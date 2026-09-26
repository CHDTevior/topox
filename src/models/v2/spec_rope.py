"""Spectral joint RoPE, ported from UniMate (outside_docs/UniMate/unimate/models/denoiser/rope.py: _rotate_half,
_apply_rotary_pos_emb, SignNetSpectralEncoder, SpectralJointRoPE with use_signnet=True, signnet_hidden 64; the
angles are produced once per sample from the rig's Laplacian eigenvectors (src/data/skeleton_spectral.py) and the
rotation is broadcast over the frames, as UniMate's forward_per_frame does).

Where it acts (as in UniMate's graph attention, blocks/graph.py:138-146): on the spatial attention's q and k, AFTER the
q/k RMS normalisation and before the dot product. It REPLACES the learned joint-slot table (UniMate keeps no additive
joint index embedding when use_spectral_rope is on; here InContextMotionDiT(use_spec_rope=True) drops j_pos), so the
model addresses joints by where they sit in the tree's spectrum instead of by a slot index shared across rigs.

H1 (2026-09-24, `hks=True`): the coordinate is the rig's heat-kernel signature (src/data/skeleton_spectral.py
heat_kernel_signature, K scales, invariant to eigenvector sign AND basis) and the encoder a plain MLP
(HeatKernelSpectralEncoder): nothing to symmetrise, so no SignNet. Same angles out, same rotation.
"""
import torch
import torch.nn as nn

from src.data.skeleton_spectral import hks_scales


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap and negate the two halves of the last dimension."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """out = x * cos + rotate_half(x) * sin on the last dim of q and k (float32 math, cast back to the input dtype);
    cos / sin must already broadcast against q and k."""
    dtype = q.dtype
    q_f, k_f = q.float(), k.float()
    q_out = (q_f * cos) + (_rotate_half(q_f) * sin)
    k_out = (k_f * cos) + (_rotate_half(k_f) * sin)
    return q_out.to(dtype), k_out.to(dtype)


class SignNetSpectralEncoder(nn.Module):
    """rho([phi(v_k) + phi(-v_k)]_{k=1..K}): phi a per-frequency scalar MLP, rho an aggregation MLP; invariant to the
    arbitrary signs of Laplacian eigenvectors because phi(v) + phi(-v) is unchanged when v is negated."""

    def __init__(self, num_eigvecs: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.num_eigvecs = num_eigvecs
        self.phi = nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.rho = nn.Sequential(nn.Linear(num_eigvecs * hidden_dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, out_dim))
        nn.init.normal_(self.rho[-1].weight, std=0.02)
        nn.init.zeros_(self.rho[-1].bias)

    def forward(self, spectral_coords: torch.Tensor) -> torch.Tensor:
        """[B, J, K] eigenvector values -> [B, J, out_dim] rotation angles."""
        v = spectral_coords.unsqueeze(-1)              # (B, J, K, 1)
        h = self.phi(v) + self.phi(-v)                  # (B, J, K, hidden)
        h = h.flatten(start_dim=-2)                     # (B, J, K * hidden)
        return self.rho(h)                              # (B, J, out_dim)


class HeatKernelSpectralEncoder(nn.Module):
    """A plain MLP from the heat-kernel signature to the rotation angles. The signature is already invariant to the
    eigenvectors' signs and to the basis inside a repeated eigenvalue, so nothing is symmetrised (no phi(v) + phi(-v)).
    The scale ladder the coordinates were computed with travels with the weights as a persistent buffer: together with
    the MLP's own key names it is the checkpoint's marker, so a SignNet checkpoint and an HKS checkpoint refuse each
    other's weights under strict loading."""

    def __init__(self, num_scales: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.num_scales = num_scales
        self.net = nn.Sequential(nn.Linear(num_scales, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, out_dim))
        self.register_buffer("scales", torch.tensor(hks_scales(num_scales), dtype=torch.float32))
        # the ladder is a pure function of num_scales, but a checkpoint measured under another ladder must be REFUSED,
        # not silently corrected by load_state_dict (same contract as trope_base in dit_motion.py)
        self._register_load_state_dict_pre_hook(self._refuse_scales_drift)

    def _refuse_scales_drift(self, state_dict, prefix, *args):
        key = prefix + "scales"
        # compared on the CPU: every consumer builds the model on CUDA first and loads a map_location="cpu" state_dict,
        # and torch.equal refuses to compare tensors on different devices
        if key in state_dict and not torch.equal(state_dict[key].detach().cpu().to(self.scales.dtype), self.scales.detach().cpu()):
            raise RuntimeError(f"{key}: checkpoint HKS scale ladder {state_dict[key].tolist()} != this model's "
                               f"{self.scales.tolist()} (num_scales {self.num_scales})")

    def forward(self, hks: torch.Tensor) -> torch.Tensor:
        """[B, J, S] heat-kernel signature -> [B, J, out_dim] rotation angles."""
        return self.net(hks)


def unimate_basic_init(module: nn.Module) -> None:
    """UniMate's denoiser-wide `_basic_init` (models/denoiser/base.py initialize_weights, applied by _finish_build AFTER the
    spectral module is built): every nn.Linear gets Xavier-uniform weights and zero biases. It overrides rope.py's own
    rho[-1] normal(0.02) init, so this -- not the class's own init -- is the initialisation UniMate actually trains from
    (codex specrope r1 P2)."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class SpectralJointRoPE(nn.Module):
    """Laplacian eigenvectors -> cos / sin of one rotation angle per pair of head channels (head_dim // 2 angles,
    duplicated over the two halves as standard RoPE does). Initialised as UniMate's denoiser leaves it: the class's own
    draws happen first (same RNG consumption order as their build), then `unimate_basic_init` over every Linear.
    With `hks` the encoder is the HeatKernelSpectralEncoder over `num_eigvecs` SCALES (same hidden width, same
    xavier-uniform / zero-bias init pass) and the coordinates it expects are the heat-kernel signature."""

    def __init__(self, head_dim: int, num_eigvecs: int = 8, signnet_hidden: int = 64, hks: bool = False):
        super().__init__()
        assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
        self.head_dim = head_dim
        self.num_eigvecs = num_eigvecs
        self.hks = bool(hks)
        if self.hks:
            self.spectral_encoder = HeatKernelSpectralEncoder(num_scales=num_eigvecs, hidden_dim=signnet_hidden,
                                                              out_dim=head_dim // 2)
        else:
            self.spectral_encoder = SignNetSpectralEncoder(num_eigvecs=num_eigvecs, hidden_dim=signnet_hidden,
                                                           out_dim=head_dim // 2)
        self.apply(unimate_basic_init)

    def cos_sin(self, spectral_coords: torch.Tensor):
        """[B, J, K] (eigenvectors, or the heat-kernel signature at K scales with hks) -> (cos, sin) each [B, J, head_dim]."""
        angles = self.spectral_encoder(spectral_coords)   # (B, J, C/2)
        angles = torch.cat([angles, angles], dim=-1)      # (B, J, C)
        return angles.cos(), angles.sin()
