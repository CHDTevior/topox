"""A flat, padded-joint baseline denoiser, built the way the comparable works build theirs.

Every multi-topology motion paper we are positioned against adapts a fixed-skeleton generator to
variable skeletons the same way, and reports it as a baseline:

  AnyTop (Gat et al., 2025) on MDM: "to match its representation format, we concatenate all joint
    features for each character, and pad them to a length of J x D", and "we concatenate the
    vectorized rest-pose embedding along the temporal axis as frame 0".
  SAMoR on T2M-GPT: "concatenating all J per-joint position-and-velocity features into a single
    flat per-frame feature vector of dimension Jmax x 6", "zero-padded to Jmax=96 joints, with
    padded positions masked", and "trained on the same unified heterogeneous corpus ... with the
    same data splits, loss functions, and training budget, to ensure a fair comparison".
  OmniZoo on MoMask/MMM: "we extend them to heterogeneous skeletons using joint padding and binary
    masking, and append species tags to text prompts for explicit conditioning".

This module is that construction on our corpus. One token per FRAME holding every joint's channels,
zero-padded to `max_joints` and masked by the batch's joint-validity flags; frames attend to frames;
the caption modulates every block through AdaLN, as in our own model. The species is already named
in the caption, so no separate species tag is appended.

WHAT IT IS GIVEN, and what it is not. Given: the same per-cell normalised corpus, the same split,
the same captions, the rig's rest pose as frame 0, and the padding mask -- and, exactly as the
per-joint model gets them, the corpus's channel-validity masks -- the fixed-DOF rotation rows and
the statistically constant cells -- which zero structurally invalid cells in the interpolant
(cfm_loss's `eff`, plus the per-frame heading validity) and are skeleton-derived. Not given: the per-joint
natural-language descriptions, the geodesic and directional attention biases, the structural joint
features, and per-joint tokens -- the four ingredients this paper is about. The claim is that those
FOUR PATHWAYS are absent, not that the baseline sees no skeleton-derived quantity at all
(codex 2026-09-10 #6). The conditioning fields
the trainer always sends for those (`joint_sem`, `joint_bias`, `struct_feats`, `updown`,
`demo_text`, `local_root`, `blueprint`) are listed in IGNORED and dropped by an explicit contract;
any OTHER keyword raises, so a new conditioning input cannot reach this model unnoticed.

Kept in its own file on purpose: the calibration gate hashes the bytes of dit_motion.py, so adding
this baseline there would invalidate every calibration artifact and every live run bound to one.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from src.models.v2.dit_motion import Attention, modulate, timestep_embedding

# conditioning the trainer always sends and this baseline deliberately does not read
IGNORED = ("joint_sem", "joint_bias", "struct_feats", "updown", "demo_text", "local_root",
           "blueprint")


class FlatBlock(nn.Module):
    """Frames attend to frames, then an MLP; both AdaLN-modulated by the conditioning vector.

    The joint axis is gone, so the spatial attention of the per-joint trunk has nothing to attend
    over and is not built: it would be self-attention over one token, spending parameters on an
    operation with no content."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0, qk_norm: bool = False):
        super().__init__()
        self.n1, self.n2 = (nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6) for _ in range(2))
        self.t_attn = Attention(dim, n_heads, qk_norm=qk_norm)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(approximate="tanh"), nn.Linear(h, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        # same initialisation contract as the per-joint trunk: modulation weights small and
        # non-zero so conditioning has a live path from step 1, gates biased to 1 so a block starts
        # as an ordinary residual layer
        nn.init.normal_(self.ada[-1].weight, std=0.5); nn.init.zeros_(self.ada[-1].bias)
        with torch.no_grad():
            for k in (2, 5):
                self.ada[-1].bias[k * dim:(k + 1) * dim].fill_(1.0)

    def forward(self, x, c, frame_valid=None):
        # x [B,T,D]
        st, sc, sg, mt, mc, mg = self.ada(c).chunk(6, dim=-1)
        e = lambda v: v[:, None, :]
        x = x + e(sg) * self.t_attn(modulate(self.n1(x), e(st), e(sc)), key_pad=frame_valid)
        x = x + e(mg) * self.mlp(modulate(self.n2(x), e(mt), e(mc)))
        return x


class FlatMotionDiT(nn.Module):
    """[demo | target] frames -> clean motion, from a flat padded pose vector and the caption."""

    def __init__(self, in_ch=17, max_joints=102, dim=384, depth=8, n_heads=6, d_text=4096,
                 mlp_ratio=4.0, grad_ckpt=False, qk_norm=False):
        super().__init__()
        self.in_ch, self.max_joints, self.dim = int(in_ch), int(max_joints), int(dim)
        self.grad_ckpt = bool(grad_ckpt)
        flat = self.max_joints * self.in_ch
        self.x_in = nn.Linear(flat, dim)
        self.mask_token = nn.Parameter(torch.zeros(dim))
        self.max_T = 4096
        self.t_pos = nn.Parameter(torch.zeros(1, self.max_T, dim))
        nn.init.normal_(self.t_pos, std=0.02)
        self.t_mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))
        self.text_mlp = nn.Sequential(nn.LayerNorm(d_text), nn.Linear(d_text, dim), nn.SiLU(),
                                      nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([FlatBlock(dim, n_heads, mlp_ratio, qk_norm=qk_norm)
                                     for _ in range(depth)])
        self.n_out = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.out = nn.Linear(dim, flat)
        nn.init.normal_(self.ada_out[-1].weight, std=0.5); nn.init.zeros_(self.ada_out[-1].bias)
        nn.init.normal_(self.out.weight, std=0.02); nn.init.zeros_(self.out.bias)
        # travels with the checkpoint: a consumer that rebuilds this as the per-joint model fails
        # strict loading instead of silently restoring a different architecture
        self.register_buffer("flat_baseline", torch.tensor(float(self.max_joints)))

    def forward(self, x, t, *, is_target, text=None, frame_valid=None, joint_valid=None, **cond):
        """x [B,T,J,C]; t [B] in [0,1]; is_target [B,T] bool. Returns [B,T,J,C]."""
        unexpected = [k for k in cond if k not in IGNORED]
        if unexpected:
            raise ValueError(f"FlatMotionDiT received conditioning it does not read and does not "
                             f"declare: {sorted(unexpected)}")
        B, T, J, C = x.shape
        if C != self.in_ch:
            raise ValueError(f"channels {C} != {self.in_ch}")
        if J > self.max_joints:
            raise ValueError(f"{J} joints exceeds max_joints={self.max_joints}")
        if joint_valid is not None:                      # the binary masking of the precedents
            x = x * joint_valid[:, None, :, None].to(x.dtype)
        # F.pad, not an allocate-and-slice-assign: with a dynamic J the latter makes inductor emit a
        # graph whose output list holds a plain int, and compilation dies with
        # "AttributeError: 'int' object has no attribute 'meta'" (observed 2026-09-10 in the 8-rank smoke).
        flat = F.pad(x, (0, 0, 0, self.max_joints - J))                  # zero-pad the joint axis
        h = self.x_in(flat.flatten(2))                                   # [B,T,J_max*C], no named B/T
        h = h + self.t_pos[:, :T]
        h = h + is_target[..., None].to(h.dtype) * self.mask_token
        c = self.t_mlp(timestep_embedding(t * 1000.0, self.dim))
        if text is not None:
            c = c + self.text_mlp(text)
        for blk in self.blocks:
            if self.grad_ckpt and self.training and torch.is_grad_enabled():
                h = torch.utils.checkpoint.checkpoint(blk, h, c, frame_valid, use_reentrant=False)
            else:
                h = blk(h, c, frame_valid=frame_valid)
        shift, scale = self.ada_out(c).chunk(2, dim=-1)
        h = modulate(self.n_out(h), shift[:, None], scale[:, None])
        return self.out(h).unflatten(-1, (self.max_joints, C))[:, :, :J]
