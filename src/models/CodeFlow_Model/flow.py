"""GraphCodeFlow — rectified-flow objective + ODE/CFG sampler over the FROZEN
Graph-VQVAE post-RVQ z_q grid.

PORTED (not imported) from CodeFlow `outside_docs/CodeFlow/models/codeflow/
motion_code_flow.py`:
  - rectified-flow interpolation + velocity target (:474-475 compute_losses):
        z_t = t*x + (1-t)*noise ;  v_target = x - noise
  - masked flow MSE over valid tokens (:509-510)
  - predict_clean_from_velocity (:405-413):  clean = z_t + (1-t)*v
  - ODE sampler + classifier-free guidance (:570-649 sample_embeddings):
        v = v_uncond + cfg_scale * (v_cond - v_uncond) ;  z = z + dt * v
  - empirical / codebook latent normalization hooks (:199-283 raw_to_model_latent
    / model_to_raw_latent)

ADAPTED for our setting:
  - mask is 2D `[B,T_lat,C]` (token_mask = coarse_mask & frame_mask_lat), NOT the
    1D time-length mask CodeFlow uses (`lengths_to_mask`). Applied at noise-init,
    in the loss reduction, in the CFG combine, in the ODE update, and (by the
    projection) before snapping.
  - latent normalization is EMPIRICAL z_q (mean/std over VALID train-set tokens),
    registered as frozen buffers, NOT codebook-stat (LOCKED recipe).
  - the velocity network is GraphStructuredCodeFlow (graph token grid), wrapped
    here; the frozen tokenizer (encode/quantize/decode + nearest_residual_ids)
    lives OUTSIDE this module and is passed in by the trainer / sampler.
  - terminal ID CE is OFF (flow-only, LOCKED recipe). Only the rectified-flow
    masked MSE is returned as the training loss.

This module owns ONLY the flow math + sampler; it does not own the tokenizer or
the dataset, keeping the post-RVQ branch decoupled from the Gaussian VAE / latent
diffusion (handoff §16).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .graph_codeflow import GraphStructuredCodeFlow
from .graph_pscf import GraphPSCFFlowNet


class GraphCodeFlow(nn.Module):
    """Rectified-flow wrapper around GraphStructuredCodeFlow.

    Holds the velocity net + the frozen empirical-norm buffers. The frozen
    Graph-VQVAE tokenizer is NOT held here (it is passed to `decode`/sampling by
    the caller) so this stays a pure flow module.

    Empirical normalization (LOCKED): set_latent_stats(mean, std) registers frozen
    buffers `latent_mean`/`latent_std` shaped [1,1,1,D]; `normalize` maps raw z_q
    -> model space, `denormalize` inverts it. Defaults are mean 0 / std 1 (identity)
    until the trainer computes empirical stats over VALID exported z_q tokens.
    """

    def __init__(
        self,
        code_dim: int = 512,
        n_heads: int = 8,
        d_ff: int | None = None,
        n_layers: int = 5,
        d_text: int = 768,
        text_token_dim: int = 768,
        dropout: float | None = None,
        noise_scale: float = 1.0,
        t_eps: float = 1e-4,
        model_variant: str = "level_a",
        depth_double: int = 6,
        depth_single: int = 12,
        max_T_lat: int = 75,
        mlp_ratio: float = 4.0,
        text_input_norm: bool = False,
        use_sentence_token: bool = False,
        text_slot_xattn: bool = False,
        parameterization: str = "v",
    ) -> None:
        super().__init__()
        if parameterization not in ("v", "x"):
            raise ValueError(
                f"unknown parameterization {parameterization!r} (expected v | x)")
        self.code_dim = code_dim
        self.model_variant = model_variant
        self.parameterization = parameterization
        self.noise_scale = float(noise_scale)
        self.t_eps = float(t_eps)
        # Resolve dropout per variant if unset (spec §5.4: graph_pscf=0.05, level_a=0.1)
        # so a bare GraphCodeFlow(model_variant="graph_pscf") does NOT inherit 0.1.
        if dropout is None:
            dropout = 0.05 if model_variant == "graph_pscf" else 0.1
        # model_variant selector (handoff §5.3): old ckpts default to level_a so they
        # still load; graph_pscf is the formal backbone. Both nets share the EXACT
        # 11-positional-arg forward contract, so predict_velocity/CFG/empirical-norm
        # below are unchanged and drop-in for either variant.
        if model_variant == "level_a":
            if text_input_norm or use_sentence_token or text_slot_xattn:
                raise ValueError(
                    "text_input_norm/use_sentence_token/text_slot_xattn are "
                    "graph_pscf-only; level_a does not implement them")
            self.net = GraphStructuredCodeFlow(
                code_dim=code_dim, n_heads=n_heads, d_ff=d_ff, n_layers=n_layers,
                d_text=d_text, text_token_dim=text_token_dim, dropout=dropout)
        elif model_variant == "graph_pscf":
            self.net = GraphPSCFFlowNet(
                code_dim=code_dim, n_heads=n_heads,
                d_ff=(2048 if d_ff is None else d_ff),
                depth_double=depth_double, depth_single=depth_single,
                d_text=d_text, text_token_dim=text_token_dim,
                max_T_lat=max_T_lat, mlp_ratio=mlp_ratio, dropout=dropout,
                text_input_norm=text_input_norm,
                use_sentence_token=use_sentence_token,
                text_slot_xattn=text_slot_xattn)

        else:
            raise ValueError(
                f"unknown model_variant {model_variant!r} (expected level_a | graph_pscf)")
        # Empirical latent stats (frozen buffers; identity until set). persistent
        # so they travel with the checkpoint (handoff §5.2: normalizer saved with
        # the backbone ckpt).
        self.register_buffer("latent_mean", torch.zeros(1, 1, 1, code_dim))
        self.register_buffer("latent_std", torch.ones(1, 1, 1, code_dim))

    # ------------------------------------------------------------------ #
    # Empirical normalization (frozen)                                   #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def set_latent_stats(self, mean: torch.Tensor, std: torch.Tensor,
                         eps: float = 1e-6) -> None:
        """Install empirical z_q stats. mean/std are [D] (or broadcastable to
        [1,1,1,D]); std is floored at eps to avoid divide-by-zero."""
        mean = mean.reshape(1, 1, 1, self.code_dim).float()
        std = std.reshape(1, 1, 1, self.code_dim).float().clamp_min(eps)
        self.latent_mean.copy_(mean)
        self.latent_std.copy_(std)

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        m = self.latent_mean.to(device=z.device, dtype=z.dtype)
        s = self.latent_std.to(device=z.device, dtype=z.dtype)
        return (z - m) / s

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        m = self.latent_mean.to(device=z.device, dtype=z.dtype)
        s = self.latent_std.to(device=z.device, dtype=z.dtype)
        return z * s + m

    # ------------------------------------------------------------------ #
    # Velocity prediction (thin pass-through to the graph net)           #
    # ------------------------------------------------------------------ #
    # x-parameterization: loss-weight cap. The v-space residual obeys the exact
    # identity v_hat - v_target == (x_hat - x)/(1-t) (z_t = t*x + (1-t)*noise),
    # so flow_loss computes the residual THAT way: the optimum stays x_hat == x
    # at EVERY t (no bias term), and this floor merely caps the per-sample
    # x-error weight at 1e4 for t > 0.99 (~1% of uniform t), where an uncapped
    # 1/(1-t)^2 would amplify bf16 output rounding into loss spikes. (Capping
    # (1-t) INSIDE the x->v conversion instead would add a (q/c - 1)*(x-noise)
    # bias to the residual and move the optimum — codex r1 #1.)
    X_PRED_MIN_ONE_MINUS_T = 1e-2

    def _net_forward(
        self,
        z_t: torch.Tensor,
        timesteps: torch.Tensor,
        cond: dict,
        *,
        validate_inputs: bool = False,
    ) -> torch.Tensor:
        """Raw net output: a velocity under parameterization='v', the clean
        latent x_hat under parameterization='x'."""
        return self.net(
            z_t, timesteps,
            cond["text_global"], cond["text_tokens"], cond["text_token_mask"],
            cond["has_text"], cond["pooled_adjacency"], cond["pooled_geodesic"],
            cond["pooled_skeleton_embeddings"], cond["coarse_mask"],
            cond["frame_mask_lat"], validate_inputs=validate_inputs)

    def predict_velocity(
        self,
        z_t: torch.Tensor,
        timesteps: torch.Tensor,
        cond: dict,
        *,
        validate_inputs: bool = False,
    ) -> torch.Tensor:
        """cond carries the (already-prepared) conditioning tensors:
        text_global, text_tokens, text_token_mask, has_text, pooled_adjacency,
        pooled_geodesic, pooled_skeleton_embeddings, coarse_mask, frame_mask_lat.

        Under parameterization="x" the net's output is read as the clean latent
        x_hat and converted here EXACTLY: v_hat = (x_hat - z_t)/(1-t) in fp32
        ((1-t) floored at t_eps=1e-4 purely against divide-by-zero — the Euler
        sampling grid tops out at t = 1 - 1/steps, so for any steps <= 10^4 the
        floor never activates and the sampler / predict_clean_from_velocity see
        the exact conversion). Training does NOT go through this path (flow_loss
        uses the identity residual, see X_PRED_MIN_ONE_MINUS_T above).
        """
        out = self._net_forward(z_t, timesteps, cond, validate_inputs=validate_inputs)
        if self.parameterization == "x":
            t = timesteps
            while t.ndim < z_t.ndim:
                t = t[..., None]
            one_minus_t = (1.0 - t.float()).clamp_min(self.t_eps)
            out = ((out.float() - z_t.float()) / one_minus_t).to(out.dtype)
        return out

    def predict_clean_from_velocity(
        self, z_t: torch.Tensor, timesteps: torch.Tensor, velocity: torch.Tensor,
    ) -> torch.Tensor:
        """clean = z_t + (1 - t) * v  (CodeFlow motion_code_flow.py:405-413)."""
        t = timesteps
        while t.ndim < z_t.ndim:
            t = t[..., None]
        return z_t + (1.0 - t).clamp_min(self.t_eps) * velocity

    # ------------------------------------------------------------------ #
    # Rectified-flow training loss (flow-only, masked over valid tokens) #
    # ------------------------------------------------------------------ #
    def forward(self, *args, **kwargs) -> dict:
        """DDP entry point: forward == flow_loss, so DDP's gradient all_reduce
        fires on the training step. Single-process callers may use flow_loss
        directly; under DDP the trainer MUST call the wrapped module via
        flow(...) (NOT flow.module.flow_loss, which would bypass the DDP hooks
        and desync gradients across ranks)."""
        return self.flow_loss(*args, **kwargs)

    def flow_loss(
        self,
        z_q: torch.Tensor,                 # [B,T_lat,C,D] RAW frozen RVQ z_q
        token_mask: torch.Tensor,          # [B,T_lat,C] bool (valid tokens)
        cond: dict,                        # conditioning (with CFG drop applied)
        *,
        noise: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        validate_inputs: bool = False,
    ) -> dict:
        """Rectified-flow masked MSE. Returns {flow_loss, velocity_pred,
        velocity_target, z_t, timesteps} (the extras let the trainer log the
        continuous-vs-snapped projection on the SAME predicted clean latent).

        Flow math (CodeFlow): work in NORMALIZED latent space.
          x = normalize(z_q)
          z_t = t*x + (1-t)*noise ;  v_target = x - noise
          v_pred = net(z_t, t, cond)
          loss = mean over (valid tokens * D) of (v_pred - v_target)^2
        2D `[T_lat,C]` masking is applied to z_t (noise-init), and the loss
        reduction divides by (#valid tokens * D). fp32 reductions.
        """
        if z_q.dim() != 4 or z_q.shape[-1] != self.code_dim:
            raise ValueError(
                f"flow_loss: z_q must be [B,T_lat,C,D={self.code_dim}], got "
                f"{tuple(z_q.shape)}")
        B, T_lat, C, D = z_q.shape
        if token_mask.shape != (B, T_lat, C) or token_mask.dtype != torch.bool:
            raise ValueError(
                f"flow_loss: token_mask must be [B,T_lat,C]={(B, T_lat, C)} bool, "
                f"got {tuple(token_mask.shape)} {token_mask.dtype}")
        x = self.normalize(z_q)
        valid = token_mask.unsqueeze(-1).to(x.dtype)       # [B,T_lat,C,1]
        # zero padded targets so noise never injects signal there.
        x = x * valid
        if noise is None:
            noise = torch.randn_like(x) * self.noise_scale
        else:
            noise = noise.to(device=x.device, dtype=x.dtype)
        noise = noise * valid
        if timesteps is None:
            t = torch.rand(B, device=x.device, dtype=x.dtype)   # uniform t (recipe)
        else:
            t = timesteps.to(device=x.device, dtype=x.dtype)
            if t.ndim == 0:
                t = t.expand(B)
        t_view = t[:, None, None, None]
        z_t = (t_view * x + (1.0 - t_view) * noise) * valid
        v_target = (x - noise) * valid

        raw = self._net_forward(z_t, t, cond, validate_inputs=validate_inputs)
        if self.parameterization == "x":
            # v-space residual via the exact identity v_hat - v_target ==
            # (x_hat - x)/(1-t): optimum stays x_hat == x at every t; the floor
            # only caps the weight (see X_PRED_MIN_ONE_MINUS_T). fp32.
            c = (1.0 - t_view.float()).clamp_min(self.X_PRED_MIN_ONE_MINUS_T)
            diff = (raw.float() - x.float()) / c
            # Exact conversion for the QA extras (predict_clean_from_velocity
            # reconstructs x_hat from this); t_eps guard against t == 1 only.
            v_pred = ((raw.float() - z_t.float())
                      / (1.0 - t_view.float()).clamp_min(self.t_eps)).to(raw.dtype)
            # Clean prediction in NORMALIZED latent space, gradient-carrying —
            # consumed by the trainer's decoded-geometry loss (ARDY-style L_dec).
            clean_pred_norm = raw
        else:
            v_pred = raw
            diff = v_pred.float() - v_target.float()
            clean_pred_norm = z_t + (1.0 - t_view) * v_pred

        # Masked flow MSE in fp32 (CodeFlow :509-510 adapted to 2D mask + sum/D).
        vmask = token_mask.unsqueeze(-1).float()           # [B,T_lat,C,1]
        diff_sq = diff.pow(2) * vmask
        denom = vmask.sum().clamp_min(1.0) * 1.0            # (#valid tokens) * D via broadcast below
        # diff_sq summed over the D axis is folded into the numerator already
        # (vmask broadcasts over D); divide by valid_token_count * D.
        n_valid_tokens = token_mask.float().sum().clamp_min(1.0)
        loss = diff_sq.sum() / (n_valid_tokens * D)
        return {
            "flow_loss": loss,
            "velocity_pred": v_pred,
            "velocity_target": v_target,
            "clean_pred_norm": clean_pred_norm,
            "z_t": z_t,
            "timesteps": t,
        }

    # ------------------------------------------------------------------ #
    # ODE + CFG sampler (continuous z_hat in RAW latent space)           #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample(
        self,
        cond: dict,
        token_mask: torch.Tensor,          # [B,T_lat,C] bool
        T_lat: int,
        C: int,
        *,
        steps: int = 50,
        cfg_scale: float = 4.0,
        validate_inputs: bool = False,
    ) -> torch.Tensor:
        """ODE-integrate from t=0 to t=1 with classifier-free guidance, returning
        the DEnormalized continuous z_hat [B,T_lat,C,D] (raw RVQ latent space, to
        be fed to tokenizer.nearest_residual_ids). Padded tokens are kept 0.

        CFG (handoff §8): the uncond branch drops BOTH text streams (has_text
        False + token mask all-masked). `cond` must contain a sibling
        `cond_uncond` dict already built with text dropped, OR has_text False; we
        build the uncond view here by flipping has_text -> all-False (the token
        cross-attn zeroes those rows; the global add is gated to 0). cfg_scale=1.0
        skips the uncond pass.
        """
        device = token_mask.device
        B = token_mask.shape[0]
        D = self.code_dim
        valid = token_mask.unsqueeze(-1).float()           # [B,T_lat,C,1]
        z = torch.randn(B, T_lat, C, D, device=device) * self.noise_scale
        z = z * valid
        grid = torch.linspace(0.0, 1.0, steps + 1, device=device)

        # Build the uncond conditioning view (text dropped) once.
        cond_uncond = dict(cond)
        cond_uncond["has_text"] = torch.zeros_like(cond["has_text"])

        for i in range(steps):
            t_cur = grid[i]
            dt = grid[i + 1] - t_cur
            t_b = t_cur.expand(B)
            v_cond = self.predict_velocity(z, t_b, cond, validate_inputs=validate_inputs)
            if float(cfg_scale) == 1.0:
                v = v_cond
            else:
                v_uncond = self.predict_velocity(
                    z, t_b, cond_uncond, validate_inputs=validate_inputs)
                v = v_uncond + float(cfg_scale) * (v_cond - v_uncond)
            z = (z + dt * v) * valid
        # De-normalize back to RAW RVQ latent space (so nearest_residual_ids snaps
        # against the real codebooks), then re-mask padded tokens to 0.
        z_hat = self.denormalize(z) * valid
        return z_hat
