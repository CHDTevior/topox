"""RawJointFlow — the ablation baseline for the pooled + quantized latent route.

Question this answers: does the Stage-1 tokenizer (graph pooling + shared RVQ codebook)
buy anything, or could we have run the same text-conditioned rectified flow directly on
the raw padded per-joint motion tensor?

To make that the ONLY variable, everything else is held identical to the main model:
  * same generative objective  — rectified flow, masked MSE on valid entries
  * same text conditioning     — dual_text (pooled -> FiLM/AdaLN, tokens -> cross-attn),
                                 gated by has_text so CFG works the same way
  * same graph conditioning    — adjacency + geodesic + per-joint skeleton embedding
  * same data + curriculum + optimizer + schedule (the trainer is shared)

The only difference is the space being modelled:
    main model : z_q   [B, T_lat=75, C=96, 512]   pooled, quantized   ( 7 200 tokens)
    this model : x     [B, T=300,    J=144,  13]   raw, per-joint      (43 200 tokens)

Implementation is deliberately thin: `GraphSaladDenoiser` is reused UNMODIFIED. Its
forward contract is [B, T, N, D] with an [B, N, N] graph over the N axis and is agnostic
to whether N indexes coarse slots or joints, so we pass joints. All this file adds is the
13 -> D input projection, the D -> 13 zero-init velocity head, and the flow math (copied
in form from CodeFlow_Model/flow.py so the two losses are numerically comparable).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.graph_salad.denoiser import GraphSaladDenoiser


class RawJointFlow(nn.Module):
    """Rectified flow over raw per-joint motion, conditioned on text + skeleton graph."""

    def __init__(
        self,
        motion_ch: int = 13,
        joint_feat_dim: int = 9,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        n_layers: int = 12,
        d_text: int = 768,
        text_token_dim: int = 768,
        dropout: float | None = None,
        noise_scale: float = 1.0,
        t_eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.motion_ch = motion_ch
        self.d_model = d_model
        self.noise_scale = noise_scale
        self.t_eps = t_eps

        self.in_proj = nn.Linear(motion_ch, d_model)
        # per-joint skeleton embedding, the joint-space analogue of the main model's
        # pooled_skeleton_embeddings; the denoiser adds it to every slot/joint token.
        self.skel_proj = nn.Linear(joint_feat_dim, d_model)

        self.net = GraphSaladDenoiser(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff, n_layers=n_layers,
            d_text=d_text, text_token_dim=text_token_dim,
            dropout=0.1 if dropout is None else dropout,
            text_mode="dual_text", spatial_mode="graph",
        )

        # zero-init velocity head: the model starts as an identity map, matching the
        # main model's zero-init readout.
        self.out_proj = nn.Linear(d_model, motion_ch)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    # ------------------------------------------------------------------ #
    def predict_velocity(self, x_t: torch.Tensor, timesteps: torch.Tensor,
                         cond: dict, *, validate_inputs: bool = False) -> torch.Tensor:
        """x_t [B,T,J,motion_ch] -> v_pred [B,T,J,motion_ch]."""
        h = self.in_proj(x_t)                                    # [B,T,J,D]
        s_j = self.skel_proj(cond["skeleton_features"])          # [B,J,D]
        # GraphSaladDenoiser requires every float input to share z_t's dtype. Under
        # bf16 autocast the projections come out bf16 while the graph and text tensors
        # arrive fp32, so align them here rather than relaxing the denoiser's contract.
        dt = h.dtype
        h = self.net(
            h, timesteps,
            cond["text_global"].to(dt),
            cond["adjacency"].to(dt), cond["geodesic_dist"].to(dt),
            cond["joint_mask"], cond["frame_mask"],
            None,
            pooled_skeleton_embeddings=s_j.to(dt),
            has_text=cond["has_text"],
            validate_inputs=validate_inputs,
            text_token_mask=cond["text_token_mask"],
            text_tokens=cond["text_tokens"].to(dt),
        )                                                        # [B,T,J,D]
        return self.out_proj(h)                                  # [B,T,J,motion_ch]

    # ------------------------------------------------------------------ #
    def forward(self, *args, **kwargs) -> dict:
        """DDP entry point: forward == flow_loss so the gradient all-reduce fires."""
        return self.flow_loss(*args, **kwargs)

    def flow_loss(
        self,
        x: torch.Tensor,                   # [B,T,J,motion_ch] NORMALIZED motion (target)
        token_mask: torch.Tensor,          # [B,T,J] bool — valid (frame AND joint)
        cond: dict,
        *,
        noise: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        validate_inputs: bool = False,
    ) -> dict:
        """Same rectified-flow math as CodeFlow_Model/flow.py, on raw motion.

          z_t = t*x + (1-t)*noise ;  v_target = x - noise
          loss = masked mean over (valid entries * motion_ch) of (v_pred - v_target)^2
        """
        if x.dim() != 4 or x.shape[-1] != self.motion_ch:
            raise ValueError(
                f"flow_loss: x must be [B,T,J,{self.motion_ch}], got {tuple(x.shape)}")
        B, T, J, _ = x.shape
        if token_mask.shape != (B, T, J) or token_mask.dtype != torch.bool:
            raise ValueError(
                f"flow_loss: token_mask must be [B,T,J]={(B, T, J)} bool, got "
                f"{tuple(token_mask.shape)} {token_mask.dtype}")

        valid = token_mask.unsqueeze(-1).to(x.dtype)             # [B,T,J,1]
        x = x * valid                                            # zero padded targets
        if noise is None:
            noise = torch.randn_like(x) * self.noise_scale
        else:
            noise = noise.to(device=x.device, dtype=x.dtype)
        noise = noise * valid

        if timesteps is None:
            t = torch.rand(B, device=x.device, dtype=x.dtype)
        else:
            t = timesteps.to(device=x.device, dtype=x.dtype)
            if t.ndim == 0:
                t = t.expand(B)
        t4 = t.view(B, 1, 1, 1)

        z_t = (t4 * x + (1.0 - t4) * noise) * valid
        v_target = (x - noise) * valid
        v_pred = self.predict_velocity(z_t, t, cond, validate_inputs=validate_inputs)
        v_pred = v_pred * valid

        diff = (v_pred.float() - v_target.float()) ** 2
        denom = token_mask.float().sum().clamp_min(1.0) * float(self.motion_ch)
        loss = diff.sum() / denom
        return {"flow_loss": loss, "velocity_pred": v_pred,
                "velocity_target": v_target, "z_t": z_t, "timesteps": t}

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample(self, cond: dict, token_mask: torch.Tensor, T: int, J: int,
               *, steps: int = 25, cfg_scale: float = 4.0,
               generator: torch.Generator | None = None) -> torch.Tensor:
        """Euler integration of the learned velocity field with CFG, mirroring
        flow.py::sample. Returns NORMALIZED motion [B,T,J,motion_ch]."""
        device = token_mask.device
        B = token_mask.shape[0]
        valid = token_mask.unsqueeze(-1).float()
        z = torch.randn(B, T, J, self.motion_ch, device=device,
                        generator=generator) * self.noise_scale * valid

        cond_u = dict(cond)
        cond_u["has_text"] = torch.zeros_like(cond["has_text"])

        dt = 1.0 / float(steps)
        for i in range(steps):
            t = torch.full((B,), i * dt, device=device, dtype=z.dtype)
            v = self.predict_velocity(z, t, cond)
            if cfg_scale != 1.0:
                v_u = self.predict_velocity(z, t, cond_u)
                v = v_u + cfg_scale * (v - v_u)
            z = (z + dt * v) * valid
        return z
