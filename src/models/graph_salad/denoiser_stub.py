"""GraphSaladDenoiserStub — Phase 2 denoiser interface lock for M1.

M1 milestone (PLAN_GAP_REPORT.md Q-C lock): denoiser stub is signature-only.
The forward signature is the load-bearing artifact — it pins what Phase 2
must accept, including the ``level2_meta`` dict that carries token →
coarse-group mappings needed for SALAD-style attention-map editing.

Phase 2 (M2 milestone) will replace ``GraphSaladDenoiserStub`` with the real
``GraphSaladDenoiser`` implementing temporal attention + graph-aware
skeletal attention (Graphormer SPD + GRPE-style edge K/V per lit survey) +
text cross-attention. The signature must remain backward-compatible.

Per codex pre-scaffold review (2026-05-20), ``level2_meta`` is added now
(rather than deferred) so that any Phase 1 wiring (e.g. eval-time noise
shape forming, sampling utility) can already pass the right argument
without rework.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GraphSaladDenoiserStub(nn.Module):
    """Phase 2 denoiser interface lock.

    ``forward()`` raises ``NotImplementedError``. The class is still nn.Module
    so ``GraphMotionVAE`` (M1.3) can compose it as ``self.denoiser`` and
    inspect/lock its signature without crashing on ``model.named_modules()``.

    Args:
        d_model: latent feature dim (must match VAE Gaussian latent dim).
        n_heads: number of attention heads (locked here; Phase 2 may revisit).
        n_temporal_layers: temporal attention block depth (locked here).
        n_skeletal_layers: graph-aware skeletal attention block depth.
        n_text_cross_layers: text cross-attention block depth.

    Note:
        No parameters are allocated. Phase 2 will replace the body with real
        modules. ``state_dict()`` of a stub instance is empty; cross-version
        ckpt compatibility is therefore preserved (M1 ckpts won't contain
        denoiser params that Phase 2 would conflict with).
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_temporal_layers: int = 4,
        n_skeletal_layers: int = 4,
        n_text_cross_layers: int = 2,
    ) -> None:
        super().__init__()
        # Record signature in attrs so VAE wiring (M1.3) can introspect
        # without instantiating a real denoiser.
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_temporal_layers = n_temporal_layers
        self.n_skeletal_layers = n_skeletal_layers
        self.n_text_cross_layers = n_text_cross_layers
        # No nn submodules — keeps state_dict empty (M1 ckpt-compat envelope
        # per PLAN_GAP_REPORT.md §3.6).

    def forward(
        self,
        z_t: torch.Tensor,
        timesteps: torch.Tensor,
        text: list[str] | torch.Tensor,
        adjacency: torch.Tensor,
        geodesic_dist: torch.Tensor,
        coarse_mask: torch.Tensor,
        frame_mask: torch.Tensor,
        level2_meta: dict | None = None,
    ) -> torch.Tensor:
        """Locked Phase 2 forward signature — raises ``NotImplementedError``.

        Args:
            z_t: noisy latent, shape ``[B, T_lat, C_max, d_model]`` where
                ``T_lat = T_max / 4`` (after VAE temporal downsample ×2).
            timesteps: ``[B]`` int64 diffusion step indices.
            text: ``list[str]`` (Phase 2 will encode via CLIP) OR pre-encoded
                ``[B, n_text_tokens, d_text]`` tensor.
            adjacency: ``[B, C_max, C_max]`` coarse-level adjacency.
            geodesic_dist: ``[B, C_max, C_max]`` coarse-level geodesic distance
                in graph hops (output of ``graph_utils.floyd_shortest_path``).
            coarse_mask: ``[B, C_max]`` bool, valid coarse nodes per sample.
            frame_mask: ``[B, T_lat]`` bool, valid latent frames per sample.
            level2_meta: optional dict carrying Phase-2-only metadata, e.g.
                ``{'token_to_coarse_group': [B] list[Tensor[T_lat * C_max]],
                  'coarse_to_chain_id': [B] list[list[int]], ...}``.
                M1 callers pass ``None``; Phase 2 will define schema strictly.

        Returns (Phase 2):
            Predicted noise (or velocity, depending on prediction parameterization),
            shape ``[B, T_lat, C_max, d_model]``.

        Raises:
            NotImplementedError: always; this is a signature-lock stub.
        """
        raise NotImplementedError(
            "GraphSaladDenoiserStub is a Phase-2 interface lock. "
            "M1 (GraphMotionVAE reconstruction across 3 pool variants) does "
            "not call denoiser.forward(). Phase 2 (M2) implementation will "
            "replace this class with the real GraphSaladDenoiser."
        )
