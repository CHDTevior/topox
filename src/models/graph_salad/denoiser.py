"""GraphSaladDenoiser — Phase-2 graph-aware latent diffusion denoiser.

Replaces ``denoiser_stub.GraphSaladDenoiserStub`` with the real implementation
per `docs/phase2_diffusion_design.md` §2. Architecture (v1):

  - SALAD-style skip-transformer with `n_layers=5` (2 enc + 1 mid + 2 dec).
  - Per-layer ordering: spatial_graph_attn → FiLM → temporal_self_attn → FiLM →
    text (mean_additive OR token_cross_attn) → FiLM. No trailing FFN (spatial
    block already carries FFN).
  - text_mode="mean_additive" (DEFAULT): mean-pooled T5 [B,768] additive broadcast
    (gated by has_text). Byte-identical to v1; old ckpts strict-load.
  - text_mode="token_cross_attn" (optional): motion tokens cross-attend token-level
    T5 [B,L,768] (key-padding-masked; CFG-uncond rows → zero). New params:
    text_token_proj + per-layer TextCrossAttention. Needs the offline token cache.
  - Spatial: `GraphAttentionBlock` from `graph_salad.attention` (purpose-built
    for Phase-2 coarse-node self-attn over `pooled_adjacency` / `pooled_geodesic`
    bias). Carries pre-norm + own FFN.
  - Temporal: `TemporalSelfAttention` from `motion_decoder` (key-masks padded
    frames; no FFN).
  - Text: mean-pooled T5-base [768] → `Linear(768, d_model)` → additive broadcast.
    `has_text=False` (CFG-uncond) zeroes the contribution.
  - FiLM: `DenseFiLM` per-slot, sharing the same timestep_emb across all layers.
    Formula `x * (scale + 1) + shift` (SALAD `featurewise_affine`).
  - Skip: decoder layer i concatenates with encoder layer (n_layers-1)/2 - i then
    `Linear(2D, D)` halves back.
  - Per-layer trailing re-mask `x * coarse_mask[:,None,:,None] * frame_mask[:,:,None,None]`
    so padded slots/frames never leak across layers.

Inputs (stub signature preserved + keyword-only extensions):
  z_t           [B, T_lat, C, D]
  timesteps     [B] long
  text          [B, 768]  — mean-pooled T5 (caller pre-encoded; v1 contract)
  adjacency     [B, C, C] — pooled_adjacency
  geodesic_dist [B, C, C] — pooled_geodesic
  coarse_mask   [B, C] bool
  frame_mask    [B, T_lat] bool
  level2_meta=None  (reserved for v2; currently unused)
  pooled_skeleton_embeddings=None  [B, C, D]  — additive slot conditioning
  has_text=None [B] bool — CFG gate (False → text contribution = 0)

Output:
  v_pred [B, T_lat, C, D]

Hot-path note: `validate_inputs=False` is passed to GraphAttentionBlock on every
inner step. Preflight (first iter / sampling step 0) should call once with
`validate_inputs=True` separately to catch graph-contract violations.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.graph_salad.attention import GraphAttentionBlock
from src.models.motion_decoder import TemporalSelfAttention


# ----------------------------------------------------------------------------
# Timestep embedding (sinusoidal → MLP)
# ----------------------------------------------------------------------------

class SinusoidalTimestepEmbedding(nn.Module):
    """SALAD-style sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even for sinusoidal embedding, got {dim}")
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # timesteps: [B] long or float
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / half
        )
        args = timesteps.to(torch.float32).unsqueeze(-1) * freqs.unsqueeze(0)  # [B, half]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)           # [B, dim]


# ----------------------------------------------------------------------------
# DenseFiLM (scale + shift from a global timestep embedding)
# ----------------------------------------------------------------------------

class DenseFiLM(nn.Module):
    """FiLM affine: out = x * (scale + 1) + shift, with (scale,shift) projected
    from a [B, D_t] timestep embedding. Broadcast over leading spatial dims.

    Per SALAD `featurewise_affine`: the `+1` keeps scale ≈ 0 at init so the
    block is approximate identity early in training.
    """

    def __init__(self, d_t: int, d_model: int) -> None:
        super().__init__()
        self.act = nn.SiLU()
        self.proj = nn.Linear(d_t, 2 * d_model)
        # Zero-init the proj so scale=shift=0 at init → block is identity.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x: [B, T_lat, C, D]; t_emb: [B, D_t]
        ss = self.proj(self.act(t_emb))         # [B, 2*D]
        scale, shift = ss.chunk(2, dim=-1)      # each [B, D]
        return x * (scale[:, None, None, :] + 1.0) + shift[:, None, None, :]


# ----------------------------------------------------------------------------
# Text cross-attention (token_cross_attn mode)
# ----------------------------------------------------------------------------

class TextCrossAttention(nn.Module):
    """Motion tokens query text tokens (SALAD `MultiheadAttention` template /
    PRISM `encoder_hidden_states` concept). q = motion [B, T*C, D]; k/v = text
    tokens [B, L, D]; key_padding_mask masks padded/uncond text columns.

    bf16-safety (item D): softmax is forced fp32 (scores.float() → softmax →
    .to(orig_dtype)), mirroring GraphAttentionBlock (attention.py:374). On the
    fp32 path this is a no-op (byte-for-byte unchanged); on bf16 the softmax
    reduction + the -1e9 sentinel run in fp32, then cast back.

    CFG-uncond zero (item 5): a row whose text key_padding_mask is ALL-True
    (has_text=False, or every token padded) would softmax over an all-(-1e9)
    row → uniform attention over zeroed values (not NaN here, but meaningless),
    so its cross-attn output is EXPLICITLY zeroed. We do NOT rely on softmax over
    all-(-inf); we zero the output for all-masked rows. This guarantees the
    uncond branch contributes exactly 0.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        # Zero-init output so the cross-attn block is ~identity at init (the
        # token path starts as a no-op residual; matches zero-init FiLM/output
        # convention so a fresh token run is training-stable).
        nn.init.zeros_(self.o_proj.weight)
        nn.init.zeros_(self.o_proj.bias)

    def forward(
        self,
        x: torch.Tensor,                 # [B, T*C, D] motion queries
        text_tokens: torch.Tensor,      # [B, L, D] projected text tokens
        key_padding_mask: torch.Tensor,  # [B, L] bool — True = MASK (ignore) this key
    ) -> torch.Tensor:
        B, Nq, D = x.shape
        L = text_tokens.shape[1]
        q = self.q_proj(self.norm_q(x))
        kv_in = self.norm_kv(text_tokens)
        k = self.k_proj(kv_in)
        v = self.v_proj(kv_in)
        q = q.view(B, Nq, self.n_heads, self.d_head).permute(0, 2, 1, 3)  # [B,H,Nq,dh]
        k = k.view(B, L, self.n_heads, self.d_head).permute(0, 2, 1, 3)   # [B,H,L,dh]
        v = v.view(B, L, self.n_heads, self.d_head).permute(0, 2, 1, 3)   # [B,H,L,dh]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B,H,Nq,L]
        # Mask padded/uncond text keys. key_padding_mask True ⇒ ignore.
        kpm = key_padding_mask[:, None, None, :]                  # [B,1,1,L]
        scores = scores.masked_fill(kpm, -1e9)
        # bf16-safe fp32 softmax (item D): fp32 path is a no-op.
        attn = F.softmax(scores.float(), dim=-1).to(scores.dtype)  # [B,H,Nq,L]
        # All-masked rows (every key ignored) softmax to uniform over -1e9 → we
        # zero them below; nan_to_num guards any residual NaN defensively.
        attn = attn.nan_to_num(0.0)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)                               # [B,H,Nq,dh]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, Nq, D)
        out = self.o_proj(out)
        # Item 5: explicitly zero the output for samples whose ALL text keys are
        # masked (has_text=False / fully-padded text) — the cross-attn must add
        # exactly 0 for CFG-uncond, regardless of softmax behavior.
        all_masked = key_padding_mask.all(dim=1)                  # [B] bool
        keep = (~all_masked).to(out.dtype)[:, None, None]         # [B,1,1]
        out = out * keep
        return self.dropout(out)


# ----------------------------------------------------------------------------
# One denoiser layer
# ----------------------------------------------------------------------------

class GraphSaladDenoiserLayer(nn.Module):
    """Per-layer block: spatial_graph_attn → FiLM → temporal_self_attn → FiLM →
    text → FiLM, then padded re-mask. No trailing FFN.

    Text sub-block depends on `text_mode`:
      - "mean_additive" (default): broadcast-add projected mean text_cond [B,D]
        (gated by has_text). Unchanged from v1.
      - "token_cross_attn": motion tokens cross-attend projected text tokens
        [B,L,D] (key-padding-masked; CFG-uncond rows zeroed). The additive mean
        path is NOT used in this mode.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        d_t: int,
        dropout: float = 0.1,
        text_mode: str = "mean_additive",
    ) -> None:
        super().__init__()
        self.text_mode = text_mode
        self.spatial = GraphAttentionBlock(d_model, n_heads, d_ff, dropout=dropout)
        self.temporal = TemporalSelfAttention(d_model, n_heads, dropout=dropout)
        self.film_after_spatial = DenseFiLM(d_t, d_model)
        self.film_after_temporal = DenseFiLM(d_t, d_model)
        self.film_after_text = DenseFiLM(d_t, d_model)
        # Token cross-attn sub-block exists only in token mode (so mean-mode
        # state_dict is byte-identical to old ckpts — strict-load preserved).
        if text_mode == "token_cross_attn":
            self.text_cross_attn = TextCrossAttention(d_model, n_heads, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,                  # [B, T_lat, C, D]
        t_emb: torch.Tensor,              # [B, D_t]
        text_cond: torch.Tensor | None,   # [B, D] projected mean text (mean_additive)
        has_text: torch.Tensor,           # [B] bool
        pooled_adj: torch.Tensor,         # [B, C, C]
        pooled_geo: torch.Tensor,         # [B, C, C]
        coarse_mask: torch.Tensor,        # [B, C] bool
        frame_mask: torch.Tensor,         # [B, T_lat] bool
        *,
        validate_inputs: bool = False,
        text_tokens: torch.Tensor | None = None,       # [B, L, D] (token_cross_attn)
        text_key_padding_mask: torch.Tensor | None = None,  # [B, L] bool, True=mask
    ) -> torch.Tensor:
        B, T_lat, C, D = x.shape

        # --- 1. Spatial graph self-attn (per frame, over C slots) ---
        # Reshape [B, T_lat, C, D] -> [B*T_lat, C, D]; expand graph tensors along T_lat.
        x_sp_in = x.reshape(B * T_lat, C, D)
        adj_exp = (
            pooled_adj.unsqueeze(1).expand(B, T_lat, C, C).reshape(B * T_lat, C, C)
        )
        geo_exp = (
            pooled_geo.unsqueeze(1).expand(B, T_lat, C, C).reshape(B * T_lat, C, C)
        )
        cmask_exp = (
            coarse_mask.unsqueeze(1).expand(B, T_lat, C).reshape(B * T_lat, C)
        )
        # `validate_inputs=False` for hot-path; preflight callers pass True once.
        x_sp = self.spatial(
            x_sp_in, adj_exp, geo_exp, cmask_exp, validate_inputs=validate_inputs
        )
        x = x_sp.reshape(B, T_lat, C, D)

        # --- 2. FiLM after spatial ---
        x = self.film_after_spatial(x, t_emb)

        # --- 3. Temporal self-attn (per slot, over T_lat frames) ---
        # [B, T_lat, C, D] -> [B*C, T_lat, D]
        x_t_in = x.permute(0, 2, 1, 3).contiguous().reshape(B * C, T_lat, D)
        fmask_exp = frame_mask.unsqueeze(1).expand(B, C, T_lat).reshape(B * C, T_lat)
        x_t = self.temporal(x_t_in, fmask_exp)
        x = x_t.reshape(B, C, T_lat, D).permute(0, 2, 1, 3).contiguous()

        # --- 4. FiLM after temporal ---
        x = self.film_after_temporal(x, t_emb)

        # --- 5. Text conditioning (mode-dependent) ---
        if self.text_mode == "token_cross_attn":
            # Motion tokens [B,T*C,D] cross-attend text tokens [B,L,D]. CFG-uncond
            # rows (all text keys masked) get zero output (TextCrossAttention).
            q = x.reshape(B, T_lat * C, D)
            ca = self.text_cross_attn(q, text_tokens, text_key_padding_mask)
            x = x + ca.reshape(B, T_lat, C, D)
        else:
            # mean_additive (default): broadcast-add projected mean text (gated).
            text_gated = text_cond * has_text[:, None].to(text_cond.dtype)  # [B, D]
            x = x + text_gated[:, None, None, :]

        # --- 6. FiLM after text ---
        x = self.film_after_text(x, t_emb)

        # --- 7. Padded re-mask: padded slots/frames must be zero after each layer ---
        cm = coarse_mask[:, None, :, None].to(x.dtype)   # [B, 1, C, 1]
        fm = frame_mask[:, :, None, None].to(x.dtype)    # [B, T_lat, 1, 1]
        x = x * cm * fm
        return x


# ----------------------------------------------------------------------------
# Top-level denoiser
# ----------------------------------------------------------------------------

class GraphSaladDenoiser(nn.Module):
    """Phase-2 v1 graph-aware latent diffusion denoiser.

    Replaces ``GraphSaladDenoiserStub`` with the implementation per design doc.
    Stub forward signature is preserved as the positional contract; new optional
    arguments (``pooled_skeleton_embeddings``, ``has_text``) are keyword-only.
    """

    def __init__(
        self,
        d_model: int = 384,
        n_heads: int = 8,
        d_ff: int | None = None,
        n_layers: int = 5,
        d_text: int = 768,
        d_t: int | None = None,
        dropout: float = 0.1,
        text_mode: str = "mean_additive",
        text_token_dim: int = 768,
    ) -> None:
        super().__init__()
        if text_mode not in ("mean_additive", "token_cross_attn"):
            raise ValueError(
                f"text_mode must be 'mean_additive' or 'token_cross_attn', "
                f"got {text_mode!r}"
            )
        if n_layers % 2 == 0:
            raise ValueError(
                f"n_layers must be odd for SALAD skip-transformer "
                f"(enc + mid + dec); got {n_layers}"
            )
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must divide n_heads ({n_heads})")
        if d_ff is None:
            d_ff = 4 * d_model
        if d_t is None:
            d_t = d_model

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.d_text = d_text
        self.d_t = d_t
        self.text_mode = text_mode
        self.text_token_dim = text_token_dim

        # --- Timestep embedding (shared across all layers' FiLMs) ---
        self.t_sin = SinusoidalTimestepEmbedding(d_t)
        self.t_mlp = nn.Sequential(
            nn.Linear(d_t, d_t * 4),
            nn.SiLU(),
            nn.Linear(d_t * 4, d_t),
        )

        # --- Text projection (T5-base 768 → d_model); shared across layers ---
        # Per design §2.3: denoiser owns its own text_proj (NOT reusing VAE's).
        # mean_additive: projects the [B,768] mean-pooled caption.
        self.text_proj = nn.Linear(d_text, d_model)
        # token_cross_attn: separate projection for token-level T5 [B,L,768]→[B,L,D]
        # (exists only in token mode → mean-mode ckpts stay byte-identical).
        if text_mode == "token_cross_attn":
            self.text_token_proj = nn.Linear(text_token_dim, d_model)

        # --- Input projection: latent z + slot conditioning ---
        self.input_proj = nn.Linear(d_model, d_model)

        # --- Skip-transformer stack ---
        # n_layers = 2*depth + 1; depth pairs (enc[i], dec[i]) + 1 middle.
        # For n_layers=5: depth=2, layers = [enc0, enc1, mid, dec0, dec1] with
        # skip(enc1->dec0) and skip(enc0->dec1).
        self.layers = nn.ModuleList(
            [
                GraphSaladDenoiserLayer(d_model, n_heads, d_ff, d_t, dropout=dropout,
                                        text_mode=text_mode)
                for _ in range(n_layers)
            ]
        )
        # Skip mergers: one per decoder layer
        self.depth = (n_layers - 1) // 2
        self.skip_mergers = nn.ModuleList(
            [nn.Linear(2 * d_model, d_model) for _ in range(self.depth)]
        )

        # --- Output projection: D → D (predicts v_pred at same dim as z_t) ---
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        # Zero-init output_proj weights so initial v_pred ≈ 0 (training-stable
        # for diffusion; common practice — e.g. DiT, U-Net).
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        z_t: torch.Tensor,
        timesteps: torch.Tensor,
        text: torch.Tensor,
        adjacency: torch.Tensor,
        geodesic_dist: torch.Tensor,
        coarse_mask: torch.Tensor,
        frame_mask: torch.Tensor,
        level2_meta: dict | None = None,
        *,
        pooled_skeleton_embeddings: torch.Tensor | None = None,
        has_text: torch.Tensor | None = None,
        validate_inputs: bool = False,
        text_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns v_pred [B, T_lat, C, D].

        See module docstring for input/output contracts.

        Args:
            validate_inputs: passed to inner GraphAttentionBlock calls. Use True
                on the very first iter / sampling step to run the full ~14
                contract checks on pooled adjacency/geodesic/mask; thereafter
                pass False to keep the hot-path lean. (Per design §2.1 spatial
                block bullet — "冷启动验,热路径关".)
        """
        if z_t.dim() != 4:
            raise ValueError(f"z_t must be [B, T_lat, C, D], got {tuple(z_t.shape)}")
        B, T_lat, C, D = z_t.shape
        if D != self.d_model:
            raise ValueError(f"z_t last dim {D} != d_model {self.d_model}")
        if pooled_skeleton_embeddings is None:
            raise ValueError(
                "pooled_skeleton_embeddings is required (additive slot "
                "conditioning per design §2.4). Pass enc['pooled_skeleton_embeddings'] "
                "from vae.encode(...) or vae.encode_skeleton_only(...)."
            )
        if pooled_skeleton_embeddings.shape != (B, C, D):
            raise ValueError(
                f"pooled_skeleton_embeddings must be [B, C, D]=[{B},{C},{D}], "
                f"got {tuple(pooled_skeleton_embeddings.shape)}"
            )
        if has_text is None:
            raise ValueError(
                "has_text is required for CFG gating. Pass a [B] bool tensor; "
                "the trainer is responsible for cond_drop (flipping True→False "
                "with cond_drop_prob)."
            )
        if has_text.shape != (B,) or has_text.dtype != torch.bool:
            raise ValueError(
                f"has_text must be [B={B}] bool, got {tuple(has_text.shape)} "
                f"dtype {has_text.dtype}"
            )
        if self.text_mode == "mean_additive":
            if text.dim() != 2 or text.shape[0] != B or text.shape[1] != self.d_text:
                raise ValueError(
                    f"text must be [B={B}, d_text={self.d_text}] for mean_additive "
                    f"(mean-pooled cache); got {tuple(text.shape)}."
                )
            if text_token_mask is not None:
                raise ValueError(
                    "text_token_mask must be None in mean_additive mode."
                )
        else:  # token_cross_attn
            if (text.dim() != 3 or text.shape[0] != B
                    or text.shape[2] != self.text_token_dim):
                raise ValueError(
                    f"text must be [B={B}, L, text_token_dim={self.text_token_dim}] "
                    f"for token_cross_attn; got {tuple(text.shape)}."
                )
            L = text.shape[1]
            if (text_token_mask is None or text_token_mask.shape != (B, L)
                    or text_token_mask.dtype != torch.bool):
                raise ValueError(
                    f"token_cross_attn requires text_token_mask [B={B}, L={L}] bool, "
                    f"got {None if text_token_mask is None else tuple(text_token_mask.shape)}"
                    f"{'' if text_token_mask is None else ' ' + str(text_token_mask.dtype)}"
                )
            if text_token_mask.device != z_t.device:
                raise ValueError(
                    f"text_token_mask.device {text_token_mask.device} != "
                    f"z_t.device {z_t.device}"
                )
        if timesteps.shape != (B,):
            raise ValueError(f"timesteps must be [B={B}], got {tuple(timesteps.shape)}")
        if coarse_mask.shape != (B, C) or coarse_mask.dtype != torch.bool:
            raise ValueError(
                f"coarse_mask must be [B={B}, C={C}] bool, got "
                f"{tuple(coarse_mask.shape)} {coarse_mask.dtype}"
            )
        if frame_mask.shape != (B, T_lat) or frame_mask.dtype != torch.bool:
            raise ValueError(
                f"frame_mask must be [B={B}, T_lat={T_lat}] bool, got "
                f"{tuple(frame_mask.shape)} {frame_mask.dtype}"
            )
        # P1 (codex 2026-05-23): explicit shape/device/dtype for graph tensors.
        # Without these, a [1, C, C] graph would broadcast-expand into the
        # B-axis via `.unsqueeze().expand()` and silently condition multiple
        # samples on the wrong topology.
        ref_device = z_t.device
        if adjacency.shape != (B, C, C):
            raise ValueError(
                f"adjacency must be [B={B}, C={C}, C={C}], got {tuple(adjacency.shape)}"
            )
        if geodesic_dist.shape != (B, C, C):
            raise ValueError(
                f"geodesic_dist must be [B={B}, C={C}, C={C}], got {tuple(geodesic_dist.shape)}"
            )
        for name, t in (
            ("adjacency", adjacency),
            ("geodesic_dist", geodesic_dist),
            ("coarse_mask", coarse_mask),
            ("frame_mask", frame_mask),
            ("text", text),
            ("pooled_skeleton_embeddings", pooled_skeleton_embeddings),
            ("has_text", has_text),
            ("timesteps", timesteps),
        ):
            if t.device != ref_device:
                raise ValueError(
                    f"GraphSaladDenoiser: {name}.device {t.device} != z_t.device {ref_device}"
                )
        # float tensors must match z_t.dtype (denoiser is fp32-only — same
        # contract as GraphAttentionBlock)
        for name, t in (
            ("adjacency", adjacency),
            ("geodesic_dist", geodesic_dist),
            ("text", text),
            ("pooled_skeleton_embeddings", pooled_skeleton_embeddings),
        ):
            if t.dtype != z_t.dtype:
                raise ValueError(
                    f"GraphSaladDenoiser: {name}.dtype {t.dtype} != z_t.dtype {z_t.dtype}"
                )

        # --- Timestep embedding (shared by all FiLMs) ---
        t_emb = self.t_mlp(self.t_sin(timesteps))         # [B, D_t]

        # --- Text prep (mode-dependent) ---
        # mean_additive: project the [B,768] mean caption → [B,D], gated per-layer.
        # token_cross_attn: project tokens [B,L,768]→[B,L,D] + build the per-layer
        #   key_padding_mask (True = ignore): pad-token OR has_text=False. An
        #   all-masked (uncond) row is handled by TextCrossAttention (output→0).
        text_cond = None
        text_tokens = None
        text_key_padding_mask = None
        if self.text_mode == "mean_additive":
            text_cond = self.text_proj(text)              # [B, D]
        else:
            text_tokens = self.text_token_proj(text)      # [B, L, D]
            # valid key = token present AND has_text=True. key_padding_mask is
            # the inverse (True ⇒ mask). has_text=False ⇒ whole row masked.
            valid = text_token_mask & has_text[:, None]   # [B, L] bool
            text_key_padding_mask = ~valid                # [B, L] True=mask

        # --- Input projection: latent + slot conditioning (additive) ---
        x = self.input_proj(z_t)                          # [B, T_lat, C, D]
        x = x + pooled_skeleton_embeddings.unsqueeze(1).expand(-1, T_lat, -1, -1)
        # Apply input padded re-mask (defense in depth)
        cm = coarse_mask[:, None, :, None].to(x.dtype)
        fm = frame_mask[:, :, None, None].to(x.dtype)
        x = x * cm * fm

        # --- Skip-transformer: enc → mid → dec with paired skips ---
        # depth = (n_layers - 1) // 2 (e.g. 2 for n_layers=5)
        # layout: layers[0..depth-1] = encoders, layers[depth] = mid,
        #         layers[depth+1..2*depth] = decoders (in order)
        # skip pairing: dec[i] consumes encoder output enc[depth-1-i] (i.e.
        # symmetric around the middle).
        enc_outputs: list[torch.Tensor] = []
        for i in range(self.depth):
            x = self.layers[i](
                x, t_emb, text_cond, has_text,
                adjacency, geodesic_dist, coarse_mask, frame_mask,
                validate_inputs=validate_inputs,
                text_tokens=text_tokens,
                text_key_padding_mask=text_key_padding_mask,
            )
            enc_outputs.append(x)

        # middle layer
        x = self.layers[self.depth](
            x, t_emb, text_cond, has_text,
            adjacency, geodesic_dist, coarse_mask, frame_mask,
            validate_inputs=validate_inputs,
            text_tokens=text_tokens,
            text_key_padding_mask=text_key_padding_mask,
        )

        # decoder layers with skip
        for i in range(self.depth):
            dec_layer = self.layers[self.depth + 1 + i]
            skip = enc_outputs[self.depth - 1 - i]    # reverse pair
            x = torch.cat([x, skip], dim=-1)          # [B, T_lat, C, 2D]
            x = self.skip_mergers[i](x)               # [B, T_lat, C, D]
            x = dec_layer(
                x, t_emb, text_cond, has_text,
                adjacency, geodesic_dist, coarse_mask, frame_mask,
                validate_inputs=validate_inputs,
                text_tokens=text_tokens,
                text_key_padding_mask=text_key_padding_mask,
            )

        # --- Output: pre-norm + zero-init linear (init v_pred ≈ 0) ---
        x = self.output_norm(x)
        v_pred = self.output_proj(x)
        # Final padded re-mask
        v_pred = v_pred * cm * fm
        return v_pred
