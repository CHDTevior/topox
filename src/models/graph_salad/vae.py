"""GraphMotionVAE — graph-aware motion VAE wrapping all M1.2 components.

M1.3 wiring per plan §8 + user Q-B (TreeIK FK path). Single-level pool
(temporal_stride=4 in one shot) for cleanest M1.3 first version; 2-level
hierarchical pool deferred to M2 if needed (avoids level-2 anchor-derivation
ambiguity).

3-way ablation via `pool_type` arg:
  - "dynamic":       DynamicGraphPool (learned soft assign + MinCut)
  - "deterministic": DeterministicGraphPool (rule-based hard argmin)
  - "none":          No skeletal pool; J unchanged; temporal AvgPool×2 only

Pipeline:
  Encoder (SkeletonEncoder + SlotNorm) → h0 [B, T, J, D]
    ↓
  Pool path (stride=4):
    dynamic/deterministic: h0 → pool → h_lat [B, T/4, C, D]
    none:                  h0 → temporal-AvgPool×2 → h_lat [B, T/4, J, D]
    ↓
  Gaussian latent head: Linear(D, 2D) → mu, logvar; reparametrize → z
    ↓
  Unpool path:
    dynamic/deterministic: z → unpool → h_fine [B, T, J, D] (via assignment + temporal repeat)
    none:                  z → temporal-Upsample×2 → h_fine [B, T, J, D]
    ↓
  TopoFKTreeIKDecoder (rest_FiLM + TreeIKBlocks + rot head + FK):
    h_fine → rot6d → hard FK with rest_offsets + parents → joint_positions
    ↓
  Output: pred_pos [B, T, J, 3], pred_vel [B, T, J, 3]
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encoder import SkeletonEncoder
from ..motion_decoder import MotionDecoder, GraphTemporalDecoderLayer
from ..slot_norm import SlotNorm
from ..treeik_decoder import TreeIKBlock, fk_persample

from .pool_deterministic import DeterministicGraphPool
from .pool_dynamic import DynamicGraphPool
from .pool_edge_segment import EdgeSegmentPool
from .unpool import DynamicGraphUnpool


def _identity_assignment(joint_mask: torch.Tensor) -> torch.Tensor:
    """Build identity assignment P[b, j, j] = joint_mask[b, j] (others 0)."""
    B, J = joint_mask.shape
    eye = torch.eye(J, device=joint_mask.device, dtype=torch.float32)
    P = eye.unsqueeze(0).expand(B, J, J).clone()
    P = P * joint_mask[:, :, None].to(P.dtype) * joint_mask[:, None, :].to(P.dtype)
    return P


class GraphMotionVAE(nn.Module):
    """Graph-aware motion VAE with 3-way pool ablation switch.

    Args:
        pool_type: "dynamic" | "deterministic" | "none"
        d_model, n_heads, d_ff, ...: encoder/decoder dims (match baseline)
        max_coarse: pool max anchor count (only used for dynamic/deterministic)
        local_radius: pool candidate anchor radius
        temporal_stride: total T compression factor (4 = single-shot stride=4)
        motion_feat_dim: 6 = local_pos(3) + velocity(3)
    """

    def __init__(
        self,
        pool_type: str = "dynamic",
        *,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        n_graph_layers: int = 4,
        n_enc_temporal_layers: int = 2,
        n_cross_layers: int = 3,
        n_dec_temporal_layers: int = 2,
        n_treeik_layers: int = 3,
        max_coarse: int = 128,
        local_radius: int = 6,
        motion_feat_dim: int = 6,
        joint_feat_dim: int = 9,
        temporal_kernel: int = 9,
        temporal_stride: int = 4,
        dropout: float = 0.1,
        pool_tau: float | None = None,
        feat_mode: str = "fk6",
        attn_mode: str = "scalar",
        use_text: bool = False,
        decoder_mode: str = "unpool_identity",
        n_graph_temporal_layers: int = 4,
    ) -> None:
        super().__init__()
        if pool_type not in ("dynamic", "deterministic", "soft_deterministic", "edge_segment", "none"):
            raise ValueError(
                f"pool_type must be 'dynamic'/'deterministic'/'soft_deterministic'/"
                f"'edge_segment'/'none', got {pool_type!r}"
            )
        # feat_mode selects the motion representation + decoder head:
        #   "fk6"      — 6ch (local_pos+vel) input, FK decoder -> pred_pos/pred_vel
        #   "anytop13" — AnyTop native 13ch input, direct 13ch decoder (no FK)
        if feat_mode not in ("fk6", "anytop13"):
            raise ValueError(f"feat_mode must be 'fk6' or 'anytop13', got {feat_mode!r}")
        # decoder_mode selects how the latent reaches MotionDecoder:
        #   "unpool_identity" — legacy: DynamicGraphUnpool to fine joints first,
        #       then MotionDecoder with an IDENTITY assignment (cross-attention
        #       degenerates — every joint attends only to itself).
        #   "coarse_xattn"    — pass the coarse slots straight to MotionDecoder
        #       with the REAL pool assignment P, so its cross-attention attends
        #       each fine joint to its coarse anchors (bias = log P). Cross-joint
        #       info then flows via shared coarse slots.
        #   "graph_temporal" — coarse_xattn, then n_graph_temporal_layers extra
        #       AnyTop-style spatial(graph)+temporal layers refine the fine
        #       features (joint↔joint + long-range temporal coordination).
        if decoder_mode not in ("unpool_identity", "coarse_xattn", "graph_temporal"):
            raise ValueError(
                f"decoder_mode must be 'unpool_identity'/'coarse_xattn'/"
                f"'graph_temporal', got {decoder_mode!r}"
            )
        # graph_temporal needs the AnyTop graph tensors (graph_dist /
        # joint_relations) + the 13ch head — i.e. the anytop13 path only.
        if decoder_mode == "graph_temporal" and feat_mode != "anytop13":
            raise ValueError(
                f"decoder_mode='graph_temporal' requires feat_mode='anytop13', "
                f"got feat_mode={feat_mode!r}"
            )
        # attn_mode selects the encoder graph-attention bias:
        #   "scalar"     — GraphAttentionBlock (legacy scalar geo/adj bias)
        #   "graphormer" — AnyTopGraphAttentionBlock (per-edge-type embedding
        #                  bias; needs batch.anytop_graph_dist + joint_relations)
        if attn_mode not in ("scalar", "graphormer"):
            raise ValueError(f"attn_mode must be 'scalar' or 'graphormer', got {attn_mode!r}")
        # pool_tau only meaningful for soft_deterministic; explicit cross-check
        if pool_type == "soft_deterministic":
            if pool_tau is None or not (pool_tau > 0):
                raise ValueError(
                    f"pool_type=soft_deterministic requires pool_tau > 0, got {pool_tau}"
                )
        elif pool_tau is not None:
            raise ValueError(
                f"pool_tau only valid with pool_type=soft_deterministic; "
                f"got pool_type={pool_type!r}, pool_tau={pool_tau}"
            )
        if not isinstance(temporal_stride, int) or isinstance(temporal_stride, bool):
            raise TypeError("temporal_stride must be strict int")
        if temporal_stride < 1:
            raise ValueError(f"temporal_stride must be ≥ 1, got {temporal_stride}")
        # Codex M1.3 R4 fix: strict positive int on dim hparams
        for name, val in (
            ("d_model", d_model),
            ("n_heads", n_heads),
            ("d_ff", d_ff),
            ("n_graph_layers", n_graph_layers),
            ("n_enc_temporal_layers", n_enc_temporal_layers),
            ("n_cross_layers", n_cross_layers),
            ("n_dec_temporal_layers", n_dec_temporal_layers),
            ("n_treeik_layers", n_treeik_layers),
            ("motion_feat_dim", motion_feat_dim),
            ("joint_feat_dim", joint_feat_dim),
            ("temporal_kernel", temporal_kernel),
            ("n_graph_temporal_layers", n_graph_temporal_layers),
        ):
            if not isinstance(val, int) or isinstance(val, bool):
                raise TypeError(f"{name} must be strict int, got {type(val).__name__}")
            if val <= 0:
                raise ValueError(f"{name} must be > 0, got {val}")
        # Codex M1.3 round 1 F3 fixes:
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        # Codex M1.3 R5 fix: temporal_kernel must be odd (encoder.py:108 uses
        # `padding = (kernel_size - 1) // 2` for centered conv — even kernels
        # produce off-by-one length mismatches at forward time).
        if temporal_kernel % 2 == 0:
            raise ValueError(
                f"temporal_kernel must be odd, got {temporal_kernel} "
                f"(encoder uses centered conv padding)"
            )

        self.pool_type = pool_type
        self.d_model = d_model
        self.temporal_stride = temporal_stride
        self.feat_mode = feat_mode
        self.attn_mode = attn_mode
        self.use_text = use_text
        self.decoder_mode = decoder_mode
        # Optional text condition: project a precomputed 768-d T5 caption
        # embedding to d_model, injected post-unpool in decode().
        if use_text:
            self.text_proj = nn.Linear(768, d_model)
        else:
            self.text_proj = None
        # anytop13 always uses AnyTop's native 13 channels; fk6 keeps the
        # caller-supplied dim (6 = local_pos+vel). Derived so the caller cannot
        # silently mis-size the encoder/decoder.
        self.motion_feat_dim = 13 if feat_mode == "anytop13" else motion_feat_dim
        _enc_motion_mode = "anytop13_split" if feat_mode == "anytop13" else "fk6_shared"

        # ---- Encoder (reuse baseline params; ckpt-compatible) ----
        self.encoder = SkeletonEncoder(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            n_graph_layers=n_graph_layers,
            n_temporal_layers=n_enc_temporal_layers,
            joint_feat_dim=joint_feat_dim,
            motion_feat_dim=self.motion_feat_dim,
            temporal_kernel=temporal_kernel,
            dropout=dropout,
            motion_mode=_enc_motion_mode,
            attn_mode=attn_mode,
        )
        self.slot_norm = SlotNorm(d_model)

        # ---- Pool path (3-way) ----
        if pool_type == "dynamic":
            self.pool = DynamicGraphPool(
                d_model=d_model, max_coarse=max_coarse,
                local_radius=local_radius,
                temporal_stride=temporal_stride,
            )
        elif pool_type == "deterministic":
            self.pool = DeterministicGraphPool(
                d_model=d_model, max_coarse=max_coarse,
                local_radius=local_radius,
                temporal_stride=temporal_stride,
                tau=None,  # hard argmin
            )
        elif pool_type == "soft_deterministic":
            self.pool = DeterministicGraphPool(
                d_model=d_model, max_coarse=max_coarse,
                local_radius=local_radius,
                temporal_stride=temporal_stride,
                tau=pool_tau,
            )
        elif pool_type == "edge_segment":
            # v2 chain-segment pool — see pool_edge_segment.py docstring.
            # Hard 1-of-K assignment, segment-mean pooled_skeleton_embeddings,
            # zero aux_losses. Does NOT use local_radius (no anchor distance
            # semantics in v2). pool_tau also unused.
            self.pool = EdgeSegmentPool(
                d_model=d_model, max_coarse=max_coarse,
                temporal_stride=temporal_stride,
            )
        else:  # 'none'
            self.pool = None
            # No-pool: explicit temporal AvgPool for T compression
            self.temporal_pool = nn.AvgPool1d(
                kernel_size=temporal_stride, stride=temporal_stride
            )

        # ---- Gaussian latent head (shared Linear(D, 2D), NOT MultiLinear(...,7)) ----
        self.dist = nn.Linear(d_model, 2 * d_model)

        # ---- Unpool path ----
        if pool_type != "none":
            self.unpool = DynamicGraphUnpool(
                d_model=d_model, temporal_stride=temporal_stride,
            )
        else:
            self.unpool = None

        # ---- Decoder split (codex M1.3 R1 F1 fix for ckpt-compat) ----
        # Baseline `Model` stores MotionDecoder weights under `decoder.*`. Our
        # VAE must match that key path 1:1 for L6/ep399 ckpt-load.
        # Wrapping in TopoFKTreeIKDecoder would nest weights under
        # `decoder.base.*` → ckpt key mismatch. Instead: keep `decoder` as the
        # raw MotionDecoder (ckpt-compat), and place TreeIK head modules under
        # `treeik_head.*` (fresh init; not in baseline ckpt, "missing" allowed).
        # MotionDecoder is always called with return_features=True (both feat
        # modes — the feat_mode-specific head does the real output), so its own
        # `output_proj` is dead weight. It is kept ONLY for ckpt-key parity with
        # the baseline `Model` (which carries `decoder.output_proj` at [6, D]).
        # Build it at the fixed baseline dim 6 so warm-starting anytop13 from a
        # 6ch/fk6 ckpt does not hit an output_proj shape mismatch (codex P1).
        self.decoder = MotionDecoder(
            d_model=d_model, n_heads=n_heads,
            n_cross_layers=n_cross_layers,
            n_temporal_layers=n_dec_temporal_layers,
            motion_feat_dim=6,
            temporal_kernel=temporal_kernel,
            dropout=dropout,
        )
        if feat_mode == "fk6":
            # TreeIK head: rest_proj + blocks + rot_head + root (separately from
            # decoder so ckpt-compat is clean). Initialization matches
            # TopoFKTreeIKDecoder (identity-6D rot, zero root delta).
            d_ff_treeik = 4 * d_model
            max_hop_treeik = 16.0  # default from TopoFKTreeIKDecoder
            self.treeik_head = nn.ModuleDict({
                "rest_proj": nn.Linear(3, d_model),
                "blocks": nn.ModuleList([
                    TreeIKBlock(d_model, n_heads, d_ff_treeik, dropout, max_hop_treeik)
                    for _ in range(n_treeik_layers)
                ]),
                "rot_head": nn.Sequential(
                    nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 6)
                ),
                "root": nn.Sequential(
                    nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 3)
                ),
            })
            # Init: identity-6D rot at step 0, zero root delta (codex constraint)
            nn.init.zeros_(self.treeik_head["rot_head"][-1].weight)
            with torch.no_grad():
                self.treeik_head["rot_head"][-1].bias[:] = torch.tensor(
                    [1, 0, 0, 0, 1, 0], dtype=torch.float32
                )
            nn.init.zeros_(self.treeik_head["root"][-1].weight)
            nn.init.zeros_(self.treeik_head["root"][-1].bias)
            self.anytop13_head = None
        else:  # anytop13 — direct 13ch decoder head, AnyTop OutputProcess style
            # decoder runs with return_features=True (output_proj unused, mirrors
            # the fk6 pattern). Root/non-root get separate output Linears since
            # their 13 channels carry different semantics.
            self.treeik_head = None
            self.anytop13_head = nn.ModuleDict({
                "out_root": nn.Linear(d_model, 13),
                "out_nonroot": nn.Linear(d_model, 13),
            })

        # ---- graph_temporal decoder refine layers ----
        # Built ONLY for decoder_mode='graph_temporal' (None otherwise) so the
        # two legacy modes carry zero extra state_dict keys. Placed last in
        # __init__ so the legacy modes' parameter-init RNG sequence is unchanged.
        if decoder_mode == "graph_temporal":
            self.graph_temporal_layers = nn.ModuleList([
                GraphTemporalDecoderLayer(d_model, n_heads, d_ff, dropout)
                for _ in range(n_graph_temporal_layers)
            ])
        else:
            self.graph_temporal_layers = None

    @staticmethod
    def reparametrize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = mu + eps * exp(0.5 * logvar), eps ~ N(0, I)."""
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, batch: "GraphMotionBatch", sample: bool | None = None) -> dict:
        """Encoder → SlotNorm → Pool → Gaussian latent head.

        Returns the latent (z, mu, logvar) plus the graph metadata needed by
        both decode() AND a future Phase-2 graph-aware denoiser:
            z, mu, logvar, s_j, assignment, coarse_mask, frame_mask_lat,
            aux_losses, pooled_adjacency, pooled_geodesic, hard_assignment,
            pooled_skeleton_embeddings, anchor_indices.

        The Phase-2 keys (pooled_adjacency / pooled_geodesic / hard_assignment /
        pooled_skeleton_embeddings / anchor_indices) are pure intermediates the
        pool already computes — exposing them is wiring-only (no new compute,
        no parameter change, no ckpt-envelope change). A frozen Phase-1 VAE
        checkpoint loads strict=True unchanged.

        For pool_type='none' the contract is uniform: pooled_adjacency /
        pooled_geodesic = batch.adjacency / batch.geodesic_dist (C == J, each
        joint is its own slot); hard_assignment / anchor_indices are identity
        (arange), with anchor_indices padded entries set to -1 per the pool
        variants' convention.

        Args:
            sample: if None (default), follows self.training (sample in train,
                    deterministic in eval). If True, force reparametrize.
                    If False, force z = mu.
        """
        if sample is None:
            sample = self.training
        # Select the motion representation by feat_mode.
        #   fk6      -> motion_features [B, T, J, 6] (local_pos+vel)
        #   anytop13 -> anytop_x [B, J, 13, T] permuted to [B, T, J, 13]
        if self.feat_mode == "anytop13":
            if batch.anytop_x is None:
                raise ValueError(
                    "GraphMotionVAE(feat_mode=anytop13) requires batch.anytop_x; "
                    "use --dataset anytop_truebones"
                )
            motion_in = batch.anytop_x.permute(0, 3, 1, 2).contiguous()  # [B,T,J,13]
        else:
            motion_in = batch.motion_features                            # [B,T,J,6]
        # Graphormer-attn bias tensors (only when attn_mode=graphormer).
        if self.attn_mode == "graphormer":
            if batch.anytop_graph_dist is None or batch.anytop_joint_relations is None:
                raise ValueError(
                    "GraphMotionVAE(attn_mode=graphormer) requires "
                    "batch.anytop_graph_dist + batch.anytop_joint_relations; "
                    "use --dataset anytop_truebones"
                )
            _gd, _jr = batch.anytop_graph_dist, batch.anytop_joint_relations
        else:
            _gd, _jr = None, None
        # Encoder forward (reuse baseline)
        h0 = self.encoder(
            motion_in,                     # [B, T, J, motion_feat_dim]
            batch.skeleton_features,       # [B, J, 9]
            batch.adjacency,
            batch.geodesic_dist,
            batch.joint_mask,
            batch.frame_mask,
            name_hashes=batch.name_hashes,
            clip_embeddings=getattr(batch, "joint_semantics", None),
            graph_dist=_gd, joint_relations=_jr,
        )  # [B, T, J, D]
        s_j = self.encoder.encode_skeleton(
            batch.skeleton_features,
            batch.adjacency,
            batch.geodesic_dist,
            batch.joint_mask,
            name_hashes=batch.name_hashes,
            clip_embeddings=getattr(batch, "joint_semantics", None),
            graph_dist=_gd, joint_relations=_jr,
        )  # [B, J, D]
        h0 = self.slot_norm(h0)

        if self.pool is not None:
            # Pool path
            pool_out = self.pool(
                joint_features=h0,
                skeleton_embeddings=s_j,
                adjacency=batch.adjacency,
                geodesic_dist=batch.geodesic_dist,
                joint_mask=batch.joint_mask,
                frame_mask=batch.frame_mask,
                parent_indices=batch.parent_indices,
            )
            h_lat = pool_out["pooled_features"]   # [B, T_lat, C, D]
            coarse_mask = pool_out["pooled_mask"]
            frame_mask_lat = pool_out["frame_mask_down"]
            assignment = pool_out["assignment"]
            aux_losses = pool_out["aux_losses"]
            # M2 Phase-2 wiring: surface the pool's coarse-graph metadata so a
            # downstream graph-aware denoiser can condition on it + sampling can
            # decode without recomputing. Pure pass-through of intermediates the
            # pool already produced.
            pooled_adjacency = pool_out["pooled_adjacency"]                  # [B, C, C]
            pooled_geodesic = pool_out["pooled_geodesic"]                    # [B, C, C]
            hard_assignment = pool_out["hard_assignment"]                    # [B, J] long
            pooled_skeleton_embeddings = pool_out["pooled_skeleton_embeddings"]  # [B, C, D]
            anchor_indices = pool_out["anchor_indices"]                      # [B, C] long, -1 = padded
        else:
            # No-pool: temporal compress only; skeletal dim unchanged
            B, T, J, D = h0.shape
            # Codex M1.3 R1 F3 fix: T must divide temporal_stride
            if T % self.temporal_stride != 0:
                raise ValueError(
                    f"GraphMotionVAE no-pool: T={T} must be divisible by "
                    f"temporal_stride={self.temporal_stride}"
                )
            T_lat = T // self.temporal_stride
            # [B, T, J, D] → [B*J*D, 1, T]
            h_flat = h0.permute(0, 2, 3, 1).reshape(B * J * D, 1, T)
            h_down = self.temporal_pool(h_flat)  # [B*J*D, 1, T_lat]
            h_lat = h_down.reshape(B, J, D, T_lat).permute(0, 3, 1, 2).contiguous()
            # coarse_mask = joint_mask (no skeletal compression)
            coarse_mask = batch.joint_mask  # [B, J]
            # frame_mask_lat: conservative AND across stride frames
            frame_mask_lat = batch.frame_mask.view(B, T_lat, self.temporal_stride).all(dim=-1)
            # Identity assignment (J × J)
            assignment = _identity_assignment(batch.joint_mask)
            aux_losses = {
                "mincut": torch.zeros((), device=h0.device, dtype=h0.dtype),
                "mincut_cut": torch.zeros((), device=h0.device, dtype=h0.dtype),
                "mincut_ortho": torch.zeros((), device=h0.device, dtype=h0.dtype),
                "locality": torch.zeros((), device=h0.device, dtype=h0.dtype),
                "entropy": torch.zeros((), device=h0.device, dtype=h0.dtype),
            }
            # M2 Phase-2 wiring: no-pool identity equivalents (C == J, each
            # joint is its own slot). Mirrors the pool variants' schema so
            # encode()'s contract is uniform across pool_type.
            arange_j = torch.arange(
                J, device=h0.device, dtype=torch.long
            ).unsqueeze(0).expand(B, J)
            pooled_adjacency = batch.adjacency                               # [B, J, J]
            pooled_geodesic = batch.geodesic_dist                            # [B, J, J]
            hard_assignment = arange_j.contiguous()                          # [B, J] j -> j
            pooled_skeleton_embeddings = s_j                                 # [B, J, D]
            anchor_indices = torch.where(
                batch.joint_mask, arange_j,
                torch.full_like(arange_j, -1),
            )                                                                 # [B, J] -1 = padded

        # Gaussian latent head
        dist_out = self.dist(h_lat)  # [B, T_lat, C, 2D]
        mu, logvar = dist_out.chunk(2, dim=-1)  # each [B, T_lat, C, D]

        # Reparametrize (only when sampling; in eval, z = mu for deterministic metrics)
        if sample:
            z = self.reparametrize(mu, logvar)
        else:
            z = mu

        # Mask latent (defense in depth — padded coarse/frame should be 0)
        z = z * coarse_mask[:, None, :, None].to(z.dtype)
        z = z * frame_mask_lat[:, :, None, None].to(z.dtype)

        return {
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "s_j": s_j,
            "assignment": assignment,
            "coarse_mask": coarse_mask,
            "frame_mask_lat": frame_mask_lat,
            "aux_losses": aux_losses,
            # M2 Phase-2 wiring (graph-aware denoiser conditioning + decode-time
            # sampling). forward() auto-propagates via {**enc, **dec, ...}.
            "pooled_adjacency": pooled_adjacency,
            "pooled_geodesic": pooled_geodesic,
            "hard_assignment": hard_assignment,
            "pooled_skeleton_embeddings": pooled_skeleton_embeddings,
            "anchor_indices": anchor_indices,
        }

    def encode_skeleton_only(self, batch: "GraphMotionBatch") -> dict:
        """Compute coarse-graph metadata (motion-independent pool outputs)
        without consuming motion frames. Used by the Phase-2 graph-aware
        denoiser at sampling time, when motion is itself the unknown being
        denoised and z_T is initialized from N(0,I).

        Returns a schema-aligned subset of encode()'s Phase-2 keys, plus s_j
        and the soft assignment matrix:
            s_j, pooled_adjacency, pooled_geodesic, coarse_mask,
            pooled_skeleton_embeddings, anchor_indices, hard_assignment, assignment

        For pool variants: delegates to pool.compute_assignment_and_graph(...).
        For pool_type='none': synthesizes identity equivalents (same convention
        as encode()'s else branch — C == J, each joint is its own slot).

        Caller is responsible for the freeze contract (vae.eval() +
        with torch.no_grad(): ...) at the Phase-2 denoiser training/sampling site.

        Note: when called on a DDP-wrapped model, dispatch through
        `vae.module.encode_skeleton_only(...)` — PyTorch's DDP only proxies
        `forward()` on the wrapper.
        """
        # Graphormer-attn bias tensors (only when attn_mode=graphormer)
        if self.attn_mode == "graphormer":
            if batch.anytop_graph_dist is None or batch.anytop_joint_relations is None:
                raise ValueError(
                    "GraphMotionVAE(attn_mode=graphormer) requires "
                    "batch.anytop_graph_dist + batch.anytop_joint_relations; "
                    "use --dataset anytop_truebones"
                )
            _gd, _jr = batch.anytop_graph_dist, batch.anytop_joint_relations
        else:
            _gd, _jr = None, None

        s_j = self.encoder.encode_skeleton(
            batch.skeleton_features,
            batch.adjacency,
            batch.geodesic_dist,
            batch.joint_mask,
            name_hashes=batch.name_hashes,
            clip_embeddings=getattr(batch, "joint_semantics", None),
            graph_dist=_gd, joint_relations=_jr,
        )  # [B, J, D]

        if self.pool is not None:
            geom = self.pool.compute_assignment_and_graph(
                skeleton_embeddings=s_j,
                adjacency=batch.adjacency,
                geodesic_dist=batch.geodesic_dist,
                joint_mask=batch.joint_mask,
                parent_indices=batch.parent_indices,
            )
            return {
                "s_j": s_j,
                # Phase-2 keys (schema-aligned with encode()'s return)
                "pooled_adjacency": geom["pooled_adjacency"],
                "pooled_geodesic": geom["pooled_geodesic"],
                "coarse_mask": geom["pooled_mask"],
                "pooled_skeleton_embeddings": geom["pooled_skeleton_embeddings"],
                "anchor_indices": geom["anchor_indices"],
                "hard_assignment": geom["hard_assignment"],
                "assignment": geom["assignment"],
            }
        else:
            # No-pool: identity equivalents (matches encode()'s else branch).
            B, J, _ = s_j.shape
            arange_j = torch.arange(
                J, device=s_j.device, dtype=torch.long
            ).unsqueeze(0).expand(B, J)
            return {
                "s_j": s_j,
                "pooled_adjacency": batch.adjacency,
                "pooled_geodesic": batch.geodesic_dist,
                "coarse_mask": batch.joint_mask,
                "pooled_skeleton_embeddings": s_j,
                "anchor_indices": torch.where(
                    batch.joint_mask, arange_j,
                    torch.full_like(arange_j, -1),
                ),
                "hard_assignment": arange_j.contiguous(),
                "assignment": _identity_assignment(batch.joint_mask),
            }

    def decode(self, encode_out: dict, batch: "GraphMotionBatch") -> dict:
        """Unpool → decoder head.

        fk6:      → TreeIK FK → pred_pos/pred_vel [B, T, J, 3].
        anytop13: → direct 13ch regression → pred_motion [B, T, J, 13].
        The return dict always carries pred_pos/pred_vel/pred_motion; the
        keys not produced by the active feat_mode are None.
        """
        z = encode_out["z"]
        s_j = encode_out["s_j"]
        assignment = encode_out["assignment"]
        coarse_mask = encode_out["coarse_mask"]
        frame_mask_lat = encode_out["frame_mask_lat"]

        # ---- Optional text conditioning ----
        # Project the T5 caption embedding once; each decoder_mode branch
        # broadcasts it into whichever tensor feeds the decoder. Gated by
        # has_text so a missing-caption sample contributes nothing (a zero
        # embedding still picks up text_proj's bias, hence the explicit gate).
        text_vec = None
        if self.use_text:
            if batch.caption_emb is None or batch.has_text is None:
                raise ValueError(
                    "GraphMotionVAE(use_text=True) requires batch.caption_emb "
                    "and batch.has_text"
                )
            text_vec = self.text_proj(batch.caption_emb)            # [B, D]
            text_vec = text_vec * batch.has_text[:, None].to(text_vec.dtype)

        if self.decoder_mode in ("coarse_xattn", "graph_temporal"):
            # Pass the coarse slots + the REAL pool assignment P straight to
            # MotionDecoder. Its step-1 einsum does the P-weighted unpool, and
            # — crucially — the cross-attention layers KEEP using P (bias =
            # log P), so each fine joint attends to its coarse anchors instead
            # of being locked to itself by an identity assignment.
            # DynamicGraphUnpool is not used on this path.
            # graph_temporal enters here too, then refines `feats` with the
            # extra spatial+temporal layers (see the block after the if/else).
            z_up = z.repeat_interleave(self.temporal_stride, dim=1)   # [B,T,C,D]
            frame_mask_recovered = frame_mask_lat.repeat_interleave(
                self.temporal_stride, dim=-1)
            if text_vec is not None:
                z_up = z_up + text_vec[:, None, None, :]              # broadcast (T,C)
            feats = self.decoder(
                z_up,                  # slot_features [B, T, C, D]
                s_j,                   # skeleton embeddings [B, J, D]
                assignment,            # REAL pool assignment P [B, J, C]
                batch.joint_mask,
                frame_mask_recovered,
                return_features=True,
            )  # [B, T, J, D]
        else:  # "unpool_identity" — legacy: unpool to fine joints, identity decoder
            if self.unpool is not None:
                unpool_out = self.unpool(
                    coarse_features=z,
                    assignment=assignment,
                    joint_mask=batch.joint_mask,
                    coarse_mask=coarse_mask,
                    frame_mask_down=frame_mask_lat,
                )
                h_fine = unpool_out["fine_features"]   # [B, T, J, D]
                frame_mask_recovered = unpool_out["frame_mask_up"]
            else:
                # No-pool: temporal upsample only
                B, T_lat, J, D = z.shape
                T_full = T_lat * self.temporal_stride
                z_flat = z.permute(0, 2, 3, 1).reshape(B * J * D, 1, T_lat)
                h_up = z_flat.repeat_interleave(self.temporal_stride, dim=-1)
                h_fine = h_up.reshape(B, J, D, T_full).permute(0, 3, 1, 2).contiguous()
                frame_mask_recovered = frame_mask_lat.repeat_interleave(
                    self.temporal_stride, dim=-1)
            if text_vec is not None:
                h_fine = h_fine + text_vec[:, None, None, :]          # broadcast (T,J)
            # MotionDecoder with an IDENTITY assignment (K == J): the cross-
            # attention degenerates to per-joint self-refinement.
            asg = _identity_assignment(batch.joint_mask)  # [B, J, J]
            feats = self.decoder(
                h_fine,                # slot_features [B, T, K=J, D]
                s_j,                   # skeleton embeddings [B, J, D]
                asg,                   # identity assignment [B, J, K=J]
                batch.joint_mask,
                frame_mask_recovered,
                return_features=True,
            )  # [B, T, J, D]

        if self.decoder_mode == "graph_temporal":
            # Refine the fine features with AnyTop-style spatial(graph)+temporal
            # layers — the joint↔joint + long-range temporal coordination
            # MotionDecoder lacks. Each layer re-masks padded joints/frames so
            # `feats` stays clean by construction.
            if batch.anytop_graph_dist is None or batch.anytop_joint_relations is None:
                raise ValueError(
                    "decoder_mode='graph_temporal' requires batch.anytop_graph_dist "
                    "and batch.anytop_joint_relations (use --dataset anytop_truebones)"
                )
            for layer in self.graph_temporal_layers:
                feats = layer(
                    feats,
                    batch.anytop_graph_dist,
                    batch.anytop_joint_relations,
                    batch.joint_mask,
                    frame_mask_recovered,
                )

        fm_b = frame_mask_recovered[:, :, None, None].to(feats.dtype)
        jm_b = batch.joint_mask[:, None, :, None].to(feats.dtype)

        if self.feat_mode == "anytop13":
            # Direct 13ch regression, AnyTop OutputProcess style: root joint
            # and non-root joints use separate output Linears.
            out_root = self.anytop13_head["out_root"](feats[:, :, 0:1, :])     # [B,T,1,13]
            out_nonroot = self.anytop13_head["out_nonroot"](feats[:, :, 1:, :])  # [B,T,J-1,13]
            pred_motion = torch.cat([out_root, out_nonroot], dim=2)            # [B,T,J,13]
            pred_motion = pred_motion * fm_b * jm_b
            return {
                "pred_motion": pred_motion,   # [B, T, J, 13]
                "pred_pos": None,
                "pred_vel": None,
                "frame_mask_recovered": frame_mask_recovered,
            }

        # fk6: TreeIK head — rest_proj + blocks + rot_head + root + FK.
        rest_embed = self.treeik_head["rest_proj"](batch.rest_offsets)  # [B, J, D]
        x = feats
        for blk in self.treeik_head["blocks"]:
            x = blk(x, rest_embed, batch.adjacency, batch.geodesic_dist,
                   batch.joint_mask, frame_mask_recovered)
        r6 = self.treeik_head["rot_head"](x)                              # [B, T, J, 6]
        root_local = self.treeik_head["root"](feats[:, :, 0])             # [B, T, 3]
        # Hard FK
        pos = fk_persample(r6, root_local, batch.parent_indices,
                          batch.rest_offsets, batch.joint_mask)            # [B, T, J, 3]
        # Velocity from numerical diff (matches TopoFKTreeIKDecoder)
        vel = torch.zeros_like(pos)
        if pos.shape[1] > 1:
            vel[:, 1:] = (pos[:, 1:] - pos[:, :-1]) * batch.fps.view(-1, 1, 1, 1)
            vel[:, 0] = vel[:, 1]
        # Codex M1.3 R6 fix: reapply frame_mask + joint_mask on pos/vel so
        # padded frames + joints output exactly zero (FK + numerical-diff
        # otherwise leaks non-zero values into padded regions).
        pos = pos * fm_b * jm_b
        vel = vel * fm_b * jm_b

        return {
            "pred_pos": pos,        # [B, T, J, 3]
            "pred_vel": vel,        # [B, T, J, 3]
            "pred_motion": None,
            "frame_mask_recovered": frame_mask_recovered,
        }

    def forward(self, batch: "GraphMotionBatch", sample: bool | None = None) -> dict:
        """Full VAE forward: encode → decode. Returns combined dict.

        Args:
            sample: controls reparametrization. If None (default), uses
                    self.training (sample in train, deterministic in eval).
        """
        if sample is None:
            sample = self.training
        enc = self.encode(batch, sample=sample)
        dec = self.decode(enc, batch)
        return {
            **enc,
            **dec,
            "pool_aux_outputs": [enc["aux_losses"]],
        }
