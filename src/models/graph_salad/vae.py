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
from ..motion_decoder import MotionDecoder
from ..slot_norm import SlotNorm
from ..treeik_decoder import TreeIKBlock, fk_persample

from .pool_deterministic import DeterministicGraphPool
from .pool_dynamic import DynamicGraphPool
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
    ) -> None:
        super().__init__()
        if pool_type not in ("dynamic", "deterministic", "soft_deterministic", "none"):
            raise ValueError(
                f"pool_type must be 'dynamic'/'deterministic'/'soft_deterministic'/'none', "
                f"got {pool_type!r}"
            )
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
        self.motion_feat_dim = motion_feat_dim

        # ---- Encoder (reuse baseline params; ckpt-compatible) ----
        self.encoder = SkeletonEncoder(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            n_graph_layers=n_graph_layers,
            n_temporal_layers=n_enc_temporal_layers,
            joint_feat_dim=joint_feat_dim,
            motion_feat_dim=motion_feat_dim,
            temporal_kernel=temporal_kernel,
            dropout=dropout,
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
        self.decoder = MotionDecoder(
            d_model=d_model, n_heads=n_heads,
            n_cross_layers=n_cross_layers,
            n_temporal_layers=n_dec_temporal_layers,
            motion_feat_dim=motion_feat_dim,
            temporal_kernel=temporal_kernel,
            dropout=dropout,
        )
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

    @staticmethod
    def reparametrize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = mu + eps * exp(0.5 * logvar), eps ~ N(0, I)."""
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, batch: "GraphMotionBatch", sample: bool | None = None) -> dict:
        """Encoder → SlotNorm → Pool → Gaussian latent head.

        Returns dict with z, mu, logvar, plus graph_info needed by decode.

        Args:
            sample: if None (default), follows self.training (sample in train,
                    deterministic in eval). If True, force reparametrize.
                    If False, force z = mu.
        """
        if sample is None:
            sample = self.training
        # Encoder forward (reuse baseline)
        h0 = self.encoder(
            batch.motion_features,        # [B, T, J, 6]
            batch.skeleton_features,       # [B, J, 9]
            batch.adjacency,
            batch.geodesic_dist,
            batch.joint_mask,
            batch.frame_mask,
            name_hashes=batch.name_hashes,
        )  # [B, T, J, D]
        s_j = self.encoder.encode_skeleton(
            batch.skeleton_features,
            batch.adjacency,
            batch.geodesic_dist,
            batch.joint_mask,
            name_hashes=batch.name_hashes,
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
        }

    def decode(self, encode_out: dict, batch: "GraphMotionBatch") -> dict:
        """Unpool → TopoFKTreeIKDecoder → (pos, vel).

        Returns dict with pred_pos [B, T, J, 3], pred_vel [B, T, J, 3].
        """
        z = encode_out["z"]
        s_j = encode_out["s_j"]
        assignment = encode_out["assignment"]
        coarse_mask = encode_out["coarse_mask"]
        frame_mask_lat = encode_out["frame_mask_lat"]

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
            frame_mask_recovered = frame_mask_lat.repeat_interleave(self.temporal_stride, dim=-1)

        # Decoder: TopoFKTreeIKDecoder expects (slot_features [B,T,K,D], s_j,
        # asg [B,J,K], joint_mask, frame_mask, parents_list, rest_tensor, fps,
        # adjacency, geodesic_dist). For our case, K == J_max because we've
        # unpooled back; assignment is identity.
        B, T, J, D = h_fine.shape
        asg = _identity_assignment(batch.joint_mask)  # [B, J, J]

        # Inlined TreeIK forward (codex M1.3 R1 F1 fix — no TopoFKTreeIKDecoder
        # wrapper; we call MotionDecoder + treeik_head sequentially so ckpt
        # keys stay `decoder.*` 1:1 with baseline).
        # 1. MotionDecoder return_features=True → per-joint features [B,T,J,D]
        feats = self.decoder(
            h_fine,                # slot_features [B, T, K=J, D]
            s_j,                   # skeleton embeddings [B, J, D]
            asg,                   # assignment [B, J, K=J]
            batch.joint_mask,
            frame_mask_recovered,
            return_features=True,
        )  # [B, T, J, D]
        # 2. TreeIK head: rest_proj + blocks + rot_head + root + FK
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
        fm_b = frame_mask_recovered[:, :, None, None].to(pos.dtype)
        jm_b = batch.joint_mask[:, None, :, None].to(pos.dtype)
        pos = pos * fm_b * jm_b
        vel = vel * fm_b * jm_b

        return {
            "pred_pos": pos,        # [B, T, J, 3]
            "pred_vel": vel,        # [B, T, J, 3]
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
