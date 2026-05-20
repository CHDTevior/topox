"""DynamicGraphPool — soft-assignment skeleton pool per plan §6.

For each sample in the batch:
1. Anchors derived from `parent_indices[b]` via `graph_utils.find_anchors_rulebased`
   (root + degree≥2 branch + leaf + chain-chunked long limbs).
2. Soft assignment P[b, j, c] = softmax_c( Wq(s_j)·Wk(s_a[c]) / sqrt(d_model)
                                           - locality_alpha * geodesic[b, j, anchor[c]] )
   restricted to a graph-local candidate mask (j may only assign to anchors
   within `local_radius` hops).
3. Mass-normalized feature pool:
       h_pool[b, t, c, d] = (Σ_j P[b,j,c] · h[b,t,j,d]) / mass[b, c]
   with `mass[b, c] = Σ_j P[b,j,c]` clamped to a small positive floor.
4. Temporal downsample T → T // temporal_stride via avg-pool over valid frames.
5. Pooled adjacency from argmax(P) via `graph_utils.build_coarse_adjacency_from_hard_assign`.
6. Pooled geodesic via `graph_utils.floyd_shortest_path` on pooled adjacency.

Aux losses (returned in `aux_losses` dict, weighted by caller):
  - mincut: spectral relaxation of normalized cut + orthogonality (MinCutPool,
    Bianchi 2020). Single λ=0.5 (paper default, user 2026-05-20 decision).
  - locality: Σ_jc P[b,j,c] · geodesic[b,j,anchor[c]] — penalises long-range
    cluster membership (plan §13.3 retained).
  - entropy: per-row entropy of P — kept moderate; collapse-to-hard is OK,
    fully-uniform is uninformative (plan §13.3 retained).

Coarse-node ordering invariant (treeik_decoder §3.3 / codex M1.0 R12):
    anchor list per sample is sorted ascending by joint index. This preserves
    the "root=0 + parents[j]<j" property of the coarse graph WHEN the input
    fine graph satisfies it (which GraphMotionBatch already validates).

Outputs (forward returns dict):
    pooled_features:          [B, T_lat, C_max, D]   (T_lat = T // temporal_stride)
    assignment:               [B, J_max, C_max]      (soft, masked to candidate set)
    hard_assignment:          [B, J_max]             (argmax along C; padded → 0)
    pooled_adjacency:         [B, C_max, C_max]      binary {0, 1}
    pooled_geodesic:          [B, C_max, C_max]      Floyd hop count (or +Inf)
    pooled_mask:              [B, C_max] bool
    pooled_skeleton_embeddings: [B, C_max, D]
    frame_mask_down:          [B, T_lat] bool
    anchor_indices:           [B, C_max] long, -1 for padded
    aux_losses:               dict[str, Tensor]
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_utils import (
    assert_root_first_parent_order,
    build_coarse_adjacency_from_hard_assign,
    find_anchors_rulebased,
    floyd_shortest_path,
)


_LARGE_NEG = -1e9
_MASS_FLOOR = 1e-8


class DynamicGraphPool(nn.Module):
    """Soft-assignment graph pool with per-sample dynamic anchors.

    Args:
        d_model: feature dim of joint / anchor embeddings.
        max_coarse: hard upper bound on number of coarse nodes per sample.
            Per-sample anchor counts can be < max_coarse; remainder padded.
            Must be ≥ 1.
        local_radius: candidate anchor mask radius (in graph hops).
            Each fine joint may only assign to anchors within this geodesic
            distance. Default 3 — covers most rule-derived anchors per plan §6.2.
        max_chain_chunk_len: passed through to find_anchors_rulebased.
        locality_alpha: weight on `-geodesic(j, anchor)` in assignment score.
            Default 1.0.
        temperature: softmax temperature on assignment scores. Default 1.0.
        temporal_stride: T → T // stride downsample factor. Default 2.
        mincut_lambda: weight on MinCut cut+ortho aux loss. Default 0.5
            (paper default; user 2026-05-20 fixed-no-warmup decision).
    """

    def __init__(
        self,
        d_model: int,
        max_coarse: int,
        *,
        local_radius: int = 3,
        max_chain_chunk_len: int = 5,
        locality_alpha: float = 1.0,
        temperature: float = 1.0,
        temporal_stride: int = 2,
        mincut_lambda: float = 0.5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # Strict int (reject bool since bool ⊂ int) — codex M1.2 round 9 R12 #1/#2
        if not isinstance(d_model, int) or isinstance(d_model, bool):
            raise TypeError(f"d_model must be strict int, got {type(d_model).__name__}")
        if not isinstance(max_coarse, int) or isinstance(max_coarse, bool):
            raise TypeError(f"max_coarse must be strict int, got {type(max_coarse).__name__}")
        for name, val in (
            ("local_radius", local_radius),
            ("max_chain_chunk_len", max_chain_chunk_len),
            ("temporal_stride", temporal_stride),
        ):
            if not isinstance(val, int) or isinstance(val, bool):
                raise TypeError(f"{name} must be strict int, got {type(val).__name__}")
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")
        if max_coarse < 1:
            raise ValueError(f"max_coarse must be ≥ 1, got {max_coarse}")
        if local_radius < 1:
            raise ValueError(f"local_radius must be ≥ 1, got {local_radius}")
        if max_chain_chunk_len < 1:
            raise ValueError(f"max_chain_chunk_len must be ≥ 1, got {max_chain_chunk_len}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if temporal_stride < 1:
            raise ValueError(f"temporal_stride must be ≥ 1, got {temporal_stride}")
        # Reject non-finite scalar hparams (codex M1.2 round 12).
        for name, val in (
            ("locality_alpha", locality_alpha),
            ("temperature", temperature),
            ("mincut_lambda", mincut_lambda),
        ):
            if not math.isfinite(float(val)):
                raise ValueError(
                    f"DynamicGraphPool: {name} must be finite, got {val}"
                )

        self.d_model = d_model
        self.max_coarse = max_coarse
        self.local_radius = local_radius
        self.max_chain_chunk_len = max_chain_chunk_len
        self.locality_alpha = locality_alpha
        self.temperature = temperature
        self.temporal_stride = temporal_stride
        self.mincut_lambda = mincut_lambda

        # Q/K projections for bilinear assignment scoring
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)

    # ------------------------------------------------------------------ #
    # Anchor selection (per-sample, CPU; cached not needed — cheap)      #
    # ------------------------------------------------------------------ #
    def _select_anchors(
        self,
        parent_indices: list[list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-sample anchor indices.

        Args:
            parent_indices: list[list[int]] of length B. Each inner list is
                length num_joints[b] (un-padded), already validated by caller.

        Returns:
            anchor_indices: [B, C_max] long; -1 in padded slots.
            coarse_mask:    [B, C_max] bool; True for valid anchor slots.
        """
        B = len(parent_indices)
        anchor_indices = torch.full((B, self.max_coarse), -1, dtype=torch.long)
        coarse_mask = torch.zeros(B, self.max_coarse, dtype=torch.bool)
        for b in range(B):
            anchors = find_anchors_rulebased(
                parent_indices[b],
                max_chain_chunk_len=self.max_chain_chunk_len,
            )
            if len(anchors) > self.max_coarse:
                raise ValueError(
                    f"DynamicGraphPool: sample {b} produced {len(anchors)} anchors "
                    f"> max_coarse={self.max_coarse}; raise max_coarse or tighten "
                    f"max_chain_chunk_len"
                )
            for c, a in enumerate(anchors):
                anchor_indices[b, c] = a
                coarse_mask[b, c] = True
        return anchor_indices, coarse_mask

    # ------------------------------------------------------------------ #
    # Soft assignment with graph-local candidate mask                    #
    # ------------------------------------------------------------------ #
    def _compute_assignment(
        self,
        skeleton_embeddings: torch.Tensor,  # [B, J, D]
        geodesic_dist: torch.Tensor,        # [B, J, J]
        joint_mask: torch.Tensor,           # [B, J]
        anchor_indices: torch.Tensor,       # [B, C_max]
        coarse_mask: torch.Tensor,          # [B, C_max]
    ) -> torch.Tensor:
        """Soft assignment P[b, j, c] over the candidate-anchor mask.

        Score: Wq(s_j) · Wk(s_anchor[c]) / sqrt(d) - alpha * geodesic[j, anchor[c]]
        Candidate mask: geodesic ≤ local_radius AND coarse_mask[c] AND joint_mask[j].
        Softmax over c on the candidate set per (b, j).

        Returns: P [B, J, C_max]; zero rows for padded joints; zero columns
        for padded anchors; rows that have ≥1 candidate sum to 1.
        """
        B, J, D = skeleton_embeddings.shape
        C = self.max_coarse
        device = skeleton_embeddings.device

        # Joint queries Wq(s_j) — [B, J, D]
        q = self.q_proj(skeleton_embeddings)
        # Anchor embeddings: gather from skeleton_embeddings at anchor positions.
        # For padded anchors (anchor_indices == -1), gather safely at idx 0 then mask.
        safe_anchor = anchor_indices.clamp(min=0)  # [B, C]
        idx = safe_anchor.unsqueeze(-1).expand(-1, -1, D)  # [B, C, D]
        anchor_embed = torch.gather(skeleton_embeddings, dim=1, index=idx)  # [B, C, D]
        # Anchor keys
        k = self.k_proj(anchor_embed)  # [B, C, D]

        # Bilinear score: [B, J, C]
        score = torch.einsum("bjd,bcd->bjc", q, k) / math.sqrt(D)

        # Locality penalty: -alpha * geodesic(j, anchor[c])
        # gather geodesic_dist[b, j, anchor[c]] → [B, J, C]
        gather_idx = anchor_indices.unsqueeze(1).expand(-1, J, -1).clamp(min=0)  # [B, J, C]
        # gather column anchor[c] from geodesic_dist[b, j, :]
        geo_jc = torch.gather(geodesic_dist, dim=2, index=gather_idx)  # [B, J, C]
        # Substitute +Inf with a large finite value (NaN/-Inf were rejected at
        # forward entry; only +Inf can appear here, from Floyd unreachable pairs).
        geo_jc_safe = geo_jc.clone()
        geo_jc_safe[torch.isposinf(geo_jc_safe)] = float(2 * self.local_radius + 1)
        score = score - self.locality_alpha * geo_jc_safe

        # Candidate mask: True iff (geodesic ≤ local_radius) AND coarse_mask[c] AND joint_mask[j].
        in_radius = geo_jc <= float(self.local_radius)
        candidate = (
            in_radius
            & coarse_mask.unsqueeze(1)        # [B, 1, C]
            & joint_mask.unsqueeze(-1)        # [B, J, 1]
        )

        # Softmax over C on candidate set per (b, j). Rows with no candidate
        # (rare — only if anchor set doesn't cover a joint within local_radius)
        # fall back to: all softmax inputs masked → uniform on candidate set
        # would be NaN. We handle this loudly: a fine joint with NO reachable
        # anchor inside local_radius is a configuration error.
        no_candidate = ~candidate.any(dim=-1)  # [B, J]
        no_candidate_for_valid = no_candidate & joint_mask
        if no_candidate_for_valid.any():
            bad = no_candidate_for_valid.nonzero(as_tuple=False)
            b0, j0 = bad[0, 0].item(), bad[0, 1].item()
            raise ValueError(
                f"DynamicGraphPool: sample {b0} joint {j0} has no candidate anchor "
                f"within local_radius={self.local_radius} hops; increase "
                f"local_radius or revise anchor rules so every joint has at "
                f"least one anchor in radius"
            )

        score = score / self.temperature
        score = score.masked_fill(~candidate, _LARGE_NEG)
        P = F.softmax(score, dim=-1)
        # Zero rows for padded joints + columns for padded anchors (defensive;
        # softmax may produce small non-zero on padded due to fp).
        P = P * joint_mask.unsqueeze(-1).to(P.dtype)
        P = P * coarse_mask.unsqueeze(1).to(P.dtype)
        return P

    # ------------------------------------------------------------------ #
    # Feature pool + temporal downsample                                  #
    # ------------------------------------------------------------------ #
    def _pool_features(
        self,
        joint_features: torch.Tensor,  # [B, T, J, D]
        P: torch.Tensor,               # [B, J, C]
        frame_mask: torch.Tensor,      # [B, T]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mass-normalized skeletal pool + temporal AvgPool1d downsample.

        Returns:
            pooled_features: [B, T_lat, C, D]
            frame_mask_down: [B, T_lat]
        """
        # Skeletal pool
        # h_pool[b, t, c, d] = sum_j P[b,j,c] * h[b,t,j,d]
        h_pool = torch.einsum("bjc,btjd->btcd", P, joint_features)
        mass = P.sum(dim=1).clamp(min=_MASS_FLOOR)  # [B, C]
        h_pool = h_pool / mass.unsqueeze(1).unsqueeze(-1)  # [B, T, C, D]

        # Temporal downsample
        B, T, C, D = h_pool.shape
        s = self.temporal_stride
        T_lat = T // s
        if T_lat == 0:
            raise ValueError(
                f"DynamicGraphPool: temporal_stride={s} > T={T} produces "
                f"T_lat=0; reduce stride or pad frames"
            )
        # Reshape [B, T, C, D] → [B*C*D, T] for 1D pool, then back
        h_pool_flat = h_pool.permute(0, 2, 3, 1).reshape(B * C * D, 1, T)
        h_down = F.avg_pool1d(h_pool_flat, kernel_size=s, stride=s)  # [B*C*D, 1, T_lat]
        h_down = h_down.reshape(B, C, D, T_lat).permute(0, 3, 1, 2).contiguous()
        # frame_mask downsample: a downsampled frame is valid iff ALL `s` source
        # frames are valid (conservative; alternatives are valid iff ANY source
        # is valid — we use ALL to match recon contract).
        fm = frame_mask.view(B, T_lat, s).all(dim=-1)
        return h_down, fm

    # ------------------------------------------------------------------ #
    # Pooled graph (adjacency + geodesic)                                 #
    # ------------------------------------------------------------------ #
    def _build_pooled_graph(
        self,
        adjacency: torch.Tensor,       # [B, J, J]
        P: torch.Tensor,               # [B, J, C]
        joint_mask: torch.Tensor,      # [B, J]
        coarse_mask: torch.Tensor,     # [B, C]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Hard-assign argmax(P) → coarse adjacency → Floyd geodesic.

        Returns:
            hard_assignment:   [B, J] long; 0 in padded slots.
            pooled_adjacency:  [B, C, C] binary
            pooled_geodesic:   [B, C, C] Floyd hop count (or +Inf)
        """
        hard = P.argmax(dim=-1)  # [B, J]
        # Mask padded joints' assignment to 0 (will be ignored by mask anyway)
        hard = hard.masked_fill(~joint_mask, 0)
        pooled_adj = build_coarse_adjacency_from_hard_assign(
            adjacency, hard, joint_mask, coarse_mask
        )
        pooled_geo = floyd_shortest_path(pooled_adj, coarse_mask)
        return hard, pooled_adj, pooled_geo

    # ------------------------------------------------------------------ #
    # Aux losses                                                          #
    # ------------------------------------------------------------------ #
    def _compute_aux_losses(
        self,
        P: torch.Tensor,                # [B, J, C]
        adjacency: torch.Tensor,        # [B, J, J]
        geodesic_dist: torch.Tensor,    # [B, J, J]
        joint_mask: torch.Tensor,       # [B, J]
        coarse_mask: torch.Tensor,      # [B, C]
        anchor_indices: torch.Tensor,   # [B, C] long; -1 for padded
    ) -> dict[str, torch.Tensor]:
        """MinCut + locality + entropy regularizers, all masked."""
        B, J, C = P.shape
        device = P.device
        eps = 1e-8

        # MinCut cut term: - tr(S^T A S) / tr(S^T D S)
        # Mask adjacency by joint_mask to prevent polluted padded rows/cols
        # from corrupting cut term (codex M1.2 round 12 R12 #20).
        valid_pair_fine = joint_mask.unsqueeze(-1) & joint_mask.unsqueeze(1)  # [B, J, J]
        adj_masked = adjacency * valid_pair_fine.to(adjacency.dtype)
        sm_a_s = torch.einsum("bjc,bjk,bkd->bcd", P, adj_masked, P)  # [B, C, C]
        cut_num = sm_a_s.diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # [B]
        degree = adj_masked.sum(dim=-1)  # [B, J] — already masked
        sm_d_s = torch.einsum("bjc,bj,bjk->bck", P, degree, P)  # [B, C, C]
        cut_den = sm_d_s.diagonal(dim1=-2, dim2=-1).sum(dim=-1).clamp(min=eps)
        loss_cut = -(cut_num / cut_den).mean()

        # MinCut ortho term: per-sample || SᵀS / ||SᵀS||_F − I_C_valid / sqrt(C_valid) ||_F^2
        # Codex M1.2 pool_dynamic round 1 fix: use per-sample C_valid (not C_max),
        # so a perfectly-orthogonal C_valid=3 cluster assignment is NOT penalized
        # by the "missing" C_max-C_valid padded slots.
        P_masked = P * joint_mask.unsqueeze(-1).to(P.dtype)
        # Restrict to valid coarse cols too:
        P_masked = P_masked * coarse_mask.unsqueeze(1).to(P_masked.dtype)
        sts = torch.bmm(P_masked.transpose(1, 2), P_masked)  # [B, C, C]
        valid_pair = coarse_mask.unsqueeze(-1) & coarse_mask.unsqueeze(1)  # [B, C, C]
        # Frobenius norm over valid-pair entries (per sample)
        sts_valid_sq_sum = (sts.pow(2) * valid_pair.to(sts.dtype)).sum(dim=(-2, -1))
        sts_norm = sts_valid_sq_sum.sqrt().clamp(min=eps)  # [B]
        sts_normalized = sts / sts_norm.view(B, 1, 1)
        sts_normalized = sts_normalized * valid_pair.to(sts_normalized.dtype)
        # Per-sample target identity: I scaled by 1/sqrt(C_valid_b) on valid-pair only.
        C_valid = coarse_mask.sum(dim=-1).float().clamp(min=1)  # [B]
        I_C = torch.eye(C, device=device).unsqueeze(0).expand(B, -1, -1).to(sts.dtype)
        target = I_C / C_valid.view(B, 1, 1).sqrt().to(I_C.dtype)
        target = target * valid_pair.to(target.dtype)
        diff_sq = (sts_normalized - target).pow(2) * valid_pair.to(P.dtype)
        loss_ortho = diff_sq.sum(dim=(-2, -1)).mean()

        loss_mincut = self.mincut_lambda * (loss_cut + loss_ortho)

        # Locality: Σ_jc P[b,j,c] * geodesic[b, j, anchor[c]]
        safe_anchor = anchor_indices.clamp(min=0)  # [B, C]
        gather_idx = safe_anchor.unsqueeze(1).expand(-1, J, -1)  # [B, J, C]
        geo_jc = torch.gather(geodesic_dist, dim=2, index=gather_idx)  # [B, J, C]
        geo_jc_safe = geo_jc.clone()
        geo_jc_safe[torch.isposinf(geo_jc_safe)] = float(2 * self.local_radius + 1)
        # mask padded joints + padded anchors before sum
        mask_jc = joint_mask.unsqueeze(-1) & coarse_mask.unsqueeze(1)  # [B, J, C]
        loss_locality = (
            P * geo_jc_safe * mask_jc.to(P.dtype)
        ).sum(dim=(1, 2)) / (joint_mask.float().sum(dim=1).clamp(min=eps))  # [B]
        loss_locality = loss_locality.mean()

        # Entropy: mean per-row entropy of P over valid joints (lower = sharper).
        # H(P_j) = - Σ_c P[j,c] log P[j,c]. Sharp assignments → near-0; uniform → log C.
        # We report it as a metric and (optionally) regularize; caller decides weight.
        P_safe = P.clamp(min=eps)
        ent = -(P * P_safe.log()).sum(dim=-1)  # [B, J]
        ent = (ent * joint_mask.to(P.dtype)).sum(dim=-1) / joint_mask.float().sum(dim=-1).clamp(min=eps)
        loss_entropy = ent.mean()

        return {
            "mincut_cut": loss_cut.detach(),
            "mincut_ortho": loss_ortho.detach(),
            "mincut": loss_mincut,
            "locality": loss_locality,
            "entropy": loss_entropy,
        }

    # ------------------------------------------------------------------ #
    # Forward                                                             #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        joint_features: torch.Tensor,        # [B, T, J, D]
        skeleton_embeddings: torch.Tensor,   # [B, J, D]
        adjacency: torch.Tensor,             # [B, J, J]
        geodesic_dist: torch.Tensor,         # [B, J, J]
        joint_mask: torch.Tensor,            # [B, J]
        frame_mask: torch.Tensor,            # [B, T]
        parent_indices: Optional[list[list[int]]] = None,
        anchor_indices: Optional[torch.Tensor] = None,   # [B, C_max] long
        coarse_mask: Optional[torch.Tensor] = None,      # [B, C_max] bool
    ) -> dict:
        """Multi-level pool contract (codex M1.2 pool_dynamic round 1 fix #3):
          - Level-1 use case (called from M1.3 VAE encoder on fine joints):
              pass `parent_indices`; this method computes anchors via
              `find_anchors_rulebased`.
          - Level-2 use case (called from M1.3 VAE on coarse level-1 features):
              pass pre-computed `anchor_indices` + `coarse_mask`; this method
              skips anchor selection.
        Exactly one of the two anchor-source pairs must be provided.
        """
        # --- Input validation (fail loud) ---
        if joint_features.dim() != 4 or joint_features.shape[-1] != self.d_model:
            raise ValueError(
                f"joint_features must be [B, T, J, {self.d_model}], "
                f"got {tuple(joint_features.shape)}"
            )
        B, T, J, D = joint_features.shape
        if B <= 0 or T <= 0 or J <= 0:
            raise ValueError(f"empty input: B={B} T={T} J={J}")
        if skeleton_embeddings.shape != (B, J, D):
            raise ValueError(
                f"skeleton_embeddings must be [{B}, {J}, {D}], "
                f"got {tuple(skeleton_embeddings.shape)}"
            )
        if adjacency.shape != (B, J, J) or geodesic_dist.shape != (B, J, J):
            raise ValueError(
                f"adjacency/geodesic_dist must be [B={B}, J={J}, J={J}], "
                f"got {tuple(adjacency.shape)} and {tuple(geodesic_dist.shape)}"
            )
        if joint_mask.shape != (B, J) or joint_mask.dtype != torch.bool:
            raise ValueError(
                f"joint_mask must be [B={B}, J={J}] bool, "
                f"got {tuple(joint_mask.shape)} dtype {joint_mask.dtype}"
            )
        if frame_mask.shape != (B, T) or frame_mask.dtype != torch.bool:
            raise ValueError(
                f"frame_mask must be [B={B}, T={T}] bool, "
                f"got {tuple(frame_mask.shape)} dtype {frame_mask.dtype}"
            )
        # --- Device consistency (codex M1.2 round 10 R12 #18 + round 11) ---
        ref_device = joint_features.device
        for name, t in (
            ("skeleton_embeddings", skeleton_embeddings),
            ("adjacency", adjacency),
            ("geodesic_dist", geodesic_dist),
            ("joint_mask", joint_mask),
            ("frame_mask", frame_mask),
        ):
            if t.device != ref_device:
                raise ValueError(
                    f"DynamicGraphPool: {name}.device {t.device} != "
                    f"joint_features.device {ref_device}"
                )
        # Module params must also match (else matmul crashes deep).
        module_dev = self.q_proj.weight.device
        if module_dev != ref_device:
            raise ValueError(
                f"DynamicGraphPool: module device {module_dev} != input device "
                f"{ref_device}; call .to(device) on the module before forward"
            )
        # Module dtype must match input float dtype (codex M1.2 round 12).
        module_dtype = self.q_proj.weight.dtype
        if module_dtype != torch.float32:
            raise ValueError(
                f"DynamicGraphPool: module dtype {module_dtype} != float32 "
                f"(expected; module must stay fp32 — fp16 overflows + fp64 "
                f"silently up-casts inputs)"
            )

        # --- Finite-value + dtype + topology guards (codex M1.2 rounds 7/9) ---
        # Float-tensor dtype: must be float32 (consistent with module dtype;
        # float64 would silently cast deep at matmul, fp16 overflow).
        for name, t in (
            ("joint_features", joint_features),
            ("skeleton_embeddings", skeleton_embeddings),
            ("adjacency", adjacency),
            ("geodesic_dist", geodesic_dist),
        ):
            if t.dtype != torch.float32:
                raise ValueError(
                    f"DynamicGraphPool: {name}.dtype must be float32, got {t.dtype}"
                )
        # joint_features and skeleton_embeddings: must be finite. Padded slots
        # are typically zero; non-finite anywhere would propagate via einsum.
        if not torch.isfinite(joint_features).all():
            raise ValueError(
                "DynamicGraphPool: joint_features contains NaN or Inf"
            )
        if not torch.isfinite(skeleton_embeddings).all():
            raise ValueError(
                "DynamicGraphPool: skeleton_embeddings contains NaN or Inf"
            )
        # Adjacency: finite, symmetric, zero diagonal (binary-or-soft in [0,1]
        # is the caller's contract via GraphMotionBatch; we re-check structure).
        if not torch.isfinite(adjacency).all():
            raise ValueError(
                "DynamicGraphPool: adjacency contains NaN or Inf"
            )
        if not torch.allclose(adjacency, adjacency.transpose(-2, -1), atol=1e-6, rtol=0.0):
            raise ValueError(
                "DynamicGraphPool: adjacency is not symmetric "
                "(undirected graph required)"
            )
        if (adjacency.diagonal(dim1=-2, dim2=-1) != 0).any():
            raise ValueError(
                "DynamicGraphPool: adjacency has non-zero diagonal "
                "(self-loops not permitted)"
            )
        # Adjacency value-domain (codex M1.2 round 8 R12 #14): binary {0,1} or
        # soft [0,1]. Negative values would silently corrupt MinCut degree and
        # bias (similar to attention.py contract).
        if (adjacency < 0).any():
            raise ValueError(
                "DynamicGraphPool: adjacency contains negative values "
                "(edge weights must be non-negative)"
            )
        if (adjacency > 1.0).any():
            raise ValueError(
                "DynamicGraphPool: adjacency contains values > 1.0 "
                "(contract is binary {0,1} or soft [0,1])"
            )
        # Geodesic: NaN/-Inf reject (only +Inf legitimate per Floyd unreachable).
        if torch.isnan(geodesic_dist).any():
            raise ValueError(
                "DynamicGraphPool: geodesic_dist contains NaN"
            )
        if (geodesic_dist == float("-inf")).any():
            raise ValueError(
                "DynamicGraphPool: geodesic_dist contains -Inf "
                "(only +Inf is legitimate for unreachable pairs)"
            )
        # Negative finite distances are invalid (Floyd output ≥ 0).
        # Codex M1.2 round 9 R12 #13.
        finite_geo = geodesic_dist[torch.isfinite(geodesic_dist)]
        if (finite_geo < 0).any():
            raise ValueError(
                "DynamicGraphPool: geodesic_dist has negative finite entries "
                "(distances must be ≥ 0)"
            )
        # (Floyd consistency check moved to AFTER all mask validation — must run
        # last so mask-violation errors fire with their own clear message rather
        # than getting masked by Floyd's downstream mismatch.)
        # Geodesic symmetry + zero diagonal (codex M1.2 round 8 K/L):
        # mirror the adjacency checks. Use finite-pattern symmetry to allow
        # +Inf at corresponding positions on both sides.
        gt = geodesic_dist.transpose(-2, -1)
        finite_g = torch.isfinite(geodesic_dist)
        finite_gt = torch.isfinite(gt)
        if not torch.equal(finite_g, finite_gt):
            raise ValueError(
                "DynamicGraphPool: geodesic_dist finite/+Inf pattern is not "
                "symmetric (asymmetric reachability)"
            )
        both_finite = finite_g & finite_gt
        if not torch.allclose(
            geodesic_dist[both_finite], gt[both_finite], atol=1e-6, rtol=0.0
        ):
            raise ValueError(
                "DynamicGraphPool: geodesic_dist is not symmetric on finite entries"
            )
        # Zero diagonal at valid nodes (i→i distance must be 0)
        # NOTE: padded slots may have anything (we don't probe them).
        diag = geodesic_dist.diagonal(dim1=-2, dim2=-1)  # [B, J]
        if ((diag != 0) & joint_mask).any():
            raise ValueError(
                "DynamicGraphPool: geodesic_dist has non-zero diagonal "
                "at valid nodes (i→i distance must be 0)"
            )

        # Per-sample non-empty joint_mask (codex M1.2 round 3 fix #2):
        # All-False joint_mask + parent_indices=[[]] silently produces finite
        # zero output otherwise.
        if not joint_mask.any(dim=-1).all():
            bad = (~joint_mask.any(dim=-1)).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(
                f"DynamicGraphPool: batch element(s) {bad} have all-False "
                f"joint_mask (sample has zero valid joints; pool undefined)"
            )
        # Per-sample joint_mask must be a CONTIGUOUS True-prefix (codex M1.2
        # round 5 R12 #11): a hole like [T,T,T,F,T,T] makes compact
        # parent_indices (length=mask.sum()) alias to wrong raw joint indices.
        # Anchors emitted in compact-index space then index back into raw
        # [B, J, D] tensors → silent topological corruption.
        for b in range(B):
            jm_b = joint_mask[b]
            j_valid_b = int(jm_b.sum().item())
            if not jm_b[:j_valid_b].all():
                raise ValueError(
                    f"DynamicGraphPool: sample {b} joint_mask is not a "
                    f"contiguous True-prefix (valid joints must occupy "
                    f"[0, {j_valid_b}); got {jm_b.tolist()})"
                )
            if j_valid_b < J and jm_b[j_valid_b:].any():
                raise ValueError(
                    f"DynamicGraphPool: sample {b} joint_mask has True in the "
                    f"padded region [{j_valid_b}:{J})"
                )

        # --- R12 #19: geodesic must equal Floyd(adjacency, joint_mask) ---
        # Runs LAST among validity checks so all mask + adj + geo structural
        # violations fire with their own messages first. Otherwise a hole in
        # joint_mask or wrong adj would surface as "geo inconsistent with adj"
        # which hides the real bug. Codex M1.2 round 11.
        expected_geo = floyd_shortest_path(adjacency, joint_mask)
        both_valid = joint_mask.unsqueeze(-1) & joint_mask.unsqueeze(1)
        finite_actual = torch.isfinite(geodesic_dist) & both_valid
        finite_expected = torch.isfinite(expected_geo) & both_valid
        if not torch.equal(finite_actual, finite_expected):
            raise ValueError(
                "DynamicGraphPool: geodesic_dist reachability pattern "
                "inconsistent with adjacency (Floyd-recomputed)"
            )
        compare_mask = finite_actual & finite_expected
        if not torch.allclose(
            geodesic_dist[compare_mask], expected_geo[compare_mask],
            atol=1e-6, rtol=0.0,
        ):
            raise ValueError(
                "DynamicGraphPool: geodesic_dist values inconsistent with "
                "shortest-path over adjacency"
            )

        # T must divide temporal_stride evenly (codex M1.2 round 2 fix #3):
        # else frame_mask.view(B, T_lat, stride) crashes deep in _pool_features.
        if T % self.temporal_stride != 0:
            raise ValueError(
                f"DynamicGraphPool: T={T} must be divisible by "
                f"temporal_stride={self.temporal_stride} (pad upstream)"
            )

        # Anchor-source contract (codex M1.2 round 2 fix #1):
        #   Provide EITHER parent_indices alone OR (anchor_indices AND coarse_mask)
        #   together. Partial (only one of anchor_indices / coarse_mask) is a bug.
        n_anchor_args = int(anchor_indices is not None) + int(coarse_mask is not None)
        if n_anchor_args == 1:
            which = "anchor_indices" if anchor_indices is not None else "coarse_mask"
            raise ValueError(
                f"DynamicGraphPool: '{which}' provided alone; the two must "
                f"come as a pair (or both be None and pass parent_indices instead)"
            )
        has_parents = parent_indices is not None
        has_anchors = n_anchor_args == 2
        if has_parents and has_anchors:
            raise ValueError(
                "DynamicGraphPool: pass either parent_indices OR "
                "(anchor_indices + coarse_mask), not both"
            )
        if not has_parents and not has_anchors:
            raise ValueError(
                "DynamicGraphPool: must pass either parent_indices OR "
                "(anchor_indices + coarse_mask) — got neither"
            )
        if has_parents:
            if not isinstance(parent_indices, list) or len(parent_indices) != B:
                raise ValueError(
                    f"parent_indices must be list of length B={B}, "
                    f"got {type(parent_indices).__name__}"
                )
            # Per-sample length must match joint_mask.sum (codex M1.2 fix #2)
            for b in range(B):
                j_valid = int(joint_mask[b].sum().item())
                if len(parent_indices[b]) != j_valid:
                    raise ValueError(
                        f"DynamicGraphPool: parent_indices[{b}] length "
                        f"{len(parent_indices[b])} != joint_mask[{b}].sum() = {j_valid}"
                    )
                # FK ordering invariant (root=0 + parents[j]<j for j>0; codex
                # M1.2 round 14 advisory promoted to required). TopoFKDecoder
                # downstream WILL crash on non-FK-ordered parents.
                try:
                    assert_root_first_parent_order(parent_indices[b])
                except ValueError as e:
                    raise ValueError(
                        f"DynamicGraphPool: parent_indices[{b}] violates FK "
                        f"ordering invariant (root=0 + parents[j]<j) — {e}"
                    ) from e
                # parent_indices must encode the SAME graph as adjacency[b]
                # (codex M1.2 round 13 R12 #21 + round 14 #22). atol=0 because
                # adjacency is binary {0,1} — any non-zero entry counts as edge.
                pi_b = parent_indices[b]
                expected_adj_sub = torch.zeros(j_valid, j_valid, dtype=adjacency.dtype,
                                               device=adjacency.device)
                for j, p in enumerate(pi_b):
                    if p >= 0:
                        expected_adj_sub[j, p] = 1.0
                        expected_adj_sub[p, j] = 1.0
                actual_adj_sub = adjacency[b, :j_valid, :j_valid]
                # Use exact equality (binary contract); the rest of the
                # forward path also assumes 0/1 entries elsewhere.
                if not torch.equal(actual_adj_sub, expected_adj_sub):
                    raise ValueError(
                        f"DynamicGraphPool: sample {b} parent_indices does not "
                        f"match adjacency exactly (binary {{0,1}} contract; any "
                        f"non-zero entry counts as an edge in Floyd)"
                    )

        device = joint_features.device

        # --- 1. Anchor selection ---
        if has_parents:
            anchor_indices, coarse_mask = self._select_anchors(parent_indices)
            anchor_indices = anchor_indices.to(device)
            coarse_mask = coarse_mask.to(device)
        else:
            # Validate provided anchor_indices + coarse_mask
            if anchor_indices.shape != (B, self.max_coarse):
                raise ValueError(
                    f"anchor_indices shape must be (B={B}, max_coarse={self.max_coarse}), "
                    f"got {tuple(anchor_indices.shape)}"
                )
            if anchor_indices.dtype != torch.long:
                raise ValueError(
                    f"anchor_indices must be int64, got {anchor_indices.dtype}"
                )
            if coarse_mask.shape != (B, self.max_coarse) or coarse_mask.dtype != torch.bool:
                raise ValueError(
                    f"coarse_mask must be ({B}, {self.max_coarse}) bool, "
                    f"got {tuple(coarse_mask.shape)} dtype {coarse_mask.dtype}"
                )
            # Padded slots (coarse_mask=False) must be exactly -1 sentinel
            # (codex M1.2 round 15 R12 #23). Match rule-based path which
            # writes -1 in padded slots.
            padded_slots = ~coarse_mask
            if padded_slots.any():
                padded_vals = anchor_indices[padded_slots]
                if (padded_vals != -1).any():
                    bad_pos = (padded_slots & (anchor_indices != -1)).nonzero(as_tuple=False)
                    raise ValueError(
                        f"DynamicGraphPool: anchor_indices[coarse_mask=False] "
                        f"must be -1 sentinel; first offending (b, slot) = "
                        f"({bad_pos[0, 0].item()}, {bad_pos[0, 1].item()})"
                    )
            # Range check on ALL anchor_indices entries (codex M1.2 round 2 fix #2):
            # even masked-False slots will pass through `gather` deep in compute
            # — they must be safe values in [-1, J), where -1 is the sentinel
            # for "padded slot" (downstream clamps to 0 + masks). Values like
            # 9999 in a masked slot would silently corrupt the gather output.
            if (anchor_indices < -1).any() or (anchor_indices >= J).any():
                bad = (anchor_indices < -1) | (anchor_indices >= J)
                pos = bad.nonzero(as_tuple=False)
                raise ValueError(
                    f"DynamicGraphPool: anchor_indices out of range [-1, {J}); "
                    f"first offending position (batch, slot) = "
                    f"({pos[0, 0].item()}, {pos[0, 1].item()}), value = "
                    f"{anchor_indices[pos[0, 0], pos[0, 1]].item()}"
                )
            # Active slots: anchor must be a valid joint, unique, strictly
            # ascending (matches rule-based path which emits sorted unique
            # anchors; codex M1.2 round 3 fix #1). Plus coarse_mask must be
            # a CONTIGUOUS prefix of True — codex M1.2 round 4 R12 fix #9:
            # rule-based `_select_anchors` writes active slots from index 0
            # contiguously, so override path must match that convention.
            for b in range(B):
                valid_anchor_mask = coarse_mask[b]
                if not valid_anchor_mask.any():
                    raise ValueError(
                        f"DynamicGraphPool: sample {b} has zero active coarse "
                        f"slots (coarse_mask all False); pool undefined"
                    )
                # Prefix-active check: True entries must be in [0, c_valid_b).
                c_valid_b = int(valid_anchor_mask.sum().item())
                if not valid_anchor_mask[:c_valid_b].all():
                    raise ValueError(
                        f"DynamicGraphPool: sample {b} coarse_mask is not a "
                        f"contiguous True-prefix (active slots must occupy "
                        f"[0, {c_valid_b}); got mask {valid_anchor_mask.tolist()})"
                    )
                if c_valid_b < self.max_coarse and valid_anchor_mask[c_valid_b:].any():
                    raise ValueError(
                        f"DynamicGraphPool: sample {b} coarse_mask has True in "
                        f"the padded region [{c_valid_b}:{self.max_coarse})"
                    )
                valid_anchors = anchor_indices[b][valid_anchor_mask]
                if (valid_anchors < 0).any():
                    raise ValueError(
                        f"DynamicGraphPool: sample {b} has -1 in an active "
                        f"coarse slot (coarse_mask=True ↔ anchor must be ≥ 0)"
                    )
                if not joint_mask[b][valid_anchors].all():
                    raise ValueError(
                        f"DynamicGraphPool: sample {b} anchor_indices point "
                        f"to padded joints (anchors must be valid joints)"
                    )
                # Strictly ascending → no duplicates AND sorted.
                if len(valid_anchors) > 1:
                    diffs = valid_anchors[1:] - valid_anchors[:-1]
                    if (diffs <= 0).any():
                        raise ValueError(
                            f"DynamicGraphPool: sample {b} anchor_indices "
                            f"active slots must be strictly ascending (sorted "
                            f"+ unique to match rule-based path contract); "
                            f"got {valid_anchors.tolist()}"
                        )
                # Root invariant: rule-based path (find_anchors_rulebased)
                # always emits joint 0 (the root) as the first anchor; override
                # path must match. Codex M1.2 round 6 R12 #12.
                if valid_anchors[0].item() != 0:
                    raise ValueError(
                        f"DynamicGraphPool: sample {b} anchor_indices missing "
                        f"root (joint 0) — first active anchor must be 0 to "
                        f"match rule-based path contract; got first anchor = "
                        f"{valid_anchors[0].item()}"
                    )
            # Device normalization (codex M1.2 round 2 fix advisory)
            anchor_indices = anchor_indices.to(device)
            coarse_mask = coarse_mask.to(device)

        # --- 2. Soft assignment ---
        P = self._compute_assignment(
            skeleton_embeddings, geodesic_dist, joint_mask,
            anchor_indices, coarse_mask,
        )

        # --- 3. Feature pool + temporal downsample ---
        pooled_features, frame_mask_down = self._pool_features(
            joint_features, P, frame_mask,
        )

        # --- 4. Pooled graph ---
        hard_assignment, pooled_adj, pooled_geo = self._build_pooled_graph(
            adjacency, P, joint_mask, coarse_mask,
        )

        # --- 5. Anchor skeleton embeddings (RAW gather; NOT Wk-projected.
        # Wk is private to assignment scoring — exposing it would corrupt
        # downstream re-projection; M1.3 VAE wants raw skeleton conditioning.
        # Codex M1.2 round 1 confirmed RAW is correct.) ---
        safe_anchor = anchor_indices.clamp(min=0)
        idx = safe_anchor.unsqueeze(-1).expand(-1, -1, D)
        pooled_skeleton_embeddings = torch.gather(skeleton_embeddings, dim=1, index=idx)
        pooled_skeleton_embeddings = pooled_skeleton_embeddings * coarse_mask.unsqueeze(-1).to(pooled_skeleton_embeddings.dtype)

        # --- 6. Aux losses ---
        aux_losses = self._compute_aux_losses(
            P, adjacency, geodesic_dist, joint_mask, coarse_mask, anchor_indices,
        )

        return {
            "pooled_features": pooled_features,
            "assignment": P,
            "hard_assignment": hard_assignment,
            "pooled_adjacency": pooled_adj,
            "pooled_geodesic": pooled_geo,
            "pooled_mask": coarse_mask,
            "pooled_skeleton_embeddings": pooled_skeleton_embeddings,
            "frame_mask_down": frame_mask_down,
            "anchor_indices": anchor_indices,
            "aux_losses": aux_losses,
        }
