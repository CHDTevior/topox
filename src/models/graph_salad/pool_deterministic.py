"""DeterministicGraphPool — rule-based skeleton pool (no learnable params).

3-way ablation companion to DynamicGraphPool: same API contract (per
PLAN_GAP_REPORT.md §5; user 2026-05-20 Gate-3 decision to compare learned
vs deterministic vs no-pool to prove the dynamic pool's learning value).

Differences from DynamicGraphPool:
  - No `q_proj` / `k_proj` projections. `state_dict()` is empty → preserves
    M1.0/M1.1 ckpt envelope (no new learnable params introduced by this path).
  - Hard rule-based assignment: each fine joint j is one-hot assigned to
    `argmin_c geodesic[j, anchor[c]]` over active coarse slots. Ties broken
    by lowest anchor index (PyTorch argmin default).
  - No MinCut aux loss. Locality is exactly the sum-of-min-geodesics (since
    each joint goes to nearest anchor). Entropy is exactly 0 (one-hot).

Identical to DynamicGraphPool:
  - Anchor selection via `graph_utils.find_anchors_rulebased`.
  - Mass-normalized skeletal pool + temporal AvgPool1d.
  - Pooled adjacency via `graph_utils.build_coarse_adjacency_from_hard_assign`.
  - Pooled geodesic via `graph_utils.floyd_shortest_path`.
  - Output dict schema (so M1.3 VAE can swap via `pool_type` arg).
  - All 23 R12 fail-loud validations from pool_dynamic.py rounds 1-16.

Validation duplication note: this module copies pool_dynamic.py's validation
block ~verbatim. Extracting to a shared `_pool_common.py` is the natural
refactor but defers until codex round-1 has reviewed both modules side-by-side
(safer to keep changes isolated to in-flight module).
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


_LARGE_FINITE_GEO_SENTINEL = 1.0e6  # used in argmin where +Inf masking unsafe
_MASS_FLOOR = 1e-8


class DeterministicGraphPool(nn.Module):
    """Rule-based graph pool (zero learnable params).

    Args:
        d_model: kept for API parity with DynamicGraphPool; not used internally
            (no projections). Must be > 0 + strict int.
        max_coarse: max coarse nodes per sample.
        local_radius: candidate anchor mask radius. Joint without any anchor in
            radius → raise (configuration error).
        max_chain_chunk_len: passed to find_anchors_rulebased.
        temporal_stride: T → T // stride downsample factor.

    Note no `mincut_lambda` / `temperature` / `locality_alpha` — assignment is
    deterministic so these would be inert.
    """

    def __init__(
        self,
        d_model: int,
        max_coarse: int,
        *,
        local_radius: int = 3,
        max_chain_chunk_len: int = 5,
        temporal_stride: int = 2,
        tau: float | None = None,
    ) -> None:
        """tau: None (default) → hard argmin one-hot assignment.
                > 0           → soft softmax(-geodesic/tau) over candidates
                                ("soft deterministic" ablation per codex M1.5 finding
                                 audit; isolates "hardness" vs "learnedness" as
                                 differentiator).
        """
        super().__init__()
        # Strict-int hparam checks (codex pool_dynamic R12 contract)
        for name, val in (
            ("d_model", d_model),
            ("max_coarse", max_coarse),
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
        if temporal_stride < 1:
            raise ValueError(f"temporal_stride must be ≥ 1, got {temporal_stride}")
        if tau is not None:
            if not isinstance(tau, (int, float)) or isinstance(tau, bool):
                raise TypeError(f"tau must be float or None, got {type(tau).__name__}")
            if not (tau > 0):
                raise ValueError(f"tau must be > 0 when set, got {tau}")
            if not math.isfinite(tau):
                raise ValueError(f"tau must be finite when set, got {tau}")

        self.d_model = d_model
        self.max_coarse = max_coarse
        self.local_radius = local_radius
        self.max_chain_chunk_len = max_chain_chunk_len
        self.temporal_stride = temporal_stride
        self.tau = float(tau) if tau is not None else None
        # No nn.Parameters / submodules. state_dict() will be empty.

    # ------------------------------------------------------------------ #
    # Anchor selection — identical to DynamicGraphPool path              #
    # ------------------------------------------------------------------ #
    def _select_anchors(
        self,
        parent_indices: list[list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
                    f"DeterministicGraphPool: sample {b} produced {len(anchors)} "
                    f"anchors > max_coarse={self.max_coarse}"
                )
            for c, a in enumerate(anchors):
                anchor_indices[b, c] = a
                coarse_mask[b, c] = True
        return anchor_indices, coarse_mask

    # ------------------------------------------------------------------ #
    # Hard rule-based assignment (argmin geodesic over active anchors)   #
    # ------------------------------------------------------------------ #
    def _compute_assignment(
        self,
        geodesic_dist: torch.Tensor,        # [B, J, J]
        joint_mask: torch.Tensor,           # [B, J]
        anchor_indices: torch.Tensor,       # [B, C]
        coarse_mask: torch.Tensor,          # [B, C]
    ) -> torch.Tensor:
        """One-hot assignment P[b, j, c] = 1 iff c == argmin_c' geodesic[j, anchor[c']].

        Argmin is over active anchors (coarse_mask[b, c]=True) within
        local_radius of joint j. Ties broken by lowest anchor index.

        Joint without any anchor in local_radius → raise (configuration error;
        same contract as DynamicGraphPool).
        """
        B, J, _ = geodesic_dist.shape
        C = self.max_coarse
        device = geodesic_dist.device

        # gather geodesic[b, j, anchor[c]] → [B, J, C]
        safe_anchor = anchor_indices.clamp(min=0)
        gather_idx = safe_anchor.unsqueeze(1).expand(-1, J, -1)  # [B, J, C]
        geo_jc = torch.gather(geodesic_dist, dim=2, index=gather_idx)  # [B, J, C]

        # Candidate mask: in_radius AND coarse_mask AND joint_mask
        in_radius = geo_jc <= float(self.local_radius)
        candidate = (
            in_radius
            & coarse_mask.unsqueeze(1)
            & joint_mask.unsqueeze(-1)
        )
        no_candidate = ~candidate.any(dim=-1)
        no_candidate_for_valid = no_candidate & joint_mask
        if no_candidate_for_valid.any():
            bad = no_candidate_for_valid.nonzero(as_tuple=False)
            b0, j0 = bad[0, 0].item(), bad[0, 1].item()
            raise ValueError(
                f"DeterministicGraphPool: sample {b0} joint {j0} has no candidate "
                f"anchor within local_radius={self.local_radius}; increase radius "
                f"or revise anchors"
            )

        # Score: substitute non-candidate entries with large finite sentinel
        score = geo_jc.clone()
        score[torch.isposinf(score)] = _LARGE_FINITE_GEO_SENTINEL
        score = torch.where(candidate, score,
                            torch.full_like(score, _LARGE_FINITE_GEO_SENTINEL * 2))

        if self.tau is None:
            # Hard argmin one-hot (default)
            argmin_c = score.argmin(dim=-1)  # [B, J] long
            P = torch.zeros(B, J, C, device=device, dtype=torch.float32)
            P.scatter_(2, argmin_c.unsqueeze(-1), 1.0)
        else:
            # Soft assignment: softmax(-score/tau) over candidates ONLY (codex M1.5 R1 P1.1 fix)
            # Bug pre-fix: dividing sentinel by tau leaked probability to non-candidates
            # at large tau (e.g., tau=1e6 → 0.119 mass on out-of-radius). Fix: mask
            # non-candidate logits to -inf-equivalent AFTER tau scaling, matching
            # DynamicGraphPool convention.
            logits = -geo_jc / self.tau  # [B, J, C], use raw geo (not sentinel-substituted)
            logits = logits.masked_fill(~candidate, -1.0e9)  # exclude non-candidates
            P = F.softmax(logits, dim=-1)  # rows sum to 1 over candidates
        # Zero rows for padded joints; zero columns for padded anchors
        P = P * joint_mask.unsqueeze(-1).to(P.dtype)
        P = P * coarse_mask.unsqueeze(1).to(P.dtype)
        return P

    # ------------------------------------------------------------------ #
    # Feature pool — identical to DynamicGraphPool                       #
    # ------------------------------------------------------------------ #
    def _pool_features(
        self,
        joint_features: torch.Tensor,
        P: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_pool = torch.einsum("bjc,btjd->btcd", P, joint_features)
        mass = P.sum(dim=1).clamp(min=_MASS_FLOOR)
        h_pool = h_pool / mass.unsqueeze(1).unsqueeze(-1)
        B, T, C, D = h_pool.shape
        s = self.temporal_stride
        T_lat = T // s
        h_pool_flat = h_pool.permute(0, 2, 3, 1).reshape(B * C * D, 1, T)
        h_down = F.avg_pool1d(h_pool_flat, kernel_size=s, stride=s)
        h_down = h_down.reshape(B, C, D, T_lat).permute(0, 3, 1, 2).contiguous()
        fm = frame_mask.view(B, T_lat, s).all(dim=-1)
        return h_down, fm

    # ------------------------------------------------------------------ #
    # Pooled graph — identical to DynamicGraphPool                       #
    # ------------------------------------------------------------------ #
    def _build_pooled_graph(
        self,
        adjacency: torch.Tensor,
        P: torch.Tensor,
        joint_mask: torch.Tensor,
        coarse_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hard = P.argmax(dim=-1)
        hard = hard.masked_fill(~joint_mask, 0)
        pooled_adj = build_coarse_adjacency_from_hard_assign(
            adjacency, hard, joint_mask, coarse_mask
        )
        pooled_geo = floyd_shortest_path(pooled_adj, coarse_mask)
        return hard, pooled_adj, pooled_geo

    # ------------------------------------------------------------------ #
    # Aux losses — only locality + entropy (entropy ≡ 0 for one-hot)     #
    # ------------------------------------------------------------------ #
    def _compute_aux_losses(
        self,
        P: torch.Tensor,                # [B, J, C]
        geodesic_dist: torch.Tensor,    # [B, J, J]
        joint_mask: torch.Tensor,       # [B, J]
        coarse_mask: torch.Tensor,      # [B, C]
        anchor_indices: torch.Tensor,   # [B, C]
    ) -> dict[str, torch.Tensor]:
        B, J, C = P.shape
        device = P.device
        eps = 1e-8

        # Locality (with one-hot P, this is exactly sum of min-geodesic-to-anchor
        # over valid joints).
        safe_anchor = anchor_indices.clamp(min=0)
        gather_idx = safe_anchor.unsqueeze(1).expand(-1, J, -1)
        geo_jc = torch.gather(geodesic_dist, dim=2, index=gather_idx)
        geo_jc_safe = geo_jc.clone()
        geo_jc_safe[torch.isposinf(geo_jc_safe)] = float(2 * self.local_radius + 1)
        mask_jc = joint_mask.unsqueeze(-1) & coarse_mask.unsqueeze(1)
        loss_locality = (
            P * geo_jc_safe * mask_jc.to(P.dtype)
        ).sum(dim=(1, 2)) / joint_mask.float().sum(dim=1).clamp(min=eps)
        loss_locality = loss_locality.mean()

        # Entropy — codex M1.5 soft-det R1 P2 fix: compute from P for all modes
        # (hard one-hot → exactly 0; soft → positive). Was hard-coded 0 (misleading
        # for soft path).
        P_log = (P + eps).log()
        entropy_per_row = -(P * P_log).sum(dim=-1)  # [B, J]
        loss_entropy = (entropy_per_row * joint_mask.to(P.dtype)).sum() \
            / joint_mask.float().sum().clamp(min=eps)
        zero_const = torch.zeros((), device=device, dtype=P.dtype)

        return {
            # All MinCut keys present for schema parity with DynamicGraphPool,
            # but zero since deterministic has no learnable assignment.
            "mincut_cut": zero_const,
            "mincut_ortho": zero_const,
            "mincut": zero_const,
            "locality": loss_locality,
            "entropy": loss_entropy,
        }

    # ------------------------------------------------------------------ #
    # Motion-independent geometry (used by VAE.encode_skeleton_only)     #
    # ------------------------------------------------------------------ #
    def compute_assignment_and_graph(
        self,
        skeleton_embeddings: torch.Tensor,
        adjacency: torch.Tensor,
        geodesic_dist: torch.Tensor,
        joint_mask: torch.Tensor,
        parent_indices: Optional[list[list[int]]] = None,
        anchor_indices: Optional[torch.Tensor] = None,
        coarse_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Compute motion-independent pool outputs (parity with DynamicGraphPool).

        Does NOT consume motion frames — used by GraphMotionVAE.encode_skeleton_only
        for the Phase-2 graph-aware denoiser at sampling time. forward() calls
        this internally for the motion-independent half of its work and then
        layers _pool_features on top.

        Returns dict with: assignment, hard_assignment, pooled_adjacency,
        pooled_geodesic, pooled_mask (= coarse_mask), pooled_skeleton_embeddings,
        anchor_indices, aux_losses.
        """
        # --- Input validation (skeleton-side; mirrors forward's contract) ---
        if skeleton_embeddings.dim() != 3 or skeleton_embeddings.shape[-1] != self.d_model:
            raise ValueError(
                f"skeleton_embeddings must be [B, J, {self.d_model}], got {tuple(skeleton_embeddings.shape)}"
            )
        B, J, D = skeleton_embeddings.shape
        if B <= 0 or J <= 0:
            raise ValueError(f"empty input: B={B} J={J}")
        if adjacency.shape != (B, J, J) or geodesic_dist.shape != (B, J, J):
            raise ValueError(
                f"adjacency/geodesic shape mismatch: {tuple(adjacency.shape)} / {tuple(geodesic_dist.shape)}"
            )
        if joint_mask.shape != (B, J) or joint_mask.dtype != torch.bool:
            raise ValueError(f"joint_mask must be [B, J] bool, got {tuple(joint_mask.shape)} {joint_mask.dtype}")

        # Device consistency
        ref_device = skeleton_embeddings.device
        for name, t in (
            ("adjacency", adjacency),
            ("geodesic_dist", geodesic_dist),
            ("joint_mask", joint_mask),
        ):
            if t.device != ref_device:
                raise ValueError(f"{name}.device {t.device} != skeleton_embeddings.device {ref_device}")

        # Dtypes
        for name, t in (
            ("skeleton_embeddings", skeleton_embeddings),
            ("adjacency", adjacency),
            ("geodesic_dist", geodesic_dist),
        ):
            if t.dtype != torch.float32:
                raise ValueError(f"{name}.dtype must be float32, got {t.dtype}")

        # Finite
        if not torch.isfinite(skeleton_embeddings).all():
            raise ValueError("skeleton_embeddings contains NaN or Inf")
        if not torch.isfinite(adjacency).all():
            raise ValueError("adjacency contains NaN or Inf")
        if (adjacency < 0).any():
            raise ValueError("adjacency contains negative values")
        if (adjacency > 1.0).any():
            raise ValueError("adjacency contains values > 1.0 (binary {0,1} contract)")
        if not torch.allclose(adjacency, adjacency.transpose(-2, -1), atol=1e-6, rtol=0.0):
            raise ValueError("adjacency is not symmetric")
        if (adjacency.diagonal(dim1=-2, dim2=-1) != 0).any():
            raise ValueError("adjacency has non-zero diagonal")

        # Geodesic structure
        if torch.isnan(geodesic_dist).any():
            raise ValueError("geodesic_dist contains NaN")
        if (geodesic_dist == float("-inf")).any():
            raise ValueError("geodesic_dist contains -Inf")
        finite_geo = geodesic_dist[torch.isfinite(geodesic_dist)]
        if (finite_geo < 0).any():
            raise ValueError("geodesic_dist has negative finite entries")
        both_valid_jj = joint_mask.unsqueeze(-1) & joint_mask.unsqueeze(1)
        valid_finite_geo = geodesic_dist[torch.isfinite(geodesic_dist) & both_valid_jj]
        if (valid_finite_geo > (J - 1)).any():
            raise ValueError(f"geodesic_dist has finite entries > {J - 1} on valid pairs")
        gt = geodesic_dist.transpose(-2, -1)
        finite_g = torch.isfinite(geodesic_dist)
        finite_gt = torch.isfinite(gt)
        if not torch.equal(finite_g, finite_gt):
            raise ValueError("geodesic_dist finite/+Inf pattern is not symmetric")
        both_finite = finite_g & finite_gt
        if not torch.allclose(geodesic_dist[both_finite], gt[both_finite], atol=1e-6, rtol=0.0):
            raise ValueError("geodesic_dist is not symmetric on finite entries")
        diag = geodesic_dist.diagonal(dim1=-2, dim2=-1)
        if ((diag != 0) & joint_mask).any():
            raise ValueError("geodesic_dist has non-zero diagonal at valid nodes")

        # Mask contiguity
        if not joint_mask.any(dim=-1).all():
            raise ValueError("joint_mask has all-False rows")
        for b in range(B):
            jm_b = joint_mask[b]
            j_valid_b = int(jm_b.sum().item())
            if not jm_b[:j_valid_b].all():
                raise ValueError(f"joint_mask[{b}] not contiguous prefix-True")
            if j_valid_b < J and jm_b[j_valid_b:].any():
                raise ValueError(f"joint_mask[{b}] True in padded region")

        # Anchor source XOR
        n_anchor_args = int(anchor_indices is not None) + int(coarse_mask is not None)
        if n_anchor_args == 1:
            which = "anchor_indices" if anchor_indices is not None else "coarse_mask"
            raise ValueError(f"DeterministicGraphPool: '{which}' provided alone")
        has_parents = parent_indices is not None
        has_anchors = n_anchor_args == 2
        if has_parents and has_anchors:
            raise ValueError("pass either parent_indices OR (anchor_indices + coarse_mask), not both")
        if not has_parents and not has_anchors:
            raise ValueError("must pass either parent_indices OR (anchor_indices + coarse_mask)")

        if has_parents:
            if not isinstance(parent_indices, list) or len(parent_indices) != B:
                raise ValueError(f"parent_indices must be list of length {B}")
            for b in range(B):
                j_valid = int(joint_mask[b].sum().item())
                if len(parent_indices[b]) != j_valid:
                    raise ValueError(
                        f"parent_indices[{b}] length {len(parent_indices[b])} != joint_mask.sum() {j_valid}"
                    )
                try:
                    assert_root_first_parent_order(parent_indices[b])
                except ValueError as e:
                    raise ValueError(f"parent_indices[{b}] violates FK ordering — {e}") from e
                pi_b = parent_indices[b]
                expected_adj_sub = torch.zeros(j_valid, j_valid, dtype=adjacency.dtype, device=adjacency.device)
                for j, p in enumerate(pi_b):
                    if p >= 0:
                        expected_adj_sub[j, p] = 1.0
                        expected_adj_sub[p, j] = 1.0
                actual_adj_sub = adjacency[b, :j_valid, :j_valid]
                if not torch.equal(actual_adj_sub, expected_adj_sub):
                    raise ValueError(f"parent_indices[{b}] does not match adjacency exactly")

        # Floyd cross-consistency
        expected_geo = floyd_shortest_path(adjacency, joint_mask)
        both_valid = joint_mask.unsqueeze(-1) & joint_mask.unsqueeze(1)
        finite_actual_pat = torch.isfinite(geodesic_dist) & both_valid
        finite_expected_pat = torch.isfinite(expected_geo) & both_valid
        if not torch.equal(finite_actual_pat, finite_expected_pat):
            raise ValueError("geodesic_dist reachability inconsistent with adjacency")
        compare_mask = finite_actual_pat & finite_expected_pat
        if not torch.allclose(
            geodesic_dist[compare_mask], expected_geo[compare_mask], atol=1e-6, rtol=0.0
        ):
            raise ValueError("geodesic_dist values inconsistent with shortest-path over adjacency")

        # --- Anchor source resolution ---
        if has_parents:
            anchor_indices, coarse_mask = self._select_anchors(parent_indices)
            anchor_indices = anchor_indices.to(ref_device)
            coarse_mask = coarse_mask.to(ref_device)
        else:
            if anchor_indices.shape != (B, self.max_coarse):
                raise ValueError(f"anchor_indices shape mismatch: {tuple(anchor_indices.shape)}")
            if anchor_indices.dtype != torch.long:
                raise ValueError(f"anchor_indices must be int64, got {anchor_indices.dtype}")
            if coarse_mask.shape != (B, self.max_coarse) or coarse_mask.dtype != torch.bool:
                raise ValueError(f"coarse_mask must be [B, max_coarse] bool")
            padded_slots = ~coarse_mask
            if padded_slots.any():
                padded_vals = anchor_indices[padded_slots]
                if (padded_vals != -1).any():
                    raise ValueError("anchor_indices[coarse_mask=False] must be -1 sentinel")
            if (anchor_indices < -1).any() or (anchor_indices >= J).any():
                raise ValueError(f"anchor_indices out of range [-1, {J})")
            for b in range(B):
                valid_anchor_mask = coarse_mask[b]
                if not valid_anchor_mask.any():
                    raise ValueError(f"sample {b} has zero active coarse slots")
                c_valid_b = int(valid_anchor_mask.sum().item())
                if not valid_anchor_mask[:c_valid_b].all():
                    raise ValueError(f"sample {b} coarse_mask not contiguous prefix-True")
                if c_valid_b < self.max_coarse and valid_anchor_mask[c_valid_b:].any():
                    raise ValueError(f"sample {b} coarse_mask True in padded region")
                valid_anchors = anchor_indices[b][valid_anchor_mask]
                if (valid_anchors < 0).any():
                    raise ValueError(f"sample {b} -1 in active coarse slot")
                if not joint_mask[b][valid_anchors].all():
                    raise ValueError(f"sample {b} anchor points to padded joint")
                if len(valid_anchors) > 1:
                    diffs = valid_anchors[1:] - valid_anchors[:-1]
                    if (diffs <= 0).any():
                        raise ValueError(f"sample {b} anchor not strictly ascending")
                if valid_anchors[0].item() != 0:
                    raise ValueError(f"sample {b} anchor_indices missing root (joint 0)")
            anchor_indices = anchor_indices.to(ref_device)
            coarse_mask = coarse_mask.to(ref_device)

        # --- 1. Hard (or soft) assignment ---
        P = self._compute_assignment(geodesic_dist, joint_mask, anchor_indices, coarse_mask)

        # --- 2. Pooled graph ---
        hard_assignment, pooled_adj, pooled_geo = self._build_pooled_graph(
            adjacency, P, joint_mask, coarse_mask
        )

        # --- 3. Anchor skeleton embeddings (raw gather) ---
        safe_anchor = anchor_indices.clamp(min=0)
        idx = safe_anchor.unsqueeze(-1).expand(-1, -1, D)
        pooled_skeleton_embeddings = torch.gather(skeleton_embeddings, dim=1, index=idx)
        pooled_skeleton_embeddings = (
            pooled_skeleton_embeddings * coarse_mask.unsqueeze(-1).to(pooled_skeleton_embeddings.dtype)
        )

        # --- 4. Aux losses ---
        aux_losses = self._compute_aux_losses(
            P, geodesic_dist, joint_mask, coarse_mask, anchor_indices
        )

        return {
            "assignment": P,
            "hard_assignment": hard_assignment,
            "pooled_adjacency": pooled_adj,
            "pooled_geodesic": pooled_geo,
            "pooled_mask": coarse_mask,
            "pooled_skeleton_embeddings": pooled_skeleton_embeddings,
            "anchor_indices": anchor_indices,
            "aux_losses": aux_losses,
        }

    # ------------------------------------------------------------------ #
    # Forward (motion-dep: validates motion inputs + delegates geometry) #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        joint_features: torch.Tensor,
        skeleton_embeddings: torch.Tensor,
        adjacency: torch.Tensor,
        geodesic_dist: torch.Tensor,
        joint_mask: torch.Tensor,
        frame_mask: torch.Tensor,
        parent_indices: Optional[list[list[int]]] = None,
        anchor_indices: Optional[torch.Tensor] = None,
        coarse_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        # --- Motion-side validation ---
        if joint_features.dim() != 4 or joint_features.shape[-1] != self.d_model:
            raise ValueError(
                f"joint_features must be [B, T, J, {self.d_model}], got {tuple(joint_features.shape)}"
            )
        B, T, J, D = joint_features.shape
        if B <= 0 or T <= 0 or J <= 0:
            raise ValueError(f"empty input: B={B} T={T} J={J}")
        if skeleton_embeddings.shape != (B, J, D):
            raise ValueError(f"skeleton_embeddings shape mismatch: {tuple(skeleton_embeddings.shape)}")
        if frame_mask.shape != (B, T) or frame_mask.dtype != torch.bool:
            raise ValueError(f"frame_mask must be [B, T] bool, got {tuple(frame_mask.shape)} {frame_mask.dtype}")
        ref_device = joint_features.device
        if frame_mask.device != ref_device:
            raise ValueError(f"frame_mask.device {frame_mask.device} != joint_features.device {ref_device}")
        if skeleton_embeddings.device != ref_device:
            raise ValueError(
                f"skeleton_embeddings.device {skeleton_embeddings.device} != joint_features.device {ref_device}"
            )
        if joint_features.dtype != torch.float32:
            raise ValueError(f"joint_features.dtype must be float32, got {joint_features.dtype}")
        if not torch.isfinite(joint_features).all():
            raise ValueError("joint_features contains NaN or Inf")
        if T % self.temporal_stride != 0:
            raise ValueError(f"T={T} must be divisible by temporal_stride={self.temporal_stride}")

        # --- Motion-independent geometry ---
        geom = self.compute_assignment_and_graph(
            skeleton_embeddings=skeleton_embeddings,
            adjacency=adjacency,
            geodesic_dist=geodesic_dist,
            joint_mask=joint_mask,
            parent_indices=parent_indices,
            anchor_indices=anchor_indices,
            coarse_mask=coarse_mask,
        )

        # --- Feature pool (motion-dependent) ---
        pooled_features, frame_mask_down = self._pool_features(joint_features, geom["assignment"], frame_mask)

        return {
            "pooled_features": pooled_features,
            "frame_mask_down": frame_mask_down,
            **geom,
        }
