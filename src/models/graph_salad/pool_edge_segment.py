"""EdgeSegmentPool — v2 chain-segment pool (Pool v2 per design doc
2026-05-23, handoff/20260523_210312_pool_v2_edge_chain_design.md).

This is a NEW pool class, **parallel** to DynamicGraphPool. It does NOT
extend / override / reuse v1's anchor path. The two pools share only the
public contract (forward + compute_assignment_and_graph signatures + return
dict keys), so GraphMotionVAE / MotionDecoder / GraphSaladDenoiser do not
need code changes to switch between them.

Core idea: coarse slot semantics change from "joint anchor cluster" to
"edge-chain segment" (= a group of consecutive fine edges along a
kinematic chain). Inspired by Skeleton-Aware Networks (SAN, Aberman 2020)
but adapted to AnyTop arbitrary topology (no homeomorphic primal-skeleton
assumption).

Differences from v1 (DynamicGraphPool):
  - assignment is HARD 1-of-K (one fine joint → one segment), not soft
    attention. Avoids long-chain averaging of articulation (Dragon wing fix).
  - No learnable params (no Wq/Wk/q_proj/k_proj). Pure rule-based +
    deterministic. Forward is trivial einsum + temporal AvgPool.
  - aux_losses are ZERO scalars (no anchor-distance locality / mincut
    semantics in v2; loss config can keep w_pool_aux=0.5 — multiplying
    zero is a no-op, no need to change train_graph_vae loss weights).
  - pooled_skeleton_embeddings is per-segment MEAN of fine s_j (not anchor
    gather), giving denoiser/decoder a segment-level summary, not a point.
  - anchor_indices returns segment-midpoint joint (semantic shifted, kept
    for debug/visualization; downstream code that ignores it is unaffected).

MVP scope (per user review 2026-05-23):
  - p=2 chain pooling, no fan-aware sibling merge
  - On C overflow: greedy-merge longest chain's adjacent segments until ≤ max_coarse
  - parent_indices required; anchor_indices/coarse_mask override NOT supported
    (raise loud — v2 has no anchor semantics)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_utils import (
    assert_root_first_parent_order,
    floyd_shortest_path,
    validate_parent_tree,
)


_MASS_FLOOR = 1e-6


def _build_segments_rulebased(
    parents: list[int], max_segments: int = 64,
) -> list[list[int]]:
    """Edge-chain segmentation for one skeleton.

    Returns: list of segments, each segment is a list of fine joint indices
    (the joint at the child end of each edge in the segment, plus the root
    joint as a singleton virtual-root segment 0).

    Contract:
      - segments[0] = [0] always (virtual root segment; contains root joint)
      - Each non-root joint j appears in exactly one segment
      - len(segments) <= max_segments (greedy-merge longest chain if overflow)
      - Two segments are topologically adjacent in the pooled graph iff any
        two fine joints (one from each) are adjacent in the original tree
        (i.e., share a parent-child edge crossing the segment boundary)
    """
    J = len(parents)
    if J == 0:
        return []
    validate_parent_tree(parents)
    # Codex P2 (2026-05-23): root must be joint 0 with parents[0]=-1 (root-first
    # FK ordering). The VAE path enforces this via GraphMotionBatch, but direct
    # callers (smoke / unit tests) need explicit assertion.
    assert_root_first_parent_order(parents)
    if J == 1:
        return [[0]]

    children: list[list[int]] = [[] for _ in range(J)]
    for j, p in enumerate(parents):
        if p >= 0:
            children[p].append(j)

    # segments[0] is always the virtual root segment.
    segments: list[list[int]] = [[0]]
    # Track which chain each non-root segment belongs to (for overflow merge).
    # chain_id is a sequential integer per chain (we don't reuse it elsewhere).
    seg_chain_id: list[int] = [-1]  # virtual root has no chain
    chain_counter = 0

    # Walk all chains via iterative DFS from root's children.
    # A "chain" = ordered list of joints (child-side of each edge), starting
    # right after a branch joint (or root) and ending at the next branch joint
    # or leaf (exclusive of the next branch).
    stack: list[int] = list(reversed(children[0]))  # start chains from root's children
    while stack:
        start = stack.pop()
        chain: list[int] = []
        curr = start
        while True:
            chain.append(curr)
            kids = children[curr]
            if len(kids) == 0:
                break  # leaf — chain ends
            if len(kids) >= 2:
                # branch — chain ends here; push children for new chains
                for k in reversed(kids):
                    stack.append(k)
                break
            # exactly 1 child — continue this chain
            curr = kids[0]

        # Group chain into p=2 segments. SAN convention: root-side remainder
        # kept as a single edge when len is odd (i.e., the segment closer to
        # the root absorbs the odd-out).
        L = len(chain)
        i = 0
        chain_seg_indices_start = len(segments)
        if L % 2 == 1:
            segments.append([chain[0]])
            seg_chain_id.append(chain_counter)
            i = 1
        while i < L:
            segments.append([chain[i], chain[i + 1]])
            seg_chain_id.append(chain_counter)
            i += 2
        chain_counter += 1

    # Overflow handling: greedy-merge adjacent segments within the longest
    # chain (= chain with most segments) until total <= max_segments.
    # Merge target: adjacent pair within the same chain, prefer ROOT-SIDE
    # (less-articulated; tip segments more important for visual leaf motion).
    while len(segments) > max_segments:
        # Count segments per chain
        seg_per_chain: dict[int, list[int]] = {}
        for seg_idx, cid in enumerate(seg_chain_id):
            if cid < 0:
                continue
            seg_per_chain.setdefault(cid, []).append(seg_idx)
        # Find chain with most segments (and at least 2 to merge)
        longest = max(
            (cid for cid, lst in seg_per_chain.items() if len(lst) >= 2),
            key=lambda cid: len(seg_per_chain[cid]),
            default=None,
        )
        if longest is None:
            # All chains have only 1 segment — can't reduce. Fail loud.
            raise RuntimeError(
                f"EdgeSegmentPool: cannot fit J={J} skeleton into "
                f"max_segments={max_segments} even after greedy merge "
                f"(all {len(segments)} segments are single-edge chains). "
                f"Increase max_coarse or revise topology."
            )
        # Merge the two root-most segments in this chain
        lst = seg_per_chain[longest]
        keep_idx, drop_idx = lst[0], lst[1]
        # Merge drop into keep
        segments[keep_idx] = segments[keep_idx] + segments[drop_idx]
        # Remove drop from segments + seg_chain_id
        del segments[drop_idx]
        del seg_chain_id[drop_idx]

    # Codex P2 (2026-05-23): coverage assertion — every joint in [0, J)
    # appears in exactly one segment. Defensive — catches segmenting bugs
    # (e.g., missed branch path, double-counted joint after merge).
    covered: set[int] = set()
    for seg in segments:
        for j in seg:
            if j in covered:
                raise RuntimeError(
                    f"_build_segments_rulebased: joint {j} appears in multiple "
                    f"segments (segmentation bug)"
                )
            covered.add(j)
    missing = set(range(J)) - covered
    if missing:
        raise RuntimeError(
            f"_build_segments_rulebased: joints {sorted(missing)[:10]} "
            f"(total {len(missing)}) not covered by any segment "
            f"(segmentation bug; expected all of [0, {J}) covered)"
        )
    return segments


class EdgeSegmentPool(nn.Module):
    """v2 chain-segment pool. See module docstring for design rationale.

    Args:
        d_model: feature dim per joint/slot. Must match VAE's d_model.
        max_coarse: cap on coarse slot count C_max. Default 64 (= v1).
        temporal_stride: temporal AvgPool downsample factor on joint_features.
        temporal_kernel: kept for API parity with v1; AvgPool1d uses
            kernel_size=temporal_stride for clean stride downsample.
        dropout: kept for API parity with v1; unused (no learnable params).

    Forward signature mirrors DynamicGraphPool exactly. anchor_indices /
    coarse_mask override is NOT supported (v2 has no anchor semantics) —
    pass parent_indices instead.
    """

    def __init__(
        self,
        d_model: int,
        max_coarse: int = 64,
        temporal_stride: int = 4,
        temporal_kernel: int = 9,  # noqa: ARG002  (API parity)
        dropout: float = 0.1,      # noqa: ARG002
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")
        if max_coarse <= 0:
            raise ValueError(f"max_coarse must be > 0, got {max_coarse}")
        if not isinstance(temporal_stride, int) or temporal_stride < 1:
            raise ValueError(f"temporal_stride must be int >= 1, got {temporal_stride}")
        self.d_model = d_model
        self.max_coarse = max_coarse
        self.temporal_stride = temporal_stride
        # No learnable params (rule-based + hard assignment + mean embed).
        # Temporal pool kept as buffer-free module for shape parity with v1.
        self.temporal_pool = nn.AvgPool1d(
            kernel_size=temporal_stride, stride=temporal_stride
        )

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
        """Same public contract as DynamicGraphPool.compute_assignment_and_graph.

        v2 semantics: hard 1-of-K assignment, segment-mean pooled_skeleton_embeddings,
        zero aux_losses. anchor_indices/coarse_mask override unsupported.
        """
        # --- Reject anchor override path (v2 has no anchor semantics) ---
        if anchor_indices is not None or coarse_mask is not None:
            raise ValueError(
                "EdgeSegmentPool does not support the anchor_indices/coarse_mask "
                "override path (level-2 nested pool case). v2 only supports the "
                "parent_indices path. If you need nested pooling, use DynamicGraphPool."
            )
        if parent_indices is None:
            raise ValueError(
                "EdgeSegmentPool requires parent_indices (list[list[int]] of length B). "
                "v2 cannot build chain segments without the parent tree."
            )

        # --- Input validation (skeleton-side, mirrors v1's contract) ---
        if skeleton_embeddings.dim() != 3 or skeleton_embeddings.shape[-1] != self.d_model:
            raise ValueError(
                f"skeleton_embeddings must be [B, J, {self.d_model}], "
                f"got {tuple(skeleton_embeddings.shape)}"
            )
        B, J, D = skeleton_embeddings.shape
        if B <= 0 or J <= 0:
            raise ValueError(f"empty input: B={B} J={J}")
        if adjacency.shape != (B, J, J) or geodesic_dist.shape != (B, J, J):
            raise ValueError(
                f"adjacency/geodesic_dist must be [B={B}, J={J}, J={J}], "
                f"got {tuple(adjacency.shape)} and {tuple(geodesic_dist.shape)}"
            )
        if joint_mask.shape != (B, J) or joint_mask.dtype != torch.bool:
            raise ValueError(
                f"joint_mask must be [B={B}, J={J}] bool, got "
                f"{tuple(joint_mask.shape)} dtype {joint_mask.dtype}"
            )
        ref_device = skeleton_embeddings.device
        for name, t in (
            ("adjacency", adjacency),
            ("geodesic_dist", geodesic_dist),
            ("joint_mask", joint_mask),
        ):
            if t.device != ref_device:
                raise ValueError(
                    f"EdgeSegmentPool: {name}.device {t.device} != "
                    f"skeleton_embeddings.device {ref_device}"
                )
        for name, t in (
            ("skeleton_embeddings", skeleton_embeddings),
            ("adjacency", adjacency),
            ("geodesic_dist", geodesic_dist),
        ):
            if t.dtype != torch.float32:
                raise ValueError(
                    f"EdgeSegmentPool: {name}.dtype must be float32, got {t.dtype}"
                )
        if not torch.isfinite(skeleton_embeddings).all():
            raise ValueError("EdgeSegmentPool: skeleton_embeddings contains NaN/Inf")
        if not isinstance(parent_indices, list) or len(parent_indices) != B:
            raise ValueError(
                f"parent_indices must be list of length B={B}, "
                f"got {type(parent_indices).__name__} len="
                f"{len(parent_indices) if hasattr(parent_indices, '__len__') else '?'}"
            )

        C_max = self.max_coarse

        # --- Build per-sample segments + assignment ---
        assignment = torch.zeros(B, J, C_max, device=ref_device, dtype=torch.float32)
        hard_assignment = torch.zeros(B, J, device=ref_device, dtype=torch.long)
        new_coarse_mask = torch.zeros(B, C_max, device=ref_device, dtype=torch.bool)
        anchor_indices_b = torch.full(
            (B, C_max), -1, device=ref_device, dtype=torch.long
        )
        pooled_skel_emb = torch.zeros(
            B, C_max, D, device=ref_device, dtype=torch.float32
        )
        pooled_adj = torch.zeros(
            B, C_max, C_max, device=ref_device, dtype=torch.float32
        )

        for b in range(B):
            parents_b = parent_indices[b]
            J_valid_b = len(parents_b)
            if J_valid_b > J:
                raise ValueError(
                    f"sample {b}: len(parent_indices[{b}])={J_valid_b} > J={J}"
                )
            # Validate joint_mask: parent_indices joints must be valid
            jm_b = joint_mask[b]
            jm_valid_count = int(jm_b.sum().item())
            if jm_valid_count != J_valid_b:
                raise ValueError(
                    f"sample {b}: parent_indices len={J_valid_b} != "
                    f"joint_mask.sum()={jm_valid_count}"
                )

            # Build segments (list of lists of fine joint indices, all within [0, J_valid_b))
            segments_b = _build_segments_rulebased(parents_b, max_segments=C_max)
            C_b = len(segments_b)
            if C_b > C_max:
                # _build_segments_rulebased should have merged; defensive
                raise RuntimeError(
                    f"sample {b}: segments={C_b} > max_coarse={C_max} "
                    f"(internal merge failed)"
                )

            # Fill assignment + hard_assignment + coarse_mask + anchor_indices
            for c, seg_joints in enumerate(segments_b):
                new_coarse_mask[b, c] = True
                for j in seg_joints:
                    assignment[b, j, c] = 1.0
                    hard_assignment[b, j] = c
                # anchor_indices: segment midpoint joint (debug/visualize)
                anchor_indices_b[b, c] = seg_joints[len(seg_joints) // 2]
                # pooled_skel: mean of s_j across valid joints in segment
                valid_in_seg = [j for j in seg_joints if jm_b[j]]
                if valid_in_seg:
                    pooled_skel_emb[b, c] = skeleton_embeddings[b, valid_in_seg].mean(dim=0)

            # Build pooled_adjacency: two segments connected iff any joints
            # in them are adjacent in the original tree.
            # seg_of[j] = c for j in [0, J_valid_b)
            seg_of = [-1] * J_valid_b
            for c, seg_joints in enumerate(segments_b):
                for j in seg_joints:
                    seg_of[j] = c
            # Iterate original edges (parent → child for j >= 1)
            for j in range(1, J_valid_b):
                p = parents_b[j]
                if p < 0:
                    continue
                cj = seg_of[j]
                cp = seg_of[p]
                if cj != cp:  # cross-segment edge
                    pooled_adj[b, cj, cp] = 1.0
                    pooled_adj[b, cp, cj] = 1.0
            # Diagonal stays 0 (no self-loops)

        # --- Pooled geodesic via Floyd on pooled_adj (batched) ---
        pooled_geo = floyd_shortest_path(pooled_adj, new_coarse_mask)

        # --- Aux losses: zero tensors (v2 has no anchor-distance semantics) ---
        zero_scalar = torch.zeros((), device=ref_device, dtype=torch.float32)
        aux_losses = {
            "mincut": zero_scalar,
            "mincut_cut": zero_scalar,
            "mincut_ortho": zero_scalar,
            "locality": zero_scalar,
            "entropy": zero_scalar,
        }

        return {
            "assignment": assignment,
            "hard_assignment": hard_assignment,
            "pooled_adjacency": pooled_adj,
            "pooled_geodesic": pooled_geo,
            "pooled_mask": new_coarse_mask,
            "pooled_skeleton_embeddings": pooled_skel_emb,
            "anchor_indices": anchor_indices_b,
            "aux_losses": aux_losses,
        }

    # ------------------------------------------------------------------ #
    # Feature pool + temporal downsample (motion-dependent, mirrors v1) #
    # ------------------------------------------------------------------ #
    def _pool_features(
        self,
        joint_features: torch.Tensor,  # [B, T, J, D]
        P: torch.Tensor,               # [B, J, C], hard 1-of-K
        frame_mask: torch.Tensor,      # [B, T]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mass-normalized skeletal pool + temporal AvgPool1d.

        For hard P (1-of-K), mass[b, c] = #joints assigned to segment c.
        Mean is over valid joints in segment (hard P guarantees integer counts).
        """
        h_pool = torch.einsum("bjc,btjd->btcd", P, joint_features)
        mass = P.sum(dim=1).clamp(min=_MASS_FLOOR)  # [B, C]
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
        """v2 EdgeSegmentPool forward. Same dict schema as DynamicGraphPool."""
        # --- Motion-side validation ---
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
        if frame_mask.shape != (B, T) or frame_mask.dtype != torch.bool:
            raise ValueError(
                f"frame_mask must be [B={B}, T={T}] bool, got "
                f"{tuple(frame_mask.shape)} dtype {frame_mask.dtype}"
            )
        ref_device = joint_features.device
        if frame_mask.device != ref_device:
            raise ValueError(
                f"EdgeSegmentPool: frame_mask.device {frame_mask.device} != "
                f"joint_features.device {ref_device}"
            )
        if skeleton_embeddings.device != ref_device:
            raise ValueError(
                f"EdgeSegmentPool: skeleton_embeddings.device "
                f"{skeleton_embeddings.device} != joint_features.device {ref_device}"
            )
        if joint_features.dtype != torch.float32:
            raise ValueError(
                f"EdgeSegmentPool: joint_features.dtype must be float32, "
                f"got {joint_features.dtype}"
            )
        if not torch.isfinite(joint_features).all():
            raise ValueError("EdgeSegmentPool: joint_features contains NaN/Inf")
        if T % self.temporal_stride != 0:
            raise ValueError(
                f"EdgeSegmentPool: T={T} must be divisible by "
                f"temporal_stride={self.temporal_stride} (pad upstream)"
            )

        # --- Geometry (delegate) ---
        geom = self.compute_assignment_and_graph(
            skeleton_embeddings=skeleton_embeddings,
            adjacency=adjacency,
            geodesic_dist=geodesic_dist,
            joint_mask=joint_mask,
            parent_indices=parent_indices,
            anchor_indices=anchor_indices,
            coarse_mask=coarse_mask,
        )

        # --- Feature pool + temporal downsample (motion-dep) ---
        pooled_features, frame_mask_down = self._pool_features(
            joint_features, geom["assignment"], frame_mask,
        )

        return {
            "pooled_features": pooled_features,
            "frame_mask_down": frame_mask_down,
            **geom,
        }
