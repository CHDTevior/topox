"""Graph utilities for graph_salad's dynamic skeleton pool.

Two function groups by usage site:

A) Torch-batched (called inside forward pass; J_max ≤ 160 for our dataset):
   - floyd_shortest_path: dense Floyd-Warshall on padded adjacency, masking padded
     joints. O(B·J^3) but J ≤ 160 → ~4M ops per batch on GPU, negligible cost.
   - build_coarse_adjacency_from_hard_assign: lift fine edges to coarse graph by
     argmax assignment.

B) Per-sample numpy / list (called once-per-skeleton during pool init or batch prep):
   - find_anchors_rulebased: plan §6.2 anchor rules (root + degree≥3 + leaf +
     chain-chunked long limbs)
   - decompose_chains: root-to-leaf path decomposition
   - topological_order_with_root_first: permutation preserving root=0 + parent
     before child (codex 3.3 invariant)
   - assert_root_first_parent_order: validator
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


_FLOYD_INF = float("inf")


def _validate_parent_tree(parent_indices: list[int]) -> None:
    """Validate that parent_indices forms a single-rooted, acyclic, connected tree.

    Looser than ``assert_root_first_parent_order``: does NOT require root=0
    or parent-before-child ordering. Use this for utilities that accept
    arbitrary parent orderings (decompose_chains, find_anchors_rulebased,
    topological_order_with_root_first).

    Raises ValueError on:
      - Multi-root or no-root (number of -1 entries != 1)
      - Out-of-range parents (parent < -1, parent >= J, or parent == self)
      - Cycles (any joint reachable from root more than once)
      - Disconnected components (any joint unreachable from root)
    """
    J = len(parent_indices)
    if J == 0:
        return
    roots = [j for j, p in enumerate(parent_indices) if p == -1]
    if len(roots) != 1:
        raise ValueError(
            f"_validate_parent_tree: expected exactly 1 root (parents[i]==-1), "
            f"found {len(roots)} at {roots}"
        )
    root = roots[0]
    for j, p in enumerate(parent_indices):
        if j == root:
            continue
        if p < -1 or p >= J:
            raise ValueError(
                f"_validate_parent_tree: parents[{j}]={p} out of range [0,{J})"
            )
        if p == j:
            raise ValueError(f"_validate_parent_tree: self-loop at parents[{j}]")
    # BFS from root: must visit each joint exactly once
    children: list[list[int]] = [[] for _ in range(J)]
    for j, p in enumerate(parent_indices):
        if p >= 0:
            children[p].append(j)
    visited = {root}
    queue = [root]
    while queue:
        curr = queue.pop(0)
        for c in children[curr]:
            if c in visited:
                raise ValueError(
                    f"_validate_parent_tree: cycle detected — joint {c} reached twice"
                )
            visited.add(c)
            queue.append(c)
    if len(visited) != J:
        unreached = sorted(set(range(J)) - visited)
        raise ValueError(
            f"_validate_parent_tree: disconnected — joints {unreached} unreachable from root {root}"
        )


def floyd_shortest_path(
    adjacency: torch.Tensor,
    joint_mask: torch.Tensor,
) -> torch.Tensor:
    """Dense Floyd-Warshall shortest path on a padded undirected graph.

    Args:
        adjacency: [B, J, J] float tensor. Nonzero entry = edge (treated as
            single hop; edge weight is ignored — we count graph hops).
        joint_mask: [B, J] bool tensor. True = valid joint.

    Returns:
        [B, J, J] float tensor of geodesic distances in hops. Distance for
        any padded (masked-off) row or column, or for unreachable pairs, is
        +inf. Diagonal is 0.

    Notes:
        - Pure no-grad operation. Output is detached from autograd.
        - Edge weight is fixed at 1 hop. If you need weighted shortest path,
          pass a graph where each existing edge encodes its own initial
          distance (this function will min-aggregate hops, not edge weights).
    """
    if adjacency.dim() != 3:
        raise ValueError(f"adjacency must be [B, J, J], got shape {tuple(adjacency.shape)}")
    if joint_mask.dim() != 2:
        raise ValueError(f"joint_mask must be [B, J], got shape {tuple(joint_mask.shape)}")
    if adjacency.shape[0] != joint_mask.shape[0]:
        raise ValueError("adjacency and joint_mask must share batch dim")
    if adjacency.shape[1] != adjacency.shape[2] or adjacency.shape[1] != joint_mask.shape[1]:
        raise ValueError("adjacency must be square and match joint_mask J")

    B, J, _ = adjacency.shape
    device = adjacency.device

    with torch.no_grad():
        # dist[b, i, j] = 1 if edge (i,j) exists AND both i,j valid, else +inf
        valid_pair = joint_mask[:, :, None] & joint_mask[:, None, :]  # [B, J, J]
        has_edge = (adjacency > 0) & valid_pair
        dist = torch.where(
            has_edge,
            torch.ones_like(adjacency),
            torch.full_like(adjacency, _FLOYD_INF),
        )
        # Diagonal = 0 for valid joints; padded rows stay +inf along diagonal too,
        # so that callers can use isfinite() as the "valid" mask of the result.
        diag_valid = joint_mask.float()  # [B, J]
        eye = torch.eye(J, device=device, dtype=adjacency.dtype).expand(B, J, J)
        dist = torch.where(eye.bool() & valid_pair, torch.zeros_like(dist), dist)

        # Floyd-Warshall
        for k in range(J):
            # Skip k if it is masked out in every batch element — minor speedup
            # for very-mixed batches; correctness holds either way because the
            # row/col of a masked k is +inf and contributes nothing.
            via_k = dist[:, :, k : k + 1] + dist[:, k : k + 1, :]  # [B, J, J]
            dist = torch.minimum(dist, via_k)

    # Re-mask padded rows/cols defensively: their diagonal got set to 0 above
    # only if the joint is valid in joint_mask. Padded diagonals stay +inf.
    return dist


def build_coarse_adjacency_from_hard_assign(
    fine_adjacency: torch.Tensor,
    hard_assignment: torch.Tensor,
    fine_mask: torch.Tensor,
    coarse_mask: torch.Tensor,
) -> torch.Tensor:
    """Lift fine-graph edges to a coarse graph via hard assignment (argmax of S).

    A coarse edge (c1, c2) exists iff there is at least one valid fine edge
    (u, v) such that hard_c(u) = c1 and hard_c(v) = c2 and c1 != c2.

    Args:
        fine_adjacency: [B, J, J] float, fine graph adjacency.
        hard_assignment: [B, J] int64, the coarse-node id each fine joint
            belongs to. Values must be in [0, C). Padded joints' assignment
            is ignored.
        fine_mask: [B, J] bool, valid fine joints.
        coarse_mask: [B, C] bool, valid coarse nodes.

    Returns:
        [B, C, C] float adjacency of the coarse graph (binary 0/1).
        Diagonal is zero (no self-loops). Padded coarse rows/cols are zero.
    """
    if fine_adjacency.dim() != 3:
        raise ValueError("fine_adjacency must be [B, J, J]")
    if hard_assignment.dim() != 2:
        raise ValueError("hard_assignment must be [B, J]")
    if fine_mask.dim() != 2 or coarse_mask.dim() != 2:
        raise ValueError("masks must be [B, J] and [B, C]")
    if hard_assignment.dtype != torch.long:
        raise ValueError(f"hard_assignment must be int64, got {hard_assignment.dtype}")
    if fine_mask.dtype != torch.bool:
        raise ValueError(f"fine_mask must be bool, got {fine_mask.dtype}")
    if coarse_mask.dtype != torch.bool:
        raise ValueError(f"coarse_mask must be bool, got {coarse_mask.dtype}")

    B, J, _ = fine_adjacency.shape
    C = coarse_mask.shape[1]
    if coarse_mask.shape[0] != B:
        raise ValueError("coarse_mask batch must match fine_adjacency batch")
    if hard_assignment.shape != (B, J):
        raise ValueError(
            f"hard_assignment shape must be ({B}, {J}), got {tuple(hard_assignment.shape)}"
        )
    if fine_mask.shape != (B, J):
        raise ValueError(
            f"fine_mask shape must be ({B}, {J}), got {tuple(fine_mask.shape)}"
        )

    # Validate assignment for VALID fine joints (padded ones are ignored).
    # We require: for every (b, j) with fine_mask[b, j]=True,
    #   - 0 <= hard_assignment[b, j] < C
    #   - coarse_mask[b, hard_assignment[b, j]] == True
    # Raise loud rather than silently clamping (codex review M1.0 R12 fix).
    valid_assign_range = (hard_assignment >= 0) & (hard_assignment < C)
    if not torch.all(valid_assign_range | ~fine_mask):
        bad_pos = ((~valid_assign_range) & fine_mask).nonzero(as_tuple=False)
        bad_vals = hard_assignment[fine_mask & ~valid_assign_range]
        raise ValueError(
            f"hard_assignment has out-of-range values for valid joints; "
            f"valid range [0, {C}). Example offending (batch, joint, value) = "
            f"({bad_pos[0, 0].item()}, {bad_pos[0, 1].item()}, {bad_vals[0].item()})"
        )
    # Now safe to gather: assignment is in-range for valid joints, padded joints
    # we still need to clamp internally (their value is garbage but masked out).
    safe_assign = hard_assignment.clamp(min=0, max=C - 1)
    assigned_coarse_valid = torch.gather(coarse_mask, 1, safe_assign)  # [B, J]
    if not torch.all(assigned_coarse_valid | ~fine_mask):
        bad_pos = ((~assigned_coarse_valid) & fine_mask).nonzero(as_tuple=False)
        raise ValueError(
            f"hard_assignment maps a valid joint to a coarse_mask=False coarse id "
            f"(inactive coarse node). Example offending (batch, joint, coarse_id) = "
            f"({bad_pos[0, 0].item()}, {bad_pos[0, 1].item()}, "
            f"{hard_assignment[bad_pos[0, 0], bad_pos[0, 1]].item()})"
        )

    device = fine_adjacency.device

    with torch.no_grad():
        # one_hot assignment with padded joints zeroed
        # P[b, j, c] = 1 if hard_assignment[b, j] = c AND fine_mask[b, j], else 0
        P = torch.zeros(B, J, C, device=device, dtype=fine_adjacency.dtype)
        P.scatter_(2, safe_assign.unsqueeze(-1), 1.0)
        P = P * fine_mask[:, :, None].to(P.dtype)

        # valid fine edges only
        edge_mask_bool = fine_mask[:, :, None] & fine_mask[:, None, :]  # [B, J, J]
        A_fine = fine_adjacency * edge_mask_bool.to(fine_adjacency.dtype)

        # A_coarse[b, c1, c2] = sum_uv P[b,u,c1] A_fine[b,u,v] P[b,v,c2]
        # einsum is clearer than chained matmul.
        A_coarse = torch.einsum("buc,buv,bvd->bcd", P, A_fine, P)

        # Binarize, zero diagonal, mask invalid coarse rows/cols
        A_coarse = (A_coarse > 0).to(fine_adjacency.dtype)
        eye = torch.eye(C, device=device, dtype=A_coarse.dtype).expand(B, C, C)
        A_coarse = A_coarse * (1.0 - eye)
        coarse_valid = coarse_mask.to(A_coarse.dtype)
        A_coarse = A_coarse * coarse_valid[:, :, None] * coarse_valid[:, None, :]

    return A_coarse


def assert_root_first_parent_order(parent_indices: Iterable[int]) -> None:
    """Validate that parents[0] == -1 (root) and parents[i] < i for all i > 0.

    This is the topo-order invariant required by treeik_decoder.py (codex
    review §3.3): root must be index 0 and every joint's parent must come
    before it in the index order. Plan §6 ordering of dynamic pool's coarse
    nodes must also satisfy this invariant.

    Raises:
        ValueError if the invariant is violated.
    """
    pi = list(parent_indices)
    J = len(pi)
    if J == 0:
        return
    if pi[0] != -1:
        raise ValueError(f"root invariant violated: parents[0]={pi[0]}, expected -1")
    for i, p in enumerate(pi):
        if i == 0:
            continue
        if p == -1:
            raise ValueError(f"multiple roots detected at index {i} (parents[{i}]=-1)")
        if p < 0 or p >= i:
            raise ValueError(
                f"parent-before-child violated at index {i}: parents[{i}]={p}, must be in [0, {i})"
            )


def topological_order_with_root_first(parent_indices: Iterable[int]) -> list[int]:
    """Return a permutation `perm` such that, if you reorder joints by `perm`,
    the resulting parent_indices satisfies root=0 + parent-before-child.

    Args:
        parent_indices: length-J iterable, -1 for the (single) root.

    Returns:
        perm: list[int] of length J. perm[new_index] = old_index.

    Notes:
        - If input already satisfies the invariant, returns identity [0, ..., J-1].
        - Uses BFS from root. Children visited in order of their original index
          (deterministic, no randomness).
        - If the graph has multiple roots or disconnected components, raises.
    """
    pi = list(parent_indices)
    J = len(pi)
    if J == 0:
        return []
    _validate_parent_tree(pi)

    root = pi.index(-1)
    children: list[list[int]] = [[] for _ in range(J)]
    for j, p in enumerate(pi):
        if p >= 0:
            children[p].append(j)

    perm: list[int] = []
    queue: list[int] = [root]
    while queue:
        curr = queue.pop(0)
        perm.append(curr)
        for child in sorted(children[curr]):
            queue.append(child)
    return perm


def decompose_chains(parent_indices: Iterable[int]) -> list[list[int]]:
    """Decompose skeleton tree into root-to-leaf chains.

    A chain is the unique path from the root to a leaf, expressed as a list
    of joint indices in root→leaf order. Internal joints belong to multiple
    chains (one per leaf in their subtree).

    Args:
        parent_indices: length-J iterable, -1 for root.

    Returns:
        list of chains. Each chain is a list[int] of joint indices starting
        at the root and ending at a leaf.

    Notes:
        - If the skeleton has L leaves, this returns L chains.
        - A leaf is a joint with no children (degree 0 in the directed-from-root
          sense, equivalently degree 1 in undirected sense for non-root joints).
        - Order of returned chains: by leaf joint index ascending.
    """
    pi = list(parent_indices)
    J = len(pi)
    if J == 0:
        return []
    _validate_parent_tree(pi)

    # Identify leaves
    has_child = [False] * J
    for j, p in enumerate(pi):
        if p >= 0:
            has_child[p] = True
    leaves = [j for j in range(J) if not has_child[j]]

    chains: list[list[int]] = []
    for leaf in sorted(leaves):
        path: list[int] = []
        curr = leaf
        while curr != -1:
            path.append(curr)
            curr = pi[curr]
        path.reverse()  # now root → leaf
        chains.append(path)

    return chains


def find_anchors_rulebased(
    parent_indices: Iterable[int],
    *,
    max_chain_chunk_len: int = 5,
) -> list[int]:
    """Plan §6.2 deterministic anchor generation rules.

    Anchors are the coarse-graph node representatives chosen from the fine
    joint set by simple graph-structure rules — no learning. Output is the
    set used by both the dynamic pool (as candidate-anchor mask) and the
    deterministic pool (as the fixed anchor set).

    Rules applied in order:
        1. Root (parents[i] == -1) is always an anchor.
        2. Every branch joint (children count ≥ 2) is an anchor.
        3. Every leaf joint (no children) is an anchor.
        4. For each root-to-leaf chain, every `max_chain_chunk_len`-th
           non-anchor internal joint is promoted to an anchor (chain chunking).
           Direction: walk leaf→root, so the joint immediately above the leaf
           at depth max_chain_chunk_len gets promoted first.

    Args:
        parent_indices: length-J iterable, -1 for root.
        max_chain_chunk_len: chain-chunk granularity. Default 5 matches plan
            §6.2's "每 N joints 生成一个 coarse anchor" for long chains.

    Returns:
        sorted list[int] of anchor joint indices.

    Notes:
        - Returns indices in ascending order (deterministic).
        - Plan §6.2 rule 6 ("short limbs of 2-4 joints poolable into one
          coarse limb node") is intentionally NOT implemented in M1 — codex
          pre-scaffold review accepted root+branch+leaf+chain-chunk as the
          minimum sufficient set for the 3-way ablation. The short-limb rule
          can be added in M1.5 only if visual QA shows over-segmentation on
          insect/spider topologies.
    """
    if max_chain_chunk_len < 1:
        raise ValueError(f"max_chain_chunk_len must be >= 1, got {max_chain_chunk_len}")

    pi = list(parent_indices)
    J = len(pi)
    if J == 0:
        return []
    _validate_parent_tree(pi)

    # Count children per joint
    children: list[list[int]] = [[] for _ in range(J)]
    for j, p in enumerate(pi):
        if p >= 0:
            children[p].append(j)

    anchors: set[int] = set()

    # Rule 1: root
    for j in range(J):
        if pi[j] == -1:
            anchors.add(j)

    # Rule 2: branch (≥ 2 children)
    for j in range(J):
        if len(children[j]) >= 2:
            anchors.add(j)

    # Rule 3: leaf (no children)
    for j in range(J):
        if len(children[j]) == 0:
            anchors.add(j)

    # Rule 4: chain chunking — walk each root-to-leaf path, and every
    # max_chain_chunk_len joints between existing anchors promote one.
    # We walk along the unique parent pointer chain from each leaf upward;
    # on the way, count consecutive non-anchor joints, and promote each
    # max_chain_chunk_len-th one (counted from the deeper end).
    leaves = [j for j in range(J) if not children[j]]
    for leaf in leaves:
        consec = 0
        curr = pi[leaf]  # the leaf itself is already an anchor
        while curr != -1:
            if curr in anchors:
                consec = 0
            else:
                consec += 1
                if consec >= max_chain_chunk_len:
                    anchors.add(curr)
                    consec = 0
            curr = pi[curr]

    return sorted(anchors)
