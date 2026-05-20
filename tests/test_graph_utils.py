"""Unit tests for src/models/graph_salad/graph_utils.py — M1.0 preflight.

Run from repo root (no pytest dep; uses stdlib unittest):
    python -m unittest tests.test_graph_utils -v
or
    python tests/test_graph_utils.py

These tests are CPU-only, fixed-seed (deterministic). They verify the
contract that DynamicGraphPool / DeterministicGraphPool / GraphMotionVAE
in M1.1+ will depend on.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

# Make src importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.graph_salad.graph_utils import (
    assert_root_first_parent_order,
    build_coarse_adjacency_from_hard_assign,
    decompose_chains,
    find_anchors_rulebased,
    floyd_shortest_path,
    topological_order_with_root_first,
)
# Private validator — exposed only for tests.
from src.models.graph_salad.graph_utils import _validate_parent_tree


def _adj_from_parents(parents: list[int]) -> torch.Tensor:
    """Build [J, J] symmetric adjacency from a parent list (-1 for root)."""
    J = len(parents)
    A = torch.zeros(J, J)
    for j, p in enumerate(parents):
        if p >= 0:
            A[j, p] = 1.0
            A[p, j] = 1.0
    return A


class FloydShortestPathTests(unittest.TestCase):

    def test_line_graph(self):
        # Line: 0 - 1 - 2 - 3 - 4
        parents = [-1, 0, 1, 2, 3]
        A = _adj_from_parents(parents).unsqueeze(0)
        mask = torch.ones(1, 5, dtype=torch.bool)
        d = floyd_shortest_path(A, mask)
        expected = torch.tensor(
            [[abs(i - j) for j in range(5)] for i in range(5)], dtype=torch.float32
        )
        self.assertTrue(torch.allclose(d[0], expected), f"line graph wrong:\n{d[0]}")

    def test_t_shape(self):
        # 0 - 1 - {2, 3}
        parents = [-1, 0, 1, 1]
        A = _adj_from_parents(parents).unsqueeze(0)
        mask = torch.ones(1, 4, dtype=torch.bool)
        d = floyd_shortest_path(A, mask)
        expected = torch.tensor(
            [[0, 1, 2, 2], [1, 0, 1, 1], [2, 1, 0, 2], [2, 1, 2, 0]], dtype=torch.float32
        )
        self.assertTrue(torch.allclose(d[0], expected), f"T-shape wrong:\n{d[0]}")

    def test_padding_mask(self):
        # 5 padded; only first 3 valid (line 0-1-2)
        parents = [-1, 0, 1, 0, 0]
        A = _adj_from_parents(parents).unsqueeze(0)
        mask = torch.tensor([[True, True, True, False, False]])
        d = floyd_shortest_path(A, mask)
        self.assertEqual(d[0, 0, 1].item(), 1.0)
        self.assertEqual(d[0, 0, 2].item(), 2.0)
        self.assertEqual(d[0, 1, 2].item(), 1.0)
        self.assertEqual(d[0, 0, 0].item(), 0.0)
        self.assertTrue(torch.isinf(d[0, 0, 3]))
        self.assertTrue(torch.isinf(d[0, 3, 0]))
        self.assertTrue(torch.isinf(d[0, 3, 3]))
        self.assertTrue(torch.isinf(d[0, 4, 4]))

    def test_batched(self):
        parents = [-1, 0, 1, 2, 3]
        A = _adj_from_parents(parents)
        A_batch = A.unsqueeze(0).expand(2, 5, 5).clone()
        mask = torch.tensor(
            [[True, True, True, True, True], [True, True, True, False, False]]
        )
        d = floyd_shortest_path(A_batch, mask)
        self.assertEqual(d[0, 0, 4].item(), 4.0)
        self.assertEqual(d[1, 0, 2].item(), 2.0)
        self.assertTrue(torch.isinf(d[1, 0, 3]))

    def test_disconnected_components(self):
        # Two components: {0,1} and {2,3}. d(0,2)=inf, d(1,3)=inf, but d(0,1)=1, d(2,3)=1.
        A = torch.zeros(1, 4, 4)
        A[0, 0, 1] = A[0, 1, 0] = 1.0
        A[0, 2, 3] = A[0, 3, 2] = 1.0
        mask = torch.ones(1, 4, dtype=torch.bool)
        d = floyd_shortest_path(A, mask)
        self.assertEqual(d[0, 0, 1].item(), 1.0)
        self.assertEqual(d[0, 2, 3].item(), 1.0)
        self.assertTrue(torch.isinf(d[0, 0, 2]))
        self.assertTrue(torch.isinf(d[0, 1, 3]))
        self.assertTrue(torch.isinf(d[0, 2, 0]))


class BuildCoarseAdjacencyTests(unittest.TestCase):

    def test_simple_split(self):
        # Fine 0-1-2-3 (line); 0,1→c0; 2,3→c1. Cross edge (1,2) → coarse (0,1)
        fine_adj = _adj_from_parents([-1, 0, 1, 2]).unsqueeze(0)
        hard = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
        fine_mask = torch.ones(1, 4, dtype=torch.bool)
        coarse_mask = torch.ones(1, 2, dtype=torch.bool)
        A_coarse = build_coarse_adjacency_from_hard_assign(
            fine_adj, hard, fine_mask, coarse_mask
        )
        expected = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
        self.assertTrue(torch.allclose(A_coarse, expected), f"wrong:\n{A_coarse}")

    def test_all_to_one_cluster(self):
        # Fine line 0-1-2; all → c0. Coarse adj = all zeros (no inter-cluster edges).
        fine_adj = _adj_from_parents([-1, 0, 1]).unsqueeze(0)
        hard = torch.tensor([[0, 0, 0]], dtype=torch.long)
        fine_mask = torch.ones(1, 3, dtype=torch.bool)
        coarse_mask = torch.tensor([[True, False]])
        A_coarse = build_coarse_adjacency_from_hard_assign(
            fine_adj, hard, fine_mask, coarse_mask
        )
        self.assertTrue(torch.all(A_coarse == 0), f"expected zero, got {A_coarse}")

    def test_padding_ignored(self):
        # Fine 0-1-2-3 with joint 3 padded. Hard assign [0,0,1,99] — joint 3
        # invalid value 99 is ignored because fine_mask[3]=False.
        fine_adj = _adj_from_parents([-1, 0, 1, 2]).unsqueeze(0)
        hard = torch.tensor([[0, 0, 1, 99]], dtype=torch.long)
        fine_mask = torch.tensor([[True, True, True, False]])
        coarse_mask = torch.tensor([[True, True]])
        A_coarse = build_coarse_adjacency_from_hard_assign(
            fine_adj, hard, fine_mask, coarse_mask
        )
        # Edge (1,2) crosses c0→c1; edge (2,3) involves padded → ignored
        expected = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
        self.assertTrue(torch.allclose(A_coarse, expected), f"padding leaked:\n{A_coarse}")

    def test_invalid_assignment_out_of_range_raises(self):
        # Joint 0 (valid) has assignment 99 → out of range. Must raise.
        fine_adj = _adj_from_parents([-1, 0]).unsqueeze(0)
        hard = torch.tensor([[99, 0]], dtype=torch.long)
        fine_mask = torch.ones(1, 2, dtype=torch.bool)
        coarse_mask = torch.ones(1, 2, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            build_coarse_adjacency_from_hard_assign(fine_adj, hard, fine_mask, coarse_mask)

    def test_invalid_assignment_negative_raises(self):
        fine_adj = _adj_from_parents([-1, 0]).unsqueeze(0)
        hard = torch.tensor([[-1, 0]], dtype=torch.long)
        fine_mask = torch.ones(1, 2, dtype=torch.bool)
        coarse_mask = torch.ones(1, 2, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            build_coarse_adjacency_from_hard_assign(fine_adj, hard, fine_mask, coarse_mask)

    def test_invalid_assignment_to_inactive_coarse_raises(self):
        # Joint 0 (valid) assigned to coarse 1, but coarse_mask[0,1]=False → inactive.
        fine_adj = _adj_from_parents([-1, 0]).unsqueeze(0)
        hard = torch.tensor([[0, 1]], dtype=torch.long)
        fine_mask = torch.ones(1, 2, dtype=torch.bool)
        coarse_mask = torch.tensor([[True, False]])  # c=1 inactive
        with self.assertRaisesRegex(ValueError, "inactive coarse"):
            build_coarse_adjacency_from_hard_assign(fine_adj, hard, fine_mask, coarse_mask)

    def test_non_bool_mask_raises(self):
        fine_adj = _adj_from_parents([-1, 0]).unsqueeze(0)
        hard = torch.tensor([[0, 0]], dtype=torch.long)
        # float mask instead of bool
        fine_mask_float = torch.ones(1, 2, dtype=torch.float32)
        coarse_mask = torch.ones(1, 1, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "fine_mask must be bool"):
            build_coarse_adjacency_from_hard_assign(fine_adj, hard, fine_mask_float, coarse_mask)

    def test_shape_mismatch_raises(self):
        fine_adj = _adj_from_parents([-1, 0, 1]).unsqueeze(0)  # J=3
        hard = torch.tensor([[0, 0]], dtype=torch.long)  # wrong shape (B=1, J=2)
        fine_mask = torch.ones(1, 3, dtype=torch.bool)
        coarse_mask = torch.ones(1, 1, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "hard_assignment shape"):
            build_coarse_adjacency_from_hard_assign(fine_adj, hard, fine_mask, coarse_mask)


class ValidateParentTreeTests(unittest.TestCase):

    def test_valid_tree(self):
        # Joint 2 is root; 0 and 1 are its children
        _validate_parent_tree([2, 2, -1])  # no exception

    def test_no_root_raises(self):
        # No -1 anywhere
        with self.assertRaisesRegex(ValueError, "expected exactly 1 root"):
            _validate_parent_tree([1, 0])

    def test_multi_root_raises(self):
        with self.assertRaisesRegex(ValueError, "expected exactly 1 root"):
            _validate_parent_tree([-1, -1, 0])

    def test_out_of_range_parent_raises(self):
        # parents[1]=5 but J=2 (only indices 0,1 are valid)
        with self.assertRaisesRegex(ValueError, "out of range"):
            _validate_parent_tree([-1, 5])

    def test_self_loop_raises(self):
        # joint 1 points to itself
        with self.assertRaisesRegex(ValueError, "self-loop"):
            _validate_parent_tree([-1, 1])

    def test_disconnected_raises(self):
        # 0 is root, 1 points to 0, 2 points to ... no one (2's parent is missing).
        # Actually [-1, 0, 0] is valid (all connected). For disconnect we need
        # 1 root + an isolated node, but every non-root must have a parent ≠ -1.
        # The simplest disconnect scenario: 2 roots. Already covered above.
        # An acyclic-but-actually-disconnected case requires a cycle hidden as
        # parent loop NOT touching root, e.g. parents = [-1, 3, 1, 2]:
        # root=0, joint 1's parent=3, joint 2's parent=1, joint 3's parent=2.
        # Joint 0 has no children -> BFS reaches only {0}; {1,2,3} form a cycle.
        with self.assertRaisesRegex(ValueError, "disconnected|cycle"):
            _validate_parent_tree([-1, 3, 1, 2])


class AssertRootFirstParentOrderTests(unittest.TestCase):

    def test_valid(self):
        assert_root_first_parent_order([-1, 0, 1, 0, 2])  # should not raise

    def test_root_not_zero(self):
        with self.assertRaisesRegex(ValueError, "root invariant"):
            assert_root_first_parent_order([0, -1, 1])

    def test_parent_after_child(self):
        with self.assertRaisesRegex(ValueError, "parent-before-child"):
            assert_root_first_parent_order([-1, 2, 0])

    def test_multiple_roots(self):
        with self.assertRaisesRegex(ValueError, "multiple roots"):
            assert_root_first_parent_order([-1, -1, 0])


class TopologicalOrderWithRootFirstTests(unittest.TestCase):

    def test_already_valid(self):
        parents = [-1, 0, 1, 0, 2]
        perm = topological_order_with_root_first(parents)
        self.assertEqual(perm[0], 0)
        self.assertEqual(len(perm), 5)
        self.assertEqual(set(perm), set(range(5)))
        inv = {old: new for new, old in enumerate(perm)}
        new_parents = [-1 if parents[old] == -1 else inv[parents[old]] for old in perm]
        assert_root_first_parent_order(new_parents)

    def test_reroots(self):
        # Joint 2 is root; need permutation that places 2 first
        parents = [2, 2, -1]
        perm = topological_order_with_root_first(parents)
        self.assertEqual(perm[0], 2)
        inv = {old: new for new, old in enumerate(perm)}
        new_parents = [-1 if parents[old] == -1 else inv[parents[old]] for old in perm]
        assert_root_first_parent_order(new_parents)

    def test_disconnected_raises(self):
        with self.assertRaises(ValueError):
            topological_order_with_root_first([-1, -1, 0])


class DecomposeChainsTests(unittest.TestCase):

    def test_single_chain(self):
        # Line 0-1-2-3
        chains = decompose_chains([-1, 0, 1, 2])
        self.assertEqual(chains, [[0, 1, 2, 3]])

    def test_t_shape(self):
        chains = decompose_chains([-1, 0, 1, 1])
        self.assertEqual(chains, [[0, 1, 2], [0, 1, 3]])

    def test_humanoid_like(self):
        # 0 root → 1 spine → {2,3,4,5} leaves
        chains = decompose_chains([-1, 0, 1, 1, 1, 1])
        self.assertEqual(len(chains), 4)
        for c in chains:
            self.assertEqual(c[0], 0)
            self.assertEqual(c[1], 1)
            self.assertIn(c[-1], {2, 3, 4, 5})

    def test_invalid_tree_raises(self):
        with self.assertRaisesRegex(ValueError, "1 root"):
            decompose_chains([-1, -1, 0])
        with self.assertRaisesRegex(ValueError, "out of range"):
            decompose_chains([-1, 5])


class FindAnchorsRulebasedTests(unittest.TestCase):

    def test_line_short(self):
        # 0 root - 1 internal - 2 leaf. With chunk=5, internal not promoted.
        anchors = find_anchors_rulebased([-1, 0, 1], max_chain_chunk_len=5)
        self.assertEqual(anchors, [0, 2])

    def test_line_long_chunked(self):
        # 11-joint line; chunk=3 → some intermediates promoted
        parents = [-1] + list(range(10))
        anchors = find_anchors_rulebased(parents, max_chain_chunk_len=3)
        self.assertIn(0, anchors)
        self.assertIn(10, anchors)
        intermediate = [a for a in anchors if a not in (0, 10)]
        self.assertGreaterEqual(
            len(intermediate), 1, f"chain chunking inactive: {anchors}"
        )
        self.assertEqual(anchors, sorted(anchors))

    def test_line_chunked_exact_positions(self):
        # Lock the chunking direction (leaf→root, every Nth non-anchor promoted).
        # 11-joint line, parents=[-1,0,1,...,9], leaf=10, chunk_len=3.
        # Algorithm walks parent(leaf)=9,8,7... incrementing consec; on every 3rd
        # consec non-anchor, promote and reset.
        #   9 (consec=1) → no
        #   8 (consec=2) → no
        #   7 (consec=3) → PROMOTE 7, reset
        #   6 (consec=1) → no
        #   5 (consec=2) → no
        #   4 (consec=3) → PROMOTE 4, reset
        #   3 (consec=1) → no
        #   2 (consec=2) → no
        #   1 (consec=3) → PROMOTE 1, reset
        #   0 = root, already in anchors
        # Final: {0, 1, 4, 7, 10}
        parents = [-1] + list(range(10))
        anchors = find_anchors_rulebased(parents, max_chain_chunk_len=3)
        self.assertEqual(
            anchors, [0, 1, 4, 7, 10],
            f"chunking direction or step is wrong; expected [0,1,4,7,10], got {anchors}",
        )

    def test_invalid_tree_raises(self):
        # multi-root via _validate_parent_tree
        with self.assertRaisesRegex(ValueError, "1 root"):
            find_anchors_rulebased([-1, -1, 0])
        # out-of-range parent
        with self.assertRaisesRegex(ValueError, "out of range"):
            find_anchors_rulebased([-1, 5])

    def test_invalid_chunk_len_raises(self):
        with self.assertRaisesRegex(ValueError, "max_chain_chunk_len must be >= 1"):
            find_anchors_rulebased([-1, 0, 1], max_chain_chunk_len=0)

    def test_branch_and_leaf(self):
        # 0 root → 1 (4 children) → {2,3,4,5} leaves
        anchors = find_anchors_rulebased([-1, 0, 1, 1, 1, 1])
        self.assertEqual(set(anchors), {0, 1, 2, 3, 4, 5})

    def test_single_joint(self):
        # 1-joint: both root and leaf — counted once
        self.assertEqual(find_anchors_rulebased([-1]), [0])

    def test_empty(self):
        self.assertEqual(find_anchors_rulebased([]), [])


class RealSkeletonSmokeTest(unittest.TestCase):

    def test_bat_sanity(self):
        """Sanity check against an actual animal skeleton from cs_sparse2full_tgt."""
        npz_path = (
            Path(__file__).resolve().parents[1]
            / "data" / "cs_sparse2full_tgt" / "skeletons" / "Bat.npz"
        )
        if not npz_path.exists():
            self.skipTest(f"Bat.npz not found at {npz_path}")
        d = np.load(npz_path, allow_pickle=True)
        parents = d["parent_indices"].tolist()
        J = len(parents)
        self.assertEqual(J, 48)

        # Root invariant
        assert_root_first_parent_order(parents)

        # Anchors should be a strict subset (< J) and include root + all leaves
        anchors = find_anchors_rulebased(parents)
        self.assertIn(0, anchors)
        self.assertTrue(1 <= len(anchors) < J)
        has_child = [False] * J
        for j, p in enumerate(parents):
            if p >= 0:
                has_child[p] = True
        leaves = {j for j in range(J) if not has_child[j]}
        self.assertTrue(leaves.issubset(set(anchors)))

        # Chain decomposition = #leaves chains
        chains = decompose_chains(parents)
        self.assertEqual(len(chains), len(leaves))

        # Floyd should match stored BFS geodesic on valid pairs
        A = torch.from_numpy(d["adjacency"]).float().unsqueeze(0)
        mask = torch.ones(1, J, dtype=torch.bool)
        d_floyd = floyd_shortest_path(A, mask)
        d_npz = torch.from_numpy(d["geodesic_dist"]).float()
        valid_finite = torch.isfinite(d_floyd[0]) & torch.isfinite(d_npz)
        diff = (d_floyd[0] - d_npz).abs()
        self.assertTrue(
            torch.all(diff[valid_finite] < 1e-5),
            f"Floyd vs stored BFS max diff = {diff[valid_finite].max().item()}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
