"""Unit tests for src/models/graph_salad/pool_dynamic.py — M1.2 step 2.

Run:
    python tests/test_pool_dynamic.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.graph_salad.pool_dynamic import DynamicGraphPool


def _adj_line(N: int) -> torch.Tensor:
    A = torch.zeros(N, N)
    for j in range(1, N):
        A[j, j - 1] = 1.0
        A[j - 1, j] = 1.0
    return A


def _geo_line(N: int) -> torch.Tensor:
    geo = torch.zeros(N, N)
    for i in range(N):
        for j in range(N):
            geo[i, j] = abs(i - j)
    return geo


def floyd_geo_from_adj_masked(adj: torch.Tensor, mask_first_n: int) -> torch.Tensor:
    """Floyd geodesic on the first mask_first_n joints; padded slots = 0."""
    B, J, _ = adj.shape
    geo = torch.zeros(B, J, J)
    INF = float("inf")
    for b in range(B):
        N = mask_first_n
        d = torch.full((N, N), INF)
        for i in range(N):
            d[i, i] = 0.0
        a_sub = adj[b, :N, :N]
        d = torch.where(a_sub > 0, torch.ones_like(a_sub), d)
        for k in range(N):
            d = torch.minimum(d, d[:, k:k+1] + d[k:k+1, :])
        geo[b, :N, :N] = d
    return geo


def _make_line_batch(B: int = 2, T: int = 8, J: int = 11, D: int = 16):
    """All B samples share same line skeleton, length J."""
    line_parents = [-1] + list(range(J - 1))
    parent_indices = [line_parents for _ in range(B)]
    joint_features = torch.randn(B, T, J, D)
    skeleton_embeddings = torch.randn(B, J, D)
    adj = _adj_line(J).expand(B, J, J).clone()
    geo = _geo_line(J).expand(B, J, J).clone()
    joint_mask = torch.ones(B, J, dtype=torch.bool)
    frame_mask = torch.ones(B, T, dtype=torch.bool)
    return joint_features, skeleton_embeddings, adj, geo, joint_mask, frame_mask, parent_indices


class DynamicGraphPoolTests(unittest.TestCase):

    def test_constructor_validation(self):
        with self.assertRaisesRegex(ValueError, "d_model must be > 0"):
            DynamicGraphPool(d_model=0, max_coarse=4)
        with self.assertRaisesRegex(ValueError, "max_coarse must be"):
            DynamicGraphPool(d_model=16, max_coarse=0)
        with self.assertRaisesRegex(TypeError, "d_model must be strict int"):
            DynamicGraphPool(d_model=16.0, max_coarse=4)
        with self.assertRaisesRegex(TypeError, "max_coarse must be strict int"):
            DynamicGraphPool(d_model=16, max_coarse=4.5)
        with self.assertRaisesRegex(TypeError, "max_coarse must be strict int"):
            DynamicGraphPool(d_model=16, max_coarse=True)  # bool subclass of int
        with self.assertRaisesRegex(ValueError, "local_radius"):
            DynamicGraphPool(d_model=16, max_coarse=4, local_radius=0)
        with self.assertRaisesRegex(ValueError, "max_chain_chunk_len"):
            DynamicGraphPool(d_model=16, max_coarse=4, max_chain_chunk_len=0)
        with self.assertRaisesRegex(ValueError, "temperature"):
            DynamicGraphPool(d_model=16, max_coarse=4, temperature=0)
        with self.assertRaisesRegex(ValueError, "temporal_stride"):
            DynamicGraphPool(d_model=16, max_coarse=4, temporal_stride=0)

    def test_forward_shapes_line_skeleton(self):
        D, T, J = 16, 8, 11
        pool = DynamicGraphPool(d_model=D, max_coarse=8, local_radius=5)
        jf, se, adj, geo, jm, fm, pi = _make_line_batch(B=2, T=T, J=J, D=D)
        out = pool(jf, se, adj, geo, jm, fm, pi)

        # Anchor count from rule on 11-line w/ chunk_len=5:
        # root=0, leaf=10 -> 2; chunking: walk 9..1, every 5th promote.
        # leaf-> consec=1..5 promotes at 5; consec=1..5 again promotes at 5 from there.
        # Result: anchors = {0, 5, 10} or similar; max 3 here. C_max=8 accommodates.
        # We just check shapes, not exact anchor count.
        B = 2
        T_lat = T // 2
        self.assertEqual(out["pooled_features"].shape, (B, T_lat, 8, D))
        self.assertEqual(out["assignment"].shape, (B, J, 8))
        self.assertEqual(out["hard_assignment"].shape, (B, J))
        self.assertEqual(out["pooled_adjacency"].shape, (B, 8, 8))
        self.assertEqual(out["pooled_geodesic"].shape, (B, 8, 8))
        self.assertEqual(out["pooled_mask"].shape, (B, 8))
        self.assertEqual(out["pooled_skeleton_embeddings"].shape, (B, 8, D))
        self.assertEqual(out["frame_mask_down"].shape, (B, T_lat))
        self.assertEqual(out["anchor_indices"].shape, (B, 8))
        # Aux losses non-NaN
        for k, v in out["aux_losses"].items():
            self.assertTrue(torch.isfinite(v).all(), f"aux_losses[{k}] non-finite")

    def test_pooled_adjacency_is_binary_symmetric_zero_diagonal(self):
        D = 16
        pool = DynamicGraphPool(d_model=D, max_coarse=8, local_radius=5)
        jf, se, adj, geo, jm, fm, pi = _make_line_batch(B=1, T=4, J=11, D=D)
        out = pool(jf, se, adj, geo, jm, fm, pi)
        adj_out = out["pooled_adjacency"][0]
        # Binary
        self.assertTrue(((adj_out == 0) | (adj_out == 1)).all())
        # Symmetric
        self.assertTrue(torch.equal(adj_out, adj_out.T))
        # Zero diagonal
        self.assertEqual(adj_out.diag().sum().item(), 0.0)

    def test_mass_normalized_pool_preserves_simple_case(self):
        # If P has rows that are one-hot to a single anchor (sharp), and the
        # one anchor receives N joints with feature value v each, mass-norm
        # output should be v (no scaling). Use temperature -> 0 to sharpen.
        # We sidestep this by directly testing pool with a synthetic P via
        # _pool_features.
        D, T = 16, 4
        pool = DynamicGraphPool(d_model=D, max_coarse=2)
        J = 6
        h = torch.full((1, T, J, D), 3.0)
        P = torch.zeros(1, J, 2)
        P[0, :3, 0] = 1.0
        P[0, 3:, 1] = 1.0
        fm = torch.ones(1, T, dtype=torch.bool)
        h_pool, _ = pool._pool_features(h, P, fm)
        # Each coarse node receives 3 joints * 3.0 / 3 (mass) = 3.0 per channel
        # after temporal pool stride=2 → T_lat=2; values still 3.0.
        self.assertTrue(torch.allclose(h_pool, torch.full_like(h_pool, 3.0)))

    def test_anchor_overflow_raises(self):
        # max_coarse too small for line skeleton's anchor count.
        pool = DynamicGraphPool(d_model=16, max_coarse=1)
        jf, se, adj, geo, jm, fm, pi = _make_line_batch(B=1, T=4, J=11, D=16)
        with self.assertRaisesRegex(ValueError, "anchors >.*max_coarse"):
            pool(jf, se, adj, geo, jm, fm, pi)

    def test_no_candidate_raises(self):
        # local_radius=1 + sparse anchors should leave some joint without
        # candidate.
        pool = DynamicGraphPool(d_model=16, max_coarse=4, local_radius=1,
                                max_chain_chunk_len=100)
        # chunk_len=100 means no chunking promotion for J=11 line → anchors
        # are only root=0 and leaf=10. local_radius=1 won't cover joint 5.
        jf, se, adj, geo, jm, fm, pi = _make_line_batch(B=1, T=4, J=11, D=16)
        with self.assertRaisesRegex(ValueError, "no candidate anchor within local_radius"):
            pool(jf, se, adj, geo, jm, fm, pi)

    def test_padded_joint_assignment_zeroed(self):
        D, T, J_max = 16, 4, 10
        pool = DynamicGraphPool(d_model=D, max_coarse=6, local_radius=5)
        # Sample with J_valid=6 inside J_max=10; padded joints have garbage features
        # but must not appear in assignment rows.
        line_parents = [-1] + list(range(5))  # 6 joints
        parent_indices = [line_parents]
        jf = torch.randn(1, T, J_max, D)
        se = torch.randn(1, J_max, D)
        adj = torch.zeros(1, J_max, J_max)
        adj[0, :6, :6] = _adj_line(6)
        geo = torch.zeros(1, J_max, J_max)
        geo[0, :6, :6] = _geo_line(6)
        # Padded geo: set to large finite so it's outside local_radius
        for i in range(6, J_max):
            for j in range(J_max):
                if i != j:
                    geo[0, i, j] = 100.0
                    geo[0, j, i] = 100.0
        jm = torch.zeros(1, J_max, dtype=torch.bool)
        jm[0, :6] = True
        fm = torch.ones(1, T, dtype=torch.bool)
        out = pool(jf, se, adj, geo, jm, fm, parent_indices)
        # Padded joints should have zero assignment rows
        self.assertTrue(out["assignment"][0, 6:, :].abs().sum().item() == 0.0)
        # Padded coarse columns also zero
        self.assertTrue(out["assignment"][0, :, ~out["pooled_mask"][0]].abs().sum().item() == 0.0)

    def test_parent_adjacency_mismatch_raises(self):
        # R12 #21: parent_indices encode a different graph than adjacency.
        pool = DynamicGraphPool(d_model=16, max_coarse=8, local_radius=5)
        jf, se, adj, geo, jm, fm, _ = _make_line_batch(B=1, T=4, J=6, D=16)
        # Star parents (all children of root 0) vs line adjacency → mismatch
        star_parents = [[-1, 0, 0, 0, 0, 0]]
        with self.assertRaisesRegex(ValueError, "parent_indices does not match adjacency"):
            pool(jf, se, adj, geo, jm, fm, parent_indices=star_parents)

    def test_local_radius_bool_raises(self):
        with self.assertRaisesRegex(TypeError, "local_radius must be strict int"):
            DynamicGraphPool(d_model=16, max_coarse=4, local_radius=True)

    def test_max_chain_chunk_len_float_raises(self):
        with self.assertRaisesRegex(TypeError, "max_chain_chunk_len must be strict int"):
            DynamicGraphPool(d_model=16, max_coarse=4, max_chain_chunk_len=5.0)

    def test_parent_not_root_first_raises(self):
        # parent_indices with root NOT at index 0 (codex round 14 advisory)
        pool = DynamicGraphPool(d_model=16, max_coarse=4, local_radius=5)
        jf, se, adj, geo, jm, fm, _ = _make_line_batch(B=1, T=4, J=6, D=16)
        # Bad parents: root at index 2 (e.g., [1, 2, -1, 2, 3, 4])
        # Must construct adj to match this so adj-match check doesn't fire first
        bad_parents = [[1, 2, -1, 2, 3, 4]]
        expected_adj = torch.zeros(1, 6, 6)
        for j, p in enumerate(bad_parents[0]):
            if p >= 0:
                expected_adj[0, j, p] = 1.0
                expected_adj[0, p, j] = 1.0
        # Floyd on this adj for the geo
        from src.models.graph_salad.graph_utils import floyd_shortest_path
        new_geo = floyd_shortest_path(expected_adj, jm)
        with self.assertRaisesRegex(ValueError, "violates FK ordering"):
            pool(jf, se, expected_adj, new_geo, jm, fm, parent_indices=bad_parents)

    def test_parent_subtle_adj_mismatch_raises(self):
        # 5e-7 spurious entry — caught by Floyd consistency (which treats any
        # >0 entry as an edge) OR by parent-adj equality (atol=0). Either way,
        # the bug fails loud rather than silently propagating.
        pool = DynamicGraphPool(d_model=16, max_coarse=4, local_radius=5)
        jf, se, adj, geo, jm, fm, pi = _make_line_batch(B=1, T=4, J=6, D=16)
        adj[0, 0, 5] = 5e-7
        adj[0, 5, 0] = 5e-7  # symmetric
        # Either parent-adj equality (atol=0) or Floyd consistency fires.
        with self.assertRaisesRegex(ValueError, "inconsistent with.*shortest-path|does not match adjacency"):
            pool(jf, se, adj, geo, jm, fm, parent_indices=pi)

    def test_parent_length_mismatch_raises(self):
        pool = DynamicGraphPool(d_model=16, max_coarse=8, local_radius=5)
        jf, se, adj, geo, jm, fm, pi = _make_line_batch(B=1, T=4, J=11, D=16)
        # Truncate parent_indices to length 5 (doesn't match joint_mask.sum()=11)
        bad_pi = [pi[0][:5]]
        with self.assertRaisesRegex(ValueError, "parent_indices.*length.*joint_mask"):
            pool(jf, se, adj, geo, jm, fm, parent_indices=bad_pi)

    def test_multi_level_anchor_override(self):
        # Pass pre-computed anchor_indices + coarse_mask instead of parent_indices
        # (level-2 use case from M1.3 VAE encoder).
        D, T, J = 16, 4, 6
        pool = DynamicGraphPool(d_model=D, max_coarse=3, local_radius=5)
        jf = torch.randn(1, T, J, D)
        se = torch.randn(1, J, D)
        adj = _adj_line(J).unsqueeze(0)
        geo = _geo_line(J).unsqueeze(0)
        jm = torch.ones(1, J, dtype=torch.bool)
        fm = torch.ones(1, T, dtype=torch.bool)
        # Custom anchors: pick joints 0, 2, 5
        anchor_indices = torch.tensor([[0, 2, 5]], dtype=torch.long)
        coarse_mask = torch.ones(1, 3, dtype=torch.bool)
        out = pool(jf, se, adj, geo, jm, fm,
                   anchor_indices=anchor_indices, coarse_mask=coarse_mask)
        self.assertEqual(out["pooled_features"].shape, (1, 2, 3, D))
        # Anchor indices preserved
        self.assertEqual(out["anchor_indices"].tolist(), [[0, 2, 5]])

    def test_both_anchor_sources_raises(self):
        pool = DynamicGraphPool(d_model=16, max_coarse=3)
        jf, se, adj, geo, jm, fm, pi = _make_line_batch(B=1, T=4, J=6, D=16)
        anchor_indices = torch.zeros(1, 3, dtype=torch.long)
        coarse_mask = torch.ones(1, 3, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "either parent_indices OR"):
            pool(jf, se, adj, geo, jm, fm,
                 parent_indices=pi, anchor_indices=anchor_indices, coarse_mask=coarse_mask)

    def test_no_anchor_source_raises(self):
        pool = DynamicGraphPool(d_model=16, max_coarse=3)
        jf, se, adj, geo, jm, fm, _ = _make_line_batch(B=1, T=4, J=6, D=16)
        with self.assertRaisesRegex(ValueError, "must pass either parent_indices OR"):
            pool(jf, se, adj, geo, jm, fm)

    def test_anchor_indices_out_of_range_raises(self):
        pool = DynamicGraphPool(d_model=16, max_coarse=3)
        jf, se, adj, geo, jm, fm, _ = _make_line_batch(B=1, T=4, J=6, D=16)
        # J=6, anchor 99 out of range
        anchor_indices = torch.tensor([[0, 2, 99]], dtype=torch.long)
        coarse_mask = torch.ones(1, 3, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "anchor_indices out of range"):
            pool(jf, se, adj, geo, jm, fm,
                 anchor_indices=anchor_indices, coarse_mask=coarse_mask)

    def test_joint_features_dtype_raises(self):
        pool = DynamicGraphPool(d_model=16, max_coarse=3)
        jf, se, adj, geo, jm, fm, pi = _make_line_batch(B=1, T=4, J=6, D=16)
        with self.assertRaisesRegex(ValueError, "joint_features.dtype must be float32"):
            pool(jf.double(), se, adj, geo, jm, fm, parent_indices=pi)

    def test_skeleton_embeddings_dtype_raises(self):
        pool = DynamicGraphPool(d_model=16, max_coarse=3)
        jf, se, adj, geo, jm, fm, pi = _make_line_batch(B=1, T=4, J=6, D=16)
        with self.assertRaisesRegex(ValueError, "skeleton_embeddings.dtype must be float32"):
            pool(jf, se.double(), adj, geo, jm, fm, parent_indices=pi)

    def test_adjacency_dtype_raises(self):
        pool = DynamicGraphPool(d_model=16, max_coarse=3)
        jf, se, adj, geo, jm, fm, pi = _make_line_batch(B=1, T=4, J=6, D=16)
        with self.assertRaisesRegex(ValueError, "adjacency.dtype must be float32"):
            pool(jf, se, adj.double(), geo, jm, fm, parent_indices=pi)

    def test_geodesic_dtype_raises(self):
        pool = DynamicGraphPool(d_model=16, max_coarse=3)
        jf, se, adj, geo, jm, fm, pi = _make_line_batch(B=1, T=4, J=6, D=16)
        with self.assertRaisesRegex(ValueError, "geodesic_dist.dtype must be float32"):
            pool(jf, se, adj, geo.double(), jm, fm, parent_indices=pi)

    def test_d_model_bool_raises(self):
        with self.assertRaisesRegex(TypeError, "d_model must be strict int"):
            DynamicGraphPool(d_model=True, max_coarse=4)

    def test_inactive_slot_outputs_zeroed(self):
        # Verify R12 #5: inactive pool slots produce zero pooled_features.
        D, T, J = 16, 4, 6
        pool = DynamicGraphPool(d_model=D, max_coarse=8, local_radius=5)  # C_max=8 > anchors
        jf, se, adj, geo, jm, fm, pi = _make_line_batch(B=1, T=T, J=J, D=D)
        out = pool(jf, se, adj, geo, jm, fm, parent_indices=pi)
        # Inactive coarse slots: pooled_features[:, :, inactive, :] should be 0
        inactive = ~out["pooled_mask"][0]
        if inactive.any():
            # Mass-normalized pool with assignment=0 → 0/eps ≈ 0
            inactive_features = out["pooled_features"][0, :, inactive, :]
            self.assertTrue(
                inactive_features.abs().max().item() < 1e-5,
                f"inactive slot features should be ~0, got max {inactive_features.abs().max().item()}",
            )
            # pooled_skeleton_embeddings should also be 0 at inactive slots
            inactive_skel = out["pooled_skeleton_embeddings"][0, inactive, :]
            self.assertTrue(
                inactive_skel.abs().max().item() < 1e-5,
                f"inactive slot skeleton embeddings should be 0",
            )

    def test_non_finite_hparam_raises(self):
        with self.assertRaisesRegex(ValueError, "locality_alpha must be finite"):
            DynamicGraphPool(d_model=16, max_coarse=4, locality_alpha=float("nan"))
        with self.assertRaisesRegex(ValueError, "temperature must be finite"):
            DynamicGraphPool(d_model=16, max_coarse=4, temperature=float("inf"))

    def test_module_dtype_raises_pool(self):
        pool = DynamicGraphPool(d_model=16, max_coarse=3)
        pool = pool.double()  # module fp64 vs input fp32
        jf, se, adj, geo, jm, fm, pi = _make_line_batch(B=1, T=4, J=6, D=16)
        # joint_features still fp32 → module dtype mismatch
        with self.assertRaisesRegex(ValueError, "module dtype.*!= float32"):
            pool(jf, se, adj, geo, jm, fm, parent_indices=pi)

    def test_padded_adjacency_pollution_masked(self):
        # R12 #20: edge added between valid joint and padded joint shouldn't
        # corrupt mincut. With masking, mincut_cut should be unchanged whether
        # padded edges are zero or polluted.
        D, T, J = 16, 4, 6
        pool = DynamicGraphPool(d_model=D, max_coarse=3, local_radius=5)
        # Use J_max=8, J_valid=6 (line). Polluted entry at adj[0, 0, 7].
        jf = torch.randn(1, T, 8, D)
        se = torch.randn(1, 8, D)
        adj_clean = torch.zeros(1, 8, 8)
        adj_clean[0, :6, :6] = _adj_line(6)
        geo = floyd_geo_from_adj_masked(adj_clean, mask_first_n=6)
        jm = torch.zeros(1, 8, dtype=torch.bool); jm[0, :6] = True
        fm = torch.ones(1, T, dtype=torch.bool)
        pi = [[-1, 0, 1, 2, 3, 4]]
        # Use the clean adj as baseline.
        out_clean = pool(jf, se, adj_clean, geo, jm, fm, parent_indices=pi)
        cut_clean = out_clean["aux_losses"]["mincut_cut"].item()
        # Polluted adj: add edge between valid joint 0 and padded joint 7.
        # Floyd consistency was checked using joint_mask, so we need geo to still match.
        adj_poll = adj_clean.clone()
        adj_poll[0, 0, 7] = adj_poll[0, 7, 0] = 1.0
        out_poll = pool(jf, se, adj_poll, geo, jm, fm, parent_indices=pi)
        cut_poll = out_poll["aux_losses"]["mincut_cut"].item()
        # With masking, both runs should give identical cut.
        self.assertAlmostEqual(cut_clean, cut_poll, places=5,
                              msg=f"polluted padded edge changed mincut_cut: "
                                  f"clean={cut_clean}, polluted={cut_poll}")

    def test_mixed_device_raises_pool(self):
        # device mismatch: skel_embed on meta, joint_features on cpu
        pool = DynamicGraphPool(d_model=16, max_coarse=3)
        jf, se, adj, geo, jm, fm, pi = _make_line_batch(B=1, T=4, J=6, D=16)
        se = se.to("meta")
        with self.assertRaisesRegex(ValueError, "skeleton_embeddings.device.*joint_features.device"):
            pool(jf, se, adj, geo, jm, fm, parent_indices=pi)

    def test_module_device_mismatch_raises(self):
        # Module on meta, inputs on cpu — should raise (we cannot actually move
        # an nn.Module trivially to meta for compute, but we can simulate by
        # moving the input tensors and module to different devices).
        pool = DynamicGraphPool(d_model=16, max_coarse=3)
        # Move module to meta (only affects param device, not actual compute)
        pool = pool.to("meta")
        jf, se, adj, geo, jm, fm, pi = _make_line_batch(B=1, T=4, J=6, D=16)
        # Inputs stay on cpu
        with self.assertRaisesRegex(ValueError, "module device meta.*input device cpu"):
            pool(jf, se, adj, geo, jm, fm, parent_indices=pi)

    def test_joint_features_pos_inf_raises(self):
        pool = DynamicGraphPool(d_model=16, max_coarse=3)
        jf, se, adj, geo, jm, fm, pi = _make_line_batch(B=1, T=4, J=6, D=16)
        jf[0, 0, 0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "joint_features contains NaN or Inf"):
            pool(jf, se, adj, geo, jm, fm, parent_indices=pi)

    def test_skeleton_embeddings_pos_inf_raises(self):
        pool = DynamicGraphPool(d_model=16, max_coarse=3)
        jf, se, adj, geo, jm, fm, pi = _make_line_batch(B=1, T=4, J=6, D=16)
        se[0, 0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "skeleton_embeddings contains NaN or Inf"):
            pool(jf, se, adj, geo, jm, fm, parent_indices=pi)

    def test_geodesic_inconsistent_with_adjacency_raises_pool(self):
        # R12 #19: geo=zeros + adj=line passes all guards but is wrong.
        pool = DynamicGraphPool(d_model=16, max_coarse=3)
        jf, se, adj, geo, jm, fm, pi = _make_line_batch(B=1, T=4, J=6, D=16)
        # Replace correct geo with all-zeros — passes finite/nonneg/symm/diag but wrong.
        bad_geo = torch.zeros_like(geo)
        with self.assertRaisesRegex(ValueError, "inconsistent with.*shortest-path"):
            pool(jf, se, adj, bad_geo, jm, fm, parent_indices=pi)

    def test_mincut_ortho_per_sample_C_valid(self):
        # Verify R12 #1 fix: mincut ortho uses per-sample C_valid, not C_max.
        # Build a sample where C_valid=2, C_max=8, and assignment is perfectly
        # orthogonal: half joints to anchor 0, half to anchor 1.
        D = 16
        pool = DynamicGraphPool(d_model=D, max_coarse=8, local_radius=5,
                                max_chain_chunk_len=10)
        # Line skeleton with 4 joints; anchors will be {root=0, leaf=3} (2 anchors).
        line_parents = [-1, 0, 1, 2]
        jf = torch.randn(1, 4, 4, D)
        se = torch.randn(1, 4, D)
        adj = _adj_line(4).unsqueeze(0)
        geo = _geo_line(4).unsqueeze(0)
        jm = torch.ones(1, 4, dtype=torch.bool)
        fm = torch.ones(1, 4, dtype=torch.bool)
        out = pool(jf, se, adj, geo, jm, fm, parent_indices=[line_parents])
        # With C_valid=2 (not C_max=8), ortho loss should be modest. The whole
        # point: it should NOT scale with C_max. Just check it's finite and
        # roughly the same magnitude regardless of C_max.
        ortho_at_8 = out["aux_losses"]["mincut_ortho"].item()
        # Now redo with max_coarse=4 — fewer pad slots
        pool2 = DynamicGraphPool(d_model=D, max_coarse=4, local_radius=5,
                                 max_chain_chunk_len=10)
        # Manually copy weights so the bilinear scoring matches
        pool2.q_proj.weight.data = pool.q_proj.weight.data.clone()
        pool2.q_proj.bias.data = pool.q_proj.bias.data.clone()
        pool2.k_proj.weight.data = pool.k_proj.weight.data.clone()
        pool2.k_proj.bias.data = pool.k_proj.bias.data.clone()
        out2 = pool2(jf, se, adj, geo, jm, fm, parent_indices=[line_parents])
        ortho_at_4 = out2["aux_losses"]["mincut_ortho"].item()
        # The two should be very close (within reasonable tol) because both
        # restrict to valid_pair entries with C_valid=2 target.
        self.assertAlmostEqual(ortho_at_8, ortho_at_4, places=4,
                              msg=f"mincut_ortho varies with C_max: "
                                  f"C_max=8 → {ortho_at_8}, C_max=4 → {ortho_at_4}")

    def test_geodesic_negative_finite_raises_pool(self):
        pool = DynamicGraphPool(d_model=16, max_coarse=3)
        jf, se, adj, geo, jm, fm, pi = _make_line_batch(B=1, T=4, J=6, D=16)
        geo[0, 0, 1] = -1.0  # finite negative — invalid Floyd output
        geo[0, 1, 0] = -1.0  # keep symmetric
        with self.assertRaisesRegex(ValueError, "geodesic_dist has negative finite"):
            pool(jf, se, adj, geo, jm, fm, parent_indices=pi)

    def test_joint_features_nan_raises(self):
        pool = DynamicGraphPool(d_model=16, max_coarse=3)
        jf, se, adj, geo, jm, fm, pi = _make_line_batch(B=1, T=4, J=6, D=16)
        jf[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "joint_features contains NaN or Inf"):
            pool(jf, se, adj, geo, jm, fm, parent_indices=pi)

    def test_skeleton_embeddings_nan_raises(self):
        pool = DynamicGraphPool(d_model=16, max_coarse=3)
        jf, se, adj, geo, jm, fm, pi = _make_line_batch(B=1, T=4, J=6, D=16)
        se[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "skeleton_embeddings contains NaN or Inf"):
            pool(jf, se, adj, geo, jm, fm, parent_indices=pi)

    def test_adjacency_asymmetric_raises_pool(self):
        pool = DynamicGraphPool(d_model=16, max_coarse=3)
        jf, se, adj, geo, jm, fm, pi = _make_line_batch(B=1, T=4, J=6, D=16)
        adj[0, 0, 1] = 0.0  # break symmetry (was 1)
        with self.assertRaisesRegex(ValueError, "adjacency is not symmetric"):
            pool(jf, se, adj, geo, jm, fm, parent_indices=pi)

    def test_adjacency_self_loop_raises_pool(self):
        pool = DynamicGraphPool(d_model=16, max_coarse=3)
        jf, se, adj, geo, jm, fm, pi = _make_line_batch(B=1, T=4, J=6, D=16)
        adj[0, 2, 2] = 1.0  # self-loop
        with self.assertRaisesRegex(ValueError, "adjacency has non-zero diagonal"):
            pool(jf, se, adj, geo, jm, fm, parent_indices=pi)

    def test_geodesic_nan_raises_pool(self):
        pool = DynamicGraphPool(d_model=16, max_coarse=3)
        jf, se, adj, geo, jm, fm, pi = _make_line_batch(B=1, T=4, J=6, D=16)
        geo[0, 0, 1] = float("nan")
        with self.assertRaisesRegex(ValueError, "geodesic_dist contains NaN"):
            pool(jf, se, adj, geo, jm, fm, parent_indices=pi)

    def test_geodesic_neg_inf_raises_pool(self):
        pool = DynamicGraphPool(d_model=16, max_coarse=3)
        jf, se, adj, geo, jm, fm, pi = _make_line_batch(B=1, T=4, J=6, D=16)
        geo[0, 0, 1] = float("-inf")
        with self.assertRaisesRegex(ValueError, "geodesic_dist contains -Inf"):
            pool(jf, se, adj, geo, jm, fm, parent_indices=pi)

    def test_geodesic_asymmetric_raises_pool(self):
        pool = DynamicGraphPool(d_model=16, max_coarse=3)
        jf, se, adj, geo, jm, fm, pi = _make_line_batch(B=1, T=4, J=6, D=16)
        geo[0, 0, 1] = 1.0
        geo[0, 1, 0] = 2.0  # asymmetric
        with self.assertRaisesRegex(ValueError, "geodesic_dist is not symmetric"):
            pool(jf, se, adj, geo, jm, fm, parent_indices=pi)

    def test_geodesic_nonzero_diagonal_raises_pool(self):
        pool = DynamicGraphPool(d_model=16, max_coarse=3)
        jf, se, adj, geo, jm, fm, pi = _make_line_batch(B=1, T=4, J=6, D=16)
        geo[0, 2, 2] = 1.0  # i→i distance should be 0
        with self.assertRaisesRegex(ValueError, "non-zero diagonal at valid nodes"):
            pool(jf, se, adj, geo, jm, fm, parent_indices=pi)

    def test_adjacency_negative_raises_pool(self):
        pool = DynamicGraphPool(d_model=16, max_coarse=3)
        jf, se, adj, geo, jm, fm, pi = _make_line_batch(B=1, T=4, J=6, D=16)
        adj[0, 2, 3] = -1.0
        adj[0, 3, 2] = -1.0  # keep symmetric
        with self.assertRaisesRegex(ValueError, "adjacency contains negative"):
            pool(jf, se, adj, geo, jm, fm, parent_indices=pi)

    def test_adjacency_above_one_raises_pool(self):
        pool = DynamicGraphPool(d_model=16, max_coarse=3)
        jf, se, adj, geo, jm, fm, pi = _make_line_batch(B=1, T=4, J=6, D=16)
        adj[0, 2, 3] = 2.0
        adj[0, 3, 2] = 2.0  # keep symmetric
        with self.assertRaisesRegex(ValueError, "adjacency contains values > 1.0"):
            pool(jf, se, adj, geo, jm, fm, parent_indices=pi)

    def test_partial_anchor_args_raises(self):
        # Only anchor_indices, no coarse_mask
        pool = DynamicGraphPool(d_model=16, max_coarse=3)
        jf, se, adj, geo, jm, fm, pi = _make_line_batch(B=1, T=4, J=6, D=16)
        with self.assertRaisesRegex(ValueError, "anchor_indices.*provided alone"):
            pool(jf, se, adj, geo, jm, fm,
                 anchor_indices=torch.zeros(1, 3, dtype=torch.long))
        # Only coarse_mask, no anchor_indices
        with self.assertRaisesRegex(ValueError, "coarse_mask.*provided alone"):
            pool(jf, se, adj, geo, jm, fm,
                 coarse_mask=torch.ones(1, 3, dtype=torch.bool))

    def test_inactive_slot_anchor_value_validated(self):
        # Active slots OK; inactive slot has 9999 instead of -1 sentinel.
        # The padded-sentinel check (R12 #23) catches this before range check.
        pool = DynamicGraphPool(d_model=16, max_coarse=4)
        jf, se, adj, geo, jm, fm, _ = _make_line_batch(B=1, T=4, J=6, D=16)
        anchor_indices = torch.tensor([[0, 2, -1, 9999]], dtype=torch.long)
        coarse_mask = torch.tensor([[True, True, False, False]])
        # Either sentinel or range check fires — both are valid R12 failures.
        with self.assertRaisesRegex(ValueError, "must be -1 sentinel|anchor_indices out of range"):
            pool(jf, se, adj, geo, jm, fm,
                 anchor_indices=anchor_indices, coarse_mask=coarse_mask)

    def test_joint_mask_hole_raises(self):
        # joint_mask = [T,T,T,F,T,T] — hole at index 3.
        # Compact parent_indices=[-1,0,1,2,3] (length 5 matches True count) would
        # alias compact joint 4 to raw joint 4 (skipping the hole at 3) — wrong.
        pool = DynamicGraphPool(d_model=16, max_coarse=3)
        jf = torch.randn(1, 4, 6, 16)
        se = torch.randn(1, 6, 16)
        adj = torch.zeros(1, 6, 6)
        # Set up some adj on indices 0,1,2,4,5 (skipping 3)
        for u, v in [(0, 1), (1, 2), (2, 4), (4, 5)]:
            adj[0, u, v] = adj[0, v, u] = 1.0
        geo = torch.zeros(1, 6, 6)
        # Geo from adj via BFS — but we just need the test to fail BEFORE compute,
        # so any consistent geo will do. Just use zeros (won't be reached).
        jm = torch.tensor([[True, True, True, False, True, True]])
        fm = torch.ones(1, 4, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "joint_mask is not a contiguous True-prefix"):
            pool(jf, se, adj, geo, jm, fm, parent_indices=[[-1, 0, 1, 2, 3]])

    def test_joint_mask_true_in_padded_raises(self):
        # joint_mask = [T,T,T,T,F,T] — True at idx 5, sum=5 (so j_valid=5 but mask[5:6]=True)
        pool = DynamicGraphPool(d_model=16, max_coarse=3)
        jf = torch.randn(1, 4, 6, 16)
        se = torch.randn(1, 6, 16)
        adj = torch.zeros(1, 6, 6)
        for u, v in [(0, 1), (1, 2), (2, 3), (3, 5)]:
            adj[0, u, v] = adj[0, v, u] = 1.0
        geo = torch.zeros(1, 6, 6)
        jm = torch.tensor([[True, True, True, True, False, True]])
        fm = torch.ones(1, 4, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "joint_mask is not a contiguous True-prefix"):
            pool(jf, se, adj, geo, jm, fm, parent_indices=[[-1, 0, 1, 2, 3]])

    def test_all_false_joint_mask_raises(self):
        pool = DynamicGraphPool(d_model=16, max_coarse=3)
        # Sample 0 has joint_mask all False → no valid joints.
        # Build a fake batch where joint_mask all False but parents=[] empty list
        # (len(parents)=0 == joint_mask.sum()=0 — passes length check).
        jf = torch.randn(1, 4, 6, 16)
        se = torch.randn(1, 6, 16)
        adj = torch.zeros(1, 6, 6)
        geo = torch.zeros(1, 6, 6)
        jm = torch.zeros(1, 6, dtype=torch.bool)  # all False
        fm = torch.ones(1, 4, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "all-False.*joint_mask"):
            pool(jf, se, adj, geo, jm, fm, parent_indices=[[]])

    def test_anchor_indices_duplicate_active_raises(self):
        pool = DynamicGraphPool(d_model=16, max_coarse=4)
        jf, se, adj, geo, jm, fm, _ = _make_line_batch(B=1, T=4, J=6, D=16)
        # Duplicate anchor 0 in active slots
        anchor_indices = torch.tensor([[0, 0, 2, 3]], dtype=torch.long)
        coarse_mask = torch.ones(1, 4, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "strictly ascending"):
            pool(jf, se, adj, geo, jm, fm,
                 anchor_indices=anchor_indices, coarse_mask=coarse_mask)

    def test_anchor_indices_unsorted_active_raises(self):
        pool = DynamicGraphPool(d_model=16, max_coarse=4)
        jf, se, adj, geo, jm, fm, _ = _make_line_batch(B=1, T=4, J=6, D=16)
        # Active anchors unsorted: 2, 0, 4, 5 (2 > 0 breaks ascending)
        anchor_indices = torch.tensor([[2, 0, 4, 5]], dtype=torch.long)
        coarse_mask = torch.ones(1, 4, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "strictly ascending"):
            pool(jf, se, adj, geo, jm, fm,
                 anchor_indices=anchor_indices, coarse_mask=coarse_mask)

    def test_padded_anchor_not_minus_one_raises(self):
        # R12 #23: padded slot (coarse_mask=False) must be -1 sentinel.
        pool = DynamicGraphPool(d_model=16, max_coarse=4)
        jf, se, adj, geo, jm, fm, _ = _make_line_batch(B=1, T=4, J=6, D=16)
        # Slot 3 is padded (coarse_mask=False) but has anchor_id=5 (not -1)
        anchor_indices = torch.tensor([[0, 2, 5, 5]], dtype=torch.long)
        coarse_mask = torch.tensor([[True, True, True, False]])
        with self.assertRaisesRegex(ValueError, "must be -1 sentinel"):
            pool(jf, se, adj, geo, jm, fm,
                 anchor_indices=anchor_indices, coarse_mask=coarse_mask)

    def test_coarse_mask_hole_raises(self):
        # coarse_mask = [F, T, T, T] — active slots not a contiguous prefix.
        # Rule-based path always writes True from slot 0 → override must match.
        pool = DynamicGraphPool(d_model=16, max_coarse=4)
        jf, se, adj, geo, jm, fm, _ = _make_line_batch(B=1, T=4, J=6, D=16)
        anchor_indices = torch.tensor([[-1, 0, 2, 5]], dtype=torch.long)
        coarse_mask = torch.tensor([[False, True, True, True]])
        with self.assertRaisesRegex(ValueError, "not a contiguous True-prefix"):
            pool(jf, se, adj, geo, jm, fm,
                 anchor_indices=anchor_indices, coarse_mask=coarse_mask)

    def test_coarse_mask_true_in_padded_raises(self):
        # coarse_mask = [T, T, F, T] — sum is 3 but True in padded region (idx 3)
        pool = DynamicGraphPool(d_model=16, max_coarse=4)
        jf, se, adj, geo, jm, fm, _ = _make_line_batch(B=1, T=4, J=6, D=16)
        anchor_indices = torch.tensor([[0, 2, -1, 5]], dtype=torch.long)
        coarse_mask = torch.tensor([[True, True, False, True]])
        with self.assertRaisesRegex(ValueError, "not a contiguous True-prefix"):
            pool(jf, se, adj, geo, jm, fm,
                 anchor_indices=anchor_indices, coarse_mask=coarse_mask)

    def test_anchor_indices_missing_root_raises(self):
        # Rule-based path always includes joint 0 (root) as first anchor.
        # Override must match.
        pool = DynamicGraphPool(d_model=16, max_coarse=3)
        jf, se, adj, geo, jm, fm, _ = _make_line_batch(B=1, T=4, J=6, D=16)
        # No anchor 0 — starts at 1
        anchor_indices = torch.tensor([[1, 2, 5]], dtype=torch.long)
        coarse_mask = torch.ones(1, 3, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "missing root.*joint 0"):
            pool(jf, se, adj, geo, jm, fm,
                 anchor_indices=anchor_indices, coarse_mask=coarse_mask)

    def test_anchor_zero_active_slots_raises(self):
        pool = DynamicGraphPool(d_model=16, max_coarse=4)
        jf, se, adj, geo, jm, fm, _ = _make_line_batch(B=1, T=4, J=6, D=16)
        # All coarse_mask=False → no active anchors
        anchor_indices = torch.full((1, 4), -1, dtype=torch.long)
        coarse_mask = torch.zeros(1, 4, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "zero active coarse slots"):
            pool(jf, se, adj, geo, jm, fm,
                 anchor_indices=anchor_indices, coarse_mask=coarse_mask)

    def test_odd_temporal_stride_raises(self):
        # T=5, stride=2 → T % stride != 0
        pool = DynamicGraphPool(d_model=16, max_coarse=3, temporal_stride=2)
        jf = torch.randn(1, 5, 6, 16)
        se = torch.randn(1, 6, 16)
        adj = _adj_line(6).unsqueeze(0)
        geo = _geo_line(6).unsqueeze(0)
        jm = torch.ones(1, 6, dtype=torch.bool)
        fm = torch.ones(1, 5, dtype=torch.bool)
        pi = [[-1, 0, 1, 2, 3, 4]]
        with self.assertRaisesRegex(ValueError, "T=5 must be divisible by temporal_stride=2"):
            pool(jf, se, adj, geo, jm, fm, parent_indices=pi)

    def test_anchor_points_to_padded_joint_raises(self):
        pool = DynamicGraphPool(d_model=16, max_coarse=3)
        # J_max=10, J_valid=6 (line)
        jf = torch.randn(1, 4, 10, 16)
        se = torch.randn(1, 10, 16)
        adj = torch.zeros(1, 10, 10)
        adj[0, :6, :6] = _adj_line(6)
        geo = torch.zeros(1, 10, 10)
        geo[0, :6, :6] = _geo_line(6)
        for i in range(6, 10):
            for j in range(10):
                if i != j:
                    geo[0, i, j] = 100.0
                    geo[0, j, i] = 100.0
        jm = torch.zeros(1, 10, dtype=torch.bool)
        jm[0, :6] = True
        fm = torch.ones(1, 4, dtype=torch.bool)
        # Anchor points to padded joint 7
        anchor_indices = torch.tensor([[0, 2, 7]], dtype=torch.long)
        coarse_mask = torch.ones(1, 3, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "anchor_indices point to padded"):
            pool(jf, se, adj, geo, jm, fm,
                 anchor_indices=anchor_indices, coarse_mask=coarse_mask)

    def test_wrong_input_shape_raises(self):
        pool = DynamicGraphPool(d_model=16, max_coarse=4)
        # Wrong d_model
        with self.assertRaisesRegex(ValueError, "joint_features must be"):
            pool(
                torch.randn(1, 4, 6, 8),  # D=8 ≠ 16
                torch.randn(1, 6, 16), _adj_line(6).unsqueeze(0),
                _geo_line(6).unsqueeze(0), torch.ones(1, 6, dtype=torch.bool),
                torch.ones(1, 4, dtype=torch.bool), [[-1, 0, 1, 2, 3, 4]],
            )
        # parent_indices wrong length
        with self.assertRaisesRegex(ValueError, "parent_indices must be list of length"):
            pool(
                torch.randn(2, 4, 6, 16), torch.randn(2, 6, 16),
                _adj_line(6).expand(2, 6, 6).clone(), _geo_line(6).expand(2, 6, 6).clone(),
                torch.ones(2, 6, dtype=torch.bool), torch.ones(2, 4, dtype=torch.bool),
                [[-1, 0, 1, 2, 3, 4]],  # length 1, expected 2
            )


class DynamicGraphPoolRealBatchTest(unittest.TestCase):
    def test_bat_skeleton_smoke(self):
        """Sanity check end-to-end forward on Bat skeleton (J=48)."""
        npz_path = (
            Path(__file__).resolve().parents[1]
            / "data" / "cs_sparse2full_tgt" / "skeletons" / "Bat.npz"
        )
        if not npz_path.exists():
            self.skipTest(f"Bat.npz not found")
        d = np.load(npz_path, allow_pickle=True)
        parents = d["parent_indices"].tolist()
        J = len(parents)
        D = 16
        # Bat (J=48) produces ~26 anchors with default chunk_len=5; use 32 to cover.
        pool = DynamicGraphPool(d_model=D, max_coarse=32, local_radius=6)

        joint_features = torch.randn(1, 8, J, D)
        skeleton_embeddings = torch.randn(1, J, D)
        adj = torch.from_numpy(d["adjacency"]).float().unsqueeze(0)
        geo = torch.from_numpy(d["geodesic_dist"]).float().unsqueeze(0)
        joint_mask = torch.ones(1, J, dtype=torch.bool)
        frame_mask = torch.ones(1, 8, dtype=torch.bool)

        out = pool(joint_features, skeleton_embeddings, adj, geo,
                   joint_mask, frame_mask, [parents])
        # Sanity: anchor count > 0, pooled_features non-NaN
        self.assertGreater(out["pooled_mask"].sum().item(), 0)
        self.assertTrue(torch.isfinite(out["pooled_features"]).all())
        self.assertTrue(torch.isfinite(out["aux_losses"]["mincut"]).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
