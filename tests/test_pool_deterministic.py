"""Unit tests for src/models/graph_salad/pool_deterministic.py — M1.2 step 3."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.graph_salad.pool_deterministic import DeterministicGraphPool


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


def _make_line_batch(B=2, T=8, J=11, D=16):
    line_parents = [-1] + list(range(J - 1))
    parent_indices = [line_parents for _ in range(B)]
    joint_features = torch.randn(B, T, J, D)
    skeleton_embeddings = torch.randn(B, J, D)
    adj = _adj_line(J).expand(B, J, J).clone()
    geo = _geo_line(J).expand(B, J, J).clone()
    joint_mask = torch.ones(B, J, dtype=torch.bool)
    frame_mask = torch.ones(B, T, dtype=torch.bool)
    return joint_features, skeleton_embeddings, adj, geo, joint_mask, frame_mask, parent_indices


class DeterministicGraphPoolTests(unittest.TestCase):

    def test_no_learnable_params(self):
        pool = DeterministicGraphPool(d_model=16, max_coarse=4)
        self.assertEqual(len(list(pool.parameters())), 0,
                        f"DeterministicGraphPool should have 0 params, "
                        f"got {[p.shape for p in pool.parameters()]}")
        self.assertEqual(len(pool.state_dict()), 0,
                        f"state_dict should be empty, got {list(pool.state_dict())}")

    def test_forward_shapes(self):
        D, T, J = 16, 8, 11
        pool = DeterministicGraphPool(d_model=D, max_coarse=8, local_radius=5)
        jf, se, adj, geo, jm, fm, pi = _make_line_batch(B=2, T=T, J=J, D=D)
        out = pool(jf, se, adj, geo, jm, fm, parent_indices=pi)
        B = 2
        T_lat = T // 2
        self.assertEqual(out["pooled_features"].shape, (B, T_lat, 8, D))
        self.assertEqual(out["assignment"].shape, (B, J, 8))
        self.assertEqual(out["pooled_adjacency"].shape, (B, 8, 8))

    def test_assignment_is_one_hot(self):
        D = 16
        pool = DeterministicGraphPool(d_model=D, max_coarse=8, local_radius=5)
        jf, se, adj, geo, jm, fm, pi = _make_line_batch(B=1, T=4, J=11, D=D)
        out = pool(jf, se, adj, geo, jm, fm, parent_indices=pi)
        P = out["assignment"][0]  # [J, C]
        # Each valid joint's row should sum to 1 (one-hot)
        row_sums = P.sum(dim=-1)
        valid_rows = row_sums[jm[0]]
        self.assertTrue(torch.allclose(valid_rows, torch.ones_like(valid_rows)))
        # Each value should be 0 or 1
        self.assertTrue(((P == 0) | (P == 1)).all())

    def test_assignment_picks_nearest_anchor(self):
        # Line J=5 with anchors {0, 4} (root + leaf, no chain chunking).
        # Joint 0 → anchor 0 (geo 0), joint 1 → anchor 0 (geo 1), joint 2 → anchor 0 (geo 2, ties to anchor 4 geo 2 → lower idx wins),
        # joint 3 → anchor 4 (geo 1), joint 4 → anchor 4 (geo 0).
        D = 16
        pool = DeterministicGraphPool(d_model=D, max_coarse=3, local_radius=4, max_chain_chunk_len=100)
        line_parents = [[-1, 0, 1, 2, 3]]
        jf = torch.randn(1, 4, 5, D)
        se = torch.randn(1, 5, D)
        adj = _adj_line(5).unsqueeze(0)
        geo = _geo_line(5).unsqueeze(0)
        jm = torch.ones(1, 5, dtype=torch.bool)
        fm = torch.ones(1, 4, dtype=torch.bool)
        out = pool(jf, se, adj, geo, jm, fm, parent_indices=line_parents)
        # Anchors should be [0, 4, -1]
        anchors = out["anchor_indices"][0].tolist()
        self.assertEqual(anchors[:2], [0, 4])
        # Hard assignment per joint (compact column index, not raw anchor id)
        hard = out["hard_assignment"][0].tolist()
        # joint 0 → col 0 (anchor 0), joint 1 → col 0, joint 2 → col 0 (tie → lower),
        # joint 3 → col 1 (anchor 4), joint 4 → col 1
        self.assertEqual(hard, [0, 0, 0, 1, 1])

    def test_entropy_zero(self):
        # One-hot assignment has exactly 0 entropy.
        D = 16
        pool = DeterministicGraphPool(d_model=D, max_coarse=4, local_radius=5)
        jf, se, adj, geo, jm, fm, pi = _make_line_batch(B=1, T=4, J=6, D=D)
        out = pool(jf, se, adj, geo, jm, fm, parent_indices=pi)
        self.assertEqual(out["aux_losses"]["entropy"].item(), 0.0)
        # MinCut N/A → all three keys (mincut/mincut_cut/mincut_ortho) zero
        self.assertEqual(out["aux_losses"]["mincut"].item(), 0.0)
        self.assertEqual(out["aux_losses"]["mincut_cut"].item(), 0.0)
        self.assertEqual(out["aux_losses"]["mincut_ortho"].item(), 0.0)

    def test_aux_losses_schema_parity_with_dynamic(self):
        # All 5 aux_losses keys must match DynamicGraphPool for M1.3 VAE swap.
        from src.models.graph_salad.pool_dynamic import DynamicGraphPool
        D = 16
        det = DeterministicGraphPool(d_model=D, max_coarse=4, local_radius=5)
        dyn = DynamicGraphPool(d_model=D, max_coarse=4, local_radius=5)
        jf, se, adj, geo, jm, fm, pi = _make_line_batch(B=1, T=4, J=6, D=D)
        det_out = det(jf, se, adj, geo, jm, fm, parent_indices=pi)
        dyn_out = dyn(jf, se, adj, geo, jm, fm, parent_indices=pi)
        self.assertEqual(set(det_out["aux_losses"]), set(dyn_out["aux_losses"]),
                        f"aux_losses key mismatch: det={set(det_out['aux_losses'])}, dyn={set(dyn_out['aux_losses'])}")
        # Also top-level output dict
        self.assertEqual(set(det_out), set(dyn_out),
                        f"output dict key mismatch: det={set(det_out)}, dyn={set(dyn_out)}")

    def test_aux_losses_dtype_matches_input(self):
        D = 16
        pool = DeterministicGraphPool(d_model=D, max_coarse=4, local_radius=5)
        jf, se, adj, geo, jm, fm, pi = _make_line_batch(B=1, T=4, J=6, D=D)
        out = pool(jf, se, adj, geo, jm, fm, parent_indices=pi)
        # All aux_losses tensors should be float32 (matches P/input dtype)
        for k, v in out["aux_losses"].items():
            self.assertEqual(v.dtype, torch.float32, f"aux_losses[{k}] dtype = {v.dtype}, expected float32")

    def test_padded_geodesic_finite_accepted(self):
        # Padded entries can have any finite value (parity with pool_dynamic).
        # Verify that geo > J-1 in PADDED entries is accepted, not raised.
        D = 16
        pool = DeterministicGraphPool(d_model=D, max_coarse=4, local_radius=5)
        # B=1 with J_max=10, J_valid=6.
        jf = torch.randn(1, 4, 10, D)
        se = torch.randn(1, 10, D)
        adj = torch.zeros(1, 10, 10)
        adj[0, :6, :6] = _adj_line(6)
        geo = torch.zeros(1, 10, 10)
        geo[0, :6, :6] = _geo_line(6)
        # Padded entries set to 999.0 — would violate hop-bound if globally checked.
        for i in range(10):
            for j in range(10):
                if i >= 6 or j >= 6:
                    if i != j:
                        geo[0, i, j] = 999.0
                        geo[0, j, i] = 999.0
        jm = torch.zeros(1, 10, dtype=torch.bool)
        jm[0, :6] = True
        fm = torch.ones(1, 4, dtype=torch.bool)
        pi = [[-1, 0, 1, 2, 3, 4]]
        # Must NOT raise (padded entries should be ignored)
        out = pool(jf, se, adj, geo, jm, fm, parent_indices=pi)
        self.assertTrue(torch.isfinite(out["pooled_features"]).all())

    def test_pooled_adjacency_binary_symmetric(self):
        D = 16
        pool = DeterministicGraphPool(d_model=D, max_coarse=4, local_radius=5)
        jf, se, adj, geo, jm, fm, pi = _make_line_batch(B=1, T=4, J=6, D=D)
        out = pool(jf, se, adj, geo, jm, fm, parent_indices=pi)
        adj_out = out["pooled_adjacency"][0]
        self.assertTrue(((adj_out == 0) | (adj_out == 1)).all())
        self.assertTrue(torch.equal(adj_out, adj_out.T))
        self.assertEqual(adj_out.diag().sum().item(), 0.0)

    # --- Validation contract tests (sample selection from pool_dynamic's 23 R12) ---

    def test_constructor_strict_int(self):
        with self.assertRaisesRegex(TypeError, "d_model must be strict int"):
            DeterministicGraphPool(d_model=16.0, max_coarse=4)
        with self.assertRaisesRegex(TypeError, "local_radius must be strict int"):
            DeterministicGraphPool(d_model=16, max_coarse=4, local_radius=True)

    def test_joint_features_nan_raises(self):
        pool = DeterministicGraphPool(d_model=16, max_coarse=4, local_radius=5)
        jf, se, adj, geo, jm, fm, pi = _make_line_batch(B=1, T=4, J=6, D=16)
        jf[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "joint_features contains NaN or Inf"):
            pool(jf, se, adj, geo, jm, fm, parent_indices=pi)

    def test_adjacency_non_symmetric_raises(self):
        pool = DeterministicGraphPool(d_model=16, max_coarse=4, local_radius=5)
        jf, se, adj, geo, jm, fm, pi = _make_line_batch(B=1, T=4, J=6, D=16)
        adj[0, 0, 1] = 0.0
        with self.assertRaisesRegex(ValueError, "adjacency is not symmetric"):
            pool(jf, se, adj, geo, jm, fm, parent_indices=pi)

    def test_geodesic_inconsistent_with_adjacency_raises(self):
        pool = DeterministicGraphPool(d_model=16, max_coarse=4, local_radius=5)
        jf, se, adj, geo, jm, fm, pi = _make_line_batch(B=1, T=4, J=6, D=16)
        bad_geo = torch.zeros_like(geo)
        with self.assertRaisesRegex(ValueError, "inconsistent with.*shortest-path"):
            pool(jf, se, adj, bad_geo, jm, fm, parent_indices=pi)

    def test_no_candidate_raises(self):
        pool = DeterministicGraphPool(d_model=16, max_coarse=4, local_radius=1,
                                      max_chain_chunk_len=100)
        jf, se, adj, geo, jm, fm, pi = _make_line_batch(B=1, T=4, J=11, D=16)
        with self.assertRaisesRegex(ValueError, "no candidate anchor"):
            pool(jf, se, adj, geo, jm, fm, parent_indices=pi)

    def test_anchor_overflow_raises(self):
        pool = DeterministicGraphPool(d_model=16, max_coarse=1)
        jf, se, adj, geo, jm, fm, pi = _make_line_batch(B=1, T=4, J=11, D=16)
        with self.assertRaisesRegex(ValueError, "anchors >.*max_coarse"):
            pool(jf, se, adj, geo, jm, fm, parent_indices=pi)

    def test_multi_level_anchor_override(self):
        D = 16
        pool = DeterministicGraphPool(d_model=D, max_coarse=3, local_radius=5)
        jf = torch.randn(1, 4, 6, D)
        se = torch.randn(1, 6, D)
        adj = _adj_line(6).unsqueeze(0)
        geo = _geo_line(6).unsqueeze(0)
        jm = torch.ones(1, 6, dtype=torch.bool)
        fm = torch.ones(1, 4, dtype=torch.bool)
        anchor_indices = torch.tensor([[0, 2, 5]], dtype=torch.long)
        coarse_mask = torch.ones(1, 3, dtype=torch.bool)
        out = pool(jf, se, adj, geo, jm, fm,
                   anchor_indices=anchor_indices, coarse_mask=coarse_mask)
        self.assertEqual(out["pooled_features"].shape, (1, 2, 3, D))


class DeterministicGraphPoolRealBatchTest(unittest.TestCase):
    def test_bat_skeleton_smoke(self):
        npz_path = (
            Path(__file__).resolve().parents[1]
            / "data" / "cs_sparse2full_tgt" / "skeletons" / "Bat.npz"
        )
        if not npz_path.exists():
            self.skipTest("Bat.npz not found")
        d = np.load(npz_path, allow_pickle=True)
        parents = d["parent_indices"].tolist()
        J = len(parents)
        D = 16
        pool = DeterministicGraphPool(d_model=D, max_coarse=32, local_radius=6)
        joint_features = torch.randn(1, 8, J, D)
        skeleton_embeddings = torch.randn(1, J, D)
        adj = torch.from_numpy(d["adjacency"]).float().unsqueeze(0)
        geo = torch.from_numpy(d["geodesic_dist"]).float().unsqueeze(0)
        joint_mask = torch.ones(1, J, dtype=torch.bool)
        frame_mask = torch.ones(1, 8, dtype=torch.bool)
        out = pool(joint_features, skeleton_embeddings, adj, geo,
                   joint_mask, frame_mask, [parents])
        self.assertGreater(out["pooled_mask"].sum().item(), 0)
        self.assertTrue(torch.isfinite(out["pooled_features"]).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
