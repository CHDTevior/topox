"""Unit tests for src/models/graph_salad/attention.py — M1.2 step 1.

Run:
    python tests/test_graph_attention.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.graph_salad.attention import GraphAttentionBlock


def _adj_line(N: int) -> torch.Tensor:
    A = torch.zeros(N, N)
    for j in range(1, N):
        A[j, j - 1] = 1.0
        A[j - 1, j] = 1.0
    return A


def _geo_from_adj(adj: torch.Tensor) -> torch.Tensor:
    # Convenience: small-N geodesic via Floyd on a single sample
    N = adj.shape[0]
    INF = float("inf")
    d = torch.full((N, N), INF)
    for i in range(N):
        d[i, i] = 0.0
    d = torch.where(adj > 0, torch.ones_like(adj), d)
    for k in range(N):
        d = torch.minimum(d, d[:, k:k+1] + d[k:k+1, :])
    return d


class GraphAttentionBlockTests(unittest.TestCase):

    def test_forward_shape(self):
        block = GraphAttentionBlock(d_model=32, n_heads=4, d_ff=64, dropout=0.0)
        B, N = 2, 6
        x = torch.randn(B, N, 32)
        adj = _adj_line(N).expand(B, N, N).clone()
        geo = _geo_from_adj(adj[0]).expand(B, N, N).clone()
        mask = torch.ones(B, N, dtype=torch.bool)
        y = block(x, adj, geo, mask)
        self.assertEqual(y.shape, (B, N, 32))

    def test_invalid_dim_raises(self):
        with self.assertRaisesRegex(ValueError, "divisible by"):
            GraphAttentionBlock(d_model=30, n_heads=4, d_ff=64)
        with self.assertRaisesRegex(ValueError, "> 0"):
            GraphAttentionBlock(d_model=32, n_heads=0, d_ff=64)
        with self.assertRaisesRegex(ValueError, "d_ff must be > 0"):
            GraphAttentionBlock(d_model=32, n_heads=4, d_ff=0)

    def test_wrong_input_shape_raises(self):
        block = GraphAttentionBlock(d_model=32, n_heads=4, d_ff=64, dropout=0.0)
        # Wrong last dim
        with self.assertRaisesRegex(ValueError, "x must be"):
            block(
                torch.randn(2, 6, 16),
                torch.zeros(2, 6, 6), torch.zeros(2, 6, 6),
                torch.ones(2, 6, dtype=torch.bool),
            )
        # adjacency shape mismatch
        with self.assertRaisesRegex(ValueError, "adjacency/geodesic_dist"):
            block(
                torch.randn(2, 6, 32),
                torch.zeros(2, 5, 5), torch.zeros(2, 6, 6),
                torch.ones(2, 6, dtype=torch.bool),
            )
        # mask shape / dtype mismatch
        with self.assertRaisesRegex(ValueError, "node_mask must be"):
            block(
                torch.randn(2, 6, 32),
                torch.zeros(2, 6, 6), torch.zeros(2, 6, 6),
                torch.ones(2, 6, dtype=torch.float32),  # wrong dtype
            )

    def test_mask_zeroes_padded_attention(self):
        # Sample 1 has only 3 valid nodes; pad the rest. Verify that padded
        # output rows are still finite (no NaN), and that valid rows depend
        # only on valid keys (mask is enforced).
        torch.manual_seed(0)
        block = GraphAttentionBlock(d_model=16, n_heads=2, d_ff=32, dropout=0.0)
        B, N = 1, 5
        x = torch.randn(B, N, 16)
        adj = _adj_line(3)
        adj_padded = torch.zeros(N, N)
        adj_padded[:3, :3] = adj
        geo_padded = torch.full((N, N), float("inf"))
        geo_padded[:3, :3] = _geo_from_adj(adj)
        mask = torch.tensor([[True, True, True, False, False]])

        y = block(x, adj_padded.unsqueeze(0), geo_padded.unsqueeze(0), mask)
        self.assertTrue(torch.isfinite(y).all(), "output contains NaN/Inf")

        # Repeat with same valid region but DIFFERENT padded values; the valid
        # rows of y should be identical (mask must isolate them).
        x2 = x.clone()
        x2[0, 3:] = 99.0  # garbage in padded region
        y2 = block(x2, adj_padded.unsqueeze(0), geo_padded.unsqueeze(0), mask)
        # NOTE: query at padded position still computes; only KEYs are masked.
        # So valid-query rows (0,1,2) should be identical.
        self.assertTrue(
            torch.allclose(y[0, :3], y2[0, :3], atol=1e-5),
            "valid query rows changed when padded keys were perturbed (mask leak)",
        )

    def test_bias_actually_used(self):
        # Two configurations identical except for non-zero adjacency_bias.weight.
        # Outputs must differ — proves the bias path participates in scores.
        torch.manual_seed(42)
        block = GraphAttentionBlock(d_model=16, n_heads=2, d_ff=32, dropout=0.0)
        block.eval()
        B, N = 1, 4
        x = torch.randn(B, N, 16)
        adj = _adj_line(N).expand(B, N, N).clone()
        geo = _geo_from_adj(adj[0]).expand(B, N, N).clone()
        mask = torch.ones(B, N, dtype=torch.bool)

        with torch.no_grad():
            y_normal = block(x, adj, geo, mask)
            # Zero out the adjacency_bias projection
            block.adjacency_bias.weight.zero_()
            y_no_adj_bias = block(x, adj, geo, mask)

        self.assertFalse(
            torch.allclose(y_normal, y_no_adj_bias, atol=1e-6),
            "outputs identical with/without adjacency_bias — bias path inert",
        )

    def test_geodesic_inf_handled(self):
        # geodesic_dist with +inf in valid positions (e.g., disconnected
        # components in valid nodes) should not produce NaN/Inf in output.
        block = GraphAttentionBlock(d_model=16, n_heads=2, d_ff=32, dropout=0.0)
        block.eval()
        B, N = 1, 4
        x = torch.randn(B, N, 16)
        adj = torch.zeros(B, N, N)
        adj[0, 0, 1] = adj[0, 1, 0] = 1.0
        adj[0, 2, 3] = adj[0, 3, 2] = 1.0
        geo = _geo_from_adj(adj[0]).unsqueeze(0)
        # Confirm geo has +inf between components
        self.assertTrue(torch.isinf(geo[0, 0, 2]))
        mask = torch.ones(B, N, dtype=torch.bool)

        y = block(x, adj, geo, mask)
        self.assertTrue(torch.isfinite(y).all(), "inf in geodesic_dist leaked into output")

    # --- Codex M1.2 round 1 R12 regression tests ---

    def test_x_nan_raises(self):
        block = GraphAttentionBlock(d_model=16, n_heads=2, d_ff=32, dropout=0.0)
        x = torch.randn(1, 4, 16)
        x[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "x contains NaN or Inf"):
            block(
                x, torch.zeros(1, 4, 4), torch.zeros(1, 4, 4),
                torch.ones(1, 4, dtype=torch.bool),
            )

    def test_adjacency_nan_raises(self):
        block = GraphAttentionBlock(d_model=16, n_heads=2, d_ff=32, dropout=0.0)
        adj = torch.zeros(1, 4, 4)
        adj[0, 0, 1] = float("nan")
        with self.assertRaisesRegex(ValueError, "adjacency contains NaN or Inf"):
            block(
                torch.randn(1, 4, 16), adj, torch.zeros(1, 4, 4),
                torch.ones(1, 4, dtype=torch.bool),
            )

    def test_geodesic_nan_raises(self):
        block = GraphAttentionBlock(d_model=16, n_heads=2, d_ff=32, dropout=0.0)
        geo = torch.zeros(1, 4, 4)
        geo[0, 0, 1] = float("nan")
        with self.assertRaisesRegex(ValueError, "geodesic_dist contains NaN"):
            block(
                torch.randn(1, 4, 16), torch.zeros(1, 4, 4), geo,
                torch.ones(1, 4, dtype=torch.bool),
            )

    def test_geodesic_neg_inf_raises(self):
        block = GraphAttentionBlock(d_model=16, n_heads=2, d_ff=32, dropout=0.0)
        geo = torch.zeros(1, 4, 4)
        geo[0, 0, 1] = float("-inf")
        with self.assertRaisesRegex(ValueError, "geodesic_dist contains -Inf"):
            block(
                torch.randn(1, 4, 16), torch.zeros(1, 4, 4), geo,
                torch.ones(1, 4, dtype=torch.bool),
            )

    def test_all_false_mask_raises(self):
        block = GraphAttentionBlock(d_model=16, n_heads=2, d_ff=32, dropout=0.0)
        mask = torch.zeros(2, 4, dtype=torch.bool)
        mask[0, :4] = True  # sample 0 OK, sample 1 all-False
        with self.assertRaisesRegex(ValueError, "all-False rows for batch element"):
            block(
                torch.randn(2, 4, 16), torch.zeros(2, 4, 4), torch.zeros(2, 4, 4),
                mask,
            )

    def test_dropout_out_of_range_raises(self):
        with self.assertRaisesRegex(ValueError, "dropout must be in"):
            GraphAttentionBlock(d_model=16, n_heads=2, d_ff=32, dropout=1.0)
        with self.assertRaisesRegex(ValueError, "dropout must be in"):
            GraphAttentionBlock(d_model=16, n_heads=2, d_ff=32, dropout=-0.1)

    def test_empty_batch_raises(self):
        block = GraphAttentionBlock(d_model=16, n_heads=2, d_ff=32, dropout=0.0)
        with self.assertRaisesRegex(ValueError, "B=0 and node count N=4 must be > 0"):
            block(
                torch.zeros(0, 4, 16),
                torch.zeros(0, 4, 4),
                torch.zeros(0, 4, 4),
                torch.zeros(0, 4, dtype=torch.bool),
            )

    def test_empty_node_count_raises(self):
        block = GraphAttentionBlock(d_model=16, n_heads=2, d_ff=32, dropout=0.0)
        with self.assertRaisesRegex(ValueError, "B=1 and node count N=0 must be > 0"):
            block(
                torch.zeros(1, 0, 16),
                torch.zeros(1, 0, 0),
                torch.zeros(1, 0, 0),
                torch.zeros(1, 0, dtype=torch.bool),
            )

    # --- Codex M1.2 round 3 R12 #8: topology semantic contracts ---

    def test_adjacency_negative_raises(self):
        block = GraphAttentionBlock(d_model=16, n_heads=2, d_ff=32, dropout=0.0)
        adj = _adj_line(4).unsqueeze(0)
        adj[0, 0, 1] = -1.0
        adj[0, 1, 0] = -1.0  # keep symmetric
        with self.assertRaisesRegex(ValueError, "adjacency contains negative"):
            block(
                torch.randn(1, 4, 16), adj, _geo_from_adj(_adj_line(4)).unsqueeze(0),
                torch.ones(1, 4, dtype=torch.bool),
            )

    def test_adjacency_asymmetric_raises(self):
        block = GraphAttentionBlock(d_model=16, n_heads=2, d_ff=32, dropout=0.0)
        adj = _adj_line(4).unsqueeze(0)
        adj[0, 0, 1] = 0.0  # break symmetry (was 1)
        with self.assertRaisesRegex(ValueError, "adjacency is not symmetric"):
            block(
                torch.randn(1, 4, 16), adj, _geo_from_adj(_adj_line(4)).unsqueeze(0),
                torch.ones(1, 4, dtype=torch.bool),
            )

    def test_adjacency_self_loop_raises(self):
        block = GraphAttentionBlock(d_model=16, n_heads=2, d_ff=32, dropout=0.0)
        adj = _adj_line(4).unsqueeze(0)
        adj[0, 2, 2] = 1.0  # self-loop
        with self.assertRaisesRegex(ValueError, "adjacency has non-zero diagonal"):
            block(
                torch.randn(1, 4, 16), adj, _geo_from_adj(_adj_line(4)).unsqueeze(0),
                torch.ones(1, 4, dtype=torch.bool),
            )

    def test_geodesic_negative_finite_raises(self):
        block = GraphAttentionBlock(d_model=16, n_heads=2, d_ff=32, dropout=0.0)
        geo = _geo_from_adj(_adj_line(4)).unsqueeze(0).clone()
        geo[0, 0, 1] = -1.0
        with self.assertRaisesRegex(ValueError, "geodesic_dist has negative finite"):
            block(
                torch.randn(1, 4, 16), _adj_line(4).unsqueeze(0), geo,
                torch.ones(1, 4, dtype=torch.bool),
            )

    def test_geodesic_asymmetric_raises(self):
        block = GraphAttentionBlock(d_model=16, n_heads=2, d_ff=32, dropout=0.0)
        geo = _geo_from_adj(_adj_line(4)).unsqueeze(0).clone()
        geo[0, 0, 1] = 1.0
        geo[0, 1, 0] = 2.0  # break symmetry
        with self.assertRaisesRegex(ValueError, "geodesic_dist is not symmetric"):
            block(
                torch.randn(1, 4, 16), _adj_line(4).unsqueeze(0), geo,
                torch.ones(1, 4, dtype=torch.bool),
            )

    def test_geodesic_nonzero_diagonal_raises(self):
        block = GraphAttentionBlock(d_model=16, n_heads=2, d_ff=32, dropout=0.0)
        geo = _geo_from_adj(_adj_line(4)).unsqueeze(0).clone()
        geo[0, 2, 2] = 1.0  # i→i distance should be 0
        with self.assertRaisesRegex(ValueError, "non-zero diagonal at valid nodes"):
            block(
                torch.randn(1, 4, 16), _adj_line(4).unsqueeze(0), geo,
                torch.ones(1, 4, dtype=torch.bool),
            )

    # --- Codex M1.2 round 4 R12 fixes ---

    def test_adjacency_subtle_asymmetry_raises(self):
        # Values within [0,1] but asymmetric by diff>atol that the default
        # rtol=1e-5 would have absorbed. rtol=0.0 catches it.
        # 0.99 vs 0.99001 — diff 1e-5, atol=1e-6, default rtol_slack = 1e-5*0.99 ≈ 9.9e-6
        # Sum = 1.099e-5 ≥ 1e-5 → passes default; rtol=0 → tolerance=1e-6 → catches.
        block = GraphAttentionBlock(d_model=16, n_heads=2, d_ff=32, dropout=0.0)
        adj = _adj_line(4).unsqueeze(0).clone()
        adj[0, 0, 1] = 0.99
        adj[0, 1, 0] = 0.99001
        with self.assertRaisesRegex(ValueError, "adjacency is not symmetric"):
            block(
                torch.randn(1, 4, 16), adj, _geo_from_adj(_adj_line(4)).unsqueeze(0),
                torch.ones(1, 4, dtype=torch.bool),
            )

    def test_geodesic_finite_vs_inf_asymmetry_raises(self):
        # geo[0,1]=+Inf but geo[1,0]=1.0 — finite/+Inf pattern asymmetry.
        # Previous allclose-on-finite-only check would miss this.
        block = GraphAttentionBlock(d_model=16, n_heads=2, d_ff=32, dropout=0.0)
        geo = _geo_from_adj(_adj_line(4)).unsqueeze(0).clone()
        geo[0, 0, 1] = float("inf")
        # geo[0, 1, 0] stays at 1.0 from line — asymmetric pattern
        with self.assertRaisesRegex(ValueError, "finite/.+Inf pattern is not symmetric"):
            block(
                torch.randn(1, 4, 16), _adj_line(4).unsqueeze(0), geo,
                torch.ones(1, 4, dtype=torch.bool),
            )

    # --- Codex M1.2 round 5 R12 fixes ---

    def test_geodesic_hop_bound_exceeded_raises(self):
        block = GraphAttentionBlock(d_model=16, n_heads=2, d_ff=32, dropout=0.0)
        geo = _geo_from_adj(_adj_line(4)).unsqueeze(0).clone()
        geo[0, 0, 3] = 1e6  # N=4, max hop=3; 1e6 violates
        geo[0, 3, 0] = 1e6  # keep symmetric so this check fires, not symmetry
        with self.assertRaisesRegex(ValueError, "has finite entries > 3"):
            block(
                torch.randn(1, 4, 16), _adj_line(4).unsqueeze(0), geo,
                torch.ones(1, 4, dtype=torch.bool),
            )

    def test_fp16_input_raises(self):
        block = GraphAttentionBlock(d_model=16, n_heads=2, d_ff=32, dropout=0.0)
        # Module is fp32 default; input fp16 mismatches both supported-dtype
        # and module-dtype checks.
        with self.assertRaisesRegex(ValueError, "dtype must be float32 or float64"):
            block(
                torch.randn(1, 4, 16).half(),
                _adj_line(4).unsqueeze(0), _geo_from_adj(_adj_line(4)).unsqueeze(0),
                torch.ones(1, 4, dtype=torch.bool),
            )

    def test_mixed_dtype_raises(self):
        block = GraphAttentionBlock(d_model=16, n_heads=2, d_ff=32, dropout=0.0)
        # x fp32, adj fp64 → mixed
        with self.assertRaisesRegex(ValueError, "adjacency.dtype torch.float64 != module dtype torch.float32"):
            block(
                torch.randn(1, 4, 16),
                _adj_line(4).unsqueeze(0).double(),
                _geo_from_adj(_adj_line(4)).unsqueeze(0),
                torch.ones(1, 4, dtype=torch.bool),
            )

    # --- Codex M1.2 round 6 R12 #11: adj/geo cross-consistency ---

    def test_geodesic_inconsistent_with_adjacency_raises(self):
        # 4-node line: correct adj, but geo[0,3]=2 instead of 3 (shortcut).
        # Symmetric + non-negative + bounded ≤ N-1, but Floyd would give 3.
        block = GraphAttentionBlock(d_model=16, n_heads=2, d_ff=32, dropout=0.0)
        adj = _adj_line(4).unsqueeze(0)
        bad_geo = torch.tensor([
            [0.0, 1.0, 2.0, 2.0],   # row 0: 0→3 wrong (says 2, true is 3)
            [1.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 1.0],
            [2.0, 2.0, 1.0, 0.0],   # row 3: 3→0 wrong (says 2, true is 3)
        ]).unsqueeze(0)
        with self.assertRaisesRegex(ValueError, "inconsistent with shortest-path"):
            block(
                torch.randn(1, 4, 16), adj, bad_geo,
                torch.ones(1, 4, dtype=torch.bool),
            )

    def test_geodesic_reachability_inconsistent_raises(self):
        # Disconnected adj (0-1 and 2-3 as 2 components), but geo claims
        # 0→2 has finite distance.
        block = GraphAttentionBlock(d_model=16, n_heads=2, d_ff=32, dropout=0.0)
        adj = torch.zeros(1, 4, 4)
        adj[0, 0, 1] = adj[0, 1, 0] = 1.0
        adj[0, 2, 3] = adj[0, 3, 2] = 1.0  # 0-1 + 2-3
        # Floyd would give geo[0,2]=+Inf (disconnected); we lie with 1.0.
        bad_geo = torch.zeros(1, 4, 4)
        bad_geo[0, 0, 1] = bad_geo[0, 1, 0] = 1.0
        bad_geo[0, 2, 3] = bad_geo[0, 3, 2] = 1.0
        bad_geo[0, 0, 2] = bad_geo[0, 2, 0] = 1.0  # finite but should be Inf
        with self.assertRaisesRegex(ValueError, "reachability pattern inconsistent"):
            block(
                torch.randn(1, 4, 16), adj, bad_geo,
                torch.ones(1, 4, dtype=torch.bool),
            )

    # --- Codex M1.2 round 7 R12 #12 + perf toggle ---

    def test_adjacency_above_one_raises(self):
        # Symmetric magnitude 1e6 adjacency passes all prior checks (finite,
        # non-neg, symmetric, zero-diag, geo consistent if recomputed) but
        # explodes the bias.
        block = GraphAttentionBlock(d_model=16, n_heads=2, d_ff=32, dropout=0.0)
        adj = _adj_line(4).unsqueeze(0) * 1e6
        # geodesic recomputed from this binary-times-1e6 ≠ correct hop count
        # since Floyd treats any >0 as 1-hop. So geo is still correct hop count.
        geo = _geo_from_adj(_adj_line(4)).unsqueeze(0)
        with self.assertRaisesRegex(ValueError, "adjacency contains values > 1.0"):
            block(
                torch.randn(1, 4, 16), adj, geo,
                torch.ones(1, 4, dtype=torch.bool),
            )

    def test_validate_inputs_false_bypasses_checks(self):
        # With validate_inputs=False, deliberately-bad inputs should NOT raise
        # (caller asserts they're pre-validated). Forward should still compute.
        block = GraphAttentionBlock(d_model=16, n_heads=2, d_ff=32, dropout=0.0)
        # Adjacency above 1.0 — would raise with default validate_inputs=True.
        adj = _adj_line(4).unsqueeze(0) * 100.0
        geo = _geo_from_adj(_adj_line(4)).unsqueeze(0)
        x = torch.randn(1, 4, 16)
        mask = torch.ones(1, 4, dtype=torch.bool)
        y = block(x, adj, geo, mask, validate_inputs=False)
        self.assertEqual(y.shape, (1, 4, 16))
        # Sanity: also confirm validate_inputs=True does raise on the same input
        with self.assertRaisesRegex(ValueError, "adjacency contains values > 1.0"):
            block(x, adj, geo, mask)  # default validate_inputs=True


if __name__ == "__main__":
    unittest.main(verbosity=2)
