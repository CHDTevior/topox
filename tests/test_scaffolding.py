"""Unit tests for M1.1 scaffolding: GraphMotionBatch + GraphSaladDenoiserStub.

CPU-only, no GPU. Run:
    python tests/test_scaffolding.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

# Make src importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.graph_salad import GraphMotionBatch, GraphSaladDenoiserStub


def _build_line_adjacency(J: int) -> torch.Tensor:
    """Undirected adjacency for a line skeleton: 0-1-2-...-J-1."""
    A = torch.zeros(J, J)
    for j in range(1, J):
        A[j, j - 1] = 1.0
        A[j - 1, j] = 1.0
    return A


def _make_synthetic_collate_dict(B: int = 2, T: int = 16, J: int = 24) -> dict:
    """Mimic the dict that UnifiedMotionDataset.collate_fn produces.

    Synthesizes a *consistent* batch where ``num_joints == joint_mask.sum(1)``
    and ``num_frames == frame_mask.sum(1)``. Sample 0 uses full J/T; sample 1
    uses J-2 / T-4 (to exercise per-sample variable validity); samples 2+
    use full J/T. Tests that need INconsistent state mutate fields after
    construction.
    """
    # Per-sample valid joint / frame counts
    nj = [J, max(J - 2, 1)] + [J] * max(B - 2, 0)
    nf = [T, max(T - 4, 1)] + [T] * max(B - 2, 0)
    nj = nj[:B]
    nf = nf[:B]

    joint_mask = torch.zeros(B, J, dtype=torch.bool)
    frame_mask = torch.zeros(B, T, dtype=torch.bool)
    for i in range(B):
        joint_mask[i, : nj[i]] = True
        frame_mask[i, : nf[i]] = True

    # Build per-sample line adjacency (matches the linear-chain parent_indices
    # set below) so graph-semantics check in batch.from_collate_dict passes.
    adj = torch.zeros(B, J, J)
    for i in range(B):
        adj[i, : nj[i], : nj[i]] = _build_line_adjacency(nj[i])

    return {
        # Padded tensors
        "motion_features": torch.randn(B, T, J, 6),
        "skeleton_features": torch.randn(B, J, 9),
        "joint_mask": joint_mask,
        "frame_mask": frame_mask,
        "adjacency": adj,
        "geodesic_dist": torch.zeros(B, J, J),
        "name_hashes": torch.zeros(B, J, dtype=torch.long),
        "root_position": torch.randn(B, T, 3),
        "root_velocity": torch.randn(B, T, 3),
        "local_rotations_6d": torch.randn(B, T, J, 6),
        "foot_contact": torch.zeros(B, T, 4),
        "bone_lengths": torch.ones(B, T, J),
        "rest_offsets": torch.randn(B, J, 3),
        # Batched scalars (collate fn stacks ints → tensor; floats too)
        "num_joints": torch.tensor(nj, dtype=torch.long),
        "num_frames": torch.tensor(nf, dtype=torch.long),
        "fps": torch.full((B,), 20.0),
        "has_rotations": torch.ones(B, dtype=torch.bool),
        # Per-sample variable-length lists — codex M1.1 round 4 R12 fix: lengths
    # MUST equal num_joints[i] (not J_max), matching unified_dataset.py upstream.
        "parent_indices":    [[-1] + list(range(nj[i] - 1)) for i in range(B)],
        "joint_names":       [[f"j{j}" for j in range(nj[i])] for i in range(B)],
        "canonical_names":   [[f"c{j}" for j in range(nj[i])] for i in range(B)],
        "bone_lengths_rest": [[0.1] * nj[i] for i in range(B)],
        # Per-sample strings
        "text": [f"sample {i} text" for i in range(B)],
        "skeleton_id": [f"skel_{i}" for i in range(B)],
        "motion_id":   [f"mot_{i}" for i in range(B)],
    }


class GraphMotionBatchTests(unittest.TestCase):

    def test_from_dict_synthetic(self):
        d = _make_synthetic_collate_dict()
        batch = GraphMotionBatch.from_collate_dict(d)
        self.assertEqual(batch.batch_size, 2)
        self.assertEqual(batch.max_frames, 16)
        self.assertEqual(batch.max_joints, 24)
        self.assertEqual(len(batch.parent_indices), 2)
        self.assertEqual(len(batch.text), 2)
        # New helper generates "sample {i} text" per index
        self.assertEqual(batch.text[0], "sample 0 text")

    def test_no_tensor_copy(self):
        # Dataclass should hold REFERENCES to the input dict's tensors, not copies.
        d = _make_synthetic_collate_dict()
        batch = GraphMotionBatch.from_collate_dict(d)
        # Mutate dict's tensor; batch's tensor should mirror the change.
        d["motion_features"][0, 0, 0, 0] = 999.0
        self.assertEqual(batch.motion_features[0, 0, 0, 0].item(), 999.0)

    def test_missing_required_key_raises(self):
        d = _make_synthetic_collate_dict()
        del d["adjacency"]
        with self.assertRaisesRegex(ValueError, "missing required keys.*adjacency"):
            GraphMotionBatch.from_collate_dict(d)

    def test_wrong_type_raises(self):
        d = _make_synthetic_collate_dict()
        d["motion_features"] = "not a tensor"
        with self.assertRaisesRegex(ValueError, "must be torch.Tensor"):
            GraphMotionBatch.from_collate_dict(d)

    def test_batch_dim_mismatch_raises(self):
        d = _make_synthetic_collate_dict()
        d["motion_features"] = torch.randn(3, 16, 24, 6)  # B=3, not 2
        with self.assertRaisesRegex(ValueError, "batch dim"):
            GraphMotionBatch.from_collate_dict(d)

    def test_list_length_mismatch_raises(self):
        d = _make_synthetic_collate_dict()
        d["text"] = ["only one text"]  # length 1, not 2
        with self.assertRaisesRegex(ValueError, "list 'text' length 1 .* batch size 2"):
            GraphMotionBatch.from_collate_dict(d)

    def test_wrong_dtype_mask_raises(self):
        # joint_mask must be bool, not float
        d = _make_synthetic_collate_dict()
        d["joint_mask"] = d["joint_mask"].float()
        with self.assertRaisesRegex(ValueError, "'joint_mask' dtype must be torch.bool"):
            GraphMotionBatch.from_collate_dict(d)

    def test_wrong_last_dim_raises(self):
        # motion_features last-dim must be 6 (pos 3 + vel 3)
        d = _make_synthetic_collate_dict()
        d["motion_features"] = torch.randn(2, 16, 24, 5)
        with self.assertRaisesRegex(ValueError, "'motion_features' last-dim must be 6"):
            GraphMotionBatch.from_collate_dict(d)

    def test_non_square_adjacency_raises(self):
        d = _make_synthetic_collate_dict()
        d["adjacency"] = torch.randn(2, 24, 25)  # non-square
        with self.assertRaisesRegex(ValueError, "'adjacency' must be square"):
            GraphMotionBatch.from_collate_dict(d)

    def test_batched_scalar_as_list_raises(self):
        # collate_fn always tensorizes int/float/bool; passing list should raise
        d = _make_synthetic_collate_dict()
        d["num_joints"] = [24, 22]  # list instead of tensor
        with self.assertRaisesRegex(ValueError, "'num_joints' must be torch.Tensor"):
            GraphMotionBatch.from_collate_dict(d)

    def test_batched_scalar_wrong_shape_raises(self):
        d = _make_synthetic_collate_dict()
        d["num_joints"] = torch.tensor([[24, 22]], dtype=torch.long)  # shape [1,2] not [2]
        with self.assertRaisesRegex(ValueError, "'num_joints' shape must be"):
            GraphMotionBatch.from_collate_dict(d)

    def test_batched_scalar_wrong_dtype_raises(self):
        d = _make_synthetic_collate_dict()
        d["fps"] = torch.tensor([20, 20], dtype=torch.long)  # int instead of float
        with self.assertRaisesRegex(ValueError, "'fps' dtype must be torch.float32"):
            GraphMotionBatch.from_collate_dict(d)

    def test_cross_tensor_t_max_mismatch_raises(self):
        d = _make_synthetic_collate_dict()
        d["frame_mask"] = torch.ones(2, 32, dtype=torch.bool)  # T=32, not 16
        with self.assertRaisesRegex(ValueError, "'frame_mask' T-dim"):
            GraphMotionBatch.from_collate_dict(d)

    def test_cross_tensor_j_max_mismatch_raises(self):
        d = _make_synthetic_collate_dict()
        d["adjacency"] = torch.zeros(2, 30, 30)  # J=30, not 24
        with self.assertRaisesRegex(ValueError, "'adjacency'.*J-dim|J_max"):
            GraphMotionBatch.from_collate_dict(d)

    def test_motion_features_scalar_rank_raises(self):
        # scalar (rank 0) for motion_features must not crash with IndexError
        d = _make_synthetic_collate_dict()
        d["motion_features"] = torch.tensor(0.0)  # rank 0 scalar
        with self.assertRaisesRegex(ValueError, "rank >= 1"):
            GraphMotionBatch.from_collate_dict(d)

    def test_zero_batch_size_raises(self):
        d = _make_synthetic_collate_dict(B=2)
        # Replace motion_features with B=0 tensor — synthetic, just to test the guard
        d["motion_features"] = torch.zeros(0, 16, 24, 6)
        with self.assertRaisesRegex(ValueError, "batch size must be > 0"):
            GraphMotionBatch.from_collate_dict(d)

    def test_nan_in_float_tensor_raises(self):
        d = _make_synthetic_collate_dict()
        d["motion_features"][0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            GraphMotionBatch.from_collate_dict(d)

    def test_inf_in_float_tensor_raises(self):
        d = _make_synthetic_collate_dict()
        d["adjacency"][1, 0, 1] = float("inf")
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            GraphMotionBatch.from_collate_dict(d)

    def test_mixed_device_raises(self):
        # Skip if no CUDA available (most CI workers); just verify code path with mock
        # by faking device on a tensor (we use meta device which always exists)
        d = _make_synthetic_collate_dict()
        d["adjacency"] = d["adjacency"].to("meta")  # different device from cpu
        with self.assertRaisesRegex(ValueError, "device .* must be on the same device"):
            GraphMotionBatch.from_collate_dict(d)

    def test_parent_indices_inner_type_raises(self):
        # length-24 list with one float (1.5) embedded → inner-type violation
        d = _make_synthetic_collate_dict()
        bad = [-1] + list(range(22)) + [1.5]  # length 24, last is float
        d["parent_indices"] = [bad, [-1] + list(range(21))]
        with self.assertRaisesRegex(ValueError, "parent_indices.*type"):
            GraphMotionBatch.from_collate_dict(d)

    def test_text_inner_type_raises(self):
        d = _make_synthetic_collate_dict()
        d["text"] = ["valid", 42]  # 42 is int, not str; length 2 matches B=2
        with self.assertRaisesRegex(ValueError, "'text'.*must be str"):
            GraphMotionBatch.from_collate_dict(d)

    def test_parent_indices_outer_inner_not_list_raises(self):
        d = _make_synthetic_collate_dict()
        d["parent_indices"] = ["not a list", [-1] + list(range(21))]  # 1st is str
        with self.assertRaisesRegex(ValueError, "parent_indices.*\\[0\\] must be list"):
            GraphMotionBatch.from_collate_dict(d)

    def test_num_joints_out_of_range_raises(self):
        d = _make_synthetic_collate_dict()
        d["num_joints"] = torch.tensor([-1, 24], dtype=torch.long)
        d["joint_mask"] = torch.zeros(2, 24, dtype=torch.bool)  # also bypass mask cross-check
        d["joint_mask"][1, :24] = True
        with self.assertRaisesRegex(ValueError, "num_joints out of range"):
            GraphMotionBatch.from_collate_dict(d)

    def test_num_joints_mask_mismatch_raises(self):
        # num_joints says 24 + 24 but joint_mask only has 22 True for sample 0
        d = _make_synthetic_collate_dict()
        d["joint_mask"][0, 22:] = False  # only 22 valid joints in sample 0
        # num_joints still says [24, 24] → mismatch
        with self.assertRaisesRegex(ValueError, "num_joints.*!= joint_mask.sum"):
            GraphMotionBatch.from_collate_dict(d)

    def test_num_frames_mask_mismatch_raises(self):
        d = _make_synthetic_collate_dict()
        d["frame_mask"][0, 10:] = False  # 10 valid frames sample 0
        with self.assertRaisesRegex(ValueError, "num_frames.*!= frame_mask.sum"):
            GraphMotionBatch.from_collate_dict(d)

    def test_fps_nan_raises(self):
        d = _make_synthetic_collate_dict()
        d["fps"] = torch.tensor([float("nan"), 20.0])
        with self.assertRaisesRegex(ValueError, "fps contains NaN or Inf"):
            GraphMotionBatch.from_collate_dict(d)

    def test_fps_negative_raises(self):
        d = _make_synthetic_collate_dict()
        d["fps"] = torch.tensor([-1.0, 20.0])
        with self.assertRaisesRegex(ValueError, "fps must be > 0"):
            GraphMotionBatch.from_collate_dict(d)

    def test_parent_indices_bool_raises(self):
        # Python: isinstance(True, int) == True, but semantically distinct.
        d = _make_synthetic_collate_dict()
        # length-24 list with one True embedded
        bad = [-1] + list(range(22)) + [True]
        d["parent_indices"] = [bad, [-1] + list(range(21))]
        with self.assertRaisesRegex(ValueError, "parent_indices.*is bool"):
            GraphMotionBatch.from_collate_dict(d)

    def test_bone_lengths_rest_bool_raises(self):
        d = _make_synthetic_collate_dict()
        # sample 0 has nj=J=24, sample 1 has nj=22
        d["bone_lengths_rest"] = [[0.1, True, 0.3] + [0.1] * 21, [0.1] * 22]
        with self.assertRaisesRegex(ValueError, "bone_lengths_rest.*is bool"):
            GraphMotionBatch.from_collate_dict(d)

    def test_parent_indices_cardinality_mismatch_raises(self):
        # num_joints[0]=24, but parent_indices[0] has only 20 entries → mismatch
        d = _make_synthetic_collate_dict()
        d["parent_indices"] = [[-1] + list(range(19)), [-1] + list(range(21))]
        with self.assertRaisesRegex(ValueError, "parent_indices.*length 20.*num_joints"):
            GraphMotionBatch.from_collate_dict(d)

    def test_joint_names_cardinality_mismatch_raises(self):
        d = _make_synthetic_collate_dict()
        d["joint_names"] = [["j0"] * 24, ["j0"] * 30]  # sample 1: 30 names but nj=22
        with self.assertRaisesRegex(ValueError, "joint_names.*length 30.*num_joints"):
            GraphMotionBatch.from_collate_dict(d)

    def test_parent_indices_invalid_tree_no_root_raises(self):
        # all non-root → no -1 entry → invalid tree
        d = _make_synthetic_collate_dict()
        d["parent_indices"] = [[0] + list(range(23)), [-1] + list(range(21))]
        # First sample has parents[0]=0 (self-loop / no root)
        with self.assertRaisesRegex(ValueError, "parent_indices\\[0\\].*not a valid rooted tree"):
            GraphMotionBatch.from_collate_dict(d)

    def test_parent_indices_multi_root_raises(self):
        d = _make_synthetic_collate_dict()
        # Two -1 entries in sample 0 → invalid tree
        bad = [-1, -1] + list(range(22))
        d["parent_indices"] = [bad, [-1] + list(range(21))]
        with self.assertRaisesRegex(ValueError, "parent_indices\\[0\\].*not a valid rooted tree"):
            GraphMotionBatch.from_collate_dict(d)

    def test_parent_indices_out_of_range_raises(self):
        d = _make_synthetic_collate_dict()
        # Parent index 999 out of range for J=24
        bad = [-1] + list(range(22)) + [999]
        d["parent_indices"] = [bad, [-1] + list(range(21))]
        with self.assertRaisesRegex(ValueError, "parent_indices\\[0\\].*not a valid rooted tree"):
            GraphMotionBatch.from_collate_dict(d)

    # --- Round 6 R12 fixes: mask contiguity + graph semantics ---
    # NOTE: validation order is num_joints==mask.sum (4c) BEFORE prefix contiguity (5b),
    # so to trigger 5b alone we must keep mask.sum() == num_joints by compensating
    # the hole with an extra True in the padded region. (The "padded-True" branch
    # in 5b is mathematically dead given 4c + prefix-True; it's kept as defense
    # in depth per codex round 5, but is unreachable in isolation.)

    def test_joint_mask_hole_raises(self):
        d = _make_synthetic_collate_dict()  # nj=[24, 22]
        # Use sample 1 (has padded room): hole at idx 5, compensate True at idx 22.
        d["joint_mask"][1, 5] = False
        d["joint_mask"][1, 22] = True  # sum stays 22 (matches num_joints[1])
        with self.assertRaisesRegex(ValueError, "joint_mask\\[1\\] is not contiguous"):
            GraphMotionBatch.from_collate_dict(d)

    def test_frame_mask_hole_raises(self):
        d = _make_synthetic_collate_dict()  # nf=[16, 12]
        d["frame_mask"][1, 5] = False
        d["frame_mask"][1, 12] = True  # sum stays 12
        with self.assertRaisesRegex(ValueError, "frame_mask\\[1\\] is not contiguous"):
            GraphMotionBatch.from_collate_dict(d)

    def test_adjacency_non_binary_raises(self):
        d = _make_synthetic_collate_dict()
        d["adjacency"][0, 0, 1] = 2.0  # not 0 or 1
        with self.assertRaisesRegex(ValueError, "adjacency.*non-binary"):
            GraphMotionBatch.from_collate_dict(d)

    def test_adjacency_non_symmetric_raises(self):
        d = _make_synthetic_collate_dict()
        d["adjacency"][0, 1, 0] = 0.0  # break symmetry while [0,0,1]=1
        with self.assertRaisesRegex(ValueError, "adjacency.*not symmetric"):
            GraphMotionBatch.from_collate_dict(d)

    def test_adjacency_diagonal_raises(self):
        d = _make_synthetic_collate_dict()
        d["adjacency"][0, 3, 3] = 1.0  # self-loop on diagonal
        with self.assertRaisesRegex(ValueError, "adjacency.*non-zero diagonal"):
            GraphMotionBatch.from_collate_dict(d)

    def test_adjacency_mismatch_parents_raises(self):
        d = _make_synthetic_collate_dict()
        # parents describe line 0-1-2-...-23 but we set adjacency[0,5,10]=1 (extra edge)
        d["adjacency"][0, 5, 10] = 1.0
        d["adjacency"][0, 10, 5] = 1.0
        with self.assertRaisesRegex(ValueError, "adjacency.*does not match.*parent_indices"):
            GraphMotionBatch.from_collate_dict(d)

    def test_adjacency_padded_nonzero_raises(self):
        d = _make_synthetic_collate_dict()
        # Sample 1 has nj=22; pollute the padded row/col with non-zero
        d["adjacency"][1, 22, 0] = 1.0
        d["adjacency"][1, 0, 22] = 1.0
        with self.assertRaisesRegex(ValueError, "adjacency.*non-zero entries in padded"):
            GraphMotionBatch.from_collate_dict(d)

    # --- Round 7 R12 #8 fix: FK-topology ordering invariant ---

    def test_parent_indices_root_not_zero_raises(self):
        # parent_indices = [1, 2, -1] is a valid tree (root=2) but violates
        # treeik_decoder's root=0 requirement. The implied graph (line 0-1-2)
        # matches the default helper's _build_line_adjacency, so no adj override.
        d = _make_synthetic_collate_dict(B=1, T=16, J=3)
        d["parent_indices"] = [[1, 2, -1]]
        with self.assertRaisesRegex(ValueError, "parent_indices.*FK ordering"):
            GraphMotionBatch.from_collate_dict(d)

    def test_parent_indices_parent_after_child_raises(self):
        # parent_indices = [-1, 2, 0] — root=0 ✓ but parents[1]=2 > 1 violates
        # parent-before-child ordering. Graph is not a line (edges (1,2),(0,2)),
        # so we must override default adj to match (else graph-semantics fires first).
        d = _make_synthetic_collate_dict(B=1, T=16, J=3)
        d["parent_indices"] = [[-1, 2, 0]]
        adj = torch.zeros(1, 3, 3)
        adj[0, 1, 2] = adj[0, 2, 1] = 1.0
        adj[0, 0, 2] = adj[0, 2, 0] = 1.0
        d["adjacency"] = adj
        with self.assertRaisesRegex(ValueError, "parent_indices.*FK ordering"):
            GraphMotionBatch.from_collate_dict(d)

    def test_real_dataset_compat(self):
        """Smoke test using actual UnifiedMotionDataset collate output."""
        try:
            from src.data.unified_dataset import UnifiedMotionDataset, collate_fn
        except ImportError:
            self.skipTest("UnifiedMotionDataset not importable")

        data_dir = (
            Path(__file__).resolve().parents[1]
            / "data" / "cs_sparse2full_tgt"
        )
        if not data_dir.exists():
            self.skipTest(f"dataset dir {data_dir} not found")

        ds = UnifiedMotionDataset(
            data_dirs=[data_dir], split="val",
            max_joints=160, max_frames=196, normalize=False,
        )
        if len(ds) < 2:
            self.skipTest(f"dataset has {len(ds)} val samples, need >= 2")
        samples = [ds[0], ds[1]]
        d = collate_fn(samples)
        batch = GraphMotionBatch.from_collate_dict(d)
        self.assertEqual(batch.batch_size, 2)
        self.assertEqual(batch.max_joints, 160)
        self.assertEqual(batch.max_frames, 196)
        # Real parent_indices should be per-sample lists
        self.assertEqual(len(batch.parent_indices), 2)
        self.assertIsInstance(batch.parent_indices[0], list)
        self.assertEqual(batch.parent_indices[0][0], -1)  # root


class GraphSaladDenoiserStubTests(unittest.TestCase):

    def test_instantiable(self):
        stub = GraphSaladDenoiserStub(d_model=128, n_heads=4)
        self.assertEqual(stub.d_model, 128)
        self.assertEqual(stub.n_heads, 4)

    def test_no_parameters(self):
        # Stub has zero learnable params — preserves ckpt-compat envelope.
        stub = GraphSaladDenoiserStub()
        params = list(stub.parameters())
        self.assertEqual(len(params), 0, f"stub leaked params: {params}")
        sd = stub.state_dict()
        self.assertEqual(len(sd), 0, f"stub leaked state_dict entries: {list(sd)}")

    def test_forward_raises_not_implemented(self):
        stub = GraphSaladDenoiserStub()
        B, T_lat, C, D = 2, 4, 8, 256
        z_t = torch.randn(B, T_lat, C, D)
        ts = torch.zeros(B, dtype=torch.long)
        text = ["a", "b"]
        adj = torch.zeros(B, C, C)
        geo = torch.zeros(B, C, C)
        cm = torch.ones(B, C, dtype=torch.bool)
        fm = torch.ones(B, T_lat, dtype=torch.bool)
        with self.assertRaisesRegex(NotImplementedError, "Phase-2"):
            stub(z_t, ts, text, adj, geo, cm, fm)

    def test_forward_signature_includes_level2_meta(self):
        # Verify the call signature accepts level2_meta keyword (codex pre-scaffold req)
        stub = GraphSaladDenoiserStub()
        import inspect
        sig = inspect.signature(stub.forward)
        params = list(sig.parameters)
        self.assertIn("level2_meta", params, f"forward params: {params}")
        self.assertEqual(
            sig.parameters["level2_meta"].default, None,
            "level2_meta must default to None for M1 callers"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
