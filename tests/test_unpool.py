"""Unit tests for src/models/graph_salad/unpool.py — M1.2 step 4."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.graph_salad.unpool import DynamicGraphUnpool


def _one_hot_assignment(B: int, J: int, C: int, assignments: list[list[int]]) -> torch.Tensor:
    """Build one-hot assignment [B, J, C] from a list of per-sample [J]-length argmax indices."""
    P = torch.zeros(B, J, C)
    for b in range(B):
        for j, c in enumerate(assignments[b]):
            if c >= 0:
                P[b, j, c] = 1.0
    return P


class DynamicGraphUnpoolTests(unittest.TestCase):

    def test_no_learnable_params(self):
        unpool = DynamicGraphUnpool(d_model=16)
        self.assertEqual(len(list(unpool.parameters())), 0)
        self.assertEqual(len(unpool.state_dict()), 0)

    def test_forward_shape(self):
        D, T_lat, J, C = 16, 4, 6, 3
        unpool = DynamicGraphUnpool(d_model=D, temporal_stride=2)
        coarse_features = torch.randn(1, T_lat, C, D)
        # One-hot assignment: joints 0,1 → c0; 2,3 → c1; 4,5 → c2
        assignment = _one_hot_assignment(1, J, C, [[0, 0, 1, 1, 2, 2]])
        joint_mask = torch.ones(1, J, dtype=torch.bool)
        coarse_mask = torch.ones(1, C, dtype=torch.bool)
        frame_mask_down = torch.ones(1, T_lat, dtype=torch.bool)
        out = unpool(coarse_features, assignment, joint_mask, coarse_mask, frame_mask_down)
        self.assertEqual(out["fine_features"].shape, (1, T_lat * 2, J, D))
        self.assertEqual(out["frame_mask_up"].shape, (1, T_lat * 2))

    def test_unpool_distributes_coarse(self):
        # With one-hot P, h_fine[t, j, :] should equal h_coarse[t, P_argmax(j), :]
        D, T_lat, J, C = 4, 2, 6, 3
        unpool = DynamicGraphUnpool(d_model=D, temporal_stride=1)
        coarse_features = torch.zeros(1, T_lat, C, D)
        coarse_features[0, :, 0, :] = 1.0  # c0 features = all-1
        coarse_features[0, :, 1, :] = 2.0
        coarse_features[0, :, 2, :] = 3.0
        assignment = _one_hot_assignment(1, J, C, [[0, 0, 1, 1, 2, 2]])
        joint_mask = torch.ones(1, J, dtype=torch.bool)
        coarse_mask = torch.ones(1, C, dtype=torch.bool)
        frame_mask_down = torch.ones(1, T_lat, dtype=torch.bool)
        out = unpool(coarse_features, assignment, joint_mask, coarse_mask, frame_mask_down)
        ff = out["fine_features"]
        # Joints 0,1 get c0's features (1.0)
        self.assertTrue(torch.allclose(ff[0, :, 0, :], torch.full((T_lat, D), 1.0)))
        self.assertTrue(torch.allclose(ff[0, :, 1, :], torch.full((T_lat, D), 1.0)))
        # Joints 2,3 get c1's (2.0)
        self.assertTrue(torch.allclose(ff[0, :, 2, :], torch.full((T_lat, D), 2.0)))
        # Joints 4,5 get c2's (3.0)
        self.assertTrue(torch.allclose(ff[0, :, 5, :], torch.full((T_lat, D), 3.0)))

    def test_padded_joint_output_zero(self):
        D, T_lat, J, C = 4, 2, 5, 2
        unpool = DynamicGraphUnpool(d_model=D, temporal_stride=1)
        coarse_features = torch.randn(1, T_lat, C, D)
        # Joint 4 is padded; assignment row 4 must be all zeros for valid input.
        assignment = _one_hot_assignment(1, J, C, [[0, 0, 1, 1, -1]])
        joint_mask = torch.tensor([[True, True, True, True, False]])
        coarse_mask = torch.ones(1, C, dtype=torch.bool)
        frame_mask_down = torch.ones(1, T_lat, dtype=torch.bool)
        out = unpool(coarse_features, assignment, joint_mask, coarse_mask, frame_mask_down)
        self.assertEqual(out["fine_features"][0, :, 4, :].abs().sum().item(), 0.0)

    def test_invalid_assignment_row_sum_raises(self):
        # Valid joint with assignment row sum != 1
        unpool = DynamicGraphUnpool(d_model=4)
        coarse_features = torch.randn(1, 2, 2, 4)
        # joint 0 valid but assignment row = [0.3, 0.3] sum 0.6 (not 1)
        assignment = torch.tensor([[[0.3, 0.3], [1.0, 0.0]]])
        joint_mask = torch.tensor([[True, True]])
        coarse_mask = torch.ones(1, 2, dtype=torch.bool)
        frame_mask_down = torch.ones(1, 2, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "row sums on valid joints must be ~1"):
            unpool(coarse_features, assignment, joint_mask, coarse_mask, frame_mask_down)

    def test_padded_row_nonzero_raises(self):
        # Padded joint with non-zero assignment row
        unpool = DynamicGraphUnpool(d_model=4)
        coarse_features = torch.randn(1, 2, 2, 4)
        assignment = torch.tensor([[[1.0, 0.0], [0.5, 0.5]]])  # joint 1 padded, row nonzero
        joint_mask = torch.tensor([[True, False]])
        coarse_mask = torch.ones(1, 2, dtype=torch.bool)
        frame_mask_down = torch.ones(1, 2, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "padded joints must be"):
            unpool(coarse_features, assignment, joint_mask, coarse_mask, frame_mask_down)

    def test_wrong_dtype_raises(self):
        unpool = DynamicGraphUnpool(d_model=4)
        coarse_features = torch.randn(1, 2, 2, 4)
        assignment = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        joint_mask = torch.ones(1, 2, dtype=torch.bool)
        coarse_mask = torch.ones(1, 2, dtype=torch.bool)
        frame_mask_down = torch.ones(1, 2, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "coarse_features.dtype must be float32"):
            unpool(coarse_features.double(), assignment, joint_mask, coarse_mask, frame_mask_down)

    def test_constructor_strict_int(self):
        with self.assertRaisesRegex(TypeError, "d_model must be strict int"):
            DynamicGraphUnpool(d_model=16.0)
        with self.assertRaisesRegex(TypeError, "temporal_stride must be strict int"):
            DynamicGraphUnpool(d_model=16, temporal_stride=True)

    def test_temporal_stride_one_is_identity(self):
        D, T_lat, J, C = 4, 3, 4, 2
        unpool = DynamicGraphUnpool(d_model=D, temporal_stride=1)
        coarse_features = torch.randn(1, T_lat, C, D)
        assignment = _one_hot_assignment(1, J, C, [[0, 0, 1, 1]])
        joint_mask = torch.ones(1, J, dtype=torch.bool)
        coarse_mask = torch.ones(1, C, dtype=torch.bool)
        frame_mask_down = torch.ones(1, T_lat, dtype=torch.bool)
        out = unpool(coarse_features, assignment, joint_mask, coarse_mask, frame_mask_down)
        self.assertEqual(out["fine_features"].shape, (1, T_lat, J, D))

    def test_assignment_negative_raises(self):
        unpool = DynamicGraphUnpool(d_model=4)
        coarse_features = torch.randn(1, 2, 2, 4)
        # Row [1.5, -0.5] sums to 1 but has negative entry
        assignment = torch.tensor([[[1.5, -0.5], [1.0, 0.0]]])
        joint_mask = torch.ones(1, 2, dtype=torch.bool)
        coarse_mask = torch.ones(1, 2, dtype=torch.bool)
        frame_mask_down = torch.ones(1, 2, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "negative entries"):
            unpool(coarse_features, assignment, joint_mask, coarse_mask, frame_mask_down)

    def test_assignment_to_padded_coarse_raises(self):
        # Joint 0 valid, assigns to coarse slot 1, but coarse_mask=[T, F]
        # (slot 1 is padded). Must raise.
        unpool = DynamicGraphUnpool(d_model=4)
        coarse_features = torch.randn(1, 2, 2, 4)
        assignment = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
        joint_mask = torch.ones(1, 2, dtype=torch.bool)
        coarse_mask = torch.tensor([[True, False]])  # slot 1 padded
        frame_mask_down = torch.ones(1, 2, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "coarse_mask=False columns"):
            unpool(coarse_features, assignment, joint_mask, coarse_mask, frame_mask_down)

    def test_temporal_mask_consistent_with_features(self):
        # repeat_interleave for both features and mask — exact agreement.
        D, T_lat, J, C = 4, 2, 2, 2
        unpool = DynamicGraphUnpool(d_model=D, temporal_stride=2)
        coarse_features = torch.zeros(1, T_lat, C, D)
        coarse_features[0, 0, 0, :] = 5.0  # down-frame 0
        coarse_features[0, 1, 0, :] = 9.0  # down-frame 1
        assignment = _one_hot_assignment(1, J, C, [[0, 0]])
        joint_mask = torch.ones(1, J, dtype=torch.bool)
        coarse_mask = torch.tensor([[True, False]])  # only c=0 valid
        # Slot 1 padded → set assignment col 1 to 0 manually
        assignment[0, :, 1] = 0.0
        frame_mask_down = torch.tensor([[True, False]])  # down-frame 1 padded
        out = unpool(coarse_features, assignment, joint_mask, coarse_mask, frame_mask_down)
        # Up frames 0, 1 ↔ down 0 (valid 5.0); Up frames 2, 3 ↔ down 1 (invalid)
        # With repeat_interleave: ff[0, 0, 0, :] = 5.0, ff[0, 1, 0, :] = 5.0,
        # ff[0, 2, 0, :] = 9.0, ff[0, 3, 0, :] = 9.0
        # mask_up = [T, T, F, F]
        # After mask zeroing, fine_up[0, 2:, ...] should be 0
        self.assertTrue(torch.allclose(out["fine_features"][0, 0, 0, :], torch.full((D,), 5.0)))
        self.assertTrue(torch.allclose(out["fine_features"][0, 1, 0, :], torch.full((D,), 5.0)))
        # Frames 2, 3 should be zeroed by mask
        self.assertEqual(out["fine_features"][0, 2, :, :].abs().sum().item(), 0.0)
        self.assertEqual(out["fine_features"][0, 3, :, :].abs().sum().item(), 0.0)
        self.assertEqual(out["frame_mask_up"].tolist(), [[True, True, False, False]])

    def test_finite_check_raises(self):
        unpool = DynamicGraphUnpool(d_model=4)
        coarse_features = torch.randn(1, 2, 2, 4)
        coarse_features[0, 0, 0, 0] = float("nan")
        assignment = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        joint_mask = torch.ones(1, 2, dtype=torch.bool)
        coarse_mask = torch.ones(1, 2, dtype=torch.bool)
        frame_mask_down = torch.ones(1, 2, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "coarse_features contains NaN"):
            unpool(coarse_features, assignment, joint_mask, coarse_mask, frame_mask_down)


if __name__ == "__main__":
    unittest.main(verbosity=2)
