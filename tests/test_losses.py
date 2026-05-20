"""Unit tests for src/models/graph_salad/losses.py — M1.2 step 5."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.graph_salad.losses import (
    aggregate_pool_aux,
    compute_total_loss,
    masked_bone_length,
    masked_kl_gaussian,
    masked_l1_pos,
    masked_l1_vel,
    masked_vel_consistency,
)


class MaskedL1PosTests(unittest.TestCase):

    def test_zero_when_equal(self):
        pred = torch.zeros(1, 4, 6, 3)
        gt = torch.zeros(1, 4, 6, 3)
        jm = torch.ones(1, 6, dtype=torch.bool)
        fm = torch.ones(1, 4, dtype=torch.bool)
        self.assertEqual(masked_l1_pos(pred, gt, jm, fm).item(), 0.0)

    def test_known_difference(self):
        pred = torch.full((1, 2, 3, 3), 1.0)
        gt = torch.zeros(1, 2, 3, 3)
        jm = torch.ones(1, 3, dtype=torch.bool)
        fm = torch.ones(1, 2, dtype=torch.bool)
        # |1.0 - 0.0| = 1.0 per dim, 3 dims → sum=3 per (j, t) pair. Mean = 3.0.
        loss = masked_l1_pos(pred, gt, jm, fm).item()
        self.assertAlmostEqual(loss, 3.0)

    def test_padded_ignored(self):
        pred = torch.full((1, 2, 3, 3), 1.0)
        gt = torch.zeros(1, 2, 3, 3)
        # Only joint 0 valid; joints 1, 2 padded → ignored
        jm = torch.tensor([[True, False, False]])
        fm = torch.ones(1, 2, dtype=torch.bool)
        loss = masked_l1_pos(pred, gt, jm, fm).item()
        # 3.0 per valid (j, t); 2 frames × 1 joint = 2 entries; mean = 3.0.
        self.assertAlmostEqual(loss, 3.0)

    def test_wrong_shape_raises(self):
        with self.assertRaisesRegex(ValueError, "must be \\[B, T, J, 3\\]"):
            masked_l1_pos(torch.zeros(1, 2, 3, 4), torch.zeros(1, 2, 3, 4),
                         torch.ones(1, 3, dtype=torch.bool),
                         torch.ones(1, 2, dtype=torch.bool))

    def test_nan_raises(self):
        pred = torch.zeros(1, 2, 3, 3)
        pred[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            masked_l1_pos(pred, torch.zeros(1, 2, 3, 3),
                         torch.ones(1, 3, dtype=torch.bool),
                         torch.ones(1, 2, dtype=torch.bool))


class MaskedKLGaussianClampTests(unittest.TestCase):

    def test_logvar_clamp_prevents_overflow(self):
        # Without clamp, logvar=100 → exp ≈ 2.7e43 → KL overflows.
        mu = torch.zeros(1, 1, 1, 2)
        logvar = torch.full((1, 1, 1, 2), 100.0)
        coarse_mask = torch.ones(1, 1, dtype=torch.bool)
        frame_mask_lat = torch.ones(1, 1, dtype=torch.bool)
        kl = masked_kl_gaussian(mu, logvar, coarse_mask, frame_mask_lat).item()
        # Should be finite due to clamp
        self.assertTrue(torch.isfinite(torch.tensor(kl)))


class MaskedKLGaussianTests(unittest.TestCase):

    def test_kl_zero_at_unit_gaussian(self):
        # mu=0, logvar=0 (sigma=1) → KL = 0
        mu = torch.zeros(1, 2, 3, 4)
        logvar = torch.zeros(1, 2, 3, 4)
        coarse_mask = torch.ones(1, 3, dtype=torch.bool)
        frame_mask_lat = torch.ones(1, 2, dtype=torch.bool)
        kl = masked_kl_gaussian(mu, logvar, coarse_mask, frame_mask_lat).item()
        self.assertAlmostEqual(kl, 0.0, places=5)

    def test_kl_positive(self):
        # mu=1, logvar=0 → KL = 0.5 * mu^2 * D = 0.5 * 1 * 4 = 2.0 per node
        mu = torch.ones(1, 1, 1, 4)
        logvar = torch.zeros(1, 1, 1, 4)
        coarse_mask = torch.ones(1, 1, dtype=torch.bool)
        frame_mask_lat = torch.ones(1, 1, dtype=torch.bool)
        kl = masked_kl_gaussian(mu, logvar, coarse_mask, frame_mask_lat).item()
        self.assertAlmostEqual(kl, 2.0, places=5)


class MaskedVelConsistencyTests(unittest.TestCase):

    def test_consistent_position_velocity(self):
        # If pos[t+1] - pos[t] == vel[t] / fps, consistency = 0
        T, J = 3, 2
        pred_pos = torch.tensor([[
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        ]])
        # pos diff per frame = 1.0 in x; fps=1 → numerical_vel = 1.0
        pred_vel = torch.zeros(1, T, J, 3)
        pred_vel[..., 0] = 1.0  # all vel x = 1
        fps = torch.tensor([1.0])
        jm = torch.ones(1, J, dtype=torch.bool)
        fm = torch.ones(1, T, dtype=torch.bool)
        loss = masked_vel_consistency(pred_pos, pred_vel, fps, jm, fm).item()
        self.assertAlmostEqual(loss, 0.0, places=5)


class MaskedBoneLengthMaskTests(unittest.TestCase):
    def test_padded_child_skipped(self):
        # Joint 2 padded but listed in parents → must not contribute.
        # parent_indices length must match joint_mask.sum() = 2 → only [-1, 0].
        # We test: parents shorter than J_max, padded joints fail-safe.
        pred_pos = torch.zeros(1, 1, 3, 3)
        pred_pos[0, 0, 2, 0] = 100.0  # joint 2 way off — would add huge loss
        rest_bone_lengths = torch.tensor([[0.0, 1.0, 1.0]])
        # Only 2 joints valid → parent_indices length 2
        parent_indices = [[-1, 0]]
        jm = torch.tensor([[True, True, False]])
        fm = torch.ones(1, 1, dtype=torch.bool)
        # Joint 1 at (0, 0, 0), parent 0 at (0, 0, 0) → bone len 0, rest 1.0 → err 1
        # Joint 2 NOT in parents list (length 2), but if it were, would skip via joint_mask
        loss = masked_bone_length(pred_pos, None, rest_bone_lengths,
                                  parent_indices, jm, fm).item()
        # Only j=1 contributes (parent=0); bone len = 0, rest = 1 → err = 1.0
        self.assertAlmostEqual(loss, 1.0, places=5)


class MaskedBoneLengthTests(unittest.TestCase):

    def test_zero_when_match_rest(self):
        # Line skeleton J=3 with parents [-1, 0, 1], rest bone length 1.0
        pred_pos = torch.zeros(1, 2, 3, 3)
        pred_pos[0, :, 1, 0] = 1.0  # joint 1 at (1, 0, 0)
        pred_pos[0, :, 2, 0] = 2.0  # joint 2 at (2, 0, 0)
        # Bone lengths: joint 1 → parent 0 distance = 1; joint 2 → parent 1 = 1
        rest_bone_lengths = torch.tensor([[0.0, 1.0, 1.0]])
        parent_indices = [[-1, 0, 1]]
        jm = torch.ones(1, 3, dtype=torch.bool)
        fm = torch.ones(1, 2, dtype=torch.bool)
        loss = masked_bone_length(pred_pos, None, rest_bone_lengths,
                                  parent_indices, jm, fm).item()
        self.assertAlmostEqual(loss, 0.0, places=5)

    def test_known_error(self):
        pred_pos = torch.zeros(1, 1, 2, 3)
        pred_pos[0, 0, 1, 0] = 2.0  # joint 1 at distance 2 from root
        rest_bone_lengths = torch.tensor([[0.0, 1.0]])  # expected length 1
        parent_indices = [[-1, 0]]
        jm = torch.ones(1, 2, dtype=torch.bool)
        fm = torch.ones(1, 1, dtype=torch.bool)
        loss = masked_bone_length(pred_pos, None, rest_bone_lengths,
                                  parent_indices, jm, fm).item()
        # |2.0 - 1.0| = 1.0, 1 edge × 1 frame = 1 entry → mean = 1.0
        self.assertAlmostEqual(loss, 1.0, places=5)


class AggregatePoolAuxTests(unittest.TestCase):

    def test_sum_with_defaults(self):
        pool_aux = {
            "mincut": torch.tensor(0.5),
            "locality": torch.tensor(0.3),
            "entropy": torch.tensor(0.1),
        }
        # Default weights: mincut=1, locality=1, entropy=0
        total = aggregate_pool_aux(pool_aux).item()
        self.assertAlmostEqual(total, 0.5 + 0.3, places=5)

    def test_custom_weights(self):
        pool_aux = {"mincut": torch.tensor(1.0), "locality": torch.tensor(2.0)}
        total = aggregate_pool_aux(pool_aux, {"mincut": 0.5, "locality": 0.25}).item()
        self.assertAlmostEqual(total, 0.5 + 0.5, places=5)

    def test_nan_in_aux_raises(self):
        pool_aux = {"mincut": torch.tensor(float("nan"))}
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            aggregate_pool_aux(pool_aux, {"mincut": 1.0})


class ComputeTotalLossTests(unittest.TestCase):

    def test_zero_pred_gt_equal(self):
        B, T, J, D = 1, 4, 3, 4
        T_lat, C = 2, 2
        pred_pos = torch.zeros(B, T, J, 3)
        gt_pos = torch.zeros(B, T, J, 3)
        pred_vel = torch.zeros(B, T, J, 3)
        gt_vel = torch.zeros(B, T, J, 3)
        mu = torch.zeros(B, T_lat, C, D)
        logvar = torch.zeros(B, T_lat, C, D)
        jm = torch.ones(B, J, dtype=torch.bool)
        fm = torch.ones(B, T, dtype=torch.bool)
        cm = torch.ones(B, C, dtype=torch.bool)
        fm_lat = torch.ones(B, T_lat, dtype=torch.bool)
        rest_bones = torch.zeros(B, J)
        parents = [[-1, 0, 1]]
        fps = torch.tensor([1.0])

        losses = compute_total_loss(
            pred_pos=pred_pos, gt_pos=gt_pos,
            pred_vel=pred_vel, gt_vel=gt_vel,
            mu=mu, logvar=logvar,
            pool_aux_outputs=None,
            joint_mask=jm, frame_mask=fm,
            coarse_mask=cm, frame_mask_lat=fm_lat,
            rest_bone_lengths=rest_bones,
            parent_indices=parents,
            fps=fps,
        )
        self.assertIn("total", losses)
        self.assertEqual(losses["pos"].item(), 0.0)
        self.assertEqual(losses["vel"].item(), 0.0)
        # KL at N(0,1) → 0
        self.assertAlmostEqual(losses["kl"].item(), 0.0, places=5)
        # Bone lengths all 0 with all-zero positions → loss 0
        self.assertEqual(losses["bone"].item(), 0.0)

    def test_total_combines_with_weights(self):
        B, T, J = 1, 2, 2
        pred_pos = torch.ones(B, T, J, 3)
        gt_pos = torch.zeros(B, T, J, 3)
        # pos loss = 3.0 (per test_known_difference)
        # Use total with only pos weight set
        losses = compute_total_loss(
            pred_pos=pred_pos, gt_pos=gt_pos,
            pred_vel=torch.zeros(B, T, J, 3), gt_vel=torch.zeros(B, T, J, 3),
            mu=torch.zeros(B, 1, 1, 4), logvar=torch.zeros(B, 1, 1, 4),
            pool_aux_outputs=None,
            joint_mask=torch.ones(B, J, dtype=torch.bool),
            frame_mask=torch.ones(B, T, dtype=torch.bool),
            coarse_mask=torch.ones(B, 1, dtype=torch.bool),
            frame_mask_lat=torch.ones(B, 1, dtype=torch.bool),
            rest_bone_lengths=torch.zeros(B, J),
            parent_indices=[[-1, 0]],
            fps=torch.tensor([1.0]),
            weights={"pos": 1.0, "vel": 0.0, "vel_consistency": 0.0,
                    "kl": 0.0, "bone": 0.0, "pool_aux": 0.0},
        )
        # Only pos contributes; pos = 3.0
        self.assertAlmostEqual(losses["total"].item(), 3.0, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
