from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ktjd17.calibration import (  # noqa: E402
    CalibrationError,
    derive_position_anchor_heading,
    gains_from_sums,
    position_anchor_heading_errors,
    summarize_distribution,
)


class CalibrationPrimitiveTests(unittest.TestCase):
    def test_reciprocal_rms_gains(self):
        gains, rms = gains_from_sums(
            {"q": 8.0, "v": 18.0, "s": 4.0},
            {"q": 2, "v": 2, "s": 4},
        )
        np.testing.assert_allclose(rms, [2.0, 3.0, 1.0])
        np.testing.assert_allclose(gains, [0.5, 1.0 / 3.0, 1.0])
        with self.assertRaises(CalibrationError):
            gains_from_sums(
                {"q": 8.0, "v": 18.0, "s": 0.0},
                {"q": 2, "v": 2, "s": 4},
            )

    def test_position_anchor_heading_gold_and_opposite(self):
        positions = np.zeros((2, 4, 3), dtype=np.float64)
        positions[:, 0] = [-1.0, 0.0, 0.0]
        positions[:, 1] = [1.0, 0.0, 0.0]
        positions[:, 2] = [-1.0, 1.0, 0.0]
        positions[:, 3] = [1.0, 1.0, 0.0]
        heading = np.asarray([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float64)
        errors, comparable, horizontal = position_anchor_heading_errors(
            positions,
            heading,
            np.ones(2, dtype=bool),
            method="lateral_pairs",
            anchor_indices=[0, 1, 2, 3],
            s_rig=2.0,
        )
        np.testing.assert_allclose(errors, [0.0, math.pi], atol=1e-12)
        self.assertTrue(np.all(comparable))
        self.assertTrue(np.all(horizontal > 0.0))
        derived, valid, _ = derive_position_anchor_heading(
            positions,
            method="lateral_pairs",
            anchor_indices=[0, 1, 2, 3],
            s_rig=2.0,
        )
        np.testing.assert_allclose(derived, [[1.0, 0.0], [1.0, 0.0]])
        self.assertTrue(np.all(valid))

    def test_distribution_rejects_nonfinite(self):
        summary = summarize_distribution([1.0, 2.0, 3.0])
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["median"], 2.0)
        with self.assertRaises(CalibrationError):
            summarize_distribution([1.0, np.nan])


if __name__ == "__main__":
    unittest.main()
