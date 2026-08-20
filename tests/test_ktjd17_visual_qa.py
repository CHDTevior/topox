from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ktjd17.visual_qa import (  # noqa: E402
    PerspectiveCamera,
    _safe_stem,
    _select_diagnostic_frames,
)


class PerspectiveContractTests(unittest.TestCase):
    def test_plus_y_is_screen_up_and_plus_x_is_screen_right(self):
        camera = PerspectiveCamera(
            center_x=0.0, center_y=0.0, camera_z=10.0, focal_px=100.0
        )
        points = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        projected, depth = camera.project(points, width=200, height=200)
        self.assertGreater(projected[1, 0], projected[0, 0])
        self.assertLess(projected[2, 1], projected[0, 1])
        np.testing.assert_allclose(depth, 10.0)

    def test_plus_z_is_toward_camera(self):
        camera = PerspectiveCamera(
            center_x=0.0, center_y=0.0, camera_z=10.0, focal_px=100.0
        )
        _, depth = camera.project(
            np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            width=200,
            height=200,
        )
        self.assertLess(depth[1], depth[0])

    def test_safe_stem_is_bounded_and_stable(self):
        value = "clip/with spaces:" + "x" * 200
        first = _safe_stem(value)
        self.assertEqual(first, _safe_stem(value))
        self.assertLessEqual(len(first), 83)
        self.assertNotIn("/", first)

    def test_diagnostic_frames_include_endpoints_and_worst_tail(self):
        errors = np.zeros(20, dtype=np.float64)
        errors[11] = 3.0
        errors[15] = 2.0
        selected = _select_diagnostic_frames(errors, frame_count=20, count=6)
        self.assertIn(0, selected)
        self.assertIn(19, selected)
        self.assertIn(11, selected)
        self.assertEqual(len(selected), 6)


if __name__ == "__main__":
    unittest.main()
