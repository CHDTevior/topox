"""Unit tests for per-species KTJD-17 population moments."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ktjd17.species_stats import (  # noqa: E402
    SpeciesAggregate,
    SpeciesStatsError,
    accumulate_motion,
    species_from_rig_id,
)


class SpeciesStatsTests(unittest.TestCase):
    def test_species_grouping(self) -> None:
        self.assertEqual(species_from_rig_id("PZ_Aardvark_Female"), "Aardvark")
        self.assertEqual(species_from_rig_id("PZ_Aardvark_Juvenile"), "Aardvark")
        self.assertEqual(species_from_rig_id("PZ_Aardvark_Male"), "Aardvark")
        self.assertEqual(species_from_rig_id("HML3D_Human"), "Human")
        with self.assertRaises(SpeciesStatsError):
            species_from_rig_id("PZ_Aardvark")

    def test_masked_population_mean_std_matches_numpy(self) -> None:
        first = np.arange(3 * 2 * 17, dtype=np.float32).reshape(3, 2, 17)
        second = (np.arange(2 * 3 * 17, dtype=np.float32).reshape(2, 3, 17) - 7.0)
        first[:, 1:, 13:17] = 0.0
        second[:, 1:, 13:17] = 0.0
        first_heading = np.asarray([True, False, True])
        second_heading = np.asarray([False, True])

        aggregate = SpeciesAggregate("Aardvark")
        accumulate_motion(
            aggregate,
            motion=first,
            heading_valid=first_heading,
            rig_id="PZ_Aardvark_Female",
        )
        accumulate_motion(
            aggregate,
            motion=second,
            heading_valid=second_heading,
            rig_id="PZ_Aardvark_Male",
        )

        expected: list[np.ndarray] = []
        for channel in range(17):
            if channel < 13:
                values = np.concatenate(
                    [first[..., channel].reshape(-1), second[..., channel].reshape(-1)]
                )
            elif channel < 15:
                values = np.concatenate([first[:, 0, channel], second[:, 0, channel]])
            else:
                values = np.concatenate(
                    [
                        first[first_heading, 0, channel],
                        second[second_heading, 0, channel],
                    ]
                )
            expected.append(values.astype(np.float64))
        np.testing.assert_allclose(
            aggregate.moments.mean,
            np.asarray([np.mean(values) for values in expected]),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            aggregate.moments.population_std(),
            np.asarray([np.std(values, ddof=0) for values in expected]),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_array_equal(
            aggregate.moments.count,
            np.asarray([12] * 13 + [5, 5, 3, 3], dtype=np.int64),
        )
        self.assertEqual(aggregate.clip_count, 2)
        self.assertEqual(aggregate.frame_count, 5)
        self.assertEqual(aggregate.physical_joint_frame_count, 12)
        self.assertEqual(aggregate.heading_valid_frame_count, 3)
        self.assertEqual(
            set(aggregate.rig_moments),
            {"PZ_Aardvark_Female", "PZ_Aardvark_Male"},
        )
        female = aggregate.rig_moments["PZ_Aardvark_Female"].finalized()
        female_count = np.asarray(female["count"])
        expected_count = np.zeros((2, 17), dtype=np.int64)
        expected_count[:, :13] = 3
        expected_count[0, 13:15] = 3
        expected_count[0, 15:17] = 2
        np.testing.assert_array_equal(female_count, expected_count)
        for joint in range(2):
            for channel in range(17):
                if channel < 13:
                    cell = first[:, joint, channel].astype(np.float64)
                elif joint == 0 and channel < 15:
                    cell = first[:, 0, channel].astype(np.float64)
                elif joint == 0:
                    cell = first[first_heading, 0, channel].astype(np.float64)
                else:
                    cell = np.asarray([], dtype=np.float64)
                if cell.size:
                    self.assertAlmostEqual(
                        float(np.asarray(female["mean"])[joint, channel]),
                        float(np.mean(cell)),
                    )
                    self.assertAlmostEqual(
                        float(np.asarray(female["std"])[joint, channel]),
                        float(np.std(cell, ddof=0)),
                    )
                else:
                    self.assertEqual(
                        float(np.asarray(female["std"])[joint, channel]), 0.0
                    )

    def test_nonroot_root_only_values_are_rejected(self) -> None:
        motion = np.zeros((2, 2, 17), dtype=np.float32)
        motion[:, 1, 13] = 1.0
        with self.assertRaisesRegex(SpeciesStatsError, "non-root"):
            accumulate_motion(
                SpeciesAggregate("Aardvark"),
                motion=motion,
                heading_valid=np.ones(2, dtype=bool),
                rig_id="PZ_Aardvark_Female",
            )


if __name__ == "__main__":
    unittest.main()
