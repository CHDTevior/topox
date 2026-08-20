from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ktjd17.codec import (  # noqa: E402
    Ktjd17CodecError,
    SmootherConfig,
    decode_column_cont6d,
    direct_decode_positions,
    encode_column_cont6d,
    encode_ktjd17_channels,
    fk_from_global_rotations,
    global_to_local_rotations,
    resample_root_and_local_rotations,
    restore_origin_xz,
    smooth_root_xz,
    timestamp_grid,
    world_velocity,
)
from src.data.ktjd17.decoder import decode_ktjd17  # noqa: E402
from src.data.ktjd17.loader import (  # noqa: E402
    build_model_view,
    crop_full_clip,
    derive_masks,
    load_motion_npz,
    yaw_augment,
    yaw_matrix,
)
from src.data.ktjd17.fixed_qa import (  # noqa: E402
    FixedQaError,
    _validate_topology_distance_bucket,
)
from src.data.ktjd17.prototype import (  # noqa: E402
    PrototypeBuildError,
    _project_failure_manifest_record,
    select_prototype_clips,
)


def _y_rotation(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


class Rot6dTests(unittest.TestCase):
    def test_gold_cases_are_exact(self):
        identity = np.eye(3, dtype=np.float64)
        y90 = _y_rotation(math.pi / 2.0)
        np.testing.assert_allclose(
            encode_column_cont6d(identity),
            np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            atol=1e-12,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            encode_column_cont6d(y90),
            np.asarray([0.0, 0.0, -1.0, 0.0, 1.0, 0.0]),
            atol=1e-12,
            rtol=0.0,
        )

    def test_random_10000_so3_roundtrip(self):
        matrices = Rotation.random(10_000, random_state=20260819).as_matrix()
        decoded = decode_column_cont6d(encode_column_cont6d(matrices))
        error = np.max(np.linalg.norm(decoded - matrices, axis=(-2, -1)))
        self.assertLessEqual(error, 1e-10)

    def test_degenerate_gt_aborts(self):
        with self.assertRaisesRegex(Ktjd17CodecError, "first column"):
            decode_column_cont6d(np.zeros(6, dtype=np.float64))
        with self.assertRaisesRegex(Ktjd17CodecError, "second column"):
            decode_column_cont6d(
                np.asarray([1.0, 0.0, 0.0, 2.0, 0.0, 0.0], dtype=np.float64)
            )


class TimestampResamplingTests(unittest.TestCase):
    def test_exact_fps_bypass_is_bitwise(self):
        roots = np.arange(15, dtype=np.float64).reshape(5, 3)
        local = np.broadcast_to(np.eye(3), (5, 2, 3, 3)).copy()
        output = resample_root_and_local_rotations(
            roots, local, fps_src=30.0, fps_target=30.0
        )
        self.assertEqual(output.mode, "exact_fps_identity_bypass")
        self.assertTrue(np.array_equal(output.root_positions, roots))
        self.assertTrue(np.array_equal(output.local_rotations, local))

    def test_timestamp_grid_does_not_stretch_endpoint(self):
        roots = np.zeros((25, 3), dtype=np.float64)
        roots[:, 0] = timestamp_grid(25, 24.0)
        local = np.broadcast_to(np.eye(3), (25, 1, 3, 3)).copy()
        output = resample_root_and_local_rotations(
            roots, local, fps_src=24.0, fps_target=30.0
        )
        self.assertEqual(len(output.target_times), 31)
        self.assertEqual(float(output.target_times[-1]), 1.0)
        np.testing.assert_allclose(output.root_positions[:, 0], output.target_times)

    def test_slerp_operates_on_local_so3(self):
        roots = np.zeros((2, 3), dtype=np.float64)
        local = np.broadcast_to(np.eye(3), (2, 2, 3, 3)).copy()
        local[1, 1] = _y_rotation(math.pi)
        output = resample_root_and_local_rotations(
            roots, local, fps_src=1.0, fps_target=2.0
        )
        np.testing.assert_allclose(
            output.local_rotations[1, 1], _y_rotation(math.pi / 2.0), atol=1e-12
        )


class EncoderDecoderTests(unittest.TestCase):
    def _fixture(self):
        parents = np.asarray([-1, 0, 1], dtype=np.int64)
        offsets = np.asarray(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        rest_global = np.broadcast_to(np.eye(3), (3, 3, 3)).copy()
        rest_local = rest_global.copy()
        roots = np.asarray(
            [[10.0, 1.0, -4.0], [11.0, 1.0, -3.0], [13.0, 1.0, -2.0]],
            dtype=np.float64,
        )
        local = np.broadcast_to(np.eye(3), (3, 3, 3, 3)).copy()
        local[1, 0] = _y_rotation(0.2)
        local[2, 0] = _y_rotation(0.4)
        local[2, 1] = _y_rotation(-0.1)
        smoother = SmootherConfig()
        encoded = encode_ktjd17_channels(
            parents=parents,
            root_positions=roots,
            local_rotations=local,
            offset_parent_local=offsets,
            R_rest_global=rest_global,
            s_rig=2.0,
            fps_target=30.0,
            smoother=smoother,
            contact_tau_h=0.1,
            contact_tau_v=1.0,
            heading_carrier_joint=0,
            u_forward_local=np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            heading_eps_h=0.05,
        )
        return parents, offsets, rest_global, rest_local, roots, local, encoded

    def test_position_direct_and_origin_are_frame_local(self):
        _, _, _, _, _, _, encoded = self._fixture()
        direct = direct_decode_positions(encoded.motion)
        np.testing.assert_allclose(direct, encoded.positions_clip, atol=1e-12)
        absolute = restore_origin_xz(direct, encoded.origin_xz)
        np.testing.assert_allclose(absolute, encoded.positions_absolute, atol=1e-12)
        edited = encoded.motion.copy()
        edited[1, 2, 0] += 7.0
        changed = direct_decode_positions(edited)
        np.testing.assert_array_equal(changed[0], direct[0])
        np.testing.assert_array_equal(changed[2], direct[2])

    def test_rotation_fk_matches_direct_for_rigid_fixture(self):
        parents, offsets, rest_global, rest_local, _, _, encoded = self._fixture()
        decoded = decode_ktjd17(
            encoded.motion,
            parents=parents,
            R_rest_global=rest_global,
            R_rest_local=rest_local,
            offset_parent_local=offsets,
            rotation_source_kind=np.asarray(["animated_dof"] * 3),
        )
        np.testing.assert_allclose(
            decoded.positions_direct, decoded.positions_fk, atol=1e-12
        )

    def test_fixed_dof_ignores_predicted_d6(self):
        parents, offsets, rest_global, rest_local, _, _, encoded = self._fixture()
        corrupted = encoded.motion.copy()
        corrupted[:, 2, 3:9] = encode_column_cont6d(_y_rotation(1.2))
        decoded = decode_ktjd17(
            corrupted,
            parents=parents,
            R_rest_global=rest_global,
            R_rest_local=rest_local,
            offset_parent_local=offsets,
            rotation_source_kind=np.asarray(
                ["animated_dof", "animated_dof", "fixed_dof"]
            ),
        )
        expected = np.matmul(decoded.global_rotations[:, 1], rest_local[2])
        np.testing.assert_allclose(decoded.global_rotations[:, 2], expected, atol=1e-12)
        self.assertTrue(decoded.fixed_dof_overridden[:, 2].all())

    def test_velocity_tail_contract(self):
        positions = np.asarray(
            [[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[3.0, 0.0, 0.0]]],
            dtype=np.float64,
        )
        velocity = world_velocity(positions, fps=2.0)
        np.testing.assert_array_equal(velocity[:, 0, 0], [2.0, 4.0, 4.0])

    def test_short_smoother_is_deterministic_ols(self):
        values = np.asarray([[0.0, 1.0], [2.0, 4.0], [4.0, 7.0]], dtype=np.float64)
        smoothed, mode = smooth_root_xz(
            values, fps=30.0, config=SmootherConfig()
        )
        self.assertEqual(mode, "ols_line")
        np.testing.assert_allclose(smoothed, values, atol=1e-12)


class LoaderTests(unittest.TestCase):
    def test_masks_are_exact(self):
        masks = derive_masks(
            T_valid=2,
            J_phys=3,
            T_max=4,
            J_max=5,
            parents=np.asarray([-1, 0, 1]),
            rotation_source_kind=np.asarray(
                ["animated_dof", "fixed_dof", "animated_dof"]
            ),
            heading_valid=np.asarray([True, False]),
        )
        np.testing.assert_array_equal(masks.frame_mask, [True, True, False, False])
        np.testing.assert_array_equal(masks.joint_mask, [True, True, True, False, False])
        self.assertTrue(masks.channel_valid_mask[0].all())
        self.assertTrue(masks.channel_valid_mask[1, :13].all())
        self.assertFalse(masks.channel_valid_mask[1, 13:].any())
        self.assertFalse(masks.channel_valid_mask[3:].any())
        np.testing.assert_array_equal(
            masks.rotation_supervised, [True, False, True, False, False]
        )
        np.testing.assert_array_equal(
            masks.fixed_rotation_mask, [False, True, False, False, False]
        )
        np.testing.assert_array_equal(
            masks.child_edge_valid, [False, True, True, False, False]
        )

    def test_crop_changes_only_smooth_root(self):
        motion = np.arange(5 * 2 * 17, dtype=np.float64).reshape(5, 2, 17)
        heading = np.asarray([True] * 5)
        cropped, _ = crop_full_clip(motion, heading, start=2, length=2)
        expected = motion[2:4].copy()
        expected[:, 0, 13:15] -= expected[0, 0, 13:15]
        np.testing.assert_array_equal(cropped, expected)

    def test_yaw_equivariance_and_invalid_heading_zero(self):
        parents = np.asarray([-1, 0], dtype=np.int64)
        rest = np.broadcast_to(np.eye(3), (2, 3, 3)).copy()
        offsets = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        roots = np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.1]], dtype=np.float64)
        local = np.broadcast_to(np.eye(3), (2, 2, 3, 3)).copy()
        encoded = encode_ktjd17_channels(
            parents=parents,
            root_positions=roots,
            local_rotations=local,
            offset_parent_local=offsets,
            R_rest_global=rest,
            s_rig=1.0,
            fps_target=30.0,
            smoother=SmootherConfig(),
            contact_tau_h=0.1,
            contact_tau_v=100.0,
            heading_carrier_joint=0,
            u_forward_local=np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
            heading_eps_h=0.05,
        )
        self.assertFalse(encoded.heading_valid.any())
        rotated = yaw_augment(
            encoded.motion,
            encoded.heading_valid,
            R_rest_global=rest,
            phi=0.7,
        )
        self.assertTrue(np.all(rotated[:, 0, 15:17] == 0.0))
        Y, _, _ = yaw_matrix(0.7)
        source_positions = direct_decode_positions(encoded.motion)
        target_positions = direct_decode_positions(rotated)
        np.testing.assert_allclose(
            target_positions,
            np.einsum("ab,tjb->tja", Y, source_positions),
            atol=1e-12,
        )

    def test_model_view_padding_is_after_normalization(self):
        parents = np.asarray([-1], dtype=np.int64)
        rest = np.eye(3, dtype=np.float64)[None]
        motion = np.zeros((2, 1, 17), dtype=np.float32)
        motion[..., 3:9] = encode_column_cont6d(np.eye(3, dtype=np.float64)).astype(
            np.float32
        )
        view = build_model_view(
            motion,
            np.asarray([True, True]),
            parents=parents,
            R_rest_global=rest,
            rotation_source_kind=np.asarray(["animated_dof"]),
            s_rig=2.0,
            gains=np.ones(3, dtype=np.float64),
            T_max=4,
            J_max=3,
        )
        self.assertEqual(view.motion.shape, (4, 3, 17))
        self.assertEqual(view.motion.dtype, np.float32)
        self.assertTrue(np.all(view.motion[2:] == 0.0))
        self.assertTrue(np.all(view.motion[:, 1:] == 0.0))

    def test_public_npz_load_feeds_model_view_without_external_cast(self):
        motion = np.zeros((2, 1, 17), dtype=np.float32)
        motion[..., 3:9] = encode_column_cont6d(np.eye(3, dtype=np.float64)).astype(
            np.float32
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.npz"
            np.savez(
                path,
                motion=motion,
                heading_valid=np.asarray([True, False]),
                clip_id=np.asarray("clip"),
                rig_id=np.asarray("rig"),
                fps_target=np.asarray(30.0),
                origin_xz=np.zeros(2, dtype=np.float64),
            )
            loaded = load_motion_npz(path, expected_fps_target=30.0)
            self.assertEqual(loaded["motion"].dtype, np.float32)
            view = build_model_view(
                loaded["motion"],
                loaded["heading_valid"],
                parents=np.asarray([-1], dtype=np.int64),
                R_rest_global=np.eye(3, dtype=np.float64)[None],
                rotation_source_kind=np.asarray(["animated_dof"]),
                s_rig=1.0,
                gains=np.ones(3, dtype=np.float64),
                T_max=2,
                J_max=1,
            )
        self.assertEqual(view.motion.dtype, np.float32)
        np.testing.assert_array_equal(view.masks.heading_valid, [True, False])

    def test_public_npz_load_rejects_schema_fps_mismatch(self):
        motion = np.zeros((1, 1, 17), dtype=np.float32)
        motion[..., 3:9] = encode_column_cont6d(np.eye(3, dtype=np.float64)).astype(
            np.float32
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.npz"
            np.savez(
                path,
                motion=motion,
                heading_valid=np.asarray([True]),
                clip_id=np.asarray("clip"),
                rig_id=np.asarray("rig"),
                fps_target=np.asarray(30.0),
                origin_xz=np.zeros(2, dtype=np.float64),
            )
            with self.assertRaisesRegex(Ktjd17CodecError, "!= schema"):
                load_motion_npz(path, expected_fps_target=24.0)

    def test_public_npz_load_rejects_noncanonical_scalar_sidecars(self):
        motion = np.zeros((1, 1, 17), dtype=np.float32)
        motion[..., 3:9] = encode_column_cont6d(np.eye(3, dtype=np.float64)).astype(
            np.float32
        )
        canonical = {
            "motion": motion,
            "heading_valid": np.asarray([True], dtype=np.bool_),
            "clip_id": np.asarray("clip"),
            "rig_id": np.asarray("rig"),
            "fps_target": np.asarray(30.0, dtype=np.float64),
            "origin_xz": np.zeros(2, dtype=np.float64),
        }
        mutations = {
            "fps_string": ("fps_target", np.asarray("30.0"), "float64 scalar"),
            "fps_vector": (
                "fps_target",
                np.asarray([30.0], dtype=np.float64),
                "float64 scalar",
            ),
            "clip_numeric": ("clip_id", np.asarray(123), "Unicode scalar"),
            "clip_vector": (
                "clip_id",
                np.asarray(["clip"]),
                "Unicode scalar",
            ),
            "rig_numeric": ("rig_id", np.asarray(456), "Unicode scalar"),
            "rig_vector": (
                "rig_id",
                np.asarray(["rig"]),
                "Unicode scalar",
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, (key, value, message) in mutations.items():
                with self.subTest(name=name):
                    payload = dict(canonical)
                    payload[key] = value
                    path = Path(directory) / f"{name}.npz"
                    np.savez(path, **payload)
                    with self.assertRaisesRegex(Ktjd17CodecError, message):
                        load_motion_npz(path, expected_fps_target=30.0)


class PrototypeSelectionTests(unittest.TestCase):
    def test_failure_manifest_preserves_topology_distance_bucket(self):
        record = {
            "clip_id": "clip",
            "topology_distance_bucket": "train_seen_topology",
            "status": "reject",
            "error": "synthetic",
        }
        projected = _project_failure_manifest_record(record)
        self.assertEqual(
            projected["topology_distance_bucket"], "train_seen_topology"
        )
        with self.assertRaisesRegex(
            PrototypeBuildError, "missing topology_distance_bucket"
        ):
            _project_failure_manifest_record({"clip_id": "clip"})

    def test_fixed_qa_requires_qa_manifest_parent_bucket_equality(self):
        parent = {"topology_distance_bucket": "train_seen_topology"}
        manifest = {"topology_distance_bucket": "train_seen_topology"}
        qa = {"topology_distance_bucket": "train_seen_topology"}
        self.assertEqual(
            _validate_topology_distance_bucket(
                clip_id="clip", manifest=manifest, qa=qa, parent_clip=parent
            ),
            "train_seen_topology",
        )
        with self.assertRaisesRegex(
            FixedQaError, "QA topology-distance bucket drifted"
        ):
            _validate_topology_distance_bucket(
                clip_id="clip",
                manifest=manifest,
                qa={"topology_distance_bucket": "held_stress_topology"},
                parent_clip=parent,
            )

    @staticmethod
    def _record(
        clip_id: str,
        topology: str,
        rig_id: str,
        *,
        eligible: bool,
        source_family: str = "truebones",
        split: str = "train",
    ) -> dict[str, object]:
        return {
            "clip_id": clip_id,
            "source": {"family": source_family},
            "topology_family": topology,
            "rig_id": rig_id,
            "split": split,
            "status": "review" if eligible else "reject",
            "split_eligible_for_train_calibration": eligible,
            "split_eligible_for_ktjd17_t04": eligible,
            "canonical_skeleton": {"status": "pass" if eligible else "reject"},
        }

    @staticmethod
    def _family_payload(selected: list[str], eligible: list[str]) -> dict[str, object]:
        eligible_set = set(eligible)
        ineligible = [clip_id for clip_id in selected if clip_id not in eligible_set]
        return {
            "required_train_clips": 30,
            "selected_train_candidates": selected,
            "canonical_skeleton_t04": {
                "eligible_selected_train_clips": eligible,
                "eligible_count": len(eligible),
                "ineligible_selected_train_clips": ineligible,
                "shortage": 30 - len(eligible),
                "selection_replaced": False,
            },
        }

    def test_t05_overlay_preserves_parent_and_records_every_replacement(self):
        clips: list[dict[str, object]] = []
        families: dict[str, object] = {}

        human = [f"human_{index:02d}" for index in range(30)]
        clips.extend(
            self._record(
                clip_id,
                "human",
                "HML3D_Human",
                eligible=False,
                source_family="motionstreamer272",
            )
            for clip_id in human
        )
        families["human"] = self._family_payload(human, [])

        specifications = {
            "quadruped": ("Alligator", 2, 28),
            "winged": ("Tukan", 30, 0),
            "spider_crab": ("HermitCrab", 30, 0),
            "dragon_or_deep_topology": ("Horse", 3, 25),
        }
        for family, (rig_id, base_eligible_count, replacement_count) in specifications.items():
            parent = [f"{family}_parent_{index:02d}" for index in range(30)]
            parent_eligible = parent[:base_eligible_count]
            clips.extend(
                self._record(
                    clip_id,
                    family,
                    rig_id,
                    eligible=clip_id in set(parent_eligible),
                )
                for clip_id in parent
            )
            clips.extend(
                self._record(
                    f"{family}_replacement_{index:02d}",
                    family,
                    rig_id,
                    eligible=True,
                )
                for index in range(replacement_count)
            )
            families[family] = self._family_payload(parent, parent_eligible)

        snakes = [f"snake_held_{index:02d}" for index in range(10)]
        clips.extend(
            self._record(
                clip_id,
                "snake",
                "KingCobra",
                eligible=True,
                split="held_representative",
            )
            for clip_id in snakes
        )
        dragons = [f"dragon_held_{index:02d}" for index in range(13)]
        clips.extend(
            self._record(
                clip_id,
                "dragon_or_deep_topology",
                "Dragon",
                eligible=True,
                split="held_stress",
            )
            for clip_id in dragons
        )

        selected, audit = select_prototype_clips(clips, {"families": families})
        self.assertEqual(len(selected), 171)
        self.assertEqual(audit["quadruped"]["calibration_selected"], 30)
        self.assertEqual(len(audit["quadruped"]["replacement_pairs"]), 28)
        self.assertEqual(
            audit["quadruped"]["selected_clips"][:2],
            ["quadruped_parent_00", "quadruped_parent_01"],
        )
        self.assertTrue(audit["quadruped"]["selection_replaced"])
        self.assertEqual(
            audit["dragon_or_deep_topology"]["calibration_shortage"], 2
        )
        self.assertEqual(audit["snake"]["calibration_selected"], 0)
        self.assertEqual(audit["snake"]["calibration_shortage"], 30)
        self.assertTrue(
            all(
                entry.replaces_parent_clip_id is not None
                for entry in selected
                if entry.selection_origin
                == "explicit_t05_replacement_from_pinned_parent_manifest"
            )
        )


if __name__ == "__main__":
    unittest.main()
