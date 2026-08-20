"""CPU gold fixtures for the KTJD-17 T03 source parsers."""

from __future__ import annotations

import dataclasses
import os
import sys
import tempfile
import unittest
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ktjd17.source_parser import (  # noqa: E402
    MOTIONSTREAMER272_JOINTS,
    SourceParserError,
    decode_source_row_cont6d,
    forward_kinematics,
    parse_bvh_numeric,
    parse_bvh_source,
    parse_motionstreamer272_source,
    require_source_fk_pass,
    source_fk_metrics,
)
from src.data.ktjd17.bvh_inventory import parse_bvh_header  # noqa: E402
from src.data.ktjd17.encoder import (  # noqa: E402
    EncoderConfig,
    Ktjd17EncoderError,
    encode_prepared_motion,
    load_skeleton,
    prepare_manifest_clip,
    write_npz_atomic,
)
from src.data.ktjd17.codec import SmootherConfig  # noqa: E402
from src.data.ktjd17.fixed_qa import (  # noqa: E402
    SOURCE_FK_MAX_NORM,
    _validate_motion,
)
from src.data.ktjd17.planetzoo_fixed_rig import (  # noqa: E402
    PlanetzooFixedRigError,
    build_planetzoo_fixed_rig,
)
from src.data.ktjd17.source_fk import _build_summary  # noqa: E402
from src.data.ktjd17.source_fk_validation import (  # noqa: E402
    SourceFkValidationError,
    write_source_fk_validation_report,
)
from src.data.ktjd17.truebones_fixed_rig import (  # noqa: E402
    ACTIVE_COND_SHA256,
    TRUEBONES_BTJD_MEAN_EDGE_TARGET,
    FixedRigGeometry,
    ForwardSpec,
    TruebonesFixedRigError,
    _fixed_geometry,
    build_fixed_rig_motion,
)


def _row_d6(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=np.float64)[..., :2, :].reshape(
        np.asarray(matrix).shape[:-2] + (6,)
    )


def _bvh(rows: list[list[float]], *, invalid_token: str | None = None) -> str:
    rendered = [" ".join(str(value) for value in row) for row in rows]
    if invalid_token is not None:
        fields = rendered[0].split()
        fields[-1] = invalid_token
        rendered[0] = " ".join(fields)
    return f"""HIERARCHY
ROOT Wrapper
{{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT Root
  {{
    OFFSET 0 0 0
    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
    JOINT Bone
    {{
      OFFSET 0 1 0
      CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
      End Site
      {{
        OFFSET 0 1 0
      }}
    }}
  }}
}}
MOTION
Frames: {len(rows)}
Frame Time: 0.03333333333333333
{chr(10).join(rendered)}
"""


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _pz_bvh(*, chest_offset_z: float = 1.0) -> str:
    return f"""HIERARCHY
ROOT def_c_root_joint
{{
  OFFSET 0 1 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT def_c_hips_joint
  {{
    OFFSET 0 -0.5 0
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT def_c_chest_joint
    {{
      OFFSET 0 0 {chest_offset_z}
      CHANNELS 3 Zrotation Xrotation Yrotation
    }}
  }}
}}
MOTION
Frames: 2
Frame Time: 0.041667
0 1 0 0 0 0 0 0 0 0 0 0
0.2 1 0.3 0 0 15 0 5 0 0 0 0
"""


class _NoTposeConditioning(Mapping[str, object]):
    def __init__(self, payload: dict[str, object]):
        self._payload = payload

    def __getitem__(self, key: str) -> object:
        if key == "tpos_first_frame":
            raise AssertionError("forbidden tpos_first_frame access")
        return self._payload[key]

    def __iter__(self) -> Iterator[str]:
        return iter((*self._payload, "tpos_first_frame"))

    def __len__(self) -> int:
        return len(self._payload) + 1


class SourceRow6dTests(unittest.TestCase):
    def test_identity_and_positive_y_quarter_turn(self):
        identity = np.eye(3, dtype=np.float64)
        positive_y = np.asarray(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        decoded = decode_source_row_cont6d(
            np.stack((_row_d6(identity), _row_d6(positive_y)))
        )
        np.testing.assert_allclose(decoded[0], identity, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(decoded[1], positive_y, atol=1e-12, rtol=0.0)
        self.assertEqual(decoded.dtype, np.float64)

    def test_degenerate_source_6d_fails_closed(self):
        with self.assertRaisesRegex(SourceParserError, "degenerate"):
            decode_source_row_cont6d(np.zeros((1, 6), dtype=np.float64))


class BvhNumericParserTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            # Wrapper, retained Root, retained Bone.  Every block is XYZ
            # translation followed by declared intrinsic Z-X-Y Euler angles.
            [1, 2, 3, 10, 20, 30, 0, 0, 0, 40, 50, 60, 0, 1, 0, 70, 80, 90],
            [2, 3, 4, -5, 10, 15, 0, 0, 0, 20, 30, 40, 0, 1, 0, 5, 15, 25],
        ]

    def test_declared_euler_order_matches_scipy_intrinsic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion.bvh"
            _write(path, _bvh(self.rows))
            parsed = parse_bvh_numeric(path)
            expected = Rotation.from_euler(
                "ZXY", np.asarray([40.0, 50.0, 60.0]), degrees=True
            ).as_matrix()
            np.testing.assert_allclose(
                parsed.local_rotations[0, 1], expected, atol=1e-12, rtol=0.0
            )
            self.assertEqual(parsed.local_positions.dtype, np.float64)
            self.assertEqual(parsed.local_rotations.dtype, np.float64)

    def test_wrapper_reroot_and_legacy_unnamed_end_site_map(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            motion = root / "motion.bvh"
            rest = root / "rest.bvh"
            _write(motion, _bvh(self.rows))
            _write(rest, _bvh([self.rows[0]]))
            parsed = parse_bvh_source(
                motion,
                retained_names=("Root", "Bone", "Bone_end_site"),
                retained_parents=(-1, 0, 1),
                expected_rotation_kinds=(
                    "animated_dof",
                    "animated_dof",
                    "fixed_dof",
                ),
                frame_slice=(0, 2),
                rest_path=rest,
                rest_mode="explicit_tpose_frame",
                family="truebones",
            )
            self.assertEqual(parsed.source_joint_indices.tolist(), [1, 2, 3])
            self.assertEqual(parsed.root_translation.shape, (2, 3))
            self.assertEqual(parsed.local_rotations.shape, (2, 3, 3, 3))
            self.assertLess(source_fk_metrics(parsed)["source_parser_fk_max_norm"], 1e-12)

    def test_invalid_numeric_token_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.bvh"
            _write(path, _bvh(self.rows, invalid_token="not-a-number"))
            with self.assertRaisesRegex(SourceParserError, "every BVH numeric token"):
                parse_bvh_numeric(path)


class ForwardKinematicsTests(unittest.TestCase):
    def test_active_column_vector_chain(self):
        parents = np.asarray([-1, 0], dtype=np.int64)
        local_positions = np.asarray([[[1, 0, 0], [0, 1, 0]]], dtype=np.float64)
        root_rotation = Rotation.from_euler("Z", 90, degrees=True).as_matrix()
        local_rotations = np.stack(
            (root_rotation, np.eye(3, dtype=np.float64)), axis=0
        )[None]
        positions, _ = forward_kinematics(
            parents, local_positions, local_rotations
        )
        np.testing.assert_allclose(
            positions[0], [[1, 0, 0], [0, 0, 0]], atol=1e-12, rtol=0.0
        )


class TruebonesFixedRigTests(unittest.TestCase):
    @staticmethod
    def _fixture(
        nonroot_xyz_count: int = 1,
    ) -> tuple[SimpleNamespace, FixedRigGeometry]:
        identity = np.eye(3, dtype=np.float64)
        source_positions = np.asarray(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.5, 0.0, 0.0], [10.0, 4.0, -3.0]],
            ],
            dtype=np.float64,
        )
        rotations = np.broadcast_to(identity, (2, 2, 3, 3)).copy()
        parsed = SimpleNamespace(
            joint_names=("Root", "Bone"),
            parents=np.asarray([-1, 0], dtype=np.int64),
            rest_global_positions=np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64
            ),
            rest_global_rotations=np.broadcast_to(identity, (2, 3, 3)).copy(),
            rest_local_rotations=np.broadcast_to(identity, (2, 3, 3)).copy(),
            global_rotations=rotations,
            local_rotations=rotations.copy(),
            source_positions=source_positions,
            root_translation=source_positions[:, 0].copy(),
            rest_status="explicit_tpose_frame",
            diagnostics={
                "nonroot_position_channel_joint_count": nonroot_xyz_count,
                "nonroot_position_channel_sample_count": 2 * nonroot_xyz_count,
                "nonroot_position_channel_max_frame_variation_norm": 9.0,
            },
        )
        target = TRUEBONES_BTJD_MEAN_EDGE_TARGET
        fixed = FixedRigGeometry(
            rig_id="Fixture",
            joint_names=("Root", "Bone"),
            parents=np.asarray([-1, 0], dtype=np.int64),
            offsets=np.asarray(
                [[0.0, 0.0, 0.0], [0.0, 0.0, target]], dtype=np.float64
            ),
            rest_positions=np.asarray(
                [[0.0, 0.0, 0.0], [0.0, 0.0, target]], dtype=np.float64
            ),
            ground_shift_y=0.0,
            payload_sha256="fixture",
            metrics={
                "cond_offsets_to_tpos_max_abs": 0.0,
                "cond_offsets_to_tpos_max_norm": 0.0,
                "cond_ground_shift_y": 0.0,
                "cond_ground_min_y_abs": 0.0,
                "cond_root_xz_max_abs": 0.0,
                "cond_mean_nonroot_edge_length": target,
                "cond_mean_edge_target_abs_error": 0.0,
            },
        )
        return parsed, fixed

    def test_nonroot_xyz_is_diagnostic_not_authoritative(self):
        parsed, fixed = self._fixture()
        result = build_fixed_rig_motion(
            parsed,
            fixed,
            ForwardSpec("declared_plus_z", tuple(), "fixture"),
        )
        expected_child = result.P_authoritative[:, 0] + np.asarray(
            [0.0, 0.0, TRUEBONES_BTJD_MEAN_EDGE_TARGET]
        )
        np.testing.assert_allclose(result.P_authoritative[:, 1], expected_child)
        self.assertLess(result.metrics["motion_rigid_edge_max_norm"], 1e-12)
        self.assertGreater(result.metrics["raw_xyz_vs_authoritative_max_norm"], 1.0)
        self.assertFalse(result.provenance["forbidden_inputs_used"])

    def test_missing_ignored_xyz_audit_fails_closed(self):
        parsed, fixed = self._fixture(nonroot_xyz_count=0)
        with self.assertRaisesRegex(
            TruebonesFixedRigError, "did not enumerate ignored non-root XYZ"
        ):
            build_fixed_rig_motion(
                parsed,
                fixed,
                ForwardSpec("declared_plus_z", tuple(), "fixture"),
            )

    def test_cond_rotation_like_channels_are_hashed_but_not_geometry(self):
        target = TRUEBONES_BTJD_MEAN_EDGE_TARGET
        tpose = np.zeros((2, 13), dtype=np.float64)
        tpose[1, 2] = target
        base = {
            "joints_names": np.asarray(["Root", "Bone"]),
            "parents": np.asarray([-1, 0], dtype=np.int64),
            "offsets": np.asarray(
                [[0.0, 0.0, 0.0], [0.0, 0.0, target]], dtype=np.float64
            ),
            "tpos_first_frame": tpose,
        }
        changed = {**base, "tpos_first_frame": tpose.copy()}
        changed["tpos_first_frame"][1, 3:9] = [0.2, -0.3, 0.4, 0.5, -0.6, 0.7]
        first = _fixed_geometry(
            "Fixture",
            base,
            expected_names=("Root", "Bone"),
            expected_parents=(-1, 0),
        )
        second = _fixed_geometry(
            "Fixture",
            changed,
            expected_names=("Root", "Bone"),
            expected_parents=(-1, 0),
        )
        np.testing.assert_array_equal(first.rest_positions, second.rest_positions)
        np.testing.assert_array_equal(first.offsets, second.offsets)
        self.assertNotEqual(first.payload_sha256, second.payload_sha256)


class PlanetzooFixedRigTests(unittest.TestCase):
    @staticmethod
    def _rig(path: Path) -> dict[str, object]:
        header = parse_bvh_header(path)
        names = list(header.joint_names)
        parents = list(header.parents)
        return {
            "rig_id": "PZ_Fixture",
            "source_family": "planetzoo",
            "topology_family": "quadruped",
            "rest_pose": {
                "source_path": str(path),
                "selection_method": "first_sorted_processed_clip",
                "rest_layout_sha256": header.rest_layout_sha256(),
            },
            "joint_map": {
                "source_joint_names": names,
                "source_parents": parents,
                "source_node_kinds": ["joint"] * 3,
                "source_rotation_layout_sha256": header.rotation_layout_sha256(),
                "source_root_index_for_btjd_root": 0,
                "btjd_joint_names": names,
                "btjd_parents": parents,
                "btjd_to_source": [0, 1, 2],
                "rotation_source_kind": ["animated_dof"] * 3,
                "direct_source_edge_count": 2,
                "source_skipping_edge_count": 0,
                "animated_dof_count": 3,
                "fixed_dof_count": 0,
                "mapping_kind": "exact_joint_name_with_source_ancestry_check",
                "joint_map_sha256": "fixture",
            },
        }

    @staticmethod
    def _cond() -> _NoTposeConditioning:
        return _NoTposeConditioning(
            {
                "joints_names": np.asarray(
                    [
                        "def_c_root_joint",
                        "def_c_hips_joint",
                        "def_c_chest_joint",
                    ]
                ),
                "parents": np.asarray([-1, 0, 1], dtype=np.int64),
                "offsets": np.asarray(
                    [[0.0, 1.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, 1.0]],
                    dtype=np.float32,
                ),
            }
        )

    def test_stage2_builder_never_reads_cond_tpose_and_encoder_closes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "PZ_Fixture_motion.bvh"
            _write(source, _pz_bvh())
            rig = self._rig(source)
            fixed = build_planetzoo_fixed_rig(
                rig_record=rig,
                representative_bvh_path=source,
                pinned_source_root=root,
                cond_entry=self._cond(),
                active_cond_sha256=ACTIVE_COND_SHA256,
            )
            skeleton_path = root / "PZ_Fixture.npz"
            write_npz_atomic(skeleton_path, fixed.payload)
            skeleton = load_skeleton(skeleton_path)
            clip = {
                "clip_id": source.stem,
                "rig_id": "PZ_Fixture",
                "source": {
                    "family": "planetzoo",
                    "path": str(source),
                    "slice_frames": [0, 2],
                },
            }
            prepared = prepare_manifest_clip(
                clip, rig, skeleton, conditioning_catalog=None
            )
            self.assertEqual(prepared.root_positions.shape, (2, 3))
            self.assertLess(
                prepared.source_parser_metrics[
                    "planetzoo_fixed_fk_source_position_max_norm"
                ],
                1e-12,
            )
            self.assertFalse(
                skeleton.metadata["position_geometry_provenance"][
                    "cond_tpos_first_frame_used"
                ]
            )

    def test_full_fixed_qa_planetzoo_path_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "PZ_Fixture_motion.bvh"
            _write(source, _pz_bvh())
            rig = self._rig(source)
            fixed = build_planetzoo_fixed_rig(
                rig_record=rig,
                representative_bvh_path=source,
                pinned_source_root=root,
                cond_entry=self._cond(),
                active_cond_sha256=ACTIVE_COND_SHA256,
            )
            skeleton_path = root / "PZ_Fixture.npz"
            write_npz_atomic(skeleton_path, fixed.payload)
            skeleton = load_skeleton(skeleton_path)
            clip = {
                "clip_id": source.stem,
                "rig_id": "PZ_Fixture",
                "topology_distance_bucket": "train_seen_topology",
                "source": {
                    "family": "planetzoo",
                    "path": str(source),
                    "slice_frames": [0, 2],
                },
            }
            prepared = prepare_manifest_clip(
                clip, rig, skeleton, conditioning_catalog=None
            )
            config = EncoderConfig(
                fps_target=prepared.fps_src,
                smoother=SmootherConfig(),
                contact_tau_h=0.1,
                contact_tau_v=1.0,
                heading_eps_h=1e-6,
            )
            encoded = encode_prepared_motion(prepared, skeleton, config)
            motion_path = root / "motions" / f"{source.stem}.npz"
            motion_sha = write_npz_atomic(motion_path, encoded.artifact_payload())
            manifest = {
                "clip_id": source.stem,
                "rig_id": "PZ_Fixture",
                "motion_relpath": f"motions/{source.stem}.npz",
                "motion_sha256": motion_sha,
                "T_target": int(encoded.motion_float32.shape[0]),
                "J_phys": int(encoded.motion_float32.shape[1]),
                "source_family": "planetzoo",
                "resample_mode": encoded.resample_mode,
                "topology_family": "quadruped",
                "topology_distance_bucket": "train_seen_topology",
                "family_role": "prototype",
                "split": "train",
                "calibration_eligible": False,
            }
            result = _validate_motion(
                root=root,
                manifest=manifest,
                qa={
                    "motion_sha256": motion_sha,
                    "topology_distance_bucket": "train_seen_topology",
                },
                parent_clip=clip,
                parent_rig=rig,
                skeleton=skeleton,
                encoder_config=config.as_record(),
                rest_cache={},
            )
            self.assertEqual(result["status"], "pass")
            self.assertEqual(SOURCE_FK_MAX_NORM["planetzoo"], 1e-10)

    def test_loaded_planetzoo_semantic_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "PZ_Fixture_motion.bvh"
            _write(source, _pz_bvh())
            rig = self._rig(source)
            fixed = build_planetzoo_fixed_rig(
                rig_record=rig,
                representative_bvh_path=source,
                pinned_source_root=root,
                cond_entry=self._cond(),
                active_cond_sha256=ACTIVE_COND_SHA256,
            )
            skeleton_path = root / "PZ_Fixture.npz"
            write_npz_atomic(skeleton_path, fixed.payload)
            skeleton = load_skeleton(skeleton_path)
            clip = {
                "clip_id": source.stem,
                "rig_id": "PZ_Fixture",
                "source": {
                    "family": "planetzoo",
                    "path": str(source),
                    "slice_frames": [0, 2],
                },
            }
            global_rest = skeleton.R_rest_global.copy()
            global_rest[0] = Rotation.from_euler("Y", 5.0, degrees=True).as_matrix()
            local_rest = skeleton.R_rest_local.copy()
            local_rest[1] = Rotation.from_euler("Y", 5.0, degrees=True).as_matrix()
            mutations = {
                "source_family": {"source_family": "truebones"},
                "artifact_status": {"artifact_status": "unreviewed"},
                "global_rest": {"R_rest_global": global_rest},
                "local_rest": {"R_rest_local": local_rest},
                "heading_carrier": {"heading_carrier_joint": 1},
                "forward_axis": {
                    "u_forward_local": np.asarray(
                        [0.0, 0.0, -1.0], dtype=np.float64
                    )
                },
            }
            for name, changes in mutations.items():
                with self.subTest(name=name):
                    drifted = dataclasses.replace(skeleton, **changes)
                    with self.assertRaises(
                        (PlanetzooFixedRigError, Ktjd17EncoderError)
                    ):
                        prepare_manifest_clip(
                            clip, rig, drifted, conditioning_catalog=None
                        )

    def test_per_clip_offset_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "PZ_Fixture_reference.bvh"
            drifted = root / "PZ_Fixture_drifted.bvh"
            _write(reference, _pz_bvh())
            _write(drifted, _pz_bvh(chest_offset_z=1.1))
            rig = self._rig(reference)
            fixed = build_planetzoo_fixed_rig(
                rig_record=rig,
                representative_bvh_path=reference,
                pinned_source_root=root,
                cond_entry=self._cond(),
                active_cond_sha256=ACTIVE_COND_SHA256,
            )
            skeleton_path = root / "PZ_Fixture.npz"
            write_npz_atomic(skeleton_path, fixed.payload)
            skeleton = load_skeleton(skeleton_path)
            clip = {
                "clip_id": drifted.stem,
                "rig_id": "PZ_Fixture",
                "source": {
                    "family": "planetzoo",
                    "path": str(drifted),
                    "slice_frames": [0, 2],
                },
            }
            with self.assertRaisesRegex(
                PlanetzooFixedRigError, "rest_layout_sha256 drifted"
            ):
                prepare_manifest_clip(clip, rig, skeleton, conditioning_catalog=None)

    def test_stage2_builder_rejects_symlink_or_out_of_root_representative(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            pinned = root / "pinned"
            pinned.mkdir()
            target = pinned / "PZ_Fixture_target.bvh"
            link = pinned / "PZ_Fixture_link.bvh"
            _write(target, _pz_bvh())
            os.symlink(target.name, link)
            rig = self._rig(link)
            with self.assertRaisesRegex(PlanetzooFixedRigError, "symlink"):
                build_planetzoo_fixed_rig(
                    rig_record=rig,
                    representative_bvh_path=link,
                    pinned_source_root=pinned,
                    cond_entry=self._cond(),
                    active_cond_sha256=ACTIVE_COND_SHA256,
                )
            outside = root / "PZ_Fixture_outside.bvh"
            _write(outside, _pz_bvh())
            rig = self._rig(outside)
            with self.assertRaisesRegex(PlanetzooFixedRigError, "direct child"):
                build_planetzoo_fixed_rig(
                    rig_record=rig,
                    representative_bvh_path=outside,
                    pinned_source_root=pinned,
                    cond_entry=self._cond(),
                    active_cond_sha256=ACTIVE_COND_SHA256,
                )


class MotionStreamerParserTests(unittest.TestCase):
    def test_official_272_layout_recovers_float64_source_fk(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            motion_path = root / "motion.npy"
            model_path = root / "neutral.npz"
            parents = np.asarray(
                [-1] + [0] * (MOTIONSTREAMER272_JOINTS - 1), dtype=np.int64
            )
            rest = np.zeros((MOTIONSTREAMER272_JOINTS, 3), dtype=np.float64)
            for joint in range(1, MOTIONSTREAMER272_JOINTS):
                rest[joint] = [joint / 20.0, 0.1 * (joint % 3), 0.05 * joint]
            vertices = rest.copy()
            regressor = np.eye(MOTIONSTREAMER272_JOINTS, dtype=np.float64)
            kintree = np.stack(
                (parents, np.arange(MOTIONSTREAMER272_JOINTS, dtype=np.int64))
            )
            np.savez(
                model_path,
                v_template=vertices,
                J_regressor=regressor,
                kintree_table=kintree,
            )
            data = np.zeros((3, 272), dtype=np.float64)
            identity_d6 = _row_d6(np.eye(3, dtype=np.float64))
            data[:, 2:8] = identity_d6
            data[:, 8:74] = np.broadcast_to(rest, (3,) + rest.shape).reshape(3, -1)
            data[:, 140:272] = np.broadcast_to(
                identity_d6, (3, MOTIONSTREAMER272_JOINTS, 6)
            ).reshape(3, -1)
            np.save(motion_path, data)

            parsed = parse_motionstreamer272_source(
                motion_path,
                joint_names=[f"j{joint}" for joint in range(MOTIONSTREAMER272_JOINTS)],
                parents=parents,
                neutral_model_path=model_path,
            )
            self.assertEqual(parsed.source_positions.dtype, np.float64)
            self.assertEqual(parsed.local_rotations.dtype, np.float64)
            self.assertLess(source_fk_metrics(parsed)["source_parser_fk_max_norm"], 1e-12)

    def test_encoder_guard_requires_explicit_source_fk_pass(self):
        require_source_fk_pass(
            {"clip_id": "ok", "source_parser_fk": {"status": "pass"}}
        )
        with self.assertRaisesRegex(SourceParserError, "cannot enter KTJD encoding"):
            require_source_fk_pass(
                {"clip_id": "bad", "source_parser_fk": {"status": "fail"}}
            )
        with self.assertRaisesRegex(SourceParserError, "cannot enter KTJD encoding"):
            require_source_fk_pass({"clip_id": "missing"})


class SourceFkFailurePathTests(unittest.TestCase):
    @staticmethod
    def _record(
        family: str,
        *,
        gate_status: str,
        metrics: dict[str, float] | None,
    ) -> dict[str, object]:
        return {
            "gate_status": gate_status,
            "audit_role": "prototype_train_calibration",
            "source_family": family,
            "topology_family": "human",
            "rig_id": f"{family}_rig",
            "clip_id": f"{family}_clip",
            "reason_code": (
                None
                if gate_status == "pass"
                else (
                    "SOURCE_FK_REPRODUCTION_FAILED"
                    if metrics is not None
                    else "SOURCE_NUMERIC_PARSE_INVALID"
                )
            ),
            "error": None if gate_status == "pass" else {"message": "fixture"},
            "metrics": metrics,
            "calibration_eligible": True,
            "encoder_called": False,
        }

    def test_whole_source_family_failure_is_summarized_not_crashed(self):
        metric = {"source_parser_fk_max_norm": 1e-15}
        summary = _build_summary(
            [
                self._record(
                    "motionstreamer272", gate_status="fail", metrics=None
                ),
                self._record(
                    "motionstreamer272",
                    gate_status="fail",
                    metrics={"source_parser_fk_max_norm": 2e-6},
                ),
                self._record("planetzoo", gate_status="pass", metrics=metric),
                self._record("truebones", gate_status="pass", metrics=metric),
            ],
            parent_generation_id="fixture-parent",
        )
        audit = summary["train_only_threshold_audit"]["motionstreamer272"]
        self.assertEqual(summary["status"], "completed_with_exclusions")
        self.assertEqual(audit["status"], "unavailable_no_passing_train_metrics")
        self.assertEqual(audit["train_clip_count"], 0)
        self.assertEqual(audit["failed_train_clip_count"], 2)
        self.assertIsNone(audit["train_q99_9"])
        self.assertFalse(audit["held_data_used"])

    def test_validation_report_refuses_immutable_generation_target(self):
        with tempfile.TemporaryDirectory() as directory:
            generation = Path(directory) / "generation"
            generation.mkdir()
            with self.assertRaisesRegex(
                SourceFkValidationError, "cannot be written inside immutable"
            ):
                write_source_fk_validation_report(
                    {"status": "pass"},
                    generation / "source_fk_validation.json",
                    immutable_manifest_root=generation,
                )
            self.assertFalse((generation / "source_fk_validation.json").exists())


if __name__ == "__main__":
    unittest.main()
