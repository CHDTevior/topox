"""Fail-closed unit tests for the PZ-311 plus Human-1 KTJD-17 builder."""

from __future__ import annotations

import hashlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.data.ktjd17.pz_human312_build as build  # noqa: E402
from src.data.ktjd17.encoder import Ktjd17EncoderError  # noqa: E402
from src.data.ktjd17.visual_qa import (  # noqa: E402
    verify_parent_manifest_authority,
)


class PzHuman312BuildTests(unittest.TestCase):
    def test_generation_requires_external_read_only_content_approval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary).resolve()
            generation_root = (
                output_root
                / build.PROTOTYPE_GENERATION_DIRECTORY
                / "generation-test"
            )
            generation_root.mkdir(parents=True)
            (generation_root / "payload.bin").write_bytes(b"payload")
            generation = {
                "build_version": build.BUILD_VERSION,
                "generation_id": generation_root.name,
                "mode": "prototype",
                "source_plan_commit": build.SOURCE_PLAN_COMMIT,
                "freeze_binding": {"generation_id": "freeze"},
                "coordinate_contract": build.COORDINATE_CONTRACT,
                "source_audit_bindings": {"planetzoo": {}, "motionstreamer272": {}},
                "source_scope_count": 312,
                "accepted_clip_count": 312,
                "rejected_clip_count": 0,
                "rig_count": 312,
                "selection_sha256": "a" * 64,
                "visual_gate_sha256": None,
                "anomaly_allowlist_sha256": None,
                "anomaly_allowlist_entry_set_sha256": "b" * 64,
                "final_source_recheck_sha256": "c" * 64,
                "prototype_conversion_authorized": True,
                "full_conversion_authorized": False,
                "files": build._file_manifest(generation_root),
            }
            build._write_json(generation_root / "generation.json", generation)
            build._freeze_tree(generation_root)
            try:
                evidence = build._generation_content_evidence(
                    generation_root, generation=generation
                )
                with self.assertRaisesRegex(
                    build.PzHuman312BuildError, "invalid build approval root"
                ):
                    build._validate_generation_approval(
                        generation_root,
                        generation=generation,
                        evidence=evidence,
                    )
                approval_path, approval = build._create_generation_approval(
                    generation_root,
                    generation=generation,
                    evidence=evidence,
                )
                self.assertEqual(
                    approval["generation_relpath"],
                    (
                        f"{build.PROTOTYPE_GENERATION_DIRECTORY}/"
                        f"{generation_root.name}"
                    ),
                )
                self.assertNotIn(str(output_root), str(approval))
                validated_path, validated = build._validate_generation_approval(
                    generation_root,
                    generation=generation,
                    evidence=evidence,
                )
                self.assertEqual(validated_path, approval_path)
                self.assertEqual(validated, approval)
                os.chmod(approval_path, 0o600)
                with self.assertRaisesRegex(
                    build.PzHuman312BuildError, "read-only single-link"
                ):
                    build._validate_generation_approval(
                        generation_root,
                        generation=generation,
                        evidence=evidence,
                    )
            finally:
                for path in sorted(
                    output_root.rglob("*"),
                    key=lambda item: len(item.parts),
                    reverse=True,
                ):
                    if path.is_file():
                        os.chmod(path, 0o600)
                    elif path.is_dir():
                        os.chmod(path, 0o700)

    def test_approved_source_bytes_reject_same_stat_content_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.bin"
            source.write_bytes(b"approved")
            observed = source.stat()
            expected = {
                "source_device": int(observed.st_dev),
                "source_inode": int(observed.st_ino),
                "source_size_bytes": int(observed.st_size),
                "source_mtime_ns": int(observed.st_mtime_ns),
                "source_nlink": int(observed.st_nlink),
                "source_sha256": hashlib.sha256(b"approved").hexdigest(),
            }
            payload, snapshot = build._read_approved_source_bytes(source, expected)
            self.assertEqual(payload, b"approved")
            self.assertEqual(snapshot["inode"], int(observed.st_ino))

            source.write_bytes(b"tampered")
            os.utime(
                source,
                ns=(int(observed.st_atime_ns), int(observed.st_mtime_ns)),
            )
            with self.assertRaisesRegex(
                build.PzHuman312BuildError, "approved source bytes drifted"
            ):
                build._read_approved_source_bytes(source, expected)

    def test_human_production_parser_is_crosschecked_not_replaced(self) -> None:
        frames = 2
        raw = np.zeros((frames, 272), dtype=np.float64)
        identity_row6d = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        raw[:, 2:8] = identity_row6d
        raw[:, 140:272] = np.tile(identity_row6d, 22)
        buffer = io.BytesIO()
        np.save(buffer, raw, allow_pickle=False)

        parents = np.asarray([-1] + list(range(21)), dtype=np.int64)
        offsets = np.zeros((22, 3), dtype=np.float64)
        offsets[1:, 1] = 1.0
        rest_positions = np.zeros((22, 3), dtype=np.float64)
        for child in range(1, 22):
            rest_positions[child] = (
                rest_positions[int(parents[child])] + offsets[child]
            )
        identity = np.broadcast_to(np.eye(3), (22, 3, 3)).copy()
        skeleton = build.SkeletonData(
            path="skeletons/HML3D_Human.npz",
            sha256="a" * 64,
            rig_id="HML3D_Human",
            source_family="motionstreamer272",
            topology_family="human",
            joint_names=tuple(f"joint_{index}" for index in range(22)),
            parents=parents,
            P_rest_global=rest_positions,
            R_rest_global=identity,
            R_rest_local=identity,
            offset_parent_local=offsets,
            rotation_source_kind=np.asarray(["animated_dof"] * 22),
            heading_carrier_joint=0,
            u_forward_local=np.asarray([0.0, 0.0, 1.0]),
            source_to_canonical_C=np.eye(3),
            source_to_canonical_alpha=1.0,
            source_to_canonical_o=np.zeros(3),
            s_rig=21.0,
            artifact_status="t05_prototype_override_pass",
            metadata={},
        )
        prepared = build._prepare_human_fast(
            {"clip_id": "Human_M000001", "rig_id": "HML3D_Human"},
            skeleton,
            {
                "source_relpath": "M000001.npy",
                "source_sha256": "b" * 64,
                "metrics": {"source_parser_fk_max_norm": 0.0},
            },
            buffer.getvalue(),
        )
        self.assertEqual(
            prepared.provenance["production_decoder"],
            "human_source_parser.parse_motionstreamer272_fixed_neutral_array",
        )
        self.assertEqual(
            prepared.provenance["independent_crosscheck_decoder"],
            "human312_audit.independent_motionstreamer272_decode",
        )
        for key, value in prepared.source_parser_metrics.items():
            if key.startswith("production_vs_independent_"):
                self.assertEqual(value, 0.0)

    def test_full_mode_requires_visual_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = build.BuildConfig(
                dataset_root=root,
                freeze_root=root,
                output_root=root,
                mode="full",
            )
            with self.assertRaisesRegex(
                build.PzHuman312BuildError, "requires --visual-gate"
            ):
                config.resolved()

    def test_only_explicit_numeric_encoder_gates_are_filterable(self) -> None:
        allowed = Ktjd17EncoderError(
            "clip: float32 direct roundtrip failed: 1e-4"
        )
        systemic = Ktjd17EncoderError(
            "clip: PlanetZoo requires the reviewed stage-2 fixed-rig artifact"
        )
        self.assertTrue(build._is_filterable_encoder_gate(allowed))
        self.assertFalse(build._is_filterable_encoder_gate(systemic))
        self.assertFalse(
            build._is_filterable_encoder_gate(FileNotFoundError("missing"))
        )

    def test_human_source_relpath_must_be_flat_and_contained(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            self.assertEqual(
                build._source_relpath(
                    {"source_relpath": "M000001.npy"},
                    family="motionstreamer272",
                    source_root=root,
                ),
                "M000001.npy",
            )
            with self.assertRaisesRegex(
                build.PzHuman312BuildError, "unsafe Human source"
            ):
                build._source_relpath(
                    {"source_relpath": "../escape.npy"},
                    family="motionstreamer272",
                    source_root=root,
                )
            with self.assertRaisesRegex(
                build.PzHuman312BuildError, "escaped flat source root"
            ):
                build._source_relpath(
                    {"source_relpath": "nested/M000001.npy"},
                    family="motionstreamer272",
                    source_root=root,
                )

    def test_unexpected_worker_exception_aborts_instead_of_filtering(self) -> None:
        task = {
            "clip": {"clip_id": "clip", "rig_id": "rig"},
            "audit": {"source_family": "planetzoo"},
        }
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(
                build, "_conversion_worker", side_effect=RuntimeError("bug")
            ):
                with self.assertRaisesRegex(
                    build.PzHuman312BuildError,
                    "unexpected conversion failure for clip",
                ):
                    build._run_conversion_tasks(
                        [task],
                        rigs={},
                        skeleton_paths={},
                        encoder=mock.Mock(),
                        output_root=Path(temporary),
                        workers=1,
                        allowed_rejections={},
                    )

    def test_identifiers_reject_paths_and_separators(self) -> None:
        self.assertEqual(
            build._safe_identifier("Human_M000001", label="test identifier"),
            "Human_M000001",
        )
        for value in ("../escape", "nested/clip", "nested\\clip", ".", ""):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    build.PzHuman312BuildError, "unsafe test identifier"
                ):
                    build._safe_identifier(value, label="test identifier")

    def test_prototype_forbids_anomaly_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = build.BuildConfig(
                dataset_root=root,
                freeze_root=root,
                output_root=root,
                mode="prototype",
                anomaly_allowlist_path=root / "allowlist.json",
            )
            with self.assertRaisesRegex(
                build.PzHuman312BuildError, "forbids anomaly filtering"
            ):
                config.resolved()

    def test_no_allowlist_means_exactly_zero_rejections(self) -> None:
        result = build._load_anomaly_allowlist(None, {})
        self.assertEqual(result["entries"], {})
        self.assertIsNone(result["payload"])
        self.assertIsNone(result["sha256"])
        self.assertEqual(
            result["entry_set_sha256"],
            build._sha256_bytes(build._canonical_json([])),
        )

    def test_relative_parent_manifest_authority_is_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset_root = Path(temporary).resolve()
            parent = dataset_root / "manifests/generation"
            parent.mkdir(parents=True)
            (parent / "clips.jsonl").write_text("{}\n", encoding="utf-8")
            (parent / "rigs.jsonl").write_text("{}\n", encoding="utf-8")
            record = {
                "selection_authority": {
                    "parent_manifest_base": "dataset_root",
                    "parent_manifest_relpath": "manifests/generation",
                    "parent_clips_jsonl_sha256": build._sha256_file(
                        parent / "clips.jsonl"
                    ),
                    "parent_rigs_jsonl_sha256": build._sha256_file(
                        parent / "rigs.jsonl"
                    ),
                }
            }
            resolved, hashes = verify_parent_manifest_authority(
                record,
                dataset_root=dataset_root,
            )
            self.assertEqual(resolved, parent)
            self.assertEqual(set(hashes), {"clips.jsonl", "rigs.jsonl"})
            record["selection_authority"]["parent_manifest_relpath"] = "../escape"
            with self.assertRaisesRegex(Exception, "unsafe"):
                verify_parent_manifest_authority(
                    record,
                    dataset_root=dataset_root,
                )

    def test_visual_review_requires_exact_artifact_bindings(self) -> None:
        binding = {
            "clip_id": "clip",
            "rig_id": "rig",
            "paths_reviewed": ["source", "position-direct", "rotation-FK"],
            "perspective_camera": True,
            "fixed_camera_across_frames_and_paths": True,
            "frame_recenter_applied": False,
            "ground_changed": False,
            "face_direction_changed": False,
            "gif": {"relpath": "clips/a.gif", "sha256": "a" * 64},
            "filmstrip": {
                "relpath": "clips/a_filmstrip.png",
                "sha256": "b" * 64,
            },
            "rest": {"relpath": "clips/a_rest.png", "sha256": "c" * 64},
        }
        expectation = {
            "prototype_generation_id": "prototype",
            "prototype_generation_sha256": "d" * 64,
            "visual_generation_id": "visual",
            "visual_generation_sha256": "e" * 64,
            "freeze_binding": {"generation_id": "freeze"},
            "coverage": {"reviewed_clip_count": 1},
            "artifact_bindings": [binding],
        }
        review = {
            "model": "gpt-5.6-sol",
            "model_reasoning_effort": "xhigh",
            "review_thread_id": "thread",
            "verdict": "pass",
            "prototype_generation_id": "prototype",
            "prototype_generation_sha256": "d" * 64,
            "visual_generation_id": "visual",
            "visual_generation_sha256": "e" * 64,
            "freeze_binding": {"generation_id": "freeze"},
            "coverage": {"reviewed_clip_count": 1},
            "artifact_reviews": [
                {**binding, "status": "pass", "native_image_reviewed": True}
            ],
            "failures": [],
        }
        self.assertEqual(build.validate_visual_review(review, expectation), review)
        forged = {**review, "artifact_reviews": [dict(review["artifact_reviews"][0])]}
        forged["artifact_reviews"][0]["gif"] = {
            "relpath": "clips/a.gif",
            "sha256": "f" * 64,
        }
        with self.assertRaisesRegex(
            build.PzHuman312BuildError, "exact 312-artifact"
        ):
            build.validate_visual_review(forged, expectation)

    def test_rejection_record_requires_specific_reason(self) -> None:
        task = {
            "clip": {
                "clip_id": "clip",
                "rig_id": "rig",
                "topology_family": "quadruped",
                "topology_distance_bucket": "train_seen_topology",
                "split": "train",
            },
            "audit": {
                "source_family": "planetzoo",
                "source_relpath": "clip.bvh",
                "source_sha256": "a" * 64,
            },
        }
        record = build._rejection_record(
            task,
            Ktjd17EncoderError("numeric gate"),
            reason_code="KTJD17_NUMERICAL_ENCODING_GATE_FAILURE",
        )
        self.assertEqual(
            record["reason_codes"], ["KTJD17_NUMERICAL_ENCODING_GATE_FAILURE"]
        )
        self.assertFalse(record["legacy_fallback_allowed"])


if __name__ == "__main__":
    unittest.main()
