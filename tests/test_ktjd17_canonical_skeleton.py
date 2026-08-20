"""CPU gold tests for the KTJD-17 T04 canonical skeleton stage."""

from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from scipy.spatial.transform import Rotation


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ktjd17.canonical_skeleton import (  # noqa: E402
    TRUEBONES_BTJD_MEAN_EDGE_TARGET,
    TRUEBONES_FORWARD_SPECS,
    BuiltSkeleton,
    CanonicalSkeletonError,
    ForwardSpec,
    _artifact_payload,
    _rest_fk,
    _rest_gate_metrics,
    _summarize_qa,
    _validate_direct_t03_parent,
    apply_canonical_positions,
    apply_canonical_rotations,
    derive_rest_local_arrays,
    resolve_manifest_skeleton_artifact,
    validate_source_to_canonical_basis,
)
from src.data.ktjd17.canonical_skeleton_validation import (  # noqa: E402
    CanonicalSkeletonValidationError,
    EXPECTED_TRUEBONES_FORWARD_SPECS,
    _require_expected_forward_provenance,
    write_canonical_skeleton_validation_report,
)
from src.data.ktjd17.inventory import _write_transaction  # noqa: E402
from src.data.ktjd17.schema import SKELETON_REQUIRED_KEYS  # noqa: E402


class CanonicalBasisTests(unittest.TestCase):
    def test_proper_and_reflection_bases_are_valid(self):
        proper = Rotation.from_euler("Y", 37.0, degrees=True).as_matrix().astype(
            np.float64
        )
        reflected = np.diag([-1.0, 1.0, 1.0]).astype(np.float64)
        self.assertAlmostEqual(
            validate_source_to_canonical_basis(proper)["basis_determinant"],
            1.0,
        )
        self.assertAlmostEqual(
            validate_source_to_canonical_basis(reflected)["basis_determinant"],
            -1.0,
        )

    def test_scaled_or_float32_basis_fails_closed(self):
        with self.assertRaisesRegex(CanonicalSkeletonError, "C.T@C"):
            validate_source_to_canonical_basis(
                np.diag([1.0, 1.0, 1.001]).astype(np.float64)
            )
        with self.assertRaisesRegex(CanonicalSkeletonError, "float64"):
            validate_source_to_canonical_basis(np.eye(3, dtype=np.float32))

    def test_position_formula_and_rotation_conjugation(self):
        points = np.asarray([[2.0, 3.0, 4.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        origin = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
        C = Rotation.from_euler("Y", 90.0, degrees=True).as_matrix().astype(
            np.float64
        )
        actual = apply_canonical_positions(points, C=C, alpha=2.0, o=origin)
        expected = np.stack([2.0 * C @ (point - origin) for point in points])
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.0)

        source_rotation = Rotation.from_euler(
            "XYZ", [10.0, 20.0, 30.0], degrees=True
        ).as_matrix().astype(np.float64)
        reflected = np.diag([-1.0, 1.0, 1.0]).astype(np.float64)
        canonical = apply_canonical_rotations(
            source_rotation[None], C=reflected
        )[0]
        np.testing.assert_allclose(
            canonical, reflected @ source_rotation @ reflected.T, atol=1e-12
        )
        self.assertAlmostEqual(float(np.linalg.det(canonical)), 1.0, places=12)


class CanonicalRestTests(unittest.TestCase):
    def test_local_rotation_and_parent_local_offset_are_same_rest(self):
        parents = np.asarray([-1, 0, 1], dtype=np.int64)
        root = Rotation.from_euler("Z", 90.0, degrees=True).as_matrix()
        child_local = Rotation.from_euler("X", 30.0, degrees=True).as_matrix()
        grandchild_local = Rotation.from_euler("Y", -20.0, degrees=True).as_matrix()
        rotations = np.stack(
            (root, root @ child_local, root @ child_local @ grandchild_local)
        ).astype(np.float64)
        expected_offsets = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            dtype=np.float64,
        )
        positions = _rest_fk(
            parents,
            np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
            rotations,
            expected_offsets,
        )
        local, offsets = derive_rest_local_arrays(parents, positions, rotations)
        np.testing.assert_allclose(local[0], root, atol=1e-12)
        np.testing.assert_allclose(local[1], child_local, atol=1e-12)
        np.testing.assert_allclose(local[2], grandchild_local, atol=1e-12)
        np.testing.assert_allclose(offsets, expected_offsets, atol=1e-12)
        np.testing.assert_allclose(
            _rest_fk(parents, positions[0], rotations, offsets),
            positions,
            atol=1e-12,
        )

    def test_truebones_scale_and_float32_gates(self):
        parents = np.asarray([-1, 0], dtype=np.int64)
        positions = np.asarray(
            [[0.0, 0.0, 0.0], [TRUEBONES_BTJD_MEAN_EDGE_TARGET, 0.0, 0.0]],
            dtype=np.float64,
        )
        rotations = np.broadcast_to(np.eye(3), (2, 3, 3)).copy()
        local, offsets = derive_rest_local_arrays(parents, positions, rotations)
        metrics = _rest_gate_metrics(
            parents=parents,
            C=np.eye(3, dtype=np.float64),
            P_rest_global=positions,
            R_rest_global=rotations,
            R_rest_local=local,
            offset_parent_local=offsets,
            source_forward=np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            alpha=TRUEBONES_BTJD_MEAN_EDGE_TARGET / 2.0,
            source_mean_edge=2.0,
            source_family="truebones",
        )
        self.assertLessEqual(metrics["rest_position_fk_float32_max_norm"], 1e-5)
        self.assertAlmostEqual(
            metrics["canonical_mean_nonroot_edge_length"],
            TRUEBONES_BTJD_MEAN_EDGE_TARGET,
        )
        self.assertAlmostEqual(
            metrics["s_rig"], TRUEBONES_BTJD_MEAN_EDGE_TARGET
        )


class ArtifactContractTests(unittest.TestCase):
    def _built(self, rest_path: Path) -> BuiltSkeleton:
        parents = np.asarray([-1, 0], dtype=np.int64)
        positions = np.asarray(
            [[0.0, 0.0, 0.0], [TRUEBONES_BTJD_MEAN_EDGE_TARGET, 0.0, 0.0]],
            dtype=np.float64,
        )
        rotations = np.broadcast_to(np.eye(3), (2, 3, 3)).copy()
        local, offsets = derive_rest_local_arrays(parents, positions, rotations)
        parsed = SimpleNamespace(
            joint_names=("Root", "Child"),
            parents=parents,
            rotation_source_kind=("animated_dof", "fixed_dof"),
            rest_path=str(rest_path),
        )
        joint_map = {
            "btjd_joint_names": ["Root", "Child"],
            "btjd_parents": [-1, 0],
            "rotation_source_kind": ["animated_dof", "fixed_dof"],
        }
        return BuiltSkeleton(
            rig_id="Tiny",
            source_family="truebones",
            topology_family="quadruped",
            representative_clip_id="clip",
            parsed=parsed,
            C=np.eye(3, dtype=np.float64),
            alpha=1.0,
            o=np.zeros(3, dtype=np.float64),
            P_rest_global=positions,
            R_rest_global=rotations,
            R_rest_local=local,
            offset_parent_local=offsets,
            heading_carrier_joint=0,
            u_forward_local=np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            source_forward=np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            forward_spec=ForwardSpec("declared_plus_z", (), "test"),
            s_rig=TRUEBONES_BTJD_MEAN_EDGE_TARGET,
            length_unit_id="test-unit",
            source_unit_to_meter=None,
            metrics={},
            artifact_status="pass",
            reason_codes=[],
            heading_provenance={"status": "test"},
            transform_provenance={"status": "test"},
            rig_record={"joint_map": joint_map},
        )

    def test_npz_payload_has_required_pickle_free_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            rest_path = Path(directory) / "rest.bvh"
            rest_path.write_text("rest evidence\n", encoding="utf-8")
            payload = _artifact_payload(self._built(rest_path))
            self.assertTrue(set(SKELETON_REQUIRED_KEYS).issubset(payload))
            self.assertFalse(any(value.dtype.hasobject for value in payload.values()))
            self.assertEqual(payload["source_unit_to_meter"].shape, (0,))
            artifact = Path(directory) / "skeleton.npz"
            np.savez_compressed(artifact, **payload)
            with np.load(artifact, allow_pickle=False) as loaded:
                for name in loaded.files:
                    self.assertFalse(loaded[name].dtype.hasobject)

    def test_manifest_resolver_uses_immutable_path_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory)
            generation = dataset / ".ktjd17_skeleton_generations" / "g1"
            generation.mkdir(parents=True)
            artifact = generation / "Tiny.npz"
            artifact.write_bytes(b"artifact")
            digest = hashlib.sha256(b"artifact").hexdigest()
            metadata = {
                "artifact_relpath": ".ktjd17_skeleton_generations/g1/Tiny.npz",
                "artifact_sha256": digest,
            }
            self.assertEqual(
                resolve_manifest_skeleton_artifact(dataset, metadata),
                artifact.resolve(),
            )
            with self.assertRaisesRegex(CanonicalSkeletonError, "hash mismatch"):
                resolve_manifest_skeleton_artifact(
                    dataset, {**metadata, "artifact_sha256": "0" * 64}
                )
            with self.assertRaisesRegex(CanonicalSkeletonError, "escapes"):
                resolve_manifest_skeleton_artifact(
                    dataset,
                    {
                        "artifact_relpath": "outside.npz",
                        "artifact_sha256": digest,
                    },
                )

    def test_post_swap_fsync_failure_cannot_delete_active_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "manifests"
            real_fsync = __import__("os").fsync
            call_count = 0

            def fail_parent_fsync(descriptor: int) -> None:
                nonlocal call_count
                call_count += 1
                if call_count == 4:
                    raise OSError("injected post-swap parent fsync failure")
                real_fsync(descriptor)

            with mock.patch(
                "src.data.ktjd17.inventory.os.fsync",
                side_effect=fail_parent_fsync,
            ):
                with self.assertRaisesRegex(OSError, "injected post-swap"):
                    _write_transaction(
                        output,
                        {"payload.json": "{}\n"},
                        overwrite=True,
                    )
            self.assertTrue(output.is_symlink())
            self.assertTrue(output.resolve().is_dir())
            self.assertEqual((output / "payload.json").read_text(), "{}\n")
            self.assertTrue((output / "inventory_generation.json").is_file())

    def test_t04_requires_direct_hash_complete_t03_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = (
                Path(directory)
                / ".ktjd17_manifest_generations"
                / "t03-generation"
            )
            root.mkdir(parents=True)
            required = {
                "clips.jsonl",
                "rigs.jsonl",
                "inventory_summary.json",
                "inventory_reason_codes.json",
                "prototype_candidates.json",
                "prototype_gaps.jsonl",
                "source_fk_qa.jsonl",
                "source_fk_summary.json",
                "source_fk_generation.json",
            }
            files = {}
            for name in sorted(required):
                path = root / name
                path.write_text(name + "\n", encoding="utf-8")
                files[name] = {
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "size_bytes": path.stat().st_size,
                }
            transaction = {"generation_id": root.name, "files": files}
            generation_id, source_files = _validate_direct_t03_parent(
                root, transaction
            )
            self.assertEqual(generation_id, root.name)
            self.assertEqual(
                set(source_files),
                {
                    "source_fk_qa.jsonl",
                    "source_fk_summary.json",
                    "source_fk_generation.json",
                },
            )
            bad = {**transaction, "files": {**files, "canonical_skeleton_qa.jsonl": {}}}
            with self.assertRaisesRegex(CanonicalSkeletonError, "direct T03"):
                _validate_direct_t03_parent(root, bad)


class SummaryPolicyTests(unittest.TestCase):
    @staticmethod
    def _record(
        rig_id: str,
        source: str,
        topology: str,
        status: str,
        *,
        artifact: bool,
        reasons: list[str] | None = None,
    ) -> dict[str, object]:
        return {
            "rig_id": rig_id,
            "source_family": source,
            "topology_family": topology,
            "gate_status": status,
            "reason_codes": reasons or [],
            "artifact_relpath": f"{rig_id}.npz" if artifact else None,
        }

    def test_exact_fail_closed_outcome_is_accepted(self):
        topologies = [
            "quadruped",
            "winged",
            "snake",
            "spider_crab",
            "dragon_or_deep_topology",
        ]
        records = []
        for index in range(31):
            review = index < 4
            records.append(
                self._record(
                    f"tb{index}",
                    "truebones",
                    topologies[index % len(topologies)],
                    "review" if review else "pass",
                    artifact=True,
                )
            )
        records.extend(
            self._record(
                f"pz{index}", "planetzoo", "quadruped", "reject", artifact=False
            )
            for index in range(26)
        )
        records.append(
            self._record(
                "human",
                "motionstreamer272",
                "human",
                "reject",
                artifact=True,
            )
        )
        summary = _summarize_qa(records)
        self.assertTrue(summary["expected_outcomes_satisfied"])
        records[0]["gate_status"] = "pass"
        self.assertFalse(_summarize_qa(records)["expected_outcomes_satisfied"])

    def test_validation_report_cannot_mutate_immutable_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            immutable = Path(directory) / "immutable"
            immutable.mkdir()
            with self.assertRaises(CanonicalSkeletonValidationError):
                write_canonical_skeleton_validation_report(
                    {"status": "pass"},
                    immutable / "report.json",
                    immutable_manifest_root=immutable,
                )

    def test_validator_owns_exact_forward_anchor_truth(self):
        self.assertEqual(set(TRUEBONES_FORWARD_SPECS), set(EXPECTED_TRUEBONES_FORWARD_SPECS))
        for rig_id, producer in TRUEBONES_FORWARD_SPECS.items():
            self.assertEqual(
                (producer.method, producer.anchor_names, producer.provenance),
                EXPECTED_TRUEBONES_FORWARD_SPECS[rig_id],
            )
        names = ["R_momo", "L_momo", "R_hiji", "L_hiji", "wrong"]
        correct = {
            "forward_method": "lateral_pairs",
            "forward_anchor_names": ["R_momo", "L_momo", "R_hiji", "L_hiji"],
            "forward_anchor_indices": [0, 1, 2, 3],
            "forward_spec_provenance": "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04",
        }
        _require_expected_forward_provenance(
            "Alligator", "truebones", names, correct
        )
        wrong = {**correct, "forward_anchor_names": ["R_momo", "L_momo", "wrong", "L_hiji"]}
        with self.assertRaisesRegex(
            CanonicalSkeletonValidationError, "frozen forward anchors"
        ):
            _require_expected_forward_provenance(
                "Alligator", "truebones", names, wrong
            )


if __name__ == "__main__":
    unittest.main()
