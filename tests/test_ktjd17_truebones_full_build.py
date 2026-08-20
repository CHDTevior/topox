from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.data.ktjd17.loader import load_motion_npz
from src.data.ktjd17.truebones_full_build import (
    PARENT_PROTOTYPE_CANDIDATES_SHA256,
    TruebonesFullBuildError,
    _representative_regression,
    _sha256_file,
    _snapshot_regular_file,
    _verify_payload_reference_closure,
    default_full_build_config,
    reviewed_representative_clip_ids,
    summarize_strata,
    validate_visual_gate,
    verify_full_generation,
)


ROOT = Path(__file__).resolve().parents[1]


class FullBuildAuthorityTests(unittest.TestCase):
    def test_live_visual_gate_binds_reviewed_generations(self) -> None:
        config = default_full_build_config(ROOT)
        gate = validate_visual_gate(
            gate_path=config.visual_gate_path,
            visual_root=config.visual_root,
            forward_audit_root=config.forward_audit_root,
        )
        self.assertEqual(gate["verdict"], "pass")
        self.assertTrue(gate["authorization"]["full_source_safe_conversion"])
        self.assertEqual(gate["visual_generation"]["rig_count"], 66)

    def test_parent_prototype_candidate_pin_is_live(self) -> None:
        config = default_full_build_config(ROOT)
        self.assertEqual(
            _sha256_file(config.manifest_root / "prototype_candidates.json"),
            PARENT_PROTOTYPE_CANDIDATES_SHA256,
        )

    def test_reviewed_representative_scope_is_exactly_the_frozen_66(self) -> None:
        config = default_full_build_config(ROOT)
        clip_ids = reviewed_representative_clip_ids(config.forward_audit_root)
        self.assertEqual(len(clip_ids), 66)
        self.assertEqual(len(set(clip_ids)), 66)

    def test_stable_source_snapshot_checks_size_mtime_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bvh"
            source.write_bytes(b"HIERARCHY\nMOTION\n")
            stat_result = source.stat()
            snapshot = _snapshot_regular_file(
                source,
                expected_size=stat_result.st_size,
                expected_mtime_ns=stat_result.st_mtime_ns,
            )
            self.assertEqual(snapshot["size_bytes"], stat_result.st_size)
            self.assertEqual(snapshot["sha256"], _sha256_file(source))
            with self.assertRaises(TruebonesFullBuildError):
                _snapshot_regular_file(source, expected_size=stat_result.st_size + 1)
            link = root / "source-link.bvh"
            link.symlink_to(source)
            with self.assertRaises(TruebonesFullBuildError):
                _snapshot_regular_file(link)

    def test_visual_gate_copy_cannot_self_authorize_after_edit(self) -> None:
        config = default_full_build_config(ROOT)
        with tempfile.TemporaryDirectory() as temporary:
            gate_path = Path(temporary) / "gate.json"
            gate = json.loads(config.visual_gate_path.read_text(encoding="utf-8"))
            gate["independent_review"]["model_reasoning_effort"] = "low"
            gate_path.write_text(json.dumps(gate) + "\n", encoding="utf-8")
            with self.assertRaises(TruebonesFullBuildError):
                validate_visual_gate(
                    gate_path=gate_path,
                    visual_root=config.visual_root,
                    forward_audit_root=config.forward_audit_root,
                )


class FullBuildRegressionTests(unittest.TestCase):
    def test_payload_reference_closure_rejects_orphan_motion(self) -> None:
        observed = {
            "motions/accepted.npz": {},
            "motions/orphan.npz": {},
            "skeletons/Rig.npz": {},
        }
        with self.assertRaisesRegex(TruebonesFullBuildError, "orphan.npz"):
            _verify_payload_reference_closure(
                observed,
                referenced_motions={"motions/accepted.npz"},
                referenced_skeletons={"skeletons/Rig.npz"},
            )

    def test_verifier_requires_complete_generation_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "incomplete-generation"
            root.mkdir()
            (root / "generation.json").write_text(
                json.dumps(
                    {
                        "generation_id": root.name,
                        "status": "conversion_incomplete",
                        "conversion_complete": False,
                        "full_conversion_authorized": False,
                        "files": {},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                TruebonesFullBuildError, "full conversion is incomplete"
            ):
                verify_full_generation(root)

    def test_reviewed_representative_requires_exact_arrays(self) -> None:
        config = default_full_build_config(ROOT)
        manifest_path = config.forward_audit_root / "manifests/clips.jsonl"
        first = json.loads(manifest_path.read_text(encoding="utf-8").splitlines()[0])
        path = config.forward_audit_root / first["motion_relpath"]
        payload = load_motion_npz(path, expected_fps_target=30.0)
        encoded = SimpleNamespace(
            motion_float32=payload["motion"].copy(),
            heading_valid=payload["heading_valid"].copy(),
            origin_xz=payload["origin_xz"].copy(),
            clip_id=payload["clip_id"],
            rig_id=payload["rig_id"],
            fps_target=float(payload["fps_target"]),
        )
        _representative_regression(encoded, audit_motion_path=path)
        encoded.motion_float32[0, 0, 0] += np.float32(1.0)
        with self.assertRaises(TruebonesFullBuildError):
            _representative_regression(encoded, audit_motion_path=path)

    def test_stratified_summary_keeps_split_rig_and_bucket(self) -> None:
        records = [
            {
                "status": "pass",
                "split": "train",
                "source_family": "truebones",
                "topology_family": "quadruped",
                "topology_distance_bucket": "train_seen_topology",
                "rig_id": "A",
                "parent_inventory_status": "accept",
                "metrics": {"direct_vs_fk_max_norm": 1e-8},
            },
            {
                "status": "pass",
                "split": "val",
                "source_family": "truebones",
                "topology_family": "winged",
                "topology_distance_bucket": "held_out_topology",
                "rig_id": "B",
                "parent_inventory_status": "review",
                "metrics": {"direct_vs_fk_max_norm": 2e-8},
            },
        ]
        summary = summarize_strata(records)
        self.assertEqual(summary["all"]["count"], 2)
        self.assertEqual(summary["split:train"]["pass"], 1)
        self.assertEqual(summary["rig_id:B"]["count"], 1)
        self.assertEqual(
            summary["all"]["metrics"]["direct_vs_fk_max_norm"]["max"],
            2e-8,
        )


if __name__ == "__main__":
    unittest.main()
