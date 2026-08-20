from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data.ktjd17.truebones_fixed_rig import (
    ACTIVE_COND_SHA256,
    ForwardSpec,
    TRUEBONES_FULL_FORWARD_SPECS,
    forward_from_rest,
    validate_full_forward_spec_catalog,
)
from src.data.ktjd17.truebones_forward_audit import (
    EXPECTED_UNAVAILABLE_RIGS,
    TruebonesForwardAuditError,
    _load_pinned_jsonl,
    _sha256_file,
    default_forward_audit_config,
    encoder_config_from_frozen_schema,
    verify_parent_manifest_files,
)
from src.data.ktjd17.visual_qa import (
    VisualQaError,
    verify_parent_manifest_authority,
    verify_visual_generation,
)


ROOT = Path(__file__).resolve().parents[1]


class FullForwardCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cond_path = (
            ROOT / "data/animo4d_L4TB_plus_human_v4b272neutral/cond.npy"
        )
        cls.cond = np.load(cls.cond_path, allow_pickle=True).item()

    def test_catalog_exactly_covers_frozen_truebones_conditioning(self) -> None:
        self.assertEqual(_sha256_file(self.cond_path), ACTIVE_COND_SHA256)
        legacy_rigs = set(
            np.load(ROOT / "data/anytop_truebones/cond.npy", allow_pickle=True)
            .item()
            .keys()
        )
        self.assertEqual(len(legacy_rigs), 70)
        self.assertEqual(set(TRUEBONES_FULL_FORWARD_SPECS), legacy_rigs)
        validate_full_forward_spec_catalog(
            {rig: self.cond[rig]["joints_names"] for rig in legacy_rigs}
        )

    def test_anaconda_uses_tail_to_head_not_coiled_hips_to_head(self) -> None:
        entry = self.cond["Anaconda"]
        names = tuple(str(value) for value in entry["joints_names"])
        positions = np.asarray(entry["tpos_first_frame"], dtype=np.float64)[:, :3]
        corrected = TRUEBONES_FULL_FORWARD_SPECS["Anaconda"]
        corrected_forward, corrected_indices = forward_from_rest(
            names, positions, corrected
        )
        legacy_wrong = ForwardSpec(
            "root_to_head", ("Hips", "BN_Tone_04"), "negative_control"
        )
        wrong_forward, _ = forward_from_rest(names, positions, legacy_wrong)
        plus_z = np.asarray([0.0, 0.0, 1.0])
        self.assertEqual(corrected_indices, (13, 26))
        self.assertLess(float(np.max(np.abs(corrected_forward - plus_z))), 1e-6)
        self.assertGreater(float(np.max(np.abs(wrong_forward - plus_z))), 0.3)

    def test_only_declared_unavailable_rigs_lack_source_safe_records(self) -> None:
        cfg = default_forward_audit_config(ROOT)
        safe_rigs: set[str] = set()
        all_rigs: set[str] = set()
        import json

        with (cfg.manifest_root / "clips.jsonl").open(
            "r", encoding="utf-8"
        ) as handle:
            for line in handle:
                record = json.loads(line)
                if record["source"]["family"] != "truebones":
                    continue
                all_rigs.add(str(record["rig_id"]))
                if record["status"] != "reject":
                    safe_rigs.add(str(record["rig_id"]))
        self.assertEqual(all_rigs - safe_rigs, EXPECTED_UNAVAILABLE_RIGS)
        self.assertEqual(len(safe_rigs), 66)

    def test_encoder_config_is_loaded_from_frozen_schema(self) -> None:
        config = encoder_config_from_frozen_schema(
            default_forward_audit_config(ROOT).freeze_root
        )
        self.assertEqual(config.calibration_status, "frozen")
        self.assertEqual(config.fps_target, 30.0)
        self.assertEqual(config.smoother.order, 4)
        self.assertEqual(config.contact_tau_h, 0.05)
        self.assertEqual(config.contact_tau_v, 0.25)
        self.assertEqual(config.heading_eps_h, 0.05)


class FullForwardVisualIntegrityTests(unittest.TestCase):
    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_parent_manifest_hashes_are_mandatory_and_fail_on_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / "parent"
            parent.mkdir()
            clips = parent / "clips.jsonl"
            rigs = parent / "rigs.jsonl"
            clips.write_text('{"clip_id":"clip"}\n', encoding="utf-8")
            rigs.write_text('{"rig_id":"rig"}\n', encoding="utf-8")
            record = {
                "selection_authority": {
                    "parent_manifest_root": str(parent),
                    "parent_clips_jsonl_sha256": self._sha256(clips),
                    "parent_rigs_jsonl_sha256": self._sha256(rigs),
                }
            }
            resolved, hashes = verify_parent_manifest_authority(record)
            self.assertEqual(resolved, parent.resolve())
            self.assertEqual(hashes["rigs.jsonl"], self._sha256(rigs))

            missing_rig_hash = json.loads(json.dumps(record))
            del missing_rig_hash["selection_authority"][
                "parent_rigs_jsonl_sha256"
            ]
            with self.assertRaises(VisualQaError):
                verify_parent_manifest_authority(missing_rig_hash)

            rigs.write_text('{"rig_id":"drifted"}\n', encoding="utf-8")
            with self.assertRaises(VisualQaError):
                verify_parent_manifest_authority(record)

    def test_visual_generation_requires_exact_closed_regular_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "visual-generation"
            root.mkdir()
            payload = root / "visual_qa_index.json"
            payload.write_text("{}\n", encoding="utf-8")
            generation = {
                "generation_id": "fixture",
                "files": {
                    "visual_qa_index.json": {
                        "sha256": self._sha256(payload),
                        "size_bytes": payload.stat().st_size,
                    }
                },
            }
            (root / "generation.json").write_text(
                json.dumps(generation) + "\n", encoding="utf-8"
            )
            self.assertEqual(
                verify_visual_generation(root)["generation_id"], "fixture"
            )

            extra = root / "untracked.png"
            extra.write_bytes(b"not an image")
            with self.assertRaises(VisualQaError):
                verify_visual_generation(root)
            extra.unlink()

            payload.unlink()
            payload.symlink_to(root / "generation.json")
            with self.assertRaises(VisualQaError):
                verify_visual_generation(root)

    def test_audit_producer_loads_and_rechecks_the_pinned_byte_stream(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / "parent"
            parent.mkdir()
            clips = parent / "clips.jsonl"
            rigs = parent / "rigs.jsonl"
            clips.write_text('{"clip_id":"clip"}\n', encoding="utf-8")
            rigs.write_text('{"rig_id":"rig"}\n', encoding="utf-8")
            clips_hash = self._sha256(clips)
            rigs_hash = self._sha256(rigs)
            expected = verify_parent_manifest_files(
                parent,
                expected_clips_sha256=clips_hash,
                expected_rigs_sha256=rigs_hash,
            )
            self.assertEqual(
                _load_pinned_jsonl(clips, expected_sha256=clips_hash),
                [{"clip_id": "clip"}],
            )
            self.assertEqual(
                verify_parent_manifest_files(
                    parent,
                    expected_clips_sha256=clips_hash,
                    expected_rigs_sha256=rigs_hash,
                ),
                expected,
            )

            clips.write_text('{"clip_id":"changed"}\n', encoding="utf-8")
            with self.assertRaises(TruebonesForwardAuditError):
                _load_pinned_jsonl(clips, expected_sha256=clips_hash)
            with self.assertRaises(TruebonesForwardAuditError):
                verify_parent_manifest_files(
                    parent,
                    expected_clips_sha256=clips_hash,
                    expected_rigs_sha256=rigs_hash,
                )


if __name__ == "__main__":
    unittest.main()
