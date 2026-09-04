"""protocol.batch binding of the gamma calibration (codex 2026-09-03 round 2): strict for every view,
integer-typed, with exactly one announced legacy exception -- a --resume whose checkpoint already
trained at this --batch under this exact artifact (path + byte sha)."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("train_v2_incontext", ROOT / "scripts" / "train_v2_incontext.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
gate = _mod._calib_batch_gate
demo_drift = _mod._calib_demo_drift

SHA = "a" * 64
PATH = "configs/x_gamma_calibration.json"


def _resume(batch=16, path=PATH, sha=SHA):
    return {"args": {"batch": batch, "ktjd_gamma_calib": path}, "ktjd_pins": {"gamma_calib_sha256": sha}}


class CalibBatchGateTests(unittest.TestCase):
    def test_matching_batch_passes_silently(self):
        self.assertIsNone(gate({"batch": 13}, 13, SHA, PATH, None))
        self.assertIsNone(gate({"batch": 13}, 13, SHA, PATH, _resume(batch=13)))

    def test_non_positive_integer_protocol_batch_is_refused(self):
        for pb in (13.0, 8.9, True, 0, -3, "13", None):
            with self.subTest(pb=pb), self.assertRaisesRegex(SystemExit, "not a positive integer"):
                gate({"batch": pb}, 13, SHA, PATH, None)

    def test_fresh_run_mismatch_is_refused_for_any_view(self):
        with self.assertRaisesRegex(SystemExit, "mechanism-checked at batch 8 but this run uses --batch 16"):
            gate({"batch": 8}, 16, SHA, PATH, None)

    def test_legacy_resume_same_batch_same_artifact_is_announced_not_silent(self):
        msg = gate({"batch": 8}, 16, SHA, PATH, _resume())
        self.assertTrue(msg.startswith("[calib] LEGACY"))
        self.assertIn("NO batch-matched certificate", msg)

    def test_resume_that_changes_anything_is_refused(self):
        cases = {
            "other_batch": _resume(batch=13),
            "other_path": _resume(path="configs/other.json"),
            "other_bytes_same_name": _resume(sha="b" * 64),
            "pre_pinning_ckpt": {"args": {}, "ktjd_pins": {}},
            "wrong_type": {"args": {"batch": "16", "ktjd_gamma_calib": PATH}, "ktjd_pins": {"gamma_calib_sha256": SHA}},
            "float_batch": _resume(batch=16.0),                  # 16.0 == 16 in Python, not the same batch
        }
        for name, res in cases.items():
            with self.subTest(case=name), self.assertRaisesRegex(SystemExit, "recalibrate with CALIB_BATCH=16"):
                gate({"batch": 8}, 16, SHA, PATH, res)

    def test_bool_ckpt_batch_is_refused(self):
        with self.assertRaisesRegex(SystemExit, "recalibrate with CALIB_BATCH=1"):
            gate({"batch": 8}, 1, SHA, PATH, _resume(batch=True))  # True == 1 in Python

    def test_malformed_resume_metadata_is_refused(self):
        for res in ({"args": None, "ktjd_pins": {}}, {"args": {"batch": 16}, "ktjd_pins": None},
                    {"args": [16], "ktjd_pins": {}}, "not-a-mapping", {}):
            with self.subTest(res=res), self.assertRaisesRegex(SystemExit, "malformed"):
                gate({"batch": 8}, 16, SHA, PATH, res)

    def test_malformed_metadata_is_refused_even_when_the_batch_matches(self):
        for res in ({"args": None, "ktjd_pins": {}}, {"args": {"batch": 13}, "ktjd_pins": None},
                    {"args": [13], "ktjd_pins": {}}, "not-a-mapping", [], 7, {}):
            with self.subTest(res=res), self.assertRaisesRegex(SystemExit, "malformed"):
                gate({"batch": 13}, 13, SHA, PATH, res)   # protocol.batch == --batch, still refused


class CalibDemoDriftTests(unittest.TestCase):
    def test_legacy_artifact_means_one_frame_rest_demo(self):
        self.assertEqual(demo_drift({"batch": 8}, True, 1), [])                 # no fields: rest/1 -> matches a rest run
        self.assertTrue(demo_drift({"batch": 8}, False, 64))                    # ... but not a 64-frame motion-demo run

    def test_recorded_demo_condition_is_compared_exactly(self):
        self.assertEqual(demo_drift({"demo_rest": False, "demo_frames": 64}, False, 64), [])
        self.assertTrue(demo_drift({"demo_rest": False, "demo_frames": 64}, True, 1))
        self.assertTrue(demo_drift({"demo_rest": False, "demo_frames": 32}, False, 64))
        msg = demo_drift({"demo_rest": True, "demo_frames": 1}, False, 64)[0]
        self.assertIn("DEMO_REST=0 DEMO_FRAMES=64", msg)

    def test_malformed_fields_are_refused_even_when_coercion_would_match(self):
        for proto in ({"demo_rest": False, "demo_frames": 64.9}, {"demo_rest": False, "demo_frames": "64"},
                      {"demo_rest": [], "demo_frames": 64}, {"demo_rest": 0, "demo_frames": 64},
                      {"demo_rest": False, "demo_frames": True}, {"demo_rest": False, "demo_frames": 0},
                      {"demo_rest": True, "demo_frames": 64}, {"demo_rest": "false", "demo_frames": 64}):
            with self.subTest(proto=proto):
                msgs = demo_drift(proto, False, 64)
                self.assertTrue(msgs and "malformed" in msgs[0])

    def test_coercible_run_inputs_are_refused(self):
        art = {"demo_rest": False, "demo_frames": 64}
        for run in ((0, 64), (1, 1), (False, 64.9), (False, "64"), (False, True), (False, 0), (True, 64), ("0", 64)):
            with self.subTest(run=run), self.assertRaisesRegex(SystemExit, "run demo condition malformed"):
                demo_drift(art, *run)
        self.assertEqual(demo_drift(art, False, 64), [])                       # parser-native bool / int passes
        self.assertIn("DEMO_REST=0 DEMO_FRAMES=64", demo_drift({"demo_rest": True, "demo_frames": 1}, False, 64)[0])

    def test_partial_record_is_refused_not_defaulted(self):
        for proto in ({"demo_rest": False}, {"demo_frames": 64}, {"demo_rest": True}, {"demo_frames": 1}):
            with self.subTest(proto=proto):
                for run in ((True, 1), (False, 64)):
                    msgs = demo_drift(proto, *run)
                    self.assertTrue(msgs and "half-recorded" in msgs[0])


if __name__ == "__main__":
    unittest.main()
