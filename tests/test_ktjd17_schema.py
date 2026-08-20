"""CPU-only tests for the KTJD-17 M0 schema contract."""

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ktjd17.schema import (
    CHANNEL_SLICES,
    KTJD17_D,
    KTJD17_REPR_VERSION,
    NON_ROOT_CHANNEL_VALIDITY,
    SchemaValidationError,
    build_schema,
    load_schema,
    validate_physical_parent_tree,
    validate_schema,
    validate_unit_metadata,
    write_schema,
)


def _frozen_schema() -> dict:
    return build_schema(
        fps_target=30.0,
        smoother_id="prototype_smoother_v1",
        smoother_params={"margin_norm": 0.02},
        short_clip_rule="identity_below_3_frames",
        heading_eps_h=1e-6,
        contact_tau_h=0.05,
        contact_tau_v=0.10,
        normalization_gains=[1.0, 2.0, 3.0],
        j_max=144,
        frozen=True,
        calibration_run_ids=["prototype-train-only-001"],
        train_split_protocol="prototype_train_v1",
        frozen_at_utc="2026-08-19T00:00:00Z",
    )


class KTJD17SchemaTests(unittest.TestCase):
    def test_candidate_schema_has_exact_core_contract(self):
        schema = build_schema()
        validate_schema(schema)
        self.assertEqual(schema["repr_version"], KTJD17_REPR_VERSION)
        self.assertEqual(schema["D"], KTJD17_D)
        self.assertEqual(schema["root_index"], 0)
        self.assertEqual(schema["channel_slices"], CHANNEL_SLICES)
        self.assertEqual(schema["channel_validity"]["physical_non_root"],
                         NON_ROOT_CHANNEL_VALIDITY)
        self.assertEqual(schema["coordinate"]["handedness"], "right")
        self.assertEqual(schema["coordinate"]["up"], "+Y")
        self.assertEqual(schema["coordinate"]["rest_forward"], "+Z")
        self.assertEqual(schema["calibration"]["status"], "unfrozen")

    def test_unfrozen_schema_fails_frozen_consumer_gate(self):
        with self.assertRaisesRegex(SchemaValidationError, "requires a frozen schema"):
            validate_schema(build_schema(), require_frozen=True)

    def test_complete_frozen_schema_passes(self):
        validate_schema(_frozen_schema(), expected_fps_target=30.0, require_frozen=True)

    def test_world_or_control_node_schema_flag_rejected(self):
        schema = build_schema()
        schema["topology"]["virtual_nodes_allowed"] = True
        with self.assertRaisesRegex(SchemaValidationError, "virtual_nodes_allowed"):
            validate_schema(schema)

    def test_missing_channel_slice_rejected(self):
        schema = build_schema()
        del schema["channel_slices"]["heading"]
        with self.assertRaisesRegex(SchemaValidationError, "missing=.*heading"):
            validate_schema(schema)

    def test_legacy_synonym_or_extra_key_rejected(self):
        schema = build_schema()
        schema["motion_dim"] = 17
        with self.assertRaisesRegex(SchemaValidationError, "extra=.*motion_dim"):
            validate_schema(schema)

    def test_wrong_source_spec_commit_rejected(self):
        schema = build_schema()
        schema["provenance"]["source_plan_commit"] = "0" * 40
        with self.assertRaisesRegex(SchemaValidationError, "source_plan_commit"):
            validate_schema(schema)

    def test_nonroot_root_only_channels_marked_valid_rejected(self):
        schema = build_schema()
        schema["channel_validity"]["physical_non_root"][16] = True
        with self.assertRaisesRegex(SchemaValidationError, "physical_non_root"):
            validate_schema(schema)

    def test_fps_mismatch_rejected(self):
        with self.assertRaisesRegex(SchemaValidationError, "mismatch"):
            validate_schema(build_schema(fps_target=30.0), expected_fps_target=24.0)

    def test_nonfinite_schema_number_rejected(self):
        schema = build_schema()
        schema["smoother"]["params"]["margin_norm"] = math.inf
        with self.assertRaisesRegex(SchemaValidationError, "must be finite"):
            validate_schema(schema)

    def test_unrepresentable_numeric_value_is_cleanly_rejected(self):
        with self.assertRaisesRegex(SchemaValidationError, "representable finite number"):
            build_schema(fps_target=10 ** 10000)

    def test_bool_is_not_accepted_as_numeric_fps(self):
        schema = build_schema()
        schema["fps_target"] = True
        with self.assertRaisesRegex(SchemaValidationError, "expected finite number"):
            validate_schema(schema)

    def test_parent_tree_and_physical_node_kinds(self):
        validate_physical_parent_tree([-1, 0, 1, 1], node_kinds=["physical_joint"] * 4)
        with self.assertRaisesRegex(SchemaValidationError, "parent < child"):
            validate_physical_parent_tree(
                [-1, 0, 2], node_kinds=["physical_joint"] * 3
            )
        with self.assertRaisesRegex(SchemaValidationError, "WORLD/control"):
            validate_physical_parent_tree(
                [-1, 0], node_kinds=["physical_joint", "WORLD"]
            )

    def test_parent_tree_requires_node_kind_evidence(self):
        with self.assertRaisesRegex(SchemaValidationError, "required to prove"):
            validate_physical_parent_tree([-1, 0])

    def test_schema_import_does_not_import_training_stack(self):
        root = Path(__file__).resolve().parents[1]
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import src.data.ktjd17.schema; "
                    "assert 'torch' not in sys.modules; "
                    "assert 'src.data.unified_dataset' not in sys.modules"
                ),
            ],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(probe.returncode, 0, probe.stderr)

    def test_meter_claim_requires_source_unit_evidence(self):
        metadata = {
            "length_unit_id": "source_native_unknown",
            "source_unit_to_meter": None,
            "canonical_scale_factor": 1.0,
            "s_rig": 2.0,
        }
        validate_unit_metadata(metadata, claims_meters=False)
        with self.assertRaisesRegex(SchemaValidationError, "meter claim requires"):
            validate_unit_metadata(metadata, claims_meters=True)
        metadata["source_unit_to_meter"] = 0.01
        validate_unit_metadata(metadata, claims_meters=True)

    def test_roundtrip_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schema.json"
            candidate = build_schema()
            write_schema(candidate, path)
            self.assertEqual(load_schema(path), candidate)
            write_schema(candidate, path)  # identical write is idempotent

            changed = copy.deepcopy(candidate)
            changed["fps_target"] = 24.0
            with self.assertRaises(FileExistsError):
                write_schema(changed, path)
            write_schema(changed, path, overwrite=True)
            self.assertEqual(load_schema(path)["fps_target"], 24.0)

    def test_frozen_schema_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schema.json"
            frozen = _frozen_schema()
            write_schema(frozen, path)
            changed = copy.deepcopy(frozen)
            changed["fps_target"] = 24.0
            with self.assertRaisesRegex(SchemaValidationError, "refusing to overwrite frozen"):
                write_schema(changed, path, overwrite=True)

    def test_json_file_with_nan_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schema.json"
            schema = build_schema()
            schema["fps_target"] = float("nan")
            path.write_text(json.dumps(schema), encoding="utf-8")
            with self.assertRaisesRegex(SchemaValidationError, "must be finite"):
                load_schema(path)


if __name__ == "__main__":
    unittest.main()
