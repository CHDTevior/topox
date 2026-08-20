"""CPU-only tests for the KTJD-17 T02 source inventory."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ktjd17.bvh_inventory import (  # noqa: E402
    BvhInventoryError,
    parse_bvh_header,
)
from src.data.ktjd17.inventory import (  # noqa: E402
    InventoryConfig,
    _build_joint_map,
    _load_split_map,
    _round_robin_select,
    _status_from_codes,
    _truebones_action_key,
    _validate_current_cond,
    classify_topology_family,
    run_inventory,
)
from src.data.ktjd17.inventory_validation import (  # noqa: E402
    InventoryValidationError,
    validate_inventory_outputs,
)


def _bvh(
    *,
    frames: int = 2,
    frame_time: float = 1.0 / 30.0,
    child_channels: str = "CHANNELS 3 Zrotation Yrotation Xrotation",
    named_end: bool = True,
) -> str:
    end = "End Site #name: Tip" if named_end else "End Site"
    channel_count = 9 if child_channels.endswith("Xrotation") else 8
    rows = "\n".join(" ".join(["0"] * channel_count) for _ in range(frames))
    return f"""HIERARCHY
ROOT Root
{{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
  JOINT Bone
  {{
    OFFSET 0 1 0
    {child_channels}
    {end}
    {{
      OFFSET 0 1 0
    }}
  }}
}}
MOTION
Frames: {frames}
Frame Time: {frame_time}
{rows}
"""


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _cond_record(names: list[str], parents: list[int]) -> dict:
    return {
        "joints_names": np.asarray(names, dtype=object),
        "parents": np.asarray(parents, dtype=np.int64),
        "offsets": np.zeros((len(names), 3), dtype=np.float32),
    }


class BvhHeaderTests(unittest.TestCase):
    def test_named_end_site_is_fixed_dof(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.bvh"
            _write(path, _bvh())
            header = parse_bvh_header(path)
            self.assertEqual(header.joint_names, ("Root", "Bone", "Tip"))
            self.assertEqual(header.parents, (-1, 0, 1))
            self.assertEqual(
                header.rotation_source_kinds(),
                ("animated_dof", "animated_dof", "fixed_dof"),
            )
            self.assertEqual(header.frames, 2)
            self.assertAlmostEqual(header.fps, 30.0)

    def test_unnamed_end_site_cannot_accidentally_match_tip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.bvh"
            _write(path, _bvh(named_end=False))
            header = parse_bvh_header(path)
            self.assertEqual(header.joint_names[-1], "Bone__unnamed_end_site_0")

    def test_partial_rotation_channels_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.bvh"
            _write(path, _bvh(child_channels="CHANNELS 2 Zrotation Xrotation"))
            with self.assertRaisesRegex(BvhInventoryError, "one X/Y/Z"):
                parse_bvh_header(path)

    def test_nonfinite_frame_time_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.bvh"
            _write(path, _bvh(frame_time=float("nan")))
            with self.assertRaisesRegex(BvhInventoryError, "finite and positive"):
                parse_bvh_header(path)


class InventoryHelperTests(unittest.TestCase):
    def test_joint_map_preserves_source_fixed_dof(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.bvh"
            _write(path, _bvh())
            header = parse_bvh_header(path)
            cond = _validate_current_cond(
                {"Rig": _cond_record(["Root", "Bone", "Tip"], [-1, 0, 1])}
            )
            mapping, codes, diagnostics = _build_joint_map(
                "Rig", cond["Rig"], header, "truebones"
            )
            self.assertEqual(mapping["status"], "binary_proven")
            self.assertEqual(
                mapping["rotation_source_kind"],
                ["animated_dof", "animated_dof", "fixed_dof"],
            )
            self.assertEqual(codes, [])
            self.assertEqual(diagnostics, [])

    def test_joint_map_records_skipped_source_edge(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.bvh"
            _write(path, _bvh())
            header = parse_bvh_header(path)
            cond = _validate_current_cond(
                {"Rig": _cond_record(["Root", "Tip"], [-1, 0])}
            )
            mapping, codes, _ = _build_joint_map(
                "Rig", cond["Rig"], header, "truebones"
            )
            self.assertEqual(mapping["status"], "binary_proven")
            self.assertEqual(mapping["source_skipping_edge_count"], 1)
            self.assertIn("JOINT_MAP_SKIPS_SOURCE_JOINTS", codes)

    def test_legacy_named_end_site_maps_to_unique_unnamed_source_site(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.bvh"
            _write(path, _bvh(named_end=False))
            header = parse_bvh_header(path)
            cond = _validate_current_cond(
                {
                    "Rig": _cond_record(
                        ["Root", "Bone", "Bone_end_site"], [-1, 0, 1]
                    )
                }
            )
            mapping, codes, diagnostics = _build_joint_map(
                "Rig", cond["Rig"], header, "truebones"
            )
            self.assertEqual(mapping["status"], "binary_proven")
            self.assertEqual(mapping["rotation_source_kind"][-1], "fixed_dof")
            self.assertEqual(len(mapping["structural_unnamed_end_site_maps"]), 1)
            self.assertEqual(codes, [])
            self.assertEqual(diagnostics, [])

    def test_topology_taxonomy_is_explicit(self):
        self.assertEqual(classify_topology_family("HML3D_Human", 4), "human")
        self.assertEqual(classify_topology_family("Anaconda", 12), "snake")
        self.assertEqual(classify_topology_family("Spider", 21), "spider_crab")
        self.assertEqual(classify_topology_family("Bird", 8), "winged")
        self.assertEqual(
            classify_topology_family("PZ_African_Elephant_Male", 18),
            "dragon_or_deep_topology",
        )
        self.assertEqual(classify_topology_family("Horse", 14), "quadruped")

    def test_truebones_action_key_only_removes_global_counter(self):
        self.assertEqual(
            _truebones_action_key("Fox_-_Die2_366", "Fox"), "-_Die2"
        )
        self.assertEqual(
            _truebones_action_key("Dragon___Attack_292", "Dragon"), "__Attack"
        )

    def test_reject_reason_dominates_review(self):
        self.assertEqual(
            _status_from_codes(
                ["HEADING_PAYLOAD_UNREVIEWED", "SOURCE_FILE_MISSING"]
            ),
            "reject",
        )

    def test_split_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(root / "train.txt", "a.npy\n")
            _write(root / "val.txt", "a.npy\n")
            _write(root / "held_representative.txt", "")
            _write(root / "held_stress.txt", "")
            with self.assertRaisesRegex(Exception, "split overlap"):
                _load_split_map(root)


class TinyEndToEndInventoryTest(unittest.TestCase):
    def test_live_fixture_writes_rotation_proven_review_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset_current"
            motions = dataset / "motions"
            motions.mkdir(parents=True)
            cond = {
                "HML3D_Human": _cond_record(
                    [f"j{i}" for i in range(22)], [-1] + [0] * 21
                ),
                "PZ_Wolf": _cond_record(["Root", "Bone"], [-1, 0]),
                "Snake": _cond_record(["Root", "Bone"], [-1, 0]),
            }
            np.save(dataset / "cond.npy", cond, allow_pickle=True)
            np.save(
                motions / "HML3D_Human_000001.npy",
                np.zeros((3, 22, 13), dtype=np.float32),
            )
            np.save(
                motions / "PZ_Wolf_walk_1.npy",
                np.zeros((1, 2, 13), dtype=np.float32),
            )
            np.save(
                motions / "Snake___Move_1.npy",
                np.zeros((1, 2, 13), dtype=np.float32),
            )
            np.save(
                motions / "Snake___Move_2.npy",
                np.zeros((1, 2, 13), dtype=np.float32),
            )

            splits = root / "splits"
            _write(
                splits / "train.txt",
                "HML3D_Human_000001.npy\nPZ_Wolf_walk_1.npy\nSnake___Move_1.npy\n",
            )
            _write(splits / "val.txt", "Snake___Move_2.npy\n")
            for name in ("held_representative", "held_stress"):
                _write(splits / f"{name}.txt", "")

            pz = root / "pz"
            _write(pz / "PZ_Wolf_walk_1.bvh", _bvh())
            truebones = root / "truebones" / "Snake"
            _write(truebones / "__Tpose.bvh", _bvh(frames=1))
            _write(truebones / "__Move.bvh", _bvh(frames=4))
            human = root / "human272" / "motion_data"
            human.mkdir(parents=True)
            np.save(human / "000001.npy", np.zeros((6, 272), dtype=np.float64))
            builder = root / "builder.py"
            model = root / "neutral_model.npz"
            lineage = root / "lineage.md"
            _write(builder, "# fixture\n")
            np.savez(model, fixture=np.array([1]))
            _write(lineage, "fixture lineage\n")
            output = root / "out"

            summary = run_inventory(
                InventoryConfig(
                    dataset_root=dataset,
                    split_root=splits,
                    pz_bvh_root=pz,
                    truebones_raw_root=root / "truebones",
                    human272_root=root / "human272",
                    output_root=output,
                    human_builder_path=builder,
                    smpl_neutral_model_path=model,
                    planetzoo_lineage_path=lineage,
                    workers=2,
                    prototype_min_train_clips=1,
                )
            )
            self.assertEqual(summary["fresh_counts"]["current_btjd_clips"], 4)
            self.assertTrue(output.is_symlink())
            self.assertTrue((output / "inventory_generation.json").is_file())
            records = [
                json.loads(line)
                for line in (output / "clips.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(records), 4)
            self.assertTrue(
                all(record["rotation_provenance"]["status"] == "proven" for record in records)
            )
            snake_records = [record for record in records if record["rig_id"] == "Snake"]
            self.assertTrue(all(record["status"] == "reject" for record in snake_records))
            self.assertTrue(
                all(
                    "RAW_SOURCE_SEQUENCE_SPLIT_OVERLAP" in record["reason_codes"]
                    and not record["split_eligible_for_train_calibration"]
                    and not record["prototype_candidate"]
                    for record in snake_records
                )
            )
            self.assertEqual(
                summary["raw_source_split_audit"]["cross_split_source_count"], 1
            )
            self.assertEqual(
                json.loads((output / "inventory_summary.json").read_text())[
                    "fresh_counts"
                ]["current_rigs"],
                3,
            )
            validation = validate_inventory_outputs(
                output, dataset_root=dataset, split_root=splits
            )
            self.assertEqual(validation["status"], "pass")
            self.assertEqual(
                validation["validated_counts"]["current_btjd_clips"], 4
            )
            with self.assertRaises(FileExistsError):
                run_inventory(
                    InventoryConfig(
                        dataset_root=dataset,
                        split_root=splits,
                        pz_bvh_root=pz,
                        truebones_raw_root=root / "truebones",
                        human272_root=root / "human272",
                        output_root=output,
                        human_builder_path=builder,
                        smpl_neutral_model_path=model,
                        planetzoo_lineage_path=lineage,
                        workers=1,
                        prototype_min_train_clips=1,
                    )
                )
            old_generation = output.resolve()
            run_inventory(
                InventoryConfig(
                    dataset_root=dataset,
                    split_root=splits,
                    pz_bvh_root=pz,
                    truebones_raw_root=root / "truebones",
                    human272_root=root / "human272",
                    output_root=output,
                    human_builder_path=builder,
                    smpl_neutral_model_path=model,
                    planetzoo_lineage_path=lineage,
                    workers=1,
                    overwrite=True,
                    prototype_min_train_clips=1,
                )
            )
            self.assertTrue(output.is_symlink())
            self.assertNotEqual(output.resolve(), old_generation)
            self.assertEqual(
                validate_inventory_outputs(
                    output, dataset_root=dataset, split_root=splits
                )["status"],
                "pass",
            )

    def test_fail_closed_invalid_rig_is_a_valid_inventory_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset_current"
            motions = dataset / "motions"
            motions.mkdir(parents=True)
            np.save(
                dataset / "cond.npy",
                {"Broken": _cond_record(["Root", "Missing"], [-1, 0])},
                allow_pickle=True,
            )
            np.save(
                motions / "Broken___Move_1.npy",
                np.zeros((1, 2, 13), dtype=np.float32),
            )
            splits = root / "splits"
            _write(splits / "train.txt", "Broken___Move_1.npy\n")
            for name in ("val", "held_representative", "held_stress"):
                _write(splits / f"{name}.txt", "")

            pz = root / "pz"
            pz.mkdir()
            truebones = root / "truebones" / "Broken"
            _write(truebones / "__Tpose.bvh", _bvh(frames=1))
            action = truebones / "__Move.bvh"
            _write(action, _bvh(frames=2))
            human = root / "human272" / "motion_data"
            human.mkdir(parents=True)
            builder = root / "builder.py"
            model = root / "neutral_model.npz"
            lineage = root / "lineage.md"
            _write(builder, "# fixture\n")
            np.savez(model, fixture=np.array([1]))
            _write(lineage, "fixture lineage\n")
            output = root / "out"

            summary = run_inventory(
                InventoryConfig(
                    dataset_root=dataset,
                    split_root=splits,
                    pz_bvh_root=pz,
                    truebones_raw_root=root / "truebones",
                    human272_root=root / "human272",
                    output_root=output,
                    human_builder_path=builder,
                    smpl_neutral_model_path=model,
                    planetzoo_lineage_path=lineage,
                    workers=1,
                    prototype_min_train_clips=1,
                )
            )
            self.assertEqual(summary["fresh_counts"]["current_max_physical_joints"], 2)
            rig = json.loads((output / "rigs.jsonl").read_text().splitlines()[0])
            clip = json.loads((output / "clips.jsonl").read_text().splitlines()[0])
            self.assertEqual(rig["rotation_provenance_status"], "invalid")
            self.assertEqual(rig["status"], "reject")
            self.assertIn("JOINT_MAP_MISSING", rig["reason_codes"])
            self.assertEqual(clip["rotation_provenance"]["status"], "invalid")
            self.assertEqual(clip["status"], "reject")
            self.assertFalse(clip["split_eligible_for_train_calibration"])
            self.assertEqual(
                validate_inventory_outputs(
                    output, dataset_root=dataset, split_root=splits
                )["status"],
                "pass",
            )

            _write(action, _bvh(frames=3))
            with self.assertRaisesRegex(
                InventoryValidationError, "source (T_src|mtime_ns|file_size_bytes)"
            ):
                validate_inventory_outputs(
                    output, dataset_root=dataset, split_root=splits
                )


if __name__ == "__main__":
    unittest.main()
