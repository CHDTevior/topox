"""Gold tests for the exhaustive PlanetZoo-311 source audit."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import shutil
import sys
import tempfile
import time
import unittest
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ktjd17.bvh_inventory import parse_bvh_header  # noqa: E402
from src.data.ktjd17.pz312_audit import (  # noqa: E402
    DUAL_ROTATION_MAX_ABS,
    AUDIT_APPROVAL_DIRECTORY,
    AUDIT_APPROVAL_LINK_NAME,
    AUDIT_GENERATION_DIRECTORY,
    AUDIT_LINK_NAME,
    CANDIDATE_STATUS,
    EXPECTED_PARENT_MANIFEST_FILES,
    INDEPENDENT_DECODER_ID,
    PZ312_AUDIT_VERSION,
    PzAuditConfig,
    Pz312AuditError,
    _assert_manifest_disk_bijection,
    _activate_approved_generation,
    _canonical_json,
    _chunk_sha256,
    _file_manifest,
    _freeze_immutable_tree,
    _distribution_content_fingerprint,
    _initialize_worker,
    _load_valid_chunk,
    _revalidate_all_live_sources,
    _records_sha256,
    _replace_symlink,
    _single_thread_spawn_environment,
    _select_representatives,
    _source_snapshot_sha256,
    _validate_parent_manifest_generation,
    _worker_process_status,
    independent_scipy_bvh_check,
    run_pz_source_audit,
    validate_active_pz_audit,
    validate_pz_audit_generation,
)
from src.data.ktjd17.encoder import write_npz_atomic  # noqa: E402
from src.data.ktjd17.inventory_validation import EXPECTED_VERSION  # noqa: E402
import src.data.ktjd17.planetzoo_fixed_rig as pz_fixed_module  # noqa: E402
import src.data.ktjd17.pz312_audit as pz_audit_module  # noqa: E402
from src.data.ktjd17.source_parser import parse_bvh_source  # noqa: E402


def _fixture_bvh() -> str:
    return """HIERARCHY
ROOT def_c_root_joint
{
  OFFSET 0 1 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT def_c_hips_joint
  {
    OFFSET 0 -0.5 0
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT def_c_chest_joint
    {
      OFFSET 0 0 1
      CHANNELS 3 Yrotation Zrotation Xrotation
    }
  }
}
MOTION
Frames: 3
Frame Time: 0.041667
0 1 0 0 0 0 0 0 0 0 0 0
0.2 1 0.3 10 20 30 -4 5 6 7 8 9
0.4 1 0.6 -20 15 80 3 -6 11 -17 2 4
"""


def _snapshot_task(path: Path, *, clip_id: str | None = None) -> dict:
    stat = path.lstat()
    return {
        "clip_id": clip_id or path.stem,
        "source_path": str(path.resolve()),
        "file_size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "source_device": int(stat.st_dev),
        "source_inode": int(stat.st_ino),
        "source_nlink": int(stat.st_nlink),
    }


def _cache_fixture(root: Path) -> tuple[Path, dict, dict]:
    source = root / "PZ_Fixture.bvh"
    source.write_text(_fixture_bvh(), encoding="utf-8")
    header = parse_bvh_header(source)
    stat = source.lstat()
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    task = {
        "clip_id": source.stem,
        "rig_id": "PZ_FixtureRig",
        "topology_family": "quadruped",
        "split": "train",
        "parent_status": "review",
        "source_path": str(source.resolve()),
        "slice_frames": [0, 3],
        "file_size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "source_device": int(stat.st_dev),
        "source_inode": int(stat.st_ino),
        "source_nlink": int(stat.st_nlink),
        "T_src": 3,
        "frame_time_src": 0.041667,
        "fps_src": 1.0 / 0.041667,
        "source_joint_count": 3,
        "source_channel_count": 12,
        "retained_joint_count": 3,
        "rotation_layout_sha256": header.rotation_layout_sha256(),
    }
    record = {
        "audit_version": PZ312_AUDIT_VERSION,
        "clip_id": task["clip_id"],
        "rig_id": task["rig_id"],
        "source_family": "planetzoo",
        "topology_family": task["topology_family"],
        "split": task["split"],
        "parent_status": task["parent_status"],
        "status": "pass",
        "reason_codes": [],
        "source_path": task["source_path"],
        "source_sha256": source_sha256,
        "source_size_bytes": task["file_size_bytes"],
        "source_mtime_ns": task["mtime_ns"],
        "source_device": task["source_device"],
        "source_inode": task["source_inode"],
        "source_nlink": task["source_nlink"],
        "slice_frames": task["slice_frames"],
        "T_src": task["T_src"],
        "J_phys": task["retained_joint_count"],
        "frame_time_src": task["frame_time_src"],
        "fps_src": task["fps_src"],
        "source_joint_count": task["source_joint_count"],
        "source_channel_count": task["source_channel_count"],
        "rotation_layout_sha256": task["rotation_layout_sha256"],
        "rest_layout_sha256": header.rest_layout_sha256(),
        "metrics": {
            "planetzoo_per_clip_declared_offset_exact": 1,
            "planetzoo_per_clip_rotation_layout_exact": 1,
            "planetzoo_per_clip_rest_layout_exact": 1,
            "planetzoo_root_translation_exact": 1,
            "planetzoo_fixed_fk_source_position_max_norm": 0.0,
            "planetzoo_fixed_fk_source_position_mpjpe_norm": 0.0,
            "planetzoo_stage2_contract": "ktjd17-planetzoo-stage2-fixed-rig-v1",
            "independent_decoder": INDEPENDENT_DECODER_ID,
            "independent_source_sha256": source_sha256,
            "independent_rotation_max_abs": 0.0,
            "independent_root_translation_max_abs": 0.0,
            "independent_frame_count": 3,
            "independent_frame_time_src": task["frame_time_src"],
            "independent_fps_src": task["fps_src"],
            "independent_source_joint_count": task["source_joint_count"],
            "independent_source_channel_count": task["source_channel_count"],
            "independent_rotation_layout_sha256": task[
                "rotation_layout_sha256"
            ],
            "independent_rest_layout_sha256": header.rest_layout_sha256(),
            "root_speed_rms_norm_per_s": 0.0,
            "rotation_speed_rms_rad_per_s": 0.0,
            "dynamic_score": 0.0,
        },
    }
    return source, task, record


def _miniature_rig_record(source: Path) -> tuple[dict, dict]:
    header = parse_bvh_header(source)
    names = list(header.joint_names)
    parents = list(header.parents)
    offsets = np.asarray([joint.offset for joint in header.joints], dtype=np.float64)
    kinds = [joint.rotation_source_kind() for joint in header.joints]
    rig_id = "PZ_FixtureRig"
    rig = {
        "rig_id": rig_id,
        "source_family": "planetzoo",
        "topology_family": "quadruped",
        "rest_pose": {
            "source_path": str(source.resolve()),
            "rest_layout_sha256": header.rest_layout_sha256(),
        },
        "joint_map": {
            "mapping_kind": "identity_direct_source_edges",
            "joint_map_sha256": "1" * 64,
            "source_joint_names": names,
            "source_parents": parents,
            "source_node_kinds": [joint.node_kind for joint in header.joints],
            "source_rotation_layout_sha256": header.rotation_layout_sha256(),
            "source_root_index_for_btjd_root": 0,
            "btjd_joint_names": names,
            "btjd_parents": parents,
            "btjd_to_source": list(range(len(names))),
            "rotation_source_kind": kinds,
            "direct_source_edge_count": len(names) - 1,
            "source_skipping_edge_count": 0,
            "animated_dof_count": sum(kind == "animated_dof" for kind in kinds),
            "fixed_dof_count": sum(kind == "fixed_dof" for kind in kinds),
        },
    }
    cond = {
        "joints_names": names,
        "parents": np.asarray(parents, dtype=np.int64),
        "offsets": offsets,
    }
    return rig, cond


def _miniature_clip_record(source: Path, rig_id: str) -> dict:
    header = parse_bvh_header(source)
    observed = source.lstat()
    return {
        "clip_id": source.stem,
        "rig_id": rig_id,
        "topology_family": "quadruped",
        "split": "train",
        "status": "review",
        "source": {
            "family": "planetzoo",
            "path": str(source.resolve()),
            "slice_frames": [0, header.frames],
            "T_src": header.frames,
            "frame_time_src": header.frame_time,
            "fps_src": header.fps,
            "file_size_bytes": int(observed.st_size),
            "mtime_ns": int(observed.st_mtime_ns),
            "source_joint_count": len(header.joints),
            "source_channel_count": header.channel_count,
            "rotation_layout_sha256": header.rotation_layout_sha256(),
        },
    }


def _write_miniature_parent(root: Path, rig: dict, clips: list[dict]) -> None:
    root.mkdir(parents=True)
    for name in sorted(EXPECTED_PARENT_MANIFEST_FILES):
        path = root / name
        if name == "rigs.jsonl":
            path.write_bytes(_canonical_json(rig) + b"\n")
        elif name == "clips.jsonl":
            path.write_bytes(b"".join(_canonical_json(clip) + b"\n" for clip in clips))
        else:
            path.write_text("{}\n", encoding="utf-8")
    files = {
        name: {
            "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
            "size_bytes": int((root / name).stat().st_size),
        }
        for name in sorted(EXPECTED_PARENT_MANIFEST_FILES)
    }
    transaction = {
        "files": files,
        "generation_id": root.name,
        "manifest_version": EXPECTED_VERSION,
        "publish_protocol": "immutable_generation_atomic_symlink_replace",
    }
    (root / "inventory_generation.json").write_bytes(
        _canonical_json(transaction) + b"\n"
    )


def _refresh_generation_closure(root: Path) -> None:
    path = root / "generation.json"
    generation = json.loads(path.read_text(encoding="utf-8"))
    generation["files"] = _file_manifest(root)
    path.write_bytes(_canonical_json(generation) + b"\n")
    _freeze_immutable_tree(root)


def _make_tree_writable(root: Path) -> None:
    os.chmod(root, root.stat().st_mode | 0o700)
    for path in root.rglob("*"):
        observed = path.lstat()
        if path.is_dir():
            os.chmod(path, observed.st_mode | 0o700)
        elif path.is_file():
            os.chmod(path, observed.st_mode | 0o600)


def _clone_generation(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    _make_tree_writable(destination)
    summary_path = destination / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["generation_id"] = destination.name
    summary_path.write_bytes(_canonical_json(summary) + b"\n")
    generation_path = destination / "generation.json"
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    generation["generation_id"] = destination.name
    generation_path.write_bytes(_canonical_json(generation) + b"\n")
    _refresh_generation_closure(destination)
    _make_tree_writable(destination)
    return destination


def _prepare_miniature_transaction(outer: Path, *, two_clips: bool) -> dict:
    pz_root = outer / "pz"
    pz_root.mkdir()
    source_a = pz_root / "PZ_Fixture_A.bvh"
    source_a.write_text(_fixture_bvh(), encoding="utf-8")
    sources = [source_a]
    if two_clips:
        source_b = pz_root / "PZ_Fixture_B.bvh"
        source_b.write_text(
            _fixture_bvh().replace(
                "0.4 1 0.6 -20 15 80 3 -6 11 -17 2 4",
                "0.1 1 0.1 -2 1 8 0 -1 1 -2 0 1",
            ),
            encoding="utf-8",
        )
        sources.append(source_b)
    rig, cond_entry = _miniature_rig_record(source_a)
    clips = sorted(
        [_miniature_clip_record(source, rig["rig_id"]) for source in sources],
        key=lambda item: item["clip_id"],
    )
    parent = outer / "parent-generation"
    _write_miniature_parent(parent, rig, clips)
    cond_path = outer / "cond.npy"
    np.save(cond_path, {rig["rig_id"]: cond_entry}, allow_pickle=True)
    return {
        "pz_root": pz_root,
        "sources": sources,
        "rig": rig,
        "clips": clips,
        "parent": parent,
        "cond_path": cond_path,
        "cond_sha256": hashlib.sha256(cond_path.read_bytes()).hexdigest(),
        "output": outer / "output",
    }


def _worker_status_after_file_barrier(
    barrier_root: str, expected_workers: int
) -> dict[str, int]:
    """Hold each spawned child until every requested worker is observable."""
    root = Path(barrier_root)
    (root / f"{os.getpid()}.ready").write_text("ready\n", encoding="utf-8")
    deadline = time.monotonic() + 15.0
    while len(list(root.glob("*.ready"))) < expected_workers:
        if time.monotonic() >= deadline:
            raise RuntimeError("spawn-worker barrier timed out")
        time.sleep(0.01)
    return _worker_process_status()


class Pz312AuditTests(unittest.TestCase):
    def test_independent_scipy_parser_matches_every_declared_euler_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.bvh"
            path.write_text(_fixture_bvh(), encoding="utf-8")
            header = parse_bvh_header(path)
            names = list(header.joint_names)
            parents = list(header.parents)
            rig = {
                "joint_map": {
                    "btjd_joint_names": names,
                    "btjd_parents": parents,
                    "btjd_to_source": list(range(3)),
                    "rotation_source_kind": ["animated_dof"] * 3,
                }
            }
            parsed = parse_bvh_source(
                path,
                retained_names=names,
                retained_parents=parents,
                expected_rotation_kinds=["animated_dof"] * 3,
                frame_slice=[0, 3],
                rest_path=path,
                rest_mode="processed_hierarchy_stage2_fixed",
                family="planetzoo",
            )
            metrics = independent_scipy_bvh_check(
                path, rig_record=rig, parsed=parsed
            )
            self.assertLess(metrics["independent_rotation_max_abs"], 1e-12)
            self.assertEqual(metrics["independent_root_translation_max_abs"], 0.0)
            self.assertEqual(metrics["independent_frame_count"], 3)
            self.assertEqual(
                metrics["independent_rest_layout_sha256"],
                header.rest_layout_sha256(),
            )

    def test_representative_selection_is_one_per_rig_and_deterministic(self):
        records = []
        for rig_id in ("PZ_A", "PZ_B"):
            for index, score in enumerate((1.0, 3.0, 3.0)):
                records.append(
                    {
                        "clip_id": f"{rig_id}_{index}",
                        "rig_id": rig_id,
                        "status": "pass",
                        "T_src": 10 + index,
                        "source_sha256": f"{index:064x}",
                        "metrics": {"dynamic_score": score},
                    }
                )
        selected = _select_representatives(records, expected_rig_count=2)
        self.assertEqual(selected["selected_count"], 2)
        self.assertEqual(
            [record["clip_id"] for record in selected["selected"]],
            ["PZ_A_2", "PZ_B_2"],
        )

    def test_rotation_dual_decoder_threshold_is_exactly_one_e_minus_twelve(self):
        self.assertEqual(DUAL_ROTATION_MAX_ABS, 1e-12)

    def test_worker_initializer_caps_loaded_blas_pools(self):
        from threadpoolctl import threadpool_info

        with tempfile.TemporaryDirectory() as directory:
            _initialize_worker({}, {}, directory)
            pools = [
                int(item["num_threads"])
                for item in threadpool_info()
                if item.get("user_api") == "blas"
            ]
            self.assertTrue(pools)
            self.assertTrue(all(count == 1 for count in pools), pools)

    def test_spawn_workers_each_have_one_os_thread(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            barrier = root / "barrier"
            barrier.mkdir()
            context = multiprocessing.get_context("spawn")
            with _single_thread_spawn_environment():
                with ProcessPoolExecutor(
                    max_workers=2,
                    mp_context=context,
                    initializer=_initialize_worker,
                    initargs=({}, {}, str(root)),
                ) as executor:
                    futures = [
                        executor.submit(
                            _worker_status_after_file_barrier,
                            str(barrier),
                            2,
                        )
                        for _ in range(2)
                    ]
                    observed = [future.result(timeout=30.0) for future in futures]
            self.assertEqual(len({row["pid"] for row in observed}), 2, observed)
            self.assertEqual([row["threads"] for row in observed], [1, 1])

    def test_runtime_distribution_bytes_are_pinned_beyond_unchanged_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            runtime = root / "package/runtime.py"
            record = root / "package.dist-info/RECORD"
            runtime.parent.mkdir(parents=True)
            record.parent.mkdir(parents=True)
            runtime.write_text("value = 1\n", encoding="utf-8")
            record.write_text(
                "package/runtime.py,sha256=unchanged,10\n",
                encoding="utf-8",
            )

            class FakeDistribution:
                files = (
                    Path("package/runtime.py"),
                    Path("package.dist-info/RECORD"),
                )

                @staticmethod
                def locate_file(relative: Path) -> Path:
                    return root / relative

            before_record = hashlib.sha256(record.read_bytes()).hexdigest()
            before = _distribution_content_fingerprint(FakeDistribution())
            runtime.write_text("value = 2\n", encoding="utf-8")
            after = _distribution_content_fingerprint(FakeDistribution())
            self.assertEqual(
                hashlib.sha256(record.read_bytes()).hexdigest(), before_record
            )
            self.assertNotEqual(
                before["actual_files_sha256"], after["actual_files_sha256"]
            )

    def test_cached_pass_is_bound_to_current_source_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source, task, record = _cache_fixture(root)
            authority = "a" * 64
            chunk = root / "chunk.json"
            chunk.write_bytes(
                _canonical_json(
                    {
                        "audit_version": PZ312_AUDIT_VERSION,
                        "authority_sha256": authority,
                        "task_sha256": _chunk_sha256([task]),
                        "records_sha256": _records_sha256([record]),
                        "worker_process_status": None,
                        "records": [record],
                    }
                )
                + b"\n"
            )
            self.assertIsNotNone(
                _load_valid_chunk(
                    chunk,
                    tasks=[task],
                    authority_sha256=authority,
                    pz_root=root,
                )
            )
            source.write_text(_fixture_bvh() + "\n", encoding="utf-8")
            self.assertIsNone(
                _load_valid_chunk(
                    chunk,
                    tasks=[task],
                    authority_sha256=authority,
                    pz_root=root,
                )
            )
            source.unlink()
            self.assertIsNone(
                _load_valid_chunk(
                    chunk,
                    tasks=[task],
                    authority_sha256=authority,
                    pz_root=root,
                )
            )

    def test_corrupt_chunk_is_never_resumed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            _, task, _ = _cache_fixture(root)
            chunk = root / "chunk.json"
            chunk.write_text("{", encoding="utf-8")
            self.assertIsNone(
                _load_valid_chunk(
                    chunk,
                    tasks=[task],
                    authority_sha256="a" * 64,
                    pz_root=root,
                )
            )

    def test_semantically_forged_cached_pass_is_never_resumed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            _, task, baseline = _cache_fixture(root)
            authority = "b" * 64
            chunk = root / "chunk.json"
            mutations = (
                ("inflated dynamic score", "dynamic_score", 9.99e99),
                (
                    "negative FK error",
                    "planetzoo_fixed_fk_source_position_max_norm",
                    -1.0,
                ),
                (
                    "wrong independent rest layout",
                    "independent_rest_layout_sha256",
                    "0" * 64,
                ),
            )
            for label, field, value in mutations:
                with self.subTest(label):
                    record = json.loads(json.dumps(baseline))
                    record["metrics"][field] = value
                    chunk.write_bytes(
                        _canonical_json(
                            {
                                "audit_version": PZ312_AUDIT_VERSION,
                                "authority_sha256": authority,
                                "task_sha256": _chunk_sha256([task]),
                                "records_sha256": _records_sha256([record]),
                                "worker_process_status": None,
                                "records": [record],
                            }
                        )
                        + b"\n"
                    )
                    self.assertIsNone(
                        _load_valid_chunk(
                            chunk,
                            tasks=[task],
                            authority_sha256=authority,
                            pz_root=root,
                        )
                    )

    def test_final_recheck_rejects_source_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source, task, record = _cache_fixture(root)
            source.write_text(_fixture_bvh() + "\n", encoding="utf-8")
            with self.assertRaises(Pz312AuditError):
                _revalidate_all_live_sources(
                    [record], [task], pz_root=root, workers=1
                )

    def test_manifest_disk_bijection_rejects_extra_missing_symlink_and_hardlink(self):
        with self.subTest("extra"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                first = root / "PZ_A.bvh"
                first.write_text("a", encoding="utf-8")
                task = _snapshot_task(first)
                (root / "PZ_EXTRA.bvh").write_text("x", encoding="utf-8")
                with self.assertRaises(Pz312AuditError):
                    _assert_manifest_disk_bijection(root, [task])
        with self.subTest("missing"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                first = root / "PZ_A.bvh"
                first.write_text("a", encoding="utf-8")
                task = _snapshot_task(first)
                first.unlink()
                with self.assertRaises(Pz312AuditError):
                    _assert_manifest_disk_bijection(root, [task])
        with self.subTest("symlink"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                first = root / "PZ_A.bvh"
                first.write_text("a", encoding="utf-8")
                task = _snapshot_task(first)
                os.symlink(first.name, root / "PZ_LINK.bvh")
                with self.assertRaises(Pz312AuditError):
                    _assert_manifest_disk_bijection(root, [task])
        with self.subTest("hardlink"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                first = root / "PZ_A.bvh"
                second = root / "PZ_B.bvh"
                first.write_text("a", encoding="utf-8")
                os.link(first, second)
                tasks = [_snapshot_task(first), _snapshot_task(second)]
                with self.assertRaises(Pz312AuditError):
                    _assert_manifest_disk_bijection(root, tasks)
        with self.subTest("external hardlink alias"):
            with tempfile.TemporaryDirectory() as directory:
                outer = Path(directory).resolve()
                root = outer / "pz"
                root.mkdir()
                first = root / "PZ_A.bvh"
                first.write_text("a", encoding="utf-8")
                task = _snapshot_task(first)
                os.link(first, outer / "outside_alias.bvh")
                with self.assertRaises(Pz312AuditError):
                    _assert_manifest_disk_bijection(root, [task])

    def test_immutable_manifest_rejects_external_hardlink_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            outer = Path(directory).resolve()
            root = outer / "generation"
            root.mkdir()
            artifact = root / "summary.json"
            artifact.write_text("{}\n", encoding="utf-8")
            os.link(artifact, outer / "outside_alias.json")
            with self.assertRaises(Pz312AuditError):
                _file_manifest(root)

    def test_parent_generation_hash_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "parent-generation"
            root.mkdir()
            files = {}
            for name in sorted(EXPECTED_PARENT_MANIFEST_FILES):
                path = root / name
                path.write_text(f"{name}\n", encoding="utf-8")
                files[name] = {
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "size_bytes": path.stat().st_size,
                }
            transaction = {
                "files": files,
                "generation_id": root.name,
                "manifest_version": EXPECTED_VERSION,
                "publish_protocol": "immutable_generation_atomic_symlink_replace",
            }
            (root / "inventory_generation.json").write_text(
                json.dumps(transaction), encoding="utf-8"
            )
            self.assertEqual(
                _validate_parent_manifest_generation(root)["generation_id"], root.name
            )
            (root / "clips.jsonl").write_text("drift\n", encoding="utf-8")
            with self.assertRaises(Pz312AuditError):
                _validate_parent_manifest_generation(root)

    def test_post_publish_deep_failure_quarantines_pending_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            outer = Path(directory).resolve()
            fixture = _prepare_miniature_transaction(outer, two_clips=False)
            with (
                mock.patch.object(pz_audit_module, "EXPECTED_PZ_RIG_COUNT", 1),
                mock.patch.object(pz_audit_module, "EXPECTED_PZ_CLIP_COUNT", 1),
                mock.patch.object(
                    pz_audit_module, "ACTIVE_COND_SHA256", fixture["cond_sha256"]
                ),
                mock.patch.object(
                    pz_fixed_module, "ACTIVE_COND_SHA256", fixture["cond_sha256"]
                ),
                mock.patch.object(
                    pz_audit_module,
                    "_validate_pz_audit_candidate",
                    side_effect=Pz312AuditError("injected post-publish failure"),
                ),
            ):
                with self.assertRaisesRegex(
                    Pz312AuditError, "injected post-publish failure"
                ):
                    run_pz_source_audit(
                        PzAuditConfig(
                            manifest_root=fixture["parent"],
                            pz_bvh_root=fixture["pz_root"],
                            active_cond_path=fixture["cond_path"],
                            output_root=fixture["output"],
                            workers=1,
                            chunk_size=1,
                            update_link=True,
                        )
                    )

            generations = fixture["output"] / AUDIT_GENERATION_DIRECTORY
            visible = [path for path in generations.iterdir() if not path.name.startswith(".")]
            rejected = [
                path
                for path in generations.iterdir()
                if path.name.startswith(".rejected-")
            ]
            self.assertEqual(visible, [])
            self.assertEqual(len(rejected), 1)
            generation = json.loads(
                (rejected[0] / "generation.json").read_text(encoding="utf-8")
            )
            summary = json.loads(
                (rejected[0] / "summary.json").read_text(encoding="utf-8")
            )
            for record in (generation, summary):
                self.assertEqual(record["status"], CANDIDATE_STATUS)
                self.assertIs(record["prototype_conversion_authorized"], False)
                self.assertIs(record["full_conversion_authorized"], False)
            self.assertFalse(
                os.path.lexists(fixture["output"] / AUDIT_LINK_NAME)
            )
            self.assertFalse(
                os.path.lexists(fixture["output"] / AUDIT_APPROVAL_LINK_NAME)
            )

    def test_post_deep_self_consistent_forgery_is_rejected_before_activation(self):
        with tempfile.TemporaryDirectory() as directory:
            outer = Path(directory).resolve()
            fixture = _prepare_miniature_transaction(outer, two_clips=True)
            real_validator = pz_audit_module._validate_pz_audit_candidate

            def validate_then_forge(root: str | Path, *, workers: int = 24) -> dict:
                proof = real_validator(root, workers=workers)
                generation_root = Path(root)
                _make_tree_writable(generation_root)
                qa_path = generation_root / "qa/pz_source_audit.jsonl"
                records = [
                    json.loads(line)
                    for line in qa_path.read_text(encoding="utf-8").splitlines()
                ]
                winner = _select_representatives(
                    records, expected_rig_count=1
                )["selected"][0]["clip_id"]
                loser = next(row for row in records if row["clip_id"] != winner)
                loser["metrics"]["root_speed_rms_norm_per_s"] += 1000.0
                loser["metrics"]["dynamic_score"] = (
                    loser["metrics"]["root_speed_rms_norm_per_s"]
                    + loser["metrics"]["rotation_speed_rms_rad_per_s"]
                )
                qa_path.write_bytes(
                    b"".join(_canonical_json(row) + b"\n" for row in records)
                )
                forged_selection = _select_representatives(
                    records, expected_rig_count=1
                )
                self.assertEqual(
                    forged_selection["selected"][0]["clip_id"], loser["clip_id"]
                )
                (generation_root / "selection/pz_representatives.json").write_bytes(
                    _canonical_json(forged_selection) + b"\n"
                )
                _refresh_generation_closure(generation_root)
                return proof

            with (
                mock.patch.object(pz_audit_module, "EXPECTED_PZ_RIG_COUNT", 1),
                mock.patch.object(pz_audit_module, "EXPECTED_PZ_CLIP_COUNT", 2),
                mock.patch.object(
                    pz_audit_module, "ACTIVE_COND_SHA256", fixture["cond_sha256"]
                ),
                mock.patch.object(
                    pz_fixed_module, "ACTIVE_COND_SHA256", fixture["cond_sha256"]
                ),
                mock.patch.object(
                    pz_audit_module,
                    "_validate_pz_audit_candidate",
                    side_effect=validate_then_forge,
                ),
            ):
                with self.assertRaisesRegex(
                    Pz312AuditError, "candidate changed after deep validation returned"
                ):
                    run_pz_source_audit(
                        PzAuditConfig(
                            manifest_root=fixture["parent"],
                            pz_bvh_root=fixture["pz_root"],
                            active_cond_path=fixture["cond_path"],
                            output_root=fixture["output"],
                            workers=1,
                            chunk_size=1,
                            update_link=True,
                        )
                    )

            generations = fixture["output"] / AUDIT_GENERATION_DIRECTORY
            self.assertEqual(
                [path for path in generations.iterdir() if not path.name.startswith(".")],
                [],
            )
            rejected = [
                path
                for path in generations.iterdir()
                if path.name.startswith(".rejected-")
            ]
            self.assertEqual(len(rejected), 1)
            self.assertFalse(
                os.path.lexists(fixture["output"] / AUDIT_LINK_NAME)
            )
            self.assertFalse(
                os.path.lexists(fixture["output"] / AUDIT_APPROVAL_LINK_NAME)
            )

    def test_post_deep_same_stat_live_source_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            outer = Path(directory).resolve()
            fixture = _prepare_miniature_transaction(outer, two_clips=False)
            source = fixture["sources"][0]
            original = source.read_bytes()
            observed = source.lstat()
            mutated = original.replace(b"\n0.2 1 0.3", b"\n0.3 1 0.3", 1)
            self.assertEqual(len(mutated), len(original))
            self.assertNotEqual(mutated, original)
            real_validator = pz_audit_module._validate_pz_audit_candidate

            def validate_then_mutate_live(
                root: str | Path, *, workers: int = 24
            ) -> dict:
                proof = real_validator(root, workers=workers)
                source.write_bytes(mutated)
                os.utime(
                    source,
                    ns=(int(observed.st_atime_ns), int(observed.st_mtime_ns)),
                )
                current = source.lstat()
                self.assertEqual(
                    (
                        current.st_size,
                        current.st_mtime_ns,
                        current.st_dev,
                        current.st_ino,
                    ),
                    (
                        observed.st_size,
                        observed.st_mtime_ns,
                        observed.st_dev,
                        observed.st_ino,
                    ),
                )
                return proof

            with (
                mock.patch.object(pz_audit_module, "EXPECTED_PZ_RIG_COUNT", 1),
                mock.patch.object(pz_audit_module, "EXPECTED_PZ_CLIP_COUNT", 1),
                mock.patch.object(
                    pz_audit_module, "ACTIVE_COND_SHA256", fixture["cond_sha256"]
                ),
                mock.patch.object(
                    pz_fixed_module, "ACTIVE_COND_SHA256", fixture["cond_sha256"]
                ),
                mock.patch.object(
                    pz_audit_module,
                    "_validate_pz_audit_candidate",
                    side_effect=validate_then_mutate_live,
                ),
            ):
                with self.assertRaisesRegex(Pz312AuditError, "live source SHA drifted"):
                    run_pz_source_audit(
                        PzAuditConfig(
                            manifest_root=fixture["parent"],
                            pz_bvh_root=fixture["pz_root"],
                            active_cond_path=fixture["cond_path"],
                            output_root=fixture["output"],
                            workers=1,
                            chunk_size=1,
                            update_link=True,
                        )
                    )

            generations = fixture["output"] / AUDIT_GENERATION_DIRECTORY
            self.assertEqual(
                [path for path in generations.iterdir() if not path.name.startswith(".")],
                [],
            )
            self.assertEqual(
                len(
                    [
                        path
                        for path in generations.iterdir()
                        if path.name.startswith(".rejected-")
                    ]
                ),
                1,
            )
            self.assertFalse(
                os.path.lexists(fixture["output"] / AUDIT_LINK_NAME)
            )
            self.assertFalse(
                os.path.lexists(fixture["output"] / AUDIT_APPROVAL_LINK_NAME)
            )

    def test_miniature_generation_rejects_all_rereview_false_pass_attacks(self):
        with tempfile.TemporaryDirectory() as directory:
            outer = Path(directory).resolve()
            pz_root = outer / "pz"
            pz_root.mkdir()
            source_a = pz_root / "PZ_Fixture_A.bvh"
            source_b = pz_root / "PZ_Fixture_B.bvh"
            source_a.write_text(_fixture_bvh(), encoding="utf-8")
            source_b.write_text(
                _fixture_bvh().replace(
                    "0.4 1 0.6 -20 15 80 3 -6 11 -17 2 4",
                    "0.1 1 0.1 -2 1 8 0 -1 1 -2 0 1",
                ),
                encoding="utf-8",
            )
            rig, cond_entry = _miniature_rig_record(source_a)
            clips = sorted(
                [
                    _miniature_clip_record(source_a, rig["rig_id"]),
                    _miniature_clip_record(source_b, rig["rig_id"]),
                ],
                key=lambda item: item["clip_id"],
            )
            parent = outer / "parent-generation"
            _write_miniature_parent(parent, rig, clips)
            cond_path = outer / "cond.npy"
            np.save(cond_path, {rig["rig_id"]: cond_entry}, allow_pickle=True)
            cond_sha256 = hashlib.sha256(cond_path.read_bytes()).hexdigest()
            output = outer / "output"
            with (
                mock.patch.object(pz_audit_module, "EXPECTED_PZ_RIG_COUNT", 1),
                mock.patch.object(pz_audit_module, "EXPECTED_PZ_CLIP_COUNT", 2),
                mock.patch.object(
                    pz_audit_module, "ACTIVE_COND_SHA256", cond_sha256
                ),
                mock.patch.object(
                    pz_fixed_module, "ACTIVE_COND_SHA256", cond_sha256
                ),
            ):
                result = run_pz_source_audit(
                    PzAuditConfig(
                        manifest_root=parent,
                        pz_bvh_root=pz_root,
                        active_cond_path=cond_path,
                        output_root=output,
                        workers=2,
                        chunk_size=1,
                        update_link=True,
                    )
                )
                generation = Path(result["generation_root"])
                self.assertEqual(result["status"], "pass")
                self.assertTrue(result["prototype_conversion_authorized"])
                self.assertEqual(
                    validate_active_pz_audit(output)["generation_root"],
                    str(generation),
                )
                producer_evidence = json.loads(
                    (
                        generation / "qa/producer_worker_process_status.json"
                    ).read_text(encoding="utf-8")
                )
                producer_statuses = producer_evidence["worker_process_statuses"]
                self.assertTrue(producer_statuses)
                self.assertTrue(
                    all(row["threads"] == 1 for row in producer_statuses),
                    producer_statuses,
                )
                approval_path = Path(result["approval_path"])
                approval = json.loads(approval_path.read_text(encoding="utf-8"))
                deep_statuses = approval[
                    "deep_validator_worker_process_statuses"
                ]
                self.assertTrue(deep_statuses)
                self.assertTrue(
                    all(row["threads"] == 1 for row in deep_statuses),
                    deep_statuses,
                )

                bad_generation = _clone_generation(
                    generation,
                    generation.parent / f"{generation.name}-bad-activation",
                )
                _freeze_immutable_tree(bad_generation)
                with self.assertRaises(Pz312AuditError):
                    _activate_approved_generation(
                        output_root=output,
                        generation_root=bad_generation,
                        approval_path=approval_path,
                        workers=1,
                    )
                self.assertEqual(
                    validate_active_pz_audit(output, workers=1)["generation_root"],
                    str(generation),
                )

                _replace_symlink(output / AUDIT_LINK_NAME, bad_generation)
                with self.assertRaises(Pz312AuditError):
                    validate_active_pz_audit(output, workers=1)
                _replace_symlink(output / AUDIT_LINK_NAME, generation)
                self.assertEqual(
                    validate_active_pz_audit(output, workers=1)["generation_root"],
                    str(generation),
                )

                replay_output = outer / "replay-output"
                replay_generation = (
                    replay_output
                    / AUDIT_GENERATION_DIRECTORY
                    / generation.name
                )
                replay_approval = (
                    replay_output / AUDIT_APPROVAL_DIRECTORY / approval_path.name
                )
                replay_generation.parent.mkdir(parents=True)
                replay_approval.parent.mkdir(parents=True)
                shutil.copytree(generation, replay_generation)
                shutil.copy2(approval_path, replay_approval)
                with self.assertRaises(Pz312AuditError):
                    _activate_approved_generation(
                        output_root=replay_output,
                        generation_root=replay_generation,
                        approval_path=replay_approval,
                        workers=1,
                    )
                self.assertFalse(
                    os.path.lexists(replay_output / AUDIT_LINK_NAME)
                )
                self.assertFalse(
                    os.path.lexists(replay_output / AUDIT_APPROVAL_LINK_NAME)
                )

                original = source_a.read_bytes()
                observed = source_a.stat()
                mutated = original.replace(b"\n0.2 1 0.3", b"\n0.3 1 0.3", 1)
                self.assertEqual(len(mutated), len(original))
                self.assertNotEqual(mutated, original)
                source_a.write_bytes(mutated)
                os.utime(
                    source_a,
                    ns=(int(observed.st_atime_ns), int(observed.st_mtime_ns)),
                )
                with self.assertRaisesRegex(
                    Pz312AuditError,
                    "fixed-skeleton payload field drifted|deep live re-audit drifted",
                ):
                    validate_pz_audit_generation(generation, workers=1)
                source_a.write_bytes(original)
                os.utime(
                    source_a,
                    ns=(int(observed.st_atime_ns), int(observed.st_mtime_ns)),
                )

                sha_attack = _clone_generation(
                    generation, outer / "source-sha-attack"
                )
                qa_path = sha_attack / "qa/pz_source_audit.jsonl"
                records = [
                    json.loads(line)
                    for line in qa_path.read_text(encoding="utf-8").splitlines()
                ]
                records[0]["source_sha256"] = "0" * 64
                records[0]["metrics"]["independent_source_sha256"] = "0" * 64
                qa_path.write_bytes(
                    b"".join(_canonical_json(record) + b"\n" for record in records)
                )
                recheck_path = sha_attack / "qa/source_snapshot_recheck.json"
                recheck = json.loads(recheck_path.read_text(encoding="utf-8"))
                recheck["source_snapshot_sha256"] = _source_snapshot_sha256(records)
                recheck_path.write_bytes(_canonical_json(recheck) + b"\n")
                selection = _select_representatives(records, expected_rig_count=1)
                (sha_attack / "selection/pz_representatives.json").write_bytes(
                    _canonical_json(selection) + b"\n"
                )
                summary_path = sha_attack / "summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                summary["source_snapshot_recheck"] = recheck
                summary_path.write_bytes(_canonical_json(summary) + b"\n")
                _refresh_generation_closure(sha_attack)
                with self.assertRaisesRegex(
                    Pz312AuditError, "deep live re-audit drifted"
                ):
                    validate_pz_audit_generation(sha_attack, workers=1)

                skeleton_attack = _clone_generation(
                    generation, outer / "skeleton-attack"
                )
                skeleton_path = (
                    skeleton_attack / "skeletons" / f"{rig['rig_id']}.npz"
                )
                with np.load(skeleton_path, allow_pickle=False) as loaded:
                    payload = {name: np.asarray(loaded[name]).copy() for name in loaded.files}
                payload["s_rig"] = np.asarray(
                    float(np.asarray(payload["s_rig"]).item()) * 2.0,
                    dtype=np.float64,
                )
                skeleton_sha256 = write_npz_atomic(skeleton_path, payload)
                rig_qa_path = skeleton_attack / "qa/rig_audit.jsonl"
                rig_qa = json.loads(rig_qa_path.read_text(encoding="utf-8"))
                rig_qa["skeleton_sha256"] = skeleton_sha256
                rig_qa_path.write_bytes(_canonical_json(rig_qa) + b"\n")
                _refresh_generation_closure(skeleton_attack)
                with self.assertRaisesRegex(
                    Pz312AuditError, "fixed-skeleton payload field drifted"
                ):
                    validate_pz_audit_generation(skeleton_attack, workers=1)

                selection_attack = _clone_generation(
                    generation, outer / "selection-attack"
                )
                qa_path = selection_attack / "qa/pz_source_audit.jsonl"
                records = [
                    json.loads(line)
                    for line in qa_path.read_text(encoding="utf-8").splitlines()
                ]
                true_selection = _select_representatives(records, expected_rig_count=1)
                true_winner = true_selection["selected"][0]["clip_id"]
                loser = next(record for record in records if record["clip_id"] != true_winner)
                loser["metrics"]["root_speed_rms_norm_per_s"] += 1000.0
                loser["metrics"]["dynamic_score"] += 1000.0
                qa_path.write_bytes(
                    b"".join(_canonical_json(record) + b"\n" for record in records)
                )
                forged_selection = _select_representatives(records, expected_rig_count=1)
                self.assertEqual(
                    forged_selection["selected"][0]["clip_id"], loser["clip_id"]
                )
                (selection_attack / "selection/pz_representatives.json").write_bytes(
                    _canonical_json(forged_selection) + b"\n"
                )
                _refresh_generation_closure(selection_attack)
                with self.assertRaisesRegex(
                    Pz312AuditError, "deep live re-audit drifted"
                ):
                    validate_pz_audit_generation(selection_attack, workers=1)

    def test_count_only_empty_generation_cannot_claim_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "fake-generation"
            (root / "selection").mkdir(parents=True)
            (root / "summary.json").write_text(
                json.dumps(
                    {"status": "pass", "clip_count": 74522, "rig_count": 311}
                ),
                encoding="utf-8",
            )
            (root / "selection/pz_representatives.json").write_text(
                json.dumps({"selected_count": 311}), encoding="utf-8"
            )
            generation = {
                "audit_version": PZ312_AUDIT_VERSION,
                "generation_id": root.name,
                "status": "pass",
                "prototype_conversion_authorized": True,
                "full_conversion_authorized": False,
                "files": _file_manifest(root),
            }
            (root / "generation.json").write_text(
                json.dumps(generation), encoding="utf-8"
            )
            with self.assertRaises(Pz312AuditError):
                validate_pz_audit_generation(root)


if __name__ == "__main__":
    unittest.main()
