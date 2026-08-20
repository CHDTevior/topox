"""Focused tests for the exhaustive Human MotionStreamer272 source audit."""

from __future__ import annotations

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

import src.data.ktjd17.human312_audit as human_audit  # noqa: E402
from src.data.ktjd17.human312_audit import (  # noqa: E402
    HUMAN312_AUDIT_VERSION,
    Human312AuditError,
    _activate_approved_generation,
    _dynamic_metrics,
    _ensure_canonical_directory,
    _file_manifest,
    _first_record_difference,
    _load_valid_chunk,
    _quarantine_failed_approval,
    _read_relative_symlink_target,
    _records_sha256,
    _reject_unstable_source_records,
    _replace_symlink,
    _rigid_edge_max_norm,
    _select_representative,
    _task_scope_sha256,
    _validate_records_against_tasks,
    _verify_live_source_record,
    _write_json,
    independent_motionstreamer272_decode,
)
from src.data.ktjd17.human_source_parser import (  # noqa: E402
    FIXED_NEUTRAL_PARSER_VERSION,
    MotionStreamer272ContentError,
    parse_motionstreamer272_fixed_neutral,
    parse_motionstreamer272_fixed_neutral_array,
)
from src.data.ktjd17.source_parser import (  # noqa: E402
    MOTIONSTREAMER272_DIM,
    MOTIONSTREAMER272_POSITION_SLICE,
    MOTIONSTREAMER272_ROTATION_SLICE,
    SourceParserError,
    decode_source_row_cont6d,
)


PARENTS = np.asarray(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
    dtype=np.int64,
)


def _row_cont6d(rotations: np.ndarray) -> np.ndarray:
    return np.asarray(rotations[..., :2, :], dtype=np.float64).reshape(
        *rotations.shape[:-2], 6
    )


def _z_rotation(angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _fixture(frames: int = 5) -> np.ndarray:
    data = np.zeros((frames, MOTIONSTREAMER272_DIM), dtype=np.float64)
    data[:, 0] = np.linspace(0.0, 0.2, frames)
    data[:, 1] = np.linspace(0.0, -0.1, frames)
    heading = np.stack([_z_rotation(0.01 * index) for index in range(frames)])
    data[:, 2:8] = _row_cont6d(heading)
    positions = np.zeros((frames, 22, 3), dtype=np.float64)
    positions[..., 1] = np.arange(22, dtype=np.float64)[None] * 0.01
    data[:, MOTIONSTREAMER272_POSITION_SLICE] = positions.reshape(frames, -1)
    local = np.broadcast_to(np.eye(3), (frames, 22, 3, 3)).copy()
    for frame in range(frames):
        local[frame, 3] = _z_rotation(0.05 * frame)
    data[:, MOTIONSTREAMER272_ROTATION_SLICE] = _row_cont6d(local).reshape(frames, -1)
    return data


def _task_and_pass_record() -> tuple[dict[str, object], dict[str, object]]:
    task: dict[str, object] = {
        "clip_id": "HML3D_Human_000000",
        "rig_id": "HML3D_Human",
        "topology_family": "human",
        "split": "train",
        "source_relpath": "000000.npy",
        "file_size_bytes": 4096,
        "mtime_ns": 123456789,
        "source_device": 7,
        "source_inode": 11,
        "source_nlink": 1,
        "T_src": 5,
        "fps_src": 30.0,
        "source_shape": [5, 272],
        "source_dtype": "float64",
        "rotation_slice": [140, 272],
        "rotation_shape": [22, 6],
    }
    metrics: dict[str, object] = {
        "independent_decoder": "numpy-independent-motionstreamer272-v1",
        "rotation_payload_sha256": "b" * 64,
        "raw_d6_first_row_unit_max_abs": 0.0,
        "raw_d6_second_row_unit_max_abs": 0.0,
        "raw_d6_row_dot_max_abs": 0.0,
        "raw_d6_cross_norm_min": 1.0,
        "independent_positions_max_abs": 0.0,
        "independent_root_translation_max_abs": 0.0,
        "independent_local_rotation_max_abs": 0.0,
        "independent_global_rotation_max_abs": 0.0,
        "source_parser_fk_max_norm": 0.0,
        "source_parser_fk_mpjpe_norm": 0.0,
        "fixed_neutral_rigid_edge_max_norm": 0.0,
        "rotation_orthogonality_max_abs": 0.0,
        "rotation_determinant_min": 1.0,
        "rotation_determinant_max": 1.0,
        "root_speed_rms_norm_per_s": 0.0,
        "rotation_speed_rms_rad_per_s": 0.0,
        "pose_excursion_rms_norm": 0.0,
        "dynamic_score": 0.0,
    }
    record: dict[str, object] = {
        "audit_version": HUMAN312_AUDIT_VERSION,
        "clip_id": task["clip_id"],
        "rig_id": "HML3D_Human",
        "source_family": "motionstreamer272",
        "topology_family": "human",
        "split": "train",
        "status": "pass",
        "reason_codes": [],
        "source_relpath": task["source_relpath"],
        "source_sha256": "a" * 64,
        "source_size_bytes": task["file_size_bytes"],
        "source_mtime_ns": task["mtime_ns"],
        "source_device": task["source_device"],
        "source_inode": task["source_inode"],
        "source_nlink": 1,
        "source_shape": task["source_shape"],
        "source_dtype": "float64",
        "rotation_slice": [140, 272],
        "rotation_shape": [22, 6],
        "T_src": 5,
        "J_phys": 22,
        "fps_src": 30.0,
        "metrics": metrics,
    }
    return task, record


def _write_fixture_task(root: Path) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    data = _fixture()
    offsets = np.zeros((22, 3), dtype=np.float64)
    offsets[1:, 1] = 0.1
    rest = np.zeros((22, 3), dtype=np.float64)
    for child in range(1, 22):
        rest[child] = rest[int(PARENTS[child])] + offsets[child]
    identity6 = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    data[:, 2:8] = identity6
    data[:, MOTIONSTREAMER272_POSITION_SLICE] = rest.reshape(1, -1)
    data[:, MOTIONSTREAMER272_ROTATION_SLICE] = np.tile(identity6, 22)
    source = root / "000000.npy"
    np.save(source, data)
    observed = source.stat()
    task: dict[str, object] = {
        "clip_id": "HML3D_Human_000000",
        "rig_id": "HML3D_Human",
        "topology_family": "human",
        "split": "train",
        "source_relpath": source.name,
        "file_size_bytes": int(observed.st_size),
        "mtime_ns": int(observed.st_mtime_ns),
        "source_device": int(observed.st_dev),
        "source_inode": int(observed.st_ino),
        "source_nlink": int(observed.st_nlink),
        "T_src": int(data.shape[0]),
        "fps_src": 30.0,
        "source_shape": list(data.shape),
        "source_dtype": "float64",
        "rotation_slice": [140, 272],
        "rotation_shape": [22, 6],
    }
    return task, rest, offsets


class Human312AuditTests(unittest.TestCase):
    def test_transient_enoent_retry_is_bounded_and_specific(self) -> None:
        attempts = 0

        def eventually_visible() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise FileNotFoundError("injected transient visibility failure")
            return "visible"

        with mock.patch.object(human_audit.time, "sleep") as sleep:
            self.assertEqual(
                human_audit._retry_transient_enoent(
                    eventually_visible, label="fixture visibility"
                ),
                "visible",
            )
        self.assertEqual(attempts, 3)
        self.assertEqual(sleep.call_count, 2)

        permanent_attempts = 0

        def permanently_absent() -> None:
            nonlocal permanent_attempts
            permanent_attempts += 1
            raise FileNotFoundError("injected permanent absence")

        with mock.patch.object(human_audit.time, "sleep"):
            with self.assertRaisesRegex(FileNotFoundError, "permanent absence"):
                human_audit._retry_transient_enoent(
                    permanently_absent, label="fixture absence"
                )
        self.assertEqual(
            permanent_attempts,
            len(human_audit.TRANSIENT_ENOENT_RETRY_DELAYS_SECONDS),
        )

        other_attempts = 0

        def other_io_failure() -> None:
            nonlocal other_attempts
            other_attempts += 1
            raise PermissionError("injected permission failure")

        with self.assertRaisesRegex(PermissionError, "permission failure"):
            human_audit._retry_transient_enoent(
                other_io_failure, label="fixture permission"
            )
        self.assertEqual(other_attempts, 1)

    def test_atomic_write_and_json_read_retry_transient_enoent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.json"
            real_fsync = os.fsync
            fsync_calls = 0

            def transient_fsync(descriptor: int) -> None:
                nonlocal fsync_calls
                fsync_calls += 1
                if fsync_calls == 1:
                    raise FileNotFoundError("injected transient fsync failure")
                real_fsync(descriptor)

            with (
                mock.patch.object(human_audit.os, "fsync", side_effect=transient_fsync),
                mock.patch.object(human_audit.time, "sleep"),
            ):
                _write_json(path, {"status": "pass"})
            self.assertGreaterEqual(fsync_calls, 3)

            real_read_text = Path.read_text
            read_calls = 0

            def transient_read_text(observed: Path, *args: object, **kwargs: object) -> str:
                nonlocal read_calls
                read_calls += 1
                if observed == path and read_calls == 1:
                    raise FileNotFoundError("injected transient read failure")
                return real_read_text(observed, *args, **kwargs)

            with (
                mock.patch.object(Path, "read_text", new=transient_read_text),
                mock.patch.object(human_audit.time, "sleep"),
            ):
                self.assertEqual(human_audit._load_json(path), {"status": "pass"})
            self.assertEqual(read_calls, 2)

    def test_fixed_neutral_primary_parser_matches_independent_decoder(self) -> None:
        data = _fixture()
        offsets = np.zeros((22, 3), dtype=np.float64)
        offsets[1:, 1] = 0.1
        rest = np.zeros((22, 3), dtype=np.float64)
        for child in range(1, 22):
            rest[child] = rest[int(PARENTS[child])] + offsets[child]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.npy"
            np.save(path, data)
            primary = parse_motionstreamer272_fixed_neutral(
                path,
                joint_names=[f"joint_{index}" for index in range(22)],
                parents=PARENTS,
                P_rest_global=rest,
                offset_parent_local=offsets,
                rest_authority="fixture",
            )
        independent = independent_motionstreamer272_decode(data, PARENTS)
        self.assertEqual(
            primary.diagnostics["parser_version"], FIXED_NEUTRAL_PARSER_VERSION
        )
        self.assertLess(
            float(np.max(np.abs(primary.source_positions - independent["positions"]))),
            1e-14,
        )
        self.assertLess(
            float(
                np.max(
                    np.abs(
                        primary.local_rotations - independent["local_rotations"]
                    )
                )
            ),
            1e-14,
        )

    def test_array_parser_uses_the_supplied_stable_snapshot(self) -> None:
        data = _fixture()
        offsets = np.zeros((22, 3), dtype=np.float64)
        offsets[1:, 1] = 0.1
        rest = np.zeros((22, 3), dtype=np.float64)
        for child in range(1, 22):
            rest[child] = rest[int(PARENTS[child])] + offsets[child]
        parsed = parse_motionstreamer272_fixed_neutral_array(
            data,
            source_identity="content-sha256:fixture",
            joint_names=[f"joint_{index}" for index in range(22)],
            parents=PARENTS,
            P_rest_global=rest,
            offset_parent_local=offsets,
            rest_authority="fixture",
        )
        self.assertEqual(parsed.path, "content-sha256:fixture")
        self.assertEqual(parsed.global_rotations.shape, (5, 22, 3, 3))

    def test_independent_decoder_matches_source_row_cont6d_contract(self) -> None:
        data = _fixture()
        decoded = independent_motionstreamer272_decode(data, PARENTS)
        source_local = decode_source_row_cont6d(
            data[:, MOTIONSTREAMER272_ROTATION_SLICE].reshape(5, 22, 6)
        )
        self.assertEqual(decoded["positions"].shape, (5, 22, 3))
        self.assertEqual(decoded["global_rotations"].shape, (5, 22, 3, 3))
        self.assertLess(
            float(np.max(np.abs(decoded["local_rotations"][:, 1:] - source_local[:, 1:]))),
            1e-15,
        )
        gram = decoded["global_rotations"] @ np.swapaxes(
            decoded["global_rotations"], -1, -2
        )
        self.assertLess(float(np.max(np.abs(gram - np.eye(3)))), 1e-12)

    def test_independent_decoder_rejects_degenerate_rows(self) -> None:
        data = _fixture()
        data[0, 2:5] = 0.0
        with self.assertRaisesRegex(Human312AuditError, "degenerate"):
            independent_motionstreamer272_decode(data, PARENTS)

    def test_rigid_edge_and_dynamic_metrics(self) -> None:
        frames = 4
        offsets = np.zeros((22, 3), dtype=np.float64)
        offsets[1:, 1] = 0.1
        positions = np.zeros((frames, 22, 3), dtype=np.float64)
        for child in range(1, 22):
            positions[:, child] = positions[:, int(PARENTS[child])] + offsets[child]
        rotations = np.broadcast_to(np.eye(3), (frames, 22, 3, 3)).copy()
        self.assertLess(
            _rigid_edge_max_norm(positions, PARENTS, offsets, 2.0), 1e-15
        )
        metrics = _dynamic_metrics(positions, rotations, fps=30.0, s_rig=2.0)
        self.assertEqual(metrics["dynamic_score"], 0.0)

    def test_representative_is_dynamic_and_prefers_long_visual_pool(self) -> None:
        records = [
            {
                "clip_id": "short",
                "status": "pass",
                "T_src": 20,
                "source_relpath": "short.npy",
                "source_sha256": "a" * 64,
                "metrics": {"dynamic_score": 100.0},
            },
            {
                "clip_id": "long",
                "status": "pass",
                "T_src": 80,
                "source_relpath": "long.npy",
                "source_sha256": "b" * 64,
                "metrics": {"dynamic_score": 3.0},
            },
        ]
        selected = _select_representative(records)
        self.assertEqual(selected["clip_id"], "long")

    def test_chunk_cache_is_bound_to_authority_tasks_and_records(self) -> None:
        task, record = _task_and_pass_record()
        status = {
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "threads": 1,
            "vmrss_kib": 1,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "chunk.json"
            payload = {
                "audit_version": HUMAN312_AUDIT_VERSION,
                "authority_sha256": "c" * 64,
                "chunk_index": 0,
                "task_scope_sha256": _task_scope_sha256([task]),
                "records_sha256": _records_sha256([record]),
                "records": [record],
                "worker_process_status": status,
            }
            _write_json(path, payload)
            self.assertIsNotNone(
                _load_valid_chunk(
                    path,
                    authority_sha256="c" * 64,
                    chunk_index=0,
                    tasks=[task],
                )
            )
            payload["records"][0]["status"] = "reject"
            _write_json(path, payload)
            self.assertIsNone(
                _load_valid_chunk(
                    path,
                    authority_sha256="c" * 64,
                    chunk_index=0,
                    tasks=[task],
                )
            )

    def test_record_difference_is_exact(self) -> None:
        self.assertEqual(_first_record_difference({"a": [1]}, {"a": [1]}), "")
        self.assertIn("$.a[0]", _first_record_difference({"a": [1]}, {"a": [2]}))

    def test_relative_symlink_activation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "generations" / "g1"
            target.mkdir(parents=True)
            link = root / "active"
            _replace_symlink(link, target)
            self.assertTrue(link.is_symlink())
            self.assertFalse(Path(os.readlink(link)).is_absolute())
            self.assertEqual(link.resolve(), target.resolve())

    def test_absolute_or_traversing_authority_link_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.json"
            target.write_text("{}", encoding="utf-8")
            absolute = root / "absolute"
            os.symlink(str(target), absolute)
            with self.assertRaisesRegex(Human312AuditError, "canonical relative"):
                _read_relative_symlink_target(absolute, label="fixture")
            traversing = root / "traversing"
            child = root / "child"
            child.mkdir()
            os.symlink("../target.json", traversing)
            with self.assertRaisesRegex(Human312AuditError, "canonical relative"):
                _read_relative_symlink_target(traversing, label="fixture")

    def test_generation_manifest_rejects_special_symlink_and_hardlink_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            regular = root / "regular.json"
            regular.write_text("{}", encoding="utf-8")
            fifo = root / "fifo"
            os.mkfifo(fifo)
            with self.assertRaisesRegex(Human312AuditError, "special entry"):
                _file_manifest(root)
            fifo.unlink()
            linked = root / "linked.json"
            os.symlink("regular.json", linked)
            with self.assertRaisesRegex(Human312AuditError, "symlink"):
                _file_manifest(root)
            linked.unlink()
            hardlink = root / "hardlink.json"
            os.link(regular, hardlink)
            with self.assertRaisesRegex(Human312AuditError, "hard-linked"):
                _file_manifest(root)

    def test_read_only_generation_validation_rejects_writable_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            child = root / "child"
            child.mkdir()
            (child / "value.json").write_text("{}", encoding="utf-8")
            os.chmod(root, 0o555)
            try:
                with self.assertRaisesRegex(Human312AuditError, "directory is writable"):
                    _file_manifest(root, require_read_only=True)
            finally:
                os.chmod(root, 0o755)

    def test_zero_record_or_source_mutation_cannot_form_an_approval_scope(self) -> None:
        task, record = _task_and_pass_record()
        with self.assertRaisesRegex(Human312AuditError, "not exhaustive"):
            _validate_records_against_tasks([], [task], expected_count=1)
        changed = dict(record)
        changed["status"] = "reject"
        changed["reason_codes"] = ["HUMAN_SOURCE_CHANGED_DURING_AUDIT"]
        changed.pop("metrics")
        changed["error_type"] = "HumanClipReject"
        changed["error"] = "changed"
        with self.assertRaisesRegex(Human312AuditError, "cannot filter"):
            _reject_unstable_source_records([changed])
        unhashed = dict(record)
        unhashed["source_sha256"] = None
        with self.assertRaisesRegex(Human312AuditError, "cannot filter"):
            _reject_unstable_source_records([unhashed])

    def test_unexpected_decoder_defect_aborts_instead_of_becoming_reject(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task, rest, offsets = _write_fixture_task(root)
            human_audit._WORKER_RIG = {
                "joint_map": {
                    "btjd_parents": PARENTS.tolist(),
                    "btjd_joint_names": [f"joint_{index}" for index in range(22)],
                }
            }
            human_audit._WORKER_FIXED = {
                "P_rest_global": rest,
                "offsets": offsets,
                "s_rig": float(np.linalg.norm(np.ptp(rest, axis=0))),
            }
            human_audit._WORKER_SOURCE_ROOT = root
            with mock.patch.object(
                human_audit,
                "parse_motionstreamer272_fixed_neutral_array",
                side_effect=RuntimeError("programmer defect"),
            ):
                with self.assertRaisesRegex(RuntimeError, "programmer defect"):
                    human_audit._audit_one(task)

            with mock.patch.object(
                human_audit,
                "independent_motionstreamer272_decode",
                side_effect=Human312AuditError("independent implementation defect"),
            ):
                with self.assertRaisesRegex(
                    Human312AuditError, "independent implementation defect"
                ):
                    human_audit._audit_one(task)

    def test_typed_source_parser_error_remains_filterable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task, rest, offsets = _write_fixture_task(root)
            human_audit._WORKER_RIG = {
                "joint_map": {
                    "btjd_parents": PARENTS.tolist(),
                    "btjd_joint_names": [f"joint_{index}" for index in range(22)],
                }
            }
            human_audit._WORKER_FIXED = {
                "P_rest_global": rest,
                "offsets": offsets,
                "s_rig": float(np.linalg.norm(np.ptp(rest, axis=0))),
            }
            human_audit._WORKER_SOURCE_ROOT = root
            with mock.patch.object(
                human_audit,
                "parse_motionstreamer272_fixed_neutral_array",
                side_effect=MotionStreamer272ContentError("recognized source defect"),
            ):
                record = human_audit._audit_one(task)
            self.assertEqual(record["status"], "reject")
            self.assertEqual(record["reason_codes"], ["HUMAN_SOURCE_PARSE_FAILURE"])
            self.assertRegex(str(record["source_sha256"]), r"^[0-9a-f]{64}$")
            with mock.patch.object(
                human_audit,
                "parse_motionstreamer272_fixed_neutral_array",
                side_effect=SourceParserError("parser contract defect"),
            ):
                with self.assertRaisesRegex(SourceParserError, "parser contract defect"):
                    human_audit._audit_one(task)

    def test_approval_partial_finalization_quarantines_pass_named_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            approval_root = output / human_audit.HUMAN_AUDIT_APPROVAL_DIRECTORY
            approval_root.mkdir()
            generation = (
                output
                / human_audit.HUMAN_AUDIT_GENERATION_DIRECTORY
                / "fixture-generation"
            )
            generation.mkdir(parents=True)
            authority = {
                "authority_sha256": "a" * 64,
                "code_closure": {},
                "runtime_fingerprint": {},
                "parent_manifest": {},
                "pinned_inputs": {},
                "chunk_size": 32,
            }
            summary = {
                "source_snapshot_sha256": "b" * 64,
                "accepted_clip_count": 26846,
                "rejected_clip_count": 0,
            }
            _write_json(generation / "authority.json", authority)
            _write_json(generation / "summary.json", summary)
            content = {
                "generation_content_sha256": "c" * 64,
                "generation_json_sha256": "d" * 64,
            }
            recheck = {
                "status": "pass",
                "validated_count": 26846,
                "hash_workers": 1,
                "disk_inventory_snapshot_sha256": "e" * 64,
                "source_snapshot_sha256": "b" * 64,
                "completed_at_utc": "now",
            }
            status = {
                "pid": os.getpid() + 1_000_000,
                "ppid": os.getpid(),
                "threads": 1,
                "vmrss_kib": 1,
            }
            chunk_count = (26846 + 31) // 32
            candidate = {
                "content_evidence": content,
                "deep_chunk_process_evidence": {
                    "executor_mode": "spawn",
                    "chunk_count": chunk_count,
                    "cached_revalidated_chunk_count": 0,
                    "fresh_spawn_chunk_count": chunk_count,
                    "fresh_spawn_chunks_with_process_status": chunk_count,
                    "cached_worker_process_status_trusted": False,
                },
            }
            real_fsync = human_audit._fsync_directory
            calls = 0

            def fail_second(path: Path) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected approval-root fsync failure")
                real_fsync(path)

            with mock.patch.object(
                human_audit, "_fsync_directory", side_effect=fail_second
            ):
                witness = human_audit._ApprovalCleanupWitness()
                with self.assertRaisesRegex(
                    Human312AuditError, "materialized approval was quarantined"
                ):
                    human_audit._create_approval(
                        output_root=output,
                        generation_root=generation,
                        candidate_proof=candidate,
                        post_deep_live_recheck=recheck,
                        deep_records_sha256="f" * 64,
                        deep_worker_statuses=[status],
                        cleanup_witness=witness,
                    )
            approval = human_audit._approval_path(output, "c" * 64)
            self.assertEqual(witness.path, approval)
            self.assertTrue(witness.owned_by_run)
            self.assertFalse(os.path.lexists(approval))
            self.assertEqual(
                len(list(approval_root.glob(".rejected-incomplete-*.json"))), 1
            )

            # Discard the successful return value to model an asynchronous
            # exception before the caller can unpack it.  The mutable witness
            # must already identify the newly owned PASS path.
            successful_witness = human_audit._ApprovalCleanupWitness()
            human_audit._create_approval(
                output_root=output,
                generation_root=generation,
                candidate_proof=candidate,
                post_deep_live_recheck=recheck,
                deep_records_sha256="f" * 64,
                deep_worker_statuses=[status],
                cleanup_witness=successful_witness,
            )
            self.assertEqual(successful_witness.path, approval)
            self.assertTrue(successful_witness.owned_by_run)
            self.assertTrue(approval.is_file())
            self.assertEqual(approval.stat().st_mode & 0o222, 0)

    def test_work_and_chunk_namespaces_reject_symlink_redirection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = root / "outside"
            outside.mkdir()
            linked = root / "linked"
            os.symlink("outside", linked)
            with self.assertRaisesRegex(Human312AuditError, "canonical directory"):
                _ensure_canonical_directory(linked, label="fixture")
            work = root / "work"
            work.mkdir()
            os.symlink("../outside", work / "deep_chunks")
            with self.assertRaisesRegex(Human312AuditError, "canonical directory"):
                human_audit._run_chunks(
                    tasks=[],
                    rig={},
                    fixed={},
                    source_root=outside,
                    work_root=work,
                    authority_sha256="a" * 64,
                    workers=1,
                    chunk_size=1,
                    resumable=False,
                )

    def test_resumable_process_evidence_distinguishes_fresh_from_cached(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "source"
            work_root = root / "work"
            source_root.mkdir()
            work_root.mkdir()
            task, rest, offsets = _write_fixture_task(source_root)
            rig = {
                "joint_map": {
                    "btjd_parents": PARENTS.tolist(),
                    "btjd_joint_names": [f"joint_{index}" for index in range(22)],
                }
            }
            fixed = {
                "parents": PARENTS,
                "P_rest_global": rest,
                "offsets": offsets,
                "s_rig": float(np.linalg.norm(np.ptp(rest, axis=0))),
            }
            first_records, first_statuses, first_execution = human_audit._run_chunks(
                tasks=[task],
                rig=rig,
                fixed=fixed,
                source_root=source_root,
                work_root=work_root,
                authority_sha256="a" * 64,
                workers=1,
                chunk_size=1,
                resumable=True,
            )
            self.assertEqual(first_records[0]["status"], "pass")
            self.assertTrue(first_statuses)
            self.assertEqual(first_execution["fresh_spawn_chunk_count"], 1)
            self.assertEqual(first_execution["cached_revalidated_chunk_count"], 0)
            self.assertFalse(first_execution["cached_worker_process_status_trusted"])
            cached_path = work_root / "chunks/chunk_000000.json"
            cached_payload = human_audit._load_json(cached_path)
            cached_payload["worker_process_status"]["pid"] = os.getpid() + 2_000_000
            _write_json(cached_path, cached_payload)
            _, second_statuses, second_execution = human_audit._run_chunks(
                tasks=[task],
                rig=rig,
                fixed=fixed,
                source_root=source_root,
                work_root=work_root,
                authority_sha256="a" * 64,
                workers=1,
                chunk_size=1,
                resumable=True,
            )
            self.assertEqual(second_statuses, [])
            self.assertEqual(second_execution["fresh_spawn_chunk_count"], 0)
            self.assertEqual(second_execution["cached_revalidated_chunk_count"], 1)
            self.assertFalse(second_execution["cached_worker_process_status_trusted"])

    def test_approval_validation_invokes_full_generation_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            generation = (
                output
                / human_audit.HUMAN_AUDIT_GENERATION_DIRECTORY
                / "fixture-generation"
            )
            generation.mkdir(parents=True)
            sentinel = Human312AuditError("semantic sentinel")
            with mock.patch.object(
                human_audit, "_validate_generation_structure", side_effect=sentinel
            ) as semantic:
                with self.assertRaisesRegex(Human312AuditError, "semantic sentinel"):
                    human_audit._validate_approval(generation, output)
            semantic.assert_called_once_with(generation)

    def test_active_validator_derives_generation_from_atomic_approval_link(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            approval_root = output / human_audit.HUMAN_AUDIT_APPROVAL_DIRECTORY
            generation_root = output / human_audit.HUMAN_AUDIT_GENERATION_DIRECTORY
            approval_root.mkdir()
            generation_root.mkdir()
            new_generation = generation_root / "new"
            old_generation = generation_root / "old"
            new_generation.mkdir()
            old_generation.mkdir()
            approval = approval_root / "approval.json"
            _write_json(
                approval,
                {
                    "generation_relpath": new_generation.relative_to(output).as_posix()
                },
            )
            _replace_symlink(
                output / human_audit.HUMAN_AUDIT_APPROVAL_LINK_NAME, approval
            )
            _replace_symlink(
                output / human_audit.HUMAN_AUDIT_LINK_NAME, old_generation
            )
            with mock.patch.object(
                human_audit,
                "_validate_approval",
                return_value=(approval, {"generation_relpath": "unused"}),
            ):
                with self.assertRaisesRegex(Human312AuditError, "mixed"):
                    human_audit.validate_active_human_audit(
                        output, rehash_sources=False
                    )

    def test_activation_failure_rolls_back_both_links(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            old_generation = output / "old-generation"
            new_generation = output / "new-generation"
            old_approval = output / "old-approval.json"
            new_approval = output / "new-approval.json"
            old_generation.mkdir()
            new_generation.mkdir()
            old_approval.write_text("{}", encoding="utf-8")
            new_approval.write_text("{}", encoding="utf-8")
            generation_link = output / human_audit.HUMAN_AUDIT_LINK_NAME
            approval_link = output / human_audit.HUMAN_AUDIT_APPROVAL_LINK_NAME
            _replace_symlink(generation_link, old_generation)
            _replace_symlink(approval_link, old_approval)
            previous_generation = os.readlink(generation_link)
            previous_approval = os.readlink(approval_link)
            with (
                mock.patch.object(
                    human_audit,
                    "_validate_approval",
                    return_value=(new_approval, {}),
                ),
                mock.patch.object(
                    human_audit, "_verify_approved_live_sources", return_value={}
                ),
                mock.patch.object(
                    human_audit,
                    "validate_active_human_audit",
                    side_effect=Human312AuditError("post-link failure"),
                ),
            ):
                with self.assertRaisesRegex(Human312AuditError, "post-link failure"):
                    _activate_approved_generation(
                        output_root=output,
                        generation_root=new_generation,
                        approval_path=new_approval,
                    )
            self.assertEqual(os.readlink(generation_link), previous_generation)
            self.assertEqual(os.readlink(approval_link), previous_approval)

    def test_committed_active_binding_is_detectable_before_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            generation = output / "generation"
            approval = output / "approval.json"
            generation.mkdir()
            approval.write_text("{}", encoding="utf-8")
            _replace_symlink(
                output / human_audit.HUMAN_AUDIT_LINK_NAME, generation
            )
            _replace_symlink(
                output / human_audit.HUMAN_AUDIT_APPROVAL_LINK_NAME, approval
            )
            self.assertTrue(
                human_audit._active_binding_points_to(
                    output_root=output,
                    generation_root=generation,
                    approval_path=approval,
                )
            )
            self.assertTrue(
                human_audit._active_links_reference_candidate(
                    output_root=output,
                    generation_root=generation,
                    approval_path=approval,
                )
            )

    def test_cleanup_cannot_quarantine_committed_active_objects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            generations = output / human_audit.HUMAN_AUDIT_GENERATION_DIRECTORY
            approvals = output / human_audit.HUMAN_AUDIT_APPROVAL_DIRECTORY
            work = output / "work"
            generations.mkdir()
            approvals.mkdir()
            work.mkdir()
            generation = generations / "committed-generation"
            generation.mkdir()
            approval = approvals / ("a" * 64 + ".json")
            approval.write_text("{}", encoding="utf-8")
            os.chmod(approval, 0o444)
            _replace_symlink(
                output / human_audit.HUMAN_AUDIT_LINK_NAME, generation
            )
            _replace_symlink(
                output / human_audit.HUMAN_AUDIT_APPROVAL_LINK_NAME, approval
            )
            with self.assertRaisesRegex(Human312AuditError, "actively referenced"):
                human_audit._quarantine_failed_candidate(
                    generation_root=generation,
                    generations_root=generations,
                    approval_path=approval,
                    work_root=work,
                    error=KeyboardInterrupt("post-commit interrupt"),
                )
            with self.assertRaisesRegex(Human312AuditError, "actively referenced"):
                human_audit._quarantine_failed_approval(
                    approval_path=approval,
                    output_root=output,
                    work_root=work,
                    error=KeyboardInterrupt("post-commit interrupt"),
                )
            self.assertTrue(generation.is_dir())
            self.assertTrue(approval.is_file())
            self.assertEqual(
                (output / human_audit.HUMAN_AUDIT_LINK_NAME).resolve(), generation
            )
            self.assertEqual(
                (output / human_audit.HUMAN_AUDIT_APPROVAL_LINK_NAME).resolve(),
                approval,
            )

    def test_candidate_quarantine_refuses_approval_only_active_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            generations = output / human_audit.HUMAN_AUDIT_GENERATION_DIRECTORY
            approvals = output / human_audit.HUMAN_AUDIT_APPROVAL_DIRECTORY
            work = output / "work"
            generations.mkdir()
            approvals.mkdir()
            work.mkdir()
            generation = generations / "committed-generation"
            generation.mkdir()
            approval = approvals / ("a" * 64 + ".json")
            _write_json(
                approval,
                {"generation_relpath": generation.relative_to(output).as_posix()},
            )
            os.chmod(approval, 0o444)
            _replace_symlink(
                output / human_audit.HUMAN_AUDIT_APPROVAL_LINK_NAME, approval
            )
            with self.assertRaisesRegex(Human312AuditError, "active approval"):
                human_audit._quarantine_failed_candidate(
                    generation_root=generation,
                    generations_root=generations,
                    approval_path=approval,
                    work_root=work,
                    error=KeyboardInterrupt("approval-only commit witness"),
                )
            self.assertTrue(generation.is_dir())
            self.assertEqual(
                (output / human_audit.HUMAN_AUDIT_APPROVAL_LINK_NAME).resolve(),
                approval,
            )

    def test_failed_new_approval_is_quarantined_not_left_pass_named(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            approval_root = output / human_audit.HUMAN_AUDIT_APPROVAL_DIRECTORY
            work = output / "work"
            approval_root.mkdir()
            work.mkdir()
            approval = approval_root / ("a" * 64 + ".json")
            approval.write_text("{}", encoding="utf-8")
            os.chmod(approval, 0o444)
            rejected = _quarantine_failed_approval(
                approval_path=approval,
                output_root=output,
                work_root=work,
                error=Human312AuditError("activation failed"),
            )
            self.assertFalse(approval.exists())
            self.assertTrue(rejected.is_file())
            self.assertTrue(rejected.name.startswith(".rejected-"))

    def test_live_source_verifier_hashes_one_stable_descriptor_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "000000.npy"
            source.write_bytes(b"fixture")
            observed = source.stat()
            digest = "d" * 64
            record = {
                "clip_id": "HML3D_Human_000000",
                "source_relpath": source.name,
                "source_sha256": digest,
                "source_size_bytes": int(observed.st_size),
                "source_mtime_ns": int(observed.st_mtime_ns),
                "source_device": int(observed.st_dev),
                "source_inode": int(observed.st_ino),
                "source_nlink": int(observed.st_nlink),
            }
            evidence = {
                "size_bytes": int(observed.st_size),
                "mtime_ns": int(observed.st_mtime_ns),
                "device": int(observed.st_dev),
                "inode": int(observed.st_ino),
                "nlink": int(observed.st_nlink),
            }
            with mock.patch.object(
                human_audit,
                "_read_stable_source_bytes",
                return_value=(b"fixture", digest, evidence),
            ) as stable:
                _verify_live_source_record(record, root)
            stable.assert_called_once_with(source)


if __name__ == "__main__":
    unittest.main()
