"""Exhaustive MotionStreamer272 source audit for the Human member of PZ+Human-312.

This module is intentionally separate from :mod:`pz312_audit`: the two source
families have different binary formats and different trusted rotation
authorities.  It audits every manifest-scoped Human clip, records explicit
per-clip rejection reasons, selects one dynamic representative, freezes an
immutable generation, deep-replays the complete audit, and only then creates a
content-addressed prototype-conversion approval.

The audit never derives rotations from positions.  Its two rotation paths are:

* the production MotionStreamer272 parser; and
* an independent NumPy implementation of the documented row-cont6d and
  heading-recovery equations.

SMPL shape coefficients are absent from MotionStreamer272.  The authoritative
KTJD geometry is therefore the separately reviewed, hash-pinned fixed-neutral
Human rig.  Raw shaped positions remain diagnostic only.
"""

from __future__ import annotations

import ast
import concurrent.futures
import contextlib
import dataclasses
import datetime as _datetime
import hashlib
import io
import importlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from collections import Counter, deque
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import threadpoolctl

from .encoder import load_skeleton
from .human312_fixed_rig import (
    HUMAN_CONTRACT_VERSION,
    HUMAN_RIG_ID,
    HumanFixedRig,
    build_current_btjd_human_fixed_rig,
)
from .human_source_parser import (
    MotionStreamer272ContentError,
    parse_motionstreamer272_fixed_neutral_array,
)
from .inventory_validation import _validate_transaction
from .source_parser import (
    MOTIONSTREAMER272_DIM,
    MOTIONSTREAMER272_JOINTS,
    MOTIONSTREAMER272_POSITION_SLICE,
    MOTIONSTREAMER272_ROTATION_SLICE,
    SOURCE_D6_DEGENERACY_EPS,
    rotation_matrix_diagnostics,
    source_fk_metrics,
)


HUMAN312_AUDIT_VERSION = "ktjd17-human1-exhaustive-source-audit-v3"
HUMAN312_APPROVAL_VERSION = "ktjd17-human1-source-audit-approval-v2"
HUMAN312_INDEPENDENT_DECODER = "numpy-independent-motionstreamer272-v1"
HUMAN_AUDIT_GENERATION_DIRECTORY = ".ktjd17_human_source_audit_generations"
HUMAN_AUDIT_APPROVAL_DIRECTORY = ".ktjd17_human_source_audit_approvals"
HUMAN_AUDIT_WORK_DIRECTORY = ".ktjd17_human_source_audit_work"
HUMAN_AUDIT_LINK_NAME = "ktjd17_human_source_audit"
HUMAN_AUDIT_APPROVAL_LINK_NAME = "ktjd17_human_source_audit_approval"
EXPECTED_HUMAN_RIG_COUNT = 1
EXPECTED_HUMAN_CLIP_COUNT = 26_846
EXPECTED_HUMAN_JOINT_COUNT = 22
EXPECTED_HUMAN_SOURCE_DIM = 272
EXPECTED_HUMAN_FPS = 30.0
SOURCE_FK_MAX_NORM = 1.0e-5
DUAL_POSITION_MAX_ABS = 1.0e-10
DUAL_ROTATION_MAX_ABS = 1.0e-12
ROTATION_ORTHOGONALITY_MAX_ABS = 1.0e-10
ROTATION_DETERMINANT_MAX_ABS = 1.0e-10
FIXED_RIGID_EDGE_MAX_NORM = 1.0e-10
RAW_D6_UNIT_NORM_MAX_ABS = 1.0e-5
RAW_D6_ROW_DOT_MAX_ABS = 1.0e-5
RAW_D6_CROSS_NORM_MIN = 1.0 - 1.0e-5
CANDIDATE_STATUS = "pending_post_publish_deep_validation"
HUMAN_CLAIM_BOUNDARY = "current_btjd_fixed_neutral_human_not_subject_shaped_amass"
HUMAN_ROTATION_AUTHORITY = (
    "motionstreamer272_real_local_rotation_channels_140_272"
)
HUMAN_PRODUCTION_DECODER = (
    "src.data.ktjd17.human_source_parser."
    "parse_motionstreamer272_fixed_neutral_array"
)
HUMAN_ANOMALY_POLICY = (
    "retain the one-rig/26846-clip source scope; reject only stable, "
    "content-hashed per-clip numeric, schema, rotation, or FK anomalies; "
    "source mutation or missing content identity aborts the audit; never "
    "silently drop the Human rig"
)
TRANSIENT_ENOENT_RETRY_DELAYS_SECONDS = (
    0.0,
    0.05,
    0.1,
    0.2,
    0.4,
    0.8,
    1.6,
    3.2,
)
EXPECTED_PARENT_MANIFEST_FILES = frozenset(
    {
        "canonical_skeleton_generation.json",
        "canonical_skeleton_qa.jsonl",
        "canonical_skeleton_summary.json",
        "clips.jsonl",
        "inventory_reason_codes.json",
        "inventory_summary.json",
        "prototype_candidates.json",
        "prototype_gaps.jsonl",
        "rigs.jsonl",
        "source_fk_generation.json",
        "source_fk_qa.jsonl",
        "source_fk_summary.json",
    }
)
PASS_RECORD_KEYS = frozenset(
    {
        "audit_version",
        "clip_id",
        "rig_id",
        "source_family",
        "topology_family",
        "split",
        "status",
        "reason_codes",
        "source_relpath",
        "source_sha256",
        "source_size_bytes",
        "source_mtime_ns",
        "source_device",
        "source_inode",
        "source_nlink",
        "source_shape",
        "source_dtype",
        "rotation_slice",
        "rotation_shape",
        "T_src",
        "J_phys",
        "fps_src",
        "metrics",
    }
)
REJECT_RECORD_KEYS = frozenset((PASS_RECORD_KEYS - {"metrics"}) | {"error_type", "error"})
PASS_METRIC_KEYS = frozenset(
    {
        "independent_decoder",
        "rotation_payload_sha256",
        "raw_d6_first_row_unit_max_abs",
        "raw_d6_second_row_unit_max_abs",
        "raw_d6_row_dot_max_abs",
        "raw_d6_cross_norm_min",
        "independent_positions_max_abs",
        "independent_root_translation_max_abs",
        "independent_local_rotation_max_abs",
        "independent_global_rotation_max_abs",
        "source_parser_fk_max_norm",
        "source_parser_fk_mpjpe_norm",
        "fixed_neutral_rigid_edge_max_norm",
        "rotation_orthogonality_max_abs",
        "rotation_determinant_min",
        "rotation_determinant_max",
        "root_speed_rms_norm_per_s",
        "rotation_speed_rms_rad_per_s",
        "pose_excursion_rms_norm",
        "dynamic_score",
    }
)
REJECTION_REASON_CODES = frozenset(
    {
        "HUMAN_SOURCE_NPY_LOAD_FAILURE",
        "HUMAN_SOURCE_SCHEMA_INVALID",
        "HUMAN_SOURCE_NONFINITE",
        "HUMAN_SOURCE_D6_DEGENERATE",
        "HUMAN_RAW_D6_NOT_ROTATION_LIKE",
        "HUMAN_INDEPENDENT_DECODER_MISMATCH",
        "HUMAN_SOURCE_FK_FAILURE",
        "HUMAN_FIXED_NEUTRAL_FK_FAILURE",
        "HUMAN_ROTATION_INVALID",
        "HUMAN_SOURCE_PARSE_FAILURE",
        "HUMAN_SOURCE_CHANGED_DURING_AUDIT",
    }
)

_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}

_WORKER_RIG: dict[str, Any] | None = None
_WORKER_FIXED: dict[str, Any] | None = None
_WORKER_SOURCE_ROOT: Path | None = None
_WORKER_THREAD_LIMITER: Any = None


class Human312AuditError(RuntimeError):
    """The exhaustive Human source authority failed closed."""


class HumanSourceContentError(Human312AuditError):
    """A recognized numeric/schema defect in one stable source payload."""


class HumanClipReject(RuntimeError):
    """Typed, stable anomaly classification for one content-hashed source clip."""

    def __init__(self, reason_code: str, message: str):
        if reason_code not in REJECTION_REASON_CODES:
            raise Human312AuditError(f"unknown Human rejection reason: {reason_code}")
        super().__init__(message)
        self.reason_code = reason_code


def _absolute_no_resolve(path: str | Path) -> Path:
    """Normalize dot components without hiding a symlink from later lstat checks."""
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


@dataclasses.dataclass(frozen=True)
class HumanAuditConfig:
    manifest_root: Path
    source_root: Path
    output_root: Path
    active_cond_path: Path
    legacy_truebones_cond_path: Path
    t04_candidate_path: Path
    workers: int = 24
    chunk_size: int = 32
    update_link: bool = True

    def resolved(self) -> "HumanAuditConfig":
        return dataclasses.replace(
            self,
            manifest_root=_absolute_no_resolve(self.manifest_root),
            source_root=_absolute_no_resolve(self.source_root),
            output_root=_absolute_no_resolve(self.output_root),
            active_cond_path=_absolute_no_resolve(self.active_cond_path),
            legacy_truebones_cond_path=_absolute_no_resolve(
                self.legacy_truebones_cond_path
            ),
            t04_candidate_path=_absolute_no_resolve(self.t04_candidate_path),
        )


@dataclasses.dataclass
class _ApprovalCleanupWitness:
    """Caller-visible ownership state for asynchronous approval cleanup."""

    path: Path | None = None
    owned_by_run: bool = False


def default_human_audit_config(repo_root: str | Path = ".") -> HumanAuditConfig:
    root = Path(repo_root).expanduser().resolve()
    manifest = (
        root
        / "dataset/.ktjd17_manifest_generations/"
        "20260819T145535975831Z-ed48b3fd2745"
    )
    return HumanAuditConfig(
        manifest_root=manifest,
        source_root=root / "scratch/humanml3d_272/motion_data",
        output_root=root / "dataset",
        active_cond_path=(
            root / "data/animo4d_L4TB_plus_human_v4b272neutral/cond.npy"
        ),
        legacy_truebones_cond_path=root / "data/anytop_truebones/cond.npy",
        t04_candidate_path=(
            root
            / "dataset/.ktjd17_skeleton_generations/"
            "20260819T145532135993Z-77bd88e242a2/candidates/HML3D_Human.npz"
        ),
    )


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _retry_transient_enoent(operation: Any, *, label: str) -> Any:
    """Retry only the observed GPFS transient ENOENT, then fail closed."""
    last_error: FileNotFoundError | None = None
    for delay in TRANSIENT_ENOENT_RETRY_DELAYS_SECONDS:
        if delay:
            time.sleep(delay)
        try:
            return operation()
        except FileNotFoundError as exc:
            last_error = exc
    if last_error is None:  # pragma: no cover - the loop is statically non-empty
        raise Human312AuditError(f"{label} retry schedule is empty")
    last_error.add_note(
        f"{label} remained unavailable after "
        f"{len(TRANSIENT_ENOENT_RETRY_DELAYS_SECONDS)} attempts"
    )
    raise last_error


def _load_json(path: Path) -> Any:
    try:
        text = _retry_transient_enoent(
            lambda: path.read_text(encoding="utf-8"),
            label=f"JSON read {path}",
        )
        return json.loads(text)
    except Exception as exc:  # noqa: BLE001
        raise Human312AuditError(f"cannot read JSON {path}: {exc}") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise Human312AuditError(f"{path}:{line_number}: blank JSONL row")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise Human312AuditError(
                        f"{path}:{line_number}: row is not an object"
                    )
                records.append(value)
    except Human312AuditError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise Human312AuditError(f"cannot read JSONL {path}: {exc}") from exc
    return records


def _write_bytes_atomic(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            _retry_transient_enoent(
                lambda: os.fsync(handle.fileno()),
                label=f"file fsync {temporary}",
            )
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, value: Any) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    _write_bytes_atomic(path, payload)


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    payload = b"".join(_canonical_json(dict(record)) + b"\n" for record in records)
    _write_bytes_atomic(path, payload)


def _write_npz_atomic(path: Path, payload: Mapping[str, np.ndarray]) -> str:
    if any(np.asarray(value).dtype.hasobject for value in payload.values()):
        raise Human312AuditError("object dtype is forbidden in Human skeleton payload")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **payload)
        with temporary.open("rb") as handle:
            _retry_transient_enoent(
                lambda: os.fsync(handle.fileno()),
                label=f"NPZ fsync {temporary}",
            )
        os.replace(temporary, path)
        _fsync_directory(path.parent)
        return _sha256_file(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_stable_source_bytes(path: Path) -> tuple[bytes, str, dict[str, int]]:
    """Read one regular source twice through one no-follow descriptor."""
    requested = path.expanduser().absolute()
    before = _regular_stat(requested, label="Human source")
    if int(before.st_nlink) != 1:
        raise Human312AuditError(
            f"Human source has hard-link aliases: {requested}"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(requested, flags)
    except OSError as exc:
        raise Human312AuditError(f"cannot open Human source {requested}: {exc}") from exc
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            opened = os.fstat(handle.fileno())
            if (
                int(opened.st_dev) != int(before.st_dev)
                or int(opened.st_ino) != int(before.st_ino)
                or int(opened.st_nlink) != 1
            ):
                raise Human312AuditError(
                    f"Human source changed while opening: {requested}"
                )
            first = handle.read()
            middle = os.fstat(handle.fileno())
            handle.seek(0)
            second = handle.read()
            after_fd = os.fstat(handle.fileno())
    except OSError as exc:
        raise Human312AuditError(f"cannot stably read Human source {requested}: {exc}") from exc
    after_path = _regular_stat(requested, label="Human source post-read")
    stat_fields = ("st_dev", "st_ino", "st_nlink", "st_size", "st_mtime_ns")
    snapshots = (before, opened, middle, after_fd, after_path)
    for field in stat_fields:
        values = {int(getattr(value, field)) for value in snapshots}
        if len(values) != 1:
            raise Human312AuditError(
                f"Human source changed during stable read ({field}): {requested}"
            )
    if first != second or len(first) != int(after_path.st_size):
        raise Human312AuditError(f"Human source bytes changed during read: {requested}")
    source_sha256 = hashlib.sha256(first).hexdigest()
    return (
        first,
        source_sha256,
        {
            "size_bytes": int(after_path.st_size),
            "mtime_ns": int(after_path.st_mtime_ns),
            "device": int(after_path.st_dev),
            "inode": int(after_path.st_ino),
            "nlink": int(after_path.st_nlink),
        },
    )


def _fsync_directory(path: Path) -> None:
    def sync_once() -> None:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    _retry_transient_enoent(sync_once, label=f"directory fsync {path}")


def _ensure_canonical_directory(path: Path, *, label: str) -> Path:
    """Create/check one directory without accepting a linked path component."""
    canonical = _absolute_no_resolve(path)
    parent = canonical.parent
    if (
        parent.is_symlink()
        or not parent.is_dir()
        or parent.resolve(strict=True) != parent
    ):
        raise Human312AuditError(f"{label} parent is not canonical: {parent}")
    if os.path.lexists(canonical):
        observed = canonical.lstat()
        if (
            stat.S_ISLNK(observed.st_mode)
            or not stat.S_ISDIR(observed.st_mode)
            or canonical.resolve(strict=True) != canonical
        ):
            raise Human312AuditError(f"{label} is not a canonical directory: {canonical}")
    else:
        os.mkdir(canonical)
        _fsync_directory(parent)
    return canonical


def _regular_stat(path: Path, *, label: str) -> os.stat_result:
    """Return lstat only for a canonical regular file with one directory entry."""
    path = _absolute_no_resolve(path)
    if path.is_symlink():
        raise Human312AuditError(f"{label} is a symlink: {path}")
    try:
        observed = path.lstat()
    except OSError as exc:
        raise Human312AuditError(f"cannot lstat {label} {path}: {exc}") from exc
    if not stat.S_ISREG(observed.st_mode):
        raise Human312AuditError(f"{label} is not a regular file: {path}")
    if int(observed.st_nlink) != 1:
        raise Human312AuditError(
            f"{label} has hard-link aliases (st_nlink={observed.st_nlink}): {path}"
        )
    return observed


def _stable_file_evidence(path: Path, *, label: str) -> dict[str, Any]:
    """Hash one pinned input and prove its path/stat identity stayed unchanged."""
    canonical = _absolute_no_resolve(path)
    if canonical.resolve(strict=True) != canonical:
        raise Human312AuditError(f"{label} has a linked path component: {canonical}")
    _, digest, observed = _read_stable_source_bytes(canonical)
    return {
        "path": str(canonical),
        "sha256": digest,
        **observed,
    }


def _file_manifest(root: Path, *, require_read_only: bool = False) -> dict[str, Any]:
    root = _absolute_no_resolve(root)
    root_stat = root.lstat()
    if (
        root.is_symlink()
        or not stat.S_ISDIR(root_stat.st_mode)
        or root.resolve(strict=True) != root
    ):
        raise Human312AuditError(f"invalid Human generation root: {root}")
    if require_read_only and int(root_stat.st_mode) & 0o222:
        raise Human312AuditError(f"generation root is writable: {root}")
    result: dict[str, Any] = {}
    inode_owner: dict[tuple[int, int], str] = {}
    for path in sorted(root.rglob("*")):
        observed = path.lstat()
        if stat.S_ISLNK(observed.st_mode):
            raise Human312AuditError(f"symlink is forbidden in generation: {path}")
        if stat.S_ISDIR(observed.st_mode):
            if require_read_only and int(observed.st_mode) & 0o222:
                raise Human312AuditError(f"generation directory is writable: {path}")
            continue
        if not stat.S_ISREG(observed.st_mode):
            raise Human312AuditError(f"special entry is forbidden in generation: {path}")
        if int(observed.st_nlink) != 1:
            raise Human312AuditError(
                f"hard-linked generation artifact is forbidden: {path}"
            )
        relpath = path.relative_to(root).as_posix()
        inode = (int(observed.st_dev), int(observed.st_ino))
        if inode in inode_owner:
            raise Human312AuditError(
                f"generation artifacts share an inode: {inode_owner[inode]} and {relpath}"
            )
        inode_owner[inode] = relpath
        if require_read_only and int(observed.st_mode) & 0o222:
            raise Human312AuditError(f"generation artifact is writable: {path}")
        if relpath == "generation.json":
            continue
        result[relpath] = {
            "sha256": _sha256_file(path),
            "size_bytes": int(observed.st_size),
        }
    return result


def _freeze_immutable_tree(root: Path) -> None:
    _file_manifest(root)
    entries = sorted(root.rglob("*"), key=lambda value: len(value.parts), reverse=True)
    for path in entries:
        observed = path.lstat()
        if stat.S_ISREG(observed.st_mode) or stat.S_ISDIR(observed.st_mode):
            os.chmod(path, int(observed.st_mode) & ~0o222)
        else:
            raise Human312AuditError(f"cannot freeze special generation entry: {path}")
    os.chmod(root, int(root.lstat().st_mode) & ~0o222)
    _file_manifest(root, require_read_only=True)
    _fsync_directory(root.parent)


def _replace_symlink(link: Path, target: Path) -> None:
    if os.path.lexists(link) and not link.is_symlink():
        raise Human312AuditError(f"refusing to replace non-symlink {link}")
    relative = os.path.relpath(target, start=link.parent)
    temporary = link.parent / f".{link.name}.{uuid.uuid4().hex}.tmp"
    os.symlink(relative, temporary)
    os.replace(temporary, link)
    _fsync_directory(link.parent)


def _read_relative_symlink_target(link: Path, *, label: str) -> Path:
    """Resolve one local authority link while rejecting absolute/traversal targets."""
    if not link.is_symlink():
        raise Human312AuditError(f"{label} is not a symlink: {link}")
    raw = os.readlink(link)
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts or raw != relative.as_posix():
        raise Human312AuditError(f"{label} target is not a canonical relative path: {raw}")
    target = _absolute_no_resolve(link.parent / relative)
    if target.resolve(strict=True) != target:
        raise Human312AuditError(f"{label} target has a linked path component: {target}")
    return target


@contextlib.contextmanager
def _single_thread_spawn_environment() -> Iterable[None]:
    previous = {key: os.environ.get(key) for key in _THREAD_ENVIRONMENT}
    try:
        os.environ.update(_THREAD_ENVIRONMENT)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _worker_process_status() -> dict[str, int]:
    result = {
        "pid": int(os.getpid()),
        "ppid": int(os.getppid()),
        "threads": -1,
        "vmrss_kib": -1,
    }
    status_path = Path("/proc/self/status")
    if not status_path.is_file():
        raise Human312AuditError("/proc/self/status is unavailable")
    for line in status_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Threads:"):
            result["threads"] = int(line.split(":", 1)[1].strip())
        elif line.startswith("VmRSS:"):
            result["vmrss_kib"] = int(line.split(":", 1)[1].split()[0])
    return _validate_worker_process_status(result)


def _validate_worker_process_status(value: Any) -> dict[str, int]:
    expected = {"pid", "ppid", "threads", "vmrss_kib"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise Human312AuditError("Human worker OS-process evidence schema drifted")
    result: dict[str, int] = {}
    for field in sorted(expected):
        observed = value[field]
        if type(observed) is not int or int(observed) <= 0:
            raise Human312AuditError(
                f"Human worker OS-process evidence is invalid: {field}={observed!r}"
            )
        result[field] = int(observed)
    if result["threads"] != 1:
        raise Human312AuditError(
            f"spawned Human audit worker has {result['threads']} OS threads, expected 1"
        )
    return result


def _worker_statuses_sha256(values: Sequence[Mapping[str, Any]]) -> str:
    normalized = [_validate_worker_process_status(value) for value in values]
    return hashlib.sha256(_canonical_json(normalized)).hexdigest()


def _validate_chunk_execution_evidence(
    value: Any, *, expected_chunk_count: int, allow_cache: bool
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Human312AuditError("Human chunk execution evidence is not an object")
    evidence = dict(value)
    expected_keys = {
        "executor_mode",
        "chunk_count",
        "cached_revalidated_chunk_count",
        "fresh_spawn_chunk_count",
        "fresh_spawn_chunks_with_process_status",
        "cached_worker_process_status_trusted",
    }
    if set(evidence) != expected_keys:
        raise Human312AuditError("Human chunk execution evidence schema drifted")
    integer_fields = (
        "chunk_count",
        "cached_revalidated_chunk_count",
        "fresh_spawn_chunk_count",
        "fresh_spawn_chunks_with_process_status",
    )
    if any(type(evidence.get(field)) is not int for field in integer_fields):
        raise Human312AuditError("Human chunk execution counts are not integers")
    cached = int(evidence["cached_revalidated_chunk_count"])
    fresh = int(evidence["fresh_spawn_chunk_count"])
    if (
        evidence.get("executor_mode") != "spawn"
        or int(evidence["chunk_count"]) != expected_chunk_count
        or cached < 0
        or fresh < 0
        or cached + fresh != expected_chunk_count
        or int(evidence["fresh_spawn_chunks_with_process_status"]) != fresh
        or evidence.get("cached_worker_process_status_trusted") is not False
        or (not allow_cache and cached != 0)
    ):
        raise Human312AuditError("Human chunk execution accounting drifted")
    return {
        "executor_mode": "spawn",
        "chunk_count": expected_chunk_count,
        "cached_revalidated_chunk_count": cached,
        "fresh_spawn_chunk_count": fresh,
        "fresh_spawn_chunks_with_process_status": fresh,
        "cached_worker_process_status_trusted": False,
    }


def _code_closure() -> dict[str, dict[str, Any]]:
    """Hash the recursively imported local producer source closure."""
    module_root = Path(__file__).resolve().parent
    pending = [module_root / "human312_audit.py"]
    init = module_root / "__init__.py"
    if init.is_file():
        pending.append(init)
    visited: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        observed = _regular_stat(path, label="Human audit producer module")
        visited.add(path)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except Exception as exc:  # noqa: BLE001
            raise Human312AuditError(f"cannot parse producer module {path}: {exc}") from exc
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level != 1:
                continue
            candidates: list[str] = []
            if node.module:
                candidates.append(node.module.split(".", 1)[0])
            else:
                candidates.extend(alias.name.split(".", 1)[0] for alias in node.names)
            for candidate in candidates:
                local_file = module_root / f"{candidate}.py"
                local_package = module_root / candidate / "__init__.py"
                if local_file.is_file():
                    pending.append(local_file)
                elif local_package.is_file():
                    pending.append(local_package)
    repo_root = module_root.parents[2]
    return {
        path.relative_to(repo_root).as_posix(): {
            "sha256": _sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
        for path in sorted(visited)
    }


def _distribution_content_fingerprint(distribution: Any) -> dict[str, Any]:
    files = distribution.files
    if files is None:
        raise Human312AuditError("installed distribution exposes no file inventory")
    entries: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for relative in sorted(files, key=lambda value: str(value)):
        path = Path(distribution.locate_file(relative))
        if path.is_symlink():
            raise Human312AuditError(f"runtime distribution file is a symlink: {path}")
        try:
            resolved = path.resolve(strict=True)
            observed = resolved.stat()
        except OSError as exc:
            raise Human312AuditError(
                f"cannot resolve runtime distribution file {path}: {exc}"
            ) from exc
        if not stat.S_ISREG(observed.st_mode):
            raise Human312AuditError(f"runtime distribution entry is not regular: {resolved}")
        canonical_path = str(resolved)
        if canonical_path in seen_paths:
            raise Human312AuditError(f"runtime distribution aliases one file: {resolved}")
        seen_paths.add(canonical_path)
        entries.append(
            {
                "relative_path": str(relative),
                "resolved_path": canonical_path,
                "sha256": _sha256_file(resolved),
                "size_bytes": int(observed.st_size),
            }
        )
    return {
        "actual_file_count": len(entries),
        "actual_total_size_bytes": sum(int(value["size_bytes"]) for value in entries),
        "actual_files_sha256": hashlib.sha256(_canonical_json(entries)).hexdigest(),
    }


def _native_library_dependency_fingerprint(roots: Sequence[Path]) -> dict[str, Any]:
    libraries: dict[str, dict[str, Any]] = {}
    pending = [path.resolve(strict=True) for path in roots]
    while pending:
        path = pending.pop()
        canonical = str(path)
        if canonical in libraries:
            continue
        observed = path.stat()
        if not stat.S_ISREG(observed.st_mode):
            raise Human312AuditError(f"native runtime dependency is not regular: {path}")
        libraries[canonical] = {
            "sha256": _sha256_file(path),
            "size_bytes": int(observed.st_size),
        }
        completed = subprocess.run(
            ["ldd", canonical], check=False, capture_output=True, text=True
        )
        output = completed.stdout + completed.stderr
        if "not found" in output:
            raise Human312AuditError(
                f"native runtime dependency is unresolved for {path}: {output.strip()}"
            )
        if completed.returncode not in (0, 1):
            raise Human312AuditError(f"ldd failed for {path}: {output.strip()}")
        for line in output.splitlines():
            value = line.strip()
            if "=>" in value:
                value = value.split("=>", 1)[1].strip()
            if not value.startswith("/"):
                continue
            dependency = Path(value.split(" (", 1)[0]).resolve(strict=True)
            if str(dependency) not in libraries:
                pending.append(dependency)
    if not libraries:
        raise Human312AuditError("no native runtime dependency was discoverable")
    ordered = {name: libraries[name] for name in sorted(libraries)}
    return {
        "library_count": len(ordered),
        "libraries": ordered,
        "closure_sha256": hashlib.sha256(_canonical_json(ordered)).hexdigest(),
    }


def _runtime_fingerprint() -> dict[str, Any]:
    with threadpoolctl.threadpool_limits(limits=1):
        pools = threadpoolctl.threadpool_info()
    if not pools:
        raise Human312AuditError("no loaded BLAS runtime was discoverable")
    normalized_pools: list[dict[str, Any]] = []
    for pool in pools:
        library = Path(str(pool.get("filepath", ""))).resolve()
        if not library.is_file():
            raise Human312AuditError(f"BLAS runtime library is absent: {library}")
        entry = {
            "user_api": str(pool.get("user_api")),
            "internal_api": str(pool.get("internal_api")),
            "prefix": str(pool.get("prefix")),
            "filepath": str(library),
            "library_sha256": _sha256_file(library),
            "library_size_bytes": int(library.stat().st_size),
            "version": str(pool.get("version")),
            "threading_layer": str(pool.get("threading_layer")),
            "architecture": str(pool.get("architecture")),
            "effective_num_threads": int(pool.get("num_threads", -1)),
        }
        if entry["effective_num_threads"] != 1:
            raise Human312AuditError(f"BLAS runtime is not capped to one thread: {entry}")
        normalized_pools.append(entry)
    normalized_pools.sort(key=lambda item: _canonical_json(item))
    extension_modules: dict[str, dict[str, Any]] = {}
    for module_name in ("numpy._core._multiarray_umath", "numpy.linalg._umath_linalg"):
        module = importlib.import_module(module_name)
        path = Path(str(module.__file__)).resolve()
        extension_modules[module_name] = {
            "path": str(path),
            "sha256": _sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
    distributions: dict[str, dict[str, Any]] = {}
    for distribution_name in ("numpy", "threadpoolctl"):
        distribution = importlib.metadata.distribution(distribution_name)
        record = distribution.read_text("RECORD")
        if record is None:
            raise Human312AuditError(
                f"installed distribution lacks RECORD: {distribution_name}"
            )
        distributions[distribution_name] = {
            "record_sha256": hashlib.sha256(record.encode("utf-8")).hexdigest(),
            **_distribution_content_fingerprint(distribution),
        }
    python_executable = Path(sys.executable).resolve()
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_full_version": sys.version,
        "python_executable": str(python_executable),
        "python_executable_sha256": _sha256_file(python_executable),
        "numpy_version": np.__version__,
        "threadpoolctl_version": threadpoolctl.__version__,
        "byteorder": sys.byteorder,
        "installed_distributions": distributions,
        "extension_modules": extension_modules,
        "blas_pools": normalized_pools,
        "native_library_dependencies": _native_library_dependency_fingerprint(
            [
                python_executable,
                *(Path(value["path"]) for value in extension_modules.values()),
                *(Path(value["filepath"]) for value in normalized_pools),
            ]
        ),
    }


def _validate_parent_manifest(root: Path) -> dict[str, Any]:
    generation_root = _absolute_no_resolve(root)
    if (
        generation_root.is_symlink()
        or not generation_root.is_dir()
        or generation_root.resolve(strict=True) != generation_root
    ):
        raise Human312AuditError(f"invalid parent inventory root: {generation_root}")
    transaction_path = generation_root / "inventory_generation.json"
    transaction_stat = _regular_stat(transaction_path, label="parent transaction")
    try:
        transaction = _validate_transaction(generation_root)
    except Exception as exc:  # noqa: BLE001
        raise Human312AuditError(f"parent inventory transaction is invalid: {exc}") from exc
    if transaction.get("generation_id") != generation_root.name:
        raise Human312AuditError("parent inventory generation id/path mismatch")
    if transaction.get("publish_protocol") != "immutable_generation_atomic_symlink_replace":
        raise Human312AuditError("parent inventory publish protocol drifted")
    files = transaction.get("files")
    if not isinstance(files, Mapping) or set(files) != EXPECTED_PARENT_MANIFEST_FILES:
        raise Human312AuditError(
            "parent inventory does not expose the required twelve-file closure"
        )
    observed: dict[str, dict[str, Any]] = {}
    inode_owner: dict[tuple[int, int], str] = {
        (int(transaction_stat.st_dev), int(transaction_stat.st_ino)):
        "inventory_generation.json"
    }
    for name, expected in sorted(files.items()):
        if not isinstance(name, str) or not isinstance(expected, Mapping):
            raise Human312AuditError("parent inventory file schema is invalid")
        path = generation_root / name
        stat_result = _regular_stat(path, label="parent inventory artifact")
        if path.resolve(strict=True) != path:
            raise Human312AuditError(f"parent artifact has a linked component: {name}")
        inode = (int(stat_result.st_dev), int(stat_result.st_ino))
        if inode in inode_owner:
            raise Human312AuditError(
                f"parent artifacts share an inode: {inode_owner[inode]} and {name}"
            )
        inode_owner[inode] = name
        actual = {
            "sha256": _sha256_file(path),
            "size_bytes": int(stat_result.st_size),
        }
        if actual != dict(expected):
            raise Human312AuditError(f"parent inventory artifact drifted: {name}")
        observed[name] = actual
    return {
        "generation_id": str(transaction.get("generation_id")),
        "manifest_version": str(transaction.get("manifest_version")),
        "publish_protocol": str(transaction.get("publish_protocol")),
        "inventory_generation_sha256": _sha256_file(transaction_path),
        "files": observed,
    }


def _load_human_scope(
    manifest_root: Path, source_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_root = _absolute_no_resolve(source_root)
    if (
        source_root.is_symlink()
        or not source_root.is_dir()
        or source_root.resolve(strict=True) != source_root
    ):
        raise Human312AuditError(f"invalid or symlinked Human source root: {source_root}")
    rigs = [
        value
        for value in _load_jsonl(manifest_root / "rigs.jsonl")
        if value.get("rig_id") == HUMAN_RIG_ID
    ]
    if len(rigs) != EXPECTED_HUMAN_RIG_COUNT:
        raise Human312AuditError(f"expected one Human rig, got {len(rigs)}")
    rig = rigs[0]
    if (
        rig.get("source_family") != "motionstreamer272"
        or rig.get("source_kind") != "motionstreamer272_rotation6d"
        or rig.get("rotation_provenance_status") != "proven"
    ):
        raise Human312AuditError("Human rig rotation authority is not proven")
    joint_map = rig.get("joint_map")
    if not isinstance(joint_map, Mapping):
        raise Human312AuditError("Human rig lacks joint_map")
    if (
        len(joint_map.get("btjd_joint_names", [])) != EXPECTED_HUMAN_JOINT_COUNT
        or list(joint_map.get("btjd_parents", []))[0] != -1
        or int(joint_map.get("fixed_dof_count", -1)) != 0
        or int(joint_map.get("animated_dof_count", -1))
        != EXPECTED_HUMAN_JOINT_COUNT
    ):
        raise Human312AuditError("Human joint map violates the 22-joint authority")

    tasks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_relpaths: set[str] = set()
    with (manifest_root / "clips.jsonl").open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise Human312AuditError(
                    f"clips.jsonl:{line_number}: blank JSONL row"
                )
            record = json.loads(line)
            source = record.get("source")
            if not isinstance(source, Mapping) or source.get("family") != "motionstreamer272":
                continue
            clip_id = str(record.get("clip_id"))
            if record.get("rig_id") != HUMAN_RIG_ID or clip_id in seen_ids:
                raise Human312AuditError(f"invalid or duplicate Human clip {clip_id}")
            source_path = _absolute_no_resolve(str(source.get("path")))
            if source_path.resolve(strict=True) != source_path:
                raise Human312AuditError(
                    f"{clip_id}: source path contains a symlink component"
                )
            try:
                relpath = source_path.relative_to(source_root).as_posix()
            except ValueError as exc:
                raise Human312AuditError(
                    f"{clip_id}: source path escapes configured Human root"
                ) from exc
            if relpath in seen_relpaths:
                raise Human312AuditError(f"duplicate Human source relpath {relpath}")
            if source_path.parent != source_root or source_path.name != (
                clip_id.removeprefix(f"{HUMAN_RIG_ID}_") + ".npy"
            ):
                raise Human312AuditError(
                    f"{clip_id}: source is not the exact direct Human-root child"
                )
            observed = _regular_stat(source_path, label=f"{clip_id} source")
            expected_shape = source.get("shape")
            if (
                source.get("kind") != "motionstreamer272_rotation6d"
                or source.get("dtype") != "float64"
                or expected_shape != [int(source.get("T_src", -1)), EXPECTED_HUMAN_SOURCE_DIM]
                or source.get("rotation_slice") != [140, 272]
                or source.get("rotation_shape") != [22, 6]
                or float(source.get("fps_src", -1.0)) != EXPECTED_HUMAN_FPS
                or source.get("slice_frames") != [0, int(source.get("T_src", -1))]
                or int(observed.st_size) != int(source.get("file_size_bytes", -1))
                or int(observed.st_mtime_ns) != int(source.get("mtime_ns", -1))
            ):
                raise Human312AuditError(f"{clip_id}: manifest source schema drifted")
            tasks.append(
                {
                    "clip_id": clip_id,
                    "rig_id": HUMAN_RIG_ID,
                    "topology_family": "human",
                    "split": record.get("split"),
                    "source_relpath": relpath,
                    "file_size_bytes": int(source["file_size_bytes"]),
                    "mtime_ns": int(source["mtime_ns"]),
                    "source_device": int(observed.st_dev),
                    "source_inode": int(observed.st_ino),
                    "source_nlink": int(observed.st_nlink),
                    "T_src": int(source["T_src"]),
                    "fps_src": float(source["fps_src"]),
                    "source_shape": list(expected_shape),
                    "source_dtype": str(source["dtype"]),
                    "rotation_slice": [140, 272],
                    "rotation_shape": [22, 6],
                }
            )
            seen_ids.add(clip_id)
            seen_relpaths.add(relpath)
    tasks.sort(key=lambda value: value["clip_id"])
    if len(tasks) != EXPECTED_HUMAN_CLIP_COUNT:
        raise Human312AuditError(
            f"expected {EXPECTED_HUMAN_CLIP_COUNT} Human clips, got {len(tasks)}"
        )

    _assert_manifest_disk_bijection(source_root, tasks)
    return rig, tasks


def _assert_manifest_disk_bijection(
    source_root: Path, tasks: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Require a flat, one-link, inode-unique NPY tree exactly equal to scope."""
    root = _absolute_no_resolve(source_root)
    if root.is_symlink() or not root.is_dir() or root.resolve(strict=True) != root:
        raise Human312AuditError(f"invalid or symlinked Human source root: {root}")
    expected = {str(task["source_relpath"]): task for task in tasks}
    if len(expected) != len(tasks):
        raise Human312AuditError("duplicate Human manifest source relpath")
    observed: dict[str, os.stat_result] = {}
    inode_owner: dict[tuple[int, int], str] = {}
    try:
        with os.scandir(root) as iterator:
            for entry in iterator:
                if entry.is_symlink():
                    raise Human312AuditError(f"symlink in Human source root: {entry.name}")
                entry_stat = entry.stat(follow_symlinks=False)
                if not stat.S_ISREG(entry_stat.st_mode) or not entry.name.endswith(".npy"):
                    raise Human312AuditError(
                        f"unexpected non-NPY/non-regular Human root entry: {entry.name}"
                    )
                if int(entry_stat.st_nlink) != 1:
                    raise Human312AuditError(
                        f"hard-linked Human source (st_nlink={entry_stat.st_nlink}): "
                        f"{entry.name}"
                    )
                inode = (int(entry_stat.st_dev), int(entry_stat.st_ino))
                if inode in inode_owner:
                    raise Human312AuditError(
                        f"same-inode Human sources: {inode_owner[inode]} and {entry.name}"
                    )
                inode_owner[inode] = entry.name
                observed[entry.name] = entry_stat
    except OSError as exc:
        raise Human312AuditError(f"cannot enumerate Human source root {root}: {exc}") from exc
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))[:20]
        extra = sorted(set(observed) - set(expected))[:20]
        raise Human312AuditError(
            f"Human manifest/disk scope mismatch: missing={missing}, extra={extra}, "
            f"manifest={len(expected)}, disk={len(observed)}"
        )
    snapshot: list[dict[str, Any]] = []
    for relpath in sorted(expected):
        task = expected[relpath]
        observed_stat = observed[relpath]
        checks = {
            "file_size_bytes": int(observed_stat.st_size),
            "mtime_ns": int(observed_stat.st_mtime_ns),
            "source_device": int(observed_stat.st_dev),
            "source_inode": int(observed_stat.st_ino),
            "source_nlink": int(observed_stat.st_nlink),
        }
        for field, actual in checks.items():
            if int(task[field]) != actual:
                raise Human312AuditError(
                    f"Human source snapshot {field} drifted for {relpath}: "
                    f"{actual} != {task[field]}"
                )
        snapshot.append({"source_relpath": relpath, **checks})
    return {
        "entry_count": len(snapshot),
        "snapshot_sha256": hashlib.sha256(_canonical_json(snapshot)).hexdigest(),
    }


def _independent_row_cont6d(value: np.ndarray) -> np.ndarray:
    source = np.asarray(value, dtype=np.float64)
    if source.shape[-1] != 6 or not np.isfinite(source).all():
        raise HumanSourceContentError("independent row-cont6d input is invalid")
    a1 = source[..., :3]
    a2 = source[..., 3:]
    n1 = np.linalg.norm(a1, axis=-1, keepdims=True)
    if np.any(n1 < SOURCE_D6_DEGENERACY_EPS):
        raise HumanSourceContentError("independent row-cont6d first row is degenerate")
    b1 = a1 / n1
    u2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    n2 = np.linalg.norm(u2, axis=-1, keepdims=True)
    if np.any(n2 < SOURCE_D6_DEGENERACY_EPS):
        raise HumanSourceContentError("independent row-cont6d second row is degenerate")
    b2 = u2 / n2
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=-2)


def independent_motionstreamer272_decode(
    data: np.ndarray, parents: Sequence[int]
) -> dict[str, np.ndarray]:
    """Independent full decode used only by the audit cross-check."""
    source = np.asarray(data)
    if (
        source.dtype != np.float64
        or source.ndim != 2
        or source.shape[1] != MOTIONSTREAMER272_DIM
        or source.shape[0] <= 0
        or not np.isfinite(source).all()
    ):
        raise HumanSourceContentError(
            f"MotionStreamer272 must be finite float64 [T,272], got {source.shape}/{source.dtype}"
        )
    parent_array = np.asarray(parents, dtype=np.int64)
    if parent_array.shape != (MOTIONSTREAMER272_JOINTS,):
        raise Human312AuditError("independent decoder parent shape drifted")
    positions_no_heading = source[:, MOTIONSTREAMER272_POSITION_SLICE].reshape(
        source.shape[0], MOTIONSTREAMER272_JOINTS, 3
    )
    heading_delta = _independent_row_cont6d(source[:, 2:8])
    heading = np.empty_like(heading_delta)
    heading[0] = heading_delta[0]
    for frame in range(1, source.shape[0]):
        heading[frame] = heading_delta[frame] @ heading[frame - 1]
    inverse_heading = np.swapaxes(heading, -1, -2)
    positions = np.einsum("tij,tkj->tki", inverse_heading, positions_no_heading)
    velocity = np.zeros((source.shape[0], 3), dtype=np.float64)
    velocity[:, 0] = source[:, 0]
    velocity[:, 2] = source[:, 1]
    if source.shape[0] > 1:
        velocity[1:] = np.einsum(
            "tij,tj->ti", inverse_heading[:-1], velocity[1:]
        )
    root_translation = np.cumsum(velocity, axis=0)
    positions[..., 0] += root_translation[:, None, 0]
    positions[..., 2] += root_translation[:, None, 2]

    local = _independent_row_cont6d(
        source[:, MOTIONSTREAMER272_ROTATION_SLICE].reshape(
            source.shape[0], MOTIONSTREAMER272_JOINTS, 6
        )
    )
    local[:, 0] = inverse_heading @ local[:, 0]
    global_rotations = np.empty_like(local)
    global_rotations[:, 0] = local[:, 0]
    for child in range(1, MOTIONSTREAMER272_JOINTS):
        global_rotations[:, child] = (
            global_rotations[:, int(parent_array[child])] @ local[:, child]
        )
    return {
        "positions": positions,
        "root_translation": positions[:, 0].copy(),
        "local_rotations": local,
        "global_rotations": global_rotations,
    }


def _fixed_neutral_fk(
    root_positions: np.ndarray,
    local_rotations: np.ndarray,
    parents: np.ndarray,
    offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frames, joints = local_rotations.shape[:2]
    positions = np.empty((frames, joints, 3), dtype=np.float64)
    global_rotations = np.empty((frames, joints, 3, 3), dtype=np.float64)
    positions[:, 0] = root_positions
    global_rotations[:, 0] = local_rotations[:, 0]
    for child in range(1, joints):
        parent = int(parents[child])
        global_rotations[:, child] = global_rotations[:, parent] @ local_rotations[:, child]
        positions[:, child] = positions[:, parent] + np.einsum(
            "tij,j->ti", global_rotations[:, parent], offsets[child]
        )
    return positions, global_rotations


def _rigid_edge_max_norm(
    positions: np.ndarray, parents: np.ndarray, offsets: np.ndarray, s_rig: float
) -> float:
    errors: list[np.ndarray] = []
    for child in range(1, len(parents)):
        parent = int(parents[child])
        observed = np.linalg.norm(positions[:, child] - positions[:, parent], axis=-1)
        errors.append(np.abs(observed - np.linalg.norm(offsets[child])))
    return float(np.max(np.stack(errors, axis=-1)) / s_rig)


def _dynamic_metrics(
    positions: np.ndarray, global_rotations: np.ndarray, *, fps: float, s_rig: float
) -> dict[str, float]:
    if positions.shape[0] <= 1:
        return {
            "root_speed_rms_norm_per_s": 0.0,
            "rotation_speed_rms_rad_per_s": 0.0,
            "pose_excursion_rms_norm": 0.0,
            "dynamic_score": 0.0,
        }
    root_step = np.diff(positions[:, 0], axis=0) * fps
    root_rms = float(np.sqrt(np.mean(np.square(root_step))) / s_rig)
    relative = np.matmul(
        global_rotations[1:], np.swapaxes(global_rotations[:-1], -1, -2)
    )
    traces = np.trace(relative, axis1=-2, axis2=-1)
    angles = np.arccos(np.clip((traces - 1.0) * 0.5, -1.0, 1.0)) * fps
    rotation_rms = float(np.sqrt(np.mean(np.square(angles))))
    centered = positions - positions[:, :1]
    pose_excursion = float(
        np.sqrt(np.mean(np.square(centered - centered[:1]))) / s_rig
    )
    return {
        "root_speed_rms_norm_per_s": root_rms,
        "rotation_speed_rms_rad_per_s": rotation_rms,
        "pose_excursion_rms_norm": pose_excursion,
        "dynamic_score": root_rms + rotation_rms + pose_excursion,
    }


def _rotation_payload_metrics(data: np.ndarray) -> dict[str, Any]:
    heading = np.asarray(data[:, 2:8], dtype=np.float64).reshape(len(data), 1, 6)
    local = np.asarray(
        data[:, MOTIONSTREAMER272_ROTATION_SLICE], dtype=np.float64
    ).reshape(len(data), EXPECTED_HUMAN_JOINT_COUNT, 6)
    rows = np.concatenate((heading, local), axis=1)
    first = rows[..., :3]
    second = rows[..., 3:]
    first_norm = np.linalg.norm(first, axis=-1)
    second_norm = np.linalg.norm(second, axis=-1)
    cross_norm = np.linalg.norm(np.cross(first, second), axis=-1)
    if (
        np.any(first_norm < SOURCE_D6_DEGENERACY_EPS)
        or np.any(second_norm < SOURCE_D6_DEGENERACY_EPS)
        or np.any(cross_norm < SOURCE_D6_DEGENERACY_EPS)
    ):
        raise HumanClipReject(
            "HUMAN_SOURCE_D6_DEGENERATE", "raw MotionStreamer272 6D rows are degenerate"
        )
    metrics = {
        "raw_d6_first_row_unit_max_abs": float(np.max(np.abs(first_norm - 1.0))),
        "raw_d6_second_row_unit_max_abs": float(np.max(np.abs(second_norm - 1.0))),
        "raw_d6_row_dot_max_abs": float(np.max(np.abs(np.sum(first * second, axis=-1)))),
        "raw_d6_cross_norm_min": float(np.min(cross_norm)),
    }
    if (
        metrics["raw_d6_first_row_unit_max_abs"] > RAW_D6_UNIT_NORM_MAX_ABS
        or metrics["raw_d6_second_row_unit_max_abs"] > RAW_D6_UNIT_NORM_MAX_ABS
        or metrics["raw_d6_row_dot_max_abs"] > RAW_D6_ROW_DOT_MAX_ABS
        or metrics["raw_d6_cross_norm_min"] < RAW_D6_CROSS_NORM_MIN
    ):
        raise HumanClipReject(
            "HUMAN_RAW_D6_NOT_ROTATION_LIKE",
            f"raw MotionStreamer272 6D rows exceed rotation-like thresholds: {metrics}",
        )
    digest = hashlib.sha256()
    digest.update(
        _canonical_json(
            {
                "dtype": str(rows.dtype),
                "shape": list(rows.shape),
                "layout": "heading_2_8_then_local_140_272_row_cont6d",
            }
        )
    )
    digest.update(np.ascontiguousarray(rows).tobytes(order="C"))
    return {"rotation_payload_sha256": digest.hexdigest(), **metrics}


def _assert_task_source_stat(
    task: Mapping[str, Any], observed: os.stat_result, *, phase: str
) -> None:
    expected = {
        "st_size": int(task["file_size_bytes"]),
        "st_mtime_ns": int(task["mtime_ns"]),
        "st_dev": int(task["source_device"]),
        "st_ino": int(task["source_inode"]),
        "st_nlink": int(task["source_nlink"]),
    }
    for field, value in expected.items():
        if int(getattr(observed, field)) != value:
            raise HumanClipReject(
                "HUMAN_SOURCE_CHANGED_DURING_AUDIT",
                f"{task['clip_id']}: source {field} drifted during {phase}",
            )


def _initialize_worker(
    rig: Mapping[str, Any], fixed: Mapping[str, Any], source_root: str
) -> None:
    global _WORKER_RIG, _WORKER_FIXED, _WORKER_SOURCE_ROOT, _WORKER_THREAD_LIMITER
    for name in _THREAD_ENVIRONMENT:
        os.environ[name] = "1"
    _WORKER_THREAD_LIMITER = threadpoolctl.threadpool_limits(limits=1)
    pools = threadpoolctl.threadpool_info()
    if not pools or any(int(pool.get("num_threads", -1)) != 1 for pool in pools):
        raise Human312AuditError(f"Human worker BLAS pools are not single-threaded: {pools}")
    _WORKER_RIG = dict(rig)
    _WORKER_FIXED = {
        key: np.asarray(value).copy() if isinstance(value, np.ndarray) else value
        for key, value in fixed.items()
    }
    _WORKER_SOURCE_ROOT = Path(source_root)


def _audit_one(task: Mapping[str, Any]) -> dict[str, Any]:
    if _WORKER_RIG is None or _WORKER_FIXED is None or _WORKER_SOURCE_ROOT is None:
        raise Human312AuditError("Human audit worker was not initialized")
    clip_id = str(task["clip_id"])
    source_path = _WORKER_SOURCE_ROOT / str(task["source_relpath"])
    common = {
        "audit_version": HUMAN312_AUDIT_VERSION,
        "clip_id": clip_id,
        "rig_id": HUMAN_RIG_ID,
        "source_family": "motionstreamer272",
        "topology_family": "human",
        "split": task.get("split"),
        "source_relpath": str(task["source_relpath"]),
        "source_size_bytes": int(task["file_size_bytes"]),
        "source_mtime_ns": int(task["mtime_ns"]),
        "source_device": int(task["source_device"]),
        "source_inode": int(task["source_inode"]),
        "source_nlink": int(task["source_nlink"]),
        "source_shape": list(task["source_shape"]),
        "source_dtype": str(task["source_dtype"]),
        "rotation_slice": list(task["rotation_slice"]),
        "rotation_shape": list(task["rotation_shape"]),
        "T_src": int(task["T_src"]),
        "J_phys": EXPECTED_HUMAN_JOINT_COUNT,
        "fps_src": float(task["fps_src"]),
    }
    source_sha256: str | None = None
    try:
        if source_path.parent != _WORKER_SOURCE_ROOT or source_path.resolve(strict=True) != source_path:
            raise HumanClipReject(
                "HUMAN_SOURCE_CHANGED_DURING_AUDIT",
                f"{clip_id}: source escaped the canonical Human root",
            )
        try:
            source_bytes, source_sha256, stable = _read_stable_source_bytes(source_path)
        except Human312AuditError as exc:
            raise HumanClipReject(
                "HUMAN_SOURCE_CHANGED_DURING_AUDIT",
                f"{clip_id}: stable source read failed: {exc}",
            ) from exc
        expected_stable = {
            "size_bytes": int(task["file_size_bytes"]),
            "mtime_ns": int(task["mtime_ns"]),
            "device": int(task["source_device"]),
            "inode": int(task["source_inode"]),
            "nlink": int(task["source_nlink"]),
        }
        if stable != expected_stable:
            raise HumanClipReject(
                "HUMAN_SOURCE_CHANGED_DURING_AUDIT",
                f"{clip_id}: stable source identity drifted",
            )
        try:
            raw = np.load(io.BytesIO(source_bytes), allow_pickle=False)
        except (ValueError, OSError, EOFError) as exc:
            raise HumanClipReject(
                "HUMAN_SOURCE_NPY_LOAD_FAILURE",
                f"{clip_id}: cannot load stable NPY bytes: {exc}",
            ) from exc
        if raw.dtype != np.float64 or list(raw.shape) != list(task["source_shape"]):
            raise HumanClipReject(
                "HUMAN_SOURCE_SCHEMA_INVALID",
                f"{clip_id}: source shape/dtype drift: {raw.shape}/{raw.dtype}"
            )
        data = np.asarray(raw, dtype=np.float64)
        if not np.isfinite(data).all():
            raise HumanClipReject(
                "HUMAN_SOURCE_NONFINITE", f"{clip_id}: source contains non-finite values"
            )
        raw_rotation_metrics = _rotation_payload_metrics(data)
        joint_map = _WORKER_RIG["joint_map"]
        parents = np.asarray(joint_map["btjd_parents"], dtype=np.int64)
        try:
            independent = independent_motionstreamer272_decode(data, parents)
        except HumanSourceContentError as exc:
            raise HumanClipReject(
                "HUMAN_SOURCE_PARSE_FAILURE",
                f"{clip_id}: independent decoder failed: {exc}",
            ) from exc
        try:
            parsed = parse_motionstreamer272_fixed_neutral_array(
                data,
                source_identity=str(task["source_relpath"]),
                joint_names=joint_map["btjd_joint_names"],
                parents=parents,
                P_rest_global=np.asarray(_WORKER_FIXED["P_rest_global"]),
                offset_parent_local=np.asarray(_WORKER_FIXED["offsets"]),
                rest_authority=HUMAN_CONTRACT_VERSION,
            )
        except MotionStreamer272ContentError as exc:
            raise HumanClipReject(
                "HUMAN_SOURCE_PARSE_FAILURE",
                f"{clip_id}: production decoder failed: {exc}",
            ) from exc
        dual = {
            "positions_max_abs": float(
                np.max(np.abs(independent["positions"] - parsed.source_positions))
            ),
            "root_translation_max_abs": float(
                np.max(
                    np.abs(
                        independent["root_translation"] - parsed.root_translation
                    )
                )
            ),
            "local_rotation_max_abs": float(
                np.max(
                    np.abs(independent["local_rotations"] - parsed.local_rotations)
                )
            ),
            "global_rotation_max_abs": float(
                np.max(
                    np.abs(
                        independent["global_rotations"] - parsed.global_rotations
                    )
                )
            ),
        }
        if (
            max(dual["positions_max_abs"], dual["root_translation_max_abs"])
            > DUAL_POSITION_MAX_ABS
            or max(dual["local_rotation_max_abs"], dual["global_rotation_max_abs"])
            > DUAL_ROTATION_MAX_ABS
        ):
            raise HumanClipReject(
                "HUMAN_INDEPENDENT_DECODER_MISMATCH",
                f"{clip_id}: independent dual decoder mismatch {dual}",
            )
        parser_fk = source_fk_metrics(parsed)
        if float(parser_fk["source_parser_fk_max_norm"]) > SOURCE_FK_MAX_NORM:
            raise HumanClipReject(
                "HUMAN_SOURCE_FK_FAILURE",
                f"{clip_id}: source-FK exceeds threshold: {parser_fk}"
            )
        offsets = np.asarray(_WORKER_FIXED["offsets"], dtype=np.float64)
        s_rig = float(_WORKER_FIXED["s_rig"])
        fixed_positions, fixed_global = _fixed_neutral_fk(
            parsed.root_translation,
            parsed.local_rotations,
            parents,
            offsets,
        )
        rigid_edge = _rigid_edge_max_norm(fixed_positions, parents, offsets, s_rig)
        if rigid_edge > FIXED_RIGID_EDGE_MAX_NORM:
            raise HumanClipReject(
                "HUMAN_FIXED_NEUTRAL_FK_FAILURE",
                f"{clip_id}: fixed-neutral rigid edge failure {rigid_edge}"
            )
        rotation_diag = rotation_matrix_diagnostics(fixed_global)
        determinant_error = max(
            abs(float(rotation_diag["rotation_determinant_min"]) - 1.0),
            abs(float(rotation_diag["rotation_determinant_max"]) - 1.0),
        )
        if (
            float(rotation_diag["rotation_orthogonality_max"])
            > ROTATION_ORTHOGONALITY_MAX_ABS
            or determinant_error > ROTATION_DETERMINANT_MAX_ABS
        ):
            raise HumanClipReject(
                "HUMAN_ROTATION_INVALID",
                f"{clip_id}: rotation SO3 failure {rotation_diag}"
            )
        metrics = {
            "independent_decoder": HUMAN312_INDEPENDENT_DECODER,
            **raw_rotation_metrics,
            "independent_positions_max_abs": dual["positions_max_abs"],
            "independent_root_translation_max_abs": dual[
                "root_translation_max_abs"
            ],
            "independent_local_rotation_max_abs": dual[
                "local_rotation_max_abs"
            ],
            "independent_global_rotation_max_abs": dual[
                "global_rotation_max_abs"
            ],
            "source_parser_fk_max_norm": float(
                parser_fk["source_parser_fk_max_norm"]
            ),
            "source_parser_fk_mpjpe_norm": float(
                parser_fk["source_parser_fk_error_norm"]
            ),
            "fixed_neutral_rigid_edge_max_norm": rigid_edge,
            "rotation_orthogonality_max_abs": float(
                rotation_diag["rotation_orthogonality_max"]
            ),
            "rotation_determinant_min": float(
                rotation_diag["rotation_determinant_min"]
            ),
            "rotation_determinant_max": float(
                rotation_diag["rotation_determinant_max"]
            ),
            **_dynamic_metrics(
                fixed_positions,
                fixed_global,
                fps=float(task["fps_src"]),
                s_rig=s_rig,
            ),
        }
        return {
            **common,
            "status": "pass",
            "reason_codes": [],
            "source_sha256": source_sha256,
            "metrics": metrics,
        }
    except HumanClipReject as exc:
        if source_sha256 is None:
            try:
                source_sha256 = _stable_file_evidence(
                    source_path, label=f"rejected Human source {clip_id}"
                )["sha256"]
            except Human312AuditError:
                source_sha256 = None
        return {
            **common,
            "status": "reject",
            "reason_codes": [exc.reason_code],
            "source_sha256": source_sha256,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _audit_chunk(tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "records": [_audit_one(task) for task in tasks],
        "worker_process_status": _worker_process_status(),
    }


def _records_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(_canonical_json(list(records))).hexdigest()


def _task_scope_sha256(tasks: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(_canonical_json(list(tasks))).hexdigest()


def _authority_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(dict(value))).hexdigest()


def _validate_record_against_task(
    record: Mapping[str, Any], task: Mapping[str, Any]
) -> None:
    status = record.get("status")
    expected_keys = PASS_RECORD_KEYS if status == "pass" else REJECT_RECORD_KEYS
    if set(record) != expected_keys:
        raise Human312AuditError(
            f"Human {status!r} record schema drifted for {task['clip_id']}: "
            f"missing={sorted(expected_keys - set(record))}, "
            f"extra={sorted(set(record) - expected_keys)}"
        )
    expected = {
        "audit_version": HUMAN312_AUDIT_VERSION,
        "clip_id": str(task["clip_id"]),
        "rig_id": HUMAN_RIG_ID,
        "source_family": "motionstreamer272",
        "topology_family": "human",
        "split": task.get("split"),
        "source_relpath": str(task["source_relpath"]),
        "source_size_bytes": int(task["file_size_bytes"]),
        "source_mtime_ns": int(task["mtime_ns"]),
        "source_device": int(task["source_device"]),
        "source_inode": int(task["source_inode"]),
        "source_nlink": 1,
        "source_shape": list(task["source_shape"]),
        "source_dtype": str(task["source_dtype"]),
        "rotation_slice": list(task["rotation_slice"]),
        "rotation_shape": list(task["rotation_shape"]),
        "T_src": int(task["T_src"]),
        "J_phys": EXPECTED_HUMAN_JOINT_COUNT,
        "fps_src": float(task["fps_src"]),
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise Human312AuditError(
                f"Human record {field} drifted for {task['clip_id']}: "
                f"{record.get(field)!r} != {value!r}"
            )
    source_sha256 = str(record.get("source_sha256", ""))
    if re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None:
        raise Human312AuditError(
            f"Human record lacks a valid content SHA: {task['clip_id']}"
        )
    if status == "reject":
        reasons = record.get("reason_codes")
        if (
            not isinstance(reasons, list)
            or len(reasons) != 1
            or reasons[0] not in REJECTION_REASON_CODES
            or not isinstance(record.get("error_type"), str)
            or not record["error_type"]
            or not isinstance(record.get("error"), str)
            or not record["error"]
        ):
            raise Human312AuditError(f"malformed Human reject record: {task['clip_id']}")
        return
    if status != "pass" or record.get("reason_codes") != []:
        raise Human312AuditError(f"malformed Human PASS record: {task['clip_id']}")
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != PASS_METRIC_KEYS:
        raise Human312AuditError(f"Human PASS metric schema drifted: {task['clip_id']}")
    if metrics.get("independent_decoder") != HUMAN312_INDEPENDENT_DECODER:
        raise Human312AuditError(f"Human decoder authority drifted: {task['clip_id']}")
    if re.fullmatch(r"[0-9a-f]{64}", str(metrics.get("rotation_payload_sha256", ""))) is None:
        raise Human312AuditError(f"Human rotation payload SHA is invalid: {task['clip_id']}")
    numeric_names = PASS_METRIC_KEYS - {"independent_decoder", "rotation_payload_sha256"}
    numeric: dict[str, float] = {}
    for name in numeric_names:
        value = metrics[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise Human312AuditError(f"Human metric type drifted ({name}): {task['clip_id']}")
        numeric[name] = float(value)
        if not math.isfinite(numeric[name]):
            raise Human312AuditError(f"Human metric is nonfinite ({name}): {task['clip_id']}")
    nonnegative = numeric_names - {
        "rotation_determinant_min",
        "rotation_determinant_max",
        "raw_d6_cross_norm_min",
    }
    if any(numeric[name] < 0.0 for name in nonnegative):
        raise Human312AuditError(f"Human metric is negative: {task['clip_id']}")
    if (
        numeric["raw_d6_first_row_unit_max_abs"] > RAW_D6_UNIT_NORM_MAX_ABS
        or numeric["raw_d6_second_row_unit_max_abs"] > RAW_D6_UNIT_NORM_MAX_ABS
        or numeric["raw_d6_row_dot_max_abs"] > RAW_D6_ROW_DOT_MAX_ABS
        or numeric["raw_d6_cross_norm_min"] < RAW_D6_CROSS_NORM_MIN
        or numeric["independent_positions_max_abs"] > DUAL_POSITION_MAX_ABS
        or numeric["independent_root_translation_max_abs"] > DUAL_POSITION_MAX_ABS
        or numeric["independent_local_rotation_max_abs"] > DUAL_ROTATION_MAX_ABS
        or numeric["independent_global_rotation_max_abs"] > DUAL_ROTATION_MAX_ABS
        or numeric["source_parser_fk_max_norm"] > SOURCE_FK_MAX_NORM
        or numeric["source_parser_fk_mpjpe_norm"] > numeric["source_parser_fk_max_norm"]
        or numeric["fixed_neutral_rigid_edge_max_norm"] > FIXED_RIGID_EDGE_MAX_NORM
        or numeric["rotation_orthogonality_max_abs"] > ROTATION_ORTHOGONALITY_MAX_ABS
        or abs(numeric["rotation_determinant_min"] - 1.0)
        > ROTATION_DETERMINANT_MAX_ABS
        or abs(numeric["rotation_determinant_max"] - 1.0)
        > ROTATION_DETERMINANT_MAX_ABS
    ):
        raise Human312AuditError(f"Human PASS metric threshold failed: {task['clip_id']}")
    dynamic = (
        numeric["root_speed_rms_norm_per_s"]
        + numeric["rotation_speed_rms_rad_per_s"]
        + numeric["pose_excursion_rms_norm"]
    )
    if numeric["dynamic_score"] != dynamic:
        raise Human312AuditError(f"Human dynamic score composition drifted: {task['clip_id']}")


def _validate_records_against_tasks(
    records: Sequence[Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
    *,
    expected_count: int = EXPECTED_HUMAN_CLIP_COUNT,
) -> None:
    """Validate the exhaustive one-to-one record/task closure in canonical order."""
    if len(records) != expected_count or len(tasks) != expected_count:
        raise Human312AuditError(
            f"Human record/task scope is not exhaustive: "
            f"records={len(records)}, tasks={len(tasks)}, expected={expected_count}"
        )
    record_ids = [str(record.get("clip_id")) for record in records]
    task_ids = [str(task.get("clip_id")) for task in tasks]
    if (
        record_ids != task_ids
        or record_ids != sorted(record_ids)
        or len(set(record_ids)) != expected_count
    ):
        raise Human312AuditError("Human record/task identity or canonical order drifted")
    source_inodes: set[tuple[int, int]] = set()
    source_relpaths: set[str] = set()
    for record, task in zip(records, tasks, strict=True):
        _validate_record_against_task(record, task)
        relpath = str(record["source_relpath"])
        inode = (int(record["source_device"]), int(record["source_inode"]))
        if relpath in source_relpaths or inode in source_inodes:
            raise Human312AuditError(
                f"duplicate Human source identity in QA closure: {record['clip_id']}"
            )
        source_relpaths.add(relpath)
        source_inodes.add(inode)


def _reject_unstable_source_records(records: Sequence[Mapping[str, Any]]) -> None:
    """Source mutation is an audit failure, never a filterable data anomaly."""
    unstable = [
        str(record.get("clip_id"))
        for record in records
        if "HUMAN_SOURCE_CHANGED_DURING_AUDIT"
        in list(record.get("reason_codes", []))
        or re.fullmatch(r"[0-9a-f]{64}", str(record.get("source_sha256", "")))
        is None
    ]
    if unstable:
        raise Human312AuditError(
            "Human source authority changed or lacked a content hash; "
            f"audit cannot filter these records: count={len(unstable)}, first={unstable[:10]}"
        )


def _verify_live_source_record(
    record: Mapping[str, Any], source_root: Path
) -> None:
    root = _absolute_no_resolve(source_root)
    source = root / str(record["source_relpath"])
    if source.parent != root or source.resolve(strict=True) != source:
        raise Human312AuditError(f"approved Human source escaped scope: {source}")
    _, digest, observed = _read_stable_source_bytes(source)
    expected = {
        "size_bytes": int(record["source_size_bytes"]),
        "mtime_ns": int(record["source_mtime_ns"]),
        "device": int(record["source_device"]),
        "inode": int(record["source_inode"]),
        "nlink": int(record["source_nlink"]),
    }
    if observed != expected:
        raise Human312AuditError(
            f"approved Human source stat identity drifted: {record['clip_id']}"
        )
    if digest != record["source_sha256"]:
        raise Human312AuditError(f"approved Human source SHA drifted: {record['clip_id']}")


def _verify_live_source_identity(
    record: Mapping[str, Any], source_root: Path
) -> None:
    """Cheap post-worker identity gate; full byte replay follows before publish."""
    root = _absolute_no_resolve(source_root)
    source = root / str(record["source_relpath"])
    if source.parent != root or source.resolve(strict=True) != source:
        raise Human312AuditError(f"Human source escaped scope: {source}")
    observed = _regular_stat(source, label="post-worker Human source")
    expected = {
        "st_size": int(record["source_size_bytes"]),
        "st_mtime_ns": int(record["source_mtime_ns"]),
        "st_dev": int(record["source_device"]),
        "st_ino": int(record["source_inode"]),
        "st_nlink": int(record["source_nlink"]),
    }
    if any(int(getattr(observed, name)) != value for name, value in expected.items()):
        raise Human312AuditError(
            f"post-worker Human source identity drifted: {record['clip_id']}"
        )


def _load_valid_chunk(
    path: Path,
    *,
    authority_sha256: str,
    chunk_index: int,
    tasks: Sequence[Mapping[str, Any]],
    source_root: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]] | None:
    if not path.is_file() or path.is_symlink():
        return None
    try:
        _regular_stat(path, label="resumable Human chunk")
        payload = _load_json(path)
        records = payload["records"]
        status = payload["worker_process_status"]
        if (
            set(payload)
            != {
                "audit_version",
                "authority_sha256",
                "chunk_index",
                "task_scope_sha256",
                "records_sha256",
                "records",
                "worker_process_status",
            }
            or
            payload.get("audit_version") != HUMAN312_AUDIT_VERSION
            or payload.get("authority_sha256") != authority_sha256
            or int(payload.get("chunk_index", -1)) != chunk_index
            or payload.get("task_scope_sha256") != _task_scope_sha256(tasks)
            or not isinstance(records, list)
            or len(records) != len(tasks)
            or payload.get("records_sha256") != _records_sha256(records)
            or [record.get("clip_id") for record in records]
            != [task.get("clip_id") for task in tasks]
        ):
            return None
        normalized = _validate_worker_process_status(status)
        for record, task in zip(records, tasks, strict=True):
            if not isinstance(record, Mapping):
                return None
            _validate_record_against_task(record, task)
            if source_root is not None:
                _verify_live_source_record(record, source_root)
        return [dict(record) for record in records], normalized
    except Exception:  # noqa: BLE001
        return None


def _write_chunk(
    path: Path,
    *,
    authority_sha256: str,
    chunk_index: int,
    tasks: Sequence[Mapping[str, Any]],
    result: Mapping[str, Any],
    source_root: Path,
) -> None:
    records = result.get("records")
    status = result.get("worker_process_status")
    if not isinstance(records, list) or not isinstance(status, Mapping):
        raise Human312AuditError(f"chunk {chunk_index} returned invalid payload")
    if len(records) != len(tasks):
        raise Human312AuditError(f"chunk {chunk_index} record count drifted")
    normalized_status = _validate_worker_process_status(status)
    for record, task in zip(records, tasks, strict=True):
        if not isinstance(record, Mapping):
            raise Human312AuditError(f"chunk {chunk_index} returned a non-record")
        _validate_record_against_task(record, task)
        _verify_live_source_identity(record, source_root)
    _write_json(
        path,
        {
            "audit_version": HUMAN312_AUDIT_VERSION,
            "authority_sha256": authority_sha256,
            "chunk_index": chunk_index,
            "task_scope_sha256": _task_scope_sha256(tasks),
            "records_sha256": _records_sha256(records),
            "records": records,
            "worker_process_status": normalized_status,
        },
    )


def _run_chunks(
    *,
    tasks: Sequence[Mapping[str, Any]],
    rig: Mapping[str, Any],
    fixed: Mapping[str, Any],
    source_root: Path,
    work_root: Path,
    authority_sha256: str,
    workers: int,
    chunk_size: int,
    resumable: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, int]], dict[str, int | str | bool]]:
    chunks = [
        (index, list(tasks[start : start + chunk_size]))
        for index, start in enumerate(range(0, len(tasks), chunk_size))
    ]
    if (
        work_root.is_symlink()
        or not work_root.is_dir()
        or work_root.resolve(strict=True) != work_root
    ):
        raise Human312AuditError(f"invalid Human audit work root: {work_root}")
    chunk_root = _ensure_canonical_directory(
        work_root / ("chunks" if resumable else "deep_chunks"),
        label="Human audit chunk root",
    )
    records_by_index: dict[int, list[dict[str, Any]]] = {}
    fresh_statuses: dict[int, dict[str, int]] = {}
    cached_revalidated_chunk_count = 0
    fresh_spawn_chunk_count = 0
    pending: deque[tuple[int, list[Mapping[str, Any]]]] = deque()
    for index, chunk_tasks in chunks:
        chunk_path = chunk_root / f"chunk_{index:06d}.json"
        cached = (
            _load_valid_chunk(
                chunk_path,
                authority_sha256=authority_sha256,
                chunk_index=index,
                tasks=chunk_tasks,
                source_root=source_root,
            )
            if resumable
            else None
        )
        if cached is None:
            pending.append((index, chunk_tasks))
        else:
            records_by_index[index], _ = cached
            cached_revalidated_chunk_count += 1
    completed = sum(len(value) for value in records_by_index.values())
    if pending:
        context = multiprocessing.get_context("spawn")
        with _single_thread_spawn_environment():
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers,
                mp_context=context,
                initializer=_initialize_worker,
                initargs=(rig, fixed, str(source_root)),
            ) as executor:
                futures: dict[
                    concurrent.futures.Future[dict[str, Any]],
                    tuple[int, list[Mapping[str, Any]]],
                ] = {}
                while pending and len(futures) < workers * 2:
                    index, chunk_tasks = pending.popleft()
                    futures[executor.submit(_audit_chunk, chunk_tasks)] = (
                        index,
                        chunk_tasks,
                    )
                while futures:
                    done, _ = concurrent.futures.wait(
                        futures,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in done:
                        index, chunk_tasks = futures.pop(future)
                        result = future.result()
                        chunk_path = chunk_root / f"chunk_{index:06d}.json"
                        _write_chunk(
                            chunk_path,
                            authority_sha256=authority_sha256,
                            chunk_index=index,
                            tasks=chunk_tasks,
                            result=result,
                            source_root=source_root,
                        )
                        records_by_index[index] = [
                            dict(value) for value in result["records"]
                        ]
                        status = _validate_worker_process_status(
                            result["worker_process_status"]
                        )
                        fresh_statuses[status["pid"]] = status
                        fresh_spawn_chunk_count += 1
                        completed += len(chunk_tasks)
                        if completed % 2000 < len(chunk_tasks) or completed == len(tasks):
                            print(
                                f"[human1-audit] audited {completed}/{len(tasks)}",
                                flush=True,
                            )
                    while pending and len(futures) < workers * 2:
                        index, chunk_tasks = pending.popleft()
                        futures[executor.submit(_audit_chunk, chunk_tasks)] = (
                            index,
                            chunk_tasks,
                        )
    ordered: list[dict[str, Any]] = []
    for index in range(len(chunks)):
        ordered.extend(records_by_index[index])
    if len(ordered) != len(tasks):
        raise Human312AuditError("Human audit record count drifted")
    execution_evidence = _validate_chunk_execution_evidence(
        {
            "executor_mode": "spawn",
            "chunk_count": len(chunks),
            "cached_revalidated_chunk_count": cached_revalidated_chunk_count,
            "fresh_spawn_chunk_count": fresh_spawn_chunk_count,
            "fresh_spawn_chunks_with_process_status": fresh_spawn_chunk_count,
            "cached_worker_process_status_trusted": False,
        },
        expected_chunk_count=len(chunks),
        allow_cache=resumable,
    )
    return (
        ordered,
        [fresh_statuses[pid] for pid in sorted(fresh_statuses)],
        execution_evidence,
    )


def _source_snapshot(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [
        {
            "clip_id": record["clip_id"],
            "source_relpath": record["source_relpath"],
            "source_sha256": record.get("source_sha256"),
            "source_size_bytes": int(record["source_size_bytes"]),
            "source_mtime_ns": int(record["source_mtime_ns"]),
            "source_device": int(record["source_device"]),
            "source_inode": int(record["source_inode"]),
            "source_nlink": int(record["source_nlink"]),
        }
        for record in records
    ]
    return {
        "validated_count": len(rows),
        "source_snapshot_sha256": hashlib.sha256(_canonical_json(rows)).hexdigest(),
    }


def _disk_snapshot_from_tasks(tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [
        {
            "source_relpath": str(task["source_relpath"]),
            "file_size_bytes": int(task["file_size_bytes"]),
            "mtime_ns": int(task["mtime_ns"]),
            "source_device": int(task["source_device"]),
            "source_inode": int(task["source_inode"]),
            "source_nlink": int(task["source_nlink"]),
        }
        for task in sorted(tasks, key=lambda value: str(value["source_relpath"]))
    ]
    return {
        "entry_count": len(rows),
        "snapshot_sha256": hashlib.sha256(_canonical_json(rows)).hexdigest(),
    }


def _revalidate_all_live_sources(
    records: Sequence[Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
    *,
    source_root: Path,
    workers: int,
) -> dict[str, Any]:
    if [record.get("clip_id") for record in records] != [
        task["clip_id"] for task in tasks
    ]:
        raise Human312AuditError("Human final source record/task ordering drifted")
    for record, task in zip(records, tasks, strict=True):
        _validate_record_against_task(record, task)
    before_inventory = _assert_manifest_disk_bijection(source_root, tasks)
    worker_count = min(max(int(workers), 1), 16)
    if worker_count == 1:
        for record in records:
            _verify_live_source_record(record, source_root)
    else:
        queue = deque(records)
        in_flight: set[concurrent.futures.Future[None]] = set()
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            while queue and len(in_flight) < worker_count * 2:
                in_flight.add(
                    executor.submit(_verify_live_source_record, queue.popleft(), source_root)
                )
            while in_flight:
                done, in_flight = concurrent.futures.wait(
                    in_flight, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    future.result()
                while queue and len(in_flight) < worker_count * 2:
                    in_flight.add(
                        executor.submit(
                            _verify_live_source_record, queue.popleft(), source_root
                        )
                    )
    after_inventory = _assert_manifest_disk_bijection(source_root, tasks)
    if after_inventory != before_inventory:
        raise Human312AuditError("Human source inventory changed during byte recheck")
    snapshot = _source_snapshot(records)
    return {
        "status": "pass",
        "validated_count": len(records),
        "hash_workers": worker_count,
        "disk_inventory_snapshot_sha256": after_inventory["snapshot_sha256"],
        "source_snapshot_sha256": snapshot["source_snapshot_sha256"],
        "completed_at_utc": _datetime.datetime.now(_datetime.UTC).isoformat(),
    }


def _select_representative(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    passed = [record for record in records if record.get("status") == "pass"]
    if not passed:
        raise Human312AuditError("Human rig has no accepted clip")
    visual_pool = [record for record in passed if int(record["T_src"]) >= 60]
    if not visual_pool:
        visual_pool = passed
    selected = max(
        visual_pool,
        key=lambda record: (
            float(record["metrics"]["dynamic_score"]),
            int(record["T_src"]),
            str(record["clip_id"]),
        ),
    )
    return {
        "selection_version": "ktjd17-human-dynamic-representative-v1",
        "rig_id": HUMAN_RIG_ID,
        "clip_id": selected["clip_id"],
        "source_relpath": selected["source_relpath"],
        "source_sha256": selected["source_sha256"],
        "T_src": int(selected["T_src"]),
        "dynamic_score": float(selected["metrics"]["dynamic_score"]),
        "eligible_pass_count": len(visual_pool),
        "minimum_visual_frames_preferred": 60,
    }


def _generation_content_evidence(root: Path) -> dict[str, Any]:
    generation_root = _absolute_no_resolve(root)
    output_root = generation_root.parent.parent
    generation_path = generation_root / "generation.json"
    observed = _regular_stat(generation_path, label="Human candidate generation manifest")
    if int(observed.st_mode) & 0o222:
        raise Human312AuditError(f"Human generation manifest is writable: {generation_path}")
    generation = _load_json(generation_path)
    files = _file_manifest(generation_root, require_read_only=True)
    if files != generation.get("files"):
        raise Human312AuditError("Human audit generation file closure drifted")
    core = {
        "audit_version": HUMAN312_AUDIT_VERSION,
        "generation_id": generation_root.name,
        "generation_root": str(generation_root),
        "output_root": str(output_root),
        "generation_json_sha256": _sha256_file(generation_path),
        "generation_json_size_bytes": int(observed.st_size),
        "authority_sha256": str(generation.get("authority_sha256")),
        "files": files,
    }
    return {
        **core,
        "generation_content_sha256": hashlib.sha256(_canonical_json(core)).hexdigest(),
    }


def _task_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "clip_id": str(record["clip_id"]),
        "rig_id": str(record["rig_id"]),
        "topology_family": str(record["topology_family"]),
        "split": record.get("split"),
        "source_relpath": str(record["source_relpath"]),
        "file_size_bytes": int(record["source_size_bytes"]),
        "mtime_ns": int(record["source_mtime_ns"]),
        "source_device": int(record["source_device"]),
        "source_inode": int(record["source_inode"]),
        "source_nlink": int(record["source_nlink"]),
        "T_src": int(record["T_src"]),
        "fps_src": float(record["fps_src"]),
        "source_shape": list(record["source_shape"]),
        "source_dtype": str(record["source_dtype"]),
        "rotation_slice": list(record["rotation_slice"]),
        "rotation_shape": list(record["rotation_shape"]),
    }


def _validate_npz_payload_exact(
    path: Path, expected_payload: Mapping[str, np.ndarray]
) -> None:
    _regular_stat(path, label="published Human fixed skeleton")
    try:
        with np.load(path, allow_pickle=False) as observed:
            if set(observed.files) != set(expected_payload):
                raise Human312AuditError("published Human skeleton key set drifted")
            for name in sorted(expected_payload):
                expected = np.asarray(expected_payload[name])
                actual = np.asarray(observed[name])
                if (
                    actual.dtype != expected.dtype
                    or actual.shape != expected.shape
                    or not np.array_equal(actual, expected)
                ):
                    raise Human312AuditError(
                        f"published Human skeleton payload drifted: {name}"
                    )
    except Human312AuditError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise Human312AuditError(f"cannot validate published Human skeleton: {exc}") from exc


def _current_pinned_inputs(authority: Mapping[str, Any]) -> dict[str, Any]:
    paths = authority.get("pinned_input_paths")
    if not isinstance(paths, Mapping) or set(paths) != {
        "active_cond",
        "legacy_truebones_cond",
        "t04_candidate",
        "neutral_model",
    }:
        raise Human312AuditError("Human pinned-input path schema drifted")
    return {
        name: _stable_file_evidence(Path(str(path)), label=f"Human {name}")
        for name, path in sorted(paths.items())
    }


def _rebuild_and_validate_published_skeleton(
    generation_root: Path,
    *,
    authority: Mapping[str, Any],
    rig: Mapping[str, Any],
) -> HumanFixedRig:
    selection = _load_json(generation_root / "selection/human_representative.json")
    representative_clip_id = str(selection.get("clip_id", ""))
    paths = authority["pinned_input_paths"]
    rebuilt = build_current_btjd_human_fixed_rig(
        rig_record=rig,
        active_cond_path=str(paths["active_cond"]),
        legacy_truebones_cond_path=str(paths["legacy_truebones_cond"]),
        t04_candidate_path=str(paths["t04_candidate"]),
        representative_clip_id=representative_clip_id,
    )
    path = generation_root / "skeletons/HML3D_Human.npz"
    _validate_npz_payload_exact(path, rebuilt.payload)
    skeleton = load_skeleton(path)
    summary = _load_json(generation_root / "summary.json")
    if (
        skeleton.sha256 != summary.get("skeleton_sha256")
        or skeleton.rig_id != HUMAN_RIG_ID
        or skeleton.source_family != "motionstreamer272"
        or skeleton.artifact_status != "t05_prototype_override_pass"
        or len(skeleton.parents) != EXPECTED_HUMAN_JOINT_COUNT
    ):
        raise Human312AuditError("published Human skeleton semantic identity drifted")
    return rebuilt


def _validate_live_recheck_evidence(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Human312AuditError("Human live-source recheck evidence is not an object")
    evidence = dict(value)
    expected = {
        "status",
        "validated_count",
        "hash_workers",
        "disk_inventory_snapshot_sha256",
        "source_snapshot_sha256",
        "completed_at_utc",
    }
    if set(evidence) != expected:
        raise Human312AuditError("Human live-source recheck evidence schema drifted")
    if (
        evidence.get("status") != "pass"
        or int(evidence.get("validated_count", -1)) != EXPECTED_HUMAN_CLIP_COUNT
        or not 1 <= int(evidence.get("hash_workers", 0)) <= 16
        or re.fullmatch(
            r"[0-9a-f]{64}", str(evidence.get("disk_inventory_snapshot_sha256", ""))
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}", str(evidence.get("source_snapshot_sha256", ""))
        )
        is None
        or not isinstance(evidence.get("completed_at_utc"), str)
        or not evidence["completed_at_utc"]
    ):
        raise Human312AuditError("Human live-source recheck evidence is invalid")
    return evidence


def _validate_generation_structure(root: Path) -> dict[str, Any]:
    generation_root = _absolute_no_resolve(root)
    if (
        generation_root.parent.name != HUMAN_AUDIT_GENERATION_DIRECTORY
        or generation_root.is_symlink()
        or not generation_root.is_dir()
        or generation_root.resolve(strict=True) != generation_root
    ):
        raise Human312AuditError(f"invalid Human audit generation: {generation_root}")
    generation = _load_json(generation_root / "generation.json")
    expected_generation_keys = {
        "audit_version",
        "generation_id",
        "created_at_utc",
        "status",
        "authority_sha256",
        "parent_manifest_root",
        "files",
        "prototype_conversion_authorized",
        "full_conversion_authorized",
    }
    if (
        set(generation) != expected_generation_keys
        or generation.get("audit_version") != HUMAN312_AUDIT_VERSION
        or generation.get("generation_id") != generation_root.name
        or generation.get("status") != CANDIDATE_STATUS
        or generation.get("prototype_conversion_authorized") is not False
        or generation.get("full_conversion_authorized") is not False
    ):
        raise Human312AuditError("Human audit generation schema/status drifted")
    content = _generation_content_evidence(generation_root)
    required_files = {
        "authority.json",
        "summary.json",
        "qa/human_source_audit.jsonl",
        "qa/rejected_clips.jsonl",
        "qa/source_snapshot_recheck.json",
        "qa/producer_worker_process_status.json",
        "selection/human_representative.json",
        "skeletons/HML3D_Human.npz",
    }
    if set(content["files"]) != required_files:
        raise Human312AuditError("Human generation semantic file closure drifted")
    authority = _load_json(generation_root / "authority.json")
    expected_authority_keys = {
        "audit_version",
        "claim_boundary",
        "rotation_authority",
        "production_decoder",
        "independent_decoder",
        "parent_manifest_root",
        "parent_manifest",
        "source_root",
        "task_scope_sha256",
        "disk_inventory_snapshot",
        "clip_count",
        "rig_count",
        "pinned_input_paths",
        "pinned_inputs",
        "human_contract_version",
        "code_closure",
        "runtime_fingerprint",
        "workers",
        "chunk_size",
        "thresholds",
        "anomaly_policy",
        "authority_sha256",
    }
    authority_sha = str(authority.get("authority_sha256", ""))
    authority_core = dict(authority)
    authority_core.pop("authority_sha256", None)
    expected_thresholds = {
        "source_fk_max_norm": SOURCE_FK_MAX_NORM,
        "dual_position_max_abs": DUAL_POSITION_MAX_ABS,
        "dual_rotation_max_abs": DUAL_ROTATION_MAX_ABS,
        "rotation_orthogonality_max_abs": ROTATION_ORTHOGONALITY_MAX_ABS,
        "rotation_determinant_max_abs": ROTATION_DETERMINANT_MAX_ABS,
        "fixed_rigid_edge_max_norm": FIXED_RIGID_EDGE_MAX_NORM,
        "raw_d6_unit_norm_max_abs": RAW_D6_UNIT_NORM_MAX_ABS,
        "raw_d6_row_dot_max_abs": RAW_D6_ROW_DOT_MAX_ABS,
        "raw_d6_cross_norm_min": RAW_D6_CROSS_NORM_MIN,
    }
    if (
        set(authority) != expected_authority_keys
        or re.fullmatch(r"[0-9a-f]{64}", authority_sha) is None
        or _authority_sha256(authority_core) != authority_sha
        or generation.get("authority_sha256") != authority_sha
        or authority.get("audit_version") != HUMAN312_AUDIT_VERSION
        or authority.get("claim_boundary") != HUMAN_CLAIM_BOUNDARY
        or authority.get("rotation_authority") != HUMAN_ROTATION_AUTHORITY
        or authority.get("production_decoder") != HUMAN_PRODUCTION_DECODER
        or authority.get("independent_decoder") != HUMAN312_INDEPENDENT_DECODER
        or authority.get("human_contract_version") != HUMAN_CONTRACT_VERSION
        or authority.get("anomaly_policy") != HUMAN_ANOMALY_POLICY
        or authority.get("thresholds") != expected_thresholds
        or int(authority.get("clip_count", -1)) != EXPECTED_HUMAN_CLIP_COUNT
        or int(authority.get("rig_count", -1)) != EXPECTED_HUMAN_RIG_COUNT
        or not 1 <= int(authority.get("workers", 0)) <= EXPECTED_HUMAN_CLIP_COUNT
        or not 1 <= int(authority.get("chunk_size", 0)) <= EXPECTED_HUMAN_CLIP_COUNT
        or not isinstance(authority.get("code_closure"), Mapping)
        or not authority["code_closure"]
        or not isinstance(authority.get("runtime_fingerprint"), Mapping)
    ):
        raise Human312AuditError("Human audit authority hash/scope drifted")
    for relpath, evidence in authority["code_closure"].items():
        if (
            not isinstance(relpath, str)
            or Path(relpath).is_absolute()
            or ".." in Path(relpath).parts
            or not isinstance(evidence, Mapping)
            or set(evidence) != {"sha256", "size_bytes"}
            or re.fullmatch(r"[0-9a-f]{64}", str(evidence.get("sha256", "")))
            is None
            or int(evidence.get("size_bytes", -1)) <= 0
        ):
            raise Human312AuditError("Human producer code-closure schema drifted")
    parent_root = _absolute_no_resolve(str(authority.get("parent_manifest_root", "")))
    parent = _validate_parent_manifest(parent_root)
    if (
        parent != authority.get("parent_manifest")
        or generation.get("parent_manifest_root") != str(parent_root)
    ):
        raise Human312AuditError("Human live parent authority drifted")
    source_root = _absolute_no_resolve(str(authority.get("source_root", "")))
    rig, tasks = _load_human_scope(parent_root, source_root)
    if _task_scope_sha256(tasks) != authority.get("task_scope_sha256"):
        raise Human312AuditError("Human source task closure drifted")
    disk_snapshot = _disk_snapshot_from_tasks(tasks)
    if (
        disk_snapshot != authority.get("disk_inventory_snapshot")
        or _assert_manifest_disk_bijection(source_root, tasks) != disk_snapshot
    ):
        raise Human312AuditError("Human live disk inventory drifted")
    records = _load_jsonl(generation_root / "qa/human_source_audit.jsonl")
    _validate_records_against_tasks(records, tasks)
    _reject_unstable_source_records(records)
    rejected = [record for record in records if record["status"] == "reject"]
    if _load_jsonl(generation_root / "qa/rejected_clips.jsonl") != rejected:
        raise Human312AuditError("Human rejected-clip manifest drifted")
    accepted = [record for record in records if record["status"] == "pass"]
    if not accepted:
        raise Human312AuditError("Human rig coverage failed: no accepted clips")
    selection = _load_json(generation_root / "selection/human_representative.json")
    if selection != _select_representative(records):
        raise Human312AuditError("Human representative selection drifted")
    summary = _load_json(generation_root / "summary.json")
    expected_summary_keys = {
        "audit_version",
        "generation_id",
        "status",
        "source_audit_status",
        "rig_count",
        "clip_count",
        "accepted_clip_count",
        "rejected_clip_count",
        "status_counts",
        "rejection_reason_counts",
        "rig_coverage_status",
        "T_src_min",
        "T_src_max",
        "source_snapshot_sha256",
        "independent_position_max_abs",
        "independent_rotation_max_abs",
        "source_parser_fk_max_norm",
        "fixed_neutral_rigid_edge_max_norm",
        "representative",
        "skeleton_sha256",
        "producer_worker_process_count",
        "producer_worker_threads_max",
        "producer_worker_process_statuses",
        "source_snapshot_recheck",
        "authority_sha256",
        "claim_boundary",
        "prototype_conversion_authorized",
        "full_conversion_authorized",
    }
    status_counts = dict(sorted(Counter(str(record["status"]) for record in records).items()))
    reason_counts = dict(
        sorted(Counter(reason for record in rejected for reason in record["reason_codes"]).items())
    )
    snapshot = _source_snapshot(records)
    metrics = [record["metrics"] for record in accepted]
    expected_metrics = {
        "T_src_min": min(int(record["T_src"]) for record in records),
        "T_src_max": max(int(record["T_src"]) for record in records),
        "independent_position_max_abs": max(
            float(value["independent_positions_max_abs"]) for value in metrics
        ),
        "independent_rotation_max_abs": max(
            max(
                float(value["independent_local_rotation_max_abs"]),
                float(value["independent_global_rotation_max_abs"]),
            )
            for value in metrics
        ),
        "source_parser_fk_max_norm": max(
            float(value["source_parser_fk_max_norm"]) for value in metrics
        ),
        "fixed_neutral_rigid_edge_max_norm": max(
            float(value["fixed_neutral_rigid_edge_max_norm"]) for value in metrics
        ),
    }
    if (
        set(summary) != expected_summary_keys
        or summary.get("audit_version") != HUMAN312_AUDIT_VERSION
        or summary.get("generation_id") != generation_root.name
        or summary.get("status") != CANDIDATE_STATUS
        or summary.get("source_audit_status")
        != ("pass" if not rejected else "pass_with_rejections")
        or int(summary.get("clip_count", -1)) != EXPECTED_HUMAN_CLIP_COUNT
        or int(summary.get("rig_count", -1)) != EXPECTED_HUMAN_RIG_COUNT
        or int(summary.get("accepted_clip_count", -1)) != len(accepted)
        or int(summary.get("rejected_clip_count", -1)) != len(rejected)
        or summary.get("status_counts") != status_counts
        or summary.get("rejection_reason_counts") != reason_counts
        or summary.get("rig_coverage_status") != "pass"
        or summary.get("source_snapshot_sha256") != snapshot["source_snapshot_sha256"]
        or summary.get("representative") != selection
        or summary.get("authority_sha256") != authority_sha
        or summary.get("claim_boundary") != HUMAN_CLAIM_BOUNDARY
        or re.fullmatch(r"[0-9a-f]{64}", str(summary.get("skeleton_sha256", "")))
        is None
        or summary.get("prototype_conversion_authorized") is not False
        or summary.get("full_conversion_authorized") is not False
    ):
        raise Human312AuditError("Human summary/coverage binding drifted")
    for field, expected_value in expected_metrics.items():
        if summary.get(field) != expected_value:
            raise Human312AuditError(f"Human summary metric drifted: {field}")
    recheck = _validate_live_recheck_evidence(
        _load_json(generation_root / "qa/source_snapshot_recheck.json")
    )
    if (
        summary.get("source_snapshot_recheck") != recheck
        or recheck["source_snapshot_sha256"] != snapshot["source_snapshot_sha256"]
        or recheck["disk_inventory_snapshot_sha256"] != disk_snapshot["snapshot_sha256"]
    ):
        raise Human312AuditError("Human published source recheck binding drifted")
    producer = _load_json(generation_root / "qa/producer_worker_process_status.json")
    expected_producer_keys = {
        "audit_version",
        "status",
        "executor_mode",
        "authority_sha256",
        "task_scope_sha256",
        "record_count",
        "records_sha256",
        "chunk_count",
        "cached_revalidated_chunk_count",
        "fresh_spawn_chunk_count",
        "fresh_spawn_chunks_with_process_status",
        "cached_worker_process_status_trusted",
        "worker_process_statuses",
        "worker_process_statuses_sha256",
    }
    statuses = [
        _validate_worker_process_status(value)
        for value in producer.get("worker_process_statuses", [])
    ] if isinstance(producer, Mapping) else []
    expected_chunks = math.ceil(EXPECTED_HUMAN_CLIP_COUNT / int(authority["chunk_size"]))
    producer_execution = (
        _validate_chunk_execution_evidence(
            {
                key: producer[key]
                for key in (
                    "executor_mode",
                    "chunk_count",
                    "cached_revalidated_chunk_count",
                    "fresh_spawn_chunk_count",
                    "fresh_spawn_chunks_with_process_status",
                    "cached_worker_process_status_trusted",
                )
            },
            expected_chunk_count=expected_chunks,
            allow_cache=True,
        )
        if isinstance(producer, Mapping) and set(producer) == expected_producer_keys
        else None
    )
    fresh_chunks = (
        int(producer_execution["fresh_spawn_chunk_count"])
        if producer_execution is not None
        else -1
    )
    if (
        not isinstance(producer, Mapping)
        or set(producer) != expected_producer_keys
        or producer.get("audit_version") != HUMAN312_AUDIT_VERSION
        or producer.get("status") != "pass"
        or producer.get("executor_mode") != "spawn"
        or producer.get("authority_sha256") != authority_sha
        or producer.get("task_scope_sha256") != authority.get("task_scope_sha256")
        or int(producer.get("record_count", -1)) != EXPECTED_HUMAN_CLIP_COUNT
        or producer.get("records_sha256") != _records_sha256(records)
        or producer_execution is None
        or (fresh_chunks > 0 and not statuses)
        or (fresh_chunks == 0 and bool(statuses))
        or len(statuses) > int(authority["workers"])
        or statuses != sorted(statuses, key=lambda value: value["pid"])
        or len({value["pid"] for value in statuses}) != len(statuses)
        or any(value["pid"] == value["ppid"] for value in statuses)
        or producer.get("worker_process_statuses_sha256")
        != _worker_statuses_sha256(statuses)
    ):
        raise Human312AuditError("Human producer worker evidence drifted")
    if (
        summary.get("producer_worker_process_statuses") != statuses
        or int(summary.get("producer_worker_process_count", -1)) != len(statuses)
        or int(summary.get("producer_worker_threads_max", -1))
        != max((int(value["threads"]) for value in statuses), default=0)
    ):
        raise Human312AuditError("Human summary producer evidence drifted")
    if _current_pinned_inputs(authority) != authority.get("pinned_inputs"):
        raise Human312AuditError("Human fixed-rig input authority drifted")
    _rebuild_and_validate_published_skeleton(
        generation_root, authority=authority, rig=rig
    )
    return generation


def _approval_path(output_root: Path, content_sha256: str) -> Path:
    return output_root / HUMAN_AUDIT_APPROVAL_DIRECTORY / f"{content_sha256}.json"


def _create_approval(
    *,
    output_root: Path,
    generation_root: Path,
    candidate_proof: Mapping[str, Any],
    post_deep_live_recheck: Mapping[str, Any],
    deep_records_sha256: str,
    deep_worker_statuses: Sequence[Mapping[str, Any]],
    cleanup_witness: _ApprovalCleanupWitness | None = None,
) -> tuple[Path, dict[str, Any], bool]:
    content = dict(candidate_proof["content_evidence"])
    authority = _load_json(generation_root / "authority.json")
    summary = _load_json(generation_root / "summary.json")
    normalized_statuses = [_validate_worker_process_status(value) for value in deep_worker_statuses]
    if not normalized_statuses:
        raise Human312AuditError("Human deep replay lacks spawned-worker evidence")
    normalized_statuses.sort(key=lambda value: value["pid"])
    controller_pid = int(os.getpid())
    if (
        len({value["pid"] for value in normalized_statuses})
        != len(normalized_statuses)
        or any(value["ppid"] != controller_pid for value in normalized_statuses)
    ):
        raise Human312AuditError(
            "Human deep replay worker evidence is not bound to this validator process"
        )
    live_recheck = _validate_live_recheck_evidence(post_deep_live_recheck)
    expected_deep_chunks = math.ceil(
        EXPECTED_HUMAN_CLIP_COUNT / int(authority["chunk_size"])
    )
    deep_execution = _validate_chunk_execution_evidence(
        candidate_proof.get("deep_chunk_process_evidence"),
        expected_chunk_count=expected_deep_chunks,
        allow_cache=False,
    )
    approval = {
        "approval_version": HUMAN312_APPROVAL_VERSION,
        "audit_version": HUMAN312_AUDIT_VERSION,
        "status": "pass",
        "generation_id": generation_root.name,
        "generation_root": str(generation_root),
        "output_root": str(output_root),
        "generation_relpath": generation_root.relative_to(output_root).as_posix(),
        "generation_content_sha256": content["generation_content_sha256"],
        "generation_json_sha256": content["generation_json_sha256"],
        "authority_sha256": authority["authority_sha256"],
        "source_snapshot_sha256": summary["source_snapshot_sha256"],
        "accepted_clip_count": int(summary["accepted_clip_count"]),
        "rejected_clip_count": int(summary["rejected_clip_count"]),
        "deep_records_sha256": deep_records_sha256,
        "deep_validated_count": int(live_recheck["validated_count"]),
        "deep_validator_controller_pid": controller_pid,
        "deep_chunk_process_evidence": deep_execution,
        "deep_chunk_process_evidence_sha256": hashlib.sha256(
            _canonical_json(deep_execution)
        ).hexdigest(),
        "deep_worker_process_statuses": normalized_statuses,
        "deep_worker_process_statuses_sha256": _worker_statuses_sha256(normalized_statuses),
        "producer_code_sha256": hashlib.sha256(
            _canonical_json(authority["code_closure"])
        ).hexdigest(),
        "runtime_fingerprint_sha256": hashlib.sha256(
            _canonical_json(authority["runtime_fingerprint"])
        ).hexdigest(),
        "parent_manifest_sha256": hashlib.sha256(
            _canonical_json(authority["parent_manifest"])
        ).hexdigest(),
        "pinned_inputs_sha256": hashlib.sha256(
            _canonical_json(authority["pinned_inputs"])
        ).hexdigest(),
        "post_deep_live_source_recheck": live_recheck,
        "post_deep_live_source_recheck_sha256": hashlib.sha256(
            _canonical_json(live_recheck)
        ).hexdigest(),
        "prototype_conversion_authorized": True,
        "full_conversion_authorized": False,
        "approved_at_utc": _datetime.datetime.now(_datetime.UTC).isoformat(),
    }
    root = _ensure_canonical_directory(
        output_root / HUMAN_AUDIT_APPROVAL_DIRECTORY,
        label="Human approval root",
    )
    path = _approval_path(output_root, str(content["generation_content_sha256"]))
    if cleanup_witness is not None:
        cleanup_witness.path = path
        cleanup_witness.owned_by_run = False
    created = False
    if os.path.lexists(path):
        existing = _regular_stat(path, label="existing Human source approval")
        if int(existing.st_mode) & 0o222 or _load_json(path) != approval:
            raise Human312AuditError(f"Human approval collision/drift: {path}")
    else:
        # Publish caller-visible ownership before materializing the PASS path.
        # This closes the asynchronous-exception window between our return and
        # the caller's tuple assignment: cleanup can still identify the path.
        if cleanup_witness is not None:
            cleanup_witness.owned_by_run = True
        try:
            _write_json(path, approval)
            os.chmod(path, int(path.stat().st_mode) & ~0o222)
            _fsync_directory(root)
            created = True
        except BaseException as exc:
            rejected: Path | None = None
            cleanup_error: BaseException | None = None
            if os.path.lexists(path):
                rejected = root / (
                    f".rejected-incomplete-{path.stem}-{uuid.uuid4().hex[:8]}.json"
                )
                try:
                    if not path.is_symlink():
                        os.chmod(path, int(path.lstat().st_mode) & ~0o222)
                    os.replace(path, rejected)
                    try:
                        _fsync_directory(root)
                    except BaseException as fsync_exc:
                        cleanup_error = fsync_exc
                except BaseException as rename_exc:
                    cleanup_error = rename_exc
                    try:
                        if os.path.lexists(path):
                            path.unlink()
                    except BaseException as unlink_exc:
                        cleanup_error = Human312AuditError(
                            f"approval quarantine failed ({rename_exc}); "
                            f"fallback unlink failed ({unlink_exc})"
                        )
            if os.path.lexists(path):
                raise Human312AuditError(
                    f"Human approval creation failed and left a PASS-named path: {path}"
                ) from (cleanup_error or exc)
            detail = (
                f"; quarantine durability warning: {cleanup_error}"
                if cleanup_error is not None
                else ""
            )
            raise Human312AuditError(
                "Human approval creation failed; any materialized approval was "
                f"quarantined as {rejected}{detail}"
            ) from exc
    return path, approval, created


def _validate_approval(
    generation_root: Path, output_root: Path
) -> tuple[Path, dict[str, Any]]:
    generation_root = _absolute_no_resolve(generation_root)
    output_root = _absolute_no_resolve(output_root)
    if generation_root.parent.parent != output_root:
        raise Human312AuditError("Human generation/output namespace binding drifted")
    _validate_generation_structure(generation_root)
    content = _generation_content_evidence(generation_root)
    approval_root = output_root / HUMAN_AUDIT_APPROVAL_DIRECTORY
    if (
        approval_root.is_symlink()
        or not approval_root.is_dir()
        or approval_root.resolve(strict=True) != approval_root
    ):
        raise Human312AuditError(f"invalid Human approval root: {approval_root}")
    path = _approval_path(output_root, str(content["generation_content_sha256"]))
    observed = _regular_stat(path, label="Human source approval")
    if int(observed.st_mode) & 0o222:
        raise Human312AuditError("Human source approval is writable")
    approval = _load_json(path)
    expected_keys = {
        "approval_version",
        "audit_version",
        "status",
        "generation_id",
        "generation_root",
        "output_root",
        "generation_relpath",
        "generation_content_sha256",
        "generation_json_sha256",
        "authority_sha256",
        "source_snapshot_sha256",
        "accepted_clip_count",
        "rejected_clip_count",
        "deep_records_sha256",
        "deep_validated_count",
        "deep_validator_controller_pid",
        "deep_chunk_process_evidence",
        "deep_chunk_process_evidence_sha256",
        "deep_worker_process_statuses",
        "deep_worker_process_statuses_sha256",
        "producer_code_sha256",
        "runtime_fingerprint_sha256",
        "parent_manifest_sha256",
        "pinned_inputs_sha256",
        "post_deep_live_source_recheck",
        "post_deep_live_source_recheck_sha256",
        "prototype_conversion_authorized",
        "full_conversion_authorized",
        "approved_at_utc",
    }
    if set(approval) != expected_keys:
        raise Human312AuditError("Human approval schema drifted")
    authority = _load_json(generation_root / "authority.json")
    summary = _load_json(generation_root / "summary.json")
    records = _load_jsonl(generation_root / "qa/human_source_audit.jsonl")
    deep_statuses = [
        _validate_worker_process_status(value)
        for value in approval.get("deep_worker_process_statuses", [])
    ]
    deep_statuses.sort(key=lambda value: value["pid"])
    deep_controller_pid = int(approval.get("deep_validator_controller_pid", -1))
    expected_deep_chunks = math.ceil(
        EXPECTED_HUMAN_CLIP_COUNT / int(authority["chunk_size"])
    )
    deep_execution = approval.get("deep_chunk_process_evidence")
    expected_deep_execution = _validate_chunk_execution_evidence(
        {
            "executor_mode": "spawn",
            "chunk_count": expected_deep_chunks,
            "cached_revalidated_chunk_count": 0,
            "fresh_spawn_chunk_count": expected_deep_chunks,
            "fresh_spawn_chunks_with_process_status": expected_deep_chunks,
            "cached_worker_process_status_trusted": False,
        },
        expected_chunk_count=expected_deep_chunks,
        allow_cache=False,
    )
    relative = Path(str(approval.get("generation_relpath", "")))
    live_recheck = _validate_live_recheck_evidence(
        approval.get("post_deep_live_source_recheck")
    )
    published_recheck = _validate_live_recheck_evidence(
        _load_json(generation_root / "qa/source_snapshot_recheck.json")
    )
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or _absolute_no_resolve(output_root / relative) != generation_root
        or
        approval.get("approval_version") != HUMAN312_APPROVAL_VERSION
        or approval.get("audit_version") != HUMAN312_AUDIT_VERSION
        or approval.get("status") != "pass"
        or approval.get("generation_id") != generation_root.name
        or approval.get("generation_root") != str(generation_root)
        or approval.get("output_root") != str(output_root)
        or content.get("generation_root") != str(generation_root)
        or content.get("output_root") != str(output_root)
        or approval.get("generation_relpath")
        != generation_root.relative_to(output_root).as_posix()
        or approval.get("generation_content_sha256")
        != content["generation_content_sha256"]
        or approval.get("generation_json_sha256") != content["generation_json_sha256"]
        or approval.get("authority_sha256") != authority.get("authority_sha256")
        or approval.get("source_snapshot_sha256")
        != summary.get("source_snapshot_sha256")
        or int(approval.get("accepted_clip_count", -1))
        != int(summary.get("accepted_clip_count", -2))
        or int(approval.get("rejected_clip_count", -1))
        != int(summary.get("rejected_clip_count", -2))
        or approval.get("deep_records_sha256") != _records_sha256(records)
        or int(approval.get("deep_validated_count", -1))
        != EXPECTED_HUMAN_CLIP_COUNT
        or deep_controller_pid <= 0
        or deep_execution != expected_deep_execution
        or approval.get("deep_chunk_process_evidence_sha256")
        != hashlib.sha256(_canonical_json(expected_deep_execution)).hexdigest()
        or not deep_statuses
        or len(deep_statuses) > int(authority["workers"])
        or len({value["pid"] for value in deep_statuses}) != len(deep_statuses)
        or any(value["ppid"] != deep_controller_pid for value in deep_statuses)
        or approval.get("deep_worker_process_statuses_sha256")
        != _worker_statuses_sha256(deep_statuses)
        or approval.get("producer_code_sha256")
        != hashlib.sha256(_canonical_json(authority.get("code_closure"))).hexdigest()
        or approval.get("runtime_fingerprint_sha256")
        != hashlib.sha256(_canonical_json(authority.get("runtime_fingerprint"))).hexdigest()
        or approval.get("parent_manifest_sha256")
        != hashlib.sha256(_canonical_json(authority.get("parent_manifest"))).hexdigest()
        or approval.get("pinned_inputs_sha256")
        != hashlib.sha256(_canonical_json(authority.get("pinned_inputs"))).hexdigest()
        or approval.get("post_deep_live_source_recheck_sha256")
        != hashlib.sha256(_canonical_json(live_recheck)).hexdigest()
        or live_recheck.get("source_snapshot_sha256")
        != summary.get("source_snapshot_sha256")
        or live_recheck.get("disk_inventory_snapshot_sha256")
        != published_recheck.get("disk_inventory_snapshot_sha256")
        or int(live_recheck.get("validated_count", -1))
        != EXPECTED_HUMAN_CLIP_COUNT
        or approval.get("prototype_conversion_authorized") is not True
        or approval.get("full_conversion_authorized") is not False
        or not isinstance(approval.get("approved_at_utc"), str)
        or not approval["approved_at_utc"]
    ):
        raise Human312AuditError("Human approval/content binding drifted")
    return path, approval


def _recheck_generation_live_sources(
    generation_root: Path, *, workers: int
) -> dict[str, Any]:
    authority = _load_json(generation_root / "authority.json")
    parent_root = _absolute_no_resolve(str(authority["parent_manifest_root"]))
    source_root = _absolute_no_resolve(str(authority["source_root"]))
    _, tasks = _load_human_scope(parent_root, source_root)
    records = _load_jsonl(generation_root / "qa/human_source_audit.jsonl")
    current = _validate_live_recheck_evidence(
        _revalidate_all_live_sources(
            records, tasks, source_root=source_root, workers=int(workers)
        )
    )
    published = _validate_live_recheck_evidence(
        _load_json(generation_root / "qa/source_snapshot_recheck.json")
    )
    for field in (
        "status",
        "validated_count",
        "disk_inventory_snapshot_sha256",
        "source_snapshot_sha256",
    ):
        if current[field] != published[field]:
            raise Human312AuditError(f"Human live source scope drifted: {field}")
    return current


def _verify_approved_live_sources(
    generation_root: Path,
    approval: Mapping[str, Any],
    *,
    workers: int,
) -> dict[str, Any]:
    approved = _validate_live_recheck_evidence(
        approval.get("post_deep_live_source_recheck")
    )
    if approval.get("post_deep_live_source_recheck_sha256") != hashlib.sha256(
        _canonical_json(approved)
    ).hexdigest():
        raise Human312AuditError("approved Human live-source digest drifted")
    current = _recheck_generation_live_sources(generation_root, workers=workers)
    for field in (
        "status",
        "validated_count",
        "disk_inventory_snapshot_sha256",
        "source_snapshot_sha256",
    ):
        if current[field] != approved[field]:
            raise Human312AuditError(
                f"approved Human live source scope is no longer current: {field}"
            )
    return current


def _validate_candidate(root: Path, *, workers: int) -> dict[str, Any]:
    generation_root = _absolute_no_resolve(root)
    generation = _validate_generation_structure(generation_root)
    initial_content = _generation_content_evidence(generation_root)
    authority = _load_json(generation_root / "authority.json")
    if _code_closure() != authority.get("code_closure"):
        raise Human312AuditError("current Human producer code closure drifted")
    if _runtime_fingerprint() != authority.get("runtime_fingerprint"):
        raise Human312AuditError("current Human numerical runtime drifted")
    if _current_pinned_inputs(authority) != authority.get("pinned_inputs"):
        raise Human312AuditError("current Human fixed-rig inputs drifted")
    parent_root = _absolute_no_resolve(str(authority["parent_manifest_root"]))
    if _validate_parent_manifest(parent_root) != authority.get("parent_manifest"):
        raise Human312AuditError("current Human parent manifest drifted")
    source_root = _absolute_no_resolve(str(authority["source_root"]))
    rig, tasks = _load_human_scope(parent_root, source_root)
    rebuilt = _rebuild_and_validate_published_skeleton(
        generation_root, authority=authority, rig=rig
    )
    fixed = {
        "parents": rebuilt.parents,
        "P_rest_global": rebuilt.P_rest_global,
        "offsets": rebuilt.offset_parent_local,
        "s_rig": rebuilt.s_rig,
    }
    records = _load_jsonl(generation_root / "qa/human_source_audit.jsonl")
    deep_records, deep_statuses, deep_execution = _run_chunks(
        tasks=tasks,
        rig=rig,
        fixed=fixed,
        source_root=source_root,
        work_root=(
            generation_root.parent.parent
            / HUMAN_AUDIT_WORK_DIRECTORY
            / str(authority["authority_sha256"])[:20]
        ),
        authority_sha256=str(authority["authority_sha256"]),
        workers=int(workers),
        chunk_size=int(authority["chunk_size"]),
        resumable=False,
    )
    difference = _first_record_difference(records, deep_records)
    if difference:
        raise Human312AuditError(f"post-publish Human deep replay drifted: {difference}")
    normalized_statuses = [_validate_worker_process_status(value) for value in deep_statuses]
    expected_chunks = math.ceil(len(tasks) / int(authority["chunk_size"]))
    deep_execution = _validate_chunk_execution_evidence(
        deep_execution,
        expected_chunk_count=expected_chunks,
        allow_cache=False,
    )
    if not normalized_statuses:
        raise Human312AuditError("Human deep validator produced no child-process evidence")
    deep_recheck = _revalidate_all_live_sources(
        records, tasks, source_root=source_root, workers=int(workers)
    )
    published = _validate_live_recheck_evidence(
        _load_json(generation_root / "qa/source_snapshot_recheck.json")
    )
    for field in (
        "status",
        "validated_count",
        "disk_inventory_snapshot_sha256",
        "source_snapshot_sha256",
    ):
        if deep_recheck[field] != published[field]:
            raise Human312AuditError(f"Human deep source recheck drifted: {field}")
    if (
        _code_closure() != authority.get("code_closure")
        or _runtime_fingerprint() != authority.get("runtime_fingerprint")
        or _current_pinned_inputs(authority) != authority.get("pinned_inputs")
        or _validate_parent_manifest(parent_root) != authority.get("parent_manifest")
    ):
        raise Human312AuditError("Human authority changed during deep validation")
    final_generation = _validate_generation_structure(generation_root)
    final_content = _generation_content_evidence(generation_root)
    if final_generation != generation or final_content != initial_content:
        raise Human312AuditError("Human candidate changed during deep validation")
    return {
        "generation": generation,
        "content_evidence": final_content,
        "deep_source_recheck": deep_recheck,
        "deep_records_sha256": _records_sha256(deep_records),
        "deep_worker_process_statuses": normalized_statuses,
        "deep_chunk_process_evidence": deep_execution,
    }


def validate_active_human_audit(
    output_root: str | Path, *, rehash_sources: bool = True
) -> dict[str, Any]:
    """Validate from the approval link, the single atomic authorization pointer."""
    output = _absolute_no_resolve(output_root)
    if output.is_symlink() or not output.is_dir() or output.resolve(strict=True) != output:
        raise Human312AuditError(f"invalid Human audit output root: {output}")
    generation_link = output / HUMAN_AUDIT_LINK_NAME
    approval_link = output / HUMAN_AUDIT_APPROVAL_LINK_NAME
    if not generation_link.is_symlink() or not approval_link.is_symlink():
        raise Human312AuditError("active Human generation/approval links are incomplete")
    approval_path = _read_relative_symlink_target(
        approval_link, label="active Human approval link"
    )
    if approval_path.parent != output / HUMAN_AUDIT_APPROVAL_DIRECTORY:
        raise Human312AuditError("active Human approval escaped its namespace")
    preliminary = _load_json(approval_path)
    relative = Path(str(preliminary.get("generation_relpath", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise Human312AuditError("active Human approval has an unsafe generation path")
    generation_root = _absolute_no_resolve(output / relative)
    expected_approval_path, approval = _validate_approval(generation_root, output)
    if approval_path != expected_approval_path:
        raise Human312AuditError("active Human approval path/content binding drifted")
    compatibility_generation = _read_relative_symlink_target(
        generation_link, label="active Human compatibility generation link"
    )
    if compatibility_generation != generation_root:
        raise Human312AuditError(
            "active Human compatibility link is mixed with the atomic approval authority"
        )
    authority = _load_json(generation_root / "authority.json")
    if _code_closure() != authority.get("code_closure"):
        raise Human312AuditError("current Human audit code closure drifted")
    if _runtime_fingerprint() != authority.get("runtime_fingerprint"):
        raise Human312AuditError("current Human audit numerical runtime drifted")
    if _current_pinned_inputs(authority) != authority.get("pinned_inputs"):
        raise Human312AuditError("current Human fixed-rig inputs drifted")
    if _validate_parent_manifest(Path(str(authority["parent_manifest_root"]))) != authority.get(
        "parent_manifest"
    ):
        raise Human312AuditError("current Human parent manifest drifted")
    active_live_recheck = None
    if rehash_sources:
        active_live_recheck = _verify_approved_live_sources(
            generation_root, approval, workers=16
        )
    return {
        **approval,
        "generation_root": str(generation_root),
        "approval_path": str(approval_path),
        "live_sources_rehashed": bool(rehash_sources),
        "active_live_source_recheck": active_live_recheck,
    }


def load_active_human_clip_binding(
    output_root: str | Path,
    clip_id: str,
    *,
    rehash_all_sources: bool = False,
) -> dict[str, Any]:
    """Resolve one accepted clip and its published skeleton from paired approval links."""
    active = validate_active_human_audit(
        output_root, rehash_sources=rehash_all_sources
    )
    generation_root = Path(str(active["generation_root"]))
    authority = _load_json(generation_root / "authority.json")
    records = _load_jsonl(generation_root / "qa/human_source_audit.jsonl")
    matches = [record for record in records if record.get("clip_id") == clip_id]
    if len(matches) != 1 or matches[0].get("status") != "pass":
        raise Human312AuditError(
            f"Human clip is absent or rejected by the active approval: {clip_id}"
        )
    record = matches[0]
    source_root = _absolute_no_resolve(str(authority["source_root"]))
    _verify_live_source_record(record, source_root)
    skeleton_path = generation_root / "skeletons/HML3D_Human.npz"
    skeleton = load_skeleton(skeleton_path)
    return {
        "audit_version": HUMAN312_AUDIT_VERSION,
        "approval_version": HUMAN312_APPROVAL_VERSION,
        "generation_id": generation_root.name,
        "generation_content_sha256": active["generation_content_sha256"],
        "approval_path": active["approval_path"],
        "source_root": str(source_root),
        "source_path": str(source_root / str(record["source_relpath"])),
        "source_record": record,
        "skeleton_path": str(skeleton_path),
        "skeleton_sha256": skeleton.sha256,
    }


def _snapshot_symlink(link: Path) -> str | None:
    if not os.path.lexists(link):
        return None
    if not link.is_symlink():
        raise Human312AuditError(f"active path is not a symlink: {link}")
    return os.readlink(link)


def _restore_symlink(link: Path, previous: str | None) -> None:
    if previous is None:
        if os.path.lexists(link):
            if not link.is_symlink():
                raise Human312AuditError(f"cannot remove non-symlink active path: {link}")
            link.unlink()
            _fsync_directory(link.parent)
        return
    temporary = link.parent / f".{link.name}.{uuid.uuid4().hex}.rollback"
    os.symlink(previous, temporary)
    os.replace(temporary, link)
    _fsync_directory(link.parent)


def _activate_approved_generation(
    *,
    output_root: Path,
    generation_root: Path,
    approval_path: Path,
    workers: int = 16,
) -> dict[str, Any]:
    generation_link = output_root / HUMAN_AUDIT_LINK_NAME
    approval_link = output_root / HUMAN_AUDIT_APPROVAL_LINK_NAME
    previous_generation = _snapshot_symlink(generation_link)
    previous_approval = _snapshot_symlink(approval_link)
    try:
        expected_approval_path, approval = _validate_approval(
            generation_root, output_root
        )
        if approval_path != expected_approval_path:
            raise Human312AuditError(
                "requested Human approval does not bind requested generation"
            )
        live_recheck = _verify_approved_live_sources(
            generation_root, approval, workers=int(workers)
        )
        # The generation link is compatibility-only.  The approval link is the
        # single authorization pointer and is atomically replaced last.
        _replace_symlink(generation_link, generation_root)
        _replace_symlink(approval_link, approval_path)
        active = validate_active_human_audit(output_root, rehash_sources=False)
        return {**active, "active_live_source_recheck": live_recheck}
    except BaseException as exc:
        rollback_errors: list[str] = []
        for link, previous in (
            (generation_link, previous_generation),
            (approval_link, previous_approval),
        ):
            try:
                _restore_symlink(link, previous)
            except Exception as rollback_exc:  # noqa: BLE001
                rollback_errors.append(f"{link}: {rollback_exc}")
        if rollback_errors:
            raise Human312AuditError(
                f"Human activation failed and rollback was incomplete: {rollback_errors}"
            ) from exc
        raise


def _active_binding_points_to(
    *, output_root: Path, generation_root: Path, approval_path: Path
) -> bool:
    """Best-effort commit detector used only to prevent post-commit quarantine."""
    try:
        return (
            _read_relative_symlink_target(
                output_root / HUMAN_AUDIT_LINK_NAME,
                label="active Human compatibility generation link",
            )
            == generation_root
            and _read_relative_symlink_target(
                output_root / HUMAN_AUDIT_APPROVAL_LINK_NAME,
                label="active Human approval link",
            )
            == approval_path
        )
    except (OSError, Human312AuditError):
        return False


def _active_links_reference_candidate(
    *, output_root: Path, generation_root: Path, approval_path: Path | None
) -> bool:
    """Prevent cleanup of either object while any active link still names it."""
    checks = [
        (output_root / HUMAN_AUDIT_LINK_NAME, generation_root),
    ]
    if approval_path is not None:
        checks.append((output_root / HUMAN_AUDIT_APPROVAL_LINK_NAME, approval_path))
    for link, expected in checks:
        try:
            if _read_relative_symlink_target(link, label="active Human link") == expected:
                return True
        except (OSError, Human312AuditError):
            continue
    return False


def _active_approval_binding(output_root: Path) -> tuple[Path, Path] | None:
    """Resolve the sole authority link to its approval and generation targets."""

    link = output_root / HUMAN_AUDIT_APPROVAL_LINK_NAME
    if not os.path.lexists(link):
        return None
    try:
        approval_path = _read_relative_symlink_target(
            link, label="active Human approval link"
        )
        _regular_stat(approval_path, label="active Human approval")
        approval = _load_json(approval_path)
        relative = Path(str(approval.get("generation_relpath", "")))
        if (
            not relative.parts
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != str(approval.get("generation_relpath", ""))
        ):
            raise Human312AuditError(
                "active Human approval has an invalid generation_relpath"
            )
        generation_root = _absolute_no_resolve(output_root / relative)
        return approval_path, generation_root
    except (OSError, Human312AuditError, ValueError, TypeError) as exc:
        raise Human312AuditError(
            "cannot prove that the active Human approval is unrelated to cleanup"
        ) from exc


def _quarantine_failed_candidate(
    *,
    generation_root: Path,
    generations_root: Path,
    approval_path: Path | None,
    work_root: Path,
    error: BaseException,
) -> Path:
    if generation_root.parent != generations_root or generation_root.is_symlink():
        raise Human312AuditError(
            f"cannot safely quarantine Human candidate: {generation_root}"
        )
    output_root = generations_root.parent
    generation_link = output_root / HUMAN_AUDIT_LINK_NAME
    if os.path.lexists(generation_link):
        try:
            if _read_relative_symlink_target(
                generation_link, label="active Human compatibility generation link"
            ) == generation_root:
                raise Human312AuditError(
                    "refusing to quarantine an actively referenced Human generation"
                )
        except FileNotFoundError as exc:
            raise Human312AuditError(
                "cannot prove that the active Human generation link is unrelated to cleanup"
            ) from exc
    active_approval = _active_approval_binding(output_root)
    if active_approval is not None:
        active_approval_path, active_generation_root = active_approval
        if active_generation_root == generation_root or (
            approval_path is not None and active_approval_path == approval_path
        ):
            raise Human312AuditError(
                "refusing to quarantine a Human generation referenced by the active approval"
            )
    rejected = generations_root / (
        f".rejected-{generation_root.name}-{uuid.uuid4().hex[:8]}"
    )
    os.replace(generation_root, rejected)
    _fsync_directory(generations_root)
    _write_json(
        work_root / "post_publish_validation_failure.json",
        {
            "status": "rejected",
            "candidate_generation_id": generation_root.name,
            "quarantined_relpath": rejected.relative_to(
                generations_root.parent
            ).as_posix(),
            "error_type": type(error).__name__,
            "error": str(error),
            "rejected_at_utc": _datetime.datetime.now(_datetime.UTC).isoformat(),
        },
    )
    return rejected


def _quarantine_failed_approval(
    *, approval_path: Path, output_root: Path, work_root: Path, error: BaseException
) -> Path:
    approval_root = output_root / HUMAN_AUDIT_APPROVAL_DIRECTORY
    if approval_path.parent != approval_root or approval_path.is_symlink():
        raise Human312AuditError(
            f"cannot safely quarantine Human approval: {approval_path}"
        )
    _regular_stat(approval_path, label="failed Human approval")
    active_link = output_root / HUMAN_AUDIT_APPROVAL_LINK_NAME
    if os.path.lexists(active_link):
        try:
            if _read_relative_symlink_target(
                active_link, label="active Human approval link"
            ) == approval_path:
                raise Human312AuditError(
                    "refusing to quarantine an actively referenced Human approval"
                )
        except FileNotFoundError as exc:
            raise Human312AuditError(
                "cannot prove that the active Human approval link is unrelated to cleanup"
            ) from exc
    rejected = approval_root / (
        f".rejected-{approval_path.stem}-{uuid.uuid4().hex[:8]}.json"
    )
    os.replace(approval_path, rejected)
    _fsync_directory(approval_root)
    _write_json(
        work_root / "approval_quarantine.json",
        {
            "status": "rejected",
            "approval_name": approval_path.name,
            "quarantined_relpath": rejected.relative_to(output_root).as_posix(),
            "error_type": type(error).__name__,
            "error": str(error),
            "rejected_at_utc": _datetime.datetime.now(_datetime.UTC).isoformat(),
        },
    )
    return rejected


def run_human_source_audit(config: HumanAuditConfig) -> dict[str, Any]:
    cfg = config.resolved()
    if cfg.workers <= 0 or cfg.chunk_size <= 0:
        raise Human312AuditError("workers and chunk_size must be positive")
    if (
        cfg.output_root.is_symlink()
        or not cfg.output_root.is_dir()
        or cfg.output_root.resolve(strict=True) != cfg.output_root
    ):
        raise Human312AuditError(f"invalid Human output root: {cfg.output_root}")
    parent = _validate_parent_manifest(cfg.manifest_root)
    rig, tasks = _load_human_scope(cfg.manifest_root, cfg.source_root)
    disk_inventory = _assert_manifest_disk_bijection(cfg.source_root, tasks)
    pinned_input_paths = {
        "active_cond": str(cfg.active_cond_path),
        "legacy_truebones_cond": str(cfg.legacy_truebones_cond_path),
        "t04_candidate": str(cfg.t04_candidate_path),
        "neutral_model": str(_absolute_no_resolve(rig["rest_pose"]["source_path"])),
    }
    pinned_inputs = {
        name: _stable_file_evidence(Path(path), label=f"Human {name}")
        for name, path in sorted(pinned_input_paths.items())
    }
    fixed_contract: HumanFixedRig = build_current_btjd_human_fixed_rig(
        rig_record=rig,
        active_cond_path=cfg.active_cond_path,
        legacy_truebones_cond_path=cfg.legacy_truebones_cond_path,
        t04_candidate_path=cfg.t04_candidate_path,
    )
    if fixed_contract.provenance.get("input_file_evidence") != pinned_inputs:
        raise Human312AuditError("Human fixed-rig builder input authority drifted")
    fixed_worker = {
        "parents": fixed_contract.parents,
        "P_rest_global": fixed_contract.P_rest_global,
        "offsets": fixed_contract.offset_parent_local,
        "s_rig": fixed_contract.s_rig,
    }
    authority_core = {
        "audit_version": HUMAN312_AUDIT_VERSION,
        "claim_boundary": HUMAN_CLAIM_BOUNDARY,
        "rotation_authority": HUMAN_ROTATION_AUTHORITY,
        "production_decoder": HUMAN_PRODUCTION_DECODER,
        "independent_decoder": HUMAN312_INDEPENDENT_DECODER,
        "parent_manifest_root": str(cfg.manifest_root),
        "parent_manifest": parent,
        "source_root": str(cfg.source_root),
        "task_scope_sha256": _task_scope_sha256(tasks),
        "disk_inventory_snapshot": disk_inventory,
        "clip_count": len(tasks),
        "rig_count": 1,
        "pinned_input_paths": pinned_input_paths,
        "pinned_inputs": pinned_inputs,
        "human_contract_version": HUMAN_CONTRACT_VERSION,
        "code_closure": _code_closure(),
        "runtime_fingerprint": _runtime_fingerprint(),
        "workers": int(cfg.workers),
        "chunk_size": int(cfg.chunk_size),
        "thresholds": {
            "source_fk_max_norm": SOURCE_FK_MAX_NORM,
            "dual_position_max_abs": DUAL_POSITION_MAX_ABS,
            "dual_rotation_max_abs": DUAL_ROTATION_MAX_ABS,
            "rotation_orthogonality_max_abs": ROTATION_ORTHOGONALITY_MAX_ABS,
            "rotation_determinant_max_abs": ROTATION_DETERMINANT_MAX_ABS,
            "fixed_rigid_edge_max_norm": FIXED_RIGID_EDGE_MAX_NORM,
            "raw_d6_unit_norm_max_abs": RAW_D6_UNIT_NORM_MAX_ABS,
            "raw_d6_row_dot_max_abs": RAW_D6_ROW_DOT_MAX_ABS,
            "raw_d6_cross_norm_min": RAW_D6_CROSS_NORM_MIN,
        },
        "anomaly_policy": HUMAN_ANOMALY_POLICY,
    }
    authority_sha = _authority_sha256(authority_core)
    authority = {**authority_core, "authority_sha256": authority_sha}
    work_namespace = _ensure_canonical_directory(
        cfg.output_root / HUMAN_AUDIT_WORK_DIRECTORY,
        label="Human audit work namespace",
    )
    work_root = _ensure_canonical_directory(
        work_namespace / authority_sha[:20],
        label="Human audit authority work root",
    )
    _write_json(work_root / "authority.json", authority)
    records, producer_statuses, producer_execution = _run_chunks(
        tasks=tasks,
        rig=rig,
        fixed=fixed_worker,
        source_root=cfg.source_root,
        work_root=work_root,
        authority_sha256=authority_sha,
        workers=cfg.workers,
        chunk_size=cfg.chunk_size,
        resumable=True,
    )
    _validate_records_against_tasks(records, tasks)
    _reject_unstable_source_records(records)
    producer_statuses = sorted(
        [_validate_worker_process_status(value) for value in producer_statuses],
        key=lambda value: value["pid"],
    )
    producer_execution = _validate_chunk_execution_evidence(
        producer_execution,
        expected_chunk_count=math.ceil(len(tasks) / cfg.chunk_size),
        allow_cache=True,
    )
    fresh_spawn_chunks = int(producer_execution["fresh_spawn_chunk_count"])
    if (
        len(producer_statuses) > cfg.workers
        or len({value["pid"] for value in producer_statuses})
        != len(producer_statuses)
        or any(status["pid"] == status["ppid"] for status in producer_statuses)
        or (fresh_spawn_chunks > 0 and not producer_statuses)
        or (fresh_spawn_chunks == 0 and producer_statuses)
    ):
        raise Human312AuditError("Human producer workers are not proven single-threaded")
    status_counts = Counter(str(record["status"]) for record in records)
    accepted_count = int(status_counts.get("pass", 0))
    rejected_count = int(status_counts.get("reject", 0))
    if accepted_count + rejected_count != EXPECTED_HUMAN_CLIP_COUNT:
        raise Human312AuditError("Human status accounting drifted")
    if accepted_count <= 0:
        raise Human312AuditError("Human rig coverage failed: no accepted clips")
    snapshot = _source_snapshot(records)
    source_snapshot_recheck = _revalidate_all_live_sources(
        records,
        tasks,
        source_root=cfg.source_root,
        workers=cfg.workers,
    )
    representative = _select_representative(records)
    published_contract = build_current_btjd_human_fixed_rig(
        rig_record=rig,
        active_cond_path=cfg.active_cond_path,
        legacy_truebones_cond_path=cfg.legacy_truebones_cond_path,
        t04_candidate_path=cfg.t04_candidate_path,
        representative_clip_id=str(representative["clip_id"]),
    )
    if (
        not np.array_equal(published_contract.parents, fixed_contract.parents)
        or not np.array_equal(
            published_contract.P_rest_global, fixed_contract.P_rest_global
        )
        or not np.array_equal(
            published_contract.offset_parent_local,
            fixed_contract.offset_parent_local,
        )
        or published_contract.s_rig != fixed_contract.s_rig
        or published_contract.provenance.get("input_file_evidence") != pinned_inputs
    ):
        raise Human312AuditError("Human representative skeleton rebuild drifted")

    producer_evidence = {
        "audit_version": HUMAN312_AUDIT_VERSION,
        "status": "pass",
        "executor_mode": producer_execution["executor_mode"],
        "authority_sha256": authority_sha,
        "task_scope_sha256": authority["task_scope_sha256"],
        "record_count": len(records),
        "records_sha256": _records_sha256(records),
        "chunk_count": int(producer_execution["chunk_count"]),
        "cached_revalidated_chunk_count": int(
            producer_execution["cached_revalidated_chunk_count"]
        ),
        "fresh_spawn_chunk_count": int(
            producer_execution["fresh_spawn_chunk_count"]
        ),
        "fresh_spawn_chunks_with_process_status": int(
            producer_execution["fresh_spawn_chunks_with_process_status"]
        ),
        "cached_worker_process_status_trusted": False,
        "worker_process_statuses": producer_statuses,
        "worker_process_statuses_sha256": _worker_statuses_sha256(
            producer_statuses
        ),
    }

    generation_id = (
        _datetime.datetime.now(_datetime.UTC).strftime("%Y%m%dT%H%M%S%fZ")
        + "-"
        + uuid.uuid4().hex[:12]
    )
    generations = _ensure_canonical_directory(
        cfg.output_root / HUMAN_AUDIT_GENERATION_DIRECTORY,
        label="Human audit generation namespace",
    )
    staging = Path(tempfile.mkdtemp(prefix=f".{generation_id}.", dir=generations))
    final = generations / generation_id
    approval_path: Path | None = None
    approval_cleanup_witness = _ApprovalCleanupWitness()
    activation_committed = False
    standalone_approval_committed = False
    active: dict[str, Any] | None = None
    result_payload: dict[str, Any] | None = None
    try:
        skeleton_sha = _write_npz_atomic(
            staging / "skeletons/HML3D_Human.npz", published_contract.payload
        )
        _write_jsonl(staging / "qa/human_source_audit.jsonl", records)
        rejected_records = [record for record in records if record["status"] == "reject"]
        _write_jsonl(staging / "qa/rejected_clips.jsonl", rejected_records)
        _write_json(
            staging / "qa/source_snapshot_recheck.json",
            source_snapshot_recheck,
        )
        _write_json(
            staging / "qa/producer_worker_process_status.json",
            producer_evidence,
        )
        _write_json(staging / "selection/human_representative.json", representative)
        _write_json(staging / "authority.json", authority)
        metrics = [record["metrics"] for record in records if record["status"] == "pass"]
        summary = {
            "audit_version": HUMAN312_AUDIT_VERSION,
            "generation_id": generation_id,
            "status": "pending_post_publish_deep_validation",
            "source_audit_status": "pass" if rejected_count == 0 else "pass_with_rejections",
            "rig_count": 1,
            "clip_count": EXPECTED_HUMAN_CLIP_COUNT,
            "accepted_clip_count": accepted_count,
            "rejected_clip_count": rejected_count,
            "status_counts": dict(sorted(status_counts.items())),
            "rejection_reason_counts": dict(
                sorted(
                    Counter(
                        reason
                        for record in rejected_records
                        for reason in record["reason_codes"]
                    ).items()
                )
            ),
            "rig_coverage_status": "pass",
            "T_src_min": min(int(record["T_src"]) for record in records),
            "T_src_max": max(int(record["T_src"]) for record in records),
            "source_snapshot_sha256": snapshot["source_snapshot_sha256"],
            "independent_position_max_abs": max(
                float(value["independent_positions_max_abs"]) for value in metrics
            ),
            "independent_rotation_max_abs": max(
                max(
                    float(value["independent_local_rotation_max_abs"]),
                    float(value["independent_global_rotation_max_abs"]),
                )
                for value in metrics
            ),
            "source_parser_fk_max_norm": max(
                float(value["source_parser_fk_max_norm"]) for value in metrics
            ),
            "fixed_neutral_rigid_edge_max_norm": max(
                float(value["fixed_neutral_rigid_edge_max_norm"]) for value in metrics
            ),
            "representative": representative,
            "skeleton_sha256": skeleton_sha,
            "producer_worker_process_count": len(producer_statuses),
            "producer_worker_threads_max": max(
                (int(value["threads"]) for value in producer_statuses),
                default=0,
            ),
            "producer_worker_process_statuses": producer_statuses,
            "source_snapshot_recheck": source_snapshot_recheck,
            "authority_sha256": authority_sha,
            "claim_boundary": HUMAN_CLAIM_BOUNDARY,
            "prototype_conversion_authorized": False,
            "full_conversion_authorized": False,
        }
        _write_json(staging / "summary.json", summary)
        files = _file_manifest(staging)
        generation = {
            "audit_version": HUMAN312_AUDIT_VERSION,
            "generation_id": generation_id,
            "created_at_utc": _datetime.datetime.now(_datetime.UTC).isoformat(),
            "status": "pending_post_publish_deep_validation",
            "authority_sha256": authority_sha,
            "parent_manifest_root": str(cfg.manifest_root),
            "files": files,
            "prototype_conversion_authorized": False,
            "full_conversion_authorized": False,
        }
        _write_json(staging / "generation.json", generation)
        _freeze_immutable_tree(staging)
        if final.exists():
            raise Human312AuditError(f"Human audit generation exists: {final}")
        os.replace(staging, final)
        _fsync_directory(generations)
        candidate_proof = _validate_candidate(final, workers=cfg.workers)
        post_return_generation = _validate_generation_structure(final)
        post_return_content = _generation_content_evidence(final)
        if (
            post_return_generation != candidate_proof["generation"]
            or post_return_content != candidate_proof["content_evidence"]
        ):
            raise Human312AuditError(
                "Human candidate changed after deep validation returned"
            )
        # _validate_candidate performs this complete stable-byte rehash after
        # the non-resumable deep replay; bind that exact evidence into approval.
        post_deep_live_recheck = _validate_live_recheck_evidence(
            candidate_proof["deep_source_recheck"]
        )
        if (
            _validate_generation_structure(final) != post_return_generation
            or _generation_content_evidence(final) != post_return_content
            or _validate_parent_manifest(cfg.manifest_root) != parent
            or _current_pinned_inputs(authority) != pinned_inputs
            or _code_closure() != authority["code_closure"]
            or _runtime_fingerprint() != authority["runtime_fingerprint"]
        ):
            raise Human312AuditError(
                "Human candidate or source authority changed before approval"
            )
        approval_path, approval, approval_created = _create_approval(
            output_root=cfg.output_root,
            generation_root=final,
            candidate_proof=candidate_proof,
            post_deep_live_recheck=post_deep_live_recheck,
            deep_records_sha256=str(candidate_proof["deep_records_sha256"]),
            deep_worker_statuses=candidate_proof[
                "deep_worker_process_statuses"
            ],
            cleanup_witness=approval_cleanup_witness,
        )
        if (
            approval_path != approval_cleanup_witness.path
            or approval_created != approval_cleanup_witness.owned_by_run
        ):
            raise Human312AuditError("Human approval cleanup ownership drifted")
        expected_approval_path, validated_approval = _validate_approval(
            final, cfg.output_root
        )
        if expected_approval_path != approval_path or validated_approval != approval:
            raise Human312AuditError("Human approval changed after creation")
        result_payload = {
            **summary,
            "status": "pass",
            "prototype_conversion_authorized": True,
            "full_conversion_authorized": False,
            "generation_root": str(final),
            "generation_content_sha256": approval["generation_content_sha256"],
            "approval_path": str(approval_path),
            "active_binding": None,
            "resumable_work_root": str(work_root),
        }
        if cfg.update_link:
            active = _activate_approved_generation(
                output_root=cfg.output_root,
                generation_root=final,
                approval_path=approval_path,
                workers=cfg.workers,
            )
            activation_committed = True
        else:
            _verify_approved_live_sources(
                final, approval, workers=cfg.workers
            )
            standalone_approval_committed = True
    except BaseException as exc:
        # An interrupt can land after the atomic approval-link swap but before
        # the caller assigns activation_committed.  Inspect the links as the
        # second commit witness and never quarantine an authorized object.
        if (
            activation_committed
            or standalone_approval_committed
            or _active_links_reference_candidate(
                output_root=cfg.output_root,
                generation_root=final,
                approval_path=approval_path or approval_cleanup_witness.path,
            )
        ):
            raise
        quarantine_errors: list[str] = []
        cleanup_approval_path = approval_path or approval_cleanup_witness.path
        if (
            approval_cleanup_witness.owned_by_run
            and cleanup_approval_path is not None
            and os.path.lexists(cleanup_approval_path)
        ):
            try:
                _quarantine_failed_approval(
                    approval_path=cleanup_approval_path,
                    output_root=cfg.output_root,
                    work_root=work_root,
                    error=exc,
                )
            except Exception as quarantine_exc:  # noqa: BLE001
                quarantine_errors.append(f"approval: {quarantine_exc}")
        if final.exists() and not final.is_symlink():
            try:
                _quarantine_failed_candidate(
                    generation_root=final,
                    generations_root=generations,
                    approval_path=cleanup_approval_path,
                    work_root=work_root,
                    error=exc,
                )
            except Exception as quarantine_exc:  # noqa: BLE001
                quarantine_errors.append(f"generation: {quarantine_exc}")
        if quarantine_errors:
            raise Human312AuditError(
                f"Human audit failed and quarantine was incomplete: {quarantine_errors}"
            ) from exc
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    if result_payload is None:
        raise Human312AuditError("Human audit completed without a result payload")
    result_payload["active_binding"] = active
    return result_payload


def _first_record_difference(expected: Any, actual: Any, path: str = "$") -> str:
    if type(expected) is not type(actual):
        return f"{path}: type {type(actual).__name__} != {type(expected).__name__}"
    if isinstance(expected, Mapping):
        if set(expected) != set(actual):
            return f"{path}: key set drifted"
        for key in sorted(expected):
            difference = _first_record_difference(expected[key], actual[key], f"{path}.{key}")
            if difference:
                return difference
        return ""
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{path}: length {len(actual)} != {len(expected)}"
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            difference = _first_record_difference(left, right, f"{path}[{index}]")
            if difference:
                return difference
        return ""
    if expected != actual:
        return f"{path}: {actual!r} != {expected!r}"
    return ""
