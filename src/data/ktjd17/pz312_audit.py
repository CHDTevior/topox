"""Exhaustive, resumable source audit for the 311 PlanetZoo stage-2 rigs.

The audit is intentionally stronger than the later encoder pass.  Every BVH is
decoded once by the production parser and once by SciPy from an independently
read numeric stream.  The production result must also close against the fixed
per-rig stage-2 skeleton.  Only a complete zero-reject corpus can publish an
immutable audit generation or choose one dynamic representative per rig.
"""

from __future__ import annotations

import ast
import concurrent.futures
import contextlib
import dataclasses
import datetime as _datetime
import hashlib
import importlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import platform
import re
import shutil
import stat as stat_module
import subprocess
import sys
import tempfile
import uuid
import warnings
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.spatial.transform import Rotation
import threadpoolctl

from .encoder import SkeletonData, load_skeleton, write_npz_atomic
from .inventory_validation import _validate_transaction
from .planetzoo_fixed_rig import (
    PLANETZOO_FIXED_RIG_VERSION,
    PLANETZOO_REST_MODE,
    PLANETZOO_SOURCE_POSITION_MAX_NORM,
    build_planetzoo_fixed_rig,
    validate_planetzoo_parsed_against_skeleton,
)
from .source_parser import ParsedSourceMotion, parse_bvh_source
from .truebones_fixed_rig import ACTIVE_COND_SHA256


PZ312_AUDIT_VERSION = "ktjd17-pz311-exhaustive-source-audit-v4"
EXPECTED_PZ_RIG_COUNT = 311
EXPECTED_PZ_CLIP_COUNT = 74_522
DUAL_ROTATION_MAX_ABS = 1e-12
DUAL_ROOT_MAX_ABS = 1e-12
AUDIT_GENERATION_DIRECTORY = ".ktjd17_pz_source_audit_generations"
AUDIT_WORK_DIRECTORY = ".ktjd17_pz_source_audit_work"
AUDIT_LINK_NAME = "ktjd17_pz_source_audit"
AUDIT_APPROVAL_DIRECTORY = ".ktjd17_pz_source_audit_approvals"
AUDIT_APPROVAL_LINK_NAME = "ktjd17_pz_source_audit_approval"
AUDIT_APPROVAL_VERSION = "ktjd17-pz311-source-audit-approval-v2"
CANDIDATE_STATUS = "pending_post_publish_deep_validation"
SINGLE_THREAD_ENVIRONMENT = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
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
        "parent_status",
        "status",
        "reason_codes",
        "source_path",
        "source_sha256",
        "source_size_bytes",
        "source_mtime_ns",
        "source_device",
        "source_inode",
        "source_nlink",
        "slice_frames",
        "T_src",
        "J_phys",
        "frame_time_src",
        "fps_src",
        "source_joint_count",
        "source_channel_count",
        "rotation_layout_sha256",
        "rest_layout_sha256",
        "metrics",
    }
)
PASS_METRIC_KEYS = frozenset(
    {
        "planetzoo_per_clip_declared_offset_exact",
        "planetzoo_per_clip_rotation_layout_exact",
        "planetzoo_per_clip_rest_layout_exact",
        "planetzoo_root_translation_exact",
        "planetzoo_fixed_fk_source_position_max_norm",
        "planetzoo_fixed_fk_source_position_mpjpe_norm",
        "planetzoo_stage2_contract",
        "independent_decoder",
        "independent_source_sha256",
        "independent_rotation_max_abs",
        "independent_root_translation_max_abs",
        "independent_frame_count",
        "independent_frame_time_src",
        "independent_fps_src",
        "independent_source_joint_count",
        "independent_source_channel_count",
        "independent_rotation_layout_sha256",
        "independent_rest_layout_sha256",
        "root_speed_rms_norm_per_s",
        "rotation_speed_rms_rad_per_s",
        "dynamic_score",
    }
)
INDEPENDENT_DECODER_ID = (
    "independent_byte_header_and_numeric_parser_plus_"
    "scipy_Rotation_from_euler_intrinsic_uppercase"
)


class Pz312AuditError(RuntimeError):
    """The full PlanetZoo source scope did not close exactly."""


@dataclasses.dataclass(frozen=True)
class _IndependentBvhJoint:
    """One BVH node decoded without the production header parser."""

    name: str
    parent: int
    node_kind: str
    offset: tuple[float, float, float]
    channels: tuple[str, ...]
    channels_declared: bool


@dataclasses.dataclass(frozen=True)
class _IndependentBvh:
    """Independent hierarchy, timing, numeric stream, and source hash."""

    joints: tuple[_IndependentBvhJoint, ...]
    frames: int
    frame_time: float
    values: np.ndarray
    source_sha256: str

    @property
    def channel_count(self) -> int:
        return sum(len(joint.channels) for joint in self.joints)

    def rotation_layout_sha256(self) -> str:
        payload = [
            {
                "name": joint.name,
                "parent": joint.parent,
                "node_kind": joint.node_kind,
                "channels": list(joint.channels),
                "channels_declared": joint.channels_declared,
            }
            for joint in self.joints
        ]
        return hashlib.sha256(_canonical_json(payload)).hexdigest()

    def rest_layout_sha256(self) -> str:
        payload = [
            {
                "name": joint.name,
                "parent": joint.parent,
                "node_kind": joint.node_kind,
                "offset": list(joint.offset),
                "channels": list(joint.channels),
                "channels_declared": joint.channels_declared,
            }
            for joint in self.joints
        ]
        return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _absolute_no_resolve(path: str | Path) -> Path:
    """Normalize ``.``/``..`` while preserving a leaf symlink for lstat checks."""
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


@dataclasses.dataclass(frozen=True)
class PzAuditConfig:
    manifest_root: Path
    pz_bvh_root: Path
    active_cond_path: Path
    output_root: Path
    workers: int = 24
    chunk_size: int = 128
    update_link: bool = True

    def resolved(self) -> "PzAuditConfig":
        return dataclasses.replace(
            self,
            manifest_root=self.manifest_root.expanduser().resolve(),
            pz_bvh_root=_absolute_no_resolve(self.pz_bvh_root),
            active_cond_path=_absolute_no_resolve(self.active_cond_path),
            output_root=_absolute_no_resolve(self.output_root),
        )


def default_pz_audit_config(repo_root: str | Path = ".") -> PzAuditConfig:
    root = Path(repo_root).expanduser().resolve()
    return PzAuditConfig(
        manifest_root=root / "dataset/manifests",
        pz_bvh_root=root / "data/animo4d_anytop/bvhs",
        active_cond_path=(
            root / "data/animo4d_L4TB_plus_human_v4b272neutral/cond.npy"
        ),
        output_root=root / "dataset",
    )


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_single_link_stat(path: Path, *, label: str) -> os.stat_result:
    """Return an lstat only for a regular file with no hard/symbolic aliases."""
    if path.is_symlink():
        raise Pz312AuditError(f"{label} is a symlink: {path}")
    try:
        observed = path.lstat()
    except OSError as exc:
        raise Pz312AuditError(f"cannot lstat {label} {path}: {exc}") from exc
    if not stat_module.S_ISREG(observed.st_mode):
        raise Pz312AuditError(f"{label} is not a regular file: {path}")
    if int(observed.st_nlink) != 1:
        raise Pz312AuditError(
            f"{label} has hard-link aliases (st_nlink={observed.st_nlink}): {path}"
        )
    return observed


def _producer_code_sha256() -> dict[str, str]:
    """Hash the complete local relative-import closure of this audit module."""
    module_root = Path(__file__).resolve().parent
    pending = [module_root / "pz312_audit.py"]
    init = module_root / "__init__.py"
    if init.is_file():
        pending.append(init)
    visited: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        _regular_single_link_stat(path, label="producer source module")
        visited.add(path)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except Exception as exc:  # noqa: BLE001
            raise Pz312AuditError(f"cannot parse producer module {path}: {exc}") from exc
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
        path.relative_to(repo_root).as_posix(): _sha256_file(path)
        for path in sorted(visited)
    }


def _distribution_content_fingerprint(distribution: Any) -> dict[str, Any]:
    """Hash every installed byte declared by one wheel distribution."""
    files = distribution.files
    if files is None:
        raise Pz312AuditError("installed distribution exposes no file inventory")
    entries: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for relative in sorted(files, key=lambda value: str(value)):
        relative_path = str(relative)
        path = Path(distribution.locate_file(relative))
        if path.is_symlink():
            raise Pz312AuditError(f"runtime distribution file is a symlink: {path}")
        try:
            resolved = path.resolve(strict=True)
            observed = resolved.stat()
        except OSError as exc:
            raise Pz312AuditError(
                f"cannot resolve runtime distribution file {path}: {exc}"
            ) from exc
        if not stat_module.S_ISREG(observed.st_mode):
            raise Pz312AuditError(
                f"runtime distribution entry is not regular: {resolved}"
            )
        canonical_path = str(resolved)
        if canonical_path in seen_paths:
            raise Pz312AuditError(
                f"runtime distribution inventory aliases one file: {resolved}"
            )
        seen_paths.add(canonical_path)
        entries.append(
            {
                "relative_path": relative_path,
                "resolved_path": canonical_path,
                "sha256": _sha256_file(resolved),
                "size_bytes": int(observed.st_size),
            }
        )
    return {
        "actual_file_count": len(entries),
        "actual_total_size_bytes": sum(
            int(entry["size_bytes"]) for entry in entries
        ),
        "actual_files_sha256": hashlib.sha256(
            _canonical_json(entries)
        ).hexdigest(),
    }


def _native_library_dependency_fingerprint(
    roots: Sequence[Path],
) -> dict[str, Any]:
    """Hash native roots and their recursively resolved dynamic dependencies."""
    libraries: dict[str, dict[str, Any]] = {}
    pending = [path.resolve(strict=True) for path in roots]
    while pending:
        path = pending.pop()
        canonical = str(path)
        if canonical in libraries:
            continue
        try:
            observed = path.stat()
        except OSError as exc:
            raise Pz312AuditError(
                f"cannot stat native runtime dependency {path}: {exc}"
            ) from exc
        if not stat_module.S_ISREG(observed.st_mode):
            raise Pz312AuditError(f"native runtime dependency is not regular: {path}")
        libraries[canonical] = {
            "sha256": _sha256_file(path),
            "size_bytes": int(observed.st_size),
        }
        completed = subprocess.run(
            ["ldd", canonical],
            check=False,
            capture_output=True,
            text=True,
        )
        output = completed.stdout + completed.stderr
        if "not found" in output:
            raise Pz312AuditError(
                f"native runtime dependency is unresolved for {path}: {output.strip()}"
            )
        if completed.returncode not in (0, 1):
            raise Pz312AuditError(
                f"ldd failed for native runtime dependency {path}: {output.strip()}"
            )
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
        raise Pz312AuditError("no native runtime dependency was discoverable")
    ordered = {name: libraries[name] for name in sorted(libraries)}
    return {
        "library_count": len(ordered),
        "libraries": ordered,
        "closure_sha256": hashlib.sha256(_canonical_json(ordered)).hexdigest(),
    }


def _runtime_fingerprint() -> dict[str, Any]:
    """Pin numerical package bytes and the effective single-thread native ABI."""
    with threadpoolctl.threadpool_limits(limits=1):
        pools = threadpoolctl.threadpool_info()
    if not pools:
        raise Pz312AuditError("no loaded BLAS runtime was discoverable")
    normalized_pools: list[dict[str, Any]] = []
    for pool in pools:
        library = Path(str(pool.get("filepath", ""))).resolve()
        if not library.is_file():
            raise Pz312AuditError(f"BLAS runtime library is absent: {library}")
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
            raise Pz312AuditError(f"BLAS runtime is not capped to one thread: {entry}")
        normalized_pools.append(entry)
    normalized_pools.sort(key=lambda item: _canonical_json(item))
    extension_modules: dict[str, dict[str, Any]] = {}
    for module_name in (
        "numpy._core._multiarray_umath",
        "numpy.linalg._umath_linalg",
        "scipy.spatial.transform._rotation",
    ):
        module = importlib.import_module(module_name)
        path = Path(str(module.__file__)).resolve()
        extension_modules[module_name] = {
            "path": str(path),
            "sha256": _sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
    distributions: dict[str, dict[str, Any]] = {}
    for distribution_name in ("numpy", "scipy", "threadpoolctl"):
        distribution = importlib.metadata.distribution(distribution_name)
        record = distribution.read_text("RECORD")
        if record is None:
            raise Pz312AuditError(
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
        "byteorder": sys.byteorder,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "threadpoolctl_version": threadpoolctl.__version__,
        "installed_distributions": distributions,
        "extension_modules": extension_modules,
        "blas_pools": normalized_pools,
        "native_library_dependencies": _native_library_dependency_fingerprint(
            [
                python_executable,
                *(Path(entry["path"]) for entry in extension_modules.values()),
                *(Path(entry["filepath"]) for entry in normalized_pools),
            ]
        ),
    }


@contextlib.contextmanager
def _single_thread_spawn_environment() -> Iterable[None]:
    """Make BLAS limits visible before a spawned interpreter imports NumPy."""
    previous = {name: os.environ.get(name) for name in SINGLE_THREAD_ENVIRONMENT}
    try:
        for name in SINGLE_THREAD_ENVIRONMENT:
            os.environ[name] = "1"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _worker_process_status() -> dict[str, int]:
    """Small spawn-safe probe used by the real OS-thread regression test."""
    result = {"pid": os.getpid(), "threads": -1, "vmrss_kib": -1}
    status = Path("/proc/self/status")
    if not status.is_file():
        raise Pz312AuditError("/proc/self/status is unavailable")
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith("Threads:"):
            result["threads"] = int(line.split(":", 1)[1].strip())
        elif line.startswith("VmRSS:"):
            result["vmrss_kib"] = int(line.split(":", 1)[1].split()[0])
    if result["threads"] <= 0 or result["vmrss_kib"] <= 0:
        raise Pz312AuditError(f"cannot read worker process status: {result}")
    return result


def _validate_worker_process_status(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != {
        "pid",
        "threads",
        "vmrss_kib",
    }:
        raise Pz312AuditError("worker OS-process evidence schema drifted")
    result: dict[str, int] = {}
    for field in ("pid", "threads", "vmrss_kib"):
        observed = value[field]
        if type(observed) is not int or int(observed) <= 0:
            raise Pz312AuditError(
                f"worker OS-process evidence is invalid: {field}={observed!r}"
            )
        result[field] = int(observed)
    if result["threads"] != 1:
        raise Pz312AuditError(
            f"spawned audit worker has {result['threads']} OS threads, expected 1"
        )
    return result


def _worker_statuses_sha256(values: Sequence[Mapping[str, Any]]) -> str:
    normalized = [_validate_worker_process_status(value) for value in values]
    return hashlib.sha256(_canonical_json(normalized)).hexdigest()


def _records_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(_canonical_json(list(records))).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for record in records:
            handle.write(_canonical_json(record) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise Pz312AuditError(f"cannot read JSON {path}: {exc}") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise Pz312AuditError(f"{path}:{line_number}: blank JSONL row")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise Pz312AuditError(
                        f"{path}:{line_number}: row is not an object"
                    )
                records.append(value)
    except Pz312AuditError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise Pz312AuditError(f"cannot read JSONL {path}: {exc}") from exc
    return records


def _file_manifest(
    root: Path, *, require_read_only: bool = False
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    inode_owner: dict[tuple[int, int], str] = {}
    root_stat = root.lstat()
    if require_read_only and int(root_stat.st_mode) & 0o222:
        raise Pz312AuditError(f"immutable audit root is writable: {root}")
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise Pz312AuditError(f"symlink inside immutable audit: {path}")
        observed = path.lstat()
        if stat_module.S_ISDIR(observed.st_mode):
            if require_read_only and int(observed.st_mode) & 0o222:
                raise Pz312AuditError(f"immutable audit directory is writable: {path}")
            continue
        if not stat_module.S_ISREG(observed.st_mode):
            raise Pz312AuditError(f"non-regular entry inside immutable audit: {path}")
        if int(observed.st_nlink) != 1:
            raise Pz312AuditError(
                f"hard-linked entry inside immutable audit: {path} "
                f"(st_nlink={observed.st_nlink})"
            )
        if require_read_only and int(observed.st_mode) & 0o222:
            raise Pz312AuditError(f"immutable audit file is writable: {path}")
        relpath = path.relative_to(root).as_posix()
        inode = (int(observed.st_dev), int(observed.st_ino))
        if inode in inode_owner:
            raise Pz312AuditError(
                f"duplicate immutable-audit inode: {inode_owner[inode]} and {relpath}"
            )
        inode_owner[inode] = relpath
        if relpath == "generation.json":
            continue
        result[relpath] = {
            "sha256": _sha256_file(path),
            "size_bytes": int(observed.st_size),
        }
    return result


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _freeze_immutable_tree(root: Path) -> None:
    """Remove all write bits after the candidate payload is fully materialized."""
    _file_manifest(root)
    entries = sorted(root.rglob("*"), key=lambda path: len(path.parts), reverse=True)
    for path in entries:
        observed = path.lstat()
        if stat_module.S_ISREG(observed.st_mode):
            os.chmod(path, int(observed.st_mode) & ~0o222)
        elif stat_module.S_ISDIR(observed.st_mode):
            os.chmod(path, int(observed.st_mode) & ~0o222)
        else:
            raise Pz312AuditError(f"cannot freeze non-regular audit entry: {path}")
    root_stat = root.lstat()
    os.chmod(root, int(root_stat.st_mode) & ~0o222)
    _file_manifest(root, require_read_only=True)
    _fsync_directory(root.parent)


def _replace_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(link) and not link.is_symlink():
        raise Pz312AuditError(f"refusing to replace non-symlink {link}")
    temporary = link.parent / f".{link.name}.{uuid.uuid4().hex}.tmp"
    os.symlink(os.path.relpath(target, start=link.parent), temporary)
    os.replace(temporary, link)
    _fsync_directory(link.parent)


def _snapshot_symlink(link: Path) -> str | None:
    if not os.path.lexists(link):
        return None
    if not link.is_symlink():
        raise Pz312AuditError(f"active authority path is not a symlink: {link}")
    return os.readlink(link)


def _restore_symlink(link: Path, previous_target: str | None) -> None:
    if previous_target is None:
        if os.path.lexists(link):
            if not link.is_symlink():
                raise Pz312AuditError(
                    f"cannot roll back non-symlink authority path: {link}"
                )
            link.unlink()
            _fsync_directory(link.parent)
        return
    temporary = link.parent / f".{link.name}.{uuid.uuid4().hex}.rollback"
    os.symlink(previous_target, temporary)
    os.replace(temporary, link)
    _fsync_directory(link.parent)


def _parse_independent_bvh_bytes(raw: bytes, *, path: Path) -> _IndependentBvh:
    """Strictly parse BVH hierarchy and samples without production BVH helpers."""
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise Pz312AuditError(f"independent UTF-8 decode failed for {path}: {exc}") from exc
    lines = [(index, line.strip()) for index, line in enumerate(text.splitlines(), 1)]
    cursor = 0

    def next_nonempty() -> tuple[int, str]:
        nonlocal cursor
        while cursor < len(lines):
            item = lines[cursor]
            cursor += 1
            if item[1]:
                return item
        raise Pz312AuditError(f"independent parser reached EOF in {path}")

    line_number, first = next_nonempty()
    if first.upper() != "HIERARCHY":
        raise Pz312AuditError(
            f"independent parser expected HIERARCHY at {path}:{line_number}"
        )

    mutable: list[dict[str, Any]] = []
    stack: list[int] = []
    pending: int | None = None
    unnamed_end_count: dict[int, int] = {}
    saw_motion = False
    declaration_re = re.compile(r"^(ROOT|JOINT)\s+(.+?)\s*$", re.IGNORECASE)
    end_re = re.compile(
        r"^End\s+Site(?:\s+#name:\s*(.+?))?\s*$", re.IGNORECASE
    )
    while cursor < len(lines):
        line_number, value = next_nonempty()
        if value.upper() == "MOTION":
            if pending is not None or stack:
                raise Pz312AuditError(
                    f"independent hierarchy is unclosed at {path}:{line_number}"
                )
            saw_motion = True
            break
        declaration = declaration_re.match(value)
        if declaration:
            if pending is not None:
                raise Pz312AuditError(
                    f"independent declaration lacks brace at {path}:{line_number}"
                )
            kind = declaration.group(1).lower()
            name = declaration.group(2).strip()
            if not name or (kind == "root" and mutable) or (kind == "joint" and not stack):
                raise Pz312AuditError(
                    f"independent invalid joint declaration at {path}:{line_number}"
                )
            mutable.append(
                {
                    "name": name,
                    "parent": -1 if kind == "root" else stack[-1],
                    "node_kind": "joint",
                    "channels": None,
                    "channels_declared": False,
                    "offset_seen": False,
                    "offset": None,
                }
            )
            pending = len(mutable) - 1
            continue
        end_site = end_re.match(value)
        if end_site:
            if pending is not None or not stack:
                raise Pz312AuditError(
                    f"independent End Site has no parent at {path}:{line_number}"
                )
            parent = stack[-1]
            explicit_name = end_site.group(1)
            if explicit_name is None:
                number = unnamed_end_count.get(parent, 0)
                unnamed_end_count[parent] = number + 1
                name = f"{mutable[parent]['name']}__unnamed_end_site_{number}"
            else:
                name = explicit_name.strip()
            if not name:
                raise Pz312AuditError(
                    f"independent empty End Site name at {path}:{line_number}"
                )
            mutable.append(
                {
                    "name": name,
                    "parent": parent,
                    "node_kind": "end_site",
                    "channels": (),
                    "channels_declared": True,
                    "offset_seen": False,
                    "offset": None,
                }
            )
            pending = len(mutable) - 1
            continue
        if value == "{":
            if pending is None:
                raise Pz312AuditError(
                    f"independent unmatched opening brace at {path}:{line_number}"
                )
            stack.append(pending)
            pending = None
            continue
        if value == "}":
            if pending is not None or not stack:
                raise Pz312AuditError(
                    f"independent unmatched closing brace at {path}:{line_number}"
                )
            stack.pop()
            continue
        if value.upper().startswith("OFFSET"):
            if not stack or mutable[stack[-1]]["offset_seen"]:
                raise Pz312AuditError(
                    f"independent invalid OFFSET at {path}:{line_number}"
                )
            fields = value.split()
            if len(fields) != 4:
                raise Pz312AuditError(
                    f"independent malformed OFFSET at {path}:{line_number}"
                )
            try:
                offset = tuple(float(item) for item in fields[1:])
            except ValueError as exc:
                raise Pz312AuditError(
                    f"independent nonnumeric OFFSET at {path}:{line_number}"
                ) from exc
            if not all(math.isfinite(item) for item in offset):
                raise Pz312AuditError(
                    f"independent nonfinite OFFSET at {path}:{line_number}"
                )
            mutable[stack[-1]]["offset_seen"] = True
            mutable[stack[-1]]["offset"] = offset
            continue
        if value.upper().startswith("CHANNELS"):
            if not stack:
                raise Pz312AuditError(
                    f"independent CHANNELS outside joint at {path}:{line_number}"
                )
            joint = mutable[stack[-1]]
            if joint["node_kind"] == "end_site" or joint["channels_declared"]:
                raise Pz312AuditError(
                    f"independent duplicate/End-Site CHANNELS at {path}:{line_number}"
                )
            fields = value.split()
            try:
                count = int(fields[1])
            except (IndexError, ValueError) as exc:
                raise Pz312AuditError(
                    f"independent malformed CHANNELS at {path}:{line_number}"
                ) from exc
            channels = tuple(fields[2:])
            if count < 0 or count != len(channels):
                raise Pz312AuditError(
                    f"independent CHANNELS count mismatch at {path}:{line_number}"
                )
            for channel in channels:
                if re.fullmatch(r"[XYZ](?:position|rotation)", channel, re.IGNORECASE) is None:
                    raise Pz312AuditError(
                        f"independent unsupported channel {channel!r} at "
                        f"{path}:{line_number}"
                    )
            joint["channels"] = channels
            joint["channels_declared"] = True
            continue
        raise Pz312AuditError(
            f"independent unsupported hierarchy line at {path}:{line_number}: {value!r}"
        )
    if not saw_motion or not mutable:
        raise Pz312AuditError(f"independent parser found no complete hierarchy in {path}")

    names: set[str] = set()
    joints: list[_IndependentBvhJoint] = []
    for index, record in enumerate(mutable):
        if not record["offset_seen"] or not record["channels_declared"]:
            raise Pz312AuditError(
                f"independent incomplete joint {record['name']!r} in {path}"
            )
        if record["name"] in names:
            raise Pz312AuditError(
                f"independent duplicate joint {record['name']!r} in {path}"
            )
        names.add(str(record["name"]))
        parent = int(record["parent"])
        if (index == 0 and parent != -1) or (index > 0 and not 0 <= parent < index):
            raise Pz312AuditError(
                f"independent parent order drifted at {record['name']!r} in {path}"
            )
        channels = tuple(record["channels"])
        rotation_axes = sorted(
            channel[0].lower()
            for channel in channels
            if channel.lower().endswith("rotation")
        )
        position_axes = sorted(
            channel[0].lower()
            for channel in channels
            if channel.lower().endswith("position")
        )
        if rotation_axes not in ([], ["x", "y", "z"]):
            raise Pz312AuditError(
                f"independent invalid rotation axes at {record['name']!r} in {path}"
            )
        if position_axes not in ([], ["x", "y", "z"]):
            raise Pz312AuditError(
                f"independent invalid position axes at {record['name']!r} in {path}"
            )
        joints.append(
            _IndependentBvhJoint(
                name=str(record["name"]),
                parent=parent,
                node_kind=str(record["node_kind"]),
                offset=tuple(float(value) for value in record["offset"]),
                channels=channels,
                channels_declared=bool(record["channels_declared"]),
            )
        )

    frames_line, frames_text = next_nonempty()
    frames_match = re.fullmatch(r"Frames\s*:\s*(\d+)\s*", frames_text, re.IGNORECASE)
    if frames_match is None or int(frames_match.group(1)) <= 0:
        raise Pz312AuditError(
            f"independent invalid Frames header at {path}:{frames_line}"
        )
    frames = int(frames_match.group(1))
    time_line, time_text = next_nonempty()
    time_match = re.fullmatch(
        r"Frame\s+Time\s*:\s*([^\s]+)\s*", time_text, re.IGNORECASE
    )
    if time_match is None:
        raise Pz312AuditError(
            f"independent invalid Frame Time header at {path}:{time_line}"
        )
    try:
        frame_time = float(time_match.group(1))
    except ValueError as exc:
        raise Pz312AuditError(
            f"independent nonnumeric Frame Time at {path}:{time_line}"
        ) from exc
    if not math.isfinite(frame_time) or frame_time <= 0.0:
        raise Pz312AuditError(
            f"independent nonpositive Frame Time at {path}:{time_line}"
        )

    numeric_tokens = [
        token
        for _, value in lines[cursor:]
        if value
        for token in value.split()
    ]
    channel_count = sum(len(joint.channels) for joint in joints)
    expected = frames * channel_count
    if len(numeric_tokens) != expected:
        raise Pz312AuditError(
            f"independent numeric token count mismatch for {path}: "
            f"{len(numeric_tokens)} != {expected}"
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        values = np.fromstring(" ".join(numeric_tokens), sep=" ", dtype=np.float64)
    if values.size != expected:
        raise Pz312AuditError(
            f"independent numeric count mismatch for {path}: {values.size} != {expected}"
        )
    values = values.reshape(frames, channel_count)
    if not np.isfinite(values).all():
        raise Pz312AuditError(f"independent numeric stream is non-finite: {path}")
    return _IndependentBvh(
        joints=tuple(joints),
        frames=frames,
        frame_time=frame_time,
        values=values,
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )


def independent_scipy_bvh_check(
    path: str | Path,
    *,
    rig_record: Mapping[str, Any],
    parsed: ParsedSourceMotion,
) -> dict[str, float | int | str]:
    """Independently decode hierarchy, channels, and retained Euler samples."""
    source = _absolute_no_resolve(path)
    independent_bvh = _parse_independent_bvh_bytes(source.read_bytes(), path=source)
    joint_map = rig_record["joint_map"]
    retained_names = tuple(str(value) for value in joint_map["btjd_joint_names"])
    retained_parents = tuple(int(value) for value in joint_map["btjd_parents"])
    expected_kinds = tuple(str(value) for value in joint_map["rotation_source_kind"])
    lookup = {joint.name: index for index, joint in enumerate(independent_bvh.joints)}
    if len(lookup) != len(independent_bvh.joints):
        raise Pz312AuditError(f"independent joint-name ambiguity in {source}")
    try:
        source_indices = tuple(lookup[name] for name in retained_names)
    except KeyError as exc:
        raise Pz312AuditError(
            f"independent retained joint is absent in {source}: {exc.args[0]}"
        ) from exc
    for retained_index in range(1, len(retained_names)):
        source_parent = independent_bvh.joints[source_indices[retained_index]].parent
        expected_parent = source_indices[retained_parents[retained_index]]
        if source_parent != expected_parent:
            raise Pz312AuditError(
                f"independent retained edge is not direct at "
                f"{retained_names[retained_index]!r} in {source}"
            )
    starts: list[int] = []
    cursor = 0
    for joint in independent_bvh.joints:
        starts.append(cursor)
        cursor += len(joint.channels)
    if cursor != independent_bvh.channel_count:
        raise Pz312AuditError(f"independent channel cursor drifted for {source}")
    start_frame = 0
    end_frame = int(parsed.local_rotations.shape[0])
    if end_frame != int(independent_bvh.frames):
        raise Pz312AuditError(
            f"independent audit requires the complete source clip: "
            f"{end_frame} != {independent_bvh.frames} for {source}"
        )
    rotation_max_abs = 0.0
    root_position = np.zeros((independent_bvh.frames, 3), dtype=np.float64)
    for retained_index, source_index in enumerate(source_indices):
        joint = independent_bvh.joints[source_index]
        block = independent_bvh.values[
            :, starts[source_index] : starts[source_index] + len(joint.channels)
        ]
        rotation_axes: list[str] = []
        rotation_columns: list[int] = []
        position_axes: list[str] = []
        for channel_index, channel in enumerate(joint.channels):
            lowered = channel.lower()
            if retained_index == 0 and lowered.endswith("position"):
                axis = "xyz".find(lowered[0])
                if axis < 0:
                    raise Pz312AuditError(
                        f"independent unsupported root position channel {channel}"
                    )
                root_position[:, axis] = block[:, channel_index]
                position_axes.append(lowered[0])
            if lowered.endswith("rotation"):
                rotation_axes.append(lowered[0])
                rotation_columns.append(channel_index)
        if retained_index == 0 and sorted(position_axes) != ["x", "y", "z"]:
            raise Pz312AuditError(
                f"independent root position layout is not XYZ at {joint.name}"
            )
        if expected_kinds[retained_index] == "fixed_dof":
            if rotation_axes:
                raise Pz312AuditError(
                    f"independent fixed joint unexpectedly rotates at {joint.name}"
                )
            independent = np.broadcast_to(
                np.eye(3, dtype=np.float64), (end_frame - start_frame, 3, 3)
            )
        elif sorted(rotation_axes) != ["x", "y", "z"] or len(rotation_axes) != 3:
            raise Pz312AuditError(
                f"independent retained rotation layout is not XYZ at {joint.name}"
            )
        else:
            angles = block[:, rotation_columns]
            independent = Rotation.from_euler(
                "".join(rotation_axes).upper(), angles, degrees=True
            ).as_matrix()[start_frame:end_frame]
        observed = np.asarray(parsed.local_rotations[:, retained_index], dtype=np.float64)
        rotation_max_abs = max(
            rotation_max_abs, float(np.max(np.abs(independent - observed)))
        )
    root_position = root_position[start_frame:end_frame]
    root_max_abs = float(np.max(np.abs(root_position - parsed.root_translation)))
    if rotation_max_abs > DUAL_ROTATION_MAX_ABS:
        raise Pz312AuditError(
            f"independent SciPy rotation disagreement {rotation_max_abs} > "
            f"{DUAL_ROTATION_MAX_ABS} for {source}"
        )
    if root_max_abs > DUAL_ROOT_MAX_ABS:
        raise Pz312AuditError(
            f"independent root disagreement {root_max_abs} > {DUAL_ROOT_MAX_ABS} "
            f"for {source}"
        )
    return {
        "independent_decoder": INDEPENDENT_DECODER_ID,
        "independent_source_sha256": independent_bvh.source_sha256,
        "independent_rotation_max_abs": rotation_max_abs,
        "independent_root_translation_max_abs": root_max_abs,
        "independent_frame_count": int(end_frame - start_frame),
        "independent_frame_time_src": float(independent_bvh.frame_time),
        "independent_fps_src": float(1.0 / independent_bvh.frame_time),
        "independent_source_joint_count": len(independent_bvh.joints),
        "independent_source_channel_count": independent_bvh.channel_count,
        "independent_rotation_layout_sha256": (
            independent_bvh.rotation_layout_sha256()
        ),
        "independent_rest_layout_sha256": independent_bvh.rest_layout_sha256(),
    }


def _motion_energy(parsed: ParsedSourceMotion) -> dict[str, float]:
    frame_count = int(parsed.local_rotations.shape[0])
    if frame_count < 2:
        return {
            "root_speed_rms_norm_per_s": 0.0,
            "rotation_speed_rms_rad_per_s": 0.0,
            "dynamic_score": 0.0,
        }
    root_steps = np.diff(parsed.root_translation, axis=0) * float(parsed.fps)
    root_speed = np.linalg.norm(root_steps, axis=-1) / float(parsed.s_rig)
    previous = np.swapaxes(parsed.local_rotations[:-1], -1, -2)
    relative = np.matmul(previous, parsed.local_rotations[1:])
    cosine = np.clip(
        (np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5,
        -1.0,
        1.0,
    )
    angular_speed = np.arccos(cosine) * float(parsed.fps)
    root_rms = float(np.sqrt(np.mean(np.square(root_speed), dtype=np.float64)))
    rotation_rms = float(
        np.sqrt(np.mean(np.square(angular_speed), dtype=np.float64))
    )
    return {
        "root_speed_rms_norm_per_s": root_rms,
        "rotation_speed_rms_rad_per_s": rotation_rms,
        "dynamic_score": root_rms + rotation_rms,
    }


_WORKER_RIGS: dict[str, dict[str, Any]] = {}
_WORKER_SKELETONS: dict[str, SkeletonData] = {}
_WORKER_PZ_ROOT: Path | None = None
_WORKER_THREAD_LIMITER: Any = None


def _initialize_worker(
    rigs: dict[str, dict[str, Any]],
    skeletons: dict[str, SkeletonData],
    pz_root: str,
) -> None:
    global _WORKER_RIGS, _WORKER_SKELETONS, _WORKER_PZ_ROOT, _WORKER_THREAD_LIMITER
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    from threadpoolctl import threadpool_limits

    _WORKER_THREAD_LIMITER = threadpool_limits(limits=1)
    _WORKER_RIGS = rigs
    _WORKER_SKELETONS = skeletons
    _WORKER_PZ_ROOT = Path(pz_root)


def _audit_one_clip(task: Mapping[str, Any]) -> dict[str, Any]:
    clip_id = str(task["clip_id"])
    rig_id = str(task["rig_id"])
    result: dict[str, Any] = {
        "audit_version": PZ312_AUDIT_VERSION,
        "clip_id": clip_id,
        "rig_id": rig_id,
        "source_family": "planetzoo",
        "topology_family": task["topology_family"],
        "split": task.get("split"),
        "parent_status": task.get("parent_status"),
    }
    try:
        if _WORKER_PZ_ROOT is None:
            raise Pz312AuditError("audit worker was not initialized")
        source = _absolute_no_resolve(str(task["source_path"]))
        if source.parent != _WORKER_PZ_ROOT:
            raise Pz312AuditError(f"source is not a direct child of PZ root: {source}")
        if source.is_symlink():
            raise Pz312AuditError(f"source is not a regular non-symlink file: {source}")
        if source.name != f"{clip_id}.bvh":
            raise Pz312AuditError(f"clip/source filename mismatch: {source.name}")
        before = source.lstat()
        if not stat_module.S_ISREG(before.st_mode):
            raise Pz312AuditError(f"source is not a regular file: {source}")
        if int(before.st_nlink) != 1 or int(task["source_nlink"]) != 1:
            raise Pz312AuditError(f"source has a hard-link alias: {clip_id}")
        if int(before.st_size) != int(task["file_size_bytes"]):
            raise Pz312AuditError(f"source size drifted for {clip_id}")
        if int(before.st_mtime_ns) != int(task["mtime_ns"]):
            raise Pz312AuditError(f"source mtime drifted for {clip_id}")
        if int(before.st_dev) != int(task["source_device"]):
            raise Pz312AuditError(f"source device drifted for {clip_id}")
        if int(before.st_ino) != int(task["source_inode"]):
            raise Pz312AuditError(f"source inode drifted for {clip_id}")
        rig = _WORKER_RIGS[rig_id]
        skeleton = _WORKER_SKELETONS[rig_id]
        joint_map = rig["joint_map"]
        parsed = parse_bvh_source(
            source,
            retained_names=joint_map["btjd_joint_names"],
            retained_parents=joint_map["btjd_parents"],
            expected_rotation_kinds=joint_map["rotation_source_kind"],
            frame_slice=task["slice_frames"],
            rest_path=source,
            rest_mode=PLANETZOO_REST_MODE,
            family="planetzoo",
        )
        closure = validate_planetzoo_parsed_against_skeleton(parsed, skeleton)
        if int(parsed.local_rotations.shape[0]) != int(task["T_src"]):
            raise Pz312AuditError(f"parent-manifest frame count drifted for {clip_id}")
        if int(parsed.local_rotations.shape[1]) != int(task["retained_joint_count"]):
            raise Pz312AuditError(f"parent-manifest retained J drifted for {clip_id}")
        if int(parsed.diagnostics["source_full_joint_count"]) != int(
            task["source_joint_count"]
        ):
            raise Pz312AuditError(f"parent-manifest source J drifted for {clip_id}")
        if int(parsed.diagnostics["source_full_frame_count"]) != int(task["T_src"]):
            raise Pz312AuditError(f"production full frame count drifted for {clip_id}")
        if abs(float(parsed.fps) - float(task["fps_src"])) > 1e-12:
            raise Pz312AuditError(f"parent-manifest source FPS drifted for {clip_id}")
        if (
            parsed.diagnostics["source_rotation_layout_sha256"]
            != task["rotation_layout_sha256"]
        ):
            raise Pz312AuditError(
                f"parent-manifest rotation layout drifted for {clip_id}"
            )
        dual = independent_scipy_bvh_check(source, rig_record=rig, parsed=parsed)
        if int(dual["independent_frame_count"]) != int(task["T_src"]):
            raise Pz312AuditError(f"independent frame count drifted for {clip_id}")
        if int(dual["independent_source_joint_count"]) != int(
            task["source_joint_count"]
        ):
            raise Pz312AuditError(f"independent source J drifted for {clip_id}")
        if int(dual["independent_source_channel_count"]) != int(
            task["source_channel_count"]
        ):
            raise Pz312AuditError(f"independent channel count drifted for {clip_id}")
        if abs(float(dual["independent_frame_time_src"]) - float(task["frame_time_src"])) > 1e-15:
            raise Pz312AuditError(f"independent Frame Time drifted for {clip_id}")
        if abs(float(dual["independent_fps_src"]) - float(task["fps_src"])) > 1e-12:
            raise Pz312AuditError(f"independent FPS drifted for {clip_id}")
        if dual["independent_rotation_layout_sha256"] != task["rotation_layout_sha256"]:
            raise Pz312AuditError(f"independent rotation layout drifted for {clip_id}")
        if (
            dual["independent_rest_layout_sha256"]
            != parsed.diagnostics["source_rest_layout_sha256"]
        ):
            raise Pz312AuditError(f"independent rest layout drifted for {clip_id}")
        after = source.lstat()
        if (
            int(after.st_size) != int(before.st_size)
            or int(after.st_mtime_ns) != int(before.st_mtime_ns)
            or int(after.st_dev) != int(before.st_dev)
            or int(after.st_ino) != int(before.st_ino)
            or int(after.st_nlink) != 1
            or not stat_module.S_ISREG(after.st_mode)
        ):
            raise Pz312AuditError(f"source changed during audit: {clip_id}")
        if _sha256_file(source) != dual["independent_source_sha256"]:
            raise Pz312AuditError(f"source bytes changed during audit: {clip_id}")
        result.update(
            {
                "status": "pass",
                "reason_codes": [],
                "source_path": str(source),
                "source_sha256": dual["independent_source_sha256"],
                "source_size_bytes": int(after.st_size),
                "source_mtime_ns": int(after.st_mtime_ns),
                "source_device": int(after.st_dev),
                "source_inode": int(after.st_ino),
                "source_nlink": int(after.st_nlink),
                "slice_frames": [int(value) for value in task["slice_frames"]],
                "T_src": int(parsed.local_rotations.shape[0]),
                "J_phys": int(parsed.local_rotations.shape[1]),
                "frame_time_src": float(task["frame_time_src"]),
                "fps_src": float(parsed.fps),
                "source_joint_count": int(
                    parsed.diagnostics["source_full_joint_count"]
                ),
                "source_channel_count": int(
                    dual["independent_source_channel_count"]
                ),
                "rotation_layout_sha256": parsed.diagnostics[
                    "source_rotation_layout_sha256"
                ],
                "rest_layout_sha256": parsed.diagnostics[
                    "source_rest_layout_sha256"
                ],
                "metrics": {**closure, **dual, **_motion_energy(parsed)},
            }
        )
    except Exception as exc:  # noqa: BLE001
        result.update(
            {
                "status": "reject",
                "reason_codes": ["PZ_EXHAUSTIVE_SOURCE_AUDIT_FAILURE"],
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    return result


def _audit_chunk(tasks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [_audit_one_clip(task) for task in tasks]


def _audit_chunk_with_process_status(
    tasks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Run one real worker chunk and fail if that child is over-threaded."""
    records = _audit_chunk(tasks)
    process_status = _validate_worker_process_status(_worker_process_status())
    return {
        "records": records,
        "worker_process_status": process_status,
    }


def _unpack_worker_chunk_result(value: Any) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not isinstance(value, Mapping) or set(value) != {
        "records",
        "worker_process_status",
    }:
        raise Pz312AuditError("spawned audit chunk result schema drifted")
    records = value["records"]
    if not isinstance(records, list):
        raise Pz312AuditError("spawned audit chunk records are not a list")
    return records, _validate_worker_process_status(value["worker_process_status"])


def _task_core(
    record: Mapping[str, Any],
    pz_root: Path,
    rig_record: Mapping[str, Any],
) -> dict[str, Any]:
    source = record.get("source")
    if not isinstance(source, Mapping):
        raise Pz312AuditError(f"{record.get('clip_id')}: source record is absent")
    clip_id = str(record["clip_id"])
    if source.get("family") != "planetzoo":
        raise Pz312AuditError(f"{clip_id}: source family is not PlanetZoo")
    path = _absolute_no_resolve(str(source["path"]))
    if path.parent != pz_root or path.name != f"{clip_id}.bvh":
        raise Pz312AuditError(f"{clip_id}: path is not an exact direct PZ-root child")
    if path.is_symlink():
        raise Pz312AuditError(f"{clip_id}: source is a symlink")
    try:
        stat = path.lstat()
    except OSError as exc:
        raise Pz312AuditError(f"{clip_id}: cannot lstat source: {exc}") from exc
    if not stat_module.S_ISREG(stat.st_mode):
        raise Pz312AuditError(f"{clip_id}: source is not a regular file")
    if int(stat.st_nlink) != 1:
        raise Pz312AuditError(
            f"{clip_id}: source has hard-link aliases (st_nlink={stat.st_nlink})"
        )
    if path.resolve(strict=True) != path:
        raise Pz312AuditError(f"{clip_id}: source path contains a symlink component")
    frame_slice = [int(value) for value in source["slice_frames"]]
    if frame_slice != [0, int(source["T_src"])]:
        raise Pz312AuditError(f"{clip_id}: PZ audit requires the complete source clip")
    frame_time = float(source["frame_time_src"])
    fps = float(source["fps_src"])
    if (
        not math.isfinite(frame_time)
        or frame_time <= 0.0
        or not math.isfinite(fps)
        or fps <= 0.0
        or abs(fps - 1.0 / frame_time) > 1e-12
    ):
        raise Pz312AuditError(f"{clip_id}: parent timing fields are inconsistent")
    if int(stat.st_size) != int(source["file_size_bytes"]):
        raise Pz312AuditError(f"{clip_id}: parent source size is stale")
    if int(stat.st_mtime_ns) != int(source["mtime_ns"]):
        raise Pz312AuditError(f"{clip_id}: parent source mtime is stale")
    return {
        "clip_id": clip_id,
        "rig_id": str(record["rig_id"]),
        "topology_family": str(record["topology_family"]),
        "split": record.get("split"),
        "parent_status": record.get("status"),
        "source_path": str(path),
        "slice_frames": frame_slice,
        "file_size_bytes": int(source["file_size_bytes"]),
        "mtime_ns": int(source["mtime_ns"]),
        "source_device": int(stat.st_dev),
        "source_inode": int(stat.st_ino),
        "source_nlink": int(stat.st_nlink),
        "T_src": int(source["T_src"]),
        "frame_time_src": frame_time,
        "fps_src": fps,
        "source_joint_count": int(source["source_joint_count"]),
        "source_channel_count": int(source["source_channel_count"]),
        "retained_joint_count": len(rig_record["joint_map"]["btjd_joint_names"]),
        "rotation_layout_sha256": str(source["rotation_layout_sha256"]),
    }


def _assert_manifest_disk_bijection(
    pz_root: Path,
    tasks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Require a flat, regular, non-linked source tree exactly equal to tasks."""
    root = _absolute_no_resolve(pz_root)
    if root.is_symlink() or not root.is_dir() or root.resolve(strict=True) != root:
        raise Pz312AuditError(f"invalid or symlinked PZ source root: {root}")
    expected: dict[str, Mapping[str, Any]] = {}
    for task in tasks:
        name = Path(str(task["source_path"])).name
        if name in expected:
            raise Pz312AuditError(f"duplicate manifest source basename: {name}")
        expected[name] = task
    observed: dict[str, os.stat_result] = {}
    inode_owner: dict[tuple[int, int], str] = {}
    try:
        with os.scandir(root) as iterator:
            for entry in iterator:
                if entry.is_symlink():
                    raise Pz312AuditError(f"symlink in PZ source root: {entry.name}")
                stat = entry.stat(follow_symlinks=False)
                if not stat_module.S_ISREG(stat.st_mode) or not entry.name.endswith(".bvh"):
                    raise Pz312AuditError(
                        f"unexpected non-BVH/non-regular PZ root entry: {entry.name}"
                    )
                if int(stat.st_nlink) != 1:
                    raise Pz312AuditError(
                        f"hard-linked PZ source (st_nlink={stat.st_nlink}): "
                        f"{entry.name}"
                    )
                inode_key = (int(stat.st_dev), int(stat.st_ino))
                if inode_key in inode_owner:
                    raise Pz312AuditError(
                        f"hard-linked PZ sources: {inode_owner[inode_key]} and {entry.name}"
                    )
                inode_owner[inode_key] = entry.name
                observed[entry.name] = stat
    except OSError as exc:
        raise Pz312AuditError(f"cannot enumerate PZ source root {root}: {exc}") from exc
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))[:20]
        extra = sorted(set(observed) - set(expected))[:20]
        raise Pz312AuditError(
            f"PZ manifest/disk scope mismatch: missing={missing}, extra={extra}, "
            f"manifest={len(expected)}, disk={len(observed)}"
        )
    snapshot: list[dict[str, Any]] = []
    for name in sorted(expected):
        task = expected[name]
        stat = observed[name]
        actual_path = root / name
        if _absolute_no_resolve(task["source_path"]) != actual_path:
            raise Pz312AuditError(f"manifest source path drifted for {name}")
        checks = {
            "file_size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "source_device": int(stat.st_dev),
            "source_inode": int(stat.st_ino),
            "source_nlink": int(stat.st_nlink),
        }
        for field, actual in checks.items():
            if int(task[field]) != actual:
                raise Pz312AuditError(
                    f"PZ source snapshot {field} drifted for {name}: "
                    f"{actual} != {task[field]}"
                )
        snapshot.append({"name": name, **checks})
    return {
        "entry_count": len(snapshot),
        "snapshot_sha256": hashlib.sha256(_canonical_json(snapshot)).hexdigest(),
    }


def _load_pz_scope(
    manifest_root: Path, pz_root: Path
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    rigs: dict[str, dict[str, Any]] = {}
    with (manifest_root / "rigs.jsonl").open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            if record.get("source_family") != "planetzoo":
                continue
            rig_id = str(record["rig_id"])
            if rig_id in rigs:
                raise Pz312AuditError(
                    f"duplicate PZ rig {rig_id} at rigs.jsonl:{line_number}"
                )
            rigs[rig_id] = record
    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    with (manifest_root / "clips.jsonl").open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            if record.get("source", {}).get("family") != "planetzoo":
                continue
            rig_id = str(record["rig_id"])
            if rig_id not in rigs:
                raise Pz312AuditError(
                    f"{record.get('clip_id')}: rig is absent from rigs.jsonl"
                )
            task = _task_core(record, pz_root, rigs[rig_id])
            clip_id = str(task["clip_id"])
            if clip_id in seen:
                raise Pz312AuditError(
                    f"duplicate PZ clip {clip_id} at clips.jsonl:{line_number}"
                )
            seen.add(clip_id)
            tasks.append(task)
    tasks.sort(key=lambda item: str(item["clip_id"]))
    if len(rigs) != EXPECTED_PZ_RIG_COUNT:
        raise Pz312AuditError(
            f"PZ rig scope drifted: {len(rigs)} != {EXPECTED_PZ_RIG_COUNT}"
        )
    if len(tasks) != EXPECTED_PZ_CLIP_COUNT:
        raise Pz312AuditError(
            f"PZ clip scope drifted: {len(tasks)} != {EXPECTED_PZ_CLIP_COUNT}"
        )
    rig_counts = Counter(str(task["rig_id"]) for task in tasks)
    if set(rig_counts) != set(rigs) or min(rig_counts.values(), default=0) <= 0:
        raise Pz312AuditError("PZ clip scope does not cover every one of the 311 rigs")
    _assert_manifest_disk_bijection(pz_root, tasks)
    return rigs, tasks


def _chunk_sha256(tasks: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(_canonical_json(list(tasks))).hexdigest()


def _validate_pass_record_against_task(
    record: Mapping[str, Any], task: Mapping[str, Any]
) -> None:
    if int(task["source_nlink"]) != 1:
        raise Pz312AuditError(f"task source has a hard-link alias: {task['clip_id']}")
    if set(record) != PASS_RECORD_KEYS:
        raise Pz312AuditError(
            f"cached PASS schema drifted for {task['clip_id']}: "
            f"missing={sorted(PASS_RECORD_KEYS - set(record))}, "
            f"extra={sorted(set(record) - PASS_RECORD_KEYS)}"
        )
    exact_fields = {
        "audit_version": PZ312_AUDIT_VERSION,
        "clip_id": str(task["clip_id"]),
        "rig_id": str(task["rig_id"]),
        "source_family": "planetzoo",
        "topology_family": task["topology_family"],
        "split": task.get("split"),
        "parent_status": task.get("parent_status"),
        "status": "pass",
        "source_path": str(task["source_path"]),
        "source_size_bytes": int(task["file_size_bytes"]),
        "source_mtime_ns": int(task["mtime_ns"]),
        "source_device": int(task["source_device"]),
        "source_inode": int(task["source_inode"]),
        "source_nlink": int(task["source_nlink"]),
        "slice_frames": [int(value) for value in task["slice_frames"]],
        "T_src": int(task["T_src"]),
        "J_phys": int(task["retained_joint_count"]),
        "source_joint_count": int(task["source_joint_count"]),
        "source_channel_count": int(task["source_channel_count"]),
        "rotation_layout_sha256": str(task["rotation_layout_sha256"]),
    }
    for field, expected in exact_fields.items():
        if record.get(field) != expected:
            raise Pz312AuditError(
                f"cached record {field} drifted for {task['clip_id']}: "
                f"{record.get(field)!r} != {expected!r}"
            )
    if record.get("reason_codes") != []:
        raise Pz312AuditError(f"cached PASS has reason codes: {task['clip_id']}")
    for field, expected, tolerance in (
        ("frame_time_src", float(task["frame_time_src"]), 1e-15),
        ("fps_src", float(task["fps_src"]), 1e-12),
    ):
        value = float(record[field])
        if not math.isfinite(value) or abs(value - expected) > tolerance:
            raise Pz312AuditError(
                f"cached record {field} drifted for {task['clip_id']}"
            )
    source_sha256 = str(record.get("source_sha256", ""))
    rotation_layout_sha256 = str(record.get("rotation_layout_sha256", ""))
    rest_layout_sha256 = str(record.get("rest_layout_sha256", ""))
    if any(
        re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in (source_sha256, rotation_layout_sha256, rest_layout_sha256)
    ):
        raise Pz312AuditError(
            f"cached source/layout SHA is invalid: {task['clip_id']}"
        )
    if int(record["source_size_bytes"]) <= 0:
        raise Pz312AuditError(f"cached source size is invalid: {task['clip_id']}")
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        raise Pz312AuditError(f"cached metrics are absent: {task['clip_id']}")
    if set(metrics) != PASS_METRIC_KEYS:
        raise Pz312AuditError(
            f"cached metric schema drifted for {task['clip_id']}: "
            f"missing={sorted(PASS_METRIC_KEYS - set(metrics))}, "
            f"extra={sorted(set(metrics) - PASS_METRIC_KEYS)}"
        )
    if (
        metrics.get("independent_source_sha256") != source_sha256
        or metrics.get("independent_rotation_layout_sha256")
        != rotation_layout_sha256
        or metrics.get("independent_rest_layout_sha256") != rest_layout_sha256
    ):
        raise Pz312AuditError(
            f"cached independent/source-layout evidence mismatch: {task['clip_id']}"
        )
    if (
        metrics.get("independent_decoder") != INDEPENDENT_DECODER_ID
        or metrics.get("planetzoo_stage2_contract")
        != PLANETZOO_FIXED_RIG_VERSION
    ):
        raise Pz312AuditError(
            f"cached decoder/fixed-rig contract drifted: {task['clip_id']}"
        )
    for name in (
        "planetzoo_per_clip_declared_offset_exact",
        "planetzoo_per_clip_rotation_layout_exact",
        "planetzoo_per_clip_rest_layout_exact",
        "planetzoo_root_translation_exact",
    ):
        if type(metrics[name]) is not int or metrics[name] != 1:
            raise Pz312AuditError(
                f"cached exact-closure flag failed ({name}): {task['clip_id']}"
            )
    exact_metric_counts = {
        "independent_frame_count": int(task["T_src"]),
        "independent_source_joint_count": int(task["source_joint_count"]),
        "independent_source_channel_count": int(task["source_channel_count"]),
    }
    for name, expected in exact_metric_counts.items():
        if type(metrics[name]) is not int or metrics[name] != expected:
            raise Pz312AuditError(
                f"cached independent count drifted ({name}): {task['clip_id']}"
            )
    for name, expected, tolerance in (
        ("independent_frame_time_src", float(task["frame_time_src"]), 1e-15),
        ("independent_fps_src", float(task["fps_src"]), 1e-12),
    ):
        value = float(metrics[name])
        if not math.isfinite(value) or abs(value - expected) > tolerance:
            raise Pz312AuditError(
                f"cached independent timing drifted ({name}): {task['clip_id']}"
            )
    numeric = {
        name: float(metrics[name])
        for name in (
            "independent_rotation_max_abs",
            "independent_root_translation_max_abs",
            "planetzoo_fixed_fk_source_position_max_norm",
            "planetzoo_fixed_fk_source_position_mpjpe_norm",
            "root_speed_rms_norm_per_s",
            "rotation_speed_rms_rad_per_s",
            "dynamic_score",
        )
    }
    if any(not math.isfinite(value) or value < 0.0 for value in numeric.values()):
        raise Pz312AuditError(
            f"cached metric is nonfinite/negative: {task['clip_id']}"
        )
    if numeric["independent_rotation_max_abs"] > DUAL_ROTATION_MAX_ABS:
        raise Pz312AuditError(
            f"cached independent rotation threshold failed: {task['clip_id']}"
        )
    if numeric["independent_root_translation_max_abs"] > DUAL_ROOT_MAX_ABS:
        raise Pz312AuditError(
            f"cached independent root threshold failed: {task['clip_id']}"
        )
    fk_max = numeric["planetzoo_fixed_fk_source_position_max_norm"]
    fk_mean = numeric["planetzoo_fixed_fk_source_position_mpjpe_norm"]
    if fk_max > PLANETZOO_SOURCE_POSITION_MAX_NORM or fk_mean > fk_max:
        raise Pz312AuditError(
            f"cached fixed-FK closure failed: {task['clip_id']}"
        )
    expected_dynamic = (
        numeric["root_speed_rms_norm_per_s"]
        + numeric["rotation_speed_rms_rad_per_s"]
    )
    if numeric["dynamic_score"] != expected_dynamic:
        raise Pz312AuditError(
            f"cached dynamic-score composition drifted: {task['clip_id']}"
        )


def _validate_fresh_record_against_task(
    record: Mapping[str, Any], task: Mapping[str, Any]
) -> None:
    if record.get("status") == "pass":
        _validate_pass_record_against_task(record, task)
        return
    expected = {
        "audit_version": PZ312_AUDIT_VERSION,
        "clip_id": str(task["clip_id"]),
        "rig_id": str(task["rig_id"]),
        "source_family": "planetzoo",
        "topology_family": task["topology_family"],
        "split": task.get("split"),
        "parent_status": task.get("parent_status"),
        "status": "reject",
        "reason_codes": ["PZ_EXHAUSTIVE_SOURCE_AUDIT_FAILURE"],
    }
    if any(record.get(name) != value for name, value in expected.items()):
        raise Pz312AuditError(f"malformed reject record: {task['clip_id']}")
    if not isinstance(record.get("error_type"), str) or not isinstance(
        record.get("error"), str
    ):
        raise Pz312AuditError(f"reject record lacks error evidence: {task['clip_id']}")


def _verify_live_source_record(record: Mapping[str, Any], pz_root: Path) -> None:
    """Re-hash one cached/final PASS and bind it to stable lstat identity."""
    source = _absolute_no_resolve(str(record["source_path"]))
    root = _absolute_no_resolve(pz_root)
    if source.parent != root or source.name != f"{record['clip_id']}.bvh":
        raise Pz312AuditError(f"live source path escaped scope: {source}")
    if source.is_symlink():
        raise Pz312AuditError(f"live source became a symlink: {source}")
    before = source.lstat()
    if not stat_module.S_ISREG(before.st_mode):
        raise Pz312AuditError(f"live source is not regular: {source}")
    if int(before.st_nlink) != 1 or int(record["source_nlink"]) != 1:
        raise Pz312AuditError(f"live source has a hard-link alias: {source}")
    expected_stat = {
        "st_size": int(record["source_size_bytes"]),
        "st_mtime_ns": int(record["source_mtime_ns"]),
        "st_dev": int(record["source_device"]),
        "st_ino": int(record["source_inode"]),
        "st_nlink": int(record["source_nlink"]),
    }
    for field, expected in expected_stat.items():
        if int(getattr(before, field)) != expected:
            raise Pz312AuditError(
                f"live source {field} drifted for {record['clip_id']}"
            )
    observed_sha256 = _sha256_file(source)
    after = source.lstat()
    if any(
        int(getattr(after, field)) != int(getattr(before, field))
        for field in expected_stat
    ) or not stat_module.S_ISREG(after.st_mode):
        raise Pz312AuditError(f"live source changed while re-hashing: {source}")
    if observed_sha256 != record["source_sha256"]:
        raise Pz312AuditError(f"live source SHA drifted for {record['clip_id']}")


def _load_chunk_payload(
    path: Path,
    *,
    tasks: Sequence[Mapping[str, Any]],
    authority_sha256: str,
) -> list[dict[str, Any]] | None:
    try:
        _regular_single_link_stat(path, label="resumable chunk")
        payload = _load_json(path)
        records = payload["records"]
        if (
            set(payload)
            != {
                "audit_version",
                "authority_sha256",
                "task_sha256",
                "records_sha256",
                "worker_process_status",
                "records",
            }
            or payload.get("audit_version") != PZ312_AUDIT_VERSION
            or payload.get("authority_sha256") != authority_sha256
            or payload.get("task_sha256") != _chunk_sha256(tasks)
            or not isinstance(records, list)
            or payload.get("records_sha256") != _records_sha256(records)
            or [record.get("clip_id") for record in records]
            != [task["clip_id"] for task in tasks]
            or any(not isinstance(record, Mapping) for record in records)
        ):
            return None
        status = payload.get("worker_process_status")
        if status is not None:
            _validate_worker_process_status(status)
        return [dict(record) for record in records]
    except Exception:  # noqa: BLE001
        return None


def _load_valid_chunk(
    path: Path,
    *,
    tasks: Sequence[Mapping[str, Any]],
    authority_sha256: str,
    pz_root: Path,
) -> list[dict[str, Any]] | None:
    """Load a resumable chunk only after semantic and current-byte validation."""
    records = _load_chunk_payload(
        path, tasks=tasks, authority_sha256=authority_sha256
    )
    if records is None:
        return None
    try:
        for record, task in zip(records, tasks, strict=True):
            _validate_pass_record_against_task(record, task)
            _verify_live_source_record(record, pz_root)
        return records
    except Exception:  # noqa: BLE001
        return None


def _source_snapshot_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: str(item["clip_id"])):
        core = {
            "clip_id": str(record["clip_id"]),
            "rig_id": str(record["rig_id"]),
            "source_path": str(record["source_path"]),
            "source_sha256": str(record["source_sha256"]),
            "source_size_bytes": int(record["source_size_bytes"]),
            "source_mtime_ns": int(record["source_mtime_ns"]),
            "source_device": int(record["source_device"]),
            "source_inode": int(record["source_inode"]),
            "source_nlink": int(record["source_nlink"]),
        }
        digest.update(_canonical_json(core) + b"\n")
    return digest.hexdigest()


def _revalidate_all_live_sources(
    records: Sequence[Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
    *,
    pz_root: Path,
    workers: int,
) -> dict[str, Any]:
    """Perform the mandatory corpus-wide byte recheck immediately pre-publish."""
    if [record.get("clip_id") for record in records] != [
        task["clip_id"] for task in tasks
    ]:
        raise Pz312AuditError("final source record/task ordering drifted")
    for record, task in zip(records, tasks, strict=True):
        _validate_pass_record_against_task(record, task)
    before_inventory = _assert_manifest_disk_bijection(pz_root, tasks)
    worker_count = min(max(int(workers), 1), 16)
    if worker_count == 1:
        for record in records:
            _verify_live_source_record(record, pz_root)
    else:
        queue = deque(records)
        in_flight: set[concurrent.futures.Future[None]] = set()
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            while queue and len(in_flight) < worker_count * 2:
                in_flight.add(
                    executor.submit(_verify_live_source_record, queue.popleft(), pz_root)
                )
            while in_flight:
                done, in_flight = concurrent.futures.wait(
                    in_flight,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    future.result()
                while queue and len(in_flight) < worker_count * 2:
                    in_flight.add(
                        executor.submit(
                            _verify_live_source_record, queue.popleft(), pz_root
                        )
                    )
    after_inventory = _assert_manifest_disk_bijection(pz_root, tasks)
    if before_inventory != after_inventory:
        raise Pz312AuditError("PZ disk inventory changed during final byte recheck")
    return {
        "status": "pass",
        "validated_count": len(records),
        "hash_workers": worker_count,
        "disk_inventory_snapshot_sha256": after_inventory["snapshot_sha256"],
        "source_snapshot_sha256": _source_snapshot_sha256(records),
        "completed_at_utc": _datetime.datetime.now(_datetime.UTC).isoformat(),
    }


def _select_representatives(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_rig_count: int = EXPECTED_PZ_RIG_COUNT,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("status") != "pass":
            raise Pz312AuditError("cannot select representatives from rejected clips")
        grouped[str(record["rig_id"])].append(record)
    selected: list[dict[str, Any]] = []
    candidates: dict[str, list[dict[str, Any]]] = {}
    for rig_id in sorted(grouped):
        ranked = sorted(
            grouped[rig_id],
            key=lambda record: (
                -float(record["metrics"]["dynamic_score"]),
                -int(record["T_src"]),
                str(record["clip_id"]),
            ),
        )
        top = ranked[:3]
        candidates[rig_id] = [
            {
                "clip_id": str(record["clip_id"]),
                "dynamic_score": float(record["metrics"]["dynamic_score"]),
                "T_src": int(record["T_src"]),
                "source_sha256": str(record["source_sha256"]),
            }
            for record in top
        ]
        selected.append(
            {
                "rig_id": rig_id,
                "clip_id": str(top[0]["clip_id"]),
                "dynamic_score": float(top[0]["metrics"]["dynamic_score"]),
                "T_src": int(top[0]["T_src"]),
                "source_sha256": str(top[0]["source_sha256"]),
                "selection_policy": (
                    "max normalized root-speed RMS plus local SO3 angular-speed RMS; "
                    "then longer T; then lexical clip id"
                ),
            }
        )
    if len(selected) != expected_rig_count:
        raise Pz312AuditError(
            f"representative scope drifted: {len(selected)} != {expected_rig_count}"
        )
    return {
        "audit_version": PZ312_AUDIT_VERSION,
        "selected_count": len(selected),
        "selected": selected,
        "top3_candidates_by_rig": candidates,
    }


def _task_from_pass_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Recover the exact authority task core stored in one published PASS row."""
    return {
        "clip_id": str(record["clip_id"]),
        "rig_id": str(record["rig_id"]),
        "topology_family": str(record["topology_family"]),
        "split": record.get("split"),
        "parent_status": record.get("parent_status"),
        "source_path": str(record["source_path"]),
        "slice_frames": [int(value) for value in record["slice_frames"]],
        "file_size_bytes": int(record["source_size_bytes"]),
        "mtime_ns": int(record["source_mtime_ns"]),
        "source_device": int(record["source_device"]),
        "source_inode": int(record["source_inode"]),
        "source_nlink": int(record["source_nlink"]),
        "T_src": int(record["T_src"]),
        "frame_time_src": float(record["frame_time_src"]),
        "fps_src": float(record["fps_src"]),
        "source_joint_count": int(record["source_joint_count"]),
        "source_channel_count": int(record["source_channel_count"]),
        "retained_joint_count": int(record["J_phys"]),
        "rotation_layout_sha256": str(record["rotation_layout_sha256"]),
    }


def _disk_snapshot_from_tasks(tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    snapshot = [
        {
            "name": Path(str(task["source_path"])).name,
            "file_size_bytes": int(task["file_size_bytes"]),
            "mtime_ns": int(task["mtime_ns"]),
            "source_device": int(task["source_device"]),
            "source_inode": int(task["source_inode"]),
            "source_nlink": int(task["source_nlink"]),
        }
        for task in sorted(tasks, key=lambda item: Path(str(item["source_path"])).name)
    ]
    return {
        "entry_count": len(snapshot),
        "snapshot_sha256": hashlib.sha256(_canonical_json(snapshot)).hexdigest(),
    }


def _validate_pz_audit_generation_structure(root: str | Path) -> dict[str, Any]:
    """Structural precheck only; this function can never authorize conversion."""
    generation_root = _absolute_no_resolve(root)
    if (
        not generation_root.is_dir()
        or generation_root.is_symlink()
        or generation_root.resolve(strict=True) != generation_root
    ):
        raise Pz312AuditError(f"invalid immutable audit root: {generation_root}")
    generation = _load_json(generation_root / "generation.json")
    if (
        generation.get("generation_id") != generation_root.name
        or generation.get("audit_version") != PZ312_AUDIT_VERSION
        or generation.get("status") != CANDIDATE_STATUS
        or generation.get("prototype_conversion_authorized") is not False
        or generation.get("full_conversion_authorized") is not False
    ):
        raise Pz312AuditError("audit generation metadata/authorization drifted")
    expected = generation.get("files")
    if not isinstance(expected, Mapping):
        raise Pz312AuditError("audit generation has no file closure")
    observed = _file_manifest(generation_root, require_read_only=True)
    if observed != dict(expected):
        raise Pz312AuditError("audit generation file closure/hash/size drifted")

    static_required = {
        "authority.json",
        "summary.json",
        "qa/pz_source_audit.jsonl",
        "qa/rig_audit.jsonl",
        "qa/source_snapshot_recheck.json",
        "qa/producer_worker_process_status.json",
        "selection/pz_representatives.json",
    }
    if not static_required <= set(expected):
        raise Pz312AuditError("audit generation lacks required semantic artifacts")

    authority = _load_json(generation_root / "authority.json")
    authority_sha256 = str(authority.get("authority_sha256", ""))
    authority_core = dict(authority)
    authority_core.pop("authority_sha256", None)
    if (
        re.fullmatch(r"[0-9a-f]{64}", authority_sha256) is None
        or hashlib.sha256(_canonical_json(authority_core)).hexdigest()
        != authority_sha256
        or generation.get("authority_sha256") != authority_sha256
        or authority.get("audit_version") != PZ312_AUDIT_VERSION
        or int(authority.get("expected_rig_count", -1)) != EXPECTED_PZ_RIG_COUNT
        or int(authority.get("expected_clip_count", -1)) != EXPECTED_PZ_CLIP_COUNT
    ):
        raise Pz312AuditError("audit authority hash/scope drifted")
    parent_root = _absolute_no_resolve(str(authority.get("parent_manifest_root", "")))
    parent_evidence = _validate_parent_manifest_generation(parent_root)
    if parent_evidence != authority.get("parent_inventory_generation"):
        raise Pz312AuditError("live parent immutable generation drifted")
    if generation.get("parent_manifest_root") != str(parent_root):
        raise Pz312AuditError("generation/authority parent root drifted")

    records = _load_jsonl(generation_root / "qa/pz_source_audit.jsonl")
    rig_records = _load_jsonl(generation_root / "qa/rig_audit.jsonl")
    if len(records) != EXPECTED_PZ_CLIP_COUNT:
        raise Pz312AuditError("audit QA does not contain exactly 74,522 clips")
    if len(rig_records) != EXPECTED_PZ_RIG_COUNT:
        raise Pz312AuditError("rig QA does not contain exactly 311 rigs")
    clip_ids = [str(record.get("clip_id")) for record in records]
    rig_ids = [str(record.get("rig_id")) for record in rig_records]
    if clip_ids != sorted(clip_ids) or len(set(clip_ids)) != len(clip_ids):
        raise Pz312AuditError("audit clip IDs are not unique and canonical-order")
    if rig_ids != sorted(rig_ids) or len(set(rig_ids)) != len(rig_ids):
        raise Pz312AuditError("audit rig IDs are not unique and canonical-order")
    rig_set = set(rig_ids)
    source_paths: set[str] = set()
    source_inodes: set[tuple[int, int]] = set()
    all_metrics: list[Mapping[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    pz_root = _absolute_no_resolve(str(authority.get("pz_bvh_root", "")))
    for record in records:
        if (
            record.get("audit_version") != PZ312_AUDIT_VERSION
            or record.get("status") != "pass"
            or record.get("reason_codes") != []
            or record.get("source_family") != "planetzoo"
            or str(record.get("rig_id")) not in rig_set
        ):
            raise Pz312AuditError(f"invalid source QA row: {record.get('clip_id')}")
        clip_id = str(record["clip_id"])
        source_path = str(record.get("source_path", ""))
        source = _absolute_no_resolve(source_path)
        if source.parent != pz_root or source.name != f"{clip_id}.bvh":
            raise Pz312AuditError(f"published source path escaped scope: {clip_id}")
        if source_path in source_paths:
            raise Pz312AuditError(f"duplicate published source path: {source_path}")
        source_paths.add(source_path)
        inode = (int(record["source_device"]), int(record["source_inode"]))
        if inode in source_inodes:
            raise Pz312AuditError(f"duplicate published source inode: {clip_id}")
        source_inodes.add(inode)
        if (
            re.fullmatch(r"[0-9a-f]{64}", str(record.get("source_sha256", "")))
            is None
            or re.fullmatch(
                r"[0-9a-f]{64}", str(record.get("rotation_layout_sha256", ""))
            )
            is None
            or re.fullmatch(
                r"[0-9a-f]{64}", str(record.get("rest_layout_sha256", ""))
            )
            is None
        ):
            raise Pz312AuditError(f"published source/layout hash is invalid: {clip_id}")
        if (
            int(record["source_size_bytes"]) <= 0
            or int(record["T_src"]) <= 0
            or int(record["J_phys"]) <= 0
            or int(record["source_joint_count"]) < int(record["J_phys"])
            or int(record["source_channel_count"]) <= 0
            or [int(value) for value in record["slice_frames"]]
            != [0, int(record["T_src"])]
        ):
            raise Pz312AuditError(f"published source shape/slice is invalid: {clip_id}")
        frame_time = float(record["frame_time_src"])
        fps = float(record["fps_src"])
        if (
            not math.isfinite(frame_time)
            or frame_time <= 0.0
            or not math.isfinite(fps)
            or abs(fps - 1.0 / frame_time) > 1e-12
        ):
            raise Pz312AuditError(f"published source timing is invalid: {clip_id}")
        metrics = record.get("metrics")
        if not isinstance(metrics, Mapping):
            raise Pz312AuditError(f"published metrics are absent: {clip_id}")
        if (
            metrics.get("independent_source_sha256") != record["source_sha256"]
            or metrics.get("independent_rotation_layout_sha256")
            != record["rotation_layout_sha256"]
            or int(metrics["independent_frame_count"]) != int(record["T_src"])
            or int(metrics["independent_source_joint_count"])
            != int(record["source_joint_count"])
            or int(metrics["independent_source_channel_count"])
            != int(record["source_channel_count"])
            or abs(float(metrics["independent_frame_time_src"]) - frame_time) > 1e-15
            or abs(float(metrics["independent_fps_src"]) - fps) > 1e-12
        ):
            raise Pz312AuditError(f"published independent evidence drifted: {clip_id}")
        numeric_metrics = (
            "independent_rotation_max_abs",
            "independent_root_translation_max_abs",
            "planetzoo_fixed_fk_source_position_max_norm",
            "dynamic_score",
        )
        if any(not math.isfinite(float(metrics[name])) for name in numeric_metrics):
            raise Pz312AuditError(f"published metric is nonfinite: {clip_id}")
        if (
            float(metrics["independent_rotation_max_abs"])
            > DUAL_ROTATION_MAX_ABS
            or float(metrics["independent_root_translation_max_abs"])
            > DUAL_ROOT_MAX_ABS
            or float(metrics["planetzoo_fixed_fk_source_position_max_norm"])
            > PLANETZOO_SOURCE_POSITION_MAX_NORM
        ):
            raise Pz312AuditError(f"published metric threshold failed: {clip_id}")
        all_metrics.append(metrics)
        tasks.append(_task_from_pass_record(record))

    if set(str(record["rig_id"]) for record in records) != rig_set:
        raise Pz312AuditError("source QA does not cover every audited rig")
    if _chunk_sha256(tasks) != authority.get("source_scope_sha256"):
        raise Pz312AuditError("published source task closure drifted")
    parent_rigs, parent_tasks = _load_pz_scope(parent_root, pz_root)
    if parent_tasks != tasks or set(parent_rigs) != rig_set:
        raise Pz312AuditError("published source QA drifted from the pinned parent")
    for record, task in zip(records, parent_tasks, strict=True):
        _validate_pass_record_against_task(record, task)
    disk_snapshot = _disk_snapshot_from_tasks(tasks)
    if disk_snapshot != authority.get("disk_inventory_snapshot"):
        raise Pz312AuditError("published manifest/disk snapshot evidence drifted")
    live_disk_snapshot = _assert_manifest_disk_bijection(pz_root, tasks)
    if live_disk_snapshot != disk_snapshot:
        raise Pz312AuditError("live disk scope drifted after source recheck")

    required_files = set(static_required)
    for rig_record in rig_records:
        rig_id = str(rig_record.get("rig_id"))
        if (
            rig_record.get("audit_version") != PZ312_AUDIT_VERSION
            or rig_record.get("status") != "pass"
            or not rig_id.startswith("PZ_")
            or int(rig_record.get("J_phys", 0)) <= 0
            or int(rig_record.get("J_phys", 0))
            != len(parent_rigs[rig_id]["joint_map"]["btjd_joint_names"])
            or re.fullmatch(
                r"[0-9a-f]{64}", str(rig_record.get("skeleton_sha256", ""))
            )
            is None
        ):
            raise Pz312AuditError(f"invalid rig QA row: {rig_id}")
        relpath = f"skeletons/{rig_id}.npz"
        required_files.add(relpath)
        skeleton = load_skeleton(generation_root / relpath)
        if (
            skeleton.sha256 != rig_record["skeleton_sha256"]
            or skeleton.rig_id != rig_id
            or skeleton.source_family != "planetzoo"
            or skeleton.artifact_status != "planetzoo_stage2_fixed_rig_pass"
            or len(skeleton.parents) != int(rig_record["J_phys"])
        ):
            raise Pz312AuditError(f"skeleton/rig QA drifted: {rig_id}")
    if set(expected) != required_files:
        raise Pz312AuditError("audit generation contains missing/extra semantic files")

    summary = _load_json(generation_root / "summary.json")
    selection = _load_json(generation_root / "selection/pz_representatives.json")
    recheck = _load_json(generation_root / "qa/source_snapshot_recheck.json")
    producer_workers = _load_json(
        generation_root / "qa/producer_worker_process_status.json"
    )
    if set(producer_workers) != {
        "audit_version",
        "status",
        "executor_mode",
        "chunk_count",
        "chunks_with_process_status",
        "worker_process_statuses",
    }:
        raise Pz312AuditError("producer worker evidence schema drifted")
    producer_statuses = [
        _validate_worker_process_status(value)
        for value in producer_workers.get("worker_process_statuses", [])
    ]
    audit_workers = int(authority.get("audit_workers", 0))
    audit_chunk_size = int(authority.get("audit_chunk_size", 0))
    if audit_workers <= 0 or audit_chunk_size <= 0:
        raise Pz312AuditError("producer worker configuration drifted")
    expected_chunk_count = math.ceil(EXPECTED_PZ_CLIP_COUNT / audit_chunk_size)
    if (
        producer_workers.get("audit_version") != PZ312_AUDIT_VERSION
        or producer_workers.get("status") != "pass"
        or int(producer_workers.get("chunk_count", -1)) != expected_chunk_count
        or (
            audit_workers > 1
            and (
                producer_workers.get("executor_mode") != "spawn"
                or int(producer_workers.get("chunks_with_process_status", -1))
                != expected_chunk_count
                or not producer_statuses
            )
        )
        or (
            audit_workers == 1
            and (
                producer_workers.get("executor_mode") != "in_process"
                or int(producer_workers.get("chunks_with_process_status", -1)) != 0
                or producer_statuses
            )
        )
    ):
        raise Pz312AuditError("producer worker OS-thread evidence drifted")
    recomputed_selection = _select_representatives(
        records, expected_rig_count=EXPECTED_PZ_RIG_COUNT
    )
    status_counts = dict(sorted(Counter(str(row["status"]) for row in records).items()))
    if (
        summary.get("audit_version") != PZ312_AUDIT_VERSION
        or summary.get("generation_id") != generation_root.name
        or summary.get("status") != CANDIDATE_STATUS
        or int(summary.get("clip_count", -1)) != EXPECTED_PZ_CLIP_COUNT
        or int(summary.get("rig_count", -1)) != EXPECTED_PZ_RIG_COUNT
        or summary.get("status_counts") != {"pass": EXPECTED_PZ_CLIP_COUNT}
        or status_counts != {"pass": EXPECTED_PZ_CLIP_COUNT}
        or int(selection.get("selected_count", -1)) != EXPECTED_PZ_RIG_COUNT
        or selection != recomputed_selection
        or summary.get("authority_sha256") != authority_sha256
        or summary.get("prototype_conversion_authorized") is not False
        or summary.get("full_conversion_authorized") is not False
    ):
        raise Pz312AuditError("audit summary/selection scope is not complete")
    expected_recheck = {
        "status": "pass",
        "validated_count": EXPECTED_PZ_CLIP_COUNT,
        "hash_workers": int(recheck.get("hash_workers", 0)),
        "disk_inventory_snapshot_sha256": disk_snapshot["snapshot_sha256"],
        "source_snapshot_sha256": _source_snapshot_sha256(records),
        "completed_at_utc": recheck.get("completed_at_utc"),
    }
    if (
        recheck != expected_recheck
        or not 1 <= expected_recheck["hash_workers"] <= 16
        or not isinstance(expected_recheck["completed_at_utc"], str)
        or not expected_recheck["completed_at_utc"]
        or summary.get("source_snapshot_recheck") != recheck
    ):
        raise Pz312AuditError("final source snapshot recheck evidence drifted")
    recomputed_summary = {
        "J_phys_min": min(int(record["J_phys"]) for record in records),
        "J_phys_max": max(int(record["J_phys"]) for record in records),
        "T_src_min": min(int(record["T_src"]) for record in records),
        "T_src_max": max(int(record["T_src"]) for record in records),
        "independent_rotation_max_abs": max(
            float(metrics["independent_rotation_max_abs"]) for metrics in all_metrics
        ),
        "independent_root_translation_max_abs": max(
            float(metrics["independent_root_translation_max_abs"])
            for metrics in all_metrics
        ),
        "fixed_fk_source_position_max_norm": max(
            float(metrics["planetzoo_fixed_fk_source_position_max_norm"])
            for metrics in all_metrics
        ),
        "representative_count": EXPECTED_PZ_RIG_COUNT,
        "producer_worker_process_count": len(producer_statuses),
        "producer_worker_threads_max": max(
            (int(value["threads"]) for value in producer_statuses),
            default=0,
        ),
    }
    for field, expected_value in recomputed_summary.items():
        if summary.get(field) != expected_value:
            raise Pz312AuditError(f"audit summary metric drifted: {field}")
    return generation


def _load_active_conditioning(path: Path, *, expected_sha256: str) -> Mapping[str, Any]:
    _regular_single_link_stat(path, label="active conditioning")
    if path.resolve(strict=True) != path:
        raise Pz312AuditError(f"active conditioning has a linked component: {path}")
    observed_sha256 = _sha256_file(path)
    if observed_sha256 != expected_sha256 or expected_sha256 != ACTIVE_COND_SHA256:
        raise Pz312AuditError(
            f"active conditioning hash drifted: {observed_sha256} != "
            f"{expected_sha256} != {ACTIVE_COND_SHA256}"
        )
    try:
        conditioning = np.load(path, allow_pickle=True).item()
    except Exception as exc:  # noqa: BLE001
        raise Pz312AuditError(f"cannot load active conditioning: {exc}") from exc
    if not isinstance(conditioning, Mapping):
        raise Pz312AuditError("active conditioning is not a mapping")
    return conditioning


def _validate_npz_payload_exact(
    path: Path, expected_payload: Mapping[str, np.ndarray]
) -> None:
    _regular_single_link_stat(path, label="published fixed skeleton")
    try:
        with np.load(path, allow_pickle=False) as observed:
            if set(observed.files) != set(expected_payload):
                raise Pz312AuditError(
                    f"fixed-skeleton payload keys drifted for {path.name}"
                )
            for name in sorted(expected_payload):
                expected = np.asarray(expected_payload[name])
                actual = np.asarray(observed[name])
                if (
                    actual.dtype != expected.dtype
                    or actual.shape != expected.shape
                    or not np.array_equal(actual, expected)
                ):
                    raise Pz312AuditError(
                        f"fixed-skeleton payload field drifted for {path.name}: {name}"
                    )
    except Pz312AuditError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise Pz312AuditError(f"cannot validate fixed skeleton {path}: {exc}") from exc


def _rebuild_and_validate_published_skeletons(
    *,
    generation_root: Path,
    rig_records: Sequence[Mapping[str, Any]],
    parent_rigs: Mapping[str, Mapping[str, Any]],
    pz_root: Path,
    conditioning: Mapping[str, Any],
    active_cond_sha256: str,
) -> dict[str, SkeletonData]:
    by_rig = {str(record.get("rig_id")): record for record in rig_records}
    if len(by_rig) != len(rig_records) or set(by_rig) != set(parent_rigs):
        raise Pz312AuditError("published rig QA does not bijectively cover parent rigs")
    skeletons: dict[str, SkeletonData] = {}
    expected_rig_keys = {
        "audit_version",
        "rig_id",
        "status",
        "J_phys",
        "skeleton_sha256",
        "metrics",
        "claim_boundary",
    }
    for index, rig_id in enumerate(sorted(parent_rigs), start=1):
        record = by_rig[rig_id]
        if set(record) != expected_rig_keys:
            raise Pz312AuditError(f"published rig QA schema drifted: {rig_id}")
        if rig_id not in conditioning:
            raise Pz312AuditError(f"active conditioning lacks {rig_id}")
        rig = parent_rigs[rig_id]
        rebuilt = build_planetzoo_fixed_rig(
            rig_record=rig,
            representative_bvh_path=rig["rest_pose"]["source_path"],
            pinned_source_root=pz_root,
            cond_entry=conditioning[rig_id],
            active_cond_sha256=active_cond_sha256,
        )
        if (
            record.get("audit_version") != PZ312_AUDIT_VERSION
            or record.get("status") != "pass"
            or int(record.get("J_phys", -1)) != len(rebuilt.parents)
            or record.get("metrics") != rebuilt.metrics
            or record.get("claim_boundary") != rebuilt.provenance["claim_boundary"]
        ):
            raise Pz312AuditError(f"rebuilt fixed-rig QA drifted: {rig_id}")
        path = generation_root / "skeletons" / f"{rig_id}.npz"
        _validate_npz_payload_exact(path, rebuilt.payload)
        skeleton = load_skeleton(path)
        if skeleton.sha256 != record.get("skeleton_sha256"):
            raise Pz312AuditError(f"rebuilt fixed-rig SHA drifted: {rig_id}")
        skeletons[rig_id] = skeleton
        if index % 50 == 0 or index == len(parent_rigs):
            print(
                f"[pz311-validator] rebuilt fixed rigs {index}/{len(parent_rigs)}",
                flush=True,
            )
    return skeletons


def _first_record_difference(expected: Any, actual: Any, path: str = "$") -> str:
    if type(expected) is not type(actual):
        return f"{path}: type {type(actual).__name__} != {type(expected).__name__}"
    if isinstance(expected, Mapping):
        if set(expected) != set(actual):
            return (
                f"{path}: keys missing={sorted(set(expected) - set(actual))}, "
                f"extra={sorted(set(actual) - set(expected))}"
            )
        for key in sorted(expected):
            difference = _first_record_difference(
                expected[key], actual[key], f"{path}.{key}"
            )
            if difference:
                return difference
        return ""
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{path}: length {len(actual)} != {len(expected)}"
        for index, (expected_item, actual_item) in enumerate(
            zip(expected, actual, strict=True)
        ):
            difference = _first_record_difference(
                expected_item, actual_item, f"{path}[{index}]"
            )
            if difference:
                return difference
        return ""
    if expected != actual:
        return f"{path}: {actual!r} != {expected!r}"
    return ""


def _deep_reaudit_published_records(
    *,
    records: Sequence[Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
    rigs: dict[str, dict[str, Any]],
    skeletons: dict[str, SkeletonData],
    pz_root: Path,
    workers: int,
    chunk_size: int,
) -> dict[str, Any]:
    """Re-run both decoders, fixed FK, energy, and SHA for every live source."""
    if workers <= 0 or chunk_size <= 0:
        raise Pz312AuditError("deep-validator workers/chunk_size must be positive")
    if len(records) != len(tasks):
        raise Pz312AuditError("deep-validator record/task count drifted")
    chunks = [
        (
            index,
            list(tasks[start : start + chunk_size]),
            list(records[start : start + chunk_size]),
        )
        for index, start in enumerate(range(0, len(tasks), chunk_size))
    ]
    before_inventory = _assert_manifest_disk_bijection(pz_root, tasks)
    completed = 0
    worker_statuses: dict[int, dict[str, int]] = {}

    def compare(
        index: int,
        task_chunk: Sequence[Mapping[str, Any]],
        expected_chunk: Sequence[Mapping[str, Any]],
        actual_chunk: Any,
        process_status: Mapping[str, Any] | None = None,
    ) -> None:
        nonlocal completed
        if process_status is not None:
            normalized_status = _validate_worker_process_status(process_status)
            worker_statuses[normalized_status["pid"]] = normalized_status
        if not isinstance(actual_chunk, list) or len(actual_chunk) != len(expected_chunk):
            raise Pz312AuditError(f"deep re-audit chunk {index} count drifted")
        for task, expected, actual in zip(
            task_chunk, expected_chunk, actual_chunk, strict=True
        ):
            if not isinstance(actual, Mapping):
                raise Pz312AuditError(
                    f"deep re-audit returned a non-record: {task['clip_id']}"
                )
            difference = _first_record_difference(expected, actual)
            if difference:
                raise Pz312AuditError(
                    f"deep live re-audit drifted for {task['clip_id']}: {difference}"
                )
        completed += len(actual_chunk)
        if completed % 5000 < len(actual_chunk) or completed == len(tasks):
            print(
                f"[pz311-validator] deep re-audit {completed}/{len(tasks)}",
                flush=True,
            )

    if workers == 1:
        _initialize_worker(rigs, skeletons, str(pz_root))
        for index, task_chunk, expected_chunk in chunks:
            compare(index, task_chunk, expected_chunk, _audit_chunk(task_chunk))
    else:
        context = multiprocessing.get_context("spawn")
        with _single_thread_spawn_environment():
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers,
                mp_context=context,
                initializer=_initialize_worker,
                initargs=(rigs, skeletons, str(pz_root)),
            ) as executor:
                queue = deque(chunks)
                futures: dict[
                    concurrent.futures.Future[dict[str, Any]],
                    tuple[
                        int,
                        list[Mapping[str, Any]],
                        list[Mapping[str, Any]],
                    ],
                ] = {}
                while queue and len(futures) < workers * 2:
                    index, task_chunk, expected_chunk = queue.popleft()
                    futures[
                        executor.submit(_audit_chunk_with_process_status, task_chunk)
                    ] = (
                        index,
                        task_chunk,
                        expected_chunk,
                    )
                while futures:
                    done, _ = concurrent.futures.wait(
                        futures,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in done:
                        index, task_chunk, expected_chunk = futures.pop(future)
                        actual_records, process_status = _unpack_worker_chunk_result(
                            future.result()
                        )
                        compare(
                            index,
                            task_chunk,
                            expected_chunk,
                            actual_records,
                            process_status,
                        )
                    while queue and len(futures) < workers * 2:
                        index, task_chunk, expected_chunk = queue.popleft()
                        futures[
                            executor.submit(_audit_chunk_with_process_status, task_chunk)
                        ] = (
                            index,
                            task_chunk,
                            expected_chunk,
                        )
    after_inventory = _assert_manifest_disk_bijection(pz_root, tasks)
    if after_inventory != before_inventory:
        raise Pz312AuditError("PZ disk inventory changed during deep re-audit")
    return {
        "source_recheck": _revalidate_all_live_sources(
            records,
            tasks,
            pz_root=pz_root,
            workers=workers,
        ),
        "worker_process_statuses": [
            worker_statuses[pid] for pid in sorted(worker_statuses)
        ],
    }


def _generation_content_evidence(root: str | Path) -> dict[str, Any]:
    generation_root = _absolute_no_resolve(root)
    output_root = generation_root.parent.parent
    generation_path = generation_root / "generation.json"
    observed_stat = _regular_single_link_stat(
        generation_path, label="candidate generation manifest"
    )
    if int(observed_stat.st_mode) & 0o222:
        raise Pz312AuditError(f"candidate generation manifest is writable: {generation_path}")
    generation = _load_json(generation_path)
    observed_files = _file_manifest(generation_root, require_read_only=True)
    if observed_files != generation.get("files"):
        raise Pz312AuditError("candidate generation content closure drifted")
    core = {
        "audit_version": PZ312_AUDIT_VERSION,
        "generation_id": generation_root.name,
        "generation_root": str(generation_root),
        "output_root": str(output_root),
        "generation_json_sha256": _sha256_file(generation_path),
        "generation_json_size_bytes": int(observed_stat.st_size),
        "authority_sha256": str(generation.get("authority_sha256")),
        "files": observed_files,
    }
    return {
        **core,
        "generation_content_sha256": hashlib.sha256(_canonical_json(core)).hexdigest(),
    }


def _validate_pz_audit_candidate(
    root: str | Path, *, workers: int = 24
) -> dict[str, Any]:
    """Deep-check a frozen pending candidate without granting authorization."""
    generation = _validate_pz_audit_generation_structure(root)
    initial_content = _generation_content_evidence(root)
    generation_root = _absolute_no_resolve(root)
    authority = _load_json(generation_root / "authority.json")
    if authority.get("producer_code_sha256") != _producer_code_sha256():
        raise Pz312AuditError("current producer import closure drifted from authority")
    if authority.get("runtime_fingerprint") != _runtime_fingerprint():
        raise Pz312AuditError("current numerical runtime drifted from authority")
    if (
        authority.get("production_decoder")
        != "src.data.ktjd17.source_parser.parse_bvh_source"
        or authority.get("independent_decoder") != INDEPENDENT_DECODER_ID
    ):
        raise Pz312AuditError("published decoder authority drifted")
    chunk_size = int(authority.get("audit_chunk_size", 0))
    parent_root = _absolute_no_resolve(str(authority["parent_manifest_root"]))
    pz_root = _absolute_no_resolve(str(authority["pz_bvh_root"]))
    active_cond_path = _absolute_no_resolve(str(authority.get("active_cond_path", "")))
    active_cond_sha256 = str(authority.get("active_cond_sha256", ""))
    conditioning = _load_active_conditioning(
        active_cond_path, expected_sha256=active_cond_sha256
    )
    parent_rigs, parent_tasks = _load_pz_scope(parent_root, pz_root)
    records = _load_jsonl(generation_root / "qa/pz_source_audit.jsonl")
    rig_records = _load_jsonl(generation_root / "qa/rig_audit.jsonl")
    if [record.get("clip_id") for record in records] != [
        task["clip_id"] for task in parent_tasks
    ]:
        raise Pz312AuditError("published records do not match parent task order")
    for record, task in zip(records, parent_tasks, strict=True):
        _validate_pass_record_against_task(record, task)
    skeletons = _rebuild_and_validate_published_skeletons(
        generation_root=generation_root,
        rig_records=rig_records,
        parent_rigs=parent_rigs,
        pz_root=pz_root,
        conditioning=conditioning,
        active_cond_sha256=active_cond_sha256,
    )
    deep_result = _deep_reaudit_published_records(
        records=records,
        tasks=parent_tasks,
        rigs=dict(parent_rigs),
        skeletons=skeletons,
        pz_root=pz_root,
        workers=int(workers),
        chunk_size=chunk_size,
    )
    deep_recheck = dict(deep_result["source_recheck"])
    deep_worker_statuses = [
        _validate_worker_process_status(value)
        for value in deep_result["worker_process_statuses"]
    ]
    if workers > 1 and not deep_worker_statuses:
        raise Pz312AuditError("deep validator produced no spawned-worker evidence")
    published_recheck = _load_json(
        generation_root / "qa/source_snapshot_recheck.json"
    )
    for field in (
        "status",
        "validated_count",
        "disk_inventory_snapshot_sha256",
        "source_snapshot_sha256",
    ):
        if deep_recheck.get(field) != published_recheck.get(field):
            raise Pz312AuditError(
                f"post-publish live source recheck drifted: {field}"
            )
    final_generation = _validate_pz_audit_generation_structure(root)
    final_content = _generation_content_evidence(root)
    if final_generation != generation or final_content != initial_content:
        raise Pz312AuditError("candidate changed during post-publish deep validation")
    return {
        "generation": generation,
        "content_evidence": final_content,
        "deep_source_recheck": deep_recheck,
        "deep_validator_worker_process_statuses": deep_worker_statuses,
    }


def _validate_live_recheck_evidence(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Pz312AuditError("live-source recheck evidence is not an object")
    evidence = dict(value)
    expected_keys = {
        "status",
        "validated_count",
        "hash_workers",
        "disk_inventory_snapshot_sha256",
        "source_snapshot_sha256",
        "completed_at_utc",
    }
    if set(evidence) != expected_keys:
        raise Pz312AuditError("live-source recheck evidence schema drifted")
    if (
        evidence.get("status") != "pass"
        or int(evidence.get("validated_count", -1)) != EXPECTED_PZ_CLIP_COUNT
        or not 1 <= int(evidence.get("hash_workers", 0)) <= 16
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(evidence.get("disk_inventory_snapshot_sha256", "")),
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}", str(evidence.get("source_snapshot_sha256", ""))
        )
        is None
        or not isinstance(evidence.get("completed_at_utc"), str)
        or not evidence["completed_at_utc"]
    ):
        raise Pz312AuditError("live-source recheck evidence is invalid")
    return evidence


def _recheck_generation_live_sources(
    root: str | Path, *, workers: int
) -> dict[str, Any]:
    """Re-hash the complete mutable BVH scope bound by one frozen generation."""
    generation_root = _absolute_no_resolve(root)
    authority = _load_json(generation_root / "authority.json")
    records = _load_jsonl(generation_root / "qa/pz_source_audit.jsonl")
    tasks = [_task_from_pass_record(record) for record in records]
    pz_root = _absolute_no_resolve(str(authority.get("pz_bvh_root", "")))
    current = _validate_live_recheck_evidence(
        _revalidate_all_live_sources(
            records,
            tasks,
            pz_root=pz_root,
            workers=int(workers),
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
            raise Pz312AuditError(
                f"live source scope drifted after deep validation: {field}"
            )
    return current


def _live_recheck_sha256(value: Mapping[str, Any]) -> str:
    evidence = _validate_live_recheck_evidence(value)
    return hashlib.sha256(_canonical_json(evidence)).hexdigest()


def _verify_approved_live_sources(
    generation_root: Path,
    approval: Mapping[str, Any],
    *,
    workers: int,
) -> dict[str, Any]:
    approved = _validate_live_recheck_evidence(
        approval.get("post_deep_live_source_recheck")
    )
    if approval.get("post_deep_live_source_recheck_sha256") != (
        _live_recheck_sha256(approved)
    ):
        raise Pz312AuditError("approved live-source recheck digest drifted")
    current = _recheck_generation_live_sources(generation_root, workers=workers)
    for field in (
        "status",
        "validated_count",
        "disk_inventory_snapshot_sha256",
        "source_snapshot_sha256",
    ):
        if current[field] != approved[field]:
            raise Pz312AuditError(
                f"approved live source scope is no longer current: {field}"
            )
    return current


def _create_generation_approval(
    *, output_root: Path, generation_root: Path, candidate_proof: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    content = dict(candidate_proof["content_evidence"])
    deep_recheck = dict(candidate_proof["deep_source_recheck"])
    post_deep_live_recheck = _validate_live_recheck_evidence(
        candidate_proof.get("post_deep_live_source_recheck")
    )
    deep_worker_statuses = [
        _validate_worker_process_status(value)
        for value in candidate_proof.get(
            "deep_validator_worker_process_statuses", []
        )
    ]
    authority = _load_json(generation_root / "authority.json")
    generation_relpath = generation_root.relative_to(output_root).as_posix()
    approval = {
        "approval_version": AUDIT_APPROVAL_VERSION,
        "audit_version": PZ312_AUDIT_VERSION,
        "status": "pass",
        "generation_id": generation_root.name,
        "generation_root": str(generation_root),
        "output_root": str(output_root),
        "generation_relpath": generation_relpath,
        "generation_content_sha256": content["generation_content_sha256"],
        "generation_json_sha256": content["generation_json_sha256"],
        "authority_sha256": content["authority_sha256"],
        "source_snapshot_sha256": deep_recheck["source_snapshot_sha256"],
        "deep_validated_count": int(deep_recheck["validated_count"]),
        "producer_code_sha256": authority["producer_code_sha256"],
        "runtime_fingerprint_sha256": hashlib.sha256(
            _canonical_json(authority["runtime_fingerprint"])
        ).hexdigest(),
        "post_deep_live_source_recheck": post_deep_live_recheck,
        "post_deep_live_source_recheck_sha256": _live_recheck_sha256(
            post_deep_live_recheck
        ),
        "deep_validator_worker_process_statuses": deep_worker_statuses,
        "deep_validator_worker_process_statuses_sha256": (
            _worker_statuses_sha256(deep_worker_statuses)
        ),
        "prototype_conversion_authorized": True,
        "full_conversion_authorized": False,
        "approved_at_utc": _datetime.datetime.now(_datetime.UTC).isoformat(),
    }
    approval_root = output_root / AUDIT_APPROVAL_DIRECTORY
    approval_root.mkdir(parents=True, exist_ok=True)
    if approval_root.is_symlink() or approval_root.resolve(strict=True) != approval_root:
        raise Pz312AuditError(f"invalid approval root: {approval_root}")
    approval_path = approval_root / f"{content['generation_content_sha256']}.json"
    if os.path.lexists(approval_path):
        observed = _load_json(approval_path)
        if observed != approval:
            raise Pz312AuditError(f"approval digest collision/drift: {approval_path}")
    else:
        _write_json_atomic(approval_path, approval)
        observed_stat = approval_path.lstat()
        os.chmod(
            approval_path,
            int(observed_stat.st_mode) & ~0o222,
        )
        _fsync_directory(approval_root)
    approval_stat = _regular_single_link_stat(
        approval_path, label="immutable generation approval"
    )
    if int(approval_stat.st_mode) & 0o222:
        raise Pz312AuditError(f"generation approval is writable: {approval_path}")
    return approval_path, approval


def _validate_generation_approval(
    generation_root: Path,
    *,
    content_evidence: Mapping[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    root = _absolute_no_resolve(generation_root)
    if root.parent.name != AUDIT_GENERATION_DIRECTORY:
        raise Pz312AuditError(f"generation is outside the approval namespace: {root}")
    output_root = root.parent.parent
    content = (
        dict(content_evidence)
        if content_evidence is not None
        else _generation_content_evidence(root)
    )
    approval_path = (
        output_root
        / AUDIT_APPROVAL_DIRECTORY
        / f"{content['generation_content_sha256']}.json"
    )
    approval_stat = _regular_single_link_stat(
        approval_path, label="immutable generation approval"
    )
    if int(approval_stat.st_mode) & 0o222:
        raise Pz312AuditError(f"generation approval is writable: {approval_path}")
    approval = _load_json(approval_path)
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
        "deep_validated_count",
        "producer_code_sha256",
        "runtime_fingerprint_sha256",
        "post_deep_live_source_recheck",
        "post_deep_live_source_recheck_sha256",
        "deep_validator_worker_process_statuses",
        "deep_validator_worker_process_statuses_sha256",
        "prototype_conversion_authorized",
        "full_conversion_authorized",
        "approved_at_utc",
    }
    if set(approval) != expected_keys:
        raise Pz312AuditError("generation approval schema drifted")
    relative = Path(str(approval["generation_relpath"]))
    resolved_from_approval = _absolute_no_resolve(output_root / relative)
    authority = _load_json(root / "authority.json")
    recheck = _load_json(root / "qa/source_snapshot_recheck.json")
    approved_live_recheck = _validate_live_recheck_evidence(
        approval.get("post_deep_live_source_recheck")
    )
    approved_deep_worker_statuses = [
        _validate_worker_process_status(value)
        for value in approval.get("deep_validator_worker_process_statuses", [])
    ]
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or resolved_from_approval != root
        or approval.get("approval_version") != AUDIT_APPROVAL_VERSION
        or approval.get("audit_version") != PZ312_AUDIT_VERSION
        or approval.get("status") != "pass"
        or approval.get("generation_id") != root.name
        or approval.get("generation_root") != str(root)
        or approval.get("output_root") != str(output_root)
        or content.get("generation_root") != str(root)
        or content.get("output_root") != str(output_root)
        or approval.get("generation_content_sha256")
        != content["generation_content_sha256"]
        or approval.get("generation_json_sha256")
        != content["generation_json_sha256"]
        or approval.get("authority_sha256") != content["authority_sha256"]
        or approval.get("source_snapshot_sha256")
        != recheck.get("source_snapshot_sha256")
        or int(approval.get("deep_validated_count", -1))
        != EXPECTED_PZ_CLIP_COUNT
        or approval.get("producer_code_sha256")
        != authority.get("producer_code_sha256")
        or approval.get("runtime_fingerprint_sha256")
        != hashlib.sha256(
            _canonical_json(authority.get("runtime_fingerprint"))
        ).hexdigest()
        or approval.get("post_deep_live_source_recheck_sha256")
        != _live_recheck_sha256(approved_live_recheck)
        or approved_live_recheck.get("source_snapshot_sha256")
        != recheck.get("source_snapshot_sha256")
        or approved_live_recheck.get("disk_inventory_snapshot_sha256")
        != recheck.get("disk_inventory_snapshot_sha256")
        or int(approved_live_recheck.get("validated_count", -1))
        != EXPECTED_PZ_CLIP_COUNT
        or approval.get("deep_validator_worker_process_statuses_sha256")
        != _worker_statuses_sha256(approved_deep_worker_statuses)
        or approval.get("prototype_conversion_authorized") is not True
        or approval.get("full_conversion_authorized") is not False
        or not isinstance(approval.get("approved_at_utc"), str)
        or not approval["approved_at_utc"]
    ):
        raise Pz312AuditError("generation approval/content binding drifted")
    return approval_path, approval


def validate_pz_audit_generation(
    root: str | Path, *, workers: int = 24
) -> dict[str, Any]:
    """Deep-validate one candidate and require its content-addressed approval."""
    proof = _validate_pz_audit_candidate(root, workers=workers)
    approval_path, approval = _validate_generation_approval(
        _absolute_no_resolve(root),
        content_evidence=proof["content_evidence"],
    )
    return {
        **approval,
        "approval_path": str(approval_path),
        "generation_root": str(_absolute_no_resolve(root)),
    }


def validate_active_pz_audit(
    output_root: str | Path, *, workers: int = 16
) -> dict[str, Any]:
    """Verify both active links and re-hash every mutable source BVH."""
    output = _absolute_no_resolve(output_root)
    generation_link = output / AUDIT_LINK_NAME
    approval_link = output / AUDIT_APPROVAL_LINK_NAME
    if not generation_link.is_symlink() or not approval_link.is_symlink():
        raise Pz312AuditError("active generation/approval links are not both symlinks")
    generation_root = generation_link.resolve(strict=True)
    approval_path = approval_link.resolve(strict=True)
    content = _generation_content_evidence(generation_root)
    expected_approval_path, approval = _validate_generation_approval(
        generation_root, content_evidence=content
    )
    if approval_path != expected_approval_path:
        raise Pz312AuditError("active approval link does not bind active generation")
    active_live_recheck = _verify_approved_live_sources(
        generation_root,
        approval,
        workers=int(workers),
    )
    return {
        **approval,
        "approval_path": str(approval_path),
        "generation_root": str(generation_root),
        "active_live_source_recheck": active_live_recheck,
    }


def _activate_approved_generation(
    *,
    output_root: Path,
    generation_root: Path,
    approval_path: Path,
    workers: int = 16,
) -> dict[str, Any]:
    generation_link = output_root / AUDIT_LINK_NAME
    approval_link = output_root / AUDIT_APPROVAL_LINK_NAME
    previous_generation = _snapshot_symlink(generation_link)
    previous_approval = _snapshot_symlink(approval_link)
    try:
        _replace_symlink(generation_link, generation_root)
        content = _generation_content_evidence(generation_root)
        expected_approval_path, approval = _validate_generation_approval(
            generation_root,
            content_evidence=content,
        )
        if approval_path != expected_approval_path:
            raise Pz312AuditError(
                "requested approval does not bind requested generation"
            )
        _verify_approved_live_sources(
            generation_root,
            approval,
            workers=int(workers),
        )
        _replace_symlink(approval_link, approval_path)
        return validate_active_pz_audit(output_root, workers=int(workers))
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
            raise Pz312AuditError(
                f"active authority failed and rollback was incomplete: {rollback_errors}"
            ) from exc
        raise


def _validate_parent_manifest_generation(root: Path) -> dict[str, Any]:
    """Validate and pin the complete twelve-file immutable parent transaction."""
    generation_root = _absolute_no_resolve(root)
    if (
        generation_root.is_symlink()
        or not generation_root.is_dir()
        or generation_root.resolve(strict=True) != generation_root
    ):
        raise Pz312AuditError(f"invalid parent inventory root: {generation_root}")
    transaction_path = generation_root / "inventory_generation.json"
    transaction_stat = _regular_single_link_stat(
        transaction_path, label="parent transaction"
    )
    try:
        transaction = _validate_transaction(generation_root)
    except Exception as exc:  # noqa: BLE001
        raise Pz312AuditError(f"parent inventory transaction is invalid: {exc}") from exc
    if transaction.get("generation_id") != generation_root.name:
        raise Pz312AuditError("parent inventory generation id/path mismatch")
    files = transaction.get("files")
    if not isinstance(files, Mapping) or set(files) != EXPECTED_PARENT_MANIFEST_FILES:
        raise Pz312AuditError(
            "parent inventory does not expose the required twelve-file closure"
        )
    inode_owner = {
        (int(transaction_stat.st_dev), int(transaction_stat.st_ino)):
        "inventory_generation.json"
    }
    for name in sorted(files):
        path = generation_root / name
        observed = _regular_single_link_stat(path, label="parent artifact")
        inode = (int(observed.st_dev), int(observed.st_ino))
        if inode in inode_owner:
            raise Pz312AuditError(
                f"parent artifacts share an inode: {inode_owner[inode]} and {name}"
            )
        inode_owner[inode] = name
    return {
        "generation_id": str(transaction["generation_id"]),
        "inventory_generation_sha256": _sha256_file(transaction_path),
        "manifest_version": str(transaction["manifest_version"]),
        "publish_protocol": str(transaction["publish_protocol"]),
        "files": dict(files),
    }


def _quarantine_failed_candidate(
    *, final: Path, generations_root: Path, work_root: Path, error: BaseException
) -> Path:
    if final.parent != generations_root or not final.is_dir() or final.is_symlink():
        raise Pz312AuditError(f"cannot safely quarantine failed candidate: {final}")
    rejected = generations_root / f".rejected-{final.name}-{uuid.uuid4().hex[:8]}"
    os.replace(final, rejected)
    _fsync_directory(generations_root)
    _write_json(
        work_root / "post_publish_validation_failure.json",
        {
            "status": "rejected",
            "candidate_generation_id": final.name,
            "quarantined_relpath": rejected.relative_to(
                generations_root.parent
            ).as_posix(),
            "error_type": type(error).__name__,
            "error": str(error),
            "rejected_at_utc": _datetime.datetime.now(_datetime.UTC).isoformat(),
        },
    )
    return rejected


def run_pz_source_audit(config: PzAuditConfig) -> dict[str, Any]:
    cfg = config.resolved()
    if cfg.workers <= 0 or cfg.chunk_size <= 0:
        raise Pz312AuditError("workers and chunk_size must be positive")
    for path in (
        cfg.manifest_root / "inventory_generation.json",
        cfg.manifest_root / "clips.jsonl",
        cfg.manifest_root / "rigs.jsonl",
        cfg.manifest_root / "prototype_candidates.json",
        cfg.active_cond_path,
    ):
        _regular_single_link_stat(path, label="required pinned input")
        if path.resolve(strict=True) != path:
            raise Pz312AuditError(f"required pinned input has a linked component: {path}")
    if not cfg.pz_bvh_root.is_dir() or cfg.pz_bvh_root.is_symlink():
        raise Pz312AuditError(f"invalid PZ BVH root: {cfg.pz_bvh_root}")
    parent_generation = _validate_parent_manifest_generation(cfg.manifest_root)
    cond_sha256 = _sha256_file(cfg.active_cond_path)
    if cond_sha256 != ACTIVE_COND_SHA256:
        raise Pz312AuditError(
            f"active cond hash drifted: {cond_sha256} != {ACTIVE_COND_SHA256}"
        )
    producer_code_sha256 = _producer_code_sha256()
    runtime_fingerprint = _runtime_fingerprint()
    rigs, tasks = _load_pz_scope(cfg.manifest_root, cfg.pz_bvh_root)
    disk_inventory = _assert_manifest_disk_bijection(cfg.pz_bvh_root, tasks)
    authority = {
        "audit_version": PZ312_AUDIT_VERSION,
        "parent_manifest_root": str(cfg.manifest_root),
        "parent_inventory_generation": parent_generation,
        "active_cond_path": str(cfg.active_cond_path),
        "active_cond_sha256": cond_sha256,
        "pz_bvh_root": str(cfg.pz_bvh_root),
        "source_scope_sha256": _chunk_sha256(tasks),
        "disk_inventory_snapshot": disk_inventory,
        "expected_rig_count": EXPECTED_PZ_RIG_COUNT,
        "expected_clip_count": EXPECTED_PZ_CLIP_COUNT,
        "audit_workers": int(cfg.workers),
        "audit_chunk_size": int(cfg.chunk_size),
        "production_decoder": "src.data.ktjd17.source_parser.parse_bvh_source",
        "independent_decoder": INDEPENDENT_DECODER_ID,
        "producer_code_sha256": producer_code_sha256,
        "runtime_fingerprint": runtime_fingerprint,
        "claim_boundary": (
            "processed_planetzoo_stage2_coordinates_only_not_native_raw_game_bvh"
        ),
    }
    authority_sha256 = hashlib.sha256(_canonical_json(authority)).hexdigest()
    work_root = cfg.output_root / AUDIT_WORK_DIRECTORY / authority_sha256[:20]
    chunk_root = work_root / "chunks"
    skeleton_root = work_root / "skeletons"
    chunk_root.mkdir(parents=True, exist_ok=True)
    skeleton_root.mkdir(parents=True, exist_ok=True)
    _write_json(work_root / "authority.json", {**authority, "authority_sha256": authority_sha256})

    conditioning = _load_active_conditioning(
        cfg.active_cond_path, expected_sha256=cond_sha256
    )
    skeletons: dict[str, SkeletonData] = {}
    rig_audit_records: list[dict[str, Any]] = []
    for index, rig_id in enumerate(sorted(rigs), start=1):
        if rig_id not in conditioning:
            raise Pz312AuditError(f"active conditioning lacks {rig_id}")
        rig = rigs[rig_id]
        fixed = build_planetzoo_fixed_rig(
            rig_record=rig,
            representative_bvh_path=rig["rest_pose"]["source_path"],
            pinned_source_root=cfg.pz_bvh_root,
            cond_entry=conditioning[rig_id],
            active_cond_sha256=cond_sha256,
        )
        path = skeleton_root / f"{rig_id}.npz"
        skeleton_sha256 = write_npz_atomic(path, fixed.payload)
        skeleton = load_skeleton(path)
        if skeleton.sha256 != skeleton_sha256:
            raise Pz312AuditError(f"{rig_id}: skeleton write/read hash drifted")
        skeletons[rig_id] = skeleton
        rig_audit_records.append(
            {
                "audit_version": PZ312_AUDIT_VERSION,
                "rig_id": rig_id,
                "status": "pass",
                "J_phys": len(skeleton.parents),
                "skeleton_sha256": skeleton_sha256,
                "metrics": fixed.metrics,
                "claim_boundary": fixed.provenance["claim_boundary"],
            }
        )
        if index % 50 == 0 or index == len(rigs):
            print(f"[pz311-audit] built fixed rigs {index}/{len(rigs)}", flush=True)
    _write_jsonl(work_root / "rig_audit.jsonl", rig_audit_records)

    chunks = [
        tasks[start : start + cfg.chunk_size]
        for start in range(0, len(tasks), cfg.chunk_size)
    ]
    pending: list[tuple[int, list[dict[str, Any]]]] = []
    completed_clips = 0
    for index, chunk in enumerate(chunks):
        path = chunk_root / f"chunk_{index:06d}.json"
        cached = _load_valid_chunk(
            path,
            tasks=chunk,
            authority_sha256=authority_sha256,
            pz_root=cfg.pz_bvh_root,
        )
        if cached is None:
            pending.append((index, chunk))
        else:
            completed_clips += len(cached)
    print(
        f"[pz311-audit] resume={completed_clips}/{len(tasks)}; "
        f"pending_chunks={len(pending)}; workers={cfg.workers}",
        flush=True,
    )

    def persist(
        index: int,
        chunk: Sequence[Mapping[str, Any]],
        records: Any,
        worker_process_status: Mapping[str, Any] | None = None,
    ) -> None:
        nonlocal completed_clips
        normalized_worker_status = (
            _validate_worker_process_status(worker_process_status)
            if worker_process_status is not None
            else None
        )
        if cfg.workers > 1 and normalized_worker_status is None:
            raise Pz312AuditError(
                f"chunk {index}: spawned worker returned no OS-process evidence"
            )
        if not isinstance(records, list) or len(records) != len(chunk):
            raise Pz312AuditError(f"chunk {index}: worker result count drifted")
        if [record.get("clip_id") for record in records] != [
            task["clip_id"] for task in chunk
        ]:
            raise Pz312AuditError(f"chunk {index}: worker result order drifted")
        for record, task in zip(records, chunk, strict=True):
            _validate_fresh_record_against_task(record, task)
        _write_json_atomic(
            chunk_root / f"chunk_{index:06d}.json",
            {
                "audit_version": PZ312_AUDIT_VERSION,
                "authority_sha256": authority_sha256,
                "task_sha256": _chunk_sha256(chunk),
                "records_sha256": _records_sha256(records),
                "worker_process_status": normalized_worker_status,
                "records": records,
            },
        )
        completed_clips += len(records)
        print(
            f"[pz311-audit] audited {completed_clips}/{len(tasks)}",
            flush=True,
        )

    if cfg.workers == 1:
        _initialize_worker(rigs, skeletons, str(cfg.pz_bvh_root))
        for index, chunk in pending:
            persist(index, chunk, _audit_chunk(chunk))
    elif pending:
        context = multiprocessing.get_context("spawn")
        with _single_thread_spawn_environment():
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=cfg.workers,
                mp_context=context,
                initializer=_initialize_worker,
                initargs=(rigs, skeletons, str(cfg.pz_bvh_root)),
            ) as executor:
                queue = deque(pending)
                futures: dict[
                    concurrent.futures.Future[dict[str, Any]],
                    tuple[int, list[dict[str, Any]]],
                ] = {}
                while queue and len(futures) < cfg.workers * 2:
                    index, chunk = queue.popleft()
                    futures[executor.submit(_audit_chunk_with_process_status, chunk)] = (
                        index,
                        chunk,
                    )
                while futures:
                    done, _ = concurrent.futures.wait(
                        futures,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in done:
                        index, chunk = futures.pop(future)
                        actual_records, process_status = _unpack_worker_chunk_result(
                            future.result()
                        )
                        persist(index, chunk, actual_records, process_status)
                    while queue and len(futures) < cfg.workers * 2:
                        index, chunk = queue.popleft()
                        futures[
                            executor.submit(_audit_chunk_with_process_status, chunk)
                        ] = (index, chunk)

    records: list[dict[str, Any]] = []
    producer_worker_statuses: dict[int, dict[str, int]] = {}
    producer_chunk_status_count = 0
    for index, chunk in enumerate(chunks):
        chunk_path = chunk_root / f"chunk_{index:06d}.json"
        cached = _load_chunk_payload(
            chunk_path,
            tasks=chunk,
            authority_sha256=authority_sha256,
        )
        if cached is None:
            raise Pz312AuditError(f"completed audit chunk {index} is invalid")
        records.extend(cached)
        chunk_payload = _load_json(chunk_path)
        worker_status = chunk_payload.get("worker_process_status")
        if worker_status is not None:
            normalized_status = _validate_worker_process_status(worker_status)
            producer_worker_statuses[normalized_status["pid"]] = normalized_status
            producer_chunk_status_count += 1
    if cfg.workers > 1 and producer_chunk_status_count != len(chunks):
        raise Pz312AuditError(
            "producer executor lacks OS-process evidence for one or more chunks"
        )
    producer_worker_evidence = {
        "audit_version": PZ312_AUDIT_VERSION,
        "status": "pass",
        "executor_mode": "spawn" if cfg.workers > 1 else "in_process",
        "chunk_count": len(chunks),
        "chunks_with_process_status": producer_chunk_status_count,
        "worker_process_statuses": [
            producer_worker_statuses[pid]
            for pid in sorted(producer_worker_statuses)
        ],
    }
    _write_json(
        work_root / "producer_worker_process_status.json",
        producer_worker_evidence,
    )
    if len(records) != EXPECTED_PZ_CLIP_COUNT:
        raise Pz312AuditError("merged exhaustive PZ record count drifted")
    for record, task in zip(records, tasks, strict=True):
        _validate_fresh_record_against_task(record, task)
    status_counts = Counter(str(record["status"]) for record in records)
    failures = [record for record in records if record["status"] != "pass"]
    if failures:
        failure_summary = {
            "audit_version": PZ312_AUDIT_VERSION,
            "authority_sha256": authority_sha256,
            "status": "fail",
            "status_counts": dict(sorted(status_counts.items())),
            "failure_count": len(failures),
            "failure_error_types": dict(
                sorted(Counter(record.get("error_type") for record in failures).items())
            ),
            "failures": failures[:100],
        }
        _write_json(work_root / "audit_failure_summary.json", failure_summary)
        raise Pz312AuditError(
            f"exhaustive PZ audit rejected {len(failures)} clips; "
            f"see {work_root / 'audit_failure_summary.json'}"
        )

    source_snapshot_recheck = _revalidate_all_live_sources(
        records,
        tasks,
        pz_root=cfg.pz_bvh_root,
        workers=cfg.workers,
    )
    representatives = _select_representatives(
        records, expected_rig_count=EXPECTED_PZ_RIG_COUNT
    )
    generation_id = (
        _datetime.datetime.now(_datetime.UTC).strftime("%Y%m%dT%H%M%S%fZ")
        + "-"
        + uuid.uuid4().hex[:12]
    )
    generations = cfg.output_root / AUDIT_GENERATION_DIRECTORY
    generations.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{generation_id}.", dir=generations))
    final = generations / generation_id
    try:
        shutil.copytree(skeleton_root, staging / "skeletons")
        _write_jsonl(staging / "qa/pz_source_audit.jsonl", records)
        _write_jsonl(staging / "qa/rig_audit.jsonl", rig_audit_records)
        _write_json(
            staging / "qa/source_snapshot_recheck.json", source_snapshot_recheck
        )
        _write_json(
            staging / "qa/producer_worker_process_status.json",
            producer_worker_evidence,
        )
        _write_json(staging / "selection/pz_representatives.json", representatives)
        _write_json(
            staging / "authority.json",
            {**authority, "authority_sha256": authority_sha256},
        )
        all_metrics = [record["metrics"] for record in records]
        summary = {
            "audit_version": PZ312_AUDIT_VERSION,
            "generation_id": generation_id,
            "status": CANDIDATE_STATUS,
            "source_audit_status": "pass",
            "rig_count": EXPECTED_PZ_RIG_COUNT,
            "clip_count": EXPECTED_PZ_CLIP_COUNT,
            "status_counts": dict(sorted(status_counts.items())),
            "J_phys_min": min(int(record["J_phys"]) for record in records),
            "J_phys_max": max(int(record["J_phys"]) for record in records),
            "T_src_min": min(int(record["T_src"]) for record in records),
            "T_src_max": max(int(record["T_src"]) for record in records),
            "independent_rotation_max_abs": max(
                float(metrics["independent_rotation_max_abs"])
                for metrics in all_metrics
            ),
            "independent_root_translation_max_abs": max(
                float(metrics["independent_root_translation_max_abs"])
                for metrics in all_metrics
            ),
            "fixed_fk_source_position_max_norm": max(
                float(metrics["planetzoo_fixed_fk_source_position_max_norm"])
                for metrics in all_metrics
            ),
            "representative_count": int(representatives["selected_count"]),
            "producer_worker_process_count": len(producer_worker_statuses),
            "producer_worker_threads_max": max(
                (
                    int(value["threads"])
                    for value in producer_worker_statuses.values()
                ),
                default=0,
            ),
            "authority_sha256": authority_sha256,
            "source_snapshot_recheck": source_snapshot_recheck,
            "claim_boundary": authority["claim_boundary"],
            "prototype_conversion_authorized": False,
            "full_conversion_authorized": False,
        }
        _write_json(staging / "summary.json", summary)
        files = _file_manifest(staging)
        generation = {
            "audit_version": PZ312_AUDIT_VERSION,
            "generation_id": generation_id,
            "created_at_utc": _datetime.datetime.now(_datetime.UTC).isoformat(),
            "status": CANDIDATE_STATUS,
            "authority_sha256": authority_sha256,
            "parent_manifest_root": str(cfg.manifest_root),
            "files": files,
            "prototype_conversion_authorized": False,
            "full_conversion_authorized": False,
        }
        _write_json(staging / "generation.json", generation)
        _freeze_immutable_tree(staging)
        if final.exists():
            raise Pz312AuditError(f"audit generation already exists: {final}")
        os.replace(staging, final)
        _fsync_directory(generations)
        try:
            candidate_proof = _validate_pz_audit_candidate(
                final, workers=cfg.workers
            )
            post_return_generation = _validate_pz_audit_generation_structure(final)
            post_return_content = _generation_content_evidence(final)
            if (
                post_return_generation != candidate_proof["generation"]
                or post_return_content != candidate_proof["content_evidence"]
            ):
                raise Pz312AuditError(
                    "candidate changed after deep validation returned"
                )
            post_deep_live_recheck = _recheck_generation_live_sources(
                final,
                workers=cfg.workers,
            )
            candidate_proof = {
                **candidate_proof,
                "post_deep_live_source_recheck": post_deep_live_recheck,
            }
            if (
                _validate_pz_audit_generation_structure(final)
                != post_return_generation
                or _generation_content_evidence(final) != post_return_content
            ):
                raise Pz312AuditError(
                    "candidate changed during post-deep live-source recheck"
                )
            approval_path, approval = _create_generation_approval(
                output_root=cfg.output_root,
                generation_root=final,
                candidate_proof=candidate_proof,
            )
            if _generation_content_evidence(final) != post_return_content:
                raise Pz312AuditError(
                    "candidate changed between approval and activation"
                )
            active = None
            if cfg.update_link:
                active = _activate_approved_generation(
                    output_root=cfg.output_root,
                    generation_root=final,
                    approval_path=approval_path,
                    workers=cfg.workers,
                )
            else:
                _verify_approved_live_sources(
                    final,
                    approval,
                    workers=cfg.workers,
                )
            return {
                **summary,
                "status": "pass",
                "candidate_status": CANDIDATE_STATUS,
                "prototype_conversion_authorized": True,
                "full_conversion_authorized": False,
                "generation_root": str(final),
                "generation_content_sha256": approval[
                    "generation_content_sha256"
                ],
                "approval_path": str(approval_path),
                "compatibility_link": (
                    str(cfg.output_root / AUDIT_LINK_NAME)
                    if cfg.update_link
                    else None
                ),
                "active_approval_link": (
                    str(cfg.output_root / AUDIT_APPROVAL_LINK_NAME)
                    if cfg.update_link
                    else None
                ),
                "active_binding": active,
                "resumable_work_root": str(work_root),
            }
        except BaseException as exc:
            if final.exists() and not final.is_symlink():
                _quarantine_failed_candidate(
                    final=final,
                    generations_root=generations,
                    work_root=work_root,
                    error=exc,
                )
            raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
