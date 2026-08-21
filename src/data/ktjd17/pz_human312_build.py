"""Research converter for PlanetZoo-311 plus Human-1 KTJD-17 data.

The converter reuses the completed source-audit pass lists, hashes each source
file while it is converted, and checks the stored KTJD-17 numerical/FK result.
It intentionally does not treat filesystem device/inode identity or the audit
program's own source hash as part of the research-data contract.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import datetime as _datetime
import hashlib
import io
import json
import multiprocessing
import os
import re
import shutil
import stat
import tempfile
import time
import uuid
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .encoder import (
    EncoderConfig,
    Ktjd17EncoderError,
    PreparedMotion,
    SkeletonData,
    encode_prepared_motion,
    load_skeleton,
    prepare_manifest_clip,
    write_npz_atomic,
)
from .fixed_qa import (
    _validate_motion as _deep_validate_motion,
    _validate_skeleton as _deep_validate_skeleton,
    independent_decode_column_cont6d,
)
from .freeze import verify_generation_file_closure
from .human312_audit import (
    independent_motionstreamer272_decode,
)
from .human_source_parser import parse_motionstreamer272_fixed_neutral_array
from .loader import derive_masks, load_motion_npz
from .truebones_forward_audit import (
    EXPECTED_FROZEN_SCHEMA_SHA256,
    FROZEN_SCHEMA_GENERATION_ID,
    encoder_config_from_frozen_schema,
)
from .truebones_full_build import FREEZE_GENERATION_SHA256, FROZEN_STATS_SHA256
from .visual_qa import verify_visual_generation


BUILD_VERSION = "ktjd17-pz-human312-build-v1"
BUILD_APPROVAL_VERSION = "ktjd17-pz-human312-build-approval-v1"
PROTOTYPE_GENERATION_DIRECTORY = ".ktjd17_pz_human312_prototype_generations"
PROTOTYPE_LINK_NAME = "ktjd17_pz_human312_prototype"
FULL_GENERATION_DIRECTORY = ".ktjd17_pz_human312_generations"
FULL_LINK_NAME = "ktjd17_pz_human312"
BUILD_APPROVAL_DIRECTORY = ".ktjd17_pz_human312_build_approvals"
PROTOTYPE_APPROVAL_LINK_NAME = "ktjd17_pz_human312_prototype_approval"
FULL_APPROVAL_LINK_NAME = "ktjd17_pz_human312_approval"
VISUAL_GATE_VERSION = "ktjd17-pz-human312-visual-gate-v2"
ANOMALY_ALLOWLIST_VERSION = "ktjd17-pz-human312-anomaly-allowlist-v1"
SOURCE_PLAN_COMMIT = "9181f5cccbad23e941bf94c2874daf36e7f288cf"
COORDINATE_CONTRACT = (
    "right-handed; +Y is screen-up; +Z points out of the screen toward the viewer"
)
EXPECTED_PZ_RIG_COUNT = 311
EXPECTED_HUMAN_RIG_COUNT = 1
EXPECTED_RIG_COUNT = 312
EXPECTED_PZ_CLIP_COUNT = 74_522
EXPECTED_HUMAN_CLIP_COUNT = 26_846
EXPECTED_CLIP_COUNT = 101_368
SPLITS = ("train", "val", "held_representative", "held_stress")
SOURCE_FAMILIES = ("planetzoo", "motionstreamer272")
MAX_ISOLATED_REJECT_FRACTION = 0.001
SAFE_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,254}")
_MACHINE_PATH_MARKERS = tuple(
    "/" + component + "/" for component in ("iridisfs", "scratch")
)
_FILE_URI_MARKER = "file" + "://"
_FILTERABLE_ENCODER_GATES = {
    "float64 direct roundtrip failed": "ENCODER_FLOAT64_DIRECT_ROUNDTRIP",
    "origin restore failed": "ENCODER_ORIGIN_RESTORE",
    "float32 storage is non-finite": "ENCODER_FLOAT32_NONFINITE",
    "float32 direct roundtrip failed": "ENCODER_FLOAT32_DIRECT_ROUNDTRIP",
    "float32 velocity gate failed": "ENCODER_FLOAT32_VELOCITY",
    "rigid edge gate failed": "ENCODER_RIGID_EDGE",
    "non-root root-global storage is not exact zero": "ENCODER_NONROOT_GLOBAL_STORAGE",
    "invalid heading sentinel is not exact zero": "ENCODER_HEADING_SENTINEL",
}


class PzHuman312BuildError(RuntimeError):
    """The approved 312-rig conversion cannot be trusted or published."""


@dataclasses.dataclass(frozen=True)
class BuildConfig:
    dataset_root: Path
    freeze_root: Path
    output_root: Path
    mode: str = "prototype"
    workers: int = 24
    source_rehash_workers: int = 16
    visual_gate_path: Path | None = None
    anomaly_allowlist_path: Path | None = None
    update_link: bool = True

    def resolved(self) -> "BuildConfig":
        mode = str(self.mode)
        if mode not in {"prototype", "full"}:
            raise PzHuman312BuildError(
                f"mode must be 'prototype' or 'full', got {mode!r}"
            )
        workers = int(self.workers)
        rehash_workers = int(self.source_rehash_workers)
        if workers <= 0 or rehash_workers <= 0:
            raise PzHuman312BuildError("worker counts must be positive")
        gate = (
            None
            if self.visual_gate_path is None
            else self.visual_gate_path.expanduser().resolve()
        )
        allowlist = (
            None
            if self.anomaly_allowlist_path is None
            else self.anomaly_allowlist_path.expanduser().resolve()
        )
        if mode == "full" and gate is None:
            raise PzHuman312BuildError("full mode requires --visual-gate")
        if mode == "prototype" and allowlist is not None:
            raise PzHuman312BuildError(
                "prototype mode forbids anomaly filtering; omit --anomaly-allowlist"
            )
        return dataclasses.replace(
            self,
            dataset_root=self.dataset_root.expanduser().resolve(),
            freeze_root=self.freeze_root.expanduser().resolve(),
            output_root=self.output_root.expanduser().absolute(),
            mode=mode,
            workers=workers,
            source_rehash_workers=rehash_workers,
            visual_gate_path=gate,
            anomaly_allowlist_path=allowlist,
        )


def default_build_config(repo_root: str | Path = ".") -> BuildConfig:
    root = Path(repo_root).expanduser().resolve()
    return BuildConfig(
        dataset_root=root / "dataset",
        freeze_root=root / "dataset/ktjd17_freeze",
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


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_identifier(value: Any, *, label: str) -> str:
    identifier = str(value)
    if (
        SAFE_IDENTIFIER_PATTERN.fullmatch(identifier) is None
        or identifier in {".", ".."}
    ):
        raise PzHuman312BuildError(f"unsafe {label}: {identifier!r}")
    return identifier


def _read_regular_bytes(
    path: str | Path,
    *,
    label: str,
    require_read_only: bool = False,
) -> bytes:
    source = Path(path).expanduser().absolute()
    observed = source.lstat()
    if (
        source.is_symlink()
        or not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
        or (require_read_only and int(observed.st_mode) & 0o222)
    ):
        qualifier = "read-only single-link" if require_read_only else "single-link"
        raise PzHuman312BuildError(
            f"{label} is not a {qualifier} regular file: {source}"
        )
    descriptor = os.open(
        source,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        if (
            int(opened.st_dev),
            int(opened.st_ino),
            int(opened.st_size),
            int(opened.st_mtime_ns),
            int(opened.st_nlink),
        ) != (
            int(observed.st_dev),
            int(observed.st_ino),
            int(observed.st_size),
            int(observed.st_mtime_ns),
            int(observed.st_nlink),
        ):
            raise PzHuman312BuildError(
                f"{label} changed before descriptor capture: {source}"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
        descriptor_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = source.lstat()
    if (
        int(descriptor_after.st_dev),
        int(descriptor_after.st_ino),
        int(descriptor_after.st_size),
        int(descriptor_after.st_mtime_ns),
        int(descriptor_after.st_nlink),
    ) != (
        int(observed.st_dev),
        int(observed.st_ino),
        int(observed.st_size),
        int(observed.st_mtime_ns),
        int(observed.st_nlink),
    ) or (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_mtime_ns),
        int(after.st_nlink),
    ) != (
        int(observed.st_dev),
        int(observed.st_ino),
        int(observed.st_size),
        int(observed.st_mtime_ns),
        int(observed.st_nlink),
    ):
        raise PzHuman312BuildError(f"{label} changed while being read: {source}")
    return payload


def _json_from_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise PzHuman312BuildError(f"cannot decode {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise PzHuman312BuildError(f"{label} root is not an object")
    return value


def _contained_path(root: Path, relative: str | Path, *, label: str) -> Path:
    base = root.resolve()
    relpath = Path(relative)
    if relpath.is_absolute() or ".." in relpath.parts:
        raise PzHuman312BuildError(f"unsafe {label} relative path: {relative}")
    candidate = (base / relpath).resolve(strict=False)
    if not candidate.is_relative_to(base) or candidate == base:
        raise PzHuman312BuildError(f"{label} escapes generation root: {relative}")
    return candidate


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise PzHuman312BuildError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PzHuman312BuildError(f"JSON root is not an object: {path}")
    return value


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise PzHuman312BuildError(
                        f"blank JSONL row at {path}:{line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise PzHuman312BuildError(
                        f"JSONL row is not an object at {path}:{line_number}"
                    )
                yield value
    except PzHuman312BuildError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise PzHuman312BuildError(f"cannot read JSONL {path}: {exc}") from exc


def _write_json(path: Path, value: Any) -> None:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(_canonical_json(record).decode("utf-8") + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_lines(path: Path, values: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(str(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_bytes_atomic(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, path)
    _fsync_directory(path.parent)
    return _sha256_bytes(payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_freeze_binding() -> dict[str, Any]:
    return {
        "generation_id": FROZEN_SCHEMA_GENERATION_ID,
        "generation_json_sha256": FREEZE_GENERATION_SHA256,
        "schema_sha256": EXPECTED_FROZEN_SCHEMA_SHA256,
        "train_block_gains_sha256": FROZEN_STATS_SHA256,
        "source_plan_commit": SOURCE_PLAN_COMMIT,
    }


def _verify_freeze(root: Path) -> dict[str, Any]:
    freeze_root = root.expanduser().resolve()
    generation = verify_generation_file_closure(freeze_root)
    binding = _expected_freeze_binding()
    if (
        freeze_root.name != FROZEN_SCHEMA_GENERATION_ID
        or _sha256_file(freeze_root / "generation.json") != FREEZE_GENERATION_SHA256
        or _sha256_file(freeze_root / "schema.json") != EXPECTED_FROZEN_SCHEMA_SHA256
        or _sha256_file(freeze_root / "stats/train_block_gains.npz")
        != FROZEN_STATS_SHA256
        or generation.get("generation_id") != FROZEN_SCHEMA_GENERATION_ID
        or generation.get("source_plan_commit") != SOURCE_PLAN_COMMIT
        or generation.get("freeze_authorized") is not True
        or generation.get("full_build_may_start") is not True
    ):
        raise PzHuman312BuildError("approved frozen encoder generation drifted")
    return binding


def _file_manifest(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise PzHuman312BuildError(f"symlink inside generation: {path}")
        if path.is_file():
            relpath = path.relative_to(root).as_posix()
            if relpath == "generation.json":
                continue
            result[relpath] = {
                "sha256": _sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
    return result


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    for directory in sorted(
        (item for item in root.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        _fsync_directory(directory)
    _fsync_directory(root)


def _freeze_tree(root: Path) -> None:
    entries = sorted(root.rglob("*"), key=lambda path: len(path.parts), reverse=True)
    for path in entries:
        observed = path.lstat()
        if not (stat.S_ISREG(observed.st_mode) or stat.S_ISDIR(observed.st_mode)):
            raise PzHuman312BuildError(
                f"cannot freeze non-regular generation entry: {path}"
            )
        os.chmod(path, int(observed.st_mode) & ~0o222)
    observed_root = root.lstat()
    if not stat.S_ISDIR(observed_root.st_mode):
        raise PzHuman312BuildError(f"generation root is not a directory: {root}")
    os.chmod(root, int(observed_root.st_mode) & ~0o222)
    _fsync_directory(root.parent)


def _replace_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() and not link.is_symlink():
        raise PzHuman312BuildError(f"refusing to replace non-symlink {link}")
    temporary = link.parent / f".{link.name}.{uuid.uuid4().hex}.tmp"
    os.symlink(os.path.relpath(target, start=link.parent), temporary)
    os.replace(temporary, link)
    _fsync_directory(link.parent)


def _copy_regular_file(source: Path, target: Path, *, expected_sha256: str) -> str:
    if source.is_symlink() or not source.is_file():
        raise PzHuman312BuildError(f"copy source is not a regular file: {source}")
    observed = _sha256_file(source)
    if observed != expected_sha256:
        raise PzHuman312BuildError(
            f"copy source hash drifted: {source}: {observed} != {expected_sha256}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    shutil.copyfile(source, temporary)
    if _sha256_file(temporary) != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise PzHuman312BuildError(f"copy target hash mismatch: {target}")
    os.replace(temporary, target)
    return expected_sha256


def _sanitize_provenance_value(
    value: Any, *, logical_roots: Mapping[str, Path]
) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _sanitize_provenance_value(item, logical_roots=logical_roots)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _sanitize_provenance_value(item, logical_roots=logical_roots)
            for item in value
        ]
    if isinstance(value, str) and (
        value.startswith("/") or any(marker in value for marker in _MACHINE_PATH_MARKERS)
    ):
        candidate = Path(value).expanduser().resolve(strict=False)
        for label, root in logical_roots.items():
            try:
                relative = candidate.relative_to(root.resolve())
            except ValueError:
                continue
            return f"{label}/{relative.as_posix()}"
        return f"external/{candidate.name}"
    return value


def _text_scalar(value: str) -> np.ndarray:
    text = str(value)
    return np.asarray(text, dtype=f"<U{max(1, len(text))}")


def _copy_sanitized_skeleton(
    source: Path,
    target: Path,
    *,
    expected_sha256: str,
    logical_roots: Mapping[str, Path],
) -> str:
    if source.is_symlink() or not source.is_file():
        raise PzHuman312BuildError(f"skeleton source is not regular: {source}")
    if _sha256_file(source) != expected_sha256:
        raise PzHuman312BuildError(f"skeleton source hash drifted: {source}")
    with np.load(source, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]).copy() for key in archive.files}
    json_fields = {
        "heading_payload_provenance",
        "source_to_canonical_provenance",
        "position_geometry_provenance",
        "conditioning_authority",
        "unit_metadata",
        "joint_map_metadata",
    }
    if "source_rest_path" in payload:
        cleaned = _sanitize_provenance_value(
            str(np.asarray(payload["source_rest_path"]).item()),
            logical_roots=logical_roots,
        )
        payload["source_rest_path"] = _text_scalar(str(cleaned))
    for key in sorted(json_fields & set(payload)):
        try:
            decoded = json.loads(str(np.asarray(payload[key]).item()))
        except Exception as exc:  # noqa: BLE001
            raise PzHuman312BuildError(
                f"cannot sanitize skeleton metadata {source}:{key}: {exc}"
            ) from exc
        cleaned = _sanitize_provenance_value(decoded, logical_roots=logical_roots)
        payload[key] = _text_scalar(_canonical_json(cleaned).decode("utf-8"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target_hash = write_npz_atomic(target, payload)
    for key, array in payload.items():
        if np.asarray(array).dtype.kind in {"U", "S"}:
            text = "\n".join(str(value) for value in np.asarray(array).reshape(-1).tolist())
            if any(marker in text for marker in _MACHINE_PATH_MARKERS):
                raise PzHuman312BuildError(
                    f"machine path survived skeleton sanitization: {target}:{key}"
                )
    before = load_skeleton(source)
    after = load_skeleton(target)
    comparisons = (
        before.rig_id == after.rig_id,
        before.source_family == after.source_family,
        before.topology_family == after.topology_family,
        before.joint_names == after.joint_names,
        np.array_equal(before.parents, after.parents),
        np.array_equal(before.P_rest_global, after.P_rest_global),
        np.array_equal(before.R_rest_global, after.R_rest_global),
        np.array_equal(before.R_rest_local, after.R_rest_local),
        np.array_equal(before.offset_parent_local, after.offset_parent_local),
        np.array_equal(before.rotation_source_kind, after.rotation_source_kind),
        before.heading_carrier_joint == after.heading_carrier_joint,
        np.array_equal(before.u_forward_local, after.u_forward_local),
        np.array_equal(before.source_to_canonical_C, after.source_to_canonical_C),
        before.source_to_canonical_alpha == after.source_to_canonical_alpha,
        np.array_equal(before.source_to_canonical_o, after.source_to_canonical_o),
        before.s_rig == after.s_rig,
        before.artifact_status == after.artifact_status,
        before.metadata.get("heading_payload_provenance")
        == after.metadata.get("heading_payload_provenance"),
    )
    if not all(comparisons) or _sha256_file(target) != target_hash:
        raise PzHuman312BuildError(
            f"skeleton semantics changed during provenance sanitization: {source}"
        )
    return target_hash


def _project_clip(record: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "clip_id",
        "rig_id",
        "source",
        "split",
        "split_protocol",
        "topology_family",
        "topology_distance_bucket",
    }
    missing = sorted(required - set(record))
    if missing:
        raise PzHuman312BuildError(
            f"parent clip {record.get('clip_id')} lacks {missing}"
        )
    source = dict(record["source"])
    for key in ("family", "path", "slice_frames"):
        if key not in source:
            raise PzHuman312BuildError(
                f"parent clip {record.get('clip_id')} source lacks {key}"
            )
    clip_id = _safe_identifier(record["clip_id"], label="clip identifier")
    rig_id = _safe_identifier(record["rig_id"], label="rig identifier")
    return {
        "clip_id": clip_id,
        "rig_id": rig_id,
        "source": source,
        "split": str(record["split"]),
        "split_protocol": str(record["split_protocol"]),
        "topology_family": str(record["topology_family"]),
        "topology_distance_bucket": str(record["topology_distance_bucket"]),
    }


def _project_rig(record: Mapping[str, Any]) -> dict[str, Any]:
    required = {"rig_id", "source_family", "joint_map", "rest_pose"}
    missing = sorted(required - set(record))
    if missing:
        raise PzHuman312BuildError(
            f"parent rig {record.get('rig_id')} lacks {missing}"
        )
    rig_id = _safe_identifier(record["rig_id"], label="rig identifier")
    return {
        "rig_id": rig_id,
        "source_family": str(record["source_family"]),
        "joint_map": dict(record["joint_map"]),
        "rest_pose": dict(record["rest_pose"]),
    }


def _source_relpath(
    record: Mapping[str, Any], *, family: str, source_root: Path
) -> str:
    if family == "planetzoo":
        source = Path(str(record["source_path"])).resolve()
        try:
            relative = source.relative_to(source_root)
        except ValueError as exc:
            raise PzHuman312BuildError(
                f"PlanetZoo source escaped approved root: {source}"
            ) from exc
    elif family == "motionstreamer272":
        relative = Path(str(record["source_relpath"]))
        source = (source_root / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts:
            raise PzHuman312BuildError(
                f"unsafe Human source relative path: {relative}"
            )
    else:
        raise PzHuman312BuildError(f"unsupported source family {family!r}")
    if source.parent != source_root and family == "motionstreamer272":
        raise PzHuman312BuildError(f"Human source escaped flat source root: {source}")
    return relative.as_posix()


def _parent_hashes(parent_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ("clips.jsonl", "rigs.jsonl"):
        path = parent_root / name
        if path.is_symlink() or not path.is_file():
            raise PzHuman312BuildError(f"missing regular parent manifest {path}")
        result[name] = _sha256_file(path)
    return result


def _load_research_source_approval(
    dataset_root: Path,
    *,
    generation_link_name: str,
    approval_link_name: str,
    expected_clip_count: int,
) -> tuple[Path, dict[str, Any]]:
    """Validate audit artifacts without pinning host/runtime identity.

    The research contract keeps the immutable file/hash closure and accepted
    clip count, but deliberately ignores st_dev/inode, runtime fingerprints,
    and the audit program's own source hash.  Each motion source is SHA-256
    checked again by the conversion worker that actually consumes it.
    """

    generation_link = dataset_root / generation_link_name
    approval_link = dataset_root / approval_link_name
    if not generation_link.is_symlink() or not approval_link.is_symlink():
        raise PzHuman312BuildError(
            f"source-audit links are incomplete: {generation_link_name}"
        )
    generation_root = generation_link.resolve(strict=True)
    approval_path = approval_link.resolve(strict=True)
    generation = _load_json(generation_root / "generation.json")
    approval = _load_json(approval_path)
    expected_files = generation.get("files")
    observed_files = _file_manifest(generation_root)
    generation_relpath = Path(str(approval.get("generation_relpath", "")))
    if (
        not isinstance(expected_files, Mapping)
        or observed_files
        != {str(path): dict(metadata) for path, metadata in expected_files.items()}
        or generation.get("generation_id") != generation_root.name
        or approval.get("generation_id") != generation_root.name
        or generation_relpath.is_absolute()
        or ".." in generation_relpath.parts
        or (dataset_root / generation_relpath).resolve() != generation_root
        or approval_path.stem != approval.get("generation_content_sha256")
        or _sha256_file(generation_root / "generation.json")
        != approval.get("generation_json_sha256")
        or generation.get("authority_sha256") != approval.get("authority_sha256")
        or approval.get("status") != "pass"
        or approval.get("prototype_conversion_authorized") is not True
        or approval.get("full_conversion_authorized") is not False
        or int(approval.get("deep_validated_count", -1)) != expected_clip_count
    ):
        raise PzHuman312BuildError(
            f"research source-audit approval drifted: {generation_link_name}"
        )
    return generation_root, {
        **approval,
        "generation_root": str(generation_root),
        "approval_path": str(approval_path),
    }


def _load_scope(cfg: BuildConfig) -> dict[str, Any]:
    freeze_binding = _verify_freeze(cfg.freeze_root)
    pz_generation, pz_active = _load_research_source_approval(
        cfg.dataset_root,
        generation_link_name="ktjd17_pz_source_audit",
        approval_link_name="ktjd17_pz_source_audit_approval",
        expected_clip_count=EXPECTED_PZ_CLIP_COUNT,
    )
    human_generation, human_active = _load_research_source_approval(
        cfg.dataset_root,
        generation_link_name="ktjd17_human_source_audit",
        approval_link_name="ktjd17_human_source_audit_approval",
        expected_clip_count=EXPECTED_HUMAN_CLIP_COUNT,
    )
    pz_authority = _load_json(pz_generation / "authority.json")
    human_authority = _load_json(human_generation / "authority.json")
    parent_root = Path(str(pz_authority["parent_manifest_root"])).resolve()
    if parent_root != Path(str(human_authority["parent_manifest_root"])).resolve():
        raise PzHuman312BuildError("PZ and Human approvals bind different parents")
    hashes_before = _parent_hashes(parent_root)
    pz_root = Path(str(pz_authority["pz_bvh_root"])).resolve()
    human_root = Path(str(human_authority["source_root"])).resolve()

    audit_records: dict[str, dict[str, Any]] = {}
    family_counts: Counter[str] = Counter()
    rig_ids: set[str] = set()
    for family, path, expected in (
        (
            "planetzoo",
            pz_generation / "qa/pz_source_audit.jsonl",
            EXPECTED_PZ_CLIP_COUNT,
        ),
        (
            "motionstreamer272",
            human_generation / "qa/human_source_audit.jsonl",
            EXPECTED_HUMAN_CLIP_COUNT,
        ),
    ):
        count = 0
        source_root = pz_root if family == "planetzoo" else human_root
        for raw in _iter_jsonl(path):
            count += 1
            clip_id = _safe_identifier(
                raw.get("clip_id"), label="approved clip identifier"
            )
            rig_id = _safe_identifier(
                raw.get("rig_id"), label="approved rig identifier"
            )
            if raw.get("status") != "pass" or raw.get("source_family") != family:
                raise PzHuman312BuildError(
                    f"approved audit row is not a pass for {clip_id}"
                )
            if clip_id in audit_records:
                raise PzHuman312BuildError(f"duplicate approved clip id {clip_id}")
            source_relpath = _source_relpath(
                raw, family=family, source_root=source_root
            )
            projected = dict(raw)
            projected["clip_id"] = clip_id
            projected["rig_id"] = rig_id
            projected["source_relpath"] = source_relpath
            audit_records[clip_id] = projected
            family_counts[family] += 1
            rig_ids.add(rig_id)
        if count != expected:
            raise PzHuman312BuildError(
                f"{family} approved row count {count} != {expected}"
            )
    if family_counts != Counter(
        {
            "planetzoo": EXPECTED_PZ_CLIP_COUNT,
            "motionstreamer272": EXPECTED_HUMAN_CLIP_COUNT,
        }
    ):
        raise PzHuman312BuildError(f"source-family scope drifted: {family_counts}")
    if len(audit_records) != EXPECTED_CLIP_COUNT or len(rig_ids) != EXPECTED_RIG_COUNT:
        raise PzHuman312BuildError(
            f"approved scope is {len(audit_records)} clips/{len(rig_ids)} rigs"
        )

    parent_clips: dict[str, dict[str, Any]] = {}
    for raw in _iter_jsonl(parent_root / "clips.jsonl"):
        clip_id = _safe_identifier(raw.get("clip_id"), label="parent clip identifier")
        if clip_id in audit_records:
            parent_clips[clip_id] = _project_clip(raw)
    if set(parent_clips) != set(audit_records):
        raise PzHuman312BuildError(
            f"approved/parent clip scope mismatch: missing={len(set(audit_records)-set(parent_clips))}"
        )
    parent_rigs: dict[str, dict[str, Any]] = {}
    for raw in _iter_jsonl(parent_root / "rigs.jsonl"):
        rig_id = _safe_identifier(raw.get("rig_id"), label="parent rig identifier")
        if rig_id in rig_ids:
            parent_rigs[rig_id] = _project_rig(raw)
    if set(parent_rigs) != rig_ids:
        raise PzHuman312BuildError("approved/parent rig scope mismatch")

    for clip_id, audit in audit_records.items():
        clip = parent_clips[clip_id]
        family = str(audit["source_family"])
        source_root = pz_root if family == "planetzoo" else human_root
        expected_path = (source_root / str(audit["source_relpath"])).resolve()
        if (
            clip["rig_id"] != str(audit["rig_id"])
            or clip["source"]["family"] != family
            or Path(str(clip["source"]["path"])).resolve() != expected_path
            or clip["split"] != str(audit["split"])
            or clip["topology_family"] != str(audit["topology_family"])
            or list(clip["source"]["slice_frames"])
            != list(audit.get("slice_frames", [0, audit["T_src"]]))
        ):
            raise PzHuman312BuildError(
                f"parent/audit source binding drifted: {clip_id}"
            )

    skeleton_paths: dict[str, str] = {}
    skeleton_hashes: dict[str, str] = {}
    for rig_id in sorted(rig_ids):
        generation = human_generation if rig_id == "HML3D_Human" else pz_generation
        path = generation / "skeletons" / f"{rig_id}.npz"
        skeleton = load_skeleton(path)
        if skeleton.rig_id != rig_id:
            raise PzHuman312BuildError(f"skeleton identity drifted: {rig_id}")
        expected_family = parent_rigs[rig_id]["source_family"]
        if skeleton.source_family != expected_family:
            raise PzHuman312BuildError(f"skeleton family drifted: {rig_id}")
        skeleton_paths[rig_id] = str(path)
        skeleton_hashes[rig_id] = skeleton.sha256

    if _parent_hashes(parent_root) != hashes_before:
        raise PzHuman312BuildError("parent manifest changed while loading scope")
    return {
        "pz_active": pz_active,
        "human_active": human_active,
        "pz_generation": str(pz_generation),
        "human_generation": str(human_generation),
        "parent_root": str(parent_root),
        "parent_hashes": hashes_before,
        "source_roots": {
            "planetzoo": str(pz_root),
            "motionstreamer272": str(human_root),
        },
        "audit_records": audit_records,
        "parent_clips": parent_clips,
        "parent_rigs": parent_rigs,
        "skeleton_paths": skeleton_paths,
        "skeleton_hashes": skeleton_hashes,
        "freeze_binding": freeze_binding,
    }


def _audit_binding(active: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "status",
        "approval_version",
        "generation_id",
        "generation_content_sha256",
        "authority_sha256",
        "source_snapshot_sha256",
        "deep_validated_count",
        "prototype_conversion_authorized",
        "full_conversion_authorized",
        "generation_root",
        "approval_path",
    )
    return {key: active.get(key) for key in keys}


def _paths_overlap(left: Path, right: Path) -> bool:
    a = left.resolve(strict=False)
    b = right.resolve(strict=False)
    return a == b or a.is_relative_to(b) or b.is_relative_to(a)


def _validate_output_namespace(
    cfg: BuildConfig,
    scope: Mapping[str, Any],
    *,
    generation_directory: str,
) -> Path:
    output = cfg.output_root
    if output.is_symlink() or not output.is_dir() or output.resolve() != output:
        raise PzHuman312BuildError(
            f"output root must be an existing non-symlink canonical directory: {output}"
        )
    generations = _contained_path(
        output, generation_directory, label="generation namespace"
    )
    if generations.exists() and (
        generations.is_symlink()
        or not generations.is_dir()
        or generations.resolve() != generations
    ):
        raise PzHuman312BuildError(
            f"generation namespace is not a canonical directory: {generations}"
        )
    protected = [
        cfg.freeze_root,
        Path(str(scope["parent_root"])),
        Path(str(scope["pz_generation"])),
        Path(str(scope["human_generation"])),
        *[Path(str(value)) for value in scope["source_roots"].values()],
    ]
    conflicts = [str(path) for path in protected if _paths_overlap(generations, path)]
    if conflicts:
        raise PzHuman312BuildError(
            f"generation namespace overlaps protected inputs: {conflicts}"
        )
    return generations


def _parent_manifest_relpath(cfg: BuildConfig, scope: Mapping[str, Any]) -> str:
    parent = Path(str(scope["parent_root"])).resolve()
    try:
        relative = parent.relative_to(cfg.dataset_root)
    except ValueError as exc:
        raise PzHuman312BuildError(
            "parent manifest is not representable relative to dataset root"
        ) from exc
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise PzHuman312BuildError("unsafe parent manifest relative provenance")
    return relative.as_posix()


def _final_scope_recheck(
    cfg: BuildConfig, scope: Mapping[str, Any]
) -> dict[str, Any]:
    """Record lightweight research provenance after per-clip SHA checking."""
    approved = {
        "planetzoo": scope["pz_active"],
        "motionstreamer272": scope["human_active"],
    }
    expected_counts = {
        "planetzoo": EXPECTED_PZ_CLIP_COUNT,
        "motionstreamer272": EXPECTED_HUMAN_CLIP_COUNT,
    }
    evidence: dict[str, Any] = {}
    for family in SOURCE_FAMILIES:
        current = approved[family]
        evidence[family] = {
            "generation_id": current["generation_id"],
            "generation_content_sha256": current["generation_content_sha256"],
            "source_snapshot_sha256": current["source_snapshot_sha256"],
            "validated_count": expected_counts[family],
            "verification": "source_sha256_checked_by_conversion_worker",
        }
    if _parent_hashes(Path(str(scope["parent_root"]))) != scope["parent_hashes"]:
        raise PzHuman312BuildError("parent manifest changed during conversion")
    if _verify_freeze(cfg.freeze_root) != scope["freeze_binding"]:
        raise PzHuman312BuildError("frozen encoder generation changed during conversion")
    for rig_id, path_text in scope["skeleton_paths"].items():
        if _sha256_file(Path(str(path_text))) != scope["skeleton_hashes"][rig_id]:
            raise PzHuman312BuildError(
                f"approved skeleton changed during conversion: {rig_id}"
            )
    return {
        "status": "pass",
        "completed_at_utc": _datetime.datetime.now(_datetime.UTC).isoformat(),
        "source_audits": evidence,
        "parent_manifest_hashes": dict(scope["parent_hashes"]),
        "freeze_binding": dict(scope["freeze_binding"]),
        "skeleton_count": len(scope["skeleton_paths"]),
    }


def _representative_ids(scope: Mapping[str, Any]) -> list[str]:
    pz_generation = Path(str(scope["pz_generation"]))
    human_generation = Path(str(scope["human_generation"]))
    pz_selection = _load_json(pz_generation / "selection/pz_representatives.json")
    human_selection = _load_json(
        human_generation / "selection/human_representative.json"
    )
    selected = [str(record["clip_id"]) for record in pz_selection.get("selected", [])]
    selected.append(str(human_selection["clip_id"]))
    if len(selected) != EXPECTED_RIG_COUNT or len(set(selected)) != len(selected):
        raise PzHuman312BuildError(
            f"representative selection is not exactly {EXPECTED_RIG_COUNT} unique clips"
        )
    audit_records = scope["audit_records"]
    selected_rigs = {str(audit_records[clip_id]["rig_id"]) for clip_id in selected}
    if len(selected_rigs) != EXPECTED_RIG_COUNT:
        raise PzHuman312BuildError("representatives do not cover every rig exactly once")
    return sorted(selected)


def _source_snapshot(path: Path, expected: Mapping[str, Any]) -> dict[str, int]:
    observed = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
        raise PzHuman312BuildError(f"source is not a single-link regular file: {path}")
    result = {
        "device": int(observed.st_dev),
        "inode": int(observed.st_ino),
        "size_bytes": int(observed.st_size),
        "mtime_ns": int(observed.st_mtime_ns),
        "nlink": int(observed.st_nlink),
    }
    if result["size_bytes"] != int(expected["source_size_bytes"]):
        raise PzHuman312BuildError(
            f"source size drifted for {path.name}: {result['size_bytes']} != "
            f"{int(expected['source_size_bytes'])}"
        )
    return result


def _read_approved_source_bytes(
    path: Path, expected: Mapping[str, Any]
) -> tuple[bytes, dict[str, int]]:
    """Read and hash exactly one approved descriptor snapshot.

    The returned bytes, rather than a later pathname reopen, are the conversion
    authority.  This closes replace/restore and in-place mutation races between
    preflight hashing and parsing.
    """

    source = path.expanduser().absolute()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise PzHuman312BuildError(f"cannot open approved source {source}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or int(before.st_nlink) != 1:
            raise PzHuman312BuildError(
                f"approved source descriptor is not a single-link regular file: {source}"
            )
        opened = {
            "device": int(before.st_dev),
            "inode": int(before.st_ino),
            "size_bytes": int(before.st_size),
            "mtime_ns": int(before.st_mtime_ns),
            "nlink": int(before.st_nlink),
        }
        if opened["size_bytes"] != int(expected["source_size_bytes"]):
            raise PzHuman312BuildError(
                f"source size drifted for {source.name}: {opened['size_bytes']} != "
                f"{int(expected['source_size_bytes'])}"
            )
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        current = source.lstat()
        after_identity = (
            int(after.st_dev),
            int(after.st_ino),
            int(after.st_size),
            int(after.st_mtime_ns),
            int(after.st_nlink),
        )
        before_identity = (
            int(before.st_dev),
            int(before.st_ino),
            int(before.st_size),
            int(before.st_mtime_ns),
            int(before.st_nlink),
        )
        pathname_identity = (
            int(current.st_dev),
            int(current.st_ino),
            int(current.st_size),
            int(current.st_mtime_ns),
            int(current.st_nlink),
        )
        if (
            after_identity != before_identity
            or pathname_identity != before_identity
            or len(payload) != int(before.st_size)
        ):
            raise PzHuman312BuildError(
                f"approved source changed while reading one descriptor: {source}"
            )
        observed_sha256 = _sha256_bytes(payload)
        if observed_sha256 != str(expected["source_sha256"]):
            raise PzHuman312BuildError(
                f"approved source bytes drifted for {source.name}: {observed_sha256}"
            )
        return payload, opened
    finally:
        os.close(descriptor)


def _filterable_encoder_failure_code(exc: BaseException) -> str | None:
    """Map only known encoder numeric gates to stable structured codes."""
    if not isinstance(exc, Ktjd17EncoderError):
        return None
    matches = [
        code for marker, code in _FILTERABLE_ENCODER_GATES.items() if marker in str(exc)
    ]
    return matches[0] if len(matches) == 1 else None


def _is_filterable_encoder_gate(exc: BaseException) -> bool:
    """Compatibility predicate; publication still requires an exact allowlist."""
    return _filterable_encoder_failure_code(exc) is not None


def _load_anomaly_allowlist(
    path: Path | None, scope: Mapping[str, Any]
) -> dict[str, Any]:
    if path is None:
        return {
            "entries": {},
            "payload": None,
            "sha256": None,
            "entry_set_sha256": _sha256_bytes(_canonical_json([])),
        }
    payload = _read_regular_bytes(
        path, label="anomaly allowlist", require_read_only=True
    )
    document = _json_from_bytes(payload, label="anomaly allowlist")
    if set(document) != {
        "allowlist_version",
        "created_at_utc",
        "review",
        "entries",
    }:
        raise PzHuman312BuildError("anomaly allowlist schema drifted")
    raw_entries = document.get("entries")
    review = document.get("review")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise PzHuman312BuildError("anomaly allowlist must contain reviewed entries")
    entries: dict[str, dict[str, Any]] = {}
    rig_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    valid_codes = set(_FILTERABLE_ENCODER_GATES.values())
    expected_entry_keys = {
        "clip_id",
        "rig_id",
        "source_family",
        "source_sha256",
        "allowed_failure_code",
        "justification",
    }
    for raw in raw_entries:
        if not isinstance(raw, Mapping) or set(raw) != expected_entry_keys:
            raise PzHuman312BuildError("anomaly allowlist entry schema drifted")
        clip_id = _safe_identifier(raw["clip_id"], label="allowlisted clip identifier")
        rig_id = _safe_identifier(raw["rig_id"], label="allowlisted rig identifier")
        family = str(raw["source_family"])
        code = str(raw["allowed_failure_code"])
        justification = str(raw["justification"]).strip()
        if clip_id in entries:
            raise PzHuman312BuildError(f"duplicate anomaly allowlist clip: {clip_id}")
        audit = scope["audit_records"].get(clip_id)
        if (
            not isinstance(audit, Mapping)
            or str(audit.get("rig_id")) != rig_id
            or str(audit.get("source_family")) != family
            or str(audit.get("source_sha256")) != str(raw["source_sha256"])
            or family not in SOURCE_FAMILIES
            or code not in valid_codes
            or len(justification) < 16
        ):
            raise PzHuman312BuildError(
                f"anomaly allowlist source/code/justification drifted: {clip_id}"
            )
        entry = {key: raw[key] for key in sorted(expected_entry_keys)}
        entry["entry_sha256"] = _sha256_bytes(_canonical_json(entry))
        entries[clip_id] = entry
        rig_counts[rig_id] += 1
        family_counts[family] += 1
    if any(count > 1 for count in rig_counts.values()):
        raise PzHuman312BuildError(
            "anomaly allowlist is clustered: more than one rejection in a rig"
        )
    family_scope = {
        "planetzoo": EXPECTED_PZ_CLIP_COUNT,
        "motionstreamer272": EXPECTED_HUMAN_CLIP_COUNT,
    }
    if (
        len(entries) / EXPECTED_CLIP_COUNT > MAX_ISOLATED_REJECT_FRACTION
        or any(
            family_counts[family] / family_scope[family]
            > MAX_ISOLATED_REJECT_FRACTION
            for family in SOURCE_FAMILIES
        )
    ):
        raise PzHuman312BuildError("anomaly allowlist exceeds isolated-failure limits")
    canonical_entries = [entries[key] for key in sorted(entries)]
    entry_set_sha = _sha256_bytes(_canonical_json(canonical_entries))
    if (
        document.get("allowlist_version") != ANOMALY_ALLOWLIST_VERSION
        or not isinstance(document.get("created_at_utc"), str)
        or not document["created_at_utc"]
        or not isinstance(review, Mapping)
        or set(review)
        != {
            "model",
            "model_reasoning_effort",
            "verdict",
            "reviewed_entry_count",
            "reviewed_entries_sha256",
            "failures",
        }
        or review.get("model") != "gpt-5.6-sol"
        or review.get("model_reasoning_effort") != "xhigh"
        or review.get("verdict") != "pass"
        or int(review.get("reviewed_entry_count", -1)) != len(entries)
        or review.get("reviewed_entries_sha256") != entry_set_sha
        or review.get("failures") != []
    ):
        raise PzHuman312BuildError(
            "anomaly allowlist lacks an exact failure-free gpt-5.6-sol xhigh review"
        )
    return {
        "entries": entries,
        "payload": payload,
        "sha256": _sha256_bytes(payload),
        "entry_set_sha256": entry_set_sha,
    }


def _prepare_human_fast(
    clip: Mapping[str, Any],
    skeleton: SkeletonData,
    audit: Mapping[str, Any],
    source_payload: bytes,
) -> PreparedMotion:
    try:
        loaded = np.load(io.BytesIO(source_payload), allow_pickle=False)
    except Exception as exc:  # noqa: BLE001
        raise PzHuman312BuildError(
            f"cannot decode approved Human source bytes for {clip['clip_id']}: {exc}"
        ) from exc
    if not isinstance(loaded, np.ndarray):
        if hasattr(loaded, "close"):
            loaded.close()
        raise PzHuman312BuildError(
            f"Human source is not one NPY array: {clip['clip_id']}"
        )
    raw = np.asarray(loaded)
    source_identity = f"human_motionstreamer272/{audit['source_relpath']}"
    parsed = parse_motionstreamer272_fixed_neutral_array(
        raw,
        source_identity=source_identity,
        joint_names=skeleton.joint_names,
        parents=skeleton.parents,
        P_rest_global=skeleton.P_rest_global,
        offset_parent_local=skeleton.offset_parent_local,
        rest_authority=f"skeletons/{skeleton.rig_id}.npz",
    )
    independent = independent_motionstreamer272_decode(raw, skeleton.parents)
    crosscheck = {
        "production_vs_independent_root_translation_max_abs": float(
            np.max(
                np.abs(
                    np.asarray(parsed.root_translation, dtype=np.float64)
                    - np.asarray(independent["root_translation"], dtype=np.float64)
                )
            )
        ),
        "production_vs_independent_local_rotation_max_abs": float(
            np.max(
                np.abs(
                    np.asarray(parsed.local_rotations, dtype=np.float64)
                    - np.asarray(independent["local_rotations"], dtype=np.float64)
                )
            )
        ),
        "production_vs_independent_source_position_max_abs": float(
            np.max(
                np.abs(
                    np.asarray(parsed.source_positions, dtype=np.float64)
                    - np.asarray(independent["positions"], dtype=np.float64)
                )
            )
        ),
    }
    if max(crosscheck.values()) > 1e-12:
        raise PzHuman312BuildError(
            f"Human production/independent decoder mismatch for {clip['clip_id']}: "
            f"{crosscheck}"
        )
    return PreparedMotion(
        clip_id=str(clip["clip_id"]),
        rig_id=str(clip["rig_id"]),
        source_family="motionstreamer272",
        topology_family="human",
        fps_src=float(parsed.fps),
        root_positions=np.asarray(parsed.root_translation, dtype=np.float64),
        local_rotations=np.asarray(parsed.local_rotations, dtype=np.float64),
        source_positions_diagnostic=np.asarray(parsed.source_positions, dtype=np.float64),
        source_global_rotations=np.asarray(parsed.global_rotations, dtype=np.float64),
        source_parser_metrics={
            **dict(parsed.diagnostics),
            **crosscheck,
            "approved_audit_source_parser_fk_max_norm": float(
                audit.get("metrics", {}).get("source_parser_fk_max_norm", 0.0)
            ),
        },
        provenance={
            "source_relpath": str(audit["source_relpath"]),
            "source_sha256": str(audit["source_sha256"]),
            "rotation_authority": (
                "motionstreamer272_real_local_rotation_channels_140_272"
            ),
            "position_authority": (
                "current_btjd_fixed_neutral_offsets_plus_motionstreamer_root"
            ),
            "production_decoder": (
                "human_source_parser.parse_motionstreamer272_fixed_neutral_array"
            ),
            "independent_crosscheck_decoder": (
                "human312_audit.independent_motionstreamer272_decode"
            ),
            "stable_source_snapshot": "single_fd_sha256_then_in_memory_npy_parse",
            "legacy_btjd13_rotation_used": False,
            "position_ik_used": False,
        },
    )


def _prepare_planetzoo_snapshot(
    clip: Mapping[str, Any],
    rig: Mapping[str, Any],
    skeleton: SkeletonData,
    audit: Mapping[str, Any],
    source_payload: bytes,
) -> PreparedMotion:
    """Parse the exact approved BVH bytes through the frozen production parser."""

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix="ktjd17-pz-source-", suffix=".bvh", delete=False
        ) as handle:
            handle.write(source_payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name).resolve()
        snapshot_clip = {
            **dict(clip),
            "source": {**dict(clip["source"]), "path": str(temporary_path)},
        }
        prepared = prepare_manifest_clip(
            snapshot_clip,
            rig,
            skeleton,
            conditioning_catalog=None,
        )
        expected_sha256 = str(audit["source_sha256"])
        if (
            prepared.provenance.get("source_sha256") != expected_sha256
            or prepared.provenance.get("source_rest_sha256") != expected_sha256
        ):
            raise PzHuman312BuildError(
                f"private PZ snapshot hash drifted for {clip['clip_id']}"
            )
        logical_source = f"planetzoo_stage2/{audit['source_relpath']}"
        return dataclasses.replace(
            prepared,
            provenance={
                **prepared.provenance,
                "source_path": logical_source,
                "source_rest_path": logical_source,
                "source_sha256": expected_sha256,
                "source_rest_sha256": expected_sha256,
                "stable_source_snapshot": (
                    "single_fd_sha256_then_private_bvh_snapshot_parse"
                ),
            },
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _fk(
    parents: np.ndarray,
    root_positions: np.ndarray,
    global_rotations: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    frames, joints = global_rotations.shape[:2]
    result = np.empty((frames, joints, 3), dtype=np.float64)
    result[:, 0] = root_positions
    for child in range(1, joints):
        parent = int(parents[child])
        result[:, child] = result[:, parent] + np.einsum(
            "tij,j->ti", global_rotations[:, parent], offsets[child]
        )
    return result


def _validate_stored_artifact(
    path: Path,
    *,
    clip_id: str,
    skeleton: SkeletonData,
    expected_sha256: str,
    fps_target: float,
) -> dict[str, float | int]:
    if _sha256_file(path) != expected_sha256:
        raise PzHuman312BuildError(f"stored motion hash drifted: {clip_id}")
    payload = load_motion_npz(path, expected_fps_target=fps_target)
    if payload["clip_id"] != clip_id or payload["rig_id"] != skeleton.rig_id:
        raise PzHuman312BuildError(f"embedded identity drifted: {clip_id}")
    motion = np.asarray(payload["motion"], dtype=np.float64)
    heading_valid = np.asarray(payload["heading_valid"], dtype=bool)
    if motion.ndim != 3 or motion.shape[1:] != (len(skeleton.parents), 17):
        raise PzHuman312BuildError(f"stored shape drifted: {clip_id}: {motion.shape}")
    if not np.isfinite(motion).all():
        raise PzHuman312BuildError(f"stored motion is non-finite: {clip_id}")
    if np.any(motion[:, 1:, 13:17] != 0.0):
        raise PzHuman312BuildError(f"non-root ch13:17 is nonzero: {clip_id}")
    if np.any(~np.isin(motion[..., 12], [0.0, 1.0])):
        raise PzHuman312BuildError(f"contact is not binary: {clip_id}")
    if np.any(motion[~heading_valid, 0, 15:17] != 0.0):
        raise PzHuman312BuildError(f"invalid heading sentinel drifted: {clip_id}")
    heading_unit = (
        float(
            np.max(
                np.abs(
                    np.linalg.norm(motion[heading_valid, 0, 15:17], axis=-1)
                    - 1.0
                )
            )
        )
        if np.any(heading_valid)
        else 0.0
    )
    if heading_unit > 2e-6:
        raise PzHuman312BuildError(f"heading unit gate failed: {clip_id}")
    direct = motion[..., 0:3].copy()
    direct[..., 0] += motion[:, 0, 13][:, None]
    direct[..., 2] += motion[:, 0, 14][:, None]
    delta = independent_decode_column_cont6d(motion[..., 3:9])
    raw_global = np.matmul(delta, skeleton.R_rest_global[None])
    global_rotations = np.empty_like(raw_global)
    kinds = skeleton.rotation_source_kind.astype(str)
    for joint in range(len(kinds)):
        if kinds[joint] == "animated_dof":
            global_rotations[:, joint] = raw_global[:, joint]
        elif joint == 0:
            global_rotations[:, joint] = skeleton.R_rest_local[0]
        elif kinds[joint] == "fixed_dof":
            global_rotations[:, joint] = np.matmul(
                global_rotations[:, int(skeleton.parents[joint])],
                skeleton.R_rest_local[joint],
            )
        else:
            raise PzHuman312BuildError(
                f"invalid rotation source kind at {clip_id}/{joint}: {kinds[joint]}"
            )
    fk = _fk(
        skeleton.parents,
        direct[:, 0],
        global_rotations,
        skeleton.offset_parent_local,
    )
    direct_fk = float(
        np.max(np.linalg.norm(direct - fk, axis=-1)) / skeleton.s_rig
    )
    if direct_fk > 1e-4:
        raise PzHuman312BuildError(
            f"stored direct/FK gate failed: {clip_id}: {direct_fk}"
        )
    expected_velocity = np.zeros_like(direct)
    if len(direct) >= 2:
        expected_velocity[:-1] = np.diff(direct, axis=0) * fps_target
        expected_velocity[-1] = expected_velocity[-2]
    velocity = float(
        np.max(np.abs(expected_velocity - motion[..., 9:12]))
        / (skeleton.s_rig * fps_target)
    )
    if velocity > 1e-5:
        raise PzHuman312BuildError(
            f"stored velocity gate failed: {clip_id}: {velocity}"
        )
    rest_lengths = np.asarray(
        [np.linalg.norm(skeleton.offset_parent_local[j]) for j in range(1, len(kinds))],
        dtype=np.float64,
    )
    observed_lengths = np.stack(
        [
            np.linalg.norm(
                direct[:, j] - direct[:, int(skeleton.parents[j])], axis=-1
            )
            for j in range(1, len(kinds))
        ],
        axis=-1,
    )
    rigid_edge = float(
        np.max(np.abs(observed_lengths - rest_lengths[None])) / skeleton.s_rig
    )
    if rigid_edge > 1e-4:
        raise PzHuman312BuildError(
            f"stored rigid-edge gate failed: {clip_id}: {rigid_edge}"
        )
    masks = derive_masks(
        T_valid=len(motion),
        J_phys=len(kinds),
        T_max=len(motion),
        J_max=len(kinds),
        parents=skeleton.parents,
        rotation_source_kind=skeleton.rotation_source_kind,
        heading_valid=heading_valid,
    )
    expected_channel = np.zeros((len(kinds), 17), dtype=bool)
    expected_channel[:, :13] = True
    expected_channel[0, 13:17] = True
    if (
        not np.all(masks.frame_mask)
        or not np.all(masks.joint_mask)
        or not np.array_equal(masks.channel_valid_mask, expected_channel)
        or not np.array_equal(masks.heading_valid, heading_valid)
    ):
        raise PzHuman312BuildError(f"derived mask gate failed: {clip_id}")
    return {
        "stored_T": int(motion.shape[0]),
        "stored_J": int(motion.shape[1]),
        "stored_direct_vs_fk_max_norm": direct_fk,
        "stored_velocity_max_norm_fps": velocity,
        "stored_rigid_edge_max_norm": rigid_edge,
        "stored_heading_unit_max_abs": heading_unit,
        "stored_heading_invalid_count": int(np.count_nonzero(~heading_valid)),
    }


_WORKER_RIGS: dict[str, dict[str, Any]] = {}
_WORKER_SKELETON_PATHS: dict[str, str] = {}
_WORKER_ENCODER: EncoderConfig | None = None
_WORKER_OUTPUT_ROOT: Path | None = None
_WORKER_SKELETON_CACHE: dict[str, SkeletonData] = {}
_WORKER_ALLOWED_REJECTIONS: dict[str, dict[str, Any]] = {}


def _initialize_conversion_worker(
    rigs: Mapping[str, Mapping[str, Any]],
    skeleton_paths: Mapping[str, str],
    encoder: EncoderConfig,
    output_root: str,
    allowed_rejections: Mapping[str, Mapping[str, Any]],
) -> None:
    global _WORKER_RIGS
    global _WORKER_SKELETON_PATHS
    global _WORKER_ENCODER
    global _WORKER_OUTPUT_ROOT
    global _WORKER_SKELETON_CACHE
    global _WORKER_ALLOWED_REJECTIONS
    _WORKER_RIGS = {str(key): dict(value) for key, value in rigs.items()}
    _WORKER_SKELETON_PATHS = {
        str(key): str(value) for key, value in skeleton_paths.items()
    }
    _WORKER_ENCODER = encoder
    _WORKER_OUTPUT_ROOT = Path(output_root)
    _WORKER_SKELETON_CACHE = {}
    _WORKER_ALLOWED_REJECTIONS = {
        str(key): dict(value) for key, value in allowed_rejections.items()
    }


def _conversion_worker_once(task: Mapping[str, Any]) -> dict[str, Any]:
    if _WORKER_ENCODER is None or _WORKER_OUTPUT_ROOT is None:
        raise PzHuman312BuildError("conversion worker was not initialized")
    clip = dict(task["clip"])
    audit = dict(task["audit"])
    clip_id = str(clip["clip_id"])
    rig_id = str(clip["rig_id"])
    family = str(audit["source_family"])
    source_path = Path(str(clip["source"]["path"])).resolve()
    source_payload, before = _read_approved_source_bytes(source_path, audit)
    if rig_id not in _WORKER_SKELETON_CACHE:
        _WORKER_SKELETON_CACHE[rig_id] = load_skeleton(
            _WORKER_SKELETON_PATHS[rig_id]
        )
    skeleton = _WORKER_SKELETON_CACHE[rig_id]
    rig = _WORKER_RIGS[rig_id]
    try:
        if family == "motionstreamer272":
            prepared = _prepare_human_fast(clip, skeleton, audit, source_payload)
        elif family == "planetzoo":
            prepared = _prepare_planetzoo_snapshot(
                clip, rig, skeleton, audit, source_payload
            )
        else:
            raise PzHuman312BuildError(
                f"unsupported family for {clip_id}: {family}"
            )
        observed_sha = str(prepared.provenance.get("source_sha256", ""))
        if observed_sha != str(audit["source_sha256"]):
            raise PzHuman312BuildError(
                f"source hash drifted while preparing {clip_id}: {observed_sha}"
            )
        encoded = encode_prepared_motion(prepared, skeleton, _WORKER_ENCODER)
    except Exception as exc:  # noqa: BLE001
        failure_code = _filterable_encoder_failure_code(exc)
        allowance = _WORKER_ALLOWED_REJECTIONS.get(clip_id)
        if (
            failure_code is not None
            and allowance is not None
            and allowance.get("allowed_failure_code") == failure_code
            and allowance.get("source_sha256") == audit.get("source_sha256")
            and allowance.get("rig_id") == rig_id
            and allowance.get("source_family") == family
        ):
            return _rejection_record(
                task,
                exc,
                reason_code=failure_code,
                allowlist_entry=allowance,
            )
        raise
    after = _source_snapshot(source_path, audit)
    if after != before:
        raise PzHuman312BuildError(f"source changed while converting {clip_id}")
    motion_relpath = f"motions/{_safe_identifier(clip_id, label='clip identifier')}.npz"
    motion_path = _contained_path(
        _WORKER_OUTPUT_ROOT, motion_relpath, label="motion output"
    )
    motion_sha = write_npz_atomic(motion_path, encoded.artifact_payload())
    stored_metrics = _validate_stored_artifact(
        motion_path,
        clip_id=clip_id,
        skeleton=skeleton,
        expected_sha256=motion_sha,
        fps_target=encoded.fps_target,
    )
    manifest = {
        "clip_id": clip_id,
        "rig_id": rig_id,
        "source_family": family,
        "topology_family": clip["topology_family"],
        "topology_distance_bucket": clip["topology_distance_bucket"],
        "family_role": rig_id,
        "audit_role": str(task["audit_role"]),
        "calibration_eligible": False,
        "selection_origin": str(task["selection_origin"]),
        "replaces_parent_clip_id": None,
        "split": clip["split"],
        "status": "accept",
        "reason_codes": [],
        "motion_relpath": motion_relpath,
        "motion_sha256": motion_sha,
        "skeleton_relpath": f"skeletons/{rig_id}.npz",
        "skeleton_sha256": str(task["skeleton_sha256"]),
        "fps_src": encoded.fps_src,
        "fps_target": encoded.fps_target,
        "T_src": encoded.metrics["T_src"],
        "T_target": encoded.metrics["T_target"],
        "J_phys": encoded.metrics["J_phys"],
        "resample_mode": encoded.resample_mode,
        "source_relpath": str(audit["source_relpath"]),
        "source_sha256": str(audit["source_sha256"]),
        "source_frame_slice": list(clip["source"]["slice_frames"]),
        "source_split_protocol": clip["split_protocol"],
        "rotation_authority": (
            "stage2_bvh_real_declared_euler_rotation_channels"
            if family == "planetzoo"
            else "motionstreamer272_real_local_rotation_channels_140_272"
        ),
        "legacy_btjd13_motion_used": False,
        "legacy_btjd13_rotation_used": False,
        "position_ik_used": False,
    }
    qa = {
        **{key: value for key, value in manifest.items() if key not in {"status"}},
        "status": "pass",
        "motion_size_bytes": int(motion_path.stat().st_size),
        "metrics": {**encoded.metrics, **stored_metrics},
        "source_audit_metrics": dict(audit.get("metrics", {})),
    }
    return {"status": "pass", "manifest": manifest, "qa": qa}


def _conversion_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    delays = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2)
    for index, delay in enumerate(delays):
        if delay:
            time.sleep(delay)
        try:
            return _conversion_worker_once(task)
        except FileNotFoundError:
            if index + 1 == len(delays):
                raise
    raise AssertionError("unreachable ENOENT retry state")


@contextlib.contextmanager
def _single_thread_worker_environment() -> Iterator[None]:
    keys = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ[key] = "1"
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _run_conversion_tasks(
    tasks: Sequence[Mapping[str, Any]],
    *,
    rigs: Mapping[str, Mapping[str, Any]],
    skeleton_paths: Mapping[str, str],
    encoder: EncoderConfig,
    output_root: Path,
    workers: int,
    allowed_rejections: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifests: list[dict[str, Any]] = []
    qa_records: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []

    def consume(task: Mapping[str, Any], result: Mapping[str, Any]) -> None:
        if result.get("status") == "pass":
            manifests.append(dict(result["manifest"]))
            qa_records.append(dict(result["qa"]))
            return
        # A worker may fail after atomically materializing its distinct motion
        # path (for example, during the storage readback gate).  A rejected
        # clip must not leave an unreferenced payload in the generation.
        rejected_clip_id = _safe_identifier(
            task["clip"]["clip_id"], label="rejected clip identifier"
        )
        rejected_path = _contained_path(
            output_root,
            f"motions/{rejected_clip_id}.npz",
            label="rejected motion output",
        )
        rejected_path.unlink(missing_ok=True)
        rejections.append(dict(result))

    if workers == 1:
        _initialize_conversion_worker(
            rigs,
            skeleton_paths,
            encoder,
            str(output_root),
            allowed_rejections,
        )
        for index, task in enumerate(tasks, start=1):
            try:
                result = _conversion_worker(task)
            except Exception as exc:  # noqa: BLE001
                raise PzHuman312BuildError(
                    f"unexpected conversion failure for {task['clip']['clip_id']}"
                ) from exc
            consume(task, result)
            if index % 100 == 0 or index == len(tasks):
                print(
                    f"[ktjd17-312] converted {index}/{len(tasks)}; "
                    f"accepted={len(manifests)} rejected={len(rejections)}",
                    flush=True,
                )
    else:
        context = multiprocessing.get_context("spawn")
        with _single_thread_worker_environment():
            pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=workers,
                mp_context=context,
                initializer=_initialize_conversion_worker,
                initargs=(
                    rigs,
                    skeleton_paths,
                    encoder,
                    str(output_root),
                    allowed_rejections,
                ),
            )
            pending_tasks = iter(enumerate(tasks))
            in_flight: dict[
                concurrent.futures.Future[dict[str, Any]],
                tuple[int, Mapping[str, Any]],
            ] = {}

            def submit_next() -> bool:
                try:
                    task_index, next_task = next(pending_tasks)
                except StopIteration:
                    return False
                future = pool.submit(_conversion_worker, next_task)
                in_flight[future] = (task_index, next_task)
                return True

            try:
                for _ in range(min(len(tasks), max(workers, workers * 4))):
                    submit_next()
                completed = 0
                while in_flight:
                    done, _ = concurrent.futures.wait(
                        tuple(in_flight),
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in done:
                        _, task = in_flight.pop(future)
                        try:
                            result = future.result()
                        except Exception as exc:  # noqa: BLE001
                            raise PzHuman312BuildError(
                                "unexpected conversion failure for "
                                f"{task['clip']['clip_id']}"
                            ) from exc
                        consume(task, result)
                        completed += 1
                        submit_next()
                        if completed % 500 == 0 or completed == len(tasks):
                            print(
                                f"[ktjd17-312] converted {completed}/{len(tasks)}; "
                                f"accepted={len(manifests)} rejected={len(rejections)}",
                                flush=True,
                            )
            except BaseException:
                for future in in_flight:
                    future.cancel()
                pool.shutdown(wait=True, cancel_futures=True)
                raise
            else:
                pool.shutdown(wait=True, cancel_futures=True)
    manifests.sort(key=lambda value: value["clip_id"])
    qa_records.sort(key=lambda value: value["clip_id"])
    rejections.sort(key=lambda value: value["clip_id"])
    combined = qa_records + rejections
    combined.sort(key=lambda value: value["clip_id"])
    return manifests, combined


def _rejection_record(
    task: Mapping[str, Any],
    exc: Exception,
    *,
    reason_code: str,
    allowlist_entry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    clip = task["clip"]
    return {
        "status": "reject",
        "clip_id": str(clip["clip_id"]),
        "rig_id": str(clip["rig_id"]),
        "source_family": str(task["audit"]["source_family"]),
        "topology_family": str(clip["topology_family"]),
        "topology_distance_bucket": str(clip["topology_distance_bucket"]),
        "split": str(clip["split"]),
        "source_relpath": str(task["audit"]["source_relpath"]),
        "source_sha256": str(task["audit"]["source_sha256"]),
        "reason_codes": [reason_code],
        "error_type": type(exc).__name__,
        "error": str(exc),
        "legacy_fallback_allowed": False,
        "allowlist_entry_sha256": (
            None if allowlist_entry is None else allowlist_entry.get("entry_sha256")
        ),
        "allowlist_justification": (
            None if allowlist_entry is None else allowlist_entry.get("justification")
        ),
    }


def _metric_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    names = sorted(
        {
            name
            for record in records
            if record.get("status") == "pass"
            for name, value in record.get("metrics", {}).items()
            if isinstance(value, (int, float)) and value is not None
        }
    )
    result: dict[str, Any] = {}
    for name in names:
        values = np.asarray(
            [
                float(record["metrics"][name])
                for record in records
                if record.get("status") == "pass"
                and record.get("metrics", {}).get(name) is not None
            ],
            dtype=np.float64,
        )
        if values.size:
            result[name] = {
                "count": int(values.size),
                "min": float(np.min(values)),
                "median": float(np.median(values)),
                "p99": float(np.percentile(values, 99)),
                "max": float(np.max(values)),
                "mean": float(np.mean(values)),
            }
    return result


def _deep_validate_one(task: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(str(task["root"])).resolve()
    skeleton_path = root / str(task["manifest"]["skeleton_relpath"])
    skeleton, skeleton_metrics = _deep_validate_skeleton(
        skeleton_path, str(task["manifest"]["skeleton_sha256"])
    )
    record = _deep_validate_motion(
        root=root,
        manifest=task["manifest"],
        qa=task["qa"],
        parent_clip=task["parent_clip"],
        parent_rig=task["parent_rig"],
        skeleton=skeleton,
        encoder_config=task["encoder_config"],
        rest_cache={},
    )
    record["skeleton_metrics"] = skeleton_metrics
    return record


def _run_prototype_deep_validation(
    *,
    root: Path,
    manifests: Sequence[Mapping[str, Any]],
    qa_records: Sequence[Mapping[str, Any]],
    scope: Mapping[str, Any],
    encoder_config: Mapping[str, Any],
    workers: int,
) -> list[dict[str, Any]]:
    qa_by_id = {
        str(record["clip_id"]): record
        for record in qa_records
        if record.get("status") == "pass"
    }
    tasks = [
        {
            "root": str(root),
            "manifest": dict(manifest),
            "qa": dict(qa_by_id[str(manifest["clip_id"])]),
            "parent_clip": dict(scope["parent_clips"][str(manifest["clip_id"])]),
            "parent_rig": dict(scope["parent_rigs"][str(manifest["rig_id"])]),
            "encoder_config": dict(encoder_config),
        }
        for manifest in manifests
    ]
    if workers == 1:
        records = [_deep_validate_one(task) for task in tasks]
    else:
        context = multiprocessing.get_context("spawn")
        with _single_thread_worker_environment():
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers, mp_context=context
            ) as pool:
                records = list(pool.map(_deep_validate_one, tasks, chunksize=1))
    records.sort(key=lambda record: record["clip_id"])
    if len(records) != EXPECTED_RIG_COUNT or any(
        record.get("status") != "pass" for record in records
    ):
        raise PzHuman312BuildError("312-rig deep prototype QA did not fully pass")
    return records


def _encoder_record(config: EncoderConfig) -> dict[str, Any]:
    return config.as_record()


def _require_read_only_tree(root: Path, *, label: str) -> None:
    base = root.resolve()
    if root.is_symlink() or not base.is_dir():
        raise PzHuman312BuildError(f"{label} is not a canonical directory: {root}")
    for path in [base, *sorted(base.rglob("*"))]:
        observed = path.lstat()
        if stat.S_ISLNK(observed.st_mode) or int(observed.st_mode) & 0o222:
            raise PzHuman312BuildError(f"{label} contains mutable/symlink entry: {path}")
        if not (stat.S_ISREG(observed.st_mode) or stat.S_ISDIR(observed.st_mode)):
            raise PzHuman312BuildError(f"{label} contains special entry: {path}")


def build_visual_review_expectation(
    prototype_root: str | Path,
    visual_root: str | Path,
    *,
    expected_freeze_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    prototype_path = Path(prototype_root).expanduser().resolve()
    visual_path = Path(visual_root).expanduser().resolve()
    prototype = verify_generation(prototype_path)
    visual = verify_visual_generation(visual_path)
    _require_read_only_tree(prototype_path, label="prototype generation")
    _require_read_only_tree(visual_path, label="visual generation")
    prototype_sha = _sha256_file(prototype_path / "generation.json")
    visual_sha = _sha256_file(visual_path / "generation.json")
    manifests = list(_iter_jsonl(prototype_path / "manifests/clips.jsonl"))
    visual_index = _load_json(visual_path / "visual_qa_index.json")
    index_records = visual_index.get("clips")
    if not isinstance(index_records, list):
        raise PzHuman312BuildError("visual index clips are absent")
    prototype_pairs = sorted(
        (
            _safe_identifier(record["clip_id"], label="prototype clip identifier"),
            _safe_identifier(record["rig_id"], label="prototype rig identifier"),
        )
        for record in manifests
    )
    index_by_clip: dict[str, Mapping[str, Any]] = {}
    for raw in index_records:
        if not isinstance(raw, Mapping):
            raise PzHuman312BuildError("visual index clip record is not an object")
        clip_id = _safe_identifier(raw.get("clip_id"), label="visual clip identifier")
        if clip_id in index_by_clip:
            raise PzHuman312BuildError(f"duplicate visual clip identifier: {clip_id}")
        index_by_clip[clip_id] = raw
    index_pairs = sorted(
        (
            clip_id,
            _safe_identifier(raw.get("rig_id"), label="visual rig identifier"),
        )
        for clip_id, raw in index_by_clip.items()
    )
    required_paths = ["source", "position-direct", "rotation-FK"]
    freeze_binding = dict(prototype.get("freeze_binding", {}))
    if (
        prototype.get("mode") != "prototype"
        or prototype.get("status")
        != "numeric_pass_pending_312_rig_dynamic_visual_review"
        or len(prototype_pairs) != EXPECTED_RIG_COUNT
        or len({rig_id for _, rig_id in prototype_pairs}) != EXPECTED_RIG_COUNT
        or prototype_pairs != index_pairs
        or visual.get("prototype_generation_id") != prototype_path.name
        or visual.get("prototype_generation_sha256") != prototype_sha
        or visual.get("freeze_binding") != freeze_binding
        or visual_index.get("prototype_generation_id") != prototype_path.name
        or visual_index.get("prototype_generation_sha256") != prototype_sha
        or visual_index.get("freeze_binding") != freeze_binding
        or visual_index.get("coordinate_contract") != COORDINATE_CONTRACT
        or visual_index.get("required_paths") != required_paths
        or visual_index.get("perspective_camera") is not True
        or visual_index.get("fixed_camera_across_frames_and_paths") is not True
        or visual_index.get("frame_recenter_applied") is not False
        or visual_index.get("ground_changed") is not False
        or visual_index.get("face_direction_changed") is not False
        or (
            expected_freeze_binding is not None
            and freeze_binding != dict(expected_freeze_binding)
        )
    ):
        raise PzHuman312BuildError(
            "prototype, visual, coordinate, or frozen-encoder binding drifted"
        )
    files = visual.get("files")
    if not isinstance(files, Mapping):
        raise PzHuman312BuildError("visual file manifest is absent")
    artifact_bindings: list[dict[str, Any]] = []
    for clip_id, rig_id in prototype_pairs:
        record = index_by_clip[clip_id]
        if (
            record.get("coordinate_contract") != COORDINATE_CONTRACT
            or record.get("fixed_camera_across_frames_and_paths") is not True
            or record.get("frame_recenter_applied") is not False
            or record.get("ground_changed") is not False
            or record.get("face_direction_changed") is not False
            or record.get("inspection_status")
            != "pending_human_and_codex_visual_review"
        ):
            raise PzHuman312BuildError(
                f"visual per-clip camera/coordinate contract drifted: {clip_id}"
            )
        assets: dict[str, dict[str, str]] = {}
        for asset, key, suffix in (
            ("gif", "gif_relpath", ".gif"),
            ("filmstrip", "filmstrip_relpath", "_filmstrip.png"),
            ("rest", "rest_relpath", "_rest.png"),
        ):
            relpath = str(record.get(key, ""))
            _contained_path(visual_path, relpath, label=f"{clip_id} {asset}")
            metadata = files.get(relpath)
            if (
                not relpath.startswith("clips/")
                or not relpath.endswith(suffix)
                or not isinstance(metadata, Mapping)
                or not re.fullmatch(r"[0-9a-f]{64}", str(metadata.get("sha256", "")))
            ):
                raise PzHuman312BuildError(
                    f"visual asset binding is invalid: {clip_id}/{asset}"
                )
            assets[asset] = {
                "relpath": relpath,
                "sha256": str(metadata["sha256"]),
            }
        artifact_bindings.append(
            {
                "clip_id": clip_id,
                "rig_id": rig_id,
                "paths_reviewed": required_paths,
                "perspective_camera": True,
                "fixed_camera_across_frames_and_paths": True,
                "frame_recenter_applied": False,
                "ground_changed": False,
                "face_direction_changed": False,
                "gif": assets["gif"],
                "filmstrip": assets["filmstrip"],
                "rest": assets["rest"],
            }
        )
    pair_records = [
        {"clip_id": clip_id, "rig_id": rig_id}
        for clip_id, rig_id in prototype_pairs
    ]
    coverage = {
        "reviewed_clip_count": EXPECTED_RIG_COUNT,
        "reviewed_rig_count": EXPECTED_RIG_COUNT,
        "clip_rig_set_sha256": _sha256_bytes(_canonical_json(pair_records)),
        "artifact_bindings_sha256": _sha256_bytes(
            _canonical_json(artifact_bindings)
        ),
        "required_paths": required_paths,
        "coordinate_contract": COORDINATE_CONTRACT,
        "perspective_camera": True,
        "fixed_camera_across_frames_and_paths": True,
        "frame_recenter_applied": False,
        "ground_changed": False,
        "face_direction_changed": False,
    }
    return {
        "prototype_generation_id": prototype_path.name,
        "prototype_generation_sha256": prototype_sha,
        "visual_generation_id": visual_path.name,
        "visual_generation_sha256": visual_sha,
        "freeze_binding": freeze_binding,
        "coverage": coverage,
        "artifact_bindings": artifact_bindings,
    }


def validate_visual_review(
    review: Mapping[str, Any], expectation: Mapping[str, Any]
) -> dict[str, Any]:
    expected_keys = {
        "model",
        "model_reasoning_effort",
        "review_thread_id",
        "verdict",
        "prototype_generation_id",
        "prototype_generation_sha256",
        "visual_generation_id",
        "visual_generation_sha256",
        "freeze_binding",
        "coverage",
        "artifact_reviews",
        "failures",
    }
    expected_artifacts = [
        {
            **binding,
            "status": "pass",
            "native_image_reviewed": True,
        }
        for binding in expectation["artifact_bindings"]
    ]
    if (
        set(review) != expected_keys
        or review.get("model") != "gpt-5.6-sol"
        or review.get("model_reasoning_effort") != "xhigh"
        or not isinstance(review.get("review_thread_id"), str)
        or not review["review_thread_id"]
        or review.get("verdict") != "pass"
        or review.get("prototype_generation_id")
        != expectation["prototype_generation_id"]
        or review.get("prototype_generation_sha256")
        != expectation["prototype_generation_sha256"]
        or review.get("visual_generation_id") != expectation["visual_generation_id"]
        or review.get("visual_generation_sha256")
        != expectation["visual_generation_sha256"]
        or review.get("freeze_binding") != expectation["freeze_binding"]
        or review.get("coverage") != expectation["coverage"]
        or review.get("artifact_reviews") != expected_artifacts
        or review.get("failures") != []
    ):
        raise PzHuman312BuildError(
            "visual review is not an exact 312-artifact gpt-5.6-sol xhigh PASS"
        )
    return dict(review)


def _validate_visual_gate(
    path: Path, scope: Mapping[str, Any], *, dataset_root: Path
) -> dict[str, Any]:
    payload = _read_regular_bytes(
        path, label="312 visual gate", require_read_only=True
    )
    gate = _json_from_bytes(payload, label="312 visual gate")
    expected_keys = {
        "gate_version",
        "created_at_utc",
        "verdict",
        "prototype_generation_id",
        "prototype_generation_sha256",
        "visual_generation_id",
        "visual_generation_sha256",
        "rig_count",
        "clip_count",
        "coordinate_contract",
        "freeze_binding",
        "source_audit_bindings",
        "review_binding",
        "review_json_sha256",
        "review",
        "full_conversion_authorized",
    }
    if set(gate) != expected_keys:
        raise PzHuman312BuildError("312 visual gate schema drifted")
    expected_bindings = {
        "planetzoo": {
            "generation_id": scope["pz_active"]["generation_id"],
            "generation_content_sha256": scope["pz_active"][
                "generation_content_sha256"
            ],
            "source_snapshot_sha256": scope["pz_active"][
                "source_snapshot_sha256"
            ],
        },
        "motionstreamer272": {
            "generation_id": scope["human_active"]["generation_id"],
            "generation_content_sha256": scope["human_active"][
                "generation_content_sha256"
            ],
            "source_snapshot_sha256": scope["human_active"][
                "source_snapshot_sha256"
            ],
        },
    }
    prototype_root = (dataset_root / PROTOTYPE_LINK_NAME).resolve(strict=True)
    visual_root = (
        dataset_root
        / "ktjd17_pz_human312_visual"
        / "ktjd17_visual_qa"
    ).resolve(strict=True)
    expectation = build_visual_review_expectation(
        prototype_root,
        visual_root,
        expected_freeze_binding=scope["freeze_binding"],
    )
    review = gate.get("review")
    if not isinstance(review, Mapping):
        raise PzHuman312BuildError("312 visual gate review is absent")
    validated_review = validate_visual_review(review, expectation)
    review_binding = {
        "review_thread_id": validated_review["review_thread_id"],
        "coverage": expectation["coverage"],
        "artifact_reviews_sha256": _sha256_bytes(
            _canonical_json(validated_review["artifact_reviews"])
        ),
    }
    if (
        gate["gate_version"] != VISUAL_GATE_VERSION
        or gate["verdict"] != "pass"
        or int(gate["rig_count"]) != EXPECTED_RIG_COUNT
        or int(gate["clip_count"]) != EXPECTED_RIG_COUNT
        or gate["coordinate_contract"] != COORDINATE_CONTRACT
        or gate["freeze_binding"] != scope["freeze_binding"]
        or gate["source_audit_bindings"] != expected_bindings
        or gate["review_binding"] != review_binding
        or gate["review_json_sha256"]
        != _sha256_bytes(_canonical_json(validated_review))
        or gate["prototype_generation_id"]
        != expectation["prototype_generation_id"]
        or gate["prototype_generation_sha256"]
        != expectation["prototype_generation_sha256"]
        or gate["visual_generation_id"] != expectation["visual_generation_id"]
        or gate["visual_generation_sha256"]
        != expectation["visual_generation_sha256"]
        or gate["full_conversion_authorized"] is not True
    ):
        raise PzHuman312BuildError("312 visual gate is not an exact authorized PASS")
    return {
        "record": gate,
        "payload": payload,
        "sha256": _sha256_bytes(payload),
        "expectation": expectation,
    }


def _assert_sanitized_generation_metadata(root: Path) -> None:
    forbidden = (*_MACHINE_PATH_MARKERS, _FILE_URI_MARKER)
    text_suffixes = {".json", ".jsonl", ".txt", ".md", ".yaml", ".yml"}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in text_suffixes:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise PzHuman312BuildError(
                    f"metadata file is not UTF-8: {path}"
                ) from exc
            if any(value in text for value in forbidden):
                raise PzHuman312BuildError(
                    f"machine-specific path leaked into generation metadata: {path}"
                )
        elif path.suffix.lower() == ".npz" and path.parent.name == "skeletons":
            with np.load(path, allow_pickle=False) as archive:
                for key in archive.files:
                    array = np.asarray(archive[key])
                    if array.dtype.kind not in {"U", "S"}:
                        continue
                    text = "\n".join(
                        str(value) for value in array.reshape(-1).tolist()
                    )
                    if any(value in text for value in forbidden):
                        raise PzHuman312BuildError(
                            f"machine-specific path leaked into skeleton: {path}:{key}"
                        )


def _verify_published_anomaly_allowlist(
    root: Path,
    generation: Mapping[str, Any],
    rejections: Sequence[Mapping[str, Any]],
) -> None:
    allowlist_sha = generation.get("anomaly_allowlist_sha256")
    entry_set_sha = generation.get("anomaly_allowlist_entry_set_sha256")
    path = root / "evidence/anomaly_allowlist.json"
    if allowlist_sha is None:
        if path.exists() or rejections or entry_set_sha != _sha256_bytes(_canonical_json([])):
            raise PzHuman312BuildError("zero-rejection anomaly binding drifted")
        return
    if (
        not re.fullmatch(r"[0-9a-f]{64}", str(allowlist_sha))
        or not path.is_file()
        or _sha256_file(path) != allowlist_sha
    ):
        raise PzHuman312BuildError("published anomaly allowlist hash drifted")
    document = _load_json(path)
    raw_entries = document.get("entries")
    review = document.get("review")
    if (
        document.get("allowlist_version") != ANOMALY_ALLOWLIST_VERSION
        or not isinstance(raw_entries, list)
        or not isinstance(review, Mapping)
        or review.get("model") != "gpt-5.6-sol"
        or review.get("model_reasoning_effort") != "xhigh"
        or review.get("verdict") != "pass"
        or review.get("failures") != []
    ):
        raise PzHuman312BuildError("published anomaly allowlist review drifted")
    entries: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for raw in raw_entries:
        if not isinstance(raw, Mapping):
            raise PzHuman312BuildError("published anomaly entry is invalid")
        entry = {key: raw[key] for key in sorted(raw)}
        entry["entry_sha256"] = _sha256_bytes(_canonical_json(entry))
        clip_id = _safe_identifier(entry.get("clip_id"), label="anomaly clip identifier")
        if clip_id in by_id:
            raise PzHuman312BuildError("published anomaly IDs are duplicated")
        by_id[clip_id] = entry
        entries.append(entry)
    entries.sort(key=lambda value: value["clip_id"])
    observed_entry_set_sha = _sha256_bytes(_canonical_json(entries))
    rejected_by_id = {str(record["clip_id"]): record for record in rejections}
    if (
        observed_entry_set_sha != entry_set_sha
        or review.get("reviewed_entries_sha256") != entry_set_sha
        or int(review.get("reviewed_entry_count", -1)) != len(entries)
        or set(rejected_by_id) != set(by_id)
        or any(
            rejected_by_id[clip_id].get("source_sha256")
            != entry.get("source_sha256")
            or rejected_by_id[clip_id].get("rig_id") != entry.get("rig_id")
            or rejected_by_id[clip_id].get("source_family")
            != entry.get("source_family")
            or rejected_by_id[clip_id].get("reason_codes")
            != [entry.get("allowed_failure_code")]
            or rejected_by_id[clip_id].get("allowlist_entry_sha256")
            != entry.get("entry_sha256")
            for clip_id, entry in by_id.items()
        )
    ):
        raise PzHuman312BuildError("published anomaly/rejection binding drifted")


def _generation_namespace(mode: str) -> tuple[str, str, str]:
    if mode == "prototype":
        return (
            PROTOTYPE_GENERATION_DIRECTORY,
            PROTOTYPE_LINK_NAME,
            PROTOTYPE_APPROVAL_LINK_NAME,
        )
    if mode == "full":
        return FULL_GENERATION_DIRECTORY, FULL_LINK_NAME, FULL_APPROVAL_LINK_NAME
    raise PzHuman312BuildError(f"invalid generation mode {mode!r}")


def _generation_content_evidence(
    root: str | Path,
    *,
    generation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind a frozen generation to one digest outside its own file map."""

    generation_root = Path(root).expanduser().resolve()
    _require_read_only_tree(generation_root, label="published KTJD-17 generation")
    generation_payload = _read_regular_bytes(
        generation_root / "generation.json",
        label="published KTJD-17 generation manifest",
        require_read_only=True,
    )
    observed_generation = _json_from_bytes(
        generation_payload, label="published KTJD-17 generation manifest"
    )
    if generation is not None and observed_generation != dict(generation):
        raise PzHuman312BuildError("generation changed before approval evidence")
    mode = str(observed_generation.get("mode"))
    generation_directory, _, _ = _generation_namespace(mode)
    if generation_root.parent.name != generation_directory:
        raise PzHuman312BuildError(
            "generation is outside its mode-specific approval namespace"
        )
    output_root = generation_root.parent.parent
    relative = generation_root.relative_to(output_root).as_posix()
    expected_relative = f"{generation_directory}/{generation_root.name}"
    observed_files = _file_manifest(generation_root)
    if (
        relative != expected_relative
        or observed_generation.get("generation_id") != generation_root.name
        or observed_generation.get("files") != observed_files
    ):
        raise PzHuman312BuildError("generation content evidence closure failed")
    core = {
        "build_version": BUILD_VERSION,
        "generation_id": generation_root.name,
        "generation_relpath": relative,
        "mode": mode,
        "generation_json_sha256": _sha256_bytes(generation_payload),
        "generation_json_size_bytes": len(generation_payload),
        "file_manifest_sha256": _sha256_bytes(_canonical_json(observed_files)),
        "file_count": len(observed_files),
    }
    return {
        **core,
        "generation_content_sha256": _sha256_bytes(_canonical_json(core)),
    }


def _approval_fields(
    generation: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    approved_at_utc: str,
) -> dict[str, Any]:
    return {
        "approval_version": BUILD_APPROVAL_VERSION,
        "build_version": BUILD_VERSION,
        "status": "pass",
        "mode": generation["mode"],
        "generation_id": generation["generation_id"],
        "generation_relpath": evidence["generation_relpath"],
        "generation_content_sha256": evidence["generation_content_sha256"],
        "generation_json_sha256": evidence["generation_json_sha256"],
        "generation_json_size_bytes": evidence["generation_json_size_bytes"],
        "file_manifest_sha256": evidence["file_manifest_sha256"],
        "file_count": evidence["file_count"],
        "source_plan_commit": generation["source_plan_commit"],
        "freeze_binding": generation["freeze_binding"],
        "coordinate_contract": generation["coordinate_contract"],
        "source_audit_bindings": generation["source_audit_bindings"],
        "source_scope_count": generation["source_scope_count"],
        "accepted_clip_count": generation["accepted_clip_count"],
        "rejected_clip_count": generation["rejected_clip_count"],
        "rig_count": generation["rig_count"],
        "selection_sha256": generation["selection_sha256"],
        "visual_gate_sha256": generation["visual_gate_sha256"],
        "anomaly_allowlist_sha256": generation["anomaly_allowlist_sha256"],
        "anomaly_allowlist_entry_set_sha256": generation[
            "anomaly_allowlist_entry_set_sha256"
        ],
        "final_source_recheck_sha256": generation["final_source_recheck_sha256"],
        "prototype_conversion_authorized": generation[
            "prototype_conversion_authorized"
        ],
        "full_conversion_authorized": generation["full_conversion_authorized"],
        "approved_at_utc": approved_at_utc,
    }


def _approval_path(output_root: Path, content_sha256: str) -> Path:
    return output_root / BUILD_APPROVAL_DIRECTORY / f"{content_sha256}.json"


def _validate_generation_approval(
    generation_root: Path,
    *,
    generation: Mapping[str, Any] | None = None,
    evidence: Mapping[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    root = generation_root.expanduser().resolve()
    observed_generation = (
        dict(generation)
        if generation is not None
        else _json_from_bytes(
            _read_regular_bytes(
                root / "generation.json",
                label="published KTJD-17 generation manifest",
                require_read_only=True,
            ),
            label="published KTJD-17 generation manifest",
        )
    )
    observed_evidence = (
        dict(evidence)
        if evidence is not None
        else _generation_content_evidence(root, generation=observed_generation)
    )
    output_root = root.parent.parent
    approval_root = output_root / BUILD_APPROVAL_DIRECTORY
    if (
        approval_root.is_symlink()
        or not approval_root.is_dir()
        or approval_root.resolve(strict=True) != approval_root
    ):
        raise PzHuman312BuildError(f"invalid build approval root: {approval_root}")
    path = _approval_path(
        output_root, str(observed_evidence["generation_content_sha256"])
    )
    payload = _read_regular_bytes(
        path, label="immutable KTJD-17 build approval", require_read_only=True
    )
    approval = _json_from_bytes(payload, label="immutable KTJD-17 build approval")
    expected_keys = set(
        _approval_fields(
            observed_generation,
            observed_evidence,
            approved_at_utc="placeholder",
        )
    )
    approved_at = approval.get("approved_at_utc")
    if (
        set(approval) != expected_keys
        or not isinstance(approved_at, str)
        or not approved_at
        or approval
        != _approval_fields(
            observed_generation,
            observed_evidence,
            approved_at_utc=approved_at,
        )
    ):
        raise PzHuman312BuildError("external build approval/content binding drifted")
    return path, approval


def _create_generation_approval(
    generation_root: Path,
    *,
    generation: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    root = generation_root.expanduser().resolve()
    output_root = root.parent.parent
    approval_root = output_root / BUILD_APPROVAL_DIRECTORY
    approval_root.mkdir(parents=True, exist_ok=True)
    if approval_root.is_symlink() or approval_root.resolve(strict=True) != approval_root:
        raise PzHuman312BuildError(f"invalid build approval root: {approval_root}")
    path = _approval_path(output_root, str(evidence["generation_content_sha256"]))
    if os.path.lexists(path):
        return _validate_generation_approval(
            root, generation=generation, evidence=evidence
        )
    approval = _approval_fields(
        generation,
        evidence,
        approved_at_utc=_datetime.datetime.now(_datetime.UTC).isoformat(),
    )
    payload = (
        json.dumps(
            approval,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _write_bytes_atomic(path, payload)
    observed = path.lstat()
    os.chmod(path, int(observed.st_mode) & ~0o222)
    _fsync_directory(approval_root)
    return _validate_generation_approval(
        root, generation=generation, evidence=evidence
    )


def _verify_generation_content(root: str | Path) -> dict[str, Any]:
    generation_root = Path(root).expanduser().resolve()
    _assert_sanitized_generation_metadata(generation_root)
    generation = _load_json(generation_root / "generation.json")
    if generation.get("generation_id") != generation_root.name:
        raise PzHuman312BuildError("generation id/path mismatch")
    expected = generation.get("files")
    if not isinstance(expected, Mapping):
        raise PzHuman312BuildError("generation file manifest is absent")
    observed = _file_manifest(generation_root)
    if observed != {key: dict(value) for key, value in expected.items()}:
        raise PzHuman312BuildError("generation file hash/size closure failed")
    manifests = list(_iter_jsonl(generation_root / "manifests/clips.jsonl"))
    rejections = list(_iter_jsonl(generation_root / "manifests/rejections.jsonl"))
    qa = list(_iter_jsonl(generation_root / "qa/encoder_qa.jsonl"))
    accepted_ids = [
        _safe_identifier(record["clip_id"], label="manifest clip identifier")
        for record in manifests
    ]
    rejected_ids = [
        _safe_identifier(record["clip_id"], label="rejected clip identifier")
        for record in rejections
    ]
    qa_ids = [
        _safe_identifier(record["clip_id"], label="QA clip identifier")
        for record in qa
    ]
    qa_by_id = {str(record["clip_id"]): record for record in qa}
    if (
        accepted_ids != sorted(accepted_ids)
        or rejected_ids != sorted(rejected_ids)
        or qa_ids != sorted(qa_ids)
        or len(accepted_ids) != len(set(accepted_ids))
        or len(rejected_ids) != len(set(rejected_ids))
        or len(qa_ids) != len(set(qa_ids))
        or set(accepted_ids) & set(rejected_ids)
        or set(qa_ids) != set(accepted_ids) | set(rejected_ids)
        or any(record.get("status") != "accept" for record in manifests)
        or any(qa_by_id[clip_id].get("status") != "pass" for clip_id in accepted_ids)
        or any(qa_by_id[clip_id].get("status") != "reject" for clip_id in rejected_ids)
        or any(
            qa_by_id[str(record["clip_id"])] != record for record in rejections
        )
    ):
        raise PzHuman312BuildError("manifest/QA identity closure failed")
    mode = str(generation.get("mode"))
    if mode not in {"prototype", "full"}:
        raise PzHuman312BuildError(f"invalid generation mode {mode!r}")
    expected_scope = EXPECTED_RIG_COUNT if mode == "prototype" else EXPECTED_CLIP_COUNT
    if len(accepted_ids) + len(rejected_ids) != expected_scope:
        raise PzHuman312BuildError("generation source scope count drifted")
    accepted_rigs = {
        _safe_identifier(record["rig_id"], label="manifest rig identifier")
        for record in manifests
    }
    expected_status = (
        "numeric_pass_pending_312_rig_dynamic_visual_review"
        if mode == "prototype"
        else "full_numeric_pass_visual_gate_bound"
    )
    if (
        int(generation.get("source_scope_count", -1)) != expected_scope
        or int(generation.get("accepted_clip_count", -1)) != len(accepted_ids)
        or int(generation.get("rejected_clip_count", -1)) != len(rejected_ids)
        or int(generation.get("rig_count", -1)) != len(accepted_rigs)
        or len(accepted_rigs) != EXPECTED_RIG_COUNT
        or generation.get("status") != expected_status
        or generation.get("freeze_binding") != _expected_freeze_binding()
        or generation.get("source_plan_commit") != SOURCE_PLAN_COMMIT
        or generation.get("prototype_conversion_authorized")
        is not (mode == "prototype")
        or generation.get("full_conversion_authorized") is not (mode == "full")
        or (mode == "prototype" and (rejected_ids or len(accepted_ids) != EXPECTED_RIG_COUNT))
        or (
            mode == "full"
            and not re.fullmatch(
                r"[0-9a-f]{64}", str(generation.get("visual_gate_sha256", ""))
            )
        )
    ):
        raise PzHuman312BuildError("generation mode/count/authorization drifted")
    if (
        observed.get("schema.json", {}).get("sha256")
        != EXPECTED_FROZEN_SCHEMA_SHA256
        or observed.get("stats/train_block_gains.npz", {}).get("sha256")
        != FROZEN_STATS_SHA256
        or observed.get("evidence/freeze_generation.json", {}).get("sha256")
        != FREEZE_GENERATION_SHA256
    ):
        raise PzHuman312BuildError("frozen encoder payload binding drifted")
    selection_path = generation_root / "manifests/prototype_selection.json"
    selection = _load_json(selection_path)
    authority = selection.get("selection_authority")
    if (
        generation.get("selection_sha256")
        != _sha256_bytes(_canonical_json(selection))
        or not isinstance(authority, Mapping)
        or authority.get("parent_manifest_base") != "dataset_root"
        or Path(str(authority.get("parent_manifest_relpath", ""))).is_absolute()
        or ".." in Path(str(authority.get("parent_manifest_relpath", ""))).parts
        or authority.get("freeze_binding") != _expected_freeze_binding()
    ):
        raise PzHuman312BuildError("selection/freeze/relative-provenance binding drifted")
    visual_evidence = generation_root / "evidence/visual_gate.json"
    if mode == "prototype":
        if generation.get("visual_gate_sha256") is not None or visual_evidence.exists():
            raise PzHuman312BuildError("prototype unexpectedly contains a visual gate")
    elif (
        not visual_evidence.is_file()
        or _sha256_file(visual_evidence) != generation.get("visual_gate_sha256")
    ):
        raise PzHuman312BuildError("full visual-gate evidence binding drifted")
    _verify_published_anomaly_allowlist(generation_root, generation, rejections)
    referenced_motions = {str(record["motion_relpath"]) for record in manifests}
    observed_motions = {
        relpath for relpath in observed if relpath.startswith("motions/")
    }
    if referenced_motions != observed_motions:
        raise PzHuman312BuildError("motion payload reference closure failed")
    referenced_skeletons = {
        str(record["skeleton_relpath"]) for record in manifests
    }
    observed_skeletons = {
        relpath for relpath in observed if relpath.startswith("skeletons/")
    }
    if referenced_skeletons != observed_skeletons:
        raise PzHuman312BuildError("skeleton payload reference closure failed")
    schema = _load_json(generation_root / "schema.json")
    fps_target = float(schema.get("fps_target", float("nan")))
    if not np.isfinite(fps_target) or fps_target != 30.0:
        raise PzHuman312BuildError("frozen FPS target drifted")
    skeleton_cache: dict[str, SkeletonData] = {}
    for record in manifests:
        clip_id = _safe_identifier(record["clip_id"], label="manifest clip identifier")
        rig_id = _safe_identifier(record["rig_id"], label="manifest rig identifier")
        motion_relpath = str(record["motion_relpath"])
        skeleton_relpath = str(record["skeleton_relpath"])
        if (
            motion_relpath != f"motions/{clip_id}.npz"
            or skeleton_relpath != f"skeletons/{rig_id}.npz"
            or _contained_path(
                generation_root, motion_relpath, label=f"{clip_id} motion"
            )
            != generation_root / motion_relpath
            or _contained_path(
                generation_root, skeleton_relpath, label=f"{clip_id} skeleton"
            )
            != generation_root / skeleton_relpath
            or motion_relpath not in observed
            or skeleton_relpath not in observed
            or observed[motion_relpath]["sha256"] != record.get("motion_sha256")
            or observed[skeleton_relpath]["sha256"]
            != record.get("skeleton_sha256")
        ):
            raise PzHuman312BuildError(
                f"manifest payload path/hash binding drifted: {clip_id}"
            )
        if rig_id not in skeleton_cache:
            skeleton_cache[rig_id] = load_skeleton(generation_root / skeleton_relpath)
        skeleton = skeleton_cache[rig_id]
        source_family = str(record.get("source_family"))
        if (
            skeleton.sha256 != str(record["skeleton_sha256"])
            or skeleton.rig_id != rig_id
            or skeleton.source_family != source_family
            or skeleton.topology_family != str(record.get("topology_family"))
            or set(skeleton.rotation_source_kind.astype(str)) != {"animated_dof"}
            or (
                source_family == "planetzoo"
                and skeleton.artifact_status != "planetzoo_stage2_fixed_rig_pass"
            )
            or (
                source_family == "motionstreamer272"
                and skeleton.artifact_status != "t05_prototype_override_pass"
            )
            or source_family not in SOURCE_FAMILIES
        ):
            raise PzHuman312BuildError(f"published skeleton semantics drifted: {rig_id}")
        stored_metrics = _validate_stored_artifact(
            generation_root / motion_relpath,
            clip_id=clip_id,
            skeleton=skeleton,
            expected_sha256=str(record["motion_sha256"]),
            fps_target=fps_target,
        )
        qa_metrics = qa_by_id[clip_id].get("metrics")
        if (
            not isinstance(qa_metrics, Mapping)
            or int(record.get("T_target", -1)) != int(stored_metrics["stored_T"])
            or int(record.get("J_phys", -1)) != int(stored_metrics["stored_J"])
            or float(record.get("fps_target", float("nan"))) != fps_target
            or any(
                key not in qa_metrics
                or not np.isclose(
                    float(qa_metrics[key]),
                    float(value),
                    rtol=0.0,
                    atol=1e-12,
                )
                for key, value in stored_metrics.items()
            )
        ):
            raise PzHuman312BuildError(f"published payload/QA semantics drifted: {clip_id}")
    selection_records = selection.get("selected")
    expected_selection = sorted(
        (
            str(record["clip_id"]),
            str(record["rig_id"]),
            str(record["source_family"]),
        )
        for record in manifests + rejections
    )
    if (
        not isinstance(selection_records, list)
        or sorted(
            (
                str(record.get("clip_id")),
                str(record.get("rig_id")),
                str(record.get("source_family")),
            )
            for record in selection_records
        )
        != expected_selection
        or int(selection.get("selected_count", -1)) != expected_scope
        or int(selection.get("rig_count", -1)) != EXPECTED_RIG_COUNT
    ):
        raise PzHuman312BuildError("selection/manifest semantic closure drifted")
    deep_path = generation_root / "qa/independent_fixed_qa.jsonl"
    if mode == "prototype":
        deep_records = list(_iter_jsonl(deep_path))
        deep_ids = [str(record.get("clip_id")) for record in deep_records]
        if (
            deep_ids != sorted(accepted_ids)
            or len(deep_ids) != EXPECTED_RIG_COUNT
            or len(set(deep_ids)) != EXPECTED_RIG_COUNT
            or any(record.get("status") != "pass" for record in deep_records)
        ):
            raise PzHuman312BuildError("prototype independent fixed-QA closure drifted")
    elif deep_path.exists():
        raise PzHuman312BuildError("full generation contains prototype-only deep QA")
    split_union: set[str] = set()
    for split in SPLITS:
        values = [
            line
            for line in (
                generation_root / f"splits/holdout_splits_v1/{split}.txt"
            ).read_text(encoding="utf-8").splitlines()
            if line
        ]
        expected_values = sorted(
            str(record["clip_id"])
            for record in manifests
            if record["split"] == split
        )
        if values != expected_values or split_union & set(values):
            raise PzHuman312BuildError(f"split membership drifted: {split}")
        split_union.update(values)
    if split_union != set(accepted_ids):
        raise PzHuman312BuildError("split files do not cover accepted clips")
    summary = _load_json(generation_root / "qa/summary.json")
    final_recheck_path = generation_root / "qa/final_source_recheck.json"
    final_recheck = _load_json(final_recheck_path)
    expected_source_bindings = {
        family: {
            "generation_id": record["generation_id"],
            "generation_content_sha256": record["generation_content_sha256"],
            "source_snapshot_sha256": record["source_snapshot_sha256"],
        }
        for family, record in final_recheck.get("source_audits", {}).items()
    }
    if (
        summary.get("mode") != mode
        or summary.get("status") != expected_status
        or int(summary.get("source_scope_count", -1)) != expected_scope
        or int(summary.get("accepted_clip_count", -1)) != len(accepted_ids)
        or int(summary.get("rejected_clip_count", -1)) != len(rejected_ids)
        or int(summary.get("accepted_rig_count", -1)) != len(accepted_rigs)
        or summary.get("final_source_recheck_status") != "pass"
        or summary.get("freeze_binding") != _expected_freeze_binding()
        or summary.get("anomaly_allowlist_sha256")
        != generation.get("anomaly_allowlist_sha256")
        or summary.get("anomaly_allowlist_entry_set_sha256")
        != generation.get("anomaly_allowlist_entry_set_sha256")
        or final_recheck.get("status") != "pass"
        or final_recheck.get("freeze_binding") != _expected_freeze_binding()
        or generation.get("source_audit_bindings") != expected_source_bindings
        or set(expected_source_bindings) != set(SOURCE_FAMILIES)
        or int(final_recheck.get("skeleton_count", -1)) != EXPECTED_RIG_COUNT
        or generation.get("final_source_recheck_sha256")
        != _sha256_file(final_recheck_path)
    ):
        raise PzHuman312BuildError("summary/final-source-recheck binding drifted")
    return generation


def verify_generation(root: str | Path) -> dict[str, Any]:
    """Verify payload semantics and the external immutable build approval."""

    generation_root = Path(root).expanduser().resolve()
    generation = _verify_generation_content(generation_root)
    evidence = _generation_content_evidence(
        generation_root, generation=generation
    )
    _validate_generation_approval(
        generation_root,
        generation=generation,
        evidence=evidence,
    )
    return generation


def run_build(config: BuildConfig) -> dict[str, Any]:
    cfg = config.resolved()
    scope = _load_scope(cfg)
    encoder = encoder_config_from_frozen_schema(cfg.freeze_root)
    visual_gate = (
        None
        if cfg.mode == "prototype"
        else _validate_visual_gate(  # type: ignore[arg-type]
            cfg.visual_gate_path, scope, dataset_root=cfg.dataset_root
        )
    )
    anomaly_allowlist = _load_anomaly_allowlist(
        cfg.anomaly_allowlist_path, scope
    )
    if cfg.mode == "prototype":
        selected_ids = _representative_ids(scope)
        generation_directory = PROTOTYPE_GENERATION_DIRECTORY
        link_name = PROTOTYPE_LINK_NAME
        approval_link_name = PROTOTYPE_APPROVAL_LINK_NAME
        audit_role = "per_rig_dynamic_prototype"
        selection_origin = "approved_source_audit_max_dynamic_per_rig"
    else:
        selected_ids = sorted(scope["audit_records"])
        generation_directory = FULL_GENERATION_DIRECTORY
        link_name = FULL_LINK_NAME
        approval_link_name = FULL_APPROVAL_LINK_NAME
        audit_role = "full_approved_source_conversion"
        selection_origin = "paired_content_addressed_source_audit_approval"
    generation_id = (
        _datetime.datetime.now(_datetime.UTC).strftime("%Y%m%dT%H%M%S%fZ")
        + "-"
        + uuid.uuid4().hex[:12]
    )
    generations = _validate_output_namespace(
        cfg, scope, generation_directory=generation_directory
    )
    generations.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{generation_id}.", dir=generations))
    final = generations / generation_id
    try:
        logical_roots = {
            "planetzoo_stage2": Path(scope["source_roots"]["planetzoo"]),
            "human_motionstreamer272": Path(
                scope["source_roots"]["motionstreamer272"]
            ),
            "dataset": cfg.dataset_root,
            "frozen_encoder": cfg.freeze_root,
        }
        skeleton_hashes: dict[str, str] = {}
        for rig_id, source_text in sorted(scope["skeleton_paths"].items()):
            source = Path(source_text)
            expected = str(scope["skeleton_hashes"][rig_id])
            target = _contained_path(
                staging,
                f"skeletons/{_safe_identifier(rig_id, label='rig identifier')}.npz",
                label="skeleton output",
            )
            skeleton_hashes[rig_id] = _copy_sanitized_skeleton(
                source,
                target,
                expected_sha256=expected,
                logical_roots=logical_roots,
            )
        _copy_regular_file(
            cfg.freeze_root / "schema.json",
            staging / "schema.json",
            expected_sha256=EXPECTED_FROZEN_SCHEMA_SHA256,
        )
        _copy_regular_file(
            cfg.freeze_root / "stats/train_block_gains.npz",
            staging / "stats/train_block_gains.npz",
            expected_sha256=FROZEN_STATS_SHA256,
        )
        _copy_regular_file(
            cfg.freeze_root / "generation.json",
            staging / "evidence/freeze_generation.json",
            expected_sha256=FREEZE_GENERATION_SHA256,
        )
        if visual_gate is not None:
            observed_gate_sha = _write_bytes_atomic(
                staging / "evidence/visual_gate.json",
                visual_gate["payload"],
            )
            if observed_gate_sha != visual_gate["sha256"]:
                raise PzHuman312BuildError("captured visual gate bytes drifted")
        if anomaly_allowlist["payload"] is not None:
            observed_allowlist_sha = _write_bytes_atomic(
                staging / "evidence/anomaly_allowlist.json",
                anomaly_allowlist["payload"],
            )
            if observed_allowlist_sha != anomaly_allowlist["sha256"]:
                raise PzHuman312BuildError("captured anomaly allowlist bytes drifted")
        encoder_record = _encoder_record(encoder)
        _write_json(staging / "config/encoder_candidate.json", encoder_record)
        selection_authority = {
            "parent_manifest_base": "dataset_root",
            "parent_manifest_relpath": _parent_manifest_relpath(cfg, scope),
            "parent_clips_jsonl_sha256": scope["parent_hashes"]["clips.jsonl"],
            "parent_rigs_jsonl_sha256": scope["parent_hashes"]["rigs.jsonl"],
            "planetzoo_source_audit_generation_id": scope["pz_active"][
                "generation_id"
            ],
            "planetzoo_source_audit_content_sha256": scope["pz_active"][
                "generation_content_sha256"
            ],
            "human_source_audit_generation_id": scope["human_active"][
                "generation_id"
            ],
            "human_source_audit_content_sha256": scope["human_active"][
                "generation_content_sha256"
            ],
            "legacy_btjd13_motion_used": False,
            "source_plan_commit": SOURCE_PLAN_COMMIT,
            "freeze_binding": scope["freeze_binding"],
        }
        selection = {
            "selection_version": BUILD_VERSION,
            "mode": cfg.mode,
            "selection_authority": selection_authority,
            "selected": [
                {
                    "clip_id": clip_id,
                    "rig_id": scope["audit_records"][clip_id]["rig_id"],
                    "source_family": scope["audit_records"][clip_id][
                        "source_family"
                    ],
                }
                for clip_id in selected_ids
            ],
            "selected_count": len(selected_ids),
            "rig_count": len(
                {
                    str(scope["audit_records"][clip_id]["rig_id"])
                    for clip_id in selected_ids
                }
            ),
            "full_conversion_authorized": bool(visual_gate),
        }
        _write_json(staging / "manifests/prototype_selection.json", selection)
        tasks = [
            {
                "clip": scope["parent_clips"][clip_id],
                "audit": scope["audit_records"][clip_id],
                "skeleton_sha256": skeleton_hashes[
                    str(scope["audit_records"][clip_id]["rig_id"])
                ],
                "audit_role": audit_role,
                "selection_origin": selection_origin,
            }
            for clip_id in selected_ids
        ]
        manifests, qa_all = _run_conversion_tasks(
            tasks,
            rigs=scope["parent_rigs"],
            skeleton_paths={
                rig_id: str(staging / "skeletons" / f"{rig_id}.npz")
                for rig_id in skeleton_hashes
            },
            encoder=encoder,
            output_root=staging,
            workers=cfg.workers,
            allowed_rejections=anomaly_allowlist["entries"],
        )
        pass_qa = [record for record in qa_all if record.get("status") == "pass"]
        rejections = [
            record for record in qa_all if record.get("status") == "reject"
        ]
        if cfg.mode == "prototype" and rejections:
            raise PzHuman312BuildError(
                f"prototype rejected {len(rejections)} of {EXPECTED_RIG_COUNT} clips"
            )
        if cfg.mode == "full":
            accepted_rigs = {record["rig_id"] for record in manifests}
            actual_rejected = {str(record["clip_id"]) for record in rejections}
            expected_rejected = set(anomaly_allowlist["entries"])
            if (
                actual_rejected != expected_rejected
                or len(accepted_rigs) != EXPECTED_RIG_COUNT
            ):
                raise PzHuman312BuildError(
                    "full rejection set is not the exact reviewed allowlist: "
                    f"actual={len(actual_rejected)} expected={len(expected_rejected)}, "
                    f"accepted_rigs={len(accepted_rigs)}/{EXPECTED_RIG_COUNT}"
                )
        deep_records: list[dict[str, Any]] = []
        if cfg.mode == "prototype":
            deep_records = _run_prototype_deep_validation(
                root=staging,
                manifests=manifests,
                qa_records=pass_qa,
                scope=scope,
                encoder_config=encoder_record,
                workers=cfg.workers,
            )
            _write_jsonl(staging / "qa/independent_fixed_qa.jsonl", deep_records)
        final_source_recheck = _final_scope_recheck(cfg, scope)
        _write_json(
            staging / "qa/final_source_recheck.json", final_source_recheck
        )
        _write_jsonl(staging / "manifests/clips.jsonl", manifests)
        _write_jsonl(staging / "manifests/rejections.jsonl", rejections)
        _write_jsonl(staging / "qa/encoder_qa.jsonl", qa_all)
        for split in SPLITS:
            _write_lines(
                staging / f"splits/holdout_splits_v1/{split}.txt",
                sorted(
                    record["clip_id"]
                    for record in manifests
                    if record["split"] == split
                ),
            )
        summary = {
            "build_version": BUILD_VERSION,
            "mode": cfg.mode,
            "status": (
                "numeric_pass_pending_312_rig_dynamic_visual_review"
                if cfg.mode == "prototype"
                else "full_numeric_pass_visual_gate_bound"
            ),
            "source_scope_count": len(selected_ids),
            "accepted_clip_count": len(manifests),
            "rejected_clip_count": len(rejections),
            "accepted_rig_count": len({record["rig_id"] for record in manifests}),
            "source_family_counts": dict(
                sorted(Counter(record["source_family"] for record in manifests).items())
            ),
            "split_counts": dict(
                sorted(Counter(record["split"] for record in manifests).items())
            ),
            "rejection_reason_counts": dict(
                sorted(
                    Counter(
                        reason
                        for record in rejections
                        for reason in record["reason_codes"]
                    ).items()
                )
            ),
            "encoder_metrics": _metric_summary(pass_qa),
            "independent_fixed_qa_metrics": _metric_summary(deep_records),
            "final_source_recheck_status": final_source_recheck["status"],
            "freeze_binding": scope["freeze_binding"],
            "anomaly_allowlist_sha256": anomaly_allowlist["sha256"],
            "anomaly_allowlist_entry_set_sha256": anomaly_allowlist[
                "entry_set_sha256"
            ],
            "coordinate_contract": COORDINATE_CONTRACT,
            "prototype_conversion_authorized": cfg.mode == "prototype",
            "full_conversion_authorized": cfg.mode == "full",
        }
        _write_json(staging / "qa/summary.json", summary)
        _assert_sanitized_generation_metadata(staging)
        files = _file_manifest(staging)
        generation = {
            "build_version": BUILD_VERSION,
            "generation_id": generation_id,
            "created_at_utc": _datetime.datetime.now(_datetime.UTC).isoformat(),
            "mode": cfg.mode,
            "status": summary["status"],
            "source_plan_commit": SOURCE_PLAN_COMMIT,
            "freeze_binding": scope["freeze_binding"],
            "coordinate_contract": COORDINATE_CONTRACT,
            "source_scope_count": len(selected_ids),
            "accepted_clip_count": len(manifests),
            "rejected_clip_count": len(rejections),
            "rig_count": len({record["rig_id"] for record in manifests}),
            "selection_sha256": hashlib.sha256(
                _canonical_json(selection)
            ).hexdigest(),
            "source_audit_bindings": {
                "planetzoo": {
                    "generation_id": scope["pz_active"]["generation_id"],
                    "generation_content_sha256": scope["pz_active"][
                        "generation_content_sha256"
                    ],
                    "source_snapshot_sha256": scope["pz_active"][
                        "source_snapshot_sha256"
                    ],
                },
                "motionstreamer272": {
                    "generation_id": scope["human_active"]["generation_id"],
                    "generation_content_sha256": scope["human_active"][
                        "generation_content_sha256"
                    ],
                    "source_snapshot_sha256": scope["human_active"][
                        "source_snapshot_sha256"
                    ],
                },
            },
            "visual_gate_sha256": (
                None if visual_gate is None else visual_gate["sha256"]
            ),
            "anomaly_allowlist_sha256": anomaly_allowlist["sha256"],
            "anomaly_allowlist_entry_set_sha256": anomaly_allowlist[
                "entry_set_sha256"
            ],
            "final_source_recheck_sha256": _sha256_file(
                staging / "qa/final_source_recheck.json"
            ),
            "prototype_conversion_authorized": cfg.mode == "prototype",
            "full_conversion_authorized": cfg.mode == "full",
            "files": files,
        }
        _write_json(staging / "generation.json", generation)
        _fsync_tree(staging)
        if final.exists():
            raise PzHuman312BuildError(f"generation already exists: {final}")
        os.replace(staging, final)
        _fsync_directory(generations)
        _freeze_tree(final)
        candidate = _verify_generation_content(final)
        evidence = _generation_content_evidence(final, generation=candidate)
        approval_path, approval = _create_generation_approval(
            final,
            generation=candidate,
            evidence=evidence,
        )
        verified = verify_generation(final)
        _require_read_only_tree(final, label="published KTJD-17 generation")
        if cfg.update_link:
            _replace_symlink(
                cfg.output_root / approval_link_name,
                approval_path,
            )
            _replace_symlink(cfg.output_root / link_name, final)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {
        "status": verified["status"],
        "mode": cfg.mode,
        "generation_id": generation_id,
        "generation_root": str(final),
        "generation_content_sha256": approval["generation_content_sha256"],
        "approval_path": str(approval_path),
        "approval_link": str(cfg.output_root / approval_link_name),
        "compatibility_link": str(cfg.output_root / link_name),
        "compatibility_link_updated": bool(cfg.update_link),
        "accepted_clip_count": int(verified["accepted_clip_count"]),
        "rejected_clip_count": int(verified["rejected_clip_count"]),
        "rig_count": int(verified["rig_count"]),
        "coordinate_contract": COORDINATE_CONTRACT,
        "prototype_conversion_authorized": cfg.mode == "prototype",
        "full_conversion_authorized": cfg.mode == "full",
    }
