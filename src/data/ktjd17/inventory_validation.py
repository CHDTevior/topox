"""Independent live-data gate for materialized KTJD T02 inventories.

This module deliberately does not import inventory-producer constants, parsers,
or hashing helpers.  It reopens every current NPY and every unique raw source,
recomputes the live snapshot, and then checks cross-artifact invariants.
"""

from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_VERSION = "ktjd17-raw-inventory-v1"
EXPECTED_FAMILIES = (
    "human",
    "quadruped",
    "winged",
    "snake",
    "spider_crab",
    "dragon_or_deep_topology",
)
BASE_MANIFEST_FILES = (
    "clips.jsonl",
    "rigs.jsonl",
    "inventory_summary.json",
    "inventory_reason_codes.json",
    "prototype_candidates.json",
    "prototype_gaps.jsonl",
)
SOURCE_FK_MANIFEST_FILES = (
    "source_fk_qa.jsonl",
    "source_fk_summary.json",
    "source_fk_generation.json",
)
CANONICAL_SKELETON_MANIFEST_FILES = (
    "canonical_skeleton_qa.jsonl",
    "canonical_skeleton_summary.json",
    "canonical_skeleton_generation.json",
)
TRANSACTION_FILENAME = "inventory_generation.json"
EXPECTED_REASON_SEVERITIES = {
    "BTJD_SHAPE_INVALID": "reject",
    "SOURCE_FILE_MISSING": "reject",
    "SOURCE_HEADER_INVALID": "reject",
    "SOURCE_LAYOUT_DRIFT": "reject",
    "SOURCE_NONRETAINED_LAYOUT_VARIANT": "review",
    "SOURCE_FRAME_MAPPING_MISMATCH": "reject",
    "RAW_SOURCE_SEQUENCE_SPLIT_OVERLAP": "reject",
    "JOINT_MAP_MISSING": "reject",
    "JOINT_MAP_AMBIGUOUS": "reject",
    "CURRENT_PARENT_NOT_SOURCE_ANCESTOR": "reject",
    "ROTATION_PROVENANCE_INVALID": "reject",
    "NUMERIC_PAYLOAD_VALIDATION_DEFERRED_T03": "review",
    "SOURCE_NUMERIC_PARSE_INVALID": "reject",
    "SOURCE_FK_REPRODUCTION_FAILED": "reject",
    "CANONICAL_TRANSFORM_PROVENANCE_INVALID": "reject",
    "CANONICAL_SKELETON_DERIVATION_FAILED": "reject",
    "HUMAN_FIXED_REST_UNRESOLVED": "reject",
    "HEADING_PAYLOAD_UNREVIEWED": "review",
    "SOURCE_TO_CANONICAL_UNREVIEWED": "review",
    "PZ_RAW_GAME_BVH_NOT_LOCAL": "review",
    "PZ_SOURCE_HAS_PER_CLIP_CANONICALIZATION": "review",
    "HUMAN_CURRENT_BRIDGE_USES_PER_CLIP_ALIGNMENT": "review",
    "REST_SOURCE_REQUIRES_RAW_TPOSE_RECOVERY": "review",
    "REST_FRAME_FALLBACK_NOT_EXPLICIT": "review",
    "SOURCE_ROOT_WRAPPER_DROPPED": "review",
    "JOINT_MAP_SKIPS_SOURCE_JOINTS": "review",
    "SOURCE_UNIT_TO_METER_UNKNOWN": "info",
    "PROTOTYPE_TRAIN_SHORTAGE": "gap",
    "EXACT_DRAGON_NOT_TRAIN_ELIGIBLE": "gap",
}


class InventoryValidationError(RuntimeError):
    """A materialized inventory disagrees with its contract or live sources."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise InventoryValidationError(f"cannot load JSON {path}: {exc}") from exc


def _iter_jsonl(path: Path):
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise InventoryValidationError(
                        f"{path}:{line_number}: blank JSONL record"
                    )
                try:
                    yield json.loads(line)
                except Exception as exc:  # noqa: BLE001
                    raise InventoryValidationError(
                        f"{path}:{line_number}: invalid JSON: {exc}"
                    ) from exc
    except OSError as exc:
        raise InventoryValidationError(f"cannot read JSONL {path}: {exc}") from exc


def _load_split_map(root: Path) -> tuple[dict[str, str], dict[str, int]]:
    mapping: dict[str, str] = {}
    counts: dict[str, int] = {}
    for split in ("train", "val", "held_representative", "held_stress"):
        path = root / f"{split}.txt"
        if not path.is_file():
            raise InventoryValidationError(f"missing frozen split file: {path}")
        count = 0
        for line_number, raw in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            name = raw.strip()
            if not name or name.startswith("#"):
                continue
            if Path(name).name != name or not name.endswith(".npy"):
                raise InventoryValidationError(
                    f"{path}:{line_number}: invalid clip basename {name!r}"
                )
            if name in mapping:
                raise InventoryValidationError(
                    f"split overlap for {name}: {mapping[name]} and {split}"
                )
            mapping[name] = split
            count += 1
        counts[split] = count
    return mapping, counts


def _status_from_codes(codes: list[str]) -> str:
    severities = {EXPECTED_REASON_SEVERITIES[code] for code in codes}
    if "reject" in severities:
        return "reject"
    if "review" in severities:
        return "review"
    return "accept"


def _read_npy_header_live(path: Path) -> dict[str, Any]:
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:  # noqa: BLE001
        raise InventoryValidationError(f"cannot independently open NPY {path}: {exc}") from exc
    stat = path.stat()
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "file_size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


_FRAMES_RE = re.compile(r"^Frames\s*:\s*([0-9]+)\s*$", re.IGNORECASE)
_FRAME_TIME_RE = re.compile(
    r"^Frame\s+Time\s*:\s*([^\s]+)\s*$", re.IGNORECASE
)


def _read_bvh_header_live(path: Path) -> dict[str, Any]:
    frames: int | None = None
    frame_time: float | None = None
    joint_count = 0
    channel_count = 0
    try:
        with path.open("r", encoding="utf-8", errors="strict") as handle:
            for line_number, raw in enumerate(handle, start=1):
                text = raw.strip()
                if text.startswith("ROOT ") or text.startswith("JOINT "):
                    joint_count += 1
                elif text.startswith("End Site"):
                    joint_count += 1
                elif text.startswith("CHANNELS "):
                    fields = text.split()
                    if len(fields) < 2:
                        raise InventoryValidationError(
                            f"{path}:{line_number}: malformed CHANNELS line"
                        )
                    declared = int(fields[1])
                    if len(fields) != declared + 2:
                        raise InventoryValidationError(
                            f"{path}:{line_number}: CHANNELS declaration mismatch"
                        )
                    channel_count += declared
                else:
                    frame_match = _FRAMES_RE.match(text)
                    if frame_match:
                        frames = int(frame_match.group(1))
                        continue
                    time_match = _FRAME_TIME_RE.match(text)
                    if time_match:
                        frame_time = float(time_match.group(1))
                        break
    except InventoryValidationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise InventoryValidationError(
            f"cannot independently parse BVH header {path}: {exc}"
        ) from exc
    if frames is None or frames <= 0:
        raise InventoryValidationError(f"{path}: missing/invalid Frames header")
    if frame_time is None or not math.isfinite(frame_time) or frame_time <= 0:
        raise InventoryValidationError(f"{path}: missing/invalid Frame Time header")
    if joint_count <= 0 or channel_count <= 0:
        raise InventoryValidationError(f"{path}: missing hierarchy/channel evidence")
    stat = path.stat()
    return {
        "T_src": frames,
        "frame_time_src": frame_time,
        "fps_src": 1.0 / frame_time,
        "source_joint_count": joint_count,
        "source_channel_count": channel_count,
        "file_size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise InventoryValidationError(f"{label} mismatch: {actual!r} != {expected!r}")


def _validate_transaction(root: Path) -> dict[str, Any]:
    transaction = _load_json(root / TRANSACTION_FILENAME)
    _require_equal(
        "transaction manifest_version",
        transaction.get("manifest_version"),
        EXPECTED_VERSION,
    )
    _require_equal(
        "transaction publish_protocol",
        transaction.get("publish_protocol"),
        "immutable_generation_atomic_symlink_replace",
    )
    files = transaction.get("files")
    if not isinstance(files, dict):
        raise InventoryValidationError("transaction files must be an object")
    file_set = set(files)
    allowed_sets = {
        frozenset(BASE_MANIFEST_FILES),
        frozenset((*BASE_MANIFEST_FILES, *SOURCE_FK_MANIFEST_FILES)),
        frozenset(
            (
                *BASE_MANIFEST_FILES,
                *SOURCE_FK_MANIFEST_FILES,
                *CANONICAL_SKELETON_MANIFEST_FILES,
            )
        ),
    }
    if frozenset(file_set) not in allowed_sets:
        raise InventoryValidationError("transaction file set mismatch")
    for name in sorted(file_set):
        path = root / name
        if not path.is_file():
            raise InventoryValidationError(f"transaction artifact missing: {path}")
        record = files[name]
        _require_equal(f"transaction {name} size", path.stat().st_size, record.get("size_bytes"))
        _require_equal(f"transaction {name} sha256", _sha256_file(path), record.get("sha256"))
    return transaction


def validate_inventory_outputs(
    manifest_root: str | Path,
    *,
    dataset_root: str | Path,
    split_root: str | Path,
) -> dict[str, Any]:
    """Re-read live headers and fail on provenance, split, or snapshot drift."""
    requested_root = Path(manifest_root).expanduser()
    root = requested_root.resolve()  # snapshot the immutable generation once
    current_root = Path(dataset_root).expanduser().resolve()
    frozen_splits = Path(split_root).expanduser().resolve()
    missing = [
        name
        for name in (*BASE_MANIFEST_FILES, TRANSACTION_FILENAME)
        if not (root / name).is_file()
    ]
    if missing:
        raise InventoryValidationError(f"inventory artifacts missing: {missing}")
    transaction = _validate_transaction(root)
    has_source_fk = set(SOURCE_FK_MANIFEST_FILES) <= set(transaction["files"])
    has_canonical_skeleton = set(CANONICAL_SKELETON_MANIFEST_FILES) <= set(
        transaction["files"]
    )

    summary = _load_json(root / "inventory_summary.json")
    reason_table = _load_json(root / "inventory_reason_codes.json")
    candidates = _load_json(root / "prototype_candidates.json")
    for label, payload in (
        ("summary", summary),
        ("reason table", reason_table),
        ("prototype candidates", candidates),
    ):
        _require_equal(
            f"{label} manifest_version", payload.get("manifest_version"), EXPECTED_VERSION
        )
    materialized_codes = reason_table.get("codes")
    source_fk_codes = {
        "SOURCE_NUMERIC_PARSE_INVALID",
        "SOURCE_FK_REPRODUCTION_FAILED",
    }
    canonical_skeleton_codes = {
        "CANONICAL_TRANSFORM_PROVENANCE_INVALID",
        "CANONICAL_SKELETON_DERIVATION_FAILED",
        "HUMAN_FIXED_REST_UNRESOLVED",
    }
    all_codes = set(EXPECTED_REASON_SEVERITIES)
    pre_t04_codes = all_codes - canonical_skeleton_codes
    legacy_t02_codes = all_codes - source_fk_codes - canonical_skeleton_codes
    allowed_reason_sets = (
        {frozenset(all_codes), frozenset(pre_t04_codes)}
        if has_source_fk
        else {
            frozenset(all_codes),
            frozenset(all_codes - source_fk_codes),
            frozenset(pre_t04_codes),
            frozenset(legacy_t02_codes),
        }
    )
    if not isinstance(materialized_codes, dict) or frozenset(materialized_codes) not in allowed_reason_sets:
        raise InventoryValidationError("materialized reason-code set differs from contract")
    for code, payload in materialized_codes.items():
        severity = EXPECTED_REASON_SEVERITIES[code]
        if payload.get("severity") != severity or not payload.get("description"):
            raise InventoryValidationError(f"reason-code contract mismatch: {code}")

    config = summary.get("config", {})
    _require_equal(
        "summary config.dataset_root", Path(config.get("dataset_root", "")).resolve(), current_root
    )
    _require_equal(
        "summary config.split_root", Path(config.get("split_root", "")).resolve(), frozen_splits
    )
    cond_path = current_root / "cond.npy"
    _require_equal("cond_sha256", summary.get("cond_sha256"), _sha256_file(cond_path))
    split_manifest = frozen_splits / "splits_manifest.json"
    expected_split_hash = _sha256_file(split_manifest) if split_manifest.is_file() else None
    _require_equal(
        "split_manifest_sha256", summary.get("split_manifest_sha256"), expected_split_hash
    )

    rigs: dict[str, dict[str, Any]] = {}
    rig_status = Counter()
    max_joints = 0
    parent_hashes: set[str] = set()
    for record in _iter_jsonl(root / "rigs.jsonl"):
        _require_equal(
            f"rig {record.get('rig_id')} version",
            record.get("manifest_version"),
            EXPECTED_VERSION,
        )
        rig_id = record.get("rig_id")
        if not isinstance(rig_id, str) or not rig_id or rig_id in rigs:
            raise InventoryValidationError(f"invalid or duplicate rig id {rig_id!r}")
        hash_payload = dict(record)
        stored_hash = hash_payload.pop("rig_evidence_sha256", None)
        _require_equal(f"rig {rig_id} evidence hash", stored_hash, _sha256_json(hash_payload))
        codes = record.get("reason_codes")
        if not isinstance(codes, list) or any(
            code not in EXPECTED_REASON_SEVERITIES for code in codes
        ):
            raise InventoryValidationError(f"rig {rig_id} has unknown reason codes")
        _require_equal(f"rig {rig_id} status", record.get("status"), _status_from_codes(codes))
        joint_map = record.get("joint_map", {})
        names = joint_map.get("btjd_joint_names")
        parents = joint_map.get("btjd_parents")
        kinds = joint_map.get("rotation_source_kind")
        if (
            not isinstance(names, list)
            or not names
            or not isinstance(parents, list)
            or len(parents) != len(names)
            or not isinstance(kinds, list)
            or any(kind not in {"animated_dof", "fixed_dof"} for kind in kinds)
        ):
            raise InventoryValidationError(f"rig {rig_id} malformed joint provenance")
        joint_count = len(names)
        missing_count = joint_map.get("missing_or_unknown_count")
        _require_equal(
            f"rig {rig_id} animated count",
            joint_map.get("animated_dof_count"),
            kinds.count("animated_dof"),
        )
        _require_equal(
            f"rig {rig_id} fixed count",
            joint_map.get("fixed_dof_count"),
            kinds.count("fixed_dof"),
        )
        _require_equal(
            f"rig {rig_id} missing count", missing_count, joint_count - len(kinds)
        )
        provenance = record.get("rotation_provenance_status")
        if provenance == "proven":
            if joint_map.get("status") != "binary_proven" or len(kinds) != joint_count:
                raise InventoryValidationError(
                    f"rig {rig_id} proven provenance is not binary-complete"
                )
        elif provenance == "invalid":
            if joint_map.get("status") != "invalid" or "reject" not in {
                EXPECTED_REASON_SEVERITIES[code] for code in codes
            }:
                raise InventoryValidationError(
                    f"rig {rig_id} invalid provenance lacks fail-closed reject evidence"
                )
        else:
            raise InventoryValidationError(
                f"rig {rig_id} invalid rotation provenance status {provenance!r}"
            )
        max_joints = max(max_joints, joint_count)
        parent_hashes.add(record["topology_parent_sha256"])
        rig_status[record["status"]] += 1
        rigs[rig_id] = record

    split_map, disk_split_counts = _load_split_map(frozen_splits)
    disk_paths = sorted((current_root / "motions").glob("*.npy"))
    disk_ids = {path.stem for path in disk_paths}
    manifest_ids: set[str] = set()
    status_counts = Counter()
    rotation_counts = Counter()
    source_counts = Counter()
    split_counts = Counter()
    prototype_flags: set[str] = set()
    reject_reasons = Counter()
    live_source_cache: dict[str, dict[str, Any] | None] = {}
    source_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    snapshot_rows: list[list[Any]] = []
    clip_index: dict[str, dict[str, Any]] = {}

    for index, record in enumerate(_iter_jsonl(root / "clips.jsonl"), start=1):
        clip_id = record.get("clip_id")
        _require_equal(
            f"clip {clip_id} version", record.get("manifest_version"), EXPECTED_VERSION
        )
        if not isinstance(clip_id, str) or not clip_id or clip_id in manifest_ids:
            raise InventoryValidationError(f"invalid or duplicate clip id {clip_id!r}")
        manifest_ids.add(clip_id)
        rig_id = record.get("rig_id")
        if rig_id not in rigs:
            raise InventoryValidationError(f"clip {clip_id} references unknown rig {rig_id!r}")
        _require_equal(
            f"clip {clip_id} rig hash",
            record.get("rotation_provenance", {}).get("rig_evidence_sha256"),
            rigs[rig_id]["rig_evidence_sha256"],
        )
        codes = record.get("reason_codes")
        if not isinstance(codes, list) or any(
            code not in EXPECTED_REASON_SEVERITIES for code in codes
        ):
            raise InventoryValidationError(f"clip {clip_id} has unknown reason codes")
        status = record.get("status")
        _require_equal(f"clip {clip_id} status", status, _status_from_codes(codes))
        severities = {EXPECTED_REASON_SEVERITIES[code] for code in codes}

        rig_joint_count = len(rigs[rig_id]["joint_map"]["btjd_joint_names"])
        rotation = record.get("rotation_provenance", {})
        rotation_status = rotation.get("status")
        if rotation_status == "proven":
            if rotation.get("missing_ik_legacy_unknown_count") != 0:
                raise InventoryValidationError(
                    f"clip {clip_id} proven rotation has missing sources"
                )
            if (
                rotation.get("animated_dof_count", 0)
                + rotation.get("fixed_dof_count", 0)
                != rig_joint_count
            ):
                raise InventoryValidationError(
                    f"clip {clip_id} proven rotation counts do not match rig J"
                )
        elif rotation_status == "invalid":
            if status != "reject" or "reject" not in severities:
                raise InventoryValidationError(
                    f"clip {clip_id} invalid rotations lack a reject record"
                )
        else:
            raise InventoryValidationError(
                f"clip {clip_id} invalid rotation status {rotation_status!r}"
            )
        if status == "accept" and rotation_status != "proven":
            raise InventoryValidationError(f"accepted clip {clip_id} lacks proven rotations")

        btjd = record.get("btjd", {})
        btjd_path = Path(btjd.get("path", ""))
        if not btjd_path.is_file():
            raise InventoryValidationError(f"clip {clip_id} current BTJD path missing")
        live_btjd = _read_npy_header_live(btjd_path)
        _require_equal(f"clip {clip_id} BTJD shape", btjd.get("shape"), live_btjd["shape"])
        _require_equal(f"clip {clip_id} BTJD dtype", btjd.get("dtype"), live_btjd["dtype"])
        _require_equal(
            f"clip {clip_id} BTJD size",
            btjd.get("file_size_bytes"),
            live_btjd["file_size_bytes"],
        )
        _require_equal(
            f"clip {clip_id} BTJD mtime", btjd.get("mtime_ns"), live_btjd["mtime_ns"]
        )
        shape = live_btjd["shape"]
        if len(shape) != 3 or shape[0] <= 0 or shape[1] != rig_joint_count or shape[2] != 13:
            raise InventoryValidationError(f"clip {clip_id} invalid live BTJD shape {shape}")

        source = record.get("source", {})
        source_path_text = source.get("path")
        if not isinstance(source_path_text, str) or not source_path_text:
            raise InventoryValidationError(f"clip {clip_id} has no source path evidence")
        source_path = Path(source_path_text)
        source_exists = source_path.is_file()
        source_live: dict[str, Any] | None = None
        if source_exists:
            if source_path_text not in live_source_cache:
                try:
                    if source.get("family") == "motionstreamer272":
                        source_live = _read_npy_header_live(source_path)
                        source_live["T_src"] = source_live["shape"][0]
                    else:
                        source_live = _read_bvh_header_live(source_path)
                except InventoryValidationError:
                    if "SOURCE_HEADER_INVALID" not in codes:
                        raise
                    source_live = None
                live_source_cache[source_path_text] = source_live
            source_live = live_source_cache[source_path_text]
        elif "SOURCE_FILE_MISSING" not in codes:
            raise InventoryValidationError(f"clip {clip_id} source path missing")

        if source_live is not None:
            for key in ("file_size_bytes", "mtime_ns", "T_src"):
                _require_equal(
                    f"clip {clip_id} source {key}", source.get(key), source_live.get(key)
                )
            if source.get("family") == "motionstreamer272":
                _require_equal(
                    f"clip {clip_id} source shape", source.get("shape"), source_live["shape"]
                )
                _require_equal(
                    f"clip {clip_id} source dtype", source.get("dtype"), source_live["dtype"]
                )
                _require_equal(f"clip {clip_id} source fps", source.get("fps_src"), 30.0)
            else:
                for key in (
                    "source_joint_count",
                    "source_channel_count",
                    "frame_time_src",
                ):
                    _require_equal(
                        f"clip {clip_id} source {key}",
                        source.get(key),
                        source_live.get(key),
                    )
                if not math.isclose(
                    float(source.get("fps_src")),
                    float(source_live["fps_src"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise InventoryValidationError(f"clip {clip_id} source FPS drift")
            source_groups[source_path_text].append(record)

        split = record.get("split")
        _require_equal(
            f"clip {clip_id} split", split, split_map.get(f"{clip_id}.npy")
        )
        canonical_reject_codes = {
            "CANONICAL_TRANSFORM_PROVENANCE_INVALID",
            "CANONICAL_SKELETON_DERIVATION_FAILED",
            "HUMAN_FIXED_REST_UNRESOLVED",
        }
        source_stage_has_reject = any(
            EXPECTED_REASON_SEVERITIES[code] == "reject"
            for code in codes
            if code not in canonical_reject_codes
        )
        source_train_eligible = (
            split == "train"
            and rotation_status == "proven"
            and not source_stage_has_reject
        )
        expected_train_eligible = (
            split == "train" and rotation_status == "proven" and "reject" not in severities
        )
        _require_equal(
            f"clip {clip_id} train eligibility",
            record.get("split_eligible_for_train_calibration"),
            expected_train_eligible,
        )
        if record.get("prototype_candidate"):
            if not source_train_eligible:
                raise InventoryValidationError(
                    f"prototype candidate {clip_id} was not safe at its frozen source/provenance stage"
                )
            prototype_flags.add(clip_id)

        source_stat = source_path.stat() if source_exists else None
        snapshot_rows.append(
            [
                clip_id,
                live_btjd["file_size_bytes"],
                live_btjd["mtime_ns"],
                source_path_text,
                source_stat.st_size if source_stat else None,
                source_stat.st_mtime_ns if source_stat else None,
            ]
        )
        status_counts[status] += 1
        rotation_counts[rotation_status] += 1
        source_counts[source.get("family")] += 1
        split_counts[split] += 1
        if status == "reject":
            for code in codes:
                if EXPECTED_REASON_SEVERITIES[code] == "reject":
                    reject_reasons[code] += 1
        clip_index[clip_id] = {
            "family": record.get("topology_family"),
            "rig_id": rig_id,
            "split": split,
            # Frozen T02/T03 prototype selection remains source-stage evidence.
            # T04 eligibility is recorded separately and must not rewrite history.
            "eligible": source_train_eligible,
            "prototype": bool(record.get("prototype_candidate")),
        }
        if index % 10000 == 0:
            print(f"[inventory-validation] live headers: {index}/{len(disk_ids)}", flush=True)

    if manifest_ids != disk_ids:
        raise InventoryValidationError(
            f"clip manifest/disk mismatch: manifest_only={len(manifest_ids-disk_ids)}, "
            f"disk_only={len(disk_ids-manifest_ids)}"
        )
    snapshot_rows.sort(key=lambda row: row[0])
    _require_equal(
        "live_snapshot_sha256", summary.get("live_snapshot_sha256"), _sha256_json(snapshot_rows)
    )

    overlap_records: list[dict[str, Any]] = []
    for source_path, records in sorted(source_groups.items()):
        splits = sorted({record["split"] for record in records})
        is_safe = len(splits) <= 1
        clip_ids = sorted(record["clip_id"] for record in records)
        for record in records:
            source = record["source"]
            _require_equal(
                f"clip {record['clip_id']} source sequence_splits",
                source.get("sequence_splits"),
                splits,
            )
            _require_equal(
                f"clip {record['clip_id']} source sequence_split_safe",
                source.get("sequence_split_safe"),
                is_safe,
            )
            has_code = "RAW_SOURCE_SEQUENCE_SPLIT_OVERLAP" in record["reason_codes"]
            if has_code != (not is_safe):
                raise InventoryValidationError(
                    f"clip {record['clip_id']} raw-sequence split reason mismatch"
                )
            if not is_safe and (
                record["status"] != "reject"
                or record["split_eligible_for_train_calibration"]
                or record.get("prototype_candidate")
            ):
                raise InventoryValidationError(
                    f"clip {record['clip_id']} cross-split source was not excluded"
                )
        if not is_safe:
            overlap_records.append(
                {
                    "source_path": source_path,
                    "splits": splits,
                    "clip_count": len(records),
                    "clip_ids": clip_ids,
                }
            )
    summary_split_audit = summary.get("raw_source_split_audit", {})
    _require_equal(
        "raw split cross-source count",
        summary_split_audit.get("cross_split_source_count"),
        len(overlap_records),
    )
    _require_equal(
        "raw split affected clip count",
        summary_split_audit.get("affected_clip_count"),
        sum(record["clip_count"] for record in overlap_records),
    )
    _require_equal(
        "raw split source groups", summary_split_audit.get("source_groups"), overlap_records
    )

    gaps = list(_iter_jsonl(root / "prototype_gaps.jsonl"))
    shortage_families = {
        gap.get("family")
        for gap in gaps
        if "PROTOTYPE_TRAIN_SHORTAGE" in gap.get("reason_codes", [])
    }
    family_payload = candidates.get("families", {})
    selected_ids: set[str] = set()
    for family in EXPECTED_FAMILIES:
        payload = family_payload.get(family)
        if not isinstance(payload, dict):
            raise InventoryValidationError(f"prototype family record missing: {family}")
        family_clips = [item for item in clip_index.values() if item["family"] == family]
        eligible_ids = {
            clip_id
            for clip_id, item in clip_index.items()
            if item["family"] == family and item["eligible"]
        }
        selected = payload.get("selected_train_candidates")
        required_count = payload.get("required_train_clips")
        if not isinstance(selected, list) or not isinstance(required_count, int):
            raise InventoryValidationError(f"prototype family record malformed: {family}")
        if not set(selected) <= eligible_ids:
            raise InventoryValidationError(
                f"prototype family {family} selected unsafe/noneligible clips"
            )
        _require_equal(
            f"prototype family {family} eligible count",
            payload.get("rotation_proven_train_candidates"),
            len(eligible_ids),
        )
        _require_equal(
            f"prototype family {family} all count",
            payload.get("all_current_clip_count"),
            len(family_clips),
        )
        _require_equal(
            f"prototype family {family} selected count",
            payload.get("selected_count"),
            len(selected),
        )
        if len(selected) >= required_count:
            _require_equal(f"prototype family {family} status", payload.get("status"), "available")
        elif family not in shortage_families or payload.get("status") != "shortage":
            raise InventoryValidationError(
                f"prototype family {family} shortage lacks gap record"
            )
        selected_ids.update(selected)
    _require_equal("prototype flag/selection set", selected_ids, prototype_flags)

    fresh = summary.get("fresh_counts", {})
    expected_counts = {
        "current_btjd_clips": len(manifest_ids),
        "current_rigs": len(rigs),
        "current_unique_parent_trees": len(parent_hashes),
        "current_max_physical_joints": max_joints,
        "split_counts": dict(disk_split_counts),
        "source_clip_counts": dict(sorted(source_counts.items())),
        "clip_status_counts": dict(sorted(status_counts.items())),
        "rig_status_counts": dict(sorted(rig_status.items())),
        "rotation_provenance_counts": dict(sorted(rotation_counts.items())),
    }
    for key, expected in expected_counts.items():
        _require_equal(f"summary fresh_counts.{key}", fresh.get(key), expected)
    _require_equal("summary prototype families", summary.get("prototype_families"), family_payload)
    _require_equal("summary prototype gaps", summary.get("prototype_gap_records"), gaps)

    return {
        "manifest_version": EXPECTED_VERSION,
        "validated_at_utc": _datetime.datetime.now(
            _datetime.timezone.utc
        ).isoformat().replace("+00:00", "Z"),
        "status": "pass",
        "generation_id": transaction.get("generation_id"),
        "validation_mode": "independent_live_npy_bvh_header_rescan",
        "validated_counts": expected_counts,
        "live_current_npy_headers_validated": len(manifest_ids),
        "live_unique_source_headers_validated": sum(
            source is not None for source in live_source_cache.values()
        ),
        "live_snapshot_sha256_verified": True,
        "raw_source_cross_split_count": len(overlap_records),
        "raw_source_cross_split_affected_clips": sum(
            record["clip_count"] for record in overlap_records
        ),
        "prototype_candidate_count": len(prototype_flags),
        "prototype_shortage_families": sorted(shortage_families),
        "reject_reason_counts": dict(sorted(reject_reasons.items())),
        "meter_claim": False,
        "source_fk_artifacts_present": has_source_fk,
        "canonical_skeleton_artifacts_present": has_canonical_skeleton,
    }


def write_validation_report(
    report: dict[str, Any],
    path: str | Path,
    *,
    immutable_manifest_root: str | Path,
) -> None:
    target = Path(path).expanduser().resolve()
    immutable_root = Path(immutable_manifest_root).expanduser().resolve()
    if target == immutable_root or immutable_root in target.parents:
        raise InventoryValidationError(
            "validation reports are post-publication evidence and cannot be written "
            f"inside immutable manifest generation {immutable_root}: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                report,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    finally:
        temp = Path(temp_name)
        if temp.exists():
            temp.unlink()
