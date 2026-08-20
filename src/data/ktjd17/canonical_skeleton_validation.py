"""Independent live validation for KTJD-17 T04 skeleton artifacts.

The validator deliberately does not import the T04 producer or its constants.
Truebones rest BVHs are decoded through the independent SciPy-based T03
validator, then the canonical transform and every rest array are recomputed.
The mutable ``dataset/skeletons`` compatibility link is never used to locate
an artifact; the active manifest's immutable relative path and SHA-256 are the
authority.
"""

from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .source_fk_validation import (
    _load_conditioning_independent,
    _load_neutral_rest,
    _parse_bvh_independent,
    _truebones_fixed_live,
)


EXPECTED_INVENTORY_VERSION = "ktjd17-raw-inventory-v1"
EXPECTED_QA_VERSION = "ktjd17-canonical-skeleton-v2"
EXPECTED_SKELETON_DIRECTORY = ".ktjd17_skeleton_generations"
EXPECTED_MANIFEST_DIRECTORY = ".ktjd17_manifest_generations"
EXPECTED_MANIFEST_FILES = {
    "clips.jsonl",
    "rigs.jsonl",
    "inventory_summary.json",
    "inventory_reason_codes.json",
    "prototype_candidates.json",
    "prototype_gaps.jsonl",
    "source_fk_qa.jsonl",
    "source_fk_summary.json",
    "source_fk_generation.json",
    "canonical_skeleton_qa.jsonl",
    "canonical_skeleton_summary.json",
    "canonical_skeleton_generation.json",
}
EXPECTED_T03_MANIFEST_FILES = EXPECTED_MANIFEST_FILES - {
    "canonical_skeleton_qa.jsonl",
    "canonical_skeleton_summary.json",
    "canonical_skeleton_generation.json",
}
EXPECTED_SOURCE_FK_EVIDENCE_FILES = {
    "source_fk_qa.jsonl",
    "source_fk_summary.json",
    "source_fk_generation.json",
}
EXPECTED_REQUIRED_KEYS = {
    "joint_names",
    "parents",
    "P_rest_global",
    "R_rest_global",
    "R_rest_local",
    "offset_parent_local",
    "rotation_source_kind",
    "heading_carrier_joint",
    "u_forward_local",
    "heading_payload_provenance",
    "source_to_canonical_C",
    "source_to_canonical_alpha",
    "source_to_canonical_o",
    "s_rig",
    "length_unit_id",
    "source_unit_to_meter",
    "canonical_scale_factor",
    "joint_map_metadata",
}
EXPECTED_EXTRA_KEYS = {
    "rig_id",
    "source_family",
    "topology_family",
    "artifact_status",
    "reason_codes",
    "skeleton_format_version",
    "representative_clip_id",
    "source_rest_path",
    "source_rest_sha256",
    "source_to_canonical_provenance",
    "position_geometry_provenance",
    "conditioning_authority",
    "conditioning_payload_sha256",
    "fixed_rig_rotation_signatures",
    "unit_metadata",
}
EXPECTED_SOURCE_COUNTS = Counter(
    {"truebones": 31, "planetzoo": 26, "motionstreamer272": 1}
)
EXPECTED_GATE_COUNTS = Counter({"pass": 27, "review": 4, "reject": 27})
EXPECTED_ARTIFACT_SOURCE_COUNTS = Counter(
    {"truebones": 31, "motionstreamer272": 1}
)
EXPECTED_TOPOLOGIES = {
    "human",
    "quadruped",
    "winged",
    "snake",
    "spider_crab",
    "dragon_or_deep_topology",
}
EXPECTED_REVIEW_RIGS = {"Anaconda", "Bird", "Lion", "Pteranodon"}
TRUEBONES_MEAN_EDGE_TARGET = 0.2092142857142857

# Independent frozen review truth.  This intentionally duplicates rather than
# imports the producer table so a self-consistent but anatomically wrong
# producer change fails validation.
EXPECTED_TRUEBONES_FORWARD_SPECS: dict[
    str, tuple[str, tuple[str, ...], str]
] = {
    "Alligator": ("lateral_pairs", ("R_momo", "L_momo", "R_hiji", "L_hiji"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Anaconda": ("root_to_head", ("Hips", "BN_Tone_04"), "coiled_snake_root_to_tongue_endpoint_reviewed_t04"),
    "Bat": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_R_UpperArm_01", "BN_L_UpperArm_01"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Bird": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_Forearm_R_01", "BN_Forearm_L_01"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Buffalo": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_Clavicle", "Bip01_L_Clavicle"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Buzzard": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_Wing_R_02", "BN_Wing_L_02"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Cat": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_Clavicle", "Bip01_L_Clavicle"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Chicken": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_Finger_R_01", "BN_Finger_L_01"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Coyote": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_Clavicle", "Bip01_L_Clavicle"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Crocodile": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Dragon": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Eagle": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_Wing_R_02", "BN_Wing_L_02"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Flamingo": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_Forearm_R_02", "BN_Forearm_L_02"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Fox": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Gazelle": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Hamster": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "HermitCrab": ("lateral_pairs", ("BN_Leg_R_09", "BN_Leg_L_09", "BN_Crab_pincers_R_02", "BN_Crab_pincers_L_02"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Hippopotamus": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_Clavicle", "Bip01_L_Clavicle"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Horse": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Hound": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "KingCobra": ("root_to_head", ("Hips", "BN_Tongue_02"), "snake_root_to_tongue_endpoint_reviewed_t04"),
    "Lion": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Lynx": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Mammoth": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Ostrich": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_Forearm_R_02", "BN_Forearm_L_02"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Parrot": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_Wing_R_02", "BN_Wing_L_02"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Parrot2": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_Wing_R_02", "BN_Wing_L_02"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Pteranodon": ("lateral_pairs", ("jt_Thigh_R", "jt_Thigh_L", "jt_Elbow_R", "jt_Elbow_L"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Scorpion": ("lateral_pairs", ("Bip01_R_Thigh_4", "Bip01_L_Thigh1_4", "Bip01_R_Forearm", "Bip01_L_Forearm"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "SpiderG": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Tukan": ("lateral_pairs", ("R_momo", "L_momo", "R_kata", "L_kata"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
}

BASIS_TOL = 1e-12
FLOAT64_ROTATION_TOL = 1e-10
FLOAT64_LOCAL_TOL = 1e-12
FLOAT64_POSITION_NORM_TOL = 1e-12
FLOAT32_LOCAL_TOL = 2e-6
FLOAT32_POSITION_NORM_TOL = 1e-5
FLOAT32_IDENTITY_TOL = 2e-6


class CanonicalSkeletonValidationError(RuntimeError):
    """Materialized T04 evidence disagrees with independent reconstruction."""


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise CanonicalSkeletonValidationError(
            f"cannot read JSON {path}: {exc}"
        ) from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise CanonicalSkeletonValidationError(
                        f"{path}:{line_number}: blank JSONL row"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise CanonicalSkeletonValidationError(
                        f"{path}:{line_number}: row is not an object"
                    )
                records.append(value)
    except CanonicalSkeletonValidationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise CanonicalSkeletonValidationError(
            f"cannot read JSONL {path}: {exc}"
        ) from exc
    return records


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise CanonicalSkeletonValidationError(
            f"{label} mismatch: {actual!r} != {expected!r}"
        )


def _require_close(
    label: str,
    actual: np.ndarray | float,
    expected: np.ndarray | float,
    *,
    atol: float,
) -> float:
    lhs = np.asarray(actual, dtype=np.float64)
    rhs = np.asarray(expected, dtype=np.float64)
    if lhs.shape != rhs.shape:
        raise CanonicalSkeletonValidationError(
            f"{label} shape mismatch: {lhs.shape} != {rhs.shape}"
        )
    if not np.isfinite(lhs).all() or not np.isfinite(rhs).all():
        raise CanonicalSkeletonValidationError(f"{label} contains non-finite values")
    error = float(np.max(np.abs(lhs - rhs))) if lhs.size else 0.0
    if error > atol:
        raise CanonicalSkeletonValidationError(
            f"{label} max error {error:.17g} exceeds {atol:.17g}"
        )
    return error


def _scalar_text(payload: Mapping[str, np.ndarray], key: str) -> str:
    value = payload[key]
    if value.shape != () or value.dtype.kind not in "US":
        raise CanonicalSkeletonValidationError(
            f"{key} must be a scalar Unicode/string array, got {value.shape}/{value.dtype}"
        )
    return str(value.item())


def _scalar_float(payload: Mapping[str, np.ndarray], key: str) -> float:
    value = payload[key]
    if value.shape != () or value.dtype != np.float64:
        raise CanonicalSkeletonValidationError(
            f"{key} must be a float64 scalar, got {value.shape}/{value.dtype}"
        )
    result = float(value.item())
    if not math.isfinite(result):
        raise CanonicalSkeletonValidationError(f"{key} is non-finite")
    return result


def _parse_json_scalar(payload: Mapping[str, np.ndarray], key: str) -> Any:
    try:
        return json.loads(_scalar_text(payload, key))
    except json.JSONDecodeError as exc:
        raise CanonicalSkeletonValidationError(
            f"{key} is not canonical JSON: {exc}"
        ) from exc


def _validate_manifest_transaction(root: Path) -> dict[str, Any]:
    transaction = _load_json(root / "inventory_generation.json")
    _require_equal(
        "manifest version",
        transaction.get("manifest_version"),
        EXPECTED_INVENTORY_VERSION,
    )
    _require_equal(
        "manifest publish protocol",
        transaction.get("publish_protocol"),
        "immutable_generation_atomic_symlink_replace",
    )
    files = transaction.get("files")
    if not isinstance(files, dict) or set(files) != EXPECTED_MANIFEST_FILES:
        raise CanonicalSkeletonValidationError(
            f"unexpected T04 manifest file set: {sorted(files) if isinstance(files, dict) else files}"
        )
    for name in sorted(EXPECTED_MANIFEST_FILES):
        path = root / name
        if not path.is_file():
            raise CanonicalSkeletonValidationError(f"manifest file missing: {path}")
        _require_equal(
            f"manifest size {name}", path.stat().st_size, files[name]["size_bytes"]
        )
        _require_equal(
            f"manifest SHA-256 {name}", _sha256_file(path), files[name]["sha256"]
        )
    return transaction


def _resolve_skeleton_generation(
    manifest_root: Path, stage_record: Mapping[str, Any]
) -> tuple[Path, Path]:
    if manifest_root.parent.name != ".ktjd17_manifest_generations":
        raise CanonicalSkeletonValidationError(
            f"manifest must resolve to an immutable generation, got {manifest_root}"
        )
    dataset_root = manifest_root.parent.parent.resolve()
    relative_text = stage_record.get("skeleton_generation_relpath")
    generation_id = stage_record.get("skeleton_generation_id")
    if not isinstance(relative_text, str) or not isinstance(generation_id, str):
        raise CanonicalSkeletonValidationError("T04 stage record lacks skeleton generation")
    relative = Path(relative_text)
    expected = Path(EXPECTED_SKELETON_DIRECTORY) / generation_id
    _require_equal("skeleton generation relative path", relative, expected)
    generation = (dataset_root / relative).resolve()
    generation_root = (dataset_root / EXPECTED_SKELETON_DIRECTORY).resolve()
    if generation.parent != generation_root or not generation.is_dir():
        raise CanonicalSkeletonValidationError(
            f"invalid immutable skeleton generation: {generation}"
        )
    return dataset_root, generation


def _validate_skeleton_transaction(
    generation: Path,
    stage_record: Mapping[str, Any],
) -> dict[str, Any]:
    transaction_path = generation / "skeleton_generation.json"
    _require_equal(
        "skeleton transaction SHA-256",
        _sha256_file(transaction_path),
        stage_record.get("skeleton_generation_transaction_sha256"),
    )
    transaction = _load_json(transaction_path)
    _require_equal("skeleton QA version", transaction.get("qa_version"), EXPECTED_QA_VERSION)
    _require_equal(
        "skeleton generation id",
        transaction.get("generation_id"),
        stage_record.get("skeleton_generation_id"),
    )
    _require_equal(
        "skeleton parent manifest",
        transaction.get("parent_manifest_generation_id"),
        stage_record.get("parent_manifest_generation_id"),
    )
    _require_equal(
        "skeleton source-FK manifest",
        transaction.get("source_fk_manifest_generation_id"),
        stage_record.get("source_fk_manifest_generation_id"),
    )
    _require_equal(
        "skeleton source-FK transaction SHA-256",
        transaction.get("source_fk_manifest_transaction_sha256"),
        stage_record.get("source_fk_manifest_transaction_sha256"),
    )
    _require_equal(
        "skeleton source-FK file evidence",
        transaction.get("source_fk_manifest_files"),
        stage_record.get("source_fk_manifest_files"),
    )
    _require_equal(
        "skeleton publish protocol",
        transaction.get("publish_protocol"),
        "immutable_skeleton_generation_then_authoritative_manifest_reference",
    )
    _require_equal(
        "skeleton symlink role",
        transaction.get("public_symlink_role"),
        "compatibility_only_non_authoritative",
    )
    files = transaction.get("files")
    if not isinstance(files, dict):
        raise CanonicalSkeletonValidationError("skeleton transaction files are absent")
    actual_files = {
        str(path.relative_to(generation))
        for path in generation.rglob("*")
        if path.is_file() and path.name != "skeleton_generation.json"
    }
    _require_equal("skeleton transaction file set", set(files), actual_files)
    for name in sorted(files):
        path = generation / name
        _require_equal(
            f"skeleton size {name}", path.stat().st_size, files[name]["size_bytes"]
        )
        _require_equal(
            f"skeleton SHA-256 {name}", _sha256_file(path), files[name]["sha256"]
        )
    return transaction


def _validate_direct_t03_lineage(
    dataset_root: Path,
    stage_record: Mapping[str, Any],
) -> Path:
    source_generation_id = stage_record.get("source_fk_manifest_generation_id")
    _require_equal(
        "direct T03 parent generation",
        stage_record.get("parent_manifest_generation_id"),
        source_generation_id,
    )
    if not isinstance(source_generation_id, str) or not source_generation_id:
        raise CanonicalSkeletonValidationError("source-FK manifest generation id is absent")
    source_root = (
        dataset_root / EXPECTED_MANIFEST_DIRECTORY / source_generation_id
    ).resolve()
    expected_parent = (dataset_root / EXPECTED_MANIFEST_DIRECTORY).resolve()
    if source_root.parent != expected_parent or not source_root.is_dir():
        raise CanonicalSkeletonValidationError(
            f"direct immutable T03 generation is missing: {source_root}"
        )
    transaction_path = source_root / "inventory_generation.json"
    _require_equal(
        "direct T03 transaction SHA-256",
        _sha256_file(transaction_path),
        stage_record.get("source_fk_manifest_transaction_sha256"),
    )
    transaction = _load_json(transaction_path)
    _require_equal(
        "direct T03 generation id", transaction.get("generation_id"), source_generation_id
    )
    _require_equal(
        "direct T03 manifest version",
        transaction.get("manifest_version"),
        EXPECTED_INVENTORY_VERSION,
    )
    _require_equal(
        "direct T03 publish protocol",
        transaction.get("publish_protocol"),
        "immutable_generation_atomic_symlink_replace",
    )
    files = transaction.get("files")
    if not isinstance(files, dict) or set(files) != EXPECTED_T03_MANIFEST_FILES:
        raise CanonicalSkeletonValidationError(
            "recorded source-FK parent is not an exact T03 manifest generation"
        )
    _require_equal(
        "direct T03 parent file evidence",
        stage_record.get("parent_manifest_files"),
        files,
    )
    _require_equal(
        "direct T03 source-FK file evidence",
        stage_record.get("source_fk_manifest_files"),
        {name: files[name] for name in EXPECTED_SOURCE_FK_EVIDENCE_FILES},
    )
    for name in sorted(files):
        path = source_root / name
        if not path.is_file():
            raise CanonicalSkeletonValidationError(
                f"direct T03 artifact is absent: {path}"
            )
        _require_equal(
            f"direct T03 size {name}", path.stat().st_size, files[name]["size_bytes"]
        )
        _require_equal(
            f"direct T03 SHA-256 {name}", _sha256_file(path), files[name]["sha256"]
        )
    return source_root


def _forward_from_provenance(
    names: Sequence[str],
    positions: np.ndarray,
    provenance: Mapping[str, Any],
) -> np.ndarray:
    anchors = provenance.get("forward_anchor_names")
    method = provenance.get("forward_method")
    if not isinstance(anchors, list) or any(not isinstance(item, str) for item in anchors):
        raise CanonicalSkeletonValidationError("heading forward anchors are malformed")
    lookup: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        lookup.setdefault(str(name), []).append(index)
    indices: list[int] = []
    for name in anchors:
        hits = lookup.get(name, [])
        if len(hits) != 1:
            raise CanonicalSkeletonValidationError(
                f"forward anchor {name!r} resolves to {hits}"
            )
        indices.append(hits[0])
    if method == "lateral_pairs" and len(indices) == 4:
        across = (
            positions[indices[0]]
            - positions[indices[1]]
            + positions[indices[2]]
            - positions[indices[3]]
        )
        forward = np.cross(np.asarray([0.0, 1.0, 0.0]), across)
    elif method == "root_to_head" and len(indices) == 2:
        forward = positions[indices[1]] - positions[indices[0]]
    elif method == "declared_plus_z" and not indices:
        forward = np.asarray([0.0, 0.0, 1.0])
    else:
        raise CanonicalSkeletonValidationError(
            f"unsupported forward provenance: method={method!r}, anchors={anchors!r}"
        )
    horizontal = np.asarray([forward[0], 0.0, forward[2]], dtype=np.float64)
    norm = float(np.linalg.norm(horizontal))
    if not math.isfinite(norm) or norm <= 0.0:
        raise CanonicalSkeletonValidationError("forward provenance is degenerate")
    return horizontal / norm


def _require_expected_forward_provenance(
    rig_id: str,
    source_family: str,
    names: Sequence[str],
    provenance: Mapping[str, Any],
) -> None:
    if source_family == "truebones":
        try:
            expected_method, expected_anchors, expected_source = (
                EXPECTED_TRUEBONES_FORWARD_SPECS[rig_id]
            )
        except KeyError as exc:
            raise CanonicalSkeletonValidationError(
                f"{rig_id}: no validator-owned reviewed forward specification"
            ) from exc
    elif source_family == "motionstreamer272":
        expected_method = "declared_plus_z"
        expected_anchors = ()
        expected_source = "motionstreamer_humanml_declared_plus_z_candidate_t04"
    else:
        raise CanonicalSkeletonValidationError(
            f"{rig_id}: unsupported artifact source family {source_family!r}"
        )
    _require_equal(
        f"{rig_id} frozen forward method",
        provenance.get("forward_method"),
        expected_method,
    )
    _require_equal(
        f"{rig_id} frozen forward anchors",
        provenance.get("forward_anchor_names"),
        list(expected_anchors),
    )
    _require_equal(
        f"{rig_id} frozen forward provenance",
        provenance.get("forward_spec_provenance"),
        expected_source,
    )
    lookup = {str(name): index for index, name in enumerate(names)}
    if len(lookup) != len(names):
        raise CanonicalSkeletonValidationError(f"{rig_id}: duplicate joint names")
    try:
        expected_indices = [lookup[name] for name in expected_anchors]
    except KeyError as exc:
        raise CanonicalSkeletonValidationError(
            f"{rig_id}: frozen forward anchor is missing: {exc.args[0]}"
        ) from exc
    _require_equal(
        f"{rig_id} frozen forward anchor indices",
        provenance.get("forward_anchor_indices"),
        expected_indices,
    )


def _basis_from_forward(forward: np.ndarray) -> np.ndarray:
    fx, fz = float(forward[0]), float(forward[2])
    norm = math.hypot(fx, fz)
    fx, fz = fx / norm, fz / norm
    return np.asarray(
        [[fz, 0.0, -fx], [0.0, 1.0, 0.0], [fx, 0.0, fz]],
        dtype=np.float64,
    )


def _independent_source_rest(
    payload: Mapping[str, np.ndarray],
    rig: Mapping[str, Any],
    heading_provenance: Mapping[str, Any],
    cond_entries: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    names = tuple(str(item) for item in payload["joint_names"].tolist())
    parents = np.asarray(payload["parents"], dtype=np.int64)
    family = _scalar_text(payload, "source_family")
    rest_path = _scalar_text(payload, "source_rest_path")
    if family == "truebones":
        rest = _parse_bvh_independent(rest_path)
        mapping = np.asarray(rig["joint_map"]["btjd_to_source"], dtype=np.int64)
        if mapping.shape != parents.shape or np.any(mapping < 0) or np.any(mapping >= len(rest.names)):
            raise CanonicalSkeletonValidationError(f"{rig['rig_id']}: invalid source rest map")
        source_names = tuple(rig["joint_map"]["source_joint_names"])
        for source_index in mapping:
            _require_equal(
                f"{rig['rig_id']} source hierarchy name {source_index}",
                rest.names[int(source_index)],
                source_names[int(source_index)],
            )
        source_positions = rest.global_positions[0, mapping].copy()
        source_rotations = rest.global_rotations[0, mapping].copy()
        forward = _forward_from_provenance(names, source_positions, heading_provenance)
        C = _basis_from_forward(forward)
        edges = np.asarray(
            [
                np.linalg.norm(
                    source_positions[child] - source_positions[int(parents[child])]
                )
                for child in range(1, len(parents))
            ],
            dtype=np.float64,
        )
        alpha = TRUEBONES_MEAN_EDGE_TARGET / float(np.mean(edges))
        try:
            cond_entry = cond_entries[str(rig["rig_id"])]
        except KeyError as exc:
            raise CanonicalSkeletonValidationError(
                f"{rig['rig_id']}: conditioning geometry is absent"
            ) from exc
        cond_names = tuple(str(value) for value in cond_entry["joints_names"])
        cond_parents = np.asarray(cond_entry["parents"], dtype=np.int64)
        _require_equal(f"{rig['rig_id']} cond joint names", cond_names, names)
        if not np.array_equal(cond_parents, parents):
            raise CanonicalSkeletonValidationError(
                f"{rig['rig_id']}: cond parent tree differs from artifact"
            )
        expected_positions = np.asarray(
            cond_entry["tpos_first_frame"], dtype=np.float64
        )[:, :3].copy()
        expected_positions[:, 1] -= float(np.min(expected_positions[:, 1]))
        origin = source_positions[0] - (C.T @ expected_positions[0]) / alpha
    elif family == "motionstreamer272":
        source_positions = _load_neutral_rest(rest_path, parents)
        source_rotations = np.broadcast_to(
            np.eye(3, dtype=np.float64), (len(parents), 3, 3)
        ).copy()
        forward = _forward_from_provenance(names, source_positions, heading_provenance)
        C = np.eye(3, dtype=np.float64)
        alpha = 1.0
        rotated = source_positions @ C.T
        canonical_anchor = np.asarray(
            [rotated[0, 0], np.min(rotated[:, 1]), rotated[0, 2]], dtype=np.float64
        )
        origin = C.T @ canonical_anchor
        expected_positions = alpha * (source_positions - origin) @ C.T
    else:
        raise CanonicalSkeletonValidationError(
            f"artifact source family cannot be {family!r}"
        )
    return source_positions, source_rotations, C, alpha, origin, expected_positions


def _rest_fk(
    parents: np.ndarray,
    root_position: np.ndarray,
    rotations: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    result = np.empty((len(parents), 3), dtype=np.result_type(root_position, rotations, offsets))
    result[0] = root_position
    for child in range(1, len(parents)):
        parent = int(parents[child])
        result[child] = result[parent] + rotations[parent] @ offsets[child]
    return result


def _column_d6(rotations: np.ndarray) -> np.ndarray:
    return np.concatenate((rotations[..., :, 0], rotations[..., :, 1]), axis=-1)


def _validate_numeric_rest(
    rig_id: str,
    payload: Mapping[str, np.ndarray],
) -> dict[str, float]:
    parents = np.asarray(payload["parents"], dtype=np.int64)
    positions = np.asarray(payload["P_rest_global"])
    rotations = np.asarray(payload["R_rest_global"])
    local = np.asarray(payload["R_rest_local"])
    offsets = np.asarray(payload["offset_parent_local"])
    joint_count = len(parents)
    if parents.dtype != np.int64 or parents.shape != (joint_count,) or parents[0] != -1:
        raise CanonicalSkeletonValidationError(f"{rig_id}: invalid int64 parent tree")
    for child in range(1, joint_count):
        if not 0 <= int(parents[child]) < child:
            raise CanonicalSkeletonValidationError(
                f"{rig_id}: parent-before-child fails at {child}"
            )
    expected_shapes = {
        "P_rest_global": (joint_count, 3),
        "R_rest_global": (joint_count, 3, 3),
        "R_rest_local": (joint_count, 3, 3),
        "offset_parent_local": (joint_count, 3),
    }
    for name, shape in expected_shapes.items():
        value = payload[name]
        if value.shape != shape or value.dtype != np.float64 or not np.isfinite(value).all():
            raise CanonicalSkeletonValidationError(
                f"{rig_id}: {name} must be finite float64 {shape}, got {value.shape}/{value.dtype}"
            )
    C = np.asarray(payload["source_to_canonical_C"])
    if C.shape != (3, 3) or C.dtype != np.float64 or not np.isfinite(C).all():
        raise CanonicalSkeletonValidationError(f"{rig_id}: invalid source basis")
    basis_orth = float(np.max(np.abs(C.T @ C - np.eye(3))))
    basis_det_error = abs(abs(float(np.linalg.det(C))) - 1.0)
    if basis_orth > BASIS_TOL or basis_det_error > BASIS_TOL:
        raise CanonicalSkeletonValidationError(
            f"{rig_id}: basis gate failed: orth={basis_orth}, det_error={basis_det_error}"
        )
    rotation_orth = float(
        np.max(np.abs(np.swapaxes(rotations, -1, -2) @ rotations - np.eye(3)))
    )
    rotation_det_min = float(np.min(np.linalg.det(rotations)))
    if rotation_orth > FLOAT64_ROTATION_TOL or rotation_det_min <= 0.0:
        raise CanonicalSkeletonValidationError(
            f"{rig_id}: canonical rotations are not SO(3)"
        )
    local_error64 = float(np.max(np.abs(local[0] - rotations[0])))
    for child in range(1, joint_count):
        parent = int(parents[child])
        local_error64 = max(
            local_error64,
            float(np.max(np.abs(rotations[parent] @ local[child] - rotations[child]))),
        )
    if local_error64 > FLOAT64_LOCAL_TOL:
        raise CanonicalSkeletonValidationError(f"{rig_id}: float64 local rotation gate failed")
    if float(np.max(np.abs(offsets[0]))) != 0.0:
        raise CanonicalSkeletonValidationError(f"{rig_id}: root rest offset must be exactly zero")
    s_rig = _scalar_float(payload, "s_rig")
    expected_scale = float(np.linalg.norm(np.ptp(positions, axis=0)))
    _require_close(f"{rig_id} s_rig", s_rig, expected_scale, atol=1e-12)
    fk64 = _rest_fk(parents, positions[0], rotations, offsets)
    position_error64 = float(np.max(np.linalg.norm(fk64 - positions, axis=-1)))
    if position_error64 / s_rig > FLOAT64_POSITION_NORM_TOL:
        raise CanonicalSkeletonValidationError(f"{rig_id}: float64 rest position FK failed")
    delta64 = rotations @ np.swapaxes(rotations, -1, -2)
    identity_d6 = _column_d6(np.eye(3, dtype=np.float64))
    identity_error64 = float(np.max(np.abs(_column_d6(delta64) - identity_d6)))
    if identity_error64 > FLOAT64_ROTATION_TOL:
        raise CanonicalSkeletonValidationError(f"{rig_id}: float64 rest identity failed")

    rotations32 = rotations.astype(np.float32)
    local32 = local.astype(np.float32)
    offsets32 = offsets.astype(np.float32)
    positions32 = positions.astype(np.float32)
    local_error32 = float(np.max(np.abs(local32[0] - rotations32[0])))
    for child in range(1, joint_count):
        parent = int(parents[child])
        local_error32 = max(
            local_error32,
            float(
                np.max(
                    np.abs(rotations32[parent] @ local32[child] - rotations32[child])
                )
            ),
        )
    if local_error32 > FLOAT32_LOCAL_TOL:
        raise CanonicalSkeletonValidationError(f"{rig_id}: float32 local rotation gate failed")
    fk32 = _rest_fk(parents, positions32[0], rotations32, offsets32)
    position_error32 = float(
        np.max(np.linalg.norm(fk32.astype(np.float64) - positions32, axis=-1))
    )
    if position_error32 / s_rig > FLOAT32_POSITION_NORM_TOL:
        raise CanonicalSkeletonValidationError(f"{rig_id}: float32 rest position FK failed")
    delta32 = rotations32 @ np.swapaxes(rotations32, -1, -2)
    identity_error32 = float(
        np.max(np.abs(_column_d6(delta32).astype(np.float64) - identity_d6))
    )
    if identity_error32 > FLOAT32_IDENTITY_TOL:
        raise CanonicalSkeletonValidationError(f"{rig_id}: float32 rest identity failed")
    return {
        "basis_orthogonality_max_abs": basis_orth,
        "basis_determinant_abs_error": basis_det_error,
        "rest_local_float64_max_abs": local_error64,
        "rest_local_float32_max_abs": local_error32,
        "rest_position_fk_float64_max_norm": position_error64 / s_rig,
        "rest_position_fk_float32_max_norm": position_error32 / s_rig,
        "rest_identity_d6_float64_max_abs": identity_error64,
        "rest_identity_d6_float32_max_abs": identity_error32,
    }


def _load_npz_pickle_free(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            payload = {name: archive[name] for name in archive.files}
    except Exception as exc:  # noqa: BLE001
        raise CanonicalSkeletonValidationError(
            f"cannot load pickle-free skeleton {path}: {exc}"
        ) from exc
    if any(value.dtype.hasobject for value in payload.values()):
        raise CanonicalSkeletonValidationError(f"object dtype in skeleton {path}")
    expected = EXPECTED_REQUIRED_KEYS | EXPECTED_EXTRA_KEYS
    _require_equal(f"{path.name} artifact keys", set(payload), expected)
    return payload


def _canonical_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": record["gate_status"],
        "qa_version": record["qa_version"],
        "artifact_relpath": record["artifact_relpath"],
        "artifact_sha256": record["artifact_sha256"],
        "artifact_size_bytes": record["artifact_size_bytes"],
        "representative_clip_id": record["representative_clip_id"],
        "reason_codes": record["reason_codes"],
        "authoritative_resolution": "active_manifest_immutable_relpath_plus_sha256",
    }


def _validate_artifact(
    dataset_root: Path,
    generation: Path,
    record: Mapping[str, Any],
    rig: Mapping[str, Any],
    representative_clip: Mapping[str, Any],
    cond_entries: Mapping[str, Any],
    conditioning_authority: Mapping[str, Any],
) -> dict[str, float]:
    relative_text = record["artifact_relpath"]
    if not isinstance(relative_text, str):
        raise CanonicalSkeletonValidationError(f"{record['rig_id']}: artifact path absent")
    artifact = (dataset_root / relative_text).resolve()
    if generation not in artifact.parents or not artifact.is_file():
        raise CanonicalSkeletonValidationError(
            f"{record['rig_id']}: artifact escapes/misses immutable generation"
        )
    _require_equal(
        f"{record['rig_id']} artifact SHA-256", _sha256_file(artifact), record["artifact_sha256"]
    )
    _require_equal(
        f"{record['rig_id']} artifact size", artifact.stat().st_size, record["artifact_size_bytes"]
    )
    if record["gate_status"] == "pass" and artifact.parent != generation:
        raise CanonicalSkeletonValidationError(f"{record['rig_id']}: pass artifact is not public")
    if record["gate_status"] != "pass" and artifact.parent != generation / "candidates":
        raise CanonicalSkeletonValidationError(f"{record['rig_id']}: blocked artifact is public")

    payload = _load_npz_pickle_free(artifact)
    rig_id = record["rig_id"]
    _require_equal(f"{rig_id} payload rig", _scalar_text(payload, "rig_id"), rig_id)
    _require_equal(
        f"{rig_id} source family", _scalar_text(payload, "source_family"), record["source_family"]
    )
    _require_equal(
        f"{rig_id} topology", _scalar_text(payload, "topology_family"), record["topology_family"]
    )
    _require_equal(
        f"{rig_id} artifact status", _scalar_text(payload, "artifact_status"), record["gate_status"]
    )
    _require_equal(
        f"{rig_id} reason codes", payload["reason_codes"].tolist(), record["reason_codes"]
    )
    _require_equal(
        f"{rig_id} skeleton version", _scalar_text(payload, "skeleton_format_version"), EXPECTED_QA_VERSION
    )
    _require_equal(
        f"{rig_id} representative clip",
        _scalar_text(payload, "representative_clip_id"),
        record["representative_clip_id"],
    )
    names = payload["joint_names"]
    kinds = payload["rotation_source_kind"]
    parents = payload["parents"]
    if names.ndim != 1 or names.dtype.kind not in "US" or len(set(names.tolist())) != len(names):
        raise CanonicalSkeletonValidationError(f"{rig_id}: invalid joint_names")
    if kinds.shape != names.shape or kinds.dtype.kind not in "US":
        raise CanonicalSkeletonValidationError(f"{rig_id}: invalid rotation_source_kind")
    _require_equal(f"{rig_id} joint names", names.tolist(), rig["joint_map"]["btjd_joint_names"])
    _require_equal(f"{rig_id} parents", parents.tolist(), rig["joint_map"]["btjd_parents"])
    _require_equal(
        f"{rig_id} rotation provenance", kinds.tolist(), rig["joint_map"]["rotation_source_kind"]
    )
    _require_equal(
        f"{rig_id} joint-map metadata", _parse_json_scalar(payload, "joint_map_metadata"), rig["joint_map"]
    )
    rest_path = Path(_scalar_text(payload, "source_rest_path")).expanduser().resolve()
    if not rest_path.is_file():
        raise CanonicalSkeletonValidationError(f"{rig_id}: source rest is missing: {rest_path}")
    rest_hash = _sha256_file(rest_path)
    _require_equal(f"{rig_id} source rest hash", _scalar_text(payload, "source_rest_sha256"), rest_hash)
    _require_equal(f"{rig_id} QA source rest hash", record["rest_source_sha256"], rest_hash)

    heading = _parse_json_scalar(payload, "heading_payload_provenance")
    transform = _parse_json_scalar(payload, "source_to_canonical_provenance")
    position_geometry = _parse_json_scalar(payload, "position_geometry_provenance")
    artifact_conditioning = _parse_json_scalar(payload, "conditioning_authority")
    artifact_rotation_signatures = _parse_json_scalar(
        payload, "fixed_rig_rotation_signatures"
    )
    unit = _parse_json_scalar(payload, "unit_metadata")
    _require_expected_forward_provenance(
        rig_id,
        record["source_family"],
        names.tolist(),
        heading,
    )
    _require_equal(f"{rig_id} heading provenance", heading, record["heading"]["provenance"])
    _require_equal(
        f"{rig_id} transform provenance", transform, record["source_to_canonical"]["provenance"]
    )
    if transform.get("no_motion_or_first_motion_frame_heading_used") is not True:
        raise CanonicalSkeletonValidationError(f"{rig_id}: motion frame influenced fixed transform")
    if transform.get("meter_claim") is not False or unit.get("meter_claim") is not False:
        raise CanonicalSkeletonValidationError(f"{rig_id}: unsupported meter claim")
    if payload["source_unit_to_meter"].shape != (0,) or payload["source_unit_to_meter"].dtype != np.float64:
        raise CanonicalSkeletonValidationError(f"{rig_id}: nullable meter field is malformed")

    numeric = _validate_numeric_rest(rig_id, payload)
    (
        source_positions,
        source_rotations,
        expected_C,
        expected_alpha,
        expected_o,
        expected_positions,
    ) = (
        _independent_source_rest(payload, rig, heading, cond_entries)
    )
    _require_close(f"{rig_id} live C", payload["source_to_canonical_C"], expected_C, atol=1e-12)
    alpha = _scalar_float(payload, "source_to_canonical_alpha")
    _require_close(f"{rig_id} live alpha", alpha, expected_alpha, atol=1e-12)
    _require_close(f"{rig_id} canonical scale alias", _scalar_float(payload, "canonical_scale_factor"), alpha, atol=0.0)
    _require_close(f"{rig_id} live origin", payload["source_to_canonical_o"], expected_o, atol=1e-12)
    expected_rotations = expected_C @ source_rotations @ expected_C.T
    _require_close(f"{rig_id} live rest positions", payload["P_rest_global"], expected_positions, atol=2e-12)
    _require_close(f"{rig_id} live rest rotations", payload["R_rest_global"], expected_rotations, atol=2e-12)

    expected_local = np.empty_like(expected_rotations)
    expected_offsets = np.zeros_like(expected_positions)
    expected_local[0] = expected_rotations[0]
    for child in range(1, len(parents)):
        parent = int(parents[child])
        expected_local[child] = expected_rotations[parent].T @ expected_rotations[child]
        expected_offsets[child] = expected_rotations[parent].T @ (
            expected_positions[child] - expected_positions[parent]
        )
    _require_close(f"{rig_id} live rest local", payload["R_rest_local"], expected_local, atol=2e-12)
    _require_close(f"{rig_id} live rest offsets", payload["offset_parent_local"], expected_offsets, atol=2e-12)
    carrier = payload["heading_carrier_joint"]
    if carrier.shape != () or carrier.dtype != np.int64 or int(carrier) != 0:
        raise CanonicalSkeletonValidationError(f"{rig_id}: heading carrier must be int64 root 0")
    expected_forward_local = expected_rotations[0].T @ np.asarray([0.0, 0.0, 1.0])
    expected_forward_local /= np.linalg.norm(expected_forward_local)
    if payload["u_forward_local"].shape != (3,) or payload["u_forward_local"].dtype != np.float64:
        raise CanonicalSkeletonValidationError(f"{rig_id}: malformed u_forward_local")
    _require_close(
        f"{rig_id} heading local forward", payload["u_forward_local"], expected_forward_local, atol=2e-12
    )
    _require_close(
        f"{rig_id} forward maps to +Z",
        expected_C @ _forward_from_provenance(names.tolist(), source_positions, heading),
        np.asarray([0.0, 0.0, 1.0]),
        atol=1e-12,
    )
    if record["source_family"] == "truebones":
        try:
            live_fixed = _truebones_fixed_live(
                dict(representative_clip), dict(rig), dict(cond_entries)
            )
        except Exception as exc:  # noqa: BLE001
            raise CanonicalSkeletonValidationError(
                f"{rig_id}: independent fixed-rig motion validation failed: {exc}"
            ) from exc
        fixed_record = record.get("fixed_rig")
        if not isinstance(fixed_record, dict) or fixed_record.get("status") != "pass":
            raise CanonicalSkeletonValidationError(
                f"{rig_id}: T04 QA lacks fixed-rig pass evidence"
            )
        _require_equal(
            f"{rig_id} conditioning authority",
            artifact_conditioning,
            dict(conditioning_authority),
        )
        _require_equal(
            f"{rig_id} QA conditioning authority",
            fixed_record.get("conditioning_authority"),
            dict(conditioning_authority),
        )
        _require_equal(
            f"{rig_id} conditioning payload hash",
            _scalar_text(payload, "conditioning_payload_sha256"),
            live_fixed["conditioning_payload_sha256"],
        )
        _require_equal(
            f"{rig_id} QA conditioning payload hash",
            fixed_record.get("conditioning_payload_sha256"),
            live_fixed["conditioning_payload_sha256"],
        )
        _require_equal(
            f"{rig_id} full rotation signatures",
            artifact_rotation_signatures,
            live_fixed["rotation_signatures"],
        )
        _require_equal(
            f"{rig_id} QA rotation signatures",
            fixed_record.get("rotation_signatures"),
            live_fixed["rotation_signatures"],
        )
        _require_equal(
            f"{rig_id} position geometry provenance",
            position_geometry,
            fixed_record.get("provenance"),
        )
        _require_equal(
            f"{rig_id} transform mixed authority",
            transform.get("mixed_authority_contract"),
            position_geometry,
        )
        if position_geometry.get("forbidden_inputs_used") is not False:
            raise CanonicalSkeletonValidationError(
                f"{rig_id}: forbidden input entered fixed-rig artifact"
            )
        for name, value in live_fixed["metrics"].items():
            _require_close(
                f"{rig_id} fixed-rig metric {name}",
                float(fixed_record["metrics"][name]),
                float(value),
                atol=1e-10,
            )
        edges = [
            np.linalg.norm(
                payload["P_rest_global"][child]
                - payload["P_rest_global"][int(parents[child])]
            )
            for child in range(1, len(parents))
        ]
        _require_close(
            f"{rig_id} Truebones mean edge", float(np.mean(edges)), TRUEBONES_MEAN_EDGE_TARGET, atol=1e-8
        )
    else:
        _require_equal(
            f"{rig_id} non-Truebones conditioning authority",
            artifact_conditioning,
            {"status": "not_applicable"},
        )
    _require_equal(f"{rig_id} unit length id", _scalar_text(payload, "length_unit_id"), unit["length_unit_id"])
    _require_close(f"{rig_id} unit alpha", unit["canonical_scale_factor"], alpha, atol=1e-12)
    _require_close(f"{rig_id} unit s_rig", unit["s_rig"], _scalar_float(payload, "s_rig"), atol=1e-12)

    for metric_name, live_value in numeric.items():
        _require_close(
            f"{rig_id} reported metric {metric_name}",
            float(record["metrics"][metric_name]),
            live_value,
            atol=2e-12,
        )
    return numeric


def validate_canonical_skeleton_outputs(manifest_root: str | Path) -> dict[str, Any]:
    """Validate an active immutable T04 manifest and all 32 skeleton artifacts."""
    root = Path(manifest_root).expanduser().resolve()
    transaction = _validate_manifest_transaction(root)
    stage_record = _load_json(root / "canonical_skeleton_generation.json")
    _require_equal("T04 stage QA version", stage_record.get("qa_version"), EXPECTED_QA_VERSION)
    _require_equal("T04 stage encoder flag", stage_record.get("encoder_called"), False)
    _require_equal("T04 stage visual flag", stage_record.get("motion_visual_qa_claimed"), False)
    inventory_summary = _load_json(root / "inventory_summary.json")
    try:
        cond_entries, conditioning_authority = _load_conditioning_independent(
            inventory_summary
        )
    except Exception as exc:  # noqa: BLE001
        raise CanonicalSkeletonValidationError(
            f"cannot independently establish conditioning authority: {exc}"
        ) from exc
    _require_equal(
        "T04 stage conditioning authority",
        stage_record.get("truebones_conditioning_authority"),
        conditioning_authority,
    )
    _require_equal("compatibility authoritative flag", stage_record.get("compatibility_symlink_authoritative"), False)
    dataset_root, generation = _resolve_skeleton_generation(root, stage_record)
    source_fk_root = _validate_direct_t03_lineage(dataset_root, stage_record)
    _validate_skeleton_transaction(generation, stage_record)

    manifest_qa = _load_jsonl(root / "canonical_skeleton_qa.jsonl")
    generation_qa = _load_jsonl(generation / "canonical_skeleton_qa.jsonl")
    _require_equal("manifest/generation QA", manifest_qa, generation_qa)
    manifest_summary = _load_json(root / "canonical_skeleton_summary.json")
    generation_summary = _load_json(generation / "canonical_skeleton_summary.json")
    _require_equal("manifest/generation summary", manifest_summary, generation_summary)
    _require_equal("summary expected outcome", manifest_summary.get("expected_outcomes_satisfied"), True)
    _require_equal(
        "summary status",
        manifest_summary.get("status"),
        "pass_with_declared_review_and_reject_records",
    )
    _require_equal("summary encoder flag", manifest_summary["scope"].get("encoder_called"), False)
    _require_equal("summary visual flag", manifest_summary["scope"].get("motion_visual_qa_claimed"), False)
    _require_equal("summary full conversion", manifest_summary.get("full_conversion_allowed"), False)
    _require_equal(
        "summary source-FK generation",
        manifest_summary.get("source_fk_manifest_generation_id"),
        stage_record.get("source_fk_manifest_generation_id"),
    )
    _require_equal(
        "summary source-FK transaction SHA-256",
        manifest_summary.get("source_fk_manifest_transaction_sha256"),
        stage_record.get("source_fk_manifest_transaction_sha256"),
    )

    if len(manifest_qa) != 58 or len({record["rig_id"] for record in manifest_qa}) != 58:
        raise CanonicalSkeletonValidationError("T04 QA must contain 58 unique rigs")
    _require_equal("QA order", [item["rig_id"] for item in manifest_qa], sorted(item["rig_id"] for item in manifest_qa))
    gate_counts = Counter(record["gate_status"] for record in manifest_qa)
    source_counts = Counter(record["source_family"] for record in manifest_qa)
    artifact_records = [record for record in manifest_qa if record["artifact_relpath"]]
    artifact_source_counts = Counter(record["source_family"] for record in artifact_records)
    _require_equal("T04 gate counts", gate_counts, EXPECTED_GATE_COUNTS)
    _require_equal("T04 source counts", source_counts, EXPECTED_SOURCE_COUNTS)
    _require_equal("T04 artifact source counts", artifact_source_counts, EXPECTED_ARTIFACT_SOURCE_COUNTS)
    _require_equal("T04 artifact count", len(artifact_records), 32)
    _require_equal(
        "validator-owned Truebones forward-spec scope",
        {
            record["rig_id"]
            for record in manifest_qa
            if record["source_family"] == "truebones"
        },
        set(EXPECTED_TRUEBONES_FORWARD_SPECS),
    )
    _require_equal("T04 artifact topology coverage", {record["topology_family"] for record in artifact_records}, EXPECTED_TOPOLOGIES)
    _require_equal(
        "T04 public topology coverage",
        {record["topology_family"] for record in artifact_records if record["gate_status"] == "pass"},
        EXPECTED_TOPOLOGIES - {"human"},
    )
    _require_equal(
        "review rig set",
        {record["rig_id"] for record in manifest_qa if record["gate_status"] == "review"},
        EXPECTED_REVIEW_RIGS,
    )

    rigs_list = _load_jsonl(root / "rigs.jsonl")
    rigs = {record["rig_id"]: record for record in rigs_list}
    if len(rigs) != len(rigs_list):
        raise CanonicalSkeletonValidationError("duplicate rig ids in T04 manifest")
    qa_by_rig = {record["rig_id"]: record for record in manifest_qa}
    clips_list = _load_jsonl(root / "clips.jsonl")
    clips = {record["clip_id"]: record for record in clips_list}
    if len(clips) != len(clips_list):
        raise CanonicalSkeletonValidationError("duplicate clip ids in T04 manifest")
    maxima: dict[str, float] = {}
    for index, record in enumerate(manifest_qa, start=1):
        rig_id = record["rig_id"]
        _require_equal(f"{rig_id} QA version", record.get("qa_version"), EXPECTED_QA_VERSION)
        _require_equal(f"{rig_id} encoder flag", record.get("encoder_called"), False)
        _require_equal(f"{rig_id} visual flag", record.get("motion_visual_qa_claimed"), False)
        rig = rigs[rig_id]
        _require_equal(f"{rig_id} rig canonical metadata", rig.get("canonical_skeleton"), _canonical_metadata(record))
        if record["source_family"] == "planetzoo":
            _require_equal(f"{rig_id} PZ gate", record["gate_status"], "reject")
            _require_equal(f"{rig_id} PZ reasons", record["reason_codes"], ["CANONICAL_TRANSFORM_PROVENANCE_INVALID"])
            _require_equal(f"{rig_id} PZ artifact", record["artifact_relpath"], None)
            if record["source_to_canonical"] is not None or record["heading"] is not None:
                raise CanonicalSkeletonValidationError(f"{rig_id}: rejected PZ has fabricated canonical metadata")
        else:
            if record["source_family"] == "motionstreamer272":
                _require_equal(f"{rig_id} human gate", record["gate_status"], "reject")
                _require_equal(f"{rig_id} human reasons", record["reason_codes"], ["HUMAN_FIXED_REST_UNRESOLVED"])
            elif rig_id in EXPECTED_REVIEW_RIGS:
                _require_equal(f"{rig_id} fallback reasons", record["reason_codes"], ["REST_FRAME_FALLBACK_NOT_EXPLICIT"])
            else:
                _require_equal(f"{rig_id} pass reasons", record["reason_codes"], [])
            representative_clip = clips[record["representative_clip_id"]]
            numeric = _validate_artifact(
                dataset_root,
                generation,
                record,
                rig,
                representative_clip,
                cond_entries,
                conditioning_authority,
            )
            for name, value in numeric.items():
                maxima[name] = max(maxima.get(name, 0.0), value)
        if index % 10 == 0 or index == len(manifest_qa):
            print(f"[canonical-skeleton-validation] checked {index}/{len(manifest_qa)} rigs", flush=True)

    clip_counts: Counter[str] = Counter()
    for clip in clips_list:
        qa = qa_by_rig.get(clip["rig_id"])
        if qa is None:
            if "canonical_skeleton" in clip or "split_eligible_for_ktjd17_t04" in clip:
                raise CanonicalSkeletonValidationError(
                    f"non-T04 clip gained canonical metadata: {clip['clip_id']}"
                )
            continue
        clip_counts[qa["gate_status"]] += 1
        _require_equal(
            f"{clip['clip_id']} canonical metadata",
            clip.get("canonical_skeleton"),
            _canonical_metadata(qa),
        )
        expected_eligible = bool(
            qa["gate_status"] == "pass"
            and clip.get("split") == "train"
            and isinstance(clip.get("source_parser_fk"), dict)
            and clip["source_parser_fk"].get("status") == "pass"
        )
        _require_equal(
            f"{clip['clip_id']} T04 eligibility",
            clip.get("split_eligible_for_ktjd17_t04"),
            expected_eligible,
        )
        for reason in qa["reason_codes"]:
            if reason not in clip.get("reason_codes", []):
                raise CanonicalSkeletonValidationError(
                    f"{clip['clip_id']} lacks T04 reason {reason}"
                )

    compatibility = dataset_root / "skeletons"
    if not compatibility.is_symlink() or compatibility.resolve() != generation:
        raise CanonicalSkeletonValidationError(
            "current compatibility skeleton link does not match the active manifest"
        )
    return {
        "qa_version": EXPECTED_QA_VERSION,
        "manifest_version": EXPECTED_INVENTORY_VERSION,
        "manifest_generation_id": transaction["generation_id"],
        "skeleton_generation_id": stage_record["skeleton_generation_id"],
        "source_fk_manifest_generation_id": source_fk_root.name,
        "validated_at_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "pass_with_declared_review_and_reject_records",
        "validation_mode": "independent_scipy_rest_reparse_transform_rederive_and_rest_fk",
        "audited_rig_count": len(manifest_qa),
        "artifact_count": len(artifact_records),
        "public_pass_artifact_count": sum(record["gate_status"] == "pass" for record in artifact_records),
        "candidate_artifact_count": sum(record["gate_status"] != "pass" for record in artifact_records),
        "gate_status_counts": dict(sorted(gate_counts.items())),
        "source_family_counts": dict(sorted(source_counts.items())),
        "affected_clip_status_counts": dict(sorted(clip_counts.items())),
        "numeric_gate_maxima": dict(sorted(maxima.items())),
        "artifact_topology_families": sorted({record["topology_family"] for record in artifact_records}),
        "authoritative_resolution": "active_manifest_immutable_relpath_plus_sha256",
        "compatibility_symlink_checked_but_non_authoritative": True,
        "encoder_invocation_count": 0,
        "motion_visual_qa_claimed": False,
        "full_conversion_allowed": False,
    }


def write_canonical_skeleton_validation_report(
    report: dict[str, Any],
    path: str | Path,
    *,
    immutable_manifest_root: str | Path,
) -> None:
    """Atomically write post-publication evidence outside immutable generations."""
    target = Path(path).expanduser().resolve()
    immutable_root = Path(immutable_manifest_root).expanduser().resolve()
    if target == immutable_root or immutable_root in target.parents:
        raise CanonicalSkeletonValidationError(
            "validation report cannot be written inside the immutable manifest generation"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()
