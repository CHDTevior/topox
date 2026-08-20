"""Fail-closed KTJD-17 T04 canonical-rest and skeleton artifact builder.

This stage consumes only T03-audited source records.  It does not encode a
motion clip.  A skeleton is public/encodable only when its fixed per-rig
source-to-canonical transform and one real source rest frame pass the numeric
gates in :mod:`.MECHANISM.md`.

The active manifest is the authoritative publication pointer.  Each manifest
record names an immutable skeleton-generation path and SHA-256.  The public
``dataset/skeletons`` symlink is only a compatibility view; readers must not
use it to choose a generation.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime as _datetime
import hashlib
import json
import math
import os
import shutil
import tempfile
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .inventory import (
    GENERATION_DIRECTORY_NAME,
    INVENTORY_VERSION,
    PROTOTYPE_FAMILIES,
    REASON_CODES,
    _canonical_json,
    _sha256_file,
    _sha256_json,
    _status_from_codes,
    _write_transaction,
)
from .schema import SKELETON_REQUIRED_KEYS
from .source_parser import (
    ParsedBvhMotion,
    ParsedSourceMotion,
    SourceParserError,
    parse_bvh_numeric,
    parse_bvh_source,
    parse_motionstreamer272_source,
)
from .truebones_fixed_rig import (
    COND_GEOMETRY_TOL,
    ConditioningCatalog,
    FixedRigMotion,
    TruebonesFixedRigError,
    build_fixed_rig_motion,
    load_conditioning_catalog,
)


CANONICAL_SKELETON_VERSION = "ktjd17-canonical-skeleton-v2"
SKELETON_GENERATION_DIRECTORY = ".ktjd17_skeleton_generations"
SKELETON_GENERATION_FILENAME = "skeleton_generation.json"
CANONICAL_QA_FILENAME = "canonical_skeleton_qa.jsonl"
CANONICAL_SUMMARY_FILENAME = "canonical_skeleton_summary.json"
MANIFEST_STAGE_FILENAME = "canonical_skeleton_generation.json"

# This is the exact current BTJD/AnyTop mean-nonroot-edge target.  It is a
# fixed canonical scale (alpha), not the KTJD normalization scale s_rig.
TRUEBONES_BTJD_MEAN_EDGE_TARGET = 0.2092142857142857

BASIS_ORTHOGONALITY_TOL = 1e-12
BASIS_DETERMINANT_TOL = 1e-12
REST_LOCAL_FLOAT64_TOL = 1e-12
REST_IDENTITY_FLOAT64_TOL = 1e-10
REST_IDENTITY_FLOAT32_TOL = 2e-6
REST_LOCAL_FLOAT32_TOL = 2e-6
REST_POSITION_FLOAT64_NORM_TOL = 1e-12
REST_POSITION_FLOAT32_NORM_TOL = 1e-5
FORWARD_ALIGNMENT_TOL = 1e-12

REQUIRED_PARENT_MANIFEST_FILES = (
    "clips.jsonl",
    "rigs.jsonl",
    "inventory_summary.json",
    "inventory_reason_codes.json",
    "prototype_candidates.json",
    "prototype_gaps.jsonl",
    "inventory_generation.json",
    "source_fk_qa.jsonl",
    "source_fk_summary.json",
    "source_fk_generation.json",
)
T03_PARENT_TRANSACTION_FILES = frozenset(
    name for name in REQUIRED_PARENT_MANIFEST_FILES if name != "inventory_generation.json"
)
SOURCE_FK_EVIDENCE_FILES = (
    "source_fk_qa.jsonl",
    "source_fk_summary.json",
    "source_fk_generation.json",
)


class CanonicalSkeletonError(RuntimeError):
    """The T04 input, derivation, or publication violates the contract."""


@dataclasses.dataclass(frozen=True)
class CanonicalSkeletonConfig:
    manifest_root: Path
    skeleton_output_root: Path
    manifest_output_root: Path
    overwrite: bool = False

    def resolved(self) -> "CanonicalSkeletonConfig":
        # Resolve the input snapshot exactly once.  Output leaves are atomic
        # symlinks and therefore must not be resolve()d through an old target.
        manifest_root = self.manifest_root.expanduser().resolve()

        def output_leaf(path: Path) -> Path:
            candidate = path.expanduser()
            if not candidate.is_absolute():
                candidate = Path.cwd() / candidate
            return candidate.parent.resolve() / candidate.name

        return dataclasses.replace(
            self,
            manifest_root=manifest_root,
            skeleton_output_root=output_leaf(self.skeleton_output_root),
            manifest_output_root=output_leaf(self.manifest_output_root),
        )


@dataclasses.dataclass(frozen=True)
class ForwardSpec:
    method: str
    anchor_names: tuple[str, ...]
    provenance: str


# These names freeze the reviewed T04 source-rest anatomy.  They are copied as
# names rather than indices so a future hierarchy reorder fails closed.
TRUEBONES_FORWARD_SPECS: dict[str, ForwardSpec] = {
    "Alligator": ForwardSpec("lateral_pairs", ("R_momo", "L_momo", "R_hiji", "L_hiji"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Anaconda": ForwardSpec("root_to_head", ("Hips", "BN_Tone_04"), "coiled_snake_root_to_tongue_endpoint_reviewed_t04"),
    "Bat": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_R_UpperArm_01", "BN_L_UpperArm_01"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Bird": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_Forearm_R_01", "BN_Forearm_L_01"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Buffalo": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_Clavicle", "Bip01_L_Clavicle"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Buzzard": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_Wing_R_02", "BN_Wing_L_02"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Cat": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_Clavicle", "Bip01_L_Clavicle"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Chicken": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_Finger_R_01", "BN_Finger_L_01"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Coyote": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_Clavicle", "Bip01_L_Clavicle"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Crocodile": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Dragon": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Eagle": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_Wing_R_02", "BN_Wing_L_02"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Flamingo": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_Forearm_R_02", "BN_Forearm_L_02"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Fox": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Gazelle": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Hamster": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "HermitCrab": ForwardSpec("lateral_pairs", ("BN_Leg_R_09", "BN_Leg_L_09", "BN_Crab_pincers_R_02", "BN_Crab_pincers_L_02"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Hippopotamus": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_Clavicle", "Bip01_L_Clavicle"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Horse": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Hound": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "KingCobra": ForwardSpec("root_to_head", ("Hips", "BN_Tongue_02"), "snake_root_to_tongue_endpoint_reviewed_t04"),
    "Lion": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Lynx": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Mammoth": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Ostrich": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_Forearm_R_02", "BN_Forearm_L_02"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Parrot": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_Wing_R_02", "BN_Wing_L_02"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Parrot2": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_Wing_R_02", "BN_Wing_L_02"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Pteranodon": ForwardSpec("lateral_pairs", ("jt_Thigh_R", "jt_Thigh_L", "jt_Elbow_R", "jt_Elbow_L"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Scorpion": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh_4", "Bip01_L_Thigh1_4", "Bip01_R_Forearm", "Bip01_L_Forearm"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "SpiderG": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Tukan": ForwardSpec("lateral_pairs", ("R_momo", "L_momo", "R_kata", "L_kata"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
}


@dataclasses.dataclass
class BuiltSkeleton:
    rig_id: str
    source_family: str
    topology_family: str
    representative_clip_id: str
    parsed: ParsedSourceMotion
    C: np.ndarray
    alpha: float
    o: np.ndarray
    P_rest_global: np.ndarray
    R_rest_global: np.ndarray
    R_rest_local: np.ndarray
    offset_parent_local: np.ndarray
    heading_carrier_joint: int
    u_forward_local: np.ndarray
    source_forward: np.ndarray
    forward_spec: ForwardSpec
    s_rig: float
    length_unit_id: str
    source_unit_to_meter: float | None
    metrics: dict[str, float]
    artifact_status: str
    reason_codes: list[str]
    heading_provenance: dict[str, Any]
    transform_provenance: dict[str, Any]
    rig_record: dict[str, Any]
    fixed_rig_motion: FixedRigMotion | None = None
    conditioning_authority: dict[str, Any] | None = None
    conditioning_payload_sha256: str | None = None
    artifact_relpath: str | None = None
    artifact_sha256: str | None = None
    artifact_size_bytes: int | None = None


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise CanonicalSkeletonError(f"cannot read JSON {path}: {exc}") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise CanonicalSkeletonError(f"{path}:{line_number}: blank JSONL row")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise CanonicalSkeletonError(
                        f"{path}:{line_number}: record is not an object"
                    )
                records.append(value)
    except CanonicalSkeletonError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise CanonicalSkeletonError(f"cannot read JSONL {path}: {exc}") from exc
    return records


def _validate_direct_t03_parent(
    root: Path, transaction: Mapping[str, Any]
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Require an immutable, hash-complete T03 input rather than a T04 rerun."""
    generation_id = transaction.get("generation_id")
    files = transaction.get("files")
    if not isinstance(generation_id, str) or not generation_id:
        raise CanonicalSkeletonError("parent manifest has no generation id")
    if root.parent.name != GENERATION_DIRECTORY_NAME or root.name != generation_id:
        raise CanonicalSkeletonError(
            "T04 input must resolve directly to its immutable T03 manifest generation"
        )
    if not isinstance(files, dict) or frozenset(files) != T03_PARENT_TRANSACTION_FILES:
        raise CanonicalSkeletonError(
            "T04 input must be the direct T03 generation with exactly the source-FK "
            "manifest file set; pass --manifest-root to that immutable generation "
            "instead of rerunning from an existing T04 manifest"
        )
    for name in sorted(files):
        path = root / name
        record = files[name]
        if not path.is_file() or not isinstance(record, dict):
            raise CanonicalSkeletonError(f"T03 transaction artifact is absent: {path}")
        if path.stat().st_size != record.get("size_bytes"):
            raise CanonicalSkeletonError(f"T03 transaction size mismatch: {path}")
        if _sha256_file(path) != record.get("sha256"):
            raise CanonicalSkeletonError(f"T03 transaction SHA-256 mismatch: {path}")
    source_fk_files = {
        name: copy.deepcopy(files[name]) for name in SOURCE_FK_EVIDENCE_FILES
    }
    return generation_id, source_fk_files


def _jsonl_chunks(records: Iterable[dict[str, Any]]) -> Iterable[str]:
    for record in records:
        yield _canonical_json(record).decode("utf-8") + "\n"


def _require_finite_float64(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.float64:
        raise CanonicalSkeletonError(f"{name} must be float64, got {array.dtype}")
    if not np.isfinite(array).all():
        bad = np.argwhere(~np.isfinite(array))[0].tolist()
        raise CanonicalSkeletonError(f"{name} contains non-finite value at {bad}")
    return array


def validate_source_to_canonical_basis(C: np.ndarray) -> dict[str, float]:
    basis = _require_finite_float64("source_to_canonical_C", np.asarray(C))
    if basis.shape != (3, 3):
        raise CanonicalSkeletonError(f"C must be [3,3], got {basis.shape}")
    orthogonality = float(np.max(np.abs(basis.T @ basis - np.eye(3))))
    determinant = float(np.linalg.det(basis))
    determinant_abs_error = abs(abs(determinant) - 1.0)
    if orthogonality > BASIS_ORTHOGONALITY_TOL:
        raise CanonicalSkeletonError(
            f"C.T@C gate failed: {orthogonality} > {BASIS_ORTHOGONALITY_TOL}"
        )
    if determinant_abs_error > BASIS_DETERMINANT_TOL:
        raise CanonicalSkeletonError(
            f"abs(abs(det(C))-1) gate failed: {determinant_abs_error}"
        )
    return {
        "basis_orthogonality_max_abs": orthogonality,
        "basis_determinant": determinant,
        "basis_determinant_abs_error": determinant_abs_error,
    }


def apply_canonical_positions(
    positions_source: np.ndarray, *, C: np.ndarray, alpha: float, o: np.ndarray
) -> np.ndarray:
    positions = _require_finite_float64(
        "positions_source", np.asarray(positions_source)
    )
    basis = _require_finite_float64("source_to_canonical_C", np.asarray(C))
    origin = _require_finite_float64("source_to_canonical_o", np.asarray(o))
    if positions.shape[-1] != 3 or origin.shape != (3,):
        raise CanonicalSkeletonError(
            f"position/origin shapes must end in 3 and [3], got {positions.shape}/{origin.shape}"
        )
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise CanonicalSkeletonError(f"alpha must be finite and positive, got {alpha}")
    return _require_finite_float64(
        "positions_canonical", alpha * np.matmul(positions - origin, basis.T)
    )


def apply_canonical_rotations(rotations_source: np.ndarray, *, C: np.ndarray) -> np.ndarray:
    rotations = _require_finite_float64(
        "rotations_source", np.asarray(rotations_source)
    )
    basis = _require_finite_float64("source_to_canonical_C", np.asarray(C))
    if rotations.shape[-2:] != (3, 3):
        raise CanonicalSkeletonError(
            f"source rotations must end in [3,3], got {rotations.shape}"
        )
    canonical = np.matmul(np.matmul(basis, rotations), basis.T)
    return _require_finite_float64("rotations_canonical", canonical)


def derive_rest_local_arrays(
    parents: np.ndarray,
    P_rest_global: np.ndarray,
    R_rest_global: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    parent_array = np.asarray(parents, dtype=np.int64)
    positions = _require_finite_float64(
        "P_rest_global", np.asarray(P_rest_global)
    )
    rotations = _require_finite_float64(
        "R_rest_global", np.asarray(R_rest_global)
    )
    joint_count = positions.shape[0]
    if positions.shape != (joint_count, 3) or rotations.shape != (joint_count, 3, 3):
        raise CanonicalSkeletonError(
            f"rest shape mismatch: P={positions.shape}, R={rotations.shape}"
        )
    if parent_array.shape != (joint_count,) or int(parent_array[0]) != -1:
        raise CanonicalSkeletonError("parents must have one root at index 0")
    for child in range(1, joint_count):
        if not 0 <= int(parent_array[child]) < child:
            raise CanonicalSkeletonError(
                f"parent-before-child failed at {child}: {parent_array[child]}"
            )

    local_rotations = np.empty_like(rotations)
    offsets = np.zeros_like(positions)
    local_rotations[0] = rotations[0]
    for child in range(1, joint_count):
        parent = int(parent_array[child])
        parent_inverse = rotations[parent].T
        local_rotations[child] = parent_inverse @ rotations[child]
        offsets[child] = parent_inverse @ (positions[child] - positions[parent])
    return (
        _require_finite_float64("R_rest_local", local_rotations),
        _require_finite_float64("offset_parent_local", offsets),
    )


def _rest_fk(
    parents: np.ndarray,
    root_position: np.ndarray,
    R_rest_global: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    dtype = np.result_type(root_position, R_rest_global, offsets)
    result = np.empty((len(parents), 3), dtype=dtype)
    result[0] = root_position
    for child in range(1, len(parents)):
        parent = int(parents[child])
        result[child] = result[parent] + R_rest_global[parent] @ offsets[child]
    return result


def _encode_column_cont6d(rotations: np.ndarray) -> np.ndarray:
    matrices = np.asarray(rotations)
    return np.concatenate((matrices[..., :, 0], matrices[..., :, 1]), axis=-1)


def _rotation_metrics(rotations: np.ndarray) -> tuple[float, float, float]:
    matrices = np.asarray(rotations, dtype=np.float64)
    gram = np.matmul(np.swapaxes(matrices, -1, -2), matrices)
    determinants = np.linalg.det(matrices)
    return (
        float(np.max(np.abs(gram - np.eye(3)))),
        float(np.min(determinants)),
        float(np.max(determinants)),
    )


def _aabb_diagonal(positions: np.ndarray) -> float:
    points = np.asarray(positions, dtype=np.float64)
    value = float(np.linalg.norm(np.ptp(points, axis=0)))
    if not math.isfinite(value) or value <= 0.0:
        raise CanonicalSkeletonError(f"rest AABB diagonal must be positive, got {value}")
    return value


def _forward_from_rest(
    joint_names: Sequence[str], positions: np.ndarray, spec: ForwardSpec
) -> tuple[np.ndarray, list[int]]:
    lookup: dict[str, list[int]] = defaultdict(list)
    for index, name in enumerate(joint_names):
        lookup[str(name)].append(index)
    indices: list[int] = []
    for name in spec.anchor_names:
        hits = lookup.get(name, [])
        if len(hits) != 1:
            raise CanonicalSkeletonError(
                f"forward anchor {name!r} must resolve once, got {hits}"
            )
        indices.append(hits[0])

    if spec.method == "lateral_pairs":
        if len(indices) != 4:
            raise CanonicalSkeletonError("lateral_pairs requires four anchors")
        across = (
            positions[indices[0]]
            - positions[indices[1]]
            + positions[indices[2]]
            - positions[indices[3]]
        )
        forward = np.cross(np.array([0.0, 1.0, 0.0]), across)
    elif spec.method == "root_to_head":
        if len(indices) != 2:
            raise CanonicalSkeletonError("root_to_head requires two anchors")
        forward = positions[indices[1]] - positions[indices[0]]
    elif spec.method == "declared_plus_z":
        forward = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        raise CanonicalSkeletonError(f"unknown forward method {spec.method!r}")
    horizontal = np.asarray([forward[0], 0.0, forward[2]], dtype=np.float64)
    norm = float(np.linalg.norm(horizontal))
    scale = _aabb_diagonal(np.asarray(positions, dtype=np.float64))
    if not math.isfinite(norm) or norm <= 1e-10 * scale:
        raise CanonicalSkeletonError(
            f"forward anchor is horizontally degenerate: norm={norm}, s_rig_src={scale}"
        )
    return horizontal / norm, indices


def _yaw_basis_to_plus_z(source_forward: np.ndarray) -> np.ndarray:
    forward = _require_finite_float64("source_forward", np.asarray(source_forward))
    if forward.shape != (3,):
        raise CanonicalSkeletonError(f"source_forward must be [3], got {forward.shape}")
    fx, fz = float(forward[0]), float(forward[2])
    horizontal_norm = math.hypot(fx, fz)
    if horizontal_norm <= 0.0:
        raise CanonicalSkeletonError("source forward has zero XZ norm")
    fx, fz = fx / horizontal_norm, fz / horizontal_norm
    # Active right-handed Y rotation with angle -atan2(fx, fz).
    return np.asarray(
        [[fz, 0.0, -fx], [0.0, 1.0, 0.0], [fx, 0.0, fz]],
        dtype=np.float64,
    )


def _fixed_origin_from_rest(positions_source: np.ndarray, C: np.ndarray) -> np.ndarray:
    rotated = np.matmul(positions_source, C.T)
    canonical_anchor = np.asarray(
        [rotated[0, 0], np.min(rotated[:, 1]), rotated[0, 2]],
        dtype=np.float64,
    )
    return _require_finite_float64(
        "source_to_canonical_o", C.T @ canonical_anchor
    )


def _rest_gate_metrics(
    *,
    parents: np.ndarray,
    C: np.ndarray,
    P_rest_global: np.ndarray,
    R_rest_global: np.ndarray,
    R_rest_local: np.ndarray,
    offset_parent_local: np.ndarray,
    source_forward: np.ndarray,
    alpha: float,
    source_mean_edge: float,
    source_family: str,
) -> dict[str, float]:
    metrics = validate_source_to_canonical_basis(C)
    rotation_orth, rotation_det_min, rotation_det_max = _rotation_metrics(
        R_rest_global
    )
    if rotation_det_min <= 0.0 or rotation_orth > 1e-10:
        raise CanonicalSkeletonError(
            f"canonical rest rotations are not SO(3): orth={rotation_orth}, det_min={rotation_det_min}"
        )

    local_error = float(np.max(np.abs(R_rest_local[0] - R_rest_global[0])))
    for child in range(1, len(parents)):
        parent = int(parents[child])
        local_error = max(
            local_error,
            float(
                np.max(
                    np.abs(
                        R_rest_global[parent] @ R_rest_local[child]
                        - R_rest_global[child]
                    )
                )
            ),
        )
    if local_error > REST_LOCAL_FLOAT64_TOL:
        raise CanonicalSkeletonError(
            f"float64 rest local gate failed: {local_error}"
        )

    s_rig = _aabb_diagonal(P_rest_global)
    fk64 = _rest_fk(
        parents, P_rest_global[0], R_rest_global, offset_parent_local
    )
    position_error64 = float(np.max(np.linalg.norm(fk64 - P_rest_global, axis=-1)))
    if position_error64 / s_rig > REST_POSITION_FLOAT64_NORM_TOL:
        raise CanonicalSkeletonError(
            f"float64 rest position gate failed: {position_error64 / s_rig}"
        )

    identity = np.eye(3, dtype=np.float64)
    delta64 = np.matmul(R_rest_global, np.swapaxes(R_rest_global, -1, -2))
    identity_d6 = _encode_column_cont6d(identity)
    identity_error64 = float(
        np.max(np.abs(_encode_column_cont6d(delta64) - identity_d6))
    )
    if identity_error64 > REST_IDENTITY_FLOAT64_TOL:
        raise CanonicalSkeletonError(
            f"float64 rest identity gate failed: {identity_error64}"
        )

    R32 = R_rest_global.astype(np.float32)
    L32 = R_rest_local.astype(np.float32)
    O32 = offset_parent_local.astype(np.float32)
    P32 = P_rest_global.astype(np.float32)
    delta32 = np.matmul(R32, np.swapaxes(R32, -1, -2))
    identity_error32 = float(
        np.max(
            np.abs(
                _encode_column_cont6d(delta32).astype(np.float64) - identity_d6
            )
        )
    )
    if identity_error32 > REST_IDENTITY_FLOAT32_TOL:
        raise CanonicalSkeletonError(
            f"float32 rest identity gate failed: {identity_error32}"
        )
    local_error32 = float(np.max(np.abs(L32[0] - R32[0])))
    for child in range(1, len(parents)):
        parent = int(parents[child])
        local_error32 = max(
            local_error32,
            float(np.max(np.abs(R32[parent] @ L32[child] - R32[child]))),
        )
    if local_error32 > REST_LOCAL_FLOAT32_TOL:
        raise CanonicalSkeletonError(
            f"float32 rest local gate failed: {local_error32}"
        )
    fk32 = _rest_fk(parents, P32[0], R32, O32)
    position_error32 = float(
        np.max(np.linalg.norm(fk32.astype(np.float64) - P32.astype(np.float64), axis=-1))
    )
    if position_error32 / s_rig > REST_POSITION_FLOAT32_NORM_TOL:
        raise CanonicalSkeletonError(
            f"float32 rest position gate failed: {position_error32 / s_rig}"
        )

    forward_error = float(
        np.max(np.abs(C @ source_forward - np.array([0.0, 0.0, 1.0])))
    )
    if forward_error > FORWARD_ALIGNMENT_TOL:
        raise CanonicalSkeletonError(
            f"canonical forward does not map to +Z: {forward_error}"
        )
    root_xz_error = float(np.max(np.abs(P_rest_global[0, [0, 2]])))
    ground_error = abs(float(np.min(P_rest_global[:, 1])))
    origin_tol = 1e-12 * max(1.0, s_rig)
    if root_xz_error > origin_tol or ground_error > origin_tol:
        raise CanonicalSkeletonError(
            f"canonical rest origin/ground gate failed: root_xz={root_xz_error}, ground={ground_error}"
        )

    canonical_edges = np.asarray(
        [
            np.linalg.norm(P_rest_global[child] - P_rest_global[int(parents[child])])
            for child in range(1, len(parents))
        ],
        dtype=np.float64,
    )
    canonical_mean_edge = float(np.mean(canonical_edges))
    if source_family == "truebones":
        target_error = abs(canonical_mean_edge - TRUEBONES_BTJD_MEAN_EDGE_TARGET)
        if target_error > COND_GEOMETRY_TOL:
            raise CanonicalSkeletonError(
                f"Truebones mean-edge scale gate failed: {canonical_mean_edge}"
            )
    else:
        target_error = 0.0

    metrics.update(
        {
            "canonical_rotation_orthogonality_max_abs": rotation_orth,
            "canonical_rotation_determinant_min": rotation_det_min,
            "canonical_rotation_determinant_max": rotation_det_max,
            "rest_local_float64_max_abs": local_error,
            "rest_local_float32_max_abs": local_error32,
            "rest_position_fk_float64_max_abs": position_error64,
            "rest_position_fk_float64_max_norm": position_error64 / s_rig,
            "rest_position_fk_float32_max_abs": position_error32,
            "rest_position_fk_float32_max_norm": position_error32 / s_rig,
            "rest_identity_d6_float64_max_abs": identity_error64,
            "rest_identity_d6_float32_max_abs": identity_error32,
            "canonical_forward_to_plus_z_max_abs": forward_error,
            "canonical_root_xz_max_abs": root_xz_error,
            "canonical_ground_min_y_abs": ground_error,
            "source_mean_nonroot_edge_length": source_mean_edge,
            "canonical_mean_nonroot_edge_length": canonical_mean_edge,
            "canonical_mean_edge_target_abs_error": target_error,
            "source_to_canonical_alpha": alpha,
            "s_rig": s_rig,
        }
    )
    return metrics


def _rest_mode(rig: Mapping[str, Any], source_family: str) -> str:
    if source_family == "planetzoo":
        return "processed_hierarchy_only_review"
    method = rig.get("rest_pose", {}).get("selection_method")
    if method == "explicit_tpose_filename":
        return "explicit_tpose_frame"
    if method in {"legacy_idle_fallback", "legacy_first_file_fallback"}:
        return "legacy_idle_fallback_review"
    raise CanonicalSkeletonError(
        f"unsupported Truebones rest selection for {rig.get('rig_id')}: {method!r}"
    )


def _parse_manifest_source(
    clip: Mapping[str, Any],
    rig: Mapping[str, Any],
    *,
    rest_cache: dict[str, ParsedBvhMotion],
) -> ParsedSourceMotion:
    source = clip["source"]
    joint_map = rig["joint_map"]
    family = source["family"]
    if family == "motionstreamer272":
        return parse_motionstreamer272_source(
            source["path"],
            joint_names=joint_map["btjd_joint_names"],
            parents=joint_map["btjd_parents"],
            neutral_model_path=rig["rest_pose"]["source_path"],
        )
    if family != "truebones":
        raise CanonicalSkeletonError(
            f"T04 does not parse canonical skeletons from {family!r}"
        )
    rest_path = str(Path(rig["rest_pose"]["source_path"]).expanduser().resolve())
    if rest_path not in rest_cache:
        rest_cache[rest_path] = parse_bvh_numeric(rest_path)
    return parse_bvh_source(
        source["path"],
        retained_names=joint_map["btjd_joint_names"],
        retained_parents=joint_map["btjd_parents"],
        expected_rotation_kinds=joint_map["rotation_source_kind"],
        frame_slice=source["slice_frames"],
        rest_path=rest_path,
        rest_mode=_rest_mode(rig, family),
        parsed_rest=rest_cache[rest_path],
        family=family,
    )


def _build_one(
    rig: dict[str, Any],
    clip: dict[str, Any],
    *,
    rest_cache: dict[str, ParsedBvhMotion],
    conditioning_catalog: ConditioningCatalog,
) -> BuiltSkeleton:
    rig_id = rig["rig_id"]
    source_family = rig["source_family"]
    parsed = _parse_manifest_source(clip, rig, rest_cache=rest_cache)
    if parsed.joint_names != tuple(rig["joint_map"]["btjd_joint_names"]):
        raise CanonicalSkeletonError(f"{rig_id}: parsed joint names drifted")
    if not np.array_equal(parsed.parents, np.asarray(rig["joint_map"]["btjd_parents"])):
        raise CanonicalSkeletonError(f"{rig_id}: parsed parent tree drifted")

    fixed_motion: FixedRigMotion | None = None
    conditioning_payload_sha256: str | None = None
    if source_family == "truebones":
        try:
            spec = TRUEBONES_FORWARD_SPECS[rig_id]
        except KeyError as exc:
            raise CanonicalSkeletonError(
                f"{rig_id}: no explicitly reviewed Truebones forward spec"
            ) from exc
        fixed_geometry = conditioning_catalog.rig(
            rig_id,
            expected_names=rig["joint_map"]["btjd_joint_names"],
            expected_parents=rig["joint_map"]["btjd_parents"],
        )
        fixed_motion = build_fixed_rig_motion(parsed, fixed_geometry, spec)
        conditioning_payload_sha256 = fixed_geometry.payload_sha256
        source_forward = fixed_motion.source_forward
        anchor_indices = list(fixed_motion.forward_anchor_indices)
        C = fixed_motion.C
        source_edges = np.asarray(
            [
                np.linalg.norm(
                    parsed.rest_global_positions[child]
                    - parsed.rest_global_positions[int(parsed.parents[child])]
                )
                for child in range(1, parsed.joint_count)
            ],
            dtype=np.float64,
        )
        source_mean_edge = float(np.mean(source_edges))
        if not math.isfinite(source_mean_edge) or source_mean_edge <= 0.0:
            raise CanonicalSkeletonError(
                f"{rig_id}: invalid source mean edge {source_mean_edge}"
            )
        alpha = fixed_motion.alpha
        length_unit_id = "truebones_btjd_canonical_mean_rest_edge"
    elif source_family == "motionstreamer272":
        spec = ForwardSpec(
            "declared_plus_z",
            tuple(),
            "motionstreamer_humanml_declared_plus_z_candidate_t04",
        )
        source_forward = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        anchor_indices = []
        C = np.eye(3, dtype=np.float64)
        source_edges = np.asarray(
            [
                np.linalg.norm(
                    parsed.rest_global_positions[child]
                    - parsed.rest_global_positions[int(parsed.parents[child])]
                )
                for child in range(1, parsed.joint_count)
            ],
            dtype=np.float64,
        )
        source_mean_edge = float(np.mean(source_edges))
        alpha = 1.0
        length_unit_id = "motionstreamer272_native_unverified"
    else:
        raise CanonicalSkeletonError(f"unsupported source family {source_family!r}")

    validate_source_to_canonical_basis(C)
    if fixed_motion is not None:
        o = fixed_motion.o
        P_rest_global = fixed_motion.P_rest_global
        R_rest_global = fixed_motion.R_rest_global
        R_rest_local = fixed_motion.R_rest_local
        offset_parent_local = fixed_motion.offset_parent_local
    else:
        o = _fixed_origin_from_rest(parsed.rest_global_positions, C)
        P_rest_global = apply_canonical_positions(
            parsed.rest_global_positions, C=C, alpha=alpha, o=o
        )
        R_rest_global = apply_canonical_rotations(parsed.rest_global_rotations, C=C)
        R_rest_local, offset_parent_local = derive_rest_local_arrays(
            parsed.parents, P_rest_global, R_rest_global
        )
    heading_carrier_joint = 0
    u_forward_local = R_rest_global[heading_carrier_joint].T @ np.asarray(
        [0.0, 0.0, 1.0], dtype=np.float64
    )
    u_forward_local /= np.linalg.norm(u_forward_local)
    s_rig = _aabb_diagonal(P_rest_global)
    metrics = _rest_gate_metrics(
        parents=parsed.parents,
        C=C,
        P_rest_global=P_rest_global,
        R_rest_global=R_rest_global,
        R_rest_local=R_rest_local,
        offset_parent_local=offset_parent_local,
        source_forward=source_forward,
        alpha=alpha,
        source_mean_edge=source_mean_edge,
        source_family=source_family,
    )
    if fixed_motion is not None:
        metrics.update(
            {
                f"truebones_fixed_rig_{name}": value
                for name, value in fixed_motion.metrics.items()
            }
        )

    reason_codes: list[str] = []
    if source_family == "motionstreamer272":
        artifact_status = "reject"
        reason_codes.append("HUMAN_FIXED_REST_UNRESOLVED")
    elif rig["rest_pose"].get("selection_method") != "explicit_tpose_filename":
        artifact_status = "review"
        reason_codes.append("REST_FRAME_FALLBACK_NOT_EXPLICIT")
    else:
        artifact_status = "pass"

    heading_provenance = {
        "status": "static_anatomical_polarity_reviewed_t04",
        "carrier_joint": heading_carrier_joint,
        "carrier_name": parsed.joint_names[heading_carrier_joint],
        "local_forward_definition": "R_rest_global[carrier].T @ canonical_plus_z",
        "canonical_rest_forward": [0.0, 0.0, 1.0],
        "forward_method": spec.method,
        "forward_anchor_names": list(spec.anchor_names),
        "forward_anchor_indices": anchor_indices,
        "forward_spec_provenance": spec.provenance,
        "polarity": "canonical_plus_z",
        "dynamic_perspective_visual_status": "pending_t05",
        "heading_epsilon_status": "unfrozen",
    }
    transform_provenance = {
        "status": "numeric_fixed_per_rig_t04",
        "position_convention": (
            "P_rest=ground_shifted_cond_xyz; motion_root=alpha*C@(raw_root-o); "
            "motion_children=fixed_rotation_fk"
            if fixed_motion is not None
            else "P_can=alpha*C@(P_src-o)"
        ),
        "rotation_convention": "R_can=C@R_src@C.T",
        "C_method": "proper_yaw_from_reviewed_source_rest_anatomy",
        "alpha_method": (
            "0.2092142857142857/mean_nonroot_retained_rest_edge"
            if source_family == "truebones"
            else "identity_scale_motionstreamer_native"
        ),
        "origin_method": (
            "o=x_raw_rest_root-alpha^-1*C.T@P_cond_root"
            if fixed_motion is not None
            else "same_rest_root_xz_zero_and_rest_min_y_zero"
        ),
        "rest_source_path": parsed.rest_path,
        "rest_status": parsed.rest_status,
        "no_motion_or_first_motion_frame_heading_used": True,
        "s_rig_definition": "canonical_rest_aabb_diagonal",
        "meter_claim": False,
    }
    if fixed_motion is not None:
        transform_provenance["mixed_authority_contract"] = fixed_motion.provenance
        transform_provenance["conditioning_authority"] = (
            conditioning_catalog.authority_record()
        )
        transform_provenance["conditioning_payload_sha256"] = (
            conditioning_payload_sha256
        )
    return BuiltSkeleton(
        rig_id=rig_id,
        source_family=source_family,
        topology_family=rig["topology_family"],
        representative_clip_id=clip["clip_id"],
        parsed=parsed,
        C=C,
        alpha=float(alpha),
        o=o,
        P_rest_global=P_rest_global,
        R_rest_global=R_rest_global,
        R_rest_local=R_rest_local,
        offset_parent_local=offset_parent_local,
        heading_carrier_joint=heading_carrier_joint,
        u_forward_local=np.asarray(u_forward_local, dtype=np.float64),
        source_forward=source_forward,
        forward_spec=spec,
        s_rig=s_rig,
        length_unit_id=length_unit_id,
        source_unit_to_meter=None,
        metrics=metrics,
        artifact_status=artifact_status,
        reason_codes=reason_codes,
        heading_provenance=heading_provenance,
        transform_provenance=transform_provenance,
        rig_record=rig,
        fixed_rig_motion=fixed_motion,
        conditioning_authority=(
            conditioning_catalog.authority_record()
            if fixed_motion is not None
            else None
        ),
        conditioning_payload_sha256=conditioning_payload_sha256,
    )


def _json_scalar(value: Any) -> np.ndarray:
    text = _canonical_json(value).decode("utf-8")
    return np.asarray(text, dtype=f"<U{max(1, len(text))}")


def _text_scalar(value: str) -> np.ndarray:
    text = str(value)
    return np.asarray(text, dtype=f"<U{max(1, len(text))}")


def _artifact_payload(built: BuiltSkeleton) -> dict[str, np.ndarray]:
    source_unit_to_meter = (
        np.asarray([], dtype=np.float64)
        if built.source_unit_to_meter is None
        else np.asarray(built.source_unit_to_meter, dtype=np.float64)
    )
    unit_metadata = {
        "length_unit_id": built.length_unit_id,
        "source_unit_to_meter": built.source_unit_to_meter,
        "canonical_scale_factor": built.alpha,
        "s_rig": built.s_rig,
        "meter_claim": False,
    }
    payload: dict[str, np.ndarray] = {
        "joint_names": np.asarray(built.parsed.joint_names, dtype=np.str_),
        "parents": built.parsed.parents.astype(np.int64, copy=True),
        "P_rest_global": built.P_rest_global.astype(np.float64, copy=True),
        "R_rest_global": built.R_rest_global.astype(np.float64, copy=True),
        "R_rest_local": built.R_rest_local.astype(np.float64, copy=True),
        "offset_parent_local": built.offset_parent_local.astype(np.float64, copy=True),
        "rotation_source_kind": np.asarray(
            built.parsed.rotation_source_kind, dtype=np.str_
        ),
        "heading_carrier_joint": np.asarray(
            built.heading_carrier_joint, dtype=np.int64
        ),
        "u_forward_local": built.u_forward_local.astype(np.float64, copy=True),
        "heading_payload_provenance": _json_scalar(built.heading_provenance),
        "source_to_canonical_C": built.C.astype(np.float64, copy=True),
        "source_to_canonical_alpha": np.asarray(built.alpha, dtype=np.float64),
        "source_to_canonical_o": built.o.astype(np.float64, copy=True),
        "s_rig": np.asarray(built.s_rig, dtype=np.float64),
        "length_unit_id": _text_scalar(built.length_unit_id),
        "source_unit_to_meter": source_unit_to_meter,
        "canonical_scale_factor": np.asarray(built.alpha, dtype=np.float64),
        "joint_map_metadata": _json_scalar(built.rig_record["joint_map"]),
        # Additional provenance is intentionally pickle-free.
        "rig_id": _text_scalar(built.rig_id),
        "source_family": _text_scalar(built.source_family),
        "topology_family": _text_scalar(built.topology_family),
        "artifact_status": _text_scalar(built.artifact_status),
        "reason_codes": np.asarray(built.reason_codes, dtype=np.str_),
        "skeleton_format_version": _text_scalar(CANONICAL_SKELETON_VERSION),
        "representative_clip_id": _text_scalar(built.representative_clip_id),
        "source_rest_path": _text_scalar(str(built.parsed.rest_path)),
        "source_rest_sha256": _text_scalar(_sha256_file(Path(str(built.parsed.rest_path)))),
        "source_to_canonical_provenance": _json_scalar(
            built.transform_provenance
        ),
        "position_geometry_provenance": _json_scalar(
            (
                built.fixed_rig_motion.provenance
                if built.fixed_rig_motion is not None
                else {"status": "not_applicable"}
            )
        ),
        "conditioning_authority": _json_scalar(
            built.conditioning_authority or {"status": "not_applicable"}
        ),
        "conditioning_payload_sha256": _text_scalar(
            built.conditioning_payload_sha256 or "not_applicable"
        ),
        "fixed_rig_rotation_signatures": _json_scalar(
            (
                built.fixed_rig_motion.rotation_signatures
                if built.fixed_rig_motion is not None
                else {"status": "not_applicable"}
            )
        ),
        "unit_metadata": _json_scalar(unit_metadata),
    }
    missing = sorted(set(SKELETON_REQUIRED_KEYS) - set(payload))
    if missing:
        raise CanonicalSkeletonError(
            f"{built.rig_id}: artifact payload missing required keys {missing}"
        )
    if any(array.dtype.hasobject for array in payload.values()):
        raise CanonicalSkeletonError(f"{built.rig_id}: object dtype is forbidden")
    return payload


def _qa_record_from_built(built: BuiltSkeleton) -> dict[str, Any]:
    return {
        "qa_version": CANONICAL_SKELETON_VERSION,
        "rig_id": built.rig_id,
        "source_family": built.source_family,
        "topology_family": built.topology_family,
        "representative_clip_id": built.representative_clip_id,
        "gate_status": built.artifact_status,
        "reason_codes": list(built.reason_codes),
        "artifact_relpath": built.artifact_relpath,
        "artifact_sha256": built.artifact_sha256,
        "artifact_size_bytes": built.artifact_size_bytes,
        "rest_source_path": built.parsed.rest_path,
        "rest_source_sha256": _sha256_file(Path(str(built.parsed.rest_path))),
        "rest_status": built.parsed.rest_status,
        "joint_count": built.parsed.joint_count,
        "source_to_canonical": {
            "C": built.C.tolist(),
            "alpha": built.alpha,
            "o": built.o.tolist(),
            "provenance": built.transform_provenance,
        },
        "heading": {
            "carrier_joint": built.heading_carrier_joint,
            "u_forward_local": built.u_forward_local.tolist(),
            "provenance": built.heading_provenance,
        },
        "unit": {
            "length_unit_id": built.length_unit_id,
            "source_unit_to_meter": built.source_unit_to_meter,
            "canonical_scale_factor": built.alpha,
            "s_rig": built.s_rig,
            "meter_claim": False,
        },
        "metrics": built.metrics,
        "fixed_rig": (
            {
                "status": "pass",
                "conditioning_authority": built.conditioning_authority,
                "conditioning_payload_sha256": built.conditioning_payload_sha256,
                "metrics": built.fixed_rig_motion.metrics,
                "rotation_signatures": built.fixed_rig_motion.rotation_signatures,
                "provenance": built.fixed_rig_motion.provenance,
            }
            if built.fixed_rig_motion is not None
            else None
        ),
        "encoder_called": False,
        "motion_visual_qa_claimed": False,
    }


def _rejection_record(
    rig: Mapping[str, Any], representative_clip_id: str, reason_code: str, message: str
) -> dict[str, Any]:
    return {
        "qa_version": CANONICAL_SKELETON_VERSION,
        "rig_id": rig["rig_id"],
        "source_family": rig["source_family"],
        "topology_family": rig["topology_family"],
        "representative_clip_id": representative_clip_id,
        "gate_status": "reject",
        "reason_codes": [reason_code],
        "artifact_relpath": None,
        "artifact_sha256": None,
        "artifact_size_bytes": None,
        "rest_source_path": rig.get("rest_pose", {}).get("source_path"),
        "rest_source_sha256": None,
        "rest_status": rig.get("rest_pose", {}).get("status"),
        "joint_count": len(rig.get("joint_map", {}).get("btjd_joint_names", [])),
        "source_to_canonical": None,
        "heading": None,
        "unit": rig.get("unit"),
        "metrics": None,
        "error": {"type": reason_code, "message": message},
        "encoder_called": False,
        "motion_visual_qa_claimed": False,
    }


def _summarize_qa(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    artifact_records = [record for record in records if record["artifact_relpath"]]
    gate_counts = Counter(record["gate_status"] for record in records)
    source_counts = Counter(record["source_family"] for record in records)
    artifact_source_counts = Counter(
        record["source_family"] for record in artifact_records
    )
    artifact_topology_counts = Counter(
        record["topology_family"] for record in artifact_records
    )
    derivation_failures = [
        record["rig_id"]
        for record in records
        if "CANONICAL_SKELETON_DERIVATION_FAILED" in record["reason_codes"]
    ]
    expected_outcomes = bool(
        len(records) == 58
        and source_counts
        == Counter({"truebones": 31, "planetzoo": 26, "motionstreamer272": 1})
        and artifact_source_counts
        == Counter({"truebones": 31, "motionstreamer272": 1})
        and set(artifact_topology_counts) == set(PROTOTYPE_FAMILIES)
        and gate_counts == Counter({"pass": 27, "review": 4, "reject": 27})
        and not derivation_failures
    )
    return {
        "qa_version": CANONICAL_SKELETON_VERSION,
        "status": (
            "pass_with_declared_review_and_reject_records"
            if expected_outcomes
            else "fail_unexpected_t04_outcome"
        ),
        "expected_outcomes_satisfied": expected_outcomes,
        "derivation_failure_rigs": derivation_failures,
        "scope": {
            "audited_rig_count": len(records),
            "artifact_count": len(artifact_records),
            "public_pass_artifact_count": sum(
                record["gate_status"] == "pass" for record in artifact_records
            ),
            "candidate_artifact_count": sum(
                record["gate_status"] != "pass" for record in artifact_records
            ),
            "encoder_called": False,
            "motion_visual_qa_claimed": False,
        },
        "counts": {
            "gate_status": dict(sorted(gate_counts.items())),
            "source_family": dict(sorted(source_counts.items())),
            "topology_family": dict(
                sorted(Counter(record["topology_family"] for record in records).items())
            ),
            "artifact_source_family": dict(sorted(artifact_source_counts.items())),
            "artifact_topology_family": dict(sorted(artifact_topology_counts.items())),
        },
        "blocked_sources": {
            "planetzoo": "native fixed per-rig transform/rest unavailable; processed sources contain per-clip yaw canonicalization",
            "motionstreamer272_human": "neutral SMPL rest is a non-encodable candidate because 272 omits shape",
        },
        "held_data_policy": {
            "static_per_rig_rest_metadata_allowed": True,
            "threshold_gain_fps_contact_heading_epsilon_tuning_allowed": False,
            "snake_train_shortage_resolved": False,
            "exact_dragon_train_eligible": False,
        },
        "full_conversion_allowed": False,
        "next_stage": "T05 Truebones-only motion prototypes; dynamic perspective visual QA remains mandatory",
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_npz_fsynced(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    with path.open("xb") as handle:
        np.savez_compressed(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())


def _publish_skeleton_generation(
    *,
    skeleton_output_root: Path,
    built_skeletons: list[BuiltSkeleton],
    rejection_records: list[dict[str, Any]],
    parent_manifest_generation_id: str,
    source_fk_manifest_transaction_sha256: str,
    source_fk_manifest_files: Mapping[str, Mapping[str, Any]],
    overwrite: bool,
) -> tuple[str, Path, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if skeleton_output_root.exists() and not skeleton_output_root.is_symlink():
        raise CanonicalSkeletonError(
            f"{skeleton_output_root} is a real path; managed skeleton output must be a symlink"
        )
    if skeleton_output_root.is_symlink() and not skeleton_output_root.exists():
        raise CanonicalSkeletonError(
            f"refusing to replace broken skeleton symlink {skeleton_output_root}"
        )
    if skeleton_output_root.exists() and not overwrite:
        raise FileExistsError(
            f"skeleton output exists; pass --overwrite after review: {skeleton_output_root}"
        )

    skeleton_output_root.parent.mkdir(parents=True, exist_ok=True)
    generation_root = skeleton_output_root.parent / SKELETON_GENERATION_DIRECTORY
    generation_root.mkdir(parents=True, exist_ok=True)
    generation_id = (
        _datetime.datetime.now(_datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + "-"
        + uuid.uuid4().hex[:12]
    )
    stage = Path(tempfile.mkdtemp(prefix=f".stage-{generation_id}-", dir=generation_root))
    final_generation = generation_root / generation_id
    try:
        for built in sorted(built_skeletons, key=lambda item: item.rig_id):
            if built.artifact_status == "pass":
                relative = Path(f"{built.rig_id}.npz")
            else:
                relative = Path("candidates") / f"{built.rig_id}.npz"
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            _write_npz_fsynced(target, _artifact_payload(built))
            built.artifact_relpath = str(
                Path(SKELETON_GENERATION_DIRECTORY) / generation_id / relative
            )
            built.artifact_sha256 = _sha256_file(target)
            built.artifact_size_bytes = target.stat().st_size

        qa_records = [
            _qa_record_from_built(built)
            for built in sorted(built_skeletons, key=lambda item: item.rig_id)
        ] + sorted(rejection_records, key=lambda record: record["rig_id"])
        qa_records.sort(key=lambda record: record["rig_id"])
        summary = _summarize_qa(qa_records)
        if not summary["expected_outcomes_satisfied"]:
            raise CanonicalSkeletonError(
                "refusing to publish an unexpected T04 outcome: "
                + _canonical_json(summary).decode("utf-8")
            )
        summary["skeleton_generation_id"] = generation_id
        summary["skeleton_generation_relpath"] = str(
            Path(SKELETON_GENERATION_DIRECTORY) / generation_id
        )
        summary["source_fk_manifest_generation_id"] = parent_manifest_generation_id
        summary["source_fk_manifest_transaction_sha256"] = (
            source_fk_manifest_transaction_sha256
        )
        summary["authoritative_resolution"] = (
            "resolve artifact_relpath and artifact_sha256 from active manifest; "
            "dataset/skeletons is compatibility-only"
        )

        qa_path = stage / CANONICAL_QA_FILENAME
        with qa_path.open("x", encoding="utf-8") as handle:
            for chunk in _jsonl_chunks(qa_records):
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        summary_path = stage / CANONICAL_SUMMARY_FILENAME
        with summary_path.open("x", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        file_paths = sorted(
            path for path in stage.rglob("*") if path.is_file()
        )
        file_records = {
            str(path.relative_to(stage)): {
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in file_paths
        }
        generation_record = {
            "qa_version": CANONICAL_SKELETON_VERSION,
            "generation_id": generation_id,
            "parent_manifest_generation_id": parent_manifest_generation_id,
            "source_fk_manifest_generation_id": parent_manifest_generation_id,
            "source_fk_manifest_transaction_sha256": (
                source_fk_manifest_transaction_sha256
            ),
            "source_fk_manifest_files": copy.deepcopy(source_fk_manifest_files),
            "publish_protocol": "immutable_skeleton_generation_then_authoritative_manifest_reference",
            "public_symlink_role": "compatibility_only_non_authoritative",
            "files": file_records,
        }
        generation_path = stage / SKELETON_GENERATION_FILENAME
        with generation_path.open("x", encoding="utf-8") as handle:
            json.dump(
                generation_record,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(stage)
        os.rename(stage, final_generation)
        _fsync_directory(generation_root)
        return generation_id, final_generation, qa_records, summary, generation_record
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _canonical_metadata_from_qa(record: Mapping[str, Any]) -> dict[str, Any]:
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


def _update_rigs(
    rigs: list[dict[str, Any]], qa_records: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    qa_by_rig = {record["rig_id"]: record for record in qa_records}
    result: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for original in rigs:
        record = copy.deepcopy(original)
        qa = qa_by_rig.get(record["rig_id"])
        if qa is not None:
            codes = set(record.get("reason_codes", []))
            record["canonical_skeleton"] = _canonical_metadata_from_qa(qa)
            if qa["artifact_relpath"] is not None:
                transform = qa["source_to_canonical"]
                record["source_to_canonical"] = {
                    "status": (
                        "numeric_fixed_per_rig_pass_t04"
                        if qa["gate_status"] == "pass"
                        else "numeric_fixed_per_rig_candidate_t04"
                    ),
                    "C": transform["C"],
                    "alpha": transform["alpha"],
                    "o": transform["o"],
                    "provenance": transform["provenance"],
                    "evidence_paths": list(
                        original.get("source_to_canonical", {}).get("evidence_paths", [])
                    ),
                }
                record["heading"] = {
                    "status": "static_anatomical_polarity_reviewed_t04",
                    "heading_carrier_joint_candidate": qa["heading"]["carrier_joint"],
                    "heading_carrier_name_candidate": record["joint_map"][
                        "btjd_joint_names"
                    ][qa["heading"]["carrier_joint"]],
                    "u_forward_local_candidate": qa["heading"]["u_forward_local"],
                    "polarity": "canonical_plus_z",
                    "provenance": qa["heading"]["provenance"],
                }
                record["unit"] = {
                    **record["unit"],
                    "length_unit_id": qa["unit"]["length_unit_id"],
                    "source_unit_to_meter": qa["unit"]["source_unit_to_meter"],
                    "canonical_scale_factor": qa["unit"]["canonical_scale_factor"],
                    "meter_claim": False,
                }
                codes.discard("SOURCE_TO_CANONICAL_UNREVIEWED")
                codes.discard("HEADING_PAYLOAD_UNREVIEWED")
            for code in qa["reason_codes"]:
                codes.add(code)
            record["reason_codes"] = sorted(codes)
            record["status"] = _status_from_codes(record["reason_codes"])
            payload = dict(record)
            payload.pop("rig_evidence_sha256", None)
            record["rig_evidence_sha256"] = _sha256_json(payload)
        hashes[record["rig_id"]] = record["rig_evidence_sha256"]
        result.append(record)
    return result, hashes


def _update_clips(
    clips: list[dict[str, Any]],
    updated_rigs: Mapping[str, dict[str, Any]],
    qa_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    qa_by_rig = {record["rig_id"]: record for record in qa_records}
    result: list[dict[str, Any]] = []
    for original in clips:
        qa = qa_by_rig.get(original["rig_id"])
        if qa is None:
            result.append(original)
            continue
        record = copy.deepcopy(original)
        rig = updated_rigs[record["rig_id"]]
        codes = set(record.get("reason_codes", []))
        record["canonical_skeleton"] = _canonical_metadata_from_qa(qa)
        if qa["artifact_relpath"] is not None:
            record["source_to_canonical"] = {
                "status": rig["source_to_canonical"]["status"],
                "rig_evidence_sha256": rig["rig_evidence_sha256"],
                "artifact_relpath": qa["artifact_relpath"],
                "artifact_sha256": qa["artifact_sha256"],
            }
            record["heading"] = {
                "status": rig["heading"]["status"],
                "carrier_joint_candidate": rig["heading"][
                    "heading_carrier_joint_candidate"
                ],
                "u_forward_local_candidate": rig["heading"][
                    "u_forward_local_candidate"
                ],
                "polarity": rig["heading"]["polarity"],
            }
            record["unit"] = rig["unit"]
            codes.discard("SOURCE_TO_CANONICAL_UNREVIEWED")
            codes.discard("HEADING_PAYLOAD_UNREVIEWED")
        for code in qa["reason_codes"]:
            codes.add(code)
        for key in ("rotation_provenance", "rest_pose"):
            if isinstance(record.get(key), dict):
                record[key]["rig_evidence_sha256"] = rig["rig_evidence_sha256"]
        record["reason_codes"] = sorted(codes)
        record["status"] = _status_from_codes(record["reason_codes"])
        if record["status"] == "reject":
            record["split_eligible_for_train_calibration"] = False
        source_fk = record.get("source_parser_fk")
        record["split_eligible_for_ktjd17_t04"] = bool(
            qa["gate_status"] == "pass"
            and isinstance(source_fk, dict)
            and source_fk.get("status") == "pass"
            and record.get("split") == "train"
        )
        result.append(record)
    return result


def _update_prototype_candidates(
    candidates: dict[str, Any], clips: Sequence[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = copy.deepcopy(candidates)
    clip_index = {record["clip_id"]: record for record in clips}
    gaps: list[dict[str, Any]] = []
    for family, payload in sorted(result["families"].items()):
        selected = list(payload.get("selected_train_candidates", []))
        eligible = [
            clip_id
            for clip_id in selected
            if clip_index[clip_id].get("split_eligible_for_ktjd17_t04")
        ]
        ineligible = [clip_id for clip_id in selected if clip_id not in set(eligible)]
        required = int(payload["required_train_clips"])
        payload["canonical_skeleton_t04"] = {
            "status": "available" if len(eligible) >= required else "shortage",
            "eligible_selected_train_clips": eligible,
            "eligible_count": len(eligible),
            "ineligible_selected_train_clips": ineligible,
            "required_train_clips": required,
            "shortage": max(0, required - len(eligible)),
            "selection_replaced": False,
        }
        if len(eligible) < required:
            gaps.append(
                {
                    "manifest_version": INVENTORY_VERSION,
                    "gap_id": f"prototype_t04_canonical_shortage:{family}",
                    "family": family,
                    "status": "gap",
                    "reason_codes": ["PROTOTYPE_TRAIN_SHORTAGE"],
                    "required_train_clips": required,
                    "canonical_skeleton_eligible_selected_train_clips": len(eligible),
                    "shortage": required - len(eligible),
                    "evidence": "T04 fixed-rest/canonical-skeleton gate; no replacement clips were selected in T04",
                }
            )
    return result, gaps


def _replace_compatibility_symlink(output_root: Path, generation_id: str) -> None:
    relative_target = Path(SKELETON_GENERATION_DIRECTORY) / generation_id
    link_tmp = output_root.parent / f".{output_root.name}.{generation_id}.tmp"
    try:
        os.symlink(str(relative_target), link_tmp)
        os.replace(link_tmp, output_root)
        _fsync_directory(output_root.parent)
    finally:
        if link_tmp.is_symlink():
            link_tmp.unlink()


def run_canonical_skeleton_audit(config: CanonicalSkeletonConfig) -> dict[str, Any]:
    """Build T04 artifacts and publish a manifest-authoritative generation."""
    config = config.resolved()
    root = config.manifest_root
    for name in REQUIRED_PARENT_MANIFEST_FILES:
        if not (root / name).is_file():
            raise CanonicalSkeletonError(f"missing T03 artifact: {root / name}")
    parent_transaction = _load_json(root / "inventory_generation.json")
    parent_generation_id, source_fk_manifest_files = _validate_direct_t03_parent(
        root, parent_transaction
    )
    source_fk_manifest_transaction_sha256 = _sha256_file(
        root / "inventory_generation.json"
    )
    inventory_summary = _load_json(root / "inventory_summary.json")
    try:
        dataset_root = Path(inventory_summary["config"]["dataset_root"])
        conditioning_catalog = load_conditioning_catalog(
            dataset_root / "cond.npy",
            expected_active_sha256=str(inventory_summary["cond_sha256"]),
        )
    except (KeyError, TypeError, TruebonesFixedRigError) as exc:
        raise CanonicalSkeletonError(
            f"cannot establish Truebones fixed-rig conditioning authority: {exc}"
        ) from exc

    clips = _load_jsonl(root / "clips.jsonl")
    rigs_list = _load_jsonl(root / "rigs.jsonl")
    qa_t03 = _load_jsonl(root / "source_fk_qa.jsonl")
    rigs = {record["rig_id"]: record for record in rigs_list}
    clip_index = {record["clip_id"]: record for record in clips}
    if len(rigs) != len(rigs_list) or len(clip_index) != len(clips):
        raise CanonicalSkeletonError("duplicate rig or clip id in parent manifest")

    qa_by_rig: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in qa_t03:
        qa_by_rig[record["rig_id"]].append(record)
    source_scope = Counter(rigs[rig_id]["source_family"] for rig_id in qa_by_rig)
    if len(qa_by_rig) != 58 or source_scope != Counter(
        {"truebones": 31, "planetzoo": 26, "motionstreamer272": 1}
    ):
        raise CanonicalSkeletonError(
            f"T04 scope drifted: rigs={len(qa_by_rig)}, sources={dict(source_scope)}"
        )
    for rig_id, records in qa_by_rig.items():
        if any(record.get("gate_status") != "pass" for record in records):
            raise CanonicalSkeletonError(
                f"{rig_id}: T04 requires every existing T03 QA record to pass"
            )
        if rigs[rig_id]["source_family"] == "truebones" and any(
            not isinstance(record.get("fixed_rig"), dict)
            or record["fixed_rig"].get("status") != "pass"
            for record in records
        ):
            raise CanonicalSkeletonError(
                f"{rig_id}: T04 requires the T03 fixed-rig Truebones contract"
            )

    built_skeletons: list[BuiltSkeleton] = []
    rejection_records: list[dict[str, Any]] = []
    rest_cache: dict[str, ParsedBvhMotion] = {}
    for index, rig_id in enumerate(sorted(qa_by_rig), start=1):
        rig = rigs[rig_id]
        representative_qa = sorted(
            qa_by_rig[rig_id], key=lambda record: record["clip_id"]
        )[0]
        representative_clip_id = representative_qa["clip_id"]
        clip = clip_index[representative_clip_id]
        if rig["source_family"] == "planetzoo":
            rejection_records.append(
                _rejection_record(
                    rig,
                    representative_clip_id,
                    "CANONICAL_TRANSFORM_PROVENANCE_INVALID",
                    "local PlanetZoo BVHs already contain per-action initial-yaw canonicalization and expose hierarchy-only rest; native fixed per-rig C/rest is unavailable",
                )
            )
        else:
            try:
                built_skeletons.append(
                    _build_one(
                        rig,
                        clip,
                        rest_cache=rest_cache,
                        conditioning_catalog=conditioning_catalog,
                    )
                )
            except (
                CanonicalSkeletonError,
                SourceParserError,
                TruebonesFixedRigError,
            ) as exc:
                rejection_records.append(
                    _rejection_record(
                        rig,
                        representative_clip_id,
                        "CANONICAL_SKELETON_DERIVATION_FAILED",
                        f"{type(exc).__name__}: {exc}",
                    )
                )
        if index % 10 == 0 or index == len(qa_by_rig):
            print(f"[canonical-skeleton] audited {index}/{len(qa_by_rig)} rigs", flush=True)

    generation_id, generation_path, qa_records, summary, skeleton_transaction = (
        _publish_skeleton_generation(
            skeleton_output_root=config.skeleton_output_root,
            built_skeletons=built_skeletons,
            rejection_records=rejection_records,
            parent_manifest_generation_id=parent_generation_id,
            source_fk_manifest_transaction_sha256=(
                source_fk_manifest_transaction_sha256
            ),
            source_fk_manifest_files=source_fk_manifest_files,
            overwrite=config.overwrite,
        )
    )

    updated_rigs_list, _ = _update_rigs(rigs_list, qa_records)
    updated_rigs = {record["rig_id"]: record for record in updated_rigs_list}
    updated_clips = _update_clips(clips, updated_rigs, qa_records)
    candidates = _load_json(root / "prototype_candidates.json")
    updated_candidates, t04_gaps = _update_prototype_candidates(
        candidates, updated_clips
    )
    existing_gaps = _load_jsonl(root / "prototype_gaps.jsonl")
    existing_gap_ids = {record["gap_id"] for record in existing_gaps}
    updated_gaps = existing_gaps + [
        record for record in t04_gaps if record["gap_id"] not in existing_gap_ids
    ]

    inventory_summary["canonical_skeleton_audit"] = {
        "qa_version": CANONICAL_SKELETON_VERSION,
        "status": summary["status"],
        "skeleton_generation_id": generation_id,
        "skeleton_generation_relpath": summary["skeleton_generation_relpath"],
        "gate_status_counts": summary["counts"]["gate_status"],
        "artifact_count": summary["scope"]["artifact_count"],
        "public_pass_artifact_count": summary["scope"][
            "public_pass_artifact_count"
        ],
        "authoritative_resolution": summary["authoritative_resolution"],
        "full_conversion_allowed": False,
        "truebones_conditioning_authority": conditioning_catalog.authority_record(),
    }
    inventory_summary["fresh_counts"]["clip_status_counts"] = dict(
        sorted(Counter(record["status"] for record in updated_clips).items())
    )
    inventory_summary["fresh_counts"]["rig_status_counts"] = dict(
        sorted(Counter(record["status"] for record in updated_rigs_list).items())
    )
    inventory_summary["prototype_families"] = updated_candidates["families"]
    inventory_summary["prototype_gap_records"] = updated_gaps

    reason_table = _load_json(root / "inventory_reason_codes.json")
    reason_table["codes"] = REASON_CODES
    manifest_stage_record = {
        "qa_version": CANONICAL_SKELETON_VERSION,
        "manifest_version": INVENTORY_VERSION,
        "parent_manifest_generation_id": parent_generation_id,
        "parent_manifest_files": parent_transaction.get("files"),
        "source_fk_manifest_generation_id": parent_generation_id,
        "source_fk_manifest_transaction_sha256": (
            source_fk_manifest_transaction_sha256
        ),
        "source_fk_manifest_files": source_fk_manifest_files,
        "skeleton_generation_id": generation_id,
        "skeleton_generation_relpath": summary["skeleton_generation_relpath"],
        "skeleton_generation_transaction_sha256": _sha256_file(
            generation_path / SKELETON_GENERATION_FILENAME
        ),
        "authoritative_resolution": summary["authoritative_resolution"],
        "compatibility_symlink": str(config.skeleton_output_root),
        "compatibility_symlink_authoritative": False,
        "encoder_called": False,
        "motion_visual_qa_claimed": False,
        "truebones_conditioning_authority": conditioning_catalog.authority_record(),
        "next_stage": "T05 Truebones-only KTJD motion prototypes",
    }

    passthrough_names = (
        "source_fk_qa.jsonl",
        "source_fk_summary.json",
        "source_fk_generation.json",
    )
    outputs: dict[str, str | Iterable[str]] = {
        "clips.jsonl": _jsonl_chunks(updated_clips),
        "rigs.jsonl": _jsonl_chunks(updated_rigs_list),
        "inventory_summary.json": json.dumps(
            inventory_summary,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        "inventory_reason_codes.json": json.dumps(
            reason_table,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        "prototype_candidates.json": json.dumps(
            updated_candidates,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        "prototype_gaps.jsonl": _jsonl_chunks(updated_gaps),
        CANONICAL_QA_FILENAME: _jsonl_chunks(qa_records),
        CANONICAL_SUMMARY_FILENAME: json.dumps(
            summary,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        MANIFEST_STAGE_FILENAME: json.dumps(
            manifest_stage_record,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
    }
    for name in passthrough_names:
        outputs[name] = (root / name).read_text(encoding="utf-8")

    manifest_transaction = _write_transaction(
        config.manifest_output_root,
        outputs,
        overwrite=config.overwrite,
    )
    # This link is intentionally switched only after the authoritative manifest
    # exists.  A crash here leaves exact immutable manifest references usable.
    _replace_compatibility_symlink(config.skeleton_output_root, generation_id)

    result = copy.deepcopy(summary)
    result["manifest_generation_id"] = manifest_transaction["generation_id"]
    result["skeleton_generation_id"] = generation_id
    result["skeleton_generation_transaction"] = skeleton_transaction
    result["compatibility_symlink_target"] = os.readlink(
        config.skeleton_output_root
    )
    return result


def resolve_manifest_skeleton_artifact(
    dataset_root: Path, canonical_metadata: Mapping[str, Any]
) -> Path:
    """Resolve and hash-check an artifact without trusting dataset/skeletons."""
    relative = canonical_metadata.get("artifact_relpath")
    expected_hash = canonical_metadata.get("artifact_sha256")
    if not isinstance(relative, str) or not relative or not isinstance(expected_hash, str):
        raise CanonicalSkeletonError("manifest has no immutable skeleton path/hash")
    root = dataset_root.expanduser().resolve()
    candidate = (root / relative).resolve()
    generation_root = (root / SKELETON_GENERATION_DIRECTORY).resolve()
    if generation_root not in candidate.parents:
        raise CanonicalSkeletonError(
            f"skeleton path escapes immutable generation root: {candidate}"
        )
    if not candidate.is_file():
        raise CanonicalSkeletonError(f"skeleton artifact is missing: {candidate}")
    actual = _sha256_file(candidate)
    if actual != expected_hash:
        raise CanonicalSkeletonError(
            f"skeleton hash mismatch for {candidate}: {actual} != {expected_hash}"
        )
    return candidate
