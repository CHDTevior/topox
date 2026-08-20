"""Independent live validation for KTJD-17 T03 source-FK artifacts.

This module deliberately does not import the T03 producer, its BVH parser, its
FK implementation, or its constants.  BVH Euler matrices are decoded with
SciPy and positions are evaluated with independent homogeneous transforms.
"""

from __future__ import annotations

import dataclasses
import datetime as _datetime
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


EXPECTED_INVENTORY_VERSION = "ktjd17-raw-inventory-v1"
EXPECTED_QA_VERSION = "ktjd17-source-fk-v2"
EXPECTED_TRANSACTION_FILES = {
    "clips.jsonl",
    "rigs.jsonl",
    "inventory_summary.json",
    "inventory_reason_codes.json",
    "prototype_candidates.json",
    "prototype_gaps.jsonl",
    "source_fk_qa.jsonl",
    "source_fk_summary.json",
    "source_fk_generation.json",
}
CANONICAL_SKELETON_TRANSACTION_FILES = {
    "canonical_skeleton_qa.jsonl",
    "canonical_skeleton_summary.json",
    "canonical_skeleton_generation.json",
}
EXPECTED_THRESHOLDS = {
    "motionstreamer272": 1e-6,
    "planetzoo": 1e-10,
    "truebones": 1e-10,
}
EXPECTED_ACTIVE_COND_SHA256 = "161795a6507e24c2908f3837c9c999f19d411ecefd7943852f308942d8949bfb"
EXPECTED_LEGACY_COND_SHA256 = "9dad7c833534edf90fa295e837d1c5e021306b9857aa40d8c3f88c17e5c33d02"
TRUEBONES_MEAN_EDGE_TARGET = 0.2092142857142857
ROTATION_QUANTIZATION_STEP = 1e-8
RIGID_EDGE_MAX_NORM = 1e-4

# Independent frozen anatomy truth.  This intentionally duplicates the
# producer table so an anatomically self-consistent producer edit fails here.
EXPECTED_TRUEBONES_FORWARD_SPECS: dict[str, tuple[str, tuple[str, ...]]] = {
    "Alligator": ("lateral_pairs", ("R_momo", "L_momo", "R_hiji", "L_hiji")),
    "Anaconda": ("root_to_head", ("Hips", "BN_Tone_04")),
    "Bat": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_R_UpperArm_01", "BN_L_UpperArm_01")),
    "Bird": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_Forearm_R_01", "BN_Forearm_L_01")),
    "Buffalo": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_Clavicle", "Bip01_L_Clavicle")),
    "Buzzard": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_Wing_R_02", "BN_Wing_L_02")),
    "Cat": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_Clavicle", "Bip01_L_Clavicle")),
    "Chicken": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_Finger_R_01", "BN_Finger_L_01")),
    "Coyote": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_Clavicle", "Bip01_L_Clavicle")),
    "Crocodile": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm")),
    "Dragon": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm")),
    "Eagle": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_Wing_R_02", "BN_Wing_L_02")),
    "Flamingo": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_Forearm_R_02", "BN_Forearm_L_02")),
    "Fox": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm")),
    "Gazelle": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm")),
    "Hamster": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm")),
    "HermitCrab": ("lateral_pairs", ("BN_Leg_R_09", "BN_Leg_L_09", "BN_Crab_pincers_R_02", "BN_Crab_pincers_L_02")),
    "Hippopotamus": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_Clavicle", "Bip01_L_Clavicle")),
    "Horse": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm")),
    "Hound": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm")),
    "KingCobra": ("root_to_head", ("Hips", "BN_Tongue_02")),
    "Lion": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm")),
    "Lynx": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm")),
    "Mammoth": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm")),
    "Ostrich": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_Forearm_R_02", "BN_Forearm_L_02")),
    "Parrot": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_Wing_R_02", "BN_Wing_L_02")),
    "Parrot2": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_Wing_R_02", "BN_Wing_L_02")),
    "Pteranodon": ("lateral_pairs", ("jt_Thigh_R", "jt_Thigh_L", "jt_Elbow_R", "jt_Elbow_L")),
    "Scorpion": ("lateral_pairs", ("Bip01_R_Thigh_4", "Bip01_L_Thigh1_4", "Bip01_R_Forearm", "Bip01_L_Forearm")),
    "SpiderG": ("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm")),
    "Tukan": ("lateral_pairs", ("R_momo", "L_momo", "R_kata", "L_kata")),
}


class SourceFkValidationError(RuntimeError):
    """Materialized T03 evidence disagrees with live independent recovery."""


@dataclasses.dataclass(frozen=True)
class _Bvh:
    path: str
    fps: float
    names: tuple[str, ...]
    parents: np.ndarray
    node_kinds: tuple[str, ...]
    offsets: np.ndarray
    channels: tuple[tuple[str, ...], ...]
    local_positions: np.ndarray
    local_rotations: np.ndarray
    global_positions: np.ndarray
    global_rotations: np.ndarray


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise SourceFkValidationError(f"cannot read JSON {path}: {exc}") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise SourceFkValidationError(
                        f"{path}:{line_number}: blank JSONL record"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise SourceFkValidationError(
                        f"{path}:{line_number}: JSONL row is not an object"
                    )
                result.append(value)
    except SourceFkValidationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise SourceFkValidationError(f"cannot read JSONL {path}: {exc}") from exc
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise SourceFkValidationError(
            f"{label} mismatch: {actual!r} != {expected!r}"
        )


def _require_close(label: str, actual: float, expected: float) -> None:
    # SciPy Euler evaluation and the producer's explicit axis-matrix product
    # accumulate round-off differently on deep (100+ joint) chains.  The
    # This tolerance is used only to compare reported round-off magnitudes;
    # pass/fail is recomputed independently against the exact family threshold.
    if not math.isclose(actual, expected, rel_tol=2e-6, abs_tol=1e-10):
        raise SourceFkValidationError(
            f"{label} mismatch: {actual:.17g} != {expected:.17g}"
        )


def _validate_transaction(root: Path) -> dict[str, Any]:
    transaction = _load_json(root / "inventory_generation.json")
    _require_equal(
        "transaction manifest version",
        transaction.get("manifest_version"),
        EXPECTED_INVENTORY_VERSION,
    )
    _require_equal(
        "transaction publish protocol",
        transaction.get("publish_protocol"),
        "immutable_generation_atomic_symlink_replace",
    )
    files = transaction.get("files")
    allowed_file_sets = {
        frozenset(EXPECTED_TRANSACTION_FILES),
        frozenset(EXPECTED_TRANSACTION_FILES | CANONICAL_SKELETON_TRANSACTION_FILES),
    }
    if not isinstance(files, dict) or frozenset(files) not in allowed_file_sets:
        raise SourceFkValidationError("T03 transaction file set is incomplete")
    for name in sorted(files):
        path = root / name
        if not path.is_file():
            raise SourceFkValidationError(f"transaction file is absent: {path}")
        _require_equal(
            f"transaction size {name}", path.stat().st_size, files[name]["size_bytes"]
        )
        _require_equal(
            f"transaction sha256 {name}", _sha256_file(path), files[name]["sha256"]
        )
    return transaction


_DECLARATION_RE = re.compile(r"^(ROOT|JOINT)\s+(.+?)\s*$", re.IGNORECASE)
_END_SITE_RE = re.compile(
    r"^End\s+Site(?:\s+#name:\s*(.+?))?\s*$", re.IGNORECASE
)


def _parse_bvh_independent(path: str | Path) -> _Bvh:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise SourceFkValidationError(f"BVH source is missing: {source}")
    lines = source.read_text(encoding="utf-8", errors="strict").splitlines()
    mutable: list[dict[str, Any]] = []
    stack: list[int] = []
    pending: int | None = None
    unnamed_counts: dict[int, int] = {}
    motion_line: int | None = None
    for line_index, raw in enumerate(lines):
        text = raw.strip()
        if not text or text.upper() == "HIERARCHY":
            continue
        if text.upper() == "MOTION":
            if pending is not None or stack:
                raise SourceFkValidationError(f"unclosed BVH hierarchy: {source}")
            motion_line = line_index
            break
        declaration = _DECLARATION_RE.match(text)
        if declaration:
            kind, name = declaration.groups()
            if pending is not None:
                raise SourceFkValidationError(f"missing opening brace in {source}")
            if kind.upper() == "ROOT" and mutable:
                raise SourceFkValidationError(f"multiple BVH roots in {source}")
            if kind.upper() == "JOINT" and not stack:
                raise SourceFkValidationError(f"parentless BVH joint in {source}")
            mutable.append(
                {
                    "name": name.strip(),
                    "parent": -1 if kind.upper() == "ROOT" else stack[-1],
                    "node_kind": "joint",
                    "offset": None,
                    "channels": None,
                }
            )
            pending = len(mutable) - 1
            continue
        end_site = _END_SITE_RE.match(text)
        if end_site:
            if pending is not None or not stack:
                raise SourceFkValidationError(f"orphan BVH End Site in {source}")
            parent = stack[-1]
            name = end_site.group(1)
            if name is None:
                count = unnamed_counts.get(parent, 0)
                unnamed_counts[parent] = count + 1
                name = f"{mutable[parent]['name']}__unnamed_end_site_{count}"
            mutable.append(
                {
                    "name": name.strip(),
                    "parent": parent,
                    "node_kind": "end_site",
                    "offset": None,
                    "channels": (),
                }
            )
            pending = len(mutable) - 1
            continue
        if text == "{":
            if pending is None:
                raise SourceFkValidationError(f"unexpected opening brace in {source}")
            stack.append(pending)
            pending = None
            continue
        if text == "}":
            if pending is not None or not stack:
                raise SourceFkValidationError(f"unexpected closing brace in {source}")
            stack.pop()
            continue
        if text.upper().startswith("OFFSET "):
            if not stack:
                raise SourceFkValidationError(f"BVH OFFSET outside a node: {source}")
            fields = text.split()
            if len(fields) != 4:
                raise SourceFkValidationError(f"malformed BVH OFFSET: {source}")
            mutable[stack[-1]]["offset"] = tuple(float(value) for value in fields[1:])
            continue
        if text.upper().startswith("CHANNELS "):
            if not stack:
                raise SourceFkValidationError(f"BVH CHANNELS outside a node: {source}")
            fields = text.split()
            count = int(fields[1])
            if len(fields) != count + 2:
                raise SourceFkValidationError(f"malformed BVH CHANNELS: {source}")
            mutable[stack[-1]]["channels"] = tuple(fields[2:])
            continue
        raise SourceFkValidationError(f"unrecognized BVH hierarchy line {text!r}")
    if motion_line is None or not mutable:
        raise SourceFkValidationError(f"BVH has no hierarchy/motion marker: {source}")
    if any(item["offset"] is None or item["channels"] is None for item in mutable):
        raise SourceFkValidationError(f"incomplete BVH hierarchy metadata: {source}")

    frames: int | None = None
    frame_time: float | None = None
    data_start: int | None = None
    for line_index in range(motion_line + 1, len(lines)):
        text = lines[line_index].strip()
        if not text:
            continue
        if text.lower().startswith("frames:"):
            frames = int(text.split(":", 1)[1].strip())
            continue
        if text.lower().startswith("frame time:"):
            frame_time = float(text.split(":", 1)[1].strip())
            data_start = line_index + 1
            break
    if frames is None or frames <= 0 or frame_time is None or frame_time <= 0:
        raise SourceFkValidationError(f"invalid BVH timing header: {source}")
    channels = tuple(tuple(item["channels"]) for item in mutable)
    channel_count = sum(len(item) for item in channels)
    tokens = " ".join(lines[data_start:]).split()
    expected = frames * channel_count
    if len(tokens) != expected:
        raise SourceFkValidationError(
            f"BVH numeric count mismatch: {len(tokens)} != {expected}: {source}"
        )
    try:
        values = np.asarray([float(token) for token in tokens], dtype=np.float64)
    except ValueError as exc:
        raise SourceFkValidationError(f"invalid BVH numeric token: {source}") from exc
    values = values.reshape(frames, channel_count)
    if not np.isfinite(values).all():
        raise SourceFkValidationError(f"non-finite BVH numeric payload: {source}")

    offsets = np.asarray([item["offset"] for item in mutable], dtype=np.float64)
    parents = np.asarray([item["parent"] for item in mutable], dtype=np.int64)
    local_positions = np.broadcast_to(offsets, (frames,) + offsets.shape).copy()
    local_rotations = np.broadcast_to(
        np.eye(3, dtype=np.float64), (frames, len(mutable), 3, 3)
    ).copy()
    cursor = 0
    for joint, joint_channels in enumerate(channels):
        block = values[:, cursor : cursor + len(joint_channels)]
        cursor += len(joint_channels)
        position_items: list[tuple[int, int]] = []
        rotation_items: list[tuple[str, int]] = []
        for channel_index, channel in enumerate(joint_channels):
            lowered = channel.lower()
            if lowered.endswith("position") and lowered[0] in "xyz":
                position_items.append(("xyz".index(lowered[0]), channel_index))
            elif lowered.endswith("rotation") and lowered[0] in "xyz":
                rotation_items.append((lowered[0], channel_index))
            else:
                raise SourceFkValidationError(
                    f"unsupported BVH channel {channel!r}: {source}"
                )
        if position_items:
            if len(position_items) != 3 or sorted(x for x, _ in position_items) != [0, 1, 2]:
                raise SourceFkValidationError(f"partial BVH translation channels: {source}")
            local_positions[:, joint] = 0.0
            for axis, channel_index in position_items:
                local_positions[:, joint, axis] = block[:, channel_index]
        if rotation_items:
            sequence = "".join(axis.upper() for axis, _ in rotation_items)
            angles = np.stack(
                [block[:, channel_index] for _, channel_index in rotation_items], axis=-1
            )
            local_rotations[:, joint] = Rotation.from_euler(
                sequence, angles, degrees=True
            ).as_matrix()

    global_positions, global_rotations = _homogeneous_fk(
        parents, local_positions, local_rotations
    )
    return _Bvh(
        path=str(source),
        fps=1.0 / frame_time,
        names=tuple(item["name"] for item in mutable),
        parents=parents,
        node_kinds=tuple(item["node_kind"] for item in mutable),
        offsets=offsets,
        channels=channels,
        local_positions=local_positions,
        local_rotations=local_rotations,
        global_positions=global_positions,
        global_rotations=global_rotations,
    )


def _homogeneous_fk(
    parents: np.ndarray,
    local_positions: np.ndarray,
    local_rotations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frame_count, joint_count = local_positions.shape[:2]
    local = np.zeros((frame_count, joint_count, 4, 4), dtype=np.float64)
    local[..., :3, :3] = local_rotations
    local[..., :3, 3] = local_positions
    local[..., 3, 3] = 1.0
    world = np.empty_like(local)
    world[:, 0] = local[:, 0]
    for child in range(1, joint_count):
        parent = int(parents[child])
        if not 0 <= parent < child:
            raise SourceFkValidationError(
                f"invalid parent-before-child tree at {child}: {parent}"
            )
        world[:, child] = world[:, parent] @ local[:, child]
    return world[..., :3, 3].copy(), world[..., :3, :3].copy()


def _reroot_retained(
    positions: np.ndarray,
    rotations: np.ndarray,
    source_indices: np.ndarray,
    retained_parents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_positions = positions[:, source_indices].copy()
    source_rotations = rotations[:, source_indices].copy()
    local_positions = np.empty_like(source_positions)
    local_rotations = np.empty_like(source_rotations)
    local_positions[:, 0] = source_positions[:, 0]
    local_rotations[:, 0] = source_rotations[:, 0]
    for child in range(1, len(source_indices)):
        parent = int(retained_parents[child])
        inverse = np.swapaxes(source_rotations[:, parent], -1, -2)
        local_positions[:, child] = np.einsum(
            "tij,tj->ti",
            inverse,
            source_positions[:, child] - source_positions[:, parent],
        )
        local_rotations[:, child] = inverse @ source_rotations[:, child]
    fk_positions, _ = _homogeneous_fk(
        retained_parents, local_positions, local_rotations
    )
    return source_positions, fk_positions, local_rotations


def _aabb_diagonal(points: np.ndarray) -> float:
    scale = float(np.linalg.norm(np.ptp(np.asarray(points, dtype=np.float64), axis=0)))
    if not math.isfinite(scale) or scale <= 0:
        raise SourceFkValidationError(f"invalid rest AABB diagonal: {scale}")
    return scale


def _bvh_live_metrics(
    clip: dict[str, Any], rig: dict[str, Any]
) -> dict[str, float]:
    motion = _parse_bvh_independent(clip["source"]["path"])
    mapping = np.asarray(rig["joint_map"]["btjd_to_source"], dtype=np.int64)
    retained_parents = np.asarray(
        rig["joint_map"]["btjd_parents"], dtype=np.int64
    )
    if mapping.shape != retained_parents.shape or np.any(mapping < 0):
        raise SourceFkValidationError(f"invalid retained map for {clip['clip_id']}")
    start, end = (int(value) for value in clip["source"]["slice_frames"])
    source_positions, fk_positions, _ = _reroot_retained(
        motion.global_positions[start:end],
        motion.global_rotations[start:end],
        mapping,
        retained_parents,
    )

    rest = _parse_bvh_independent(rig["rest_pose"]["source_path"])
    if clip["source"]["family"] == "planetzoo":
        rest_local_positions = np.broadcast_to(
            rest.offsets, (1,) + rest.offsets.shape
        ).copy()
        rest_local_rotations = np.broadcast_to(
            np.eye(3, dtype=np.float64), (1, len(rest.names), 3, 3)
        ).copy()
        rest_positions, rest_rotations = _homogeneous_fk(
            rest.parents, rest_local_positions, rest_local_rotations
        )
    else:
        rest_positions = rest.global_positions[:1]
        rest_rotations = rest.global_rotations[:1]
    rest_mapping = np.asarray(rig["joint_map"]["btjd_to_source"], dtype=np.int64)
    mapped_rest = rest_positions[0, rest_mapping]
    scale = _aabb_diagonal(mapped_rest)
    return _error_metrics(source_positions, fk_positions, scale)


def _decode_row6d(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    first = source[..., :3]
    second = source[..., 3:]
    first = first / np.linalg.norm(first, axis=-1, keepdims=True)
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second = second / np.linalg.norm(second, axis=-1, keepdims=True)
    third = np.cross(first, second)
    result = np.stack((first, second, third), axis=-2)
    if not np.isfinite(result).all():
        raise SourceFkValidationError("non-finite independent row-6D decode")
    return result


def _load_neutral_rest(path: str | Path, parents: np.ndarray) -> np.ndarray:
    try:
        with np.load(path, allow_pickle=True) as model:
            vertices = np.asarray(model["v_template"], dtype=np.float64)
            regressor_raw = model["J_regressor"]
            if hasattr(regressor_raw, "toarray"):
                regressor_raw = regressor_raw.toarray()
            regressor = np.asarray(regressor_raw, dtype=np.float64)
            kintree = np.asarray(model["kintree_table"])[0, : len(parents)].astype(
                np.int64
            )
    except Exception as exc:  # noqa: BLE001
        raise SourceFkValidationError(f"cannot independently load neutral rest: {exc}") from exc
    kintree[0] = -1
    if not np.array_equal(kintree, parents):
        raise SourceFkValidationError("neutral SMPL tree differs from manifest tree")
    return np.asarray(regressor @ vertices, dtype=np.float64)[: len(parents)]


def _human_live_metrics(
    clip: dict[str, Any], rig: dict[str, Any]
) -> dict[str, float]:
    data = np.asarray(
        np.load(clip["source"]["path"], allow_pickle=False), dtype=np.float64
    )
    if data.ndim != 2 or data.shape[1] != 272 or data.shape[0] <= 0:
        raise SourceFkValidationError(f"invalid MotionStreamer272 shape: {data.shape}")
    frame_count = data.shape[0]
    heading_delta = _decode_row6d(data[:, 2:8])
    heading = np.empty_like(heading_delta)
    heading[0] = heading_delta[0]
    for frame in range(1, frame_count):
        heading[frame] = heading_delta[frame] @ heading[frame - 1]
    inverse_heading = np.swapaxes(heading, -1, -2)
    positions = data[:, 8:74].reshape(frame_count, 22, 3)
    positions = np.einsum("tij,tkj->tki", inverse_heading, positions)
    root_velocity = np.zeros((frame_count, 3), dtype=np.float64)
    root_velocity[:, 0] = data[:, 0]
    root_velocity[:, 2] = data[:, 1]
    if frame_count > 1:
        root_velocity[1:] = np.einsum(
            "tij,tj->ti", inverse_heading[:-1], root_velocity[1:]
        )
    root_translation = np.cumsum(root_velocity, axis=0)
    positions[..., 0] += root_translation[:, None, 0]
    positions[..., 2] += root_translation[:, None, 2]

    local_rotations = _decode_row6d(data[:, 140:272].reshape(frame_count, 22, 6))
    local_rotations[:, 0] = inverse_heading @ local_rotations[:, 0]
    parents = np.asarray(rig["joint_map"]["btjd_parents"], dtype=np.int64)
    global_rotations = np.empty_like(local_rotations)
    global_rotations[:, 0] = local_rotations[:, 0]
    for child in range(1, 22):
        global_rotations[:, child] = (
            global_rotations[:, int(parents[child])] @ local_rotations[:, child]
        )
    observed_offsets = np.zeros_like(positions)
    for child in range(1, 22):
        parent = int(parents[child])
        observed_offsets[:, child] = np.einsum(
            "tij,tj->ti",
            np.swapaxes(global_rotations[:, parent], -1, -2),
            positions[:, child] - positions[:, parent],
        )
    shaped_offsets = np.median(observed_offsets, axis=0)
    local_positions = np.broadcast_to(shaped_offsets, positions.shape).copy()
    local_positions[:, 0] = positions[:, 0]
    fk_positions, _ = _homogeneous_fk(parents, local_positions, local_rotations)
    neutral = _load_neutral_rest(rig["rest_pose"]["source_path"], parents)
    return _error_metrics(positions, fk_positions, _aabb_diagonal(neutral))


def _error_metrics(
    source_positions: np.ndarray, fk_positions: np.ndarray, scale: float
) -> dict[str, float]:
    errors = np.linalg.norm(fk_positions - source_positions, axis=-1)
    return {
        "s_rig": scale,
        "mpjpe_abs": float(np.mean(errors)),
        "p99_abs": float(np.percentile(errors, 99)),
        "max_abs": float(np.max(errors)),
        "source_parser_fk_error_norm": float(np.mean(errors) / scale),
        "source_parser_fk_p99_norm": float(np.percentile(errors, 99) / scale),
        "source_parser_fk_max_norm": float(np.max(errors) / scale),
    }


def _array_record(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _conditioning_payload_sha256(entry: dict[str, Any]) -> str:
    payload = {
        "parents": _array_record(entry["parents"]),
        "offsets": _array_record(entry["offsets"]),
        "tpos_first_frame": _array_record(entry["tpos_first_frame"]),
        "joints_names": [str(value) for value in entry["joints_names"]],
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_conditioning_independent(
    inventory_summary: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        active = Path(inventory_summary["config"]["dataset_root"]) / "cond.npy"
        expected_manifest_sha = str(inventory_summary["cond_sha256"])
    except (KeyError, TypeError) as exc:
        raise SourceFkValidationError("inventory lacks cond authority") from exc
    active = active.expanduser().resolve()
    legacy = active.parent.parent.joinpath("anytop_truebones", "cond.npy").resolve()
    active_sha = _sha256_file(active)
    legacy_sha = _sha256_file(legacy)
    _require_equal("active cond manifest hash", active_sha, expected_manifest_sha)
    _require_equal("active cond frozen hash", active_sha, EXPECTED_ACTIVE_COND_SHA256)
    _require_equal("legacy Truebones cond hash", legacy_sha, EXPECTED_LEGACY_COND_SHA256)
    try:
        active_entries = np.load(active, allow_pickle=True).item()
        legacy_entries = np.load(legacy, allow_pickle=True).item()
    except Exception as exc:  # noqa: BLE001
        raise SourceFkValidationError(f"cannot load cond authority: {exc}") from exc
    if not isinstance(active_entries, dict) or not isinstance(legacy_entries, dict):
        raise SourceFkValidationError("cond authority is not a dict")
    missing = sorted(set(legacy_entries) - set(active_entries))
    if missing:
        raise SourceFkValidationError(f"active cond misses legacy rigs: {missing[:10]}")
    for rig_id in sorted(legacy_entries):
        left = legacy_entries[rig_id]
        right = active_entries[rig_id]
        if [str(x) for x in left["joints_names"]] != [
            str(x) for x in right["joints_names"]
        ]:
            raise SourceFkValidationError(f"{rig_id} cond names drifted")
        for key in ("parents", "offsets", "tpos_first_frame"):
            if _array_record(left[key]) != _array_record(right[key]):
                raise SourceFkValidationError(f"{rig_id} cond {key} drifted")
    authority = {
        "authority_kind": "current_btjd_fixed_physical_rig_geometry_only",
        "active_cond_path": str(active),
        "active_cond_sha256": active_sha,
        "expected_active_cond_sha256": EXPECTED_ACTIVE_COND_SHA256,
        "legacy_truebones_cond_path": str(legacy),
        "legacy_truebones_cond_sha256": legacy_sha,
        "expected_legacy_truebones_cond_sha256": EXPECTED_LEGACY_COND_SHA256,
        "legacy_rig_count": len(legacy_entries),
        "active_rig_count": len(active_entries),
        "legacy_keys_present_in_active": True,
        "legacy_geometry_payloads_exact_in_active": True,
        "allowed_fields": [
            "joints_names",
            "parents",
            "offsets",
            "tpos_first_frame[:,0:3]",
        ],
        "forbidden_authority_fields": [
            "tpos_first_frame[:,3:9]",
            "mean",
            "std",
            "legacy_btjd_motion_channels",
        ],
    }
    return active_entries, authority


def _forward_independent(
    names: tuple[str, ...], positions: np.ndarray, rig_id: str
) -> np.ndarray:
    try:
        method, anchor_names = EXPECTED_TRUEBONES_FORWARD_SPECS[rig_id]
    except KeyError as exc:
        raise SourceFkValidationError(f"no independent forward spec for {rig_id}") from exc
    lookup: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        lookup.setdefault(name, []).append(index)
    indices: list[int] = []
    for name in anchor_names:
        hits = lookup.get(name, [])
        if len(hits) != 1:
            raise SourceFkValidationError(
                f"{rig_id} forward anchor {name!r} resolves to {hits}"
            )
        indices.append(hits[0])
    if method == "lateral_pairs":
        across = (
            positions[indices[0]] - positions[indices[1]]
            + positions[indices[2]] - positions[indices[3]]
        )
        value = np.cross(np.asarray([0.0, 1.0, 0.0]), across)
    elif method == "root_to_head":
        value = positions[indices[1]] - positions[indices[0]]
    else:
        raise SourceFkValidationError(f"unsupported forward method {method}")
    horizontal = np.asarray([value[0], 0.0, value[2]], dtype=np.float64)
    norm = float(np.linalg.norm(horizontal))
    if not math.isfinite(norm) or norm <= 1e-10 * _aabb_diagonal(positions):
        raise SourceFkValidationError(f"{rig_id} independent forward is degenerate")
    return horizontal / norm


def _basis_independent(forward: np.ndarray) -> np.ndarray:
    fx, fz = float(forward[0]), float(forward[2])
    norm = math.hypot(fx, fz)
    fx, fz = fx / norm, fz / norm
    return np.asarray(
        [[fz, 0.0, -fx], [0.0, 1.0, 0.0], [fx, 0.0, fz]],
        dtype=np.float64,
    )


def _fixed_fk_independent(
    parents: np.ndarray,
    root_positions: np.ndarray,
    rotations: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    roots = np.asarray(root_positions, dtype=np.float64)
    matrices = np.asarray(rotations, dtype=np.float64)
    single = roots.ndim == 1
    if single:
        roots = roots[None]
        matrices = matrices[None]
    result = np.empty((len(roots), len(parents), 3), dtype=np.float64)
    result[:, 0] = roots
    for child in range(1, len(parents)):
        parent = int(parents[child])
        result[:, child] = result[:, parent] + np.einsum(
            "tij,j->ti", matrices[:, parent], offsets[child]
        )
    return result[0] if single else result


def _rotation_signature_independent(rotations: np.ndarray) -> str:
    values = np.asarray(rotations, dtype=np.float64)
    quantized = np.ascontiguousarray(
        np.rint(values / ROTATION_QUANTIZATION_STEP).astype("<i8", copy=False)
    )
    digest = hashlib.sha256()
    digest.update(json.dumps(list(values.shape), separators=(",", ":")).encode("ascii"))
    digest.update(quantized.tobytes(order="C"))
    return digest.hexdigest()


def _truebones_fixed_live(
    clip: dict[str, Any],
    rig: dict[str, Any],
    cond_entries: dict[str, Any],
) -> dict[str, Any]:
    rig_id = str(clip["rig_id"])
    entry = cond_entries[rig_id]
    names = tuple(str(value) for value in rig["joint_map"]["btjd_joint_names"])
    parents = np.asarray(rig["joint_map"]["btjd_parents"], dtype=np.int64)
    cond_names = tuple(str(value) for value in entry["joints_names"])
    cond_parents = np.asarray(entry["parents"], dtype=np.int64)
    if cond_names != names or not np.array_equal(cond_parents, parents):
        raise SourceFkValidationError(f"{rig_id} cond topology differs from manifest")
    offsets_cond = np.asarray(entry["offsets"], dtype=np.float64)
    tpose = np.asarray(entry["tpos_first_frame"], dtype=np.float64)
    raw_cond = tpose[:, :3].copy()
    cumulative = np.empty_like(raw_cond)
    cumulative[0] = raw_cond[0]
    for child in range(1, len(parents)):
        cumulative[child] = cumulative[int(parents[child])] + offsets_cond[child]
    cond_delta = cumulative - raw_cond
    P_rest = raw_cond.copy()
    ground_shift = -float(np.min(P_rest[:, 1]))
    P_rest[:, 1] += ground_shift
    rest_edges = np.asarray(
        [
            np.linalg.norm(P_rest[child] - P_rest[int(parents[child])])
            for child in range(1, len(parents))
        ],
        dtype=np.float64,
    )
    mean_edge = float(np.mean(rest_edges))
    if abs(mean_edge - TRUEBONES_MEAN_EDGE_TARGET) > 1e-8:
        raise SourceFkValidationError(f"{rig_id} independent cond mean edge drift")

    motion = _parse_bvh_independent(clip["source"]["path"])
    mapping = np.asarray(rig["joint_map"]["btjd_to_source"], dtype=np.int64)
    start, end = (int(value) for value in clip["source"]["slice_frames"])
    source_positions, _, local_rotations = _reroot_retained(
        motion.global_positions[start:end],
        motion.global_rotations[start:end],
        mapping,
        parents,
    )
    global_rotations = motion.global_rotations[start:end, mapping].copy()
    rest = _parse_bvh_independent(rig["rest_pose"]["source_path"])
    raw_rest_positions, _, raw_rest_local_rotations = _reroot_retained(
        rest.global_positions[:1], rest.global_rotations[:1], mapping, parents
    )
    raw_rest_positions = raw_rest_positions[0]
    raw_rest_global_rotations = rest.global_rotations[0, mapping].copy()
    raw_rest_local_rotations = raw_rest_local_rotations[0]

    source_forward = _forward_independent(names, raw_rest_positions, rig_id)
    C = _basis_independent(source_forward)
    raw_edges = np.asarray(
        [
            np.linalg.norm(
                raw_rest_positions[child] - raw_rest_positions[int(parents[child])]
            )
            for child in range(1, len(parents))
        ],
        dtype=np.float64,
    )
    raw_mean_edge = float(np.mean(raw_edges))
    alpha = TRUEBONES_MEAN_EDGE_TARGET / raw_mean_edge
    o = raw_rest_positions[0] - (C.T @ P_rest[0]) / alpha
    R_rest_global = np.einsum(
        "ab,jbc,dc->jad", C, raw_rest_global_rotations, C
    )
    R_rest_local = np.empty_like(R_rest_global)
    fixed_offsets = np.zeros_like(P_rest)
    R_rest_local[0] = R_rest_global[0]
    for child in range(1, len(parents)):
        parent = int(parents[child])
        R_rest_local[child] = R_rest_global[parent].T @ R_rest_global[child]
        fixed_offsets[child] = R_rest_global[parent].T @ (
            P_rest[child] - P_rest[parent]
        )
    rest_fk64 = _fixed_fk_independent(
        parents, P_rest[0], R_rest_global, fixed_offsets
    )
    s_rig = _aabb_diagonal(P_rest)
    rest_fk64_norm = float(
        np.max(np.linalg.norm(rest_fk64 - P_rest, axis=-1)) / s_rig
    )
    rest_fk32 = _fixed_fk_independent(
        parents,
        P_rest[0].astype(np.float32).astype(np.float64),
        R_rest_global.astype(np.float32).astype(np.float64),
        fixed_offsets.astype(np.float32).astype(np.float64),
    )
    rest_fk32_norm = float(
        np.max(
            np.linalg.norm(
                rest_fk32 - P_rest.astype(np.float32).astype(np.float64), axis=-1
            )
        )
        / s_rig
    )
    R_global = np.einsum("ab,tjbc,dc->tjad", C, global_rotations, C)
    authoritative_root = alpha * ((source_positions[:, 0] - o) @ C.T)
    authoritative = _fixed_fk_independent(
        parents, authoritative_root, R_global, fixed_offsets
    )
    motion_lengths = np.stack(
        [
            np.linalg.norm(
                authoritative[:, child] - authoritative[:, int(parents[child])],
                axis=-1,
            )
            for child in range(1, len(parents))
        ],
        axis=-1,
    )
    rigid_norm = float(np.max(np.abs(motion_lengths - rest_edges)) / s_rig)
    raw_xyz_canonical = alpha * ((source_positions - o) @ C.T)
    discrepancy = np.linalg.norm(raw_xyz_canonical - authoritative, axis=-1)

    nonroot_source_indices = [
        retained_index
        for retained_index, source_index in enumerate(mapping)
        if retained_index != 0
        and any(
            channel.lower().endswith("position")
            for channel in motion.channels[int(source_index)]
        )
    ]
    direct = motion.local_positions[
        start:end, mapping[nonroot_source_indices]
    ]
    direct_median = np.median(direct, axis=0)
    nonroot_variation = float(
        np.max(np.linalg.norm(direct - direct_median[None], axis=-1))
    )
    cond_forward = _forward_independent(names, P_rest, rig_id)
    metrics = {
        "cond_offsets_to_tpos_max_abs": float(np.max(np.abs(cond_delta))),
        "cond_offsets_to_tpos_max_norm": float(
            np.max(np.linalg.norm(cond_delta, axis=-1))
        ),
        "cond_ground_shift_y": ground_shift,
        "cond_ground_min_y_abs": abs(float(np.min(P_rest[:, 1]))),
        "cond_root_xz_max_abs": float(np.max(np.abs(P_rest[0, [0, 2]]))),
        "cond_mean_nonroot_edge_length": mean_edge,
        "cond_mean_edge_target_abs_error": abs(
            mean_edge - TRUEBONES_MEAN_EDGE_TARGET
        ),
        "source_raw_rest_mean_nonroot_edge_length": raw_mean_edge,
        "source_to_canonical_alpha": alpha,
        "s_rig": s_rig,
        "rest_fk_float64_max_norm": rest_fk64_norm,
        "rest_fk_float32_max_norm": rest_fk32_norm,
        "motion_rigid_edge_max_norm": rigid_norm,
        "raw_root_translation_max_abs": 0.0,
        "ignored_nonroot_xyz_joint_count": float(len(nonroot_source_indices)),
        "ignored_nonroot_xyz_sample_count": float(
            (end - start) * len(nonroot_source_indices)
        ),
        "ignored_nonroot_xyz_max_frame_variation_norm": nonroot_variation,
        "raw_xyz_vs_authoritative_mpjpe_norm": float(np.mean(discrepancy) / s_rig),
        "raw_xyz_vs_authoritative_max_norm": float(np.max(discrepancy) / s_rig),
        "conditioning_forward_to_plus_z_max_abs": float(
            np.max(np.abs(cond_forward - np.asarray([0.0, 0.0, 1.0])))
        ),
    }
    signatures = {
        "quantization_step": ROTATION_QUANTIZATION_STEP,
        "local_rotation_sha256": _rotation_signature_independent(local_rotations),
        "global_rotation_sha256": _rotation_signature_independent(global_rotations),
        "rest_local_rotation_sha256": _rotation_signature_independent(
            raw_rest_local_rotations
        ),
        "rest_global_rotation_sha256": _rotation_signature_independent(
            raw_rest_global_rotations
        ),
    }
    return {
        "metrics": metrics,
        "rotation_signatures": signatures,
        "conditioning_payload_sha256": _conditioning_payload_sha256(entry),
    }


def _expected_scope(
    clips: dict[str, dict[str, Any]], candidates: dict[str, Any]
) -> list[tuple[str, str, bool]]:
    expected: list[tuple[str, str, bool]] = []
    for family in (
        "human",
        "quadruped",
        "winged",
        "spider_crab",
        "dragon_or_deep_topology",
    ):
        selected = candidates["families"][family]["selected_train_candidates"]
        if len(selected) != 30:
            raise SourceFkValidationError(f"{family} does not retain 30 T03 prototypes")
        expected.extend(
            (clip_id, "prototype_train_calibration", True) for clip_id in selected
        )
    expected.extend(
        (
            clip_id,
            "held_representative_read_only",
            False,
        )
        for clip_id, record in sorted(clips.items())
        if record.get("topology_family") == "snake"
        and record.get("split") == "held_representative"
        and "SOURCE_NUMERIC_PARSE_INVALID" not in record.get("reason_codes", [])
        and "SOURCE_FK_REPRODUCTION_FAILED" not in record.get("reason_codes", [])
    )
    expected.extend(
        (clip_id, "held_stress_exact_dragon_read_only", False)
        for clip_id, record in sorted(clips.items())
        if record.get("rig_id") == "Dragon"
        and record.get("split") == "held_stress"
        and "SOURCE_NUMERIC_PARSE_INVALID" not in record.get("reason_codes", [])
        and "SOURCE_FK_REPRODUCTION_FAILED" not in record.get("reason_codes", [])
    )
    if len(expected) != 193 or len({item[0] for item in expected}) != 193:
        raise SourceFkValidationError(
            f"independent T03 scope is not 193 unique clips: {len(expected)}"
        )
    return expected


def validate_source_fk_outputs(manifest_root: str | Path) -> dict[str, Any]:
    """Reparse every T03 target and validate saved metrics and clip gates."""
    root = Path(manifest_root).expanduser().resolve()
    transaction = _validate_transaction(root)
    clips_list = _load_jsonl(root / "clips.jsonl")
    rigs_list = _load_jsonl(root / "rigs.jsonl")
    qa_records = _load_jsonl(root / "source_fk_qa.jsonl")
    clips = {record["clip_id"]: record for record in clips_list}
    rigs = {record["rig_id"]: record for record in rigs_list}
    qa = {record["clip_id"]: record for record in qa_records}
    if len(clips) != len(clips_list) or len(rigs) != len(rigs_list) or len(qa) != len(qa_records):
        raise SourceFkValidationError("duplicate ids in T03 artifacts")
    candidates = _load_json(root / "prototype_candidates.json")
    inventory_summary = _load_json(root / "inventory_summary.json")
    cond_entries, conditioning_authority = _load_conditioning_independent(
        inventory_summary
    )
    summary = _load_json(root / "source_fk_summary.json")
    generation = _load_json(root / "source_fk_generation.json")
    _require_equal("summary QA version", summary.get("qa_version"), EXPECTED_QA_VERSION)
    _require_equal("generation QA version", generation.get("qa_version"), EXPECTED_QA_VERSION)
    fixed_summary = summary.get("truebones_fixed_rig_contract")
    if not isinstance(fixed_summary, dict):
        raise SourceFkValidationError("summary lacks Truebones fixed-rig contract")
    _require_equal(
        "summary conditioning authority",
        fixed_summary.get("conditioning_authority"),
        conditioning_authority,
    )
    _require_equal(
        "generation conditioning authority",
        generation.get("truebones_conditioning_authority"),
        conditioning_authority,
    )
    _require_equal("summary thresholds", {
        family: payload["max"] for family, payload in summary["thresholds"].items()
    }, EXPECTED_THRESHOLDS)
    if summary["scope"].get("held_data_influenced_thresholds") is not False:
        raise SourceFkValidationError("held data was marked as influencing thresholds")
    if summary.get("encoder_invocation_count") != 0 or generation.get("encoder_called") is not False:
        raise SourceFkValidationError("T03 artifact claims an encoder invocation")
    threshold_audit = summary.get("train_only_threshold_audit")
    if not isinstance(threshold_audit, dict):
        raise SourceFkValidationError("train-only threshold audit is absent")
    for family, engineering_floor in EXPECTED_THRESHOLDS.items():
        values = np.asarray(
            [
                float(record["metrics"]["source_parser_fk_max_norm"])
                for record in qa_records
                if record.get("source_family") == family
                and record.get("calibration_eligible") is True
                and isinstance(record.get("metrics"), dict)
            ],
            dtype=np.float64,
        )
        if values.size == 0:
            raise SourceFkValidationError(
                f"source family {family} has no train-only threshold evidence"
            )
        q99_9 = float(np.percentile(values, 99.9))
        payload = threshold_audit.get(family)
        if not isinstance(payload, dict):
            raise SourceFkValidationError(
                f"source family {family} has no threshold-audit record"
            )
        _require_equal(
            f"{family} threshold train count",
            payload.get("train_clip_count"),
            int(values.size),
        )
        _require_close(
            f"{family} train Q99.9", float(payload["train_q99_9"]), q99_9
        )
        _require_close(
            f"{family} 1.5x train Q99.9",
            float(payload["one_point_five_times_train_q99_9"]),
            1.5 * q99_9,
        )
        _require_equal(
            f"{family} engineering floor",
            payload.get("engineering_floor"),
            engineering_floor,
        )
        _require_equal(
            f"{family} selected threshold",
            payload.get("selected_threshold"),
            EXPECTED_THRESHOLDS[family],
        )
        if payload.get("held_data_used") is not False:
            raise SourceFkValidationError(
                f"{family} threshold audit used held data"
            )

    expected = _expected_scope(clips, candidates)
    expected_ids = [item[0] for item in expected]
    _require_equal("QA record order/scope", [record["clip_id"] for record in qa_records], expected_ids)
    gate_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    topology_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    live_max: dict[str, float] = {}
    fixed_validated_count = 0
    fixed_live_maxima: dict[str, float] = {}
    rotation_agreement_observed_max = 0.0
    for index, (clip_id, role, calibration_eligible) in enumerate(expected, start=1):
        record = qa[clip_id]
        clip = clips[clip_id]
        rig = rigs[clip["rig_id"]]
        _require_equal(f"{clip_id} QA version", record.get("qa_version"), EXPECTED_QA_VERSION)
        _require_equal(f"{clip_id} role", record.get("audit_role"), role)
        _require_equal(
            f"{clip_id} calibration eligibility",
            record.get("calibration_eligible"),
            calibration_eligible,
        )
        if calibration_eligible and clip.get("split") != "train":
            raise SourceFkValidationError(f"held clip entered calibration: {clip_id}")
        family = clip["source"]["family"]
        _require_equal(
            f"{clip_id} threshold", record.get("threshold_max_norm"), EXPECTED_THRESHOLDS[family]
        )
        if record.get("encoder_called") is not False:
            raise SourceFkValidationError(f"encoder_called is not false for {clip_id}")
        try:
            if family == "motionstreamer272":
                metrics = _human_live_metrics(clip, rig)
            else:
                metrics = _bvh_live_metrics(clip, rig)
        except Exception as exc:
            if record.get("parser_status") != "fail" or record.get("gate_status") != "fail":
                raise SourceFkValidationError(
                    f"independent parse failed but saved QA passed for {clip_id}: {exc}"
                ) from exc
            continue
        if record.get("parser_status") != "pass" or not isinstance(record.get("metrics"), dict):
            raise SourceFkValidationError(
                f"independent parse passed but saved parser failed for {clip_id}"
            )
        for name, value in metrics.items():
            _require_close(
                f"{clip_id} metric {name}",
                float(record["metrics"][name]),
                value,
            )
        fixed_gate_passed = True
        if family == "truebones":
            fixed_live = _truebones_fixed_live(clip, rig, cond_entries)
            fixed_saved = record.get("fixed_rig")
            if not isinstance(fixed_saved, dict):
                raise SourceFkValidationError(
                    f"{clip_id} lacks saved fixed-rig evidence"
                )
            _require_equal(f"{clip_id} fixed-rig status", fixed_saved.get("status"), "pass")
            _require_equal(
                f"{clip_id} fixed-rig threshold",
                fixed_saved.get("threshold_max_norm"),
                RIGID_EDGE_MAX_NORM,
            )
            expected_conditioning = {
                **conditioning_authority,
                "rig_payload_sha256": fixed_live["conditioning_payload_sha256"],
            }
            _require_equal(
                f"{clip_id} conditioning authority",
                fixed_saved.get("conditioning"),
                expected_conditioning,
            )
            saved_fixed_metrics = fixed_saved.get("metrics")
            if not isinstance(saved_fixed_metrics, dict):
                raise SourceFkValidationError(
                    f"{clip_id} fixed-rig metrics are absent"
                )
            for name, value in fixed_live["metrics"].items():
                _require_close(
                    f"{clip_id} fixed-rig metric {name}",
                    float(saved_fixed_metrics[name]),
                    float(value),
                )
                fixed_live_maxima[name] = max(
                    fixed_live_maxima.get(name, 0.0), float(value)
                )
            _require_equal(
                f"{clip_id} full rotation signatures",
                fixed_saved.get("rotation_signatures"),
                fixed_live["rotation_signatures"],
            )
            rotation_agreement = fixed_saved.get("rotation_agreement")
            if not isinstance(rotation_agreement, dict):
                raise SourceFkValidationError(
                    f"{clip_id} lacks direct producer-vs-SciPy rotation agreement"
                )
            _require_equal(
                f"{clip_id} rotation agreement threshold",
                rotation_agreement.get("threshold_max_abs"),
                1e-12,
            )
            for name in (
                "motion_local_rotation_scipy_max_abs",
                "motion_global_rotation_scipy_max_abs",
                "rest_local_rotation_scipy_max_abs",
                "rest_global_rotation_scipy_max_abs",
            ):
                value = float(rotation_agreement[name])
                if not math.isfinite(value) or value > 1e-12:
                    raise SourceFkValidationError(
                        f"{clip_id} rotation agreement gate failed for {name}: {value}"
                    )
                rotation_agreement_observed_max = max(
                    rotation_agreement_observed_max, value
                )
            provenance = fixed_saved.get("provenance")
            if not isinstance(provenance, dict):
                raise SourceFkValidationError(
                    f"{clip_id} fixed-rig provenance is absent"
                )
            if provenance.get("forbidden_inputs_used") is not False:
                raise SourceFkValidationError(
                    f"{clip_id} used a forbidden fixed-rig input"
                )
            _require_equal(
                f"{clip_id} raw non-root XYZ role",
                provenance.get("raw_nonroot_xyz_role"),
                "diagnostic_only_never_rest_offset_or_motion_authority",
            )
            fixed_gate_passed = bool(
                fixed_live["metrics"]["motion_rigid_edge_max_norm"]
                <= RIGID_EDGE_MAX_NORM
                and fixed_live["metrics"]["rest_fk_float64_max_norm"] <= 1e-10
                and fixed_live["metrics"]["rest_fk_float32_max_norm"] <= 1e-5
            )
            fixed_validated_count += 1
        expected_gate = (
            "pass"
            if metrics["source_parser_fk_max_norm"] <= EXPECTED_THRESHOLDS[family]
            and fixed_gate_passed
            else "fail"
        )
        _require_equal(f"{clip_id} gate", record.get("gate_status"), expected_gate)
        clip_gate = clip.get("source_parser_fk")
        if not isinstance(clip_gate, dict):
            raise SourceFkValidationError(f"clip manifest lacks T03 gate: {clip_id}")
        _require_equal(f"{clip_id} clip gate", clip_gate.get("status"), expected_gate)
        _require_equal(f"{clip_id} clip gate role", clip_gate.get("audit_role"), role)
        if family == "truebones":
            _require_equal(
                f"{clip_id} clip fixed-rig evidence",
                clip_gate.get("fixed_rig"),
                record.get("fixed_rig"),
            )
        if "NUMERIC_PAYLOAD_VALIDATION_DEFERRED_T03" in clip["reason_codes"]:
            raise SourceFkValidationError(f"audited clip remains deferred: {clip_id}")
        if expected_gate == "pass" and any(
            code in clip["reason_codes"]
            for code in (
                "SOURCE_NUMERIC_PARSE_INVALID",
                "SOURCE_FK_REPRODUCTION_FAILED",
            )
        ):
            raise SourceFkValidationError(f"passing clip carries a T03 reject: {clip_id}")
        gate_counts[expected_gate] += 1
        source_counts[family] += 1
        topology_counts[clip["topology_family"]] += 1
        role_counts[role] += 1
        live_max[family] = max(
            live_max.get(family, 0.0), metrics["source_parser_fk_max_norm"]
        )
        if index % 10 == 0 or index == len(expected):
            print(f"[source-fk-validation] reparsed {index}/{len(expected)}", flush=True)

    _require_equal(
        "summary gate counts", summary["counts"]["gate_status"], dict(sorted(gate_counts.items()))
    )
    _require_equal(
        "summary source counts", summary["counts"]["source_family"], dict(sorted(source_counts.items()))
    )
    _require_equal(
        "summary topology counts", summary["counts"]["topology_family"], dict(sorted(topology_counts.items()))
    )
    _require_equal(
        "summary role counts", summary["counts"]["audit_role"], dict(sorted(role_counts.items()))
    )
    _require_equal(
        "summary fixed-rig passing clip count",
        fixed_summary.get("passing_clip_count"),
        fixed_validated_count,
    )
    for name, expected_value in fixed_summary.get("metric_maxima", {}).items():
        _require_close(
            f"summary fixed-rig maximum {name}",
            float(expected_value),
            float(fixed_live_maxima[name]),
        )
    _require_close(
        "summary rotation agreement observed maximum",
        float(fixed_summary["rotation_agreement_observed_max_abs"]),
        rotation_agreement_observed_max,
    )
    return {
        "qa_version": EXPECTED_QA_VERSION,
        "manifest_version": EXPECTED_INVENTORY_VERSION,
        "generation_id": transaction.get("generation_id"),
        "validated_at_utc": _datetime.datetime.now(
            _datetime.timezone.utc
        ).isoformat().replace("+00:00", "Z"),
        "status": "pass",
        "validation_mode": "independent_scipy_euler_and_homogeneous_fk_live_reparse",
        "live_reparsed_clip_count": len(expected),
        "calibration_train_clip_count": sum(item[2] for item in expected),
        "held_read_only_clip_count": sum(not item[2] for item in expected),
        "held_data_influenced_thresholds": False,
        "truebones_fixed_rig_validated_clip_count": fixed_validated_count,
        "truebones_fixed_rig_live_maxima": fixed_live_maxima,
        "truebones_conditioning_authority": conditioning_authority,
        "rotation_validation": (
            "producer_vs_independent_scipy_max_abs_1e-12_plus_integrity_sha256_step_1e-8"
        ),
        "gate_status_counts": dict(sorted(gate_counts.items())),
        "source_family_counts": dict(sorted(source_counts.items())),
        "topology_family_counts": dict(sorted(topology_counts.items())),
        "live_source_family_max_norm": dict(sorted(live_max.items())),
        "encoder_invocation_count": 0,
        "visual_qa_claimed": False,
        "canonicalization_claimed": False,
        "ktjd_encoding_claimed": False,
    }


def write_source_fk_validation_report(
    report: dict[str, Any],
    path: str | Path,
    *,
    immutable_manifest_root: str | Path,
) -> None:
    target = Path(path).expanduser().resolve()
    immutable_root = Path(immutable_manifest_root).expanduser().resolve()
    if target == immutable_root or immutable_root in target.parents:
        raise SourceFkValidationError(
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
        temporary = Path(temp_name)
        if temporary.exists():
            temporary.unlink()
