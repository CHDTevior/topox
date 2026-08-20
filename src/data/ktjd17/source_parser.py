"""Strict numeric source parsers for the KTJD-17 source-FK gate.

This module stops before canonicalization and KTJD encoding.  Its only job is
to recover the transforms that are *actually* carried by each source and prove
that those transforms reproduce the source positions in float64.

Important source-specific distinctions:

* BVH Euler rotations follow each joint's declared channel order.  A joint
  with XYZ position channels uses those values as its complete local
  translation, matching the AnyTop/Truebones lineage (the values replace the
  declared OFFSET rather than being silently added to it).
* MotionStreamer272 stores PyTorch3D-style row-cont6d, not the KTJD-17
  column-cont6d public codec.  It also omits SMPL shape coefficients.  We may
  audit a clip-constant shaped rest offset recovered from true rotations plus
  source positions, but that offset is explicitly not a frozen per-rig rest.

No function in this module imports or calls a KTJD encoder.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .bvh_inventory import BvhHeader, BvhInventoryError, parse_bvh_header


MOTIONSTREAMER272_DIM = 272
MOTIONSTREAMER272_JOINTS = 22
MOTIONSTREAMER272_POSITION_SLICE = slice(8, 8 + 3 * MOTIONSTREAMER272_JOINTS)
MOTIONSTREAMER272_ROTATION_SLICE = slice(
    8 + 6 * MOTIONSTREAMER272_JOINTS,
    8 + 12 * MOTIONSTREAMER272_JOINTS,
)
SOURCE_D6_DEGENERACY_EPS = 1e-6


class SourceParserError(RuntimeError):
    """A source payload cannot satisfy the lossless numeric parser contract."""


@dataclasses.dataclass(frozen=True)
class ParsedBvhMotion:
    """Full BVH hierarchy and numeric samples before retained-joint mapping."""

    path: str
    fps: float
    joint_names: tuple[str, ...]
    parents: np.ndarray
    node_kinds: tuple[str, ...]
    rotation_source_kind: tuple[str, ...]
    offsets: np.ndarray
    channels: tuple[tuple[str, ...], ...]
    local_positions: np.ndarray
    local_rotations: np.ndarray
    global_positions: np.ndarray
    global_rotations: np.ndarray


@dataclasses.dataclass(frozen=True)
class ParsedSourceMotion:
    """Common retained-physical-joint parser output used by the T03 gate."""

    family: str
    path: str
    fps: float
    joint_names: tuple[str, ...]
    parents: np.ndarray
    rotation_source_kind: tuple[str, ...]
    source_joint_indices: np.ndarray
    root_translation: np.ndarray
    local_positions: np.ndarray
    local_rotations: np.ndarray
    global_positions: np.ndarray
    global_rotations: np.ndarray
    source_positions: np.ndarray
    fk_positions: np.ndarray
    rest_local_positions: np.ndarray
    rest_local_rotations: np.ndarray
    rest_global_positions: np.ndarray
    rest_global_rotations: np.ndarray
    rest_status: str
    rest_path: str | None
    diagnostics: dict[str, Any]

    @property
    def frame_count(self) -> int:
        return int(self.source_positions.shape[0])

    @property
    def joint_count(self) -> int:
        return int(self.source_positions.shape[1])

    @property
    def s_rig(self) -> float:
        return aabb_diagonal(self.rest_global_positions)


def _require_float64_finite(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.float64:
        raise SourceParserError(f"{name} must be float64, got {array.dtype}")
    if not np.isfinite(array).all():
        bad = np.argwhere(~np.isfinite(array))[0].tolist()
        raise SourceParserError(f"{name} contains a non-finite value at {bad}")
    return array


def _validate_parent_tree(parents: np.ndarray, joint_count: int) -> np.ndarray:
    parent_array = np.asarray(parents, dtype=np.int64)
    if parent_array.shape != (joint_count,):
        raise SourceParserError(
            f"parents must have shape ({joint_count},), got {parent_array.shape}"
        )
    if int(parent_array[0]) != -1:
        raise SourceParserError(f"physical root parent must be -1, got {parent_array[0]}")
    for child in range(1, joint_count):
        parent = int(parent_array[child])
        if not 0 <= parent < child:
            raise SourceParserError(
                f"parent-before-child violated at joint {child}: parent={parent}"
            )
    return parent_array


def aabb_diagonal(positions: np.ndarray) -> float:
    points = _require_float64_finite("rest_global_positions", np.asarray(positions))
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise SourceParserError(
            f"rest_global_positions must be [J,3], got {points.shape}"
        )
    scale = float(np.linalg.norm(np.ptp(points, axis=0)))
    if not math.isfinite(scale) or scale <= 0.0:
        raise SourceParserError(f"rest-pose AABB diagonal must be positive, got {scale}")
    return scale


def decode_source_row_cont6d(d6: np.ndarray) -> np.ndarray:
    """Decode MotionStreamer/PyTorch3D row-cont6d in strict float64 mode."""
    source = _require_float64_finite("source_row_cont6d", np.asarray(d6))
    if source.shape[-1] != 6:
        raise SourceParserError(
            f"source row-cont6d last dimension must be 6, got {source.shape}"
        )
    a1 = source[..., 0:3]
    a2 = source[..., 3:6]
    norm1 = np.linalg.norm(a1, axis=-1, keepdims=True)
    if np.any(norm1 < SOURCE_D6_DEGENERACY_EPS):
        bad = np.argwhere(norm1[..., 0] < SOURCE_D6_DEGENERACY_EPS)[0].tolist()
        raise SourceParserError(
            f"source row-cont6d first row is degenerate at {bad}: "
            f"norm={float(norm1[tuple(bad) + (0,)])}"
        )
    b1 = a1 / norm1
    u2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    norm2 = np.linalg.norm(u2, axis=-1, keepdims=True)
    if np.any(norm2 < SOURCE_D6_DEGENERACY_EPS):
        bad = np.argwhere(norm2[..., 0] < SOURCE_D6_DEGENERACY_EPS)[0].tolist()
        raise SourceParserError(
            f"source row-cont6d second row is degenerate at {bad}: "
            f"norm={float(norm2[tuple(bad) + (0,)])}"
        )
    b2 = u2 / norm2
    b3 = np.cross(b1, b2)
    matrices = np.stack((b1, b2, b3), axis=-2)
    return _require_float64_finite("decoded_source_row_cont6d", matrices)


def rotation_matrix_diagnostics(rotations: np.ndarray) -> dict[str, float]:
    matrices = _require_float64_finite("rotation_matrices", np.asarray(rotations))
    if matrices.shape[-2:] != (3, 3):
        raise SourceParserError(f"rotations must end in [3,3], got {matrices.shape}")
    gram = np.matmul(np.swapaxes(matrices, -1, -2), matrices)
    orthogonality_max = float(np.max(np.abs(gram - np.eye(3, dtype=np.float64))))
    determinants = np.linalg.det(matrices)
    return {
        "rotation_orthogonality_max": orthogonality_max,
        "rotation_determinant_min": float(np.min(determinants)),
        "rotation_determinant_max": float(np.max(determinants)),
    }


def forward_kinematics(
    parents: np.ndarray,
    local_positions: np.ndarray,
    local_rotations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate active column-vector local transforms in parent-before-child order."""
    positions_local = _require_float64_finite(
        "local_positions", np.asarray(local_positions)
    )
    rotations_local = _require_float64_finite(
        "local_rotations", np.asarray(local_rotations)
    )
    if positions_local.ndim != 3 or positions_local.shape[-1] != 3:
        raise SourceParserError(
            f"local_positions must be [T,J,3], got {positions_local.shape}"
        )
    if rotations_local.shape != positions_local.shape[:2] + (3, 3):
        raise SourceParserError(
            "local rotation/position shape mismatch: "
            f"{rotations_local.shape} vs {positions_local.shape}"
        )
    frame_count, joint_count = positions_local.shape[:2]
    parent_array = _validate_parent_tree(parents, joint_count)
    global_positions = np.empty((frame_count, joint_count, 3), dtype=np.float64)
    global_rotations = np.empty(
        (frame_count, joint_count, 3, 3), dtype=np.float64
    )
    global_positions[:, 0] = positions_local[:, 0]
    global_rotations[:, 0] = rotations_local[:, 0]
    for child in range(1, joint_count):
        parent = int(parent_array[child])
        global_positions[:, child] = global_positions[:, parent] + np.einsum(
            "tij,tj->ti", global_rotations[:, parent], positions_local[:, child]
        )
        global_rotations[:, child] = np.matmul(
            global_rotations[:, parent], rotations_local[:, child]
        )
    return (
        _require_float64_finite("global_positions", global_positions),
        _require_float64_finite("global_rotations", global_rotations),
    )


def _homogeneous_positions(
    parents: np.ndarray,
    local_positions: np.ndarray,
    local_rotations: np.ndarray,
) -> np.ndarray:
    """Second source-position evaluator used inside the producer-side gate."""
    frame_count, joint_count = local_positions.shape[:2]
    transforms = np.zeros((frame_count, joint_count, 4, 4), dtype=np.float64)
    transforms[..., 3, 3] = 1.0
    transforms[..., :3, :3] = local_rotations
    transforms[..., :3, 3] = local_positions
    global_transforms = np.empty_like(transforms)
    global_transforms[:, 0] = transforms[:, 0]
    for child in range(1, joint_count):
        global_transforms[:, child] = np.matmul(
            global_transforms[:, int(parents[child])], transforms[:, child]
        )
    return _require_float64_finite(
        "homogeneous_source_positions", global_transforms[..., :3, 3]
    )


def _axis_rotation(axis: str, angles_radians: np.ndarray) -> np.ndarray:
    values = _require_float64_finite("euler_angles_radians", angles_radians)
    cosine = np.cos(values)
    sine = np.sin(values)
    result = np.zeros(values.shape + (3, 3), dtype=np.float64)
    if axis == "x":
        result[..., 0, 0] = 1.0
        result[..., 1, 1] = cosine
        result[..., 1, 2] = -sine
        result[..., 2, 1] = sine
        result[..., 2, 2] = cosine
    elif axis == "y":
        result[..., 0, 0] = cosine
        result[..., 0, 2] = sine
        result[..., 1, 1] = 1.0
        result[..., 2, 0] = -sine
        result[..., 2, 2] = cosine
    elif axis == "z":
        result[..., 0, 0] = cosine
        result[..., 0, 1] = -sine
        result[..., 1, 0] = sine
        result[..., 1, 1] = cosine
        result[..., 2, 2] = 1.0
    else:
        raise SourceParserError(f"unsupported Euler axis {axis!r}")
    return result


def _read_bvh_values(path: Path, header: BvhHeader) -> np.ndarray:
    saw_frame_time = False
    motion_text: list[str] = []
    try:
        with path.open("r", encoding="utf-8", errors="strict") as handle:
            for raw_line in handle:
                text = raw_line.strip()
                if not saw_frame_time:
                    if text.lower().startswith("frame time"):
                        saw_frame_time = True
                    continue
                if text:
                    motion_text.append(text)
    except OSError as exc:
        raise SourceParserError(f"cannot read BVH motion samples {path}: {exc}") from exc
    if not saw_frame_time:
        raise SourceParserError(f"BVH has no Frame Time line: {path}")
    expected = header.frames * header.channel_count
    tokens = " ".join(motion_text).split()
    if len(tokens) != expected:
        raise SourceParserError(
            f"BVH numeric count mismatch for {path}: {len(tokens)} != {expected} "
            f"({header.frames} frames x {header.channel_count} channels)"
        )
    try:
        values = np.asarray([float(token) for token in tokens], dtype=np.float64)
    except ValueError as exc:
        raise SourceParserError(
            f"cannot parse every BVH numeric token in {path}: {exc}"
        ) from exc
    values = values.reshape(header.frames, header.channel_count)
    return _require_float64_finite("bvh_motion_values", values)


def parse_bvh_numeric(path: str | Path) -> ParsedBvhMotion:
    """Parse arbitrary per-joint BVH channel layouts without legacy fallbacks."""
    source = Path(path).expanduser().resolve()
    try:
        header = parse_bvh_header(source)
    except BvhInventoryError as exc:
        raise SourceParserError(str(exc)) from exc
    values = _read_bvh_values(source, header)
    frame_count = header.frames
    joint_count = len(header.joints)
    offsets = np.asarray([joint.offset for joint in header.joints], dtype=np.float64)
    parents = np.asarray(header.parents, dtype=np.int64)
    local_positions = np.broadcast_to(
        offsets[None], (frame_count, joint_count, 3)
    ).copy()
    local_rotations = np.broadcast_to(
        np.eye(3, dtype=np.float64), (frame_count, joint_count, 3, 3)
    ).copy()

    cursor = 0
    for joint_index, joint in enumerate(header.joints):
        channel_count = len(joint.channels)
        block = values[:, cursor : cursor + channel_count]
        cursor += channel_count
        position_channels: list[tuple[int, int]] = []
        rotation_channels: list[tuple[str, int]] = []
        for channel_index, channel in enumerate(joint.channels):
            lowered = channel.lower()
            if lowered.endswith("position") and lowered[0] in "xyz":
                position_channels.append(("xyz".index(lowered[0]), channel_index))
            elif lowered.endswith("rotation") and lowered[0] in "xyz":
                rotation_channels.append((lowered[0], channel_index))
            else:
                raise SourceParserError(
                    f"unsupported BVH channel {channel!r} at joint {joint.name!r}"
                )

        if position_channels:
            axes = sorted(axis for axis, _ in position_channels)
            if axes != [0, 1, 2] or len(position_channels) != 3:
                raise SourceParserError(
                    f"joint {joint.name!r} must expose either zero or one X/Y/Z "
                    f"position channel each, got {joint.channels}"
                )
            # The local Truebones/PZ lineage replaces OFFSET when XYZ position
            # channels exist.  Truebones non-root values are the offsets.
            local_positions[:, joint_index] = 0.0
            for axis, channel_index in position_channels:
                local_positions[:, joint_index, axis] = block[:, channel_index]

        if rotation_channels:
            axes = sorted(axis for axis, _ in rotation_channels)
            if axes != ["x", "y", "z"] or len(rotation_channels) != 3:
                raise SourceParserError(
                    f"joint {joint.name!r} must expose one X/Y/Z rotation channel "
                    f"each, got {joint.channels}"
                )
            result = np.broadcast_to(
                np.eye(3, dtype=np.float64), (frame_count, 3, 3)
            ).copy()
            for axis, channel_index in rotation_channels:
                elemental = _axis_rotation(
                    axis, np.deg2rad(block[:, channel_index].astype(np.float64))
                )
                result = np.matmul(result, elemental)
            local_rotations[:, joint_index] = result
        elif joint.rotation_source_kind() != "fixed_dof":
            raise SourceParserError(
                f"joint {joint.name!r} has no numeric rotation and is not fixed"
            )
    if cursor != header.channel_count:
        raise SourceParserError(
            f"BVH channel cursor mismatch for {source}: {cursor} != {header.channel_count}"
        )

    global_positions, global_rotations = forward_kinematics(
        parents, local_positions, local_rotations
    )
    reference_positions = _homogeneous_positions(
        parents, local_positions, local_rotations
    )
    if not np.array_equal(global_positions, reference_positions):
        max_error = float(np.max(np.abs(global_positions - reference_positions)))
        if max_error > 1e-12:
            raise SourceParserError(
                f"internal BVH transform evaluators disagree for {source}: {max_error}"
            )
    return ParsedBvhMotion(
        path=str(source),
        fps=header.fps,
        joint_names=header.joint_names,
        parents=parents,
        node_kinds=tuple(joint.node_kind for joint in header.joints),
        rotation_source_kind=header.rotation_source_kinds(),
        offsets=offsets,
        channels=tuple(joint.channels for joint in header.joints),
        local_positions=local_positions,
        local_rotations=local_rotations,
        global_positions=reference_positions,
        global_rotations=global_rotations,
    )


def _slice_frames(
    motion: ParsedBvhMotion, frame_slice: Sequence[int] | None
) -> tuple[int, int]:
    if frame_slice is None:
        return 0, motion.local_positions.shape[0]
    if len(frame_slice) != 2:
        raise SourceParserError(f"frame_slice must be [start,end], got {frame_slice}")
    start, end = (int(frame_slice[0]), int(frame_slice[1]))
    if not 0 <= start < end <= motion.local_positions.shape[0]:
        raise SourceParserError(
            f"invalid frame_slice [{start},{end}) for T={motion.local_positions.shape[0]}"
        )
    return start, end


def _map_retained(
    motion: ParsedBvhMotion,
    *,
    retained_names: Sequence[str],
    retained_parents: Sequence[int],
    expected_rotation_kinds: Sequence[str],
    frame_slice: Sequence[int] | None,
) -> dict[str, Any]:
    names = tuple(str(name) for name in retained_names)
    if len(names) == 0 or len(set(names)) != len(names):
        raise SourceParserError("retained joint names must be unique and non-empty")
    parents = _validate_parent_tree(np.asarray(retained_parents), len(names))
    if len(expected_rotation_kinds) != len(names):
        raise SourceParserError("retained rotation kind count does not match joint count")
    source_lookup: dict[str, list[int]] = {}
    for source_index, source_name in enumerate(motion.joint_names):
        source_lookup.setdefault(source_name, []).append(source_index)
    resolved: list[int] = []
    used: set[int] = set()
    for retained_index, name in enumerate(names):
        hits = source_lookup.get(name, [])
        if len(hits) == 1:
            source_index = hits[0]
        elif len(hits) > 1:
            raise SourceParserError(
                f"retained joint {name!r} is ambiguous in {motion.path}: {hits}"
            )
        else:
            # The local AnyTop lineage serializes a previously unnamed BVH End
            # Site as ``<parent>_end_site``.  Recreate the T02 structural map
            # only when parent order is already proven and exactly one unused
            # unnamed End Site exists below the mapped source parent.
            retained_parent = int(parents[retained_index])
            candidates: list[int] = []
            if (
                retained_index > 0
                and name.endswith("_end_site")
                and retained_parent < len(resolved)
            ):
                source_parent = resolved[retained_parent]
                candidates = [
                    source_index
                    for source_index, (source_name, source_kind) in enumerate(
                        zip(motion.joint_names, motion.node_kinds, strict=True)
                    )
                    if int(motion.parents[source_index]) == source_parent
                    and source_kind == "end_site"
                    and "__unnamed_end_site_" in source_name
                    and source_index not in used
                ]
            if len(candidates) != 1:
                raise SourceParserError(
                    f"retained joint {name!r} is missing from {motion.path}; "
                    f"structural End Site candidates={candidates}"
                )
            source_index = candidates[0]
        if source_index in used:
            raise SourceParserError(
                f"source joint {source_index} maps to more than one retained joint"
            )
        resolved.append(source_index)
        used.add(source_index)
    source_indices = np.asarray(resolved, dtype=np.int64)

    for child in range(1, len(names)):
        expected_parent = int(source_indices[int(parents[child])])
        source_cursor = int(motion.parents[int(source_indices[child])])
        while source_cursor >= 0 and source_cursor != expected_parent:
            source_cursor = int(motion.parents[source_cursor])
        if source_cursor != expected_parent:
            raise SourceParserError(
                f"retained edge {names[int(parents[child])]!r}->{names[child]!r} "
                f"is not a source ancestry edge in {motion.path}"
            )
    actual_kinds = tuple(motion.rotation_source_kind[index] for index in source_indices)
    if actual_kinds != tuple(expected_rotation_kinds):
        raise SourceParserError(
            f"rotation-source kinds drifted for {motion.path}: "
            f"{actual_kinds} != {tuple(expected_rotation_kinds)}"
        )

    start, end = _slice_frames(motion, frame_slice)
    positions = motion.global_positions[start:end, source_indices].copy()
    rotations = motion.global_rotations[start:end, source_indices].copy()
    frame_count = end - start
    local_positions = np.empty((frame_count, len(names), 3), dtype=np.float64)
    local_rotations = np.empty((frame_count, len(names), 3, 3), dtype=np.float64)
    local_positions[:, 0] = positions[:, 0]
    local_rotations[:, 0] = rotations[:, 0]
    for child in range(1, len(names)):
        parent = int(parents[child])
        parent_inverse = np.swapaxes(rotations[:, parent], -1, -2)
        local_positions[:, child] = np.einsum(
            "tij,tj->ti", parent_inverse, positions[:, child] - positions[:, parent]
        )
        local_rotations[:, child] = np.matmul(parent_inverse, rotations[:, child])
    fk_positions, fk_rotations = forward_kinematics(
        parents, local_positions, local_rotations
    )
    rotation_error = float(np.max(np.abs(fk_rotations - rotations)))
    if rotation_error > 1e-12:
        raise SourceParserError(
            f"retained rotation re-rooting failed for {motion.path}: {rotation_error}"
        )
    return {
        "names": names,
        "parents": parents,
        "kinds": actual_kinds,
        "source_indices": source_indices,
        "positions": positions,
        "rotations": rotations,
        "local_positions": local_positions,
        "local_rotations": local_rotations,
        "fk_positions": fk_positions,
    }


def _rest_from_bvh(
    rest_motion: ParsedBvhMotion,
    *,
    retained_names: Sequence[str],
    retained_parents: Sequence[int],
    expected_rotation_kinds: Sequence[str],
    use_first_numeric_frame: bool,
) -> dict[str, np.ndarray]:
    if use_first_numeric_frame:
        mapped = _map_retained(
            rest_motion,
            retained_names=retained_names,
            retained_parents=retained_parents,
            expected_rotation_kinds=expected_rotation_kinds,
            frame_slice=(0, 1),
        )
        return {
            "local_positions": mapped["local_positions"][0],
            "local_rotations": mapped["local_rotations"][0],
            "global_positions": mapped["positions"][0],
            "global_rotations": mapped["rotations"][0],
        }

    # Processed PlanetZoo has no local raw rest file.  Hierarchy offsets remain
    # useful scale/topology evidence, but identity rotations are explicitly a
    # review-only candidate and not accepted as the final T04 rest.
    frame_count = rest_motion.local_positions.shape[0]
    identity_local_positions = np.broadcast_to(
        rest_motion.offsets[None], (frame_count, len(rest_motion.offsets), 3)
    ).copy()
    identity_local_rotations = np.broadcast_to(
        np.eye(3, dtype=np.float64),
        (frame_count, len(rest_motion.offsets), 3, 3),
    ).copy()
    identity_positions, identity_rotations = forward_kinematics(
        rest_motion.parents, identity_local_positions, identity_local_rotations
    )
    identity_motion = dataclasses.replace(
        rest_motion,
        local_positions=identity_local_positions,
        local_rotations=identity_local_rotations,
        global_positions=identity_positions,
        global_rotations=identity_rotations,
    )
    mapped = _map_retained(
        identity_motion,
        retained_names=retained_names,
        retained_parents=retained_parents,
        expected_rotation_kinds=expected_rotation_kinds,
        frame_slice=(0, 1),
    )
    return {
        "local_positions": mapped["local_positions"][0],
        "local_rotations": mapped["local_rotations"][0],
        "global_positions": mapped["positions"][0],
        "global_rotations": mapped["rotations"][0],
    }


def parse_bvh_source(
    path: str | Path,
    *,
    retained_names: Sequence[str],
    retained_parents: Sequence[int],
    expected_rotation_kinds: Sequence[str],
    frame_slice: Sequence[int] | None,
    rest_path: str | Path,
    rest_mode: str,
    parsed_rest: ParsedBvhMotion | None = None,
    family: str,
) -> ParsedSourceMotion:
    """Parse one Truebones or processed PlanetZoo clip into the common API."""
    if family not in {"truebones", "planetzoo"}:
        raise SourceParserError(f"unsupported BVH source family {family!r}")
    if rest_mode not in {
        "explicit_tpose_frame",
        "legacy_idle_fallback_review",
        "processed_hierarchy_only_review",
    }:
        raise SourceParserError(f"unsupported BVH rest_mode {rest_mode!r}")
    motion = parse_bvh_numeric(path)
    mapped = _map_retained(
        motion,
        retained_names=retained_names,
        retained_parents=retained_parents,
        expected_rotation_kinds=expected_rotation_kinds,
        frame_slice=frame_slice,
    )
    rest_source = parsed_rest or parse_bvh_numeric(rest_path)
    rest = _rest_from_bvh(
        rest_source,
        retained_names=retained_names,
        retained_parents=retained_parents,
        expected_rotation_kinds=expected_rotation_kinds,
        use_first_numeric_frame=rest_mode
        in {"explicit_tpose_frame", "legacy_idle_fallback_review"},
    )
    diagnostics = rotation_matrix_diagnostics(mapped["local_rotations"])
    nonroot_static = 0.0
    if mapped["local_positions"].shape[1] > 1:
        median = np.median(mapped["local_positions"][:, 1:], axis=0)
        nonroot_static = float(
            np.max(
                np.linalg.norm(
                    mapped["local_positions"][:, 1:] - median[None], axis=-1
                )
            )
        )
    start, end = _slice_frames(motion, frame_slice)
    retained_position_channel_indices = [
        retained_index
        for retained_index, source_index in enumerate(mapped["source_indices"])
        if any(
            channel.lower().endswith("position")
            for channel in motion.channels[int(source_index)]
        )
    ]
    nonroot_position_channel_indices = [
        index for index in retained_position_channel_indices if index != 0
    ]
    nonroot_position_channel_variation = 0.0
    if nonroot_position_channel_indices:
        source_indices = mapped["source_indices"][nonroot_position_channel_indices]
        direct_values = motion.local_positions[start:end, source_indices]
        median = np.median(direct_values, axis=0)
        nonroot_position_channel_variation = float(
            np.max(np.linalg.norm(direct_values - median[None], axis=-1))
        )
    diagnostics.update(
        {
            "source_full_joint_count": len(motion.joint_names),
            "source_full_frame_count": int(motion.local_positions.shape[0]),
            "nonroot_local_translation_static_max_abs": nonroot_static,
            "retained_position_channel_joint_count": len(
                retained_position_channel_indices
            ),
            "nonroot_position_channel_joint_count": len(
                nonroot_position_channel_indices
            ),
            "nonroot_position_channel_sample_count": (
                (end - start) * len(nonroot_position_channel_indices)
            ),
            "nonroot_position_channel_max_frame_variation_norm": (
                nonroot_position_channel_variation
            ),
            "position_channel_semantics": "xyz_channels_replace_offset",
            "euler_semantics": "active_intrinsic_declared_order",
        }
    )
    return ParsedSourceMotion(
        family=family,
        path=motion.path,
        fps=motion.fps,
        joint_names=mapped["names"],
        parents=mapped["parents"],
        rotation_source_kind=mapped["kinds"],
        source_joint_indices=mapped["source_indices"],
        root_translation=mapped["local_positions"][:, 0].copy(),
        local_positions=mapped["local_positions"],
        local_rotations=mapped["local_rotations"],
        global_positions=mapped["positions"],
        global_rotations=mapped["rotations"],
        source_positions=mapped["positions"].copy(),
        fk_positions=mapped["fk_positions"],
        rest_local_positions=rest["local_positions"],
        rest_local_rotations=rest["local_rotations"],
        rest_global_positions=rest["global_positions"],
        rest_global_rotations=rest["global_rotations"],
        rest_status=rest_mode,
        rest_path=str(Path(rest_path).expanduser().resolve()),
        diagnostics=diagnostics,
    )


def _recover_motionstreamer_positions(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame_count = data.shape[0]
    positions_no_heading = data[:, MOTIONSTREAMER272_POSITION_SLICE].reshape(
        frame_count, MOTIONSTREAMER272_JOINTS, 3
    )
    heading_delta = decode_source_row_cont6d(data[:, 2:8])
    heading = np.empty_like(heading_delta)
    heading[0] = heading_delta[0]
    for frame in range(1, frame_count):
        heading[frame] = np.matmul(heading_delta[frame], heading[frame - 1])
    inverse_heading = np.swapaxes(heading, -1, -2)
    positions = np.einsum(
        "tij,tkj->tki", inverse_heading, positions_no_heading
    )
    root_velocity = np.zeros((frame_count, 3), dtype=np.float64)
    root_velocity[:, 0] = data[:, 0]
    root_velocity[:, 2] = data[:, 1]
    if frame_count > 1:
        root_velocity[1:] = np.einsum(
            "tij,tj->ti", inverse_heading[:-1], root_velocity[1:]
        )
    root_translation = np.cumsum(root_velocity, axis=0)
    positions[:, :, 0] += root_translation[:, None, 0]
    positions[:, :, 2] += root_translation[:, None, 2]
    return (
        _require_float64_finite("motionstreamer_source_positions", positions),
        inverse_heading,
    )


def _load_smpl_neutral_rest(
    model_path: str | Path, parents: np.ndarray
) -> np.ndarray:
    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise SourceParserError(f"SMPL neutral model is missing: {path}")
    try:
        model = np.load(path, allow_pickle=True)
        vertices = np.asarray(model["v_template"], dtype=np.float64)
        regressor_raw = model["J_regressor"]
        if hasattr(regressor_raw, "toarray"):
            regressor_raw = regressor_raw.toarray()
        regressor = np.asarray(regressor_raw, dtype=np.float64)
        kintree = np.asarray(model["kintree_table"])[0, : len(parents)].astype(
            np.int64
        )
    except Exception as exc:  # noqa: BLE001
        raise SourceParserError(f"cannot read SMPL neutral rest {path}: {exc}") from exc
    kintree[0] = -1
    if not np.array_equal(kintree, parents):
        raise SourceParserError(
            f"SMPL neutral parent tree differs from MotionStreamer schema: "
            f"{kintree.tolist()} != {parents.tolist()}"
        )
    rest = np.asarray(regressor @ vertices, dtype=np.float64)[: len(parents)]
    return _require_float64_finite("smpl_neutral_rest_positions", rest)


def parse_motionstreamer272_source(
    path: str | Path,
    *,
    joint_names: Sequence[str],
    parents: Sequence[int],
    neutral_model_path: str | Path,
) -> ParsedSourceMotion:
    """Decode one official MotionStreamer272 payload without position IK."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise SourceParserError(f"MotionStreamer272 source is missing: {source}")
    try:
        raw = np.load(source, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001
        raise SourceParserError(f"cannot load MotionStreamer272 source {source}: {exc}") from exc
    if raw.ndim != 2 or raw.shape[1] != MOTIONSTREAMER272_DIM or raw.shape[0] <= 0:
        raise SourceParserError(
            f"MotionStreamer272 must be [T,272], got {raw.shape} at {source}"
        )
    data = np.asarray(raw, dtype=np.float64)
    _require_float64_finite("motionstreamer272", data)
    names = tuple(str(name) for name in joint_names)
    if len(names) != MOTIONSTREAMER272_JOINTS or len(set(names)) != len(names):
        raise SourceParserError(
            f"MotionStreamer joint names must contain 22 unique entries, got {len(names)}"
        )
    parent_array = _validate_parent_tree(
        np.asarray(parents), MOTIONSTREAMER272_JOINTS
    )
    source_positions, inverse_heading = _recover_motionstreamer_positions(data)

    local_rotations = decode_source_row_cont6d(
        data[:, MOTIONSTREAMER272_ROTATION_SLICE].reshape(
            data.shape[0], MOTIONSTREAMER272_JOINTS, 6
        )
    )
    # The root source rotation was made heading-free during representation
    # construction.  Restore its world orientation exactly as the official
    # recover_from_local_rotation implementation does.
    local_rotations[:, 0] = np.matmul(
        inverse_heading, local_rotations[:, 0]
    )
    global_rotations = np.empty_like(local_rotations)
    global_rotations[:, 0] = local_rotations[:, 0]
    for child in range(1, MOTIONSTREAMER272_JOINTS):
        global_rotations[:, child] = np.matmul(
            global_rotations[:, int(parent_array[child])],
            local_rotations[:, child],
        )

    # Shape coefficients are absent from 272.  True rotations plus source
    # positions nevertheless expose a clip-constant shaped offset.  Recovering
    # this offset does not infer rotations; it is an audit-only rest candidate
    # and remains explicitly unfrozen for T04.
    observed_offsets = np.zeros(
        (data.shape[0], MOTIONSTREAMER272_JOINTS, 3), dtype=np.float64
    )
    for child in range(1, MOTIONSTREAMER272_JOINTS):
        parent = int(parent_array[child])
        observed_offsets[:, child] = np.einsum(
            "tij,tj->ti",
            np.swapaxes(global_rotations[:, parent], -1, -2),
            source_positions[:, child] - source_positions[:, parent],
        )
    shaped_offsets = np.median(observed_offsets, axis=0)
    local_positions = np.broadcast_to(
        shaped_offsets[None], source_positions.shape
    ).copy()
    local_positions[:, 0] = source_positions[:, 0]
    fk_positions, fk_rotations = forward_kinematics(
        parent_array, local_positions, local_rotations
    )
    global_rotation_error = float(
        np.max(np.abs(fk_rotations - global_rotations))
    )
    if global_rotation_error > 1e-12:
        raise SourceParserError(
            f"MotionStreamer rotation FK recurrence disagrees: {global_rotation_error}"
        )

    neutral_positions = _load_smpl_neutral_rest(neutral_model_path, parent_array)
    neutral_offsets = np.zeros_like(neutral_positions)
    for child in range(1, MOTIONSTREAMER272_JOINTS):
        neutral_offsets[child] = (
            neutral_positions[child]
            - neutral_positions[int(parent_array[child])]
        )
    neutral_fk_local = np.broadcast_to(
        neutral_offsets[None], source_positions.shape
    ).copy()
    neutral_fk_local[:, 0] = source_positions[:, 0]
    neutral_fk_positions, _ = forward_kinematics(
        parent_array, neutral_fk_local, local_rotations
    )
    rest_local_rotations = np.broadcast_to(
        np.eye(3, dtype=np.float64),
        (MOTIONSTREAMER272_JOINTS, 3, 3),
    ).copy()
    diagnostics = rotation_matrix_diagnostics(local_rotations)
    offset_deviation = np.linalg.norm(
        observed_offsets - shaped_offsets[None], axis=-1
    )
    neutral_error = np.linalg.norm(
        neutral_fk_positions - source_positions, axis=-1
    )
    diagnostics.update(
        {
            "source_full_joint_count": MOTIONSTREAMER272_JOINTS,
            "source_full_frame_count": int(data.shape[0]),
            "source_d6_semantics": "row_cont6d_first_two_rows",
            "heading_recovery_semantics": "left_accumulate_then_transpose",
            "observed_shaped_offset_variation_max_abs": float(
                np.max(offset_deviation)
            ),
            "neutral_rest_fk_mpjpe_abs": float(np.mean(neutral_error)),
            "neutral_rest_fk_p99_abs": float(np.percentile(neutral_error, 99)),
            "neutral_rest_fk_max_abs": float(np.max(neutral_error)),
            "observed_vs_neutral_offset_mean_abs": float(
                np.mean(np.linalg.norm(shaped_offsets - neutral_offsets, axis=-1))
            ),
            "observed_vs_neutral_offset_max_abs": float(
                np.max(np.linalg.norm(shaped_offsets - neutral_offsets, axis=-1))
            ),
        }
    )
    return ParsedSourceMotion(
        family="motionstreamer272",
        path=str(source),
        fps=30.0,
        joint_names=names,
        parents=parent_array,
        rotation_source_kind=tuple("animated_dof" for _ in names),
        source_joint_indices=np.arange(MOTIONSTREAMER272_JOINTS, dtype=np.int64),
        root_translation=source_positions[:, 0].copy(),
        local_positions=local_positions,
        local_rotations=local_rotations,
        global_positions=fk_positions,
        global_rotations=global_rotations,
        source_positions=source_positions,
        fk_positions=fk_positions,
        rest_local_positions=neutral_offsets,
        rest_local_rotations=rest_local_rotations,
        rest_global_positions=neutral_positions,
        rest_global_rotations=rest_local_rotations.copy(),
        rest_status="neutral_smpl_rest_with_missing_shape_coefficients_review",
        rest_path=str(Path(neutral_model_path).expanduser().resolve()),
        diagnostics=diagnostics,
    )


def source_fk_metrics(parsed: ParsedSourceMotion) -> dict[str, float | str]:
    """Compute the normalized source-parser FK gate metrics."""
    difference = _require_float64_finite(
        "source_fk_difference", parsed.fk_positions - parsed.source_positions
    )
    errors = np.linalg.norm(difference, axis=-1)
    scale = parsed.s_rig
    return {
        "s_rig": scale,
        "s_rig_definition": "rest_pose_aabb_diagonal",
        "mpjpe_abs": float(np.mean(errors)),
        "p99_abs": float(np.percentile(errors, 99)),
        "max_abs": float(np.max(errors)),
        "source_parser_fk_error_norm": float(np.mean(errors) / scale),
        "source_parser_fk_p99_norm": float(np.percentile(errors, 99) / scale),
        "source_parser_fk_max_norm": float(np.max(errors) / scale),
    }


def require_source_fk_pass(record: dict[str, Any]) -> None:
    """Hard guard for later encoders; T03 itself never invokes an encoder."""
    gate = record.get("source_parser_fk")
    if not isinstance(gate, dict) or gate.get("status") != "pass":
        raise SourceParserError(
            f"clip {record.get('clip_id')!r} cannot enter KTJD encoding: "
            f"source_parser_fk.status={None if not isinstance(gate, dict) else gate.get('status')!r}"
        )
