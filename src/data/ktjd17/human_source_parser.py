"""Efficient fixed-neutral MotionStreamer272 parser for Human KTJD-17.

The historical source parser reloads the 84 MiB SMPL-H model for every clip to
produce diagnostics. PZ+Human-312 has one separately reviewed, hash-pinned
fixed-neutral Human rig, so full conversion instead passes that rest geometry
explicitly. Every time-varying rotation still comes directly from
MotionStreamer272 channels 140:272; this module performs no position IK.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .source_parser import (
    MOTIONSTREAMER272_DIM,
    MOTIONSTREAMER272_JOINTS,
    MOTIONSTREAMER272_POSITION_SLICE,
    MOTIONSTREAMER272_ROTATION_SLICE,
    ParsedSourceMotion,
    SourceParserError,
    decode_source_row_cont6d,
    forward_kinematics,
    rotation_matrix_diagnostics,
)


FIXED_NEUTRAL_PARSER_VERSION = "motionstreamer272-fixed-neutral-v1"


class MotionStreamer272ContentError(SourceParserError):
    """A stable MotionStreamer272 payload violates an explicit numeric contract."""


def _finite_float64(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.float64 or not np.isfinite(array).all():
        raise SourceParserError(f"{name} must be finite float64, got {array.dtype}")
    return array


def _parents(value: Sequence[int]) -> np.ndarray:
    result = np.asarray(value, dtype=np.int64)
    if result.shape != (MOTIONSTREAMER272_JOINTS,) or int(result[0]) != -1:
        raise SourceParserError("fixed-neutral Human parents are invalid")
    for child in range(1, len(result)):
        if not 0 <= int(result[child]) < child:
            raise SourceParserError(
                f"fixed-neutral Human parent-before-child fails at {child}"
            )
    return result


def _recover_positions(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frames = data.shape[0]
    positions_no_heading = data[:, MOTIONSTREAMER272_POSITION_SLICE].reshape(
        frames, MOTIONSTREAMER272_JOINTS, 3
    )
    try:
        heading_delta = decode_source_row_cont6d(data[:, 2:8])
    except SourceParserError as exc:
        raise MotionStreamer272ContentError(
            f"invalid MotionStreamer272 heading rows: {exc}"
        ) from exc
    heading = np.empty_like(heading_delta)
    heading[0] = heading_delta[0]
    for frame in range(1, frames):
        heading[frame] = heading_delta[frame] @ heading[frame - 1]
    inverse_heading = np.swapaxes(heading, -1, -2)
    positions = np.einsum("tij,tkj->tki", inverse_heading, positions_no_heading)
    root_velocity = np.zeros((frames, 3), dtype=np.float64)
    root_velocity[:, 0] = data[:, 0]
    root_velocity[:, 2] = data[:, 1]
    if frames > 1:
        root_velocity[1:] = np.einsum(
            "tij,tj->ti", inverse_heading[:-1], root_velocity[1:]
        )
    root_translation = np.cumsum(root_velocity, axis=0)
    positions[..., 0] += root_translation[:, None, 0]
    positions[..., 2] += root_translation[:, None, 2]
    try:
        positions = _finite_float64("fixed_neutral_source_positions", positions)
    except SourceParserError as exc:
        raise MotionStreamer272ContentError(
            f"invalid recovered MotionStreamer272 positions: {exc}"
        ) from exc
    return positions, inverse_heading


def parse_motionstreamer272_fixed_neutral_array(
    data: np.ndarray,
    *,
    source_identity: str,
    joint_names: Sequence[str],
    parents: Sequence[int],
    P_rest_global: np.ndarray,
    offset_parent_local: np.ndarray,
    rest_authority: str,
) -> ParsedSourceMotion:
    """Decode one already-stable float64 array against a fixed-neutral rig."""
    raw = np.asarray(data)
    if raw.ndim != 2 or raw.shape[1] != MOTIONSTREAMER272_DIM or raw.shape[0] <= 0:
        raise MotionStreamer272ContentError(
            f"MotionStreamer272 must be [T,272], got {raw.shape} at {source_identity}"
        )
    try:
        data = _finite_float64("motionstreamer272_fixed_neutral", np.asarray(raw))
    except SourceParserError as exc:
        raise MotionStreamer272ContentError(str(exc)) from exc
    names = tuple(str(name) for name in joint_names)
    if len(names) != MOTIONSTREAMER272_JOINTS or len(set(names)) != len(names):
        raise SourceParserError("fixed-neutral Human names must be 22 unique joints")
    parent_array = _parents(parents)
    rest_positions = _finite_float64(
        "fixed_neutral_P_rest_global", np.asarray(P_rest_global)
    )
    offsets = _finite_float64(
        "fixed_neutral_offset_parent_local", np.asarray(offset_parent_local)
    )
    if rest_positions.shape != (MOTIONSTREAMER272_JOINTS, 3) or offsets.shape != (
        MOTIONSTREAMER272_JOINTS,
        3,
    ):
        raise SourceParserError("fixed-neutral Human rest geometry has invalid shape")
    for child in range(1, MOTIONSTREAMER272_JOINTS):
        expected = rest_positions[child] - rest_positions[int(parent_array[child])]
        if float(np.max(np.abs(expected - offsets[child]))) > 1e-7:
            raise SourceParserError(
                f"fixed-neutral Human offset/rest mismatch at joint {child}"
            )

    source_positions, inverse_heading = _recover_positions(data)
    try:
        local_rotations = decode_source_row_cont6d(
            data[:, MOTIONSTREAMER272_ROTATION_SLICE].reshape(
                data.shape[0], MOTIONSTREAMER272_JOINTS, 6
            )
        )
    except SourceParserError as exc:
        raise MotionStreamer272ContentError(
            f"invalid MotionStreamer272 local-rotation rows: {exc}"
        ) from exc
    local_rotations[:, 0] = inverse_heading @ local_rotations[:, 0]
    global_rotations = np.empty_like(local_rotations)
    global_rotations[:, 0] = local_rotations[:, 0]
    for child in range(1, MOTIONSTREAMER272_JOINTS):
        global_rotations[:, child] = (
            global_rotations[:, int(parent_array[child])] @ local_rotations[:, child]
        )

    observed_offsets = np.zeros_like(source_positions)
    for child in range(1, MOTIONSTREAMER272_JOINTS):
        parent = int(parent_array[child])
        observed_offsets[:, child] = np.einsum(
            "tij,tj->ti",
            np.swapaxes(global_rotations[:, parent], -1, -2),
            source_positions[:, child] - source_positions[:, parent],
        )
    shaped_offsets = np.median(observed_offsets, axis=0)
    shaped_local_positions = np.broadcast_to(
        shaped_offsets[None], source_positions.shape
    ).copy()
    shaped_local_positions[:, 0] = source_positions[:, 0]
    fk_positions, fk_rotations = forward_kinematics(
        parent_array, shaped_local_positions, local_rotations
    )
    recurrence_error = float(np.max(np.abs(fk_rotations - global_rotations)))
    if recurrence_error > 1e-12:
        raise SourceParserError(
            f"fixed-neutral Human rotation recurrence disagrees: {recurrence_error}"
        )

    fixed_local_positions = np.broadcast_to(offsets[None], source_positions.shape).copy()
    fixed_local_positions[:, 0] = source_positions[:, 0]
    fixed_positions, _ = forward_kinematics(
        parent_array, fixed_local_positions, local_rotations
    )
    shaped_offset_deviation = np.linalg.norm(
        observed_offsets - shaped_offsets[None], axis=-1
    )
    fixed_error = np.linalg.norm(fixed_positions - source_positions, axis=-1)
    try:
        diagnostics = rotation_matrix_diagnostics(local_rotations)
    except SourceParserError as exc:
        raise MotionStreamer272ContentError(
            f"invalid decoded MotionStreamer272 rotations: {exc}"
        ) from exc
    diagnostics.update(
        {
            "parser_version": FIXED_NEUTRAL_PARSER_VERSION,
            "source_full_joint_count": MOTIONSTREAMER272_JOINTS,
            "source_full_frame_count": int(data.shape[0]),
            "source_d6_semantics": "row_cont6d_first_two_rows",
            "heading_recovery_semantics": "left_accumulate_then_transpose",
            "observed_shaped_offset_variation_max_abs": float(
                np.max(shaped_offset_deviation)
            ),
            "fixed_neutral_fk_mpjpe_abs": float(np.mean(fixed_error)),
            "fixed_neutral_fk_p99_abs": float(np.percentile(fixed_error, 99)),
            "fixed_neutral_fk_max_abs": float(np.max(fixed_error)),
            "fixed_neutral_position_role": "authoritative_ktjd_geometry",
            "raw_shaped_position_role": "diagnostic_only",
            "position_ik_used": False,
        }
    )
    identity = np.broadcast_to(
        np.eye(3, dtype=np.float64),
        (MOTIONSTREAMER272_JOINTS, 3, 3),
    ).copy()
    return ParsedSourceMotion(
        family="motionstreamer272",
        path=str(source_identity),
        fps=30.0,
        joint_names=names,
        parents=parent_array,
        rotation_source_kind=tuple("animated_dof" for _ in names),
        source_joint_indices=np.arange(MOTIONSTREAMER272_JOINTS, dtype=np.int64),
        root_translation=source_positions[:, 0].copy(),
        local_positions=shaped_local_positions,
        local_rotations=local_rotations,
        global_positions=fk_positions,
        global_rotations=global_rotations,
        source_positions=source_positions,
        fk_positions=fk_positions,
        rest_local_positions=offsets.copy(),
        rest_declared_offsets=offsets.copy(),
        rest_local_rotations=identity,
        rest_global_positions=rest_positions.copy(),
        rest_global_rotations=identity.copy(),
        rest_status="current_btjd_fixed_neutral_human",
        rest_path=rest_authority,
        diagnostics=diagnostics,
    )


def parse_motionstreamer272_fixed_neutral(
    path: str | Path,
    *,
    joint_names: Sequence[str],
    parents: Sequence[int],
    P_rest_global: np.ndarray,
    offset_parent_local: np.ndarray,
    rest_authority: str,
) -> ParsedSourceMotion:
    """Convenience path wrapper; exhaustive audits use the stable-array API."""
    requested = Path(path).expanduser().absolute()
    if requested.is_symlink():
        raise SourceParserError(f"MotionStreamer272 source is linked: {requested}")
    source = requested.resolve()
    if not source.is_file():
        raise SourceParserError(f"MotionStreamer272 source is missing: {source}")
    try:
        raw = np.load(source, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001
        raise SourceParserError(
            f"cannot load MotionStreamer272 source {source}: {exc}"
        ) from exc
    return parse_motionstreamer272_fixed_neutral_array(
        np.asarray(raw),
        source_identity=str(source),
        joint_names=joint_names,
        parents=parents,
        P_rest_global=P_rest_global,
        offset_parent_local=offset_parent_local,
        rest_authority=rest_authority,
    )
