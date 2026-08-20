"""Read-only, source-backed fixed algebraic QA for KTJD-17 prototypes.

This module intentionally recomputes the public representation from stored
artifacts and raw rotation sources.  It does not update thresholds, schema,
statistics, selections, or any dataset generation.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation, Slerp

from .encoder import SkeletonData, load_skeleton
from .codec import encode_column_cont6d
from .loader import derive_masks, load_motion_npz, yaw_augment
from .source_parser import (
    ParsedBvhMotion,
    ParsedSourceMotion,
    parse_bvh_numeric,
    parse_bvh_source,
    parse_motionstreamer272_source,
)


FIXED_QA_VERSION = "ktjd17-fixed-qa-v2"
SOURCE_FK_MAX_NORM = {"truebones": 1e-10, "motionstreamer272": 1e-6}
FORBIDDEN_VIRTUAL_NAMES = {"WORLD", "WORLD_NODE", "__WORLD__", "CONTROL"}


class FixedQaError(RuntimeError):
    """A prototype cannot satisfy the immutable fixed-QA contract."""


@dataclasses.dataclass(frozen=True)
class SourceReference:
    parsed: ParsedSourceMotion
    root_positions: np.ndarray
    local_rotations: np.ndarray


@dataclasses.dataclass(frozen=True)
class AlignedSourceReference:
    """Source-backed motion aligned to the stored clip-canonical world."""

    positions_clip: np.ndarray
    positions_absolute: np.ndarray
    global_rotations: np.ndarray
    resample_mode: str


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise FixedQaError(f"cannot read JSON {path}: {exc}") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise FixedQaError(f"{path}:{line_number}: blank JSONL row")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise FixedQaError(f"{path}:{line_number}: row is not an object")
                records.append(value)
    except FixedQaError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise FixedQaError(f"cannot read JSONL {path}: {exc}") from exc
    return records


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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FixedQaError(message)


def _resolve_generation_path(root: Path, relpath: Any, *, label: str) -> Path:
    _require(isinstance(relpath, str) and bool(relpath), f"{label}: invalid relpath")
    relative = Path(relpath)
    _require(not relative.is_absolute(), f"{label}: absolute path is forbidden")
    resolved = (root / relative).resolve()
    _require(resolved.is_relative_to(root), f"{label}: path escapes generation")
    return resolved


def _validate_topology_distance_bucket(
    *,
    clip_id: str,
    manifest: Mapping[str, Any],
    qa: Mapping[str, Any],
    parent_clip: Mapping[str, Any],
) -> str:
    """Require the producer and both audit rows to preserve the pinned bucket."""
    bucket = parent_clip.get("topology_distance_bucket")
    _require(
        isinstance(bucket, str) and bool(bucket),
        f"{clip_id}: parent topology-distance bucket is absent",
    )
    _require(
        manifest.get("topology_distance_bucket") == bucket,
        f"{clip_id}: manifest topology-distance bucket drifted",
    )
    _require(
        qa.get("topology_distance_bucket") == bucket,
        f"{clip_id}: QA topology-distance bucket drifted",
    )
    return bucket


def _finite(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if not np.isfinite(array).all():
        location = np.argwhere(~np.isfinite(array))[0].tolist()
        raise FixedQaError(f"{name}: non-finite value at {location}")
    return array


def _validate_tree(parents: np.ndarray) -> np.ndarray:
    values = np.asarray(parents, dtype=np.int64)
    _require(values.ndim == 1 and len(values) > 0, "parents must be nonempty [J]")
    _require(int(values[0]) == -1, "physical root 0 must have parent -1")
    for child in range(1, len(values)):
        parent = int(values[child])
        _require(
            0 <= parent < child,
            f"parent-before-child failed at child={child}, parent={parent}",
        )
    return values


def _rotation_diagnostics(rotations: np.ndarray) -> tuple[float, float, float]:
    matrices = _finite("rotations", np.asarray(rotations, dtype=np.float64))
    gram = np.matmul(np.swapaxes(matrices, -1, -2), matrices)
    determinants = np.linalg.det(matrices)
    return (
        float(np.max(np.abs(gram - np.eye(3, dtype=np.float64)))),
        float(np.min(determinants)),
        float(np.max(determinants)),
    )


def independent_decode_column_cont6d(d6: np.ndarray) -> np.ndarray:
    """Independent NumPy implementation of the normative column-cont6d decoder."""
    values = _finite("d6", np.asarray(d6, dtype=np.float64))
    _require(values.shape[-1] == 6, f"d6 must end in 6, got {values.shape}")
    first = values[..., :3]
    second = values[..., 3:]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    _require(
        bool(np.all(first_norm[..., 0] >= 1e-6)),
        "stored GT d6 contains a degenerate first column",
    )
    b1 = first / first_norm
    second_orthogonal = second - np.sum(b1 * second, axis=-1, keepdims=True) * b1
    second_norm = np.linalg.norm(second_orthogonal, axis=-1, keepdims=True)
    _require(
        bool(np.all(second_norm[..., 0] >= 1e-6)),
        "stored GT d6 contains a degenerate second column",
    )
    b2 = second_orthogonal / second_norm
    b3 = np.cross(b1, b2)
    result = np.stack((b1, b2, b3), axis=-1)
    orthogonality, determinant_min, determinant_max = _rotation_diagnostics(result)
    _require(orthogonality <= 2e-6, f"decoded d6 is not orthogonal: {orthogonality}")
    _require(
        determinant_min > 0.0
        and abs(determinant_min - 1.0) <= 2e-6
        and abs(determinant_max - 1.0) <= 2e-6,
        f"decoded d6 is not right-handed: [{determinant_min},{determinant_max}]",
    )
    return result


def _validate_global_algebra() -> dict[str, float]:
    identity = np.eye(3, dtype=np.float64)
    y90 = np.asarray(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    expected = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    gold = encode_column_cont6d(np.stack((identity, y90)))
    gold_error = float(np.max(np.abs(gold - expected)))
    _require(gold_error <= 1e-12, f"6D gold error {gold_error}")
    gold_decode_error = float(
        np.max(
            np.linalg.norm(
                independent_decode_column_cont6d(gold)
                - np.stack((identity, y90)),
                axis=(-2, -1),
            )
        )
    )
    _require(gold_decode_error <= 1e-12, f"6D gold decode {gold_decode_error}")
    random_rotations = Rotation.random(10_000, random_state=20260819).as_matrix()
    random_decoded = independent_decode_column_cont6d(
        encode_column_cont6d(random_rotations)
    )
    random_error = float(
        np.max(np.linalg.norm(random_decoded - random_rotations, axis=(-2, -1)))
    )
    _require(random_error <= 1e-10, f"6D random roundtrip {random_error}")
    return {
        "gold_encode_max_abs": gold_error,
        "gold_decode_frobenius_max": gold_decode_error,
        "random_10000_frobenius_max": random_error,
    }


def _global_to_local(parents: np.ndarray, global_rotations: np.ndarray) -> np.ndarray:
    local = np.empty_like(global_rotations)
    local[:, 0] = global_rotations[:, 0]
    for child in range(1, len(parents)):
        parent = int(parents[child])
        local[:, child] = np.matmul(
            np.swapaxes(global_rotations[:, parent], -1, -2),
            global_rotations[:, child],
        )
    return local


def _local_to_global(parents: np.ndarray, local_rotations: np.ndarray) -> np.ndarray:
    global_rotations = np.empty_like(local_rotations)
    global_rotations[:, 0] = local_rotations[:, 0]
    for child in range(1, len(parents)):
        parent = int(parents[child])
        global_rotations[:, child] = np.matmul(
            global_rotations[:, parent], local_rotations[:, child]
        )
    return global_rotations


def _fk(
    parents: np.ndarray,
    roots: np.ndarray,
    global_rotations: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    positions = np.empty(
        (global_rotations.shape[0], global_rotations.shape[1], 3),
        dtype=np.float64,
    )
    positions[:, 0] = roots
    for child in range(1, len(parents)):
        parent = int(parents[child])
        positions[:, child] = positions[:, parent] + np.einsum(
            "tij,j->ti", global_rotations[:, parent], offsets[child]
        )
    return positions


def _resample(
    roots: np.ndarray,
    local_rotations: np.ndarray,
    *,
    fps_src: float,
    fps_target: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    if fps_src == fps_target:
        return roots.copy(), local_rotations.copy(), "exact_fps_identity_bypass"
    if len(roots) == 1:
        return roots.copy(), local_rotations.copy(), "single_frame_identity"
    source_times = np.arange(len(roots), dtype=np.float64) / float(fps_src)
    target_count = math.floor(float(source_times[-1]) * float(fps_target)) + 1
    target_times = np.arange(target_count, dtype=np.float64) / float(fps_target)
    _require(
        bool(target_times[-1] <= source_times[-1]),
        "independent timestamp resampling attempted extrapolation",
    )
    target_roots = np.stack(
        [np.interp(target_times, source_times, roots[:, axis]) for axis in range(3)],
        axis=-1,
    )
    target_local = np.empty(
        (target_count, local_rotations.shape[1], 3, 3), dtype=np.float64
    )
    for joint in range(local_rotations.shape[1]):
        target_local[:, joint] = Slerp(
            source_times, Rotation.from_matrix(local_rotations[:, joint])
        )(target_times).as_matrix()
    return target_roots, target_local, "timestamp_linear_root_local_so3_slerp_then_fk"


def _velocity(positions: np.ndarray, fps: float) -> np.ndarray:
    velocity = np.zeros_like(positions)
    if len(positions) >= 2:
        velocity[:-1] = (positions[1:] - positions[:-1]) * float(fps)
        velocity[-1] = velocity[-2]
    return velocity


def _smooth_root(
    root_xz: np.ndarray, *, fps: float, smoother: Mapping[str, Any]
) -> tuple[np.ndarray, str]:
    _require(smoother.get("id") == "butterworth_filtfilt", "unknown smoother")
    params = smoother.get("params")
    _require(isinstance(params, Mapping), "smoother.params is absent")
    order = int(params["order"])
    cutoff = float(params["cutoff_hz"])
    padtype = str(params["padtype"])
    padlen = int(params["padlen"])
    cycles = float(params["short_clip_cycles"])
    _require(order == 4 and padtype == "odd" and padlen == 15, "smoother drifted")
    if len(root_xz) == 1:
        return root_xz.copy(), "single_frame_identity"
    if len(root_xz) < cycles * round(float(fps) / cutoff) or len(root_xz) <= padlen:
        x = np.arange(len(root_xz), dtype=np.float64)
        design = np.stack((x, np.ones_like(x)), axis=-1)
        coefficients, _, _, _ = np.linalg.lstsq(design, root_xz, rcond=None)
        return design @ coefficients, "ols_line"
    b, a = butter(order, cutoff / (0.5 * float(fps)), btype="low", analog=False)
    return (
        filtfilt(b, a, root_xz, axis=0, padtype=padtype, padlen=padlen),
        "butterworth_filtfilt",
    )


def _heading(
    global_rotations: np.ndarray,
    *,
    carrier: int,
    forward_local: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = np.einsum(
        "tij,j->ti", global_rotations[:, int(carrier)], forward_local
    )
    horizontal = np.hypot(forward[:, 0], forward[:, 2])
    valid = horizontal >= float(epsilon)
    heading = np.zeros((len(global_rotations), 2), dtype=np.float64)
    heading[valid, 0] = forward[valid, 2] / horizontal[valid]
    heading[valid, 1] = forward[valid, 0] / horizontal[valid]
    return heading, valid, horizontal


def _longest_false_run(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in np.asarray(mask, dtype=bool):
        current = 0 if value else current + 1
        longest = max(longest, current)
    return longest


def _validate_loader_masks(
    *,
    T: int,
    J: int,
    parents: np.ndarray,
    rotation_source_kind: np.ndarray,
    heading_valid: np.ndarray,
) -> int:
    T_max = T + 2
    J_max = J + 2
    observed = derive_masks(
        T_valid=T,
        J_phys=J,
        T_max=T_max,
        J_max=J_max,
        parents=parents,
        rotation_source_kind=rotation_source_kind,
        heading_valid=heading_valid,
    )
    frame = np.zeros(T_max, dtype=bool)
    frame[:T] = True
    joint = np.zeros(J_max, dtype=bool)
    joint[:J] = True
    channel = np.zeros((J_max, 17), dtype=bool)
    channel[:J, :13] = True
    channel[0, 13:17] = True
    kinds = np.asarray(rotation_source_kind).astype(str)
    rotation = np.zeros(J_max, dtype=bool)
    rotation[:J] = kinds == "animated_dof"
    fixed = np.zeros(J_max, dtype=bool)
    fixed[:J] = kinds == "fixed_dof"
    child_edge = np.zeros(J_max, dtype=bool)
    for child in range(1, J):
        child_edge[child] = joint[child] and joint[int(parents[child])]
    heading = np.zeros(T_max, dtype=bool)
    heading[:T] = heading_valid
    pairs = (
        ("frame_mask", observed.frame_mask, frame),
        ("joint_mask", observed.joint_mask, joint),
        ("channel_valid_mask", observed.channel_valid_mask, channel),
        ("rotation_supervised", observed.rotation_supervised, rotation),
        ("fixed_rotation_mask", observed.fixed_rotation_mask, fixed),
        ("contact_supervised", observed.contact_supervised, joint),
        ("child_edge_valid", observed.child_edge_valid, child_edge),
        ("heading_valid", observed.heading_valid, heading),
    )
    mismatch = 0
    for name, actual, expected in pairs:
        current = int(np.count_nonzero(np.asarray(actual) != expected))
        _require(current == 0, f"{name}: {current} mask entries drifted")
        mismatch += current
    return mismatch


def _rotation_angle_error(reference: np.ndarray, observed: np.ndarray) -> np.ndarray:
    relative = np.matmul(np.swapaxes(reference, -1, -2), observed)
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cosine)


def _acceleration_rms(positions: np.ndarray, fps: float, scale: float) -> float:
    if len(positions) < 3:
        return 0.0
    acceleration = np.diff(positions, n=2, axis=0) * float(fps) ** 2
    return float(np.sqrt(np.mean(np.square(acceleration / float(scale)))))


def _rest_mode(rig: Mapping[str, Any]) -> str:
    method = rig.get("rest_pose", {}).get("selection_method")
    if method == "explicit_tpose_filename":
        return "explicit_tpose_frame"
    if method in {"legacy_idle_fallback", "legacy_first_file_fallback"}:
        return "legacy_idle_fallback_review"
    raise FixedQaError(f"unsupported Truebones rest mode {method!r}")


def _source_reference(
    clip: Mapping[str, Any],
    rig: Mapping[str, Any],
    skeleton: SkeletonData,
    *,
    rest_cache: dict[str, ParsedBvhMotion],
) -> SourceReference:
    family = str(clip["source"]["family"])
    if family == "motionstreamer272":
        parsed = parse_motionstreamer272_source(
            clip["source"]["path"],
            joint_names=rig["joint_map"]["btjd_joint_names"],
            parents=rig["joint_map"]["btjd_parents"],
            neutral_model_path=rig["rest_pose"]["source_path"],
        )
        return SourceReference(
            parsed=parsed,
            root_positions=np.asarray(parsed.source_positions[:, 0], dtype=np.float64),
            local_rotations=np.asarray(parsed.local_rotations, dtype=np.float64),
        )
    if family != "truebones":
        raise FixedQaError(f"unsupported source family {family!r}")
    rest_path = str(Path(rig["rest_pose"]["source_path"]).expanduser().resolve())
    if rest_path not in rest_cache:
        rest_cache[rest_path] = parse_bvh_numeric(rest_path)
    parsed = parse_bvh_source(
        clip["source"]["path"],
        retained_names=rig["joint_map"]["btjd_joint_names"],
        retained_parents=rig["joint_map"]["btjd_parents"],
        expected_rotation_kinds=rig["joint_map"]["rotation_source_kind"],
        frame_slice=clip["source"]["slice_frames"],
        rest_path=rest_path,
        rest_mode=_rest_mode(rig),
        parsed_rest=rest_cache[rest_path],
        family="truebones",
    )
    canonical_global = np.einsum(
        "ab,tjbc,dc->tjad",
        skeleton.source_to_canonical_C,
        parsed.global_rotations,
        skeleton.source_to_canonical_C,
    )
    canonical_root = skeleton.source_to_canonical_alpha * (
        (np.asarray(parsed.source_positions[:, 0], dtype=np.float64)
         - skeleton.source_to_canonical_o)
        @ skeleton.source_to_canonical_C.T
    )
    return SourceReference(
        parsed=parsed,
        root_positions=canonical_root,
        local_rotations=_global_to_local(skeleton.parents, canonical_global),
    )


def reconstruct_aligned_source_reference(
    *,
    parent_clip: Mapping[str, Any],
    parent_rig: Mapping[str, Any],
    skeleton: SkeletonData,
    fps_target: float,
    origin_xz: np.ndarray,
    rest_cache: dict[str, ParsedBvhMotion] | None = None,
) -> AlignedSourceReference:
    """Rebuild the independent source route used by numeric and visual QA."""
    cache = {} if rest_cache is None else rest_cache
    source = _source_reference(
        parent_clip, parent_rig, skeleton, rest_cache=cache
    )
    roots, local, mode = _resample(
        source.root_positions,
        source.local_rotations,
        fps_src=float(source.parsed.fps),
        fps_target=float(fps_target),
    )
    global_rotations = _local_to_global(skeleton.parents, local)
    positions_absolute = _fk(
        skeleton.parents,
        roots,
        global_rotations,
        skeleton.offset_parent_local,
    )
    positions_absolute[..., 1] -= float(np.min(positions_absolute[..., 1]))
    origin = np.asarray(origin_xz, dtype=np.float64)
    _require(origin.shape == (2,) and np.isfinite(origin).all(), "invalid origin_xz")
    positions_clip = positions_absolute.copy()
    positions_clip[..., 0] -= origin[0]
    positions_clip[..., 2] -= origin[1]
    return AlignedSourceReference(
        positions_clip=_finite("aligned source positions", positions_clip),
        positions_absolute=_finite("absolute source positions", positions_absolute),
        global_rotations=_finite("aligned source rotations", global_rotations),
        resample_mode=mode,
    )


def _validate_skeleton(path: Path, expected_sha: str) -> tuple[SkeletonData, dict[str, float]]:
    _require(path.is_file(), f"missing skeleton {path}")
    _require(_sha256_file(path) == expected_sha, f"skeleton hash drifted: {path}")
    skeleton = load_skeleton(path)
    parents = _validate_tree(skeleton.parents)
    _require(
        not any(name.strip().upper() in FORBIDDEN_VIRTUAL_NAMES for name in skeleton.joint_names),
        f"{skeleton.rig_id}: virtual WORLD/control joint found",
    )
    _require(
        set(skeleton.rotation_source_kind.astype(str)).issubset(
            {"animated_dof", "fixed_dof"}
        ),
        f"{skeleton.rig_id}: invalid rotation source kind",
    )
    C = skeleton.source_to_canonical_C
    basis_error = float(np.max(np.abs(C.T @ C - np.eye(3, dtype=np.float64))))
    determinant_error = abs(abs(float(np.linalg.det(C))) - 1.0)
    _require(basis_error <= 1e-12, f"{skeleton.rig_id}: basis error {basis_error}")
    _require(
        determinant_error <= 1e-12,
        f"{skeleton.rig_id}: basis determinant error {determinant_error}",
    )
    orthogonality, determinant_min, determinant_max = _rotation_diagnostics(
        skeleton.R_rest_global
    )
    _require(
        orthogonality <= 1e-12
        and abs(determinant_min - 1.0) <= 1e-12
        and abs(determinant_max - 1.0) <= 1e-12,
        f"{skeleton.rig_id}: rest rotations are not SO(3)",
    )
    expected_local = np.empty_like(skeleton.R_rest_local)
    expected_local[0] = skeleton.R_rest_global[0]
    expected_offsets = np.zeros_like(skeleton.offset_parent_local)
    for child in range(1, len(parents)):
        parent = int(parents[child])
        expected_local[child] = (
            skeleton.R_rest_global[parent].T @ skeleton.R_rest_global[child]
        )
        expected_offsets[child] = skeleton.R_rest_global[parent].T @ (
            skeleton.P_rest_global[child] - skeleton.P_rest_global[parent]
        )
    rest_local_error = float(np.max(np.abs(expected_local - skeleton.R_rest_local)))
    offset_error = float(
        np.max(np.abs(expected_offsets - skeleton.offset_parent_local))
    )
    _require(rest_local_error <= 1e-12, f"rest-local error {rest_local_error}")
    _require(offset_error <= 1e-12, f"rest-offset error {offset_error}")
    rest_fk = _fk(
        parents,
        skeleton.P_rest_global[0][None],
        skeleton.R_rest_global[None],
        skeleton.offset_parent_local,
    )[0]
    rest_fk_norm = float(
        np.max(np.linalg.norm(rest_fk - skeleton.P_rest_global, axis=-1))
        / skeleton.s_rig
    )
    _require(rest_fk_norm <= 1e-10, f"rest FK error {rest_fk_norm}")
    rest_delta = np.matmul(
        skeleton.R_rest_global,
        np.swapaxes(skeleton.R_rest_global, -1, -2),
    )
    rest_d6 = np.concatenate((rest_delta[..., :, 0], rest_delta[..., :, 1]), axis=-1)
    identity_d6 = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    rest_identity_f64 = float(np.max(np.abs(rest_d6 - identity_d6)))
    rest_identity_f32 = float(
        np.max(np.abs(rest_d6.astype(np.float32).astype(np.float64) - identity_d6))
    )
    _require(rest_identity_f64 <= 1e-10, f"rest identity f64 {rest_identity_f64}")
    _require(rest_identity_f32 <= 2e-6, f"rest identity f32 {rest_identity_f32}")
    scale = float(np.linalg.norm(np.ptp(skeleton.P_rest_global, axis=0)))
    _require(
        abs(scale - skeleton.s_rig) <= 1e-12 * max(1.0, scale),
        f"s_rig mismatch {skeleton.s_rig} != {scale}",
    )
    rest_forward = (
        skeleton.R_rest_global[skeleton.heading_carrier_joint]
        @ skeleton.u_forward_local
    )
    forward_error = float(
        np.max(np.abs(rest_forward - np.asarray([0.0, 0.0, 1.0])))
    )
    _require(forward_error <= 1e-6, f"rest forward is not +Z: {forward_error}")
    return skeleton, {
        "basis_orthogonality_max_abs": basis_error,
        "basis_abs_determinant_error": determinant_error,
        "rest_rotation_orthogonality_max_abs": orthogonality,
        "rest_rotation_determinant_min": determinant_min,
        "rest_rotation_determinant_max": determinant_max,
        "rest_local_max_abs": rest_local_error,
        "rest_offset_max_abs": offset_error,
        "rest_fk_max_norm": rest_fk_norm,
        "rest_identity_d6_float64_max_abs": rest_identity_f64,
        "rest_identity_d6_float32_max_abs": rest_identity_f32,
        "rest_forward_plus_z_max_abs": forward_error,
    }


def _validate_motion(
    *,
    root: Path,
    manifest: Mapping[str, Any],
    qa: Mapping[str, Any],
    parent_clip: Mapping[str, Any],
    parent_rig: Mapping[str, Any],
    skeleton: SkeletonData,
    encoder_config: Mapping[str, Any],
    rest_cache: dict[str, ParsedBvhMotion],
) -> dict[str, Any]:
    clip_id = str(manifest["clip_id"])
    _validate_topology_distance_bucket(
        clip_id=clip_id,
        manifest=manifest,
        qa=qa,
        parent_clip=parent_clip,
    )
    motion_path = _resolve_generation_path(
        root, manifest["motion_relpath"], label=f"{clip_id}: motion"
    )
    _require(motion_path.is_file(), f"{clip_id}: motion is missing")
    motion_sha = _sha256_file(motion_path)
    _require(motion_sha == manifest["motion_sha256"], f"{clip_id}: motion hash drifted")
    _require(motion_sha == qa["motion_sha256"], f"{clip_id}: QA motion hash drifted")
    fps_target = float(encoder_config["fps_target"])
    payload = load_motion_npz(motion_path, expected_fps_target=fps_target)
    motion = np.asarray(payload["motion"], dtype=np.float64)
    heading_valid = np.asarray(payload["heading_valid"], dtype=bool)
    origin_xz = np.asarray(payload["origin_xz"], dtype=np.float64)
    T, J, D = motion.shape
    _require(D == 17 and J == len(skeleton.parents), f"{clip_id}: shape drifted")
    _require(T == int(manifest["T_target"]), f"{clip_id}: T_target drifted")
    _require(J == int(manifest["J_phys"]), f"{clip_id}: J_phys drifted")
    _require(payload["clip_id"] == clip_id, f"{clip_id}: embedded clip id drifted")
    _require(payload["rig_id"] == skeleton.rig_id, f"{clip_id}: embedded rig id drifted")
    mask_mismatch = _validate_loader_masks(
        T=T,
        J=J,
        parents=skeleton.parents,
        rotation_source_kind=skeleton.rotation_source_kind,
        heading_valid=heading_valid,
    )
    _require(
        bool(np.all(motion[:, 1:, 13:17] == 0.0)),
        f"{clip_id}: non-root channels 13:17 are not exact zero",
    )
    _require(
        bool(np.all(np.logical_or(motion[..., 12] == 0.0, motion[..., 12] == 1.0))),
        f"{clip_id}: contact is not binary",
    )
    _require(
        bool(np.all(motion[~heading_valid, 0, 15:17] == 0.0)),
        f"{clip_id}: invalid heading sentinel is not zero",
    )
    if np.any(heading_valid):
        heading_unit_error = float(
            np.max(
                np.abs(
                    np.linalg.norm(motion[heading_valid, 0, 15:17], axis=-1) - 1.0
                )
            )
        )
    else:
        heading_unit_error = 0.0
    _require(heading_unit_error <= 2e-6, f"{clip_id}: heading unit error")

    direct = motion[..., 0:3].copy()
    direct[..., 0] += motion[:, 0, 13][:, None]
    direct[..., 2] += motion[:, 0, 14][:, None]
    delta = independent_decode_column_cont6d(motion[..., 3:9])
    decoded_global_raw = np.matmul(delta, skeleton.R_rest_global[None])
    decoded_global = np.empty_like(decoded_global_raw)
    kinds = skeleton.rotation_source_kind.astype(str)
    for joint in range(J):
        if kinds[joint] == "animated_dof":
            decoded_global[:, joint] = decoded_global_raw[:, joint]
        elif joint == 0:
            decoded_global[:, joint] = skeleton.R_rest_local[0]
        else:
            decoded_global[:, joint] = np.matmul(
                decoded_global[:, int(skeleton.parents[joint])],
                skeleton.R_rest_local[joint],
            )
    fk = _fk(
        skeleton.parents,
        direct[:, 0],
        decoded_global,
        skeleton.offset_parent_local,
    )
    direct_fk_errors = np.linalg.norm(direct - fk, axis=-1) / skeleton.s_rig
    direct_fk_max = float(np.max(direct_fk_errors))
    direct_fk_p99 = float(np.percentile(direct_fk_errors, 99))
    direct_fk_mpjpe = float(np.mean(direct_fk_errors))
    _require(direct_fk_max <= 1e-4, f"{clip_id}: direct/FK {direct_fk_max}")

    expected_velocity = _velocity(direct, fps_target)
    velocity_error = float(np.max(np.abs(motion[..., 9:12] - expected_velocity)))
    velocity_error_norm_fps = velocity_error / (skeleton.s_rig * fps_target)
    _require(
        velocity_error_norm_fps <= 1e-5,
        f"{clip_id}: velocity error {velocity_error_norm_fps}",
    )
    rest_lengths = np.asarray(
        [
            np.linalg.norm(
                skeleton.P_rest_global[child]
                - skeleton.P_rest_global[int(skeleton.parents[child])]
            )
            for child in range(1, J)
        ],
        dtype=np.float64,
    )
    edge_lengths = np.stack(
        [
            np.linalg.norm(
                direct[:, child] - direct[:, int(skeleton.parents[child])], axis=-1
            )
            for child in range(1, J)
        ],
        axis=-1,
    )
    rigid_edge_max = float(
        np.max(np.abs(edge_lengths - rest_lengths[None])) / skeleton.s_rig
    )
    _require(rigid_edge_max <= 1e-4, f"{clip_id}: rigid edge {rigid_edge_max}")

    source = _source_reference(
        parent_clip, parent_rig, skeleton, rest_cache=rest_cache
    )
    parsed_error = np.linalg.norm(
        source.parsed.fk_positions - source.parsed.source_positions, axis=-1
    )
    source_fk_max = float(np.max(parsed_error) / source.parsed.s_rig)
    _require(
        source_fk_max <= SOURCE_FK_MAX_NORM[str(manifest["source_family"])],
        f"{clip_id}: source FK {source_fk_max}",
    )
    roots_target, local_target, resample_mode = _resample(
        source.root_positions,
        source.local_rotations,
        fps_src=float(source.parsed.fps),
        fps_target=fps_target,
    )
    _require(resample_mode == manifest["resample_mode"], f"{clip_id}: resample mode")
    _require(len(roots_target) == T, f"{clip_id}: resample length drifted")
    reference_global = _local_to_global(skeleton.parents, local_target)
    reference_positions_ungrounded = _fk(
        skeleton.parents,
        roots_target,
        reference_global,
        skeleton.offset_parent_local,
    )
    ground_shift = -float(np.min(reference_positions_ungrounded[..., 1]))
    reference_positions = reference_positions_ungrounded.copy()
    reference_positions[..., 1] += ground_shift
    restored_direct = direct.copy()
    restored_direct[..., 0] += origin_xz[0]
    restored_direct[..., 2] += origin_xz[1]
    source_position_error = float(
        np.max(np.abs(restored_direct - reference_positions)) / skeleton.s_rig
    )
    _require(
        source_position_error <= 1e-5,
        f"{clip_id}: source position roundtrip {source_position_error}",
    )
    rotation_angles = _rotation_angle_error(reference_global, decoded_global)
    source_rotation_geodesic_max = float(np.max(rotation_angles))
    source_rotation_geodesic_mean = float(np.mean(rotation_angles))
    _require(
        source_rotation_geodesic_max <= 2e-6,
        f"{clip_id}: source rotation geodesic {source_rotation_geodesic_max}",
    )

    expected_smooth, smooth_mode = _smooth_root(
        reference_positions[:, 0][:, [0, 2]],
        fps=fps_target,
        smoother=encoder_config["smoother"],
    )
    stored_smooth_absolute = motion[:, 0, 13:15] + origin_xz[None]
    smooth_error = float(
        np.max(np.abs(stored_smooth_absolute - expected_smooth)) / skeleton.s_rig
    )
    _require(smooth_error <= 1e-5, f"{clip_id}: smooth-root {smooth_error}")
    expected_q = reference_positions.copy()
    expected_q[..., 0] -= expected_smooth[:, None, 0]
    expected_q[..., 2] -= expected_smooth[:, None, 1]
    q_error = float(
        np.max(np.abs(motion[..., 0:3] - expected_q)) / skeleton.s_rig
    )
    _require(q_error <= 1e-5, f"{clip_id}: q source roundtrip {q_error}")

    reference_velocity = _velocity(reference_positions, fps_target)
    source_velocity_error = float(
        np.max(np.abs(motion[..., 9:12] - reference_velocity))
        / (skeleton.s_rig * fps_target)
    )
    _require(
        source_velocity_error <= 1e-5,
        f"{clip_id}: source velocity {source_velocity_error}",
    )
    contact_config = encoder_config["contact"]
    height_norm = reference_positions[..., 1] / skeleton.s_rig
    speed_norm = np.linalg.norm(reference_velocity, axis=-1) / skeleton.s_rig
    expected_contact = (
        (height_norm <= float(contact_config["tau_h"]))
        & (speed_norm <= float(contact_config["tau_v"]))
    )
    if T >= 2:
        expected_contact[-1] = expected_contact[-2]
    contact_mismatch = int(np.count_nonzero(motion[..., 12].astype(bool) != expected_contact))
    _require(contact_mismatch == 0, f"{clip_id}: contact mismatch {contact_mismatch}")
    expected_heading, expected_heading_valid, horizontal_norm = _heading(
        reference_global,
        carrier=skeleton.heading_carrier_joint,
        forward_local=skeleton.u_forward_local,
        epsilon=float(encoder_config["heading"]["eps_h"]),
    )
    heading_mask_mismatch = int(np.count_nonzero(heading_valid != expected_heading_valid))
    heading_error = float(np.max(np.abs(motion[:, 0, 15:17] - expected_heading)))
    _require(heading_mask_mismatch == 0, f"{clip_id}: heading mask mismatch")
    _require(heading_error <= 2e-6, f"{clip_id}: heading error {heading_error}")

    # Exercise the actual loader yaw path against independent expected direct,
    # rotation, heading, and invariant contact values.
    phi = 0.371
    augmented = yaw_augment(
        motion,
        heading_valid,
        R_rest_global=skeleton.R_rest_global,
        phi=phi,
    )
    cosine, sine = math.cos(phi), math.sin(phi)
    yaw = np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )
    direct_augmented = augmented[..., 0:3].copy()
    direct_augmented[..., 0] += augmented[:, 0, 13][:, None]
    direct_augmented[..., 2] += augmented[:, 0, 14][:, None]
    yaw_position_error = float(
        np.max(
            np.abs(
                direct_augmented - np.einsum("ab,tjb->tja", yaw, direct)
            )
        )
        / skeleton.s_rig
    )
    augmented_delta = independent_decode_column_cont6d(augmented[..., 3:9])
    augmented_global = np.matmul(augmented_delta, skeleton.R_rest_global[None])
    yaw_rotation_error = float(
        np.max(
            np.linalg.norm(
                augmented_global - np.einsum("ab,tjbc->tjac", yaw, decoded_global_raw),
                axis=(-2, -1),
            )
        )
    )
    heading_rot2 = np.asarray(
        [[cosine, -sine], [sine, cosine]], dtype=np.float64
    )
    expected_yaw_heading = np.zeros((T, 2), dtype=np.float64)
    expected_yaw_heading[heading_valid] = np.einsum(
        "ab,tb->ta", heading_rot2, motion[heading_valid, 0, 15:17]
    )
    yaw_heading_error = float(
        np.max(np.abs(augmented[:, 0, 15:17] - expected_yaw_heading))
    )
    _require(yaw_position_error <= 1e-5, f"{clip_id}: yaw position")
    _require(yaw_rotation_error <= 2e-6, f"{clip_id}: yaw rotation")
    _require(yaw_heading_error <= 2e-6, f"{clip_id}: yaw heading")
    _require(
        bool(np.array_equal(augmented[..., 12], motion[..., 12])),
        f"{clip_id}: yaw changed contact",
    )

    edited = motion.copy()
    middle = T // 2
    edited[middle, min(1, J - 1), 0] += 0.123
    edited_direct = edited[..., 0:3].copy()
    edited_direct[..., 0] += edited[:, 0, 13][:, None]
    edited_direct[..., 2] += edited[:, 0, 14][:, None]
    other_frames = np.ones(T, dtype=bool)
    other_frames[middle] = False
    locality_error = float(np.max(np.abs(edited_direct[other_frames] - direct[other_frames])))
    _require(locality_error == 0.0, f"{clip_id}: decode is not frame-local")

    source_global = _local_to_global(skeleton.parents, source.local_rotations)
    source_positions = _fk(
        skeleton.parents,
        source.root_positions,
        source_global,
        skeleton.offset_parent_local,
    )
    source_acceleration = _acceleration_rms(
        source_positions, source.parsed.fps, skeleton.s_rig
    )
    target_acceleration = _acceleration_rms(
        reference_positions_ungrounded, fps_target, skeleton.s_rig
    )
    acceleration_jitter_ratio = (
        target_acceleration / source_acceleration
        if source_acceleration > 1e-12
        else None
    )
    return {
        "clip_id": clip_id,
        "rig_id": skeleton.rig_id,
        "source_family": manifest["source_family"],
        "topology_family": manifest["topology_family"],
        "topology_distance_bucket": manifest["topology_distance_bucket"],
        "family_role": manifest["family_role"],
        "split": manifest["split"],
        "calibration_eligible": bool(manifest["calibration_eligible"]),
        "status": "pass",
        "T": T,
        "J": J,
        "metrics": {
            "source_parser_fk_max_norm": source_fk_max,
            "source_position_roundtrip_max_norm": source_position_error,
            "source_global_rotation_geodesic_max_rad": source_rotation_geodesic_max,
            "source_global_rotation_geodesic_mean_rad": source_rotation_geodesic_mean,
            "direct_vs_fk_max_norm": direct_fk_max,
            "direct_vs_fk_p99_norm": direct_fk_p99,
            "direct_vs_fk_mpjpe_norm": direct_fk_mpjpe,
            "velocity_max_norm_fps": velocity_error_norm_fps,
            "source_velocity_max_norm_fps": source_velocity_error,
            "rigid_edge_max_norm": rigid_edge_max,
            "smooth_root_max_norm": smooth_error,
            "q_source_roundtrip_max_norm": q_error,
            "heading_max_abs": heading_error,
            "heading_unit_max_abs": heading_unit_error,
            "heading_valid_fraction": float(np.mean(heading_valid)),
            "heading_invalid_longest_run": _longest_false_run(heading_valid),
            "heading_horizontal_min": float(np.min(horizontal_norm)),
            "contact_positive_rate": float(np.mean(expected_contact)),
            "contact_mismatch_count": contact_mismatch,
            "mask_mismatch_count": mask_mismatch,
            "ground_min_y_norm": float(
                np.min(reference_positions[..., 1]) / skeleton.s_rig
            ),
            "smooth_root_residual_rms_norm": float(
                np.sqrt(
                    np.mean(
                        np.square(
                            (
                                reference_positions[:, 0][:, [0, 2]]
                                - expected_smooth
                            )
                            / skeleton.s_rig
                        )
                    )
                )
            ),
            "source_acceleration_rms_norm": source_acceleration,
            "target_acceleration_rms_norm": target_acceleration,
            "resample_acceleration_jitter_ratio": acceleration_jitter_ratio,
            "yaw_position_max_norm": yaw_position_error,
            "yaw_rotation_frobenius_max": yaw_rotation_error,
            "yaw_heading_max_abs": yaw_heading_error,
            "locality_other_frame_max_abs": locality_error,
        },
        "smoother_mode": smooth_mode,
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
    summary: dict[str, Any] = {}
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
            summary[name] = {
                "count": int(values.size),
                "min": float(np.min(values)),
                "median": float(np.median(values)),
                "p99": float(np.percentile(values, 99)),
                "max": float(np.max(values)),
                "mean": float(np.mean(values)),
            }
    return summary


def validate_prototype(prototype_root: str | Path) -> dict[str, Any]:
    """Validate one immutable prototype generation without mutating it."""
    root = Path(prototype_root).expanduser().resolve()
    generation = _load_json(root / "generation.json")
    summary = _load_json(root / "qa/encoder_summary.json")
    selection = _load_json(root / "manifests/prototype_selection.json")
    encoder_config = _load_json(root / "config/encoder_candidate.json")
    manifest_records = _load_jsonl(root / "manifests/clips.jsonl")
    qa_records = _load_jsonl(root / "qa/encoder_qa.jsonl")
    global_algebra_metrics = _validate_global_algebra()
    _require(generation["generation_id"] == root.name, "generation id/path mismatch")
    _require(generation["status"] == summary["status"], "generation status drifted")

    expected_files = generation.get("files")
    _require(isinstance(expected_files, Mapping), "generation file manifest is absent")
    actual_relpaths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() != "generation.json"
    }
    _require(actual_relpaths == set(expected_files), "generation file set drifted")
    for relpath, payload in expected_files.items():
        path = root / relpath
        _require(path.stat().st_size == int(payload["size_bytes"]), f"size drift {relpath}")
        _require(_sha256_file(path) == payload["sha256"], f"hash drift {relpath}")

    selection_core = {
        "selection_authority": selection["selection_authority"],
        "selection_counts": selection["selection_counts"],
        "selected": selection["selected"],
    }
    selection_sha = hashlib.sha256(_canonical_json(selection_core)).hexdigest()
    _require(selection_sha == selection["selection_sha256"], "selection hash drifted")
    _require(selection_sha == generation["selection_sha256"], "generation selection hash")
    parent_root = Path(
        selection["selection_authority"]["parent_manifest_root"]
    ).expanduser().resolve()
    _require(
        _sha256_file(parent_root / "clips.jsonl")
        == selection["selection_authority"]["parent_clips_jsonl_sha256"],
        "parent clips manifest hash drifted",
    )
    _require(
        _sha256_file(parent_root / "prototype_candidates.json")
        == selection["selection_authority"]["parent_prototype_candidates_sha256"],
        "parent prototype candidates hash drifted",
    )
    parent_clips = {
        record["clip_id"]: record for record in _load_jsonl(parent_root / "clips.jsonl")
    }
    parent_rigs = {
        record["rig_id"]: record for record in _load_jsonl(parent_root / "rigs.jsonl")
    }
    manifests = {record["clip_id"]: record for record in manifest_records}
    qa = {record["clip_id"]: record for record in qa_records}
    _require(len(manifests) == len(manifest_records), "duplicate manifest clip ids")
    _require(len(qa) == len(qa_records), "duplicate QA clip ids")
    selected_ids = [record["clip_id"] for record in selection["selected"]]
    _require(len(selected_ids) == len(set(selected_ids)), "selection ids are duplicated")
    _require(set(selected_ids) == set(manifests) == set(qa), "prototype scopes differ")
    _require(len(selected_ids) == int(summary["selected_count"]), "selected count drift")
    _require(selection.get("held_data_used_for_calibration") is False, "held leakage")

    skeleton_paths: dict[str, tuple[Path, str]] = {}
    for record in manifest_records:
        rig_id = str(record["rig_id"])
        reference = (
            _resolve_generation_path(
                root,
                record["skeleton_relpath"],
                label=f"{rig_id}: skeleton",
            ),
            str(record["skeleton_sha256"]),
        )
        if rig_id in skeleton_paths:
            _require(
                skeleton_paths[rig_id] == reference,
                f"{rig_id}: inconsistent skeleton references",
            )
        else:
            skeleton_paths[rig_id] = reference
    skeletons: dict[str, SkeletonData] = {}
    skeleton_metrics: dict[str, dict[str, float]] = {}
    for rig_id, (path, expected_sha) in sorted(skeleton_paths.items()):
        skeleton, metrics = _validate_skeleton(path, expected_sha)
        _require(skeleton.rig_id == rig_id, f"skeleton identity drifted for {rig_id}")
        skeletons[rig_id] = skeleton
        skeleton_metrics[rig_id] = metrics

    rest_cache: dict[str, ParsedBvhMotion] = {}
    clip_results: list[dict[str, Any]] = []
    for index, clip_id in enumerate(selected_ids, start=1):
        manifest = manifests[clip_id]
        qa_record = qa[clip_id]
        try:
            _require(manifest.get("status") == "accept", f"{clip_id}: not accepted")
            _require(qa_record.get("status") == "pass", f"{clip_id}: QA did not pass")
            provenance = qa_record.get("provenance")
            _require(isinstance(provenance, Mapping), f"{clip_id}: provenance absent")
            _require("skeleton_path" not in provenance, f"{clip_id}: staging path leaked")
            _require(
                provenance.get("skeleton_relpath") == manifest["skeleton_relpath"],
                f"{clip_id}: skeleton provenance relpath drifted",
            )
            _require(
                provenance.get("skeleton_resolution")
                == "generation_relative_relpath_plus_sha256",
                f"{clip_id}: skeleton resolution authority drifted",
            )
            result = _validate_motion(
                root=root,
                manifest=manifest,
                qa=qa_record,
                parent_clip=parent_clips[clip_id],
                parent_rig=parent_rigs[str(manifest["rig_id"])],
                skeleton=skeletons[str(manifest["rig_id"])],
                encoder_config=encoder_config,
                rest_cache=rest_cache,
            )
        except Exception as exc:  # noqa: BLE001
            result = {
                "clip_id": clip_id,
                "rig_id": manifest.get("rig_id"),
                "source_family": manifest.get("source_family"),
                "topology_family": manifest.get("topology_family"),
                "topology_distance_bucket": manifest.get(
                    "topology_distance_bucket"
                ),
                "family_role": manifest.get("family_role"),
                "split": manifest.get("split"),
                "calibration_eligible": manifest.get("calibration_eligible"),
                "status": "fail",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        clip_results.append(result)
        if index % 25 == 0 or index == len(selected_ids):
            print(f"[ktjd17-fixed-qa] validated {index}/{len(selected_ids)}", flush=True)

    failures = [record for record in clip_results if record["status"] != "pass"]
    calibration_records = [
        record
        for record in clip_results
        if record["status"] == "pass" and record["calibration_eligible"] is True
    ]
    held_records = [
        record
        for record in clip_results
        if record["status"] == "pass" and record["calibration_eligible"] is False
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in clip_results:
        for key in (
            f"source_family:{record['source_family']}",
            f"topology_family:{record['topology_family']}",
            f"topology_distance_bucket:{record['topology_distance_bucket']}",
            f"family_role:{record['family_role']}",
            f"rig:{record['rig_id']}",
        ):
            grouped[key].append(record)
    stratified = {
        key: {
            "count": len(records),
            "pass": sum(record["status"] == "pass" for record in records),
            "fail": sum(record["status"] != "pass" for record in records),
            "metrics": _metric_summary(records),
        }
        for key, records in sorted(grouped.items())
    }
    return {
        "qa_version": FIXED_QA_VERSION,
        "prototype_root": str(root),
        "generation_id": generation["generation_id"],
        "selection_sha256": selection_sha,
        "read_only": True,
        "calibration_or_schema_written": False,
        "status": "pass" if not failures else "fail",
        "clip_count": len(clip_results),
        "pass_count": len(clip_results) - len(failures),
        "fail_count": len(failures),
        "calibration_eligible_pass_count": len(calibration_records),
        "held_read_only_pass_count": len(held_records),
        "skeleton_count": len(skeletons),
        "J_phys_max": max(len(skeleton.parents) for skeleton in skeletons.values()),
        "T_max_observed": max(int(record.get("T", 0)) for record in clip_results),
        "fixed_thresholds": {
            "source_fk_max_norm": SOURCE_FK_MAX_NORM,
            "direct_vs_fk_max_norm": 1e-4,
            "source_position_roundtrip_max_norm": 1e-5,
            "source_global_rotation_geodesic_max_rad": 2e-6,
            "velocity_max_norm_fps": 1e-5,
            "rigid_edge_max_norm": 1e-4,
            "smooth_root_max_norm": 1e-5,
            "heading_max_abs": 2e-6,
            "yaw_position_max_norm": 1e-5,
            "yaw_rotation_frobenius_max": 2e-6,
            "yaw_heading_max_abs": 2e-6,
        },
        "global_algebra_metrics": global_algebra_metrics,
        "metrics_all": _metric_summary(clip_results),
        "metrics_train_calibration_only": _metric_summary(calibration_records),
        "metrics_held_read_only": _metric_summary(held_records),
        "skeleton_metrics": skeleton_metrics,
        "stratified": stratified,
        "failures": failures,
        "clips": clip_results,
    }
