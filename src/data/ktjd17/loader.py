"""Loader-only masks, crop, yaw, normalization, and padding for KTJD-17."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np

from .codec import (
    Ktjd17CodecError,
    decode_column_cont6d,
    encode_column_cont6d,
    require_float64_finite,
    require_so3,
    validate_parent_tree,
)


@dataclasses.dataclass(frozen=True)
class DerivedMasks:
    frame_mask: np.ndarray
    joint_mask: np.ndarray
    channel_valid_mask: np.ndarray
    rotation_supervised: np.ndarray
    fixed_rotation_mask: np.ndarray
    contact_supervised: np.ndarray
    child_edge_valid: np.ndarray
    heading_valid: np.ndarray


@dataclasses.dataclass(frozen=True)
class ModelView:
    motion: np.ndarray
    masks: DerivedMasks
    T_valid: int
    J_phys: int
    crop_start: int
    yaw_radians: float


def derive_masks(
    *,
    T_valid: int,
    J_phys: int,
    T_max: int,
    J_max: int,
    parents: np.ndarray,
    rotation_source_kind: np.ndarray,
    heading_valid: np.ndarray,
) -> DerivedMasks:
    if not 0 < T_valid <= T_max or not 0 < J_phys <= J_max:
        raise Ktjd17CodecError(
            f"invalid valid/padded extents T={T_valid}/{T_max}, J={J_phys}/{J_max}"
        )
    parent_array = validate_parent_tree(parents, J_phys)
    kinds = np.asarray(rotation_source_kind).astype(str)
    if kinds.shape != (J_phys,) or not set(kinds).issubset(
        {"animated_dof", "fixed_dof"}
    ):
        raise Ktjd17CodecError("invalid rotation_source_kind payload")
    heading = np.asarray(heading_valid, dtype=bool)
    if heading.shape != (T_valid,):
        raise Ktjd17CodecError(
            f"heading_valid must have shape ({T_valid},), got {heading.shape}"
        )
    frame_mask = np.zeros(T_max, dtype=bool)
    frame_mask[:T_valid] = True
    joint_mask = np.zeros(J_max, dtype=bool)
    joint_mask[:J_phys] = True
    channel_valid = np.zeros((J_max, 17), dtype=bool)
    channel_valid[:J_phys, :13] = True
    channel_valid[0, 13:17] = True
    rotation_supervised = np.zeros(J_max, dtype=bool)
    fixed_rotation_mask = np.zeros(J_max, dtype=bool)
    rotation_supervised[:J_phys] = kinds == "animated_dof"
    fixed_rotation_mask[:J_phys] = kinds == "fixed_dof"
    contact_supervised = joint_mask.copy()
    child_edge_valid = np.zeros(J_max, dtype=bool)
    for child in range(1, J_phys):
        child_edge_valid[child] = joint_mask[child] and joint_mask[
            int(parent_array[child])
        ]
    heading_padded = np.zeros(T_max, dtype=bool)
    heading_padded[:T_valid] = heading
    return DerivedMasks(
        frame_mask=frame_mask,
        joint_mask=joint_mask,
        channel_valid_mask=channel_valid,
        rotation_supervised=rotation_supervised,
        fixed_rotation_mask=fixed_rotation_mask,
        contact_supervised=contact_supervised,
        child_edge_valid=child_edge_valid,
        heading_valid=heading_padded,
    )


def crop_full_clip(
    motion: np.ndarray,
    heading_valid: np.ndarray,
    *,
    start: int,
    length: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = require_float64_finite("motion", np.asarray(motion))
    heading = np.asarray(heading_valid, dtype=bool)
    if values.ndim != 3 or values.shape[-1] != 17 or heading.shape != (
        values.shape[0],
    ):
        raise Ktjd17CodecError("invalid motion/heading payload for crop")
    begin = int(start)
    count = int(length)
    if begin < 0 or count <= 0 or begin + count > values.shape[0]:
        raise Ktjd17CodecError(
            f"invalid crop [{begin}:{begin+count}) for T={values.shape[0]}"
        )
    cropped = values[begin : begin + count].copy()
    # The crop origin changes only the stored smooth-root trajectory.  q, d6,
    # velocity, contact, and heading retain their semantic values.
    cropped[:, 0, 13:15] -= cropped[0, 0, 13:15]
    return cropped, heading[begin : begin + count].copy()


def yaw_matrix(phi: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angle = float(phi)
    if not math.isfinite(angle):
        raise Ktjd17CodecError(f"yaw must be finite, got {phi!r}")
    cosine, sine = math.cos(angle), math.sin(angle)
    Y = np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )
    Y_xz = np.asarray([[cosine, sine], [-sine, cosine]], dtype=np.float64)
    heading_rot2 = np.asarray(
        [[cosine, -sine], [sine, cosine]], dtype=np.float64
    )
    return Y, Y_xz, heading_rot2


def yaw_augment(
    motion: np.ndarray,
    heading_valid: np.ndarray,
    *,
    R_rest_global: np.ndarray,
    phi: float,
) -> np.ndarray:
    values = require_float64_finite("motion", np.asarray(motion)).copy()
    heading_mask = np.asarray(heading_valid, dtype=bool)
    if values.ndim != 3 or values.shape[-1] != 17 or heading_mask.shape != (
        values.shape[0],
    ):
        raise Ktjd17CodecError("invalid motion/heading payload for yaw")
    rest = require_so3("R_rest_global", np.asarray(R_rest_global))
    if rest.shape != (values.shape[1], 3, 3):
        raise Ktjd17CodecError("rest rotations do not match motion J")
    Y, Y_xz, heading_rot2 = yaw_matrix(phi)
    values[..., 0:3] = np.einsum("ab,tjb->tja", Y, values[..., 0:3])
    values[..., 9:12] = np.einsum("ab,tjb->tja", Y, values[..., 9:12])
    values[:, 0, 13:15] = np.einsum(
        "ab,tb->ta", Y_xz, values[:, 0, 13:15]
    )
    delta = decode_column_cont6d(values[..., 3:9])
    global_rotations = np.matmul(delta, rest[None])
    rotated_global = np.einsum("ab,tjbc->tjac", Y, global_rotations)
    rotated_delta = np.matmul(rotated_global, np.swapaxes(rest, -1, -2)[None])
    values[..., 3:9] = encode_column_cont6d(rotated_delta)
    values[:, 0, 15:17] = 0.0
    values[heading_mask, 0, 15:17] = np.einsum(
        "ab,tb->ta", heading_rot2, motion[heading_mask, 0, 15:17]
    )
    if np.any(values[~heading_mask, 0, 15:17] != 0.0):
        raise Ktjd17CodecError("invalid headings must remain exact zero after yaw")
    if np.any(values[:, 1:, 13:17] != 0.0):
        raise Ktjd17CodecError("non-root root-global channels changed under yaw")
    return require_float64_finite("yaw_augmented_motion", values)


def normalize_model_motion(
    motion: np.ndarray,
    *,
    s_rig: float,
    gains: np.ndarray,
) -> np.ndarray:
    values = require_float64_finite("motion", np.asarray(motion)).copy()
    scale = float(s_rig)
    gain_values = require_float64_finite("gains", np.asarray(gains))
    if not math.isfinite(scale) or scale <= 0.0 or gain_values.shape != (3,):
        raise Ktjd17CodecError(f"invalid scale/gains: {scale}, {gain_values.shape}")
    if np.any(gain_values <= 0.0):
        raise Ktjd17CodecError(f"normalization gains must be positive: {gain_values}")
    values[..., 0:3] *= gain_values[0] / scale
    values[..., 9:12] *= gain_values[1] / scale
    values[:, 0, 13:15] *= gain_values[2] / scale
    return require_float64_finite("normalized_motion", values)


def build_model_view(
    motion: np.ndarray,
    heading_valid: np.ndarray,
    *,
    parents: np.ndarray,
    R_rest_global: np.ndarray,
    rotation_source_kind: np.ndarray,
    s_rig: float,
    gains: np.ndarray,
    T_max: int,
    J_max: int,
    crop_start: int = 0,
    crop_length: int | None = None,
    yaw_radians: float = 0.0,
) -> ModelView:
    # Published KTJD artifacts deliberately use float32 storage, while all
    # crop/yaw/normalization algebra is evaluated in float64.  Accept exactly
    # those two public representations and make the precision boundary here;
    # callers must not need an undocumented cast between load_motion_npz() and
    # build_model_view().
    stored = np.asarray(motion)
    if stored.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise Ktjd17CodecError(
            f"motion must be float32 storage or float64 working data, got {stored.dtype}"
        )
    if not np.isfinite(stored).all():
        location = np.argwhere(~np.isfinite(stored))[0].tolist()
        raise Ktjd17CodecError(f"motion contains non-finite value at {location}")
    source = require_float64_finite(
        "motion", stored.astype(np.float64, copy=False)
    )
    length = source.shape[0] - int(crop_start) if crop_length is None else int(crop_length)
    cropped, heading = crop_full_clip(
        source, heading_valid, start=int(crop_start), length=length
    )
    augmented = yaw_augment(
        cropped,
        heading,
        R_rest_global=R_rest_global,
        phi=yaw_radians,
    )
    normalized = normalize_model_motion(augmented, s_rig=s_rig, gains=gains)
    T_valid, J_phys = normalized.shape[:2]
    if T_valid > T_max or J_phys > J_max:
        raise Ktjd17CodecError(
            f"sample exceeds padded bounds T={T_valid}/{T_max}, J={J_phys}/{J_max}"
        )
    padded = np.zeros((T_max, J_max, 17), dtype=np.float32)
    padded[:T_valid, :J_phys] = normalized.astype(np.float32)
    masks = derive_masks(
        T_valid=T_valid,
        J_phys=J_phys,
        T_max=T_max,
        J_max=J_max,
        parents=parents,
        rotation_source_kind=rotation_source_kind,
        heading_valid=heading,
    )
    return ModelView(
        motion=padded,
        masks=masks,
        T_valid=T_valid,
        J_phys=J_phys,
        crop_start=int(crop_start),
        yaw_radians=float(yaw_radians),
    )


def load_motion_npz(
    path: str | Path, *, expected_fps_target: float | None = None
) -> dict[str, object]:
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as payload:
        required = {"motion", "heading_valid", "clip_id", "rig_id", "fps_target", "origin_xz"}
        missing = sorted(required - set(payload.files))
        if missing:
            raise Ktjd17CodecError(f"{source}: missing motion keys {missing}")
        motion = np.asarray(payload["motion"])
        if motion.dtype != np.float32 or motion.ndim != 3 or motion.shape[-1] != 17:
            raise Ktjd17CodecError(
                f"{source}: motion must be float32 [T,J,17], got {motion.dtype} {motion.shape}"
            )
        if not np.isfinite(motion).all():
            location = np.argwhere(~np.isfinite(motion))[0].tolist()
            raise Ktjd17CodecError(f"{source}: non-finite motion at {location}")
        heading_valid = np.asarray(payload["heading_valid"])
        if heading_valid.dtype != np.bool_ or heading_valid.shape != (motion.shape[0],):
            raise Ktjd17CodecError(
                f"{source}: heading_valid must be bool [{motion.shape[0]}], "
                f"got {heading_valid.dtype} {heading_valid.shape}"
            )
        fps_payload = np.asarray(payload["fps_target"])
        if fps_payload.shape != () or fps_payload.dtype != np.dtype(np.float64):
            raise Ktjd17CodecError(
                f"{source}: fps_target must be a float64 scalar, "
                f"got {fps_payload.dtype} {fps_payload.shape}"
            )
        fps_target = float(fps_payload.item())
        if not math.isfinite(fps_target) or fps_target <= 0.0:
            raise Ktjd17CodecError(f"{source}: invalid fps_target {fps_target}")
        if expected_fps_target is not None and fps_target != float(
            expected_fps_target
        ):
            raise Ktjd17CodecError(
                f"{source}: fps_target {fps_target} != schema {expected_fps_target}"
            )
        origin_xz = np.asarray(payload["origin_xz"])
        if (
            origin_xz.dtype != np.float64
            or origin_xz.shape != (2,)
            or not np.isfinite(origin_xz).all()
        ):
            raise Ktjd17CodecError(
                f"{source}: origin_xz must be finite float64 [2], "
                f"got {origin_xz.dtype} {origin_xz.shape}"
            )
        clip_payload = np.asarray(payload["clip_id"])
        rig_payload = np.asarray(payload["rig_id"])
        if clip_payload.shape != () or clip_payload.dtype.kind != "U":
            raise Ktjd17CodecError(
                f"{source}: clip_id must be a Unicode scalar, "
                f"got {clip_payload.dtype} {clip_payload.shape}"
            )
        if rig_payload.shape != () or rig_payload.dtype.kind != "U":
            raise Ktjd17CodecError(
                f"{source}: rig_id must be a Unicode scalar, "
                f"got {rig_payload.dtype} {rig_payload.shape}"
            )
        clip_id = str(clip_payload.item())
        rig_id = str(rig_payload.item())
        if not clip_id or not rig_id:
            raise Ktjd17CodecError(f"{source}: clip_id and rig_id must be nonempty")
        return {
            "motion": motion,
            "heading_valid": heading_valid,
            "clip_id": clip_id,
            "rig_id": rig_id,
            "fps_target": fps_target,
            "origin_xz": origin_xz,
        }
