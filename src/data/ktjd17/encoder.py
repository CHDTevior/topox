"""Strict source-to-KTJD-17 encoder and atomic raw motion writer."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .codec import (
    EncodedChannels,
    Ktjd17CodecError,
    SmootherConfig,
    direct_decode_positions,
    encode_ktjd17_channels,
    global_to_local_rotations,
    local_to_global_rotations,
    require_float64_finite,
    require_so3,
    resample_root_and_local_rotations,
    restore_origin_xz,
    rotation_diagnostics,
    world_velocity,
)
from .decoder import decode_ktjd17
from .human_fixed_rig import HumanFixedRig
from .source_parser import (
    ParsedBvhMotion,
    ParsedSourceMotion,
    parse_bvh_numeric,
    parse_bvh_source,
    parse_motionstreamer272_source,
    source_fk_metrics,
)
from .truebones_fixed_rig import (
    ConditioningCatalog,
    ForwardSpec,
    TRUEBONES_FORWARD_SPECS,
    build_fixed_rig_motion,
)


ENCODER_VERSION = "ktjd17-encoder-v1"
SOURCE_PLAN_COMMIT = "9181f5cccbad23e941bf94c2874daf36e7f288cf"


class Ktjd17EncoderError(RuntimeError):
    """One source clip cannot enter the lossless KTJD-17 build."""


@dataclasses.dataclass(frozen=True)
class EncoderConfig:
    fps_target: float
    smoother: SmootherConfig
    contact_tau_h: float
    contact_tau_v: float
    heading_eps_h: float
    calibration_status: str = "candidate_unfrozen"

    def validate(self) -> None:
        values = {
            "fps_target": self.fps_target,
            "contact_tau_h": self.contact_tau_h,
            "contact_tau_v": self.contact_tau_v,
            "heading_eps_h": self.heading_eps_h,
        }
        for name, value in values.items():
            number = float(value)
            if not math.isfinite(number) or number <= 0.0:
                raise Ktjd17EncoderError(f"{name} must be finite and >0, got {value}")
        if self.calibration_status not in {"candidate_unfrozen", "frozen"}:
            raise Ktjd17EncoderError(
                f"invalid calibration_status {self.calibration_status!r}"
            )
        self.smoother.validate(float(self.fps_target))

    def as_record(self) -> dict[str, Any]:
        return {
            "fps_target": float(self.fps_target),
            "smoother": {
                "id": self.smoother.method,
                "params": self.smoother.as_schema_params(),
                "short_clip_rule": self.smoother.schema_short_clip_rule,
            },
            "contact": {
                "tau_h": float(self.contact_tau_h),
                "tau_v": float(self.contact_tau_v),
            },
            "heading": {"eps_h": float(self.heading_eps_h)},
            "calibration_status": self.calibration_status,
        }


@dataclasses.dataclass(frozen=True)
class SkeletonData:
    path: str
    sha256: str
    rig_id: str
    source_family: str
    topology_family: str
    joint_names: tuple[str, ...]
    parents: np.ndarray
    P_rest_global: np.ndarray
    R_rest_global: np.ndarray
    R_rest_local: np.ndarray
    offset_parent_local: np.ndarray
    rotation_source_kind: np.ndarray
    heading_carrier_joint: int
    u_forward_local: np.ndarray
    source_to_canonical_C: np.ndarray
    source_to_canonical_alpha: float
    source_to_canonical_o: np.ndarray
    s_rig: float
    artifact_status: str
    metadata: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class PreparedMotion:
    clip_id: str
    rig_id: str
    source_family: str
    topology_family: str
    fps_src: float
    root_positions: np.ndarray
    local_rotations: np.ndarray
    source_positions_diagnostic: np.ndarray
    source_global_rotations: np.ndarray
    source_parser_metrics: dict[str, Any]
    provenance: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class EncodedMotion:
    clip_id: str
    rig_id: str
    motion_float64: np.ndarray
    motion_float32: np.ndarray
    heading_valid: np.ndarray
    origin_xz: np.ndarray
    encoded: EncodedChannels
    resample_mode: str
    fps_src: float
    fps_target: float
    metrics: dict[str, Any]
    provenance: dict[str, Any]

    def artifact_payload(self) -> dict[str, np.ndarray]:
        return {
            "motion": self.motion_float32.copy(),
            "heading_valid": self.heading_valid.astype(bool, copy=True),
            "clip_id": _text_scalar(self.clip_id),
            "rig_id": _text_scalar(self.rig_id),
            "fps_target": np.asarray(self.fps_target, dtype=np.float64),
            "origin_xz": self.origin_xz.astype(np.float64, copy=True),
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_scalar(value: str) -> np.ndarray:
    text = str(value)
    return np.asarray(text, dtype=f"<U{max(1, len(text))}")


def _json_from_scalar(array: np.ndarray, name: str) -> dict[str, Any]:
    try:
        value = json.loads(str(np.asarray(array).item()))
    except Exception as exc:  # noqa: BLE001
        raise Ktjd17EncoderError(f"cannot parse skeleton JSON field {name}: {exc}") from exc
    if not isinstance(value, dict):
        raise Ktjd17EncoderError(f"skeleton JSON field {name} is not an object")
    return value


def load_skeleton(path: str | Path) -> SkeletonData:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise Ktjd17EncoderError(f"skeleton artifact is missing: {source}")
    with np.load(source, allow_pickle=False) as payload:
        required = {
            "joint_names",
            "parents",
            "P_rest_global",
            "R_rest_global",
            "R_rest_local",
            "offset_parent_local",
            "rotation_source_kind",
            "heading_carrier_joint",
            "u_forward_local",
            "source_to_canonical_C",
            "source_to_canonical_alpha",
            "source_to_canonical_o",
            "s_rig",
            "rig_id",
            "source_family",
            "topology_family",
            "artifact_status",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise Ktjd17EncoderError(f"{source}: missing skeleton keys {missing}")
        names = tuple(str(value) for value in payload["joint_names"].tolist())
        parents = np.asarray(payload["parents"], dtype=np.int64)
        P_rest = np.asarray(payload["P_rest_global"], dtype=np.float64)
        R_rest_global = np.asarray(payload["R_rest_global"], dtype=np.float64)
        R_rest_local = np.asarray(payload["R_rest_local"], dtype=np.float64)
        offsets = np.asarray(payload["offset_parent_local"], dtype=np.float64)
        kinds = np.asarray(payload["rotation_source_kind"]).astype(str)
        joint_count = len(names)
        if (
            len(set(names)) != joint_count
            or parents.shape != (joint_count,)
            or P_rest.shape != (joint_count, 3)
            or R_rest_global.shape != (joint_count, 3, 3)
            or R_rest_local.shape != (joint_count, 3, 3)
            or offsets.shape != (joint_count, 3)
            or kinds.shape != (joint_count,)
        ):
            raise Ktjd17EncoderError(f"{source}: inconsistent skeleton shapes")
        for name, array in {
            "P_rest_global": P_rest,
            "R_rest_global": R_rest_global,
            "R_rest_local": R_rest_local,
            "offset_parent_local": offsets,
        }.items():
            require_float64_finite(name, array)
        require_so3("R_rest_global", R_rest_global)
        require_so3("R_rest_local", R_rest_local)
        metadata = {
            key: _json_from_scalar(payload[key], key)
            for key in (
                "heading_payload_provenance",
                "source_to_canonical_provenance",
                "position_geometry_provenance",
                "conditioning_authority",
                "unit_metadata",
                "joint_map_metadata",
            )
            if key in payload.files
        }
        return SkeletonData(
            path=str(source),
            sha256=_sha256_file(source),
            rig_id=str(np.asarray(payload["rig_id"]).item()),
            source_family=str(np.asarray(payload["source_family"]).item()),
            topology_family=str(np.asarray(payload["topology_family"]).item()),
            joint_names=names,
            parents=parents,
            P_rest_global=P_rest,
            R_rest_global=R_rest_global,
            R_rest_local=R_rest_local,
            offset_parent_local=offsets,
            rotation_source_kind=kinds,
            heading_carrier_joint=int(np.asarray(payload["heading_carrier_joint"]).item()),
            u_forward_local=np.asarray(payload["u_forward_local"], dtype=np.float64),
            source_to_canonical_C=np.asarray(
                payload["source_to_canonical_C"], dtype=np.float64
            ),
            source_to_canonical_alpha=float(
                np.asarray(payload["source_to_canonical_alpha"]).item()
            ),
            source_to_canonical_o=np.asarray(
                payload["source_to_canonical_o"], dtype=np.float64
            ),
            s_rig=float(np.asarray(payload["s_rig"]).item()),
            artifact_status=str(np.asarray(payload["artifact_status"]).item()),
            metadata=metadata,
        )


def skeleton_from_human_contract(contract: HumanFixedRig, *, path: str) -> SkeletonData:
    return SkeletonData(
        path=path,
        sha256="pending_write",
        rig_id="HML3D_Human",
        source_family="motionstreamer272",
        topology_family="human",
        joint_names=contract.joint_names,
        parents=contract.parents,
        P_rest_global=contract.P_rest_global,
        R_rest_global=contract.R_rest_global,
        R_rest_local=contract.R_rest_local,
        offset_parent_local=contract.offset_parent_local,
        rotation_source_kind=contract.rotation_source_kind.astype(str),
        heading_carrier_joint=contract.heading_carrier_joint,
        u_forward_local=contract.u_forward_local,
        source_to_canonical_C=np.eye(3, dtype=np.float64),
        source_to_canonical_alpha=1.0,
        source_to_canonical_o=np.zeros(3, dtype=np.float64),
        s_rig=contract.s_rig,
        artifact_status="t05_prototype_override_pass",
        metadata={"position_geometry_provenance": contract.provenance},
    )


def _rest_mode(rig_record: Mapping[str, Any]) -> str:
    method = rig_record.get("rest_pose", {}).get("selection_method")
    if method == "explicit_tpose_filename":
        return "explicit_tpose_frame"
    if method in {"legacy_idle_fallback", "legacy_first_file_fallback"}:
        return "legacy_idle_fallback_review"
    raise Ktjd17EncoderError(
        f"unsupported Truebones rest selection {method!r} for {rig_record.get('rig_id')}"
    )


def _parse_truebones(
    clip_record: Mapping[str, Any],
    rig_record: Mapping[str, Any],
    *,
    rest_cache: dict[str, ParsedBvhMotion],
) -> ParsedSourceMotion:
    source = clip_record["source"]
    joint_map = rig_record["joint_map"]
    rest_path = str(Path(rig_record["rest_pose"]["source_path"]).expanduser().resolve())
    if rest_path not in rest_cache:
        rest_cache[rest_path] = parse_bvh_numeric(rest_path)
    return parse_bvh_source(
        source["path"],
        retained_names=joint_map["btjd_joint_names"],
        retained_parents=joint_map["btjd_parents"],
        expected_rotation_kinds=joint_map["rotation_source_kind"],
        frame_slice=source["slice_frames"],
        rest_path=rest_path,
        rest_mode=_rest_mode(rig_record),
        parsed_rest=rest_cache[rest_path],
        family="truebones",
    )


def prepare_manifest_clip(
    clip_record: Mapping[str, Any],
    rig_record: Mapping[str, Any],
    skeleton: SkeletonData,
    *,
    conditioning_catalog: ConditioningCatalog | None,
    rest_cache: dict[str, ParsedBvhMotion] | None = None,
    truebones_forward_specs: Mapping[str, ForwardSpec] | None = None,
) -> PreparedMotion:
    clip_id = str(clip_record["clip_id"])
    rig_id = str(clip_record["rig_id"])
    if rig_id != skeleton.rig_id or rig_id != rig_record.get("rig_id"):
        raise Ktjd17EncoderError(f"{clip_id}: clip/rig/skeleton identity mismatch")
    source_family = str(clip_record["source"]["family"])
    cache = {} if rest_cache is None else rest_cache
    if source_family == "truebones":
        if conditioning_catalog is None:
            raise Ktjd17EncoderError("Truebones preparation requires conditioning catalog")
        if skeleton.artifact_status not in {"pass", "review"}:
            raise Ktjd17EncoderError(
                f"{clip_id}: skeleton status {skeleton.artifact_status!r} is not encodable"
            )
        specs = (
            TRUEBONES_FORWARD_SPECS
            if truebones_forward_specs is None
            else truebones_forward_specs
        )
        try:
            spec = specs[rig_id]
        except KeyError as exc:
            raise Ktjd17EncoderError(
                f"{clip_id}: no reviewed Truebones forward spec for {rig_id}"
            ) from exc
        parsed = _parse_truebones(clip_record, rig_record, rest_cache=cache)
        fixed_geometry = conditioning_catalog.rig(
            rig_id,
            expected_names=rig_record["joint_map"]["btjd_joint_names"],
            expected_parents=rig_record["joint_map"]["btjd_parents"],
        )
        fixed = build_fixed_rig_motion(parsed, fixed_geometry, spec)
        comparisons = {
            "P_rest_global": float(
                np.max(np.abs(fixed.P_rest_global - skeleton.P_rest_global))
            ),
            "R_rest_global": float(
                np.max(np.abs(fixed.R_rest_global - skeleton.R_rest_global))
            ),
            "R_rest_local": float(
                np.max(np.abs(fixed.R_rest_local - skeleton.R_rest_local))
            ),
            "offset_parent_local": float(
                np.max(
                    np.abs(fixed.offset_parent_local - skeleton.offset_parent_local)
                )
            ),
            "C": float(np.max(np.abs(fixed.C - skeleton.source_to_canonical_C))),
            "o": float(np.max(np.abs(fixed.o - skeleton.source_to_canonical_o))),
            "alpha": abs(fixed.alpha - skeleton.source_to_canonical_alpha),
        }
        if max(comparisons.values()) > 1e-10:
            raise Ktjd17EncoderError(
                f"{clip_id}: rebuilt fixed-rig payload differs from skeleton: {comparisons}"
            )
        local_rotations = global_to_local_rotations(
            skeleton.parents, fixed.R_global
        )
        raw_positions_canonical = skeleton.source_to_canonical_alpha * (
            (np.asarray(parsed.source_positions, dtype=np.float64)
             - skeleton.source_to_canonical_o)
            @ skeleton.source_to_canonical_C.T
        )
        return PreparedMotion(
            clip_id=clip_id,
            rig_id=rig_id,
            source_family=source_family,
            topology_family=skeleton.topology_family,
            fps_src=float(parsed.fps),
            root_positions=fixed.P_authoritative[:, 0].copy(),
            local_rotations=local_rotations,
            source_positions_diagnostic=raw_positions_canonical,
            source_global_rotations=fixed.R_global,
            source_parser_metrics={
                **source_fk_metrics(parsed),
                **{f"fixed_rig_{key}": value for key, value in fixed.metrics.items()},
            },
            provenance={
                "source_path": parsed.path,
                "source_sha256": _sha256_file(Path(parsed.path)),
                "source_rest_path": parsed.rest_path,
                "source_rest_sha256": _sha256_file(Path(str(parsed.rest_path))),
                "source_frame_slice": list(clip_record["source"]["slice_frames"]),
                "rotation_authority": "original_bvh_declared_rotation_channels_only",
                "position_authority": "current_btjd_fixed_cond_geometry_plus_raw_root",
                "raw_nonroot_xyz_role": "diagnostic_only",
                "skeleton_rebuild_max_abs": comparisons,
            },
        )
    if source_family == "motionstreamer272":
        if skeleton.artifact_status != "t05_prototype_override_pass":
            raise Ktjd17EncoderError(
                f"{clip_id}: Human requires explicit fixed-neutral prototype override"
            )
        parsed = parse_motionstreamer272_source(
            clip_record["source"]["path"],
            joint_names=rig_record["joint_map"]["btjd_joint_names"],
            parents=rig_record["joint_map"]["btjd_parents"],
            neutral_model_path=rig_record["rest_pose"]["source_path"],
        )
        global_rotations = local_to_global_rotations(
            skeleton.parents, parsed.local_rotations
        )
        return PreparedMotion(
            clip_id=clip_id,
            rig_id=rig_id,
            source_family=source_family,
            topology_family="human",
            fps_src=float(parsed.fps),
            root_positions=np.asarray(parsed.source_positions[:, 0], dtype=np.float64),
            local_rotations=np.asarray(parsed.local_rotations, dtype=np.float64),
            source_positions_diagnostic=np.asarray(
                parsed.source_positions, dtype=np.float64
            ),
            source_global_rotations=global_rotations,
            source_parser_metrics=source_fk_metrics(parsed),
            provenance={
                "source_path": parsed.path,
                "source_sha256": _sha256_file(Path(parsed.path)),
                "rotation_authority": "motionstreamer272_real_local_rotation_channels_140_272",
                "position_authority": "current_btjd_fixed_neutral_offsets_plus_motionstreamer_root",
                "raw_shaped_position_role": "diagnostic_only_not_fixed_neutral_authority",
                "claim_boundary": "current_btjd_fixed_neutral_human_not_subject_shaped_amass",
            },
        )
    raise Ktjd17EncoderError(
        f"{clip_id}: unsupported source family {source_family!r}"
    )


def _rigid_edge_max_norm(
    positions: np.ndarray,
    skeleton: SkeletonData,
) -> float:
    rest_lengths = np.asarray(
        [
            np.linalg.norm(
                skeleton.P_rest_global[child]
                - skeleton.P_rest_global[int(skeleton.parents[child])]
            )
            for child in range(1, len(skeleton.parents))
        ],
        dtype=np.float64,
    )
    motion_lengths = np.stack(
        [
            np.linalg.norm(
                positions[:, child]
                - positions[:, int(skeleton.parents[child])],
                axis=-1,
            )
            for child in range(1, len(skeleton.parents))
        ],
        axis=-1,
    )
    return float(np.max(np.abs(motion_lengths - rest_lengths)) / skeleton.s_rig)


def encode_prepared_motion(
    prepared: PreparedMotion,
    skeleton: SkeletonData,
    config: EncoderConfig,
) -> EncodedMotion:
    config.validate()
    if prepared.rig_id != skeleton.rig_id:
        raise Ktjd17EncoderError("prepared motion and skeleton rig differ")
    resampled = resample_root_and_local_rotations(
        prepared.root_positions,
        prepared.local_rotations,
        fps_src=prepared.fps_src,
        fps_target=config.fps_target,
    )
    encoded = encode_ktjd17_channels(
        parents=skeleton.parents,
        root_positions=resampled.root_positions,
        local_rotations=resampled.local_rotations,
        offset_parent_local=skeleton.offset_parent_local,
        R_rest_global=skeleton.R_rest_global,
        s_rig=skeleton.s_rig,
        fps_target=config.fps_target,
        smoother=config.smoother,
        contact_tau_h=config.contact_tau_h,
        contact_tau_v=config.contact_tau_v,
        heading_carrier_joint=skeleton.heading_carrier_joint,
        u_forward_local=skeleton.u_forward_local,
        heading_eps_h=config.heading_eps_h,
    )
    motion64 = encoded.motion
    direct64 = direct_decode_positions(motion64)
    direct64_error = float(np.max(np.abs(direct64 - encoded.positions_clip)))
    if direct64_error > 1e-10 * skeleton.s_rig:
        raise Ktjd17EncoderError(
            f"{prepared.clip_id}: float64 direct roundtrip failed: {direct64_error}"
        )
    restored64 = restore_origin_xz(direct64, encoded.origin_xz)
    origin64_error = float(np.max(np.abs(restored64 - encoded.positions_absolute)))
    if origin64_error > 1e-10 * skeleton.s_rig:
        raise Ktjd17EncoderError(
            f"{prepared.clip_id}: origin restore failed: {origin64_error}"
        )
    motion32 = motion64.astype(np.float32)
    if not np.isfinite(motion32).all():
        location = np.argwhere(~np.isfinite(motion32))[0].tolist()
        raise Ktjd17EncoderError(
            f"{prepared.clip_id}: float32 storage is non-finite at {location}"
        )
    direct32 = direct_decode_positions(motion32.astype(np.float64))
    reference32 = encoded.positions_clip.astype(np.float32).astype(np.float64)
    direct32_error = float(np.max(np.abs(direct32 - reference32)))
    if direct32_error > 1e-5 * skeleton.s_rig:
        raise Ktjd17EncoderError(
            f"{prepared.clip_id}: float32 direct roundtrip failed: {direct32_error}"
        )
    expected_velocity32 = world_velocity(direct32, fps=config.fps_target)
    velocity32_error = float(
        np.max(
            np.abs(
                motion32[..., 9:12].astype(np.float64) - expected_velocity32
            )
        )
    )
    if velocity32_error > 1e-5 * skeleton.s_rig * config.fps_target:
        raise Ktjd17EncoderError(
            f"{prepared.clip_id}: float32 velocity gate failed: {velocity32_error}"
        )
    decoded = decode_ktjd17(
        motion32.astype(np.float64),
        parents=skeleton.parents,
        R_rest_global=skeleton.R_rest_global,
        R_rest_local=skeleton.R_rest_local,
        offset_parent_local=skeleton.offset_parent_local,
        rotation_source_kind=skeleton.rotation_source_kind,
        strict_gt=True,
    )
    direct_fk_error_norm = float(
        np.max(np.linalg.norm(decoded.positions_direct_minus_fk, axis=-1))
        / skeleton.s_rig
    )
    rigid_edge_norm = _rigid_edge_max_norm(direct32, skeleton)
    if rigid_edge_norm > 1e-4:
        raise Ktjd17EncoderError(
            f"{prepared.clip_id}: rigid edge gate failed: {rigid_edge_norm}"
        )
    rotation_diag = rotation_diagnostics(decoded.global_rotations)
    if np.any(motion32[:, 1:, 13:17] != 0.0):
        raise Ktjd17EncoderError(
            f"{prepared.clip_id}: non-root root-global storage is not exact zero"
        )
    invalid_heading_nonzero = float(
        np.max(
            np.abs(motion32[~encoded.heading_valid, 0, 15:17]), initial=0.0
        )
    )
    if invalid_heading_nonzero != 0.0:
        raise Ktjd17EncoderError(
            f"{prepared.clip_id}: invalid heading sentinel is not exact zero"
        )
    metrics = {
        "T_src": int(prepared.root_positions.shape[0]),
        "T_target": int(motion32.shape[0]),
        "J_phys": int(motion32.shape[1]),
        "s_rig": float(skeleton.s_rig),
        "ground_shift_y": float(encoded.ground_shift_y),
        "direct_roundtrip_float64_max_abs": direct64_error,
        "direct_roundtrip_float64_max_norm": direct64_error / skeleton.s_rig,
        "direct_roundtrip_float32_max_abs": direct32_error,
        "direct_roundtrip_float32_max_norm": direct32_error / skeleton.s_rig,
        "origin_restore_float64_max_abs": origin64_error,
        "velocity_float32_max_abs": velocity32_error,
        "velocity_float32_max_norm_fps": velocity32_error
        / (skeleton.s_rig * config.fps_target),
        "rigid_edge_max_norm": rigid_edge_norm,
        "direct_vs_fk_max_norm": direct_fk_error_norm,
        "direct_vs_fk_mpjpe_norm": float(
            np.mean(np.linalg.norm(decoded.positions_direct_minus_fk, axis=-1))
            / skeleton.s_rig
        ),
        "heading_valid_fraction": float(np.mean(encoded.heading_valid)),
        "heading_invalid_longest_run": _longest_false_run(encoded.heading_valid),
        "contact_positive_rate": float(np.mean(motion32[..., 12])),
        "rotation_orthogonality_max_abs": rotation_diag["orthogonality_max_abs"],
        "rotation_determinant_min": rotation_diag["determinant_min"],
        "rotation_determinant_max": rotation_diag["determinant_max"],
        "q_rms_normalized_by_s_rig": float(
            np.sqrt(np.mean(np.square(motion64[..., 0:3] / skeleton.s_rig)))
        ),
        "velocity_rms_normalized_by_s_rig": float(
            np.sqrt(np.mean(np.square(motion64[..., 9:12] / skeleton.s_rig)))
        ),
        "smooth_root_rms_normalized_by_s_rig": float(
            np.sqrt(
                np.mean(
                    np.square(motion64[:, 0, 13:15] / skeleton.s_rig)
                )
            )
        ),
    }
    return EncodedMotion(
        clip_id=prepared.clip_id,
        rig_id=prepared.rig_id,
        motion_float64=motion64,
        motion_float32=motion32,
        heading_valid=encoded.heading_valid,
        origin_xz=encoded.origin_xz,
        encoded=encoded,
        resample_mode=resampled.mode,
        fps_src=float(prepared.fps_src),
        fps_target=float(config.fps_target),
        metrics=metrics,
        provenance={
            "encoder_version": ENCODER_VERSION,
            "source_plan_commit": SOURCE_PLAN_COMMIT,
            "config": config.as_record(),
            "resample_mode": resampled.mode,
            "source_timestep_seconds": 1.0 / float(prepared.fps_src),
            "target_timestep_seconds": 1.0 / float(config.fps_target),
            "source": prepared.provenance,
            "source_parser_metrics": prepared.source_parser_metrics,
            "skeleton_path": skeleton.path,
            "skeleton_sha256": skeleton.sha256,
            "position_direct_authority": True,
            "velocity_used_for_decode": False,
            "previous_frame_used_for_decode": False,
            "raw_storage_normalized": False,
            "raw_storage_padded": False,
        },
    )


def _longest_false_run(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in np.asarray(mask, dtype=bool):
        if value:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def write_npz_atomic(path: str | Path, payload: Mapping[str, np.ndarray]) -> str:
    target = Path(path).expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    if any(np.asarray(value).dtype.hasobject for value in payload.values()):
        raise Ktjd17EncoderError(f"{target}: object dtype is forbidden")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp.npz", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **payload)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _sha256_file(target)
