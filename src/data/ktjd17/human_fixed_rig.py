"""Narrow fixed-neutral Human contract for current BTJD prototype data.

MotionStreamer272 contains real local rotations but no SMPL shape coefficients.
The current BTJD corpus intentionally re-FKs those rotations on one hash-pinned
neutral conditioning skeleton.  This module records that exact, narrow contract;
it does not claim recovery of the original subject-specific AMASS geometry.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .codec import Ktjd17CodecError, fk_from_global_rotations, require_float64_finite
from .truebones_fixed_rig import (
    ACTIVE_COND_SHA256,
    conditioning_payload_sha256,
    load_conditioning_catalog,
)


HUMAN_RIG_ID = "HML3D_Human"
HUMAN_CONDITIONING_PAYLOAD_SHA256 = (
    "f4b65120fe35f621a2e54f761728f3f249594eac0e41d7e94b5000ec8a6f5c02"
)
HUMAN_SMPL_NEUTRAL_SHA256 = (
    "a60a7e29d33f09ef1a6352907e2f485e161f471330ca40592764374063e751df"
)
HUMAN_CONTRACT_VERSION = "ktjd17-current-btjd-human-neutral-v2"


@dataclasses.dataclass(frozen=True)
class HumanFixedRig:
    joint_names: tuple[str, ...]
    parents: np.ndarray
    P_rest_global: np.ndarray
    R_rest_global: np.ndarray
    R_rest_local: np.ndarray
    offset_parent_local: np.ndarray
    rotation_source_kind: np.ndarray
    heading_carrier_joint: int
    u_forward_local: np.ndarray
    s_rig: float
    payload: dict[str, np.ndarray]
    provenance: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_scalar(value: Any) -> np.ndarray:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return np.asarray(text, dtype=f"<U{max(1, len(text))}")


def _text_scalar(value: str) -> np.ndarray:
    text = str(value)
    return np.asarray(text, dtype=f"<U{max(1, len(text))}")


def build_current_btjd_human_fixed_rig(
    *,
    rig_record: Mapping[str, Any],
    active_cond_path: str | Path,
    legacy_truebones_cond_path: str | Path,
    t04_candidate_path: str | Path,
) -> HumanFixedRig:
    if rig_record.get("rig_id") != HUMAN_RIG_ID:
        raise Ktjd17CodecError(
            f"Human fixed-rig builder requires {HUMAN_RIG_ID}, got {rig_record.get('rig_id')!r}"
        )
    catalog = load_conditioning_catalog(
        active_cond_path,
        expected_active_sha256=ACTIVE_COND_SHA256,
        legacy_path=legacy_truebones_cond_path,
    )
    try:
        entry = catalog.active_entries[HUMAN_RIG_ID]
    except KeyError as exc:
        raise Ktjd17CodecError("active cond lacks HML3D_Human") from exc
    payload_hash = conditioning_payload_sha256(entry)
    if payload_hash != HUMAN_CONDITIONING_PAYLOAD_SHA256:
        raise Ktjd17CodecError(
            f"Human conditioning payload drifted: {payload_hash}"
        )

    joint_map = rig_record.get("joint_map")
    if not isinstance(joint_map, Mapping):
        raise Ktjd17CodecError("Human rig record lacks joint_map")
    names = tuple(str(value) for value in entry["joints_names"])
    expected_names = tuple(str(value) for value in joint_map["btjd_joint_names"])
    parents = np.asarray(entry["parents"], dtype=np.int64)
    expected_parents = np.asarray(joint_map["btjd_parents"], dtype=np.int64)
    if names != expected_names or not np.array_equal(parents, expected_parents):
        raise Ktjd17CodecError("Human cond names/parents differ from frozen inventory")
    offsets = np.asarray(entry["offsets"], dtype=np.float64)
    tpose = np.asarray(entry["tpos_first_frame"], dtype=np.float64)
    if offsets.shape != (len(names), 3) or tpose.shape[0] != len(names):
        raise Ktjd17CodecError("Human conditioning geometry has invalid shape")
    P_rest_raw = require_float64_finite(
        "human_P_rest_global_raw", tpose[:, :3].copy()
    )
    cumulative = np.empty_like(P_rest_raw)
    cumulative[0] = P_rest_raw[0]
    for child in range(1, len(names)):
        parent = int(parents[child])
        if not 0 <= parent < child:
            raise Ktjd17CodecError("Human parent tree is not parent-before-child")
        cumulative[child] = cumulative[parent] + offsets[child]
    geometry_error = float(np.max(np.abs(cumulative - P_rest_raw)))
    if geometry_error > 1e-7:
        raise Ktjd17CodecError(
            f"Human cond offsets do not reproduce tpose: {geometry_error}"
        )

    # The pinned conditioning payload stores offsets and its t-pose at finite
    # precision. They pass the reviewed source-quantization gate above, but are
    # not bitwise one geometry. Current BTJD motion FK uses the offsets, so use
    # those same offsets as the single rest-position authority. The t-pose
    # residual remains an explicit provenance diagnostic.
    ground_shift_y = -float(np.min(cumulative[:, 1]))
    P_rest = cumulative.copy()
    P_rest[:, 1] += ground_shift_y
    if abs(float(np.min(P_rest[:, 1]))) > 1e-7:
        raise Ktjd17CodecError("Human conditioning rest grounding failed")

    candidate = Path(t04_candidate_path).expanduser().resolve()
    with np.load(candidate, allow_pickle=False) as old:
        candidate_error = float(
            np.max(np.abs(np.asarray(old["P_rest_global"], dtype=np.float64) - P_rest))
        )
        candidate_offset_error = float(
            np.max(
                np.abs(
                    np.asarray(old["offset_parent_local"], dtype=np.float64)
                    - offsets
                )
            )
        )
        neutral_model_to_conditioning_o = np.asarray(
            old["source_to_canonical_o"], dtype=np.float64
        )
    if candidate_error > 1e-7 or candidate_offset_error > 1e-7:
        raise Ktjd17CodecError(
            "T04 Human candidate differs from current fixed-neutral conditioning "
            f"geometry: P={candidate_error}, offsets={candidate_offset_error}"
        )

    neutral_path = Path(rig_record["rest_pose"]["source_path"]).expanduser().resolve()
    neutral_sha = _sha256_file(neutral_path)
    if neutral_sha != HUMAN_SMPL_NEUTRAL_SHA256:
        raise Ktjd17CodecError(f"SMPL neutral model drifted: {neutral_sha}")

    identity = np.eye(3, dtype=np.float64)
    R_rest_global = np.broadcast_to(identity, (len(names), 3, 3)).copy()
    R_rest_local = R_rest_global.copy()
    root = np.broadcast_to(P_rest[0], (1, 3)).copy()
    rest_fk = fk_from_global_rotations(
        parents,
        root,
        R_rest_global[None],
        offsets,
    )[0]
    rest_fk_error = float(np.max(np.abs(rest_fk - P_rest)))
    if rest_fk_error > 1e-7:
        raise Ktjd17CodecError(f"Human fixed-neutral rest FK failed: {rest_fk_error}")
    s_rig = float(np.linalg.norm(np.ptp(P_rest, axis=0)))
    if not math.isfinite(s_rig) or s_rig <= 0.0:
        raise Ktjd17CodecError(f"invalid Human s_rig {s_rig}")

    provenance = {
        "contract_version": HUMAN_CONTRACT_VERSION,
        "claim_boundary": "current_btjd_fixed_neutral_human_not_subject_shaped_amass",
        "position_authority": "active_cond_fixed_neutral_offsets_plus_motionstreamer272_root",
        "rotation_authority": "motionstreamer272_real_local_rotation_channels_140_272",
        "rest_position_authority": "active_cond_hml3d_human_offsets_fk",
        "conditioning_tpose_role": "quantized_consistency_diagnostic_only",
        "rest_rotation_authority": "smpl_neutral_identity_local_axes",
        "source_stream_coordinate_status": "already_canonical_y_up_plus_z_forward",
        "source_to_canonical": {"C": "identity", "alpha": 1.0, "o": [0.0, 0.0, 0.0]},
        "neutral_model_to_conditioning_o_diagnostic": neutral_model_to_conditioning_o.tolist(),
        "raw_motionstreamer_shaped_positions_role": "diagnostic_only",
        "legacy_btjd_rotation_channels_used": False,
        "position_ik_used": False,
        "leaf_identity_imputation_used": False,
        "active_cond_sha256": catalog.active_sha256,
        "conditioning_payload_sha256": payload_hash,
        "smpl_neutral_sha256": neutral_sha,
        "t04_candidate_sha256": _sha256_file(candidate),
        "t04_candidate_position_max_abs": candidate_error,
        "t04_candidate_offset_max_abs": candidate_offset_error,
        "cond_offsets_to_tpose_max_abs": geometry_error,
        "cond_offset_fk_rest_ground_shift_y": ground_shift_y,
        "rest_fk_max_abs": rest_fk_error,
    }
    heading_provenance = {
        "status": "explicit_reviewed_current_btjd_human",
        "carrier_joint": 0,
        "carrier_name": names[0],
        "forward_method": "lateral_pairs",
        "forward_anchor_names": [
            "right_hip",
            "left_hip",
            "right_shoulder",
            "left_shoulder",
        ],
        "forward_anchor_indices": [2, 1, 17, 16],
        "forward_spec_provenance": (
            "current_btjd_human_neutral_anatomy_reviewed_t05"
        ),
        "local_forward_definition": "canonical_plus_z_in_smpl_neutral_root_axes",
        "canonical_rest_forward": [0.0, 0.0, 1.0],
        "polarity": "canonical_plus_z",
        "dynamic_perspective_visual_status": "pending_t05",
    }
    unit_metadata = {
        "length_unit_id": "motionstreamer272_current_btjd_neutral_unverified",
        "source_unit_to_meter": None,
        "canonical_scale_factor": 1.0,
        "s_rig": s_rig,
        "meter_claim": False,
    }
    payload: dict[str, np.ndarray] = {
        "joint_names": np.asarray(names, dtype=np.str_),
        "parents": parents.copy(),
        "P_rest_global": P_rest,
        "R_rest_global": R_rest_global,
        "R_rest_local": R_rest_local,
        "offset_parent_local": offsets.copy(),
        "rotation_source_kind": np.asarray(
            ["animated_dof"] * len(names), dtype=np.str_
        ),
        "heading_carrier_joint": np.asarray(0, dtype=np.int64),
        "u_forward_local": np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        "heading_payload_provenance": _json_scalar(heading_provenance),
        "source_to_canonical_C": identity.copy(),
        "source_to_canonical_alpha": np.asarray(1.0, dtype=np.float64),
        "source_to_canonical_o": np.zeros(3, dtype=np.float64),
        "s_rig": np.asarray(s_rig, dtype=np.float64),
        "length_unit_id": _text_scalar(unit_metadata["length_unit_id"]),
        "source_unit_to_meter": np.asarray([], dtype=np.float64),
        "canonical_scale_factor": np.asarray(1.0, dtype=np.float64),
        "joint_map_metadata": _json_scalar(dict(joint_map)),
        "rig_id": _text_scalar(HUMAN_RIG_ID),
        "source_family": _text_scalar("motionstreamer272"),
        "topology_family": _text_scalar("human"),
        "artifact_status": _text_scalar("t05_prototype_override_pass"),
        "reason_codes": np.asarray([], dtype=np.str_),
        "skeleton_format_version": _text_scalar(HUMAN_CONTRACT_VERSION),
        "representative_clip_id": _text_scalar("HML3D_Human_000000"),
        "source_rest_path": _text_scalar(str(neutral_path)),
        "source_rest_sha256": _text_scalar(neutral_sha),
        "source_to_canonical_provenance": _json_scalar(provenance),
        "position_geometry_provenance": _json_scalar(provenance),
        "conditioning_authority": _json_scalar(catalog.authority_record()),
        "conditioning_payload_sha256": _text_scalar(payload_hash),
        "fixed_rig_rotation_signatures": _json_scalar(
            {"status": "per_clip_motionstreamer272_real_rotations"}
        ),
        "unit_metadata": _json_scalar(unit_metadata),
    }
    if any(value.dtype.hasobject for value in payload.values()):
        raise Ktjd17CodecError("Human skeleton payload must be pickle-free")
    return HumanFixedRig(
        joint_names=names,
        parents=parents,
        P_rest_global=P_rest,
        R_rest_global=R_rest_global,
        R_rest_local=R_rest_local,
        offset_parent_local=offsets,
        rotation_source_kind=payload["rotation_source_kind"],
        heading_carrier_joint=0,
        u_forward_local=payload["u_forward_local"],
        s_rig=s_rig,
        payload=payload,
        provenance=provenance,
    )
