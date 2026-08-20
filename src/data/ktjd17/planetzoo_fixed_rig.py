"""Fixed-rig contract for the locally available PlanetZoo stage-2 BVHs.

The raw game BVHs are not present locally.  This module therefore makes a
deliberately narrow claim: the processed stage-2 BVH is the coordinate and
rotation authority.  Its hierarchy offsets have already been transformed into
the aligned rest basis, while every motion clip still carries real declared
Euler rotation channels.  No legacy 13-D positions, rotation channels, or
``cond.tpos_first_frame`` values may become motion/rest authority here.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import stat as stat_module
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .bvh_inventory import BvhHeader, parse_bvh_header
from .codec import require_float64_finite, require_so3
from .source_parser import ParsedSourceMotion
from .truebones_fixed_rig import ACTIVE_COND_SHA256


PLANETZOO_FIXED_RIG_VERSION = "ktjd17-planetzoo-stage2-fixed-rig-v1"
PLANETZOO_REST_MODE = "processed_hierarchy_stage2_fixed"
PLANETZOO_COND_OFFSET_MAX_ABS = 1e-6
PLANETZOO_SOURCE_POSITION_MAX_NORM = 1e-10
PLANETZOO_FORWARD_MAX_ABS_X = 1e-5
PLANETZOO_FORWARD_MIN_Z = 0.999999
COORDINATE_CONTRACT = (
    "right-handed; +Y is screen-up; +Z points out of the screen toward the viewer"
)


class PlanetzooFixedRigError(RuntimeError):
    """A PlanetZoo rig/clip violates the explicit stage-2 contract."""


@dataclasses.dataclass(frozen=True)
class PlanetzooFixedRig:
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
    source_header_root_offset: np.ndarray
    payload: dict[str, np.ndarray]
    provenance: dict[str, Any]
    metrics: dict[str, float | int | str]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PlanetzooFixedRigError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_scalar(value: Any) -> np.ndarray:
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return np.asarray(text, dtype=f"<U{max(1, len(text))}")


def _text_scalar(value: str) -> np.ndarray:
    text = str(value)
    return np.asarray(text, dtype=f"<U{max(1, len(text))}")


def _text_vector(values: Sequence[str]) -> np.ndarray:
    texts = [str(value) for value in values]
    width = max((len(value) for value in texts), default=1)
    return np.asarray(texts, dtype=f"<U{width}")


def _validate_parent_tree(parents: np.ndarray) -> None:
    _require(parents.ndim == 1 and len(parents) > 1, "invalid parent array")
    _require(int(parents[0]) == -1, "physical root parent must be -1")
    for child in range(1, len(parents)):
        parent = int(parents[child])
        _require(
            0 <= parent < child,
            f"parent-before-child violated at {child}: parent={parent}",
        )


def _fixed_fk(
    parents: np.ndarray,
    root_positions: np.ndarray,
    global_rotations: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    roots = require_float64_finite("PZ root positions", np.asarray(root_positions))
    rotations = require_float64_finite(
        "PZ global rotations", np.asarray(global_rotations)
    )
    if roots.ndim == 1:
        roots = roots[None]
        rotations = rotations[None]
        squeeze = True
    else:
        squeeze = False
    result = np.empty((len(roots), len(parents), 3), dtype=np.float64)
    result[:, 0] = roots
    for child in range(1, len(parents)):
        parent = int(parents[child])
        result[:, child] = result[:, parent] + np.einsum(
            "tij,j->ti", rotations[:, parent], offsets[child]
        )
    return result[0] if squeeze else result


def _conditioning_geometry_sha256(
    names: Sequence[str], parents: np.ndarray, offsets: np.ndarray
) -> str:
    """Hash only the three allowed PZ conditioning fields."""
    payload = {
        "joints_names": [str(value) for value in names],
        "parents": np.asarray(parents, dtype=np.int64).tolist(),
        "offsets": np.asarray(offsets, dtype=np.float64).tolist(),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def audit_planetzoo_header(
    header: BvhHeader, rig_record: Mapping[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Prove the full stage-2 header and retained direct-edge mapping."""
    rig_id = str(rig_record.get("rig_id"))
    _require(
        rig_record.get("source_family") == "planetzoo",
        f"{rig_id}: source family is not PlanetZoo",
    )
    joint_map = rig_record.get("joint_map")
    _require(isinstance(joint_map, Mapping), f"{rig_id}: joint_map is absent")
    expected_source_names = tuple(str(x) for x in joint_map["source_joint_names"])
    expected_source_parents = tuple(int(x) for x in joint_map["source_parents"])
    expected_node_kinds = tuple(str(x) for x in joint_map["source_node_kinds"])
    _require(
        header.joint_names == expected_source_names,
        f"{rig_id}: full source joint names drifted",
    )
    _require(
        header.parents == expected_source_parents,
        f"{rig_id}: full source parents drifted",
    )
    _require(
        tuple(joint.node_kind for joint in header.joints) == expected_node_kinds,
        f"{rig_id}: full source node kinds drifted",
    )
    expected_rotation_layout = str(joint_map["source_rotation_layout_sha256"])
    _require(
        header.rotation_layout_sha256() == expected_rotation_layout,
        f"{rig_id}: source rotation/channel layout drifted",
    )
    expected_rest_layout = str(rig_record["rest_pose"]["rest_layout_sha256"])
    _require(
        header.rest_layout_sha256() == expected_rest_layout,
        f"{rig_id}: stage-2 rest-offset layout drifted",
    )

    source_root = int(joint_map["source_root_index_for_btjd_root"])
    _require(source_root == 0, f"{rig_id}: retained physical root is not source root")
    position_joint_indices: list[int] = []
    animated_count = 0
    fixed_count = 0
    for index, joint in enumerate(header.joints):
        lowered = tuple(channel.lower() for channel in joint.channels)
        positions = tuple(channel for channel in lowered if channel.endswith("position"))
        rotations = tuple(channel for channel in lowered if channel.endswith("rotation"))
        if positions:
            position_joint_indices.append(index)
            _require(
                index == source_root
                and len(positions) == 3
                and sorted(channel[0] for channel in positions) == ["x", "y", "z"],
                f"{rig_id}: XYZ position channels are not root-only",
            )
        kind = joint.rotation_source_kind()
        if kind == "animated_dof":
            animated_count += 1
            _require(
                len(rotations) == 3
                and sorted(channel[0] for channel in rotations) == ["x", "y", "z"],
                f"{rig_id}: {joint.name} lacks declared XYZ rotation channels",
            )
        else:
            fixed_count += 1
            _require(not rotations, f"{rig_id}: fixed joint has numeric rotations")
    _require(
        position_joint_indices == [source_root],
        f"{rig_id}: expected exactly one root XYZ carrier, got {position_joint_indices}",
    )

    retained_names = tuple(str(x) for x in joint_map["btjd_joint_names"])
    parents = np.asarray(joint_map["btjd_parents"], dtype=np.int64)
    source_indices = np.asarray(joint_map["btjd_to_source"], dtype=np.int64)
    kinds = tuple(str(x) for x in joint_map["rotation_source_kind"])
    _validate_parent_tree(parents)
    _require(
        source_indices.shape == parents.shape
        and len(np.unique(source_indices)) == len(source_indices),
        f"{rig_id}: invalid retained source map",
    )
    _require(
        tuple(header.joint_names[int(index)] for index in source_indices)
        == retained_names,
        f"{rig_id}: retained joint names do not resolve exactly",
    )
    retained_kinds = tuple(
        header.joints[int(index)].rotation_source_kind() for index in source_indices
    )
    _require(retained_kinds == kinds, f"{rig_id}: retained rotation kinds drifted")
    direct_edges = 0
    for child in range(1, len(parents)):
        source_child = int(source_indices[child])
        source_parent = int(source_indices[int(parents[child])])
        _require(
            int(header.joints[source_child].parent) == source_parent,
            f"{rig_id}: retained edge {child} skips a source joint",
        )
        direct_edges += 1
    _require(
        direct_edges == len(parents) - 1
        and int(joint_map["direct_source_edge_count"]) == direct_edges
        and int(joint_map["source_skipping_edge_count"]) == 0,
        f"{rig_id}: direct-edge inventory drifted",
    )
    _require(
        int(joint_map["animated_dof_count"]) == sum(x == "animated_dof" for x in kinds)
        and int(joint_map["fixed_dof_count"]) == sum(x == "fixed_dof" for x in kinds),
        f"{rig_id}: retained DOF counts drifted",
    )
    retained_offsets = np.asarray(
        [header.joints[int(index)].offset for index in source_indices], dtype=np.float64
    )
    require_float64_finite("PZ retained declared offsets", retained_offsets)
    _require(
        float(np.max(np.abs(retained_offsets[0, [0, 2]]))) <= 1e-12,
        f"{rig_id}: stage-2 hierarchy root offset has nonzero XZ",
    )
    return retained_offsets, {
        "source_joint_count": len(header.joints),
        "retained_joint_count": len(retained_names),
        "source_animated_dof_count": animated_count,
        "source_fixed_dof_count": fixed_count,
        "retained_direct_edge_count": direct_edges,
        "root_position_channel_joint_count": len(position_joint_indices),
        "source_rotation_layout_sha256": header.rotation_layout_sha256(),
        "source_rest_layout_sha256": header.rest_layout_sha256(),
    }


def build_planetzoo_fixed_rig(
    *,
    rig_record: Mapping[str, Any],
    representative_bvh_path: str | Path,
    pinned_source_root: str | Path,
    cond_entry: Mapping[str, Any],
    active_cond_sha256: str,
) -> PlanetzooFixedRig:
    """Build one identity-rest fixed rig from a pinned stage-2 BVH header."""
    rig_id = str(rig_record.get("rig_id"))
    _require(rig_id.startswith("PZ_"), f"invalid PlanetZoo rig id {rig_id!r}")
    _require(
        active_cond_sha256 == ACTIVE_COND_SHA256,
        f"{rig_id}: active cond hash drifted: {active_cond_sha256}",
    )
    source_root = Path(
        os.path.abspath(os.fspath(Path(pinned_source_root).expanduser()))
    )
    representative = Path(
        os.path.abspath(os.fspath(Path(representative_bvh_path).expanduser()))
    )
    declared_representative = Path(
        os.path.abspath(
            os.fspath(Path(str(rig_record["rest_pose"]["source_path"])).expanduser())
        )
    )
    _require(
        source_root.is_dir()
        and not source_root.is_symlink()
        and source_root.resolve(strict=True) == source_root,
        f"{rig_id}: pinned PlanetZoo source root is invalid or symlinked",
    )
    _require(
        representative == declared_representative,
        f"{rig_id}: representative is not the pinned first stage-2 clip",
    )
    _require(
        representative.parent == source_root,
        f"{rig_id}: representative is not a direct child of the pinned source root",
    )
    _require(
        not representative.is_symlink(),
        f"{rig_id}: representative is a symlink",
    )
    try:
        representative_stat = representative.lstat()
    except OSError as exc:
        raise PlanetzooFixedRigError(
            f"{rig_id}: cannot lstat representative: {exc}"
        ) from exc
    _require(
        stat_module.S_ISREG(representative_stat.st_mode)
        and int(representative_stat.st_nlink) == 1
        and representative.resolve(strict=True) == representative,
        f"{rig_id}: representative is not a canonical single-link regular file",
    )
    header = parse_bvh_header(representative)
    declared_offsets, header_metrics = audit_planetzoo_header(header, rig_record)
    joint_map = rig_record["joint_map"]
    names = tuple(str(x) for x in joint_map["btjd_joint_names"])
    parents = np.asarray(joint_map["btjd_parents"], dtype=np.int64)
    kinds = np.asarray(joint_map["rotation_source_kind"]).astype(str)

    # Deliberately access only these three fields.  In particular,
    # tpos_first_frame and every legacy 13-D statistic are forbidden inputs.
    cond_names = tuple(str(x) for x in cond_entry["joints_names"])
    cond_parents = np.asarray(cond_entry["parents"], dtype=np.int64)
    cond_offsets = np.asarray(cond_entry["offsets"], dtype=np.float64)
    _require(cond_names == names, f"{rig_id}: cond joint names drifted")
    _require(np.array_equal(cond_parents, parents), f"{rig_id}: cond parents drifted")
    _require(
        cond_offsets.shape == declared_offsets.shape and np.isfinite(cond_offsets).all(),
        f"{rig_id}: cond offsets have invalid shape/values",
    )
    cond_offset_error = float(np.max(np.abs(cond_offsets - declared_offsets)))
    _require(
        cond_offset_error <= PLANETZOO_COND_OFFSET_MAX_ABS,
        f"{rig_id}: cond/header offset diagnostic exceeded 1e-6: {cond_offset_error}",
    )

    raw_rest = np.empty_like(declared_offsets)
    raw_rest[0] = declared_offsets[0]
    for child in range(1, len(parents)):
        raw_rest[child] = raw_rest[int(parents[child])] + declared_offsets[child]
    ground_shift_y = -float(np.min(raw_rest[:, 1]))
    P_rest = raw_rest.copy()
    P_rest[:, 1] += ground_shift_y
    _require(
        float(np.max(np.abs(P_rest[0, [0, 2]]))) <= 1e-12,
        f"{rig_id}: grounded rest root XZ is not zero",
    )
    _require(
        abs(float(np.min(P_rest[:, 1]))) <= 1e-12,
        f"{rig_id}: grounded rest minimum Y is not zero",
    )
    identity = np.eye(3, dtype=np.float64)
    R_rest_global = np.broadcast_to(identity, (len(names), 3, 3)).copy()
    R_rest_local = R_rest_global.copy()
    offsets = declared_offsets.copy()
    offsets[0] = 0.0
    rest_fk = _fixed_fk(parents, P_rest[0], R_rest_global, offsets)
    rest_fk_error = float(np.max(np.abs(rest_fk - P_rest)))
    _require(rest_fk_error <= 1e-12, f"{rig_id}: fixed rest FK failed")
    s_rig = float(np.linalg.norm(np.ptp(P_rest, axis=0)))
    _require(math.isfinite(s_rig) and s_rig > 0.0, f"{rig_id}: invalid s_rig")

    try:
        hips = names.index("def_c_hips_joint")
        chest = names.index("def_c_chest_joint")
    except ValueError as exc:
        raise PlanetzooFixedRigError(
            f"{rig_id}: reviewed hips/chest forward anchors are absent"
        ) from exc
    forward = P_rest[chest] - P_rest[hips]
    horizontal_norm = float(np.hypot(forward[0], forward[2]))
    _require(
        horizontal_norm > 1e-8 * s_rig,
        f"{rig_id}: hips-to-chest horizontal forward is degenerate",
    )
    forward_x = float(forward[0] / horizontal_norm)
    forward_z = float(forward[2] / horizontal_norm)
    _require(
        abs(forward_x) <= PLANETZOO_FORWARD_MAX_ABS_X
        and forward_z >= PLANETZOO_FORWARD_MIN_Z,
        f"{rig_id}: processed rest forward is not +Z: x={forward_x}, z={forward_z}",
    )

    source_sha = _sha256_file(representative)
    representative_after = representative.lstat()
    _require(
        stat_module.S_ISREG(representative_after.st_mode)
        and int(representative_after.st_nlink) == 1
        and all(
            int(getattr(representative_after, field))
            == int(getattr(representative_stat, field))
            for field in ("st_size", "st_mtime_ns", "st_dev", "st_ino", "st_nlink")
        ),
        f"{rig_id}: representative changed during fixed-rig construction",
    )
    conditioning_geometry_sha = _conditioning_geometry_sha256(
        cond_names, cond_parents, cond_offsets
    )
    metrics: dict[str, float | int | str] = {
        **header_metrics,
        "cond_offsets_to_header_max_abs": cond_offset_error,
        "rest_ground_shift_y": ground_shift_y,
        "rest_ground_min_y_abs": abs(float(np.min(P_rest[:, 1]))),
        "rest_root_xz_max_abs": float(np.max(np.abs(P_rest[0, [0, 2]]))),
        "rest_fk_max_abs": rest_fk_error,
        "rest_forward_horizontal_x": forward_x,
        "rest_forward_horizontal_z": forward_z,
        "s_rig": s_rig,
    }
    provenance = {
        "contract_version": PLANETZOO_FIXED_RIG_VERSION,
        "claim_boundary": "processed_planetzoo_stage2_coordinates_only_not_native_raw_game_bvh",
        "position_authority": "stage2_bvh_numeric_root_translation_plus_stage2_hierarchy_offsets_fk",
        "rotation_authority": "stage2_bvh_real_declared_euler_rotation_channels",
        "rest_position_authority": "stage2_bvh_hierarchy_offsets_identity_rest_grounded",
        "rest_rotation_authority": "identity_in_already_rebased_stage2_rest_basis",
        "source_to_canonical": {"C": "identity", "alpha": 1.0, "o": [0.0, 0.0, 0.0]},
        "authoritative_fk_formula": "P_child=P_parent+R_global_parent@offset_parent_local",
        "raw_game_bvh_available": False,
        "raw_game_orientation_reconstruction_claimed": False,
        "ktjd_position_ik_used": False,
        "time_varying_position_ik_used_as_authority": False,
        "upstream_static_tpose_position_ik_may_exist": True,
        "legacy_btjd13_motion_used": False,
        "legacy_btjd13_rotation_used": False,
        "cond_tpos_first_frame_used": False,
        "leaf_identity_imputation_used": False,
        "forbidden_inputs_used": False,
        "source_rotation_layout_sha256": header.rotation_layout_sha256(),
        "source_rest_layout_sha256": header.rest_layout_sha256(),
        "source_header_root_offset": declared_offsets[0].tolist(),
        "representative_source_sha256": source_sha,
        "active_cond_sha256": active_cond_sha256,
        "conditioning_geometry_sha256_allowed_fields_only": conditioning_geometry_sha,
        "cond_offsets_to_header_max_abs": cond_offset_error,
        "ground_shift_y": ground_shift_y,
    }
    heading_provenance = {
        "status": "static_anatomical_polarity_reviewed_stage2_numeric_pass",
        "carrier_joint": 0,
        "carrier_name": names[0],
        "forward_method": "hips_to_chest",
        "forward_anchor_names": [names[hips], names[chest]],
        "forward_anchor_indices": [hips, chest],
        "forward_spec_provenance": "processed_stage2_hips_to_chest_horizontal_axis_audited_all_311_rigs",
        "local_forward_definition": "canonical_plus_z_in_identity_stage2_root_axes",
        "canonical_rest_forward": [0.0, 0.0, 1.0],
        "polarity": "canonical_plus_z",
        "coordinate_contract": COORDINATE_CONTRACT,
        "dynamic_perspective_visual_status": "pending_312_rig_visual_gate",
    }
    unit_metadata = {
        "length_unit_id": "anytop_planetzoo_stage2_canonical_unlabeled",
        "source_unit_to_meter": None,
        "canonical_scale_factor": 1.0,
        "s_rig": s_rig,
        "meter_claim": False,
    }
    conditioning_authority = {
        "authority_kind": "planetzoo_stage2_header_primary_cond_offsets_diagnostic_only",
        "active_cond_sha256": active_cond_sha256,
        "allowed_fields_read": ["joints_names", "parents", "offsets"],
        "forbidden_fields_not_read": [
            "tpos_first_frame",
            "mean",
            "std",
            "legacy_btjd_motion_channels",
        ],
        "conditioning_geometry_sha256_allowed_fields_only": conditioning_geometry_sha,
        "cond_offsets_to_header_max_abs": cond_offset_error,
    }
    joint_map_metadata = {
        "mapping_kind": str(joint_map["mapping_kind"]),
        "joint_map_sha256": str(joint_map["joint_map_sha256"]),
        "source_rotation_layout_sha256": header.rotation_layout_sha256(),
        "source_rest_layout_sha256": header.rest_layout_sha256(),
        "direct_source_edge_count": len(parents) - 1,
        "source_skipping_edge_count": 0,
        "rotation_authority": "stage2_bvh_real_declared_euler_rotation_channels",
        "legacy_btjd_motion_channels_used": False,
    }
    identity_sha = hashlib.sha256(
        np.ascontiguousarray(R_rest_global.astype("<f8")).tobytes()
    ).hexdigest()
    payload: dict[str, np.ndarray] = {
        "joint_names": _text_vector(names),
        "parents": parents.copy(),
        "P_rest_global": P_rest,
        "R_rest_global": R_rest_global,
        "R_rest_local": R_rest_local,
        "offset_parent_local": offsets,
        "rotation_source_kind": _text_vector(kinds.tolist()),
        "heading_carrier_joint": np.asarray(0, dtype=np.int64),
        "u_forward_local": np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        "source_to_canonical_C": identity.copy(),
        "source_to_canonical_alpha": np.asarray(1.0, dtype=np.float64),
        "source_to_canonical_o": np.zeros(3, dtype=np.float64),
        "s_rig": np.asarray(s_rig, dtype=np.float64),
        "rig_id": _text_scalar(rig_id),
        "source_family": _text_scalar("planetzoo"),
        "topology_family": _text_scalar(str(rig_record["topology_family"])),
        "artifact_status": _text_scalar("planetzoo_stage2_fixed_rig_pass"),
        "reason_codes": _text_vector(()),
        "skeleton_format_version": _text_scalar(PLANETZOO_FIXED_RIG_VERSION),
        "representative_clip_id": _text_scalar(representative.stem),
        "source_rest_path": _text_scalar(str(representative)),
        "source_rest_sha256": _text_scalar(source_sha),
        "heading_payload_provenance": _json_scalar(heading_provenance),
        "source_to_canonical_provenance": _json_scalar(provenance),
        "position_geometry_provenance": _json_scalar(provenance),
        "conditioning_authority": _json_scalar(conditioning_authority),
        "unit_metadata": _json_scalar(unit_metadata),
        "joint_map_metadata": _json_scalar(joint_map_metadata),
        "fixed_rig_rotation_signatures": _json_scalar(
            {"status": "identity_stage2_rest", "rest_global_rotation_sha256": identity_sha}
        ),
    }
    _require(
        not any(np.asarray(value).dtype.hasobject for value in payload.values()),
        f"{rig_id}: object dtype is forbidden in skeleton payload",
    )
    return PlanetzooFixedRig(
        joint_names=names,
        parents=parents,
        P_rest_global=P_rest,
        R_rest_global=R_rest_global,
        R_rest_local=R_rest_local,
        offset_parent_local=offsets,
        rotation_source_kind=kinds,
        heading_carrier_joint=0,
        u_forward_local=payload["u_forward_local"],
        s_rig=s_rig,
        source_header_root_offset=declared_offsets[0].copy(),
        payload=payload,
        provenance=provenance,
        metrics=metrics,
    )


def validate_planetzoo_parsed_against_skeleton(
    parsed: ParsedSourceMotion, skeleton: Any
) -> dict[str, float | int | str]:
    """Hard per-clip closure against one already-built stage-2 fixed rig."""
    rig_id = str(skeleton.rig_id)
    _require(
        skeleton.source_family == "planetzoo",
        f"{rig_id}: loaded skeleton source family drifted",
    )
    _require(
        skeleton.artifact_status == "planetzoo_stage2_fixed_rig_pass",
        f"{rig_id}: loaded skeleton artifact status drifted",
    )
    joint_count = len(skeleton.joint_names)
    identity = np.eye(3, dtype=np.float64)
    rest_identity = np.broadcast_to(identity, (joint_count, 3, 3))
    _require(
        np.array_equal(skeleton.R_rest_global, rest_identity),
        f"{rig_id}: loaded global rest rotations are not exact identity",
    )
    _require(
        np.array_equal(skeleton.R_rest_local, rest_identity),
        f"{rig_id}: loaded local rest rotations are not exact identity",
    )
    _require(
        np.array_equal(skeleton.source_to_canonical_C, identity)
        and float(skeleton.source_to_canonical_alpha) == 1.0
        and np.array_equal(
            skeleton.source_to_canonical_o, np.zeros(3, dtype=np.float64)
        ),
        f"{rig_id}: loaded stage-2 transform is not exact C=I, alpha=1, o=0",
    )
    _require(
        int(skeleton.heading_carrier_joint) == 0,
        f"{rig_id}: loaded heading carrier is not the physical root",
    )
    _require(
        np.array_equal(
            skeleton.u_forward_local, np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        ),
        f"{rig_id}: loaded local forward is not exact canonical +Z",
    )
    _require(parsed.family == "planetzoo", f"{rig_id}: parsed family drifted")
    _require(
        parsed.rest_status == PLANETZOO_REST_MODE,
        f"{rig_id}: parsed rest mode is not the fixed stage-2 contract",
    )
    _require(parsed.joint_names == skeleton.joint_names, f"{rig_id}: names drifted")
    _require(
        np.array_equal(parsed.parents, skeleton.parents), f"{rig_id}: parents drifted"
    )
    _require(
        tuple(parsed.rotation_source_kind)
        == tuple(skeleton.rotation_source_kind.astype(str)),
        f"{rig_id}: rotation-source kinds drifted",
    )
    _require(
        bool(np.all(skeleton.rotation_source_kind.astype(str) == "animated_dof")),
        f"{rig_id}: PlanetZoo fixed rig contains a non-animated rotation source",
    )
    provenance = skeleton.metadata.get("position_geometry_provenance")
    _require(isinstance(provenance, Mapping), f"{rig_id}: PZ provenance is absent")
    _require(
        provenance.get("contract_version") == PLANETZOO_FIXED_RIG_VERSION,
        f"{rig_id}: PZ contract version drifted",
    )
    for diagnostic_name, provenance_name in (
        ("source_rotation_layout_sha256", "source_rotation_layout_sha256"),
        ("source_rest_layout_sha256", "source_rest_layout_sha256"),
    ):
        _require(
            parsed.diagnostics.get(diagnostic_name) == provenance.get(provenance_name),
            f"{rig_id}: per-clip {diagnostic_name} drifted",
        )
    _require(
        int(parsed.diagnostics.get("retained_position_channel_joint_count", -1)) == 1
        and int(parsed.diagnostics.get("nonroot_position_channel_joint_count", -1)) == 0,
        f"{rig_id}: per-clip XYZ channel carrier is not root-only",
    )
    declared = np.asarray(parsed.rest_declared_offsets, dtype=np.float64)
    root_offset = np.asarray(provenance.get("source_header_root_offset"), dtype=np.float64)
    _require(
        declared.shape == skeleton.offset_parent_local.shape,
        f"{rig_id}: declared offset shape drifted",
    )
    _require(
        np.array_equal(declared[1:], skeleton.offset_parent_local[1:]),
        f"{rig_id}: per-clip non-root declared offsets differ from fixed rig",
    )
    _require(
        root_offset.shape == (3,) and np.array_equal(declared[0], root_offset),
        f"{rig_id}: per-clip header root offset differs from fixed provenance",
    )
    _require(
        np.array_equal(
            skeleton.offset_parent_local[0], np.zeros(3, dtype=np.float64)
        ),
        f"{rig_id}: fixed-rig root FK offset is not exact zero",
    )
    raw_rest = np.empty_like(declared)
    raw_rest[0] = declared[0]
    for child in range(1, joint_count):
        raw_rest[child] = raw_rest[int(skeleton.parents[child])] + declared[child]
    expected_rest = raw_rest.copy()
    expected_rest[:, 1] -= float(np.min(raw_rest[:, 1]))
    rest_error = float(np.max(np.abs(skeleton.P_rest_global - expected_rest)))
    _require(
        rest_error <= 1e-12,
        f"{rig_id}: loaded grounded rest geometry drifted: {rest_error}",
    )
    expected_s_rig = float(np.linalg.norm(np.ptp(expected_rest, axis=0)))
    _require(
        math.isfinite(expected_s_rig)
        and expected_s_rig > 0.0
        and abs(float(skeleton.s_rig) - expected_s_rig)
        <= 1e-12 * max(1.0, expected_s_rig),
        f"{rig_id}: loaded s_rig drifted from reconstructed stage-2 rest",
    )
    heading = skeleton.metadata.get("heading_payload_provenance")
    _require(isinstance(heading, Mapping), f"{rig_id}: heading provenance is absent")
    try:
        hips = skeleton.joint_names.index("def_c_hips_joint")
        chest = skeleton.joint_names.index("def_c_chest_joint")
    except ValueError as exc:
        raise PlanetzooFixedRigError(
            f"{rig_id}: reviewed hips/chest forward anchors are absent"
        ) from exc
    _require(
        heading.get("forward_method") == "hips_to_chest"
        and tuple(heading.get("forward_anchor_indices", ())) == (hips, chest)
        and tuple(heading.get("forward_anchor_names", ()))
        == (skeleton.joint_names[hips], skeleton.joint_names[chest])
        and int(heading.get("carrier_joint", -1)) == 0
        and heading.get("carrier_name") == skeleton.joint_names[0]
        and np.array_equal(
            np.asarray(heading.get("canonical_rest_forward"), dtype=np.float64),
            np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        ),
        f"{rig_id}: loaded hips-to-chest heading provenance drifted",
    )
    _require(
        np.array_equal(parsed.source_positions[:, 0], parsed.root_translation),
        f"{rig_id}: numeric root translation was not preserved exactly",
    )
    require_so3("PZ parsed global rotations", parsed.global_rotations)
    fixed_positions = _fixed_fk(
        skeleton.parents,
        parsed.root_translation,
        parsed.global_rotations,
        skeleton.offset_parent_local,
    )
    difference = np.linalg.norm(fixed_positions - parsed.source_positions, axis=-1)
    source_position_max_norm = float(np.max(difference) / float(skeleton.s_rig))
    source_position_mpjpe_norm = float(np.mean(difference) / float(skeleton.s_rig))
    _require(
        source_position_max_norm <= PLANETZOO_SOURCE_POSITION_MAX_NORM,
        f"{rig_id}: stage-2 fixed FK/source positions disagree: {source_position_max_norm}",
    )
    return {
        "planetzoo_per_clip_declared_offset_exact": 1,
        "planetzoo_per_clip_rotation_layout_exact": 1,
        "planetzoo_per_clip_rest_layout_exact": 1,
        "planetzoo_root_translation_exact": 1,
        "planetzoo_fixed_fk_source_position_max_norm": source_position_max_norm,
        "planetzoo_fixed_fk_source_position_mpjpe_norm": source_position_mpjpe_norm,
        "planetzoo_stage2_contract": PLANETZOO_FIXED_RIG_VERSION,
    }
