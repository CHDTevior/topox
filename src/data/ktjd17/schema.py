"""Writer and fail-closed validator for the KTJD-17 v1 dataset schema.

This module covers the static M0 contract only.  Calibration-dependent values
may be represented in an explicitly ``unfrozen`` schema, but downstream full
conversion, evaluation, and training must call :func:`validate_schema` with
``require_frozen=True``.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


KTJD17_REPR_VERSION = "ktjd17-v1"
KTJD17_D = 17
KTJD17_ROOT_INDEX = 0
KTJD17_SOURCE_PLAN = "handoff/20260819_ktjd17_multitopology_design.md"
KTJD17_SOURCE_PLAN_COMMIT = "9181f5cccbad23e941bf94c2874daf36e7f288cf"

CHANNEL_SLICES: dict[str, list[int]] = {
    "q_position": [0, 3],
    "global_rest_delta_6d": [3, 9],
    "world_velocity": [9, 12],
    "contact": [12, 13],
    "smooth_root_xz": [13, 15],
    "heading": [15, 17],
}

ROOT_CHANNEL_VALIDITY = [True] * KTJD17_D
NON_ROOT_CHANNEL_VALIDITY = [True] * 13 + [False] * 4
PADDED_CHANNEL_VALIDITY = [False] * KTJD17_D

MOTION_REQUIRED_KEYS = [
    "motion",
    "heading_valid",
    "clip_id",
    "rig_id",
    "fps_target",
    "origin_xz",
]
SKELETON_REQUIRED_KEYS = [
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
]

CALIBRATION_FIELDS = [
    "fps_target",
    "smoother.id",
    "smoother.params",
    "smoother.short_clip_rule",
    "heading.eps_h",
    "contact.tau_h",
    "contact.tau_v",
    "normalization.gains",
    "topology.J_max",
]

MASK_DERIVATION = {
    "frame_mask": "frame_mask[t] = (t < T_valid)",
    "joint_mask": "joint_mask[j] = (j < J_phys)",
    "channel_valid_mask_common": "channel_valid_mask[j,0:13] = joint_mask[j]",
    "channel_valid_mask_root": "channel_valid_mask[0,13:17] = joint_mask[0]",
    "channel_valid_mask_nonroot": "channel_valid_mask[j>0,13:17] = False",
    "rotation_supervised": (
        "rotation_supervised[j] = joint_mask[j] and "
        "rotation_source_kind[j] == animated_dof"
    ),
    "fixed_rotation_mask": (
        "fixed_rotation_mask[j] = joint_mask[j] and "
        "rotation_source_kind[j] == fixed_dof"
    ),
    "contact_supervised": "contact_supervised[j] = joint_mask[j]",
    "child_edge_valid_root": "child_edge_valid[0] = False",
    "child_edge_valid_child": (
        "child_edge_valid[c>0] = joint_mask[c] and joint_mask[parents[c]]"
    ),
}

TOP_LEVEL_KEYS = {
    "repr_version",
    "D",
    "root_index",
    "channel_slices",
    "channel_validity",
    "coordinate",
    "rot6d",
    "fps_target",
    "velocity",
    "smoother",
    "heading",
    "contact",
    "normalization",
    "yaw",
    "dtype",
    "topology",
    "artifacts",
    "mask_derivation",
    "units",
    "calibration",
    "provenance",
}


class SchemaValidationError(ValueError):
    """A KTJD-17 schema or associated static payload violates the contract."""


def _fail(path: str, message: str) -> None:
    raise SchemaValidationError(f"{path}: {message}")


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, f"expected object, got {type(value).__name__}")
    return value


def _require_exact_keys(value: Any, expected: set[str], path: str) -> Mapping[str, Any]:
    mapping = _require_mapping(value, path)
    actual = set(mapping)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        _fail(path, f"key mismatch; missing={missing}, extra={extra}")
    return mapping


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite_number(value: Any, path: str, *, positive: bool = False,
                   nonnegative: bool = False) -> float:
    if not _is_number(value):
        _fail(path, f"expected finite number, got {value!r}")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise SchemaValidationError(
            f"{path}: expected representable finite number, got {type(value).__name__}"
        ) from exc
    if not math.isfinite(number):
        _fail(path, f"must be finite, got {value!r}")
    if positive and number <= 0:
        _fail(path, f"must be > 0, got {number}")
    if nonnegative and number < 0:
        _fail(path, f"must be >= 0, got {number}")
    return number


def _validate_json_finite(value: Any, path: str) -> None:
    """Reject non-JSON values and every NaN/Inf without silently replacing it."""
    if value is None or isinstance(value, (str, bool)):
        return
    if _is_number(value):
        _finite_number(value, path)
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                _fail(path, f"JSON object key must be str, got {type(key).__name__}")
            _validate_json_finite(child, f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _validate_json_finite(child, f"{path}[{index}]")
        return
    _fail(path, f"not a JSON-compatible value: {type(value).__name__}")


def _require_exact_value(actual: Any, expected: Any, path: str) -> None:
    if actual != expected:
        _fail(path, f"expected {expected!r}, got {actual!r}")


def _require_optional_text(value: Any, path: str, *, frozen: bool) -> None:
    if value is None and not frozen:
        return
    if not isinstance(value, str) or not value.strip():
        _fail(path, "must be a non-empty string" + ("" if frozen else " or null while unfrozen"))


def _require_optional_positive(value: Any, path: str, *, frozen: bool,
                               nonnegative: bool = False) -> None:
    if value is None and not frozen:
        return
    _finite_number(value, path, positive=not nonnegative, nonnegative=nonnegative)


def build_schema(
    *,
    fps_target: float = 30.0,
    smoother_id: str | None = None,
    smoother_params: Mapping[str, Any] | None = None,
    short_clip_rule: str | None = None,
    heading_eps_h: float | None = None,
    contact_tau_h: float | None = None,
    contact_tau_v: float | None = None,
    normalization_gains: Sequence[float] | None = None,
    j_max: int | None = None,
    frozen: bool = False,
    source_plan: str = KTJD17_SOURCE_PLAN,
    source_plan_commit: str = KTJD17_SOURCE_PLAN_COMMIT,
    calibration_run_ids: Sequence[str] | None = None,
    train_split_protocol: str | None = None,
    frozen_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic KTJD-17 schema object and validate it.

    ``fps_target=30`` is only a candidate until ``frozen=True``.  An unfrozen
    schema deliberately carries null calibration values and cannot pass a
    ``require_frozen`` consumer gate.
    """
    fps = _finite_number(fps_target, "fps_target", positive=True)
    gains = list(normalization_gains) if normalization_gains is not None else [None] * 3
    schema: dict[str, Any] = {
        "repr_version": KTJD17_REPR_VERSION,
        "D": KTJD17_D,
        "root_index": KTJD17_ROOT_INDEX,
        "channel_slices": {key: list(value) for key, value in CHANNEL_SLICES.items()},
        "channel_validity": {
            "physical_root": list(ROOT_CHANNEL_VALIDITY),
            "physical_non_root": list(NON_ROOT_CHANNEL_VALIDITY),
            "padded_joint": list(PADDED_CHANNEL_VALIDITY),
            "invalid_storage_value": 0.0,
            "invalid_policy": "mask_out_of_loss_and_statistics",
        },
        "coordinate": {
            "handedness": "right",
            "up": "+Y",
            "ground_plane": "XZ",
            "rest_forward": "+Z",
            "rotation_action": "active",
            "vector_convention": "column",
        },
        "rot6d": {
            "order": "R00,R10,R20,R01,R11,R21",
            "decoder": "gram_schmidt",
            "eps": 1e-8,
            "third_axis": "b1_cross_b2",
        },
        "fps_target": fps,
        "velocity": {
            "difference": "forward",
            "units": "length_per_second",
            "tail": "repeat_last",
        },
        "smoother": {
            "id": smoother_id,
            "params": dict(smoother_params or {}),
            "short_clip_rule": short_clip_rule,
        },
        "heading": {
            "eps_h": heading_eps_h,
            "invalid_sentinel": [0.0, 0.0],
            "invalid_policy": "mask",
        },
        "contact": {
            "definition": "joint_proxy_ground_support",
            "tau_h": contact_tau_h,
            "tau_v": contact_tau_v,
            "tail": "repeat_last",
        },
        "normalization": {
            "rig_scale": "rest_aabb_diagonal",
            "gains": gains,
            "center": False,
        },
        "yaw": {
            "Y": "[[c,0,s],[0,1,0],[-s,0,c]]",
            "Y_xz": "[[c,s],[-s,c]]",
            "heading_rot2": "[[c,-s],[s,c]]",
            "invalid_heading": "keep_zero",
        },
        "dtype": {"build": "float64", "storage": "float32"},
        "topology": {
            "node_semantics": "physical_joint_only",
            "virtual_nodes_allowed": False,
            "root_index": 0,
            "parent_order": "parents[0]=-1;parents[c]<c",
            "J_max": j_max,
        },
        "artifacts": {
            "dataset_root": "dataset",
            "schema": "dataset/schema.json",
            "motion_pattern": "dataset/motions/<clip_id>.npz",
            "skeleton_pattern": "dataset/skeletons/<rig_id>.npz",
            "manifest": "dataset/manifests/clips.jsonl",
            "split_pattern": "dataset/splits/<protocol>/*.txt",
            "train_stats": "dataset/stats/train_block_gains.npz",
            "motion_required_keys": list(MOTION_REQUIRED_KEYS),
            "skeleton_required_keys": list(SKELETON_REQUIRED_KEYS),
            "raw_motion_padding": "none",
            "raw_motion_normalization": "none",
        },
        "mask_derivation": dict(MASK_DERIVATION),
        "units": {
            "default_value_label": "TopoX canonical length unit",
            "default_claims_meters": False,
            "meter_claim_requires": "finite_positive_source_unit_to_meter",
            "required_skeleton_fields": [
                "length_unit_id",
                "source_unit_to_meter",
                "canonical_scale_factor",
                "s_rig",
            ],
        },
        "calibration": {
            "status": "frozen" if frozen else "unfrozen",
            "scope": "train_only",
            "validation_or_held_tuning_allowed": False,
            "unresolved_fields": [] if frozen else list(CALIBRATION_FIELDS),
        },
        "provenance": {
            "source_plan": source_plan,
            "source_plan_commit": source_plan_commit,
            "schema_writer": "src.data.ktjd17.schema",
            "schema_writer_version": 1,
            "calibration_run_ids": list(calibration_run_ids or []),
            "train_split_protocol": train_split_protocol,
            "frozen_at_utc": frozen_at_utc,
        },
    }
    validate_schema(schema, require_frozen=frozen)
    return schema


def validate_schema(
    schema: Mapping[str, Any],
    *,
    expected_fps_target: float | None = None,
    require_frozen: bool = False,
) -> None:
    """Validate the exact KTJD-17 v1 schema, with no legacy-key fallback."""
    root = _require_exact_keys(schema, TOP_LEVEL_KEYS, "schema")
    _validate_json_finite(root, "schema")

    _require_exact_value(root["repr_version"], KTJD17_REPR_VERSION, "schema.repr_version")
    _require_exact_value(root["D"], KTJD17_D, "schema.D")
    _require_exact_value(root["root_index"], KTJD17_ROOT_INDEX, "schema.root_index")

    slices = _require_exact_keys(root["channel_slices"], set(CHANNEL_SLICES),
                                 "schema.channel_slices")
    for name, expected in CHANNEL_SLICES.items():
        _require_exact_value(slices[name], expected, f"schema.channel_slices.{name}")

    validity = _require_exact_keys(
        root["channel_validity"],
        {"physical_root", "physical_non_root", "padded_joint", "invalid_storage_value",
         "invalid_policy"},
        "schema.channel_validity",
    )
    _require_exact_value(validity["physical_root"], ROOT_CHANNEL_VALIDITY,
                         "schema.channel_validity.physical_root")
    _require_exact_value(validity["physical_non_root"], NON_ROOT_CHANNEL_VALIDITY,
                         "schema.channel_validity.physical_non_root")
    _require_exact_value(validity["padded_joint"], PADDED_CHANNEL_VALIDITY,
                         "schema.channel_validity.padded_joint")
    _require_exact_value(validity["invalid_storage_value"], 0.0,
                         "schema.channel_validity.invalid_storage_value")
    _require_exact_value(validity["invalid_policy"], "mask_out_of_loss_and_statistics",
                         "schema.channel_validity.invalid_policy")

    coordinate = _require_exact_keys(
        root["coordinate"],
        {"handedness", "up", "ground_plane", "rest_forward", "rotation_action",
         "vector_convention"},
        "schema.coordinate",
    )
    expected_coordinate = {
        "handedness": "right",
        "up": "+Y",
        "ground_plane": "XZ",
        "rest_forward": "+Z",
        "rotation_action": "active",
        "vector_convention": "column",
    }
    _require_exact_value(dict(coordinate), expected_coordinate, "schema.coordinate")

    rot6d = _require_exact_keys(root["rot6d"], {"order", "decoder", "eps", "third_axis"},
                                "schema.rot6d")
    _require_exact_value(rot6d["order"], "R00,R10,R20,R01,R11,R21", "schema.rot6d.order")
    _require_exact_value(rot6d["decoder"], "gram_schmidt", "schema.rot6d.decoder")
    _require_exact_value(rot6d["eps"], 1e-8, "schema.rot6d.eps")
    _require_exact_value(rot6d["third_axis"], "b1_cross_b2", "schema.rot6d.third_axis")

    fps = _finite_number(root["fps_target"], "schema.fps_target", positive=True)
    if expected_fps_target is not None:
        expected_fps = _finite_number(expected_fps_target, "expected_fps_target", positive=True)
        if fps != expected_fps:
            _fail("schema.fps_target", f"mismatch: schema={fps}, expected={expected_fps}")

    velocity = _require_exact_keys(root["velocity"], {"difference", "units", "tail"},
                                   "schema.velocity")
    _require_exact_value(dict(velocity), {
        "difference": "forward",
        "units": "length_per_second",
        "tail": "repeat_last",
    }, "schema.velocity")

    calibration = _require_exact_keys(
        root["calibration"],
        {"status", "scope", "validation_or_held_tuning_allowed", "unresolved_fields"},
        "schema.calibration",
    )
    status = calibration["status"]
    if status not in {"unfrozen", "frozen"}:
        _fail("schema.calibration.status", f"expected 'unfrozen' or 'frozen', got {status!r}")
    frozen = status == "frozen"
    _require_exact_value(calibration["scope"], "train_only", "schema.calibration.scope")
    _require_exact_value(calibration["validation_or_held_tuning_allowed"], False,
                         "schema.calibration.validation_or_held_tuning_allowed")
    _require_exact_value(
        calibration["unresolved_fields"],
        [] if frozen else CALIBRATION_FIELDS,
        "schema.calibration.unresolved_fields",
    )
    if require_frozen and not frozen:
        _fail("schema.calibration.status", "consumer requires a frozen schema")

    smoother = _require_exact_keys(root["smoother"], {"id", "params", "short_clip_rule"},
                                   "schema.smoother")
    _require_optional_text(smoother["id"], "schema.smoother.id", frozen=frozen)
    _require_mapping(smoother["params"], "schema.smoother.params")
    _require_optional_text(smoother["short_clip_rule"], "schema.smoother.short_clip_rule",
                           frozen=frozen)

    heading = _require_exact_keys(root["heading"], {"eps_h", "invalid_sentinel", "invalid_policy"},
                                  "schema.heading")
    _require_optional_positive(heading["eps_h"], "schema.heading.eps_h", frozen=frozen)
    _require_exact_value(heading["invalid_sentinel"], [0.0, 0.0],
                         "schema.heading.invalid_sentinel")
    _require_exact_value(heading["invalid_policy"], "mask", "schema.heading.invalid_policy")

    contact = _require_exact_keys(root["contact"], {"definition", "tau_h", "tau_v", "tail"},
                                  "schema.contact")
    _require_exact_value(contact["definition"], "joint_proxy_ground_support",
                         "schema.contact.definition")
    _require_optional_positive(contact["tau_h"], "schema.contact.tau_h", frozen=frozen,
                               nonnegative=True)
    _require_optional_positive(contact["tau_v"], "schema.contact.tau_v", frozen=frozen,
                               nonnegative=True)
    _require_exact_value(contact["tail"], "repeat_last", "schema.contact.tail")

    normalization = _require_exact_keys(root["normalization"], {"rig_scale", "gains", "center"},
                                        "schema.normalization")
    _require_exact_value(normalization["rig_scale"], "rest_aabb_diagonal",
                         "schema.normalization.rig_scale")
    gains = normalization["gains"]
    if not isinstance(gains, list) or len(gains) != 3:
        _fail("schema.normalization.gains", f"expected length-3 list, got {gains!r}")
    for index, gain in enumerate(gains):
        _require_optional_positive(gain, f"schema.normalization.gains[{index}]", frozen=frozen)
    _require_exact_value(normalization["center"], False, "schema.normalization.center")

    yaw = _require_exact_keys(root["yaw"], {"Y", "Y_xz", "heading_rot2", "invalid_heading"},
                              "schema.yaw")
    _require_exact_value(dict(yaw), {
        "Y": "[[c,0,s],[0,1,0],[-s,0,c]]",
        "Y_xz": "[[c,s],[-s,c]]",
        "heading_rot2": "[[c,-s],[s,c]]",
        "invalid_heading": "keep_zero",
    }, "schema.yaw")
    dtype = _require_exact_keys(root["dtype"], {"build", "storage"}, "schema.dtype")
    _require_exact_value(dict(dtype), {"build": "float64", "storage": "float32"},
                         "schema.dtype")

    topology = _require_exact_keys(
        root["topology"],
        {"node_semantics", "virtual_nodes_allowed", "root_index", "parent_order", "J_max"},
        "schema.topology",
    )
    _require_exact_value(topology["node_semantics"], "physical_joint_only",
                         "schema.topology.node_semantics")
    _require_exact_value(topology["virtual_nodes_allowed"], False,
                         "schema.topology.virtual_nodes_allowed")
    _require_exact_value(topology["root_index"], 0, "schema.topology.root_index")
    _require_exact_value(topology["parent_order"], "parents[0]=-1;parents[c]<c",
                         "schema.topology.parent_order")
    j_max = topology["J_max"]
    if j_max is None and not frozen:
        pass
    elif not isinstance(j_max, int) or isinstance(j_max, bool) or j_max <= 0:
        _fail("schema.topology.J_max", f"must be a positive integer, got {j_max!r}")

    artifacts = _require_exact_keys(
        root["artifacts"],
        {"dataset_root", "schema", "motion_pattern", "skeleton_pattern", "manifest",
         "split_pattern", "train_stats", "motion_required_keys", "skeleton_required_keys",
         "raw_motion_padding", "raw_motion_normalization"},
        "schema.artifacts",
    )
    expected_artifacts = {
        "dataset_root": "dataset",
        "schema": "dataset/schema.json",
        "motion_pattern": "dataset/motions/<clip_id>.npz",
        "skeleton_pattern": "dataset/skeletons/<rig_id>.npz",
        "manifest": "dataset/manifests/clips.jsonl",
        "split_pattern": "dataset/splits/<protocol>/*.txt",
        "train_stats": "dataset/stats/train_block_gains.npz",
        "motion_required_keys": MOTION_REQUIRED_KEYS,
        "skeleton_required_keys": SKELETON_REQUIRED_KEYS,
        "raw_motion_padding": "none",
        "raw_motion_normalization": "none",
    }
    _require_exact_value(dict(artifacts), expected_artifacts, "schema.artifacts")

    masks = _require_exact_keys(root["mask_derivation"], set(MASK_DERIVATION),
                                "schema.mask_derivation")
    _require_exact_value(dict(masks), MASK_DERIVATION, "schema.mask_derivation")

    units = _require_exact_keys(
        root["units"],
        {"default_value_label", "default_claims_meters", "meter_claim_requires",
         "required_skeleton_fields"},
        "schema.units",
    )
    _require_exact_value(dict(units), {
        "default_value_label": "TopoX canonical length unit",
        "default_claims_meters": False,
        "meter_claim_requires": "finite_positive_source_unit_to_meter",
        "required_skeleton_fields": [
            "length_unit_id", "source_unit_to_meter", "canonical_scale_factor", "s_rig"
        ],
    }, "schema.units")

    provenance = _require_exact_keys(
        root["provenance"],
        {"source_plan", "source_plan_commit", "schema_writer", "schema_writer_version",
         "calibration_run_ids", "train_split_protocol", "frozen_at_utc"},
        "schema.provenance",
    )
    for key in ("source_plan", "source_plan_commit", "schema_writer"):
        _require_optional_text(provenance[key], f"schema.provenance.{key}", frozen=True)
    _require_exact_value(provenance["source_plan"], KTJD17_SOURCE_PLAN,
                         "schema.provenance.source_plan")
    _require_exact_value(provenance["source_plan_commit"], KTJD17_SOURCE_PLAN_COMMIT,
                         "schema.provenance.source_plan_commit")
    _require_exact_value(provenance["schema_writer"], "src.data.ktjd17.schema",
                         "schema.provenance.schema_writer")
    _require_exact_value(provenance["schema_writer_version"], 1,
                         "schema.provenance.schema_writer_version")
    run_ids = provenance["calibration_run_ids"]
    if not isinstance(run_ids, list) or any(not isinstance(item, str) or not item.strip()
                                            for item in run_ids):
        _fail("schema.provenance.calibration_run_ids", "must be a list of non-empty strings")
    if frozen:
        if not run_ids:
            _fail("schema.provenance.calibration_run_ids", "frozen schema requires run ids")
        _require_optional_text(provenance["train_split_protocol"],
                               "schema.provenance.train_split_protocol", frozen=True)
        _require_optional_text(provenance["frozen_at_utc"],
                               "schema.provenance.frozen_at_utc", frozen=True)
    else:
        if provenance["train_split_protocol"] is not None or provenance["frozen_at_utc"] is not None:
            _fail("schema.provenance", "unfrozen schema cannot claim frozen split/time provenance")


def validate_physical_parent_tree(
    parents: Sequence[int],
    *,
    node_kinds: Sequence[str] | None = None,
) -> None:
    """Validate root-first physical-joint-only parent order.

    Required ``node_kinds`` avoids unsafe name heuristics: a source joint name
    may contain the word "world", but its normalized node kind must still be
    ``physical_joint``.  Virtual WORLD/control tokens therefore fail closed.
    """
    if isinstance(parents, (str, bytes, bytearray)) or not isinstance(parents, Sequence):
        _fail("parents", f"expected sequence, got {type(parents).__name__}")
    if not parents:
        _fail("parents", "tree must contain at least one physical joint")
    for index, parent in enumerate(parents):
        if not isinstance(parent, int) or isinstance(parent, bool):
            _fail(f"parents[{index}]", f"expected int, got {parent!r}")
        if index == 0:
            if parent != -1:
                _fail("parents[0]", f"physical root must be -1, got {parent}")
        elif parent < 0 or parent >= index:
            _fail(f"parents[{index}]", f"must satisfy 0 <= parent < child index {index}, got {parent}")

    if node_kinds is None:
        _fail("node_kinds", "required to prove every stored node is a physical joint")
    if isinstance(node_kinds, (str, bytes, bytearray)) or not isinstance(node_kinds, Sequence):
        _fail("node_kinds", f"expected sequence, got {type(node_kinds).__name__}")
    if len(node_kinds) != len(parents):
        _fail("node_kinds", f"length {len(node_kinds)} != parents length {len(parents)}")
    for index, kind in enumerate(node_kinds):
        if kind != "physical_joint":
            _fail(f"node_kinds[{index}]", f"virtual/WORLD/control node kind is forbidden: {kind!r}")


def validate_unit_metadata(metadata: Mapping[str, Any], *, claims_meters: bool = False) -> None:
    """Validate per-rig length-unit evidence without inventing metric units."""
    value = _require_exact_keys(
        metadata,
        {"length_unit_id", "source_unit_to_meter", "canonical_scale_factor", "s_rig"},
        "unit_metadata",
    )
    if not isinstance(value["length_unit_id"], str) or not value["length_unit_id"].strip():
        _fail("unit_metadata.length_unit_id", "must be a non-empty string")
    source_scale = value["source_unit_to_meter"]
    if source_scale is not None:
        _finite_number(source_scale, "unit_metadata.source_unit_to_meter", positive=True)
    if claims_meters and source_scale is None:
        _fail("unit_metadata.source_unit_to_meter",
              "a meter claim requires finite positive source-unit evidence")
    _finite_number(value["canonical_scale_factor"], "unit_metadata.canonical_scale_factor",
                   positive=True)
    _finite_number(value["s_rig"], "unit_metadata.s_rig", positive=True)


def load_schema(
    path: str | Path,
    *,
    expected_fps_target: float | None = None,
    require_frozen: bool = False,
) -> dict[str, Any]:
    """Load JSON and validate before returning it."""
    schema_path = Path(path)
    if not schema_path.is_file():
        raise FileNotFoundError(f"KTJD-17 schema not found: {schema_path}")
    try:
        value = json.loads(schema_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(f"{schema_path}: invalid JSON: {exc}") from exc
    validate_schema(value, expected_fps_target=expected_fps_target, require_frozen=require_frozen)
    return value


def write_schema(
    schema: Mapping[str, Any],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Validate and atomically write a schema, protecting frozen artifacts."""
    validate_schema(schema)
    output = Path(path)
    serialized = json.dumps(schema, ensure_ascii=False, indent=2) + "\n"

    if output.exists():
        existing = load_schema(output)
        existing_text = json.dumps(existing, ensure_ascii=False, indent=2) + "\n"
        if existing_text == serialized:
            return output
        if existing["calibration"]["status"] == "frozen":
            raise SchemaValidationError(f"refusing to overwrite frozen KTJD-17 schema: {output}")
        if not overwrite:
            raise FileExistsError(f"schema exists with different content: {output}; pass overwrite=True")

    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp",
                                          dir=str(output.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return output
