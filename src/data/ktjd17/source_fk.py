"""T03 source-parser and source-FK audit orchestration.

The audit stops before KTJD-17 channel encoding.  In addition to diagnostic
source-FK reproduction, Truebones clips must pass the current-BTJD fixed-rig
contract: cond geometry, original BVH rotations, retained-root translation,
canonical-ready rigid FK, and explicit ignored-XYZ evidence.  It evaluates the
frozen T02 prototype selection plus the read-only held snake and exact-Dragon
strata, then publishes a new immutable manifest generation.
"""

from __future__ import annotations

import dataclasses
import datetime as _datetime
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .inventory import (
    INVENTORY_VERSION,
    REASON_CODES,
    _canonical_json,
    _sha256_file,
    _status_from_codes,
    _write_transaction,
)
from .source_parser import (
    ParsedBvhMotion,
    ParsedSourceMotion,
    SourceParserError,
    parse_bvh_numeric,
    parse_bvh_source,
    parse_motionstreamer272_source,
    source_fk_metrics,
)
from .truebones_fixed_rig import (
    RIGID_EDGE_MAX_NORM,
    ConditioningCatalog,
    FixedRigMotion,
    TRUEBONES_FORWARD_SPECS,
    TruebonesFixedRigError,
    build_fixed_rig_motion,
    load_conditioning_catalog,
)


SOURCE_FK_QA_VERSION = "ktjd17-source-fk-v2"
BASE_MANIFEST_FILES = (
    "clips.jsonl",
    "rigs.jsonl",
    "inventory_summary.json",
    "inventory_reason_codes.json",
    "prototype_candidates.json",
    "prototype_gaps.jsonl",
)
SOURCE_FK_FILES = (
    "source_fk_qa.jsonl",
    "source_fk_summary.json",
    "source_fk_generation.json",
)
SOURCE_FK_MAX_NORM_THRESHOLDS = {
    # BVH source positions and rotations are evaluated in float64.  This floor
    # is deliberately much looser than observed round-off while remaining an
    # exact parser-engineering gate rather than a learned data threshold.
    "truebones": 1e-10,
    "planetzoo": 1e-10,
    # MotionStreamer272 stores rotations/positions after finite-precision
    # preprocessing and omits shape coefficients.  The audit recovers one
    # clip-constant shaped offset from true source rotations and positions.
    "motionstreamer272": 1e-6,
}


class SourceFkAuditError(RuntimeError):
    """The T03 audit scope or manifest transaction is inconsistent."""


@dataclasses.dataclass(frozen=True)
class SourceFkConfig:
    manifest_root: Path
    output_root: Path
    overwrite: bool = False

    def resolved(self) -> "SourceFkConfig":
        return dataclasses.replace(
            self,
            manifest_root=self.manifest_root.expanduser().resolve(),
            output_root=self.output_root.expanduser().absolute(),
        )


@dataclasses.dataclass(frozen=True)
class AuditTarget:
    clip_id: str
    audit_role: str
    calibration_eligible: bool


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise SourceFkAuditError(f"cannot read JSON {path}: {exc}") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise SourceFkAuditError(f"{path}:{line_number}: blank JSONL row")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise SourceFkAuditError(
                        f"{path}:{line_number}: record is not an object"
                    )
                records.append(value)
    except SourceFkAuditError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise SourceFkAuditError(f"cannot read JSONL {path}: {exc}") from exc
    return records


def _jsonl(records: Iterable[dict[str, Any]]) -> str:
    return "".join(_canonical_json(record).decode("utf-8") + "\n" for record in records)


def _select_audit_targets(
    clips: list[dict[str, Any]], candidates: dict[str, Any]
) -> list[AuditTarget]:
    clip_index = {record["clip_id"]: record for record in clips}
    if len(clip_index) != len(clips):
        raise SourceFkAuditError("clips.jsonl contains duplicate clip ids")
    targets: list[AuditTarget] = []

    families = candidates.get("families")
    if not isinstance(families, dict):
        raise SourceFkAuditError("prototype_candidates.json has no families mapping")
    for family in (
        "human",
        "quadruped",
        "winged",
        "spider_crab",
        "dragon_or_deep_topology",
    ):
        payload = families.get(family)
        selected = None if not isinstance(payload, dict) else payload.get(
            "selected_train_candidates"
        )
        if not isinstance(selected, list) or len(selected) != 30:
            raise SourceFkAuditError(
                f"T03 requires the frozen 30-clip train selection for {family}, "
                f"got {None if selected is None else len(selected)}"
            )
        for clip_id in selected:
            record = clip_index.get(clip_id)
            if (
                record is None
                or record.get("topology_family") != family
                or record.get("split") != "train"
                or not record.get("split_eligible_for_train_calibration")
                or not record.get("prototype_candidate")
                or record.get("status") == "reject"
            ):
                raise SourceFkAuditError(
                    f"unsafe or drifted train prototype selection: {family}/{clip_id}"
                )
            targets.append(
                AuditTarget(
                    clip_id=clip_id,
                    audit_role="prototype_train_calibration",
                    calibration_eligible=True,
                )
            )

    snakes = sorted(
        record["clip_id"]
        for record in clips
        if record.get("topology_family") == "snake"
        and record.get("split") == "held_representative"
        and record.get("status") != "reject"
    )
    if len(snakes) != 30:
        raise SourceFkAuditError(
            f"expected 30 read-only held-representative snake clips, got {len(snakes)}"
        )
    targets.extend(
        AuditTarget(
            clip_id=clip_id,
            audit_role="held_representative_read_only",
            calibration_eligible=False,
        )
        for clip_id in snakes
    )

    exact_dragons = sorted(
        record["clip_id"]
        for record in clips
        if record.get("rig_id") == "Dragon"
        and record.get("split") == "held_stress"
        and record.get("status") != "reject"
    )
    if len(exact_dragons) != 13:
        raise SourceFkAuditError(
            f"expected 13 read-only held-stress exact Dragon clips, got {len(exact_dragons)}"
        )
    targets.extend(
        AuditTarget(
            clip_id=clip_id,
            audit_role="held_stress_exact_dragon_read_only",
            calibration_eligible=False,
        )
        for clip_id in exact_dragons
    )

    ids = [target.clip_id for target in targets]
    if len(ids) != 193 or len(set(ids)) != len(ids):
        raise SourceFkAuditError(
            f"T03 audit scope must contain 193 unique clips, got {len(ids)} / "
            f"{len(set(ids))} unique"
        )
    return targets


def _rest_mode(rig: dict[str, Any], source_family: str) -> str:
    if source_family == "planetzoo":
        return "processed_hierarchy_only_review"
    method = rig.get("rest_pose", {}).get("selection_method")
    if method == "explicit_tpose_filename":
        return "explicit_tpose_frame"
    if method in {"legacy_idle_fallback", "legacy_first_file_fallback"}:
        return "legacy_idle_fallback_review"
    raise SourceFkAuditError(
        f"unsupported Truebones rest selection for {rig.get('rig_id')}: {method!r}"
    )


def _parse_target(
    clip: dict[str, Any],
    rig: dict[str, Any],
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
    if family not in {"truebones", "planetzoo"}:
        raise SourceFkAuditError(f"unsupported source family {family!r}")
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


def _float64_output_contract(
    parsed: ParsedSourceMotion, fixed_motion: FixedRigMotion | None = None
) -> dict[str, str]:
    arrays = {
        "parents": parsed.parents,
        "source_joint_indices": parsed.source_joint_indices,
        "root_translation": parsed.root_translation,
        "local_positions": parsed.local_positions,
        "local_rotations": parsed.local_rotations,
        "global_positions": parsed.global_positions,
        "global_rotations": parsed.global_rotations,
        "source_positions": parsed.source_positions,
        "fk_positions": parsed.fk_positions,
        "rest_local_positions": parsed.rest_local_positions,
        "rest_local_rotations": parsed.rest_local_rotations,
        "rest_global_positions": parsed.rest_global_positions,
        "rest_global_rotations": parsed.rest_global_rotations,
    }
    if fixed_motion is not None:
        arrays.update(
            {
                "fixed_C": fixed_motion.C,
                "fixed_o": fixed_motion.o,
                "fixed_P_rest_global": fixed_motion.P_rest_global,
                "fixed_R_rest_global": fixed_motion.R_rest_global,
                "fixed_R_rest_local": fixed_motion.R_rest_local,
                "fixed_offset_parent_local": fixed_motion.offset_parent_local,
                "fixed_P_authoritative": fixed_motion.P_authoritative,
                "fixed_R_global": fixed_motion.R_global,
            }
        )
    result: dict[str, str] = {}
    for name, value in arrays.items():
        array = np.asarray(value)
        if name in {"parents", "source_joint_indices"}:
            if array.dtype != np.int64:
                raise SourceParserError(f"{name} must be int64, got {array.dtype}")
        elif array.dtype != np.float64:
            raise SourceParserError(f"{name} must be float64, got {array.dtype}")
        if not np.isfinite(array).all():
            raise SourceParserError(f"{name} contains non-finite values")
        result[name] = str(array.dtype)
    return result


def _independent_truebones_rotation_agreement(
    parsed: ParsedSourceMotion,
    clip: dict[str, Any],
    rig: dict[str, Any],
) -> dict[str, float | str]:
    """Directly compare producer rotations with the separate SciPy parser."""
    # Local import preserves the validator's one-way rule: the independent
    # module never imports the producer.  T03 runs this same comparison before
    # publication, and the post-publication validator reparses again.
    from .source_fk_validation import (  # noqa: PLC0415
        _parse_bvh_independent,
        _reroot_retained,
    )

    mapping = np.asarray(rig["joint_map"]["btjd_to_source"], dtype=np.int64)
    parents = np.asarray(rig["joint_map"]["btjd_parents"], dtype=np.int64)
    start, end = (int(value) for value in clip["source"]["slice_frames"])
    motion = _parse_bvh_independent(clip["source"]["path"])
    _, _, independent_local = _reroot_retained(
        motion.global_positions[start:end],
        motion.global_rotations[start:end],
        mapping,
        parents,
    )
    independent_global = motion.global_rotations[start:end, mapping]
    rest = _parse_bvh_independent(rig["rest_pose"]["source_path"])
    _, _, independent_rest_local = _reroot_retained(
        rest.global_positions[:1], rest.global_rotations[:1], mapping, parents
    )
    independent_rest_global = rest.global_rotations[0, mapping]
    pairs = {
        "motion_local_rotation_scipy_max_abs": (
            parsed.local_rotations,
            independent_local,
        ),
        "motion_global_rotation_scipy_max_abs": (
            parsed.global_rotations,
            independent_global,
        ),
        "rest_local_rotation_scipy_max_abs": (
            parsed.rest_local_rotations,
            independent_rest_local[0],
        ),
        "rest_global_rotation_scipy_max_abs": (
            parsed.rest_global_rotations,
            independent_rest_global,
        ),
    }
    result: dict[str, float | str] = {
        "comparison": "producer_explicit_axis_product_vs_independent_scipy_full_arrays",
        "threshold_metric": "matrix_max_abs",
        "threshold_max_abs": 1e-12,
    }
    for name, (producer, independent) in pairs.items():
        lhs = np.asarray(producer, dtype=np.float64)
        rhs = np.asarray(independent, dtype=np.float64)
        if lhs.shape != rhs.shape:
            raise TruebonesFixedRigError(
                f"rotation cross-check shape mismatch for {name}: {lhs.shape} != {rhs.shape}"
            )
        value = float(np.max(np.abs(lhs - rhs)))
        if value > 1e-12:
            raise TruebonesFixedRigError(
                f"rotation cross-check failed for {name}: {value}"
            )
        result[name] = value
    return result


def _audit_one(
    target: AuditTarget,
    clip: dict[str, Any],
    rig: dict[str, Any],
    *,
    rest_cache: dict[str, ParsedBvhMotion],
    conditioning_catalog: ConditioningCatalog,
) -> dict[str, Any]:
    family = clip["source"]["family"]
    threshold = SOURCE_FK_MAX_NORM_THRESHOLDS[family]
    base = {
        "qa_version": SOURCE_FK_QA_VERSION,
        "clip_id": target.clip_id,
        "rig_id": clip["rig_id"],
        "topology_family": clip["topology_family"],
        "topology_distance_bucket": clip["topology_distance_bucket"],
        "source_family": family,
        "source_path": clip["source"]["path"],
        "split": clip["split"],
        "audit_role": target.audit_role,
        "calibration_eligible": target.calibration_eligible,
        "threshold_metric": "source_parser_fk_max_norm",
        "threshold_max_norm": threshold,
        "threshold_policy": "predeclared_source_family_engineering_floor",
        "encoder_called": False,
    }
    try:
        parsed = _parse_target(clip, rig, rest_cache=rest_cache)
    except SourceParserError as exc:
        return {
            **base,
            "parser_status": "fail",
            "gate_status": "fail",
            "reason_code": "SOURCE_NUMERIC_PARSE_INVALID",
            "frame_count": None,
            "joint_count": None,
            "fps_src": clip["source"].get("fps_src"),
            "rest_status": None,
            "rest_path": rig.get("rest_pose", {}).get("source_path"),
            "output_dtypes": None,
            "metrics": None,
            "fixed_rig": None,
            "diagnostics": {},
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }

    fixed_motion: FixedRigMotion | None = None
    fixed_record: dict[str, Any] | None = None
    try:
        if family == "truebones":
            try:
                spec = TRUEBONES_FORWARD_SPECS[clip["rig_id"]]
            except KeyError as exc:
                raise TruebonesFixedRigError(
                    f"{clip['rig_id']}: no reviewed forward specification"
                ) from exc
            joint_map = rig["joint_map"]
            fixed_geometry = conditioning_catalog.rig(
                clip["rig_id"],
                expected_names=joint_map["btjd_joint_names"],
                expected_parents=joint_map["btjd_parents"],
            )
            fixed_motion = build_fixed_rig_motion(parsed, fixed_geometry, spec)
            rotation_agreement = _independent_truebones_rotation_agreement(
                parsed, clip, rig
            )
            fixed_record = {
                "status": "pass",
                "threshold_metric": "motion_rigid_edge_max_norm",
                "threshold_max_norm": RIGID_EDGE_MAX_NORM,
                "conditioning": {
                    **conditioning_catalog.authority_record(),
                    "rig_payload_sha256": fixed_geometry.payload_sha256,
                },
                "metrics": fixed_motion.metrics,
                "rotation_signatures": fixed_motion.rotation_signatures,
                "rotation_agreement": rotation_agreement,
                "provenance": {
                    **fixed_motion.provenance,
                    "source_motion_sha256": _sha256_file(
                        Path(str(parsed.path))
                    ),
                    "source_rest_sha256": _sha256_file(
                        Path(str(parsed.rest_path))
                    ),
                },
            }
        dtypes = _float64_output_contract(parsed, fixed_motion)
    except TruebonesFixedRigError as exc:
        return {
            **base,
            "parser_status": "pass",
            "gate_status": "fail",
            "reason_code": "SOURCE_FK_REPRODUCTION_FAILED",
            "frame_count": parsed.frame_count,
            "joint_count": parsed.joint_count,
            "fps_src": parsed.fps,
            "rest_status": parsed.rest_status,
            "rest_path": parsed.rest_path,
            "output_dtypes": None,
            "metrics": source_fk_metrics(parsed),
            "fixed_rig": {
                "status": "fail",
                "threshold_metric": "motion_rigid_edge_max_norm",
                "threshold_max_norm": RIGID_EDGE_MAX_NORM,
            },
            "diagnostics": parsed.diagnostics,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }

    metrics = source_fk_metrics(parsed)
    source_fk_passed = float(metrics["source_parser_fk_max_norm"]) <= threshold
    fixed_passed = fixed_record is None or fixed_record["status"] == "pass"
    passed = source_fk_passed and fixed_passed
    reason_code = None if passed else "SOURCE_FK_REPRODUCTION_FAILED"
    return {
        **base,
        "parser_status": "pass",
        "gate_status": "pass" if passed else "fail",
        "reason_code": reason_code,
        "frame_count": parsed.frame_count,
        "joint_count": parsed.joint_count,
        "fps_src": parsed.fps,
        "rest_status": parsed.rest_status,
        "rest_path": parsed.rest_path,
        "output_dtypes": dtypes,
        "metrics": metrics,
        "fixed_rig": fixed_record,
        "diagnostics": parsed.diagnostics,
        "error": None,
    }


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "p95": None,
            "p99": None,
            "p99_9": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise SourceFkAuditError("cannot summarize non-finite source-FK values")
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "p99_9": float(np.percentile(array, 99.9)),
        "max": float(np.max(array)),
    }


def _group_distributions(
    records: list[dict[str, Any]], *, calibration_eligible: bool
) -> dict[str, Any]:
    groups: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        if record["calibration_eligible"] != calibration_eligible:
            continue
        metrics = record.get("metrics")
        if not isinstance(metrics, dict):
            continue
        value = float(metrics["source_parser_fk_max_norm"])
        groups["source_family"][record["source_family"]].append(value)
        groups["topology_family"][record["topology_family"]].append(value)
        groups["rig_id"][record["rig_id"]].append(value)
    return {
        dimension: {
            name: _distribution(values) for name, values in sorted(by_name.items())
        }
        for dimension, by_name in sorted(groups.items())
    }


def _build_summary(
    records: list[dict[str, Any]],
    *,
    parent_generation_id: str | None,
    conditioning_authority: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gate_counts = Counter(record["gate_status"] for record in records)
    role_counts = Counter(record["audit_role"] for record in records)
    source_counts = Counter(record["source_family"] for record in records)
    topology_counts = Counter(record["topology_family"] for record in records)
    failures = [
        {
            "clip_id": record["clip_id"],
            "reason_code": record["reason_code"],
            "error": record["error"],
            "metrics": record["metrics"],
        }
        for record in records
        if record["gate_status"] != "pass"
    ]
    train_threshold_audit: dict[str, Any] = {}
    for family, engineering_floor in sorted(
        SOURCE_FK_MAX_NORM_THRESHOLDS.items()
    ):
        values = [
            float(record["metrics"]["source_parser_fk_max_norm"])
            for record in records
            if record["source_family"] == family
            and record["calibration_eligible"]
            and record["gate_status"] == "pass"
            and isinstance(record.get("metrics"), dict)
        ]
        selected = SOURCE_FK_MAX_NORM_THRESHOLDS[family]
        failed_train_count = sum(
            record["source_family"] == family
            and bool(record["calibration_eligible"])
            and record["gate_status"] != "pass"
            for record in records
        )
        if not values:
            # A parser outage for a complete source family must still reach the
            # transaction writer so its clips become explicit rejects and its
            # prototype family becomes a shortage.  There is deliberately no
            # percentile fallback to held data or to failed/non-numeric rows.
            if selected < engineering_floor:
                raise SourceFkAuditError(
                    f"{family} source-FK threshold {selected} is below its "
                    f"engineering floor {engineering_floor}"
                )
            train_threshold_audit[family] = {
                "status": "unavailable_no_passing_train_metrics",
                "train_clip_count": 0,
                "failed_train_clip_count": failed_train_count,
                "train_q99_9": None,
                "one_point_five_times_train_q99_9": None,
                "engineering_floor": engineering_floor,
                "selected_threshold": selected,
                "held_data_used": False,
            }
            continue
        q99_9 = float(np.percentile(np.asarray(values, dtype=np.float64), 99.9))
        percentile_candidate = 1.5 * q99_9
        if selected < max(engineering_floor, percentile_candidate):
            raise SourceFkAuditError(
                f"{family} source-FK threshold {selected} is below its train-only "
                f"proposal {max(engineering_floor, percentile_candidate)}"
            )
        train_threshold_audit[family] = {
            "status": "available",
            "train_clip_count": len(values),
            "failed_train_clip_count": failed_train_count,
            "train_q99_9": q99_9,
            "one_point_five_times_train_q99_9": percentile_candidate,
            "engineering_floor": engineering_floor,
            "selected_threshold": selected,
            "held_data_used": False,
        }

    fixed_records = [
        record["fixed_rig"]
        for record in records
        if record.get("source_family") == "truebones"
        and isinstance(record.get("fixed_rig"), dict)
        and record["fixed_rig"].get("status") == "pass"
    ]
    fixed_metric_names = (
        "motion_rigid_edge_max_norm",
        "rest_fk_float64_max_norm",
        "rest_fk_float32_max_norm",
        "raw_xyz_vs_authoritative_max_norm",
        "ignored_nonroot_xyz_max_frame_variation_norm",
    )
    fixed_maxima = {
        name: (
            max(float(record["metrics"][name]) for record in fixed_records)
            if fixed_records
            else None
        )
        for name in fixed_metric_names
    }
    rotation_agreement_max = (
        max(
            float(value)
            for record in fixed_records
            for name, value in record["rotation_agreement"].items()
            if name.endswith("_max_abs") and name != "threshold_max_abs"
        )
        if fixed_records
        else None
    )
    return {
        "qa_version": SOURCE_FK_QA_VERSION,
        "manifest_version": INVENTORY_VERSION,
        "created_at_utc": _datetime.datetime.now(
            _datetime.timezone.utc
        ).isoformat().replace("+00:00", "Z"),
        "parent_inventory_generation_id": parent_generation_id,
        "status": "pass" if not failures else "completed_with_exclusions",
        "scope": {
            "selection": (
                "frozen T02 30x5 safe-train prototypes plus all 30 held-read-only "
                "snake clips and all 13 held-read-only exact Dragon clips"
            ),
            "audited_clip_count": len(records),
            "calibration_clip_count": sum(
                bool(record["calibration_eligible"]) for record in records
            ),
            "read_only_reporting_clip_count": sum(
                not bool(record["calibration_eligible"]) for record in records
            ),
            "held_data_influenced_thresholds": False,
        },
        "thresholds": {
            family: {
                "metric": "source_parser_fk_max_norm",
                "max": threshold,
                "policy": "predeclared_source_family_engineering_floor",
            }
            for family, threshold in sorted(SOURCE_FK_MAX_NORM_THRESHOLDS.items())
        },
        "train_only_threshold_audit": train_threshold_audit,
        "truebones_fixed_rig_contract": {
            "status": "pass" if fixed_records else "unavailable",
            "passing_clip_count": len(fixed_records),
            "rigid_edge_threshold_max_norm": RIGID_EDGE_MAX_NORM,
            "conditioning_authority": conditioning_authority or {},
            "metric_maxima": fixed_maxima,
            "rotation_agreement_metric": "producer_vs_independent_scipy_matrix_max_abs",
            "rotation_agreement_threshold_max_abs": 1e-12,
            "rotation_agreement_observed_max_abs": rotation_agreement_max,
            "rotation_integrity_fingerprint_quantization_step": 1e-8,
            "legacy_btjd_authority": False,
            "raw_nonroot_xyz_authority": False,
        },
        "counts": {
            "gate_status": dict(sorted(gate_counts.items())),
            "audit_role": dict(sorted(role_counts.items())),
            "source_family": dict(sorted(source_counts.items())),
            "topology_family": dict(sorted(topology_counts.items())),
        },
        "calibration_train_only_distributions": _group_distributions(
            records, calibration_eligible=True
        ),
        "held_read_only_distributions": _group_distributions(
            records, calibration_eligible=False
        ),
        "failure_records": failures,
        "encoder_invocation_count": sum(bool(record["encoder_called"]) for record in records),
        "visual_qa_claimed": False,
        "canonicalization_claimed": False,
        "canonical_ready_fixed_rig_audit_claimed": True,
        "ktjd_encoding_claimed": False,
    }


def _update_clip(
    clip: dict[str, Any], qa: dict[str, Any]
) -> dict[str, Any]:
    updated = dict(clip)
    codes = [
        code
        for code in updated["reason_codes"]
        if code != "NUMERIC_PAYLOAD_VALIDATION_DEFERRED_T03"
        and code not in {
            "SOURCE_NUMERIC_PARSE_INVALID",
            "SOURCE_FK_REPRODUCTION_FAILED",
        }
    ]
    if qa["gate_status"] != "pass":
        codes.append(qa["reason_code"])
        updated["prototype_candidate"] = False
        updated["split_eligible_for_train_calibration"] = False
    updated["reason_codes"] = sorted(set(codes))
    updated["status"] = _status_from_codes(updated["reason_codes"])
    updated["source_parser_fk"] = {
        "qa_version": qa["qa_version"],
        "status": qa["gate_status"],
        "parser_status": qa["parser_status"],
        "audit_role": qa["audit_role"],
        "calibration_eligible": qa["calibration_eligible"],
        "threshold_metric": qa["threshold_metric"],
        "threshold_max_norm": qa["threshold_max_norm"],
        "metrics": qa["metrics"],
        "rest_status": qa["rest_status"],
        "encoder_called": qa["encoder_called"],
        "reason_code": qa["reason_code"],
        "fixed_rig": qa.get("fixed_rig"),
    }
    return updated


def _update_candidates_after_failures(
    candidates: dict[str, Any],
    updated_clips: list[dict[str, Any]],
    qa_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    updated = json.loads(json.dumps(candidates))
    clips_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for clip in updated_clips:
        clips_by_family[clip["topology_family"]].append(clip)
    for family, payload in updated["families"].items():
        selected = payload.get("selected_train_candidates", [])
        retained = [
            clip_id
            for clip_id in selected
            if clip_id not in qa_index or qa_index[clip_id]["gate_status"] == "pass"
        ]
        payload["selected_train_candidates"] = retained
        payload["selected_count"] = len(retained)
        payload["rotation_proven_train_candidates"] = sum(
            bool(clip["split_eligible_for_train_calibration"])
            for clip in clips_by_family[family]
        )
        if len(retained) < int(payload["required_train_clips"]):
            payload["status"] = "shortage"
    return updated


def _updated_prototype_gaps(
    existing: list[dict[str, Any]], candidates: dict[str, Any]
) -> list[dict[str, Any]]:
    result = [
        record
        for record in existing
        if not str(record.get("gap_id", "")).startswith("source_fk_shortage:")
    ]
    existing_shortages = {
        record.get("family")
        for record in result
        if "PROTOTYPE_TRAIN_SHORTAGE" in record.get("reason_codes", [])
    }
    for family, payload in candidates["families"].items():
        required = int(payload["required_train_clips"])
        selected = int(payload["selected_count"])
        if selected >= required or family in existing_shortages:
            continue
        result.append(
            {
                "manifest_version": INVENTORY_VERSION,
                "gap_id": f"source_fk_shortage:{family}",
                "family": family,
                "status": "gap",
                "reason_codes": ["PROTOTYPE_TRAIN_SHORTAGE"],
                "required_train_clips": required,
                "source_fk_passing_selected_train_clips": selected,
                "shortage": required - selected,
                "evidence": "T03 source-parser/FK failure removed a frozen T02 prototype",
            }
        )
    return result


def run_source_fk_audit(config: SourceFkConfig) -> dict[str, Any]:
    """Audit and atomically publish the T03 prototype source-FK generation."""
    config = config.resolved()
    root = config.manifest_root
    for name in (*BASE_MANIFEST_FILES, "inventory_generation.json"):
        if not (root / name).is_file():
            raise SourceFkAuditError(f"missing T02 input artifact: {root / name}")
    parent_transaction = _load_json(root / "inventory_generation.json")
    inventory_summary = _load_json(root / "inventory_summary.json")
    try:
        dataset_root = Path(inventory_summary["config"]["dataset_root"])
        conditioning_catalog = load_conditioning_catalog(
            dataset_root / "cond.npy",
            expected_active_sha256=str(inventory_summary["cond_sha256"]),
        )
    except (KeyError, TypeError, TruebonesFixedRigError) as exc:
        raise SourceFkAuditError(
            f"cannot establish Truebones fixed-rig conditioning authority: {exc}"
        ) from exc
    clips = _load_jsonl(root / "clips.jsonl")
    rigs_list = _load_jsonl(root / "rigs.jsonl")
    rigs = {record["rig_id"]: record for record in rigs_list}
    if len(rigs) != len(rigs_list):
        raise SourceFkAuditError("rigs.jsonl contains duplicate rig ids")
    candidates = _load_json(root / "prototype_candidates.json")
    targets = _select_audit_targets(clips, candidates)
    clip_index = {record["clip_id"]: record for record in clips}

    rest_cache: dict[str, ParsedBvhMotion] = {}
    qa_records: list[dict[str, Any]] = []
    for index, target in enumerate(targets, start=1):
        clip = clip_index[target.clip_id]
        rig = rigs[clip["rig_id"]]
        qa_records.append(
            _audit_one(
                target,
                clip,
                rig,
                rest_cache=rest_cache,
                conditioning_catalog=conditioning_catalog,
            )
        )
        if index % 10 == 0 or index == len(targets):
            print(f"[source-fk] audited {index}/{len(targets)}", flush=True)

    qa_index = {record["clip_id"]: record for record in qa_records}
    if len(qa_index) != len(qa_records):
        raise SourceFkAuditError("duplicate source-FK QA records")
    updated_clips = [
        _update_clip(record, qa_index[record["clip_id"]])
        if record["clip_id"] in qa_index
        else record
        for record in clips
    ]
    updated_candidates = _update_candidates_after_failures(
        candidates, updated_clips, qa_index
    )
    existing_gaps = _load_jsonl(root / "prototype_gaps.jsonl")
    updated_gaps = _updated_prototype_gaps(existing_gaps, updated_candidates)
    summary = _build_summary(
        qa_records,
        parent_generation_id=parent_transaction.get("generation_id"),
        conditioning_authority=conditioning_catalog.authority_record(),
    )

    inventory_summary["source_fk_audit"] = {
        "qa_version": SOURCE_FK_QA_VERSION,
        "status": summary["status"],
        "audited_clip_count": summary["scope"]["audited_clip_count"],
        "calibration_clip_count": summary["scope"]["calibration_clip_count"],
        "read_only_reporting_clip_count": summary["scope"][
            "read_only_reporting_clip_count"
        ],
        "gate_status_counts": summary["counts"]["gate_status"],
        "held_data_influenced_thresholds": False,
        "truebones_fixed_rig_contract": summary["truebones_fixed_rig_contract"],
    }
    inventory_summary["fresh_counts"]["clip_status_counts"] = dict(
        sorted(Counter(record["status"] for record in updated_clips).items())
    )
    inventory_summary["prototype_families"] = updated_candidates["families"]
    inventory_summary["prototype_gap_records"] = updated_gaps
    reason_table = _load_json(root / "inventory_reason_codes.json")
    reason_table["codes"] = REASON_CODES

    generation_record = {
        "qa_version": SOURCE_FK_QA_VERSION,
        "manifest_version": INVENTORY_VERSION,
        "parent_inventory_generation_id": parent_transaction.get("generation_id"),
        "parent_inventory_files": parent_transaction.get("files"),
        "audit_scope_clip_count": len(qa_records),
        "truebones_conditioning_authority": conditioning_catalog.authority_record(),
        "encoder_called": False,
        "next_stage": "T04 canonical skeleton derivation",
    }
    outputs: dict[str, str] = {
        "clips.jsonl": _jsonl(updated_clips),
        "rigs.jsonl": _jsonl(rigs_list),
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
        "prototype_gaps.jsonl": _jsonl(updated_gaps),
        "source_fk_qa.jsonl": _jsonl(qa_records),
        "source_fk_summary.json": json.dumps(
            summary,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        "source_fk_generation.json": json.dumps(
            generation_record,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
    }
    transaction = _write_transaction(
        config.output_root,
        outputs,
        overwrite=config.overwrite,
    )
    result = dict(summary)
    result["generation_id"] = transaction["generation_id"]
    result["transaction_file_count"] = len(transaction["files"])
    return result
