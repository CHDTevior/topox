"""Train-only numeric calibration for an immutable KTJD-17 prototype.

The calibration pass never mutates the prototype, schema, or held data.  It
publishes an immutable candidate generation whose gains and distributions can
only become frozen after the separate visual/review gate.
"""

from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import math
import os
import shutil
import tempfile
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .decoder import decode_ktjd17
from .encoder import load_skeleton
from .loader import load_motion_npz


CALIBRATION_VERSION = "ktjd17-train-only-calibration-v1"
CALIBRATION_GENERATION_DIRECTORY = ".ktjd17_calibration_generations"
CALIBRATION_LINK_NAME = "ktjd17_calibration_candidate"
POSITION_ANCHOR_EPS_NORM = 1e-8
POSITION_ANCHOR_REVIEW_RAD = math.pi / 6.0
HEADING_EPS_SWEEP = (0.01, 0.02, 0.05, 0.10)


class CalibrationError(RuntimeError):
    """Calibration input or output violated the train-only contract."""


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise CalibrationError(f"cannot read JSON {path}: {exc}") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise CalibrationError(f"{path}:{line_number}: blank JSONL row")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise CalibrationError(f"{path}:{line_number}: row is not an object")
                records.append(value)
    except CalibrationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise CalibrationError(f"cannot read JSONL {path}: {exc}") from exc
    return records


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())


def _resolve_generation_path(root: Path, relpath: Any, *, label: str) -> Path:
    if not isinstance(relpath, str) or not relpath:
        raise CalibrationError(f"{label}: invalid relative path")
    relative = Path(relpath)
    if relative.is_absolute():
        raise CalibrationError(f"{label}: absolute path is forbidden")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise CalibrationError(f"{label}: path escapes prototype generation")
    return resolved


def summarize_distribution(values: np.ndarray | Sequence[float]) -> dict[str, Any]:
    """Return deterministic finite distribution statistics with tail quantiles."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {"count": 0}
    if not np.isfinite(array).all():
        location = int(np.flatnonzero(~np.isfinite(array))[0])
        raise CalibrationError(f"distribution contains non-finite value at {location}")
    quantiles = np.percentile(array, [0.1, 1.0, 5.0, 50.0, 95.0, 99.0, 99.9])
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "q00_1": float(quantiles[0]),
        "q01": float(quantiles[1]),
        "q05": float(quantiles[2]),
        "median": float(quantiles[3]),
        "q95": float(quantiles[4]),
        "q99": float(quantiles[5]),
        "q99_9": float(quantiles[6]),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "rms": float(np.sqrt(np.mean(np.square(array)))),
    }


def gains_from_sums(
    sum_squares: Mapping[str, float], counts: Mapping[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Compute reciprocal-RMS gains for q, velocity, and smooth-root blocks."""
    rms_values: list[float] = []
    gain_values: list[float] = []
    for key in ("q", "v", "s"):
        count = int(counts.get(key, 0))
        square_sum = float(sum_squares.get(key, math.nan))
        if count <= 0 or not math.isfinite(square_sum) or square_sum <= 0.0:
            raise CalibrationError(
                f"cannot calibrate {key}: count={count}, sum_squares={square_sum}"
            )
        rms = math.sqrt(square_sum / count)
        gain = 1.0 / rms
        if not math.isfinite(gain) or gain <= 0.0:
            raise CalibrationError(f"invalid {key} normalization gain {gain}")
        rms_values.append(rms)
        gain_values.append(gain)
    return np.asarray(gain_values, dtype=np.float64), np.asarray(
        rms_values, dtype=np.float64
    )


def derive_position_anchor_heading(
    positions: np.ndarray,
    *,
    method: str,
    anchor_indices: Sequence[int],
    s_rig: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive an independent per-frame heading from reviewed position anchors."""
    points = np.asarray(positions, dtype=np.float64)
    scale = float(s_rig)
    if (
        points.ndim != 3
        or points.shape[-1] != 3
        or not np.isfinite(points).all()
        or not math.isfinite(scale)
        or scale <= 0.0
    ):
        raise CalibrationError("invalid position-anchor heading inputs")
    indices = tuple(int(value) for value in anchor_indices)
    if any(index < 0 or index >= points.shape[1] for index in indices):
        raise CalibrationError(f"position-anchor index outside J={points.shape[1]}")
    if method == "lateral_pairs":
        if len(indices) != 4:
            raise CalibrationError("lateral_pairs requires four anchors")
        across = (
            points[:, indices[0]]
            - points[:, indices[1]]
            + points[:, indices[2]]
            - points[:, indices[3]]
        )
        forward = np.cross(
            np.broadcast_to(np.asarray([0.0, 1.0, 0.0]), across.shape), across
        )
    elif method == "root_to_head":
        if len(indices) != 2:
            raise CalibrationError("root_to_head requires two anchors")
        forward = points[:, indices[1]] - points[:, indices[0]]
    elif method == "declared_plus_z":
        if indices:
            raise CalibrationError("declared_plus_z does not accept anchors")
        forward = np.broadcast_to(
            np.asarray([0.0, 0.0, 1.0]), (points.shape[0], 3)
        ).copy()
    else:
        raise CalibrationError(f"unsupported position-anchor method {method!r}")
    horizontal = np.hypot(forward[:, 0], forward[:, 2])
    position_valid = horizontal >= POSITION_ANCHOR_EPS_NORM * scale
    position_heading = np.zeros((points.shape[0], 2), dtype=np.float64)
    position_heading[position_valid, 0] = (
        forward[position_valid, 2] / horizontal[position_valid]
    )
    position_heading[position_valid, 1] = (
        forward[position_valid, 0] / horizontal[position_valid]
    )
    return position_heading, position_valid, horizontal / scale


def position_anchor_heading_errors(
    positions: np.ndarray,
    heading: np.ndarray,
    heading_valid: np.ndarray,
    *,
    method: str,
    anchor_indices: Sequence[int],
    s_rig: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compare rotation heading with a separate heading derived from positions."""
    points = np.asarray(positions, dtype=np.float64)
    stored = np.asarray(heading, dtype=np.float64)
    stored_valid = np.asarray(heading_valid, dtype=bool)
    if (
        stored.shape != (points.shape[0], 2)
        or stored_valid.shape != (points.shape[0],)
        or not np.isfinite(stored).all()
    ):
        raise CalibrationError("invalid stored heading inputs")
    position_heading, position_valid, horizontal_norm = (
        derive_position_anchor_heading(
            points,
            method=method,
            anchor_indices=anchor_indices,
            s_rig=s_rig,
        )
    )
    compare = stored_valid & position_valid
    if not np.any(compare):
        raise CalibrationError("position-anchor heading has no comparable frame")
    cross = (
        stored[compare, 0] * position_heading[compare, 1]
        - stored[compare, 1] * position_heading[compare, 0]
    )
    dot = np.sum(stored[compare] * position_heading[compare], axis=-1)
    errors = np.abs(np.arctan2(cross, dot))
    return errors, compare, horizontal_norm


def _grouped_metric_distributions(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        group_keys = (
            f"source_family:{record['source_family']}",
            f"topology_distance_bucket:{record['topology_distance_bucket']}",
            f"family_role:{record['family_role']}",
            f"rig:{record['rig_id']}",
        )
        for group in group_keys:
            for name, value in record["metrics"].items():
                if isinstance(value, (int, float)) and value is not None:
                    grouped[group][name].append(float(value))
    return {
        group: {
            name: summarize_distribution(values)
            for name, values in sorted(metrics.items())
        }
        for group, metrics in sorted(grouped.items())
    }


def _sample_group_distributions(
    records: Sequence[Mapping[str, Any]],
    sample_blocks: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record, samples in zip(records, sample_blocks, strict=True):
        group_keys = (
            f"source_family:{record['source_family']}",
            f"topology_distance_bucket:{record['topology_distance_bucket']}",
            f"family_role:{record['family_role']}",
            f"rig:{record['rig_id']}",
        )
        for group in group_keys:
            for name, values in samples.items():
                grouped[group][name].append(np.asarray(values).reshape(-1))
    return {
        group: {
            name: summarize_distribution(np.concatenate(chunks))
            for name, chunks in sorted(metrics.items())
        }
        for group, metrics in sorted(grouped.items())
    }


def _candidate_error_thresholds(
    records: Sequence[Mapping[str, Any]],
    fixed_qa: Mapping[str, Any],
) -> dict[str, Any]:
    floors = {
        "direct_vs_fk_max_norm": 1e-5,
        "direct_vs_fk_p99_norm": 1e-5,
        "source_position_roundtrip_max_norm": 1e-5,
        "source_global_rotation_geodesic_max_rad": 2e-6,
        "source_velocity_max_norm_fps": 1e-5,
        "velocity_max_norm_fps": 1e-5,
        "rigid_edge_max_norm": 1e-4,
        "smooth_root_max_norm": 1e-5,
        "q_source_roundtrip_max_norm": 1e-5,
    }
    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_source[str(record["source_family"])].append(record)
    result: dict[str, Any] = {}
    source_fk_floors = fixed_qa["fixed_thresholds"]["source_fk_max_norm"]
    for source_family, source_records in sorted(by_source.items()):
        source_result: dict[str, Any] = {}
        source_metrics = [record["metrics"] for record in source_records]
        for name, floor in floors.items():
            values = np.asarray([float(metrics[name]) for metrics in source_metrics])
            q99_9 = float(np.percentile(values, 99.9))
            source_result[name] = {
                "engineering_floor": float(floor),
                "train_q99_9": q99_9,
                "multiplier": 1.5,
                "candidate_max": max(float(floor), 1.5 * q99_9),
            }
        source_fk_values = np.asarray(
            [float(metrics["source_parser_fk_max_norm"]) for metrics in source_metrics]
        )
        source_fk_q99_9 = float(np.percentile(source_fk_values, 99.9))
        source_fk_floor = float(source_fk_floors[source_family])
        source_result["source_parser_fk_max_norm"] = {
            "engineering_floor": source_fk_floor,
            "train_q99_9": source_fk_q99_9,
            "multiplier": 1.5,
            "candidate_max": max(source_fk_floor, 1.5 * source_fk_q99_9),
        }
        result[source_family] = source_result
    return result


def _file_manifest(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relpath = path.relative_to(root).as_posix()
        result[relpath] = {
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return result


def _replace_symlink(link: Path, target: Path) -> None:
    if link.exists() and not link.is_symlink():
        raise CalibrationError(f"refusing to replace non-symlink {link}")
    relative = os.path.relpath(target, start=link.parent)
    temporary = link.parent / f".{link.name}.{uuid.uuid4().hex}.tmp"
    os.symlink(relative, temporary)
    os.replace(temporary, link)


def run_prototype_calibration(
    *,
    prototype_root: str | Path,
    fixed_qa_report: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Publish numeric calibration candidates derived only from train records."""
    root = Path(prototype_root).expanduser().resolve()
    report_path = Path(fixed_qa_report).expanduser().resolve()
    output = Path(output_root).expanduser().absolute()
    generation = _load_json(root / "generation.json")
    encoder_config = _load_json(root / "config/encoder_candidate.json")
    selection = _load_json(root / "manifests/prototype_selection.json")
    fixed_qa = _load_json(report_path)
    if fixed_qa.get("status") != "pass" or fixed_qa.get("fail_count") != 0:
        raise CalibrationError("fixed QA must pass before calibration")
    if fixed_qa.get("generation_id") != generation.get("generation_id"):
        raise CalibrationError("fixed-QA generation does not match prototype")
    if Path(str(fixed_qa.get("prototype_root"))).resolve() != root:
        raise CalibrationError("fixed-QA prototype root does not match live generation")
    manifests = _load_jsonl(root / "manifests/clips.jsonl")
    manifest_ids = [str(record.get("clip_id")) for record in manifests]
    if len(manifest_ids) != len(set(manifest_ids)):
        raise CalibrationError("duplicate prototype manifest clip ids")
    train_records = [
        record
        for record in manifests
        if record.get("status") == "accept"
        and record.get("calibration_eligible") is True
    ]
    if any(record.get("split") != "train" for record in train_records):
        raise CalibrationError("held/validation/test record entered calibration")
    if any(
        record.get("calibration_eligible") is True
        and record.get("split") != "train"
        for record in manifests
    ):
        raise CalibrationError("non-train record is marked calibration eligible")
    if len(train_records) != int(fixed_qa["calibration_eligible_pass_count"]):
        raise CalibrationError("calibration scope differs from fixed-QA scope")
    if len(train_records) != 148:
        raise CalibrationError(f"expected pinned train scope 148, got {len(train_records)}")
    fixed_by_clip = {record["clip_id"]: record for record in fixed_qa["clips"]}
    if set(record["clip_id"] for record in train_records) - set(fixed_by_clip):
        raise CalibrationError("fixed QA omits a calibration clip")

    train_records.sort(key=lambda item: item["clip_id"])
    sum_squares = {"q": 0.0, "v": 0.0, "s": 0.0}
    counts = {"q": 0, "v": 0, "s": 0}
    calibrated_records: list[dict[str, Any]] = []
    sample_blocks: list[dict[str, np.ndarray]] = []
    q_chunks: list[np.ndarray] = []
    v_chunks: list[np.ndarray] = []
    smooth_chunks: list[np.ndarray] = []
    height_chunks: list[np.ndarray] = []
    speed_chunks: list[np.ndarray] = []
    contact_chunks: list[np.ndarray] = []
    joint_clip_index_chunks: list[np.ndarray] = []
    heading_error_chunks: list[np.ndarray] = []
    heading_clip_index_chunks: list[np.ndarray] = []
    heading_horizontal_chunks: list[np.ndarray] = []
    fps_src_values: list[float] = []

    for clip_index, manifest in enumerate(train_records):
        clip_id = str(manifest["clip_id"])
        motion_path = _resolve_generation_path(
            root, manifest["motion_relpath"], label=f"{clip_id}: motion"
        )
        skeleton_path = _resolve_generation_path(
            root, manifest["skeleton_relpath"], label=f"{clip_id}: skeleton"
        )
        if _sha256_file(motion_path) != manifest["motion_sha256"]:
            raise CalibrationError(f"{clip_id}: motion hash drifted")
        if _sha256_file(skeleton_path) != manifest["skeleton_sha256"]:
            raise CalibrationError(f"{clip_id}: skeleton hash drifted")
        skeleton = load_skeleton(skeleton_path)
        payload = load_motion_npz(
            motion_path, expected_fps_target=float(encoder_config["fps_target"])
        )
        motion = np.asarray(payload["motion"], dtype=np.float64)
        heading_valid = np.asarray(payload["heading_valid"], dtype=bool)
        decoded = decode_ktjd17(
            motion,
            parents=skeleton.parents,
            R_rest_global=skeleton.R_rest_global,
            R_rest_local=skeleton.R_rest_local,
            offset_parent_local=skeleton.offset_parent_local,
            rotation_source_kind=skeleton.rotation_source_kind,
            strict_gt=True,
        )
        scale = float(skeleton.s_rig)
        q_norm = motion[..., 0:3] / scale
        v_norm = motion[..., 9:12] / scale
        smooth_norm = motion[:, 0, 13:15] / scale
        for key, values in (("q", q_norm), ("v", v_norm), ("s", smooth_norm)):
            sum_squares[key] += float(np.sum(np.square(values), dtype=np.float64))
            counts[key] += int(values.size)
        height_norm = decoded.positions_direct[..., 1] / scale
        speed_norm = np.linalg.norm(motion[..., 9:12], axis=-1) / scale
        contact = motion[..., 12].astype(np.uint8)
        heading_provenance = skeleton.metadata.get("heading_payload_provenance")
        if not isinstance(heading_provenance, Mapping):
            raise CalibrationError(f"{clip_id}: heading payload provenance is absent")
        anchor_names = tuple(
            str(value) for value in heading_provenance["forward_anchor_names"]
        )
        anchor_indices = tuple(
            int(value) for value in heading_provenance["forward_anchor_indices"]
        )
        resolved_names = tuple(skeleton.joint_names[index] for index in anchor_indices)
        if resolved_names != anchor_names:
            raise CalibrationError(
                f"{clip_id}: heading anchor name/index mapping drifted: "
                f"{resolved_names} != {anchor_names}"
            )
        heading_errors, heading_compare, position_horizontal_norm = (
            position_anchor_heading_errors(
                decoded.positions_direct,
                motion[:, 0, 15:17],
                heading_valid,
                method=str(heading_provenance["forward_method"]),
                anchor_indices=anchor_indices,
                s_rig=scale,
            )
        )
        carrier_forward = np.einsum(
            "tij,j->ti",
            decoded.global_rotations[:, skeleton.heading_carrier_joint],
            skeleton.u_forward_local,
        )
        carrier_horizontal = np.hypot(carrier_forward[:, 0], carrier_forward[:, 2])
        fixed_metrics = dict(fixed_by_clip[clip_id]["metrics"])
        fixed_metrics.update(
            {
                "position_anchor_heading_circular_median_rad": float(
                    np.median(heading_errors)
                ),
                "position_anchor_heading_circular_p99_rad": float(
                    np.percentile(heading_errors, 99)
                ),
                "position_anchor_heading_circular_max_rad": float(
                    np.max(heading_errors)
                ),
                "position_anchor_comparable_fraction": float(
                    np.mean(heading_compare)
                ),
                "position_anchor_horizontal_min_norm": float(
                    np.min(position_horizontal_norm)
                ),
                "normalized_joint_height_median": float(np.median(height_norm)),
                "normalized_joint_speed_median": float(np.median(speed_norm)),
                "q_normalized_rms": float(np.sqrt(np.mean(np.square(q_norm)))),
                "v_normalized_rms": float(np.sqrt(np.mean(np.square(v_norm)))),
                "smooth_root_normalized_rms": float(
                    np.sqrt(np.mean(np.square(smooth_norm)))
                ),
            }
        )
        calibrated = {
            "clip_id": clip_id,
            "rig_id": str(manifest["rig_id"]),
            "source_family": str(manifest["source_family"]),
            "topology_family": str(manifest["topology_family"]),
            "topology_distance_bucket": str(
                manifest["topology_distance_bucket"]
            ),
            "family_role": str(manifest["family_role"]),
            "split": str(manifest["split"]),
            "calibration_eligible": True,
            "T": int(motion.shape[0]),
            "J": int(motion.shape[1]),
            "fps_src": float(manifest["fps_src"]),
            "fps_target": float(manifest["fps_target"]),
            "resample_mode": str(manifest["resample_mode"]),
            "heading_anchor_method": str(heading_provenance["forward_method"]),
            "heading_anchor_names": list(anchor_names),
            "metrics": fixed_metrics,
        }
        calibrated_records.append(calibrated)
        samples = {
            "normalized_joint_height": height_norm,
            "normalized_joint_speed": speed_norm,
            "contact": contact,
            "position_anchor_heading_circular_error_rad": heading_errors,
            "carrier_heading_horizontal_norm": carrier_horizontal,
        }
        sample_blocks.append(samples)
        q_chunks.append(q_norm.astype(np.float32).reshape(-1))
        v_chunks.append(v_norm.astype(np.float32).reshape(-1))
        smooth_chunks.append(smooth_norm.astype(np.float32).reshape(-1))
        height_chunks.append(height_norm.astype(np.float32).reshape(-1))
        speed_chunks.append(speed_norm.astype(np.float32).reshape(-1))
        contact_chunks.append(contact.reshape(-1))
        joint_clip_index_chunks.append(
            np.full(height_norm.size, clip_index, dtype=np.int32)
        )
        heading_error_chunks.append(heading_errors.astype(np.float32))
        heading_clip_index_chunks.append(
            np.full(heading_errors.size, clip_index, dtype=np.int32)
        )
        heading_horizontal_chunks.append(carrier_horizontal.astype(np.float32))
        fps_src_values.append(float(manifest["fps_src"]))

    gains, rms_values = gains_from_sums(sum_squares, counts)
    grouped_clip = _grouped_metric_distributions(calibrated_records)
    grouped_samples = _sample_group_distributions(calibrated_records, sample_blocks)
    all_heading_errors = np.concatenate(heading_error_chunks).astype(np.float64)
    all_carrier_horizontal = np.concatenate(heading_horizontal_chunks).astype(np.float64)
    heading_sweep = {
        f"eps_{epsilon:g}": {
            "epsilon": float(epsilon),
            "invalid_fraction": float(np.mean(all_carrier_horizontal < epsilon)),
            "valid_fraction": float(np.mean(all_carrier_horizontal >= epsilon)),
        }
        for epsilon in HEADING_EPS_SWEEP
    }
    jitter_values = np.asarray(
        [
            record["metrics"]["resample_acceleration_jitter_ratio"]
            for record in calibrated_records
            if record["metrics"]["resample_acceleration_jitter_ratio"] is not None
        ],
        dtype=np.float64,
    )
    jitter_deviation = np.abs(jitter_values - 1.0)
    jitter_candidate_abs_max = max(
        0.10, 1.5 * float(np.percentile(jitter_deviation, 99.9))
    )
    candidate_thresholds = _candidate_error_thresholds(
        calibrated_records, fixed_qa
    )
    family_counts: dict[str, int] = defaultdict(int)
    for record in calibrated_records:
        family_counts[str(record["family_role"])] += 1
    coverage_gaps = dict(
        _load_json(root / "qa/encoder_summary.json")["calibration_coverage_gaps"]
    )
    review_items: list[dict[str, Any]] = []
    for family, shortage in sorted(coverage_gaps.items()):
        review_items.append(
            {
                "code": "KTJD17_CALIBRATION_COVERAGE_SHORTAGE",
                "family_role": family,
                "shortage": int(shortage),
                "held_data_used_to_fill": False,
            }
        )
    heading_distribution = summarize_distribution(all_heading_errors)
    if float(heading_distribution["q99"]) > POSITION_ANCHOR_REVIEW_RAD:
        review_items.append(
            {
                "code": "KTJD17_POSITION_ANCHOR_HEADING_TAIL_REVIEW",
                "review_trigger_rad": POSITION_ANCHOR_REVIEW_RAD,
                "train_q99_rad": float(heading_distribution["q99"]),
                "action": "inspect synchronized perspective visual QA; do not tune on held",
            }
        )
    review_items.extend(
        [
            {
                "code": "KTJD17_CONTACT_VISUAL_REVIEW_REQUIRED",
                "source_contact_labels_available": False,
                "candidate_tau_h": float(encoder_config["contact"]["tau_h"]),
                "candidate_tau_v": float(encoder_config["contact"]["tau_v"]),
            },
            {
                "code": "KTJD17_VISUAL_QA_PENDING",
                "required_families": [
                    "human",
                    "quadruped",
                    "winged",
                    "snake",
                    "spider_crab",
                    "dragon_exact",
                ],
            },
        ]
    )
    candidate_freeze = {
        "status": "candidate_unfrozen",
        "fps_target": float(encoder_config["fps_target"]),
        "fps_audit": {
            "native_fps_distribution": summarize_distribution(fps_src_values),
            "resample_acceleration_jitter_ratio": summarize_distribution(
                jitter_values
            ),
            "candidate_abs_deviation_max": jitter_candidate_abs_max,
        },
        "smoother": dict(encoder_config["smoother"]),
        "contact": dict(encoder_config["contact"]),
        "heading": {
            **dict(encoder_config["heading"]),
            "epsilon_sweep": heading_sweep,
            "position_anchor_circular_error_rad": heading_distribution,
        },
        "normalization": {
            "definition": "reciprocal RMS of train-only valid q/s_rig, v/s_rig, and root smooth_xz/s_rig entries",
            "gains_order": ["g_q", "g_v", "g_s"],
            "gains": gains.tolist(),
            "source_rms": rms_values.tolist(),
            "valid_scalar_counts": [counts["q"], counts["v"], counts["s"]],
        },
        "topology": {"J_max": int(fixed_qa["J_phys_max"])},
        "candidate_error_thresholds_by_source_family": candidate_thresholds,
        "review_items": review_items,
        "freeze_authorized": False,
        "full_conversion_authorized": False,
    }
    calibration_report = {
        "calibration_version": CALIBRATION_VERSION,
        "status": "numeric_pass_pending_visual_and_coverage_review",
        "prototype_generation_id": generation["generation_id"],
        "prototype_selection_sha256": generation["selection_sha256"],
        "fixed_qa_version": fixed_qa["qa_version"],
        "fixed_qa_sha256": _sha256_file(report_path),
        "source_plan_commit": generation["source_plan_commit"],
        "scope": {
            "split": "train",
            "calibration_eligible_only": True,
            "clip_count": len(calibrated_records),
            "held_clip_count": 0,
            "validation_or_held_tuning_used": False,
            "family_counts": dict(sorted(family_counts.items())),
            "coverage_gaps": coverage_gaps,
            "selection_sha256": selection["selection_sha256"],
        },
        "normalization": candidate_freeze["normalization"],
        "candidate_freeze": candidate_freeze,
        "metric_distributions_by_group": grouped_clip,
        "sample_distributions_by_group": grouped_samples,
        "global_sample_distributions": {
            "normalized_joint_height": summarize_distribution(
                np.concatenate(height_chunks)
            ),
            "normalized_joint_speed": summarize_distribution(
                np.concatenate(speed_chunks)
            ),
            "contact": summarize_distribution(np.concatenate(contact_chunks)),
            "position_anchor_heading_circular_error_rad": heading_distribution,
            "carrier_heading_horizontal_norm": summarize_distribution(
                all_carrier_horizontal
            ),
        },
        "raw_distribution_artifact": "train_distribution_samples.npz",
        "per_clip_artifact": "train_clip_metrics.jsonl",
        "visual_qa_status": "pending",
        "freeze_authorized": False,
        "full_conversion_authorized": False,
        "review_items": review_items,
    }

    generation_id = (
        _datetime.datetime.now(_datetime.UTC).strftime("%Y%m%dT%H%M%S%fZ")
        + "-"
        + uuid.uuid4().hex[:12]
    )
    generations = output / CALIBRATION_GENERATION_DIRECTORY
    generations.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{generation_id}.", dir=generations))
    final = generations / generation_id
    try:
        _write_json(staging / "calibration_report.json", calibration_report)
        _write_json(staging / "candidate_freeze.json", candidate_freeze)
        _write_jsonl(staging / "train_clip_metrics.jsonl", calibrated_records)
        np.savez_compressed(
            staging / "candidate_train_block_gains.npz",
            gains=gains,
            g_q=np.asarray(gains[0], dtype=np.float64),
            g_v=np.asarray(gains[1], dtype=np.float64),
            g_s=np.asarray(gains[2], dtype=np.float64),
            source_rms=rms_values,
            valid_scalar_counts=np.asarray(
                [counts["q"], counts["v"], counts["s"]], dtype=np.int64
            ),
            clip_ids=np.asarray(
                [record["clip_id"] for record in calibrated_records]
            ),
            prototype_generation_id=np.asarray(generation["generation_id"]),
            split=np.asarray("train"),
            calibration_version=np.asarray(CALIBRATION_VERSION),
            frozen=np.asarray(False, dtype=np.bool_),
        )
        np.savez_compressed(
            staging / "train_distribution_samples.npz",
            q_over_s_rig=np.concatenate(q_chunks),
            v_over_s_rig=np.concatenate(v_chunks),
            smooth_root_xz_over_s_rig=np.concatenate(smooth_chunks),
            normalized_joint_height=np.concatenate(height_chunks),
            normalized_joint_speed=np.concatenate(speed_chunks),
            contact=np.concatenate(contact_chunks),
            joint_sample_clip_index=np.concatenate(joint_clip_index_chunks),
            position_anchor_heading_circular_error_rad=np.concatenate(
                heading_error_chunks
            ),
            heading_sample_clip_index=np.concatenate(heading_clip_index_chunks),
            carrier_heading_horizontal_norm=np.concatenate(
                heading_horizontal_chunks
            ),
            clip_ids=np.asarray(
                [record["clip_id"] for record in calibrated_records]
            ),
        )
        files = _file_manifest(staging)
        calibration_generation = {
            "calibration_version": CALIBRATION_VERSION,
            "generation_id": generation_id,
            "created_at_utc": _datetime.datetime.now(_datetime.UTC).isoformat(),
            "prototype_generation_id": generation["generation_id"],
            "prototype_root": str(root),
            "prototype_generation_sha256": _sha256_file(root / "generation.json"),
            "fixed_qa_report": str(report_path),
            "fixed_qa_sha256": _sha256_file(report_path),
            "scope": "train_only",
            "status": calibration_report["status"],
            "files": files,
            "visual_qa_status": "pending",
            "freeze_authorized": False,
            "full_conversion_authorized": False,
        }
        _write_json(staging / "generation.json", calibration_generation)
        if final.exists():
            raise CalibrationError(f"calibration generation already exists: {final}")
        os.replace(staging, final)
        parent_fd = os.open(generations, os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        _replace_symlink(output / CALIBRATION_LINK_NAME, final)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {
        "status": calibration_report["status"],
        "generation_id": generation_id,
        "generation_root": str(final),
        "compatibility_link": str(output / CALIBRATION_LINK_NAME),
        "train_clip_count": len(calibrated_records),
        "held_clip_count": 0,
        "family_counts": dict(sorted(family_counts.items())),
        "coverage_gaps": coverage_gaps,
        "gains": gains.tolist(),
        "position_anchor_heading_q99_rad": float(heading_distribution["q99"]),
        "visual_qa_status": "pending",
        "freeze_authorized": False,
        "full_conversion_authorized": False,
    }
