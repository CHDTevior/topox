"""Full-catalog Truebones direction prototype for the frozen KTJD-17 build.

This is intentionally a pre-conversion gate.  It parses every source-safe
current BTJD Truebones clip from original BVH rotation channels, selects one
motion-rich representative per encodable rig, constructs a fixed physical rig
from the current ``cond.npy`` geometry, and publishes only the representative
motions for synchronized source/direct/FK visual review.

No legacy BTJD-13 motion channel is read by this module.  The legacy dataset is
represented only by the pinned inventory and fixed conditioning geometry.
"""

from __future__ import annotations

import dataclasses
import datetime as _datetime
import hashlib
import json
import math
import os
import shutil
import tempfile
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .codec import SmootherConfig
from .encoder import (
    EncoderConfig,
    encode_prepared_motion,
    load_skeleton,
    prepare_manifest_clip,
    write_npz_atomic,
)
from .schema import load_schema
from .source_parser import ParsedBvhMotion, ParsedSourceMotion
from .truebones_fixed_rig import (
    ACTIVE_COND_SHA256,
    FULL_TRUEBONES_FORWARD_SPEC_VERSION,
    LEGACY_TRUEBONES_COND_SHA256,
    TRUEBONES_FULL_FORWARD_SPECS,
    build_fixed_rig_motion,
    load_conditioning_catalog,
    validate_full_forward_spec_catalog,
)


TRUEBONES_FORWARD_AUDIT_VERSION = "ktjd17-truebones-forward-audit-v1"
AUDIT_GENERATION_DIRECTORY = ".ktjd17_truebones_forward_audit_generations"
AUDIT_LINK_NAME = "ktjd17_truebones_forward_audit"
SOURCE_PLAN_COMMIT = "9181f5cccbad23e941bf94c2874daf36e7f288cf"
PARENT_MANIFEST_GENERATION_ID = "20260819T145535975831Z-ed48b3fd2745"
FROZEN_SCHEMA_GENERATION_ID = "20260819T192429040697Z-fe820492caaa"
EXPECTED_CLIPS_SHA256 = (
    "f7cd0d05ad2208924c43ede43f31a13d6ec893a2af92ec3229f344e76b23e9f3"
)
EXPECTED_RIGS_SHA256 = (
    "108cb904684617e2bdcf24eaa80f9e5976d14a38acf56054b3996237e5cc5271"
)
EXPECTED_FROZEN_SCHEMA_SHA256 = (
    "9132e7b11573062569361d4959e561f84dafac1d3ea25e737506be3bf6da7edf"
)
EXPECTED_SCOPE = {
    "clip_count": 1070,
    "rig_count": 70,
    "source_safe_clip_count": 986,
    "upstream_reject_count": 84,
    "source_layout_drift_count": 67,
    "sequence_split_overlap_count": 17,
    "encodable_rig_count": 66,
}
EXPECTED_UNAVAILABLE_RIGS = {"Ant", "Crab", "Deer", "Jaguar"}
FALLBACK_REST_METHODS = {"legacy_idle_fallback", "legacy_first_file_fallback"}
FORWARD_TO_PLUS_Z_TOL = 1e-6
J_MAX = 142


class TruebonesForwardAuditError(RuntimeError):
    """The full-rig direction prototype cannot be trusted or published."""


@dataclasses.dataclass(frozen=True)
class ForwardAuditConfig:
    manifest_root: Path
    freeze_root: Path
    output_root: Path
    active_cond_path: Path
    legacy_cond_path: Path
    overwrite_link: bool = True

    def resolved(self) -> "ForwardAuditConfig":
        return dataclasses.replace(
            self,
            manifest_root=self.manifest_root.expanduser().resolve(),
            freeze_root=self.freeze_root.expanduser().resolve(),
            output_root=self.output_root.expanduser().absolute(),
            active_cond_path=self.active_cond_path.expanduser().resolve(),
            legacy_cond_path=self.legacy_cond_path.expanduser().resolve(),
        )


def default_forward_audit_config(repo_root: str | Path = ".") -> ForwardAuditConfig:
    root = Path(repo_root).expanduser().resolve()
    return ForwardAuditConfig(
        manifest_root=root
        / "dataset/.ktjd17_manifest_generations"
        / PARENT_MANIFEST_GENERATION_ID,
        freeze_root=root / "dataset/ktjd17_freeze",
        output_root=root / "dataset",
        active_cond_path=root
        / "data/animo4d_L4TB_plus_human_v4b272neutral/cond.npy",
        legacy_cond_path=root / "data/anytop_truebones/cond.npy",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _load_pinned_jsonl(path: Path, *, expected_sha256: str) -> list[dict[str, Any]]:
    """Parse the same byte stream whose digest is checked against authority."""
    records: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                digest.update(raw_line)
                if not raw_line.strip():
                    raise TruebonesForwardAuditError(
                        f"{path}:{line_number}: blank JSONL row"
                    )
                try:
                    value = json.loads(raw_line.decode("utf-8"))
                except Exception as exc:  # noqa: BLE001
                    raise TruebonesForwardAuditError(
                        f"{path}:{line_number}: invalid UTF-8 JSON: {exc}"
                    ) from exc
                if not isinstance(value, dict):
                    raise TruebonesForwardAuditError(
                        f"{path}:{line_number}: row is not an object"
                    )
                records.append(value)
    except TruebonesForwardAuditError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise TruebonesForwardAuditError(f"cannot read {path}: {exc}") from exc
    observed = digest.hexdigest()
    if observed != expected_sha256:
        raise TruebonesForwardAuditError(
            f"loaded byte stream hash drifted for {path.name}: "
            f"{observed} != {expected_sha256}"
        )
    return records


def verify_parent_manifest_files(
    manifest_root: str | Path,
    *,
    expected_clips_sha256: str = EXPECTED_CLIPS_SHA256,
    expected_rigs_sha256: str = EXPECTED_RIGS_SHA256,
) -> dict[str, str]:
    """Hash-check the two regular parent files used by the audit producer."""
    root = Path(manifest_root).expanduser().resolve()
    verified: dict[str, str] = {}
    for filename, expected in (
        ("clips.jsonl", expected_clips_sha256),
        ("rigs.jsonl", expected_rigs_sha256),
    ):
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise TruebonesForwardAuditError(
                f"invalid expected parent hash for {filename}"
            )
        path = root / filename
        if path.is_symlink() or not path.is_file():
            raise TruebonesForwardAuditError(
                f"parent manifest input is not a regular file: {path}"
            )
        observed = _sha256_file(path)
        if observed != expected:
            raise TruebonesForwardAuditError(
                f"parent {filename} hash drifted: {observed} != {expected}"
            )
        verified[filename] = observed
    return verified


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ) + "\n"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(_canonical_json(record).decode("utf-8") + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _text_scalar(value: str) -> np.ndarray:
    text = str(value)
    return np.asarray(text, dtype=f"<U{max(1, len(text))}")


def _text_vector(values: Sequence[str]) -> np.ndarray:
    strings = [str(value) for value in values]
    width = max([1, *(len(value) for value in strings)])
    return np.asarray(strings, dtype=f"<U{width}")


def _json_scalar(value: Any) -> np.ndarray:
    return _text_scalar(_canonical_json(value).decode("utf-8"))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _replace_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() and not link.is_symlink():
        raise TruebonesForwardAuditError(f"refusing to replace non-symlink {link}")
    temporary = link.parent / f".{link.name}.{uuid.uuid4().hex}.tmp"
    os.symlink(os.path.relpath(target, start=link.parent), temporary)
    os.replace(temporary, link)
    _fsync_directory(link.parent)


def _file_manifest(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise TruebonesForwardAuditError(
                f"symlink is forbidden inside audit generation: {path}"
            )
        if path.is_file():
            relpath = path.relative_to(root).as_posix()
            result[relpath] = {
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    return result


def verify_forward_audit_generation(root: str | Path) -> dict[str, Any]:
    generation_root = Path(root).expanduser().resolve()
    generation_path = generation_root / "generation.json"
    try:
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise TruebonesForwardAuditError(
            f"cannot read audit generation manifest: {exc}"
        ) from exc
    expected = generation.get("files")
    if not isinstance(expected, dict):
        raise TruebonesForwardAuditError("audit generation files map is absent")
    observed = _file_manifest(generation_root)
    observed.pop("generation.json", None)
    if set(observed) != set(expected):
        raise TruebonesForwardAuditError(
            "audit generation file closure failed: "
            f"missing={sorted(set(expected) - set(observed))}, "
            f"extra={sorted(set(observed) - set(expected))}"
        )
    for relpath, record in expected.items():
        if observed[relpath] != record:
            raise TruebonesForwardAuditError(
                f"audit generation hash/size drift: {relpath}"
            )
    return generation


def encoder_config_from_frozen_schema(freeze_root: str | Path) -> EncoderConfig:
    root = Path(freeze_root).expanduser().resolve()
    if root.name != FROZEN_SCHEMA_GENERATION_ID:
        raise TruebonesForwardAuditError(
            f"freeze link resolves to unexpected generation {root.name}"
        )
    schema_path = root / "schema.json"
    if _sha256_file(schema_path) != EXPECTED_FROZEN_SCHEMA_SHA256:
        raise TruebonesForwardAuditError("frozen schema hash drifted")
    schema = load_schema(schema_path, expected_fps_target=30.0, require_frozen=True)
    if _sha256_file(schema_path) != EXPECTED_FROZEN_SCHEMA_SHA256:
        raise TruebonesForwardAuditError("frozen schema changed while loading")
    smoother = schema["smoother"]
    params = smoother["params"]
    config = EncoderConfig(
        fps_target=float(schema["fps_target"]),
        smoother=SmootherConfig(
            method=str(smoother["id"]),
            order=int(params["order"]),
            cutoff_hz=float(params["cutoff_hz"]),
            padtype=str(params["padtype"]),
            padlen=int(params["padlen"]),
            short_clip_cycles=float(params["short_clip_cycles"]),
        ),
        contact_tau_h=float(schema["contact"]["tau_h"]),
        contact_tau_v=float(schema["contact"]["tau_v"]),
        heading_eps_h=float(schema["heading"]["eps_h"]),
        calibration_status="frozen",
    )
    config.validate()
    if int(schema["topology"]["J_max"]) != J_MAX:
        raise TruebonesForwardAuditError("frozen J_max drifted")
    return config


def _motion_energy(parsed: ParsedSourceMotion) -> dict[str, float]:
    """Source-only motion score used solely to choose a revealing visual clip."""
    rotations = np.asarray(parsed.global_rotations, dtype=np.float64)
    roots = np.asarray(parsed.root_translation, dtype=np.float64)
    if len(rotations) < 2 or rotations.shape[:2] != roots.shape[:1] + (parsed.joint_count,):
        raise TruebonesForwardAuditError(
            f"{parsed.path}: invalid motion-energy shapes"
        )
    relative = np.matmul(
        np.swapaxes(rotations[:-1], -1, -2), rotations[1:]
    )
    cosine = np.clip(
        (np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5,
        -1.0,
        1.0,
    )
    angular_speed = np.arccos(cosine) * float(parsed.fps)
    root_speed = (
        np.linalg.norm(np.diff(roots, axis=0), axis=-1)
        * float(parsed.fps)
        / float(parsed.s_rig)
    )
    angular_p95 = float(np.percentile(angular_speed, 95))
    root_p95 = float(np.percentile(root_speed, 95))
    score = angular_p95 + min(root_p95, 10.0)
    if not all(math.isfinite(value) for value in (angular_p95, root_p95, score)):
        raise TruebonesForwardAuditError(f"{parsed.path}: non-finite motion energy")
    return {
        "selection_score": score,
        "global_angular_speed_p95_rad_s": angular_p95,
        "root_speed_p95_s_rig_s": root_p95,
    }


def _parse_truebones_record(
    clip: Mapping[str, Any],
    rig: Mapping[str, Any],
    *,
    rest_cache: dict[str, ParsedBvhMotion],
) -> ParsedSourceMotion:
    # Kept local instead of using the encoder's historical 31-rig lookup.
    from .encoder import _parse_truebones  # local import documents the boundary

    return _parse_truebones(clip, rig, rest_cache=rest_cache)


def _validate_parent_scope(
    clips: Sequence[Mapping[str, Any]], rigs: Mapping[str, Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    truebones = [
        dict(record)
        for record in clips
        if record.get("source", {}).get("family") == "truebones"
    ]
    truebones_rigs = {
        rig_id: record
        for rig_id, record in rigs.items()
        if record.get("source_family") == "truebones"
    }
    if len(truebones) != EXPECTED_SCOPE["clip_count"]:
        raise TruebonesForwardAuditError("current Truebones clip count drifted")
    if len(truebones_rigs) != EXPECTED_SCOPE["rig_count"]:
        raise TruebonesForwardAuditError("current Truebones rig count drifted")
    safe = [record for record in truebones if record.get("status") != "reject"]
    rejected = [record for record in truebones if record.get("status") == "reject"]
    if len(safe) != EXPECTED_SCOPE["source_safe_clip_count"]:
        raise TruebonesForwardAuditError("source-safe clip count drifted")
    if len(rejected) != EXPECTED_SCOPE["upstream_reject_count"]:
        raise TruebonesForwardAuditError("upstream reject count drifted")
    for record in safe:
        if record.get("status") not in {"accept", "review"}:
            raise TruebonesForwardAuditError(
                f"{record.get('clip_id')}: unexpected source-safe status"
            )
        if record.get("source", {}).get("sequence_split_safe") is not True:
            raise TruebonesForwardAuditError(
                f"{record.get('clip_id')}: source-safe clip crosses split boundary"
            )
    layout = 0
    overlap = 0
    for record in rejected:
        reasons = set(record.get("reason_codes", []))
        has_layout = "SOURCE_LAYOUT_DRIFT" in reasons
        has_overlap = "RAW_SOURCE_SEQUENCE_SPLIT_OVERLAP" in reasons
        if has_layout == has_overlap:
            raise TruebonesForwardAuditError(
                f"{record.get('clip_id')}: reject does not have exactly one source blocker"
            )
        layout += int(has_layout)
        overlap += int(has_overlap)
    if layout != EXPECTED_SCOPE["source_layout_drift_count"]:
        raise TruebonesForwardAuditError("source-layout reject count drifted")
    if overlap != EXPECTED_SCOPE["sequence_split_overlap_count"]:
        raise TruebonesForwardAuditError("split-overlap reject count drifted")
    return safe, rejected


def _select_representatives(
    safe: Sequence[dict[str, Any]],
    rigs: Mapping[str, dict[str, Any]],
    *,
    rest_cache: dict[str, ParsedBvhMotion],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    candidates: dict[str, list[tuple[dict[str, Any], dict[str, float]]]] = defaultdict(list)
    parse_qa: list[dict[str, Any]] = []
    for index, clip in enumerate(sorted(safe, key=lambda item: item["clip_id"]), start=1):
        rig_id = str(clip["rig_id"])
        try:
            parsed = _parse_truebones_record(
                clip, rigs[rig_id], rest_cache=rest_cache
            )
            energy = _motion_energy(parsed)
        except Exception as exc:  # noqa: BLE001
            raise TruebonesForwardAuditError(
                f"source-safe parse failed for {clip['clip_id']}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        record = {
            "clip_id": clip["clip_id"],
            "rig_id": rig_id,
            "split": clip["split"],
            "T_src": int(parsed.frame_count),
            "J_phys": int(parsed.joint_count),
            "fps_src": float(parsed.fps),
            "status": "pass",
            "selection_metrics": energy,
        }
        parse_qa.append(record)
        candidates[rig_id].append((clip, energy))
        if index % 100 == 0 or index == len(safe):
            print(
                f"[ktjd17-forward-audit] source parsed {index}/{len(safe)}",
                flush=True,
            )
    selected: dict[str, dict[str, Any]] = {}
    for rig_id, values in candidates.items():
        ranked = sorted(
            values,
            key=lambda item: (
                -float(item[1]["selection_score"]),
                -int(item[0]["source"]["T_src"]),
                str(item[0]["clip_id"]),
            ),
        )
        selected[rig_id] = ranked[0][0]
    unavailable = set(rigs) - set(selected)
    if unavailable != EXPECTED_UNAVAILABLE_RIGS:
        raise TruebonesForwardAuditError(
            f"unexpected unavailable rig set: {sorted(unavailable)}"
        )
    if len(selected) != EXPECTED_SCOPE["encodable_rig_count"]:
        raise TruebonesForwardAuditError("encodable rig count drifted")
    return selected, parse_qa


def _skeleton_payload(
    *,
    rig: Mapping[str, Any],
    clip: Mapping[str, Any],
    parsed: ParsedSourceMotion,
    fixed: Any,
    conditioning_authority: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    rig_id = str(rig["rig_id"])
    spec = TRUEBONES_FULL_FORWARD_SPECS[rig_id]
    rest_method = str(rig["rest_pose"]["selection_method"])
    fallback = rest_method in FALLBACK_REST_METHODS
    s_rig = float(np.linalg.norm(np.ptp(fixed.P_rest_global, axis=0)))
    if len(parsed.joint_names) > J_MAX:
        raise TruebonesForwardAuditError(
            f"{rig_id}: J={len(parsed.joint_names)} exceeds frozen J_max={J_MAX}"
        )
    if fixed.metrics["conditioning_forward_to_plus_z_max_abs"] > FORWARD_TO_PLUS_Z_TOL:
        raise TruebonesForwardAuditError(
            f"{rig_id}: conditioning forward is not +Z: "
            f"{fixed.metrics['conditioning_forward_to_plus_z_max_abs']}"
        )
    forward_local = fixed.R_rest_global[0].T @ np.asarray(
        [0.0, 0.0, 1.0], dtype=np.float64
    )
    reason_codes = ["REST_FRAME_FALLBACK_REQUIRES_VISUAL_QA"] if fallback else []
    heading_provenance = {
        "forward_spec_version": FULL_TRUEBONES_FORWARD_SPEC_VERSION,
        "forward_method": spec.method,
        "forward_anchor_names": list(spec.anchor_names),
        "forward_anchor_indices": list(fixed.forward_anchor_indices),
        "legacy_anchor_indices": (
            None
            if spec.legacy_anchor_indices is None
            else list(spec.legacy_anchor_indices)
        ),
        "forward_spec_provenance": spec.provenance,
        "carrier_joint": 0,
        "carrier_name": parsed.joint_names[0],
        "canonical_rest_forward": [0.0, 0.0, 1.0],
        "coordinate_contract": (
            "right-handed; +Y up/screen-up; +Z out of screen toward viewer"
        ),
        "visual_status": "pending_multiframe_perspective_review",
    }
    unit_metadata = {
        "length_unit_id": "truebones_btjd_canonical_mean_rest_edge",
        "source_unit_to_meter": None,
        "canonical_scale_factor": float(fixed.alpha),
        "s_rig": s_rig,
        "meter_claim": False,
    }
    joint_map_metadata = {
        "mapping_kind": rig["joint_map"]["mapping_kind"],
        "joint_map_sha256": rig["joint_map"]["joint_map_sha256"],
        "source_rotation_layout_sha256": rig["joint_map"][
            "source_rotation_layout_sha256"
        ],
        "rotation_authority": "original_bvh_declared_rotation_channels_only",
        "legacy_btjd_motion_channels_used": False,
    }
    return {
        "joint_names": _text_vector(parsed.joint_names),
        "parents": np.asarray(parsed.parents, dtype=np.int64),
        "P_rest_global": np.asarray(fixed.P_rest_global, dtype=np.float64),
        "R_rest_global": np.asarray(fixed.R_rest_global, dtype=np.float64),
        "R_rest_local": np.asarray(fixed.R_rest_local, dtype=np.float64),
        "offset_parent_local": np.asarray(
            fixed.offset_parent_local, dtype=np.float64
        ),
        "rotation_source_kind": _text_vector(parsed.rotation_source_kind),
        "heading_carrier_joint": np.asarray(0, dtype=np.int64),
        "u_forward_local": np.asarray(forward_local, dtype=np.float64),
        "source_to_canonical_C": np.asarray(fixed.C, dtype=np.float64),
        "source_to_canonical_alpha": np.asarray(fixed.alpha, dtype=np.float64),
        "source_to_canonical_o": np.asarray(fixed.o, dtype=np.float64),
        "s_rig": np.asarray(s_rig, dtype=np.float64),
        "rig_id": _text_scalar(rig_id),
        "source_family": _text_scalar("truebones"),
        "topology_family": _text_scalar(str(rig["topology_family"])),
        "artifact_status": _text_scalar("review"),
        "reason_codes": _text_vector(reason_codes),
        "skeleton_format_version": _text_scalar(
            "ktjd17-full-truebones-forward-candidate-v1"
        ),
        "representative_clip_id": _text_scalar(str(clip["clip_id"])),
        "source_rest_path": _text_scalar(str(parsed.rest_path)),
        "heading_payload_provenance": _json_scalar(heading_provenance),
        "source_to_canonical_provenance": _json_scalar(
            {
                **fixed.provenance,
                "status": "numeric_pass_visual_pending",
                "rest_selection_method": rest_method,
            }
        ),
        "position_geometry_provenance": _json_scalar(fixed.provenance),
        "conditioning_authority": _json_scalar(dict(conditioning_authority)),
        "unit_metadata": _json_scalar(unit_metadata),
        "joint_map_metadata": _json_scalar(joint_map_metadata),
        "fixed_rig_rotation_signatures": _json_scalar(
            fixed.rotation_signatures
        ),
    }


def _upstream_rejection_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "clip_id": record["clip_id"],
        "rig_id": record["rig_id"],
        "split": record["split"],
        "topology_family": record["topology_family"],
        "topology_distance_bucket": record["topology_distance_bucket"],
        "status": "reject",
        "reason_codes": list(record["reason_codes"]),
        "source_path": record["source"]["path"],
        "source_sequence_split_safe": bool(
            record["source"]["sequence_split_safe"]
        ),
        "conversion_attempted": False,
        "legacy_btjd_fallback_allowed": False,
    }


def run_truebones_forward_audit(config: ForwardAuditConfig) -> dict[str, Any]:
    cfg = config.resolved()
    clips_path = cfg.manifest_root / "clips.jsonl"
    rigs_path = cfg.manifest_root / "rigs.jsonl"
    parent_hashes_before = verify_parent_manifest_files(cfg.manifest_root)
    encoder = encoder_config_from_frozen_schema(cfg.freeze_root)
    clips = _load_pinned_jsonl(
        clips_path, expected_sha256=EXPECTED_CLIPS_SHA256
    )
    rig_records = _load_pinned_jsonl(
        rigs_path, expected_sha256=EXPECTED_RIGS_SHA256
    )
    parent_hashes_after = verify_parent_manifest_files(cfg.manifest_root)
    if parent_hashes_after != parent_hashes_before:
        raise TruebonesForwardAuditError(
            "parent manifest authority changed while loading"
        )
    rigs = {str(record["rig_id"]): record for record in rig_records}
    if len(rigs) != len(rig_records):
        raise TruebonesForwardAuditError("duplicate rig ids in parent manifest")
    truebones_rigs = {
        key: value
        for key, value in rigs.items()
        if value.get("source_family") == "truebones"
    }
    safe, rejected = _validate_parent_scope(clips, truebones_rigs)
    conditioning = load_conditioning_catalog(
        cfg.active_cond_path,
        expected_active_sha256=ACTIVE_COND_SHA256,
        legacy_path=cfg.legacy_cond_path,
    )
    if conditioning.legacy_sha256 != LEGACY_TRUEBONES_COND_SHA256:
        raise TruebonesForwardAuditError("legacy conditioning hash drifted")
    validate_full_forward_spec_catalog(
        {
            rig_id: conditioning.active_entries[rig_id]["joints_names"]
            for rig_id in truebones_rigs
        }
    )
    rest_cache: dict[str, ParsedBvhMotion] = {}
    selected, parse_qa = _select_representatives(
        safe, truebones_rigs, rest_cache=rest_cache
    )

    generation_id = (
        _datetime.datetime.now(_datetime.UTC).strftime("%Y%m%dT%H%M%S%fZ")
        + "-"
        + uuid.uuid4().hex[:12]
    )
    generations = cfg.output_root / AUDIT_GENERATION_DIRECTORY
    generations.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{generation_id}.", dir=generations))
    final = generations / generation_id
    manifest_records: list[dict[str, Any]] = []
    rig_qa_records: list[dict[str, Any]] = []
    try:
        for index, rig_id in enumerate(sorted(selected), start=1):
            clip = selected[rig_id]
            rig = truebones_rigs[rig_id]
            parsed = _parse_truebones_record(
                clip, rig, rest_cache=rest_cache
            )
            fixed_geometry = conditioning.rig(
                rig_id,
                expected_names=rig["joint_map"]["btjd_joint_names"],
                expected_parents=rig["joint_map"]["btjd_parents"],
            )
            fixed = build_fixed_rig_motion(
                parsed, fixed_geometry, TRUEBONES_FULL_FORWARD_SPECS[rig_id]
            )
            skeleton_relpath = f"skeletons/{rig_id}.npz"
            skeleton_path = staging / skeleton_relpath
            payload = _skeleton_payload(
                rig=rig,
                clip=clip,
                parsed=parsed,
                fixed=fixed,
                conditioning_authority={
                    **conditioning.authority_record(),
                    "rig_payload_sha256": fixed_geometry.payload_sha256,
                },
            )
            skeleton_sha = write_npz_atomic(skeleton_path, payload)
            skeleton = load_skeleton(skeleton_path)
            if skeleton.sha256 != skeleton_sha:
                raise TruebonesForwardAuditError(
                    f"{rig_id}: skeleton write/hash verification failed"
                )
            prepared = prepare_manifest_clip(
                clip,
                rig,
                skeleton,
                conditioning_catalog=conditioning,
                rest_cache=rest_cache,
                truebones_forward_specs=TRUEBONES_FULL_FORWARD_SPECS,
            )
            encoded = encode_prepared_motion(prepared, skeleton, encoder)
            motion_relpath = f"motions/{encoded.clip_id}.npz"
            motion_path = staging / motion_relpath
            motion_sha = write_npz_atomic(
                motion_path, encoded.artifact_payload()
            )
            parse_record = next(
                record
                for record in parse_qa
                if record["clip_id"] == encoded.clip_id
            )
            rest_method = str(rig["rest_pose"]["selection_method"])
            fallback = rest_method in FALLBACK_REST_METHODS
            manifest_records.append(
                {
                    "clip_id": encoded.clip_id,
                    "rig_id": rig_id,
                    "source_family": "truebones",
                    "topology_family": rig["topology_family"],
                    "topology_distance_bucket": clip[
                        "topology_distance_bucket"
                    ],
                    "family_role": rig_id,
                    "audit_role": "one_motion_rich_source_safe_clip_per_rig",
                    "calibration_eligible": False,
                    "selection_origin": "raw_bvh_motion_energy_ranking",
                    "split": clip["split"],
                    "status": "accept",
                    "reason_codes": (
                        ["REST_FRAME_FALLBACK_REQUIRES_VISUAL_QA"]
                        if fallback
                        else []
                    ),
                    "motion_relpath": motion_relpath,
                    "motion_sha256": motion_sha,
                    "skeleton_relpath": skeleton_relpath,
                    "skeleton_sha256": skeleton_sha,
                    "fps_src": encoded.fps_src,
                    "fps_target": encoded.fps_target,
                    "T_src": encoded.metrics["T_src"],
                    "T_target": encoded.metrics["T_target"],
                    "J_phys": encoded.metrics["J_phys"],
                    "resample_mode": encoded.resample_mode,
                    "source_path": clip["source"]["path"],
                    "source_split_protocol": clip["split_protocol"],
                    "visual_status": "pending_multiframe_perspective_review",
                }
            )
            rig_qa_records.append(
                {
                    "audit_version": TRUEBONES_FORWARD_AUDIT_VERSION,
                    "clip_id": encoded.clip_id,
                    "rig_id": rig_id,
                    "topology_family": rig["topology_family"],
                    "split": clip["split"],
                    "rest_selection_method": rest_method,
                    "rest_fallback": fallback,
                    "status": "numeric_pass_visual_pending",
                    "selection_metrics": parse_record["selection_metrics"],
                    "fixed_rig_metrics": fixed.metrics,
                    "encoder_metrics": encoded.metrics,
                    "forward_spec": dataclasses.asdict(
                        TRUEBONES_FULL_FORWARD_SPECS[rig_id]
                    ),
                    "rotation_authority": (
                        "original_bvh_declared_rotation_channels_only"
                    ),
                    "legacy_btjd_motion_channels_used": False,
                    "motion_sha256": motion_sha,
                    "skeleton_sha256": skeleton_sha,
                }
            )
            if index % 10 == 0 or index == len(selected):
                print(
                    f"[ktjd17-forward-audit] encoded {index}/{len(selected)} rigs",
                    flush=True,
                )

        manifest_records.sort(key=lambda record: record["clip_id"])
        rig_qa_records.sort(key=lambda record: record["rig_id"])
        parse_qa.sort(key=lambda record: record["clip_id"])
        upstream_rejections = sorted(
            (_upstream_rejection_record(record) for record in rejected),
            key=lambda record: record["clip_id"],
        )
        selected_records = [
            {
                "rig_id": record["rig_id"],
                "clip_id": record["clip_id"],
                "split": record["split"],
                "selection_metrics": next(
                    item["selection_metrics"]
                    for item in parse_qa
                    if item["clip_id"] == record["clip_id"]
                ),
            }
            for record in rig_qa_records
        ]
        selection_authority = {
            "policy_version": "ktjd17-one-motion-rich-raw-bvh-clip-per-rig-v1",
            "parent_manifest_root": str(cfg.manifest_root),
            "parent_clips_jsonl_sha256": EXPECTED_CLIPS_SHA256,
            "parent_rigs_jsonl_sha256": EXPECTED_RIGS_SHA256,
            "selection_input": "original_bvh_root_and_rotation_channels_only",
            "legacy_btjd_motion_channels_used": False,
        }
        selection_sha = hashlib.sha256(
            _canonical_json(
                {
                    "selection_authority": selection_authority,
                    "selected": selected_records,
                }
            )
        ).hexdigest()
        _write_jsonl(staging / "manifests/clips.jsonl", manifest_records)
        _write_jsonl(staging / "qa/rig_audit.jsonl", rig_qa_records)
        _write_jsonl(staging / "qa/source_safe_parse.jsonl", parse_qa)
        _write_jsonl(
            staging / "manifests/upstream_rejections.jsonl",
            upstream_rejections,
        )
        _write_json(
            staging / "manifests/prototype_selection.json",
            {
                "audit_version": TRUEBONES_FORWARD_AUDIT_VERSION,
                "selection_authority": selection_authority,
                "selection_sha256": selection_sha,
                "selected_count": len(selected_records),
                "selected": selected_records,
                "held_data_used_for_calibration": False,
            },
        )
        _write_json(
            staging / "config/encoder_candidate.json", encoder.as_record()
        )
        unavailable_records = []
        for rig_id in sorted(EXPECTED_UNAVAILABLE_RIGS):
            rig_rejections = [
                record for record in upstream_rejections if record["rig_id"] == rig_id
            ]
            unavailable_records.append(
                {
                    "rig_id": rig_id,
                    "status": "reject",
                    "source_safe_clip_count": 0,
                    "upstream_reject_count": len(rig_rejections),
                    "reason_codes": sorted(
                        {
                            reason
                            for record in rig_rejections
                            for reason in record["reason_codes"]
                        }
                    ),
                    "legacy_btjd_fallback_allowed": False,
                }
            )
        _write_jsonl(
            staging / "manifests/unavailable_rigs.jsonl", unavailable_records
        )
        fallback_rigs = sorted(
            record["rig_id"] for record in rig_qa_records if record["rest_fallback"]
        )
        summary = {
            "audit_version": TRUEBONES_FORWARD_AUDIT_VERSION,
            "status": "numeric_pass_visual_pending",
            "source_plan_commit": SOURCE_PLAN_COMMIT,
            "frozen_schema_generation_id": FROZEN_SCHEMA_GENERATION_ID,
            "frozen_schema_sha256": EXPECTED_FROZEN_SCHEMA_SHA256,
            "forward_spec_version": FULL_TRUEBONES_FORWARD_SPEC_VERSION,
            "scope": EXPECTED_SCOPE,
            "split_counts_all": dict(
                sorted(Counter(record["split"] for record in safe + rejected).items())
            ),
            "source_safe_parse_pass_count": len(parse_qa),
            "representative_count": len(manifest_records),
            "unavailable_rigs": sorted(EXPECTED_UNAVAILABLE_RIGS),
            "fallback_rest_rigs": fallback_rigs,
            "fallback_rest_rig_count": len(fallback_rigs),
            "max_J_phys": max(record["J_phys"] for record in manifest_records),
            "max_T_target": max(record["T_target"] for record in manifest_records),
            "coordinate_contract": (
                "right-handed; +Y up/screen-up; +Z out of screen toward viewer"
            ),
            "rotation_authority": "original_bvh_declared_rotation_channels_only",
            "legacy_btjd_motion_channels_used": False,
            "calibration_or_threshold_refit_performed": False,
            "visual_review_required_before_full_conversion": True,
            "full_conversion_authorized": False,
        }
        _write_json(staging / "audit_summary.json", summary)
        files = _file_manifest(staging)
        generation = {
            "audit_version": TRUEBONES_FORWARD_AUDIT_VERSION,
            "generation_id": generation_id,
            "created_at_utc": _datetime.datetime.now(_datetime.UTC).isoformat(),
            "source_plan_commit": SOURCE_PLAN_COMMIT,
            "status": summary["status"],
            "parent_manifest_generation_id": PARENT_MANIFEST_GENERATION_ID,
            "frozen_schema_generation_id": FROZEN_SCHEMA_GENERATION_ID,
            "selection_sha256": selection_sha,
            "files": files,
            "full_conversion_authorized": False,
        }
        _write_json(staging / "generation.json", generation)
        _fsync_directory(staging)
        if final.exists():
            raise TruebonesForwardAuditError(
                f"audit generation already exists: {final}"
            )
        os.replace(staging, final)
        _fsync_directory(generations)
        if cfg.overwrite_link:
            _replace_symlink(cfg.output_root / AUDIT_LINK_NAME, final)
        verify_forward_audit_generation(final)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {
        "status": "numeric_pass_visual_pending",
        "generation_id": generation_id,
        "generation_root": str(final),
        "compatibility_link": str(cfg.output_root / AUDIT_LINK_NAME),
        "source_safe_parse_pass_count": len(parse_qa),
        "representative_count": len(manifest_records),
        "upstream_reject_count": len(rejected),
        "unavailable_rigs": sorted(EXPECTED_UNAVAILABLE_RIGS),
        "full_conversion_authorized": False,
    }
