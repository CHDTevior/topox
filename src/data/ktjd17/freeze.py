"""Immutable KTJD-17 calibration freeze publication.

The prototype, train-only calibration, visual QA, and independent review are
separate immutable inputs.  This module closes those inputs, preserves the
declared prototype shortages, and publishes the only schema/stats pair that a
formal full conversion may consume.
"""

from __future__ import annotations

import dataclasses
import datetime as _datetime
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from .schema import KTJD17_SOURCE_PLAN_COMMIT, build_schema, write_schema


FREEZE_VERSION = "ktjd17-prototype-freeze-v1"
FREEZE_GENERATION_DIRECTORY = ".ktjd17_freeze_generations"
FREEZE_LINK_NAME = "ktjd17_freeze"
EXPECTED_FAMILY_COUNTS = {
    "dragon_or_deep_topology": 28,
    "human": 30,
    "quadruped": 30,
    "spider_crab": 30,
    "winged": 30,
}
EXPECTED_COVERAGE_GAPS = {"dragon_or_deep_topology": 2, "snake": 30}
EXPECTED_HELD_SELECTION_COUNTS = {"dragon_exact": 13, "snake": 10}
EXPECTED_GAINS = np.asarray(
    [3.867547101351066, 2.943516881261983, 3.3212471860907744],
    dtype=np.float64,
)
EXPECTED_SOURCE_RMS = np.asarray(
    [0.25856181548523766, 0.3397296636434668, 0.3010917116280753],
    dtype=np.float64,
)
EXPECTED_VALID_SCALAR_COUNTS = np.asarray(
    [2589480, 2589480, 37004], dtype=np.int64
)
EXPECTED_PROTOTYPE_GENERATION_ID = "20260819T175812150524Z-7e7115d87c89"
EXPECTED_PROTOTYPE_GENERATION_SHA256 = (
    "f6a0263d8e1bd6415a885cae168bf3ffd35f9d2442e13e1b60d3ab1a3c7f6d2f"
)
EXPECTED_CALIBRATION_GENERATION_ID = "20260819T184426865851Z-d3e11ee4b327"
EXPECTED_CALIBRATION_GENERATION_SHA256 = (
    "dfcf228cf1871b9a9ee0debf65cbc7785045956264e56e82e0a5c4a3a0b7ff4e"
)
EXPECTED_VISUAL_GENERATION_ID = "20260819T184450911885Z-020d74cdf492"
EXPECTED_VISUAL_GENERATION_SHA256 = (
    "dec51a41b15d18a718ba0a4a4ddca5796e0377d505a23643ab719aaedf859ce7"
)
EXPECTED_FIXED_QA_SHA256 = (
    "1865f4f77d2083ed1953e71df07a76f6b87569d2920cf42e7ab88310d36effa6"
)
EXPECTED_CODEX_REVIEW_SHA256 = (
    "5aaa2a8568f7602b08069dec23355da50a76fceb5c8356fe265bcca09b56b52c"
)
EXPECTED_CODEX_THREAD_ID = "01a01b59-1ed6-7a40-beb5-38afcd3176e7"


class FreezeError(RuntimeError):
    """A freeze input or immutable-publication invariant failed."""


@dataclasses.dataclass(frozen=True)
class FreezeConfig:
    prototype_root: Path
    fixed_qa_report: Path
    calibration_root: Path
    visual_root: Path
    codex_review: Path
    output_root: Path
    codex_thread_id: str
    overwrite_link: bool = True

    def resolved(self) -> "FreezeConfig":
        return dataclasses.replace(
            self,
            prototype_root=self.prototype_root.expanduser().resolve(),
            fixed_qa_report=self.fixed_qa_report.expanduser().resolve(),
            calibration_root=self.calibration_root.expanduser().resolve(),
            visual_root=self.visual_root.expanduser().resolve(),
            codex_review=self.codex_review.expanduser().resolve(),
            output_root=self.output_root.expanduser().absolute(),
        )


def default_freeze_config(repo_root: str | Path = ".") -> FreezeConfig:
    root = Path(repo_root).expanduser().resolve()
    return FreezeConfig(
        prototype_root=root
        / "dataset/.ktjd17_motion_generations/20260819T175812150524Z-7e7115d87c89",
        fixed_qa_report=root / "scratch/ktjd17_t08_fixed_qa.json",
        calibration_root=root
        / "dataset/.ktjd17_calibration_generations/20260819T184426865851Z-d3e11ee4b327",
        visual_root=root
        / "dataset/.ktjd17_visual_qa_generations/20260819T184450911885Z-020d74cdf492",
        codex_review=root / "scratch/_codex_ktjd17_calibration_visual_review.md",
        output_root=root / "dataset",
        codex_thread_id="01a01b59-1ed6-7a40-beb5-38afcd3176e7",
    )


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise FreezeError(f"cannot read JSON {path}: {exc}") from exc


def _write_json(path: Path, value: Any) -> None:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _file_manifest(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relpath = path.relative_to(root).as_posix()
        if relpath == "generation.json":
            continue
        result[relpath] = {
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return result


def verify_generation_file_closure(root: str | Path) -> dict[str, Any]:
    """Verify every generation-listed file and reject missing or extra files."""
    path = Path(root).expanduser().resolve()
    generation_path = path / "generation.json"
    generation = _load_json(generation_path)
    files = generation.get("files")
    if not isinstance(files, dict):
        raise FreezeError(f"{generation_path}: files must be an object")
    symlinks = [item for item in path.rglob("*") if item.is_symlink()]
    if symlinks:
        raise FreezeError(
            f"{path}: symlinks are forbidden inside immutable generations: "
            f"{[item.relative_to(path).as_posix() for item in symlinks]}"
        )
    actual = {
        item.relative_to(path).as_posix()
        for item in path.rglob("*")
        if item.is_file() and item.relative_to(path).as_posix() != "generation.json"
    }
    expected = set(files)
    if actual != expected:
        raise FreezeError(
            f"{path}: file closure mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    for relpath, metadata in files.items():
        target = path / relpath
        if target.resolve().parent != target.parent.resolve() or not target.resolve().is_relative_to(path):
            raise FreezeError(f"{target}: path escapes immutable generation")
        expected_size = int(metadata["size_bytes"])
        expected_sha = str(metadata["sha256"])
        if target.stat().st_size != expected_size:
            raise FreezeError(f"{target}: size drifted")
        if _sha256_file(target) != expected_sha:
            raise FreezeError(f"{target}: sha256 drifted")
    return generation


def validate_codex_freeze_review(text: str) -> None:
    """Require the exact independently issued freeze verdict."""
    required = (
        "VERDICT: PASS",
        "FREEZE RECOMMENDATION: PROCEED_WITH_DECLARED_SHORTAGES",
        "zero snake train calibration",
        "10 held snakes visual/read-only only",
        "deep-topology train coverage 28/30",
    )
    missing = [value for value in required if value not in text]
    if missing:
        raise FreezeError(f"Codex freeze review lacks required verdict text: {missing}")
    if "**Blocking Findings**\nNone." not in text:
        raise FreezeError("Codex freeze review does not state zero Blocking findings")
    if "**Major Findings**\nNone." not in text:
        raise FreezeError("Codex freeze review does not state zero Major findings")


def _replace_symlink(link: Path, target: Path) -> None:
    if link.exists() and not link.is_symlink():
        raise FreezeError(f"refusing to replace non-symlink {link}")
    relative = os.path.relpath(target, start=link.parent)
    temporary = link.parent / f".{link.name}.{uuid.uuid4().hex}.tmp"
    os.symlink(relative, temporary)
    os.replace(temporary, link)
    _fsync_directory(link.parent)


def _require_exact_array(label: str, actual: np.ndarray, expected: np.ndarray) -> None:
    if actual.dtype.kind != expected.dtype.kind or not np.array_equal(actual, expected):
        raise FreezeError(f"{label} drifted: {actual.tolist()} != {expected.tolist()}")


def run_freeze(config: FreezeConfig) -> dict[str, Any]:
    """Validate all freeze evidence and publish one immutable freeze generation."""
    cfg = config.resolved()
    for required in (
        cfg.prototype_root,
        cfg.calibration_root,
        cfg.visual_root,
    ):
        if not required.is_dir():
            raise FreezeError(f"missing generation directory {required}")
    for required in (cfg.fixed_qa_report, cfg.codex_review):
        if not required.is_file():
            raise FreezeError(f"missing freeze evidence {required}")

    prototype = verify_generation_file_closure(cfg.prototype_root)
    calibration = verify_generation_file_closure(cfg.calibration_root)
    visual = verify_generation_file_closure(cfg.visual_root)
    fixed_qa = _load_json(cfg.fixed_qa_report)
    calibration_report = _load_json(cfg.calibration_root / "calibration_report.json")
    candidate = _load_json(cfg.calibration_root / "candidate_freeze.json")
    visual_index = _load_json(cfg.visual_root / "visual_qa_index.json")
    review_text = cfg.codex_review.read_text(encoding="utf-8")
    validate_codex_freeze_review(review_text)

    pinned_inputs = (
        (
            cfg.prototype_root / "generation.json",
            EXPECTED_PROTOTYPE_GENERATION_SHA256,
            "prototype generation",
        ),
        (
            cfg.calibration_root / "generation.json",
            EXPECTED_CALIBRATION_GENERATION_SHA256,
            "calibration generation",
        ),
        (
            cfg.visual_root / "generation.json",
            EXPECTED_VISUAL_GENERATION_SHA256,
            "visual generation",
        ),
        (cfg.fixed_qa_report, EXPECTED_FIXED_QA_SHA256, "fixed QA"),
        (cfg.codex_review, EXPECTED_CODEX_REVIEW_SHA256, "Codex freeze review"),
    )
    for pinned_path, expected_sha, label in pinned_inputs:
        actual_sha = _sha256_file(pinned_path)
        if actual_sha != expected_sha:
            raise FreezeError(
                f"{label} is not the exact reviewed input: {actual_sha} != {expected_sha}"
            )
    if cfg.codex_thread_id != EXPECTED_CODEX_THREAD_ID:
        raise FreezeError(
            f"Codex thread id drifted: {cfg.codex_thread_id} != {EXPECTED_CODEX_THREAD_ID}"
        )

    prototype_id = str(prototype.get("generation_id"))
    calibration_id = str(calibration.get("generation_id"))
    visual_id = str(visual.get("generation_id"))
    if prototype_id != EXPECTED_PROTOTYPE_GENERATION_ID:
        raise FreezeError("prototype generation id drifted")
    if calibration_id != EXPECTED_CALIBRATION_GENERATION_ID:
        raise FreezeError("calibration generation id drifted")
    if visual_id != EXPECTED_VISUAL_GENERATION_ID:
        raise FreezeError("visual generation id drifted")
    if prototype.get("source_plan_commit") != KTJD17_SOURCE_PLAN_COMMIT:
        raise FreezeError("prototype source-plan commit drifted")
    if calibration.get("prototype_generation_id") != prototype_id:
        raise FreezeError("calibration/prototype generation relation drifted")
    if visual.get("prototype_generation_id") != prototype_id:
        raise FreezeError("visual/prototype generation relation drifted")
    if visual.get("calibration_generation_id") != calibration_id:
        raise FreezeError("visual/calibration generation relation drifted")
    if _sha256_file(cfg.fixed_qa_report) != calibration.get("fixed_qa_sha256"):
        raise FreezeError("fixed-QA report hash differs from calibration authority")
    if prototype.get("status") != "incomplete" or prototype.get(
        "full_conversion_authorized"
    ) is not False:
        raise FreezeError("prototype pre-freeze status/authorization drifted")
    if calibration.get("status") != "numeric_pass_pending_visual_and_coverage_review":
        raise FreezeError("calibration pre-freeze status drifted")
    if calibration.get("freeze_authorized") is not False or calibration.get(
        "full_conversion_authorized"
    ) is not False:
        raise FreezeError("calibration unexpectedly self-authorizes")
    if visual.get("status") != "pending_human_and_codex_visual_review":
        raise FreezeError("visual pre-freeze status drifted")
    if visual.get("freeze_authorized") is not False or visual.get(
        "full_conversion_authorized"
    ) is not False:
        raise FreezeError("visual generation unexpectedly self-authorizes")
    if fixed_qa.get("status") != "pass" or int(fixed_qa.get("fail_count", -1)) != 0:
        raise FreezeError("fixed-QA report is not a zero-failure pass")
    if int(fixed_qa.get("clip_count", -1)) != 171:
        raise FreezeError("fixed-QA scope is not 171 clips")

    scope = calibration_report.get("scope", {})
    if scope.get("split") != "train" or scope.get("validation_or_held_tuning_used") is not False:
        raise FreezeError("calibration is not strictly train-only")
    if int(scope.get("clip_count", -1)) != 148 or int(scope.get("held_clip_count", -1)) != 0:
        raise FreezeError("calibration train/held counts drifted")
    if scope.get("family_counts") != EXPECTED_FAMILY_COUNTS:
        raise FreezeError("calibration family counts drifted")
    if scope.get("coverage_gaps") != EXPECTED_COVERAGE_GAPS:
        raise FreezeError("declared calibration shortages drifted")
    if candidate.get("status") != "candidate_unfrozen":
        raise FreezeError("input calibration candidate is not explicitly unfrozen")
    if candidate.get("freeze_authorized") is not False:
        raise FreezeError("input candidate unexpectedly self-authorizes freeze")
    if candidate.get("full_conversion_authorized") is not False:
        raise FreezeError("input candidate unexpectedly self-authorizes full conversion")
    if calibration_report.get("candidate_freeze") != candidate:
        raise FreezeError("calibration report and candidate freeze payload differ")

    selection = _load_json(
        cfg.prototype_root / "manifests/prototype_selection.json"
    )
    selection_counts = selection.get("selection_counts", {})
    snake_selection = selection_counts.get("snake", {})
    dragon_selection = selection_counts.get("dragon_exact_held", {})
    if (
        snake_selection.get("calibration_selected") != 0
        or snake_selection.get("calibration_shortage") != 30
        or snake_selection.get("read_only_selected") != 10
        or selection.get("held_data_used_for_calibration") is not False
    ):
        raise FreezeError("prototype snake shortage/read-only evidence drifted")
    if (
        dragon_selection.get("read_only_selected") != 13
        or dragon_selection.get("read_only_shortage") != 0
    ):
        raise FreezeError("prototype exact-Dragon read-only evidence drifted")

    if visual_index.get("coordinate_contract") != (
        "right-handed; +Y is screen-up; +Z points out of the screen toward the viewer"
    ):
        raise FreezeError("visual coordinate contract drifted")
    for key, expected in (
        ("perspective_camera", True),
        ("fixed_camera_across_frames_and_paths", True),
        ("frame_recenter_applied", False),
        ("ground_changed", False),
        ("face_direction_changed", False),
    ):
        if visual_index.get(key) is not expected:
            raise FreezeError(f"visual QA flag {key} drifted")
    if visual_index.get("required_paths") != [
        "source",
        "position-direct",
        "rotation-FK",
    ]:
        raise FreezeError("visual QA path contract drifted")
    held_visual_counts: dict[str, int] = {
        key: 0 for key in EXPECTED_HELD_SELECTION_COUNTS
    }
    for record in visual_index.get("clips", []):
        role = str(record.get("visual_role"))
        if role in held_visual_counts:
            # Counts are read from the immutable prototype selection, not merely
            # the one rendered representative per held role.
            held_visual_counts[role] = 1
    if held_visual_counts != {"dragon_exact": 1, "snake": 1}:
        raise FreezeError("visual QA lacks held snake or exact-Dragon representative")

    with np.load(
        cfg.calibration_root / "candidate_train_block_gains.npz",
        allow_pickle=False,
    ) as source_stats:
        gains = np.asarray(source_stats["gains"], dtype=np.float64)
        source_rms = np.asarray(source_stats["source_rms"], dtype=np.float64)
        valid_counts = np.asarray(source_stats["valid_scalar_counts"], dtype=np.int64)
        clip_ids = np.asarray(source_stats["clip_ids"])
    _require_exact_array("normalization gains", gains, EXPECTED_GAINS)
    _require_exact_array("normalization source RMS", source_rms, EXPECTED_SOURCE_RMS)
    _require_exact_array(
        "normalization valid scalar counts", valid_counts, EXPECTED_VALID_SCALAR_COUNTS
    )
    if clip_ids.shape != (148,) or len(set(clip_ids.tolist())) != 148:
        raise FreezeError("normalization clip ids are not 148 unique train clips")
    if not np.allclose(gains * source_rms, np.ones(3), rtol=0.0, atol=2e-15):
        raise FreezeError("normalization gains are not reciprocal RMS")

    smoother = candidate["smoother"]
    generation_id = (
        _datetime.datetime.now(_datetime.UTC).strftime("%Y%m%dT%H%M%S%fZ")
        + "-"
        + uuid.uuid4().hex[:12]
    )
    created_at = _datetime.datetime.now(_datetime.UTC).isoformat()
    schema = build_schema(
        fps_target=float(candidate["fps_target"]),
        smoother_id=str(smoother["id"]),
        smoother_params=dict(smoother["params"]),
        short_clip_rule=str(smoother["short_clip_rule"]),
        heading_eps_h=float(candidate["heading"]["eps_h"]),
        contact_tau_h=float(candidate["contact"]["tau_h"]),
        contact_tau_v=float(candidate["contact"]["tau_v"]),
        normalization_gains=gains.tolist(),
        j_max=int(candidate["topology"]["J_max"]),
        frozen=True,
        calibration_run_ids=[calibration_id, visual_id],
        train_split_protocol=(
            "prototype-train-only-selection:"
            + str(scope["selection_sha256"])
        ),
        frozen_at_utc=created_at,
    )
    decision = {
        "freeze_version": FREEZE_VERSION,
        "generation_id": generation_id,
        "status": "frozen_with_declared_shortages",
        "source_plan_commit": KTJD17_SOURCE_PLAN_COMMIT,
        "freeze_authorized": True,
        "full_build_may_start": True,
        "full_build_complete": False,
        "scope": "KTJD-17 v1 schema parameters from train-only six-family prototype",
        "coordinate_contract": (
            "right-handed; +Y up/screen-up; +Z out of screen toward viewer"
        ),
        "calibration_scope": {
            "train_clip_count": 148,
            "held_or_validation_clip_count": 0,
            "family_counts": EXPECTED_FAMILY_COUNTS,
            "validation_or_held_tuning_used": False,
        },
        "declared_limitations": {
            "dragon_or_deep_topology": {
                "target_train_clips": 30,
                "actual_train_clips": 28,
                "shortage": 2,
            },
            "snake": {
                "target_train_clips": 30,
                "actual_train_clips": 0,
                "shortage": 30,
                "held_snake_clips": 10,
                "held_snake_role": "visual_and_read_only_only",
                "held_used_for_calibration": False,
            },
            "exact_dragon": {
                "held_clips": 13,
                "held_role": "visual_and_read_only_only",
                "held_used_for_calibration": False,
            },
            "contact": {
                "semantics": "deterministic_joint_proxy_ground_support",
                "source_contact_labels_available": False,
                "claim_boundary": (
                    "zero recomputation mismatch proves formula consistency only; "
                    "it is not source-label contact validation"
                ),
            },
        },
        "inputs": {
            "prototype": {
                "generation_id": prototype_id,
                "generation_json_sha256": _sha256_file(
                    cfg.prototype_root / "generation.json"
                ),
            },
            "fixed_qa": {
                "sha256": _sha256_file(cfg.fixed_qa_report),
                "clip_count": 171,
                "pass_count": 171,
            },
            "calibration": {
                "generation_id": calibration_id,
                "generation_json_sha256": _sha256_file(
                    cfg.calibration_root / "generation.json"
                ),
            },
            "visual_qa": {
                "generation_id": visual_id,
                "generation_json_sha256": _sha256_file(
                    cfg.visual_root / "generation.json"
                ),
                "rendered_roles": [
                    str(record["visual_role"]) for record in visual_index["clips"]
                ],
            },
            "codex_review": {
                "model": "gpt-5.5",
                "reasoning_effort": "xhigh",
                "thread_id": cfg.codex_thread_id,
                "sha256": _sha256_file(cfg.codex_review),
                "verdict": "PASS",
                "freeze_recommendation": "PROCEED_WITH_DECLARED_SHORTAGES",
            },
        },
        "frozen_values": {
            "fps_target": schema["fps_target"],
            "smoother": schema["smoother"],
            "contact": schema["contact"],
            "heading": schema["heading"],
            "normalization": schema["normalization"],
            "topology": schema["topology"],
        },
    }

    generations = cfg.output_root / FREEZE_GENERATION_DIRECTORY
    generations.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{generation_id}.", dir=generations))
    final = generations / generation_id
    try:
        write_schema(schema, staging / "schema.json")
        (staging / "stats").mkdir(parents=True, exist_ok=False)
        np.savez_compressed(
            staging / "stats/train_block_gains.npz",
            gains=gains,
            g_q=np.asarray(gains[0], dtype=np.float64),
            g_v=np.asarray(gains[1], dtype=np.float64),
            g_s=np.asarray(gains[2], dtype=np.float64),
            source_rms=source_rms,
            valid_scalar_counts=valid_counts,
            clip_ids=clip_ids,
            prototype_generation_id=np.asarray(prototype_id),
            calibration_generation_id=np.asarray(calibration_id),
            visual_generation_id=np.asarray(visual_id),
            freeze_generation_id=np.asarray(generation_id),
            split=np.asarray("train"),
            calibration_version=np.asarray(calibration["calibration_version"]),
            frozen=np.asarray(True, dtype=np.bool_),
        )
        _fsync_file(staging / "stats/train_block_gains.npz")
        _write_json(staging / "freeze_decision.json", decision)
        (staging / "reviews").mkdir(parents=True, exist_ok=False)
        (staging / "evidence").mkdir(parents=True, exist_ok=False)
        shutil.copy2(cfg.codex_review, staging / "reviews/codex_freeze_review.md")
        shutil.copy2(cfg.fixed_qa_report, staging / "evidence/fixed_qa_report.json")
        shutil.copy2(
            cfg.calibration_root / "candidate_freeze.json",
            staging / "evidence/candidate_freeze.json",
        )
        generation = {
            "freeze_version": FREEZE_VERSION,
            "generation_id": generation_id,
            "created_at_utc": created_at,
            "status": "frozen_with_declared_shortages",
            "source_plan_commit": KTJD17_SOURCE_PLAN_COMMIT,
            "freeze_authorized": True,
            "full_build_may_start": True,
            "full_build_complete": False,
            "prototype_generation_id": prototype_id,
            "calibration_generation_id": calibration_id,
            "visual_generation_id": visual_id,
            "codex_review_sha256": _sha256_file(cfg.codex_review),
            "files": _file_manifest(staging),
        }
        _write_json(staging / "generation.json", generation)
        for directory in (
            staging / "stats",
            staging / "reviews",
            staging / "evidence",
            staging,
        ):
            _fsync_directory(directory)
        os.replace(staging, final)
        _fsync_directory(generations)
        if cfg.overwrite_link:
            _replace_symlink(cfg.output_root / FREEZE_LINK_NAME, final)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    result = {
        "status": "frozen_with_declared_shortages",
        "generation_id": generation_id,
        "generation_root": str(final),
        "schema_path": str(final / "schema.json"),
        "stats_path": str(final / "stats/train_block_gains.npz"),
        "freeze_authorized": True,
        "full_build_may_start": True,
        "declared_coverage_gaps": EXPECTED_COVERAGE_GAPS,
    }
    return result
