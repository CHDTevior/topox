"""Six-family KTJD-17 prototype selection, conversion, and publication."""

from __future__ import annotations

import dataclasses
import datetime as _datetime
import hashlib
import json
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
    Ktjd17EncoderError,
    encode_prepared_motion,
    load_skeleton,
    prepare_manifest_clip,
    skeleton_from_human_contract,
    write_npz_atomic,
)
from .human_fixed_rig import build_current_btjd_human_fixed_rig
from .truebones_fixed_rig import (
    ACTIVE_COND_SHA256,
    TRUEBONES_FORWARD_SPECS,
    load_conditioning_catalog,
)


PROTOTYPE_VERSION = "ktjd17-six-family-prototype-v2"
MOTION_GENERATION_DIRECTORY = ".ktjd17_motion_generations"
PROTOTYPE_LINK_NAME = "ktjd17_prototype"
_FAILURE_MANIFEST_FIELDS = (
    "clip_id",
    "rig_id",
    "source_family",
    "topology_family",
    "topology_distance_bucket",
    "family_role",
    "audit_role",
    "calibration_eligible",
    "selection_origin",
    "replaces_parent_clip_id",
    "split",
    "status",
    "reason_codes",
    "error_type",
    "error",
)


class PrototypeBuildError(RuntimeError):
    """Prototype selection/publication violated a fail-closed contract."""


def _project_failure_manifest_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Project an encoder failure without losing required stratification fields."""
    bucket = record.get("topology_distance_bucket")
    if not isinstance(bucket, str) or not bucket:
        raise PrototypeBuildError(
            "encoder failure record is missing topology_distance_bucket"
        )
    return {key: record[key] for key in _FAILURE_MANIFEST_FIELDS if key in record}


@dataclasses.dataclass(frozen=True)
class PrototypeConfig:
    manifest_root: Path
    dataset_root: Path
    output_root: Path
    active_cond_path: Path
    legacy_truebones_cond_path: Path
    encoder: EncoderConfig
    overwrite_link: bool = True

    def resolved(self) -> "PrototypeConfig":
        return dataclasses.replace(
            self,
            manifest_root=self.manifest_root.expanduser().resolve(),
            dataset_root=self.dataset_root.expanduser().resolve(),
            output_root=self.output_root.expanduser().absolute(),
            active_cond_path=self.active_cond_path.expanduser().resolve(),
            legacy_truebones_cond_path=self.legacy_truebones_cond_path.expanduser().resolve(),
        )


@dataclasses.dataclass(frozen=True)
class SelectedClip:
    clip_id: str
    family_role: str
    audit_role: str
    calibration_eligible: bool
    selection_origin: str
    replaces_parent_clip_id: str | None = None


def default_prototype_config(repo_root: str | Path = ".") -> PrototypeConfig:
    root = Path(repo_root).expanduser().resolve()
    return PrototypeConfig(
        manifest_root=root
        / "dataset/.ktjd17_manifest_generations/20260819T145535975831Z-ed48b3fd2745",
        dataset_root=root / "dataset",
        output_root=root / "dataset",
        active_cond_path=root / "data/animo4d_L4TB_plus_human_v4b272neutral/cond.npy",
        legacy_truebones_cond_path=root / "data/anytop_truebones/cond.npy",
        encoder=EncoderConfig(
            fps_target=30.0,
            smoother=SmootherConfig(),
            contact_tau_h=0.05,
            contact_tau_v=0.25,
            heading_eps_h=0.05,
            calibration_status="candidate_unfrozen",
        ),
    )


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise PrototypeBuildError(f"cannot read JSON {path}: {exc}") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise PrototypeBuildError(f"{path}:{line_number}: blank JSONL row")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise PrototypeBuildError(
                        f"{path}:{line_number}: row is not an object"
                    )
                records.append(value)
    except PrototypeBuildError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise PrototypeBuildError(f"cannot read JSONL {path}: {exc}") from exc
    return records


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


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


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(_canonical_json(record).decode("utf-8") + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _round_robin(records: Sequence[dict[str, Any]], limit: int) -> list[str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in sorted(records, key=lambda item: (item["rig_id"], item["clip_id"])):
        grouped[str(record["rig_id"])].append(str(record["clip_id"]))
    result: list[str] = []
    keys = sorted(grouped)
    while len(result) < limit:
        progressed = False
        for key in keys:
            if grouped[key] and len(result) < limit:
                result.append(grouped[key].pop(0))
                progressed = True
        if not progressed:
            break
    return result


def _require_string_list(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise PrototypeBuildError(f"{label} must be a list of strings")
    if len(value) != len(set(value)):
        raise PrototypeBuildError(f"{label} contains duplicate clip ids")
    return list(value)


def _parent_candidate_lists(
    families: Mapping[str, Any], family: str
) -> tuple[int, list[str], list[str], list[str]]:
    payload = families.get(family)
    if not isinstance(payload, Mapping):
        raise PrototypeBuildError(f"prototype_candidates.json lacks family {family}")
    target = int(payload.get("required_train_clips", -1))
    parent_selected = _require_string_list(
        payload.get("selected_train_candidates"),
        label=f"{family}.selected_train_candidates",
    )
    if target <= 0 or len(parent_selected) != target:
        raise PrototypeBuildError(
            f"{family}: immutable parent selection has {len(parent_selected)} clips "
            f"for target {target}"
        )
    t04 = payload.get("canonical_skeleton_t04")
    if not isinstance(t04, Mapping):
        raise PrototypeBuildError(f"{family}: canonical_skeleton_t04 is absent")
    eligible = _require_string_list(
        t04.get("eligible_selected_train_clips"),
        label=f"{family}.canonical_skeleton_t04.eligible_selected_train_clips",
    )
    ineligible = _require_string_list(
        t04.get("ineligible_selected_train_clips"),
        label=f"{family}.canonical_skeleton_t04.ineligible_selected_train_clips",
    )
    if set(eligible).intersection(ineligible) or set(eligible).union(ineligible) != set(
        parent_selected
    ):
        raise PrototypeBuildError(
            f"{family}: T04 eligible/ineligible partition does not exactly cover "
            "the immutable parent selection"
        )
    if int(t04.get("eligible_count", -1)) != len(eligible):
        raise PrototypeBuildError(f"{family}: T04 eligible_count drifted")
    if int(t04.get("shortage", -1)) != target - len(eligible):
        raise PrototypeBuildError(f"{family}: T04 shortage drifted")
    if t04.get("selection_replaced") is not False:
        raise PrototypeBuildError(
            f"{family}: expected immutable T04 selection_replaced=false"
        )
    return target, parent_selected, eligible, ineligible


def _truebones_train_eligible(record: Mapping[str, Any], family: str) -> bool:
    source_fk = record.get("source_parser_fk")
    source_fk_not_known_failed = not isinstance(source_fk, Mapping) or (
        source_fk.get("status") == "pass"
    )
    return bool(
        record.get("source", {}).get("family") == "truebones"
        and record.get("topology_family") == family
        and record.get("split") == "train"
        and record.get("split_eligible_for_train_calibration") is True
        and record.get("status") != "reject"
        and record.get("canonical_skeleton", {}).get("status") == "pass"
        and record.get("rig_id") in TRUEBONES_FORWARD_SPECS
        and source_fk_not_known_failed
    )


def _select_train_family_overlay(
    *,
    family: str,
    clips: Sequence[dict[str, Any]],
    clip_index: Mapping[str, dict[str, Any]],
    families: Mapping[str, Any],
) -> tuple[list[SelectedClip], dict[str, Any]]:
    target, parent_selected, parent_eligible, parent_ineligible = (
        _parent_candidate_lists(families, family)
    )
    for clip_id in parent_eligible:
        record = clip_index.get(clip_id)
        if record is None or not _truebones_train_eligible(record, family):
            raise PrototypeBuildError(
                f"{family}: immutable T04-eligible parent clip drifted: {clip_id}"
            )

    # T04 intentionally did not replace failed frozen candidates.  T05 keeps
    # every eligible frozen candidate, then creates an explicit deterministic
    # overlay from the same immutable clips.jsonl.  Every substitution and its
    # displaced parent id is serialized in prototype_selection.json.
    parent_ids = set(parent_selected)
    replacement_pool = [
        record
        for record in clips
        if str(record.get("clip_id")) not in parent_ids
        and _truebones_train_eligible(record, family)
    ]
    additions = _round_robin(replacement_pool, target - len(parent_eligible))
    replacement_pairs = [
        {"parent_clip_id": parent_id, "replacement_clip_id": replacement_id}
        for parent_id, replacement_id in zip(parent_ineligible, additions, strict=False)
    ]
    selected = [
        SelectedClip(
            clip_id,
            family,
            "prototype_train_calibration",
            True,
            "immutable_t04_eligible_parent_candidate",
        )
        for clip_id in parent_eligible
    ]
    selected.extend(
        SelectedClip(
            replacement_id,
            family,
            "prototype_train_calibration",
            True,
            "explicit_t05_replacement_from_pinned_parent_manifest",
            parent_id,
        )
        for parent_id, replacement_id in zip(parent_ineligible, additions, strict=False)
    )
    selected_ids = [entry.clip_id for entry in selected]
    audit = {
        "target_train_calibration_clips": target,
        "selected": len(selected),
        "calibration_selected": len(selected),
        "calibration_shortage": target - len(selected),
        "parent_selected_train_candidates": parent_selected,
        "parent_t04_eligible_selected_train_clips": parent_eligible,
        "parent_t04_ineligible_selected_train_clips": parent_ineligible,
        "t05_added_replacement_clips": additions,
        "replacement_pairs": replacement_pairs,
        "selected_clips": selected_ids,
        "selection_replaced": bool(additions),
        "selection_policy": (
            "preserve immutable T04-eligible candidates, then fill from the "
            "pinned T04 clips manifest with deterministic rig round-robin; "
            "all additions rerun source-FK and fixed-rig gates during T05 encoding"
        ),
    }
    return selected, audit


def select_prototype_clips(
    clips: Sequence[dict[str, Any]],
    prototype_candidates: Mapping[str, Any],
) -> tuple[list[SelectedClip], dict[str, Any]]:
    clip_index = {str(record["clip_id"]): record for record in clips}
    if len(clip_index) != len(clips):
        raise PrototypeBuildError("clips.jsonl contains duplicate clip ids")
    families = prototype_candidates.get("families")
    if not isinstance(families, Mapping):
        raise PrototypeBuildError("prototype_candidates.json lacks families")

    selected: list[SelectedClip] = []
    human_target, human, human_t04_eligible, human_t04_ineligible = (
        _parent_candidate_lists(families, "human")
    )
    if human_t04_eligible:
        raise PrototypeBuildError(
            "Human T05 override expected the immutable T04 Human selection to be ineligible"
        )
    for clip_id in human:
        record = clip_index.get(str(clip_id))
        if (
            record is None
            or record.get("source", {}).get("family") != "motionstreamer272"
            or record.get("topology_family") != "human"
            or record.get("split") != "train"
        ):
            raise PrototypeBuildError(f"unsafe Human prototype {clip_id}")
        selected.append(
            SelectedClip(
                str(clip_id),
                "human",
                "prototype_train_calibration",
                True,
                "immutable_parent_candidate_with_reviewed_t05_human_fixed_rig_override",
            )
        )

    selection_counts: dict[str, dict[str, Any]] = {
        "human": {
            "target_train_calibration_clips": human_target,
            "selected": len(human),
            "calibration_selected": len(human),
            "calibration_shortage": 0,
            "parent_selected_train_candidates": human,
            "parent_t04_eligible_selected_train_clips": human_t04_eligible,
            "parent_t04_ineligible_selected_train_clips": human_t04_ineligible,
            "t05_added_replacement_clips": [],
            "replacement_pairs": [],
            "selected_clips": human,
            "selection_replaced": False,
            "fixed_rig_gate_overridden": True,
            "selection_policy": (
                "preserve immutable Human candidate ids; use the separately reviewed "
                "T05 fixed-neutral Human rig override"
            ),
        }
    }
    for family in ("quadruped", "winged", "spider_crab"):
        family_selected, family_audit = _select_train_family_overlay(
            family=family,
            clips=clips,
            clip_index=clip_index,
            families=families,
        )
        if family_audit["calibration_shortage"]:
            raise PrototypeBuildError(
                f"{family} requires {family_audit['target_train_calibration_clips']} "
                f"pinned pass/train clips, got {family_audit['calibration_selected']}"
            )
        selected.extend(family_selected)
        selection_counts[family] = family_audit

    deep_selected, deep_audit = _select_train_family_overlay(
        family="dragon_or_deep_topology",
        clips=clips,
        clip_index=clip_index,
        families=families,
    )
    selected.extend(deep_selected)
    deep_audit["reason"] = (
        "the pinned T04 manifest exposes only accepted fixed-rig Truebones clips; "
        "rejected PlanetZoo candidates are never substituted silently"
    )
    selection_counts["dragon_or_deep_topology"] = deep_audit

    snakes = sorted(
        str(record["clip_id"])
        for record in clips
        if record.get("source", {}).get("family") == "truebones"
        and record.get("rig_id") == "KingCobra"
        and record.get("split") == "held_representative"
        and record.get("status") != "reject"
        and record.get("canonical_skeleton", {}).get("status") == "pass"
    )
    selected.extend(
        SelectedClip(
            clip_id,
            "snake",
            "held_representative_read_only",
            False,
            "pinned_parent_manifest_held_query",
        )
        for clip_id in snakes
    )
    selection_counts["snake"] = {
        "selected": len(snakes),
        "read_only_selected": len(snakes),
        "selected_clips": snakes,
        "target_train_calibration_clips": 30,
        "calibration_selected": 0,
        "calibration_shortage": 30,
        "selection_policy": "exact sorted held_representative query on pinned clips.jsonl",
        "reason": (
            "no accepted train-split snake clips; held clips are visual/read-only "
            "and never satisfy calibration coverage"
        ),
    }

    dragons = sorted(
        str(record["clip_id"])
        for record in clips
        if record.get("source", {}).get("family") == "truebones"
        and record.get("rig_id") == "Dragon"
        and record.get("split") == "held_stress"
        and record.get("status") != "reject"
        and record.get("canonical_skeleton", {}).get("status") == "pass"
    )
    selected.extend(
        SelectedClip(
            clip_id,
            "dragon_exact",
            "held_stress_exact_dragon_read_only",
            False,
            "pinned_parent_manifest_held_query",
        )
        for clip_id in dragons
    )
    selection_counts["dragon_exact_held"] = {
        "selected": len(dragons),
        "read_only_selected": len(dragons),
        "selected_clips": dragons,
        "expected_read_only_clips": 13,
        "read_only_shortage": max(0, 13 - len(dragons)),
        "selection_policy": "exact sorted held_stress Dragon query on pinned clips.jsonl",
        "reason": "exact Dragon is held-stress read-only",
    }
    ids = [entry.clip_id for entry in selected]
    if len(ids) != len(set(ids)):
        raise PrototypeBuildError("prototype selection contains duplicate clips")
    expected_total = sum(
        int(selection_counts[family]["selected"])
        for family in (
            "human",
            "quadruped",
            "winged",
            "spider_crab",
            "dragon_or_deep_topology",
            "snake",
            "dragon_exact_held",
        )
    )
    if len(selected) != expected_total:
        raise PrototypeBuildError("prototype selection count arithmetic drifted")
    return selected, selection_counts


def _resolve_t04_skeleton(dataset_root: Path, clip: Mapping[str, Any]) -> Path:
    payload = clip.get("canonical_skeleton")
    if not isinstance(payload, Mapping):
        raise PrototypeBuildError(f"{clip.get('clip_id')}: no canonical skeleton metadata")
    relpath = payload.get("artifact_relpath")
    expected_sha = payload.get("artifact_sha256")
    if not isinstance(relpath, str) or not isinstance(expected_sha, str):
        raise PrototypeBuildError(f"{clip.get('clip_id')}: incomplete skeleton reference")
    path = (dataset_root / relpath).resolve()
    actual_sha = _sha256_file(path)
    if actual_sha != expected_sha:
        raise PrototypeBuildError(
            f"{clip.get('clip_id')}: skeleton hash drifted {actual_sha} != {expected_sha}"
        )
    return path


def _replace_symlink(link: Path, target: Path) -> None:
    if link.exists() and not link.is_symlink():
        raise PrototypeBuildError(f"refusing to replace non-symlink {link}")
    relative = os.path.relpath(target, start=link.parent)
    temporary = link.parent / f".{link.name}.{uuid.uuid4().hex}.tmp"
    os.symlink(relative, temporary)
    os.replace(temporary, link)


def _file_manifest(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relpath = path.relative_to(root).as_posix()
        result[relpath] = {
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return result


def run_prototype_build(config: PrototypeConfig) -> dict[str, Any]:
    cfg = config.resolved()
    cfg.encoder.validate()
    required = [
        "clips.jsonl",
        "rigs.jsonl",
        "prototype_candidates.json",
        "canonical_skeleton_qa.jsonl",
        "canonical_skeleton_generation.json",
    ]
    missing = [name for name in required if not (cfg.manifest_root / name).is_file()]
    if missing:
        raise PrototypeBuildError(f"manifest root is incomplete: {missing}")
    clips_path = cfg.manifest_root / "clips.jsonl"
    candidates_path = cfg.manifest_root / "prototype_candidates.json"
    clips = _load_jsonl(clips_path)
    rigs = {record["rig_id"]: record for record in _load_jsonl(cfg.manifest_root / "rigs.jsonl")}
    prototype_candidates = _load_json(candidates_path)
    selected, selection_counts = select_prototype_clips(
        clips, prototype_candidates
    )
    selected_records = [dataclasses.asdict(value) for value in selected]
    selection_authority = {
        "policy_version": "ktjd17-t05-explicit-selection-overlay-v1",
        "parent_manifest_root": str(cfg.manifest_root),
        "parent_clips_jsonl_sha256": _sha256_file(clips_path),
        "parent_prototype_candidates_sha256": _sha256_file(candidates_path),
        "parent_selection_is_immutable": True,
        "silent_replacement_allowed": False,
    }
    selection_sha256 = hashlib.sha256(
        _canonical_json(
            {
                "selection_authority": selection_authority,
                "selection_counts": selection_counts,
                "selected": selected_records,
            }
        )
    ).hexdigest()
    clip_index = {record["clip_id"]: record for record in clips}

    generation_id = (
        _datetime.datetime.now(_datetime.UTC).strftime("%Y%m%dT%H%M%S%fZ")
        + "-"
        + uuid.uuid4().hex[:12]
    )
    generations = cfg.output_root / MOTION_GENERATION_DIRECTORY
    generations.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{generation_id}.", dir=generations)
    )
    final = generations / generation_id
    rest_cache: dict[str, ParsedBvhMotion] = {}
    conditioning = load_conditioning_catalog(
        cfg.active_cond_path,
        expected_active_sha256=ACTIVE_COND_SHA256,
        legacy_path=cfg.legacy_truebones_cond_path,
    )
    qa_records: list[dict[str, Any]] = []
    manifest_records: list[dict[str, Any]] = []
    skeleton_cache: dict[str, Any] = {}
    copied_skeletons: dict[str, Path] = {}
    try:
        human_candidate = (
            cfg.dataset_root
            / ".ktjd17_skeleton_generations/20260819T145532135993Z-77bd88e242a2/candidates/HML3D_Human.npz"
        )
        human_contract = build_current_btjd_human_fixed_rig(
            rig_record=rigs["HML3D_Human"],
            active_cond_path=cfg.active_cond_path,
            legacy_truebones_cond_path=cfg.legacy_truebones_cond_path,
            t04_candidate_path=human_candidate,
        )
        human_path = staging / "skeletons/HML3D_Human.npz"
        human_sha = write_npz_atomic(human_path, human_contract.payload)
        human_skeleton = load_skeleton(human_path)
        if human_sha != human_skeleton.sha256:
            raise PrototypeBuildError("Human skeleton write hash verification failed")
        skeleton_cache["HML3D_Human"] = human_skeleton
        copied_skeletons["HML3D_Human"] = human_path

        for index, selected_clip in enumerate(selected, start=1):
            clip = clip_index[selected_clip.clip_id]
            rig = rigs[clip["rig_id"]]
            rig_id = str(clip["rig_id"])
            try:
                if rig_id not in skeleton_cache:
                    source_skeleton = _resolve_t04_skeleton(cfg.dataset_root, clip)
                    target_skeleton = staging / "skeletons" / f"{rig_id}.npz"
                    target_skeleton.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_skeleton, target_skeleton)
                    if _sha256_file(target_skeleton) != _sha256_file(source_skeleton):
                        raise PrototypeBuildError(
                            f"{rig_id}: copied skeleton hash verification failed"
                        )
                    skeleton_cache[rig_id] = load_skeleton(target_skeleton)
                    copied_skeletons[rig_id] = target_skeleton
                skeleton = skeleton_cache[rig_id]
                prepared = prepare_manifest_clip(
                    clip,
                    rig,
                    skeleton,
                    conditioning_catalog=(
                        None if rig_id == "HML3D_Human" else conditioning
                    ),
                    rest_cache=rest_cache,
                )
                encoded = encode_prepared_motion(prepared, skeleton, cfg.encoder)
                motion_relpath = f"motions/{encoded.clip_id}.npz"
                motion_path = staging / motion_relpath
                motion_sha = write_npz_atomic(motion_path, encoded.artifact_payload())
                published_provenance = dict(encoded.provenance)
                published_provenance.pop("skeleton_path", None)
                published_provenance["skeleton_relpath"] = f"skeletons/{rig_id}.npz"
                published_provenance["skeleton_resolution"] = (
                    "generation_relative_relpath_plus_sha256"
                )
                qa = {
                    "prototype_version": PROTOTYPE_VERSION,
                    "clip_id": encoded.clip_id,
                    "rig_id": encoded.rig_id,
                    "source_family": prepared.source_family,
                    "topology_family": prepared.topology_family,
                    "topology_distance_bucket": clip["topology_distance_bucket"],
                    "family_role": selected_clip.family_role,
                    "audit_role": selected_clip.audit_role,
                    "calibration_eligible": selected_clip.calibration_eligible,
                    "selection_origin": selected_clip.selection_origin,
                    "replaces_parent_clip_id": selected_clip.replaces_parent_clip_id,
                    "split": clip.get("split"),
                    "status": "pass",
                    "reason_codes": [],
                    "motion_relpath": motion_relpath,
                    "motion_sha256": motion_sha,
                    "motion_size_bytes": motion_path.stat().st_size,
                    "skeleton_relpath": f"skeletons/{rig_id}.npz",
                    "skeleton_sha256": skeleton.sha256,
                    "metrics": encoded.metrics,
                    "provenance": published_provenance,
                }
                qa_records.append(qa)
                manifest_records.append(
                    {
                        "clip_id": encoded.clip_id,
                        "rig_id": encoded.rig_id,
                        "source_family": prepared.source_family,
                        "topology_family": prepared.topology_family,
                        "topology_distance_bucket": clip["topology_distance_bucket"],
                        "family_role": selected_clip.family_role,
                        "audit_role": selected_clip.audit_role,
                        "calibration_eligible": selected_clip.calibration_eligible,
                        "selection_origin": selected_clip.selection_origin,
                        "replaces_parent_clip_id": selected_clip.replaces_parent_clip_id,
                        "split": clip.get("split"),
                        "status": "accept",
                        "reason_codes": [],
                        "motion_relpath": motion_relpath,
                        "motion_sha256": motion_sha,
                        "skeleton_relpath": f"skeletons/{rig_id}.npz",
                        "skeleton_sha256": skeleton.sha256,
                        "fps_src": encoded.fps_src,
                        "fps_target": encoded.fps_target,
                        "T_src": encoded.metrics["T_src"],
                        "T_target": encoded.metrics["T_target"],
                        "J_phys": encoded.metrics["J_phys"],
                        "resample_mode": encoded.resample_mode,
                        "source_path": clip["source"]["path"],
                        "source_split_protocol": clip.get("split_protocol"),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                record = {
                    "prototype_version": PROTOTYPE_VERSION,
                    "clip_id": selected_clip.clip_id,
                    "rig_id": rig_id,
                    "source_family": clip.get("source", {}).get("family"),
                    "topology_family": clip.get("topology_family"),
                    "topology_distance_bucket": clip.get(
                        "topology_distance_bucket"
                    ),
                    "family_role": selected_clip.family_role,
                    "audit_role": selected_clip.audit_role,
                    "calibration_eligible": selected_clip.calibration_eligible,
                    "selection_origin": selected_clip.selection_origin,
                    "replaces_parent_clip_id": selected_clip.replaces_parent_clip_id,
                    "split": clip.get("split"),
                    "status": "reject",
                    "reason_codes": ["KTJD17_ENCODER_FAILURE"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                qa_records.append(record)
                manifest_records.append(_project_failure_manifest_record(record))
            if index % 10 == 0 or index == len(selected):
                print(f"[ktjd17-prototype] encoded {index}/{len(selected)}", flush=True)

        qa_records.sort(key=lambda item: item["clip_id"])
        manifest_records.sort(key=lambda item: item["clip_id"])
        _write_jsonl(staging / "qa/encoder_qa.jsonl", qa_records)
        _write_jsonl(staging / "manifests/clips.jsonl", manifest_records)
        _write_json(
            staging / "manifests/prototype_selection.json",
            {
                "prototype_version": PROTOTYPE_VERSION,
                "selection_authority": selection_authority,
                "selection_sha256": selection_sha256,
                "selection_counts": selection_counts,
                "selected_count": len(selected),
                "selected": selected_records,
                "held_data_used_for_calibration": False,
            },
        )
        _write_json(staging / "config/encoder_candidate.json", cfg.encoder.as_record())
        statuses = Counter(record["status"] for record in qa_records)
        family_statuses = Counter(
            (record["family_role"], record["status"]) for record in qa_records
        )
        calibration_coverage_gaps = {
            family: int(selection_counts[family]["calibration_shortage"])
            for family in (
                "human",
                "quadruped",
                "winged",
                "spider_crab",
                "dragon_or_deep_topology",
                "snake",
            )
            if int(selection_counts[family]["calibration_shortage"]) > 0
        }
        encoder_status = "pass" if statuses.get("reject", 0) == 0 else "fail"
        coverage_status = "pass" if not calibration_coverage_gaps else "incomplete"
        overall_status = (
            "fail"
            if encoder_status == "fail"
            else "pass" if coverage_status == "pass" else "incomplete"
        )
        summary = {
            "prototype_version": PROTOTYPE_VERSION,
            "generation_id": generation_id,
            "status": overall_status,
            "encoder_status": encoder_status,
            "train_calibration_coverage_status": coverage_status,
            "calibration_coverage_gaps": calibration_coverage_gaps,
            "selected_count": len(selected),
            "status_counts": dict(sorted(statuses.items())),
            "family_status_counts": {
                f"{family}:{status}": count
                for (family, status), count in sorted(family_statuses.items())
            },
            "skeleton_count": len(copied_skeletons),
            "selection_authority": selection_authority,
            "selection_sha256": selection_sha256,
            "selection_counts": selection_counts,
            "calibration_eligible_pass_count": sum(
                record["status"] == "pass" and record["calibration_eligible"]
                for record in qa_records
            ),
            "read_only_pass_count": sum(
                record["status"] == "pass" and not record["calibration_eligible"]
                for record in qa_records
            ),
            "human_claim_boundary": human_contract.provenance["claim_boundary"],
            "visual_qa_status": "pending",
            "full_conversion_authorized": False,
        }
        _write_json(staging / "qa/encoder_summary.json", summary)
        files_before_generation = _file_manifest(staging)
        generation = {
            "prototype_version": PROTOTYPE_VERSION,
            "generation_id": generation_id,
            "created_at_utc": _datetime.datetime.now(_datetime.UTC).isoformat(),
            "parent_manifest_root": str(cfg.manifest_root),
            "parent_canonical_generation": _load_json(
                cfg.manifest_root / "canonical_skeleton_generation.json"
            ),
            "source_plan_commit": "9181f5cccbad23e941bf94c2874daf36e7f288cf",
            "selection_authority": selection_authority,
            "selection_sha256": selection_sha256,
            "encoder_config": cfg.encoder.as_record(),
            "files": files_before_generation,
            "status": summary["status"],
            "visual_qa_status": "pending",
            "full_conversion_authorized": False,
        }
        _write_json(staging / "generation.json", generation)
        if final.exists():
            raise PrototypeBuildError(f"generation already exists: {final}")
        os.replace(staging, final)
        parent_fd = os.open(generations, os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        if cfg.overwrite_link:
            _replace_symlink(cfg.output_root / PROTOTYPE_LINK_NAME, final)
        return {
            **summary,
            "generation_root": str(final),
            "compatibility_link": str(cfg.output_root / PROTOTYPE_LINK_NAME),
        }
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
