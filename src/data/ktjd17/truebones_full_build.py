"""Frozen full-catalog Truebones to KTJD-17 conversion.

The producer consumes only source-safe rows from the pinned current-BTJD
inventory, original BVH rotation channels, the frozen KTJD-17 schema, the
66-rig reviewed forward audit, and its separately signed visual gate.  Legacy
BTJD-13 motion channels remain inventory witnesses and are never decoded here.
"""

from __future__ import annotations

import dataclasses
import datetime as _datetime
import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
import uuid
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .encoder import (
    EncodedMotion,
    encode_prepared_motion,
    load_skeleton,
    prepare_manifest_clip,
    write_npz_atomic,
)
from .freeze import verify_generation_file_closure
from .loader import load_motion_npz
from .source_parser import ParsedBvhMotion
from .truebones_fixed_rig import (
    ACTIVE_COND_SHA256,
    FULL_TRUEBONES_FORWARD_SPEC_VERSION,
    LEGACY_TRUEBONES_COND_SHA256,
    TRUEBONES_FULL_FORWARD_SPECS,
    load_conditioning_catalog,
)
from .truebones_forward_audit import (
    AUDIT_GENERATION_DIRECTORY,
    EXPECTED_CLIPS_SHA256,
    EXPECTED_FROZEN_SCHEMA_SHA256,
    EXPECTED_RIGS_SHA256,
    EXPECTED_SCOPE,
    EXPECTED_UNAVAILABLE_RIGS,
    FROZEN_SCHEMA_GENERATION_ID,
    PARENT_MANIFEST_GENERATION_ID,
    SOURCE_PLAN_COMMIT,
    _canonical_json,
    _load_pinned_jsonl,
    _upstream_rejection_record,
    _validate_parent_scope,
    _write_json,
    _write_jsonl,
    encoder_config_from_frozen_schema,
    verify_forward_audit_generation,
    verify_parent_manifest_files,
)
from .visual_qa import verify_visual_generation


FULL_BUILD_VERSION = "ktjd17-truebones-frozen-full-v1"
FULL_GENERATION_DIRECTORY = ".ktjd17_truebones_generations"
FULL_LINK_NAME = "ktjd17_truebones"
FORWARD_AUDIT_GENERATION_ID = "20260819T203306371942Z-8541b68c8480"
FORWARD_AUDIT_GENERATION_SHA256 = (
    "787313054cd4b75e370a9e0bb83e9fbc8a004b02a76a09a5b2248fd3c092aa9d"
)
FORWARD_AUDIT_SUMMARY_SHA256 = (
    "be9115da85fdbdc9ab234c290fdac54f2ab5d295e86d6550c9af788e8cdcb120"
)
FORWARD_AUDIT_RIG_QA_SHA256 = (
    "dadb4bdbe336cf352019030f29b7f8a6ce0a52bdc889c4729846753540ffaf66"
)
FORWARD_AUDIT_PARSE_QA_SHA256 = (
    "c2dd9a5014c57b8b973cf46fa1ef8b23b08254689a11a3fc47e348f3ba778c31"
)
VISUAL_GENERATION_ID = "20260819T203413394509Z-c8a431c08118"
VISUAL_GENERATION_SHA256 = (
    "f5a4c7199f5211925c459eb115a7fa4360ca6cfd17e15e676762dfab39b6c2be"
)
VISUAL_INDEX_SHA256 = (
    "f9dcc6d37bb034e12f94d9d33a5691b54e5995a39cde110781a0260d822584d2"
)
VISUAL_GATE_SHA256 = (
    "506b177ba2ec9df6bdcc479fbe9606a5d2acf9001dcd63580ace011e443f1bd8"
)
VISUAL_REVIEW_THREAD_ID = "01a01bc9-6b29-7873-86f2-ef38ec76f7b4"
VISUAL_RIG_SET_SHA256 = (
    "92b7aaeeb922867a68d65f15f7f7c728915a6f4531575df1a041d619dff86530"
)
VISUAL_CLIP_SET_SHA256 = (
    "ae176100c8c6dc86675e7dd0809df95313e44992e1e1518994ef273bd89d1d20"
)
FREEZE_GENERATION_SHA256 = (
    "0053d028e4cc8ec96ec3dafd841c8122ddab2c807f7519cf4293ff20cb3486b5"
)
FROZEN_STATS_SHA256 = (
    "dcde268e52a6c629475ab7529c666e202904df51729de7792c551f8a5615434b"
)
PARENT_PROTOTYPE_CANDIDATES_SHA256 = (
    "e03baf7561745f929b396c5d10cf0c761b16a15cf1605913addf8de3a4f800da"
)
COORDINATE_CONTRACT = (
    "right-handed; +Y is screen-up; +Z points out of the screen toward the viewer"
)
SPLITS = ("train", "val", "held_representative", "held_stress")


class TruebonesFullBuildError(RuntimeError):
    """The frozen full conversion cannot be trusted or published."""


@dataclasses.dataclass(frozen=True)
class FullBuildConfig:
    manifest_root: Path
    freeze_root: Path
    forward_audit_root: Path
    visual_root: Path
    visual_gate_path: Path
    output_root: Path
    active_cond_path: Path
    legacy_cond_path: Path
    update_link: bool = True

    def resolved(self) -> "FullBuildConfig":
        return dataclasses.replace(
            self,
            manifest_root=self.manifest_root.expanduser().resolve(),
            freeze_root=self.freeze_root.expanduser().resolve(),
            forward_audit_root=self.forward_audit_root.expanduser().resolve(),
            visual_root=self.visual_root.expanduser().resolve(),
            visual_gate_path=self.visual_gate_path.expanduser().resolve(),
            output_root=self.output_root.expanduser().absolute(),
            active_cond_path=self.active_cond_path.expanduser().resolve(),
            legacy_cond_path=self.legacy_cond_path.expanduser().resolve(),
        )


def default_full_build_config(repo_root: str | Path = ".") -> FullBuildConfig:
    root = Path(repo_root).expanduser().resolve()
    return FullBuildConfig(
        manifest_root=(
            root
            / "dataset/.ktjd17_manifest_generations"
            / PARENT_MANIFEST_GENERATION_ID
        ),
        freeze_root=(
            root
            / "dataset/.ktjd17_freeze_generations"
            / FROZEN_SCHEMA_GENERATION_ID
        ),
        forward_audit_root=(
            root / "dataset" / AUDIT_GENERATION_DIRECTORY / FORWARD_AUDIT_GENERATION_ID
        ),
        visual_root=(
            root
            / "dataset/ktjd17_truebones_forward_visual"
            / ".ktjd17_visual_qa_generations"
            / VISUAL_GENERATION_ID
        ),
        visual_gate_path=root / "dataset/ktjd17_truebones_forward_visual_gate.json",
        output_root=root / "dataset",
        active_cond_path=(
            root / "data/animo4d_L4TB_plus_human_v4b272neutral/cond.npy"
        ),
        legacy_cond_path=root / "data/anytop_truebones/cond.npy",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise TruebonesFullBuildError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TruebonesFullBuildError(f"JSON root is not an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise TruebonesFullBuildError(
                        f"{path}:{line_number}: blank JSONL row"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TruebonesFullBuildError(
                        f"{path}:{line_number}: row is not an object"
                    )
                records.append(value)
    except TruebonesFullBuildError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise TruebonesFullBuildError(f"cannot read JSONL {path}: {exc}") from exc
    return records


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


def _fsync_tree(root: Path) -> None:
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        _fsync_file(path)
    for path in sorted(
        (item for item in root.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        _fsync_directory(path)
    _fsync_directory(root)


def _file_manifest(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise TruebonesFullBuildError(
                f"symlink is forbidden inside full generation: {path}"
            )
        if path.is_file():
            relpath = path.relative_to(root).as_posix()
            if relpath == "generation.json":
                continue
            result[relpath] = {
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    return result


def _copy_regular_file(
    source: Path,
    target: Path,
    *,
    expected_sha256: str | None = None,
) -> str:
    if source.is_symlink() or not source.is_file():
        raise TruebonesFullBuildError(f"copy source is not a regular file: {source}")
    if target.exists() or target.is_symlink():
        raise TruebonesFullBuildError(f"copy target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    _fsync_file(target)
    observed = _sha256_file(target)
    if expected_sha256 is not None and observed != expected_sha256:
        raise TruebonesFullBuildError(
            f"copied file hash drifted for {source}: {observed} != {expected_sha256}"
        )
    return observed


def _replace_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() and not link.is_symlink():
        raise TruebonesFullBuildError(f"refusing to replace non-symlink {link}")
    if link.is_symlink() and not link.exists():
        raise TruebonesFullBuildError(f"refusing to replace broken symlink {link}")
    temporary = link.parent / f".{link.name}.{uuid.uuid4().hex}.tmp"
    os.symlink(os.path.relpath(target, start=link.parent), temporary)
    os.replace(temporary, link)
    _fsync_directory(link.parent)


def _write_lines(path: Path, values: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            if "\n" in value or "\r" in value:
                raise TruebonesFullBuildError(f"newline in split member {value!r}")
            handle.write(value + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _sorted_set_sha256(values: Sequence[str]) -> str:
    payload = ("\n".join(sorted(str(value) for value in values)) + "\n").encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _verify_parent_build_files(manifest_root: Path) -> dict[str, str]:
    """Pin every parent-manifest file named by the full-build authority."""
    verified = verify_parent_manifest_files(manifest_root)
    candidates = manifest_root / "prototype_candidates.json"
    if candidates.is_symlink() or not candidates.is_file():
        raise TruebonesFullBuildError(
            f"parent prototype candidates are not a regular file: {candidates}"
        )
    observed = _sha256_file(candidates)
    if observed != PARENT_PROTOTYPE_CANDIDATES_SHA256:
        raise TruebonesFullBuildError(
            "parent prototype-candidate hash drifted: "
            f"{observed} != {PARENT_PROTOTYPE_CANDIDATES_SHA256}"
        )
    verified["prototype_candidates.json"] = observed
    return verified


def _snapshot_regular_file(
    path: Path,
    *,
    expected_size: int | None = None,
    expected_mtime_ns: int | None = None,
) -> dict[str, Any]:
    """Hash one stable regular-file stream and bind it to its path identity."""
    target = path.expanduser().resolve(strict=False)
    try:
        path_stat = path.lstat()
    except OSError as exc:
        raise TruebonesFullBuildError(f"cannot stat source file {path}: {exc}") from exc
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
        raise TruebonesFullBuildError(f"source is not a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if (before.st_dev, before.st_ino) != (path_stat.st_dev, path_stat.st_ino):
                raise TruebonesFullBuildError(f"source identity raced before read: {path}")
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
            after = os.fstat(handle.fileno())
    except TruebonesFullBuildError:
        raise
    except OSError as exc:
        raise TruebonesFullBuildError(f"cannot hash source file {path}: {exc}") from exc
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise TruebonesFullBuildError(f"source changed while hashing: {path}")
    final_stat = path.lstat()
    if (
        stat.S_ISLNK(final_stat.st_mode)
        or (final_stat.st_dev, final_stat.st_ino, final_stat.st_size, final_stat.st_mtime_ns)
        != identity_after
    ):
        raise TruebonesFullBuildError(f"source path changed after hashing: {path}")
    if expected_size is not None and before.st_size != int(expected_size):
        raise TruebonesFullBuildError(
            f"source size drifted for {path}: {before.st_size} != {expected_size}"
        )
    if expected_mtime_ns is not None and before.st_mtime_ns != int(expected_mtime_ns):
        raise TruebonesFullBuildError(
            f"source mtime drifted for {path}: "
            f"{before.st_mtime_ns} != {expected_mtime_ns}"
        )
    return {
        "path": str(target),
        "sha256": digest.hexdigest(),
        "size_bytes": int(before.st_size),
        "mtime_ns": int(before.st_mtime_ns),
        "device": int(before.st_dev),
        "inode": int(before.st_ino),
    }


def _require_same_snapshot(
    expected: Mapping[str, Any], observed: Mapping[str, Any], *, label: str
) -> None:
    keys = ("path", "sha256", "size_bytes", "mtime_ns", "device", "inode")
    if any(expected.get(key) != observed.get(key) for key in keys):
        raise TruebonesFullBuildError(
            f"{label} changed during conversion: "
            f"before={dict(expected)}, after={dict(observed)}"
        )


def validate_visual_gate(
    *,
    gate_path: str | Path,
    visual_root: str | Path,
    forward_audit_root: str | Path,
) -> dict[str, Any]:
    gate_file = Path(gate_path).expanduser().resolve()
    if gate_file.is_symlink() or not gate_file.is_file():
        raise TruebonesFullBuildError(f"visual gate is not a regular file: {gate_file}")
    if _sha256_file(gate_file) != VISUAL_GATE_SHA256:
        raise TruebonesFullBuildError("visual gate hash drifted")
    gate = _load_json(gate_file)
    if gate.get("verdict") != "pass":
        raise TruebonesFullBuildError("visual gate did not pass")
    authorization = gate.get("authorization")
    if not isinstance(authorization, Mapping) or authorization.get(
        "full_source_safe_conversion"
    ) is not True:
        raise TruebonesFullBuildError("visual gate does not authorize full conversion")
    if gate.get("coordinate_contract") != COORDINATE_CONTRACT:
        raise TruebonesFullBuildError("visual gate coordinate contract drifted")
    primary = gate.get("primary_review")
    independent = gate.get("independent_review")
    if not isinstance(primary, Mapping) or primary.get("verdict") != "pass":
        raise TruebonesFullBuildError("primary visual review is not a pass")
    if not isinstance(independent, Mapping) or independent.get("verdict") != "pass":
        raise TruebonesFullBuildError("independent visual review is not a pass")
    if (
        independent.get("model") != "gpt-5.5"
        or independent.get("model_reasoning_effort") != "xhigh"
        or independent.get("thread_id") != VISUAL_REVIEW_THREAD_ID
    ):
        raise TruebonesFullBuildError("independent visual-review authority drifted")
    if primary.get("failures") != [] or independent.get("failures") != []:
        raise TruebonesFullBuildError("visual gate contains a failure")

    audit_root = Path(forward_audit_root).expanduser().resolve()
    audit_generation = verify_forward_audit_generation(audit_root)
    if audit_root.name != FORWARD_AUDIT_GENERATION_ID:
        raise TruebonesFullBuildError("forward-audit generation id drifted")
    audit_pins = gate.get("forward_audit_generation")
    if not isinstance(audit_pins, Mapping):
        raise TruebonesFullBuildError("visual gate lacks forward-audit pins")
    actual_audit_pins = {
        "generation_id": audit_root.name,
        "generation_json_sha256": _sha256_file(audit_root / "generation.json"),
        "audit_summary_sha256": _sha256_file(audit_root / "audit_summary.json"),
        "rig_audit_sha256": _sha256_file(audit_root / "qa/rig_audit.jsonl"),
        "source_safe_parse_sha256": _sha256_file(
            audit_root / "qa/source_safe_parse.jsonl"
        ),
    }
    expected_audit_pins = {
        "generation_id": FORWARD_AUDIT_GENERATION_ID,
        "generation_json_sha256": FORWARD_AUDIT_GENERATION_SHA256,
        "audit_summary_sha256": FORWARD_AUDIT_SUMMARY_SHA256,
        "rig_audit_sha256": FORWARD_AUDIT_RIG_QA_SHA256,
        "source_safe_parse_sha256": FORWARD_AUDIT_PARSE_QA_SHA256,
    }
    if dict(audit_pins) != expected_audit_pins or actual_audit_pins != expected_audit_pins:
        raise TruebonesFullBuildError("forward-audit visual-gate pins drifted")
    if audit_generation.get("status") != "numeric_pass_visual_pending":
        raise TruebonesFullBuildError("forward audit has unexpected status")

    visual = Path(visual_root).expanduser().resolve()
    visual_generation = verify_visual_generation(visual)
    if visual.name != VISUAL_GENERATION_ID:
        raise TruebonesFullBuildError("visual generation id drifted")
    visual_pins = gate.get("visual_generation")
    if not isinstance(visual_pins, Mapping):
        raise TruebonesFullBuildError("visual gate lacks visual-generation pins")
    visual_index_path = visual / "visual_qa_index.json"
    visual_index = _load_json(visual_index_path)
    clips = visual_index.get("clips")
    if not isinstance(clips, list) or len(clips) != EXPECTED_SCOPE["encodable_rig_count"]:
        raise TruebonesFullBuildError("visual index does not cover all available rigs")
    rig_ids = [str(record["rig_id"]) for record in clips]
    clip_ids = [str(record["clip_id"]) for record in clips]
    expected_visual_pins = {
        "generation_id": VISUAL_GENERATION_ID,
        "generation_json_sha256": VISUAL_GENERATION_SHA256,
        "visual_qa_index_sha256": VISUAL_INDEX_SHA256,
        "rig_count": 66,
        "clip_count": 66,
        "rig_set_sha256_newline_sorted": VISUAL_RIG_SET_SHA256,
        "clip_set_sha256_newline_sorted": VISUAL_CLIP_SET_SHA256,
    }
    actual_visual_pins = {
        "generation_id": visual.name,
        "generation_json_sha256": _sha256_file(visual / "generation.json"),
        "visual_qa_index_sha256": _sha256_file(visual_index_path),
        "rig_count": len(set(rig_ids)),
        "clip_count": len(set(clip_ids)),
        "rig_set_sha256_newline_sorted": _sorted_set_sha256(rig_ids),
        "clip_set_sha256_newline_sorted": _sorted_set_sha256(clip_ids),
    }
    if dict(visual_pins) != expected_visual_pins or actual_visual_pins != expected_visual_pins:
        raise TruebonesFullBuildError("visual-generation gate pins drifted")
    if visual_generation.get("prototype_generation_id") != FORWARD_AUDIT_GENERATION_ID:
        raise TruebonesFullBuildError("visual/audit generation relation drifted")
    if visual_index.get("coordinate_contract") != COORDINATE_CONTRACT:
        raise TruebonesFullBuildError("visual index coordinate contract drifted")
    return gate


def _verify_freeze(root: Path) -> dict[str, Any]:
    generation = verify_generation_file_closure(root)
    if root.name != FROZEN_SCHEMA_GENERATION_ID:
        raise TruebonesFullBuildError("freeze generation id drifted")
    if _sha256_file(root / "generation.json") != FREEZE_GENERATION_SHA256:
        raise TruebonesFullBuildError("freeze generation hash drifted")
    if generation.get("source_plan_commit") != SOURCE_PLAN_COMMIT:
        raise TruebonesFullBuildError("freeze source-plan commit drifted")
    if generation.get("freeze_authorized") is not True or generation.get(
        "full_build_may_start"
    ) is not True:
        raise TruebonesFullBuildError("freeze does not authorize a full build")
    if _sha256_file(root / "schema.json") != EXPECTED_FROZEN_SCHEMA_SHA256:
        raise TruebonesFullBuildError("frozen schema hash drifted")
    if _sha256_file(root / "stats/train_block_gains.npz") != FROZEN_STATS_SHA256:
        raise TruebonesFullBuildError("frozen train gains hash drifted")
    return generation


def _audit_skeleton_authority(
    audit_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    manifests = _load_jsonl(audit_root / "manifests/clips.jsonl")
    if len(manifests) != EXPECTED_SCOPE["encodable_rig_count"]:
        raise TruebonesFullBuildError("forward audit representative count drifted")
    by_rig: dict[str, dict[str, Any]] = {}
    representatives: dict[str, str] = {}
    for record in manifests:
        rig_id = str(record["rig_id"])
        clip_id = str(record["clip_id"])
        if rig_id in by_rig or clip_id in representatives:
            raise TruebonesFullBuildError("duplicate audit rig or representative clip")
        if record.get("status") != "accept":
            raise TruebonesFullBuildError(f"audit representative is not accepted: {clip_id}")
        by_rig[rig_id] = record
        representatives[clip_id] = rig_id
    if _sorted_set_sha256(list(by_rig)) != VISUAL_RIG_SET_SHA256:
        raise TruebonesFullBuildError("audit skeleton rig set differs from visual gate")
    if _sorted_set_sha256(list(representatives)) != VISUAL_CLIP_SET_SHA256:
        raise TruebonesFullBuildError(
            "audit representative clip set differs from visual gate"
        )
    parse_records = _load_jsonl(audit_root / "qa/source_safe_parse.jsonl")
    if (
        len(parse_records) != EXPECTED_SCOPE["source_safe_clip_count"]
        or any(record.get("status") != "pass" for record in parse_records)
    ):
        raise TruebonesFullBuildError("forward audit did not parse all source-safe clips")
    return by_rig, representatives


def reviewed_representative_clip_ids(
    forward_audit_root: str | Path,
) -> list[str]:
    """Load only the exact 66-clip representative authority reviewed visually."""
    root = Path(forward_audit_root).expanduser().resolve()
    if root.name != FORWARD_AUDIT_GENERATION_ID:
        raise TruebonesFullBuildError("forward-audit generation id drifted")
    verify_forward_audit_generation(root)
    if _sha256_file(root / "generation.json") != FORWARD_AUDIT_GENERATION_SHA256:
        raise TruebonesFullBuildError("forward-audit generation hash drifted")
    _, representatives = _audit_skeleton_authority(root)
    verify_forward_audit_generation(root)
    if _sha256_file(root / "generation.json") != FORWARD_AUDIT_GENERATION_SHA256:
        raise TruebonesFullBuildError(
            "forward-audit authority changed while loading representatives"
        )
    clip_ids = sorted(representatives)
    if (
        len(clip_ids) != EXPECTED_SCOPE["encodable_rig_count"]
        or _sorted_set_sha256(clip_ids) != VISUAL_CLIP_SET_SHA256
    ):
        raise TruebonesFullBuildError("reviewed representative scope drifted")
    return clip_ids


def _representative_regression(
    encoded: EncodedMotion,
    *,
    audit_motion_path: Path,
) -> None:
    reference = load_motion_npz(audit_motion_path, expected_fps_target=30.0)
    checks = {
        "motion": np.array_equal(reference["motion"], encoded.motion_float32),
        "heading_valid": np.array_equal(
            reference["heading_valid"], encoded.heading_valid
        ),
        "origin_xz": np.array_equal(reference["origin_xz"], encoded.origin_xz),
        "clip_id": reference["clip_id"] == encoded.clip_id,
        "rig_id": reference["rig_id"] == encoded.rig_id,
        "fps_target": float(reference["fps_target"]) == encoded.fps_target,
    }
    if not all(checks.values()):
        raise TruebonesFullBuildError(
            f"{encoded.clip_id}: reviewed representative regression failed: {checks}"
        )


def _published_provenance(
    encoded: EncodedMotion,
    *,
    rig_id: str,
    source_snapshot: Mapping[str, Any],
    rest_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    provenance = dict(encoded.provenance)
    provenance.pop("skeleton_path", None)
    provenance["skeleton_relpath"] = f"skeletons/{rig_id}.npz"
    provenance["skeleton_resolution"] = "generation_relative_relpath_plus_sha256"
    source = dict(provenance["source"])
    source["source_file_snapshot"] = dict(source_snapshot)
    source["rest_file_snapshot"] = dict(rest_snapshot)
    provenance["source"] = source
    provenance["visual_gate_sha256"] = VISUAL_GATE_SHA256
    provenance["forward_audit_generation_id"] = FORWARD_AUDIT_GENERATION_ID
    provenance["legacy_btjd_motion_channels_used"] = False
    return provenance


def _metric_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    names = sorted(
        {
            name
            for record in records
            if record.get("status") == "pass"
            for name, value in record.get("metrics", {}).items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value is not None
            and math.isfinite(float(value))
        }
    )
    result: dict[str, Any] = {}
    for name in names:
        values = np.asarray(
            [
                float(record["metrics"][name])
                for record in records
                if record.get("status") == "pass"
                and isinstance(record.get("metrics", {}).get(name), (int, float))
                and not isinstance(record.get("metrics", {}).get(name), bool)
                and math.isfinite(float(record["metrics"][name]))
            ],
            dtype=np.float64,
        )
        if values.size:
            result[name] = {
                "count": int(values.size),
                "min": float(np.min(values)),
                "median": float(np.median(values)),
                "p99": float(np.percentile(values, 99)),
                "max": float(np.max(values)),
                "mean": float(np.mean(values)),
            }
    return result


def summarize_strata(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped["all"].append(record)
        for field in (
            "split",
            "source_family",
            "topology_family",
            "topology_distance_bucket",
            "rig_id",
            "parent_inventory_status",
        ):
            grouped[f"{field}:{record.get(field)}"].append(record)
    return {
        key: {
            "count": len(values),
            "pass": sum(value.get("status") == "pass" for value in values),
            "fail": sum(value.get("status") != "pass" for value in values),
            "metrics": _metric_summary(values),
        }
        for key, values in sorted(grouped.items())
    }


def _source_authority_record(
    snapshots: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    entries = [
        {
            "path": path,
            "sha256": record["sha256"],
            "size_bytes": record["size_bytes"],
            "mtime_ns": record["mtime_ns"],
        }
        for path, record in sorted(snapshots.items())
    ]
    return {
        "file_count": len(entries),
        "entry_stream_sha256": hashlib.sha256(_canonical_json(entries)).hexdigest(),
        "entries": entries,
    }


def _verify_payload_reference_closure(
    observed_files: Mapping[str, Mapping[str, Any]],
    *,
    referenced_motions: set[str],
    referenced_skeletons: set[str],
) -> None:
    payload_motions = {
        relpath for relpath in observed_files if relpath.startswith("motions/")
    }
    if payload_motions != referenced_motions:
        raise TruebonesFullBuildError(
            "motion payload/reference closure failed: "
            f"orphan={sorted(payload_motions - referenced_motions)}, "
            f"missing={sorted(referenced_motions - payload_motions)}"
        )
    payload_skeletons = {
        relpath for relpath in observed_files if relpath.startswith("skeletons/")
    }
    if payload_skeletons != referenced_skeletons:
        raise TruebonesFullBuildError(
            "skeleton payload/reference closure failed: "
            f"orphan={sorted(payload_skeletons - referenced_skeletons)}, "
            f"missing={sorted(referenced_skeletons - payload_skeletons)}"
        )


def verify_full_generation(
    root: str | Path,
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Verify immutable closure and every public full-dataset reference."""
    generation_root = Path(root).expanduser().resolve()
    generation = _load_json(generation_root / "generation.json")
    if generation.get("generation_id") != generation_root.name:
        raise TruebonesFullBuildError("full generation id/path mismatch")
    if require_complete and generation.get("conversion_complete") is not True:
        raise TruebonesFullBuildError(
            "full conversion is incomplete; this generation is not a valid full dataset"
        )
    expected_files = generation.get("files")
    if not isinstance(expected_files, Mapping):
        raise TruebonesFullBuildError("full generation file manifest is absent")
    observed_files = _file_manifest(generation_root)
    if set(observed_files) != set(expected_files):
        raise TruebonesFullBuildError(
            "full generation file closure failed: "
            f"missing={sorted(set(expected_files) - set(observed_files))}, "
            f"extra={sorted(set(observed_files) - set(expected_files))}"
        )
    for relpath, metadata in expected_files.items():
        if observed_files[relpath] != dict(metadata):
            raise TruebonesFullBuildError(f"full generation hash/size drift: {relpath}")
    if _sha256_file(generation_root / "schema.json") != EXPECTED_FROZEN_SCHEMA_SHA256:
        raise TruebonesFullBuildError("published schema hash drifted")
    if (
        _sha256_file(generation_root / "stats/train_block_gains.npz")
        != FROZEN_STATS_SHA256
    ):
        raise TruebonesFullBuildError("published train gains hash drifted")
    if _sha256_file(generation_root / "evidence/visual_gate.json") != VISUAL_GATE_SHA256:
        raise TruebonesFullBuildError("published visual gate hash drifted")

    manifests = _load_jsonl(generation_root / "manifests/clips.jsonl")
    qa_records = _load_jsonl(generation_root / "qa/encoder_qa.jsonl")
    upstream = _load_jsonl(
        generation_root / "manifests/upstream_rejections.jsonl"
    )
    conversion = _load_jsonl(
        generation_root / "manifests/conversion_rejections.jsonl"
    )
    unavailable = _load_jsonl(
        generation_root / "manifests/unavailable_rigs.jsonl"
    )
    manifest_by_id = {str(record["clip_id"]): record for record in manifests}
    qa_by_id = {str(record["clip_id"]): record for record in qa_records}
    if len(manifest_by_id) != len(manifests) or len(qa_by_id) != len(qa_records):
        raise TruebonesFullBuildError("duplicate full-build clip id")
    if set(manifest_by_id) != set(qa_by_id):
        raise TruebonesFullBuildError("full manifest and QA scopes differ")
    upstream_ids = [str(record["clip_id"]) for record in upstream]
    conversion_ids = [str(record["clip_id"]) for record in conversion]
    if len(upstream_ids) != len(set(upstream_ids)) or len(conversion_ids) != len(
        set(conversion_ids)
    ):
        raise TruebonesFullBuildError("duplicate rejected clip id")
    rejected_ids = set(upstream_ids + conversion_ids)
    if set(upstream_ids) & set(conversion_ids):
        raise TruebonesFullBuildError("upstream and conversion rejects overlap")
    if set(manifest_by_id) & rejected_ids:
        raise TruebonesFullBuildError("accepted and rejected scopes overlap")
    if len(set(manifest_by_id) | rejected_ids) != EXPECTED_SCOPE["clip_count"]:
        raise TruebonesFullBuildError("full source scope does not close to 1070 clips")
    if len(upstream) != EXPECTED_SCOPE["upstream_reject_count"]:
        raise TruebonesFullBuildError("upstream reject count drifted")
    if len(unavailable) != len(EXPECTED_UNAVAILABLE_RIGS):
        raise TruebonesFullBuildError("unavailable rig count drifted")
    complete = generation.get("conversion_complete") is True
    if complete and (
        len(manifests) != EXPECTED_SCOPE["source_safe_clip_count"] or conversion
    ):
        raise TruebonesFullBuildError("complete generation does not encode all source-safe clips")
    if complete != (generation.get("full_conversion_authorized") is True):
        raise TruebonesFullBuildError("conversion-complete/authorization flags disagree")
    if complete and generation.get("status") != "numeric_pass_visual_regression_pending":
        raise TruebonesFullBuildError("complete generation has unexpected status")
    if not complete and generation.get("status") != "conversion_incomplete":
        raise TruebonesFullBuildError("incomplete generation has unexpected status")
    referenced_skeletons: dict[str, tuple[str, str]] = {}
    referenced_motions: set[str] = set()
    expected_fps = 30.0
    for clip_id, manifest in sorted(manifest_by_id.items()):
        qa = qa_by_id[clip_id]
        if manifest.get("status") != "accept" or qa.get("status") != "pass":
            raise TruebonesFullBuildError(f"accepted row did not pass: {clip_id}")
        motion_relpath = str(manifest["motion_relpath"])
        if motion_relpath != f"motions/{clip_id}.npz":
            raise TruebonesFullBuildError(
                f"non-canonical motion reference for {clip_id}: {motion_relpath}"
            )
        referenced_motions.add(motion_relpath)
        motion_path = generation_root / motion_relpath
        motion_sha = _sha256_file(motion_path)
        if (
            motion_sha != manifest.get("motion_sha256")
            or motion_sha != qa.get("motion_sha256")
            or motion_path.stat().st_size != int(qa.get("motion_size_bytes", -1))
        ):
            raise TruebonesFullBuildError(f"motion reference drifted: {clip_id}")
        payload = load_motion_npz(motion_path, expected_fps_target=expected_fps)
        if (
            payload["clip_id"] != clip_id
            or payload["rig_id"] != manifest["rig_id"]
            or payload["motion"].shape
            != (int(manifest["T_target"]), int(manifest["J_phys"]), 17)
        ):
            raise TruebonesFullBuildError(f"embedded motion identity drifted: {clip_id}")
        rig_id = str(manifest["rig_id"])
        skeleton_relpath = str(manifest["skeleton_relpath"])
        if skeleton_relpath != f"skeletons/{rig_id}.npz":
            raise TruebonesFullBuildError(
                f"non-canonical skeleton reference for {rig_id}: {skeleton_relpath}"
            )
        reference = (
            skeleton_relpath,
            str(manifest["skeleton_sha256"]),
        )
        if rig_id in referenced_skeletons and referenced_skeletons[rig_id] != reference:
            raise TruebonesFullBuildError(f"inconsistent skeleton reference: {rig_id}")
        referenced_skeletons[rig_id] = reference
    if complete and len(referenced_skeletons) != EXPECTED_SCOPE["encodable_rig_count"]:
        raise TruebonesFullBuildError("complete generation skeleton count drifted")
    for rig_id, (relpath, expected_sha) in sorted(referenced_skeletons.items()):
        skeleton_path = generation_root / relpath
        if _sha256_file(skeleton_path) != expected_sha:
            raise TruebonesFullBuildError(f"skeleton reference drifted: {rig_id}")
        skeleton = load_skeleton(skeleton_path)
        if skeleton.rig_id != rig_id:
            raise TruebonesFullBuildError(f"skeleton identity drifted: {rig_id}")

    referenced_skeleton_paths = {
        relpath for relpath, _expected_sha in referenced_skeletons.values()
    }
    _verify_payload_reference_closure(
        observed_files,
        referenced_motions=referenced_motions,
        referenced_skeletons=referenced_skeleton_paths,
    )

    split_union: set[str] = set()
    for split in SPLITS:
        path = generation_root / f"splits/holdout_splits_v1/{split}.txt"
        values = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        if values != sorted(values) or len(values) != len(set(values)):
            raise TruebonesFullBuildError(f"split is unsorted or duplicated: {split}")
        expected = sorted(
            clip_id
            for clip_id, record in manifest_by_id.items()
            if record["split"] == split
        )
        if values != expected or split_union & set(values):
            raise TruebonesFullBuildError(f"split membership drifted: {split}")
        split_union.update(values)
    if split_union != set(manifest_by_id):
        raise TruebonesFullBuildError("split files do not cover accepted clips")
    return generation


def run_truebones_full_build(config: FullBuildConfig) -> dict[str, Any]:
    cfg = config.resolved()
    parent_hashes_before = _verify_parent_build_files(cfg.manifest_root)
    freeze_generation = _verify_freeze(cfg.freeze_root)
    encoder = encoder_config_from_frozen_schema(cfg.freeze_root)
    gate = validate_visual_gate(
        gate_path=cfg.visual_gate_path,
        visual_root=cfg.visual_root,
        forward_audit_root=cfg.forward_audit_root,
    )
    audit_generation = verify_forward_audit_generation(cfg.forward_audit_root)
    audit_by_rig, representative_rigs = _audit_skeleton_authority(
        cfg.forward_audit_root
    )
    clips = _load_pinned_jsonl(
        cfg.manifest_root / "clips.jsonl", expected_sha256=EXPECTED_CLIPS_SHA256
    )
    rig_records = _load_pinned_jsonl(
        cfg.manifest_root / "rigs.jsonl", expected_sha256=EXPECTED_RIGS_SHA256
    )
    parent_hashes_after = _verify_parent_build_files(cfg.manifest_root)
    if parent_hashes_after != parent_hashes_before:
        raise TruebonesFullBuildError("parent manifest authority changed while loading")
    rigs = {str(record["rig_id"]): record for record in rig_records}
    if len(rigs) != len(rig_records):
        raise TruebonesFullBuildError("duplicate rig ids in parent manifest")
    truebones_rigs = {
        rig_id: record
        for rig_id, record in rigs.items()
        if record.get("source_family") == "truebones"
    }
    safe, rejected = _validate_parent_scope(clips, truebones_rigs)
    safe.sort(key=lambda record: str(record["clip_id"]))
    conditioning = load_conditioning_catalog(
        cfg.active_cond_path,
        expected_active_sha256=ACTIVE_COND_SHA256,
        legacy_path=cfg.legacy_cond_path,
    )
    if conditioning.legacy_sha256 != LEGACY_TRUEBONES_COND_SHA256:
        raise TruebonesFullBuildError("legacy conditioning hash drifted")

    generation_id = (
        _datetime.datetime.now(_datetime.UTC).strftime("%Y%m%dT%H%M%S%fZ")
        + "-"
        + uuid.uuid4().hex[:12]
    )
    generations = cfg.output_root / FULL_GENERATION_DIRECTORY
    generations.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{generation_id}.", dir=generations))
    final = generations / generation_id
    rest_cache: dict[str, ParsedBvhMotion] = {}
    source_snapshots: dict[str, dict[str, Any]] = {}
    rest_snapshots: dict[str, dict[str, Any]] = {}
    skeletons: dict[str, Any] = {}
    skeleton_hashes: dict[str, str] = {}
    manifest_records: list[dict[str, Any]] = []
    qa_records: list[dict[str, Any]] = []
    conversion_rejections: list[dict[str, Any]] = []
    representative_regression_count = 0
    try:
        for rig_id, audit_record in sorted(audit_by_rig.items()):
            source_relpath = str(audit_record["skeleton_relpath"])
            source_path = cfg.forward_audit_root / source_relpath
            target_relpath = f"skeletons/{rig_id}.npz"
            target_path = staging / target_relpath
            skeleton_sha = _copy_regular_file(
                source_path,
                target_path,
                expected_sha256=str(audit_record["skeleton_sha256"]),
            )
            skeleton = load_skeleton(target_path)
            if skeleton.rig_id != rig_id or skeleton.sha256 != skeleton_sha:
                raise TruebonesFullBuildError(f"reviewed skeleton identity drift: {rig_id}")
            skeletons[rig_id] = skeleton
            skeleton_hashes[rig_id] = skeleton_sha

        for index, clip in enumerate(safe, start=1):
            clip_id = str(clip["clip_id"])
            rig_id = str(clip["rig_id"])
            rig = truebones_rigs[rig_id]
            source_path = Path(str(clip["source"]["path"]))
            rest_path = Path(str(rig["rest_pose"]["source_path"]))
            motion_path: Path | None = None
            try:
                source_before = _snapshot_regular_file(
                    source_path,
                    expected_size=int(clip["source"]["file_size_bytes"]),
                    expected_mtime_ns=int(clip["source"]["mtime_ns"]),
                )
                rest_before = _snapshot_regular_file(rest_path)
                if str(source_path.resolve()) in source_snapshots:
                    _require_same_snapshot(
                        source_snapshots[str(source_path.resolve())],
                        source_before,
                        label=f"{clip_id} source authority",
                    )
                else:
                    source_snapshots[str(source_path.resolve())] = source_before
                if str(rest_path.resolve()) in rest_snapshots:
                    _require_same_snapshot(
                        rest_snapshots[str(rest_path.resolve())],
                        rest_before,
                        label=f"{rig_id} rest authority",
                    )
                else:
                    rest_snapshots[str(rest_path.resolve())] = rest_before

                prepared = prepare_manifest_clip(
                    clip,
                    rig,
                    skeletons[rig_id],
                    conditioning_catalog=conditioning,
                    rest_cache=rest_cache,
                    truebones_forward_specs=TRUEBONES_FULL_FORWARD_SPECS,
                )
                source_after = _snapshot_regular_file(
                    source_path,
                    expected_size=int(clip["source"]["file_size_bytes"]),
                    expected_mtime_ns=int(clip["source"]["mtime_ns"]),
                )
                rest_after = _snapshot_regular_file(rest_path)
                _require_same_snapshot(
                    source_before, source_after, label=f"{clip_id} source"
                )
                _require_same_snapshot(
                    rest_before, rest_after, label=f"{clip_id} rest"
                )
                if prepared.provenance.get("source_sha256") != source_before["sha256"]:
                    raise TruebonesFullBuildError(
                        f"{clip_id}: parsed source hash differs from stable source stream"
                    )
                if (
                    prepared.provenance.get("source_rest_sha256")
                    != rest_before["sha256"]
                ):
                    raise TruebonesFullBuildError(
                        f"{clip_id}: parsed rest hash differs from stable rest stream"
                    )
                encoded = encode_prepared_motion(prepared, skeletons[rig_id], encoder)
                if clip_id in representative_rigs:
                    audit_record = audit_by_rig[rig_id]
                    _representative_regression(
                        encoded,
                        audit_motion_path=(
                            cfg.forward_audit_root / str(audit_record["motion_relpath"])
                        ),
                    )
                    representative_regression_count += 1
                motion_relpath = f"motions/{clip_id}.npz"
                motion_path = staging / motion_relpath
                motion_sha = write_npz_atomic(
                    motion_path, encoded.artifact_payload()
                )
                provenance = _published_provenance(
                    encoded,
                    rig_id=rig_id,
                    source_snapshot=source_before,
                    rest_snapshot=rest_before,
                )
                common = {
                    "full_build_version": FULL_BUILD_VERSION,
                    "clip_id": clip_id,
                    "rig_id": rig_id,
                    "source_family": "truebones",
                    "topology_family": clip["topology_family"],
                    "topology_distance_bucket": clip["topology_distance_bucket"],
                    "family_role": rig_id,
                    "audit_role": "full_source_safe_conversion",
                    "calibration_eligible": False,
                    "selection_origin": "pinned_source_safe_manifest",
                    "replaces_parent_clip_id": None,
                    "split": clip["split"],
                    "parent_inventory_status": clip["status"],
                    "status": "pass",
                    "reason_codes": [],
                    "motion_relpath": motion_relpath,
                    "motion_sha256": motion_sha,
                    "motion_size_bytes": motion_path.stat().st_size,
                    "skeleton_relpath": f"skeletons/{rig_id}.npz",
                    "skeleton_sha256": skeleton_hashes[rig_id],
                    "metrics": encoded.metrics,
                    "provenance": provenance,
                    "reviewed_representative_regression": clip_id in representative_rigs,
                }
                manifest_record = {
                    "clip_id": clip_id,
                    "rig_id": rig_id,
                    "source_family": "truebones",
                    "topology_family": clip["topology_family"],
                    "topology_distance_bucket": clip["topology_distance_bucket"],
                    "family_role": rig_id,
                    "audit_role": "full_source_safe_conversion",
                    "calibration_eligible": False,
                    "selection_origin": "pinned_source_safe_manifest",
                    "replaces_parent_clip_id": None,
                    "split": clip["split"],
                    "parent_inventory_status": clip["status"],
                    "status": "accept",
                    "reason_codes": [],
                    "motion_relpath": motion_relpath,
                    "motion_sha256": motion_sha,
                    "skeleton_relpath": f"skeletons/{rig_id}.npz",
                    "skeleton_sha256": skeleton_hashes[rig_id],
                    "fps_src": encoded.fps_src,
                    "fps_target": encoded.fps_target,
                    "T_src": encoded.metrics["T_src"],
                    "T_target": encoded.metrics["T_target"],
                    "J_phys": encoded.metrics["J_phys"],
                    "resample_mode": encoded.resample_mode,
                    "source_path": clip["source"]["path"],
                    "source_sha256": source_before["sha256"],
                    "source_rest_path": str(rest_path.resolve()),
                    "source_rest_sha256": rest_before["sha256"],
                    "source_frame_slice": list(clip["source"]["slice_frames"]),
                    "source_split_protocol": clip["split_protocol"],
                    "rotation_authority": (
                        "original_bvh_declared_rotation_channels_only"
                    ),
                    "legacy_btjd_motion_channels_used": False,
                }
                qa_records.append(common)
                manifest_records.append(manifest_record)
            except Exception as exc:  # noqa: BLE001
                if motion_path is not None and motion_path.exists():
                    motion_path.unlink()
                conversion_rejections.append(
                    {
                        "full_build_version": FULL_BUILD_VERSION,
                        "clip_id": clip_id,
                        "rig_id": rig_id,
                        "split": clip.get("split"),
                        "topology_family": clip.get("topology_family"),
                        "topology_distance_bucket": clip.get(
                            "topology_distance_bucket"
                        ),
                        "parent_inventory_status": clip.get("status"),
                        "status": "reject",
                        "reason_codes": ["KTJD17_FROZEN_FULL_ENCODER_FAILURE"],
                        "conversion_attempted": True,
                        "legacy_btjd_fallback_allowed": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            if index % 25 == 0 or index == len(safe):
                print(
                    f"[ktjd17-full] processed {index}/{len(safe)}; "
                    f"accepted={len(manifest_records)} rejected={len(conversion_rejections)}",
                    flush=True,
                )

        for path, expected in sorted(source_snapshots.items()):
            observed = _snapshot_regular_file(Path(path))
            _require_same_snapshot(expected, observed, label=f"final source {path}")
        for path, expected in sorted(rest_snapshots.items()):
            observed = _snapshot_regular_file(Path(path))
            _require_same_snapshot(expected, observed, label=f"final rest {path}")
        verify_forward_audit_generation(cfg.forward_audit_root)
        if (
            _sha256_file(cfg.forward_audit_root / "generation.json")
            != FORWARD_AUDIT_GENERATION_SHA256
        ):
            raise TruebonesFullBuildError("forward-audit authority changed during build")
        parent_hashes_final = _verify_parent_build_files(cfg.manifest_root)
        if parent_hashes_final != parent_hashes_before:
            raise TruebonesFullBuildError("parent manifest authority changed during build")
        if representative_regression_count != EXPECTED_SCOPE["encodable_rig_count"]:
            raise TruebonesFullBuildError(
                "not every reviewed representative passed array-level regression"
            )

        manifest_records.sort(key=lambda record: record["clip_id"])
        qa_records.sort(key=lambda record: record["clip_id"])
        conversion_rejections.sort(key=lambda record: record["clip_id"])
        upstream_rejections = sorted(
            (_upstream_rejection_record(record) for record in rejected),
            key=lambda record: record["clip_id"],
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
        split_members = {
            split: sorted(
                record["clip_id"]
                for record in manifest_records
                if record["split"] == split
            )
            for split in SPLITS
        }
        if sum(len(values) for values in split_members.values()) != len(
            manifest_records
        ):
            raise TruebonesFullBuildError("accepted split membership does not close")

        selection_counts = {
            split: {"selected": len(split_members[split])} for split in SPLITS
        }
        selection_authority = {
            "selection_kind": "full_source_safe_conversion",
            "parent_manifest_root": str(cfg.manifest_root),
            "parent_clips_jsonl_sha256": EXPECTED_CLIPS_SHA256,
            "parent_rigs_jsonl_sha256": EXPECTED_RIGS_SHA256,
            "parent_prototype_candidates_sha256": (
                PARENT_PROTOTYPE_CANDIDATES_SHA256
            ),
            "frozen_schema_generation_id": FROZEN_SCHEMA_GENERATION_ID,
            "forward_audit_generation_id": FORWARD_AUDIT_GENERATION_ID,
            "visual_gate_sha256": VISUAL_GATE_SHA256,
            "legacy_btjd_motion_channels_used": False,
        }
        selected_records = [
            {
                "clip_id": record["clip_id"],
                "rig_id": record["rig_id"],
                "split": record["split"],
            }
            for record in manifest_records
        ]
        selection_core = {
            "selection_authority": selection_authority,
            "selection_counts": selection_counts,
            "selected": selected_records,
        }
        selection_sha = hashlib.sha256(_canonical_json(selection_core)).hexdigest()
        complete = not conversion_rejections and len(manifest_records) == EXPECTED_SCOPE[
            "source_safe_clip_count"
        ]
        status = (
            "numeric_pass_visual_regression_pending"
            if complete
            else "conversion_incomplete"
        )

        _write_jsonl(staging / "manifests/clips.jsonl", manifest_records)
        _write_jsonl(staging / "qa/encoder_qa.jsonl", qa_records)
        _write_jsonl(
            staging / "manifests/upstream_rejections.jsonl", upstream_rejections
        )
        _write_jsonl(
            staging / "manifests/conversion_rejections.jsonl",
            conversion_rejections,
        )
        _write_jsonl(
            staging / "manifests/unavailable_rigs.jsonl", unavailable_records
        )
        selection_payload = {
            "full_build_version": FULL_BUILD_VERSION,
            **selection_core,
            "selection_sha256": selection_sha,
            "selected_count": len(selected_records),
            "held_data_used_for_calibration": False,
            "frozen_calibration_updated": False,
        }
        _write_json(
            staging / "manifests/prototype_selection.json", selection_payload
        )
        _write_json(staging / "manifests/full_selection.json", selection_payload)
        for split, values in split_members.items():
            _write_lines(
                staging / f"splits/holdout_splits_v1/{split}.txt", values
            )
        _write_json(staging / "config/encoder_frozen.json", encoder.as_record())
        _write_json(staging / "config/encoder_candidate.json", encoder.as_record())
        _copy_regular_file(
            cfg.freeze_root / "schema.json",
            staging / "schema.json",
            expected_sha256=EXPECTED_FROZEN_SCHEMA_SHA256,
        )
        _copy_regular_file(
            cfg.freeze_root / "stats/train_block_gains.npz",
            staging / "stats/train_block_gains.npz",
            expected_sha256=FROZEN_STATS_SHA256,
        )
        _copy_regular_file(
            cfg.visual_gate_path,
            staging / "evidence/visual_gate.json",
            expected_sha256=VISUAL_GATE_SHA256,
        )
        _write_json(
            staging / "qa/stratified_encoder_metrics.json",
            {
                "full_build_version": FULL_BUILD_VERSION,
                "status": "pass" if complete else "fail",
                "stratified": summarize_strata(qa_records),
            },
        )
        source_authority = _source_authority_record(source_snapshots)
        rest_authority = _source_authority_record(rest_snapshots)
        _write_json(
            staging / "qa/source_file_authority.json",
            {
                "source_bvh": source_authority,
                "rest_bvh": rest_authority,
                "stable_pre_post_hash_required": True,
                "manifest_size_and_mtime_required": True,
            },
        )
        inventory = {
            "full_build_version": FULL_BUILD_VERSION,
            "status": "pass" if complete else "fail",
            "parent_truebones_clip_count": EXPECTED_SCOPE["clip_count"],
            "parent_truebones_rig_count": EXPECTED_SCOPE["rig_count"],
            "source_safe_clip_count": EXPECTED_SCOPE["source_safe_clip_count"],
            "accepted_clip_count": len(manifest_records),
            "upstream_reject_count": len(upstream_rejections),
            "conversion_reject_count": len(conversion_rejections),
            "available_rig_count": len(skeletons),
            "unavailable_rig_count": len(unavailable_records),
            "split_counts_accepted": {
                split: len(values) for split, values in split_members.items()
            },
            "topology_family_counts_accepted": dict(
                sorted(Counter(record["topology_family"] for record in manifest_records).items())
            ),
            "topology_distance_bucket_counts_accepted": dict(
                sorted(
                    Counter(
                        record["topology_distance_bucket"]
                        for record in manifest_records
                    ).items()
                )
            ),
            "parent_inventory_status_counts_accepted": dict(
                sorted(
                    Counter(
                        record["parent_inventory_status"]
                        for record in manifest_records
                    ).items()
                )
            ),
            "total_target_frames": sum(
                int(record["T_target"]) for record in manifest_records
            ),
            "max_T_target": max(
                (int(record["T_target"]) for record in manifest_records), default=0
            ),
            "max_J_phys": max(
                (int(record["J_phys"]) for record in manifest_records), default=0
            ),
            "unique_source_bvh_count": source_authority["file_count"],
            "unique_rest_bvh_count": rest_authority["file_count"],
            "representative_array_regression_pass_count": (
                representative_regression_count
            ),
        }
        _write_json(staging / "qa/fresh_inventory.json", inventory)
        summary = {
            "full_build_version": FULL_BUILD_VERSION,
            "generation_id": generation_id,
            "status": status,
            "selected_count": len(manifest_records),
            "status_counts": {
                "pass": len(qa_records),
                "reject": len(conversion_rejections),
            },
            "skeleton_count": len(skeletons),
            "selection_authority": selection_authority,
            "selection_sha256": selection_sha,
            "selection_counts": selection_counts,
            "calibration_eligible_pass_count": 0,
            "read_only_pass_count": len(qa_records),
            "visual_qa_status": "prebuild_66_rig_gate_pass_postbuild_regression_pending",
            "conversion_complete": complete,
            "full_conversion_authorized": bool(complete),
            "ready_for_training": False,
        }
        _write_json(staging / "qa/encoder_summary.json", summary)
        producer_dir = Path(__file__).resolve().parent
        producer_files = {
            path.name: _sha256_file(path)
            for path in (
                Path(__file__).resolve(),
                producer_dir / "encoder.py",
                producer_dir / "codec.py",
                producer_dir / "source_parser.py",
                producer_dir / "truebones_fixed_rig.py",
            )
        }
        files = _file_manifest(staging)
        generation = {
            "full_build_version": FULL_BUILD_VERSION,
            "generation_id": generation_id,
            "created_at_utc": _datetime.datetime.now(_datetime.UTC).isoformat(),
            "status": status,
            "source_plan_commit": SOURCE_PLAN_COMMIT,
            "parent_manifest_generation_id": PARENT_MANIFEST_GENERATION_ID,
            "parent_manifest_hashes": parent_hashes_before,
            "freeze_generation_id": freeze_generation["generation_id"],
            "freeze_generation_sha256": FREEZE_GENERATION_SHA256,
            "forward_audit_generation_id": audit_generation["generation_id"],
            "forward_audit_generation_sha256": FORWARD_AUDIT_GENERATION_SHA256,
            "visual_generation_id": VISUAL_GENERATION_ID,
            "visual_gate_sha256": VISUAL_GATE_SHA256,
            "visual_review_thread_id": VISUAL_REVIEW_THREAD_ID,
            "forward_spec_version": FULL_TRUEBONES_FORWARD_SPEC_VERSION,
            "encoder_config": encoder.as_record(),
            "selection_sha256": selection_sha,
            "coordinate_contract": COORDINATE_CONTRACT,
            "rotation_authority": "original_bvh_declared_rotation_channels_only",
            "legacy_btjd_motion_channels_used": False,
            "producer_file_sha256": producer_files,
            "source_authority_stream_sha256": source_authority[
                "entry_stream_sha256"
            ],
            "rest_authority_stream_sha256": rest_authority[
                "entry_stream_sha256"
            ],
            "scope": inventory,
            "files": files,
            "conversion_complete": complete,
            "full_conversion_authorized": bool(complete),
            "postbuild_fixed_qa_complete": False,
            "postbuild_visual_regression_complete": False,
            "ready_for_training": False,
        }
        _write_json(staging / "generation.json", generation)
        _fsync_tree(staging)
        if final.exists():
            raise TruebonesFullBuildError(f"full generation already exists: {final}")
        os.replace(staging, final)
        _fsync_directory(generations)
        verify_full_generation(final, require_complete=complete)
        link_updated = False
        if cfg.update_link and complete:
            _replace_symlink(cfg.output_root / FULL_LINK_NAME, final)
            link_updated = True
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {
        "status": status,
        "generation_id": generation_id,
        "generation_root": str(final),
        "compatibility_link": str(cfg.output_root / FULL_LINK_NAME),
        "compatibility_link_updated": link_updated,
        "accepted_clip_count": len(manifest_records),
        "upstream_reject_count": len(rejected),
        "conversion_reject_count": len(conversion_rejections),
        "available_rig_count": len(skeletons),
        "representative_array_regression_pass_count": (
            representative_regression_count
        ),
        "conversion_complete": complete,
        "ready_for_training": False,
    }
