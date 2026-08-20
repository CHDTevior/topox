"""Fail-closed T02 raw-source inventory for the local BTJD-13 corpus.

The inventory proves where a rotation channel comes from; it does not decode
Euler/6D samples, perform source FK, choose heading polarity, or bless a
source-to-canonical transform.  Those later gates remain explicit ``review``
reasons rather than being silently inferred from legacy AnyTop13 values.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import datetime as _datetime
import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .bvh_inventory import BvhHeader, BvhInventoryError, parse_bvh_header


INVENTORY_VERSION = "ktjd17-raw-inventory-v1"
PROTOTYPE_FAMILIES = (
    "human",
    "quadruped",
    "winged",
    "snake",
    "spider_crab",
    "dragon_or_deep_topology",
)
PROTOTYPE_MIN_TRAIN_CLIPS = 30
OUTPUT_FILENAMES = (
    "clips.jsonl",
    "rigs.jsonl",
    "inventory_summary.json",
    "inventory_reason_codes.json",
    "prototype_candidates.json",
    "prototype_gaps.jsonl",
)
TRANSACTION_FILENAME = "inventory_generation.json"
GENERATION_DIRECTORY_NAME = ".ktjd17_manifest_generations"


REASON_CODES: dict[str, dict[str, str]] = {
    "BTJD_SHAPE_INVALID": {
        "severity": "reject",
        "description": "Current payload is not a numeric [T,J,13] NPY header matching cond J.",
    },
    "SOURCE_FILE_MISSING": {
        "severity": "reject",
        "description": "The declared raw rotation authority is absent on the live filesystem.",
    },
    "SOURCE_HEADER_INVALID": {
        "severity": "reject",
        "description": "The source hierarchy/timing or MotionStreamer NPY header is malformed.",
    },
    "SOURCE_LAYOUT_DRIFT": {
        "severity": "reject",
        "description": "A clip's retained-joint hierarchy/rotation provenance differs from its rig evidence layout.",
    },
    "SOURCE_NONRETAINED_LAYOUT_VARIANT": {
        "severity": "review",
        "description": "The full source hierarchy differs, but every retained joint remaps with the same binary rotation provenance; T03 must audit the variant explicitly.",
    },
    "SOURCE_FRAME_MAPPING_MISMATCH": {
        "severity": "reject",
        "description": "Source frames cannot reproduce the current BTJD clip length under the documented preprocessing rule.",
    },
    "RAW_SOURCE_SEQUENCE_SPLIT_OVERLAP": {
        "severity": "reject",
        "description": "One raw temporal source contributes clips to more than one frozen split; every affected slice is excluded until a source-grouped split is approved.",
    },
    "JOINT_MAP_MISSING": {
        "severity": "reject",
        "description": "At least one current physical joint has no exact source-hierarchy/schema name match.",
    },
    "JOINT_MAP_AMBIGUOUS": {
        "severity": "reject",
        "description": "A current physical joint maps to more than one source joint.",
    },
    "CURRENT_PARENT_NOT_SOURCE_ANCESTOR": {
        "severity": "reject",
        "description": "A current parent-child edge is not an ancestor relation in the source hierarchy.",
    },
    "ROTATION_PROVENANCE_INVALID": {
        "severity": "reject",
        "description": "A retained joint is neither source animated_dof nor source-proven fixed_dof.",
    },
    "NUMERIC_PAYLOAD_VALIDATION_DEFERRED_T03": {
        "severity": "review",
        "description": "T02 reads hierarchy/header evidence only; exhaustive finite decode and source-FK reproduction are T03 gates.",
    },
    "SOURCE_NUMERIC_PARSE_INVALID": {
        "severity": "reject",
        "description": "T03 could not decode the declared source payload into finite float64 transforms under its source-specific numeric contract.",
    },
    "SOURCE_FK_REPRODUCTION_FAILED": {
        "severity": "reject",
        "description": "T03 source-parser FK reproduction exceeded the declared source-family threshold; the clip is blocked before KTJD encoding.",
    },
    "CANONICAL_TRANSFORM_PROVENANCE_INVALID": {
        "severity": "reject",
        "description": "T04 cannot prove one fixed native per-rig source-to-canonical transform and rest frame; no lossless KTJD skeleton or motion may be accepted.",
    },
    "CANONICAL_SKELETON_DERIVATION_FAILED": {
        "severity": "reject",
        "description": "T04 could not derive a finite, mutually consistent canonical rest skeleton under the fixed algebraic gates.",
    },
    "HUMAN_FIXED_REST_UNRESOLVED": {
        "severity": "reject",
        "description": "MotionStreamer272 omits SMPL shape, so the neutral SMPL rest is a non-encodable review candidate until a fixed-rig shape/rest policy is approved.",
    },
    "HEADING_PAYLOAD_UNREVIEWED": {
        "severity": "review",
        "description": "Carrier, local forward vector, and polarity still require train-only numeric plus perspective visual review.",
    },
    "SOURCE_TO_CANONICAL_UNREVIEWED": {
        "severity": "review",
        "description": "A single numeric per-rig C/alpha/o transform has not yet been rederived and audited.",
    },
    "PZ_RAW_GAME_BVH_NOT_LOCAL": {
        "severity": "review",
        "description": "Only processed full-frame Planet Zoo BVHs are local; the native MANIS-exported raw BVHs are absent.",
    },
    "PZ_SOURCE_HAS_PER_CLIP_CANONICALIZATION": {
        "severity": "review",
        "description": "The local Planet Zoo BVH lineage documents per-action initial-yaw canonicalization, which must not be mistaken for a fixed per-rig KTJD transform.",
    },
    "HUMAN_CURRENT_BRIDGE_USES_PER_CLIP_ALIGNMENT": {
        "severity": "review",
        "description": "The current BTJD human builder uses per-clip Kabsch alignment; KTJD must rebuild from MotionStreamer272 with a fixed rig transform.",
    },
    "REST_SOURCE_REQUIRES_RAW_TPOSE_RECOVERY": {
        "severity": "review",
        "description": "The local source provides derived hierarchy offsets/cond rest metadata but not the original explicit T-pose file.",
    },
    "REST_FRAME_FALLBACK_NOT_EXPLICIT": {
        "severity": "review",
        "description": "The Truebones rig has no named T-pose; legacy preprocessing selected an idle/first clip as rest evidence.",
    },
    "SOURCE_ROOT_WRAPPER_DROPPED": {
        "severity": "review",
        "description": "Current physical root is a descendant of a removed source wrapper; T03 must compose its transform exactly.",
    },
    "JOINT_MAP_SKIPS_SOURCE_JOINTS": {
        "severity": "review",
        "description": "At least one retained edge skips source joints; T03 must show that the reduced physical tree remains a valid independent FK audit.",
    },
    "SOURCE_UNIT_TO_METER_UNKNOWN": {
        "severity": "info",
        "description": "Source units are retained as named native/canonical units; no meter claim is made.",
    },
    "PROTOTYPE_TRAIN_SHORTAGE": {
        "severity": "gap",
        "description": "Fewer than 30 rotation-proven clips from this prototype family are eligible in the frozen train split.",
    },
    "EXACT_DRAGON_NOT_TRAIN_ELIGIBLE": {
        "severity": "gap",
        "description": "The exact Truebones Dragon rig is frozen held-stress; train calibration must use a declared deep-topology substitute or revise the protocol explicitly.",
    },
}


class InventoryError(RuntimeError):
    """Global inventory integrity failure; no output should be committed."""


@dataclasses.dataclass(frozen=True)
class InventoryConfig:
    dataset_root: Path
    split_root: Path
    pz_bvh_root: Path
    truebones_raw_root: Path
    human272_root: Path
    output_root: Path
    human_builder_path: Path
    smpl_neutral_model_path: Path
    planetzoo_lineage_path: Path
    workers: int = 16
    overwrite: bool = False
    prototype_min_train_clips: int = PROTOTYPE_MIN_TRAIN_CLIPS

    def resolved(self) -> "InventoryConfig":
        values = dataclasses.asdict(self)
        for key in (
            "dataset_root",
            "split_root",
            "pz_bvh_root",
            "truebones_raw_root",
            "human272_root",
            "human_builder_path",
            "smpl_neutral_model_path",
            "planetzoo_lineage_path",
        ):
            values[key] = Path(values[key]).expanduser().resolve()
        # Keep the leaf identity of output_root: it is an atomic symlink pointer
        # to an immutable generation directory.  Calling resolve() on the leaf
        # would follow the old generation and make publication non-atomic.
        output_root = Path(values["output_root"]).expanduser()
        if not output_root.is_absolute():
            output_root = Path.cwd() / output_root
        values["output_root"] = output_root.parent.resolve() / output_root.name
        return InventoryConfig(**values)


@dataclasses.dataclass(frozen=True)
class NpyHeader:
    path: str
    shape: tuple[int, ...]
    dtype: str
    file_size_bytes: int
    mtime_ns: int
    error: str | None = None


@dataclasses.dataclass(frozen=True)
class BvhCompact:
    path: str
    frames: int | None
    frame_time: float | None
    fps_src: float | None
    source_joint_count: int | None
    source_channel_count: int | None
    rotation_layout_sha256: str | None
    rest_layout_sha256: str | None
    file_size_bytes: int | None
    mtime_ns: int | None
    error: str | None = None


@dataclasses.dataclass
class ClipDescriptor:
    clip_id: str
    rig_id: str
    btjd: NpyHeader
    split: str
    topology_family: str
    topology_distance_bucket: str
    source_family: str
    source_kind: str
    source_path: str
    source: BvhCompact | NpyHeader | None
    source_slice: tuple[int, int] | None = None
    source_frame_mapping: str | None = None
    source_sequence_splits: tuple[str, ...] = ()
    clip_reason_codes: list[str] = dataclasses.field(default_factory=list)
    diagnostics: list[str] = dataclasses.field(default_factory=list)
    prototype_candidate: bool = False


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _add_code(codes: list[str], code: str) -> None:
    if code not in REASON_CODES:
        raise InventoryError(f"unknown reason code {code!r}")
    if code not in codes:
        codes.append(code)


def _status_from_codes(codes: Iterable[str]) -> str:
    severities = {REASON_CODES[code]["severity"] for code in codes}
    if "reject" in severities:
        return "reject"
    if "review" in severities:
        return "review"
    return "accept"


def _tree_depth(parents: Sequence[int]) -> int:
    if not parents or int(parents[0]) != -1:
        raise InventoryError("physical parent tree must start with -1")
    depths = [0] * len(parents)
    for index in range(1, len(parents)):
        parent = int(parents[index])
        if not 0 <= parent < index:
            raise InventoryError(
                f"physical parent tree violates parent-before-child at {index}: {parent}"
            )
        depths[index] = depths[parent] + 1
    return max(depths)


_WINGED_TERMS = {
    "bat", "bird", "buzzard", "chicken", "eagle", "flamingo", "ostrich",
    "parrot", "pigeon", "pteranodon", "tukan", "peafowl", "penguin",
    "crane", "swan", "duck", "goose",
}
_SPIDER_CRAB_TERMS = {"spider", "spiderg", "crab", "hermitcrab", "scorpion"}
_SNAKE_TERMS = {"anaconda", "kingcobra", "cobra", "snake"}
_QUADRUPED_TERMS = {
    "aardvark", "alpaca", "alligator", "armadillo", "bear", "beaver",
    "bison", "buffalo", "camel", "caracal", "cat", "cheetah", "cougar",
    "coyote", "crocodile", "deer", "dog", "elephant", "fossa", "fox",
    "gazelle", "goat", "hamster", "hippopotamus", "horse", "hound",
    "hyena", "jaguar", "leopard", "leapord", "lion", "llama", "lynx",
    "mammoth", "moose", "otter", "panda", "pig", "polarbear", "porcupine",
    "puppy", "quokka", "rabbit", "raccoon", "rat", "raindeer", "reindeer",
    "rhino", "sandcat", "seal", "skunk", "tiger", "wolf", "wolverine",
    "wombat", "zebra",
}


def classify_topology_family(rig_id: str, tree_depth: int) -> str:
    """Deterministic, explicit prototype taxonomy; no learned/legacy labels."""
    normalized = re.sub(r"[^a-z0-9]+", "", rig_id.lower().removeprefix("pz_"))
    if rig_id == "HML3D_Human":
        return "human"
    if any(term in normalized for term in _SNAKE_TERMS):
        return "snake"
    if any(term in normalized for term in _SPIDER_CRAB_TERMS):
        return "spider_crab"
    if any(term in normalized for term in _WINGED_TERMS):
        return "winged"
    if rig_id == "Dragon" or tree_depth >= 15:
        return "dragon_or_deep_topology"
    if any(term in normalized for term in _QUADRUPED_TERMS):
        return "quadruped"
    return "other"


def _topology_bucket(split: str) -> str:
    return {
        "train": "train_seen_topology",
        "val": "val_seen_topology",
        "held_representative": "held_representative_topology",
        "held_stress": "held_stress_topology",
    }[split]


def _read_npy_header(path: str) -> NpyHeader:
    source = Path(path)
    try:
        stat = source.stat()
        array = np.load(source, mmap_mode="r", allow_pickle=False)
        shape = tuple(int(value) for value in array.shape)
        dtype = str(array.dtype)
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            mmap.close()
        del array
        return NpyHeader(
            path=str(source.resolve()),
            shape=shape,
            dtype=dtype,
            file_size_bytes=int(stat.st_size),
            mtime_ns=int(stat.st_mtime_ns),
        )
    except Exception as exc:  # noqa: BLE001 - persisted as evidence, not swallowed
        try:
            stat = source.stat()
            size, mtime = int(stat.st_size), int(stat.st_mtime_ns)
        except OSError:
            size, mtime = -1, -1
        return NpyHeader(
            path=str(source.resolve()),
            shape=(),
            dtype="unknown",
            file_size_bytes=size,
            mtime_ns=mtime,
            error=f"{type(exc).__name__}: {exc}",
        )


def _read_bvh_compact(path: str) -> BvhCompact:
    source = Path(path)
    try:
        stat = source.stat()
        header = parse_bvh_header(source)
        compact = header.compact_dict()
        return BvhCompact(
            path=str(source.resolve()),
            frames=int(compact["frames"]),
            frame_time=float(compact["frame_time"]),
            fps_src=float(compact["fps_src"]),
            source_joint_count=int(compact["source_joint_count"]),
            source_channel_count=int(compact["source_channel_count"]),
            rotation_layout_sha256=str(compact["rotation_layout_sha256"]),
            rest_layout_sha256=str(compact["rest_layout_sha256"]),
            file_size_bytes=int(stat.st_size),
            mtime_ns=int(stat.st_mtime_ns),
        )
    except Exception as exc:  # noqa: BLE001 - exact source error belongs in manifest
        try:
            stat = source.stat()
            size, mtime = int(stat.st_size), int(stat.st_mtime_ns)
        except OSError:
            size, mtime = None, None
        return BvhCompact(
            path=str(source.resolve()),
            frames=None,
            frame_time=None,
            fps_src=None,
            source_joint_count=None,
            source_channel_count=None,
            rotation_layout_sha256=None,
            rest_layout_sha256=None,
            file_size_bytes=size,
            mtime_ns=mtime,
            error=f"{type(exc).__name__}: {exc}",
        )


def _parallel_map(
    paths: Sequence[str],
    worker: Any,
    *,
    workers: int,
    label: str,
    progress_every: int = 5000,
) -> dict[str, Any]:
    if workers <= 0:
        raise InventoryError("workers must be positive")
    result: dict[str, Any] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for index, item in enumerate(
            executor.map(worker, paths, chunksize=32), start=1
        ):
            result[item.path] = item
            if progress_every and index % progress_every == 0:
                print(f"[inventory] {label}: {index}/{len(paths)}", flush=True)
    print(f"[inventory] {label}: {len(paths)}/{len(paths)}", flush=True)
    return result


def _load_split_map(split_root: Path) -> tuple[dict[str, str], dict[str, int]]:
    mapping: dict[str, str] = {}
    counts: dict[str, int] = {}
    for split in ("train", "val", "held_representative", "held_stress"):
        path = split_root / f"{split}.txt"
        if not path.is_file():
            raise InventoryError(f"missing frozen split file: {path}")
        count = 0
        for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            name = raw.strip()
            if not name or name.startswith("#"):
                continue
            if Path(name).name != name or not name.endswith(".npy"):
                raise InventoryError(f"{path}:{line_number}: invalid clip basename {name!r}")
            if name in mapping:
                raise InventoryError(
                    f"split overlap for {name}: {mapping[name]} and {split}"
                )
            mapping[name] = split
            count += 1
        counts[split] = count
    return mapping, counts


def _resolve_rig_id(filename: str, rig_prefixes: Sequence[tuple[str, str]]) -> str:
    stem = Path(filename).stem
    for prefix, rig_id in rig_prefixes:
        if stem.startswith(prefix):
            return rig_id
    raise InventoryError(f"cannot resolve rig from current filename {filename!r}")


def _truebones_action_key(clip_id: str, rig_id: str) -> str:
    prefix = rig_id + "_"
    if not clip_id.startswith(prefix):
        raise InventoryError(f"{clip_id!r} does not start with rig prefix {prefix!r}")
    remainder = clip_id[len(prefix):]
    match = re.match(r"^(.+)_([0-9]+)$", remainder)
    if not match:
        raise InventoryError(f"Truebones clip has no global numeric suffix: {clip_id}")
    return match.group(1)


def _truebones_counter(clip_id: str) -> int:
    match = re.search(r"_([0-9]+)$", clip_id)
    if not match:
        raise InventoryError(f"Truebones clip has no numeric suffix: {clip_id}")
    return int(match.group(1))


def _choose_truebones_rest(rig_dir: Path) -> tuple[Path, str]:
    files = sorted(rig_dir.glob("*.bvh"), key=lambda path: path.name.casefold())
    if not files:
        raise InventoryError(f"Truebones rig contains no BVH: {rig_dir}")
    tposes = [path for path in files if "tpos" in path.name.casefold()]
    if tposes:
        return tposes[0], "explicit_tpose_filename"
    idles = [
        path for path in files
        if path.stem.casefold().startswith(("idle", "__idle"))
    ]
    if idles:
        return idles[0], "legacy_idle_fallback"
    return files[0], "legacy_first_file_fallback"


def _source_kind_for_rig(rig_id: str) -> tuple[str, str]:
    if rig_id == "HML3D_Human":
        return "motionstreamer272", "motionstreamer272_rotation6d"
    if rig_id.startswith("PZ_"):
        return "planetzoo", "processed_bvh_rotation_channels"
    return "truebones", "raw_bvh_rotation_channels"


def _validate_current_cond(cond: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    validated: dict[str, dict[str, Any]] = {}
    for rig_id, raw in cond.items():
        if not isinstance(rig_id, str) or not rig_id:
            raise InventoryError(f"invalid cond rig id {rig_id!r}")
        if not isinstance(raw, Mapping):
            raise InventoryError(f"cond[{rig_id!r}] is not a mapping")
        for key in ("parents", "joints_names", "offsets"):
            if key not in raw:
                raise InventoryError(f"cond[{rig_id!r}] missing {key}")
        parents = [int(value) for value in np.asarray(raw["parents"]).tolist()]
        names = [str(value) for value in np.asarray(raw["joints_names"]).tolist()]
        offsets = np.asarray(raw["offsets"], dtype=np.float64)
        if len(parents) != len(names) or offsets.shape != (len(names), 3):
            raise InventoryError(f"cond[{rig_id!r}] joint metadata shape mismatch")
        if len(set(names)) != len(names):
            raise InventoryError(f"cond[{rig_id!r}] has duplicate joint names")
        if not np.isfinite(offsets).all():
            raise InventoryError(f"cond[{rig_id!r}] has non-finite offsets")
        depth = _tree_depth(parents)
        validated[rig_id] = {
            "parents": parents,
            "joint_names": names,
            "joint_count": len(names),
            "tree_depth": depth,
            "topology_parent_sha256": _sha256_json(parents),
            "topology_named_sha256": _sha256_json(
                {"joint_names": names, "parents": parents}
            ),
        }
    return validated


def _source_ancestors(header: BvhHeader, index: int) -> list[int]:
    result: list[int] = []
    parent = int(header.joints[index].parent)
    while parent >= 0:
        result.append(parent)
        parent = int(header.joints[parent].parent)
    return result


def _build_joint_map(
    rig_id: str,
    current: Mapping[str, Any],
    source_header: BvhHeader | None,
    source_family: str,
) -> tuple[dict[str, Any], list[str], list[str]]:
    codes: list[str] = []
    diagnostics: list[str] = []
    current_names = list(current["joint_names"])
    current_parents = list(current["parents"])
    if source_family == "motionstreamer272":
        mapping = list(range(len(current_names)))
        kinds = ["animated_dof"] * len(current_names)
        source_names = current_names
        source_parents = current_parents
        source_node_kinds = ["joint"] * len(current_names)
        source_rotation_layout = _sha256_json(
            {
                "schema": "MotionStreamer272[140:272].reshape(T,22,6)",
                "joint_names": source_names,
                "parents": source_parents,
            }
        )
        direct_edges = len(current_names) - 1
        skipped_edges = 0
        source_root_index = 0
    else:
        if source_header is None:
            _add_code(codes, "SOURCE_HEADER_INVALID")
            diagnostics.append("rig baseline BVH header unavailable")
            payload = {
                "status": "invalid",
                "mapping_kind": "exact_joint_name_with_source_ancestry_check",
                "btjd_joint_names": current_names,
                "btjd_parents": current_parents,
                "source_joint_names": [],
                "source_parents": [],
                "source_node_kinds": [],
                "btjd_to_source": [],
                "source_root_index_for_btjd_root": -1,
                "rotation_source_kind": [],
                "animated_dof_count": 0,
                "fixed_dof_count": 0,
                "missing_or_unknown_count": len(current_names),
                "direct_source_edge_count": 0,
                "source_skipping_edge_count": 0,
                "source_rotation_layout_sha256": None,
                "structural_unnamed_end_site_maps": [],
            }
            payload["joint_map_sha256"] = _sha256_json(
                {
                    "btjd_joint_names": current_names,
                    "btjd_parents": current_parents,
                    "btjd_to_source": [],
                    "rotation_source_kind": [],
                }
            )
            return payload, codes, diagnostics
        positions: dict[str, list[int]] = defaultdict(list)
        for index, name in enumerate(source_header.joint_names):
            positions[name].append(index)
        mapping: list[int] = []
        missing = []
        ambiguous = []
        structural_end_site_maps: list[dict[str, Any]] = []
        used_source_indices: set[int] = set()
        for current_index, name in enumerate(current_names):
            hits = positions.get(name, [])
            if not hits:
                # The legacy BVH saver names a previously unnamed End Site as
                # ``<parent>_end_site``.  Recover that association only when
                # parent order is already proven and exactly one unused,
                # source-declared End Site exists under the mapped parent.
                candidate_indices: list[int] = []
                current_parent = current_parents[current_index]
                if (
                    current_index > 0
                    and name.endswith("_end_site")
                    and current_parent >= 0
                    and current_parent < len(mapping)
                    and mapping[current_parent] >= 0
                ):
                    source_parent = mapping[current_parent]
                    candidate_indices = [
                        source_index
                        for source_index, joint in enumerate(source_header.joints)
                        if joint.parent == source_parent
                        and joint.node_kind == "end_site"
                        and "__unnamed_end_site_" in joint.name
                        and source_index not in used_source_indices
                    ]
                if len(candidate_indices) == 1:
                    source_index = candidate_indices[0]
                    mapping.append(source_index)
                    used_source_indices.add(source_index)
                    structural_end_site_maps.append(
                        {
                            "btjd_joint": name,
                            "source_joint": source_header.joints[source_index].name,
                            "mapped_parent": current_names[current_parent],
                            "evidence": "unique unnamed End Site under exact mapped parent",
                        }
                    )
                elif len(candidate_indices) > 1:
                    ambiguous.append(name)
                    mapping.append(-1)
                else:
                    missing.append(name)
                    mapping.append(-1)
            elif len(hits) > 1:
                ambiguous.append(name)
                mapping.append(-1)
            else:
                mapping.append(hits[0])
                used_source_indices.add(hits[0])
        if missing:
            _add_code(codes, "JOINT_MAP_MISSING")
            diagnostics.append(f"missing source joints: {missing[:12]}")
        if ambiguous:
            _add_code(codes, "JOINT_MAP_AMBIGUOUS")
            diagnostics.append(f"ambiguous source joints: {ambiguous[:12]}")
        kinds = []
        if not missing and not ambiguous:
            try:
                all_kinds = source_header.rotation_source_kinds()
                kinds = [all_kinds[index] for index in mapping]
            except BvhInventoryError as exc:
                _add_code(codes, "ROTATION_PROVENANCE_INVALID")
                diagnostics.append(str(exc))
        if kinds and any(kind not in {"animated_dof", "fixed_dof"} for kind in kinds):
            _add_code(codes, "ROTATION_PROVENANCE_INVALID")

        direct_edges = 0
        skipped_edges = 0
        if not missing and not ambiguous:
            for child in range(1, len(mapping)):
                expected_parent = mapping[current_parents[child]]
                source_parent = source_header.parents[mapping[child]]
                if source_parent == expected_parent:
                    direct_edges += 1
                    continue
                ancestors = _source_ancestors(source_header, mapping[child])
                if expected_parent not in ancestors:
                    _add_code(codes, "CURRENT_PARENT_NOT_SOURCE_ANCESTOR")
                    diagnostics.append(
                        f"current edge {current_names[current_parents[child]]!r} -> "
                        f"{current_names[child]!r} is not a source ancestor edge"
                    )
                else:
                    skipped_edges += 1
            if skipped_edges:
                _add_code(codes, "JOINT_MAP_SKIPS_SOURCE_JOINTS")
        source_root_index = mapping[0] if mapping else -1
        if source_root_index != 0 and source_root_index >= 0:
            _add_code(codes, "SOURCE_ROOT_WRAPPER_DROPPED")
        source_names = list(source_header.joint_names)
        source_parents = list(source_header.parents)
        source_node_kinds = [joint.node_kind for joint in source_header.joints]
        source_rotation_layout = source_header.rotation_layout_sha256()

    payload = {
        "status": "binary_proven" if not any(
            REASON_CODES[code]["severity"] == "reject" for code in codes
        ) else "invalid",
        "mapping_kind": (
            "motionstreamer_schema_index_identity"
            if source_family == "motionstreamer272"
            else "exact_joint_name_with_source_ancestry_check"
        ),
        "btjd_joint_names": current_names,
        "btjd_parents": current_parents,
        "source_joint_names": source_names,
        "source_parents": source_parents,
        "source_node_kinds": source_node_kinds,
        "btjd_to_source": mapping,
        "source_root_index_for_btjd_root": source_root_index,
        "rotation_source_kind": kinds,
        "animated_dof_count": kinds.count("animated_dof"),
        "fixed_dof_count": kinds.count("fixed_dof"),
        "missing_or_unknown_count": len(current_names) - len(kinds),
        "direct_source_edge_count": direct_edges,
        "source_skipping_edge_count": skipped_edges,
        "source_rotation_layout_sha256": source_rotation_layout,
        "structural_unnamed_end_site_maps": (
            structural_end_site_maps if source_family != "motionstreamer272" else []
        ),
    }
    payload["joint_map_sha256"] = _sha256_json(
        {
            "btjd_joint_names": current_names,
            "btjd_parents": current_parents,
            "btjd_to_source": mapping,
            "rotation_source_kind": kinds,
        }
    )
    return payload, codes, diagnostics


def _build_rig_records(
    config: InventoryConfig,
    cond: Mapping[str, Mapping[str, Any]],
    clips_by_rig: Mapping[str, list[str]],
    source_path_by_clip: Mapping[str, str],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    records: dict[str, dict[str, Any]] = {}
    baseline_path_by_rig: dict[str, str] = {}
    for rig_id in sorted(cond):
        source_family, source_kind = _source_kind_for_rig(rig_id)
        codes: list[str] = []
        diagnostics: list[str] = []
        source_header: BvhHeader | None = None
        rest: dict[str, Any]
        evidence_paths: list[str]
        if source_family == "truebones":
            rig_dir = config.truebones_raw_root / rig_id
            try:
                rest_path, rest_method = _choose_truebones_rest(rig_dir)
                source_header = parse_bvh_header(rest_path)
                baseline_path_by_rig[rig_id] = str(rest_path.resolve())
                rest = {
                    "status": "source_hierarchy_offsets_present",
                    "source_path": str(rest_path.resolve()),
                    "selection_method": rest_method,
                    "rest_layout_sha256": source_header.rest_layout_sha256(),
                }
                if rest_method != "explicit_tpose_filename":
                    _add_code(codes, "REST_FRAME_FALLBACK_NOT_EXPLICIT")
            except Exception as exc:  # noqa: BLE001
                _add_code(codes, "SOURCE_HEADER_INVALID")
                diagnostics.append(f"rest evidence: {type(exc).__name__}: {exc}")
                rest = {"status": "invalid", "source_path": None}
            evidence_paths = [str(config.truebones_raw_root)]
        elif source_family == "planetzoo":
            rig_clips = sorted(clips_by_rig.get(rig_id, []))
            if not rig_clips:
                _add_code(codes, "SOURCE_FILE_MISSING")
                rest = {"status": "invalid", "source_path": None}
            else:
                baseline = Path(source_path_by_clip[rig_clips[0]])
                try:
                    source_header = parse_bvh_header(baseline)
                    baseline_path_by_rig[rig_id] = str(baseline.resolve())
                    rest = {
                        "status": "processed_bvh_hierarchy_offsets_only",
                        "source_path": str(baseline.resolve()),
                        "selection_method": "first_sorted_processed_clip",
                        "rest_layout_sha256": source_header.rest_layout_sha256(),
                    }
                except Exception as exc:  # noqa: BLE001
                    _add_code(codes, "SOURCE_HEADER_INVALID")
                    diagnostics.append(f"rig baseline: {type(exc).__name__}: {exc}")
                    rest = {"status": "invalid", "source_path": str(baseline.resolve())}
            _add_code(codes, "PZ_RAW_GAME_BVH_NOT_LOCAL")
            _add_code(codes, "PZ_SOURCE_HAS_PER_CLIP_CANONICALIZATION")
            _add_code(codes, "REST_SOURCE_REQUIRES_RAW_TPOSE_RECOVERY")
            evidence_paths = [str(config.planetzoo_lineage_path)]
        else:
            rest = {
                "status": "smpl_neutral_model_present",
                "source_path": str(config.smpl_neutral_model_path),
                "selection_method": "SMPLH neutral betas zero via current audited builder",
                "rest_layout_sha256": _sha256_json(
                    {
                        "path": str(config.smpl_neutral_model_path),
                        "size": config.smpl_neutral_model_path.stat().st_size
                        if config.smpl_neutral_model_path.is_file() else None,
                    }
                ),
            }
            if not config.smpl_neutral_model_path.is_file():
                _add_code(codes, "SOURCE_FILE_MISSING")
                diagnostics.append("SMPL-neutral model missing")
            _add_code(codes, "HUMAN_CURRENT_BRIDGE_USES_PER_CLIP_ALIGNMENT")
            evidence_paths = [
                str(config.human_builder_path),
                str(config.smpl_neutral_model_path),
            ]

        joint_map, map_codes, map_diagnostics = _build_joint_map(
            rig_id, cond[rig_id], source_header, source_family
        )
        for code in map_codes:
            _add_code(codes, code)
        diagnostics.extend(map_diagnostics)
        _add_code(codes, "NUMERIC_PAYLOAD_VALIDATION_DEFERRED_T03")
        _add_code(codes, "HEADING_PAYLOAD_UNREVIEWED")
        _add_code(codes, "SOURCE_TO_CANONICAL_UNREVIEWED")
        _add_code(codes, "SOURCE_UNIT_TO_METER_UNKNOWN")

        current = cond[rig_id]
        topology_family = classify_topology_family(
            rig_id, int(current["tree_depth"])
        )
        record = {
            "manifest_version": INVENTORY_VERSION,
            "rig_id": rig_id,
            "source_family": source_family,
            "source_kind": source_kind,
            "source_rotation_authority": (
                "MotionStreamer272 rotation6d slice 140:272"
                if source_family == "motionstreamer272"
                else (
                    "processed BVH channels derived from native MANIS/BVH rotations"
                    if source_family == "planetzoo"
                    else "original Truebones BVH rotation channels"
                )
            ),
            "current_clip_count": len(clips_by_rig.get(rig_id, [])),
            "topology_family": topology_family,
            "topology_tree_depth": int(current["tree_depth"]),
            "topology_parent_sha256": current["topology_parent_sha256"],
            "topology_named_sha256": current["topology_named_sha256"],
            "rest_pose": rest,
            "joint_map": joint_map,
            "rotation_provenance_status": (
                "proven" if joint_map.get("status") == "binary_proven" else "invalid"
            ),
            "unit": {
                "length_unit_id": {
                    "truebones": "truebones_bvh_native_unlabeled",
                    "planetzoo": "anytop_planetzoo_canonical_unlabeled",
                    "motionstreamer272": "motionstreamer272_native_unverified",
                }[source_family],
                "source_unit_to_meter": None,
                "canonical_scale_factor": None,
                "meter_claim": False,
            },
            "source_to_canonical": {
                "status": "numeric_per_rig_transform_unreviewed",
                "C": None,
                "alpha": None,
                "o": None,
                "evidence_paths": evidence_paths,
            },
            "heading": {
                "status": "candidate_unreviewed",
                "heading_carrier_joint_candidate": 0,
                "heading_carrier_name_candidate": current["joint_names"][0],
                "u_forward_local_candidate": None,
                "polarity": "unreviewed",
                "provenance": "unreviewed",
            },
            "status": _status_from_codes(codes),
            "reason_codes": sorted(codes),
            "diagnostics": diagnostics,
        }
        evidence_payload = dict(record)
        record["rig_evidence_sha256"] = _sha256_json(evidence_payload)
        records[rig_id] = record
    return records, baseline_path_by_rig


def _build_truebones_source_index(root: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for rig_dir in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda p: p.name):
        index: dict[str, str] = {}
        collisions: set[str] = set()
        for path in sorted(rig_dir.glob("*.bvh"), key=lambda p: p.name.casefold()):
            key = path.stem.casefold()
            if key in index:
                collisions.add(key)
            index[key] = str(path.resolve())
        for key in collisions:
            index.pop(key, None)
        result[rig_dir.name] = index
    return result


def _round_robin_select(descriptors: Sequence[ClipDescriptor], limit: int) -> list[str]:
    groups: dict[str, deque[str]] = defaultdict(deque)
    for descriptor in sorted(descriptors, key=lambda item: (item.rig_id, item.clip_id)):
        groups[descriptor.rig_id].append(descriptor.clip_id)
    queue = deque(sorted(groups))
    selected: list[str] = []
    while queue and len(selected) < limit:
        rig_id = queue.popleft()
        selected.append(groups[rig_id].popleft())
        if groups[rig_id]:
            queue.append(rig_id)
    return selected


def _mark_raw_source_split_overlaps(
    descriptors: Sequence[ClipDescriptor],
) -> list[dict[str, Any]]:
    """Reject every slice of a live raw sequence that crosses frozen splits.

    Truebones long BVHs were historically sliced before the clip-level split
    was frozen.  Treating those slices as independent would let adjacent frames
    from one temporal source enter train and validation.  We preserve the
    frozen clip labels as evidence, but fail closed for every affected slice.
    """
    groups: dict[str, list[ClipDescriptor]] = defaultdict(list)
    for descriptor in descriptors:
        if descriptor.source is None or getattr(descriptor.source, "error", None):
            continue
        groups[descriptor.source_path].append(descriptor)

    audit: list[dict[str, Any]] = []
    for source_path, group in sorted(groups.items()):
        splits = tuple(sorted({item.split for item in group}))
        for descriptor in group:
            descriptor.source_sequence_splits = splits
        if len(splits) <= 1:
            continue
        clip_ids = sorted(item.clip_id for item in group)
        for descriptor in group:
            _add_code(
                descriptor.clip_reason_codes,
                "RAW_SOURCE_SEQUENCE_SPLIT_OVERLAP",
            )
            descriptor.diagnostics.append(
                "raw temporal source crosses frozen splits "
                f"{list(splits)}; all {len(group)} slices excluded"
            )
        audit.append(
            {
                "source_path": source_path,
                "splits": list(splits),
                "clip_count": len(group),
                "clip_ids": clip_ids,
            }
        )
    return audit


def _clip_record(
    descriptor: ClipDescriptor,
    rig_record: Mapping[str, Any],
) -> dict[str, Any]:
    codes = list(rig_record["reason_codes"])
    for code in descriptor.clip_reason_codes:
        _add_code(codes, code)
    source_payload: dict[str, Any] = {
        "family": descriptor.source_family,
        "kind": descriptor.source_kind,
        "path": descriptor.source_path,
        "slice_frames": list(descriptor.source_slice) if descriptor.source_slice else None,
        "frame_mapping": descriptor.source_frame_mapping,
        "sequence_splits": list(descriptor.source_sequence_splits),
        "sequence_split_safe": (
            len(descriptor.source_sequence_splits) <= 1
            if descriptor.source_sequence_splits
            else None
        ),
    }
    if isinstance(descriptor.source, BvhCompact):
        source_payload.update(
            {
                "T_src": descriptor.source.frames,
                "frame_time_src": descriptor.source.frame_time,
                "fps_src": descriptor.source.fps_src,
                "native_fps_evidence": "BVH Frame Time header",
                "source_joint_count": descriptor.source.source_joint_count,
                "source_channel_count": descriptor.source.source_channel_count,
                "rotation_layout_sha256": descriptor.source.rotation_layout_sha256,
                "file_size_bytes": descriptor.source.file_size_bytes,
                "mtime_ns": descriptor.source.mtime_ns,
                "header_error": descriptor.source.error,
            }
        )
    elif isinstance(descriptor.source, NpyHeader):
        source_payload.update(
            {
                "shape": list(descriptor.source.shape),
                "dtype": descriptor.source.dtype,
                "T_src": descriptor.source.shape[0] if descriptor.source.shape else None,
                "frame_time_src": 1.0 / 30.0,
                "fps_src": 30.0,
                "native_fps_evidence": "MotionStreamer272 local builder contract",
                "rotation_slice": [140, 272],
                "rotation_shape": [22, 6],
                "file_size_bytes": descriptor.source.file_size_bytes,
                "mtime_ns": descriptor.source.mtime_ns,
                "header_error": descriptor.source.error,
            }
        )
    else:
        source_payload["header_error"] = "source metadata unavailable"

    joint_map = rig_record["joint_map"]
    rotation_invalidating_codes = {
        code
        for code in descriptor.clip_reason_codes
        if REASON_CODES[code]["severity"] == "reject"
        and code != "RAW_SOURCE_SEQUENCE_SPLIT_OVERLAP"
    }
    rotation_valid = (
        rig_record["rotation_provenance_status"] == "proven"
        and not rotation_invalidating_codes
    )
    has_reject = any(REASON_CODES[code]["severity"] == "reject" for code in codes)
    return {
        "manifest_version": INVENTORY_VERSION,
        "clip_id": descriptor.clip_id,
        "rig_id": descriptor.rig_id,
        "btjd": {
            "path": descriptor.btjd.path,
            "shape": list(descriptor.btjd.shape),
            "dtype": descriptor.btjd.dtype,
            "file_size_bytes": descriptor.btjd.file_size_bytes,
            "mtime_ns": descriptor.btjd.mtime_ns,
        },
        "split": descriptor.split,
        "split_protocol": "holdout_splits_v1",
        "split_eligible_for_train_calibration": (
            descriptor.split == "train" and rotation_valid and not has_reject
        ),
        "topology_family": descriptor.topology_family,
        "topology_distance_bucket": descriptor.topology_distance_bucket,
        "topology_tree_depth": rig_record["topology_tree_depth"],
        "source": source_payload,
        "rotation_provenance": {
            "status": "proven" if rotation_valid else "invalid",
            "allowed_kinds": ["animated_dof", "fixed_dof"],
            "animated_dof_count": joint_map.get("animated_dof_count", 0),
            "fixed_dof_count": joint_map.get("fixed_dof_count", 0),
            "missing_ik_legacy_unknown_count": joint_map.get(
                "missing_or_unknown_count", len(descriptor.btjd.shape)
            ),
            "joint_map_sha256": joint_map.get("joint_map_sha256"),
            "rig_evidence_sha256": rig_record["rig_evidence_sha256"],
        },
        "rest_pose": {
            "status": rig_record["rest_pose"].get("status"),
            "source_path": rig_record["rest_pose"].get("source_path"),
            "rig_evidence_sha256": rig_record["rig_evidence_sha256"],
        },
        "unit": rig_record["unit"],
        "source_to_canonical": {
            "status": rig_record["source_to_canonical"]["status"],
            "rig_evidence_sha256": rig_record["rig_evidence_sha256"],
        },
        "heading": {
            "status": rig_record["heading"]["status"],
            "carrier_joint_candidate": rig_record["heading"][
                "heading_carrier_joint_candidate"
            ],
            "u_forward_local_candidate": rig_record["heading"][
                "u_forward_local_candidate"
            ],
            "polarity": rig_record["heading"]["polarity"],
        },
        "prototype_candidate": descriptor.prototype_candidate,
        "status": _status_from_codes(codes),
        "reason_codes": sorted(codes),
        "diagnostics": descriptor.diagnostics,
    }


def _jsonl_line(value: Any) -> str:
    return _canonical_json(value).decode("utf-8") + "\n"


def _write_transaction(
    output_root: Path,
    outputs: Mapping[str, Iterable[str] | str],
    *,
    overwrite: bool,
) -> dict[str, Any]:
    """Publish one immutable manifest generation through an atomic symlink.

    GPFS does not support Linux ``RENAME_EXCHANGE``.  Replacing several public
    files one by one can therefore expose a mixed generation after a crash.
    Every file is instead fsynced in an immutable sibling directory, and the
    single public ``dataset/manifests`` symlink is replaced atomically.  Readers
    resolve that link once to obtain a stable snapshot.
    """
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if output_root.exists() and not output_root.is_symlink():
        raise InventoryError(
            f"{output_root} is a real directory; migrate it once to the managed "
            f"{GENERATION_DIRECTORY_NAME} symlink layout before overwrite"
        )
    if output_root.is_symlink() and not output_root.exists():
        raise InventoryError(f"refusing to replace broken manifest symlink: {output_root}")

    existing = [
        str(output_root / name)
        for name in outputs
        if (output_root / name).exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            "inventory outputs already exist; pass --overwrite after review: "
            + ", ".join(existing)
        )

    generation_root = output_root.parent / GENERATION_DIRECTORY_NAME
    generation_root.mkdir(parents=True, exist_ok=True)
    generation_id = (
        _datetime.datetime.now(_datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + "-"
        + uuid.uuid4().hex[:12]
    )
    stage = Path(
        tempfile.mkdtemp(prefix=f".stage-{generation_id}-", dir=generation_root)
    )
    final_generation = generation_root / generation_id
    link_tmp = output_root.parent / f".{output_root.name}.{generation_id}.tmp"
    published = False
    try:
        for name, content in outputs.items():
            target = stage / name
            with target.open("x", encoding="utf-8") as handle:
                if isinstance(content, str):
                    handle.write(content)
                else:
                    for chunk in content:
                        handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())

        file_records = {
            name: {
                "sha256": _sha256_file(stage / name),
                "size_bytes": (stage / name).stat().st_size,
            }
            for name in sorted(outputs)
        }
        transaction = {
            "manifest_version": INVENTORY_VERSION,
            "generation_id": generation_id,
            "publish_protocol": "immutable_generation_atomic_symlink_replace",
            "files": file_records,
        }
        transaction_path = stage / TRANSACTION_FILENAME
        with transaction_path.open("x", encoding="utf-8") as handle:
            json.dump(
                transaction,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        directory_fd = os.open(stage, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(stage, final_generation)

        relative_target = os.path.relpath(final_generation, output_root.parent)
        os.symlink(relative_target, link_tmp)
        os.replace(link_tmp, output_root)
        # From this point onward the public authoritative symlink may resolve
        # to ``final_generation``.  Never let a later durability error make the
        # cleanup path delete that now-active generation.
        published = True
        parent_fd = os.open(output_root.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return transaction
    finally:
        if link_tmp.is_symlink():
            link_tmp.unlink()
        if stage.exists():
            shutil.rmtree(stage)
        if not published and final_generation.exists():
            active_target: Path | None = None
            if output_root.is_symlink():
                try:
                    link_text = os.readlink(output_root)
                    active_target = (output_root.parent / link_text).resolve()
                except OSError:
                    active_target = None
            if active_target != final_generation.resolve():
                shutil.rmtree(final_generation)


def run_inventory(config: InventoryConfig) -> dict[str, Any]:
    """Scan the live corpus and atomically materialize all T02 artifacts."""
    config = config.resolved()
    if config.prototype_min_train_clips <= 0:
        raise InventoryError("prototype_min_train_clips must be positive")
    existing_outputs = [
        str(config.output_root / name)
        for name in OUTPUT_FILENAMES
        if (config.output_root / name).exists()
    ]
    if existing_outputs and not config.overwrite:
        raise FileExistsError(
            "inventory outputs already exist; pass --overwrite after review: "
            + ", ".join(existing_outputs)
        )
    for path, label in (
        (config.dataset_root, "current dataset root"),
        (config.split_root, "frozen split root"),
        (config.pz_bvh_root, "Planet Zoo BVH root"),
        (config.truebones_raw_root, "Truebones raw root"),
        (config.human272_root, "MotionStreamer272 root"),
    ):
        if not path.is_dir():
            raise InventoryError(f"missing {label}: {path}")
    cond_path = config.dataset_root / "cond.npy"
    motion_root = config.dataset_root / "motions"
    if not cond_path.is_file() or not motion_root.is_dir():
        raise InventoryError(f"invalid current dataset layout: {config.dataset_root}")

    raw_cond = np.load(cond_path, allow_pickle=True).item()
    if not isinstance(raw_cond, Mapping):
        raise InventoryError("cond.npy does not contain a rig mapping")
    cond = _validate_current_cond(raw_cond)
    split_map, split_counts = _load_split_map(config.split_root)
    motion_paths = sorted(motion_root.glob("*.npy"), key=lambda path: path.name)
    motion_names = {path.name for path in motion_paths}
    if motion_names != set(split_map):
        missing_split = sorted(motion_names - set(split_map))
        missing_disk = sorted(set(split_map) - motion_names)
        raise InventoryError(
            "frozen split coverage mismatch: "
            f"unassigned_on_disk={missing_split[:10]}, missing_on_disk={missing_disk[:10]}"
        )
    if sum(split_counts.values()) != len(motion_paths):
        raise InventoryError("split counts do not sum to current motion count")

    current_headers = _parallel_map(
        [str(path.resolve()) for path in motion_paths],
        _read_npy_header,
        workers=config.workers,
        label="current BTJD NPY headers",
    )
    rig_prefixes = sorted(
        ((rig_id + "_", rig_id) for rig_id in cond),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    clips_by_rig: dict[str, list[str]] = defaultdict(list)
    rig_by_clip: dict[str, str] = {}
    for path in motion_paths:
        rig_id = _resolve_rig_id(path.name, rig_prefixes)
        rig_by_clip[path.stem] = rig_id
        clips_by_rig[rig_id].append(path.stem)
    missing_rigs = sorted(set(cond) - set(clips_by_rig))
    if missing_rigs:
        raise InventoryError(f"cond rigs with no current clips: {missing_rigs}")

    truebones_index = _build_truebones_source_index(config.truebones_raw_root)
    source_path_by_clip: dict[str, str] = {}
    pre_source_errors: dict[str, tuple[str, str]] = {}
    for clip_id, rig_id in rig_by_clip.items():
        source_family, _ = _source_kind_for_rig(rig_id)
        if source_family == "motionstreamer272":
            source_id = clip_id.removeprefix("HML3D_Human_")
            source_path = config.human272_root / "motion_data" / f"{source_id}.npy"
        elif source_family == "planetzoo":
            source_path = config.pz_bvh_root / f"{clip_id}.bvh"
        else:
            try:
                action = _truebones_action_key(clip_id, rig_id)
                resolved = truebones_index.get(rig_id, {}).get(action.casefold())
                if resolved is None:
                    raise InventoryError(
                        f"raw action {action!r} missing or ambiguous under rig {rig_id!r}"
                    )
                source_path = Path(resolved)
            except Exception as exc:  # noqa: BLE001
                source_path = config.truebones_raw_root / rig_id / "__MISSING__.bvh"
                pre_source_errors[clip_id] = (
                    "SOURCE_FILE_MISSING",
                    f"{type(exc).__name__}: {exc}",
                )
        source_path_by_clip[clip_id] = str(source_path.resolve())

    rig_records, _ = _build_rig_records(
        config, cond, clips_by_rig, source_path_by_clip
    )

    bvh_paths = sorted(
        {
            path
            for clip_id, path in source_path_by_clip.items()
            if _source_kind_for_rig(rig_by_clip[clip_id])[0] != "motionstreamer272"
            and Path(path).is_file()
        }
    )
    bvh_compact = _parallel_map(
        bvh_paths,
        _read_bvh_compact,
        workers=config.workers,
        label="source BVH headers",
    )
    human_paths = sorted(
        {
            path
            for clip_id, path in source_path_by_clip.items()
            if _source_kind_for_rig(rig_by_clip[clip_id])[0] == "motionstreamer272"
        }
    )
    human_headers = _parallel_map(
        human_paths,
        _read_npy_header,
        workers=config.workers,
        label="MotionStreamer272 NPY headers",
    )

    descriptors: list[ClipDescriptor] = []
    descriptor_by_id: dict[str, ClipDescriptor] = {}
    for path in motion_paths:
        clip_id = path.stem
        rig_id = rig_by_clip[clip_id]
        current = current_headers[str(path.resolve())]
        source_family, source_kind = _source_kind_for_rig(rig_id)
        source_path = source_path_by_clip[clip_id]
        source: BvhCompact | NpyHeader | None
        source = (
            human_headers.get(source_path)
            if source_family == "motionstreamer272"
            else bvh_compact.get(source_path)
        )
        descriptor = ClipDescriptor(
            clip_id=clip_id,
            rig_id=rig_id,
            btjd=current,
            split=split_map[path.name],
            topology_family=rig_records[rig_id]["topology_family"],
            topology_distance_bucket=_topology_bucket(split_map[path.name]),
            source_family=source_family,
            source_kind=source_kind,
            source_path=source_path,
            source=source,
        )
        if (
            current.error
            or len(current.shape) != 3
            or current.shape[-1] != 13
            or current.dtype not in {"float32", "float64"}
        ):
            _add_code(descriptor.clip_reason_codes, "BTJD_SHAPE_INVALID")
            descriptor.diagnostics.append(f"BTJD header: {current.error or current.shape}")
        elif current.shape[1] != cond[rig_id]["joint_count"] or current.shape[0] <= 0:
            _add_code(descriptor.clip_reason_codes, "BTJD_SHAPE_INVALID")
            descriptor.diagnostics.append(
                f"BTJD shape {current.shape} vs rig J={cond[rig_id]['joint_count']}"
            )
        if clip_id in pre_source_errors:
            code, diagnostic = pre_source_errors[clip_id]
            _add_code(descriptor.clip_reason_codes, code)
            descriptor.diagnostics.append(diagnostic)
        if source is None:
            _add_code(descriptor.clip_reason_codes, "SOURCE_FILE_MISSING")
            descriptor.diagnostics.append(f"source not scanned: {source_path}")
        elif source.error:
            _add_code(descriptor.clip_reason_codes, "SOURCE_HEADER_INVALID")
            descriptor.diagnostics.append(source.error)
        descriptor_by_id[clip_id] = descriptor
        descriptors.append(descriptor)

    # Per-clip layout and frame mapping checks.
    truebones_groups: dict[str, list[ClipDescriptor]] = defaultdict(list)
    for descriptor in descriptors:
        rig = rig_records[descriptor.rig_id]
        expected_layout = rig["joint_map"].get("source_rotation_layout_sha256")
        if isinstance(descriptor.source, BvhCompact) and not descriptor.source.error:
            if descriptor.source.rotation_layout_sha256 != expected_layout:
                try:
                    variant_header = parse_bvh_header(descriptor.source_path)
                    variant_map, variant_codes, variant_diagnostics = _build_joint_map(
                        descriptor.rig_id,
                        cond[descriptor.rig_id],
                        variant_header,
                        descriptor.source_family,
                    )
                    retained_compatible = (
                        variant_map.get("status") == "binary_proven"
                        and variant_map.get("rotation_source_kind")
                        == rig["joint_map"].get("rotation_source_kind")
                        and not any(
                            REASON_CODES[code]["severity"] == "reject"
                            for code in variant_codes
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    retained_compatible = False
                    variant_diagnostics = [f"{type(exc).__name__}: {exc}"]
                if retained_compatible:
                    # Full-layout drift can come from unused wrappers/helpers
                    # or unnamed terminal sites.  It remains reviewable, but it
                    # does not erase binary provenance for retained joints.
                    descriptor.diagnostics.append(
                        "full source layout differs; retained-joint rotation provenance remaps identically"
                    )
                    # This review code is clip-local rather than a rig-wide
                    # source-layout claim.
                    _add_code(
                        descriptor.clip_reason_codes,
                        "SOURCE_NONRETAINED_LAYOUT_VARIANT",
                    )
                else:
                    _add_code(descriptor.clip_reason_codes, "SOURCE_LAYOUT_DRIFT")
                    descriptor.diagnostics.append(
                        f"layout {descriptor.source.rotation_layout_sha256} != rig {expected_layout}; "
                        f"retained remap failed: {variant_diagnostics[:3]}"
                    )
            if descriptor.source_family == "planetzoo":
                descriptor.source_slice = (0, int(descriptor.source.frames or 0))
                descriptor.source_frame_mapping = "BTJD_T = source_BVH_frames - 1"
                if (
                    descriptor.btjd.shape
                    and descriptor.source.frames is not None
                    and descriptor.btjd.shape[0] != descriptor.source.frames - 1
                ):
                    _add_code(descriptor.clip_reason_codes, "SOURCE_FRAME_MAPPING_MISMATCH")
                    descriptor.diagnostics.append(
                        f"BTJD T={descriptor.btjd.shape[0]} vs BVH frames={descriptor.source.frames}"
                    )
            else:
                truebones_groups[descriptor.source_path].append(descriptor)
        elif isinstance(descriptor.source, NpyHeader) and not descriptor.source.error:
            source_t_raw = (
                int(descriptor.source.shape[0]) if descriptor.source.shape else 0
            )
            descriptor.source_slice = (0, source_t_raw)
            descriptor.source_frame_mapping = (
                "current BTJD only: pad source to >=6; T20=max(4,round(T30*20/30)); BTJD_T=T20-1"
            )
            if (
                len(descriptor.source.shape) != 2
                or descriptor.source.shape[1] != 272
                or descriptor.source.dtype not in {"float32", "float64"}
            ):
                _add_code(descriptor.clip_reason_codes, "SOURCE_HEADER_INVALID")
                descriptor.diagnostics.append(
                    f"MotionStreamer source shape {descriptor.source.shape}, expected [T,272]"
                )
            elif descriptor.btjd.shape:
                source_t = max(6, source_t_raw)
                expected = max(4, int(round(source_t * 20.0 / 30.0))) - 1
                if descriptor.btjd.shape[0] != expected:
                    _add_code(descriptor.clip_reason_codes, "SOURCE_FRAME_MAPPING_MISMATCH")
                    descriptor.diagnostics.append(
                        f"BTJD T={descriptor.btjd.shape[0]} vs documented current builder T={expected}"
                    )

    for source_path, group in truebones_groups.items():
        group.sort(key=lambda descriptor: _truebones_counter(descriptor.clip_id))
        source = group[0].source
        if not isinstance(source, BvhCompact) or source.frames is None:
            continue
        cursor = 0
        for descriptor in group:
            if not descriptor.btjd.shape:
                continue
            segment_frames = descriptor.btjd.shape[0] + 1
            descriptor.source_slice = (cursor, cursor + segment_frames)
            descriptor.source_frame_mapping = (
                "legacy Truebones contiguous <=200-frame slices; BTJD_T = slice_frames - 1"
            )
            cursor += segment_frames
        if cursor != source.frames:
            for descriptor in group:
                _add_code(descriptor.clip_reason_codes, "SOURCE_FRAME_MAPPING_MISMATCH")
                descriptor.diagnostics.append(
                    f"mapped slice coverage={cursor}, raw BVH frames={source.frames}"
                )

    source_split_audit = _mark_raw_source_split_overlaps(descriptors)

    # Select rotation-proven, non-rejected train candidates with rig round-robin diversity.
    prototype_records: dict[str, dict[str, Any]] = {}
    selected_ids: set[str] = set()
    gap_records: list[dict[str, Any]] = []
    for family in PROTOTYPE_FAMILIES:
        family_all = [item for item in descriptors if item.topology_family == family]
        eligible = [
            item for item in family_all
            if item.split == "train"
            and rig_records[item.rig_id]["rotation_provenance_status"] == "proven"
            and not any(
                REASON_CODES[code]["severity"] == "reject"
                for code in item.clip_reason_codes
            )
        ]
        selected = _round_robin_select(eligible, config.prototype_min_train_clips)
        selected_ids.update(selected)
        split_counter = Counter(item.split for item in family_all)
        record = {
            "family": family,
            "required_train_clips": config.prototype_min_train_clips,
            "rotation_proven_train_candidates": len(eligible),
            "selected_train_candidates": selected,
            "selected_count": len(selected),
            "all_current_clip_count": len(family_all),
            "split_counts": dict(sorted(split_counter.items())),
            "rig_ids": sorted({item.rig_id for item in family_all}),
            "status": (
                "available"
                if len(selected) >= config.prototype_min_train_clips
                else "shortage"
            ),
        }
        prototype_records[family] = record
        if record["status"] == "shortage":
            gap_records.append(
                {
                    "manifest_version": INVENTORY_VERSION,
                    "gap_id": f"prototype_train_shortage:{family}",
                    "family": family,
                    "status": "gap",
                    "reason_codes": ["PROTOTYPE_TRAIN_SHORTAGE"],
                    "required_train_clips": config.prototype_min_train_clips,
                    "available_rotation_proven_train_clips": len(eligible),
                    "shortage": config.prototype_min_train_clips - len(selected),
                    "all_current_clip_count": len(family_all),
                    "split_counts": dict(sorted(split_counter.items())),
                    "rig_ids": sorted({item.rig_id for item in family_all}),
                }
            )
    for descriptor in descriptors:
        descriptor.prototype_candidate = descriptor.clip_id in selected_ids

    exact_dragon = [item for item in descriptors if item.rig_id == "Dragon"]
    exact_dragon_train = [item for item in exact_dragon if item.split == "train"]
    if exact_dragon and not exact_dragon_train:
        gap_records.append(
            {
                "manifest_version": INVENTORY_VERSION,
                "gap_id": "exact_dragon_not_train_eligible",
                "family": "dragon_or_deep_topology",
                "status": "gap",
                "reason_codes": ["EXACT_DRAGON_NOT_TRAIN_ELIGIBLE"],
                "required_train_clips": config.prototype_min_train_clips,
                "exact_dragon_current_clips": len(exact_dragon),
                "exact_dragon_split_counts": dict(
                    sorted(Counter(item.split for item in exact_dragon).items())
                ),
                "deep_topology_selected_substitute_rigs": sorted(
                    {
                        descriptor_by_id[clip_id].rig_id
                        for clip_id in prototype_records[
                            "dragon_or_deep_topology"
                        ]["selected_train_candidates"]
                    }
                ),
            }
        )

    status_counts: Counter[str] = Counter()
    rotation_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    fps_counts: dict[str, Counter[str]] = defaultdict(Counter)
    sorted_descriptors = sorted(descriptors, key=lambda item: item.clip_id)
    for descriptor in sorted_descriptors:
        record = _clip_record(descriptor, rig_records[descriptor.rig_id])
        status_counts[record["status"]] += 1
        rotation_counts[record["rotation_provenance"]["status"]] += 1
        source_counts[descriptor.source_family] += 1
        family_counts[descriptor.topology_family][descriptor.split] += 1
        fps = record["source"].get("fps_src")
        if fps is not None:
            fps_counts[descriptor.source_family][f"{float(fps):.6f}"] += 1

    rig_status_counts = Counter(record["status"] for record in rig_records.values())
    parent_tree_hashes = {record["topology_parent_sha256"] for record in rig_records.values()}
    snapshot_rows = [
        [
            descriptor.clip_id,
            descriptor.btjd.file_size_bytes,
            descriptor.btjd.mtime_ns,
            descriptor.source_path,
            getattr(descriptor.source, "file_size_bytes", None),
            getattr(descriptor.source, "mtime_ns", None),
        ]
        for descriptor in sorted(descriptors, key=lambda item: item.clip_id)
    ]
    summary = {
        "manifest_version": INVENTORY_VERSION,
        "generated_at_utc": _datetime.datetime.now(
            _datetime.timezone.utc
        ).isoformat().replace("+00:00", "Z"),
        "evidence_mode": "fresh_live_filesystem_scan_not_archived_counts",
        "config": {
            "dataset_root": str(config.dataset_root),
            "split_root": str(config.split_root),
            "pz_bvh_root": str(config.pz_bvh_root),
            "truebones_raw_root": str(config.truebones_raw_root),
            "human272_root": str(config.human272_root),
            "prototype_min_train_clips": config.prototype_min_train_clips,
        },
        "fresh_counts": {
            "current_btjd_clips": len(descriptors),
            "current_rigs": len(rig_records),
            "current_unique_parent_trees": len(parent_tree_hashes),
            "current_max_physical_joints": max(
                len(record["joint_map"].get("btjd_joint_names", []))
                for record in rig_records.values()
            ),
            "split_counts": split_counts,
            "source_clip_counts": dict(sorted(source_counts.items())),
            "clip_status_counts": dict(sorted(status_counts.items())),
            "rig_status_counts": dict(sorted(rig_status_counts.items())),
            "rotation_provenance_counts": dict(sorted(rotation_counts.items())),
        },
        "live_snapshot_sha256": _sha256_json(snapshot_rows),
        "cond_sha256": _sha256_file(cond_path),
        "split_manifest_sha256": (
            _sha256_file(config.split_root / "splits_manifest.json")
            if (config.split_root / "splits_manifest.json").is_file()
            else None
        ),
        "native_fps_counts_by_source_family": {
            family: dict(sorted(counter.items()))
            for family, counter in sorted(fps_counts.items())
        },
        "topology_family_split_counts": {
            family: dict(sorted(counter.items()))
            for family, counter in sorted(family_counts.items())
        },
        "prototype_families": prototype_records,
        "prototype_gap_records": gap_records,
        "raw_source_split_audit": {
            "policy": "one live raw temporal source must belong to exactly one frozen split",
            "cross_split_source_count": len(source_split_audit),
            "affected_clip_count": sum(
                record["clip_count"] for record in source_split_audit
            ),
            "resolution": (
                "all affected slices reject RAW_SOURCE_SEQUENCE_SPLIT_OVERLAP and are "
                "ineligible for train calibration/prototype selection"
            ),
            "source_groups": source_split_audit,
        },
        "interpretation": {
            "acceptance_scope": (
                "rotation provenance and source timing inventory only; all clips remain review "
                "until T03 numeric source decode/FK and later heading/canonical visual gates"
            ),
            "planetzoo_limitation": (
                "processed BVH rotation channels are live, but native raw game BVHs are absent "
                "and the processed lineage includes per-clip initial-yaw canonicalization"
            ),
            "human_limitation": (
                "MotionStreamer272 rotations are live; the current BTJD per-clip alignment is "
                "not an admissible KTJD per-rig transform and will not be reused"
            ),
            "meter_claim": False,
        },
    }

    reason_payload = {
        "manifest_version": INVENTORY_VERSION,
        "codes": REASON_CODES,
    }
    candidate_payload = {
        "manifest_version": INVENTORY_VERSION,
        "selection": "deterministic rig-round-robin over rotation-proven frozen-train candidates",
        "families": prototype_records,
    }
    outputs: dict[str, Iterable[str] | str] = {
        "clips.jsonl": (
            _jsonl_line(_clip_record(descriptor, rig_records[descriptor.rig_id]))
            for descriptor in sorted_descriptors
        ),
        "rigs.jsonl": [
            _jsonl_line(rig_records[rig_id]) for rig_id in sorted(rig_records)
        ],
        "inventory_summary.json": json.dumps(
            summary, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        ) + "\n",
        "inventory_reason_codes.json": json.dumps(
            reason_payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        ) + "\n",
        "prototype_candidates.json": json.dumps(
            candidate_payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        ) + "\n",
        "prototype_gaps.jsonl": [_jsonl_line(record) for record in gap_records],
    }
    _write_transaction(config.output_root, outputs, overwrite=config.overwrite)
    print(
        "[inventory] wrote "
        f"{len(descriptors)} clips, {len(rig_records)} rigs -> {config.output_root}",
        flush=True,
    )
    return summary


def default_inventory_config(repo_root: str | Path = ".") -> InventoryConfig:
    root = Path(repo_root).expanduser().resolve()
    workspace = root.parent
    return InventoryConfig(
        dataset_root=root / "data" / "animo4d_L4TB_plus_human_v4b272neutral",
        split_root=root / "data" / "holdout_splits_v1",
        pz_bvh_root=root / "data" / "animo4d_anytop" / "bvhs",
        truebones_raw_root=(
            workspace
            / "Anytop"
            / "AnyTop"
            / "dataset"
            / "truebones"
            / "zoo"
            / "Truebone_Z-OO"
        ),
        human272_root=root / "scratch" / "humanml3d_272",
        output_root=root / "dataset" / "manifests",
        human_builder_path=root / "scripts" / "_v4_build_from_272.py",
        smpl_neutral_model_path=(
            workspace
            / "motion-latent-diffusion-main"
            / "datasets"
            / "humanml3d"
            / "body_models"
            / "smplh"
            / "neutral"
            / "model.npz"
        ),
        planetzoo_lineage_path=(
            workspace
            / "planetzoo-anytop-pipeline"
            / "docs"
            / "ANIMO4D_ANYTOP_DATA_LINEAGE.md"
        ),
    )
