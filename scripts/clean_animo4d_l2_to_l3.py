#!/usr/bin/env python3
"""Derive stricter AniMo4D AnyTop clean datasets from an existing clean root.

The source root is read-only and the output root is written separately. The
default mode derives L3 from L2. ``--cleaning-level L4_safe`` derives a more
compact safe-clean set from L3 by removing residual skin/volume helper leaves.
``--cleaning-level L5`` derives a body-only-foot set from L4-safe by removing
toe/claw/hoof terminal branches while keeping foot joints.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.anytop_dataset import _create_topology_edge_relations
from src.models.graph_salad.pool_edge_segment import _build_segments_rulebased


STATS_FLOOR = 1e-6

SUBTREE_HELPER_RE = re.compile(
    r"twist|breath|food|anus|"
    r"jowl|nostril|snout|muzzle|mouth|jaw|lip|tongue|tooth|teeth|"
    r"cheek|eye|brow|whisk|vibrissa|"
    r"helper|joey|squash|ikblend|"
    r"(?:^|[_\W])ear[0-9a-z]*|def_ear[0-9a-z]*",
    re.IGNORECASE,
)

LEAF_HELPER_RE = re.compile(
    r"(?:^|[_\W])l[0-9](?:[_\W]|$)|_[lL][0-9]$|"
    r"(?:_end|\.end|end$)",
    re.IGNORECASE,
)

L4_SAFE_SUBTREE_HELPER_RE = re.compile(
    r"volume|fold|scale|bubble|pouch|sternum|throat|wobble",
    re.IGNORECASE,
)

L5_SUBTREE_HELPER_RE = re.compile(
    r"toe|claw|hoof",
    re.IGNORECASE,
)

EMPTY_HELPER_RE = re.compile(r"a^")

COPY_TOPLEVEL_FILES = [
    "motion_text_manifest.csv",
    "motion_text_manifest.json",
    "motion_text_manifest.jsonl",
    "motion_texts_by_file_with_animo4d_official.json",
    "motion_texts_by_file_with_animosty4d_matches.json",
    "motion_texts_by_file_with_codex_drafts.json",
    "motion_texts_by_file_with_codex_drafts_summary.json",
    "metadata.txt",
    "missing_processed_current.jsonl",
    "pack_manifest.jsonl",
]

_WORK_KEYS: list[str] = []
_WORK_KEEP: dict[str, np.ndarray] = {}
_WORK_SRC_MOTIONS: Path | None = None
_WORK_DST_MOTIONS: Path | None = None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/animo4d_anytop_clean_L2")
    ap.add_argument("--output", default="data/animo4d_anytop_clean_L3")
    ap.add_argument(
        "--cleaning-level",
        choices=["L3", "L4_safe", "L5"],
        default="L3",
        help="L3: remove helper/deformer/facial leftovers from L2. "
             "L4_safe: remove residual skin/volume helpers from L3. "
             "L5: remove toe/claw/hoof branches from L4-safe, keeping foot.",
    )
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report-every", type=int, default=5000)
    return ap.parse_args()


def cleaning_patterns(level: str) -> tuple[re.Pattern, re.Pattern]:
    if level == "L3":
        return SUBTREE_HELPER_RE, LEAF_HELPER_RE
    if level == "L4_safe":
        return L4_SAFE_SUBTREE_HELPER_RE, EMPTY_HELPER_RE
    if level == "L5":
        return L5_SUBTREE_HELPER_RE, EMPTY_HELPER_RE
    raise ValueError(level)


def level_label(level: str) -> str:
    if level == "L4_safe":
        return "L4-safe"
    if level == "L5":
        return "L5"
    return level


def children_from_parents(parents: np.ndarray) -> list[list[int]]:
    children: list[list[int]] = [[] for _ in range(len(parents))]
    for j, p in enumerate(parents.tolist()):
        if p >= 0:
            children[int(p)].append(j)
    return children


def subtree_descendants(children: list[list[int]], seeds: Iterable[int]) -> set[int]:
    remove: set[int] = set()
    stack = list(seeds)
    while stack:
        j = stack.pop()
        if j in remove:
            continue
        remove.add(j)
        stack.extend(children[j])
    remove.discard(0)
    return remove


def remap_parents(parents: np.ndarray, keep: list[int]) -> np.ndarray:
    old_to_new = {old: new for new, old in enumerate(keep)}
    new_parents: list[int] = []
    for old in keep:
        p = int(parents[old])
        if p < 0:
            new_parents.append(-1)
        elif p in old_to_new:
            new_parents.append(old_to_new[p])
        else:
            raise RuntimeError(
                f"kept joint {old} has removed parent {p}; subtree delete failed"
            )
    for j, p in enumerate(new_parents):
        if j == 0 and p != -1:
            raise RuntimeError(f"root parent must be -1, got {p}")
        if j > 0 and not (0 <= p < j):
            raise RuntimeError(f"parents must be root-first; parents[{j}]={p}")
    return np.asarray(new_parents, dtype=np.int32)


def kinematic_chains_from_parents(parents: np.ndarray) -> list[list[int]]:
    children = children_from_parents(parents)
    leaves = [i for i, ch in enumerate(children) if not ch]
    chains: list[list[int]] = []
    for leaf in leaves:
        cur = leaf
        path: list[int] = []
        while cur >= 0:
            path.append(cur)
            cur = int(parents[cur])
        chains.append(list(reversed(path)))
    return chains


def build_clean_index(parents: np.ndarray, names: list[str], level: str) -> dict:
    subtree_re, leaf_re = cleaning_patterns(level)
    children = children_from_parents(parents)
    subtree_seeds: list[int] = []
    leaf_seeds: list[int] = []
    for i, name in enumerate(names):
        if i == 0:
            continue
        if subtree_re.search(name):
            subtree_seeds.append(i)
        elif leaf_re.search(name) and not children[i]:
            leaf_seeds.append(i)
    remove = subtree_descendants(children, subtree_seeds) | set(leaf_seeds)
    if 0 in remove:
        raise RuntimeError(f"{level_label(level)} rule would remove root joint 0")
    keep = [i for i in range(len(parents)) if i not in remove]
    new_parents = remap_parents(parents, keep)
    return {
        "keep": keep,
        "remove": sorted(remove),
        "subtree_seeds": subtree_seeds,
        "leaf_seeds": leaf_seeds,
        "parents": new_parents,
    }


def clean_cond_entry(entry: dict, level: str) -> tuple[dict, dict]:
    parents = np.asarray(entry["parents"], dtype=np.int32)
    names = [str(n) for n in np.asarray(entry["joints_names"]).tolist()]
    idx = build_clean_index(parents, names, level)
    keep = np.asarray(idx["keep"], dtype=np.int64)
    new_parents = idx["parents"]
    joint_rel, graph_dist = _create_topology_edge_relations(new_parents)

    cleaned = dict(entry)
    cleaned["parents"] = new_parents
    cleaned["offsets"] = np.asarray(entry["offsets"])[keep].astype(np.float32)
    cleaned["tpos_first_frame"] = (
        np.asarray(entry["tpos_first_frame"])[keep].astype(np.float32)
    )
    cleaned["mean"] = np.asarray(entry["mean"])[keep].astype(np.float32)
    cleaned["std"] = np.asarray(entry["std"])[keep].astype(np.float32)
    cleaned["joints_names"] = [names[i] for i in keep.tolist()]
    cleaned["joint_relations"] = joint_rel.astype(np.float32)
    cleaned["joints_graph_dist"] = graph_dist.astype(np.float32)
    cleaned["kinematic_chains"] = kinematic_chains_from_parents(new_parents)

    manifest = {
        "source_joints": int(len(parents)),
        "cleaned_joints": int(len(new_parents)),
        "removed_joints": int(len(idx["remove"])),
        "subtree_seed_joints": int(len(idx["subtree_seeds"])),
        "leaf_seed_joints": int(len(idx["leaf_seeds"])),
        "keep_indices": idx["keep"],
        "removed_indices": idx["remove"],
        "removed_joint_names": [names[i] for i in idx["remove"]],
        "subtree_seed_joint_names": [names[i] for i in idx["subtree_seeds"]],
        "leaf_seed_joint_names": [names[i] for i in idx["leaf_seeds"]],
    }
    return cleaned, manifest


def longest_prefix_match(fname: str, keys: list[str]) -> str | None:
    for key in keys:
        if fname.startswith(f"{key}_"):
            return key
    return None


def worker_init(src_motions: str, dst_motions: str, keys: list[str],
                keep_by_obj: dict[str, np.ndarray]) -> None:
    global _WORK_KEYS, _WORK_KEEP, _WORK_SRC_MOTIONS, _WORK_DST_MOTIONS
    _WORK_KEYS = keys
    _WORK_KEEP = keep_by_obj
    _WORK_SRC_MOTIONS = Path(src_motions)
    _WORK_DST_MOTIONS = Path(dst_motions)


def process_motion_file(name: str) -> dict:
    if _WORK_SRC_MOTIONS is None or _WORK_DST_MOTIONS is None:
        raise RuntimeError("worker not initialized")
    obj = longest_prefix_match(name, _WORK_KEYS)
    if obj is None:
        raise ValueError(f"motion did not match any object prefix: {name}")
    src = _WORK_SRC_MOTIONS / name
    dst = _WORK_DST_MOTIONS / name
    keep = _WORK_KEEP[obj]
    arr = np.load(src, mmap_mode="r")
    sliced = np.asarray(arr[:, keep, :], dtype=np.float32)
    if sliced.ndim != 3 or sliced.shape[2] != 13:
        raise ValueError(f"bad sliced shape for {src}: {sliced.shape}")
    finite = bool(np.isfinite(sliced).all())
    np.save(dst, sliced, allow_pickle=False)
    arr64 = sliced.astype(np.float64, copy=False)
    return {
        "obj": obj,
        "frames": int(sliced.shape[0]),
        "shape": tuple(int(v) for v in sliced.shape),
        "dtype": str(sliced.dtype),
        "finite": finite,
        "max_abs": float(np.nanmax(np.abs(sliced))) if sliced.size else 0.0,
        "sum": arr64.sum(axis=0),
        "sumsq": np.square(arr64).sum(axis=0),
    }


def copy_metadata(src_root: Path, out_root: Path) -> None:
    for name in COPY_TOPLEVEL_FILES:
        src = src_root / name
        if src.exists():
            shutil.copy2(src, out_root / name)

    src_anim = src_root / "animations"
    if src_anim.is_dir():
        dst_anim = out_root / "animations"
        dst_anim.mkdir(exist_ok=True)
        for path in src_anim.iterdir():
            if path.is_file():
                shutil.copy2(path, dst_anim / path.name)

    # Keep provenance without copying multi-GB removed motion payloads.
    removed = src_root / "proximal_rotation_removed_20260608"
    if removed.is_dir():
        dst = out_root / "source_metadata" / "proximal_rotation_removed_20260608"
        dst.mkdir(parents=True, exist_ok=True)
        for path in removed.iterdir():
            if path.is_file():
                shutil.copy2(path, dst / path.name)

    src_summary = src_root / "pack_summary.json"
    if src_summary.exists():
        (out_root / "source_metadata").mkdir(exist_ok=True)
        shutil.copy2(src_summary, out_root / "source_metadata" / f"{src_root.name}_pack_summary.json")
    src_info = src_root / "DATASET_INFO.md"
    if src_info.exists():
        (out_root / "source_metadata").mkdir(exist_ok=True)
        shutil.copy2(src_info, out_root / "source_metadata" / f"{src_root.name}_DATASET_INFO.md")


def load_object_index(path: Path) -> tuple[list[dict], list[str]]:
    if not path.exists():
        return [], ["object_name", "source_dir", "motions", "bvhs", "animations", "joints", "frames"]
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames or []


def write_object_index(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    if not fieldnames:
        fieldnames = ["object_name", "source_dir", "motions", "bvhs", "animations", "joints", "frames"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_dataset_info(out_root: Path, summary: dict) -> None:
    label = level_label(str(summary["level"]))
    if summary["level"] == "L3":
        rule_text = """L3 keeps the L2 main body skeleton and additionally removes likely skinning /
deformation helper joints:

- limb twist helpers: `*Twist`, `*HalfTwist`, `*AllTwist`
- breathing / food / anus helper leaves
- leftover facial-detail leaves: jowl, nostril, snout, muzzle, jaw, lip, tongue,
  teeth, cheek, eye, brow, whisker/vibrissa
- helper/squash/IKBlend controls
- remaining numbered ear chains missed by the older L2 regex
- joey auxiliary mini-skeletons
- leaf-only `L0`-style locators and leaf-only `_end` joints

The rule deliberately does **not** remove `horselink`, `foot`, `toe`, `claw`,
`tail`, or `trunk` body joints because those carry subject motion or species
shape."""
    elif summary["level"] == "L4_safe":
        rule_text = """L4-safe is derived from L3 and removes only residual
skin/volume helper leaves:

- `*Volume*`, `*Fold*`, `*Scale*`
- `*Bubble*`, `*Pouch*`, `*Sternum*`, `*Throat*`
- `*Wobble*`

The rule deliberately does **not** remove `horselink`, `foot`, `toe`, `claw`,
`tail`, or `trunk` body joints."""
    else:
        rule_text = """L5 is derived from L4-safe and removes terminal foot
detail branches:

- `*toe*`
- `*claw*`
- `*hoof*`

The rule deliberately keeps `frontFoot`, `rearFoot`, and `horselink` joints as
the limb endpoints. It also keeps `tail` and `trunk` body chains."""

    text = f"""# AniMo4D AnyTop Clean {label}

Generated from:

```text
{summary['source_root']}
```

{label} is a derived dataset; the source root is not modified.

## Cleaning Rule

{rule_text}

Kept joints preserve source order. Motion filenames and caption metadata remain
unchanged. `cond.npy` topology and per-object mean/std are recomputed from the
{label} motion tensors. `std` is floored at {summary['std_floor']}.

## Counts

| Item | Count |
|---|---:|
| Objects | {summary['objects']} |
| Motion `.npy` files | {summary['motions']} |
| Total frames | {summary['frames']} |
| Source max joints | {summary['source_max_joints']} |
| {label} max joints | {summary['clean_max_joints']} |
| {label} min joints | {summary['clean_min_joints']} |
| {label} mean joints | {summary['clean_mean_joints']:.2f} |
| {label} median joints | {summary['clean_median_joints']} |
| Removed joints total vs source | {summary['removed_joints']} |
| Nonfinite motion files while writing | {summary['nonfinite_motion_files']} |
| Max abs in {label} motions | {summary['motion_absmax_max']:.6f} |

## Layout Notes

- `motions/` contains {label} AnyTop 13ch tensors with unchanged filenames.
- `cond.npy` contains matching {label} topology and recomputed statistics.
- `bvhs/` is intentionally empty: original BVHs do not match the cleaned joint
  axis.
- Source provenance metadata is under `source_metadata/`.
"""
    (out_root / "DATASET_INFO.md").write_text(text)
    (out_root / "README.md").write_text(
        f"Derived cleaned AniMo4D AnyTop dataset ({label}). See DATASET_INFO.md.\n"
    )


def percentile(values: list[int] | list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


def summarize_segments(cond: dict[str, dict]) -> dict:
    counts = []
    for sk in cond.values():
        parents = np.asarray(sk["parents"], dtype=np.int64)
        counts.append(len(_build_segments_rulebased(parents, max_segments=10000)))
    return {
        "natural_edge_segment_p2_min": int(min(counts)),
        "natural_edge_segment_p2_max": int(max(counts)),
        "natural_edge_segment_p2_mean": float(np.mean(counts)),
        "natural_edge_segment_p2_median": float(np.median(counts)),
        "natural_edge_segment_p2_p95": percentile(counts, 95),
        "natural_edge_segment_p2_p99": percentile(counts, 99),
        "natural_edge_segment_p2_gt64": int(sum(c > 64 for c in counts)),
        "natural_edge_segment_p2_gt96": int(sum(c > 96 for c in counts)),
        "natural_edge_segment_p2_gt128": int(sum(c > 128 for c in counts)),
    }


def build_clean_dataset(src_root: Path, out_root: Path, level: str, workers: int,
                        overwrite: bool, dry_run: bool, report_every: int) -> dict:
    t0 = time.time()
    label = level_label(level)
    if not (src_root / "cond.npy").exists() or not (src_root / "motions").is_dir():
        raise SystemExit(f"source is not an AnyTop-style dataset root: {src_root}")
    if out_root.exists() and not (overwrite or dry_run):
        raise FileExistsError(f"{out_root} exists; pass --overwrite")

    src_cond = np.load(src_root / "cond.npy", allow_pickle=True).item()
    clean_cond: dict[str, dict] = {}
    manifest: dict[str, dict] = {}
    for obj, entry in src_cond.items():
        clean_entry, info = clean_cond_entry(entry, level)
        clean_cond[obj] = clean_entry
        manifest[obj] = info

    source_j = [int(v["source_joints"]) for v in manifest.values()]
    clean_j = [int(v["cleaned_joints"]) for v in manifest.values()]
    removed_total = int(sum(v["removed_joints"] for v in manifest.values()))

    if dry_run:
        summary = {
            "source_root": str(src_root),
            "output_root": str(out_root),
            "level": level,
            "objects": len(clean_cond),
            "source_min_joints": int(min(source_j)),
            "source_max_joints": int(max(source_j)),
            "source_mean_joints": float(np.mean(source_j)),
            "source_median_joints": float(np.median(source_j)),
            "clean_min_joints": int(min(clean_j)),
            "clean_max_joints": int(max(clean_j)),
            "clean_mean_joints": float(np.mean(clean_j)),
            "clean_median_joints": float(np.median(clean_j)),
            "removed_joints": removed_total,
            **summarize_segments(clean_cond),
        }
        print(json.dumps(summary, indent=2))
        return summary

    if out_root.exists() and overwrite:
        shutil.rmtree(out_root)
    (out_root / "motions").mkdir(parents=True)
    (out_root / "bvhs").mkdir()
    (out_root / "animations").mkdir()
    (out_root / "source_metadata").mkdir()

    np.save(out_root / "cond.npy", clean_cond, allow_pickle=True)
    copy_metadata(src_root, out_root)

    keys = sorted(clean_cond.keys(), key=lambda k: -len(k))
    keep_by_obj = {
        obj: np.asarray(manifest[obj]["keep_indices"], dtype=np.int64)
        for obj in clean_cond
    }
    sums = {
        obj: np.zeros((int(manifest[obj]["cleaned_joints"]), 13), dtype=np.float64)
        for obj in clean_cond
    }
    sumsqs = {obj: np.zeros_like(arr) for obj, arr in sums.items()}
    motion_counts: Counter[str] = Counter()
    frame_counts: Counter[str] = Counter()
    shape_counts: Counter[tuple[int, int, int]] = Counter()
    dtype_counts: Counter[str] = Counter()
    nonfinite = 0
    max_abs = 0.0

    motion_names = sorted(path.name for path in (src_root / "motions").glob("*.npy"))
    if workers <= 1:
        worker_init(str(src_root / "motions"), str(out_root / "motions"), keys, keep_by_obj)
        iterator = (process_motion_file(name) for name in motion_names)
        for i, result in enumerate(iterator, start=1):
            obj = result["obj"]
            sums[obj] += result["sum"]
            sumsqs[obj] += result["sumsq"]
            motion_counts[obj] += 1
            frame_counts[obj] += int(result["frames"])
            shape_counts[result["shape"]] += 1
            dtype_counts[result["dtype"]] += 1
            nonfinite += 0 if result["finite"] else 1
            max_abs = max(max_abs, float(result["max_abs"]))
            if report_every and i % report_every == 0:
                print(f"[{label}] wrote {i}/{len(motion_names)} motions", flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=worker_init,
            initargs=(str(src_root / "motions"), str(out_root / "motions"), keys, keep_by_obj),
        ) as ex:
            futures = [ex.submit(process_motion_file, name) for name in motion_names]
            for i, fut in enumerate(as_completed(futures), start=1):
                result = fut.result()
                obj = result["obj"]
                sums[obj] += result["sum"]
                sumsqs[obj] += result["sumsq"]
                motion_counts[obj] += 1
                frame_counts[obj] += int(result["frames"])
                shape_counts[result["shape"]] += 1
                dtype_counts[result["dtype"]] += 1
                nonfinite += 0 if result["finite"] else 1
                max_abs = max(max_abs, float(result["max_abs"]))
                if report_every and i % report_every == 0:
                    print(f"[{label}] wrote {i}/{len(motion_names)} motions", flush=True)

    for obj in sorted(clean_cond):
        n = frame_counts[obj]
        if n <= 0:
            raise RuntimeError(f"object has no frames: {obj}")
        mean = sums[obj] / float(n)
        var = np.maximum(sumsqs[obj] / float(n) - np.square(mean), 0.0)
        std = np.maximum(np.sqrt(var), STATS_FLOOR)
        clean_cond[obj]["mean"] = mean.astype(np.float32)
        clean_cond[obj]["std"] = std.astype(np.float32)
    np.save(out_root / "cond.npy", clean_cond, allow_pickle=True)

    index_rows, fieldnames = load_object_index(src_root / "object_index.csv")
    if index_rows:
        for row in index_rows:
            obj = row.get("object_name", "")
            if obj in manifest:
                row["motions"] = str(int(motion_counts[obj]))
                row["bvhs"] = "0"
                row["animations"] = "0"
                row["joints"] = str(int(manifest[obj]["cleaned_joints"]))
                row["frames"] = str(int(frame_counts[obj]))
    else:
        fieldnames = ["object_name", "source_dir", "motions", "bvhs", "animations", "joints", "frames"]
        index_rows = [
            {
                "object_name": obj,
                "source_dir": "",
                "motions": str(int(motion_counts[obj])),
                "bvhs": "0",
                "animations": "0",
                "joints": str(int(manifest[obj]["cleaned_joints"])),
                "frames": str(int(frame_counts[obj])),
            }
            for obj in sorted(clean_cond)
        ]
    write_object_index(out_root / "object_index.csv", index_rows, fieldnames)

    summary = {
        "source_root": str(src_root),
        "level": level,
        "objects": len(clean_cond),
        "motions": int(sum(motion_counts.values())),
        "frames": int(sum(frame_counts.values())),
        "source_min_joints": int(min(source_j)),
        "source_max_joints": int(max(source_j)),
        "source_mean_joints": float(np.mean(source_j)),
        "source_median_joints": float(np.median(source_j)),
        "clean_min_joints": int(min(clean_j)),
        "clean_max_joints": int(max(clean_j)),
        "clean_mean_joints": float(np.mean(clean_j)),
        "clean_median_joints": float(np.median(clean_j)),
        "removed_joints": removed_total,
        "std_floor": STATS_FLOOR,
        "nonfinite_motion_files": int(nonfinite),
        "motion_absmax_max": float(max_abs),
        "shape_counts_top20": [
            {"shape": list(k), "count": int(v)}
            for k, v in shape_counts.most_common(20)
        ],
        "dtype_counts": dict(dtype_counts),
        "elapsed_sec": round(time.time() - t0, 2),
        **summarize_segments(clean_cond),
    }
    (out_root / "clean_filter_manifest.json").write_text(
        json.dumps(
            {
                "level": level,
                "source_root": str(src_root),
                "rule": {
                    "subtree_helper_regex": cleaning_patterns(level)[0].pattern,
                    "leaf_helper_regex": cleaning_patterns(level)[1].pattern,
                    "notes": [
                        "source root is not modified.",
                        "horselink/foot/tail/trunk body joints are not targeted.",
                        "L3 mode removes L0-style locators and _end joints only when leaf-only.",
                        "L4_safe mode only removes residual skin/volume helpers.",
                        "L5 mode removes toe/claw/hoof terminal branches while keeping foot.",
                    ],
                },
                "std_floor": STATS_FLOOR,
                "objects": manifest,
            },
            indent=2,
        )
    )
    (out_root / "pack_summary.json").write_text(json.dumps(summary, indent=2))
    write_dataset_info(out_root, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> int:
    args = parse_args()
    workers = max(1, int(args.workers))
    return 0 if build_clean_dataset(
        Path(args.source),
        Path(args.output),
        level=args.cleaning_level,
        workers=workers,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        report_every=args.report_every,
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
