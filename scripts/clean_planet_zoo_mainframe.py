#!/usr/bin/env python3
"""Build cleaned Planet Zoo AnyTop-style datasets.

The source dataset is never modified. Two cleaning levels are supported:

L1: remove end sites and facial-expression/detail controls.
L2: L1 plus ears and cosmetic/decorative appendages such as mane/crest/feather.

Both levels delete matched joints as whole subtrees, then slice the same joint
indices out of cond.npy and every motion tensor. Filenames and caption JSON keys
stay unchanged, so the existing AnyTopDataset caption path still works.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.anytop_dataset import _create_topology_edge_relations


L1_PATTERN = re.compile(
    r"end[_ ]?site|endsite|"
    r"jaw|mouth|lip|tongue|tooth|teeth|gum|chin|cheek|"
    r"eye|eyelid|brow|blink|pupil|iris|"
    r"nose|snout|muzzle|whisk|vibrissa",
    re.IGNORECASE,
)

L2_EXTRA_PATTERN = re.compile(
    r"(?:^|[_\W])ear(?:[_\W]|$)|"
    r"mane|crest|wattle|comb|beard|tuft|puff|skin|fur|"
    r"feather|plume|ruff|snood|dewlap",
    re.IGNORECASE,
)

COPY_TEXT_FILES = [
    "motion_texts_by_file_with_codex_drafts.json",
    "motion_texts_by_file_with_codex_drafts_summary.json",
    "motion_texts_by_file_with_codex_drafts_audit.json",
    "motion_texts_by_file_with_animosty4d_matches.json",
    "motion_text_match_summary.json",
    "metadata.txt",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        default="/scratch/ts1v23/workspace/noKslot_clean/data/anytop_planet_zoo",
        help="Original packed AnyTop-style Planet Zoo dataset root.",
    )
    ap.add_argument(
        "--output-parent",
        default="/scratch/ts1v23/workspace/noKslot_clean/data",
        help="Parent directory for anytop_planet_zoo_clean_L1/L2.",
    )
    ap.add_argument(
        "--levels",
        nargs="+",
        default=["L1", "L2"],
        choices=["L1", "L2"],
        help="Cleaning levels to generate.",
    )
    ap.add_argument(
        "--motion-dtype",
        choices=["source", "float32"],
        default="float32",
        help="Saved motion dtype. float32 is enough for the local training loader.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing clean output directories before writing.",
    )
    return ap.parse_args()


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
    return remove


def level_pattern(level: str) -> re.Pattern:
    if level == "L1":
        return L1_PATTERN
    if level == "L2":
        return re.compile(
            f"(?:{L1_PATTERN.pattern})|(?:{L2_EXTRA_PATTERN.pattern})",
            re.IGNORECASE,
        )
    raise ValueError(level)


def build_clean_index(parents: np.ndarray, names: list[str], level: str) -> dict:
    pat = level_pattern(level)
    children = children_from_parents(parents)
    seeds = [i for i, name in enumerate(names) if pat.search(str(name))]
    remove = subtree_descendants(children, seeds)
    if 0 in remove:
        raise RuntimeError("cleaning rule would remove root joint 0")
    keep = [i for i in range(len(parents)) if i not in remove]
    old_to_new = {old: new for new, old in enumerate(keep)}
    new_parents = []
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
    return {
        "keep": keep,
        "remove": sorted(remove),
        "seeds": seeds,
        "parents": np.asarray(new_parents, dtype=np.int32),
    }


def kinematic_chains_from_parents(parents: np.ndarray) -> list[list[int]]:
    children = children_from_parents(parents)
    leaves = [i for i, ch in enumerate(children) if not ch]
    chains: list[list[int]] = []
    for leaf in leaves:
        cur = leaf
        path = []
        while cur >= 0:
            path.append(cur)
            cur = int(parents[cur])
        chains.append(list(reversed(path)))
    return chains


def clean_cond_entry(entry: dict, level: str) -> tuple[dict, dict]:
    parents = np.asarray(entry["parents"], dtype=np.int32)
    names = [str(n) for n in entry["joints_names"]]
    idx = build_clean_index(parents, names, level)
    keep = np.asarray(idx["keep"], dtype=np.int64)
    new_parents = idx["parents"]
    new_names = [names[i] for i in keep.tolist()]
    joint_rel, graph_dist = _create_topology_edge_relations(new_parents)
    cleaned = dict(entry)
    cleaned["parents"] = new_parents
    cleaned["offsets"] = np.asarray(entry["offsets"])[keep].astype(np.float32)
    cleaned["tpos_first_frame"] = (
        np.asarray(entry["tpos_first_frame"])[keep].astype(np.float32)
    )
    cleaned["mean"] = np.asarray(entry["mean"])[keep].astype(np.float32)
    cleaned["std"] = np.asarray(entry["std"])[keep].astype(np.float32)
    cleaned["joints_names"] = new_names
    cleaned["joint_relations"] = joint_rel.astype(np.float32)
    cleaned["joints_graph_dist"] = graph_dist.astype(np.float32)
    cleaned["kinematic_chains"] = kinematic_chains_from_parents(new_parents)
    manifest = {
        "original_joints": int(len(parents)),
        "cleaned_joints": int(len(new_parents)),
        "removed_joints": int(len(idx["remove"])),
        "seed_joints": int(len(idx["seeds"])),
        "keep_indices": idx["keep"],
        "removed_indices": idx["remove"],
        "removed_joint_names": [names[i] for i in idx["remove"]],
        "seed_joint_names": [names[i] for i in idx["seeds"]],
    }
    return cleaned, manifest


def longest_prefix_match(fname: str, keys: list[str]) -> str | None:
    for key in keys:
        if fname.startswith(f"{key}_"):
            return key
    return None


def load_original_object_index(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            out[row["object_name"]] = row
    return out


def write_object_index(path: Path, rows: list[dict]) -> None:
    fieldnames = ["object_name", "source_dir", "motions", "bvhs", "animations", "joints", "frames"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_dataset_info(out_root: Path, level: str, summary: dict) -> None:
    text = f"""# AnyTopo Planet Zoo Clean {level}

Generated from:

```text
{summary['source_root']}
```

This is a derived main-motion dataset. The original dataset is not modified.

## Cleaning Rule

- L1 removes end-site joints and facial-expression/detail controls:
  jaw, mouth, lip, tongue, teeth, cheek, eye, eyelid, brow, nose, snout,
  muzzle, whisker, and descendants of matched joints.
- L2 applies L1 and additionally removes ear/cosmetic/decorative branches:
  ears, mane, crest, wattle, comb, beard, tuft, puff, skin, fur, feather,
  plume, ruff, snood, dewlap, and descendants.

Deletion is subtree-based. Kept joints keep their original order, and motion
tensors are sliced on the joint axis with the same kept indices.

## Counts

| Item | Count |
|---|---:|
| Objects | {summary['objects']} |
| Motion `.npy` files | {summary['motions']} |
| Total frames | {summary['frames']} |
| Original max joints | {summary['orig_max_joints']} |
| Clean max joints | {summary['clean_max_joints']} |
| Clean median joints | {summary['clean_median_joints']} |
| Removed joints total | {summary['removed_joints']} |

## Layout Notes

- `motions/` contains cleaned AnyTop-style tensors with unchanged filenames.
- `cond.npy` contains the matching cleaned per-object topology and statistics.
- Caption JSON files are copied unchanged because motion filenames are unchanged.
- `bvhs/` is intentionally not regenerated; original BVHs would not match the
  cleaned joint axis. Use `data/anytop_planet_zoo_clean_visual_qa/` renders for
  inspection.
"""
    (out_root / "DATASET_INFO.md").write_text(text)


def copy_metadata_files(src: Path, dst: Path) -> None:
    for name in COPY_TEXT_FILES:
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)


def build_level(src_root: Path, out_parent: Path, level: str, motion_dtype: str, overwrite: bool) -> None:
    out_root = out_parent / f"anytop_planet_zoo_clean_{level}"
    if out_root.exists():
        if not overwrite:
            raise FileExistsError(f"{out_root} exists; pass --overwrite")
        shutil.rmtree(out_root)
    (out_root / "motions").mkdir(parents=True)
    (out_root / "animations").mkdir()
    (out_root / "bvhs").mkdir()

    cond = np.load(src_root / "cond.npy", allow_pickle=True).item()
    clean_cond: dict[str, dict] = {}
    manifest: dict[str, dict] = {}
    for obj, entry in cond.items():
        clean_entry, info = clean_cond_entry(entry, level)
        clean_cond[obj] = clean_entry
        manifest[obj] = info
    np.save(out_root / "cond.npy", clean_cond, allow_pickle=True)

    keys = sorted(clean_cond.keys(), key=lambda k: -len(k))
    keep_by_obj = {
        obj: np.asarray(manifest[obj]["keep_indices"], dtype=np.int64)
        for obj in clean_cond
    }
    motion_counts = Counter()
    frame_counts = Counter()
    shape_counts = Counter()
    dtype_counts = Counter()
    unmatched = []
    motions_src = src_root / "motions"
    motions_dst = out_root / "motions"
    for i, motion_path in enumerate(sorted(motions_src.glob("*.npy")), start=1):
        obj = longest_prefix_match(motion_path.name, keys)
        if obj is None:
            unmatched.append(motion_path.name)
            continue
        arr = np.load(motion_path, mmap_mode="r")
        sliced = np.asarray(arr[:, keep_by_obj[obj], :])
        if motion_dtype == "float32":
            sliced = sliced.astype(np.float32, copy=False)
        np.save(motions_dst / motion_path.name, sliced, allow_pickle=False)
        motion_counts[obj] += 1
        frame_counts[obj] += int(sliced.shape[0])
        shape_counts[tuple(int(v) for v in sliced.shape)] += 1
        dtype_counts[str(sliced.dtype)] += 1
        if i % 5000 == 0:
            print(f"[{level}] wrote {i} motions", flush=True)
    if unmatched:
        raise RuntimeError(f"{len(unmatched)} unmatched motions, first={unmatched[:5]}")

    original_index = load_original_object_index(src_root / "object_index.csv")
    rows = []
    for obj in sorted(clean_cond):
        orig = original_index.get(obj, {})
        rows.append(
            {
                "object_name": obj,
                "source_dir": orig.get("source_dir", ""),
                "motions": int(motion_counts[obj]),
                "bvhs": 0,
                "animations": 0,
                "joints": int(manifest[obj]["cleaned_joints"]),
                "frames": int(frame_counts[obj]),
            }
        )
    write_object_index(out_root / "object_index.csv", rows)

    copy_metadata_files(src_root, out_root)
    (out_root / "clean_filter_manifest.json").write_text(
        json.dumps(
            {
                "level": level,
                "source_root": str(src_root),
                "motion_dtype": motion_dtype,
                "objects": manifest,
            },
            indent=2,
        )
    )
    total_frames = int(sum(frame_counts.values()))
    clean_j = [int(v["cleaned_joints"]) for v in manifest.values()]
    orig_j = [int(v["original_joints"]) for v in manifest.values()]
    summary = {
        "source_root": str(src_root),
        "level": level,
        "objects": len(clean_cond),
        "motions": int(sum(motion_counts.values())),
        "frames": total_frames,
        "orig_max_joints": int(max(orig_j)),
        "clean_max_joints": int(max(clean_j)),
        "clean_median_joints": int(np.median(clean_j)),
        "removed_joints": int(sum(v["removed_joints"] for v in manifest.values())),
        "shape_counts_top10": [
            {"shape": list(k), "count": int(v)}
            for k, v in shape_counts.most_common(10)
        ],
        "dtype_counts": dict(dtype_counts),
    }
    (out_root / "pack_summary.json").write_text(json.dumps(summary, indent=2))
    write_dataset_info(out_root, level, summary)
    (out_root / "README.md").write_text(
        f"Derived cleaned Planet Zoo dataset ({level}). See DATASET_INFO.md.\n"
    )
    print(f"[{level}] done -> {out_root}", flush=True)


def main() -> int:
    args = parse_args()
    src_root = Path(args.source)
    out_parent = Path(args.output_parent)
    if not (src_root / "cond.npy").exists() or not (src_root / "motions").is_dir():
        raise SystemExit(f"source is not an AnyTop-style dataset root: {src_root}")
    for level in args.levels:
        build_level(src_root, out_parent, level, args.motion_dtype, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
