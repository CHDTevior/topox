#!/usr/bin/env python3
"""Move Planet Zoo velocity-risk clips aside and recompute cond mean/std.

The dataset root names stay unchanged. Risk clips keep their original filenames
under ``risk_files/`` so training code can keep reading the same dataset roots.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np


DEFAULT_RISK_CSV = (
    "runs/diagnostics/planetzoo_p9999_abslt100_L2/"
    "L2_velocity_risk_absge100_or_gt22p53.csv"
)
DEFAULT_DATASETS = [
    "data/anytop_planet_zoo",
    "data/anytop_planet_zoo_clean_L1",
    "data/anytop_planet_zoo_clean_L2",
]
STATS_FIELDS = ("mean", "std")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--risk-csv", default=DEFAULT_RISK_CSV)
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def load_risk_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty risk csv: {path}")
    return rows


def longest_prefix_match(fname: str, keys: list[str]) -> str | None:
    for key in keys:
        if fname.startswith(f"{key}_"):
            return key
    return None


def move_if_exists(src: Path, dst: Path, dry_run: bool) -> str:
    if src.exists():
        if dst.exists():
            raise FileExistsError(f"destination already exists: {dst}")
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        return "moved"
    if dst.exists():
        return "already_in_risk"
    return "missing"


def copy_if_exists(src: Path, dst: Path, dry_run: bool) -> bool:
    if not src.exists():
        return False
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return True


def backup_once(src: Path, dst: Path, dry_run: bool) -> bool:
    if not src.exists() or dst.exists():
        return False
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return True


def write_csv(path: Path, rows: list[dict], dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def filter_caption_json(root: Path, risk_files: set[str], risk_dir: Path, dry_run: bool) -> dict:
    result = {}
    for name in (
        "motion_texts_by_file_with_codex_drafts.json",
        "motion_texts_by_file_with_animosty4d_matches.json",
    ):
        path = root / name
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        removed = {k: data[k] for k in risk_files if k in data}
        if not removed:
            result[name] = {"removed": 0, "kept": len(data)}
            continue
        kept = {k: v for k, v in data.items() if k not in risk_files}
        if not dry_run:
            backup_once(path, risk_dir / "metadata_before_filter" / name, dry_run=False)
            (risk_dir / "metadata").mkdir(parents=True, exist_ok=True)
            (risk_dir / "metadata" / name).write_text(json.dumps(removed, indent=2))
            path.write_text(json.dumps(kept, indent=2))
        result[name] = {"removed": len(removed), "kept": len(kept)}
    return result


def filter_motion_manifest_json(root: Path, risk_files: set[str], risk_dir: Path, dry_run: bool) -> dict:
    path = root / "motion_text_manifest.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict) or not isinstance(data.get("rows"), list):
        return {}
    rows = data["rows"]
    kept = []
    removed = []
    for row in rows:
        raw = row.get("processed_motion") or row.get("motion") or row.get("file") or ""
        fname = Path(str(raw).replace("\\", "/")).name
        if fname in risk_files:
            removed.append(row)
        else:
            kept.append(row)
    if removed and not dry_run:
        backup_once(path, risk_dir / "metadata_before_filter" / path.name, dry_run=False)
        (risk_dir / "metadata").mkdir(parents=True, exist_ok=True)
        (risk_dir / "metadata" / path.name).write_text(json.dumps({"rows": removed}, indent=2))
        data["rows"] = kept
        if isinstance(data.get("summary"), dict):
            data["summary"]["rows"] = len(kept)
        path.write_text(json.dumps(data, indent=2))
    return {"removed": len(removed), "kept": len(kept)}


def filter_jsonl_by_motion_name(path: Path, risk_files: set[str], risk_dir: Path, dry_run: bool) -> dict:
    if not path.exists():
        return {}
    kept_lines = []
    removed_lines = []
    with path.open() as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                kept_lines.append(line)
                continue
            raw = (
                row.get("processed_motion")
                or row.get("destination")
                or row.get("motion")
                or row.get("file")
                or ""
            )
            fname = Path(str(raw).replace("\\", "/")).name
            if fname in risk_files:
                removed_lines.append(line)
            else:
                kept_lines.append(line)
    if removed_lines and not dry_run:
        backup_once(path, risk_dir / "metadata_before_filter" / path.name, dry_run=False)
        (risk_dir / "metadata").mkdir(parents=True, exist_ok=True)
        (risk_dir / "metadata" / path.name).write_text("".join(removed_lines))
        path.write_text("".join(kept_lines))
    return {"removed": len(removed_lines), "kept": len(kept_lines)}


def filter_csv_by_motion_name(path: Path, risk_files: set[str], risk_dir: Path, dry_run: bool) -> dict:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if not rows:
        return {"removed": 0, "kept": 0}
    kept = []
    removed = []
    for row in rows:
        raw = row.get("processed_motion") or row.get("motion") or row.get("file") or ""
        fname = Path(str(raw).replace("\\", "/")).name
        if fname in risk_files:
            removed.append(row)
        else:
            kept.append(row)
    if removed and not dry_run:
        backup_once(path, risk_dir / "metadata_before_filter" / path.name, dry_run=False)
        (risk_dir / "metadata").mkdir(parents=True, exist_ok=True)
        with (risk_dir / "metadata" / path.name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(removed)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept)
    return {"removed": len(removed), "kept": len(kept)}


def recompute_stats_for_all_objects(root: Path, skip_files: set[str], dry_run: bool) -> dict:
    cond_path = root / "cond.npy"
    cond = np.load(cond_path, allow_pickle=True).item()
    keys = sorted(cond.keys(), key=lambda k: -len(k))
    sums: dict[str, np.ndarray] = {}
    sumsqs: dict[str, np.ndarray] = {}
    counts: Counter[str] = Counter()

    for obj in cond:
        J = int(np.asarray(cond[obj]["parents"]).shape[0])
        sums[obj] = np.zeros((J, 13), dtype=np.float64)
        sumsqs[obj] = np.zeros((J, 13), dtype=np.float64)

    for i, path in enumerate(sorted((root / "motions").glob("*.npy")), start=1):
        if path.name in skip_files:
            continue
        obj = longest_prefix_match(path.name, keys)
        if obj not in sums:
            raise ValueError(f"motion did not match any cond object: {path}")
        arr = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float64)
        if arr.ndim != 3 or arr.shape[1:] != sums[obj].shape:
            raise ValueError(f"shape mismatch for {path}: {arr.shape}, expected [T,{sums[obj].shape[0]},13]")
        if not np.isfinite(arr).all():
            raise ValueError(f"nonfinite values remain in {path}")
        sums[obj] += arr.sum(axis=0)
        sumsqs[obj] += np.square(arr).sum(axis=0)
        counts[obj] += int(arr.shape[0])
        if i % 10000 == 0:
            print(f"  scanned {i} motions for full mean/std in {root.name}", flush=True)

    updated = {}
    for obj in sorted(sums):
        n = counts[obj]
        if n <= 0:
            raise ValueError(f"no remaining frames for object {obj}")
        mean = sums[obj] / float(n)
        var = np.maximum(sumsqs[obj] / float(n) - np.square(mean), 0.0)
        std = np.maximum(np.sqrt(var), 1e-6)
        dtype = np.asarray(cond[obj]["mean"]).dtype
        cond[obj]["mean"] = mean.astype(dtype, copy=False)
        cond[obj]["std"] = std.astype(dtype, copy=False)
        updated[obj] = {
            "frames": int(n),
            "max_abs_mean": float(np.max(np.abs(cond[obj]["mean"]))),
            "max_abs_std": float(np.max(np.abs(cond[obj]["std"]))),
        }
    if not dry_run:
        np.save(cond_path, cond, allow_pickle=True)
    return updated


def update_object_index(root: Path, dry_run: bool) -> dict:
    path = root / "object_index.csv"
    if not path.exists():
        return {}
    cond = np.load(root / "cond.npy", allow_pickle=True).item()
    keys = sorted(cond.keys(), key=lambda k: -len(k))
    motion_counts: Counter[str] = Counter()
    frame_counts: Counter[str] = Counter()
    bvh_counts: Counter[str] = Counter()
    for p in sorted((root / "motions").glob("*.npy")):
        obj = longest_prefix_match(p.name, keys)
        if obj is None:
            continue
        arr = np.load(p, mmap_mode="r")
        motion_counts[obj] += 1
        frame_counts[obj] += int(arr.shape[0])
    for p in sorted((root / "bvhs").glob("*.bvh")):
        obj = longest_prefix_match(p.name, keys)
        if obj is not None:
            bvh_counts[obj] += 1
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    for row in rows:
        obj = row.get("object_name", "")
        if obj in cond:
            row["motions"] = str(int(motion_counts[obj]))
            row["frames"] = str(int(frame_counts[obj]))
            row["joints"] = str(int(np.asarray(cond[obj]["parents"]).shape[0]))
            if "bvhs" in row:
                row["bvhs"] = str(int(bvh_counts[obj]))
    if not dry_run:
        backup_once(path, root / "risk_files" / "metadata_before_filter" / path.name, dry_run=False)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return {
        "objects": len(rows),
        "motions": int(sum(motion_counts.values())),
        "frames": int(sum(frame_counts.values())),
    }


def update_pack_summary(root: Path, index_summary: dict, dry_run: bool) -> None:
    path = root / "pack_summary.json"
    if not path.exists() or not index_summary:
        return
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return
    for key in ("motions", "text_manifest_rows"):
        if key in data:
            data[key] = int(index_summary["motions"])
    for key in ("frames", "total_frames"):
        if key in data:
            data[key] = int(index_summary["frames"])
    if "bvhs" in data:
        data["bvhs"] = len(list((root / "bvhs").glob("*.bvh")))
    if not dry_run:
        backup_once(path, root / "risk_files" / "metadata_before_filter" / path.name, dry_run=False)
        path.write_text(json.dumps(data, indent=2))


def move_stale_cond_cache(root: Path, dry_run: bool) -> list[str]:
    moved = []
    for path in sorted(root.glob("_cond_normalized_*.pkl")):
        dst = root / "risk_files" / "stale_cond_cache" / path.name
        status = move_if_exists(path, dst, dry_run)
        moved.append(f"{path.name}:{status}")
    return moved


def process_dataset(root: Path, risk_rows: list[dict], dry_run: bool) -> dict:
    risk_files = {row["file"] for row in risk_rows}
    risk_dir = root / "risk_files"
    cond = np.load(root / "cond.npy", allow_pickle=True).item()
    keys = sorted(cond.keys(), key=lambda k: -len(k))

    affected_objects = set()
    move_rows = []
    for row in risk_rows:
        fname = row["file"]
        obj = longest_prefix_match(fname, keys)
        if obj is None:
            obj = row.get("object_prefix", "")
        affected_objects.add(obj)
        motion_src = root / "motions" / fname
        motion_dst = risk_dir / "motions" / fname
        bvh_src = root / "bvhs" / f"{Path(fname).stem}.bvh"
        bvh_dst = risk_dir / "bvhs" / f"{Path(fname).stem}.bvh"
        move_rows.append(
            {
                **row,
                "matched_object": obj,
                "motion_move_status": move_if_exists(motion_src, motion_dst, dry_run),
                "bvh_move_status": move_if_exists(bvh_src, bvh_dst, dry_run),
            }
        )

    if not dry_run:
        risk_dir.mkdir(parents=True, exist_ok=True)
        backup_once(root / "cond.npy", risk_dir / "cond_before_velocity_filter.npy", dry_run=False)
    write_csv(risk_dir / "velocity_risk_files.csv", move_rows, dry_run)

    metadata_result = {
        "captions": filter_caption_json(root, risk_files, risk_dir, dry_run),
        "motion_text_manifest_json": filter_motion_manifest_json(root, risk_files, risk_dir, dry_run),
        "motion_text_manifest_jsonl": filter_jsonl_by_motion_name(
            root / "motion_text_manifest.jsonl", risk_files, risk_dir, dry_run
        ),
        "motion_text_manifest_csv": filter_csv_by_motion_name(
            root / "motion_text_manifest.csv", risk_files, risk_dir, dry_run
        ),
        "pack_manifest_jsonl": filter_jsonl_by_motion_name(
            root / "pack_manifest.jsonl", risk_files, risk_dir, dry_run
        ),
    }
    cache_result = move_stale_cond_cache(root, dry_run)
    stats_result = recompute_stats_for_all_objects(root, risk_files, dry_run)
    index_result = update_object_index(root, dry_run)
    update_pack_summary(root, index_result, dry_run)

    summary = {
        "root": str(root),
        "risk_files": len(risk_files),
        "affected_objects": sorted(affected_objects),
        "move_status_counts": dict(Counter(r["motion_move_status"] for r in move_rows)),
        "bvh_status_counts": dict(Counter(r["bvh_move_status"] for r in move_rows)),
        "metadata": metadata_result,
        "stale_cache": cache_result,
        "recomputed_stats": stats_result,
        "index": index_result,
    }
    if not dry_run:
        (risk_dir / "velocity_filter_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    args = parse_args()
    risk_rows = load_risk_rows(Path(args.risk_csv))
    summaries = []
    for dataset in args.datasets:
        root = Path(dataset)
        if not (root / "cond.npy").exists() or not (root / "motions").is_dir():
            raise SystemExit(f"not an AnyTop dataset root: {root}")
        print(f"[process] {root}", flush=True)
        summaries.append(process_dataset(root, risk_rows, dry_run=args.dry_run))
    print(json.dumps(summaries, indent=2)[:12000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
