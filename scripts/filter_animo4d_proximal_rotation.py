#!/usr/bin/env python3
"""Filter AniMo4D AnyTop clips flagged by proximal-rotation QC.

The dataset root names stay unchanged. Flagged clips are moved aside under
``proximal_rotation_removed_20260608/`` so training code can continue to point
at the same dataset roots while excluded samples remain recoverable.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_REMOVE_DIR = (
    "data/animo4d_anytop/proximal_rotation_remove_list_20260608"
)
DEFAULT_DATASETS = [
    "data/animo4d_anytop",
    "data/animo4d_anytop_clean_L1",
    "data/animo4d_anytop_clean_L2",
]
REMOVED_DIR_NAME = "proximal_rotation_removed_20260608"
STATS_FLOOR = 1e-6


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remove-dir", default=DEFAULT_REMOVE_DIR)
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"empty jsonl: {path}")
    return rows


def load_remove_rows(remove_dir: Path) -> list[dict]:
    rows = read_jsonl(remove_dir / "remove_motion_rows.jsonl")
    seen = set()
    out = []
    for row in rows:
        fname = row.get("motion_file") or Path(row.get("motion_relpath", "")).name
        if not fname:
            raise ValueError(f"remove row has no motion filename: {row}")
        if fname in seen:
            continue
        seen.add(fname)
        row = dict(row)
        row["motion_file"] = fname
        row["motion_relpath"] = row.get("motion_relpath") or f"motions/{fname}"
        row["bvh_relpath"] = row.get("bvh_relpath") or f"bvhs/{Path(fname).stem}.bvh"
        out.append(row)
    return out


def longest_prefix_match(fname: str, keys: list[str]) -> str | None:
    for key in keys:
        if fname.startswith(f"{key}_"):
            return key
    return None


def backup_once(src: Path, dst: Path, dry_run: bool) -> bool:
    if not src.exists() or dst.exists():
        return False
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return True


def copy_if_exists(src: Path, dst: Path, dry_run: bool) -> bool:
    if not src.exists():
        return False
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return True


def move_if_exists(src: Path, dst: Path, dry_run: bool) -> str:
    if src.exists():
        if dst.exists():
            return "source_and_removed_exist"
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        return "moved"
    if dst.exists():
        return "already_removed"
    return "missing"


def motion_basename_from_row(row: dict) -> str:
    raw = (
        row.get("processed_motion")
        or row.get("destination")
        or row.get("motion")
        or row.get("file")
        or row.get("motion_file")
        or ""
    )
    return Path(str(raw).replace("\\", "/")).name


def write_json(path: Path, data: object, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def write_jsonl(path: Path, rows: Iterable[dict], dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_csv(path: Path, rows: list[dict], dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def filter_caption_dict_json(root: Path, remove_files: set[str], removed_dir: Path,
                             dry_run: bool) -> dict:
    result = {}
    for path in sorted(root.glob("motion_texts_by_file*.json")):
        if path.name.endswith("_summary.json") or path.name.endswith("_audit.json"):
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        removed = {k: data[k] for k in remove_files if k in data}
        kept = {k: v for k, v in data.items() if k not in remove_files}
        if removed and not dry_run:
            backup_once(path, removed_dir / "metadata_before_filter" / path.name, False)
            write_json(removed_dir / "metadata_removed" / path.name, removed, False)
            path.write_text(json.dumps(kept, indent=2))
        result[path.name] = {"removed": len(removed), "kept": len(kept)}
    return result


def filter_motion_text_manifest_json(root: Path, remove_files: set[str],
                                     removed_dir: Path, dry_run: bool) -> dict:
    path = root / "motion_text_manifest.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        return {}
    items = data.get("items")
    if not isinstance(items, list):
        return {}
    kept = []
    removed = []
    for row in items:
        if motion_basename_from_row(row) in remove_files:
            removed.append(row)
        else:
            kept.append(row)
    if removed and not dry_run:
        backup_once(path, removed_dir / "metadata_before_filter" / path.name, False)
        write_json(removed_dir / "metadata_removed" / path.name, {"items": removed}, False)
        data["items"] = kept
        data["rows"] = len(kept)
        if isinstance(data.get("summary"), dict):
            data["summary"]["rows"] = len(kept)
            if "status_counts" in data["summary"]:
                data["summary"]["status_counts"] = {"matched": len(kept)}
        if isinstance(data.get("status_counts"), dict):
            data["status_counts"] = {"matched": len(kept)}
        path.write_text(json.dumps(data, indent=2))
    return {"removed": len(removed), "kept": len(kept)}


def filter_jsonl_by_motion_name(path: Path, remove_files: set[str],
                                removed_dir: Path, dry_run: bool) -> dict:
    if not path.exists():
        return {}
    kept_lines = []
    removed_lines = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                kept_lines.append(line)
                continue
            if motion_basename_from_row(row) in remove_files:
                removed_lines.append(line)
            else:
                kept_lines.append(line)
    if removed_lines and not dry_run:
        backup_once(path, removed_dir / "metadata_before_filter" / path.name, False)
        (removed_dir / "metadata_removed").mkdir(parents=True, exist_ok=True)
        (removed_dir / "metadata_removed" / path.name).write_text("".join(removed_lines))
        path.write_text("".join(kept_lines))
    return {"removed": len(removed_lines), "kept": len(kept_lines)}


def filter_csv_by_motion_name(path: Path, remove_files: set[str],
                              removed_dir: Path, dry_run: bool) -> dict:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    kept = []
    removed = []
    for row in rows:
        if motion_basename_from_row(row) in remove_files:
            removed.append(row)
        else:
            kept.append(row)
    if removed and not dry_run:
        backup_once(path, removed_dir / "metadata_before_filter" / path.name, False)
        (removed_dir / "metadata_removed").mkdir(parents=True, exist_ok=True)
        with (removed_dir / "metadata_removed" / path.name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(removed)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept)
    return {"removed": len(removed), "kept": len(kept)}


def move_stale_cond_cache(root: Path, removed_dir: Path, dry_run: bool) -> list[dict]:
    out = []
    for path in sorted(root.glob("_cond_normalized_*.pkl")):
        dst = removed_dir / "stale_cond_cache" / path.name
        out.append({"file": path.name, "status": move_if_exists(path, dst, dry_run)})
    return out


def recompute_stats(root: Path, dry_run: bool) -> dict:
    cond_path = root / "cond.npy"
    cond = np.load(cond_path, allow_pickle=True).item()
    keys = sorted(cond.keys(), key=lambda k: -len(k))
    sums: dict[str, np.ndarray] = {}
    sumsqs: dict[str, np.ndarray] = {}
    counts: Counter[str] = Counter()

    for obj, sk in cond.items():
        j = int(np.asarray(sk["parents"]).shape[0])
        sums[obj] = np.zeros((j, 13), dtype=np.float64)
        sumsqs[obj] = np.zeros((j, 13), dtype=np.float64)

    motion_paths = sorted((root / "motions").glob("*.npy"))
    for i, path in enumerate(motion_paths, start=1):
        obj = longest_prefix_match(path.name, keys)
        if obj is None:
            raise ValueError(f"motion did not match cond object: {path}")
        arr = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float64)
        expected = sums[obj].shape
        if arr.ndim != 3 or arr.shape[1:] != expected:
            raise ValueError(
                f"shape mismatch for {path}: {arr.shape}, expected [T,{expected[0]},13]"
            )
        if not np.isfinite(arr).all():
            raise ValueError(f"nonfinite values remain in {path}")
        sums[obj] += arr.sum(axis=0)
        sumsqs[obj] += np.square(arr).sum(axis=0)
        counts[obj] += int(arr.shape[0])
        if i % 10000 == 0:
            print(f"  [{root.name}] scanned {i}/{len(motion_paths)} motions", flush=True)

    stats = {}
    for obj in sorted(cond):
        n = counts[obj]
        if n <= 0:
            raise ValueError(f"object has no remaining frames after filter: {obj}")
        mean = sums[obj] / float(n)
        var = np.maximum(sumsqs[obj] / float(n) - np.square(mean), 0.0)
        std = np.maximum(np.sqrt(var), STATS_FLOOR)
        cond[obj]["mean"] = mean.astype(np.float32)
        cond[obj]["std"] = std.astype(np.float32)
        stats[obj] = {
            "frames": int(n),
            "max_abs_mean": float(np.max(np.abs(cond[obj]["mean"]))),
            "max_abs_std": float(np.max(np.abs(cond[obj]["std"]))),
            "min_std": float(np.min(cond[obj]["std"])),
        }
    if not dry_run:
        np.save(cond_path, cond, allow_pickle=True)
    return {
        "objects": len(cond),
        "motions_scanned": len(motion_paths),
        "frames": int(sum(counts.values())),
        "std_floor": STATS_FLOOR,
        "per_object": stats,
    }


def collect_counts(root: Path) -> dict:
    cond = np.load(root / "cond.npy", allow_pickle=True).item()
    keys = sorted(cond.keys(), key=lambda k: -len(k))
    motion_counts: Counter[str] = Counter()
    frame_counts: Counter[str] = Counter()
    bvh_counts: Counter[str] = Counter()
    joint_counts = {obj: int(np.asarray(sk["parents"]).shape[0]) for obj, sk in cond.items()}
    max_abs = 0.0
    nonfinite = 0
    for path in sorted((root / "motions").glob("*.npy")):
        obj = longest_prefix_match(path.name, keys)
        if obj is None:
            continue
        arr = np.load(path, mmap_mode="r")
        motion_counts[obj] += 1
        frame_counts[obj] += int(arr.shape[0])
        finite = bool(np.isfinite(arr).all())
        if not finite:
            nonfinite += 1
        max_abs = max(max_abs, float(np.nanmax(np.abs(arr))))
    for path in sorted((root / "bvhs").glob("*.bvh")):
        obj = longest_prefix_match(path.name, keys)
        if obj is not None:
            bvh_counts[obj] += 1
    return {
        "objects": len(cond),
        "motions": int(sum(motion_counts.values())),
        "frames": int(sum(frame_counts.values())),
        "bvhs": int(sum(bvh_counts.values())),
        "motion_counts": motion_counts,
        "frame_counts": frame_counts,
        "bvh_counts": bvh_counts,
        "joint_counts": joint_counts,
        "clean_min_joints": int(min(joint_counts.values())),
        "clean_max_joints": int(max(joint_counts.values())),
        "clean_mean_joints": float(np.mean(list(joint_counts.values()))),
        "clean_median_joints": float(np.median(list(joint_counts.values()))),
        "nonfinite_motion_files": int(nonfinite),
        "motion_absmax_max": max_abs,
    }


def update_object_index(root: Path, counts: dict, removed_dir: Path, dry_run: bool) -> dict:
    path = root / "object_index.csv"
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    for row in rows:
        obj = row.get("object_name", "")
        if obj in counts["joint_counts"]:
            row["motions"] = str(int(counts["motion_counts"][obj]))
            row["frames"] = str(int(counts["frame_counts"][obj]))
            row["joints"] = str(int(counts["joint_counts"][obj]))
            if "bvhs" in row:
                row["bvhs"] = str(int(counts["bvh_counts"][obj]))
    if not dry_run:
        backup_once(path, removed_dir / "metadata_before_filter" / path.name, False)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return {"objects": len(rows)}


def update_pack_summary(root: Path, counts: dict, removed_dir: Path, dry_run: bool) -> dict:
    path = root / "pack_summary.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    updates = {
        "motions": counts["motions"],
        "text_manifest_rows": counts["motions"],
        "total_frames": counts["frames"],
        "frames": counts["frames"],
        "bvhs": counts["bvhs"],
    }
    for key, value in updates.items():
        if key in data:
            data[key] = int(value)
    data["proximal_rotation_filter"] = {
        "date": "20260608",
        "removed_motions": 3372,
        "rule": "severe + candidate + borderline proximal-limb rotation QC",
    }
    if not dry_run:
        backup_once(path, removed_dir / "metadata_before_filter" / path.name, False)
        path.write_text(json.dumps(data, indent=2))
    return {k: data[k] for k in ("motions", "bvhs", "total_frames", "text_manifest_rows") if k in data}


def update_dataset_summary(root: Path, remove_files: set[str], counts: dict,
                           removed_dir: Path, dry_run: bool) -> dict:
    path = root / "dataset_summary.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    if isinstance(data.get("clips"), list):
        kept = []
        removed = []
        for row in data["clips"]:
            if motion_basename_from_row(row) in remove_files:
                removed.append(row)
            else:
                kept.append(row)
        data["clips"] = kept
        data["total_clips"] = len(kept)
        lengths = [int(row.get("length", 0)) for row in kept if int(row.get("length", 0)) > 0]
        nodes = [int(row.get("nodes", 0)) for row in kept if int(row.get("nodes", 0)) > 0]
        if lengths:
            data["clip_length_range"] = [min(lengths), max(lengths)]
            data["clip_length_distribution"] = sorted(Counter(lengths).items())
        if nodes:
            data["node_count_range"] = [min(nodes), max(nodes)]
            data["node_count_distribution"] = sorted(Counter(nodes).items())
    else:
        removed = []
        data["total_clips"] = counts["motions"]
    if isinstance(data.get("objects"), list):
        # The source summary has a single coarse object row in this AniMo4D dump.
        # Keep its identity but refresh clip/frame-level counts if present.
        for row in data["objects"]:
            if "clips" in row:
                row["clips"] = counts["motions"]
            if "length_min" in row and data.get("clip_length_range"):
                row["length_min"] = data["clip_length_range"][0]
            if "length_max" in row and data.get("clip_length_range"):
                row["length_max"] = data["clip_length_range"][1]
    if not dry_run and removed:
        backup_once(path, removed_dir / "metadata_before_filter" / path.name, False)
        write_json(removed_dir / "metadata_removed" / path.name, {"clips": removed}, False)
        path.write_text(json.dumps(data, indent=2))
    return {"removed": len(removed), "kept": data.get("total_clips")}


def update_dataset_summary_objects(root: Path, counts: dict, removed_dir: Path,
                                   dry_run: bool) -> dict:
    path = root / "dataset_summary_objects.csv"
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    for row in rows:
        if "clips" in row:
            row["clips"] = str(counts["motions"])
        if "length_min" in row:
            row["length_min"] = ""
        if "length_max" in row:
            row["length_max"] = ""
    if not dry_run:
        backup_once(path, removed_dir / "metadata_before_filter" / path.name, False)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return {"rows": len(rows)}


def update_dataset_info(root: Path, counts: dict, removed_dir: Path, dry_run: bool) -> dict:
    path = root / "DATASET_INFO.md"
    if not path.exists():
        return {}
    text = path.read_text()
    replacements = {
        "| Motion `.npy` files |": counts["motions"],
        "| Total frames |": counts["frames"],
        "| Clean max joints |": counts["clean_max_joints"],
        "| Clean median joints |": int(counts["clean_median_joints"]),
        "| Nonfinite motion files while writing |": counts["nonfinite_motion_files"],
        "| Max abs in cleaned motions |": f"{counts['motion_absmax_max']:.6f}",
    }
    lines = []
    changed = False
    for line in text.splitlines():
        new_line = line
        for prefix, value in replacements.items():
            if line.startswith(prefix):
                new_line = f"{prefix} {value} |"
                changed = True
                break
        lines.append(new_line)
    if "Proximal Rotation Filter" not in text:
        lines.extend([
            "",
            "## Proximal Rotation Filter",
            "",
            "On 2026-06-08, clips listed in `source_metadata/proximal_rotation_remove_list_20260608` were excluded from `motions/` and metadata. Removed files are retained under `proximal_rotation_removed_20260608/`.",
            "",
            f"- Removed motions: 3372",
            f"- Remaining motions: {counts['motions']}",
            f"- Remaining frames: {counts['frames']}",
        ])
        changed = True
    if changed and not dry_run:
        backup_once(path, removed_dir / "metadata_before_filter" / path.name, False)
        path.write_text("\n".join(lines) + "\n")
    return {"changed": changed}


def copy_remove_manifest(remove_dir: Path, removed_dir: Path, dry_run: bool) -> dict:
    copied = {}
    for path in sorted(remove_dir.iterdir()):
        if path.is_file():
            copied[path.name] = copy_if_exists(path, removed_dir / "qc_manifest" / path.name, dry_run)
    return copied


def process_dataset(root: Path, remove_dir: Path, rows: list[dict],
                    dry_run: bool) -> dict:
    if not (root / "cond.npy").exists() or not (root / "motions").is_dir():
        raise ValueError(f"not an AnyTop dataset root: {root}")
    removed_dir = root / REMOVED_DIR_NAME
    remove_files = {row["motion_file"] for row in rows}
    remove_motion_or_bvh_files = remove_files | {
        f"{Path(name).stem}.bvh" for name in remove_files
    }
    cond = np.load(root / "cond.npy", allow_pickle=True).item()
    keys = sorted(cond.keys(), key=lambda k: -len(k))

    move_rows = []
    severity_counts = Counter()
    affected_objects = Counter()
    for row in rows:
        fname = row["motion_file"]
        obj = longest_prefix_match(fname, keys) or row.get("object_name", "")
        severity = row.get("severity", "")
        severity_counts[severity] += 1
        affected_objects[obj] += 1
        motion_status = move_if_exists(
            root / "motions" / fname,
            removed_dir / "motions" / fname,
            dry_run,
        )
        bvh_name = Path(row["bvh_relpath"]).name
        bvh_status = move_if_exists(
            root / "bvhs" / bvh_name,
            removed_dir / "bvhs" / bvh_name,
            dry_run,
        )
        move_rows.append({
            "motion_file": fname,
            "object_name": obj,
            "severity": severity,
            "flag": row.get("flag", ""),
            "motion_status": motion_status,
            "bvh_status": bvh_status,
            "trigger_joint_name": row.get("trigger_joint_name", ""),
            "trigger_frame_raw": row.get("trigger_frame_raw", ""),
            "score": row.get("score", ""),
            "text": row.get("text", ""),
        })

    if not dry_run:
        removed_dir.mkdir(parents=True, exist_ok=True)
        backup_once(root / "cond.npy", removed_dir / "metadata_before_filter" / "cond.npy", False)
        copy_remove_manifest(remove_dir, removed_dir, False)
        write_jsonl(removed_dir / "removed_motion_rows.jsonl", rows, False)
        write_csv(removed_dir / "removed_motion_rows.csv", move_rows, False)

    metadata = {
        "caption_dicts": filter_caption_dict_json(root, remove_files, removed_dir, dry_run),
        "motion_text_manifest_json": filter_motion_text_manifest_json(root, remove_files, removed_dir, dry_run),
        "motion_text_manifest_jsonl": filter_jsonl_by_motion_name(
            root / "motion_text_manifest.jsonl", remove_files, removed_dir, dry_run),
        "motion_text_manifest_csv": filter_csv_by_motion_name(
            root / "motion_text_manifest.csv", remove_files, removed_dir, dry_run),
        "pack_manifest_jsonl": filter_jsonl_by_motion_name(
            root / "pack_manifest.jsonl", remove_motion_or_bvh_files, removed_dir, dry_run),
    }
    stale_cache = move_stale_cond_cache(root, removed_dir, dry_run)
    if dry_run:
        summary = {
            "root": str(root),
            "dry_run": True,
            "remove_files_requested": len(remove_files),
            "severity_counts": dict(severity_counts),
            "affected_objects": len(affected_objects),
            "top_affected_objects": affected_objects.most_common(30),
            "motion_status_counts": dict(Counter(r["motion_status"] for r in move_rows)),
            "bvh_status_counts": dict(Counter(r["bvh_status"] for r in move_rows)),
            "metadata": metadata,
            "stale_cache": stale_cache,
            "note": "dry-run skips full mean/std recompute and full motion scan",
        }
        return summary
    stats = recompute_stats(root, dry_run)
    counts = collect_counts(root)
    index = update_object_index(root, counts, removed_dir, dry_run)
    pack_summary = update_pack_summary(root, counts, removed_dir, dry_run)
    dataset_summary = update_dataset_summary(root, remove_files, counts, removed_dir, dry_run)
    dataset_summary_objects = update_dataset_summary_objects(root, counts, removed_dir, dry_run)
    dataset_info = update_dataset_info(root, counts, removed_dir, dry_run)

    summary = {
        "root": str(root),
        "dry_run": dry_run,
        "remove_files_requested": len(remove_files),
        "severity_counts": dict(severity_counts),
        "affected_objects": len(affected_objects),
        "top_affected_objects": affected_objects.most_common(30),
        "motion_status_counts": dict(Counter(r["motion_status"] for r in move_rows)),
        "bvh_status_counts": dict(Counter(r["bvh_status"] for r in move_rows)),
        "remaining": {
            "objects": counts["objects"],
            "motions": counts["motions"],
            "bvhs": counts["bvhs"],
            "frames": counts["frames"],
            "clean_min_joints": counts["clean_min_joints"],
            "clean_max_joints": counts["clean_max_joints"],
            "clean_mean_joints": counts["clean_mean_joints"],
            "clean_median_joints": counts["clean_median_joints"],
            "nonfinite_motion_files": counts["nonfinite_motion_files"],
            "motion_absmax_max": counts["motion_absmax_max"],
        },
        "metadata": metadata,
        "stale_cache": stale_cache,
        "stats": {
            "objects": stats["objects"],
            "motions_scanned": stats["motions_scanned"],
            "frames": stats["frames"],
            "std_floor": stats["std_floor"],
        },
        "index": index,
        "pack_summary": pack_summary,
        "dataset_summary": dataset_summary,
        "dataset_summary_objects": dataset_summary_objects,
        "dataset_info": dataset_info,
    }
    if not dry_run:
        write_json(removed_dir / "proximal_rotation_filter_summary.json", summary, False)
    return summary


def main() -> int:
    args = parse_args()
    remove_dir = Path(args.remove_dir)
    rows = load_remove_rows(remove_dir)
    print(f"[remove-list] {remove_dir} rows={len(rows)}", flush=True)
    summaries = []
    for dataset in args.datasets:
        root = Path(dataset)
        print(f"[process] {root}", flush=True)
        summaries.append(process_dataset(root, remove_dir, rows, args.dry_run))
    print(json.dumps(summaries, indent=2)[:30000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
