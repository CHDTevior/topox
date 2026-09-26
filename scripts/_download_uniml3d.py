#!/usr/bin/env python3
"""Pin/index UniML3D Objaverse exports, then download selected physical-source NPZs."""
import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from huggingface_hub import HfApi, snapshot_download

REPO = "Linzhan/UniML3D"
REVISION = "85be3505794200daf7a60fabf8eaedf39185bb27"


def rate_limit_response(error):
    """The SDK may wrap metadata HTTP errors in LocalEntryNotFoundError."""
    pending=[error];seen=set()
    while pending:
        exc=pending.pop()
        if id(exc) in seen:continue
        seen.add(id(exc))
        response=getattr(exc,"response",None)
        if response is not None and response.status_code==429:return response
        for inner in (getattr(exc,"__cause__",None),getattr(exc,"__context__",None)):
            if inner is not None:pending.append(inner)
    return None


def prepare(root):
    index = root / "selection.json"
    if index.exists():
        return json.loads(index.read_text())
    info = HfApi().dataset_info(REPO, revision=REVISION, files_metadata=True)
    files = {x.rfilename: x.size for x in info.siblings}
    snapshot_download(REPO, repo_type="dataset", revision=REVISION, local_dir=root,
                      allow_patterns=["README.md", "clips.csv", "skeletons.csv",
                                      "export/objaverse/*.json", "export/objaverse/*.txt"],
                      max_workers=6)
    def obj_json(name):
        return json.loads((root / "export/objaverse" / (name + ".json")).read_text())
    def exclusions(name):
        lines = (root / "export/objaverse" / (name + ".txt")).read_text().splitlines()
        return {line.split("#", 1)[0].strip() for line in lines if line.split("#", 1)[0].strip()}
    faces, flags = obj_json("face_joint_names"), obj_json("rig_flags")
    excluded = exclusions("filtered_objects")
    badflags = {"tpose_wrong", "facing_wrong", "object_no_front"}
    for flag in badflags:
        excluded.update(flags.get("categories", {}).get(flag, []))
    for rig, entry in flags.get("rigs", {}).items():
        if entry.get("category") in badflags:
            excluded.add(rig)
    filtered_clips = exclusions("filtered_clips")
    skeletons = [r for r in csv.DictReader((root / "skeletons.csv").open())
                 if r["dataset"] == "objaverse"]
    clips = [r for r in csv.DictReader((root / "clips.csv").open())
             if r["dataset"] == "objaverse"]
    motion_files = {Path(p).stem: p for p in files if p.startswith("export/objaverse/motions/")}
    byrig = defaultdict(list)
    rejected_clips = []
    for row in clips:
        cid = row["clip"]
        if cid in filtered_clips:
            rejected_clips.append({"official_id": cid, "reason": "upstream_filtered_clip"})
            continue
        if cid not in motion_files:
            raise FileNotFoundError(f"No export NPZ for indexed clip {cid}")
        row["source_npz"] = motion_files[cid]
        row["source_bytes"] = files[row["source_npz"]]
        byrig[row["object_type"]].append(row)
    selected, rejected = [], []
    for row in skeletons:
        rig = row["object_type"]
        face = faces.get(rig, {})
        reason = None
        if rig in excluded:
            reason = "upstream_bad_rest_or_facing"
        elif int(row["num_joints"]) > 142:
            reason = "joint_count_over_142"
        elif row["category"] == "articulated_rigid":
            reason = "articulated_rigid"
        elif not face.get("r_hip", {}).get("raw") or not face.get("l_hip", {}).get("raw"):
            reason = "missing_face_pair"
        elif not byrig[rig]:
            reason = "no_unfiltered_clips"
        if reason:
            rejected.append({"object_type": rig, "category": row["category"], "reason": reason})
        else:
            selected.append({**row, "clips": sorted(byrig[rig], key=lambda x: x["clip"]),
                             "license": "not_provided_per_asset_in_UniML3D_export",
                             "asset_url": (f"https://sketchfab.com/3d-models/{rig}"
                                           if len(rig) == 32 else None)})
    # Stratified order makes any prefix useful for the first numerical/visual batch.
    groups = defaultdict(list)
    for row in selected:
        groups[row["category"]].append(row)
    for rows in groups.values():
        rows.sort(key=lambda x: (sum(c["source_bytes"] for c in x["clips"]), x["object_type"]))
    ordered = []
    while any(groups.values()):
        for category in sorted(groups):
            if groups[category]:
                ordered.append(groups[category].pop(0))
    result = {"repo_id": REPO, "revision": REVISION,
              "source_rigs": len(skeletons), "source_clips": len(clips),
              "source_categories": dict(Counter(x["category"] for x in skeletons)),
              "selected": ordered, "rejected_rigs": rejected,
              "rejected_clips": rejected_clips,
              "note": "Single-tree check requires NPZ parents; CSV has no tree-count field. "
                      "Tpose reference files are PNG only; rest is embedded in each motion NPZ."}
    index.write_text(json.dumps(result, indent=2) + "\n")
    (root / "remote_files.json").write_text(json.dumps(files, indent=2) + "\n")
    print(json.dumps({"candidate_rigs": len(ordered),
                      "candidate_clips": sum(len(r["clips"]) for r in ordered),
                      "candidate_GiB": sum(c["source_bytes"] for r in ordered for c in r["clips"])/2**30,
                      "rejected": dict(Counter(x["reason"] for x in rejected))}), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("dataset/uniml3d"))
    p.add_argument("--limit-rigs", type=int, default=0, help="0: every eligible candidate")
    p.add_argument("--metadata-only", action="store_true")
    p.add_argument("--references", action="store_true", help="Also cache upstream PNG/MP4 previews")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    data = prepare(a.root)
    if a.metadata_only:
        return
    rows = data["selected"][:a.limit_rigs or None]
    paths = [c["source_npz"] for r in rows for c in r["clips"]]
    if a.references:
        paths += [r["tpose"] for r in rows]
        paths += [c["video"] for r in rows for c in r["clips"]]
    print(f"Downloading {len(paths)} files for {len(rows)} rigs, revision {data['revision']}", flush=True)
    sizes=json.loads((a.root/"remote_files.json").read_text())
    for attempt in range(5):
        remaining=[p for p in paths if not (a.root/p).is_file() or (a.root/p).stat().st_size != sizes[p]]
        if not remaining:
            break
        print(f"Verified by pinned remote size: {len(paths)-len(remaining)} cached; {len(remaining)} remain",flush=True)
        try:
            mismatched=[p for p in remaining if (a.root/p).is_file()]
            if mismatched:
                snapshot_download(data["repo_id"],repo_type="dataset",revision=data["revision"],
                                  local_dir=a.root,allow_patterns=mismatched,
                                  force_download=True,max_workers=a.workers)
            snapshot_download(data["repo_id"], repo_type="dataset", revision=data["revision"],
                              local_dir=a.root, allow_patterns=remaining, max_workers=a.workers)
            break
        except Exception as exc:
            response=rate_limit_response(exc)
            if response is None or attempt == 4:
                raise
            retry = response.headers.get("Retry-After", "300")
            try:
                delay=max(0.,float(retry))
            except ValueError:
                delay=max(0.,(parsedate_to_datetime(retry)-datetime.now(timezone.utc)).total_seconds())
            print(f"HF rate limit: retaining cached files, retrying in {delay}s", flush=True)
            time.sleep(delay)
    missing = [x for x in paths if not (a.root / x).is_file() or (a.root/x).stat().st_size != sizes[x]]
    if missing:
        raise FileNotFoundError(missing)
    print(f"Complete: {len(paths)} files", flush=True)


if __name__ == "__main__":
    main()
