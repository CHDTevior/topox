#!/usr/bin/env python3
"""Build ktjd17_pzh312_noik_v1 = new no-IK PZ (311 rigs) + old HML3D_Human (1 rig).

Two things make this more than a file copy:

1. The new corpus stores the root's smooth-root track (ch13:15) in ABSOLUTE world XZ, while this
   project's codec stores it RELATIVE to the clip's first frame and keeps that origin in
   `origin_xz` (codec.py:530). Measured: 199/200 new clips have a nonzero first frame, every old
   clip has exactly [0,0]. Mixing both conventions would put two different meanings in the same
   channel. So the root track is re-based here and the real origin recorded, which also makes the
   new clips decodable back to world coordinates.
   ch0:3 (q = position - smooth_root) is UNAFFECTED: the origin cancels in the difference.

2. The loader requires clip_id / rig_id / fps_target / origin_xz in every payload
   (ktjd17/loader.py:271) and validates them against the manifest row. The new npz files carry
   only motion + heading_valid.

Human clips are already in the target convention and are hard-linked, not rewritten.
"""
import json, os, sys, hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

NEW = Path(os.environ.get("NEW_ROOT",
    "data/animo4d_anytop_noik_canonical/data"))
OLD = Path("dataset/ktjd17_pz_human312")
OUT = Path(os.environ.get("OUT_ROOT", "dataset/ktjd17_pzh312_noik_v2"))
WORKERS = int(os.environ.get("WORKERS", "16"))

def convert(row):
    """Rewrite one new-corpus clip into the project's payload contract."""
    src = NEW / row["motion_file"]
    dst = OUT / "motions" / f"{row['clip_id']}.npz"
    if dst.exists():
        # An interrupted np.savez leaves a truncated file that a later run would silently keep.
        # Trust an existing output only if it actually opens and carries the expected keys.
        try:
            with np.load(dst) as _z:
                if {"motion", "heading_valid", "clip_id", "rig_id", "fps_target",
                    "origin_xz"} <= set(_z.files):
                    return ("skip", row["clip_id"], 0.0)
        except Exception:
            pass
        dst.unlink(missing_ok=True)
    with np.load(src) as z:
        motion = np.asarray(z["motion"], dtype=np.float32).copy()
        hv = np.asarray(z["heading_valid"], dtype=bool)
    # re-base the root smooth track onto its own first frame, recording the origin
    origin = motion[0, 0, 13:15].astype(np.float64).copy()
    motion[:, 0, 13:15] -= origin.astype(np.float32)
    # atomic: a crash mid-write leaves the .tmp behind, never a half-written final file
    # np.savez APPENDS .npz to a path that lacks it, so the temp name must already end in .npz
    # or os.replace would chase a file that was never created (codex 2026-08-24).
    tmp = dst.with_name(dst.name + ".tmp.npz")
    np.savez(tmp, motion=motion, heading_valid=hv,
             clip_id=np.array(row["clip_id"], dtype="<U32"),
             rig_id=np.array(row["rig_id"], dtype="<U40"),
             fps_target=np.float64(30.0),
             origin_xz=origin)
    os.replace(tmp, dst)
    return ("ok", row["clip_id"], float(np.abs(motion[0, 0, 13:15]).max()))

def main():
    (OUT / "motions").mkdir(parents=True, exist_ok=True)
    (OUT / "skeletons").mkdir(exist_ok=True)
    (OUT / "manifests").mkdir(exist_ok=True)

    rows = [json.loads(l) for l in open(NEW / "manifests" / "clips.jsonl")]
    print(f"[new] {len(rows)} clips, {len({r['rig_id'] for r in rows})} rigs", flush=True)

    done = 0
    with ProcessPoolExecutor(WORKERS) as ex:
        for st, cid, resid in ex.map(convert, rows, chunksize=64):
            done += 1
            if resid > 1e-6:
                print(f"[WARN] {cid}: first frame not zeroed ({resid:.2e})", flush=True)
            if done % 10000 == 0:
                print(f"  converted {done}/{len(rows)}", flush=True)
    print(f"[new] converted {done}", flush=True)

    # skeletons: 311 new + 1 human
    import shutil
    n_sk = 0
    for s in sorted((NEW / "skeletons").glob("*.npz")):
        shutil.copy2(s, OUT / "skeletons" / s.name); n_sk += 1
    shutil.copy2(OLD / "skeletons" / "HML3D_Human.npz", OUT / "skeletons" / "HML3D_Human.npz")
    n_sk += 1
    print(f"[skel] {n_sk} skeletons", flush=True)

    # human motions: already in the right convention -- hard-link to avoid 5 GB of duplication
    hu = sorted((OLD / "motions").glob("HML3D_Human_*.npz"))
    linked = 0
    for h in hu:
        d = OUT / "motions" / h.name
        if not d.exists():
            try:
                os.link(h, d)
            except OSError:
                shutil.copy2(h, d)
            linked += 1
    print(f"[human] linked {linked} of {len(hu)} clips", flush=True)
    print(f"[total] motions: {len(list((OUT/'motions').glob('*.npz')))}", flush=True)

    # The manifest is part of the corpus, not a separate step: the statistics job and the loader
    # both require OUT/manifests/clips.jsonl, so a build that omits it is not reproducible
    # (codex 2026-08-24). Delegate to the dedicated builder rather than duplicating the logic.
    import subprocess, sys as _sys
    print("[manifest] building clips.jsonl", flush=True)
    subprocess.run([_sys.executable, "scripts/_build_merged_manifest_stats.py"], check=True)

if __name__ == "__main__":
    main()
