#!/usr/bin/env python3
"""Per-(rig, joint, channel) statistics for the merged no-IK corpus, from scratch.

THE CONTRACT (src/data/ktjd17/species_stats.py:220-250) has three partitions, and getting it wrong
silently corrupts the normalisation -- it already cost this project two artifacts and a full gamma
calibration on 2026-08-21:
    ch0:13   every frame, every joint
    ch13:15  every frame, ROOT ROW ONLY   (the smooth-root track)
    ch15:17  ROOT ROW ONLY and only on heading_valid frames (invalid frames store zeros)

Accumulated in float64. Human clips are unchanged from the old corpus, so their recomputed
statistics must match the old artifact -- that comparison is run separately as the no-op check.
"""
import json, os, sys
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "dataset/ktjd17_pzh312_noik_v1")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "data/noik_rig_stats_v1.npz")
WORKERS = int(os.environ.get("WORKERS", "16"))
C = 17

def one(args):
    rig, paths, Jmax = args
    n = np.zeros((Jmax, C), np.float64)
    s = np.zeros((Jmax, C), np.float64)
    q = np.zeros((Jmax, C), np.float64)
    lo = np.full((Jmax, C), np.inf); hi = np.full((Jmax, C), -np.inf)
    nhead = 0
    for p in paths:
        with np.load(p) as z:
            m = np.asarray(z["motion"], dtype=np.float64)
            hv = np.asarray(z["heading_valid"], dtype=bool)
        T, J = m.shape[0], m.shape[1]
        # ch0:13 -- all frames, all joints
        blk = m[:, :, 0:13]
        n[:J, 0:13] += T
        s[:J, 0:13] += blk.sum(0); q[:J, 0:13] += (blk * blk).sum(0)
        np.minimum(lo[:J, 0:13], blk.min(0), out=lo[:J, 0:13])
        np.maximum(hi[:J, 0:13], blk.max(0), out=hi[:J, 0:13])
        # ch13:15 -- all frames, ROOT row only
        r = m[:, 0, 13:15]
        n[0, 13:15] += T
        s[0, 13:15] += r.sum(0); q[0, 13:15] += (r * r).sum(0)
        np.minimum(lo[0, 13:15], r.min(0), out=lo[0, 13:15])
        np.maximum(hi[0, 13:15], r.max(0), out=hi[0, 13:15])
        # ch15:17 -- ROOT row, heading_valid frames only
        if hv.any():
            h = m[hv, 0, 15:17]
            nhead += int(hv.sum())
            n[0, 15:17] += h.shape[0]
            s[0, 15:17] += h.sum(0); q[0, 15:17] += (h * h).sum(0)
            np.minimum(lo[0, 15:17], h.min(0), out=lo[0, 15:17])
            np.maximum(hi[0, 15:17], h.max(0), out=hi[0, 15:17])
    return rig, n, s, q, lo, hi, nhead

def main():
    rows = [json.loads(l) for l in open(ROOT / "manifests" / "clips.jsonl")]
    by_rig = defaultdict(list)
    for r in rows:
        by_rig[r["rig_id"]].append(ROOT / r["motion_relpath"])
    rigs = sorted(by_rig)
    Jn = {}
    for rig in rigs:
        with np.load(ROOT / "skeletons" / f"{rig}.npz", allow_pickle=True) as z:
            Jn[rig] = len(z["joint_names"])
    Jmax = max(Jn.values())
    print(f"[stats] {len(rigs)} rigs, {len(rows)} clips, Jmax={Jmax}", flush=True)

    mean = np.zeros((len(rigs), Jmax, C)); std = np.zeros_like(mean)
    cnt = np.zeros_like(mean); vmask = np.zeros((len(rigs), Jmax, C), bool)
    mn = np.zeros_like(mean); mx = np.zeros_like(mean)
    heads = np.zeros(len(rigs), np.int64)
    done = 0
    with ProcessPoolExecutor(WORKERS) as ex:
        for rig, n, s, q, lo, hi, nh in ex.map(
                one, [(r, by_rig[r], Jmax) for r in rigs], chunksize=1):
            i = rigs.index(rig)
            with np.errstate(invalid="ignore", divide="ignore"):
                mu = np.where(n > 0, s / np.maximum(n, 1), 0.0)
                var = np.where(n > 0, q / np.maximum(n, 1) - mu * mu, 0.0)
            mean[i] = mu; std[i] = np.sqrt(np.maximum(var, 0.0)); cnt[i] = n
            vmask[i] = n > 0
            mn[i] = np.where(np.isfinite(lo), lo, 0.0)
            mx[i] = np.where(np.isfinite(hi), hi, 0.0)
            heads[i] = nh
            done += 1
            if done % 40 == 0:
                print(f"  {done}/{len(rigs)} rigs", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, rig_ids=np.array(rigs), mean=mean, std=std, count=cnt,
             valid_mask=vmask, minimum=mn, maximum=mx,
             heading_valid_frame_count=heads, joint_counts=np.array([Jn[r] for r in rigs]))
    print(f"[stats] wrote {OUT}  valid cells {int(vmask.sum()):,}  "
          f"heading frames {int(heads.sum()):,}", flush=True)

if __name__ == "__main__":
    main()
