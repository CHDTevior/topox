#!/usr/bin/env python3
"""Teleport detection by SPIKINESS, not magnitude.

The magnitude-only scan (scripts/_scan_ktjd17_teleports.py) showed why magnitude cannot work: the
median clip already has a frame moving 5.7% of rig scale, and a 2% cut hits 90.8% of clips. Fast
motion IS large displacement. What separates the user-rejected artifact from fast motion is that
the artifact is ISOLATED -- the tortoise pop was 11.63 with 0.03 and 0.56 on either side.

For each (clip, joint): d_t = |v_t| / fps, and
    spike_t = d_t / max(median_t(d), floor)
A frame is a teleport if spike_t > K AND d_t > amin * rig_scale (the second guard stops a joint
that is essentially still all clip from registering any ordinary motion as a huge ratio).
"""
import json, sys, collections
from pathlib import Path
import numpy as np

D = Path("dataset/ktjd17_pz_human312")
KS = [5.0, 10.0, 20.0, 40.0]
AMIN = 0.02
OUT = Path("configs/pzh312_teleport_spike_scan.json")

scale = {}
for p in sorted((D / "skeletons").glob("*.npz")):
    P = np.asarray(np.load(p, allow_pickle=True)["P_rest_global"], np.float64)
    scale[p.stem] = float(np.linalg.norm(P.max(0) - P.min(0)))

rows = [json.loads(l) for l in (D / "manifests/clips.jsonl").open()]
print(f"{len(rows)} clips, {len(scale)} rigs, amin={AMIN}", flush=True)
n_f = collections.Counter(); n_c = collections.Counter()
rigs = collections.defaultdict(set); hits = collections.defaultdict(dict)
top = []
tot_f = 0
for i, r in enumerate(rows):
    rig, cid = r["rig_id"], r["clip_id"]
    z = np.load(D / r["motion_relpath"], allow_pickle=True)
    m = z["motion"] if "motion" in z else z[list(z.keys())[0]]
    fps = float(r.get("fps_target", 30.0))
    s = max(scale[rig], 1e-9)
    d = np.linalg.norm(np.asarray(m[:, :, 9:12], np.float64), axis=-1) / fps / s   # [T,J], rig-rel
    tot_f += d.shape[0]
    med = np.median(d, axis=0, keepdims=True)                       # per joint, whole clip
    spike = d / np.maximum(med, 1e-6)
    big = d > AMIN
    for K in KS:
        hit = big & (spike > K)
        nf = int(hit.any(axis=1).sum())
        if nf:
            n_f[K] += nf; n_c[K] += 1; rigs[K].add(rig); hits[K][cid] = nf
    hh = big & (spike > KS[1])
    if hh.any():
        t, j = np.unravel_index(int(np.where(hh, spike, 0).argmax()), spike.shape)
        top.append((float(spike[t, j]), float(d[t, j]), cid, rig, int(j), int(t), int(d.shape[0])))
    if i % 20000 == 0:
        print(f"  ...{i}/{len(rows)}", flush=True)

print(f"\ntotal frames = {tot_f:,} over {len(rows):,} clips")
print(f"{'K':>6} {'frames':>10} {'frame%':>8} {'clips':>8} {'clip%':>8} {'rigs':>6}")
for K in KS:
    print(f"{K:6.0f} {n_f[K]:10,} {100*n_f[K]/tot_f:7.4f}% {n_c[K]:8,} "
          f"{100*n_c[K]/len(rows):7.3f}% {len(rigs[K]):6d}")
top.sort(reverse=True)
print(f"\ntop 15 spikes (spike_ratio, displacement/scale, rig, joint, frame, clip_len):")
for sp, dd, cid, rg, j, t, T in top[:15]:
    print(f"  {sp:9.1f}  {dd:.4f}  {rg:32s} j{j:<3d} f{t:<4d}/{T}")
OUT.write_text(json.dumps({"K": KS, "amin": AMIN, "total_frames": tot_f,
                           "total_clips": len(rows),
                           "criterion": "d_t=|v|/fps/rig_scale; spike=d/median_clip(d); "
                                        "teleport = d>amin AND spike>K",
                           "per_K": {str(K): {"frames": n_f[K], "clips": n_c[K],
                                              "rigs": sorted(rigs[K])} for K in KS},
                           "clip_hits": {str(K): hits[K] for K in KS},
                           "top": [{"spike": sp, "disp_rel": dd, "clip_id": cid, "rig": rg,
                                    "j": j, "frame": t, "T": T} for sp, dd, cid, rg, j, t, T in top[:200]]}))
print(f"\n[OK] wrote {OUT}")
