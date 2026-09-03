#!/usr/bin/env python3
"""Teleport detection by PARENT-RELATIVE discontinuity, scaled to each clip's own normal level.

Two earlier criteria failed on measured data and are recorded so nobody retries them:
  * absolute displacement / rig scale -- the median clip already has a frame at 5.7%, and a 2% cut
    hits 90.8% of clips. Fast motion IS large displacement.
  * spike ratio on WORLD displacement -- dominated by near-still joints, whose median is ~0, so a
    tiny motion registers as a 38,000x spike. K=40 still hit 35.6% of clips.

What works (validated against the one user-adjudicated teleport and four fast-locomotion controls):
    d_local[t,j] = || (P[t,j]-P[t,pa]) - (P[t-1,j]-P[t-1,pa]) || / bone_length[j]
strips the parent's motion, so an end joint carried fast by its limb reads small while a joint that
jumps relative to its own parent reads large. Then compare each frame to the SAME CLIP's p99:
    ratio = d_local[t,j] / max(p99_clip(d_local), FLOOR)
Measured: tortoise teleport 14.4x, cheetah run 1.05x, clouded-leopard jump 1.04x, jaguar 1.06x.
"""
import json, sys, collections
from pathlib import Path
import numpy as np

D = Path("dataset/ktjd17_pz_human312")
KS = [3.0, 5.0, 8.0, 12.0]
FLOOR = 0.10                      # bone-lengths; stops an almost-still clip inflating every ratio
OUT = Path("configs/pzh312_teleport_local_scan.json")

sk_cache = {}
def skel(rig):
    if rig not in sk_cache:
        z = np.load(D / "skeletons" / f"{rig}.npz", allow_pickle=True)
        par = np.asarray(z["parents"]).astype(int)
        Pr = np.asarray(z["P_rest_global"], np.float64)
        bone = np.linalg.norm(Pr - Pr[np.maximum(par, 0)], axis=-1)
        bone[par < 0] = np.linalg.norm(Pr.max(0) - Pr.min(0))
        sk_cache[rig] = (par, np.maximum(bone, 1e-6))
    return sk_cache[rig]

rows = [json.loads(l) for l in (D / "manifests/clips.jsonl").open()]
print(f"{len(rows)} clips | K={KS} floor={FLOOR}", flush=True)
n_f = collections.Counter(); n_c = collections.Counter()
rigs = collections.defaultdict(set); hits = collections.defaultdict(dict)
tops, tot_f = [], 0
for i, r in enumerate(rows):
    rig, cid = r["rig_id"], r["clip_id"]
    z = np.load(D / r["motion_relpath"], allow_pickle=True)
    m = np.asarray(z["motion"] if "motion" in z else z[list(z.keys())[0]], np.float64)
    par, bone = skel(rig)
    P = m[:, :, 0:3].copy(); P[:, :, 0] += m[:, 0:1, 13]; P[:, :, 2] += m[:, 0:1, 14]
    rel = P - P[:, np.maximum(par, 0)]
    rel[:, par < 0] = P[:, par < 0]
    d = np.linalg.norm(np.diff(rel, axis=0), axis=-1) / bone            # [T-1,J]
    tot_f += d.shape[0]
    base = max(float(np.percentile(d, 99)), FLOOR)
    ratio = d / base
    for K in KS:
        hit = ratio > K
        nf = int(hit.any(axis=1).sum())
        if nf:
            n_f[K] += nf; n_c[K] += 1; rigs[K].add(rig); hits[K][cid] = nf
    mr = float(ratio.max())
    if mr > KS[0]:
        t, j = np.unravel_index(int(ratio.argmax()), ratio.shape)
        tops.append((mr, float(d[t, j]), base, cid, rig, int(j), int(t) + 1, int(d.shape[0]) + 1))
    if i % 20000 == 0:
        print(f"  ...{i}/{len(rows)}", flush=True)

print(f"\ntotal transition-frames = {tot_f:,}")
print(f"{'K':>5} {'frames':>10} {'frame%':>9} {'clips':>8} {'clip%':>8} {'rigs':>6}")
for K in KS:
    print(f"{K:5.0f} {n_f[K]:10,} {100*n_f[K]/tot_f:8.4f}% {n_c[K]:8,} "
          f"{100*n_c[K]/len(rows):7.3f}% {len(rigs[K]):6d}")
tops.sort(reverse=True)
print("\ntop 20 (ratio, d_local bone-lengths, clip p99 base, rig, joint, frame/T):")
for mr, dd, b, cid, rg, j, t, T in tops[:20]:
    print(f"  {mr:8.1f}  {dd:7.3f}  {b:6.3f}  {rg:31s} j{j:<3d} f{t:<4d}/{T}")
OUT.write_text(json.dumps({"K": KS, "floor": FLOOR, "total_frames": tot_f,
    "total_clips": len(rows),
    "criterion": "d_local=||delta(P_j - P_parent)||/bone; ratio=d_local/max(p99_clip,floor)",
    "validated_on": {"tortoise_teleport": 14.4, "cheetah_run": 1.05,
                     "clouded_leopard_jump": 1.04, "jaguar_turn": 1.06},
    "per_K": {str(K): {"frames": n_f[K], "clips": n_c[K], "rigs": sorted(rigs[K])} for K in KS},
    "clip_hits": {str(K): hits[K] for K in KS},
    "top": [{"ratio": mr, "d_local": dd, "base": b, "clip_id": cid, "rig": rg, "j": j,
             "frame": t, "T": T} for mr, dd, b, cid, rg, j, t, T in tops[:400]]}))
print(f"\n[OK] wrote {OUT}")
