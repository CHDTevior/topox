#!/usr/bin/env python3
"""Position discontinuity scan -- the criterion the user's verdicts actually select.

User 2026-08-21, adjudicating rendered GT: an Aldabra tortoise toe and an American alligator tail
are defects ("突变", "瞬移的都不能要"); a cheetah's rear toe at full run and a clouded leopard's
front toe mid-jump are NORMAL ("这俩都是正常的"), and the thing to detect is "瞬间变化不连续的".

Four criteria were tried and rejected against those four verdicts, recorded so they are not retried:
  * displacement / rig scale                 median clip already at 5.7%; a 2% cut takes 90.8%
  * world spike vs clip-wide median          near-still joints give 38,000x on trivial motion
  * peak acceleration / bone length          cheetah toe 3.80 > rejected alligator 1.43
  * peak jerk / bone length                  cheetah toe 6.91 > rejected tortoise 3.58
  * step vs the CLIP's p90 (parent-relative or world)   clouded leopard 3.77 > alligator 2.46
The last two fail because a short bone or a clip that is fast throughout rescales everything. What
the verdicts track is a step that is out of line with ITS OWN IMMEDIATE NEIGHBOURHOOD:
    d[t,j]   = ||P[t+1,j] - P[t,j]|| / rig_scale          (world; that is what the user watched)
    base[t,j]= median of d[.,j] over a 9-frame window centred at t
    ratio    = d / max(base, floor)
Measured: alligator 7.29, tortoise 6.82  vs  clouded leopard 3.10, cheetah 1.35, human 1.39,
meerkat 1.83, elephant 1.03, jaguar 1.02, kangaroo 1.21, komodo 1.34.
"""
import json, sys, collections
from pathlib import Path
import numpy as np
from scipy.ndimage import median_filter

D = Path("dataset/ktjd17_pz_human312")
KS = [4.0, 5.0, 6.0, 8.0]
W, FFRAC = 9, 0.02
OUT = Path("configs/pzh312_discontinuity_scan.json")

scale = {}
for p in sorted((D / "skeletons").glob("*.npz")):
    P = np.asarray(np.load(p, allow_pickle=True)["P_rest_global"], np.float64)
    scale[p.stem] = float(np.linalg.norm(P.max(0) - P.min(0)))

rows = [json.loads(l) for l in (D / "manifests/clips.jsonl").open()]
print(f"{len(rows)} clips | K={KS} window={W}", flush=True)
n_c = collections.Counter(); rigs = collections.defaultdict(set); hits = collections.defaultdict(dict)
allmax, tops = [], []
for i, r in enumerate(rows):
    rig, cid = r["rig_id"], r["clip_id"]
    z = np.load(D / r["motion_relpath"], allow_pickle=True)
    m = np.asarray(z["motion"] if "motion" in z else z[list(z.keys())[0]], np.float64)
    if m.shape[0] < W + 1:
        allmax.append(0.0); continue
    P = m[:, :, 0:3].copy(); P[:, :, 0] += m[:, 0:1, 13]; P[:, :, 2] += m[:, 0:1, 14]
    d = np.linalg.norm(np.diff(P, axis=0), axis=-1) / max(scale[rig], 1e-9)
    loc = median_filter(d, size=(W, 1), mode="nearest")
    fl = FFRAC * float(np.percentile(d, 90))
    ratio = d / np.maximum(loc, max(fl, 1e-9))
    mx = float(ratio.max()); allmax.append(mx)
    for K in KS:
        if mx > K:
            n_c[K] += 1; rigs[K].add(rig); hits[K][cid] = mx
    if mx > KS[0]:
        t, j = np.unravel_index(int(ratio.argmax()), ratio.shape)
        tops.append((mx, cid, rig, int(j), int(t) + 1, int(m.shape[0])))
    if i % 20000 == 0:
        print(f"  ...{i}/{len(rows)}", flush=True)

a = np.array(allmax)
print(f"\nper-clip max discontinuity ratio:")
print("   " + "  ".join(f"p{q}={np.percentile(a,q):.2f}" for q in (50, 90, 99, 99.9)) +
      f"  max={a.max():.1f}")
print(f"\n{'K':>5} {'clips':>8} {'clip%':>8} {'rigs':>6}")
for K in KS:
    print(f"{K:5.1f} {n_c[K]:8,} {100*n_c[K]/len(a):7.3f}% {len(rigs[K]):6d}")
tops.sort(reverse=True)
print("\ntop 12 (ratio, rig, joint, frame/T):")
for mx, cid, rg, j, t, T in tops[:12]:
    print(f"  {mx:8.1f}  {rg:33s} j{j:<3d} f{t:<4d}/{T}")
OUT.write_text(json.dumps({"K": KS, "window": W, "floor_frac": FFRAC, "total_clips": len(a),
    "criterion": "d=||dP||/rig_scale (world); base=median_filter(d,W) per joint; ratio=d/base",
    "verdicts": {"alligator_tail_REJECTED": 7.29, "tortoise_toe_REJECTED": 6.82,
                 "clouded_leopard_toe_OK": 3.10, "cheetah_toe_OK": 1.35},
    "percentiles": {f"p{q}": float(np.percentile(a, q)) for q in (50, 90, 99, 99.9)},
    "max": float(a.max()),
    "per_K": {str(K): {"clips": n_c[K], "rigs": sorted(rigs[K])} for K in KS},
    "clip_hits": {str(K): hits[K] for K in KS},
    "top": [{"ratio": mx, "clip_id": c, "rig": rg, "j": j, "frame": t, "T": T}
            for mx, c, rg, j, t, T in tops[:500]]}))
print(f"\n[OK] wrote {OUT}")
