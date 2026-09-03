#!/usr/bin/env python3
"""Second corpus defect class: JITTER / oscillation (user verdict 2026-08-21 on the alligator tail).

The teleport criterion (scripts/_scan_ktjd17_teleports_local.py) finds ISOLATED discontinuities.
It does not fire on a joint that swings back and forth every frame, because relative to that
clip's own p99 no single step stands out. The user rejected such a clip as well.

Discriminated on known cases (max over joints of peak parent-relative acceleration / bone length):
    REJECTED   alligator tail 1.428    tortoise toe 1.994
    ACCEPTED   cheetah run 0.012       clouded-leopard jump 0.628    human 0.057
Acceleration, not jerk-to-velocity ratio and not reversal rate: both of those put a clouded-leopard
jump (0.813 / 0.083) on top of the rejected alligator (0.841 / 0.050) and cannot separate them.
"""
import json, sys, collections
from pathlib import Path
import numpy as np

D = Path("dataset/ktjd17_pz_human312")
BS = [0.8, 1.0, 1.25, 1.5, 2.0]
OUT = Path("configs/pzh312_jitter_scan.json")

sk_cache = {}
def skel(rig):
    if rig not in sk_cache:
        z = np.load(D / "skeletons" / f"{rig}.npz", allow_pickle=True)
        par = np.asarray(z["parents"]).astype(int)
        Pr = np.asarray(z["P_rest_global"], np.float64)
        bone = np.linalg.norm(Pr - Pr[np.maximum(par, 0)], axis=-1)
        bone[par < 0] = np.linalg.norm(Pr.max(0) - Pr.min(0))
        sk_cache[rig] = (par, np.maximum(bone, 1e-6)[:, None])
    return sk_cache[rig]

rows = [json.loads(l) for l in (D / "manifests/clips.jsonl").open()]
print(f"{len(rows)} clips | thresholds {BS}", flush=True)
n_c = collections.Counter(); rigs = collections.defaultdict(set); hits = collections.defaultdict(dict)
allmax, tops = [], []
for i, r in enumerate(rows):
    rig, cid = r["rig_id"], r["clip_id"]
    z = np.load(D / r["motion_relpath"], allow_pickle=True)
    m = np.asarray(z["motion"] if "motion" in z else z[list(z.keys())[0]], np.float64)
    if m.shape[0] < 3:
        continue
    par, bone = skel(rig)
    P = m[:, :, 0:3].copy(); P[:, :, 0] += m[:, 0:1, 13]; P[:, :, 2] += m[:, 0:1, 14]
    rel = P - P[:, np.maximum(par, 0)]
    rel[:, par < 0] = P[:, par < 0]
    acc = np.linalg.norm(np.diff(rel, n=2, axis=0), axis=-1) / bone.T     # [T-2,J]
    mx = float(acc.max()); allmax.append(mx)
    for B in BS:
        if mx > B:
            n_c[B] += 1; rigs[B].add(rig); hits[B][cid] = mx
    if mx > BS[0]:
        t, j = np.unravel_index(int(acc.argmax()), acc.shape)
        tops.append((mx, cid, rig, int(j), int(t) + 1, int(m.shape[0])))
    if i % 20000 == 0:
        print(f"  ...{i}/{len(rows)}", flush=True)

a = np.array(allmax)
print(f"\nper-clip max parent-relative acceleration (bone-lengths):")
print("   " + "  ".join(f"p{q}={np.percentile(a,q):.3f}" for q in (50, 90, 99, 99.9)) +
      f"  max={a.max():.3f}")
print(f"\n{'B':>6} {'clips':>8} {'clip%':>8} {'rigs':>6}")
for B in BS:
    print(f"{B:6.2f} {n_c[B]:8,} {100*n_c[B]/len(a):7.3f}% {len(rigs[B]):6d}")
tops.sort(reverse=True)
print("\ntop 15 (peak accel, rig, joint, frame/T):")
for mx, cid, rg, j, t, T in tops[:15]:
    print(f"  {mx:8.2f}  {rg:33s} j{j:<3d} f{t:<4d}/{T}")
OUT.write_text(json.dumps({"thresholds": BS, "total_clips": len(a),
    "criterion": "max_{t,j} ||d2/dt2 (P_j - P_parent)|| / bone_length",
    "known": {"alligator_tail_REJECTED": 1.428, "tortoise_toe_REJECTED": 1.994,
              "cheetah_run_OK": 0.012, "clouded_leopard_jump_OK": 0.628, "human_OK": 0.057},
    "percentiles": {f"p{q}": float(np.percentile(a, q)) for q in (50, 90, 99, 99.9)},
    "max": float(a.max()),
    "per_B": {str(B): {"clips": n_c[B], "rigs": sorted(rigs[B])} for B in BS},
    "clip_hits": {str(B): hits[B] for B in BS},
    "top": [{"acc": mx, "clip_id": c, "rig": rg, "j": j, "frame": t, "T": T}
            for mx, c, rg, j, t, T in tops[:400]]}))
print(f"\n[OK] wrote {OUT}")
