"""How much do the PZ clips in the TRAINING dataset actually MOVE?

The 311-species pack verified FK==RIC (self-consistency). A FROZEN clip passes that
trivially. This audits the thing that check never looked at: motion ENERGY.

Two energies, both scale-free (divided by the animal's own body extent):
  root_travel  : how far the root translates over the shot / body extent
  joint_motion : mean per-joint displacement between consecutive frames / body extent

Split by the Planet Zoo naming convention:
  'animationmotionextracted...'     -> root motion IS baked into the animation
  'animationnotmotionextracted...'  -> animation plays IN PLACE (engine moves the actor)
"""
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, "/scratch/ts1v23/workspace/noKslot_clean")
from src.data.anytop_rot6d_fk import recover_from_bvh_rot_np

R = "data/animo4d_L4TB_plus_human_v4b272neutral"
N_SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 400

cond = np.load(os.path.join(R, "cond.npy"), allow_pickle=True).item()
keys = sorted(cond.keys(), key=len, reverse=True)
files = [f for f in sorted(os.listdir(os.path.join(R, "motions"))) if f.startswith("PZ_")]
random.seed(0)
sample = random.sample(files, min(N_SAMPLE, len(files)))

rows = []
for i, fn in enumerate(sample, 1):
    stem = fn[:-4]
    ot = next((k for k in keys if stem.startswith(k)), None)
    if ot is None:
        continue
    c = cond[ot]
    parents = np.asarray(c["parents"], dtype=int)
    offsets = np.asarray(c["offsets"], dtype=np.float64)
    J = len(parents)
    m = np.load(os.path.join(R, "motions", fn))
    if m.shape[0] < 4:
        continue
    w = np.asarray(recover_from_bvh_rot_np(m[:, :J, :].astype(np.float64), parents, offsets),
                   dtype=np.float64)[:, :J, :3]

    body = float(np.linalg.norm(w[0].max(0) - w[0].min(0))) + 1e-8
    root_travel = float(np.linalg.norm(np.diff(w[:, 0], axis=0), axis=-1).sum()) / body
    rel = w - w[:, :1, :]                                   # root-relative -> pure articulation
    joint_motion = float(np.linalg.norm(np.diff(rel, axis=0), axis=-1).mean()) / body * m.shape[0]

    rows.append({
        "file": fn, "object_type": ot, "frames": int(m.shape[0]),
        "motion_extracted": ("notmotionextracted" not in fn.lower()),
        "root_travel_bodylens": root_travel,
        "joint_motion_bodylens": joint_motion,
    })
    if i % 100 == 0:
        print(f"  ...{i}/{len(sample)}", flush=True)

json.dump(rows, open("scratch/_motion_energy_audit.json", "w"), indent=1)

STATIC = 0.02          # < 2% of a body length of articulation over the whole shot
me = [r for r in rows if r["motion_extracted"]]
nm = [r for r in rows if not r["motion_extracted"]]


def stat(g, key):
    v = sorted(r[key] for r in g)
    if not v:
        return "n/a"
    return f"median {v[len(v)//2]:.4f}  p10 {v[len(v)//10]:.4f}  p90 {v[len(v)*9//10]:.4f}"


print(f"\n=== sampled {len(rows)} PZ clips from the TRAINING dataset ===")
for name, g in (("motion-extracted    ", me), ("NOT-motion-extracted", nm)):
    if not g:
        continue
    frozen = sum(1 for r in g if r["joint_motion_bodylens"] < STATIC)
    print(f"\n{name}  n={len(g)}")
    print(f"   root travel  (body lengths): {stat(g, 'root_travel_bodylens')}")
    print(f"   joint motion (body lengths): {stat(g, 'joint_motion_bodylens')}")
    print(f"   NEAR-STATIC (joint motion < {STATIC}): {frozen}/{len(g)} = {100*frozen/len(g):.1f}%")

allf = sum(1 for r in rows if r["joint_motion_bodylens"] < STATIC)
print(f"\n>>> OVERALL near-static: {allf}/{len(rows)} = {100*allf/len(rows):.1f}% of sampled PZ training clips")
worst = sorted(rows, key=lambda r: r["joint_motion_bodylens"])[:5]
print("\nmost frozen clips:")
for r in worst:
    print(f"   {r['joint_motion_bodylens']:.5f}  {r['file'][:88]}")
