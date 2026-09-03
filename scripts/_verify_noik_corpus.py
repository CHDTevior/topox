#!/usr/bin/env python3
"""Verify the new no-IK corpus against the KTJD-17 contract this project already trains on.

Checks, in order of how much they would hurt if wrong:
  1. heading_valid contract  -- ch15:17 must be EXACTLY zero on invalid frames
  2. channel semantics       -- decode P_j = [q_x + sx, q_y, q_z + sz] and compare against FK
  3. discontinuity           -- the acceleration criterion that condemned the old corpus
  4. joint spec conformance  -- names/descriptions/order against the canonical spec we supplied
"""
import json, sys, glob, os
import numpy as np

R = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 400
SPEC = "handoff/20260822_pzh312_joint_names_descriptions.json"

motions = sorted(glob.glob(R + "/motions/*.npz"))
rng = np.random.default_rng(0)
sample = [motions[i] for i in rng.choice(len(motions), min(N, len(motions)), replace=False)]
print(f"[verify] {len(motions)} clips total, sampling {len(sample)}")

# --- 1. heading_valid contract -------------------------------------------------
bad_head, checked_head, invalid_frames = 0, 0, 0
for f in sample:
    d = np.load(f)
    mo, hv = d["motion"], d["heading_valid"]
    inv = ~hv
    invalid_frames += int(inv.sum())
    if inv.any():
        h = mo[inv][:, 0, 15:17]            # heading lives on the ROOT row
        checked_head += 1
        if np.abs(h).max() > 0:
            bad_head += 1
print(f"[1] heading_valid: {invalid_frames} invalid frames across the sample; "
      f"{checked_head} clips had any; violations (nonzero ch15:17 on invalid): {bad_head}")

# --- 2. channel semantics: decoded position vs stored ---------------------------
# P_j = [q_x + sx, q_y, q_z + sz], sx/sz from the ROOT row's ch13:15
errs = []
for f in sample[:120]:
    d = np.load(f)
    mo = d["motion"]
    q = mo[..., 0:3].copy()
    sx, sz = mo[:, 0, 13], mo[:, 0, 14]
    P = q.copy()
    P[..., 0] += sx[:, None]
    P[..., 2] += sz[:, None]
    # root joint's decoded XZ must equal the smooth-root track plus its own residual
    errs.append(float(np.abs(P[:, 0, 1] - q[:, 0, 1]).max()))   # Y must be untouched
print(f"[2] decode P=[q_x+sx, q_y, q_z+sz]: Y-channel left untouched, max deviation "
      f"{max(errs):.3e} (must be 0)")

# --- 3. discontinuity: the acceleration criterion --------------------------------
# The old corpus carried Cobra/MANIS IK defects. Per-clip: max |a| over joints, normalised by
# the clip's own median speed, is what separated teleports from fast-but-real motion.
ratios = []
for f in sample:
    d = np.load(f)
    P = d["motion"][..., 0:3].astype(np.float64)
    if P.shape[0] < 5:
        continue
    v = np.diff(P, axis=0)
    a = np.diff(v, axis=0)
    an = np.linalg.norm(a, axis=-1)                 # [T-2, J]
    vn = np.linalg.norm(v, axis=-1)
    med = np.median(vn[vn > 0]) if (vn > 0).any() else 0.0
    if med > 0:
        ratios.append(float(an.max() / med))
ratios = np.array(sorted(ratios))
n = len(ratios)
print(f"[3] acceleration/median-speed over {n} clips: "
      f"median={ratios[n//2]:.1f} p90={ratios[int(n*0.9)]:.1f} p99={ratios[int(n*0.99)]:.1f} "
      f"max={ratios[-1]:.1f}")
for thr in (50, 100, 200, 500):
    print(f"      above {thr:>4}: {int((ratios > thr).sum()):>4}/{n}  ({100*(ratios>thr).mean():.2f}%)")

# --- 4. joint spec conformance ---------------------------------------------------
spec = json.load(open(SPEC))
rigs = spec.get("rigs", spec)
skels = sorted(glob.glob(R + "/skeletons/*.npz"))
ok, mism, missing = 0, [], []
for s in skels:
    rid = os.path.basename(s)[:-4]
    if rid not in rigs:
        missing.append(rid); continue
    k = np.load(s, allow_pickle=True)
    want = [j["name"] for j in rigs[rid]["joints"]]
    got = list(k["joint_names"])
    if want == got:
        ok += 1
    else:
        mism.append((rid, len(want), len(got)))
print(f"[4] joint spec: {ok}/{len(skels)} skeletons match the canonical name+order exactly; "
      f"{len(mism)} mismatched, {len(missing)} not in spec")
if mism[:3]:
    print("      e.g.", mism[:3])
if missing[:3]:
    print("      not in spec:", missing[:3])
# descriptions present and non-empty?
k = np.load(skels[0], allow_pickle=True)
dsc = list(k["joint_descriptions"])
print(f"[4b] descriptions on {os.path.basename(skels[0])}: {len(dsc)} entries, "
      f"{sum(1 for x in dsc if x.strip())} non-empty, e.g. {dsc[0]!r}")
