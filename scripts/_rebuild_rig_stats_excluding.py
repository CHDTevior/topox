#!/usr/bin/env python3
"""Recompute per-(rig,joint,channel) statistics with teleport-affected data removed.

The corpus is frozen and sha-pinned, so nothing here writes to dataset/. The exclusion list is an
explicit, hashable artifact and the rebuilt statistics record which list produced them; training
pins both, so a run can always be traced back to exactly what it was allowed to see.

Exclusion granularity is whatever the list says:
    {"clips": {"<clip_id>": "all"}}                  drop the whole clip
    {"clips": {"<clip_id>": [12, 13, 14]}}           drop those frames only
Frame-level drops also remove the frame BEFORE each dropped frame from the velocity channels
(9:12), because ch9:11 is a forward difference: v[t] describes the step t -> t+1, so a discontinuity
at t+1 corrupts v[t], not v[t+1].
"""
import json, sys, hashlib
from pathlib import Path
import numpy as np

D = Path("dataset/ktjd17_pz_human312")
EXC = Path(sys.argv[1])
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "data/pzh312_rig_stats_excluded.npz")
SRC = np.load(D.parent / "ktjd17_pz_human312_species_stats/rig_stats.npz", allow_pickle=True)

exc = json.loads(EXC.read_text())
drop = exc["clips"]
exc_sha = hashlib.sha256(EXC.read_bytes()).hexdigest()
print(f"exclusion list {EXC} sha {exc_sha[:16]} -- {len(drop)} clips affected")

rig_ids = [str(r) for r in SRC["rig_ids"]]
ridx = {r: i for i, r in enumerate(rig_ids)}
R, J, C = SRC["mean"].shape
n = np.zeros((R, J, C), np.int64)
s1 = np.zeros((R, J, C), np.float64)
s2 = np.zeros((R, J, C), np.float64)
lo = np.full((R, J, C), np.inf)
hi = np.full((R, J, C), -np.inf)
n_clip = np.zeros(R, np.int64); n_frame = np.zeros(R, np.int64)
n_head = np.zeros(R, np.int64)          # SURVIVING heading-valid frames, not the original count
dropped_clips = dropped_frames = 0

rows = [json.loads(l) for l in (D / "manifests/clips.jsonl").open()]
for i, r in enumerate(rows):
    cid, rig = r["clip_id"], r["rig_id"]
    d = drop.get(cid)
    if d == "all":
        dropped_clips += 1
        continue
    z = np.load(D / r["motion_relpath"], allow_pickle=True)
    m = np.asarray(z["motion"] if "motion" in z else z[list(z.keys())[0]], np.float64)
    hv = np.asarray(z["heading_valid"], bool)
    keep = np.ones(m.shape[0], bool)
    if d:
        f = np.asarray(d, int)
        keep[f] = False
        prev = f - 1                                  # forward-difference neighbour, see docstring
        keep[prev[prev >= 0]] = False
        dropped_frames += int((~keep).sum())
    m, hv = m[keep], hv[keep]
    if not len(m):
        dropped_clips += 1
        continue
    k, Jr = ridx[rig], m.shape[1]
    # THE SOURCE CONTRACT, reproduced exactly (src/data/ktjd17/species_stats.py:220-250). Summing
    # every channel over every frame -- what this script did before -- pulls the ZEROS that invalid
    # heading frames carry in ch15:17 into the heading mean and std. codex measured the damage:
    # 102 of 312 rigs wrong, 8,857 invalid-heading frames counted, 3,214 of them HML3D_Human.
    #   ch0:13   every joint, every frame
    #   ch13:15  ROOT ONLY (smooth-root XZ), every frame
    #   ch15:17  ROOT ONLY (heading cos/sin), ONLY frames where heading_valid
    T = m.shape[0]
    c = m[:, :, :13]
    n[k, :Jr, :13] += T
    s1[k, :Jr, :13] += c.sum(0); s2[k, :Jr, :13] += (c ** 2).sum(0)
    np.minimum(lo[k, :Jr, :13], c.min(0), out=lo[k, :Jr, :13])
    np.maximum(hi[k, :Jr, :13], c.max(0), out=hi[k, :Jr, :13])
    rxz = m[:, 0, 13:15]
    n[k, 0, 13:15] += T
    s1[k, 0, 13:15] += rxz.sum(0); s2[k, 0, 13:15] += (rxz ** 2).sum(0)
    np.minimum(lo[k, 0, 13:15], rxz.min(0), out=lo[k, 0, 13:15])
    np.maximum(hi[k, 0, 13:15], rxz.max(0), out=hi[k, 0, 13:15])
    nh = int(hv.sum())
    if nh:
        hd = m[hv, 0, 15:17]
        n[k, 0, 15:17] += nh
        s1[k, 0, 15:17] += hd.sum(0); s2[k, 0, 15:17] += (hd ** 2).sum(0)
        np.minimum(lo[k, 0, 15:17], hd.min(0), out=lo[k, 0, 15:17])
        np.maximum(hi[k, 0, 15:17], hd.max(0), out=hi[k, 0, 15:17])
    n_head[k] += nh
    n_clip[k] += 1; n_frame[k] += T
    if i % 20000 == 0:
        print(f"  ...{i}/{len(rows)}", flush=True)

mean = np.where(n > 0, s1 / np.maximum(n, 1), 0.0)
var = np.where(n > 0, s2 / np.maximum(n, 1) - mean ** 2, 0.0)
std = np.sqrt(np.maximum(var, 0.0))
lo[~np.isfinite(lo)] = 0.0; hi[~np.isfinite(hi)] = 0.0
vm = np.asarray(SRC["valid_mask"])                   # channel validity is structural, not statistical
print(f"\ndropped {dropped_clips} whole clips, {dropped_frames} individual frames")
print(f"rigs with zero surviving clips: {int((n_clip == 0).sum())}")
old_std = np.asarray(SRC["std"])
ch = (np.abs(std - old_std) > 1e-9) & vm
print(f"cells whose std changed: {int(ch.sum()):,} of {int(vm.sum()):,} valid")
hv_old = np.asarray(SRC["heading_valid_frame_count"])
print(f"heading-valid frames: {int(hv_old.sum()):,} -> {int(n_head.sum()):,} "
      f"({int(hv_old.sum() - n_head.sum()):,} removed with the cut clips)")
print(f"max |std change| on valid cells: {float(np.abs(std - old_std)[vm].max()):.6f}")
np.savez_compressed(OUT, rig_ids=np.array(rig_ids), joint_count=SRC["joint_count"],
                    channel_names=SRC["channel_names"], mean=mean, std=std, count=n,
                    valid_mask=vm, minimum=lo, maximum=hi, clip_count=n_clip,
                    frame_count=n_frame,
                    biological_species_ids=SRC["biological_species_ids"],
                    heading_valid_frame_count=n_head,
                    __exclusion_sha256=np.array(exc_sha),
                    __source_sha256=np.array(hashlib.sha256(
                        (D.parent / "ktjd17_pz_human312_species_stats/rig_stats.npz").read_bytes()
                    ).hexdigest()))
print(f"[OK] wrote {OUT} sha {hashlib.sha256(OUT.read_bytes()).hexdigest()[:16]}")
