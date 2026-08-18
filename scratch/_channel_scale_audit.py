#!/usr/bin/env python3
"""Per-channel scale audit on the TrueBones subset, in BOTH the normalised space (what the loss
actually sees) and the raw space (what is physically meaningful). Read-only.

An unweighted MSE over [T,J,13] is implicitly a weighting by each channel's variance in the
NORMALISED space. This measures that implicit weighting so we can replace it deliberately.
"""
import sys, pickle
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.anytop_dataset import AnyTopDataset, _STD_FLOOR

R = "data/animo4d_L4TB_plus_human_v4b272neutral"
tb = set(k for k in pickle.load(open(f"{R}/_cond_normalized_J144.pkl", "rb")).keys()
         if not k.startswith("PZ_") and not k.startswith("HML3D"))
ds = AnyTopDataset(data_root=R, split="all", num_frames=300, max_joints=144,
                   load_captions=False, caption_emb_cache=None, random_caption=False, augment=False)
print(f"[audit] dataset {len(ds)}", flush=True)

NAMES = ["ric_x","ric_y","ric_z", "r6d_0","r6d_1","r6d_2","r6d_3","r6d_4","r6d_5",
         "vel_x","vel_y","vel_z", "contact"]
norm_sq, raw_sq, n_frames = np.zeros(13), np.zeros(13), 0
seen = 0
for i in range(len(ds)):
    try:
        item = ds[i]
    except Exception:
        continue
    ot = item.get("object_type") or str(item.get("motion_id","?")).split("___")[0]
    if ot not in tb:
        continue
    T, J = int(item["num_frames"]), int(item["num_joints"])
    xn = np.asarray(item["anytop_x"])[:J, :, :T].transpose(2,0,1)          # [T,J,13] NORMALISED
    mean = np.asarray(item["anytop_mean"])[:J]; std = np.asarray(item["anytop_std"])[:J]
    xr = xn * (std[None] + _STD_FLOOR) + mean[None]                        # RAW
    norm_sq += (xn**2).sum(axis=(0,1)); raw_sq += (xr**2).sum(axis=(0,1))
    n_frames += T*J; seen += 1
    if seen >= 400:
        break

print(f"[audit] {seen} TrueBones clips, {n_frames} joint-frames\n")
nm, rm = norm_sq/n_frames, raw_sq/n_frames
print(f"{'channel':10s} {'E[x^2] norm':>13s} {'share of loss':>14s} {'E[x^2] raw':>13s}")
tot = nm.sum()
for i,(a,b) in enumerate(zip(nm, rm)):
    print(f"{NAMES[i]:10s} {a:13.4f} {100*a/tot:13.2f}% {b:13.4f}")
print(f"\n分组占比(未加权 MSE 下各组实际拿到的权重):")
for lab, sl in [("RIC 位置 0:3", slice(0,3)), ("rot6d 3:9", slice(3,9)),
                ("速度 9:12", slice(9,12)), ("contact 12", slice(12,13))]:
    print(f"  {lab:16s} {100*nm[sl].sum()/tot:6.2f}%   ({sl.stop-sl.start} 通道)")
