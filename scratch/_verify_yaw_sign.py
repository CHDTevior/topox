#!/usr/bin/env python3
"""Referee the heading sign against a quantity that involves NO rot6d at all: the world path tangent.

If the body turns left, the world root path must also curve left over the same clip (for locomotion
clips that actually travel). Whichever of yaw(q) / yaw(conj q) agrees with the path decides the
convention. Read-only.
"""
import sys, re
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.anytop_rot6d_fk import _recover_root_quat_and_pos_np, _quat_neg
from src.data.anytop_dataset import AnyTopDataset, _STD_FLOOR

def yaw_of(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z))

def net_deg(a):
    return float(np.degrees(np.unwrap(a)[-1] - np.unwrap(a)[0]))

ds = AnyTopDataset(data_root="data/animo4d_L4TB_plus_human_v4b272neutral", split="all",
                   num_frames=300, max_joints=144, load_captions=False,
                   caption_emb_cache=None, random_caption=False, augment=False)
print(f"[yaw] dataset {len(ds)} clips", flush=True)

PAT = re.compile(r"turn.?l|turnleft|walkturnl", re.I)
PATR = re.compile(r"turn.?r|turnright|walkturnr", re.I)
buckets = {"LEFT": [], "RIGHT": []}
for i in range(len(ds)):
    try:
        item = ds[i]
    except Exception:
        continue
    mid = str(item.get("motion_id", ""))
    lab = "LEFT" if PAT.search(mid) else ("RIGHT" if PATR.search(mid) else None)
    if lab is None:
        continue
    T, J = int(item["num_frames"]), int(item["num_joints"])
    x = np.asarray(item["anytop_x"])[:J, :, :T].transpose(2, 0, 1)
    raw = x * (np.asarray(item["anytop_std"])[:J][None] + _STD_FLOOR) + np.asarray(item["anytop_mean"])[:J][None]
    q, pos = _recover_root_quat_and_pos_np(raw[:, 0, :])
    d = np.diff(pos[:, [0, 2]], axis=0)                       # world XZ displacement
    step = np.linalg.norm(d, axis=-1)
    keep = step > np.percentile(step, 60)                     # only frames that actually travel
    if keep.sum() < 8:
        continue
    path = np.arctan2(d[keep, 0], d[keep, 1])                 # heading of travel in world XZ
    tot = float(np.linalg.norm(pos[-1, [0, 2]] - pos[0, [0, 2]]))
    buckets[lab].append((net_deg(yaw_of(q)), net_deg(yaw_of(_quat_neg(q))), net_deg(path), tot, mid))
    if sum(len(v) for v in buckets.values()) >= 120:
        break

for lab, rows in buckets.items():
    if not rows:
        print(f"{lab}: none found"); continue
    yq  = np.array([r[0] for r in rows]); yc = np.array([r[1] for r in rows])
    pth = np.array([r[2] for r in rows]); dist = np.array([r[3] for r in rows])
    m = np.abs(pth) > 15.0                                    # clips whose path really turns
    print(f"\n{lab}: n={len(rows)}  (with |path turn|>15deg: {m.sum()})")
    print(f"  mean yaw(q)      = {yq[m].mean():+8.1f} deg")
    print(f"  mean yaw(conj q) = {yc[m].mean():+8.1f} deg")
    print(f"  mean path turn   = {pth[m].mean():+8.1f} deg   (mean travel {dist[m].mean():.2f})")
    if m.sum():
        print(f"  sign(yaw(q))      == sign(path): {np.mean(np.sign(yq[m])==np.sign(pth[m])):.3f}")
        print(f"  sign(yaw(conj q)) == sign(path): {np.mean(np.sign(yc[m])==np.sign(pth[m])):.3f}")
    for r in rows[:3]:
        print(f"    {r[4][:52]:52s} yaw(q)={r[0]:+7.1f} conj={r[1]:+7.1f} path={r[2]:+7.1f}")
