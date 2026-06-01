"""Diagnose user's observation: is GT_FK double-rotating the root vs GT_RIC?

Take the max-mismatch Saiga clip, compute per-frame GLOBAL orientation (xz angle
of the non-root centroid relative to root) for both routes. If FK's total angle
sweep ~= 2x RIC's, it's a root double-rotation bug. Also report the per-frame
root-frame rotation each route implies, and where mismatch concentrates.
"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.anytop_dataset import AnyTopDataset, _recover_world_positions, _STD_FLOOR  # noqa
from src.data.anytop_rot6d_fk import recover_from_bvh_rot_np  # noqa

ds = AnyTopDataset(split="train", val_frac=0.05, seed=42,
                   data_root=str(ROOT / "data/anytop_planet_zoo_clean_L2"),
                   num_frames=64, max_joints=144, caption_emb_cache=None)

sidx = [i for i, s in enumerate(ds.samples) if "Saiga" in s["object_type"]][:8]
best = None
for i in sidx:
    it = ds[i]; J = int(it["num_joints"]); T = int(it["num_frames"])
    ax = np.asarray(it["anytop_x"], np.float32)
    mean = np.asarray(it["anytop_mean"], np.float32); std = np.asarray(it["anytop_std"], np.float32)
    raw = np.transpose(ax, (2, 0, 1))[:T, :J, :] * (std[:J][None] + _STD_FLOOR) + mean[:J][None]
    parents = [int(p) for p in it["parent_indices"][:J]]
    offsets = np.asarray(it["rest_offsets"], np.float32)[:J]
    ric = _recover_world_positions(raw)
    fk = recover_from_bvh_rot_np(raw, parents, offsets)
    l1 = float(np.abs(fk - ric).sum(-1).mean())
    if best is None or l1 > best[0]:
        best = (l1, ric, fk, it["object_type"], T)

l1, ric, fk, name, T = best
print("clip: %s  absL1=%.4f  T=%d" % (name, l1, T), flush=True)


def gangle(P):  # xz angle of (non-root centroid - root)
    c = P[:, 1:, :].mean(1) - P[:, 0, :]
    return np.degrees(np.arctan2(c[:, 2], c[:, 0]))


def unwrap(a):
    return np.degrees(np.unwrap(np.radians(a)))


ar = unwrap(gangle(ric)); af = unwrap(gangle(fk))
print("\nframe   RIC_ang   FK_ang   FK-RIC", flush=True)
for t in range(0, T, max(1, T // 12)):
    print("  %3d   %8.1f  %8.1f   %8.1f" % (t, ar[t], af[t], af[t] - ar[t]), flush=True)
sweep_r = ar[-1] - ar[0]; sweep_f = af[-1] - af[0]
print("\nRIC total angle sweep (f0->fT): %.1f deg" % sweep_r, flush=True)
print("FK  total angle sweep (f0->fT): %.1f deg" % sweep_f, flush=True)
print("ratio FK/RIC sweep = %.2f  (==2.0 would mean DOUBLE rotation)" % (sweep_f / (sweep_r if abs(sweep_r) > 1e-6 else 1e-6)), flush=True)

# per-joint mismatch: is it uniform (global rot) or concentrated (few joints)?
perj = np.abs(fk - ric).sum(-1).mean(0)  # [J]
order = np.argsort(-perj)
print("\nper-joint absL1 (top8 worst / and root):", flush=True)
print("  root(j0): %.4f" % perj[0], flush=True)
for j in order[:8]:
    print("  j%-3d: %.4f" % (j, perj[j]), flush=True)
print("  mean over joints: %.4f  median: %.4f" % (perj.mean(), np.median(perj)), flush=True)
