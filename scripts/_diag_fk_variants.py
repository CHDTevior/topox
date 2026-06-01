"""Locate the FK double-rotation bug by brute-forcing rot_q construction variants
against the RELIABLE ground truth = RIC path (_recover_world_positions).

The correct variant has absL1 ~= 0 vs RIC and sweep ratio ~= 1.0. The current
port (A) is known to be ratio ~1.98 (double). We test reindex-to-parent vs
self-slot, and root correction (mul -r_rot_quat) vs direct overwrite vs none.
"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.anytop_dataset import AnyTopDataset, _recover_world_positions, _STD_FLOOR  # noqa
from src.data.anytop_rot6d_fk import (  # noqa
    _recover_root_quat_and_pos_np, _rotation_6d_to_matrix_np, _quat_from_transforms,
    _quat_to_matrix, _quat_mul, _quat_neg, _positions_global, recover_from_bvh_rot_np)

ds = AnyTopDataset(split="train", val_frac=0.05, seed=42,
                   data_root=str(ROOT / "data/anytop_planet_zoo_clean_L2"),
                   num_frames=64, max_joints=144, caption_emb_cache=None)
sidx = [i for i, s in enumerate(ds.samples) if "Saiga" in s["object_type"]][:8]


def getraw(it):
    J = int(it["num_joints"]); T = int(it["num_frames"])
    ax = np.asarray(it["anytop_x"], np.float32)
    mean = np.asarray(it["anytop_mean"], np.float32); std = np.asarray(it["anytop_std"], np.float32)
    raw = np.transpose(ax, (2, 0, 1))[:T, :J, :] * (std[:J][None] + _STD_FLOOR) + mean[:J][None]
    parents = np.asarray([int(p) for p in it["parent_indices"][:J]], int)
    offsets = np.asarray(it["rest_offsets"], np.float32)[:J]
    return raw.astype(np.float64), parents, offsets, T, J


best = None
for i in sidx:
    raw, parents, offsets, T, J = getraw(ds[i])
    ric = _recover_world_positions(raw.astype(np.float32))
    fk = recover_from_bvh_rot_np(raw, parents, offsets)
    l1 = float(np.abs(fk - ric).sum(-1).mean())
    if best is None or l1 > best[0]:
        best = (l1, raw, parents, offsets, T, J, ric)
l1, raw, parents, offsets, T, J, ric = best
print("Saiga max clip: absL1(cur)=%.4f  T=%d J=%d" % (l1, T, J), flush=True)

r_rot_quat, r_pos = _recover_root_quat_and_pos_np(raw[:, 0])
r_rot_mat = _quat_to_matrix(r_rot_quat)
nonroot_mat = _rotation_6d_to_matrix_np(raw[:, 1:, 3:9])
all_mat = np.concatenate([r_rot_mat[:, None], nonroot_mat], axis=1)
all_q = _quat_from_transforms(all_mat)  # all_q[:,0]=r_rot, [:,1:]=nonroot
pos = np.repeat(offsets[None], T, axis=0).astype(np.float64); pos[:, 0] = r_pos


def sweep(P):
    c = P[:, 1:, :].mean(1) - P[:, 0, :]
    a = np.degrees(np.unwrap(np.arctan2(c[:, 2], c[:, 0])))
    return a[-1] - a[0]


def absl1(P):
    return float(np.abs(P - ric).sum(-1).mean())


sweep_ric = sweep(ric)
print("RIC sweep=%.1f deg  (target ratio=1.00, target absL1~0)\n" % sweep_ric, flush=True)


def run(rot_q, name):
    P = _positions_global(rot_q, pos, parents).astype(np.float32)
    print("  %-26s absL1=%.4f  sweep=%7.1f  ratio=%.2f" % (name, absl1(P), sweep(P), sweep(P) / (sweep_ric if abs(sweep_ric) > 1e-6 else 1e-9)), flush=True)


def reindex():
    rq = np.zeros((T, J, 4)); rq[..., 0] = 1.0
    for j, p in enumerate(parents[1:], 1):
        rq[:, p] = all_q[:, j]
    return rq


rq = reindex(); rq[:, 0] = _quat_mul(_quat_neg(r_rot_quat), rq[:, 0]); run(rq, "A reindex+corr(CURRENT)")
run(reindex(), "B reindex,nocorr")
rq = all_q.copy(); rq[:, 0] = _quat_mul(_quat_neg(r_rot_quat), rq[:, 0]); run(rq, "C self-slot+corr")
run(all_q.copy(), "D self-slot,nocorr")
rq = all_q.copy(); rq[:, 0] = _quat_neg(r_rot_quat); run(rq, "E self,root=-rrot(overwrite)")
rq = all_q.copy(); rq[:, 0] = r_rot_quat.copy(); run(rq, "F self,root=+rrot(overwrite)")
rq = reindex(); rq[:, 0] = _quat_neg(r_rot_quat); run(rq, "G reindex,root=-rrot(overwrite)")
rq = reindex(); rq[:, 0] = r_rot_quat.copy(); run(rq, "H reindex,root=+rrot(overwrite)")
# I: self-slot, root=identity (no root rotation at all -> should be root-frame)
rq = all_q.copy(); rq[:, 0] = np.array([1.0, 0, 0, 0]); run(rq, "I self,root=identity")
