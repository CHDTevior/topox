"""CROSS-VALIDATE the FK double-rotation bug on the OLD AnyTop truebones dataset
(1071 official motions, paper-validated). If A(current, with root correction)
doubles and B(drop correction) restores FK==RIC here too, the bug is in the FK
recovery CODE (universal), NOT our clean_L2 data processing.

Scan all 1071 for largest root rotation (fast: angle of r_rot ch3:9[0]), then on
the top large-rotation clips compare A vs B (absL1 + sweep ratio vs RIC).
sweep RATIO is scale-invariant (angle), so it isolates double-rotation even if
motion were normalized; absL1~0 additionally confirms raw scale + exact match.
"""
import sys
import glob
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.anytop_dataset import _recover_world_positions  # noqa
from src.data.anytop_rot6d_fk import (  # noqa
    _recover_root_quat_and_pos_np, _rotation_6d_to_matrix_np, _quat_from_transforms,
    _quat_to_matrix, _positions_global, recover_from_bvh_rot_np)

D = ROOT / "data/anytop_truebones"
cond = np.load(D / "cond.npy", allow_pickle=True).item()
mfiles = sorted(glob.glob(str(D / "motions/*.npy")))
print("n motions:", len(mfiles), flush=True)


def species_of(mf):
    return Path(mf).stem.split("___")[0]


def root_sweep_fast(raw):  # Y-rotation angle of r_rot (ch3:9 of root), scale-free
    m = _rotation_6d_to_matrix_np(raw[:, 0, 3:9])  # [T,3,3]
    a = np.degrees(np.unwrap(np.arctan2(m[:, 0, 2], m[:, 0, 0])))
    return a[-1] - a[0]


def gsweep(P):  # global orientation sweep of recovered skeleton
    c = P[:, 1:, :].mean(1) - P[:, 0, :]
    a = np.degrees(np.unwrap(np.arctan2(c[:, 2], c[:, 0])))
    return a[-1] - a[0]


cands = []
for mf in mfiles:
    sp = species_of(mf)
    if sp not in cond:
        continue
    try:
        raw = np.load(mf).astype(np.float64)
    except Exception:
        continue
    if raw.ndim != 3 or raw.shape[-1] != 13 or raw.shape[0] < 10:
        continue
    cands.append((abs(root_sweep_fast(raw)), mf, sp, raw.shape[0], raw.shape[1]))
cands.sort(reverse=True)
print("\ntop12 largest root-rotation clips (|r_rot Y-sweep| deg):", flush=True)
for sw, mf, sp, T, J in cands[:12]:
    print("  %7.1f  %-32s T=%d J=%d" % (sw, Path(mf).stem[:32], T, J), flush=True)

print("\n=== RIC vs FK variant A(corr,current) vs B(nocorr) on top-6 ===", flush=True)
for sw, mf, sp, T, J in cands[:6]:
    raw = np.load(mf).astype(np.float64)
    par = np.asarray(cond[sp]["parents"], int)
    off = np.asarray(cond[sp]["offsets"], np.float64)
    ric = _recover_world_positions(raw.astype(np.float32))
    fkA = recover_from_bvh_rot_np(raw, par, off)
    # variant B: identical pipeline minus the root correction line
    r_rot_quat, r_pos = _recover_root_quat_and_pos_np(raw[:, 0])
    r_rot_mat = _quat_to_matrix(r_rot_quat)
    nonroot_mat = _rotation_6d_to_matrix_np(raw[:, 1:, 3:9])
    all_mat = np.concatenate([r_rot_mat[:, None], nonroot_mat], axis=1)
    all_q = _quat_from_transforms(all_mat)
    pos = np.repeat(off[None], T, axis=0).astype(np.float64); pos[:, 0] = r_pos
    rq = np.zeros((T, J, 4)); rq[..., 0] = 1.0
    for j, p in enumerate(par[1:], 1):
        rq[:, p] = all_q[:, j]
    fkB = _positions_global(rq, pos, par).astype(np.float32)
    sr = gsweep(ric) or 1e-9
    l1A = float(np.abs(fkA - ric).sum(-1).mean()); l1B = float(np.abs(fkB - ric).sum(-1).mean())
    # raw-scale check: recovered bone(0->1) length vs offsets[1] length
    bl = float(np.linalg.norm(ric[0, 1] - ric[0, 0])); ol = float(np.linalg.norm(off[1]))
    print("  %-30s RICsweep=%6.1f | A absL1=%.4f r=%.2f | B absL1=%.4f r=%.2f | bone/off=%.2f/%.2f" % (
        Path(mf).stem[:30], gsweep(ric), l1A, gsweep(fkA) / sr, l1B, gsweep(fkB) / sr, bl, ol), flush=True)
