"""Re-render rot6d FK by faithfully RE-IMPLEMENTING the official AnyTop/SALAD
recover_from_bvh_rot_np in self-contained numpy (user request 2026-06-01),
fixing my earlier hand-rolled FK mistakes.

Every operator is a verbatim port of the authoritative source I read from the
user's pipeline + /tmp/salad:
  - rotation_6d_to_matrix_np  : utils/rotation_conversions.py:536 (x=norm(a1),
                                z=norm(a1xa2), y=zxx ; columns [x,y,z])
  - Quaternions.from_transforms / __mul__(q*q, q*vec) / __neg__ / transforms()
                              : /tmp/salad/visualization/Quaternions.py
  - positions_global (4x4 local->global matmul chain) : Animation.py
  - recover_root_quat_and_pos_np / recover_from_bvh_ric_np /
    recover_from_bvh_rot_np : motion_process.py:700/738/750 (verbatim)

NO external lib import (the SALAD lib needs numpy.core.umath_tests which is gone)
-> everything below is plain numpy.

Output: red = ric (0:3-position, the render path the user actually used),
blue = rot6d FK (official). SELF-CHECK: both share recover_root_quat_and_pos_np
so the ROOT trajectory MUST match ~0; the non-root diff is the real question.

Run on rose11: python scripts/_render_rot6d_official_fk.py
"""
import sys
import importlib.util
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ============== verbatim ports (no external deps) ==============
def rotation_6d_to_matrix_np(cont6d):
    """rotation_conversions.py:536 verbatim. cont6d[...,6] -> [...,3,3] cols [x,y,z]."""
    x_raw = cont6d[..., 0:3]
    y_raw = cont6d[..., 3:6]
    x = x_raw / np.linalg.norm(x_raw, axis=-1, keepdims=True)
    z = np.cross(x, y_raw, axis=-1)
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    y = np.cross(z, x, axis=-1)
    return np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)


def quat_from_transforms(ts):
    """Quaternions.from_transforms verbatim (rotation matrix -> wxyz quat)."""
    d0, d1, d2 = ts[..., 0, 0], ts[..., 1, 1], ts[..., 2, 2]
    q0 = (d0 + d1 + d2 + 1.0) / 4.0
    q1 = (d0 - d1 - d2 + 1.0) / 4.0
    q2 = (-d0 + d1 - d2 + 1.0) / 4.0
    q3 = (-d0 - d1 + d2 + 1.0) / 4.0
    q0 = np.sqrt(q0.clip(0, None)); q1 = np.sqrt(q1.clip(0, None))
    q2 = np.sqrt(q2.clip(0, None)); q3 = np.sqrt(q3.clip(0, None))
    c0 = (q0 >= q1) & (q0 >= q2) & (q0 >= q3)
    c1 = (q1 >= q0) & (q1 >= q2) & (q1 >= q3)
    c2 = (q2 >= q0) & (q2 >= q1) & (q2 >= q3)
    c3 = (q3 >= q0) & (q3 >= q1) & (q3 >= q2)
    q1[c0] *= np.sign(ts[c0, 2, 1] - ts[c0, 1, 2])
    q2[c0] *= np.sign(ts[c0, 0, 2] - ts[c0, 2, 0])
    q3[c0] *= np.sign(ts[c0, 1, 0] - ts[c0, 0, 1])
    q0[c1] *= np.sign(ts[c1, 2, 1] - ts[c1, 1, 2])
    q2[c1] *= np.sign(ts[c1, 1, 0] + ts[c1, 0, 1])
    q3[c1] *= np.sign(ts[c1, 0, 2] + ts[c1, 2, 0])
    q0[c2] *= np.sign(ts[c2, 0, 2] - ts[c2, 2, 0])
    q1[c2] *= np.sign(ts[c2, 1, 0] + ts[c2, 0, 1])
    q3[c2] *= np.sign(ts[c2, 2, 1] + ts[c2, 1, 2])
    q0[c3] *= np.sign(ts[c3, 1, 0] - ts[c3, 0, 1])
    q1[c3] *= np.sign(ts[c3, 2, 0] + ts[c3, 0, 2])
    q2[c3] *= np.sign(ts[c3, 2, 1] + ts[c3, 1, 2])
    qs = np.empty(ts.shape[:-2] + (4,))
    qs[..., 0] = q0; qs[..., 1] = q1; qs[..., 2] = q2; qs[..., 3] = q3
    return qs


def quat_mul_quat(sqs, oqs):
    """Quaternions.__mul__ (q*q) verbatim."""
    sqs, oqs = np.broadcast_arrays(sqs, oqs)
    q0 = sqs[..., 0]; q1 = sqs[..., 1]; q2 = sqs[..., 2]; q3 = sqs[..., 3]
    r0 = oqs[..., 0]; r1 = oqs[..., 1]; r2 = oqs[..., 2]; r3 = oqs[..., 3]
    out = np.empty(sqs.shape)
    out[..., 0] = r0 * q0 - r1 * q1 - r2 * q2 - r3 * q3
    out[..., 1] = r0 * q1 + r1 * q0 - r2 * q3 + r3 * q2
    out[..., 2] = r0 * q2 + r1 * q3 + r2 * q0 - r3 * q1
    out[..., 3] = r0 * q3 - r1 * q2 + r2 * q1 + r3 * q0
    return out


def quat_neg(qs):
    """Quaternions.__neg__ : conjugate (w,-x,-y,-z)."""
    return qs * np.array([1.0, -1.0, -1.0, -1.0])


def quat_mul_vec(qs, v):
    """Quaternions.__mul__ (q*vec): (self * (vs * -self)).imaginaries, vs=[0,v]."""
    vs = np.concatenate([np.zeros(v.shape[:-1] + (1,)), v], axis=-1)
    return quat_mul_quat(qs, quat_mul_quat(vs, quat_neg(qs)))[..., 1:]


def quat_to_matrix(qs):
    """Quaternions.transforms() verbatim (wxyz -> 3x3)."""
    qw = qs[..., 0]; qx = qs[..., 1]; qy = qs[..., 2]; qz = qs[..., 3]
    x2 = qx + qx; y2 = qy + qy; z2 = qz + qz
    xx = qx * x2; yy = qy * y2; wx = qw * x2; xy = qx * y2; yz = qy * z2
    wy = qw * y2; xz = qx * z2; zz = qz * z2; wz = qw * z2
    m = np.empty(qs.shape[:-1] + (3, 3))
    m[..., 0, 0] = 1.0 - (yy + zz); m[..., 0, 1] = xy - wz; m[..., 0, 2] = xz + wy
    m[..., 1, 0] = xy + wz; m[..., 1, 1] = 1.0 - (xx + zz); m[..., 1, 2] = yz - wx
    m[..., 2, 0] = xz - wy; m[..., 2, 1] = yz + wx; m[..., 2, 2] = 1.0 - (xx + yy)
    return m


def positions_global_from_quat(rotations_q, positions, parents):
    """Animation.positions_global verbatim: build 4x4 local transforms, chain
    multiply parent->child, return global xyz. rotations_q [F,J,4]; positions
    [F,J,3]; parents [J]."""
    F, J = rotations_q.shape[:2]
    R = quat_to_matrix(rotations_q)                       # [F,J,3,3]
    loc = np.zeros((F, J, 4, 4))
    loc[:, :, :3, :3] = R
    loc[:, :, :3, 3] = positions
    loc[:, :, 3, 3] = 1.0
    glob = np.zeros((F, J, 4, 4))
    glob[:, 0] = loc[:, 0]
    for i in range(1, J):
        glob[:, i] = np.matmul(glob[:, int(parents[i])], loc[:, i])
    p = glob[:, :, :, 3]
    return p[:, :, :3] / p[:, :, 3, None]


# ============== verbatim recover funcs (motion_process.py) ==============
def recover_root_quat_and_pos_np(data):  # :700
    r_rot_quat = quat_from_transforms(rotation_6d_to_matrix_np(data[:, 3:9]))  # [T,4]
    r_pos = np.zeros(data.shape[:-1] + (3,))
    r_pos[..., 1:, [0, 2]] = data[..., :-1, [9, 11]]
    r_pos = quat_mul_vec(quat_neg(r_rot_quat), r_pos)   # -r_rot_quat * r_pos
    r_pos = np.cumsum(r_pos, axis=-2)
    r_pos[..., 1] = data[..., 1]
    return r_rot_quat, r_pos


def recover_from_bvh_ric_np(data):  # :738 (0:3-position path = current render)
    r_rot_quat, r_pos = recover_root_quat_and_pos_np(data[..., 0, :])
    positions = data[..., 1:, :3].copy()
    nrep = positions.shape[-2]
    neg_q = np.repeat(quat_neg(r_rot_quat)[..., None, :], nrep, axis=-2)  # [T,J-1,4]
    positions = quat_mul_vec(neg_q, positions)
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    return np.concatenate([r_pos[..., None, :], positions], axis=-2)


def recover_from_bvh_rot_np(data, parents, offsets):  # :750 (rot6d FK path)
    r_rot_quat, r_pos = recover_root_quat_and_pos_np(data[:, 0])           # [T,4],[T,3]
    r_rot_mat = quat_to_matrix(r_rot_quat)                                  # [T,3,3]
    nonroot_mat = rotation_6d_to_matrix_np(data[..., 1:, 3:9])             # [T,J-1,3,3]
    allmat = np.concatenate([r_rot_mat[:, None], nonroot_mat], axis=1)     # [T,J,3,3]
    allq_hml = quat_from_transforms(allmat)                                 # [T,J,4]
    # parent reindex: rotations[:,p] = hml[:,j]  (motion_process.py:758-759)
    T, J = allq_hml.shape[:2]
    rot_q = np.zeros((T, J, 4)); rot_q[..., 0] = 1.0   # identity wxyz
    for j, p in enumerate(parents[1:], 1):
        rot_q[:, p] = allq_hml[:, j]
    # root: rotations[:,0] = -r_rot_quat * rotations[:,0]
    rot_q[:, 0] = quat_mul_quat(quat_neg(r_rot_quat), rot_q[:, 0])
    pos = np.repeat(offsets[None], T, axis=0).astype(float)               # [T,J,3] bone offsets
    pos[:, 0] = r_pos
    return positions_global_from_quat(rot_q, pos, parents)


# ============== run on long-chain species ==============
from src.data.anytop_dataset import AnyTopDataset  # noqa: E402
_spec = importlib.util.spec_from_file_location("aa13", str(ROOT / "scripts" / "animate_anytop13.py"))
aa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(aa)

OUT = ROOT / "runs" / "_rot6d_official_fk"
OUT.mkdir(parents=True, exist_ok=True)
# Authoritative rest-pose offsets + parents come from cond.npy (the user's
# pipeline FK route uses cond[obj]['offsets'] / ['parents'], NOT the dataset
# item's rest_offsets — those are a different/placeholder version, verified
# 2026-06-01: item[2]=[0,0,0] vs cond[2]=[0,0.004,0.053], diff mean=0.079).
COND = np.load(str(ROOT / "data/anytop_planet_zoo_clean_L2/cond.npy"),
               allow_pickle=True).item()
ds = AnyTopDataset(split="val", val_frac=0.05, seed=42,
                   data_root=str(ROOT / "data/anytop_planet_zoo_clean_L2"),
                   num_frames=64, max_joints=144, caption_emb_cache=None)
want = ["PZ_Asian_Water_Monitor_Male", "PZ_Komodo_Dragon_Male",
        "PZ_Saltwater_Crocodile_Male", "PZ_Grey_Seal_Male"]
done = set()
for i in range(len(ds)):
    it = ds[i]
    sp = it["object_type"]
    if sp not in want or sp in done:
        continue
    done.add(sp)
    J = int(it["num_joints"]); T = int(it["num_frames"])
    ax = np.asarray(it["anytop_x"], np.float32)
    mean = np.asarray(it["anytop_mean"], np.float32); std = np.asarray(it["anytop_std"], np.float32)
    raw = (np.transpose(ax, (2, 0, 1))[:T, :J, :] * (std[:J][None] + 1e-6) + mean[:J][None]).astype(np.float64)
    # FK inputs from cond.npy (authoritative), matching the user's pipeline route.
    co = COND[sp]
    parents = np.asarray(co["parents"], dtype=int)
    offsets = np.asarray(co["offsets"], dtype=np.float64)
    assert parents.shape[0] == J and offsets.shape[0] == J, \
        f"{sp}: cond J={parents.shape[0]} vs item J={J} mismatch"

    ric = recover_from_bvh_ric_np(raw).astype(np.float32)              # red
    rot = recover_from_bvh_rot_np(raw, parents, offsets).astype(np.float32)  # blue

    root_diff = np.abs(ric[:, 0] - rot[:, 0]).mean()
    nr = slice(1, J)
    nr_diff = np.abs(ric[:, nr] - rot[:, nr]).mean()
    scale = float(np.abs(ric[:, nr]).mean())
    print(f"{sp} J={J} T={T} root_diff={root_diff:.4f} nonroot|ric-rot|={nr_diff:.4f} "
          f"(scale={scale:.4f} rel={nr_diff/max(scale,1e-9):.2f})", flush=True)
    ttl = f"{sp} RED=ric(0:3) BLUE=rot6dFK(official) nr={nr_diff:.3f}"
    aa.contact_sheet(rot, ric, list(parents), str(OUT / f"{sp}_sheet_obl.png"), ttl, elev=12, azim=-70)
    aa.contact_sheet(rot, ric, list(parents), str(OUT / f"{sp}_sheet_top.png"), ttl, elev=75, azim=-90)
    aa.animate_clip(rot, ric, list(parents), str(OUT / f"{sp}_gtvs.gif"), ttl, 2, 12)

print("DONE", flush=True)
