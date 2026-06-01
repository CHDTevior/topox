"""Run the USER'S original verification logic verbatim on RAW (un-clean) data:
recover_from_bvh_ric_np vs recover_from_bvh_rot_np, motion + cond from the SAME
original pipeline (data/anytop_planet_zoo). This is the authoritative baseline —
if the two routes agree here (small err), then any large err on the cleaned/
normalized dataset is introduced by our L2/normalize/reorder pipeline, NOT by the
data representation. Uses my self-contained verbatim numpy port of the official
recover funcs (root_diff=0 already verified).

Run on rose11: python scripts/_verify_official_rot_vs_ric_rawdata.py
"""
import csv
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# reuse the verbatim official recover funcs from the render script
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "offfk", str(ROOT / "scripts" / "_render_rot6d_official_fk.py"))
# we only want the functions, not the render side-effects -> load module text up
# to the "# ===== run" marker by importing carefully. Simpler: re-import the
# pure funcs by exec of just the function defs. They are side-effect free until
# the dataset loop. To avoid running the loop, we re-declare via import of the
# module is unsafe (it runs the loop). Instead, copy the 3 recover funcs here is
# heavy; better: guard the render script. For now, inline-import the helpers.

# --- minimal re-port (identical to _render_rot6d_official_fk.py) ---
def rotation_6d_to_matrix_np(c):
    x = c[..., 0:3] / np.linalg.norm(c[..., 0:3], axis=-1, keepdims=True)
    z = np.cross(x, c[..., 3:6], axis=-1); z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    y = np.cross(z, x, axis=-1)
    return np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)


def quat_from_transforms(ts):
    d0, d1, d2 = ts[..., 0, 0], ts[..., 1, 1], ts[..., 2, 2]
    q0 = np.sqrt(((d0 + d1 + d2 + 1) / 4).clip(0, None)); q1 = np.sqrt(((d0 - d1 - d2 + 1) / 4).clip(0, None))
    q2 = np.sqrt(((-d0 + d1 - d2 + 1) / 4).clip(0, None)); q3 = np.sqrt(((-d0 - d1 + d2 + 1) / 4).clip(0, None))
    c0 = (q0 >= q1) & (q0 >= q2) & (q0 >= q3); c1 = (q1 >= q0) & (q1 >= q2) & (q1 >= q3)
    c2 = (q2 >= q0) & (q2 >= q1) & (q2 >= q3); c3 = (q3 >= q0) & (q3 >= q1) & (q3 >= q2)
    q1[c0] *= np.sign(ts[c0, 2, 1] - ts[c0, 1, 2]); q2[c0] *= np.sign(ts[c0, 0, 2] - ts[c0, 2, 0]); q3[c0] *= np.sign(ts[c0, 1, 0] - ts[c0, 0, 1])
    q0[c1] *= np.sign(ts[c1, 2, 1] - ts[c1, 1, 2]); q2[c1] *= np.sign(ts[c1, 1, 0] + ts[c1, 0, 1]); q3[c1] *= np.sign(ts[c1, 0, 2] + ts[c1, 2, 0])
    q0[c2] *= np.sign(ts[c2, 0, 2] - ts[c2, 2, 0]); q1[c2] *= np.sign(ts[c2, 1, 0] + ts[c2, 0, 1]); q3[c2] *= np.sign(ts[c2, 2, 1] + ts[c2, 1, 2])
    q0[c3] *= np.sign(ts[c3, 1, 0] - ts[c3, 0, 1]); q1[c3] *= np.sign(ts[c3, 2, 0] + ts[c3, 0, 2]); q2[c3] *= np.sign(ts[c3, 2, 1] + ts[c3, 1, 2])
    return np.stack([q0, q1, q2, q3], axis=-1)


def qmul(sqs, oqs):
    sqs, oqs = np.broadcast_arrays(sqs, oqs)
    q0, q1, q2, q3 = sqs[..., 0], sqs[..., 1], sqs[..., 2], sqs[..., 3]
    r0, r1, r2, r3 = oqs[..., 0], oqs[..., 1], oqs[..., 2], oqs[..., 3]
    return np.stack([r0 * q0 - r1 * q1 - r2 * q2 - r3 * q3,
                     r0 * q1 + r1 * q0 - r2 * q3 + r3 * q2,
                     r0 * q2 + r1 * q3 + r2 * q0 - r3 * q1,
                     r0 * q3 - r1 * q2 + r2 * q1 + r3 * q0], axis=-1)


def qneg(qs):
    return qs * np.array([1.0, -1.0, -1.0, -1.0])


def qmulvec(qs, v):
    vs = np.concatenate([np.zeros(v.shape[:-1] + (1,)), v], axis=-1)
    return qmul(qs, qmul(vs, qneg(qs)))[..., 1:]


def qmat(qs):
    qw, qx, qy, qz = qs[..., 0], qs[..., 1], qs[..., 2], qs[..., 3]
    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    xx, yy, wx, xy, yz = qx * x2, qy * y2, qw * x2, qx * y2, qy * z2
    wy, xz, zz, wz = qw * y2, qx * z2, qz * z2, qw * z2
    m = np.empty(qs.shape[:-1] + (3, 3))
    m[..., 0, 0] = 1 - (yy + zz); m[..., 0, 1] = xy - wz; m[..., 0, 2] = xz + wy
    m[..., 1, 0] = xy + wz; m[..., 1, 1] = 1 - (xx + zz); m[..., 1, 2] = yz - wx
    m[..., 2, 0] = xz - wy; m[..., 2, 1] = yz + wx; m[..., 2, 2] = 1 - (xx + yy)
    return m


def pos_global(rq, pos, parents):
    F, J = rq.shape[:2]
    R = qmat(rq); loc = np.zeros((F, J, 4, 4)); loc[:, :, :3, :3] = R
    loc[:, :, :3, 3] = pos; loc[:, :, 3, 3] = 1
    g = np.zeros((F, J, 4, 4)); g[:, 0] = loc[:, 0]
    for i in range(1, J):
        g[:, i] = np.matmul(g[:, int(parents[i])], loc[:, i])
    p = g[:, :, :, 3]
    return p[:, :, :3] / p[:, :, 3, None]


def root_quat_pos(data):
    rq = quat_from_transforms(rotation_6d_to_matrix_np(data[:, 3:9]))
    rp = np.zeros(data.shape[:-1] + (3,)); rp[..., 1:, [0, 2]] = data[..., :-1, [9, 11]]
    rp = qmulvec(qneg(rq), rp); rp = np.cumsum(rp, axis=-2); rp[..., 1] = data[..., 1]
    return rq, rp


def ric(data):
    rq, rp = root_quat_pos(data[..., 0, :])
    p = data[..., 1:, :3].copy()
    nq = np.repeat(qneg(rq)[..., None, :], p.shape[-2], axis=-2)
    p = qmulvec(nq, p); p[..., 0] += rp[..., 0:1]; p[..., 2] += rp[..., 2:3]
    return np.concatenate([rp[..., None, :], p], axis=-2)


def rot(data, parents, offsets):
    rq, rp = root_quat_pos(data[:, 0])
    rmat = qmat(rq); nrm = rotation_6d_to_matrix_np(data[..., 1:, 3:9])
    allm = np.concatenate([rmat[:, None], nrm], axis=1); allq = quat_from_transforms(allm)
    T, J = allq.shape[:2]; rqj = np.zeros((T, J, 4)); rqj[..., 0] = 1
    for j, p in enumerate(parents[1:], 1):
        rqj[:, p] = allq[:, j]
    rqj[:, 0] = qmul(qneg(rq), rqj[:, 0])
    pos = np.repeat(offsets[None], T, axis=0).astype(float); pos[:, 0] = rp
    return pos_global(rqj, pos, parents)


# ---- run on RAW data with RAW cond (the user's original config) ----
RAW = ROOT / "data" / "anytop_planet_zoo"
cond = np.load(RAW / "cond.npy", allow_pickle=True).item()
with (RAW / "object_index.csv").open(newline="", encoding="utf-8") as f:
    objs = sorted([r["object_name"] for r in csv.DictReader(f)], key=len, reverse=True)


def obj_of(name):
    for o in objs:
        if name.startswith(o + "_"):
            return o
    return name.rsplit("_", 1)[0]


import os
mdir = RAW / "motions"
picked = []
for fn in sorted(os.listdir(mdir)):
    if "Asian_Water_Monitor" in fn or "Komodo_Dragon" in fn or "Grey_Seal" in fn:
        picked.append(fn)
    if len(picked) >= 3:
        break

for fn in picked:
    data = np.load(mdir / fn).astype(np.float64)
    o = obj_of(fn)
    parents = np.asarray(cond[o]["parents"], int)
    offsets = np.asarray(cond[o]["offsets"], float)
    pric = ric(data)
    prot = rot(data, parents, offsets)
    err = np.linalg.norm(prot - pric, axis=-1)
    bbox = pric.reshape(-1, 3)
    diag = np.linalg.norm(bbox.max(0) - bbox.min(0))
    print(f"{o} shape={list(data.shape)} J={parents.shape[0]} "
          f"mean_err={err.mean():.4f} p95={np.percentile(err,95):.4f} "
          f"root_err={err[:,0].mean():.4f} mean%bbox={err.mean()/diag*100:.2f} "
          f"p95%bbox={np.percentile(err,95)/diag*100:.2f}", flush=True)
print("DONE", flush=True)
