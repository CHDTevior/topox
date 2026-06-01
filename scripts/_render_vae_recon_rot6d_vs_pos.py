"""End-to-end: use the FROZEN VAE (the one diffusion trains on) to reconstruct
long-chain clips, then render the reconstruction TWO ways and compare:
  RED  = pos/RIC path   : recover_from_bvh_ric_np(pred_13ch)  (channel 0:3)
  BLUE = rot6d FK path  : recover_from_bvh_rot_np(pred_13ch, parents, offsets)
                          (channel 3:9 official FK, offsets/parents from the
                           SAME dataset item -> same new_to_old_perm ordering,
                           so it's aligned, unlike my earlier cond.npy mistake)

Also a GT self-check: GT ric vs GT rot must agree (<~1%bbox) using item's own
rest_offsets/parents -> confirms the dataset fields are self-consistent.

This shows how good the VAE's reconstructed ROTATION channels are (blue) vs its
reconstructed POSITION channels (red) — the question the user asked.

Run on rose11: python scripts/_render_vae_recon_rot6d_vs_pos.py
"""
import sys
import importlib.util
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---- verbatim official recover funcs (root_diff=0 verified; <1% on raw data) ----
def r6m(c):
    x = c[..., 0:3] / np.linalg.norm(c[..., 0:3], axis=-1, keepdims=True)
    z = np.cross(x, c[..., 3:6], axis=-1); z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    y = np.cross(z, x, axis=-1)
    return np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)


def qft(ts):
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


def qm(s, o):
    s, o = np.broadcast_arrays(s, o)
    q0, q1, q2, q3 = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    r0, r1, r2, r3 = o[..., 0], o[..., 1], o[..., 2], o[..., 3]
    return np.stack([r0*q0-r1*q1-r2*q2-r3*q3, r0*q1+r1*q0-r2*q3+r3*q2,
                     r0*q2+r1*q3+r2*q0-r3*q1, r0*q3-r1*q2+r2*q1+r3*q0], axis=-1)


def qn(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def qmv(q, v):
    vs = np.concatenate([np.zeros(v.shape[:-1] + (1,)), v], axis=-1)
    return qm(q, qm(vs, qn(q)))[..., 1:]


def qmat(q):
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    x2, y2, z2 = qx+qx, qy+qy, qz+qz
    xx, yy, wx, xy, yz = qx*x2, qy*y2, qw*x2, qx*y2, qy*z2
    wy, xz, zz, wz = qw*y2, qx*z2, qz*z2, qw*z2
    m = np.empty(q.shape[:-1] + (3, 3))
    m[..., 0, 0] = 1-(yy+zz); m[..., 0, 1] = xy-wz; m[..., 0, 2] = xz+wy
    m[..., 1, 0] = xy+wz; m[..., 1, 1] = 1-(xx+zz); m[..., 1, 2] = yz-wx
    m[..., 2, 0] = xz-wy; m[..., 2, 1] = yz+wx; m[..., 2, 2] = 1-(xx+yy)
    return m


def pg(rq, pos, par):
    F, J = rq.shape[:2]; R = qmat(rq); loc = np.zeros((F, J, 4, 4))
    loc[:, :, :3, :3] = R; loc[:, :, :3, 3] = pos; loc[:, :, 3, 3] = 1
    g = np.zeros((F, J, 4, 4)); g[:, 0] = loc[:, 0]
    for i in range(1, J):
        g[:, i] = np.matmul(g[:, int(par[i])], loc[:, i])
    p = g[:, :, :, 3]; return p[:, :, :3] / p[:, :, 3, None]


def rqp(data):
    rq = qft(r6m(data[:, 3:9])); rp = np.zeros(data.shape[:-1] + (3,))
    rp[..., 1:, [0, 2]] = data[..., :-1, [9, 11]]
    rp = qmv(qn(rq), rp); rp = np.cumsum(rp, axis=-2); rp[..., 1] = data[..., 1]
    return rq, rp


def ric(data):
    rq, rp = rqp(data[..., 0, :]); p = data[..., 1:, :3].copy()
    nq = np.repeat(qn(rq)[..., None, :], p.shape[-2], axis=-2)
    p = qmv(nq, p); p[..., 0] += rp[..., 0:1]; p[..., 2] += rp[..., 2:3]
    return np.concatenate([rp[..., None, :], p], axis=-2)


def rot(data, par, off):
    rq, rp = rqp(data[:, 0]); rm = qmat(rq); nrm = r6m(data[..., 1:, 3:9])
    allm = np.concatenate([rm[:, None], nrm], axis=1); allq = qft(allm)
    T, J = allq.shape[:2]; rqj = np.zeros((T, J, 4)); rqj[..., 0] = 1
    for j, p in enumerate(par[1:], 1):
        rqj[:, p] = allq[:, j]
    rqj[:, 0] = qm(qn(rq), rqj[:, 0])
    pos = np.repeat(off[None], T, axis=0).astype(float); pos[:, 0] = rp
    return pg(rqj, pos, par)


# ---- VAE load (same as animate_anytop13) ----
from src.data.anytop_dataset import AnyTopDataset, _STD_FLOOR  # noqa: E402
from src.models.graph_salad import GraphMotionBatch, GraphMotionVAE  # noqa: E402
_spec = importlib.util.spec_from_file_location("aa13", str(ROOT / "scripts" / "animate_anytop13.py"))
aa = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(aa)

VAE_CKPT = str(ROOT / "runs/_baseline_cleanL2_ep34_for_p1diag_compare/best_recon_model.pt")
OUT = ROOT / "runs" / "_vae_recon_rot6d_vs_pos"; OUT.mkdir(parents=True, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae, ta = aa.load_anytop13_vae(VAE_CKPT, dev)
from src.data.anytop_dataset import collate_fn  # noqa: E402

ds = AnyTopDataset(split="val", val_frac=0.05, seed=42,
                   data_root=str(ROOT / "data/anytop_planet_zoo_clean_L2"),
                   num_frames=64, max_joints=144, caption_emb_cache=None)
want = ["PZ_Asian_Water_Monitor_Male", "PZ_Komodo_Dragon_Male",
        "PZ_Saltwater_Crocodile_Male", "PZ_Grey_Seal_Male"]
done = set()
with torch.no_grad():
    for i in range(len(ds)):
        it = ds[i]
        sp = it["object_type"]
        if sp not in want or sp in done:
            continue
        done.add(sp)
        J = int(it["num_joints"]); T = int(it["num_frames"])
        parents = np.asarray([int(p) for p in it["parent_indices"][:J]], dtype=int)
        offsets = np.asarray(it["rest_offsets"], np.float32)[:J].astype(np.float64)  # SAME perm as motion
        raw = collate_fn([it])
        raw = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in raw.items()}
        batch = GraphMotionBatch.from_collate_dict(raw)
        out = vae(batch, sample=False)
        Tv = int(out["frame_mask_recovered"][0].sum().item()); T = min(T, Tv)
        std = raw["anytop_std"][0, :J].cpu().numpy(); mean = raw["anytop_mean"][0, :J].cpu().numpy()
        pred = out["pred_motion"][0, :T, :J, :].cpu().numpy() * (std[None] + _STD_FLOOR) + mean[None]  # [T,J,13] raw
        pred = pred.astype(np.float64)

        # GT self-check (item fields self-consistent?)
        gt = (np.transpose(np.asarray(it["anytop_x"], np.float32), (2, 0, 1))[:T, :J, :]
              * (std[None] + _STD_FLOOR) + mean[None]).astype(np.float64)
        gt_ric = ric(gt); gt_rot = rot(gt, parents, offsets)
        gtbbox = np.linalg.norm(gt_ric.reshape(-1, 3).max(0) - gt_ric.reshape(-1, 3).min(0))
        gt_err = np.linalg.norm(gt_ric - gt_rot, axis=-1)

        # VAE recon: two render paths
        pred_ric = ric(pred).astype(np.float32)      # RED  (pos channel)
        pred_rot = rot(pred, parents, offsets).astype(np.float32)  # BLUE (rot6d channel)
        rr_err = np.linalg.norm(pred_ric - pred_rot, axis=-1)

        print(f"{sp} J={J} T={T} | GT_self_check ric-vs-rot mean%bbox={gt_err.mean()/gtbbox*100:.2f} "
              f"p95%bbox={np.percentile(gt_err,95)/gtbbox*100:.2f} (should be <~2%) | "
              f"VAE recon pos-vs-rot6d mean%bbox={rr_err.mean()/gtbbox*100:.2f}", flush=True)

        # Render: BLUE=rot6d-FK recon  vs  RED=pos-RIC recon
        ttl = f"{sp} RED=recon-pos(0:3) BLUE=recon-rot6dFK(3:9) GTchk={gt_err.mean()/gtbbox*100:.1f}%"
        aa.contact_sheet(pred_rot, pred_ric, list(parents), str(OUT / f"{sp}_sheet_obl.png"), ttl, elev=12, azim=-70)
        aa.contact_sheet(pred_rot, pred_ric, list(parents), str(OUT / f"{sp}_sheet_top.png"), ttl, elev=75, azim=-90)
        aa.animate_clip(pred_rot, pred_ric, list(parents), str(OUT / f"{sp}_gtvs.gif"), ttl, 2, 12)

print("DONE", flush=True)
