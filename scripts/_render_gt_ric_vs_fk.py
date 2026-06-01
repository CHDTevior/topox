"""Render GT_RIC (position route) vs GT_FK (rot6d FK route) for the PREFLIGHT
ACTUAL worst objects + a low-divergence control, picking each object's
MAX-mismatch clip (the real outlier), NOT just the first clip found.

This is the GT data's own two-route disagreement (= gt_fk_mismatch), no model.
Root-relative + bbox cubic axis. Title shows abs-L1 + L2/bbox%.

Run on rose11: python scripts/_render_gt_ric_vs_fk.py
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.anytop_dataset import AnyTopDataset, _recover_world_positions, _STD_FLOOR  # noqa
from src.data.anytop_rot6d_fk import recover_from_bvh_rot_np  # noqa

ds = AnyTopDataset(split="train", val_frac=0.05, seed=42,
                   data_root=str(ROOT / "data/anytop_planet_zoo_clean_L2"),
                   num_frames=64, max_joints=144, caption_emb_cache=None)
OUT = ROOT / "_qa_gt_ric_vs_fk"
OUT.mkdir(exist_ok=True)

# preflight ACTUAL worst objects (p95 50-87%) + one low control
WANT = ["Pronghorn_Antelope", "Saiga", "Western_Lowland_Gorilla",
        "Southern_White_Rhinoceros", "Komodo"]  # Komodo = reptile control


def get(it):
    J = int(it["num_joints"]); T = int(it["num_frames"])
    ax = np.asarray(it["anytop_x"], np.float32)
    mean = np.asarray(it["anytop_mean"], np.float32); std = np.asarray(it["anytop_std"], np.float32)
    raw = np.transpose(ax, (2, 0, 1))[:T, :J, :] * (std[:J][None] + _STD_FLOOR) + mean[:J][None]
    parents = [int(p) for p in it["parent_indices"][:J]]
    offsets = np.asarray(it["rest_offsets"], np.float32)[:J]
    ric = _recover_world_positions(raw)
    fk = recover_from_bvh_rot_np(raw, parents, offsets)
    return ric, fk, parents, it["object_type"]


def root_rel(P):
    Q = np.array(P, float, copy=True)
    Q[:, :, 0] -= P[:, 0:1, 0]
    Q[:, :, 2] -= P[:, 0:1, 2]
    return Q


def render(ric, fk, par, name, path):
    l1 = float(np.abs(fk - ric).sum(-1).mean())
    bb = ric.reshape(-1, 3); diag = float(np.linalg.norm(bb.max(0) - bb.min(0))) or 1e-9
    pct = 100 * float(np.linalg.norm(fk - ric, axis=-1).mean()) / diag
    g = root_rel(ric); f = root_rel(fk)
    T = g.shape[0]; idx = list(range(0, T, 2))
    if idx[-1] != T - 1:
        idx.append(T - 1)
    allc = np.concatenate([g.reshape(-1, 3), f.reshape(-1, 3)], 0)
    lo, hi = allc.min(0), allc.max(0); ctr = (lo + hi) * 0.5
    rad = max(float((hi - lo).max()) * 0.5, 1e-3) * 1.10
    fig = plt.figure(figsize=(12, 6))
    axes = [fig.add_subplot(1, 2, k + 1, projection='3d') for k in range(2)]

    def draw(ax, P, nm, col):
        ax.clear()
        for j, pj in enumerate(par):
            if 0 <= pj < P.shape[0] and j < P.shape[0]:
                ax.plot3D([P[j, 0], P[pj, 0]], [P[j, 2], P[pj, 2]], [P[j, 1], P[pj, 1]], color='#888', lw=1.4)
        ax.scatter3D(P[:, 0], P[:, 2], P[:, 1], c=col, s=12)
        ax.set_xlim(ctr[0] - rad, ctr[0] + rad); ax.set_ylim(ctr[2] - rad, ctr[2] + rad); ax.set_zlim(ctr[1] - rad, ctr[1] + rad)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.view_init(12, -70); ax.set_title(nm, fontsize=10)

    def upd(fi):
        draw(axes[0], g[fi], 'GT_RIC (position route) f%d' % fi, '#e74c3c')
        draw(axes[1], f[fi], 'GT_FK (rot6d-FK route) f%d' % fi, '#27ae60')
        fig.suptitle('%s   absL1=%.4f   L2/bbox=%.1f%%   (GT two recovery routes)' % (name, l1, pct), fontsize=11)
        return axes

    ani = FuncAnimation(fig, upd, frames=idx, blit=False)
    ani.save(path, writer=PillowWriter(fps=8))
    plt.close(fig)
    return l1, pct


# LIGHT: find target idx via ds.samples (dict, NO motion load), load only those.
idx_by = {t: [] for t in WANT}
for i, s in enumerate(ds.samples):
    o = s["object_type"]
    k = next((t for t in WANT if t in o), None)
    if k:
        idx_by[k].append(i)
print("target idx counts:", {k: len(v) for k, v in idx_by.items()}, flush=True)

CAP = 8  # load up to 8 clips per target, render the MAX-mismatch one
for k in WANT:
    idxs = idx_by[k]
    if not idxs:
        print("  %s: NOT in train split" % k, flush=True); continue
    best = None
    for i in idxs[:CAP]:
        ric, fk, par, name = get(ds[i])
        l1 = float(np.abs(fk - ric).sum(-1).mean())
        if best is None or l1 > best[0]:
            best = (l1, ric, fk, par, name)
    l1, ric, fk, par, name = best
    rl1, pct = render(ric, fk, par, name, str(OUT / ("WORST_%s.gif" % name)))
    tag = "reptile-control" if k == "Komodo" else "MAMMAL-outlier"
    print("  [%s] %-42s absL1=%.4f L2/bbox=%.1f%% -> WORST_%s.gif" % (tag, name, rl1, pct, name), flush=True)
print("DONE ->", OUT, flush=True)
