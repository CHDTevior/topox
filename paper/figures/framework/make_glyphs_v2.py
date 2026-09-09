"""Bolder glyphs for the framework figure (v6): thick round-capped bones, white-filled joint dots, saturated per-state colours.
States: rest (teal), noisy target frames (indigo, jittered), predicted clean frames (ink), final motion strip (clay gradient)."""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "glyphs_v2"; OUT.mkdir(exist_ok=True)
DUMP = ROOT / "renders/clean_vis_ep289_g0/A_PZ_Alaskan_Moose_Female__d34d76e2c0a3e5b0d5e3.world.npz"
TEAL, TEAL_D, INDIGO, INK, CLAY, CLAY_L, GOLD = "#1F7F93", "#0E4F5E", "#5B5FA8", "#2B2B2B", "#B5552F", "#E8B79E", "#D9A441"
z = np.load(DUMP, allow_pickle=True); gen, rest, par = z["gen_fk"], z["demo_w"][0], z["parents"].astype(int); names = [str(s) for s in z["joint_names"]]
T, J = gen.shape[:2]
hips = next(i for i, s in enumerate(names) if "hips" in s.lower()); head = next(i for i, s in enumerate(names) if s.lower().endswith("head_joint"))
hv = (gen[:, head] - gen[:, hips]).mean(0); hv[1] = 0; hv /= np.linalg.norm(hv)
side = lambda P: (P @ hv, P[..., 1])
def draw(ax, P, col, lw, dot, edge=None, noise=0.0, seed=0, alpha=1.0, shadow=True, s_bl=1.0):
    Q = P + (np.random.default_rng(seed).normal(0, noise, P.shape) if noise > 0 else 0)
    x, y = side(Q)
    if shadow:
        ax.add_patch(Ellipse(((x.max() + x.min()) / 2, y.min() - 0.02 * s_bl), (x.max() - x.min()) * 1.05, 0.35 * s_bl, color="#000000", alpha=0.10 * alpha, lw=0, zorder=1))
    for j in range(J):
        p = par[j]
        if p < 0: continue
        ax.plot([x[j], x[p]], [y[j], y[p]], color=col, lw=lw, alpha=alpha, solid_capstyle="round", zorder=2)
    ax.scatter(x, y, s=dot * 0.45, color=edge or col, alpha=alpha, linewidths=0, zorder=3)
    if noise > 0:                                  # noise speckle around a noisy frame
        rng = np.random.default_rng(seed + 100); n = 60
        ax.scatter(x.mean() + rng.normal(0, np.ptp(x) * 0.55, n), y.mean() + rng.normal(0, np.ptp(y) * 0.55, n), s=dot * 0.35, color=col, alpha=0.35, linewidths=0, zorder=1)
    ax.set_aspect("equal"); ax.axis("off"); return x, y
def frame(ax, x, y, pad=0.10):
    cx, cy = (x.max() + x.min()) / 2, (y.max() + y.min()) / 2; r = max(np.ptp(x), np.ptp(y)) * (0.5 + pad)
    ax.set_xlim(cx - r, cx + r); ax.set_ylim(cy - r * 0.95, cy + r * 0.85)
sk = np.load(ROOT / "dataset/ktjd17_pzh312_noik_v2/skeletons/PZ_Alaskan_Moose_Female.npz", allow_pickle=True)
bl = float(np.linalg.norm(sk["offset_parent_local"][par >= 0], axis=1).mean()); body = float(np.ptp(side(rest)[1]))
# rest pose, teal, bold
fig, ax = plt.subplots(figsize=(1.6, 1.4)); x, y = draw(ax, rest, TEAL, 3.8, 14, edge=TEAL_D, s_bl=body); frame(ax, x, y); fig.savefig(OUT / "rest.png", dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
# noisy target frames (indigo, jittered) and predicted clean frames (ink)
idx = np.linspace(0, T - 1, 4).astype(int)
for k, f in enumerate(idx):
    for tag, nz, col, edge in (("noisy", 0.6 * bl, INDIGO, "#3A3D7A"), ("clean", 0.0, INK, INK)):
        fig, ax = plt.subplots(figsize=(1.3, 1.3)); x, y = draw(ax, gen[f], col, 2.6 if nz else 3.0, 9, edge=edge, noise=nz * 0.6, seed=f, alpha=0.8 if nz else 1.0, shadow=not nz, s_bl=body)
        xc, yc = side(gen[f]); frame(ax, xc, yc, pad=0.28 if nz else 0.10)
        fig.savefig(OUT / f"{tag}_{k}.png", dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
# motion strip: 5 poses, clay gradient light -> dark, one ground shadow each
fig, ax = plt.subplots(figsize=(3.6, 1.3))
fs = np.linspace(0, T - 1, 5).astype(int)
for i, f in enumerate(fs):
    t = i / (len(fs) - 1); col = tuple(np.array(matplotlib.colors.to_rgb(CLAY_L)) * (1 - t) + np.array(matplotlib.colors.to_rgb(CLAY)) * t)
    draw(ax, gen[f], col, 2.6, 8, edge=CLAY, s_bl=body)
x, y = side(gen.reshape(-1, 3)); ax.set_xlim(x.min() - 0.4 * bl, x.max() + 0.4 * bl); ax.set_ylim(y.min() - 0.5 * bl, y.max() + 0.4 * bl)
fig.savefig(OUT / "motion_strip.png", dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
# three rigs (rest poses, side view) with the same bold treatment, different topologies
sys.path.insert(0, str(ROOT / "paper/figures")); from skeleton_art import load as sk_load, project as sk_project
for rig, fname in (("PZ_Alaskan_Moose_Female", "rig_moose.png"), ("PZ_Gharial_Male", "rig_gharial.png"), ("PZ_Red_Ruffed_Lemur_Male", "rig_lemur.png")):
    P, par_, names_ = sk_load(ROOT / f"dataset/ktjd17_pzh312_noik_v2/skeletons/{rig}.npz"); x, y, dpt = sk_project(P, "side")
    fig, ax = plt.subplots(figsize=(1.7, 1.2))
    bodyr = float(np.ptp(y)); ax.add_patch(Ellipse(((x.max() + x.min()) / 2, y.min() - 0.02 * bodyr), (x.max() - x.min()) * 1.05, 0.3 * bodyr, color="#000000", alpha=0.10, lw=0, zorder=1))
    for j in np.argsort(dpt):
        p_ = par_[j]
        if p_ < 0: continue
        ax.plot([x[j], x[p_]], [y[j], y[p_]], color=TEAL, lw=3.0, solid_capstyle="round", zorder=2)
    ax.scatter(x, y, s=5, color=TEAL_D, linewidths=0, zorder=3); ax.set_aspect("equal"); ax.axis("off")
    cx, cy = (x.max() + x.min()) / 2, (y.max() + y.min()) / 2; r = max(np.ptp(x), np.ptp(y)) * 0.58; ax.set_xlim(cx - r, cx + r); ax.set_ylim(cy - r * 0.75, cy + r * 0.7)
    fig.savefig(OUT / fname, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
# tree-distance bias matrix, indigo colormap to match the transformer's attention motif
G = np.full((J, J), np.inf); G[np.arange(J), np.arange(J)] = 0
for j in range(J):
    if par[j] >= 0: G[j, par[j]] = G[par[j], j] = 1
for k in range(J): G = np.minimum(G, G[:, [k]] + G[[k], :])
fig, ax = plt.subplots(figsize=(0.8, 0.8)); ax.imshow(-np.minimum(G, 8), cmap="Purples_r", interpolation="nearest"); ax.axis("off")
fig.savefig(OUT / "bias.png", dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.0); plt.close(fig)
print("[glyphs v2] wrote", sorted(p.name for p in OUT.glob("*.png")))
