"""Render rest/T-pose skeletons from cond.npy (offsets+parents FK) + world axes.
One-off QA viz (user 2026-06-07): see the T-pose + coordinate system for a few species.
Usage: python scripts/_render_tpose.py [Sp1 Sp2 Sp3]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COND = "data/anytop_truebones/cond.npy"
species = sys.argv[1:] or ["Horse", "Spider", "Flamingo"]
c = np.load(COND, allow_pickle=True).item()


def rest_fk(offsets, parents):
    """Rest pose = cumulative offset sum along the kinematic chain (identity rot)."""
    J = len(parents)
    pos = np.zeros((J, 3), dtype=float)
    for j in range(J):
        p = int(parents[j])
        pos[j] = offsets[j] if p < 0 else pos[p] + offsets[j]
    return pos


fig = plt.figure(figsize=(6 * len(species), 6.5))
for i, sp in enumerate(species):
    v = c[sp]
    off = np.asarray(v["offsets"], dtype=float)
    par = np.asarray(v["parents"])
    pos = rest_fk(off, par)
    ax = fig.add_subplot(1, len(species), i + 1, projection="3d")
    # bones (parent->child)
    for j in range(len(par)):
        p = int(par[j])
        if p >= 0:
            ax.plot([pos[j, 0], pos[p, 0]], [pos[j, 1], pos[p, 1]],
                    [pos[j, 2], pos[p, 2]], "-", color="0.35", lw=1.2)
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c="tab:blue", s=10, depthshade=False)
    ax.scatter([pos[0, 0]], [pos[0, 1]], [pos[0, 2]], c="orange", s=40, label="root")  # root
    # world coordinate axes at origin (R=X, G=Y, B=Z)
    rng = float(np.ptp(pos, axis=0).max())
    L = max(rng * 0.35, 1e-3)
    ax.quiver(0, 0, 0, L, 0, 0, color="r", arrow_length_ratio=0.12, lw=2)
    ax.quiver(0, 0, 0, 0, L, 0, color="g", arrow_length_ratio=0.12, lw=2)
    ax.quiver(0, 0, 0, 0, 0, L, color="b", arrow_length_ratio=0.12, lw=2)
    ax.text(L, 0, 0, "X", color="r", fontsize=11, weight="bold")
    ax.text(0, L, 0, "Y", color="g", fontsize=11, weight="bold")
    ax.text(0, 0, L, "Z", color="b", fontsize=11, weight="bold")
    # equal aspect around the skeleton (include origin)
    allpts = np.vstack([pos, np.zeros((1, 3))])
    ctr = allpts.mean(0)
    r = max(np.ptp(allpts, axis=0).max() * 0.55, 1e-3)
    ax.set_xlim(ctr[0] - r, ctr[0] + r)
    ax.set_ylim(ctr[1] - r, ctr[1] + r)
    ax.set_zlim(ctr[2] - r, ctr[2] + r)
    ax.set_box_aspect([1, 1, 1])
    ax.set_title(f"{sp}  (J={len(par)})", fontsize=12)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=12, azim=-60)

fig.suptitle("Truebones T-pose (rest FK from cond.npy offsets+parents) + world axes  R=X G=Y B=Z",
             fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.96])
import os
os.makedirs("runs/_qa_tpose", exist_ok=True)
out = "runs/_qa_tpose/truebones_tpose_" + "_".join(species) + ".png"
plt.savefig(out, dpi=115, bbox_inches="tight")
print("SAVED", out, "| species:", species,
      "| pos ranges:", {s: np.round(np.ptp(rest_fk(np.asarray(c[s]['offsets'],float), np.asarray(c[s]['parents'])),0),2).tolist() for s in species})
