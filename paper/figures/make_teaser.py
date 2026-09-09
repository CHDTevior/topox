"""Figure 1 (teaser): one library of skeletons -> one model -> motion on any of them, plus a new rig adapted from a rest pose.

  python paper/figures/make_teaser.py <spec.json> <out.pdf>

spec.json: {"rest_rigs": [{"npz": "...skeletons/PZ_x.npz", "name": "Cheetah"}, ... (8)],
            "gen": [{"dir": "<paper-style stills dir>", "view": "threequarter", "caption": "...", "name": "Plains zebra"}, ... (4)],
            "unseen": {"npz": "<world dump .world.npz>", "name": "Dragon", "note": "rest pose + 10 clips, adapter"}}
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
sys.path.insert(0, str(Path(__file__).parent))
from skeleton_art import draw_rest
from compose_figs import load_stills, union_bbox, strip, ACCENT, INK, MUTE

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"], "font.size": 8,
                     "pdf.fonttype": 42})
PAPER = "#f7f4ef"

def draw_world_frames(axes, npz, n, color=ACCENT):
    """Side-view line art of n evenly spaced frames of a generated FK motion (world dump v3: +Y up)."""
    z = np.load(npz, allow_pickle=True); X, par = z["gen_fk"], z["parents"]
    names = [str(s) for s in z["joint_names"]]
    hips = next((i for i, s in enumerate(names) if "pelvis" in s.lower() or "hips" in s.lower()), 0)
    head = next((i for i, s in enumerate(names) if "head" in s.lower()), len(names) - 1)
    hv = (X[:, head] - X[:, hips]).mean(0); hv[1] = 0; hv /= np.linalg.norm(hv) + 1e-9
    C = X.reshape(-1, 3).mean(0)
    idxs = np.linspace(0, X.shape[0] - 1, n).astype(int)
    xs_all = ((X - C) @ hv); ys_all = (X - C)[..., 1]
    lim = max(np.ptp(xs_all), np.ptp(ys_all)) * 0.55
    for ax, f in zip(axes, idxs):
        x, y = xs_all[f], ys_all[f]
        for j in range(len(par)):
            p = par[j]
            if p < 0: continue
            ax.plot([x[j], x[p]], [y[j], y[p]], color=color, lw=0.8, solid_capstyle="round")
        ax.scatter(x, y, s=0.7, color=INK, linewidths=0)
        cx, cy = x.mean(), y.mean()
        ax.set_xlim(cx - lim, cx + lim); ax.set_ylim(cy - lim, cy + lim); ax.set_aspect("equal"); ax.axis("off")

def main(spec_path, out_path):
    spec = json.load(open(spec_path))
    fig = plt.figure(figsize=(6.75, 2.55))
    outer = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[2.3, 0.95, 3.5], wspace=0.05, left=0.005, right=0.995, top=0.86, bottom=0.03)
    def title(cell, text):
        bb = cell.get_position(fig)
        fig.text(bb.x0, 0.975, text, fontsize=7.4, color=INK, ha="left", va="center", weight="bold")

    # ---- A: the library --------------------------------------------------------------------------
    rests = spec["rest_rigs"]
    gA = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=outer[0], wspace=0.06, hspace=0.30)
    for i, r in enumerate(rests[:8]):
        ax = fig.add_subplot(gA[i // 4, i % 4])
        n = draw_rest(ax, r["npz"], "side", lw=0.9, dot=1.1)
        ax.set_title(f"{r['name']} · {n} j.", fontsize=6.0, color=MUTE, pad=1.2)
    title(outer[0], "A  One library: 311 skeletons, 34–102 joints")

    # ---- B: the model ----------------------------------------------------------------------------
    axB = fig.add_subplot(outer[1]); axB.set_xlim(0, 1); axB.set_ylim(0, 1); axB.axis("off")
    def box(x, y, w, h, text, fc, ec, fs=6.6, weight="normal", color=INK):
        axB.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04", fc=fc, ec=ec, lw=0.9))
        axB.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, weight=weight, linespacing=1.15)
    box(0.04, 0.76, 0.92, 0.15, "rest pose of the rig\n(skeleton in context)", "white", MUTE, fs=5.4)
    box(0.04, 0.53, 0.92, 0.15, "caption\n“the zebra fights and flees”", "white", MUTE, fs=5.4)
    box(0.10, 0.12, 0.80, 0.24, "TopX\nflow-matching DiT", ACCENT, ACCENT, fs=6.6, weight="bold", color="white")
    axB.add_patch(FancyArrowPatch((0.5, 0.76), (0.5, 0.69), arrowstyle="-|>", mutation_scale=6, lw=0.8, color=MUTE))
    axB.add_patch(FancyArrowPatch((0.5, 0.53), (0.5, 0.375), arrowstyle="-|>", mutation_scale=6, lw=0.8, color=MUTE))
    axB.add_patch(FancyArrowPatch((0.93, 0.24), (1.08, 0.24), arrowstyle="-|>", mutation_scale=8, lw=1.1, color=ACCENT, clip_on=False))
    title(outer[1], "B  One model")

    # ---- C: motion on library rigs + an unseen rig ----------------------------------------------
    gens = spec["gen"][:4]
    gC = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[2], height_ratios=[1, 1, 0.9], hspace=0.62)
    for row in range(2):
        gRow = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gC[row], wspace=0.06)
        for col in range(2):
            g = gens[row * 2 + col] if row * 2 + col < len(gens) else None
            if g is None: continue
            ims = load_stills(g["dir"], g.get("view", "threequarter"))[:3]
            arrs = strip(ims, union_bbox(ims))
            gS = gridspec.GridSpecFromSubplotSpec(1, len(arrs), subplot_spec=gRow[col], wspace=0.02)
            for j, a in enumerate(arrs):
                ax = fig.add_subplot(gS[j]); ax.imshow(a); ax.axis("off")
                if j == 0:
                    import textwrap
                    ax.set_title("\n".join(textwrap.wrap(f"{g['name']}: “{g['caption']}”", 46)), loc="left", fontsize=5.9, color=INK, pad=2.0, linespacing=1.1)
    u = spec["unseen"]
    gU = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gC[2], wspace=0.02)
    axes = [fig.add_subplot(gU[j]) for j in range(5)]
    draw_world_frames(axes, u["npz"], 5)
    axes[0].set_title(f"unseen rig: {u['name']} — {u['note']}", loc="left", fontsize=6.3, color=ACCENT, pad=2.0)
    title(outer[2], "C  Motion from text on any of them, and on a rig it has never seen")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("wrote", out_path)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
