#!/usr/bin/env python3
"""Visualization-STYLE comparison (user 2026-06-15): render the SAME motion in
(a) our PIL renderer (animate_denoiser.make_t2m_large_gif) and (b) a faithful
replica of AnyTop's DEFAULT matplotlib stick-figure style
(data_loaders/.../plot_script.py:130 get_general_skeleton_3d_motion): mpl_toolkits
mplot3d, view_init(elev=120, azim=-90), ax.dist=7.5, xz ground plane that follows
the root trajectory, per-frame root-xz centering, dataset scale (truebones *=1.3),
red bone lines. AnyTop writes MP4 via moviepy; moviepy/ffmpeg are unavailable here,
so we write a GIF via matplotlib PillowWriter — the DRAWING is identical, only the
container differs. READ-ONLY QA, no model: GT positions come from
motion_features[..., :3], which the dataset already builds as
world_pos = _recover_world_positions(raw_13ch) (anytop_dataset.py:1016), i.e. the
exact AnyTop recover_from_bvh_ric_np position route.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.animate import fk_rest_pose  # noqa: E402
from scripts.animate_denoiser import make_t2m_large_gif  # noqa: E402
from src.data.anytop_dataset import AnyTopDataset  # noqa: E402


def anytop_style_gif(joints, parents, out_path, title="", dataset="truebones",
                     figsize=(7, 7), radius=5, fps=8):
    """Faithful replica of AnyTop get_general_skeleton_3d_motion (plot_script.py:130),
    written as GIF (PillowWriter) instead of moviepy VideoClip/MP4. `joints` = [T,J,3]
    world positions; `parents` = parent index per joint (root parent ignored)."""
    data = np.asarray(joints, dtype=np.float64).reshape(len(joints), -1, 3).copy()
    # dataset-specific visualization scale (verbatim from AnyTop plot_script.py:158-165)
    if dataset in ("humanml", "truebones", "humanml_mat"):
        data *= 1.3
    elif dataset == "kit":
        data *= 0.003
    elif dataset in ("humanact12", "uestc"):
        data *= -1.5

    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    n_frames = data.shape[0]
    height_offset = MINS[1]
    data[:, :, 1] -= height_offset            # drop skeleton onto the ground plane
    trajec = data[:, 0, [0, 2]]               # root xz trajectory (for ground follow)
    data[..., 0] -= data[:, 0:1, 0]           # per-frame root-xz centering: animal
    data[..., 2] -= data[:, 0:1, 2]           # pinned to view center, ground moves

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")  # AnyTop uses p3.Axes3D(fig); equivalent

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        verts = [[minx, miny, minz], [minx, miny, maxz],
                 [maxx, miny, maxz], [maxx, miny, minz]]
        pl = Poly3DCollection([verts])
        pl.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(pl)

    def update(index):
        # Mirror AnyTop update() EXACTLY (plot_script.py:184): ax.clear() then view +
        # ground + bones, with NO per-frame axis-limit reset — AnyTop lets the cleared
        # 3D axes autoscale to the (ground-plane + bones) extent, so framing must match.
        ax.clear()
        ax.view_init(elev=120, azim=-90)
        try:
            ax.dist = 7.5  # removed in newer mpl; harmless if it no-ops
        except Exception:
            pass
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0,
                     MINS[2] - trajec[index, 1], MAXS[2] - trajec[index, 1])
        for joint, parent in enumerate(parents[1:], start=1):
            ax.plot3D(data[index, [joint, parent], 0],
                      data[index, [joint, parent], 1],
                      data[index, [joint, parent], 2],
                      color="red", solid_capstyle="round")
        ax.set_axis_off()
        if title:
            fig.suptitle(title, fontsize=10)
        return []

    ani = FuncAnimation(fig, update, frames=n_frames,
                        interval=1000.0 / max(fps, 1), blit=False)
    ani.save(str(out_path), writer=PillowWriter(fps=fps))
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anytop_root", required=True)
    ap.add_argument("--clip_names", required=True, help="comma-sep clip basenames")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_joints", type=int, default=144)
    ap.add_argument("--num_frames", type=int, default=300)
    ap.add_argument("--dataset_scale", default="truebones",
                    help="AnyTop dataset key controlling the *=1.3 (truebones/humanml) "
                         "visualization scale")
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    want = set(c.strip() for c in args.clip_names.split(",") if c.strip())

    # READ-ONLY guard (codex 019ecdf3): AnyTopDataset rebuilds+WRITES
    # _cond_normalized_J{max_joints}.pkl into --anytop_root unless a fresh cache exists
    # (anytop_dataset.py:554-591, write condition = NOT (cache exists AND newer than
    # cond.npy)). Refuse to run if that write would trigger, so this QA script never
    # mutates a dataset root. Pre-build the cache once via a normal dataset load if needed.
    _root = Path(args.anytop_root)
    _cache = _root / f"_cond_normalized_J{args.max_joints}.pkl"
    _cond = _root / "cond.npy"
    if not (_cache.exists() and _cond.exists()
            and _cache.stat().st_mtime > _cond.stat().st_mtime):
        raise SystemExit(
            f"[read-only guard] {_cache.name} missing/stale vs cond.npy in {_root}; "
            f"AnyTopDataset would build+WRITE it. Pre-build the cache once (normal "
            f"dataset load at max_joints={args.max_joints}) before running this QA script.")

    ds = AnyTopDataset(split="all", num_frames=args.num_frames,
                       max_joints=args.max_joints, data_root=args.anytop_root,
                       load_captions=False)
    done: set[str] = set()
    for i in range(len(ds)):
        bn = Path(ds.samples[i]["path"]).name
        if bn not in want or bn in done:
            continue
        item = ds[i]
        J = int(item["num_joints"])
        T = min(args.num_frames, int(item["num_frames"]))
        # GT world positions (== AnyTop recover_from_bvh_ric_np route, precomputed).
        pos = item["motion_features"][:T, :J, :3].float().cpu().numpy()
        parents = [int(p) for p in item["parent_indices"][:J]]
        offsets = np.asarray(item["rest_offsets"])[:J]
        static_pose = fk_rest_pose(offsets, parents)
        label = bn[:-4] if bn.endswith(".npy") else bn

        anytop_style_gif(pos, parents, out / f"{label}__ANYTOP_style.gif",
                         title=label[:30], dataset=args.dataset_scale, fps=args.fps)
        # Our PIL style: feed the SAME GT positions; gt=pos draws the red GT panel too.
        # max_frames=T so ours renders the full clip (AnyTop renders all frames too) —
        # avoids a frame-count confound in the style comparison.
        make_t2m_large_gif(pos, pos, static_pose, parents, "GT (style demo)",
                           str(out / f"{label}__OURS_style.gif"), max_frames=T,
                           fps=args.fps, gt=pos,
                           pred_labels=("our-style (oblique PIL)", "(same motion)"))
        done.add(bn)
        print(f"rendered {label}: J={J} T={T}")
        if done >= want:
            break
    missing = sorted(want - done)
    print(f"DONE {len(done)} clips x 2 styles -> {out}")
    if missing:
        print("MISSING:", missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
