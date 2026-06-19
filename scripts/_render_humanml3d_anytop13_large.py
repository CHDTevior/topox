"""High-res orientation QA for converted HumanML3D->AnyTop13, using OUR AnyTop
PIL renderer (scripts/_pil_skeleton_render.py, 900x760/panel, oblique Y-up
projection + ground grid + coordinate axes + root trail).

Two panels per clip: official HumanML3D recover_from_ric (red) | converted
AnyTop13 _recover_world_positions (blue), BOTH with axes drawn, so facing /
upright / left-right can be checked against the world frame. Gate B is exact so
the panels coincide.

Usage: python _render_humanml3d_anytop13_large.py --ids 000005 000006 ...
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HM = "/iridisfs/scratch/ts1v23/workspace/motion-latent-diffusion-main"
REPO = "/iridisfs/scratch/ts1v23/workspace/noKslot_clean"
SRC = HM + "/datasets/humanml3d/HumanML3D"
OUT = REPO + "/data/humanml3d_anytop13"
sys.path.insert(0, HM)
sys.path.insert(0, REPO)

from mld.data.humanml.scripts.motion_process import recover_from_ric
from src.data.anytop_dataset import _recover_world_positions
import scripts._pil_skeleton_render as pr

OBJ = "HML3D_Human"
cond = np.load(Path(OUT) / "cond.npy", allow_pickle=True).item()[OBJ]
parents = [int(p) for p in cond["parents"]]
caps = json.loads((Path(OUT) / "motion_texts_by_file.json").read_text())


def find_raw(mid):
    for d in ("motions", "motions_heldout"):
        p = Path(OUT) / d / f"{OBJ}_{mid}.npy"
        if p.exists():
            return np.load(p)
    return None


def render(mid, qa, cell, zoom, pad, fps, max_frames):
    raw = find_raw(mid)
    if raw is None:
        print(f"  [skip] {mid}")
        return
    x = np.load(Path(SRC) / "new_joint_vecs" / f"{mid}.npy")
    official = recover_from_ric(torch.from_numpy(x).float(), 22).numpy().astype(np.float64)
    converted = _recover_world_positions(raw.astype(np.float32)).astype(np.float64)
    # feet to ground per the renderer convention
    for a in (official, converted):
        a[..., 1] -= a[..., 1].min()

    arrs = [(official, "official recover_from_ric", (200, 60, 60)),
            (converted, "converted AnyTop13 RIC", (35, 112, 180))]
    idxs = pr.sample_indices(official.shape[0], max_frames)
    # shared transform over root-centered sampled frames (matches make_t2m_large_gif)
    ps = []
    for a, *_ in arrs:
        c = a.copy(); roots = c[:, 0].copy()
        c[..., 0] -= roots[:, None, 0]; c[..., 2] -= roots[:, None, 2]
        ps += [c[k] for k in idxs]
    transform = pr.compute_transform(ps, cell, pad, zoom)
    cap = caps.get(f"{OBJ}_{mid}.npy", {}).get("primary_caption", "")
    header = f"{mid}  T={official.shape[0]}   |   {cap}"
    frames = []
    for k in idxs:
        panels = [{"positions": a, "parents": parents, "title": t,
                   "color": col, "axes": True, "static": False}
                  for (a, t, col) in arrs]
        frames.append(pr.make_row_frame(panels, k, transform, cell, 3, 4,
                                        header=header, header_h=84))
    gif = qa / f"{mid}_large.gif"
    pr.save_gif(frames, str(gif), fps)
    d = float(np.abs(official - converted).max())
    print(f"  [ok] {mid}  T={official.shape[0]}  off-vs-conv maxabs={d:.2e}  -> {gif.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*",
                    default=["000005", "000006", "008970", "001305",
                             "000000", "000001", "004822", "014611"])
    ap.add_argument("--cell", type=int, nargs=2, default=[900, 760])
    ap.add_argument("--zoom", type=float, default=1.15)
    ap.add_argument("--pad", type=float, default=0.06)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--max_frames", type=int, default=64)
    ap.add_argument("--tag", default="20260619_large")
    args = ap.parse_args()
    qa = Path(OUT) / "animations" / f"conversion_qa_{args.tag}"
    qa.mkdir(parents=True, exist_ok=True)
    for mid in args.ids:
        render(mid, qa, tuple(args.cell), args.zoom, args.pad, args.fps, args.max_frames)
    print(f"\nGIFs in: {qa}")


if __name__ == "__main__":
    main()
