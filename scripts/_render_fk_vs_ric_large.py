"""High-res RIC-route | rot6d-FK-route side-by-side GIFs, to eyeball pose/FK
separation on the worst gt_fk_mismatch clips of the re-encoded v2 dataset.

Left  = RIC route (_recover_world_positions, ch0:3) — the faithful position path.
Right = FK route  (recover_from_bvh_rot_np, ch3:9 + offsets) — the re-encoded rot6d.
Both with axes. If they overlap, FK==RIC; visible separation == the mismatch.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = "/iridisfs/scratch/ts1v23/workspace/noKslot_clean"
V = REPO + "/data/humanml3d_anytop13_v2_shared_reencoded"
sys.path.insert(0, REPO)
from src.data.anytop_dataset import _recover_world_positions
from src.data.anytop_rot6d_fk import recover_from_bvh_rot_np
import scripts._pil_skeleton_render as pr

OBJ = "HML3D_Human"
cond = np.load(Path(V) / "cond.npy", allow_pickle=True).item()[OBJ]
parents_list = [int(p) for p in cond["parents"]]
parents = np.asarray(cond["parents"], np.int64)
offsets = np.asarray(cond["offsets"], np.float64)
caps = json.loads((Path(V) / "motion_texts_by_file.json").read_text())


def find_raw(mid):
    for d in ("motions", "motions_heldout"):
        p = Path(V) / d / f"{OBJ}_{mid}.npy"
        if p.exists():
            return np.load(p)
    return None


def render(mid, qa, cell=(900, 760), zoom=1.15, pad=0.06, fps=12, max_frames=64):
    raw = find_raw(mid)
    if raw is None:
        print(f"  [skip] {mid}"); return
    ric = _recover_world_positions(raw.astype(np.float32)).astype(np.float64)
    fk = recover_from_bvh_rot_np(raw.astype(np.float64), parents, offsets).astype(np.float64)
    for a in (ric, fk):
        a[..., 1] -= a[..., 1].min()
    d = np.linalg.norm(ric - fk, axis=-1)
    bbox = np.linalg.norm(ric.reshape(-1, 3).max(0) - ric.reshape(-1, 3).min(0)) + 1e-8
    arrs = [(ric, "RIC route (faithful)", (35, 112, 180)),
            (fk, "rot6d-FK route", (30, 150, 55))]
    idxs = pr.sample_indices(ric.shape[0], max_frames)
    ps = []
    for a, *_ in arrs:
        c = a.copy(); r = c[:, 0].copy()
        c[..., 0] -= r[:, None, 0]; c[..., 2] -= r[:, None, 2]
        ps += [c[k] for k in idxs]
    transform = pr.compute_transform(ps, cell, pad, zoom)
    cap = caps.get(f"{OBJ}_{mid}.npy", {}).get("primary_caption", "")
    header = f"{mid} T={ric.shape[0]}  FK-vs-RIC mean={100*d.mean()/bbox:.2f}% max={100*d.max()/bbox:.2f}%  |  {cap}"
    frames = []
    for k in idxs:
        panels = [{"positions": a, "parents": parents_list, "title": t, "color": col,
                   "axes": True, "static": False} for (a, t, col) in arrs]
        frames.append(pr.make_row_frame(panels, k, transform, cell, 3, 4, header=header, header_h=84))
    gif = qa / f"{mid}_fkvsric.gif"
    pr.save_gif(frames, str(gif), fps)
    print(f"  [ok] {mid} T={ric.shape[0]} mean={100*d.mean()/bbox:.2f}% max={100*d.max()/bbox:.2f}% -> {gif.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--tag", default="fk_mismatch_top")
    args = ap.parse_args()
    qa = Path(V) / "animations" / f"qa_{args.tag}"
    qa.mkdir(parents=True, exist_ok=True)
    for mid in args.ids:
        render(mid, qa)
    print(f"\nGIFs in: {qa}")


if __name__ == "__main__":
    main()
