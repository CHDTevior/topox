"""Gate D visual QA: render side-by-side [official HumanML3D recover_from_ric |
converted AnyTop13 _recover_world_positions] GIFs for the converted dataset.

Gate B is numerically exact, so the two panels coincide -> the GIF confirms (a)
no conversion distortion and (b) the motion is coherent human motion matching
the caption. Uses the RIC renderer (the faithful HumanML3D visualization path),
NOT the rot6d-FK route (which is the known rigid-BVH approximation for SMPL).

Usage: python _render_humanml3d_anytop13_qa.py --ids 000000 000001 ...
       python _render_humanml3d_anytop13_qa.py --auto 15
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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
from scripts.animate import animate_clip

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


def auto_ids(n):
    import csv
    rows = list(csv.DictReader(open(Path(OUT) / "object_index.csv")))
    # prefer one per action tag + short + long + a few splits
    picked, seen_tag = [], set()
    for r in rows:
        t = r["tag"]
        key = t.split(":")[0] if t else ""
        if key and key not in seen_tag:
            seen_tag.add(key)
            picked.append(r["filename"].replace(f"{OBJ}_", "").replace(".npy", ""))
    for r in rows:
        if len(picked) >= n:
            break
        mid = r["filename"].replace(f"{OBJ}_", "").replace(".npy", "")
        if mid not in picked:
            picked.append(mid)
    return picked[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--auto", type=int, default=0)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--tag", default="20260619")
    args = ap.parse_args()

    ids = args.ids if args.ids else auto_ids(args.auto or 15)
    qa = Path(OUT) / "animations" / f"conversion_qa_{args.tag}"
    qa.mkdir(parents=True, exist_ok=True)

    for mid in ids:
        raw = find_raw(mid)
        if raw is None:
            print(f"  [skip] {mid} not found")
            continue
        x = np.load(Path(SRC) / "new_joint_vecs" / f"{mid}.npy")
        official = recover_from_ric(torch.from_numpy(x).float(), 22).numpy()
        converted = _recover_world_positions(raw.astype(np.float32))
        cap = caps.get(f"{OBJ}_{mid}.npy", {}).get("primary_caption", "")
        title = f"{mid} T={raw.shape[0]}  |  {cap[:60]}"
        gif = qa / f"{mid}.gif"
        # animate_clip(pp=PRED/right, gp=GT/left, ...): left=official, right=converted
        animate_clip(converted, official, parents, str(gif), title, args.stride, args.fps)
        d = float(np.abs(official - converted).max())
        print(f"  [ok] {mid}  T={raw.shape[0]}  off-vs-conv maxabs={d:.2e}  -> {gif.name}")
    print(f"\nGIFs in: {qa}")


if __name__ == "__main__":
    main()
