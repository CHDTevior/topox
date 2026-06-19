"""Full-dataset QA: gt_fk_mismatch = RIC(gt) vs rot6d-FK(gt), %bbox, over ALL
29226 converted clips of the re-encoded v2 dataset. Reports mean/median/p90/p95/
p99/max and the top-20 worst clips; writes per-clip JSONL.

RIC(gt) = official recover_from_ric(source 263) (== our _recover_world_positions
on ch0:3, Gate B). FK(gt) = recover_from_bvh_rot_np(reencoded raw13, parents, cond offsets).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HM = "/iridisfs/scratch/ts1v23/workspace/motion-latent-diffusion-main"
REPO = "/iridisfs/scratch/ts1v23/workspace/noKslot_clean"
SRC = HM + "/datasets/humanml3d/HumanML3D"
V = REPO + "/data/humanml3d_anytop13_v2_shared_reencoded"
sys.path.insert(0, HM); sys.path.insert(0, REPO)
from mld.data.humanml.scripts.motion_process import recover_from_ric
from src.data.anytop_rot6d_fk import recover_from_bvh_rot_np

OBJ = "HML3D_Human"
J = 22
cond = np.load(Path(V) / "cond.npy", allow_pickle=True).item()[OBJ]
parents = np.asarray(cond["parents"], np.int64)
offsets = np.asarray(cond["offsets"], np.float64)
split_of = {}
for s in ("train", "val", "test"):
    for ln in (Path(V) / "splits" / f"{s}.txt").read_text().splitlines():
        if ln.strip():
            split_of[ln.strip()] = s

files = sorted(list((Path(V) / "motions").glob(f"{OBJ}_*.npy")) +
               list((Path(V) / "motions_heldout").glob(f"{OBJ}_*.npy")))
rows = []
for i, f in enumerate(files):
    mid = f.stem.replace(f"{OBJ}_", "")
    raw = np.load(f).astype(np.float64)
    x = np.load(Path(SRC) / "new_joint_vecs" / f"{mid}.npy")
    ric = recover_from_ric(torch.from_numpy(x).float(), J).numpy()
    fk = recover_from_bvh_rot_np(raw, parents, offsets)
    d = np.linalg.norm(fk - ric, axis=-1)
    bbox = np.linalg.norm(ric.reshape(-1, 3).max(0) - ric.reshape(-1, 3).min(0)) + 1e-8
    rows.append({"id": mid, "T": int(raw.shape[0]), "split": split_of.get(f.name, "?"),
                 "pct": float(100 * d.mean() / bbox), "maxpct": float(100 * d.max() / bbox)})
    if (i + 1) % 4000 == 0:
        print(f"  ... {i + 1}/{len(files)}", flush=True)

with open(Path(V) / "fk_mismatch.jsonl", "w") as fo:
    for r in rows:
        fo.write(json.dumps(r) + "\n")

pct = np.array([r["pct"] for r in rows])
loader = np.array([r["pct"] for r in rows if r["split"] in ("train", "val")])


def stats(a, name):
    print(f"  {name:>16} (n={len(a)}): mean={a.mean():.3f}  median={np.median(a):.3f}  "
          f"p90={np.percentile(a,90):.3f}  p95={np.percentile(a,95):.3f}  "
          f"p99={np.percentile(a,99):.3f}  max={a.max():.3f}  (%bbox)")


print(f"\n=== gt_fk_mismatch  RIC(gt) vs rot6d-FK(gt)  [%bbox, mean-per-clip] ===")
stats(pct, "ALL 29226")
stats(loader, "train+val")
print(f"\n=== top-20 worst (mean %bbox) ===")
for r in sorted(rows, key=lambda r: -r["pct"])[:20]:
    print(f"  {r['id']}  T={r['T']:>3}  split={r['split']:>5}  mean={r['pct']:.2f}%  max={r['maxpct']:.2f}%")
print(f"\nwrote {Path(V)/'fk_mismatch.jsonl'}")
