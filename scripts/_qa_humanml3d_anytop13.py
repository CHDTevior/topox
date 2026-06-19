"""QA gates B/C/E for the converted data/humanml3d_anytop13 dataset.

Gate B: official HumanML3D recover_from_ric(263) vs our _recover_world_positions(13)  [EXACT expected]
Gate C: our recover_from_bvh_rot_np(13) vs _recover_world_positions(13)             [WARNING — rigid-BVH approx]
Gate E: AnyTopDataset loader smoke (shapes, masks, graph fields)
"""
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
from src.data.anytop_dataset import _recover_world_positions, AnyTopDataset
from src.data.anytop_rot6d_fk import recover_from_bvh_rot_np

OBJ = "HML3D_Human"
cond = np.load(Path(OUT) / "cond.npy", allow_pickle=True).item()[OBJ]
parents = np.asarray(cond["parents"], dtype=np.int64)
offsets = np.asarray(cond["offsets"], dtype=np.float64)

mot_dir = Path(OUT) / "motions"
files = sorted(mot_dir.glob(f"{OBJ}_*.npy"))
# pick a spread of clips for Gate B/C
sample = files[:: max(1, len(files) // 20)][:20]

print(f"=== Gate B / Gate C on {len(sample)} clips ===")
print(f"{'clip':>22} {'T':>5} | {'B.mean':>9} {'B.max':>9} | {'C.mean':>8} {'C.%bbox':>8}")
bmax_all, cmean_all = 0.0, []
for f in sample:
    mid = f.stem.replace(f"{OBJ}_", "")
    raw = np.load(f)
    T = raw.shape[0]
    x = np.load(Path(SRC) / "new_joint_vecs" / f"{mid}.npy")
    ric_off = recover_from_ric(torch.from_numpy(x).float(), 22).numpy()
    wpos = _recover_world_positions(raw.astype(np.float32))
    db = np.linalg.norm(ric_off - wpos, axis=-1)
    fk = recover_from_bvh_rot_np(raw, parents, offsets)
    dc = np.linalg.norm(fk - wpos, axis=-1)
    bbox = np.linalg.norm(wpos.reshape(-1, 3).max(0) - wpos.reshape(-1, 3).min(0))
    bmax_all = max(bmax_all, db.max())
    cmean_all.append(dc.mean())
    print(f"{mid:>22} {T:>5} | {db.mean():>9.6f} {db.max():>9.6f} | {dc.mean():>8.4f} {100*dc.mean()/bbox:>7.2f}")
print(f"\nGate B: max abs error across all = {bmax_all:.2e}  (PASS if < 1e-3)")
print(f"Gate C: mean = {np.mean(cmean_all):.4f}  (WARNING — rigid-BVH approx, not a converter bug)")

print(f"\n=== Gate E: loader smoke ===")
ds = AnyTopDataset(data_root=OUT, split="train", num_frames=300, max_joints=144,
                   random_crop=False, load_captions=True)
print(f"  dataset len(train) = {len(ds)}")
it = ds[0]
ax = it["anytop_x"]
jm = it["joint_mask"]
fm = it["frame_mask"]
am, asd = it["anytop_mean"], it["anytop_std"]
print(f"  anytop_x: {tuple(ax.shape)}  (expect [144,13,300])")
print(f"  joint_mask.sum() = {int(jm.sum())}  (expect 22)")
print(f"  frame_mask.sum() = {int(fm.sum())}")
print(f"  anytop_mean: {tuple(am.shape)}  anytop_std: {tuple(asd.shape)}  (expect [144,13])")
gd = it["anytop_graph_dist"]
print(f"  anytop_graph_dist: {tuple(gd.shape)}  finite={bool(torch.isfinite(gd).all())}")
print(f"  caption: {it.get('caption','')[:70]!r}")
print(f"  anytop_x finite: {bool(torch.isfinite(ax).all())}")
print("  Gate E PASS" if tuple(ax.shape) == (144, 13, 300) and int(jm.sum()) == 22
      and tuple(am.shape) == (144, 13) else "  Gate E CHECK")
