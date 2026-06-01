"""Diagnose the 18-34% rot6d-FK vs 0:3-position gap: is it at depth-1 (root
rotation direction / rot6d convention) or does it accumulate with chain depth
(FK chain convention)? Pure data, no model. Read-only.

For Asian_Water_Monitor_Male:
  - root (j=0) diff vs world03  -> should be ~0 (shared root_local)
  - mean |world03 - fk| grouped by hop-depth-from-root
  - for depth-1 joints, compare forward Rr@rest vs inverse Rr^T@rest to see
    which root-rotation direction better matches world03.
"""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.anytop_dataset import AnyTopDataset, _recover_world_positions  # noqa
from src.models.treeik_decoder import fk_one, rot6d_to_matrix  # noqa

ds = AnyTopDataset(split="val", val_frac=0.05, seed=42,
                   data_root=str(ROOT / "data/anytop_planet_zoo_clean_L2"),
                   num_frames=64, max_joints=144, caption_emb_cache=None)
it = None
for i in range(len(ds)):
    x = ds[i]
    if x["object_type"] == "PZ_Asian_Water_Monitor_Male":
        it = x
        break
J = int(it["num_joints"]); T = int(it["num_frames"])
ax = np.asarray(it["anytop_x"], np.float32)
mean = np.asarray(it["anytop_mean"], np.float32); std = np.asarray(it["anytop_std"], np.float32)
raw = np.transpose(ax, (2, 0, 1))[:T, :J, :] * (std[:J][None] + 1e-6) + mean[:J][None]
world03 = _recover_world_positions(raw)  # [T,J,3]
lr6 = np.asarray(it["local_rotations_6d"], np.float32)[:T, :J, :]
rot6d = torch.tensor(lr6)
rest = torch.tensor(np.asarray(it["rest_offsets"], np.float32)[:J])
parents = [int(p) for p in it["parent_indices"][:J]]
root_local = torch.tensor(world03[:, 0, :])
fk = fk_one(rot6d, root_local, parents, rest).numpy()  # [T,J,3]

# hop depth
depth = [0] * J
for j in range(J):
    d = 0; p = parents[j]
    while p >= 0:
        d += 1; p = parents[p]
    depth[j] = d

dif = np.abs(world03 - fk).mean(axis=(0, 2))  # [J] per-joint mean over T,xyz
print(f"J={J} T={T} maxdepth={max(depth)}", flush=True)
print(f"root(j0) diff = {dif[0]:.4f}  (expect ~0, shared)", flush=True)
for d in sorted(set(depth)):
    js = [j for j in range(J) if depth[j] == d]
    print(f"  depth={d:2d} njoints={len(js):3d} mean|w03-fk|={np.mean([dif[j] for j in js]):.4f}", flush=True)

# depth-1 joints: forward Rr@rest vs inverse Rr^T@rest, which matches world03?
R0 = rot6d_to_matrix(rot6d[:, 0]).numpy()  # [T,3,3] root rotation
restnp = rest.numpy()
d1 = [j for j in range(J) if depth[j] == 1]
fwd_err = inv_err = 0.0
for j in d1:
    off = restnp[j]
    fwd = world03[:, 0, :] + np.einsum("tij,j->ti", R0, off)        # root + Rr@off
    inv = world03[:, 0, :] + np.einsum("tij,j->ti", np.transpose(R0, (0, 2, 1)), off)  # root + Rr^T@off
    fwd_err += np.abs(world03[:, j] - fwd).mean()
    inv_err += np.abs(world03[:, j] - inv).mean()
n = max(len(d1), 1)
print(f"depth1 joints={len(d1)}  forward Rr@rest err={fwd_err/n:.4f}  "
      f"inverse Rr^T@rest err={inv_err/n:.4f}", flush=True)
print("DONE", flush=True)
