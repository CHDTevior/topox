"""Answer: is there DIRECT per-frame root translation/rotation in the data, or
only RIFKE velocity that must be integrated? And is root per-frame (not static)?
Pure read-only data check for PZ_Asian_Water_Monitor_Male.
"""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.anytop_dataset import AnyTopDataset, _recover_world_positions  # noqa
from src.models.treeik_decoder import rot6d_to_matrix  # noqa

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

# 1. what root-related fields exist directly in the item?
print("ROOT-RELATED item keys:", [k for k in it.keys() if "root" in k.lower()], flush=True)
for k in ["root_position", "root_velocity"]:
    v = it.get(k)
    if v is not None:
        v = np.asarray(v)
        print(f"  {k}: shape={v.shape}", flush=True)

# 2. my recovered root world pos (integrated) — is it per-frame (moving)?
ax = np.asarray(it["anytop_x"], np.float32)
mean = np.asarray(it["anytop_mean"], np.float32); std = np.asarray(it["anytop_std"], np.float32)
raw = np.transpose(ax, (2, 0, 1))[:T, :J, :] * (std[:J][None] + 1e-6) + mean[:J][None]
world03 = _recover_world_positions(raw)
rootw = world03[:, 0, :]  # [T,3]
disp_total = float(np.linalg.norm(rootw[-1] - rootw[0]))
disp_pf = float(np.linalg.norm(np.diff(rootw, axis=0), axis=-1).mean())
print(f"\nRECOVERED root world pos: total_disp={disp_total:.4f} perframe_mean={disp_pf:.4f} "
      f"(if ~0 -> static/overlapping; if >0 -> moving each frame)", flush=True)
print("  first3:", [np.round(rootw[t], 3).tolist() for t in range(min(3, T))], flush=True)
print("  last3: ", [np.round(rootw[t], 3).tolist() for t in range(max(0, T - 3), T)], flush=True)

# 3. direct root_position field vs my recovered (if it exists)
rp = it.get("root_position")
if rp is not None:
    rp = np.asarray(rp, np.float32)
    rp = rp[:T] if rp.ndim == 2 else rp
    print(f"\nDIRECT root_position shape={rp.shape}", flush=True)
    print("  first3:", [np.round(rp[t], 3).tolist() for t in range(min(3, T))], flush=True)
    if rp.shape == rootw.shape:
        print(f"  |direct - recovered| mean={np.abs(rp - rootw).mean():.4f} "
              f"max={np.abs(rp - rootw).max():.4f}  "
              f"(0 -> same; >0 -> I used a different/worse root!)", flush=True)
        dt = float(np.linalg.norm(rp[-1] - rp[0]))
        print(f"  direct root_position total_disp={dt:.4f}", flush=True)

# 4. root rotation per frame — does it change (root turning)?
R0 = rot6d_to_matrix(torch.tensor(raw[:, 0, 3:9])).numpy()  # [T,3,3]
angs = []
for t in range(T):
    Rrel = R0[t] @ R0[0].T
    angs.append(np.degrees(np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1))))
print(f"\nROOT rotation vs frame0 (deg): min={min(angs):.1f} max={max(angs):.1f} "
      f"mean={np.mean(angs):.1f}  (if ~0 -> root not rotating; if >0 -> per-frame rotation present)", flush=True)
print("  channels: root vel_x(ch9), vel_z(ch11), height(ch1) first3:", flush=True)
for t in range(min(3, T)):
    print(f"    t={t} vx={raw[t,0,9]:.4f} vz={raw[t,0,11]:.4f} h={raw[t,0,1]:.4f}", flush=True)
print("DONE", flush=True)
