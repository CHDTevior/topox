"""Smoke-verify the FK double-rotation FIX (root correction removed) on BOTH the
numpy and the torch (training-loss) FK paths, against the RIC ground truth.

PASS criteria (Saiga max-mismatch clip):
  numpy FK(fixed) vs RIC          absL1 ~ 0          (was 0.6522 with bug)
  torch FK(fixed) vs torch RIC    absL1 ~ 0
  torch FK(fixed) vs numpy RIC    absL1 ~ 0
  torch FK(fixed) vs numpy FK     absL1 ~ 0          (impl parity, was 1.19e-6)
  autograd backward through torch FK: grad finite    (loss still differentiable)
"""
import sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.anytop_dataset import AnyTopDataset, _recover_world_positions, _STD_FLOOR  # noqa
from src.data.anytop_rot6d_fk import recover_from_bvh_rot_np  # noqa
from src.models.graph_salad.rot6d_fk_recovery import recover_rot6d_fk_positions_torch  # noqa
from src.models.graph_salad.world_recovery import recover_world_positions_torch  # noqa

ds = AnyTopDataset(split="train", val_frac=0.05, seed=42,
                   data_root=str(ROOT / "data/anytop_planet_zoo_clean_L2"),
                   num_frames=64, max_joints=144, caption_emb_cache=None)
sidx = [i for i, s in enumerate(ds.samples) if "Saiga" in s["object_type"]][:8]


def getraw(it):
    J = int(it["num_joints"]); T = int(it["num_frames"])
    ax = np.asarray(it["anytop_x"], np.float32)
    mean = np.asarray(it["anytop_mean"], np.float32); std = np.asarray(it["anytop_std"], np.float32)
    raw = np.transpose(ax, (2, 0, 1))[:T, :J, :] * (std[:J][None] + _STD_FLOOR) + mean[:J][None]
    parents = np.asarray([int(p) for p in it["parent_indices"][:J]], int)
    offsets = np.asarray(it["rest_offsets"], np.float32)[:J]
    return raw, parents, offsets, T, J


best = None
for i in sidx:
    raw, parents, offsets, T, J = getraw(ds[i])
    ric = _recover_world_positions(raw.astype(np.float32))
    fk = recover_from_bvh_rot_np(raw.astype(np.float64), parents, offsets)
    l1 = float(np.abs(fk - ric).sum(-1).mean())
    if best is None or l1 > best[0]:
        best = (l1, raw, parents, offsets, T, J)
l1, raw, parents, offsets, T, J = best
print("Saiga max clip: T=%d J=%d" % (T, J), flush=True)

ric = _recover_world_positions(raw.astype(np.float32))
fk_np = recover_from_bvh_rot_np(raw.astype(np.float64), parents, offsets)
print("[1] numpy FK(fixed) vs numpy RIC   absL1=%.7f" % float(np.abs(fk_np - ric).sum(-1).mean()), flush=True)

mt = torch.from_numpy(raw[None].astype(np.float32))     # [1,T,J,13]
off_t = torch.from_numpy(offsets[None].astype(np.float32))
jm = torch.ones(1, J, dtype=torch.bool)
fk_t = recover_rot6d_fk_positions_torch(mt, [list(parents)], off_t, jm)[0].numpy()
ric_t = recover_world_positions_torch(mt)[0].numpy()
print("[2] torch FK(fixed) vs torch RIC   absL1=%.7f" % float(np.abs(fk_t - ric_t).sum(-1).mean()), flush=True)
print("[3] torch FK(fixed) vs numpy RIC   absL1=%.7f" % float(np.abs(fk_t - ric).sum(-1).mean()), flush=True)
print("[4] torch FK(fixed) vs numpy FK    absL1=%.7f  (impl parity)" % float(np.abs(fk_t - fk_np).sum(-1).mean()), flush=True)

mt2 = torch.from_numpy(raw[None].astype(np.float32)).requires_grad_(True)
fk2 = recover_rot6d_fk_positions_torch(mt2, [list(parents)], off_t, jm)
loss = fk2.sum(); loss.backward()
gfin = bool(torch.isfinite(mt2.grad).all()); gn = float(mt2.grad.norm())
print("[5] autograd backward: grad_finite=%s grad_norm=%.4f" % (gfin, gn), flush=True)

ok = (np.abs(fk_np - ric).sum(-1).mean() < 1e-3 and np.abs(fk_t - ric).sum(-1).mean() < 1e-3
      and np.abs(fk_t - fk_np).sum(-1).mean() < 1e-3 and gfin)
print("\nRESULT: %s" % ("PASS (FK==RIC, fix correct, differentiable)" if ok else "FAIL"), flush=True)
