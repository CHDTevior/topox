"""Reproduce review P1 ONLY: does world+traj give gradient to non-root rotation?
Writes plain result lines to stdout (read via ssh to avoid render pollution).
"""
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.anytop_dataset import AnyTopDataset, collate_fn  # noqa
from src.models.graph_salad.batch import GraphMotionBatch  # noqa
from src.models.graph_salad.losses import compute_world_geometry_terms  # noqa

ANYTOP_ROOT = str(ROOT / "data" / "anytop_planet_zoo_clean_L2")

ds = AnyTopDataset(split="val", val_frac=0.05, seed=42, data_root=ANYTOP_ROOT,
                   num_frames=64, max_joints=144, caption_emb_cache=None)
d = collate_fn([ds[i] for i in range(4)])
b = GraphMotionBatch.from_collate_dict(d)
gt = b.anytop_x.permute(0, 3, 1, 2).contiguous()

pred = gt.clone().detach().requires_grad_(True)
noise = torch.full_like(gt, 0.05)
t = compute_world_geometry_terms(
    pred_motion=pred + noise, gt_motion=gt,
    anytop_mean=b.anytop_mean, anytop_std=b.anytop_std,
    joint_mask=b.joint_mask, frame_mask=b.frame_mask)
(t["world"] + t["traj"]).backward()
g = pred.grad

print("RESULT root_rot_j0_3:9   =", round(g[:, :, 0, 3:9].abs().sum().item(), 6))
print("RESULT nonroot_rot_3:9   =", round(g[:, :, 1:, 3:9].abs().sum().item(), 6))
print("RESULT nonroot_pos_0:3   =", round(g[:, :, 1:, 0:3].abs().sum().item(), 6))
print("RESULT nonroot_vel_9:12  =", round(g[:, :, 1:, 9:12].abs().sum().item(), 6))
print("RESULT root_h_j0_ch1     =", round(g[:, :, 0, 1].abs().sum().item(), 6))
nr_rot = g[:, :, 1:, 3:9].abs().sum().item()
nr_pos = g[:, :, 1:, 0:3].abs().sum().item()
r_rot = g[:, :, 0, 3:9].abs().sum().item()
print("RESULT P1_CONFIRMED      =", (nr_rot == 0.0 and nr_pos > 0 and r_rot > 0))
print("VERIFY_P1_DONE")
