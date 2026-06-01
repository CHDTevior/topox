"""Deterministically reproduce the 3 review findings before acting on them.

P1: world+traj give ZERO gradient to NON-root rotation channels (3:9 of joints>=1)
    -> confirms it is a world-geometry loss, NOT PRISM FK (rotation-chain) loss.
    Sanity: root rotation (3:9 of joint 0) AND non-root position (0:3) DO get grad.
P2: passing {"world":..,"traj":..} into compute_total_loss_13ch's `weights` alone
    does NOT add them to total (its loop skips keys not in its own losses dict).

Run: python scripts/_verify_review_findings.py
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.anytop_dataset import AnyTopDataset, collate_fn  # noqa: E402
from src.models.graph_salad.batch import GraphMotionBatch  # noqa: E402
from src.models.graph_salad.losses import (  # noqa: E402
    compute_world_geometry_terms, compute_total_loss_13ch,
)

ANYTOP_ROOT = str(ROOT / "data" / "anytop_planet_zoo_clean_L2")


def main():
    ds = AnyTopDataset(split="val", val_frac=0.05, seed=42, data_root=ANYTOP_ROOT,
                       num_frames=64, max_joints=144, caption_emb_cache=None)
    d = collate_fn([ds[i] for i in range(4)])
    batch = GraphMotionBatch.from_collate_dict(d)
    gt = batch.anytop_x.permute(0, 3, 1, 2).contiguous()  # [B,T,J,13]

    # ---------- P1: which channels get gradient from world+traj? ----------
    pred = gt.clone().detach().requires_grad_(True)
    noise = torch.zeros_like(gt)
    noise[..., :] = 0.05  # perturb ALL channels so grad can flow to any used one
    terms = compute_world_geometry_terms(
        pred_motion=pred + noise, gt_motion=gt,
        anytop_mean=batch.anytop_mean, anytop_std=batch.anytop_std,
        joint_mask=batch.joint_mask, frame_mask=batch.frame_mask,
    )
    (terms["world"] + terms["traj"]).backward()
    g = pred.grad  # [B,T,J,13]

    # joint 0 = root ; joints 1: = non-root
    root_rot_grad = g[:, :, 0, 3:9].abs().sum().item()
    nonroot_rot_grad = g[:, :, 1:, 3:9].abs().sum().item()
    nonroot_pos_grad = g[:, :, 1:, 0:3].abs().sum().item()
    nonroot_vel_grad = g[:, :, 1:, 9:12].abs().sum().item()
    root_h_grad = g[:, :, 0, 1].abs().sum().item()
    root_vel_grad = (g[:, :, 0, 9].abs().sum() + g[:, :, 0, 11].abs().sum()).item()

    print("=== P1: gradient reaching each channel group (world+traj) ===", flush=True)
    print(f"  root    rot6d (j0, 3:9)   grad_abs_sum = {root_rot_grad:.4e}  (expect >0 used)", flush=True)
    print(f"  root    height (j0, ch1)  grad_abs_sum = {root_h_grad:.4e}  (expect >0 used)", flush=True)
    print(f"  root    vel (j0, ch9,11)  grad_abs_sum = {root_vel_grad:.4e}  (expect >0 used)", flush=True)
    print(f"  NONroot pos (j1:, 0:3)    grad_abs_sum = {nonroot_pos_grad:.4e}  (expect >0 used)", flush=True)
    print(f"  NONroot rot6d (j1:, 3:9)  grad_abs_sum = {nonroot_rot_grad:.4e}  (P1 claim: ==0)", flush=True)
    print(f"  NONroot vel (j1:, 9:12)   grad_abs_sum = {nonroot_vel_grad:.4e}  (expect ==0 unused)", flush=True)
    p1_confirmed = (nonroot_rot_grad == 0.0 and root_rot_grad > 0 and nonroot_pos_grad > 0)
    print(f"  >>> P1 CONFIRMED (non-root rotation gets NO geometric gradient): {p1_confirmed}", flush=True)

    # ---------- P2: does passing world/traj via weights alone work? ----------
    print("\n=== P2: weights-only injection into compute_total_loss_13ch ===", flush=True)
    losses = compute_total_loss_13ch(
        pred_motion=gt.clone(), gt_motion=gt,
        foot_contact_per_joint=batch.foot_contact_per_joint,
        mu=torch.zeros(4, 17, 128, 8), logvar=torch.zeros(4, 17, 128, 8),
        pool_aux_outputs=None,
        joint_mask=batch.joint_mask, frame_mask=batch.frame_mask,
        coarse_mask=torch.ones(4, 128, dtype=torch.bool),
        frame_mask_lat=torch.ones(4, 17, dtype=torch.bool),
        weights={"pos": 1.0, "rot": 1.0, "vel": 1.0, "contact": 0.1,
                 "kl": 1e-3, "pool_aux": 0.5, "world": 0.5, "traj": 0.25},
    )
    has_world = "world" in losses
    print(f"  'world' key present in returned losses dict: {has_world}  (P2: should be False)", flush=True)
    print(f"  returned loss keys: {sorted(losses.keys())}", flush=True)
    print(f"  >>> P2 CONFIRMED (weights-only does NOT add world/traj): {not has_world}", flush=True)

    print("\nVERIFY_DONE", flush=True)


if __name__ == "__main__":
    main()
