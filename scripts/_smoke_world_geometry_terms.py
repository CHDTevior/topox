"""Smoke (Step2): compute_world_geometry_terms — execution-only verification.

Verifies the new loss term via PYTHON EXECUTION (clean stdout channel), not by
reading source. Checks:
  1. world & traj terms finite on a real batch.
  2. pred==gt  -> world==0 and traj==0 (exact identity sanity).
  3. pred!=gt  -> world>0 and traj>0 (term actually responds to error).
  4. autograd flows back to pred (grad finite + nonzero).
  5. denorm round-trip: recover_world(denorm(anytop_x)) ~= motion_features[...,:3]
     (the dataset's own world GT) — proves the whole geometry path is correct.

Run: python scripts/_smoke_world_geometry_terms.py
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.anytop_dataset import AnyTopDataset, collate_fn  # noqa: E402
from src.models.graph_salad.batch import GraphMotionBatch  # noqa: E402
from src.models.graph_salad.losses import (  # noqa: E402
    compute_world_geometry_terms,
)
from src.models.graph_salad.world_recovery import (  # noqa: E402
    recover_world_positions_torch,
)

ANYTOP_ROOT = str(ROOT / "data" / "anytop_planet_zoo_clean_L2")


def main():
    ds = AnyTopDataset(
        split="val", val_frac=0.05, seed=42, data_root=ANYTOP_ROOT,
        num_frames=64, max_joints=144, caption_emb_cache=None,
    )
    items = [ds[i] for i in range(4)]
    d = collate_fn(items)
    batch = GraphMotionBatch.from_collate_dict(d)

    gt_motion = batch.anytop_x.permute(0, 3, 1, 2).contiguous()  # [B,T,J,13]
    B, T, J, _ = gt_motion.shape
    print(f"batch: B={B} T={T} J={J}", flush=True)

    # ---- check 5 FIRST: denorm round-trip vs dataset's own world GT ----
    # motion_features[...,:3] is dataset's _recover_world_positions(raw) (world GT).
    from src.models.graph_salad.losses import _denorm_13ch
    gt_raw = _denorm_13ch(gt_motion, batch.anytop_mean, batch.anytop_std)
    world_from_terms = recover_world_positions_torch(gt_raw)        # [B,T,J,3]
    world_dataset = batch.motion_features[..., :3]                  # [B,T,J,3]
    mask = (batch.joint_mask.unsqueeze(1) & batch.frame_mask.unsqueeze(-1))
    m3 = mask.unsqueeze(-1).float()
    rt_diff = ((world_from_terms - world_dataset).abs() * m3).sum() / m3.sum().clamp(min=1e-8)
    print(f"check5 denorm->recover vs dataset world GT: mean|diff|={rt_diff.item():.3e}", flush=True)

    # ---- check 2: pred==gt -> zero ----
    pred_eq = gt_motion.clone()
    terms_eq = compute_world_geometry_terms(
        pred_motion=pred_eq, gt_motion=gt_motion,
        anytop_mean=batch.anytop_mean, anytop_std=batch.anytop_std,
        joint_mask=batch.joint_mask, frame_mask=batch.frame_mask,
    )
    print(f"check2 pred==gt: world={terms_eq['world'].item():.3e} "
          f"traj={terms_eq['traj'].item():.3e} (expect ~0)", flush=True)

    # ---- check 1+3+4: pred!=gt (perturb rotation channels) ----
    pred = gt_motion.clone().detach().requires_grad_(True)
    pred_perturbed = pred + 0.0  # keep graph; perturb via a separate tensor
    noise = torch.zeros_like(gt_motion)
    noise[..., 3:9] = 0.1  # perturb 6D rotation channels
    terms = compute_world_geometry_terms(
        pred_motion=pred + noise, gt_motion=gt_motion,
        anytop_mean=batch.anytop_mean, anytop_std=batch.anytop_std,
        joint_mask=batch.joint_mask, frame_mask=batch.frame_mask,
    )
    w = terms["world"]
    tr = terms["traj"]
    print(f"check1/3 pred!=gt: world={w.item():.4f} traj={tr.item():.4f} "
          f"finite={bool(torch.isfinite(w).item() and torch.isfinite(tr).item())}",
          flush=True)
    (w + tr).backward()
    g = pred.grad
    grad_finite = bool(torch.isfinite(g).all().item())
    grad_sum = float(g.abs().sum().item())
    print(f"check4 autograd: grad_finite={grad_finite} grad_abs_sum={grad_sum:.3e}",
          flush=True)

    ok = (
        rt_diff.item() < 1e-3
        and terms_eq["world"].item() < 1e-6 and terms_eq["traj"].item() < 1e-6
        and w.item() > 0 and tr.item() > 0
        and grad_finite and grad_sum > 0
    )
    if ok:
        print("STEP2_GATE PASS", flush=True)
    else:
        print(f"STEP2_GATE FAIL (rt={rt_diff.item():.3e} "
              f"eq_world={terms_eq['world'].item():.3e} "
              f"w={w.item():.3e} tr={tr.item():.3e} "
              f"grad_finite={grad_finite} grad_sum={grad_sum:.3e})", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
