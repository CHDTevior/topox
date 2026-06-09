"""Diagnostics for the Graph-VQVAE tokenizer — root-drift/jitter QA gate (F2 /
amendment 4).

The review flagged the ROOT as the highest-risk token under quantization: the
world-recovery integrates the root velocity by cumsum (world_recovery.py:78), so
quantization noise on the root token accumulates into TRAJECTORY DRIFT, and the
joint-averaged `traj` loss term DILUTES root supervision (it is one joint among
J). Per fork F2 we keep v1.0 coarse-slot-only (no separate global/root branch)
but log root drift + root jitter EXPLICITLY during smoke/training so a root
failure is caught BEFORE committing to the 300-epoch run.

All metrics are computed from the SAME differentiable world recovery the loss /
visual QA use (recover_world_positions_torch), on DE-NORMALIZED motion, in fp32.
These are torch.no_grad diagnostics — they do not affect the loss.
"""

from __future__ import annotations

import torch

from src.models.graph_salad.losses import _denorm_13ch
from src.models.graph_salad.world_recovery import recover_world_positions_torch


_EPS = 1e-8


@torch.no_grad()
def root_drift_jitter_qa(
    pred_motion: torch.Tensor,   # [B, T, J, 13] normalized decoder output
    gt_motion: torch.Tensor,     # [B, T, J, 13] normalized anytop_x
    anytop_mean: torch.Tensor,   # [B, J, 13]
    anytop_std: torch.Tensor,    # [B, J, 13]
    frame_mask: torch.Tensor,    # [B, T] bool (use frame_mask_recovered)
) -> dict[str, float]:
    """Root-only trajectory drift + jitter diagnostics.

    Returns:
      root_drift_mean : mean over valid frames of ||root_pred - root_gt||_2
                        (world space) — catches cumsum-amplified trajectory drift.
      root_drift_max  : max over valid frames of the same (worst-frame drift).
      root_jitter_pred: mean ||Δ²root_pred||_2 over valid interior frames
                        (2nd time-difference = acceleration magnitude). High = the
                        quantized root trajectory is shaky / high-frequency.
      root_jitter_gt  : same for GT (the natural jitter floor for this batch).
      root_jitter_ratio: root_jitter_pred / max(root_jitter_gt, eps). ~1 means
                        pred jitter matches GT; >>1 means quantization injected
                        jitter the GT does not have.
    """
    pred_raw = _denorm_13ch(pred_motion, anytop_mean, anytop_std)
    gt_raw = _denorm_13ch(gt_motion, anytop_mean, anytop_std)
    world_pred = recover_world_positions_torch(pred_raw)  # [B,T,J,3]
    world_gt = recover_world_positions_torch(gt_raw)

    root_pred = world_pred[:, :, 0, :]  # [B, T, 3]
    root_gt = world_gt[:, :, 0, :]      # [B, T, 3]
    fm = frame_mask.to(root_pred.dtype)  # [B, T]

    # ---- drift: per-frame L2 of root position error ----
    drift = (root_pred - root_gt).pow(2).sum(dim=-1).clamp(min=0).sqrt()  # [B,T]
    drift_masked = drift * fm
    n_frames = fm.sum().clamp(min=_EPS)
    drift_mean = (drift_masked.sum() / n_frames).item()
    drift_max = drift_masked.max().item() if fm.any() else 0.0

    # ---- jitter: L2 of 2nd time-difference (acceleration) at interior frames ----
    # accel[t] = root[t+1] - 2 root[t] + root[t-1]; valid where t-1,t,t+1 all valid.
    def _jitter(root: torch.Tensor) -> float:
        if root.shape[1] < 3:
            return 0.0
        accel = root[:, 2:, :] - 2.0 * root[:, 1:-1, :] + root[:, :-2, :]  # [B,T-2,3]
        amag = accel.pow(2).sum(dim=-1).clamp(min=0).sqrt()                # [B,T-2]
        # interior validity: frame t-1, t, t+1 all real
        fm_b = frame_mask.bool()
        interior = fm_b[:, 2:] & fm_b[:, 1:-1] & fm_b[:, :-2]              # [B,T-2]
        im = interior.to(amag.dtype)
        denom = im.sum().clamp(min=_EPS)
        return (amag * im).sum().div(denom).item()

    jitter_pred = _jitter(root_pred)
    jitter_gt = _jitter(root_gt)
    jitter_ratio = jitter_pred / max(jitter_gt, _EPS)

    return {
        "root_drift_mean": drift_mean,
        "root_drift_max": drift_max,
        "root_jitter_pred": jitter_pred,
        "root_jitter_gt": jitter_gt,
        "root_jitter_ratio": jitter_ratio,
    }
