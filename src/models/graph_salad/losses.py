"""Loss orchestration for Graph-SALAD VAE training (M1.2 step 5).

Pure functions (no nn.Module — caller composes them per pool_type and target
format). Designed to be ckpt-state-free (no learnable buffers).

Coverage per plan §13:
  - Reconstruction:
      * masked_l1_pos       — joint position L1 (per plan §13.1)
      * masked_l1_vel       — joint velocity L1 (per plan §13.1)
      * masked_vel_consistency — numerical-vel vs predicted-vel L1
  - Graph / morphology:
      * masked_bone_length   — | ||p_u - p_v|| - rest_bone | per edge (plan §13.2)
  - VAE bottleneck:
      * masked_kl_gaussian   — KL(q(z|x,G) || N(0,I)), per plan §13.1
  - Pool aux passthrough:
      * aggregate_pool_aux   — sum pool_dynamic / pool_deterministic aux_losses
                                weighted by user choice (caller passes weights)

All loss functions return scalar tensors (mean-reduced over valid mask entries).
Padded joints / padded coarse / invalid frames are excluded by mask. fp32 only.

For TreeIK path (user 2026-05-20 Q-B decision: rot → hard FK), the VAE outputs
6D rotation that goes through TopoFKDecoder to obtain joint_positions. The
position loss runs on the FK'd positions, NOT on the raw rotations directly.
A separate rotation loss can be added later if needed (rot_geodesic_loss already
exists in treeik_decoder.py). For M1.2 we provide only position/velocity/bone
losses; rotation loss is M1.3 wiring concern.
"""

from __future__ import annotations

import math

import torch


_EPS = 1e-8


def _broadcast_pos_vel_mask(joint_mask: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
    """Build [B, T, J] mask from [B, J] joint_mask + [B, T] frame_mask."""
    return joint_mask.unsqueeze(1) & frame_mask.unsqueeze(-1)


def masked_l1_pos(
    pred_pos: torch.Tensor,   # [B, T, J, 3]
    gt_pos: torch.Tensor,     # [B, T, J, 3]
    joint_mask: torch.Tensor,  # [B, J]
    frame_mask: torch.Tensor,  # [B, T]
) -> torch.Tensor:
    """L1(pred_pos, gt_pos) averaged over valid (joint, frame) pairs."""
    if pred_pos.shape != gt_pos.shape:
        raise ValueError(
            f"pred_pos {tuple(pred_pos.shape)} != gt_pos {tuple(gt_pos.shape)}"
        )
    if pred_pos.dim() != 4 or pred_pos.shape[-1] != 3:
        raise ValueError(f"pred_pos must be [B, T, J, 3], got {tuple(pred_pos.shape)}")
    if not torch.isfinite(pred_pos).all() or not torch.isfinite(gt_pos).all():
        raise ValueError("pred_pos / gt_pos contains NaN or Inf")
    mask = _broadcast_pos_vel_mask(joint_mask, frame_mask)  # [B, T, J]
    diff = (pred_pos - gt_pos).abs().sum(dim=-1)  # [B, T, J]
    diff = diff * mask.to(diff.dtype)
    valid_count = mask.sum().clamp(min=_EPS)
    return diff.sum() / valid_count


def masked_l1_vel(
    pred_vel: torch.Tensor,
    gt_vel: torch.Tensor,
    joint_mask: torch.Tensor,
    frame_mask: torch.Tensor,
) -> torch.Tensor:
    """L1(pred_vel, gt_vel) averaged over valid (joint, frame) pairs."""
    if pred_vel.shape != gt_vel.shape:
        raise ValueError(
            f"pred_vel {tuple(pred_vel.shape)} != gt_vel {tuple(gt_vel.shape)}"
        )
    if pred_vel.dim() != 4 or pred_vel.shape[-1] != 3:
        raise ValueError(f"pred_vel must be [B, T, J, 3], got {tuple(pred_vel.shape)}")
    if not torch.isfinite(pred_vel).all() or not torch.isfinite(gt_vel).all():
        raise ValueError("pred_vel / gt_vel contains NaN or Inf")
    mask = _broadcast_pos_vel_mask(joint_mask, frame_mask)
    diff = (pred_vel - gt_vel).abs().sum(dim=-1)
    diff = diff * mask.to(diff.dtype)
    valid_count = mask.sum().clamp(min=_EPS)
    return diff.sum() / valid_count


def masked_vel_consistency(
    pred_pos: torch.Tensor,    # [B, T, J, 3]
    pred_vel: torch.Tensor,    # [B, T, J, 3]
    fps: torch.Tensor,         # [B] float — sample frame rate
    joint_mask: torch.Tensor,
    frame_mask: torch.Tensor,
) -> torch.Tensor:
    """L1(numerical_velocity, pred_vel) where numerical_vel[t] = (pos[t+1] - pos[t]) * fps.

    Only valid for t < T-1 with both t and t+1 frame_mask=True.
    """
    if pred_pos.shape != pred_vel.shape:
        raise ValueError(
            f"pred_pos {tuple(pred_pos.shape)} != pred_vel {tuple(pred_vel.shape)}"
        )
    if pred_pos.dim() != 4 or pred_pos.shape[-1] != 3:
        raise ValueError(f"pred_pos must be [B, T, J, 3], got {tuple(pred_pos.shape)}")
    if not torch.isfinite(pred_pos).all() or not torch.isfinite(pred_vel).all():
        raise ValueError("pred_pos / pred_vel contains NaN or Inf")
    if fps.shape != (pred_pos.shape[0],):
        raise ValueError(f"fps shape must be [B], got {tuple(fps.shape)}")
    if (fps <= 0).any() or not torch.isfinite(fps).all():
        raise ValueError("fps must be positive finite")
    if joint_mask.shape != (pred_pos.shape[0], pred_pos.shape[2]) or joint_mask.dtype != torch.bool:
        raise ValueError(f"joint_mask must be [B, J] bool, got {tuple(joint_mask.shape)}")
    if frame_mask.shape != (pred_pos.shape[0], pred_pos.shape[1]) or frame_mask.dtype != torch.bool:
        raise ValueError(f"frame_mask must be [B, T] bool, got {tuple(frame_mask.shape)}")

    B, T, J, _ = pred_pos.shape
    if T < 2:
        # Can't compute numerical velocity with fewer than 2 frames
        return torch.zeros((), device=pred_pos.device, dtype=pred_pos.dtype)
    numerical_vel = (pred_pos[:, 1:] - pred_pos[:, :-1]) * fps.view(B, 1, 1, 1)  # [B, T-1, J, 3]
    pred_vel_aligned = pred_vel[:, :-1]  # [B, T-1, J, 3]
    diff = (numerical_vel - pred_vel_aligned).abs().sum(dim=-1)  # [B, T-1, J]
    # Mask: both frame_mask[t] AND frame_mask[t+1] valid; joint_mask[j] valid
    frame_mask_pair = frame_mask[:, :-1] & frame_mask[:, 1:]  # [B, T-1]
    mask = joint_mask.unsqueeze(1) & frame_mask_pair.unsqueeze(-1)  # [B, T-1, J]
    diff = diff * mask.to(diff.dtype)
    valid_count = mask.sum().clamp(min=_EPS)
    return diff.sum() / valid_count


def masked_kl_gaussian(
    mu: torch.Tensor,        # [B, T_lat, C_max, D]
    logvar: torch.Tensor,    # [B, T_lat, C_max, D]
    coarse_mask: torch.Tensor,        # [B, C_max]
    frame_mask_lat: torch.Tensor,     # [B, T_lat]
    logvar_clamp: tuple[float, float] = (-30.0, 30.0),
) -> torch.Tensor:
    """KL(N(mu, exp(logvar)) || N(0, I)) averaged over valid (latent_frame, coarse_node).

    Standard VAE KL: KL = -0.5 * sum_d (1 + logvar - mu^2 - exp(logvar)).
    Sums over the feature dim D, then mean over masked-in (B, T_lat, C_max).

    Codex M1.2 losses R1 fixes:
    - Clamp logvar to [logvar_clamp_min, logvar_clamp_max] BEFORE exp to avoid
      overflow → Inf → propagates to NaN via inf*0 on masked positions.
    - Apply mask FIRST: zero out mu/logvar at padded positions, so even if
      upstream wrote garbage at those positions, the computation stays clean.
    """
    if mu.shape != logvar.shape:
        raise ValueError(f"mu {tuple(mu.shape)} != logvar {tuple(logvar.shape)}")
    if mu.dim() != 4:
        raise ValueError(f"mu must be [B, T_lat, C_max, D], got {tuple(mu.shape)}")
    if not torch.isfinite(mu).all() or not torch.isfinite(logvar).all():
        raise ValueError("mu / logvar contains NaN or Inf")
    B, T_lat, C, D = mu.shape
    if frame_mask_lat.shape != (B, T_lat) or frame_mask_lat.dtype != torch.bool:
        raise ValueError(f"frame_mask_lat must be [B, T_lat] bool, got {tuple(frame_mask_lat.shape)}")
    if coarse_mask.shape != (B, C) or coarse_mask.dtype != torch.bool:
        raise ValueError(f"coarse_mask must be [B, C] bool, got {tuple(coarse_mask.shape)}")

    # Zero mu/logvar at padded positions BEFORE arithmetic
    mask = frame_mask_lat.unsqueeze(-1) & coarse_mask.unsqueeze(1)  # [B, T_lat, C]
    mask_4d = mask.unsqueeze(-1).to(mu.dtype)  # [B, T_lat, C, 1]
    mu_safe = mu * mask_4d
    logvar_safe = logvar * mask_4d
    # Clamp logvar to safe range before exp (codex R1 fix #1)
    logvar_clamped = logvar_safe.clamp(min=logvar_clamp[0], max=logvar_clamp[1])
    # KL per (b, t, c, d)
    kl_per_dim = -0.5 * (1 + logvar_clamped - mu_safe.pow(2) - logvar_clamped.exp())
    # Re-mask (the +1 constant contaminates padded positions if not zeroed)
    kl_per_dim = kl_per_dim * mask_4d
    kl_per_node = kl_per_dim.sum(dim=-1)  # [B, T_lat, C]
    valid_count = mask.sum().clamp(min=_EPS)
    return kl_per_node.sum() / valid_count


def masked_bone_length(
    pred_pos: torch.Tensor,        # [B, T, J, 3]
    adjacency: torch.Tensor,       # [B, J, J] binary symmetric
    rest_bone_lengths: torch.Tensor,  # [B, J] bone length from joint j to parent (parent stored in batch)
    parent_indices: list[list[int]],  # length B, each length J_valid
    joint_mask: torch.Tensor,      # [B, J]
    frame_mask: torch.Tensor,      # [B, T]
) -> torch.Tensor:
    """Bone length error: for each valid edge (j, parent[j]), penalize
    ||pred_pos[t, j] - pred_pos[t, parent[j]]|| - rest_bone_lengths[j].

    Edges are derived from parent_indices (root has -1, skipped). Bone length
    is stored at the child joint per skeleton_graph convention (`bone_lengths[j]
    = ||rest_offset[j]||`).
    """
    if pred_pos.dim() != 4 or pred_pos.shape[-1] != 3:
        raise ValueError(f"pred_pos must be [B, T, J, 3], got {tuple(pred_pos.shape)}")
    if not torch.isfinite(pred_pos).all():
        raise ValueError("pred_pos contains NaN or Inf")
    B, T, J, _ = pred_pos.shape
    if rest_bone_lengths.shape != (B, J):
        raise ValueError(
            f"rest_bone_lengths must be [B, J], got {tuple(rest_bone_lengths.shape)}"
        )
    if not isinstance(parent_indices, list) or len(parent_indices) != B:
        raise ValueError(f"parent_indices must be list of length B={B}")

    total_loss = torch.zeros((), device=pred_pos.device, dtype=pred_pos.dtype)
    total_count = 0
    for b in range(B):
        pi_b = parent_indices[b]
        j_valid_count = int(joint_mask[b].sum().item())
        # For each non-root child j with parent p:
        # Codex R1 fix #2: skip if child j or parent p is padded (joint_mask=False).
        for j, p in enumerate(pi_b):
            if p < 0:
                continue
            # Both j and p must be valid joints
            if j >= j_valid_count or p >= j_valid_count:
                continue
            if not (joint_mask[b, j] and joint_mask[b, p]):
                continue
            # Predicted bone length at each frame
            diff_vec = pred_pos[b, :, j, :] - pred_pos[b, :, p, :]  # [T, 3]
            pred_len = diff_vec.norm(dim=-1)  # [T]
            rest_len = rest_bone_lengths[b, j]  # scalar
            err = (pred_len - rest_len).abs()  # [T]
            # Mask by frame_mask[b]
            err = err * frame_mask[b].to(err.dtype)
            total_loss = total_loss + err.sum()
            total_count += int(frame_mask[b].sum().item())
    if total_count == 0:
        return total_loss  # zero
    return total_loss / total_count


def aggregate_pool_aux(
    pool_aux: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Sum pool aux_losses weighted by user-provided weights.

    Default weights (when `weights=None`):
        mincut: 1.0   (already scaled by mincut_lambda inside pool)
        locality: 1.0
        entropy: 0.0  (metric, not regularized by default)

    Only keys present in BOTH pool_aux and weights contribute. Missing keys
    in weights → use default. Missing keys in pool_aux → skipped silently
    (e.g., DeterministicGraphPool emits zeros for mincut_* but they're 0
    anyway).
    """
    default_weights = {"mincut": 1.0, "locality": 1.0, "entropy": 0.0,
                      "mincut_cut": 0.0, "mincut_ortho": 0.0}
    if weights is None:
        weights = default_weights
    else:
        merged = dict(default_weights)
        merged.update(weights)
        weights = merged

    total = None
    for key, w in weights.items():
        if key not in pool_aux:
            continue
        if w == 0.0:
            continue
        v = pool_aux[key]
        if not torch.isfinite(v).all():
            raise ValueError(f"pool_aux[{key}] contains NaN or Inf")
        contribution = v * w
        total = contribution if total is None else total + contribution
    if total is None:
        # All weights zero or no overlap — return zero scalar on default device
        device = next(iter(pool_aux.values())).device
        return torch.zeros((), device=device, dtype=torch.float32)
    return total


def compute_total_loss(
    *,
    pred_pos: torch.Tensor,
    gt_pos: torch.Tensor,
    pred_vel: torch.Tensor,
    gt_vel: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    pool_aux_outputs: list[dict[str, torch.Tensor]] | None,  # one dict per pool level
    joint_mask: torch.Tensor,
    frame_mask: torch.Tensor,
    coarse_mask: torch.Tensor,        # for KL
    frame_mask_lat: torch.Tensor,     # for KL
    rest_bone_lengths: torch.Tensor,  # [B, J] static bone-length at each child
    parent_indices: list[list[int]],
    fps: torch.Tensor,
    weights: dict[str, float] | None = None,
    pool_aux_weights: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """End-to-end VAE recon loss aggregation.

    Returns dict of named scalar losses plus a `total` summary key for
    backward(). Caller sets `weights` to balance reconstruction vs KL vs aux.
    Default weights are conservative starting points; tune per dataset.

    Default weights:
        pos: 1.0
        vel: 1.0
        vel_consistency: 0.5
        kl: 0.001          (small KL warmup; ramp up if needed)
        bone: 1.0
        pool_aux: 0.5      (multiplier on aggregated pool aux from all levels)
    """
    default_weights = {
        "pos": 1.0,
        "vel": 1.0,
        "vel_consistency": 0.5,
        "kl": 1e-3,
        "bone": 1.0,
        "pool_aux": 0.5,
    }
    if weights is None:
        weights = default_weights
    else:
        merged = dict(default_weights)
        merged.update(weights)
        weights = merged

    losses: dict[str, torch.Tensor] = {}
    losses["pos"] = masked_l1_pos(pred_pos, gt_pos, joint_mask, frame_mask)
    losses["vel"] = masked_l1_vel(pred_vel, gt_vel, joint_mask, frame_mask)
    losses["vel_consistency"] = masked_vel_consistency(
        pred_pos, pred_vel, fps, joint_mask, frame_mask
    )
    losses["kl"] = masked_kl_gaussian(mu, logvar, coarse_mask, frame_mask_lat)
    losses["bone"] = masked_bone_length(
        pred_pos, None, rest_bone_lengths, parent_indices, joint_mask, frame_mask
    )

    # Pool aux: aggregate over levels (codex R1 fix: pool_aux_weights kwarg
    # routes per-key weights to aggregate_pool_aux; default still passes None
    # to use aggregate_pool_aux's defaults).
    if pool_aux_outputs is None or len(pool_aux_outputs) == 0:
        losses["pool_aux"] = torch.zeros(
            (), device=pred_pos.device, dtype=pred_pos.dtype
        )
    else:
        per_level = [aggregate_pool_aux(aux, pool_aux_weights) for aux in pool_aux_outputs]
        losses["pool_aux"] = sum(per_level)

    # Total
    total = torch.zeros((), device=pred_pos.device, dtype=pred_pos.dtype)
    for key, w in weights.items():
        if key not in losses or w == 0.0:
            continue
        total = total + losses[key] * w
    losses["total"] = total
    return losses
