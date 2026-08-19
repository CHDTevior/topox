"""Differentiable torch port of the OFFICIAL rot-path FK (src/data/anytop_rot6d_fk.py), for the
gamma_7-style FK<->RIC consistency loss.

FIDELITY CONTRACT
    fk_positions_torch(raw, parents, offsets) must match recover_from_bvh_rot_np to ~1e-4 on real
    clips (checked by scratch/_test_gamma7_gtzero.py before the term may ever be enabled). The
    port keeps two non-obvious properties of the numpy original VERBATIM:
      - the parent-slot reindex writes each child's rotation channels onto its PARENT's slot,
        LAST child wins (idempotent on real data where siblings carry identical parent rotations;
        on predicted data only the last sibling's channels receive FK gradient -- documented, not
        "fixed", because fidelity to the official recovery outranks tidiness);
      - the spurious root double-rotation correction stays REMOVED (anytop_rot6d_fk.py:150-159:
        with it FK-vs-RIC absL1=0.65, without it 0.0000 -- verified there on 1070 clips).
    Matrices are used throughout instead of the numpy path's rot6d->matrix->quat->matrix detour;
    the math is identical (the quat leg was a historical artifact) and stays branch-free, which
    autograd prefers. The conjugate-rotation of the root velocity becomes multiplication by the
    TRANSPOSED root matrix (conj(q) == R^T for rotation matrices).
"""
from __future__ import annotations

import torch

# ONE shared rot6d conversion for BOTH families (codex 01a01939 fix 1): the FK side must use the
# SAME kernel as the RIC side (world_recovery), or degenerate predicted rot6d (near-zero /
# near-parallel a1,a2) converts differently on the two paths and the loss punishes an ARTIFICIAL
# residual that no real disagreement produced. A local F.normalize-based copy did exactly that.
from src.models.graph_salad.world_recovery import _rot6d_to_matrix_torch as _rot6d_to_matrix


def fk_positions_torch(raw: torch.Tensor, parents, offsets: torch.Tensor) -> torch.Tensor:
    """World joint positions from the ROTATION family, one sample.

    Args:
      raw:     [T, J, 13] RAW (de-normalized) AnyTop motion, autograd-traceable.
      parents: [J] ints, FK order (parents[0] = -1, parents[j] < j).
      offsets: [J, 3] rest bone offsets, same joint order.
    Returns: [T, J, 3] world positions (differentiable in raw).
    """
    T, J, _ = raw.shape
    dev, dt = raw.device, raw.dtype
    root = raw[:, 0]                                              # [T,13]

    # ---- root orientation + integrated translation (mirrors _recover_root_quat_and_pos_np) ----
    R_root = _rot6d_to_matrix(root[:, 3:9])                       # [T,3,3] world->local
    v_loc = torch.zeros(T, 3, device=dev, dtype=dt)
    if T > 1:
        v_loc = torch.cat([torch.zeros(1, 3, device=dev, dtype=dt),
                           torch.stack([root[:-1, 9],
                                        torch.zeros(T - 1, device=dev, dtype=dt),
                                        root[:-1, 11]], dim=-1)], dim=0)
    # conj-quat rotation == R^T @ v (local -> world)
    v_world = torch.einsum("tji,tj->ti", R_root, v_loc)
    r_pos = torch.cumsum(v_world, dim=0)
    r_pos = torch.cat([r_pos[:, 0:1], root[:, 1:2], r_pos[:, 2:3]], dim=-1)   # y from ch1 directly

    # ---- per-joint matrices + the official parent-slot reindex (last child wins) ----
    all_mat = _rot6d_to_matrix(raw[:, :, 3:9])                    # [T,J,3,3]
    eye = torch.eye(3, device=dev, dtype=dt).expand(T, J, 3, 3)
    R_used = eye.clone()
    for j in range(1, J):
        p = int(parents[j])
        R_used[:, p] = all_mat[:, j]

    # ---- Animation.positions_global: 4x4 local->global chain ----
    loc = torch.zeros(T, J, 4, 4, device=dev, dtype=dt)
    loc[:, :, :3, :3] = R_used
    loc[:, :, :3, 3] = offsets.to(dt)[None].expand(T, J, 3).clone()
    loc[:, 0, :3, 3] = r_pos
    loc[:, :, 3, 3] = 1.0
    glob = [loc[:, 0]]
    for j in range(1, J):
        glob.append(torch.matmul(glob[int(parents[j])], loc[:, j]))
    g = torch.stack(glob, dim=1)                                  # [T,J,4,4]
    return g[:, :, :3, 3]


FK_SCALE_FRAC = 0.15  # smooth-L1 knee at 0.15 x mean bone length -- mirrors the user's proven
#                       hy273 recipe (fk_scale_m=0.05m against ~0.33m human mean bone ~= 0.15)


def fk_ric_consistency_loss(pred_norm, mean, std, std_floor, parents, offsets, n_joints,
                            frame_mask, ric_world_fn, want_diag=True):
    """gamma_7 core (Kimodo Eq.1 term 7): FK(pred rotations) vs RIC(pred positions) consistency.

    Follows the user's prior human-data implementation of the same Kimodo-style term
    (moge_UMO_ST models/raw_motion/hy273_multitask_losses.py): residual divided by a physical
    scale, then smooth-L1 (beta=1.0) -- quadratic near zero, bounded gradient on the huge
    mismatches an early-training x1_pred produces. (Kimodo's paper states plain ||.||_1; the
    user's own recipe is the field-tested variant and was explicitly offered as the reference.)

    pred_norm [B,T,J,13] NORMALIZED prediction; mean/std [B,J,13]; parents [B,J] long (pad -1,
    consumed as python ints on CPU); offsets [B,J,3]; n_joints [B]; frame_mask [B,T] marks the
    frames the term applies to (REAL target frames); ric_world_fn = the differentiable RIC
    recovery (world_recovery.recover_world_positions_torch), passed in to keep this module
    import-light.

    DELIBERATE DEVIATIONS FROM BOTH REFERENCES, forced by heterogeneous rigs (both references
    are single-human, fixed physical scale):
      - the scale is per-rig, FK_SCALE_FRAC x that rig's mean bone length, not a fixed 0.05m --
        otherwise a Trex contributes ~100x a Chick and gamma_fk trains large rigs only;
      - reduction is the per-element mean over (frames, joints, xyz), stable when J varies
        24..102 across the batch.
    Everything runs in fp32 with autocast DISABLED: a 100-link matmul chain in bf16 loses the
    small FK-vs-RIC differences this term exists to punish. Both families share the identical
    integrated root translation, so it cancels in the difference -- the term measures pure pose
    disagreement (exactly the H4 numerator).

    Returns (loss_term, diag_dist):
      loss_term -- scalar, smooth-L1 of scaled residual (train on gamma_fk * this);
      diag_dist -- scalar, DETACHED mean |FK-RIC| in mean-bone-length units (weight-0 diagnostic,
                   the online H4 monitor; mirrors hy273's fk_distance_cm).
    """
    dev_type = "cuda" if pred_norm.is_cuda else "cpu"
    # Hot-path sync discipline (codex 01a01939 fix 3): parents and n_joints arrive as CPU tensors
    # (the trainer's to_dev skips them), the frame mask is copied to CPU ONCE per call, and
    # per-sample frame selection uses index_select with an async H2D index instead of GPU boolean
    # masking (whose data-dependent output shape forces a sync per sample).
    fm_cpu = frame_mask.detach().to("cpu", non_blocking=False)
    with torch.autocast(device_type=dev_type, enabled=False):
        B = pred_norm.shape[0]
        total = pred_norm.sum() * 0.0
        diag_terms, count = [], 0
        for b in range(B):
            Jb = int(n_joints[b])
            idx_cpu = torch.nonzero(fm_cpu[b], as_tuple=False).flatten()
            if idx_cpu.numel() < 1:
                continue
            idx = idx_cpu.to(pred_norm.device, non_blocking=True)
            sel = pred_norm[b].index_select(0, idx)[:, :Jb].float()
            raw = sel * (std[b, :Jb][None].float() + std_floor) + mean[b, :Jb][None].float()
            fk = fk_positions_torch(raw, parents[b, :Jb].tolist(), offsets[b, :Jb].float())
            ric = ric_world_fn(raw[None])[0]
            # mean bone length of THIS rig (root offset is not a bone; guard tiny/degenerate rigs)
            bone = offsets[b, 1:Jb].float().norm(dim=-1).mean().clamp_min(1e-3) if Jb > 1 \
                else offsets.new_tensor(1.0)
            resid = (fk - ric) / (FK_SCALE_FRAC * bone)
            total = total + torch.nn.functional.smooth_l1_loss(
                resid, torch.zeros_like(resid), reduction="mean", beta=1.0)
            if want_diag:
                diag_terms.append((fk - ric).detach().norm(dim=-1).mean() / bone)
            count += 1
        n = max(count, 1)
        # diag floats sync -- computed only when the caller asked for parts (val/probes, not the
        # per-step train path)
        diag = float(torch.stack(diag_terms).mean()) if (want_diag and diag_terms) else 0.0
        return total / n, diag
