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
            bone = offsets[b, 1:Jb].float().norm(dim=-1).mean() + 1e-3 if Jb > 1 \
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


def _ktjd_cont6d_torch(d6: torch.Tensor) -> torch.Tensor:
    """Differentiable mirror of codec.decode_column_cont6d (COLUMN Gram-Schmidt, columns stacked
    on the LAST axis). [*,6] -> [*,3,3].

    THE FLOOR IS ADDITIVE, NOT A CLAMP -- and that is a gradient property, not a style choice
    (2026-08-21, after this term diverged a 400-epoch run):

        n / clamp_min(n, 1e-8)   is EXACTLY the true norm whenever n > 1e-8, so a predicted 6D
                                 vector passing near zero (say n = 1e-7) still divides by 1e-7 and
                                 contributes d/dn ~ 1/n^2 ~ 1e14 to the backward pass. The clamp
                                 only fires BELOW the floor, where it zeroes the gradient instead
                                 -- so the singularity survives in the band just above it.
        n + 1e-8                 is bounded below by the floor EVERYWHERE and is smooth, so the
                                 derivative is bounded by 1/(1e-8)^2 only in the limit and is
                                 continuous through zero.

    The numpy codec can use a clamp because it never back-propagates. The 13ch kernel this mirrors
    (world_recovery._rot6d_to_matrix_torch) uses the additive form for exactly this reason, and the
    13ch runs -- which never enabled an FK term at all -- never exercised this path. Measured
    consequence of the clamp: gradient norm went 4.2e2 -> 5.0e10 -> 4.1e12 across three epochs with
    the non-finite counter still at zero, i.e. a live near-singularity rather than a diverging lr.
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / (a1.norm(dim=-1, keepdim=True) + 1e-8)
    u2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = u2 / (u2.norm(dim=-1, keepdim=True) + 1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def fk_ktjd_consistency_loss(pred_norm, mean, std, std_floor, parents, offsets, R_rest_global,
                             n_joints, frame_mask, want_diag=True):
    """gamma_7 for KTJD-17: FK(pred rotations) vs DIRECT positions, mirroring the official
    decode_ktjd17 (src/data/ktjd17/decoder.py) exactly, channel for channel:
      direct  = raw[...,0:3] with the root's smooth-root ch13/ch14 added to every joint's x/z
                (codec.direct_decode_positions);
      R_global = cont6d(raw[...,3:9]) @ R_rest_global   (decoder proposed_global);
      fk       = chain p_child = p_parent + R_global(parent) @ offset_child, root at direct[:,0]
                (codec.fk_from_global_rotations).
    NO temporal integration, NO velocity use -- both paths are frame-local, per the KTJD
    contract. fixed_dof rows are NOT implemented (the current 66 rigs have zero of them); the
    pack builder asserts that so a future rig fails loud instead of silently mis-decoding.

    pred_norm [B,T,J,17] NORMALIZED; mean/std [B,J,>=17] (the adapter's rest-centering offset in
    raw units and its de-normalization scale -- raw = x*(std+floor)+mean, the standard
    convention; mean is NOT zero since the 2026-08-20 rest-centering fix); parents [B,J] long
    CPU; offsets [B,J,3]; R_rest_global [B,J,3,3]; n_joints [B]; frame_mask [B,T]. Same per-rig
    bone-length scaling, smooth-L1, fp32-under-autocast-off, and sync discipline as
    fk_ric_consistency_loss above.
    """
    dev_type = "cuda" if pred_norm.is_cuda else "cpu"
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
            sel = pred_norm[b].index_select(0, idx)[:, :Jb].float()          # [F,J,17]
            raw = (sel * (std[b, :Jb, :17][None].float() + std_floor)
                   + mean[b, :Jb, :17][None].float())
            direct = raw[..., 0:3].clone()
            direct[..., 0] = direct[..., 0] + raw[:, 0:1, 13]
            direct[..., 2] = direct[..., 2] + raw[:, 0:1, 14]
            Rg = _ktjd_cont6d_torch(raw[..., 3:9]) @ R_rest_global[b, :Jb].float()  # [F,J,3,3]
            par = parents[b, :Jb].tolist()
            pos = [direct[:, 0]]
            off = offsets[b, :Jb].float()
            for child in range(1, Jb):
                p = int(par[child])
                pos.append(pos[p] + (Rg[:, p] @ off[child]))
            fk = torch.stack(pos, dim=1)                                     # [F,J,3]
            bone = off[1:].norm(dim=-1).mean() + 1e-3 if Jb > 1 \
                else off.new_tensor(1.0)
            resid = (fk - direct) / (FK_SCALE_FRAC * bone)
            total = total + torch.nn.functional.smooth_l1_loss(
                resid, torch.zeros_like(resid), reduction="mean", beta=1.0)
            if want_diag:
                diag_terms.append((fk - direct).detach().norm(dim=-1).mean() / bone)
            count += 1
        n = max(count, 1)
        diag = float(torch.stack(diag_terms).mean()) if (want_diag and diag_terms) else 0.0
        return total / n, diag


# UMO supervises (x[t+1]-x[t])*fps in metres and weights it 0.01. KTJD skeletons are ALREADY
# scale-normalized at storage time -- measured mean bone length 0.209 on every rig sampled -- so a
# single shared scale is as meaningful here as metres are there. The fps factor is deliberately
# ABSORBED into that scale: it is a constant, and reusing gamma_7's own unit (FK_SCALE_FRAC x mean
# bone length, applied to per-FRAME displacement) keeps the two physical terms on one convention.
# Measured GT displacement in that unit: 0.02 .. 7.1, median ~1.4 -- i.e. right on smooth-L1's
# beta=1 knee, the same regime UMO's ~1 m/s residuals sit in, so their 0.01 weight transfers.
# NOT used: the window's own GT speed as denominator. 345/986 clips have a static root and many
# windows are near-static, so that denominator collapses and the term explodes exactly where the
# correct answer is "stay still".


def ktjd_dynamics_losses(pred_norm, x1_norm, mean, std, std_floor, offsets, n_joints, frame_mask,
                         contact_on=None, want_diag=True):
    """UMO's anti-degenerate pair, ported to KTJD-17 (2026-08-20 frozen-pose fix).

    UMO/HY273 carries FOUR terms Eq.1 does not, two of which exist specifically to make a frozen
    output impossible (train_hy273_raw_flow.py:733-757, weights 0.01 each):

      clean_root_velocity / clean_joint_velocity -- finite differences on the DENORMALIZED
        prediction, supervised against the same statistic of GT. Their own comment: "a static
        output has zero predicted velocity against non-zero targets, so these terms are exactly
        the ones a frozen solution cannot satisfy". This is NOT the same as supervising the ch9:12
        velocity CHANNELS (which KTJD never integrates, and which a frozen output can satisfy by
        emitting the right channel value while holding the pose still): here the constraint couples
        consecutive POSITION frames, so it can only be met by actually moving.

      foot_lock -- the asymmetric counterpart: zero displacement demanded ONLY where GT contact is
        on at both endpoints of the pair. Together the two block "frozen everywhere" and "sliding
        everywhere" from opposite sides. KTJD's contact is PER JOINT (ch12 on every row), not four
        feet, so the term generalizes to every contacting joint.

    Both operate on KTJD's direct-decode world positions -- q_position plus the root's smooth-root
    XZ (codec.direct_decode_positions), the same quantity the renderer draws.

    pred_norm/x1_norm [B,T,J,17]; mean/std [B,J,>=17]; offsets [B,J,3] rest bone offsets;
    n_joints [B]; frame_mask [B,T] real target frames; contact_on [B,T,J] bool (GT contact) or
    None to skip foot_lock. Returns (vel_term, lock_term, diag_articulation_ratio, n_windows); the
    count is how many windows contributed a ratio, so the caller can average without zero-imputing
    static-GT batches. The diagnostic is
    the predicted POSE-RELATIVE (root-subtracted) articulation speed over GT's, computed only over
    windows whose GT actually articulates -- a world-space ratio would read ~1x for the frozen-body-
    dragged-by-a-moving-root failure this monitor exists to catch.
    """
    dev_type = "cuda" if pred_norm.is_cuda else "cpu"
    fm_cpu = frame_mask.detach().to("cpu", non_blocking=False)
    with torch.autocast(device_type=dev_type, enabled=False):
        B = pred_norm.shape[0]
        vel_t = pred_norm.sum() * 0.0
        lock_t = pred_norm.sum() * 0.0
        ratios, nv, nl = [], 0, 0
        for b in range(B):
            Jb = int(n_joints[b])
            idx_cpu = torch.nonzero(fm_cpu[b], as_tuple=False).flatten()
            if idx_cpu.numel() < 2:                     # a pair of frames is required
                continue
            idx = idx_cpu.to(pred_norm.device, non_blocking=True)
            sc = std[b, :Jb, :17][None].float() + std_floor
            mn = mean[b, :Jb, :17][None].float()

            def world(t_norm):
                raw = t_norm.index_select(0, idx)[:, :Jb].float() * sc + mn
                w = raw[..., 0:3].clone()
                w[..., 0] = w[..., 0] + raw[:, 0:1, 13]
                w[..., 2] = w[..., 2] + raw[:, 0:1, 14]
                return w                                                  # [F,J,3]

            wp, wg = world(pred_norm[b]), world(x1_norm[b])
            off = offsets[b, :Jb].float()
            # additive floor, not clamp_min: see _ktjd_cont6d_torch -- a clamped
            # denominator keeps the true (possibly tiny) value in its gradient.
            bone = off[1:].norm(dim=-1).mean() + 1e-3 if Jb > 1 else off.new_tensor(1.0)
            scale = FK_SCALE_FRAC * bone                                  # gamma_7's own unit
            dp = (wp[1:] - wp[:-1]) / scale
            dg = (wg[1:] - wg[:-1]) / scale
            vel_t = vel_t + torch.nn.functional.smooth_l1_loss(
                dp, dg, reduction="mean", beta=1.0)
            nv += 1
            if want_diag and Jb > 1:
                # POSE-RELATIVE, not world (codex round-S2 #6): the production failure was a frozen
                # body dragged by a moving root, and a WORLD speed ratio reports ~1x for exactly
                # that case -- it would have missed the very collapse this monitor exists to catch.
                # Subtracting the root per frame and dropping the (now identically zero) root row
                # measures ARTICULATION only: the diagnosis measured 0.098-0.114 here while world
                # motion ran at 1.6-1.8x GT.
                rp = (wp - wp[:, 0:1])[:, 1:]
                rg = (wg - wg[:, 0:1])[:, 1:]
                dpr = (rp[1:] - rp[:-1]) / scale
                dgr = (rg[1:] - rg[:-1]) / scale
                gs = dgr.norm(dim=-1).mean().detach()
                if float(gs) > 1e-3:                    # a static GT window has no ratio to report
                    ratios.append(dpr.norm(dim=-1).mean().detach() / gs)
            if contact_on is not None:
                c = contact_on[b].index_select(0, idx)[:, :Jb]             # [F,J] bool
                pair = c[1:] & c[:-1]
                if bool(pair.any()):
                    lock_t = lock_t + (dp ** 2 * pair[..., None]).sum() / pair.sum().clamp_min(1)
                    nl += 1
        # Return the COUNT alongside the mean (codex round-S3): a batch whose GT does not
        # articulate produces NO ratio, and folding its 0.0 into a running mean zero-imputes it
        # and mis-weights every other batch. The caller accumulates sum/count instead.
        diag = float(torch.stack(ratios).mean()) if (want_diag and ratios) else 0.0
        return vel_t / max(nv, 1), lock_t / max(nl, 1), diag, len(ratios)
