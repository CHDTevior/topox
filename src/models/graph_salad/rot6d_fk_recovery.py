"""Differentiable (torch) AnyTop rot6d-FK position recovery.

A torch, matrix-composition port (official-derived, PATCHED 2026-06-01) of the
AnyTop/SALAD `recover_from_bvh_rot_np` (truebones motion_process.py:750) —
recovers world joint positions from the ROTATION channels (3:9) via FK on each
sample's parent
tree with rest offsets, plus the integrated root translation. This is the
"Route B" the user's diagnostic measures; the differentiable version lets the
VAE be supervised on rot6d-FK geometry (loss_mode="anytop13_world_rot6d_fk").

DESIGN (plan §4): matrix composition only — NO quaternion sign branches. The
numpy reference (src/data/anytop_rot6d_fk.py) routes matrix->quat->reindex->
quat->matrix purely to reuse the SALAD Quaternions class; that round-trip is an
identity on rotations, so the reindex is done DIRECTLY on the 3x3 matrices here:
  - parent reindex `rot_q[:,p] = all_q[:,j]` (motion_process.py:758) becomes
    `local_R[:, :, p] = all_R[:, :, j]`.

PATCH (2026-06-01): the official source applies a root correction
`local_R[0] = root_R^T @ local_R[0]` (== numpy's `-r_rot_quat * rot_q[0]`) that
DOUBLE-applies the root global rotation (turn yaw twice). REMOVED here — the
ch3:9 root/child tokens already carry the global facing; the reindex puts the
correct root orientation in local_R[0]. Verified: WITH it FK-vs-RIC absL1=0.65
(sweep ~1.98); WITHOUT it absL1=0.0000 (FK == RIC) on clean_L2 + 1070
old-truebones largest-rotation clips.
Smoke gate 1 (plan §8): max_abs_error vs the numpy port < 1e-4 on real clips
(both paths patched).

Root recovery reuses the verified RIC root path (world_recovery._rot6d_to_matrix
_torch + the same vel-cumsum + height logic), so root trajectory matches RIC to
~0 (the two routes share root channels).
"""
from __future__ import annotations

import torch

from src.models.graph_salad.world_recovery import _rot6d_to_matrix_torch


def _recover_root_R_and_pos(root_13: torch.Tensor):
    """Root rotation matrix [B,T,3,3] + world root position [B,T,3] from the root
    joint's 13ch. Mirrors world_recovery (RIC root path) exactly so the FK root
    trajectory == the RIC root trajectory.

    root_13: [B,T,13].
    """
    B, T, _ = root_13.shape
    R = _rot6d_to_matrix_torch(root_13[:, :, 3:9])            # [B,T,3,3]
    Rt = R.transpose(-1, -2)                                  # inverse rotation
    rpos_local = torch.zeros(B, T, 3, dtype=torch.float32, device=root_13.device)
    if T > 1:
        rpos_local[:, 1:, 0] = root_13[:, :-1, 9]            # vel_x at t-1
        rpos_local[:, 1:, 2] = root_13[:, :-1, 11]           # vel_z at t-1
    rpos_world = torch.einsum("btij,btj->bti", Rt, rpos_local)  # inverse-rotate
    rpos_world = torch.cumsum(rpos_world, dim=1)
    rpos_world = rpos_world.clone()
    rpos_world[:, :, 1] = root_13[:, :, 1]                    # root height (ch1)
    return R, rpos_world


def recover_rot6d_fk_positions_torch(
    motion_13ch: torch.Tensor,          # [B,T,J,13], RAW (un-normalized)
    parent_indices,                     # list length B of list[int]
    rest_offsets: torch.Tensor,         # [B,J,3]
    joint_mask: torch.Tensor,           # [B,J] bool
) -> torch.Tensor:                      # [B,T,J,3]
    """Differentiable rot6d-FK world positions (official Route-B logic).

    Per-sample loop (B small; J<=144; this is a VAE loss, not a sampling loop).
    autograd-traceable through motion_13ch (the 6D rotation channels 3:9) and
    rest_offsets.
    """
    if motion_13ch.dim() != 4 or motion_13ch.shape[-1] != 13:
        raise ValueError(
            f"recover_rot6d_fk_positions_torch: expected [B,T,J,13], got "
            f"{tuple(motion_13ch.shape)}")
    B, T, J, _ = motion_13ch.shape
    dev = motion_13ch.device
    m = motion_13ch.to(torch.float32)
    rest = rest_offsets.to(torch.float32)
    out = m.new_zeros(B, T, J, 3)

    for b in range(B):
        par = [int(p) for p in parent_indices[b]]
        Jb = min(len(par), J, int(joint_mask[b].sum().item()) if joint_mask is not None else J)
        if Jb < 1:
            continue
        # 1. root R + world root pos (shares RIC root path)
        root_R, root_pos = _recover_root_R_and_pos(m[b:b + 1, :, 0, :])  # [1,T,3,3],[1,T,3]
        root_R = root_R[0]                                              # [T,3,3]
        root_pos = root_pos[0]                                         # [T,3]
        # 2. non-root local rotation matrices from ch3:9
        nonroot_R = _rot6d_to_matrix_torch(m[b, :, 1:Jb, 3:9])         # [T,Jb-1,3,3]
        # 3. all_R = concat(root_R, nonroot_R)  [T,Jb,3,3]
        all_R = torch.cat([root_R[:, None], nonroot_R], dim=1)        # [T,Jb,3,3]
        # 4. parent reindex (motion_process.py:758): local_R[p] = all_R[j].
        #    Done on matrices (identity vs the numpy quat round-trip). Build a
        #    per-joint list (autograd-safe; no inplace on a graph tensor).
        eye = torch.eye(3, dtype=torch.float32, device=dev).expand(T, 3, 3)
        local_R_list = [eye for _ in range(Jb)]               # [T,3,3] each
        for j in range(1, Jb):
            p = par[j]
            if 0 <= p < Jb:
                local_R_list[p] = all_R[:, j]
        # NOTE: official applies a root correction here (local_R[0] =
        # root_R^T @ local_R[0], == numpy's -r_rot_quat * rot_q[0]). REMOVED
        # 2026-06-01 — it DOUBLE-applies the root global rotation (turn yaw
        # applied twice). The ch3:9 root/child tokens already carry the global
        # facing; the parent reindex puts the correct orientation into
        # local_R[0]. Verified (RIC = ground truth): WITH it FK-vs-RIC absL1=
        # 0.65 (sweep ~1.98); WITHOUT it absL1=0.0000 on clean_L2 + 1070
        # old-truebones largest-rotation clips. local_R[0] keeps reindexed value.
        # 6. local translations: offsets per joint; root = world root pos
        local_t_list = [rest[b, j].unsqueeze(0).expand(T, 3) for j in range(Jb)]
        local_t_list[0] = root_pos                            # [T,3]
        # 7. FK chain on 4x4 homogeneous transforms (build global per-joint;
        #    list accumulation keeps autograd happy — no inplace writes).
        bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev).expand(T, 1, 4)

        def _to_T(R3, t3):  # [T,3,3],[T,3] -> [T,4,4]
            top = torch.cat([R3, t3[:, :, None]], dim=-1)     # [T,3,4]
            return torch.cat([top, bottom], dim=1)            # [T,4,4]

        glob_list = [None] * Jb
        glob_list[0] = _to_T(local_R_list[0], local_t_list[0])
        for j in range(1, Jb):
            p = par[j]
            pp = p if 0 <= p < Jb else 0
            loc_j = _to_T(local_R_list[j], local_t_list[j])
            glob_list[j] = torch.matmul(glob_list[pp], loc_j)
        glob = torch.stack(glob_list, dim=1)                  # [T,Jb,4,4]
        out = out.clone()
        out[b, :, :Jb] = glob[:, :, :3, 3]
    # zero padded joints
    return out * joint_mask[:, None, :, None].to(out.dtype)
