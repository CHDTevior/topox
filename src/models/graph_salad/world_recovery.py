"""Differentiable (torch) AnyTop RIFKE -> world-position recovery.

A torch port of `src.data.anytop_dataset._recover_world_positions` (numpy/scipy)
so that world-space joint geometry can be supervised inside the training loop
(loss_mode="anytop13_world_geometry").

WHY a torch port instead of reusing the numpy one:
  The numpy version uses scipy.Rotation (.inv().apply()) + np.cumsum — not
  autograd-traceable. To put a geometry loss on the decoder's predicted 13ch
  motion we need the recovery to be differentiable end-to-end.

CORRECTNESS CONTRACT (smoke gate 1):
  recover_world_positions_torch(x) must match
  src.data.anytop_dataset._recover_world_positions(x) to ~1e-4 on real clips.
  Both implement AnyTop motion_process.recover_from_bvh_ric_np:
    1. root rotation per frame from 6D rot (channels 3:9)
    2. root xz: inverse-rotate per-frame vel (ch 9=x, 11=z, shifted by 1), cumsum
    3. root y: channel 1 directly (not integrated)
    4. non-root joints: channels 0:3 are root-relative; inverse-rotate to world,
       then add root xz (NOT root y — non-root carries its own y in 0:3).

The 6D->matrix convention is identical to numpy `_rotation_6d_to_matrix_np` and
torch `treeik_decoder.rot6d_to_matrix` (Zhou 2019 Gram-Schmidt, columns
[b1,b2,b3]); verified equivalent. R[...,:,k] is the k-th column, so
inverse-rotate v is R^T @ v == einsum('...ij,...j->...i', R.transpose(-1,-2), v).
"""
from __future__ import annotations

import torch


def _rot6d_to_matrix_torch(d6: torch.Tensor) -> torch.Tensor:
    """6D rotation -> [...,3,3], columns [b1,b2,b3]. Matches numpy
    `_rotation_6d_to_matrix_np` and `treeik_decoder.rot6d_to_matrix` bit-for-bit
    in formula (Gram-Schmidt, +1e-8 norm floor)."""
    a1 = d6[..., :3]
    a2 = d6[..., 3:]
    b1 = a1 / (a1.norm(dim=-1, keepdim=True) + 1e-8)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = b2 / (b2.norm(dim=-1, keepdim=True) + 1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # [...,3,3]


def recover_world_positions_torch(motion_13ch: torch.Tensor) -> torch.Tensor:
    """Differentiable world-space [B,T,J,3] from AnyTop RIFKE [B,T,J,13] (raw).

    Args:
      motion_13ch: [B,T,J,13] RAW (un-normalized) AnyTop encoding.
    Returns:
      [B,T,J,3] world-space joint positions (autograd-traceable in motion_13ch).

    Mirrors numpy `_recover_world_positions` (anytop_dataset.py:282) exactly;
    see module docstring for the 4-step contract.
    """
    if motion_13ch.dim() != 4 or motion_13ch.shape[-1] != 13:
        raise ValueError(
            f"recover_world_positions_torch: expected [B,T,J,13], got "
            f"{tuple(motion_13ch.shape)}"
        )
    B, T, J, _ = motion_13ch.shape
    m = motion_13ch.to(torch.float32)
    root = m[:, :, 0, :]  # [B,T,13]

    # 1. Root rotation per frame from 6D rot (channels 3:9). R[...,:,k]=col k.
    R = _rot6d_to_matrix_torch(root[:, :, 3:9])  # [B,T,3,3]
    Rt = R.transpose(-1, -2)  # inverse rotation (R orthonormal) == scipy .inv()

    # 2. Root xz integration. AnyTop: vel_x=ch9, vel_z=ch11, shifted by 1 frame
    #    (no motion at t=0); idx 10 NOT used in root recovery. Inverse-rotate the
    #    per-frame local velocity into world frame, then cumsum over time.
    rpos_local = torch.zeros(B, T, 3, dtype=torch.float32, device=m.device)
    if T > 1:
        rpos_local[:, 1:, 0] = root[:, :-1, 9]   # vel_x at t-1
        rpos_local[:, 1:, 2] = root[:, :-1, 11]  # vel_z at t-1
    # inverse-rotate per frame: Rt @ v  (einsum over the matrix's row/col)
    rpos_world = torch.einsum("btij,btj->bti", Rt, rpos_local)  # [B,T,3]
    rpos_world = torch.cumsum(rpos_world, dim=1)
    # Root y from channel 1 directly (overwrite integrated y, which was 0 anyway).
    rpos_world = rpos_world.clone()
    rpos_world[:, :, 1] = root[:, :, 1]

    # 3. Non-root joints: channels 0:3 are root-relative; inverse-rotate to world,
    #    then add root xz origin shift (NOT root y).
    if J > 1:
        rel = m[:, :, 1:, :3]  # [B,T,J-1,3]
        # world_rel[b,t,j] = Rt[b,t] @ rel[b,t,j]
        world_rel = torch.einsum("btij,btnj->btni", Rt, rel)  # [B,T,J-1,3]
        # add root xz (broadcast over joints); y untouched
        shift = torch.zeros_like(world_rel)
        shift[..., 0] = rpos_world[:, :, None, 0]
        shift[..., 2] = rpos_world[:, :, None, 2]
        world_rel = world_rel + shift
        world = torch.cat([rpos_world[:, :, None, :], world_rel], dim=2)
    else:
        world = rpos_world[:, :, None, :]
    return world  # [B,T,J,3]
