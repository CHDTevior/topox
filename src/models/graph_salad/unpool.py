"""DynamicGraphUnpool — assignment-based unpool + temporal upsample.

Inverse of DynamicGraphPool / DeterministicGraphPool. Used by M1.3 VAE
decoder to lift latent (coarse) features back to fine joints.

Operation per plan §7:
  1. Skeletal unpool: h_fine[b,t,j,d] = sum_c P[b,j,c] * h_coarse[b,t,c,d]
     (P is the assignment matrix saved from the corresponding pool — same
      orientation, not transposed; pool was h_coarse = P^T·h_fine, unpool is
      h_fine = P·h_coarse. Source slot_assignment.py uses the same convention.)
  2. Temporal upsample T_lat → T_lat * stride via repeat_interleave (nearest).

The unpool is symmetric for the dynamic + deterministic + no-pool paths:
all three produce an assignment P with shape [B, J_max, C_max] (one-hot in
deterministic and no-pool cases; soft in dynamic).

For the no-pool VAE variant we don't actually pool, so unpool is a no-op:
the caller skips this module entirely and uses h_coarse=h_fine (with C=J).
This module's contract assumes C ≠ J (else stride=1 makes it trivial).

Aux outputs return frame_mask_up so M1.3 VAE can stack mask information
through the encoder/decoder pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DynamicGraphUnpool(nn.Module):
    """Assignment-based unpool + temporal upsample.

    Args:
        d_model: feature dim of coarse / fine features (kept for API parity;
            no projections used). Must be > 0 strict int.
        temporal_stride: T_lat → T_lat * stride upsample factor. Must equal
            the pool's temporal_stride for round-trip consistency. Default 2.
    """

    def __init__(
        self,
        d_model: int,
        *,
        temporal_stride: int = 2,
    ) -> None:
        super().__init__()
        if not isinstance(d_model, int) or isinstance(d_model, bool):
            raise TypeError(f"d_model must be strict int, got {type(d_model).__name__}")
        if not isinstance(temporal_stride, int) or isinstance(temporal_stride, bool):
            raise TypeError(f"temporal_stride must be strict int, got {type(temporal_stride).__name__}")
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")
        if temporal_stride < 1:
            raise ValueError(f"temporal_stride must be ≥ 1, got {temporal_stride}")

        self.d_model = d_model
        self.temporal_stride = temporal_stride
        # No nn.Parameters — state_dict() empty, ckpt envelope preserved.

    def forward(
        self,
        coarse_features: torch.Tensor,  # [B, T_lat, C_max, D]
        assignment: torch.Tensor,        # [B, J_max, C_max]
        joint_mask: torch.Tensor,        # [B, J_max]
        coarse_mask: torch.Tensor,       # [B, C_max]
        frame_mask_down: torch.Tensor,   # [B, T_lat]
    ) -> dict:
        """Returns dict with:
            fine_features:  [B, T_lat * stride, J_max, D]
            frame_mask_up:  [B, T_lat * stride] bool (repeat-interpolate from down)
        """
        # --- Input validation ---
        if coarse_features.dim() != 4 or coarse_features.shape[-1] != self.d_model:
            raise ValueError(
                f"coarse_features must be [B, T_lat, C, {self.d_model}], "
                f"got {tuple(coarse_features.shape)}"
            )
        B, T_lat, C, D = coarse_features.shape
        if B <= 0 or T_lat <= 0 or C <= 0:
            raise ValueError(f"empty input: B={B} T_lat={T_lat} C={C}")
        if assignment.dim() != 3:
            raise ValueError(f"assignment must be [B, J, C], got {tuple(assignment.shape)}")
        if assignment.shape[0] != B or assignment.shape[2] != C:
            raise ValueError(
                f"assignment shape {tuple(assignment.shape)} != [B={B}, J, C={C}]"
            )
        J = assignment.shape[1]
        if joint_mask.shape != (B, J) or joint_mask.dtype != torch.bool:
            raise ValueError(
                f"joint_mask must be [B={B}, J={J}] bool, got "
                f"{tuple(joint_mask.shape)} {joint_mask.dtype}"
            )
        if coarse_mask.shape != (B, C) or coarse_mask.dtype != torch.bool:
            raise ValueError(
                f"coarse_mask must be [B={B}, C={C}] bool, got "
                f"{tuple(coarse_mask.shape)} {coarse_mask.dtype}"
            )
        if frame_mask_down.shape != (B, T_lat) or frame_mask_down.dtype != torch.bool:
            raise ValueError(
                f"frame_mask_down must be [B={B}, T_lat={T_lat}] bool, got "
                f"{tuple(frame_mask_down.shape)} {frame_mask_down.dtype}"
            )

        # Dtype + device consistency
        ref_device = coarse_features.device
        for name, t in (("assignment", assignment),):
            if t.device != ref_device:
                raise ValueError(
                    f"{name}.device {t.device} != coarse_features.device {ref_device}"
                )
            if t.dtype != torch.float32:
                raise ValueError(f"{name}.dtype must be float32, got {t.dtype}")
        if coarse_features.dtype != torch.float32:
            raise ValueError(f"coarse_features.dtype must be float32, got {coarse_features.dtype}")

        # Finite check
        if not torch.isfinite(coarse_features).all():
            raise ValueError("coarse_features contains NaN or Inf")
        if not torch.isfinite(assignment).all():
            raise ValueError("assignment contains NaN or Inf")

        # assignment row contract: padded joints' rows ~0; valid joints' rows sum ~1
        # (soft) or exactly 1 (one-hot). We accept either.
        # Codex M1.2 unpool round 1 fixes:
        # (a) require non-negative entries (catches extrapolation row like [1.5, -0.5])
        # (b) require padded-coarse-column entries are 0 (catches valid joint
        #     reading padded coarse via row=[0,1] + coarse_mask=[T,F])
        if (assignment < -1e-6).any():
            raise ValueError(
                "DynamicGraphUnpool: assignment contains negative entries "
                "(must be a convex combination, not extrapolation)"
            )
        # Padded coarse columns: any non-zero entry in coarse_mask=False columns
        # means a fine joint is reading from a padded coarse slot.
        padded_col = ~coarse_mask  # [B, C]
        if padded_col.any():
            # Mask out padded columns of assignment; should be all zero on these.
            padded_col_assign = assignment * padded_col.unsqueeze(1).to(assignment.dtype)
            if padded_col_assign.abs().max() > 1e-6:
                raise ValueError(
                    "DynamicGraphUnpool: assignment has non-zero entries in "
                    "coarse_mask=False columns (fine joint would read padded "
                    "coarse features)"
                )
        row_sums = assignment.sum(dim=-1)  # [B, J]
        # Handle empty-valid case (no valid joints in batch → vacuous OK)
        if joint_mask.any():
            valid_row_sums = row_sums[joint_mask]
            if not torch.allclose(valid_row_sums, torch.ones_like(valid_row_sums), atol=1e-5):
                raise ValueError(
                    f"DynamicGraphUnpool: assignment row sums on valid joints must "
                    f"be ~1.0, got min/max = "
                    f"{valid_row_sums.min().item()}/{valid_row_sums.max().item()}"
                )
        # Padded rows must be ~0
        padded_row_sums = row_sums[~joint_mask]
        if padded_row_sums.numel() > 0:
            if padded_row_sums.abs().max() > 1e-5:
                raise ValueError(
                    f"DynamicGraphUnpool: assignment rows for padded joints "
                    f"must be ≈0, got max |sum| = {padded_row_sums.abs().max().item()}"
                )

        # --- 1. Skeletal unpool: h_fine = P · h_coarse ---
        # P [B,J,C] @ h_coarse [B,T,C,D] → h_fine [B,T,J,D]
        fine_features = torch.einsum("bjc,btcd->btjd", assignment, coarse_features)
        # Zero padded joints (defense in depth; assignment already zeros them)
        fine_features = fine_features * joint_mask[:, None, :, None].to(fine_features.dtype)

        # --- 2. Temporal upsample T_lat → T_lat * stride ---
        # Codex M1.2 unpool round 1 fix #3: features and mask MUST use the same
        # kernel. Previously features=linear, mask=repeat_interleave → an
        # interpolated boundary up-frame depended on an invalid neighbor but
        # was marked True. Switch features to NEAREST (mode='nearest-exact')
        # so both use the same per-frame mapping: up-frame t_up ↔ down-frame
        # t_down = t_up // stride. Plan §7.3 lists 'nearest' as acceptable
        # alternative to linear; we trade smoothness for mask-correctness.
        s = self.temporal_stride
        T_up = T_lat * s
        if s == 1:
            fine_up = fine_features
            frame_mask_up = frame_mask_down
        else:
            # repeat_interleave is exact-nearest for both features and mask.
            # Each down-frame's value repeats `s` times in up-frame.
            fine_up = fine_features.repeat_interleave(s, dim=1)
            frame_mask_up = frame_mask_down.repeat_interleave(s, dim=-1)

        # Zero invalid frames (defense in depth)
        fine_up = fine_up * frame_mask_up[:, :, None, None].to(fine_up.dtype)

        return {
            "fine_features": fine_up,
            "frame_mask_up": frame_mask_up,
        }
