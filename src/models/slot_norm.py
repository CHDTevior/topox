"""noKslot_clean / src.models.slot_norm — SlotNorm extracted from source repo's
src/models/slot_assignment.py:256-265.

SlotNorm is a per-feature LayerNorm wrapper used by the no_k_slot path to
calibrate the per-joint encoder features `h_tj` after they are bypassed past
the K=24 Sinkhorn bottleneck. The default (K=24) path used the same SlotNorm
on the K-pooled slot features; the no_k_slot path applies it identically with
K=Jpad (identity passthrough; per-joint features become the "slots").

Intentionally NOT brought over from source slot_assignment.py:
  - SlotAssignment class (the K=24 Sinkhorn OT bottleneck — user decision
    2026-05-19 Hold-not-sunset; bypass via masked-identity assignment in the
    no_k_slot path is the whole point of this clean repo)
  - sinkhorn_normalization helper (only used by SlotAssignment)
  - assign_anchor_loss function (Phase 2b paired-gate anchor loss, K-slot only)
"""

import torch
import torch.nn as nn


class SlotNorm(nn.Module):
    """Per-slot LayerNorm to calibrate feature scale across different J."""

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, slot_features: torch.Tensor) -> torch.Tensor:
        """slot_features: [B, T, K, D]"""
        return self.norm(slot_features)
