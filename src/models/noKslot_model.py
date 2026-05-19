"""noKslot_clean / src.models.noKslot_model — minimal model wrapper for the
no_k_slot training / eval / animate path.

Wraps three subcomponents that the no_k_slot forward pass uses (see source
scripts/train_paired_gate.py:208-231, the no_k_slot=True branch of
encode_decode):
  - encoder    : src.models.encoder.SkeletonEncoder
                 (h_tj per-joint encoder features + encode_skeleton method)
  - slot_norm  : src.models.slot_norm.SlotNorm
                 (per-feature LayerNorm calibrating h_tj scale)
  - decoder    : src.models.motion_decoder.MotionDecoder
                 (used as TopoFKTreeIKDecoder.base via return_features=True)

Intentionally does NOT contain:
  - SlotAssignment (K=24 Sinkhorn OT bottleneck) — user 2026-05-19 K=24
    Hold-not-sunset decision; the no_k_slot path bypasses SlotAssignment
    entirely via masked-identity assignment (_nok_identity_assignment in
    scripts/train.py), so it has no role here.

Loading the L6_anchor_h100_seed42 pre-trained ckpt:
  This wrapper is intentionally constructor-arg-compatible with source
  src/models/slot_ae.py:SlotAE for parameter-key compatibility — encoder.*,
  slot_norm.*, and decoder.* state-dict keys load 1:1 from the L6 ckpt.
  load_state_dict(..., strict=False) is REQUIRED because the L6 ckpt also
  contains slot_assignment.* keys which this minimal model intentionally
  omits; train.py logs the dropped/missed key lists for the audit record.

The TopoFKTreeIK head (src.models.treeik_decoder.TopoFKTreeIKDecoder) is
instantiated SEPARATELY in train.py / eval.py / animate.py and takes this
model.decoder as its base_decoder. The TreeIK head's optimizer params come
from TopoFKTreeIKDecoder.new_parameters() (which excludes self.base ==
model.decoder, matching the source C3 lesson at scripts/topofk_decoder.py:
256-262) so the shared decoder is registered exactly once in the optimizer.

Use sites:
  scripts/train.py    — instantiate, load L6 init (strict=False), wrap with
                        TopoFKTreeIKDecoder, train via no_k_slot path.
  scripts/eval.py     — load baseline ckpt, instantiate, eval same as train.
  scripts/animate.py  — same as eval.
"""

import torch.nn as nn

from .encoder import SkeletonEncoder
from .motion_decoder import MotionDecoder
from .slot_norm import SlotNorm


class NoKslotModel(nn.Module):
    """Minimal model: encoder + slot_norm + decoder. No SlotAssignment.

    Constructor args mirror source SlotAE.__init__ for L6 ckpt parameter-key
    compatibility (same defaults; only the slot_assignment-only `n_slots` arg
    is omitted since this minimal model has no SlotAssignment to size).
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        # Encoder
        n_graph_layers: int = 4,
        n_enc_temporal_layers: int = 2,
        # Decoder
        n_cross_layers: int = 3,
        n_dec_temporal_layers: int = 2,
        # Shared
        motion_feat_dim: int = 6,
        joint_feat_dim: int = 9,
        temporal_kernel: int = 9,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = SkeletonEncoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_graph_layers=n_graph_layers,
            n_temporal_layers=n_enc_temporal_layers,
            joint_feat_dim=joint_feat_dim,
            motion_feat_dim=motion_feat_dim,
            temporal_kernel=temporal_kernel,
            dropout=dropout,
        )

        # Per-feature LayerNorm calibrating encoder output scale; used by
        # the no_k_slot path on h_tj (K=Jpad identity-passthrough) — same
        # LayerNorm the source K-slot path applied to K-pooled slot features.
        self.slot_norm = SlotNorm(d_model)

        self.decoder = MotionDecoder(
            d_model=d_model,
            n_heads=n_heads,
            n_cross_layers=n_cross_layers,
            n_temporal_layers=n_dec_temporal_layers,
            motion_feat_dim=motion_feat_dim,
            temporal_kernel=temporal_kernel,
            dropout=dropout,
        )
