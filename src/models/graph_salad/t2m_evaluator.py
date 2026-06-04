"""AnyTop T2M Evaluator (M2) — independent text↔motion contrastive evaluator.

Design doc: handoff/20260604_0121_anytop_t2m_evaluator_impl_proposal.md
  (§1c hard requirements + §5-M2) and
  handoff/20260604_0257_anytop_t2m_evaluator_impl_progress.md (M2 NEXT).

WHAT THIS IS
------------
A small (~14M) two-tower model that maps a text caption and a motion clip into a
*shared* 384-d coembedding space, trained with symmetric, multi-positive InfoNCE
on REAL training motions. Its frozen embeddings drive the paper's T2M metrics
(group-aware R-precision / multi-positive matching score / FID) for the AnyTop
VAE + diffusion generators — i.e. it is the *measuring instrument*, never part of
the generator.

INDEPENDENT-FROZEN-EVALUATOR CONTRACT (§1c hard requirements 1-3)
----------------------------------------------------------------
  1. This evaluator is a STANDALONE model. Its motion tower is a *freshly
     initialized* `SkeletonEncoder` — it shares NO weights with the VAE encoder
     or the diffusion denoiser, and NEVER loads a VAE/denoiser checkpoint.
  2. It is trained ONLY on real training motions (never on generated samples).
  3. It does NOT consume the VAE latent z. The motion tower ingests the raw
     AnyTop 13-channel motion (`batch.anytop_x`) + graph tensors + masks,
     exactly the way `GraphMotionVAE.encode` feeds its own encoder — so the eval
     distribution matches the training distribution — but with its own weights.
  Downstream (M3/M4), the trained evaluator is loaded, `.eval()`-ed, and run
  under `torch.no_grad()`; the generators are never backproped through it.

TWO TOWERS (§1c-4/5, shared coemb_dim=384 — NO motion_proj)
-----------------------------------------------------------
  text tower:   T5 caption embedding [B,768] --MLP--> text_emb [B,384]
  motion tower: SkeletonEncoder(d_model=384, ...) -> [B,T,J,384]
                --masked mean over valid (t,j)--> motion_emb [B,384]
  SkeletonEncoder's d_model IS 384, so the masked pool already lands in the
  shared 384-d space; no extra motion projection is needed. Both embeddings are
  L2-normalized so InfoNCE operates on cosine similarity.

METADATA-ONLY INVARIANT (§1c-7)
-------------------------------
  `source_motion_id` / `object_type` (species) NEVER enter either tower. They are
  used solely to build the multi-positive false-negative mask and for grouping
  in the M3/M4 metrics. The ONLY text signal the model sees is the T5
  `caption_emb` of the selected caption view.

MULTI-POSITIVE FALSE-NEGATIVE MASK (§1c-8 / A2)
-----------------------------------------------
  AnyTop captions are highly duplicated (one val caption appears up to 56×).
  A vanilla InfoNCE treats every off-diagonal (text_i, motion_j) as a negative,
  which would punish a *semantically correct* same-caption / same-motion pair as
  if it were a mismatch. `build_multi_positive_mask` marks every off-diagonal
  pair (i, j), i != j, that shares ANY of:
      (same motion_id) ∪ (same source_motion_id) ∪ (same caption_text)
  as a false negative; `symmetric_infonce` then removes those entries from the
  denominator (logit -> -inf) while ALWAYS keeping the diagonal positive. This is
  the SAME grouping used by the group-aware R-precision / matching metrics, so
  training and evaluation agree on what "a positive" means.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encoder import SkeletonEncoder
from .batch import GraphMotionBatch


# A large finite negative used to drop masked-out logits from the InfoNCE
# denominator. -inf would make a fully-masked row produce NaN in log-softmax;
# a finite floor keeps softmax well-defined (the diagonal positive is always
# kept, so every row has at least one finite entry anyway).
_NEG_INF = -1e9


def build_multi_positive_mask(
    motion_id: list[str],
    source_motion_id: list[str],
    caption_text: list[str],
) -> torch.Tensor:
    """Build the [B, B] boolean false-negative mask for multi-positive InfoNCE.

    `mask[i, j] == True` iff (i, j) is an OFF-diagonal pair (i != j) that should
    NOT be treated as a negative, because items i and j share content under the
    grouping rule (§1c-8):
        (same motion_id) ∪ (same source_motion_id) ∪ (same caption_text).

    The diagonal is always `False` here (diagonal entries are the positives the
    loss keeps; they are handled by the loss, not by this mask). Returns a CPU
    bool tensor; the caller moves it to the logits' device.

    Args:
        motion_id:        per-sample motion id strings, length B.
        source_motion_id: per-sample canonical (species_gender@action) keys,
                          length B. METADATA ONLY — never fed to a tower.
        caption_text:     per-sample caption text for the active view, length B.

    Returns:
        mask: [B, B] bool. True = false-negative (mask out of the denominator).
    """
    if not (len(motion_id) == len(source_motion_id) == len(caption_text)):
        raise ValueError(
            f"build_multi_positive_mask: length mismatch "
            f"motion_id={len(motion_id)} source_motion_id={len(source_motion_id)} "
            f"caption_text={len(caption_text)}"
        )
    B = len(motion_id)

    def _eq_matrix(keys: list[str]) -> torch.Tensor:
        # [B, B] bool: same[i, j] = (keys[i] == keys[j]).
        same = torch.zeros(B, B, dtype=torch.bool)
        for i in range(B):
            ki = keys[i]
            for j in range(B):
                if keys[j] == ki:
                    same[i, j] = True
        return same

    shared = (
        _eq_matrix(motion_id)
        | _eq_matrix(source_motion_id)
        | _eq_matrix(caption_text)
    )
    # Exclude the diagonal — those are the kept positives, not false negatives.
    shared.fill_diagonal_(False)
    return shared


def symmetric_infonce(
    text_emb: torch.Tensor,      # [B, D] L2-normalized
    motion_emb: torch.Tensor,    # [B, D] L2-normalized
    logit_scale: torch.Tensor,   # scalar: 1 / temperature (already exp'd if learnable)
    false_neg_mask: Optional[torch.Tensor] = None,  # [B, B] bool, True = drop
) -> torch.Tensor:
    """Symmetric (text->motion + motion->text) multi-positive InfoNCE.

    Logits are `logit_scale * text_emb @ motion_emb.T`. The diagonal is the
    positive pair for each row/column. Off-diagonal entries flagged by
    `false_neg_mask` (shared content, §1c-8) are removed from BOTH the
    text->motion and the motion->text denominators (set to a large finite
    negative), so a duplicate caption / duplicate motion never acts as a hard
    negative against its twin. The diagonal is never masked.

    Returns the scalar mean of the two cross-entropy directions.
    """
    B = text_emb.shape[0]
    logits = logit_scale * text_emb @ motion_emb.t()  # [B, B], rows=text, cols=motion
    if false_neg_mask is not None:
        if false_neg_mask.shape != (B, B):
            raise ValueError(
                f"symmetric_infonce: false_neg_mask shape {tuple(false_neg_mask.shape)} "
                f"!= ({B}, {B})"
            )
        mask = false_neg_mask.to(logits.device)
        # Defensive: never mask the diagonal (the positive).
        mask = mask.clone()
        mask.fill_diagonal_(False)
        logits = logits.masked_fill(mask, _NEG_INF)
    labels = torch.arange(B, device=logits.device)
    # Row direction (text query -> matching motion) and column direction
    # (motion query -> matching text). logits.t() masks transpose correctly:
    # the mask is symmetric in construction, but transpose keeps it exact.
    loss_t2m = F.cross_entropy(logits, labels)
    loss_m2t = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_t2m + loss_m2t)


@dataclass
class EvaluatorOutput:
    """Container for a forward pass of `AnyTopT2MEvaluator`."""

    text_emb: torch.Tensor      # [B, D] L2-normalized
    motion_emb: torch.Tensor    # [B, D] L2-normalized
    logit_scale: torch.Tensor   # scalar effective 1/temperature


class AnyTopT2MEvaluator(nn.Module):
    """Two-tower text↔motion contrastive evaluator for AnyTop T2M metrics.

    Args:
        coemb_dim:   shared coembedding dim (= SkeletonEncoder d_model). 384.
        text_in_dim: T5 caption embedding dim. 768.
        n_heads / d_ff / n_graph_layers / n_temporal_layers: motion-tower
            SkeletonEncoder hyperparameters (defaults match the §1b/§5-M2 spec).
        motion_feat_dim: AnyTop motion channels (13). Fixed; the tower uses the
            same `motion_mode="anytop13_split"` + `attn_mode="graphormer"` path
            as `GraphMotionVAE(feat_mode="anytop13", attn_mode="graphormer")`.
        joint_feat_dim: static per-joint feature dim (9), as in the VAE encoder.
        temporal_kernel / dropout: SkeletonEncoder temporal conv kernel / dropout.
        learnable_temperature: if True, the InfoNCE temperature is a learnable
            log-scale (CLIP-style, clamped); else a fixed `temperature`.
        temperature: fixed softmax temperature when not learnable (default 0.07).
        max_logit_scale: clamp ceiling for the learnable log scale (CLIP uses
            ln(100)); ignored when temperature is fixed.
    """

    def __init__(
        self,
        *,
        coemb_dim: int = 384,
        text_in_dim: int = 768,
        n_heads: int = 8,
        d_ff: int = 1536,
        n_graph_layers: int = 4,
        n_temporal_layers: int = 2,
        motion_feat_dim: int = 13,
        joint_feat_dim: int = 9,
        temporal_kernel: int = 9,
        dropout: float = 0.1,
        learnable_temperature: bool = True,
        temperature: float = 0.07,
        max_logit_scale: float = 4.6051702,  # ln(100), CLIP cap
    ) -> None:
        super().__init__()
        if coemb_dim % n_heads != 0:
            raise ValueError(
                f"coemb_dim ({coemb_dim}) must be divisible by n_heads ({n_heads})"
            )
        if motion_feat_dim != 13:
            raise ValueError(
                f"AnyTopT2MEvaluator motion tower is AnyTop-13ch; "
                f"motion_feat_dim must be 13, got {motion_feat_dim}"
            )
        if not (0.0 < temperature):
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.coemb_dim = coemb_dim
        self.learnable_temperature = bool(learnable_temperature)
        self.max_logit_scale = float(max_logit_scale)

        # ---- Text tower: T5 [768] -> coemb_dim MLP (§1c-5). ----
        # One hidden layer (GELU) — a true MLP projection, not a bare Linear, so
        # the sentence-level T5 vector can be reshaped into the motion-aligned
        # coembedding space.
        self.text_proj = nn.Sequential(
            nn.Linear(text_in_dim, coemb_dim),
            nn.GELU(),
            nn.Linear(coemb_dim, coemb_dim),
        )

        # ---- Motion tower: freshly initialized SkeletonEncoder (§1c-1/3/4). ----
        # Same anytop13_split + graphormer path as GraphMotionVAE.encode, so the
        # motion is processed identically to training — but with INDEPENDENT
        # weights (never loaded from a VAE/denoiser ckpt).
        self.motion_encoder = SkeletonEncoder(
            d_model=coemb_dim,
            n_heads=n_heads,
            d_ff=d_ff,
            n_graph_layers=n_graph_layers,
            n_temporal_layers=n_temporal_layers,
            joint_feat_dim=joint_feat_dim,
            motion_feat_dim=motion_feat_dim,
            temporal_kernel=temporal_kernel,
            dropout=dropout,
            motion_mode="anytop13_split",
            attn_mode="graphormer",
        )

        # ---- Temperature / logit scale. ----
        if self.learnable_temperature:
            # CLIP-style learnable log scale, initialized to 1/temperature.
            init_log_scale = torch.log(torch.tensor(1.0 / temperature))
            self.log_logit_scale = nn.Parameter(init_log_scale)
        else:
            self.register_buffer(
                "fixed_logit_scale",
                torch.tensor(1.0 / temperature),
                persistent=True,
            )

    # ------------------------------------------------------------------ #
    # Towers
    # ------------------------------------------------------------------ #
    def encode_text(self, caption_emb: torch.Tensor) -> torch.Tensor:
        """T5 caption embedding [B, 768] -> L2-normalized text_emb [B, coemb_dim]."""
        x = self.text_proj(caption_emb)
        return F.normalize(x, dim=-1)

    def encode_motion(self, batch: GraphMotionBatch) -> torch.Tensor:
        """AnyTop motion -> L2-normalized motion_emb [B, coemb_dim].

        Mirrors `GraphMotionVAE.encode`'s encoder call EXACTLY (same tensors,
        same permute, same graphormer bias inputs), then masked-mean pools the
        per-(frame, joint) embeddings over the VALID region into a single
        [B, coemb_dim] vector.
        """
        if batch.anytop_x is None:
            raise ValueError(
                "AnyTopT2MEvaluator.encode_motion requires batch.anytop_x "
                "(use the AnyTop 13ch path / --dataset anytop_truebones)"
            )
        if batch.anytop_graph_dist is None or batch.anytop_joint_relations is None:
            raise ValueError(
                "AnyTopT2MEvaluator.encode_motion requires batch.anytop_graph_dist "
                "+ batch.anytop_joint_relations (graphormer attn bias)"
            )
        # [B, J, 13, T] -> [B, T, J, 13], identical to GraphMotionVAE.encode.
        motion_in = batch.anytop_x.permute(0, 3, 1, 2).contiguous()  # [B,T,J,13]
        h = self.motion_encoder(
            motion_in,                       # [B, T, J, 13]
            batch.skeleton_features,         # [B, J, 9]
            batch.adjacency,                 # [B, J, J]
            batch.geodesic_dist,             # [B, J, J]
            batch.joint_mask,                # [B, J]
            batch.frame_mask,                # [B, T]
            name_hashes=batch.name_hashes,   # [B, J]
            graph_dist=batch.anytop_graph_dist,            # [B, J, J]
            joint_relations=batch.anytop_joint_relations,  # [B, J, J]
        )  # [B, T, J, coemb_dim]
        pooled = self._masked_mean_pool(h, batch.frame_mask, batch.joint_mask)
        return F.normalize(pooled, dim=-1)

    @staticmethod
    def _masked_mean_pool(
        h: torch.Tensor,            # [B, T, J, D]
        frame_mask: torch.Tensor,   # [B, T] bool
        joint_mask: torch.Tensor,   # [B, J] bool
    ) -> torch.Tensor:
        """Mean over VALID (frame, joint) positions -> [B, D].

        `SkeletonEncoder` already zeros padded joints, and its temporal layers
        zero padded frames, but we still build the explicit (t, j) validity mask
        and divide by its count so the mean is over real positions only (never
        diluted by padding). Empty masks are guarded with a clamp(min=1) on the
        denominator (cannot happen for real batches — every sample has >= 1
        valid joint and frame — but keeps the op NaN-free).
        """
        B, T, J, D = h.shape
        # [B, T, J] validity = frame valid AND joint valid.
        valid = frame_mask[:, :, None] & joint_mask[:, None, :]   # [B, T, J]
        w = valid.to(h.dtype).unsqueeze(-1)                       # [B, T, J, 1]
        summed = (h * w).sum(dim=(1, 2))                          # [B, D]
        count = w.sum(dim=(1, 2)).clamp(min=1.0)                  # [B, 1]
        return summed / count

    @property
    def logit_scale(self) -> torch.Tensor:
        """Effective InfoNCE logit scale (= 1 / temperature).

        Learnable path: `exp(clamp(log_logit_scale))` (CLIP-style, clamped at
        `max_logit_scale` to avoid runaway sharpening). Fixed path: the constant
        buffer `1 / temperature`.
        """
        if self.learnable_temperature:
            return self.log_logit_scale.clamp(max=self.max_logit_scale).exp()
        return self.fixed_logit_scale

    # ------------------------------------------------------------------ #
    # Forward / loss
    # ------------------------------------------------------------------ #
    def forward(self, batch: GraphMotionBatch, caption_emb: torch.Tensor) -> EvaluatorOutput:
        """Encode both towers. `caption_emb` is the T5 [B, 768] of the active view.

        `caption_emb` is passed explicitly (not read off `batch`) because the
        eval dataset's caption view (full / species_stripped) is selected
        upstream; the typed `GraphMotionBatch` carries the generic `caption_emb`,
        but keeping it an explicit arg makes the view choice unambiguous at the
        call site.
        """
        text_emb = self.encode_text(caption_emb)
        motion_emb = self.encode_motion(batch)
        return EvaluatorOutput(
            text_emb=text_emb,
            motion_emb=motion_emb,
            logit_scale=self.logit_scale,
        )

    def contrastive_loss(
        self,
        out: EvaluatorOutput,
        false_neg_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Symmetric multi-positive InfoNCE on a forward output."""
        return symmetric_infonce(
            out.text_emb,
            out.motion_emb,
            out.logit_scale,
            false_neg_mask=false_neg_mask,
        )
