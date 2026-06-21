"""AnyTop T2M Evaluator (M2) — independent text↔motion contrastive evaluator.

Design doc: handoff/20260604_0121_anytop_t2m_evaluator_impl_proposal.md
  (§1c hard requirements + §5-M2) and
  handoff/20260604_0257_anytop_t2m_evaluator_impl_progress.md (M2 NEXT).

WHAT THIS IS
------------
A two-tower model (~108M total: ~42M trainable + a frozen DistilBERT backbone) that
maps a raw text caption and a motion clip into a *shared* 512-d coembedding space,
trained with symmetric, FALSE-NEGATIVE-MASKED (duplicate-aware) InfoNCE on REAL
training motions. Its frozen embeddings drive the project's T2M metrics
(group-aware R-precision / matching score / FID) for the AnyTop Graph-VQVAE +
Graph-CodeFlow generators — i.e. it is the *measuring instrument*, never part of
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

TWO TOWERS (shared coemb_dim=512 — NO motion_proj)  [VQ/CodeFlow revision]
--------------------------------------------------------------------------
  text tower:   raw caption str --frozen DistilBERT + masked-mean + MLP head--> text_emb [B,512]
                (primary; "t5_cache" T5-[768]->MLP is a fallback/ablation only)
  motion tower: SkeletonEncoder(d_model=512, n_graph=6, n_temporal=4, anytop13_split,
                graphormer) -> [B,T,J,512] --masked mean over valid (t,j)--> motion_emb [B,512]
  SkeletonEncoder's d_model IS 512, so the masked pool already lands in the
  shared 512-d space; no extra motion projection is needed. Both embeddings are
  L2-normalized so InfoNCE operates on cosine similarity.

METADATA-ONLY INVARIANT (§1c-7)
-------------------------------
  `source_motion_id` / `object_type` (species) NEVER enter either tower. They are
  used solely to build the false-negative mask and for grouping in the metrics.
  The ONLY text signal the model sees is the raw caption STRING (DistilBERT-encoded)
  of the selected caption view (or the T5 `caption_emb` in the t5_cache ablation).

DUPLICATE-AWARE FALSE-NEGATIVE MASK (§1c-8 / A2)
-----------------------------------------------
  AnyTop captions are highly duplicated (one val caption appears up to 56×).
  A vanilla InfoNCE treats every off-diagonal (text_i, motion_j) as a negative,
  which would punish a *semantically correct* same-caption / same-motion pair as
  if it were a mismatch. `build_false_negative_mask` marks every off-diagonal
  pair (i, j), i != j, that shares ANY of:
      (same motion_id) ∪ (same source_motion_id) ∪ (same caption_text)
  as a false negative; `symmetric_infonce` then removes those entries from the
  denominator (logit -> -inf) while ALWAYS keeping the diagonal positive. This is
  the SAME grouping used by the group-aware R-precision / matching metrics, so
  training and evaluation agree on what "a positive" means.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Union

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


def build_false_negative_mask(
    motion_id: list[str],
    source_motion_id: list[str],
    caption_text: list[str],
) -> torch.Tensor:
    """Build the [B, B] boolean false-negative mask for the false-negative-masked InfoNCE.

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
            f"build_false_negative_mask: length mismatch "
            f"motion_id={len(motion_id)} source_motion_id={len(source_motion_id)} "
            f"caption_text={len(caption_text)}"
        )
    B = len(motion_id)

    def _eq_matrix(keys: list[str]) -> torch.Tensor:
        # [B, B] bool: same[i, j] = (keys[i] == keys[j]). Vectorized via int-id
        # hashing (O(B) map + one O(B^2) tensor compare); the old Python double
        # loop is too slow at global DDP batch sizes (256/512).
        uniq: dict[str, int] = {}
        ids = [uniq.setdefault(k, len(uniq)) for k in keys]
        idt = torch.tensor(ids, dtype=torch.long)
        return idt[:, None] == idt[None, :]

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
    """Symmetric (text->motion + motion->text) FALSE-NEGATIVE-MASKED InfoNCE.

    NOTE ON NAMING (be honest): this is NOT strict multi-positive InfoNCE. The
    ONLY positive in the numerator is the DIAGONAL pair; off-diagonal entries
    flagged by `false_neg_mask` (duplicate motion_id / source_motion_id /
    caption_text) are merely REMOVED from the denominator (set to a large finite
    negative), NOT added to the numerator. So duplicates are "duplicate-aware"
    (never act as a hard negative against their twin) but are not rewarded as
    extra positives. Strict multi-positive would need a positive-logsumexp
    numerator; at global batch 256/512 the mask is sparse, so this masked-diagonal
    form keeps essentially all real negatives and is the standard HumanML3D-style choice.

    Logits are `logit_scale * text_emb @ motion_emb.T`. The diagonal is never masked.
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


class DistilBertTextTower(nn.Module):
    """Frozen DistilBERT raw-caption text tower + trainable MASKED-MEAN pooling +
    MLP head -> text_emb [B, coemb_dim]. PRIMARY evaluator text tower (spec §5).

    Loads LOCAL distilbert-base-uncased, FREEZE it (requires_grad=False; a
    train()-override pins it in eval so its dropout never reactivates while the
    head trains), takes last_hidden_state [B,L,768] + attention_mask, MASKED-MEAN
    pools over valid tokens -> [B,768], then a 2-layer MLP -> [B,coemb_dim].
    Only the MLP head is trainable; the DistilBERT backbone is never updated
    (independent-evaluator contract). Masked-mean (vs a learnable-query transformer
    pool) is used deliberately: the query-token pool collapsed to a near-constant
    embedding at init (text off-diag cosine 0.985 -> InfoNCE pinned at ln(B));
    masked-mean of distinct token sequences is input-dependent by construction and
    does not collapse. Spec §5 asks for a "projection/pooling head" without pinning
    the mechanism. (TMR / MotionCLIP-style text pooling.)
    """

    def __init__(
        self,
        model_path: str,
        coemb_dim: int = 512,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_length: int = 64,
    ) -> None:
        super().__init__()
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from transformers import AutoModel, AutoTokenizer
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.text_model = AutoModel.from_pretrained(model_path)
        self.text_model.eval()
        for p in self.text_model.parameters():
            p.requires_grad_(False)
        hidden = int(self.text_model.config.hidden_size)  # 768
        self.max_length = int(max_length)
        # Trainable MLP head over the masked-mean of DistilBERT token states.
        self.head = nn.Sequential(
            nn.Linear(hidden, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, coemb_dim),
        )

    def train(self, mode: bool = True):  # noqa: D401
        # Keep the FROZEN DistilBERT backbone in eval (dropout off) even when the
        # evaluator is .train()-ed — only the head trains.
        super().train(mode)
        self.text_model.eval()
        return self

    @property
    def device(self) -> torch.device:
        return self.head[0].weight.device

    def forward(self, captions: list[str]) -> torch.Tensor:
        """list[str] raw captions -> pooled text features [B, coemb_dim] (un-normalized)."""
        enc = self.tokenizer(
            list(captions), return_tensors="pt", padding=True,
            truncation=True, max_length=self.max_length,
        ).to(self.device)
        # Frozen backbone: no grad to DistilBERT; grad still flows to the head below.
        with torch.no_grad():
            hs = self.text_model(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
            ).last_hidden_state                              # [B, L, 768]
        mask = enc["attention_mask"].bool().unsqueeze(-1).to(hs.dtype)  # [B, L, 1]
        mean = (hs * mask).sum(1) / mask.sum(1).clamp(min=1.0)          # [B, 768] masked mean
        return self.head(mean)                                          # [B, coemb_dim]


class AnyTopT2MEvaluator(nn.Module):
    """Two-tower text↔motion contrastive evaluator for AnyTop T2M metrics.

    Args:
        coemb_dim:   shared coembedding dim (= SkeletonEncoder d_model). 512.
        text_in_dim: T5 caption embedding dim. 768.
        n_heads / d_ff / n_graph_layers / n_temporal_layers: motion-tower
            SkeletonEncoder hyperparameters (defaults match the §1b/§5-M2 spec).
        motion_feat_dim: motion channels fed to the tower. 13 = full AnyTop;
            12 = drop the trailing contact channel (ch12) — a per-joint-normalized
            near-binary channel whose std~0 makes it pollute the embedding (the
            contact-free tower is the clean motion-semantics metric). The tower uses
            the same `motion_mode="anytop13_split"` + `attn_mode="graphormer"` path
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
        coemb_dim: int = 512,
        text_tower: str = "distilbert",
        distilbert_path: str = "checkpoints/text_encoders/distilbert-base-uncased",
        text_max_length: int = 64,
        text_in_dim: int = 768,  # T5 fallback caption-emb dim only
        n_heads: int = 8,
        d_ff: int = 2048,
        n_graph_layers: int = 6,
        n_temporal_layers: int = 4,
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
        if motion_feat_dim not in (12, 13):
            raise ValueError(
                f"AnyTopT2MEvaluator motion tower is AnyTop-13ch; motion_feat_dim must be "
                f"13 (full) or 12 (drop the trailing contact channel ch12), got {motion_feat_dim}"
            )
        self.motion_feat_dim = int(motion_feat_dim)
        if not (0.0 < temperature):
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.coemb_dim = coemb_dim
        self.text_tower = text_tower
        self.learnable_temperature = bool(learnable_temperature)
        self.max_logit_scale = float(max_logit_scale)

        # ---- Text tower (spec §5). PRIMARY "distilbert" = frozen DistilBERT
        # raw-caption backbone + trainable learnable-query pooling head; fallback
        # "t5_cache" = the old T5 mean-pooled [768]->coemb MLP (ablation only). ----
        if text_tower == "distilbert":
            self.text_distilbert = DistilBertTextTower(
                distilbert_path, coemb_dim=coemb_dim, d_ff=d_ff,
                dropout=dropout, max_length=text_max_length,
            )
        elif text_tower == "t5_cache":
            self.text_proj = nn.Sequential(
                nn.Linear(text_in_dim, coemb_dim),
                nn.GELU(),
                nn.Linear(coemb_dim, coemb_dim),
            )
        else:
            raise ValueError(
                f"text_tower must be 'distilbert' or 't5_cache', got {text_tower!r}"
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
    def encode_text(self, text_input: Union[list[str], torch.Tensor]) -> torch.Tensor:
        """Raw captions (DistilBERT) or T5 emb (fallback) -> L2-normalized text_emb [B, coemb_dim].

        text_tower="distilbert": `text_input` is a list[str] of raw captions.
        text_tower="t5_cache":  `text_input` is the T5 caption_emb [B, 768].
        """
        if self.text_tower == "distilbert":
            x = self.text_distilbert(text_input)
        else:
            x = self.text_proj(text_input)
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
        # [B, J, 13, T] -> [B, T, J, 13], identical to GraphMotionVAE.encode. When
        # motion_feat_dim==12, drop the TRAILING contact channel (ch12) -> [B,T,J,12]
        # (kept channels: pos 0:3, rot6d 3:9, vel 9:12). Contact is per-joint-normalized
        # near-binary (std~0 -> any recon error blows up in normalized space and pollutes
        # the embedding), so the 12ch tower is the clean motion-semantics metric.
        if batch.anytop_x.shape[2] != 13:                         # slice assumes ch12==contact
            raise ValueError(f"encode_motion expects AnyTop 13ch anytop_x [B,J,13,T], "
                             f"got C={batch.anytop_x.shape[2]}")
        motion_in = batch.anytop_x.permute(0, 3, 1, 2)[..., :self.motion_feat_dim].contiguous()  # [B,T,J,motion_feat_dim]
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
    def forward(
        self,
        batch: GraphMotionBatch,
        text_input: Union[list[str], torch.Tensor],
    ) -> EvaluatorOutput:
        """Encode both towers. `text_input` is the active caption view:
        a list[str] of raw captions (DistilBERT primary), or the T5 caption_emb
        [B, 768] (t5_cache fallback). Passed explicitly (not read off `batch`) so
        the view choice (full / species_stripped) is unambiguous at the call site.
        """
        text_emb = self.encode_text(text_input)
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
        """Symmetric false-negative-masked InfoNCE on a forward output."""
        return symmetric_infonce(
            out.text_emb,
            out.motion_emb,
            out.logit_scale,
            false_neg_mask=false_neg_mask,
        )
