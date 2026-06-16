# AnyTopo T2M Evaluator Plan — VQ/CodeFlow Revision

Date: 2026-06-14
Updated: 2026-06-16

This document replaces the older diffusion-oriented evaluator wording with the
current Graph-VQVAE + Graph-CodeFlow route. The core evaluator idea is unchanged:
train an independent graph-aware text-motion matching evaluator, then freeze it
and use its embedding space for R-precision, matching, FID, diversity, and visual
QA reports.

## 1. What the older evaluator documents said

Relevant old documents:

- `handoff/20260530_2119_anytop_t2m_evaluator_plan.md`
- `handoff/20260604_0015_anytop_t2m_evaluator_split_plan.md`
- `handoff/20260604_0121_anytop_t2m_evaluator_impl_proposal.md`
- `handoff/20260604_0257_anytop_t2m_evaluator_impl_progress.md`

The old plan established these points:

1. We cannot reuse HumanML3D / KIT / AniMo evaluator checkpoints because those
   assume a fixed human or canonical animal topology.
2. We should reuse the T2M/SALAD/AniMo evaluation protocol conceptually:
   R-precision, matching score, FID, diversity, multimodality, and real-data
   upper bound.
3. The evaluator must be independent:
   - no generator weights;
   - no VAE latent as paper metric embedding;
   - no generated samples during evaluator training.
4. The evaluator motion encoder must be graph-aware and mask-aware, so it can
   consume arbitrary AnyTop skeletons with variable joint counts.
5. Captions are duplicated, so InfoNCE and R-precision must be multi-positive /
   duplicate-aware rather than strict diagonal-only matching.
6. Full caption is the main metric view; species-stripped caption is a sanity
   view against species-name shortcuts.

These points are still correct.

## 2. What changes because the main route is now Graph-VQVAE + CodeFlow

The old text still says "VAE / denoiser" in several places. In the current route
the generator stack is:

```text
real motion [B,T,J,13]
  -> Graph-VQVAE tokenizer
  -> RVQ indices / post-RVQ z_q
  -> Graph-CodeFlow backbone
  -> predicted continuous z_hat
  -> residual nearest snap to RVQ indices
  -> frozen Graph-VQVAE decode
  -> pred_motion [B,T,J,13]
```

So the metric adapters should be renamed conceptually:

- old `eval_anytop_vae_t2m.py`
  -> new `eval_anytop_vqvae_t2m.py`
- old `eval_anytop_denoiser_t2m.py`
  -> new `eval_graph_codeflow_t2m.py`

The evaluator itself should not become an RVQ evaluator. It should still judge
motions after decode, in fine-joint AnyTop 13-channel space.

## 3. Current dataset target

The older evaluator plan was written for:

```text
data/anytop_planet_zoo_clean_L2
```

The current VQ/CodeFlow route should target:

```text
data/animo4d_anytop_clean_L4_safe_plus_truebones
```

Dataset choice is intentional: use the merged dataset made from
`animo4d_anytop_clean_L4_safe` plus `anytop_truebones`. Do not switch this
evaluator plan to L5-only, L4-only, or truebones-only unless a separate ablation
is explicitly requested.

Current dataset facts from the dataset-local files:

```text
motions:      75592
objects:      381
train / val:  71784 / 3808
max T:        299 frames
max J:        142 joints
```

Recommended evaluator loader caps:

```text
num_frames = 300
max_joints = 144
```

Rationale:

- `num_frames=300` covers the current max length 299 without action truncation.
- `max_joints=144` covers max joint count 142 while matching the current
  training convention.

Do not copy the old L2 values `num_frames=260` or the old L2 split counts into
new evaluator reports.

## 4. Evaluator encoder: yes, it is the VAE-style encoder, but not VAE weights

The current implemented evaluator is:

```text
src/models/graph_salad/t2m_evaluator.py
```

Its motion tower is:

```text
AnyTop 13ch motion [B,J,13,T]
  -> permute to [B,T,J,13]
  -> SkeletonEncoder(d_model=384, motion_mode="anytop13_split", attn_mode="graphormer")
  -> h [B,T,J,384]
  -> masked mean over valid frames and joints
  -> motion_emb [B,384]
```

This is "VAE-style" because `GraphMotionVAE.encode()` also uses
`SkeletonEncoder` with the AnyTop13 + graphormer path. But it is not the trained
VAE encoder:

- The evaluator creates a fresh `SkeletonEncoder`.
- It does not load VAE / VQVAE / CodeFlow checkpoints.
- It is trained only with real text-motion matching.
- It never consumes `z`, `z_q`, or RVQ `indices` as paper metric embedding.

This separation is important. The evaluator is the measuring instrument. If it
used the same compressed RVQ latent as the generator, it could inherit the
generator's blind spots and miss failures in wings, tails, long chains, or
fine-end structures.

## 5. Text encoder

Updated decision: use raw-caption DistilBERT as the primary evaluator text tower,
not the generator-side T5 caption cache.

Reason:

- MotionMillion also separates generation text conditioning from evaluator text
  matching. Their generator can use Flan-T5-XL/XXL, while their evaluator wrapper
  loads an independent DistilBERT-based text encoder.
- Using DistilBERT raw captions makes our evaluator less coupled to the current
  Graph-CodeFlow text-conditioning path, improving metric fairness.
- We still keep the evaluator independent: no generator weights, no VQVAE/RVQ
  latents, and no generated samples during evaluator training.

Primary evaluator text tower:

```text
raw caption strings
  -> frozen DistilBERT tokenizer + encoder
  -> token hidden states [B,L,768] + attention_mask [B,L]
  -> evaluator trainable projection / pooling head
  -> text_emb [B,512]
```

Local pretrained text backbone:

```text
checkpoints/text_encoders/distilbert-base-uncased
```

Local MotionMillion reference code:

```text
outside_docs/MotionMillion-Codes/models/evaluator_wrapper_motionmillion_rpr272.py
outside_docs/MotionMillion-Codes/mld/models/architectures/temos/textencoder/distillbert.py
outside_docs/MotionMillion-Codes/mld/models/architectures/temos/textencoder/distillbert_actor.py
outside_docs/MotionMillion-Codes/mld/models/architectures/temos/motionencoder/actor.py
outside_docs/MotionMillion-Codes/dataset/dataset_TM_eval_motionmillion.py
outside_docs/MotionMillion-Codes/utils/eval_trans.py
```

Local MotionMillion reference evaluator checkpoint:

```text
outside_docs/MotionMillion-Codes/checkpoints/evaluator/epoch=199.ckpt
```

This checkpoint is downloaded for inspection/reproducibility of the reference
paper. It is not usable as our AnyTopo evaluator checkpoint because its motion
tower is fixed-topology human `nfeats=272`.

Relevant MotionMillion behavior:

```text
DistilbertActorAgnosticEncoder("distilbert-base-uncased", num_layers=4, latent_dim=512)
ActorAgnosticEncoder(nfeats=272, vae=True, num_layers=4, max_len=300, latent_dim=512)
checkpoint: checkpoints/evaluator/epoch=199.ckpt
```

We should not copy their fixed-topology motion tower. It assumes fixed human
motion vectors (`nfeats=272`) and `max_len=300`. We only borrow the evaluator
separation principle and the DistilBERT raw-caption text tower idea. Our motion
tower remains AnyTop graph-aware.

Former T5-cache text tower:

```text
caption_emb [B,768]  # T5 mean-pooled caption embedding
  -> MLP
  -> text_emb [B,512]
```

This is now a fallback / ablation path, not the primary evaluator plan.

Species / object name / source motion id are metadata only:

- used for grouping;
- used for multi-positive masks;
- used for reporting;
- not directly fed as separate conditioning channels.

Caption quality note for the merged L4-safe + truebones dataset:

- A hard-corruption audit found only 5 truly broken caption strings: one
  repeated `NTNT...` text and four one-letter placeholders `A/B/C/D`.
- These are too few to justify reworking the dataset. If all-captions random
  evaluator training is used, blacklist only those 5 caption entries during
  evaluator manifest/cache selection.
- The larger `gimmick` / `take 001` group is not JSON corruption. It is weak or
  game-action-style wording, but can remain unless a later caption-cleaning pass
  is explicitly requested.

Main text view:

```text
full caption is the primary evaluator training/eval view.
```

The old `species_stripped` plan should be downgraded to a sanity-only view for
the current merged dataset. The current human-filtered caption distribution has
very low natural animal-level coverage, so species-stripped is not reliable as
a primary metric view unless we later generate a separate rule-rewritten T5
cache.

## 6. Evaluator training objective

Training data:

```text
real motions only
```

No generated motions should appear in evaluator training.

Loss:

```text
text_emb   = normalize(text_encoder(caption))
motion_emb = normalize(motion_encoder(motion, graph, masks))
logits     = text_emb @ motion_emb.T * logit_scale

loss = symmetric InfoNCE(text -> motion, motion -> text)
```

The denominator masks false negatives. Off-diagonal pairs are not treated as
ordinary negatives if they share any of:

```text
same motion_id
same source_motion_id
same caption_text under the active view
```

This is needed because animal captions have many duplicated or near-duplicated
texts.

Important implementation requirement:

```text
InfoNCE must be computed over the GLOBAL DDP batch, not only per-rank batches.
```

For a serious evaluator run, gather `text_emb`, `motion_emb`, and the metadata
lists across ranks before computing logits and the multi-positive mask. A small
per-rank batch with local-only negatives is too weak and makes the evaluator
easier to overfit to shallow species/gender cues.

The evaluator should continue to use:

```text
loss = symmetric multi-positive InfoNCE
temperature = learnable CLIP-style logit scale
```

and should not add generator-specific losses, RVQ losses, or decoded-geometry
losses. This model is a measuring instrument, not a generator component.

## 6.1 Recommended Training Configuration

Recommended first full evaluator run on
`data/animo4d_anytop_clean_L4_safe_plus_truebones`:

```text
data_root      = data/animo4d_anytop_clean_L4_safe_plus_truebones
num_frames     = 300
max_joints     = 144
caption view   = full
motion input   = decoded/fine AnyTop 13ch motion [B,T,J,13]
text input     = raw caption string -> frozen DistilBERT [B,L,768]
```

Model:

```text
coemb_dim          = 512
text_backbone      = checkpoints/text_encoders/distilbert-base-uncased
text_backbone_grad = frozen
n_heads            = 8
d_ff               = 2048
n_graph_layers     = 6
n_temporal_layers  = 4
dropout            = 0.1
temperature        = learnable, init 0.07
```

Optimizer:

```text
optimizer       = AdamW
global_batch    = 256 minimum, 512 preferred if memory/throughput allow
lr              = 2e-4 for global_batch 256
lr              = 3e-4 to 4e-4 for global_batch 512, after smoke
warmup_steps    = 2000
schedule        = cosine decay
weight_decay    = 1e-6 to 1e-4
grad_clip       = 1.0
precision       = bf16 encoder forward, fp32 logits / cross entropy
```

This is intentionally larger than the old ~14M smoke evaluator. The merged
dataset has ~72k train motions, so a 512-d graph-aware evaluator is justified.
The goal is not a tiny smoke model; the goal is a reliable frozen metric model.

## 6.2 Required Evaluator Validity Gates

Before using evaluator scores as paper/project evidence, require:

1. Tiny overfit gate:
   a tiny subset should reach near-perfect retrieval. If it cannot overfit, the
   model/data wiring is wrong.
2. Held-out retrieval gate:
   `val_all` R@1/R@2/R@3 and matching score must be clearly above random.
3. Shuffled-caption gate:
   randomly permuting captions should sharply reduce retrieval quality.
4. Within-species retrieval gate:
   report retrieval where the candidate pool is restricted to the same
   species/object or same species+gender group. This is the main guard against
   the evaluator winning by species-name shortcuts.
5. Source split gate:
   report AniMo4D and truebones subsets separately. Truebones is small but
   important because it contains motion/topology styles underrepresented in
   AniMo4D.
6. VQVAE reconstruction gate:
   frozen VQVAE reconstructions should score close to GT in evaluator embedding
   space before evaluating CodeFlow generations.
7. Visual QA gate:
   evaluator metrics do not replace rendered GT-vs-pred GIF/video checks.

## 7. Metrics after evaluator is trained

Once the evaluator passes validity gates, freeze it and compute:

- group-aware R-precision top1/top2/top3;
- multi-positive matching score;
- FID in evaluator motion-embedding space;
- diversity;
- multimodality if repeated samples per condition are available;
- real-data upper bound;
- shuffled-caption sanity;
- species-stripped sanity if a usable species-stripped view/cache exists;
- within-species retrieval sanity;
- per-source reports for AniMo4D vs truebones;
- visual QA GIFs or videos.

Metric priority:

```text
R-precision / matching + visual QA  >  FID alone
```

FID is useful, but in a multi-topology, multi-species distribution it should not
be the only decision metric.

## 8. VQ-VAE reconstruction evaluation

Script to add:

```text
scripts/eval_anytop_vqvae_t2m.py
```

Flow:

```text
GT batch from AnyTopDataset
  -> frozen Graph-VQVAE tokenizer forward
  -> pred_motion [B,T,J,13]

GT motion and pred_motion
  -> frozen AnyTopo T2M evaluator
  -> motion embeddings / text embeddings
  -> reconstruction-space evaluator metrics
  -> visual QA
```

Purpose:

- verify evaluator integration before evaluating generation;
- check whether VQVAE reconstruction is close to GT in evaluator space;
- catch mask, length, normalization, and graph-field bugs.

## 9. Graph-CodeFlow generation evaluation

Script to add:

```text
scripts/eval_graph_codeflow_t2m.py
```

Flow:

```text
target skeleton + caption
  -> tokenizer.prepare_skeleton_only(...)
  -> GraphCodeFlow.sample(...)          # ODE + CFG -> continuous z_hat
  -> tokenizer.nearest_residual_ids(...)# residual nearest snap
  -> tokenizer.decode_from_indices(...) # frozen VQVAE decode
  -> pred_motion [B,T,J,13]

pred_motion + GT/text
  -> frozen AnyTopo T2M evaluator
  -> R-precision / matching / FID / diversity
  -> visual QA
```

Important: CodeFlow internal diagnostics such as `projection_error`,
`code_usage`, or `continuous-vs-snapped` are still useful, but they are not a
replacement for evaluator metrics on decoded motion.

## 10. Updated implementation checklist

M0. Rebuild evaluator manifests for:

```text
data/animo4d_anytop_clean_L4_safe_plus_truebones/eval_splits/
```

Expected manifest files:

```text
train_main.json
val_all.json
val_action_clean.json
val_action_overlap.json
val_animo4d.json
val_truebones.json
split_audit.json
```

M1. Update `AnyTopT2MEvalDataset` defaults or launch args:

```text
data_root=data/animo4d_anytop_clean_L4_safe_plus_truebones
num_frames=300
max_joints=144
```

This `data_root` is the L4-safe + truebones merged dataset. It is not the L5
dataset used in some tokenizer/backbone experiments.

M2. Replace the primary text path with raw-caption DistilBERT:

- dataset returns raw caption strings and caption metadata;
- text tower loads local `checkpoints/text_encoders/distilbert-base-uncased`;
- DistilBERT backbone is frozen by default;
- train only the evaluator projection/pooling head;
- T5 mean-pooled cache can remain as an optional ablation, but not the primary
  metric model.

M3. Implement/verify DDP global-batch InfoNCE:

- all-gather text embeddings;
- all-gather motion embeddings;
- gather metadata strings for the multi-positive mask;
- compute one global contrastive loss per step.

M4. Train the independent evaluator on real train motions only.

- primary run: full caption view;
- blacklist only the 5 hard-corrupt caption entries if using random
  all-caption sampling;
- do not feed species/object metadata into either tower.

M5. Validate evaluator itself:

- tiny overfit;
- held-out real retrieval above random;
- shuffled-caption drops;
- within-species retrieval sanity;
- truebones and AniMo4D subsets reported separately;
- species-stripped sanity only if an adequate rewritten/cache-backed view exists.

M6. Add VQ-VAE reconstruction eval.

M7. Add Graph-CodeFlow generation eval.

Do not use evaluator scores as a paper claim until the validation gates and
VQ-VAE reconstruction eval pass.

## 11. Current code pointers

Implemented evaluator model:

- `src/models/graph_salad/t2m_evaluator.py`

Evaluator dataset wrapper:

- `src/data/anytop_t2m_eval_dataset.py`

Evaluator training script:

- `scripts/train_anytop_t2m_evaluator.py`

Smoke:

- `scripts/_smoke_anytop_t2m_evaluator.py`

Generator-side current route:

- `src/models/vq_model/graph_vq_tokenizer.py`
- `src/models/CodeFlow_Model/flow.py`
- `src/models/CodeFlow_Model/graph_pscf.py`
- `scripts/train_graph_codeflow.py`
- `scripts/animate_graph_codeflow.py`
