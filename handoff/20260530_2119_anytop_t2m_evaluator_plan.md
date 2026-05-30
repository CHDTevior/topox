# AnyTopo T2M Evaluator Plan

Date: 2026-05-30
Scope: research and implementation plan only. No code changes are included in this document.

## 0. Goal

Current any-topology motion generation has visual QA and training losses, but no paper-grade T2M evaluation equivalent to HumanML3D/KIT workflows. The goal is to build an independent, frozen, graph-aware text-motion evaluator for AnyTopo/PlanetZoo so we can compute:

- FID in evaluator motion-embedding space
- R-precision top1/top2/top3
- matching score
- diversity
- multimodality for repeated sampling

The important point is not to directly reuse the HumanML3D evaluator checkpoint. The important point is to reuse the EricGuo/SALAD evaluation protocol while replacing the fixed-human motion encoder with an AnyTopo graph-temporal motion encoder.

## 1. Reference Code Reviewed

### 1.1 EricGuo text-to-motion

Repository snapshot: `/tmp/text-to-motion`, cloned from `https://github.com/EricGuo5513/text-to-motion`.

Key files:

- `/tmp/text-to-motion/train_tex_mot_match.py:15-30`
  - Builds the evaluator model family: `MovementConvEncoder`, `TextEncoderBiGRUCo`, `MotionEncoderBiGRUCo`.
- `/tmp/text-to-motion/train_tex_mot_match.py:52-73`
  - Hard-codes HumanML3D/KIT assumptions: `dim_pose=263` for T2M, `dim_pose=251` for KIT, fixed joint counts.
- `/tmp/text-to-motion/train_tex_mot_match.py:96-106`
  - Trains `TextMotionMatchTrainer` on text-motion batches.
- `/tmp/text-to-motion/networks/trainers.py:881-999`
  - `TextMotionMatchTrainer`, the core evaluator trainer.
- `/tmp/text-to-motion/networks/trainers.py:943-965`
  - Forward pass: motion is encoded by frozen movement encoder plus motion encoder; text is encoded by text encoder.
- `/tmp/text-to-motion/networks/trainers.py:968-981`
  - Positive text-motion pairs are pulled together, shifted negative pairs are pushed apart.
- `/tmp/text-to-motion/networks/modules.py:79-98`
  - `MovementConvEncoder`, temporal conv feature extractor over fixed human motion vectors.
- `/tmp/text-to-motion/networks/modules.py:311-350`
  - `TextEncoderBiGRUCo`.
- `/tmp/text-to-motion/networks/modules.py:353-386`
  - `MotionEncoderBiGRUCo`.
- `/tmp/text-to-motion/utils/metrics.py:37-56`
  - R-precision and matching score.
- `/tmp/text-to-motion/utils/metrics.py:60-70`
  - Activation mean/covariance for FID.
- `/tmp/text-to-motion/utils/metrics.py:73-92`
  - Diversity and multimodality.
- `/tmp/text-to-motion/utils/metrics.py:95-140`
  - Frechet distance.

What to borrow:

- Independent text-motion evaluator as frozen judge.
- Retrieval-based top-k metrics.
- FID over learned motion embeddings.
- Real-motion upper bound reporting.

What not to borrow directly:

- Fixed `dim_pose=263/251`.
- Fixed 22/21-joint human representation.
- GloVe/POS text pipeline, because our code already uses T5 caption embeddings.
- Single shifted negative as the only contrastive signal. For 80K PlanetZoo clips, batch-wide InfoNCE is stronger.

### 1.2 SALAD use of traditional evaluator

Key files:

- `outside_docs/SALAD/models/t2m_eval_wrapper.py:5-24`
  - Builds evaluator and loads `text_mot_match/model/finest.tar`.
- `outside_docs/SALAD/models/t2m_eval_wrapper.py:27-59`
  - `EvaluatorModelWrapper` freezes the loaded evaluator and sets fixed HumanML3D/KIT dimensions.
- `outside_docs/SALAD/models/t2m_eval_wrapper.py:61-94`
  - `get_co_embeddings()` and `get_motion_embeddings()` expose frozen text/motion embeddings.
- `outside_docs/SALAD/utils/eval_t2m.py:21-120`
  - VAE reconstruction evaluation using the frozen evaluator.
- `outside_docs/SALAD/utils/eval_t2m.py:240-367`
  - Denoiser generation evaluation: real embedding, generated embedding, FID, diversity, R-precision, matching.

What to borrow:

- Wrapper pattern: one frozen evaluator object with `get_co_embeddings()` and `get_motion_embeddings()`.
- Eval loop structure: collect real embeddings and generated embeddings, then compute metrics.
- Best-checkpoint reporting hooks, but only after our evaluator passes sanity checks.

What not to borrow directly:

- The pretrained human evaluator checkpoint.
- SALAD's assumption that all motion samples live in one fixed vector space with a single human skeleton.

### 1.3 Current noKslot_clean code

Dataset:

- `src/data/anytop_dataset.py:604-640`
  - Multi-caption JSON loading, with primary caption plus de-duplicated caption list.
- `src/data/anytop_dataset.py:642-687`
  - Multi-caption T5 cache loading. Keys are grouped as `<motion_id>__cap<i>`.
- `src/data/anytop_dataset.py:760-775`
  - Raw motion load and joint reorder to FK/BFS order.
- `src/data/anytop_dataset.py:791-800`
  - AnyTop 13ch normalization with per-object mean/std.
- `src/data/anytop_dataset.py:802-811`
  - Recovered world position and velocity view.
- `src/data/anytop_dataset.py:817-860`
  - Full-motion crop/pad and `frame_mask`.
- `src/data/anytop_dataset.py:866-878`
  - Padded adjacency, geodesic distance, rest offsets.
- `src/data/anytop_dataset.py:920-936`
  - Emits normalized `anytop_x`, AnyTop graph distance, joint relations, mean/std.
- `src/data/anytop_dataset.py:944-965`
  - Random-caption train path and deterministic primary-caption val path.
- `src/data/anytop_dataset.py:967-1014`
  - Return schema with graph fields, caption embedding, caption text, motion id, skeleton id.

VAE:

- `src/models/graph_salad/vae.py:347-360`
  - `encode()` returns latent and Phase-2 graph metadata.
- `src/models/graph_salad/vae.py:375-386`
  - `feat_mode=anytop13` consumes `anytop_x`.
- `src/models/graph_salad/vae.py:419-443`
  - Pool path exposes `pooled_adjacency`, `pooled_geodesic`, hard assignment, pooled skeleton embedding, anchors.
- `src/models/graph_salad/vae.py:500-516`
  - Return dict includes all metadata currently used by diffusion.
- `src/models/graph_salad/vae.py:518-579`
  - `encode_skeleton_only()` enables sampling-time graph metadata without a motion input.
- `src/models/graph_salad/vae.py:600-719`
  - Decode path returns `pred_motion [B,T,J,13]` and `frame_mask_recovered`.

Denoiser:

- `src/models/graph_salad/denoiser.py:1-38`
  - Graph-aware latent diffusion design and input contract.
- `src/models/graph_salad/denoiser.py:112-188`
  - Per-layer coarse-node spatial graph attention, temporal attention, text additive conditioning, FiLM, remask.
- `src/models/graph_salad/denoiser.py:272-430`
  - Forward contract with pooled adjacency/geodesic, coarse/frame masks, T5 text, pooled skeleton embeddings.

Training:

- `scripts/train_denoiser.py:300-313`
  - Denoiser can run full-motion `max_frames=260` while VAE was trained on 64-frame windows.
- `scripts/train_denoiser.py:315-371`
  - AnyTopDataset construction, full-motion mode, random caption train, deterministic val.
- `scripts/train_denoiser.py:377-417`
  - Preflight checks for max-frame coverage and multi-caption cache coverage.
- `scripts/train_denoiser.py:527-570`
  - Frozen VAE encode, diffusion v-pred target, graph-conditioned denoiser forward.
- `scripts/train_denoiser.py:607-670`
  - Current validation only computes masked v-MSE `val_denoise`. No T2M FID/top3 generation eval exists.

## 2. Key Design Decision

Build a new evaluator:

```text
caption / T5 [B,768]
    -> AnyTopoTextEncoder
    -> text_emb [B,512]

motion [B,T,J,13]
+ joint_mask [B,J]
+ frame_mask [B,T]
+ adjacency / geodesic / joint_relations
    -> AnyTopoGraphTemporalMotionEncoder
    -> motion_emb [B,512]

training objective:
    matched caption-motion pairs close
    mismatched pairs far

generation eval:
    generated motion -> frozen motion encoder
    real motion      -> frozen motion encoder
    caption          -> frozen text encoder
    metrics: FID, R-precision, matching, diversity, multimodality
```

This is analogous to EricGuo/SALAD, but the motion branch is topology-aware and can handle variable joint count.

## 3. Data Plan

### 3.1 Default dataset

Use:

```text
data/anytop_planet_zoo_clean_L2
```

Current checked state:

- `81994` motion files under `motions/`
- `473` object groups in `cond.npy`
- `51` quarantined risk files under `risk_files/`

This is the default active cleaned dataset because it removes high-risk velocity/normalization outliers and recomputes mean/std per object group.

### 3.2 Splits

Need explicit evaluator splits, not ad hoc train/val slicing.

Recommended split files:

```text
data/anytop_planet_zoo_clean_L2/eval_splits/
  train_seen.json
  val_seen.json
  test_seen.json
  val_unseen_topology.json
  test_unseen_topology.json
```

Definitions:

- `seen`: same object/topology groups may appear in train and eval, but motion clips are disjoint.
- `unseen_topology`: object/topology groups are held out entirely.

Why both:

- Seen split measures whether evaluator can capture text-motion semantics when topology is familiar.
- Unseen split measures the any-topology claim.
- If only seen split is reported, reviewers can argue the evaluator learned object-specific shortcuts.

### 3.3 Caption policy

Train:

- `random_caption=True`
- use all available captions per motion through the existing T5 cache grouping.

Val/test:

- deterministic primary caption for main metrics.
- optional repeated-caption evaluation to estimate caption variance.

Required caption checks:

- cache coverage: every evaluated motion should have at least one text embedding.
- multi-caption average: train split should have average captions per motion greater than 1.5 unless intentionally running single-caption mode.
- species-stripped sanity: make an alternate caption view that removes species/object names where possible. This catches a bad evaluator that retrieves by species name rather than motion semantics.

### 3.4 Motion length

Use full-motion mode:

- `max_frames=260`
- `random_crop=False`
- `frame_mask` controls valid frames.

Reason:

- We already found crop-caption mismatch is a real risk for full-clip captions.
- Evaluation should measure full action semantics, not a random local window.

## 4. Model Plan

### 4.1 Text branch

Input:

```text
caption_emb [B,768]
has_text [B]
```

Recommended v1:

```text
T5 mean-pooled embedding
  -> LayerNorm
  -> MLP 768 -> 1024 -> 512
  -> L2 normalize
```

Reason:

- This matches our current denoiser text path.
- It avoids adding a token-level T5 pipeline before the evaluator is validated.

Possible v2:

- token-level T5 cross attention.
- only do this if v1 evaluator fails because mean-pooled text is too weak.

### 4.2 Motion branch

Input:

```text
anytop_x [B,J,13,T] or permuted [B,T,J,13]
joint_mask [B,J]
frame_mask [B,T]
adjacency [B,J,J]
geodesic_dist [B,J,J]
anytop_graph_dist [B,J,J]
anytop_joint_relations [B,J,J]
skeleton_features [B,J,9]
rest_offsets [B,J,3]
```

Recommended v1:

```text
per-joint input projection:
  normalized anytop13 + skeleton/rest embedding -> D

N x GraphTemporalEvaluatorLayer:
  spatial graph attention over fine joints per frame
  temporal self-attention or BiGRU over frames per joint
  FFN
  mask after each layer

pooling:
  mask-aware temporal pooling
  mask-aware joint pooling
  optional CLS token

output:
  motion_emb [B,512]
  L2 normalize
```

Recommended dimensions:

- `d_model=384` or `512`
- `n_layers=4`
- `n_heads=8`
- output embedding `512`

Important anti-self-hype choice:

- Do not use the VAE latent `z` as the evaluator motion embedding in v1.
- Do not share evaluator weights with VAE/denoiser.
- It is acceptable to reuse code patterns such as graph attention modules, but checkpoint weights must be independent.

Why fine-joint graph rather than pooled latent:

- The evaluator is the judge. If it uses the same compression bottleneck as the generator, it can inherit the generator's blind spots.
- Fine-joint graph attention keeps wings, tails, faces, and long chains visible to the evaluator.

### 4.3 Optional fast baseline evaluator

Implement only for sanity, not as final metric:

```text
frozen VAE encoder mu -> temporal/slot pooling -> motion_emb
```

Use case:

- Debug pipeline quickly.

Do not use as paper metric:

- Too close to our generator stack.

## 5. Training Objective

Use batch-wide symmetric InfoNCE instead of EricGuo's single shifted negative.

```text
t = normalize(text_emb)
m = normalize(motion_emb)
logits = t @ m.T / temperature
labels = arange(B)

loss_t2m = CE(logits, labels)
loss_m2t = CE(logits.T, labels)
loss = 0.5 * (loss_t2m + loss_m2t)
```

Recommended additions:

- temperature learnable or fixed at `0.07`.
- balanced batches with multiple species/actions when possible.
- hard negatives:
  - same object, different motion
  - similar action, different object
  - same high-level locomotion word, different speed/phase if captions allow

Do not train evaluator on generated motions.

## 6. Metrics and Evaluation Protocol

### 6.1 Evaluator validation metrics

Before using this evaluator to score generators, it must pass:

- real text-motion R@1/R@2/R@3 on held-out seen split.
- real text-motion R@1/R@2/R@3 on held-out unseen-topology split.
- matching score: matched pairs closer than shuffled pairs.
- shuffled-caption sanity: R@3 should drop close to random.
- species-stripped sanity: R@3 should remain meaningfully above random.
- action-only sanity if possible.

### 6.2 Generator metrics

For VAE reconstruction:

- real motion embedding vs reconstructed motion embedding.
- FID, matching, top3.
- This is a bridge sanity before scoring denoiser.

For denoiser generation:

- Generate motion for each `(caption, target skeleton)` in eval split.
- Decode through frozen VAE to `pred_motion [B,T,J,13]`.
- Feed generated and real motion to frozen AnyTopo evaluator.
- Compute:
  - FID
  - top1/top2/top3
  - matching score
  - diversity
  - multimodality for repeated samples per condition

### 6.3 Real upper bound

Always report:

- `R_precision_real`
- `matching_score_real`
- `diversity_real`

Reason:

- SALAD does this, and it is essential for knowing how far generated samples are from real-data evaluator performance.

## 7. Anti-Self-Hype Gates

This is the most important part.

Gate A: evaluator independence

- evaluator trained separately from VAE/denoiser.
- evaluator weights are frozen during generation evaluation.
- evaluator never trains on generated samples.

Gate B: architecture separation

- evaluator v1 should use fine-joint graph-temporal motion encoding, not VAE pooled latent `z`.
- no shared checkpoint weights with generator.

Gate C: split discipline

- report seen-topology and unseen-topology metrics separately.
- unseen topology is required for the any-topology claim.

Gate D: negative controls

- shuffled captions should fail.
- random generated motions should fail.
- static/zero motion baseline should fail.
- nearest-neighbor retrieval baseline should be reported or at least audited.

Gate E: caption leakage control

- run species-stripped or action-only caption sanity.
- if full-caption metric is high but stripped-caption metric collapses, the evaluator is learning species names more than motion.

Gate F: visual QA remains primary for failure diagnosis

- metric passing does not replace GIF/render checks.
- every major metric result should have a small visual panel: GT, VAE recon, denoiser sample.

## 8. Implementation Plan

### Step 1: split builder

Add a script:

```text
scripts/build_anytop_t2m_eval_splits.py
```

Responsibilities:

- read `cond.npy` and motion file inventory.
- create seen and unseen-topology split manifests.
- record object group, species/action tokens, motion id, path, caption count.
- fail if any split has missing captions.

Outputs:

```text
data/anytop_planet_zoo_clean_L2/eval_splits/*.json
```

### Step 2: evaluator dataset wrapper

Add:

```text
src/data/anytop_t2m_eval_dataset.py
```

or reuse `AnyTopDataset` directly with a manifest filter.

Responsibilities:

- return all graph/motion fields required by evaluator.
- preserve full-motion `frame_mask`.
- expose caption text and T5 embedding.
- deterministic val/test caption selection.

### Step 3: evaluator model

Add:

```text
src/models/graph_salad/t2m_evaluator.py
```

Classes:

- `AnyTopoTextEncoder`
- `AnyTopoMotionEncoder`
- `AnyTopoT2MEvaluator`
- `AnyTopoT2MEvaluatorWrapper`

Wrapper API should mirror SALAD:

```python
get_co_embeddings(batch) -> (text_emb, motion_emb)
get_motion_embeddings(batch_or_motion) -> motion_emb
```

### Step 4: evaluator training script

Add:

```text
scripts/train_anytop_t2m_evaluator.py
```

Training:

- AdamW
- InfoNCE
- batch-wide negatives
- train on real data only
- save `latest.pt` and `finest.pt`
- best checkpoint by validation R@3 or matching score, not train loss

Logging:

- train loss
- val seen R@1/R@2/R@3
- val unseen R@1/R@2/R@3
- matching/shuffled matching
- species-stripped sanity if implemented

### Step 5: metric utilities

Add:

```text
src/metrics/anytop_t2m_metrics.py
```

Borrow formulas from:

- `/tmp/text-to-motion/utils/metrics.py:37-56`
- `/tmp/text-to-motion/utils/metrics.py:60-70`
- `/tmp/text-to-motion/utils/metrics.py:73-92`
- `/tmp/text-to-motion/utils/metrics.py:95-140`

Keep numerical behavior close to T2M/SALAD.

### Step 6: VAE reconstruction evaluator

Add:

```text
scripts/eval_anytop_vae_t2m.py
```

Purpose:

- first integration target before denoiser.
- feeds GT and VAE reconstruction through frozen evaluator.
- catches mask/shape/normalization bugs.

### Step 7: denoiser evaluator

Add:

```text
scripts/eval_anytop_denoiser_t2m.py
```

Purpose:

- load frozen VAE.
- load denoiser.
- load frozen AnyTopo evaluator.
- sample generated motion for eval conditions.
- compute FID/top3/matching/diversity/multimodality.
- save a small visual QA sheet/GIF set for the same samples.

## 9. Milestones and Decision Gates

| Milestone | Goal | Must Pass | Failure Means |
|---|---|---|---|
| M0 | split/cache sanity | no missing captions, no overlength truncation, clean masks | data pipeline not ready |
| M1 | evaluator overfit sanity | tiny subset reaches high retrieval | model/loss wiring bug if it cannot overfit |
| M2 | evaluator real validation | real heldout R@3 clearly above random, shuffled drops | evaluator not a usable judge |
| M3 | VAE recon eval | VAE recon close to GT in evaluator space and visuals align | metric or VAE representation mismatch |
| M4 | denoiser eval | FID/top3/matching computed on seen and unseen topology | final paper metric pipeline ready |

Do not use evaluator metrics as main claims until M2 and M3 pass.

## 10. Expected Risks

Risk: evaluator learns species names instead of motion.

Mitigation:

- species-stripped sanity.
- unseen topology split.
- hard negatives with same/similar species where possible.

Risk: evaluator is biased toward our architecture.

Mitigation:

- fine-joint graph encoder rather than VAE pooled latent.
- no shared weights.
- random/static/nearest-neighbor baselines.

Risk: captions are too weak or too templated.

Mitigation:

- report evaluator upper-bound.
- inspect failure cases.
- optionally group captions by action tokens and add action-balanced negatives.

Risk: FID improves while visual quality degrades.

Mitigation:

- keep visual QA as a required report artifact.
- never accept metric-only PASS.

Risk: variable topology makes batch negatives too easy.

Mitigation:

- include hard negatives from similar topology/action.
- report seen and unseen separately.

## 11. First Implementation Prompt

Use this prompt for the implementation agent:

```text
We need to implement an AnyTopo text-to-motion evaluator for /iridisfs/scratch/ts1v23/workspace/noKslot_clean, without changing the current VAE/denoiser training behavior.

Context:
- The old HumanML3D/T2M evaluator comes from EricGuo text-to-motion:
  - /tmp/text-to-motion/train_tex_mot_match.py:15-30 builds MovementConvEncoder + TextEncoderBiGRUCo + MotionEncoderBiGRUCo.
  - /tmp/text-to-motion/networks/trainers.py:881-999 trains TextMotionMatchTrainer.
  - /tmp/text-to-motion/networks/trainers.py:968-981 uses positive/negative contrastive text-motion pairs.
  - /tmp/text-to-motion/utils/metrics.py:37-140 defines R-precision, matching, diversity, multimodality, and FID.
- SALAD uses the frozen evaluator through:
  - outside_docs/SALAD/models/t2m_eval_wrapper.py:5-94
  - outside_docs/SALAD/utils/eval_t2m.py:21-120 and 240-367
- We cannot reuse the fixed HumanML3D evaluator checkpoint because our motion is arbitrary topology, variable J, AnyTop 13ch, with graph fields.

Current repo facts:
- AnyTopDataset already exposes multi-caption T5 cache and graph/motion fields:
  - src/data/anytop_dataset.py:604-687 for multi-caption and T5 cache.
  - src/data/anytop_dataset.py:760-811 for raw motion, normalized 13ch, recovered world pos/vel.
  - src/data/anytop_dataset.py:817-878 for crop/pad, masks, graph fields.
  - src/data/anytop_dataset.py:920-1014 for anytop_x, graph_dist, joint_relations, mean/std, caption_emb, has_text.
- VAE decode returns pred_motion [B,T,J,13]:
  - src/models/graph_salad/vae.py:600-719
- Denoiser currently validates only masked v-MSE:
  - scripts/train_denoiser.py:527-570 and 607-670

Task:
1. Build an implementation plan and then implement in small stages.
2. First add split/cache sanity tooling for data/anytop_planet_zoo_clean_L2:
   - seen topology split
   - unseen topology split
   - caption coverage checks
3. Implement a frozen-evaluator-style model:
   - AnyTopoTextEncoder: T5 [768] -> embedding [512]
   - AnyTopoMotionEncoder: graph-temporal fine-joint encoder over [B,T,J,13] plus masks and graph fields -> embedding [512]
   - AnyTopoT2MEvaluatorWrapper with get_co_embeddings() and get_motion_embeddings()
4. Train evaluator with symmetric batch-wide InfoNCE, not generated data.
5. Implement metric utilities matching EricGuo/SALAD formulas.
6. Add evaluator validation gates before using it on generators:
   - real text-motion retrieval above random
   - shuffled-caption retrieval drops
   - seen and unseen topology metrics reported separately
   - species-stripped/action-only sanity if feasible
7. Add VAE reconstruction eval before denoiser eval.
8. Add denoiser eval only after evaluator passes gates.

Hard constraints:
- Do not change current VAE or denoiser training behavior.
- Do not use VAE latent z as the paper evaluator embedding in v1.
- Do not share evaluator checkpoint weights with generator components.
- Do not train evaluator on generated samples.
- Preserve visual QA as a required artifact alongside metric reports.
- Keep all changes staged and reviewable; after implementation, run smoke tests and report exact commands/results.

Primary output expected before coding:
- a concrete file-by-file implementation checklist
- then code changes + smoke results in the order above
```

## 12. Recommendation

Proceed, but only if the first deliverable is evaluator validity, not generator score.

Minimum acceptable first milestone:

- split manifests exist.
- evaluator can overfit a tiny subset.
- heldout real text-motion R@3 is meaningfully above random.
- shuffled-caption R@3 drops.

Only after that should FID/top3 on denoiser samples be treated as meaningful.
