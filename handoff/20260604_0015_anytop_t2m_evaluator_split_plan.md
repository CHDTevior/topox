# AnyTopo T2M Evaluator Split Plan

Date: 2026-06-04

Scope: planning document only. No training code is changed by this document.

## 0. Current Repo State Verified

The active PlanetZoo L2 dataset now has materialized train/val split files:

```text
data/anytop_planet_zoo_clean_L2/splits/train.txt
data/anytop_planet_zoo_clean_L2/splits/val.txt
```

`AnyTopDataset` now prefers these files for `split="train"` / `split="val"` when both exist and `use_split_file=True`:

```text
src/data/anytop_dataset.py
  use_split_file default: True
  file mode: data_root/splits/{train,val}.txt
  fallback: original per-object md5-seeded stratified algorithm
```

Verified counts:

```text
train.txt entries: 77882
val.txt entries:   4112
total:             81994
motions on disk:   81994
overlap:           0
duplicates:        0
missing on disk:   0
uncovered on disk: 0
```

Instantiating the loader confirms it reads the files:

```text
AnyTopDataset [train]: 77882 motions, 473 object types
AnyTopDataset [val]:   4112 motions, 473 object types
```

Important consequence: for this data root, `val_frac` no longer determines the split unless `use_split_file=False` is explicitly passed. This is good for reproducibility because VAE and diffusion now share exactly the same motion holdout.

## 1. What HumanML3D / SALAD Did

EricGuo text-to-motion evaluator uses fixed dataset split files:

```text
HumanML3D/train.txt
HumanML3D/val.txt
HumanML3D/test.txt
```

Evaluator training reads `train.txt` and `val.txt`. Final generation evaluation uses the dataset test split.

SALAD follows the same protocol: it trains VAE / denoiser with the dataset train/val files and uses a frozen evaluator wrapper to compute FID, R-precision, matching score, diversity, and multimodality.

That protocol assumes one fixed human topology. It does not need a topology holdout.

## 2. No-Extra-Motion Constraint

The dataset has no spare motion pool. There are only 81994 usable clean-L2 motions, and the current 77882/4112 split already accounts for all of them.

So the evaluator plan must not carve out another large train/val/test split that removes additional actions from training. Instead:

- keep the existing `splits/train.txt` as the evaluator training pool
- keep the existing `splits/val.txt` as the evaluator validation / reporting pool
- derive stricter audit subsets inside the existing val split
- label what each subset does and does not prove

The current `splits/train.txt` and `splits/val.txt` are correct for VAE / diffusion training consistency:

- same data root
- same motion ids
- same train/val split for VAE and diffusion
- deterministic and inspectable

They are also the right starting point for evaluator training because they do not waste any extra motion.

The caveat is that this split is not a clean unseen-action or unseen-topology benchmark:

1. They are clip-level holdout within the same object/topology set.
2. All 473 object types appear in both train and val.
3. PlanetZoo contains many near-duplicate action templates across male/female or juvenile/adult variants.
4. Captions contain species and sex words, so a retrieval evaluator can cheat by matching object names.

Measured on the current split using caption JSON `source_motion_id` as the canonical action key:

```text
val motions:                         4112
val canonical action keys:           3987
val files whose action key in train:  3288
val files clean vs train action key:   824
template-overlap rate:               79.96%
```

Therefore the evaluator should not pretend the whole val split is an unseen-action benchmark. It should report the full val split and the clean/overlap subsets separately.

## 3. Keep One Main Split, Add Derived Eval Manifests

Use one main split system and add derived evaluator manifests.

### 3.1 Main split for VAE / diffusion / evaluator training

Keep the current files:

```text
data/anytop_planet_zoo_clean_L2/splits/train.txt
data/anytop_planet_zoo_clean_L2/splits/val.txt
```

Purpose:

- VAE reconstruction validation
- diffusion denoise validation
- stable training curves
- exact VAE/diffusion split alignment
- evaluator training / validation without throwing away additional motions

Do not change these for evaluator research unless intentionally re-running all training.

### 3.2 Derived evaluator manifests

Add derived manifest files:

```text
data/anytop_planet_zoo_clean_L2/eval_splits/
  train_main.json
  val_all.json
  val_action_clean.json
  val_action_overlap.json
  split_audit.json
```

Purpose:

- train the independent text-motion evaluator
- validate evaluator quality
- evaluate VAE recon and denoiser generation
- report full val and action-clean val metrics separately
- avoid wasting motions by creating another large holdout

Do not make `test_seen` / `test_unseen_topology` mandatory in v1. With no extra motions, those would only reduce evaluator training data or produce tiny unreliable subsets.

## 4. Recommended Evaluator Manifest Logic

### Step A: Read existing split files

Use the materialized split files as source of truth:

```text
train_main = splits/train.txt  # 77882
val_all    = splits/val.txt    # 4112
```

Do not regenerate a different evaluator train split.

### Step B: Build canonical action groups

For every motion file, derive:

```text
object_type
motion_id
source_file
source_motion_id
canonical_action_key
```

Preferred `canonical_action_key` source:

1. caption JSON field `source_motion_id`, if present
2. caption JSON field `source_file`, if present
3. fallback parse from filename: `maniset<hash>__<action_name>`

All files with the same `canonical_action_key` should stay in the same split. This prevents male/female or similar repeated clips from leaking across train/test.

### Step C: Derive val subsets

Use the train canonical action key set to partition val:

```text
val_action_clean:
  val motions whose canonical_action_key does NOT appear in train_main

val_action_overlap:
  val motions whose canonical_action_key DOES appear in train_main
```

Current expected counts:

```text
val_action_clean:    824
val_action_overlap: 3288
```

Interpretation:

- `val_all`: main validation metric, uses all available held-out clips.
- `val_action_clean`: stricter semantic retrieval sanity, smaller but less template-leaky.
- `val_action_overlap`: diagnostic, tells how much metric depends on repeated templates.

### Step D: Optional topology diagnostic

Do not make unseen-topology a main v1 metric unless we are willing to withhold object types from evaluator training.

If needed, add a small diagnostic only:

```text
val_object_fold_diagnostic.json
```

This would train a second evaluator variant with a small set of object types removed from evaluator training. It is useful for evaluator generalization analysis, but it should not be the default because it spends scarce training motion.

### Step E: Caption coverage and leakage checks

Fail loudly if:

- any motion lacks caption JSON
- any motion lacks T5 cache entry
- any `train_main` / `val_all` entry is missing from disk
- `train_main` and `val_all` overlap
- any duplicate entry exists in either split file
- `val_action_clean` is empty or too small for meaningful retrieval
- derived counts drift from the current audited counts without explanation

## 5. Evaluator Training Data

Train the evaluator only on:

```text
train_main.json
```

Validation:

```text
val_all.json
val_action_clean.json
val_action_overlap.json
```

Main reporting:

```text
val_all.json
val_action_clean.json
```

The evaluator must not train on generated samples.

## 6. Caption Views

The evaluator should support three caption views:

```text
full
species_stripped
action_only
```

`full` uses the original caption.

`species_stripped` removes animal/object/sex tokens where possible, e.g.:

```text
The female aardvark runs to a standstill, then turns left.
-> The animal runs to a standstill, then turns left.
```

`action_only` removes species identity even more aggressively:

```text
runs to a standstill, then turns left
```

Species-stripped and action-only sanity are required, not optional. If full-caption retrieval is high but species-stripped/action-only collapses, the evaluator is not a trustworthy motion semantic judge.

## 7. Model / Metric Plan

Use the earlier AnyTopo evaluator design, with these split-specific constraints:

- text encoder: T5 mean-pooled `[768] -> [512]`
- motion encoder: fine-joint graph-temporal encoder over `[B,T,J,13]`
- no VAE latent `z` as paper metric embedding
- no generator checkpoint weights
- train objective: symmetric batch-wide InfoNCE
- false-negative mask:
  - same `motion_id` is not a negative
  - same `canonical_action_key` is not a negative

Metrics:

```text
R-precision top1/top2/top3
matching score
FID
diversity
multimodality
```

Report every metric separately for:

```text
val_all
val_action_clean
val_action_overlap
full caption
species_stripped caption
action_only caption
```

## 8. VAE / Denoiser Evaluation Adapter

Real motions from `AnyTopDataset` arrive as:

```text
anytop_x [B,J,13,T]
```

Evaluator motion encoder should consume:

```text
motion [B,T,J,13]
```

So real motion needs:

```python
real_motion = batch.anytop_x.permute(0, 3, 1, 2).contiguous()
```

VAE / denoiser decoded motion already has:

```text
pred_motion [B,T,J,13]
frame_mask_recovered [B,T]
```

When comparing real vs pred, evaluate only over the common valid recovered frames. Do not use the original full `frame_mask` blindly on decoded VAE predictions, because latent temporal stride can mask incomplete trailing windows.

## 9. Implementation Order

### M0: evaluator manifest builder only

Implement:

```text
scripts/build_anytop_t2m_eval_splits.py
```

Outputs:

```text
data/anytop_planet_zoo_clean_L2/eval_splits/*.json
```

Must pass:

- no overlap
- no duplicate
- no missing caption/cache
- train/val files cover exactly all current motions
- `val_action_clean` and `val_action_overlap` are derived and counted
- no claim that `val_all` is an unseen-action benchmark

### M1: evaluator dataset adapter

Implement:

```text
src/data/anytop_t2m_eval_dataset.py
```

Must pass:

- loads all evaluator split manifests
- returns graph fields, masks, captions, canonical keys
- no default `AnyTopDataset` train/val split is used for official evaluator splits

### M2: evaluator model and tiny overfit

Implement:

```text
src/models/graph_salad/t2m_evaluator.py
scripts/train_anytop_t2m_evaluator.py
```

Must pass:

- tiny subset overfit
- real heldout retrieval above random
- shuffled caption retrieval near random
- species-stripped/action-only sanity remains meaningful

### M3: VAE reconstruction evaluator

Implement:

```text
scripts/eval_anytop_vae_t2m.py
```

Must pass:

- GT and VAE recon can both be embedded
- metrics and visual QA agree at least qualitatively

### M4: denoiser generation evaluator

Implement:

```text
scripts/eval_anytop_denoiser_t2m.py
```

Must pass:

- `val_all` / `val_action_clean` / `val_action_overlap` metrics; do not report unseen-topology as a v1 main metric
- full/stripped/action-only caption metrics
- visual QA GIFs saved for the same samples used in metric report

## 10. Agent Prompt

Use this prompt for implementation:

```text
Implement the AnyTopo T2M evaluator split + evaluator pipeline in noKslot_clean.

Do not modify current VAE or denoiser training behavior. The existing
data/anytop_planet_zoo_clean_L2/splits/train.txt and val.txt are the single source
of truth for VAE/diffusion/evaluator v1. They must remain untouched.

First create derived evaluator manifests under:
data/anytop_planet_zoo_clean_L2/eval_splits/

Do NOT carve out additional large test/unseen splits. There are no spare motions.
Use:
- train_main.json = current splits/train.txt
- val_all.json = current splits/val.txt
- val_action_clean.json = val motions whose canonical_action_key is absent from train_main
- val_action_overlap.json = val motions whose canonical_action_key is present in train_main

canonical_action_key should prefer caption JSON source_motion_id or source_file,
then fallback to filename maniset+action parsing.

Before model code, run split/cache preflight:
- caption JSON coverage 100%
- T5 cache coverage 100%
- no duplicate entries
- no train/eval overlap
- train_main + val_all cover all current motions exactly
- val_action_clean / val_action_overlap counts are reported
- val_all is labelled as held-out clip validation, not unseen-action validation

Then implement:
1. src/data/anytop_t2m_eval_dataset.py
2. src/models/graph_salad/t2m_evaluator.py
3. scripts/train_anytop_t2m_evaluator.py
4. src/metrics/anytop_t2m_metrics.py
5. scripts/eval_anytop_vae_t2m.py
6. scripts/eval_anytop_denoiser_t2m.py

Evaluator constraints:
- train only on real motions
- no VAE latent z as paper metric embedding
- no shared generator checkpoint weights
- motion encoder uses fine-joint graph-temporal [B,T,J,13]
- text encoder uses T5 [768] -> [512]
- symmetric batch-wide InfoNCE with false-negative mask for same motion_id and same canonical_action_key
- report val_all and val_action_clean separately
- full caption, species_stripped caption, and action_only caption sanity are required
- visual QA remains required for every major metric report

Stop after M0-M2 first and report exact commands/results before implementing
VAE/denoiser evaluator scripts.
```

## 11. Recommendation

Proceed with this plan.

The current materialized `splits/train.txt` / `val.txt` are good and should stay as the stable VAE/diffusion split.

For evaluator work, do not overload those files and do not create another large holdout. Add `eval_splits/*.json` so metrics can separately report:

- all held-out validation clips
- action-clean held-out validation clips
- action-overlap diagnostic clips
- text-motion semantics without species-name shortcut
