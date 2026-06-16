# AniMo4D Clean L2 Proximal-Filtered BF16 VAE Training Plan

## Goal

Train the main AnyTop13 Graph-VAE on the **currently cleaned and proximal-rotation-filtered** AniMo4D L2 dataset.

Dataset root:

```text
data/animo4d_anytop_clean_L2
```

Training length:

```text
300 epochs
BF16
```

This replaces the earlier pre-filter L2 VAE plan. Use this document as the current source of truth.

## Current Dataset State

The dataset has already gone through two filtering stages:

1. L2 skeleton cleaning: removes face/detail controls plus ears/mane/fur/feather/decorative branches.
2. Proximal-rotation QC filtering on 2026-06-08: removes all severe + candidate + borderline proximal-limb rotation flagged clips.

Current L2 stats:

| Item | Value |
|---|---:|
| Objects | 311 |
| Motions | 74522 |
| Removed by proximal-rotation QC | 3372 |
| Total frames | 5942181 |
| Clean min joints | 60 |
| Clean max joints | 140 |
| Clean mean joints | 91.23 |
| Clean median joints | 92 |
| Feature dim | 13 |
| dtype | float32 |
| Nonfinite motion files | 0 |
| Max abs cleaned motion | 10.143964 |
| `cond.npy` mean/std | recomputed after filtering |
| `std` floor | 1e-6 |

Filtered clips are retained for audit under:

```text
data/animo4d_anytop_clean_L2/proximal_rotation_removed_20260608/
```

Do not train from that removed directory.

## Split

Use the existing `AnyTopDataset` loader path:

```text
--dataset anytop_truebones
--anytop_root data/animo4d_anytop_clean_L2
```

There is currently no materialized `splits/train.txt` / `splits/val.txt`, so the loader will use its deterministic per-object stratified fallback split.

Use:

```text
--val_frac 0.05
```

This gives roughly a 19:1 train/val split over the 74522 remaining clips.

Do **not** use `--full_data_val_species` for this main VAE run.

## Model

Keep the same architecture as the previous BF16 L2 plan.

```text
feat_mode       = anytop13
attn_mode       = graphormer
pool_type       = edge_segment
decoder_mode    = coarse_xattn
amp_dtype       = bf16
```

Recommended model settings:

| Setting | Value |
|---|---:|
| `d_model` | 512 |
| `d_ff` | 1536 |
| `n_heads` | 8 |
| `n_graph_layers` | 4 |
| `n_enc_temporal_layers` | 2 |
| `n_cross_layers` | 3 |
| `n_dec_temporal_layers` | 2 |
| `n_treeik_layers` | 3 |
| `max_joints` | 144 |
| `max_coarse` | 128 |
| `local_radius` | 8 |
| `temporal_stride` | 4 |
| `max_frames` | 64 |
| `use_name_embed` | true |

Rationale:

- L2 max J is 140, so `max_joints=144` covers all remaining skeletons.
- `edge_segment + max_coarse=128` is the current main pooling route.
- `coarse_xattn` is the stable decoder default; do not switch to `graph_temporal` for this first large AniMo4D VAE run.
- `use_name_embed` should stay enabled for multi-species transfer.

## Loss

Use the same geometry-aware AnyTop13 VAE loss:

```text
--loss_mode anytop13_world_rot6d_fk
```

Loss weights:

| Term | Weight |
|---|---:|
| `w_pos` | 1.0 |
| `w_rot` | 1.0 |
| `w_vel` | 1.0 |
| `w_contact` | 0.1 |
| `w_kl` | 1e-3 |
| `w_pool_aux` | 0.5 |
| `w_world` | 0.25 |
| `w_fk` | 1.0 |
| `w_traj` | 0.10 |

Meaning:

- The base AnyTop13 loss supervises normalized position, rot6d, velocity, contact, KL, and pool auxiliary terms.
- `world` supervises recovered world/RIC joint positions.
- `fk` supervises true rot6d-FK geometry and gives non-root rotation channels a geometry-level signal.
- `traj` supervises root trajectory consistency.

Do **not** enable VAE text conditioning for this run. Text conditioning belongs to the diffusion/backbone stage.

## BF16 Batch / LR

Use BF16 autocast:

```text
--amp_dtype bf16
```

Reference scaling:

```text
global_batch = 384
lr = 8e-4
```

Use linear scaling by actual global batch:

```text
lr = 8e-4 * global_batch / 384
```

Suggested settings:

| GPUs | Per-GPU batch | Global batch | LR |
|---:|---:|---:|---:|
| 8 | 48 | 384 | 8e-4 |
| 6 | 48 | 288 | 6e-4 |
| 4 | 48 | 192 | 4e-4 |
| 2 | 48 | 96 | 2e-4 |

If per-GPU batch 48 OOMs, reduce per-GPU batch and recompute LR from the actual global batch. Keep the 300-epoch target unchanged.

## Checkpoints

Use:

```text
--epochs 300
--save_every 5
--periodic_save_every 50
--seed 42
```

Suggested output directory:

```text
runs/m1_animo4dL2_proxfiltered_bf16_rot6dfk_C128_d512_h8_ff1536_300ep_seed42
```

Expected artifacts:

```text
train.log
metrics.jsonl
diagnostics.jsonl
best_model.pt
best_recon_model.pt
last_model.pt
ep50_model.pt
ep100_model.pt
ep150_model.pt
ep200_model.pt
ep250_model.pt
ep300_model.pt
```

## Preflight / Smoke

Before full training, run the exact same config with `--smoke`.

Smoke must verify:

- Dataset root is `data/animo4d_anytop_clean_L2`.
- Dataset count is based on 74522 remaining motions, not the old 77894 count.
- First train batch reports expected latent shape:

```text
z = [B, 16, 128, 512]
```

- BF16 autocast is enabled.
- Loss is finite.
- Backward pass and optimizer step complete.
- No NaN/Inf gradients.

If smoke fails, do not start the 300-epoch run.

## Visual QA

After training, render GT-vs-reconstruction GIFs. Do not judge this run from scalar metrics alone.

Minimum QA objects:

```text
PZ_Grey_Seal_Female
PZ_Caracal_Male
PZ_West_African_Lion_Male
PZ_Red_River_Hog_Male
PZ_Indian_Elephant_Male
PZ_B_W_Ruffed_Lemur_Male
PZ_Striped_Hyena_Male
PZ_Red_Kangaroo_Female
PZ_Fossa_Male
PZ_Tasmanian_Devil_Female
```

Inspect:

- GT vs recon motion energy.
- Long tails/spines/flippers/forelimbs.
- Frozen limbs.
- High-frequency jitter.
- Limb popping beyond what exists in the raw source.
- Large-J species versus medium-J species.

Render at least:

```text
best_recon_model.pt
last_model.pt
ep100_model.pt
ep200_model.pt
ep300_model.pt
```

If scalar best and visual best disagree, use the visually better checkpoint for the downstream diffusion VAE.

## Do Not Do

- Do not train on pre-filtered 77894-clip L2.
- Do not use `proximal_rotation_removed_20260608/` as training data.
- Do not enable `graph_temporal` for this first main run.
- Do not use `pool_type=none`.
- Do not enable VAE text conditioning.
- Do not use `full_data_val_species`.
- Do not judge by `val_recon` alone.

## Executor Prompt

Please train the BF16 AnyTop13 Graph-VAE on the currently filtered AniMo4D L2 dataset:

```text
data/animo4d_anytop_clean_L2
```

Important: this dataset has already removed 3372 proximal-rotation QC flagged clips and now contains 74522 motions. Use the current root as-is; do not train from the removed-file directory.

Use the existing `scripts/train_graph_vae.py` AnyTop path with:

```text
dataset=anytop_truebones
anytop_root=data/animo4d_anytop_clean_L2
feat_mode=anytop13
attn_mode=graphormer
pool_type=edge_segment
decoder_mode=coarse_xattn
loss_mode=anytop13_world_rot6d_fk
amp_dtype=bf16
max_joints=144
max_coarse=128
d_model=512
d_ff=1536
n_heads=8
n_graph_layers=4
n_enc_temporal_layers=2
n_cross_layers=3
n_dec_temporal_layers=2
n_treeik_layers=3
temporal_stride=4
local_radius=8
max_frames=64
use_name_embed=true
epochs=300
val_frac=0.05
seed=42
save_every=5
periodic_save_every=50
```

Use these loss weights:

```text
w_pos=1.0
w_rot=1.0
w_vel=1.0
w_contact=0.1
w_kl=1e-3
w_pool_aux=0.5
w_world=0.25
w_fk=1.0
w_traj=0.10
```

Use BF16 batch/LR scaling:

```text
lr = 8e-4 * global_batch / 384
```

Reference: per-GPU batch 48. If using 8 GPUs, use global batch 384 and LR 8e-4. If using fewer GPUs, keep per-GPU batch 48 if memory allows and scale LR by actual global batch.

First run a smoke test with the exact same config plus `--smoke`. Only start the 300-epoch run after smoke passes. After training, render GT-vs-recon GIFs for the listed QA species and pick the downstream checkpoint by visual reconstruction quality, not scalar metric alone.
