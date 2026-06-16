# AniMo4D AnyTop Clean L2 BF16 VAE Training Plan

## Goal

Train a new AnyTop 13-channel Graph-VAE on the cleaned AniMo4D AnyTop dataset.

Primary target:

```text
data/animo4d_anytop_clean_L2
```

Run length:

```text
300 epochs, BF16
```

This run is meant to produce the VAE checkpoint used by the next diffusion/backbone stage, so the priority is a stable, high-quality reconstruction VAE over the cleaned body-motion skeletons.

## Dataset Choice

Use **L2** as the primary training dataset.

Reason:

- L2 keeps the same 77894 motion clips and 311 objects as L1.
- L2 removes more non-body/detail controls: ears, mane, fur, feather, tuft, plume, dewlap, etc.
- L2 is closer to the main motion skeleton we care about for text-to-motion generation.
- L2 max joint count is 140, so `max_joints=144` covers the dataset cleanly.

Current dataset stats:

| Item | L2 |
|---|---:|
| Objects | 311 |
| Motions | 77894 |
| Total frames | 6179510 |
| Clean min joints | 60 |
| Clean max joints | 140 |
| Clean mean joints | 91.23 |
| Clean median joints | 92 |
| Nonfinite motion files | 0 |
| Max abs cleaned motion | 10.143964 |

Keep L1 as a later visual ablation only if L2 removes body parts we actually want to model. Do not start with L1.

## Data Split

Use the existing `AnyTopDataset` path:

```text
--dataset anytop_truebones
--anytop_root data/animo4d_anytop_clean_L2
```

The new clean dataset currently has no materialized `splits/train.txt` and `splits/val.txt`, so the loader will fall back to its deterministic per-object stratified split.

Use:

```text
--val_frac 0.05
```

This gives an approximate 19:1 train/val split per object, which is appropriate for 77894 clips.

Do not use `--full_data_val_species` for this run. That mode is for overlap-style reconstruction diagnostics, not the main large-data VAE.

## Model Configuration

Use the current proven AnyTop 13ch VAE path:

```text
feat_mode       = anytop13
attn_mode       = graphormer
pool_type       = edge_segment
decoder_mode    = coarse_xattn
amp_dtype       = bf16
```

Recommended architecture:

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
| `temporal_stride` | 4 |
| `max_frames` | 64 |
| `local_radius` | 8 |
| `use_name_embed` | true |

Rationale:

- `max_joints=144` covers L2 max J=140 with small padding.
- `max_coarse=128` preserves most topology detail while still using the edge-segment abstraction.
- `coarse_xattn` is the stable decoder mode; do not start with `graph_temporal` for this large first run.
- `use_name_embed` should stay on because this is a multi-species, multi-topology dataset.

## Loss

Use the current geometry-aware AnyTop13 VAE loss:

```text
loss_mode = anytop13_world_rot6d_fk
```

Recommended weights:

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

Interpretation:

- Base AnyTop13 loss still supervises normalized position, rot6d, velocity, contact, KL, and pool auxiliary terms.
- `world` supervises recovered world/RIC joint positions.
- `fk` supervises true rot6d-FK geometry, giving useful gradient to non-root rotation channels.
- `traj` supervises root trajectory consistency.

Do not use text conditioning for VAE training in this run. This is reconstruction pretraining; text belongs to the diffusion/backbone stage.

## Batch / LR Scaling

Use BF16 and scale LR by global batch.

Reference setting:

```text
global_batch = 384
lr = 8e-4
```

So:

| GPUs | Per-GPU batch | Global batch | LR |
|---:|---:|---:|---:|
| 8 | 48 | 384 | 8e-4 |
| 6 | 48 | 288 | 6e-4 |
| 4 | 48 | 192 | 4e-4 |
| 2 | 48 | 96 | 2e-4 |

If per-GPU batch 48 OOMs, reduce per-GPU batch and scale LR by the actual global batch:

```text
lr = 8e-4 * global_batch / 384
```

Keep the epoch count at 300. With 77894 motions, 300 epochs is a real large-data run, not a tiny overfit probe.

## Checkpoints / Logging

Use:

```text
--epochs 300
--save_every 5
--periodic_save_every 50
--seed 42
```

Expected important artifacts:

```text
runs/<run_name>/train.log
runs/<run_name>/metrics.jsonl
runs/<run_name>/diagnostics.jsonl
runs/<run_name>/best_model.pt
runs/<run_name>/best_recon_model.pt
runs/<run_name>/last_model.pt
runs/<run_name>/ep50_model.pt
runs/<run_name>/ep100_model.pt
runs/<run_name>/ep150_model.pt
runs/<run_name>/ep200_model.pt
runs/<run_name>/ep250_model.pt
runs/<run_name>/ep300_model.pt
```

Suggested run name:

```text
runs/m1_animo4dL2_bf16_rot6dfk_C128_d512_h8_ff1536_300ep_seed42
```

## Smoke Test Before Full Run

Before the 300-epoch run, execute a short smoke using the exact same config plus `--smoke`.

Smoke pass criteria:

- Dataset initializes from `data/animo4d_anytop_clean_L2`.
- Train and val split counts are nonzero.
- First batch gate reports `z=[B,16,128,512]`.
- Loss is finite.
- BF16 autocast is enabled.
- Backward and optimizer step complete.
- No NaN/Inf in losses or gradients.

Do not start the full run if the smoke fails.

## Visual QA

Metric is not enough for this project. Render reconstruction GIFs during or immediately after training.

Minimum QA set:

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

What to inspect:

- Static body skeleton remains coherent.
- GT vs recon motion speed is close.
- No frozen limbs.
- No new high-frequency jitter beyond raw-data artifacts.
- Long tails / spines / flippers / forelimbs remain temporally coherent.
- The known Caracal and Grey Seal raw clip discontinuities should not be counted as model-created artifacts unless recon makes them worse.

Render at least:

```text
best_recon_model.pt
last_model.pt
ep100_model.pt
ep200_model.pt
ep300_model.pt
```

## Success Criteria

This run is acceptable if:

- Training reaches 300 epochs without NaN/Inf.
- `val_recon` and visual recon improve over the first 50-100 epochs and do not collapse later.
- Reconstruction GIFs preserve body-scale motion energy.
- Large-J species are not visibly worse than medium-J species.
- The best reconstruction checkpoint is visually usable as the frozen VAE for Phase-2 diffusion/backbone training.

If the scalar best checkpoint and visual best checkpoint disagree, prefer the visual checkpoint for downstream diffusion.

## Do Not Do In This Run

- Do not train on L1 first.
- Do not use `graph_temporal` decoder first.
- Do not use `pool_type=none`.
- Do not enable VAE text conditioning.
- Do not use `full_data_val_species`.
- Do not change the cleaned dataset files during training.
- Do not judge the run from `val_recon` alone; render GIFs.

## Executor Prompt

Please launch a BF16 AnyTop13 Graph-VAE training run on:

```text
data/animo4d_anytop_clean_L2
```

Use the existing `scripts/train_graph_vae.py` AnyTop path with:

```text
dataset=anytop_truebones
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

Reference: per-GPU batch 48. If 8 GPUs are available, use global batch 384 and LR 8e-4. If fewer GPUs are used, keep per-GPU batch 48 if possible and scale LR by actual global batch.

First run a smoke test with the exact same config plus `--smoke`. Only start the 300-epoch run after smoke passes. During/after training, render GT-vs-recon GIFs for high-risk species and select the downstream VAE checkpoint by visual reconstruction quality, not scalar metric alone.
