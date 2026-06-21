# L4-safe + HumanML3D VQVAE and T5 Cache Plan

Date: 2026-06-19

## 0. Goal

Train a new Graph-VQVAE tokenizer and build matching T5 caption caches for:

```text
data/animo4d_anytop_clean_L4_safe_plus_humanml3d
```

This is a new dataset root, separate from the earlier L4+truebones and L5 runs.
Do not reuse old T5 caches or old frozen tokenizers.

## 1. Dataset Facts

Audit files already written in the dataset root:

- `DATASET_INFO.md`
- `MERGE_SUMMARY.json`
- `DATASET_NUMERIC_AUDIT.json`
- `DATASET_POOL_STATS.json`

Key facts:

| item | value |
|---|---:|
| cond objects | 312 = 311 animal + 1 human |
| motions/train+val | 99,360 |
| train | 94,170 |
| val | 5,190 |
| human heldout | 4,388 |
| max joints | 102 |
| max joints setting | 144 |
| EdgeSegmentPool max natural coarse slots | 71 |
| recommended max_coarse | 72 |
| Human joints | 22 |
| Human coarse slots | 12 |
| total caption entries | 103,748 |
| total captions | 263,871 |
| avg captions / motion | 2.54 |

Dataset QA already passed:

- `cond.npy`: finite, no `std <= 0`, no mean/std explosion.
- motion scan: no NaN/Inf, no `abs >= 22.53`, no `abs >= 100`.
- train/val split: no overlap, no missing, no uncovered `motions/` files.
- caption JSON: readable, no empty captions, no overlong/NTNT corrupted captions after one Wisent caption filter.
- loader smoke: `AnyTopDataset(... max_joints=144, num_frames=300)` passed.

## 2. Important Design Choice

### VQVAE trains on 64-frame windows

Keep VQVAE tokenizer training at:

```text
max_frames = 64
temporal_stride = 4
T_lat = 16
```

This matches the existing tokenizer training path. Train split uses random temporal crop by default; val uses deterministic start=0. This is correct for tokenizer learning.

Do not confuse this with CodeFlow/backbone export. The later CodeFlow token export should use full-length or `num_frames=300`, depending on the downstream backbone setting.

### New tokenizer shape

For this dataset:

```text
max_joints = 144
max_coarse = 72
num_quantizers = 4
code_dim = 512
d_model = 512
```

Per 64-frame training clip, the dense RVQ index tensor shape is:

```text
[T_lat=16, C=72, Q=4]
```

Valid coarse slots are masked. Human clips only use 12 coarse slots; padded slots must remain masked.

## 3. T5 Caption Cache Plan

Use the new merged caption file:

```text
data/animo4d_anytop_clean_L4_safe_plus_humanml3d/motion_texts_by_file.json
```

Build this cache prefix:

```text
data/anytop_caption_t5_l4safe_human_multi
```

Required outputs:

```text
data/anytop_caption_t5_l4safe_human_multi.npz
data/anytop_caption_t5_l4safe_human_multi.embs.npy
data/anytop_caption_t5_l4safe_human_multi.keys.json
data/anytop_caption_t5_l4safe_human_multi.tokens.npy
data/anytop_caption_t5_l4safe_human_multi.token_mask.npy
```

Sequence:

1. Mean-pooled T5 cache:

```bash
python scripts/precompute_t5_captions.py \
  --texts_json data/animo4d_anytop_clean_L4_safe_plus_humanml3d/motion_texts_by_file.json \
  --out data/anytop_caption_t5_l4safe_human_multi.npz \
  --batch_size 64
```

2. Convert `.npz` to fast sidecar:

```bash
python scripts/convert_caption_npz_to_npy.py \
  --src data/anytop_caption_t5_l4safe_human_multi.npz
```

3. Token length preflight:

```bash
python scripts/precompute_t5_caption_tokens.py \
  --texts_json data/animo4d_anytop_clean_L4_safe_plus_humanml3d/motion_texts_by_file.json \
  --out_prefix data/anytop_caption_t5_l4safe_human_multi \
  --max_length 64 \
  --dtype fp16 \
  --batch_size 64 \
  --lengths_only
```

4. Token-level T5 cache:

```bash
python scripts/precompute_t5_caption_tokens.py \
  --texts_json data/animo4d_anytop_clean_L4_safe_plus_humanml3d/motion_texts_by_file.json \
  --out_prefix data/anytop_caption_t5_l4safe_human_multi \
  --max_length 64 \
  --dtype fp16 \
  --batch_size 64
```

Fail-fast requirements:

- Coverage must be 100% for all train/val entries.
- `keys.json` count must equal mean embedding rows and token rows.
- Token cache must align row-for-row with `.keys.json`.
- Any missing caption or zero embedding is a blocker.

The existing `scripts/preflight_t5_coverage.py` is older and defaults to truebones. Either extend it with `--data_root`, `--max_joints`, and `--num_frames`, or run a short custom `AnyTopDataset` coverage script against this new root.

## 4. VQVAE Training Plan

Recommended mainline: use the stronger codebook setting from the later L4safeTB run.

```text
num_codes = 8192
num_quantizers = 4
code_dim = 512
d_model = 512
d_ff = 1536
n_heads = 8
n_graph_layers = 4
n_enc_temporal_layers = 2
n_pre_vq_layers = 2
n_post_vq_layers = 2
n_cross_layers = 3
n_dec_temporal_layers = 2
temporal_stride = 4
max_frames = 64
max_joints = 144
max_coarse = 72
amp_dtype = bf16
epochs = 300
```

Loss weights: keep current `train_graph_vqvae.py` defaults:

```text
w_pos     = 1.0
w_rot     = 1.0
w_vel     = 1.0
w_contact = 0.1
w_world   = 0.25
w_fk      = 1.0
w_traj    = 0.10
w_commit  = 0.02
```

HumanML3D has a small accepted `gt_fk_mismatch` floor under the shared human skeleton. Keep `w_fk=1.0` for this first run, but explicitly monitor human reconstruction QA. If human batches visibly degrade or FK loss dominates, stop and report; do not silently change loss weights mid-run.

## 5. Batch / LR Recommendation

Evidence from existing L4safeTB VQVAE runs:

- n2048 run: `max_coarse=96`, `num_codes=2048`, `batch_size=32`, `lr=6.65e-5`, best val_total around `1.0478`.
- n8192 run: `max_coarse=96`, `num_codes=8192`, `batch_size=16`, `lr=6.65e-5`, best val_total around `0.8777`.

For this new dataset, use:

```text
base global batch = 64
base lr = 6.65e-5
```

Examples:

| hardware | per-GPU batch | global batch | lr |
|---|---:|---:|---:|
| 4 GPUs | 16 | 64 | 6.65e-5 |
| 6 GPUs | 10 or 11 | 60 or 66 | about 6.65e-5 |
| 8 GPUs | 8 | 64 | 6.65e-5 |

If the executor deliberately raises global batch, use linear LR scaling from this base and add a short warmup:

```text
lr = 6.65e-5 * global_batch / 64
warmup_steps = 2000 if global_batch changes substantially
```

But the preferred first run is the proven global-64 regime, not a throughput experiment.

Suggested run name:

```text
runs/vqvae_L4safeHuman_C72_J144_d512_Q4_n8192_b16g64_300ep_seed42
```

## 6. Smoke / Gate Order

Do not start the 300-epoch run until these pass:

1. VQVAE unit checks:

```bash
torchrun --standalone --nproc_per_node=1 scripts/train_graph_vqvae.py \
  --unit_checks \
  --out /tmp/vqvae_l4human_unit \
  --overwrite
```

2. Data loader smoke with the actual root:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_graph_vqvae.py \
  --anytop_root data/animo4d_anytop_clean_L4_safe_plus_humanml3d \
  --max_joints 144 \
  --max_coarse 72 \
  --max_frames 64 \
  --num_codes 8192 \
  --batch_size 2 \
  --lr 6.65e-5 \
  --amp_dtype bf16 \
  --smoke \
  --smoke_iters 4 \
  --out /tmp/vqvae_l4human_smoke \
  --overwrite
```

3. Confirm smoke logs:

- no pool overflow
- `z_q` / `indices` shapes reflect `C=72`, `Q=4`
- loss finite
- grad finite
- perplexity not collapsed to 1
- root drift/jitter QA finite
- val pass runs

4. Caption cache preflight:

- mean cache coverage 100%
- token cache coverage 100%
- train/val examples produce nonzero `caption_emb`, nonempty `caption_token_mask`

## 7. Training Launch Notes

You may reuse `scripts/_launch_graph_vqvae.sh` or the existing cross-alloc wrappers, but override these env vars explicitly:

```bash
ANYTOP_ROOT=data/animo4d_anytop_clean_L4_safe_plus_humanml3d
MAX_JOINTS=144
MAX_COARSE=72
MAX_FRAMES=64
NUM_CODES=8192
AMP_DTYPE=bf16
EPOCHS=300
LR=6.65e-5
```

If using a wrapper originally written for L4safeTB, do not forget that many wrappers default to:

```text
ANYTOP_ROOT=data/animo4d_anytop_clean_L4_safe_plus_truebones
MAX_COARSE=96
```

Those defaults are wrong for this run and must be overridden.

## 8. Visual QA

After early checkpoints and final checkpoint, render reconstruction GIFs with `animate_vqvae_recon*.py`.

Required categories:

- animal dense skeleton: `PZ_Giant_Anteater_Female` or `PZ_Giant_Anteater_Male`
- animal medium quadruped: `PZ_Siberian_Tiger_*` or `PZ_Snow_Leopard_*`
- animal swimming/long body if present in L4
- human locomotion
- human jump/kick/gesture

Visual checks:

- no frozen output
- no high-frequency jitter
- no root trajectory explosion
- limbs/fingers/toes are not expected because L4 has animal terminal detail but HumanML3D only has 22 joints
- compare RIC/recovered pose and FK route where possible

Metric alone is not sufficient. Visual QA is a gate.

## 9. Deliverables

Executor should return:

1. T5 cache paths and coverage report.
2. VQVAE run path.
3. Exact launch command / env.
4. Smoke logs summary.
5. Training epoch / best checkpoint summary.
6. Reconstruction GIF paths.
7. Any code changes, if made, and codex review result.

## 10. Out of Scope

Do not build CodeFlow token exports in this task.

Do not train CodeFlow in this task.

Do not alter the merged dataset contents unless a fail-fast preflight finds a concrete bug. If a bug is found, report it first.
