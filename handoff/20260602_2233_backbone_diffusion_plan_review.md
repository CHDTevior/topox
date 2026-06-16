# Review: Backbone Diffusion Plan with B rot6d-FK VAE

Date: 2026-06-02 22:33 BST.

Reviewed plan: `handoff/20260602_2220_backbone_diffusion_plan.md`

Scope: data path, VAE checkpoint compatibility, denoiser/training mechanics,
6-card launch design, and resource decision. No code was changed during this
review.

## Verdict

The main plan is sound:

```text
freeze B rot6d-FK VAE
use its encoder latents z [B,65,128,512]
train GraphSaladDenoiser from scratch on clean_L2 T2M captions
run on 6 H100s via same-node cross-allocation DDP
```

However, the launch/smoke adaptation has two hard issues that must be fixed
before starting real training:

1. Existing smoke logic would silently run as 1 GPU, so it would not validate
   the intended 6-card rendezvous/NCCL path.
2. Existing launch variables conflate global `WORLD_SIZE` with local
   `nproc_per_node`; a 6-card cross-allocation launcher must split those.

After those launcher fixes, the plan can proceed to codex review, true 6-rank
smoke, and then formal training.

## Confirmed Facts

### B VAE checkpoint is compatible with `train_denoiser.py`

Checkpoint:

```text
runs/m1_l2_anytop13_edgeseg_C128_rot6dfk_w025f100t010_g128_4card_seed42/best_model.pt
```

Actual loaded checkpoint metadata:

```text
epoch        = 79
val_loss     = 1.6218433943829795
val_recon    = 1.5049276157217317
pool_type    = edge_segment
feat_mode    = anytop13
attn_mode    = graphormer
decoder_mode = coarse_xattn
loss_mode    = anytop13_world_rot6d_fk
d_model      = 512
d_ff         = 1536
n_heads      = 8
max_coarse   = 128
max_joints   = 144
max_frames   = 64
temporal_stride = 4
val_frac     = 0.05
seed         = 42
use_text     = false
```

Direct `load_frozen_vae()` check passed:

```text
LOAD_OK edge_segment anytop13 coarse_xattn 512 128 144 4 anytop13_world_rot6d_fk
```

This is stronger evidence than merely saying the state dict has 192 keys, because
it verifies the actual Phase-2 loading path used by `scripts/train_denoiser.py`.

Relevant code:
- `scripts/train_denoiser.py:58-105` rebuilds and strict-loads the frozen VAE.
- `scripts/train_denoiser.py:545-553` uses `vae.encode(batch, sample=True)` and
  consumes `z`, pooled graph metadata, masks, and pooled skeleton embeddings.
- `scripts/train_denoiser.py:640-650` uses `vae.encode(batch, sample=False)` for
  deterministic validation.

### Caption cache and dataset are correct

Cache:

```text
data/anytop_caption_t5_cleanL2_multi.npz
data/anytop_caption_t5_cleanL2_multi.embs.npy
data/anytop_caption_t5_cleanL2_multi.keys.json
```

Verified:

```text
81994 motions
409970 caption embeddings
avg 5.0 captions / motion
```

The sidecar fast path exists and is used by the dataset:

```text
data/anytop_caption_t5_cleanL2_multi.embs.npy   1.2G
data/anytop_caption_t5_cleanL2_multi.keys.json  58M
```

This avoids slow per-key decompression from the 1.4G npz.

Relevant code:
- `src/data/anytop_dataset.py:652-707` loads caption embeddings and groups
  keys by `<motion_id>__cap<i>`.
- `src/data/anytop_dataset.py:974-990` samples random caption embeddings during
  train and uses index 0 for deterministic validation.
- `scripts/train_denoiser.py:414-434` performs fail-loud caption coverage and
  multi-caption preflight.

### Split alignment is credible

B VAE and planned diffusion both use:

```text
data_root = data/anytop_planet_zoo_clean_L2
val_frac  = 0.05
seed      = 42
```

`AnyTopDataset` uses a stable per-object split:

```text
object-specific md5 seed offset
sorted motion ids
shuffle with random.Random(seed + offset)
n_val=max(1, round(n * val_frac)) for n>=2
```

Relevant code:
- `src/data/anytop_dataset.py:471-487` constructor defaults include `seed=42`.
- `src/data/anytop_dataset.py:576-602` implements stable per-object train/val
  split.
- `scripts/train_denoiser.py:315-386` constructs train/val with
  `random_crop=False`, train `random_caption=True`, val `random_caption=False`.

No data leakage issue was found in the default split path.

### Denoiser training objective is as described

The current Phase-2 denoiser objective is:

```text
DDIMScheduler(prediction_type="v_prediction")
z_t = add_noise(z0, noise, timestep)
v_target = get_velocity(z0, noise, timestep)
loss = masked MSE(v_pred, v_target)
```

Relevant code:
- `scripts/train_denoiser.py:513-519` constructs the scheduler.
- `scripts/train_denoiser.py:565-586` builds `z_t`, `v_target`, and the loss.
- `scripts/train_denoiser.py:557-563` performs CFG text-drop by zeroing text
  embeddings when `has_text=False`.

This is not PRISM flow matching and not per-token timestep training. It is the
existing Graph-SALAD DDIM/v-pred setup.

## Hard Issues

### H1. Current smoke path would not validate 6-card training

Current script:

```text
scripts/_launch_diffusion_t2m.sh:53  NPROC="$WORLD_SIZE"
scripts/_launch_diffusion_t2m.sh:55  if [ "$SMOKE" = 1 ]; then
scripts/_launch_diffusion_t2m.sh:56      NPROC=1
```

For the existing 2-card script this was acceptable as a cheap single-GPU smoke.
For the new 6-card cross-allocation plan it is a hard issue.

If reused as-is, `SMOKE=1` would:

```text
run one local process
skip WORLD_SIZE=6 behavior
skip multi-allocation rendezvous validation
skip NCCL collective validation
not test real per-GPU batch under 6-rank DDP
```

Required fix:

```text
In the 6-card launcher, SMOKE must only add --smoke to train_denoiser.py.
It must not reduce nproc_per_node or nnodes.
```

Expected smoke gate:

```text
nnodes=3
nproc_per_node=2
WORLD_SIZE=6 in rank-0 train.log
NCCL initializes across all six ranks
one smoke epoch completes
loss finite
val finite
```

### H2. `WORLD_SIZE` must not be used as local process count

Current script uses:

```text
WORLD_SIZE="${WORLD_SIZE:-2}"
GLOBAL=$(( PER_GPU_BATCH * WORLD_SIZE ))
NPROC="$WORLD_SIZE"
torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC"
```

This is fine only for single-node/single-allocation use. For 3 allocations
with 2 GPUs each, it becomes ambiguous and dangerous.

Required fix:

```text
NNODES=3
NPROC_PER_NODE=2
GLOBAL=$(( PER_GPU_BATCH * NNODES * NPROC_PER_NODE ))
```

Each allocation should launch:

```text
torchrun \
  --nnodes=3 \
  --node_rank=<0|1|2> \
  --master_addr=swarmh1002-ib0 \
  --master_port=<shared_port> \
  --nproc_per_node=2 \
  scripts/train_denoiser.py ...
```

Do not set `NPROC_PER_NODE=6`; that would try to spawn six local ranks inside
each 2-GPU allocation.

### H3. Resource description should say same-node cross-allocation, not cross-node

Current available H100 jobs:

```text
944459  swarm_h10  RUNNING  swarmh1002  gres/gpu:2
944460  swarm_h10  RUNNING  swarmh1002  gres/gpu:2
944461  swarm_h10  RUNNING  swarmh1002  gres/gpu:2
```

All three allocations are on the same physical node, `swarmh1002`.

The correct wording is:

```text
same-node cross-allocation DDP
```

not:

```text
cross-node DDP
```

Static rendezvous is still needed because the three Slurm allocations are
separate. Reusing the previous `swarmh1002-ib0` + static rendezvous approach is
reasonable.

### H4. Old diffusion continuation is a weak use of 4 A100s

Current old-VAE diffusion references:

```text
runs/m2_t2m_cleanL2_ep34edgeseg_d512C128_h200x2_seed42/best_model.pt
  epoch=45, val_denoise=0.3685538057074499

runs/m2_t2m_cleanL2_cont_swarma1004/best_model.pt
  epoch=45, val_denoise=0.37236162535517203
```

The A100 continuation on swarma1004 has not beaten the original H200 run. It is
currently occupying 4 A100s and gives limited additional control value because
the better old-VAE checkpoint/log already exist.

Recommendation:

```text
stop the swarma1004 old-VAE continuation process
keep checkpoints and logs as baseline/control artifacts
use resources for the B-VAE diffusion run
```

Safe stop command for the user, preserving allocation:

```bash
ssh swarma1004 "pkill -f 'torchrun.*m2_t2m_cleanL2_cont_swarma1004' || true; pkill -f 'train_denoiser.py.*m2_t2m_cleanL2_cont_swarma1004' || true"
```

Do not `scancel` unless the user explicitly decides to release the allocation.

## Corrections to the Plan Text

### C1. Replace "state_dict strict load passed (192 keys)"

Better wording:

```text
`scripts.train_denoiser.load_frozen_vae()` strict rebuild/load passed for B ckpt.
```

Reason: this verifies the actual diffusion loading path, including constructor
args and strict state dict load.

### C2. Clarify decoder-agnostic wording

The statement "diffusion is decoder-agnostic" is basically correct, because
`train_denoiser.py` only calls `vae.encode()` and never calls `vae.decode()`.

But the B loss affects the whole VAE training, including the encoder. Better
wording:

```text
Diffusion does not call the VAE decoder, so decoder_mode differences do not
directly change denoiser training. However, the B rot6d-FK training objective
changes the learned encoder latent distribution, so this is a new diffusion
experiment on B latents, not a continuation of the old-VAE diffusion run.
```

### C3. Make smoke definition explicit

Required smoke wording:

```text
Smoke must be a true 6-rank run:
  nnodes=3
  nproc_per_node=2
  WORLD_SIZE=6
  same PER_GPU_BATCH as planned formal run
  --smoke passed to train_denoiser.py
```

Do not let smoke collapse to 1 GPU.

### C4. Split launch variables

Use:

```text
PER_GPU_BATCH
NNODES
NPROC_PER_NODE
GLOBAL_BATCH
LR
```

Avoid:

```text
WORLD_SIZE as both local nproc and global world size
```

## Recommended Hyperparameters

Start conservative:

```text
PER_GPU_BATCH = 16
NNODES = 3
NPROC_PER_NODE = 2
GLOBAL_BATCH = 96
LR = 1.0e-3
WARMUP_ITERS = 4000
EPOCHS = 500
```

Rationale:
- Existing A100 continuation used global 96 / lr 1e-3 and remained finite.
- B VAE latents are a new distribution; avoid jumping immediately to global 144
  / lr 1.5e-3.
- 6 H100s should handle batch 16 comfortably; smoke can test batch 24 later if
  we want speed.

Optional speed trial:

```text
PER_GPU_BATCH = 24
GLOBAL_BATCH = 144
LR = 1.5e-3
```

But do not make this the default first formal run unless smoke plus early val
look clean.

## Required Smoke Checklist

Before formal training:

1. B VAE checkpoint path is the one above.
2. `OUT` is a new clean directory, not old-VAE output.
3. Rank-0 log shows:

```text
world_size=6
pool_type=edge_segment
feat_mode=anytop13
decoder_mode=coarse_xattn
d_model=512
max_coarse=128
denoiser_max_frames=260
T_lat=65
```

4. Caption preflight reports 100% coverage and avg 5 captions/motion.
5. NCCL initializes across six ranks.
6. One smoke epoch completes.
7. Train loss is finite.
8. Val loss is finite.
9. Saved smoke checkpoint contains `vae_ckpt_args` from B, not old ep34 VAE.

## Final Recommendation

Proceed, but only after launcher fixes.

Approved direction:

```text
B rot6d-FK VAE ep79 -> frozen encoder latents -> new T2M diffusion run
```

Not approved as-is:

```text
reusing current _launch_diffusion_t2m.sh smoke behavior for 6-card validation
```

Implementation sequence:

1. Stop old swarma1004 old-VAE continuation if resources are needed.
2. Patch launch/orchestrator only:
   - set B VAE path;
   - set new OUT;
   - split `NNODES` and `NPROC_PER_NODE`;
   - keep smoke as true 6-rank DDP;
   - use static rendezvous over `swarmh1002-ib0`;
   - keep `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=1`, `NCCL_SOCKET_IFNAME=ib0`.
3. Codex review the launch changes.
4. Run true 6-card smoke.
5. If smoke passes, start formal training.
6. Do not judge only by `val_denoise`; render T2M samples for visual QA once
   useful checkpoints exist.

No model/data code change is required for this plan.
