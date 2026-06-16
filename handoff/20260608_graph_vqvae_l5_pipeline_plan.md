# Graph-Aware VQ-VAE Pipeline Plan for AniMo4D AnyTop L5

Date: 2026-06-08

## Goal

Build a **separate** VQ tokenizer / VQ-VAE pipeline for
`data/animo4d_anytop_clean_L5`, inspired by MoGenTS-style joint-structured
VQ-VAE, but adapted to our arbitrary-topology AnyTop setting.

Hard boundary:

- Do **not** modify or entangle the existing Gaussian Graph-VAE pipeline.
- New VQ code should live under a separate package, e.g. `src/models/vq_model/`.
- Existing `GraphMotionVAE`, `scripts/train_graph_vae.py`, and current diffusion
  code should keep working unchanged.

The first milestone is a trained VQ tokenizer with strong reconstruction and
usable discrete tokens. Token generation from text is a later stage.

## References Read

### Local report

- `outside_docs/per_joint_vqvae_motion_generation_report.md`

Key takeaway: MoGenTS changes the tokenizer structure from whole-body 1D tokens
to a spatial-temporal token map. It does not use a truly separate codebook for
each original joint; it pads fixed human joints and downsamples them into a
small latent spatial axis.

### MoGenTS official code

Temporarily inspected from the official repo:

- `weihaosky/mogents`
- `models/vq/model.py`
- `models/vq/residual_vq.py`
- `models/vq/quantizer.py`
- `models/vq/vq_trainer.py`
- `data/t2m_dataset.py`

Relevant implementation facts:

- `RVQVAE` has a 1D global branch and a 2D joint branch:
  - `models/vq/model.py:23-39`: 1D encoder/decoder + RVQ.
  - `models/vq/model.py:47-68`: 2D joint encoder/decoder + RVQ.
- 2D branch pads fixed human joints, runs Conv2D, then quantizes each latent
  spatial slot independently:
  - `models/vq/model.py:86-95`
  - `models/vq/model.py:108-119`
- Reconstruction repacks the decoded joint map back into the fixed HumanML
  vector:
  - `models/vq/model.py:121-134`
- RVQ is residual, multi-stage, EMA-updated, and supports quantizer dropout:
  - `models/vq/residual_vq.py:28-169`
  - `models/vq/quantizer.py:35-158`
- Trainer uses final reconstruction plus branch auxiliary losses and commit loss:
  - `models/vq/vq_trainer.py:33-72`
  - `models/vq/vq_trainer.py:138-148`
- Dataset constructs a per-joint map from position / rotation / velocity:
  - `data/t2m_dataset.py:98-116`

Important adaptation warning:

MoGenTS assumes fixed `J=22/21` human topology and uses Conv2D spatial
downsampling. We should **not** copy that part. Our spatial downsampling must
remain `EdgeSegmentPool`, because L5 has variable topology and variable valid
coarse slot count.

## Current L5 Facts

Dataset:

- `data_root = data/animo4d_anytop_clean_L5`
- motions: `74522`
- objects: `311`
- actual max joints: `62`
- natural EdgeSegmentPool p=2 max coarse slots: `49`
- recommended compact VAE limits:
  - `max_joints = 64`
  - `max_coarse = 50`

So the VQ token grid should be:

```text
T = 64
temporal_stride = 4
T_lat = 16
C_max = 50
valid C <= 49
```

## Core Design Decision

Use a **coarse-slot structured RVQ**, not per-raw-joint codebooks.

```text
AnyTop13 motion [B,T,J,13]
  -> graph-aware fine encoder
  -> EdgeSegmentPool
  -> h_lat [B,16,C,512]
  -> residual vector quantization
  -> code indices [B,16,C,Q]
  -> dequantized z_q [B,16,C,512]
  -> coarse_xattn decoder
  -> pred_motion [B,64,J,13]
```

This is the direct analogue of MoGenTS's 2D token map, but our spatial axis is
`edge-segment coarse slots`, not fixed human joint-grid positions.

Why this handles variable topology:

- The codebook is shared across all **valid** coarse-slot tokens.
- Padded slots get `index = -1` and are ignored by codebook update, loss, and
  token generation.
- The codebook is not tied to absolute slot id `c`; topology semantics come from
  pooled graph metadata and pooled skeleton embeddings.

Do not implement one codebook per slot in v1. Slot meaning is not globally
consistent across arbitrary skeletons, and L5 has variable valid C.

## Proposed Module Layout

New package:

```text
src/models/vq_model/
  __init__.py
  quantizer.py
  graph_vq_tokenizer.py
  losses.py
  utils.py
```

New scripts:

```text
scripts/train_graph_vqvae.py
scripts/animate_graph_vqvae.py
scripts/export_graph_vq_tokens.py
```

Future, not v1:

```text
scripts/train_graph_vq_transformer.py
scripts/animate_graph_vq_t2m.py
```

## Model Architecture

### Encoder

Reuse our existing modules, but instantiate them inside the new VQ model so the
checkpoint and training loop are separate.

```text
GraphVQTokenizer.encode(batch)

batch.anytop_x [B,J,13,T]
  -> permute [B,T,J,13]
  -> SkeletonEncoder(
       motion_mode="anytop13_split",
       attn_mode="graphormer",
       d_model=512,
       d_ff=1536,
       n_heads=8
     )
  -> SlotNorm
  -> EdgeSegmentPool(max_coarse=50, temporal_stride=4)
  -> h_pool [B,16,C,512]
```

Reuse points from current code:

- `SkeletonEncoder` already handles anytop13 root/non-root split and graphormer
  graph attention.
- `EdgeSegmentPool` already builds hard segment assignment, pooled adjacency,
  pooled geodesic, pooled skeleton embeddings, and frame/coarse masks.

### Optional coarse graph refine before VQ

Add 1-2 lightweight graph-temporal coarse layers before quantization:

```text
for each layer:
  spatial coarse GraphAttentionBlock over C using pooled_adjacency/pooled_geodesic
  temporal self-attn or temporal conv over T_lat per coarse slot
  FFN
  re-mask frame_mask_lat * coarse_mask
```

This keeps the VQ latent graph-aware after pooling. Use
`src/models/graph_salad/attention.py::GraphAttentionBlock` for coarse graph
attention. Do not use MoGenTS Conv2D spatial layers.

Recommended v1:

- `n_pre_vq_layers = 2`
- `n_post_vq_layers = 2`

### Quantizer

Implement a mask-aware Residual VQ:

```text
MaskedResidualVQ.forward(
  x:    [B,T_lat,C,D],
  mask: [B,T_lat,C] bool
) -> {
  quantized:   [B,T_lat,C,D],
  indices:     [B,T_lat,C,Q], long, -1 on padded tokens,
  commit_loss: scalar,
  perplexity:  [Q] or scalar,
  usage:       diagnostics
}
```

Hard requirements:

- Quantize only valid tokens. Padded frame/slot positions must not update the
  codebook.
- Commit loss denominator is valid-token count times D, not padded token count.
- Codebook distance computation should run in fp32 even under BF16 training.
- Codebook buffers must be device-neutral. Do not copy MoGenTS's hardcoded
  `.cuda()` in `QuantizeEMAReset.reset_codebook`.
- Return `-1` indices for padded tokens; downstream token generators must treat
  these as padding.

Recommended quantizer config:

```text
code_dim = 512
num_codes = 512
num_quantizers = 4
shared_codebook = false
ema_mu = 0.99
commit_weight = 0.02
quantize_dropout_prob = 0.1
```

Why Q=4 first:

- MoGenTS uses residual quantizers; its public defaults/configs vary between 3
  and 6 quantizers.
- Our tokens are already graph-pooled into semantic coarse slots, so Q=4 is a
  good first stability point.
- If reconstruction is clearly codebook-limited, Q=6 is the first ablation.

### Decoder

Use our coarse-slot decoder path, not MoGenTS Decoder2D.

```text
z_q [B,16,C,512]
  -> optional post-VQ graph-temporal refine
  -> repeat_interleave temporal_stride=4
  -> MotionDecoder coarse_xattn with real assignment P [B,J,C]
  -> anytop13 root/nonroot output heads
  -> pred_motion [B,64,J,13]
```

Reuse the VAE decoder logic conceptually:

- `GraphMotionVAE.decode` coarse_xattn path uses real assignment and
  MotionDecoder to map coarse slots back to fine joints.
- The VQ model should own its own `MotionDecoder` and output heads under
  `src/models/vq_model/`, even if the module classes are imported.

Do not call or subclass `GraphMotionVAE` in the VQ model. That would entangle
checkpoint format and training behavior.

## Optional Global / Root Branch

MoGenTS has a 1D global branch in addition to the 2D joint branch. For our v1,
do **not** make this mandatory.

Reason:

- Our coarse-slot map already contains a root segment and uses world / FK / traj
  supervision.
- Adding a second global token stream makes the first tokenizer and future token
  generator more complex.

Recommended staging:

- v1.0: coarse-slot RVQ only.
- v1.1 ablation if root trajectory is weak:
  - add `global_tokens [B,T_lat,Qg]`
  - build global feature by masked mean over coarse slots plus explicit root
    slot feature
  - inject global dequantized feature into every coarse slot before decoder
  - use `num_quantizers_global=2`

## Loss

No KL term. VQ replaces Gaussian latent regularization.

Use the same 13-channel reconstruction family as our current VAE:

```text
L_total =
  L_13ch
  + w_world * L_world
  + w_fk    * L_rot6d_fk
  + w_traj  * L_traj
  + w_commit * L_commit
```

Recommended weights:

```text
w_pos      = 1.0
w_rot      = 1.0
w_vel      = 1.0
w_contact  = 0.1
w_world    = 0.25
w_fk       = 1.0
w_traj     = 0.10
w_commit   = 0.02
```

Implementation reuse:

- `compute_total_loss_13ch`
- `compute_world_rot6d_fk_terms`

Important:

- `frame_mask` for final reconstruction must be the recovered full-resolution
  frame mask.
- Geometry losses need de-normalized motion via `batch.anytop_mean/std`.
- Geometry loss should run in fp32 under BF16 training.

Do not include:

- KL loss
- pool aux loss
- diffusion losses
- text conditioning

The VQ tokenizer is unconditional reconstruction.

## Data and Split

Use:

```text
data_root = data/animo4d_anytop_clean_L5
max_joints = 64
max_coarse = 50
max_frames = 64
temporal_stride = 4
```

Before the first real run, materialize `splits/train.txt` and `splits/val.txt`
for L5 if they are not already present. Use a stable `95/5` split with
`seed=42`, consistent with the large-data VAE convention.

Rationale:

- `AnyTopDataset` can fall back to md5 object-stratified splitting, but explicit
  split files make VAE / VQ / later token generator comparisons reproducible.
- Do not let the VQ tokenizer use a different split from later generator eval.

Training dataset should use random 64-frame windows, same as VAE tokenizer
training. Full-motion generation/eval is a separate downstream concern.

## Training Config

Recommended first run:

```text
dataset = anytop_truebones
anytop_root = data/animo4d_anytop_clean_L5
feat_mode = anytop13
attn_mode = graphormer
pool_type = edge_segment
max_joints = 64
max_coarse = 50
max_frames = 64
temporal_stride = 4
d_model = 512
d_ff = 1536
n_heads = 8
n_graph_layers = 4
n_enc_temporal_layers = 2
n_pre_vq_layers = 2
n_post_vq_layers = 2
n_cross_layers = 3
n_dec_temporal_layers = 2
code_dim = 512
num_codes = 512
num_quantizers = 4
commit_weight = 0.02
amp_dtype = bf16
epochs = 300
lr = 2e-4
```

Batch size:

- Determine by smoke on the target allocation.
- L5 is compact, so start higher than the old J144/C128 runs.
- Keep linear LR scaling only if batch is deliberately scaled from a known
  reference; otherwise start at `2e-4` and do not over-tune before smoke.

Checkpoint fields:

- model state
- optimizer state
- epoch / global step
- full args
- codebook usage stats
- split manifest hash
- dataset root and max_joints/max_coarse

## Diagnostics to Log

Must log per epoch:

- train / val `total`
- `pos`, `rot`, `vel`, `contact`
- `world`, `fk`, `traj`, `gt_fk_mismatch`
- `commit`
- codebook perplexity per quantizer
- active code count per quantizer
- dead code count per quantizer
- valid token count
- mean / p95 valid coarse slot count
- reconstruction speed ratio on a fixed QA subset

Codebook health gates:

- If perplexity is near 1 for most quantizers after warmup, codebook collapsed.
- If later quantizers are always unused, lower Q or increase dropout schedule.
- If reconstruction is good but code usage is tiny, future token generator may
  overfit; keep that as a generator-stage risk.

## Visual QA

Metric alone is not sufficient. Render GT-vs-VQ-recon GIFs.

Required first QA set:

- `PZ_Giant_Anteater_Male`
- `PZ_Grey_Seal_Female`
- `PZ_Caracal_Male`
- `PZ_California_Sea_Lion_Female`
- `PZ_Nine_Banded_Armadillo_Male`
- `PZ_African_Crested_Porcupine_Male`
- `PZ_Giant_Otter_Male`
- `PZ_Reticulated_Giraffe_Male`

Use the L5 dense render manifest as a source for dense topology cases:

```text
data/animo4d_anytop_clean_L5/animations/l5_dense_random10_20260608/manifest.json
```

Success criterion for tokenizer:

- No frozen / collapse / severe jitter in VQ reconstruction.
- Dense-topology animals preserve main motion framework.
- Reconstruction is visually close enough that a token generator would not be
  blamed for tokenizer failure.

## Implementation Milestones

### M0: Preflight

- Verify L5 has 74522 motions and `J_max=62`.
- Materialize stable splits if missing.
- Build one `AnyTopDataset` batch with `max_joints=64`.
- Run `EdgeSegmentPool(max_coarse=50)` and assert max valid C <= 49.

Stop if:

- any object is skipped by `max_joints=64`
- any sample overflows `max_coarse=50`
- mean/std or motion tensors are nonfinite

### M1: Quantizer Unit Tests

Implement `MaskedResidualVQ`.

Smoke tests:

- valid tokens only update codebook
- padded tokens return `indices=-1`
- dequantized padded outputs are zero after mask
- BF16 model path still computes VQ distance/commit in fp32
- save/load preserves codebooks

### M2: GraphVQTokenizer Forward Smoke

Implement model forward only.

Expected shapes:

```text
pred_motion: [B,64,J,13]
indices:     [B,16,50,4]
z_q:         [B,16,50,512]
```

Smoke:

- 5 train iterations
- finite loss
- nonzero gradients in encoder, quantizer-adjacent projections, decoder
- no gradients into padded tokens
- old Gaussian VAE scripts still import and run smoke unchanged

### M3: First Real Tokenizer Run

Train 300 epochs on L5.

Save:

- `best_model.pt` by val total
- `best_recon_model.pt` by val reconstruction
- `last_model.pt`
- periodic snapshots every 25-50 epochs

Render QA at early/mid/final checkpoints.

### M4: Token Export

After tokenizer passes visual QA:

```text
scripts/export_graph_vq_tokens.py
```

Output per motion:

```text
tokens:          [T_lat,C,Q] int16, -1 padded
coarse_mask:     [C] bool
frame_mask_lat:  [T_lat] bool
pooled graph metadata or reference to object topology
motion_id / object_type / captions
```

Do not train a token generator until this export format is stable.

### M5: Token Generator, Later

Future text-to-motion generator should use:

- token grid `[T_lat,C,Q]`
- graph-aware spatial + temporal transformer
- dual text conditioning if the diffusion experiments confirm it remains useful
- multi-positive caption handling if using evaluator/retrieval losses

This is not part of the first VQ tokenizer implementation.

## Ablations After First Tokenizer

Only run these after v1 has a clean reconstruction:

1. `Q=4` vs `Q=6`
2. `num_codes=512` vs `1024`
3. `n_pre/post_vq_layers=0` vs `2`
4. coarse-slot-only vs added global/root branch
5. L5 compact `J64/C50` vs compatibility `J144/C72`

Do not start with these. First establish that the VQ tokenizer can reconstruct.

## Main Risks

### Risk 1: Codebook collapse

Mitigation:

- mask-aware valid-token-only quantization
- EMA codebook reset from valid tokens
- perplexity / dead-code logging
- Q=4 first, Q=6 only if needed

### Risk 2: Quantization hurts long-chain or dense skeletons

Mitigation:

- required dense-topology visual QA
- keep graph-aware pre/post VQ layers
- keep world + FK geometry losses

### Risk 3: Future token generator has too many tokens

L5 compact token count is manageable:

```text
16 * <=50 * Q
```

For Q=4 this is up to 3200 discrete indices per 64-frame clip if every residual
level is generated independently. The generator should probably factorize over
residual levels rather than flattening everything as one long sequence.

Do not solve this before tokenizer QA.

## Executor Prompt

Implement the Graph-aware VQ-VAE tokenizer plan in
`handoff/20260608_graph_vqvae_l5_pipeline_plan.md`.

Hard requirements:

1. Keep the current Gaussian Graph-VAE and diffusion code unchanged.
2. Put VQ-specific modules under `src/models/vq_model/`.
3. Use `data/animo4d_anytop_clean_L5` with `max_joints=64`,
   `max_coarse=50`, `max_frames=64`, `temporal_stride=4`.
4. Use our `SkeletonEncoder`, `SlotNorm`, `EdgeSegmentPool`, graph-aware coarse
   attention, and `MotionDecoder`/anytop13-style output heads. Do not copy
   MoGenTS Conv2D spatial downsampling.
5. Implement a mask-aware Residual VQ that ignores padded coarse slots and
   padded latent frames, returns `-1` for padded indices, and computes VQ
   distance/commit in fp32 under BF16.
6. Train tokenizer reconstruction only; no text conditioning and no token
   generator in the first implementation.
7. Loss = 13ch reconstruction + world/RIC + rot6d-FK + root traj + commit.
   No KL.
8. Every milestone needs a smoke test and visual QA GIFs. Dense L5 animals must
   be included before claiming the tokenizer works.

Start with M0-M2 only, run codex review after the initial implementation, and
do not launch a 300-epoch run until smoke + shape + codebook-health gates pass.
