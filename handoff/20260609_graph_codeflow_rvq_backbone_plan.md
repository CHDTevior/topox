# Graph-CodeFlow RVQ Backbone Plan

Date: 2026-06-09

Status: planning / pre-implementation. This document records the provisional
backbone design after the Graph-VQVAE tokenizer. No training code is changed by
this document.

## 0. Decision Summary

We tentatively make **Graph-CodeFlow continuous embedding generation** the main
post-RVQ backbone path.

The core idea:

```text
frozen Graph-VQVAE tokenizer
  motion -> RVQ z_q [B,T_lat,C,D] + indices [B,T_lat,C,Q]

Graph-CodeFlow backbone
  text + skeleton graph + noise -> generated continuous z_hat [B,T_lat,C,D]

RVQ projection
  z_hat -> residual-nearest indices_hat [B,T_lat,C,Q] -> z_snap [B,T_lat,C,D]

frozen Graph-VQVAE decoder
  z_snap + assignment + skeleton -> generated anytop13 motion [B,T,J,13]
```

This is different from a pure token classifier. The main model learns a
rectified-flow velocity field over **quantized code embedding space** (`z_q`),
then snaps generated embeddings back to the frozen RVQ codebooks before decode.

The earlier `handoff/20260609_rvq_token_backbone_preparation.md` MaskGIT /
base-residual token plan remains useful, but should be treated as a secondary
ablation or v1.1 fallback, not the first primary route.

## 1. Source References

External source:

- GitHub: <https://github.com/PengchengFang-cs/CodeFlow>
- Local clone: `outside_docs/CodeFlow`
- Commit read: `cc8e1bccf7f2adfa22828bcf2eb641851e4a194e`
- Local notes: `outside_docs/codeflow_research_notes.md`

Most relevant CodeFlow files:

- `outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py`
- `outside_docs/CodeFlow/models/codeflow/motion_code_flow.py`
- `outside_docs/CodeFlow/models/codeflow/dit_blocks.py`
- `outside_docs/CodeFlow/models/codeflow/kv_vq.py`
- `outside_docs/CodeFlow/models/codeflow/momask_vq.py`
- `outside_docs/CodeFlow/scripts/launch/train_humanml3d_pscf_standard.sh`

Our relevant files:

- `src/models/vq_model/graph_vq_tokenizer.py`
- `src/models/vq_model/quantizer.py`
- `src/models/vq_model/masked_motion_decoder.py`
- `src/models/graph_salad/pool_edge_segment.py`
- `src/models/graph_salad/attention.py`
- `src/models/graph_salad/denoiser.py`
- `handoff/20260609_rvq_token_backbone_preparation.md`

## 2. What CodeFlow Actually Does

CodeFlow's strong path is not direct motion diffusion and not plain ID
classification. It is:

```text
HumanML3D motion
  -> frozen part-aware VQ tokenizer
  -> code embeddings [B,T,part,D]
  -> rectified flow prior over code embeddings
  -> nearest / terminal projection to code IDs
  -> frozen VQ decoder
  -> motion
```

For PS-CF, `part` is a fixed human body-part axis. In the standard setting:

```text
num_parts = 6
code_dim = 128
hidden_size = num_parts * part_hidden_dim
```

PS-CF packs one latent frame by projecting each of the 6 parts, then
concatenating those part vectors along the feature/channel dimension. This is
why the code enforces:

```text
hidden_size == num_parts * part_hidden_dim
```

The useful transferable ideas are:

1. Use a frozen motion tokenizer.
2. Train the generator in code embedding space, not raw motion space.
3. Use rectified flow / ODE sampling instead of v-pred diffusion.
4. Use strong text conditioning: pooled text plus token-level text attention.
5. Project the sampled continuous embedding back to codebook space before
   decode.

The non-transferable part is the fixed 6-body-part frame-grouped packing.

## 3. Why We Should Not Directly Copy PS-CF

Our Graph-VQVAE token map is:

```text
z_q     [B,T_lat,C,D]
indices [B,T_lat,C,Q]
```

where:

- `T_lat` is latent time, usually 16 for 64-frame training with stride 4.
- `C` is a padded coarse-slot graph axis from EdgeSegmentPool.
- `D` is code dimension, currently 512.
- `Q` is residual quantizer depth, currently 4.

Important distinction:

```text
CodeFlow PS-CF part axis: fixed semantic human body parts.
Our C axis: variable arbitrary-topology coarse graph slots.
Our Q axis: residual quantizer layers, not spatial parts.
```

Therefore:

- Do not set `num_parts = Q`.
- Do not treat RVQ depth as body part structure.
- Do not concatenate all `C` slots into `hidden_size = C * part_hidden_dim` as
  the main design. `C` is variable and graph-structured.
- Do not flatten `[T,C,Q]` into a blind 1D stream for v1.

The correct adaptation is to keep `C` as a graph node axis and let the backbone
operate on `[B,T_lat,C,D]` with masks and pooled graph metadata.

## 4. Current Graph-VQVAE Token Contract

Tokenizer training path:

```text
anytop13 motion [B,T,J,13]
  -> SkeletonEncoder
  -> SlotNorm
  -> EdgeSegmentPool
  -> h_lat [B,T_lat,C,D]
  -> MaskedResidualVQ
       z_q     [B,T_lat,C,D]
       indices [B,T_lat,C,Q]
  -> MaskedMotionDecoder
  -> pred_motion [B,T,J,13]
```

Current L5 v1 expected defaults:

```text
T      = 64
T_lat  = 16
C_max  = 50 or configured max_coarse
D      = 512
Q      = 4
K      = 512 codes per RVQ layer
```

Valid tokens:

```text
token_mask = frame_mask_lat[:, :, None] & coarse_mask[:, None, :]
shape      = [B,T_lat,C]
```

Padding contract:

```text
token_mask=False -> indices = -1, z_q = 0, no loss, no attention leakage
```

### 4.1 Symbol-Level Information Flow

Use the following symbols throughout the backbone plan:

```text
B = batch size
T = original motion frames, v1 fixed to 64
J = fine joints after dataset padding / cleaning
C = padded coarse-slot count from EdgeSegmentPool
D = RVQ code embedding dimension
Q = residual quantizer depth
K = codebook size per RVQ stage
```

Current Graph-VQVAE / L5 reference values:

```text
T = 64
T_lat = T / temporal_stride = 16
C = C_max from Graph-VQVAE config, expected 50 for the current L5 tokenizer plan
D = 512
Q = 4
K = 512
```

For one training batch:

```text
motion:
  x_motion [B,T,J,13]

frozen Graph-VQVAE encoder + pool:
  h_lat [B,T_lat,C,D]

frozen RVQ quantizer:
  z_q     [B,T_lat,C,D]
  indices [B,T_lat,C,Q]

validity:
  coarse_mask    [B,C]
  frame_mask_lat [B,T_lat]
  token_mask     [B,T_lat,C]
```

The backbone sees only the post-RVQ target:

```text
target_raw = z_q [B,T_lat,C,D]
```

It does **not** see the original continuous Gaussian VAE latent, and it does not
train against the pre-quantization `h_lat` in v1.

Rectified-flow training tensor shapes:

```text
noise           [B,T_lat,C,D]
t               [B]
z_t             [B,T_lat,C,D]
velocity_target [B,T_lat,C,D]
velocity_pred   [B,T_lat,C,D]
```

At inference:

```text
sampled continuous:
  z_hat [B,T_lat,C,D]

RVQ residual-nearest projection:
  indices_hat [B,T_lat,C,Q]
  z_snap      [B,T_lat,C,D]

frozen decoder output:
  pred_motion [B,T,J,13]
```

So the planned generation backbone is:

```text
text + skeleton graph + noise[B,T_lat,C,D]
  -> Graph-CodeFlow
  -> z_hat[B,T_lat,C,D]
  -> residual-nearest RVQ snap
  -> indices_hat[B,T_lat,C,Q] + z_snap[B,T_lat,C,D]
  -> frozen Graph-VQVAE decoder
  -> motion[B,T,J,13]
```

## 5. Proposed Data Flow After RVQ

### 5.1 Offline Export

After a Graph-VQVAE checkpoint is accepted, export an offline RVQ dataset:

```text
for each real motion:
  tokenizer.encode(batch)
  quantizer(...)

store:
  z_q                     [T_lat,C,D]
  indices                 [T_lat,C,Q]
  token_mask              [T_lat,C]
  coarse_mask             [C]
  frame_mask_lat          [T_lat]
  pooled_adjacency        [C,C]
  pooled_geodesic         [C,C]
  pooled_skeleton_embeddings / slot embeddings if available
  assignment              [J,C]       for decode
  skeleton metadata       parent_indices, rest offsets, joint names/features
  caption ids / caption_emb / caption_token_emb references
```

Main training should not run the tokenizer encoder online every step. The
tokenizer is frozen, and backbone training should read cached `z_q` and graph
metadata.

### 5.2 Backbone Training Target

The main target is `z_q`, not raw `h_lat` and not only discrete `indices`.

```text
target_raw = z_q [B,T_lat,C,D]
valid      = token_mask [B,T_lat,C]
```

Optional normalization:

```text
target = normalize(target_raw)
```

Preferred first choice:

- Codebook-stat normalization if robust after the VQ checkpoint is fixed.
- Otherwise empirical train-set `z_q` mean/std over valid tokens.

The normalizer must be frozen and saved with the backbone checkpoint.

## 6. Graph-CodeFlow Objective

Use CodeFlow-style rectified flow:

```text
noise ~ N(0, I)
t     ~ Uniform(0, 1)

z_t = t * target + (1 - t) * noise
velocity_target = target - noise

velocity_pred = model(
  z_t,
  t,
  text_global,
  text_tokens,
  pooled_adjacency,
  pooled_geodesic,
  token_mask,
)

loss = masked_mse(velocity_pred, velocity_target, valid=token_mask)
```

Masking:

```text
loss numerator: valid token positions only
loss denominator: valid tokens * D
```

Default v1 loss:

```text
total_loss = flow_loss
```

Do not add terminal CE, decoded geometry loss, or residual ID CE in the first
training run unless the base flow fails a diagnostic gate. Keep v1 clean so the
result answers whether CodeFlow-style latent flow works.

### 6.1 Training Recipe And Hyperparameters

We should start from CodeFlow's public PS-CF standard recipe, but adapt the data
source and tensor shape to our Graph-VQVAE tokens.

CodeFlow standard recipe, from
`outside_docs/CodeFlow/scripts/launch/train_humanml3d_pscf_standard.sh`:

```text
batch_size              = 64
max_epoch               = 600
lr                      = 1e-4
lr_scheduler            = half_cosine
eta_min_ratio           = 0.01
warmup_steps            = 2000
weight_decay            = 0.01
grad_clip               = 1.0
amp_dtype               = bf16
seed                    = 42
cond_drop_prob          = 0.1
time_schedule           = uniform
sampling_method         = ode
latent_norm_mode        = codebook
terminal_mode           = tied_logits
terminal_tau_mode       = codebook_nn
terminal_loss_weight    = 0.0
clean_loss_weight       = 0.0
flow_loss_weight        = 1.0
eval_every_epoch        = 10
eval_steps              = 96
eval_cond_scale         = 6.0
```

The parts we should preserve directly:

```text
optimizer/lr:
  AdamW
  lr = 1e-4 initial reference
  weight_decay = 0.01
  grad_clip = 1.0

schedule:
  warmup_steps = 2000
  half_cosine decay
  eta_min_ratio = 0.01

flow:
  uniform t sampling
  ODE sampling
  flow-only objective in v1
  terminal_loss_weight = 0.0
  clean_loss_weight = 0.0

conditioning:
  cond_drop_prob = 0.1
  CFG eval cond_scale = 6.0 initial reference

precision:
  bf16 autocast
  fp32 for mask reductions / codebook distances / numerically sensitive losses
```

The parts we must adapt:

```text
data_root:
  CodeFlow original: HumanML3D
  ours: data/animo4d_anytop_clean_L5

tokenizer:
  CodeFlow original: frozen KV / human VQ tokenizer
  ours: frozen Graph-VQVAE tokenizer checkpoint

target:
  CodeFlow original: [B,T_code,6,128]
  ours: [B,T_lat,C,512]

architecture:
  CodeFlow original: fixed 6-part frame-grouped PS-CF
  ours: graph-structured C-slot CodeFlow
```

Initial adapted v1 training config:

```text
dataset_root             = data/animo4d_anytop_clean_L5
tokenizer_ckpt           = accepted Graph-VQVAE L5 checkpoint
offline_token_dataset    = exported z_q / indices / graph metadata

T                        = 64
T_lat                    = 16
C                        = Graph-VQVAE C_max, expected 50 for current L5 plan
D                        = 512
Q                        = 4
K                        = 512

batch_size               = start from 64 if memory allows
lr                       = 1e-4
epochs                   = 600 reference; first real run can stop early by visual QA
warmup_steps             = 2000
lr_scheduler             = half_cosine
eta_min_ratio            = 0.01
weight_decay             = 0.01
grad_clip                = 1.0
amp_dtype                = bf16
cond_drop_prob           = 0.1
flow_loss_weight         = 1.0
terminal_loss_weight     = 0.0
clean_loss_weight        = 0.0
sampling_method          = ode
eval_steps               = 96
eval_cond_scale          = 6.0
seed                     = 42
```

Batch-size rule:

```text
Use batch_size=64 as the CodeFlow-faithful starting point.
If the graph C-slot model is memory-heavy, lower batch before changing the loss.
If we scale global batch up/down substantially, retune lr explicitly rather than
silently changing the CodeFlow recipe.
```

The first implementation should log both the CodeFlow-style metrics and our RVQ
specific metrics:

```text
flow_loss
projection_error = mse(z_hat, z_snap) on valid tokens
nearest residual ID usage / perplexity per q
continuous-decode vs snapped-decode QA
decoded motion GIFs
```

## 7. Backbone Architecture

Recommended v1 name:

```text
GraphStructuredCodeFlow
```

Input/output:

```text
input  z_t [B,T_lat,C,D]
output v   [B,T_lat,C,D]
```

Conditioning:

```text
time embedding t
global caption embedding [B,768]
token-level caption embeddings [B,L,768]
pooled graph metadata [B,C,C]
coarse/token masks
```

Layer shape:

```text
x [B,T_lat,C,D]

per layer:
  graph-spatial attention over C
    uses pooled_adjacency / pooled_geodesic

  temporal attention over T_lat
    per coarse slot

  text token cross-attention or CodeFlow-style motion-text joint attention

  timestep/global-text FiLM or AdaLN-Zero modulation

  FFN / SwiGLU

  re-mask x *= token_mask[...,None]
```

Two implementation levels:

### Level A: minimal proof model

Reuse our existing graph-temporal denoiser ingredients:

- `GraphAttentionBlock`
- `TemporalSelfAttention`
- dual text conditioning from `graph_salad/denoiser.py`
- masked residual discipline

This gives a quick controlled proof of the objective with less source churn.

### Level B: CodeFlow-like main model

Adopt stronger CodeFlow DiT ingredients:

- AdaLN-Zero / timestep-conditioned modulation
- double-stream text-motion attention
- single-stream joint text-motion blocks
- SwiGLU / DiT-style blocks

But preserve our graph-spatial sub-blocks. CodeFlow's `FrameMotionTextDiT`
operates on frame tokens; our version should operate on graph-structured
motion tokens.

Recommended path:

```text
M2 minimal Graph-CodeFlow first
M3 CodeFlow-like Graph DiT after M2 proves the target and sampling path
```

## 8. Inference Flow

Given a target skeleton and text prompt:

```text
1. Build skeleton-only graph metadata
   - coarse_mask
   - token_mask for target T_lat
   - pooled_adjacency / pooled_geodesic
   - assignment [J,C]
   - skeleton embeddings for decoder

2. Sample initial noise
   z_0_noise [B,T_lat,C,D]
   zero padded tokens

3. ODE integrate from t=0 to t=1
   for step in sampler_grid:
     v = GraphCodeFlow(z_t, t, text, graph, masks)
     z_t = z_t + dt * v

4. Denormalize
   z_hat [B,T_lat,C,D]

5. Residual-nearest RVQ projection
   z_hat -> indices_hat [B,T_lat,C,Q] -> z_snap [B,T_lat,C,D]

6. Decode
   frozen Graph-VQVAE decoder(z_snap, skeleton metadata)
   -> generated motion [B,T,J,13]
```

CFG:

```text
conditional: text_global + text_tokens
unconditional: both global and token text dropped together
v = v_uncond + cfg_scale * (v_cond - v_uncond)
```

Default sampler:

```text
ODE, 50-100 steps for first QA
```

## 9. How To Get Full RVQ Indices At Inference

CodeFlow's nearest projection is one-stage for a single codebook or per-part
codebooks. Our RVQ needs **sequential residual nearest**.

For each valid `(b,t,c)`:

```text
r0 = z_hat[b,t,c]

idx0 = nearest(r0, codebook_0)
e0   = codebook_0[idx0]
r1   = r0 - e0

idx1 = nearest(r1, codebook_1)
e1   = codebook_1[idx1]
r2   = r1 - e1

idx2 = nearest(r2, codebook_2)
e2   = codebook_2[idx2]
r3   = r2 - e2

idx3 = nearest(r3, codebook_3)
e3   = codebook_3[idx3]

indices_hat = [idx0, idx1, idx2, idx3]
z_snap      = e0 + e1 + e2 + e3
```

Padded positions:

```text
token_mask=False -> indices_hat = -1, z_snap = 0
```

This gives the complete RVQ ID map:

```text
indices_hat [B,T_lat,C,Q]
```

and a codebook-consistent continuous latent:

```text
z_snap [B,T_lat,C,D]
```

The decoder should default to `z_snap`, not raw `z_hat`, for the main generation
path. Raw continuous decode can remain a diagnostic upper bound.

## 10. Required Tokenizer Utility APIs

Before implementing the backbone, add small frozen-tokenizer utilities:

```text
GraphVQTokenizer.ids_to_embeddings(indices, token_mask) -> z_q

GraphVQTokenizer.nearest_residual_ids(z_hat, token_mask)
  -> indices_hat, z_snap, projection_error

GraphVQTokenizer.prepare_skeleton_only(batch_or_skeleton, T_lat)
  -> decode metadata:
     assignment, coarse_mask, frame_mask_lat, token_mask,
     pooled_adjacency, pooled_geodesic, s_j
```

Implementation notes:

- `ids_to_embeddings` sums codebook vectors across RVQ stages.
- `nearest_residual_ids` mirrors the quantizer's residual loop but no EMA update.
- All distance/argmin math should run in fp32.
- Returned indices must be `-1` on padded positions.
- `projection_error = ||z_hat - z_snap||^2` over valid tokens is a required QA
  metric.

## 11. Why This Uses RVQ Well

This plan uses RVQ in three ways:

1. Training target is quantized embedding `z_q`, not pre-VQ encoder output.
2. Sampling output is snapped back through frozen residual codebooks.
3. Generated samples expose both continuous quality (`z_hat`) and discrete
   quality (`indices_hat`, code usage, projection error).

Bad path to avoid:

```text
h_lat -> continuous flow -> decoder
```

That would bypass the codebooks and degrade the RVQ tokenizer into a regular
continuous autoencoder.

Preferred path:

```text
h_lat -> RVQ z_q / indices -> flow over z_q -> residual nearest -> decoder
```

## 12. Relationship To Earlier MaskGIT / Base-Residual Plan

The previous plan proposed:

```text
BaseGraphMaskTransformer for q0
SharedResidualGraphTransformer for q1..q3
```

Keep this idea, but move it to secondary status.

Recommended priority:

```text
Primary v1:
  Graph-CodeFlow over z_q + residual nearest projection

Fallback / ablation:
  MaskGIT-style base/residual ID generator

Hybrid later:
  Graph-CodeFlow z_hat
  + terminal / residual ID corrector
```

When to revisit ID generators:

- `projection_error` remains high.
- nearest-projected `indices_hat` have poor code usage or visible artifacts.
- continuous decode looks good but snapped decode degrades strongly.
- we need explicit discrete editing or token-level control.

## 13. Training Milestones

### M0: VQ checkpoint gate

Inputs:

- accepted Graph-VQVAE checkpoint on `animo4d_anytop_clean_L5`
- frozen tokenizer config and codebooks

Gates:

- VQ reconstruction GIFs pass visual QA.
- codebook perplexity / active code usage healthy.
- root drift / jitter acceptable.
- tokenizer can decode both `z_q` and `ids_to_embeddings(indices)`.

### M1: RVQ dataset export

Write exporter:

```text
real motion -> z_q / indices / masks / graph metadata / caption pointers
```

Gates:

- strict shape audit.
- padded IDs exactly `-1`.
- no valid ID outside `[0,K-1]`.
- `ids_to_embeddings(indices)` reconstructs cached `z_q` within tolerance.
- train/val split mirrors source split.

### M2: Minimal Graph-CodeFlow

Implement minimal model with our existing graph-temporal blocks and dual text.

Gates:

- tiny overfit on a small subset.
- flow loss finite and decreasing.
- generated `z_hat` finite.
- residual-nearest projection finite.
- decoder output finite.
- GIFs on 10 fixed QA prompts show non-static motion.

### M3: CodeFlow-like Graph DiT

If M2 validates the objective, implement stronger CodeFlow-style blocks:

- AdaLN-Zero.
- double-stream / single-stream text-motion attention.
- graph-spatial block kept explicit.
- ODE + CFG sampling.

Gates:

- improves visual control over M2.
- no loss of graph/topology validity.
- projection error lower or equal.

### M4: ID / residual ablations

Only after M2/M3:

- add terminal residual ID logits.
- add residual ID corrector.
- compare against MaskGIT base/residual generator.
- compare snapped decode vs continuous decode.

## 14. Evaluation And QA

Metric logs:

- flow loss.
- projection error `||z_hat - z_snap||`.
- nearest residual reconstruction error.
- generated code usage per RVQ stage.
- active codes / perplexity on generated IDs.
- decoded anytop13 loss against GT for reconstruction-style probes.

Visual QA:

- text-to-motion GIFs.
- static skeleton / generated / GT side by side when GT exists.
- long-chain species.
- dense topology species.
- high-energy and low-energy actions.
- slow actions, to catch CFG/energy overshoot.

Early decision gates:

```text
continuous decode good, snapped decode bad:
  RVQ projection / codebook mismatch problem.

continuous and snapped both bad:
  flow backbone / conditioning problem.

snapped decode good but motion bad:
  tokenizer decoder bottleneck or skeleton metadata issue.
```

Visual QA remains the primary gate.

## 15. Open Design Choices

1. Normalization:

```text
Option A: codebook-stat normalization.
Option B: empirical z_q train-set normalization.
```

Default: start with empirical `z_q` stats if codebook-stat behavior is uncertain,
then ablate codebook stats.

2. Decode path:

```text
Default: z_snap from residual-nearest projection.
Diagnostic: raw z_hat continuous decode.
```

3. Architecture level:

```text
M2 minimal graph flow first.
M3 CodeFlow-like graph DiT after M2 proves the target.
```

4. Terminal ID loss:

```text
Default v1: off.
Add only if projection_error / snapped decode says it is needed.
```

5. Motion length:

```text
Default v1: fixed 64 frames, T_lat=16.
Variable length later.
```

## 16. Implementation Boundary

This should be a separate backbone pipeline from Gaussian VAE diffusion and
Graph-VQVAE tokenizer training.

Suggested directory:

```text
src/models/CodeFlow_Model/
scripts/export_graph_vq_tokens.py
scripts/train_graph_codeflow.py
scripts/animate_graph_codeflow.py
```

This branch is **not** the existing continuous-space VAE / latent diffusion
route. It should not be implemented as another `graph_salad` latent diffusion
mode and should not change the current Gaussian Graph-VAE, frozen VAE diffusion,
or denoiser training behavior.

Allowed reuse:

- graph-aware attention blocks, e.g. skeleton / pooled-graph attention
- temporal attention helpers
- dual text-conditioning utilities
- AnyTop batch / caption loading utilities
- frozen Graph-VQVAE tokenizer and decoder APIs

Not allowed in v1:

- changing existing Gaussian VAE decode / diffusion semantics
- making current latent diffusion depend on CodeFlow modules
- placing CodeFlow-specific model logic inside the old denoiser path
- silently replacing the old continuous-latent backbone

The preferred engineering boundary is:

```text
Existing lines:
  Graph-VAE / latent diffusion       stays as-is
  Graph-VQVAE tokenizer              stays as frozen tokenizer after training

New line:
  CodeFlow_Model / Graph-CodeFlow    separate post-RVQ generation branch
```

Shared modules can be imported, but new CodeFlow-specific logic should live in
the new `CodeFlow_Model` folder so this experiment remains reversible and does
not perturb earlier baselines.

## 17. Current Recommendation

Proceed later with:

```text
Graph-CodeFlow continuous embedding backbone
  target: frozen RVQ z_q [B,T_lat,C,D]
  condition: dual text + skeleton pooled graph
  objective: rectified-flow velocity MSE
  inference: ODE + CFG
  terminal: residual-nearest projection to full indices [B,T_lat,C,Q]
  decode: frozen Graph-VQVAE decoder from z_snap
```

This best matches the strong part of CodeFlow while respecting our arbitrary
topology design. It uses RVQ as the latent manifold and discrete projection
mechanism, without incorrectly treating RVQ depth as a body-part axis.
