# RVQ Token Backbone Preparation

Date: 2026-06-09

Status: planning / pre-implementation. This document records the current design
thinking for the generation backbone that will consume the Graph-VQVAE RVQ
tokens. It does not change the running Graph-VQVAE tokenizer training.

## Scope

We are preparing the stage after the Graph-VQVAE tokenizer:

```text
text + skeleton / pooled graph
  -> token backbone
  -> RVQ code indices [B, T_lat, C, Q]
  -> frozen Graph-VQVAE decoder
  -> AnyTop13 motion [B, T, J, 13]
```

This document is about the token generator / backbone only. The tokenizer itself
is tracked by:

- `handoff/20260608_graph_vqvae_l5_pipeline_plan.md`
- `handoff/20260608_0600_graph_vqvae_review_verdict_and_forks.md`
- `handoff/20260608_0807_graph_vqvae_training_walkthrough.md`

The external reference note is:

- `outside_docs/per_joint_vqvae_motion_generation_report.md`

## Current Graph-VQVAE Token Contract

The tokenizer input and output contract is:

```text
AnyTop motion:
  batch.anytop_x [B, J_max, 13, T=64]

Tokenizer continuous latent before RVQ:
  h_lat [B, T_lat=16, C_max=50, D=512]
  token_mask [B, 16, 50]

RVQ output:
  indices [B, 16, 50, Q=4]
  z_q     [B, 16, 50, 512]
```

Semantics:

- `T_lat = 16` comes from `temporal_stride = 4`.
- `C_max = 50` is the padded coarse-slot axis after `EdgeSegmentPool`.
- `C_valid <= 49` for L5; padded slots use `token_mask=False` and index `-1`.
- `Q = 4` is the RVQ residual depth.
- Each valid `(t, c)` location is represented by four code IDs:

```text
indices[b, t, c, :] = [q0, q1, q2, q3]
```

The code vocabulary size is currently:

```text
num_codes = 512
valid code id range = 0 ... 511
padding id = -1
```

## What To Learn From MoGenTS

MoGenTS uses a fixed-human setting:

```text
Human joints: 22
pad to:       24
2D encoder spatial downsample:
  24 -> 6 latent spatial slots

RVQ indices:
  2D branch [B, T, 6, Q]
```

The useful idea is not the fixed `6` spatial slots. That part is human-specific.
The useful ideas are:

1. Preserve a structured token map instead of flattening all IDs into one long
   sequence.
2. Generate the first RVQ layer (`q0`) separately from later residual layers.
3. Use masked parallel generation rather than left-to-right autoregressive
   generation over every ID.
4. Use spatial and temporal attention patterns that respect the token map.

MoGenTS does:

```text
text
  -> Mask Transformer for base IDs q0
  -> Residual Transformer for q1...qQ-1
  -> VQ decoder
```

We should adapt that to arbitrary topology:

```text
text + pooled graph
  -> Graph Mask Transformer for base IDs q0
  -> Graph Residual Transformer for q1...q3
  -> frozen Graph-VQVAE decoder
```

## Main Design Decision

Use two backbone stages:

```text
Stage A: Base Graph Mask Transformer
  predicts indices[..., 0]  -> [B, 16, C]

Stage B: Shared Graph Residual Transformer
  predicts indices[..., q] for q = 1, 2, 3
```

Do not train one independent model per residual layer in v1. A single shared
residual transformer with a learned `q_layer_embedding` is simpler and should
generalize better.

Do not flatten `[T, C, Q]` into a single 1D token stream in v1. Flattening erases
the fact that:

- `T` is time.
- `C` is a pooled skeleton graph axis.
- `Q` is residual quantizer depth, not another spatial or temporal axis.

## Proposed Shape Flow

### Offline Token Extraction

After training a Graph-VQVAE tokenizer:

```text
for each real motion:
  tokenizer.encode + quantizer
  -> indices       [16, 50, 4]
  -> token_mask    [16, 50]
  -> coarse_mask   [50]
  -> frame_mask_lat[16]
  -> pooled_adjacency [50, 50]
  -> pooled_geodesic  [50, 50]
  -> pooled_skeleton_embeddings [50, 512]
  -> assignment / hard_assignment for decode-time recovery
```

These should be cached into a token dataset. The token backbone should not
re-encode motion online during its main training loop.

### Base Transformer Training Target

```text
target_base = indices[..., 0]  # [B, 16, 50]
valid       = token_mask       # [B, 16, 50]
```

Training task:

```text
randomly mask valid base IDs
predict q0 code id at masked valid positions
loss = cross_entropy(logits, target_base), valid masked positions only
```

Output:

```text
base_logits [B, 16, 50, 512]
```

### Residual Transformer Training Target

For each residual layer `q in {1,2,3}`:

```text
history_sum_q =
  codebook_0[indices[..., 0]]
  + ...
  + codebook_{q-1}[indices[..., q-1]]

target_res_q = indices[..., q]  # [B, 16, 50]
```

Training task:

```text
input = history_sum_q + q_layer_embedding + text/graph conditioning
predict target_res_q
loss = cross_entropy(logits, target_res_q), valid positions only
```

Recommended v1:

```text
one shared residual transformer
q_layer_embedding tells it which residual layer is being predicted
teacher-forcing history during training
sequential generation during inference
```

## Graph-Aware Backbone Structure

The spatial axis is not a fixed 2D grid. It is a padded graph of edge-segment
coarse slots. Therefore the backbone should use graph-aware spatial blocks:

```text
Token state x [B, 16, 50, D]

per layer:
  1. graph-spatial attention over C using pooled_adjacency / pooled_geodesic
  2. temporal attention over T for each coarse slot
  3. text conditioning
  4. FFN
  5. re-mask with token_mask / coarse_mask / frame_mask_lat
```

This can reuse design patterns from:

- `src/models/graph_salad/attention.py::GraphAttentionBlock`
- `src/models/motion_decoder.py::TemporalSelfAttention`
- `src/models/graph_salad/denoiser.py` text conditioning patterns

We should not copy MoGenTS fixed 2D positional encoding directly. Instead:

- temporal position embedding for `T_lat = 16`
- coarse slot / pooled skeleton embedding for `C`
- optional graph-distance bias from `pooled_geodesic`
- optional segment/root type embedding if later needed

## Text Conditioning

Use the stronger dual text pathway by default for this future backbone:

```text
global text embedding:
  caption_emb [B, 768] -> global additive / FiLM conditioning

token-level text embedding:
  caption_token_emb [B, L, 768] + mask -> cross-attention
```

Reason:

- The old diffusion backbone showed weak text disambiguation.
- Token generation needs action details to choose the right discrete motion
  family.
- We already have token-level caption infrastructure in the repo.

CFG-style condition dropout can be used in training:

```text
with probability p_drop:
  drop both global text and token text together
```

## Masking Strategy

Base transformer masking should be topology-aware:

1. Never mask or train on padded slots:

```text
valid = token_mask
```

2. Use mixed temporal-spatial masking:

```text
temporal frame mask:
  mask whole latent frames for some samples

spatial slot mask:
  mask subsets of valid coarse slots inside unmasked frames

random token mask:
  mask scattered valid positions
```

3. Inference starts from all-mask valid positions:

```text
ids = MASK for token_mask=True
ids = PAD  for token_mask=False
```

Then run iterative MaskGIT-style refinement:

```text
for step in 1..N:
  predict logits at masked valid positions
  sample / choose code ids
  keep high-confidence ids
  re-mask low-confidence ids
```

Initial `N` can be 10, matching the MoGenTS / MaskGIT-style pattern.

Residual transformer can be simpler in v1:

```text
for q in 1..3:
  predict all valid residual positions in parallel
```

If residual quality is poor, later add iterative residual refinement too.

## Inference Flow

Given a skeleton and prompt:

```text
1. Build skeleton-only coarse graph metadata:
   pooled_adjacency, pooled_geodesic, coarse_mask, pooled_skeleton_embeddings,
   assignment, frame_mask_lat.

2. Base generation:
   base_ids [B, 16, 50] = BaseGraphMaskTransformer.generate(...)

3. Residual generation:
   full_ids = [base_ids]
   for q in 1..3:
     res_q = ResidualGraphTransformer.generate(full_ids_so_far, q, ...)
     append res_q

4. Full token map:
   indices [B, 16, 50, 4]

5. Frozen Graph-VQVAE decode:
   indices -> z_q -> pred_motion [B, 64, J, 13]
```

The decode path must respect:

- `coarse_mask`
- `token_mask`
- `frame_mask_lat`
- assignment from target skeleton to coarse slots

## v1 Model Count

Recommended v1 trains two token-generator models:

```text
1. BaseGraphMaskTransformer
2. SharedResidualGraphTransformer
```

Do not train four separate models:

```text
q0 model
q1 model
q2 model
q3 model
```

Reasons:

- More checkpoints and more training workflows increase operational cost.
- Residual layers are related; `q_layer_embedding` should be enough.
- We first need a minimal working token-generation baseline before adding
  fragmentation.

## Why Not 1D GPT First

Rejected v1 baseline:

```text
flatten [16, 50, 4] -> sequence length 3200
train autoregressive GPT
```

Reason:

- It is slow at inference.
- It treats time, graph slot, and RVQ depth as the same kind of axis.
- It creates long-range dependencies that are artificial.
- It does not use pooled graph metadata, weakening the arbitrary-topology claim.

A 1D flatten baseline can be a later ablation, not the primary design.

## Evaluation Plan

Backbone evaluation must use both metrics and visual QA.

Sanity gates:

- token overfit on a tiny split
- base-token accuracy on held-out real tokens
- residual-token accuracy per layer
- full-token decode finite and no padded-slot leakage
- generated IDs contain no invalid code IDs except `PAD=-1` on padded positions

Motion QA:

- decode generated tokens with frozen Graph-VQVAE
- render GT-vs-generated GIFs
- check frozen/static collapse
- check high-energy / low-energy actions
- check long-chain species and dense topologies

Metrics:

- token CE / accuracy for q0 and q1-q3
- reconstructed decoded motion losses through frozen decoder
- world / FK / traj diagnostics after decode
- evaluator metrics later, after AnyTopo evaluator is ready

Visual QA remains the decisive early gate.

## Minimal Implementation Milestones

### M0: Token Dataset Export

Export frozen tokenizer outputs:

```text
indices [N, 16, 50, 4]
token_mask [N, 16, 50]
coarse graph metadata
caption pointers / caption embeddings
decode metadata
```

Gate:

- strict shape check
- no train/val leakage
- no invalid IDs on valid positions
- padded IDs exactly `-1`

### M1: Base Graph Mask Transformer

Train q0 generator:

```text
input: partially masked q0 map [B,16,50]
condition: text + pooled graph
target: q0 [B,16,50]
loss: masked CE over valid masked positions
```

Gate:

- tiny overfit
- held-out token accuracy above random
- generated q0 decodes to non-garbage when residuals are copied from GT

### M2: Shared Residual Graph Transformer

Train q1-q3 generator:

```text
input: history_sum + q_layer_embedding
target: q_i
loss: CE over valid positions
```

Gate:

- per-layer residual accuracy above random
- generated full IDs decode without collapse

### M3: End-to-End Text-to-Motion Token Generation

Run:

```text
text + skeleton -> q0 -> q1-q3 -> frozen VQ decode -> motion GIF
```

Gate:

- visual QA on L5 dense/long-chain/high-energy samples
- compare against current latent diffusion backbone qualitatively

### M4: Ablations

Only after M1-M3 work:

- base/residual split vs one-shot `[T,C,Q]` prediction
- graph-spatial attention vs plain spatial attention
- token-level text vs mean text
- optional global/root branch if root drift is a visible failure

## Open Decisions

1. Codebook embedding source for residual transformer:

```text
Option A: read frozen quantizer codebook buffers directly.
Option B: train separate ID embeddings initialized from codebook vectors.
```

Default: use frozen codebook vectors projected into transformer dim. This keeps
residual generation aligned with the decoder's latent geometry.

2. Whether to add a 1D/global branch:

Default: not in v1. Add only if generated samples show root/global drift that
the coarse-slot tokens cannot handle.

3. Whether residual generation should use iterative MaskGIT refinement:

Default: no. Predict residual layers in parallel per layer first. Add iterative
residual refinement if residual visual quality is poor.

4. Motion length:

Default for L5 v1 is fixed 64 frames, therefore fixed `T_lat=16`. Variable
length can come later.

## Current Recommendation

The next backbone after Graph-VQVAE should be:

```text
Graph-aware MaskGIT-style token generator
  with:
    BaseGraphMaskTransformer for q0
    SharedResidualGraphTransformer for q1-q3
    graph-spatial + temporal attention
    dual text conditioning
    frozen Graph-VQVAE decoder
```

This preserves the main idea from MoGenTS while making the spatial axis
arbitrary-topology aware.

