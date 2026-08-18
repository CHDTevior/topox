# Graph-VQVAE Rest-FiLM Adaptation Plan

**Date**: 2026-06-25  
**Dataset**: `data/animo4d_anytop_clean_L4_safe_plus_humanml3d`  
**Goal**: add an optional rest-pose-conditioned FiLM path to the current Graph-VQVAE so the decoder can use skeleton geometry more actively, while preserving the existing assignment+skeleton-embedding routing and keeping the RVQ token semantics stable.

---

## 0. Executive Summary

Current Graph-VQVAE already uses skeleton/template information, but mostly through:

1. `skeleton_features + name_hashes + graph` -> `s_j[B,J,D]`;
2. `EdgeSegmentPool` -> `assignment[B,J,C]`;
3. `assignment + s_j` in the decoder;
4. `rest_offsets + parents` inside FK geometry loss.

It does **not** currently use rest-pose FiLM:

```text
gamma,beta = f(rest_pose)
h = h * (1 + gamma) + beta
```

This plan adds Rest-FiLM as an optional decoder-side conditioning mechanism:

```text
slot latent z_q
  -> unpool with assignment
  -> h_joint = unpool + s_j
  -> Rest-FiLM from skeleton/rest features
  -> MaskedMotionDecoder blocks
  -> pred_anytop13
```

The main experiment should be **decoder-side fine-joint Rest-FiLM**, not encoder-side Rest-FiLM. Encoder-side modulation is more likely to make RVQ codes skeleton-specific and weaken cross-topology reuse. Decoder-side FiLM keeps token meaning mostly motion-centric and lets the frozen/generated token be interpreted differently under different skeleton templates.

---

## 1. Current Code Contract

### 1.1 Dataset Contract

Dataset root:

```text
data/animo4d_anytop_clean_L4_safe_plus_humanml3d
```

Current dataset facts from `DATASET_INFO.md`:

```text
cond objects: 312 = 311 animal + 1 human
motions/: 99360 clips = 74522 animal + 24838 human train/val
motions_heldout/: 4388 human heldout clips
train: 94170
val: 5190
max actual joints: 102
HumanML3D joints: 22
EdgeSegmentPool max coarse slots with max_segments=72: 71
HumanML3D coarse slots: 12
max T: 469
```

Current VQVAE training uses padded:

```text
max_joints = 144
max_coarse = 72
max_frames = 64
d_model = 512
num_quantizers = 4
num_codes = 8192 for the L4safe+Human mainline
temporal_stride = 4
```

Dataset item exposes the fields we need:

```text
anytop_x                 [Jmax,13,T]      normalized motion
skeleton_features        [Jmax,9]
name_hashes              [Jmax]
adjacency                [Jmax,Jmax]
geodesic_dist            [Jmax,Jmax]
anytop_graph_dist        [Jmax,Jmax]
anytop_joint_relations   [Jmax,Jmax]
rest_offsets             [Jmax,3]
parent_indices           list[int]
anytop_tpos_first_frame  [Jmax,13]
anytop_mean/std          [Jmax,13]
joint_mask               [Jmax]
frame_mask               [T]
```

Relevant code:

- `src/data/anytop_dataset.py`: derives `skeleton_features` from `parents + offsets + joint_names`.
- `src/data/anytop_dataset.py`: pads `rest_offsets`, `name_hashes`, graph tensors, `anytop_tpos_first_frame`.
- `src/models/vq_model/graph_vq_tokenizer.py`: passes `skeleton_features/name_hashes/graph` into `SkeletonEncoder`.
- `src/models/vq_model/losses.py`: passes `rest_offsets` to `compute_world_rot6d_fk_terms`.

### 1.2 Current Model Contract

Current VQVAE information flow:

```text
anytop_x[B,J,13,T]
  -> permute
motion[B,T,J,13]
  -> SkeletonEncoder(... skeleton_features, name_hashes, graph ...)
h_fine[B,T,J,D]

skeleton_features/name_hashes/graph
  -> SkeletonEncoder.encode_skeleton
s_j[B,J,D]

h_fine + s_j + parents/graph
  -> EdgeSegmentPool
pooled_feat[B,T_lat,C,D]
assignment[B,J,C]
coarse graph[B,C,C]
pooled_skeleton_embeddings[B,C,D]

pooled_feat
  -> pre-VQ graph/temporal refine
  -> RVQ
z_q[B,T_lat,C,D], indices[B,T_lat,C,Q]

z_q
  -> post-VQ refine
  -> repeat_interleave
slot_feat[B,T,C,D]
  -> unpool with assignment
h0[B,T,J,D]
  -> + s_j
h_joint[B,T,J,D]
  -> MaskedMotionDecoder
  -> anytop13 heads
pred_motion[B,T,J,13]
```

Critical current decoder line:

```python
unpool_features = torch.einsum('bjk,btkd->btjd', assignment, slot_features)
joint_features = unpool_features + skeleton_embeddings.unsqueeze(1).expand(-1, T, -1, -1)
```

This means current skeleton conditioning is additive identity injection:

```text
dynamic coarse motion + fine-joint identity
```

Rest-FiLM would add feature-wise modulation:

```text
h = h * (1 + gamma(rest)) + beta(rest)
```

---

## 2. Research Claim

### Primary Claim

Decoder-side rest-pose FiLM improves topology-specific reconstruction of fine joint motion, especially for long chains, wings/tails, dense limb ends, and HumanML3D joints, without destroying RVQ token sharing.

### Supporting Claim

The useful place for stronger rest-pose conditioning is the decoder, not the encoder before quantization. Decoder-side conditioning helps interpret a shared motion code under different skeleton templates; encoder-side conditioning risks making RVQ codes skeleton-specific.

### Anti-Claim to Rule Out

Any improvement is not merely from adding parameters. The plan includes a parameter-matched or decoder-MLP-only sanity if the first result is ambiguous.

---

## 3. Method Design

## 3.1 Do Not Replace Assignment + `s_j`

Keep the current routing:

```text
assignment[B,J,C] : coarse slot -> fine joint routing
s_j[B,J,D]        : fine joint identity
```

Rest-FiLM is an additional conditioning layer, not a replacement.

Required invariant:

```text
Rest-FiLM must never remove:
  - assignment unpool
  - strict padded-slot key mask
  - s_j addition
  - FK/world/traj loss terms
```

## 3.2 Main Variant: `rest_film_mode=decoder_fine`

Add a small module owned by `src/models/vq_model/`:

```text
FineRestFiLM
inputs:
  s_j[B,J,D]
  skeleton_features[B,J,9]
  rest_offsets[B,J,3]
  optional anytop_tpos_first_frame[B,J,13]
  joint_mask[B,J]

outputs:
  gamma[B,J,D]
  beta[B,J,D]
```

Apply after initial unpool and `+s_j`:

```text
slot_feat[B,T,C,D]
assignment[B,J,C]
      │
      ▼
unpool_features[B,T,J,D]
      │
      │ + s_j[B,J,D]
      ▼
joint_features[B,T,J,D]
      │
      │ Rest-FiLM
      ▼
joint_features = joint_features * (1 + gamma[:,None]) + beta[:,None]
      │
      ▼
MaskedMotionDecoder cross-attn/FFN/temporal
```

Recommended implementation detail:

```python
gamma_beta = self.rest_film(rest_cond)  # [B,J,2D]
gamma, beta = gamma_beta.chunk(2, dim=-1)
gamma = gamma.tanh() * gamma_scale
joint_features = joint_features * (1.0 + gamma[:, None]) + beta[:, None]
```

But for stability, the last projection should be zero-initialized:

```text
init gamma=0, beta=0
=> new branch is exact no-op at initialization
```

This matters because we want a clean warm start / easy regression check.

## 3.3 Rest Condition Source

Use a conservative rest condition first:

```text
rest_cond_j = concat(
  s_j[B,J,D],                    # already graph/name/geometry encoded
  rest_proj(rest_offsets[B,J,3]),
  skel_proj(skeleton_features[B,J,9])
)
```

Then:

```text
rest_cond_j -> MLP -> gamma,beta[B,J,D]
```

Why include `s_j`?

`s_j` already contains:

```text
skeleton_features
joint name embedding
graphormer topology propagation
```

So `s_j` is the safest high-level rest/template embedding.

Why include raw `rest_offsets`?

It gives the FiLM branch direct access to bone direction/length in the template, useful for:

```text
long chain amplitude
wing/tail endpoint scaling
human limb geometry
FK-consistent local interpretation
```

Why not use full `anytop_tpos_first_frame` in v1?

`anytop_tpos_first_frame[J,13]` is available, but it is normalized 13ch and includes channels beyond static offsets. It may leak motion-normalization quirks into the FiLM branch. Keep it as optional v1.1 if decoder_fine is positive but weak.

## 3.4 Where to Put the Module

Recommended file structure:

```text
src/models/vq_model/rest_film.py
  - FineRestFiLM
  - CoarseRestFiLM later if needed

src/models/vq_model/masked_motion_decoder.py
  - add optional rest_film module / mode
  - keep default None / disabled

src/models/vq_model/graph_vq_tokenizer.py
  - add ctor args:
      rest_film_mode: str = "none"
      rest_film_hidden_mult: int = 2
      rest_film_gamma_scale: float = 0.1
      rest_film_use_offsets: bool = True
      rest_film_use_skeleton_features: bool = True
      rest_film_use_tpos: bool = False
  - pass batch.rest_offsets / skeleton_features into decoder when enabled
```

Default must be `rest_film_mode="none"` so all existing checkpoints and experiments remain unchanged.

## 3.5 Optional Later Variant: `decoder_layerwise`

If single-entry FiLM helps but not enough, apply Rest-FiLM inside every decoder cross-attn layer:

```text
for layer:
  joint_features = RestFiLM(joint_features)
  joint_features = slot cross-attn(...)
  joint_features = ffn(...)
```

This is stronger but less minimal. It should not be v1 because it changes every decoder layer.

## 3.6 Optional Later Variant: `coarse_postvq`

Use `pooled_skeleton_embeddings[B,C,D]` to FiLM the coarse slot features after RVQ:

```text
z_q[B,T_lat,C,D]
pooled_skeleton_embeddings[B,C,D]
      │
      ▼
coarse gamma,beta[B,C,D]
      │
      ▼
z_q = z_q * (1 + gamma[:,None]) + beta[:,None]
```

This may help CodeFlow because generation happens in `z_q` space. But it also changes the interpretation of the post-RVQ latent. It should be tested only after `decoder_fine`.

## 3.7 Do Not Do Initially: Encoder-Side Rest-FiLM

Avoid:

```text
h_fine before pool = RestFiLM(h_fine, rest)
pooled_feat before RVQ = RestFiLM(pooled_feat, rest)
```

Reason:

The RVQ codebook is supposed to represent reusable motion primitives. Strong skeleton-specific modulation before quantization may make the code indices depend too much on skeleton shape, which can hurt downstream CodeFlow generalization.

---

## 4. Tensor Flow for Proposed Main Variant

```text
DATA
──────────────────────────────────────────────────────────────
anytop_x[B,J,13,T]
skeleton_features[B,J,9]
name_hashes[B,J]
rest_offsets[B,J,3]
parents[B,J]
graph_dist / joint_relations[B,J,J]
joint_mask[B,J], frame_mask[B,T]


ENCODE
──────────────────────────────────────────────────────────────
anytop_x[B,J,13,T]
   -> motion[B,T,J,13]
   -> SkeletonEncoder(...)
   -> h_fine[B,T,J,D]

skeleton_features + name_hashes + graph
   -> encode_skeleton
   -> s_j[B,J,D]


POOL + RVQ
──────────────────────────────────────────────────────────────
h_fine + s_j + parents/graph
   -> EdgeSegmentPool
   -> pooled_feat[B,T_lat,C,D]
   -> pre_vq_layers
   -> z_e[B,T_lat,C,D]
   -> RVQ
   -> z_q[B,T_lat,C,D], indices[B,T_lat,C,Q]


DECODE WITH REST-FILM
──────────────────────────────────────────────────────────────
z_q[B,T_lat,C,D]
   -> post_vq_layers
   -> repeat_interleave
   -> slot_feat[B,T,C,D]

slot_feat + assignment[B,J,C]
   -> unpool
   -> h0[B,T,J,D]

h0 + s_j[:,None]
   -> h_joint[B,T,J,D]

rest condition:
  s_j[B,J,D]
  rest_offsets[B,J,3]
  skeleton_features[B,J,9]
      -> FineRestFiLM
      -> gamma,beta[B,J,D]

h_joint = h_joint * (1 + gamma[:,None]) + beta[:,None]
   -> MaskedMotionDecoder
   -> anytop13 heads
   -> pred_motion[B,T,J,13]
```

---

## 5. Training Plan

## 5.1 Baseline to Compare Against

Use current L4safe+Human mainline as baseline:

```text
data_root = data/animo4d_anytop_clean_L4_safe_plus_humanml3d
max_joints = 144
max_coarse = 72
max_frames = 64
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
num_quantizers = 4
num_codes = 8192
loss = compute_vq_loss_13ch
weights:
  pos=1.0, rot=1.0, vel=1.0, contact=0.1,
  world=0.25, fk=1.0, traj=0.10, commit=0.02
```

If the current baseline run is accepted as reference, do not retrain baseline unless code drift makes comparison ambiguous.

## 5.2 Main Rest-FiLM Run

Run ID:

```text
vqvae_L4safeHuman_restfilm_decoderFine_C72_J144_d512_Q4_n8192_300ep_seed42
```

Variant:

```text
rest_film_mode = decoder_fine
rest_film_gamma_scale = 0.1
rest_film_hidden_mult = 2
rest_film_use_offsets = true
rest_film_use_skeleton_features = true
rest_film_use_tpos = false
zero_init_last = true
```

Training recipe should mirror baseline:

```text
epochs = 300
bf16 = true
same max_frames=64
same global batch and lr policy as baseline
same human upsampling policy if active in baseline
```

Important: because the FiLM branch is zero-init no-op, this run should not need an aggressive LR reduction. If there is no warm-start, use the baseline LR. If warm-starting from a trained baseline checkpoint, use a short low-LR finetune first.

## 5.3 Two Valid Training Strategies

### Strategy A: Train From Scratch

Use when we want the cleanest paper comparison:

```text
same data
same seed
same epochs
baseline vs rest_film from scratch
```

Pros:

```text
cleanest comparison
FiLM can co-adapt with codebook
```

Cons:

```text
costs full 300ep
harder to know quickly if the branch helps
```

### Strategy B: Warm-Start From Existing Baseline

Use as fast probe:

```text
load baseline Graph-VQVAE checkpoint with strict=False
missing keys allowed only rest_film.*
freeze nothing
train 30-80 epochs at lower LR
```

Pros:

```text
fast answer
tests whether decoder conditioning can improve a trained tokenizer
```

Cons:

```text
codebook was trained without FiLM
may underestimate full benefit
```

Recommended order:

```text
1. Warm-start probe 50 epochs.
2. If recon/visual improves or does not regress, run full 300ep from scratch.
```

## 5.4 Loss

Keep the existing VQ loss unchanged:

```text
L_total =
  pos + rot + vel + 0.1 contact
  + 0.25 world
  + 1.0 fk
  + 0.10 traj
  + 0.02 commit
```

Do not add a special Rest-FiLM regularizer in v1.

Reason:

The branch is zero-init and bounded by `gamma_scale`. If it overpowers the model, this will show in reconstruction/visual/codebook metrics. Extra regularizers would complicate interpretation.

Optional logging only:

```text
mean_abs_gamma
mean_abs_beta
max_abs_gamma
max_abs_beta
```

These are diagnostics, not loss terms.

---

## 6. Validation and QA

## 6.1 Unit / Smoke Gates

Before real training:

1. `rest_film_mode=none` exact behavior:

```text
old checkpoint strict-loads
state_dict keys unchanged when mode none
forward output identical to current HEAD for same seed/input if no code path changed
```

2. `decoder_fine` construction:

```text
new keys only under rest_film / decoder rest modules
forward finite
loss finite
grad finite
z_q shape [B,T_lat,72,512]
indices shape [B,T_lat,72,4]
```

3. zero-init no-op:

At initialization:

```text
gamma == 0
beta == 0
decoder_fine output before training should match no-FiLM up to numerical tolerance
```

4. mask safety:

```text
padded joints have gamma/beta zeroed
padded frames remain zero in pred_motion
padded coarse slots still strictly masked
```

5. human batch gate:

Run a small batch containing HumanML3D and animal samples:

```text
gt_fk_mismatch logged
human samples do not NaN
rest_offsets finite
```

## 6.2 Training Metrics to Track

Existing:

```text
train/val total
pos, rot, vel, contact
world, fk, traj
commit
gt_fk_mismatch
perplexity / active codes per quantizer
root drift / jitter QA
```

Add:

```text
restfilm_gamma_abs_mean
restfilm_beta_abs_mean
restfilm_gamma_abs_max
restfilm_beta_abs_max
animal/human val split losses
long-chain / high-C object subset losses if available
```

Critical: report animal and human separately, because L4safe+Human has very different skeleton regimes.

## 6.3 Visual QA Must Be Primary

For this task, metric alone is not enough. Render GT vs reconstruction GIFs.

Required QA buckets:

```text
Animal dense topology:
  high-joint-count L4safe animals near max J / max C

Long chains:
  crocodile / monitor / seal / long tail-like examples if present

Wings or fan-like structures:
  any bird/wing-like or dense limb fan examples if present

Human:
  locomotion
  jump
  kick
  gesture / upper-body motion

Slow motion:
  idle / stand / slow walk

Fast motion:
  run / leap / sharp turn
```

Render at least:

```text
baseline recon vs rest_film recon vs GT
same clip, same camera, same frame support
```

Failure modes to inspect:

```text
wing/tail stiffness
long-chain traveling wave flattening
human arm/leg mismatch
root drift
jitter / high-frequency instability
over-smoothing
bone-length visual inconsistency
RVQ snap artifacts if later used by CodeFlow
```

## 6.4 Downstream Token Compatibility

After VQVAE training, export tokens and check:

```text
z_q reconstruction works
ids_to_embeddings(indices) == z_q from quantizer path within tolerance
token cache manifest uses max_coarse=72, D=512, Q=4
continuous-vs-snapped decode visual QA
```

Then train or finetune CodeFlow only if VQVAE recon is visually acceptable.

---

## 7. Experiment Blocks

## Block 1: Warm-Start Rest-FiLM Probe

**Claim tested**: Decoder-side Rest-FiLM can improve reconstruction using an already trained RVQ codebook.

Setup:

```text
baseline checkpoint: current L4safeHuman VQVAE best/selected epoch
variant: rest_film_mode=decoder_fine
load baseline strict=False, allow missing rest_film keys only
epochs: 50
lr: 0.2x to 0.5x baseline LR
data: L4safe+Human
```

Success criterion:

```text
no regression in val total/recon
visual improvement on hard buckets
no codebook collapse
gamma/beta non-zero but bounded
```

Failure interpretation:

```text
If no effect: either codebook already bottlenecked, or FiLM needs from-scratch co-adaptation.
If worse: decoder-side modulation destabilizes trained codebook interpretation.
```

Priority: MUST-RUN.

## Block 2: Full From-Scratch Rest-FiLM

**Claim tested**: Rest-FiLM improves the actual tokenizer when trained end-to-end with RVQ.

Setup:

```text
same baseline recipe
same data
same seed if possible
rest_film_mode=decoder_fine
epochs=300
```

Success criterion:

```text
recon visual equal or better overall
clear improvement on long-chain / dense topology / human buckets
code usage remains healthy
no animal regression from human adaptation
```

Priority: MUST-RUN if Block 1 is non-negative.

## Block 3: Parameter-Control Sanity

Only run if metrics improve but visual gains are unclear.

Variant:

```text
decoder_extra_mlp_no_rest
same parameter count roughly as Rest-FiLM
no rest_offsets/skeleton_features into gamma/beta
```

Purpose:

Rule out “just more parameters”.

Priority: NICE-TO-HAVE, not first-line.

## Block 4: Stronger Rest-FiLM Variants

Only run after `decoder_fine` is positive.

Candidates:

```text
decoder_layerwise:
  Rest-FiLM before every decoder cross-attn block

coarse_postvq:
  pooled_skeleton_embeddings -> gamma,beta for z_q/slot_feat

fine_plus_coarse:
  combine both
```

Priority: FUTURE.

## Block 5: Encoder-Side Rest-FiLM

Only run if decoder-side FiLM improves recon but CodeFlow still fails to use tokens well, and only as a deliberate ablation.

Priority: APPENDIX / FUTURE, not mainline.

---

## 8. Implementation Checklist

## 8.1 Files to Add

```text
src/models/vq_model/rest_film.py
```

Classes:

```text
FineRestFiLM
  input:
    s_j [B,J,D]
    rest_offsets [B,J,3]
    skeleton_features [B,J,9]
    joint_mask [B,J]
  output:
    gamma,beta [B,J,D]

zero_init_last_linear(module)
```

## 8.2 Files to Modify

```text
src/models/vq_model/masked_motion_decoder.py
```

Add optional args:

```python
rest_film: Optional[nn.Module] = None
rest_film_mode: str = "none"
```

Forward optional inputs:

```python
rest_offsets=None
skeleton_features=None
rest_extra=None
```

Apply after `unpool + s_j`.

```text
Do not change cross-attn assignment bias or coarse_mask behavior.
```

```text
src/models/vq_model/graph_vq_tokenizer.py
```

Add ctor args and pass batch fields into decoder.

Keep default:

```text
rest_film_mode = "none"
```

```text
scripts/train_graph_vqvae.py
```

Add CLI:

```text
--rest_film_mode {none,decoder_fine}
--rest_film_hidden_mult 2
--rest_film_gamma_scale 0.1
--rest_film_use_offsets
--rest_film_use_skeleton_features
--rest_film_use_tpos
```

Add logging:

```text
restfilm_gamma_abs_mean
restfilm_beta_abs_mean
```

Allow init from baseline with missing rest-film keys only if implementing warm-start.

```text
scripts/_launch_graph_vqvae*.sh
```

Pass env flags only when non-default.

## 8.3 Checkpoint Compatibility

Rules:

```text
mode=none:
  no new state_dict keys if possible, or no active new module.
  existing ckpts strict-load unchanged.

mode=decoder_fine:
  new rest_film keys expected.
  old ckpt warm-start allowed only with explicit flag and missing-key allowlist.
```

Recommended:

Instantiate Rest-FiLM module only when `rest_film_mode != "none"`. This keeps old mode state_dict clean.

---

## 9. Code Review Gates

Any implementation must pass:

1. Codex review with focus on:

```text
mask safety
zero-init no-op
state_dict compatibility
no assignment/coarse_mask regression
no unintentional encoder-side FiLM
```

2. Unit checks:

```text
mode none old ckpt strict-load
mode decoder_fine finite forward/loss
gamma/beta zero at init
padded joints gamma/beta zero
same output as no-FiLM at init within tolerance
```

3. Smoke training:

```text
2-rank smoke, 5-10 iterations
animal+human batch present
loss finite
grad finite
z_q/indices shapes correct
```

4. Visual smoke:

```text
render 4-6 recon GIFs:
  animal dense
  long chain
  human locomotion
  human gesture
```

---

## 10. Expected Outcomes

## 10.1 What Success Looks Like

```text
Visual:
  less stiffness in wings/tails/long chains
  better human limb detail
  fewer joint-specific reconstruction mistakes

Metrics:
  val pos/rot/fk/world modestly better or equal
  human val improves without animal regression
  code usage remains healthy
  commit does not spike

Downstream:
  token export works unchanged
  CodeFlow can still train on z_q/indices
```

## 10.2 What Failure Looks Like

```text
gamma/beta grows too large
val improves but visual gets jittery
codebook active codes drop
animal quality regresses while human improves
FK/world mismatch increases
CodeFlow snapped decode becomes worse
```

## 10.3 Decision Tree

```text
Warm-start improves visually:
  -> run full 300ep from scratch

Warm-start neutral, no regression:
  -> still consider full from-scratch if budget allows

Warm-start worse:
  -> do not full-train; inspect gamma/beta and maybe lower gamma_scale

Full train improves VQVAE but hurts CodeFlow:
  -> keep VQVAE result but test coarse_postvq or lower FiLM strength

Full train improves human but hurts animal:
  -> check human upsampling ratio and split losses; Rest-FiLM may be overfitting human template
```

---

## 11. Recommended Run Order

### M0: Implementation and Safety

```text
Add rest_film.py
Wire decoder_fine as optional branch
Add train args/logging
Run codex review
Run shape/unit smoke
```

Gate:

```text
mode none is unchanged
mode decoder_fine starts as no-op
```

### M1: Warm-Start Probe

```text
load current selected L4safeHuman VQVAE ckpt
rest_film_mode=decoder_fine
50 epochs
lower LR
render hard QA buckets
```

Gate:

```text
no visual regression
some hard-bucket improvement OR clear metric improvement
```

### M2: Full 300-Epoch Training

```text
from scratch
same baseline recipe
same dataset
same max_joints/max_coarse/max_frames
```

Gate:

```text
visual QA passes
animal/human split metrics not worse
codebook healthy
```

### M3: Token Export and CodeFlow Compatibility

```text
export full-length tokens T=300
verify RVQ identity
train short CodeFlow smoke
render continuous-vs-snapped decode
```

Gate:

```text
snapped decode quality not worse than baseline tokenizer
```

### M4: Optional Stronger Variants

Only if M2 is positive:

```text
decoder_layerwise
coarse_postvq
fine_plus_coarse
```

---

## 12. Minimal Prompt for Executor

```text
Implement optional decoder-side Rest-FiLM for Graph-VQVAE according to
handoff/20260625_rest_film_graph_vqvae_plan.md.

Hard requirements:
1. Default rest_film_mode="none" must preserve current behavior and old checkpoint strict-load.
2. Main new mode is rest_film_mode="decoder_fine".
3. Keep assignment unpool + s_j addition + strict padded-slot key mask unchanged.
4. Rest-FiLM is applied after unpool+s_j and before MaskedMotionDecoder refinement.
5. Use zero-init final projection so decoder_fine is an exact/no-near no-op at init.
6. Use rest condition from s_j + rest_offsets + skeleton_features. Do not use tpos in v1 unless explicitly enabled.
7. Add gamma/beta diagnostics but no new loss term.
8. Run unit checks, DDP smoke, and Codex review before any long training.
9. Dataset for experiments is data/animo4d_anytop_clean_L4_safe_plus_humanml3d.
10. Do not change CodeFlow or existing VQVAE baseline behavior.
```

---

## 13. Bottom Line

Recommended first real experiment:

```text
rest_film_mode = decoder_fine
zero-init gamma/beta
condition = s_j + rest_offsets + skeleton_features
warm-start 50ep from current L4safeHuman VQVAE
visual QA on animal dense / long-chain / human motion buckets
```

If this is not worse, run the full 300ep from scratch.

This is the lowest-risk way to test Rest-FiLM because it strengthens skeleton-specific decoding while preserving:

```text
RVQ codebook semantics
assignment-based coarse-to-fine routing
current FK/world geometry supervision
existing CodeFlow token interface
```

