# PRISM-Inspired VAE Plan for Long Chains, Wings, and Tails

Date: 2026-05-30
Scope: design report only. No code changes in this document.

## 0. Motivation

Recent visual QA still shows a recurring weak spot: long-chain or membrane-like structures can reconstruct with acceptable global speed, but local articulation looks under-modeled. The screenshot discussed on 2026-05-30 shows:

```text
PZ_Asian_Water_Monitor_Male clip0
J = 114, T = 36
pool_type = edge_segment
speed_ratio = 1.106
```

The speed ratio is not catastrophic, so this is not simply a "frozen prediction" case. The concern is more specific:

- long tails may lose traveling-wave phase along the chain;
- wings / fins / membrane structures may look like cloth or skirt bones with correlated but not identical local motion;
- segment-level pooling can preserve coarse displacement while smoothing fine per-joint phase differences.

The question is whether PRISM's VAE idea can help: it keeps one latent token per body joint instead of compressing an entire frame into one vector.

## 1. PRISM Takeaway

PRISM's relevant contribution is not skeleton pooling. It is almost the opposite:

```text
SMPL motion x [B, T, K, C]
  K = per-joint/body tokens

causal VAE

latent z [B, T_lat, K, D]
  T_lat = ceil(T / 4)
  K is preserved
```

Evidence from PRISM code:

- `/tmp/PRISM/prism/models/autoencoders/autoencoder_prism/autoencoder_kl_prism_2d.py:166-176`
  - `encode()` takes `[B,T,K,C]`, permutes to `[B,C,T,K]`, and returns latent over `[B,2*z_dim,T_lat,K]`.
- `/tmp/PRISM/prism/models/autoencoders/autoencoder_prism/autoencoder_kl_prism_2d.py:207-218`
  - `decode()` maps `[B,z_dim,T_lat,K]` back to `[B,T,K,C]`.
- `/tmp/PRISM/prism/pipelines/prism_t2m_pipeline.py:73-81`
  - latent shape is `[B, C, num_latent_frames, num_joints]`.
- `/tmp/PRISM/prism/models/transformers/motion_prism/transformer_prism.py:154-155`
  - asserts `patch_size[-1] == 1`, i.e. it does not patchify / pool over joints.
- `/tmp/PRISM/prism/models/autoencoders/autoencoder_prism/wan_causalconv.py:136-139`
  - causal conv does not stride or convolve over the joint axis.
- `/tmp/PRISM/prism/models/autoencoders/autoencoder_prism/wan_attention.py:163-208`
  - `WanKWiseAttention` mixes information across joint tokens, but does not merge them.

So the useful idea for us is:

```text
Do not destroy joint-token identity too early.
Use temporal compression, but preserve fine spatial tokens where fine articulation matters.
```

## 2. Current VAE Mechanics

Current `edge_segment` path:

```text
anytop_x [B,J,13,T]
  -> encoder
h0 [B,T,J,D]
  -> EdgeSegmentPool
z [B,T_lat,C,D]
  -> Gaussian latent
  -> coarse_xattn decoder
pred_motion [B,T,J,13]
```

Important code points:

- `GraphMotionVAE` supports `pool_type="edge_segment"` and `pool_type="none"` already:
  - `src/models/graph_salad/vae.py:99-103`
  - `src/models/graph_salad/vae.py:221-257`
- anytop13 encoder input:
  - `src/models/graph_salad/vae.py:375-386`
- edge-segment pool is called at:
  - `src/models/graph_salad/vae.py:419-443`
- no-pool path keeps `C == J` and only temporally pools:
  - `src/models/graph_salad/vae.py:444-484`
- decoder uses real assignment for `coarse_xattn` / `graph_temporal`:
  - `src/models/graph_salad/vae.py:629-650`
- graph-temporal decoder refinement exists as optional post-unpool fine-joint refinement:
  - `src/models/graph_salad/vae.py:685-702`

Current EdgeSegmentPool details:

- It builds chain segments of p=2 edges:
  - `src/models/graph_salad/pool_edge_segment.py:116-129`
- If segment count exceeds `max_coarse`, it greedily merges root-side adjacent segments in the longest chain:
  - `src/models/graph_salad/pool_edge_segment.py:132-165`
- It uses hard 1-of-K assignment:
  - `src/models/graph_salad/pool_edge_segment.py:309-356`
- It pools motion features by segment mean:
  - `src/models/graph_salad/pool_edge_segment.py:408-429`
- It temporally averages with `temporal_stride=4`:
  - `src/models/graph_salad/pool_edge_segment.py:423-428`

This means a fine joint's latent information is not preserved individually once it shares a segment with another joint:

```text
h_pool[c] = mean_j_in_segment(h[j])
```

That is a reasonable compression for rigid-ish local parts. It is risky for long chains where adjacent joints have phase-shifted motion.

## 3. Hypothesis for Long-Chain Failure

The likely failure is not that `edge_segment` loses the animal's global motion. It can still get speed roughly right. The failure is local phase/detail:

```text
tail/wing chain:
  j0 -> j1 -> j2 -> j3 -> j4 -> ...

traveling wave:
  phase(j0) != phase(j1) != phase(j2) ...

edge_segment p=2:
  [j0,j1], [j2,j3], ...

pooled latent:
  z_seg0 ~= average(j0,j1)
  z_seg1 ~= average(j2,j3)

decoder:
  each joint receives a segment-conditioned feature plus static skeleton embedding
```

This can reconstruct the average trajectory while flattening:

- tip timing;
- wave propagation;
- membrane fold / fan articulation;
- small alternating local rotations.

The problem becomes more visible when:

- chain length is high;
- motion is wave-like rather than rigid;
- adjacent joints are similar in rest geometry but different in phase;
- the segment count hits the `max_coarse` cap and overflow merge coarsens root-side chain segments;
- decoder is `coarse_xattn` and relies on pooled segment information.

## 4. Why PRISM Helps Conceptually

PRISM's per-joint latent does this:

```text
joint j has its own latent token z[:, :, j, :]
```

For long chains:

```text
z_tail_0, z_tail_1, z_tail_2, ...
```

can each carry different phase and amplitude. The model can still use joint-wise attention to coordinate them, but it does not need to reconstruct fine phase after averaging it away.

For us, a literal PRISM copy would be:

```text
pool_type = none
C = J
z [B,T_lat,J,D]
```

This already exists in `GraphMotionVAE`; the open question is whether it is tractable and whether it improves the specific failure cases.

## 5. Candidate VAE Improvement Branches

### Branch A: No-Code PRISM-Style Diagnostic

Run a VAE with:

```text
pool_type = none
feat_mode = anytop13
attn_mode = graphormer
decoder_mode = graph_temporal or coarse_xattn
max_joints = 144
temporal_stride = 4
```

Interpretation:

- If long tails/wings improve clearly, pooling is the bottleneck.
- If they do not improve, the bottleneck is more likely decoder/loss/data representation.

This is the first test because it uses existing code. It costs training time but avoids implementing a speculative pool.

Expected tradeoff:

- Better fine-chain preservation.
- Larger latent token count for diffusion later.
- No downstream latent compatibility with existing denoiser.

Important nuance:

- With `pool_type=none`, `C == J`.
- `coarse_xattn` uses identity assignment in this case, so it behaves like per-joint latent decoding.
- `graph_temporal` may be more valuable here than it was in earlier small-data runs because it gives fine-joint coordination after preserving per-joint tokens.

### Branch B: Hybrid PRISM-Segment Pool

New pool idea:

```text
protect high-risk chains as singleton joint tokens
pool low-risk body regions with edge segments
```

Rough segmentation:

```text
long chain / wing / tail:
  j0, j1, j2, j3, ...  -> singleton tokens

torso / rigid limb sections:
  p=2 edge segments or larger segments
```

This keeps the same external schema:

```text
assignment [B,J,C]
pooled_adjacency [B,C,C]
pooled_geodesic [B,C,C]
z [B,T_lat,C,D]
```

so the denoiser and decoder can still consume the usual pool metadata.

Possible protection rules:

1. Chain-length rule:
   - if a non-branching chain length >= L, make all joints on that chain singleton.
2. Leaf-distance rule:
   - protect the distal N joints near long tails / wing tips.
3. Degree/fan rule:
   - for fan-like membrane structures, protect sibling leaves or make small singleton groups instead of averaging.
4. Motion-prior rule:
   - compute per-joint velocity variance over a small calibration set; protect joints whose local motion variance differs strongly from neighbors.

Why this is attractive:

- It is less expensive than full `pool_type=none`.
- It directly targets dragon wing / long tail failure modes.
- It preserves downstream `[B,T,C,D]` interface if `C <= max_coarse`.

Risk:

- Rule design can become brittle.
- If too many joints are protected, it degenerates into no-pool anyway.
- If max_coarse is too low, overflow merge may undo the intended protection unless protected segments are marked non-mergeable.

### Branch C: Per-Joint Residual Detail Head

Keep current edge-segment latent, but add a fine-joint residual path:

```text
coarse z -> decoder -> coarse-conditioned fine feature
fine skeleton/motion skip -> residual head
pred = base + residual
```

This is less pure than PRISM. It lets the VAE reconstruct details, but diffusion still only models coarse latent unless the residual is deterministic from skeleton or encoded as a second latent.

Useful if:

- VAE reconstruction is the immediate bottleneck;
- diffusion does not need to model every local residual explicitly.

Risk:

- It may improve reconstruction but hide information outside `z`, weakening the diffusion stage.
- If the residual depends on input motion during encode/decode asymmetrically, sampling may not work.

For generation, avoid any residual path that requires GT motion at decode time.

### Branch D: Multi-Resolution Latent

Two latent grids:

```text
z_coarse [B,T_lat,C,D]    body / global structure
z_fine   [B,T_lat,J_f,Df] protected chains / wings / tails
```

Decoder consumes both:

```text
coarse_xattn(z_coarse) + fine_xattn(z_fine)
```

This is the most expressive but also the most invasive:

- VAE state dict changes.
- Denoiser must generate two latent tensors or a packed multi-level latent.
- Training/eval scripts need more changes.

This should be a later design if Branch A/B show clear evidence.

## 6. Recommended Order

### Step 1: Targeted QA Set

Create a small fixed visual suite:

- Dragon wing clips
- long tail reptiles
- monitor lizard / crocodile-like long-chain clips
- a few anti-regression short-chain animals

For each clip render:

- GT vs recon GIF
- multiple frames, not only frame 0
- local zoom or at least consistent camera for tail/wing
- speed ratio plus local-chain displacement ratio

Reason:

- The failure is local motion detail; global speed ratio is not enough.

### Step 2: Existing-Code Diagnostic

Train or at least overfit-smoke:

```text
VAE-A current:
  pool_type=edge_segment
  max_coarse=128
  decoder_mode=coarse_xattn

VAE-B PRISM diagnostic:
  pool_type=none
  decoder_mode=graph_temporal

Optional VAE-C:
  pool_type=none
  decoder_mode=coarse_xattn
```

Run on the same clean L2 data and same QA set.

Decision:

- If B improves long chains without harming normal animals, implement Branch B hybrid pool.
- If B only improves with graph_temporal, decoder coordination is the bottleneck.
- If B does not improve, revisit loss/data representation before pool design.

### Step 3: Hybrid Pool Prototype

Implement a new pool mode, not mutate `edge_segment`:

```text
pool_type = "hybrid_prism_segment"
```

Rules:

- build chains exactly like EdgeSegmentPool;
- mark protected chains/segments before p=2 grouping;
- protected joints become singleton segments;
- overflow merge cannot merge protected singleton joints;
- non-protected regions use current p=2 segments;
- if still over cap, fail loud or increase max_coarse rather than silently merging protected long-chain tokens.

Return contract should match EdgeSegmentPool exactly:

```text
assignment
hard_assignment
pooled_adjacency
pooled_geodesic
pooled_mask
pooled_skeleton_embeddings
anchor_indices
aux_losses
```

### Step 4: Loss/Metric Additions

Add targeted diagnostics before changing the main training objective:

- per-chain velocity L1;
- distal-chain displacement ratio;
- wave smoothness / phase preservation proxy;
- tip displacement error for long chains;
- wing/tail local visual QA.

Only add training loss terms if the diagnostics show current losses are blind.

Potential loss additions:

```text
long_chain_vel_loss:
  velocity L1 weighted higher for protected long-chain joints

tip_motion_loss:
  displacement magnitude error on distal leaf/tip joints

local_relative_motion_loss:
  error on (joint_pos[j] - joint_pos[parent[j]]) over time
```

Be careful:

- over-weighting tips can make bodies worse;
- any loss must be masked by valid joints and valid frames;
- anytop13 normalized-channel loss alone may not reflect world-space tail/wing motion.

## 7. What Not To Do First

Do not immediately replace the whole VAE with PRISM.

Reasons:

- PRISM assumes fixed SMPL topology.
- Our problem is variable topology and downstream diffusion conditioning.
- Our current code already has a no-pool per-joint latent diagnostic.

Do not immediately add multi-resolution diffusion.

Reasons:

- Too many moving pieces.
- If no-pool diagnostic does not help, multi-resolution latent will not fix the root cause.

Do not judge by aggregate val_recon alone.

Reasons:

- long-chain failures can be visually important but statistically small;
- many small face/end-effector joints can dominate count-based averages;
- global speed ratio can look fine while traveling-wave phase is wrong.

## 8. Success Criteria

For a VAE improvement to count as real:

1. Visual:
   - Dragon wing / long tail clips show less stiffness and less phase flattening.
   - No new collapse on normal quadrupeds / spiders / short-chain examples.

2. Local metrics:
   - distal-chain tip displacement error improves;
   - per-chain velocity error improves;
   - local parent-child relative motion improves.

3. Global metrics:
   - val_recon does not regress substantially;
   - speed_ratio stays close to 1.0.

4. Downstream feasibility:
   - latent shape and metadata can still be consumed by denoiser, or the denoiser changes are explicitly scoped.

## 9. My Recommendation

Run the no-code PRISM diagnostic first:

```text
pool_type=none
decoder_mode=graph_temporal
```

This is the cleanest test of the hypothesis:

```text
Does preserving per-joint latent tokens fix long-chain/wing/tail reconstruction?
```

If yes, implement `hybrid_prism_segment` as the practical version for arbitrary topology:

```text
per-joint latent where needed;
edge-segment compression where safe.
```

If no, focus on decoder/loss/data representation rather than pool design.

## 10. Implementation Prompt

Use this when asking an implementation agent to start the diagnostic phase:

```text
We want to test whether PRISM-style per-joint latent preservation helps noKslot_clean VAE reconstruction on long-chain, wing, and tail structures.

Read:
- handoff/20260530_2155_prism_inspired_vae_long_chain_plan.md
- src/models/graph_salad/vae.py
- src/models/graph_salad/pool_edge_segment.py
- scripts/train_graph_vae.py
- existing render QA scripts used for VAE GT-vs-recon GIFs

Do not implement a new pool yet.

First task:
1. Build a fixed QA list containing Dragon wing / long tail / reptile clips plus short-chain anti-regression examples.
2. Launch or prepare a VAE diagnostic run using existing code:
   - anytop_root = data/anytop_planet_zoo_clean_L2
   - feat_mode = anytop13
   - attn_mode = graphormer
   - pool_type = none
   - decoder_mode = graph_temporal
   - max_joints = 144
   - temporal_stride = 4
3. Compare against the current edge_segment/coarse_xattn VAE on the fixed QA set.
4. Report visual differences first, then local-chain metrics, then aggregate val_recon.

Hard constraints:
- Do not mutate edge_segment yet.
- Do not change denoiser.
- Do not judge only by aggregate metric.
- Render multi-frame GIFs for the target failure modes.

Decision after diagnostic:
- If per-joint latent improves long-chain/wing/tail details, design a new pool_type="hybrid_prism_segment".
- If it does not, investigate decoder/loss/data representation before pool changes.
```
