# PRISM Diffusion Backbone Ideas for Later Iteration

Date: 2026-06-02 20:38 BST.

Purpose: record PRISM diffusion/backbone mechanisms that may be useful in later
Graph-SALAD Phase-2 iterations. This is a research note only; no current training
code is changed by this document.

References:
- Paper: PRISM: Streaming Human Motion Generation with Per-Joint Latent Decomposition, arXiv:2603.08590.
- Code: https://github.com/ZeyuLing/PRISM
- Inspected files: `prism/pipelines/prism_tp2m_pipeline.py`,
  `prism/pipelines/prism_ar_t2m_pipeline.py`,
  `prism/models/transformers/motion_prism/transformer_prism.py`.

## What PRISM Does Differently

### 1. Flow matching instead of DDIM/DDPM v-prediction

PRISM trains a DiT-style latent generator with flow matching:

```text
z1 ~ N(0, I)
z0 = normalized VAE latent
model predicts velocity field v_theta(z_t, t, text)
```

Inference uses `FlowMatchEulerDiscreteScheduler` with 1000 train timesteps,
50 Euler steps, and CFG scale 5.0 in the released pipeline.

This differs from our current `scripts/train_denoiser.py`, which uses:

```text
DDIMScheduler + prediction_type="v_prediction"
z_t = scheduler.add_noise(z0, noise, t)
loss = masked MSE(v_pred, v_target)
```

### 2. Per-token timestep / noise-free condition injection

This is PRISM's most relevant backbone-side idea.

Instead of assigning one scalar timestep to the whole latent sequence, PRISM lets
each latent token carry its own timestep. During training it randomly selects a
clean prefix of F frames:

```text
prefix tokens:     timestep = 0   (clean condition)
remaining tokens:  timestep = t>0 (denoising targets)
```

With probability 0.5, `F = 0`, so the same model also trains pure T2M.

Effect:
- `F = 0`: text-to-motion.
- `F > 0`: pose/prefix-conditioned generation.
- autoregressive streaming: previous segment tail is injected as clean prefix
  into the next segment.

Code evidence from PRISM inference:
- `condition[:, :, :1, :] = first_frame_latents`
- `first_frame_mask[:, :, :1, :] = 0`
- model input is `(1 - mask) * condition + mask * latents`
- token timestep is `(first_frame_mask[0][0] * t).flatten()`, so condition
  tokens get `t=0` and generated tokens get current denoising timestep.
- after each scheduler step, condition latents are force-restored so they remain
  noise-free.

### 3. Self-forcing for long autoregressive rollouts

PRISM notes that teacher-forced training conditions on ground-truth previous
segments, while streaming inference conditions on the model's own previous
outputs. This train-inference gap causes boundary drift and long-rollout
collapse.

Their fix is self-forcing:

```text
generate segment
decode through VAE
re-encode generated output
use it as the next segment's condition during training
```

The paper says Distribution Matching Distillation supplies the training signal
for these self-conditioned rollouts. In practice, this is a later-stage feature:
it matters most after prefix-conditioned generation already works.

### 4. 2D latent grid + 2D RoPE

PRISM keeps the latent as a joint-factorized 2D grid:

```text
[T_lat, J, D]
```

The DiT uses rotary position embeddings along both time and joint axes. The
released transformer accepts hidden states as `[B, C, T, J]` and flattens
`T x J` tokens after a 2D patch embedding.

This is conceptually close to our latent layout:

```text
z [B, T_lat, C, D]
```

but our `C` is pooled graph slots, not fixed human joints. If adopted, the joint
axis RoPE idea should become a coarse-slot/graph-position encoding, not a literal
joint-index RoPE.

## Difference from Our Current Phase-2 Denoiser

Current Graph-SALAD denoiser:

```text
frozen Graph-VAE encode -> z0 [B,T_lat,C,D]
sample one timestep per sample
DDIM v-prediction target
graph-conditioned denoiser:
  pooled_adjacency
  pooled_geodesic
  pooled_skeleton_embeddings
  mean-pooled T5 caption embedding
CFG via has_text dropout
```

PRISM backbone:

```text
VAE latent [B,C,T,J]
flow matching velocity target
per-token timestep
clean prefix condition tokens
token-level T5-XXL cross-attention
2D time/joint RoPE
self-forcing for streaming
```

The main overlap is the structured latent grid. The main mismatch is that PRISM
has fixed human joints, while our topology-dependent coarse slots need graph-aware
conditioning and masks.

## What Is Worth Considering Later

Priority 1: per-token timestep + clean prefix condition

This is the cleanest transplant. It would let us train one any-topology denoiser
for:
- pure T2M (`F=0`);
- first-frame or prefix-conditioned motion generation (`F>0`);
- future long-motion chaining.

Implementation sketch for our setting:

```text
z0 [B,T_lat,C,D]
choose prefix length F_lat per sample
condition = z0 prefix, zeros elsewhere
mask = 0 for prefix, 1 for generated region
z_t = condition * (1-mask) + noisy_z * mask
timestep_grid [B,T_lat,C]:
  0 for prefix tokens
  t for generated tokens
denoiser must accept per-token timestep, not only [B]
loss only on generated/noised valid tokens
```

Open issue: our current denoiser timestep MLP assumes `timesteps [B]`. This would
need a real architectural change and codex review.

Priority 2: token-level text cross-attention

PRISM uses frozen T5 token embeddings with cross-attention. Our v1 uses mean-pooled
T5 `[768]` additive conditioning. Moving to token-level text is likely useful, but
it is separate from noise-free condition injection. It should be a controlled
ablation, not bundled with per-token timestep at first.

Priority 3: flow matching

Flow matching may be worth testing after the current DDIM/v-pred baseline is
stable. It is a larger scheduler/objective change and should not be mixed with
prefix-conditioning in the first iteration.

Priority 4: self-forcing

This is only useful after prefix-conditioned generation works. It is designed for
streaming / multi-segment rollout, not for the first plain T2M baseline.

## Recommended Future Experiment Order

1. Keep current T2M baseline unchanged until we have visual QA and curves.
2. Add a separate `condition_mode=prefix_clean` experiment:
   - keep DDIM/v-pred first;
   - add per-token timestep support;
   - train with `F=0` probability 0.5 and random clean prefix otherwise;
   - evaluate T2M plus first-frame/prefix-conditioned generation.
3. If prefix mode works, add autoregressive segment chaining inference.
4. Only then consider self-forcing training.
5. Flow matching and token-level T5 should be separate ablations.

## Do Not Overclaim

PRISM's published gains come from a combination of:
- joint-factorized latent VAE;
- FK-supervised causal VAE;
- flow-matching DiT;
- noise-free condition injection;
- self-forcing;
- much larger data/model scale.

For our project, the directly relevant idea is not "copy PRISM backbone wholesale".
The useful idea is:

```text
condition frames can be injected as clean latent tokens by giving each token its
own timestep.
```

That is the part most compatible with our any-topology graph latent design.
