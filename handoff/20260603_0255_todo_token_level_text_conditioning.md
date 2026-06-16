# TODO: Token-Level Text Conditioning for Graph-SALAD Diffusion

Date: 2026-06-03 02:55 BST.

Status: deferred follow-up. Do not mix this into the current B-VAE diffusion
launch. This document records a future upgrade path.

## Current Text Conditioning

Current `GraphSaladDenoiser` text path:

```text
caption string
  -> T5-base embedding cache
  -> mean-pooled vector [768]
  -> text_proj Linear(768, d_model)
  -> additive broadcast into every denoiser layer
```

Relevant current code:

```text
src/models/graph_salad/denoiser.py
  text input: [B,768]
  text_proj: Linear(768, d_model)
  per-layer text_additive gated by has_text

scripts/train_denoiser.py
  batch.caption_emb [B,768]
  cond_drop_prob flips has_text False
  text_emb is zeroed for CFG-uncond samples
```

This is simple and works as a first version, but it compresses an entire caption
into one global vector.

## Problem

Mean-pooled text loses token-level structure:

```text
"walk forward, then turn left and raise the front legs"
```

After mean pooling, the denoiser receives only one global semantic vector. It has
weak access to:

- word order;
- action composition;
- temporal sequencing such as "then";
- local action phrases;
- which body part a phrase refers to.

This can make T2M look like "average action energy" rather than precise caption
following, especially for multi-clause or fine-control captions.

## PRISM Comparison

PRISM uses token-level T5-XXL text embeddings with cross-attention:

```text
motion latent tokens query text tokens
text tokens provide key/value context
```

So each motion token can attend to different words or phrases. This is a more
complete use of text than our current mean-pooled additive conditioning.

PRISM also combines this with a much larger DiT, flow matching, per-token
timestep, and noise-free condition injection. This TODO is only about the text
conditioning part.

## Proposed Upgrade

Upgrade text conditioning from:

```text
text [B,768] -> additive broadcast
```

to:

```text
text_tokens [B,L,768]
text_mask   [B,L]
motion/query tokens [B,T_lat,C,D]
  -> cross-attention to text_tokens
```

The denoiser layer order could become:

```text
spatial graph attention
FiLM(t)
temporal attention
FiLM(t)
text cross-attention
FiLM(t)
re-mask padded frames/slots
```

This preserves the current graph-aware spatial and temporal structure while
giving the model access to full caption tokens.

## Required Data Change

The current cache stores one vector per caption:

```text
caption_emb [768]
```

Future cache should store token embeddings:

```text
caption_token_emb [L,768]
caption_token_mask [L]
```

Implementation options:

1. Fixed max token length, e.g. `L=64` or `L=128`, pad/truncate.
2. Store token embeddings in a sidecar `.embs.npy` with shape `[N,L,768]`.
3. Store masks in sidecar `.mask.npy` or compact JSON/np array.
4. Keep old mean-pooled cache for backward compatibility and ablations.

Train split behavior should stay the same:

```text
train: random caption among each motion's captions
val: primary caption only
CFG: cond_drop_prob flips has_text False
```

For CFG-uncond samples, either pass zero token embeddings with `text_mask=False`,
or pass a learned/null text embedding. Start simple with zero + mask.

## Required Model Change

Add a text cross-attention block to `GraphSaladDenoiserLayer`.

Sketch:

```text
x: [B,T,C,D]
q = motion tokens flattened to [B,T*C,D]
k,v = projected text tokens [B,L,D]
mask text padding and CFG-uncond text
out = cross_attn(q,k,v)
x = x + out.reshape(B,T,C,D)
```

Important details:

- Re-mask `x` after every layer, as current code already does.
- Keep `coarse_mask` and `frame_mask` out of text keys; they mask motion tokens,
  not text tokens.
- For CFG-uncond, text mask should make the cross-attention contribute zero.
- Keep existing additive text path as `text_mode="mean_additive"` for regression.
- Add new mode `text_mode="token_cross_attn"` for this upgrade.

## Suggested Experiment Order

Do not combine this with the current B-VAE diffusion launch.

Recommended order:

1. Finish B-VAE diffusion baseline with current text path.
2. Render T2M visual QA and identify whether failures are text-following failures.
3. Implement token-level caption cache.
4. Add `text_mode` switch:

```text
mean_additive      # current default
token_cross_attn   # new experiment
```

5. Smoke:
   - cache coverage;
   - batch collation shapes;
   - CFG-uncond behavior;
   - forward finite;
   - DDP smoke.
6. Train A/B with same VAE and same denoiser size:

```text
A: mean_additive
B: token_cross_attn
```

7. Evaluate with visual QA:
   - simple captions;
   - multi-action captions;
   - body-part-specific captions;
   - slow/low-energy captions.

## Risks

- Token cache is much larger than mean-pooled cache.
- Cross-attention increases activation memory.
- Caption quality may still be the bottleneck if captions are generic or noisy.
- Token-level text may improve semantic control without improving motion realism;
  VAE/pooling quality remains a separate bottleneck.

## Decision

Record as future work:

```text
Upgrade diffusion text conditioning from mean-pooled additive T5 to token-level
T5 cross-attention.
```

Priority: high after the current B-VAE diffusion baseline produces visual QA.

Do not implement until the current B-VAE diffusion run has a clean baseline.
