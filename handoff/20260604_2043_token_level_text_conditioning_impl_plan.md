# Token-Level Text Conditioning Plan for Graph-SALAD Diffusion

Date: 2026-06-04 20:43 BST.

Status: implementation plan only. No code is changed by this document.

Goal: upgrade the Phase-2 T2M denoiser from mean-pooled T5 additive conditioning
to optional token-level T5 cross-attention, while preserving the current
`mean_additive` path as the default and keeping the change small enough to review
and ablate cleanly.

## 0. Decision Summary

Implement a new optional text mode:

```text
text_mode = "mean_additive"      # current default, unchanged behavior
text_mode = "token_cross_attn"   # new experiment
```

Do not change:

- VAE architecture or VAE checkpoint loading.
- latent diffusion objective: DDIM scheduler + v_prediction + masked MSE.
- graph/spatial attention over pooled slots.
- temporal attention over latent frames.
- current mean-pooled cache / current denoiser checkpoints.

Only add:

- token-level T5 cache sidecar;
- optional token fields in dataset/batch;
- optional text cross-attention block in `GraphSaladDenoiserLayer`;
- train/render script args to route the new fields.

## 1. Reference Code To Reuse Conceptually

This section is intentionally concrete so the implementation does not become a
from-scratch exploration.

### 1.1 PRISM: token embeddings as `encoder_hidden_states`

Reference files:

- `/tmp/PRISM/prism/pipelines/prism_t2m_pipeline.py`
- `/tmp/PRISM/prism/pipelines/prism_tp2m_pipeline.py`
- `/tmp/PRISM/prism/models/transformers/motion_prism/transformer_prism.py`
- `/tmp/PRISM/prism/models/transformers/motion_prism/block_with_mask.py`

What to take:

1. Encode text as token-level hidden states, not one pooled vector.
2. Pass the result to the denoiser/backbone as `encoder_hidden_states`.
3. Pass a separate text token mask for padding.
4. The cross-attention query is motion/latent tokens; keys and values are text tokens.

Concrete evidence:

- `prism_t2m_pipeline.py:340-355` tokenizes prompt with `padding="max_length"`,
  `return_attention_mask=True`, then calls the T5 text encoder and keeps
  `last_hidden_state`.
- `prism_t2m_pipeline.py:171-176` calls the transformer with
  `encoder_hidden_states=prompt_embeds`.
- `transformer_prism.py:245-253` documents text hidden states as
  `[B, N_ctx, text_dim]` plus `encoder_hidden_states_mask`.
- `transformer_prism.py:354-368` converts text mask `[B, N_ctx]` to an attention
  bias for cross-attention.
- `block_with_mask.py:237-248` performs cross-attention with motion tokens as
  queries and `encoder_hidden_states` as key/value source.

What not to take:

- Flow matching.
- per-token timestep / clean prefix conditioning.
- Wan block / 2D patch embedding / 2D RoPE.
- UMT5-XXL dependency.

Those are larger PRISM backbone changes and should remain separate ablations.

### 1.2 SALAD: small cross-attention block inside a skip-transformer

Reference files:

- `outside_docs/SALAD/models/denoiser/model.py`
- `outside_docs/SALAD/models/denoiser/transformer.py`

What to take:

1. Keep graph/spatial and temporal blocks, then add text cross-attention as a
   separate sub-block.
2. Flatten motion tokens to `[B, T*J, D]` for text cross-attention.
3. Use `key_padding_mask` on text tokens.
4. Apply FiLM after the cross-attention output.

Concrete evidence:

- `model.py:99-107` encodes text into token embeddings and projects them to model
  dimension.
- `transformer.py:173-181` implements `x query, memory key/value` cross-attention.
- `transformer.py:217-224` reshapes `[B,T,J,D] -> [B,T*J,D]`, applies cross-attn
  to text memory, reshapes back, FiLMs, and residual-adds.
- `transformer.py:255-305` shows the same cross-attention block repeated in the
  SALAD skip-transformer.

What not to take:

- CLIP online encoding in the training loop. Our project already committed to
  offline T5 cache for training scalability.
- SALAD fixed human skeleton assumptions.

## 2. Current Code Truth

Current implementation is still mean-pooled additive text conditioning.

### 2.1 Denoiser

File: `src/models/graph_salad/denoiser.py`

Current path:

```text
text [B,768]
  -> self.text_proj Linear(768,d_model)
  -> text_cond [B,D]
  -> every GraphSaladDenoiserLayer adds text_cond[:,None,None,:]
```

Evidence:

- `denoiser.py:15-16`: module docstring says mean-pooled T5 additive broadcast.
- `denoiser.py:135-136`: layer receives already projected `text_cond [B,D]`.
- `denoiser.py:177-180`: layer gates `text_cond` with `has_text` and adds it to
  every latent slot/frame.
- `denoiser.py:325-330`: top-level forward rejects token-level text and states
  `[B,n_tok,d_text]` is v2.
- `denoiser.py:386-387`: `text_cond = self.text_proj(text)`.

### 2.2 Dataset and cache

File: `src/data/anytop_dataset.py`

Current cache path:

```text
data/anytop_caption_t5_cleanL2_multi.npz
data/anytop_caption_t5_cleanL2_multi.embs.npy   [409970,768]
data/anytop_caption_t5_cleanL2_multi.keys.json  409970 keys
```

Current dataset behavior:

- `anytop_dataset.py:691-727` loads caption strings and keeps primary + multi
  caption lists.
- `anytop_dataset.py:760-789` loads mean-pooled caption embeddings, preferring
  `.embs.npy` + `.keys.json` sidecar.
- `anytop_dataset.py:1056-1077` randomly picks one caption during training and
  returns the matching mean-pooled embedding and caption string.
- `anytop_dataset.py:1121-1123` returns `caption_emb [768]` and `has_text`.

Current scale:

```text
clean_L2 motions:      81,994
caption embeddings:   409,970
avg captions/motion:  5.0
current split files:  train 77,892 / val 4,122
```

### 2.3 Batch schema

File: `src/models/graph_salad/batch.py`

Current optional text fields:

- `caption_emb: Optional[Tensor]` with shape `[B,768]`.
- `has_text: Optional[Tensor]` with shape `[B]`.

Evidence:

- `batch.py:137-138` fields.
- `batch.py:511-519` optional tensor validation includes only `caption_emb`.
- `batch.py:599-600` stores only `caption_emb` and `has_text`.

### 2.4 Train and render scripts

Files:

- `scripts/train_denoiser.py`
- `scripts/animate_denoiser.py`

Current train path:

- `train_denoiser.py:576-582`: CFG dropout modifies `has_text`, then zeroes
  `batch.caption_emb`.
- `train_denoiser.py:599-607`: passes `text=text_emb [B,768]`.
- `train_denoiser.py:471-474`: constructs `GraphSaladDenoiser(..., d_text=768)`.

Current sampling path:

- `animate_denoiser.py:64-68`: rebuilds denoiser with no `text_mode`.
- `animate_denoiser.py:111-115`: repeats `caption_emb [B,768]` for CFG.
- `animate_denoiser.py:122-129`: passes `text=text2 [2B,768]`.
- `animate_denoiser.py:229-250`: custom prompt path re-encodes T5 and mean-pools.

## 3. Proposed Minimal Design

### 3.1 Data representation

Add token cache files parallel to the existing mean-pooled cache:

```text
data/anytop_caption_t5_cleanL2_multi.tokens.npy      [N, L, 768] float16 or float32
data/anytop_caption_t5_cleanL2_multi.token_mask.npy  [N, L] bool
data/anytop_caption_t5_cleanL2_multi.keys.json       existing key list
```

Recommended first setting:

```text
L = 64
dtype on disk = float16
dtype in batch = float32 for current fp32 denoiser contract
```

Reason:

- Existing captions are short enough for `L=64`.
- `409,970 * 64 * 768 * fp16` is about 40 GB, large but manageable on project
  storage; float32 would be about 80 GB and is not worth it for a first pass.
- The current train path casts denoiser inputs to fp32; loader can cast token
  arrays to float32 on a per-batch basis.

Do not replace the existing mean-pooled sidecar. Keep both for ablation and
backward compatibility.

### 3.2 Dataset changes

Extend `AnyTopDataset` with optional args:

```python
caption_token_cache: str | Path | None = None
return_caption_tokens: bool = False
caption_token_max_len: int = 64
```

When `return_caption_tokens=False`, behavior must be identical to today.

When `return_caption_tokens=True`:

1. Load token sidecars using the same key grouping as mean-pooled cache.
2. Preserve the exact same random caption index selection as `caption_emb`.
3. Return:

```text
caption_token_emb   [L,768] float32
caption_token_mask  [L] bool
```

Important: the selected token row must match the selected caption string and
selected mean-pooled embedding. The random index cannot be sampled twice.

Implementation detail:

Current code at `anytop_dataset.py:1061-1068` samples `idx` inside `__getitem__`.
Use this same `idx` for all three:

```text
caption_emb
caption_token_emb
caption string
```

### 3.3 Batch changes

Add optional fields to `GraphMotionBatch`:

```python
caption_token_emb: Optional[torch.Tensor] = None   # [B,L,768]
caption_token_mask: Optional[torch.Tensor] = None  # [B,L] bool
```

Validation:

- `caption_token_emb`: rank 3, shape `[B,L,768]`, float32, finite.
- `caption_token_mask`: rank 2, shape `[B,L]`, bool, same device.
- If `caption_token_emb` exists, `caption_token_mask` must exist.
- If `caption_token_mask` exists, `caption_token_emb` must exist.

This is cleaner than reading raw dict keys in `train_denoiser.py`, because all
diffusion code already normalizes through `GraphMotionBatch.from_collate_dict`.

### 3.4 Model changes

Add a mode switch to `GraphSaladDenoiser`:

```python
text_mode: Literal["mean_additive", "token_cross_attn"] = "mean_additive"
text_token_dim: int = 768
```

Keep current constructor defaults equivalent to the current checkpoint surface.

For `mean_additive`:

- existing `self.text_proj = nn.Linear(768, d_model)`;
- existing layer call with `text_cond [B,D]`;
- existing strict load behavior for old checkpoints.

For `token_cross_attn`:

- add `self.text_token_proj = nn.Linear(768, d_model)`;
- add a small `TextCrossAttention` module to each `GraphSaladDenoiserLayer`;
- layer forward receives:

```text
text_tokens [B,L,D]
text_token_mask [B,L] bool
has_text [B] bool
```

Recommended cross-attention shape:

```text
x [B,T,C,D]
q = x.reshape(B, T*C, D)
kv = text_tokens [B,L,D]
key_padding_mask = ~(text_token_mask & has_text[:,None])
out = cross_attn(q, kv, kv, key_padding_mask)
x = x + out.reshape(B,T,C,D)
x = FiLM_after_text(x, t_emb)
x = x * coarse_mask * frame_mask
```

Use a local simple module, not PRISM `WanAttention`, to keep this patch small.
The SALAD `MultiheadAttention` implementation is the right conceptual template.

Recommended implementation:

```python
class TextCrossAttention(nn.Module):
    norm_q = LayerNorm(D)
    norm_kv = LayerNorm(D)
    q/k/v/o Linear
    softmax over text length
    key padding mask on text tokens
    all-masked rows return zero
```

Hard detail: CFG-uncond samples must contribute zero, not NaN. If a sample has
`has_text=False`, set its effective text mask to all False and explicitly zero
the cross-attn output for that sample after attention.

### 3.5 Forward signature

Keep the existing positional signature backward-compatible:

```python
def forward(
    z_t,
    timesteps,
    text,
    adjacency,
    geodesic_dist,
    coarse_mask,
    frame_mask,
    level2_meta=None,
    *,
    pooled_skeleton_embeddings=None,
    has_text=None,
    validate_inputs=False,
    text_token_mask=None,
):
```

Mode contracts:

```text
mean_additive:
  text must be [B,768]
  text_token_mask must be None

token_cross_attn:
  text must be [B,L,768]
  text_token_mask must be [B,L] bool
```

This avoids adding another positional argument and keeps old callers valid.

### 3.6 Training changes

Add args to `scripts/train_denoiser.py`:

```text
--text_mode {mean_additive,token_cross_attn} default=mean_additive
--caption_token_cache optional
--caption_token_max_len default=64
```

Construction:

- pass `return_caption_tokens=(text_mode=="token_cross_attn")` to `AnyTopDataset`.
- require `--caption_token_cache` when `text_mode=token_cross_attn`.
- construct `GraphSaladDenoiser(text_mode=args.text_mode, ...)`.

CFG:

For mean mode, keep current logic unchanged.

For token mode:

```python
has_text = batch.has_text & (~drop_mask)
text_tokens = batch.caption_token_emb
text_token_mask = batch.caption_token_mask & has_text[:, None]
```

Do not multiply token embeddings by zero as the primary gate; key masking is the
contract. Multiplying is acceptable as defense-in-depth, but mask must drive the
attention.

Checkpoint:

- save `text_mode`, `caption_token_cache`, `caption_token_max_len` in args.
- old checkpoints load with default `mean_additive`.
- token checkpoints strict-load only when reconstructed with token mode.

### 3.7 Sampling / animation changes

Files:

- `scripts/animate_denoiser.py`
- any newer generic T2M render scripts that call `GraphSaladDenoiser` directly.

Required changes:

1. Rebuild denoiser with `text_mode=da.get("text_mode","mean_additive")`.
2. If `text_mode=mean_additive`, keep current path.
3. If `text_mode=token_cross_attn`, load/return caption tokens from dataset and
   repeat both `caption_token_emb` and `caption_token_mask` for CFG.
4. For custom prompt / generic prompt, add token-level T5 encode helper that
   returns both:

```text
token_emb [1,L,768]
token_mask [1,L]
```

Do not mean-pool custom prompt in token mode.

CFG sampling in token mode:

```text
text2 = cat(tokens, tokens)
mask2 = cat(token_mask, zeros_like(token_mask))
has_text2 = cat(True, False)
```

The uncond branch should see no valid text keys.

## 4. Implementation Checklist

### M0: Token cache script

Create or extend a script:

```text
scripts/precompute_t5_caption_tokens.py
```

Inputs:

- `--texts_json data/anytop_planet_zoo_clean_L2/motion_texts_by_file_with_codex_drafts.json`
- `--out_prefix data/anytop_caption_t5_cleanL2_multi`
- `--max_length 64`
- `--dtype fp16`

Outputs:

- `.tokens.npy`
- `.token_mask.npy`
- `.keys.json` compatibility check against existing keys.

Smoke:

- `--limit 16`;
- verify shape `[16,64,768]`;
- verify mask has at least one valid token per caption;
- verify mean-pooling tokens with mask approximately matches existing `.embs.npy`
  for the same keys when both are float32-encoded from the same T5 model. With
  fp16 storage, allow a small tolerance.

### M1: Dataset + batch optional fields

Files:

- `src/data/anytop_dataset.py`
- `src/models/graph_salad/batch.py`

Add optional token loading and validation.

Smoke:

- `AnyTopDataset(..., return_caption_tokens=False)` returns exactly current keys
  and shape.
- token mode returns `caption_token_emb [64,768]` and mask `[64]`.
- random-caption selection keeps `caption`, `caption_emb`, and token cache index
  aligned.
- `GraphMotionBatch.from_collate_dict` accepts token fields and rejects missing
  mask / wrong dtype / wrong shape.

### M2: Denoiser text mode

File:

- `src/models/graph_salad/denoiser.py`

Add:

- `TextCrossAttention`;
- `text_mode`;
- optional token path in layer and top-level forward.

Smoke:

- mean mode old smoke still passes.
- token mode forward finite for `[B,T,C,D]` and `[B,L,768]`.
- `has_text=False` output differs from `has_text=True`, and no NaN appears for
  all-uncond batch.
- padded frame/slot outputs remain exactly zero.
- old checkpoint strict-load still works when `text_mode` defaults to
  `mean_additive`.

### M3: Train script wiring

File:

- `scripts/train_denoiser.py`

Add CLI args and route token fields.

Smoke:

- mean mode smoke produces identical behavior to current code for a fixed seed
  if no token args are passed.
- token mode smoke, 5 iterations, finite loss, gradients reach
  `text_token_proj` / cross-attn weights.
- DDP smoke true multi-rank if this will be launched on the 6-card path.

### M4: Sampling / visual QA wiring

Files:

- `scripts/animate_denoiser.py`
- any generic prompt scripts currently used for T2M QA.

Smoke:

- load mean checkpoint and render unchanged.
- load token checkpoint and sample with dataset caption.
- custom prompt in token mode uses token helper, not mean-pool helper.
- generate GIFs for:
  - simple locomotion captions;
  - multi-action captions;
  - slow / low-energy captions;
  - body-part-specific captions if present.

## 5. Experiment Setup

Baseline A:

```text
text_mode=mean_additive
same VAE
same denoiser size
same split
same caption set
same max_frames=260
```

Token B:

```text
text_mode=token_cross_attn
same everything else
```

Do not compare a token model with a larger layer count or different VAE in the
first ablation. The point is to isolate text conditioning.

If using current large backbone settings:

```text
d_model=512
d_ff=1536
n_heads=8
n_layers=11
max_coarse=128
T_lat=65
```

Expected added parameters:

Per layer text cross-attn is roughly:

```text
q/k/v/o: 4 * D * D
norms: small
```

For `D=512`, about `1.05M` parameters per layer. At `n_layers=11`, added params
are roughly `11.5M`, plus the token projection. This is acceptable relative to
the current ~63.5M denoiser, but activation memory also grows by attention
scores `[B, heads, T_lat*C, L]`.

Memory estimate:

```text
T_lat*C = 65*128 = 8320 motion queries
L = 64 text keys
heads = 8
attention scores per sample per layer = 8*8320*64 ≈ 4.26M
```

This is the main extra memory cost. If OOM:

1. enable gradient checkpointing per denoiser layer;
2. reduce per-GPU batch;
3. test cross-attn only in every other layer as a later ablation.

Do not start with every-other-layer; first implementation should be simple and
faithful.

## 6. Hard Requirements

1. `mean_additive` is the default and must remain load-compatible with existing
   denoiser checkpoints.
2. Token cache is offline. No online T5 inside training.
3. Token mode must keep random-caption train and primary-caption val behavior.
4. CFG-uncond token mode must not NaN on all-masked text.
5. VAE remains frozen and `use_text=False`.
6. No change to diffusion target/loss/scheduler in this ablation.
7. Visual QA is mandatory before accepting metric gains.

## 7. What This Can and Cannot Fix

Likely helps:

- multi-clause captions;
- word order / "then" phrases;
- body-part words when present in captions;
- generic "average action energy" behavior caused by weak text conditioning.

Does not directly fix:

- noisy or wrong captions;
- VAE reconstruction errors;
- pooling failures on long chains / wings / tails;
- low-speed bias if the dataset distribution lacks slow examples.

## 8. Review Gate

Before any full training launch:

1. codex review the token cache script and dataset/batch schema changes.
2. codex review denoiser token path and CFG all-masked behavior.
3. smoke mean mode for regression.
4. smoke token mode for finite loss and gradients.
5. render at least one token-mode sample GIF before long training if a checkpoint
   is available from smoke.

## 9. Implementation Prompt

Use this prompt for the implementation agent:

```text
Implement optional token-level T5 cross-attention for Graph-SALAD diffusion,
following handoff/20260604_2043_token_level_text_conditioning_impl_plan.md.

Hard constraints:
- Do not change VAE code or VAE checkpoint behavior.
- Do not change diffusion loss/scheduler/objective.
- Preserve current mean-pooled additive path as default:
  text_mode="mean_additive" must strict-load and run old checkpoints.
- Add text_mode="token_cross_attn" as a pure optional experiment.
- Use offline token cache sidecars; no online T5 in training.

Reference behavior:
- PRISM: token T5 hidden states are passed as encoder_hidden_states with a text
  mask.
- SALAD: motion tokens query text tokens via cross-attention inside each
  transformer layer.

Implementation order:
1. Add scripts/precompute_t5_caption_tokens.py producing
   .tokens.npy [N,L,768], .token_mask.npy [N,L], compatible with existing
   .keys.json order.
2. Extend AnyTopDataset with caption_token_cache / return_caption_tokens and
   return caption_token_emb + caption_token_mask aligned to the same randomly
   selected caption index as caption_emb.
3. Extend GraphMotionBatch optional validation for token fields.
4. Add TextCrossAttention and text_mode to src/models/graph_salad/denoiser.py.
5. Wire scripts/train_denoiser.py with --text_mode and --caption_token_cache.
6. Wire scripts/animate_denoiser.py so token checkpoints can sample and custom
   prompt uses token embeddings instead of mean-pooled embeddings.
7. Run smoke tests:
   - mean_additive regression smoke;
   - token cache limit smoke;
   - token dataloader/collate smoke;
   - token denoiser forward with has_text all False;
   - 5-iter token train smoke;
   - render smoke if a token checkpoint exists.
8. Save commands/results in a handoff report.

Do not launch a full training run until smoke and code review pass.
```

