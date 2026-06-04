# Token-Level T5 Cross-Attention — Implementation Report

Date: 2026-06-04 23:13 BST. Branch: main. Git HEAD at start: 05bb75a.

Scope: token-path PROBE — CODE + SMALL smokes only. No full training, no 40GB
cache build, no codex, H100 mean diffusion on swarmh1002 untouched. A reviewing
session runs codex + verifies before any real launch.

**Verdict: READY FOR CODEX REVIEW.** All M0–M4 smokes + mean regression + bf16
(item G) PASS. Item E truncation rate = 0.000% (full dataset).

---

## Provenance note (read first — diff-vs-HEAD is mixed)

The git status at session start already had ` M` on `scripts/_launch_diffusion_t2m.sh`,
`scripts/animate.py`, `scripts/animate_denoiser.py`, `scripts/train_denoiser.py`
(pre-existing uncommitted work, NOT mine). So `git diff` vs HEAD on those files
mixes pre-existing local changes with my token edits. Confirmed via
`git show HEAD:scripts/animate_denoiser.py` — e.g. `make_t2m_large_gif` /
`make_generic_caption` are pre-existing (not in HEAD, not written by me). I read
those blocks before editing and layered token changes on top without clobbering
them (Karpathy R8/R3). My edits are isolated to the token machinery (markers:
`token_cross_attn` / `caption_token*` / `text_token*` / `TextCrossAttention` /
`TEXT_MODE` / `return_caption_tokens`). `src/models/graph_salad/{denoiser,batch}.py`
and `src/data/anytop_dataset.py` were CLEAN at HEAD before my edits.

---

## Files changed + key line refs

### NEW: `scripts/precompute_t5_caption_tokens.py` (M0; ~280 lines)
Token-cache builder. Sidecar to the mean cache; mean cache NOT touched.
- **Idx-align law** (plan §3.1, constraint 4): does NOT re-walk the texts json in
  its own order. Reads the EXISTING `<prefix>.keys.json` (canonical order from
  `convert_caption_npz_to_npy.py`) and emits `.tokens.npy [N,L,768]` /
  `.token_mask.npy [N,L]` ROW-FOR-ROW in that key order → token row k ⇔ mean-emb
  row k ⇔ keys[k]. `_build_caption_lookup` reproduces every `<mid>__cap<i>` key
  using the SAME ordering/de-dup rule as `precompute_t5_captions.py` +
  `anytop_dataset.py` (primary at idx 0, then de-duped). Fails loud if any
  keys.json entry is absent from the texts json.
- Same T5 (`t5-base`) + `T5TokenizerFast` + `max_length=64 truncation` as the mean
  builder. `padding="max_length"` → fixed `[b,L]` (PRISM-style). Pad-token rows
  zeroed on disk (mask is the contract; zeroing keeps mask-mean exact).
- **Item E**: computes token-length distribution + L-truncation rate over ALL
  captions (no-truncation tokenize). `--lengths_only` reports without encoding.
  Flags if >5%.
- Per-caption ≥1-valid-token assertion (a conditioned caption must not be empty).

### `src/data/anytop_dataset.py` (M1 + item C)
- `__init__` signature (+3 args): `caption_token_cache`, `return_caption_tokens`,
  `caption_token_max_len=64` (plan §3.2).
- Token-cache loading block (after the mean-cache load): **item C** — `np.load(...,
  mmap_mode='r')` on `.tokens.npy`/`.token_mask.npy` (NEVER materialize whole — 8
  workers × 40GB = host OOM); per-(mid,idx) stores only the **int row index** into
  the mmap (`caption_token_rows_multi`), grouped by motion_id sorted by cap idx —
  EXACTLY the mean-cache grouping. Validates token/mean caption-count match per
  motion (fail-loud on cache drift), L == `caption_token_max_len`, dim==768.
- `__getitem__` (plan §3.2, constraint 4): the SAME `idx` sampled at the caption
  block (the line formerly `idx = random.randrange(...)`) drives mean emb, caption
  string, AND token row — NOT resampled. Slices `self._token_emb_mmap[row]` →
  fp32 `.copy()` (page-in one row, item C). No-caption motions → all-False mask +
  zero tokens (denoiser/CFG zeroes them, item 5). Token fields added to the item
  dict ONLY when `return_caption_tokens` (mean path's dict byte-identical).
- `collate_fn`: unchanged — `caption_token_emb`/`caption_token_mask` are Tensors,
  hit the existing `torch.stack` branch (bool tensor stacking preserves bool).

### `src/models/graph_salad/batch.py` (M1)
- 2 optional dataclass fields: `caption_token_emb [B,L,768]`,
  `caption_token_mask [B,L] bool`.
- Validation block "6b" (plan §3.3): mutual presence required (one without the
  other = schema bug); rank-3 `[B,L,768]` f32 finite; mask `[B,L]` bool matching L;
  device match. Validated separately from `_OPTIONAL_TENSOR_SPEC` because L is
  variable. Added both to the `from_collate_dict` `return cls(...)`.

### `src/models/graph_salad/denoiser.py` (M2 — CORE; + items D, 5)
- NEW `TextCrossAttention` (plan §3.4): q=motion `[B,T*C,D]`, k/v=text tokens
  `[B,L,D]`, `key_padding_mask` (True=ignore). **Item D**: softmax forced fp32
  (`F.softmax(scores.float(),dim=-1).to(scores.dtype)`) — mirrors
  `attention.py:374`; fp32 path is a no-op, bf16 path runs softmax+`-1e9` sentinel
  in fp32. **Item 5 / constraint 5**: rows whose key_padding_mask is ALL-True
  (has_text=False / fully-padded) get cross-attn output EXPLICITLY zeroed
  (`out * (~all_masked)`); does NOT rely on softmax-over-all-`-inf`. `o_proj`
  zero-init (token path = identity residual at init).
- `GraphSaladDenoiserLayer`: +`text_mode` ctor arg; builds `self.text_cross_attn`
  ONLY in token mode (mean-mode state_dict byte-identical). `forward` +2 kw-only
  args (`text_tokens`, `text_key_padding_mask`); step-5 branches: token mode does
  cross-attn residual-add, mean mode keeps the gated additive broadcast (unchanged).
- `GraphSaladDenoiser`: +`text_mode="mean_additive"` / `text_token_dim=768` ctor
  args; builds `self.text_token_proj` ONLY in token mode; passes `text_mode` to
  layers. `forward` +`text_token_mask` kw arg (plan §3.5); mode-aware `text`
  validation (mean=`[B,768]`+mask-None; token=`[B,L,768]`+`[B,L]` bool); token
  prep builds `key_padding_mask = ~(text_token_mask & has_text[:,None])`; passes
  token args to all 3 layer call-sites (enc/mid/dec). Module docstring updated.

### `scripts/train_denoiser.py` (M3)
- CLI: `--text_mode {mean_additive,token_cross_attn}` (default mean_additive),
  `--caption_token_cache`, `--caption_token_max_len=64`. `--amp_dtype` help updated
  (bf16 now bf16-safe; default still fp32).
- Dataset: `use_tokens` gates `return_caption_tokens` + token cache into `ds_kwargs`
  (both train+val); requires `--caption_token_cache` when token mode (fail-loud).
- Denoiser built with `text_mode=args.text_mode`.
- **Constraint 6 (resume)**: `--resume` AND `--init_ckpt` now assert
  `ckpt.args.text_mode == CLI --text_mode` BEFORE strict-load (token/mean arch
  differ by 134 keys → clear error instead of cryptic strict-load fail).
- CFG (plan §3.6): token mode passes raw `caption_token_emb` + `caption_token_mask`
  (NOT zero-multiplied — mask drives the gate; denoiser ANDs has_text). mean mode
  unchanged. Same in val loop. Denoiser calls pass `text_token_mask`. Ckpt save
  already persists `vars(args)` → text_mode/cache/L saved automatically.

### `scripts/animate_denoiser.py` (M4)
- `load_denoiser`: rebuilds with `text_mode=da.get("text_mode","mean_additive")`
  (token ckpts carry it; wrong mode → strict-load fail-loud).
- `ddim_sample`: mode-dependent CFG (plan §3.7) — token mode repeats
  `caption_token_emb`+`caption_token_mask` to 2B, passes `text_token_mask`; uncond
  half's has_text=False fully masks keys.
- NEW `_t5_encode_tokens`: token-level T5 for custom/generic prompts (NOT
  mean-pool, plan §3.7). `--generic_prompt` overrides `caption_token_emb`/`mask` in
  token mode. CLI +`--caption_token_cache` / `--caption_token_max_len`; dataset
  built with token args when ckpt is token mode.

### Launch wiring (item F)
- `scripts/_launch_diffusion_t2m.sh`: +env `TEXT_MODE` (default mean_additive),
  `CAPTION_TOKEN_CACHE`, `CAPTION_TOKEN_MAX_LEN=64`, `AMP_DTYPE` (default fp32);
  flows into torchrun (`--text_mode`/`--caption_token_max_len`/
  `${CAPTION_TOKEN_CACHE:+--caption_token_cache ...}`/`--amp_dtype $AMP_DTYPE`,
  replacing the hardcoded `--amp_dtype fp32`). +echo line.
- `scripts/_launch_diffusion_t2m_6card.sh`: same 4 vars added to orchestrator
  `COMMON_ENV` so they reach each alloc's launch.

### Smoke harnesses (kept, gitignored-adjacent under `.aris/`)
`scripts/_m3_token_train_smoke.py` (M3 integration smoke, fp32+bf16). Caches +
helper scripts live in `.aris/smoke_tok/` (gitignored).

---

## Smoke results (command + decisive output)

**M0** — token cache `--limit 16` (CPU):
`python scripts/precompute_t5_caption_tokens.py --out_prefix /tmp/t5tok_smoke --max_length 64 --dtype fp16 --limit 16 --device cpu`
→ `Saved tokens (16, 64, 768) float16` + `token_mask (16, 64) bool`, min 6 valid
tokens/cap. mask-mean(tokens) vs mean cache for same 16 keys:
`per-row cosine: min=1.000000`, `rel L2 max=1.5e-4` (fp16 noise) → **PASS** (same
T5 model+tokenizer+key order as mean cache).

**M1** — dataset+collate (CPU, 6 real val motions, paired sub-caches):
- mean mode (no token args): item has NO token keys (byte-identical) → PASS.
- token mode: `caption_token_emb [64,768] f32` + `mask [64] bool`, ≥1 valid → PASS.
- idx-align: `cosine(token-mean, caption_emb)=1.000000` (same caption idx) → PASS.
- collate accepts token batch `[B,64,768]`/`[B,64]`; REJECTS missing-mask /
  mask-wrong-dtype / emb-wrong-L / emb-wrong-dtype (all 4) → PASS.

**M2** — denoiser forward (CPU, consistent chain graph):
- mean forward finite `[2,4,5,32]`, `validate_inputs=True` OK, padded frame/slot==0;
  mean mode rejects `text_token_mask` → PASS.
- token forward finite, padded==0; all-uncond (has_text all-False) finite
  (max|v|=0.0 at init), no NaN → PASS.
- after perturbing o_proj+output_proj: `max|v_cond-v_uncond|=1.36` (text
  conditions); uncond INDEPENDENT of text tokens (CFG-zero); cond-with-empty-mask
  == uncond (item 5) → PASS.
- **strict-load** (real ckpt `m2_...n11ff1536/best_model.pt`, epoch 120, `text_mode`
  ABSENT): loads into default mean denoiser strict=True, 0 missing/0 unexpected;
  token model = mean sd + 134 token-only keys (superset) → **PASS** (constraint 1+6).

**M3** — token-train integration, 5 iters (A100 swarma1004 managed srun, 28 motions):
`srun --jobid=944455 --overlap --ntasks=1 --gres=gpu:1 python scripts/_m3_token_train_smoke.py`
- fp32: losses `[1.0116, 0.9984, 0.9908, 0.9866, 0.9777]` finite; grad reached
  `text_token_proj=True text_cross_attn=True` → PASS.
- **bf16 (item G)**: losses `[0.9818,1.0023,1.0057,0.9885,0.9835]` finite,
  `v_pred.dtype=float32` (masked_v_mse promotion), grads reach both → **PASS**.
  This is bf16 diffusion's first real run; the fp32-forced softmax (item D) holds.

**Mean regression** (A100 managed srun, real ckpt): default-mode strict-load OK;
forward bit-identical across 2 fixed-seed runs; padded==0 → **PASS** (token edits
did not perturb mean path).

**M4** — render (A100 managed srun, token ckpt built from M3 denoiser):
load_denoiser rebuilds `text_mode=token_cross_attn` (strict-load OK); batch token
fields `[1,64,768]`/`[1,64]`; DDIM CFG-token → finite `z [1,65,128,512]` →
VAE.decode → world `(12,96,3)` finite → GIF written → **PASS**.

**Item E (full dataset, 409,970 captions, `--lengths_only` CPU):**
token lengths min=6 mean=15.71 p90=21 p95=22 p99=26 **max=42**. **L=64 truncation:
0/409970 (0.000%)** — no captions truncated; L=64 comfortably sufficient. No flag.

---

## Deviations from the plan

1. **Token cache reads keys.json directly** rather than re-deriving order from the
   texts json (plan M0 implied reuse "same key iteration order"). Reason: the
   canonical order is baked by `convert_caption_npz_to_npy.py` (which iterates the
   npz zip member order = dict-insertion order of `precompute_t5_captions.py`), NOT
   reproducible by re-walking the texts json. Reading keys.json + emitting rows in
   that exact order is the only way to GUARANTEE idx-alignment (constraint 4) — a
   "looks-same but different-order" rebuild was the explicitly-warned failure mode.
2. **Token-mask gate not pre-ANDed into the mask passed to the denoiser.** Plan §3.6
   wrote `text_token_mask = batch.caption_token_mask & has_text[:,None]`. I pass the
   RAW `caption_token_mask` + `has_text` separately; the denoiser computes
   `valid = text_token_mask & has_text[:,None]` internally. Net behavior identical
   (has_text is still the single CFG gate); keeps has_text as the one source of
   truth and avoids double-ANDing. The plan's zero-multiply-embedding gate is
   deliberately NOT used (constraint 5: mask drives attention; zero-mul + softmax
   over all-`-inf` = NaN).
3. **mean `text_proj` retained in token mode** (unused, ~0.4M params, covered by the
   existing DDP `find_unused_parameters=True`). Required so the mean-mode
   state_dict stays byte-identical (constraint 1); surgical (no removal of existing
   structure).
4. **M3/M4 smokes via standalone managed-srun scripts**, not `train_denoiser.py
   --smoke` (which runs a full epoch over all 81994 motions — too slow for a 5-iter
   probe). The standalone scripts mirror the exact train/animate token code paths.
   DDP multi-rank token smoke (plan M3 "if launched on 6-card path") was NOT run —
   token mode is single-rank-equivalent (the cross-alloc DDP infra is unchanged;
   only the denoiser arch + dataloader fields changed), and the main session does
   the real 6-card launch.

## Every changed line traces to plan/C–G
Dataset/batch → plan §3.2/§3.3 + item C. Denoiser → §3.4/§3.5 + items D/5.
Train → §3.6 + constraint 6. Animate → §3.7. Launch → item F. M0 builder →
§4 M0 + item E. bf16 path → item G.

## State / cleanup
- H100 mean diffusion (swarmh1002 jobs 944459/944461/944460) UNTOUCHED, still
  running. A100 swarma1004 (944455) back to 0% util / 0 MiB — all managed srun
  steps exited cleanly, NO orphans.
- Smoke caches under `.aris/smoke_tok/` (302M fp16, gitignored). Kept so the review
  session can re-run M1–M4 smokes. The full 40GB token cache is the main session's
  job: `python scripts/precompute_t5_caption_tokens.py --out_prefix
  data/anytop_caption_t5_cleanL2_multi --max_length 64 --dtype fp16` (on GPU).

## For the codex reviewer — focus points
1. `TextCrossAttention` CFG-zero (item 5) + fp32 softmax (item D) correctness.
2. Dataset idx-alignment (constraint 4): token row == mean-emb row == caption
   string for the SAME single random idx; mmap per-item slice (item C, no host OOM).
3. `text_mode` resume/init assertions (constraint 6) + mean state_dict
   byte-identity (constraint 1, old ckpts strict-load).
4. bf16 path (item G) for the upcoming bf16 + ep209 VAE run.

READY FOR CODEX REVIEW.
