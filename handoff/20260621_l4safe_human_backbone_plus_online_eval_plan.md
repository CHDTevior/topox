# L4safe+HumanML3D Backbone Training + Online Evaluator Hook — Plan (for review)

Date: 2026-06-21
Two goals in one doc:
- **Part A** — prepare + train the Graph-CodeFlow **backbone** on `data/animo4d_anytop_clean_L4_safe_plus_humanml3d` (the animal+human dataset).
- **Part B** — hook the **frozen L4safeHuman evaluator** into backbone training for **online text→motion eval every N epochs**.

Mapped via a 5-reader workflow over: `train_graph_codeflow.py` + `flow.py`, the token-export pipeline, `_eval_codeflow_gen_in_evalspace.py`, the backbone launchers/watchdogs, and on-disk readiness.

---

## 0. Status, critical-path blocker, dependency order

**Assets that EXIST:**
- Frozen evaluator: `runs/anytop_t2m_evaluator_distilbert_coemb512_gb128_lr1e-4_mfd12_l4human_seed42/best_model.pt` (12ch contact-free; animal R@1 0.957 / human 0.576). ✓
- Caption cache: `data/anytop_caption_t5_l4safe_human_multi.*`. ✓
- eval_splits: `data/animo4d_anytop_clean_L4_safe_plus_humanml3d/eval_splits/{val_all,val_human,val_animal,...}.json`. ✓
- Frozen VQVAE candidates: **n8192** (`runs/vqvae_L4safeHuman_C72_J144_d512_Q4_n8192_b16g64_300ep_seed42`, still training ~ep112; ep100_model.pt saved) and **n4096** (just launched ~ep2, NO ckpt yet → not usable near-term).

**CRITICAL-PATH BLOCKER:** the **L4safeHuman token cache does NOT exist** (only `data/codeflow_tokens_mergedL4TB_*` and `_cleanL5_*`). The backbone trains on pre-exported frozen RVQ latents → **cannot launch until tokens are exported.**

**Dependency DAG (must be in order):**
```
freeze a VQVAE epoch  →  export token cache (num_frames=288)  →  pre-warm empirical_stats.pt
   →  backbone launcher+watchdog (codex)  →  smoke  →  train backbone
                                                          └→ (Part B) online eval hook (codex) lands BEFORE the real run
```

---

## PART A — Backbone training prep + launch

### A1. Freeze a VQVAE epoch (DECISION needed)
- The exported cache **bakes in the exact frozen latents** of whatever VQVAE ckpt you pick. `best_model.pt` keeps moving (overwrites as training improves to ep300), so **pin a specific `epNNN_model.pt`** for reproducibility.
- n8192 recon already plateaued (ep79: recon→GT R@1 0.969 / FID 0.0045). Options:
  - **(i) freeze a mid ckpt now (≥ep100, e.g. `ep100_model.pt` or the next ep-multiple)** → unblock the whole pipeline immediately. Defensible given plateau.
  - **(ii) wait for ep300** → cleanest final tokenizer, but ~1.5 days more.
- n4096 excluded near-term (no ckpt). If we later want an n4096 backbone, repeat A2–A6 with the n4096 ckpt into a separate cache.
- **Naming:** encode the chosen epoch in the cache dir (mirrors `*_ep199_*`).

### A2. Export token cache (the blocker)
**FRAME COUNT = 300 (USER DECISION 2026-06-21).** Correcting the earlier draft: `T_lat = ceil(num_frames / temporal_stride)`, it is NOT a `max_coarse × stride` hard limit. 300 frames → **T_lat = 75 = max_T_lat (default)**, which the CodeFlow net CAN express. 288 would merely be a deliberate clamp to T_lat=72 — we do NOT use it. **Use the full 300-frame training window** (matches the evaluator's stored num_frames=300, so no clamp/reconcile needed across export/backbone/online-eval). Clips longer than 300 (human up to T=469) are windowed/truncated to 300 as usual.

Script `scripts/export_graph_vq_tokens.py`. **MUST override caption caches** (defaults point at cleanL5):
```
--frozen_vqvae_ckpt runs/vqvae_L4safeHuman_C72_J144_d512_Q4_n8192_b16g64_300ep_seed42/ep<E>_model.pt
--out data/codeflow_tokens_L4safeHuman_n8192_ep<E>_fulllen300
--splits train,val --num_frames 300
--caption_emb_cache   data/anytop_caption_t5_l4safe_human_multi.npz
--caption_token_cache data/anytop_caption_t5_l4safe_human_multi
--min_text_coverage 0.99
```
(`anytop_root`/C72/J144/K8192 auto-resolve from the ckpt args.) Output ~94170 train + ~5190 val `.npz` (each: z_q[75,72,512], indices, masks, pooled graph fields, decode meta, caption_emb/token_emb baked in). Parallelize: **copy `scripts/_run_export_parallel_n2048.sh` → L4safeHuman variant** (set CKPT/OUT/CAP, keep inner `--num_frames 300`, ALLOCS=current idle jobids), run under **setsid nohup on a compute node** (the mergedL4TB export was once killed by a Slurm termination mid-run; disjoint `{i:06d}.npz` make partial re-runs safe). ~1–1.5h on ~10 GPUs; auto-runs `merge_export_shards.py`.
- **EPOCH POLICY (USER DECISION):** wait for n8192 **ep300** to freeze the FINAL tokenizer/cache for the real long backbone run. Use **ep100_model.pt for a 300-frame token-export SMOKE only** (validate the export+prewarm+backbone-smoke pipeline now while VQVAE finishes) — do NOT use ep100 as the final long-run tokenizer.
- Note: a 300-frame export at T_lat=75 is the first time we export at the max_T_lat cap exactly (75==75) — the export-smoke on ep100 also validates this.
- **Verify text coverage FIRST** on a `--max_clips` sample — exporter PREFLIGHT aborts if coverage < 0.99. HumanML3D captions should be fine (cache coverage was 100% earlier) but check before the full run; lower threshold only deliberately.

### A3. Pre-warm `empirical_stats.pt` (MANDATORY — the #1 footgun)
A real DDP launch against a cache WITHOUT a pre-warmed full-set `empirical_stats.pt` → rank-0 cold-scans ~40 min while peers wait at `dist.broadcast` → **NCCL ~10min watchdog → SIGABRT**. Before any DDP run:
```
single-process (NOT DDP): train_graph_codeflow.py --epochs 0 --empirical_stats_max_clips 0 \
  --token_cache data/codeflow_tokens_L4safeHuman_n8192_ep<E>_fulllen288 \
  --frozen_vqvae_ckpt <same n8192 ckpt> --out /tmp/prewarm_l4human --overwrite
```
Then `torch.load(<cache>/empirical_stats.pt)['count']` must be **≥10M** (the A100/H200 launcher abort-guard rejects <10M). ⚠ Do NOT smoke with `EMPIRICAL_MAX=256` against the real cache — it writes a 256-clip-keyed stats file the full run rejects → forces a DDP cold-scan anyway.

### A4. Backbone launcher config
No trainer code change to train on the new data (it only reads `--token_cache` + `--frozen_vqvae_ckpt`). Reuse a backbone launcher (`_launch_graph_pscf_2node_h200.sh` / `_2node_a100.sh` / 6card) with env OVERRIDES (all launcher defaults are mergedL4TB/C96/truebones = WRONG):
```
TOKEN_CACHE=data/codeflow_tokens_L4safeHuman_n8192_ep<E>_fulllen288
FROZEN_CKPT=runs/vqvae_L4safeHuman_C72_J144_d512_Q4_n8192_b16g64_300ep_seed42/ep<E>_model.pt
OUT=runs/codeflow_graph_pscf_L4safeHuman_n8192_b<B>_lr<LR>_seed42
EMPIRICAL_MAX=0   # hits the pre-warmed stats instantly
```
- Recipe (LOCKED defaults, prior): model_variant=graph_pscf (~287M), code_dim/hidden_size 512, d_ff 2048, depth_double6/single12, **batch 64-effective via per-GPU×WORLD (Goyal: lr scaled from global96=1.2e-4)**, epochs 600, half_cosine, warmup 2000, cond_drop 0.1, bf16, **max_T_lat 75 (≥72, no change)**, eval_cond_scale 4.0, eval_steps 50, seed 42.
- **C72 < C96** → the A100 B8 / H200 B16 ceilings relax slightly; re-smoke for a possibly larger B. K=8192 changes only the frozen codebook, not backbone D=512 → params unchanged.
- **Distinct** lock/pid names + RDZV_PORT (don't collide with any concurrent run).

### A5. Backbone auto-resume watchdog (CODE CHANGE → codex)
`_watchdog_h200_backbone.sh` / `_watchdog_a100_backbone.sh` **hardcode OUT_REL=mergedL4TB** AND their resume command **does NOT forward FROZEN_CKPT/TOKEN_CACHE** → an auto-resume would relaunch with the launcher's **wrong mergedL4TB/C96 defaults**, corrupting the run. Need a **copy parameterized like the VQVAE watchdog we already fixed**: OUT_REL + resume env (FROZEN_CKPT/TOKEN_CACHE/BATCH_SIZE/LR/OUT) + a fail-fast guard ("L4safeHuman run must use the humanml3d token cache + the n8192 ckpt"). Discover whitelist is hardware-specific (h200 flamingo0[12]/blossom0[1-4]; a100 swarma100[1-9]). → **codex review** (same pattern as the VQVAE watchdog patch).
- Alternatively: if the chosen alloc walltime > full run time, run **bare** (no watchdog) like the evaluator — simpler, no code change. Backbone is 600ep though (long) → likely needs the watchdog. **DECISION: hardware/walltime dependent.**

### A6. Smoke → launch (gated)
1. Token-cache preflight: trainer auto-checks `cache T_lat (72) ≤ max_T_lat (75)` (line 380) — passes.
2. 2-rank (or full-topology) smoke on the real cache: confirm DDP joins, flow_loss finite, empirical_stats hits the pre-warmed cache (no cold scan), z_q shape [.,72,72,512].
3. Launch real run.

---

## PART B — Online evaluator hook (text→motion, every N epochs)

### B1. Where it goes
`scripts/train_graph_codeflow.py` **do_val block (lines ~607-647)**: already `if do_val and is_main and dl_val is not None:` (RANK-0 only), bracketed by `raw_flow.eval()` (609) / `raw_flow.train()` (645), with the **single sync `dist.barrier()` at line 647** after. **Insert the online gen-eval inside this is_main block, before line 647.** Non-rank-0 ranks idle at that barrier while rank-0 evals → naturally DDP-safe. **Never add a barrier inside the rank-0-only block (deadlock).**

### B2. Shared helper (no drift)
Refactor `_eval_codeflow_gen_in_evalspace.py` main body (lines ~199-289) + metric fns (rprec_pool/pooled_rprec/fid/mean_pair_l2/subset_metrics) into a **shared module** e.g. `src/eval/gen_eval_in_evalspace.py: run_gen_eval(flow, tokenizer, evaluator, t5tok, t5, ds, idxs, *, steps, cfg_scale, gen_batch, num_frames, max_joints, pool, seed) -> report`. **Both** the offline CLI and the trainer import it → online eval == offline eval, guaranteed. (This refactor touches the offline script + new module → **codex**.)

### B3. New args (opt-in, threaded through launchers) — USER-DECIDED defaults
- `--gen_eval` (bool, opt-in; no hook unless set) + `--evaluator_ckpt <l4human eval best_model.pt>` + `--gen_eval_every N` (**default 50**) + `--gen_eval_n` (**default 256** strided; scale to 512 only after it's proven stable).
- Thread `EVALUATOR_CKPT` / `GEN_EVAL_EVERY` / `GEN_EVAL_N` through the launchers' COMMON_ENV (optional → existing mergedL4TB runs unaffected).
- These are TRAIN args (safe on resume); do NOT add to the arch-restore list.

### B4. What it does (rank-0)
Load frozen evaluator + T5 **ONCE at trainer startup (rank-0, eval/no_grad)** — not per-eval (it's hundreds of MB). Each gen-eval: T5-encode raw captions from `eval_splits/{val_animal,val_human}.json` → `tokenizer.prepare_skeleton_only` (+ **per-clip frame_mask_lat/token_mask overwrite** — the codex blocking fix) → `raw_flow.sample(cond, token_mask, T_lat, C, steps, cfg_scale)` → `tokenizer.decode(z_hat)['pred_motion']` continuous → `evaluator.encode_motion(gen)/encode_motion(GT)/encode_text(caps)` → **R@1/2/3 + matching + diversity, animal vs human separately**. Log to metrics.jsonl (+ optionally track best-by-animal-R@1).

### B5. Cost control
- Sparse cadence (`--gen_eval_every 30-50`), small `--gen_eval_n 256-512` (→ **FID auto-skipped at n<1024**, cheap; R-precision/matching/diversity only), `steps=25` (halve vs 50), `gen_batch=32`, CFG=4.0 (2 passes/step). Rank-0 only; the other ranks idle at the barrier so keep cadence sparse.

### B6. Footguns (must handle in the hook — all from the maps)
1. **RNG perturbation**: `flow.sample` uses global `torch.randn` → **save/restore `torch.get_rng_state()/cuda` around the eval**, else training dropout/noise shifts after every eval.
2. **Never crash training**: the offline place-into-gen-x HARD-FAIL (`decode T < gt_T`, lines 250-252) would SIGABRT the whole job → **wrap the hook in try/except + clamp gt_T to ≤288 (T_lat≤72)**; eval failure logs + skips, never aborts.
3. **num_frames = 300 everywhere (RESOLVED)**: export/backbone/online-eval all use 300 (T_lat=75=max_T_lat), matching the evaluator's stored num_frames=300 — no clamp/reconcile mismatch. The try/except (#2) still guards against any transient per-clip `decode T < gt_T` (e.g. a clip whose gt_T>300 if windowing ever changes), but with a uniform 300 window this should not fire.
4. **`raw_flow.sample` not `flow.sample`** (DDP wraps only forward()).
5. **has_text all-True** in the eval cond (CFG flips it internally); do NOT reuse a train-time CFG-dropped cond.
6. **animal/human split**: the offline script's `PZ_` prefix heuristic is WRONG for this dataset → use `val_animal.json`/`val_human.json`.
7. evaluator is **12ch** — `motion_feat_dim` auto-read from its ckpt; encode_motion slices `[...,:12]`. Don't pass a 13ch evaluator.

---

## PART C — Code changes requiring codex review
1. **Backbone watchdog** (copy + parameterize OUT_REL/resume-env + fail-fast guard) — A5.
2. **Shared gen-eval helper** refactor (offline script → importable module) — B2.
3. **Trainer hook** in `train_graph_codeflow.py` (load evaluator+T5 once; the do_val gen-eval block; new args; RNG save/restore; try/except) — B1-B6.
4. **Launcher arg threading** (`EVALUATOR_CKPT`/`GEN_EVAL_*` through COMMON_ENV) — B3.
(Token export + pre-warm + backbone-config are config/runs, NOT code changes — no codex, just smoke.)

---

## DECISIONS (RESOLVED 2026-06-21, user review)
1. **VQVAE epoch**: wait for n8192 **ep300** for the final tokenizer; use **ep100 for a token-export SMOKE only** (build/validate the pipeline now). ✓
2. **num_frames = 300** (full training window; T_lat=75=max_T_lat — the earlier "288=coarse budget" framing was wrong, corrected in A2). ✓
3. **Backbone hardware + auto-resume**: 600ep is long → **add the watchdog**, but parameterize + codex-review it FIRST. (Hardware/alloc still TBD — pick at launch time, ≥1.5d away while VQVAE finishes; watchdog must be parameterized for whichever node class.)
4. **Online eval**: **land the code BEFORE the backbone launch**; defaults `--gen_eval_every 50`, `--gen_eval_n 256`, **animal/human reported separately** (scale n→512 after stable). ✓
5. **n4096**: **keep bare run** (no watchdog), watch swarma1003 expiry (~7h margin). ✓

## EXECUTION ORDER (per user: fix first, then export, then launch)
1. ✅ Fix doc (288→300 framing, decisions baked in) — DONE.
2. Code fixes (each → codex): **(a) shared gen-eval helper** (refactor offline script: human/animal split via val_animal/val_human, try/except, no SystemExit, RNG-safe) → **(b) trainer hook** in train_graph_codeflow.py (load evaluator+T5 once, do_val block, new args, RNG save/restore) + launcher arg threading → **(c) backbone watchdog** (parameterize OUT_REL + resume FROZEN_CKPT/TOKEN_CACHE + fail-fast guard).
3. Token export (ep100 SMOKE @300 → ep300 FINAL @300) + empirical_stats prewarm (count≥10M) + backbone smoke.
4. Launch real backbone (with online eval already wired).

## Out of scope (unchanged)
The two VQVAE trainings (n8192 H200 + n4096 A100) continue independently. This plan does NOT touch them. Token export must use a *frozen pinned* VQVAE epoch, not the live tip.
