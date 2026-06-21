# L4safe+HumanML3D T2M Evaluator — Training Plan (for review)

Date: 2026-06-20
Dataset: `data/animo4d_anytop_clean_L4_safe_plus_humanml3d` (312 obj = 311 animal + 1 human shared skel 22 joints; train 94170 / val 5190; max actual joints 102, pad to J144; motion .npy = 13ch AnyTop).
Goal: train a NEW frozen two-tower text↔motion evaluator on THIS dataset, mirroring the prior contact-free 12ch evaluator (`runs/anytop_t2m_evaluator_distilbert_coemb512_gb128_lr1e-4_mfd12_seed42`, trained on L4safe+truebones), so we can score VQVAE recon + future CodeFlow generation on the human-inclusive distribution.

## 0. What carries over UNCHANGED (verified by code map)
- Trainer `scripts/train_anytop_t2m_evaluator.py` is **dataset-agnostic by CLI** — nothing about truebones is hardcoded inside the `.py`; only `--manifest / --data_root / --val_manifest` point at data. Hardcoded truebones lives in the **launchers** only.
- Model `AnyTopT2MEvaluator`: DistilBERT (frozen) text tower + graph-aware motion tower. **No per-species/object embedding** → generic over 312 types incl. the 22-joint human. `motion_feat_dim=12` drops contact ch12 **inside** `encode_motion` (input MUST stay raw 13ch; do NOT pre-strip).
- Loss = symmetric **false-negative-masked InfoNCE** (diagonal positive; off-diagonal same-`motion_id`/`source_motion_id`/`caption_text` removed from denominator). DDP grad-safe all-gather. Best ckpt by **val R@1**.
- Recipe to replicate: DistilBERT, coemb512, n_heads8, d_ff2048, n_graph6, n_temporal4, dropout0.1, learnable temp0.07, **mfd12**, **gb128 / lr1e-4**, epochs100, seed42, bf16, num_frames300, max_joints144.
- T5 mean cache `data/anytop_caption_t5_l4safe_human_multi.*` is **NOT used by the DistilBERT mainline** (only by the `--text_tower t5_cache` ablation). Built+coverage-verified; leave unused for the mainline.

## 1. BLOCKER — `eval_splits/` manifests do NOT exist for this dataset
The new dataset has only `splits/{train,val,test,all}.txt` (filename lists), NOT the rich `eval_splits/*.json` manifests the trainer requires (`--manifest=<root>/eval_splits/train_main.json`, `--val_manifest=<root>/eval_splits/val_all.json`). These must be built FIRST.

## 2. Phase 1 — build eval_splits manifests + preflight
Builder: `scripts/build_anytop_t2m_eval_splits.py`. **Defaults are stale truebones** (data_root + `--expect_train 71784 --expect_val 3808`) → hard-fail unless overridden.

Proposed command:
```bash
python3 scripts/build_anytop_t2m_eval_splits.py \
  --data_root data/animo4d_anytop_clean_L4_safe_plus_humanml3d \
  --expect_train 94170 --expect_val 5190 \
  --cap_json data/animo4d_anytop_clean_L4_safe_plus_humanml3d/motion_texts_by_file.json \
  --t5_keys data/anytop_caption_t5_l4safe_human_multi.keys.json
```
Outputs `eval_splits/{train_main,val_all,val_action_clean,val_action_overlap,val_animo4d,val_truebones,split_audit}.json` (each record: filename/motion_id/source_motion_id/source/object_type/captions/t5_keys/species_stripped_cap_idx/has_species_stripped).

Phase-1 preflight (fail-loud, must pass before training):
- (a) **caption coverage**: every train+val filename has a caption entry (caps JSON has 103748 entries ⊇ 99360 split → should be 100%).
- (b) **`--t5_keys` validation**: every emitted `{stem}__cap{i}` exists in the T5 keys.json (ensures manifest captions align with the cache; sanity even though mainline ignores T5).
- (c) **motion-file resolution**: ⚠ human clips are split across `motions/` AND `motions_heldout/`. Verify EVERY val/train manifest filename actually loads via the dataset (construct `AnyTopT2MEvalDataset` on val_all + iterate a few human `HML3D_*` rows). The prior run never hit motions_heldout; this is a NEW risk.
- (d) **caption-duplication check**: count near-duplicate `caption_text` within random val batches of 32 — HumanML3D has many similar captions; heavy duplication inflates the false-negative mask and can flatten the loss / distort R-precision. Report the duplication rate; if pathological, consider de-dup or larger val pool.

## 3. Phase 2 — smoke gate (before the real run)
1. Tiny-overfit: train on a tiny subset (e.g. `--max_steps` small) → retrieval should go near-perfect (sanity the loss can fit).
2. Single-GPU forward + **2-rank DDP** loss on the REAL new manifests with `--motion_feat_dim 12` → confirm: dataset builds from new eval_splits, anytop_x is 13ch + model slices to 12ch (no shape-guard fire), loss finite, DDP all-gather + false-neg mask runs, val gate produces R@1/2/3 + matching.
3. Caption-shuffle control: shuffle captions vs motions → retrieval must COLLAPSE (confirms the metric is real, not degenerate).
4. Confirm human rows participate (a batch with `HML3D_*` produces nonzero text/motion embeddings).

## 4. Phase 3 — train (recipe + hardware)
Run name (proposed): `runs/anytop_t2m_evaluator_distilbert_coemb512_gb128_lr1e-4_mfd12_l4human_seed42`
Recipe (= prior, NOT scaled — keep comparable + stable on the bigger set): DistilBERT, coemb512, **mfd12**, **gb128 / lr1e-4**, epochs100, seed42, bf16, num_frames300, max_joints144, warmup2000, wd1e-4, val_every5, FULL val (val_max_batches0), val_batch_size32.
Launcher: reuse `_launch_anytop_t2m_evaluator_crossalloc.sh` with env overrides `DATA=<human root>` + `OUT=<run name>` + fresh `JOB_A/JOB_B/NODE/RDZV_HOST/RDZV_PORT` (the defaults are stale 976839/976840/swarmh1002/10.6.15.69). MUST keep `MOTION_FEAT_DIM=12`.
Est. ~16h on 4×H100 (prior ~12h on ~72k; this is ~1.3× data). **Evaluator has NO built-in auto-resume** — see decision #5.

## 5. Acceptance gates (quality)
- Tiny-subset near-perfect retrieval ✓ (Phase 2).
- val_all R@1/2/3 ≫ random (random R@1≈1/poolsize=1/32≈3%); target R@1 ≈ prior's 0.96 ballpark.
- Shuffled captions → retrieval drops sharply ✓ (Phase 2).
- **Report subsets separately: animo4d (animal) vs HumanML3D (human)** R@1 — the human subset is the new thing we must validate the evaluator can discriminate.
- within-species / within-group retrieval reported.
- Metric is a TOOL, not a visual substitute (CV-primacy still governs the generator QA).

## 6. OPEN DECISIONS (need your call before I execute)
1. **Caption JSON source**: I built the T5 cache + VQVAE from `motion_texts_by_file.json`. The builder DEFAULTS to `motion_texts_by_file_with_codex_drafts.json` (both ~68MB). For evaluator↔backbone consistency I recommend `motion_texts_by_file.json` (same captions the backbone's tokens come from). Confirm — or is `_with_codex_drafts` the canonical/cleaner caption set? (I will diff them in preflight.)
2. **Human val split**: the builder's `source` filter is the literal substring `truebones` → on this dataset `val_truebones.json` is EMPTY and `val_animo4d.json == val_all.json` (humans get lumped into animo4d). To report **human-vs-animal R@1 separately** (recommended — human is the addition we care about), the builder needs a small code change (filter on `HumanML3D`/source tag → emit `val_human.json`). That is a code change → codex. Want it? (else humans are only visible in val_all aggregate.)
3. **species_stripped for human**: humans have no species to strip → `has_species_stripped=False` → with default `drop_uncovered_species_stripped=True` they are dropped from the species_stripped SANITY view (FULL view still includes them). Accept (sanity view = animals only) or handle specially?
4. **Hardware**: prior used 4×H100 cross-alloc. The H200s are busy (VQVAE). Free H100s: 977959(1)/977960(1)/976841(2). Use 4×H100 cross-alloc (2+1+1 or 2+2)? Or a single 4-GPU alloc if available? (resource decision = yours.)
5. **Auto-resume**: evaluator has none. If the chosen alloc walltime > ~18h, run bare (like prior). If not, I build a small auto-resume watchdog (codex'd) like the VQVAE one. Which?

## 7. Out of scope
No CodeFlow / backbone / token export. This task = build eval_splits + train the evaluator + acceptance gates. (The L4safeHuman VQVAE training continues independently.)
