# AnyTop T2M Evaluator (VQ/CodeFlow revision) — impl progress: M0–M2 DONE, M3+smoke NEXT

**Produced:** 2026-06-16T21:35Z. Spec: `handoff/20260614_anytop_t2m_evaluator_vq_codeflow_revision.md`.
**Gate:** implement + smoke ONLY; user reviews BEFORE any training. Free cross-alloc H100s reserved
for the gated training; smoke runs on a spare GPU. All code → codex (gpt-5.5 xhigh, fresh thread) before smoke.

## STATE
- status: M0 (manifests) + M1 (dataset) + M2 (model) implemented & `py_compile`-clean; NOT codex-reviewed yet; NOT smoked.
- next-critical: M3 train-script DDP global multi-positive InfoNCE (grad-safe), then smoke harness, then codex, then run smoke.
- resource: spare swarmh1002 GPU for smoke. Two backbones (512 ep~193 / 2048 ep~76) + blossom03 H200 migration continue under existing monitoring.
- design decision (LOCKED): source_motion_id — truebones use the real JSON id (e.g. Alligator_0000, groups augmented clips); animo4d clips are 1:1 with source (verified 0 shared source_file) so source_motion_id := motion_id (correct, not degenerate). multi-positive mask = motion_id ∨ source_motion_id ∨ caption_text.

## DONE
**M0** `scripts/build_anytop_t2m_eval_splits.py` (rewritten, argparse, merged-default). Ran OK →
`data/animo4d_anytop_clean_L4_safe_plus_truebones/eval_splits/`: train_main=71784, val_all=3808,
val_action_clean=3800/overlap=8 (8 = truebones augmentations split across train/val; animo4d all clean by 1:1),
val_animo4d=3730, val_truebones=78, split_audit.json. Variable caption count (1..7), blacklist corrupt captions
[A,B,C,D] (0 motions lost), per-record fields add `source`+`object_type`. action counts computed (no stale L2 assert).
**M1** `src/data/anytop_t2m_eval_dataset.py`: `num_frames` default 260→300; `caption_emb_cache` now OPTIONAL
(None → DistilBERT path uses the always-attached raw `caption_text`; T5 cache loaded only when given); T5-key
validation gated on cache presence; __getitem__ attaches `source` metadata; raw `caption_text` is the DistilBERT input.
**M2** `src/models/graph_salad/t2m_evaluator.py`: added `DistilBertTextTower` (frozen distilbert-base-uncased from
checkpoints/text_encoders/, train()-override pins backbone eval, projection ReLU+Linear(768→coemb), learnable query
token + sinusoidal PE + nn.TransformerEncoder pool → text_emb[B,coemb]) + `_SinusoidalPositionalEncoding`. Evaluator
`__init__` now: coemb_dim 384→512, d_ff 1536→2048, n_graph_layers 4→6, n_temporal_layers 2→4; `text_tower=
'distilbert'(primary)|'t5_cache'(fallback)` dispatch; `encode_text(list[str]|Tensor)` + `forward(batch, text_input)`
dispatch. Vectorized `build_multi_positive_mask` (int-id hashing, was O(B²) Python loop). Motion tower unchanged
(fresh SkeletonEncoder, anytop13_split+graphormer, no VAE/VQVAE weights), now d_model=512.

## M3 (NEXT) — `scripts/train_anytop_t2m_evaluator.py`: DDP global multi-positive InfoNCE
Current run_step calls `model(batch, caption_emb)` + `build_multi_positive_mask` + `contrastive_loss` per-rank.
Required changes:
- run_step must pass the RAW CAPTIONS list: `batch["caption_text"]` (collate keeps string lists) → `model(batch, captions)`.
- DDP: init_process_group(nccl), DistributedSampler(**drop_last=True** — symmetric per-rank batch is REQUIRED so all_gather shapes match across ranks), wrap model in DDP. Frozen DistilBERT params produce no grad → use `find_unused_parameters=True` OR `static_graph=True` OR exclude the backbone (it's already requires_grad=False; DDP still registers it — find_unused_parameters=True is the safe default).
- GLOBAL-batch InfoNCE (spec §6 — the load-bearing requirement): gather text_emb, motion_emb across ranks with **`torch.distributed.nn.functional.all_gather`** (GRAD-SAFE — plain `torch.distributed.all_gather` DETACHES grad → silently trains on local-only negatives, the exact weakness the spec warns about; ⚠ RISK #1). Gather metadata (motion_id/source_motion_id/caption_text lists) via `all_gather_object` PRESERVING rank order (concat in rank order to align with the gathered embedding rows). Build the multi-positive mask on the GLOBAL [WB,WB]; compute symmetric_infonce on the global logits. fp32 logits/CE under bf16 encoder autocast (mirror commit ef1ed84).
- schedule: AdamW, warmup 2000 + cosine decay, grad_clip 1.0, weight_decay 1e-6..1e-4, lr 2e-4@gb256. New defaults: coemb_dim=512, text_tower='distilbert', global_batch 256 (smoke much smaller).
- launcher draft `scripts/_launch_anytop_t2m_evaluator.sh` (torchrun; NOT executed — gated).

## SMOKE (NEXT) — update `scripts/_smoke_anytop_t2m_evaluator.py` (+ optional DDP smoke)
4 gates (spec §6.2): (1) tiny-overfit → near-perfect retrieval on a tiny subset; (2) single-GPU forward (B≤8, bf16
encoder + fp32 logits, distilbert path); (3) DDP 2-rank loss — assert loss finite + **grads actually flow to the
local slice** (the grad-safe all_gather check) + identical loss across ranks; (4) caption-shuffle → retrieval drops.
⚠ Update the stale `_EXPECTED_NO_GRAD` set (now includes ALL frozen DistilBERT params) + the param-count band
(10-20M is stale; recompute exact at impl time, fail-loud — don't guess). transformers must import + distilbert
loadable from checkpoints/text_encoders/distilbert-base-uncased (confirmed on disk). HF offline env already set in tower.

## RISKS (from survey, carry forward)
1. grad-detach all_gather (use torch.distributed.nn.functional.all_gather) — correctness footgun, smoke gate 3 must assert grad flow.
2. frozen DistilBERT must stay eval (dropout off) — train()-override implemented in DistilBertTextTower; verify model.train() doesn't flip it.
3. DistributedSampler drop_last=True (ragged final batch desyncs all_gather → NCCL hang).
4. fp32 logits/CE (bf16 CE underflows with sharp logit_scale).
5. metadata all_gather_object must preserve rank order (else mislabeled positives).
6. merged motions/*.npy are symlinks to source roots — must stay mounted at load.

## CODE POINTERS
M0 scripts/build_anytop_t2m_eval_splits.py · M1 src/data/anytop_t2m_eval_dataset.py · M2 src/models/graph_salad/t2m_evaluator.py
M3 scripts/train_anytop_t2m_evaluator.py · smoke scripts/_smoke_anytop_t2m_evaluator.py
motion tower SkeletonEncoder src/models/encoder.py (anytop13_split/graphormer; forward args incl graph_dist+joint_relations)
MotionMillion ref outside_docs/MotionMillion-Codes/ (text tower + eval_trans.py metrics; do NOT copy nfeats=272 motion tower)
Survey output (full evidence): the wf_bd563adf survey result (existing-code map / MM extract / graph API / manifests).
