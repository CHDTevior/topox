# AnyTop T2M Evaluator — training-flow walkthrough (for review)

Human-perspective pass through DATA → MODEL → TRAINING, with key code line numbers,
the launch script, and hyperparameters. Spec: handoff/20260614_anytop_t2m_evaluator_vq_codeflow_revision.md.
Status: M0–M3 + smoke DONE & codex-PASSED (20/20 single-proc + DDP gate); training GATED on this review.

## 0. What this model is (one paragraph)
An INDEPENDENT, frozen "measuring instrument": a two-tower text↔motion contrastive model.
Text tower = frozen DistilBERT + trainable head → text_emb[512]. Motion tower = a FRESH
graph-aware SkeletonEncoder over AnyTop 13ch motion → motion_emb[512]. Trained on REAL
motion-caption pairs with symmetric multi-positive InfoNCE. After training+freezing, its
embedding space gives R-precision / matching / FID / diversity for VQVAE recon and CodeFlow
generations. It shares NO weights with the generator and never sees z/z_q/RVQ or generated samples.

## 1. DATA
### 1a. M0 manifests — scripts/build_anytop_t2m_eval_splits.py
- `main()` :47 — reads splits/{train,val}.txt + caption JSON; hard-asserts train=71784/val=3808 + disjoint (:75).
- `build_record()` :89 — per motion: variable-length captions, drops corrupt strings via `is_corrupt_caption()` :42 (blacklisted A/B/C/D; 0 motions lost), **source_motion_id = real(truebones) | motion_id(animo4d, 1:1)** :104, `source` from source_dataset :106, t5_keys (ablation only), species_stripped idx.
- writes train_main / val_all / val_action_clean / val_action_overlap / val_animo4d / val_truebones / split_audit :145–:174.
- RAN & VERIFIED: train 71784, val 3808 (clean 3800 / overlap 8 = truebones augmentations; animo4d all clean), animo4d 3730 / truebones 78.

### 1b. M1 dataset — src/data/anytop_t2m_eval_dataset.py
- `AnyTopT2MEvalDataset` :140, `__init__` :167 (num_frames default **300** :174).
- THIN WRAPPER: instantiates the REAL `AnyTopDataset` for ALL preprocessing (FK reorder / mean-std norm / graph fields / masks / 13ch) :203–213 — eval distribution == training distribution by construction.
- T5 cache OPTIONAL :220 (None ⇒ DistilBERT path; cache only for the t5 ablation).
- `__getitem__` :345 — verbatim base sample + attaches: raw `caption_text` :362 (the DistilBERT input), `source` metadata :366, motion_id / source_motion_id (mask/grouping only).
- Data flow per item: motion .npy → AnyTopDataset → `anytop_x[J,13,T]` + graph fields (adjacency/geodesic/`anytop_graph_dist`/`anytop_joint_relations`) + masks (joint_mask/frame_mask) + skeleton_features[J,9]; caption string from manifest.

## 2. MODEL DESIGN — src/models/graph_salad/t2m_evaluator.py
### 2a. Text tower — DistilBertTextTower :180
- `__init__` :197 — loads LOCAL distilbert-base-uncased, FREEZES it (requires_grad=False); `head` = Linear(768→2048)→GELU→Dropout→Linear(2048→512) :220.
- `train()` override :225 — pins the frozen backbone in eval (dropout off) even when the evaluator is .train()-ed.
- `forward(captions)` :236 — tokenize(padding, truncation, max_len 64) → frozen backbone last_hidden_state[B,L,768] (under no_grad) → **MASKED-MEAN over valid tokens** :248 → head → text_emb[B,512]. (Masked-mean, NOT a learnable-query transformer: the query-pool collapsed at init — diagnosed, see §5.)
### 2b. Motion tower — fresh SkeletonEncoder :332 (defined in src/models/encoder.py:252)
- built `d_model=512, n_heads=8, d_ff=2048, n_graph_layers=6, n_temporal_layers=4, motion_mode="anytop13_split", attn_mode="graphormer"` — NO VAE/VQVAE/CodeFlow weights (independent init).
- `encode_motion(batch)` :373 — permute anytop_x[B,J,13,T]→[B,T,J,13]; SkeletonEncoder forward (encoder.py:439 anytop13_split root/non-root Linears; graphormer graph attn :321) → [B,T,J,512]; `_masked_mean_pool` over valid (frame,joint) :408 → [B,512]; L2-normalize.
### 2c. Shared space + loss
- both embeddings L2-normalized → cosine; learnable CLIP-style temperature `log_logit_scale` (init log(1/0.07)) :350, clamped `logit_scale` :439.
- `build_multi_positive_mask(motion_id, source_motion_id, caption_text)` :80 — [B,B] bool; off-diagonal pairs sharing ANY of the three keys are removed from the InfoNCE denominator (vectorized int-id hashing).
- `symmetric_infonce` :132 — logits = logit_scale·textᵀmotion; mask false-negs to −inf; mean of CE(text→motion) + CE(motion→text). Diagonal always kept.
- METADATA-ONLY: species/object/source_motion_id only build the mask / reports — never a tower input.

## 3. TRAINING — scripts/train_anytop_t2m_evaluator.py
- `setup_ddp()` :60 — torchrun env → (rank, world_size, local_rank, device); nccl; single-proc if WORLD_SIZE unset.
- `build_model()` :142, `build_dataset()` — text_tower=distilbert ⇒ caption_emb_cache=None.
- `run_step()` :174 — the core loop:
  1. ENCODER forward under bf16 autocast, scoped to model(...) ONLY :185–186 (DDP forward → local text_emb/motion_emb[b,512]).
  2. cast to fp32 (`.float()`) BEFORE gather/logits (spec: fp32 logits/CE).
  3. **GRAD-SAFE all-gather** `dist_nn.all_gather` (NOT torch.distributed.all_gather — that detaches grad → local-only negatives) :197–198 → global [WB,512].
  4. metadata gathered in RANK ORDER (`_gather_meta` :158, all_gather_object) → multi-positive mask on the GLOBAL batch.
  5. `symmetric_infonce` on global logits in fp32 :205.
- `main()` :209 — DistributedSampler(drop_last=True; symmetric per-rank for all_gather); `DDP(core, find_unused_parameters=True)` :244 (frozen DistilBERT + disabled encoder branches make no grad); AdamW; warmup(:256) + cosine LambdaLR; grad_clip; rank-0 logging/atomic ckpt; barrier+destroy.

## 4. LAUNCH SCRIPT + HYPERPARAMETERS
Launcher: scripts/_launch_anytop_t2m_evaluator.sh (single-node torchrun; cross-alloc reuses
the backbone orchestrator scripts/_launch_graph_pscf_2node_h200.sh, swapping the python command).

| Hyperparameter | Value | Source |
|---|---|---|
| data_root | data/animo4d_anytop_clean_L4_safe_plus_truebones | spec §3 |
| num_frames / max_joints | 300 / 144 | spec §3 |
| caption view | full (primary) | spec §5 |
| text tower | frozen DistilBERT + masked-mean MLP head | spec §5 (+ §5 collapse fix) |
| coemb_dim | 512 | spec §6.1 |
| n_heads / d_ff | 8 / 2048 | spec §6.1 |
| n_graph_layers / n_temporal_layers | 6 / 4 | spec §6.1 |
| dropout | 0.1 | spec §6.1 |
| temperature | learnable, init 0.07 | spec §6.1 |
| optimizer | AdamW (betas 0.9/0.99) | spec §6.1 |
| global_batch | 256 (= NGPU·PER_RANK; e.g. 8·32) | spec §6.1 (min) |
| lr | 2e-4 @ gb256 (→3-4e-4 @ gb512, Goyal) | spec §6.1 |
| warmup / schedule | 2000 steps + cosine decay | spec §6.1 |
| weight_decay / grad_clip | 1e-4 / 1.0 | spec §6.1 |
| precision | bf16 encoder forward, fp32 logits/CE | spec §6.1 |
| epochs | 100 (≈28k steps; tunable; no early-stop, gated by M5 gates) | proposed |

## 5. NOTE — collapse fix found by smoke (transparency)
The originally-planned learnable-query-token text pooling COLLAPSED at init (text off-diag cosine
0.985 → InfoNCE pinned at ln B). Diagnosed (scripts/_diag_eval_overfit.py) → switched to masked-mean
pooling (init cosine 0.813; overfits to ~0 at lr 2e-4). Smoke lr is 2e-4 (2e-3 destabilizes the bigger model).
codex re-PASSED.

## 6. AFTER TRAINING — validity gates (M5) before any metric is trusted
tiny-overfit ✓(smoke) · held-out val_all R@1/2/3 + matching > random · shuffled-caption drop ·
within-species retrieval · per-source (animo4d vs truebones) reports · then M6 VQVAE-recon eval ·
M7 CodeFlow-gen eval. No evaluator score is a paper claim until these pass.

## 7. KEY FILES
M0 scripts/build_anytop_t2m_eval_splits.py · M1 src/data/anytop_t2m_eval_dataset.py
M2 src/models/graph_salad/t2m_evaluator.py (motion tower src/models/encoder.py:252)
M3 scripts/train_anytop_t2m_evaluator.py · smoke scripts/_smoke_anytop_t2m_evaluator.py
launcher scripts/_launch_anytop_t2m_evaluator.sh
