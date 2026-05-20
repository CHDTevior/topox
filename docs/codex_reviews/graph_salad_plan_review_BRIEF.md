# Codex Review Brief — graph-SALAD plan vs noKslot_clean repo

**Model:** gpt-5.5, reasoning effort: xhigh
**Date:** 2026-05-20
**threadId:** `019e42fe-95b9-7150-bc0e-851a043b889b`
**Full output:** `docs/codex_reviews/graph_salad_plan_review_FULL.txt`

## TL;DR

- **TreeIK chain:** GOOD-TO-REUSE — rot → hard FK → bone/edge invariance is correct; one adapter need flagged for dynamic graph metadata.
- **Plan vs repo:** §4 DIRECT REUSE, §9 MINOR ADAPT, §6/§7/§8 NEW BUILD, §11 SPEC GAP (text field).
- **Critical bugs in current code:** None found.
- **M1 ready to start** under the 7-item acceptance checklist below.

## 1. TreeIK logic chain (PRIMARY) — GOOD-TO-REUSE

- (a) `RestFiLM` conditions only on `rest_tensor` projection — geometry-only, leak-free for target-skeleton generation. [treeik_decoder.py:158-170, 274]
- (b) `TreeGraphAttention` uses dense `[B,J,J]` adjacency/geodesic bias, masks padded keys, zeros padded queries — but does NOT consume `edge_index/edge_mask`. **Adapter needed:** graph-salad must convert dynamic graph metadata to dense `[B,J,J]` (or add an adapter). [treeik_decoder.py:194-210]
- (c) FK in `fk_one` validates root=0 and parent-before-child topo order per batch sample, truncates to `Jpad`, zeros padded joints. **Hidden invariant:** any pooled/unpooled order must preserve root=0 and topological parent-before-child. [treeik_decoder.py:87-114]
- (d) 6D → rot uses standard Gram-Schmidt; geodesic loss clamps acos input. Near-colinear 6D vectors can yield non-orthonormal matrices — monitor but not fatal. [treeik_decoder.py:57-63, 299-311]
- (e) `TopoFKTreeIKDecoder.forward` consumes all batch tensors correctly. Train/eval currently pass source `frame_mask` into target decode — safe for same-clip self-recon, but graph-salad should pass batch/target frame mask explicitly. [treeik_decoder.py:269-288; train.py:468-473; eval.py:413-416]

## 2. Plan vs repo compatibility audit

| Section | Verdict | Reason (one line) |
|---|---|---|
| §4 SkeletonEncoder reuse | DIRECT REUSE | Interface matches; downstream must keep frame masking (encoder only final-masks joints) |
| §6 DynamicGraphPool | NEW BUILD | No module exists; current identity assignment is baseline-only |
| §7 DynamicGraphUnpool | NEW BUILD | MotionDecoder unpools by assignment but has no dynamic pool metadata interface |
| §8 GraphMotionVAE | NEW BUILD | Current Model = encoder+slot_norm+decoder only, no latent head/pool/unpool |
| §9 MotionDecoder reuse | MINOR ADAPT | Reuse temporal/cross-attn/output; add `coarse_mask` (current attention only has assignment log-bias) |
| §11 diffusion training | SPEC GAP | `GraphMotionBatch.text` absent though §11 calls `batch.text` |

## 3. Hidden risks (top 5)

1. **Dragon J=143** — attention/pool memory and masks must handle near-`max_joints`. Repo default `max_joints=160` covers it, but dynamic graph code must avoid accidental fixed-128 assumptions. [train.py:217-219]
2. **Normalization stats** — dataset loads stats under `data_dir.name` but species samples use species `skeleton_id`, so per-species stats would not apply as written. Current train/eval force `normalize=False`; recon loss is raw masked MSE. [unified_dataset.py:80-83, 138-142; train.py:276-278; utils.py:112-117]
3. **Text handling** — collate returns `text` as `list[str]`; plan dataclass omits text. Denoiser interface must define this now even if M1 does not train it. [unified_dataset.py:351-362]
4. **Checkpoint compatibility** — L6/baseline load requires `strict=False`; missing must stay empty; only `slot_assignment.*` may be unexpected. [model.py:20-26; train.py:349-371; step12_final_OUT.txt:55-60]
5. **Plan §14 outdated** — still says "current SlotAE/fixed n_slots" smoke; locked decision supersedes it: use existing noKslot ep399 baseline; do NOT revive source `slot_ae.py`/`slot_assignment.py`. [plan:856-870; model.py:14-18]

## 4. M1 acceptance checklist (7 items)

- [ ] `src/models/graph_salad/` contains `__init__.py`, batch, pool, unpool, VAE, attention, losses, graph_utils, and denoiser-interface stub files.
- [ ] CPU smoke: B=2 mixed J, forward/backward, no NaN, z shape `[B,T/4,C2_max,D]`, no hard-coded 7.
- [ ] Padding gate: padded joints/coarse nodes are exactly zeroed and excluded from recon/KL/pool losses.
- [ ] Ckpt gate: L6 and ep399 baseline load through existing `Model` with missing=[], unexpected only `slot_assignment.*`.
- [ ] Recon gate: fixed mini-split masked recon loss decreases; ep399-equivalent run targets `train_diag <= 0.33` or reports regression.
- [ ] Denoiser stub exposes `forward(z_t, timesteps, text, adjacency, geodesic_dist, coarse_mask, frame_mask)`.
- [ ] Visual QA: render GT-vs-pred multi-frame GIF/contact sheets for at least Bat/Crab/Horse and one Dragon clip; metric-only PASS is invalid.

## 5. Critical bugs flagged

**None found.**
