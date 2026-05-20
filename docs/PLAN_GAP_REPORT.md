# Plan Gap Report — Graph-SALAD on noKslot_clean

**Generated**: 2026-05-20
**Inputs synthesized**:
- `outside_docs/graph_salad_implementation_plan.md` (1143 lines, 27KB)
- Repo state at HEAD `a14eccb` (step 12 final PASS, 25 commits, 4534 lines)
- Codex review `docs/codex_reviews/graph_salad_plan_review_{BRIEF.md,FULL.txt}` (gpt-5.5 xhigh, threadId `019e42fe-95b9-7150-bc0e-851a043b889b`)
- Lit survey `docs/graph_vae_lit_survey.md` (10 papers across 5 sub-topics)
- SALAD reference repo cloned to `outside_docs/SALAD/` (1903 LoC in models/)
- User-locked decisions from `/research-pipeline` Gate-1 (2026-05-20)

---

## 1. 状态总览 (what's settled vs what needs decision)

### 1.1 Fact-locked (no user decision needed; closed by evidence)

| 项 | Fact | Implication |
|---|---|---|
| Phase 0 smoke test (plan §14) | noKslot ep399 baseline (K=Jpad identity assignment, step 12 PASS) is the equivalent | **Skip** cloning source `slot_ae.py`; plan §14 is superseded |
| SkeletonEncoder reuse (plan §4) | `encoder.py::SkeletonEncoder` forward signature 100% matches plan §4.1 | **Direct reuse**, zero modification needed |
| GraphMotionBatch dataclass (plan §3) | `UnifiedMotionDataset + collate_fn` already emit padding-mask-aware dict with `motion_features / skeleton_features / adjacency / geodesic_dist / joint_mask / frame_mask / texts (list[str])` | **No rewrite needed**; dataclass-ification is optional syntactic sugar |
| Text annotation availability | Every motion file has `texts` field (319 chars typical), collate already passes through | **t2m path is data-ready** |
| Dragon J=143 vs max_joints | train/eval default `max_joints=160` ≥ 143; `unified_dataset.py` default 128 is overridden by scripts | **Not a bug**, but graph_salad must use `args.max_joints` throughout, never hard-code |
| 6D rotation availability | Every motion file has `local_rotations_6d [T, J-1, 6]` (root has no rotation) | **TreeIK supervision path data-ready** |

### 1.2 User-locked decisions (2026-05-20)

**Gate 1 decisions:**

| Decision | User choice | Implication |
|---|---|---|
| SALAD acquisition | Clone to `outside_docs/SALAD/` as read-only ref; clean reimplementation in `src/models/graph_salad/` | ✓ Done — `outside_docs/SALAD/` cloned (1903 LoC in models/) |
| VAE output format | **Rot → hard FK → bone/edge invariance** path; reuse `treeik_decoder.py::TopoFKDecoder` | VAE outputs `local_rotations_6d`, then TopoFKDecoder does FK to `joint_positions` |
| M1 milestone | GraphMotionVAE reconstruction + denoiser interface preparation (no denoiser training yet) | Phase 1 only; Phase 2 reserved for after VAE PASSes |
| Review pace | Parallel: codex review + graph VAE lit survey | ✓ Both done |

**Gate 2 decisions (post-codex/lit-survey):**

| Decision | User choice | Implication |
|---|---|---|
| Q-A AnyTop positioning | Related work; differentiate via 39-species animal multi-topology benchmark + SALAD ancestry (continuous VAE + DDIM + text editing) | Cite AnyTop in related work; do **NOT** pivot architecture; differentiation = benchmark + framework choice, not novelty in attention mechanism |
| Q-B MinCutPool aux loss | **Yes, M1 includes it; `lambda_mincut=0.5` fixed (paper default); no warmup** | Implement cut + ortho loss in `losses.py`; covers plan §13.3 `L_pool_connectivity` + `L_pool_mass` + part of `L_graph_preserve`; keep `L_pool_locality` + `L_pool_entropy` from plan §13.3 separately |
| Q-C Denoiser stub depth | **Signature-only stub (`raise NotImplementedError`)** | Lock interface, zero implementation; Phase 2 fills it |
| Q-D Normalization keying | **No fix** — source repo has same keying by design; Zoo-style multi-species datasets intentionally don't normalize (no `stats.npz` for Zoo data in source repo); graph_salad continues `normalize=False` | `unified_dataset.py` untouched; Karpathy R3 surgical |

**Gate 3 decisions (post pre-scaffold sanity check, 2026-05-20):**

| Decision | User choice | Implication |
|---|---|---|
| Pool ablation scope | **3-way: dynamic learned + deterministic-anchor + no-pool (full skeletal attention)** | M1 trains all three variants on same mini-split + full-split. Acceptance: per-species recon comparison table + 3-way visual QA. Paper narrative locked in by data, not pre-commit. |
| M1 timeline | 12-17 days (extended from 7-10 days) | One M1.0 preflight + 3 architecture variants + comparison report |
| Reviewer-2 defense | Pre-built via the 3-way ablation | "Did you try deterministic / no-pool?" answered with hard numbers before paper writing starts |

---

## 2. Concurrent work — AnyTop (acknowledged, no architecture pivot per Q-A)

**AnyTop** (arXiv 2502.17327, SIGGRAPH 2025, Tel Aviv University, project: https://anytop2025.github.io/Anytop-page/):
- "Diffusion model for diverse characters, using only skeletal structure as input"
- Transformer-based denoising network with **topology integrated into attention**
- **Textual joint descriptions** for cross-skeleton semantic correspondence
- **3 training examples per topology** generalization
- **Unseen-skeleton** generation supported

**This is the closest sibling to Graph-SALAD's denoiser** (lit survey verdict). Concurrency timeline: AnyTop Feb 24, 2025 → SALAD Mar 18, 2025 → our work May 2026 (15 months later).

**User decision (Q-A 2026-05-20)**: AnyTop is **related work**, no deep read, no architecture pivot. Differentiation rests on:
- 39-species animal multi-topology benchmark (J range 18-143) vs. AnyTop's training set
- SALAD ancestry: continuous latent VAE + DDIM + text cross-attention + attention-editing framework
- Hierarchical Pool×2 + Unpool×2 (plan §6/§7) — to be verified vs. AnyTop's exact architecture during paper-writing related-work section, NOT pre-coding

⚠ **Residual reviewer-2 risk** (codex flagged 2026-05-20 pre-scaffold sanity check): codex audit warned that "39 species + SALAD framework" alone may read as benchmark engineering to top-venue reviewers. User has chosen to absorb this risk; revisit if M1 visual QA reveals architecture-specific behavior distinct from AnyTop's published demos.

---

## 3. Codex review findings (7 items)

### 3.1 TreeIK logic chain — GOOD-TO-REUSE ✓
- `RestFiLM` conditions only on `rest_tensor` projection → leak-free
- 6D→rot via standard Gram-Schmidt; geodesic loss clamps acos input
- FK validates root=0 + parent-before-child topo order, zeros padded joints
- `TopoFKTreeIKDecoder.forward` consumes batch tensors correctly

### 3.2 TreeGraphAttention interface (adapter need)
- Uses **dense** `[B,J,J]` adjacency/geodesic bias
- Does NOT consume `edge_index/edge_mask`
- ⇒ `DynamicGraphPool` output MUST be dense `pooled_adjacency [B,C,C]` + `pooled_geodesic [B,C,C]` (already matches plan §6.1)

### 3.3 Hidden TreeIK invariant ⚠
- Any pool/unpool ordering MUST preserve **root=0 + parent-before-child topo order**
- Plan §6.5 does not explicitly require this
- ⇒ `DynamicGraphPool` cannot sort coarse nodes arbitrarily; must preserve a derived parent ordering on the coarse graph

### 3.4 Normalization keying — RECLASSIFIED as documented design (post-Q-D 2026-05-20)
- `UnifiedMotionDataset._load_data_source` stores stats with `data_dir.name` key
- Samples' `skeleton_id` is the species name (e.g., "Bat")
- ⇒ Looks like a bug, BUT: source repo has **identical** keying, and source repo's `data/processed_v1_backup/` only ships `stats.npz` for single-skeleton datasets (humanml3d/mixamo/cmu_mocap/100style/bandai_namco) — **NOT** for Zoo-style multi-species datasets
- ⇒ Conclusion: this is **intentional design**, not a bug. Zoo-style data goes through raw-scale, multi-species robust path.
- ⇒ graph_salad: keep `normalize=False` contract; do NOT touch `unified_dataset.py`

### 3.5 Plan §11 SPEC GAP — text in dataclass
- `batch.text` referenced but plan §3 `GraphMotionBatch` dataclass omits text field
- `collate_fn` already emits `text: list[str]`
- ⇒ Denoiser stub interface MUST be `forward(z_t, timesteps, text, adjacency, geodesic_dist, coarse_mask, frame_mask)`

### 3.6 Ckpt-compat envelope is narrow
- L6 + ep399 baseline load requires `strict=False` with `missing=[]`
- Only `slot_assignment.*` keys allowed as unexpected
- ⇒ graph_salad components must not introduce new missing keys

### 3.7 Dragon J=143 + dynamic max_joints
- Verified not a bug (scripts use `max_joints=160`)
- ⇒ graph_salad must use `args.max_joints` everywhere; no hard-coded 128 (or 22 or 7)

---

## 4. Lit survey — top 5 + 3 design recommendations

### 4.1 Must-read papers

1. **Skeleton-Aware Networks** (Aberman, SIGGRAPH 2020, 2005.05732) — direct motion-domain ancestor, but **homeomorphism-limited** (cannot handle J=18 vs J=143)
2. **DiffPool** (NeurIPS 2018, 1806.08804) — canonical soft-assignment pooling with `A' = S^T A S`
3. **Graph U-Nets** (ICML 2019, 1905.05178) — gPool/gUnpool with **saved-index restore**, direct template for Pool×2 / Unpool×2 stack
4. **Graphormer** (NeurIPS 2021, 2106.05234) — most-cited `attn += SPD_bias + edge_enc + centrality`, exactly plan §10.3
5. **AnyTop** (SIGGRAPH 2025, 2502.17327) — see §2 above, closest sibling

### 4.2 Design recommendations

1. **DynamicGraphPool composition** = DiffPool (soft assign + pooled-A) **+** SAGPool (topology-aware anchor score) **+** MinCutPool (**orthogonality / balanced-cut aux loss** — **key for Dragon J=143 and Anaconda J=27** to prevent clusters straddling limb boundaries)
2. **Phase 2 attention bias** = Graphormer SPD **+** GRPE-style edge-aware K/V (push bone-length/parent-child edge semantics into Q/K/V)
3. **Phase 2 backbone** = MLD's latent-diffusion recipe (SALAD's immediate ancestor); escalate to DiGress-style discrete graph diffusion only if Phase 3 unseen-topology needs it

---

## 5. M1 file inventory (proposed — needs user sign-off)

Following plan §16 + codex M1 checklist + Karpathy R2 (simplicity, no premature abstraction):

```
src/models/graph_salad/
  __init__.py
  batch.py                  # GraphMotionBatch dataclass (wraps existing dict, no new collate)
  graph_utils.py            # PREFLIGHT (M1.0): shortest_path, edge-crossing coarse graph,
                            #   root-to-leaf chain, anchor generation (rule-based:
                            #   root + degree≥3 branch + leaf + chain chunking).
                            #   Pool depends on this — separate codex review FIRST.
  pool_dynamic.py           # DynamicGraphPool (DiffPool soft-assign + topology-aware anchor
                            #   from graph_utils + MinCut aux loss + preserves root=0 +
                            #   parent ordering, codex 3.3 invariant)
  pool_deterministic.py     # DeterministicGraphPool (Gate-3 ablation): same anchors from
                            #   graph_utils + hard rule-based assignment (argmin geodesic
                            #   or chain-chunk membership). No learnable S, no MinCut.
  unpool.py                 # DynamicGraphUnpool (saved-index restore, à la Graph U-Net).
                            #   Shared between dynamic + deterministic (P interface same).
  attention.py              # GraphAttentionBlock (Graphormer SPD bias + adjacency bias +
                            #   coarse_mask, reusable across Phase 1 & 2; also used by
                            #   no-pool variant for full skeletal attention).
  vae.py                    # GraphMotionVAE with pool_type ∈ {dynamic, deterministic, none}:
                            #   SkeletonEncoder → {pool×2 OR no-pool} → Gaussian latent →
                            #   {unpool×2 OR identity} → TopoFKDecoder; rot→FK→pos
                            #   invariance preserved across all 3 variants.
  losses.py                 # masked L1 pos + masked L1 vel + KL + bone + FK + pool aux:
                            #   MinCut cut+ortho (lambda=0.5 fixed, active when
                            #   pool_type=dynamic only), L_pool_locality, L_pool_entropy
                            #   (plan §13.3 retained, dynamic-only)
  denoiser_stub.py          # Interface only: forward(z_t, timesteps, text, adjacency,
                            #   geodesic_dist, coarse_mask, frame_mask, level2_meta) —
                            #   raises NotImplementedError. level2_meta added per codex
                            #   pre-scaffold review (token→coarse-group mapping for
                            #   Phase 2 attention editing).

scripts/
  train_graph_vae.py        # NEW, modeled on existing train.py; reconstruction phase only
  eval_graph_vae.py         # NEW, reuses metric framework from eval.py
  self_test_graph_vae.py    # NEW, CPU smoke (~5s), mixed-J B=2 fwd/bwd

tests/                      # NEW directory (does not exist yet)
  test_pool_padding.py      # padded joints zeroed; mass-normalized
  test_pool_locality.py     # geodesic locality of assignment
  test_pool_connectivity.py # each coarse node covers connected fine nodes
  test_unpool_inverse.py    # pool→unpool round-trip identity on fine
  test_topology_invariant.py # root=0 + parent ordering preserved
  test_mixed_J_forward.py   # B=2 with J in [18, 143], no NaN
  test_no_hard_seven.py     # z.shape[2] varies with target graph
```

**File count**: 10 new src files (graph_salad/) + 3 new scripts + 7 new tests = **20 new files** (+1 vs Gate-2 due to pool_deterministic.py split).

**LoC estimate**: ~2500-3500 lines (graph_salad ~1500-2000, scripts ~600-800, tests ~400-700).

---

## 6. M1 acceptance gates (codex 7-item checklist + pre-scaffold strengthening, locked)

1. **Structure**: `src/models/graph_salad/` contains all 9 files listed above + denoiser stub
2. **CPU smoke**: B=2 mixed-J, fwd/bwd, no NaN, `z.shape = [B, T/4, C2_max, D]`, no hard-coded 7
3. **Padding gate**: padded joints/coarse nodes zeroed AND excluded from recon/KL/pool losses; **NaN-with-zero-gradient guard** (test that loss.backward() produces non-zero grads on at least one parameter per batch)
4. **Ckpt gate**: L6 + ep399 baseline load with `missing=[]`, unexpected only `slot_assignment.*`
5. **Recon gate**: mini-split masked recon decreases; target = ep399-equivalent `train_diag ≤ 0.33` or explicit regression report; **PLUS per-species recon for Bat, Crab, Horse, Dragon** (catches the case where global recon is good but Dragon J=143 is catastrophic)
6. **Denoiser stub**: signature `forward(z_t, timesteps, text, adjacency, geodesic_dist, coarse_mask, frame_mask, level2_meta)` locked
7. **Visual QA** (CV rule, not optional): GT-vs-pred multi-frame GIFs for Bat, Crab, Horse + 1 Dragon clip; absolute paths in report. Metric-only PASS violates CV primacy rule.

### Pool diagnostics (added per codex pre-scaffold review, 2026-05-20)
8. **Pool diagnostics** logged every N iterations:
   - `active_coarse_count` per batch (median, min, max) — should track target ~ C∝J/4
   - `mass_min_max_ratio` per cluster (catches all-to-one collapse)
   - `assignment_entropy` (per-row) — should not be near 0 (hard collapse) or log(C) (uniform = uninformative)
   - `per_topology_recon` table at every eval — catches cross-topology mode collapse

---

## 7. Open questions — ALL RESOLVED (2026-05-20)

All Gate-2 questions have user lock-in (see §1.2). Summary:

- **Q-A AnyTop** → Related work; differentiate via 39-species benchmark + SALAD ancestry. No architecture pivot.
- **Q-B MinCutPool aux loss** → **Yes**, `lambda_mincut=0.5` fixed, no warmup. Implement in `losses.py`.
- **Q-C Denoiser stub** → **Signature-only stub** (`raise NotImplementedError`).
- **Q-D Normalization keying** → **No fix** (intentional Zoo-style design, source repo same; graph_salad `normalize=False`).

---

## 8. Next-action menu

After user answers Q-A through Q-D:

- [Pre-coding] AnyTop paper read (if Q-A=1) — 30 min
- [Pre-coding] M1 file inventory final-lock with any adjustments — 5 min
- [Coding] graph_salad scaffolding (empty files + imports) — 30 min, **first codex review here**
- [Coding] DynamicGraphPool (the hardest module, ~500 LoC) — 1-2 days, **codex review**
- [Coding] DynamicGraphUnpool — 0.5 day, **codex review**
- [Coding] GraphAttentionBlock — 0.5 day, **codex review**
- [Coding] GraphMotionVAE wiring — 0.5 day, **codex review**
- [Coding] losses.py — 0.5 day, **codex review**
- [Coding] denoiser_stub.py — 0.25 day, **codex review** (signature lock)
- [Test] 7 unit tests — 0.5 day
- [Train] mini-split reconstruction (CPU/single GPU smoke) — 0.5 day
- [Train] full Phase 1 (real GPU run, target `train_diag ≤ 0.33`) — 1-2 days
- [QA] Visual QA gif rendering Bat/Crab/Horse/Dragon — 0.5 day
- [M1 finalize] codex final review + acceptance gate verification — 0.5 day

**Estimated M1 elapsed**: 7-10 days (assuming codex reviews are mostly PASS first round).

---

## 9. Evaluation strategy (added 2026-05-20 per user)

Source: `outside_docs/animo_metric_borrowing_plan_for_graph_salad.md` (21.5KB) — AniMo-borrowing plan for our open-topology evaluation needs.

**Core rule (user 2026-05-20)**: **VAE recon (M1.0-M1.6) needs ONLY loss**; metrics infrastructure is **not** in M1 scope. Phase 2 (denoiser/t2m training) is where metrics become mandatory.

### 9.1 What we borrow from AniMo (CVPR 2025)

- Reconstruction eval ↔ generation eval **separation** (`eval_graph_vae.py` + `eval_graph_denoiser.py`)
- Metric **concepts**: FID / R-Precision / Matching / Diversity / Multimodality
- Text-motion **evaluator embedding-space** evaluation paradigm
- `repeat_time=10` + 95% confidence interval statistics
- Root-aligned **MPJPE** thinking
- `eval.log` + checkpoint-sweep engineering organization

### 9.2 What we do NOT borrow

- Fixed `nb_joints=30` / `dim_pose=359` canonical pose
- `EvaluatorModelWrapper` (binds to 30-joint flat vector)
- `recover_from_ric(num_joint=30)`
- AnimalML3D-style OOD ≠ unseen-topology

### 9.3 New `metrics/` module needed before Phase 2 starts

```
metrics/
  reconstruction.py           — masked pos/vel MAE + masked root-aligned MPJPE
  physical.py                 — bone length error / edge stretch / contact sliding
  pool_metrics.py             — compression / mass / entropy / locality / connectivity / edge recall
                                (Animo has nothing equivalent — we are inventing this)
  graph_text_motion_evaluator.py — graph-aware contrastive text-motion model
                                (replaces Animo's fixed 30-joint EvaluatorModelWrapper)
  generation_metrics.py       — Graph-FID / Graph-R@K / Matching / Diversity / Multimodality
                                computed in our graph-aware embedding space
  split_report.py             — per-topology split reporting (seen/unseen species,
                                J-bucket, body-plan, limb-count). Animo has no equivalent.
  logging.py                  — eval.log + stdout consistent with Animo organization
```

### 9.4 Hard constraints (forbidden actions during metric implementation)

- ❌ Don't remap to fixed 30 joints
- ❌ Don't set `dim_pose=359`
- ❌ Don't fix root index across samples
- ❌ Don't fix contact to foot joints — use `node_attr.can_contact`
- ❌ Don't include padded joints/coarse nodes in any metric
- ❌ Don't equate AnimalML3D OOD with unseen-topology protocol

### 9.5 Acceptance gates for metric infrastructure (Phase 2 entry-block)

1. Mixed-J batch (B=2, J=22 + J=37) → all reconstruction metrics work
2. Padded joints/coarse nodes excluded from every metric
3. `edge_index` varies per sample → `bone_length_error` works
4. Dynamic pool's `K_i` varies → `pool_metrics` work
5. `GraphTextMotionEvaluator` accepts `[B,T,J,F] + graph` (no `dim_pose=359`)
6. `Graph-FID` / `Graph-RPrecision` topology-agnostic
7. `split_report` reports seen + unseen topology separately
8. `eval_graph_denoiser.py` repeats 10× + outputs mean ± 95% CI

### 9.6 Integration with M1 milestones

- M1.0-M1.6 (VAE) — **no metrics yet**, only loss
- M1.6 → Phase 2 transition gate: **build `metrics/` module first** as a separate milestone (M2.0 metric preflight, mirroring M1.0 graph_utils preflight pattern)
- Phase 2 denoiser training reads `metrics/` for eval & paper claims

---

## 10. Phase 1 closeout

- M1.0 codex PASS, committed at `3a41aa2`
- M1.1 codex round 7 PASS (R12 convergence at 8 categories; 0 new in round 7)
- M1.1 commit pending: source + tests + 7 codex review pairs + animo plan + this update

**End of report**. M1.2 (pool/unpool/attention/losses) is the next active milestone.
