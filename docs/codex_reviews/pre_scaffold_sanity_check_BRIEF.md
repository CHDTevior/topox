# Pre-Scaffold Sanity Check — BRIEF

**Date**: 2026-05-20
**Model**: gpt-5.5, xhigh reasoning
**Thread**: `019e4323-bfe7-7530-94c3-7e396913b622`
**Scope**: META-REVIEW of planning artifacts before M1 scaffolding (no code review)
**Inputs**: `PLAN_GAP_REPORT.md` (full), `graph_salad_implementation_plan.md` (full, 1143L)
**Full output**: `pre_scaffold_sanity_check_FULL.txt`

---

## Verdict

| Axis | Verdict | One-line justification |
|---|---|---|
| (a) PLAN_GAP_REPORT.md soundness | **YELLOW** | Internal contradiction: §2 says AnyTop MUST be addressed before coding; §7/§8 treat Q-A as resolved + AnyTop reading conditional. File-count mismatch: §5 says "11 new src files" but only 9 listed in `src/models/graph_salad/`. |
| (b) M1 6-step roadmap | **YELLOW** | Scaffolding can start after small plan fixes. `graph_utils.py` is listed but not as distinct dependency before `pool.py`. Dataset/dataloader wiring + loss-weight scheduling are implicit (vanish into "graph_salad scaffolding"). |

---

## Top 5 hidden risks / omissions

1. **AnyTop novelty risk under-priced**. arXiv confirms AnyTop already does topology-aware denoising + textual joint descriptions + few-shot topology generalization + unseen skeleton support. "39 species + SALAD framework" alone reads as benchmark engineering. Needs a distinct architectural/capability advantage answer before code.

2. **Fixed `lambda_mincut=0.5` (no warmup) failure modes are under-declared**. Cold-start all-to-one cluster collapse, empty coarse nodes, early FK/recon gradient conflict, topology-dependent C_i instability all plausible. A ramp should be pre-authorized as fail-loud fallback if diagnostics show collapse.

3. **Denoiser stub signature missing metadata for editing**. Raw `text: list[str]` is OK for M1 only if plan records that Phase 2 may switch to cached text embeddings. Should add `level2_meta` (token→coarse-group mapping) now if attention-editing is a planned Phase 2 deliverable, else attention-map editing requires Phase 2 signature break.

4. **Q-D normalization reclassification is not airtight**. 3rd interpretation ("source repo never finished Zoo stats") not surfaced. Failure mode if wrong: raw-scale training overweights large skeletons (Dragon J=143), hides small-species (Bat) failure, makes `train_diag ≤ 0.33` misleading.

5. **M1 acceptance gates have 5 blind spots**: (i) NaN-with-zero-gradient-after-masking, (ii) all-joints-to-one-cluster pool collapse, (iii) empty-coarse-node collapse, (iv) static-FK/T-pose decoder degenerate solution, (v) mode collapse across topology families. Visual QA needs per-topology + motion-dynamics checks, not just single GIFs.

---

## Top 3 changes recommended before scaffold starts

1. **Make AnyTop audit mandatory + define minimum differentiator before coding** — Why: reviewer-2 pushback will be "AnyTop already does this"; Graph-SALAD needs prewritten answer beyond benchmark size. Resolves §2 vs §7/§8 contradiction.

2. **Split `graph_utils`/anchor-ordering preflight into its own step before `DynamicGraphPool`** — Why: pool correctness depends on `shortest_path`, anchor generation, root=0 preservation, parent-before-child topological order. Currently lumped under "graph_salad scaffolding".

3. **Strengthen M1 gates with pool diagnostics + relative baseline** — Why: add active coarse count, mass min/max, assignment entropy, per-topology recon/velocity, and zero-pool/identity-assignment comparison. Replace absolute `train_diag ≤ 0.33` (ep399 baseline is structurally different) with relative-to-no-pool baseline.

---

## Most critical question codex would ask the implementer

> What is the smallest M1 that proves the hard claim — that **dynamic graph pooling helps multi-topology reconstruction** rather than merely adding complexity? I would ask for a locked comparison against a simpler deterministic-anchor or no-learned-pool baseline, with per-species visual QA and assignment diagnostics. Without that, M1 can pass smoke tests while still violating Karpathy R2/R9: too much architecture, not enough evidence that the architecture is the reason reconstruction works.

---

**Karpathy lens applied**: R1 (surface uncertainty — §2 vs §7 contradiction), R2 (simplicity — 21-file M1 may be over-scoped), R7 (don't average conflicts — Q-D 3rd interpretation), R9 (verification must catch failures — gate blind spots), R12 (fail loud — pre-authorize MinCut ramp fallback).
