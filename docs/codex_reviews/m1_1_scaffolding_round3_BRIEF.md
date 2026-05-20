# M1.1 Scaffolding — Codex Round 3 Review BRIEF

**Thread (fresh):** `019e4365-de22-7de0-92d3-18765076c511`
**Model:** gpt-5.5 / xhigh
**Date:** 2026-05-20
**Scope:** batch.py (370 LoC) + denoiser_stub.py + __init__.py + test_scaffolding.py (27 tests)

## Verdict
**NEEDS-FIX** (minor, value-level not structural)

## M1.2 unblock status
**NO-GO** until 3 value-level guards land — M1.2 will read `num_joints` / `num_frames` as authoritative lengths.

## Remaining silent failures (R12)
1. **Scalar value consistency** — `num_joints` / `num_frames` accept `[-1, 999]` or values inconsistent with `joint_mask.sum(1)` / `frame_mask.sum(1)`.
2. **`fps` NaN/Inf** — Scalar floats not covered by the finite guard (which runs only on `_TENSOR_SHAPE_SPEC` padded tensors).
3. **`bool` as `parent_indices` int** — `isinstance(True, int) == True` because `bool ⊂ int`. Currently silently accepted.

## R2 simplicity
PASS. 4-stage validation split (padded / scalar / list-of-list / string-list) is the natural minimum, not over-engineered. Codex says don't merge spec tables.

## R3 surgical
Procedurally flagged by codex: untracked `docs/codex_reviews/m1_1_scaffolding*` files exist outside the `src/models/graph_salad/` + `tests/` allowed scope. These are review artifacts (Claude's audit trail), not source spillover. User decides whether to commit-with-M1.1 / defer / .gitignore.

`src/data/unified_dataset.py` confirmed NOT modified.

## Fix list (round 4)
- `batch.py:218` — after scalar loop, add: `num_joints > 0`, `num_joints <= J_max`, `num_joints == joint_mask.sum(1)`; same for `num_frames` / `frame_mask`.
- `batch.py:218` — `torch.isfinite(fps).all()` + `fps > 0`.
- `batch.py:272` — exclude `bool` from parent-indices int check (`isinstance(item, int) and not isinstance(item, bool)`).
- `tests/test_scaffolding.py` — add 4 regression tests: scalar mixed-device, length/mask mismatch, `fps=nan`, bool parent-index.

## Per-Q quick map
- Q1 (schema completeness): mostly PASS, scalar-value gap noted
- Q2 (NaN guard scope): PASS — no legit Inf-sentinel path in unified_dataset.py
- Q3 (device consistency): PASS — both padded and scalar checked (batch.py:197-202, 237-241)
- Q4 (denoiser stub): PASS — no params, `level2_meta=None`, raises NotImplementedError
- Q5 (tests soundness): PASS structurally; coverage gap on the 4 new guards above
- Q6 (R2 simplicity): PASS
- Q7 (R3 surgical): procedurally NEEDS-FIX (untracked docs)
- Q8 (M1.2 compat): mostly PASS; `num_joints` / `num_frames` are surprise risk

## Files
- FULL: `/scratch/ts1v23/workspace/noKslot_clean/docs/codex_reviews/m1_1_scaffolding_round3_FULL.txt`
- BRIEF: this file
