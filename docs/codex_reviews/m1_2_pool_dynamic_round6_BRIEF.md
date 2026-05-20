# M1.2 pool_dynamic.py — Round 6 codex review BRIEF

- Model: gpt-5.5 xhigh, fresh thread
- threadId: `019e4593-96af-71d3-994c-8bee35534245`
- Date: 2026-05-20

## Verdict
**NEEDS-FIX** — 12th R12 found (override-root omission).

## Q1 — 11 prior R12 categories
All 11 SEALED in implementation. Code/test line citations recorded in FULL. Soft note (carried from round 5): mincut orthogonality (#1) has finite-aux check but no numeric oracle unit test (codex independently probed perfect `C_valid=3/2` assignments and got `mincut_ortho=3.55e-15` — math is right, just no checked-in oracle).

Round-6 fix at `pool_dynamic.py:447-465` (per-sample joint_mask prefix-True guard) + 2 negative tests `test_pool_dynamic.py:246-277`. B>1 mixed case (one sample OK, one with hole) verified to catch the bad sample.

## Q2 — 12th R12: NEW
**`anchor_indices` override path does not require root (joint 0) to be among the active anchors.**

- Contract evidence: anchor rule "root must be anchor" at `pool_dynamic.py:4`, root/order invariant at `:25`, `graph_utils.py:380`, plan at `outside_docs/graph_salad_implementation_plan.md:356`. Current override validation at `pool_dynamic.py:575` only checks active anchors are valid / prefix / sorted / unique — no root-presence check.
- Adversarial trace (codex actually ran): `B=1, T=4, J=6, D=4`, line graph, `joint_mask=frame_mask=all True`, `anchor_indices=[[1,2,5]]`, `coarse_mask=[[T,T,T]]`. Expected `ValueError` (missing root 0). Actual: success — pooled adjacency `[[0,1,0],[1,0,1],[0,1,0]]`, finite features, rows sum to 1. Silent wrong-root pooled graph.
- Fix location: override validation block at `pool_dynamic.py:575` (just after sorted/unique check), add `valid_anchors[0] == 0` per sample.

## Q3 — R3 surgical compliance
**Yes.** Round 6 added only the joint_mask prefix guard + 2 negative tests. No drift into mincut, assignment, pooling, pooled-graph, or graph_utils math. Suite: 26/26 PASS in 0.124s (codex re-ran).

## Q4 — pool_deterministic.py unblock
**NO-GO** — deterministic counterpart shares the same anchor/root contract from graph_utils, so it would inherit the override-root-omission hole. Fix root-presence guard first.

## Fix list (round 7)
1. In `_select_anchors()` / override validation at `pool_dynamic.py:~575`: require `valid_anchors[0] == 0` for every sample (root must be the first active anchor). Raise `ValueError` with `f"DynamicGraphPool: sample {b} anchor_indices missing root (joint 0); got first active anchor {valid_anchors[0]}"`.
2. Add `test_anchor_indices_missing_root_raises` with `anchor_indices=[[1,2,5]]`, `coarse_mask=all True`, expecting `ValueError` matching `"missing root"`.
3. Re-run full suite (expected 27 tests PASS), then re-review round 7.

## Notes for future rounds
- Mincut numeric oracle is the only outstanding "soft" item across all 11 sealed categories. Not a R12 (math is correct; it's a test-coverage cosmetic). Defer unless explicitly requested.
