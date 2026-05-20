# M1.1 Scaffolding - Codex Review Round 7 BRIEF

- **Model**: gpt-5.5, xhigh reasoning
- **Thread**: `019e438e-bf5e-7731-a2f0-71fafa11f9d3` (FRESH, no continuation)
- **Date**: 2026-05-20
- **Files reviewed**: batch.py / graph_utils.py / denoiser_stub.py / __init__.py / tests/test_scaffolding.py / tests/test_graph_utils.py
- **Verdict**: **NEEDS-FIX** (Q3 hygiene only; R12 convergence achieved)

## Headline

Round 7 seals **R12 category #8**: `GraphMotionBatch.from_collate_dict()` now rejects valid-tree-but-FK-invalid `parent_indices` by calling both `validate_parent_tree()` and `assert_root_first_parent_order()`.

No new R12 category #9 found. The remaining blocker is repository hygiene: `outside_docs/animo_metric_borrowing_plan_for_graph_salad.md` is still untracked outside the allowed implementation/test scope.

## Answers

| Q | Answer |
|---|---|
| Q1 all 8 R12 sealed? | **YES** - convergence achieved; no new category #9 despite searching parent-order variants, geodesic consistency, rest/bone/contact semantics, numeric-list finiteness, feature ranges, and string metadata |
| Q2 validation order | **PASS** - dependency order is structural before value use; minor docstring order is stale |
| Q3 R3 surgical scope | **FAIL** - `outside_docs/animo_metric_borrowing_plan_for_graph_salad.md` remains untracked outside `src/models/graph_salad/` + `tests/` |
| Q4 denoiser stub | **PASS** - signature-only, `level2_meta=None`, no params/state, raises `NotImplementedError`; caveat: file is untracked so git cannot prove unchanged since R1 |
| Q5 M1.2 forward compatibility | **GO for GraphMotionBatch invariants**; remaining geodesic/contact/bone/rest semantics are deferred consumer/data-health checks, not immediate pool/VAE/FK crashes |

## R12 Category Landscape

| # | Category | Status |
|---|----------|--------|
| 1 | tensor schema | sealed |
| 2 | cross-tensor T/J | sealed |
| 3 | scalar value-range + mask sum | sealed |
| 4 | NaN/Inf scalars+tensors | sealed |
| 5 | bool-as-int | sealed |
| 6 | list cardinality + tree-validity | sealed |
| 7 | graph semantics + mask prefix | sealed |
| 8 | parent ordering / FK topology | **sealed in R7** |
| 9 | new category | **not found** |

## MUST-FIX-NOW

- Clean, track, or explicitly scope-document `outside_docs/animo_metric_borrowing_plan_for_graph_salad.md` before claiming Karpathy R3 surgical compliance.

## DEFERRED-TO-M1.x

- Optional `geodesic_dist` recompute/cross-check when graph attention/pool code consumes it.
- Optional contact/bone/rest semantic validators in the loss/metric modules that actually depend on those ranges.

## M1.2 Unblock Status

**NO-GO** as a clean handoff until the Q3 outside-doc spillage is resolved.

Code-level GraphMotionBatch R12 status is **GO**: M1.2 consumers can trust parent tree/order, adjacency, masks, counts, and fps after the repository hygiene fix.

## Verification

- Requested `.venv/bin/python -m pytest ...` could not run: `.venv/bin/python` does not exist; current Python also lacks `pytest`.
- `python tests/test_scaffolding.py -v` -> 48 tests PASS in 1.851s, including real-dataset compat.
- `python tests/test_graph_utils.py -v` -> 39 tests PASS in 0.046s.

## Artifacts

- Full: `/scratch/ts1v23/workspace/noKslot_clean/docs/codex_reviews/m1_1_scaffolding_round7_FULL.txt`
- Brief: `/scratch/ts1v23/workspace/noKslot_clean/docs/codex_reviews/m1_1_scaffolding_round7_BRIEF.md`
