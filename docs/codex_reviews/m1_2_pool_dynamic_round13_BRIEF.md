# M1.2 pool_dynamic.py Round 13 — codex review BRIEF

- **Model**: gpt-5.5, xhigh
- **Thread**: FRESH (no continuation); threadId `019e45ce-9e22-76d1-b5ae-cfa01d36629f`
- **Date**: 2026-05-20
- **Local verification**: `python tests/test_pool_dynamic.py` -> 53 OK; `python -m unittest discover -s tests` -> 169 OK.

## VERDICT: NEEDS-FIX

## Q1 — 20 R12 seal table

- **SEALED (17)**: #1, #2, #3, #4, #5, #6, #7, #8, #9, #10, #11, #12, #13, #14, #16, #17 (named branches), #20 (new round-13 padded-adj mincut mask).
- **PARTIAL (1)**: #15 — adjacency source guards present (`pool_dynamic.py:510-538`), but no direct pool regression for adjacency NaN/Inf branch.
- **NOT-SEALED (2)**:
  - **#18 (override-tensor device)** — author defer NOT defensible. `pool_dynamic.py:693-781` validates override tensors with `.item()` calls at L719-720, L772-777 BEFORE device normalization. A meta `anchor_indices`/`coarse_mask` raises RuntimeError instead of fail-loud ValueError. Override mode is the documented level-2 path (`pool_dynamic.py:419-425`), so this is on the critical surface.
  - **#19 (Floyd exact-equality)** — author "fp32 noise floor ~1e-6" defense NOT defensible. `graph_utils.floyd_shortest_path` initializes exact 0/1/+Inf and uses min-add over integer hop counts (`graph_utils.py:125-146`); fp32 representation is exact for these. A +5e-7 symmetric geodesic perturbation passes through `torch.allclose(..., atol=1e-6, rtol=0.0)` at `pool_dynamic.py:631-634`.
- **Test gap, #17**: `mincut_lambda` non-finite guard exists in source (`pool_dynamic.py:115-123`) but `test_non_finite_hparam_raises` (`tests/test_pool_dynamic.py:297-301`) only covers `locality_alpha` and `temperature`.

## Q2 — 21st R12 (CONCRETE)

**FOUND**: `parent_indices` <-> adjacency/geodesic consistency is NOT enforced on the direct `DynamicGraphPool.forward` path. `pool_dynamic.py:670-683` checks only list length; anchor selection at L688-691 then trusts the parent tree, while Floyd at L621-638 only cross-checks adjacency vs geodesic — never parent topology vs adjacency.

- **Repro**: J=6, line adjacency/geodesic, joint_mask=all-True, max_coarse=6, local_radius=5, but pass star parents `[-1,0,0,0,0,0]` instead of line `[-1,0,1,2,3,4]`.
- **Failure mode**: No ValueError. Anchors derived from star tree become `[0,1,2,3,4,5]` instead of correct line anchors `[0,5]`. Pool geometry silently corrupted; aux losses propagate.
- **Note**: `batch.py:426-472` has the equivalent batch-side guard, so the direct API is the only leak.

**Additional hardening gap**: `local_radius`, `max_chain_chunk_len`, `temporal_stride` are not strict-int/non-bool. `temporal_stride=2.0` fails deep at `pool_dynamic.py:279-293`; `max_chain_chunk_len=NaN` passes constructor and silently disables chunk promotion via `consec >= nan` at `graph_utils.py:452`.

## Q3 — M1.3 pool_deterministic.py unblock: NO-GO

## Fix list (priority order)

1. Add `parent_indices` <-> `adjacency` consistency validation in `DynamicGraphPool.forward` (or hard-require GraphMotionBatch and remove the direct list path). Add regression with star-vs-line mismatch.
2. Move override-tensor (`anchor_indices`, `coarse_mask`) device check/normalization BEFORE any `.item()` call. Add meta-device override regression.
3. Tighten Floyd check to exact equality (`torch.equal` on masked finite entries, or `allclose(..., atol=0, rtol=0)`). Add sub-1e-6 mismatch regression + finite-pattern mismatch regression on disconnected graph.
4. Add strict int + non-bool validation for `local_radius`, `max_chain_chunk_len`, `temporal_stride` (mirroring the d_model/max_coarse pattern at `pool_dynamic.py:97-105`).
5. Fill test gaps: adjacency NaN/Inf direct pool test; `mincut_lambda` non-finite hparam test.

## Paths

- FULL: `docs/codex_reviews/m1_2_pool_dynamic_round13_FULL.txt`
- BRIEF: `docs/codex_reviews/m1_2_pool_dynamic_round13_BRIEF.md`
