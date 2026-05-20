# M1.2 pool_dynamic.py round 10 — codex review BRIEF

- **Model**: gpt-5.5 xhigh
- **Thread**: `019e45ae-bd7f-73a1-b59e-862fc943a5df` (FRESH)
- **Date**: 2026-05-20
- **Local verify**: 39 tests PASS in 0.162s

## Verdict: NEEDS-FIX

Three R12 categories not fully sealed by test evidence (code OK), plus one new lurking gap (Q2: device consistency).

## R12 seal status (17 categories)

- SEALED (12): parents-len (#2), full XOR (#3), partial XOR (#4), odd-T (#6), all-false joint_mask (#7), override ascending+unique (#8), override zero-active (#9), coarse_mask prefix (#10), joint_mask prefix (#11), override missing root (#12), adj contract (#15), geo contract (#16)
- SEALED-by-code, partial-test (2): joint_features finite #13 (NaN tested, Inf missing), skeleton_embeddings finite #14 (same)
- **NOT-SEALED** (3):
  - #1 mincut ortho per-sample — code per-sample (`pool_dynamic.py:337-357`), but tests only check finiteness (`:94`, `:531`); no per-sample-vs-batch regression
  - #5 inactive slot range — code zeros via `P` masking (`:239-243`, `:732-735`), but tests only assert assignment columns are zero (`:171-174`); no assertion that `pooled_features` / `pooled_skeleton_embeddings` of inactive slots are zero
  - #17 dtype/strict — code OK (`:96-100`, `:443-455`), but tests miss `d_model=True`, and float32-reject on `skeleton_embeddings` / `adjacency` / `geodesic_dist`

## Q2: 18th failure mode — FOUND (pool_dynamic-owned)

**Device consistency unchecked.** `forward` records `device = joint_features.device` at `:614` and only moves generated anchors and `coarse_mask`. It does not assert that `joint_features`, `skeleton_embeddings`, `adjacency`, `geodesic_dist`, `joint_mask`, `frame_mask`, and module params share device. Mixed CPU/CUDA inputs fail deep at `_compute_assignment` (`:188-204`), override indexing (`:677-683`), or einsum (`:262`) instead of fail-loud contract error.

No gradient-flow blocker: detached returns are diagnostic only (`mincut_cut`/`mincut_ortho` at `:382-385`); trainable `mincut` stays attached (`:359`, `:385`).

## Q3: pool_deterministic unblock — **NO-GO**

3 unsealed categories + 1 owned-here 18th gap → block downstream.

## Fix list

- FIX-1: per-sample mincut ortho regression test @ `tests/test_pool_dynamic.py:94`
- FIX-2: assert inactive-slot `pooled_features` / `pooled_skeleton_embeddings` == 0 @ `tests/test_pool_dynamic.py:148`
- FIX-3: dtype tests for `d_model=True`, `skeleton_embeddings.double()`, `adjacency.double()`, `geodesic_dist.double()` @ `tests/test_pool_dynamic.py:52`, `:229`
- FIX-4: explicit device-consistency validation in forward @ `src/models/graph_salad/pool_dynamic.py:443`

## Paths

- FULL: `docs/codex_reviews/m1_2_pool_dynamic_round10_FULL.txt`
- BRIEF: `docs/codex_reviews/m1_2_pool_dynamic_round10_BRIEF.md`
