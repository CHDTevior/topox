# M1.2 pool_dynamic.py round 11 — codex review BRIEF

- **Model**: gpt-5.5 xhigh
- **Thread (FRESH)**: `019e45b6-7cbb-74f0-a256-f98449e85081`
- **Date**: 2026-05-20
- **Local verify**: 45 tests PASS (graph_salad env, 6.89s)

## Verdict: NEEDS-FIX

Round-11 closes dtype/`d_model=True`/inactive-output/one mixed-device path, but 4 R12 categories still not sealed by test evidence, plus 1 new 19th failure mode found.

## R12 seal status (18 categories)

- **SEALED (12)**: #2 parents-len, #3 full XOR, #4 partial XOR, #5 inactive zero outputs (now with explicit feature/embedding zero asserts), #6 odd-T, #7 all-false joint_mask, #8 override ascending+unique, #9 zero-active, #10 coarse_mask prefix, #11 joint_mask prefix, #12 missing root, #15 adj contract, #16 geo contract, #17 dtype/strict (incl. `d_model=True` + 4 float .double() tests)
- **SEALED-by-code-only (2)**: #13 joint_features finite — NaN tested, Inf untested (`test_pool_dynamic.py:295-300`); #14 skeleton_embeddings finite — same (`:302-307`)
- **NOT-SEALED (2)**:
  - #1 mincut ortho per-sample — code per-sample (`pool_dynamic.py:337-357`), tests only finite-smoke aux losses (`:94-96`, `:583-586`); no per-sample-vs-Cmax regression
  - #18 device consistency — code checks 5 input tensors only (`pool_dynamic.py:443-456`); `test_mixed_device_raises_pool` covers `skeleton_embeddings` meta-device only (`:279-285`); **module-param device guard absent** despite Round-11 claim

## Q2: 19th failure mode — FOUND (pool_dynamic-owned)

**`geodesic_dist` topologically inconsistent with `adjacency` / `parent_indices`.**

Concrete attack: J=6 line skeleton, valid adjacency/parents, but `geodesic_dist = torch.zeros(1, 6, 6)`. Passes current geodesic guards (finite, nonnegative, symmetric, zero-diagonal) at `pool_dynamic.py:511-549`. Then `_compute_assignment` uses bogus geodesic for locality and candidate masks (`:200-217`), every anchor appears distance 0, graph-local pooling collapses to embedding-only assignment. Codex probed it: forward passed, locality loss = 0.0. Silent-corruption class.

## Q3: pool_deterministic unblock — **NO-GO**

#1, #13/#14, #18, plus new 19th geodesic-consistency gap → block M1.3.

## Fix list (Round-12 targets)

1. **FIX-1**: per-sample mincut orthogonality regression test that would fail under old `C_max` target, not just finite aux check @ `tests/test_pool_dynamic.py:94-96`
2. **FIX-2**: explicit `+Inf` tests for `joint_features` and `skeleton_embeddings` (#13/#14 Inf-branch) @ `tests/test_pool_dynamic.py:295-307`
3. **FIX-3**: extend device validation in `forward(...)` to module parameters and override tensors (`anchor_indices`, `coarse_mask`); add module-param mixed-device test @ `pool_dynamic.py:443` + new test
4. **FIX-4**: validate `geodesic_dist` against `adjacency`/`joint_mask` (Floyd or equivalent edge-distance consistency); add all-zero-geodesic mismatch regression @ `pool_dynamic.py:511` + new test

## Paths

- FULL: `docs/codex_reviews/m1_2_pool_dynamic_round11_FULL.txt`
- BRIEF: `docs/codex_reviews/m1_2_pool_dynamic_round11_BRIEF.md`
