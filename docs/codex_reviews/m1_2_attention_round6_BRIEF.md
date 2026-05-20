# M1.2 attention.py round 6 — codex review BRIEF

- **Model**: gpt-5.5, xhigh
- **Thread ID** (fresh): `019e43cf-1ac2-7352-86df-b3e2ccc9955d`
- **Date**: 2026-05-20
- **Files**: `src/models/graph_salad/attention.py` (313 LoC), `tests/test_graph_attention.py` (363 LoC, 25 tests, all PASS)

## Verdict

**NEEDS-FIX — UNBLOCK: NO-GO**

## What was sealed (rounds 1-6)

All 12 R12 categories from rounds 1-5 verified sealed with concrete file:line traces (NaN/Inf, all-False mask, dropout range, empty B/N, topology semantics, large-abs asymmetry rtol=0, finite-vs-+Inf pattern, Floyd hop upper bound, dtype contract). 25/25 tests PASS in 0.07s.

## New R12 found (round 6, 11th category)

**adjacency/geodesic consistency** is unsealed. A geodesic matrix that is finite, symmetric, non-negative, diagonal-zero, ≤ N-1, but **numerically inconsistent with the adjacency** (e.g., 4-node line with `geo[0,3]=2.0` instead of 3.0) passes all current validators. Attention silently consumes a topology-corrupting bias.

**Concrete repro**: 4-node line adjacency + geo `[[0,1,2,2],[1,0,1,2],[2,1,0,1],[2,2,1,0]]` (should have `geo[0,3]=geo[3,0]=3.0`).

**Minimal fix**:
1. `attention.py:37`: import `floyd_shortest_path` from `graph_utils`.
2. After `attention.py:264`: call `floyd_shortest_path(adjacency, node_mask)` and `allclose`-check vs `geodesic_dist` on valid pairs only; raise ValueError on mismatch.
3. Add regression test for the 4-node line shortcut above.

## Secondary advisories (non-blocking, optional cleanup)

- Inf branches for `x` and `adjacency` (same `isfinite` predicate as NaN) are not separately tested — coverage gap, not a bug.
- `geo` large-absolute asymmetry (analogous to existing `adj` test at L300-311) lacks dedicated test.
- Boundary-INCLUSIVE hop-bound test (max == N-1 OK) only covered indirectly via `test_forward_shape`.
- `both_finite = finite_g & finite_gt` at line 241 is redundant (after pattern-equality check passes, `finite_g == finite_gt`); harmless.
- No device check (mixed CPU/CUDA raises opaquely, not silently corrupt — same hygiene class as dtype).

## Q5/Q6

- **R3 surgical**: code edits scoped to `src/models/graph_salad/` + `tests/`. The current worktree also has `docs/codex_reviews/*` artifacts which are out-of-scope from the code-change perspective.
- **R2 simplicity**: 313 LoC justified by 5 rounds of real defect-sealing; no over-engineering, no dead code.

## Decision

**NO-GO for `pool_dynamic.py` consumption** until adjacency↔geodesic consistency is enforced. `pool_dynamic.py` could hand in a bounded-but-wrong geodesic and attention would silently corrupt training.

## Saved outputs

- Full: `/scratch/ts1v23/workspace/noKslot_clean/docs/codex_reviews/m1_2_attention_round6_FULL.txt`
- Brief: `/scratch/ts1v23/workspace/noKslot_clean/docs/codex_reviews/m1_2_attention_round6_BRIEF.md`
