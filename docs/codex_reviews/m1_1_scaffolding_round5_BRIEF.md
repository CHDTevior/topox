# M1.1 Scaffolding — Codex Review Round 5 (BRIEF)

- **Model / effort**: gpt-5.5, xhigh
- **Thread**: `019e437d-259e-7491-bc71-b27fadb23658` (FRESH, no continuation)
- **Date**: 2026-05-20

## Verdict
**NEEDS-FIX** — round 5 sealed cardinality + parent-tree validity (R4 leak), but fresh-thread surfaced a **7th R12 category**: graph-semantics consistency (adjacency / geodesic_dist vs parent_indices) is silent.

## R12 categories
| Cat | Status | Evidence |
|---|---|---|
| (a) tensor schema | SEALED | batch.py:172-205, 211-218 |
| (b) cross-tensor T/J | SEALED | batch.py:245-267 |
| (c) scalar range + mask | **PARTIAL LEAK** | batch.py:279-294 — sum-only; non-prefix mask holes pass |
| (d) NaN/Inf | SEALED | batch.py:205-209, 303-311 |
| (e) bool-as-int | SEALED | batch.py:357-365 |
| (f) cardinality + tree | SEALED | batch.py:351-356, 374-382; graph_utils.py:31-84 |
| **(g) graph semantics** | **LEAK (NEW)** | batch.py:211-218 — adjacency / geodesic_dist never cross-checked against parent_indices, symmetry, binary, zero diagonal, padded rows/cols |

## R2 over-engineering
**None.** All ~440 LoC branches are load-bearing. Only suggest refreshing stale validation-order docstring around batch.py:128-134.

## M1.2 unblock
**NO-GO** — pool/VAE could receive a valid `parent_indices[b]` while `adjacency[b]` / `geodesic_dist[b]` describe an unrelated (or all-zero) graph. Synthetic happy-path test currently uses zero adjacency with a valid tree (test_scaffolding.py:48,64,77-80), proving the leak is exercised but undetected.

## Fix list
1. **batch.py:211-218 + after :374-382** — per-sample graph-semantics validator: build expected undirected adj from `parent_indices[b]`, compare to `adjacency[b,:nj,:nj]`; enforce binary, symmetric, zero diagonal, zero padded rows/cols under `joint_mask`.
2. **batch.py:279-294** — upgrade mask checks from sum-only to prefix-contiguous: `joint_mask[b,:nj].all()` AND `~joint_mask[b,nj:].all()`; same for `frame_mask` / `num_frames`.
3. **batch.py:211-218** — validate `geodesic_dist` against `adjacency` (recompute or cross-check); current finite-only accepts all-zero distances on disconnected pairs.
4. **graph_utils.py:31 + batch.py:26** — promote `_validate_parent_tree` → public `validate_parent_tree` (cross-module reuse should not import `_`-prefixed); keep `_validate_parent_tree = validate_parent_tree` alias.

## Other findings
- Validation order is dependency-safe (cardinality uses `num_joints.tolist()` only after scalar dtype/shape validation), but does not exactly match prompt/docstring (batched scalars checked before cross-tensor T/J — non-blocking, just refresh comment).
- All 39 tests pass; no round-5 test is no-op'd by cardinality preemption.
- Denoiser stub has no M1 blocker but is signature-only — negative `d_model`/`n_heads` accepted (no allocator validates). Not a round-5 regression; tracked as M1.x deferred.

## Saved
- FULL: `/scratch/ts1v23/workspace/noKslot_clean/docs/codex_reviews/m1_1_scaffolding_round5_FULL.txt`
- BRIEF: `/scratch/ts1v23/workspace/noKslot_clean/docs/codex_reviews/m1_1_scaffolding_round5_BRIEF.md`
