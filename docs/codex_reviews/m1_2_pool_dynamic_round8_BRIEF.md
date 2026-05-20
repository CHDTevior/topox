# M1.2 pool_dynamic.py — Round 8 Codex Review (BRIEF)

**Model**: gpt-5.5 xhigh
**Thread**: `019e45a0-7c96-7580-acd8-d5251574f4a4` (FRESH — no continuation)
**Date**: 2026-05-20
**Verdict**: **NEEDS-FIX**

## Q1 — 13 R12 categories status

| # | Category | Status | File:line |
|---|---|---|---|
| A | joint_features.shape | PASS | pool_dynamic.py:410-417 |
| B | joint_features finite | PASS (new) | pool_dynamic.py:441-444 |
| C | skeleton_embeddings.shape | PASS | pool_dynamic.py:418-422 |
| D | skeleton_embeddings finite | PASS (new) | pool_dynamic.py:445-448 |
| E | adjacency.shape | PASS | pool_dynamic.py:423-427 |
| F | adjacency finite | PASS (new) | pool_dynamic.py:451-454 |
| G | adjacency symmetry | PASS (new) | pool_dynamic.py:455-459 |
| H | adjacency zero-diagonal | PASS (new) | pool_dynamic.py:460-464 |
| I | geodesic.shape | PASS | pool_dynamic.py:423-427 |
| J | geodesic finite-or-+Inf | PASS (new) | pool_dynamic.py:465-474 |
| **K** | **geodesic symmetry** | **FAIL** | guard missing; reaches pool_dynamic.py:648-652 |
| **L** | **geodesic zero diagonal** | **FAIL** | guard missing; gathers at :199 and :359 |
| M | K_max / pool dim consistency | PASS | pool_dynamic.py:98-99, 147-152, 559-572 |

## Q2 — 14th R12 found (concrete)

**Adjacency value-domain not validated**. Negative or `>1` weights pass all current guards.
- Input: valid `B=1,T=4,J=6,D=16` line skeleton with `adj[0,2,3] = adj[0,3,2] = -1.0`
- Expected: raise (adjacency contract = binary/soft `[0,1]`)
- Actual (codex probed): `forward()` returned `NO_RAISE` — negative edge silently dropped by coarse binarization, distorts mincut degree.

## Q3 — pool_deterministic.py unblock

**NO-GO**. Geodesic symmetry/diagonal still missing + adjacency value-domain leak (14th R12). Three blockers.

## Fix list

1. Add `geodesic_dist` symmetry guard (transpose/allclose with `rtol=0.0`, handling `+Inf` pattern equality).
2. Add geodesic diagonal-zero guard.
3. Add adjacency value-domain guard: reject `<0` and `>1` (or document an explicit relaxed contract).
4. Add 5 regression tests: geo_asymmetric_pool, geo_nonzero_diag_pool, adj_negative_pool, adj_greater_than_one_pool, adj_nan_or_inf_pool (latter may already exist as F's adj_nan/inf — confirm coverage).

## Artifacts

- Full: `/scratch/ts1v23/workspace/noKslot_clean/docs/codex_reviews/m1_2_pool_dynamic_round8_FULL.txt`
- Brief: `/scratch/ts1v23/workspace/noKslot_clean/docs/codex_reviews/m1_2_pool_dynamic_round8_BRIEF.md`
