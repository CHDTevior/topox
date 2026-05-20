# M1.2 pool_dynamic.py Round 9 — Codex Review BRIEF

- **Verdict**: `NEEDS-FIX` (NO convergence)
- **threadId**: `019e45a8-760c-77e0-a75b-24849c706074`
- **Reviewer**: gpt-5.5 xhigh, fresh thread (no codex-reply)
- **Date**: 2026-05-20
- **Files**: `src/models/graph_salad/pool_dynamic.py` (P), `tests/test_pool_dynamic.py` (T)
- **Test suite**: 37 tests PASS, but contract still not sealed.

## R12 13-invariant status (post round 9)

Sealed (7/13): #3 finite features (NaN only), #7 adj symmetric, #8 adj zero diag, #9 adj `[0,1]`, #12 geo zero diag, #14 anchor override contract; #11 finite-value branch of geo symmetric.

Unsealed (3 critical + several test gaps):

| # | Invariant | Gap |
|---|-----------|-----|
| 1 | K positive **int** | `P:96-99` only checks numeric value; `max_coarse=3.5` and `=True` accepted |
| 2 | d positive **int** | `P:96-97` same issue; `d_model=16.0` only raises indirectly via `nn.Linear` |
| 5 | feature dtype float32 | `P:410-448` no dtype check; no test |
| 13 | geodesic **non-negative** | `P:478-513` checks `-Inf` and NaN but NOT finite `<0`; `geo[i,j]=-1` accepted |

Test gaps: #4 seed reproducibility untested; #6/#10 bad-shape adj/geo untested; #11 +Inf-pattern asymmetry branch (`P:494-498`) untested.

## Q2 — 15th lurking
- adjacency/geodesic **dtype** absent (real gap if M1.3 needs fail-loud topology boundary)
- empty graph N=0 covered in code (`P:416-417`) but untested
- single-node N=1 untested
- disconnected geodesic `+Inf` pattern branch untested

## Q3 — pool_deterministic unblock
**NO-GO**. M1.3 cannot scaffold against this as a frozen R12 fail-loud contract until #1, #2, #5, #13 sealed.

## Fix list (intent only)

- `P:96-99`: add `isinstance(x, int) and not isinstance(x, bool)` BEFORE value check for `d_model` and `max_coarse`
- `P:410-448`: add explicit `dtype == torch.float32` checks for `joint_features` and `skeleton_embeddings`
- `P:478-513`: add `if (geodesic_dist < 0).any(): raise ValueError(...)` (still allow `+Inf`)
- `T`: add fail tests for non-int/bool K/d, non-float32 features, negative-finite geodesic, +Inf-pattern asymmetry, seed reproducibility, bad-shape adj/geo

## Paths
- FULL: `docs/codex_reviews/m1_2_pool_dynamic_round9_FULL.txt`
- BRIEF: `docs/codex_reviews/m1_2_pool_dynamic_round9_BRIEF.md`
