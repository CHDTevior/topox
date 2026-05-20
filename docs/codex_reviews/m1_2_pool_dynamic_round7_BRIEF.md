# M1.2 pool_dynamic.py — Round 7 Codex Review (BRIEF)

- **Reviewer**: codex / gpt-5.5 / xhigh (FRESH thread, no codex-reply)
- **Thread**: `019e459a-136e-71f0-af5f-cfeababca83f`
- **Date**: 2026-05-20
- **Files**: `src/models/graph_salad/pool_dynamic.py` (651 LoC), `tests/test_pool_dynamic.py` (445 LoC, 27 cases, 27/27 PASS in 0.07s)

## Verdict

**`--NEEDS-FIX`**. Convergence **NOT** reached after Round 7. R12 #12 (override missing root) IS sealed, but codex (in fresh-context audit) finds 4 previously-claimed-sealed R12 are actually **unsealed**, 3 are **partial** (missing negative tests), plus a candidate 13th leak.

## Q1 — Status of the 12 known R12 (concrete file:line)

| # | Contract | Status | Code trace | Test trace |
|---|---|---|---|---|
| 1 | T < anchor count | sealed | `pool_dynamic.py:148` | `:128` |
| 2 | K_total / C_max mismatch | **partial** | `:521,:530` | missing negative test |
| 3 | slot/model dim mismatch | **partial** | `:410,:418` | only `joint_features` covered (`:396`) |
| 4 | empty graph V=0 | **partial** | `:416,:441` | missing `J=0` test (`:290` covers all-false) |
| 5 | NaN/Inf in features | **UNSEALED** | no finite guard before `:611` | missing |
| 6 | adjacency not symmetric | **UNSEALED** | only shape checked `:423` | missing |
| 7 | adjacency self-loop policy | **UNSEALED** | no input rejection | missing rejection test (`:102` is output-side smoke) |
| 8 | undefined distance | sealed | `:224` | `:139` |
| 9 | joint coverage <100% | sealed | `:221,:224` | `:139` |
| 10 | slot assignment ties | **UNSEALED** | hard argmax `:297` | missing |
| 11 | override invalid anchors | sealed | `:540,:581,:589` | `:219,:299,:309,:389` |
| 12 | override missing root (NEW) | **sealed** | `:599,:600` | `:342` |

## Q2 — 13th R12 / convergence

**Convergence: NO.** Codex finds concrete unsealed silent paths:
- NaN in `joint_features` / `skeleton_embeddings` → non-finite pooled output, no raise.
- Asymmetric adjacency + nonzero diagonal → silently accepted.
- Tie repro: `locality_alpha=0` + zero `skeleton_embeddings` + anchors `[0,2,5]` → uniform `[1/3,1/3,1/3]`; hard `argmax` at `:297` silently picks slot 0.
- Candidate 13th: `geodesic_dist` NaN/-Inf accepted — `:203,:361` substitute all `isinf` incl. `-Inf`; NaN masked at `:232` without raising.

No `try/except: pass` or `warnings.warn`. Fallback is missing semantic validation + `isinf` substitution on geodesics.

## Q3 — pool_deterministic.py unblock

**NO-GO.** Would inherit unresolved finite-input, graph-topology, geodesic, and tie-policy contract gaps.

## Fix list (Round 8 scope)

1. **R12 #5** — finite-value guard on `joint_features`, `skeleton_embeddings`, `geodesic_dist` (NaN + +Inf + -Inf) → raise.
2. **R12 #6** — adjacency symmetry check → raise.
3. **R12 #7** — adjacency self-loop policy (decide: forbid or document; if forbid → raise on nonzero diagonal).
4. **R12 #10** — slot assignment tie policy (deterministic tiebreak OR raise on detected tie under exact-zero embedding + zero locality_alpha).
5. **R12 candidate-13** — geodesic NaN/-Inf semantic validation (separate "+Inf = unreachable" from "-Inf/NaN = invalid input").
6. **Negative tests** — add for R12 #2 (K_total/C_max), #3 (slot dim mismatch on `skeleton_embeddings` not just `joint_features`), #4 (`J=0` case).

## Paths

- FULL: `/scratch/ts1v23/workspace/noKslot_clean/docs/codex_reviews/m1_2_pool_dynamic_round7_FULL.txt`
- BRIEF: `/scratch/ts1v23/workspace/noKslot_clean/docs/codex_reviews/m1_2_pool_dynamic_round7_BRIEF.md`
