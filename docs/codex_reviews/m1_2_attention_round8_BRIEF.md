# M1.2 attention.py — ROUND 8 review (BRIEF)

- **Codex thread**: `019e43e0-01a6-7692-aad1-7c0654da6304`
- **Model**: gpt-5.5, xhigh, fresh thread (no codex-reply continuation)
- **Date**: 2026-05-20
- **Files reviewed**: `src/models/graph_salad/attention.py` (~360 LoC), `tests/test_graph_attention.py` (29 cases, all PASS in ~0.07s)

## Verdict
**PASS** — 29/29 tests pass.

## R12 coverage (12 categories all sealed)
All 12 R12 fail-loud categories enforced with line-anchored citations + corresponding tests, including round 7's adjacency-range cap (category 6: `attention.py:L202-L207` ↔ `test_adjacency_above_one_raises`).

## 13th R12 search
**Convergence declared.** No new `validate_inputs=True` fail-loud gap found. Earlier rounds moved from broad contract holes to narrow numeric/topology edge cases; round 8 sealed the last known magnitude hole. Remaining ambiguities are policy choices, not silent corruption (soft-edge Floyd semantics, disconnected-valid neutralized `+Inf` bias, padded-query caller masking — all by design).

## validate_inputs=False bypass safety
**Safe-with-contract** (unsafe as a public unchecked API — by design).

Per-input audit (from codex's actual verification):
- `adj > 1` / `adj < 0` → finite but semantically wrong (out-of-contract).
- geodesic NaN → propagates NaN (verified `has_nan=True`).
- geodesic `-Inf` → silently neutralized at `attention.py:L346-L347` (out-of-contract).
- all-False `node_mask` → returns finite garbage `(1,4,16)` (out-of-contract).
- batch singleton mismatch (B=2 vs B=1) → silent broadcast — **biggest footgun**, but default validation catches it.
- int/bool dtype mismatch → crashes in `Linear` / `masked_fill` (loud, not silent).
- `d_model` not divisible by heads → impossible (constructor guard `L85-L88`).

Codex does NOT call this NEEDS-FIX: bypass is explicitly documented at `attention.py:L134-L138` as "caller asserts inputs already validated". Hot-path callers (denoiser timestep loop) validate once before entering the loop. Contract is sufficient.

## R3 surgical scope
**OK** — implementation/tests under `src/models/graph_salad/` + `tests/`. Advisory: untracked `docs/codex_reviews/*` artifacts can be committed separately from M1.2 code commit if R3 is strict.

## R2 simplicity / scope-creep
**Not over-engineered.** 360 LoC is mostly explicit fail-loud checks + comments; `_compute()` stays simple. **No deletions recommended** — adj `[0,1]` cap does NOT make prior checks redundant (NaN comparisons don't catch NaN; symmetry/zero-diagonal independent; Floyd consistency still catches bounded-but-wrong geodesics).

## M1.3 pool_dynamic.py
**GO** — attention contract sealed; only non-functional review-artifact commit hygiene remains.

## Fix list
None (PASS).
