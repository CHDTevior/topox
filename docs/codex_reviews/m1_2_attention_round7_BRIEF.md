# M1.2 attention.py Round 7 — Codex Review BRIEF

- **Thread**: `019e43d7-e108-7d63-8216-7f115cebf3ee` (fresh, no threadId reuse)
- **Model**: gpt-5.5, reasoning effort xhigh
- **Date**: 2026-05-20

## Verdict

| Field | Value |
| --- | --- |
| VERDICT | **NEEDS-FIX** |
| CONVERGED | **no** (new R12 found: 12th category) |
| PERFORMANCE | **add-toggle** (`validate_inputs: bool = True`) |
| UNBLOCK_POOL_DYNAMIC | **NO-GO** until R12 #12 fixed |

## Q1 — All 11 R12 sealed (including round-6 adj/geo cross-consistency)?

Yes. All 11 enforced; codex traced each to a specific line in `attention.py`.
Round-7 cross-consistency: Floyd recompute catches BOTH (a) shortcut distances
(via finite-value compare at `attention.py:286`) and (b) finite distance
between disconnected valid nodes (via finite-pattern compare at
`attention.py:277`). New tests at `test_graph_attention.py:363,380` cover both.

## Q2 — 12th R12 (NEW)

**Unbounded adjacency magnitude.**

- **Scenario**: line-graph adjacency with edges `1_000_000.0` instead of `1.0`,
  geodesic equal to correct hop distances.
- **Why accepted**: adjacency check (`attention.py:179`) only enforces
  finite/non-negative/symmetric/zero-diagonal; Floyd only checks `> 0`, so
  cross-consistency still passes.
- **Why it corrupts**: `adjacency_bias(adjacency.unsqueeze(-1))` at
  `attention.py:318` injects raw magnitude into attention logits — softmax
  saturates. Codex verified: `adj * 1_000_000` changed output by max abs
  diff ~0.74 vs binary adjacency.
- **Local producer contract**: `batch.py:426` and `graph_utils.py:246`
  produce binary adjacency, so `[0, 1]` bound is the natural contract.

## Q3 — Simplicity / redundancy

**No strict redundancy.** Geodesic-only checks overlap with Floyd consistency
on valid pairs but also catch padded/invalid-pair garbage that the
cross-consistency check intentionally masks out (`both_valid` at line 276).
Validation-heavy but not over-engineered under R12 fail-loud goals.

## Q4 — Surgical scope

**Scope drift advisory** (cosmetic): code changes correctly confined to
`src/models/graph_salad/` + `tests/`, but `docs/codex_reviews/` untracked
files are outside the stated round-7 scope. Not a functional issue.

## Q5 — Performance

**Add toggle.** Floyd is Python-loop with N iterations per forward; not
"4M ops" but N kernel launches. Acceptable for one-shot module call at
N≤160, NOT acceptable as repeated denoiser hot-path across diffusion
timesteps × layers. Recommend `validate_inputs: bool = True` default-on,
caller can validate once at batch/pool boundary and skip in hot loops.

## Q6 — pool_dynamic.py

**NO-GO.** Dynamic pooling is precisely where soft-adjacency ambiguity
enters; sealing the magnitude contract first prevents bad interface
inheritance.

## FIX_LIST

1. `src/models/graph_salad/attention.py:179` — enforce valid-pair
   adjacency in `[0, 1]`, plus regression test for symmetric huge
   adjacency that currently passes silently.
2. `src/models/graph_salad/attention.py:119` — add `validate_inputs:
   bool = True` ctor/forward arg for hot-path bypass after one
   explicit validation point.

## Tests

Codex ran the 27 tests via direct unittest invocation (pytest not
installed in its env): **27/27 PASS**.

## Saved artifacts

- `docs/codex_reviews/m1_2_attention_round7_FULL.txt`
- `docs/codex_reviews/m1_2_attention_round7_BRIEF.md`
