# M1.2 attention.py — Round 5 codex fresh-eyes review BRIEF

- **Verdict**: NEEDS-FIX
- **Model**: gpt-5.5 / xhigh
- **Thread**: `019e43c6-fd24-7aa2-a964-7b17e75b69ec` (FRESH; no continuation)
- **Date**: 2026-05-20
- **Tests**: 22/22 PASS at review time

## Round-4 leak status (all 3 sealed)

| Leak | Status | Evidence |
|---|---|---|
| (a) adjacency large-absolute asymmetry | SEALED | `atol=1e-6, rtol=0.0` at attention.py:170; regression test at test_graph_attention.py:300 (adj=1e6 vs 1e6+1, asserts ValueError). Float64 does not bypass. |
| (b) geodesic finite/+Inf pattern asymmetry | SEALED (stronger than reported) | NaN rejected at attention.py:185 BEFORE finite-pattern check; two-stage finite/+Inf symmetry at attention.py:206-217. NaN cannot slip past as "infinite". |
| (c) docstring binary→weighted | SEALED | attention.py:50-54 says "non-negative weighted (binary OK as special case), symmetric, zero diagonal"; validation matches at lines 158/162/170/177. |

## New 10th R12 hole (concrete, runnable)

**Dtype contract not fail-loud.** Two runnable failure traces:

1. `block.half()` + fp16 inputs: passes validation, crashes at attention.py:268 with `RuntimeError: value cannot be converted to type c10::Half without overflow`.
2. Mixed dtype (x=float32, adj/geo=float64): `RuntimeError expected m1 and m2 to have the same dtype`.

Both errors emerge AFTER compute begins, violating validation-order-monotonicity.

## Floyd hop-count deferral: NOT defensible

Probe: `geo[0,9]=geo[9,0]=1e6` with N=10, `geodesic_bias.weight=1`, zero q/k, valid line graph → forward passes with finite output, but head-0 row-0 attention becomes `[0,0,0,0,0,0,0,0,0,1]`. Since `geodesic_bias` is unconstrained, a corrupt large finite geodesic value silently dominates attention. **MUST-FIX-NOW.**

## pool_dynamic.py unblock: NO-GO

Blocks: missing dtype fail-loud + missing finite geodesic upper-bound check. Structural adj/geo cross-consistency may remain deferred if upstream generation is trusted.

## Fix list

- **FIX-1** (geodesic hop-bound) → `src/models/graph_salad/attention.py:195-200`:
  ```python
  if (finite_geo > (N - 1)).any():
      raise ValueError(f"GraphAttentionBlock: geodesic_dist finite entries exceed Floyd hop bound N-1={N - 1}")
  ```
- **FIX-2** (dtype fail-loud) → `src/models/graph_salad/attention.py:141-146`:
  ```python
  if x.dtype not in (torch.float32, torch.float64) or x.dtype != self.q_proj.weight.dtype or adjacency.dtype != x.dtype or geodesic_dist.dtype != x.dtype:
      raise ValueError("GraphAttentionBlock: x/adjacency/geodesic_dist must be float32/float64 and match module dtype")
  ```

## Non-blocking observations

- atol=1e-6 boundary is inclusive; `adj[0,1]=1.0, adj[1,0]=1.0+1e-6` passes — undocumented except by code tolerance.
- Disconnected `+Inf` sub-block passes (matches docstring intent — OK).
- adj/geo cross-consistency not checked (`adj=0 but geo=1` passes) — defer unless upstream trust assumption fails.
- R3 surgical scope NOT cleanly confirmed: worktree shows `M src/models/graph_salad/__init__.py` plus untracked review docs; codex did not open review docs per instruction. Verify before next commit.
- R2 simplicity: 286 LoC justified; a `_validate_inputs(...)` helper would be reasonable after FIX-1/FIX-2 land but not required.

## Paths

- Full transcript: `/scratch/ts1v23/workspace/noKslot_clean/docs/codex_reviews/m1_2_attention_round5_FULL.txt`
- Brief (this file): `/scratch/ts1v23/workspace/noKslot_clean/docs/codex_reviews/m1_2_attention_round5_BRIEF.md`
