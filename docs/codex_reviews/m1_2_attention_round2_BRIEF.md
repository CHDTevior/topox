# M1.2 attention.py — Codex Round 2 BRIEF

**Verdict**: NEEDS-FIX
**Thread**: `019e43a6-7c77-77a2-a208-642f07e1ae3a` (FRESH, no continuation)
**Model**: gpt-5.5, xhigh
**Date**: 2026-05-20

## Round-1 R12 sealing — all 6 actually enforced

| Round-1 silent-failure mode | Guard | Test | Sealed |
|---|---|---|---|
| nonfinite_x | attention.py:139-142 | test_graph_attention.py:158-166 | Y |
| geodesic_nan_or_minus_inf | attention.py:150-157 | test_graph_attention.py:178-196 | Y |
| adjacency_nan | attention.py:144-147 | test_graph_attention.py:168-176 | Y |
| all_false_node_mask | attention.py:162-167 | test_graph_attention.py:198-206 | Y |
| dropout_p_eq_1 | attention.py:79-80 | test_graph_attention.py:208-212 | Y |
| comment lie | attention.py:181-189 | test_graph_attention.py:138-154 (behavior, not text) | Y |

All 12/12 tests pass; messages are explicit and regex-asserted in tests.

## NEW silent-failure mode (7th)

**`B=0` empty batch silently returns shape `(0, N, D)`** instead of raising. Verified `B=0,N=4` returns `(0,4,16)`. Guards at attention.py:118-132 accept zero batch, mask sanity at :162-167 is vacuous for empty batch.

Other hunted cases (clean): x=-Inf caught, n_heads=1 works, N=0 raises via mask sanity, dim/head divisibility caught at :70-73, dtype mismatch fails via LayerNorm/Linear, dropout=0.0 accepted, geodesic cloned before in-place substitution (:188-189).

## R3 surgical scope — NOT clean

Two unstated edits beyond the two requested files:
- `src/models/graph_salad/__init__.py:10, 22-24` modified to export `GraphAttentionBlock`
- Untracked stale round-1 review artifacts: `docs/codex_reviews/m1_2_attention_BRIEF.md`, `m1_2_attention_FULL.txt`

No spillover in encoder.py, graph_utils.py, batch.py, denoiser_stub.py, or scaffolding tests.

## R2 simplicity

+30 LoC justified for fail-loud. Minor: `node_mask.any(dim=1)` recomputed at :162-163 — cache once if touching.

## Ckpt-key isomorphism vs encoder.py

ALL keys + shapes match (q_proj/k_proj/v_proj/o_proj, geodesic_bias.weight, adjacency_bias.weight, norm1/norm2, ff.0/ff.3). Constructor signatures match (attention.py:60-66 vs encoder.py:30). No temperature/scale buffers in either. **Drop-in warm-start verified.**

## MUST-FIX-NOW

1. **attention.py:122** — add explicit `B > 0` (and `N > 0`) guard before finite checks; add empty-batch and zero-nodes regression tests.
2. **`__init__.py:10, 22-24`** — either declare this export edit as in-scope for M1.2 or revert.
3. **stale round-1 review artifacts** (`m1_2_attention_BRIEF.md`, `m1_2_attention_FULL.txt`) — remove or replace before claiming surgical scope.

## DEFERRED

- Add explicit -Inf-on-x, +Inf-on-adj, dtype/device-mismatch tests if direct (non-GraphMotionBatch) use becomes common.
- Document zero-diagonal/no-self-loop adjacency contract at attention.py:50-51 to match batch.py:427-452.

## pool_dynamic.py unblock

**NO-GO** — `B=0` silently returns + R3 scope not clean. Seal MUST-FIX-NOW #1 (and ideally #2/#3) before starting pool_dynamic.py.
