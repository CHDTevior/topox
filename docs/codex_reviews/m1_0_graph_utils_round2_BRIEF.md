# Codex Round 2 Review — M1.0 graph_utils.py — BRIEF

- **Date:** 2026-05-20
- **Model:** gpt-5.5 (xhigh)
- **Thread:** 019e433f-8ccf-7b70-a67b-42a47f4b2811 (continued)
- **Round 1 verdict:** NEEDS-FIX (4 P0 items)
- **Round 2 verdict:** **PASS**

## P0 fix items — one-liners

| # | Item | Status | Evidence |
|---|------|--------|----------|
| P0-1 | `build_coarse_adjacency_from_hard_assign` no-clamp / fail-loud | **PASS** | Raises on `<0`/`>=C` and on inactive `coarse_mask=False` ids; padded joints stay ignored. (graph_utils.py:203-232) |
| P0-2 | Shared `_validate_parent_tree` called from 3 sites | **PASS** | Validator covers single-root, range/self-loop, BFS reachability (graph_utils.py:31-84); invoked at graph_utils.py:300, 342, 409. |
| P0-3 | Remove dead `short_limb_chunk_len` | **PASS** | Public signature now only `max_chain_chunk_len` (graph_utils.py:364-368). |
| P0-4 | New test coverage | **PASS** | 39 tests pass; covers invalid assign, inactive coarse, mask dtype, shape mismatch, Floyd disconnected, parent validation, exact chunk positions. |

## Regressions

**None detected.** Padded-invalid case still passes (test_graph_utils.py:138-150) — the new range check did not break `test_padding_ignored`.

## Validator coverage

**Sufficient for M1.0.** `J=0` returns OK and callers return empty; `J=1` root-only works. Cycles not connected to root are reported as disconnected rather than always "cycle", but still fail loud — enough for the pool contract.

## Test semantics

**Sound.** `test_line_chunked_exact_positions` correctly locks leaf-to-root chunking direction: line `0..10`, chunk 3 → `[0,1,4,7,10]` (test_graph_utils.py:324-345), matching implementation (graph_utils.py:434-451).

## M1.1 unblock status

**GO.** M1.0 graph utility P0 blockers are resolved. M1.1 (scaffolding + batch + denoiser stub) can start.

## Remaining concerns (P1, non-blocking)

- Add explicit square check for `fine_adjacency.shape[1] == fine_adjacency.shape[2]` — today non-square input fails loudly later with PyTorch `RuntimeError`, not a clean `ValueError`.
- Consider device-consistency checks if pool code may pass CPU/GPU-mixed tensors.

## Files saved

- Full: `/scratch/ts1v23/workspace/noKslot_clean/docs/codex_reviews/m1_0_graph_utils_round2_FULL.txt`
- Brief: `/scratch/ts1v23/workspace/noKslot_clean/docs/codex_reviews/m1_0_graph_utils_round2_BRIEF.md`
