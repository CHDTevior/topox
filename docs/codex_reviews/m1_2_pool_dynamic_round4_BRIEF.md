# M1.2 pool_dynamic.py — Round 4 Codex Review (BRIEF)

- Model: gpt-5.5 xhigh
- Thread: `019e4584-0eff-7fa1-b9df-9898b697393c` (fresh)
- Date: 2026-05-20

## Verdict: NEEDS-FIX

## (1) Eight R12 categories — all sealed YES

| # | category | code-trace |
|---|----------|------------|
| a | mincut ortho per-sample C_valid | `pool_dynamic.py:336-352` |
| b | parent length per-b vs joint_mask.sum | `pool_dynamic.py:478-491` |
| c | full XOR (parents + anchor pair) | `pool_dynamic.py:466-472` |
| d | partial XOR (one of anchor pair) | `pool_dynamic.py:459-465` |
| e | inactive-slot anchor range | `pool_dynamic.py:516-529` |
| f | odd T % stride | `pool_dynamic.py:448-454` |
| g | all-False joint_mask | `pool_dynamic.py:438-446` |
| h | override anchor invariants (dup / unsorted / zero-active / -1-active / padded-target) | `pool_dynamic.py:533-560` |

Tests: 22/22 PASS (verified by codex `python tests/test_pool_dynamic.py`).

## (2) 9th R12 — YES (concrete probe)

**Coarse_mask prefix-active contract not enforced.** Rule-based `_select_anchors` (`pool_dynamic.py:140-155` + `graph_utils.py:457`) writes active anchors contiguously from slot 0 then pads the tail. Override path validates sort/uniqueness but allows holes.

Probe (silent finite output):
```python
anchor_indices = torch.tensor([[-1, 0, 2, 5]])
coarse_mask   = torch.tensor([[False, True, True, True]])  # hole at slot 0
joint_mask    = torch.ones(1, 6, dtype=torch.bool)
```
Passes all current checks → `pooled_mask=[[F,T,T,T]]`, finite aux losses. Bakes an incompatible slot convention into the pooling contract.

## (3) R3 surgical: FAIL (cosmetic)

Inside `pool_dynamic.py` + `tests/test_pool_dynamic.py` edits are validation-only & surgical. `git status` shows `src/models/graph_salad/__init__.py` (M) and `docs/codex_reviews/*` (??) — same review-artifact drift as round 3; declare carve-out.

## (4) Math correctness: PASS

- Feature pooling `S^T X / mass` unchanged at `:257-259`
- Coarse adjacency (hard-argmax binary) unchanged at `:297-303`
- MinCut cut + per-sample ortho unchanged at `:323-354`

## (5) pool_deterministic.py unblock: NO-GO

Shared pooling contract (slot prefix convention) must be fail-loud before deterministic twin is written. Otherwise deterministic path inherits the same hole.

## Fix list (round 5)

1. **`pool_dynamic.py:533`** — add per-sample prefix-active validation: first `C_valid` slots of `coarse_mask[b]` must be True, remaining False. (E.g., `coarse_mask[b].nonzero() == arange(C_valid)`.)
2. **`tests/test_pool_dynamic.py`** — add `test_anchor_coarse_mask_hole_raises` with `anchor_indices=[[-1,0,2,5]]`, `coarse_mask=[[F,T,T,T]]`, expecting `ValueError`.

## Paths

- Full: `/scratch/ts1v23/workspace/noKslot_clean/docs/codex_reviews/m1_2_pool_dynamic_round4_FULL.txt`
- Brief: this file
