# M1.2 pool_dynamic.py — Round 3 Codex Review (BRIEF)

- Model: gpt-5.5 xhigh
- Thread: 019e457c-49bb-72d1-ad2d-894f4af2615b (fresh)
- Date: 2026-05-20

## Verdict: NEEDS-FIX

## (1) Six R12 categories from rounds 1+2 — all sealed YES

| R12 | Status | Code-trace |
|-----|--------|------------|
| mincut ortho per-sample (C_valid) | sealed | `pool_dynamic.py:336-352` (per-batch mask + `1/sqrt(C_valid_b)` at `:347-350`) |
| parent length mismatch | sealed | `:468-481` (`len(parent_indices)==B` + per-b `len==joint_mask[b].sum()`) |
| multi-level XOR (full) | sealed | `:449-461` (both supplied → raise) |
| partial XOR (one of two) | sealed | `:449-455` (count==1 raises) |
| inactive slot anchor range | sealed | `:511-519` (`[-1, J)` enforced for all entries) |
| odd T % stride | sealed | `:438-444` (raised before `_pool_features`) |

Tests: 18/18 PASS. Minor test-coverage note: `test_inactive_slot_anchor_value` exercises only `>= J`, not `< -1`.

## (2) 7th R12 — YES (two new fail-loud holes)

**(a) Override-anchor invariants not validated (`:520-534`):**
- Module doc at `:25-28` promises sorted/unique active anchors; rule-based path enforces this via `graph_utils.py:421,457`.
- Override path validates only range + padded-joint validity, not duplicates or sorting.
- Probe: `anchor_indices=[[0,0,2,3]]` with `coarse_mask=all True` succeeds silently → duplicate active coarse slots.

**(b) Semantic-empty sample not rejected (`:416-417`, `:474-481`):**
- `J > 0` checked, but `joint_mask.any()` per sample is not.
- `joint_mask` all-False with `parent_indices=[[]]` passes length check, `_select_anchors` returns all inactive, mass/loss clamps return finite zero — should fail loud.

## (3) Math correctness: PASS (with scope caveat)

- Feature pooling: soft `S^T X` + mass normalization (`:257-259`).
- Coarse adjacency: hard-argmax binary (intentional, not literal `S^T A S`) via `:297-303` + `graph_utils.py:242-247`.
- MinCut cut + per-sample ortho coherent (`:323-354`).
- DMoN / TopK not present in this file; no method-string dispatch surface to audit.

## (4) R3 surgical: FAIL (cosmetic)

- Allowed roots OK: `src/models/graph_salad/__init__.py`, `pool_dynamic.py`, `tests/test_pool_dynamic.py`.
- Drift: untracked `docs/codex_reviews/m1_2_pool_dynamic*_*.{md,txt}` — review artifacts; safe to declare intentional exclusion.

## (5) pool_deterministic unblock: NO-GO

Shared pooling contract (override-anchor invariants + per-sample empty rejection) must be fail-loud before deterministic twin is written against it. Otherwise pool_deterministic.py inherits the same blind spots.

## Fix list (round 4)

1. **`pool_dynamic.py:520-534`** — for active override anchors, require:
   - nonempty active set,
   - strictly ascending,
   - unique indices,
   - root/ordering invariant if downstream TreeIK assumes root-first.
   Add tests: duplicate / unsorted / root-missing active anchors.

2. **`pool_dynamic.py:428-437`** — after joint_mask validation, reject any batch item with `joint_mask[b].sum() == 0`. Add an all-False `joint_mask` test.

3. **(Optional, cosmetic)** declare `docs/codex_reviews/` as intentional review-artifact carve-out for R3 surgical scope; not a code defect.

## Paths

- Full: `/scratch/ts1v23/workspace/noKslot_clean/docs/codex_reviews/m1_2_pool_dynamic_round3_FULL.txt`
- Brief: this file
