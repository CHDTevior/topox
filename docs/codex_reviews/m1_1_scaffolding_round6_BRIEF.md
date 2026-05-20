# M1.1 Scaffolding — Codex Review Round 6 BRIEF

- **Model**: gpt-5.5, xhigh reasoning
- **Thread**: `019e4385-faea-7cb2-9b8c-7c91b9db6218` (FRESH, no continuation)
- **Date**: 2026-05-20
- **Files reviewed**: batch.py / graph_utils.py / denoiser_stub.py / __init__.py / tests/test_scaffolding.py
- **Verdict**: **NEEDS-FIX**

## Headline

Codex caught **R12 category #8** — parent ordering / FK topology invariant.
Plus Q5 surgical spillage outside `src/models/graph_salad/` + `tests/test_scaffolding.py`.

## Answers

| Q | Answer |
|---|---|
| Q1 8th category | **FOUND**: parent ordering / FK topology — root must be index 0 + parent-before-child order |
| Q2 convergent | **OPEN** (specific remaining R12, not fishing) |
| Q3 geodesic defer | **AGREE** (defer to M1.x; not load-bearing for M1.2) |
| Q4 padded-True defense | **AGREE-KEEP** (cheap, future-proofs reordering) |
| Q5 surgical scope | **FAIL** — `tests/test_graph_utils.py` modified + `outside_docs/animo_metric_borrowing_plan_for_graph_salad.md` untracked |
| Q6 M1.2 unblock | **NO-GO** until parent ordering sealed |
| Q7 denoiser stub | **PASS** |

## R12 category landscape (now 8, was 7)

| # | Category | Sealed in round |
|---|----------|----------------|
| 1 | tensor schema | R1 |
| 2 | cross-tensor T/J | R1 |
| 3 | scalar value-range + mask sum | R3 |
| 4 | NaN/Inf scalars+tensors | R2/R3 |
| 5 | bool-as-int | R3 |
| 6 | list cardinality + tree-validity | R4 |
| 7 | graph semantics + mask prefix | R5 |
| **8** | **parent ordering / FK topology** | **R6 NEEDS-FIX** |

## Concrete failure case codex surfaced

```python
parent_indices = [2, 2, -1]   # root at index 2 (not 0); parents-before-children violated
# Currently passes GraphMotionBatch.from_collate_dict, but treeik_decoder.py:66 requires root=0
```

## Fix list

- **MUST-FIX-NOW (R12 #8)**: `batch.py:374` — after `validate_parent_tree(inner)`, add `assert_root_first_parent_order(inner)` (or inline equivalent). Tests for `[2,2,-1]` (root not 0) and `[-1,2,0]` (parent-after-child).
- **MUST-FIX-NOW (Q5 R3)**: clean or scope-document `tests/test_graph_utils.py` mods + `outside_docs/animo_metric_borrowing_plan_for_graph_salad.md` before claiming surgical R3 compliance.
- **DEFERRED-TO-M1.x**: exact `geodesic_dist` vs adjacency cross-check (tautology + O(B·J³)).

## Verification (codex-side)

- `tests/test_scaffolding.py -v` → 46 PASS
- `tests/test_graph_utils.py -v` → 39 PASS

## Artifacts

- Full: `/scratch/ts1v23/workspace/noKslot_clean/docs/codex_reviews/m1_1_scaffolding_round6_FULL.txt`
- Brief: this file
