# M1.1 Graph-SALAD Scaffolding — Codex Review Round 4 (Brief)

**Date:** 2026-05-20
**Reviewer:** Codex MCP, gpt-5.5, xhigh reasoning, **fresh thread**
**Thread ID:** `019e4374-2f42-7513-9d29-b8bbdce3d1ee`
**Continuation of round 1/2/3?** No — fresh thread per user 2026-05-20 rule.

## Verdict

**NEEDS-FIX** — NO-GO for M1.2.

## Summary

5 of 5 R12 silent-failure categories from rounds 1-3 are sealed. However, a
**new sixth category** — per-sample list **cardinality** (length consistency
with `num_joints[b]`) — surfaced with fresh eyes and is **NOT** sealed. The
prior 3 rounds focused on tensor / scalar guards; the per-sample list block at
`batch.py:323-353` only validates outer length (`== B`) and inner element type,
never `len(inner) == num_joints[b]`.

Validation order, M1.2 forward compat (for tensor fields), tests adequacy, and
denoiser stub all PASS in isolation — but the cardinality gap blocks M1.2 graph
pool because pool consumes `batch.parent_indices` per-joint.

## Findings by question

| Q | Topic                          | Result   | Notes                                                           |
|---|--------------------------------|----------|-----------------------------------------------------------------|
| 1a| Schema drift                   | NOT-SEALED | `len(parent_indices[b]) == num_joints[b]` not enforced       |
| 1b| Cross-tensor T_max/J_max       | SEALED   | batch.py:243-265                                                |
| 1c| Scalar range + mask consistency| SEALED   | batch.py:271-297                                                |
| 1d| NaN/Inf in tensors + scalars   | SEALED   | batch.py:203, 301-309                                           |
| 1e| bool-as-int in lists           | SEALED   | batch.py:340-353                                                |
| 2 | Validation order monotonic     | PASS     | structural-before-value; locally diagnosable error messages     |
| 3 | R2 simplicity vs R12 fail-loud | JUSTIFIED| no redundant guards; missing fix should add a guard, not remove |
| 4 | M1.2 forward compatibility     | **NO-GO**| per-joint list cardinality not enforced; pool can read padding  |
| 5 | Tests adequacy                 | **GAP**  | synthetic helper uses full-J lists even when nj=J-2; blind spot |
| 6 | Denoiser stub                  | PASS     | unchanged since round 1; no contract refinement forces change   |

## Independent evidence (CC verification)

Codex's finding was verified directly against the source files:

1. **`batch.py:323-353`** — per-sample list loop validates outer length and
   inner element type only; no `len(inner) == num_joints[i]` check.

2. **`tests/test_scaffolding.py:30-66`** — synthetic helper has
   `nj = [J, J-2, ...]` but all per-sample lists are length J for every
   sample. So sample 1 has `num_joints=22` but `parent_indices` of length 24.
   `GraphMotionBatch.from_collate_dict` accepts this silently.

3. **`unified_dataset.py:271-291`** — upstream emits lists at length
   `J = num_joints[b]` (per-sample, NOT J_max). Variable per-sample list
   lengths are real in real data. Comment at `unified_dataset.py:321`
   explicitly says "list[int] length J".

Conclusion: the gap is materially load-bearing for M1.2, not academic.

## Fix list

1. **R12, batch.py:323** — after inner type/bool validation, add
   `len(inner) == int(num_joints[i])` per per-sample-list key. For
   `parent_indices`, additionally enforce exactly one `-1` root and all
   non-root parent ids in `[0, num_joints[i])` (or call existing tree
   validator from `graph_utils.py`).

2. **R12, tests/test_scaffolding.py:63** — make `_make_synthetic_collate_dict`
   generate per-sample list lengths from `nj[i]` (not full `J`). Add negative
   regression tests for:
   - parent length mismatch (`len != num_joints`)
   - parent index out of range (`>= num_joints` or `< -1`)
   - missing root (no `-1` in parent_indices)
   - multiple roots (more than one `-1`)
   Also strengthen real-dataset smoke test to assert
   `len(parent_indices[b]) == num_joints[b]` per sample.

## Bottom line

**NO-GO for M1.2** until per-sample list cardinality (and parent-index range)
is fail-loud at `GraphMotionBatch.from_collate_dict` construction time.

## Trail of round verdicts

| Round | Result    | Categories found                                           |
|-------|-----------|------------------------------------------------------------|
| 1     | NEEDS-FIX | annotation looseness, dtype, last-dim, square check        |
| 2     | NEEDS-FIX | scalar rank, B<=0, mixed device, NaN/Inf, inner list types |
| 3     | NEEDS-FIX | num_joints/frames range+mask.sum, fps finite+positive, bool-as-int |
| 4     | NEEDS-FIX | per-sample list cardinality vs num_joints[b]               |
