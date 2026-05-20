# M1.2 pool_dynamic.py — Round 5 codex review BRIEF

- Model: gpt-5.5 xhigh, fresh thread
- threadId: `019e4589-dfc6-7d21-8bef-26e90e65e8ca`
- Date: 2026-05-20

## Verdict
**NEEDS-FIX** — 11th R12 found.

## Q1 — 10 prior R12 categories
All 10 SEALED. Code line + test line per category recorded in FULL. Coarse_mask round-5 guard at `pool_dynamic.py:543-555` and tests at `test_pool_dynamic.py:280-299` are tight. One soft note: mincut orthogonality (#1) has finite-aux check but no numeric oracle.

## Q2 — 11th R12: NEW
**`joint_mask` prefix-active contract NOT enforced — compact/raw index aliasing.**

- Adversarial: `B=1,T=4,J=6,D=4` with `joint_mask=[[T,T,T,F,T,T]]` and compact `parent_indices=[[-1,0,1,2,3]]`. Raw valid graph 0-1-2-4-5, slot 3 is a hole.
- Codex actually executed this and got finite output `anchor_indices=[[0,4,-1,-1]]`. Compact leaf `4` should map to raw joint `5`, but module anchors raw joint `4`. Silent wrong mass.
- Why it slips: `pool_dynamic.py:441` only rejects all-false; `:487` only checks length. `_select_anchors()` returns compact indices, then `_compute_assignment()` + raw gather (`:186`, `:601`) use them as raw indices.
- Fix location: `pool_dynamic.py:441-446`, immediately after all-false check.

## Q3 — R3 surgical compliance
Within the two scoped files: surgical, only coarse_mask guard + 2 tests added. No drift in mincut/assignment/pooling/graph math.
- Caveat: files are untracked, can't mechanically `git diff`; workspace also has unrelated `__init__.py` mod (not part of round 5).

## Q4 — pool_deterministic.py unblock
**NO-GO** — deterministic counterpart will inherit the same compact/raw mask assumption. Fix joint_mask prefix guard first.

## Fix list
1. Add per-sample `joint_mask[b]` contiguous-True-prefix guard in `forward()` (parallel to coarse_mask guard at :543-555), placed at `:441-446` right after the all-false check.
2. Add negative test `test_joint_mask_hole_raises` with `joint_mask=[[T,T,T,F,T,T]]` and compact `parent_indices=[[-1,0,1,2,3]]`, expecting `ValueError`.
3. Re-run full suite (expected 25 tests PASS), then re-review round 6.
