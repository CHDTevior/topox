# M1.2 Step 4 DynamicGraphUnpool Review - Round 1

**VERDICT: NEEDS-FIX**

Core math is correct: `unpool.py:147` implements `h_fine[b,t,j,d] = sum_c P[b,j,c] * h_coarse[b,t,c,d]`, matching the pool contract where dynamic and deterministic pool both return `assignment: P` with shape `[B,J,C]` (`pool_dynamic.py:863-866`, `pool_deterministic.py:503-506`). No unpool-side `1/|cluster|` factor is missing; pooling already divides by cluster/soft mass (`pool_dynamic.py:280-283`, `pool_deterministic.py:201-203`).

## Top 3 Findings

1. **Padded coarse columns can silently leak.** `coarse_mask` is only shape/dtype-checked at `unpool.py:97-101`; it is never used before the einsum at `unpool.py:147`. A valid joint can assign row `[0,1]` while `coarse_mask=[True,False]`, pass row-sum validation, and read a padded coarse feature.

2. **Assignment is not checked for non-negativity.** `unpool.py:128-135` checks row sums only. A valid row `[1.5,-0.5]` sums to 1 and passes, but produces an extrapolation rather than a convex soft-P combination.

3. **Temporal mask semantics are not kernel-aware.** Features are linearly interpolated at `unpool.py:160-162`, but masks are repeated at `unpool.py:166`. With stride 2, `[10,20]` interpolates to `[10,12.5,17.5,20]`; if mask is `[True,False]`, repeat gives `[True,True,False,False]`, marking `12.5` valid even though it depends on the invalid down frame.

## GO / NO-GO for Step 5

**NO-GO until fixes land.** The interface is sufficient for `losses.py` and should not need a breaking signature change, but step 5 should wait for: non-negative P validation, zero mass in `coarse_mask=False` columns, padded-row absolute-mass validation, empty/all-padded mask handling, mask device checks, and tests for soft P, B>1, padded coarse columns, negative P, and frame-mask boundary behavior.
