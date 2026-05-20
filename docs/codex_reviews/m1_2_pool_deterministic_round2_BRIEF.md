VERDICT: PASS

UNBLOCK_STEP_4: GO for `unpool.py`.

## Round-1 Closure

| Item | Sealed? | Evidence |
|---|---:|---|
| Hop-bound scoped to `both_valid_jj` | Y | `pool_deterministic.py:354-359`; Floyd parity `pool_deterministic.py:423-433` vs `pool_dynamic.py:629-640` |
| `aux_losses` 5-key parity | Y | deterministic `mincut_cut/mincut_ortho/mincut/locality/entropy` at `pool_deterministic.py:264-272`; dynamic oracle at `pool_dynamic.py:404-410` |
| Zero aux dtype uses `P.dtype` | Y | `pool_deterministic.py:261-262`, used at `267-271` |
| 3 new tests added | Y | schema `tests/test_pool_deterministic.py:114-127`; dtype `129-136`; padded geodesic `138-163` |

## New Findings

None. No new R12 fail-loud issue, no R7 surface conflict, no silent fallback,
phantom check, wrong-dtype leak, or mis-scoped padded-geodesic assert found.

Note: tests were not executed per instruction; user reports 17/17 PASS.

## Step 4

GO. Deterministic pool is now contract-compatible with the sealed dynamic pool
for the reviewed downstream surfaces.
