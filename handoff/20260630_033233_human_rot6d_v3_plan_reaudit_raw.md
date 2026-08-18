PASS.

The handoff resolves the prior plan blockers:

Gate C is now explicitly FK position-invariance, not token-invariance. Gates F-I cover sibling-shared storage, telescoping/reindex round-trip, root recovery unaffected, and cond offset/cache hygiene. v3b now specifies deterministic continuity: v3a direction-anchored single-child handling, spine3-only continuity, sign alignment, ordering continuity, post-choice det correction, and hysteresis.

The actual split is also confirmed from `PARENTS`: single-child parents are `[1,2,3,4,5,6,7,8,12,13,14,16,17,18,19]`; multi-child parents are `[0,9]`; leaves are `[10,11,15,20,21]`.

No remaining showstopper in the plan.