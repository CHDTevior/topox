# M1.2 DynamicGraphPool Review

**Verdict: NEEDS-FIX.** `python tests/test_pool_dynamic.py` passes 9/9, but the module is not safe to use as the M1.2 contract yet.

**Math bugs:** MinCut orthogonality is wrong for padded dynamic pools. `pool_dynamic.py:332-342` masks invalid coarse rows/cols but still uses `I_C / sqrt(C_max)`. For `C_valid=3, C_max=8`, a perfect orthogonal assignment gets nonzero loss (~0.150). Target scale must use per-sample `C_valid`.

**Silent-failure modes:** public `forward` only checks coarse shapes and mask dtypes (`pool_dynamic.py:388-421`). It does not locally validate dtype/device/finite inputs, parent length vs `joint_mask`, root-first ordering, adjacency/geodesic consistency, or mask coverage. Concrete trap: if `parent_indices` includes a padded joint, `_select_anchors` can mark it as a valid coarse anchor, and valid joints can assign to it silently.

**Plan contract bug:** plan section 6.1 uses `graph_meta` and returns `pool_meta`; current forward requires `parent_indices` and returns no pooled parent metadata. This blocks the planned pool x2 path unless M1.3 invents missing metadata outside the module.

**Q4 raw vs Wk:** raw gathered `skeleton_embeddings` are correct. Do not expose `Wk(anchor)`; Wk is a private assignment-key projection. If needed later, consider mass-pooled raw skeleton embeddings, not projected keys.

**pool_deterministic unblock:** **NO-GO.**

MUST-FIX-NOW:
1. Use per-sample `C_valid` in MinCut ortho.
2. Add R12 input/cross-consistency validation, including parent-vs-mask cardinality.
3. Resolve `graph_meta`/`pool_meta`/pooled-parent contract for pool x2.

DEFERRED:
1. same_chain/body_part bias terms.
2. Entropy as two-sided regularizer vs metric.
3. Explicit odd-`T` temporal-stride validation.
