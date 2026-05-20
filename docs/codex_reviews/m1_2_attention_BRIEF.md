# M1.2 Attention Review Brief

The attention algebra is correct: Q/K/V reshape to `[B,H,N,d_head]`, scores use scaled `QK^T`, topology bias lands on `[B,H,N,N]`, key-side mask broadcasts correctly, and output projection/FFN match `encoder.py::GraphAttentionBlock`. A state_dict probe confirmed exact same block-level key order and shapes as legacy encoder.

Blocker is R12, not math. `forward()` silently accepts bad numeric inputs: NaN in padded `x` can contaminate valid rows; `geodesic_dist` NaN is collapsed to `0.0`; adjacency NaN can be hidden into finite output; all-false `node_mask` returns finite nonzero output; and `dropout=1.0` is accepted despite a `[0,1)` contract. Also, attention.py:141-146 says "large finite sentinel" but actually substitutes `0.0`.

SPD bucketing should wait. Scalar geodesic bias is weaker than Graphormer, but it is ckpt-compatible and enough for M1.2 plumbing. Do not claim it is true Graphormer SPD bucketing until bucket embeddings exist.

Query-side masking is not required under the stated downstream-zero contract. The real fix is fail-loud validation, not relying on `nan_to_num`.

VERDICT: NEEDS-FIX
Q2_SPD_BUCKETING: LATER
Q3_MASK_QUERY_SIDE: OK_AS_IS
Q4_SILENT_FAILURES: nonfinite_x, geodesic_nan_or_minus_inf, adjacency_nan, all_false_node_mask, dropout_p_eq_1
NEXT_MODULE_UNBLOCK: NO-GO
FIX_LIST: MUST-FIX-NOW: add fail-loud finite/sentinel/mask/dropout validation, fix the false geodesic sentinel comment, add R12 regression tests; DEFERRED: SPD bucket/unreachable bucket, state_dict/backward/eval/single-head/mixed-batch tests
