# M1.2 attention.py ROUND 3 BRIEF

VERDICT: NEEDS-FIX

Local test signal: `pytest` is unavailable in this shell (`command not found`, and `python -m pytest` has no pytest module). The file's own unittest runner passes 14/14 via `python tests/test_graph_attention.py`.

The named numeric/mask R12 cases are sealed: nonfinite `x`, geodesic NaN/-Inf, adjacency NaN/Inf, all-False per-batch `node_mask`, dropout out of range, and B/N=0 all raise explicit errors.

New silent failure: topology semantics are not validated. `attention.py` documents `adjacency` as binary `{0,1}` and symmetric, but non-binary 0.5, negative -1, and asymmetric adjacency all return finite outputs. `geodesic_dist` is documented as Floyd hop-count/+Inf, but negative finite distances, nonzero diagonal, and asymmetric distances also return finite outputs. These invalid tensors become learned additive bias, so graph bugs can stay wrong-but-finite. Also, the `nan_to_num` comment still claims uniform all-masked rows are forced to zero; that is not what the code does.

R3 scope claims:
- `__init__.py` export: AGREE. It is package-local wiring for the new block, not scope creep.
- Untracked review artifacts: AGREE. They are docs/review outputs, not runtime source; make an intentional commit/remove decision at M1.2 finalization.

pool_dynamic.py: NO-GO. The attention block must first settle and enforce its topology tensor contract.

Fix list:
1. Enforce or rewrite the adjacency contract: binary/symmetric, or explicitly weighted finite/nonnegative/symmetric if pool_dynamic needs weighted pooled adjacency.
2. Add geodesic semantic validation/tests for negative finite values, nonzero valid-node diagonal, and asymmetry, or document a broader intentional contract.
3. Fix the stale all-masked `nan_to_num` comment.
