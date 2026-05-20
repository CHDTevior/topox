Verdict: **NEEDS-FIX**.

State-dict isomorphism passes: new `GraphAttentionBlock(...).state_dict()` matches `encoder.py::GraphAttentionBlock` key-for-key and shape-for-shape, including edge-bias weights, LayerNorms, and FFN indices. The 20 local tests pass under `python tests/test_graph_attention.py`.

9th R12 mode: finite but impossible geodesic magnitudes are accepted. Example: `geo[0,0,1]=geo[0,1,0]=1e20` produces finite output instead of failing, even though documented Floyd hop-count distances should be `<= valid_node_count-1` or `+Inf`.

More importantly, two existing seals are leaky:
- **MUST-FIX-NOW:** `torch.allclose(..., atol=1e-6)` leaves default `rtol=1e-5` active. Large absolute adjacency/geo asymmetries can pass.
- **MUST-FIX-NOW:** geodesic symmetry masks to `both_finite`, so `+Inf` one way and finite the other way passes.
- **MUST-FIX-NOW:** add regression tests for both traces.
- **MUST-FIX-NOW:** align adjacency doc/code contract: doc says binary, code allows weighted non-negative.
- **DEFERRED/decision:** enforce finite valid-valid geodesic upper bound if `geo` is strictly Floyd hop-count.

Defense-in-depth: **yes, justified**. `GraphAttentionBlock` is public/exported and plausible outside `GraphMotionBatch`; O(BN²) checks are acceptable relative to attention cost.

`pool_dynamic.py` unblock: **NO-GO** until the symmetry leaks are fixed. After that, the block is a stable dependency.
