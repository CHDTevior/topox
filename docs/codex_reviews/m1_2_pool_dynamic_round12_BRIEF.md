# M1.2 Pool Dynamic Round 12 Brief

**VERDICT: NEEDS-FIX**

**Convergence claim: "19 R12s sealed"**

Not confirmed. Most named items are sealed, but:

1. **#19 Floyd consistency is not exact as claimed.** The source recomputes Floyd at `src/models/graph_salad/pool_dynamic.py:594-616`, but value comparison uses `atol=1e-6, rtol=0.0` at `pool_dynamic.py:609-612`. A `+5e-7` symmetric geodesic perturbation is accepted. The regression at `tests/test_pool_dynamic.py:313-320` only catches a gross all-zero mismatch.
2. **#18 device consistency is only partially sealed.** Core input tensors and module device are checked at `pool_dynamic.py:443-463`, with tests at `tests/test_pool_dynamic.py:279-297`. Override `anchor_indices/coarse_mask` are still validated before device normalization at `pool_dynamic.py:671-759`; meta override tensors RuntimeError instead of fail-loud ValueError.

**20th R12 status: FOUND**

Padded adjacency rows/cols are accepted and corrupt mincut. `pool_dynamic.py:488-516` does not reject non-zero adjacency outside `joint_mask`; `_compute_aux_losses` uses unmasked `degree = adjacency.sum(dim=-1)` at `pool_dynamic.py:332`. Probe: adding only `adj[0,0,7]=adj[0,7,0]=1` with joint 7 padded leaves assignment and pooled adjacency unchanged, but changes `mincut_cut` from `-0.9097329` to `-0.8132305`.

**GO/NO-GO for M1.3 pool_deterministic.py: NO-GO**

Fix list:

1. Make Floyd value consistency exact (`atol=0, rtol=0` or masked `torch.equal`) and add sub-1e-6 plus finite-pattern mismatch tests.
2. Reject or mask padded adjacency rows/cols inside `DynamicGraphPool`; add polluted-padded-edge regression.
3. Finish override tensor device handling before validation; add mismatch regression.
4. Add module dtype guard/regression for `pool.double()` or other non-float32 module params.
5. Reject non-finite scalar hparams (`temperature`, `locality_alpha`, `mincut_lambda`, etc.) and add tests.
