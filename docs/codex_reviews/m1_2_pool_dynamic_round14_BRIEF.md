# M1.2 pool_dynamic.py Round 14 BRIEF

- **Model/thread**: gpt-5.5 xhigh, fresh `codex exec` cross-check (`019e45dc-2749-7b10-aa8f-2dc29eab5e0a`)
- **Local verify**: `python tests/test_pool_dynamic.py` -> 56/56 PASS
- **Verdict**: **NEEDS-FIX**
- **GO/NO-GO for `pool_deterministic.py`**: **NO-GO**

- **18/21 code-sealed**.
- **Not sealed**: #18 override-tensor device; #19 Floyd exact equality.
- **Partial**: #21 parent/adjacency consistency catches gross mismatch, but not sub-epsilon topology mismatch.
- Minor test gaps remain: #15 adjacency NaN/Inf branch; #17 `mincut_lambda` non-finite branch.

- **22nd R12: YES**. Parent/adjacency uses `allclose(atol=1e-6)` (`pool_dynamic.py:702`), but Floyd treats any `adjacency > 0` as an edge (`graph_utils.py:128`).
- **Trace**: add `adj[0,0,5]=adj[0,5,0]=5e-7` to line parents, set `geo=floyd_shortest_path(adj,jm)`, then `pool(...)` returns `NO_RAISE geo_0_5=1.0`.

- Override device defer fails: validation and `.item()`/implicit item occur before normalization (`pool_dynamic.py:737`, `:743-744`, `:796-801`; `.to(device)` only `:803-805`); meta override raises RuntimeError, not ValueError.
- Floyd `atol=1e-6` defer fails: finite Floyd distances are exact integer hops from 0/1/+Inf min-add (`graph_utils.py:125-146`); `geo += 5e-7` passes at `pool_dynamic.py:637-645`.

- Direct parent path also lacks `assert_root_first_parent_order`; GraphMotionBatch enforces it (`batch.py:372-394`) and TreeIK requires it (`treeik_decoder.py:66-76`), but `pool_dynamic.py:677-715` does not.

**Required before GO**: exact parent-vs-adjacency equality on valid binary subgraph; override tensor device check/normalize before scalar extraction; exact Floyd value equality; direct parent root-order regression if that path remains supported.
