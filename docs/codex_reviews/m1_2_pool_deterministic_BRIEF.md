VERDICT: NEEDS-FIX

Round-1 review of `src/models/graph_salad/pool_deterministic.py` against sealed `pool_dynamic.py` oracle. Hard assignment and zero-param contract are mostly clean, but deterministic is not yet a drop-in ablation pool.

## Findings

- BLOCKER: padded geodesic validation is stricter than DynamicGraphPool and violates "padding nodes ignored". Deterministic rejects any finite `geodesic_dist > J-1` globally (`src/models/graph_salad/pool_deterministic.py:345-349`); dynamic only compares Floyd values under `both_valid` (`src/models/graph_salad/pool_dynamic.py:629-640`). Repro: dynamic accepts padded finite sentinel 100; deterministic raises.
- BLOCKER: `aux_losses` nested schema is missing oracle keys. Dynamic returns `mincut_cut`, `mincut_ortho`, `mincut`, `locality`, `entropy` (`src/models/graph_salad/pool_dynamic.py:404-410`); deterministic returns only `mincut`, `locality`, `entropy` (`src/models/graph_salad/pool_deterministic.py:262-266`).
- MAJOR: zero aux constants omit dtype (`src/models/graph_salad/pool_deterministic.py:260,263`). Under `torch.set_default_dtype(torch.float64)`, deterministic returns float64 `mincut`/`entropy` despite float32 inputs.
- MINOR: tests pass but miss the parity gaps. Add schema parity, padded finite geodesic, aux dtype, and padded row/slot zeroing tests (`tests/test_pool_deterministic.py:102-110`, `tests/test_pool_deterministic.py:122`).

## Verified OK

- Nearest-anchor one-hot math: gathers `geodesic[j, anchor[c]]`, masks by radius/coarse/joint validity, raises on no candidate, then argmins (`src/models/graph_salad/pool_deterministic.py:152-182`).
- Tie-break is lowest slot/anchor index: `torch.argmin` returned first-min in verification; rule anchors are sorted (`src/models/graph_salad/graph_utils.py:394-398,457`) and override anchors are strictly ascending/root-first (`src/models/graph_salad/pool_deterministic.py:455-465`).
- No learnable params: `super().__init__()` is called and only scalar hparams are stored (`src/models/graph_salad/pool_deterministic.py:76,98-103`); verified `param_count=0`, `state_dict_keys=[]`, no children.
- Top-level output keys match dynamic (`src/models/graph_salad/pool_deterministic.py:493-504` vs `src/models/graph_salad/pool_dynamic.py:863-874`).
- `python tests/test_pool_deterministic.py`: 14/14 OK. `pytest` unavailable (`No module named pytest`).

## Step 4 Decision

UNBLOCK_STEP_4: NO-GO. Fix geodesic padding parity and aux-loss schema/dtype first; otherwise unpool/loss integration will not be testing the same contract as dynamic.

## Fix List

- Remove deterministic global `finite_geo > J-1`, or scope it to `both_valid` only.
- Return `mincut_cut`, `mincut_ortho`, `mincut`, `locality`, `entropy` with zero constants using `dtype=P.dtype`.
- Add tests for schema parity, padded finite geodesic acceptance, aux dtype, and padded joint/slot zeroing.
- Optional: use `+inf` non-candidate masking after no-candidate rows are rejected instead of fixed `1e6/2e6` sentinel.
