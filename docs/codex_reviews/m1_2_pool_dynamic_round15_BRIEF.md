# Verdict: NEEDS-FIX-with-CONCRETE-trace

Q1. The prior 22 represented R12 fail-loud categories look sealed in current repo truth. Spot checks: FK parent ordering is enforced at `pool_dynamic.py:678-701` via `graph_utils.py:256-281`; exact parent-adjacency match uses `torch.equal` at `pool_dynamic.py:702-720` and Floyd consistency at `624-646`; mask/anchor-source guards cover all-false/prefix `joint_mask` at `595-623` and XOR override contract at `656-677`.

Q2. A concrete 23rd R12 survives. Override mode accepts inactive coarse slots containing valid non-`-1` anchor ids and returns them unchanged, despite the output contract `anchor_indices: ... -1 for padded` at `pool_dynamic.py:39`. Runnable trace: `anchor_indices=[[0,2,5,0]]`, `coarse_mask=[[True,True,True,False]]` on a 6-joint line graph succeeds and returns `[[0,2,5,0]]` with `pooled_mask=[[True,True,True,False]]`. Slot 3 is padded but not `-1`: silent metadata contract violation.

Q3. `pool_deterministic.py` sibling module unblock: NO-GO.

Tests: `python tests/test_pool_dynamic.py` ran 58 tests OK. `pytest` was unavailable (`No module named pytest`). Independent `codex exec` session: `019e45ec-d2ef-7e20-99b3-02d231aec097`.

Fix list:
- In override path, require every `coarse_mask=False` slot to have `anchor_indices == -1`, or explicitly redefine the contract.
- Add a regression test for inactive valid-id sentinel drift.
