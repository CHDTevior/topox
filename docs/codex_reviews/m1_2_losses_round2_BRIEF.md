# M1.2 Losses Round 2

Verdict: **PASS** with non-blocking test gaps.

M1.3 GraphMotionVAE wiring: **GO**.

Test run: `python tests/test_losses.py -v` -> **17 passed in 0.045s**. `pytest` is not installed in this shell.

Round-1 fixes:
- `masked_kl_gaussian`: **sealed in code**. Masks `mu/logvar` before arithmetic, clamps `logvar` before `exp`, then re-masks KL. Test covers active overflow, not masked padded overflow.
- `masked_bone_length`: **sealed in code**. Skips padded child/parent and `j >= joint_mask.sum()`. Existing padded-child test does not actually list the padded child, so regression coverage is incomplete.
- `masked_vel_consistency`: **sealed in code**. Finite check plus bool shape checks are present before use/T<2 return. Tests only cover the happy path.
- `compute_total_loss`: **sealed in code**. `pool_aux_weights` is passed into `aggregate_pool_aux` per level. Tests cover custom aux weights only at `aggregate_pool_aux`, not through `compute_total_loss`.

Top residual issues:
1. Missing regression tests for padded bone edges, velocity validation, and compute-level pool aux passthrough.
2. `compute_total_loss` disables bone/vel by zero weight only; required tensors are still computed and must be supplied by M1.3.
3. `aggregate_pool_aux({})` is not handled if an empty dict appears inside `pool_aux_outputs`.

M1.3 adapter requirements:
- Split `gt_pos/gt_vel` from `batch.motion_features[..., :3]` and `[..., 3:6]`.
- Produce `pred_pos` via rot -> TopoFKDecoder and provide/derive `pred_vel`.
- Pad `batch.bone_lengths_rest` to `[B,J]`, pass final KL masks, and route pool aux dicts from dynamic/deterministic pool levels.
