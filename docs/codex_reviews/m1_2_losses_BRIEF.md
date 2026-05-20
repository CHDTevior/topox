# M1.2 losses.py review, round 1

Verdict: **NEEDS-FIX**

M1.3 unblock: **NO-GO** for drop-in training-loop use.

Test result: bare `conda activate graph_salad` failed in this non-interactive shell with `CondaError`; the same env via conda profile hook passed, `15 passed` in ~5-6s, not 0.02s.

Fresh review used local inspection plus independent `codex exec --model gpt-5.5 --config model_reasoning_effort=xhigh`.

`vae.py` does **not** exist yet.

`masked_l1_pos` / `masked_l1_vel`: divisor is valid joint/frame pairs, not xyz elements. Correct.

`masked_kl_gaussian`: sign is correct, D is summed before valid-node mean. Formula correct.

`masked_vel_consistency`: uses `(pos[t+1] - pos[t]) * fps` and masks both t and t+1. Formula correct.

`masked_bone_length`: child-parent edge formula and root exclusion are correct only if `parent_indices` is already a compact valid tree.

1. **KL overflow / padded KL poison**: losses.py line 145 calls `logvar.exp()` after only a finite-input check. Large finite logvar returns Inf; masked-out large logvar can produce `inf * 0 -> nan`.

2. **Bone loss ignores `joint_mask`**: losses.py lines 181-196 trust `parent_indices` and never check child/parent validity against the mask. A padded child can contribute loss if present in the parent list.

3. **R12 contracts are too soft for M1.3**: `masked_vel_consistency` lacks pred_pos/pred_vel finite checks and returns zero for T<2 before validating tensors; mask shape/dtype/device checks are missing across functions.

KL default `1e-3` is fine as a conservative start; warmup is only an implicit caller override.

Pool aux defaults are not cleanly aligned with plan §13 / PLAN_GAP_REPORT §1.2 + §5: DynamicGraphPool already applies `mincut_lambda=0.5`, then `compute_total_loss` applies `pool_aux=0.5`, making effective default MinCut scale 0.25; entropy is default-off and per-key aux weights are not exposed through `compute_total_loss`.

`compute_total_loss` requires `pred_vel`, `gt_vel`, final `coarse_mask`, final `frame_mask_lat`, `[B,J] rest_bone_lengths`, and `pool_aux_outputs`.

Current `GraphMotionBatch` exposes `bone_lengths [B,T,J]` and `bone_lengths_rest` as lists, not the required `[B,J]` tensor.

The locked rot -> FK path naturally yields positions, not necessarily a velocity head.

Add shared shape/dtype/device/mask validators.

Fix KL overflow by clamping/failing loud and computing only valid entries before exp.

Make bone loss either enforce compact valid `parent_indices` or explicitly skip invalid child/parent pairs via `joint_mask`.

Validate velocity tensors before T<2 early return.

Expose/clarify pool aux key weights and effective MinCut scale.

Add tests for KL overflow, padded KL, padded bone, velocity NaN, empty masks, empty aux dict, wrong masks, and one GraphMotionBatch-shaped integration case.
