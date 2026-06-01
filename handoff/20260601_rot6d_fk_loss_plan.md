# anytop13 World + True Rot6D-FK Loss Plan

Date: 2026-06-01
Status: design document only. No training-code changes in this document.
Supersedes: `handoff/20260530_2243_prism_fk_loss_experiment_plan.md` and the older `anytop13_prism_fk` wording.

## 0. Decision

Implement a combined geometry supervision branch for the VAE:

1. keep the accepted world/RIC geometry supervision, so the current pose/render
   route stays aligned with GT;
2. add true FK supervision using the existing AnyTop 13-channel rotation field:

```text
motion[..., 3:9] = per-joint 6D rotation
```

The new branch should be named by what it is:

```text
loss_mode = "anytop13_world_rot6d_fk"
```

It does not replace the base anytop13 channel loss. It also does not delete
`anytop13_world_geometry`; that mode remains available as the world-only
ablation. The combined objective is:

```text
L_total =
  L_anytop13_base
+ w_world * L_world_ric
+ w_fk    * L_rot6d_fk
+ w_traj  * L_root_traj
```

No pool, decoder, dataset, max-frame, or model-width changes in this experiment.
The intended base config remains the current clean-L2 VAE setup:

```text
data_root      = data/anytop_planet_zoo_clean_L2
feat_mode      = anytop13
pool_type      = edge_segment
decoder_mode   = coarse_xattn
max_coarse     = 128
max_joints     = 144
max_frames     = 64
d_model        = 512
n_heads        = 8
d_ff           = 1536
temporal_stride= 4
use_name_embed = true
```

## 1. Why Combine world_geometry And true FK

Current `anytop13_world_geometry` does this:

```text
recover_world_positions_torch(pred_raw)
vs
recover_world_positions_torch(gt_raw)
```

That route is AnyTop RIC/world-position recovery. It uses:

```text
root ch 3:9   root 6D rotation
root ch 9/11  root x/z velocity
root ch 1     root height
non-root ch0:3 root-relative position
```

It does not use non-root `[..., 3:9]` rotations. We verified that non-root
rotation gradient is zero for that loss. But it is still important because it
directly supervises the route our current visual QA/demo renders:

```text
P_pred_ric ~= P_gt_ric
```

The new FK part must additionally use the rot6d-FK route:

```text
pred_raw[..., 3:9]
+ parent_indices
+ rest_offsets
-> recover_from_bvh_rot-style FK
-> predicted 3D joint positions
```

This makes proximal rotation errors affect distal tail/wing/limb positions
through the kinematic chain:

```text
P_pred_fk ~= P_gt_ric
```

Together, the two losses pull both output semantics toward the same GT world
skeleton:

```text
0:3 pose/RIC route  -> GT pose/world skeleton
3:9 rot6d/FK route  -> GT pose/world skeleton
```

This is better than FK-only for our current pipeline, because FK-only could
improve the rotation channels while leaving the current `0:3` render route
visually unchanged.

## 2. Target Definition

Use the user's Route-A / Route-B logic:

```text
Route A: pose_from_pos   = recover_from_bvh_ric_np(data)
Route B: pose_from_rot6d = recover_from_bvh_rot_np(data, parents, offsets)
```

For training, both geometry branches should target Route A from the ground
truth, because that is the currently trusted visual/world-space target:

```text
P_gt_ric = recover_world_positions_torch(gt_raw)
```

World/RIC branch:

```text
P_pred_ric = recover_world_positions_torch(pred_raw)
L_world_ric =
  mean_valid_joints_frames( || P_pred_ric - P_gt_ric ||_1 )
```

FK branch:

```text
P_pred_fk = recover_rot6d_fk_positions_torch(
    pred_raw,
    parent_indices=batch.parent_indices,
    rest_offsets=batch.rest_offsets,
    joint_mask=batch.joint_mask,
)
L_rot6d_fk =
  mean_valid_joints_frames( || P_pred_fk - P_gt_ric ||_1 )
```

Root trajectory term:

```text
L_root_traj =
  mean_valid_frames( || P_pred_ric[:, :, 0, :] - P_gt_ric[:, :, 0, :] ||_1 )
```

Do not add a separate `fk_traj` in v1. The FK and RIC routes share the same root
channels; adding both would over-penalize root drift.

Important: if `pred_raw == gt_raw`, `L_rot6d_fk` is not guaranteed to be exactly
zero. It equals the dataset's own Route-B-vs-Route-A mismatch. That is expected.
The preflight scan must measure this mismatch before training.

## 3. Why The Target Is RIC(gt), Not FK(gt)

There are two possible targets:

```text
Option 1: FK(pred_rot6d) vs FK(gt_rot6d)
Option 2: FK(pred_rot6d) vs RIC(gt_position)
```

Use Option 2 for v1.

Reason:

- The visual QA path and current rendered skeleton target are RIC/world positions.
- The user's diagnostic script explicitly measures `recover_from_bvh_rot_np`
  against `recover_from_bvh_ric_np`.
- If GT rot6d and GT position channels disagree because of conversion,
  helper bones, cleaning, or parent reindexing, training against `FK(gt_rot6d)`
  would preserve the disagreement instead of correcting it.

Still log this diagnostic:

```text
L_gt_route_mismatch =
  mean_valid( || FK(gt_raw[...,3:9]) - RIC(gt_raw) ||_1 )
```

Do not include it in `total`; use it to judge data consistency and to interpret
the loss floor.

## 4. Torch FK Recovery Module

Add a new module:

```text
src/models/graph_salad/rot6d_fk_recovery.py
```

Main API:

```python
def recover_rot6d_fk_positions_torch(
    motion_13ch: torch.Tensor,          # [B,T,J,13], RAW denormalized
    parent_indices: list[list[int]],    # length B
    rest_offsets: torch.Tensor,         # [B,J,3]
    joint_mask: torch.Tensor,           # [B,J]
) -> torch.Tensor:                      # [B,T,J,3]
    ...
```

This should be a torch port of the official AnyTop/SALAD
`recover_from_bvh_rot_np` logic, not a generic `fk_persample(local_rot6d)` call.

The expected logic is:

```text
1. root_R, root_pos = recover root rotation and root trajectory from root channels
   - root_R from root ch3:9
   - root x/z from ch9/ch11 cumsum, rotated by inverse root rotation
   - root y from ch1

2. nonroot_R = rot6d_to_matrix(motion[..., 1:, 3:9])

3. all_R = concat(root_R, nonroot_R)  # [B,T,J,3,3]

4. parent reindex, matching recover_from_bvh_rot_np:
   local_R starts as identity.
   For each non-root joint j:
       p = parents[j]
       local_R[p] = all_R[j]

5. root correction:
       local_R[0] = root_R^T @ local_R[0]

6. local translation:
       local_pos[j] = rest_offsets[j]
       local_pos[0] = root_pos

7. FK chain:
       G[0] = T(local_R[0], local_pos[0])
       G[j] = G[parent[j]] @ T(local_R[j], local_pos[j])
       P[j] = G[j].translation
```

Use matrix composition directly. Avoid quaternion sign branches in the training
path; the matrix version is equivalent for FK and is smoother for autograd.

It is okay to loop over batch and joints. J is at most 144 and this is a VAE
loss, not diffusion inner-loop sampling.

## 5. Loss Function Wiring

Append a new standalone function in `src/models/graph_salad/losses.py`:

```python
def compute_world_rot6d_fk_terms(
    *,
    pred_motion: torch.Tensor,      # [B,T,J,13], normalized
    gt_motion: torch.Tensor,        # [B,T,J,13], normalized
    anytop_mean: torch.Tensor,      # [B,J,13]
    anytop_std: torch.Tensor,       # [B,J,13]
    parent_indices: list[list[int]],
    rest_offsets: torch.Tensor,     # [B,J,3]
    joint_mask: torch.Tensor,       # [B,J]
    frame_mask: torch.Tensor,       # [B,T], use frame_mask_recovered
) -> dict[str, torch.Tensor]:
    ...
```

Implementation:

```text
pred_raw = _denorm_13ch(pred_motion, anytop_mean, anytop_std)
gt_raw   = _denorm_13ch(gt_motion,   anytop_mean, anytop_std)

P_pred_fk = recover_rot6d_fk_positions_torch(
    pred_raw, parent_indices, rest_offsets, joint_mask
)
P_pred_ric = recover_world_positions_torch(pred_raw)
P_gt_ric   = recover_world_positions_torch(gt_raw)

world = masked_l1_xyz(P_pred_ric, P_gt_ric, joint_mask, frame_mask)
fk    = masked_l1_xyz(P_pred_fk,  P_gt_ric, joint_mask, frame_mask)
traj  = masked_l1_xyz(P_pred_ric[:, :, 0], P_gt_ric[:, :, 0], frame_mask)
gt_fk_mismatch = masked_l1_xyz(
    recover_rot6d_fk_positions_torch(gt_raw, parent_indices, rest_offsets, joint_mask),
    P_gt_ric,
    joint_mask,
    frame_mask,
)

return {"world": world, "fk": fk, "traj": traj, "gt_fk_mismatch": gt_fk_mismatch}
```

Do not modify `compute_total_loss_13ch`. Keep the default path byte-identical.

In `scripts/train_graph_vae.py::run_loss`:

```text
if loss_mode == "anytop13_world_rot6d_fk":
    terms = compute_world_rot6d_fk_terms(...)
    losses["world"] = terms["world"]
    losses["fk"] = terms["fk"]
    losses["traj"] = terms["traj"]
    losses["gt_fk_mismatch"] = terms["gt_fk_mismatch"]  # diagnostic only
    losses["total"] = (
        losses["total"]
        + w_world * terms["world"]
        + w_fk    * terms["fk"]
        + w_traj  * terms["traj"]
    )
```

CLI additions:

```text
--loss_mode choices += anytop13_world_rot6d_fk
--w_fk       default 0.25
```

Reuse existing `--w_world` and `--w_traj` for the world/RIC and root trajectory
terms. Do not introduce `--w_fk_traj` in v1.

Export `compute_world_rot6d_fk_terms` from `src/models/graph_salad/__init__.py`.

## 6. Weight Plan

Start with equal nominal weights for the two geometry branches:

```text
w_world = 0.25
w_fk    = 0.25
w_traj  = 0.10
```

Reason:

- We want the pose/RIC route and rot6d/FK route to be equally important.
- Equal numeric weights are only a starting point. The real criterion is equal
  weighted contribution after measuring raw scale.
- The accepted `world_geometry` branch used `w_world=0.5, w_traj=0.25`; combined
  world+FK should start lower so the geometry terms do not immediately overpower
  the base normalized channel losses.

Keep all base anytop13 weights unchanged:

```text
w_pos     = 1.0
w_rot     = 1.0
w_vel     = 1.0
w_contact = 0.1
w_kl      = 1e-3
w_pool_aux= 0.5
```

Calibration rule before the full run:

```text
Run a smoke/calibration batch and log:
  base_total
  pos / rot / vel / contact / kl / pool_aux
  world
  fk
  traj
  w_world * world
  w_fk * fk
  w_traj * traj
```

Then:

```text
if w_world*world and w_fk*fk are both < 10% of base_total:
    also run a stronger arm w_world=0.5, w_fk=0.5, w_traj=0.25

if either weighted geometry term is > 60% of base_total at epoch 0:
    reduce both geometry weights to w_world=0.10, w_fk=0.10, w_traj=0.05

if one raw term is much larger than the other:
    adjust weights so w_world*world ~= w_fk*fk, not necessarily w_world == w_fk

otherwise:
    keep w_world=0.25, w_fk=0.25, w_traj=0.10
```

Recommended experiment arms:

```text
A: existing anytop13 baseline
   runs/_baseline_cleanL2_ep34_for_p1diag_compare/best_recon_model.pt

W: accepted world_geometry run/checkpoint
   loss_mode=anytop13_world_geometry, w_world=0.5, w_traj=0.25

B: new combined world+FK run
   loss_mode=anytop13_world_rot6d_fk, w_world=0.25, w_fk=0.25, w_traj=0.10

C: optional stronger combined world+FK run if resources allow or calibration says weak
   loss_mode=anytop13_world_rot6d_fk, w_world=0.5, w_fk=0.5, w_traj=0.25
```

If 6 same-architecture A100s are available, run B and C together. Do not rerun A
unless we need same-code-version strictness.

## 7. Data Preflight

Before training, adapt the user's scanner to the local L2 path:

```text
data/anytop_planet_zoo_clean_L2
```

For train and val samples, compute:

```text
RIC(gt) vs FK(gt_rot6d)
mean / median / p95 / p99 / max
percent of motion bbox diagonal
main_nonhelper stats
root stats
```

Use the same helper-name rule from the user's script:

```text
helper if joint name contains:
  end_site, twist, srb, breath
```

Interpretation:

```text
p95 main_nonhelper <= 2% bbox:
    good, FK target is consistent enough

2% < p95 <= 5% bbox:
    usable, but report the mismatch floor in the training doc

p95 > 5% bbox or many max outliers:
    do not hide it; either fix those samples or use lower w_fk first
```

This is not a defensive blocker. It is required because `pred==gt` will not have
zero FK loss if the two GT routes disagree.

## 8. Smoke Gates

Add or extend smoke scripts with these gates:

1. Torch FK recovery matches the existing numpy official port on a small set of
   real clips:

```text
max_abs_error < 1e-4
mean_error_percent_bbox matches numpy report
```

2. `loss_mode=anytop13` remains numerically identical to the current default.

3. `loss_mode=anytop13_world_rot6d_fk` returns finite keys:

```text
world
fk
traj
gt_fk_mismatch
total
```

4. Backward pass gives nonzero gradient to non-root rotation channels:

```text
grad(pred_motion[:, :, 1:, 3:9]) > 0
```

This is the key difference from `world_geometry`.

5. World/RIC branch gives nonzero gradient to the current pose route:

```text
grad(pred_motion[:, :, 1:, 0:3]) > 0
```

6. Masking uses `frame_mask_recovered`, not raw `batch.frame_mask`, so stride-tail
   padded frames are not penalized.

7. GT route mismatch is logged. Do not assert zero.

## 9. Visual QA

Metric is not enough. Render the long-chain and membrane-risk set:

```text
PZ_Asian_Water_Monitor_*
PZ_Komodo_Dragon_*
PZ_Saltwater_Crocodile_*
PZ_Grey_Seal_*
dragon/wing analogs if present
```

For each checkpoint, render three comparisons:

```text
1. current visual route:
   GT_RIC vs PRED_RIC

2. new FK route:
   GT_RIC vs PRED_FK_ROT6D

3. consistency route:
   PRED_RIC vs PRED_FK_ROT6D
```

Decision logic:

```text
If PRED_RIC improves but PRED_FK does not:
    world supervision helps the current renderer, but rotations are still not
    geometrically meaningful. Increase/repair FK branch before claiming FK.

If PRED_FK improves but PRED_RIC does not:
    FK supervision helps rotation channels, but world/RIC is too weak or base
    channel losses dominate. Recalibrate w_world vs w_fk.

If both PRED_FK and PRED_RIC improve:
    combined world+FK supervision is doing what we want.

If neither improves:
    long-chain problem is more likely pool/decoder/data distribution than loss.
```

## 10. What Not To Do In v1

- Do not add a new rotation head.
- Do not add `pred_aux_rot6d`.
- Do not change the renderer by default.
- Do not remove the base `w_rot` channel loss.
- Do not remove `anytop13_world_geometry`; keep it as the world-only ablation.
- Do not add long-chain-only weights yet.
- Do not use generic `treeik_decoder.fk_persample` unless a smoke proves it is
  numerically equivalent to the official AnyTop `recover_from_bvh_rot_np` route.

## 11. Implementation Order

1. Run the local L2 Route-A-vs-Route-B preflight scan and save JSON/summary.
2. Add `rot6d_fk_recovery.py` with matrix-only torch official-route FK.
3. Add `compute_world_rot6d_fk_terms` as a pure append in `losses.py`.
4. Export the new function in `src/models/graph_salad/__init__.py`.
5. Wire `train_graph_vae.py`:
   - loss_mode choice
   - `--w_world`
   - `--w_fk`
   - `--w_traj`
   - explicit total accumulation in `run_loss`
6. Add smoke:
   - numpy parity
   - default path unchanged
   - FK grad to non-root rot
   - one real train-loop smoke
7. Codex review before spending GPUs.
8. Launch B (`0.25/0.25/0.10`) and optionally C (`0.5/0.5/0.25`).
9. Render visual QA before judging from metric.

## 12. Expected Claim If It Works

Safe wording:

```text
We add combined world/RIC and true rot6d-FK geometric supervision for
arbitrary-topology VAE training. The world/RIC term keeps the current pose/render
route aligned with the AnyTop world-space target. The FK term maps predicted 6D
joint rotations through each sample's parent tree and rest offsets, then
supervises the resulting joint positions against the same target. This makes
both the pose route and non-root rotation channels geometrically meaningful,
while exposing accumulated distal errors on long chains, tails, and wings.
```

Do not claim this is PRISM's exact loss implementation. The correct claim is:

```text
PRISM-inspired geometric consequence supervision, adapted to AnyTop's arbitrary
topology and 13-channel RIC/rot6d representation.
```
