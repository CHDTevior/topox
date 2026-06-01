# PRISM-Style FK Loss Experiment Plan

Date: 2026-05-30
Scope: design document only. No code changes in this file.

## 0. Position

This is the planned follow-up after the current PRISM-inspired per-joint latent
diagnostic:

```text
pool_type = none
decoder_mode = graph_temporal
max_joints = 144
```

The current diagnostic asks whether preserving one latent token per joint helps
long chains, wings, tails, and membrane-like structures.

This document describes the next experiment:

```text
Does PRISM-style FK / trajectory supervision improve VAE reconstruction quality
under the same architecture?
```

This is an exploration branch, not a default replacement for the current
anytop13 loss.

## 1. PRISM Loss Takeaway

PRISM's VAE is useful to us for two related reasons:

1. It keeps a per-joint latent grid instead of compressing all joints into one
   monolithic frame token.
2. Its VAE loss supervises the physical consequence of rotations using
   forward kinematics.

The relevant PRISM objective is:

```text
L_vae =
  lambda_param  * L_param
+ lambda_joints * L_joints
+ lambda_traj   * L_traj
+ lambda_KL     * L_KL
```

Meaning:

```text
L_param  : L1 on native motion parameters, such as rotations and root motion
L_joints : FK reconstructed 3D joints vs GT 3D joints
L_traj   : cumulative root trajectory supervision
L_KL     : VAE latent regularization
```

The important part for our long-chain problem is not just "add another loss".
It is this:

```text
small proximal rotation error
  -> large distal tail / wing / foot position error after FK
```

A plain per-channel rotation loss treats the proximal and distal consequences
too equally. FK supervision exposes the accumulated geometric error.

## 2. Current noKslot_clean Loss

For `feat_mode=anytop13`, the current loss path is:

```text
scripts/train_graph_vae.py::run_loss
  -> src/models/graph_salad/losses.py::compute_total_loss_13ch
```

Current loss:

```text
L_current =
  w_pos     * L1(pred[..., 0:3],  gt[..., 0:3])
+ w_rot     * L1(pred[..., 3:9],  gt[..., 3:9])
+ w_vel     * L1(pred[..., 9:12], gt[..., 9:12])
+ w_contact * BCE(pred[..., 12],  gt_contact)
+ w_KL      * KL(q(z|x) || N(0,I))
+ w_aux     * pool_aux
```

Default weights:

```text
w_pos     = 1.0
w_rot     = 1.0
w_vel     = 1.0
w_contact = 0.1
w_KL      = 1e-3
w_aux     = 0.5
```

For `pool_type=edge_segment`, v2 pool aux is zero.
For `pool_type=none`, pool aux is also zero.

So the current practical objective is mostly:

```text
channel-space reconstruction + KL
```

This is simple and stable, but it can miss geometry-level failures:

- predicted position channels may look reasonable while predicted rotations are
  kinematically inconsistent;
- rotation errors at the base of a tail or wing can produce large distal visual
  errors but only small average channel loss;
- velocity channels can match per-frame values while cumulative trajectory
  still drifts or overshoots.

## 3. Proposed Loss Mode

Add a new optional loss mode:

```text
loss_mode = "anytop13_prism_fk"
```

Keep the existing mode as default:

```text
loss_mode = "anytop13"
```

The new total should be:

```text
L_prism_fk =
  w_param  * L_param_current
+ w_fk     * L_fk_joints
+ w_traj   * L_traj_cumsum
+ w_KL     * L_KL
+ w_aux    * L_pool_aux
```

Where:

```text
L_param_current =
  w_pos * L_pos + w_rot * L_rot + w_vel * L_vel + w_contact * L_contact
```

Initial exploratory weights:

```text
w_param = 1.0
w_fk    = 0.5
w_traj  = 0.25
w_KL    = 1e-3
w_aux   = 0.5
```

Do not add long-chain-specific weights in the first version. First test whether
plain PRISM-style geometry supervision already helps.

## 4. FK Joint Loss

Use raw, un-normalized 6D rotations:

```text
pred_rot_raw = denorm(pred_motion[..., 3:9])
gt_rot_raw   = denorm(gt_motion[..., 3:9])
```

Then compute FK positions:

```text
P_fk_pred = FK(pred_rot_raw, pred_root, rest_offsets, parent_indices)
P_fk_gt   = FK(gt_rot_raw,   gt_root,   rest_offsets, parent_indices)
```

Loss:

```text
L_fk_joints = mean_masked |P_fk_pred - P_fk_gt|
```

Use valid joint and frame masks:

```text
joint_mask [B,J]
frame_mask_recovered [B,T]
```

Why FK-vs-FK instead of immediately using recovered AnyTop world positions:

- This is closer to PRISM.
- It forces the predicted rotation channels to carry real geometric meaning.
- It directly targets accumulated rotation-chain error.

However, the implementation should also log a non-training diagnostic:

```text
recover_world(pred_raw) vs recover_world(gt_raw)
```

because that is what our current visual QA renders.

## 5. Trajectory Cumsum Loss

Use root velocity channels in raw anytop13 space:

```text
pred_root_vel = pred_raw[:, :, root, 9:12]
gt_root_vel   = gt_raw[:, :, root, 9:12]
```

Then accumulate:

```text
R_pred[t] = sum_{i <= t} pred_root_vel[i] * dt
R_gt[t]   = sum_{i <= t} gt_root_vel[i]   * dt
```

Loss:

```text
L_traj_cumsum = mean_masked |R_pred - R_gt|
```

This is intended to catch:

- root drift;
- overshoot;
- jitter from locally plausible but cumulatively wrong velocity;
- low-speed clips where small velocity errors become visually large.

Implementation detail:

AnyTop world recovery uses root x/z velocity plus root rotation. For v1 of this
loss, keep `L_traj_cumsum` simple and compute it in the same raw root-velocity
coordinate frame. If this proves useful, v2 can switch to the exact AnyTop
world-recovery trajectory.

## 6. Code Touch Points

Expected implementation files:

```text
src/models/graph_salad/losses.py
scripts/train_graph_vae.py
src/models/graph_salad/batch.py
src/data/anytop_dataset.py  # likely no change, but verify emitted fields
```

Existing helpers to reuse:

```text
src/models/treeik_decoder.py::rot6d_to_matrix
src/models/treeik_decoder.py::fk_persample
```

Current data already contains:

```text
batch.parent_indices
batch.rest_offsets
batch.local_rotations_6d
raw["anytop_mean"]
raw["anytop_std"]
```

But `GraphMotionBatch` currently validates and stores only some optional
AnyTop fields. The implementation likely needs to add:

```text
anytop_mean: Optional[torch.Tensor]  # [B,J,13]
anytop_std:  Optional[torch.Tensor]  # [B,J,13]
```

to `GraphMotionBatch`, because the loss must denormalize `pred_motion` inside
the training loop.

CLI additions:

```text
--loss_mode anytop13 | anytop13_prism_fk
--w_fk_joints 0.5
--w_traj_cumsum 0.25
```

All new weights must default to zero or be inactive unless
`--loss_mode anytop13_prism_fk` is selected, so old experiments remain
reproducible.

## 7. Minimal Experiment Matrix

Run after the per-joint latent diagnostic finishes.

Use the same active cleaned dataset:

```text
data/anytop_planet_zoo_clean_L2
```

Recommended first A/B:

```text
A: winning architecture from p=1 diagnostic + loss_mode=anytop13
B: same architecture                     + loss_mode=anytop13_prism_fk
```

If p=1/no-pool clearly wins:

```text
A: pool_type=none, decoder_mode=graph_temporal, original loss
B: pool_type=none, decoder_mode=graph_temporal, prism_fk loss
```

If edge_segment remains better:

```text
A: pool_type=edge_segment, decoder_mode=coarse_xattn or graph_temporal, original loss
B: same architecture, prism_fk loss
```

Do not change pool, decoder, d_model, dataset, or max_frames in the loss A/B.
The only experimental variable should be the loss mode.

## 8. QA Set

Use the same fixed visual QA set as the p=1 diagnostic:

```text
long-chain / wing / membrane:
  Dragon-like wing clips if present
  Asian Water Monitor
  Grey Seal
  Crocodile / Alligator tail
  Elephant trunk / long appendage if visually useful

anti-regression:
  ordinary quadrupeds
  spider / multi-leg examples
  short-chain clean movers
```

Required visual outputs:

```text
GT vs recon GIF
same camera
same epoch or best_recon checkpoint
multi-frame, not frame-0 only
```

Do not accept a result based only on aggregate `val_recon`.

## 9. Metrics To Log

Training loss components:

```text
pos
rot
vel
contact
kl
pool_aux
fk_joints
traj_cumsum
total
```

Validation diagnostics:

```text
val_recon_original_terms
val_fk_joints
val_traj_cumsum
speed_ratio_mean
macro_per_species_pos
```

Long-chain local diagnostics, if cheap:

```text
distal_tip_displacement_error
parent_child_edge_error
chain_velocity_error
```

These can be metric-only diagnostics first. Do not make them training losses in
v1.

## 10. Expected Outcomes

### Good outcome

```text
prism_fk loss improves Dragon / Water Monitor / Seal long-chain visuals
without harming normal animals
```

Interpretation:

```text
the VAE needed geometry-level supervision;
pool design was not the only bottleneck.
```

Next step:

```text
keep prism_fk as candidate default for the next VAE training branch;
then test with hybrid_prism_segment if p=1 diagnostic also supports it.
```

### Neutral outcome

```text
metrics move but visuals do not improve
```

Interpretation:

```text
FK loss may be optimizing a geometric proxy that does not match AnyTop visual recovery.
```

Next step:

```text
try world-recovery loss instead of FK-vs-FK, or stop loss branch.
```

### Bad outcome

```text
loss becomes unstable or normal animals regress
```

Likely causes:

```text
6D rotation denormalization / FK precision issue;
FK loss weight too high;
root trajectory coordinate-frame mismatch;
```

First response:

```text
reduce w_fk_joints and w_traj_cumsum by 2-4x;
verify pred_raw and gt_raw finite before FK;
render rest-pose / FK sanity examples.
```

## 11. Smoke Gates

Before full training:

1. One batch forward/backward with `loss_mode=anytop13_prism_fk`.
2. All loss components finite.
3. `fk_joints` and `traj_cumsum` are non-zero on moving clips.
4. Gradients finite after backward.
5. Existing `loss_mode=anytop13` produces identical loss values to pre-change
   code on the same batch.
6. Render one GT-vs-recon GIF after a tiny overfit or smoke checkpoint to ensure
   visualization path still works.

## 12. Implementation Prompt

Use this after the current p=1/no-pool diagnostic finishes:

```text
Implement an optional PRISM-style FK loss experiment for noKslot_clean VAE.

Read:
- handoff/20260530_2243_prism_fk_loss_experiment_plan.md
- handoff/20260530_2155_prism_inspired_vae_long_chain_plan.md
- src/models/graph_salad/losses.py
- scripts/train_graph_vae.py
- src/models/graph_salad/batch.py
- src/models/treeik_decoder.py
- src/data/anytop_dataset.py

Goal:
- Add loss_mode="anytop13_prism_fk" without changing the default anytop13 loss.
- Implement PRISM-like:
  L = L_param_current + w_fk_joints * L_fk_joints + w_traj_cumsum * L_traj_cumsum + KL + pool_aux.
- Reuse 6D rotation FK helpers where possible.
- Denormalize pred_motion and gt_motion using anytop_mean / anytop_std inside the loss.
- Add CLI flags for loss_mode, w_fk_joints, w_traj_cumsum.
- Keep old checkpoints and old training configs loadable.

Validation:
- Smoke original loss and prism_fk loss.
- Verify old loss path is numerically unchanged.
- Run a short A/B on the winning architecture from the p=1 diagnostic.
- Render long-chain GT-vs-recon GIFs before judging.

Do not:
- Change pool_type.
- Change decoder architecture.
- Add long-chain-specific training weights in v1.
- Change denoiser.
```

