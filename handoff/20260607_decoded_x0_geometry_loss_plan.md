# Decoded-x0 Geometry Loss for Backbone Diffusion

Date: 2026-06-07

Status: **plan only**. This document does not change code.

## 0. Bottom Line

We have **not** run this experiment yet.

What we have already tried is a latent-only dynamics loss:

```text
L = L_v
  + 0.05 * ||Delta z0_hat - Delta z0||^2
  + 0.02 * ||Delta2 z0_hat - Delta2 z0||^2
```

That experiment is documented in `handoff/20260606_latent_dynamics_loss_results.md` and was negative: FAST target speed ratio stayed essentially unchanged (`0.325 -> 0.321`).

The experiment proposed here is different:

```text
v_pred -> z0_hat -> frozen VAE decode -> pred_motion
then supervise pred_motion in world/FK/speed space.
```

So this is **not** another latent dz/ddz run. It is decoded clean-latent geometry supervision.

Update after plan review:

- Keep this experiment. The direction is still technically sound.
- Add one pre-training conditioning probe to decide whether decoded loss alone is likely enough, or whether explicit speed conditioning is required.
- Add one early transfer probe to detect train/inference mismatch before spending a full run.
- Tighten implementation details: fp32 decoded branch, safe log-speed clamp/calibration, and report both RIC/pose speed and FK speed.

## 1. Research Context

### MDM / human diffusion lesson

MDM predicts the clean motion `x0` rather than only noise. The useful consequence is not the name "x0 prediction" itself; it is that the training loop can directly apply geometric losses on the model's estimated clean motion: joint positions, velocities, and foot-contact-style constraints.

Reference:
- MDM paper: https://arxiv.org/abs/2209.14916

### PRISM lesson

PRISM uses a larger recipe: joint-factorized latent VAE, FK-supervised VAE, flow matching DiT, token-level text, and noise-free condition injection. Flow matching may be useful later, but it is not required just to attach geometry supervision.

Reference:
- PRISM paper: https://arxiv.org/abs/2603.08590
- Local note: `handoff/20260602_2038_prism_diffusion_backbone_iteration_notes.md`

### Our interpretation

Our current backbone uses DDIM `v_prediction`, not x0 prediction. That is fine. For v-prediction, the implied clean latent is still available:

```text
z0_hat = sqrt(alpha_bar_t) * z_t - sqrt(1 - alpha_bar_t) * v_pred
```

Current code already has this helper:

- `scripts/train_denoiser.py:143-149` — `predict_z0_from_v`

Therefore the minimal transplant of the MDM-style idea is:

```text
keep v-prediction as the base objective
recover z0_hat from v_pred
decode z0_hat through the frozen VAE
apply geometry / speed losses on decoded motion
```

Do **not** switch to flow matching in this experiment. Flow matching is a larger scheduler/objective/sampler rewrite and would confound the result.

## 2. Current Code Truth

Current Phase-2 training path:

- `scripts/train_denoiser.py:798-804` — frozen VAE encode, `sample=True`, producing `z0`
- `scripts/train_denoiser.py:840-848` — add noise and get `v_target`
- `scripts/train_denoiser.py:854-867` — denoiser forward and base `masked_v_mse`
- `scripts/train_denoiser.py:871-883` — optional latent-only `z0_hat` x0/dz/ddz losses

Current optional loss is still latent-space only:

- `scripts/train_denoiser.py:152-184` — `masked_latent_mse`, `masked_latent_dz_mse`, `masked_latent_ddz_mse`
- It never calls `vae.decode`.
- It never computes recovered world positions.
- It never measures visible speed from decoded skeleton motion.

Existing geometry helpers are VAE-side and reusable:

- `src/models/graph_salad/losses.py:627-674` — `compute_world_geometry_terms`
- `src/models/graph_salad/losses.py:689-747` — `compute_world_rot6d_fk_terms`

Sampling already proves the decode pattern exists:

- `scripts/animate_denoiser.py:162-179` — builds fake encode dict with a replacement `z`
- `scripts/animate_denoiser.py:493-496` — `vae.decode(fake_enc, batch)` produces `pred_motion [B,T,J,13]`

The training experiment should reuse this pattern, but **without** `torch.no_grad()` around decode, because gradients must flow:

```text
loss_geo -> pred_motion -> frozen VAE decoder -> z0_hat -> v_pred -> denoiser
```

The VAE parameters remain frozen; only the gradient path through its operations is needed.

## 3. Hypothesis

The failure mode is not VAE capacity. VAE reconstruction preserves fast and slow motion energy; diffusion sampling collapses fast targets toward conditional mean / low-energy motion.

The latent dz/ddz loss did not fix this because latent derivatives are only an indirect proxy. A decoded geometry/speed loss attacks the visual failure directly:

```text
The generated clean latent must decode into a skeleton whose world motion
has the right positions, root trajectory, and speed magnitude.
```

The most important term for the current problem is speed / energy. World L1 and root trajectory help, but the fast-vs-slow failure is directly visible as wrong mean speed.

## 3.5 Pre-training Conditioning Probe

Before launching the decoded-loss run, run a cheap no-training probe:

```text
input:  cached text condition + skeleton / graph metadata
target: log(GT_speed)
task:   regress or rank per-motion speed
```

Purpose:

```text
Does the current conditioning contain enough information to infer motion energy?
```

Recommended variants:

```text
P0 text-only:       caption_emb or caption_token_emb -> log(GT_speed)
P1 skeleton-only:   pooled skeleton / species proxy -> log(GT_speed)
P2 text+skeleton:   both -> log(GT_speed)
```

Report:

```text
R2
Spearman rho
MAE on log speed
fast/slow bucket accuracy
```

Interpretation:

- If text+skeleton predicts speed well, decoded geometry loss is plausible: the information is present, but the current v-MSE objective does not expose the visible speed error.
- If text+skeleton predicts speed poorly, decoded loss alone is unlikely to recover the correct speed from an underdetermined condition. Then explicit speed conditioning, oracle-speed ablation, or richer caption conditioning becomes necessary.

This probe does not replace the decoded-loss experiment; it tells us how to read the result.

## 4. Proposed Objective

Keep the base objective:

```text
L_v = masked MSE(v_pred, v_target)
```

Add decoded clean-latent geometry:

```text
z0_hat = predict_z0_from_v(z_t, v_pred, timesteps, scheduler)
fake_enc = shallow_copy(enc)
fake_enc["z"] = z0_hat
dec = frozen_vae.decode(fake_enc, batch)
pred_motion = dec["pred_motion"]              # [B,T,J,13], normalized
frame_mask_rec = dec["frame_mask_recovered"]  # [B,T]
gt_motion = batch.anytop_x.permute(0, 3, 1, 2)
```

Then:

```text
L_total =
    L_v
  + w_world * L_world
  + w_traj  * L_traj
  + w_speed * L_speed
  + optional w_fk * L_fk
```

### 4.1 World and root trajectory

Use existing AnyTop-native geometry route first:

```text
terms = compute_world_geometry_terms(
    pred_motion=pred_motion,
    gt_motion=gt_motion,
    anytop_mean=batch.anytop_mean,
    anytop_std=batch.anytop_std,
    joint_mask=batch.joint_mask,
    frame_mask=frame_mask_rec,
)

L_world = terms["world"]
L_traj  = terms["traj"]
```

This matches our renderer/visual QA route and has already been made differentiable.

### 4.2 Speed / energy loss

Add a new helper that recovers world positions and compares temporal speed magnitude:

```text
P_pred = recover_world_positions_torch(denorm(pred_motion))
P_gt   = recover_world_positions_torch(denorm(gt_motion))

speed_pred = ||P_pred[:, 1:] - P_pred[:, :-1]||_2
speed_gt   = ||P_gt[:,   1:] - P_gt[:,   :-1]||_2

valid = joint_mask
      & frame_mask_rec[:, 1:]
      & frame_mask_rec[:, :-1]

L_speed = masked Huber(
    log(speed_pred + eps) - log(speed_gt + eps)
)
```

Use log-speed rather than raw speed for the first experiment. Raw L1 can be dominated by high-speed clips; log-speed treats "2x too fast" and "0.5x too slow" symmetrically.

Clamp both predicted and target speeds before the log:

```text
speed_pred_safe = clamp(speed_pred, min=speed_floor, max=speed_ceil)
speed_gt_safe   = clamp(speed_gt,   min=speed_floor, max=speed_ceil)
```

Do **not** clamp only GT. If `speed_pred -> 0`, `log(speed_pred + eps)` can create very large gradients and destabilize the decoded branch.

Choose `speed_floor` and `speed_ceil` by a short calibration scan over real training data, preferably quantile-based:

```text
speed_floor = max(1e-4, q001_speed)
speed_ceil  = q999_speed
```

Also log the gradient norm contribution of the decoded branch during smoke. If `w_dec_speed * grad_norm_speed` dominates the base v-loss gradient, reduce `w_dec_speed`.

For near-static GT speeds, either skip or downweight:

```text
speed_gt > speed_floor
```

Suggested initial `speed_floor`: calibrate from one training batch; if no calibration is available, start with `1e-4` in raw recovered world units and log how many valid speed entries are skipped.

### 4.3 Optional true FK branch

Do **not** enable FK in the first decoded-geometry run unless world+speed improves but rotations still look inconsistent.

If needed later:

```text
compute_world_rot6d_fk_terms(...)
L_fk = terms["fk"]
```

Reason: FK is useful for supervising ch3:9 rotations, but the current visual failure is primarily motion energy. Adding FK immediately makes the first experiment harder to interpret.

## 5. Timestep Gating

Do not apply decoded geometry at all timesteps initially.

At high noise, `z0_hat` can be a poor clean estimate early in training, and decoding it through VAE may inject noisy gradients. First run:

```text
geom_t_max = 400
apply geometry only when timestep < geom_t_max
```

Implementation:

```text
geom_sample_mask = timesteps < geom_t_max
```

Only decode the selected sub-batch if possible. If sub-batch slicing is too invasive, decode full batch but multiply geometry terms by `geom_sample_mask` in the frame/joint mask. Sub-batch is more memory efficient; full-batch is simpler.

Also add:

```text
geom_every = 1 by default
```

If memory/time is bad, use `geom_every=2` or `geom_every=4` so geometry is computed every N training steps.

Load-bearing risk:

```text
training supervises one-step z0_hat
inference uses a closed-loop 50-step DDIM trajectory
```

So geometry loss may improve teacher-forced `z0_hat` decode but fail to transfer to iterative sampling. This is the main risk of the plan.

Add an early transfer probe:

```text
teacher_forced_fast_ratio = speed_ratio(decode(z0_hat from validation noised latents))
closed_loop_fast_ratio    = speed_ratio(50-step DDIM sample)
transfer_ratio            = closed_loop_fast_ratio / teacher_forced_fast_ratio
```

Run this at an early checkpoint, for example ep20/ep50 on the 20-species probe.

Decision:

- If teacher-forced speed improves but closed-loop speed remains baseline-like, stop the run early. The loss is not transferring through the sampler.
- If both improve, continue training.
- If neither improves, weights or conditioning are ineffective.

## 6. Proposed CLI / Config

Add zero-default arguments to `scripts/train_denoiser.py`:

```text
--w_dec_world float default 0.0
--w_dec_traj  float default 0.0
--w_dec_speed float default 0.0
--w_dec_fk    float default 0.0
--dec_geom_t_max int default 400
--dec_geom_every int default 1
--dec_speed_floor float default 1e-4
--dec_speed_loss choices log_huber,log_l1,raw_l1 default log_huber
```

Zero weights must preserve current training behavior exactly, apart from harmless args/log fields.

For implementation clarity, keep the existing latent losses separate:

```text
w_lat_*      = latent-space losses
w_dec_*      = decoded motion/world-space losses
```

Do not reuse `w_lat_x0` for this experiment. That name means latent x0 MSE in the current code and would confuse the result.

## 7. First Experiment

Run on the same 20-species capacity probe, because that is where the failure is best isolated and turnaround is fast.

Baseline:

```text
A: dual_text + v_loss only
```

New run:

```text
B: dual_text + v_loss + decoded-x0 world/traj/speed
```

Keep identical:

- data root: `data/anytop_planet_zoo_clean_L2`
- same 20-species subset
- same train split policy
- same frozen VAE checkpoint
- same denoiser size
- same LR schedule
- same batch/resource shape if feasible
- same CFG/render prompts

Suggested initial weights after one-batch calibration:

```text
w_dec_world = choose so weighted term is ~2-5% of L_v
w_dec_traj  = choose so weighted term is ~1-3% of L_v
w_dec_speed = choose so weighted term is ~5-10% of L_v
w_dec_fk    = 0.0 for first run
```

If calibration is skipped, a conservative starting point:

```text
w_dec_world = 0.02
w_dec_traj  = 0.02
w_dec_speed = 0.10
w_dec_fk    = 0.0
dec_geom_t_max = 400
```

But calibration is strongly preferred because raw world-space loss scale depends on the dataset normalization stats.

## 8. Calibration Gate

Before launching training, run a smoke/calibration on one or a few batches:

Log raw values:

```text
loss_v
dec_world
dec_traj
dec_speed
weighted_total_extra / loss_v
valid_geom_batch_fraction
valid_speed_entry_fraction
```

Decision rule:

```text
weighted geometry total should start around 5-15% of L_v
speed should be the largest decoded term but not exceed L_v by itself
```

If weighted decoded losses exceed 30% of `L_v`, reduce weights before real training.

Implementation detail:

Run the decoded VAE branch and world-recovery branch in fp32. The denoiser forward can remain under `amp_ctx`, but decoded geometry should avoid bf16 accumulation/log/rotation numerical noise:

```text
with amp_ctx():
    v_pred = denoiser(...)
    loss_v = masked_v_mse(...)

if decoded_loss_active:
    with torch.cuda.amp.autocast(enabled=False):
        z0_hat = predict_z0_from_v(z_t.float(), v_pred.float(), ...)
        dec = vae.decode(fake_enc_float, batch)
        compute world / speed terms in fp32
```

The VAE is still frozen. This only changes numeric precision for the loss path.

## 9. Smoke Gates

Mandatory before real training:

1. **Zero-weight regression**
   - `w_dec_* = 0`
   - old checkpoint strict-loads
   - first-step loss equals `loss_v`
   - no VAE decode is called on zero-weight path

2. **Active decoded loss finite**
   - small smoke with `w_dec_speed > 0`
   - `loss_v`, `dec_speed`, `total` finite
   - backward succeeds

3. **VAE frozen but gradient flows**
   - all VAE params `requires_grad=False`
   - `v_pred.grad` / denoiser grad nonzero when decoded loss active
   - no VAE parameter receives optimizer update

4. **Mask correctness**
   - use `dec["frame_mask_recovered"]`, not raw `batch.frame_mask`, for decoded motion losses
   - use `batch.joint_mask`
   - skip invalid padded frames/joints

5. **No high-noise instability**
   - `dec_geom_t_max=400` path works
   - if `geom_sample_mask.sum()==0`, decoded terms become zero cleanly

## 10. Evaluation

Do not judge this experiment by `val_denoise` alone. It may worsen slightly because the new objective is not optimizing only v-MSE.

Primary QA:

- same 20-species GIF set as prior capacity probe
- 4-column layout: input skeleton | PRED pose | PRED FK | GT
- report both `PRED_pose_speed / GT_pose_speed` and `PRED_fk_speed / GT_fk_speed`
- split by slow / medium / fast targets

Success criterion:

```text
FAST targets improve substantially without making slow targets over-energetic.
```

Concrete first-pass gate:

```text
FAST mean ratio should move from ~0.32 toward >=0.55 at cfg1.5
slow mean ratio should not exceed baseline by >20%
visual QA should show less frozen/mean-motion behavior
```

Failure interpretation:

- If `dec_speed` falls but sampled speed ratio does not improve: training-time decoded `z0_hat` supervision is not transferring to iterative sampling; consider x0-pred or flow matching later.
- If teacher-forced decoded speed improves but closed-loop speed does not, the issue is sampler transfer rather than the decoded loss itself.
- If speed improves but motion becomes jittery: add smoothness/acceleration decoded term or reduce `w_dec_speed`.
- If slow clips overshoot: speed loss needs conditional weighting or log-speed clipping.
- If memory is unacceptable: use sub-batch decode or `dec_geom_every > 1`.

## 11. Why Not Flow Matching First

Flow matching could be useful later, especially if v-pred sampling dynamics are the limiting factor. But switching now would change several things at once:

- training target
- scheduler
- sampler
- possibly timestep embedding semantics
- calibration of CFG

That would make it unclear whether improvements come from geometry supervision or from the generative objective change.

The clean first test is:

```text
same v-pred backbone
same sampler
same VAE
same text mode
only add decoded clean-latent geometry/speed supervision
```

If this fails cleanly, then a second-stage objective change is justified:

1. x0-pred latent diffusion: model directly predicts `z0_hat`; decoded loss becomes simpler.
2. flow matching: model predicts a velocity field from noise to clean latent; decoded clean endpoint supervision can still be attached, but implementation is larger.

## 12. Implementation Checklist for Executor

1. Add `w_dec_*` CLI args with zero defaults.
2. Add helper to shallow-copy `enc` and replace `"z"` with `z0_hat`.
3. Add decoded geometry branch after `loss_v`.
4. Do **not** wrap `vae.decode` in `torch.no_grad()` for training geometry branch.
5. Keep VAE in `eval()` and all VAE params frozen.
6. Use `gt_motion = batch.anytop_x.permute(0, 3, 1, 2)`.
7. Use `dec["frame_mask_recovered"]` for decoded losses.
8. Implement `decoded_speed_log_huber` helper, preferably reusing `recover_world_positions_torch`.
   - compute in fp32
   - clamp both pred and GT speed before log
   - calibrate speed floor/ceil from quantiles
9. Add train/val metric logging:
   - `train_dec_world`
   - `train_dec_traj`
   - `train_dec_speed`
   - `train_dec_total`
   - `val_dec_world`
   - `val_dec_traj`
   - `val_dec_speed`
10. Add launcher env threading only after smoke passes.
11. Run codex review before launching long training.
12. Render GIFs before claiming success.
13. Before long training, run the conditioning probe.
14. During early training, run the teacher-forced-vs-closed-loop transfer probe.

## 13. One-Sentence Prompt for Implementation

Implement an optional decoded-x0 geometry loss in `scripts/train_denoiser.py`: keep the current DDIM v-prediction loss, recover `z0_hat` from `v_pred`, decode `z0_hat` through the frozen Graph-VAE with gradients flowing through the decoder but no VAE parameter updates, and add zero-default `w_dec_world/w_dec_traj/w_dec_speed` losses on recovered AnyTop world positions/speed using `frame_mask_recovered` and `joint_mask`; compute the decoded loss branch in fp32, clamp both predicted and GT speed before log-speed loss, report both RIC/pose and FK speed ratios, run a cheap conditioning probe before training, and run an early teacher-forced-vs-closed-loop transfer probe; first experiment should run on the existing 20-species dual_text capacity probe against a v-loss-only baseline, with visual speed-ratio GIF QA as the primary gate.
