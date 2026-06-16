# Latent Temporal Dynamics Loss Experiment

Date: 2026-06-05

Scope: small-data Phase-2 diffusion experiment on the existing 20-species capacity probe. This document is a plan only; it does not change code.

## 1. Problem

The 20-species capacity probe shows a consistent mismatch between VAE capacity and diffusion sampling quality.

Observed QA:

| Case | VAE recon speed ratio | diffusion cfg1.5 ratio | diffusion cfg7.5 ratio |
| --- | ---: | ---: | ---: |
| Hippo slow clip | 1.11 | 1.32 | 1.72 |
| Jaguar fast climb transition | 0.96 | 0.18 | 0.25 |
| Koala jump/climb | 1.05 | 0.78 | 1.26 |
| Proboscis monkey fast climb | 1.00 | 0.39 | 0.91 |
| Tiger run | 1.01 | 0.48 | 0.97 |

Interpretation:

- VAE is not the bottleneck for high-energy motion. Deterministic VAE recon preserves both fast and slow motion energy.
- Backbone diffusion is the weak stage. It produces latent trajectories whose decoded motion is either low-energy / mean-like, or becomes over-energized under high global CFG.
- A single scalar CFG cannot solve this because different samples need different energy correction.

Current train objective in `scripts/train_denoiser.py`:

```text
vae.encode(batch, sample=True) -> z0
noise, t
z_t = scheduler.add_noise(z0, noise, t)
v_target = scheduler.get_velocity(z0, noise, t)
v_pred = denoiser(z_t, t, text, graph)
loss = masked_mse(v_pred, v_target)
```

This objective supervises one-step velocity prediction in latent diffusion space. It does not directly ask the predicted clean latent trajectory to match the ground-truth temporal dynamics.

## 2. Hypothesis

Adding a small temporal dynamics penalty on the implied clean latent `z0_hat` will reduce latent dynamics collapse and improve decoded motion energy without changing VAE, data, text path, scheduler, or model architecture.

The key idea:

```text
v_pred predicts the diffusion velocity.
From v_pred we can reconstruct the model's implied clean latent:

z0_hat = sqrt(alpha_t) * z_t - sqrt(1 - alpha_t) * v_pred
```

Then compare temporal differences:

```text
L_dz  = MSE( z0_hat[:, 1:] - z0_hat[:, :-1],
             z0[:,     1:] - z0[:,     :-1] )

L_ddz = MSE( z0_hat[:, 2:] - 2*z0_hat[:, 1:-1] + z0_hat[:, :-2],
             z0[:,     2:] - 2*z0[:,     1:-1] + z0[:,     :-2] )
```

This targets exactly the observed failure mode: the sampled latent sequence has the wrong speed / acceleration profile even when VAE can decode correct motion from true latents.

## 3. What This Experiment Does Not Do

Do not add oracle energy conditioning in this experiment.

Do not change:

- frozen VAE checkpoint
- VAE training
- denoiser architecture
- text mode
- dataset split / species subset
- DDIM scheduler
- CFG formula
- rendering code

The only intended variable is the extra loss term.

## 4. Implementation Plan

File to modify:

- `scripts/train_denoiser.py`

Add CLI args with zero defaults so existing runs are byte-equivalent when the flags are not used:

```text
--w_lat_dz float default 0.0
--w_lat_ddz float default 0.0
--w_lat_x0 float default 0.0
--latent_dyn_target choices sample,mu default sample
```

Recommended first active setting:

```text
--w_lat_dz 0.05
--w_lat_ddz 0.02
--w_lat_x0 0.0
--latent_dyn_target sample
```

Rationale:

- `sample` is consistent with the existing diffusion target because current training uses `vae.encode(..., sample=True)`.
- Posterior-noise diagnostics did not show posterior sampling itself causing motion jitter, so supervising sampled `z0` is acceptable as the first clean test.
- `mu` should remain an available switch if `sample` makes the temporal loss noisy.

Add helper functions near `masked_v_mse`:

```python
def predict_z0_from_v(z_t, v_pred, timesteps, scheduler):
    alphas = scheduler.alphas_cumprod.to(device=z_t.device, dtype=z_t.dtype)
    a = alphas[timesteps].sqrt().view(-1, 1, 1, 1)
    b = (1.0 - alphas[timesteps]).sqrt().view(-1, 1, 1, 1)
    return a * z_t - b * v_pred

def masked_latent_mse(pred, target, mask):
    mask_f = mask.float()
    diff_sq = (pred.float() - target.float()).pow(2) * mask_f
    return diff_sq.sum() / (mask_f.sum() * pred.shape[-1]).clamp(min=1.0)

def masked_latent_dz_mse(z0_hat, z0_target, coarse_mask, frame_mask):
    dz_p = z0_hat[:, 1:] - z0_hat[:, :-1]
    dz_t = z0_target[:, 1:] - z0_target[:, :-1]
    m = (
        coarse_mask[:, None, :, None]
        & frame_mask[:, 1:, None, None]
        & frame_mask[:, :-1, None, None]
    )
    return masked_latent_mse(dz_p, dz_t, m)

def masked_latent_ddz_mse(z0_hat, z0_target, coarse_mask, frame_mask):
    ddz_p = z0_hat[:, 2:] - 2.0 * z0_hat[:, 1:-1] + z0_hat[:, :-2]
    ddz_t = z0_target[:, 2:] - 2.0 * z0_target[:, 1:-1] + z0_target[:, :-2]
    m = (
        coarse_mask[:, None, :, None]
        & frame_mask[:, 2:, None, None]
        & frame_mask[:, 1:-1, None, None]
        & frame_mask[:, :-2, None, None]
    )
    return masked_latent_mse(ddz_p, ddz_t, m)
```

Inside the train step, after `v_pred`:

```python
loss_v = masked_v_mse(v_pred, v_target, coarse_mask, frame_mask)
loss = loss_v

if args.w_lat_dz > 0 or args.w_lat_ddz > 0 or args.w_lat_x0 > 0:
    z0_hat = predict_z0_from_v(z_t.float(), v_pred.float(), timesteps, sched)

    if args.latent_dyn_target == "mu":
        z0_dyn_target = enc["mu"].float()
        z0_dyn_target = z0_dyn_target * mask_4d
    else:
        z0_dyn_target = z0

    if args.w_lat_x0 > 0:
        loss_x0 = masked_latent_mse(z0_hat, z0_dyn_target, mask_4d.bool())
        loss = loss + args.w_lat_x0 * loss_x0
    if args.w_lat_dz > 0:
        loss_dz = masked_latent_dz_mse(z0_hat, z0_dyn_target, coarse_mask, frame_mask)
        loss = loss + args.w_lat_dz * loss_dz
    if args.w_lat_ddz > 0:
        loss_ddz = masked_latent_ddz_mse(z0_hat, z0_dyn_target, coarse_mask, frame_mask)
        loss = loss + args.w_lat_ddz * loss_ddz
```

Do the same component computation in validation for logging. The best-checkpoint gate can remain `val_denoise` at first, but the metrics file must also record:

```text
train_v_mse
train_lat_dz
train_lat_ddz
train_total
val_denoise
val_lat_dz
val_lat_ddz
```

Important: the extra losses must be computed in fp32, same as the existing masked loss.

## 5. Equivalence Gate

Before running the active experiment, run a smoke with all new weights at zero:

```text
--w_lat_dz 0 --w_lat_ddz 0 --w_lat_x0 0
```

Expected:

- old checkpoints still strict-load
- no new model state_dict keys
- forward / backward unchanged except for args and logging
- first-iteration `loss` matches the previous `masked_v_mse` path

This is mandatory because `train_denoiser.py` is also used by the full-data runs.

## 6. Experiment Setup

Use the exact 20-species capacity probe setup as the baseline:

```text
data root:
  data/anytop_planet_zoo_clean_L2

VAE:
  runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42/best_recon_model.pt

caption cache:
  data/anytop_caption_t5_cleanL2_multi.npz

species whitelist:
  same 20 species as runs/m2_capacity_pz20_bf16MEAN_lr6.67e-5cos_a100x8_seed42

train_split:
  all

denoiser:
  text_mode mean_additive
  n_layers 11
  d_model 512
  d_ff 1536
  params about 63.45M

training:
  max_frames 260
  max_joints 144
  batch_size 8 per GPU
  lr 6.667e-5
  warmup_iters 400
  lr_schedule cosine
  amp_dtype bf16
  cond_drop_prob 0.1
  val_every 5
```

Suggested output:

```text
runs/m2_capacity_pz20_latdyn_dz005_ddz002_bf16MEAN_lr6.67e-5cos_a100x8_seed42
```

Use the existing baseline run as A:

```text
runs/m2_capacity_pz20_bf16MEAN_lr6.67e-5cos_a100x8_seed42
```

Train B at least to epoch 300. If visual and latent diagnostics improve, continue to 500 before deciding whether to promote to the full 473-species run.

## 7. Evaluation Gates

Do not judge by `val_denoise` alone.

Required checks at ep100, ep200, ep300:

1. Standard metric curve:
   - `val_denoise`
   - `val_lat_dz`
   - `val_lat_ddz`

2. Latent dynamics diagnostic:
   - true latent speed/accel ratio vs sampled latent speed/accel ratio
   - reuse the same five val species used in the current QA:
     - Hippo
     - Jaguar
     - Koala
     - Proboscis monkey
     - Tiger

3. Motion energy summary:
   - VAE recon ratio
   - diffusion cfg1.5 ratio
   - diffusion cfg7.5 ratio

4. Visual QA:
   - 4-column gifs: input skeleton | PRED pose | PRED FK | GT
   - same clips as the current ep260 energy QA
   - compare against the existing baseline gifs, not only against metrics

Success criteria:

- Jaguar / Tiger / Proboscis speed ratios improve at cfg1.5 without making Hippo / Koala over-energetic.
- Latent speed/accel ratio moves closer to 1.
- Visual motion shows correct large-scale transition / run / climb behavior, not just higher jitter.
- `val_denoise` may stay similar or slightly worsen; that is acceptable if visual QA and latent dynamics improve. A large degradation in `val_denoise` with no visual gain is a fail.

Failure criteria:

- `val_lat_ddz` drops but visual motion becomes more jittery. That means the loss is fitting latent finite differences without improving decoded motion.
- High-energy samples improve only under cfg7.5 while slow samples over-shoot more. That means the experiment did not solve calibration.
- `sample` target makes `loss_dz/loss_ddz` noisy and unstable. If this happens, rerun with `--latent_dyn_target mu`.

## 8. Weight Sweep Policy

Start with one run:

```text
w_lat_dz=0.05
w_lat_ddz=0.02
w_lat_x0=0.0
latent_dyn_target=sample
```

If the effect is too weak by ep100:

```text
w_lat_dz=0.10
w_lat_ddz=0.05
```

If motion becomes smoother but under-energetic:

- keep `w_lat_ddz`
- increase `w_lat_dz`

If motion becomes noisy / over-constrained:

- lower `w_lat_ddz` first
- keep or slightly lower `w_lat_dz`

Do not turn on `w_lat_x0` in the first run. Direct `x0` loss is a stronger objective change and can obscure whether the improvement came from temporal dynamics specifically.

## 9. Why This Is the Right First Fix

The failure is visible after full DDIM sampling, but training currently supervises only one-step `v` prediction. The model can reduce v-MSE while still producing latent trajectories whose temporal derivatives decode to wrong motion energy.

This experiment adds the smallest possible signal that directly touches that gap:

```text
current:  "predict the correct v at each noised point"
new:      "and make the implied clean latent move through time like the real latent"
```

It is cheaper and cleaner than switching to flow matching immediately, because it preserves the current scheduler, sampler, VAE, data loader, text pipeline, and denoiser architecture.

## 10. Agent Prompt

Implement the latent temporal dynamics loss experiment described in `handoff/20260605_latent_temporal_dynamics_loss_experiment.md`.

Hard requirements:

1. Only modify `scripts/train_denoiser.py` unless a smoke test proves another file is necessary.
2. Add zero-default CLI flags:
   - `--w_lat_dz`
   - `--w_lat_ddz`
   - `--w_lat_x0`
   - `--latent_dyn_target {sample,mu}`
3. With all weights zero, existing behavior must remain unchanged:
   - no new denoiser state_dict keys
   - old checkpoints strict-load
   - `loss == masked_v_mse` path remains the active loss
4. Use the v-prediction reconstruction:
   - `z0_hat = sqrt(alpha_t) * z_t - sqrt(1 - alpha_t) * v_pred`
5. Compute all extra losses in fp32 with the same valid `coarse_mask × frame_mask_lat` semantics as the existing loss.
6. Log train/val component losses separately.
7. Run smoke tests:
   - zero-weight equivalence smoke
   - active-loss smoke with `w_lat_dz=0.05,w_lat_ddz=0.02`
8. Run codex review on the code diff before launching the long B experiment.
9. After smoke + review, run the 20-species B experiment with the same setup as `runs/m2_capacity_pz20_bf16MEAN_lr6.67e-5cos_a100x8_seed42`, changing only the new loss flags and output directory.
10. Render the same ep100/ep200/ep300 visual QA set and report speed ratios plus visual judgment.

Do not implement oracle energy conditioning in this task.
