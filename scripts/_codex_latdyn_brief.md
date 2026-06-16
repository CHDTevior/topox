# Codex review: M2 latent temporal dynamics loss (train_denoiser.py)

You are reviewing a code change BEFORE a multi-hour training run. Be adversarial; the goal is to catch correctness bugs, silent no-ops, mask/shape errors, fp32/equivalence breaks, and anything that would waste the run. Verdict format: PASS / NEEDS-FIX with a numbered list.

## What to read (in the repo, current working tree)
1. The plan/spec (authoritative): `handoff/20260605_latent_temporal_dynamics_loss_experiment.md`
2. The diff: run `git diff scripts/train_denoiser.py`
3. Smoke logs (already run, should both succeed): `scripts/_smoke_latdyn_A.log` (zero-weight + init_ckpt strict-load) and `scripts/_smoke_latdyn_B.log` (active w_lat_dz0.05 w_lat_ddz0.02).

## Context (enough to review; do not require more)
- Phase-2 latent diffusion T2M. `GraphSaladDenoiser` predicts v (v-prediction) on FROZEN GraphMotionVAE latents `z0` of shape `[B, T_lat, C, D]` (T_lat=65 latent frames, C=coarse slots, D=feature dim). Existing loss = `masked_v_mse(v_pred, v_target, coarse_mask, frame_mask)`, fp32-safe, over valid `coarse_mask[:,None,:,None] & frame_mask[:,:,None,None]` positions.
- Diagnosis motivating this change: the VAE reconstructs fast/slow motion energy faithfully (recon ratio ~1), but the diffusion-sampled latent has wrong temporal dynamics (latent jitter ~3x real z0; decoded motion energy regresses to a low/mean value for fast targets). v-MSE supervises one-step velocity, NOT the cross-time trajectory of the implied clean latent.
- Fix (this diff): add an OPTIONAL extra loss on the temporal derivatives of the v-implied clean latent `z0_hat = sqrt(alpha_bar_t)*z_t - sqrt(1-alpha_bar_t)*v_pred`:
  `L = L_v + w_lat_dz*MSE(Δz0_hat, Δz0) + w_lat_ddz*MSE(Δ²z0_hat, Δ²z0) [+ w_lat_x0*MSE(z0_hat,z0)]`
  along the latent-time axis (axis=1). New CLI args `--w_lat_dz/--w_lat_ddz/--w_lat_x0` (default 0.0) and `--latent_dyn_target {sample,mu}` (default sample). First active run: w_lat_dz=0.05, w_lat_ddz=0.02, w_lat_x0=0, target=sample.

## Required correctness checks (be specific)
1. **v-param inversion**: is `z0_hat = sqrt(alpha_bar_t)*z_t - sqrt(1-alpha_bar_t)*v_pred` the correct clean-sample reconstruction for diffusers' v_prediction given `z_t = sqrt(abar)*z0 + sqrt(1-abar)*noise`? Check `predict_z0_from_v` broadcasting (`alphas[timesteps].sqrt().view(-1,1,1,1)` against `[B,T_lat,C,D]`).
2. **Δ / Δ² masking**: in `masked_latent_dz_mse` / `masked_latent_ddz_mse`, the temporal differences reduce T_lat by 1 / 2. Verify the masks `coarse_mask[:,None,:,None] & frame_mask[:,1:,...] & frame_mask[:,:-1,...]` (and the 3-frame version) correctly count ONLY positions where ALL contributing frames are valid, and align index-wise with `dz_p`/`ddz_p`. Any off-by-one in the frame_mask slicing vs the difference slicing?
3. **fp32 safety**: are all extra losses computed in fp32 like `masked_v_mse` (so the bf16 autocast denominator-rounding bug does not reappear)? `predict_z0_from_v(z_t.float(), v_pred.float(), ...)` and `masked_latent_mse` `.float()`.
4. **Zero-weight byte-equivalence (CRITICAL — train_denoiser.py is also used by full-data runs)**: with `w_lat_dz=w_lat_ddz=w_lat_x0=0`, is the loss EXACTLY `masked_v_mse` (loss == loss_v), with no extra compute, no extra metrics keys, identical epoch-log format, and no change to the denoiser nn.Module (so old ckpts strict-load and there are no new state_dict keys)? Confirm `lat_active` gating in BOTH the train step and the val loop. Verify smoke A (`--init_ckpt` of an old capacity ckpt, strict=True) loaded with 0 missing/unexpected keys.
5. **z0 sampled vs target**: train uses `z0 = vae.encode(sample=True)`. The dynamics target with `--latent_dyn_target sample` is this sampled z0; with `mu` it is `enc["mu"]`. Is `enc["mu"]` a valid key and correctly masked (`* mask_4d`)? Is using a stochastic per-step z0 as the dynamics target sound, or does it inject avoidable variance (the handoff offers `mu` as the fallback)?
6. **Autocast placement**: the extra-loss block runs inside `with amp_ctx():` (bf16 autocast). Since z_t/v_pred are `.float()`ed, are the elementwise diffs/MSE actually fp32 under autocast, or could any op be downcast? Is the total `loss` (v + dynamics) covered by the `torch.isfinite(loss)` fail-fast and `clip_grad_norm_(error_if_nonfinite=True)`?
7. **Grad flow**: do `w_lat_dz/w_lat_ddz` terms actually backprop into the denoiser (z0_hat depends on v_pred which is the denoiser output)? Any detached tensor that silently zeroes the new gradient?
8. **Val logging**: val computes `val_lat_dz/val_lat_ddz` as batch-mean-of-per-batch-means (diagnostic only; best-ckpt gate stays element-weighted `val_denoise`). Acceptable? Any divide-by-zero when a val batch has T_lat<3 (ddz needs ≥3 frames)? Note some clips have few latent frames.
9. **Metrics schema**: when active, metrics.jsonl rows gain `train_v_mse/train_lat_dz/train_lat_ddz/train_total/val_lat_dz/val_lat_ddz` (handoff §4). When inactive, identical to before. Confirm.

## Scope note
The diff also contains PRE-REVIEWED hunks (codex threads 019e95f0 lr_schedule/cosine + 019e98dc species_whitelist/train_split) — `--lr_schedule/--lr_min/--species_whitelist/--train_split`, `lr_for()` cosine, `total_iters`, ds_train `split=args.train_split`. You may sanity-check them but the NEW work to scrutinize is the latent temporal dynamics loss (helpers `predict_z0_from_v/masked_latent_mse/masked_latent_dz_mse/masked_latent_ddz_mse`, the train-step component block, the train/val accumulation + logging, the metrics additions, and the 4 new args).

Only `scripts/train_denoiser.py` is modified. The launcher/orchestrator that threads `--w_lat_*` env vars for the real 8×A100 run is NOT in this diff (will be a separate review). Report PASS / NEEDS-FIX.
