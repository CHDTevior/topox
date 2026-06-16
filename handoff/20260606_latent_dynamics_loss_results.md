# Latent Temporal Dynamics Loss — Experiment Results (NEGATIVE)

Date: 2026-06-06
Status: **B-sample concluded — latent-dynamics loss is a no-op on generation quality. Pivoting to dual-stream text conditioning.** (B-mu / mu-target arm still training as a confirmatory diagnostic.)

## 1. What was tested

Hypothesis (handoff/20260605_latent_temporal_dynamics_loss_experiment.md): the diffusion v-MSE only supervises one-step velocity, not the temporal trajectory of the implied clean latent. Adding a small penalty on the temporal derivatives of `z0_hat = √ᾱ_t·z_t − √(1−ᾱ_t)·v_pred` should fix the diagnosed failure (fast-action targets decode to static / mean-energy motion).

Loss: `L = L_v + 0.05·‖Δz0_hat−Δz0‖² + 0.02·‖Δ²z0_hat−Δ²z0‖²` (w_lat_x0=0). Two arms differing ONLY in the Δ/Δ² target:
- **B-sample** — target = sampled z0 (`latent_dyn_target=sample`). 8×A100. `runs/m2_capacity_pz20_latdyn_dz005_ddz002_bf16MEAN_lr6.67e-5cos_a100x8_seed42`. Stopped at ep120 (best val_denoise 0.3951).
- **B-mu** — target = posterior mean mu (`latent_dyn_target=mu`). 4×H100. `runs/m2_capacity_pz20_latdyn_dz005_ddz002_MUtarget_h100x4_lr4.17e-5cos_seed42`. Resumed/continuing (was at ep52, best val 0.3618).

Baseline A (no latdyn loss, w=0): `runs/m2_capacity_pz20_bf16MEAN_lr6.67e-5cos_a100x8_seed42` (the 20-species capacity probe, ep310). Setup otherwise identical: 20 high-count species, train_split=all, frozen bf16 VAE (ep209), mean_additive pooled-T5 text, n11/d_ff1536 (63.45M), DDIM v-prediction.

Code: `train_denoiser.py` latdyn loss (codex PASS 019e99fb), launcher threading (codex PASS 019e9a10), 4-card H100 orchestrator (codex PASS 019e9a44). All smoke-passed. **Uncommitted.**

## 2. Result — speed_ratio (PRED_pose_speed / GT_speed; 1.0 = correct energy)

ep100 matched checkpoints (ep0100_model.pt), cfg1.5 (deploy), all 20 val species, --with_gt.
Artifacts: `runs/_qa_ep100_all20/{Bsample_cfg1.5,baselineA_cfg1.5}/` (20 gifs each + animate_summary.txt).

| species | GT_speed | A (no latdyn) | B-sample (latdyn) | |
|---|---:|---:|---:|---|
| Hamadryas_Baboon | 0.025 | 1.83 | 1.59 | slow |
| Sun_Bear | 0.027 | 2.38 | 2.38 | slow |
| Hippopotamus | 0.027 | 2.52 | 2.21 | slow |
| Ocelot | 0.039 | 3.40 | 1.59 | slow |
| Formosan_Black_Bear | 0.045 | 1.96 | 3.74 | slow (worse) |
| Little_Penguin | 0.049 | 1.15 | 1.24 | slow |
| King_Penguin | 0.054 | 1.19 | 1.17 | med |
| Amur_Leopard | 0.055 | 2.12 | 2.13 | med |
| Red_Panda | 0.078 | 1.12 | 1.15 | med |
| Siamang | 0.082 | 0.96 | 1.04 | med |
| Bush_Dog | 0.094 | 1.22 | 1.27 | med |
| Cougar | 0.095 | 1.47 | 1.30 | med |
| Koala | 0.096 | 0.54 | 0.74 | med |
| Raccoon | 0.121 | 0.59 | 0.47 | med |
| Western_Chimpanzee | 0.139 | 0.82 | 0.78 | med |
| Bonobo | 0.215 | 0.61 | 0.61 | FAST |
| Japanese_Macaque | 0.260 | 0.10 | 0.10 | FAST (collapsed) |
| Proboscis_Monkey | 0.313 | 0.19 | 0.18 | FAST |
| Jaguar | 0.323 | 0.27 | 0.29 | FAST |
| Siberian_Tiger | 0.350 | 0.45 | 0.43 | FAST |

**Aggregates:** overall mean ratio A=1.245 → B=1.220 (unchanged). **FAST targets (GT>0.2, n=5): A=0.325 → B=0.321 — statistically identical.**

5-species cfg1.5 + cfg7.5 (earlier, same conclusion): `runs/_qa_ep100/{Bsample,baselineA}_cfg{1.5,7.5}/`.

## 3. Verdict

**The latent temporal dynamics loss (sample target, w=0.05/0.02, ep100) does NOT improve generation.**
- Fast targets (the actual problem — they decode static, ratio 0.1–0.6) are UNCHANGED vs baseline.
- Slow targets shuffle noisily (some better e.g. Ocelot 3.40→1.59, some worse e.g. Formosan 1.96→3.74), no systematic direction.
- Confirmed by (a) USER visual review of cfg1.5 gifs ("没啥改善 / 完全不行") and (b) train `lat_dz` staying flat over ep45–120 (1.14–1.22) — the term never reduced, i.e. it was a weak constant pressure that did not reshape the decoded motion.
- This matches the pre-experiment honesty note: val_lat_dz fell only in step with val_denoise, so the dynamics terms had no measurable effect beyond v-MSE.

cfg7.5 raised B-sample's Jaguar (0.59→0.90) but over-energizes slow targets (Hippo 2.2→? ; not a deploy setting).

## 4. Why (diagnosis context — see also the decoupler from 2026-06-05)

Earlier read-only decoupling established the failure is NOT the VAE (deterministic VAE recon preserves both fast & slow energy, recon_ratio ≈1 incl. fastest Jaguar 0.96) and NOT undertraining/capacity (controlled ep100/200/260 trajectory flat). It is the **diffusion regressing to the conditional mean** under under-specified conditioning. The latent-dynamics loss was the minimal attempt to inject the missing temporal-energy signal — but at these weights it does not overpower the conditional-mean collapse.

Root suspicion now centers on the **conditioning being too weak**: text is `mean_additive` — the whole caption is mean-pooled into one [768] T5 vector and broadcast-added uniformly to every latent token, so speed/dynamics words ("runs", "slowly") are diluted into a bag-of-meaning. The model cannot attend to specific words → the target action's energy is under-specified → conditional-mean collapse. (See denoiser.py:280 — mean_additive and token_cross_attn are mutually-exclusive branches.)

## 5. Artifacts / paths

- Run dirs (ckpts + metrics): `runs/m2_capacity_pz20_latdyn_dz005_ddz002_bf16MEAN_lr6.67e-5cos_a100x8_seed42/` (B-sample, ep120), `runs/m2_capacity_pz20_latdyn_dz005_ddz002_MUtarget_h100x4_lr4.17e-5cos_seed42/` (B-mu, continuing).
- Baseline A: `runs/m2_capacity_pz20_bf16MEAN_lr6.67e-5cos_a100x8_seed42/`.
- ep100 QA gifs (4-col: input skeleton | PRED pose | PRED FK | GT):
  - all-20 cfg1.5: `runs/_qa_ep100_all20/Bsample_cfg1.5/`, `runs/_qa_ep100_all20/baselineA_cfg1.5/`
  - 5-species cfg1.5+7.5: `runs/_qa_ep100/Bsample_cfg{1.5,7.5}/`, `runs/_qa_ep100/baselineA_cfg{1.5,7.5}/`
- Decoupler (VAE faithful / not-capacity): `runs/m2_capacity_pz20_bf16MEAN_lr6.67e-5cos_a100x8_seed42/qa_VAErecon_val5/`, `qa_captraj_ep{100,200,260}_val5_withGT/`, `qa_cfgsweep_ep260_cfg{3,5,7.5}_val5/`.
- Code (uncommitted): `train_denoiser.py` (latdyn loss + helpers), `_launch_diffusion_t2m.sh` + `_launch_token_diffusion_8card_a100.sh` (W_LAT_* threading), `_launch_diffusion_t2m_4card.sh` (4-card H100 orchestrator).

## 6. Next direction

The minimal latent-dynamics loss is insufficient. **Pivot: dual-stream text conditioning** (`text_mode=dual_text` = global mean-pool stream + token-level cross-attn stream together, instead of the current either/or). Goal: richer conditioning so the target action (incl. its speed/energy) is better specified, attacking the conditional-mean collapse at its source. All future experiments to use dual-stream text. Data layer already emits both `caption_emb [768]` and `caption_token_emb [L,768]` (idx-aligned); denoiser + train + animate need the new fused mode.
