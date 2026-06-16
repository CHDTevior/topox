#!/bin/bash
# Render one truebones T2M group (10 truebones species, DDIM50, CFG1.5, with GT energy ratio).
# Mirrors EXACTLY the proven ep500 render (scripts/_render_truebones_diffusion_ep500.log):
# truebones specVAE + truebones caption caches + dual_text (token cache) + --split all (1070).
# Sibling of scripts/_render_one_t2m.sh (which is pz20-only); kept separate so the pz20 path is untouched.
# Args: GPU  CKPT(rel)  OUTDIR(rel)
set -eu
cd /scratch/ts1v23/workspace/noKslot_clean
GPU=$1; CKPT=$2; OUT=$3
PY=/scratch/ts1v23/.conda/bin/python3
VAE=runs/m1_bf16_anytop13_TRUEBONES_rot6dfk_w025f100t010_C128_4card_seed42/best_recon_model.pt
SP=Bat,Centipede,Crab,Eagle,Flamingo,Horse,Lion,Raptor,Spider,Trex
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"; export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"  # compute node no-internet
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=$GPU $PY -m scripts.animate_denoiser \
  --vae_ckpt "$VAE" --denoiser_ckpt "$CKPT" \
  --caption_emb_cache data/anytop_caption_t5_truebones_multi.npz \
  --caption_token_cache data/anytop_caption_t5_truebones_multi \
  --anytop_root data/anytop_truebones --split all \
  --species "$SP" --n_per 1 --cond_scale 1.5 --n_ddim_steps 50 --large --with_gt --seed 42 \
  --out "$OUT"
echo "RENDER_DONE $OUT"
