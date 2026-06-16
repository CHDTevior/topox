#!/bin/bash
# Render one T2M group (20 PZ species, DDIM50, CFG1.5, with GT energy ratio).
# Replicates the prior _qa_latest render exactly, just swapping the checkpoint.
# Args: GPU  CKPT(rel)  OUTDIR(rel)  USE_TOKEN(1=dual_text/0=global)
set -eu
cd /scratch/ts1v23/workspace/noKslot_clean
GPU=$1; CKPT=$2; OUT=$3; USE_TOKEN=$4
PY=/scratch/ts1v23/.conda/bin/python3
VAE=runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42/best_recon_model.pt
SP=PZ_Amur_Leopard_Juvenile,PZ_Bonobo_Juvenile,PZ_Bush_Dog_Male,PZ_Cougar_Male,PZ_Formosan_Black_Bear_Male,PZ_Hamadryas_Baboon_Male,PZ_Hippopotamus_Male,PZ_Jaguar_Female,PZ_Japanese_Macaque_Juvenile,PZ_King_Penguin_Male,PZ_Koala_Female,PZ_Little_Penguin_Male,PZ_Ocelot_Female,PZ_Proboscis_Monkey_Juvenile,PZ_Raccoon_Juvenile,PZ_Red_Panda_Male,PZ_Siamang_Male,PZ_Siberian_Tiger_Juvenile,PZ_Sun_Bear_Female,PZ_Western_Chimpanzee_Male
TOKARG=""
[ "$USE_TOKEN" = "1" ] && TOKARG="--caption_token_cache data/anytop_caption_t5_cleanL2_multi"
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=$GPU $PY -m scripts.animate_denoiser \
  --vae_ckpt "$VAE" --denoiser_ckpt "$CKPT" \
  --caption_emb_cache data/anytop_caption_t5_cleanL2_multi.npz $TOKARG \
  --anytop_root data/anytop_planet_zoo_clean_L2 --split val \
  --species "$SP" --n_per 1 --cond_scale 1.5 --n_ddim_steps 50 --large --with_gt --seed 42 \
  --out "$OUT"
echo "RENDER_DONE $OUT"
