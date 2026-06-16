#!/bin/bash
# Trigger-render: wait for B-sample's ep0100 ckpt, then render B-sample-ep100 vs
# baseline-A-ep100 (matched epoch) on the same 5 val species, cfg1.5 + cfg7.5,
# --with_gt, on IDLE blossom03 H200 (not training cards). External-condition poll
# (waiting on a training process the harness doesn't track) -> poll loop is right.
set -uo pipefail
cd /iridisfs/scratch/ts1v23/workspace/noKslot_clean

O_S=runs/m2_capacity_pz20_latdyn_dz005_ddz002_bf16MEAN_lr6.67e-5cos_a100x8_seed42   # B-sample
O_A=runs/m2_capacity_pz20_bf16MEAN_lr6.67e-5cos_a100x8_seed42                        # baseline A (no latdyn)
VAE=runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42/best_recon_model.pt
SP=PZ_Koala_Female,PZ_Jaguar_Female,PZ_Proboscis_Monkey_Juvenile,PZ_Siberian_Tiger_Juvenile,PZ_Hippopotamus_Male

echo "[ep100] waiting for $O_S/ep0100_model.pt ..."
until [ -f "$O_S/ep0100_model.pt" ]; do sleep 30; done
echo "[ep100] B-sample ep0100 ckpt present; A ep0100 exists=$([ -f $O_A/ep0100_model.pt ] && echo yes || echo NO)"

render() {  # tag ckpt cfg
  local tag="$1" ck="$2" cfg="$3"
  srun --jobid=976856 --overlap --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --time=25:00 \
    bash -lc "python -m scripts.animate_denoiser --vae_ckpt $VAE --denoiser_ckpt $ck \
      --caption_emb_cache data/anytop_caption_t5_cleanL2_multi.npz \
      --anytop_root data/anytop_planet_zoo_clean_L2 --split val --species $SP \
      --n_per 1 --cond_scale $cfg --n_ddim_steps 50 --large --with_gt --seed 42 \
      --out runs/_qa_ep100/${tag}_cfg${cfg}" && echo "[ep100] $tag cfg$cfg done"
}

for CFG in 1.5 7.5; do
  render Bsample "$O_S/ep0100_model.pt" "$CFG"
  render baselineA "$O_A/ep0100_model.pt" "$CFG"
done
echo "EP100_RENDER_DONE"
