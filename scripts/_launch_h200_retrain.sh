#!/bin/bash
# From-scratch VAE retrain on CLEANED L2 (81994 motions, 41 velocity-risk
# clips removed, cond.npy std recomputed + floored). 2x H200 (alloc 976854,
# CVD=0,1 only — GPU 2,3 belong to yx1g22, DO NOT touch).
#
# Config = proven cont1 recipe, rescaled for 2 ranks:
#   global batch 128 = 64/rank x 2  (smoke: peak 128.4/143GB OK)
#   lr 4e-4 UNCHANGED (global batch unchanged => same optimization dynamics)
#   from scratch (no init_ckpt), epochs 300, seed 42
#
# Run durably via:
#   setsid nohup srun --jobid=976854 --overlap --ntasks=1 \
#       bash scripts/_launch_h200_retrain.sh > <log> 2>&1 < /dev/null &
set -u

P=/scratch/ts1v23/workspace/noKslot_clean
cd "$P" || exit 1
OUT=runs/m1_l2_anytop13_C128_d512_h8_cleanL2_h200x2_seed42

# Guard: never double-launch this run
if pgrep -f "train_graph_vae.py.*cleanL2_h200x2" >/dev/null 2>&1; then
    echo "[launch] ABORT: cleanL2_h200x2 training already running"; exit 0
fi
mkdir -p "$OUT"

echo "[launch] $(date '+%F %T %Z') CVD=$CUDA_VISIBLE_DEVICES host=$(hostname)"
echo "[launch] out=$OUT  global_batch=128 (64/rank x2)  lr=4e-4  from-scratch"

torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_graph_vae.py \
  --dataset anytop_truebones --feat_mode anytop13 --attn_mode graphormer \
  --decoder_mode coarse_xattn --pool_type edge_segment \
  --anytop_root /scratch/ts1v23/workspace/noKslot_clean/data/anytop_planet_zoo_clean_L2 \
  --val_frac 0.05 --batch_size 64 --lr 4e-4 --seed 42 --epochs 300 \
  --save_every 5 --periodic_save_every 50 --d_model 512 --n_heads 8 --d_ff 1536 \
  --n_graph_layers 4 --n_enc_temporal_layers 2 --n_cross_layers 3 \
  --n_dec_temporal_layers 2 --n_treeik_layers 3 --max_coarse 128 --local_radius 8 \
  --temporal_stride 4 --max_frames 64 --max_joints 144 --use_name_embed \
  --out "$OUT" --overwrite
echo "[launch] $(date '+%F %T %Z') torchrun EXITED rc=$?"
