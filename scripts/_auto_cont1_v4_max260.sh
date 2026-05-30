#!/bin/bash
# Auto-launch cont1 (ep1001-2000) for v4 max_frames=260 denoiser when ep1000 training_complete.
# Deploy via: ssh swarma1003 "setsid nohup bash <this>.sh > <this>.log 2>&1 < /dev/null &"

set -u
SRC=runs/m2_denoiser_v4_max260_C96_ddp2a100_lr5e-4_1000ep_fulldata_seed42
DST=runs/m2_denoiser_v4_max260_C96_ddp2a100_lr5e-4_1000ep_fulldata_seed42_cont1
WORKDIR=/scratch/ts1v23/workspace/noKslot_clean
JOBID=925438
INIT_CKPT=$SRC/last_model.pt
VAE_CKPT=runs/m1_7_anytop13_edge_segment_C96_fulldata_ddp2a100_seed42/last_model.pt
CAP_CACHE=data/anytop_caption_t5_1070_multi.npz

cd $WORKDIR

echo "[auto_cont1 v4] $(date) waiting for '=== training complete ===' in $SRC/train.log"
until grep -q "=== training complete ===" $SRC/train.log 2>/dev/null; do sleep 30; done
echo "[auto_cont1 v4] $(date) training complete detected. sleeping 30s for ckpt finalization."
sleep 30

if [ ! -s "$INIT_CKPT" ]; then
  echo "[auto_cont1 v4] FATAL: $INIT_CKPT missing/empty. abort."; exit 1
fi
echo "[auto_cont1 v4] init_ckpt OK: $INIT_CKPT ($(stat -c%s $INIT_CKPT) bytes)"

mkdir -p $DST
echo "[auto_cont1 v4] $(date) launching cont1 in alloc $JOBID -> $DST"

srun --jobid=$JOBID --overlap --ntasks=1 --gres=gpu:2 bash -c "
source /scratch/ts1v23/.conda/etc/profile.d/conda.sh && conda activate graph_salad
cd $WORKDIR
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTHONUNBUFFERED=1 \
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
scripts/train_denoiser.py \
  --init_ckpt $INIT_CKPT \
  --vae_ckpt $VAE_CKPT \
  --caption_emb_cache $CAP_CACHE \
  --max_frames 260 \
  --epochs 1000 --batch_size 16 --lr 5e-4 --weight_decay 1e-6 \
  --warmup_iters 2000 --grad_clip 1.0 \
  --n_layers 5 --dropout 0.1 \
  --num_train_timesteps 1000 --beta_start 0.00085 --beta_end 0.012 --beta_schedule scaled_linear \
  --cond_drop_prob 0.1 --val_every 10 --save_every 20 \
  --periodic_save_every 500 \
  --full_data_val_species 'Dragon,Monkey,Centipede,Horse' \
  --seed 42 \
  --out $DST --overwrite
" > $DST/_launch_stdout.log 2>&1

echo "[auto_cont1 v4] $(date) srun returned exit=$?"
