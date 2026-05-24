#!/bin/bash
# Auto-launch cont1 (ep1001-2000) for v2 edge_segment C=96 when ep1000 training_complete.
# Deploy via: ssh swarma1003 "setsid nohup bash <this>.sh > <this>.log 2>&1 < /dev/null &"

set -u
SRC=runs/m1_7_anytop13_edge_segment_C96_fulldata_ddp2a100_seed42
DST=runs/m1_7_anytop13_edge_segment_C96_fulldata_ddp2a100_seed42_cont1
WORKDIR=/scratch/ts1v23/workspace/noKslot_clean
JOBID=925437
INIT_CKPT=$SRC/last_model.pt
MAX_COARSE=96

cd $WORKDIR

echo "[auto_cont1 C=96] $(date) waiting for '=== training complete ===' in $SRC/train.log"
until grep -q "=== training complete ===" $SRC/train.log 2>/dev/null; do sleep 30; done
echo "[auto_cont1 C=96] $(date) training complete detected. sleeping 30s for ckpt finalization."
sleep 30

if [ ! -s "$INIT_CKPT" ]; then
  echo "[auto_cont1 C=96] FATAL: $INIT_CKPT missing/empty. abort."; exit 1
fi
echo "[auto_cont1 C=96] init_ckpt OK: $INIT_CKPT ($(stat -c%s $INIT_CKPT) bytes)"

mkdir -p $DST
echo "[auto_cont1 C=96] $(date) launching cont1 in alloc $JOBID -> $DST"

srun --jobid=$JOBID --overlap --ntasks=1 --gres=gpu:2 bash -c "
source /scratch/ts1v23/.conda/etc/profile.d/conda.sh && conda activate graph_salad
cd $WORKDIR
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTHONUNBUFFERED=1 \
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
scripts/train_graph_vae.py \
  --init_ckpt $INIT_CKPT \
  --dataset anytop_truebones --feat_mode anytop13 \
  --attn_mode graphormer --decoder_mode coarse_xattn \
  --pool_type edge_segment \
  --batch_size 16 --lr 4e-4 --seed 42 \
  --epochs 1000 --save_every 10 \
  --d_model 384 --n_heads 8 --d_ff 1024 \
  --n_graph_layers 4 --n_enc_temporal_layers 2 --n_cross_layers 3 \
  --n_dec_temporal_layers 2 --n_treeik_layers 3 \
  --max_coarse $MAX_COARSE --local_radius 8 --temporal_stride 4 \
  --max_frames 64 --max_joints 143 \
  --use_name_embed \
  --full_data_val_species 'Dragon,Monkey,Centipede,Horse' \
  --out $DST --overwrite
" > $DST/_launch_stdout.log 2>&1

echo "[auto_cont1 C=96] $(date) srun returned exit=$?"
