#!/usr/bin/env bash
# Launch the AnyTop T2M evaluator training (M4). GATED: written for review; do NOT run
# until the user approves. Independent frozen metric model — trains ONLY on real motions
# (train_main manifest), no generator/VQVAE/CodeFlow weights, frozen DistilBERT text backbone.
#
# TOPOLOGY: single-node multi-GPU via torchrun (simplest). For CROSS-ALLOC over several
# H100 allocs on the same node, reuse the proven backbone orchestrator pattern in
# scripts/_launch_graph_pscf_2node_h200.sh (static rendezvous + NCCL P2P/SHM-disable +
# node_rank 0..k-1); the only change is swapping its python command for the one below.
# The evaluator needs NO token-export / empirical-prewarm (it trains directly on motions).
set -euo pipefail
ROOT=/scratch/ts1v23/workspace/noKslot_clean
PY=/scratch/ts1v23/.conda/bin/python
cd "$ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# ============================ HYPERPARAMETERS (review) ============================
NGPU=${NGPU:-8}                 # GPUs on this node (ranks). global_batch = NGPU * PER_RANK
PER_RANK=${PER_RANK:-32}        # per-rank batch -> global 8*32 = 256 (spec §6.1 minimum)
LR=${LR:-2e-4}                  # 2e-4 @ global 256 (Goyal: scale to 3-4e-4 if global -> 512)
EPOCHS=${EPOCHS:-100}           # ~72k train motions / gb256 ~= 280 steps/epoch -> ~28k steps
WARMUP=${WARMUP:-2000}          # linear warmup steps, then cosine decay
WD=${WD:-1e-4}                  # weight decay (spec 1e-6..1e-4)
GRAD_CLIP=${GRAD_CLIP:-1.0}
MASTER_PORT=${MASTER_PORT:-29531}
NUM_WORKERS=${NUM_WORKERS:-6}

DATA=data/animo4d_anytop_clean_L4_safe_plus_truebones
OUT=runs/anytop_t2m_evaluator_distilbert_coemb512_gb$((NGPU*PER_RANK))_lr${LR}_seed42
# ==================================================================================

echo "[launch] evaluator: NGPU=$NGPU PER_RANK=$PER_RANK global_batch=$((NGPU*PER_RANK)) "
echo "         lr=$LR epochs=$EPOCHS warmup=$WARMUP -> OUT=$OUT"

$PY -m torch.distributed.run \
  --nproc_per_node="$NGPU" --master_port="$MASTER_PORT" \
  scripts/train_anytop_t2m_evaluator.py \
    --manifest "$DATA/eval_splits/train_main.json" \
    --data_root "$DATA" \
    --split train --view full \
    --text_tower distilbert \
    --distilbert_path checkpoints/text_encoders/distilbert-base-uncased \
    --num_frames 300 --max_joints 144 \
    --coemb_dim 512 --n_heads 8 --d_ff 2048 \
    --n_graph_layers 6 --n_temporal_layers 4 --dropout 0.1 \
    --temperature 0.07 \
    --batch_size "$PER_RANK" --lr "$LR" --weight_decay "$WD" \
    --warmup_steps "$WARMUP" --grad_clip "$GRAD_CLIP" --epochs "$EPOCHS" \
    --bf16 --num_workers "$NUM_WORKERS" \
    --val_manifest "$DATA/eval_splits/val_all.json" \
    --val_every "${VAL_EVERY:-5}" --val_batch_size 32 --val_max_batches "${VAL_MAX_BATCHES:-64}" \
    --out "$OUT" --save_every 5 --log_every 50

echo "[launch] DONE -> $OUT"
