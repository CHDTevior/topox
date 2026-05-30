#!/bin/bash
# EXPERIMENTAL — 8-card cross-alloc 2-node DDP VAE retrain on CLEANED L2.
# swarma1003 (alloc 925438, node_rank 0, MASTER) + swarma1001 (alloc 925439,
# node_rank 1), 4 A100 each = 8 ranks. Standalone experiment, separate run dir,
# does NOT touch train_graph_vae.py or any verified launch script. If NCCL
# cross-node handshake fails, just delete this file — zero rollback risk.
#
# Called once per node with NODE_RANK env set:
#   NODE_RANK=0 bash _exp_8card_2node_ddp.sh   # on swarma1003 (master)
#   NODE_RANK=1 bash _exp_8card_2node_ddp.sh   # on swarma1001
#
# cross-alloc cgroup blocks NCCL P2P/SHM -> force TCP/IB (lesson §8.13):
#   NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_SOCKET_IFNAME=ib0
set -u

P=/scratch/ts1v23/workspace/noKslot_clean
cd "$P" || exit 1

MASTER_ADDR="${MASTER_ADDR:-10.6.15.9}"   # swarma1003 ib0
MASTER_PORT="${MASTER_PORT:-29517}"
NODE_RANK="${NODE_RANK:?set NODE_RANK=0 (master swarma1003) or 1 (swarma1001)}"
SMOKE="${SMOKE:-0}"

OUT=runs/_exp_m1_l2_cleanL2_8card2node_seed42
mkdir -p "$OUT"

echo "[exp8] $(date '+%F %T %Z') host=$(hostname) NODE_RANK=$NODE_RANK MASTER=$MASTER_ADDR:$MASTER_PORT CVD=${CUDA_VISIBLE_DEVICES:-unset} SMOKE=$SMOKE"

SMOKE_FLAG=""
[ "$SMOKE" = "1" ] && SMOKE_FLAG="--smoke"

# global batch target ~256 over 8 ranks => 32/rank (same per-rank as orig A100
# cont1). global batch 256 vs cont1's 128 => lr scaled x2 per Goyal linear rule:
# lr 4e-4 -> 8e-4. (epochs halved to keep epoch-count if it ran full, but this
# is a 12h experiment so just let it run.)
NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_SOCKET_IFNAME=ib0 \
NCCL_IB_DISABLE=0 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTHONUNBUFFERED=1 \
torchrun \
  --nnodes=2 --node_rank="$NODE_RANK" --nproc_per_node=4 \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  scripts/train_graph_vae.py \
  --dataset anytop_truebones --feat_mode anytop13 --attn_mode graphormer \
  --decoder_mode coarse_xattn --pool_type edge_segment \
  --anytop_root /scratch/ts1v23/workspace/noKslot_clean/data/anytop_planet_zoo_clean_L2 \
  --val_frac 0.05 --batch_size 32 --lr 8e-4 --seed 42 --epochs 300 \
  --save_every 5 --periodic_save_every 50 --d_model 512 --n_heads 8 --d_ff 1536 \
  --n_graph_layers 4 --n_enc_temporal_layers 2 --n_cross_layers 3 \
  --n_dec_temporal_layers 2 --n_treeik_layers 3 --max_coarse 128 --local_radius 8 \
  --temporal_stride 4 --max_frames 64 --max_joints 144 --use_name_embed \
  --out "$OUT" --overwrite $SMOKE_FLAG
echo "[exp8] $(date '+%F %T %Z') node_rank=$NODE_RANK torchrun EXITED rc=$?"
