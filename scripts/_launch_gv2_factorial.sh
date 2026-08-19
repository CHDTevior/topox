#!/bin/bash
# graph-v2 factorial launcher -- ONE arm per invocation, single GPU, run ON the compute node:
#   ssh <node> "cd /scratch/ts1v23/workspace/noKslot_clean && setsid nohup \
#     env ARM=e3 GPU=0 bash scripts/_launch_gv2_factorial.sh > runs/v2_tb_gv2_both/launch.log \
#     2>&1 < /dev/null &"
#
# The four arms (codex 01a01b1a: E0 is the causal baseline -- single-GPU B8 like its peers, NOT
# the 4-rank run-1d; grouped_loss normalises per-process, so B2x4 gradients != one B8 loss):
#   e0 = gamma_fk only            -> runs/v2_tb_gv2_e0
#   e1 = + struct_feats           -> runs/v2_tb_gv2_struct
#   e2 = + dir_bias               -> runs/v2_tb_gv2_dir
#   e3 = + both                   -> runs/v2_tb_gv2_both
# Shared protocol (byte-matches the gamma-dose study): TrueBones, 500 ep x 91 steps = 46k steps,
# B8, lr 3e-4, no warmup, gamma_fk 1.0 / fk_warmup 1000, seed 0. All four arms start from the
# SAME function (zero-init knives) + identical shared weights + identical data stream (explicit
# DataLoader generator), so endpoint deltas are attributable to the knives.
set -euo pipefail
cd "$(dirname "$0")/.."

ARM=${ARM:?e0|e1|e2|e3}
GPU=${GPU:?CUDA device index}
PY=${PY:-/scratch/ts1v23/.conda/bin/python3}
EPOCHS=${EPOCHS:-500}

case "$ARM" in
  e0) OUT=runs/v2_tb_gv2_e0;     FLAGS="" ;;
  e1) OUT=runs/v2_tb_gv2_struct; FLAGS="--struct_feats" ;;
  e2) OUT=runs/v2_tb_gv2_dir;    FLAGS="--dir_bias" ;;
  e3) OUT=runs/v2_tb_gv2_both;   FLAGS="--struct_feats --dir_bias" ;;
  *) echo "unknown ARM=$ARM"; exit 1 ;;
esac
mkdir -p "$OUT"

exec 9>".aris/meta/.gv2_${ARM}.lock"
flock -n 9 || { echo "[gv2] arm $ARM already running"; exit 1; }

RESUME=""
[ -f "$OUT/last_model.pt" ] && RESUME="--resume $OUT/last_model.pt"

CUDA_VISIBLE_DEVICES=$GPU $PY scripts/train_v2_incontext.py \
  --out "$OUT" --epochs "$EPOCHS" --lr 3e-4 --batch 8 \
  --gamma_fk 1.0 --fk_warmup_steps 1000 \
  --val_every 5 --ckpt_every 25 --num_workers 4 --seed 0 \
  $FLAGS $RESUME 2>&1 | sed -u "s/^/[$ARM] /"
