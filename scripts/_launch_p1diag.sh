#!/bin/bash
# scripts/_launch_p1diag.sh
# Per-joint latent (pool_type=none) LONG-CHAIN diagnostic — dragon-wing / long-tail
# reconstruction. Two modes for attribution (user code review 2026-05-30):
#
#   MODE=A : pool_type=none + decoder_mode=coarse_xattn   (attribution control)
#            vs baseline(edge_segment+coarse_xattn) = isolates "NO spatial pool"
#   MODE=B : pool_type=none + decoder_mode=graph_temporal (main diagnostic)
#            vs A = isolates "graph-temporal decoder"; vs baseline = joint effect
#
# baseline = edge_segment + coarse_xattn (separate H200 run 976854, best ep34 val=1.378).
#
# ── Batch/LR convention (train_graph_vae.py:259: --batch_size is PER-GPU) ──
#   per_gpu_batch          = $PER_GPU_BATCH        (smoke-tested, set explicitly)
#   world_size             = $WORLD_SIZE           (= torchrun nproc_per_node)
#   global_batch           = per_gpu_batch × world_size
#   reference_global_batch = 128                   (H200 baseline: 64/gpu × 2)
#   lr (Goyal linear)      = 4e-4 × global_batch / 128
#
# From-scratch only (NO --init_ckpt): pool_type=none has no `unpool.*`, so
# warm-starting from an edge_segment ckpt would throw unexpected-key errors.
#
# Usage (smoke, single GPU, real-size model, ~2ep×4steps to test OOM/peak mem):
#   SMOKE=1 MODE=B PER_GPU_BATCH=48 bash scripts/_launch_p1diag.sh
# Usage (real run, e.g. parallel 2+2 on swarma1001):
#   CUDA_VISIBLE_DEVICES=2,3 MODE=B PER_GPU_BATCH=<smoked> WORLD_SIZE=2 \
#     setsid nohup bash scripts/_launch_p1diag.sh > <log> 2>&1 < /dev/null &
set -u

P=/scratch/ts1v23/workspace/noKslot_clean
cd "$P" || exit 1

MODE="${MODE:?set MODE=A (none+coarse_xattn) or B (none+graph_temporal)}"
PER_GPU_BATCH="${PER_GPU_BATCH:?set PER_GPU_BATCH (per-GPU, smoke-tested)}"
WORLD_SIZE="${WORLD_SIZE:-2}"
SMOKE="${SMOKE:-0}"

case "$MODE" in
  A) DEC=coarse_xattn;   TAG=coarse_p1diagA ;;
  B) DEC=graph_temporal; TAG=gtemporal_p1diagB ;;
  *) echo "[p1diag] bad MODE=$MODE (want A or B)"; exit 2 ;;
esac

GLOBAL=$(( PER_GPU_BATCH * WORLD_SIZE ))
REF_GLOBAL=128
LR=$(awk "BEGIN{printf \"%.3e\", 4e-4 * $GLOBAL / $REF_GLOBAL}")

OUT="runs/m1_l2_anytop13_noneJ144_${TAG}_seed42"

# smoke: single GPU, real-size model (NOT --smoke_tiny_model) to measure true
# graph_temporal B·T·J² memory; real run: WORLD_SIZE GPUs.
NPROC="$WORLD_SIZE"
SMOKE_FLAG=""
if [ "$SMOKE" = 1 ]; then
    NPROC=1
    SMOKE_FLAG="--smoke"
    OUT="${OUT}_smoke"
fi

# Guard: never double-launch the same real run
if [ "$SMOKE" != 1 ] && pgrep -f "train_graph_vae.py.*${TAG}_seed42" >/dev/null 2>&1; then
    echo "[p1diag] ABORT: ${TAG} run already training"; exit 0
fi
mkdir -p "$OUT"

echo "[p1diag] $(date '+%F %T %Z') host=$(hostname) CVD=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[p1diag] MODE=$MODE pool=none decoder=$DEC | per_gpu=$PER_GPU_BATCH world=$WORLD_SIZE global=$GLOBAL ref=$REF_GLOBAL lr=$LR | smoke=$SMOKE nproc=$NPROC"
echo "[p1diag] out=$OUT  from-scratch(no init_ckpt)"

# expandable_segments: defensive vs long-run fragmentation on J=144 per-joint
# path (bz32 fits bare in smoke; DDP long runs fragment). PyTorch's own OOM msg
# recommends it. Cheap insurance.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
echo "[p1diag] PYTORCH_ALLOC_CONF=$PYTORCH_ALLOC_CONF"

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" scripts/train_graph_vae.py \
  --dataset anytop_truebones --feat_mode anytop13 --attn_mode graphormer \
  --decoder_mode "$DEC" --pool_type none \
  --anytop_root "$P/data/anytop_planet_zoo_clean_L2" \
  --val_frac 0.05 --batch_size "$PER_GPU_BATCH" --lr "$LR" --seed 42 --epochs 300 \
  --save_every 5 --periodic_save_every 50 --d_model 512 --n_heads 8 --d_ff 1536 \
  --n_graph_layers 4 --n_enc_temporal_layers 2 --n_cross_layers 3 \
  --n_dec_temporal_layers 2 --n_treeik_layers 3 --max_coarse 144 --local_radius 8 \
  --temporal_stride 4 --max_frames 64 --max_joints 144 --use_name_embed \
  --out "$OUT" --overwrite $SMOKE_FLAG
echo "[p1diag] $(date '+%F %T %Z') torchrun EXITED rc=$?"
