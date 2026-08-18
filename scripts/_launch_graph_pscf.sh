#!/bin/bash
# scripts/_launch_graph_pscf.sh
# INNER per-alloc launcher for the graph_pscf backbone (scripts/train_graph_codeflow.py).
# Mirrors scripts/_launch_graph_vqvae.sh's PROVEN cross-alloc 6-card pattern, but runs
# the graph_pscf rectified-flow training command.
#
# ── Launch modes ──
#   NNODES=1 (default): torchrun --standalone (single-alloc 2-GPU smoke).
#   NNODES=3 (6-card same-node cross-alloc): orchestrator _launch_graph_pscf_6card.sh
#   sets NNODES=3 + NODE_RANK(0/1/2) + MASTER_ADDR=swarmh1002-ib0 + NPROC_PER_NODE=2
#   → STATIC rendezvous, WORLD_SIZE=6. (NOT c10d auto-host: same-node agents'
#   hostname != IB rdzv host, so c10d host election fails — verified on the VQVAE run.)
#
# ── Batch (train_graph_codeflow.py --batch_size is PER-GPU) ──
#   GLOBAL = BATCH_SIZE × NNODES × NPROC_PER_NODE   (16×3×2 = 96 on the 6-card run)
#   dropout is NOT passed → train resolves graph_pscf's 0.05 default.
set -u
P=/scratch/ts1v23/workspace/noKslot_clean
cd "$P" || exit 1
# torchrun's shebang is the conda python3, so it launches train under the right
# interpreter — call torchrun by absolute path (srun's PATH may lack the conda bin).
TORCHRUN=/scratch/ts1v23/.conda/bin/torchrun

BATCH_SIZE="${BATCH_SIZE:?set BATCH_SIZE (per-GPU)}"
LR="${LR:?set LR}"
TOKEN_CACHE="${TOKEN_CACHE:?set TOKEN_CACHE}"
FROZEN_CKPT="${FROZEN_CKPT:?set FROZEN_CKPT}"
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-29505}"
SMOKE="${SMOKE:-0}"
CVD="${CVD:-0,1}"
EPOCHS="${EPOCHS:-600}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
DEPTH_DOUBLE="${DEPTH_DOUBLE:-6}"
DEPTH_SINGLE="${DEPTH_SINGLE:-12}"
MAX_T_LAT="${MAX_T_LAT:-75}"
COND_DROP_PROB="${COND_DROP_PROB:-0.1}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
NUM_WORKERS="${NUM_WORKERS:-6}"
LOG_EVERY="${LOG_EVERY:-50}"
QA_EVERY="${QA_EVERY:-200}"
SAVE_EVERY="${SAVE_EVERY:-10}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"   # >1 preserves global batch/lr on fewer GPUs (per_gpu*world*accum); 1 = unchanged
SEED="${SEED:-42}"
# Text architecture (LLM2Vec run: TEXT_DIM=4096 TEXT_INPUT_NORM=1 USE_SENTENCE_TOKEN=1
# TEXT_SLOT_XATTN=1 GEN_EVAL_CAPTION_CACHE=<sidecar prefix>). Defaults reproduce T5 legacy.
TEXT_DIM="${TEXT_DIM:-768}"
TEXT_INPUT_NORM="${TEXT_INPUT_NORM:-0}"
USE_SENTENCE_TOKEN="${USE_SENTENCE_TOKEN:-0}"
TEXT_SLOT_XATTN="${TEXT_SLOT_XATTN:-0}"
GEN_EVAL_CAPTION_CACHE="${GEN_EVAL_CAPTION_CACHE:-}"
CAPTION_SAMPLING="${CAPTION_SAMPLING:-fixed}"
CAPTION_SIDECAR="${CAPTION_SIDECAR:-}"
PARAMETERIZATION="${PARAMETERIZATION:-}"   # empty = trainer default (v); set x for x-pred
W_DEC_WORLD="${W_DEC_WORLD:-}"             # decoded-geometry loss weights; empty = off
W_DEC_TRAJ="${W_DEC_TRAJ:-}"
W_DEC_SPEED="${W_DEC_SPEED:-}"
DEC_GEOM_T_MIN="${DEC_GEOM_T_MIN:-}"
DEC_GEOM_EVERY="${DEC_GEOM_EVERY:-}"
GEN_EVAL_MANIFEST="${GEN_EVAL_MANIFEST:-}"
PROTOCOL="${PROTOCOL:-legacy}"
HOLDOUT_ART="${HOLDOUT_ART:-}"
HOLDOUT_SHA="${HOLDOUT_SHA:-}"
EMPIRICAL_MAX="${EMPIRICAL_MAX:-0}"     # 0 = full train-set empirical z_q norm (LOCKED real run);
                                       # a positive cap is for the SMOKE only (skip the full scan)
RESUME_CKPT="${RESUME_CKPT:-}"          # FULL resume (model+optimizer+epoch+step)
OUT="${OUT:?set OUT}"
OVERWRITE="${OVERWRITE:-1}"
# Online text->motion gen-eval (opt-in): non-empty GEN_EVAL enables the trainer's
# frozen-evaluator hook. Off by default -> existing behavior byte-unchanged.
GEN_EVAL="${GEN_EVAL:-}"
EVALUATOR_CKPT="${EVALUATOR_CKPT:-}"
GEN_EVAL_EVERY="${GEN_EVAL_EVERY:-50}"
GEN_EVAL_N="${GEN_EVAL_N:-256}"
GEN_EVAL_BATCH="${GEN_EVAL_BATCH:-8}"
# Two-phase human-upsampling curriculum (opt-in): factor 1.0 / start -1 = OFF (unchanged).
# phase2 (VQVAE-style): phase2_factor 1.0 / phase2_start -1 = OFF (single-phase).
HUMAN_UPSAMPLE_FACTOR="${HUMAN_UPSAMPLE_FACTOR:-1.0}"
HUMAN_UPSAMPLE_START_EPOCH="${HUMAN_UPSAMPLE_START_EPOCH:--1}"
HUMAN_UPSAMPLE_PHASE2_FACTOR="${HUMAN_UPSAMPLE_PHASE2_FACTOR:-1.0}"
HUMAN_UPSAMPLE_PHASE2_START_EPOCH="${HUMAN_UPSAMPLE_PHASE2_START_EPOCH:--1}"

GLOBAL=$(( BATCH_SIZE * NNODES * NPROC_PER_NODE * GRAD_ACCUM ))   # EFFECTIVE global batch (incl grad accumulation)
SMOKE_FLAG=""; [ "$SMOKE" = 1 ] && SMOKE_FLAG="--smoke"
OVERWRITE_FLAG=""; [ "$OVERWRITE" = 1 ] && OVERWRITE_FLAG="--overwrite"
GEN_EVAL_ARGS=""
# "0" must DISABLE (codex r4 #1c): the old non-empty test made GEN_EVAL=0 arm the eval.
if [ -n "$GEN_EVAL" ] && [ "$GEN_EVAL" != "0" ]; then
    [ -n "$EVALUATOR_CKPT" ] || { echo "[gpscf] FAIL: GEN_EVAL set but EVALUATOR_CKPT empty"; exit 2; }
    GEN_EVAL_ARGS="--gen_eval --evaluator_ckpt $EVALUATOR_CKPT --gen_eval_every $GEN_EVAL_EVERY --gen_eval_n $GEN_EVAL_N --gen_eval_batch $GEN_EVAL_BATCH"
fi
mkdir -p "$OUT"
# Compute nodes have NO internet: force HF/transformers offline so the online gen-eval's
# DistilBERT(local path) + T5(local HF cache) load from disk instead of hitting HF Hub
# (else "Cannot send a request, client closed" -> gen-eval silently DISABLED). Harmless
# off-path (the backbone trainer itself loads no HF model). Covers manual + watchdog-resume.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="$CVD"

# torchrun mode: standalone (single alloc) vs static rendezvous (cross-alloc).
if [ "$NNODES" -gt 1 ]; then
    [ -z "$MASTER_ADDR" ] && { echo "[gpscf] FAIL: NNODES>1 requires MASTER_ADDR"; exit 2; }
    # same-node cross-cgroup: Slurm isolates P2P/SHM between allocs → disable + force IB.
    export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
    export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib0}"
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
    export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
    export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
    RDZV_ARGS="--nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nproc_per_node=$NPROC_PER_NODE"
else
    RDZV_ARGS="--standalone --nnodes=1 --nproc_per_node=$NPROC_PER_NODE"
fi

echo "[gpscf] $(date '+%F %T %Z') host=$(hostname) CVD=$CVD nnodes=$NNODES nproc=$NPROC_PER_NODE node_rank=$NODE_RANK"
echo "[gpscf] per_gpu=$BATCH_SIZE global=$GLOBAL(=${BATCH_SIZE}x${NNODES}x${NPROC_PER_NODE}x accum${GRAD_ACCUM}) lr=$LR warmup=$WARMUP_STEPS epochs=$EPOCHS smoke=$SMOKE"
echo "[gpscf] master=${MASTER_ADDR:-<standalone>}:$MASTER_PORT nccl_ifname=${NCCL_SOCKET_IFNAME:-<n/a>} cache=$TOKEN_CACHE out=$OUT"
echo "[gpscf] resume=${RESUME_CKPT:-<none>}"

"$TORCHRUN" $RDZV_ARGS scripts/train_graph_codeflow.py \
  --model_variant graph_pscf \
  --token_cache "$TOKEN_CACHE" --frozen_vqvae_ckpt "$FROZEN_CKPT" \
  --code_dim 512 --hidden_size 512 --n_heads 8 --d_ff 2048 \
  --depth_double "$DEPTH_DOUBLE" --depth_single "$DEPTH_SINGLE" --mlp_ratio 4.0 \
  --max_T_lat "$MAX_T_LAT" \
  --batch_size "$BATCH_SIZE" --lr "$LR" \
  --epochs "$EPOCHS" --warmup_steps "$WARMUP_STEPS" --lr_scheduler half_cosine \
  --cond_drop_prob "$COND_DROP_PROB" \
  --amp_dtype "$AMP_DTYPE" --seed "$SEED" --num_workers "$NUM_WORKERS" \
  --empirical_stats_max_clips "$EMPIRICAL_MAX" \
  --log_every "$LOG_EVERY" --qa_every "$QA_EVERY" --save_every "$SAVE_EVERY" \
  --grad_accum "$GRAD_ACCUM" \
  --human_upsample_factor "$HUMAN_UPSAMPLE_FACTOR" --human_upsample_start_epoch "$HUMAN_UPSAMPLE_START_EPOCH" \
  --human_upsample_phase2_factor "$HUMAN_UPSAMPLE_PHASE2_FACTOR" --human_upsample_phase2_start_epoch "$HUMAN_UPSAMPLE_PHASE2_START_EPOCH" \
  --text_dim "$TEXT_DIM" --text_input_norm "$TEXT_INPUT_NORM" \
  --use_sentence_token "$USE_SENTENCE_TOKEN" --text_slot_xattn "$TEXT_SLOT_XATTN" \
  ${GEN_EVAL_CAPTION_CACHE:+--gen_eval_caption_cache "$GEN_EVAL_CAPTION_CACHE"} \
  --caption_sampling "$CAPTION_SAMPLING" \
  ${CAPTION_SIDECAR:+--caption_sidecar "$CAPTION_SIDECAR"} \
  ${PARAMETERIZATION:+--parameterization "$PARAMETERIZATION"} \
  ${W_DEC_WORLD:+--w_dec_world "$W_DEC_WORLD"} \
  ${W_DEC_TRAJ:+--w_dec_traj "$W_DEC_TRAJ"} \
  ${W_DEC_SPEED:+--w_dec_speed "$W_DEC_SPEED"} \
  ${DEC_GEOM_T_MIN:+--dec_geom_t_min "$DEC_GEOM_T_MIN"} \
  ${DEC_GEOM_EVERY:+--dec_geom_every "$DEC_GEOM_EVERY"} \
  ${GEN_EVAL_MANIFEST:+--gen_eval_manifest "$GEN_EVAL_MANIFEST"} \
  --protocol "$PROTOCOL" \
  ${HOLDOUT_ART:+--holdout_artifact "$HOLDOUT_ART"} \
  ${HOLDOUT_SHA:+--holdout_sha "$HOLDOUT_SHA"} \
  $GEN_EVAL_ARGS \
  ${RESUME_CKPT:+--resume "$RESUME_CKPT"} \
  --out "$OUT" $OVERWRITE_FLAG $SMOKE_FLAG
rc=$?
echo "[gpscf] $(date '+%F %T %Z') torchrun EXITED rc=$rc"
exit "$rc"
