#!/bin/bash
# ============================================================================
# noKslot_clean / scripts/_deploy_train_anytop13.sh — PPID=1 setsid srun
# launcher for the M1.7 AnyTop-native 13ch end-to-end Graph-VAE.
#
# Adapted from scripts/_deploy_train_graph_vae.sh — same PPID=1 launcher
# pattern, but targets `--dataset anytop_truebones --feat_mode anytop13` and
# exposes the iter-2 knobs (attn_mode / use_text / augment).
#
# Cross-project rule: NO sbatch / NO scancel — single-attempt --overlap into
# an EXISTING user-owned RUNNING allocation provided via $JOBID.
#
# Required env vars:
#   JOBID       Slurm jobid of a RUNNING alloc (user-owned)
#   NODE        compute node hostname
#   POOL_TYPE   dynamic | deterministic | soft_deterministic | none
#
# Optional env vars (defaults = M1.5R-proven config, AnyTop-native attn):
#   POOL_TAU          required iff POOL_TYPE=soft_deterministic (e.g., 1.0)
#   GPUS_PER_TASK     1
#   NGPU              1  (>1 = DDP via torchrun over NGPU GPUs of the alloc; the
#                     global batch becomes BATCH_SIZE×NGPU, so rescale LR by the
#                     linear scaling rule — lr ∝ global batch — at launch time)
#   EPOCHS            1000
#   SAVE_EVERY        10
#   LR                4e-4
#   BATCH_SIZE        16
#   D_MODEL           384      N_HEADS 8      D_FF 1024
#   MAX_COARSE        64       LOCAL_RADIUS 8 TEMPORAL_STRIDE 4
#   MAX_FRAMES        64       MAX_JOINTS 143 (AnyTop max is 142)
#   SEED              42
#   ATTN_MODE         graphormer   (AnyTop edge-type bias; or 'scalar')
#   DECODER_MODE      coarse_xattn (default) | unpool_identity (legacy) |
#                     graph_temporal (coarse_xattn + AnyTop spatial+temporal layers)
#   N_GRAPH_TEMPORAL_LAYERS 4  (only for DECODER_MODE=graph_temporal)
#   USE_TEXT          0  (1 = inject T5 caption embeddings; needs CAPTION_EMB_CACHE)
#   CAPTION_EMB_CACHE "" (path to scripts/precompute_t5_captions.py .npz output)
#   AUGMENT           0  (1 = AnyTop remove-joints augmentation, train split only)
#   AUGMENT_PROB      0.3      REMOVAL_RATE 0.5
#   USE_NAME_EMBED    1  (encoder name embedding; 0 or 1)
#   W_POS 1.0  W_ROT 1.0  W_VEL 1.0  W_CONTACT 0.1  W_KL 1e-3  W_POOL_AUX 0.5
#   INIT_CKPT         "" (no warm-start)
#   OUT               runs/m1_7_anytop13_<POOL_TYPE>_seed<SEED>
#   ANYTOP_ROOT       "" (override AnyTop processed-data root)
#
# Launch example (run from login node; reparents to init via setsid):
#   JOBID=925437 NODE=swarma1003 POOL_TYPE=dynamic \
#     ssh swarma1003 "setsid nohup bash \
#       /scratch/ts1v23/workspace/noKslot_clean/scripts/_deploy_train_anytop13.sh \
#       > /scratch/ts1v23/workspace/noKslot_clean/logs/deploy_anytop13_dynamic.out 2>&1 < /dev/null &"
# ============================================================================
set -u

# ---- self-anchor: project root from script location ------------------------
P="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"
cd "$P" || exit 1
mkdir -p logs runs .aris/meta

# ---- required env vars -----------------------------------------------------
: "${JOBID:?ERROR: JOBID env var required (existing RUNNING alloc jobid)}"
: "${NODE:?ERROR: NODE env var required (compute node hostname)}"
: "${POOL_TYPE:?ERROR: POOL_TYPE env var required (dynamic|deterministic|soft_deterministic|edge_segment|none)}"

case "$POOL_TYPE" in
    dynamic|deterministic|soft_deterministic|edge_segment|none) : ;;
    *) echo "[deploy_anytop13] FATAL: POOL_TYPE='$POOL_TYPE' invalid" >&2 ; exit 2 ;;
esac

POOL_TAU="${POOL_TAU:-}"
if [ "$POOL_TYPE" = "soft_deterministic" ] && [ -z "$POOL_TAU" ]; then
    echo "[deploy_anytop13] FATAL: POOL_TYPE=soft_deterministic requires POOL_TAU" >&2
    exit 2
elif [ "$POOL_TYPE" != "soft_deterministic" ] && [ -n "$POOL_TAU" ]; then
    # GraphMotionVAE rejects pool_tau outside soft_deterministic — fail here
    # rather than let the launched python job crash minutes later.
    echo "[deploy_anytop13] FATAL: POOL_TAU set but POOL_TYPE=$POOL_TYPE (only valid for soft_deterministic)" >&2
    exit 2
fi

# ---- optional env vars (defaults) ------------------------------------------
GPUS_PER_TASK="${GPUS_PER_TASK:-1}"
NGPU="${NGPU:-1}"
EPOCHS="${EPOCHS:-1000}"
SAVE_EVERY="${SAVE_EVERY:-10}"
LR="${LR:-4e-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
D_MODEL="${D_MODEL:-384}"
N_HEADS="${N_HEADS:-8}"
D_FF="${D_FF:-1024}"
MAX_COARSE="${MAX_COARSE:-64}"
LOCAL_RADIUS="${LOCAL_RADIUS:-8}"
TEMPORAL_STRIDE="${TEMPORAL_STRIDE:-4}"
MAX_FRAMES="${MAX_FRAMES:-64}"
MAX_JOINTS="${MAX_JOINTS:-143}"
SEED="${SEED:-42}"
ATTN_MODE="${ATTN_MODE:-graphormer}"
DECODER_MODE="${DECODER_MODE:-coarse_xattn}"
N_GRAPH_TEMPORAL_LAYERS="${N_GRAPH_TEMPORAL_LAYERS:-4}"
W_POS="${W_POS:-1.0}"
W_ROT="${W_ROT:-1.0}"
W_VEL="${W_VEL:-1.0}"
W_CONTACT="${W_CONTACT:-0.1}"
W_KL="${W_KL:-1e-3}"
W_POOL_AUX="${W_POOL_AUX:-0.5}"
INIT_CKPT="${INIT_CKPT:-}"
ANYTOP_ROOT="${ANYTOP_ROOT:-}"
CAPTION_EMB_CACHE="${CAPTION_EMB_CACHE:-}"
AUGMENT_PROB="${AUGMENT_PROB:-0.3}"
REMOVAL_RATE="${REMOVAL_RATE:-0.5}"
# 0/1 flags — no-colon default so an explicitly-empty value fails validation
USE_TEXT="${USE_TEXT-0}"
AUGMENT="${AUGMENT-0}"
USE_NAME_EMBED="${USE_NAME_EMBED-1}"
for FLAG_NAME in USE_TEXT AUGMENT USE_NAME_EMBED; do
    case "${!FLAG_NAME}" in
        0|1) : ;;
        *) echo "[deploy_anytop13] FATAL: $FLAG_NAME='${!FLAG_NAME}' must be 0 or 1" >&2 ; exit 2 ;;
    esac
done
case "$ATTN_MODE" in
    scalar|graphormer) : ;;
    *) echo "[deploy_anytop13] FATAL: ATTN_MODE='$ATTN_MODE' must be scalar|graphormer" >&2 ; exit 2 ;;
esac
case "$DECODER_MODE" in
    unpool_identity|coarse_xattn|graph_temporal) : ;;
    *) echo "[deploy_anytop13] FATAL: DECODER_MODE='$DECODER_MODE' must be unpool_identity|coarse_xattn|graph_temporal" >&2 ; exit 2 ;;
esac
case "$NGPU" in
    1|2|3|4) : ;;
    *) echo "[deploy_anytop13] FATAL: NGPU='$NGPU' must be 1-4 (GPUs per node)" >&2 ; exit 2 ;;
esac
if [ "$USE_TEXT" = "1" ] && [ -z "$CAPTION_EMB_CACHE" ]; then
    echo "[deploy_anytop13] WARN: USE_TEXT=1 but CAPTION_EMB_CACHE empty — text will be a no-op" >&2
fi

OUT="${OUT:-runs/m1_7_anytop13_${POOL_TYPE}_seed${SEED}}"
# Key LOG + flock on the run's OUT basename (RUN_TAG), NOT POOL_TYPE — so two
# same-pool runs (e.g. an A/B over decoder_mode) get separate deploy logs and
# locks and never cross-contaminate each other's ep0-watch.
RUN_TAG="$(basename "$OUT")"
LOG="$P/logs/deploy_${RUN_TAG}.log"

# ---- validate alloc is RUNNING and on the named node -----------------------
if ! squeue -j "$JOBID" -h 2>/dev/null | grep -q ' R '; then
    echo "[deploy_anytop13] FATAL: alloc $JOBID not RUNNING; NOT sbatch-ing anything" >&2
    exit 2
fi
SQ_NODE=$(squeue -j "$JOBID" -h -o '%N' 2>/dev/null | tr -d '[:space:]')
SQ_STATE=$(squeue -j "$JOBID" -h -o '%T' 2>/dev/null | tr -d '[:space:]')
if [ "$SQ_NODE" != "$NODE" ] || [ "$SQ_STATE" != "RUNNING" ]; then
    echo "[deploy_anytop13] FATAL: job $JOBID node='$SQ_NODE' state='$SQ_STATE'; require node==$NODE & RUNNING — refusing (no sbatch/scancel)" >&2
    exit 2
fi
# Owner check: only ever inject into the current user's own alloc — never
# touch another user's / another project's job (cross-project rule).
SQ_USER=$(squeue -j "$JOBID" -h -o '%u' 2>/dev/null | tr -d '[:space:]')
if [ "$SQ_USER" != "$USER" ]; then
    echo "[deploy_anytop13] FATAL: job $JOBID owned by '$SQ_USER' != \$USER ($USER) — refusing" >&2
    exit 2
fi
echo "[deploy_anytop13] squeue validated: job $JOBID RUNNING on $SQ_NODE (owner $SQ_USER)"

# ---- single-instance flock (per RUN_TAG) -----------------------------------
exec 9>".aris/meta/.deploy_${RUN_TAG}.lock"
flock -n -o 9 || {
    echo "[deploy_anytop13] another deploy (RUN_TAG=$RUN_TAG) already running — exit"
    exit 0
}

# ---- guard: never double-launch THIS training's python ---------------------
if ssh "$NODE" "pgrep -f '[p]ython.*train_graph_vae.py.*$OUT' >/dev/null 2>&1" 2>/dev/null; then
    echo "[deploy_anytop13] ABORT: a train_graph_vae.py with OUT=$OUT already running on $NODE"
    exit 1
fi

# ---- conda activate (graph_salad) ------------------------------------------
CONDA_BASE=$(conda info --base 2>/dev/null)
[ -z "$CONDA_BASE" ] && CONDA_BASE="/scratch/ts1v23/.conda"
ACTIVATE="source $CONDA_BASE/etc/profile.d/conda.sh && conda activate graph_salad"

# ---- build optional args ---------------------------------------------------
INIT_CKPT_ARG="" ; [ -n "$INIT_CKPT" ] && INIT_CKPT_ARG="--init_ckpt $INIT_CKPT"
POOL_TAU_ARG=""  ; [ -n "$POOL_TAU" ]  && POOL_TAU_ARG="--pool_tau $POOL_TAU"
ANYTOP_ROOT_ARG="" ; [ -n "$ANYTOP_ROOT" ] && ANYTOP_ROOT_ARG="--anytop_root $ANYTOP_ROOT"
CAPTION_ARG="" ; [ -n "$CAPTION_EMB_CACHE" ] && CAPTION_ARG="--caption_emb_cache $CAPTION_EMB_CACHE"

echo "[deploy_anytop13] PPID=1 setsid srun(1 task, --overlap) -> train_graph_vae.py"
echo "[deploy_anytop13] alloc=$JOBID@$NODE POOL_TYPE=$POOL_TYPE ATTN_MODE=$ATTN_MODE DECODER_MODE=$DECODER_MODE OUT=$OUT"
echo "[deploy_anytop13] EPOCHS=$EPOCHS LR=$LR BATCH=$BATCH_SIZE NGPU=$NGPU USE_TEXT=$USE_TEXT AUGMENT=$AUGMENT"

# ---- training: ONE srun task, --overlap, --no-kill -------------------------
# NGPU==1 -> plain `python -u` (single GPU, unchanged). NGPU>1 -> `torchrun`
# spawns NGPU procs INSIDE the one srun task for DDP; the task gets NGPU GPUs.
CUDA_PIN=""
DDP_ENV=""
if [ "$NGPU" -gt 1 ]; then
    LAUNCHER="torchrun --standalone --nnodes=1 --nproc_per_node=$NGPU"
    GRES_ARG="--gres=gpu:$NGPU"
    DDP_ENV="TORCH_NCCL_ASYNC_ERROR_HANDLING=1 "
    if [ -n "${CUDA_VISIBLE_DEVICES_OVERRIDE:-}" ]; then
        echo "[deploy_anytop13] WARN: CUDA_VISIBLE_DEVICES_OVERRIDE ignored for NGPU>1 (DDP)" >&2
    fi
else
    LAUNCHER="python -u"
    GRES_ARG="--gres=gpu:$GPUS_PER_TASK"
    if [ -n "${CUDA_VISIBLE_DEVICES_OVERRIDE:-}" ]; then
        # Manual GPU pin: DROP --gres (srun --gres forces mask {0} for the first
        # injected task, blanking a CUDA_VISIBLE_DEVICES=1 override). Root-cause
        # fix verified in M1.5R.
        CUDA_PIN="export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_OVERRIDE} && "
        GRES_ARG=""
    fi
fi
setsid nohup srun --jobid="$JOBID" -w "$NODE" --overlap --ntasks=1 \
    $GRES_ARG --no-kill \
    bash -lc \
    "$ACTIVATE && cd $P && ${CUDA_PIN}${DDP_ENV}PYTHONUNBUFFERED=1 $LAUNCHER scripts/train_graph_vae.py \
     --dataset anytop_truebones --feat_mode anytop13 --attn_mode $ATTN_MODE \
     --decoder_mode $DECODER_MODE \
     --n_graph_temporal_layers $N_GRAPH_TEMPORAL_LAYERS \
     --pool_type $POOL_TYPE \
     --epochs $EPOCHS --save_every $SAVE_EVERY --lr $LR --batch_size $BATCH_SIZE \
     --seed $SEED --device cuda \
     --d_model $D_MODEL --n_heads $N_HEADS --d_ff $D_FF \
     --max_coarse $MAX_COARSE --local_radius $LOCAL_RADIUS \
     --temporal_stride $TEMPORAL_STRIDE \
     --max_frames $MAX_FRAMES --max_joints $MAX_JOINTS \
     --w_pos $W_POS --w_rot $W_ROT --w_vel $W_VEL --w_contact $W_CONTACT \
     --w_kl $W_KL --w_pool_aux $W_POOL_AUX \
     --augment_prob $AUGMENT_PROB --removal_rate $REMOVAL_RATE \
     $([ \"$USE_NAME_EMBED\" = \"1\" ] && echo \"--use_name_embed\") \
     $([ \"$USE_TEXT\" = \"1\" ] && echo \"--use_text\") \
     $([ \"$AUGMENT\" = \"1\" ] && echo \"--augment\") \
     $INIT_CKPT_ARG $POOL_TAU_ARG $ANYTOP_ROOT_ARG $CAPTION_ARG \
     --out $OUT --overwrite" \
    > "$LOG" 2>&1 < /dev/null &
disown
echo "[deploy_anytop13] launched (log: $LOG); watching startup -> ep0"

# ---- wait for ep0 or fail fast ---------------------------------------------
for i in $(seq 1 60); do            # up to ~20min (init + ep0)
    if grep -q "epoch 0 done" "$LOG" 2>/dev/null; then
        echo "[deploy_anytop13] EP0_REACHED iter $i"
        grep -E "AnyTopDataset|VAE params|gate2 ok|stride-tail|epoch 0 done" "$LOG" | tail -10
        timeout 12 ssh "$NODE" "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -8" 2>/dev/null
        break
    fi
    if grep -qE "GATE.*FAIL|AssertionError|Traceback|RuntimeError|ARGS FAIL" "$LOG" 2>/dev/null \
        && ! grep -q "epoch 0 done" "$LOG" 2>/dev/null; then
        echo "[deploy_anytop13] ABORT before ep0 at iter $i"
        tail -40 "$LOG"
        exit 3
    fi
    if ! pgrep -f "srun --jobid=${JOBID}.*${NODE}" >/dev/null 2>&1 \
        && ! grep -q "epoch 0 done" "$LOG" 2>/dev/null; then
        echo "[deploy_anytop13] SRUN_GONE before ep0 at iter $i"
        tail -25 "$LOG"
        exit 4
    fi
    sleep 20
done
if ! grep -q "epoch 0 done" "$LOG" 2>/dev/null; then
    echo "[deploy_anytop13] EP0_TIMEOUT (20min); inspect: tail -200 $LOG"
    exit 5
fi
echo "[deploy_anytop13] handing off; ep0 logged. Monitor: tail -f $LOG"
exit 0
