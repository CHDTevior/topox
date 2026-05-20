#!/bin/bash
# ============================================================================
# noKslot_clean / scripts/_deploy_train_graph_vae.sh — PPID=1 setsid srun
# launcher for Graph-SALAD VAE training (M1.5 milestone).
#
# Adapted from scripts/_deploy_train.sh — same launcher pattern, but
# targets scripts/train_graph_vae.py and parameterizes over POOL_TYPE
# {dynamic, deterministic, none} for 3-way ablation.
#
# Cross-project rule: NO sbatch / NO scancel — single attempt --overlap
# into an EXISTING allocation provided via $JOBID.
#
# Required env vars:
#   JOBID       Slurm jobid of RUNNING alloc (user-owned)
#   NODE        compute node hostname
#   POOL_TYPE   dynamic | deterministic | none
#
# Optional env vars (defaults appropriate for H100 / A100-80GB):
#   GPUS_PER_TASK   default 1
#   EPOCHS          default 100
#   SAVE_EVERY      default 10
#   LR              default 2e-4
#   BATCH_SIZE      default 8
#   D_MODEL         default 256
#   N_HEADS         default 4 (matches baseline ep399)
#   D_FF            default 512
#   MAX_COARSE      default 64
#   LOCAL_RADIUS    default 8
#   TEMPORAL_STRIDE default 4
#   MAX_FRAMES      default 64
#   MAX_JOINTS      default 160
#   SEED            default 42
#   INIT_CKPT       default "" (no warm-start; pass baseline ep399 path to warm-start)
#   OUT             default runs/m1_5_graph_vae_<POOL_TYPE>_seed<SEED>
#
# Launch example (run from login node, will reparent to init via setsid):
#   JOBID=925435 NODE=swarmh1002 POOL_TYPE=dynamic \
#     ssh swarmh1002 "setsid nohup bash \
#       /scratch/ts1v23/workspace/noKslot_clean/scripts/_deploy_train_graph_vae.sh \
#       > /scratch/ts1v23/workspace/noKslot_clean/logs/deploy_graph_dynamic.out 2>&1 < /dev/null &"
# ============================================================================
set -u

# ---- self-anchor: project root from script location ------------------------
P="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"
cd "$P" || exit 1
mkdir -p logs runs .aris/meta

# ---- required env vars -----------------------------------------------------
: "${JOBID:?ERROR: JOBID env var required (existing RUNNING alloc jobid)}"
: "${NODE:?ERROR: NODE env var required (compute node hostname)}"
: "${POOL_TYPE:?ERROR: POOL_TYPE env var required (dynamic | deterministic | none)}"

case "$POOL_TYPE" in
    dynamic|deterministic|none) : ;;
    *) echo "[deploy_graph] FATAL: POOL_TYPE='$POOL_TYPE' not in {dynamic,deterministic,none}" >&2 ; exit 2 ;;
esac

# ---- optional env vars (defaults) ------------------------------------------
GPUS_PER_TASK="${GPUS_PER_TASK:-1}"
EPOCHS="${EPOCHS:-100}"
SAVE_EVERY="${SAVE_EVERY:-10}"
LR="${LR:-2e-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
D_MODEL="${D_MODEL:-256}"
N_HEADS="${N_HEADS:-4}"
D_FF="${D_FF:-512}"
MAX_COARSE="${MAX_COARSE:-64}"
LOCAL_RADIUS="${LOCAL_RADIUS:-8}"
TEMPORAL_STRIDE="${TEMPORAL_STRIDE:-4}"
MAX_FRAMES="${MAX_FRAMES:-64}"
MAX_JOINTS="${MAX_JOINTS:-160}"
SEED="${SEED:-42}"
INIT_CKPT="${INIT_CKPT:-}"
OUT="${OUT:-runs/m1_5_graph_vae_${POOL_TYPE}_seed${SEED}}"

LOG="$P/logs/deploy_graph_${POOL_TYPE}.log"

# ---- validate alloc is RUNNING and on the named node -----------------------
if ! squeue -j "$JOBID" -h 2>/dev/null | grep -q ' R '; then
    echo "[deploy_graph] FATAL: alloc $JOBID is not RUNNING; NOT sbatch-ing anything" >&2
    exit 2
fi
SQ_NODE=$(squeue -j "$JOBID" -h -o '%N' 2>/dev/null | tr -d '[:space:]')
SQ_STATE=$(squeue -j "$JOBID" -h -o '%T' 2>/dev/null | tr -d '[:space:]')
if [ "$SQ_NODE" != "$NODE" ] || [ "$SQ_STATE" != "RUNNING" ]; then
    echo "[deploy_graph] FATAL: squeue says job $JOBID node='$SQ_NODE' state='$SQ_STATE'; require node==$NODE & state==RUNNING — refusing (no sbatch/scancel)" >&2
    exit 2
fi
echo "[deploy_graph] squeue validated: job $JOBID is RUNNING on $SQ_NODE (== $NODE)"

# ---- single-instance flock (per POOL_TYPE) ---------------------------------
exec 9>".aris/meta/.deploy_graph_${POOL_TYPE}.lock"
flock -n -o 9 || {
    echo "[deploy_graph] another deploy_graph (POOL_TYPE=$POOL_TYPE) already running — exit"
    exit 0
}

# ---- guard: never double-launch THIS training's python ---------------------
if ssh "$NODE" "pgrep -f '[p]ython.*train_graph_vae.py.*$OUT' >/dev/null 2>&1" 2>/dev/null; then
    echo "[deploy_graph] ABORT: a train_graph_vae.py with OUT=$OUT already running on $NODE"
    exit 1
fi

# ---- conda activate environment (graph_salad) ------------------------------
CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -z "$CONDA_BASE" ]; then
    CONDA_BASE="/scratch/ts1v23/.conda"
fi
ACTIVATE="source $CONDA_BASE/etc/profile.d/conda.sh && conda activate graph_salad"

# ---- build training command ------------------------------------------------
INIT_CKPT_ARG=""
if [ -n "$INIT_CKPT" ]; then
    INIT_CKPT_ARG="--init_ckpt $INIT_CKPT"
fi

echo "[deploy_graph] starting clean PPID=1 setsid srun(1 task, --overlap) -> python (single GPU)"
echo "[deploy_graph] alloc=$JOBID@$NODE POOL_TYPE=$POOL_TYPE OUT=$OUT"
echo "[deploy_graph] EPOCHS=$EPOCHS LR=$LR BATCH=$BATCH_SIZE D_MODEL=$D_MODEL N_HEADS=$N_HEADS"
echo "[deploy_graph] MAX_COARSE=$MAX_COARSE LOCAL_RADIUS=$LOCAL_RADIUS STRIDE=$TEMPORAL_STRIDE"

# ---- training: ONE srun task, --gres=gpu:N, --no-kill ----------------------
setsid nohup srun --jobid="$JOBID" -w "$NODE" --overlap --ntasks=1 \
    --gres="gpu:$GPUS_PER_TASK" --no-kill \
    bash -lc \
    "$ACTIVATE && cd $P && PYTHONUNBUFFERED=1 python -u scripts/train_graph_vae.py \
     --pool_type $POOL_TYPE \
     --data_dir data/cs_sparse2full_tgt \
     --epochs $EPOCHS --save_every $SAVE_EVERY --lr $LR --batch_size $BATCH_SIZE \
     --seed $SEED --device cuda \
     --d_model $D_MODEL --n_heads $N_HEADS --d_ff $D_FF \
     --max_coarse $MAX_COARSE --local_radius $LOCAL_RADIUS \
     --temporal_stride $TEMPORAL_STRIDE \
     --max_frames $MAX_FRAMES --max_joints $MAX_JOINTS \
     $INIT_CKPT_ARG \
     --out $OUT --overwrite" \
    > "$LOG" 2>&1 < /dev/null &
disown
echo "[deploy_graph] launched (log: $LOG); watching startup -> ep0"

# ---- wait for ep0 or fail fast ---------------------------------------------
for i in $(seq 1 60); do            # up to ~20min (init + ep0)
    if grep -q "epoch 0 done" "$LOG" 2>/dev/null; then
        echo "[deploy_graph] EP0_REACHED iter $i"
        grep -E "UnifiedMotionDataset|VAE params|gate2 ok|stride-tail|epoch 0 done" "$LOG" | tail -10
        timeout 12 ssh "$NODE" "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -8" 2>/dev/null
        break
    fi
    if grep -qE "GATE.*FAIL|AssertionError|Traceback|RuntimeError" "$LOG" 2>/dev/null \
        && ! grep -q "epoch 0 done" "$LOG" 2>/dev/null; then
        echo "[deploy_graph] ABORT before ep0 at iter $i"
        tail -40 "$LOG"
        exit 3
    fi
    if ! pgrep -f "srun --jobid=${JOBID}.*${NODE}" >/dev/null 2>&1 \
        && ! grep -q "epoch 0 done" "$LOG" 2>/dev/null; then
        echo "[deploy_graph] SRUN_GONE before ep0 at iter $i"
        tail -25 "$LOG"
        exit 4
    fi
    sleep 20
done
if ! grep -q "epoch 0 done" "$LOG" 2>/dev/null; then
    echo "[deploy_graph] EP0_TIMEOUT (20min); inspect: tail -200 $LOG"
    exit 5
fi

echo "[deploy_graph] handing off to background; ep0 logged. Monitor via: tail -f $LOG"
exit 0
