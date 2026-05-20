#!/bin/bash
# ============================================================================
# noKslot_clean / scripts/_deploy_train.sh — PPID=1 setsid srun launcher for
# the noKslot reproducible baseline training. Parametric template (no
# hardcoded alloc / node / paths) — user supplies JOBID + NODE via env.
#
# Adapted from source motion_representation_study/scripts/
# _deploy_noKslot_diag_a1004.sh. K-slot-specific safety invariants (hardcoded
# DECISIVE_JOBID/DECISIVE_NODE exclusion, JOBID/NODE literal-readonly pin,
# cs_sparse2full source-of-truth assertion) are removed since this clean
# repo has no K-slot/decisive concept — the user is responsible for picking
# a free alloc/node that does not conflict with other jobs.
#
# PRESERVED safety invariants:
#   * NO sbatch / NO scancel — single attempt --overlap into an EXISTING
#     allocation provided via $JOBID.
#   * flock single-instance (idempotent re-launch is a no-op).
#   * double-launch guard via pgrep on the compute node.
#   * PPID=1 setsid srun → python (survives the launching ssh disconnect).
#   * preflight is internal to scripts/train.py (write_preflight_manifest_
#     nokslot + assert_name_policy + assert_no_raw_rotation_supervision +
#     IK coverage + noK runtime same-topo bitwise check); a python
#     preflight failure aborts before the first training step.
#   * auto-chained eval.py + animate.py on DONE (visual-QA-primacy gate).
#
# REQUIRED env vars (no defaults — fail-fast):
#   JOBID    Slurm jobid of an EXISTING RUNNING allocation owned by user
#   NODE     compute node name (must match the alloc's host)
#
# OPTIONAL env vars (sensible defaults; override as needed):
#   GPUS_PER_TASK   gres count for srun (default 1 for single-GPU; pass 4
#                   + run torchrun yourself if you want DDP — this
#                   launcher does single-task per srun, no torchrun)
#   EPOCHS          default 400 (matches noKslot diagnostic predeclared)
#   LR              default 2e-4
#   BATCH_SIZE      default 8 (per-GPU)
#   SEED            default 42
#   OUT             output run dir (default runs/noKslot_baseline)
#   SPECIES         eval/animate species filter (default Bat,Crab,Horse)
#
# Launch (from login node, will reparent to init via setsid):
#   JOBID=<job> NODE=<node> ssh <node> "setsid nohup bash \\
#       $(pwd)/scripts/_deploy_train.sh \\
#       > $(pwd)/logs/deploy_train.out 2>&1 < /dev/null &"
# ============================================================================
set -u

# ---- self-anchor: project root from script location ------------------------
P="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"
cd "$P" || exit 1
mkdir -p logs runs .aris/meta

# ---- required env vars -----------------------------------------------------
: "${JOBID:?ERROR: JOBID env var required (existing RUNNING alloc jobid)}"
: "${NODE:?ERROR: NODE env var required (compute node hostname)}"

# ---- optional env vars (defaults) ------------------------------------------
GPUS_PER_TASK="${GPUS_PER_TASK:-1}"
EPOCHS="${EPOCHS:-400}"
LR="${LR:-2e-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-42}"
OUT="${OUT:-runs/noKslot_baseline}"
SPECIES="${SPECIES:-Bat,Crab,Horse}"
MAX_FRAMES=196
MAX_JOINTS=160

LOG="$P/logs/deploy_train.log"
EVAL_LOG="$P/logs/deploy_train_eval.log"

# ---- validate alloc is RUNNING and on the named node -----------------------
if ! squeue -j "$JOBID" -h 2>/dev/null | grep -q ' R '; then
    echo "[deploy_train] FATAL: alloc $JOBID is not RUNNING; NOT sbatch-ing anything" >&2
    exit 2
fi
SQ_NODE=$(squeue -j "$JOBID" -h -o '%N' 2>/dev/null | tr -d '[:space:]')
SQ_STATE=$(squeue -j "$JOBID" -h -o '%T' 2>/dev/null | tr -d '[:space:]')
if [ "$SQ_NODE" != "$NODE" ] || [ "$SQ_STATE" != "RUNNING" ]; then
    echo "[deploy_train] FATAL: squeue says job $JOBID node='$SQ_NODE' state='$SQ_STATE'; require node==$NODE & state==RUNNING — refusing (no sbatch/scancel)" >&2
    exit 2
fi
echo "[deploy_train] squeue validated: job $JOBID is RUNNING on $SQ_NODE (== $NODE)"

# ---- single-instance flock -------------------------------------------------
exec 9>.aris/meta/.deploy_train.lock
flock -n -o 9 || {
    echo "[deploy_train] another deploy_train already running — exit"
    exit 0
}

# ---- guard: never double-launch THIS training's python ---------------------
if ssh "$NODE" "pgrep -f '[p]ython.*scripts/train.py.*$OUT' >/dev/null 2>&1" 2>/dev/null; then
    echo "[deploy_train] ABORT: a real train.py with OUT=$OUT already running on $NODE"
    exit 1
fi

echo "[deploy_train] starting clean PPID=1 setsid srun(1 task, --overlap) -> python (single GPU)"
echo "[deploy_train] alloc=$JOBID@$NODE OUT=$OUT EPOCHS=$EPOCHS LR=$LR BATCH=$BATCH_SIZE SEED=$SEED"

# ---- training: ONE srun task, --gres=gpu:N, --no-kill (flaky step shouldn't
# nuke the borrowed alloc). Defaults to noKslot reproducible config.
setsid nohup srun --jobid="$JOBID" -w "$NODE" --overlap --ntasks=1 \
    --gres="gpu:$GPUS_PER_TASK" --no-kill \
    bash -lc \
    "cd $P && PYTHONUNBUFFERED=1 python -u scripts/train.py \
     --src_dir data/cs_sparse2full_tgt \
     --tgt_dir data/cs_sparse2full_tgt \
     --ik_dir data/cs_sparse2full_ik_rot \
     --init_ckpt runs/L6_anchor_h100_seed42/best_model.pt \
     --epochs $EPOCHS --save_every 25 --lr $LR --batch_size $BATCH_SIZE \
     --max_frames $MAX_FRAMES --max_joints $MAX_JOINTS --seed $SEED \
     --w_rot_ik 0.1 --w_acc 0.01 --w_vel_consistency 0.5 \
     --freeze_name_embed 1 --out $OUT" \
    > "$LOG" 2>&1 < /dev/null &
disown
echo "[deploy_train] launched (log: $LOG); watching startup -> ep0"

# ---- wait for ep0 or fail fast ---------------------------------------------
for i in $(seq 1 90); do            # up to ~30min (preflight + init + ep0)
    if grep -q "ep0 train=" "$LOG" 2>/dev/null; then
        echo "[deploy_train] EP0_REACHED iter $i"
        grep -E "UnifiedMotionDataset|paired |IK-retained|split_manifest|leakage|PREFLIGHT|raw-rotation|fine-tune init|NOKSLOT|ep0 train=" "$LOG" | tail -16
        timeout 12 ssh "$NODE" "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -8" 2>/dev/null
        break
    fi
    if grep -qE "PREFLIGHT.*(ABORT|FAIL)|AssertionError|Traceback|NaN/Inf loss" "$LOG" 2>/dev/null \
        && ! grep -q "ep0 train=" "$LOG" 2>/dev/null; then
        echo "[deploy_train] PREFLIGHT/ABORT before ep0 at iter $i"
        tail -40 "$LOG"
        exit 3
    fi
    if ! pgrep -f "srun --jobid=${JOBID}.*${NODE}" >/dev/null 2>&1 \
        && ! grep -q "ep0 train=" "$LOG" 2>/dev/null; then
        echo "[deploy_train] SRUN_GONE before ep0 at iter $i"
        tail -25 "$LOG"
        exit 4
    fi
    sleep 20
done
if ! grep -q "ep0 train=" "$LOG" 2>/dev/null; then
    echo "[deploy_train] TIMEOUT ~30min no ep0"
    tail -25 "$LOG"
    exit 5
fi

# ============================================================================
# AUTO-CHAINED forced eval + GT-vs-pred visual QA after training DONE.
# Uses scripts/eval.py which itself auto-chains scripts/animate.py for
# multi-frame gif + dual-view contact-sheet (visual-QA-primacy hard gate).
# Runs on a verified-running GPU within the same alloc (--overlap).
# ============================================================================
EVAL_OUT="$OUT/eval_out"
echo "[deploy_train] training launched; will auto-eval on DONE (eval+animate auto-chain)"
for i in $(seq 1 100000); do
    if grep -aqE "DONE head=topofk_treeik([[:space:]]|$)" "$LOG" 2>/dev/null; then
        echo "[deploy_train] training DONE -> auto-chained eval + animate"
        mkdir -p "$EVAL_OUT"
        timeout 7200 srun --jobid="$JOBID" -w "$NODE" --overlap --ntasks=1 \
            --gres="gpu:1" --no-kill \
            bash -lc \
            "cd $P && PYTHONUNBUFFERED=1 python -u scripts/eval.py \
                --ckpt $OUT/last_model.pt \
                --src_dir data/cs_sparse2full_tgt \
                --tgt_dir data/cs_sparse2full_tgt \
                --ik_dir data/cs_sparse2full_ik_rot \
                --split val --species $SPECIES \
                --max_frames $MAX_FRAMES --max_joints $MAX_JOINTS \
                --out $EVAL_OUT --device cuda" \
            > "$EVAL_LOG" 2>&1
        EVAL_RC=$?
        echo "[deploy_train] eval+animate done rc=$EVAL_RC -> $EVAL_OUT/gate_verdict.json"
        tail -8 "$EVAL_LOG" 2>/dev/null
        exit $EVAL_RC
    fi
    if grep -aqE 'NaN/Inf loss|Traceback|CUDA out of memory|PREFLIGHT.*(ABORT|FAIL)' "$LOG" 2>/dev/null; then
        echo "[deploy_train] training FAILED (see $LOG)"
        tail -25 "$LOG"
        exit 6
    fi
    if ! ssh "$NODE" "pgrep -f '[p]ython.*scripts/train.py.*$OUT' >/dev/null 2>&1" 2>/dev/null \
        && ! grep -aqE "DONE head=topofk_treeik([[:space:]]|$)" "$LOG" 2>/dev/null; then
        echo "[deploy_train] train python vanished before DONE (see $LOG)"
        tail -25 "$LOG"
        exit 7
    fi
    sleep 300
done
