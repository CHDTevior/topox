#!/bin/bash
# Same-NODE cross-ALLOC 4-card H100 DDP orchestrator for the AnyTop T2M evaluator
# (scripts/train_anytop_t2m_evaluator.py). User-approved topology (2026-06-16):
# 4 cards H100, global batch 128, lr 1e-4.
#
# swarmh1002: alloc 976839 (2xH100, node_rank 0/master) + 976840 (2xH100, node_rank 1)
#   -> WORLD_SIZE=4 via torchrun STATIC rendezvous over IB (ib0/mlx5_0, 10.6.15.69).
# Both allocs are MINE and IDLE (the n1024/n2048 VQVAE ablation that used to occupy
# them was intentionally stopped 2026-06-15; verified 0% util / 0 MiB on all 4 GPUs
# before writing this). Not another project's cards.
#
# ── Why the NCCL config is the OPPOSITE of the cross-NODE backbone template ──
#   _launch_graph_pscf_2node_h200.sh keeps NVLink P2P/SHM ON (each node = ONE 2-GPU
#   cgroup, intra-node NVLink). HERE the two allocs are TWO Slurm cgroups on ONE
#   physical node, so GPU P2P/SHM is BLOCKED across the cgroup boundary -> MUST
#   disable P2P/SHM and force ALL 4 ranks over IB. (cross-alloc recipe item #2.)
#
# ── PLUMBING SMOKE (verify rendezvous + NCCL via NET/IB + WORLD_SIZE=4, ~6 steps) ──
#   SMOKE=1 OUT=/tmp/eval_crossalloc_smoke NCCL_DEBUG=INFO \
#     bash scripts/_launch_anytop_t2m_evaluator_crossalloc.sh 2>&1 | tee scripts/_smoke_eval_crossalloc.log
#   (confirm: "via NET/IB" mlx5_0/ib0, WORLD_SIZE=4, no-OOM at the real per-rank batch)
#
# ── REAL run (DURABLE) — run ON swarmh1002, PPID=1 survives ssh drop ──
#   ssh swarmh1002 "cd /scratch/ts1v23/workspace/noKslot_clean && \
#     OUT=runs/anytop_t2m_evaluator_distilbert_coemb512_gb128_lr1e-4_seed42 setsid nohup \
#     bash scripts/_launch_anytop_t2m_evaluator_crossalloc.sh \
#     > scripts/_train_eval_crossalloc.log 2>&1 </dev/null &"
set -uo pipefail
P=/scratch/ts1v23/workspace/noKslot_clean
cd "$P" || exit 1
PY=/scratch/ts1v23/.conda/bin/python

JOB_A="${JOB_A:-976839}"                 # node_rank 0 (master, runs TCPStore)
JOB_B="${JOB_B:-976840}"                 # node_rank 1
NODE="${NODE:-swarmh1002}"               # both allocs are on this single physical node
RDZV_HOST="${RDZV_HOST:-10.6.15.69}"     # swarmh1002 ib0 IP (IP, not hostname; DNS-alias safety)
RDZV_PORT="${RDZV_PORT:-29533}"          # distinct from backbone 29507 / 6card 29505
[[ "$RDZV_PORT" =~ ^[0-9]+$ ]] || { echo "[eval-xalloc] ABORT: RDZV_PORT='$RDZV_PORT' not numeric"; exit 64; }
NCCL_IFACE="${NCCL_SOCKET_IFNAME:-ib0}"  # swarmh1002 active IB iface (verified ib0 UP, mlx5_0)
NCCL_HCA="${NCCL_IB_HCA:-mlx5_0}"
SMOKE="${SMOKE:-0}"

# ============================ HYPERPARAMETERS (review) ============================
DATA="${DATA:-data/animo4d_anytop_clean_L4_safe_plus_truebones}"
PER_RANK="${PER_RANK:-32}"               # per-rank batch -> global = 4 x 32 = 128
LR="${LR:-1e-4}"                         # user-chosen @ global 128 (Goyal anchor)
EPOCHS="${EPOCHS:-100}"                  # ~72k train motions / gb128 ~= 560 steps/epoch
WARMUP="${WARMUP:-2000}"                 # linear warmup steps, then cosine decay
WD="${WD:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
NUM_FRAMES="${NUM_FRAMES:-300}"
MAX_JOINTS="${MAX_JOINTS:-144}"
COEMB="${COEMB:-512}"
N_HEADS="${N_HEADS:-8}"
D_FF="${D_FF:-2048}"
N_GRAPH="${N_GRAPH:-6}"
N_TEMPORAL="${N_TEMPORAL:-4}"
MOTION_FEAT_DIM="${MOTION_FEAT_DIM:-12}"  # 12 = contact-free (drop ch12) clean motion-semantics tower; 13 = full AnyTop
DROPOUT="${DROPOUT:-0.1}"
TEMPERATURE="${TEMPERATURE:-0.07}"
NUM_WORKERS="${NUM_WORKERS:-6}"
VAL_EVERY="${VAL_EVERY:-5}"
VAL_BATCH="${VAL_BATCH:-32}"
VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-0}"  # 0 = FULL val (all ~119 batches) for a stable best_model.pt
SAVE_EVERY="${SAVE_EVERY:-5}"
LOG_EVERY="${LOG_EVERY:-50}"
SEED="${SEED:-42}"
OUT="${OUT:?set OUT (use /tmp/eval_crossalloc_smoke for smoke, runs/... for real)}"
# ==================================================================================

# SMOKE -> short plumbing run: a handful of steps, write to /tmp, never touch val/ckpt.
MAX_STEPS="${MAX_STEPS:-0}"
if [ "$SMOKE" = "1" ]; then
    [ "$MAX_STEPS" = "0" ] && MAX_STEPS=6  # honor an explicit MAX_STEPS override, else 6
    VAL_EVERY=100000                      # max_steps stops mid-epoch-0 anyway; belt-and-braces
    SAVE_EVERY=100000
fi

GLOBAL=$(( PER_RANK * 4 ))

# Single-instance lock (no inner pgrep guard for the cross-alloc case; pgrep would
# also match the peer alloc's rank on this shared node and false-ABORT).
mkdir -p .aris/meta
exec 9>".aris/meta/.eval_crossalloc.lock"
flock -n 9 || { echo "[eval-xalloc] ABORT: already running"; exit 75; }
echo $$ > ".aris/meta/.eval_crossalloc_orch.pid" 2>/dev/null || true

# Offline HF + NCCL env. SAME-NODE cross-CGROUP -> P2P/SHM OFF, force IB.
COMMON_ENV="HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 CUDA_VISIBLE_DEVICES=0,1 \
NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=0 \
NCCL_SOCKET_IFNAME=$NCCL_IFACE NCCL_IB_HCA=$NCCL_HCA"
[ -n "${NCCL_DEBUG:-}" ] && COMMON_ENV="$COMMON_ENV NCCL_DEBUG=$NCCL_DEBUG"

echo "[eval-xalloc] $(date '+%F %T %Z') same-NODE 4-card H100 DDP: $JOB_A(r0)+$JOB_B(r1) on $NODE via $RDZV_HOST:$RDZV_PORT smoke=$SMOKE"
echo "[eval-xalloc] global=$GLOBAL (4xbs$PER_RANK) lr=$LR warmup=$WARMUP epochs=$EPOCHS frames=$NUM_FRAMES motion_feat_dim=$MOTION_FEAT_DIM max_steps=$MAX_STEPS out=$OUT"
echo "[eval-xalloc] NCCL P2P/SHM OFF, force IB $NCCL_IFACE/$NCCL_HCA (cross-cgroup on one node)"

TRAIN_ARGS="\
  --manifest $DATA/eval_splits/train_main.json \
  --data_root $DATA \
  --split train --view full \
  --text_tower distilbert \
  --distilbert_path checkpoints/text_encoders/distilbert-base-uncased \
  --num_frames $NUM_FRAMES --max_joints $MAX_JOINTS \
  --coemb_dim $COEMB --n_heads $N_HEADS --d_ff $D_FF \
  --n_graph_layers $N_GRAPH --n_temporal_layers $N_TEMPORAL --dropout $DROPOUT \
  --motion_feat_dim $MOTION_FEAT_DIM \
  --temperature $TEMPERATURE \
  --batch_size $PER_RANK --lr $LR --weight_decay $WD \
  --warmup_steps $WARMUP --grad_clip $GRAD_CLIP --epochs $EPOCHS \
  --bf16 --num_workers $NUM_WORKERS \
  --val_manifest $DATA/eval_splits/val_all.json \
  --val_every $VAL_EVERY --val_batch_size $VAL_BATCH --val_max_batches $VAL_MAX_BATCHES \
  --out $OUT --save_every $SAVE_EVERY --log_every $LOG_EVERY --max_steps $MAX_STEPS"

# ---- PREFLIGHT: both jobs RUNNING on the expected node, rendezvous port free ----
# Fail loud BEFORE spawning rather than burning a 60s torchrun rendezvous timeout.
preflight() {
    local j st node
    for j in "$JOB_A" "$JOB_B"; do
        st=$(squeue -h -j "$j" -o "%T" 2>/dev/null | head -1)
        node=$(squeue -h -j "$j" -o "%N" 2>/dev/null | head -1)
        [ "$st" = "RUNNING" ] || { echo "[eval-xalloc] PREFLIGHT FAIL: job $j state='${st:-<none>}' (need RUNNING)"; exit 70; }
        [ "$node" = "$NODE" ] || { echo "[eval-xalloc] PREFLIGHT FAIL: job $j on '${node:-<none>}' != expected NODE='$NODE'"; exit 70; }
    done
    # The TCPStore binds $RDZV_PORT on $NODE; probe the node's listen sockets via the alloc.
    local pf
    pf=$(timeout 25 srun --jobid="$JOB_A" --overlap --nodes=1 --ntasks=1 --cpus-per-task=1 \
           bash -c 'if command -v ss >/dev/null 2>&1; then ss -Hltn 2>/dev/null | grep -qE ":'"$RDZV_PORT"'[[:space:]]" && echo INUSE || echo FREE; else echo NOSS; fi' \
           2>/dev/null | grep -Eo "INUSE|FREE|NOSS" | tail -1)
    case "$pf" in
        INUSE) echo "[eval-xalloc] PREFLIGHT FAIL: rendezvous port $RDZV_PORT already LISTENing on $NODE (pick another RDZV_PORT)"; exit 70 ;;
        FREE)  echo "[eval-xalloc] preflight OK: $JOB_A & $JOB_B RUNNING on $NODE, port $RDZV_PORT free, OUT=$OUT" ;;
        NOSS)  echo "[eval-xalloc] preflight: 'ss' unavailable on $NODE -> port check skipped (jobs+node OK), OUT=$OUT" ;;
        *)     echo "[eval-xalloc] preflight WARN: port probe inconclusive ('${pf:-empty}'), proceeding (rendezvous fails loud if taken); jobs+node OK, OUT=$OUT" ;;
    esac
}
preflight

# Per-rank log prefixing via NAMED FIFOs so the sed prefixer runs as a SEPARATELY
# TRACKED background process. This gives us BOTH: (a) $! = srun's OWN pid (the trap can
# reach it), and (b) a waitable prefixer we DRAIN before exit. A bare process-sub sed
# (`srun > >(sed) &`) is NOT waited on -> tail log lines can be lost/reordered past the
# final EXITED line (codex-flagged regression vs the old `srun | sed` pipeline, where
# bash implicitly waited for sed). srun writes the fifo; when srun exits its write end
# closes -> sed gets EOF, flushes, and exits, so `wait $SED_*` drains it.
FIFO_DIR=$(mktemp -d "${TMPDIR:-/tmp}/eval_xalloc.XXXXXX") || { echo "[eval-xalloc] ABORT: mktemp failed"; exit 1; }
trap 'rm -rf "$FIFO_DIR" 2>/dev/null' EXIT

# Kill both child srun steps if the orchestrator is SIGNALLED (kill/Ctrl-C), so the
# orchestrator dying never leaves orphaned ranks holding the 4 GPUs. SIGTERM to the
# srun CLIENT cancels its Slurm step -> torchrun + ranks torn down. We do NOT scancel
# (iron rule); we only signal our own srun client pids. PID_A/PID_B init "" so the trap
# is set -u safe even if a signal arrives before they are assigned.
PID_A=""; PID_B=""
cleanup() {
    trap - INT TERM
    # Kill ALL still-running direct background children (srun clients + sed prefixers),
    # read from the LIVE job table -> covers a signal landing in the window after a
    # fifo-reader sed is started but before its srun writer is tracked (an untracked
    # writerless sed would otherwise hang the bare `wait`). codex-flagged. We only signal
    # our OWN children: SIGTERM to an srun client cancels its Slurm step -> never scancel.
    local pids; mapfile -t pids < <(jobs -pr)
    echo "[eval-xalloc] $(date '+%F %T %Z') SIGNAL -> terminating ${#pids[@]} child proc(s): ${pids[*]:-<none>}"
    [ "${#pids[@]}" -gt 0 ] && kill "${pids[@]}" 2>/dev/null
    wait 2>/dev/null            # reap srun + the sed prefixers (which EOF when srun closes)
    exit 130                    # EXIT trap then removes $FIFO_DIR
}
trap cleanup INT TERM

# One torchrun group per alloc; static rendezvous joins them into WORLD_SIZE=4.
# Start the sed prefixer (reader) on the fifo FIRST so srun's open-for-write connects.
run_alloc() {   # sets RUN_PID (srun pid) and SED_PID (prefixer pid)
    local job="$1" noderank="$2" fifo="$3"
    stdbuf -oL sed "s/^/[r$noderank:$job] /" < "$fifo" & SED_PID=$!
    srun --jobid="$job" --overlap --nodes=1 --ntasks=1 \
      --gres=gpu:2 --cpus-per-task=16 --no-kill \
      bash -c "cd '$P' && $COMMON_ENV \
        $PY -m torch.distributed.run \
          --nnodes=2 --nproc_per_node=2 --node_rank=$noderank \
          --master_addr=$RDZV_HOST --master_port=$RDZV_PORT \
          scripts/train_anytop_t2m_evaluator.py $TRAIN_ARGS" \
      > "$fifo" 2>&1 &
    RUN_PID=$!
}
FIFO_A="$FIFO_DIR/r0"; FIFO_B="$FIFO_DIR/r1"
mkfifo "$FIFO_A" "$FIFO_B" || { echo "[eval-xalloc] ABORT: mkfifo failed"; exit 1; }
run_alloc "$JOB_A" 0 "$FIFO_A"; PID_A=$RUN_PID; SED_A=$SED_PID
run_alloc "$JOB_B" 1 "$FIFO_B"; PID_B=$RUN_PID; SED_B=$SED_PID

wait "$PID_A"; RC_A=$?
wait "$PID_B"; RC_B=$?
wait "$SED_A" "$SED_B" 2>/dev/null   # drain BOTH prefixers (sed exits on fifo EOF) before EXITED
echo "[eval-xalloc] $(date '+%F %T %Z') EXITED rc_A($JOB_A,r0)=$RC_A rc_B($JOB_B,r1)=$RC_B"
if [ "$RC_A" -ne 0 ] || [ "$RC_B" -ne 0 ]; then exit 1; fi
exit 0
