#!/bin/bash
# Cross-alloc 8-card H100 DDP orchestrator for the graph_pscf backbone
# (scripts/train_graph_codeflow.py). ADAPTED from the PROVEN same-node cross-alloc
# scripts/_launch_graph_pscf_6card.sh. Joins SIX same-node (swarmh1002) swarm_h100
# allocs into one 8-rank DDP job via torchrun STATIC rendezvous over IB
# (swarmh1002-ib0). Topology (UNIFORM nproc_per_node=1 -> torchrun global-rank
# math is clean):
#   - 4 single-GPU allocs -> 1 rank each (--gres=gpu:1, CVD=0), node_rank 0..3
#   - 2 double-GPU allocs -> 2 ranks each (--gres=gpu:2, CVD=0 and CVD=1 pick the
#     two physical GPUs), node_rank 4/5 (alloc E) and 6/7 (alloc F)
# => WORLD_SIZE=8, all nproc_per_node=1.
#
# WHY batch8 (not the H200 batch16): batch16 backbone needs ~78GB and OOMs a single
# H100 80GB (verified). batch8 (~40-50GB) fits. To keep the SAME learning efficiency
# as the 4xH200 run (global batch 64, lr 8e-5): 8 ranks x bs8 x accum1 = global 64.
# Only global rank 0 writes ckpts (train_graph_codeflow.py is_main guard). Gradients
# all_reduce EVERY step across the 6 allocs over IB (the smoke verifies this).
#
# SMOKE (true 8-rank; verify rdzv + IB NCCL + grad all_reduce, then EXIT):
#   SMOKE=1 EMPIRICAL_MAX=256 NCCL_DEBUG=WARN OUT=/tmp/gpscf_8card_smoke \
#   bash scripts/_launch_graph_pscf_8card_h100.sh 2>&1 | tee scripts/_smoke_gpscf_8card.log
# REAL run (DURABLE, on the compute node so PPID=1 survives ssh disconnect):
#   ssh swarmh1002 "cd /scratch/ts1v23/workspace/noKslot_clean && setsid nohup \
#     RESUME_CKPT=runs/<OUT>/last_model.pt OVERWRITE=0 OUT=runs/<OUT> \
#     bash scripts/_launch_graph_pscf_8card_h100.sh > scripts/_train_gpscf_8card.log 2>&1 < /dev/null &"
set -uo pipefail
P=/scratch/ts1v23/workspace/noKslot_clean
cd "$P" || exit 1

# The 6 swarm_h100 allocs on swarmh1002 (4x 1-GPU + 2x 2-GPU). Env-overridable.
JOB_A="${JOB_A:-977973}"   # 1-GPU
JOB_B="${JOB_B:-977974}"   # 1-GPU
JOB_C="${JOB_C:-977975}"   # 1-GPU
JOB_D="${JOB_D:-977976}"   # 1-GPU
JOB_E="${JOB_E:-988069}"   # 2-GPU (split into 2 ranks)
JOB_F="${JOB_F:-988070}"   # 2-GPU (split into 2 ranks)
RDZV_HOST="${RDZV_HOST:-swarmh1002-ib0}"
RDZV_PORT="${RDZV_PORT:-29506}"   # distinct from vqvae 29503 / t2m 29501 / 6card 29505
SMOKE="${SMOKE:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"     # per-GPU; global = 8 ranks x bs8 x accum1 = 64
LR="${LR:-8e-5}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"     # 8 x bs8 = 64 already, no accumulation needed
TOKEN_CACHE="${TOKEN_CACHE:-data/codeflow_tokens_v4b272neutral_n8192_ep219_fulllen300}"
FROZEN_CKPT="${FROZEN_CKPT:-runs/vqvae_v4b272neutral_C96_J144_d512_Q4_n8192_b16g64_300ep_curric50to60_seed42/best_model.pt}"
EPOCHS="${EPOCHS:-600}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
NUM_WORKERS="${NUM_WORKERS:-6}"
LOG_EVERY="${LOG_EVERY:-50}"
QA_EVERY="${QA_EVERY:-200}"
SAVE_EVERY="${SAVE_EVERY:-5}"
SEED="${SEED:-42}"
EMPIRICAL_MAX="${EMPIRICAL_MAX:-0}"   # 0 = full-set empirical z_q norm (real run, cache-hit); cap for SMOKE only
RESUME_CKPT="${RESUME_CKPT:-}"        # FULL path (parent must == OUT for resume_in_place)
OVERWRITE="${OVERWRITE:-1}"           # 0 for in-place resume into the run's OWN dir
# Two-phase human curriculum (v4b) + online gen-eval (opt-in) forwarded to the trainer.
HUMAN_UPSAMPLE_FACTOR="${HUMAN_UPSAMPLE_FACTOR:-3.0}"
HUMAN_UPSAMPLE_START_EPOCH="${HUMAN_UPSAMPLE_START_EPOCH:-0}"
HUMAN_UPSAMPLE_PHASE2_FACTOR="${HUMAN_UPSAMPLE_PHASE2_FACTOR:-4.5}"
HUMAN_UPSAMPLE_PHASE2_START_EPOCH="${HUMAN_UPSAMPLE_PHASE2_START_EPOCH:-50}"
GEN_EVAL="${GEN_EVAL:-}"
EVALUATOR_CKPT="${EVALUATOR_CKPT:-}"
OUT="${OUT:?set OUT (use /tmp/gpscf_8card_smoke for the smoke, runs/... for real)}"

# Single-instance lock: the inner launch has NO pgrep double-launch guard for the
# cross-alloc case, so prevent a double orchestrator run HERE.
mkdir -p .aris/meta
exec 9>".aris/meta/.gpscf8card.lock"
flock -n 9 || { echo "[gpscf-8card] ABORT: already running"; exit 0; }

# Shared env every rank's inner launch inherits. NNODES=8 -> static-rendezvous branch;
# NPROC_PER_NODE=1 (uniform); CVD is passed PER-RANK below (0 for 1-GPU allocs and the
# first GPU of a 2-GPU alloc, 1 for the second GPU of a 2-GPU alloc).
COMMON_ENV="NNODES=8 NPROC_PER_NODE=1 MASTER_ADDR=$RDZV_HOST MASTER_PORT=$RDZV_PORT BATCH_SIZE=$BATCH_SIZE LR=$LR GRAD_ACCUM=$GRAD_ACCUM TOKEN_CACHE=$TOKEN_CACHE FROZEN_CKPT=$FROZEN_CKPT EPOCHS=$EPOCHS WARMUP_STEPS=$WARMUP_STEPS NUM_WORKERS=$NUM_WORKERS LOG_EVERY=$LOG_EVERY QA_EVERY=$QA_EVERY SAVE_EVERY=$SAVE_EVERY SEED=$SEED EMPIRICAL_MAX=$EMPIRICAL_MAX RESUME_CKPT=$RESUME_CKPT OVERWRITE=$OVERWRITE OUT=$OUT SMOKE=$SMOKE HUMAN_UPSAMPLE_FACTOR=$HUMAN_UPSAMPLE_FACTOR HUMAN_UPSAMPLE_START_EPOCH=$HUMAN_UPSAMPLE_START_EPOCH HUMAN_UPSAMPLE_PHASE2_FACTOR=$HUMAN_UPSAMPLE_PHASE2_FACTOR HUMAN_UPSAMPLE_PHASE2_START_EPOCH=$HUMAN_UPSAMPLE_PHASE2_START_EPOCH GEN_EVAL=$GEN_EVAL EVALUATOR_CKPT=$EVALUATOR_CKPT"

echo "[gpscf-8card] $(date '+%F %T %Z') cross-alloc 8-card H100 DDP: 1gpu[$JOB_A,$JOB_B,$JOB_C,$JOB_D] + 2gpu[$JOB_E,$JOB_F] via $RDZV_HOST:$RDZV_PORT smoke=$SMOKE"
echo "[gpscf-8card] global=$(( BATCH_SIZE*8*GRAD_ACCUM )) (8 ranks x bs$BATCH_SIZE x accum$GRAD_ACCUM) lr=$LR warmup=$WARMUP_STEPS epochs=$EPOCHS overwrite=$OVERWRITE out=$OUT"
echo "[gpscf-8card] cache=$TOKEN_CACHE frozen=$FROZEN_CKPT resume=${RESUME_CKPT:-<none>}"

# One torchrun group per RANK. gres: 1-GPU allocs request gpu:1 (CVD=0 picks it);
# 2-GPU allocs run TWO overlapping steps each requesting gpu:2 so both GPUs are visible,
# and CVD (0/1) selects which physical GPU that rank uses. --no-kill so one rank's
# transient failure does not tear down the step.
run_rank() {
    local tag="$1" job="$2" gres="$3" cvd="$4" nr="$5"
    srun --jobid="$job" --overlap --nodes=1 --ntasks=1 \
      --gres="$gres" --cpus-per-task=8 --no-kill \
      bash -c "cd '$P' && NODE_RANK=$nr CVD=$cvd $COMMON_ENV bash scripts/_launch_graph_pscf.sh" \
      2>&1 | stdbuf -oL sed "s/^/[$tag] /"
}

# node_rank 0 = master (starts the TCPStore). All ranks nproc_per_node=1.
run_rank r0 "$JOB_A" gpu:1 0 0 & PIDS=($!)
run_rank r1 "$JOB_B" gpu:1 0 1 & PIDS+=($!)
run_rank r2 "$JOB_C" gpu:1 0 2 & PIDS+=($!)
run_rank r3 "$JOB_D" gpu:1 0 3 & PIDS+=($!)
run_rank r4 "$JOB_E" gpu:2 0 4 & PIDS+=($!)
run_rank r5 "$JOB_E" gpu:2 1 5 & PIDS+=($!)
run_rank r6 "$JOB_F" gpu:2 0 6 & PIDS+=($!)
run_rank r7 "$JOB_F" gpu:2 1 7 & PIDS+=($!)

rc_all=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || rc_all=1
done
echo "[gpscf-8card] $(date '+%F %T %Z') ALL RANKS EXITED rc_all=$rc_all"
exit "$rc_all"
