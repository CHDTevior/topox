#!/bin/bash
# Cross-NODE 4-card H200 DDP orchestrator for the graph_pscf backbone
# (scripts/train_graph_codeflow.py) on the MERGED n512 token cache.
# ADAPTED from the PROVEN scripts/_launch_graph_pscf_6card.sh (same-node 6-card),
# turned into a genuine 2-NODE cross-node run: flamingo01 (993171, 2xH200) +
# blossom03 (988074, 2xH200) -> WORLD_SIZE=4 via torchrun STATIC rendezvous over IB.
#
# ── Why this differs from the same-node 6-card orchestrator ──
#   * flamingo01 and blossom03 are DIFFERENT physical nodes -> genuine cross-NODE DDP.
#   * Each node = ONE 2-GPU alloc (both H200 in ONE cgroup) -> intra-node P2P/SHM
#     (NVLink) MUST stay ENABLED; only the inter-node ring goes over IB. So we
#     EXPLICITLY export NCCL_P2P_DISABLE=0 / NCCL_SHM_DISABLE=0 to OVERRIDE the inner
#     launcher's ${VAR:-1} same-node-cross-cgroup defaults (wrong here).
#   * ACTIVE HCA on these nodes is mlx5_1 / ib1 (mlx5_0/ib0 are DOWN), verified
#     2026-06-13: flamingo01 ib1=10.6.15.127, blossom03 ib1=10.6.15.132, same /22,
#     ping 0% loss, 200 Gb/s (2X NDR). So NCCL_IB_HCA=mlx5_1 NCCL_SOCKET_IFNAME=ib1.
#   * RDZV_HOST = flamingo01 ib1 IP (10.6.15.127), NOT a hostname (DNS alias safety).
#
# ── Batch/lr (H200 141GB relaxes the C=96 B=8 limit that 80GB cards hit) ──
#   per-GPU BATCH_SIZE x 4 = global; lr = 1.2e-4 x (global/96) (Goyal, old anchor
#   global96). SMOKE to find max B (12/16/24) under 141GB. e.g. B=24->global96->lr1.2e-4.
#
# ── PLUMBING SMOKE (verify rendezvous + NCCL via NET/IB + WORLD_SIZE=4; 4-iter) ──
#   SMOKE=1 EMPIRICAL_MAX=256 NCCL_DEBUG=INFO BATCH_SIZE=16 LR=8e-5 OUT=/tmp/gpscf_h200_smoke \
#     bash scripts/_launch_graph_pscf_2node_h200.sh 2>&1 | tee scripts/_smoke_gpscf_h200.log
#   (confirm: inter-node ring "via NET/IB" mlx5_1/ib1, intra-node NVLink/P2P, WORLD_SIZE=4)
#
# ── REAL run (DURABLE) — run ON flamingo01 (master node), PPID=1 survives ssh drop ──
#   ssh flamingo01 "cd /scratch/ts1v23/workspace/noKslot_clean && BATCH_SIZE=.. LR=.. \
#     OUT=runs/codeflow_graph_pscf_mergedL4TB_n512_..._4xh200_seed42 setsid nohup \
#     bash scripts/_launch_graph_pscf_2node_h200.sh > scripts/_train_gpscf_h200.log 2>&1 </dev/null &"
set -uo pipefail
P=/scratch/ts1v23/workspace/noKslot_clean
cd "$P" || exit 1

JOB_A="${JOB_A:-993171}"            # master alloc (node_rank 0)
JOB_B="${JOB_B:-988074}"            # worker alloc (node_rank 1)
MASTER_NODE="${MASTER_NODE:-flamingo01}"   # node_rank 0 host (overridable: H200 alloc may land on flamingo02/blossom0X)
WORKER_NODE="${WORKER_NODE:-blossom03}"    # node_rank 1 host
RDZV_HOST="${RDZV_HOST:-10.6.15.127}"   # MASTER_NODE ib IP (master); IP not hostname
RDZV_PORT="${RDZV_PORT:-29507}"         # distinct from 6card/8card 29505
# Active IB iface/HCA on these H200 nodes (verified ib1/mlx5_1; overridable so the
# watchdog can pass whatever it detects UP on the actual resume nodes).
NCCL_IFACE="${NCCL_SOCKET_IFNAME:-ib1}"
NCCL_HCA="${NCCL_IB_HCA:-mlx5_1}"
SMOKE="${SMOKE:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"          # per-GPU; global = 4 x BATCH_SIZE
LR="${LR:-8e-5}"                        # Goyal: 1.2e-4 x (global/96); B16->global64->8e-5
TOKEN_CACHE="${TOKEN_CACHE:-data/codeflow_tokens_mergedL4TB_n512_ep209_fulllen300}"
FROZEN_CKPT="${FROZEN_CKPT:-runs/vqvae_L4safeTB_C96_J144_d512_Q4_n512_b32_300ep_seed42/best_model.pt}"
EPOCHS="${EPOCHS:-600}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
DEPTH_DOUBLE="${DEPTH_DOUBLE:-6}"
DEPTH_SINGLE="${DEPTH_SINGLE:-12}"
MAX_T_LAT="${MAX_T_LAT:-75}"
COND_DROP_PROB="${COND_DROP_PROB:-0.1}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
NUM_WORKERS="${NUM_WORKERS:-4}"   # dual_h200 alloc has only 8 CPUs total; 2 ranks x 4 workers fits
LOG_EVERY="${LOG_EVERY:-50}"
QA_EVERY="${QA_EVERY:-200}"
SAVE_EVERY="${SAVE_EVERY:-5}"     # H200 alloc may drop -> checkpoint more often (drop-safety)
SEED="${SEED:-42}"
EMPIRICAL_MAX="${EMPIRICAL_MAX:-0}"     # 0 = full-set empirical z_q norm (real run); cap for SMOKE
RESUME_CKPT="${RESUME_CKPT:-}"
OVERWRITE="${OVERWRITE:-1}"             # 0 for in-place resume into the run's OWN dir
OUT="${OUT:?set OUT (use /tmp/gpscf_h200_smoke for smoke, runs/... for real)}"

# In-place resume guard (train_graph_codeflow.py) requires the resume ckpt's parent
# dir to be exactly OUT. A bare RESUME_CKPT=last_model.pt (this script cd's to the
# workspace) would resolve to workspace-root, not $OUT -> [OUT FAIL]. Qualify it.
if [ -n "$RESUME_CKPT" ] && [[ "$RESUME_CKPT" != /* && "$RESUME_CKPT" != */* ]]; then
    RESUME_CKPT="$OUT/$RESUME_CKPT"
fi

GLOBAL=$(( BATCH_SIZE * 4 ))

# Single-instance lock (the inner launch has no pgrep guard for the cross case).
mkdir -p .aris/meta
exec 9>".aris/meta/.gpscf_h200_2node.lock"
flock -n 9 || { echo "[gpscf-h200] ABORT: already running"; exit 75; }
# Record orchestrator PID so the watchdog can verify THIS launch came up (PID-based,
# not pgrep-by-name which would also match the launching ssh wrapper's argv).
echo $$ > ".aris/meta/.gpscf_h200_orch.pid" 2>/dev/null || true

# Shared env each node's inner launch inherits. NNODES=2 -> static-rendezvous branch.
# CROSS-NODE NCCL: keep intra-node NVLink (P2P/SHM ON), inter-node over IB mlx5_1/ib1.
COMMON_ENV="NNODES=2 NPROC_PER_NODE=2 MASTER_ADDR=$RDZV_HOST MASTER_PORT=$RDZV_PORT CVD=0,1 \
NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=0 NCCL_SOCKET_IFNAME=$NCCL_IFACE NCCL_IB_DISABLE=0 NCCL_IB_HCA=$NCCL_HCA \
BATCH_SIZE=$BATCH_SIZE LR=$LR TOKEN_CACHE=$TOKEN_CACHE FROZEN_CKPT=$FROZEN_CKPT \
EPOCHS=$EPOCHS WARMUP_STEPS=$WARMUP_STEPS DEPTH_DOUBLE=$DEPTH_DOUBLE DEPTH_SINGLE=$DEPTH_SINGLE \
MAX_T_LAT=$MAX_T_LAT COND_DROP_PROB=$COND_DROP_PROB AMP_DTYPE=$AMP_DTYPE NUM_WORKERS=$NUM_WORKERS \
LOG_EVERY=$LOG_EVERY QA_EVERY=$QA_EVERY SAVE_EVERY=$SAVE_EVERY SEED=$SEED \
EMPIRICAL_MAX=$EMPIRICAL_MAX RESUME_CKPT=$RESUME_CKPT OVERWRITE=$OVERWRITE OUT=$OUT SMOKE=$SMOKE"

echo "[gpscf-h200] $(date '+%F %T %Z') cross-NODE 4-card H200 DDP: $JOB_A($MASTER_NODE,r0)+$JOB_B($WORKER_NODE,r1) via $RDZV_HOST:$RDZV_PORT smoke=$SMOKE"
echo "[gpscf-h200] global=$GLOBAL (4xbs$BATCH_SIZE) lr=$LR warmup=$WARMUP_STEPS epochs=$EPOCHS overwrite=$OVERWRITE out=$OUT"
echo "[gpscf-h200] cache=$TOKEN_CACHE frozen=$FROZEN_CKPT resume=${RESUME_CKPT:-<none>} (NCCL $NCCL_IFACE/$NCCL_HCA, intra-node P2P/SHM ON)"

# One torchrun group per node; static rendezvous joins them into WORLD_SIZE=4.
run_alloc() {
    local tag="$1" job="$2" noderank="$3"
    srun --jobid="$job" --overlap --nodes=1 --ntasks=1 \
      --gres=gpu:2 --cpus-per-task=8 --no-kill \
      bash -c "cd '$P' && NODE_RANK=$noderank $COMMON_ENV bash scripts/_launch_graph_pscf.sh" \
      2>&1 | stdbuf -oL sed "s/^/[$tag] /"
}
run_alloc "$MASTER_NODE" "$JOB_A" 0 & PID_A=$!
run_alloc "$WORKER_NODE" "$JOB_B" 1 & PID_B=$!

wait "$PID_A"; RC_A=$?
wait "$PID_B"; RC_B=$?
echo "[gpscf-h200] $(date '+%F %T %Z') EXITED rc_A($MASTER_NODE)=$RC_A rc_B($WORKER_NODE)=$RC_B"
if [ "$RC_A" -ne 0 ] || [ "$RC_B" -ne 0 ]; then exit 1; fi
exit 0
