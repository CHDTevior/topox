#!/bin/bash
# Cross-NODE 8-card A100 DDP orchestrator for the graph_pscf backbone on the
# MERGED n2048 token cache (codebook-size comparison vs the live 512-backbone).
# Sibling of scripts/_launch_graph_pscf_2node_h200.sh, adapted for 2x (4xA100) nodes.
#
# ── Differences vs the H200 2-node launcher ──
#   * Each A100 node = ONE 4-GPU alloc -> NPROC_PER_NODE=4, CVD=0,1,2,3 (vs 2 on H200).
#     WORLD_SIZE = 2 nodes x 4 = 8.
#   * ACTIVE IB rail on swarma nodes is ib0 / mlx5_0 (ib1 DOWN) -- the OPPOSITE of the
#     flamingo/blossom H200 nodes (ib1/mlx5_1). Verified 2026-06-15: swarma1003
#     ib0=10.6.15.9, swarma1004 ib0=10.6.15.68, same /22. So NCCL_SOCKET_IFNAME=ib0
#     NCCL_IB_HCA=mlx5_0, RDZV_HOST=swarma1003 ib0 IP.
#   * Each node is a FULL 4-GPU alloc (own cgroup) -> intra-node NVLink P2P/SHM stays
#     ENABLED; only the inter-node ring goes over IB (same as H200).
#   * Goyal: per-GPU B=8 x 8 = global64 -> lr 8e-5 (anchor global96=1.2e-4). Same
#     effective global64/lr8e-5 as the 4xH200 512-backbone -> clean codebook comparison.
#     C=96 on 80GB A100 -> B=8 (B12/16 OOM, per merged-coldscan note).
#
# ── PLUMBING SMOKE (rendezvous + NCCL via NET/IB + WORLD_SIZE=8; 4-iter) ──
#   Run AFTER pre-warming empirical_stats.pt; smoke with EMPIRICAL_MAX=0 so it uses the
#   warmed full cache and does NOT overwrite it with a 256-clip cache (which would
#   make the real launch cold-scan in DDP). NEVER smoke with EMPIRICAL_MAX=256 here.
#   SMOKE=1 EMPIRICAL_MAX=0 NCCL_DEBUG=INFO OUT=/tmp/gpscf_a100_smoke \
#     bash scripts/_launch_graph_pscf_2node_a100.sh 2>&1 | tee scripts/_smoke_gpscf_a100.log
#
# ── REAL run (DURABLE) — run ON swarma1003 (master), PPID=1 survives ssh drop ──
#   ssh swarma1003 "cd /scratch/ts1v23/workspace/noKslot_clean && setsid nohup \
#     bash scripts/_launch_graph_pscf_2node_a100.sh > <OUT>/orch.log 2>&1 </dev/null &"
set -uo pipefail
P=/scratch/ts1v23/workspace/noKslot_clean
cd "$P" || exit 1

JOB_A="${JOB_A:-976853}"            # swarma1003 (master, node_rank 0)
JOB_B="${JOB_B:-974143}"            # swarma1004 (worker, node_rank 1)
MASTER_NODE="${MASTER_NODE:-swarma1003}"   # node_rank 0 host (overridable for resume on other swarma)
WORKER_NODE="${WORKER_NODE:-swarma1004}"   # node_rank 1 host
RDZV_HOST="${RDZV_HOST:-10.6.15.9}"        # MASTER_NODE ib0 IP (master); IP not hostname
RDZV_PORT="${RDZV_PORT:-29509}"            # distinct from h200(29507)/6card/8card
# Active IB iface/HCA on swarma A100 nodes (ib0/mlx5_0; overridable for resume).
NCCL_IFACE="${NCCL_SOCKET_IFNAME:-ib0}"
NCCL_HCA="${NCCL_IB_HCA:-mlx5_0}"
SMOKE="${SMOKE:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"              # per-GPU; global = 8 x BATCH_SIZE (C=96 -> B8 on 80GB)
LR="${LR:-8e-5}"                           # Goyal: 1.2e-4 x (global/96); B8x8=global64 -> 8e-5
TOKEN_CACHE="${TOKEN_CACHE:-data/codeflow_tokens_mergedL4TB_n2048_ep199_fulllen300}"
FROZEN_CKPT="${FROZEN_CKPT:-runs/vqvae_L4safeTB_C96_J144_d512_Q4_n2048_b32_300ep_seed42/best_model.pt}"
EPOCHS="${EPOCHS:-600}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
DEPTH_DOUBLE="${DEPTH_DOUBLE:-6}"
DEPTH_SINGLE="${DEPTH_SINGLE:-12}"
MAX_T_LAT="${MAX_T_LAT:-75}"
COND_DROP_PROB="${COND_DROP_PROB:-0.1}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
NUM_WORKERS="${NUM_WORKERS:-4}"            # 4 ranks x 4 workers = 16 < 32 CPUs/alloc
LOG_EVERY="${LOG_EVERY:-50}"
QA_EVERY="${QA_EVERY:-200}"
SAVE_EVERY="${SAVE_EVERY:-5}"
SEED="${SEED:-42}"
EMPIRICAL_MAX="${EMPIRICAL_MAX:-0}"        # 0 = full-set empirical z_q norm (pre-warm first!)
RESUME_CKPT="${RESUME_CKPT:-}"
OVERWRITE="${OVERWRITE:-1}"                # 0 for in-place resume into the run's OWN dir
OUT="${OUT:?set OUT (use /tmp/gpscf_a100_smoke for smoke, runs/... for real)}"

# In-place resume guard: a bare RESUME_CKPT must be qualified to $OUT (this script cd's to $P).
if [ -n "$RESUME_CKPT" ] && [[ "$RESUME_CKPT" != /* && "$RESUME_CKPT" != */* ]]; then
    RESUME_CKPT="$OUT/$RESUME_CKPT"
fi

# Empirical-stats PREFLIGHT: a real run with EMPIRICAL_MAX=0 MUST hit a pre-warmed
# FULL-SET cache, else rank-0 cold-scans (~40min) while other ranks wait at broadcast
# -> NCCL 10-min watchdog -> SIGABRT. Existence is NOT enough: a smoke run with
# EMPIRICAL_MAX=256 would leave a small (max_clips=256) cache whose key the trainer
# rejects -> cold-scan in DDP anyway. So require the stored count to be full-set
# (~55M tokens; a 256-clip cache is <1M). NEVER smoke with EMPIRICAL_MAX=256 here.
if [ "$SMOKE" = 0 ] && [ "$EMPIRICAL_MAX" = 0 ]; then
    _cnt=$(/scratch/ts1v23/.conda/bin/python3 -c "import torch;print(int(torch.load('$TOKEN_CACHE/empirical_stats.pt',map_location='cpu').get('count',0)))" 2>/dev/null || echo 0)
    if [ "${_cnt:-0}" -lt 10000000 ]; then
        echo "[gpscf-a100] ABORT: $TOKEN_CACHE/empirical_stats.pt missing or not full-set (count=${_cnt:-0} < 10M) -- pre-warm single-process EMPIRICAL_MAX=0 BEFORE DDP (do NOT smoke with EMPIRICAL_MAX=256)"; exit 3
    fi
fi

GLOBAL=$(( BATCH_SIZE * 8 ))

mkdir -p .aris/meta
exec 9>".aris/meta/.gpscf_a100_2node.lock"
flock -n 9 || { echo "[gpscf-a100] ABORT: already running"; exit 75; }
# Record orchestrator PID so a watchdog can verify THIS launch came up (PID-based).
echo $$ > ".aris/meta/.gpscf_a100_orch.pid" 2>/dev/null || true

# NNODES=2 -> static-rendezvous branch. Full-node allocs -> intra-node NVLink P2P/SHM ON.
COMMON_ENV="NNODES=2 NPROC_PER_NODE=4 MASTER_ADDR=$RDZV_HOST MASTER_PORT=$RDZV_PORT CVD=0,1,2,3 \
NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=0 NCCL_SOCKET_IFNAME=$NCCL_IFACE NCCL_IB_DISABLE=0 NCCL_IB_HCA=$NCCL_HCA \
BATCH_SIZE=$BATCH_SIZE LR=$LR TOKEN_CACHE=$TOKEN_CACHE FROZEN_CKPT=$FROZEN_CKPT \
EPOCHS=$EPOCHS WARMUP_STEPS=$WARMUP_STEPS DEPTH_DOUBLE=$DEPTH_DOUBLE DEPTH_SINGLE=$DEPTH_SINGLE \
MAX_T_LAT=$MAX_T_LAT COND_DROP_PROB=$COND_DROP_PROB AMP_DTYPE=$AMP_DTYPE NUM_WORKERS=$NUM_WORKERS \
LOG_EVERY=$LOG_EVERY QA_EVERY=$QA_EVERY SAVE_EVERY=$SAVE_EVERY SEED=$SEED \
EMPIRICAL_MAX=$EMPIRICAL_MAX RESUME_CKPT=$RESUME_CKPT OVERWRITE=$OVERWRITE OUT=$OUT SMOKE=$SMOKE"

echo "[gpscf-a100] $(date '+%F %T %Z') cross-NODE 8-card A100 DDP: $JOB_A($MASTER_NODE,r0)+$JOB_B($WORKER_NODE,r1) via $RDZV_HOST:$RDZV_PORT smoke=$SMOKE"
echo "[gpscf-a100] global=$GLOBAL (8xbs$BATCH_SIZE) lr=$LR warmup=$WARMUP_STEPS epochs=$EPOCHS overwrite=$OVERWRITE out=$OUT"
echo "[gpscf-a100] cache=$TOKEN_CACHE frozen=$FROZEN_CKPT resume=${RESUME_CKPT:-<none>} (NCCL $NCCL_IFACE/$NCCL_HCA, intra-node P2P/SHM ON)"

# One torchrun group per node; static rendezvous joins them into WORLD_SIZE=8.
run_alloc() {
    local tag="$1" job="$2" noderank="$3"
    srun --jobid="$job" --overlap --nodes=1 --ntasks=1 \
      --gres=gpu:4 --cpus-per-task=32 --no-kill \
      bash -c "cd '$P' && NODE_RANK=$noderank $COMMON_ENV bash scripts/_launch_graph_pscf.sh" \
      2>&1 | stdbuf -oL sed "s/^/[$tag] /"
}
run_alloc "$MASTER_NODE" "$JOB_A" 0 & PID_A=$!
run_alloc "$WORKER_NODE" "$JOB_B" 1 & PID_B=$!

wait "$PID_A"; RC_A=$?
wait "$PID_B"; RC_B=$?
echo "[gpscf-a100] $(date '+%F %T %Z') EXITED rc_A($MASTER_NODE)=$RC_A rc_B($WORKER_NODE)=$RC_B"
if [ "$RC_A" -ne 0 ] || [ "$RC_B" -ne 0 ]; then exit 1; fi
exit 0
