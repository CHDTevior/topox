#!/bin/bash
# Cross-NODE 4-card H200 DDP orchestrator for the GRAPH-VQVAE (scripts/train_graph_vqvae.py),
# n8192 codebook ablation on the MERGED L4-safe+truebones dataset.
# ADAPTED VERBATIM from the PROVEN scripts/_launch_graph_pscf_2node_h200.sh (the live
# 512-backbone cross-node H200 orchestrator) — ONLY the inner training launcher
# (_launch_graph_vqvae.sh) and its env block differ; the cross-node infra (NPROC=2,
# gres=gpu:2, static IB rendezvous, intra-node NVLink + inter-node IB, lock, orch-pid,
# run_alloc, overridable NCCL iface/HCA) is unchanged.
#
# ── Clean codebook-size ablation (vs n2048) ──
#   train_graph_vqvae.py arch + loss-weight DEFAULTS already == the n2048 recipe
#   (d_ff 1536, n_graph 4, code_dim 512, Q4, ema 0.99, w_contact 0.1, w_world 0.25 ...),
#   so ONLY --num_codes changes (2048 -> 8192). Keep n2048's GLOBAL BATCH (64) + lr
#   (6.65e-5) + 300 epochs identical -> same #steps/epoch, same optimization, only the
#   per-quantizer codebook size differs. On 4 cards: per-GPU BATCH_SIZE=16 (16x2x2=64).
#
# ── Cross-NODE NCCL (H200) ──
#   flamingo0[12] (dual_h200, MASTER node_rank 0) + blossom0[1-4] (quad_h200, WORKER
#   node_rank 1), 2 GPU each (CVD=0,1). Intra-node NVLink P2P/SHM ON; inter-node ring
#   over IB. Active HCA on these nodes is mlx5_1/ib1 (mlx5_0/ib0 DOWN) — overridable so
#   the watchdog passes whatever it detects UP on the actual resume nodes.
#
# ── PLUMBING SMOKE (rendezvous + NCCL via NET/IB + WORLD_SIZE=4; 4-iter) ──
#   SMOKE=1 NCCL_DEBUG=INFO OVERWRITE=1 OUT=/tmp/vqvae8192_h200_smoke \
#     bash scripts/_launch_graph_vqvae_2node_h200.sh 2>&1 | tee scripts/_smoke_vqvae8192_h200.log
#
# ── REAL run (DURABLE) — run ON the master node (flamingo0X), PPID=1 survives ssh drop ──
#   ssh <master> "cd /scratch/ts1v23/workspace/noKslot_clean && JOB_A=.. JOB_B=.. \
#     MASTER_NODE=.. WORKER_NODE=.. RDZV_HOST=<master-ib-ip> NCCL_SOCKET_IFNAME=.. NCCL_IB_HCA=.. \
#     OUT=runs/vqvae_L4safeTB_C96_J144_d512_Q4_n8192_b16_300ep_seed42 setsid nohup \
#     bash scripts/_launch_graph_vqvae_2node_h200.sh > scripts/_train_vqvae8192_h200.log 2>&1 </dev/null &"
set -uo pipefail
P=/scratch/ts1v23/workspace/noKslot_clean
cd "$P" || exit 1

JOB_A="${JOB_A:-993172}"            # master alloc (node_rank 0), flamingo01 (dual_h200)
JOB_B="${JOB_B:-993170}"            # worker alloc (node_rank 1), blossom03 (quad_h200)
MASTER_NODE="${MASTER_NODE:-flamingo01}"
WORKER_NODE="${WORKER_NODE:-blossom03}"
RDZV_HOST="${RDZV_HOST:-10.6.15.127}"   # MASTER_NODE ib IP (IP not hostname; watchdog overrides per-resume)
RDZV_PORT="${RDZV_PORT:-29515}"         # distinct from backbone 29507 / 6card 29505 / 8card-a100 29505
NCCL_IFACE="${NCCL_SOCKET_IFNAME:-ib1}"
NCCL_HCA="${NCCL_IB_HCA:-mlx5_1}"
SMOKE="${SMOKE:-0}"
# ── n2048-identical recipe, only num_codes -> 8192. per-GPU 16 x 4 = global 64 (== n2048). ──
BATCH_SIZE="${BATCH_SIZE:-16}"          # per-GPU; global = 4 x BATCH_SIZE = 64
LR="${LR:-6.65e-5}"                     # == n2048 (global 64); flat (warmup 0) for fresh run
NUM_CODES="${NUM_CODES:-8192}"          # THE ablation knob (n512/n2048 -> n8192)
WARMUP_STEPS="${WARMUP_STEPS:-0}"       # n2048 used 0 (fresh-from-scratch flat lr)
EPOCHS="${EPOCHS:-300}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
NUM_WORKERS="${NUM_WORKERS:-3}"         # dual_h200 alloc ~8 CPUs; 2 ranks x 3 = 6 (n2048 used 3)
LOG_EVERY="${LOG_EVERY:-50}"
QA_EVERY="${QA_EVERY:-100}"
SAVE_EVERY="${SAVE_EVERY:-10}"          # last_model.pt every 10 ep (watchdog resumes from it on drop)
PERIODIC_SAVE_EVERY="${PERIODIC_SAVE_EVERY:-25}"
SEED="${SEED:-42}"
ANYTOP_ROOT="${ANYTOP_ROOT:-data/animo4d_anytop_clean_L4_safe_plus_truebones}"
MAX_JOINTS="${MAX_JOINTS:-144}"
MAX_COARSE="${MAX_COARSE:-96}"
MAX_FRAMES="${MAX_FRAMES:-64}"
RESUME_CKPT="${RESUME_CKPT:-}"          # FULL resume (model+opt+epoch+step); watchdog passes last_model.pt
OVERWRITE="${OVERWRITE:-1}"             # 0 for in-place resume into the run's OWN dir
OUT="${OUT:?set OUT (use /tmp/vqvae8192_h200_smoke for smoke, runs/... for real)}"

# In-place resume: train_graph_vqvae.py's resume_in_place guard needs the resume ckpt's
# parent dir to be exactly OUT. A bare RESUME_CKPT=last_model.pt (we cd to workspace)
# would resolve to workspace-root -> guard fail. Qualify it to $OUT/last_model.pt.
if [ -n "$RESUME_CKPT" ] && [[ "$RESUME_CKPT" != /* && "$RESUME_CKPT" != */* ]]; then
    RESUME_CKPT="$OUT/$RESUME_CKPT"
fi

GLOBAL=$(( BATCH_SIZE * 4 ))

# Single-instance lock (DISTINCT from the backbone's .gpscf_h200_2node.lock).
mkdir -p .aris/meta
exec 9>".aris/meta/.vqvae_h200_2node.lock"
flock -n 9 || { echo "[vqvae-h200] ABORT: already running"; exit 75; }
# Record orchestrator PID so the watchdog verifies THIS launch came up (PID-based).
echo $$ > ".aris/meta/.vqvae_h200_orch.pid" 2>/dev/null || true

# Shared env each node's inner launch inherits. NNODES=2 -> static-rendezvous branch.
# CROSS-NODE NCCL: intra-node NVLink (P2P/SHM ON), inter-node over IB (NCCL_IFACE/HCA).
COMMON_ENV="NNODES=2 NPROC_PER_NODE=2 MASTER_ADDR=$RDZV_HOST MASTER_PORT=$RDZV_PORT CVD=0,1 \
NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=0 NCCL_SOCKET_IFNAME=$NCCL_IFACE NCCL_IB_DISABLE=0 NCCL_IB_HCA=$NCCL_HCA \
BATCH_SIZE=$BATCH_SIZE LR=$LR NUM_CODES=$NUM_CODES WARMUP_STEPS=$WARMUP_STEPS AMP_DTYPE=$AMP_DTYPE \
EPOCHS=$EPOCHS NUM_WORKERS=$NUM_WORKERS LOG_EVERY=$LOG_EVERY QA_EVERY=$QA_EVERY SAVE_EVERY=$SAVE_EVERY \
PERIODIC_SAVE_EVERY=$PERIODIC_SAVE_EVERY SEED=$SEED ANYTOP_ROOT=$ANYTOP_ROOT \
MAX_JOINTS=$MAX_JOINTS MAX_COARSE=$MAX_COARSE MAX_FRAMES=$MAX_FRAMES RESUME_CKPT=$RESUME_CKPT OVERWRITE=$OVERWRITE OUT=$OUT SMOKE=$SMOKE"

echo "[vqvae-h200] $(date '+%F %T %Z') cross-NODE 4-card H200 DDP: $JOB_A($MASTER_NODE,r0)+$JOB_B($WORKER_NODE,r1) via $RDZV_HOST:$RDZV_PORT smoke=$SMOKE"
echo "[vqvae-h200] global=$GLOBAL (4xbs$BATCH_SIZE) lr=$LR num_codes=$NUM_CODES warmup=$WARMUP_STEPS epochs=$EPOCHS overwrite=$OVERWRITE out=$OUT"
echo "[vqvae-h200] anytop_root=$ANYTOP_ROOT max_joints=$MAX_JOINTS max_coarse=$MAX_COARSE max_frames=$MAX_FRAMES resume=${RESUME_CKPT:-<none>} (NCCL $NCCL_IFACE/$NCCL_HCA, intra-node P2P/SHM ON)"

# One torchrun group per node; static rendezvous joins them into WORLD_SIZE=4.
run_alloc() {
    local tag="$1" job="$2" noderank="$3"
    srun --jobid="$job" --overlap --nodes=1 --ntasks=1 \
      --gres=gpu:2 --cpus-per-task=8 --no-kill \
      bash -c "cd '$P' && NODE_RANK=$noderank $COMMON_ENV bash scripts/_launch_graph_vqvae.sh" \
      2>&1 | stdbuf -oL sed "s/^/[$tag] /"
}
run_alloc "$MASTER_NODE" "$JOB_A" 0 & PID_A=$!
run_alloc "$WORKER_NODE" "$JOB_B" 1 & PID_B=$!

wait "$PID_A"; RC_A=$?
wait "$PID_B"; RC_B=$?
echo "[vqvae-h200] $(date '+%F %T %Z') EXITED rc_A($MASTER_NODE)=$RC_A rc_B($WORKER_NODE)=$RC_B"
if [ "$RC_A" -ne 0 ] || [ "$RC_B" -ne 0 ]; then exit 1; fi
exit 0
