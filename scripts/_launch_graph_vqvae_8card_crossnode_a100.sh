#!/bin/bash
# Cross-NODE 8-card A100 DDP orchestrator for the GRAPH-VQVAE (train_graph_vqvae.py).
# RE-PARAMETERIZED COPY of the proven scripts/_launch_graph_vqvae_6card.sh (do NOT edit
# that same-node file). Joins TWO different 4xA100-80GB nodes into one 8-rank DDP job via
# torchrun STATIC rendezvous over IB. Used to CONTINUE (--resume) the live 4-card merged-
# dataset VQVAE as an 8-card run once a 2nd 4-card A100 alloc lands on another node.
#
# ── Why this differs from the 6-card SAME-NODE orchestrator (verified 2026-06-11) ──
#   * swarma1004 is a 4-GPU node, so the 2nd alloc is on a DIFFERENT physical node →
#     genuine 2-node cross-NODE DDP (NOT the same-node cross-cgroup swarmh1002 trick).
#   * Each node = ONE full 4-GPU alloc (all 4 local GPUs in ONE cgroup) → intra-node P2P/
#     SHM (NVLink) MUST stay ENABLED; only the inter-node ring goes over IB. So we EXPLICITLY
#     export NCCL_P2P_DISABLE=0 / NCCL_SHM_DISABLE=0 to OVERRIDE the inner launcher's
#     ${VAR:-1} defaults (which were correct for same-node-cross-cgroup, WRONG here — they
#     would needlessly cripple the fast intra-node NVLink path and HURT throughput).
#   * NNODES=2 NPROC_PER_NODE=4 (WORLD_SIZE=8), RDZV_HOST=swarma1004-ib0 (10.6.15.68),
#     RDZV_PORT=29505. srun --gres=gpu:4 --cpus-per-task=32 (full node each).
#   * Inner launcher scripts/_launch_graph_vqvae.sh needs NO edit (its NNODES>1 branch already
#     builds the correct static-rendezvous args and reads MAX_JOINTS/MAX_COARSE/MAX_FRAMES +
#     RESUME_CKPT from env).
#
# ── Goyal linear scaling for the 4→8 card resume ──
#   global batch 4x32=128 → 8x32=256 (k=2). lr 1.33e-4 → 2.66e-4. Re-warm WARMUP_STEPS=500
#   (keyed to steps-SINCE-LAUNCH in train_graph_vqvae.py, so --resume re-warms lr from 0 at
#   this launch; the live run used warmup 0 — this run MUST re-warm the doubled lr). EPOCHS
#   stays 300 (Goyal: same data-traversal-per-epoch, fewer larger-batch steps from cutover on).
#
# ── PREREQ (HARD) ──
#   * last_model.pt MUST exist (save_every=10 → first ckpt at end of ep9). Cannot --resume
#     before then. Verify: ls runs/vqvae_L4safeTB_C96_J144_d512_Q4_n512_b32_300ep_seed42/last_model.pt
#   * Stop the live 4-card run FIRST (pkill the torchrun on swarma1004, bracket pattern; NEVER
#     scancel). Verify 0 train procs + 4 GPUs freed before launching this.
#   * Verify the 2nd node's 4 GPUs are idle AND not another project's (cross-project card rule).
#
# ── PLUMBING SMOKE (verify rendezvous + NCCL via NET/IB + WORLD_SIZE=8; 4-iter, no resume) ──
#   JOB_B=<new_alloc> SMOKE=1 RESUME_CKPT= OVERWRITE=1 NCCL_DEBUG=INFO OUT=/tmp/vqvae_8card_smoke \
#     bash scripts/_launch_graph_vqvae_8card_crossnode_a100.sh 2>&1 | tee scripts/_smoke_vqvae_8card.log
#   (confirm log shows inter-node ring "via NET/IB" (mlx5_0/ib0), intra-node via NVLink/P2P, WORLD_SIZE=8;
#    OVERWRITE=1 so a re-run into the /tmp smoke dir does not trip the non-empty-dir guard.)
#
# ── THROUGHPUT SMOKE (after ckpt exists; measure 8-card items/s vs live 4-card ~75.8) ──
#   JOB_B=<new_alloc> SMOKE=0 OVERWRITE=1 OUT=/tmp/vqvae_8card_tput \
#     RESUME_CKPT=runs/vqvae_L4safeTB_C96_J144_d512_Q4_n512_b32_300ep_seed42/last_model.pt \
#     bash scripts/_launch_graph_vqvae_8card_crossnode_a100.sh   (tear down after a clean window)
#
# ── REAL resume (DURABLE) — run ON swarma1004 (master compute node), PPID=1 ──
#   ssh swarma1004 "cd /scratch/ts1v23/workspace/noKslot_clean && JOB_B=<new_alloc> setsid nohup \
#     bash scripts/_launch_graph_vqvae_8card_crossnode_a100.sh > scripts/_train_vqvae_8card.log 2>&1 </dev/null &"
set -uo pipefail
P=/scratch/ts1v23/workspace/noKslot_clean
cd "$P" || exit 1

JOB_A="${JOB_A:-974143}"                 # swarma1004 (master, node_rank 0); the live 4-card alloc.
JOB_B="${JOB_B:?set JOB_B = the NEW 4-card A100 alloc jobid on the 2nd node (node_rank 1)}"
RDZV_HOST="${RDZV_HOST:-swarma1004-ib0}" # master node IB hostname (10.6.15.68); use the IP if DNS flaky.
RDZV_PORT="${RDZV_PORT:-29505}"          # distinct from 6card 29503 / t2m 29501; verified free on swarma1004.
SMOKE="${SMOKE:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"           # per-GPU (UNCHANGED from the 4-card run); global = 8x32 = 256.
LR="${LR:-2.66e-4}"                      # Goyal: 1.33e-4 x (256/128).
WARMUP_STEPS="${WARMUP_STEPS:-500}"      # re-warm the doubled lr from 0 over first 500 steps of THIS launch.
AMP_DTYPE="${AMP_DTYPE:-bf16}"
EPOCHS="${EPOCHS:-300}"
NUM_WORKERS="${NUM_WORKERS:-6}"
LOG_EVERY="${LOG_EVERY:-50}"
QA_EVERY="${QA_EVERY:-100}"
SAVE_EVERY="${SAVE_EVERY:-10}"
PERIODIC_SAVE_EVERY="${PERIODIC_SAVE_EVERY:-25}"
OVERWRITE="${OVERWRITE:-0}"              # 0 = in-place resume into the run's OWN dir (no log truncation).
SEED="${SEED:-42}"
ANYTOP_ROOT="${ANYTOP_ROOT:-data/animo4d_anytop_clean_L4_safe_plus_truebones}"
MAX_JOINTS="${MAX_JOINTS:-144}"          # MUST match the live run (Dragon J=142) or strict resume load fails.
MAX_COARSE="${MAX_COARSE:-96}"
MAX_FRAMES="${MAX_FRAMES:-64}"
RESUME_CKPT="${RESUME_CKPT-runs/vqvae_L4safeTB_C96_J144_d512_Q4_n512_b32_300ep_seed42/last_model.pt}"
OUT="${OUT:-runs/vqvae_L4safeTB_C96_J144_d512_Q4_n512_b32_300ep_seed42}"

# Single-instance lock (DISTINCT from the 6card lock). The inner launch has no cross-node
# pgrep guard, so prevent a double orchestrator run here.
mkdir -p .aris/meta
exec 9>".aris/meta/.vqvae8card.lock"
flock -n 9 || { echo "[vqvae-8card] ABORT: already running"; exit 0; }

# Cross-NODE NCCL: KEEP IB (SOCKET_IFNAME=ib0, IB_DISABLE=0); pin the verbs HCA to mlx5_0
# (ib1/mlx5_1 is DOWN on swarma1004, so do NOT let NCCL auto-pick and stumble onto it —
# SOCKET_IFNAME only controls socket/bootstrap, NCCL_IB_HCA controls RDMA HCA selection;
# mirrors the proven scripts/_launch_token_diffusion_8card_a100.sh). OVERRIDE the inner
# launcher's P2P/SHM :-1 defaults to 0 so intra-node 4-GPU NVLink/SHM stays enabled (inter-node uses IB).
COMMON_ENV="NNODES=2 NPROC_PER_NODE=4 MASTER_ADDR=$RDZV_HOST MASTER_PORT=$RDZV_PORT CVD=0,1,2,3 \
NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=0 NCCL_SOCKET_IFNAME=ib0 NCCL_IB_DISABLE=0 NCCL_IB_HCA=mlx5_0 \
BATCH_SIZE=$BATCH_SIZE LR=$LR WARMUP_STEPS=$WARMUP_STEPS AMP_DTYPE=$AMP_DTYPE EPOCHS=$EPOCHS \
NUM_WORKERS=$NUM_WORKERS LOG_EVERY=$LOG_EVERY QA_EVERY=$QA_EVERY SAVE_EVERY=$SAVE_EVERY \
PERIODIC_SAVE_EVERY=$PERIODIC_SAVE_EVERY OVERWRITE=$OVERWRITE SEED=$SEED ANYTOP_ROOT=$ANYTOP_ROOT \
MAX_JOINTS=$MAX_JOINTS MAX_COARSE=$MAX_COARSE MAX_FRAMES=$MAX_FRAMES RESUME_CKPT=$RESUME_CKPT OUT=$OUT SMOKE=$SMOKE"

echo "[vqvae-8card] $(date '+%F %T %Z') cross-NODE 8-card DDP: A=$JOB_A(node0/master) + B=$JOB_B(node1) via $RDZV_HOST:$RDZV_PORT smoke=$SMOKE"
echo "[vqvae-8card] global=$(( BATCH_SIZE*8 )) (8xbs$BATCH_SIZE) lr=$LR warmup_steps=$WARMUP_STEPS overwrite=$OVERWRITE amp=$AMP_DTYPE epochs=$EPOCHS"
echo "[vqvae-8card] anytop_root=$ANYTOP_ROOT max_joints=$MAX_JOINTS max_coarse=$MAX_COARSE max_frames=$MAX_FRAMES"
echo "[vqvae-8card] resume=${RESUME_CKPT:-<none>} out=$OUT"

# One torchrun group per alloc; static rendezvous joins them into 8 global ranks.
run_alloc() {
    local tag="$1" job="$2" noderank="$3"
    srun --jobid="$job" --overlap --nodes=1 --ntasks=1 \
      --gres=gpu:4 --cpus-per-task=32 --no-kill \
      bash -c "cd '$P' && NODE_RANK=$noderank $COMMON_ENV bash scripts/_launch_graph_vqvae.sh" \
      2>&1 | stdbuf -oL sed "s/^/[$tag] /"
}
# allocA = node_rank 0 (master on swarma1004, starts the TCPStore); allocB = node_rank 1 (2nd node).
run_alloc allocA "$JOB_A" 0 & PID_A=$!
run_alloc allocB "$JOB_B" 1 & PID_B=$!

wait "$PID_A"; RC_A=$?
wait "$PID_B"; RC_B=$?
echo "[vqvae-8card] $(date '+%F %T %Z') EXITED rc_A=$RC_A rc_B=$RC_B"
if [ "$RC_A" -ne 0 ] || [ "$RC_B" -ne 0 ]; then exit 1; fi
exit 0
