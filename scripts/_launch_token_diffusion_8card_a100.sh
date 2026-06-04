#!/bin/bash
# Cross-NODE 8-card A100 DDP launcher for the TOKEN-CROSS-ATTN diffusion PROBE.
# Two allocs on two physical nodes joined via static rendezvous → WORLD_SIZE=8:
#   944455 swarma1004 (4xA100, NODE_RANK 0 = master, starts TCPStore)
#   944456 swarma1001 (4xA100, NODE_RANK 1)
# Reuses the proven cross-alloc pattern (mem: same-node 6-card + xnode VAE):
# static rendezvous over IB, NCCL P2P/SHM disabled + IB forced, srun --overlap
# managed steps, rank-0-only ckpt, durable orchestrator on a compute node.
#
# Experiment: token-path PROBE (NOT a strict A/B). bf16 + ep209 best VAE +
# text_mode=token_cross_attn. Mean diffusion on H100 (swarmh1002) is the coarse
# baseline and is NOT touched here.
#
# Usage (SMOKE FIRST — cross-node rendezvous + token DDP grad-sync + bf16 no-OOM,
# 1 epoch; MUST pass before the real run):
#   SMOKE=1 NCCL_DEBUG=INFO bash scripts/_launch_token_diffusion_8card_a100.sh 2>&1 | tee scripts/_smoke_token_8card.log
# Usage (real, DURABLE — orchestrator ON a compute node so PPID=1):
#   ssh swarma1004 "cd /scratch/ts1v23/workspace/noKslot_clean && setsid nohup bash scripts/_launch_token_diffusion_8card_a100.sh > scripts/_train_token_8card.log 2>&1 < /dev/null &"
set -uo pipefail
P=/scratch/ts1v23/workspace/noKslot_clean
cd "$P" || exit 1

JOB_A="${JOB_A:-944455}"                 # swarma1004, NODE_RANK 0 (master)
JOB_B="${JOB_B:-944456}"                 # swarma1001, NODE_RANK 1
RDZV_HOST="${RDZV_HOST:-swarma1004-ib0}" # master = alloc A's IB host (verify reachable)
RDZV_PORT="${RDZV_PORT:-29511}"          # distinct from the H100 6-card run (29501)
SMOKE="${SMOKE:-0}"
# token cross-attn adds activation (scores [B,heads,T_lat*C,L]); start conservative,
# smoke-tune up. Goyal: global = PER_GPU_BATCH * 8, lr = 5e-4 * global / 48.
PER_GPU_BATCH="${PER_GPU_BATCH:-8}"
LR="${LR:-$(awk "BEGIN{printf \"%.3e\", 5e-4 * ($PER_GPU_BATCH*8) / 48}")}"
OUT="${OUT:-runs/m2_token_cleanL2_bf16ep209_d512C128_n11ff1536_a100x8_seed42}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
TEXT_MODE="${TEXT_MODE:-token_cross_attn}"
CAPTION_TOKEN_CACHE="${CAPTION_TOKEN_CACHE:-data/anytop_caption_t5_cleanL2_multi}"
CAPTION_TOKEN_MAX_LEN="${CAPTION_TOKEN_MAX_LEN:-64}"
# Frozen VAE = bf16 ep209 best (val_recon 1.3983), archived to main.
VAE_CKPT="${VAE_CKPT:-runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42/best_recon_model.pt}"

# Single-instance lock (cross-alloc: per-launch pgrep guard is disabled for NNODES>1).
mkdir -p .aris/meta
exec 9>".aris/meta/.token8card.lock"
flock -n 9 || { echo "[token-8card] ABORT: already running"; exit 0; }

# Shared env every alloc's launch inherits. NNODES=2 → static-rendezvous branch in
# _launch_diffusion_t2m.sh; CVD=0,1,2,3 = each alloc's 4 local A100s.
# NCCL (codex 019e94d2 P1): this is TRUE cross-NODE (not same-node cross-cgroup).
# Each node's 4 A100-SXM4 are NV4 NVLink → ENABLE intra-node P2P/SHM, overriding
# the _launch_diffusion_t2m.sh NNODES>1 defaults (P2P/SHM=disabled, which were for
# same-node cross-alloc cgroup isolation and would route intra-node collectives
# through slow host/NET). Matches the proven xnode VAE launcher. IB_HCA=mlx5_0
# (ibdev2netdev: mlx5_0->ib0 Up on both nodes).
COMMON_ENV="NNODES=2 NPROC_PER_NODE=4 MASTER_ADDR=$RDZV_HOST MASTER_PORT=$RDZV_PORT CVD=0,1,2,3 NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=0 NCCL_IB_HCA=mlx5_0 PER_GPU_BATCH=$PER_GPU_BATCH LR=$LR OUT=$OUT SMOKE=$SMOKE AMP_DTYPE=$AMP_DTYPE TEXT_MODE=$TEXT_MODE CAPTION_TOKEN_CACHE=$CAPTION_TOKEN_CACHE CAPTION_TOKEN_MAX_LEN=$CAPTION_TOKEN_MAX_LEN VAE_CKPT=$VAE_CKPT"

echo "[token-8card] $(date '+%F %T %Z') cross-node 8-card A100 DDP: $JOB_A(1004,r0)+$JOB_B(1001,r1) via $RDZV_HOST:$RDZV_PORT smoke=$SMOKE"
echo "[token-8card] text_mode=$TEXT_MODE amp=$AMP_DTYPE vae=$VAE_CKPT token_cache=$CAPTION_TOKEN_CACHE L=$CAPTION_TOKEN_MAX_LEN"
echo "[token-8card] global=$(( PER_GPU_BATCH*8 )) (8xbs$PER_GPU_BATCH) lr=$LR out=$OUT"

# One torchrun group per alloc; static rendezvous joins them into 8 global ranks.
# Explicit --gres/--cpus so each srun step gets its alloc's 4 GPUs + CPU for 4 ranks
# x dataloaders; --no-kill so one rank's transient failure does not tear down the step.
run_alloc() {
    local tag="$1" job="$2" noderank="$3"
    srun --jobid="$job" --overlap --nodes=1 --ntasks=1 \
      --gres=gpu:4 --cpus-per-task=32 --no-kill \
      bash -c "cd '$P' && NODE_RANK=$noderank $COMMON_ENV bash scripts/_launch_diffusion_t2m.sh" \
      2>&1 | stdbuf -oL sed "s/^/[$tag] /"
}
# allocA = NODE_RANK 0 (master, starts the TCPStore); allocB = 1.
run_alloc allocA "$JOB_A" 0 & PID_A=$!
run_alloc allocB "$JOB_B" 1 & PID_B=$!

wait "$PID_A"; RC_A=$?
wait "$PID_B"; RC_B=$?
echo "[token-8card] $(date '+%F %T %Z') EXITED rc_A=$RC_A rc_B=$RC_B"
if [ "$RC_A" -ne 0 ] || [ "$RC_B" -ne 0 ]; then exit 1; fi
exit 0
