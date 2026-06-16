#!/bin/bash
# Cross-alloc 4-card H100 DDP launcher (2 same-node allocs on swarmh1002).
# DIAGNOSTIC B-mu: latent temporal dynamics loss with --latent_dyn_target mu
# (deterministic posterior-mean latent trajectory as the dz/ddz reference) vs the
# main B run's `sample` target. Reuses the proven cross-alloc pattern (mem: 6-card
# same-node + 8-card xnode): each alloc's srun runs the SAME _launch_diffusion_t2m.sh
# with NNODES=2 + shared MASTER + explicit NODE_RANK; static rendezvous over IB.
#
# Adapted from _launch_diffusion_t2m_6card.sh (3-alloc) -> 2-alloc; COMMON_ENV adds
# the 20-species capacity whitelist + train_split=all + latent-dynamics-loss flags
# (the inner launcher already threads SPECIES_WHITELIST/TRAIN_SPLIT/W_LAT_*/LATENT_DYN_TARGET,
# both codex-PASSED 019e98dc / 019e9a10).
#
# Usage (SMOKE FIRST -- TRUE 4-rank, verify 2-alloc rendezvous + IB NCCL + bs no-OOM,
# 1 epoch; MUST pass before the real run):
#   SMOKE=1 NCCL_DEBUG=INFO bash scripts/_launch_diffusion_t2m_4card.sh 2>&1 | tee scripts/_smoke_t2m_4card.log
# Usage (real, DURABLE -- orchestrator ON a compute node so PPID=1):
#   ssh swarmh1002 "cd /scratch/ts1v23/workspace/noKslot_clean && setsid nohup bash scripts/_launch_diffusion_t2m_4card.sh > scripts/_train_latdyn_mu_4card.log 2>&1 < /dev/null &"
set -uo pipefail
P=/scratch/ts1v23/workspace/noKslot_clean
cd "$P" || exit 1

JOB_A="${JOB_A:-944460}"                 # swarmh1002, NODE_RANK 0 (master, starts TCPStore)
JOB_B="${JOB_B:-944461}"                 # swarmh1002, NODE_RANK 1
RDZV_HOST="${RDZV_HOST:-swarmh1002-ib0}"
RDZV_PORT="${RDZV_PORT:-29541}"          # distinct from 29501(6card)/29511(cap)/29531(B)
SMOKE="${SMOKE:-0}"
# 4-card global = PER_GPU_BATCH*4. Goyal off the pz20 low-LR line (global64->6.667e-5):
# global40 (4xbs10) -> lr 4.17e-5. bs10 smoke-tested no-OOM @64.8GB on H100 (6-card mem).
PER_GPU_BATCH="${PER_GPU_BATCH:-10}"
LR="${LR:-4.17e-5}"
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
LR_MIN="${LR_MIN:-0.0}"
WARMUP_ITERS="${WARMUP_ITERS:-400}"
EPOCHS="${EPOCHS:-1500}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
TEXT_MODE="${TEXT_MODE:-mean_additive}"
# 20-species capacity probe set (== B run); train on all clips, eval on val.
SPECIES_WHITELIST="${SPECIES_WHITELIST:-PZ_Koala_Female,PZ_Jaguar_Female,PZ_Siberian_Tiger_Juvenile,PZ_Ocelot_Female,PZ_Amur_Leopard_Juvenile,PZ_Cougar_Male,PZ_Bush_Dog_Male,PZ_Raccoon_Juvenile,PZ_Sun_Bear_Female,PZ_Formosan_Black_Bear_Male,PZ_Red_Panda_Male,PZ_Proboscis_Monkey_Juvenile,PZ_Hamadryas_Baboon_Male,PZ_Western_Chimpanzee_Male,PZ_Bonobo_Juvenile,PZ_Siamang_Male,PZ_Japanese_Macaque_Juvenile,PZ_King_Penguin_Male,PZ_Little_Penguin_Male,PZ_Hippopotamus_Male}"
TRAIN_SPLIT="${TRAIN_SPLIT:-all}"
# latent temporal dynamics loss -- B-mu: SAME weights as B, target=mu (vs B's sample).
W_LAT_DZ="${W_LAT_DZ:-0.05}"
W_LAT_DDZ="${W_LAT_DDZ:-0.02}"
W_LAT_X0="${W_LAT_X0:-0}"
LATENT_DYN_TARGET="${LATENT_DYN_TARGET:-mu}"
SPATIAL_MODE="${SPATIAL_MODE:-graph}"    # graph (default) | plain (no_graph_spatial ablation)
CAPTION_TOKEN_CACHE="${CAPTION_TOKEN_CACHE:-}"        # required when TEXT_MODE=token_cross_attn/dual_text
CAPTION_TOKEN_MAX_LEN="${CAPTION_TOKEN_MAX_LEN:-64}"
VAE_CKPT="${VAE_CKPT:-runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42/best_recon_model.pt}"
OUT="${OUT:-runs/m2_capacity_pz20_latdyn_dz005_ddz002_MUtarget_h100x4_lr4.17e-5cos_seed42}"
RESUME_CKPT="${RESUME_CKPT:-}"           # full crash-resume (model+opt+epoch+best_val+global_it); inner launcher passes --resume. cosine resume re-passes same lr_schedule/epochs (above).

# Single-instance lock (per-launch pgrep guard disabled for NNODES>1).
mkdir -p .aris/meta
exec 9>".aris/meta/.t2m4card.lock"
flock -n 9 || { echo "[t2m-4card] ABORT: already running"; exit 0; }

# NNODES=2 triggers the static-rendezvous branch in _launch_diffusion_t2m.sh;
# CVD=0,1 = each alloc's 2 local H100s. Same-node cross-cgroup -> inner launcher
# disables P2P/SHM + forces IB (proven on the 6-card same-node run).
COMMON_ENV="NNODES=2 NPROC_PER_NODE=2 MASTER_ADDR=$RDZV_HOST MASTER_PORT=$RDZV_PORT CVD=0,1 PER_GPU_BATCH=$PER_GPU_BATCH LR=$LR LR_SCHEDULE=$LR_SCHEDULE LR_MIN=$LR_MIN WARMUP_ITERS=$WARMUP_ITERS EPOCHS=$EPOCHS AMP_DTYPE=$AMP_DTYPE TEXT_MODE=$TEXT_MODE SPECIES_WHITELIST=$SPECIES_WHITELIST TRAIN_SPLIT=$TRAIN_SPLIT W_LAT_DZ=$W_LAT_DZ W_LAT_DDZ=$W_LAT_DDZ W_LAT_X0=$W_LAT_X0 LATENT_DYN_TARGET=$LATENT_DYN_TARGET SPATIAL_MODE=$SPATIAL_MODE CAPTION_TOKEN_CACHE=$CAPTION_TOKEN_CACHE CAPTION_TOKEN_MAX_LEN=$CAPTION_TOKEN_MAX_LEN VAE_CKPT=$VAE_CKPT OUT=$OUT SMOKE=$SMOKE RESUME_CKPT=$RESUME_CKPT"

echo "[t2m-4card] $(date '+%F %T %Z') cross-alloc 4-card H100 DDP: $JOB_A(r0)+$JOB_B(r1) via $RDZV_HOST:$RDZV_PORT smoke=$SMOKE"
echo "[t2m-4card] global=$(( PER_GPU_BATCH*4 )) (4xbs$PER_GPU_BATCH) lr=$LR sched=$LR_SCHEDULE/lr_min=$LR_MIN warmup=$WARMUP_ITERS epochs=$EPOCHS amp=$AMP_DTYPE"
echo "[t2m-4card] latdyn: w_lat_dz=$W_LAT_DZ w_lat_ddz=$W_LAT_DDZ w_lat_x0=$W_LAT_X0 TARGET=$LATENT_DYN_TARGET (B-mu diagnostic)"
echo "[t2m-4card] VAE=$VAE_CKPT text=$TEXT_MODE split=$TRAIN_SPLIT out=$OUT"

# One torchrun group per alloc; static rendezvous joins them into 4 global ranks.
run_alloc() {
    local tag="$1" job="$2" noderank="$3"
    srun --jobid="$job" --overlap --nodes=1 --ntasks=1 \
      --gres=gpu:2 --cpus-per-task=16 --no-kill \
      bash -c "cd '$P' && NODE_RANK=$noderank $COMMON_ENV bash scripts/_launch_diffusion_t2m.sh" \
      2>&1 | stdbuf -oL sed "s/^/[$tag] /"
}
# allocA = node_rank 0 (master, starts the TCPStore); allocB = 1.
run_alloc allocA "$JOB_A" 0 & PID_A=$!
run_alloc allocB "$JOB_B" 1 & PID_B=$!

wait "$PID_A"; RC_A=$?
wait "$PID_B"; RC_B=$?
echo "[t2m-4card] $(date '+%F %T %Z') EXITED rc_A=$RC_A rc_B=$RC_B"
if [ "$RC_A" -ne 0 ] || [ "$RC_B" -ne 0 ]; then exit 1; fi
exit 0
