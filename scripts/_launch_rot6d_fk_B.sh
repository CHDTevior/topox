#!/bin/bash
# scripts/_launch_rot6d_fk_B.sh
# rot6d-FK COMBINED geometry loss, arm B (user-decided 2026-06-01, "先 B，不先 C").
#
#   loss_mode = anytop13_world_rot6d_fk
#   L_total = L_anytop13_base + w_world*L_world_ric + w_fk*L_rot6d_fk + w_traj*L_root_traj
#   arm B weights = w_world 0.25 / w_fk 1.00 / w_traj 0.10
#     (POST-FIX calibration 2026-06-01, AFTER the double-root-rotation fix in
#      recover_rot6d_fk_positions_torch: gt_fk_mismatch 0.29->0.0000, raw fk≈1.13×
#      world (was buggy 1.82×). User decision: w_fk=1.0 (NOT 0.5) so FK
#      rotation-chain supervision is a REAL, visible training signal for
#      long-chain/wings. At w_fk=1.0: weighted fk=0.176=12.1% of base, total
#      geometry=0.229=15.7% of base — does NOT swamp main recon but is a hard FK
#      signal (w_fk=0.5 was only 6% base = too soft to test the FK hypothesis).
#      The earlier "preflight p95=29.86% / defer C=0.5/0.5/0.25" reasoning was
#      based on the BUGGY double-rotated FK and no longer applies. See
#      handoff/20260601_2102_rot6d_fk_double_rotation_fix.md.)
#
# A/B/B' comparability — loss is the SOLE experimental variable. Config replicates
# baseline A EXACTLY (verified from its ckpt args): edge_segment / coarse_xattn /
# graphormer / max_coarse128 / d512 h8 dff1536 / val_frac0.05 / lr4e-4 / seed42 /
# epochs300 / stride4 / frames64 / joints144 / use_name_embed.
#
# GLOBAL BATCH MATCH (user invariant): baseline A = 2×H200 × bs32 = global 64.
#   This arm B = 2×H100 × bs32 = global 64 → IDENTICAL global batch AND identical
#   per-GPU batch (32) as A, so lr stays 4e-4 (no Goyal scaling). Most comparable
#   config to A of all arms (world_geometry B used 4×A100×bs16, per-GPU 16).
#
# RESOURCE (verified 2026-06-01, NOT grabbing other projects' cards):
#   swarmh1002 alloc 944459 = MY 2×H100 (UUID GPU-8681af2f / GPU-38df6f29),
#   Slurm GRES cgroup-isolated. Node has 8×H100 total, shared with jb3c20 (2) and
#   mr21g23 (4) on the OTHER 6 physical cards — pam_slurm_adopt confirmed ssh
#   index 0,1 == my alloc's UUIDs. Do NOT touch swarma1001(world_geometry)/blossom04(diffusion).
#
# QA NOTE: compare best_model.pt (best-by-total, INCLUDES world/fk/traj) —
#   NOT best_recon_model.pt (recon-only).
#
# Usage (smoke — user HARD precondition: per-GPU bs32 must smoke-PASS before real):
#   SMOKE=1 CVD=0,1 bash scripts/_launch_rot6d_fk_B.sh
#     -> 2×H100 DDP, bs32, 5 iters; verifies DDP starts + bs32 no-OOM + loss branch.
# Usage (real, durable):
#   CVD=0,1 setsid nohup bash scripts/_launch_rot6d_fk_B.sh > LOG 2>&1 </dev/null &
set -u
P=/scratch/ts1v23/workspace/noKslot_clean
cd "$P" || exit 1

CVD="${CVD:-0,1}"
SMOKE="${SMOKE:-0}"
W_WORLD="${W_WORLD:-0.25}"
W_FK="${W_FK:-1.00}"
W_TRAJ="${W_TRAJ:-0.10}"
BS="${BS:-32}"
LR="${LR:-4.000e-04}"
OUT="${OUT:-runs/m1_l2_anytop13_edgeseg_C128_rot6dfk_w025f100t010_seed42}"

# Multi-node (cross-alloc) DDP via torchrun c10d rendezvous. Default NNODES=1 =
# single-alloc standalone (the unchanged 2-GPU path, lr 4e-4 global 64). The
# 4-card cross-alloc orchestrator (_launch_rot6d_fk_B_4card.sh) sets NNODES=2 +
# RDZV_ENDPOINT=swarmh1002-ib0:PORT + a shared RDZV_ID, BS=32, LR=8e-4 (Goyal
# linear scaling for global 128 vs the 2-card global-64 baseline).
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-29500}"

SMOKE_FLAG=""
if [ "$SMOKE" = 1 ]; then
    SMOKE_FLAG="--smoke"
    OUT="${OUT}_smoke"
    # NOTE: smoke keeps the FULL 2-GPU DDP + bs32 (real memory pressure) on purpose
    # — the user's precondition is a per-GPU bs32 OOM/DDP check, not a 1-GPU toy run.
fi
NPROC=$(echo "$CVD" | tr ',' '\n' | grep -c .)

# Guard: never double-launch the real run (single-alloc only). The cross-alloc
# 4-card run is managed by its orchestrator; same-node pgrep would otherwise
# false-match the peer alloc's rank and make each side self-abort.
if [ "$SMOKE" != 1 ] && [ "$NNODES" -le 1 ] && pgrep -f "train_graph_vae.py.*rot6dfk_w025f100t010" >/dev/null 2>&1; then
    echo "[fkB] ABORT: this rot6d_fk run already training"; exit 0
fi
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="$CVD"

# torchrun launch mode: standalone (single alloc) vs c10d rendezvous (cross-alloc).
if [ "$NNODES" -gt 1 ]; then
    [ -z "$MASTER_ADDR" ] && { echo "[fkB] FAIL: NNODES>1 requires MASTER_ADDR"; exit 2; }
    # cross-alloc multi-node DDP over IB (user-verified swarmh1002-ib0 reachable, 200G).
    # Same-node cross-cgroup: Slurm isolates P2P/SHM between the two allocs, so they
    # must be disabled or NCCL errors out; force comms onto the IB socket. NCCL_DEBUG
    # =INFO at smoke time confirms it uses ib0 + WORLD_SIZE=4 (codex 019e84f9).
    export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
    export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib0}"
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
    export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
    export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
    # Static rendezvous + explicit node_rank: node 0 is the unambiguous master that
    # starts the TCPStore. c10d auto-host election FAILED (smoke 2026-06-01) because
    # the agents' hostname (swarmh1002) != the IB rdzv host (swarmh1002-ib0), so
    # nobody hosted the store and both sides timed out as clients.
    RDZV_ARGS="--nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nproc_per_node=$NPROC"
else
    RDZV_ARGS="--standalone --nnodes=1 --nproc_per_node=$NPROC"
fi
GLOBAL=$(( NPROC * NNODES * BS ))

echo "[fkB] $(date '+%F %T %Z') host=$(hostname) CVD=$CVD nproc=$NPROC nnodes=$NNODES smoke=$SMOKE bs=$BS lr=$LR global=$GLOBAL"
echo "[fkB] loss=anytop13_world_rot6d_fk w_world=$W_WORLD w_fk=$W_FK w_traj=$W_TRAJ out=$OUT"
echo "[fkB] master=${MASTER_ADDR:-<standalone>}:$MASTER_PORT node_rank=$NODE_RANK nccl_ifname=${NCCL_SOCKET_IFNAME:-<n/a>}"

torchrun $RDZV_ARGS scripts/train_graph_vae.py \
  --dataset anytop_truebones --feat_mode anytop13 --attn_mode graphormer \
  --decoder_mode coarse_xattn --pool_type edge_segment \
  --loss_mode anytop13_world_rot6d_fk --w_world "$W_WORLD" --w_fk "$W_FK" --w_traj "$W_TRAJ" \
  --anytop_root "$P/data/anytop_planet_zoo_clean_L2" \
  --val_frac 0.05 --batch_size "$BS" --lr "$LR" --seed 42 \
  --epochs 300 --save_every 5 --periodic_save_every 50 \
  --d_model 512 --n_heads 8 --d_ff 1536 \
  --n_graph_layers 4 --n_enc_temporal_layers 2 --n_cross_layers 3 \
  --n_dec_temporal_layers 2 --n_treeik_layers 3 \
  --max_coarse 128 --local_radius 8 --temporal_stride 4 \
  --max_frames 64 --max_joints 144 --use_name_embed \
  --out "$OUT" --overwrite $SMOKE_FLAG
rc=$?
echo "[fkB] $(date '+%F %T %Z') torchrun EXITED rc=$rc"
exit "$rc"
