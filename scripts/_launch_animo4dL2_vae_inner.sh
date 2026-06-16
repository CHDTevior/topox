#!/bin/bash
# Inner per-node launcher for the animo4d-L2 bf16 rot6d-FK Graph-VAE.
# Runs ONE torchrun group on this node; the cross-node orchestrator
# (_launch_animo4dL2_vae_8card_xnode.sh) invokes this once per node with a
# shared MASTER_ADDR/PORT + explicit NODE_RANK to join them into one DDP world.
#
# Arg set is a BYTE-MATCH of the running 4-card config (read from that run's
# train.log args, git_sha ef1ed84): dataset anytop_truebones / anytop_root
# animo4d_anytop_clean_L2 / loss anytop13_world_rot6d_fk w0.25/1.0/0.10 /
# graphormer / coarse_xattn / edge_segment / anytop13 / d512 h8 dff1536 /
# n_graph4 n_enc2 n_cross3 n_dec2 n_treeik3 / max_coarse128 / local_radius8 /
# temporal_stride4 / max_frames64 / max_joints144 / use_name_embed / val_frac0.05
# / seed42 / amp bf16. The ONLY things that change for the 8-card resume are
# global batch (env, via DDP world size) and lr (env, 8e-4 = validated global384
# value) and --resume (continue, not fresh).
#
# train_graph_vae.py reads WORLD_SIZE/RANK/LOCAL_RANK from env (set by torchrun)
# for DDP, and supports --resume (model+optimizer+epoch+best-val). No warmup/
# scheduler exists in the train script (fixed lr by design, see its --resume help
# text) -> resume at a flat lr.
#
# Usage: invoked by the orchestrator. Direct single-node use also works:
#   CVD=0,1,2,3 NNODES=1 LR=8e-4 BS=48 OUT=runs/... RESUME_CKPT=runs/.../last_model.pt \
#     bash scripts/_launch_animo4dL2_vae_inner.sh
set -u
P="${P:-/scratch/ts1v23/workspace/noKslot_clean}"
cd "$P" || exit 1
PY=/scratch/ts1v23/.conda/bin/python3

CVD="${CVD:?set CVD (e.g. 0,1,2,3)}"
BS="${BS:-48}"                                   # per-GPU batch (= running run)
LR="${LR:?set LR (8e-4 = validated global384 value)}"
EPOCHS="${EPOCHS:-300}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
W_WORLD="${W_WORLD:-0.25}"; W_FK="${W_FK:-1.00}"; W_TRAJ="${W_TRAJ:-0.10}"
ANYTOP_ROOT="${ANYTOP_ROOT:-data/animo4d_anytop_clean_L2}"
VAL_FRAC="${VAL_FRAC:-0.05}"
SEED="${SEED:-42}"
OUT="${OUT:?set OUT}"
RESUME_CKPT="${RESUME_CKPT:-}"                   # continue from this ckpt (model+opt+epoch)
SMOKE="${SMOKE:-0}"                              # 1 -> add --smoke (5-iter preflight)
OVERWRITE="${OVERWRITE:-0}"                      # 0 = in-place resume into the run's OWN dir (no log truncate)

# Multi-node (cross-NODE) DDP via torchrun static rendezvous. NNODES=1 = single-node
# standalone (unchanged path). NNODES>1: orchestrator sets MASTER_ADDR (IB IP) +
# explicit NODE_RANK; node_rank 0 hosts the TCPStore.
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-29500}"

NPROC=$(echo "$CVD" | tr ',' '\n' | grep -c .)
GLOBAL=$(( NPROC * NNODES * BS ))

# Old-writer guard (runs ON each node via the orchestrator's srun): refuse to start
# if a train_graph_vae.py bound to THIS run dir is already alive on this node. The
# 8-card move replaces a previously-running 4-card run that used the SAME OUT; if the
# old run was not fully killed, two writers would race the same ckpts/logs. Skip for
# smoke (writes to /tmp, no collision) and use a bracket pattern so this guard does
# not match its own ssh/shell. (codex 019ea977 BLOCKER: lock only stops a 2nd
# orchestrator, not the old single-node writer.)
if [ "$SMOKE" != 1 ]; then
    _outbase="$(basename "$OUT")"
    if pgrep -f "[t]rain_graph_vae.py.*${_outbase}" >/dev/null 2>&1; then
        echo "[animo4dL2-vae] ABORT on $(hostname): a train_graph_vae.py for OUT=$_outbase is ALREADY running here — kill the old run first"; exit 3
    fi
fi

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="$CVD"

# torchrun launch mode: standalone (single node) vs static rendezvous (cross-node).
if [ "$NNODES" -gt 1 ]; then
    [ -z "$MASTER_ADDR" ] && { echo "[animo4dL2-vae] FAIL: NNODES>1 requires MASTER_ADDR"; exit 2; }
    # Cross-NODE: each node's 4 GPUs are ONE alloc (one cgroup) -> intra-node NVLink
    # P2P/SHM is safe and fast, so do NOT disable it (that was only for the same-node
    # cross-cgroup VQVAE case). Only inter-node hops use IB. NCCL_SOCKET_IFNAME picks
    # the IB iface that carries the 10.6.15.x IP (A100 nodes: ib0; H200 smoke: ib1).
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib0}"
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
    export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
    export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
    export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
    export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
    # Static rendezvous + explicit node_rank (c10d auto-host election fails when the
    # agent hostname != the IB rdzv host -> nobody hosts the store; verified upstream).
    RDZV_ARGS="--nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nproc_per_node=$NPROC"
else
    RDZV_ARGS="--standalone --nnodes=1 --nproc_per_node=$NPROC"
fi

SMOKE_FLAG=""; [ "$SMOKE" = 1 ] && SMOKE_FLAG="--smoke"
OVERWRITE_FLAG=""; [ "$OVERWRITE" = 1 ] && OVERWRITE_FLAG="--overwrite"
RESUME_FLAG=""; [ -n "$RESUME_CKPT" ] && RESUME_FLAG="--resume $RESUME_CKPT"

echo "[animo4dL2-vae] $(date '+%F %T %Z') host=$(hostname) CVD=$CVD nproc=$NPROC nnodes=$NNODES node_rank=$NODE_RANK"
echo "[animo4dL2-vae] bs=$BS lr=$LR global=$GLOBAL epochs=$EPOCHS amp=$AMP_DTYPE smoke=$SMOKE overwrite=$OVERWRITE"
echo "[animo4dL2-vae] root=$ANYTOP_ROOT out=$OUT resume=${RESUME_CKPT:-<none>}"
echo "[animo4dL2-vae] master=${MASTER_ADDR:-<standalone>}:$MASTER_PORT nccl_ifname=${NCCL_SOCKET_IFNAME:-<n/a>}"

torchrun $RDZV_ARGS scripts/train_graph_vae.py \
  --dataset anytop_truebones --feat_mode anytop13 --attn_mode graphormer \
  --decoder_mode coarse_xattn --pool_type edge_segment \
  --loss_mode anytop13_world_rot6d_fk --w_world "$W_WORLD" --w_fk "$W_FK" --w_traj "$W_TRAJ" \
  --anytop_root "$ANYTOP_ROOT" \
  --val_frac "$VAL_FRAC" --batch_size "$BS" --lr "$LR" --seed "$SEED" \
  --epochs "$EPOCHS" --save_every 5 --periodic_save_every 50 \
  --d_model 512 --n_heads 8 --d_ff 1536 \
  --n_graph_layers 4 --n_enc_temporal_layers 2 --n_cross_layers 3 \
  --n_dec_temporal_layers 2 --n_treeik_layers 3 \
  --max_coarse 128 --local_radius 8 --temporal_stride 4 \
  --max_frames 64 --max_joints 144 --use_name_embed \
  --amp_dtype "$AMP_DTYPE" \
  --out "$OUT" $OVERWRITE_FLAG $RESUME_FLAG $SMOKE_FLAG
rc=$?
echo "[animo4dL2-vae] $(date '+%F %T %Z') torchrun EXITED rc=$rc"
exit "$rc"
