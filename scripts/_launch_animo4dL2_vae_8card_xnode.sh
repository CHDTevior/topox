#!/bin/bash
# Cross-NODE 8-card A100 DDP orchestrator for the animo4d-L2 bf16 rot6d-FK Graph-VAE.
# Joins TWO physical nodes -- swarma1004 (alloc 944457, 4xA100, node_rank0/master) +
# swarma1001 (alloc 944458, 4xA100, node_rank1) -- into one 8-rank torchrun DDP world
# via STATIC rendezvous over IB. ADAPTED from the proven same-node cross-alloc
# _launch_graph_vqvae_6card.sh; the cross-NODE differences (vs the same-node VQVAE):
#   - 2 DIFFERENT physical nodes, one alloc per node (not 3 allocs on one node).
#   - WITHIN each node the 4 GPUs are ONE alloc (one cgroup) -> intra-node NVLink
#     P2P/SHM is safe + fast, so do NOT blanket-disable P2P/SHM. Only INTER-node
#     hops use IB. (Same-node-cross-cgroup disabling was a VQVAE-only workaround.)
#   - NCCL_SOCKET_IFNAME=ib0 (A100 nodes carry the 10.6.15.x IP on ib0; verified
#     swarma1004 ib0=10.6.15.68 / swarma1001 ib0=10.6.15.8, cross-node ping 0.2ms).
#
# Each node's srun runs scripts/_launch_animo4dL2_vae_inner.sh with NNODES=2 + the
# shared MASTER_ADDR/PORT + explicit NODE_RANK. Only global rank 0 (on swarma1004)
# writes ckpts (train_graph_vae.py is_main guard). global = 4 x 2 x BS(48) = 384.
# lr = 8e-4 (the VALIDATED global384 value; 2.4e-3 collapses to mean-pose). The
# train script has no warmup/scheduler (fixed lr by design) -> resume at flat 8e-4
# (VAE is robust at ep~57). --resume = continue model+optimizer+epoch (NOT fresh).
#
# SMOKE (true 4-rank cross-NODE on idle H200s; verify rendezvous + IB NCCL + a few
# steps -- WITHOUT touching the running A100 VAE). H200 nodes carry their 10.6.15.x
# IP on ib1 (NOT ib0), so override NCCL_SOCKET_IFNAME + MASTER_IB:
#   SMOKE=1 NCCL_DEBUG=INFO \
#     JOB_A=896281 JOB_B=976857 NODE_A=flamingo01 NODE_B=blossom03 \
#     MASTER_IB=10.6.15.127 NCCL_IFNAME=ib1 NPROC=2 \
#     OUT=/tmp/animo4dL2_vae_xnode_smoke MASTER_PORT=29507 \
#     bash scripts/_launch_animo4dL2_vae_8card_xnode.sh 2>&1 | tee scripts/_smoke_animo4dL2_vae_xnode.log
#
# REAL run (DURABLE) -- run the orchestrator ON the master compute node (PPID=1).
# NOTE: env-var assignments must go through `env` because `setsid nohup FOO=bar bash`
# would pass FOO=bar as argv to nohup (codex 019ea977 BLOCKER). RESUME_CKPT + OUT =
# the SAME run dir the 4-card run used (in-place continue):
#   ssh swarma1004 "cd /scratch/ts1v23/workspace/noKslot_clean && setsid nohup env \
#     RESUME_CKPT=runs/m1_animo4dL2_proxfiltered_bf16_rot6dfk_C128_d512_h8_ff1536_300ep_seed42/last_model.pt \
#     OUT=runs/m1_animo4dL2_proxfiltered_bf16_rot6dfk_C128_d512_h8_ff1536_300ep_seed42 \
#     bash scripts/_launch_animo4dL2_vae_8card_xnode.sh > scripts/_train_animo4dL2_vae_8card.log 2>&1 < /dev/null &"
set -uo pipefail
P=/scratch/ts1v23/workspace/noKslot_clean
cd "$P" || exit 1

# The two A100 allocs (verified 2026-06-08): swarma1004 944457 / swarma1001 944458.
JOB_A="${JOB_A:-944457}"          # node_rank 0 (master, hosts TCPStore)
JOB_B="${JOB_B:-944458}"          # node_rank 1
NODE_A="${NODE_A:-swarma1004}"
NODE_B="${NODE_B:-swarma1001}"
MASTER_IB="${MASTER_IB:-10.6.15.68}"     # swarma1004 ib0 (verified)
NCCL_IFNAME="${NCCL_IFNAME:-ib0}"        # A100 nodes: ib0. H200 smoke: ib1.
MASTER_PORT="${MASTER_PORT:-29506}"
SMOKE="${SMOKE:-0}"
NPROC="${NPROC:-4}"               # GPUs per node (A100=4; H200 smoke=2)
BS="${BS:-48}"                    # per-GPU batch
LR="${LR:-8.000e-04}"            # validated global384 value (NOT 2.4e-3 -> collapses)
AMP_DTYPE="${AMP_DTYPE:-bf16}"
EPOCHS="${EPOCHS:-300}"
W_WORLD="${W_WORLD:-0.25}"; W_FK="${W_FK:-1.00}"; W_TRAJ="${W_TRAJ:-0.10}"
ANYTOP_ROOT="${ANYTOP_ROOT:-data/animo4d_anytop_clean_L2}"
VAL_FRAC="${VAL_FRAC:-0.05}"
SEED="${SEED:-42}"
# OVERWRITE=1: the train script REFUSES a non-empty --out unless --overwrite
# (train_graph_vae.py:485-491). The real resume continues into the SAME (non-empty)
# run dir, so --overwrite is REQUIRED. It does NOT wipe the dir (no rmtree) and
# train.log/metrics/diagnostics open in append mode (line 501) -> existing ckpts +
# logs are PRESERVED; the run just overwrites last_model.pt on its own save cadence.
OVERWRITE="${OVERWRITE:-1}"
RESUME_CKPT="${RESUME_CKPT:-}"
OUT="${OUT:?set OUT (the run dir for the real run; /tmp/... for the smoke)}"

# Real-run safety: a real (non-smoke) run resumes into a non-empty dir with
# OVERWRITE=1, so a MISSING RESUME_CKPT would silently start from epoch 0 ON TOP of
# the existing run (the train script defaults start_epoch=0 without --resume). Hard-
# require RESUME_CKPT for real runs so an empty value cannot wipe-by-restart the run
# (codex 019ea977 BLOCKER).
if [ "$SMOKE" != 1 ]; then
    : "${RESUME_CKPT:?real run requires RESUME_CKPT (the run last_model.pt) - refusing to start epoch 0 on top of an existing run}"
    [ -f "$RESUME_CKPT" ] || { echo "[animo4dL2-8card] ABORT: RESUME_CKPT=$RESUME_CKPT does not exist"; exit 2; }
fi

# Single-instance lock (orchestrator runs on the master node). The inner launcher
# has NO pgrep double-launch guard (same-node pgrep would false-match a peer rank),
# so prevent a double orchestrator HERE (would share MASTER_PORT/OUT).
mkdir -p .aris/meta
exec 9>".aris/meta/.animo4dL2vae8card.lock"
flock -n 9 || { echo "[animo4dL2-8card] ABORT: already running"; exit 0; }

# Shared env every node's inner launch inherits. NNODES=2 -> static-rendezvous branch.
COMMON_ENV="NNODES=2 MASTER_ADDR=$MASTER_IB MASTER_PORT=$MASTER_PORT CVD=$(seq -s, 0 $((NPROC-1))) BS=$BS LR=$LR AMP_DTYPE=$AMP_DTYPE EPOCHS=$EPOCHS W_WORLD=$W_WORLD W_FK=$W_FK W_TRAJ=$W_TRAJ ANYTOP_ROOT=$ANYTOP_ROOT VAL_FRAC=$VAL_FRAC SEED=$SEED OVERWRITE=$OVERWRITE RESUME_CKPT=$RESUME_CKPT OUT=$OUT SMOKE=$SMOKE NCCL_SOCKET_IFNAME=$NCCL_IFNAME NCCL_IB_DISABLE=0 NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=0 NCCL_DEBUG=${NCCL_DEBUG:-WARN}"

echo "[animo4dL2-8card] $(date '+%F %T %Z') cross-NODE DDP: $JOB_A($NODE_A,rank0)+$JOB_B($NODE_B,rank1) via $MASTER_IB:$MASTER_PORT ifname=$NCCL_IFNAME nproc=$NPROC smoke=$SMOKE"
echo "[animo4dL2-8card] global=$(( NPROC*2*BS )) (${NPROC}x2xbs$BS) lr=$LR amp=$AMP_DTYPE epochs=$EPOCHS out=$OUT"
echo "[animo4dL2-8card] resume=${RESUME_CKPT:-<none>}"

# One srun step per node (into that node's alloc). These allocs are BATCH-step
# allocs (944457.batch / 944458.batch) -- the batch step already HOLDS the full
# GRES + CPU and exposes CUDA_VISIBLE_DEVICES to overlap steps. Re-requesting
# --gres/--cpus-per-task on an --overlap step into a batch-step alloc fails with
# "Requested nodes are busy" (the batch step owns them); verified 2026-06-08 on the
# H200 smoke. So DO NOT request --gres/--cpus here -- the inner launcher sets
# CUDA_VISIBLE_DEVICES=$CVD explicitly (= the alloc's GPUs). (The same-node VQVAE
# orchestrator could request --gres because those were interactive salloc allocs
# whose primary step does not hold all the GRES.) --no-kill so one rank's transient
# blip does not tear down the step. node_rank 0 hosts the TCPStore.
run_node() {
    local tag="$1" job="$2" node="$3" noderank="$4"
    srun --jobid="$job" --overlap --nodelist="$node" --nodes=1 --ntasks=1 --no-kill \
      bash -c "cd '$P' && NODE_RANK=$noderank $COMMON_ENV bash scripts/_launch_animo4dL2_vae_inner.sh" \
      2>&1 | stdbuf -oL sed "s/^/[$tag] /"
}
run_node nodeA "$JOB_A" "$NODE_A" 0 & PID_A=$!
run_node nodeB "$JOB_B" "$NODE_B" 1 & PID_B=$!

wait "$PID_A"; RC_A=$?
wait "$PID_B"; RC_B=$?
echo "[animo4dL2-8card] $(date '+%F %T %Z') EXITED rc_A=$RC_A rc_B=$RC_B"
if [ "$RC_A" -ne 0 ] || [ "$RC_B" -ne 0 ]; then exit 1; fi
exit 0
