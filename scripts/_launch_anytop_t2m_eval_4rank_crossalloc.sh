#!/usr/bin/env bash
# Manual-rank 4-process DDP launcher for the L4safe+HumanML3D AnyTop T2M evaluator across
# THREE same-node (swarmh1002) swarm_h100 allocs with HETEROGENEOUS gpu counts:
#   JOB_2GPU (gpu:2) -> global ranks 0,1   |  JOB_1A (gpu:1) -> rank 2  |  JOB_1B (gpu:1) -> rank 3
# torchrun --nproc_per_node cannot express 2+1+1, so we set the torch.distributed env vars by
# hand (WORLD_SIZE=4 + per-task RANK/LOCAL_RANK/MASTER_ADDR/MASTER_PORT) and let
# train_anytop_t2m_evaluator.py:setup_ddp() (env:// init) join the 4 ranks. Each srun step spawns
# $ngpu tasks; RANK = rank_base + SLURM_PROCID, LOCAL_RANK = SLURM_LOCALID (-> cuda:LOCAL_RANK).
# Same-node cross-cgroup => NCCL P2P/SHM OFF, force IB (ib0/mlx5_0). Run on swarmh1002 (setsid
# nohup for durability). NEVER scancel. Smoke first (SMOKE=1 / tiny manifests) before the real run.
set -uo pipefail
P=/scratch/ts1v23/workspace/noKslot_clean
cd "$P"
PY=/scratch/ts1v23/.conda/bin/python

# --- allocs / rendezvous (env-overridable) ---
JOB_2GPU="${JOB_2GPU:-976841}"             # global ranks 0,1
JOB_1A="${JOB_1A:-977959}"                 # global rank 2
JOB_1B="${JOB_1B:-977960}"                 # global rank 3
MASTER_ADDR="${MASTER_ADDR:-10.6.15.69}"   # swarmh1002 ib0 IP (rank-0 host)
MASTER_PORT="${MASTER_PORT:-29551}"
WORLD_SIZE=4

# --- data / recipe (env-overridable; defaults = prior contact-free 12ch recipe) ---
DATA="${DATA:-data/animo4d_anytop_clean_L4_safe_plus_humanml3d}"
OUT="${OUT:?set OUT=runs/anytop_t2m_evaluator_..._l4human_seed42}"
MANIFEST="${MANIFEST:-$DATA/eval_splits/train_main.json}"
VAL_MANIFEST="${VAL_MANIFEST:-$DATA/eval_splits/val_all.json}"
MOTION_FEAT_DIM="${MOTION_FEAT_DIM:-12}"   # CONTACT-FREE (default 13 in trainer => MUST pin 12)
COEMB="${COEMB:-512}"
BS="${BS:-32}"                             # per-rank; global = 4*BS = 128
LR="${LR:-1e-4}"
EPOCHS="${EPOCHS:-100}"
SEED="${SEED:-42}"
VAL_EVERY="${VAL_EVERY:-5}"
SAVE_EVERY="${SAVE_EVERY:-10}"
# SMOKE=1 => step-bounded smoke (default 40 steps) unless MAX_STEPS is given explicitly.
MAX_STEPS="${MAX_STEPS:-}"
[ "${SMOKE:-0}" = 1 ] && [ -z "$MAX_STEPS" ] && MAX_STEPS=40
MAX_STEPS="${MAX_STEPS:-0}"                # >0 stops early
NUM_WORKERS="${NUM_WORKERS:-4}"

MS_FLAG=""; [ "$MAX_STEPS" -gt 0 ] && MS_FLAG="--max_steps $MAX_STEPS"

NCCL_ENV="NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_SOCKET_IFNAME=ib0 NCCL_IB_DISABLE=0 NCCL_IB_HCA=mlx5_0 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false"

TRAIN_ARGS="--manifest $MANIFEST --val_manifest $VAL_MANIFEST --data_root $DATA \
--text_tower distilbert --motion_feat_dim $MOTION_FEAT_DIM --coemb_dim $COEMB \
--batch_size $BS --lr $LR --epochs $EPOCHS --seed $SEED --bf16 --num_frames 300 --max_joints 144 \
--val_every $VAL_EVERY --val_max_batches 0 --val_batch_size 32 --num_workers $NUM_WORKERS \
--out $OUT --save_every $SAVE_EVERY $MS_FLAG"

# one srun step per alloc; srun spawns $ngpu tasks; each task derives its global RANK + LOCAL_RANK.
run_group() {  # $1=jobid $2=ngpu $3=rank_base $4=tag
  srun --jobid="$1" --overlap --nodes=1 --ntasks="$2" --ntasks-per-node="$2" \
       --gres=gpu:"$2" --cpus-per-task=8 --no-kill \
    bash -c "export RANK=\$(( $3 + SLURM_PROCID )) LOCAL_RANK=\${SLURM_LOCALID:-0} WORLD_SIZE=$WORLD_SIZE MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT $NCCL_ENV; cd '$P'; echo \"[rank \$RANK localrank \$LOCAL_RANK] host=\$(hostname) cvd=\${CUDA_VISIBLE_DEVICES:-?}\"; exec $PY scripts/train_anytop_t2m_evaluator.py $TRAIN_ARGS" \
    2>&1 | stdbuf -oL sed "s/^/[$4] /"
}

echo "[eval-4rank] WORLD_SIZE=4 master=$MASTER_ADDR:$MASTER_PORT out=$OUT"
echo "[eval-4rank] mfd=$MOTION_FEAT_DIM coemb=$COEMB bs=$BS(x4=global$((BS*4))) lr=$LR epochs=$EPOCHS seed=$SEED val_every=$VAL_EVERY max_steps=$MAX_STEPS"
echo "[eval-4rank] allocs: 2gpu=$JOB_2GPU(r0,r1) 1a=$JOB_1A(r2) 1b=$JOB_1B(r3) | manifest=$MANIFEST"
run_group "$JOB_2GPU" 2 0 "a2g" & PA=$!
run_group "$JOB_1A"   1 2 "b1a" & PB=$!
run_group "$JOB_1B"   1 3 "c1b" & PC=$!
wait $PA; ra=$?
wait $PB; rb=$?
wait $PC; rc=$?
echo "[eval-4rank] EXITED a2g(r0,1)=$ra b1a(r2)=$rb c1b(r3)=$rc"
if [ "$ra" -ne 0 ] || [ "$rb" -ne 0 ] || [ "$rc" -ne 0 ]; then exit 1; fi
exit 0
