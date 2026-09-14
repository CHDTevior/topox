#!/bin/bash
# Cross-NODE 4-card H200 DDP orchestrator for the v2 in-context trainer.
#
# TWO SEPARATE NODES, one 2-GPU alloc each (user 2026-08-21: "用H200，2 alloc"). That geometry
# differs from the same-node cross-alloc playbook in exactly one way that matters:
#   * within an alloc both H200 sit in ONE cgroup -> NVLink P2P/SHM MUST STAY ON. The
#     P2P_DISABLE=1/SHM_DISABLE=1 in the same-node cross-alloc launchers is medicine for two
#     cgroups sharing a node; copying it here would disable NVLink inside each pair for nothing.
#   * only the inter-node ring goes over IB. Verified on these hosts 2026-08-21:
#     blossom02 ib1=10.6.15.131, flamingo02 ib1=10.6.15.128, same /22, mlx5_1 ACTIVE
#     (mlx5_0 is DOWN on both -- do not fall back to ib0/mlx5_0).
#   * RDZV host is the master's IB IP, never a hostname: torchrun's c10d host election dies on the
#     hostname-vs-ib-alias mismatch, so this uses STATIC rendezvous with explicit node_rank.
#
# The alloc walltime (13-19h) is far shorter than the run (500 epochs ~ 5 days), so this WILL be
# resumed many times. `--resume` is passed through; the trainer restores optimizer, RNG, gstep,
# best_val and the articulation-gate strike count, and refuses to resume across a data-pin change.
#
# SMOKE (verify WORLD_SIZE=4, rendezvous, NCCL via NET/IB, then exit):
#   SMOKE=1 OUT=/tmp/v2_2node_smoke bash scripts/_launch_v2_ddp_2node_h200.sh
#
# REAL (DURABLE -- run ON the master node so the srun clients survive ssh teardown):
#   ssh blossom02 "cd /scratch/ts1v23/workspace/noKslot_clean && mkdir -p runs/<OUT> && setsid \
#     nohup env JOB_A=.. JOB_B=.. MASTER_NODE=blossom02 WORKER_NODE=flamingo02 \
#     RDZV_HOST=10.6.15.131 OUT=runs/<OUT> EPOCHS=500 LR=1.2e-3 BATCH=8 EXTRA='...' \
#     bash scripts/_launch_v2_ddp_2node_h200.sh > runs/<OUT>/orchestrator.log 2>&1 </dev/null &"
set -uo pipefail
P=/scratch/ts1v23/workspace/noKslot_clean
cd "$P" || exit 1

# SINGLE SOURCE OF TRUTH. The 4-rank gate sources this same file, so a gate PASS is evidence about
# the configuration that actually launches (codex 2026-08-23 BLOCKING 1). Every value inside is
# written `${VAR:-default}`, so anything already in the environment -- the watchdog's fresh alloc
# ids and node names after a card change -- still wins over the file.
CFG=${CFG:-configs/run6_env.sh}
[ -f "$CFG" ] || { echo "[orch] PREFLIGHT FAIL: config $CFG not found"; exit 1; }
set -a; . "$CFG"; set +a

JOB_A=${JOB_A:?master alloc jobid (node_rank 0)}
JOB_B=${JOB_B:?worker alloc jobid (node_rank 1)}
MASTER_NODE=${MASTER_NODE:?node hosting JOB_A}
WORKER_NODE=${WORKER_NODE:?node hosting JOB_B}
RDZV_HOST=${RDZV_HOST:?master IB IP (not a hostname)}
RDZV_PORT=${RDZV_PORT:-29531}
GPUS_PER=${GPUS_PER:-2}
OUT=${OUT:?output dir}
EPOCHS=${EPOCHS:-500}
LR=${LR:?Goyal-scaled for global batch = BATCH x 2 x GPUS_PER}
BATCH=${BATCH:-8}
# These H200 allocs carry 8 CPUs each. Requesting more does not queue -- srun retries forever
# with "step creation temporarily disabled", which looks exactly like a hung launch (cost me a
# 28-minute phantom calibration on 2026-08-21). Keep the request inside the alloc.
CPUS=${CPUS:-8}
RESUME=${RESUME:-}
# Data + objective bindings are REQUIRED, never optional EXTRA (codex 2026-08-21 (A)5): the
# trainer's own defaults point at the RETIRED TrueBones corpus, the old per-cell stats and
# huber_delta=0, so a dropped EXTRA would not fail -- it would quietly train a different run.
KTJD_ROOT=${KTJD_ROOT:?corpus root}
PERCELL=${PERCELL:?per-cell normalization stats npz}
CALIB=${CALIB:?gamma calibration artifact}
CUT=${CUT:?clip-exclusion artifact}
HUBER=${HUBER:?huber knee -- must equal protocol.huber_delta in the calibration}
# Architecture and schedule are REQUIRED too (codex 2026-08-23 blocker 4): the trainer's own
# defaults are dim256/depth6, uncompiled, flat-lr. Passing them through EXTRA means a dropped
# string launches a completely different (and silently smaller) run that still looks valid.
DIM=${DIM:?model width}
DEPTH=${DEPTH:?model depth}
LR_SCHED=${LR_SCHED:?lr_scheduler: half_cosine|none}
LR_DECAY_EPOCHS=${LR_DECAY_EPOCHS:?cosine horizon in epochs (0 = full --epochs)}
ETA_MIN_RATIO=${ETA_MIN_RATIO:?cosine floor as a fraction of LR}
WARMUP=${WARMUP:?warmup steps}
WD=${WD:?weight decay}
GRAD_CLIP=${GRAD_CLIP:?gradient clip}
# Reject (not clip) post-warmup gradient spikes. Bound because a dropped value silently restores
# the failure mode that killed six runs; clipping alone does not stop it.
GRAD_SPIKE=${GRAD_SPIKE:?pre-clip gradient norm above which a step is REJECTED (0 disables)}
PARAM_RESYNC_STEPS=${PARAM_RESYNC_STEPS:?broadcast rank-0 parameters every N steps (0 disables)}
QK_NORM=${QK_NORM:?1 or 0 -- RMS-normalise q,k per head (architecture-defining)}
# strict 0/1: "true"/"2"/"01" would expand $([ = 1 ]) to NOTHING and silently train the OLD
# architecture while the config says otherwise (codex 2026-08-26 blocker 2)
case "$QK_NORM" in 0|1) ;; *) echo "[orch] QK_NORM must be exactly 0 or 1, got '$QK_NORM'"; exit 1 ;; esac
SIGMA_MIN=${SIGMA_MIN:?v_space weight floor}
GRAD_CKPT=${GRAD_CKPT:?1 or 0 -- activation checkpointing}
COMPILE=${COMPILE:?1 or 0 -- torch.compile}
JOINT_SEM=${JOINT_SEM:?joint-semantics npz}
CAPTION_CACHE=${CAPTION_CACHE:?LLM2Vec caption-embedding cache prefix}
TEXTS_JSON=${TEXTS_JSON:?caption text table the cache was built from}
# OBJECTIVE + ARCHITECTURE bindings (codex 2026-08-23 final gate, BLOCKING 1 and 2). Every one of
# these has a trainer default that is NOT what this project trains: gamma_fk/gamma_vel/gamma_lock
# default to 0.0, and v_space/bf16/demo_rest/struct_feats/dir_bias default to OFF. Passing them
# through EXTRA meant a dropped string launched a DIFFERENT objective that still looked valid --
# e.g. SIGMA_MIN would be inert because the v-space loss it floors was never enabled.
HEADS=${HEADS:?attention heads}
BF16=${BF16:?1 or 0 -- bf16 autocast}
V_SPACE=${V_SPACE:?1 or 0 -- JiT velocity-space loss (SIGMA_MIN is inert when 0)}
GAMMA_FK=${GAMMA_FK:?FK auxiliary weight}
FK_WARMUP=${FK_WARMUP:?FK warmup steps}
GAMMA_VEL=${GAMMA_VEL:?velocity auxiliary weight}
GAMMA_ACC=${GAMMA_ACC:?acceleration-matching weight (0 disables)}
GAMMA_LOCK=${GAMMA_LOCK:?foot-lock auxiliary weight}
DEMO_REST=${DEMO_REST:?1 or 0 -- rest-pose demo}
DEMO_FRAMES=${DEMO_FRAMES:?demo frame count}
STRUCT_FEATS=${STRUCT_FEATS:?1 or 0 -- graph-v2 structural features}
DIR_BIAS=${DIR_BIAS:?1 or 0 -- graph-v2 directional bias}
RANDOM_CAPTION=${RANDOM_CAPTION:?1 or 0 -- caption rotation}
ANCHOR=${ANCHOR:?none|rest|demo}
P_DROP_TEXT=${P_DROP_TEXT:?CFG text-drop probability}
P_DROP_DEMO=${P_DROP_DEMO:?CFG demo-drop probability}
P_DROP_BOTH=${P_DROP_BOTH:?CFG both-drop probability}
T_SAMPLER=${T_SAMPLER:?uniform|logitnormal}
IDLE_MIB=${IDLE_MIB:-1024}
NCCL_IFACE=${NCCL_SOCKET_IFNAME:-ib1}
NCCL_HCA=${NCCL_IB_HCA:-mlx5_1}
SMOKE=${SMOKE:-0}
EXTRA=${EXTRA:-}
case "$EXTRA" in *\'*|*\"*) echo "[orch] PREFLIGHT FAIL: EXTRA must not contain quotes"; exit 1;; esac
for dup in --out --epochs --lr --batch --resume --corpus --ktjd_root --joint_sem \
           --caption_cache --texts_json \
           --ktjd_percell_stats --ktjd_gamma_calib --exclude_clips --huber_delta \
           --dim --depth --lr_scheduler --lr_decay_epochs --eta_min_ratio --warmup_steps \
           --wd --grad_clip --grad_spike_reject --param_resync_steps --sigma_min --grad_ckpt --compile \
           --heads --bf16 --v_space --gamma_fk --fk_warmup_steps --gamma_vel --gamma_lock --gamma_acc --qk_norm \
           --demo_rest --demo_frames --struct_feats --dir_bias --random_caption --anchor \
           --p_drop_text --p_drop_demo --p_drop_both --t_sampler; do
  case " $EXTRA " in *" $dup "*|*" $dup="*) echo "[orch] PREFLIGHT FAIL: EXTRA must not set $dup"; exit 1;; esac
done

mkdir -p "$OUT" .aris/meta
# Lock on the ALLOC PAIR: a second orchestrator on another port would double-launch these ranks.
exec 9>".aris/meta/.v2_2node_${JOB_A}_${JOB_B}.lock"
flock -n 9 || { echo "[orch] lock held for ${JOB_A}+${JOB_B} -- refusing double launch"; exit 1; }
# PID file written only AFTER the lock is held: the watchdog confirms a relaunch by PID, which a
# process-name match cannot do safely (the launching ssh wrapper carries this script in its argv).
PIDFILE=${PIDFILE:-.aris/meta/.v2_2node_orch.pid}
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

# ---- preflight ----
[ "$(hostname -s)" = "$MASTER_NODE" ] \
  || { echo "[orch] PREFLIGHT FAIL: orchestrator must run ON $MASTER_NODE, is on $(hostname -s)"; exit 1; }
check() {  # jobid expected_node
  local st; st=$(squeue -j "$1" -h -o "%T %N" 2>/dev/null)
  [ -n "$st" ] || { echo "[orch] PREFLIGHT FAIL: job $1 not found"; return 1; }
  set -- $st "$2"
  [ "$1" = RUNNING ] || { echo "[orch] PREFLIGHT FAIL: job state $1"; return 1; }
  [ "$2" = "$3" ] || { echo "[orch] PREFLIGHT FAIL: job on $2, expected $3"; return 1; }
}
check "$JOB_A" "$MASTER_NODE" || exit 1
check "$JOB_B" "$WORKER_NODE" || exit 1
for J in "$JOB_A" "$JOB_B"; do
  ncpu=$(scontrol show job "$J" | grep -oP 'NumCPUs=\K[0-9]+' | head -1)
  [ -n "$ncpu" ] && [ "$ncpu" -ge "$CPUS" ] \
    || { echo "[orch] PREFLIGHT FAIL: alloc $J has ${ncpu:-?} CPUs, CPUS=$CPUS requested "\
              "(srun would retry forever instead of failing)"; exit 1; }
  # a memory census that fails to answer counts as occupied (codex heldout r3 P3: a non-zero exit with partial rows passed);
  # its stderr stays out of the rows (r4 P3: a diagnostic line would count as a GPU) and lands in the orchestrator log
  info=$(srun --jobid="$J" --overlap -n1 -N1 nvidia-smi --query-gpu=index,memory.used \
         --format=csv,noheader,nounits); st=$?
  [ $st -eq 0 ] || { echo "[orch] PREFLIGHT FAIL: alloc $J GPU memory census failed (status $st): $info"; exit 1; }
  [ "$(echo "$info" | wc -l)" -ge "$GPUS_PER" ] \
    || { echo "[orch] PREFLIGHT FAIL: alloc $J has fewer than $GPUS_PER GPUs"; exit 1; }
  busy=$(echo "$info" | awk -F', ' -v m="$IDLE_MIB" '$2+0>m{print $1":"$2}' | tr '\n' ' ')
  [ -z "$busy" ] || { echo "[orch] PREFLIGHT FAIL: alloc $J GPUs in use: $busy"; exit 1; }
  # any compute process on the node is "in use", whatever its memory (codex heldout r2 P2-1: a 512 MiB process passed the
  # memory test); a census that fails to answer counts as occupied (user rule: 查询失败一律当作被占用)
  apps=$(srun --jobid="$J" --overlap -n1 -N1 nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>&1); st=$?
  [ $st -eq 0 ] || { echo "[orch] PREFLIGHT FAIL: alloc $J compute-apps census failed (status $st): $apps"; exit 1; }
  [ -z "$apps" ] || { echo "[orch] PREFLIGHT FAIL: alloc $J has compute processes on its GPUs: $(echo "$apps" | tr '\n' ' ')"; exit 1; }
done
ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":${RDZV_PORT}\$" \
  && { echo "[orch] PREFLIGHT FAIL: port $RDZV_PORT already bound on $MASTER_NODE"; exit 1; }
[ -z "$RESUME" ] || [ -f "$RESUME" ] \
  || { echo "[orch] PREFLIGHT FAIL: --resume target $RESUME does not exist"; exit 1; }

SMOKE_ARGS=""
# the smoke must still take real optimizer steps: with drop_last the trainer refuses a
# clip budget below one global batch (codex 2026-09-02 P0-3), so the cap scales with BATCH x ranks x 2
# --epoch_draws is stated by the arms that must match another arm's epoch, and the balanced sampler draws
# with replacement, so a smoke that kept the real value would run the real number of steps (578, not two) over
# --limit_train_clips clips: the budget the smoke exists to bound (codex 2026-09-10 r2 P2-3). SMOKE_ARGS is
# placed AFTER $EXTRA on the command line, so this occurrence is the one argparse keeps.
[ "$SMOKE" = 1 ] && SMOKE_ARGS="--epochs 1 --val_every 1000 --ckpt_every 1000 --limit_train_clips $(( BATCH * GPUS_PER * 2 * 2 )) --epoch_draws $(( BATCH * GPUS_PER * 2 * 2 ))"
RES_ARG=""; [ -n "$RESUME" ] && RES_ARG="--resume $RESUME"

echo "[orch] $(date -u +%FT%TZ) master=$MASTER_NODE($JOB_A) worker=$WORKER_NODE($JOB_B)"
echo "[orch] rdzv=$RDZV_HOST:$RDZV_PORT nccl=$NCCL_IFACE/$NCCL_HCA (intra-node P2P/SHM ON)"
echo "[orch] world=$((2*GPUS_PER)) batch=$BATCH/rank -> global $((2*GPUS_PER*BATCH)) lr=$LR epochs=$EPOCHS"
echo "[orch] out=$OUT resume=${RESUME:-<none>} smoke=$SMOKE"
echo "[orch] root=$KTJD_ROOT"
echo "[orch] percell=$PERCELL calib=$CALIB"
echo "[orch] captions=$CAPTION_CACHE texts=$TEXTS_JSON"
echo "[orch] cut=$CUT huber=$HUBER joint_sem=$JOINT_SEM"
echo "[orch] arch dim=$DIM depth=$DEPTH grad_ckpt=$GRAD_CKPT compile=$COMPILE qk_norm=$QK_NORM"
echo "[orch] sched=$LR_SCHED decay_ep=$LR_DECAY_EPOCHS eta_min=$ETA_MIN_RATIO warmup=$WARMUP"
echo "[orch] wd=$WD grad_clip=$GRAD_CLIP spike_reject=$GRAD_SPIKE sigma_min=$SIGMA_MIN v_space=$V_SPACE bf16=$BF16"
echo "[orch] gamma fk=$GAMMA_FK/warm$FK_WARMUP vel=$GAMMA_VEL lock=$GAMMA_LOCK acc=$GAMMA_ACC t_sampler=$T_SAMPLER"
echo "[orch] demo_rest=$DEMO_REST frames=$DEMO_FRAMES struct=$STRUCT_FEATS dir_bias=$DIR_BIAS"
echo "[orch] heads=$HEADS anchor=$ANCHOR rand_cap=$RANDOM_CAPTION drops=$P_DROP_TEXT/$P_DROP_DEMO/$P_DROP_BOTH"
for f in "$PERCELL" "$CALIB" "$CUT" "$JOINT_SEM"; do
  [ -f "$f" ] || { echo "[orch] PREFLIGHT FAIL: missing artifact $f"; exit 1; }
done
[ -d "$KTJD_ROOT" ] || { echo "[orch] PREFLIGHT FAIL: corpus root $KTJD_ROOT not a directory"; exit 1; }
CH=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['protocol'].get('huber_delta',0.0))" "$CALIB")
python3 -c "import sys;sys.exit(0 if abs(float(sys.argv[1])-float(sys.argv[2]))<1e-9 else 1)" "$CH" "$HUBER" \
  || { echo "[orch] PREFLIGHT FAIL: calibration huber_delta=$CH but HUBER=$HUBER"; exit 1; }

NCCL_ENV="NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=0 NCCL_IB_DISABLE=0 \
NCCL_SOCKET_IFNAME=$NCCL_IFACE NCCL_IB_HCA=$NCCL_HCA TORCH_NCCL_ASYNC_ERROR_HANDLING=1"
[ "$SMOKE" = 1 ] && NCCL_ENV="$NCCL_ENV NCCL_DEBUG=INFO"

run_rank() {  # $1 jobid  $2 node_rank
  srun --jobid="$1" --overlap --ntasks=1 --gres=gpu:${GPUS_PER} --cpus-per-task="$CPUS" --no-kill \
    bash -c "cd $P && env $NCCL_ENV torchrun \
      --nnodes=2 --nproc_per_node=${GPUS_PER} --node_rank=$2 \
      --master_addr=$RDZV_HOST --master_port=$RDZV_PORT \
      scripts/train_v2_incontext.py $EXTRA \
      --corpus ktjd17 --ktjd_root $KTJD_ROOT --joint_sem $JOINT_SEM \
      --caption_cache $CAPTION_CACHE --texts_json $TEXTS_JSON \
      --ktjd_percell_stats $PERCELL --ktjd_gamma_calib $CALIB --exclude_clips $CUT \
      --huber_delta $HUBER --sigma_min $SIGMA_MIN \
      --dim $DIM --depth $DEPTH --heads $HEADS \
      --warmup_steps $WARMUP --wd $WD --grad_clip $GRAD_CLIP --grad_spike_reject $GRAD_SPIKE --param_resync_steps $PARAM_RESYNC_STEPS \
      --gamma_fk $GAMMA_FK --fk_warmup_steps $FK_WARMUP \
      --gamma_vel $GAMMA_VEL --gamma_lock $GAMMA_LOCK --gamma_acc $GAMMA_ACC \
      --demo_frames $DEMO_FRAMES --anchor $ANCHOR --t_sampler $T_SAMPLER \
      --p_drop_text $P_DROP_TEXT --p_drop_demo $P_DROP_DEMO --p_drop_both $P_DROP_BOTH \
      $([ "$BF16" = 1 ] && echo --bf16) $([ "$V_SPACE" = 1 ] && echo --v_space) \
      $([ "$DEMO_REST" = 1 ] && echo --demo_rest) \
      $([ "$STRUCT_FEATS" = 1 ] && echo --struct_feats) $([ "$DIR_BIAS" = 1 ] && echo --dir_bias) \
      $([ "$RANDOM_CAPTION" = 1 ] && echo --random_caption) \
      --lr_scheduler $LR_SCHED --lr_decay_epochs $LR_DECAY_EPOCHS --eta_min_ratio $ETA_MIN_RATIO \
      $([ "$GRAD_CKPT" = 1 ] && echo --grad_ckpt) $([ "$COMPILE" = 1 ] && echo --compile) \
      $([ "$QK_NORM" = 1 ] && echo --qk_norm) \
      --out $OUT --epochs $EPOCHS --lr $LR --batch $BATCH $RES_ARG $SMOKE_ARGS" \
    2>&1 | stdbuf -oL sed "s/^/[r$2] /"
}

# APPEND, never truncate. A resume used to open these with '>' and wipe the log of the instance
# that had just died -- which is exactly the evidence needed to explain why it died (run5,
# 2026-08-22: the crash cause was destroyed by its own recovery). A banner separates instances.
for L in "$OUT/orch_rank0.log" "$OUT/orch_rank1.log"; do
  { echo; echo "===== launch $(date -u +%FT%TZ) resume=${RESUME:-<none>} pid=$$ ====="; } >> "$L"
done
run_rank "$JOB_A" 0 >> "$OUT/orch_rank0.log" 2>&1 &
PID0=$!
run_rank "$JOB_B" 1 >> "$OUT/orch_rank1.log" 2>&1 &
PID1=$!
echo "[orch] rank0 pid $PID0 -> $OUT/orch_rank0.log | rank1 pid $PID1 -> $OUT/orch_rank1.log"
wait $PID0; RC0=$?
wait $PID1; RC1=$?
echo "[orch] $(date -u +%FT%TZ) exit rc rank0=$RC0 rank1=$RC1"
exit $(( RC0 | RC1 ))
