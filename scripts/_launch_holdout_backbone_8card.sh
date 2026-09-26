#!/bin/bash
# THE single source of truth for the holdout LLM2Vec backbone run: 8xH100 cross-alloc DDP
# (4 same-node allocs x 2 GPUs on swarmh1001), graph_pscf + LLM2Vec-4096 text architecture,
# strict unseen-topology protocol, ep150 semantic VQVAE frozen tokenizer.
#
# VERBATIM structure of scripts/_launch_graph_pscf_6card.sh (the proven cross-alloc pattern)
# extended to 4 allocs; every EXPERIMENT-DEFINING value is pinned HERE, not passed by callers.
# The watchdog resumes by re-running this file with RESUME_CKPT=last_model.pt OVERWRITE=0 and
# nothing else, so no orchestration layer can drop a flag (codex round-4 #1/#3).
#
# GEN_EVAL=0 DELIBERATELY: the only trained T2M evaluator saw the held-out topologies, so the
# strict protocol refuses it as an online instrument (and it would be scientifically wrong).
# Online gen-eval stays off until a retained-topology evaluator exists; health = loss curves +
# offline renders. GEN_EVAL_CAPTION_CACHE/MANIFEST are pre-wired for when that lands.
#
# Smoke:  SMOKE=1 OUT=/tmp/gpscf_holdout_smoke bash scripts/_launch_holdout_backbone_8card.sh
# Real:   OUT=runs/holdout_backbone_llm2vec_8card_v2 bash scripts/_launch_holdout_backbone_8card.sh
#         (v2 = caption_sampling=random from step 0; v1 was primary-only, stopped at ep76)
set -uo pipefail
P=/scratch/ts1v23/workspace/noKslot_clean
cd "$P"

# ---- allocs (4x 2xH100 on swarmh1001) ----
JOB_A="${JOB_A:-1332170}"
JOB_B="${JOB_B:-1294991}"
JOB_C="${JOB_C:-1294992}"
JOB_D="${JOB_D:-1332171}"
RDZV_HOST="${RDZV_HOST:-swarmh1001-ib0}"
RDZV_PORT="${RDZV_PORT:-29517}"

# ---- experiment contract (PINNED; the config digest hashes what the trainer receives) ----
BATCH_SIZE="${BATCH_SIZE:-8}"   # x8 GPUs (x accum) = global 64 — same optimisation problem as the v4b
GRAD_ACCUM="${GRAD_ACCUM:-1}"   # B4+accum2 keeps global 64 with half the activation memory (dec-loss runs)
LR=8e-5                         # flagship (global64/lr8e-5); B8 fits per the A100 C96 precedent
EPOCHS="${EPOCHS:-300}"
WARMUP_STEPS=2000
SEED=42
TOKEN_CACHE=data/codeflow_tokens_holdout_semantic_ep150_fulllen300
FROZEN_CKPT=runs/holdout_vqvae_semantic_8card_v1/ep150_model.pt
TEXT_DIM=4096
TEXT_INPUT_NORM=1
USE_SENTENCE_TOKEN=1
TEXT_SLOT_XATTN=1
GEN_EVAL=0
GEN_EVAL_CAPTION_CACHE=data/anytop_caption_llm2vec_v4b272neutral_multi
GEN_EVAL_MANIFEST=data/animo4d_L4TB_plus_human_v4b272neutral/eval_splits/val_all_clean_v1.json
PROTOCOL=unseen_topology_v1
HOLDOUT_ART=data/holdout_topologies_v1.json
HOLDOUT_SHA=0baf7bcfb82266d504f9bb45d0ec4f22980043ee49e53c0d7d13b40ebc858e0c
HUMAN_UPSAMPLE_FACTOR=3.0
HUMAN_UPSAMPLE_START_EPOCH=0
HUMAN_UPSAMPLE_PHASE2_FACTOR=4.5
HUMAN_UPSAMPLE_PHASE2_START_EPOCH=50
CAPTION_SAMPLING=random
CAPTION_SIDECAR=data/anytop_caption_llm2vec_v4b272neutral_multi

# ---- operational (callers may override) ----
SMOKE="${SMOKE:-0}"
PARAMETERIZATION="${PARAMETERIZATION:-}"  # empty = v (default); x = x-pred + v-space loss
W_DEC_WORLD="${W_DEC_WORLD:-}"            # decoded-geometry loss weights; empty = off
W_DEC_TRAJ="${W_DEC_TRAJ:-}"
W_DEC_SPEED="${W_DEC_SPEED:-}"
DEC_GEOM_T_MIN="${DEC_GEOM_T_MIN:-}"
DEC_GEOM_EVERY="${DEC_GEOM_EVERY:-}"
NUM_WORKERS="${NUM_WORKERS:-6}"   # 6: B4 micro-batching doubles loader request rate; 3 starved the step (py-spy 2/5 samples waiting)
LOG_EVERY="${LOG_EVERY:-50}"
QA_EVERY="${QA_EVERY:-200}"
SAVE_EVERY="${SAVE_EVERY:-10}"
EMPIRICAL_MAX="${EMPIRICAL_MAX:-0}"     # 0 = full-set empirical z_q norm (iron rule)
RESUME_CKPT="${RESUME_CKPT:-}"
OVERWRITE="${OVERWRITE:-1}"             # 0 for in-place watchdog resume
OUT="${OUT:?set OUT (use /tmp/gpscf_holdout_smoke for the smoke, runs/... for real)}"
# Qualify a bare resume filename against $OUT (codex r5 B1): "last_model.pt" alone would
# resolve under the repo root, miss, and the non-empty-OUT guard would then refuse the
# legitimate in-place resume. Same pattern as _launch_graph_pscf_2node_h200.sh.
if [ -n "$RESUME_CKPT" ] && [ ! -f "$RESUME_CKPT" ] && [ -f "$OUT/$RESUME_CKPT" ]; then
  RESUME_CKPT="$OUT/$RESUME_CKPT"
fi
if [ -n "$RESUME_CKPT" ] && [ ! -f "$RESUME_CKPT" ]; then
  echo "[gpscf-8card] ABORT: RESUME_CKPT=$RESUME_CKPT not found (neither as given nor under $OUT)"
  exit 1
fi

# ---- preflights: cache + tokenizer + manifest exist; GPUs in OUR allocs are idle ----
for f in "$TOKEN_CACHE/manifest.json" "$FROZEN_CKPT" "$GEN_EVAL_MANIFEST" \
         "$GEN_EVAL_CAPTION_CACHE.meta.json" "$HOLDOUT_ART"; do
  [ -e "$f" ] || { echo "[gpscf-8card] ABORT: missing $f"; exit 1; }
done
for j in "$JOB_A" "$JOB_B" "$JOB_C" "$JOB_D"; do
  squeue -h -j "$j" -t RUNNING -o %i 2>/dev/null | grep -q "^$j$" \
    || { echo "[gpscf-8card] ABORT: alloc $j not RUNNING"; exit 1; }
done

mkdir -p .aris/meta
exec 9>".aris/meta/.gpscf_holdout8.lock"
flock -n 9 || { echo "[gpscf-8card] ABORT: already running"; exit 0; }
echo $$ > .aris/meta/.gpscf_holdout8_orch.pid

COMMON_ENV="NNODES=4 NPROC_PER_NODE=2 MASTER_ADDR=$RDZV_HOST MASTER_PORT=$RDZV_PORT CVD=0,1 \
BATCH_SIZE=$BATCH_SIZE GRAD_ACCUM=$GRAD_ACCUM LR=$LR TOKEN_CACHE=$TOKEN_CACHE FROZEN_CKPT=$FROZEN_CKPT \
EPOCHS=$EPOCHS WARMUP_STEPS=$WARMUP_STEPS NUM_WORKERS=$NUM_WORKERS LOG_EVERY=$LOG_EVERY \
QA_EVERY=$QA_EVERY SAVE_EVERY=$SAVE_EVERY SEED=$SEED EMPIRICAL_MAX=$EMPIRICAL_MAX \
TEXT_DIM=$TEXT_DIM TEXT_INPUT_NORM=$TEXT_INPUT_NORM USE_SENTENCE_TOKEN=$USE_SENTENCE_TOKEN \
TEXT_SLOT_XATTN=$TEXT_SLOT_XATTN GEN_EVAL=$GEN_EVAL \
GEN_EVAL_CAPTION_CACHE=$GEN_EVAL_CAPTION_CACHE GEN_EVAL_MANIFEST=$GEN_EVAL_MANIFEST \
PROTOCOL=$PROTOCOL HOLDOUT_ART=$HOLDOUT_ART HOLDOUT_SHA=$HOLDOUT_SHA \
HUMAN_UPSAMPLE_FACTOR=$HUMAN_UPSAMPLE_FACTOR HUMAN_UPSAMPLE_START_EPOCH=$HUMAN_UPSAMPLE_START_EPOCH \
HUMAN_UPSAMPLE_PHASE2_FACTOR=$HUMAN_UPSAMPLE_PHASE2_FACTOR \
HUMAN_UPSAMPLE_PHASE2_START_EPOCH=$HUMAN_UPSAMPLE_PHASE2_START_EPOCH \
CAPTION_SAMPLING=$CAPTION_SAMPLING CAPTION_SIDECAR=$CAPTION_SIDECAR \
PARAMETERIZATION=$PARAMETERIZATION \
W_DEC_WORLD=$W_DEC_WORLD W_DEC_TRAJ=$W_DEC_TRAJ W_DEC_SPEED=$W_DEC_SPEED \
DEC_GEOM_T_MIN=$DEC_GEOM_T_MIN DEC_GEOM_EVERY=$DEC_GEOM_EVERY \
RESUME_CKPT=$RESUME_CKPT OVERWRITE=$OVERWRITE OUT=$OUT SMOKE=$SMOKE"

echo "[gpscf-8card] $(date '+%F %T %Z') cross-alloc 8-card DDP: $JOB_A+$JOB_B+$JOB_C+$JOB_D via $RDZV_HOST:$RDZV_PORT smoke=$SMOKE"
echo "[gpscf-8card] global=$(( BATCH_SIZE*8*GRAD_ACCUM )) (8xbs${BATCH_SIZE}xacc${GRAD_ACCUM}) lr=$LR epochs=$EPOCHS text_dim=$TEXT_DIM protocol=$PROTOCOL gen_eval=$GEN_EVAL"
echo "[gpscf-8card] cache=$TOKEN_CACHE frozen=$FROZEN_CKPT resume=${RESUME_CKPT:-<none>} out=$OUT"

run_alloc() {
    local tag="$1" job="$2" noderank="$3"
    srun --jobid="$job" --overlap --nodes=1 --ntasks=1 \
      --gres=gpu:2 --cpus-per-task=16 --no-kill \
      bash -c "cd '$P' && NODE_RANK=$noderank $COMMON_ENV bash scripts/_launch_graph_pscf.sh" \
      2>&1 | stdbuf -oL sed "s/^/[$tag] /"
}
run_alloc allocA "$JOB_A" 0 & PID_A=$!
run_alloc allocB "$JOB_B" 1 & PID_B=$!
run_alloc allocC "$JOB_C" 2 & PID_C=$!
run_alloc allocD "$JOB_D" 3 & PID_D=$!

wait "$PID_A"; RC_A=$?
wait "$PID_B"; RC_B=$?
wait "$PID_C"; RC_C=$?
wait "$PID_D"; RC_D=$?
echo "[gpscf-8card] $(date '+%F %T %Z') EXITED rc_A=$RC_A rc_B=$RC_B rc_C=$RC_C rc_D=$RC_D"
if [ "$RC_A" -ne 0 ] || [ "$RC_B" -ne 0 ] || [ "$RC_C" -ne 0 ] || [ "$RC_D" -ne 0 ]; then exit 1; fi
exit 0
