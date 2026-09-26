#!/bin/bash
# v2 CONTINUATION branch (v2b): 8xA100 CROSS-NODE DDP (swarma1002 + swarma1004), EPOCHS=400.
#
# WHY THIS FILE EXISTS: the parent run (_launch_holdout_backbone_8card.sh, 8xH100) finished
# its cosine at ep298/300 with lr decayed to 8.0e-07 (1% of the 8e-5 peak) — the ep240 val
# plateau is schedule exhaustion, not saturation. Continuing needs a longer schedule, which
# changes the config digest, so this is a NEW BRANCH with its own OUT dir; the ep300
# artefacts stay untouched.
#
# WHY 8xA100 RATHER THAN 4xH100: the 4-card attempt (per-GPU 8, accum 2) OOMed at ~79.1/79.2
# GiB — accumulation keeps a second copy of the 309M-param fp32 gradients alive, which is
# exactly what does not fit in an 80GB H100. Eight cards restore the PARENT'S EXACT SHAPE:
# per-GPU 8 x accum 1 x 8 ranks = global 64, no accumulation, no extra gradient copy. That
# also removes accumulation as a confound when comparing this continuation to the parent.
#
# TOPOLOGY — CROSS-NODE, NOT CROSS-CGROUP: each alloc owns a whole node's 4 A100-SXM4-80GB,
# which are NV4-fully-connected, so NVLink must stay ENABLED inside a node (P2P/SHM ON) and
# only the inter-node ring crosses IB (ib0 / mlx5_0 on these nodes — note this is the
# OPPOSITE interface from the H200 pair, which uses ib1/mlx5_1). Two allocs landing on the
# same node would be the cross-cgroup case with different NCCL requirements; the preflight
# refuses that explicitly.
#
# RENDEZVOUS: the master's ib0 IPv4 is read ON the node and used raw. Hostname-based
# rendezvous has failed here before (agent hostname != IB name -> no TCPStore -> all ranks
# time out).
#
# PARENT-RUN PROTECTION: refuses to write into — or resume directly from — any protected
# completed-run directory. The branch resumes from a COPY inside its own OUT dir.
#
# Smoke: SMOKE=1 OUT=<shared-fs dir> RESUME_CKPT=<that dir>/resume_seed_ep289.pt \
#        bash scripts/_launch_holdout_backbone_8card_a100.sh
#        (OUT must be on the SHARED filesystem — /tmp is node-local and invisible to the
#         other node's ranks.)
# Real:  OUT=runs/holdout_backbone_llm2vec_v2b_ep400 \
#        RESUME_CKPT=runs/holdout_backbone_llm2vec_v2b_ep400/resume_seed_ep289.pt \
#        bash scripts/_launch_holdout_backbone_8card_a100.sh
set -uo pipefail
P=/scratch/ts1v23/workspace/noKslot_clean
cd "$P"

# ---- allocs (2x 2xH100 on swarmh1001) ----
JOB_A="${JOB_A:-1355479}"   # swarma1002, 4x A100-SXM4-80GB
JOB_B="${JOB_B:-1355480}"   # swarma1004, 4x A100-SXM4-80GB
RDZV_HOST="${RDZV_HOST:-}"   # derived from NODE_A's IB IP below
RDZV_PORT="${RDZV_PORT:-29521}"   # distinct from 29517 (8-card H100) and 29519 (4-card H100)

# ---- experiment contract (PINNED; the config digest hashes what the trainer receives) ----
BATCH_SIZE="${BATCH_SIZE:-8}"   # x8 GPUs x accum 1 = global 64 — IDENTICAL to the parent run
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR=8e-5                         # unchanged peak; the cosine tail is what gets extended
# EPOCHS is PINNED, not overridable (codex v2b r2 BLOCKING-2): with EPOCHS=300 an
# UNMIGRATED seed passes the preflight AND digests identically to the parent stamp, so the
# run would silently finish at 300 having burned the allocation for nothing.
EPOCHS="${EPOCHS:-400}"         # v2b branch=400; v3 continuation=300 (pass explicitly)
# Expected resume-contract stamp for THIS configuration, i.e.
#   codeflow_training_config_sha256({...parent args..., epochs: 400}, world_size)
# with global_batch = 64. Recompute and update this constant if ANY pinned experiment
# value below changes; a mismatch means the seed was not migrated (or was migrated to a
# different schedule) and the trainer would refuse after minutes of setup.
# Per-EXPERIMENT resume-contract stamp. Overridable because this launcher serves more than
# one run: v2b(ep400 branch)=d65e7585…, v3(x-pred, epochs=300)=b41e0d89…. Both were
# verified to be identical under 4x(B8,acc2) and 8x(B8,acc1) since global_batch=64 either
# way. A wrong value fails the preflight in seconds instead of after DDP+dataset init.
EXPECTED_STAMP="${EXPECTED_STAMP:-d65e7585d3f40c184b890c3b60a695694a131f8005d687b094f2469643bb7938}"
# Continuing a finished cosine is a DELIBERATE lr re-lift, not a smooth extension: the
# ep289 ckpt carries lr=1.019e-06, while the 400-epoch curve at ep290 sits at ~1.477e-05
# (~14.5x jump). The smoke gate below asserts the first logged lr lands in this window —
# outside it means the schedule was wired differently than intended (codex v2b r2 MAJOR-2).
EXPECTED_FIRST_LR_MIN="${EXPECTED_FIRST_LR_MIN:-1.30e-05}"
EXPECTED_FIRST_LR_MAX="${EXPECTED_FIRST_LR_MAX:-1.65e-05}"
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
PARAMETERIZATION="${PARAMETERIZATION:-}"  # empty = v-pred (this branch's parent)
W_DEC_WORLD="${W_DEC_WORLD:-}"
W_DEC_TRAJ="${W_DEC_TRAJ:-}"
W_DEC_SPEED="${W_DEC_SPEED:-}"
DEC_GEOM_T_MIN="${DEC_GEOM_T_MIN:-}"
DEC_GEOM_EVERY="${DEC_GEOM_EVERY:-}"
NUM_WORKERS="${NUM_WORKERS:-6}"
LOG_EVERY="${LOG_EVERY:-50}"
QA_EVERY="${QA_EVERY:-200}"
SAVE_EVERY="${SAVE_EVERY:-10}"
EMPIRICAL_MAX="${EMPIRICAL_MAX:-0}"     # 0 = full-set empirical z_q norm (iron rule)
RESUME_CKPT="${RESUME_CKPT:-}"
OVERWRITE="${OVERWRITE:-0}"             # 0: never clear OUT — this branch ALWAYS resumes
OUT="${OUT:?set OUT (use /tmp/gpscf_v2b_smoke for the smoke, runs/... for real)}"

# ---- protected parents: never write into, never resume directly from ----
PROTECTED_DIRS=(
  "runs/holdout_backbone_llm2vec_8card_v2"
  "runs/holdout_backbone_llm2vec_8card_v3_xpred"
  "runs/holdout_backbone_llm2vec_8card_v1"
  "runs/holdout_vqvae_semantic_8card_v1"
)
canon() { readlink -m -- "$1"; }
assert_outside_protected() {   # $1=path $2=what
  local real; real="$(canon "$1")"
  local d prot
  for d in "${PROTECTED_DIRS[@]}"; do
    prot="$(canon "$P/$d")"
    if [ "$real" = "$prot" ] || case "$real" in "$prot"/*) true;; *) false;; esac; then
      echo "[gpscf-8a100] ABORT: $2 '$1' is inside PROTECTED run dir $d"
      echo "[gpscf-8a100]        (the parent run's artefacts must stay byte-identical;"
      echo "[gpscf-8a100]         copy the ckpt into \$OUT and migrate the copy instead)"
      exit 1
    fi
  done
}
# A continuation of a protected run must write into that very directory. Allow it ONLY
# when the operator says so AND the resume ckpt lives in that same dir (i.e. we are
# genuinely continuing THAT run, not writing one run's output into another's folder).
ALLOW_PROTECTED_OUT="${ALLOW_PROTECTED_OUT:-0}"
if [ "$ALLOW_PROTECTED_OUT" = "1" ]; then
  echo "[gpscf-8a100] NOTE: ALLOW_PROTECTED_OUT=1 — OUT may be a protected run dir"
  echo "[gpscf-8a100]       (in-place continuation). Parent-run protection is RELAXED for"
  echo "[gpscf-8a100]       this launch; the resume ckpt must live in that same dir."
else
  assert_outside_protected "$OUT" "OUT"
fi

# Qualify a bare resume filename against $OUT.
if [ -n "$RESUME_CKPT" ] && [ ! -f "$RESUME_CKPT" ] && [ -f "$OUT/$RESUME_CKPT" ]; then
  RESUME_CKPT="$OUT/$RESUME_CKPT"
fi
# A continuation branch has no legitimate from-scratch mode: an empty RESUME_CKPT would
# silently start at epoch 0 and bypass the resume contract entirely (BLOCKING-4).
if [ -z "$RESUME_CKPT" ]; then
  echo "[gpscf-8a100] ABORT: RESUME_CKPT is required — this launcher only continues an"
  echo "[gpscf-8a100]        existing run (set it to \$OUT/resume_seed_ep289.pt or to this"
  echo "[gpscf-8a100]        branch's own last_model.pt)"
  exit 1
fi
if [ ! -f "$RESUME_CKPT" ]; then
  echo "[gpscf-8a100] ABORT: RESUME_CKPT=$RESUME_CKPT not found (neither as given nor under $OUT)"
  exit 1
fi
if [ "$ALLOW_PROTECTED_OUT" = "1" ]; then
  RC_PARENT="$(canon "$(dirname -- "$RESUME_CKPT")")"
  if [ "$RC_PARENT" != "$(canon "$OUT")" ]; then
    echo "[gpscf-8a100] ABORT: ALLOW_PROTECTED_OUT=1 requires RESUME_CKPT to live in \$OUT"
    echo "[gpscf-8a100]        (got $RC_PARENT vs $(canon "$OUT")) — refusing to mix runs"
    exit 1
  fi
else
  assert_outside_protected "$RESUME_CKPT" "RESUME_CKPT"
fi

# The trainer only treats a resume as "in place" when the ckpt's parent dir IS $OUT; a
# cross-directory resume into a non-empty OUT is refused — but only AFTER rendezvous, DDP
# init and dataset load have burned minutes of 4xH100. Catch it here instead. (Found by
# the first 4-card smoke, which pointed OUT at /tmp while the seed lived under runs/.)
RESUME_PARENT="$(canon "$(dirname -- "$RESUME_CKPT")")"
OUT_CANON="$(canon "$OUT")"
if [ "$RESUME_PARENT" != "$OUT_CANON" ] && [ "$OVERWRITE" != "1" ] \
   && [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
  echo "[gpscf-8a100] ABORT: RESUME_CKPT lives in $RESUME_PARENT but OUT is $OUT_CANON,"
  echo "[gpscf-8a100]        and OUT is non-empty with OVERWRITE=0 — the trainer would"
  echo "[gpscf-8a100]        refuse this as a non-in-place resume."
  echo "[gpscf-8a100]        Fix: copy the seed into \$OUT and point RESUME_CKPT at it."
  exit 1
fi

# ---- preflights: cache + tokenizer + manifest exist ----
for f in "$TOKEN_CACHE/manifest.json" "$FROZEN_CKPT" "$GEN_EVAL_MANIFEST" \
         "$GEN_EVAL_CAPTION_CACHE.meta.json" "$HOLDOUT_ART"; do
  [ -e "$f" ] || { echo "[gpscf-8a100] ABORT: missing $f"; exit 1; }
done

# ---- contract preflight: the resume ckpt must already carry THIS schedule ----
# Catches "forgot to migrate" and "migrated to the wrong number" before torchrun spends
# minutes on rendezvous + cache warmup only to be refused by the trainer (BLOCKING-4).
python3 - "$RESUME_CKPT" "$EPOCHS" "$EXPECTED_STAMP" <<'PY' || exit 1
import sys, torch
sys.path.insert(0, "/scratch/ts1v23/workspace/noKslot_clean")
from src.data import provenance as prov
path, want, want_stamp = sys.argv[1], int(sys.argv[2]), sys.argv[3]
# mmap keeps the 3.7GB of weights off the heap — we only need args + the stamp
# (codex v2b r2 MINOR-2). Fall back if the torch build predates mmap support.
try:
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
except (TypeError, RuntimeError):
    ck = torch.load(path, map_location="cpu", weights_only=False)
a = ck.get("args") or {}
have = a.get("epochs")
if have != want:
    print(f"[gpscf-8a100] ABORT: resume ckpt records epochs={have} but this launcher "
          f"pins EPOCHS={want}; run scripts/_migrate_contract_epochs.py on the COPY first")
    sys.exit(1)
p = prov.read(ck) or {}
stamp = p.get("training_config_sha256")
# Compare the FULL stamp, not just epochs: a wrong global_batch/parameterization/text_dim
# would otherwise only surface after DDP + tokenizer + dataset init burn minutes of 4xH100.
if stamp != want_stamp:
    print(f"[gpscf-8a100] ABORT: resume ckpt stamp {str(stamp)[:16]}... != expected "
          f"{want_stamp[:16]}... — the seed is not migrated for THIS configuration")
    sys.exit(1)
print(f"[gpscf-8a100] contract preflight OK: ckpt epochs={have} epoch_reached={ck.get('epoch')} "
      f"stamp={str(stamp)[:16]}...")
PY

# ---- alloc preflights ----
if [ "$JOB_A" = "$JOB_B" ]; then
  echo "[gpscf-8a100] ABORT: JOB_A and JOB_B are the same alloc ($JOB_A) — both node ranks"
  echo "[gpscf-8a100]        would land on the same two GPUs"
  exit 1
fi
for j in "$JOB_A" "$JOB_B"; do
  squeue -h -j "$j" -t RUNNING -o %i 2>/dev/null | grep -q "^$j$" \
    || { echo "[gpscf-8a100] ABORT: alloc $j not RUNNING"; exit 1; }
done
# Both allocs must occupy exactly ONE and the SAME node: the pinned NCCL settings below
# assume the one-node/two-cgroup topology. Compare expanded hostnames, not the compact
# NodeList string, and require %D==1 for each (MAJOR-3).
node_of() {   # $1=jobid -> single hostname, or empty on any anomaly
  local n d
  d=$(squeue -h -j "$1" -o %D 2>/dev/null | tr -d ' ')
  [ "$d" = "1" ] || return 1
  n=$(scontrol show hostnames "$(squeue -h -j "$1" -o %N 2>/dev/null)" 2>/dev/null)
  [ "$(printf '%s\n' "$n" | grep -c .)" = "1" ] || return 1
  printf '%s' "$n"
}
NODE_A=$(node_of "$JOB_A") || { echo "[gpscf-8a100] ABORT: alloc $JOB_A is not on exactly one node"; exit 1; }
NODE_B=$(node_of "$JOB_B") || { echo "[gpscf-8a100] ABORT: alloc $JOB_B is not on exactly one node"; exit 1; }
if [ -z "$NODE_A" ] || [ -z "$NODE_B" ]; then
  echo "[gpscf-8a100] ABORT: could not resolve alloc nodes (A=$NODE_A B=$NODE_B)"
  exit 1
fi
# This launcher is the CROSS-NODE shape: each alloc owns a whole node's 4 GPUs, so NVLink
# stays enabled inside a node and only the inter-node ring crosses IB. Two allocs on the
# SAME node would be the cross-cgroup case, which needs P2P/SHM OFF — refuse it here.
if [ "$NODE_A" = "$NODE_B" ]; then
  echo "[gpscf-8a100] ABORT: both allocs are on $NODE_A — this launcher assumes one whole"
  echo "[gpscf-8a100]        node per alloc (NVLink intra-node). Use the 4-card same-node"
  echo "[gpscf-8a100]        launcher for the cross-cgroup topology."
  exit 1
fi
# The rendezvous host is DERIVED from the resolved node, never prefix-matched: a stale
# override like "swarmh1001-garbage" would pass a prefix test and then hang the whole
# 4-card rendezvous (codex v2b r2 MAJOR-3). Callers may only override the IB suffix.
# Resolve the master's IB address ON the node itself and use the raw IP: hostname-based
# rendezvous has bitten this project before (agent hostname != IB name -> nobody starts the
# TCPStore -> every rank times out).
DERIVED_RDZV=$(ssh -o ConnectTimeout=10 -o BatchMode=yes "$NODE_A" \
  "ip -o -4 addr show dev ib0 2>/dev/null | awk '{print \$4}' | cut -d/ -f1" 2>/dev/null)
if [ -z "$DERIVED_RDZV" ]; then
  echo "[gpscf-8a100] ABORT: could not read ib0 IPv4 address on $NODE_A"
  exit 1
fi
if [ -n "${RDZV_HOST:-}" ] && [ "$RDZV_HOST" != "$DERIVED_RDZV" ]; then
  echo "[gpscf-8a100] ABORT: RDZV_HOST=$RDZV_HOST != $NODE_A's ib0 address $DERIVED_RDZV"
  exit 1
fi
RDZV_HOST="$DERIVED_RDZV"

mkdir -p .aris/meta "$OUT"
# Byte offset of train.log BEFORE this launch: the smoke gates below must parse only what
# THIS run appended, otherwise a re-smoke into the same OUT reads the previous run's lr and
# reports SMOKE-OK for a run that produced zero steps (codex v2b r3 BLOCKING-1).
LOG_OFFSET_BEFORE=$(stat -c %s "$OUT/train.log" 2>/dev/null || echo 0)
for t in allocA allocB; do : > "$OUT/orch_${t}.log"; done   # fresh per-launch orch sinks

exec 9>".aris/meta/.gpscf_v2b_8a100.lock"
# A lock conflict means NOTHING was launched — must not look like success to a watchdog
# or a monitor (MAJOR-5).
flock -n 9 || { echo "[gpscf-8a100] ABORT: already running (lock held)"; exit 3; }
echo $$ > .aris/meta/.gpscf_v2b_8a100_orch.pid

# NCCL PINNED (not defaulted): same-node cross-cgroup requires P2P+SHM off over IB.
# In SMOKE we additionally force NCCL_DEBUG so the transport can be ASSERTED rather than
# assumed — NCCL_IB_DISABLE=0 only permits IB, it does not prove the ring took it
# (codex v2b r3 MAJOR-4).
NCCL_DEBUG_ENV=""
if [ "$SMOKE" = "1" ]; then
  NCCL_DEBUG_ENV="NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET "
fi
COMMON_ENV="NNODES=2 NPROC_PER_NODE=4 MASTER_ADDR=$RDZV_HOST MASTER_PORT=$RDZV_PORT CVD=0,1,2,3 \
${NCCL_DEBUG_ENV}NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=0 NCCL_SOCKET_IFNAME=ib0 NCCL_IB_DISABLE=0 NCCL_IB_HCA=mlx5_0 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
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

echo "[gpscf-8a100] $(date '+%F %T %Z') cross-alloc 4-card DDP: $JOB_A+$JOB_B on $NODE_A via $RDZV_HOST:$RDZV_PORT smoke=$SMOKE"
echo "[gpscf-8a100] global=$(( BATCH_SIZE*8*GRAD_ACCUM )) (8xbs${BATCH_SIZE}xacc${GRAD_ACCUM}) lr=$LR epochs=$EPOCHS text_dim=$TEXT_DIM protocol=$PROTOCOL gen_eval=$GEN_EVAL"
echo "[gpscf-8a100] cache=$TOKEN_CACHE frozen=$FROZEN_CKPT resume=$RESUME_CKPT out=$OUT overwrite=$OVERWRITE"

run_alloc() {
    local tag="$1" job="$2" noderank="$3"
    # Each alloc ALSO writes its own file: two concurrent seds sharing the orchestrator's
    # stdout can interleave mid-line on long tracebacks / NCCL dumps, which is exactly when
    # the evidence matters (codex v2b r2 MINOR-3). tee keeps the live view as well.
    srun --jobid="$job" --overlap --nodes=1 --ntasks=1 \
      --gres=gpu:4 --cpus-per-task=32 --no-kill \
      bash -c "cd '$P' && NODE_RANK=$noderank $COMMON_ENV bash scripts/_launch_graph_pscf.sh" \
      2>&1 | stdbuf -oL sed "s/^/[$tag] /" | tee -a "$OUT/orch_${tag}.log"
    return "${PIPESTATUS[0]}"
}
run_alloc allocA "$JOB_A" 0 & PID_A=$!
run_alloc allocB "$JOB_B" 1 & PID_B=$!

wait "$PID_A"; RC_A=$?
wait "$PID_B"; RC_B=$?
echo "[gpscf-8a100] $(date '+%F %T %Z') EXITED rc_A=$RC_A rc_B=$RC_B"

# Smoke gate on the lr re-lift: the whole point of this branch is that the extended cosine
# hands the optimiser a usable lr again. If the first logged lr is outside the expected
# window the schedule is not what we think it is, and a green smoke would be misleading.
if [ "$SMOKE" = "1" ] && [ "$RC_A" -eq 0 ] && [ "$RC_B" -eq 0 ]; then
  # Parse ONLY this launch's appended bytes (see LOG_OFFSET_BEFORE).
  NEW_LOG=$(tail -c "+$((LOG_OFFSET_BEFORE + 1))" "$OUT/train.log" 2>/dev/null)
  # (1) transport: the inter-rank ring must have taken IB, with all 4 ranks present.
  IB_HITS=$(grep -c "NET/IB" "$OUT/orch_allocA.log" "$OUT/orch_allocB.log" 2>/dev/null | awk -F: '{s+=$2} END{print s+0}')
  SOCK_HITS=$(grep -cE "NET/Socket" "$OUT/orch_allocA.log" "$OUT/orch_allocB.log" 2>/dev/null | awk -F: '{s+=$2} END{print s+0}')
  WS_OK=$(printf '%s\n' "$NEW_LOG" | grep -cE "world_size[ =:]+8|WORLD_SIZE[ =:]+8")
  if [ "$IB_HITS" -eq 0 ]; then
    echo "[gpscf-8a100] SMOKE-FAIL: no NET/IB in NCCL init logs — inter-node ring did not take IB"
    exit 4
  fi
  if [ "$SOCK_HITS" -gt 0 ]; then
    echo "[gpscf-8a100] SMOKE-FAIL: NCCL fell back to NET/Socket ($SOCK_HITS hits)"
    exit 4
  fi
  if [ "$WS_OK" -eq 0 ]; then
    echo "[gpscf-8a100] SMOKE-FAIL: world_size=8 not present in THIS launch's train.log —"
    echo "[gpscf-8a100]             the ring is not the 8-rank shape this launcher assumes"
    exit 4
  fi
  # (2) the lr re-lift.
  FIRST_LR=$(printf '%s\n' "$NEW_LOG" | grep -oE 'lr=[0-9.e+-]+' | head -1 | cut -d= -f2)
  if [ -z "$FIRST_LR" ]; then
    echo "[gpscf-8a100] SMOKE-FAIL: no lr line appended by THIS launch — zero steps executed"
    exit 4
  fi
  if ! awk -v v="$FIRST_LR" -v lo="$EXPECTED_FIRST_LR_MIN" -v hi="$EXPECTED_FIRST_LR_MAX" \
       'BEGIN{exit !(v+0>=lo+0 && v+0<=hi+0)}'; then
    echo "[gpscf-8a100] SMOKE-FAIL: first lr=$FIRST_LR outside expected re-lift window "
    echo "[gpscf-8a100]             [$EXPECTED_FIRST_LR_MIN, $EXPECTED_FIRST_LR_MAX]"
    exit 4
  fi
  echo "[gpscf-8a100] SMOKE-OK: first lr=$FIRST_LR within re-lift window"
fi

if [ "$RC_A" -ne 0 ] || [ "$RC_B" -ne 0 ]; then exit 1; fi
exit 0
