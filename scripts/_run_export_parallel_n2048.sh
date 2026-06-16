#!/usr/bin/env bash
# Durable cross-alloc PARALLEL Graph-CodeFlow RVQ token export for the MERGED
# L4_safe+truebones dataset (full-length 300), frozen on the merged n2048 VQVAE
# best_model.pt (ep199 — val_total 1.0478; val diverged after, so best is the freeze).
#
# Copy of scripts/_run_export_parallel_mergedL4TB.sh with three changes:
#   1. CKPT -> n2048 best_model.pt ; OUT -> the n2048 token cache dir.
#   2. ALLOCS -> the currently-idle A100 allocs ONLY (NOT flamingo01/blossom03,
#      which are running the live H200 512-backbone).
#   3. CAP unchanged (reuse the merged T5 caption cache).
# Export shards are INDEPENDENT processes (not DDP) -> mixing GPU counts is fine.
#
# Caller wraps with setsid nohup on a COMPUTE node (PPID=1, survives ssh drop).
set -uo pipefail

REPO=/scratch/ts1v23/workspace/noKslot_clean
PY=/scratch/ts1v23/.conda/bin/python3
CKPT=runs/vqvae_L4safeTB_C96_J144_d512_Q4_n2048_b32_300ep_seed42/best_model.pt
OUT=/scratch/ts1v23/workspace/noKslot_clean/data/codeflow_tokens_mergedL4TB_n2048_ep199_fulllen300
CAP=data/anytop_caption_t5_mergedL4TB_multi

# "jobid:ngpu" per idle A100 alloc. Shard base assigned cumulatively in list order.
#   976853 swarma1003 (4xA100)  974143 swarma1004 (4xA100)  993168 rose09 (2xA100)
#   -> 10 GPUs total -> NUM_SHARDS=10
ALLOCS=("976853:4" "974143:4" "993168:2")

NUM_SHARDS=0
for ag in "${ALLOCS[@]}"; do NUM_SHARDS=$(( NUM_SHARDS + ${ag#*:} )); done

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

cd "$REPO"
# Stale-output guard: refuse to write into an existing (possibly partial/successful)
# cache -- that would poison prewarm/train. Override with ALLOW_EXISTING=1 for re-runs.
if [ "${ALLOW_EXISTING:-0}" != 1 ] && { [ -f "$OUT/manifest.json" ] || [ -n "$(ls -A "$OUT/train" 2>/dev/null)" ] || [ -n "$(ls -A "$OUT/val" 2>/dev/null)" ]; }; then
  echo "[orch] $(ts) ABORT: $OUT already has manifest.json or non-empty train/val (stale export?); set ALLOW_EXISTING=1 to override"; exit 2
fi
mkdir -p "$OUT/logs"
echo "[orch] $(ts) START host=$(hostname) num_shards=$NUM_SHARDS allocs=${ALLOCS[*]}"
echo "[orch] $(ts) ckpt=$CKPT out=$OUT cap=$CAP"

pids=()
allocs_for_pid=()
base=0
for ag in "${ALLOCS[@]}"; do
  alloc="${ag%:*}"; ngpu="${ag#*:}"
  log="$OUT/logs/shard_alloc_${alloc}.log"
  echo "[orch] $(ts) LAUNCH alloc=$alloc ngpu=$ngpu shards=$base..$((base+ngpu-1)) -> $log"
  BASE="$base" NGPU="$ngpu" REPO="$REPO" PY="$PY" CKPT="$CKPT" OUT="$OUT" CAP="$CAP" NUM_SHARDS="$NUM_SHARDS" \
  srun --overlap --jobid="$alloc" --gres=gpu:"$ngpu" --cpus-per-task=8 \
       --ntasks=1 --nodes=1 --no-kill \
       bash -c '
         cd "$REPO"
         # Respect Slurm cgroup mask: split the INHERITED CUDA_VISIBLE_DEVICES and give
         # each child the g-th allocated device (never override to local 0..N-1, which on
         # a partial-node alloc would address another tenant'\''s GPUs).
         IFS=, read -ra DEVS <<< "${CUDA_VISIBLE_DEVICES:-}"
         if [ "${#DEVS[@]}" -lt "$NGPU" ]; then echo "FATAL: alloc exposes ${#DEVS[@]} GPUs < NGPU=$NGPU (CVD=${CUDA_VISIBLE_DEVICES:-unset})"; exit 3; fi
         ipids=()
         for (( g=0; g<NGPU; g++ )); do
           CUDA_VISIBLE_DEVICES="${DEVS[$g]}" "$PY" scripts/export_graph_vq_tokens.py \
             --frozen_vqvae_ckpt "$CKPT" \
             --out "$OUT" \
             --splits train,val \
             --num_frames 300 \
             --caption_emb_cache "$CAP.npz" \
             --caption_token_cache "$CAP" \
             --min_text_coverage 0.99 \
             --device cuda \
             --num_shards "$NUM_SHARDS" \
             --shard_idx $(( BASE + g )) &
           ipids+=($!)
         done
         irc=0
         for ip in "${ipids[@]}"; do wait "$ip" || irc=$?; done
         exit $irc
       ' > "$log" 2>&1 &
  pids+=("$!")
  allocs_for_pid+=("$alloc")
  base=$(( base + ngpu ))
done

echo "[orch] $(ts) all ${#pids[@]} srun steps launched; pids=${pids[*]}"

fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[orch] $(ts) alloc=${allocs_for_pid[$i]} srun step OK"
  else
    rc=$?; fail=1
    echo "[orch] $(ts) alloc=${allocs_for_pid[$i]} srun step FAILED rc=$rc -- see $OUT/logs/shard_alloc_${allocs_for_pid[$i]}.log"
  fi
done

if [[ "$fail" -ne 0 ]]; then
  echo "[orch] $(ts) ALL-DONE with FAILURES -- NOT merging (inspect logs, re-run failed shards before merge)"
  exit 1
fi

echo "[orch] $(ts) ALL-DONE ($NUM_SHARDS shards OK) -- merging"
"$PY" scripts/merge_export_shards.py --out "$OUT" --num_shards "$NUM_SHARDS" --splits train,val
mrc=$?
echo "[orch] $(ts) MERGE rc=$mrc -> $OUT"
exit "$mrc"
