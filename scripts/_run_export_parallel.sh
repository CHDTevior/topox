#!/usr/bin/env bash
# Durable cross-alloc PARALLEL Graph-CodeFlow RVQ token export (full-length 300).
#
# Embarrassingly parallel: each clip -> independent per-clip {i:06d}.npz, and the
# export script's --num_shards/--shard_idx slices clips disjointly by dataset index
# (i % num_shards == shard_idx), so 12 shards write disjoint npz filenames into the
# SAME split dir with no collision. We spread 12 shards over 6 idle 2-GPU allocs.
#
# Run node-local under setsid nohup so it survives ssh disconnect (PPID=1), e.g.:
#   ssh <compute-node> "cd $REPO && setsid nohup bash scripts/_run_export_parallel.sh \
#                       > $OUT/orch.log 2>&1 </dev/null &"
# Do NOT add setsid inside this script -- the caller wraps it.
#
# set -u (catch unset vars) and pipefail, but NOT -e: one failed shard must not
# abort the `wait` on the other 5 srun steps; non-zero srun exits are logged.
set -uo pipefail

REPO=/scratch/ts1v23/workspace/noKslot_clean
PY=/scratch/ts1v23/.conda/bin/python3
CKPT=runs/vqvae_L5_C50_d512_Q4_n512_b96_300ep_seed42/best_model.pt
OUT=/scratch/ts1v23/workspace/noKslot_clean/data/codeflow_tokens_cleanL5_ep280_fulllen300_par

NUM_SHARDS=12
# 6 idle allocs, each gres/gpu=2 (VAE allocs 944457/944458 are BUSY -- not listed).
#   swarmh1002: 974142 974141 944462  (H100)
#   flamingo01: 896281                (H200)
#   blossom03:  988073                (H200)
#   rose09:     977668                (A100)
ALLOCS=(974142 974141 944462 896281 988073 977668)

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

cd "$REPO"
mkdir -p "$OUT/logs"

echo "[orch] $(ts) START host=$(hostname) num_shards=$NUM_SHARDS allocs=${ALLOCS[*]}"
echo "[orch] $(ts) ckpt=$CKPT out=$OUT"

pids=()
allocs_for_pid=()
for a in "${!ALLOCS[@]}"; do
  alloc="${ALLOCS[$a]}"
  base=$(( a * 2 ))             # global shard for local gpu 0; gpu 1 uses base+1
  log="$OUT/logs/shard_alloc_${alloc}.log"
  echo "[orch] $(ts) LAUNCH alloc=$alloc (alloc_idx=$a) shards=$base,$((base+1)) -> $log"

  # One srun step per alloc; inside it, 2 export procs (one per GPU) run concurrently.
  # BASE is exported into the srun bash so the two inner procs use BASE+0 and BASE+1.
  BASE="$base" REPO="$REPO" PY="$PY" CKPT="$CKPT" OUT="$OUT" NUM_SHARDS="$NUM_SHARDS" \
  srun --overlap --jobid="$alloc" --gres=gpu:2 --cpus-per-task=6 \
       --ntasks=1 --nodes=1 --no-kill \
       bash -c '
         cd "$REPO"
         ipids=()
         for g in 0 1; do
           CUDA_VISIBLE_DEVICES=$g "$PY" scripts/export_graph_vq_tokens.py \
             --frozen_vqvae_ckpt "$CKPT" \
             --out "$OUT" \
             --splits train,val \
             --num_frames 300 \
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
done

echo "[orch] $(ts) all 6 srun steps launched; pids=${pids[*]}"

# Wait for EVERY srun step (do not let one failure short-circuit the others).
fail=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  alloc="${allocs_for_pid[$i]}"
  if wait "$pid"; then
    echo "[orch] $(ts) alloc=$alloc srun step OK (pid=$pid)"
  else
    rc=$?
    fail=1
    echo "[orch] $(ts) alloc=$alloc srun step FAILED rc=$rc (pid=$pid) -- see $OUT/logs/shard_alloc_${alloc}.log"
  fi
done

if [[ "$fail" -ne 0 ]]; then
  echo "[orch] $(ts) ALL-DONE with FAILURES -- NOT merging (inspect logs, re-run failed shards before merge)"
  exit 1
fi

echo "[orch] $(ts) ALL-DONE (12 shards OK) -- merging"
"$PY" scripts/merge_export_shards.py --out "$OUT" --num_shards "$NUM_SHARDS" --splits train,val
mrc=$?
if [[ "$mrc" -ne 0 ]]; then
  echo "[orch] $(ts) MERGE FAILED rc=$mrc"
  exit "$mrc"
fi

echo "[orch] $(ts) MERGE-DONE -> $OUT"
