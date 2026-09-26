#!/usr/bin/env bash
# Durable cross-alloc PARALLEL RVQ token export: holdout-protocol semantic VQVAE ep150,
# LLM2Vec ragged caption sidecars, full-length 300.
#
# VERBATIM copy of scripts/_run_export_parallel_l4safe_human.sh except CKPT/OUT/CAP/ALLOCS
# and the four protocol args:
#   - CKPT = runs/holdout_vqvae_semantic_8card_v1/ep150_model.pt  (user-chosen frozen tokenizer)
#   - OUT  = data/codeflow_tokens_holdout_semantic_ep150_fulllen300
#   - CAP  = data/anytop_caption_llm2vec_v4b272neutral_multi  (RAGGED sidecars; no .npz — the
#            dataset accepts a sidecar-only prefix, so caption_emb_cache passes the PREFIX)
#   - --caption_token_max_len 114 (ragged bound check; corpus clean max 113, allowlisted
#     truncations capped at 114)
#   - protocol args mirror the VQVAE training run exactly: unseen_topology_v1 + splits_dir
#     data/holdout_splits_v1 + the pre-registered artifact and its self-certifying sha.
#   - ALLOCS = four freed 2xH100 allocs on swarmh1001 -> NUM_SHARDS=8.
#
# Caller wraps with setsid nohup on a COMPUTE node (PPID=1, survives ssh drop):
#   ssh swarmh1001 "cd /scratch/ts1v23/workspace/noKslot_clean && setsid nohup bash \
#     scripts/_run_export_parallel_holdout_ep150.sh > data/codeflow_tokens_holdout_semantic_ep150_fulllen300/orch.log 2>&1 </dev/null &"
set -uo pipefail

REPO=/scratch/ts1v23/workspace/noKslot_clean
PY=/scratch/ts1v23/.conda/bin/python3
CKPT=runs/holdout_vqvae_semantic_8card_v1/ep150_model.pt
OUT=/scratch/ts1v23/workspace/noKslot_clean/data/codeflow_tokens_holdout_semantic_ep150_fulllen300
CAP=data/anytop_caption_llm2vec_v4b272neutral_multi
SPLITS_DIR=data/holdout_splits_v1
HOLDOUT_ART=data/holdout_topologies_v1.json
HOLDOUT_SHA=0baf7bcfb82266d504f9bb45d0ec4f22980043ee49e53c0d7d13b40ebc858e0c

ALLOCS=("1141929:2" "1041113:2" "1077414:2" "1123147:2")

NUM_SHARDS=0
for ag in "${ALLOCS[@]}"; do NUM_SHARDS=$(( NUM_SHARDS + ${ag#*:} )); done

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

cd "$REPO"
# The LLM2Vec cache must exist and be COMPLETE before any GPU is spent.
for f in "$CAP.embs.npy" "$CAP.tokens.npy" "$CAP.offsets.npy" "$CAP.keys.json" "$CAP.meta.json"; do
  [ -f "$f" ] || { echo "[orch] $(ts) ABORT: missing $f (build the LLM2Vec cache first)"; exit 1; }
done
mkdir -p "$OUT/logs"
echo "[orch] $(ts) START host=$(hostname) num_shards=$NUM_SHARDS allocs=${ALLOCS[*]}"
echo "[orch] $(ts) ckpt=$CKPT out=$OUT cap=$CAP protocol=unseen_topology_v1"

pids=()
allocs_for_pid=()
base=0
for ag in "${ALLOCS[@]}"; do
  alloc="${ag%:*}"; ngpu="${ag#*:}"
  log="$OUT/logs/shard_alloc_${alloc}.log"
  echo "[orch] $(ts) LAUNCH alloc=$alloc ngpu=$ngpu shards=$base..$((base+ngpu-1)) -> $log"
  BASE="$base" NGPU="$ngpu" REPO="$REPO" PY="$PY" CKPT="$CKPT" OUT="$OUT" CAP="$CAP" \
  NUM_SHARDS="$NUM_SHARDS" SPLITS_DIR="$SPLITS_DIR" HOLDOUT_ART="$HOLDOUT_ART" HOLDOUT_SHA="$HOLDOUT_SHA" \
  srun --overlap --jobid="$alloc" --gres=gpu:"$ngpu" --cpus-per-task=8 \
       --ntasks=1 --nodes=1 --no-kill \
       bash -c '
         cd "$REPO"
         ipids=()
         for (( g=0; g<NGPU; g++ )); do
           CUDA_VISIBLE_DEVICES=$g "$PY" scripts/export_graph_vq_tokens.py \
             --frozen_vqvae_ckpt "$CKPT" \
             --out "$OUT" \
             --splits train,val \
             --num_frames 300 \
             --caption_emb_cache "$CAP" \
             --caption_token_cache "$CAP" \
             --caption_token_max_len 113 \
             --texts_json_name motion_texts_by_file_clean_v1.json \
             --min_text_coverage 0.99 \
             --splits_dir "$SPLITS_DIR" \
             --protocol unseen_topology_v1 \
             --holdout_artifact "$HOLDOUT_ART" \
             --holdout_sha "$HOLDOUT_SHA" \
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
