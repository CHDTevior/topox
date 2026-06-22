#!/usr/bin/env bash
# Durable cross-alloc PARALLEL Graph-CodeFlow RVQ token export for the L4safe+HumanML3D
# dataset (full-length 300), frozen on the n8192 VQVAE best ckpt (ep239, val 1.3386).
#
# VERBATIM copy of scripts/_run_export_parallel_mergedL4TB.sh except CKPT/OUT/CAP/ALLOCS:
#   - CKPT = the n8192 L4safeHuman VQVAE best_model.pt (ep239) — the user-chosen frozen tokenizer.
#   - OUT  = data/codeflow_tokens_L4safeHuman_n8192_ep239_fulllen300
#   - CAP  = data/anytop_caption_t5_l4safe_human_multi  (the L4safeHuman T5 cache; coverage-PASS 1.0)
#   - ALLOCS = the two freed 2xH200 allocs (flamingo01 1014952 + blossom03 1014950) -> NUM_SHARDS=4.
# Mechanism unchanged: each alloc's srun runs one export proc per local GPU with a cumulative
# global shard_idx; per-clip npz keeps its {i:06d}.npz dataset-index filename so all shards write
# disjoint files into the SAME split dir; each shard writes index_shard{idx:03d}.jsonl; merged at end.
#
# Caller wraps with setsid nohup on a COMPUTE node (PPID=1, survives ssh drop):
#   ssh <compute> "cd $REPO && setsid nohup bash scripts/_run_export_parallel_l4safe_human.sh \
#                  > $OUT/orch.log 2>&1 </dev/null &"
set -uo pipefail

REPO=/scratch/ts1v23/workspace/noKslot_clean
PY=/scratch/ts1v23/.conda/bin/python3
CKPT=runs/vqvae_L4safeHuman_C72_J144_d512_Q4_n8192_b16g64_300ep_seed42/best_model.pt
OUT=/scratch/ts1v23/workspace/noKslot_clean/data/codeflow_tokens_L4safeHuman_n8192_ep239_fulllen300
CAP=data/anytop_caption_t5_l4safe_human_multi

# "jobid:ngpu" per idle alloc. Shard base assigned cumulatively in list order.
#   1014952 flamingo01 (2xH200)   1014950 blossom03 (2xH200)  -> 4 GPUs total -> NUM_SHARDS=4
ALLOCS=("1014952:2" "1014950:2")

NUM_SHARDS=0
for ag in "${ALLOCS[@]}"; do NUM_SHARDS=$(( NUM_SHARDS + ${ag#*:} )); done

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

cd "$REPO"
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
         ipids=()
         for (( g=0; g<NGPU; g++ )); do
           CUDA_VISIBLE_DEVICES=$g "$PY" scripts/export_graph_vq_tokens.py \
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
