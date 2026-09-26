#!/bin/bash
# MultiModality generation: R repeats of ONE rig shard of the frozen protocol, differing only in the sampling seed.
# Everything that produces samples is the canonical scripts/_eval_v2_gen_in_evalspace.py --save_gen, unchanged, so a
# repeat is an ordinary protocol shard carrying ordinary shard metadata; this script only schedules those calls.
#
# WHY A RIG SHARD and not a clip list: the noise of a sample is a function of the generation PLAN (torch.manual_seed
# per plan chunk), and --nshards only changes which rigs a process takes, never how any rig is chunked. So shard S of
# N is exactly what the full pass would produce for those rigs, and the repeats stay on the canonical path.
#
# COMPLETION IS THE EXIT STATUS. manifest.json is deleted on entry and written only when every requested seed is
# present, so a partial pass cannot be mistaken for a whole one by the scorer (codex divmm r1 P1-1, r2 P1).
#
# usage: _mm_gen_queue.sh <alloc> <label> <ckpt> <NG> <NSHARDS> <SHARD> <seed> [seed ...]
set -uo pipefail
cd /iridisfs/scratch/ts1v23/workspace/noKslot_clean
J=$1; LABEL=$2; CK=$3; NG=$4; NS=$5; SH=$6; shift 6; SEEDS=("$@")
EV=runs/evaluator_ktjd16_pz_v1/best_model.pt
[[ "$J" =~ ^[0-9]+$ ]] || { echo "alloc id must be numeric, got '$J'"; exit 1; }
[[ "$LABEL" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "LABEL must be a plain identifier, got '$LABEL'"; exit 1; }
# INVALIDATE THE MANIFEST HERE -- the first moment the label is known and therefore the first moment the directory is
# known. Every check below can exit, and an exit that happens while a previous request's manifest is still on disk
# leaves that older, possibly different-shard run scoreable (codex divmm r2 P1, r3 P1). rm is checked, and the
# absence is then verified, because `set -uo pipefail` does not stop on a failed rm.
D=runs/_divmm/mm/$LABEL
# rm BEFORE mkdir: rm -f is happy with a path that does not exist, while a failing mkdir (say .aris/meta is a file)
# would exit with the old manifest still on disk and still scoreable (codex divmm r4).
rm -f "$D/manifest.json"
[ ! -e "$D/manifest.json" ] || { echo "could not remove the previous $D/manifest.json -- refusing to run, because an older request would stay scoreable"; exit 1; }
mkdir -p "$D" .aris/meta || { echo "cannot create $D"; exit 1; }
[ -s "$CK" ] || { echo "checkpoint '$CK' is not a readable file"; exit 1; }
[[ "$NG" =~ ^[1-9][0-9]*$ ]] || { echo "NG must be a positive integer, got '$NG'"; exit 1; }
[[ "$NS" =~ ^[1-9][0-9]*$ ]] || { echo "NSHARDS must be a positive integer, got '$NS'"; exit 1; }
[[ "$SH" =~ ^(0|[1-9][0-9]*)$ ]] && [ "$SH" -lt "$NS" ] || { echo "SHARD must lie in [0,NSHARDS), got '$SH'"; exit 1; }
[ ${#SEEDS[@]} -ge 1 ] || { echo "give at least one sampling seed"; exit 1; }
for s in "${SEEDS[@]}"; do [[ "$s" =~ ^(0|[1-9][0-9]*)$ ]] || { echo "seed '$s' is not a non-negative integer"; exit 1; }; done
# repeated seeds would be the same samples counted twice; the scorer refuses them, so refuse before burning cards
[ "$(printf '%s\n' "${SEEDS[@]}" | sort -u | wc -l)" = "${#SEEDS[@]}" ] || { echo "the seed list repeats a seed"; exit 1; }
[ -z "${NVIDIA_TF32_OVERRIDE+x}" ] || { echo "NVIDIA_TF32_OVERRIDE is set -- unset it; the runtime record cannot see it"; exit 1; }

ENV="/usr/bin/env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
CKSHA=$(sha256sum "$CK" | cut -d" " -f1)
[[ "$CKSHA" =~ ^[0-9a-f]{64}$ ]] || { echo "could not digest the checkpoint '$CK'"; exit 1; }
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$D/chain.log"; }
SR="srun --jobid=$J --overlap --gres=gpu:$NG -N1 -n1 --mem=64G --cpus-per-task=8"
take_lock() { local __fd; exec {__fd}>"$1"
  if flock -n "$__fd"; then printf -v "$2" '%s' "$__fd"; return 0; fi
  exec {__fd}>&-; return 1; }
card_lock() { take_lock ".aris/meta/.gpu_pin_${J}_$1.lock" "$2"; }
# The card is addressed by UUID, not by ordinal. nvidia-smi's NVML index and CUDA's device ordinal need not agree
# (CUDA orders by speed unless told otherwise), so gating index k and then running on ordinal k could have gated one
# card and used another. CUDA accepts a UUID in CUDA_VISIBLE_DEVICES, which makes the gated and the used device the
# same device by construction (codex divmm r2 P2-9).
uuid_of() { $SR nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$1" 2>/dev/null | tr -d '[:space:]'; }
# fail-CLOSED gate: a failed query, an unparseable uuid, or any resident process counts as busy
gate() { local u=$1 out n
  [[ "$u" =~ ^GPU-[0-9a-f-]{36}$ ]] || { log "gate: '$u' is not a GPU uuid"; return 1; }
  out=$($SR nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>&1) \
    || { log "gate: compute-apps query failed (rc=$?): $(echo "$out" | tail -n 1)"; return 1; }
  n=$(printf '%s\n' "$out" | grep -c -F "$u"); [ "$n" = 0 ]; }
finished() { [ -s "$D/rep_seed$1.npz" ] && tail -n 1 "$D/rep_seed$1.log" 2>/dev/null | grep -qF -- "-> $D/rep_seed$1.npz"; }
# A reused repeat must be THIS pass: the checkpoint BYTES, the rig shard, the guidance, the steps, the seed and TF32.
# Path/steps/seed alone let a changed SHARD reuse a different caption subset (codex divmm r1 P1-2).
same_pass() { $ENV python - "$D/rep_seed$1.npz" "$CKSHA" "$1" "$NS" "$SH" <<'PY'
import json, sys, numpy as np
m = json.loads(str(np.load(sys.argv[1], allow_pickle=False)["__meta"]))
rt = m.get("runtime")
sys.exit(0 if (m.get("gen_ckpt_sha256") == sys.argv[2]
               and type(m.get("steps")) is int and m["steps"] == 20
               and float(m.get("cfg_text", -1)) == 2.0
               and type(m.get("seed")) is int and m["seed"] == int(sys.argv[3])
               and type(m.get("nshards")) is int and m["nshards"] == int(sys.argv[4])
               and type(m.get("shard")) is int and m["shard"] == int(sys.argv[5])
               and isinstance(rt, dict) and rt.get("allow_tf32_matmul") is True) else 1)
PY
}

todo=(); bad=0
for s in "${SEEDS[@]}"; do
  if finished "$s"; then
    same_pass "$s" && log "seed $s finished earlier -- reused" \
      || { log "seed $s was generated for another ckpt/steps/shard/tf32 setting -- REFUSED (use a fresh LABEL)"; bad=1; }
  else todo+=("$s"); fi
done
[ $bad = 0 ] || { log "refusing: existing files under this label are not this request"; exit 1; }
if [ ${#todo[@]} -gt 0 ]; then
  log "start ckpt=$CK sha=${CKSHA:0:12} cards=$NG nshards=$NS shard=$SH repeats_todo=${todo[*]}"
  pids=()
  for k in $(seq 0 $((NG-1))); do
    ( card_lock "$k" LFD || { log "GPU$k is locked by another launcher -- worker $k stops; its repeats stay for a re-run"; exit 0; }
      U=$(uuid_of "$k")
      [[ "$U" =~ ^GPU-[0-9a-f-]{36}$ ]] || { log "could not resolve a uuid for GPU$k ('$U') -- worker $k stops"; exit 0; }
      i=0; for s in "${todo[@]}"; do
        if [ $((i % NG)) = "$k" ]; then
          gate "$U" || { log "GPU$k ($U) busy before seed $s -- worker $k stops; remaining repeats stay for a re-run"; break; }
          VAR=(); [ "$s" != 42 ] && VAR=(--protocol_variant seed)   # the eval script refuses an unstamped non-42 seed
          if timeout 21600 $SR $ENV CUDA_VISIBLE_DEVICES="$U" python scripts/_eval_v2_gen_in_evalspace.py \
               --gen_ckpt "$CK" --eval_ckpt $EV --steps 20 --tf32 "${VAR[@]}" --seed "$s" \
               --nshards "$NS" --shard "$SH" --save_gen "$D/rep_seed$s.npz" > "$D/rep_seed$s.log" 2>&1; then
            log "seed $s done on GPU$k"
          else log "seed $s FAILED on GPU$k (rc=$?; see $D/rep_seed$s.log)"; fi
        fi
        i=$((i+1)); done ) & pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
fi
missing=(); for s in "${SEEDS[@]}"; do finished "$s" || missing+=("$s"); done
if [ ${#missing[@]} != 0 ]; then
  log "repeats still missing: ${missing[*]} -- re-run this label on free cards to continue"
  exit 1
fi
# json.dump, not printf: a quote in a path would make invalid JSON and a backslash-t would silently become a TAB,
# recording a path that is not the one used (codex divmm r2 P2). Publication is checked, not assumed.
$ENV python - "$D/manifest.json.tmp" "$LABEL" "$CK" "$CKSHA" "$NS" "$SH" "${SEEDS[@]}" <<'PY'
import json, sys, datetime
out, label, ck, cksha, ns, sh, *seeds = sys.argv[1:]
doc = {"format": "mm-repeat-manifest-v1", "label": label, "gen_ckpt": ck, "gen_ckpt_sha256": cksha,
       "steps": 20, "cfg_text": 2.0, "tf32": True, "nshards": int(ns), "shard": int(sh),
       "seeds": [int(s) for s in seeds],
       "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
with open(out, "w") as f:          # `with`, so a flush/close failure raises instead of being swallowed at GC
    json.dump(doc, f, indent=1, allow_nan=False)
    f.flush()
PY
[ $? = 0 ] || { log "manifest could not be written -- this pass is NOT scoreable"; rm -f "$D/manifest.json.tmp"; exit 1; }
mv "$D/manifest.json.tmp" "$D/manifest.json" \
  || { log "manifest could not be published -- this pass is NOT scoreable"; rm -f "$D/manifest.json.tmp"; exit 1; }
log "ALL-REPEATS-DONE ${#SEEDS[@]} in $D (manifest $D/manifest.json)"
