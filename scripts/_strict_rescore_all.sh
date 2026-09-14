#!/usr/bin/env bash
# Re-score a saved generation under the STANDARD acceptance rule (the caption's own clip only).
#
# The paper reported R-precision under a relaxed acceptance set; every number is being moved to the rule the rest
# of the field uses, so that the metric reads the same way as a published R-precision and the paper needs no
# prose about what else counts. Scoring only: the same shards, the same plan, the same pools, the same evaluator.
#
# usage: REPORT=<existing gen_eval json> GPU_PIN=0 SKIN_JOBID=<alloc> bash scripts/_strict_rescore_all.sh
set -uo pipefail
cd /iridisfs/scratch/ts1v23/workspace/noKslot_clean
REPORT=${REPORT:?an existing gen-eval report to re-score}
GPU_PIN=${GPU_PIN:?card index}
SKIN_JOBID=${SKIN_JOBID:?alloc id}
GPUS_TOTAL=${GPUS_TOTAL:-4}
OUT=${REPORT%.json}_strict.json
[ -s "$OUT" ] && { echo "[skip] $OUT exists"; exit 0; }

read -r GEN EVAL POOL STEPS CFG SEED SHARDS SUBSET TF32 VARIANT GENBATCH COHORT EVSPLIT <<<"$(python - "$REPORT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); p = d["protocol"]; g = p["generation"]
sh = g.get("shards")
if not isinstance(sh, list):
    sys.exit("[refuse] report was not produced from saved shards")
paths = ",".join(s["path"] for s in sh)
sub = (p.get("subset") or {}).get("path") or "-"
# shards carry the arithmetic they were sampled under, and --merge refuses a runtime that differs
rt = (p.get("runtime") or (g.get("runtime") or {}) or {})
tf32 = "1" if rt.get("allow_tf32_matmul") else "0"
var = p.get("variant") or "-"
gb = g.get("gen_batch") or p.get("gen_batch") or "-"
# a cohort-override report (the held-out study) must be re-scored on its cohort and its split, or --merge refuses its shards
coh = p.get("cohort_exclude_clips") if p.get("cohort_override") else "-"
evs = p.get("eval_split") or "-"
print(p["gen_ckpt"], p["eval_ckpt"], p["pool"], p["steps"], p["cfg_text"], p["seed"], paths, sub, tf32, var, gb, coh or "-", evs)
PY
)" || exit 1

ARGS=(--gen_ckpt "$GEN" --eval_ckpt "$EVAL" --merge "$SHARDS" --pool "$POOL" --steps "$STEPS"
      --cfg_text "$CFG" --seed "$SEED" --strict_acceptance --out "$OUT")
[ "$SUBSET" != "-" ] && ARGS+=(--score_subset "$SUBSET")
[ "$TF32" = 1 ] && ARGS+=(--tf32)
[ "$VARIANT" != "-" ] && ARGS+=(--protocol_variant "$VARIANT")
[ "$GENBATCH" != "-" ] && ARGS+=(--gen_batch "$GENBATCH")
[ "$COHORT" != "-" ] && ARGS+=(--eval_exclude "$COHORT")
[ "$EVSPLIT" = all ] && ARGS+=(--eval_split all)
srun --jobid="$SKIN_JOBID" --overlap --nodes=1 --ntasks=1 --gres=gpu:${GPUS_TOTAL} --cpus-per-task=8 \
  /usr/bin/env python scripts/_gpu_gate_exec.py "$GPU_PIN" scripts/_eval_v2_gen_in_evalspace.py "${ARGS[@]}" \
  > "${OUT%.json}.log" 2>&1 || { echo "[FAIL] $REPORT -- see ${OUT%.json}.log"; exit 1; }
python - "$REPORT" "$OUT" <<'PY'
import json, sys
a, b = (json.load(open(x)) for x in sys.argv[1:3])
print(f"[strict] {sys.argv[1].split('/')[-1]:52s} R@1 {a['text_to_gen']['rprec']['1']:.4f} -> {b['text_to_gen']['rprec']['1']:.4f}"
      f"   ceiling {a['text_to_gt_ceiling']['rprec']['1']:.4f} -> {b['text_to_gt_ceiling']['rprec']['1']:.4f}"
      f"   FID {a['fid_gen_vs_gt']:.5f} -> {b['fid_gen_vs_gt']:.5f}")
PY
