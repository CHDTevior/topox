#!/usr/bin/env bash
# Run the retrieval baselines of the 303M main-table arm inside its own allocation. This half only chooses the
# arm's artifacts and pins the step to a node; the GPU gate is scripts/_gpu_gate_exec.py, which runs on the node
# itself and execs the evaluation with the device it checked (codex 2026-09-10).
#   bash scripts/_run_nonparam_r12.sh <jobid> <node> <gpu-index-within-the-step> smoke|full
set -euo pipefail
JOBID=${1:?jobid}; NODE=${2:?node}; GPU=${3:?gpu index}; MODE=${4:?smoke|full}
REPO=/iridisfs/scratch/ts1v23/workspace/noKslot_clean
CKPT=runs/v2_noik_run12_896_r1acc/best_model.pt
REPORT=runs/v2_noik_run12_896_r1acc/gen_eval_ep289_h200x4_s20_pool64_bound.json
case "$MODE" in
  smoke) OUT=runs/_nonparam/nonparam_r12_ep289_smoke.json; EXTRA=(--limit 256) ;;
  full)  OUT=runs/_nonparam/nonparam_r12_ep289.json;       EXTRA=() ;;
  *) echo "[refuse] mode must be smoke or full"; exit 2 ;;
esac
cd "$REPO"
srun --jobid="$JOBID" --overlap --nodelist="$NODE" --nodes=1 --ntasks=1 --gres=gpu:2 --cpus-per-task=8 \
  /usr/bin/env python scripts/_gpu_gate_exec.py "$GPU" \
    scripts/_eval_nonparametric_baselines_ktjd17.py \
    --ckpt "$CKPT" --report "$REPORT" --out "$OUT" "${EXTRA[@]}"
