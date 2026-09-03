#!/usr/bin/env bash
# KTJD-17 Step-1 four-arm launcher (2026-08-20). One arm per H200:
#   N  flamingo02 GPU0  plain 17ch (no anchor)          -- new-data baseline
#   A  RETIRED -- rest-centering (2026-08-20) makes the rest pose the origin, subsuming this arm
#   B  blossom02  GPU0  demo anchor + identity_p 0.1    (UMO SOURCE_IDENTITY)
#   D  blossom02  GPU1  two-stage root->body denoiser   (Kimodo/UMO variant D)
# Shared: 32.6M dim384x7 (D: +6.6M root tower), B8 lr3e-4 500ep, graph-v2 both knives,
# CFG drops 0.1x3, random_caption,
# 2026-08-20 UMO-ALIGNMENT (after the frozen-pose collapse: the arms held one pose while the root
# dragged them). Four changes, each traced to UMO's actual recipe rather than to Eq.1's names:
#   * target REST-CENTERED in the adapter -- UMO z-scores (x-mean)/std, KTJD's spec normalization
#     is scale-only, so a constant pose captured 75% of the objective (13ch: 45%). Now 36.5%.
#     (std division is deliberately NOT copied: measured, it doubles the frozen basin here,
#     because KTJD channels still carry static content that UMO's root-relative rep does not.)
#   * gamma_vel/gamma_lock 0.01 -- UMO's clean_root/joint_velocity + foot_lock, the terms a frozen
#     output CANNOT satisfy (train_hy273_raw_flow.py:733-757). We had never ported them.
#   * gamma_fk 0.07 warmup 5000 -- UMO's actual weight; we were running 1.0/1000 (14x too strong),
#     and a static pose is a PERFECT solution of the consistency term, so it was rewarding freezing.
#   * --v_space on -- UMO's velocity-space objective (1/max(1-t,0.05)^2), which up-weights exactly
#     the clean-end timesteps where mean-pose collapse is otherwise cheapest. Implemented earlier,
#     never switched on.
# gammas from configs/ktjd17_gamma_calibration_v2.json (rest-centered; trainer refuses without it).
# Gate: ready_for_training=false overridden via --ktjd_training_authorized (user 2026-08-20,
# recorded in args.json + ktjd_pins); the data artifact is untouched.
# Memory: worst-case J=142 batch measured 77.3G single / 102.4G two-stage on H200 141G.
# Usage: ARM=N|A|B|D bash scripts/_launch_ktjd_arms.sh   (run ON the arm's node)
set -euo pipefail
cd "$(dirname "$0")/.."

ARM="${ARM:?set ARM=N|A|B|D}"
case "$ARM" in
  N) GPU=0; EXTRA=(--anchor none) ;;
  A) echo "arm A (rest anchor) is SUBSUMED by rest-centering -- see train_v2_incontext.py"; exit 1 ;;
  B) GPU=0; EXTRA=(--anchor demo --identity_p 0.1) ;;
  D) GPU=1; EXTRA=(--anchor none --two_stage --root_dim 192) ;;
  *) echo "bad ARM=$ARM"; exit 1 ;;
esac
# FRESH LINEAGE (codex round-S2 #7): the 2026-08-20 UMO-alignment changes the OBJECTIVE, so the
# retained runs/v2_ktjd_arm_*_30m checkpoints belong to a different one. Auto-resume would pick one
# up and the trainer's pin check would (correctly) refuse to start. New run dir = new lineage; the
# old dirs stay untouched as the frozen-collapse archive.
OUT="runs/v2_ktjd_${ARM}_umoalign"
mkdir -p "$OUT"

exec 9>"$OUT/.launch.lock"
flock -n 9 || { echo "[launch] $ARM already running (lock held)"; exit 0; }

RESUME=()
[ -f "$OUT/last_model.pt" ] && RESUME=(--resume "$OUT/last_model.pt")

export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_ALLOC_CONF=expandable_segments:True
echo "[launch] arm $ARM on GPU $GPU -> $OUT (resume: ${RESUME[*]:-fresh})"
exec python3 scripts/train_v2_incontext.py \
  --corpus ktjd17 --ktjd_training_authorized \
  --dim 384 --depth 7 --heads 8 --batch 8 --lr 3e-4 --epochs 500 \
  --val_every 5 --ckpt_every 50 --num_workers 4 --seed 0 --bf16 \
  --struct_feats --dir_bias \
  --gamma_fk 0.07 --fk_warmup_steps 5000 --gamma_vel 0.01 --gamma_lock 0.01 --v_space \
  --p_drop_text 0.1 --p_drop_demo 0.1 --p_drop_both 0.1 --random_caption \
  "${EXTRA[@]}" "${RESUME[@]}" \
  --out "$OUT" >> "$OUT/train.log" 2>&1
