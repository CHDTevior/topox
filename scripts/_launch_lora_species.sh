#!/usr/bin/env bash
# Per-species LoRA fine-tune of the run12 backbone on ONE TrueBones rig (user 2026-09-02).
#   SKIN_JOBID=<alloc> RIG=Buffalo BATCH=5 bash scripts/_launch_lora_species.sh [SMOKE=1]
# BATCH must divide the rig's n_train (rig_table.json); LR is derived by linear scaling from B8/1e-4.
# Single GPU, one srun step inside our own allocation. Every objective / architecture flag is the
# run12 keeper recipe (configs/run12_896_r1acc_env.sh) verbatim -- the only differences are the
# corpus (TrueBones LoRA corpus, one rig via the exclusion artifact), the per-rig calibration and
# normalization artifacts, and the LoRA / init_from / lr / schedule knobs listed under "LoRA".
set -uo pipefail
: "${SKIN_JOBID:?alloc id}"; : "${RIG:?TrueBones rig id, e.g. Buffalo}"
cd /scratch/ts1v23/workspace/noKslot_clean
SMOKE=${SMOKE:-0}
INIT=${INIT:-runs/v2_noik_run12_896_r1acc/best_model.pt}
# OUT defaults to a name derived from the data VIEW so two views can never share a run dir
# (codex 2026-09-02 r7 #6): dataset/ktjd17_truebones_lora_v1 -> runs/lora_tb_<rig>_r64_v1,
# dataset/ktjd17_truebones_lora_v2_mainbody -> runs/lora_tb_<rig>_r64_v2_mainbody.
KTJD_ROOT=${KTJD_ROOT:-dataset/ktjd17_truebones_lora_v1}
VIEW_TAG=$(basename "$KTJD_ROOT"); VIEW_TAG=${VIEW_TAG#ktjd17_truebones_lora_}
OUT=${OUT:-runs/lora_tb_${RIG}_r64_${VIEW_TAG}}
CALIB=${CALIB:-configs/tb_$(echo "$RIG" | tr 'A-Z' 'a-z')_gamma_calibration_v1.json}
CUT=configs/tb_lora_${RIG}_only_exclusions.json   # written only for ELIGIBLE rigs by the builder
# Data view (defaults = the v1 view). The pruned "main body" view v2 (user 2026-09-02) is selected by
# KTJD_ROOT/PERCELL/JOINT_SEM together with its own CALIB; the three must belong to ONE view --
# Ktjd17Base verifies the stats / rig_table / exclusion shas against the view's derivation.json.
PERCELL=${PERCELL:-data/tb_norm_stats_v2.npz}
JOINT_SEM=${JOINT_SEM:-data/joint_semantics_llm2vec_ktjd17_v1.npz}
for f in "$INIT" "$CALIB" "$CUT" "$PERCELL" "$JOINT_SEM" "$KTJD_ROOT/derivation.json" data/tb_caption_llm2vec_pzstyle_v1.keys.json \
         data/tb_motion_texts_pzstyle_v1.json; do
  [ -s "$f" ] || { echo "[lora] missing $f"; exit 1; }
done
# LoRA knobs (rank 64 per user; alpha = r -> scale 1; lr 1e-4 is the usual LoRA lr, ~0.7x the
# backbone's peak 1.5e-4; short constant-then-cosine schedule sized for ~20 clips)
LORA_R=${LORA_R:-64}; LORA_ALPHA=${LORA_ALPHA:-64}; LORA_TARGETS=${LORA_TARGETS:-attn,ffn,cond}
# Launch contract (codex 2026-09-02): the batch must DIVIDE the rig's train count so that, with
# --balance clip (each clip exactly once per epoch, no replacement) and drop_last, an epoch is
# every clip once; the lr follows the linear scaling rule from the B8 / 1e-4 reference and is NOT
# free -- an LR override that breaks the rule is refused.
BATCH=${BATCH:-8}; LR_REF=1e-4; BATCH_REF=8
LR=$(python -c "print(f'{${LR_REF} * ${BATCH} / ${BATCH_REF}:.6g}')")
if [ -n "${LR_OVERRIDE:-}" ]; then LR=$LR_OVERRIDE; echo "[lora] WARNING: LR_OVERRIDE=$LR breaks the linear-scaling contract on purpose"; fi
EPOCHS=${EPOCHS:-300}; WARMUP=${WARMUP:-100}; LR_DECAY_EPOCHS=${LR_DECAY_EPOCHS:-250}
# FK term warmup: run12 used 5000 steps over ~100k; a 600-step LoRA run would end at 12% of the FK
# weight while validation uses the full weight -- so the warmup is sized to this run (codex 2026-09-02).
FK_WARMUP=${FK_WARMUP:-100}
# torch.compile is deliberately OFF here (run12 had COMPILE=1): ~76 s of compilation is not worth it
# for a ~600-step run, and LoRA + compile has no smoke of its own yet. Throughput-only difference.
[ "$SMOKE" = 1 ] && { EPOCHS=2; OUT=${OUT}_smoke; }
# never write into a run dir that already holds a training log (a second launch would truncate
# the first run's train.log through tee); remove it explicitly or pick another OUT
if [ -e "$OUT/train.log" ] && [ "${FORCE_OUT:-0}" != 1 ]; then echo "[lora] refuse: $OUT/train.log exists (set OUT=... or FORCE_OUT=1)"; exit 1; fi
# the rig must have at least one full training batch and a source-disjoint val set (rig_table.json)
python - "$RIG" "$BATCH" "$KTJD_ROOT" << 'RIGCHECK'
import json, sys
rig, batch = sys.argv[1], int(sys.argv[2])
t = json.load(open(sys.argv[3] + "/splits/lora_v1/rig_table.json"))
if rig not in t: raise SystemExit(f"[lora] refuse: rig {rig} not in rig_table.json")
r = t[rig]
if not r["eligible"] or r["n_train"] < batch or r["n_val"] < 1:
    raise SystemExit(f"[lora] refuse: rig {rig} is not eligible for a per-species LoRA: {r}")
if r["n_train"] % batch:
    divs = [b for b in range(2, 9) if r["n_train"] % b == 0]
    raise SystemExit(f"[lora] refuse: BATCH={batch} does not divide n_train={r['n_train']} (choose BATCH in {divs}); "
                     f"with drop_last some clips would never be seen in an epoch")
print(f"[lora] rig table: {rig} usable {r['usable']} = train {r['n_train']} (source groups {r['source_groups']}) "
      f"+ val {r['n_val']} -> {r['n_train'] // batch} steps/epoch at batch {batch}, every clip once per epoch")
RIGCHECK
[ $? -eq 0 ] || exit 1
mkdir -p "$OUT"
# The TrueBones release gate says ready_for_training=false only because the post-build visual
# regression was never marked complete in the artifact, while its own visual gate (66/66 rigs,
# verdict pass) and fixed QA (986 clips, 0 fail) both passed. Training here is the user's explicit
# OOD experiment (2026-09-02); the flag records that authorization without touching the artifact.
echo "[lora] rig=$RIG out=$OUT view=$KTJD_ROOT stats=$PERCELL sem=$JOINT_SEM init=$INIT calib=$CALIB cut=$CUT r=$LORA_R alpha=$LORA_ALPHA targets=$LORA_TARGETS lr=$LR epochs=$EPOCHS"
srun --jobid="$SKIN_JOBID" --ntasks=1 --cpus-per-task=${CPUS:-8} --gres=gpu:1 --mem=${MEM:-96G} \
  /usr/bin/env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/train_v2_incontext.py \
  --out "$OUT" --corpus ktjd17 --ktjd_root "$KTJD_ROOT" \
  --joint_sem "$JOINT_SEM" --caption_cache data/tb_caption_llm2vec_pzstyle_v1 \
  --texts_json data/tb_motion_texts_pzstyle_v1.json --ktjd_percell_stats "$PERCELL" \
  --ktjd_gamma_calib "$CALIB" --exclude_clips "$CUT" \
  --dim 896 --depth 14 --heads 14 --qk_norm --grad_ckpt \
  --lr "$LR" --batch "$BATCH" --epochs "$EPOCHS" --warmup_steps "$WARMUP" --wd 0.01 --grad_clip 1.0 \
  --lr_scheduler half_cosine --lr_decay_epochs "$LR_DECAY_EPOCHS" --eta_min_ratio 0.01 --grad_spike_reject 200 \
  --v_space --sigma_min 0.2 --huber_delta 10 --bf16 --t_sampler uniform \
  --gamma_fk 0.07 --fk_warmup_steps "$FK_WARMUP" --gamma_vel 0.01 --gamma_lock 0.01 --gamma_acc 1.0 \
  --demo_rest --demo_frames 1 --struct_feats --dir_bias --anchor none --balance clip \
  --p_drop_text 0.1 --p_drop_demo 0.0 --p_drop_both 0.0 --target_frames 240 \
  --init_from "$INIT" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_targets "$LORA_TARGETS" \
  ${AUTH_FLAG:---ktjd_training_authorized} \
  ${EXTRA:-} 2>&1 | tee "$OUT/train.log" | { grep -E '^\[(lora|init_from|train|resume|ktjd|calib)\]|=== epoch|\[val\]|SPIKE|FATAL|refuse|Traceback|Error|error' || true; } | cut -c1-200
rc=${PIPESTATUS[0]}
echo "[lora] rc=$rc (full log: $OUT/train.log)"
exit "$rc"
