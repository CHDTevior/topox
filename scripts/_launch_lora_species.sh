#!/usr/bin/env bash
# Per-species LoRA fine-tune of the run12 backbone on ONE TrueBones rig (user 2026-09-02).
#   SKIN_JOBID=<alloc> RIG=Buffalo BATCH=5 bash scripts/_launch_lora_species.sh [SMOKE=1]
# BATCH must divide the rig's n_train (rig_table.json); LR is derived by linear scaling from B8/1e-4.
# Single GPU, one srun step inside our own allocation. Every objective / architecture flag is the
# run12 keeper recipe (configs/run12_896_r1acc_env.sh) verbatim -- the only differences are the
# corpus (TrueBones LoRA corpus, one rig via the exclusion artifact), the per-rig calibration and
# normalization artifacts, and the LoRA / init_from / lr / schedule knobs listed under "LoRA".
set -uo pipefail
cd /scratch/ts1v23/workspace/noKslot_clean
# VIEW profile (codex 2026-09-03 r4): a derived view needs ITS calibration, ITS main-body sidecars and ITS
# split/group settings -- VIEW_ENV=configs/lora_view_<name>.env holds those defaults in one tracked place
# (every value stays overridable by an explicit env var). PREFLIGHT=1 resolves everything, prints the
# trainer argv and exits before srun: the launch that runs is the launch that was checked.
PREFLIGHT=${PREFLIGHT:-0}
if [ -n "${VIEW_ENV:-}" ]; then [ -s "$VIEW_ENV" ] || { echo "[lora] missing VIEW_ENV $VIEW_ENV"; exit 1; }; source "$VIEW_ENV"; fi
# BB_ENV=configs/lora_backbone_<name>.env selects the BACKBONE (init ckpt + arch + demo condition); default = run12
if [ -n "${BB_ENV:-}" ]; then [ -s "$BB_ENV" ] || { echo "[lora] missing BB_ENV $BB_ENV"; exit 1; }; source "$BB_ENV"; fi
[ "$PREFLIGHT" = 1 ] || : "${SKIN_JOBID:?alloc id}"
# GROUP mode (user 2026-09-03 "flying group"): GROUP=<name> trains ONE LoRA on the usable clips of several
# rigs; the cut is configs/tb_lora_group_<name>_exclusions.json (written by scripts/_make_tb_group_exclusion.py
# and DECLARED by the view's derivation.json). GROUP_RIGS defaults to the cut artifact's group_rigs and, when
# given explicitly, must equal it; the rig-table check sums the group's train/val counts. RIG doubles as the
# run-name stem.
if [ -n "${GROUP:-}" ]; then RIG=group_${GROUP}; fi
: "${RIG:?TrueBones rig id, e.g. Buffalo (or GROUP=...)}"
SMOKE=${SMOKE:-0}
INIT=${INIT:-runs/v2_noik_run12_896_r1acc/best_model.pt}
# Backbone architecture and demo condition (user 2026-09-04: repeat zero-shot + LoRA on the 36M rest-demo and
# demo-64 pilots). --init_from loads STRICTLY, so DIM/DEPTH/HEADS must equal the backbone's; DEMO_REST/DEMO_FRAMES
# must equal what the backbone was trained with (the trainer's calibration gate also binds them to CALIB).
# Defaults reproduce the run12 launches byte-for-byte (896/14/14, qk-norm, grad_ckpt, 1-frame rest demo).
DIM=${DIM:-896}; DEPTH=${DEPTH:-14}; HEADS=${HEADS:-14}; QK_NORM=${QK_NORM:-1}; GRAD_CKPT=${GRAD_CKPT:-1}
DEMO_REST=${DEMO_REST:-1}; DEMO_FRAMES=${DEMO_FRAMES:-1}
for v in DEMO_REST QK_NORM GRAD_CKPT; do case "${!v}" in 0|1) ;; *) echo "[lora] $v must be 0 or 1 (got '${!v}')"; exit 1;; esac; done
for v in DIM DEPTH HEADS DEMO_FRAMES; do [[ "${!v}" =~ ^[1-9][0-9]*$ ]] || { echo "[lora] $v must be a positive integer (got '${!v}')"; exit 1; }; done
if [ "$DEMO_REST" = 1 ] && [ "$DEMO_FRAMES" != 1 ]; then echo "[lora] rest demo implies DEMO_FRAMES=1"; exit 1; fi
# LoRA knobs (rank 64 per user; alpha defaults to r -> scale 1 whatever the rank (codex 2026-09-03 r3);
# lr 1e-4 is the usual LoRA lr, ~0.7x the backbone's peak 1.5e-4; short constant-then-cosine schedule)
LORA_R=${LORA_R:-64}; LORA_ALPHA=${LORA_ALPHA:-$LORA_R}; LORA_TARGETS=${LORA_TARGETS:-attn,ffn,cond}
# OUT defaults to a name derived from the RANK and the data VIEW so two views / two ranks can never
# share a run dir (codex 2026-09-02 r7 #6, 2026-09-03 r3): dataset/ktjd17_truebones_lora_v1 ->
# runs/lora_tb_<rig>_r64_v1, dataset/ktjd17_truebones_lora_v2_mainbody -> runs/lora_tb_<rig>_r128_v2_mainbody.
KTJD_ROOT=${KTJD_ROOT:-dataset/ktjd17_truebones_lora_v1}
VIEW_TAG=$(basename "$KTJD_ROOT"); VIEW_TAG=${VIEW_TAG#ktjd17_truebones_lora_}
OUT=${OUT:-runs/lora_tb_${RIG}_r${LORA_R}_${VIEW_TAG}${BB_TAG:+_$BB_TAG}}
BATCH=${BATCH:-8}
# Calibration artifact: the main-body view binds it to (rig, LoRA batch, demo condition) -- the trainer refuses a
# batch or demo mismatch -- so it is derived here unless CALIB is given; other views keep the legacy default.
_calib_tag=$([ "$DEMO_REST" = 1 ] && echo rest1 || echo demo${DEMO_FRAMES})
if [ -z "${CALIB:-}" ] && [ "$VIEW_TAG" = "v2_mainbody" ]; then
  CALIB=configs/tb_$(echo "$RIG" | tr 'A-Z' 'a-z')_mainbody_gamma_calibration_b${BATCH}_${_calib_tag}_v1.json
fi
CALIB=${CALIB:-configs/tb_$(echo "$RIG" | tr 'A-Z' 'a-z')_gamma_calibration_v1.json}
CUT=configs/tb_lora_${RIG}_only_exclusions.json   # written only for ELIGIBLE rigs by the builder
[ -n "${GROUP:-}" ] && CUT=configs/tb_lora_group_${GROUP}_exclusions.json
if [ -n "${GROUP:-}" ] && [ -z "${GROUP_RIGS:-}" ]; then
  [ -s "$CUT" ] || { echo "[lora] missing $CUT"; exit 1; }
  GROUP_RIGS=$(python -c "import json, sys; print(','.join(json.load(open(sys.argv[1]))['group_rigs']))" "$CUT") || exit 1
  echo "[lora] GROUP_RIGS taken from $CUT: $GROUP_RIGS"
fi
# Data view (defaults = the v1 view). The pruned "main body" view v2 (user 2026-09-02) is selected by
# KTJD_ROOT/PERCELL/JOINT_SEM together with its own CALIB; the three must belong to ONE view --
# Ktjd17Base verifies the stats / rig_table / exclusion shas against the view's derivation.json.
PERCELL=${PERCELL:-data/tb_norm_stats_v2.npz}
JOINT_SEM=${JOINT_SEM:-data/joint_semantics_llm2vec_ktjd17_v1.npz}
for f in "$INIT" "$CALIB" "$CUT" "$PERCELL" "$JOINT_SEM" "$KTJD_ROOT/derivation.json" data/tb_caption_llm2vec_pzstyle_v1.keys.json \
         data/tb_motion_texts_pzstyle_v1.json; do
  [ -s "$f" ] || { echo "[lora] missing $f"; exit 1; }
done
# periodic epNNNN snapshots are OFF for LoRA runs (best_model.pt / last_model.pt suffice; 12-36 x 1.3-1.9 GB per
# run otherwise -- 166 GB found 2026-09-03). CKPT_EVERY=<epochs> re-enables them.
# Launch contract (codex 2026-09-02): the batch must DIVIDE the rig's train count so that, with
# --balance clip (each clip exactly once per epoch, no replacement) and drop_last, an epoch is
# every clip once; the lr follows the linear scaling rule from the B8 / 1e-4 reference and is NOT
# free -- an LR override that breaks the rule is refused.
LR_REF=1e-4; BATCH_REF=8
LR=$(python -c "print(f'{${LR_REF} * ${BATCH} / ${BATCH_REF}:.6g}')")
if [ -n "${LR_OVERRIDE:-}" ]; then LR=$LR_OVERRIDE; echo "[lora] WARNING: LR_OVERRIDE=$LR breaks the linear-scaling contract on purpose"; fi
# the cosine decay spans 5/6 of the run unless pinned (300 -> 250 as before; 900 -> 750; 200 -> 166):
# a 250-epoch default would leave a 900-epoch run flat at the floor for 650 epochs (codex 2026-09-03 r3)
EPOCHS=${EPOCHS:-300}; WARMUP=${WARMUP:-100}; LR_DECAY_EPOCHS=${LR_DECAY_EPOCHS:-$((EPOCHS * 5 / 6))}
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
python - "$RIG" "$BATCH" "$KTJD_ROOT" "${GROUP_RIGS:-}" "$CUT" << 'RIGCHECK'
import json, sys
rig, batch = sys.argv[1], int(sys.argv[2])
t = json.load(open(sys.argv[3] + "/splits/lora_v1/rig_table.json"))
group = [g.strip() for g in sys.argv[4].split(",") if g.strip()]
if group:
    if len(set(group)) != len(group): raise SystemExit(f"[lora] refuse: GROUP_RIGS has duplicates: {group}")
    cut = json.load(open(sys.argv[5]))
    if sorted(cut.get("group_rigs") or []) != sorted(group):
        raise SystemExit(f"[lora] refuse: GROUP_RIGS {sorted(group)} != the cut artifact's group_rigs "
                         f"{sorted(cut.get('group_rigs') or [])} ({sys.argv[5]})")
    miss = [g for g in group if g not in t]
    if miss: raise SystemExit(f"[lora] refuse: group rigs not in rig_table.json: {miss}")
    gc = cut.get("group_counts") or {}
    want_tr, want_va = sum(t[g]["n_train"] for g in group), sum(t[g]["n_val"] for g in group)
    if (gc.get("train"), gc.get("val")) != (want_tr, want_va):
        raise SystemExit(f"[lora] refuse: cut artifact counts train/val {gc.get('train')}/{gc.get('val')} != rig_table "
                         f"{want_tr}/{want_va} -- the artifact was built for another view/split")
    r = {"usable": sum(t[g]["usable"] for g in group), "source_groups": sum(t[g]["source_groups"] for g in group),
         "n_train": sum(t[g]["n_train"] for g in group), "n_val": sum(t[g]["n_val"] for g in group), "eligible": True,
         "reason": "", "group": group}
elif rig not in t:
    raise SystemExit(f"[lora] refuse: rig {rig} not in rig_table.json")
else:
    r = t[rig]
import os
allow_no_val = os.environ.get("ALLOW_NO_VAL") == "1"
if r["n_val"] < 1 and allow_no_val and r.get("all_train_override") and r["n_train"] >= batch:
    # all-train view (scripts/_derive_tb_view_split_override.py): no val set by design; the trainer
    # skips validation and writes last_model.pt only (user 2026-09-03)
    print(f"[lora] NOTE: ALLOW_NO_VAL=1 -- rig {rig} trains on all {r['n_train']} clips, no validation, no best_model.pt")
elif not r["eligible"] or r["n_train"] < batch or r["n_val"] < 1:
    raise SystemExit(f"[lora] refuse: rig {rig} is not eligible for a per-species LoRA: {r}")
if r["n_train"] % batch:
    divs = [b for b in range(2, 9) if r["n_train"] % b == 0]
    raise SystemExit(f"[lora] refuse: BATCH={batch} does not divide n_train={r['n_train']} (choose BATCH in {divs}); "
                     f"with drop_last some clips would never be seen in an epoch")
print(f"[lora] rig table: {rig} usable {r['usable']} = train {r['n_train']} (source groups {r['source_groups']}) "
      f"+ val {r['n_val']} -> {r['n_train'] // batch} steps/epoch at batch {batch}, every clip once per epoch")
RIGCHECK
[ $? -eq 0 ] || exit 1
# The TrueBones release gate says ready_for_training=false only because the post-build visual
# regression was never marked complete in the artifact, while its own visual gate (66/66 rigs,
# verdict pass) and fixed QA (986 clips, 0 fail) both passed. Training here is the user's explicit
# OOD experiment (2026-09-02); the flag records that authorization without touching the artifact.
echo "[lora] backbone dim=$DIM depth=$DEPTH heads=$HEADS qk_norm=$QK_NORM grad_ckpt=$GRAD_CKPT demo_rest=$DEMO_REST demo_frames=$DEMO_FRAMES"
echo "[lora] rig=$RIG out=$OUT view=$KTJD_ROOT stats=$PERCELL sem=$JOINT_SEM init=$INIT calib=$CALIB cut=$CUT r=$LORA_R alpha=$LORA_ALPHA targets=$LORA_TARGETS lr=$LR epochs=$EPOCHS warmup_steps=$WARMUP lr_decay_epochs=$LR_DECAY_EPOCHS"
CMD=(python scripts/train_v2_incontext.py
  --out "$OUT" --corpus ktjd17 --ktjd_root "$KTJD_ROOT"
  --joint_sem "$JOINT_SEM" --caption_cache data/tb_caption_llm2vec_pzstyle_v1
  --texts_json data/tb_motion_texts_pzstyle_v1.json --ktjd_percell_stats "$PERCELL"
  --ktjd_gamma_calib "$CALIB" --exclude_clips "$CUT"
  --dim "$DIM" --depth "$DEPTH" --heads "$HEADS" $([ "$QK_NORM" = 1 ] && echo --qk_norm) $([ "$GRAD_CKPT" = 1 ] && echo --grad_ckpt)
  --lr "$LR" --batch "$BATCH" --epochs "$EPOCHS" --warmup_steps "$WARMUP" --wd 0.01 --grad_clip 1.0
  --lr_scheduler half_cosine --lr_decay_epochs "$LR_DECAY_EPOCHS" --eta_min_ratio 0.01 --grad_spike_reject 200
  --v_space --sigma_min 0.2 --huber_delta 10 --bf16 --t_sampler uniform
  --gamma_fk 0.07 --fk_warmup_steps "$FK_WARMUP" --gamma_vel 0.01 --gamma_lock 0.01 --gamma_acc 1.0
  $([ "$DEMO_REST" = 1 ] && echo --demo_rest) --demo_frames "$DEMO_FRAMES" --struct_feats --dir_bias --anchor none --balance clip
  --p_drop_text 0.1 --p_drop_demo 0.0 --p_drop_both 0.0 --target_frames 240
  --init_from "$INIT" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_targets "$LORA_TARGETS"
  --ckpt_every "${CKPT_EVERY:-1000000}"
  ${AUTH_FLAG:---ktjd_training_authorized}
  ${EXTRA:-})
if [ "$PREFLIGHT" = 1 ]; then echo "[lora] PREFLIGHT OK -- would run: ${CMD[*]}"; exit 0; fi
mkdir -p "$OUT"
# Shared alloc (user 2026-09-04): GPU_PIN=<k> pins the run to ONE card of an alloc other work is using -- --gres=gpu:1
# alone does not avoid busy cards -- with an --overlap step over all GPUS_TOTAL cards, CUDA_VISIBLE_DEVICES=k, and a
# fail-closed gate: any compute process already on card k refuses the launch (no card sharing across projects).
GRES_ARGS=(--gres=gpu:1)
if [ -n "${GPU_PIN:-}" ]; then
  GPUS_TOTAL=${GPUS_TOTAL:-4}
  [[ "$GPU_PIN" =~ ^[0-9]+$ && "$GPUS_TOTAL" =~ ^[1-9][0-9]*$ ]] && [ "$GPU_PIN" -lt "$GPUS_TOTAL" ] \
    || { echo "[lora] refuse: GPU_PIN='$GPU_PIN' must be an integer in [0, GPUS_TOTAL=$GPUS_TOTAL)"; exit 1; }
  # one launcher per (alloc, card) at a time: the lock is held for the whole run, so a second cooperative launcher
  # cannot pass the idle probe between our probe and our training step (TOCTOU, codex 2026-09-04)
  mkdir -p .aris/meta; exec 8>".aris/meta/.gpu_pin_${SKIN_JOBID}_${GPU_PIN}.lock"
  flock -n 8 || { echo "[lora] refuse: GPU$GPU_PIN of alloc $SKIN_JOBID is being launched on by another launcher"; exit 1; }
  # the probe itself must succeed: a failed srun/nvidia-smi is NOT an idle card
  probe=$(srun --jobid="$SKIN_JOBID" --overlap --gres=gpu:${GPUS_TOTAL} -N1 -n1 nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU_PIN" 2>&1); st=$?
  [ "$st" -eq 0 ] || { echo "[lora] refuse: GPU probe failed (rc=$st): $probe"; exit 1; }
  busy=$(printf '%s\n' "$probe" | grep -c '[0-9]')
  [ "$busy" -eq 0 ] || { echo "[lora] refuse: GPU$GPU_PIN of alloc $SKIN_JOBID has $busy compute process(es)"; exit 1; }
  GRES_ARGS=(--overlap --gres=gpu:${GPUS_TOTAL} -N1)
  echo "[lora] pinned to GPU$GPU_PIN of alloc $SKIN_JOBID (idle at launch, lock held)"
fi
srun --jobid="$SKIN_JOBID" --ntasks=1 --cpus-per-task=${CPUS:-8} "${GRES_ARGS[@]}" --mem=${MEM:-96G} \
  /usr/bin/env ${GPU_PIN:+CUDA_VISIBLE_DEVICES=$GPU_PIN} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "${CMD[@]}" 2>&1 | tee "$OUT/train.log" | { grep -E '^\[(lora|init_from|train|resume|ktjd|calib)\]|=== epoch|\[val\]|SPIKE|FATAL|refuse|Traceback|Error|error' || true; } | cut -c1-200
rc=${PIPESTATUS[0]}
echo "[lora] rc=$rc (full log: $OUT/train.log)"
exit "$rc"
