#!/bin/bash
# scripts/_launch_worldgeom_resume.sh
# RESUME the interrupted world_geometry arm (SIGKILL @ ep21 on swarma1001) via the
# new --resume (model+optimizer+epoch). Continues the SAME experiment from
# last_model.pt (ep19) — exact comparability holds because training uses a fixed lr
# with NO scheduler, so restoring model+AdamW+start_epoch == uninterrupted run
# (codex 2026-06-01, threads 019e818d decision + 019e8198 code PASS).
#
# Writes to a FRESH OUT (_resumed) so the ORIGINAL run dir stays intact (its ep0-19
# ckpts + log are preserved, not overwritten).
#
# Config = identical to the original world_geometry run (= baseline A except loss):
#   loss_mode=anytop13_world_geometry w_world0.5 w_traj0.25, edge_segment/coarse_xattn/
#   graphormer/max_coarse128/d512 h8 dff1536/val_frac0.05/lr4e-4/seed42/epochs300/
#   stride4/frames64/joints144/use_name_embed, 4×A100×bs16 = global64.
#
# RESOURCE: swarma1001 alloc 925439, MY 4×A100 (the original arm's OWN alloc, idle
# since it was SIGKILLed). NOT grabbing other projects' cards. Do NOT touch
# swarmh1002(rot6d_fk B) / blossom04(diffusion).
#
# Smoke (verify resume load + DDP + no-OOM BEFORE the real run):
#   SMOKE=1 bash scripts/_launch_worldgeom_resume.sh
#     -> 4×A100 DDP, --smoke 5 iters from ep20. CRITICAL CHECK: loss must continue
#        LOW (~0.5, i.e. a trained model), NOT ~11.7 (fresh-init = resume failed).
# Real (durable):
#   setsid nohup bash scripts/_launch_worldgeom_resume.sh > LOG 2>&1 </dev/null &
set -u
P=/scratch/ts1v23/workspace/noKslot_clean
cd "$P" || exit 1

CVD="${CVD:-0,1,2,3}"
SMOKE="${SMOKE:-0}"
SRC="${SRC:-runs/m1_l2_anytop13_edgeseg_C128_worldgeom_w05t025_seed42}"   # original (interrupted) run dir
RESUME="${RESUME:-$SRC/last_model.pt}"
OUT="${OUT:-runs/m1_l2_anytop13_edgeseg_C128_worldgeom_w05t025_seed42_resumed}"

SMOKE_FLAG=""
if [ "$SMOKE" = 1 ]; then
    SMOKE_FLAG="--smoke"
    OUT="${OUT}_smoke"
    # keep the FULL 4-GPU DDP for the resume smoke (verify DDP + optimizer-device
    # restore, not a 1-GPU toy run).
fi
NPROC=$(echo "$CVD" | tr ',' '\n' | grep -c .)

if [ ! -f "$RESUME" ]; then echo "[wgR] ABORT: resume ckpt not found: $RESUME"; exit 1; fi
# Guard: never double-launch the real resumed run
if [ "$SMOKE" != 1 ] && pgrep -f "train_graph_vae.py.*worldgeom_w05t025_seed42_resumed" >/dev/null 2>&1; then
    echo "[wgR] ABORT: resumed world_geometry already training"; exit 0
fi
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="$CVD"

echo "[wgR] $(date '+%F %T %Z') host=$(hostname) CVD=$CVD nproc=$NPROC smoke=$SMOKE"
echo "[wgR] RESUME=$RESUME -> OUT=$OUT"

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" scripts/train_graph_vae.py \
  --dataset anytop_truebones --feat_mode anytop13 --attn_mode graphormer \
  --decoder_mode coarse_xattn --pool_type edge_segment \
  --loss_mode anytop13_world_geometry --w_world 0.5 --w_traj 0.25 \
  --anytop_root "$P/data/anytop_planet_zoo_clean_L2" \
  --val_frac 0.05 --batch_size 16 --lr 4.000e-04 --seed 42 \
  --epochs 300 --save_every 5 --periodic_save_every 50 \
  --d_model 512 --n_heads 8 --d_ff 1536 \
  --n_graph_layers 4 --n_enc_temporal_layers 2 --n_cross_layers 3 \
  --n_dec_temporal_layers 2 --n_treeik_layers 3 \
  --max_coarse 128 --local_radius 8 --temporal_stride 4 \
  --max_frames 64 --max_joints 144 --use_name_embed \
  --resume "$RESUME" \
  --out "$OUT" --overwrite $SMOKE_FLAG
rc=$?
echo "[wgR] $(date '+%F %T %Z') torchrun EXITED rc=$rc"
exit "$rc"
