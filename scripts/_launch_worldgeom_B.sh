#!/bin/bash
# scripts/_launch_worldgeom_B.sh
# A/B experiment arm B: anytop13_world_geometry loss on the edge_segment baseline.
#
#   A = edge_segment + coarse_xattn + ORIGINAL anytop13 loss
#       (= existing baseline runs/_baseline_cleanL2_ep34_for_p1diag_compare,
#        val_recon=1.3784 @ ep34 — NO retrain needed, reuse it)
#   B = SAME architecture + loss_mode=anytop13_world_geometry (this script)
#
# ONLY experimental variable = the loss. Config replicates the baseline EXACTLY
# (verified from its ckpt args 2026-05-31): edge_segment / coarse_xattn /
# max_coarse=128 / d512 h8 dff1536 / val_frac0.05 lr4e-4 seed42 epochs300 /
# stride4 frames64 joints144 / use_name_embed. We ADD only:
#   --loss_mode anytop13_world_geometry --w_world 0.5 --w_traj 0.25
#
# GLOBAL BATCH MATCH (user 2026-05-31): baseline A = 2×H200 × bs32 = global 64.
# B runs 4×A100 × bs16 = global 64 → IDENTICAL global batch, so lr stays 4e-4
# (no Goyal scaling needed). This keeps the loss as the SOLE experimental
# variable (global batch / lr / optim dynamics all matched to A).
#
# world_geometry = supervise AnyTop RIFKE-recovered world joint positions (the
# space the visual QA renders). NOT FK supervision (zero grad to non-root rot).
# Step3 wiring smoke-PASS + codex-reviewed 2026-05-31.
#
# Runs on swarma1001 alloc 925439 (4×A100, free — A-diagnostic stopped).
# QA NOTE: for this experiment compare best_model.pt (best-by-total, INCLUDES
# world/traj) — NOT best_recon_model.pt (recon-only, excludes the new terms).
#
# Usage (smoke, 1 GPU, 5 iters, verifies wiring in the REAL train loop):
#   SMOKE=1 bash scripts/_launch_worldgeom_B.sh
# Usage (real, 4×A100):
#   CVD=0,1,2,3 setsid nohup bash scripts/_launch_worldgeom_B.sh > LOG 2>&1 </dev/null &
set -u
P=/scratch/ts1v23/workspace/noKslot_clean
cd "$P" || exit 1

CVD="${CVD:-0,1,2,3}"
SMOKE="${SMOKE:-0}"
W_WORLD="${W_WORLD:-0.5}"
W_TRAJ="${W_TRAJ:-0.25}"
OUT="${OUT:-runs/m1_l2_anytop13_edgeseg_C128_worldgeom_w05t025_seed42}"

SMOKE_FLAG=""
if [ "$SMOKE" = 1 ]; then
    CVD="${CVD%%,*}"        # first GPU only for smoke
    SMOKE_FLAG="--smoke"
    OUT="${OUT}_smoke"
fi
NPROC=$(echo "$CVD" | tr ',' '\n' | grep -c .)

# Guard: never double-launch the real run
if [ "$SMOKE" != 1 ] && pgrep -f "train_graph_vae.py.*worldgeom_w05t025" >/dev/null 2>&1; then
    echo "[wgB] ABORT: this world_geometry run already training"; exit 0
fi
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="$CVD"

echo "[wgB] $(date '+%F %T %Z') host=$(hostname) CVD=$CVD nproc=$NPROC smoke=$SMOKE"
echo "[wgB] loss=anytop13_world_geometry w_world=$W_WORLD w_traj=$W_TRAJ out=$OUT"

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" scripts/train_graph_vae.py \
  --dataset anytop_truebones --feat_mode anytop13 --attn_mode graphormer \
  --decoder_mode coarse_xattn --pool_type edge_segment \
  --loss_mode anytop13_world_geometry --w_world "$W_WORLD" --w_traj "$W_TRAJ" \
  --anytop_root "$P/data/anytop_planet_zoo_clean_L2" \
  --val_frac 0.05 --batch_size 16 --lr 4.000e-04 --seed 42 \
  --epochs 300 --save_every 5 --periodic_save_every 50 \
  --d_model 512 --n_heads 8 --d_ff 1536 \
  --n_graph_layers 4 --n_enc_temporal_layers 2 --n_cross_layers 3 \
  --n_dec_temporal_layers 2 --n_treeik_layers 3 \
  --max_coarse 128 --local_radius 8 --temporal_stride 4 \
  --max_frames 64 --max_joints 144 --use_name_embed \
  --out "$OUT" --overwrite $SMOKE_FLAG
echo "[wgB] $(date '+%F %T %Z') torchrun EXITED rc=$?"
