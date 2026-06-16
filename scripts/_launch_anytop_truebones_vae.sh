#!/bin/bash
# Single-alloc bf16 VAE training on AnyTop TRUEBONES (1070 clips / 70 species),
# full-data all/all split (train=all 1070, val=all 1070 via full_data_val_species).
# Replicates the diffusion VAE (runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card
# _xnode run-4: batch48/lr8e-4/global384@8card, loss=anytop13_world_rot6d_fk
# w_world0.25/w_fk1.0/w_traj0.10, graphormer/coarse_xattn/edge_segment, anytop13,
# d512 h8 dff1536, max_coarse128, use_name_embed, bf16) — Goyal-scaled to the idle
# card count via LR. NNODES=1 standalone only (no cross-alloc). The ONLY diffs vs
# run-4: anytop_root=truebones, epochs=200, full_data_val_species set (all/all).
set -u
cd /scratch/ts1v23/workspace/noKslot_clean
CVD="${CVD:?set CVD (e.g. 0,1,2,3)}"
BS="${BS:-48}"                                   # per-GPU batch (= run-4)
LR="${LR:?set LR (Goyal: 8e-4 * global/384)}"
EPOCHS="${EPOCHS:-200}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
W_WORLD="${W_WORLD:-0.25}"; W_FK="${W_FK:-1.00}"; W_TRAJ="${W_TRAJ:-0.10}"
ANYTOP_ROOT="${ANYTOP_ROOT:-data/anytop_truebones}"
FULL_DATA_VAL_SPECIES="${FULL_DATA_VAL_SPECIES:-}"   # set -> full-data all/all; empty -> val_frac holdout split
VAL_FRAC="${VAL_FRAC:-0.05}"                          # holdout fraction when full_data empty
SMOKE="${SMOKE:-}"                                    # non-empty -> add --smoke (preflight, no full run)
OUT="${OUT:?set OUT}"
PY=/scratch/ts1v23/.conda/bin/python3
NPROC=$(echo "$CVD" | tr ',' '\n' | grep -c .)
GLOBAL=$(( NPROC * BS ))

# Guard: never double-launch THIS run (matches the OUT basename in the cmdline).
if pgrep -f "train_graph_vae.py.*$(basename "$OUT")" >/dev/null 2>&1; then
    echo "[truebones-vae] ABORT: $OUT already training"; exit 0
fi
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="$CVD"

echo "[truebones-vae] $(date '+%F %T %Z') host=$(hostname) CVD=$CVD nproc=$NPROC bs=$BS lr=$LR global=$GLOBAL epochs=$EPOCHS"
echo "[truebones-vae] root=$ANYTOP_ROOT out=$OUT amp=$AMP_DTYPE w_world=$W_WORLD w_fk=$W_FK w_traj=$W_TRAJ"
if [ -n "$FULL_DATA_VAL_SPECIES" ]; then echo "[anytop-vae] split=full-data all/all (full_data_val_species set)"; else echo "[anytop-vae] split=val_frac=$VAL_FRAC holdout (no full_data)"; fi
[ -n "$SMOKE" ] && echo "[anytop-vae] SMOKE preflight (--smoke)"

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" scripts/train_graph_vae.py \
  --dataset anytop_truebones --feat_mode anytop13 --attn_mode graphormer \
  --decoder_mode coarse_xattn --pool_type edge_segment \
  --loss_mode anytop13_world_rot6d_fk --w_world "$W_WORLD" --w_fk "$W_FK" --w_traj "$W_TRAJ" \
  --anytop_root "$ANYTOP_ROOT" \
  ${FULL_DATA_VAL_SPECIES:+--full_data_val_species "$FULL_DATA_VAL_SPECIES"} \
  --val_frac "$VAL_FRAC" --batch_size "$BS" --lr "$LR" --seed 42 \
  ${SMOKE:+--smoke} \
  --epochs "$EPOCHS" --save_every 5 --periodic_save_every 50 \
  --d_model 512 --n_heads 8 --d_ff 1536 \
  --n_graph_layers 4 --n_enc_temporal_layers 2 --n_cross_layers 3 \
  --n_dec_temporal_layers 2 --n_treeik_layers 3 \
  --max_coarse 128 --local_radius 8 --temporal_stride 4 \
  --max_frames 64 --max_joints 144 --use_name_embed \
  --amp_dtype "$AMP_DTYPE" \
  --out "$OUT" --overwrite
rc=$?
echo "[truebones-vae] $(date '+%F %T %Z') torchrun EXITED rc=$rc"
exit "$rc"
