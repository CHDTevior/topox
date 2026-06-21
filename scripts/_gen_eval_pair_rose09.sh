#!/bin/bash
# Throwaway orchestration launcher: run the (already codex-PASS) batched gen-eval for the
# 512-backbone (GPU0) + 2048-backbone (GPU1) IN PARALLEL on the rose09 A100x2 alloc, one
# per GPU via explicit CUDA_VISIBLE_DEVICES (srun --gres=gpu:1 --overlap collided on GPU0).
# ONE srun --gres=gpu:2 owns both GPUs; the two python procs pin CVD=0 / CVD=1.
set -uo pipefail
cd /scratch/ts1v23/workspace/noKslot_clean
PY=/scratch/ts1v23/.conda/bin/python
D=data/animo4d_anytop_clean_L4_safe_plus_truebones
EV=runs/anytop_t2m_evaluator_distilbert_coemb512_gb128_lr1e-4_mfd12_seed42/best_model.pt
ARGS="--eval_ckpt $EV --val_manifest $D/eval_splits/val_all.json --data_root $D --exclude_truebones --n_samples 1024 --gen_batch 16 --pool 32 --steps 50 --cfg_scale 4.0 --num_frames 300"
R512=runs/codeflow_graph_pscf_mergedL4TB_n512_b16_lr8e5_4xh200_seed42
R2048=runs/codeflow_graph_pscf_mergedL4TB_n2048_b8_lr8e5_8xa100_seed42
srun --jobid=1014946 --overlap --gres=gpu:2 --cpus-per-task=16 --no-kill bash -c "
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
  CUDA_VISIBLE_DEVICES=0 $PY scripts/_eval_codeflow_gen_in_evalspace.py --flow_ckpt $R512/best_model.pt $ARGS --out $R512/gen_evalspace_12ch_animo4d_n1024.json > $R512/gen_n1024.log 2>&1 &
  p0=\$!
  CUDA_VISIBLE_DEVICES=1 $PY scripts/_eval_codeflow_gen_in_evalspace.py --flow_ckpt $R2048/best_model_snap_for_eval.pt $ARGS --out $R2048/gen_evalspace_12ch_animo4d_n1024.json > $R2048/gen_n1024.log 2>&1 &
  p1=\$!
  wait \$p0; rc0=\$?
  wait \$p1; rc1=\$?
  echo \"[gen-pair] EXITED rc_512=\$rc0 rc_2048=\$rc1\"
"
