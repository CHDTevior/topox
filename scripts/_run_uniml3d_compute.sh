#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=""
PY=/scratch/ts1v23/.conda/bin/python
export PYTHONPATH=.
"$PY" tests/test_uniml3d_conversion.py
"$PY" scripts/_build_uniml3d_ktjd17.py --output dataset/ktjd17_uniml3d_v1 --workers 4 --wait-download-seconds 7200
"$PY" scripts/_analyze_uniml3d_ktjd17.py --root dataset/ktjd17_uniml3d_v1 --render-rigs 30
"$PY" scripts/_verify_uniml3d_channels.py dataset/ktjd17_uniml3d_v1 --samples 400
KTJD_JOINT_SPEC=dataset/ktjd17_uniml3d_v1/analysis/joint_spec.json \
    "$PY" scripts/_verify_noik_corpus.py dataset/ktjd17_uniml3d_v1 400
"$PY" scripts/_verify_noik_corpus.py dataset/ktjd17_pzh312_noik_v2 400
"$PY" scripts/_render_uniml3d_channels.py dataset/ktjd17_uniml3d_v1 --count 14
echo UNIML3D_COMPUTE_COMPLETE
