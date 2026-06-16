#!/usr/bin/env bash
# graph_pscf 泛化实验 QA (user 2026-06-11)。SPARE GPU 节点 only (绝不碰训练卡)。
# Exp1: OOD 文本(跨物种动作迁移) × 5 个 L5 已有骨架 — GT 栏 dropped(无对应 GT)。
# Exp2: truebones 5 个 unseen 拓扑骨架 × 各自 caption(已知文本类型) — 带 GT 栏对照。
set -euo pipefail
# T5 离线加载（计算节点不出网；必须在 python 进程启动前设，否则 huggingface_hub
# 在 import 时已把 HF_HUB_OFFLINE 读成模块常量，函数内再设太晚 — 踩过 2026-06-11）。
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ROOT=/scratch/ts1v23/workspace/noKslot_clean
PY=/scratch/ts1v23/.conda/bin/python
FLOW=runs/codeflow_graph_pscf_L5_b16_lr1p2e4_seed42/last_model.pt
FROZEN=runs/vqvae_L5_C50_d512_Q4_n512_b96_300ep_seed42/best_model.pt
OUT=runs/codeflow_graph_pscf_L5_b16_lr1p2e4_seed42
cd "$ROOT"

echo "######## Exp1: OOD 文本(跨物种动作迁移) × L5 骨架 ########"
# 每对 = (该物种没见过、属于别物种典型动作的 prompt)。动词模型在别物种上见过。
declare -a SP=(
  "PZ_Galapagos_Giant_Tortoise_Male|The giant tortoise gallops rapidly and leaps into the air"
  "PZ_Reticulated_Giraffe_Male|The reticulated giraffe climbs up and then climbs down"
  "PZ_Cheetah_Male|The cheetah swims through deep water"
  "PZ_Plains_Zebra_Male|The plains zebra jumps repeatedly and spins around"
  "PZ_Mandrill_Male|The mandrill gallops forward like a horse"
)
for pair in "${SP[@]}"; do
  sp="${pair%%|*}"; txt="${pair#*|}"
  echo "=== [exp1] $sp <- OOD: $txt ==="
  $PY scripts/animate_graph_codeflow.py \
    --flow_ckpt "$FLOW" --frozen_vqvae_ckpt "$FROZEN" \
    --out "$OUT/qa_gen_exp1_oodtext" --split val --species "$sp" \
    --n_per 1 --num_frames 300 --ood_text "$txt"
done

echo "######## Exp2: truebones unseen 骨架 × 各自 caption ########"
# truebones: 与 L5 零重叠、J<=64。Trex(双足)/Scorpion(节肢)/Anaconda(蛇形)/Bird(鸟翼)/BrownBear(熊)
$PY scripts/animate_graph_codeflow.py \
  --flow_ckpt "$FLOW" --frozen_vqvae_ckpt "$FROZEN" \
  --out "$OUT/qa_gen_exp2_newskel" --split train \
  --anytop_root data/anytop_truebones \
  --caption_emb_cache data/anytop_caption_t5_truebones_multi.npz \
  --caption_token_cache data/anytop_caption_t5_truebones_multi \
  --species "Trex,Scorpion-2,Anaconda,Bird,BrownBear" \
  --n_per 1 --num_frames 300

echo "DONE: $OUT/qa_gen_exp1_oodtext + qa_gen_exp2_newskel"
