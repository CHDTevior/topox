#!/bin/bash
# **清理后语料 v2 上的 73M 臂**（user 2026-09-20：先"用我们的谱rope，时间rope，增广开的路线，训一下这个组合
# 清理后的新数据"，后"只跑73的吧，我们减少变量"）。
#
# 相对 v1 的臂2 只改两件事，其余逐字继承：
#   (1) 语料换成 v2（8,610 clip / 6,360 rig，v1 是 6,881 / 5,263；v1 全部内容是 v2 的真子集，已逐字节验过）
#   (2) 模型从 384/8/6 放大到 512/10/8（73.31M，对齐 UniMate 自己的 denoiser 74.10M），lr 2e-4 → 1.5e-4
#
# **lr 为什么必须降**：同一配方在 512/10/8 下用 2e-4 炸过（runs/v2_noik_uniml3d_bothrope_aug_d512）。
# 两条臂在训练的**同一位置**撞上同一个不稳定窗口 —— warmup(4000) 刚结束、lr 还在峰值 99%：
# 384/8/6 的梯度是 213–249（擦过 200 阈值，4 步后走出来，此后 74 个 epoch 零尖峰）；
# 512/10/8 的是 523–734（3 倍），越过后再没回来，累计拒绝数万步、val artic 从 1.97 崩到 7.77。
# 1.5e-4 是**未经验证**的修复，起训后必须盯 ep11–13 那个窗口。
#
# 继承链：本文件 → ..._bothrope_aug_d512_lr15 → ..._aug_d512 → ..._uniml3d_bothrope_aug → ..._heldout_bothrope_aug
_v2_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_v2_out=${OUT:-}; _v2_port=${RDZV_PORT:-}
_v2_ja=${JOB_A:-}; _v2_jb=${JOB_B:-}; _v2_host=${RDZV_HOST:-}
source "$_v2_here/pilot36m_uniml3d_bothrope_aug_d512_lr15_2node_env.sh"

# ---- 语料 v2 的全部 sidecar（每一个都经过空操作自检：v1 也有的部分逐字节/逐行一致）----
export KTJD_ROOT=dataset/ktjd17_uniml3d_v2
export PERCELL=data/uniml3d_norm_stats_v2.npz                      # 共有 rig 中恰好 36 个变化，且恰好是多了 clip 的那 36 个
export JOINT_SEM=data/joint_semantics_llm2vec_uniml3d_v2corpus.npz # 占位行 0，confident_frac 全 1.0
export CAPTION_CACHE=data/uniml3d_caption_llm2vec_v2               # pooled_vs_tokens_max_err 0.00e+00
export TEXTS_JSON=data/uniml3d_motion_texts_v2.json                # v1 的 6,881 行逐行一致，新增 1,729
export CUT=configs/uniml3d_v2_visual_exclusions.json               # 内容同 v1，只换 generation 绑定

# ---- 授权 generation 必须换成 v2 的，否则训练器的语料绑定检查会拒 ----
# 用替换而不是追加：追加虽然靠 argparse "后者胜出"也能工作，但 EXTRA 里留着一个失效的旧 id 是误导。
_v1gen=20260915T174929822046Z-6c4c46786a1e
# 无条件 exit，不用 return：本文件唯一的真实用法是被 source（runs/_heldout/_resume.sh、
# scripts/_launch_v2_ddp_2node_h200.sh、校准 runner），而那三个 caller 都是 `set -uo pipefail` 无 -e、
# 且**全部丢弃 `.` 的返回码** —— 写 return 的话它们会带着半套配置（v2 语料 + v1 的 EXTRA/OUT/端口/CALIB）
# 继续往下跑，最后靠训练器的 calib generation 不匹配才拒，白跑一轮启动且真因埋在日志两百行外。
# exit 会中止 caller，那正是想要的（审查 2026-09-20：R9「不会 fail 的检查 = 无效检查」/ R12 fail loud）。
export EXTRA="${EXTRA//$_v1gen/20260920T162314097778Z-b54d5f79e630}"
# 查**后置条件**（替换后 v2 的 id 在不在），不查前置条件：前者更强也更直接 —— 继承链若变得不含 v1 的 id，
# 替换就是空操作，v2 的 id 便不可能出现。无条件 exit 不用 return：本文件唯一的真实用法是被 source
# （runs/_heldout/_resume.sh、scripts/_launch_v2_ddp_2node_h200.sh、校准 runner），三个 caller 都是
# `set -uo pipefail` 无 -e 且**全部丢弃 `.` 的返回码**，写 return 会让它们带着半套配置
# （v2 语料 + v1 的 EXTRA/OUT/端口/CALIB）继续跑，最后靠训练器的 calib generation 不匹配才拒 ——
# 白跑一轮启动且真因埋在日志两百行外（审查 2026-09-20：R9「不会 fail 的检查 = 无效检查」/ R12 fail loud）。
case "$EXTRA" in
  *"--ktjd_auth_generation 20260920T162314097778Z-b54d5f79e630"*) ;;
  *) echo "[refuse] EXTRA 里没有 v2 的 auth generation（20260920T162314097778Z-b54d5f79e630）—— 继承链变了，替换成了空操作" >&2; exit 1;;
esac
case "$EXTRA" in
  *"$_v1gen"*) echo "[refuse] EXTRA 里仍残留 v1 的 auth generation（$_v1gen）" >&2; exit 1;;
esac

# lr 调度铺满全程（user 2026-09-20 选"cosine 铺满全程"）。
# 继承链从 120/40 带下来的"保持形状"在 240 epoch 上变成：warmup 0→4,000 步、half-cosine 4,000→34,960 步
# （= LR_DECAY_EPOCHS 80 × 437），此后 **69,920 步（67%，约 14.4 小时）平在地板 1.5e-6 = 峰值的 1%**。
# 那样"240 epoch ≈ 对齐 UniMate 的 120k 步"在效果上站不住 —— 有效训练只有 34,960 步。
# 改成 240 后 cosine 覆盖全部 104,880 步，warmup 占比 3.8% 不变。
# 代价：与 v1 的 120/40 系列不再是同一调度形状；但 user 已选只跑 73M，本来就没有严格可比的 v1 臂。
export LR_DECAY_EPOCHS=240

export OUT=${_v2_out:-runs/v2_noik_uniml3dv2_bothrope_aug_d512_lr15}
export RDZV_PORT=${_v2_port:-29551}   # 29546 臂1 / 29547 臂2 / 29548 臂3 / 29549 d512 / 29550 d512lr15
export JOB_A=${_v2_ja:-$JOB_A} JOB_B=${_v2_jb:-$JOB_B} RDZV_HOST=${_v2_host:-$RDZV_HOST}
export MASTER_NODE=pink7002 WORKER_NODE=pink7003
export CALIB=${UNIML3DV2_D512_CALIB:-configs/pilot_uniml3dv2_bothrope_aug_d512_gamma_calibration_b16_v1.json}
