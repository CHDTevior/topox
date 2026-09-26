#!/bin/bash
# **UniMate 受控对照：两边训练同一批片段 id**（user 2026-09-22："让两边训同样的数据量，对应的 ID 也相同 …
# 每边用自己的格式，我们这边就还用我们这边的格式"；配方选"73M 主线"，训练长度选"各用各的默认"）。
#
# 相对 v2 语料上的 73M 臂（configs/pilot36m_uniml3dv2_bothrope_aug_d512_lr15_2node_env.sh）只改训练集：
#   CUT = configs/uniml3d_v2_common_unimate_exclusions.json —— 排除 1,604 条不在两边共同集里的片段
#   （1,331 条骨架关节数超出 UniMate 的 [5, 60]、150 条被它的预处理筛掉、122 条骨架不在它的特征集、1 条原有的肉眼排除），
#   剩 6,747 条训练 + 259 条留出（其中 255 条的骨架两边都有训练片段，对照以这 255 条为准；生成用 last_model.pt），与 UniMate 侧 outside_docs/UniMate/dataset/features/objaverse_common_v2 同一批 id
#   （scripts/_build_common_unimate_subset.py 生成，两边的描述与关节数逐条比对一致）。
# 训练集变了 → gamma 校准必须在这批数据上重测（scripts/_calib_uniml3dv2common_bothrope_aug_d512_b16_v1.sh），
# 产物、输出目录、端口都独立。其余（模型 512/10/8、lr 1.5e-4、240 epoch、cosine 铺满、增广）逐字继承；节点见文末（续期换了节点）。
_c_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_c_out=${OUT:-}; _c_port=${RDZV_PORT:-}
source "$_c_here/pilot36m_uniml3dv2_bothrope_aug_d512_lr15_2node_env.sh"
export CUT=configs/uniml3d_v2_common_unimate_exclusions.json
export OUT=${_c_out:-runs/v2_noik_uniml3dv2common_bothrope_aug_d512_lr15}
export RDZV_PORT=${_c_port:-29552}   # 29551 v2 73M 臂
export CALIB=configs/pilot_uniml3dv2common_bothrope_aug_d512_gamma_calibration_b16_v1.json
# 2026-09-22 18:36Z：pink7002/7003 的 alloc 到期，续期（i7_h200 1586864/1586865）落在 pink7006/pink7008。
# 两台都有 ib1 与 mlx5_3 ACTIVE InfiniBand（与继承链里的 NCCL_SOCKET_IFNAME=ib1 NCCL_IB_HCA=mlx5_3 一致），
# 各自只有这一个 alloc、4 张卡全空。只改本臂：基础配置（全语料 73M 臂）里的 pink7002/7003 保持原样。
export MASTER_NODE=pink7006 WORKER_NODE=pink7008
