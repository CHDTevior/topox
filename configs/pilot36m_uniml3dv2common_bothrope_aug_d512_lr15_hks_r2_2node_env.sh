#!/bin/bash
# R2 臂（user 2026-09-25：在常驻默认 H1 之上做"静止姿态进输入 + 静止构型增广"，保留 S1 的抽样与 H1 的热核签名）。
# 与 H1 配置（configs/pilot36m_uniml3dv2common_bothrope_aug_d512_lr15_hks_2node_env.sh）逐字相同，只多两件事：
#   1. STRUCT_WORLD_REST=1 —— 结构特征 8 → 14 维：每根骨头静止时的世界系方向（P_rest_j − P_rest_parent 的单位向量，与 R_rest[parent]·offset 等价；3 维）+ 静止世界位置
#      （减根 XZ、按平均骨长归一后径向 log 压缩，含离地高度；3 维），来自骨架文件；增广样本按变换后的树重算（src/data/incontext_pairs.py _world_rest_feats）。
#   2. 静止构型增广通道：每个样本以概率 AUG_REST_P=0.5（与 UniMate 式 one_of 操作独立抽样）把每个关节的 rest 旋转乘一个随机旋转
#      （角度 U(0, 30°)，随机轴），世界系动作不变、服务的 rest 相对旋转随之重算；该样本的归一化均值 = 变换后的静止姿态、
#      静止示范 = 该骨架自己的静止帧（全零）——即一副按新约定制作的骨架会给出的输入（src/data/ktjd17_augment.py REST_RULE）。
# 校准：dit_motion.py（struct_rest_in 层）与增广协议（rest_p/rest_deg）都进校准绑定 → 独立工件（scripts/_calib_uniml3dv2common_bothrope_aug_d512_hks_r2_b16_v1.sh）。
_r2_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$_r2_here/pilot36m_uniml3dv2common_bothrope_aug_d512_lr15_hks_2node_env.sh"
export OUT=runs/v2_noik_uniml3dv2common_bothrope_aug_d512_lr15_hks_alpha05_r2
export RDZV_PORT=29555   # 29552 对照组 / 29553 S1 / 29554 H1
export STRUCT_WORLD_REST=1
export AUG_REST_DEG=30 AUG_REST_P=0.5
# 父链把 --aug_rest_deg 0 写进了 EXTRA：原地换成 30（不重复出现），再追加独立概率；两者与上面的变量一致（校准 runner 读变量，训练器读 EXTRA，
# 训练器把自己的增广协议与工件比对——不一致会拒绝启动）。
export EXTRA="${EXTRA/--aug_rest_deg 0 /--aug_rest_deg $AUG_REST_DEG } --aug_rest_p $AUG_REST_P"
export CALIB=configs/pilot_uniml3dv2common_bothrope_aug_d512_hks_r2_gamma_calibration_b16_v1.json
