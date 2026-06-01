# Codex 审: rot6d-FK recovery double-root-rotation 修复 (2026-06-01)

## 任务
审两处**已完成**的代码修复:删除 rot6d-FK 恢复路径里多余的 root correction 行
(double-applies root global rotation)。判定修复正确性 + 有无孤儿/边界问题。

## 背景与诊断 (决定性)
rot6d-FK 恢复(从 ch3:9 6D rotation token + rest offsets + parent FK 链恢复 world
joint pos)有 double root rotation bug。用户看 GT_FK vs GT_RIC 渲染发现 FK 骨架比
RIC 多转近一倍全局旋转(turn yaw)。

RIC 路(从 ch0:3 root-invariant position 恢复, `_recover_world_positions`)= 可靠
ground truth:它是 pos loss 的 target、训练实际在用、逻辑无 FK 递归(只一次
-r_rot_quat 旋转 + 加 root xz)。**判据: FK(data) 必须 == RIC(data)**。

- clean_L2 Saiga max-mismatch clip: FK(buggy) vs RIC absL1=0.6522,全局朝向 sweep
  ratio=1.98 (≈2.0 = double)。
- 1070 条老 AnyTop truebones 数据集(论文验证过)6 个最大旋转 clip(Parrot/Bird
  CircleFly 720/714°, Trex turn_180 396°, Lynx/Raptor2): FK(buggy) absL1=0.46~1.06。
- **删 root correction 后:全部 absL1=0.0000(逐点完美 == RIC)**。
- 用户独立验证:删后 rot6d-FK 与 ch0:3 pose 路误差 1e-9。

channel 语义(用户确认,与老 AnyTop 一致): 0:3 root-invariant local pos / 3:9
AnyTop 6D rotation token / 9:12 local vel / 12 foot contact。3:9 的 root/child
token **已含全局朝向变化**;reindex 后 rot_q[:,0] 已是正确 root 朝向,再乘
-r_rot_quat 就把 turn yaw 用第二遍。

## 修复 (本次改动,2 个文件)
1. **src/data/anytop_rot6d_fk.py** (numpy, `recover_from_bvh_rot_np`):
   删 `rot_q[:, 0] = _quat_mul(_quat_neg(r_rot_quat), rot_q[:, 0])`,替为注释。
2. **src/models/graph_salad/rot6d_fk_recovery.py** (torch,
   `recover_rot6d_fk_positions_torch`, 训练 fk loss 用的就是它):
   删 `local_R_list[0] = torch.matmul(root_R.transpose(-1,-2), local_R_list[0])`,
   替为注释。

## smoke 验证 (scripts/_smoke_fk_fix_torch.py, RESULT=PASS)
- numpy FK(fixed) vs numpy RIC  absL1=2e-7
- torch FK(fixed) vs torch RIC  absL1=3e-7
- torch FK(fixed) vs numpy RIC  absL1=3e-7
- torch FK(fixed) vs numpy FK   absL1=2e-7 (impl parity, 此前 SMOKE1 带 bug 时是 1.19e-6)
- autograd backward through torch FK: grad_finite=True, grad_norm=20876 (仍可微)

## 影响范围
- world_geometry loss(`world_recovery.py` RIC torch path)无 bug,**不受影响**。
- B 训练(`_launch_rot6d_fk_B.sh`, loss_mode=anytop13_world_rot6d_fk, **w_fk=0.25**)
  的 fk loss 用 buggy torch FK → 已训 25 ep 的 fk target 一直被 double 污染。修复后
  fk target 变正确。
- `gt_fk_mismatch` diagnostic(训练实测 0.29)是 bug 产物,修复后两路一致应≈0。

## 审查点 (请逐条判定)
1. **修复正确性**:删 root correction 是否正确?reindex(`rot_q[:,p]=all_q[:,j]` /
   `local_R[p]=all_R[:,j]`)后 root 槽是否真的已含正确 root 朝向、不需再乘?
   1070 数据集 absL1=0.0000 + smoke 2e-7 是否足以判定?
2. **孤儿/死代码**:删 correction 后 — numpy 的 `r_rot_mat`/`all_mat` index-0 槽、
   torch 的 `root_R`(仍用于 `all_R` concat 占位 + `_recover_root_R_and_pos` 的 root
   pos)是否仍必要?`_quat_neg`/`_quat_mul` 是否仍被 `_quat_mul_vec` 使用(非孤儿)?
   有无我改动产生的、应删未删的孤儿?(Karpathy R3: 只删自己改动产生的孤儿)
3. **autograd**:torch 删 correction 后 list 累积 FK 链不破图?grad_finite 已验。
4. **边界/既有问题**:多 root-child 时 reindex `local_R[p]=all_R[j]` 会覆盖(只留
   最后一个 child)— 这是既有逻辑(非本次引入)。删 correction 后是否暴露新问题?root
   槽(parents[j]==0 的 j)的 reindex 来源是否唯一?
5. **B 训练建议**:修复后 fk target 正确(≈ world term)。B 已训 25 ep 带污染。
   w_fk=0.25 vs w_rot=1.0(base rot loss 直接监督 ch3:9,无 bug)。建议 B 重启(干净)
   还是续训(省 13.5h,但 pred 需 unlearn 已学的补偿)?

## 相关文件 (可读)
- src/data/anytop_rot6d_fk.py (numpy, 已改, 完整审)
- src/models/graph_salad/rot6d_fk_recovery.py (torch, 已改, 完整审)
- src/data/anytop_dataset.py:282 `_recover_world_positions` (RIC ground truth, 对照)
- src/models/graph_salad/world_recovery.py (RIC torch, 对照)
- src/models/graph_salad/losses.py:684 `compute_world_rot6d_fk_terms` (fk loss 调用点)
- scripts/_smoke_fk_fix_torch.py (smoke)
- scripts/_diag_fk_variants.py / _diag_oldset_fk_variants.py (variant 诊断)
- /scratch/ts1v23/_pz_pipeline/data_loaders/truebones/truebones_utils/motion_process.py:750
  (codex 写的官方版,自带同 bug,本次不改它,仅说明同源)
