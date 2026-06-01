你是独立代码审计者(gpt-5.5 xhigh),全新 clean context。审 VAE 新 loss_mode=anytop13_world_rot6d_fk 的实现,给 [PASS | NEEDS-FIX]。

## 背景(已验证,聚焦代码不复述)
按 handoff/20260601_rot6d_fk_loss_plan.md 实现组合几何监督: L_total = L_anytop13_base + w_world*L_world_ric + w_fk*L_rot6d_fk + w_traj*L_root_traj。保留现有 anytop13_world_geometry(world-only ablation)不删。FK 路线用官方 recover_from_bvh_rot_np(matrix port)对非 root 旋转有梯度,这是与 world_geometry 的核心区别。

已验证(确定性):
- SMOKE1: torch FK(rot6d_fk_recovery) vs numpy 官方(anytop_rot6d_fk.py) max_diff=1.19e-6 (<1e-4)。
- WIRING SMOKE: G2 默认 anytop13 total==direct(1.996637,无 geo keys); G3 world/fk/traj/gt_fk_mismatch/total 全 finite; G4 非 root rot6d(3:9) grad=6.02(>0,FK签名); G5 非 root pos(0:3) grad=3.00(>0,world路线); G7 gt_fk_mismatch=0.158 不 assert 零。
- Preflight: clean_L2 上 RIC(gt) vs FK(gt) mismatch median 1.2% 但 p95 30%(哺乳动物 outlier),长链物种小。故 fk loss 有数据固有地板,calibration 后用低 w_fk 起步。

## 改动(审这些)
1. **新模块 src/models/graph_salad/rot6d_fk_recovery.py**: recover_rot6d_fk_positions_torch([B,T,J,13]raw, parent_indices, rest_offsets[B,J,3], joint_mask[B,J])->[B,T,J,3]。matrix-only torch port 官方 recover_from_bvh_rot_np:root R+pos 复用 RIC root 路径(world_recovery._rot6d_to_matrix_torch + vel cumsum + height); 非 root R from ch3:9; parent reindex local_R[p]=all_R[j](矩阵直接做,不走 numpy 的 quat round-trip); root 修正 local_R[0]=root_R^T@local_R[0](=numpy 的 -r_rot_quat*rot_q[0]); 4x4 FK 链(list 累积避免 autograd inplace)。
2. **losses.py 纯追加(+164 -0, compute_total_loss_13ch 0 删除)**: compute_world_rot6d_fk_terms + _masked_l1_xyz helper。返回 world/fk/traj/gt_fk_mismatch。target=RIC(gt)(plan §3 Option 2)。
3. **__init__.py**: export compute_world_rot6d_fk_terms。
4. **train_graph_vae.py(+81 -4)**: run_loss 加 w_fk 参数 + anytop13_world_rot6d_fk 分支(显式 total += w_world*world+w_fk*fk+w_traj*traj, gt_fk_mismatch 不进 total); CLI --loss_mode choices 加 + --w_fk default 0.25; 两调用点(:772/:883)传 w_fk。

## 请审(聚焦)
1. **FK 正确性**: rot6d_fk_recovery.py 的 matrix-only 版是否与官方 recover_from_bvh_rot_np 逻辑等价? 重点: parent reindex(local_R[p]=all_R[j] 而非 [j]=[p])、root 修正 root_R^T@local_R[0]、root pos/R 复用 RIC 路径是否一致? smoke 已证 vs numpy 1.19e-6,但请核对逻辑无隐藏 off-by-one/排列错。autograd: list 累积 + out.clone() 是否真无 inplace 破图(smoke backward 成功)?
2. **target 语义**: world 和 fk 都 target RIC(gt)(不是 FK(gt))。这符合 plan §3 Option 2。pred_raw/gt_raw 用 _denorm_13ch(同 world_geometry)。对吗?
3. **默认零回归**: loss_mode=anytop13 不进任何 geo 分支, total bit 不变(G2 已证)。run_loss 签名加 w_fk=0.0 默认。有遗漏吗?
4. **mask**: 用 effective_frame_mask(=frame_mask_recovered)惩罚, 不用 raw batch.frame_mask(plan 硬约束5)。_masked_l1_xyz 用 _broadcast_pos_vel_mask(joint_mask, frame_mask)。对吗?
5. **gt_fk_mismatch 只诊断**: 不进 total, 只 losses[] 记录。确认。
6. **val 侧安全**: run_loss 在 val 也调(:883), 返回 world/fk/traj/gt_fk_mismatch 进 val_losses。下游 val_recon 白名单(pos/rot/vel/contact)+ loss_weights.get → 不会 KeyError? 各 torch.save 安全?
7. **DDP/dtype**: FK per-sample loop(B 小)在 DDP 下安全(无新可学参数)? 返回 float32 [B,J,T,3]? 数据约束: 不改 pool/decoder/dataset/renderer。

## 文件
- src/models/graph_salad/rot6d_fk_recovery.py(新,完整审)
- src/models/graph_salad/losses.py(compute_world_rot6d_fk_terms + _masked_l1_xyz, 末尾)
- scripts/train_graph_vae.py(run_loss :56-130, CLI :372-388, 调用点 :772/:883)
- 对照官方: src/data/anytop_rot6d_fk.py(numpy 版,已验证) / /scratch/ts1v23/_pz_pipeline/data_loaders/truebones/truebones_utils/motion_process.py:750(如可读)

## 输出
明确 [PASS | NEEDS-FIX]。NEEDS-FIX 给行号+具体修法。聚焦 FK 正确性 + 默认零回归 + autograd 安全。不复述背景。
