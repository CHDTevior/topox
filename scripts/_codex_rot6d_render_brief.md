你是独立代码审计者(gpt-5.5 xhigh),全新 clean context。审两处改动,给 [PASS | NEEDS-FIX]。

## 背景
用户决定:VAE/diffusion 的可视化 QA 渲染,GT 和 pred **统一改用 rot6d FK 路径**(官方 recover_from_bvh_rot_np:从 channel 3:9 旋转 + 骨骼 offsets + 父链 FK 恢复世界坐标,骨长严格守恒),取代之前的 pos/RIC 路径(channel 0:3 position)。GT 和 pred 用**同一种**恢复 → apples-to-apples 比较。

已验证(确定性):
- 官方 recover_from_bvh_rot_np 与 recover_from_bvh_ric_np 在原始数据上误差 <1% bbox(root_diff=0.0000),即两路径等价、FK 正确。
- 端到端测试 EXIT=0:用冻结 VAE 重建水巨蜥,rot6d 模式渲染成功,speed_ratio=1.252(合理),GT/pred 骨架完整连贯。

## 改动 1:新模块 src/data/anytop_rot6d_fk.py
自包含 numpy 复刻官方 `recover_from_bvh_rot_np`(motion_process.py:750)+ 依赖(`recover_root_quat_and_pos_np`:700、Quaternions.__mul__/from_transforms/__neg__/transforms、Animation.positions_global 4x4 matmul 链、rotation_6d_to_matrix_np)。每个算子逐行 verbatim 官方源。**为何自包含**:原 SALAD 库依赖 numpy.core.umath_tests(新 numpy 已删)+ 重 BVH 依赖链,import 不进来。

## 改动 2:scripts/animate_anytop13.py(共享渲染器,被 6+ QA 脚本调用)
- import recover_from_bvh_rot_np
- 加 `--render_mode {rot6d,pos}` default=rot6d
- 渲染逻辑:render_mode=="rot6d" 时,GT 和 pred 都用 recover_from_bvh_rot_np(从各自 raw 13ch + item 的 rest_offsets/parent_indices 恢复);否则走旧 pos 路径(pred=_recover_world_positions, gt=motion_features[...,:3])。

## 请审(聚焦正确性 + 不破坏现有调用方)
1. **rot6d FK 复刻正确性**:anytop_rot6d_fk.py 的算子是否与官方 verbatim 一致(quaternion 乘法/共轭/from_transforms 符号、parent reindex `rot_q[:,p]=hml[:,j]`、root 修正 `-r_rot_quat*rot_q[:,0]`、positions_global 的 4x4 父子 matmul)?有无 off-by-one / 轴序 / dtype 隐患?
2. **GT/pred 对齐**:rot6d 模式下 GT 用 `np.asarray(item["anytop_x"]).transpose(2,0,1)` 反 de-norm 得 raw,pred 用 out["pred_motion"] 反 de-norm。两者 de-norm 公式一致吗(都 *(std+_STD_FLOOR)+mean)?offsets/parents 来自同一 item(同 new_to_old_perm 排列),与 anytop_x 对齐吗?
3. **不破坏旧调用方**:6+ QA 脚本(_render_longchain*.sh / _render_cleanL2_poison15_qa.sh 等)不传 --render_mode → 默认 rot6d。这是**行为变更**(它们之前隐式用 pos)。这个变更是否安全(渲染逻辑、speed_ratio 计算 line 183-184 用 gt_world/pred_world 仍 work)?有无脚本依赖 pos 特定行为会因此坏?
4. **边界**:T 裁剪(min(T_clip,T_valid))在 rot6d 路径下对 GT raw 切片 [:T] 是否正确?J 切片?contact_sheet/animate_clip 接口不变(都收 [T,J,3]+parents)?
5. dtype/shape:recover_from_bvh_rot_np 返回 float32 [T,J,3],与下游 np.diff/animate 兼容?

## 文件
- src/data/anytop_rot6d_fk.py(新,完整审)
- scripts/animate_anytop13.py(diff:import / argparse :104 / 渲染 :180-200)
- 参照官方:/scratch/ts1v23/_pz_pipeline 的 data_loaders/truebones/truebones_utils/motion_process.py:700/750(如可读)

## 输出
明确 [PASS | NEEDS-FIX]。NEEDS-FIX 给行号+具体修法。聚焦 FK 正确性 + 默认行为变更安全性。不复述背景。
