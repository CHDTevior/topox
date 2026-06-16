# Codex Review: animate_denoiser rot6d-FK + pose 双路渲染

## 背景
用户要 T2M 生成渲染**同时显示 rot6d-FK 恢复 和 pose(RIC) 恢复两路**(验证生成 motion 两条恢复路自洽 —— 像之前 rot6d-FK double-root-rotation 修复时的 RIC-vs-FK 对比)。原 animate_denoiser 只渲 pose route。

## 改动(2 文件)
### scripts/animate.py — animate_t2m_input_pred
- 加可选 `pred_fk` 参数 + pred_label/pred_fk_label
- `pred_fk is None` → 2 panel(旧: input skel + pred); 非 None → 3 panel(input skel + PRED_pose + PRED_FK)
- n_panel/union/axes/update 全条件化(pred_fk is None 走旧路径, 数值/布局与改前一致)

### scripts/animate_denoiser.py
- import recover_rot6d_fk_positions_torch
- pred_world(pose, `_recover_world_positions` ch0:3) 后加 rot6d-FK route:
  - `pred_raw_t = torch.from_numpy(pred_raw).float()[None]`  → [1,T,J,13]
  - `rest_off_t = torch.from_numpy(rest_off).float()[None]`  → [1,J,3]
  - `jmask_t = torch.ones(1,J,dtype=bool)`
  - `pred_world_fk = recover_rot6d_fk_positions_torch(pred_raw_t, [parents], rest_off_t, jmask_t)[0].cpu().numpy()`
- 传 `pred_fk=pred_world_fk` + log 加 PRED_fk_speed

## 审查点(请逐一)
1. **recover_rot6d_fk_positions_torch 输入契约匹配?** 函数签名 `[B,T,J,13] RAW + parent_indices(list[B] of list) + rest_offsets[B,J,3] + joint_mask[B,J]bool`。传入: pred_raw[T,J,13]→[1,..], [parents](单元素list), rest_off[J,3]→[1,..], ones[1,J]bool — 形状/类型对吗?
2. **pred_raw 是 de-normalized RAW 13ch?** `pred_raw = pred_norm*(std+_STD_FLOOR)+mean` (animate_denoiser:307)。recover_rot6d_fk 内部从 RAW 13ch 取 ch3:9(6D rot)+root —— 它要的是 un-normalized RAW, pred_raw 正是 de-norm 后的, 对吗(不能传 normed)?
3. **pose route 与 FK route 用同一 pred_raw?** pose `_recover_world_positions(pred_raw)` + FK `recover_rot6d_fk_positions_torch(pred_raw_t,...)` 同源 pred_raw — 两路对比才有意义(自洽性检验)。确认无误传成 normed 或不同源。
4. **rot6d-FK 函数是修复版?** recover_rot6d_fk_positions_torch 应是 2026-06-01 double-root-rotation 修复后的版本(删了 root correction)。确认当前文件是 patched 版(否则两路会差 ~2x 旋转)。
5. **fp32/2-panel 回退等价?** pred_fk=None 时 animate_t2m_input_pred 与改动前 2-panel 布局/数值完全一致?
6. **stride-aware clipping 对 FK 路同样成立?** pred_raw 已 `[:T]` 切(T=min(T_clip,T_valid)), pred_world_fk 从同一 pred_raw 算 → 同样 T。确认无 tail。

请读 scripts/animate.py(animate_t2m_input_pred) + scripts/animate_denoiser.py(decode→恢复段 ~:288-355) + src/models/graph_salad/rot6d_fk_recovery.py(:60 签名+逻辑). 逐点结论 + PASS/NEEDS-FIX。这是重渲前的代码审(重渲会实测两路是否一致)。
