# rot6d-FK double-root-rotation bug 修复报告

**产出时刻**: 2026-06-01 21:02 BST
**状态**: 修复完成 + smoke PASS + codex PASS。**等用户审查 OK 后才启动训练(绝不自启动)**
**触发**: 用户看 GT_FK vs GT_RIC 渲染(Saiga)发现绿色(FK)比红色(RIC)多转近一倍全局旋转

---

## STATE (compact — 优先读这块)
- **bug**: rot6d-FK 恢复路径多乘一次 root global rotation(turn yaw 用了两遍)→ ~2× 全局旋转
- **修复**: 删 2 个文件各 1 行 root correction。numpy+torch smoke PASS(FK==RIC 2e-7),codex PASS(019e84c0)
- **影响**: world_geometry 安全;旧 B rot6d_fk(2卡 ep25) fk loss 曾被 double 污染 → 已停 + 4 卡重启;gt_fk_mismatch=0.29 是 bug 产物;之前 rot6d/三栏 QA 渲染全 double
- **DONE 2026-06-01**: FK 修复 + docstring + post-fix calibration(w_fk=1.0 用户定, 12.1% base) + 重渲 QA(全 0.0000) + cross-alloc 4 卡 infra(codex 019e84f9 PASS + smoke PASS IB NCCL) + **B 4 卡 durable 真跑**(global128 lr8e-4 w_fk=1.0)
- **B 4 卡监控**: `runs/m1_l2_anytop13_edgeseg_C128_rot6dfk_w025f100t010_g128_4card_seed42/train.log`(rank0 写, 绕过 orchestrator sed buffer); orchestrator PPID=1 durable(swarmh1002); 旧 2 卡污染 run 保留作诊断; 4 卡 launcher = scripts/_launch_rot6d_fk_B_4card.sh(static rendezvous + IB NCCL)
- **pending(用户定)**: diffusion ep10 lr 决策(swarma1004); world_geom resume 去向(swarma1001 GONE, 表现极好 val 1.395)

---

## 1. 发现 (visual QA primacy 的胜利)
用户看 GT_RIC vs GT_FK 渲染(Saiga),肉眼发现绿色 FK 骨架比红色 RIC 多转近一倍全局旋转。
这是**所有数字检查都没抓到**的 bug:
- SMOKE1 之前验证 torch FK == numpy FK = 1.19e-6 → 两个一起 double,一致但都错。
- "RIC-vs-FK <1%" 的旧验证用的是 root 几乎不动的样本(idle),没暴露(Black_Rhino idle 0.1%)。
- 数字看不出,运动可视化看得出 → 印证"CV 任务可视化优先于 metric"。

## 2. 诊断 (决定性, 双重铁证)
**判据**: RIC 路(`_recover_world_positions`, 从 ch0:3 position 恢复)= 可靠 ground truth
(pos loss target、训练在用、逻辑无 FK 递归)。正解必须 FK(data)==RIC(data)。

**clean_L2 Saiga** (scripts/_diag_fk_variants.py, 9 个 variant brute-force):

| variant | absL1 vs RIC | sweep ratio |
|---|---|---|
| A reindex+corr (旧/当前线上) | 0.6522 | 1.98 (double) |
| **B reindex, NO corr (修复)** | **0.0000** | **1.00** |

**老 AnyTop truebones 1070 条 cross-validation** (scripts/_diag_oldset_fk_variants.py):
扫描全 1070 找 root 旋转最大的 clip,top: Parrot_CircleFly 720°, Bird_CircleLand 714°,
Trex_turn180 396°。这 6 个最大旋转 clip: A(旧) absL1=0.46~1.06,**B(修复) absL1=0.0000(全部
逐点完美)**。bone/off=0.17/0.17 ... 1.00/1.00 → RIC 恢复的 bone length 精确匹配 offsets(物理
尺度正确)。

**用户独立验证**: 删 correction 后 rot6d-FK 与 ch0:3 pose 路误差 1e-9。

→ **不是数据处理问题**(老数据集和 clean_L2 表现完全一致),**不是 RIC 可视化错**(bone length
精确、修好的 FK 在 1070 条官方数据逐点==RIC),**是 FK recovery 那行 root correction 代码 bug**。
数据的 rotation(ch3:9)/position(ch0:3)通道是自洽的。

## 3. 根因
channel 语义(用户确认, 与老 AnyTop 一致): 0:3 root-invariant local pos / 3:9 AnyTop 6D
rotation token / 9:12 local vel / 12 foot contact。

3:9 的 root/child token **已含全局朝向变化**。recover 脚本 reindex(`rot_q[:,p]=hml[:,j]`)后,
root 槽 `rot_q[:,0]` 已是正确 root 朝向。旧脚本再做:
```
rotations[:, 0] = -r_rot_quat * rotations[:, 0]   # numpy
local_R[0]      = root_R^T @ local_R[0]            # torch (数学等价)
```
把 root global rotation(turn yaw)**用了第二遍** → 近似 2× 全局旋转。

## 4. 修复 (2 文件各删 1 行 root correction)
1. **src/data/anytop_rot6d_fk.py** (numpy `recover_from_bvh_rot_np`):
   删 `rot_q[:,0] = _quat_mul(_quat_neg(r_rot_quat), rot_q[:,0])`。
2. **src/models/graph_salad/rot6d_fk_recovery.py** (torch `recover_rot6d_fk_positions_torch`,
   **训练 fk loss 用它**): 删 `local_R_list[0] = torch.matmul(root_R.transpose(-1,-2), local_R_list[0])`。

两处都替为解释性注释 + 实证数据(absL1 0.65→0.0000)。`rot_q[:,0]`/`local_R[0]` 保留 reindex 后
的值(root child 的 rotation token, 已含正确 root 朝向)。

## 5. 验证
**smoke** (scripts/_smoke_fk_fix_torch.py, RESULT=PASS):
- numpy FK(fixed) vs RIC absL1=2e-7  (was 0.6522)
- torch FK(fixed) vs torch RIC absL1=3e-7
- torch FK(fixed) vs numpy FK absL1=2e-7  (impl parity 保持)
- autograd backward: grad_finite=True, norm=20876  (fk loss 仍可微)

**codex PASS** (gpt-5.5 xhigh, fresh thread `019e84c0`):
- 修复正确性 PASS(codex 复跑 smoke 确认)
- **多 root-child 安全**: codex 统计 clean_L2 全 473 skeleton(root child count 1~14, 确有多 child),
  抽样修复后 FK-vs-RIC 最大 absL1=1.54e-6, 同一 root 的多个 child rotation token 彼此 diff=0 →
  删 correction 不暴露新错误
- autograd PASS
- **建议重启 B**(不续训)
- 非阻塞建议: 2 文件顶部 docstring "verbatim/official"→"official-derived, patched"

## 6. 影响范围

| 组件 | 受影响? | 说明 |
|---|---|---|
| world_geometry loss (RIC torch, world_recovery.py) | ✅ 安全 | 走 position 通道, 无 FK 递归, 无 bug |
| **B rot6d_fk 训练 (swarmh1002, ep25)** | ❌ 污染 | fk loss w_fk=0.25 用 buggy torch FK, target double。**需重启** |
| gt_fk_mismatch diagnostic (实测 0.29) | ❌ bug 产物 | 修复后两路一致应≈0 |
| 之前 rot6d/三栏 QA 渲染 | ❌ double | 渲染脚本调 FK(已修), 重渲即可, 代码无需改 |
| 之前给的"哺乳动物 RIC-FK 分歧 60-87%"分析 | ❌ 大部分是 bug | 不是真实数据分歧; 修复后两路一致 |
| pos/vel/rot/bone base loss | ✅ 安全 | 不涉及 FK |

## 7. 待办 (等用户审查本报告 OK 后执行)
1. **FK loss 权重加大** (task #21): 见 §8。
2. **重渲 buggy QA 图** (task #22): 代码无需改; 重渲三栏(GT_RIC|PRED_RIC|PRED_FK)/GT_RIC-vs-GT_FK
   (现应几乎重合)/之前所有 rot6d render_mode 输出; 人眼复核。
3. **重启 B** (task #23): codex 建议; 用加大的 w_fk; 旧 run 保留作诊断。
4. **docstring 修正**: 2 文件顶部(codex 非阻塞建议)。
5. **re-measure gt_fk_mismatch**: 修复后应≈0, 重启 ep0 loss log 确认。

## 8. FK 权重提议 (待用户定具体值)
- 修复前: w_world=0.25, w_fk=0.25, w_traj=0.10。旧 launch 注释"raw fk≈1.82×world, keep
  w_world==w_fk" —— **那个 1.82× 是 buggy FK 的 double 放大, 不是真实信号强度**。
- 修复后: fk 和 world 同 target(RIC gt), 但 fk 监督 **ch3:9 rotation(via FK)**, world 监督
  **ch0:3 position(via RIC)**。fk 是 novel rotation 几何监督信号(区别于 base rot loss 的直接
  L1-on-rot6d)。
- **提议**: w_fk 0.25 → **0.5** (2×, 强调 rotation 几何监督), w_world 保持 0.25, w_traj 0.10。
  但**需 measure 重启后 ep0/ep1 的 fk raw**(修复后 scale 变了)确认 fk term 数值不压过 base loss,
  必要时 re-calibrate。**最终值待你定**(可指定别的倍数)。

## 9. 关键纪律
- **不自启动训练** —— 等用户审查本报告 OK 后才重启 B (不能 self-submit Slurm)。
- 旧 B run `runs/m1_l2_anytop13_edgeseg_C128_rot6dfk_w025f025t010_seed42` 保留作诊断(不删)。

## 附: 相关文件
- 修复(已改): src/data/anytop_rot6d_fk.py, src/models/graph_salad/rot6d_fk_recovery.py
- 诊断: scripts/_diag_fk_variants.py, _diag_oldset_fk_variants.py, _diag_root_double_rot.py
- smoke: scripts/_smoke_fk_fix_torch.py
- codex brief: scripts/_codex_fk_fix_brief.md
- RIC ground truth(对照): src/data/anytop_dataset.py:282, src/models/graph_salad/world_recovery.py
- fk loss 调用: src/models/graph_salad/losses.py:684 compute_world_rot6d_fk_terms
- B launch: scripts/_launch_rot6d_fk_B.sh
