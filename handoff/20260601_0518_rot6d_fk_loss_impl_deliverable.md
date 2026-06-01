# rot6d-FK Loss 实现交付 / 审核文档

> **loss_mode = `anytop13_world_rot6d_fk`** — 按 `handoff/20260601_rot6d_fk_loss_plan.md` 实施
> 产出时刻: 2026-06-01 05:18 | 实施者: Claude (Opus 4.8)

```
STATE  (更新 2026-06-01 05:49 — 用户审完，已开跑 arm B)
  status:        ✅ arm B RUNNING (swarmh1002 2×H100 DDP, ep0 loss 11.7→4.2, ERR0, bs32 no-OOM)
  next-critical: 监控 B；ep30-50 / 首轮长链 QA 后由用户定是否跑 C(0.5/0.5/0.25)
  resource:      swarmh1002 alloc944459 我的 2×H100(UUID GPU-8681af2f/38df6f29 坐实非抢卡;
                 节点另6张 H100 属 jb3c20/mr21g23 不碰); diffusion=blossom04 不碰;
                 world_geometry(swarma1001) ep21@01:27 已死, 用户说不碰, 未重起
  config:        loss_mode=anytop13_world_rot6d_fk w0.25/f0.25/t0.10, bs32×2=global64,
                 lr4e-4 seed42 300ep, OUT=runs/m1_l2_anytop13_edgeseg_C128_rot6dfk_w025f025t010_seed42
  smoke:         PASS (rc0, no OOM/nan, val_recon 含 geometry=P2 fix 生效, DDP nproc2)
  uncommitted:   是(新 _launch_rot6d_fk_B.sh codex-PASS rc-fix; anytop_rot6d_fk.py 已 git add -f); 未 commit 留审
```

---

## §0 TL;DR（30 秒）

- 新增**组合几何监督** loss，**不替代**现有 `anytop13_world_geometry`：
  `L_total = L_anytop13_base + w_world·L_world_ric + w_fk·L_rot6d_fk + w_traj·L_root_traj`
- 与 world_geometry 的**核心区别**：fk 项让**非 root 旋转通道 ch3:9 有梯度**（world_geometry 对该通道零梯度）。已 smoke 证明（G4 grad=6.02）。
- FK 路线 = 官方 `recover_from_bvh_rot_np` 的 torch matrix port，**vs numpy 官方版误差 1.19e-6**。
- 默认 `loss_mode=anytop13` **数值零回归**（G2: total bit 一致，无 geo keys）。
- Calibration: @0.25/0.25/0.10 几何项占 base **8.5%**（偏弱）→ plan §6 规则触发**也要跑 0.5/0.5/0.25**。
- **推荐**: B(0.25/0.25/0.10) + C(0.5/0.5/0.25) 两个 arm 都跑，保持 w_world=w_fk。
- **待你定**: 权重方案 + 资源分配。**我不会自启训练。**

---

## §1 设计与公式

```
L_total = L_anytop13_base                          # 原 13ch recon（pos/rot/vel/contact/kl/pool_aux），不变
        + w_world · L_world_ric                     # RIC 恢复的 world joint 位置 L1，target=RIC(gt)
        + w_fk    · L_rot6d_fk                       # rot6d-FK 恢复的 world joint 位置 L1，target=RIC(gt)
        + w_traj  · L_root_traj                      # root xz 轨迹 L1
        (+ gt_fk_mismatch 仅诊断记录，不进 total)
```

**target 语义（plan §3 Option 2）**: world 和 fk **都** target `RIC(gt)`（不是 FK(gt)）。
- `L_world_ric = L1( RIC(pred), RIC(gt) )` — 两边同走 RIC 恢复路，**地板=0**（pred=gt 时归零）。
- `L_rot6d_fk = L1( FK(pred),  RIC(gt) )` — pred 走 FK 路、target 走 RIC 路。详见 §8 地板说明。

**为何用现有 ch3:9 不加 rotation head**: 第一版直接监督现有 anytop13 通道 `pred_motion[...,3:9]`（6D 旋转），不新增 `pred_aux_rot6d`/额外 head（plan 硬约束、用户 prompt 明确）。

---

## §2 改动文件列表（git numstat，确定性）

**新增:**
| 文件 | 大小 | 内容 |
|---|---|---|
| `src/models/graph_salad/rot6d_fk_recovery.py` | 6212 B | `recover_rot6d_fk_positions_torch` — torch matrix-only FK，port 官方 `recover_from_bvh_rot_np` |
| `src/data/anytop_rot6d_fk.py` | 7113 B | numpy 逐行官方版 = parity ground-truth。⚠ `.gitignore` 含 `data/` → commit 须 `git add -f` |

**修改（tracked）:**
| 文件 | +/− | 内容 |
|---|---|---|
| `src/models/graph_salad/losses.py` | **+164 −0** | 纯追加 `compute_world_rot6d_fk_terms` + `_masked_l1_xyz`；`compute_total_loss_13ch` **0 删除** |
| `src/models/graph_salad/__init__.py` | +2 −0 | export `compute_world_rot6d_fk_terms` |
| `scripts/train_graph_vae.py` | **+105 −7** | run_loss 新分支 + CLI(`--loss_mode` choice / `--w_fk` default 0.25) + P1 fail-loud + P2 val_recon 含 geo |
| `scripts/animate_anytop13.py` | +27 −3 | rot6d 默认渲染（**独立早前任务**，已单独 codex PASS，非本 loss 改动） |

**新增 smoke/util（untracked）**: `_smoke_rot6d_fk_torch.py`、`_smoke_world_rot6d_fk_wiring.py`、`_calibration_world_rot6d_fk.py` + 各自 `_out.txt` 存盘 + `_preflight_ric_vs_fk_L2.py` 等 diag。

> **git 状态**: 全部 **uncommitted**，留你审后再 commit。`anytop_rot6d_fk.py` 需 `git add -f`。

---

## §3 实现细节（审核重点）

### 3.1 FK 路径 `recover_rot6d_fk_positions_torch`（rot6d_fk_recovery.py）
签名: `(motion_13ch[B,T,J,13], parent_indices, rest_offsets[B,J,3], joint_mask[B,J]) -> [B,T,J,3]`

port 官方 `recover_from_bvh_rot_np` 的步骤（matrix-only，避免 quaternion 符号分支，autograd-friendly）：
1. **root R + pos**: 复用 RIC root 路径 — `_rot6d_to_matrix_torch(ch3:9)` 得 root_R；ch9/ch11 cumsum 得 root xz；ch1 得 root y。
2. **non-root R**: 从 ch3:9 直接转矩阵。
3. **parent reindex**: `local_R_list[p] = all_R[:, j]`（官方语义：把关节 j 的旋转赋到其父 p 槽位；**非** `[j]=[p]`）。
4. **root 修正**: `local_R_list[0] = root_R.transpose@local_R_list[0]`（= numpy 的 `-r_rot_quat * rot_q[0]`）。
5. **4×4 FK 链**: `glob_list[j] = matmul(glob_list[parent], local_j)`，**list 累积** + `out=out.clone()` 后赋值 — 避免 autograd inplace 破图。
6. 返回 `out * joint_mask[:,None,:,None]`。

✅ **正确性已证**: vs numpy 官方版 max_diff **1.192e-6**（§4 SMOKE1，4 物种）。codex 逐行核对 parent reindex / root 修正 / 排列无 off-by-one → PASS。
✅ **未偷用** `treeik_decoder.fk_persample`（plan 硬约束 4）。

### 3.2 loss 函数 `compute_world_rot6d_fk_terms`（losses.py 末尾纯追加）
- `pred_raw/gt_raw = _denorm_13ch(...)`（同 world_geometry）。
- `world = _masked_l1_xyz( recover_world_positions_torch(pred_raw), recover_world_positions_torch(gt_raw) )`
- `fk    = _masked_l1_xyz( recover_rot6d_fk_positions_torch(pred_raw,...), recover_world_positions_torch(gt_raw) )`
- `traj  = root xz L1`
- `gt_fk_mismatch = _masked_l1_xyz( FK(gt), RIC(gt) )` — **仅记录，不进 total**。
- **mask**: 全部用 `frame_mask`（run_loss 传入 `effective_frame_mask = frame_mask_recovered`），**不用** raw `batch.frame_mask`（plan 硬约束 5）；joint 维用 `joint_mask`。

### 3.3 train wiring（train_graph_vae.py）
- run_loss 加 `w_fk=0.0` 默认 + `anytop13_world_rot6d_fk` 分支: `total += w_world*world + w_fk*fk + w_traj*traj`（gt_fk_mismatch 不进 total）。
- CLI: `--loss_mode` choices 加该值；`--w_fk` default 0.25；复用 `--w_world`/`--w_traj`。
- 两调用点（train ~:772 / val ~:883）传 `w_fk=args.w_fk`。
- **P1 fail-loud**（~:413-418）: geometry loss_mode 配非 anytop13 feat_mode/dataset → `raise RuntimeError`，防静默忽略几何监督。
- **P2 val_recon**（~:948-968）: geometry mode 下 val_recon/best_recon_model.pt 现含 world/fk/traj（args 权重），gt_fk_mismatch 排除；默认 anytop13 时 geo_w={} → val_recon 不变（无回归）。

---

## §4 Smoke 结果（本轮 rose11 全部新跑、存盘，非记忆复述）

| Gate | 检验 | 结果 | 落盘 |
|---|---|---|---|
| **SMOKE1** | torch FK vs 官方 numpy（真实 clip） | max_diff **1.192e-6** < 1e-4 ✅ | `scripts/_smoke_rot6d_fk_torch_out.txt` |
| **G2** | `loss_mode=anytop13` 零回归(total==direct, 无 geo keys) | **1.996638 == 1.996638** ✅ | `scripts/_smoke_world_rot6d_fk_wiring_out.txt` |
| **G3** | world/fk/traj/gt_fk_mismatch/total 全 finite | 0.0708/0.2063/0.0556/0.1581/2.0715 ✅ | 同上 |
| **G4** | non-root rot6d(3:9) grad>0 = **FK 签名** | **6.0167** ✅ | 同上 |
| **G5** | non-root pos(0:3) grad>0 = world/RIC route | **3.0037** ✅ | 同上 |
| **G7** | gt_fk_mismatch 只记录、不 assert 零 | 0.1581 ✅ | 同上 |

SMOKE1 逐物种: 水巨蜥(J114/T64)=1.192e-6 · 灰海豹(J140/T32)=9.54e-7 · 科莫多(J92/T64)=7.15e-7 · 咸水鳄(J96/T48)=9.54e-7。

> ⚠ **G3 数值 ≠ §6 calibration 数值**，不矛盾：G3 用 `ds[0:4]` 前 4 样本（world=0.071）；§6 calibration 跨 val set 5 batch×4 linspace（world=0.156）。来源不同。

---

## §5 Codex 审查（gpt-5.5 xhigh, fresh thread，clean-context 规则）

- **初审**（FK 路径 + losses + 新模块 + 默认零回归 + autograd 安全）: **[PASS]**。
- **复审**（P1 fail-loud + P2 val_recon geo 两修复，确认无新回归）: **[PASS]**。
  - P1: 校验覆盖 world_geometry 与 world_rot6d_fk 两个 geometry mode，位置在 arg parse 后训练前，不误伤合法组合。
  - P2: geometry mode 下 val_recon 含 base recon_keys + world/fk/traj(args 权重)，gt_fk_mismatch 正确排除；默认 anytop13 时 val_recon 不变（无回归）；下游 metrics 无 KeyError。

---

## §6 Calibration scale 表（frozen baseline VAE，5 batch×4，val set；确定性复现）

落盘: `scripts/_calibration_world_rot6d_fk_out.txt`。baseline ckpt = `runs/_baseline_cleanL2_ep34_for_p1diag_compare/best_recon_model.pt`。

| 项 | raw | 加权 @0.25/0.25/0.10 | 占 base |
|---|---|---|---|
| **base_total** | **1.4598** | — | 100% |
| └ pos | 0.2253 | | |
| └ rot | 0.8500 | | |
| └ vel | 0.2590 | | |
| └ contact | 0.0319 | | |
| └ kl | 122.3244 (×1e-3) | | |
| └ pool_aux | 0.0000 | | |
| **world** (RIC) | 0.1561 | 0.0390 | **2.7%** |
| **fk** (rot6d-FK) | 0.2847 | 0.0712 | **4.9%** |
| **traj** (root xz) | 0.1411 | 0.0141 | **1.0%** |
| **几何合计** | | **0.1243** | **8.5%** |
| gt_fk_mismatch（诊断 floor，**不进 total**） | 0.1433 | — | — |

**plan §6 自动判读（脚本输出）:**
- world(2.7%) 与 fk(4.9%) **都 <10% of base** → **ALSO 跑 0.5/0.5/0.25**。
- 等权下 world:fk raw = **1 : 1.82** → 严格等权需 w_fk≈0.137。

---

## §7 推荐权重 + B/C arm（待你审）

**判读:**
1. @0.25/0.25/0.10 几何项仅占 base 8.5% → 温和 nudge，不冲击 base recon。
2. world:fk raw = 1:1.82，等权下 fk 贡献 ~1.8× world。你定调「不死守数字相等」，且 **fk(旋转监督)是本实验 novel 项，让它略强反而合理** → **建议保持 w_world=w_fk，不把 w_fk 降到 0.137**（降了削弱本就弱的信号）。
3. 见 §8 地板：fk 收敛后 raw 约 0.14，不归零，但梯度充足。

**推荐两个 arm 都跑:**
| arm | w_world / w_fk / w_traj | 几何占 base | 依据 |
|---|---|---|---|
| **B** | 0.25 / 0.25 / 0.10 | 8.5% | plan 钦定第一版，温和 |
| **C** | 0.50 / 0.50 / 0.25 | ~17% | **plan §6 规则触发**：B 下 world/fk 都<10% |

B/C bracket 几何压力有用区间（8.5%→17%），看 dose-response：C 是否进一步改善 world-space pose 而不伤 base recon。

**备选（不推荐，列出供你定）**: 等权 arm w_fk=0.137（严格 world≈fk 贡献）；或单跑 B 省资源。

> ⚠ **命名澄清**: 这俩是 **rot6d_fk 模式**新 arm，与 swarma1001 正跑的 `world_geometry`（world-only ablation, w0.5/t0.25）是**两个不同实验**，勿混。

---

## §8 数据固有地板 gt_fk_mismatch（务必理解再定权重）

- `gt_fk_mismatch = L1(FK(gt), RIC(gt)) = 0.1433`：RIC 与 FK 两条**恢复路在数据集 gt 上本身就有分歧**（preflight: clean_L2 median 1.2% 但 p95 30%，主要哺乳类 outlier；长链物种小）。该值**只依赖 gt、与训练无关 → 是常数**。
- 因 fk 项 target=RIC(gt)（plan Option 2），**fk 的最优值 = 该地板 ≈ 0.1433**，不会归零。
  - baseline fk=0.2847 → 可降信号 = 0.2847−0.1433 = **0.141**，梯度充足，能训。
  - 收敛后 fk raw ≈ 0.14，加权 ≈ w_fk×0.14。
- **mitigation**: world 项（同 target RIC(gt)、但走 RIC 路、**地板=0**）干净地把 pred 锚向 gt；fk 项额外在旋转空间加压。两者组合：world 保 pred→gt，fk 给非 root 旋转梯度。
- **监控**: 训练日志已记 gt_fk_mismatch；若 fk raw 卡在 ≫0.14 不降 = 旋转没学好；若 ≈0.14 = 已到地板（正常）。

---

## §9 plan 硬约束逐条核对（审核 checklist）

| # | 硬约束 | 遵守证据 |
|---|---|---|
| 1 | 不改 pool/decoder/dataset/renderer/d_model/max_frames/max_joints | 仅新增 loss 文件 + losses.py 纯追加(+164−0) + train wiring；无任何 model/data/render 改动 ✅ |
| 2 | 不改 compute_total_loss_13ch 默认行为；anytop13 数值不变 | G2: total bit 一致(1.996638)、无 geo keys ✅ |
| 3 | 不删 anytop13_world_geometry（world-only ablation 保留） | `compute_world_geometry_terms` 仍 export；CLI choices 仍含该 mode ✅ |
| 4 | FK 对齐官方 recover_from_bvh_rot_np，不偷用 treeik_decoder.fk_persample | rot6d_fk_recovery.py port 官方；SMOKE1 vs numpy 1.19e-6；未引用 treeik ✅ |
| 5 | masking 用 frame_mask_recovered，不用 raw batch.frame_mask | run_loss 传 effective_frame_mask=frame_mask_recovered；_masked_l1_xyz 用之 ✅ |

**用户 prompt §6.7 要求的 5 项 smoke 核对:**
torch FK vs numpy<1e-4 ✅(1.19e-6) · 默认数值一致 ✅(G2) · 新 mode 全 finite ✅(G3) · ch3:9 非零梯度 ✅(G4=6.02) · ch0:3 非零梯度 ✅(G5=3.00)。

---

## §10 资源选项（待你定）

- 每 arm = 一次 **300ep VAE 重训**；global batch 须匹配 baseline A 的 **64**（4×A100×bs16 或 2×H200×bs32，lr 4e-4 不变，无需 Goyal scaling）。
- **占用中不可动**: swarma1001(4×A100)=world_geometry; blossom04(2×H200)=diffusion T2M(ep37, ERR0)。
- **我手上空 alloc**: rose11(2×A100, 渲染节点)、swarmh1002(944459, 刚起 4h)。
- 选项: (a) 等 world_geometry 跑完腾 swarma1001 顺序 B→C; (b) swarmh1002 起一个; (c) 你另分配。

---

## §11 待你确认（确认后我才开跑，不自启）

1. **权重**: 照推荐 B(0.25/0.25/0.10)+C(0.5/0.5/0.25) 都跑？还是只跑一个 / 改权重 / 加等权 arm(w_fk=0.137)？
2. **资源**: 用哪个节点；顺序还是并行？

---

## §12 如何自查（你可独立复现，全确定性）

```bash
# 节点: rose11(我的 alloc 944466, 2×A100 空闲) | cd /scratch/ts1v23/workspace/noKslot_clean
# 1) FK vs 官方 numpy parity
CUDA_VISIBLE_DEVICES=0 python scripts/_smoke_rot6d_fk_torch.py        # 期望 SMOKE1 PASS, max_diff~1.19e-6
# 2) 接线 + 零回归 + 梯度签名
CUDA_VISIBLE_DEVICES=0 python scripts/_smoke_world_rot6d_fk_wiring.py # 期望 WIRING_SMOKE PASS, G2 total==direct
# 3) calibration scale
CUDA_VISIBLE_DEVICES=0 python scripts/_calibration_world_rot6d_fk.py  # 期望复现 §6 表
# 4) 看改动
git diff --numstat scripts/train_graph_vae.py src/models/graph_salad/losses.py src/models/graph_salad/__init__.py
git diff scripts/train_graph_vae.py   # 审 run_loss 分支 + P1(:413) + P2(:948)
```

落盘输出: `scripts/_smoke_rot6d_fk_torch_out.txt` · `scripts/_smoke_world_rot6d_fk_wiring_out.txt` · `scripts/_calibration_world_rot6d_fk_out.txt`
