# Kimodo Eq.1 全项对齐记录 — γ₇ FK↔RIC 一致性项实装 + run-1/run-3 重跑

**日期**: 2026-08-19
**指令(user 原话)**: "kimodo有的你都得加,你先停了现有的,然后做一下对齐kimodo的run1和run-3,并且记录一下"
**补充(user)**: 参考仓 https://github.com/CHDTevior/moge_UMO_ST(user 本人在人体数据上写过同款
Kimodo 式 loss);**但它只是参考,本轮按我们现有 13ch/多骨架格式适配**;user 正在安排一个"类似
他数据格式"的转换,转换好之后可以试用该格式再跑 run-1/run-3(→ 待办,见 §6)。

## 1. 停训记录(user 显式授权 "你先停了现有的")

- **run-3 (`runs/v2_pzh_262m`, 262M×PZ+human, 4×H200)** 于 **2026-08-19T08:35:50Z 停止**。
  停点:g36200/ep6;val 轨迹 47.4→…→3.29→3.10(g35000)→**2.81(g36000, best)**,仍在下降;NF=0。
  盘上 20 个 ckpt 完整保留(g2000..g36000 每 2000 步 + best + last)——
  **这是 262M 的无-γ₇ 基线**,run-3-kimodo 的 A/B 参照(尤其 H4 复测锚点 g36000)。
- 杀法:双节点 pkill(TERM→KILL),核验两节点 train 进程 0、四卡 0%/0MiB。**未动 alloc**
  (1383550 flamingo02 / 1383547 blossom01 保留给重跑)。
- 三个监视/接力 cron(48b0bd4d/6650b136/f24a1fa3)已删,防止把停掉的训练自动拉起。
- run-1 (`runs/v2_incontext_run1`) 早已自然收尾(1000ep),无需停。

## 2. 对齐审计:Kimodo Eq.1 七项 vs 我们

| # | Kimodo 项 | γ | 我们(改前) | 我们(改后) |
|---|---|---|---|---|
| 1 | r^p root 位置/速度 | 10.0 | ✅ root_pos=10.0 | 同 |
| 2 | r^a root 朝向 | 2.0 | ✅ root_rot=2.0 | 同 |
| 3 | j^p 非根位置 | 10.0 | ✅ body_pos=10.0 | 同 |
| 4 | j^v 非根速度 | 3.0 | ✅ body_vel=3.0 | 同 |
| 5 | j^a 非根旋转 | 10.0 | ✅ body_rot=10.0 | 同 |
| 6 | f 接触 | 4.0 | ✅ contact=4.0 | 同 |
| 7 | **FK(ĵᵃ) vs ĵᵖ 一致性** | **5.0** | **❌ 刻意推迟(有代码注释存证)** | **✅ 本次实装** |

结论:差集恰好 = γ₇。1000-epoch run-1 诊断实证了缺它的后果:H4(FK↔RIC 不一致)未见骨架
4–19%、随训练恶化、纯训练治不了 —— γ₇ 正是为此存在的项。

## 3. 有记录的适配偏差(多骨架现实所迫,非随意)

1. **√(N_i/N_tot) 组尺寸归一(保留)** — 纯组均值在我们 J=24..102 异质骨架上让 root 抢
   95.14% 梯度(share∝γ²E/N 推导);Kimodo 单人骨架无此问题。run-1 起就有、已审、已验证。
2. **γ₇ 残差尺度 = 每骨架 0.15×平均骨长**(替代 hy273 的固定 5cm/Kimodo 的隐式人体尺度)——
   否则 Trex 的物理失配贡献 ~100× Chick,γ₇ 变成"只训大骨架"。0.15 比例 ≈ hy273 的
   0.05m/人体平均骨长 0.33m。
3. **smooth-L1(beta=1.0) 替代论文的纯 L1** — 采用 user 自己的 hy273 实战配方
   (moge_UMO_ST `models/raw_motion/hy273_multitask_losses.py`):早期 x̂₁ 大失配时梯度有界。
4. **fk_warmup_steps 线性爬坡**(hy273 配方,默认 5000 步;run-1-kimodo 用 1000,数据小步数少)。
5. **weight-0 诊断项 fk_dist**(hy273 的 fk_distance_cm 同款):|FK−RIC| 骨长单位,
   训练日志里免费的在线 H4 监视,不进梯度。
6. **γ₇ 不乘 v_space 的 1/(1−t)² 权** — 一致性是 x̂₁ 在任意 t 的属性,两个参照实现都不加权。
7. **父槽重索引"最后子节点胜出"逐字保留**(官方 recover 的怪癖;预测数据上只有最后一个兄弟
   的通道吃到 FK 梯度 —— 记录在案,不"修",忠实性优先)。
8. **root 双旋转修正保持移除态**(anytop_rot6d_fk.py 2026-06-01 的修正,FK==RIC absL1=0 的
   那个版本;torch 移植带同款修正)。

## 3.5 γ 权重定案 = 0.25(斜率等效对齐,非照抄 5.0)— codex round-1 fix 2

Kimodo 的 γ₇=5.0 作用在**米制原始残差**上;我们把残差除以 0.15×骨长(人≈0.05m)后,照抄 5.0
等于把物理斜率放大 ~20×(codex 实测随机 init 下 FK 梯度 12-80× 组 loss)。**斜率等效换算**:
γ = 5.0 × 0.15 × 0.33m ≈ **0.25** —— 对人体尺寸骨架,物理单位斜率与 Kimodo 完全一致。
实测标定(run-1 ep1000 真病灶,‖∇(γL_fk)‖/‖∇L_grouped‖):

| γ | Rat J18 | Rhino J43 | Centipede J83 | 混合 |
|---|---|---|---|---|
| 0.25 | 0.28 | 0.35 | 0.20 | 0.14 |
| 0.5 | 0.56 | 0.69 | 0.39 | 0.29 |

γ=0.25 落在目标窗 0.1-0.35,深度差异温和(codex 噪声探针的 80× 深度炸裂在真实预测上不出现)。
顺带病灶读数(run-1 ep1000 fk_dist):J18 0.14bl / J43 0.24bl / J83 0.51bl,深链更糟,与 H4 %
读数一致。**发射参数:--gamma_fk 0.25;--fk_warmup_steps 1000(TB)/ 5000(262M)。**

codex 审查(thread 01a01939,gpt-5.6-sol max):round-1 NEEDS-FIX 3 项已修——
① FK/RIC 曾用不同 rot6d 归一化内核(退化输入造幻影残差)→ 共用 world_recovery 同一内核,
退化 gate(零/微小/平行)3/3 PASS;② γ 标定(上表);③ 热路径同步(parents/n_joints 留 CPU、
frame_mask 单次 CPU 拷贝 + index_select、诊断 float 仅 return_parts 时算)。

## 4. 实装清单

- `src/models/v2/fk_torch.py`(新)— 官方 rot-FK 的可微 torch 移植(矩阵链,免分支)+
  `fk_ric_consistency_loss`(smooth-L1/骨长尺度/fp32+autocast 关断/返回 loss+诊断)。
- `src/models/v2/dit_motion.py` — `cfm_loss(..., gamma_fk=0.0, fk_pack=None)`;历史注释更新。
- `src/data/incontext_pairs.py` — `InContextPairs(emit_fk_fields=)` 增发
  anytop_mean/std、parents、rest_offsets(与 preflight FK gate 同源同序);collate 补 pad。
- `scripts/train_v2_incontext.py` — `--gamma_fk`(默认 0.0=旧目标,resume 契约已加此键)、
  `--fk_warmup_steps`;train 侧爬坡、val 侧全权重;val 行打印 fk/fkdist。
- `scratch/_test_gamma7_gtzero.py` — 使能 gate ×4:torch-FK 忠实性(vs numpy ≤1e-3)、
  GT-zero(GT 上 fkdist≈0)、双通道族梯度、collate 混合-J 批路径=逐样本路径。

## 5. 重跑计划(A/B 各自单变量)

- **run-1-kimodo**(TrueBones):run-1 原目标 + γ₇(γ_fk=0.25 斜率等效,见 §3.5),**不带** JiT/CFG(vs run-1 单变量)。
  4×H200 DDP(user 硬约束"下一次实验起 DDP")global32/lr1.2e-3(Goyal k=4,roadmap 预定案;
  小数据大 batch 盯 val 形状,退路 global8)。500ep 先行(~2-3h),验收:**H4 从 4-19% → ~0**。
- **run-3-kimodo**(262M PZ+human):run-3 原配方(JiT+CFG+logitnormal)+ γ₇(γ_fk=0.25)
  (vs 停点基线单变量),4×H200 跨节点,fk_warmup 5000 步。
- 顺序:先 run-1-kimodo(便宜、快、直接回答"γ₇ 是否关门"),PASS 再烧 run-3-kimodo。
- 流程铁律不变:codex 审 → GT-zero gate → smoke(标定已完成,见 §3.5)→ 真跑。

## 6. 待办 / 触发件

- [x] codex round-1(NEEDS-FIX 3 项)→ 全修;round-2 确认发出(thread 01a01939)
- [x] γ₇ 梯度占比标定(§3.5,γ=0.25 定案)
- [x] 诊断仪 pzh 参数化 + **262M 停点无-γ₇ H4 基线已测**(g36000/ep6,GT 全 0.0000%):
      人 3.97% / Fossa 8.82% / Cheetah 5.78% / Wolverine 4.65% / Panda 11.79% / Caracal 7.84%
      → 4-12%,与 run-1 同族;大模型 ep6 未自愈。JSON: scratch/_diag_262m_g36000_noG7.json。
      (顺带:262M 的 H1 jitter 已普遍 <1.0,大容量对抖动病有效,病灶聚焦 H4。)
- [ ] codex round-2 确认 → 4 卡 smoke(1ep,验 fk 日志行/NF/吞吐)→ run-1-kimodo 发射
- [ ] run-1-kimodo ep100/300/500 H4 复测(验收 4-19% → ~0)
- [ ] run-3-kimodo 发射(今晚 alloc 续期后,满 walltime 起跑;--gamma_fk 0.25 --fk_warmup_steps 5000)
- [ ] **user 的 hy273 式数据格式转换完成后**:用该格式再试 run-1/run-3(user 2026-08-19 预告)
