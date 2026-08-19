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

## 7. run-1-kimodo(run1a)判决:FAIL-混淆,已纠正重跑 run-1b(2026-08-19 ~12:30Z)

**run1a(global32/lr1.2e-3/500ep)完赛读数**:H4 best@479 = Alligator 6.84 / Trex 7.77 /
BrownBear 10.43 / Elephant 13.73 / Monkey 15.21 / Raptor 8.96%(锚:run-1 ep500 无γ₇ =
3.12/5.24/12.35/18.38%)→ **H4 没关门,原地踏步**。同时 **H1 抖动崩 3-17×**(Alligator
3.04→52.9、Trex 1.49→24.8 @50步)。

**根因=实验设计错误(我的)**:Goyal 缩 batch/lr 但保 epoch 数 → 500ep×22步=**1.1 万步**,
而锚点 run-1 ep500 = **4.6 万步**。run-1 自己在 ~1 万步(ep100)就是"有形状但很抖"阶段 ——
run1a 卡在同一欠火候区,三变量混淆(γ₇/lr 区制/步数),γ₇ 疗效不可判读。
**教训(重要,写入记忆):小数据集(TB 734 targets)收敛是步数驱动;跨 batch 配置比较必须
step-matched,不能 epoch-matched。roadmap 里 run-2 的 350ep@global32 计划(7.7k 步)同样中招,作废。**
另:run1a 尾段 fkdist 反弹(0.52→0.64)、last H4 劣于 best —— lr1.2e-3 区制尾段不稳的旁证。

**run-1b 已发射**:run-1 逐字配方(global8=B2×4rank / lr3e-4 / 无 warmup / 500ep=46k 步)+
唯一新变量 γ_fk=0.25(fk_warmup 1000 步≈11ep)。~40s/ep,ETA ~5.5h。监视 cron 942c1d09。
渲染已发 user:kimodo_best 6 骨架 + run1_ep500_ref 同条件对照 6 骨架(视觉裁决权归 user)。
run1a 产物保留 runs/v2_tb_kimodo_run1(它意外成了"γ₇ 在高 lr 大 batch 区制下也压不住欠训抖动"
的数据点,以及 val-fkdist 1.5→0.52bl 半程下降的记录)。

## 8. γ 剂量研究(run-1b/1c/1d,2026-08-19 12:30-18:30Z)— 完整曲线与判决

四跑同配方(run-1 逐字:global8=B2×4rank / lr3e-4 / 500ep=46k 步),唯一变量 γ_fk。
H4(last ckpt,ODE 10 步完整采样,GT 全 0.0000%):

| 骨架 | 无γ₇ | γ0.25 | γ0.5 | γ1.0 | γ1.0 best@444 |
|---|---|---|---|---|---|
| Alligator(已见) | 3.12 | 3.31 | 2.72 | 1.83 | 1.86 |
| Trex(已见) | 5.24 | 5.50 | 2.39 | 2.37 | 2.11 |
| BrownBear(未见) | 12.35 | 10.16 | 8.19 | 7.53 | 6.19 |
| Elephant(未见) | 18.38 | 10.11 | 8.24 | 7.49 | 8.92 |
| Monkey(未见) | ~15带内 | 23.01 | 12.18 | 10.57 | 9.18 |
| Raptor(未见) | — | 7.56 | 4.97 | 3.81 | 4.17 |
| **4骨架均值** | **9.77** | 7.27 | 5.39 | **4.81** | 4.77 |

判决三条:
1. **已见骨架 H4 已进 0-2% 验收区**(γ1.0:Alligator 1.83-1.86 / Trex 2.11-2.37)——
   γ₇ 在训练分布内把两族嘴关上了。**剩余缺口全部集中在未见骨架**(7.5-10.6%),
   即残余问题已从"目标函数缺项"转化为"泛化缺口" —— 这正是 262M+PZ 312 骨架的用武之地。
2. **零质量代价,多处白赚**:全 γ 档 H1 不劣化;γ1.0 下 Monkey 抖动 0.2×(远稳于 GT)、
   Trex 1.3×;flow-only val ≈3.4-4.0 vs 无γ₇ 基线 best 4.437(**γ₇ 顺手压掉了 ep312 的
   过拟合拐点,best 从 ep312 挪到 ep444-494**)。
3. **收益边际递减**(4骨架均值:9.77→7.27→5.39→4.81),0.5→1.0 只再赚 0.6pp,权重轴到头;
   未见骨架的残余靠数据/容量,不靠再加 γ。

**定案:run-3-kimodo 用 γ_fk=1.0,fk_warmup_steps=5000**(1.0 在每个测量轴上 ≥0.5,无任何劣化)。
诊断 JSON:scratch/_diag_kimodo_run1{b,c,d}_{best,last}.json;渲染四组已全部 SendUserFile
(renders/20260819_kimodo_run1/{run1_ep500_ref,kimodo_best,kimodo_run1c_best,kimodo_run1d_best})。

## 9. Graph-v2:结构化关节特征 + 方向化逐 head 偏置(2026-08-19 晚,user 指令"先做小实验,大的往后放")

**代码取证(user 要求核实,全部属实)**:偏置=−clip(hops,8) 单标量(incontext_pairs.py:58/256/301,
注释自认 hops 达 ~20);所有 head 共用(dit_motion.py:70 expand);j_pos=纯槽位表(:157-165);
模型 forward 零静息骨架信息(rest_offsets 只进 γ₇ loss);Floyd 测地无向。

**两刀(全 flag 门控,关=逐位不变)**:
1. `--struct_feats`:每关节 8 维结构描述子(单位 offset xyz/log 骨长/跳数深度/物理深度/子数/叶旗标,
   本骨架平均骨长为单位),小 MLP 进 token;按骨架缓存(_graph_v2_tables)。
2. `--dir_bias`:LCA 分解 (up,down) 跳数矩阵(clip 15),两张 Embedding(16, n_heads) 零初始化表,
   **加在** −clip(geo,8) 之上 → 步 0 与基线逐位相同,表学逐 head 方向修正。

**自测 6/6 PASS**(scratch/_test_graphv2.py):LCA up+down==Floyd 全部 55 TB 骨架(建缓存内置
assert);零初始化 dir 模型与基线逐位相同(0.00e+00);struct 改变输出+双模块梯度通;混合 J(18+83)
pad 前向有限;特征范围合规。诊断/渲染脚本已兼容 v2 ckpt(flags 从 ckpt args 读)。

**实验矩阵(codex 01a01b1a 三轮审后定案:E0-E3 四臂因果 factorial,全部 γ_fk=1.0,46k 步协议)**:
E0=无刀因果基线(flamingo02 GPU1)/ E1=struct only(blossom01 GPU1)/ E2=dir only(flamingo02
GPU0)/ E3=both(blossom01 GPU0)。发射器 scripts/_launch_gv2_factorial.sh(单臂单卡 B8 lr3e-4,
flock,自动 resume 至满 500ep —— alloc 到期截断后续期补齐,四臂终点 step-matched)。
codex 三轮修复:①偏置零物化(SDPA 广播 [B,1,H,J,J],省 0.75-1.57GiB,反快 2s/ep);②可选模块
挪 __init__ 末 + struct 头零初始化 → 四臂共享权重逐位同、t=0 函数同一;③DataLoader 显式
generator(a.seed+7777)→ 四臂数据流同一。E0 而非 run-1d 作因果基线(grouped_loss 按进程本地
归一,B2×4 梯度均值≠单卡 B8;E0 同时把注意力重构本身控制住)。~4.7h/臂。
**验收指标:B 桶(未见骨架)H4 地板 7.5-10.6% 是否被砸穿 + 深链 fkdist + H1 不劣化 + 渲染目验。**
**大实验(run-3-kimodo)推迟至本结论出炉(user 2026-08-19 晚指令);续期守望 cron 已删。**
