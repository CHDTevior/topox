# 架构问题记录:decoder 缺图-时序关节协调

> 记录于 2026-05-22。状态:**已裁决 (2026-05-22, A/B 实验)** — 见文末「裁决」。
> 简述:coarse_xattn (选项 A) = 小而稳的真实改进,建议设默认;选项 B 不上。

## 裁决 (2026-05-22 — A/B 实验结果)

跑了一个并行 A/B (各 1000ep,anytop13+dynamic+graphormer,只差 decoder_mode):
- A `unpool_identity` (本文记录的 baseline 弱点路径) — best val_recon **2.0725** (ep909)
- B `coarse_xattn` (选项 A 修复) — best val_recon **2.0442** (ep829),低 ~1.4%

结果:
- **metric**: B 一致小赚 (~1.4% val_recon),收敛 train_loss 基本打平。
- **视觉** (best_recon ckpt,Alligator 25J 最易看清): A 和 B 都产出连贯、跟 GT 的骨架,
  **A 没有可见的退化 artifact**。本文预测的「簇边界不连续 / 关节脱耦」未在视觉上出现。
  唯一一致差异:B 的逐帧位移幅度更贴 GT,A 略 over-shoot — 细微,非肉眼坏。
- **强假设「identity decoder → 可见运动 artifact」证据不支持** — decoder 不是质量瓶颈。
- caveat:全程只看 frame-0 + 6 帧 contact sheet,**「时间块状」(T_lat 16→64) 仍未证伪** —
  要连续帧渲染才能验。

决定:
1. `coarse_xattn` 设为默认 `decoder_mode`(零新参数、codex-PASS、val 小赚、位移更贴 GT,无下行)。
2. **选项 B (decoder 加图传播层) 不上** — 没有可见 artifact justify 这个大改。
3. 「时间块状」若以后要查 — 改 QA 渲染成连续帧,另一个小任务。

---

> 以下为 2026-05-22 实验前的原始记录,保留作背景。

## 问题

M1.7 anytop13 的 VAE 是**不对称**的:encoder 端图-时序很厚,decoder 端逐
关节很薄 —— decode 阶段没有任何**可学的**跨关节运动传播,也没有图-时序推理。

## 事实(代码核实,行号 git 4a6abb1)

decode 路径 `z → 输出`:

1. **`DynamicGraphUnpool`** (`vae.py:438`) — `unpool.py:59` 明确 "No nn.Parameters"。
   - `unpool.py:167`: `h_fine = einsum("bjc,btcd->btjd", P, z)` — 纯线性 scatter。
   - 时间维 `repeat_interleave` (`unpool.py:187`) — 最近邻复制(T_lat=16→64 块状)。
   - → unpool 零可学参数、零图传播。joint j 拿到 `Σ_c P[j,c]·z[c]` 的固定加权。

2. **`MotionDecoder`**,传入 `asg = _identity_assignment` (`vae.py:476`):
   - 内部 step-1 unpool (`motion_decoder.py:170`):identity → einsum 退化成恒等,no-op。
   - cross-attention ×3 (`motion_decoder.py:62-63`):`scores += assign_bias_scale·log(assignment)`,
     identity → 对角 0、非对角 ≈-18 → softmax 几乎只看自己。**退化成 per-joint 自我精修**。
   - temporal refine ×2 (`motion_decoder.py:187-190`):`[B*J,T,D]` 上的逐关节 1D conv,
     无跨关节混合。

## Nuance(两个方向都列,不只挑对结论有利的)

- **削弱担忧**:`s_j` 静态骨架嵌入有加到每个 joint (`motion_decoder.py:174`),
  来自 encoder graph_layers,是图感知的 —— 但**静态**(per-skeleton,不随运动变)。
- **削弱担忧**:dynamic pool 的 assignment 是软的,边界关节 row 分散到 2+ coarse
  node → unpool scatter 有一点跨簇混合。
- **逃逸口但弱**:`assign_bias_scale` 可学 (`motion_decoder.py:36`,init 1.0)。
  理论上能学到 →0 关掉 bias,但那时 cross-attn 变成 143 关节的**无结构 all-to-all**
  自注意力(没有 encoder 那种 adjacency/geodesic 边 bias)。init 强自偏。

## 为什么会这样(来源)

`_identity_assignment` 是 **M1.3 的 ckpt-兼容遗留物**:当时为复用 baseline `Model`
的 MotionDecoder、保 `decoder.*` key 1:1,才"先 unpool 再喂 identity"。
**这个理由对 anytop13 已不存在** —— anytop13 从头训、13ch、不 warm-start baseline。

## 这是 bug 吗

不是 bug,是**质量天花板**。VAE 的 decoder 表达力直接是重建质量上界。模型能
训、能学,问题是能否重建得**足够好**。

## 弱 decoder 的可预测失败模式(可视化 QA 时专门找这三个)

1. **簇边界不连续** — 同 coarse node 的关节拿到相同特征,不同簇间无 decoder 侧
   图平滑 → coarse-node 边界处关节漂移不一致。
2. **时间块状** — T_lat=16→64 的最近邻复制块状,逐关节 conv 平滑了单关节但跨
   关节时间协调(如左右脚步态同步)没建模。
3. **关节运动脱耦** — 运动学链上相邻关节不协调(髋-膝-踝)。

## 决策

先 launch 一次 anytop13 训练(dynamic pool + graphormer),用 `animate_anytop13.py`
渲 GT-vs-pred,**专门看上面三个失败模式**:
- 重建视觉干净 → encoder+pool+z 扛住了,decoder 别动。
- 出现簇边界 / 块状 / 脱耦 → decoder 升级是 evidence-driven 的,优先级清楚。

## 若需修复(evidence-driven 后)

原则性修法 —— 给 decoder 加图-时序层,对称于 encoder:
- **选项 A**:让 MotionDecoder 用**真** assignment(直接喂 coarse z,K=C,真
  `[B,J,C]` 软 assignment)而非 unpool-first + identity。恢复 cross-attention 的
  设计意图(slot→joint),但仍非 joint-joint 图传播。半措施。
- **选项 B**:给 decoder 加 graph-attention 层(同 encoder 的 graph_layers /
  `AnyTopGraphAttentionBlock`),在 unpool 后的逐关节特征上做拓扑感知的关节消息
  传递。真正的对称修法。
- **选项 C**:AnyTop 式 —— decoder 每层交替 spatial(图)+ temporal 注意力。

任何一个都是需专门规划 + codex 审的架构改动,不是顺手改。
