# Kimodo/BTJD 表示迁移方案 — 绝对 root + heading 通道的 [B,T,J,D'] 重构

> **codex 审查:r1 NEEDS-FIX 已修订(意见 12 条)。**
>
> **定位:方案制定,未排期实施,供 user 决策。** 本文档综合三份调研(依赖面审计 / heading 设计 / 通道布局设计)成一份可决策的迁移方案;所有数字与行号均来自输入材料与 repo 现状,未虚构。

---

## 1 动机

### 1.1 积分漂移的机制

当前 13ch 表示中,root 世界 XZ 平移是**唯一的积分量**:恢复路径为
`root XZ = quat_neg(root_rot) · [ch9, 0, ch11] 逐帧 cumsum`(`src/data/anytop_rot6d_fk.py:113-122`,Y 直取 ch1)。即逐帧局部系速度先逆旋回世界系、再沿时间累加。后果:

- 任何一帧的 yaw / 速度误差都被 cumsum **永久带入后续所有帧**,把整条世界轨迹弯掉——误差不是逐帧局部的,而是累积放大的("cumsum-amplified drift",`src/models/vq_model/utils.py:4-10` 的 QA 工具即为此而写)。
- 需要澄清的边界(依赖面审计确认):**只有 root 平移在积分**。root 朝向 ch3:9 已是逐帧绝对 rot6d(恢复直接读);root ch0(yaw 角速度)不被任何恢复函数使用;root ch2 在 human 转换器里写 0(`convert_humanml3d_to_anytop13.py:319`)。所以"绝对 root 化" = 只替换 ch9/11-cumsum 这一条 xz 路径。

### 1.2 为什么之前的 QA 没抓到

已有诊断结论:**骨架 GIF QA 是 root-centered 的**(`animate_denoiser.py` 的 `make_t2m_large_gif` 在 :213-214 对世界位置做 root-centering),渲染前把 root 平移减掉了——积分漂移恰好被这一步抹平,GIF 上不可见。**漂移只在蒙皮导出(PZ skinning,世界系)才暴露**。这也是"可视化 demo 准确度 > metric"规则下需要修的表示级缺陷:QA 通路本身对该失败模式盲。

### 1.3 kimodo / ARDY 的绝对 root 论证

- **kimodo**:root 平移显式存绝对量(smooth-root + 残差),恢复变成纯加法,零积分;误差逐帧局部化、不累积。速度仍保留但只作输入特征,不作恢复依赖。
- **ARDY 论据**:heading 作为**存储通道**而非逐帧 canonicalize 算子——heading 定义在奇异点附近退化时,只是该通道数据快摆,不会像逐帧转正那样把整个 pose 表示撕裂。
- kimodo 对"逐帧 heading 旋转关节位置"的批评同样命中我们现有非 root ch0:3(root-heading 局部系 RIC 位置),因此本方案把非 root 位置一并改为"世界轴对齐、只减 root-XZ"(见 §2;r1 修订后参考 raw root)。

---

## 2 目标表示设计

### 2.1 推荐布局:Option A "local-rot 最小重构",D=14(保持 [B,T,J,D] 图结构)

核心转变:世界 XZ 从"逐帧局部速度 cumsum 积分"改为"显式存储绝对量"。root / 非 root 通道语义分裂沿用现有 `anytop13_split` 双 Linear 模式(`encoder.py:349-351, 448-456`),无新机制。

**表示代数(r1 修订,codex 意见 1):存 raw 绝对 root,非 root 参考 raw root;root / 非 root 恢复方程分开写、全部与速度通道无关:**

- **root 恢复**:`p_root(t) = [ch0, ch1, ch2](t)` — 直读(crop-canonical 系,§2.3)世界绝对位置,零积分、零平滑器依赖。
- **非 root 恢复**(j≠root):`p_j(t) = [ch0_j(t) + ch0_root(t), ch1_j(t), ch2_j(t) + ch2_root(t)]` — XZ 存"世界位置 − raw root XZ"、恢复逐帧加回当帧 root XZ;Y 直读世界高度。
- 两条方程均逐帧局部、无 cumsum、不读 ch9:12。r1 首稿"root 存 smooth_root、非 root 减 smooth_root_XZ"的代数**不闭合**——raw 与 smooth 的残差无通道承载,raw root 不可恢复,已作废。
- **A2 备选(仅当 A1 的 raw-root 通道在 Phase 3 recon 对照中被证实 ill-conditioned)**:kimodo-faithful 双存——root 增 2 通道(D=16,root-only,非 root 零填充 + 零 loss 掩码)存 smooth_root XZ,ch0/ch2 改存残差(raw − smooth);恢复 `p_root = residual + smooth`、`p_j = ch0:3_j + smooth_XZ`。代价:通道预算 + SavGol 构建与短 clip 边界行为回到关键路径(§2.3、G-cpu6)。

| 通道 | root (j=0) | 非 root | 说明 |
|---|---|---|---|
| ch0:3 | **raw 绝对世界位置 `[x, y, z]`**(crop-canonical 系,§2.3) | 世界位置 XZ − raw root XZ;Y 世界高度(**不再转 root 局部系**) | 恢复纯加法(方程见上),零积分、与速度无关 |
| ch3:9 | global rot6d(现状不变) | **per-parent 局部 rot6d(v4b 现数据不动)** | v4b real-twist 已验证 jitter ~1.0× / FK-floor 0 / MPJPE ~20mm(来源见 §6.12),是 learnable 编码;勿为 kimodo-faithful 重蹈 inferred-DOF ill-conditioned target 覆辙 |
| ch9:12 | 世界系差分速度 xyz(原 ch9/11 局部系 XZ 改世界系,语义统一) | 世界系差分速度(对应 kimodo [203:269]) | **保留但降级为纯特征,恢复路径完全不用**(理由见 §2.4;autograd 零梯度 gate 见 G-cpu2) |
| ch12:14 / ch12,13 | heading `[cosθ, sinθ]`(定义见 §3) | ch12 = contact 0/1,ch13 = 0 填充 | root-heading 与非 root-contact 是**不同语义、不同监督**:loss 按 root/非 root 语义掩码分开路由(§4.3,codex 意见 8);root 原 contact 位是否可弃须 Phase 0 语料审计支持(§2.4);非 root ch13 padding 零 loss、零梯度;heading/contact 走**归一化 bypass 掩码**而非 z-score(见下注) |

> **归一化注(r1 修订,codex 意见 9)**:r1 首稿"heading/contact 强制 identity moments(mean=0, std=1)"方案**作废**——假矩会被 moments 重建/审计工具误读为真实统计。改为:归一化代码(现 `anytop_dataset.py:1184-1188` 对全 13ch 无差别 z-score)按 §2.5 schema 的**显式 bypass 掩码**跳过 heading/contact/padding 通道;per-rig moments 只在 **train split、per-crop 重锚后的分布**上重建(§2.3),`moment_source` 抽象(`anytop_dataset.py:533,547,1174-1181`)语义同步更新;`anytop_dataset.py:1186/1188` 的 `nan_to_num` 移除,换成**携带 clip/joint/channel 标识的有限性断言**(fail loud,不静默清 NaN)。

### 2.2 备选布局

| 方案 | 内容 | 取舍 |
|---|---|---|
| **B:kimodo-faithful 全局旋转,D=14** | 布局同 A,但非 root ch3:9 改 GLOBAL cont6d(kimodo [71:203] 原样) | 优点:蒙皮 LBS 直接吃全局旋转、免 FK 链累积。代价:90k clip × 179 拓扑旋转全量重 derive(v4b re-encode 级工程)、twist 约定重新钉死(ill-conditioned target 复发风险,须先小规模 jitter 验证)、FK 工具链全重写。**仅当 A 的蒙皮质量被链式误差卡住时升级** |
| **C:精简无速度,D=11** | 删全部速度通道(位置差分可重算;decode-loss 的速度监督在解码 x0 上做,不需要通道) | 优点:D 最小、无"预测速度与位置差分矛盾"的冗余不一致。风险:encoder 失去显式速度输入(AnyTop/kimodo 都保留),能量塌缩史提示速度信号重要;**须 A/C 小规模 recon 对照后才可采** |
| **D:最小 13ch 绝对 root(依赖面审计的推荐)** | 不加通道、不做 smooth root/heading:root ch0←abs_x、ch2←abs_z(ch0 yaw-vel 恢复不用、ch2 human 侧已 dead),ch1 高度、ch3:9、ch9/11 速度、ch12 contact 均不动,保持 [B,T,J,13] 与 pos-group 0:3 切片语义 | 代码面最小(小改 + 一个重编码脚本;r1 行数估计按意见 12 移除),**只修积分漂移、不修 heading/canonicalize/非root参考系**。可作为"分两步走"的第一步或低风险回退方案;但不解决 kimodo 指出的逐帧 heading 旋转问题 |

> **冲突显式化(不糅合)**:输入 A(依赖面)推荐的是方案 D(13ch 最小改),输入 C(通道布局)推荐的是方案 A(D=14 全量 kimodo 化)。两者不是矛盾而是两档 scope:D 是 A 的严格子集(只动 root xz 通道语义)。本文档主推 **A(D=14)** 作为长期表示修的目标形态,D 作为"若 user 只想先消积分漂移"的降档选项;若选 D,§3 heading 与 §2.3 smooth-root/canonicalize 全部推迟。

### 2.3 crop 语义、canonicalize 与(仅 A2)smooth root

- **crop 语义在归一化 / tokenize 之前钉死(r1 修订,codex 意见 5)**。绝对量表示下 crop 不再天然平移/朝向不变;r1 首稿"build-time 每 clip 锚定、crop 无需重锚、`__getitem__` 零改动"的说法**作废**——crop 起点的绝对 XZ/θ 泄漏 clip 内位置且无条件承载。二选一:
  - **推荐:per-crop 重锚 + 重算起始 heading**——`__getitem__`(`anytop_dataset.py:1207-1235`)取 crop 后,平移使 crop frame-0 raw root XZ = 原点、刚体 yaw 旋转使 crop frame-0 heading θ0 = 0(旋转全部世界系量:位置 / root global rot6d 左乘 / 速度 / heading cos-sin)。分布对 crop 位置与朝向不变;代价:per-rig mean/std 必须在**重锚后的 crop 分布**上(train-only)重建,不能沿用整 clip 统计(§2.1 归一化注)。
  - 备选:build-time 每 clip 锚定 + 把 **crop 起点平移与 crop 起始 heading 作为显式条件**注入 backbone(条件 plumbing 见 §4.8;采样时须给出合理的条件分布,链路显著更重)。
- **canonicalize 推论(主推路线)**:per-crop 重锚使每个样本都处于"crop 起点 = 原点、θ0 = 0"的 canonical 系;生成结果的世界摆放由消费方(渲染 / 蒙皮 / 拼接)以刚体变换后置施加。此时 first_heading 条件恒为 0,**heading_proj 条件注入不需要**;heading 通道仍保留,承载 clip 内相对朝向演化 θ(t)−θ0(encoder 特征 + loss 监督)。若 Phase 0 选备选路线:heading_proj = zero-init `Linear(2→code_dim)` 加进 backbone cond-sum(`graph_pscf.py:18` `cond = timestep_emb + pooled_text` → 所有 DenseFiLM/AdaLN),`cond = t_emb + text_pooled + heading_emb`;heading 是几何条件,**不参与文本 CFG dropout**、CFG uncond 分支同样喂入;zero-init 保证起步 no-op。VQVAE 是重建、输入自含信息,无需条件。
- **yaw 增广 × 离线 latent cache(codex 意见 6)**:主推路线下"随机 yaw 增广 → 再 per-crop 重锚"是恒等操作,backbone 阶段 yaw 增广**无意义、不做**;VQVAE 阶段可选重锚后绕原点的小幅 yaw jitter 作数值稳健性(非必需)。备选路线下,离线 z_q token cache 无法在线增广——只能在 token 导出前按 K 个 yaw 副本展开(cache 体积 ×K)或放弃增广;不存在"离线 cache 上在线增广"的第三条路。此为 Phase 0 决策项 (e)。
- **SavGol smooth root(仅 A2 生效;A1 主推路线无平滑器)**:Savitzky-Golay(窗口 ≈1.5s 按各源 fps 换算、polyorder 3、`mode='interp'`),确定性、线性(与刚体 canonicalize 可交换)、边界原生多项式拟合无反射-padding 瞬态;短于窗口的 clip 行为须显式定义并有 CPU gate(G-cpu6);**Gate:每 clip max|smooth−raw|_XZ < 骨架尺度 15%。**

### 2.4 速度通道与 contact 的处理

- **速度:保留、降级为纯特征**。理由:kimodo 保留;能量塌缩史上 decode-loss 依赖速度监督;encoder 失去速度输入有 recon 时序锐度风险。恢复路径完全不用它(彻底切断积分依赖)。
- **contact:保持 BCE**(`vq_model/losses.py:110-111` 已是 masked_contact_bce,目标 raw 0/1),走归一化 bypass 掩码(§2.1 注),contact 通道往返 bit-exact。
- **root contact 的去留由语料审计决定(codex 意见 8)**:新布局把 root ch12:14 让给 heading,前提是"root 行 contact 在语料中无监督价值"。Phase 0 跑全语料 root-contact 审计(root ch12 非零率、与足端 contact 的相关性);审计支持 → 显式 retire root contact 并记录为设计决策;不支持 → root contact 保留、heading 另定通道位。不静默丢弃。

### 2.5 表示 schema 单一事实源 + 版本/哈希锁(codex 意见 7)

- 新增中心模块(如 `src/data/repr_schema.py`):单点定义 D、通道切片与语义(root/非 root 语义掩码、归一化 bypass 掩码、loss 路由掩码)、恢复方程版本;导出 `REPR_VERSION`(语义版本号)+ 内容哈希。所有消费点的切片/掩码从 schema 取,禁止各处 hardcode。
- version+hash 写入并在加载时校验:重编码数据目录 meta、per-rig moments、`_cond_normalized_*.pkl` 缓存、token 导出 manifest 与 `empirical_stats.pt`、VQVAE/backbone/evaluator checkpoint、eval splits/manifests。
- **不匹配一律硬 abort**(fail loud),无静默回退、无自动转换;老 13ch 资产与新表示资产靠版本号硬隔离,系统性防"新代码读旧缓存"级联(new_to_old_perm footgun 的同类防线)。schema 拒绝路径有 CPU gate(G-cpu5)。

---

## 3 多拓扑 heading 定义

### 3.1 候选对比

| 方案 | 一句话 | 判定 | 关键破绽 / 价值 |
|---|---|---|---|
| (a) root global rot6d 固定局部前向轴提取 yaw | 逐帧只读 ch3:9 | **BROKEN as-is** | 实测跨家族约定不齐:PZ rest 朝向 ≈ 世界 −X 且局部前向 = −z,TrueBones/human ≈ +Z 且前向 = +z → 直接错 90°~180°;前向轴 yaw 空翻过竖直点跳 π(kimodo 弃 facing-vector 的原因)。但加 rest 校准后是 (f) 的骨干 |
| (b) rest-pose 主轴/PCA + 当前帧旋转 | 几何主轴 | 不推荐 | 特征向量无符号(头/尾 180° 二义);直立人 / 立式鱼(Pirrana,rest |XZ|=0.27 vs 高 1.35)主轴竖直对 yaw 无用;近各向同性 rig(Crab)PCA 不稳;仅可作 (f) 校准第三级候选 |
| (c) 运动方向速度 EMA | 速度定义朝向 | **排除出表示,仅留作 QA** | 静止未定义;后退走 / 蟹式侧移语义直接反向;EMA 滞后;heading 不再是 pose 的函数 → canonicalize 不可逆。唯一价值:离线一致性探针,验证 φ_skel 有没有差 π |
| (d) per-拓扑锚点关节对(髋向量泛化) | L/R 对叉 world-up | 可行但**不作运行时方案** | 379/382 骨架有可名字解析 L/R 对,但 Anaconda/Alligator/SabreToothTiger 无对必须 fallback;对的选择有陷阱(Bird 翅膀扇动振荡);逐帧需 FK 且生成侧与 pose 可能自相矛盾。其"侧向轴抗空翻"性质以代数形式继承进 (f);锚点对降级为离线零点校准工具 |
| (e) 混合(有腿肢对 / 无腿 root yaw)逐帧提取 | 两条运行时路径 | 工程最差 | 分支边界任意(KingCobra hood 对算不算肢?),两提取器须在分界处语义一致否则断裂;每新增拓扑重走判定。混合思想的正确位置在离线校准,不在运行时 |
| **(f) rest-校准 root-delta twist-about-up** | **θ(t) = φ_skel + twistY(q_root(t)·q_rest⁻¹)** | **推荐** | 见下 |

### 3.2 推荐方案 (f) 详述

- **运行时(全 179 拓扑同一条代码路径)**:`θ(t) = φ_skel + twistY(q_root(t) · q_rest_root⁻¹)`,其中 `twistY(q) = 2·atan2(q_y, q_w)`,q_rest 取 cond.npy 里 `tpos_first_frame` 的 ch3:9;存 `[cosθ, sinθ]` 于 root token 通道。
- **代数性质(kimodo 髋向量的代数推广)**:纯 pitch 旋转 `q=[cos, sinθ/2, 0, 0]` 的 q_y=0 → twistY≡0;静止 / 倒立均良定义;不需要任何肢体锚点参与逐帧计算(蛇 / 鱼 / 鸟统一)。
- **奇异点策略(r1 修订,codex 意见 3:取代首稿的"连续性论证 / 测度零容忍")**:`twistY = 2·atan2(q_y, q_w)` 在 `s(t) = q_w²+q_y² → 0`(水平轴 π 旋转邻域)数值未定义/放大。显式策略五件套:
  1. **ε 阈值**:`s(t) < ε` 判退化(ε 初值 1e-4,由第 4 条语料审计定标);
  2. **fallback 状态**:退化帧 heading = hold-last-valid(clip 首帧即退化则取 φ_skel 的 rest 朝向),并在重编码 meta 写 per-帧 `heading_valid` 标记(不占通道);
  3. **时间跳变上限**:构建时断言逐帧 |Δθ| ≤ θ_max(初值 30°/帧 @20fps,按语料 p99.9 定标);超限记录 clip/帧号进人工复核清单,**不静默 clamp**;
  4. **全语料 s(t) 审计(Phase 1)**:扫全部 clip 全帧的 `q_w²+q_y²` 最小值分布(per-clip min + 全局直方图),量化真实退化暴露面;
  5. **合成穿越用例(CPU 测试)**:合成纯 roll、纯 pitch、组合 inversion 穿越序列,验证穿越点行为符合上述定义(纯 pitch 全程不跳 π、fallback 触发正确、跳变断言生效)。
- **canonicalize 兼容**:heading 是 root rot6d 的函数 → 整段刚体 yaw 预旋转 Δ 使所有帧 θ(t)→θ(t)+Δ,`[cos,sin]` 通道按 2×2 旋转矩阵闭式变换、完全可逆;first_heading = θ(0) 可作显式条件(仅 §2.3 备选路线需要;主推 per-crop 重锚路线下恒为 0、无需条件)——与 kimodo 配方逐条对齐。
- **拓扑差异全部压进离线常量 φ_skel——per-rig 常量,非 per-canonical-topology 常量(r1 修订,codex 意见 4)**:φ_skel 按 **rig(cond.npy 条目)**逐一校准,以 rig 标识 + cond 条目内容哈希锚定进 φ_skel 表;同一 canonical topology(179 量级)下不同 rig 的 rest 朝向可不同,**禁止**按 canonical topology 共享常量。校准输入一律取**运行时真实 rig 元数据**:dataset 实际加载路径下的 `tpos_first_frame(J,13)/parents/offsets/joints_names`(即经 new_to_old_perm 处理后的 dataset joint order),不得从磁盘原始顺序或第三方清单另取一份。层级:
  1. 优先侧向 L/R 锚点对(名字 regex 解析;输入 B 实测 379/382 可解析、含 KingCobra hood 对与 Pirrana 胸鳍对——数字待 Phase 1 校准脚本复核;取 graph 距 root 最近的对;left−right 叉 world-up 定符号);
  2. 纯链骨架(Anaconda 等)退回 root→head 脊柱轴 XZ 投影;
  3. 怪例(Crab 侧行语义、Pirrana 立式 rest)手工覆盖表;
  4. **全部 rig 的 rest-pose 朝向箭头 contact-sheet 一次性视觉 QA(专抓 π 翻转),发 user 审。**
- r1 曾把"LLM2Vec joint_semantics 通路兜底"列入第 1 级——该通路作为模型侧 per-joint 语义嵌入基建**存在**(`anytop_dataset.py:534,548-552`,builder `scripts/_build_joint_semantic_embeddings.py`),但它不是现成的 L/R 配对解析器,拿来当校准兜底属未落实的引申,**已从校准方案移除**(核对说明见 §6.11)。
- 基建:cond.npy 每 rig 已存 `tpos_first_frame(J,13)/parents/offsets/joints_names` → 离线校准零新增数据依赖。

> **冲突显式化**:输入 C 的通道布局草案中 heading 用"root rot6d 前向轴 XZ 投影 atan2 + unwrap、退化帧 hold-last"——这本质是方案 (a),已被输入 B 的实测证伪(跨家族 rest 朝向差 ~90°、局部前向轴符号相反)。**以输入 B(专项 heading 调研、带实测)为准,采 (f);输入 C 的 heading 草案作废**,其余通道布局结论不受影响。

---

## 4 迁移影响面

以下按子系统分组(来自只读依赖面审计;行号为审计时快照)。注意:该审计的 scope 是"方案 D:13ch 绝对 root",**方案 A(D=14)在其上追加**通道扩维、heading 校准、per-crop 重锚、schema 等项,故下列工作面对方案 A 是**下界**。**r1 的全部行数(LOC)估计无生成依据,已按 codex 意见 12 移除**,改用档位:小 = 十行级、中 = 数十行级、新脚本 = 独立文件;保留数字的来源清单见 §6.12。

### 4.1 数据侧

| 位置 | 改动 |
|---|---|
| **新增** `src/data/repr_schema.py`(§2.5) | 新模块:通道 schema + 掩码 + version/hash;各消费点接校验(中) |
| `src/data/anytop_dataset.py:5-14` 通道契约 docstring | 重写(小) |
| `anytop_dataset.py:307-370 _recover_world_positions` | root 步骤换直读绝对量,非 root 路径按 §2.1 方程调整(小-中) |
| `anytop_dataset.py:1174-1190` 归一化 | z-score 接 schema bypass 掩码;`:1186/1188` nan_to_num 换携带 clip/joint/channel 标识的有限性断言;**所有 per-rig mean/std 在重编码数据 + 重锚 crop 分布上、train-only 重建**,`moment_source`(`:533,547,1174-1181`)语义同步更新(中) |
| `anytop_dataset.py:1207-1235` 随机时间 crop | per-crop 重锚(平移 + yaw + heading 重算,§2.3 推荐路线)(中) |
| `_cond_normalized_J144.pkl` 缓存 | 重建 + 写入 schema version/hash;注意 new_to_old_perm footgun(转换器必须在 dataset joint order 下运行或显式存 perm) |
| `scripts/convert_humanml3d_to_anytop13.py:295-328/343-356/546-550` | 转换器写绝对量 + finalize_std 分组平滑重审 + TRAIN mean/std 重跑(中) |
| `scripts/_v4_build_from_272.py` | 继承转换器改动 + 全量 human 重生成(小 + 重跑) |
| 动物/TrueBones 语料(~90k clips / 179 拓扑——输入 A 口径,Phase 0 复核;cond.npy 由外部 planetzoo-anytop-pipeline 产出) | **最省路径:新写离线重编码脚本**(现 13ch → 新表示:逐 clip 跑今天的确定性恢复得世界坐标,再按新布局写回;重算 per-rig mean/std;带 Gate 式 RIC/FK 自检),新脚本,**不动外部编码器** |

### 4.2 恢复 / FK(4 处并行实现 + 外部第 5 处,必须同改)

| 实现 | 位置 | 量级 |
|---|---|---|
| numpy RIC | `anytop_dataset.py:341-349` | 小 |
| torch RIC | `graph_salad/world_recovery.py:72-81` | 小 |
| numpy FK root | `anytop_rot6d_fk.py:118-121` | 小 |
| torch FK root | `graph_salad/rot6d_fk_recovery.py:51-56` | 小 |
| **外部第 5 处**:PZ skinning `build_planetzoo_anytop_npy_skinning_poc.py:315-327 recover_root_positions`(自带 verbatim cumsum) | **首选:迁移 PZ 管线直读绝对 root(r1 修订,codex 意见 2)**。shim 仅作短期验证桥,且约束为:shim 速度**只从预测的绝对 root 差分导出**(按 legacy 约定旋回局部系写 ch9/11,绝不透传模型预测的速度通道),并附**shim 往返等价 gate**——shim 导出 → legacy cumsum 恢复 == 预测绝对 root,逐帧浮点容差内成立,证明整条链与预测速度无关 | 小(两侧各) |
| `vq_model/utils.py:29-90 root_drift_jitter_qa` | 代码不变,docstring 过时;该指标恰好成为迁移的 payoff 验证器 | ≈0 |
| 诊断脚本 verbatim cumsum 拷贝 ×10(`_render_rot6d_official_fk.py:138` 等) | 更新或标 stale | 可选 |

### 4.3 Loss(几乎零代码——全部走共享恢复函数)

- `graph_salad/losses.py:558-566` 与 `vq_model/losses.py:104-111` 分组 L1 切片(pos 0:3 / rot 3:9 / vel 9:12 / contact 12):方案 D 下零改;方案 A 下切片改从 §2.5 schema 取,并加 **root/非 root 语义掩码路由(codex 意见 8)**——heading 项只作用于 root ch12:14,contact BCE 只作用于非 root ch12,非 root ch13 padding 恒零 loss、零梯度;路由正确性有 CPU gate(G-cpu4)。(分块 reduce 已是现状,kimodo 防高维块霸权天然满足。)
- `compute_world_geometry_terms` / `compute_world_rot6d_fk_terms` / `train_denoiser.py:960-996` 与 `train_graph_codeflow.py:1001-1048` 的 dec-loss 分支:代码不变;**root 通道梯度从 cumsum 中介变为直接**(这正是改动的目的);"cumsum precision" 注释更新。
- `vq_model/losses.py:57-60` 默认权重(traj 0.10 是对着漂移调的):可选重调,无代码。

### 4.4 渲染 / 导出

- `animate_graph_codeflow.py` / `animate_denoiser.py` / `animate_vqvae_recon.py` / `animate_anytop13.py`:共享恢复函数更新后 **0 行**;`make_t2m_large_gif` root-centering 作用在世界坐标上,不受影响。
- 导出 .npy 语义改变 → PZ 侧必须匹配(首选迁移外部 recover_root_positions;短期 shim 须带往返等价 gate,见 §4.2)。
- `expand_minipack_motion_to_full_rig.py:127-128`:review only(小或零改)。

### 4.5 Eval(12ch contact-free evaluator)

- `t2m_evaluator.py:293-311/402-409`:切 `anytop_x[...,:12]`,**必须重训**——它消费语义被改的 root 通道,否则 R-precision/FID 空间失效。过往一次 ~24h on 4×H100(来源见 §6.12)。方案 A 下切片上下界从 §2.5 schema 取(contact/heading 通道位置不 hardcode),并做 schema version 校验。
- `train_anytop_t2m_evaluator.py` / `anytop_t2m_eval_dataset.py` / `build_anytop_t2m_eval_splits.py`:重跑;`codeflow_gen_eval.py:210-227` 布局位无改。

### 4.6 零代码但必须重训的级联

encoder 13-d 投影(`encoder.py:267,345-351`)、tokenizer 13-d 双头(`graph_vq_tokenizer.py:234-236,361-363`)在方案 D 下零改(方案 A 下 13→14 共 **4 处**:encoder motion_feat_dim + 双 Linear 入维、anytop13_head 双 Linear 出维、dataset shape 链 + cond 缓存重建、losses 索引);`batch.py`、CodeFlow_Model(z_q 空间,root-agnostic)零改。但完整重训链全部重跑:**VQVAE → token 导出(export + merge)→ empirical_stats.pt → backbone → evaluator**。C96=EdgeSegmentPool 槽数,与通道数 D 完全正交,Pool/Graphormer/RVQ 零改。

### 4.7 工作量粗估

| 项 | 量级(档位;r1 行数估计按意见 12 移除,以实际 diff 为准) |
|---|---|
| 核心代码(方案 D scope) | 约 8 个核心文件的小-中改 |
| 新离线重编码 + 统计 + gate 脚本 | 新脚本 |
| 诊断脚本更新(可选) | 小,批量 |
| 方案 A 追加 | 4 处扩维 + schema 模块 + per-crop 重锚 + heading 校准脚本 +(仅备选条件路线)heading plumbing(§4.8) |
| 数据 | 全语料重编码 + 全部 cond.npy mean/std(train-only、重锚 crop 分布)+ splits/manifests 重建(全部带 schema version/hash) |
| 计算(**主要成本**) | VQVAE 重训 + token 重导出 + backbone 重训 + evaluator 重训(evaluator ~24h/4×H100 为过往实跑参考,来源见 §6.12) |

### 4.8 heading 全链路 plumbing(codex 意见 6)

heading 作为**通道**(两条路线都有)与作为**条件**(仅 §2.3 备选路线)的触点全列:

- **通道维(两条路线都要)**:D=13→14 的 shape 链贯通 token 导出(`export_graph_vq_tokens.py`)→ merge(`merge_export_shards.py`)→ `empirical_stats.pt`(维度随 z_q,不直接含 heading,但 schema version 须写入)→ `CodeFlow_Model/token_dataset.py`/collate → 渲染/导出(`animate_graph_codeflow.py` 等)→ eval(`codeflow_gen_eval.py`、evaluator 切片)。所有通道位从 §2.5 schema 取,不 hardcode。
- **条件维(仅备选路线启用)**:token 导出时 per-crop first_heading(+ crop 起点平移)写入 token manifest → token_dataset/collate 携带成 batch 字段 → `graph_pscf.py` cond-sum 注入 heading_proj → **CFG:heading 为几何条件,不随文本 dropout,uncond 分支同样喂入**(`flow.py` 采样的双分支都传)→ sampler 入口签名加 heading → `animate_graph_codeflow.py` / `codeflow_gen_eval.py` 调用点传入(采样时条件分布须显式定义)。
- **yaw 增广 × 离线 cache**:主推路线 backbone 无增广(§2.3);备选路线只能在导出前按 K 个 yaw 副本展开 cache 或放弃增广——不存在"离线 cache 上在线增广"的第三条路。
- 主推路线把条件维整条省掉——这是 §2.3 推荐 per-crop 重锚的工程论据之一。

---

## 5 分阶段实施计划

每阶段带 verify gate;G 编号 gate 清单来自通道布局调研,穿插于各阶段。全程铁律不变:代码 diff 必经 codex 审(G11)、不 self-submit/cancel Slurm、QA 可视化发 user 审。

### Phase 0 — 设计决策钉死 + 前提审计(零训练成本)
1. 验证 root ch2 在动物语料上确实 dead(human 侧已知写 0)——通道占位的前提。
2. `tpos_first_frame` 批量验证确为规范 T-pose(已抽 12 家族,余量待扫)。
3. **root-contact 全语料审计(codex 意见 8)**:root ch12 非零率 + 与足端 contact 相关性,决定 retire 还是保留 root contact(§2.4)。
4. **语料口径复核(codex 意见 12)**:clips / 拓扑 / rig 数(~90k / 179 / 382 为输入调研口径),脚本输出实际数并回写本文档。
5. 钉死决策:(a) 方案 A1 vs A2 vs D(scope 档位);(b) crop 语义 = per-crop 重锚(推荐)vs 显式条件(§2.3);(c) 速度通道保留为特征(推荐)vs 方案 C;(d) PZ 导出契约 = 迁移外部管线(首选)vs 短期 shim + 往返等价 gate(§4.2);(e) 备选路线下的 yaw 增广 × cache 策略(§2.3)。
- **Verify**:决策记录进本文档修订 + user 拍板。

### Phase 1 — heading 离线校准 + 奇异点审计(仅方案 A;零训练成本)
1. 校准脚本扫全部 cond 条目(382 量级,脚本输出实际数)产 per-rig φ_skel 表(锚点对 → 脊柱轴 fallback → 手工覆盖表;输入 = 运行时真实 rig 元数据,§3.2)。
2. 方案 (c) 作离线探针:前进类 clip 上校准后 heading 与移动方向应高相关(抓差 π)。
3. **全语料 `q_w²+q_y²` 奇异点审计 + ε/θ_max 定标 + 合成 roll/pitch/inversion 穿越用例**(§3.2 奇异点策略第 1/3/4/5 条,codex 意见 3)。
- **Verify gate**:全部 rig rest-pose 朝向箭头 contact-sheet 视觉 QA,发 user 审(G3 前置);奇异点审计报告(per-clip min s(t) 直方图 + 跳变超限清单)。

### Phase 2 — 数据重导出
1. 新离线重编码脚本(动物语料)+ 转换器改动重跑(human/v4 链)+ 全部 mean/std、splits、manifests 重建。
- **Verify gates(有序)**:
  - G1 转换器往返:世界坐标→新表示→世界坐标,179 拓扑全采样,max abs err < 1e-4×bbox(无积分,可逼恒等);
  - G2 rot-FK 一致:局部 rot6d FK 位置 vs 存储位置 ≤ v4b gt_fk_mismatch 基线(mean 0.46%,来源见 §6.12);
  - G3 heading 重算一致 <1°(退化帧豁免 + frame-0 投影范数>ε 断言);
  - G4 速度 = 解码位置差分 ≈0;
  - G5 canonicalize 可逆:de-canonicalize(θ0 + frame-0 偏移)== 原始世界 <1e-4;
  - G6 yaw 等变性:rep(rotate(x,φ)) == rotate_rep(rep(x),φ) <1e-4;
  - G7 训练前 GT 可视化 QA(**全部在任何 GPU 训练之前完成,codex 意见 11**):(i) GT 往返多帧 GIF、GT-vs-roundtrip 并排,必含高 yaw 历史 bug-catcher clip(CircleFly 720°/turn_180 396°);(ii) **非 root-centered 世界系轨迹可视化**(俯视 XZ 轨迹 + 世界系动画——专抓 root-centering QA 抹掉的漂移类缺陷,§1.2);(iii) **GT-过新表示的蒙皮导出 QA**(PZ 链路按 Phase 0 决策的契约);全部**发 user 审**;
  - G8 contact 通道往返 bit-exact;G9 moments builder [J,14] train-only + bypass 掩码通道无矩断言 + std floor + schema version 写入;G10 cond 缓存重建核对 new_to_old_perm;
  - **G-cpu CPU gate 套件(codex 意见 10,全部无 GPU、进 CI 级脚本)**:
    - G-cpu1 时序脉冲响应:扰动单帧 root 通道 → 恢复误差只落在该帧,后续帧零变化(反 cumsum 判据);
    - G-cpu2 速度零梯度:autograd 验证 ∂(恢复位置)/∂(ch9:12) ≡ 0;
    - G-cpu3 D=14 完整前向/反向:encoder→tokenizer→losses CPU 全链跑通;
    - G-cpu4 contact/heading 梯度路由:contact loss 梯度只落非 root ch12、heading 项只落 root ch12:14、非 root ch13 恒零梯度;
    - G-cpu5 schema 拒绝:version/hash 不匹配的数据 / moments / cache / ckpt 必 abort;
    - G-cpu6 短 clip SavGol 行为(仅 A2):clip 短于窗口时行为符合显式定义;
    - G-cpu7 per-topology 尾部误差:往返误差按拓扑分组报 max/p99(不许只报全局均值),逐拓扑过阈。

### Phase 3 — VQVAE 重训
1. 扩维改动(方案 A 的 4 处)+ 恢复函数改动合入;G11 codex 审。
2. G12 smoke 用一次性缓存副本(empirical-clobber 教训:smoke 的 EMPIRICAL_MAX 会覆写共享 empirical_stats.pt)。
- **Verify gates**:
  - G13 recon 协议复跑:textR text→recon R-prec vs GT ceiling、FK jitter ~1.0×、MPJPE ≤ v4b 基线(~20mm,来源见 §6.12);
  - **G14 决定性收益 gate:长程 locomotion clip 世界轨迹终点误差**——旧表示积分漂移的弱点,新表示应 ≈0,这是本次迁移的 payoff 指标(root_drift_jitter_qa 即验证器);
  - 蒙皮导出链路(PZ 契约按 Phase 0 决策)recon 级世界系可视化,发 user 审(GT 级蒙皮与世界轨迹可视化已在 G7 于任何 GPU 训练前完成,codex 意见 11)。

### Phase 4 — token 重导出 + backbone 重训
1. export_graph_vq_tokens + merge shards + empirical_stats.pt **预暖**(NCCL 冷扫描 ~40min > 10min PG 超时的坑;冷缓存首启会 SIGABRT)。
2. 仅 §2.3 备选条件路线:first_heading heading_proj(zero-init)注入 graph_pscf cond-sum + §4.8 条件维 plumbing 全链;主推 per-crop 重锚路线**无此步、亦无 backbone yaw 增广**(§2.3);codex 审。
- **Verify**:smoke(一次性缓存副本)→ 真跑;gen-eval + 生成侧 heading↔rot6d 一致性检查(§6 风险 2)。

### Phase 5 — evaluator 重训 + eval 全线复跑
1. evaluator 在新表示数据上重训(~24h/4×H100 参考);eval splits/manifests 重建。
- **Verify**:gen-eval / recon-textR 在新 evaluator 空间复跑,与 v4b 基线并列报告。
- **G15 回滚保障(贯穿全程)**:v4b 数据与 ckpt 保留可回滚(沿用 KEEP-v1 先例),新表示数据落新目录,不覆写。

---

## 6 风险与开放问题

1. **heading 与 rot6d 冗余不一致(生成侧)**:heading 是 root rot6d 的函数,但生成时两者是独立通道,可能自相矛盾。对策:consistency loss,或 decode 侧只信 rot6d(heading 仅作条件/特征)。未定,需在 Phase 4 前拍板。
2. **twist-about-up 奇异点**:已由 §3.2 显式策略覆盖(ε 阈值 + hold-last fallback + |Δθ| 跳变上限 + 全语料 s(t) 审计 + 合成穿越用例;r1 的"测度零、可达最优"论证按 codex 意见 3 作废)。剩余风险:ε/θ_max 定标依赖 Phase 1 审计的真实分布;复合 roll+pitch 杂技下"面向"本体论含混,fallback 只保证有定义、不保证语义完美。
3. **φ_skel 差 π 的跨骨架文本对齐翻转**:contact-sheet QA(Phase 1)+ 运动方向探针专抓此项;怪例依赖手工覆盖表的完备性。
4. **非 root ch0:3 参考点与参考系同时改**(root-局部系 RIC → 世界轴对齐减 raw-root-XZ):方向方差靠 per-crop 重锚(θ0=0)+ heading 通道特征承载——这是方案 A 相对 v4b 最大的分布变化,G7 往返可视化与 Phase 3 recon 对照是主要防线。
5. **ill-conditioned target 复发风险(仅备选 B)**:全局旋转重 derive 需重新钉死 twist 约定;历史教训(human rot6d jitter 15×→1.18× 靠编码修而非 loss 修)表明必须先小规模 jitter 验证再全量。
6. **前提验证未完成**:动物语料 root ch2 deadness、`tpos_first_frame` 全量 T-pose 扫描(Phase 0 项)。
7. **crop 语义**:r1 的 build-time 锚定方案已作废(§2.3,codex 意见 5);现推荐 per-crop 重锚。剩余风险:重锚后 crop 分布上的 moments 定义与实现(采样多少 crop 估计 mean/std、train/val 隔离)需在重编码/统计脚本中钉死并过 G9。
8. **主要成本是重训级联而非代码**:VQVAE → token → backbone → evaluator 全链重跑;GPU 排期与是否/何时启动属主要资源决策,**由 user 拍板**(本方案未排期)。
9. **PZ 导出契约(r1 修订,codex 意见 2)**:首选**迁移外部 PZ 管线**直读绝对 root;shim 仅作短期验证桥,且必须满足:shim 速度只从预测绝对 root 差分导出(不透传模型速度通道)+ shim 往返等价 gate(shim 导出 → legacy cumsum 恢复 == 预测绝对 root)。时序(直接迁移 vs 先 shim 后迁移)供 user 定。
10. **方案 C(无速度)的可行性**:需 A/C 小规模 recon 对照数据支撑,当前无。
11. **对 codex 意见 4 中"LLM2Vec fallback 不存在"的核对说明(部分不同意其表述,同意并落实其实质)**:LLM2Vec joint_semantics 通路在 repo 中**存在**——`anytop_dataset.py:534`(参数)与 `:548-552`(per-joint 语义嵌入加载,含 joint-order hash 校验),builder 为 `scripts/_build_joint_semantic_embeddings.py`。codex "nonexistent" 的字面表述不准确;但其实质成立:该通路是模型侧输入基建,不是 L/R 配对解析器,r1 把它列为 φ_skel 校准兜底属未落实的引申。已按实质落实:从校准方案移除该兜底,校准仅依赖名字 regex + 脊柱轴 fallback + 手工覆盖表 + 视觉 QA(§3.2)。
12. **数字来源清单(codex 意见 12)**:r1 的全部 LOC 估计(120-180 / 200-300 / 50-80 / 6-12 行等)无生成依据,已从全文移除,改档位表述。保留数字及来源:
    - v4b jitter ~1.0× / FK-floor 0 / MPJPE ~20mm / text→recon R@1 0.992:v4b 验收 eval 实跑(`scripts/_eval_vqvae_recon_textR.py` 协议,两轮 codex-PASS;记录于项目记忆 v4b 条目 `project_v4b_272_smpl_rest_construction`);
    - gt_fk_mismatch mean 0.46% / p95 2.31%:HumanML3D→AnyTop13 转换 Gate 实测(项目记忆 `project_humanml3d_to_anytop13_conversion`);
    - evaluator ~24h / 4×H100:v4b T2M evaluator 实跑 walltime(项目记忆 v4b 条目);
    - NCCL 冷扫描 ~40min > PG 10min 超时:merged backbone 首启事故实录(项目记忆 `project_backbone_merged_coldscan_nccl_footgun`);
    - ~90k clips / 179 拓扑 / 382 rig / 379 可解析 L/R 对 / 12 家族抽检 / Pirrana rest |XZ|=0.27:输入 A/B 调研口径,**未在本轮独立复核**——Phase 0 条目 4 与 Phase 1 校准脚本输出实际数并回写本文档;
    - CircleFly 720° / turn_180 396°:历史 bug-catcher clip 的实测 yaw 行程(既有渲染 QA 记录)。

---

## 7 与当前实验线的关系

- **v4 decode-loss 是正交的短期修**:decoded-x0 几何/速度 loss 解决的是能量塌缩/conditioning 问题(已确认有效:pose 能量比偏离 −41%,无需推理时喂速度),作用在"模型如何被监督";本方案是**长期表示修**,解决的是"表示本身把误差积分放大 + heading 无跨拓扑定义",作用在"数据编码是什么"。两者机制不同、不冲突。
- **两者可叠加**:新表示下 decode-loss 依然适用,且更好——traj/世界几何项对 root 通道的梯度从 cumsum 中介变为直接(§4.3),decode-loss 的监督信号质量随表示迁移而提升。
- **对在跑实验零影响**:本方案处于"方案制定、未排期"状态,不触碰当前训练线(含 v2 holdout LLM2Vec backbone)与 v4b 数据/ckpt;G15 保证 v4b 全程可回滚。启动时点、GPU 预算、方案 A vs D 档位选择,均待 user 决策。

---

*来源:输入 A(root-encoding 依赖面只读审计)、输入 B(多拓扑 heading 设计调研)、输入 C(kimodo 通道布局设计);repo 现状 /iridisfs/scratch/ts1v23/workspace/noKslot_clean(+ 外部 planetzoo-anytop-pipeline)。2026-08-11。*
