# BTJD-K 数据标准 v3.0 —— 多拓扑双流全局化动作表示

> **审查状态:v2.3 经 codex clean-context 敌对终审 r4 判定 `--NEEDS-FIX`(5 项 BLOCKING + 8 项 MAJOR + 2 项 MINOR)。本版 v3.0 是针对这 13+2 条意见的修订版。** 落实逐条对照见 **§11**。
> 修订历史:r1(3 项 blocking 数学错误 + 12 条表述矛盾)、r2(7 项规格完备性 + 2 处 stale 引用)、r3(7 项可执行性)已在 v2.1/v2.2/v2.3 落实;**r4 首次触及设计错误层面**(heading 几何方向反了、旋转来源契约缺失、6d 行列约定未冻结、骨架未入契约、contact 无损承诺不成立),故 v3.0 相对 v2.3 **含语义变化**,不是纯规格补全。
> 事实源:Kimodo 技术报告 §4.1 verbatim + 官方实现(commit 1aece8c)+ 13ch 编码代码权威表 + Phase01 全语料审计(102,438 clips / 382 rigs / 194 canonical topologies,`scratch/btjd_phase01/*.json`)+ 2026-08-12 源侧逐 clip 实测(PZ 74,522 BVH / TrueBones 1,070 BVH / human 26,846 × 272-dim)。
> **采用本标准 = 全部 v1 工件(数据/moments/VQVAE/token cache/backbone/evaluator)不兼容,schema 硬断,全链重建。** `repr_schema_version = "btjdk-v3.0"`。

---

## ⚠⚠ 文档状态(2026-08-12,user 决定本轮不换数据表示,本文档转入存档待用)

**启用本标准前必须先补三件事**,否则不要照着实施:

1. **文档里的源侧"实测"数字尚未落盘,不可复核。** 引自 `scratch/btjd_phase01/*.json` 的 Phase01 审计数字是可信的(382 rig / 194 topology / 102,438 clip / 9,985,438 帧 / root=exactly-yaw-only / root_nnz_frames=3687 等);但 2026-08-12 这一轮新增的源侧测量——PZ/TB 的 root tilt 分布、G-motion-rot 的 FK 一致性、路径 B 的叶关节占比与 twist 损失、素材覆盖率与解析耗时、6d 往返浮点余量、300-clip 归一化代理统计——**跑它们的脚本与输出都没有留档**,磁盘上找不到对应工件。这与本标准自己要求数据必须带 `source_sha256`/manifest 的纪律相违。启用前须重跑并落盘到 `scratch/btjd_evidence_<日期>/`。
2. **一处已知数字错误**:§「撤销 3」把 Anaconda 的 `n_frames`(2,686)与 KingCobra 的 `n_frames`(1,105)误当成 root-contact 帧数;`root_contact_audit_full.json` 里真实的 `root_nnz_frames` 是 **Anaconda 2,647 / KingCobra 1,040**(和为 3,687,与同句的总数自洽)。§8 引用的 `root_on_rate` 0.9855 / 0.9412 是对的。
3. **两处规格缺陷未修**:(a) §2.1.3 的水平性判据 `‖w_xz‖ ≥ 0.1‖w‖` 对 limb-pair 分支**恒真**(`w=(p_L−p_R)×e_y` 结构上 y 分量恒为 0),这个 gate 抓不到任何东西,且左右锚点近竖直分离时是 0/0;应改为对真实 forward `f(t)=D_root(t)·u_rig` 生效,并给 `‖w‖` 加下限守卫。(b) §2.2B(路径 B)与 §9.2 G-motion-rot.6 的「13ch 是非法输入」直接冲突,按现文路径 B 不可执行;选路径 B 时须同步放宽 §9.2 的对应条款。

已确证且不受上述影响的结论:**原始逐帧完整旋转三条链路都在磁盘上**(PZ 74,522 BVH 含 root 全 6 通道与 twist 关节 / TrueBones 原始 BVH / human 272-dim 真 SMPL 旋转 + AMASS 独立备份),所以路径 A 在素材层面可行,`j_a` 不必靠 IK 反推。

---

## ⚠ 顶部阻塞项:j_a 的数据来源方向待 user 拍板,Phase 2 实现在拍板前不启动

`j_a`(真全局 rest-delta 旋转)有两条互斥的数据来源路径,**本标准并列写全两条,不替 user 做决定**(§2.2A / §2.2B):

- **路径 A —— raw-source 重建**:从磁盘上的原始逐帧旋转(PZ/TB 的 BVH、human 的 272 向量)重建全部关节的真全局旋转。素材已确证 **102,438 / 102,438 clip 全覆盖,缺失 0**;源解析成本实测约 **0.8 CPU-hour 单核**。此路径下「信息无损」在 j_a 上**可以真正成立**。
- **路径 B —— legacy 13ch 派生**:只用现有 13ch。非叶关节旋转可精确重建(测地误差 ≤ 7.82e-6°),但**叶关节自身旋转在 13ch 中没有载体行,完全丢失**(占关节槽 **6,305 / 23,037 = 27.4%**,per-rig 7.4%–66.7%,且正是足端/手/头/尾尖)。此路径下标准**必须改名 `btjdk-v3.0-lossy` 并删除全部无损承诺**。

**拍板前不得启动 Phase 2 构建**:§9 的 G-motion-rot / G-skel-m / G1b / G1b2 全部依赖该决定;§2.1.1 的 heading 运行时公式在两条路径下的可用性也不同(见 §2.2B 末尾的交叉说明)。该决定记为 **§10 开放问题 6**。

---

## ⚠ 本版撤销的 v2.3 论据(诚实记录,必读)

### 撤销 1:「真 root 旋转翻滚 78°–177° tilt」—— 论据错误,已撤销

v2.3 在 §2.1 与 §2.2 两处用「raw-source 探针证明真 root rest-delta 在翻滚 clip 达 78°–177° tilt,非 yaw-only」论证「heading 不从 root 全局旋转推导」。**该论据是类别错误,予以撤销**:

- `scratch/btjd_phase01/corpus_singularity_audit_full.json` 的全语料结果是 —— legacy 13ch 的 **root 行 ch3:9 精确是 yaw-only**:`R[1,1] = 1`、y 行/列非对角元 `== 0.0` 到机器精度;`singularity.per_clip_min_s.min = 0.9999999999999994`;`eps_exposure_table` 中 eps 从 `1e-6` 到 `0.05` 的 `frames_below` **全部为 0**;即便最花的片段(Parrot CircleFly / SabreToothTiger 180Flip / Raptor2 RunJumpRoll)也如此。
- 78°–177° 的数字来自**子关节**行(child `|R11−1|` 最大 1.65),身体 pitch/roll 活在子关节旋转里。把子关节数据当作 root 数据引用,与 v2.1 时期的「r1 意见 a2」是同型错误(载体行混淆)。
- 该 JSON 的 `scope_caveat` 原文也明确「this scan cannot see raw-source tilt」——即它**没有**同级持久化证据支持 78°–177°。

**替换为真实证据(2026-08-12 源侧实测,`angle(R_glob(root)·e_y, e_y)` 的每-clip 最大值)**:PZ(25 clip 抽样)median **25.8°** / p90 **47.2°** / max **67.2°**;TrueBones(60 clip 抽样)median **58.1°** / p90 **168.7°** / max **178.3°**。结论方向不变(**源侧 root 确实不是 yaw-only**,设计意图成立),但数字、样本量与测量定义必须按上式书写,**不得再引用 78°–177°**。

### 撤销 2:heading 叉积方向与校准产物相反 —— 已修正

v2.3 §2.1 写 `v_fwd = cross(e_y, p_L − p_R)`。校准脚本 `scratch/btjd_phase01/_calibrate_phi_skel.py` 实际构造 `[−(z_L−z_R), (y_L−y_R), (x_L−x_R)]`,`phi_skel_calibration_full.json` 的 `convention.phi_skel` 原文为 `f = (posL-posR) x up`,AnyTop 官方为 `e_y × (p_R − p_L)` —— 三者互相一致,**与 v2.3 文档相差 π**。**v3.0 冻结 `w = (p_L − p_R) × e_y`**(§2.1.1(b)),v2.3 的写法作废。

### 撤销 3:「root contact 退役 = 可忽略损失(全语料 ~2 个非冗余帧)」—— 已撤销

全量审计实录 **3,687 帧 root-contact = 1、涉 30 clip**(Anaconda 2,686 帧 / KingCobra 1,105 帧)。「~2 个非冗余帧」是**语义冗余**估计(root 与其脊柱子关节 agreement 100.0% / 99.81%),**不等于位级可逆**。v3.0 保留 root contact(§2.2 ch12),该整句删除。

### 撤销 4:`phi_skel_deg` 382-rig 表的数值 —— 全表作废,须重算

现表定义式 `phi_skel = circmean_t[angle_XZ(f(t)) − twistY(q_root(t) ⊗ conj(q_rest))]` 的两个锚点均被本标准废弃:(i) `q_root` 取自 legacy yaw 载体 root 行,而新标准 root 行是真全局旋转;(ii) `q_rest` 取自 cond `tpos_first_frame` root 行,而 §2.2/G-rest 已判定其对 PZ rig 处于资产系不可直用(`calibration.tpos_frame_mismatch.pz_median_absdelta_deg = 87.48`,`pz_max_absdelta_deg = 179.86`)。**数值全部作废**;可继承的只有离散量(`method` / `anchor_left` / `anchor_right` / `spine_tip` / `pol`),继承须由 §9 G-heading-B 机检背书。

### 撤销 5:「104 rig 人工 review 清零」口径 —— 不充分,已替换

`review.reasons_histogram` 中**不含** `NO_MOVING_CLIP` 状态。逐行核对 `phi_skel_per_rig_full.tsv`:64 个 `NO_MOVING_CLIP` rig 中 **43 个 `needs_review == False`**,即旧口径下它们会**在零动态证据下静默通过**(含全部翼类 Bat / Bird / Buzzard / Eagle / Flamingo / Parrot / Pteranodon / Tukan)。v3.0 改为 §9 G-heading-A 的**覆盖性状态机**,现表实测下的人工清单为 **154 rig**(而非 104)。

---

## 1 设计原则(不变,kimodo 五条 rationale 的移植)

P1 全局优先(误差不积分、支持稀疏世界系约束)/ P2 不逐帧 heading 转正(翻滚不连续)/ P3 全局旋转(免链 FK、世界系约束)/ P4 平滑 root(foot-skate 消融证据)/ P5 平移-only canonicalize + 朝向随机化/条件。出处:Kimodo 技术报告 §4.1(p8/p10 原文)。

> **v3.0 对 P2 的澄清**:§2.1.1 改用「root 全局旋转作用于 rig 校准的 rest-forward 轴」求 heading,**不违反 P2** —— 我们仍然不做逐帧 canonicalize,只是把 heading 的**估计器**从「逐帧锚点位置几何」换成「体固连轴的方位角」。

## 2 表示结构:全局流 ⊕ 关节流

### 2.1 全局流 rglob:[B, T, 4](图外侧通道,对应我们架构的 frame/holder 流)

> **⚠ 维度由 v2.3 的 [B,T,5] 改为 [B,T,4]**(r4-MINOR-01):`r_s.y` 与 `j_p[root].y` 在 GT 中逐比特同源(§5 的 canonicalize 只减 xz,不动 y),而 §3 恢复只读 `j_p[root].y`,`r_s.y` **在恢复路径上完全不被消费**却在 §8 单独计一次 loss(等价于把 root 高度监督加倍),生成时二者还可不一致且无任何检查。**裁决:唯一权威 root 高度 = `j_p[root].y`,删除 `r_s.y` 冗余。** 与 kimodo 官方 rglob 维度不再逐位对齐 —— 移植的是「平滑 root 参考 + 逐关节残差」的**机制**而非维度,后续实现者不得按 kimodo 的 5 维复原。

| 分量 | 定义 |
|---|---|
| `r_s_xz ∈ R²` | **平滑 root 的绝对世界 xz**:物理 root 的 xz 经零相位低通(§5 S2)。canonicalize 后首帧位于原点。**不含 y 分量 —— root 高度唯一由 `j_p[root].y`(§2.2)承载,表示中不得存在第二份 root 高度。** |
| `h ∈ R²` | **显式 heading `[cos θ, sin θ]`**,由 §2.1.1 的「rest-forward 轴 ⊗ root rest-delta」唯一确定;退化与连续性见 §2.1.2,支持域见 §2.1.3 |

**全局流处理完备性(normative)——凡关节流有定义的处理,全局流同样有定义**:① canonicalize(§5 S5):`r_s_xz ← r_s_xz − q_xz`;`h` 平移不变故不变。② 整段刚体 yaw(§6):`r_s_xz′ = Y₂(φ)·r_s_xz`,`h′ = Rot2(φ)·h`。③ 归一化(§7):`r_s_xz` 参与 N1′ 尺度组;`h` **不做任何缩放**,decode 侧全域投影回 S¹。④ loss(§8):`r_s_xz`、`h` 各为独立分块。⑤ `frame_mask`(§4):padding 帧从全局流一切 loss/统计归约中剔除,与关节流同规则。

#### 2.1.1 heading 定义(normative,冻结)

**(a) 角度基(不变,与 §6 自洽)**:`θ = atan2(f_x, f_z)`;零参考轴 = **+Z**;绕 **+Y** 右手为正;`h = [cos θ, sin θ]`。

**(b) 叉积序冻结(修正 v2.2/v2.3 的方向错误,见「撤销 2」)**:由左右锚点求前向轴时,**normative 形式为 `w = (p_L − p_R) × e_y`**,展开 `w = (−(p_L−p_R)_z, 0, (p_L−p_R)_x)`。
语义推导核验(Y-up 右手系):角色面向 +Z、up = +Y 时,其右手方向 = `f × u = e_z × e_y = −e_x`,故左手方向 = `+e_x`;代入 `p_L − p_R = (+1,0,0)` 得 `w = (0,0,1) = +Z = 前向` ✓。该式与校准脚本、`phi_skel_calibration_full.json` 的 `convention.phi_skel`、AnyTop 官方 `e_y × (p_R − p_L)` 三者**完全一致**。v2.2/v2.3 的 `cross(e_y, p_L − p_R)` **作废**。

**(c) 运行时公式(唯一定义;消费校准表)**:每 rig 冻结一个 **rest-forward 单位轴 `u_rig ∈ R³`**,表达在 **G-rest 定义的真全局 rest 世界系**(与 `R_rest_global` 同系)。逐帧:

```
D_root(t) = 6d⁻¹(j_a[root, t]) = R_global(root,t) · R_rest_global(root)⁻¹   # root 行 rest-delta
f(t)      = D_root(t) · u_rig                    # ‖f(t)‖₂ ≡ 1(SO(3) 作用于单位向量)
n(t)      = hypot(f(t)_x, f(t)_z) ∈ [0,1]        # 无量纲"水平度" = |cos(前向俯仰角)|
θ(t)      = atan2(f(t)_x, f(t)_z)                # 退化与连续性见 §2.1.2
h(t)      = [cos θ(t), sin θ(t)]
```

**(c1) 设计反转声明(取代 v2.3「不从 root 全局旋转推导」一句)**:原有理由(78°–177° tilt)**无证据支撑,已撤销**(见顶部「撤销 1」)。改用本式的真实理由是 §2.1.2:**基于逐帧锚点位置的估计器存在物理滚转奇异性,本式没有**。
**关键澄清**:本式给出的 θ **不是**「root 旋转的 yaw」(在 tilt 下 yaw 分解本身病态),而是**体固连前向轴的 XZ 方位角**,对任何「前向像不竖直」的旋转都良定。
**§6 等变性核验**:yaw 增广下 `R_global′ = Y(φ)·R_global`、`R_rest` 不变 ⇒ `D_root′ = Y(φ)·D_root` ⇒ `f′ = Y(φ)·f` ⇒ `θ′ = θ + φ`、`n′ = n`,与 §6 的 `h′ = Rot2(φ)·h` 逐帧一致 ✓。

**(d) rest-forward 轴的确定(两段式,per rig)**

- **几何段(确定性,只用 G-rest 真全局 rest 姿态的关节位置,不用 motion)**,按 `method` 取 `w`:
  - `limb-pair`:`w = (P_rest(L) − P_rest(R)) × e_y`,(L,R) = 表中 `anchor_left` / `anchor_right`(存关节**名**,不存索引)。
  - `spine-axis`:`w = P_rest(tip) − P_rest(root)`;**tip 选择规则冻结**(与现校准脚本同):关节名小写含 `head` / `atama` 者中图距 root 最大者;若无,则在不含 `tail` / `sippo` / `shippo` 的非 root 关节中取图距 root 最大者;若仍无 → 按 §2.1.3 处理。
  - `explicit`:`w = (rest_forward_xz[0], 0, rest_forward_xz[1])`,由 rig 元数据给出(§2.1.3 S3)。
  - **rest 水平性判据(normative)**:要求 `‖(w_x, 0, w_z)‖₂ ≥ 0.1 · ‖w‖₂`;否则该 method 对该 rig 不可用(rest 前向轴近竖直,例如直立脊柱的人形**不得**用 spine-axis),按 §2.1.3 降级或 REJECT。
- **极性段(语义;来自 motion probe + 人工评审,§9 G-heading-A)**:`pol ∈ {+1, −1}`。
- **合成**:`u_rig = pol · (w_x, 0, w_z) / ‖(w_x, 0, w_z)‖₂`;表中记 `phi_rest_deg = degrees(atan2(u_rig_x, u_rig_z))`,**取代旧字段 `phi_skel_deg`**。
- **FLIPPED rig 的修正方式 = `pol = −1`**(等价于交换 `anchor_left` / `anchor_right`),**不得再写「phi + 180°」** —— 新定义下极性是 `u_rig` 的符号,不是角度偏移量。

**(e) 382-rig 表必须重算(硬性)**:理由见顶部「撤销 4」。**可继承的只有离散量** `method` / `anchor_left` / `anchor_right` / `spine_tip` / `pol` —— 它们只依赖关节命名与身体左右语义,不依赖 rest 帧,故 104 + 64 rig 的人工判定工作不作废;继承必须由 §9 G-heading-B 机检背书,不得口头继承。极性 probe 须在新数据上按新定义**重跑一次**(全语料一次角度扫描;现表生成 `runtime_sec = 52.7`,成本可忽略),旧 `probe_verdict` 仅作为工作清单先验。

**(f) `heading_table_hash` 的决定性(修正「哈希无法唯一决定 h」)**:h 由且仅由〔`u_rig`、`pol`、`method`、`anchor_*` / `spine_tip`、(a) 的角度基、`eps_h` 与 §2.1.2 连续性规则版本号、`j_a[root]`〕决定,而 `j_a[root]` 又依赖 `R_rest_global`。因此:

- `heading_table` 每条记录必含:`rig`、`method`、`anchor_left`、`anchor_right`、`spine_tip`、`u_rig`(float64 三元组)、`pol`、`phi_rest_deg`、`eps_h`、`continuity_rule_version`,以及其所依据的 **`skeleton_table_hash`**;
- `heading_table_hash` 按 §4.2 冻结的 canonical envelope 覆盖上述全部字段(外加 §9 G-heading-A(d) 的评审字段);
- 构建/加载时校验 `heading_table.skeleton_table_hash == manifest.skeleton_table_hash`,不等硬 abort。

#### 2.1.2 heading 退化判据与连续性规则(normative)

**(a) 旧估计器的物理奇异性(作废理由)**:v2.3 的逐帧锚点估计器 `w(t) = (p_L(t) − p_R(t)) × e_y` 在身体绕**前向轴滚转**时退化:滚转角 β 下 `‖w_xz‖ ∝ |cos β|`,β = 90° 时 `‖w_xz‖ = 6.12e-17`(数值实测),β 越过 90° 后 w 反号、θ 阶跃 180°。`[cos θ, sin θ]` 只消除角度在 ±π 处的**参数化**断点,**不消除该物理奇异**;`hold-last` 只把 π 跳变推迟到退化区间结束。此外其 ε 还是**有量纲**量(位置差的叉积,单位为长度²,却与 §7 的无量纲 `ε = 1e-6` 共用),跨 rig 尺度不可比。**全部作废。**

**(b) 新定义的退化集合(严格更小且与滚转无关)**:按 §2.1.1(c),`‖f(t)‖₂ ≡ 1`,故 `n(t) = hypot(f_x, f_z) = |cos(前向俯仰角)|` 是**无量纲**量;退化当且仅当**体固连前向轴接近竖直**(直立仰起、俯冲、垂直攀爬),与绕前向轴的滚转**完全无关**。

**(c) 阈值与状态机(冻结)**

```
ε_h = 0.05                       # 无量纲;= sin(2.87°),即前向轴距竖直 2.87° 以内判退化【待定,见 §10】
deg(t)   = [ n(t) < ε_h ]
θ_raw(t) = atan2(f(t)_x, f(t)_z)                 # 仅在 deg(t)=False 时有定义
t0 = min{ t : deg(t) = False }
若 t0 不存在(整段退化) → 该 clip REJECT(fail-closed,写入 heading_degenerate 名单;
                                        禁止静默回退到 h=[1,0])
θ(t) = θ_raw(t)      若 deg(t) = False
     = θ(t−1)        若 deg(t) = True 且 t > t0        # hold-last
     = θ(t0)         若 t < t0                          # back-fill 首段
h(t) = [cos θ(t), sin θ(t)]
```

**计算位置冻结**:该状态机在 §5 S3(通道编码)内、**crop 与重采样之后、yaw 增广之前**一次性计算并随 clip 缓存;dataloader **不得**在不同 crop 上重算(否则同一物理帧在不同 crop 下 h 不同)。规则版本号 `continuity_rule_version = "btjdk-h-cont-1"` 写入 `heading_table_hash`。

**(d) 等变性(可机检 → G-heading-E)**:`deg(t)` 只依赖 `n(t)`,而 `n` 在 yaw 下不变(`‖Y(φ)f‖_xz = ‖f‖_xz`);hold/back-fill 只在角度上加常数。故「先增广后计算」与「先计算后 `Rot2(φ)`」逐帧一致。

**(e) per-clip 诊断字段(写入 build manifest,normative)**:`heading_deg_frac`(退化帧比例)、`heading_deg_run_max`(最长连续退化帧数)、`heading_deg_jump_max_deg`(各退化区间跨越的真实 heading 变化最大值)、`heading_all_degenerate`(bool)。

**(f) 禁止的证据迁移(避免重犯类别错误)**:`corpus_singularity_audit_full.json` 的 `eps_exposure_table`(全部 `frames_below = 0`)测的是 **legacy yaw 载体行**的 `s = q_w² + q_y²`,**不是** `n(t)`;**不得**用它论证新定义下退化频度为零。§9 G-heading-D 的阈值必须用**重建后数据**上实测的直方图复核(§10)。

#### 2.1.3 rig 支持域声明(normative,fail-closed)

每个 rig 至少满足下列之一,按 S1 → S2 → S3 顺序尝试;三者皆不满足 → `status = REJECTED`,该 rig 的全部 clip 剔除出语料(§9 G-heading-A(a) 有断言核验)。

- **S1 `limb-pair`**:存在可用左右锚点对(关节名可判左右、双侧命名不冲突),且 rest 水平性判据 `‖(w_x,0,w_z)‖₂ ≥ 0.1·‖w‖₂` 成立。
- **S2 `spine-axis`**:无可用左右对,但存在 head-token 关节或非 tail-token 的最深关节作为 `tip`(规则见 §2.1.1(d)),且 rest 水平性判据成立。
- **S3 `explicit`**:rig 元数据显式提供 `rest_forward_xz`(2 向量,构建期归一化为 `u_rig`,`pol` 恒为 +1 因方向已由人给定)与 `physical_scale`(> 0,直接充当 §7 的 `s_rig`)。`status = EXPLICIT_METADATA`,须走 §9 G-heading-A(d) 的完整证据流程。

**J = 1 rig**:S1/S2 结构上不可能(无第二个关节可构成锚点对或 spine tip),且 rest bbox 对角线 = 0 使 §7 的 `s_rig` 无定义。故 **J = 1 必须走 S3,否则 REJECTED**。当前语料实测 `n_joints` 最小为 **9**(`phi_skel_per_rig_full.tsv`),因此 **v3.0 语料内不存在 J = 1 / J = 2 rig**;S3 只是为未来语料保留的最小接口,除 §9 G-heading-F 的 fixture 外**不实现任何额外逻辑**(最简可行,不做投机性设计)。

**链状 / 星状 / 无 L-R 语义 rig**:走 S2。**重要改进(须显式记明)**:在 §2.1.1(c) 的新定义下 `u_rig` 是 **rest 姿态常量**,`f(t) = D_root(t)·u_rig` 只随 root 刚体旋转变化,因此 **运行时躯干卷曲(蛇盘绕、链状弯折)不再导致 heading 退化** —— r4 担心的「链状卷曲时 spine 投影退化」是**旧的逐帧锚点几何定义**的性质,新定义下不成立。S2 的退化只在 **rest 姿态**上判定,由水平性判据在**构建期**一次性 fail-closed 捕获(例如「rest 脊柱竖直的人形」必须被 S2 拒绝而落回 S1)。

**空 block 语义(normative,补 §4 / §7 / §8)**:任一 mask 的有效计数为 0 时(`joint_mask.sum() == 0`、`edge_mask.sum() == 0`、`contact_supervised_mask.sum() == 0`),该 block 的 loss **记为 0 并跳过**:reduce 分母取 `max(count, 1)`,且该 block 的加权项乘以指示子 `1[count > 0]`;**禁止除零、禁止产生 NaN**。每个训练 step 必须记录各 block 的 `count`,`count == 0` 的 block 在训练日志中逐 block 显式计数上报(fail-loud —— 恒 0 的 loss 会掩盖 mask 构造 bug)。`frame_mask.sum() == 0` 的样本视为数据错误,**直接 abort 而非跳过**。

### 2.2 关节流 b:[B, T, J, 13](J 行**真同构**,root 行无任何特例)

| ch | 所有关节行(含 root 行)统一语义 | 坐标系 |
|---|---|---|
| 0:3 | **j_p**:`x,z = P_j.xz − r_s_xz`;`y = P_j.y`(世界绝对)。**root 行 = 物理 pelvis 相对平滑 root 的 xz 残差 + y 绝对值**(非无约束 3D 残差;y 通道即世界绝对高度、且是 root 高度的**唯一权威值**,§2.1;kimodo 官方 hips_offset 同款) | 世界轴对齐平移相对 |
| 3:9 | **j_a**:真全局旋转的 rest-delta,`6d(R_global(j) · R_rest_global(j)⁻¹)`(6d codec 逐元素冻结见 §2.3)。**唯一合法输入 = §9 G-motion-rot 通过的逐帧原始旋转;禁止由 legacy 13ch 或关节位置反推(见 §9 G-motion-rot.6)。** root 行同式(**可含 tilt,不再是 yaw 载体**)。**数据来源路径 A / B 待 user 拍板,见 §2.2A / §2.2B** | 全局(rest-delta) |
| 9:12 | **j_v**:全局速度特征,`v_t = (P_world(t+1) − P_world(t)) · fps_tgt` 存于 t;**尾帧重复最后有效速度**(kimodo 官方);**平滑与速度在全长上计算、crop 取其限制**(§5.9 变更 1,改正 v2.3 的「先裁后算」);**T=1 时 j_v ≡ 0**。恢复零依赖 | 世界系 |
| 12 | **f**:contact **特征位 0/1**(编码侧);decoder 输出为 **logit**(sigmoid + 阈值后才是 contact,两者规范区分)。**编码侧按源逐关节原样保存,不做任何关节级删除或语义改写**:`f[t,j] = source_contact[t,j]`,**包含 root 行**。BCE 的归约域 = `contact_supervised_mask`(§4),但**存储与 G1 往返判据用的是未 mask 的原始位** | — |

**撤销 root contact 退役**:理由见顶部「撤销 3」。双流设计下 root 行本就有 ch12 通道,保留它**不增加任何维度**,是零成本的无损。

**足端语义 contact 另设,不覆盖源位(normative)**:若下游需要「足端语义 contact」(foot-skate 指标、蒙皮落地判定),**另设派生量**:`f_foot = f ⊙ foot_semantic_mask`,**仅用于指标 / 可视化**,不进 VQ 张量、不进 G1、不进 repr 往返判据。
**为什么不能用足端名单制 mask 当 ch12 的合格域(实测证据)**:两个无肢蛇 rig 的**全部关节**都是接触合格的 —— **Anaconda 27/27**(含 `Hips`、`BN_Head`、`BN_Jaw`、`BN_Tone_01..04`)、**KingCobra 19/19**(含 `Hips`、`BN_Head_01`、`BN_Mouth_01`、`BN_Tongue_01/02`),对应 **67,015 个 contact-ON 位**(Anaconda 54,810 + KingCobra 12,205)。足端名单制会把它们**全部清零**,损失是 root-退役方案(3,687 位)的 **18 倍**。「叶关节 only」的退路同样不成立:蛇的合格关节里只有 2/27 与 4/19 是叶。另按每 rig 6 clip 抽样(382 rig,11,040 个合格关节槽),有 6 个 rig 的合格关节名不含任何足端词根 —— `PZ_Giant_Panda_Female/Male`(`def_FalseThumb_joint.L/.R`)、`Anaconda`、`KingCobra`、`Deer`(`Elk*PhalanxPrima`)、`Roach`(`Bone16/17/18/22`)。**合格域必须由数据统计得出,名字规则只能作为 `foot_semantic_mask` 的输入。**

**「信息无损」在 contact 上收敛为一条可机检命题**:`decode(encode(f)) == f` **逐位相等**(`contact_bits_sha256` 一致),包含 root 行与全部非足端合格关节。任何「已审计的可忽略损失 / 设计接受」类措辞从本标准删除;若某来源确实无法位级还原,该来源**整体不得进入本 schema**。判据见 §9 G1e。

**rest-delta 的前提(§10 开放问题 1 / §9 G-rest,前置 gate)**:`R_rest_global(j)` 必须是与 motion 同世界系的**真全局 rest 变换** —— 现行 `tpos_first_frame` 的 root 行同样是 legacy yaw 载体、PZ 的 tpos 在资产系(中位 87.48° 偏移、最大 179.86°),**均不可直接用**。Phase 2 必须先从 raw 源重建真全局 rest 帧并定义镜像约定;「rest-校准使跨 rig 同姿态可比」在该 gate 通过前不作为本标准的承诺。**追加契约条款(v3.0 新增)**:`P_rest_canonical`(位置)与 `R_rest_global`(旋转)**必须同源同帧** —— §2.1.1(c) 的 `u_rig` 用 rest 位置、`D_root` 用 rest 旋转,两者不同源会破坏该式;不同源会被 §9 G-heading-B 抓成常数角偏移而 fail-loud。

---

#### 2.2A 路径 A —— raw-source 重建(标准按字面成立)

**素材充足性已确证**(2026-08-12 逐 stem 双向比对,缺失 0):

| 源 | clips | 原始旋转载体 | 对应关系 |
|---|---|---|---|
| PZ | 74,522 | `data/animo4d_anytop/bvhs/<stem>.bvh` | stem 1:1,双向缺失 0 |
| TrueBones | 1,070 | `.../AnyTop/dataset/truebones/zoo/truebones_processed/bvhs/<stem>.bvh` | stem 1:1,双向缺失 0 |
| Human v4b | 26,846 | `scratch/humanml3d_272/motion_data/<key>.npy`,`key = stem[len("HML3D_Human_"):]` | 1:1 |

因此:G-motion-rot **不需要** reject 分支;**不需要**「只有位置或 legacy 13ch」的 lossy 子版本;§4 的「信息无损」承诺在 j_a 上**可以真正成立**。

**读取规范已冻结且实测通过**(§9 G-motion-rot.1/.2):PZ 25/25 clip、TB 60/60 clip 的 FK-vs-源位置相对误差 ≤ **3.26e-5**(阈值 1e-4);非叶全局旋转与源一致到 **7.8e-6°**。

**得到什么**:全部 J 行(含叶关节、含 root)的真全局旋转;root 的完整 3-DOF 朝向 —— 这也是 §2.1.1(c) heading 运行时公式的**唯一完整合法输入**。

**代价(须计入 Phase 2 预算)**:
1. **源解析本身极便宜**:实测单核 PZ **0.031 s/clip**、human 272 解码+Slerp **0.027 s/clip** → 全语料旋转提取合计约 **0.8 CPU-hour 单核**(PZ ≈ 0.6 h,human ≈ 0.20 h,TB ≈ 0.01 h),可并行、无需 GPU。**「重跑 10 万 clip 太贵」不成立**;真正的代价是 §4 已声明的全链重建(重编码全通道 + 归一化 moments + token cache + VQVAE/backbone/evaluator 重训),与选 A 还是 B 无关。
2. L4_safe 的 name filter 须按 `clean_filter_manifest.json` 重新施加(L3 中间目录已删,见 §5.0)。
3. **PZ stage-1(`00_raw_bvh_target` / OVL 提取)只在 Windows H: 盘,本文件系统不可重跑** —— 磁盘上的是 stage-2 `02_anytop_layout` 产物。故 stage-1 的一切约定(单位、世界系、ROOT 节点选择、24 fps、`_end_site` 命名)**冻结在现状且不可复查**,标准只能以 stage-2 BVH 为「原始」(§10 已知边界)。
4. 动物侧 24 fps → 统一目标 fps 的重采样此前不存在,需新增并重算 j_v(§5 S1)。

#### 2.2B 路径 B —— 13ch 派生(标准必须显式改名为 lossy)

**j_a 降级为「仅非叶关节可得」。** 由 13ch 可精确重建的是:`{有子关节的关节}` 的局部旋转(存在其子关节行,`R_local(parent(j)) = 6d⁻¹(row_j)`),测地误差 ≤ **7.82e-6°**;以及 13ch 世界系与源世界系之间的**逐帧纯 yaw**(`|R[1,1]−1| ≤ 1.11e-15`)。

**丢失什么**:叶关节自身的旋转,占关节槽 **6,305 / 23,037 = 27.4%**(per-rig **7.4% – 66.7%**)。以恒等替代的真实误差:PZ mean **10.80°** / p95 up to **153.6°** / max **179.3°**;human mean **4.40–33.00°** / max **113.2°**;TrueBones **≤ 3.2e-6°**(其叶关节全部是 End-Site 提升关节,源侧本就 `R_local ≡ I`)。丢失的叶关节正是**足端与末端**(实测叶名例:`def_toeFrontIndex1_joint.*`、`def_toeRearPinky1_joint.*`、`def_c_head_joint`、`def_c_tail5_joint`;human 的 `left_foot` / `right_foot` / `head` / `left_wrist` / `right_wrist`),即接触与末端语义所在,**不在无关紧要处**。

**走 B 则必须同时执行**:
1. 标题与 §4 改名为 **`btjdk-v3.0-lossy`**(`repr_schema_version` 同改),**删除**「信息无损」「G1 往返可精确」全部字样;
2. 叶关节的 j_a 行进 `leaf_rot_unobserved_mask`(与 §5.0 的 `rot_supervised_mask` **分开两个字段**),decode 侧输出显式标注 unconstrained;**绝不**用 IK / 位置反推填充 —— 否则重蹈 `handoff/20260630_190042_human_rot6d_data_encoding_lessons.md` 的 ill-conditioned target(从低 DOF 推断高 DOF 会留下一个不受约束的自由度,它作为 GT 合法但作为学习目标不可学);
3. §9 G1 往返判据按 mask 分块重述,叶关节的旋转块从往返判据中显式排除并在 gate 报告里逐条列出;
4. §2.2 rest-delta 前提段中「rest-校准使跨 rig 同姿态可比」的承诺,改为「**仅在非叶关节上成立**」。

**B 路径下相对现行 13ch 还剩多少优势(诚实清点)**:全局化位置(误差不积分)、显式 heading 双流、root 行同构、速度免积分恢复、平移-only canonicalize + yaw 增广/条件 —— 这些**全部保留**,仍是相对 13ch 的实质改进。唯独「真全局 rest-delta 旋转让跨 rig 同姿态可比」这一条**只在 72.6% 的关节槽上成立**。

**交叉说明(给 §2.1 heading 组)**:若选 B,§2.1.1(c) 的 heading 公式**在 root 上仍可执行** —— root 的完整朝向在 13ch 的 row-1 里,实测与源只差一个纯 yaw(`|R[1,1]−1| ≤ 1.11e-15`),而纯 yaw 差只使全语料 θ 整体偏移一个已知常数,可由 §9 G-heading-B 检出并吸收进 `u_rig`。故 heading 设计不因 B 路径失效。

---

### 2.3 6d 旋转编解码冻结(normative)

**为什么必须冻结:本仓库现存两套互为转置的 6d 约定,且同时活跃**(2026-08-12 实测 `gs_rows(d6) == gs_cols(d6)ᵀ`,Linf 差 0.0):

- **列约定**:`src/data/anytop_dataset.py:149 _rotation_6d_to_matrix_np`、`src/models/graph_salad/world_recovery.py:32 _rot6d_to_matrix_torch`、`src/data/anytop_rot6d_fk.py:_rotation_6d_to_matrix_np`;AnyTop `utils/rotation_conversions.py:536-548` 与 `550-566`。
- **行约定**:AnyTop `utils/rotation_conversions.py:513-534 rotation_6d_to_matrix`(`torch.stack(..., dim=-2)`)与 `567-582 matrix_to_rotation_6d`(drop last **row**);**以及人类 v4b 构建器 `scripts/_v4_build_from_272.py:49 decode_6d_rows`**(MotionStreamer-272 上游约定)。
- 因 `6d(I) = [1,0,0,0,1,0]` 在两约定下**完全相同**,任何以恒等姿态为目标的自检(含 v2.3 §9 G-rest 判据)对转置错误**零判别力**。

**BTJD-K 冻结列约定**(与 Kimodo codec 及本仓库全部 decode/FK 路径一致)。

**基本约定(全部 normative)**:向量为**列向量**;旋转为**主动**旋转(`v′ = R v`,转的是物体不是坐标系);复合按**左乘**(`R_total = R₂ R₁` 表示先 R₁ 后 R₂);世界系 Y-up 右手,绕轴正向按右手定则。

**编码 `d6 = 6d(R)`(逐元素冻结)**:
```
6d(R) ≜ [ R[0,0], R[1,0], R[2,0],  R[0,1], R[1,1], R[2,1] ]
       = concat( R·e_x , R·e_y )        # R 的第 0、1 两"列"
```
输入前置断言(fail-closed,不做投影修复):`‖RᵀR − I‖_∞ < 1e-6` 且 `det(R) > 0`,任一不满足即 abort。

**解码 `R = 6d⁻¹(d6)`(Gram–Schmidt,逐元素冻结)**:
```
a1 = d6[0:3] ;  a2 = d6[3:6]
b1 = a1 / max(‖a1‖₂, ε_6d)
u  = a2 − (b1·a2) · b1
b2 = u  / max(‖u‖₂ , ε_6d)
b3 = b1 × b2                              # 右手叉积
6d⁻¹(d6) ≜ [ b1 | b2 | b3 ]               # 按"列"拼接,R[:,k] = b_k
```
`ε_6d = 1e-8`(冻结)。形式是 `max(‖·‖, ε)` 而**非**现存实现的 `‖·‖ + 1e-8`(后者在单位模处引入 1e-8 相对偏差);该变更属 §4 硬 schema 断的一部分。**禁止**用 SVD / 极分解 / 对称正交化替代 Gram–Schmidt(判别用例 GOLD-5);**禁止**在解码前对 d6 做任何归一化或 clamp。

**下游公式消歧(全部按列约定重读)**:`j_a = 6d(R_global(j)·R_rest_global(j)ᵀ)`(§2.2);`R_global(j) = 6d⁻¹(j_a[j])·R_rest_global(j)`(§3);`j_a′ = 6d(Y(φ)·R_global(j)·R_rest_global(j)ᵀ)`(§6) —— **Y(φ) 左乘**在列约定下即「世界系再转 φ」;若实现误用行约定,该左乘等价于右乘转置,整段 yaw 增广语义即错,且该错误不会被任何 round-trip 自检发现。

**实现纪律(normative)**:BTJD-K 侧一切代码只允许调用本标准自带的 `btjdk_6d_encode` / `btjdk_6d_decode`;**禁止 import 或 verbatim copy** AnyTop `utils/rotation_conversions.py` 的 torch 路径与 `scripts/_v4_build_from_272.py:decode_6d_rows`。跨源接入时必须**显式转置**并在 per-clip manifest 记录 `source_6d_convention ∈ {"col","row"}`。建议配一条 CI grep(禁止 `from utils.rotation_conversions import` 与 `decode_6d_rows` 出现在 BTJD-K 代码路径)。

**退化监控**:解码侧统计 `‖a1‖₂ < 1e-3` 或 `‖u‖₂ < 1e-3` 的帧-关节比例 `deg_rate_6d`,写入训练/评测日志。GT 数据上 `deg_rate_6d` 必须 `== 0`(否则 abort);生成侧只监控不 abort。

## 3 恢复路径(零积分,精确)

```
# 前置:先按 §7 逆公式反归一化(x = x̃·s_rig/g_G;h 全域投影回 S¹:‖h‖₂ ≥ ε ? h/‖h‖₂ : [1,0]),再执行以下恢复
P_root = [r_s_xz.x + j_p[root].x,  j_p[root].y,  r_s_xz.z + j_p[root].z]   # xz 残差加回、y 直读
P_j    = [j_p[j].x + r_s_xz.x,     j_p[j].y,     j_p[j].z + r_s_xz.z]      # 所有关节行统一同式,root 无特例
R_global(j) = 6d⁻¹(j_a[j]) · R_rest_global(j)                              # 6d⁻¹ 见 §2.3(列约定)
heading     = atan2(h.sin, h.cos)   # 直读,不推导;基 = 零轴 +Z、绕 +Y 右手(§2.1.1(a))
```

任何帧误差只影响该帧。**FK 一致项是 parent-edge 方程**(r1 意见 b):

```
P_c − P_p   = R_global(p) · offset[c]                                       # 列向量右乘,§2.3 列约定
R_global(p) = 6d⁻¹(j_a[p]) · R_rest_global(p)
offset[c]   = R_rest_global(parents[c])ᵀ · ( P_rest_canonical[c] − P_rest_canonical[parents[c]] )
offset[root] ≜ (0,0,0)                                                      # root 位置由 r_s_xz / j_p 承载,不入骨架
```

即 **`offset[c]` 位于父关节的 rest 局部系**(v2.3 缺此定义,r4-BLOCKING-05)。`edge_mask` 的定义域精确化为
`{ (parents[c], c) | c ≠ root, joint_mask[c] ∧ joint_mask[parents[c]] }` —— **含叶关节**(叶关节位置仍受父→叶入边约束;仅叶关节的**旋转**无出边位置约束,故另设 SO(3) 旋转重建项 §8 j_a 块)。若位置纯由旋转生成,则完整恢复仍是链遍历。

## 4 与旧 13ch 的关系:**全量不兼容**

逐帧转正 RIFKE→平移相对;parent-local(兄弟共享)→真全局 rest-delta;heading-local 速度 + root 积分→世界系特征;root 杂烩行→同构行 + 全局流。**一切现有消费者(recovery / losses / renderer / token export / VQVAE / backbone / 12ch evaluator)都假设 legacy 语义,不能混用** —— evaluator 亦须重训。`repr_schema_version = "btjdk-v3.0"` + 七个契约哈希写入所有工件,不匹配硬 abort。

**v3.0 相对 v2.3 的破坏性清单(补充,防未来误判为 bug)**:① rglob 5→4 维(§2.1,MINOR-01);② 6d 解码由 `‖·‖ + 1e-8` 改为 `max(‖·‖, 1e-8)`(§2.3);③ heading 叉积方向修正 π 且改用 root-rotation 派生(§2.1.1);④ `phi_skel_deg` 全表作废、改为 `u_rig` / `phi_rest_deg`;⑤ 新增 24/30 → `fps_tgt` 重采样(§5 S1),动物侧全部 clip 帧数与 j_v 改变;⑥ 删除逐 clip heading 转正(§5 S0④);⑦ contact 阈值改为 fps 无关速度阈(§5 S3);⑧ root contact 保留、contact 合格域由数据统计得出;⑨ `rest_table_hash` 撤销、并入 `skeleton_table_hash`;⑩ 哈希构造改为 TLV envelope(§4.2)。

**Schema 必含字段(normative,按工件类型分区;缺失其适用分区的任一字段即 abort)**

- **通用(所有工件)**:`repr_schema_version`、`semantic_hash`、`preprocess_hash`、`skeleton_table_hash`、`heading_table_hash`、`contract_hash`、`fps_tgt`、`eps`。
- **per-rig**(存于 skeleton payload §4.1 与 moments 工件):`s_rig`(§7,供反演)、分组标量增益 `g_G`(train-only,§7)、`source_contact_eligibility [J]`、`foot_semantic_mask [J] | null`、`rot_supervised_mask [J]`、(仅路径 B)`leaf_rot_unobserved_mask [J]`、`scale_factor`、`y_floor`、`physical_scale | null`。
- **per-clip mask(五类,v2.3 为四类)及其归约作用域**:
  - `frame_mask [T]` —— padding 帧;作用于**全局流与关节流**的一切 loss/统计归约;
  - `joint_mask [J]` —— padded/无效关节;作用于关节流归约;
  - `edge_mask` —— 定义域见 §3;作用于 parent-edge FK 一致项;
  - `contact_supervised_mask [J] = source_contact_eligibility ∧ joint_mask` —— 作用于 contact BCE 归约(**取代 v2.3 的 `contact_joint_mask` 命名**);
  - `rot_supervised_mask [J]`(**新增**)`= (rot_source[j] ∈ {"channel","smpl_local"}) ∧ joint_mask[j]` —— 作用于 §8 j_a 块与 §9 j_a 相关 gate 的归约域。
- **per-clip 溯源与预处理字段**:见 §5 末的 per-clip manifest 清单。
- **per-clip contact**:`contact_bits_sha256` = 原始 `f [T,J]` 按 C 序展平、`np.packbits` 后的 sha256(供 §9 G1e 位级核对)。
- **token-cache 专属**:`yaw_seed`、实际采样角 `phi`、`cdir`(§6 方案 A)。其余工件类型此三字段不适用:写显式 null sentinel(`0x07`),消费侧不得读取。
- **ckpt 专属**:**必含 `moments_hash`**(v2.3 的「ckpt 无新增字段」作废 —— 否则同语义配不同 `g_G` 仍可加载)、`loss_contract_hash`(§8)。

### 4.1 skeleton payload(per-rig,normative 字段清单;顺序即 envelope 顺序)

| # | 字段 | 类型/shape | 定义 |
|---|---|---|---|
| 1 | `rig_name` | utf8 | rig 唯一名(与 cond / manifest 一致) |
| 2 | `topology_id` | utf8 | 194 canonical topologies 之一 |
| 3 | `J` | i64 | canonical 关节数 |
| 4 | `joint_names_canonical` | utf8[J] | **canonical(new / FK)序**关节名 |
| 5 | `joint_names_source` | utf8[J] | **源(old / raw)序**关节名 |
| 6 | `new_to_old_perm` | i64[J] | `new_to_old_perm[new] = old`(同 `src/data/anytop_dataset.py:1121`;当前缓存 **381/382 rig 非恒等**) |
| 7 | `old_to_new_perm` | i64[J] | `old_to_new_perm[old] = new`;冗余存储、被同一哈希覆盖,不一致过不了 G-skel-c |
| 8 | `root_index` | i64 | **冻结 = 0**(`_reorder_joints_fk` 保证 `parents[0] == −1`) |
| 9 | `parents` | i64[J] | canonical 序父索引;`parents[0] = −1`,`∀j>0: 0 ≤ parents[j] < j` |
| 10 | `P_rest_canonical` | f64[J,3] | 真全局 rest 关节位置(与 motion 同世界系,§9 G-rest) |
| 11 | `R_rest_global` | f64[J,3,3] | 真全局 rest 旋转(**列约定**,§2.3) |
| 12 | `offset` | f64[J,3] | §3 定义式;`offset[root] = (0,0,0)` |
| 13 | `lr_pairs` | i64[P,2] | 已登记左右对称关节对 (L,R),canonical 索引,按 (L,R) 字典序;无镜像语义写 shape `[0,2]` |
| 14 | `mirror_matrix` | f64[3,3] | 冻结 `M = diag(−1,1,1)`;不允许 per-rig 变体 |
| 15 | `heading_mode` | utf8 | `"limb_pair"` / `"spine_axis"` / `"explicit"`(与 §2.1.1 一致) |
| 16 | `heading_anchor_names` | utf8[2] | limb_pair = (L 名, R 名);spine_axis = (tip 名, root 名)。**以名为键**(perm 不变) |
| 17 | `heading_anchor_indices` | i64[2] | 上两名解析出的 **canonical** 索引;与校准 TSV 的 `*_idx_dataset`(new)/`*_idx_raw`(old)交叉核对 |
| 18 | `contact_eligible_source` | u8[J] | **源** contact 合格域,原样搬运,不做语义再解释 |
| 19 | `contact_eligible_foot` | u8[J] 或 null(`0x07`) | 派生「足端语义」合格域;不派生时写显式 null sentinel |
| 20 | `source_joint_map` | 嵌套 envelope(`0x08`) | `canonical_name → (源节点路径, rot_source ∈ {"channel","endsite_identity","smpl_local"})` |
| 21 | `rot_supervised_mask` | u8[J] | §4 定义 |
| 22 | `scale_factor` / `y_floor` | f64 / f64 | §5 S0② / S0③ 的 per-rig 常量 |
| 23 | `eligibility_clip_list_hash` | utf8 | 统计 `contact_eligible_source` 所用 clip 清单的 sha256 |
| 24 | `schema_note` | utf8 | 冻结常量 `"btjdk-skeleton-v1"` |

`skeleton_hash` ≜ 按上表顺序做 §4.2 envelope 后 sha256。
`skeleton_table_hash` ≜ 全部 rig 的 skeleton payload 按 `rig_name` UTF-8 字节序排列,每 rig 作为一个 `0x08`(嵌套 envelope)字段,外层再套一次 envelope 后 sha256。
**撤销 `rest_table_hash`** —— `R_rest_global` / `P_rest_canonical` 已在 payload 内,两张表并存只会制造第二个可漂移的真值源。§9 G-rest 判据全部保留,但「整表内容哈希写入 manifest」改为写 `skeleton_table_hash`。(落地时须全文 grep `rest_table_hash` 清零。)

### 4.2 canonical binary envelope(normative,**取代** v2.3 的「canonical JSON + float64 LE 拼接」)

v2.3 构造**无 framing**,结构不同的数据可序列化成同一字节流(例:字段名 `"ab"` + 值 `[1.0]` 与字段名 `"a"` + 值 `["b", 1.0]`;嵌套 `[[1,2],[3]]` 与 `[[1],[2,3]]` 展平后同流)。全部哈希改用下列 TLV envelope。

```
envelope(fields) :=
    MAGIC                       8 bytes  = b"BTJDK\x01\x00\x00"
    u32_le  n_fields
    for each field in 冻结顺序(不是按名排序;顺序由各字段清单表格定义):
        u32_le          len(name_utf8)
        bytes           name_utf8
        u8              type_code
        u8              ndim
        u32_le[ndim]    shape                # C 序;ndim = 0 表示标量
        u64_le          payload_len
        bytes           payload
    u64_le  n_fields                          # 尾部重复,封口
hash := sha256(envelope_bytes)
```

`type_code` 冻结:`0x01` f64(IEEE754 LE)/ `0x02` i64_le / `0x03` u8 / `0x04` utf8 标量字符串(ndim=0)/ `0x05` utf8 字符串数组(ndim=1,shape=[n],payload = 逐个 `u32_le len + utf8 bytes`)/ `0x06` bool(存为 u8 0/1)/ `0x07` **显式 null sentinel**(ndim=0,payload_len=0)/ `0x08` 嵌套 envelope(payload = 子 envelope 全字节)。

序列化规则(全部 normative):
- **浮点一律先转 f64 再序列化**;禁止 f32/f16/bf16 进入被哈希 payload(避免 dtype 漂移改哈希,保证 numpy / torch / 纯 python 三方可复现)。
- **NaN 与 ±Inf 禁止**出现在被哈希 payload → abort。
- **负零归一**:序列化前 `−0.0 → +0.0`。
- **字段不得省略**:不适用的字段必须写 `0x07` null sentinel;省略会改变 `n_fields`,即视为不同契约。加载侧必须把 `0x07` 当成「有意为空」而非「未填写」。
- **字段顺序 normative**:按各字段清单的表格顺序;**表类哈希的条目顺序**按键的 UTF-8 字节序(rig 名 / clip id)。
- 被哈希的浮点必须来自**以 f64 持久化的工件**,不得由 f32 工件上采样重算。

**envelope 自身的回归用例(必须随实现落地,先于任何数据构建)**:构造一组「旧拼接构造会碰撞、新 envelope 必须不碰撞」的负例(至少:`{"ab": [1.0]}` vs `{"a": ["b"], "x": [1.0]}`;`[[1,2],[3]]` vs `[[1],[2,3]]`;`utf8[2]=["a","bc"]` vs `utf8[2]=["ab","c"]`),断言各对哈希互不相等;并断言同一 payload 在 numpy / torch / 纯 python 三条实现路径上字节流完全一致。

### 4.3 哈希分工与加载核对(normative,fail-closed)

| 哈希 | 覆盖内容 | 定义处 |
|---|---|---|
| `semantic_hash` | 通道与公式语义:§2.1 表、§2.2 表、§2.3 codec 块、§3 恢复公式块、§6 yaw 变换块。**按锚点抽取**:每块用 `<!--BTJDK:SEM:BEGIN:<id>-->` / `<!--BTJDK:SEM:END:<id>-->` 包裹,按 `<id>` 字典序做 `0x04` 字段入 envelope | §2 / §3 / §6 |
| `preprocess_hash` | §5 冻结的确定性变换链参数块(键清单由 §5 冻结) | §5 |
| `skeleton_table_hash` | §4.1 全部 rig 的骨架 payload(**取代** `rest_table_hash`) | §4.1 |
| `heading_table_hash` | §2.1.1(f) 的 per-rig 参数 + §9 G-heading-A(d) 的**决策记录**。**只存名不存索引**(索引在 skeleton 侧,避免两表互引成环) | §2.1 / §9 |
| `moments_hash` | `s_rig_def`(标识串,如 `"rest_bbox_diag_v3.0"`)、per-rig `s_rig`、per-group `g_G`、`ε`、每组有效样本数、per-rig/全局 `pos_weight`、train split 标识 `split_manifest_hash` | §7 / §8 |
| `source_manifest_hash` | per-clip 溯源(§5 清单)+ `source_6d_convention` | §4 / §5 |
| `contract_hash` | 上述六者 + `repr_schema_version` 的 envelope(不适用项写 `0x07`) | §4.3 |

(`loss_contract_hash` 另立,见 §8;它**不进入** `semantic_hash`,但 `semantic_hash` 是它的输入字段之一。)

**每类工件必含的哈希集合**(缺一即 abort,禁止「缺失 = 跳过」):

| 工件 | semantic | preprocess | skeleton_table | heading_table | moments | source_manifest | contract |
|---|---|---|---|---|---|---|---|
| per-clip 数据分片 | ✓ | ✓ | ✓ | ✓ | null | ✓ | ✓ |
| moments 工件 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| token cache | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| VQVAE / backbone / evaluator ckpt | ✓ | ✓ | ✓ | ✓ | **✓(新增)** | null | ✓ |

**加载核对流程(normative,fail-closed)**:
1. 读工件 `repr_schema_version`,与代码常量做**精确字符串相等**;不等 → abort。
2. 按工件类型取必含集合。任一字段**键不存在** → abort;要求 ✓ 而值为 null → abort;要求 null 而写了具体值 → abort(禁止「多写无害」)。
3. 对本进程手上有原始输入的哈希(如已加载 skeleton payload、moments 数组),**重算**并与记录值比较;无原始输入的,与同批其它工件的记录值**逐字段相等比较**。
4. 比较按上表列顺序进行,**第一处不等即 abort**;异常信息必须同时打印:字段名、期望值、实际值、两侧工件路径。
5. **禁止任何形式的容忍**:不得有哈希白名单 / 多值 `in (...)` 回退 / warn-and-continue / env 开关跳过。唯一允许的旁路是显式 `--rebuild`(重建工件),它**不加载**旧工件。
6. `contract_hash` 相等**不免除**第 2–4 步的逐项核对 —— 它只是一次性快速判定,逐项核对提供可诊断性。
7. 实现收敛为单一 `verify_contract(artifact, artifact_type)` 函数,数据加载 / 训练 / 导出 / 评测四处只调它,不各写一份。

## 5 预处理链(normative,确定性;整体取代 v2.3 的「七步顺序」)

### 5.0 确定性契约 + 源 → canonical 关节保留规则

**确定性契约**:本节定义源文件→训练张量的**唯一**变换链 S0–S8。每步冻结:输入(形态/dtype/joint order/坐标系)、输出、算法与参数、失败条件(一律 fail-closed,不静默回退)。链被拆为**离线段 S0–S4**(每 clip 一次,产物落盘并入 `preprocess_hash`)与**在线段 S5–S8**(每次取样,由 `(crop_seed, clip_id, epoch)` 与 `(yaw_seed, clip_id)` 完全决定)。同 `preprocess_hash` 的任意两个实现必须逐位产生相同的离线工件。

**关节保留规则(normative)**:PZ 源 BVH 的关节数远多于 canonical 表(实测:源 ROOT/JOINT 节点最多 **206**,canonical 最多 **102**;单 rig 差额可达 80+)。差额是 locator(`L0`…`L5`)、面部/内脏 helper(brow / eye / eyelid / lip / tongue / jaw / ear / cheek / anus / chestBreath …)与 **twist helper**(`*AllTwist_joint.*` / `*HalfTwist_joint.*`,每 rig 约 16 个)。
**沿用 L4_safe 的保留集,Phase 2 不得改动。** 权威表述 = `cond[rig]["joints_names"]`;等价地按 `data/animo4d_anytop_clean_L4_safe/clean_filter_manifest.json` 记录的 name filter(`subtree_helper_regex = "volume|fold|scale|bubble|pouch|sternum|throat|wobble"`,`leaf_helper_regex = "a^"`,含 L3 的 locator / `_end` leaf-only 移除)重新施加 —— **L3 中间目录已删除,该 manifest 是唯一权威**。改保留集 = §4 全部哈希、rest 表、heading 表、eval split 同时作废。

**该规则对 j_a 无损,理由(已全量核验,非假设)**:
1. **子树闭合**:全部 311 个 PZ rig,每个保留关节的源父节点也在保留集内 —— **0 违例**。故任一保留关节从 ROOT 起的 FK 链**只经过保留关节**,其 `R_glob` 与 `P_world` 由保留关节自己的源通道**完全确定**。
2. **twist helper 全部是叶**:全部 311 rig 中,名字含 `Twist` 的关节**无一有子节点**(0 例外),且**无一被保留**。它们是纯蒙皮驱动件,不承载任何保留关节的刚体自由度。
3. **被裁骨段自身的 twist 并未丢失**:该 twist 自由度在**保留的**骨段关节自己的 3-DOF Euler 通道里,由 §9 G-motion-rot.1 直接读出。twist helper 只是把同一个 twist 按比例分配到蒙皮上。

**明确声明为 lossy 的部分(与 j_a 无关,不得混为一谈)**:被裁掉的是**蒙皮形变**信息(twist 分配比例、面部/口腔/眼睑等表情骨)。本标准**不承诺**表情与蒙皮细节可逆。§9 G-motion-rot 的 M1–M8 **只在保留集上评估**;被裁关节不进任何 gate、不进任何 loss、不进任何哈希。未来若要做面部/蒙皮,须**新开 schema 版本重跑保留集**,不可在 btjdk-v3.x 内热改。

**TrueBones 的对称规则 + `rot_supervised_mask`**:TB 的保留集**包含**提升后的 End Site(§9 G-motion-rot.1)。这些关节的 `R_local ≡ I` 是**源侧事实**(BVH End Site 节点根本没有 CHANNELS),不是推断或填充;在 `source_joint_map` 中标 `rot_source = "endsite_identity"`。其 j_a 恒等于 `6d(R_glob(parent(j))·R_rest_global(j)⁻¹)`,是父关节旋转的确定函数,**不构成独立监督信号**,故 §8 的 j_a 块只在 `rot_supervised_mask`(§4)内归约。
**注**:若走路径 B,`rot_supervised_mask` 还须再与「非叶关节」取交 —— 两个条件来源不同(一个是源侧无通道,一个是编码侧无载体行),**必须分成两个字段**(`rot_supervised_mask` / `leaf_rot_unobserved_mask`),不得合并,否则 A/B 切换时语义静默改变。

### S0 源装载与世界系规范化(离线)

- 输入(per clip,由 §9 G-motion-rot 提供;缺任一项 **reject**):`P_src[T₀,J_src,3]` f64、`R_src_local[T₀,J_src,3,3]` f64、`root_xform`、`joint_names_src[J_src]`、`parents_src[J_src]`、`fps_src`、`unit_scale`、`axis_up`、`source_sha256`。
- **native fps 的确定**:BVH 源取头部 `Frame Time` Δt,`fps_src = round(1/Δt)`,要求 `|1/Δt − fps_src| ≤ 0.01·fps_src`,否则 reject;非 BVH 源须在源 manifest 显式声明常量 fps。**仅支持均匀时基**;若源带逐帧时间戳,须验证 `max_i |t[i+1]−t[i] − 1/fps_src| ≤ 1e-4/fps_src`,否则 reject。实测:PZ `Frame Time 0.041667` → **24**;AnyTop `truebones_processed` BVH → **24**(与 PZ 同规格,见 §9 G-motion-rot.1);raw `Truebone_Z-OO` 的 1,153 BVH 为 `0.0333333` → 30,但**该 raw 目录不是我们的源**(§9 G-motion-rot 表下的 ⚠);human 272 源 → **30**。
- **世界系规范化(顺序定死,per-rig 常量)**:① 轴系转 **Y-up 右手**;② 尺度 `scale_factor(r) = HML_AVG_BONELEN / mean_j‖offset_rest(r,j)‖`,`HML_AVG_BONELEN = 0.2092142857142857`(沿用 AnyTop,per-rig 常量,由该 rig 的 rest/T-pose 导出,**非逐 clip**);③ 落地 `y_floor(r) = min_j P_rest_global(r,j).y`(scale 之后,per-rig 常量,使 `y_ground ≡ 0`);允许 per-rig 显式覆盖(AnyTop 对 Dragon 用 `ground_height = 0`),覆盖表登记进 `skeleton_hash`;④ **不做任何 heading 旋转** —— 现行 AnyTop 的 `rotate_to_hml_orientation`(逐 clip 把首帧转向 +Z)在 BTJD-K 中**删除**(P2/P5,朝向由 §6 yaw 随机化承担)。`scale_factor` 与 `y_floor` 写入 `skeleton_hash`。
- **关节重排**:按 `new_to_old_perm` 的逆(`perm_src_to_canonical`)**一次性**重排到 canonical joint order(§4.1);此后全链不再重排;`parents`、L/R 锚点、contact 合格域、rest 表一律以 canonical order 表达。
- 输出:`P0[T₀,J,3]`、`Rloc0[T₀,J,3,3]`,f64,canonical order,canonical world frame。
- 失败条件:`T₀ < 1`;任一非有限值;`|det(R)−1| > 1e-6`;`J_src ≠ J_canonical`;perm 非双射。

### S1 重采样到 `fps_tgt`(离线)

> **`fps_tgt` 待定(§10 开放问题 7)。以下公式与 gate 与具体取值无关;20 为 v2.3 沿用值,写作占位。**

- **重采样作用于最小 DOF 集,不作用于位置**:`root_t[T₀,3]`(root 世界平移)线性插值 + `Rloc0` 逐关节 SLERP,随后**一次 FK** 得 `P1`。理由:位置逐分量线性插值会收缩骨长并破坏 §3 parent-edge / §9 G2 的 FK 一致性;分别插位置与旋转会产出两套互不相容的几何。
- **时间网格(定死,不做时长拉伸)**:`D = (T₀−1)/fps_src`;`T₁ = floor(D·fps_tgt) + 1`;`t_k = k/fps_tgt, k = 0..T₁−1`(恒有 `t_k ≤ D`,不外插);源时刻 `t_i = i/fps_src`。**禁止 `linspace(0, T₀−1, T₁)` 这类按索引拉伸的写法**(它把时长按 `(T₀−1)/(T₁−1)·fps_tgt/fps_src` 微量缩放)。
- 位置/平移插值 = 逐分量 `numpy.interp`(f64,端点 clamp)。
- **SLERP 约定(定死)**:`scipy.spatial.transform.Slerp`,以 `Rotation.from_matrix(Rloc0)` 在 `t_i` 上构造、在 `t_k` 上求值;四元数沿时间轴取**最短弧** —— 自 t=0 起逐帧强制 `⟨q_i, q_{i−1}⟩ ≥ 0`,否则 `q_i ← −q_i`(实现须**显式断言**,不依赖库的隐式行为)。`T₀ = 1` → `T₁ = 1`,直接拷贝。
- **`fps_src == fps_tgt` → 恒等旁路**(不得进入插值路径),保证同帧率源逐位不变。
- 输出:`P1[T₁,J,3]`、`Rg1[T₁,J,3,3]`(FK 后全局旋转),f64。`resample_mode ∈ {identity, dof_slerp}` 记入 manifest。
- ⚠ **迁移必需(实证)**:现行语料 fps 不一致却被统一按 20fps 消费 —— PZ 24fps(BVH 118 帧 → npy 117 帧,1:1 无重采样)、TB 24fps、human 20fps(272 源已 30→20);`src/data/anytop_dataset.py` 只有单一 `target_fps = 20.0`,全仓库**不存在** `native_fps` / `frame_time` 字段。后果:动物侧以 0.83× 速度播放,`j_v = ΔP·20` 对 PZ 低估 1.2×。附带:`corpus_singularity_audit_full.json` 的 `hours_at_20fps = 138.7` 与 `theta_max` 的「deg/frame at 20fps」标定对动物侧偏差 20%,须按 source 分别重算。**BTJD-K 构建必须执行 S1,不得沿用现有 npy。**

### S2 平滑 root(离线,全长)

- 输入 `P1[:, root, [0,2]]`(f64,**整段、未 crop**)。滤波器:`butter(N=4, Wn=f_c/(fps_tgt/2))`,`f_c = 1 Hz`;`filtfilt(padtype='odd', padlen=15)`(4 阶下 `len(a)=len(b)=5`,`padlen = 3·max(len(a),len(b)) = 15`)。**y 不进入全局流**(§2.1,MINOR-01)。
- 短 clip 回退:`L_f ≜ round(fps_tgt/f_c)`;`T₁ < 3·L_f` → x、z 各做全 clip 帧号 OLS 直线拟合 `x(t) ≈ a + b·t` 取拟合值;`T₁ = 1` → `r_s_xz = P1[0, root, [0,2]]`。
- 输出 `r_s_xz_full[T₁,2]` f64。失败:filtfilt 异常或输出非有限。
- 与 kimodo 官方 ADMM 平滑的差异为已知偏离(确定性、免解器依赖);蛇/鱼/蜂鸟高频推进的截止敏感性走 §9 G7 视觉裁决。

### S3 通道编码(离线,全长,未 canonicalize / 未 yaw / 未归一化)—— 全部 f64

- `j_p[t,j] = (P1[t,j].x − r_s_xz_full[t].x, P1[t,j].y, P1[t,j].z − r_s_xz_full[t].z)`(**平移不变**)。
- `j_a[t,j] = 6d(Rg1[t,j] · R_rest_global(j)⁻¹)`(codec §2.3)。
- `j_v[t,j] = (P1[t+1,j] − P1[t,j])·fps_tgt`,`t ≤ T₁−2`;**尾帧** `j_v[T₁−1] = j_v[T₁−2]`;`T₁ = 1` → `j_v ≡ 0`。
- `h[t]` 按 §2.1.1 + §2.1.2 的状态机(**在此一次性计算并缓存**,dataloader 不得重算)。
- **`f`:contact 是派生量,统一在此重算。** 所有源都无原生 contact 通道(PZ/TB 为 BVH,human 为 HumanML3D 检测器产物),因此**不存在 nearest / OR 搬运的选项**:
  ```
  f[t,j] = 1  ⟺  j ∈ source_contact_eligibility
                ∧ ‖P1[t+1,j] − P1[t,j]‖₂ · fps_tgt ≤ v_c
                ∧ |P1[t+1,j].y| ≤ h_c            ,  t ≤ T₁−2
  f[T₁−1] = f[T₁−2]        # 尾帧,与 j_v 同规则
  T₁ = 1 → f ≡ 0
  ```
  **阈值(定死,fps 无关)**:`h_c = 0.3`(AnyTop `FOOT_CONTACT_HEIGHT_THRESH` 原值);`v_c = √0.002 × 20 = 0.8944271909999159` 单位/秒 —— 由 AnyTop/HumanML3D 的 `FOOT_CONTACT_VEL_THRESH = 0.002`(**每帧位移平方**、在 20fps 下标定)换算:`‖ΔP‖² ≤ 0.002 @20fps ⟺ ‖ΔP‖·fps ≤ √0.002·20`。**v2.3 未做该换算**;直接沿用 0.002 会让 24fps 的 PZ 判据实际收紧 1.44×、30fps 收紧 2.25×。单位 = AnyTop-scaled(S0②);若未来改用物理米,`v_c` / `h_c` 必须同步换算。
  索引约定:速度取 `t→t+1`、高度取 `t+1` 帧 —— **与 AnyTop 原实现逐位一致**,以便 §9 G-contact 复现门可用。contact 对 S5 canonicalize(仅平移)与 S6 yaw(绕 +Y)不变,故在此计算与在链末计算等价。
- **dtype 下转(全链唯一一次)**:`j_p/j_a/j_v/h/r_s_xz` → f32(`astype(np.float32)`,IEEE-754 round-to-nearest-even;**禁止任何显式 round / quantize / clip**);`f` → uint8 ∈ {0,1};masks → uint8。
- 失败:任一输出非有限;`f` 出现 {0,1} 以外值。

### S4 离线工件落盘

**全长、未 crop、未 pad、未 canonicalize、未 yaw、未归一化**;`T_clip = T₁` 与 per-clip manifest 一并写盘。

### S5 crop + canonicalize(在线,每次取样)

- 窗口长 `T_w`(配置项,入 `preprocess_hash`;具体值待定,§10)。
- 起点 `t₀`:train = `RNG(crop_seed, sha256(clip_id), epoch).randint(0, max(0, T_clip − T_w) + 1)`,**必须与 DataLoader worker 数无关且可复现**(现行 `np.random.randint` 不满足);val / test / token 导出 = `t₀ = 0`;`T_clip ≤ T_w` → `t₀ = 0`。
- 对 `r_s_xz / h / j_p / j_a / j_v / f` 施加**同一** slice `[t₀, t₀ + min(T_w, T_clip − t₀))`。
- canonicalize:`q_xz = r_s_xz[t₀]`;`r_s_xz ← r_s_xz − q_xz`;**其余通道不变**(j_p / j_a / j_v / h / f 皆平移不变)—— 与 v2.3「从每帧全部 P_world 中减 q」逐值等价,但省掉一次 P_world 重建。

### S6 整段刚体 yaw(在线 / 导出,可选)

变换定义见 §6,不变。

### S7 归一化(在线)

见 §7 N1′;**至少 f32**,禁止 fp16/bf16。

### S8 padding(在线,最后一步)

- `T_win < T_w` → **尾部右 pad**;**pad 值 = 全 0**(含 `j_a` —— pad 帧刻意不是合法 rot6d:它被 `frame_mask` 从一切 loss/统计/归一化归约中剔除,任何消费 pad 帧的实现即缺陷);`f` pad = 0;`frame_mask[0:T_win] = 1`,其余 0。关节维 pad 到 `J_max` 同规则(尾部、全 0、`joint_mask`)。
- pad **必须在归一化之后**(N1′ 是纯缩放,0 仍为 0;定死顺序仅为消除实现分歧)。

### 5.9 相对 v2.3 的语义变更(必须显式记录)

1. **平滑与速度改在全长上计算,crop 取其限制**:v2.3 的「先裁后算」使每个窗口自带 filtfilt 边界瞬态,并在窗口末帧造出假的「尾帧重复速度」;改后窗口内 `j_v` 除非窗口触到 clip 末帧,否则都是真实前向差分,且 `T < 3L_f` 的线性趋势回退只对**真短 clip** 生效。
2. **重采样作用于 DOF 而非位置**(消除骨长收缩)。
3. **contact 阈值改为 fps 无关速度阈**(数值在 20fps 处与 AnyTop 完全一致)。
4. **删除逐 clip heading 转正**(legacy 13ch 与 BTJD-K 相差一个 per-clip yaw;contact 对 yaw 不变,故 §9 G-contact 位级复现仍成立)。**副作用**:首帧朝向不再统一,任何隐含「首帧朝 +Z」的渲染/评测脚本会静默出错,须逐个排查。
5. 现行 npy 语料不可复用。

### per-clip manifest 必含字段(normative,缺任一即 abort)

`clip_id`、`source_path`、`source_sha256`(源文件**全字节** sha256)、`source_kind ∈ {pz_bvh, truebones_bvh, smpl272}`、`fps_src`、`frame_time_src`、`T_src`(BVH `Frames:` / 272 的 T30)、`fps_tgt`、`T_clip`、`resample_mode`、`scale_factor`、`y_floor`、`axis_convention`、`angle_unit ∈ {deg, rot6d}`、`euler_order ∈ {ZYX_intrinsic, null}`、`matrix_convention = "column_vector_active"`、`compose_order = "R_glob(c) = R_glob(p) @ R_local(c)"`、`rot6d_codec_id`、`source_6d_convention ∈ {col, row}`、`joint_map_hash`、`endsite_promoted`(bool)、`smooth_mode ∈ {butter4, ols_line, single_frame}`、`f_c`、`Wn`、`padlen`、`contact_params = {v_c, h_c, index_convention: "vel[t,t+1] & height[t+1]"}`、`contact_bits_sha256`、`tail_rule = "repeat_last"`、`dtype_store = {continuous: "float32", contact: "uint8"}`、`heading_deg_frac`、`heading_deg_run_max`、`heading_deg_jump_max_deg`、`heading_all_degenerate`、`preprocess_hash`。
取样期(train 日志 / token cache)另记:`crop_seed`、`epoch`、`t0`、`T_win`、`T_w`、`yaw_seed`、`phi`、`cdir`。

## 6 Canonicalize 与 yaw-cache 契约(normative 选择)

- 存储态:仅平移标准化(首帧 `r_s_xz` → 原点)。不做任何 heading 旋转。
- **整段刚体 yaw 变换定义(normative,两流统一;Y-up 右手系)**:
  ```
  Y(φ)    = [[cos φ, 0, sin φ], [0, 1, 0], [−sin φ, 0, cos φ]]     # 3D,绕 +Y 右手
  Y₂(φ)   = [[cos φ,  sin φ], [−sin φ,  cos φ]]                    # Y(φ) 的 xz 子块,作用于 r_s_xz
  Rot2(φ) = [[cos φ, −sin φ], [ sin φ,  cos φ]]                    # 作用于 h = [cos θ, sin θ]
  ```
  **⚠ `Y₂ ≠ Rot2`,二者互为转置**(v3.0 新引入的易错点):`Y₂` 作用于按 (x,z) 序排列的水平坐标,`Rot2` 作用于按 (cos, sin) 序排列的 heading 向量。写错即产生 φ → −φ 的整段镜转。
  **一致性核对(以 θ′ = θ + φ 为准)**:`Rot2(φ)·[cos θ, sin θ]ᵀ = [cos(θ+φ), sin(θ+φ)]ᵀ` ✓;按 §2.1.1(a) 基,前向向量 = `(sin θ, 0, cos θ)`,`Y(φ)·(sin θ, 0, cos θ)ᵀ = (sin(θ+φ), 0, cos(θ+φ))ᵀ` ✓;对水平二维量按 (x,z) 序 `Y₂(φ)·(sin θ, cos θ)ᵀ = (sin(θ+φ), cos(θ+φ))ᵀ` ✓(已数值核验,残差 1.1e-16)。
- 对 canonicalize 后的量施加:`P′_world = Y(φ)·P_world`、`r_s_xz′ = Y₂(φ)·r_s_xz`、`j_v′ = Y(φ)·j_v`、`R_global′(j) = Y(φ)·R_global(j)`、`h′ = Rot2(φ)·h`;contact `f` 不变;`j_a′ = 6d(Y(φ)·R_global(j)·R_rest_global(j)⁻¹)` **对固定 rest 表重编码**(`R_rest_global` 不旋转);`j_p` 由 `P′` / `r_s_xz′` 重编码(xz 残差整体随转,y 不变)。
- VQVAE 训练:在线整段刚体 yaw 随机化(完整 kimodo 式增广)。
- **token cache 契约 = 方案 A(normative)**:每 clip 施加**一个确定性随机采样的 yaw 角 φ**(seed 与 φ 均记入 manifest)后导出单副本。**区分两个角**:φ = 采样的刚体旋转量;`θ₀′` = 施加后的结果首帧朝向,`θ₀′ = θ₀ + φ`(wrap 到 (−π, π]);`cdir = [cos θ₀′, sin θ₀′]` 随 cache 存储并作为 backbone 条件 token。**诚实声明:backbone 只见朝向的边际分布,不见同动作反事实旋转**(该能力属 K=4 多副本方案,4× 存储,列为升级路径而非本版承诺)。现行导出器两者皆无(单 latent、无 yaw、无 cdir),Phase 2 重写。
- 生成:cdir 条件直接产出目标朝向的世界动作,**无 de-canonicalize 步骤**(kimodo §4.2 同);事后场景放置(平移/整段 yaw)是应用层操作,不属表示。

## 7 归一化 N1′(normative;退化路径全部 fail-closed)

> 撤销 v2.3 的「kimodo 无归一化」错误归属 —— 官方 Stats 模块实做 mean/std。

### 7① per-rig 尺度 `s_rig`

- 定义:`s_rig(r) = ‖max_j P_rest_canonical(r,j) − min_j P_rest_canonical(r,j)‖₂`(rest 姿态全局关节位置的 AABB 对角线长,AnyTop-scaled 单位,**f64**)。
- 判据(`s_min = 1e-3`):`s_rig ≥ s_min` → 直接使用;`s_rig < s_min` → 查 per-rig 登记表 `physical_scale`(§2.1.3 S3,f64,入 `skeleton_hash`),存在且 `≥ s_min` → `s_rig ← physical_scale`;否则 **abort 并拒绝该 rig 入语料**(fail-closed,与 §2.1.3 支持域同一真源)。
- **撤销 v2.3 的「s_rig < ε 回退全局中位数」**:中位数是编造的物理尺度,且在「全部候选均退化」时仍为零。实证:382 rigs 的 `s_rig ∈ [0.9012, 5.0107]`,中位 2.1248,p1 1.2808,**< 0.1 的为 0 个** → 该回退分支在现语料上是死代码,删除无能力损失;`s_min = 1e-3` 比实测最小值低约 900×,是纯 fail-closed 触发线而非活动 clamp。
- `s_rig` 以 **f64** 存入 moments 工件与 per-rig manifest,禁止存 f32 后再反演。`ε = 1e-6` 自此**仅**用于 h 的单位圆投影(§2.1 / §7③),不再与 `s_rig` 共用。

### 7② 分组标量增益 `g_G`

- 定义域(冻结):train split(以 `split_manifest_hash` 锚定,不随 epoch/子采样变化)、`frame_mask ∧ joint_mask` 内、经 ① 之后的标量条目;组 `G ∈ {j_p, j_v, r_s_xz}`(v2.3 的 `r_s` 组因 MINOR-01 去掉 y 分量而改名并只统计 xz 两个标量 —— **副带收益**:原先把分布迥异的高度 y 与水平 xz 混入同一 pooled std,增益估计本就被高度污染)。
- **前置 finite gate(必须先于任何统计、先于任何比较式)**:`np.isfinite(x).all()`;任一非有限 → abort,报告 `(rig, clip_id, t, j, ch)`。**禁止 `nanmean/nanstd/nanpercentile` 等 NaN-aware 归约,禁止把 finite 检查寄托在任何比较式上** —— NaN 参与的比较恒为 False,`Q₀.₉₉₉ > 8` 这类判据永远不会因 NaN 触发。
- **样本量 gate**:`N_G ≥ N_min = 1e6` 个有效标量,否则 abort;`N_G = 0` 的空组走同一路径,**不存在静默跳过**。
- **std 估计**:两遍算法、f64、`ddof = 0`:先 `mu_G = Σx/N`,再 `std_G = sqrt(Σ(x−mu_G)²/N)`;分片累加用按 clip 的 f64 partial sums(**禁一遍 `Σx² − (Σx)²/N` 平方和**,灾难性抵消)。
- **std gate = 常数组 fail-closed(不 clamp)**:要求 `σ_min ≤ std_G ≤ σ_max`,`σ_min = 1e-3`、`σ_max = 1e3`;越界 → abort 并报告 `(G, std_G, N_G, mu_G)`。常数组意味着数据或管线错误,不是可归一化的情形。**放宽该 gate 须经 codex 审。**
- `g_G = 1 / std_G`,因而 `g_G ≤ g_max = 1/σ_min = 1e3`(**上限由 gate 保证,不做 clamp**)。**撤销 v2.3 的 `g_G = 1/max(std_G, ε)`**:`ε = 1e-6` 会产出 `g_G = 1e6`,与合法数值相乘即在 fp16 下溢出。
- 实测量级(300 clips / 4.44M 标量 / legacy 13ch 代理):`std(j_p/s_rig) ≈ 0.219 → g ≈ 4.6`;`std(j_v/s_rig)` 换算到 BTJD-K 的 `ΔP·fps` 约定 `≈ 0.236 → g ≈ 4.2`。故 `[σ_min, σ_max] = [1e-3, 1e3]` 两侧各留约 200× 余量。
- **正归一化** `x̃ = g_G · x / s_rig`(① 组;`h / j_a / f` 恒 `x̃ = x`),**反归一化** `x = x̃ · s_rig / g_G`。`g_G` 以 **f64** 存入 moments 工件;`moments_hash` 见 §4.3。

### 7③ 精度契约(新增,normative)

- 统计量估计(`s_rig`、`mu_G`、`std_G`、百分位)**强制 f64**。
- 归一化 / 反归一化:**至少 f32**;**禁止在 fp16/bf16 下执行** —— bf16 尾数 8 位、相对误差 ~4e-3,反归一化后直接击穿 §9 G1a 的 `1e-4·s_rig` 判据。混精训练只允许在归一化完成之后、模型输入处 cast。
- **h、rot6d 与 contact 不做任何缩放**(流形量/布尔量);h 在 decode 侧全域投影回 S¹:`h ← ‖h‖₂ ≥ ε ? h/‖h‖₂ : [1,0]`,`ε = 1e-6`。
- **反归一化必须先于 §3 恢复公式执行;§8 的 parent-edge FK 一致项按归一化态定义(用 `d̃_c`),训练路径不做反归一化;仅 §9 G2 的物理单位报告指标在反归一化后计算。**(修正 v2.3 末句,r4-MAJOR-06。)

### 7④ 百分位 gate

- 执行顺序定死:finite gate → 样本量 gate → 百分位。
- `Q₀.₉₉₉` 用 `numpy.percentile(..., method='linear')`,f64。
- **每 `(τ, G)` 组合最小样本量** `N_{τ,G} ≥ 1e4`(τ = 194 canonical topology 之一);不足 → abort,或在 manifest 的 `gate_exemptions` 表中**显式登记**(rig 清单 + 理由),该表进 `preprocess_hash`。**不得静默跳过。**
- **阈值改为 per-group 且必须实测冻结**:v2.3 的单一常数 8 有在合法数据上误 abort 的实证风险 —— 同一 300-clip 代理测量给出 `Q₀.₉₉₉(|x̃|)`:`j_p ≈ 5.98`(< 8 ✓)、`j_v ≈ 9.28`(**> 8 ✗**)。速度通道的重尾是物理性质,不是 bbox 尺度错误。规范:`thr_G` 在 Phase-2 首次全量构建时按**实测 `Q₀.₉₉₉` × 1.5 安全系数**冻结并写入 `preprocess_hash`;`{j_p: 8, r_s_xz: 8}` 作为初值;**`thr_{j_v}` 待定**(§10)。
- 备选分母(骨链总长)**仅列为未来 schema 版本 btjdk-v3.1+ 的候选,不在 v3.0 内切换**。禁止 per-axis 几何 z-score(旧方案的各向异性扭曲)。

**现有 VQVAE / codebook / cache / loss 权重 / evaluator 全部随 schema 失效**(§4)。

## 8 loss 契约(normative:逐块 loss 函数 / reduce / 权重 / 超参冻结)

> v2.3 标题「smooth-L1 分块」与 geodesic / BCE / FK 不符,已删除;j_a 的「SO(3) 测地或 6d L1」二选一已冻结为单一默认(r4-MINOR-02)。

**统一 reduce(所有块共用)**:每块先在**样本内**其 mask 作用域上求算术平均得一标量,再在 batch 内对**有效样本**(该块 mask 域非空)求平均;某块在某样本上 mask 域为空 ⇒ 该样本对该块跳过(不入分母);全 batch 为空 ⇒ 该块 loss = 0 并计数上报,同一块**连续 200 步全空 ⇒ fail-closed 报警**(疑似 mask 构造错误)。**所有 loss 归约在 fp32 中进行**(即使 autocast 为 bf16)。**smooth-L1 一律取 `β = 0.1`**:由 §7 的 `g_G = 1/std_G` ⇒ 归一化量 pooled std ≈ 1,β=0.1 即「0.1σ 以下二次、以上线性」;chordal 旋转块分量域为 [−1,1],0.1 亦为 5% 满量程,故同一常数通用。

| block | 张量 / 定义域 | loss(冻结) | mask 作用域 | 权重 |
|---|---|---|---|---|
| `r_s_xz` | rglob 归一化 [B,T,2] | smooth_l1(β=0.1) 逐分量 | frame | 10 |
| `h` | rglob [B,T,2](不归一化) | smooth_l1(β=0.1) 逐分量 + `λ_unit·(‖ĥ‖₂−1)²`,`λ_unit = 1.0` | frame | 2 |
| `j_p` | b 归一化 [B,T,J,3] | smooth_l1(β=0.1) 逐分量 | frame ∧ joint | 10 |
| `j_a` | b [B,T,J,6](不归一化) | **chordal**:`smooth_l1(R̂−R; β=0.1)` 对 3×3 共 9 分量取均值,`R̂ = GS(6d⁻¹(â))` | frame ∧ **rot_supervised** | 10 |
| `j_v` | b 归一化 [B,T,J,3] | smooth_l1(β=0.1) 逐分量(**含尾帧**,其值有定义) | frame ∧ joint | 3 |
| `f` | b [B,T,J,1] logit | `BCEWithLogits(pos_weight = w_pos)`(禁止手写 sigmoid+log) | frame ∧ **contact_supervised** | 4 |
| FK | 由 `j̃_p`、`j_a` 派生 | `smooth_l1(r_e; β=0.1)`,`r_e` 见下 | frame ∧ edge | 5 |

**parent-edge FK 一致块(normative,尺度不变形式)**

(1) 前提(依赖 §4.1):骨架工件登记 `P_rest_canonical`,派生 rest 边向量 `d_c ≜ P_rest[c] − P_rest[p]` 与 `offset[c] = R_rest_global(p)ᵀ·d_c`,同入 `skeleton_hash`。
(2) **恒等化简(实现一律按右式)**:由 §3 的 `R_global(p) = Δ_p·R_rest_global(p)`(`Δ_p ≜ 6d⁻¹(j_a[p])`)得 `R_global(p)·offset[c] ≡ Δ_p·d_c`。故 FK 项只消费 `j_a` 与 `d_c`,与 6d 行列/rest 系约定解耦(转置风险由 §9 G1b 对独立源旋转捕获,不由本项承担)。
(3) **单位(训练态不反归一化)**:由 §2.2 ch0:3 有 `j_p[c] − j_p[p] = P_c − P_p`(`r_s_xz` 精确抵消,y 本为世界绝对;已数值核验残差 = 0),故归一化后 `j̃_p[c] − j̃_p[p] = (g_jp/s_rig)(P_c − P_p)`。定义 per-rig 归一化 rest 边 `d̃_c ≜ (g_jp/s_rig)·d_c`(随 skeleton artifact 预计算;`g_jp` 或 `s_rig` 变更须重算,并使 `skeleton_hash × moments_hash` 组合失配 abort)。逐边残差:
```
e_e(t) = ( j̃_p[c,t] − j̃_p[p,t] ) − Δ_p(t)·d̃_c ,   r_e(t) = ‖e_e(t)‖₂
       = g_jp · ‖(P_c−P_p) − R_global(p)·offset[c]‖ / s_rig
```
与 r4 建议式仅差全局常数 `g_jp`(被块权重吸收),但与 `j_p` 块量纲一致且训练路径零反归一化。**同等相对骨长误差在大小 rig 上给出相同 loss/梯度**(v2.3 的「先反归一化再 FK + 固定权重 5」使残差处于物理长度单位、大 rig 被加权,**废止**)。
(4) **reduce**:`L_FK = mean_{i∈有效样本}[ mean_{(t,e)∈frame∧edge} smooth_l1(r_e; β=0.1) ]` —— **先样本内 masked mean(逐边等权 ⇒ 高 J rig 不因边多占更大权重),再 batch 内对有效样本平均**。梯度**同时**回传 `j_p` 与 `j_a`(不设 stop-grad;冻结选择,变更须改 `loss_contract_hash`)。边有效性:`edge_mask` 要求父子端点 `joint_mask` 均为 1。
(5) **零长度边**:`ℓ_rest(e) ≜ ‖d_c‖₂`;`ℓ_rest < ε_ℓ`(`ε_ℓ ≜ 1e-4·s_rig`)的边**保留在 `edge_mask`**(约束仍良定:两关节须重合;本形式不除以边长故无奇异),仅在 §9 G2 的相对指标中排除并单独计数上报。
(6) **刚体性前置断言**(FK 项与 G2 的成立前提):源若含 joint 平移通道则边长时变、残差不应为零。构建期逐 clip 逐边检查 `max_t |‖P_c−P_p‖ − ℓ_rest| / s_rig < τ_rigid`;违反边从 `edge_mask` 移除并记入 per-clip manifest,违反边占比 > 5% 的 clip 拒收。**`τ_rigid` 待定**(§10)。

**j_a 冻结与理由(替换 v2.3 的「SO(3) 测地或 6d L1」二选一)**:默认 = **chordal(Frobenius 型)**。(a) 对 R 而非对 6d 计,与 6d 参数化的非唯一性无关;(b) 无 arccos 奇异(测地 loss 在 θ→0 与 θ→π 梯度病态);(c) 单调于测地角(`‖R̂−R‖_F = 2√2·sin(θ/2)`,已数值核验),与 §9 的 geodesic **指标**可互相换算 —— 故「loss 用 chordal / 指标与 gate 用 geodesic」不是两个目标,而是同一序关系的两种读数。**6d-L1 与 geodesic-loss 一并降级为 Phase 3 消融项**,切换须改 `loss_contract_hash`。

**GS 稳定 clamp(冻结)**:给定 `â = [a₁; a₂] ∈ R³×R³`,
```
u₁ = a₁ / max(‖a₁‖₂, ε_gs);  w = a₂ − ⟨u₁,a₂⟩·u₁;  u₂ = w / max(‖w‖₂, ε_gs);  u₃ = u₁ × u₂
R̂  = [u₁ u₂ u₃]        # 列向量、主动旋转,逐元素 codec 见 §2.3
ε_gs = 1e-6
```
用 `max(·, ε_gs)` 而非 `+ε`,以保证 `‖a‖ ≫ ε` 时严格无偏。训练中监控**退化 6d 率** `= P(‖a₁‖₂ < 10·ε_gs ∨ ‖w‖₂ < 10·ε_gs)`:GT 侧必须为 0,模型侧 > 0.1% 写 warn。
**geodesic 指标(仅报告/gate,不入梯度)**:`θ_geo = 2·arcsin(clamp(‖R̂−R‖_F/(2√2), 0, 1))`(与 §9 G1b 同式)。

**h 的 unit-circle 约束(补齐)**:训练侧**不做硬投影**(硬投影使梯度在 `‖ĥ‖→0` 处发散),改为上表的径向惩罚 `λ_unit·(‖ĥ‖₂−1)²`,`λ_unit = 1.0`(与块权重 2 相乘后等效权重 2);`‖ĥ‖₂` 用 `max(·, ε)` 计算,`ε = 1e-6`。**decode 侧仍执行 §7③ 的全域硬投影**;分工写死:**训练 = 软约束,推理 = 硬投影**。
**零向量率监控(normative)**:每次 eval 统计并写入日志/报告 —— `ρ₀ ≜ P(‖ĥ‖₂ < ε)`(触发 fallback `[1,0]` 的比例)与 `ρ_low ≜ P(‖ĥ‖₂ < 0.5)`。**GT 侧 `ρ₀ = ρ_low = 0`(硬 gate)**;模型侧初值 `ρ_low > 1%` warn、`> 5%` 判 gate-fail,**该两阈值待标定**(§10)。

**contact 类均衡**:`w_pos` **按 per-rig 估计**(估计集合 = train split × 该 rig 全部 clip × `contact_supervised_mask` 内条目),`w_pos(r) = clamp(N_neg(r)/N_pos(r), 1, 20)`。**不得用全局正类率**:蛇 rig 的正类率近 1(Anaconda `root_on_rate = 0.9855`、KingCobra 0.9412,且 27/27、19/19 关节全合格),四足 rig 稀疏,全局估计会让蛇主导 contact 梯度。`w_pos` 表随 moments 工件存储并进 `moments_hash`。上限 20 为守卫(防极稀疏语料使正样本梯度爆炸)。**数值待 Phase-2 实测**(§10)。

**`loss_contract_hash`(normative,§4 通用分区之外的独立字段)**:
```
loss_contract_hash = sha256(§4.2 envelope of {
  "repr_schema_version", "semantic_hash",
  "blocks": { <block>: {"loss_fn": <枚举 id>, "reduce": "sample_masked_mean→batch_mean",
                        "weight": <f64>, "mask_scope": [...], "hparams": {...}} },
  "globals": {"beta":0.1, "eps":1e-6, "eps_gs":1e-6, "lambda_unit":1.0,
              "w_pos_mode":"per_rig", "loss_dtype":"fp32", "fk_stop_grad":false} })
```
写入 **ckpt、train log 首行、experiment manifest**;resume 时与 ckpt 内值逐字段核对,不一致即 abort(除非显式传 `--allow_loss_contract_change`,该 flag 必须同时写入 log 与新 ckpt,使 ablation 可溯源)。权重(10/2/10/10/3/4/5)沿用 kimodo Eq.1 起点,Phase 3 实调后回写。

## 9 验证 gates

### 9.0 总则

gate 分四层,**依赖顺序不可颠倒**:
```
L0 codec 自检        : G-6d                                   (无需数据,先于一切)
L1 源与骨架契约      : G-motion-rot → G-rest → G-skel-c → G-skel-m → G-heading-A/B/C/F
L2 构建期逐 clip     : G-resample、G-contact、G-det、G-heading-D/E
L3 往返与一致性      : G1a–G1g、G2-a/b/c、G-cpu1、G-cpu2
L4 视觉裁决          : G7(优先级高于一切数值阈值)
```
所有 gate **fail-closed**:不过即 abort,不得 warn-and-continue、不得跳过、不得以「总体通过率」替代逐项判据。**禁止跨 sub-gate 取平均。** 任一 gate 出现 NaN/Inf 即 FAIL(禁止 `nanmax`/`nanmean` 等规避写法)。

### 9.1 G-6d(L0,codec 冻结自检;**一律 fp64**)

六组用例全过,方可实现 j_a 通道。

| 用例 | R | 冻结 `6d(R)`(列约定) | 行约定会得到 | Linf 判别余量 |
|---|---|---|---|---|
| GOLD-1 | `Rx(90°) = [[1,0,0],[0,0,−1],[0,1,0]]` | `[1, 0, 0, 0, 0, 1]` | `[1, 0, 0, 0, 0, −1]` | 2.0 |
| GOLD-2 | `Ry(90°) = [[0,0,1],[0,1,0],[−1,0,0]]` | `[0, 0, −1, 0, 1, 0]` | `[0, 0, 1, 0, 1, 0]` | 2.0 |
| GOLD-3 | `Rz(90°) = [[0,−1,0],[1,0,0],[0,0,1]]` | `[0, 1, 0, −1, 0, 0]` | `[0, −1, 0, 1, 0, 0]` | 2.0 |

判据:`‖6d(R) − 冻结值‖_∞ < 1e-12`。

**GOLD-4(复合非对称,抓「整体转置」而非仅「角度变号」)**:`R = Rz(30°)·Ry(20°)·Rx(10°)`
- 列约定冻结值:`[0.813797681349, 0.469846310393, −0.342020143326, −0.440969610530, 0.882564119259, 0.163175911167]`
- 行约定会得到:`[0.813797681349, −0.440969610530, 0.378522306370, 0.469846310393, 0.882564119259, 0.018028311236]`(Linf 差 **0.9108159209**)
- 判据 `‖·‖_∞ < 1e-12`。**单轴用例的转置等价于角度变号,GOLD-4 的转置不等于任何单轴旋转,二者必须并存。**

**GOLD-5(非正交输入,锁死 Gram–Schmidt、排除极分解 / 排除「先正交后归一」)**:
- `6d⁻¹([1,0,0, 1,1,0]) = I`(判据 Linf < 1e-12)。**极分解实现在此例给出与 I 相差 Linf 0.4472 的矩阵。**
- `6d⁻¹([2,0,0, 0,3,0]) = I`(锁死「先归一化 a1」)。

**GOLD-6(随机 SO(3) 往返)**:N = 100,000,`numpy.random.default_rng(12345)` 采样单位四元数转矩阵,fp64:
1. `‖6d⁻¹(6d(R)) − R‖_∞ < 1e-12`(实测 max **1.221e-15**);
2. `‖6d(6d⁻¹(d6)) − d6‖_∞ < 1e-12`,`d6 = 6d(R)`(实测 max **1.110e-15**);
3. 测地误差 `< 1e-6 rad`(实测 max **4.215e-08**)。

**fp32 不适用上述阈值**:同一实验在 fp32 下元素 Linf max **2.384e-07**、测地 max **4.814e-04 rad**(arccos 在 tr≈3 处病态,测地误差 ≈ √元素误差)。故:**codec golden 判据一律 fp64;运行时 fp32 路径的对应阈值另取「元素 Linf < 1e-5、测地 < 2e-3 rad」,且测地量始终提升到 fp64 计算。阈值表必须与精度绑定书写**(若误在 fp32 下套 1e-12,会 100% 假失败)。

**G-6d 不能证明什么(必须并列声明,防 false-PASS)**:编解码互逆 + 全部 golden 通过,仍**不能**证明 `R_rest_global` 表本身正确 —— 编码侧与解码侧共用同一张错表时,rest-delta 恒等自检恒过。`R_rest_global` 的正确性只能由**不经过 6d 的独立量**判定,即 **G-skel-m** 的 parent-edge 位置残差。两者必须都过,缺一不可宣称 j_a 正确。

### 9.2 G-motion-rot(L1,前置硬性 pre-build gate)

**目的**:`j_a` = 真全局 rest-delta 旋转,只能由**逐帧原始旋转**构造。本 gate 冻结「原始旋转从哪来、怎么读、怎么核验」。**legacy 13ch 不是合法输入**(§9 G-motion-rot.6)。

**语料覆盖表见 §2.2A。**
**⚠ TrueBones 必须用 AnyTop 的 `truebones_processed/bvhs`,不得用 raw `Truebone_Z-OO`**:raw 的 1,153 个 BVH 是**另一套约定**(ROOT 与**每个** JOINT 都是 `CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation`,即 **ZXY** 轴序 + 逐关节平移通道;`Frame Time 0.0333333` = 30 fps),抽样 60/60 一致;与我们 motions 的 1:1 来源不符,且逐关节平移意味着骨长逐帧可变。

#### G-motion-rot.1 BVH 读取规范(PZ 与 TrueBones **共用同一读取器**)

两源通道规格完全一致(各抽样 60 clip,100% 命中):
- `ROOT` : `CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation`
- `JOINT`: `CHANNELS 3 Zrotation Yrotation Xrotation`
- `Frame Time 0.041667`(**native_fps = 24.0**);角度单位 = **度**,读入后必须 `deg2rad`。

**局部旋转(列向量、主动旋转、内旋 Z-Y-X)**:
```
R_local(j,t) = Rz(z_jt) · Ry(y_jt) · Rx(x_jt)
Rz(a)=[[c,-s,0],[s,c,0],[0,0,1]]   Ry(a)=[[c,0,s],[0,1,0],[-s,0,c]]   Rx(a)=[[1,0,0],[0,c,-s],[0,s,c]]
```
**全局递推(按 HIERARCHY 的 DFS 顺序,父先于子)**:
```
R_glob(root,t) = R_local(root,t)          P_world(root,t) = (Xposition, Yposition, Zposition)(t)
R_glob(c,t)    = R_glob(p,t) · R_local(c,t)
P_world(c,t)   = P_world(p,t) + R_glob(p,t) · OFFSET(c)
```
`OFFSET(root)` **不参与** root 位置。MOTION 段通道按 HIERARCHY 的 DFS 出现顺序逐节点消费,读完必须断言 `consumed_cols == n_cols`(fail-closed)。

**End Site 提升(source-specific,normative)**:
- **TrueBones**:End Site **提升为真关节**,`R_local ≡ I`(该节点无 CHANNELS),其 `OFFSET` 参与父→子边。命名解析顺序:① `End Site #name: <N>` 的 `<N>`;② 回退 `<parent_name>_end_site`;③ per-rig 显式覆盖表。实测:70 个 TB rig 中 66 个走规则 ①;仅 **Ant / Crab / Deer / Jaguar** 4 个 rig(共 28 个关节)走规则 ②;当前语料**不需要**规则 ③(但须在全量 M6 后才可宣布覆盖表为空,§10)。
- **PZ**:End Site **不提升**(现行 311 个 PZ rig 的 cond 中名为 `*_end_site` 的关节数 = **0**,已全量核验)。

#### G-motion-rot.2 SMPL-272 读取规范(human)

- 输入 `x = np.load(motion_data/<key>.npy)`,`[T30, 272]`,**native_fps = 30.0**。
- 逐关节**局部** rot6d:`x[:, 140:272].reshape(T30, 22, 6)`(含 pelvis 行,即完整 3-DOF 全局朝向,非从 263 反推)。
- 6d→R 用 §2.3 冻结的 `btjdk_6d_decode`。**上游 272 是行约定**(`scripts/_v4_build_from_272.py:49 decode_6d_rows`),故接入时必须**显式转置**并在 manifest 记 `source_6d_convention = "row"`;约定写反会使 M4 爆表(不会静默通过)。
- 全局递推:`R_glob(0) = R_local(0)`;`R_glob(j) = R_glob(parent(j)) · R_local(j)`;
  `parents = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]`。
- 重采样 30 → `fps_tgt`:按 §5 S1(逐关节 Slerp + 位置线性);**先重采样、后算 j_v**。

#### G-motion-rot.3 joint mapping(锚定,normative)

canonical joint name 表 = `cond[rig]["joints_names"]`(FK 序)。映射**按名字**,绝不按下标:
```
src_node[i] = 源层级中名为 joints_names[i] 的节点
              (BVH: ROOT / JOINT / 提升后的 End Site ; 272: SMPL-22 固定名表)
```
已全量核验(311 个 PZ rig,每 rig 一个 clip):**子树闭合 0 违例**;`cond[rig]["parents"]` 与源层级(限制到保留集)**完全一致 0 违例**;`joints_names` 顺序 == 源 DFS 顺序过滤保留集 **0 违例**;`cond[rig]["offsets"][1:]` 与源 `OFFSET` 最大绝对差 **5.1e-7**(PZ)/ **5.0e-7**(TB):**无缩放、无坐标变换、无单位换算**。
每 rig 落表 `source_joint_map`(§4.1 字段 20),整表进 `skeleton_hash`。

#### G-motion-rot.4 per-clip manifest 必含字段

见 §5 末的 per-clip manifest 清单(`source_kind` / `source_path` / `source_sha256` / `source_frames` / `native_fps`(**逐 clip 从文件读出,不得写死**)/ `angle_unit` / `euler_order` / `matrix_convention` / `compose_order` / `rot6d_codec_id` / `source_6d_convention` / `joint_map_hash` / `endsite_promoted`)。

#### G-motion-rot.5 pass 判据(逐 clip,全部 fail-closed)

设 `s_bone = mean_{t,c} ‖P_world(c,t) − P_world(p,t)‖`(源单位平均骨长;`s_bone ≤ 0` 直接 abort)。

| # | 判据 | 阈值 | 依据(实测) |
|---|---|---|---|
| M1 | 有限性 | `R_glob` 全部 finite | — |
| M2 | 正交性 | `max ‖RᵀR − I‖_∞ < 1e-6` | BVH 路径 ~1e-15 |
| M3 | 行列式 | `max |det(R) − 1| < 1e-6` | 同上 |
| M4 | **FK 位置 vs 源 P_world** | `max_{t,j} ‖P_FK − P_src‖ / s_bone < 1e-4` | 25 PZ clip ≤ **3.26e-5**;60 TB clip ≤ **1.95e-5** |
| M5 | 通道消费完整 | `consumed_cols == n_cols` | — |
| M6 | 名字映射完备 | `joints_names` 全部命中源节点,无重名、无遗漏 | PZ 311/311、TB 60/60(**须全量重跑**,§10) |
| M7 | fps 一致 | `native_fps` == 该 `source_kind` 的期望值,否则进人工清单 | PZ/TB 24.0、272 30.0 |
| M8 | 子树闭合 | 每个保留关节的源父节点也在保留集内 | PZ 311/311,0 违例;TB 须全量跑 |

**M4 是唯一能抓出转置 / 轴序 / 单位 / 乘法顺序错误的判据**:`6d(I) = [1,0,0,0,1,0]` 在行/列两种约定下相同,恒等自检抓不到;而任一约定写反会让 M4 立刻 ≫ 1e-4。
**注**:M4 以 `cond[rig]["offsets"]` 为 FK 参照。若 G-rest 从 raw 重建后改动了 rest offsets,M4 的参照必须同步换成新 offsets,否则会全语料误报失败。

#### G-motion-rot.6 与 legacy 13ch 的关系(normative,禁止项)

**13ch 不是合法的 j_a 输入。** 实测(25 PZ + 60 TB + 4 human clip,与源逐帧比对):
- 13ch 的 ch3:9 只承载「**有子关节的**关节的局部旋转」,存放在其**子关节行**里(`R_local(parent(j)) = 6d⁻¹(row_j)`)。故由 13ch 重建的**非叶**关节全局旋转与源**精确一致**:测地误差 max **7.82e-6°**(PZ)/ **3.82e-6°**(TB)/ **0.0000°**(human)。
- **叶关节自身的局部旋转在 13ch 中没有载体行,完全丢失**(量化见 §2.2B)。
- 13ch 世界系与源世界系之差**恰为逐帧纯 yaw**(`|R[1,1] − 1| ≤ 1.11e-15`,BVH 路径;human 路径 ≤ 2.8e-4,为 Slerp 重采样噪声),故位置与非叶旋转的差异只是一个**已知的 yaw**,不是信息缺失。

⇒ 由 13ch 或关节位置反推叶关节旋转 = 把源里存在、编码里被丢弃的自由度当作学习目标,正是 `handoff/20260630_190042_human_rot6d_data_encoding_lessons.md` 记录的 ill-conditioned target。**禁止用 IK / 位置反推补齐任何旋转自由度。**

### 9.3 G-rest(L1,前置)

per-rig `R_rest_global` / `P_rest_canonical` 从 raw 源重建为与 motion 同世界系的真全局 rest 帧。
- **来源约定冻结**:BVH raw rest 优先,且必须是**与 motion 同一批 stage-2 BVH 文件**(不是别处的资产绑定姿态);资产绑定姿态仅在先对齐世界系后可用(PZ 资产系中位 87.48° / 最大 179.86° 偏移,不可直用)。
- **位置与旋转同源同帧**(§2.2 新增契约条款);不同源会被 G-heading-B 抓成常数角偏移。
- 镜像 rig(~25 个)约定显式登记(`lr_pairs`,§4.1 字段 13)。整表哈希写 `skeleton_table_hash`(不再是 `rest_table_hash`)。
- **pass 判据**:恒等自检 —— 每 rig 以 rest 姿态计算 `6d(R_global · R_rest_global⁻¹)`,恒等目标 `= 6d(I) = [1,0,0,0,1,0]`,误差 = per-joint 6 分量 **L∞**,聚合 = 全 rig × 全 valid joint 取 max,阈值 **max < 1e-4**。
- **⚠ 该判据的已知盲区(诚实声明)**:编码侧与解码侧共用同一张错表时它恒过(r4-BLOCKING-04)。**它不是 rest 表正确性的证据**;正确性由 **G-skel-m** 与 **G-heading-B** 提供。未过则 j_a 通道冻结不实现。

### 9.4 G-skel-c(L1,结构一致性 —— 只抓错位/损坏,不是正确性证明)

每 rig 全过,任一不过即 abort:
1. **置换自洽**:`old_to_new_perm[new_to_old_perm[i]] == i` 且反向亦然,∀i。
2. **名单一致(最强且最便宜的 perm gate)**:`[joint_names_source[new_to_old_perm[i]] for i in range(J)] == joint_names_canonical`,**逐个字符串相等**。381/382 非恒等 perm 的任何错用都在此暴露。
3. **FK 序**:`parents[0] == −1` ∧ `∀j>0: 0 ≤ parents[j] < j` ∧ 从 0 出发 BFS 覆盖全部 J 个关节。
4. **rest 重建(rest-edge gate)**:`P̂[root] = P_rest_canonical[root]`;按 c 升序 `P̂[c] = P̂[parents[c]] + R_rest_global(parents[c])·offset[c]`。判据 `max_j ‖P̂[j] − P_rest_canonical[j]‖_∞ < 1e-9 · s_rig`。**诚实声明:该式由 `offset` 的定义反解得到,是一致性 / 序列化 / 置换检查,不是 rest 表正确性证明。**
5. **rest 旋转合法**:∀j,`‖R_rest_globalᵀ·R_rest_global − I‖_∞ < 1e-9` 且 `det > 0`。
6. **镜像**(仅 `lr_pairs` 非空):`‖M·R_rest_global(L)·M − R_rest_global(R)‖_∞ < 1e-3` **并且新增位置项** `‖M·(P_rest[L] − P_rest[root]) − (P_rest[R] − P_rest[root])‖_∞ < 1e-3·s_rig`(纯旋转项对近对称姿态判别力弱)。
7. **anchor 解析**:`joint_names_canonical[heading_anchor_indices[k]] == heading_anchor_names[k]`(k=0,1),且 `new_to_old_perm[heading_anchor_indices[k]] == anchor_*_idx_raw`(与校准 TSV 交叉核对)。
8. **contact**:`len(contact_eligible_source) == J` 且取值 ⊆ {0,1};其 root 位取值不受本 gate 约束(归 §2.2)。

### 9.5 G-skel-m(L1,**非循环**正确性 —— 唯一能同时证伪 6d 转置 / R_rest 错 / offset 参考系错 / perm 错用 / parents 错 的判据)

对每 rig 抽样 `min(20, 该 rig 全部 clip)` 个 clip、每 clip 全帧,取**源侧独立量**:源 world 位置 `P_src` 与源侧全局旋转 `R_src_global`(由 G-motion-rot 提供),计算逐边归一化残差
```
res(c,t) = ‖ ( P_src[t,c] − P_src[t,parents[c]] ) − R_src_global[t,parents[c]] · offset[c] ‖₂ / s_rig
```
统计 per-rig 的 p50 / p99 / max 并落表(进 manifest)。判据分两级:
- **硬失败线(现在即可冻结)**:`max_{c,t} res > 1e-2` → abort。任何约定级错误(转置、参考系错、perm 错、parents 错)的残差都在 `O(1)·rest_edge_len / s_rig` 量级,远超此线。
- **紧阈值:待定(不编数,§10)**。对 BVH-native 源族(PZ / TrueBones)**提议**冻结 `max res < 1e-6`;对 272-derived 人类源族无法先验给数,须先在 ≥200 clip/源族上跑上式给出 p50/p99/max 分布,再按 `ceil_1sig(10 × p99)` 冻结。在测量落表前,人类源族**只走硬失败线**,manifest 标 `tight_gate_status = "pending_measurement"`,且在此状态下**不得宣称 j_a 精度**。
- **依赖声明**:本 gate 需要 `R_src_global`。**在 G-motion-rot 落地前 G-skel-m 不可执行,j_a 通道必须保持冻结** —— 否则会出现「G-skel-c 过了就以为骨架对了」的 false-PASS。

### 9.6 G-heading 族(L1/L2)

**G-heading-A(覆盖性状态机,机检,硬性 pre-build gate)** —— 取代 v2.3 的「104 rig 清零」

**(a) 记录完备性**:构建前读取 `heading_table.json`,以 rig 名为主键。
```
STATUS = {AUTO_OK, MANUAL_CONFIRMED, MANUAL_CORRECTED, EXPLICIT_METADATA, REJECTED}   # 闭集,无默认值
assert set(heading_table.keys()) == set(corpus_manifest.rigs)          # 逐名相等,不是计数相等
assert all(rec.status in STATUS for rec in heading_table.values())
assert all(rec.status == REJECTED or (rec.method in {limb-pair, spine-axis, explicit}
           and abs(norm(rec.u_rig) - 1) < 1e-9 and rec.pol in {+1, -1}) for rec in ...)
assert 语料中属于 status == REJECTED 的 rig 的 clip 数 == 0             # 必须实际剔除,不只是标注
```
任一断言失败 → 硬 abort。

**(b) AUTO_OK 的机检充要条件(全部满足才可自动通过)**
1. `probe_verdict == "OK"`(7 值闭集 {OK, NO_MOVING_CLIP, LOW_EVIDENCE, AMBIGUOUS, MIXED, SIDEWAYS, FLIPPED},实测恰好划分 382 rig:252+64+7+1+39+15+4 = 382);
2. `probe_n_clips ≥ 2` **且** `probe_n_move_frames ≥ 10`(= `params.min_move_frames`)—— **本条直接排除全部 64 个 `NO_MOVING_CLIP` rig**(实测其 `probe_n_clips == 0` 且 `probe_n_move_frames == 0`);
3. `needs_review == False`;
4. `calib_R ≥ 0.9`(= `params.good_R`;v2.3 隐含的 0.8 门槛抬到 0.9,是有意的保守取舍);
5. `method == limb-pair` 时 `pair_phi_spread_deg ≤ 30`(= `params.spread_review_deg`);`spine-axis` 时该字段为 NaN,改由 G-heading-B 背书;
6. `calib_pose_only == False`;
7. **G-heading-B 通过**。

**(c) 为什么必须废弃「104 rig 清零」口径** —— 见顶部「撤销 5」。按 (b) 在现表上实测:**AUTO_OK 228 rig / 需人工决定 154 rig**,后者 = 104(`needs_review`)∪ 43(未被 needs_review 覆盖的 `NO_MOVING_CLIP`)∪ 7(`calib_R ∈ [0.8, 0.9)` 的边界 rig:`PZ_B_W_Ruffed_Lemur_Male 0.874 / PZ_Capuchin_Monkey_Juvenile 0.869 / PZ_Giant_Anteater_Juvenile 0.804 / PZ_Highland_Cattle_Male 0.816 / PZ_Indian_Elephant_Female 0.868 / PZ_Indian_Elephant_Male 0.893 / PZ_Ring_Tailed_Lemur_Male 0.842`)。
43 个「静默通过」rig 全名单:`Bat, Bear, Bird, Buzzard, Camel, Cat, Centipede, Chicken, Comodoa, Crab, Cricket, Crocodile, Deer, Dragon, Eagle, Elephant, Flamingo, Gazelle, Giantbee, Hamster, HermitCrab, Hippopotamus, Horse, Hound, Isopetra, KingCobra, Monkey, Ostrich, PZ_Blue_Wildebeest_Male, PZ_Red_River_Hog_Female, Parrot, PolarBear, PolarBearB, Pteranodon, Raindeer, Rat, Rhino, SandMouse, Scorpion-2, SpiderG, Tricera, Tukan, Turtle`(其中 41 个为非 PZ_ 的 TrueBones rig)。
**⚠ 154 / 228 是基于现表的实测值;按 §2.1.1(e) 重跑 probe 后须重新统计并复核,不得当成固定承诺。**

**(d) 非 AUTO_OK 记录的必含证据字段(缺一即 abort)**
`reviewer`(人名/ID)、`decided_utc`、`evidence_clips`(≥ 1 个 clip 文件名)、`evidence_render`(路径 + `sha256`)、`decision`(≥ 1 句自由文本理由)。`MANUAL_CORRECTED` 另必含 `before` / `after` 两个子对象,各记 `anchor_left` / `anchor_right` / `spine_tip` / `pol` / `method`。
**零运动证据 rig 的专门规定(`probe_n_clips == 0`,64 个)**:只能取 `MANUAL_CONFIRMED` / `MANUAL_CORRECTED` / `EXPLICIT_METADATA` / `REJECTED`,**不得**以「不在 104 清单里」为由通过;其 `evidence_render` 必须是**真全局 rest 姿态 + heading 箭头叠加**的渲染,且因这 64 个 rig 在语料中全部有 clip,还须附一段该 clip 的箭头叠加**动画**(遵循项目「可视化优先 / 动作类必用多帧」规则,静帧不算证据)。
**已核实拓扑继承不可行,故不设该机制**:64 个零运动 rig 中与某个 AUTO_OK rig 同 `topo` 的**数量为 0**(194 个 canonical topology 中 131 个为单 rig 拓扑),不存在可用 donor。

**(e) 决策入哈希**:`heading_table_hash` 的输入必须包含每条记录的 `status`、`pol`、`method`、`anchor_*`、`spine_tip`、`u_rig`、`phi_rest_deg`、`reviewer`、`decided_utc`、`evidence_render.sha256`、`decision` 文本。任何评审决定的变更 ⇒ 哈希变更 ⇒ 全部既有工件失配 abort。**因此评审必须一次性做完再进入构建,不能边构建边补评审。**

---

**G-heading-B(锚点-rest 互证,机检,前置)**
对每 rig 的校准 clip,逐帧计算**纯位置派生**的 `f_anchor(t) = (P(L,t) − P(R,t)) × e_y`(spine-axis rig 用 `P(tip,t) − P(root,t)`),与 §2.1.1(c) 的 `f(t) = D_root(t)·u_rig` 比较;在 `n(t) ≥ ε_h` 且 `‖f_anchor,xz‖ ≥ 0.5·‖f_anchor‖` 的帧上取圆均值:判据 `|circmean(wrap(θ_anchor − θ))| ≤ 30°`(= `params.spread_review_deg`)且圆一致度 `R ≥ 0.9`(= `params.good_R`),否则该 rig 不得进入构建。
**为何该 gate 有效**:`f_anchor` 只用位置、与 `R_rest_global` 的**旋转**无关;`f` 用 `R_rest_global` 的旋转。若 rest 根旋转被整体污染 E(即误用 `R_rest·E`),则 `f′ = D·(R_rest E⁻¹ R_rest⁻¹)·u_rig`,表现为**恒定角偏移**,必被本 gate 抓到 —— 这同时补上了 G-rest 恒等自检的盲区(§9.3)。

**G-heading-C(h 与 j_a[root] 的一致性,训练/生成侧)**
h 在新定义下是 `j_a[root]` 的确定性函数 ⇒ GT 内存在冗余,生成时模型可能输出互不自洽的 `(h, j_a[root])`。判据:`|wrap(θ(h) − θ(D_root·u_rig))|` 的 **p99 < 5°**(**该阈值为工程默认,待定**,§10;GT 中该误差应恒为 0),且训练期逐 step 监控其均值。**权威值声明**:h 为条件与评测的权威值。

**G-heading-D(退化区间 fail-closed,逐 clip,构建期;三条独立、任一不过即丢弃该 clip 并计入名单)**
- **D1 频度**:`heading_deg_frac ≤ 0.05`。
- **D2 hold-last 合法性(核心判据)**:对每个内部极大退化区间 `[t0, t1]`(`0 < t0 ≤ t1 < T−1`),要求 `|wrap(θ_raw(t1+1) − θ_raw(t0−1))| < 30°`。**这才是 hold-last 是否无损的真判据**:只有当真实 heading 在退化期间基本未变时,沿用上一有效值才不引入 π 级错误。
- **D3 端点外推**:首段(`t0 > 0`)与末段(`t1 = T−1`)的退化长度 `≤ round(0.25 · fps_tgt)` 帧 —— 端点退化没有「区间两侧」可供 D2 验证,只能限长。
- 被丢弃的 clip 名单 `heading_degenerate.tsv`(含 §2.1.2(e) 四个诊断字段)与其计数写入 build manifest,并在构建日志中显式上报(**fail-loud,不得静默过滤**)。
- **丢弃率 > 0.5% 或任一 rig 丢弃率 > 20% 触发人工裁决,不得自动执行。**

**G-heading-E(yaw 等变性,机检)**:随机抽 64 个 clip × 8 个随机 φ,核验 `max_t |wrap(θ_先增广(t) − θ_后增广(t) − φ)| < 1e-6`,同时 `deg_先增广 ≡ deg_后增广`(逐帧布尔相等)。

**G-heading-F(支持域 golden fixtures,构建期必跑,fail-closed)**
- **合成 rig(4 个,人造,不入训练集)**:`J=1`;`J=2 链`;`J=5 星(全部子关节名无 L/R 语义)`;`rest 脊柱竖直的人形`。判据(逐个断言,期望值写死):前三个在无 S3 元数据时 `status == REJECTED`、给了 S3 元数据后 `status == EXPLICIT_METADATA` 且 `‖u_rig‖ = 1`;第四个必须**被 S2 拒绝**(水平性判据不过)而**被 S1 接受**。
- **语料内困难 rig(4 个,真实)**:`PZ_Grey_Seal_Juvenile`(`pair_phi_spread_deg = 156.76`)、`Spider`(179.99999674745015,锚点对几乎完全反向)、`PZ_Spotted_Hyena_Male`(`probe_verdict = FLIPPED`,`n_ok = 4 / n_flip = 19`)、`Bat`(`NO_MOVING_CLIP` 翼类且 `needs_review == False`)。判据:四个**都不得**取得 `AUTO_OK`(分别被 (b)5、(b)5、(b)1、(b)1+2 拦下)—— fixture 直接验证 G-heading-A 的判据确实能拦住已知的四类失败模式。
- fixture 结果与其断言逐条写入 build manifest,任一条不符即 abort。
- **承诺边界(必须原样写出)**:v3.0 只承诺对 J=1 / 链 / 星「**fail-closed 地拒绝**」,**不承诺支持**这些拓扑。

### 9.7 构建期确定性 gates(L2)

- **G-resample**:对 `fps_src == fps_tgt` 的全部 clip,S1 走恒等旁路,`P1` 与 `P0` **逐位**相同(非近似)。
- **G-contact**:在 `fps_src == 20` 的子集抽 ≥1000 clips,S3 重算的 `f` 与 legacy 13ch `ch12` **位级一致**,mismatch 率 = 0;`fps_src ≠ 20` 的 clip 不入该门,其 legacy 位存 sidecar 供审计并显式声明不可比。(该门在 §5.9 变更 4 之后仍成立:contact 对 yaw 不变。)
- **G-det**:同一 clip、同一 `preprocess_hash`,在两台机器/两次运行下构建,离线工件 sha256 逐位相同。

### 9.8 G1 往返(L3,normative,**分块 fail-closed**;取代 v2.3 的一句话判据)

**评测集**:194 canonical topologies × 382 rigs,每 rig ≥3 clips(最长/最短/随机;不足 3 则全取),外加 G-heading-F 的 golden fixtures。全部 sub-gate 在 **fp64** 中计算(输入为实际 f32 存储值)。任一 sub-gate FAIL ⇒ G1 FAIL ⇒ 不得进入下一 Phase。结果写 `g1_report.json`(逐 rig × 逐 sub-gate 的 mean/p99/max + pass/fail + 采样 clip 清单),`g1_report_hash` 入 manifest。
**注(与 §4.3 协调)**:`g1_report_hash` 是「gate 结果」类哈希,与「数据内容」类哈希分开管理 —— 重跑 gate 会改前者,不得因此判定数据失配。

- **G1a 位置往返**(编码→反归一化→§3 恢复 vs 源 `P_world`):`e_pos(t,j) = ‖P̂_j − P_j‖₂`,域 `frame ∧ joint`。判据(两条同时):`max e_pos < max(1e-6·U, 1e-4·s_rig)`(`U` = length_unit 的一个单位,即绝对下限;**修正 v2.3 的 `<1e-4×bbox` 在 bbox→0 时要求误差严格小于零、理论不可通过**)且 `mean e_pos / s_rig < 1e-5`。说明:该往返代数上恒等,残差纯为 f32 量化(相对精度 ~6e-8),故阈值取紧;超阈即语义错误而非精度问题。现语料 `s_rig ≥ 0.9012` ⇒ `1e-4·s_rig ≥ 9.0e-5 ≫ 1e-6`,绝对下限只对未来退化 rig 生效。
- **G1b 旋转往返(对独立源旋转,非 repr→repr)**:参照 = G-motion-rot 给出的源逐帧全局旋转 `R_src`;`R̂ = 6d⁻¹(j_a)·R_rest_global`。测地角定义冻结:
  `θ_geo(R̂,R) = 2·arcsin( clamp( ‖R̂−R‖_F/(2√2), 0, 1 ) )`(等价 `arccos((tr(R̂ᵀR)−1)/2)`,但 θ→0 与 θ→π 均稳定;已数值核验三例逐位一致)。
  判据:`max θ_geo < 1e-4 rad` 且 `mean < 1e-6 rad`;并 `max‖R̂ᵀR̂−I‖_∞ < 1e-5`、`min det(R̂) > 0.999`。域 = `frame ∧ rot_supervised`。
  **必须对源比较**:encode→decode 用同一 6d 约定,自比无法暴露转置。**若源旋转契约缺失 ⇒ 本 sub-gate fail-closed(不得跳过)。**
- **G1b2 旋转→FK 位置**(转置/约定错误的第二道独立捕获):由 `R̂` 与 rest 骨架沿 parent 链遍历重建 `P_FK`(root 位置取 GT),对源 `P_world` 比较:`mean‖P_FK−P‖₂/s_rig < 1e-4`,`max < 1e-3·s_rig`(**该 max 待定**,§10)。理由:转置旋转在 G1a 中不可见(位置通道独立存储),却使 FK 链严重偏离。
- **G1c heading 往返**:`chord = ‖ĥ−h‖₂`,`ang = |wrap_π(atan2(ĥ.sin, ĥ.cos) − θ)|`,域 `frame`。判据:`max chord < 1e-6` 且 `max ang < 1e-6 rad` 且 `max|‖ĥ‖₂−1| < 1e-6`(投影后)。另**必报**:退化帧逐 clip 占比与最长连续退化段长度写入 `g1_report.json`(其阈值判据属 G-heading-D,本项只管往返一致性与单位圆性质)。
- **G1d 速度一致 + 尾帧规则**:`e_v(t,j) = ‖v̂−v‖₂/(s_rig·fps_tgt)`(无量纲),判据 `max e_v < 1e-5`。并显式断言(捕获「先算后裁」「尾帧写错」):
  ① `T≥2` 时 `j_v[T−1]` 与 `j_v[T−2]` **逐元素精确相等**(差为 0,非容差);
  ② `T=1` 时 `j_v ≡ 0`;
  ③ **crop 交换律**:取随机 `t0`、`L≥3` 且 `t0+L < T_full`,判据 = `encode(crop(clip,t0,L)).j_v` 在 `t < L−1` 上与 `crop(encode(clip).j_v)` 逐元素相等,且在 `t = L−1` 上满足 ①(诊断项:另报告 `t=L−1` 处两条路径是否巧合相等,仅用于识别匀速片段,不作判据);
  ④ `max‖j_v(t) − fps_tgt·(P(t+1)−P(t))‖/(s_rig·fps_tgt) < 1e-5`,`t < T−1`。
  须在报告中给出 ③ 的**实际覆盖率**(短 clip 多时覆盖率下降)。
- **G1e contact 位级一致**:判据 `popcount(f̂ ⊕ f_target) == 0`,域 = **全部** `(t,j)`(含 root 行与非合格关节:这些位必须恒 0);`f̂` 取 decode 的编码侧特征位路径(不经 sigmoid/阈值)。`contact_bits_sha256` 须与 per-clip manifest 匹配;`contact_supervised_mask` 与 `source_contact_eligibility` 的索引一致性同时核对。
  **与 §2.2 的耦合**:采纳「source contact 原样保存」⇒ `f_target ≜ f_src`。若未来退回 lossy 变体,则 manifest 必须记录被丢弃源位数 `N_drop` 与 clip 清单,并断言 `N_drop` 与声明的 lossy 边界**完全一致**(现审计值 3,687 位 / 30 clips),多一位即 abort。
- **G1f mask 与索引一致**:`frame_mask / joint_mask / edge_mask / contact_supervised_mask / rot_supervised_mask` 往返后 dtype、shape、逐元素**完全相等**(无容差)。关节序:decode 输出的 joint 顺序须逐名等于 `skeleton_table_hash` 中的 canonical joint names;并用缓存的 `new_to_old_perm`(现 381/382 非恒等)做显式复原断言 —— `P̂[:, new_to_old_perm, :]` 须逐元素等于源顺序下的 `P_src`(容差同 G1a)。左右锚点 (L,R) 与 contact eligibility 索引往返后同样逐元素相等。
- **G1g repr canonical round-trip(encode→decode→encode,捕获非规范输出)**:规范投影 Π:`h ← ‖h‖₂ ≥ ε ? h/‖h‖₂ : [1,0]`;`j_a ← 6d(GS(6d⁻¹(·)))`;其余恒等。
  ① **幂等性**(无需数据):对随机非规范 x,`‖Π(Π(x)) − Π(x)‖_∞ < 1e-6`。
  ② **GT 侧恒等**:`‖Π(x_GT) − x_GT‖_∞ < 1e-6`,且 `E(D(x_GT))` 与 `x_GT` 逐元素相对误差 `< 1e-6`。
  ③ **非规范度量**(模型输出报告,Phase 3 起 gate):`c_h = |‖ĥ‖₂−1|`;`c_rot = ‖R̂ᵀR̂−I‖_∞`(GS 前的原始 6d 两列);`c_v = ‖ĵ_v − fps_tgt·ΔP̂‖/(s_rig·fps_tgt)`;`c_fk = r_e`(§8/G2)。GT 侧四项均须 `< 1e-5`(fail-closed);模型侧写报告,**阈值待定,与 G2-c 同批冻结**(§10)。
  (注:v2.3 拟议的 `c_root = |r_s.y − j_p[root].y|` 因 MINOR-01 删除冗余而不存在,不再列入。)

**G1 的诚实边界**:全部 sub-gate 用 fp64 评测而生产用 f32,存在「gate 通过但训练态精度更差」的缝隙,故 G1a/G1b 的阈值均按 f32 量化误差留 3 个数量级余量;若未来改 bf16 存储,阈值须重标。

### 9.9 G2 parent-edge FK 一致(L3,三段,各自 fail-closed)

- **G2-a 数值地板(合成 fixture,可先于数据)**:对已知刚体链 rig(含 J=1/链/星,复用 G-heading-F fixtures),由随机 SO(3) 生成 P 与 j_a:fp64 下 `max r_e < 1e-6`,f32 存储下 `max r_e < 1e-5`。不过即实现有误,**禁止进入数据构建**。
- **G2-b 数据地板(GT 侧,构建期硬 gate)**:在 194 canonical topologies × 每 rig ≥3 clips(最长/最短/随机)上算 GT 的 `r_e`,产出 **`fk_baseline.json`**:按 corpus(PZ / TrueBones / human-v4b)与 rig 记 `mean/p99/max(r_e)`、`ℓ_rest` 分位、零长边数、被 `τ_rigid` 剔除的边数;`fk_baseline_hash = sha256(§4.2 envelope)` 入 manifest 与 ckpt。
  **阈值 = 首次构建实测后冻结**:`thr_corpus ≜ max(1e-5, 2×p99_corpus)` 回写同一 artifact;此后任何重建须 `p99 ≤ thr_corpus` 且 `max ≤ 10×thr_corpus`,否则 abort。**在实测填入前不得声明 G2 通过**(禁用「≤ 基线」这类无数值表述 —— v2.3 原文即属此类)。
- **G2-c 模型侧(报告项,Phase 3 起 gate)**:对重建/生成输出算同一 `r_e`;基线 = `fk_baseline.json` 的 `model_reference` 字段(`run_id` + ckpt sha256 + 该 run 的 `r_e` 中位/p99)。**v1 与本 schema 不兼容,当前不存在合法 legacy 基线**,故在本 schema 内第一个通过 G7 视觉 QA 的重建模型被登记为 `model_reference` 之前,G2-c 只报告不 gate;登记后判据 `p99_model ≤ 1.0 × p99_reference`。
- **物理单位报告指标(与训练 loss 分离,不参与梯度)**:
  ```
  E_FK_abs = mean_mask ‖(P_c−P_p) − Δ_p·d_c‖₂                       # 单位 = schema 的 length_unit,报告换算为 mm
  E_FK_rel = mean_{mask ∧ ℓ_rest ≥ ε_ℓ} ‖·‖₂ / ℓ_rest               # 无量纲,按 ℓ_rest 十分位分桶报告
  ```
  `E_FK_abs` 的跨源可比性要求 §5 冻结 `length_unit`(**是否统一换算到米待定**,§10;per-source 换算因子入 manifest)。两指标与 `r_e` 一并写入 gate 报告。

### 9.10 其余 gates

- **G-cpu1 脉冲响应(反积分)**:单帧扰动只影响该帧的恢复结果。
- **G-cpu2 `∂恢复/∂j_v ≡ 0`**:恢复路径不消费 `j_v`。
- **G7 训练前视觉 QA(**优先级高于一切数值阈值**)**:GT 往返 GIF(**多帧、真原速全帧 @ fps_tgt**,不得用单静帧)+ **世界系俯视轨迹** + 蒙皮导出 mp4 + heading 箭头 contact-sheet。
  **heading 箭头复核范围更新**:必须覆盖**全部 154 个非 AUTO_OK rig**(而非 v2.3 的「原 104 rig」),并单列 64 个零运动 rig 的 rest 箭头图 + 其 clip 箭头叠加动画。
  全部发 user 裁决 —— **视觉裁决权归 user;metric 与视觉冲突时以视觉为准**。

## 10 开放问题与待定项

> 全部按「**待定:<需要什么证据才能定>**」书写。**不编造数值充数。** 标 **[USER]** 的须 user 拍板,不由本标准代决。

### 10.1 需 user 拍板的方向性决策

1. **[USER] j_a 的数据来源路径 A vs B**(见顶部阻塞项 / §2.2A / §2.2B)。**待定:user 决定。** 素材已确证支持 A(覆盖 102,438/102,438,源解析 ~0.8 CPU-hour);B 只在明确不愿重跑源转换时可选,代价是标准降级为 lossy、末端(足/手/头/尾尖)旋转不可学。**Phase 2 在拍板前不启动。**
2. **[USER] 64 个零运动 rig 的处置路线**(全人工视觉评审 vs 全部 REJECT vs 混合)。**待定:需要 (a) 这 64 rig 在语料中的 clip 数与帧数占比(决定 REJECT 的数据损失量);(b) 它们覆盖多少 canonical topology(已知无任何 AUTO_OK 同拓扑 donor,很可能是 64 个独立拓扑);(c) 抽 5 个 rig 试做 rest 箭头 + clip 箭头叠加渲染并计时,估算 64 个的总工时。** 注:43 个「静默通过」rig 中 41 个是 TrueBones,直接 REJECT 会显著削减 TrueBones 侧的拓扑多样性(本项目多拓扑迁移的核心卖点)。
3. **[USER] 154 rig 人工评审的排期**:评审结论进 `heading_table_hash` ⇒ 任何事后修改都会使已构建数据/moments/token-cache/ckpt 全部失配。**必须作为构建的前置里程碑而非并行任务。** 待定:user 确认工时预算。
4. **[USER] `semantic_hash` 采用「markdown 锚点抽取」还是「机器可读 `btjdk_semantics.json`」**。前者使**文档本身成为构建输入**(锚点块内任何排版/空白改动都会改哈希并使全部工件失效),需一条「锚点块内禁止非语义编辑」纪律 + 显式 `--rebump-semantic` 流程;后者哈希对象是 JSON 而非排版,纪律风险消失,代价是改构建流程。**待定:流程取舍,非技术未知,需 user 拍板。**
5. **[USER] `data/animo4d_anytop/proximal_rotation_removed_20260608/bvhs` 里 3,372 个被 proximal-rotation 过滤的 PZ clip 是否重新纳入**。**待定:数据范围决策,须 user 拍板;若纳入,382 rig 的 heading 表、skeleton 表与 194 topology 计数全部要重算,所有哈希失效。**
6. **[USER] 是否保留 `h` 作为独立通道**(冗余但可直接条件化/监督)还是删除并从 `j_a[root]` 派生。建议保留(cdir 条件与 kimodo 一致、heading loss 可直接监督),但这是显式取舍。**待定:Phase-3 一个消融 —— 有 h 通道 vs 无 h 通道时 cdir 条件的朝向可控性(生成动作首帧朝向与 cdir 的圆误差分布)。**
7. **[USER] 统一目标 `fps_tgt` 取 20 还是 24/30**。20 与现行渲染/滤波常数一致但丢约 20% 帧(PZ 丢 1/6),且 TrueBones per-skeleton 本已稀缺(中位 11 clips/skel);24 保动物侧帧(占语料 74%)但需重定 `Wn`、`L_f`、`v_c` 并重测 §7 阈值。**待定:需要 (a) 现行 backbone/evaluator 是否对 20 fps 有硬依赖的核查,(b) TrueBones 稀缺 skel 在 20 vs 24/30 下的 per-skel 帧数,(c) 24→20 下采样对动物高频步态(蛇/鸟/蜂鸟)的 G7 视觉裁决。**
8. **[USER] 是否恢复 twist helper 以支持蒙皮/表情导出**。代价:canonical J 上限从 102 抬到 206(PZ 侧),`max_joints`、token 长度、显存与全链缓存都要变,且 TB / human 无对应关节(跨拓扑不齐,大量 padding)。**待定:需 user 明确「蒙皮/表情质量」是否属本 schema 目标;证据 = 用现行保留集做一份蒙皮导出 QA(现有 PZ skinning 导出管线已具备),看 twist 缺失在视觉上是否真的可见。**

### 10.2 需实测才能冻结的阈值

9. **`ε_h`(默认 0.05)、G-heading-D1 的 0.05、D3 的 0.25 s。** **待定:在 G-rest 通过并按新契约重建旋转后,对全语料出三张直方图 —— (a) `n(t)` 的全帧分布(尤其左尾 n<0.2 的质量与其所属 rig/动作标签),(b) `heading_deg_run_max` 的 clip 级分布,(c) `heading_deg_jump_max_deg` 的区间级分布;并交叉列出「丢弃率 by rig / by canonical topology」。** 阈值应取 (c) 的主峰与长尾之间的谷点并使总丢弃率 ≤ 0.5%;**若 (c) 无双峰结构,则说明 hold-last 在本语料上不安全,须改为把退化区间的 h 显式标为无效并在 loss/条件中 mask 掉(第二方案,须另行 codex 审)。**
   ⚠ 现行语料对该风险**零信息量**:legacy root 是 yaw-only,直立类动作(熊直立、袋鼠、人类跳跃仰身)在真实旋转数据下才会真的触发退化。
10. **G-heading-C 的 p99 < 5°。** **待定:需 Phase-2 GT 上的实测分布(GT 中该误差应恒为 0)与 Phase-3 生成样本的实测分布。**
11. **`§2.1.1(d)` 的 rest 水平性系数 0.1。** 可与 `ε_h` 一并在 Phase-2 直方图上复核;因它只作用于**每 rig 一次的 rest 姿态**(非逐帧),取值不敏感:实测当前 382 rig 无一接近该边界(spine-axis 仅 Anaconda / Deer 两个,均为水平体轴)。
12. **`G-skel-m` 的紧阈值。** **待定:(a) 272-derived 人类源族 —— 在 ≥200 clip 上跑 `res(c,t)` 并给出按 rig 分层的 p50/p99/max 直方图,据此按 `ceil_1sig(10×p99)` 冻结;(b) BVH-native 源族提议的 1e-6 —— 在 PZ 与 TrueBones 各 ≥50 clip 上实跑确认 max res 落在 fp64 链式误差量级(预期 <1e-9),否则说明源 BVH 存在非刚性骨长或单位/轴序差异,须先归因再定阈。**
13. **`G1b2` 的 max 阈值(链累积)。** **待定:在最深链 rig(报告 J 与最大链深)上统计 f32 存储下 FK 重建误差的 p99/max 分布,取实测 max 上取整一个数量级。**
14. **`G1g③` 模型侧非规范度量阈值 + `G2-c` 的 `model_reference`。** **待定:待本 schema 内首个通过 G7 的重建模型实测,同批冻结。**
15. **`τ_rigid`(§8 FK 刚体性断言)。** **待定:对 PZ / TrueBones / human-v4b 各随机 ≥1000 clips 统计 `max_t|‖P_c−P_p‖−ℓ_rest|/s_rig` 的分布(重点确认 TrueBones 是否存在 joint 平移通道),取 p99.9 上取整一个数量级。**
16. **`thr_{j_v}`(§7④)。** **待定:上文 9.28 来自 legacy heading-local 13ch 代理(ch9:12 是每帧位移、非 ΔP·fps),不可直接采信;需 BTJD-K 真实世界系 `j_v` 在全量语料上的 per-τ `Q₀.₉₉₉` 实测。** 注:`thr_G` 需「先构建后冻结」两阶段流程 —— 第一遍全量构建只为实测,第二遍才带最终 `preprocess_hash` 正式产出;两遍之间任何代码改动都会使第一遍测量作废。
17. **`w_pos`(per-rig)实际数值。** **待定:Phase-2 构建后在 train split 上、`frame ∧ contact_supervised` 域内实测 `N_pos/N_neg`。**
18. **`ρ_low` 的 warn/fail 阈值(初值 1%/5%)。** **待定:首个重建模型 val 上的 `‖ĥ‖₂` 直方图 + 「heading 角误差 vs 模长」散点(确认 `ρ_low` 与角误差确有相关性,否则该监控无判别力)。**
19. **蛇类 contact 是否需要在 `w_pos` 之外再加 per-rig 权重**(「整段贴地」与「足端触地」不是同一量纲)。**待定:需 Phase 3 的 per-rig 分组 contact 预测 PR 曲线 + 蛇/四足各自的贴地视觉 QA。**
20. **`T_w`(窗口长)与长 clip 尾部覆盖策略。** **待定:(a) `T_w` 需 backbone 显存/序列长度预算;(b) val/test/token 导出 `t₀=0` 会丢弃 `T_clip − T_w` 帧,若改确定性 tiling(`t₀ ∈ {0, S, 2S, …}`)须把 `window_index` / `stride S` 写入 manifest —— 需 token-cache 存储预算 + 被丢弃帧占比统计。**
21. **`length_unit` 是否由 AnyTop-scaled 统一换算到物理米。** 若改,`v_c` / `h_c` / `s_min` / `τ_abs` / `E_FK_abs` 全部需换算。**待定:需 §5 三语料原生单位的核对;建议 btjdk-v3.1+ 再议。**
22. **torch(CPU/CUDA)上的 GOLD-6 实测余量。** **待定:在目标训练镜像里跑同一 seed/同一 N 的 GOLD-6,记录元素 Linf 与测地 max,确认 fp64 路径 <1e-12、fp32 路径 <1e-5/2e-3;有实测数才把 torch 侧阈值落表。**
23. **全语料 `min ‖a1‖`(GT 侧 6d 第一列模长最小值)分布。** **待定:Phase-2 构建前对全部 clip×joint 扫一遍;若存在 <1e-3 的条目,须先决定是拒绝该 rig 还是放宽 `deg_rate_6d` 的 GT 判据,不得先冻结再补丁。**
24. **`N_min = 1e6` 与 `N_{τ,G} ≥ 1e4`** 是按现语料规模估的下界。**待定:若未来加入小规模新语料需重估,证据 = 各 τ 的实际有效标量数直方图。**

### 10.3 需一次全量复跑才能落表的项

25. **`source_contact_eligibility` 全量重算。** 现值仅来自**每 rig 6 clip 抽样**(382 rig / 11,040 合格关节槽 / 耗时 8 s),会漏掉低频合格关节。**待定:正式落表前必须在全部 102,438 clip 上重跑(按抽样耗时外推约 20–30 min,I/O-bound)。** 可确定的是**规则本身**(由数据统计、非名字规则),这一条不待定。
26. **TB End Site `<parent>_end_site` 回退规则是否穷尽。** 现依据是全 cond 名表扫描(仅 Ant/Crab/Deer/Jaguar 共 28 个关节使用)+ 60-clip 抽样 100% 通过,**未跑满 1,070 clip 的 M6**。**待定:落表前必须全量跑一次 M6;若出现第 5 种命名,须补 per-rig 覆盖表。**
27. **`lr_pairs` 的登记覆盖。** ~25 个镜像 rig 已知需登记,其余 357 rig 是「确无左右语义」还是「尚未登记」?**待定:逐 rig 的登记状态枚举 —— 与 G-heading-A 的 382-rig 状态表合并做,避免两张表。**
28. **`physical_scale` 登记表 schema。** 现语料 0 个退化 rig,故无实例。**待定:等 §2.1.3 支持域(J=1/链/星)定稿后合并为同一张表,避免重复定义。**
29. **`foot_semantic_mask` 的人工确认覆盖。** 若它只服务于指标而不进训练,可先只对参与 foot-skate 报告的 rig 子集做,表里显式留 `null` 表示未确认(禁隐式缺省)。**待定:是否接受这种部分覆盖,须 user 确认。**
30. **ckpt 是否必须携带 `split_manifest_hash`。** 当前定为经 `moments_hash → split_manifest_hash` 间接锁。**待定:确认现行训练脚本能否在保存时拿到 split manifest;若能则直接加(零风险收紧),若不能则保持间接锁并在此记为已知弱点。**
31. **`preprocess_hash` 的键清单与 §5 参数的逐项核对。** §4.3 只冻结 envelope 形式,键清单是 §5 的产出。**待定:落地时必须核对「§5 列出的每个参数都在键清单里」,否则会出现哈希覆盖不到的自由参数。**

### 10.4 已知边界(不可复查,只能记录)

32. **PZ stage-1(`00_raw_bvh_target` / OVL 提取)只在 Windows H: 盘,本文件系统不可重跑。** 磁盘上的是 stage-2 `02_anytop_layout` 产物。stage-1 的一切约定(单位、世界系、ROOT 节点选择、24 fps、`_end_site` 命名)**冻结在现状且不可复查**。本标准以 stage-2 BVH 为「原始」,不假装它是绝对原始。
33. **humanact12(~1,190 clip)在 272 集内永久缺失**,A/B 两条路径都补不了。若要补须走 AMASS SMPL-H 备份路径(`motion-latent-diffusion-main/datasets/amass/motion_data`,18,342 npz + SMPL-H neutral),但那是另一套 rest / 单位 / 帧率约定,须**单独立一个 G-motion-rot 子契约**并单独验 M1–M8。当前标为待定,不阻塞 A/B 决策。
34. **`clean_filter_manifest.json` 的 name filter 只覆盖 L4_safe(L3→L4)那一层**;L3 自身的规则(L0-style locator + leaf-only `_end` 移除)只有文字描述(manifest `rule.notes`),没有可执行的正则。重新施加时若与当前 `joints_names` 对不上,**只能以 `cond[rig]["joints_names"]` 为准反推**。是否需要把 L3 规则也补成可执行形式,取决于是否要支持从 stage-2 BVH 完全重建保留集(而非读 cond)—— 待定。
35. **v2.3 遗留开放问题的处置**:① G-rest per-rig 来源逐 rig 落表 —— 机制已在 §9.3 冻结,余下为 382 rig 的逐个来源确认与 ~25 镜像 rig 的约定登记(并入第 27 项);② `s_rig` 分母对蛇/翼类的稳健性 —— bbox 已冻结为 v3.0 分母,备选(骨链总长)属 btjdk-v3.1+ 的破坏性变更;③ K=4 yaw 副本升级路径的触发条件(若 cdir 条件在 A 方案下不可控);④ 平滑截止对高频推进类的敏感性(G7);⑤ contact 独立头 vs 留在 VQ 张量(留张量占码本容量,Phase 3 消融定,若留须报 contact-free 重建指标)。

## 11 r4 意见落实对照表

> **编号说明**:任务书称本节为「§10」;因 §10 已被「开放问题」占用、且本文多处交叉引用「§10 开放问题 N」,本节编号为 **§11**,内容与要求一致(逐条列出落实位置、怎么改的、是否有残留待定项)。

| # | 意见摘要 | 落实章节 | 怎么改的 | 残留待定项 |
|---|---|---|---|---|
| **BLOCKING-01** | heading 规范公式与校准产物方向差 π;不消费 `phi_skel`;spine-axis 无公式;`heading_table_hash` 无法唯一决定 h | §2.1(h 行)、**新增 §2.1.1**、§9.6 G-heading-B、顶部「撤销 2 / 撤销 4」 | ① 叉积序冻结为 `w = (p_L−p_R) × e_y`(与校准脚本 / JSON / AnyTop 三者一致),v2.3 写法作废;② 运行时公式改为 `f(t) = D_root(t)·u_rig`(root rest-delta 作用于 rig 冻结的 rest-forward 轴),补全 limb-pair / spine-axis / explicit 三种 `w` 的确定式与 tip 选择规则;③ FLIPPED 修正改为 `pol = −1`,不再是「phi+180°」;④ `phi_skel_deg` 全表作废、改为 `u_rig` / `phi_rest_deg`,离散量可继承但须 G-heading-B 机检背书;⑤ `heading_table_hash` 字段清单冻结并绑 `skeleton_table_hash` | **有**:§10 第 6 项(h 是否保留为独立通道,USER + Phase-3 消融)、第 10 项(G-heading-C 的 p99<5°)、第 11 项(水平性系数 0.1) |
| **BLOCKING-02** | `[cosθ,sinθ]` 不消除物理滚转奇异;hold-last 只延迟 π 跳变;ε 有量纲 | **新增 §2.1.2**、§9.6 G-heading-D / G-heading-E | ① 记录旧估计器的滚转奇异性(β=90° 时 `‖w_xz‖=6.12e-17`)并作废;② 新退化判据 `n(t) = hypot(f_x,f_z)` 为**无量纲**且**与滚转无关**,只在「前向轴近竖直」时退化;③ 冻结 hold-last + back-fill + 整段退化 REJECT 的状态机与计算位置(S3 内、crop 后、yaw 前,随 clip 缓存);④ 新增 D1 频度 / **D2 hold-last 合法性(核心)** / D3 端点限长三条 fail-closed 判据 + 四个诊断字段;⑤ 新增 G-heading-E yaw 等变性机检;⑥ 显式禁止把 `eps_exposure_table` 当作新定义下的退化证据 | **有**:§10 第 9 项(`ε_h` / D1 / D3 三个阈值,须三张直方图;若无双峰须改 mask 方案并另行 codex 审) |
| **BLOCKING-03** | 缺逐帧真实旋转的权威输入契约;13ch 叶端 twist 未保存;78–177° 无同级证据 | **新增 §9.2 G-motion-rot(.1–.6)**、§2.2A / §2.2B、§5.0、顶部「撤销 1」+ 顶部阻塞项 | ① 冻结 BVH(PZ+TB 共用读取器,ZYX 内旋、24fps、End Site 提升规则)与 SMPL-272 两套读取规范、joint mapping、per-clip manifest 字段、M1–M8 判据(M4 是唯一能抓转置/轴序/单位/乘序的判据);② 语料覆盖实测 102,438/102,438,故**不需要 reject 分支**;③ 明确 13ch 非法输入 + 禁止 IK/位置反推;④ 撤销 78–177° 并换成源侧实测(PZ max 67.2° / TB max 178.3°);⑤ **路径 A / B 并列写全,顶部醒目标注待 user 拍板、Phase 2 不启动**;⑥ 新增 `rot_supervised_mask`(与路径 B 的 `leaf_rot_unobserved_mask` 分字段) | **有**:§10 第 1 项(**A/B 方向,USER,顶级阻塞**)、第 5 项(3,372 proximal-removed clip,USER)、第 7 项(`fps_tgt`,USER)、第 26 项(TB End Site 全量 M6)、第 32/33 项(stage-1 不可复查、humanact12 永久缺失) |
| **BLOCKING-04** | `6d()` 行列约定未冻结;`6d(I)` 在两约定下相同,恒等自检零判别力 | **新增 §2.3**、§9.1 G-6d | ① 列出本仓库两套互为转置的活跃实现(含 human v4b 的 `decode_6d_rows` 是行约定);② 冻结列约定 + 逐元素 encode/decode + 列向量/主动旋转/左乘复合 + `ε_6d=1e-8` 用 `max(·,ε)` 而非 `+ε`;③ 六组 golden(GOLD-4 复合非对称抓「整体转置」、GOLD-5 锁死 GS 排除极分解)+ 10 万随机 SO(3) 往返,**fp64/fp32 阈值分别书写**;④ 实现纪律 + CI grep 建议;⑤ `deg_rate_6d` 退化监控;⑥ **并列声明 G-6d 不能证明 `R_rest_global` 正确 —— 须 G-skel-m** | **有**:§10 第 22 项(torch CPU/CUDA 上的 GOLD-6 实测)、第 23 项(全语料 `min‖a1‖` 分布) |
| **BLOCKING-05** | `offset_c` 未定义;有序骨架未入契约;381/382 perm 非恒等无防错 | §3(补全 offset 定义)、**新增 §4.1 skeleton payload**、§9.4 G-skel-c、§9.5 G-skel-m | ① 补全 `offset[c] = R_rest_global(p)ᵀ·(P_rest[c]−P_rest[p])`、`offset[root]=0`、`edge_mask` 精确定义域;② 24 字段的 per-rig skeleton payload(含 canonical/source 双名表、双向 perm、parents、rest 位置与旋转、`lr_pairs`、heading anchor **名**、contact 合格域、`source_joint_map`、`rot_supervised_mask`)→ `skeleton_hash` / `skeleton_table_hash`;③ **撤销 `rest_table_hash`**(消除第二个可漂移真值源);④ G-skel-c 八条结构一致性(名单一致是最强 perm gate;镜像判据补位置项);⑤ **G-skel-m 非循环正确性**:对源侧独立量的逐边残差,硬失败线 1e-2 立即冻结 | **有**:§10 第 12 项(G-skel-m 紧阈值,两源族分别实测)、第 27 项(`lr_pairs` 登记覆盖)、第 30 项(ckpt 是否带 `split_manifest_hash`);**依赖**:G-skel-m 需 G-motion-rot 先落地,否则 j_a 通道保持冻结 |
| **MAJOR-01** | contact 无损承诺不成立(3,687 帧 root-contact 实录);foot-only mask 会丢蛇类 eligibility | §2.2 ch12 行、§4(新增字段)、§8(per-rig `w_pos`)、§9.8 G1e、顶部「撤销 3」 | ① 撤销 root contact 退役,源位原样保存(含 root 行),零维度成本;② `source_contact_eligibility` **由数据统计得出、非名字规则**(实证:Anaconda 27/27、KingCobra 19/19 全关节合格,足端名单会清零 67,015 位 = root 方案的 18 倍);③ 足端语义另设 `foot_semantic_mask` / `f_foot`,不进 VQ、不进 G1;④ 新增 `contact_supervised_mask`、`contact_bits_sha256`、`eligibility_clip_list_hash`;⑤ 无损承诺收敛为「位级往返」可机检命题 + G1e;⑥ `w_pos` 改 per-rig(蛇正类率 0.9855 会主导全局估计) | **有**:§10 第 25 项(全量重跑 eligibility,现值仅 6 clip/rig 抽样)、第 17 项(`w_pos` 实测)、第 19 项(蛇类是否再加权)、第 29 项(`foot_semantic_mask` 人工确认覆盖) |
| **MAJOR-02** | §5 是步骤标签而非确定性变换链 | **§5 整节重写(§5.0 + S0–S8 + §5.9 + manifest 清单)** | ① 拆离线 S0–S4 / 在线 S5–S8,每步冻结输入/输出/算法/参数/失败条件;② native fps 逐 clip 读出(实证现语料混用 24 / 20 fps 且 schema 无 fps 字段);③ **重采样作用于 DOF(root 平移线性 + 逐关节 SLERP + 一次 FK),不作用于位置**;时间网格禁止 `linspace` 索引拉伸;最短弧显式断言;同帧率恒等旁路;④ **平滑与速度改在全长上算、crop 取其限制**(消除窗口 filtfilt 瞬态与假尾帧速度);⑤ **contact 统一重算**(所有源都无原生 contact 通道,不存在搬运选项)+ `v_c` 由 `0.002@20fps` 换算为 fps 无关的 `0.8944271909999159`;⑥ **删除逐 clip heading 转正**;⑦ crop RNG 与 worker 数无关;⑧ dtype 下转唯一一次;⑨ pad 全 0 且在归一化之后;⑩ 关节保留规则(L4_safe 保留集,子树闭合 0 违例 / twist 全叶 0 保留,故对 j_a 无损;蒙皮与表情显式声明 lossy);⑪ per-clip manifest 全字段;⑫ 新增 G-resample / G-contact / G-det | **有**:§10 第 7 项(`fps_tgt`,USER)、第 20 项(`T_w` 与长 clip tiling)、第 21 项(`length_unit`)、第 34 项(L3 规则无可执行正则) |
| **MAJOR-03** | 「104 rig 清零」不是充分且可机检的 gate;64 个 `NO_MOVING_CLIP` 不在直方图内 | §9.6 **G-heading-A(整条替换)**、§2.1(删除 104/4 那句)、顶部「撤销 5」 | ① 5 值闭集 STATUS + 四条断言(逐名相等、无默认值、REJECTED 必须实际剔除 clip);② AUTO_OK 的七条机检充要条件(第 2 条直接排除全部 64 个零运动 rig;`calib_R` 门槛 0.8→0.9);③ 实测**人工清单 154 rig**(104 ∪ 43 ∪ 7)并列出 43 个「静默通过」rig 全名单(含全部 8 个翼类);④ 非 AUTO_OK 的必含证据字段;零运动 rig 必须 rest 箭头 + clip 箭头**动画**,静帧不算;⑤ 核实同拓扑 donor 数为 0,故不设拓扑继承;⑥ 决策全部入 `heading_table_hash`;⑦ G7 复核范围 104 → 154 | **有**:§10 第 2 项(**64 个零运动 rig 处置路线,USER**)、第 3 项(154 rig 评审排期,USER);且 154/228 标注为「现表实测值,重跑 probe 后须重新统计」 |
| **MAJOR-04** | 382-rig 校准只覆盖当前语料;J=1 / 链 / 星 / 空 mask 未定义 | **新增 §2.1.3**、§4(五类 mask)、§7/§8(空 block)、§9.6 G-heading-F | ① S1/S2/S3 三段支持域 + 顺序 + rest 水平性判据 0.1;② J=1 必须走 S3 否则 REJECTED(现语料 J_min=9,故不存在);S3 除 fixture 外不实现额外逻辑;③ **澄清「链状卷曲退化」是旧逐帧几何定义的性质,新定义下不成立** —— S2 的退化只在 rest 姿态上判定;④ 空 block 一律 skip(分母 `max(count,1)` + 指示子),禁止除零/NaN,per-block count 必须逐 step 上报,`frame_mask.sum()==0` 直接 abort;⑤ 8 个 golden fixtures(4 合成 + 4 语料内困难 rig,后者验证 gate 确实拦得住四类已知失败);⑥ 显式写出「只承诺 fail-closed 拒绝,不承诺支持」 | **无独立待定项**(唯一外部依赖是水平性系数 0.1,已并入 §10 第 11 项复核;实测 382 rig 无一接近边界) |
| **MAJOR-05** | `s_rig<ε` 回退中位数病态;G1 `<1e-4×bbox` 在 bbox=0 不可通过;`g_G=1e6` fp16 溢出;空组/NaN 无 fail-closed | **§7 ①②③④ 重写**、§9.8 G1a 容差 | ① `s_rig` 退化改 `physical_scale` 登记表 → 否则 **拒绝该 rig**;撤销中位数回退(实测 382 rig `s_rig ∈ [0.9012, 5.0107]`,`<0.1` 为 0 个,该分支是死代码);② **finite gate 必须先于任何统计与比较式**,禁 NaN-aware 归约;③ 样本量 gate `N_min=1e6`,空组走同一路径;④ std 两遍 f64 算法,禁平方和;⑤ **std gate 常数组 fail-closed**(`[1e-3,1e3]`,越界 abort 而非 clamp)⇒ `g_G ≤ 1e3`;撤销 `1/max(std,ε)`;⑥ **新增精度契约**:统计强制 f64、归一化至少 f32、**禁 fp16/bf16**;⑦ 百分位 gate 改 per-group 且必须实测冻结(实证 `j_v` 代理 9.28 > 8,常数 8 会误 abort);⑧ G1a 改 `max(1e-6·U, 1e-4·s_rig)` 正绝对下限 | **有**:§10 第 16 项(`thr_{j_v}` 与两阶段构建流程)、第 24 项(`N_min` / `N_{τ,G}` 重估)、第 28 项(`physical_scale` 表 schema) |
| **MAJOR-06** | FK 先反归一化使残差处于物理长度单位,固定权重 5 偏袒大 rig;G2「≤基线」无定义 | **§8 FK 块重写**、§9.9 G2(三段)、§7③ 末句改写 | ① 恒等化简 `R_global(p)·offset[c] ≡ Δ_p·d_c`,FK 项只消费 `j_a` 与 `d_c`;② **训练态不反归一化**,用 `d̃_c = (g_jp/s_rig)·d_c`,残差 = `g_jp·‖·‖/s_rig`,同等相对骨长误差在大小 rig 上给出相同梯度;③ reduce 冻结为「样本内 masked mean(逐边等权)→ batch 内有效样本平均」,空则 skip,梯度同回 `j_p` 与 `j_a`;④ 零长度边保留在 `edge_mask`(本形式不除边长故无奇异),仅在相对指标中排除;⑤ 新增刚体性前置断言(源含 joint 平移会使边长时变);⑥ G2 拆 a(合成数值地板)/ b(GT 数据地板 + `fk_baseline.json` + `thr_corpus ≜ max(1e-5, 2×p99)` 实测后冻结)/ c(模型侧,**当前无合法 legacy 基线故只报告不 gate**);⑦ 物理单位报告指标 `E_FK_abs` / `E_FK_rel` 与训练 loss 分离 | **有**:§10 第 15 项(`τ_rigid`)、第 14 项(`thr_corpus` 与 `model_reference` 首次构建/首个模型后填)、第 21 项(`length_unit`) |
| **MAJOR-07** | `semantic_hash` 覆盖过窄;ckpt 无 `moments_hash`;拼接无 framing 可碰撞 | **§4.2 envelope + §4.3 哈希分工(整段替换 v2.3 哈希构造)** | ① TLV envelope(MAGIC + 逐字段 name/type/ndim/shape/payload_len + 尾部封口),8 种 type_code 含**显式 null sentinel**;f64-only、禁 NaN/Inf、负零归一、字段不得省略、顺序 normative;② **必须随实现落地的碰撞负例回归**(旧构造会碰撞的三对)+ numpy/torch/纯 python 三路字节一致;③ 七个哈希分工表(`semantic` 改锚点抽取、新增 `preprocess` / `skeleton_table` / `moments` / `source_manifest` / `contract`,撤销 `rest_table`);④ **ckpt 必含 `moments_hash`**(v2.3「无新增字段」作废);⑤ 六步加载核对流程,**禁止白名单/warn-and-continue/env 跳过**,唯一旁路是 `--rebuild`;⑥ 收敛为单一 `verify_contract()` | **有**:§10 第 4 项(**`semantic_hash` 锚点 vs JSON,USER**)、第 30 项(ckpt 是否带 `split_manifest_hash`)、第 31 项(`preprocess_hash` 键清单核对) |
| **MAJOR-08** | G1 只给 `max abs err` 相对 bbox,转置旋转/contact 删除/尾帧错误/非规范预测均可漏过 | **§9.8 G1 拆为 G1a–G1g(含 G1b2)** | ① 评测集与报告格式冻结(`g1_report.json` + hash);② G1a 位置(绝对下限 + 相对均值双判据);③ **G1b 对独立源旋转**比较(自比无法暴露转置),测地角用 `2·arcsin(‖·‖_F/2√2)` 稳定式;④ **新增 G1b2 旋转→FK 位置**作为转置的第二道独立捕获;⑤ G1c heading chord/角度/单位圆;⑥ G1d 速度 + **尾帧逐元素精确相等** + **crop 交换律**(抓「先算后裁」);⑦ G1e contact **位级**(popcount==0,含 root 与非合格关节恒 0);⑧ G1f 五类 mask 与 perm 复原逐元素相等;⑨ G1g canonical round-trip(幂等性 + GT 恒等 + 四项非规范度量);⑩ 全部 fp64、禁 nan-aware、禁跨 sub-gate 平均 | **有**:§10 第 13 项(G1b2 max 阈值)、第 14 项(G1g③ 模型侧阈值);**耦合**:G1b/G1b2 依赖 BLOCKING-03 落地,G1e 的 `f_target` 依赖 MAJOR-01 裁决(已裁定为源位原样) |
| **MINOR-01** | `r_s.y` 与 `j_p[root].y` 冗余;恢复只读后者,前者仍单独计 loss | §2.1(rglob 5→**4**)、§3、§5 S2/S5、§6、§7①②、§8、§9.8 G1g | ① 裁决**唯一权威 root 高度 = `j_p[root].y`**,删除 `r_s.y`(它在恢复路径零消费,删除不损失信息);② rglob 改 `[B,T,4]`,`r_s` → `r_s_xz`;③ §6 新增 **`Y₂(φ)`(注意 `Y₂ ≠ Rot2`,互为转置)** 并给出一致性核验;④ §7 尺度组改 `{j_p, j_v, r_s_xz}`(副带收益:pooled std 不再被高度污染);⑤ §8 块改名、通道 3→2;⑥ G1g 的 `c_root` 项删除;⑦ 记录退回方案(若架构确有固定 5 维硬约束则保留槽位但声明 derived read-only,二选一禁并存);⑧ 显式声明与 kimodo 维度不再对齐、移植的是机制非维度 | **有**:§10 未单列,判据明确 —— **待定:Phase 3 的 frame/holder 流架构是否对全局流宽度有硬性约束(证据 = 架构定义文档/代码)。** 若有,按退回方案保留槽位,权威值仍为 `j_p[root].y` |
| **MINOR-02** | `j_a` 二选一未定;标题与实际 loss 不符;h 无 unit-circle 约束与零向量率监控 | **§8 整节重写(loss 契约)** | ① 标题改「loss 契约」;② 统一 reduce(样本内 masked mean → batch 内有效样本平均;空则 skip;**连续 200 步全空 fail-closed 报警**;归约 fp32);③ `β=0.1` 冻结并给出理由(N1′ 使 pooled std≈1);④ **j_a 冻结为 chordal**(对 R 计、无 arccos 奇异、单调于测地角,故 loss/指标是同一序关系的两种读数),6d-L1 与 geodesic 降级为 Phase 3 消融;⑤ GS 稳定 clamp `max(·, ε_gs=1e-6)` + 退化 6d 率监控;⑥ h 改**训练软约束(径向惩罚 λ_unit=1.0)+ 推理硬投影**,新增 `ρ₀` / `ρ_low` 零向量率监控(GT 侧必须为 0);⑦ `w_pos` per-rig 定死;⑧ 新增 **`loss_contract_hash`** 写入 ckpt/log/manifest,resume 逐字段核对 | **有**:§10 第 17 项(`w_pos` 实测)、第 18 项(`ρ_low` 阈值 + 判别力验证)、块权重 Phase 3 实调后回写 |

### 落实状态汇总

- **15 条意见全部落实到具体章节,无一条被搁置或降级。**
- **1 条无残留待定项**:MAJOR-04(支持域 / 空 block / fixtures)。
- **14 条留有待定项**,其中:
  - **8 项需 user 拍板**(§10.1),最关键的是 **BLOCKING-03 的路径 A/B** —— 它是 Phase 2 的顶级阻塞;
  - **16 项需实测/全量复跑后冻结**(§10.2 / §10.3),全部按「待定:<需要什么证据>」书写,未编造数值;
  - **4 项属不可复查的已知边界**(§10.4),只记录不假装解决。

### 已知的实现风险(供 Phase 2 排期参考,不属 gate)

1. **串行依赖变长**:`G-motion-rot → G-rest → G-skel-m / G-heading-B → 重算 382 表 → 重跑 probe → 154 rig 评审 → 构建`。评审结论进哈希 ⇒ 必须一次性做完再构建。
2. **fail-closed 面显著扩大**:首次全量构建大概率被中断若干次;构建脚本必须把 abort 信息落地成可定位的 `(rig, clip_id, t, j, ch)` 清单,否则会退化为「反复重跑」。
3. **两阶段构建**:`thr_G`(§7④)与 `thr_corpus`(G2-b)都需「先构建实测 → 再冻结 → 正式产出」,多一次全量构建成本;两遍之间任何代码改动都会使第一遍测量作废。
4. **可能整类丢 clip**:G-heading-D 会丢弃翼类俯冲、蛇/猫科垂直攀爬、体操式翻滚等 clip;必须先出直方图与分组丢弃率再冻阈值。
5. **在线段吞吐**:离线全长 + 在线 crop 使 DataLoader 变为变长读取,每样本多做一次 canonicalize/归一化/pad;VQVAE 已被实测为 launch-bound(~157 items/s),Phase 2 必须先做吞吐 smoke 再定 worker 数。存储反而下降(全长均值 ~120 帧 < 现行 pad 到 300)。
6. **重采样的样本损失**:若 `fps_tgt = 20`,对 PZ 丢 1/6 帧,全语料总帧数下降约 20%;TrueBones per-skeleton 本已稀缺(中位 11 clips/skel)。
7. **新代码必经 codex 审**:§4.2 envelope 是所有 gate 的地基,一旦有 bug 会以「哈希看似匹配」的形式静默通过 —— **负例回归用例必须先于任何数据构建落地**。








