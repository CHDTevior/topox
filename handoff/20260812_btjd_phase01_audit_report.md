# BTJD Phase 0-1 审计报告

- **日期**: 2026-08-12
- **对照计划文档**: `handoff/20260811_kimodo_btjd_repr_migration_plan.md`(下称"计划")
- **审计范围**: 计划 Phase 0(设计前提审计,items 1-4)+ Phase 1(heading 离线校准 + 极性验证 + 奇异点审计),全部只读、零训练成本,均在 swarma1004 上完成
- **覆盖面**: 全语料 102,438 clips / 9,985,438 帧 / 382 rigs(cond 条目),动物 + human,四次全量扫描共 0 读取错误
- **产物目录**: `scratch/btjd_phase01/`(脚本 + JSON/TSV 结果,详见文末附录)

---

## 一、各项审计的数字结论

### P0.1 — root ch2 死通道验证(计划 Phase 0 item 1)

**结论:root ch2 全语料 DEAD,复用为新通道零信息损失。**

- root ch2 全局 max|abs| = **0.0**,覆盖 102,438 clips / 9,985,438 帧的**每一帧**(动物 75,592 clips + human 26,846 clips),任何阈值下非零 clip 数 = 0。
- 计划"通道占位"前提在动物侧(此前仅 human 侧已知写 0)完全成立。
- **Caveat(重要)**: cond.npy 的 per-rig std 在 root ch2 上**非零**(AnyTop group-smoothed std,如 Alligator 0.0062)——**绝不可用 cond std 作死通道探针**,clip 级扫描才是权威。
- 全量扫描 19.5s(48 workers),0 错误。

### P0.2 — tpos_first_frame 规范 T-pose 验证(计划 Phase 0 item 2)

**结论:382/382 rigs 的 tpos_first_frame 几何上精确等于 rest-offset pose。**

- Kabsch 对齐相对 RMSD ≤ **2.4e-8**(判据门槛 1e-2 = rest bbox 对角线的 1%;判据为仓库 `_recover_world_positions` 的单帧特化、root-centered、对 `restp[j]=restp[parent]+offsets[j]` 做 Kabsch)。
- **两个对迁移关键的约定发现**:
  1. **311/382 rigs(全部 PZ_\*)存 NON-identity rest 局部 rot6d**——精确的轴置换矩阵,偏离 identity 118-120°,且 root 处有 ~87-90° 的全局 rest 朝向偏移(直接 RMSD ~0.5 vs Kabsch ~1e-8)。71 个 TrueBones+human rigs 为 identity。**任何假设 identity rest 旋转的 heading/q_rest 代码对每一个 PZ rig 都是错的**;计划的 rest 校准 delta 形式 `θ = φ_skel + twistY(q_root·q_rest⁻¹)` 正是使其安全的原因。
  2. **12 rigs 的 tpos 速度通道非零**(max 0.165,SabreToothTiger)——仅元数据污染,几何仍精确 rest;但若任何消费方假设 tpos vel==0(如把 tpos 当静态帧做归一化或 FK-vel 检查)则会踩到。
- **clip 首帧不是 T-pose**: 抽样 764 clips(每 rig 2 条,seeded)仅 **6/764** 首帧几何上为 rest(5 个拓扑: Bird/Crab/Cricket/FireAnt/Tukan);首帧 RMSD 中位数 9.4% bbox。规范 T-pose 只存在于 cond 的 tpos_first_frame 中。

### P0.3 — root-contact 审计 → D=14 可行性(计划 Phase 0 item 3,codex 意见 8)

**结论:root ch12 实质 DEAD,可退役并将 root ch12:14 重分配给 heading [cosθ, sinθ]——D=14 布局可行。**

- root ch12 非零: **3,687 / 9,985,438 帧(3.69e-04)**,**30 / 102,438 clips(2.93e-04)**;**380/382 拓扑逐帧精确为 0.0**(= 102,408 clips,99.97% clips / 99.96% 帧)。
- 唯二触发拓扑为两种无肢蛇: **Anaconda**(20 clips,root-on 98.5% 帧)与 **KingCobra**(10 clips,94.1%)——蛇的 root 接触是真实的(全身贴地),但近恒 ON,且与 root 的子脊柱关节 ch12 几乎完全冗余:**Anaconda 100.0% 一致(min 100%,BN_Tail_01);KingCobra 99.81% mean / 98.88% min(BN_Spine_01)**——全语料约 **2 帧**不可冗余恢复。
- 探针有效性 fail-loud 自证: 叶关节确实承载 contact(any-leaf contact 率全语料均值 **0.859**),全部 contact 值严格二值 {0,1}(非二值计数 **0**)。
- **设计决策记录要求(计划 §2.4)**: 退役的依据是**冗余可恢复**(蛇 root 接触可从 root 的第一个脊柱/尾子关节以 99.8-100% 一致恢复),不是"绝对全零";若未来新增无肢拓扑须重审。
- 方法确认: motions/\*.npy 为 RAW 关节序(与 raw cond.npy 一致,置换仅发生在 `__getitem__`)——审计已按已知 footgun 正确处理。全量 23s(~4,400 clips/s)。

### P0.4 — 语料口径实数(计划 Phase 0 item 4,codex 意见 12)

**结论:计划文档口径 "~90k clips / 179 拓扑" 须回写为 102,438 / 194 / 382。**

| 口径 | 实数 | 计划旧口径 |
|---|---|---|
| clips 总数 | **102,438**(train 97,288 / val 5,150,split 精确全覆盖) | ~90k |
| — 其中 human / animal | 26,846 / 75,592 | — |
| rigs(cond 条目) | **382**,全部有 ≥1 clip | 382 ✓ |
| canonical 拓扑(唯一 parents 结构,BFS-root-first 归一化去重) | **194** | 179 |
| 总帧数 | **9,985,438**(~138.7h @20fps) | — |

- **179 在任何结构定义下均不可复现**(raw-parents 去重 = 194、BFS 归一化 = 194、名字族 = 379、唯一关节数 = 68)。计划 §P0.4 与 §3.2 "全 179 拓扑同一条代码路径" 等表述应以 **194 拓扑 / 382 rigs** 修订,并注明定义 = "唯一 parents 结构"。

### P1.1 — φ_skel 表覆盖率 + manual 清单(计划 Phase 1 item 1)

**结论:382/382 rigs 全覆盖出表,0 个无方法 rig;但校准路径与计划 §3.2 不同——tpos rest-pose 校准对 PZ 无效,已改用 motion-frame 校准。**

- 覆盖: **380 limb-pair**(L/R 从 `_cond_normalized_J144.pkl` 运行时关节名解析,per codex 意见 4)+ **2 spine-axis**(Anaconda、Deer),**0 method-less**;锚点索引同时给出 dataset 序与 raw 序(new_to_old_perm 已处理);raw-vs-pkl root 索引交叉核对 382/382 PASS。
- **KEY FINDING(须回写计划 §3.2)**: cond tpos_first_frame 的世界几何对 **311 个 PZ rigs 不在 motion 世界系内**(rolled/轴置换的资产系,|φ_tpos − φ_motion| 中位数 **87.5°**,max 179.9°,**非均一**——至少 2 个子约定,含 ~25 个 mirror-like rigs),vs 非 PZ 中位数 6.1°(同时反证估计器正确)。→ **计划 §3.2 "从 tpos rest pose 校准 φ_skel" 的路径对 311/382 rigs 无效**;已实现的替代 = 从实测直线/水平 locomotion 帧做 circmean 稳健校准(逐 clip 共识 + circular-median 离群剔除)。**q_rest 取 tpos root 行 rot6d 仍有效**(yaw-only),per-rig φ 吸收其约定。
- 质量: calib_R 中位数 **0.995** / p5 0.868;**10 rigs R<0.8** 已标记。human 合理性检查: φ_skel = **−3.28°**,R = 0.910(期望 ~0,+Z 朝向)✓。
- **manual 清单: needs_review 共 104 rigs**(多原因计数): 46 pair-spread>30° / 39 mixed-votes / 15 sideways / 10 low-R / 9 tb-tpos-disagree>30° / 8 pooled-fallback / 7 low-evidence / 4 π-flip / 1 ambiguous。清单文件: `phi_skel_manual_needed_full.tsv`。
- 扫描: 102,438 clips(6 条 T<6 帧跳过),0 错误,~53s / 48 workers。

### P1.2 — 极性(差 π)验证(计划 Phase 1 item 2,方案 (c) 离线探针)

**结论:校准后 heading 与位移方向全局强一致,极性总体正确;4 个 FLIPPED rig 待人工复核,64 个 rig 无法用速度探针验证。**

- 全局分布: 620 条代表 clip、34,794 个移动帧中,heading-vs-位移角差 **87.2% 落在 ±45° 内,仅 3.5% 接近 π**——分布按要求集中在 0。
- per-rig 判定: **OK 252 / MIXED 39 / SIDEWAYS 15 / FLIPPED 4 / AMBIGUOUS 1 / LOW_EVIDENCE 7 / NO_MOVING_CLIP 64**(共 382)。
- per-topology: 131/194 拓扑可探测,**118/131 |median diff| ≤ 15°(p50 = 1.4°)**;flip_rate>0.5 仅 2 个拓扑。
- **FLIPPED 4 rigs + φ+180° 建议(复核假设,未自动套用)**: PZ_Spotted_Hyena_Male(−81.4→+98.6,flip 票 19:4,同族 rigs 在 +81 而它在 −81,**唯一强真镜像候选**);PZ_Fennec_Fox_Female、PZ_Saiga_Juvenile、PZ_Somali_Wild_Ass_Juvenile 三者票型窄(10/15、5/8、2/4)且 φ 落在健康 +85..88° 簇内——更像倒退步态样本占多,**review 而非 auto-apply**。
- **64 NO_MOVING_CLIP + 7 LOW_EVIDENCE**(多为 TrueBones 原地/踏步动画)速度探针不可用,其 φ 来自 pose-only/少 clip 估计 → 最终仲裁 = 计划 Phase 1 的 rest 朝向箭头 contact-sheet 视觉 QA(G3 前置),按 QA-primacy 发 user 审。
- MIXED/SIDEWAYS 多反映真实动作混合(倒退、strafe、游泳变体)而非校准错;应看 per-rig 票数列(probe_n_ok/n_flip/n_side/n_amb)而非单一 verdict 标签。

### P1.3 — 奇异点 s(t) 分布 + ε/θ_max 定标(计划 Phase 1 item 3,codex 意见 3)

**结论:当前编码下奇异点暴露面为零——§3.2 五件套在现数据上是 no-op;ε=1e-4 保留为纯 tripwire;θ_max 定标数据齐备待拍板。**

- **s(t) = q_w²+q_y²(在 q_delta = q_root·q_rest⁻¹ 上)全语料每 clip 每帧 min ≥ 0.9999999999999994**;s<0.05/0.02/0.01 的 clip 清单**全空**;ε 在 [1e-8, 0.5] 任何取值都不会触发(观测浮点偏差 ~6e-16)。
- **根因已独立在矩阵级验证**: root ch3:9 rot6d **精确 yaw-only by construction**(y 行/列非对角元 ≡ 0.0 到机器精度),连最杂技的 clip(Parrot CircleFly、SabreToothTiger 180Flip、Raptor2 RunJumpRoll、Scorpion-2 Flipped、Hamster RollAttack)也如此;pitch/roll 全在子关节旋转里(子关节 |R11−1| 达 1.65)。q_rest 对 382/382 rigs yaw-only。故 twistY(q_delta) 在现数据上就是精确 yaw 角,§3.2 fallback 路径永不触发,ε 与 heading_valid 降格为回归 tripwire。
- **ε 建议: 保留 1e-4 纯 tripwire**——未来若触发,信号含义是"上游管线变了",不是预期代码路径。
- **θ_max 定标(|ΔtwistY|/帧 @20fps)**: p50 0.21° / p99 5.75° / **p99.9 17.43°** / p99.99 53.9° / max 179.65°;计划初值 **30°/帧 ≈ p99.97 → 2,781 帧对 / 2,323 clips(2.3%)进 manual-review 清单**(可按 `per_clip_smin_full.tsv` 的 dtheta_max_all_deg 排序)。最恶劣的 ~179-180°/帧单帧 yaw 翻转(HML3D_Human 001766/M001906、PZ Ring-Tailed Lemur Juvenile T=22、PZ Quokka T=30、PZ Capuchin)形似数据 teleport/wrap 毛刺。
- **SCOPE caveat(关键)**: 零暴露是**已 yaw-collapsed 的现编码**(v4b ch3:9)的性质——上游 BVH/SMPL→13ch 管线已做了 yaw 提取。**若 Phase 2 改为从 raw 源的完整 3D root 朝向重推 heading,真实奇异点暴露必须在 raw 源级重审**;若沿用同一上游 yaw 提取路径则编码级暴露保持为零。
- Δθ 在 delta-quat twist 上测得,φ_skel 常量抵消 → 该定标对未来 heading 通道直接可复用。s_raw(不减 rest)亦 ≡1.0,两种读法一致。

---

## 二、待 user 拍板的 Phase 0.5 决策清单(计划 Phase 0 item 5)

> 每项附审计数据支持的推荐。铁律不变: 决策记录回写计划文档;涉及代码的一切改动仍必经 codex 审。

### (a) 方案 A1 vs A2 vs D(scope 档位)

**推荐: A1(D=14,raw 绝对 root + heading 通道,无平滑器)。**
审计支持:
- P0.1: root ch2 全语料精确 0.0 → 复用零信息损失,A1 的通道预算前提成立;
- P0.3: root ch12 冗余可退役(380/382 拓扑全零,蛇 99.8-100% 可从子脊柱恢复)→ root ch12:14 = heading [cos,sin] 的 D=14 布局成立;
- P1.3: twistY heading 在全语料上精确良定义(零奇异暴露)→ A1 的 heading 通道无数值风险;
- A2 的 SavGol/短 clip 边界复杂度(G-cpu6)按计划仅在"Phase 3 recon 对照证实 raw-root ill-conditioned"时才启用——维持 A2 为触发式备选即可;
- D 只修积分漂移、不修 heading,而 P0.2 显示 311 PZ rigs 的非 identity rest 约定使 heading 问题真实存在,D 不解决核心诉求。

### (b) crop 语义: per-crop 重锚(推荐)vs 显式条件

**推荐: per-crop 重锚(计划 §2.3 推荐路线)。**
审计支持:
- P1.3: θ(t) 全语料处处良定义且为精确 yaw → crop frame-0 的 θ0 重算平凡安全,重锚的刚体 yaw 旋转闭式可逆(G5/G6 可精确测试);
- 主推路线免去 heading 条件 plumbing 整条链与 yaw 增广×cache 难题(见 (e)),工程面显著更小;
- 代价已知且已在计划中钉死: per-rig mean/std 必须在**重锚后 crop 分布、train-only** 上重建(P0.1 的 cond-std caveat 同时提醒: 重建 moments 时勿沿用旧 cond 统计语义)。

### (c) 速度通道: 保留为特征(推荐)vs 方案 C(D=11 删除)

**推荐: 保留(方案 A 布局),维持计划"A/C 需小规模 recon 对照后才可采 C"的门槛。**
审计支持:
- 本轮审计没有产生任何指向"速度通道有害"的证据;能量塌缩史(项目记忆)提示显式速度输入重要,删除属高风险;
- P0.2 附带发现: **12 rigs 的 tpos 速度通道非零(max 0.165)**——保留速度通道时,任何假设 tpos vel==0 的消费方(归一化/FK-vel 检查)须显式处理该污染,建议在重编码时顺手清零并记录。

### (d) PZ 导出契约: 迁移外部管线(首选)vs 短期 shim + 往返等价 gate

**推荐: 直接迁移外部 PZ 管线读绝对 root(计划首选);若时序需要 shim,严格执行"shim 速度只从预测绝对 root 差分导出 + 往返等价 gate"。**
审计支持(本轮审计显著加重了"少过一层 legacy 约定"的分量):
- P0.2: 311/382 PZ rigs 非 identity rest 局部 rot6d(118-120° 轴置换)+ root ~87-90° rest 朝向偏移——任何 identity-rest 假设对每个 PZ rig 都错;
- P1.1: PZ tpos 世界系与 motion 世界系失配中位数 87.5°、至少 2 个子约定、~25 个 mirror-like——PZ 侧 legacy 约定是雷区,契约桥接层越薄越好;
- 时序(直接迁移 vs 先 shim 后迁移)按计划留 user 定,本报告推荐直接迁移以避免在雷区上多架一座桥。

### (e) yaw 增广 × 离线 cache 策略(仅备选条件路线需要)

**推荐: 采 (b) 主推 per-crop 重锚路线,则 backbone yaw 增广恒等、无意义、不做——本决策项随 (b) 一并消解。**
审计支持:
- P1.3: 全语料 root 旋转本就精确 yaw-only,重锚后再随机 yaw 是恒等操作(计划 §2.3 结论被数据坐实);
- VQVAE 阶段重锚后绕原点小幅 yaw jitter 保持"可选、非必需"。
- 仅当 user 在 (b) 选备选条件路线时才需在"K 份 yaw 副本展开 cache(体积 ×K)vs 放弃增广"之间二选一。

### 审计新增的拍板项(计划外,由数据浮出)

| # | 决策 | 数据 | 推荐 |
|---|---|---|---|
| (f) θ_max 取值 | 30°/帧 ≈ p99.97 → 2,323 clips 进清单;严格 p99.9 = 17.43° 会标更多 clip | **保持 30° 初值**作构建断言,先人工抽查清单头部的 ~180° teleport 样本(HML3D 001766/M001906 等),再决定是否收紧 |
| (g) 4 个 FLIPPED rig 处置 | 仅 Spotted_Hyena_Male 是强真镜像候选(同族 +81 vs 它 −81);另 3 个票型窄且 φ 在健康簇 | φ+180 **只作复核假设**,全部进 contact-sheet 视觉 QA 仲裁,不自动套用 |
| (h) root-contact 退役记录 | 蛇 30 clips 真实接触,99.8-100% 冗余可恢复 | 按计划 §2.4 **作显式设计决策记录**(依据=冗余非缺失);新增无肢拓扑时重审 |
| (i) 计划文档口径回写 | 102,438 clips / 194 拓扑 / 382 rigs;§3.2 tpos 校准路径对 PZ 无效 | 修订计划 §P0.4、§3.2(校准输入改为 motion-frame,数字改 194/382 并注明定义) |

---

## 三、Phase 2 就绪度评估

### 前提已满足

| Gate / 前提 | 状态 | 依据 |
|---|---|---|
| D=14 通道布局前提(ch2 复用 + root ch12:14 heading) | ✅ 满足 | P0.1 ch2 全零 + P0.3 ch12 冗余可退役 |
| 方案 (f) heading 运行时代数 | ✅ 满足 | q_rest yaw-only 382/382;twistY = 精确 yaw;零奇异暴露 |
| φ_skel 表输入(G3 heading 重算一致 <1° 的前提) | ✅ 数据齐 | 382/382 全覆盖、0 method-less、root 索引交叉核对 PASS |
| ε / θ_max 定标数据 | ✅ 齐备 | ε=1e-4 tripwire;θ_max 分布 p50-p99.99 + 2,323-clip 清单在册 |
| G1/G5/G6 可实施性 | ✅ 无新障碍 | 拓扑口径用 194/382 修订;yaw 等变闭式变换被 yaw-only 事实简化 |
| tpos 作 q_rest 来源 | ✅ 有效 | tpos 几何精确 rest(2.4e-8),root 行 rot6d yaw-only |
| 语料口径回写 | ✅ 数字就绪 | 102,438 / 194 / 382 / 9,985,438 帧 |

### 受阻 / 未完成(Phase 2 启动前必须清掉)

| 项 | 性质 | 说明 |
|---|---|---|
| **Phase 0.5 决策 (a)-(e) 未拍板** | ⛔ 硬阻塞 | 计划 Phase 0 的 Verify = "决策记录进文档 + user 拍板";重导出 scope 取决于 (a)/(b) |
| **rest 朝向箭头 contact-sheet 视觉 QA 未做** | ⛔ Phase 1 Verify gate 未过 | G3 前置;64 NO_MOVING_CLIP + 104 needs_review + 4 FLIPPED 只能靠它仲裁;按 QA-primacy 发 user 审 |
| **校准脚本约定未过 codex 审** | ⛔ 铁律 | θ=atan2(x,z)、forward=(L−R)×up、φ_skel 表等约定 bake 进迁移代码前必经 codex 审(校准产物 caveat 已自我声明) |
| **合成 roll/pitch/inversion 穿越用例未写** | ⚠ §3.2 第 5 条 | P1.3 只完成语料扫描;CPU 合成用例仍需实现(纯 pitch 不跳 π / fallback / 断言),纳入 G-cpu 套件 |
| **计划 §3.2 校准路径须先修订** | ⚠ 文档阻塞 | tpos rest 校准对 311 PZ rigs 无效已实证;motion-frame 校准转正 + 179→194/382 回写后 Phase 2 脚本才有正确规格 |
| **Δθ 超限清单(2,323 clips)未抽查** | ⚠ 数据质量 | ~180°/帧 teleport 样本(HML3D 001766/M001906、PZ Lemur/Quokka/Capuchin)需人工定性: 数据毛刺 vs 真动作 |
| **raw 源级奇异点重审(条件性)** | ⚠ 视 (a)/重编码方案而定 | 仅当 Phase 2 从 raw 源完整 3D root 朝向重推 heading 时必做;沿用现上游 yaw 提取则免 |

**一句话结论**: 数据侧前提(死通道、T-pose、contact 冗余、heading 代数、φ_skel 覆盖、定标分布)已全部实证到位且相互自洽,方案 A1 + per-crop 重锚的证据链完整;Phase 2 唯一的阻塞是**决策拍板 + contact-sheet 视觉 QA + 校准约定 codex 审**三件流程事,不是数据事。

---

## 附录: 脚本与产物路径

**脚本**(均在 `scratch/btjd_phase01/`,只读扫描):
- `_audit_ch2_tpos.py` — P0.1 + P0.2
- `_audit_root_contact.py` — P0.3
- `_audit_corpus_singularity.py` — P0.4 + P1.3
- `_calibrate_phi_skel.py` — P1.1 + P1.2

**关键结果文件**:
- P0.1/P0.2: `audit_ch2_tpos_full.json`、`conclusions_ch2_tpos.json`、`part_a_root_ch2_per_topology.tsv`、`part_b1_tpos_per_rig.tsv`
- P0.3: `root_contact_audit_full.json` / `.tsv`(+ sample500 对照)
- P0.4/P1.3: `corpus_singularity_audit_full.json`、`per_clip_smin_full.tsv`(19MB/102k 行,含 dtheta_max 列)、`per_rig_summary_full.tsv`、`_full_scan.log`
- P1.1/P1.2: `phi_skel_per_rig_full.tsv`(382 行主表)、`phi_skel_manual_needed_full.tsv`(104 rigs)、`probe_per_topology_full.tsv`(194 拓扑)、`probe_selected_clips_full.tsv`(620 clips)、`phi_skel_calibration_full.json`(方法学披露)、`_phi_skel_full.log`

注: 同目录 `full_scan.log`(无下划线前缀)属兄弟 Phase-0 任务,非本轮审计产物。
