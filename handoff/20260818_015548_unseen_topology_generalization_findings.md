# 未见拓扑泛化:问题重定义与诊断计划

> 状态:诊断进行中。本文档的「已确认事实」部分是读代码/读协议得出的硬结论,不会因诊断结果改变;
> 「待诊断」部分等 GPU 测量回来后补。

## 摘要

用户要求"提升泛化能力,允许微调,要经济的方案"。在设计任何方案之前先做了协议与代码的取证,
发现了三件会直接改变方案选择的事。按重要性排序:

1. **held 集不是同质的**,它横跨从"同一物种的另一性别"到"Dragon/Crab"的整个光谱。
   把它们平均成一个"unseen topology 性能"数字会掩盖真实情况。
2. **现有 evaluator 不符合协议**,它见过全部 held 拓扑。论文数字需要重训一个合规 evaluator。
3. **拓扑距离与数据稀缺高度耦合**,在 held 集内部无法归因。需要 retained 内部的对照组。

---

## 一、已确认事实(读协议与代码得出)

### 1.1 协议构成

`protocol/SEAL.json` 定义的 `unseen_topology_v1`(预注册、sha 锚定):

| split | clips | 用途 |
|---|---|---|
| train | 90,277 | 训练 |
| val | 4,777 | **唯一**允许用于选 checkpoint 的信号 |
| held_representative | 4,509 | TEST QUERY,冻结后才能碰 |
| held_stress | 2,875 | TEST QUERY,冻结后才能碰 |

- 179 棵拓扑树中 hold 掉 35 棵(48 个 object_type),移除 7,011 个训练 clip = 7.21%
- 协议红线原文:"No checkpoint, threshold, architecture or normalisation may be tuned against them."

### 1.2 held 集的真实构成 —— 按到最近保留拓扑的描述子距离分层

用 `holdout_topologies_v1.json` 的 `dist_to_nearest_retained` 对 7,384 个 held clip 分层
(全部 48 物种成功映射,无遗漏):

| 分层 | clips | 占比 | 物种数 | 该拓扑原有训练 clip(median) | 来源 |
|---|---|---|---|---|---|
| **A 近乎同拓扑** dist<0.1 | 3,514 | 47.6% | 18 | **300** | 全 PZ |
| **B 近亲** 0.1–0.5 | 1,843 | 25.0% | 11 | 365 | 混合 |
| **C 中距** 0.5–1.5 | 1,386 | 18.8% | 10 | 333 | 混合 |
| **D 远/陌生** >1.5 | 641 | **8.7%** | 9 | **28** | 主要 TrueBones |

A 层里有大量"同物种的另一变体":

| dist | held 物种 | 训练集里的最近邻 |
|---|---|---|
| 0.020 | PZ_Striped_Skunk | PZ_European_Badger_Juvenile |
| 0.049 | **PZ_Aardvark_Male** | **PZ_Aardvark_Female** |
| 0.052 | **PZ_Alpaca_Juvenile** | **PZ_Alpaca_Male** |
| 0.053 | **PZ_Siberian_Tiger_Female** | **PZ_Siberian_Tiger_Male** |
| 0.063 | **PZ_Alpine_Goat_Female/Male** | **PZ_Alpine_Goat_Juvenile** |

D 层才是真正陌生的:Scorpion-2 (3.385)、Dragon (3.067)、Bear (2.345)、Spider (2.251)、Crab (2.154)。

**推论一(评估口径)**:必须按分层报告,不能只报 representative/stress 两个总数。
后者会被 A 层的同物种变体拉高。

**推论二(产品视角)**:游戏里"加一个新动物"对应 A+B 层(新品种的猫科/熊科)= 72.5% 的 held 数据。
D 层是学术挑战,不是部署需求。若 A/B 层表现良好,方法对游戏落地已够用。

**推论三(诊断设计)**:A 层是**无借口层** —— 拓扑几乎相同 + 数据充足(median 300 clips)。
它崩 = 根本性泛化失败,没有数据稀缺可以解释。这是最干净的信号。

### 1.3 confound:拓扑距离 与 数据稀缺 高度耦合

A/B/C 层的原有训练数据 median 都是 300–365 clips,D 层只有 28 —— **差一个量级**。
协议的 `selection` 字段自己也写明,held 是从"训练 clip 数 ≤ max(2×中位数, 40)"的低资源拓扑里抽的,
并承认这只估计"合格子集"上的性能,不是整个语料。

后果:在 held 集内部**无法区分**"拓扑没见过"和"这类拓扑本来数据就少"。
必须用 retained 内部同样低资源的拓扑做对照组才能归因。

### 1.4 evaluator 不符合协议(必须修,影响论文)

- 协议 README:"train_main_retained is the ONLY evaluator training set"
- 实际:唯一的 evaluator(`anytop_t2m_evaluator_distilbert_..._v4b272_seed42`,2026-07-04 训)
  用的是 `train_main.json` **97,288 clips(全量)**,不是 `train_main_retained.json` **90,277**
- 即该 evaluator **见过全部 7,011 个 held clip 的拓扑**

影响:
- 论文里不能用它声称 unseen-topology 性能
- 但做**诊断**仍有效,且方向保守:连一个认识这些骨架的 evaluator 都判我们差,那就是真的差
- GT ceiling 与 recon/gen 共用同一 evaluator,**比值**不受这个偏差影响

代价:重训一个 retained-only evaluator ≈ 1 天 × 4 卡。目前空闲的 4×A100 只剩 23.7h,时间不够,
需要更长的 alloc。**这是必需品,不是可选项。**

### 1.5 架构与工程约束

- **模型本身是拓扑无关的**:`GraphPSCFFlowNet` 把 `pooled_adjacency` [B,C,C]、`pooled_geodesic` [B,C,C]
  和每关节的 LLM2Vec 语义嵌入作为**输入条件**,不是硬编码。所以泛化失败不是"架构不支持新骨架",
  而是分布问题。(`src/models/CodeFlow_Model/graph_pscf.py:369-410`)
- **VQVAE 是质量天花板**:flow 在冻结 VQVAE 的 latent 空间里生成,tokenizer 重建不了的东西
  flow 不可能生成出来。先验证据:animal-only VQVAE 重建 human,position L2 0.14 vs 同分布 0.027(差 5×)。
- **token cache 不含 held**:`data/codeflow_tokens_holdout_semantic_ep150_fulllen300` 只有
  train(90,277)+ val(4,777)。做 flow 侧的 teacher-forced 诊断需要先导出 held 的 token。
- **导出 held token 需要改代码**:`AnyTopDataset` 第 663 行硬拒绝 train/val/all 以外的 split 名
  (`src/data/anytop_dataset.py:663-664`)。而该文件正被两个在跑的训练加载 —— 改动有风险,
  只在确认必要后再做。
- **训练时拓扑增强不经济**:flow 训练跑在预导出的 token cache 上,任何训练时数据增强都要求
  重新导出整个 cache(142GB)+ 重训 flow。这条路成本按天算,不符合"经济"要求。
  相比之下 **per-skeleton few-shot 微调只需为目标骨架导出几十个 clip 的 token**,便宜得多。

### 1.6 关节语义对 held 骨架完整覆盖(不是瓶颈)

`data/joint_semantics_llm2vec_v1.npz` 按 object_type 存 [J, 4096] 的每关节 LLM2Vec 语义嵌入,
`data/joint_descriptions_v1.json` 按**关节名**索引 1,507 条描述(823 规则生成 + 684 LLM 生成,
94.3% 标为高置信)。

检查结果:**全部 48 个 held 物种都有嵌入**(Anaconda 27、Crab 54、Dragon 142、Spider 71、Pigeon 9,
PZ 的 held 物种同样齐全)。所以模型在推理 unseen 骨架时能拿到有意义的关节语义,这条路不是瓶颈。

附带的好消息:因为描述按**关节名**索引且由规则/LLM 生成(不依赖 motion 数据),
**部署时新增一个动物,只要有骨架关节名就能生成描述→嵌入,无需重训即可获得语义输入**。
这也意味着它不构成对 held 的信息泄漏(新骨架在部署时同样可得)。

### 1.7 评估 held 集的两个静默陷阱(codex r1 抓到,已修)

准备诊断时提交 codex(gpt-5.6-sol max)审查,verdict = NEEDS-FIX,其中两条是 CRITICAL,
**若不修则诊断会给出看似正常、实则错误的数字**:

1. **`split="val"` 硬编码 → 静默只加载 5% 数据。**
   `scripts/_eval_vqvae_recon_textR.py` 原本把 `split="val"` 写死传给底层 dataset。但一个被
   hold 掉的拓扑会把它**全部** clip(train+val 两边)贡献给测试集,所以 held manifest 里
   只有 227/4,509(representative)和 146/2,875(stress)属于原始 val 列表 —— 其余来自 train,
   会被静默丢弃,而脚本照常输出完整报告。
   修法:新增可选 `--base_split`(默认 None = 保持旧行为),held 集必须传 `all`。
   `AnyTopT2MEvalDataset` 早已为此预留了 `base_split` 参数,源码注释也写明了这个用法
   (`src/data/anytop_t2m_eval_dataset.py:206-212`)。

2. **joint semantics 未传 → encode 直接失败。**
   该 VQVAE 训练时带 4096-D 关节语义表(`--joint_semantics data/joint_semantics_llm2vec_v1.npz`),
   但 recon 脚本没把它传给 dataset。修法:从 VQVAE ckpt 自己的 args 取
   (`ta.get("joint_semantics")`),这样永远不会和 tokenizer 训练时用的表漂移;ckpt 若不带语义
   则取到 None,旧行为逐字节不变。

另两条 HIGH 也已修:`set -euo pipefail` + 原子写 + JSON 完整性校验(防止崩溃的半个 JSON 被当作
"已完成"跳过);以及**证明**诊断只写 scratch/ —— `AnyTopDataset` 在 `_cond_normalized_J144.pkl`
缺失或旧于 `cond.npy` 时会重算并重写它(`src/data/anytop_dataset.py:595`),而那个文件正被两个在跑的
训练共用。脚本现在在开头断言该缓存存在且更新(实测新 87 秒),不满足就 fail-closed 拒绝运行,
从而保证走只读分支;另加 PYTHONDONTWRITEBYTECODE / XDG_CACHE_HOME / MPLCONFIGDIR / HF_HOME /
CUDA_CACHE_PATH / TMPDIR 全部指向 scratch/。

**教训(跨任务可复用)**:凡是拿 held_* manifest 做评估,`base_split="all"` 是必需的;
漏了不会报错,只会安静地少算 95%。

### 1.8 species-stripped 消融不可行(负面结论,免得重复尝试)

manifest 每条 clip 带多个 caption 变体,部分含 species-stripped 版本("An animal is..."
而非 "A bear is..."),看起来可以用来测"物种名陌生是否导致文本路失效"。**实测覆盖率不足**:

| split | has_species_stripped | 覆盖率 |
|---|---|---|
| val_all_retained | 53 / 4,777 | 1.1% |
| held_representative | 102 / 4,509 | 2.3% |
| held_stress | 110 / 2,875 | 3.8% |

且 `species_stripped_cap_idx` 不固定(0/1/2/3 都出现),cap0 有时本身就是 stripped 版
(`Bear___BackUp_85` 的 cap0 = "An animal is cautiously walking backward"),口径不干净。
样本量与口径都不支持有统计力的消融 → **放弃这条**。
替代:直接看 held 上的 **text→GT ceiling**(诊断已内建)。若连真实动作都检索不出来,
说明文本-动作对齐在 held 上本身就难,再进一步区分是物种名陌生还是 evaluator OOD。

---

## 一点五、方法开发的协议死结,与建议的解法

**死结**:few-shot 微调的方法开发,本质上需要一个"没见过目标拓扑"的基座模型来验证。而:
- 用 retained 拓扑模拟 few-shot → 基座早已见过它,结果虚高,不是真 few-shot
- 用 held 拓扑开发 → 违反协议红线("No checkpoint, threshold, architecture or normalisation
  may be tuned against them")
- 重训一个 leave-one-topology-out 基座 → 几天 GPU,不符合"经济"要求

**建议解法:把两个 held 集拆成 DEV / TEST。**

| | A <0.1 | B 0.1–0.5 | C 0.5–1.5 | D >1.5 | 总计 |
|---|---|---|---|---|---|
| held_representative | 2,736 | 1,061 | 651 | **61** | 4,509 |
| held_stress | 778 | 782 | 735 | **580** | 2,875 |

**DEV = held_representative(4,509)/ TEST = held_stress(2,875)**,理由:
- stress 的四层分布**相当均匀**(778/782/735/580),做最终测试能可靠覆盖各距离段
- representative 的 D 层只有 61 clips,当 TEST 用无法测远距离泛化;当 DEV 用则 A/B/C 有 4,448 clips,足够
- 语义上也吻合原命名:representative = 平均情况(开发),stress = 压力测试(验证)

**代价**:失去 representative 的独立测试意义,论文里必须明确声明这一拆分及其理由。
**这是需要用户拍板的方法论决策**,因为它修改了预注册协议的使用方式。

---

## 二、诊断计划(先定位瓶颈层,再谈干预)

修错层会浪费几天,所以先诊断。四个互斥假设:

| 假设 | 含义 | 判据 |
|---|---|---|
| A | 冻结 VQVAE 是墙 —— 它重建不了 unseen 拓扑 | VQVAE recon 在 held 上显著劣于 retained |
| B | flow 是墙 —— tokenizer 重建良好但 flow 生成不出对的 token 分布 | recon 好但 gen 差 |
| C | evaluator 在 held 上 OOD,指标本身不可信 | GT ceiling 在 held 上也大幅下降 |
| D | 文本条件路对未见物种名/关节语义失效 | 需 joint-semantics 消融 |

### 第一步(进行中):VQVAE 在 held 上的重建质量

`scratch/_diag_vqvae_heldtopo.sh` — 三个 split(val_all_retained / held_representative / held_stress)
过冻结 VQVAE(ep150)做 encode→RVQ snap→decode,在 evaluator 空间量 text→recon vs text→GT(ceiling)、
FID、diversity。

为什么先做这个:
- 它决定 flow 的天花板。若 tokenizer 是墙,所有 flow 侧方案(LoRA/adapter/增强)都无效。
- **零代码改动**,走 manifest 不走 splits 机制。
- VQVAE 自 2026-08-03 起已冻结且正被两个训练当作 tokenizer 使用,对它做 held 测量是
  只读 TEST 测量,不选择、不调参,**不触协议红线**。
- 结果按 §1.2 的 A/B/C/D 分层解读,重点看 A 层。

（v2b / v3 的 held 生成评估要等它们训完冻结后再做,现在做会构成"用 held 选 ckpt"。）

### 2.1 判据 —— 在看到任何数字之前声明(防止事后定标准)

诊断脚本报告 `protocol_text_to_recon_rprec`(text→重建)、`protocol_text_to_gt_rprec`(text→真实动作,
即 ceiling)、`fid_gt_vs_recon`、`matching_*`。因为 evaluator 在 held 上本身偏乐观(§1.4),
**绝对值不可跨 split 比较**,只有比值可比。故定义:

> **保真比 F = R@1(text→recon) / R@1(text→GT ceiling)**,在同一 split 内计算,
> 分子分母共享同一 evaluator 偏差,故偏差在比值里大部分抵消。

以 `val_all_retained` 的保真比为基准 B,held 各 bucket 的保真比为 H,**事先约定**:

| H / B | 判定 |
|---|---|
| ≥ 0.95 | tokenizer 在该 bucket 上泛化良好,**不是**瓶颈 → 转向 flow 侧诊断 |
| 0.80 – 0.95 | 轻度劣化,次要因素 → 记录但不作为主攻方向 |
| < 0.80 | tokenizer 显著劣化,**是主要瓶颈** → 所有 flow 侧方案(LoRA/增强/微调)无效,必须先修 tokenizer |

**独立的第二判据**(纯 motion-space,不经过文本,故不受物种名陌生影响):
`fid_gt_vs_recon` 在 held 上若 > 3× retained 的值 → 重建分布明显偏移。
两个判据互为交叉验证;若二者结论冲突,以 FID 为准并记录冲突(说明文本路另有问题)。

**注意统计力**:truebones bucket 在两个 held split 上分别只有 150 / 125 clips,低于 `fid_min=1024`,
**不会有 FID**,只有 R-precision;且 pool=32 时它们的 R-precision 噪声较大。
所以 truebones bucket 的结论只能是定性的,**不得**据此下强判断。
主判断落在 animal bucket(4,359 / 2,750 clips),那也正是 A/B 层所在、且数据充足的部分 —— §1.2 的"无借口层"。

---

## 二点五、我独立提出的候选(待与文献调研结果合并排序)

### 候选 A:低资源拓扑上采样 + 短程微调 —— 最便宜、有项目先例

**机制**:改 sampler,把训练集中低资源拓扑(train_clips ≤ 40,共 53 棵树 / 620 clips,
全为 TrueBones,见 §1.3 对照组分析)的采样概率提高,在现有 v2b/v3 上微调 20–50 epoch。

**为什么可能有效**:held 拓扑是从低资源集合里抽的(协议 `selection` 字段),所以泛化目标天然
偏向稀缺形态。提高稀缺形态的梯度贡献,模型或许能学到更通用的「拓扑→动作」映射。

**成本**:20–50 epoch × 27–35 min × 4 卡 ≈ **36–90 GPU-h**,在 326 GPU-h 预算内。
**关键优势:不需要重导 142GB token cache** —— 上采样只改采样顺序,与预导出 cache 完全兼容。

**先例**:`src/data/human_curriculum_sampler.py` 已实现同类机制(当初按 human/animal 分组做
60% 上采样),实测 human 重建 rot6d MSE −41%、FID −52%。换个分组维度(按拓扑资源量而非物种类别)
即可复用,代码改动小。

**诚实的局限**:上采样直接改善的是**训练集里**的低资源拓扑;held 拓扑根本不在训练集里,采样不到。
它对 held 的帮助是**间接且假设性的** —— 赌"见过更多样的稀缺形态 → 对新形态泛化更好"。
这个假设可以用 20 epoch 微调便宜地证伪,但**不得当作已知有效**。
另外它与 §1.3 的 confound 直接相关:若诊断显示稀缺才是主因,此候选优先级上升;
若显示拓扑距离才是主因,则此候选大概率无效。

**因此:此候选的优先级取决于诊断结果,不应在诊断前启动。**


---

## 二点七、对抗审推翻了 §2 的诊断设计(2026-08-18,必读)

5 路文献/代码调研 + 综合 + 3 路对抗审的 workflow 跑完。**综合 agent 因 API 断连失败,`plan` 返回 null。
对抗审 agent 没有编造计划来评审 —— 它 fail loud 声明了这一点,转而审查本文档**,并给出以下结论。
综合已 resume 重跑。

### 被推翻的(必须在跑诊断前修,否则数字无意义)

**F1 — §2.1 的判据挂在错误的分组上。** `motion_id_bucket()`(`src/eval/codeflow_gen_eval.py:123-132`)
是 `HML3D*→human` / `PZ_*→animal` / 其余→`truebones`,这是**数据集来源**轴,不是拓扑距离轴。
我自己的数字就证伪了自己的解读:held_stress 有 **580** 个 dist>1.5 的 D 层 clip,但 truebones bucket
只有 **~125** 个 —— 所以**至少 455 个最难的远拓扑 clip 藏在 "animal" 里**(PZ_Armadillo dist 1.94、
PZ_Porcupine 1.27、PZ_Anteater 0.85 等)。"animal" 是 A 层与 D 层的未知比例混合,判据用上去什么也决定不了;
而 §1.2 单独拎出 A 层的全部意义 —— 那个"无借口"数字 —— 从未被真正计算过。
**修法:按 `dist_to_nearest_retained` 分 A/B/C/D + per-species 报告。**

**F2 — 主判据选错了工具。** 假设 A 是**几何重建**问题(项目自己的先验证据也是几何的:
position L2 0.14 vs 0.027),却要用 evaluator 空间的检索指标测,而 evaluator 已知不合规。三重问题:
(i) pool-32 caption 检索太粗 —— "一个重建成抽搐 blob 的 Crab 仍会被 'an animal moves forward' 检索到",
这正是本项目反复栽的 metric-乐观陷阱;(ii) 项目已测得 recon 在 text-align 上近无损(F≈0.99),
F 被钉在天花板上,向下几乎没有动态范围,噪声主导;(iii) 把 tokenizer 问题绕道经过唯一已知坏掉的组件。
**修法:主读数改为 evaluator-free 的几何误差**(world-position L2 / rot6d→FK / jitter / speed_ratio),
直接可比先前的 human 5× 结果,n=150 时统计力好得多,协议上无疑义。text→recon F 降为**次要佐证**。

**F5 — "比值抵消 evaluator 偏差"是断言而非成立,且方向未定。** 只有当偏差是两侧的乘性常数时才成立。
一个**记住了** held GT 的 evaluator 会很好地检索 GT,却可能在稍微偏离记忆流形的 recon 上崩掉 ——
这会特别压低 held 的 F,让 tokenizer 显得是瓶颈而其实不是;反方向同样可论证。
"方向保守"只对**失败**结果有效,对**通过**结果零保护 —— 而通过是更可能的结果。

**F7 — 判据没有置信区间,且 `reps=20` 测错了方差。** `shuffle_pool_rprec` 是对**固定 clip 集**做 20 次
重洗,那是 pool 分配噪声,不是 clip 抽样噪声。n=150 + pool=32 → 约 4 pools/rep,R@1 的 SE 约 ±0.04,
两个这种量的比值更差。**修法:对 clip 做 bootstrap,报 95% CI,CI 跨越阈值则不行动。**

**F3 — 最可能的结果("不是 A")会撞墙。** §1.5 已记录 held token 不在 cache、`AnyTopDataset:663-664`
拒绝非 train/val/all 的 split,而该文件被两个在跑的训练 import。所以走到"下一步做 flow 侧诊断"时无路可走。
**修法:现在就把 held token 导出的范围与成本定下来,并优先用 manifest 驱动的读路径,而不是放宽 split 白名单。**

**F6 — FID 跨 split 不可比**(n=4777/4509/2875,FID 强烈受样本量偏置,"held > 3× retained" 会部分因 n 触发)。
**F4 — 四个假设并非互斥**(C 是与 A/B/D 共存的测量混淆,应降级为被测量的 nuisance;D 已被 §1.8 杀掉)。

### 我的两处硬错误(如实记录)

**算力预算算错了。** §2.5 写的 326 GPU-h 是错的 —— 实际是 88(4×H100×22h)+ 112(4×H200×28h)= **200**,
而且是**两个不可合并的窗口**;我错误地把 1×H100(30)和 4×A100(96)也并进同一个池子,而它们型号不同、
窗口更短。同时 50-epoch 微调的上界 50×35min = 29.2h wall × 4 = **117 GPU-h,不是 90**,
且 29.2h wall **超过两个 idle 窗口**,需要我没预算的 resume 机制。
合规 evaluator(96)+ 50-ep 微调(117)= **213 > 200,在做任何诊断之前就已超额**。

**§1.5 的 DEV/TEST 提议不成立,撤回。** 我以为 rep 当 DEV、stress 当 TEST 就干净,实测两边**不独立**:
- `Siberian_Tiger` 家族横跨两边(rep: Juvenile 208 clips / stress: Female 316 clips)
- 用"共享同一最近保留邻居"作拓扑邻近代理,查出 3 组交叉:
  PZ_Aardvark_Female 组(DEV: Monkey+PZ_Aardvark_Male / TEST: PZ_Porcupine 351)、
  PZ_African_Elephant_Female 组(DEV: Scorpion-2 / TEST: PZ_Indian_Elephant ×3 = 673)、
  PZ_Alpaca_Male 组(DEV: PZ_Alpaca_Juvenile / TEST: PZ_Dromedary_Camel + PZ_Llama = 277)
- **受污染的 TEST clip:1,301 / 2,875 = 45.3%**(还没算 Siberian_Tiger 的 316)

在 rep 上调的任何超参会按构造迁移到 stress。若仍要拆分,最低诚实形式是**按拓扑家族+邻近分区、
隔离两边都出现的家族**,TEST 将缩到约 1,250 clips(统计力显著下降),并在论文里声明协议修改。
**这是需要用户拍板的取舍,不是我能单方面决定的。**

### 被确认站得住的

对抗审明确背书:§1.2 的 A/B/C/D 分层("文档里最有价值的东西",47.6% 的 held 是 dist<0.1 的近重复,
把整个问题重新定义了)、§1.4 抓出 evaluator 不合规("正是那种防止论文被撤回的自审")、
§1.3 明确命名 confound、§1.7 的完整性修复("careful, correct engineering",尤其那条保护两个在跑训练的
fail-closed cond-cache 断言)、§1.8 记录负面结论的纪律、§1.6 关节语义的部署含义。

### 已执行的修复

新增 `scripts/_diag_vqvae_geom_stratified.py`(238 行,已提交 codex 审):**完全不用 evaluator**,
按 `dist_to_nearest_retained` 分 A/B/C/D + per-species,输出 per-clip 记录供 bootstrap CI;
用 `recover_from_bvh_rot_np`(严禁脚本内复制,double-root-rotation 已踩两次)做 rot6d FK;
内建 **GT 自检 gate**(同一 GT 用 rot6d-FK 与 RIC 两条路各恢复一次世界坐标,不一致超过容差就中止 ——
因为那意味着 de-norm/FK 路径错了,报告里每个数字都无效)。
静态验证已抓到一个真 bug:字段是 `num_joints` 不是 `n_joints`;已改为全部走 `GraphMotionBatch` dataclass 字段。


---

## 二点八、方案综合(workflow 重跑成功)+ 一个可能是主因的新发现

### ★ 最重要的发现:我们的评估用了一个部署时不存在的条件

`src/data/moment_source.py` 与 **FACTS.md C10**(2026-08-02 实测,在 334 个 retained object type 上拟合、
48 个 held 上评分,从未拟合到 held)记录:

| 矩通道 | 从静态 rest pose 估计的误差(fold) |
|---|---|
| non-root rotation (3:9) | 1.15× |
| non-root position (0:3) | 1.26× |
| root heading | 1.33× |
| non-root velocity | 1.38× |
| root XZ velocity | 1.73× |
| contact | 1.82× |
| **root height** | **3.02×** |

原文:*"A 3x error on root height is an animal that floats or sinks. **It is invisible to retrieval metrics.**"*
且没有静态特征能救(vertical extent 2.82×、median bone length 4.10×、常数 3.08×)。

**推论(此前无人指出)**:所有评估默认 `--moment_policy own`,即从**该骨架自己的 motion** 取归一化统计。
这是 **transductive** 的 —— 一个真正的新骨架**没有**自己的 motion 统计可用。部署("游戏里加一个新动物")
只能用 `estimated`,而 root height 差 3×,动物会漂浮或下沉,**且 R-precision 完全看不出来**。

**所以我们很可能系统性高估了真实的"加新动物"性能,而这与拓扑新颖性无关。**
这个假设可以用 ~1.5 GPU-h 在 **retained** 数据上证伪:同一批 clip,`own` vs `estimated` 各跑一次。
若在 retained 上换成 estimated 就崩,那么我们一直归因于"unseen-topology failure"的东西,
有很大一部分其实是**归一化失败** —— 而它的修复是改进矩估计器(~0 GPU-h),不是重训模型。
基础设施已就位:`scratch/_moment_est_v2.npz`(83KB)+ `scripts/_fit_restpose_moment_estimator.py`。

### 方案的核心设计:整个诊断只用 RETAINED,完全不读 held

比"预注册一次读 held"更严格,且不增加成本。机制在 retained 内部作为**剂量-反应曲线**测量后外推:
- 对每个 retained object type 算 `nov_LOO` = 到最近的**其他** retained type 的描述子距离
  (留一法新颖度,不需要任何 held 数据)
- held 的新颖度水平只从**已冻结**的 artifact 字段 `dist_to_nearest_retained` 读(范围 0.0204–3.3854)
- 分层量 `tokenizer_gap(nov) = 1 − rho_recon`、`flow_gap(nov) = rho_recon − rho_gen`,对 `nov_LOO` 拟合
- **决策规则:gap 随新颖度斜率更陡的那层是墙。nov=0 处的绝对值说明不了任何泛化问题,只有斜率有意义。**
- **内建证伪**:若两条曲线都平坦 → 结构新颖性不是机制(27 个 held_representative 类型里 12 个
  dist<0.07,如 PZ_Aardvark_Male 紧挨 retained 的 PZ_Aardvark_Female)→ 转向 per-species 数据量与归一化

这一刀绕开了 §2.7 记录的协议死结:**不需要 DEV/TEST 拆分(我那个 45% 污染的提议作废),不需要读 held。**

### 预算算术(决定性,且推翻"诊断哪层就修哪层")

实测:flow 全量重训 1633 s/ep × 8 卡 = 3.63 GPU-h/ep → 300 ep ≈ **1090 GPU-h(预算 200 的 5.5 倍)**;
VQVAE 重训 177 GPU-h + token 重导 ~10 GPU-h,**但新 VQVAE 改变 latent 空间 → flow 完全失效 → 强制那 1090**。

> **任何 tokenizer 侧的修复在这个窗口里端到端都负担不起。**
> 所以诊断的真正任务不是"找到坏的那层然后修",而是:决定这个窗口买到的是一个**改进**,
> 还是一份**刻画 + 正确计价的提案**。两者都是正当结果。

### 执行序列与 gate(每个 gate 都能早停,避免烧掉读不懂的实验)

- **PHASE 0**(0 GPU-h):D0 结构新颖性审计 + DEV split + PREREG.json;codex 审两个小脚本;
  **用 git worktree 隔离,确保任何编辑都碰不到 LIVE 的 v3_xpred**
  - ⚠ **地雷**:`scripts/_build_holdout_splits.py` 只从**原始** splits 减去它自己 artifact 的树 ——
    在 dev artifact 上跑会**静默把所有 held_* clip 重新放回 dev_train**。
    必须改用对冻结 holdout 列表做集合差,并硬断言 `dev_train ∩ (held_rep ∪ held_stress ∪ dev_held) = ∅`
- **PHASE 1**(~10 GPU-h,只用 retained):D1 地板+仪器合理性 → D2 剂量-反应 → D3 矩探针 → D4 视觉 gate → C3 cfg sweep
  - **GATE A**:两条 gap 都平坦 → 停止拓扑路线,立即报告用户(框架需修订,"这比任何 80 GPU-h 的臂更有价值")
  - **GATE B**:tokenizer 斜率占优 → **不要**在本窗口启动 VQVAE 重训;把预算用于刻画到论文质量 +
    LOSO 确认 + 交付一份 ~1270 GPU-h 的联合重训计价提案。"告诉用户这个窗口修不了、这是真实价格和正确配方,
    是正确答案,不是失败。"
  - **GATE C**:若 cfg sweep 已关闭相当部分外推 gap → 记账(0 GPU-h / 0 推理成本)但重新基线化
- **PHASE 2**(~80–100 GPU-h,仍不碰 held):先 D5 leak gate(含第二控制种子测 run-to-run 噪声)
  - **GATE D**:控制组 dev_held 与 retained-val 无法区分 → DEV 无判别力 → **不要跑 80 GPU-h 你读不懂的实验**
  - 有判别力则跑**恰好两个**臂 ×20 epoch,分置不同节点型号(H100/H200 不混):
    ARM-CTRL(纯微调)vs ARM-C1C2(bucketed topology bias + per-block skeleton re-injection,均零初始化,
    + skeleton-conditioning dropout p=0.3)
  - **两条硬 gate 覆盖 metric**:(1) retained-val 不得退化 —— 用遗忘换来的 unseen 提升不算提升;
    (2) 三栏渲染必须发用户并通过视觉裁决(按项目历史 bone_cos 0.92 却视觉崩,渲染高于数字)
  - **GATE E**:delta 落在 D5 噪声内 → 平实报告负面结果


---

## 二点九、PHASE 0 已执行(0 GPU-h),方案比预期更强

今夜实际跑完了方案 PHASE 0 的核心分析,三项结果:

### (1) 描述子度量已对齐冻结 artifact —— 相关系数 1.000

6 个描述子(J / max_branching / n_leaves / max_depth / mean_depth / n_branch_nodes)**全部可从
`scratch/_trees.json` 的 `sig`(父节点数组)现算**,不需要读 `cond.npy`(18.4 MB),零 IO、秒级。

用「全树 z-score 后欧氏距离」复现 artifact 已记录的 `dist_to_nearest_retained`:
**Spearman = 1.000,Pearson = 1.000,中位比值 1.02**(2% 尺度差来自 z-score 样本集不同:我用 194 树,
artifact 用 179)。序完全一致 ⇒ `nov_LOO` 与 artifact 的 `dist` **直接可比**,这是后续一切的前提。

### (2) retained 内部的新颖度范围**完整覆盖** held —— 不需要外推

| 分位 | retained `nov_LOO`(n=158) | held `dist`(n=35) |
|---|---|---|
| p25 | 0.050 | 0.058 |
| p50 | 0.201 | 0.405 |
| p75 | 0.575 | 1.326 |
| p90 | 0.888 | 2.212 |
| **p100** | **3.459** | **3.385** |

retained 最大 **3.459 > held 最大 3.385**。方案原本设计为"在 retained 测曲线、外推到 held 新颖度水平",
**实测根本不用外推** —— retained 内部就能直接覆盖 held 的全部新颖度区间(含最极端的 Dragon/Scorpion-2 那一档)。
这比方案预期强:剂量-反应曲线在支撑集内插值即可,外推假设(线性?)不再是弱点。

### (3) §1.3 的 confound 可以拆开 —— retained 内部存在「高新颖 + 数据充足」象限

按 `nov_LOO ≥ 0.5` × `train_clips ≥ 100` 切 2×2(retained 158 棵树):

| 象限 | 树数 | train clips | val clips |
|---|---|---|---|
| 低新颖 / 少数据 | 23 | 601 | 39 |
| 低新颖 / 多数据 | 90 | 86,467 | 4,554 |
| 高新颖 / 少数据 | 37 | 529 | 43 |
| **高新颖 / 多数据** | **8** | **2,680** | **141** |

关键第四象限的成员:

| object type | nov_LOO | train | val |
|---|---|---|---|
| PZ_Giant_Anteater_Juvenile | **3.459** | 224 | 12 |
| PZ_Aldabra_Giant_Tortoise_Female | 1.111 | **790** | 41 |
| PZ_Aardvark_Female | 0.888 | 175 | 9 |
| PZ_Wisent_Male | 0.687 | 522 | 28 |
| PZ_Red_Ruffed_Lemur_Male | 0.578 | 485 | 26 |

**在 held 集里这个象限不存在**(D 层的 train_clips median 只有 28),所以"拓扑新颖 vs 数据稀缺"
的归因在 held 内部**不可能**做;在 retained 内部**可以**。141 个 val clip 对几何指标
(per-clip 标量 + clip-bootstrap CI)足够;对分布指标(FID,需 n≥1024)不够,故几何为主判据、FID 不用于此对照。

**结论:方案 PHASE 1 的 D2 剂量-反应曲线可以直接执行,且比原设计多一个自变量(数据量),
可同时回答"是拓扑新颖还是数据稀缺"。**


---

## 三、交接状态(2026-08-18 06:20 BST)—— 诊断脚本就绪但**未跑**

### 几何诊断脚本 `scripts/_diag_vqvae_geom_stratified.py` 的审查历程(如实记录)

| 轮次 | 结果 |
|---|---|
| r1(完整 brief,6 大项) | **失败**:跑 2h20m 后退出,输出 **0 字节**。前三轮审查分别产出 8101/8090/5333 字节,故确定是失败而非通过。原因不明(API 错误或输出被吞)。教训:brief 给太重(6 大项 + 跨文件比对)。 |
| r2(精简 brief,3 项,输出落盘不走管道) | **NEEDS-FIX**,4 个 finding,**但确认核心几何正确**:"De-normalization/FK path is **identical in effect** to `animate_vqvae_recon.py`" |
| r3(delta 复审) | **跑偏**:输出膨胀到 1.1 MB,内容是全仓 grep 历史审计文件而非验证 delta(它读到一个要求"独立审查"的 skill,触发全仓扫描),始终未给 verdict。 |

### r2 指出并已修复的 4 项

1. **[HIGH 运行时阻塞]** `_load_vq()` 从 `src.models.vq_model` import 了不存在的 `load_graph_vqvae`
   (该包只导出 `GraphVQTokenizer` / `semantic_config_from_ckpt`)。已改为与
   `scripts/_eval_vqvae_recon_textR.py:29-33` 相同的方式:importlib 动态加载
   `animate_vqvae_recon.py` 并取其 `load_vq_tokenizer`。
2. **[MED]** `--selfcheck_tol` 默认 0.05 **松了 5 个数量级**。实测 GT self-check 为
   **3.6e-08 ~ 5.9e-07**(`renders/20260807_vqrecon_flagship/recon_summary.txt` 等三份佐证)。
   已改为 **1e-3**(仍留约 1000× 余量)。
3. **[secondary]** `vq_cap = max_coarse * stride` 是范畴错误 —— `max_coarse` 是**空间**上限
   (coarse joint groups,96 是为容纳 Dragon 的 142 关节),不是帧预算。已移除,直接用
   `num_frames=300`(项目 live 假设 T_fine_max=300 / stride 4 / T_lat 75,token cache 正是这么导的)。
4. **[HIGH]** `--out` 现在强制解析路径中必须含 `scratch`,防止写到共享数据根或 run 目录。

另:r2 确认 **private mount namespace + 只读 bind mount 足以**保护共享数据根
(所有 cache 写都在 `cache_path.parent == data_root` 下),该方案已在计算节点实测
(`WRITE_BLOCKED_GOOD` / `READ_OK` / `SCRATCH_WRITABLE_GOOD`)。

### 为什么没有跑

四项修复是照 codex 明确指示做的,但**修完后未再获得成功的复审**(r3 跑偏)。
按铁律"代码改必经 codex 审"才跑,故**不启动**。今晚铁律已两次拦下会毁掉结论的问题
(静默只跑 5% 数据、encode 直接失败),不值得为赶时间破例。

### 建议的第一步(醒来后)

先跑 40-clip smoke —— 脚本内建的 **GT 自检 gate** 是对几何路径最强的实证检验:
同一份 GT 用 rot6d-FK 与 RIC 两条独立路径各恢复一次世界坐标,若 de-norm/FK 有任何错误,
自检不可能落在 1e-3 以内(实测正确时约 1e-7)。

```
srun --jobid=<空闲alloc> --overlap --gres=gpu:1 python3 scripts/_diag_vqvae_geom_stratified.py \
  --vqvae_ckpt runs/holdout_vqvae_semantic_8card_v1/ep150_model.pt \
  --manifest data/holdout_eval_splits_v1/val_all_retained.json \
  --data_root data/animo4d_L4TB_plus_human_v4b272neutral \
  --base_split all --encode_batch 8 --max_clips 40 \
  --out scratch/_smoke_geom_retval.json
```
(用 `val_all_retained` 而非 held —— 符合方案"只用 retained"的原则,零协议风险。)
自检过了再跑全量,并优先做 **D3 矩探针**(§2.8 ★),它可能推翻当前归因。


## 三、待补(原)

- [ ] VQVAE held 重建结果 + 分层解读
- [ ] 文献调研与方案排序(5 路并行调研 + 综合 + 3 路对抗审,进行中)
- [ ] retained 内部低资源拓扑对照组(拆解 §1.3 的 confound)
- [ ] joint-semantics 消融(测量它对拓扑泛化的实际贡献)
- [ ] 合规 evaluator 重训的资源决策(用户拍板)

---

## 四、对 held 集的一次性视觉读取(2026-08-18,用户显式要求)

用户看过 v2b / v3 的 in-distribution 渲染后,判定 "v3 感觉比 v2 要好不少",并要求
"找一些 test 里面别的没训练过的测试一下"。据此对 held(未训练)骨架做了一次**只读视觉检查**。

**协议记录**:此次读取**不选择 checkpoint、不调整任何参数、不作为任何训练或超参决策的依据**。
它是交付给用户的视觉证据。若日后需要报告协议数字,仍须用冻结模型 + 合规 evaluator 重跑(见 §1.4)。

**意外发现**:此前的"in-distribution"渲染其实已经包含一个未训练骨架 ——
`PZ_Caracal_Male` 属于 held_representative(dist 0.145,最近训练骨架 PZ_Cheetah_Female)。
原因是 `scripts/animate_graph_codeflow.py` 用 `AnyTopDataset(split='val')` 且**不传 splits_dir**,
因此读的是数据集原始 `splits/val.txt`(仍含 held 物种),而非 `data/holdout_splits_v1/val.txt`。
⚠ 这意味着**任何用该渲染脚本产出的 "val" 结果都混有 held 物种**,报告时必须注明。

而 Caracal 的表现是全场最好之一:v3 20 步 proj_err **0.2446(五个物种里最低)**、5 步 0.2347 ——
**未见骨架上的几何精度反而优于见过的骨架**,与"A 层是近乎同拓扑的变体"这一结构性发现一致(§1.2)。

**本次选取的 9 个未训练骨架**(按 dist_to_nearest_retained 排序,覆盖全谱):
PZ_Quokka_Female (0.048) / PZ_Wolverine_Male (0.049) / PZ_Indian_Elephant_Male (0.279) /
PZ_Giant_Panda_Male (0.653) / PZ_African_Crested_Porcupine_Male (1.265) /
PZ_Nine_Banded_Armadillo_Male (1.940) / Spider (2.251,八足) / Dragon (3.067,142 关节) /
Scorpion-2 (3.385,最陌生)

脚本 `scratch/_render_v3_unseen.sh`,输出 `renders/20260818_v3_ep286_UNSEEN_cfg2_s5`。
