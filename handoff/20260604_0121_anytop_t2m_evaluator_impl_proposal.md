# AnyTop T2M Evaluator 实现方案（供审核）

Date: 2026-06-04 01:21 UTC
Scope: 方案文档，供你审核。本文不改任何训练代码。
输入: `handoff/20260604_0015_anytop_t2m_evaluator_split_plan.md`（你的设计）
关系: 我读了相关代码、验证了你计划里的全部代码假设、盘点了可复用资产，并就两个有张力的设计点给出分析。审核通过后再进入实现（M0–M2）。

---

## 1. 一句话结论

你的计划**整体可行、假设全部验证通过、824/3288 数字可代码复现**。可以按 M0→M2 推进。但有 **2 个设计点建议你先拍板**（都关系到评估的可信度与实现量），见 §3。

---

## 1b. 审核结论与最终决策（2026-06-04，用户 PASS with required amendments）

状态：**PASS with required amendments** — 核心路线无硬伤，按以下定论实施。

### 必改（实施前生效）
- **A1. M1 做 thin wrapper，不重写 AnyTop 预处理**。FK reorder / mean-std normalize / graph fields / mask / 13ch / rot6d / root traj 已由 `src/data/anytop_dataset.py`(L~879) 处理。M1 只做：读 manifest 控制样本子集与顺序，底层仍实例化 `AnyTopDataset` 做预处理。禁止复制预处理逻辑（会与训练 loader 分叉）。见改后 §5-M1。
- **A2. R-precision / InfoNCE 必须 duplicate-aware / multi-positive**。cap3/cap4 高度重复（实测 val cap3 仅 1806 unique caption / 4112 motion，最高频一句出现 56 次）。SALAD 默认对角线唯一正解（`outside_docs/SALAD/utils/metrics.py:58`）会把语义正确的同 caption motion 误判为负样本。**正解判定与 InfoNCE 负样本 mask 都要按 (same motion_id) ∪ (same source_motion_id) ∪ (same caption_text of current view) 分组**。见改后 §5-M2。

### 4 决策点最终定论
1. **canonical key = A**（`source_motion_id` 主键，标注「物种+性别+动作 组合级 holdout」）。用户实算：剥性别后 clean 仅 40 条、纯动作模板 clean = 0 → **不做 action-stem 主评估**（数据不支持）。`val_action_clean`(824) 作主诊断，明确标注语义边界（非动作模板级 unseen）。
2. **caption views（方案 A，见 §3.2）**：`full=cap0`；`species_stripped` = per-motion 首个匹配 regex `\b(an|the) animal\b` 的 caption（无的标 `has_species_stripped=False` 并排除，覆盖 ~70.5%），直接用已缓存 T5。`category_level`(cap4) / `action_only` = optional future，v1 不做。
3. **指标主次**：**R-precision / matching score 为主裁，FID 仅参考**（多拓扑多物种混分布下 FID 解释性弱；保留但不作唯一裁决）。
4. **evaluator 规模（定）**：`d_model=384, n_heads=8, d_ff=1536, n_graph_layers=4, n_temporal_layers=2`（含 text proj ~14.1M）。d256(6.5M)偏弱、d512(22M+)易记忆，384 折中。

### 补充：序列 / 关节上限
- L2 实测 **T_max=237, J_max=140** → evaluator 用 **`num_frames=260, max_joints=144`** 全覆盖，**不裁动作**（评估须看完整动作语义，不能像 VAE 用 64 帧裁尾）。

---

## 1c. AniMo (CVPR 2025) 对齐修订（2026-06-04）

参考 AniMo 论文 + 代码（github `WandererXX/AniMo`）。实读 `models/t2m_eval_wrapper.py` + `utils/eval_t2m.py` 印证方向：

**AniMo 代码印证（实读确认）:**
- `get_co_embeddings(word_embs, pos_ohot, cap_lens, motions, m_lens)` —— **只吃 caption 特征 + motion，不吃 species/object_type/gender**。✓ 支持「evaluator 不把物种名当答案捷径」。
- 三件套 `MovementConvEncoder + TextEncoderBiGRUCo + MotionEncoderBiGRUCo`，**固定 `dim_pose=359`**（统一 30 joints / T2M format，movement 处理 359−4=355）。✗ 我们 variable-J + graph，**不能照抄**。
- R-precision（`calculate_R_precision`）+ matching（`euclidean_distance_matrix(et,em).trace()`）都是 **diagonal / single-positive**（每 query 唯一正解=配对那条）。✗ 我们 cap3/cap4 高重复（一句最多 56×），diagonal 会把同义正样本误判负样本 → **必须 multi-positive**。
- FID = motion embedding 上 `calculate_activation_statistics` + `calculate_frechet_distance`。✓ 保留作参考。
- AniMo 未公开 evaluator 训练脚本（仅 `text_mot_match/finest.tar` 下载）→ 训练目标我们自定（对称 InfoNCE）。

**与 T2M/AniMo 的有意偏离（记录）:** AniMo text encoder 是 word-level `TextEncoderBiGRUCo`（word_embs+pos_ohot）；**我们用已有 T5 mean-pooled `[768]` → proj 384**（sentence-level，更强且已缓存，省 word-token 管线；与 motion-emb 同维 `coemb_dim=384`）。

### Hard requirements（v1 必须）
1. 独立 **frozen** evaluator，**不共享** VAE/denoiser 权重。
2. evaluator **只用真实训练 motion** 训练，**不用 generated samples**。
3. **不用 VAE latent z** 作 paper metric embedding。
4. motion encoder = 自己的 **graph-aware SkeletonEncoder**（`d384/h8/dff1536/4graph/2temporal`, ~14.1M），输入 AnyTop 13ch + `frame_mask`/`joint_mask` + `adjacency`/`geodesic_dist`/`anytop_joint_relations`；**不照抄 AniMo 359-dim GRU**。
5. text encoder = 已有 **T5 caption embedding `[768]` → proj 384**（与 motion-emb 同维，**共享 `coemb_dim=384`**，无需 motion_proj）。
6. **full caption(cap0) 主评估**；species_stripped 仅防物种捷径 sanity view。
7. **species / object_type / source_motion_id 只作 metadata / 分组报告 / mask，绝不喂 text encoder**。
8. **multi-positive**（R-precision / matching / InfoNCE 一致）：`(same motion_id) ∪ (same source_motion_id) ∪ (same caption_text of view)` 都**不算普通负样本**。
9. **FID 保留但仅参考**；主裁 = R-precision / matching + visual QA。
10. M0 manifest 用现有 `splits/{train,val}.txt`，**不额外扣数据**，派生 train_main / val_all / val_action_clean / val_action_overlap。
11. species_stripped = **方案 A**：per-motion 从 cap0–cap4 选**第一条匹配 regex `\b(an|the) animal\b`** 的 caption；无的标 `has_species_stripped=False` 并从该 sanity eval 排除（val 覆盖 ~70.5%，报告覆盖率，**不重写文本、不重算 T5**）。

### Optional future ablation（v1 不做，记录备查）
- `action_only` view（去主语纯动作短语）—— 需 NLP 改写引入噪声。
- `category_level` view（"a quadruped/bird"，cap 位置不固定）—— 次要。
- `action_stem` clean（剥性别/物种的动作模板 holdout）—— 实测剥性别 clean 仅 40、纯动作 0 条，**数据不支持**。
- unseen-topology benchmark —— 需扣物种，与无备用 motion 冲突。
- species_stripped **全覆盖**（方案 B/C：NLP 补全 + 重算 T5）。
- multimodality metric（需 per-text 多次生成）。

---

## 2. 代码验证结论（你计划的假设逐条核实，全部用代码/真实数据确认）

| 计划假设 | 验证 | 证据 |
|---|---|---|
| split 文件已就位，77882/4112，0 overlap/dup/missing/uncovered | ✅ | `splits/{train,val}.txt`，loader 实测读取 |
| caption JSON 有 `source_motion_id` / `source_file` 字段 | ✅ 100% 覆盖 | `motion_texts_by_file_with_codex_drafts.json`，每条都有 |
| `source_motion_id` 可作 canonical_action_key | ✅ 但含 gender（见 §3.1） | 格式 `aardvark_female@runbase` |
| **824 clean / 3288 overlap / 3987 keys / 79.96%** 可复现 | ✅ **逐数字吻合** | agent 用 train.txt+val.txt+caption JSON 实算,完全一致 |
| T5 cache 100% 覆盖，768 维 | ✅ | 409970 keys(5 cap × 81994 motion),float32,`.embs.npy`+`.keys.json`(+`.npz`) |
| `batch.anytop_x` 是 `[B,J,13,T]`，permute(0,3,1,2)→`[B,T,J,13]` | ✅ 核实正确 | `anytop_dataset.py` collate + `vae.py:384` 同款 permute |
| VAE decode 出 `pred_motion [B,T,J,13]` + `frame_mask_recovered [B,T]` | ✅ | `vae.py:712/718`；anytop13 路径 |
| temporal_stride=4 的尾窗 mask 风险（别用原 frame_mask） | ✅ 你的警告对 | `frame_mask_recovered` = 保守 AND + repeat_interleave，须用它 |
| batch 含 evaluator 需要的全部字段 | ✅ | motion + graph(adjacency/geodesic/joint_relations/skeleton_features) + `frame_mask`/`joint_mask` + `caption_emb[B,768]` + `object_type`/`motion_id` |
| 已有可复用的 graph-temporal motion encoder | ✅ `SkeletonEncoder` | `src/models/encoder.py:252`，吃 `[B,T,J,13]`+graph，anytop13_split + graphormer |
| 已有 evaluator / FID / R-precision 代码 | ❌ **无本地实现** | 只有 `outside_docs/SALAD/`（固定拓扑，不适用 AnyTop）；要新写/移植 |
| `losses.py` 有现成 InfoNCE | ❌ 没有 | 需新写（标准对称 InfoNCE，~30 行） |

---

## 3. 两个建议你先拍板的设计点

### 3.1 ⚠ canonical_action_key 含 gender → `val_action_clean` 的真实语义比字面弱

`source_motion_id` = `物种_性别@动作`（如 `aardvark_female@runbase`）。所以:

- `val_action_clean`(824) 的真实含义是 **「物种+性别+动作 这个精确组合」未在 train 出现**；
- **不是**「动作模板未见」。例如 val 里 `aardvark_female@runbase` 算 clean(它的 key 不在 train),但 train 里可能有 `aardvark_male@runbase` —— **同一动作模板 `runbase` 通过另一性别泄漏了**。

后果:evaluator 在 `val_action_clean` 上即使拿高分,也可能部分来自「动作模板记忆」,而非真正的「未见动作泛化」。这与你计划 §2 的初衷（防 male/female 近重复泄漏）有张力 —— 用 source_motion_id 做 key **挡不住** male/female 泄漏。

**我的建议（请选）:**
- **(A) 维持现状**:v1 就用 source_motion_id（824 clean,已验证可复现），但在所有报告里**明确标注**「组合级 holdout,非动作模板级」,不宣称 unseen-action。← 最省、和你「v1 不宣称 unseen benchmark」一致，**推荐**。
- **(B) 加一个更严格诊断子集**:额外派生 `action_stem`（从 source_motion_id 剥掉性别 → `aardvark@runbase`，或进一步剥物种 → 纯动作词 `runbase`），算 `val_action_stem_clean`。更接近「未见动作模板」，但子集会更小（可能 < 824，需实测是否够大到可靠）。作为**附加诊断**,不替代 val_action_clean。

> 注:无论 A/B,InfoNCE 的 false-negative mask 也要相应定:用 source_motion_id(含gender)做「同动作不算负样本」会比用 action_stem 更宽松。这关系到 evaluator 训练，建议和上面同一选择保持一致。

### 3.2 caption views — 方案 A（per-motion 内容选择；最终定，见 §1c-11）

⚠ **纠正**：先前以为「cap3 恒为 species_stripped」是**错的**。实测（全 81994）："an animal" caption 出现在 **cap3 占 59% / cap4 占 12% / 完全没有 28%**（如 `addax` 5 句全含物种）。所以**不能用固定 cap index**。

**方案 A（hard，零 NLP、零重算 T5）:**
- `full` = **cap0**（主评估）。
- `species_stripped` = **per-motion 扫 cap0→cap4，取第一条匹配 regex `\b(an|the) animal\b` 的 caption** 及其已缓存 T5 embedding；标 `has_species_stripped=True` + 记录命中的 cap index。
- 无 animal-level caption 的 motion（val 约 29.5%）→ `has_species_stripped=False`，**从 species_stripped sanity eval 排除**，report 标注覆盖率（~70.5%）。
- `category_level`（"a quadruped/bird"）、`action_only`（去主语）→ **optional future**（§1c），v1 不做。

T5 cache 已对全部 5 caption 算好 embedding（`<motion_id>__cap{0..4}`），所以方案 A 只是「按内容选对的那个 cap 后缀」，**不重写文本、不重算 T5**。

---

## 4. 复用 vs 新写（最小化新代码，Karpathy「能复用就别造」）

**复用（已存在、验证可用）:**
- `src/models/encoder.py::SkeletonEncoder` — evaluator 的 motion encoder 基础（`[B,T,J,13]`+graph → `[B,T,J,D]`），加一层 masked 时间池化 → `[B,D]`
- `src/models/graph_salad/batch.py::GraphMotionBatch` + `graph_utils.py` — 拓扑/mask 对齐
- `AnyTopDataset`（已支持 split 文件）+ T5 cache（5 级 caption embedding）
- `outside_docs/SALAD/utils/metrics.py` — R-precision/FID/diversity/multimodality/matching_score 的**参考公式**（移植到 `src/metrics/`，按 AnyTop embedding 适配）

**新写（无现成）:**
- `scripts/build_anytop_t2m_eval_splits.py`（M0）— manifest 生成器 + preflight
- `src/data/anytop_t2m_eval_dataset.py`（M1）— 读 eval manifest，吐 graph+mask+caption(多 view)+canonical key
- `src/models/graph_salad/t2m_evaluator.py`（M2）— text proj(768→384) + SkeletonEncoder+masked 时间池化(→384，`coemb_dim=384`) + 对称 multi-positive InfoNCE
- `scripts/train_anytop_t2m_evaluator.py`（M2）
- `src/metrics/anytop_t2m_metrics.py`（M3 用，移植 SALAD）
- `scripts/eval_anytop_{vae,denoiser}_t2m.py`（M3/M4）

---

## 5. 细化的 M0–M2 可执行步骤（你计划要求 Stop after M0–M2）

### M0 — manifest 生成器 + preflight（纯数据，零模型；纯 json，不 import torch）
产出 `data/anytop_planet_zoo_clean_L2/eval_splits/{train_main,val_all,val_action_clean,val_action_overlap,split_audit}.json`
- 读 `splits/{train,val}.txt`（source of truth，不重分）。
- **每 motion 一条 manifest 记录**：
  - `filename`（带 `.npy`，= split / caption JSON 的 key）+ `motion_id`（**stem，无 `.npy`**，= AnyTopDataset 返回的 motion_id）
  - `source_motion_id`（= canonical_action_key，来自 caption JSON，**仅作 metadata / 分组 / mask，不喂 text encoder**，§1c-7）
  - `captions`（cap0–cap4 五条原文）+ `t5_keys`（`{motion_id}__cap{0..4}`，即 stem 拼 cap 后缀）
  - `species_stripped_cap_idx`（方案 A：首个匹配 regex `\b(an|the) animal\b` 的 cap index）+ `has_species_stripped`（bool）
  - （`object_type` 不存 manifest → M1 运行时从底层 `AnyTopDataset` 拿，避免分叉，§1c-A1）
- 派生 `val_action_clean`(824) / `val_action_overlap`(3288)（按 source_motion_id 是否在 train 出现）。
- **preflight 全 hard-fail**：caption 覆盖 100% / 5 个 t5_key 全在 cache / 无 dup / train∩val=0 / train+val=81994 / 824+3288=4112 / 与本文审计数一致；**并报告 `has_species_stripped` 覆盖率（预期 val ~70.5%）**。
- **验收**：json 生成 + preflight 全绿 + 数字 == 824/3288/4112 + species_stripped 覆盖率打印（codex 审）。

### M1 — evaluator dataset adapter（thin wrapper，必改 A1）
产出 `src/data/anytop_t2m_eval_dataset.py`
- **底层复用 `AnyTopDataset`**（`split="train"/"val"`, `num_frames=260, max_joints=144`）做全部预处理；wrapper 只做：① 按 M0 manifest 的 motion_id 选子集 + 定顺序（train_main / val_all / val_action_clean / val_action_overlap）；② 附加当前 caption view（`full`=cap0 / `species_stripped`=manifest 的 animal-level cap）的 T5 embedding + canonical_action_key + caption_text。**绝不复制预处理逻辑**。
- 返回：AnyTopDataset 原样字段（`anytop_x`/graph/`frame_mask`/`joint_mask`）+ view 的 caption_emb / caption_text + canonical_action_key + motion_id。
- **验收**：4 个 manifest 都能 load；wrapper 的 `anytop_x`/graph 与直接调 `AnyTopDataset` 同 motion **逐元素一致**（证明无分叉）；canonical key 抽查正确（+ codex 审）。

### M2 — evaluator 模型 + tiny overfit + sanity
产出 `t2m_evaluator.py` + `train_anytop_t2m_evaluator.py`
- text: 768→**384** MLP；motion: SkeletonEncoder(`d384/h8/dff1536/4graph/2temporal`, ~14.1M) + masked 时间池化 → **384**（SkeletonEncoder 池化天然 384 → **共享 `coemb_dim=384`，无需 motion_proj**）
- 对称 batch-wide InfoNCE，**duplicate-aware false-negative mask（必改 A2）= (同 motion_id) ∪ (同 source_motion_id) ∪ (同 caption_text of current view)** —— 三者任一都不算负样本，否则 cap3/cap4 高重复会把同义正样本当负样本压低对齐。
- **R-precision 用 multi-positive / group-aware（必改 A2）**：给定 text query，top-k 命中**任一**与 query 同组（同 motion_id / source_motion_id / caption_text）的 motion 即算正确，而非只认对角线那条；否则高重复 caption 下 R-precision 被系统性低估。
- **matching score 也 multi-positive（必改 A2）**：`matching_score = mean over text queries of min L2 distance to ANY positive motion in the same group`（同 motion_id / source_motion_id / caption_text）；**不再用 diagonal `trace()`**（diagonal 在高重复下会把更近的同义正样本当错配惩罚）。
- **验收 gate（4 条,你计划 M2 要求）**:
  1. tiny 子集 overfit（train retrieval→~100%）
  2. held-out val（group-aware）R-precision **显著高于随机**（≈ group 大小 / N baseline）
  3. **打乱 caption** retrieval **≈ 随机**（防止它学了 trivial shortcut）
  4. species_stripped(animal-level subset) retrieval **仍 meaningful**（若 full 高但它崩 → evaluator 在靠物种名作弊，不可信）
- **跑完报你确切命令 + 4 条 gate 的数值**，再决定是否进 M3/M4。

> M3（VAE recon eval）/ M4（denoiser generation eval）按你计划在 M2 通过后再做，且每个 metric 报告都配 GT-vs-pred 可视化 gif（你的 CV 铁律）。本方案先不展开 M3/M4 细节，M2 通过后单独出。

---

## 6. 决策点 — 全部已定（2026-06-04，见 §1b / §1c）

1. **canonical key = A**：`source_motion_id` 主键，标注「物种+性别+动作 组合级 holdout」。action_stem clean 数据不支持（剥性别 40 / 纯动作 0），不做。
2. **caption views = 方案 A**（§3.2）：full=cap0；species_stripped=per-motion 首个 animal-level caption（覆盖 ~70.5%，无的排除）；category_level / action_only = optional future。
3. **指标主次**：R-precision / matching 为主裁，**FID 仅参考**（evaluator motion-emb 分布上算，多拓扑混分布解释性弱）。
4. **evaluator 规模**：`d_model=384, n_heads=8, d_ff=1536, n_graph_layers=4, n_temporal_layers=2`（~14.1M）。
5. **AniMo 对齐**（§1c）：独立 frozen evaluator / 不喂 species 给 text encoder / multi-positive R-precision+matching+InfoNCE / motion encoder 用 graph-aware SkeletonEncoder（非 359-dim GRU）/ text 用 T5 proj。

---

## 7. 执行纪律（实现阶段）

- **不碰 running 训练**:evaluator 是独立栈，不改 VAE/diffusion 的 split/代码/ckpt。两训（diffusion ep69 / bf16 VAE ep100）继续。
- **每个 milestone 过 codex（gpt-5.5 xhigh）审** + 关键产物 hard-fail preflight。
- **建议实现走新 session**:本 session 已较长（监控 + split 适配 3 轮 codex + commit）。审核通过后,M0–M2 的实现是个大 arc，建议在**新 session** 用本方案文档继续，避免 context 膨胀。
- **可视化优先**:M3/M4 每个 metric 报告必配 GT-vs-pred 多帧 gif。

---

## 附:本方案的代码验证由 3 个并行 Explore agent 完成
- caption/T5/action-key 复现（824/3288 逐数字吻合）
- batch 字段 + VAE/denoiser decode 形状核实（permute 正确）
- 已有 evaluator/metrics/encoder 盘点（无本地 evaluator，SkeletonEncoder 可复用，InfoNCE 需新写）
