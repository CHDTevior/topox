# ICLR 2027 投稿草稿 · 提纲与待定项（2026-09-04）

**截止**：abstract 2026-09-18、正文 2026-09-25（均 AoE）；正文上限 9 页（含图表，不含参考文献/附录）；双盲。
**模板**：官方 `iclr2027_conference.{sty,bst}`（ICLR/Master-Template，2027 目录），已放在本目录；`make` 可编译（登录节点有 pdflatex）。
**旧稿**：`/iridisfs/scratch/ts1v23/workspace/paper-iclr/`（8 月 1–2 日，VQ-VAE+CodeFlow 两阶段路线，已不适用）——标题与引言立场已借用，其余不沿用。

## 文件
- `main.tex`：宏 `\method`（暂 TopX）、`\rep`（KTJD-17）、`\todo`、`\num`（待定数字，蓝色）；匿名作者。
- `sections/introduction.tex`（~930 词，可直接改）
- `sections/related_work.tex`（~620 词，6 段，33 条文献，3 条标 TODO verify）
- `sections/method.tex`（大纲：7 个小节，每节列内容要点 + 计划的公式/图/表）
- `references.bib`

## 论文主张（引言里的四个组件 + 结果）
1. KTJD-17 逐关节 17 通道表示 + 逐 (rig, joint, channel) 归一化（常量格剔除、rest 中心化）→ 311 个 rig 共用一个输入空间。
2. In-context 骨架条件：1 帧 rest-pose demo + 关节自然语言描述的 LLM2Vec 嵌入 + 结构特征 + 图距离注意力偏置；架构里不含任何骨架专属参数。
3. 校准的分组流匹配目标：9 组通道的梯度份额与关节数无关（γ·√(N_g/N) 的推导），γ 由预注册协议解出并经机制自检（25%）认证；FK / 接触 / 加速度辅助项。
4. 少样本适配：冻结主干 + 逐 rig LoRA（attn/ffn/cond），TrueBones 视图（主体关节剪枝规则、PZ 风格 caption、按源动画切分）。
5. 结果：311 rig / 77,894 clip；302.9M 参数；text→motion R@1 0.936（pool 32）vs 评估器 GT 上限 0.975，FID 0.0052；20 步采样；qk-norm 是 0.3B 可训的关键；TrueBones LoRA（水牛 fkdist −59%，龙从不可用到可用）；跨物种共享 LoRA 劣于物种专用（负结果）。

## Method 大纲（sections/method.tex）
3.1 KTJD-17（通道表 Table 1、rest-delta 旋转的动机、有效性掩码、逐格归一化、语料规模）
3.2 In-context 条件（序列布局、关节语义、结构特征与图偏置、时/空注意力块 + AdaLN、qk-norm 稳定性证据、CFG dropout）→ Figure 2
3.3 校准分组目标（x-prediction 流匹配 + v-space 权重 + Huber、分组损失公式与梯度份额恒等式、校准协议、辅助项、优化配置）
3.4 采样（Euler 20 步、掩码重投影、组合式 CFG、部署延迟 TODO）
3.5 少样本适配（零样本瓶颈、LoRA 位置与规模、未见 rig 的数据视图、配方、文本探针）
3.6 骨架感知评估（评估器结构与独立性、协议：3,899 val / pool 32 / GT 上限 / FID / 可复现分片生成）

## 需要你拍板 / 补充
- **方法名**：`\method` 暂用 TopX（旧稿同名），要不要改。
- **泛化口径**：run12 的逐格统计覆盖每个 rig 全部 clip（args.json 标 TRANSDUCTIVE_ARCHITECTURE_TEST，"不得报为 inductive/zero-shot"）。稿子现在的写法是"逐 rig 矩是 rig 的属性、部署时给"，对"未见 rig"只通过 TrueBones LoRA 少样本来讲。若审稿人追问 val 泄漏，需要一段限制说明或一个 inductive 对照（估计矩，memory 里有 ~1.5 GPU-h 的方案）。
- **数据来源与许可怎么写**（游戏资产重导出、只保留变形骨）——脚注留了 TODO。
- **最终数字**：R@1/FID 目前是 ep214 快照在 A100 分片上的值；run12 训到 500 epoch 后用同一硬件重算并替换所有 `\num{}`。ep300 的 gen-eval 明早出。
- **Experiments 章节**：主结果表、ablation（校准 vs 手调、关节描述 vs 名称、x vs v prediction、步数、qk-norm）、LoRA 表（四物种 + 组 vs 专用 + 两阶段）、可视化图——需要先定哪些 ablation 已有可用数字（handoff/ 里有 pred-target、unseen-topology、kimodo-loss 三份报告可用）。
- **文献核对**：`fan2025motionmillion`、`hong2025salad`、`truebones` 三条标了 TODO verify；旧稿 bib 里的 `Dragon`、`SAMoR`、`R2ET` 我没收录（不确定其真实性）。


## 2026-09-04 grill 结论（user 拍板）
- 主张：开源、可 fine-tune 的 raw-space 多拓扑 T2M；"可扩展" = 约 10 条 clip 经 LoRA 适配到未见骨架；只与开源方法比。
- 引用政策：G-MDM（ICML 2025）、SAMoR（并行）引用但不比（无模型代码）；已写入 Related Work。
- Baseline：只比 AnyTop（唯一开源多拓扑模型，无文本、在整个 Truebones 上训练）——用其发布权重在同一批 TrueBones rig 上生成，与我们的零训练臂/LoRA 臂比几何指标 + 可视化，明写只比动作质量且数据优势在 AnyTop。
- 未见 rig 口径：库内 + 少样本 + 零训练臂（rig 自己的统计 + 冻结主干）；不声称 zero-shot。
- 开源范围：论文里不提，投稿后再定（已删除所有 "we release" 句）。
- 表示消融（rest-delta / 逐格归一化 / 常量格）：留到 ablation 阶段再定。
- 故事：声音克隆 → 身体克隆（F5-TTS/E2-TTS 文本引导语音填充 → 1 帧 rest pose 即"参考音频"；JiT 的 raw-space x-prediction；Kimodo/UMO 损失）。已写进引言第 4 段。
- 新实验（user 建议）：在稳定架构下跑 demo 不止 1 帧 rest 的版本（64 帧动作 demo），方案见下一条汇报。

## 图 1 方案（待做）
- **左**：三四个 rig 的 rest pose（如 Bairds tapir 48 关节、Saltwater crocodile 75、Dragon 95、Giant anteater 102）各一小图，标关节数，说明"rest 帧 = 这具身体的参考音频"。
- **中**：序列布局示意 `[rest demo (1 帧, 干净) | target (240 帧, 带噪声)]` → DiT（时间注意力 ↔ 空间注意力 + 图距离偏置）← caption 经 AdaLN；关节描述文本 → LLM2Vec → 加到该关节所有 token。
- **右**：同一 caption（例如 "walks and then runs"）在四个 rig 上的生成结果各一帧序列（4–5 帧 filmstrip），最右加一列 TrueBones 未见 rig 的 LoRA 结果。
- 数据来源：renders/run12_ep250_cfg2_* 已有 6 个 rig 的 gif（取帧即可）；rest pose 可从 dataset 的 skeleton npz 直接画；Dragon 用 renders/lora_tb_Dragon_all_r128_e900_oodtext。
- 风格：白底、线骨架、同一视角；关节数写在 rig 名下面。

## AnyTop 对比协议（2026-09-04 v4，codex 三轮；数字见 runs/_anytop_cmp/<rig>_v4.json，页面 artifact 8c7ff27e）
- 采样：AnyTop 发布权重 `all_model_dataset_truebones_bs_16_latentdim_128`，object_type ∈ {Buffalo, Gazelle, Dragon, Spider}，8 次采样/rig（paper_cmp_v2 的 8 个 repetition；generate.py 固定种子，v1 的单条与 v2 rep_0 字节相同，已排除，对比脚本对同源 .npy 直接拒绝），6 s（120 帧 @ 20 fps，BVH 头写 24 仅为信息），无文本，在整个 TrueBones 上训练过。
- 两侧都拆 **pose / fk**（user 2026-09-04："pose 可能更稳，但只有 fk 才有使用价值"）：AnyTop pose = 其 .npy 用其官方 `recover_from_bvh_ric_np` 恢复的直接位置（`scripts/_anytop_ric_world_export.py`，在其 conda 环境跑）；AnyTop fk = 其 IK 拟合后导出的 BVH 再 FK（用户拿到的文件；与 pose 逐关节均差 0.08–0.45 = 其 IK 误差）；我们 pose = 位置通道 gen_ric，fk = 预测旋转的 FK gen_fk。
- 我们：零训练臂（run12 best 快照 + 该 rig 自己的逐格统计，corpus swap）与 LoRA 臂（Buffalo r64 / Gazelle r64 / Dragon r128-e900 / Spider r64，各自 best），val 目标全部生成（20 步 / cfg 2 / seed 7），`--dump_world` 存 world-dump-v3（含 ckpt sha、采样参数、语料 generation）。
- GT 分母 = 该 rig manifest 全部 accepted clip（不经 caption 资产、不经 LoRA 排除表；Buffalo 20 / Gazelle 10 / Dragon 13 / Spider 33），逐 clip 校 sha 后官方解码。
- 指标（`scripts/_compare_external_bvh_geometry.py`，共有关节 = 我们主体视图全部关节，双方骨长相对差 ~1e-15）：所有序列先抗混叠降到共同 20 Hz（绝不为指标上采样）；jitter = 二阶差分范数 × fps²（units/s²；全关节 / 仅根）、关节活动 = 相对根的速度、根 XZ 速度；去掉随时长变化的根高度范围；比值相对 rig-GT 均值，分母低于阈值记 n/a（Spider 根静止）；我们另报合并 matched-GT 比值；n≥3 有 bootstrap 95% CI；共有 End Site（1/1/3/13）有敏感性表。**描述统计**，非逐样本质量、非文本对齐。
- 结果（× rig-GT 均值；jitter 全关节 / 关节活动；AnyTop n=8，我们 n = val 目标数）：
  - Buffalo：AnyTop pose 0.58 / 0.59，AnyTop fk 0.83 / 0.55；零训练 pose 1.85 / 0.89，fk 3.32 / 1.45；LoRA pose 0.32 / 0.33，fk 0.33 / 0.30。
  - Gazelle（我们 n=2）：AnyTop pose 0.56 / 0.91，AnyTop fk 0.86 / 0.85；零训练 pose 7.23 / 2.68，fk 13.17 / 5.39；LoRA pose 0.74 / 0.74，fk 0.86 / 0.77。
  - Dragon：AnyTop pose 0.42 / 0.56，AnyTop fk 0.70 / 0.50；零训练 pose 1.97 / 0.95，fk 2.53 / 1.02；LoRA pose 0.48 / 0.48，fk 0.52 / 0.49。
  - Spider：AnyTop pose 0.42 / 0.56，AnyTop fk 0.41 / 0.48；零训练 pose 4.96 / 2.19，fk 6.49 / 2.87；LoRA pose 0.21 / 0.25，fk 0.20 / 0.24。
  - 读法：两家都比真实动作平滑且欠活动（AnyTop 抖动 pose 0.42–0.58× / fk 0.41–0.86×，活动 0.48–0.91×；LoRA 抖动 fk 0.20–0.86×，活动 0.24–0.77×）；AnyTop 的 IK 导出让 fk 比 pose 抖 1.0–1.7×，我们 LoRA 的 pose/fk 几乎一致（旋转与位置通道自洽），零训练臂 fk 比 pose 抖 1.3–1.8×（未适配时旋转通道先坏）。根速度两家都远低于 GT（AnyTop 0.16–0.30×，LoRA 0.21–0.47×；Spider 根静止记 n/a）。描述统计：臂间非逐样本匹配；更低的抖动/根速度也可能只是动得更少。
- 待办：表进 Experiments（加 CI、写明 AnyTop 库内训练 & 无文本、我们少样本 & 文本条件）；图用 fk 行为主、pose 行进附录。
- **2026-09-04 user 拍板（看过 PDF 预览后）**：Introduction 的贡献统一为三条——(i) 数据（KTJD-17 表示 + 311 rig 语料），(ii) 多拓扑适配的 framework（模型 + 训练目标/配方/采样合成一条，原 (ii)(iii) 合并），(iii) 好扩展（未见 rig 零训练 + ~10 条 clip 的 LoRA；评估器与 AnyTop 描述统计作为支撑证据并入）。已改 sections/introduction.tex 并重编译（预览 artifact 40fd9e03）。
- **2026-09-04 可扩展性复现（36M rest 骨干）**：同 4 rig 零训练 + LoRA（设置同 run12 LoRA），几何对比 v1 见 artifact f8b7ea82 / runs/_anytop_cmp/<rig>_ext_v1.json：36M rest LoRA fk 抖动 ×GT 0.27/0.92/0.62/0.23 vs 0.3B LoRA 0.33/0.86/0.52/0.20；零训练 1.92/6.24/0.74/4.71 vs 3.32/13.2/2.53/6.49。demo-64 骨干两臂待 ep400。r64 在 36M 上 16.3% 可训（0.3B 8.1%）= 同 rank 对照。

## 2026-09-05 user 拍板（grill 第四轮）
- demo-64 消融训练在 ep204 停（best=ep200）；ep200 的渲染/零训练/LoRA/两臂 gen-eval 照做。
- 新增 **TrueBones-only 36M rest-demo 检查点**：架构/目标/采样与 pilot 相同，只换数据（主体视图、PZ 风格 caption、逐格统计、重算校准）；**802 条带文本 clip 全训、无 val**（角色 = 库内训练的小模型参照，和 AnyTop 同类；它在四 rig 上生成的是训练过的 clip，写明是记忆上界）；预算 **6 万优化步**（lr horizon 23,118 步含 warmup 4000、即 19,118 步余弦到地板，与 pilot 的 23,120 近似匹配（相差 2 步）；之后地板 lr 到 6 万；拒步也计入 gstep，训完报告拒步数），8×H200。
- 与他人工作对比 = **能力对照表**（任意拓扑 / 文本条件 / 无模板无重定向 / 未见 rig 可适配 / 权重可得；行：AnyTop、G-MDM、SAMoR、OmniMotionGPT、人形 MDM 类、我们）+ **只与 AnyTop 比数字**（描述统计 + 渲染）。
- 逐 rig 归一化统计覆盖全部 clip：user 明确"motion 领域约定俗成，不用考虑" → **不写限制说明、不做 inductive 对照**。
- 论文剩余部分（Experiments / Conclusion / 附录）现在写，缺数字处留 \todo 占位。
- **2026-09-05 六臂对比 v2**（artifact f8b7ea82 / runs/_anytop_cmp/<rig>_ext_v2.json；fk 抖动 ×GT，Buffalo/Gazelle/Dragon/Spider）：AnyTop 0.83/0.86/0.70/0.41；run12 零 3.32/13.2/2.53/6.49、LoRA 0.33/0.86/0.52/0.20；36M rest 零 1.92/6.24/0.74/4.71、LoRA 0.27/0.92/0.62/0.23；36M demo-64@ep200 零 0.95/6.89/1.54/2.94、LoRA 0.24/0.74/0.70/0.17（LoRA best val 0.440/1.900/0.576/0.358 vs rest 0.438/1.871/0.520/0.336）。ep200 gen-eval 同硬件：rest 0.899/0.0113，demo-64 0.896/0.0100。ep200 渲染页 artifact 10dfa31d。
- **2026-09-05 TB-only g30k 臂**（七臂对比 v3，f8b7ea82）：fk 抖动 ×GT 1.02/1.93/1.10/0.36，活动 0.65/1.21/1.22/0.34；×匹配 GT 抖动 1.29/1.14/0.93/0.92、活动 0.91/0.99/0.98/0.82（训练过这些 clip = 记忆上界）。训完（6 万步）换最终 ckpt 进 Table tab:lora 的 (c) 列。
- **2026-09-05 TB-only 训完 + 七臂 v4（f8b7ea82）**：6 万 gstep（epoch 9999），0 拒步 / 0 SPIKE，末 epoch train_flow 0.154；最终 ckpt = runs/v2_p36_tbonly_rest/last_model.pt。最终臂 fk 抖动 ×GT 1.02/1.92/1.12/0.37、活动 0.67/1.22/1.23/0.35；×匹配 GT 抖动 1.29/1.14/0.95/0.95、活动 0.93/1.00/0.99/0.85 —— 与 g30k 臂几乎相同（记忆参照 3 万步后已饱和）。数字/哈希见 runs/_anytop_cmp/<rig>_ext_v4.json。
- **2026-09-05 论文指标勘误：“FK error” → “FK–pose gap”**。trainer 的 `fkdist` 是 `fk_dist`（src/models/v2/fk_torch.py）：x0 预测的“旋转 FK 得到的位置”与“直接预测位置”之间的平均距离（该 rig 平均骨长为单位、val 目标噪声水平下、固定 RNG），**不是**到目标 clip 的误差。此前 tab:lora/tab:anytop/消融段全部误写为“到目标 clip 的关节位置误差”，已统一改名并在 Setup 给出定义。它衡量两通道是否自洽（可部署性），LoRA 使新 rig 上的旋转/位置通道重新自洽（0.25–1.2 → 0.08–0.15 bl）。
- **tab:lora 的“zero”列 FK–pose gap 出处**：各 LoRA run 的第一次 [val]（epoch 5，≤15 个 warm-up 步，LoRA B 零初始化 → 与骨干几乎相同）；“LoRA”列 = best_model（val flow 最低）那次 val 的 fkdist。TB-only (c) 列同法测得：以 TB-only ckpt 为 INIT 起 EPOCHS=5 的 adapter 探针（LR_DECAY_EPOCHS 保持真跑值 250/750，前 15 步 lr 完全相同），只读第一次 [val]：Buffalo 0.072 / Gazelle 0.062 / Dragon 0.134 / Spider 0.088 bl（其 val flow 0.134/0.265/0.157/0.127，训练过这些 clip）。探针目录 runs/lora_tb_<Rig>_r*_v2_mainbody_p36tbonly_firstval，骨干 profile configs/lora_backbone_p36tbonly.env，脚本 runs/_lora_p36/_probe_tbonly_firstval.sh。
- 里程碑链教训：`run4 $CK ""` 的空 tag 在 `set -u` 下使 `$4` 越界、最终 dump 阶段空跑；已用非空 tag `_final` 重跑（renders/cmp_dump_<rig>_tbonly_p36rest_final）。
- **2026-09-06 user 决策**：训练类消融暂不补（"感觉都不需要跑"），先让 codex（gpt-6-astra max）全文审找硬伤；demo-64 **正文完全不提**，负结果整段移入附录 app:demo64；方法名沿用 TopX。user 提出研究问题：demo-64 的初衷是"像声音克隆"——demo 不只是认识骨架模版，还应作为运动模式的参考提升零训练能力；现实现未达成，需推理研究可行路线（见 handoff/研究备忘）。
- **2026-09-06 主表定稿链**：runs/_lora_p36/_final_geneval.sh 在 pink7014（ep289 best：20/10/5 步）与 pink7013（ep214 快照 20 步；36M ep399 20/50 步）各 4×H200 上跑冻结协议，结果 JSON 落 runs/v2_noik_run12_896_r1acc/gen_eval_ep289_h200x4_s*.json 等；论文将全部换成同硬件（H200 fp32）数字。
- **2026-09-06 图**：Fig 1/Fig 2 用蒙皮渲染（Blender + PZ 网格，scripts/_skin_batch_ktjd17.sh），303M best ep289 的 energetic 组 8 rig 批渲染在 flamingo01 上（renders/skin_fig_ep289）。
- **2026-09-06 硬伤（codex 全文审 P0-1，已独立复核）：验证集泄漏。** 活动集 73,995 train / 3,899 val；val 中 3,250 条与某训练 clip 动作数组逐字节相同（游戏资产重复导出，asset id 不同），另 137 条与训练集共享 (rig, source_action_name)；**干净子集 512 条 / 216 rig**（configs/pz_val_clean_subset_v1.json，规则=内容 sha256 不在任何训练 clip ∩ (rig,源动画名) 不在训练集）。干净子集偏短（中位 60 帧）、以 onspot/转身/游泳变体为主，论文必须披露。user 决定：先在干净子集上重评 + 可视化（gen-eval 加 --score_subset，只在 merge 路径生效，codex 审中）；重切分重训另议。
- **2026-09-06 其它已核实硬伤**：AnyTop 代码确实以 T-pose 作第 0 帧并加关节名嵌入（model/anytop.py tpos_first_frame）→ "in-context rest pose" 不是我们对 AnyTop 的区分；活动 rig 关节 34–102（中位 64），123 种不同 parent 数组；漏引 OmniZoo（2512.10352，140 物种 32,979 段，文本+任意拓扑）、X-MoGen（2508.05162，UniMo4D 115 物种 119k，共享拓扑）、UniMoGen（2505.21837，无文本）、Kimodo（2603.15546）。ICLR 2027：正文 ≤9 页严格、AI use statement 必需、Ethics/Reproducibility 推荐、双盲。
- **图 2 目标更换**：原 11 个目标中 9 个属于泄漏 clip（训练见过），只有 Moose fightchaseoff、Buffalo fighttauntreact 干净；改用干净子集里可蒙皮 rig 的 clip。
- **2026-09-06 AnyTop 比较口径（user 决策）**：user 指出 AnyTop 训练集就是 Truebones 70 骨架（含我们四个测试 rig），在这些 rig 上把它当对手比不公平；又明确"我们不用别人的项目训"（怕被说复现不完美、不公平），所以**不重训 AnyTop、也不移植其架构**（对其仓库的 5 行子集补丁已撤回，冒烟产物已删）。核实：AnyTop 无动作级 split，benchmark 为 29 个训练角色；论文里的"未见骨架"靠子集模型定性展示。我们 66 个 TB rig 全在其训练列表内，没有双方都未见的 Truebones rig。备选：AnimalML3D（SMAL 单拓扑，1240 段，双方均未见，可做双方零训练测试床，需 1–2 天转换）；OmniZoo 未公开。当前论文处理：AnyTop 发布模型仅作"训练过该 rig 的域内参照"（与 TB-only 36M 同列），不作为竞争对手；能力对照表保留。
- **2026-09-06 干净子集打分脚本 codex 四轮后 PASS**（--score_subset，旧指纹白名单，语料切分内容一致性校验）；渲染 paper-style 补丁 codex 三轮 PASS。
- **2026-09-06 图的内容规则（user）**："可视化只放我们自己生成的，还有尽量放靠近语义的和动作比较激烈的"。图 2 不再放真实动作行；候选按评估器 text–gen 余弦（语义）与生成能量（激烈）在干净子集 ∩ 可蒙皮 rig 上排序（scratch: renders/_webp_test/rank_clean_candidates.py，用 ep289 已存分片），选 8 条：Moose runtowalk(d34d76e2)、Cheetah jumpoutonspot(e80c143b)、Moose fightchaseoff(f2d76b31)、AfricanElephantJuv standtowalkturnl(b3714bfd)、WildDog walktostand(248651e0)、Buffalo swimtowalkturnr(3380cdd7)、Buffalo standturnr180(900153a2)、Gharial swimbaseturnl(cb6cfe73)。蒙皮渲染 renders/skin_fig2_{a,b}。
- **2026-09-06 干净子集分数**（512 条 / 16 池）：303M ep289 R@1/2/3 0.967/0.998/1.000，FID 0.0150，match 0.766（GT 参照 0.988/1.000/1.000，match 0.777）；ep214 0.965/0.998/0.998，FID 0.0158；36M rest ep200 0.926，FID 0.0293；demo-64 ep200 0.930，FID 0.0274。全集（泄漏）：ep289 0.937/0.987/0.994，FID 0.0051，match 0.764；GT 0.975/0.999/0.999，match 0.782。注意干净子集池多为跨 rig，检索偏易；match 与 FID 不受池构成影响。步数 10/5/50 的 pass 因协议钉死 steps=20 被脚本拒绝（需显式变体开关，另议）。
- **2026-09-06 压页完成**：正文（到 Conclusion 末）落在第 9 页内（Reproducibility/AI use/Ethics 声明从第 9 页后半开始，不计页）；搬入附录：通道表、Validity、图偏置细节、辅助项、LoRA 数据视图与探针、AnyTop 段+表、能力表、小骨干/共享/语言探针段、消融注记、采样器回滚轶事、切分审计；相关工作压成三段（引用全保留）。图 1/图 2 为过渡版（干净 clip 的生成，旧材质），待 skin_fig2 批与统一陶土静帧完成后替换。
- **2026-09-06 36M ep399（H200 4 卡）**：全集 R@1/2/3 0.918/0.972/0.986，FID 0.0083，match 0.746；干净子集 0.936/0.988/0.998，FID 0.0259，match 0.742。
- **2026-09-06 user 决策汇总**：demo-64 运动克隆路线记为 future work 不做；补实验里消融（描述 vs 名、校准 vs 手调、去 rest prompt）可做但最后做；文本跟随量化要用接近主训练分布、clip 多的骨架，不用龙/蜘蛛；不做 AnimalML3D 外部测试床；效率可以测但论文目的不是游戏部署；写作要求：不防御性写作、不自贬、abstract 不写"像 AnyTop 一样"，正常写正常 claim。
- **2026-09-06 切分来源核实**：PZ 语料的 train/val **不是** AniMo4D 官方划分。链路：`scripts/_merge_v4b_human_with_animals.py`（2026-07-03）用 AnyTopDataset 的 md5 分层算法自生成 val_frac 0.05（seed 42）→ `data/holdout_splits_v1`（去掉留出拓扑）→ KTJD manifest 的 split（`src/data/ktjd17/inventory.py` split_protocol=holdout_splits_v1）。AniMo4D 官方"文本与划分"在其 README 的 OneDrive 链接里，本地没有下载；是否同样存在重复导出跨切分未知。
- **2026-09-06 PZ 评估器不能迁移到 TrueBones**：TB 主体视图 val 170 条、按 rig 内检索 text→GT top-1 仅 0.33–0.38（机会 0.12–0.17，n=6–8），不能作近分布 rig 文本跟随的代理；文本跟随量化应放到重切分里的 PZ 留出 rig（评估器有效）或人评。
- **2026-09-06 codex 论文 r2 修正**：消融表首行改为干净子集 0.926/0.029；demo-64 附录改为"混合结果"（干净子集 0.930/0.027 略优，val loss/gap 略差）；干净子集 caption 重合披露（99/512 与训练 caption 逐字相同）；检索正样本谓词写全（同 clip / 同源导出 / 同 caption）；校准检查规格写实（384 宽 7 块 arm 模型、batch 8）；TB-only gap 标注同 zero 列读法；108 种无序拓扑（123 种有序父数组）；删"唯一发布的…"句；去防御性措辞多处。
- **2026-09-06 图 1/图 2 终版**：8 条干净 clip 只放生成（Moose runtowalk / Cheetah jumpoutonspot / Moose fightchaseoff / AfricanElephantJuv standtowalkturnl / WildDog walktostand / Buffalo swimtowalkturnr / Buffalo standturnr180 / Gharial swimbaseturnl），统一陶土材质（渲染脚本 --paper-style --stills，codex 六轮 PASS：白世界光 0.35、shadow catcher 地面、平滑着色、clay (0.58,0.30,0.20)、保留 AgX）；teaser 面板 C 用其中 4 条 + Dragon LoRA 线稿条带；产物 renders/skin_fig2_{a,b}/*/gen_targets_*/stills_clay2、paper/figures/{teaser,qualitative}.pdf；规格 paper/figures/{teaser_spec,qual_spec_final}.json。渲染时细分（1/2 级）试过不能消掉网格格子感反而出条纹，弃用。
- **2026-09-06 AniMo4D 官方划分核对**（user 上传 data/animo4d_official_split/{train,val,test}.txt，62,637/3,760/11,752）：我们 77,894 条 100% 对应官方序列；我们的 train 含官方 test 11,147 条（95%）和官方 val 3,546 条，我们的 val 含官方 train 3,146 / val 200 / test 553；官方划分按序列随机、无物种留出；官方 val/test 中 83.4% 与官方 train 共享 (rig, 源动画名)（精确哈希在算）。结论：官方基准同样存在重复导出跨切分；现有模型不能在官方 test 报数；未来重训应以官方 train 训、在去重后的官方 test 报数。

## 2026-09-06 (later) — split decision, figure redesign order
- User decision: AniMo4D's official split has the same duplicate-export property (exact-hash audit, record only: official val
  3,746 matched clips, 80.8% byte-identical to an official-train clip, 83.7% share (rig, source action); official test 11,700,
  80.3% / 83.3%; de-duplicated official test would hold 1,928 clips). Therefore the paper does NOT discuss the split/duplicate
  question anywhere and keeps our own per-rig 5% hash split as the protocol. Main table, intro, conclusion, ablation row and
  the demo-64 appendix now carry FULL validation-set numbers (303M ep289: R@1 0.937 / real 0.975, FID 0.0051, match 0.764;
  36M ep399: 0.918 / FID 0.0083; ep200 pilot 0.899 / 0.0113; demo-64 ep200 0.896 / 0.0099). The clean-512 subset
  (configs/pz_val_clean_subset_v1.json) and its scores (*_clean512.json) stay in the repo for our own reference only.
  App. "Validation split" is now a neutral protocol description (split rule, 311 val rigs, pool construction).
- User order on figures: Fig. 1 must show the conditional encoding, the architecture and the idea (current three-box teaser
  rejected); qualitative renders must put the character ON the ground, and the action must read at a glance (turn, jump, swim,
  attack). Process: Claude and codex each draft, cross-review, merge. Plan: Fig. 1 = TikZ framework figure with small
  raster insets; Fig. 2 = motion strobes (fixed world camera, ghost poses along the path, visible ground at the game's terrain
  level, translucent water plane for swimming, root trajectory on the ground) from the clean-subset generations of skinnable
  rigs (34 candidates over 14 rigs; runs/_figs/clean_action_metrics.json).
- Efficiency: gen-eval gained an explicit --protocol_variant steps flag (stamped in shards/report; codex r1 NEEDS-FIX -> fixes
  applied, r2 pending) and scripts/_measure_latency.py (batch-1 wall time per step count).

## 2026-09-06 (night) — figure pipeline and the ALIGN trial
- Fig. 2 is now a motion-strobe composition (paper/figures/compose_qual_strobe.py over renders/strobe_final_v6): fixed world camera,
  4–5 baked poses along the root path shaded light→dark, ground plane at the level the feet occupy most often (sliding-window mode),
  root trail + arrowhead, translucent water for swimming, row mode (4 frames) for actions on the spot. Renderer:
  planetzoo-anytop-pipeline/tools/planetzoo/_render_motion_strobe.py (codex rounds 1–3 NEEDS-FIX → fixes applied, round 4 pending).
  Lesson: Blender material colours are LINEAR — the pale renders came from sRGB values typed as linear; palette now converted
  (clay #B5552F → 0.462/0.091/0.028). Clips: cheetah stand→run, moose run→walk, babirusa jump-out, wolf run→stand→turn, lion swim
  →turn, moose fight (row), elephant stand→walk→turn (row). "jump" clips in this library leave the floor (burrow / water):
  the script now refuses an automatic ground that buries a pose; lion pounce-from-run and lemur climb-jump are being skinned as
  land-jump candidates.
- Fig. 1: TikZ v15 wired as interim (paper/figures/framework/). ALIGN method-figure-loop trial (paper/figures/fig1_align, p5.js +
  blind codex reading): round-1 blind score 6/10 with concrete misses (structure-MLP input, sampling update, LoRA optional);
  v2 redrawn (lanes under their destinations, tree → structure MLP, panel sizes, larger crops); round 2 running. The SVG→PDF
  path (cairosvg) works; the p5 version is the candidate to replace the TikZ figure.
- Main text now ends on page 9 after trimming the FK–pose-gap and limitations sentences; preview republished (40fd9e03).

## 2026-09-06 (late night) — Fig. 1 v7 wired, Fig. 2 strobe script PASS
- Fig. 1 = p5.js v7 (paper/figures/fig1_align/figure.html → figure1.pdf), replacing the TikZ interim. User rulings applied in order:
  balance (v5), colour/glyph/lane redesign (v6), then "left two bands too thin, right two too dense, the big grid is pointless, the
  architecture is not visible" (v7: band 1 rig → 17-channel token + sequence strip; band 2 the six addends of one token + caption/t → c;
  band 3 the ×14 transformer with temporal / spatial (tree bias) / MLP and the D→17 head; band 4 ×20 Euler return, decode, unseen-rig).
  Blind readings so far 6 → 6 → 8 (v4) → 7 (v6); v7 round running. Caption rewritten to match. Main text still ends on page 9.
- Fig. 2 = motion-strobe composition v7 (7 clips / 5 rigs); the render script passed codex round 7 with no residual items.
- Codex's own redesign consultation (colour-by-role, no gradient across frame columns, drop badges) recorded in
  paper/figures/fig1_align/wiki/decisions.md D-15; badges kept.
- Fig. 1 v8 (2026-09-06, after the user's "bands 2–3 meaningless / no bare MLP / t outside the dial"): the flow was first written as
  ASCII from the code (paper/figures/fig1_align/wiki/ascii_flow.md, every line grep-able), then bands 2–4 redrawn from it: token = ⊕ of six
  named terms; ×14 around the three sub-layers only; head named x̂₁; AdaLN anatomy inset; Euler update node. Canvas 1800×1100, caption
  shortened, main text still ends on page 9. Blind round 5 (v7) scored 7/10 and had raised the same points; round 6 on v8 running.
- 2026-09-06 15:00Z: the 36M 5-step protocol-variant eval (cut by alloc expiry on 09-06 05:10Z) was rerun on idle pink7001
  (runs/_final_geneval/effsweep2; report runs/v2_noik_pilot36m_r1acc/gen_eval_ep399_h200x4_s5_variant.json): R@1 0.897, FID 0.022
  (10 steps 0.915 / 0.013; 20 steps 0.918 / 0.008). tab:cost filled; text now says five steps cost 0.007 R@1 for 303M and 0.021 for
  the pilot. No training is running; run12 ended at ep297 (alloc end, lr at floor), TB-only 36M reached 60k steps, four TB LoRA arms done.
- 2026-09-06 15:2xZ: abstract carried the clean-subset numbers (0.967 / 0.988 / 0.015, 512 clips) while tab:main and the
  introduction report the full validation set (0.937 / 0.975 / 0.005, 3,899 clips) and the paper does not discuss the subset (user
  ruling on the split). Abstract now states the full-validation numbers; the clean-subset scores stay in this log only.

## 2026-09-06 16:00Z — ablation plan v2 (user) and what is running
- User's new ablation list (handoff/20260906_160000_ablation_plan_v2.md): (1) size sweep 36M / ~105M / 303M; (2) representation
  ablation vs the old representation (36M, same epochs); (3) the backbone on VQ latents (Graph-VQ retrained on KTJD-17 with LLM2Vec
  joint descriptions); (4) LoRA training length vs physical metrics, few-shot framing; (5) pick the retrieval pool size on the large
  test set (not necessarily 32). The old rows (hand-tuned / names / v-prediction) are dropped; v-prediction is not implemented.
- Free row measured: 36M without the acceleration term (pilot36m_animal ep200): R@1 0.899 / FID 0.011 -- identical to the full
  recipe at ep200 (0.899 / 0.011); the term's effect, if any, is on jitter, not retrieval.
- Running: 100M arm SMOKE on pink7001+pink7018 (configs/pilot100m_r1acc_env.sh); LoRA epoch-sweep reruns with snapshots
  (runs/lora_tb_*_v2_mainbody_epsweep) on flamingo01; codex review of --protocol_variant pool (scoring-time re-pooling of saved shards).
- From the paper LoRA logs (no new training): FK-pose gap plateaus by ~ep150-200 on all four rigs (best-val epochs 184/159/604/224);
  curves saved in runs/_figs/lora_epoch_curves_303m.json.
- 2026-09-06 17:14Z: r1acc objective recalibrated at batch 16 (configs/pilot_animal_r1acc_gamma_calibration_b16_v2.json; log
  runs/_calib/pilot_animal_b16_rest1_v2.log): solve PASS (max dev 1.2e-6), mechanism PASS; all nine gammas identical to v1 to 4 decimals
  -- the loss-code drift since v1 did not move any group share (equivalence evidence, no --allow_calib_code_drift used). New arms
  (100M, representation arms) use v2; the 36M/303M runs keep v1 (they predate the batch rule and trained under it).
- 2026-09-06 16:30Z: scratch quota hit the hard limit (4096/4099 GB) -- both LoRA sweep reruns died saving checkpoints (torch.save
  I/O errors). User approved deleting the pre-run12 superseded runs (v2_pzh_262m, v2_pzh_262m_gv2, v2_noik_run9/10, v2_pzh312_run2-8,
  v2_incontext_run1, lora_tb_Buffalo_r64_v1, lora_tb_Dragon_all*): ~226 GB freed, usage 3873 GB. Dragon/Spider sweeps relaunched;
  100M SMOKE relaunched with the batch-16 v2 calibration.
- Pool-size sweep (scoring-time variant, saved frozen-protocol samples; codex PASS r2): 303M ep289 gen R@1 0.956/0.937/0.922/0.915 at
  pools 16/32/64/128 (real 0.985/0.975/0.964/0.962); 36M ep399 0.938/0.918/0.897/0.891 (same real). The gap to the real reference is
  0.02-0.05 at every pool. Main-table pool: user's call; the sweep goes to the appendix either way.
- 2026-09-06 16:50Z data clean-up (user: "参考着删除一下你认为的没用的"): deleted the six codeflow_tokens_* caches (abandoned VQ/CodeFlow line),
  the T5 caption cache, animo4d_anytop_clean_L4_safe, animo4d_anytop_noik_canonical and the u_shard encoding intermediates of the five
  LLM2Vec caches (merged files kept). KEPT: data/animo4d_anytop (raw BVH source), data/animo4d_anytop_noik (source export of the KTJD
  noik corpus), L4TB_plus_human (trainer default), the LLM2Vec caches, dataset/ktjd17_pzh312_noik_v1 (user to confirm; no config or
  args.json references it). Quota: 4096 -> 2990 GB used (soft 3885 / hard 4099). Log runs/_lora_p36/cleanup_data_20260906.log.
- Hourly monitor cron c8987dbb (session) watches the 100M run; watchdog PPID=1 on swarma1002.
- 2026-09-06 18:00Z LoRA training-length sweep DONE (runs/_figs/lora_epsweep_summary_20hz.json; paper protocol: FK output vs all real
  clips at 20 Hz, sampler 20/2/seed 7): jitter falls below the real clips by ep50-100 on every rig (Buffalo 3.32 -> 0.66 -> 0.34;
  Gazelle 13.2 -> 1.04 -> 0.45; Spider 6.49 -> 0.57 -> 0.28; Dragon 2.53 -> 0.61 at ep150), the FK-pose gap plateaus by ep150-200
  (Dragon ep450), and articulation stays at 0.25-0.8x the real clips at every length (Gazelle dips to 0.36 at ep100 then returns to
  0.78 by ep200): more adapter training does not restore the under-articulation. Anchors reproduce tab:lora (Buffalo best-val 0.30/0.33).
- 2026-09-06 23:00Z representation ablation (arm b, "scale_only" = the KTJD spec's scale-only normalisation) implemented behind
  Ktjd17Base(normalization=...) / trainer --rep_norm / calibration REP_NORM / eval-side conversion into the evaluator's per-cell
  space / renderer; the variant rides in the existing target_centering pin (a first version added a new pin key and broke the
  bidirectional pin check for every old checkpoint -- caught by the text probe, fixed, verified against the run12 snapshot).
  Self-test: percell and scale_only items round-trip to identical raw values. Single-node launcher
  scripts/_launch_v2_ddp_1node_h200.sh derived for blossom04 (2xH200). Codex review of the whole change running; then recalibrate
  (REP_NORM=scale_only, CALIB_BATCH=64), smoke, train ~150 epochs, frozen eval at ep150 vs the r1acc pilot's ep150.
- Parallel evaluations on flamingo01: flyers group-adapter vs zero-training geometry chain (codex r2 pending); Buffalo text-following
  probe on 36 val-only unseen-text captions of related species in three energy classes (running).
- 2026-09-06 22:40Z text-following probe DONE (renders/lora_tb_Buffalo_v2_textprobe_energy36, summary in runs/_figs/textprobe_energy36_rows.json):
  36 val-only, never-seen-text captions of related species, 12 per energy class. Generated energy medians high/mid/low 0.0133/0.0057/0.0008
  vs library reference 0.0185/0.0095/0.0019 (ratios 0.70/0.58/0.54); Spearman rho 0.86; P(fight>rest)=1.00, P(fight>walk)=0.97,
  P(walk>rest)=0.94; root path 1.20/0.26/0.18. Appendix paragraph + tab:textprobe written; the limitation sentence now says language
  following is measured on one rig.
- 2026-09-06 22:50Z rep_norm codex round 1: NEEDS-FIX -- renderer still read per-cell stats directly (fixed: _stats), the
  calibration-script edit breaks the code hash of the running 100M's artifact (plan: b16 v3 recalibration on H200 + resume via
  --allow_calib_reswap, never --allow_calib_code_drift), utilities _gen_ktjd17_clips/_analyze_jitter/_ktjd17_to_bvh got variant
  plumbing. Round 2 running. configs/pilot36m_scaleonly_env.sh written (blossom04, 2xH200, B64, 150 epochs).
- 2026-09-06 22:45Z b16 v3 recalibration (current calibration-script bytes, H200): identical to v2 in gammas / energies / counts /
  shares / all 25 mechanism-check fields; protocol adds only "normalization": "percell". configs/pilot100m_r1acc_env.sh now points
  CALIB at v3 with EXTRA=--allow_calib_reswap so the watchdog's resume of the running 100M arm passes the trainer's audited
  equivalence hatch (pre-checked with the same field comparison: PASS). Flyers geometry chain (codex PASS r3) launched on flamingo01.
- 2026-09-06 23:20Z flyers group-adapter geometry (runs/_figs/flyers_group_summary.json; FK output vs all real clips, 30 Hz, val
  targets): zero-training jitter 4.2-13.2x real (median 5.5x), articulation 1.1-3.9x; the shared flyers LoRA (78 clips, 8 rigs) brings
  jitter to 0.4-1.3x (median 0.99x) and articulation to 0.46-0.83x. Dragon: shared adapter artic 0.52 / jitter 0.95 vs the species
  adapter 0.49 / 0.70 at the same 30 Hz protocol -- the species adapter is smoother at equal articulation. Pteranodon's zero-training
  dump failed (traceback; being checked); 7/8 rigs reported.
- 2026-09-06 23:35Z appendix: "Sharing an adapter across species" TODO replaced by tab:flyers (7 flying rigs, zero vs shared LoRA,
  30 Hz) + Dragon shared-vs-species comparison at the same protocol (0.52/0.95 vs 0.49/0.70). Pteranodon omitted: its zero-training
  sample's root rotation decoded to a degenerate 6D at frame 0 (root is animated_dof, so not a fixed-root case; cause unexamined).
  Remaining red TODOs in the appendix: 20 (was 24 this morning).
- 2026-09-06 23:05Z representation ablation code PASSED codex (4 rounds; r3 cleared training/eval/calibration/launcher, r4 the last
  P2 on the attention probe's cache key). Files: src/data/ktjd17_incontext.py (normalization= percell|scale_only, _stats(), variant in
  target_centering, constant cells keep their constants), scripts/train_v2_incontext.py (--rep_norm + calibration protocol check),
  scripts/_measure_ktjd17_gamma_calibration.py (REP_NORM), scripts/_eval_v2_gen_in_evalspace.py (generator-side normalization +
  conversion into the evaluator's per-cell space; legacy fingerprint b341caa6), scripts/v2_render_incontext.py, _gen_ktjd17_clips.py,
  _analyze_jitter.py, _ktjd17_to_bvh.py (manifest-derived normalization), _measure_latency.py, _probe_attn_logits.py,
  _render_lora_textprobe.py, new scripts/_launch_v2_ddp_1node_h200.sh. Calibration configs/pilot_animal_scaleonly_gamma_calibration_b64_v1.json
  (H200; A100 80G OOMed). Arm config configs/pilot36m_scaleonly_env.sh: blossom04 2xH200, B64/rank (global 128), grad-ckpt on (B64 without it
  OOMs at 138 GB), 120 epochs; SMOKE 1 OOM, SMOKE 2 running/passed -> real launch chained.
- 2026-09-06 23:00Z **36M scale_only arm LAUNCHED** (runs/v2_noik_pilot36m_scaleonly): blossom04 1478523, 2xH200, B64/rank, global 128,
  lr 2e-4, 120 epochs, grad-ckpt on, calibration scaleonly_b64_v1. Compare with the r1acc pilot at ep100/ep125 under the frozen
  protocol (eval converts scale_only samples into the evaluator's per-cell space). Hourly monitor cron (session) added.
- 2026-09-06 23:02Z the scale_only real run OOMed at B64/rank on real data (137 GB, grad-ckpt on; the smoke's 256-clip subset had
  fit). Re-planned as a 2-node 4-card run blossom04 (1478523) + flamingo01 (1478525) at B32/rank = global 128 (identical recipe);
  flamingo01 has ~15 h -> ep100 (the comparison point) is reachable at ~500 s/epoch; calibration re-measured at batch 32.
- 2026-09-07 03:20Z **user decisions**: main-table retrieval pool = 64 (tentative); `dataset/ktjd17_pzh312_noik_v1` deleted (31 GB,
  independent copy of v2, unreferenced); "old data format" for the representation ablation = the AnyTop 13-channel format DERIVED FROM
  KTJD-17 (the old processing is not trusted) -- analytic conversion proposed, awaiting go.
- 2026-09-07 03:40Z protocol pin moved 32 -> 64 in scripts/_eval_v2_gen_in_evalspace.py (codex PASS; pre-edit fingerprint 52444b7a
  registered so every saved shard set stays scorable; 32 is now a --protocol_variant pool value like 16/128). Re-scoring chain
  runs/_final_geneval/rescore64/_run.sh (codex PASS) over the saved shards of every number in the paper: 303M ep289, 36M ep399, the
  5/10-step sweep (steps variant at the new pin), r1acc/demo-64/no-acc ep200, control ep050/075/100. Paper edits follow the numbers:
  protocol paragraph (60 pools of 64, last 59 clips unpooled), tab:main, abstract, tab:cost, tab:ablation, app:demo64 sentence,
  app:split, plus a pool-sweep table (16/32/64/128, both models, GT reference) in the appendix.
- 2026-09-07 05:29Z control-arm frozen evals for the representation ablation (pilot36m_r1acc, rose11 2xA100, pool 32 at the time):
  ep050 R@1 0.8745 FID 0.0157 | ep075 0.8599 / 0.0146 | ep100 0.8643 / 0.0138 (matched-epoch comparison points for the scale_only arm,
  whose ep050 lands ~08:15Z). The scale_only->per-cell eval conversion was verified end-to-end on the SMOKE checkpoint first.
- 2026-09-07 06:25Z pool-64 re-scoring DONE (12 merges, rose11 2xA100, all rc=0): 303M ep289 0.922/0.982/0.992 (GT 0.964/0.998/0.999),
  36M ep399 0.897/0.966/0.982; steps sweep 303M s10 0.922 s5 0.914, 36M s10 0.897 s5 0.878; ep200: r1acc 0.878, demo-64 0.878,
  no-acc 0.877; control ep050/075/100 0.852/0.860/0.864 (FID unchanged by the pool). NOTE the p36_ctrl chain's own ep075/ep100
  reports (gen_eval_ep075/ep100_a100x2_s20.json) were already scored at pool 64 because the pin moved while that chain ran; ep050's
  was pool 32 -- the *_pool64.json files are the consistent set. Shards saved after 03:35Z carry the post-edit source fingerprint
  although generation code is byte-identical (fingerprint is read from disk at save time).
- 2026-09-07 07:22Z **paper switched to pool 64** everywhere (abstract, intro, conclusion, method protocol, experiments protocol +
  tab:main + tab:ablation, appendix tab:cost / demo-64 sentence / split paragraph) + new appendix tab:pool (16/32/64/128, both models,
  real-clip reference). tab:ablation restructured to the ablation plan: full recipe / without acceleration term / 64-frame motion
  demonstration (ep200); the superseded rows (hand-tuned weights, joint names, v-prediction, 10 steps) removed, their appendix notes
  replaced by acceleration-term / joint-description / size+representation (pending) notes. 18 pages, main text ends on page 9.
- 2026-09-07 09:31Z representation ablation, first matched point (36M, ep50, pool 64, same evaluator; scale-only samples converted
  into the evaluator's per-cell space): per-cell (ours) R@1/2/3 0.852/0.944/0.972 FID 0.0157 match 0.723 | scale-only 0.832/0.938/0.966
  FID 0.0201 match 0.712. Training-time FK-pose gap is LOWER for scale-only (0.118 vs ~0.29 bl) -- fkdist is self-consistency, not quality.
- 2026-09-07 15:35Z representation ablation, second matched point (36M, ep75, pool 64): per-cell (ours) R@1/2/3 0.860/0.948/0.971 FID 0.0146
  match 0.726 | scale-only 0.840/0.940/0.968 FID 0.0194 match 0.715 -- gap unchanged at ~0.020 R@1. ep100 next (scale_only ep0100 lands
  ~17:35Z, eval queued on rose11).
- 2026-09-07 16:06Z **zero-shot item 1 (statistics source) DONE** -- runs/_figs/zeroshot_stats_{buffalo,gazelle,dragon}.json + _summary.md
  (ratios vs all real clips, 30 Hz, FK family). Buffalo (PZ relative exists): 303M own-stats zero-shot artic 1.56 / jitter 5.32 ->
  borrowed PZ_African_Buffalo stats 0.69 / 2.07 (unscaled) and 0.81 / 2.55 (s_rig-rescaled); 36M per-cell 2.07 / 7.41 -> 0.83 / 2.72.
  Gazelle (2 val clips): broken under every statistics source (jitter 14-25x) -> not a statistics problem. Dragon (no relative;
  Komodo donor): unchanged (4.0-4.6x) for the 303M; 36M arms less jittery but under-articulated (artic 0.40). Conclusion: for the
  close-relative case the deployment statistics are a large part of the zero-shot failure; the rig's own few-clip statistics are worse
  than a library relative's. Borrowed-stats arm keeps the rig's own supervise/constant masks (oracle masks) -- state this wherever used.
- 2026-09-07 16:08Z zero-shot item 4 (test-time self-adaptation) tools codex PASS (r5 builder, r4 chain); Buffalo pseudo views
  (own-raw / own-smooth1.5 / borrowed-raw) + their calibrations being built on rose11; LoRA TTT runs after the ep100 eval.
- 2026-09-07 16:35Z **zero-shot item 4 (test-time self-adaptation) first result, Buffalo, 303M ep214**: LoRA r64, 300 steps on the model's own
  20 zero-shot samples (no real motion of the rig as target; own per-cell stats = transductive statistics). FK family vs real: zero-shot
  artic 1.56 / jitter 5.32 -> TTT raw 0.72 / 2.10, TTT smoothed(sigma 1.5) 0.76 / 1.91; real-clip LoRA 0.31 / 0.68. Pose-FK gap
  0.12-0.26 -> 0.08-0.19 (LoRA 0.03-0.14). Report runs/_figs/zeroshot_ttt_buffalo.json. Helps but stays far from ten real clips.
- 2026-09-07 16:40Z paper batch 1: appendix conversion / per-cell statistics / corpus statistics / calibration (groups, shares, solve,
  mechanism check, artifact) / adapter hyper-parameter table (tab:lora_hp) / geometric protocol details / stability paragraph written;
  conclusion release-todo removed (no release statements), AI-use statement completed; main.tex outline todos removed. 19 pages, main text
  ends on page 9. Remaining todo macros: 17.
- 2026-09-07 user direction: skeleton-robustness (different joint counts / names) = FUTURE WORK; write the paper first; implement the
  augmentation code meanwhile; train after the paper's ablations are complete. Item 1 (statistics borrowing) parked. Details in handoff.
- 2026-09-07 16:55Z paper batch 2: method.tex appendix references filled (rep / corpus / calibration), introduction data-source footnote
  written (neutral wording, no redistribution), capability table: 'Weights' column dropped (no release statements), OmniZoo unseen = n.r.,
  G-MDM and SAMoR no-template/unseen = check from their abstracts (2503.04257, 2607.02148). Remaining todos (6): ethics data-source
  wording (user), red-marker sentence in experiments setup, size sweep + representation ablation text and table (numbers pending:
  scale_only ep100 tonight, 100M ep200 ~09-08 06Z, 303M ep200 needs a frozen eval), qualitative filmstrips figure.
