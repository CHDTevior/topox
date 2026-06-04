# Token-Level 文本条件 Diffusion — 数据→模型→训练 审核走读

Date: 2026-06-05. 给人审核用,不是代码 dump。每节:**这是什么 → 怎么设计 → 为什么 → 红线**。
对应实现细节见 `handoff/20260604_231329_token_cross_attn_impl_report.md`(agent 报告 + 行号);本文是它的人话版。

---

## 0. 一句话:这次改了什么

**之前(mean_additive)**:一条 caption → T5 → **mean-pool 成 1 个 [768] 向量** → denoiser 每层把它**广播加**到所有 motion slot 上。文本被压成一个点,词序、"先 A 再 B"、身体部位这些细节全丢了。

**现在(token_cross_attn,可选)**:一条 caption → T5 → **保留逐 token 的 [L,768] 序列** → motion tokens **cross-attend** 这串 text token。文本以"一串词"的形式被运动查询,理论上能抓多动作 / 词序 / 部位词。

**最重要的一条红线**:`mean_additive` 是**默认**,旧 ckpt 和你**正在 H100 上跑的那个 fp32 mean diffusion 完全不受影响**。token 是一个**新开关**(`text_mode=token_cross_attn`),不开就是原样。

这次是 **token-path PROBE**(探路),**不是严格 A/B**(原因见第 3 节)。

---

## 1. 数据视角:一条 caption 怎么变成模型能用的 token

### 1.1 三层数据
- **起点**:`data/anytop_planet_zoo_clean_L2/motion_texts_by_file_with_codex_drafts.json` — 每个 motion 5 条 caption。
- **mean cache(早就有的,不动)**:每条 caption → T5-base → mean-pool → `[768]`。存在 `data/anytop_caption_t5_cleanL2_multi.embs.npy [409970,768]` + `.keys.json`(409970 个 key,**定义了行顺序**)。
- **token cache(这次新建)**:每条 caption → 同一个 T5-base → **保留 token 序列** → `[L=64,768]` + `mask[64]`。存 `.tokens.npy [409970,64,768] fp16 = 37.5GB` + `.token_mask.npy [409970,64] bool`。
  - 生成脚本:`scripts/precompute_t5_caption_tokens.py`(新)。

### 1.2 三个关键设计(你审核重点)
- **对齐(idx-align,你叮嘱的铁律)**:token cache **不自己重新走 json**,而是**读 mean cache 的 `.keys.json`,逐行按它的顺序写** → `key k ⇔ token 行 k ⇔ mean 行 k`。这样训练时随机抽到的那条 caption,它的 mean 向量、token 序列、原文字符串**是同一条**,不会错位。
  - **验证**:token 用 mask 做 mean-pool 后,和 mean cache 的余弦相似度 = **1.000000**(逐条)。
- **L=64 够不够**:统计了**全部 409970 条** caption 的 token 长度 → 最长 **42**,L=64 **截断率 0.000%**(没有任何 caption 被切)。
- **40GB 怎么不爆内存**:dataset 用 **`np.load(mmap_mode='r')` 内存映射**,不把 37.5GB 读进 RAM(8 个 dataloader worker × 40GB 会把主机内存撑爆),每条样本只 page-in 一行。

### 1.3 dataset 怎么取(`src/data/anytop_dataset.py`)
- 多了 3 个可选参数:`caption_token_cache` / `return_caption_tokens` / `caption_token_max_len=64`。**不开 token 时,返回的东西和现在逐字节一样**。
- 开 token 时:`__getitem__` 里**那个随机选 caption 的 idx 只抽一次**,mean 向量、原文、token 行**都用这同一个 idx** — 这是对齐不出错的关键点。
- 没 caption 的 motion → 全 False mask + 全 0 token(后面 denoiser/CFG 会把它当 uncond 归零)。

**红线**:不开 token,dataset 行为 byte-identical;token 行必须和 mean 行同一个随机 idx。

---

## 2. 模型视角:denoiser 怎么"读"这串 token

### 2.1 每层的文本子块从"加法"变"cross-attention"
denoiser 每层顺序:`spatial graph-attn → FiLM → temporal-attn → FiLM →` **文本** `→ FiLM`。
这次只改"**文本**"这一格,变成二选一(`src/models/graph_salad/denoiser.py`):
- **mean_additive(默认)**:mean 向量 `[B,768]→[B,D]`,按 `has_text` 门控后**广播加**。**和以前一字不差**。
- **token_cross_attn(新)**:把 motion 展平成 `[B, T×C, D]` 当 **query**,去 cross-attend text token `[B, L, D]`(key/value)。

### 2.2 新模块 `TextCrossAttention`(本次核心,我逐行审过)
标准多头 cross-attn(q=motion、k/v=text),但有 3 个为我们场景加的关键设计:
- **(a) bf16 安全 — softmax 强制 fp32**:`softmax(scores.float()).to(dtype)`。这次要 bf16 训,softmax 在 bf16 下数值不稳(还有 -1e9 mask sentinel),所以强制 fp32 算再转回。**fp32 训时这步是 no-op**。和 GraphAttentionBlock 同款做法。
- **(b) CFG-uncond 严格归零**:如果一条样本 `has_text=False`(无条件分支),它的所有 text key 全被 mask → 这条样本的 cross-attn 输出被**显式乘 0**。**不靠"softmax over 全 -inf"**(那会出 NaN)。保证无条件分支对文本贡献**精确为 0**。
- **(c) 输出投影 zero-init**:`o_proj` 权重/bias 初始化为 0 → token 路径在训练开始时是**恒等残差**(等于没加),从一个稳定起点慢慢学。新加的分支不会一上来就扰动。

### 2.3 向后兼容(你审核重点 — 关系到正在跑的 mean diffusion)
- `TextCrossAttention` 和 `text_token_proj` **只在 token mode 才建**。mean mode 下模型结构、state_dict **和旧 ckpt 逐位一致**。
- **实测**:旧的 mean ckpt(epoch 120,args 里**没有** text_mode)直接 strict-load 进默认 mean 模型 → **0 missing / 0 unexpected**;mean forward 和改动前 **max_abs = 0.0**(逐位相同)。token 模型 = mean 的 state_dict **超集**(多 134 个 token 专属 key)。
- 参数量:mean **63.5M** → token **75.4M**(+12M 是 11 层 cross-attn)。

**红线**:不开 token,模型与旧 ckpt 逐位一致;CFG uncond 严格 0、不 NaN;softmax fp32。

---

## 3. 训练视角:怎么训、和正在跑的 mean diffusion 什么关系

### 3.1 训练配置(token B 这次)
- **目标/loss/scheduler 完全没动**:还是 DDIM + v-prediction + masked MSE。这次只换"文本怎么进 denoiser"。
- **CFG**:10% 概率把 text drop 成 uncond。token mode 下 uncond = 上面的 cross-attn 归零。
- **bf16 混合精度**:autocast 包 VAE-encode + denoiser forward;**scheduler 数学 + loss 保 fp32**。
  - ⚠ 这次顺手修了一个 bf16 数值 bug(codex 抓的):`masked_v_mse` 的分母在 bf16 下会被舍入(mask_sum 83200 被舍成 83456,loss 偏 +0.31%)。改成 fp32 算分母。**这个 fix 对 fp32 是 no-op(逐位一致),所以你正在跑的 fp32 mean diffusion 续训零影响** — 我专门验证过 `fp32 new-vs-old absdiff = 0`。
- **冻结 VAE**:用 **bf16 训的 ep209 best VAE**(`val_recon 1.3983`),只 encode、不解码、不训。
- **资源**:8 卡 A100 **cross-node DDP**(swarma1004 + swarma1001,各 4 卡,IB 连),global batch 64,lr 6.67e-4(Goyal 线性缩放)。

### 3.2 和 mean diffusion 的关系(最重要的标签,别搞混)
- 这是 **token-path PROBE**,**不是严格 A/B 论文结论**。
- 你正在 H100 上跑的 mean diffusion 用的是**另一个 VAE**(4 卡 fp32 rot6d_fk ep79)+ fp32。**因为 VAE 不同,它只能当粗参考 sanity,不能和 token B 直接下最终结论**。
- **判断路径**:token B 如果(① 训练稳、loss/val 不差 ② 视觉上文本响应明显比 mean 更听话),**再补一个同 VAE 的 mean baseline** 做真正隔离 text 变量的 A/B。如果 token B 很差 / OOM / 不稳,**先修 token 实现 / 调权重,不浪费资源训 mean A**。

### 3.3 验收三件事(你定的)
1. token_cross_attn 训练 loss 有限、梯度真打到 `text_token_proj` + `text_cross_attn`。
2. 和 mean diffusion 做**非严格视觉对照**,重点看多动作 / body-part / slow-energy caption。
3. 明显有希望,再重训 same-VAE mean A 做正式结论。

---

## 4. 红线汇总(你审核时的"出错就停"清单)
1. `mean_additive` 默认、旧 ckpt strict-load → **正在跑的 fp32 mean diffusion 续训零影响**(已逐位验证)。
2. `masked_v_mse` fix **fp32 byte-identical**(absdiff=0),只修 bf16 分母。
3. **idx-align**:token 行 ⇔ mean 行 ⇔ caption 原文,同一个随机 idx(余弦 1.0 验证)。
4. **CFG-uncond 严格 0**,不 NaN(显式归零,不靠 softmax-over-(-inf))。
5. VAE frozen、loss/scheduler/objective/target 全没动。
6. token cache **mmap**,不进 RAM。

---

## 5. 当前状态(审核背景 — 状态表述已按 user review 2026-06-05 修正)
- token 实现:**codex PASS**(thread 019e94b8)。
- masked_v_mse fix:已应用 + 双验证(fp32 absdiff=0 / bf16 0.99219→1.0)。
- cross-node orchestrator(`scripts/_launch_token_diffusion_8card_a100.sh`):**codex PASS-after-fix**(019e94d2;修了一个 cross-node NCCL P2P/SHM 该 enable 没 enable 的坑)。
- 全量 token cache:**37.5GB 完成**,idx-align 验证(余弦 1.0)。
- **token 8 卡 INFRA 已验证(注意:这≠完整 smoke pass)**:cross-node rendezvous(WORLD_SIZE=8)+ NCCL via IB + bf16 autocast ON + denoiser 构建(75.4M)+ dataloader/preflight + GPU 吃满 no-OOM(46/80GB),都已在日志确认。**但 SMOKE 在 training-entered 后即被 kill 去真跑** → `scripts/_smoke_token_8card.log` 里 torchrun 是被 kill 的 **SIGTERM / rc=1(不是正常完成)**,smoke **没跑到第一个 loss/epoch**。
- **⏳ 完整 pass(首个 finite loss + 梯度到 cross-attn + ep0)证据尚未出现** —— 等 token B 真跑跑出第一个 ep0/loss 后再盖章(= 验收第①件)。**红线:不要把"已启动且未崩"当 smoke pass。**
- **token B 真跑:已启动 + training-entered + GPU 吃满**(setsid durable,主 orch 243943 PPID=1),**未到第一个 loss/ep0**(在跑)。

## 6. 关键文件(你审核可查)
- 数据:`scripts/precompute_t5_caption_tokens.py`(token cache 生成) / `src/data/anytop_dataset.py`(token 读取 + mmap)。
- 模型:`src/models/graph_salad/denoiser.py`(`TextCrossAttention` + text_mode 路由) / `src/models/graph_salad/batch.py`(token 字段 + 校验)。
- 训练:`scripts/train_denoiser.py`(CLI + CFG + masked_v_mse fix + resume-assert) / `scripts/_launch_token_diffusion_8card_a100.sh`(8 卡 cross-node orchestrator)。
- 推理:`scripts/animate_denoiser.py`(token 采样 + token prompt helper)。

**注**:本次改动均**未 commit**(等你审核 + token B 真跑 confirm 后再 commit 到 main)。
