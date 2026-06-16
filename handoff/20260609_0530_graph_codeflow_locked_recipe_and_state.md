# Graph-CodeFlow Phase-1 锁定配方 + Session 检查点

> 2026-06-09 ~05:30 BST。user 审完 review verdict(`handoff/20260609_0500_graph_codeflow_plan_review_verdict.md`)+ 拍板全部 5 个 fork。
> 计划源: `handoff/20260609_graph_codeflow_rvq_backbone_plan.md`(966 行)。verdict: SOUND-WITH-CHANGES 无硬伤。

## 🔒 LOCKED 第一版配方(user 2026-06-09 拍板,执行者不再问 fork)

Phase 1 **只回答一个问题**: frozen Graph-VQVAE 的 **post-RVQ z_q** 空间,能不能被 Graph-CodeFlow 学出**可 decode、可 snap、视觉上能动**的生成?

- **Level-A minimal graph-flow**(非 Level-B CodeFlow-like Graph DiT)
- **empirical z_q normalization**(扫训练集导出的真实 z_q,valid-token 上算 mean/std;**不**做 codebook-stat ablation,除非训不稳/projection_error 异常)
- **terminal ID CE OFF** —— flow-only loss(纯 rectified-flow MSE),只有 snapped 退化才加 CE
- **continuous-vs-snapped QA 必做**(最关键 gate): 同时看 `continuous decode(z_hat)` + `snapped decode(z_snap)` + `projection_error=mse(z_hat,z_snap)`
- **batch 64 起步**(lr 1e-4)+ profile 一次定能跑多大(**不**做 batch 消融,加 batch 同步加 lr)
- **CFG scale 不固定 6.0** —— 只作 QA sweep 起点(项目有 energy-overshoot 史)
- **strict visual QA**(单 gif T2M 布局: 静态输入骨架 + prompt + pred,去 GT 栏)

**失败类型决策树**(过不了时按此定位):
- flow loss 不降 / continuous decode 也差 → backbone / conditioning 问题
- continuous decode 好 / snapped decode 差 → RVQ projection / terminal ID / residual corrector
- snapped decode 也好 / motion 视觉差 → tokenizer decoder 或 数据/文本条件

过了 Phase 1 → 再升 Level-B。

## 6 个实现期必补缺口(来自 review verdict)
1. `encode()` 暴露 `pooled_skeleton_embeddings`(pool 算了被丢)
2. 加 `decode_from_indices(indices, skeleton_meta, batch)` 入口
3. Exporter 开 captions(VQ 训练是 `load_captions=False`)
4. Export 时 tokenizer `eval()` + full Q(training 模式有 quantizer-dropout 截断)
5. 文本编码器用仓库 T5-768 dual-text,不是 CodeFlow 的 CLIP(只迁机制)
6. `eval_cond_scale` 当 sweep 起点不固定

## 实现状态: ✅ 完成(2026-06-09 ~06:00, smoke 7/7 PASS + codex PASS thread `019eaaba`)
- **已建**: tokenizer APIs(`graph_vq_tokenizer.py` 加 encode() 返回 pooled_skeleton_embeddings + ids_to_embeddings/nearest_residual_ids/prepare_skeleton_only/decode_from_indices,全 read-only 加法,encode/decode/quantizer 输出不变)+ `src/models/CodeFlow_Model/`(graph_codeflow.py Level-A + flow.py(rectified-flow+ODE/CFG+empirical-norm)+ token_dataset.py)+ `scripts/{export_graph_vq_tokens,train_graph_codeflow,animate_graph_codeflow,_smoke_graph_codeflow}.py`。
- **7 步 smoke 全过**(blossom03,2 真 L5 clip,当前 ckpt ep200): RVQ 恒等 valid 9.5e-7/padded 0; projection_error 0.1331 finite; 两路 decode finite; **skeleton-only 自迁移 byte-identical(err 0.0)**; 一步 flow grad finite。
- **codex PASS**(独立验 RVQ 数学/2D mask/target=z_q/flow-only/empirical-norm/无回归 + RVQ-equiv probe 1.19e-7)。
- **0 改共享行为**(denoiser/attention/anytop_dataset 的 M 是早先的,没碰;CodeFlow 只 import)。CFG 默认 4.0(sweep 起点);empirical norm 默认全训练集。
## ✅ L5 文本 cache blocker 修复完成(2026-06-09 ~07:00, user 审出, workflow 修+验, codex PASS `019eaaf2`)
user 审文档+码发现真 blocker: CodeFlow 脚本默认 cleanL2 caption cache 只覆盖 L5 的 510/74522(0.68%) → 会训成 unconditional flow; 老 smoke 用 cleanL2 没证文本路径。已修:
- **L5 T5 cache 建好**: `data/anytop_caption_t5_cleanL5_multi.{npz,embs.npy,keys.json,tokens.npy,token_mask.npy}`, **覆盖 74522/74522**, 格式同 cleanL2(adapter `scripts/build_l5_t5_caption_cache.py`)。
- **3 脚本切 cleanL5**: export_graph_vq_tokens.py / animate_graph_codeflow.py 默认 cleanL5; _smoke_graph_codeflow.py 参数化(`--frozen_vqvae_ckpt`/`--caption_cache`, 去硬编码)。
- **export preflight fail-loud gate**: `--min_text_coverage`(默认 0.99), 文本覆盖不足/全零 → raise 中止。
- **text-positive smoke 全过**: caption_emb 非零 / token_mask>0 / global+token 两路各自改输出(Δv 0.09/0.39)/ CFG cond≠uncond(Δv 0.705); preflight cleanL2 fail / cleanL5 pass。

## ⏭ 下一步(下个对话, 等 VQVAE ep300 ckpt)
1. VQVAE ep300 训完(~11:30-12:30 BST)→ cron `1f9f7ed3` 自动渲重建 QA。
2. 用**最终冻结 ckpt + cleanL5 cache** 重跑 text-positive 定版 smoke。
3. 全量 token export(`scripts/export_graph_vq_tokens.py`, 默认已 cleanL5 + preflight gate, captions ON/eval/full Q)。
4. `scripts/train_graph_codeflow.py --mem_profile` 或小 batch smoke → 起 Level-A 正式训练(batch64/lr1e-4, LOCKED 配方, 空闲卡 flamingo/blossom 4×H200 + rose09 2×A100, 或 VAE/VQVAE 腾出的卡)。
5. **continuous-vs-snapped QA 是决定性 gate** + 失败类型决策树。

## 并行训练状态(此 session)
- **Graph-VQVAE tokenizer**: ep~190/300, swarmh1002 6×H100, ETA ~10:40 BST(它是 CodeFlow 的 frozen tokenizer)。cron `1f9f7ed3`。
- **animo4d L2 VAE**: ep~85/300, swarma1004+1001 8×A100 cross-node, spike 已恢复 speed_ratio 1.04 健康。cron `2c9512b6`。
- **decode-loss 扩散**: ✅ 完成 ep1500, 能量塌缩修复 −41% 已验收交付。

## 空闲卡(扩容后)
flamingo01 2×H200 + blossom03 2×H200 + rose09 2×A100 = 6 GPU 空闲。
