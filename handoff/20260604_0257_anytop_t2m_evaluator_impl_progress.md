# AnyTop T2M Evaluator — 实施进度 Handoff

Date: 2026-06-04 02:57 UTC
设计文档(final, 已 PASS-with-amendments + AniMo 对齐): `handoff/20260604_0121_anytop_t2m_evaluator_impl_proposal.md`（读其 §1c hard requirements + §5 M0-M2 + §6 决策）。
用户约束: **evaluator 只需 smoke 通**(暂无空闲 GPU 训 evaluator —— 两训占着 6×H100+8×A100);M2 只做 CPU forward/backward + tiny-step smoke, 不实际训练。

## ✅ 最新进展 (2026-06-04 ~05:00Z) — 覆盖下方旧状态
- **bf16 适配已 commit → bf16-vae 分支** (`acf88d1`, 13 files, 4136 ins;核心代码 61 行 + 脚本/文档)。**只动 noKslot_bf16vae,noKslot_clean 一字未碰 → diffusion 续训零影响**。
- **fp32 bit-identical 验证 PASS(实测)**:GraphAttentionBlock 的 fp32 `_compute`(diffusion denoiser hot-path 的调法)在 main 版 vs bf16 版 md5 **逐位一致** `04517cbce918bc5d9150085ceced2d41` → **merge bf16→main 安全,diffusion fp32 续训 byte-for-byte 不变**。其余 4 文件(encoder/losses/motion_decoder/treeik_decoder)是同款 `F.softmax(x.float()).to(dtype)` no-op 模式(fp32 处处 no-op;losses 是 VAE-only diffusion 不用)。
- **M2 22/22 PASS**(M2 agent srun-managed CPU step):**overfit-flat 根因已查清** = 相邻 val 样本 near-duplicate,被 multi-positive mask 正确剥光对比信号;换 content-distinct 样本后 overfit **2.0814→0.6820** 正常收敛 → **非 model bug**。另发现裸 ssh 重 CPU 进程会被节点 reap(之前 smoke 跑不完/255 的根因),要用 srun --overlap --gres=gpu:0 managed step。M2 文件:`src/models/graph_salad/t2m_evaluator.py`(AnyTopT2MEvaluator + build_multi_positive_mask + symmetric_infonce) + `scripts/train_anytop_t2m_evaluator.py` + `scripts/_smoke_anytop_t2m_evaluator.py`。
- **✅ 全部完成 (2026-06-04 ~05:30Z)**:① M2 codex PASS (019e90d6) → ② evaluator commit 到 main (`92758d7`, 6 files) → ③ **merge bf16-vae → main** (`62d0408`, ort 无冲突)。**main 现统一**:diffusion + split 适配 + evaluator + bf16 VAE。merge 验证全绿(split 保留 / bf16 进 / evaluator 在 / 无冲突)。diffusion fp32 续训 bit-identical 安全。
- **单文件夹后续(用户要"以后都在 main 下做")**:`noKslot_bf16vae` worktree(bf16-vae 分支)现仍在跑 bf16 VAE(ep115+),**不能删**;VAE 训完后 `git worktree remove noKslot_bf16vae` + `git branch -d bf16-vae` 即可清成单 main 文件夹。bf16-vae 分支的代码已全在 main,删分支不丢东西。

## STATE (更新 2026-06-04 ~04:0xZ)
- **M0 DONE** (codex PASS, thread 019e907a) — manifest 生成器
- **M1 DONE** (codex PASS, thread 019e9085) — thin-wrapper dataset
- **M2 代码 DONE,smoke 部分通,未 codex 审/未 commit** — `src/models/graph_salad/t2m_evaluator.py`(381) + `scripts/train_anytop_t2m_evaluator.py`(266) + `scripts/_smoke_anytop_t2m_evaluator.py`(274) 已写、语法 OK。smoke **[1-4] PASS**(实例化 / forward 双 384 emb / InfoNCE backward / **multi-positive mask 全对**:同 motion_id∪source_motion_id∪caption_text 正确 mask,real-batch 自然 dup 也对)。但 **[5] tiny-overfit loss flat ~2.08 未降** 待查 —— M2 agent 报 synthetic 向量能 overfit(→ model forward/backward 本身 work),real 8-sample flat,疑似 smoke 配置(lr=2e-3 / 小 batch multi-positive 把多数对 mask 掉 → 对比信号弱),**非必然 model bug**。下个 session 先查这个再 codex 审 + commit。
- **⚠ M2 smoke 是 CPU 泥潭**:CPU ~28min/run(SkeletonEncoder J144 graph-attn forward 慢) + swarmh1002 ssh 反复 255 + M2 agent 孤儿 background monitor 循环重启 smoke(CPU 进程,**不抢 diffusion GPU**,随 alloc 清)。**建议:别再 CPU 28min 跑全量 smoke** —— 减 batch(8→4)/steps(40→10) 或借一张卡几秒验 overfit,快速判 [5]。
- **⚠ bf16 适配从未 commit**(用户 2026-06-04 发现):`amp_dtype` 等 **61 行(7 文件)** + bf16 launch/render **2 脚本** 全在 `noKslot_bf16vae` **工作区未提交**(main=0 / bf16-vae=0 amp_dtype,工作区=3)。改的文件:train_graph_vae.py(+37)/attention.py(+23,bf16-safe guard)/encoder.py/losses.py/motion_decoder.py/treeik_decoder.py/_launch_rot6d_fk_B.sh + untracked `_launch_bf16_vae_8card_xnode.sh`(78)/`_render_bf16_vae_recon_large.py`(240)。**"merge bf16-vae 回 main"旧计划失效**(bf16-vae 无 bf16 commit,比 main 旧 1 commit)。**待用户定**:先 commit bf16 工作区→bf16-vae(版本保护) → 验 fp32 bit-identical → merge 回 main(碰核心文件,影响 diffusion)。
- **VAE recon QA DONE**:ckpt best_recon(ep114),7 物种(J52→140)recon-vs-GT 3panel gif + montage,**GT self-check 全 PASS(0%bbox,无 double-root)**,视觉准确(主线程亲验 Elephant J137/Tiger J121),recon_err 0.92–3.23%bbox,无塌缩/抖动/穿插/frozen。gif: `noKslot_bf16vae/runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42/qa_recon_ep113/PZ_*_recon_3panel_large.gif`。
- 不碰两训 / 不抢卡 / 每 milestone codex 审(gpt-5.5 xhigh) / 两训健康:diffusion ep77 val0.3742 / bf16 VAE ep115 val_recon1.6342(均在降)。

## M0 DONE — `scripts/build_anytop_t2m_eval_splits.py`
- 产出 `data/anytop_planet_zoo_clean_L2/eval_splits/{train_main,val_all,val_action_clean,val_action_overlap,split_audit}.json`(gitignored)
- 实跑全绿: train 77882 / val 4112 / clean 824 / overlap 3288 / species_stripped 覆盖 0.7045(2897/4112)
- manifest 每条: `filename`(.npy) / `motion_id`(stem) / `source_motion_id`(canonical key, 仅 metadata) / `captions`[5] / `t5_keys`[5] / `species_stripped_cap_idx` / `has_species_stripped`
- 重跑: `ssh swarmh1002 'cd /scratch/ts1v23/workspace/noKslot_clean && /iridisfs/scratch/ts1v23/.conda/bin/python3.12 scripts/build_anytop_t2m_eval_splits.py'`
- 备注: train_main.json 114M(每条存 5 caption+5 t5_key); codex 判可接受(读一次), M1 已改用 manifest 的 t5_keys, slim 是 optional cleanup

## M1 DONE — `src/data/anytop_t2m_eval_dataset.py` (+ `scripts/_smoke_anytop_t2m_eval_dataset.py`)
- `AnyTopT2MEvalDataset(manifest_path, data_root, caption_emb_cache, split, view="full"|"species_stripped", num_frames=260, max_joints=144, drop_uncovered_species_stripped=True)` + `collate_fn`
- thin-wrapper: 底层实例化 `AnyTopDataset` verbatim forward(零预处理重写); 只做 manifest 子集/排序 + caption view 附加
- smoke 13/13 PASS(CPU): 与底层 AnyTopDataset **7 字段 torch.equal 逐元素一致**(无分叉); counts exact; caption_emb[768]; species_stripped drop uncovered→2897
- codex 2 fix 已修+验证: `has_text`=caption_valid(full→True/uncovered→False {2897,1215}); dup motion_id(base+manifest) hard-fail
- 返回字段: AnyTopDataset 原样 + `caption_emb`[768]/`caption_text`/`caption_view`/`caption_valid`/`has_text`/`canonical_action_key`(=source_motion_id, metadata)/`motion_id`
- smoke: `ssh swarmh1002 'cd /scratch/ts1v23/workspace/noKslot_clean && /iridisfs/scratch/ts1v23/.conda/bin/python3.12 scripts/_smoke_anytop_t2m_eval_dataset.py'`

## M2 NEXT — evaluator model + train + CPU smoke
产出: `src/models/graph_salad/t2m_evaluator.py` + `scripts/train_anytop_t2m_evaluator.py`
spec(设计文档 §1c + §5-M2):
- **text encoder**: 已有 T5 `[768]` → proj **384**(MLP)
- **motion encoder**: 复用 `src/models/encoder.py::SkeletonEncoder`(`d384/h8/dff1536/4graph/2temporal`, ~14.1M) + masked 时间池化 → **384**。SkeletonEncoder 池化天然 384, **共享 `coemb_dim=384`, 无需 motion_proj**。输入 AnyTop 13ch + frame_mask/joint_mask + adjacency/geodesic/joint_relations
- **独立 frozen evaluator**: 不共享 VAE/denoiser 权重; 不用 VAE latent z; 只用真实 motion 训练(本任务只 smoke 不实训)
- **对称 batch-wide InfoNCE**, **multi-positive false-negative mask** = (same motion_id) ∪ (same source_motion_id) ∪ (same caption_text of view) 都不算负样本
- **metrics(M3/M4 用, 移植 outside_docs/SALAD/utils/metrics.py 适配 multi-positive)**: R-precision group-aware(top-k 命中同组任一即对); matching_score = mean over queries of **min L2 to any positive in group**(弃 diagonal trace); FID 仅参考
- **CPU smoke 验收(用户: 不实训)**: 模型能实例化; tiny batch(几样本)forward 出 text_emb[B,384]+motion_emb[B,384]; InfoNCE loss 有限且 backward 通; multi-positive mask 形状/逻辑对; tiny overfit 几步 loss 下降。**不跑完整 4-gate 训练**(无空闲卡)
- 完成送 codex 审(gpt-5.5 xhigh, fresh thread)

## 关键路径
- 数据: `data/anytop_planet_zoo_clean_L2/`(cond.npy/motions/splits/eval_splits); T5 cache prefix `data/anytop_caption_t5_cleanL2_multi`(.embs.npy[409970,768]+.keys.json)
- 复用: `src/models/encoder.py::SkeletonEncoder`; `src/models/graph_salad/batch.py::GraphMotionBatch`; `src/data/anytop_dataset.py::AnyTopDataset`
- 无现成 evaluator/InfoNCE/metrics(outside_docs/SALAD 是固定拓扑 359-dim, 不照抄)

## 两训监控(独立, 与 evaluator 无关)
- cron `8cf8ac36`(每1h :13, session-only); diffusion ep74 val0.3743 / bf16 VAE ep109 loss0.5794 val_recon1.677@ep104; 均健康 PPID=1
