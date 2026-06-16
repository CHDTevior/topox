# Graph-CodeFlow 隔夜决策 + Ready-to-Launch 状态简报

> 产出 2026-06-09 ~11:56 BST(user 睡前授权"接下来 13h 你和 codex 商量着决策"期间)。
> 本文是**决策/状态简报**,不重复深层细节。深层文档:
> - 配方+实现状态: `handoff/20260609_0530_graph_codeflow_locked_recipe_and_state.md`
> - 逐模块行号+启动脚本审核: `handoff/20260609_0600_graph_codeflow_training_walkthrough.md`
> - 设计审 verdict: `handoff/20260609_0500_graph_codeflow_plan_review_verdict.md`

---

## TL;DR(一句话)
VQVAE 训完冻结于 **ep280**(best_model.pt, val 0.945);RVQ"塌缩"虚惊已查实为**日志假象**(codex 确认),RVQ 健康;全量 token export **正在跑、durable(PPID=1)、内容已抽检正确(含文本)**,ETA ~13:00 BST;codex 判定 **READY-when-export-finishes**。**我没有启动 CodeFlow 训练**(prep-and-wait,见 §8)——一切就绪,等 user 醒来拍板启动。

---

## 1. VQVAE 冻结决定: ep280(非 ep300)
- VQVAE `runs/vqvae_L5_C50_d512_Q4_n512_b96_300ep_seed42/` 训到 ep300 完成。
- **冻结 tokenizer = `best_model.pt`(ep280, val 0.945)**,**不用 ep300**:ep300 重建 QA 相比 ep280 漂移(各物种重建 L2 回退);ep280 在全 6 物种重建上胜 ep300(−17% 到 −70%)。
- ep280 重建 QA 已**主动渲染并发 user 审**(SendUserFile)。
- RVQ 4 个 stage 全活(eval 全深度 510–512 codes,residual 133→27 单调下降,full-Q 胜 depth-1)。

## 2. RVQ"塌缩"虚惊 → 查实为日志假象(已闭环)
- 触发:ep300 重建 QA 顺手查 ckpt 的 `codebook_active=[512,0,0,0]` + 重建 L2 ep89→ep300 回退,疑 RVQ 塌缩。
- **HELD CodeFlow 准备**,先查根因 → 是 `train_graph_vqvae.py:833` **从最后一个 TRAIN batch(带 quantizer-dropout 截断)**存的 `codebook_active`,不是 eval 全深度 → 假的 [512,0,0,0]。**RVQ 实际健康**(§1 eval 证据)。
- codex 独立确认(fresh thread `019eabd5`):logging artifact,非塌缩,non-blocking。
- 决定:冻结 ep280,继续 CodeFlow 准备。`_diag_rvq_collapse.py` 是为此写的只读诊断脚本。

## 3. Token export 状态(正在跑)
- 脚本 `scripts/export_graph_vq_tokens.py`,wrapper `scripts/_run_codeflow_token_export_full.sh`(setsid nohup,node-local)。
- **位置**: swarmh1002 GPU0,pid 1241085,**PPID 链到 1(init-adopted)→ durable**,ssh 断开不影响。GPU util ~6%(host-bound,符合 VQVAE launch-bound 史)。
- **进度**: ~27.2k/70792 train npz(11:56 BST),~12–13 clips/s,**ETA train ~13:00 BST + val ~5min**。manifest.json / index.jsonl 在各 split 结束时才写(现在还没,正常)。
- **内容已抽检正确**(抽 train/013600.npz):22 个 key 全对 —
  `z_q[16,50,512]` / `indices[16,50,4]int16` / `token_mask[16,50]` / `coarse_mask[50]` / `frame_mask_lat[16]` / `pooled_adjacency,pooled_geodesic[50,50]` / **`pooled_skeleton_embeddings[50,512]`**(实现期缺口#1 已补,在) / `assignment[64,50]` / `s_j[64,512]` / `joint_mask,rest_offsets,anytop_mean/std,parent_indices,num_joints` / **`caption_emb[768] 非零` + `caption_token_emb[64,768] 非零` + `has_text=True`**(L5 文本 cache 修复在真 export 里生效,不会训成 unconditional)。
- export 不变量已确保:tokenizer `eval()` + full Q(无 quantizer-dropout 截断)+ captions ON + `--min_text_coverage 0.99` preflight fail-loud gate。

## 4. codex verdict: READY-when-export-finishes
- 准备链 workflow 重跑(用 ep280 frozen ckpt + cleanL5 cache):
  - **7 步 end-to-end smoke 全过** + text-positive smoke 全过(caption_emb 非零 / token_mask>0 / global+token 两路各改输出 / CFG cond≠uncond)。
  - **mem/throughput profile 完成**:dataloader 非瓶颈(386 clips/s);short-train smoke flow_loss 2.04→1.22 健康下降。
  - export 当时 INCOMPLETE → codex 判 **NOT-READY = READY-when-export-finishes**(fresh thread `019eabf3`)。
- codex 启动前置建议(2 条,见 §7、§8)。

## 5. Ready-to-Launch(user 醒来的单一动作)
**前提门**(export 完成后,启动前必跑一遍,fail-loud):
1. `ls data/codeflow_tokens_cleanL5_ep280/manifest.json` 存在 + train/val 的 `index.jsonl` 存在;train npz ≈ 70792、val npz ≈ source val 数。
2. 用**最终冻结 ckpt + 真 export** 重跑定版 smoke:
   `python scripts/_smoke_graph_codeflow.py --frozen_vqvae_ckpt runs/vqvae_L5_C50_d512_Q4_n512_b96_300ep_seed42/best_model.pt --caption_cache data/anytop_caption_t5_cleanL5_multi` + text-positive 变体 → 全过。

**启动命令(single-node 2×H200 DDP,codex 推荐;flamingo01 或 blossom03 任一空闲)**:
```bash
# 在目标 2×H200 节点上,node-local setsid durable(用与 VQVAE/VAE 同款 wrapper 模式)
torchrun --standalone --nproc_per_node=2 scripts/train_graph_codeflow.py \
  --token_cache data/codeflow_tokens_cleanL5_ep280 \
  --frozen_vqvae_ckpt runs/vqvae_L5_C50_d512_Q4_n512_b96_300ep_seed42/best_model.pt \
  --batch_size 256 --lr 4e-4 \
  --epochs 600 --warmup_steps 2000 --lr_scheduler half_cosine --eta_min_ratio 0.01 \
  --cond_drop_prob 0.1 \
  --out runs/codeflow_L5_levelA_b256_lr4e4_seed42
```
- **batch/lr**: profile 后 codex 推荐 **batch256/lr4e-4**(per-process 256 × 2 卡 = global 512;相对 locked 起步 global128 = 4× → lr 1e-4→4e-4,Goyal 一致)。
  **保守替代**(locked-recipe 字面起步): `--batch_size 64 --lr 1e-4`。**我的推荐:走 profile 后的 batch256/lr4e-4**(profile 已验显存可吃、flow_loss 健康下降),但这条触及 locked"起步"语义 → **留给 user 拍板**。
- single-node 单 alloc → **不需要 cross-alloc orchestrator**;普通 torchrun 即可。
- **durable**: 真起时套 setsid nohup wrapper(`ssh <node> "cd … && setsid nohup bash _launch_codeflow.sh > log 2>&1 </dev/null &"`)。该 wrapper 是**新脚本 = 代码改 → 必经 codex 审**才提交(见 §8 第 2 条),所以我没预写。

## 6. Phase-1 决定性 gate: continuous-vs-snapped QA(最关键)
Phase-1 **只回答一个问题**:frozen post-RVQ z_q 能不能被 Graph-CodeFlow 学成**可 decode、可 snap、视觉能动**的生成?
- 训练中 `projection_qa()`(`train_graph_codeflow.py:142-189`)每 `qa_every` 步报:flow_loss / grad_norm / lr + **proj_err=mse(z_hat,z_snap)** + 逐 q code 用量 + continuous decode(z_hat) 与 snapped decode(z_snap) 均 finite + cont_vs_snap_maxabs。
- **失败类型决策树**:
  - flow loss 不降 / continuous decode 也差 → backbone/conditioning 问题。
  - continuous decode 好 / snapped decode 差 → RVQ projection(加 terminal ID CE / residual corrector)。
  - snapped decode 也好 / motion 视觉差 → tokenizer decoder 或 数据/文本条件。
- **CV 铁律**:metric 不算数,必渲染**单 gif T2M 布局(静态输入骨架 + prompt + pred,去 GT 栏)**发 user 审,视觉裁决权归 user。CFG scale 不固定 6.0,只作 QA sweep 起点(项目有 energy-overshoot 史)。

## 7. codex 启动前置建议(2 条)
1. **提交未跟踪代码**(下方 §8 列表)——属 user 的决定(commit/push 只在 user 要求时做)。
2. 启动前重跑定版 smoke + 验 manifest(已并入 §5 前提门)。

## 8. 我**没做**的(prep-and-wait)+ 原因
- ❌ **没启动 CodeFlow 训练**。原因:① codex NOT-READY-until-export-done(export 还在跑);② codex 建议先提交未跟踪代码,而 commit 是 **user 的决定**(铁律:commit/push 只在 user 要求时);③ user 原话"准备进行 vqvae 对应的 backbone 的训练"——是"准备",启动权在 user。
- ❌ **没预写 durable launch wrapper / cross-alloc orchestrator**。原因:新脚本 = 代码改 → 必经 codex 审;且 single-node 不需要 cross-alloc;留 user 拍板配置后再起草+送审。
- ✅ 做了:抽检 export 内容正确性、确认 durable、写本简报、保持监控。

**未跟踪 CodeFlow 代码(codex 建议提交,待 user 决定)**:
```
src/models/CodeFlow_Model/        (graph_codeflow.py / flow.py / token_dataset.py / __init__.py)
scripts/export_graph_vq_tokens.py
scripts/train_graph_codeflow.py
scripts/animate_graph_codeflow.py
scripts/_smoke_graph_codeflow.py  (+ _smoke_graph_codeflow_textpos.py / _textpositive.py — 两个 text-pos 变体,建议合并保一个)
scripts/build_l5_t5_caption_cache.py
scripts/_run_codeflow_token_export_full.sh
scripts/_diag_rvq_collapse.py
src/models/graph_salad/graph_vq_tokenizer.py  (加 encode pooled / ids_to_embeddings / nearest_residual_ids / prepare_skeleton_only / decode_from_indices — 全 read-only 加法)
```
（注:`graph_vq_tokenizer.py` 在 working tree,改动是纯加法,冻结 encode/decode/quantizer 行为不变,codex thread `019eaaba` 已 PASS。）

## 9. 并行训练状态(本 session)
| 训练 | 状态 | 卡 | 健康 |
|---|---|---|---|
| Graph-VQVAE tokenizer (L5, 74522 clips) | ✅ ep300 完成,冻结 ep280 | swarmh1002 6×H100(已释为 export 用) | RVQ 健康,recon QA 已交付 |
| animo4d L2 VAE (74522 clips, bf16 rot6d-FK) | 🟢 ep132/300 跑 | swarma1004+1001 8×A100 cross-node | loss ~0.52,grad bounded,active_C 70–94,ep109 spike 已恢复零复发 |
| decode-loss 扩散 | ✅ ep1500 完成 | — | 能量塌缩 −41% 已验收交付 |
| Token export | 🟡 ~27k/70792 跑 | swarmh1002 GPU0 | durable PPID=1,内容抽检正确 |

监控 cron:`2c9512b6`(VAE 8 卡,每 2h)。export 完成监控见 §10。

## 10. export 完成后我会自动做的
设了 export-completion 监控:检测到 `manifest.json` 出现 → ① 验 train/val npz 计数 + manifest 完整性;② 重跑定版 smoke(最终 ckpt + 真 export);③ 把结果追加进本文 + 报 user。**仍不启动训练**(等 user 醒来拍板 batch/lr + commit)。

---

## 11. ✅ Export 完成报告(2026-06-09 11:55 UTC / 12:55 BST,自动监控 tick 产出)

**Export 完成**(DONE 11:46:29Z,墙钟 ~87min,~15–17 clips/s host-bound):
- `data/codeflow_tokens_cleanL5_ep280/` — **train 70792 npz**(=目标,0 丢失)+ **val 3730 npz** + train/val `index.jsonl` + `manifest.json`。
- manifest:`ckpt_epoch=279`(=ep280 best_model)、D=512/Q=4/K=512/max_coarse=50/stride=4、amp=bf16、**RVQ-identity max_id_err_fp32=1.91e-06**(train+val,ids_to_embeddings≈z_q 近乎精确)。
- 抽检 train+val npz:20 个 expected key 全在(missing=NONE)、`has_text=True`、`caption_emb` 非零、z_q[16,50,512]、indices[16,50,4]。

**定版 smoke(最终冻结 ckpt + 真 cleanL5 cache,swarmh1002 GPU0 我方空闲卡)**:
- **主 7 步 smoke 全过**:RVQ-identity 9.5e-7 / projection_error 0.111 finite / 两路 decode finite / skeleton-only self-transfer byte-identical(err 0.0)/ flow_loss 1.996 grad_norm 0.181 finite / ODE+CFG finite。
- **文本参与 probe(`_smoke_graph_codeflow_textpos.py`)全过 4 项**:caption 非零(abs.sum 66.7/70.7,真 caption)/ token_mask 19,16 / **两路文本都改输出**(GLOBAL FiLM Δv=0.079、TOKEN cross-attn Δv=0.717、COMBINED Δv=0.499)/ CFG cond≠uncond Δv=0.912、sampler finite。

**⚠ 一个 false-alarm 已查实(非问题)**:另一个 probe `_smoke_graph_codeflow_textpositive.py`(10:52 版)step E 报 "text route DEAD"(Δv=0.000e+00)。**查实为该 probe 的设计缺陷**:它在 at-init 模型上直接测 final `v_pred`,但 `output_proj` 是 **zero-init**(`graph_codeflow.py:191-192`,DiT 风格,初始 v_pred≡0)→ v(text-on)与 v(text-off) 都恒为 0 → delta 必为 0,**与文本死活无关**。正确的 `_textpos.py` 先 de-zero 27 个 zero-init 张量(output_proj + text-cross o_proj + FiLM)再测,已证文本真参与(上一段 Δv≠0)。佐证:主 smoke STEP 6a grad_norm 0.181 非零=梯度确实流到文本参数。**→ 建议:删除/修正 `_textpositive.py`(at-init 无法通过,是误导),保留 `_textpos.py` 为唯一文本 probe**(呼应 §8 "两个 text-pos 变体建议合并保一个",现确定保哪个)。

**结论:CodeFlow READY-to-launch。** export 完整 + 主 smoke 全过 + 文本真参与。启动命令见 §5,启动权/batch-lr/commit 决定留 user(prep-and-wait,本 tick 未启动训练)。export-completion 监控 cron 已自删。
