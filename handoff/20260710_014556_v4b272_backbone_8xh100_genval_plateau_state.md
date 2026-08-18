# v4b-272 CodeFlow Backbone — 8×H100 训练中 + 人体 gen-eval 平台期 (STATE)

**STATE (2026-07-10 01:46Z) — 精简读取区,新会话只读这 5 行即可接手**
- **status**: 训练健康运行中,ep203/600,无 NaN,best val_flow **0.21821**
- **current stage**: Graph-CodeFlow backbone (graph_pscf 287M) 在冻结 v4b VQVAE 的 post-RVQ z_q 上做 rectified-flow
- **next-critical**: (a) 短板 alloc **977973 剩 ~23h** → 到期需 re-topology(先报 user);(b) **ep300** 跑大样本离线 gen-eval(user 已定)
- **resource**: 8×H100 同节点 swarmh1002 cross-alloc = 6 个 swarm_h100 alloc,**无 watchdog**(cron 33c67891 每 30min 是唯一安全网)
- **pending**: 大样本离线 gen-eval 推迟到 ep300(user 2026-07-10 决定);视觉 QA 已 **user 判定 PASS**

---

## 1. 当前训练(唯一在跑的东西)

```
OUT      = runs/codeflow_graph_pscf_v4b272neutral_n8192_b16g64_lr8e5_4xh200_seed42
orchestr = scripts/_launch_graph_pscf_8card_h100.sh   (codex-PASS + smoke-PASS)
log      = scripts/_train_gpscf_8card.log
frozen   = runs/vqvae_v4b272neutral_C96_J144_d512_Q4_n8192_b16g64_300ep_curric50to60_seed42/best_model.pt (ep219, val 0.864)
evalckpt = runs/anytop_t2m_evaluator_distilbert_coemb512_gb128_lr1e-4_mfd12_v4b272_seed42/best_model.pt
data     = data/animo4d_L4TB_plus_human_v4b272neutral   (J144, splits/val.txt = 5150 clips)
tokens   = data/codeflow_tokens_v4b272neutral_n8192_ep219_fulllen300  (empirical_stats.pt count=66,219,731)
```

**拓扑**:6 个 alloc → 8 rank,**全部 nproc_per_node=1**(uniform,torchrun global-rank 才干净)
| rank | jobid | gres | CVD |
|---|---|---|---|
| 0-3 | 977973 / 977974 / 977975 / 977976 | gpu:1 | 0 |
| 4,5 | 988069 (双卡拆 2 rank) | gpu:2 | 0 / 1 |
| 6,7 | 988070 (双卡拆 2 rank) | gpu:2 | 0 / 1 |

**超参**:`batch8 × 8 rank × accum1 = global 64`,lr 8e-5,600 ep,warmup 2000,两阶段 human curriculum(3.0@ep0 → 4.5@ep50),gen-eval 每 50ep。
> H100 80GB **装不下 batch16**(~78GB OOM),所以 batch8 + 8 卡凑回 global64 → **学习效率与原 4×H200 完全一致**。

**alloc 剩余时间(@2026-07-10 01:46Z)**:988070 3d10h / 988069 3d10h / 977976 2d18h / 977975 2d18h / 977974 1d02h / **977973 23h24m ← 短板**

### 977973 到期后怎么办(重要,别临时拍脑袋)
8 卡缺 1 = 7 卡,**7 不整除 global64**。→ 从剩余长命卡里重新拓扑成**能整除的卡数**,保持 global64:
- **4×H100** = `BATCH_SIZE=8 GRAD_ACCUM=2`(推荐)
- 2×H100 = `BATCH_SIZE=8 GRAD_ACCUM=4`

改 `JOB_A..F` env + `BATCH_SIZE`/`GRAD_ACCUM` 后重起 orchestrator。**先报 user 再动**(重起/迁移属 user 已授权范畴,但要吱一声)。

### 重起 8 卡(6 alloc 都在时)
```bash
ssh swarmh1002 "cd /scratch/ts1v23/workspace/noKslot_clean && \
  RESUME_CKPT=runs/codeflow_graph_pscf_v4b272neutral_n8192_b16g64_lr8e5_4xh200_seed42/last_model.pt \
  OVERWRITE=0 EMPIRICAL_MAX=0 GEN_EVAL=1 \
  EVALUATOR_CKPT=runs/anytop_t2m_evaluator_distilbert_coemb512_gb128_lr1e-4_mfd12_v4b272_seed42/best_model.pt \
  OUT=runs/codeflow_graph_pscf_v4b272neutral_n8192_b16g64_lr8e5_4xh200_seed42 \
  setsid nohup bash scripts/_launch_graph_pscf_8card_h100.sh > scripts/_train_gpscf_8card.log 2>&1 </dev/null &"
```
- `RESUME_CKPT` **必须全路径**(trainer 靠 `parent == out_dir` 判 resume_in_place),`OVERWRITE=0`
- `EMPIRICAL_MAX=0` = 命中预暖缓存。**绝不用小值**(会 clobber `empirical_stats.pt` → 下次冷扫描 ~27-40min)
- 停 orchestrator 用 `fuser -k .aris/meta/.gpscf8card.lock`(按名 pkill 会误杀)

---

## 2. 在线 gen-eval:人体 R@1 **在 ep149→ep199 之间停止上涨**

协议:pool-32 / shuffle=True / euclidean / reps=20,text↔motion(R-precision 永远是 text↔motion)。n=256,其中**人体仅 67**。

| epoch | 人体 R@1 | 人体 R@2 | 人体 R@3 | 动物 R@1 | 整体 R@1 |
|---|---|---|---|---|---|
| 49  | 0.451 | — | — | 0.9997 | 0.903 |
| 99  | 0.522 | — | — | 0.9997 | 0.918 |
| 149 | **0.661** | 0.796 | 0.885 | 0.9997 | 0.952 |
| 199 | **0.654** | 0.774 | 0.838 | 0.9997 | 0.945 |

**怎么读(别粉饰也别恐慌)**
- R@1 的 −0.007 **落在噪声里**(n=67,翻一个样本 = 1/67 ≈ 0.015)。**不能说退化**。
- 但前三段斜率 +0.071 / +0.139,ep149→ep199 是 **+0** → **平台期是真的**,不是噪声能解释的"本该继续涨"。
- R@3 掉 0.048(≈3 个样本),方向一致,值得警惕。
- **loss 仍在降**(best 0.22135→0.21821)。flow-matching loss 与检索精度不是同一件事 → 再降 loss 不保证人体对齐提升。
- 动物饱和 0.9997(R@2/R@3=1.0),**没被 human curriculum 拖累**。
- 人体 diversity_gen 1.348 vs GT 1.373(轻微欠多样)。
- `fid` 一直 `null`(n=256 < 1024 被跳过)。

**天花板**:人体 text→RECON(重建上限)= **0.929**。生成停在 0.654,**gap 0.275**。不是到顶,是这套配置在这个点不再往上走。

---

## 3. 视觉 QA:**user 判定 PASS**(2026-07-10)

**这是关键交叉验证。** 动作视觉上是对的,但 R@1 仍差天花板 0.275 → **gap 更可能出在"文本-动作可判别性 / 评估器 / n=67 小样本",而不是"生成的动作不像人"**。此假设留到 ep300 用大样本 eval 验。

渲染产物:`runs/<OUT>/qa_ep190_v4b/` 12 个 gif(8 人体 + 4 动物)+ `t2m_summary.txt`
布局:`input骨架 | PRED-snapped(部署真实产物) | PRED-continuous(snap前) | GT(红,最右)`,按 GT 真实长度生成。

**复现命令**(在**空闲卡**上跑,绝不碰训练卡):
```bash
srun --jobid=<IDLE_JOB> --overlap --ntasks=1 --gres=gpu:1 --cpus-per-task=8 \
  python scripts/animate_graph_codeflow.py \
  --flow_ckpt runs/<OUT>/last_model.pt \
  --frozen_vqvae_ckpt runs/vqvae_v4b272neutral_.../best_model.pt \
  --out runs/<OUT>/qa_epNNN --split val --num_frames 300 \
  --clip_names "HML3D_Human_000002.npy,...,Deer___Gallop_271.npy" \
  --caption_emb_cache   data/anytop_caption_t5_v4b272neutral_multi.npz \
  --caption_token_cache data/anytop_caption_t5_v4b272neutral_multi \
  --anytop_root         data/animo4d_L4TB_plus_human_v4b272neutral
```

### 用这脚本的 4 个坑(踩过,别再踩)
1. **`--species` 选不到人体** —— 人体 `object_type=None`,分类靠 `motion_id` 前缀(`HML3D*`=human / `PZ_*`=animal / else=truebones)。选人体必须用 **`--clip_names <basename>.npy`**。
2. **caption cache 默认是旧的 `cleanL5`**,必须显式传 v4b 的。且 v4b 是**文件前缀式**(`.tokens.npy`/`.token_mask.npy`/`.keys.json`/`.npz`),不是目录。
3. **`ta` 是冻结 VQVAE 的 train_args,不是 backbone 的** —— `max_joints=144` / `temporal_stride=4` / `anytop_root` 都从 VQVAE ckpt 读(backbone ckpt args 里**没有**这些键)。所以 `--stride` 不用传,但 `--anytop_root` 建议显式传。
4. **`graph_pscf` 必须显式 `--num_frames 300`**(否则脚本 fail-loud 拒绝 64 帧默认值)。T_lat = 300/4 = 75 = `max_T_lat`。
5. 数据集 `split='val'` 读的是 **`splits/val.txt`(5150 clips)**,不是 val_frac/seed 内部随机划分;`eval_splits/val_human.json`(1342)/`val_animal.json`(3808)里的 clip 都在其中。

### 12 个 clip 的客观数值(speed_ratio = pred能量/GT能量;proj_err = flow输出离最近码本点)
人体 proj_err **全部低(0.34–0.92)**,动物偏高(1.07–2.00,且鹿 T=18/熊 T=25 极短、大象近静止)。

---

## 4. 待办 / 已定决策

| 项 | 状态 |
|---|---|
| 视觉 QA(12 gif) | ✅ **user 判定"对的没问题"** |
| 大样本离线 gen-eval(全量 1342 人体 + FID) | ⏸ **user 定:等 ep300 再跑**(2026-07-10) |
| 在线 gen-eval | 自动在 **ep249 / ep299** 各跑一次,照常报 user |
| 977973 到期 | 到期前报 user,re-topology 到 4×H100 (bs8×4×accum2) |
| alloc 剩 <2h 且无接替 | 提醒 user 续卡(**我不能 self-submit**) |
| 训练终点 | 600ep 跑完 或 user 叫停 → 报 user 并删 cron 33c67891 |

**大样本 eval 的目的**:定性"平台期 vs 噪声"。现有证据(n=67、无 FID)**不足以判定**。届时需找空闲卡(rose07 那个 alloc 那时早过期)。

---

## 5. 铁律(不可降级)

不 self-submit / cancel slurm ・ 不抢别项目正在用的卡(**swarma1004 = tlcontrol,勿动**) ・ 不 kill 无显式授权 ・ 代码新增/改必过 codex(gpt-5.6-sol max;MCP 断则 `codex exec`,不传 `--sandbox`) ・ smoke 不过不真跑 ・ **CV 质量看可视化不只看 metric** ・ QA 渲染直接 SendUserFile 发 user 审,**不自己下结论** ・ R-precision 永远 text↔motion ・ `EMPIRICAL_MAX=0` 不用小值 ・ node-local ssh(绝不读登录节点 iridisfs 热文件)

---

## 6. 会话备注(非项目状态)

2026-07-09~10 的监控 tick 里,我的文本输出出现过大量无意义的 `court` token —— 是**高重复度 tick 模板导致的采样退化**,已核验 **未污染任何 fingerprint / launcher / train.log / 命令**(grep 命中 0)。读旧 transcript 时忽略即可。修法:打散 tick 输出模板,不复读同一段话。
