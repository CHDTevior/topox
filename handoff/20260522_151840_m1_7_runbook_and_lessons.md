# M1.7 anytop13 — 操作手册 + harness 流程 + 失败经验教训

> 写于 2026-05-22 15:18。这份文档不是进度交接（进度见
> `20260522_005629_m1_7_anytop_progress.md`），而是**怎么操作 + 我们的工作
> 流程 + 这一路（M1.5R → M1.7）踩过的坑**。新人接手 / 未来 session 先读这份。

## STATE
- **status**: M1.7 anytop13 decoder A/B 已完成；`coarse_xattn` 已定为默认 decoder
- **current stage**: Part 1 (coarse_xattn 设默认) 已改码、待 codex；Part 3 (graph-temporal decoder 可选分支) 待专门规划
- **next-critical**: codex 审 Part 1 → 规划 + codex Part 3
- **resource**: allocs 925437@swarma1003、925436@swarma1004（A/B 训练已完成，GPU 现空闲，可 smoke）
- **pending**: M1.7 全部改动 + A/B 工作仍**未 commit**；Part 3 实现

---

## §1 如何执行命令（绝对路径）

工作目录 `/scratch/ts1v23/workspace/noKslot_clean`（== `/iridisfs/scratch/ts1v23/workspace/noKslot_clean`，同一文件系统两个挂载点；脚本里统一用 `/scratch/...`）。

### 1.1 环境
```bash
source /scratch/ts1v23/.conda/etc/profile.d/conda.sh && conda activate graph_salad
```
conda 根 `/scratch/ts1v23/.conda`，env 名 `graph_salad`（torch + transformers 4.39.3）。

### 1.2 启动训练（deploy 脚本）
```bash
cd /scratch/ts1v23/workspace/noKslot_clean
JOBID=<活alloc的id> NODE=<节点> POOL_TYPE=dynamic OUT=runs/<run名> \
  ssh <节点> "cd /scratch/ts1v23/workspace/noKslot_clean && \
    JOBID=<id> NODE=<节点> POOL_TYPE=dynamic OUT=runs/<run名> \
    setsid nohup bash scripts/_deploy_train_anytop13.sh \
      > logs/deploy_<run名>.out 2>&1 < /dev/null &"
```
- 环境变量**必须写进 ssh 命令字符串内部**（见 §3 坑 #3）。
- `scripts/_deploy_train_anytop13.sh` 内部：校验 alloc RUNNING + owner==你 →
  `setsid nohup srun --jobid=$JOBID --overlap` 注入已有 alloc（**不 sbatch、不 scancel**）→ PPID=1，ssh 断了训练不死。
- 可覆盖的 env 默认值（脚本 `:76-99` 区）：`EPOCHS=1000 LR=4e-4 BATCH_SIZE=16
  D_MODEL=384 ATTN_MODE=graphormer DECODER_MODE=coarse_xattn`（2026-05-22 起默认
  coarse_xattn）`W_POS/W_ROT/W_VEL/W_CONTACT/W_KL=1/1/1/0.1/1e-3`。

### 1.3 可视化 QA（跨项目铁律：视觉 > metric）
```bash
cd /scratch/ts1v23/workspace/noKslot_clean
source /scratch/ts1v23/.conda/etc/profile.d/conda.sh && conda activate graph_salad
python scripts/animate_anytop13.py \
  --ckpt runs/<run名>/best_recon_model.pt \
  --out runs/<run名>/qa --species Alligator,Spider,Trex,Dragon --n_per 2 --device cpu
```
产出 `<物种>_clipN_gtvspred.gif`（GT-vs-pred 动画）+ `_sheet_{obl,top}.png`（6 帧蒙太奇）。**必须人眼看 gif**。

### 1.4 文本 caption 预计算（use_text 训练前一次性）
```bash
srun --jobid=<活alloc> --overlap --ntasks=1 bash -c \
  'source /scratch/ts1v23/.conda/etc/profile.d/conda.sh && conda activate graph_salad && \
   cd /scratch/ts1v23/workspace/noKslot_clean && \
   python scripts/precompute_t5_captions.py --out data/anytop_caption_t5.npz'
```
T5-base 冻结 encoder → `data/anytop_caption_t5.npz`（885/1070 有 caption）。

### 1.5 Smoke 测试（任何改码后，必经；走 srun 不在 login node）
```bash
srun --jobid=<活alloc> --overlap --ntasks=1 bash -c \
  'source /scratch/ts1v23/.conda/etc/profile.d/conda.sh && conda activate graph_salad && \
   cd /scratch/ts1v23/workspace/noKslot_clean && \
   python scripts/train_graph_vae.py --smoke --dataset anytop_truebones \
     --feat_mode anytop13 --attn_mode graphormer --out runs/_smoke --overwrite --device cuda'
```
`--smoke` = 5 iter 跑通验证；改 fk6 路径要同时 `--dataset unified --feat_mode fk6` 回归。

### 1.6 关键绝对路径
| 角色 | 路径 |
|------|------|
| 工作目录 | `/scratch/ts1v23/workspace/noKslot_clean` |
| 数据（本地副本，1070 motion / 70 物种） | `data/anytop_truebones/` |
| caption 缓存 | `data/anytop_caption_t5.npz` |
| 训练产出 | `runs/<run名>/`（`train.log`、`best_recon_model.pt`、`args` 存在 ckpt 内） |
| conda | `/scratch/ts1v23/.conda` env `graph_salad` |
| 监控指纹 | `.aris/meta/.last_monitor_status` / `monitor_heartbeat.log` |

---

## §2 harness 流程（我们怎么干活）

一次改动 / 一次训练的标准流水线：

```
改码 → smoke (srun, 不在 login node) → codex 审 (gpt-5.5 xhigh, 里程碑用 fresh thread)
  → 按 NEEDS-FIX 修 → 重审至 PASS
  → 部署训练 (deploy 脚本, PPID=1 setsid srun --overlap 进已有 alloc)
  → 监控 (/loop 每小时 lean-read: tail log / grep val,GATE,Traceback / squeue / nvidia-smi)
  → 可视化 QA (animate, 人眼看 gif — 视觉 > metric)
  → 里程碑 codex 综合审
```

**铁律（不可降级，未来 session 必须遵守）：**
1. **alloc 永远不主动 cancel，也不主动申请 —— 申请和取消只由用户执行。** 我可以
   kill 自己的 python/srun 进程，但**绝不 `scancel`**，绝不 srun 进 / 碰别项目的 job。
2. **每一处代码新增/改动必经 codex 审**（gpt-5.5、`model_reasoning_effort=xhigh`、
   不传 `sandbox`；里程碑审用 fresh thread 避免 round-1 verdict 偏置；fix 复审可 codex-reply）。
   MCP 断了 fallback `codex exec --model gpt-5.5 --config model_reasoning_effort=xhigh`。
3. **CV 任务可视化 demo 准确度 > metric** —— 不能只看 loss/speed_ratio，必须渲染
   GT-vs-pred 动画人眼检查。视觉与 metric 冲突时以视觉为准。
4. **不抢别项目正在用的卡** —— 起训练前 `nvidia-smi` + `squeue -w <node>` 核验空闲。
5. **长任务持久化靠 PPID=1**：`ssh <node> "setsid nohup ..."` 让进程被 init 收养；
   login-node 的 `nohup &` 和 Agent subagent 都会在 ~1.5h 内死。
6. **Karpathy R1 优先**：动手前显式列假设；有歧义/多解读/更简做法 → 先停下问用户，
   不因"是 on-contract follow-up"就直接 fire。

**监控（/loop）**：CronCreate 起的 /loop 是 session 内的，session 关了就停；多小时 /
需跨 session 的监控要用 §2 铁律 5 的 PPID=1 durable monitor。停止条件达成时主动 tear down
（CronDelete），不留空转基建。

**交接文档命名**：放进 `handoff/` 的文档名 = `<时间戳>_<内容后缀>.md`，时间戳在前
（产出时刻），后缀按内容定义。阅读顺序进 `handoff/README.md` 索引。

---

## §3 失败的经验教训（M1.5R → M1.7 一路踩的坑）

按"坑 → 教训"列。这些是返工成本最高的，未来别再踩。

### 模型 / 评估类
1. **frozen-pred 退化（M1.5R 反复栽）** —— 模型塌成一个静态 pose，recon loss
   看着还行但动作是死的。→ 催生了 `speed_ratio` 指标（逐帧位移 pred/gt 比值，
   <0.1=🥶FROZEN）**和**"可视化 > metric"这条跨项目铁律。**教训：低 recon loss
   ≠ 动作活的，永远要渲染人眼看。**
2. **metric-乐观 / false-PASS** —— 曾经 bone_cos 0.92 但可视化是一团乱；codex 靠
   读 live data 才抓到。**教训：数字好 ≠ 对；视觉/数据可推翻 metric，也可推翻一个
   review verdict。单静帧 ≠ 运动对（frozen/抖动/塌缩静帧看不出）。**
3. **gif 的 Read 只能看到首帧** —— Claude 读 gif 只看到 frame 0。A/B 视觉对比时
   只能靠 contact sheet（6 帧蒙太奇）凑合。**教训：时序失败模式（时间块状）需要
   连续帧渲染才能验，6 帧稀疏采样验不出 —— 这个 QA 缺口至今未补。**

### 部署 / 基建类
4. **ssh 不传递 env 变量** —— deploy 脚本在 `ssh` 之前 export 的变量到不了远端
   shell。**修复：env 变量必须写进 ssh 命令字符串内部。**
5. **deploy LOG/flock 用 POOL_TYPE 做 key** —— 两个同 pool 的 run 撞 log + flock。
   **修复：用 `RUN_TAG = basename(OUT)` 做 key。**
6. **srun `--gres` 与 CUDA pin 冲突** —— 设了 `CUDA_VISIBLE_DEVICES_OVERRIDE` 时
   `--gres` 会冲突。**修复：override 设了就 drop `--gres`。**
7. **monitor 进程死** —— login-node `nohup &` 和 Agent subagent ~1.5h 内死。
   **修复：`ssh <node> "setsid nohup bash monitor.sh ..."` → PPID=1 被 init 收养。**
8. **session context 溢出 → harness 挂起** —— 一个长 codex-review-loop 弧把整段
   调试读进主线程（~20 个 codex 全文 ≈ brief 的 26 倍），10.5MB transcript →
   harness 静默挂起。**教训：主线程读 codex 的 `*_brief_*.txt` 不读全文；大文档
   不重复读；弧太长时主动 rotate 到新 session，别等挂。**

### 数据 / 语义类（codex 抓出来的）
9. **RIFKE 语义误读（iter-1 codex NEEDS-FIX）** —— AnyTop 13ch 的 root 通道是
   RIFKE 状态不是纯位置；最初按纯位置处理。**教训：先把数据集语义搞对，再写
   loss —— 我们的 loss 不兼容就改我们的去适配 AnyTop 原生。**
10. **hash() split 非确定** —— 用 `hash()` 做 train/val split，受 PYTHONHASHSEED
    影响跨进程不一致。**修复：改 md5-稳定 split。**
11. **contact BCE target 用错** —— 差点拿归一化后的 `anytop_x[...,12]` 当 BCE
    target；必须用 **raw `foot_contact_per_joint`（0/1）**。静默 bug 陷阱。
12. **AnyTopGraphAttentionBlock 缩放顺序**（iter-2 Task4 codex NEEDS-FIX）——
    先把 qk 缩放再加 topo/edge bias；AnyTop 是对**求和后的 score** 缩放。
    **修复：先 `qk + topo_bias + edge_bias` 再 `× 1/sqrt(d_head)`。**
13. **ckpt warm-start 断裂** —— MotionDecoder 按 `motion_feat_dim=13` 建会让
    output_proj 维度对不上旧 ckpt。**修复：output_proj 固定建在 dim 6，过滤
    obsolete key。**
14. **T5Tokenizer 缺 sentencepiece** —— `T5Tokenizer` 需要 sentencepiece 依赖。
    **修复：换 `T5TokenizerFast`。**
15. **QA 工具读不到 anytop_mean/std** —— 它们是反归一化统计量，挂在 raw collate
    dict，不在 typed `GraphMotionBatch` 上。**教训：de-norm 统计量走 raw dict。**

### decoder A/B（2026-05-22 本轮结论）
16. **decoder graph-temporal gap** —— `unpool_identity` decoder 退化成 per-joint
    identity 自精修。A/B 实验（1000ep×2）结论：`coarse_xattn`（零参数修复）val
    小赚 ~1.4%、位移更贴 GT → 已设默认；但 baseline 也无可见 artifact，doc
    `docs/decoder_graphtemporal_gap.md` 的"选项 B（decoder 加图层）"无视觉证据
    justify → 暂不上。**教训：架构弱点要 A/B + 可视化证伪，别凭直觉大改。**

---

## 文件清单
| 角色 | 文件 |
|------|------|
| 启动 | `scripts/_deploy_train_anytop13.sh` |
| 训练入口 | `scripts/train_graph_vae.py` |
| 数据 | `src/data/anytop_dataset.py` + `data/anytop_truebones/` |
| 模型 | `src/models/graph_salad/vae.py`、`src/models/encoder.py`、`src/models/motion_decoder.py` |
| loss | `src/models/graph_salad/losses.py` |
| 可视化 QA | `scripts/animate_anytop13.py` |
| 文本预计算 | `scripts/precompute_t5_captions.py` |
| 代码逐步导览 | `docs/anytop13_training_walkthrough.md` |
| decoder 架构记录 | `docs/decoder_graphtemporal_gap.md` |
