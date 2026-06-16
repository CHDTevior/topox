# Handoff — graph_pscf 训练监控 + commit 7c68441 + startup-opt（2026-06-09 23:43 BST）

> 接力对象：下一个对话。**你的主职 = 独立监控两个正在跑的训练，按里程碑/异常汇报 user，ep600 训完做视觉 QA。**
> 本文档自洽（不打开别的文件也能开始监控）。深度细节见文末「延伸阅读」。
> 上一份全量交接：`handoff/20260609_2245_session_handoff_graph_pscf_training.md`（11 条失败教训全量 + harness 流程）。

---

## 0. 一句话状态

graph_pscf 正式训练 **健康**（ep4，flow_loss 0.94 一路从 2.08 降，4 信号全绿），代码已 **commit 7c68441 到 main**，empirical-stats **startup 加速已做完 + codex PASS**。
**⚠ 你接手后第一件事：监控 cron 会随上个 session 退出而死（session-only），你必须自己重建监控**（见 §3）。

---

## 1. 本次会话做了什么（相对 2245 交接的 delta）

### 1.1 startup 加速（empirical z_q 归一化扫描）—— 已做 + codex PASS
- **问题**：训练启动时要扫全量 train token cache（70792 个全长 npz）算 z_q 的 mean/std，6-rank 全长扫 ~30min，纯 CPU 解压瓶颈。
- **做法**（`scripts/train_graph_codeflow.py` 的 `compute_empirical_stats` ~L116-185 + 调用点 ~L427-446）：
  1. **disk-cache**：把 mean/std 存到 `<token_cache>/empirical_stats.pt`，下次启动若 cache_key 匹配 → **瞬载（~0.7ms）跳过整轮扫描**。
  2. **cache_key = 内容指纹**：`manifest.json 的 md5` + `train/index.jsonl 的全文 md5（index_md5，不是 byte-size）` + `n/D/max_clips`。re-export / clip-set / 顺序 / 内容任何变化 → key 变 → 自动失效重扫。（24MB index 的 md5 一次性 rank-0 仅 34.5ms。）
  3. **DDP rank-0-only**：只有 rank 0 扫/载/写 cache，再 `dist.broadcast` mean/std/count 给所有 rank。避免每个 rank 重复解压全量 cache（这是 30min 的真正来源）。
  4. **坏 cache self-heal**：load 路径的 shape 检查用**显式 `raise`（非 assert，`python -O` 安全）**；坏 shape 被外层 `try/except` 接住 → `loaded=False` → 走完整 re-scan + 重写（自愈，不卡 3.5 天的训练）。
- **验证**：冷扫→暖载 byte-identical（mean/std/count 一致）；同 byte-size 不同内容 → 重扫（正是 byte-size 抓不到的 case）；codex **PASS**（fresh thread `019eae7d`，gpt-5.5 xhigh）。
- **⚠ 只在下次 resume/restart/新实验生效**，不碰当前在跑的训练（empirical stats 只在 startup 算一次，之后不再读 `empirical_stats.pt`）。
- codex 非阻塞 caveat：key 不 hash 每个 .npz payload，故「index/manifest 不变但原地改 .npz」不会失效——正常 export/merge 流程不会发生，acceptable。

### 1.2 commit 7c68441（graph_pscf 自洽包 → main）
- **范围（user 拍板「graph_pscf 自洽包」）= 28 文件**：
  - `src/models/CodeFlow_Model/`（graph_pscf / flow / dit_blocks / token_dataset / graph_codeflow / __init__）
  - `src/models/vq_model/`（graph_pscf 运行时加载的冻结 tokenizer：tokenizer / quantizer / masked_motion_decoder / losses / utils / __init__）
  - scripts：train_graph_codeflow / train_graph_vqvae + 跨 alloc DDP 启动器（_launch_graph_pscf{,_6card}、_launch_graph_vqvae{,_6card}）+ 分片导出（export_graph_vq_tokens / merge_export_shards / _run_export_parallel / _run_codeflow_token_export_full）+ animate_graph_codeflow + 3 个 smoke
  - **3 个被 import 的已跟踪修改**：`graph_salad/attention.py`（加 `use_graph_bias` 消融开关）、`graph_salad/denoiser.py`、`data/anytop_dataset.py` —— 必须带上，否则 commit 不自洽（committed 代码 import 未提交的改动）。
- **故意没提交**（仍在工作树里，96 项）：40 个 handoff 文档、几十个一次性诊断/渲染脚本（`_measure_/_render_/_diag_/_plot_/_check_/_sanity_/_t2m_`）、无关的 diffusion/VAE 工作（`train_denoiser.py`、`animate*.py`、`_launch_diffusion*`、`train_graph_vae.py`、`REPO_AUDIT.md` 等 8 个 modified 文件）。
- **`.gitignore` 已保护** `data/ runs/ logs/ .aris/ __pycache__/ *.pyc *.log`，所以 token cache / ckpt / 日志 / pyc 都不会进库。
- 预提交校验：所有 staged `.py` py_compile OK，所有 `.sh` `bash -n` OK。
- **没 push**（user 没要求 push，规则：commit/push 只在 user 明确要求时）。`src/data/anytop_dataset.py` 因 `.gitignore` 的 `data/` 模式需 `git add -f`（它已是 tracked，force 只是绕过重加警告）。

---

## 2. 正在跑的两个训练（绝对路径）

### 2.1 graph_pscf 正式训练（**你的主监控对象**）
| 项 | 值 |
|---|---|
| run dir | `/scratch/ts1v23/workspace/noKslot_clean/runs/codeflow_graph_pscf_L5_b16_lr1p2e4_seed42` |
| orchestrator 日志（**实时、监控看这个**） | `/scratch/ts1v23/workspace/noKslot_clean/scripts/_train_gpscf_6card.log` |
| 节点 / 卡 | swarmh1002，3 alloc cross-alloc DDP：**974142 + 974141 + 944462**，world_size **6**（6×H100） |
| 启动器 | 外 `scripts/_launch_graph_pscf_6card.sh` + 内 `scripts/_launch_graph_pscf.sh`（套 CLAUDE.md cross-alloc 8 条） |
| 配置 | batch16/GPU × 6 = **global 96**，**lr 1.2e-4**，600ep，warmup2000，half_cosine，dropout0.05，cond_drop0.1 |
| token cache | `data/codeflow_tokens_cleanL5_ep280_fulllen300_par`（**全长 T_lat=75**，train 70792 / val 3730） |
| 冻结 tokenizer | `runs/vqvae_L5_C50_d512_Q4_n512_b96_300ep_seed42/best_model.pt`（ep280，val 0.945） |
| 最新（23:43 BST） | `[ep4 it252 n_iter=3200] flow_loss=0.944 grad_norm=0.165 lr=1.2e-4 \| proj_err=3.51 code_usage/q=[497,507,495,474]` |
| ETA | ~3.5 天（~22:38 BST 6-09 启动 → 约 6-13） |
| orchestrator PPID | **1**（setsid，durable，ssh 断不死；活到 Slurm alloc 过期为止） |

### 2.2 animo4dL2 BF16 Graph-VAE（**次要监控，别碰它的卡**）
| 项 | 值 |
|---|---|
| run dir | `runs/m1_animo4dL2_proxfiltered_bf16_rot6dfk_C128_d512_h8_ff1536_300ep_seed42` |
| 日志 | `<run dir>/train.log` |
| 节点 / 卡 | **swarma1004**，4 卡，bs48/lr4e-4/global192/300ep |
| 最新（23:43 BST） | `[ep206 ...] loss~0.43 active_C~70 rowsum=[1,1]` 健康 |
| ETA | ep300 约 6-10 午后 → 训完做 recon QA |
| **⚠ 铁律** | **不抢 swarma1004 的卡**；graph_pscf 用的是 swarmh1002，互不干涉 |

---

## 3. ⚠ 监控必须由你（下一会话）重建 —— cron session-only 退出即死

上个 session 的监控 cron（`90b29c7c` graph_pscf / `2c9512b6` VAE）**会随该 session 退出而消失**。`CronList` 在你的新 session 里大概率是空的。**你接手后必须自己重新建立监控。**

### 3.1 立刻手动核活（node-local ssh 快路径——登录节点直读 iridisfs 热文件会卡 harness）
```bash
# 活性：PPID 应=1，train procs 应 >=6
ssh swarmh1002 'ps -o ppid= -p $(pgrep -f "[_]launch_graph_pscf_6card"|head -1); pgrep -f "[t]rain_graph_codeflow"|wc -l; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader|head -8'
# 4 信号最新几行（本地直读 orchestrator log，它 stdbuf line-buffered 实时）：
grep -E "\[ep[0-9]+ it" /scratch/ts1v23/workspace/noKslot_clean/scripts/_train_gpscf_6card.log | tail -6
# 报错扫描：
grep -iE "traceback|error|nan|oom|nccl error" /scratch/ts1v23/workspace/noKslot_clean/scripts/_train_gpscf_6card.log | tail
```

### 3.2 重建自动监控（推荐 `/loop`，或 `CronCreate`）—— 直接复用下面这份 brief
间隔建议 ~20-30min（progressive backoff：健康就静默，只在里程碑/异常/崩溃汇报）。把以下 brief 作为监控 prompt：

> **[监控 graph_pscf 正式训练 287M 6×H100 cross-alloc DDP]** 只在实质变化/里程碑/崩溃时汇报，否则静默。
> ORCH_LOG=`/scratch/ts1v23/workspace/noKslot_clean/scripts/_train_gpscf_6card.log`，RUN=`.../runs/codeflow_graph_pscf_L5_b16_lr1p2e4_seed42`，NODE=swarmh1002。
> 1. 核活：ssh NODE 查 orchestrator PPID=1 + `pgrep -f "[t]rain_graph_codeflow"|wc -l`>=6 + GPU util；本地 `stat -c %y $ORCH_LOG`（<20min 在动=活）。
> 2. **4 信号健康早筛**（见 §4）：从 ORCH_LOG grep 最近几行。任一异常（flow_loss 反升/grad 暴涨>5/NaN/proj_err 爆到 100+/code_usage 塌到个位数）→ PushNotification 报 user + 给 log 证据。
> 3. **崩溃 → fail-loud 不自动 resume**：orchestrator 没了 / train procs=0 / log >20min 不动 → 查 ORCH_LOG 尾 + Traceback/NCCL/OOM/NaN → PushNotification 报 user（哪个 rank + 什么错）。**不自行重启**（首训，崩溃可能是 bug 非 transient，由 user 判断）。
> 4. 里程碑：每 ep50 记一笔；ep600 训完 → 渲 continuous-vs-snapped 视觉 QA（见 §6）发 user 审。
> 5. fingerprint 原子写 `.aris/meta/.last_monitor_status_gpscf`：ISO \| graph_pscf \| epXXX/600 \| 信号 \| 6card-PPID1 \| ALIVE/CRASHED \| unchanged=N。
> 6. 停止条件：ep600 训完且 QA 发出 → 汇报后自删监控；或 user 重新分配卡。
> 铁律：不抢别项目卡（VAE swarma1004 别动）；不 self-submit/cancel Slurm；报数前自查真实回显不编造；fail-loud 不静默跳步。

VAE（swarma1004）监控可同理另起一份（间隔可松到 2h），brief 见 2245 交接 / 旧 cron `2c9512b6` 文案。

---

## 4. 4 健康信号 + 阈值（user 拍板的核心 gate）
| 信号 | init 基线 | 健康 | 异常（→报 user） |
|---|---|---|---|
| **flow_loss** | ~2.0 | 缓降（现 ep4=0.94） | 反升 / 震荡发散 / NaN |
| **grad_norm** | ~0.09-0.2 | 稳（现 0.11-0.29） | 暴涨 >5 / NaN |
| **proj_err** = mse(z_hat, z_snap) | 7-8 | 不发散（现 3.5，在降） | 爆到 100+ |
| **code_usage/q**（RVQ 4 stage 各用了多少 code，满 512） | [429-477] | 保持高覆盖（现 [497,507,495,474]） | 塌到个位数 = RVQ 退化 |
- QA decode 行 `cont_finite/snap_finite` 应恒 True。`cont_vs_snap_maxabs` 是连续-vs-snap 的最大偏差（监控趋势即可，不是 gate）。

---

## 5. 可复现命令 + 绝对路径

```bash
# —— 查状态 —— （node-local 快路径，见 §3.1）

# —— 崩溃后 resume（仅 user 批准后）：RESUME_CKPT 指向 last_model.pt，重跑外层启动器 ——
ssh swarmh1002 "cd /scratch/ts1v23/workspace/noKslot_clean && setsid nohup \
  OUT=runs/codeflow_graph_pscf_L5_b16_lr1p2e4_seed42 \
  RESUME_CKPT=runs/codeflow_graph_pscf_L5_b16_lr1p2e4_seed42/last_model.pt \
  bash scripts/_launch_graph_pscf_6card.sh > scripts/_train_gpscf_6card.log 2>&1 < /dev/null &"
# 注意：alloc 974142/974141/944462 须仍存活；flock .aris/meta/.gpscf6card.lock 防双启动；
#       startup 优化已生效 → resume 时 empirical stats 会从 empirical_stats.pt 瞬载。

# —— ep600 训完视觉 QA（CV 铁律，见 §6）——
#   必须 --num_frames 300（全长！animate_graph_codeflow.py 对 graph_pscf 无 --num_frames 会 fail-loud 拒跑，防误用 64 帧）
python scripts/animate_graph_codeflow.py --ckpt runs/codeflow_graph_pscf_L5_b16_lr1p2e4_seed42/best_model.pt \
  --num_frames 300 --frozen_ckpt runs/vqvae_L5_C50_d512_Q4_n512_b96_300ep_seed42/best_model.pt ...
```

---

## 6. 里程碑 / 停止条件
- **ep50 每档**：记一笔（健康则静默或简报）。
- **ep600 训完**（约 6-13）→ **continuous-vs-snapped 视觉 QA**（CV 铁律，**优先于 metric**）：
  - 渲 continuous decode + snapped decode 两路；T2M 单 gif 布局（静态输入骨架 + prompt 文本 + pred 动画，**去 GT 栏**——T2M 推理无 GT）。
  - 覆盖 slow / fast / long-chain / high-branch 物种（最能暴露失败模式）。
  - **`SendUserFile` 发 user 审**（视觉裁决权归 user，自己别先下结论）。
  - 失败类型树（见项目记忆 `project-graph-codeflow-direction`）：flow 不降/continuous 也差→backbone/conditioning；continuous 好/snapped 差→RVQ projection；snapped 好/视觉差→tokenizer decoder 或 数据/文本。
- **停止条件**：ep600 训完 + QA 发 user 后 → 汇报、拆掉监控（自删 cron/loop）。或 user 重新分配卡。

---

## 7. 铁律 / 约束（不可降级）
- **不 self-submit / 不 self-cancel Slurm**（job 由 user 管）。
- **不抢别项目正在用的卡**：VAE 在 swarma1004，graph_pscf 在 swarmh1002，互不碰；若发现 graph_pscf 被别 job `--overlap` 抢（util 骤降/吞吐掉）→ 立即报 user + 给让卡 prompt，自己不 scancel。
- **代码新增/改 必经 codex 审**（gpt-5.5 xhigh，milestone 用 fresh thread，**不传 sandbox 参数**；MCP 断开 fallback `codex exec --model gpt-5.5 --config model_reasoning_effort=xhigh`）。
- **CV 可视化优先于 metric**；QA 默认 `SendUserFile` 发 user 审，不自己先看图下结论。
- **崩溃 fail-loud 不自动 resume**（首训谨慎）；报数前自查本轮真实回显，不编造/不误报；不静默跳步。
- **Karpathy 第 1 条（Think Before Coding）凌驾自主决策**：codex PASS 的 follow-up 若有歧义/多解读/更简做法 → 先停下问 user。
- commit/push 只在 user 明确要求时。

---

## 8. 关键失败教训（carry-forward；全量 11 条见 2245 交接）
1. **cross-alloc DDP 必先 smoke**：6-card smoke 暴露了「train loop 调 `flow.flow_loss` 但 DDP-wrapped 无此属性」的崩溃——单卡 + 三轮 codex 都没抓到。修法：`flow.py` 加 `forward==flow_loss`，train loop 用 `flow(...)`。
2. **node-local ssh 快路径**：登录节点经 iridisfs 读热文件/大文件会卡 harness；ssh 计算节点本地 `tail`/`nvidia-smi`/`grep` 秒回。计算节点不出网（codex/git 走登录节点）。
3. **`ls *.npz` ARG_MAX**：7 万+ 文件时 `ls *.npz` 静默返回 0；用 `find … | wc -l`。
4. **cron/loop session-only**：退出即死（见 §3，本次交接核心）。
5. **monitor 持久性靠 PPID**：durable monitor 用 `ssh <node> "setsid nohup … </dev/null &"`（PPID=1），不是登录节点 nohup（~1.5h 死）。
6. **cross-alloc CPU 配额坑**：各 alloc CPU 数不齐 → `--cpus-per-task` 取 min（曾因 16>8 导致 srun 永久 retry）。

---

## 9. 延伸阅读（深度细节）
- `handoff/20260609_2245_session_handoff_graph_pscf_training.md` —— 全量交接（11 失败教训 + harness 流程 + cron 重建模板）。
- `handoff/20260609_1840_graph_pscf_training_walkthrough.md` —— 逐模块真实行号 + 287M 参数分解 + 启动脚本 + 超参。
- `handoff/20260609_2120_graph_pscf_final_review.md` —— 启动前最终审核。
- 项目记忆 `project-graph-codeflow-direction` —— Phase-1 LOCKED 配方 + 4 决策 + 失败类型树 + 本次 startup 优化记录。
- commit `7c68441`（main）—— graph_pscf 自洽包。

---

## 10. ⚠ ep200 中途 QA（user 2026-06-10 要求）+ 渲染管线更新

**触发**：训练跨过 ep200（约 6-11 上午，~4500 iter/h）→ 哨兵 cron 自动跑：
```bash
bash scripts/_render_gpscf_qa_ep200_20260610.sh   # 在空闲 GPU 节点（blossom03/rose09/flamingo01），绝不碰训练卡
```
- 固化了 **val 8 + train 8 = 16 个多样物种**（小/中/大关节 × 慢/快 × 猫科/灵长/爬行/有蹄/巨型；val 含 Giant Anteater J=62 拓扑极值）。
- 输出 `runs/.../qa_ep200_val8` + `qa_ep200_train8`，16 个 gif 全部 `SendUserFile` 发 user 审，自己不下结论。

**渲染管线本次更新（已 codex 双轮 PASS）**：
- `animate_graph_codeflow.py` 改用 `make_t2m_large_gif`（PIL 大渲染）+ **GT 红色最右栏**（布局 `input|PRED snapped|PRED continuous|GT红`）。
- **按 GT 真实长度生成**（`T_lat_i=ceil(item["num_frames"]/stride)`，`--num_frames 300` 仅作容器上限）；val 全部 clip 都 <300（19-299），所以这个改动对全部样本生效。
- GT 渲染忠实性已验证：与数据集导出参照渲染数值 bit-identical（同 `_recover_world_positions`，右手系 y-up 面朝 z+）。
- 一次性自检脚本 `scripts/_check_gt_render_fidelity_20260610.py`。
- **教训**：speed_ratio 必须等长比较——之前按 300 帧统一生成会把短 GT（如狮子 25 帧）稀释成假"冻结"（假 ratio 0.117）；等长后真值 2.0，无塌缩。

**质量评判铁律（user 2026-06-10）**：无 benchmark，loss/val 只作健康信号看趋势；生成质量唯一 gate = QA 渲染；绝不主动 early stop，跑满 ep600。

**cron 教训**：SSH 断线重建监控时若 session 未真正重启，会产生重复 cron（本次踩过：4 个 cron 双触发导致 tick 堆叠）。重建前先 CronList 查重，重复则 CronDelete 清理。当前干净 cron：哨兵 851009cc(:09/:29/:49) + 汇报 a5480ba4(:21)。
