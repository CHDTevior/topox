# Session 交接 — graph_pscf 训练启动 + 双训练监控

> 产出 2026-06-09 ~22:45 BST。给**下一个对话**接手。本 session 把 graph_pscf 从设计审 → 实现 → 全长 cache → profile → launcher → smoke → 启动正式训练全部跑完,现在 2 个训练并行跑。
> **绝对路径根**: `/scratch/ts1v23/workspace/noKslot_clean`(下文 `REPO`)。conda python: `/scratch/ts1v23/.conda/bin/python3`。

---

## 0. ⚠️ 最重要:监控 cron 是 session-only,本对话退出即死 → 下个对话必须重建

当前有 2 个 in-memory cron(本 session 退出就没了):
- `90b29c7c` — graph_pscf 训练监控(每 23min)
- `2c9512b6` — animo4d VAE 训练监控(每 2h)

**下个对话开场第一件事: 用 CronCreate 重建这两个监控**(prompt 模板在 §6)。否则两个训练无人盯。

---

## 1. 现在正在跑的(2 个训练)

### A. graph_pscf backbone(本 session 刚启动,主线)
- **run dir**: `REPO/runs/codeflow_graph_pscf_L5_b16_lr1p2e4_seed42`
- **orchestrator log**: `REPO/scripts/_train_gpscf_6card.log`(所有 rank stdout,stdbuf line-buffered 实时)
- **6×H100 cross-alloc DDP**: swarmh1002 的 3 个 swarm_h100 alloc(974142+974141+944462,各 2×H100),world_size 6,static rendezvous over IB(swarmh1002-ib0:29505)。
- **配置(user 拍板锁定)**: 287M graph_pscf,batch16/GPU × 6 = **global 96**,**lr 1.2e-4**,600 epoch,warmup 2000,half_cosine,dropout 0.05(自动),cond_drop 0.1,全长 cache T_lat=75,empirical norm 全扫。
- **当前状态**(22:43Z): orchestrator PPID=1 durable,9 train procs,在 **empirical 全扫 startup**(empirical_stats_max_clips=0 扫 70792×6rank,~10min)。**还没出第一个 [ep0 it0]** —— 第一个 epoch 含 empirical 全扫会慢(~10-15min),之后稳定 ~6.4min/epoch。
- **ETA**: ~3.5 天(600 epoch)。
- **4 信号监控**(user 重点,前 1-2 epoch 尤其): flow_loss↓ / grad_norm 不爆 / proj_err 不爆 / code_usage/q 高覆盖。smoke 验证基线: flow_loss~2.0, grad_norm~0.09, proj_err 7-8, code_usage [429-476]/512。

### B. animo4d L2 VAE(背景,接近尾声)
- **run dir**: `REPO/runs/m1_animo4dL2_proxfiltered_bf16_rot6dfk_C128_d512_h8_ff1536_300ep_seed42`
- **8×A100 cross-NODE DDP**: swarma1004(944457)+swarma1001(944458),各 4×A100。orchestrator on swarma1004。
- **当前**: ep200/300,loss 0.41 健康,speed_ratio 0.97,ETA ep300 ~15h(明天午后)。
- ep109/ep171 各有过一次 transient spike 都已完全恢复(非塌缩)。

---

## 2. 已经做完的(graph_pscf 完整链,本 session)

| 阶段 | 产物 | 验证 |
|---|---|---|
| 设计审 | 原方案 + workflow 4角度审 + codex `019eaa81`/`019ead02` | SOUND-WITH-CHANGES,B1 Floyd 硬伤,4 决策 |
| **实现 M0-M4** | `src/models/CodeFlow_Model/{dit_blocks,graph_pscf,flow}.py` + `scripts/train_graph_codeflow.py` | 三轮 codex(`019ead5e`/`019ead75`/`019ead96`),GPU smoke 8步 @ 287.19M |
| 审核者审 → 3 findings | 全长 export override / Gate-2 geo+adj / max_T_lat preflight | 全响应 |
| **全长 token cache** | `data/codeflow_tokens_cleanL5_ep280_fulllen300_par`(train70792+val3730,T_lat=75) | 12 卡并行 ~33min,抽查 PASS |
| 并行 export infra | `scripts/export_graph_vq_tokens.py`(--shard_idx/--num_shards)+`merge_export_shards.py`+`_run_export_parallel.sh` | codex `019eadd3`,smoke |
| profile | batch16=64GB(batch20=80GB满),0.52s/iter | mem_profile + throughput |
| **6×H100 DDP launcher** | `scripts/_launch_graph_pscf_6card.sh`+`_launch_graph_pscf.sh` | codex `019eae3f` PASS |
| 6-card smoke → DDP fix | flow.py `forward==flow_loss` + train:480 `flow(...)` | codex `019eae3f` PASS,smoke rc=0+world6+4信号健康 |
| **启动训练** | run dir 上方 | orchestrator PPID=1,world_size 6 |

**完整文档链**(给审核/下个对话): `handoff/20260609_2120_graph_pscf_final_review.md`(最终审核,真实行号+287M+启动脚本+超参) + `..._1650_..._executor_spec.md`(4决策+必改+defaults) + `..._1625_..._plan_review_verdict.md` + `..._graph_codeflow_pscf_double_single_impl_plan.md`(原方案)。

**关键代码行号**(当前 working tree,未提交): graph_pscf.py(SlotTemporal 66/re-mask108,122; Coupling 128非图/zero-init161,170; FlowNet223/frame_seed287/output-zero319/fwd336/cond429/double460/single493/v_pred512)。flow.py(GraphCodeFlow42/forward(=flow_loss)/dropout-resolve78/selector85/flow_loss168). train(main192/args198-262/maxTlat-preflight333/dropout-resolve365/construct373/DDP-wrap429/train-loop flow(...)480).

---

## 3. 将要干的(下一步,按顺序)

1. **盯 graph_pscf 前 1-2 epoch 的 4 信号**(empirical scan 完后出 [ep0])。健康(flow_loss 缓降/grad 不爆/proj_err 不爆/code_usage 高)→ 让它跑;异常 → fail-loud 报 user。
2. graph_pscf ep600 训完(或早期有意义 ckpt)→ **continuous-vs-snapped 视觉 QA**(CV 铁律): `scripts/animate_graph_codeflow.py`,**必带 `--model_variant`(从 ckpt 读)+ `--num_frames 300`**(否则 graph_pscf fail-loud 拒绝默认 64-frame),单 gif T2M 布局(静态骨架+prompt+pred,去GT栏),slow/fast/long-chain/high-branch 物种,**SendUserFile 发 user 审**(视觉裁决权归 user)。
3. VAE ep300 训完 → 渲 GT-vs-recon GIF(10 PZ 物种)发 user 审 → 自删 VAE cron。
4. (未提交代码: 全部 graph_pscf + VQVAE + vq_model 仍是 working-tree `??`/`M`,**commit 只在 user 明确要求时做**。)

---

## 4. 如何执行命令(可复现追踪,绝对路径)

### graph_pscf 监控(只读)
```bash
# 4 信号(orchestrator log,实时):
ssh swarmh1002 'grep -E "allocA.*\[ep" /scratch/ts1v23/workspace/noKslot_clean/scripts/_train_gpscf_6card.log | tail -8'
# orchestrator 活 + PPID:
ssh swarmh1002 'ps -o ppid= -p $(pgrep -f "[_]launch_graph_pscf_6card"|head -1); pgrep -f "[t]rain_graph_codeflow"|wc -l'
# ckpt:
ls -lt /scratch/ts1v23/workspace/noKslot_clean/runs/codeflow_graph_pscf_L5_b16_lr1p2e4_seed42/*.pt
```

### graph_pscf 崩溃后 resume(若需,user 判断后)
```bash
# durable,从 last_model.pt 全量 resume(model+optimizer+epoch),OVERWRITE=0 原地续:
ssh swarmh1002 "cd /scratch/ts1v23/workspace/noKslot_clean && \
  OUT=runs/codeflow_graph_pscf_L5_b16_lr1p2e4_seed42 OVERWRITE=0 \
  RESUME_CKPT=runs/codeflow_graph_pscf_L5_b16_lr1p2e4_seed42/last_model.pt \
  setsid nohup bash scripts/_launch_graph_pscf_6card.sh > scripts/_train_gpscf_6card.log 2>&1 </dev/null &"
```

### graph_pscf 重新启动(若 run dir 要重来,OVERWRITE=1 覆盖)
```bash
ssh swarmh1002 "cd /scratch/ts1v23/workspace/noKslot_clean && \
  OUT=runs/codeflow_graph_pscf_L5_b16_lr1p2e4_seed42 \
  setsid nohup bash scripts/_launch_graph_pscf_6card.sh > scripts/_train_gpscf_6card.log 2>&1 </dev/null &"
# (默认 SMOKE=0/EMPIRICAL_MAX=0/batch16/lr1.2e-4/600ep/全长cache/ep280 frozen)
```

### graph_pscf 停训(换配置/停)
```bash
# 括号防自匹配 + 多节点:先杀 orchestrator + srun + train,核验全 0
ssh swarmh1002 'pkill -f "[_]launch_graph_pscf_6card"; pkill -f "[s]run --overlap --jobid"; pkill -f "[t]rain_graph_codeflow"; sleep 3; pgrep -f "[t]rain_graph_codeflow"|wc -l'
```

### VAE 监控/resume: 见 cron `2c9512b6` prompt(§6)。launcher: `scripts/_launch_animo4dL2_vae_8card_xnode.sh`。

---

## 5. harness 流程(下个对话要懂的)

- **登录节点经 iridisfs 读热写大文件会卡 harness**(项目记忆 `project_iridisfs_onnode_fastpath`)。所以: 所有 nvidia-smi/tail/进程查 **走 ssh 计算节点本地**(秒回);主线只读共享盘状态文件。计算节点**不出网**(codex/git 走登录节点)。
- **durable 训练/export = compute node `setsid nohup ... </dev/null &`**(PPID=1 init-adopted,survive ssh 断)。登录节点 nohup + subagent 都会 ~1.5h 内死。
- **cross-alloc DDP 8 条**(CLAUDE.md,已验证): static rendezvous(非 c10d)+ NODE_RANK + NCCL P2P/SHM disable + IB(NCCL_SOCKET_IFNAME=ib0)+ srun --overlap --jobid --gres + setsid durable + flock 单实例 + rank-0 ckpt。
- **cron**: session-only(退出即死),durable=false。监控 tick fire 时是 fresh agent context,prompt 要自包含(绝对路径+判据+停止条件)。
- **codex 审**: 每代码改必经(铁律),`mcp__codex__codex` model gpt-5.5 + config model_reasoning_effort xhigh,**不传 sandbox**,milestone 用 fresh thread。MCP 断 → fallback `codex exec --model gpt-5.5 --config model_reasoning_effort=xhigh`。
- **不能 self-submit/cancel Slurm**;**不抢别项目正在用的卡**(VAE 的 swarma1004/1001 别碰;启动前 nvidia-smi+squeue 核验空闲)。
- **CV 任务可视化 demo 优先于 metric**,QA 默认 SendUserFile 发 user 审(视觉裁决权归 user)。
- **verify before claim**(报数前自查本轮真实回显,不编造);**fail-loud 不静默跳步**。

---

## 6. 监控 cron 重建模板(下个对话开场执行)

**graph_pscf cron**(每 23min,`7,30,53 * * * *`):
> [自动 tick — 监控 graph_pscf 正式训练(287M,6×H100 cross-alloc DDP,global96/lr1.2e-4/600ep,全长T_lat=75)]。ORCH_LOG=/scratch/ts1v23/workspace/noKslot_clean/scripts/_train_gpscf_6card.log,RUN=/scratch/ts1v23/workspace/noKslot_clean/runs/codeflow_graph_pscf_L5_b16_lr1p2e4_seed42,NODE=swarmh1002。① ssh $NODE 核 orchestrator PPID=1 + train procs≥6 + ORCH_LOG mtime<20min。② **4 信号健康早筛**(前1-2ep尤其): grep "allocA.*\[ep" → flow_loss 稳降(初~2.0)/grad_norm 不暴涨(~0.09-0.2,>5警示)/proj_err 不爆(初7-8,不发散100+)/code_usage/q 高覆盖(初[429-476]/512,塌个位=RVQ退化)。任一异常→PushNotification 报 user+log证据。③ 崩溃(orch没了/procs=0/log>20min不动)→ 查 log 尾找 Traceback/NCCL/OOM/NaN,PushNotification 报 user,**不自行重启**(首训,resume 由 user 判断,命令见 handoff §4)。④ 里程碑: ep1-2 首次健康报一次"正常起跑";每ep50记一笔;ep600→continuous-vs-snapped 视觉QA(animate_graph_codeflow.py --model_variant graph_pscf --num_frames 300,发user审)→自删cron。⑤ fingerprint 原子写 .aris/meta/.last_monitor_status_gpscf。铁律: 不抢卡/不self-submit-Slurm/不编造/fail-loud。

**VAE cron**(每2h,`13 */2 * * *`): 完整 prompt 在本 session 的 cron `2c9512b6`(也在更早 handoff)。要点: RUN=...m1_animo4dL2...,核 PPID=1 + log<15min,只在 loss反升/NaN/active_C塌个位/speed_ratio<0.3 才报(spike 在降是正常),崩溃→ssh swarma1004 重起 `_launch_animo4dL2_vae_8card_xnode.sh`(带--resume last_model.pt,括号防自匹配),ep300→渲GT-vs-recon GIF发user审→自删。fingerprint .aris/meta/.last_monitor_status_animo4dvae(当前 ep200/loss0.41/speed0.97)。

---

## 7. 失败的经验教训(本 session 踩过的坑,给下个对话避雷)

1. **cross-alloc DDP 必先 smoke**: graph_pscf 训练 train loop 调 `flow.flow_loss`,DDP-wrap 后无此属性(`AttributeError`)——**单卡 + 三轮 codex 都没暴露**,只有 6-card smoke 暴露。修: flow.py 加 `forward==flow_loss` + train 用 `flow(...)`。教训: DDP-only bug 必须真 6-rank smoke 才现形,别跳过。
2. **cross-alloc 各 alloc 的 CPU 配额不齐**: `--cpus-per-task=16` 超过 flamingo/blossom 的 8 CPU → srun 永久 retry("Requested nodes are busy")→ orchestrator 卡死。教训: cross-alloc 前 `scontrol show job <j>|grep cpu` 核各 alloc CPU,`--cpus-per-task` 取 **min**(用了 6)。
3. **并行 agent 写代码要对齐契约 + verify 真跑**: 12-shard 并行 export 的 3 个 agent 各自假设了不同契约(index_shard `03d` vs 无填充,缺 manifest_shard sidecar,inner bare `wait` 吞失败)→ 3 个 blocking bug。verify agent + codex 抓到。教训: 并行写的接口契约要主线统一 + verify 实跑(smoke 2-shard+merge 验证)。
4. **`ls *.npz` 在 70k 文件上 ARG_MAX 静默返回 0**: 差点误判数据丢失。教训: 大目录数文件用 `find ... -name '*.npz'|wc -l`,不用 `ls *.npz|wc -l`。
5. **at-init probe 要先 de-zero**: `_smoke_graph_codeflow_textpositive.py` 报"text route DEAD"(Δv=0)是假阳性——zero-init output_proj 让 v_pred 恒 0,与 text 无关。正确 probe(`_textpos.py`)先 de-zero 27 个 zero-init 张量。教训: 测 at-init 模型的 conditioning 参与,先扰动 zero-init projections。
6. **日志 artifact ≠ 真问题**: VQVAE ckpt `codebook_active=[512,0,0,0]` 是 train 脚本从带 quantizer-dropout 的 last train batch 存的假象,RVQ 实际健康。教训: 见可疑数字先查它来自哪段代码(verify before claim),别直接当塌缩报警。
7. **残留的旧帧长假设**: spec 一度写 T_lat=16(64-frame 窗口残留),user 发现要全长(caption=全长语义)。VQVAE 帧长无关已 QA(64帧训练重建 T_lat=74 OK)。教训: 帧长/窗口假设跨阶段会残留,正式训练前核 cache 实际 T_lat。
8. **VAE transient spike 别误报**: ep109(0.59→8.70)、ep171(0.47→1.19)都是单 epoch transient,几 epoch 内恢复,非塌缩(speed_ratio 健康/active_C 稳/无 NaN)。教训: loss spike 在降 + speed_ratio 健康 = 正常,只在趋势反升/NaN/塌缩才报。
9. **ssh 内用相对路径 + CWD**: ssh 命令里 cd 在子 shell,`tail relative.log` 可能空。用绝对路径。
10. **PushNotification "terminal has focus" = 没发出**: 不代表失败,in-session 报告仍可见。
11. **empirical stats 全扫在 DDP+全长下极慢(启动卡 ~20-30min)**: `compute_empirical_stats` 扫全 train cache 算 z_q mean/std,但 DDP 下**每个 rank 各扫一遍** 70792 npz,全长 z_q 又比 64-frame 大 4.7×(~58GB/rank × 6 rank = ~350GB IO)→ 首启动 empirical scan 卡 20-30min(GPU util 0%、log 停在 preflight 不动 = 正常,不是崩)。**优化建议(下个对话/resume 前可做,codex 审)**: 让 `compute_empirical_stats` 只 rank-0 扫 + `dist.broadcast` mean/std 给其它 rank(6× 加速 → ~5min);或 resume 时若 ckpt 已存 latent_mean/std 则跳过重扫。当前首训接受慢启动(只一次)。判活: GPU mem>0(model loaded)+ procs 活 + 无 Error = 正常扫,等它出 [ep0]。

---

## 8. 一句话给下个对话
**先重建 2 个监控 cron(§6) → 盯 graph_pscf 第一个 epoch 的 4 信号(empirical scan 完后出)→ 健康就让它跑 ~3.5天 → ep600 渲视觉 QA 发 user 审。VAE ep300(~明天午后)渲 recon QA 发 user。任何代码改必经 codex(gpt-5.5 xhigh)。不抢 VAE 的卡,不 self-submit Slurm。**
