# v4b-272 CodeFlow Backbone — READY TO LAUNCH (staged, awaiting user "go")

**状态 (2026-07-05 12:24Z): LAUNCHED — user 显式发令("现在就起")后已执行步骤 A+B。** 与 staged 的唯一差异:步骤 A
实际执行时补了 `GEN_EVAL=1 EVALUATOR_CKPT=<v4b evaluator>`(与步骤 B/下文 gen-eval 注释的意图一致,staged A 漏写)。
起跑核验全绿:orch PPID=1、world_size=4、empirical 缓存命中(无冷扫描)、两阶段 curriculum ON(3.0@0→4.5@50)、
gate ok z_q=[16,75,96,512]、ep0 flow_loss 1.90、两节点 100% util/126GB。watchdog swarmh1002 pid 305274 PPID=1
(CHECK_SEC=300)。flamingo01(1123160)~20:05Z 到期 → watchdog auto-resume,需 user 补新 dual_h200。
监控 cron b844893b(每小时 :18)。

**原 staged 状态 (2026-07-05 06:52Z):** backbone prep 全部完成 + 验证。**不自动起真跑,等 user 发令。**

## 决策(user 已定 2026-07-05)
- **模型**: `graph_pscf` 正式生成骨干(~287M params, DiT-style graph rectified-flow on frozen VQVAE 的 post-RVQ z_q)
- **人体 curriculum**: 跟 VQVAE 一样的**两阶段** — phase1 factor 3.0 @ep0 (~52% human) → phase2 factor 4.5 @ep50 (~62% human)

## Prep 完成清单(全验证)
- 冻结 VQVAE: `runs/vqvae_v4b272neutral_C96_J144_d512_Q4_n8192_b16g64_300ep_curric50to60_seed42/best_model.pt`(ep219, best_val 0.864, d512/C96/J144/K8192/Q4/stride4)
- 模型代码 arch-agnostic(零改动接受 v4b)
- caption 缓存 `data/anytop_caption_t5_v4b272neutral_multi.*`(coverage 1.0)
- **token 缓存** `data/codeflow_tokens_v4b272neutral_n8192_ep219_fulllen300`(142G, train 97288 + val 5150, coverage 1.0, RVQ-identity fp32 err 3.8e-6, T_lat=75)
- **empirical_stats.pt**(count 66,219,731 / D 512, 预暖好, smoke 证明缓存命中→无冷扫描 SIGABRT)
- **两阶段 curriculum 代码** (train_graph_codeflow.py) — codex-PASS
- **launcher phase2 转发** (_launch_graph_pscf.sh + _launch_graph_pscf_2node_h200.sh) — codex-PASS + smoke 证明 args 到达
- **watchdog v4b 适配** (_watchdog_h200_backbone.sh: 默认→v4b / guard→n8192 / phase2 转发 / EMPIRICAL_MAX=0) — codex 审中
- **SMOKE ×2**(裸 + 带 curriculum): WORLD_SIZE=4 / 287M params / gate ok z_q=[16,75,96,512] / 投影 QA cont+snap finite / flow_loss 有限无 NaN / rc_A=rc_B=0

## 资源
4×H200 = flamingo01(1123160 master) + blossom01(1123159 worker), IB RDZV_HOST=10.6.15.127 ib1/mlx5_1

---
## 步骤 A — REAL 起训(durable, setsid nohup 在 master, PPID=1)
```bash
ssh flamingo01 "cd /scratch/ts1v23/workspace/noKslot_clean && \
  JOB_A=1123160 JOB_B=1123159 MASTER_NODE=flamingo01 WORKER_NODE=blossom01 \
  RDZV_HOST=10.6.15.127 NCCL_SOCKET_IFNAME=ib1 NCCL_IB_HCA=mlx5_1 \
  BATCH_SIZE=16 LR=8e-5 EPOCHS=600 WARMUP_STEPS=2000 EMPIRICAL_MAX=0 OVERWRITE=1 \
  FROZEN_CKPT=runs/vqvae_v4b272neutral_C96_J144_d512_Q4_n8192_b16g64_300ep_curric50to60_seed42/best_model.pt \
  TOKEN_CACHE=data/codeflow_tokens_v4b272neutral_n8192_ep219_fulllen300 \
  HUMAN_UPSAMPLE_FACTOR=3.0 HUMAN_UPSAMPLE_START_EPOCH=0 \
  HUMAN_UPSAMPLE_PHASE2_FACTOR=4.5 HUMAN_UPSAMPLE_PHASE2_START_EPOCH=50 \
  OUT=runs/codeflow_graph_pscf_v4b272neutral_n8192_b16g64_lr8e5_4xh200_seed42 \
  setsid nohup bash scripts/_launch_graph_pscf_2node_h200.sh > scripts/_train_gpscf_v4b.log 2>&1 </dev/null &"
```
global batch = 4×16 = 64, lr 8e-5 (Goyal for global64), 600 epochs, warmup 2000, empirical-norm full-set(缓存命中), 两阶段 curriculum。

## 步骤 B — WATCHDOG(durable auto-resume, setsid nohup 在 swarmh1002)
```bash
ssh swarmh1002 "cd /scratch/ts1v23/workspace/noKslot_clean && \
  OUT_REL=runs/codeflow_graph_pscf_v4b272neutral_n8192_b16g64_lr8e5_4xh200_seed42 \
  FROZEN_CKPT=runs/vqvae_v4b272neutral_C96_J144_d512_Q4_n8192_b16g64_300ep_curric50to60_seed42/best_model.pt \
  TOKEN_CACHE=data/codeflow_tokens_v4b272neutral_n8192_ep219_fulllen300 \
  BATCH_SIZE=16 LR=8e-5 \
  HUMAN_UPSAMPLE_FACTOR=3.0 HUMAN_UPSAMPLE_START_EPOCH=0 \
  HUMAN_UPSAMPLE_PHASE2_FACTOR=4.5 HUMAN_UPSAMPLE_PHASE2_START_EPOCH=50 \
  GEN_EVAL=1 EVALUATOR_CKPT=runs/anytop_t2m_evaluator_distilbert_coemb512_gb128_lr1e-4_mfd12_v4b272_seed42/best_model.pt \
  setsid nohup bash scripts/_watchdog_h200_backbone.sh > .aris/meta/watchdog_h200_backbone.boot.log 2>&1 </dev/null &"
```
（起 watchdog 前 `pgrep -f '[w]atchdog_h200_backbone'` 应为空; 与 VQVAE watchdog 不同 lock, 但 VQVAE 已停所以无冲突。GEN_EVAL 用 v4b evaluator, root 匹配 → online R-precision/FID 每 50 epoch。）

## 起后核验
两节点各有 train_graph_codeflow proc + orch/watchdog PPID=1 + train.log flow_loss 下降 + [human-upsample] curriculum ON 两阶段 log + ep50 phase2 切换。

---
## FALLBACK — flamingo01 到期且无替补 dual_h200 时的 2×H200 bridge（user 授权 2026-07-05 "别的空闲卡都可以用,尽量保证原来的学习效率"）
**先等 dual_h200**（watchdog 自动 resume 4×H200，配置完全不变=首选）。等不到（"检查一段时间"后仍无）→ 用 blossom01 剩下的 2×H200 + **梯度累积 2** 续跑：`bs16 × 2gpu × accum2 = global 64, lr 8e-5` = **学习效率一模一样**（只 ~2× 慢）。grad-accum 已加进 trainer（`--grad_accum`，默认 1 与原版字节等价）+ launcher（`GRAD_ACCUM`）— codex-PASS + smoke-PASS（`[ep0 it1 n_iter=0]` 证明每 2 micro-batch 才 step；scheduler `steps_per_epoch=len(dl)//accum` 使 LR 曲线与 4gpu-accum1 完全一致）。**4×H100 出局**（bs16 需 ~78GB，单 H100 80GB OOM）。

**执行步骤（仅当无 dual_h200）**：
1. 停当前 H200 watchdog（它只认 4×H200 拓扑，会与 2 卡 job 抢 blossom01）：`ssh swarmh1002 "fuser -k .aris/meta/<watchdog lockfile>"`（按名 pkill 会误杀，用 lockfile fuser；见 project_h200_watchdog_autoresume 记忆）。确认 `pgrep -f '[w]atchdog_h200_backbone'` 为空。
2. 确认 flamingo01 已死、blossom01 idle（`pgrep train_graph_codeflow`=0）。
3. 起 2×H200 standalone bridge（durable, PPID=1, RESUME from last_model.pt）：
```bash
ssh blossom01 "cd /scratch/ts1v23/workspace/noKslot_clean && \
  NNODES=1 NPROC_PER_NODE=2 CVD=0,1 \
  BATCH_SIZE=16 LR=8e-5 GRAD_ACCUM=2 EPOCHS=600 WARMUP_STEPS=2000 EMPIRICAL_MAX=0 \
  RESUME_CKPT=last_model.pt OVERWRITE=0 \
  FROZEN_CKPT=runs/vqvae_v4b272neutral_C96_J144_d512_Q4_n8192_b16g64_300ep_curric50to60_seed42/best_model.pt \
  TOKEN_CACHE=data/codeflow_tokens_v4b272neutral_n8192_ep219_fulllen300 \
  HUMAN_UPSAMPLE_FACTOR=3.0 HUMAN_UPSAMPLE_START_EPOCH=0 \
  HUMAN_UPSAMPLE_PHASE2_FACTOR=4.5 HUMAN_UPSAMPLE_PHASE2_START_EPOCH=50 \
  GEN_EVAL=1 EVALUATOR_CKPT=runs/anytop_t2m_evaluator_distilbert_coemb512_gb128_lr1e-4_mfd12_v4b272_seed42/best_model.pt \
  OUT=runs/codeflow_graph_pscf_v4b272neutral_n8192_b16g64_lr8e5_4xh200_seed42 \
  setsid nohup bash scripts/_launch_graph_pscf.sh > scripts/_train_gpscf_v4b_2gpu_bridge.log 2>&1 </dev/null &"
```
   核验：log 里 `global=64(=16x1x2x accum2)`、`RESUME start_epoch=<ep15+>`、flow_loss 续上不跳、PPID=1。（bridge 无 watchdog，靠 cron 盯，像当初 A100 VQVAE bridge。）
4. **迁回 4×H200**：一旦出现 dual_h200 → 停 2 卡 bridge（bracket-pkill '[t]rain_graph_codeflow' on blossom01，非 scancel）→ 重起 watchdog（步骤 B），watchdog 会从 last_model.pt resume 4×H200（accum=1 默认，回到全速）。
