# v4b-272-neutral 合并数据集 Graph-VQVAE — A100 桥接训练中 + 待迁 4×H200

**状态 (2026-07-03 06:03Z UPDATE):** 用户改主意 —— **先在 swarma1004 4×A100 起(已起,健康运行),之后新 dual_h200 到位就 RESUME 到 4×H200(优先 H200)。每小时检查新卡+剩余资源。**
- **A100 桥接跑 LIVE**: swarma1004(1041117, 4×A100-80GB, ~3d), 单节点 standalone, global64(16×1×4)/lr6.65e-5/n8192/两阶段curriculum3.0@0→4.5@50/bf16, **SAVE_EVERY=5**(~ep5 首存以便尽早无损迁移), OUT=runs/vqvae_v4b272neutral_C96_J144_d512_Q4_n8192_b16g64_300ep_curric50to60_seed42, orchestrator PPID=1, **无 watchdog**(靠每小时检查兜底). launcher=scripts/_launch_graph_vqvae.sh. codex-verified via workflow(A100→H200 resume 无缝/无 OOM/recipe 一致, 3 路对抗验全未 refute).
- 待迁触发同下(新 dual_h200 + blossom03). 迁移用 **RESUME(OVERWRITE=0)** 不是 fresh launch —— 见文末「MIGRATION」节.

---
## (历史) 原「等 dual_h200 fresh 起」方案 — 现改为 A100 桥接先起
数据全就绪。flamingo01(1077424 dual_h200)已于 04:03Z TIMEOUT 到期。

## 为什么在等
- flamingo01 (1077424, dual_h200) 已于 **2026-07-03 04:03Z TIMEOUT 到期**(walltime 2-12:00 耗尽)。
- 现只剩 **blossom03** (1123158, quad_h200, gpu:2, 剩 ~1d5h) —— worker 半边。
- 需要一个新的 **dual_h200**(partition=dual_h200, 节点 flamingo0[12], gpu:2, 我的 alloc)当 master。
- swarma1004 有 4×A100 全空(~3d),但用户明确不选 A100、也不选 blossom-2卡。

## 数据 / 前置(全部 DONE)
- 合并集: `data/animo4d_L4TB_plus_human_v4b272neutral` — 382 objects(381 animal + HML3D_Human)、102438 motions、splits train 97288 / val 5150。coverage-OK、codex-PASS。
- 人体新集: `data/humanml3d_anytop13_v4b_272_neutral` — 26846 clips,真 twist(272 SMPL local rot6d),readback 0.0001mm PASS,全 captioned。
- GT twist QA 已发用户并**确认无误**(v4b 真 twist vs v3a 零 twist)。
- launcher/watchdog 均已 codex-PASS(既有 infra,未改代码)。IB/partition 已核:H200 用 `ib1`/`mlx5_1`,同 /22 子网。

## 触发条件(dual_h200 一到即执行)
1. 新 alloc: partition=`dual_h200`,节点 `flamingo0[12]`,`gres/gpu:2`,我的、GPU 空闲、ssh 可进(pam_slurm_adopt 放行=alloc active)。
2. blossom03 (`1123158`) 仍在(worker)。
把 `<MNODE>`/`<MJOB>`/`<MIP>` 填成新 dual_h200 的节点名/jobid/ib1-IP(`ssh <MNODE> "ip -o -4 addr show ib1"`)。

## 步骤 A — SMOKE(必做,验 rendezvous + NCCL via IB + WORLD_SIZE=4)
```bash
ssh <MNODE> "cd /scratch/ts1v23/workspace/noKslot_clean && \
  JOB_A=<MJOB> JOB_B=1123158 MASTER_NODE=<MNODE> WORKER_NODE=blossom03 \
  RDZV_HOST=<MIP> NCCL_SOCKET_IFNAME=ib1 NCCL_IB_HCA=mlx5_1 \
  SMOKE=1 NCCL_DEBUG=INFO OVERWRITE=1 OUT=/tmp/vqvae_v4b_smoke \
  ANYTOP_ROOT=data/animo4d_L4TB_plus_human_v4b272neutral MAX_JOINTS=144 MAX_COARSE=96 \
  NUM_CODES=8192 BATCH_SIZE=16 \
  bash scripts/_launch_graph_vqvae_2node_h200.sh 2>&1 | tee scripts/_smoke_vqvae_v4b.log"
```
PASS 判据:日志出现 `WORLD_SIZE=4`、`via NET/IB/0`、4 iter 无 error/无 SIGABRT。
(smoke 也顺带预暖 `data/.../\_cond_normalized_J144.pkl` 缓存,真跑首启不再冷扫描。)

## 步骤 B — REAL(durable, setsid nohup 在 master, PPID=1)
```bash
ssh <MNODE> "cd /scratch/ts1v23/workspace/noKslot_clean && \
  JOB_A=<MJOB> JOB_B=1123158 MASTER_NODE=<MNODE> WORKER_NODE=blossom03 \
  RDZV_HOST=<MIP> NCCL_SOCKET_IFNAME=ib1 NCCL_IB_HCA=mlx5_1 \
  ANYTOP_ROOT=data/animo4d_L4TB_plus_human_v4b272neutral MAX_JOINTS=144 MAX_COARSE=96 \
  NUM_CODES=8192 BATCH_SIZE=16 LR=6.65e-5 EPOCHS=300 SEED=42 AMP_DTYPE=bf16 \
  HUMAN_UPSAMPLE_FACTOR=3.0 HUMAN_UPSAMPLE_START_EPOCH=0 \
  HUMAN_UPSAMPLE_PHASE2_FACTOR=4.5 HUMAN_UPSAMPLE_PHASE2_START_EPOCH=50 \
  OVERWRITE=1 OUT=runs/vqvae_v4b272neutral_C96_J144_d512_Q4_n8192_b16g64_300ep_curric50to60_seed42 \
  setsid nohup bash scripts/_launch_graph_vqvae_2node_h200.sh > scripts/_train_vqvae_v4b.log 2>&1 </dev/null &"
```
global = 4×16 = 64,lr 6.65e-5(与 n2048 一致),300ep,bf16。curriculum: phase1 factor3.0@ep0(~50% human),phase2 factor4.5@ep50(~60% human)。
OUT 名不含 `L4safeHuman`(过 watchdog guard);含 `curric`+factor>1+start≥0(过 curriculum guard)。

## 步骤 C — WATCHDOG(durable auto-resume, setsid nohup 在 swarmh1002 稳定节点)
```bash
ssh swarmh1002 "cd /scratch/ts1v23/workspace/noKslot_clean && \
  OUT_REL=runs/vqvae_v4b272neutral_C96_J144_d512_Q4_n8192_b16g64_300ep_curric50to60_seed42 \
  ANYTOP_ROOT=data/animo4d_L4TB_plus_human_v4b272neutral MAX_COARSE=96 MAX_JOINTS=144 \
  NUM_CODES=8192 BATCH_SIZE=16 LR=6.65e-5 \
  HUMAN_UPSAMPLE_FACTOR=3.0 HUMAN_UPSAMPLE_START_EPOCH=0 \
  HUMAN_UPSAMPLE_PHASE2_FACTOR=4.5 HUMAN_UPSAMPLE_PHASE2_START_EPOCH=50 \
  setsid nohup bash scripts/_watchdog_h200_vqvae.sh > .aris/meta/watchdog_h200_vqvae.boot.log 2>&1 </dev/null &"
```
watchdog `discover_h200` 自动认 dual_h200(flamingo0[12]) + quad_h200(blossom0[1-4]) 各 gpu:2;DOWN 2 连测且 GPU 空闲且有 `last_model.pt` → resume。**必须先停任何别的 H200 watchdog**(backbone watchdog 用不同 lock,会抢同样的 H200 alloc)。启动前核 `pgrep -u ts1v23 -f '[_]watchdog_h200'` 应为空。

## 起后核验
- `ssh <MNODE> "pgrep -u ts1v23 -fc '[t]rain_graph_vqvae'"` ≥1;blossom03 同样 ≥1(两节点各有 train proc = healthy)。
- orchestrator PPID=1: `.aris/meta/.vqvae_h200_orch.pid` 存在且进程活。
- watchdog PPID=1: `ps -p <wd_pid> -o ppid=` → 1。
- 监控 `runs/.../train.log` 的 iter/loss/codebook active-dead;QA gif 每 QA_EVERY=100 出。
- **CV 铁律:训练后必渲染 recon GT-vs-pred gif 人眼/发用户审,不只看 metric。**

---
## MIGRATION — A100 桥接 → 4×H200 RESUME(新 dual_h200 到位时执行)
**与上面「fresh 起」的关键区别:是 RESUME 不是 fresh —— `OVERWRITE=0` + `RESUME_CKPT=last_model.pt`,绝不 OVERWRITE=1(会抹掉 A100 已训进度)。** OUT 目录不变(同一个 runs/vqvae_v4b272neutral_...,在 /iridisfs 共享 fs,H200 节点可读)。recipe 与 A100 完全一致(global64/lr6.65e-5/n8192/curriculum 3.0@0/4.5@50/bf16)→ 续接无缝(workflow 对抗验已确认:epoch 续 ckpt.epoch+1 不重置、LR/curriculum-phase 正确、ckpt 设备无关、per-rank seed 跨 A100↔H200 一致)。

**触发:** 新 dual_h200(partition=dual_h200, 节点 flamingo0[12], gpu:2, 我的, 空闲, ssh 可进) + blossom03(1123158 quad gpu:2)仍在。

**步骤 0 — 前置:** 确认 A100 上已有 `runs/vqvae_v4b272neutral_.../last_model.pt`(SAVE_EVERY=5 → ~ep5 后有)。**没有就等它到 ep5 再迁,别丢进度**(或用户要 H200 ASAP 则接受从头,罕见)。

**步骤 1 — 优雅停 A100(用户已授权此迁移;绝不 scancel):**
```bash
ssh swarma1004 "pkill -u ts1v23 -f '[t]rain_graph_vqvae.*vqvae_v4b272neutral' ; pkill -u ts1v23 -f '[_]launch_graph_vqvae.sh'"
# 核验:ssh swarma1004 "pgrep -u ts1v23 -fc '[t]rain_graph_vqvae'" == 0 且 nvidia-smi util 4 卡全 0。alloc 1041117 保留(不 scancel)。
```

**步骤 2 — H200 SMOKE(必做,同 fresh 的步骤A,但可跳过——因 A100 已证 recipe;仍建议 4-iter 验跨节点 rendezvous+IB)。填 MNODE/MJOB/MIP=新 dual_h200 节点名/jobid/ib1-IP。**

**步骤 3 — H200 RESUME 真跑(durable, setsid nohup 在 master, PPID=1):** 同文首「步骤 B」的命令,但把 `OVERWRITE=1` 改成 **`OVERWRITE=0 RESUME_CKPT=last_model.pt`**,其余(OUT/ANYTOP_ROOT/MAX_COARSE=96/curriculum/lr/n8192/bf16)不变。日志应显示从 A100 的 epoch 续上(非 ep0)。

**步骤 4 — H200 WATCHDOG(同文首「步骤 C」,起前 `pgrep '[_]watchdog_h200'` 必空)。** watchdog 的 discover=dual_h200(flamingo0[12])+quad_h200(blossom0[1-4]);之后 H200 断线它会自动 resume。

**步骤 5 — 核验 + 报 user:** 两节点各有 train proc、orch/watchdog PPID=1、train.log 从正确 epoch 续、loss 连续(无跳变)。
