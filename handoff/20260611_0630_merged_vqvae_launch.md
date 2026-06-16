# 合并数据集 Graph-VQVAE 训练启动 (2026-06-11 06:30 BST)

## 做了什么
在合并数据集 `data/animo4d_anytop_clean_L4_safe_plus_truebones`（75592 motions, 71784/3808 train/val, 381 object types, 最大骨架 Dragon J=142）上启动了一个新的 Graph-VQVAE 300ep 训练。这是 [[project-merge-datasets-direction]] 合并方向的**第一步**（先训 tokenizer）。

## 配置（相对 L5 VQVAE 基线的 delta）
L5 基线（runs/vqvae_L5_C50_d512_Q4_n512_b96_300ep_seed42/train.log 实测）: world_size=2, batch_size=96/GPU → global=192, lr=2e-4, bf16, 其余全 argparse 默认。

新 run 的改动**只有数据/形状/批量**，arch+loss 全部不变（= argparse 默认 = L5 recipe）:
- `--anytop_root data/animo4d_anytop_clean_L4_safe_plus_truebones`
- `--max_joints 144`（容纳 Dragon J=142；merged 已预建 `_cond_normalized_J144.pkl`）
- `--max_coarse 96`（user 定：完整保 Dragon 的 89 段，caps nothing）
- `--max_frames 64`（窗口不变 = L5）
- `--batch_size 32`/GPU × 4 卡 = **global 128**；`--lr 1.33e-4`（Goyal: 2e-4×128/192）；warmup 0
- Q=4 / num_codes=512 / code_dim=512 / d_model512 / d_ff1536 等全部不变；300ep; seed42; bf16

### 为什么 B=32 而不是 B=48
- B=48（=exact L5 global 192）smoke 峰值显存 **72.9GB/80GB = 89%**（且 4s 采样可能低估真峰）→ 多天无人值守跑 OOM 风险高。
- VQVAE 是 **launch-bound**（~157 items/s 跨 batch 全平，大 batch 无吞吐增益，见 [[project-vqvae-throughput-launchbound]]）→ 没有理由追大 batch。
- B=32 smoke 峰值 **49.4GB/80GB = 60%**（1s 密采样，可靠），余量充足；Goyal 缩 lr 后与 L5 每样本梯度动态等价。
- 故选 B=32：安全 + 动态等价 + 零吞吐损失。

## 资源 / 启动方式
- node=swarma1004, alloc=974143 (4×A100-SXM4-80GB, EndTime 2026-06-16 05:58 ≈5天, 全空闲且是 user 指定/我方 alloc)。
- 单节点 standalone torchrun（NNODES=1 NPROC_PER_NODE=4 CVD=0,1,2,3）；非 cross-alloc，无需 IB/NCCL-P2P-disable。
- ssh swarma1004 已被 pam_slurm_adopt 纳入 974143 的 cgroup（SLURM_JOB_ID 自动设, 4 卡可见）→ 直接 `ssh swarma1004 "setsid nohup env ... bash scripts/_launch_graph_vqvae.sh > scripts/_train_vqvae_4card_a100.log 2>&1 </dev/null &"`。进程 PID=726509 PPID=1 durable（已验证）。
- OUT=`runs/vqvae_L4safeTB_C96_J144_d512_Q4_n512_b32_300ep_seed42`。控制台日志 `scripts/_train_vqvae_4card_a100.log`（node-local）。

## 代码改动（唯一）+ 审查
`scripts/_launch_graph_vqvae.sh`: 加 `MAX_JOINTS`/`MAX_COARSE`/`MAX_FRAMES` env→arg 透传，默认 64/50/64（向后兼容，6card orchestrator 等现有调用者行为不变）。
**codex gpt-5.5 xhigh fresh thread (019eb517) VERDICT: PASS** — 且 codex 自己在内存里跑了 Dragon 单样本 J144/C96/T64 smoke（z_q=(1,16,96,D), valid coarse 89, pred=(1,64,144,13), loss finite）确认 J=144/C=96 路径端到端可用；确认 EdgeSegmentPool(96) 覆盖 Dragon 89 无溢出、rank-0 ckpt guard、EMA collectives all-rank、Goyal 公式正确。**未 commit（user 未要求）。**

## 启动健康（已验证）
- ep0 it200 n_iter=200, total recon 11.4→4.87 下降, grad_norm 25.9（早期正常）, lr=1.33e-4。
- 4 个 RVQ quantizer 全活: ppl=[246,356,424,247] active=[336,459,487,342] dead=[0,0,0,137]（dead 下降中，无塌缩）。
- 4 卡均 49.4GB, util 20-38%（launch-bound 波动，正常）。
- 预计 ~14 min/epoch（粗估）→ 300ep ≈ 2.5-3 天，落在 974143 的 5 天 alloc 内。

## 监控（session-only crons，本会话重建）
旧的 851009cc/a5480ba4 已删（fingerprint stale=未可靠触发）。新建覆盖**两个**训练:
- **哨兵 3a397870**（:13/:33/:53）: 静默健康哨兵；node-local ssh 读两训练 train.log+util；写 `.aris/meta/.last_monitor_status_gpscf` + `.last_monitor_status_vqvae`；里程碑（graph_pscf ep200 QA / ep600 / VQVAE ep300）+ 异常（traceback/oom/nan/util0/alloc<12h/抢卡）才发声。
- **每小时汇报 415eb0b7**（:41）: 两训练具体数字 + SLURM 队列，必发 user。
两 cron session-only，会话死则失效；7 天自动过期。

## 待办 / 风险
- VQVAE: 跑到合适里程碑（如 ep100）后做 **recon 可视化 QA**（CV 任务可视化优先于 metric），SendUserFile 发 user 审。当前仅 metric 健康信号。
- graph_pscf alloc **944462 于 2026-06-12 11:50 到期**（≈29h）→ 6 卡 DDP 丢 2 rank 会整体崩，~ep373。resume 方案见 §5 of 20260609_2343 handoff（RESUME_CKPT=last_model.pt + empirical stats 秒载 + flock guard）。哨兵 <12h 预警。
- 合并方向后续：tokenizer 训好后，flow backbone 也要在并集上重训才能真正补 unseen 拓扑泛化（见 [[project-merge-datasets-direction]]）。

## 8-card cross-node A100 续训方案（user 2026-06-11，就绪待触发）
user 指示：若之后又申请到一个 4 卡 A100 节点 → 升 8 卡 A100 VQVAE，**直接 --resume 续训 + LR 重预热**；目标 = 训练稳定 + 吞吐。

**关键事实（analysis+对抗验证 workflow wf_ac6dfc1c-8e0 确认）**：
- swarma1004 是 **4-GPU 节点** → 第二个 alloc 必在**另一物理节点** → 真·**cross-NODE** 2 节点 DDP（非 6card 的 same-node cross-cgroup）。
- 当前 4 卡 = **host/launch-bound**（median util ~51%，4 卡同步 25%↔100% 振荡；瓶颈 = 每 step ~13 个 quantizer/finite-gate 的 `.item()` GPU→CPU 同步，**per-rank 非 rank0 串行** → 跨节点可并行）。75.8 items/s, ~947s/epoch。
- **8 卡跨节点预测吞吐 1.7-1.9×**（对抗 agent 无法反驳；per-GPU batch 不变=32，global128→256 减半 iters/epoch；跨节点 IB comm <2% of 1.69s step，IB 已验证 200Gb/s HDR mlx5_0 active）。300ep ~79h → **~42-46h**。
- **硬前提：save_every=10 → 第一个 last_model.pt 在 ep9 末（~08:50 BST）才落盘**，之前无法 --resume。切换最早 ep10 后；最好在某次 save 刚落盘后切（少丢 epoch）。

**resume + LR 重预热（Goyal）**：global 128→256 (k=2) → lr 1.33e-4→**2.66e-4**；`--warmup_steps` 键于 steps-since-launch（train_graph_vqvae.py:680-684）→ --resume 从 0 重预热；用 **warmup 500**（L5 precedent 用 300 应对 3× 跳变，这里 2× 用 500 更保守，stability 优先）。OUT 必须 == ckpt 父目录（同 run dir）+ OVERWRITE=0（resume_in_place 守卫 :410-411）。dataset-shape env (MAX_JOINTS=144/MAX_COARSE=96/MAX_FRAMES=64+ANYTOP_ROOT=merged) 必须显式传否则 strict load 崩。

**orchestrator**：`scripts/_launch_graph_vqvae_8card_crossnode_a100.sh`（6card 的 re-param 拷贝，**未动 6card 文件**）。NNODES=2 NPROC_PER_NODE=4，node0=swarma1004(974143,master)/node1=新alloc；RDZV_HOST=swarma1004-ib0(10.6.15.68) PORT=29505。**cross-node NCCL：显式 NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=0**（覆盖 inner 的 :-1，保 intra-node NVLink）**+ NCCL_IB_HCA=mlx5_0**（ib1 DOWN，pin mlx5_0）+ NCCL_SOCKET_IFNAME=ib0 NCCL_IB_DISABLE=0。flock .vqvae8card.lock。inner launcher **无需改**。**codex 019eb568 PASS**（NEEDS-FIX 仅 NCCL_IB_HCA=mlx5_0，已加；对照 repo 内已验证的 _launch_token_diffusion_8card_a100.sh 同款 cross-node NCCL）。

**切换序（待 user 确认时机）**：(1) 确认 last_model.pt 存在(ep≥10)；(2) 确认新节点 4 卡空闲且**非他项目**(不抢卡)；(3) `JOB_B=<新alloc> SMOKE=1 OVERWRITE=1 NCCL_DEBUG=INFO OUT=/tmp/vqvae_8card_smoke bash scripts/_launch_graph_vqvae_8card_crossnode_a100.sh` plumbing smoke 验 WORLD_SIZE=8 via NET/IB；(4) pkill 括号 `[t]orchrun.*train_graph_vqvae` 停 4 卡 + 验 GPU 释放（**不 scancel**）；(5) `ssh swarma1004 "JOB_B=<新alloc> setsid nohup bash scripts/_launch_graph_vqvae_8card_crossnode_a100.sh > scripts/_train_vqvae_8card.log 2>&1 </dev/null &"` durable 续训。切完监控加 8 卡 fingerprint + watch grad_norm/perplexity/dead/val（doubled lr+batch 改了学习动态）。

**触发检测**：哨兵 cron 9446199c 每 tick 查 RUNNING 的新 4 卡 A100 alloc(≠974143、非 jupyter)，检测到→报 user(marker 防重复)+列就绪状态，**不自动切**（等 user 说切）。
