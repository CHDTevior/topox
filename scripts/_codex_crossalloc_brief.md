# Codex 审: cross-alloc 4-card H100 DDP infra (rot6d_fk arm B)

## 任务
审 cross-alloc 4 卡 DDP infra(把 B 训练从 2 卡扩到 4 卡, 跨两个**同节点** Slurm alloc)。
判定正确性 + race / rendezvous / NCCL / durable 隐患。

## 背景
- B = rot6d_fk arm (loss_mode=anytop13_world_rot6d_fk, w_world=0.25 / w_fk=1.0 / w_traj=0.10)。
- FK double-rotation bug 已修(codex thread 019e84c0 PASS)。w_fk=1.0 由 post-fix calibration 定
  (weighted fk=12.1% of base)。
- 之前 2×H100(单 alloc 944459) global 64 lr 4e-4。**用户决定**改 4 卡 cross-alloc:
  944459+944460(都在**同节点 swarmh1002**, 各 2×H100), global 128, lr 8e-4(Goyal linear scaling)。
- 用户实测连通性: swarmh1002-ib0 IB reachable 200G; cross-alloc DDP 走 IB。
- 用户提醒: 跨节点 P2P 不是 NVLink 而是 TCP/IB/RDMA; 真正 GPU P2P 只同节点谈。

## 改动
1. **scripts/_launch_rot6d_fk_B.sh**(改): 加 LR / NNODES / RDZV_ENDPOINT / RDZV_ID 参数;
   NNODES>1 走 c10d rendezvous(`--rdzv_backend=c10d --rdzv_endpoint --rdzv_id`)+ IB NCCL env
   (NCCL_SOCKET_IFNAME=ib, NCCL_IB_DISABLE=0, NCCL_DEBUG=WARN); 默认 NNODES=1 standalone 不变
   (2 卡路径无行为变化)。guard 加 `[ "$NNODES" -le 1 ]` 例外(同节点 pgrep 会误匹配 peer alloc
   的 rank → 自我 ABORT)。`--lr 4.000e-04` → `--lr "$LR"` 参数化。
2. **scripts/_launch_rot6d_fk_B_4card.sh**(新 orchestrator): `srun --jobid=944459/944460 --overlap
   --nodes=1 --ntasks=1` 各跑一次 launch(NNODES=2, 同 RDZV_ENDPOINT=swarmh1002-ib0:29500 + 同
   RDZV_ID=fkB4card, CVD=0,1, BS=32, LR=8e-4, OUT 共享); c10d 自动 assign 4 global ranks;
   两 srun 后台 + wait。
3. **train_graph_vae.py**: 不改。标准 torchrun DDP(_ddp_setup:230-245 用 RANK/LOCAL_RANK/
   WORLD_SIZE, set_device(local_rank), DDP device_ids=[local_rank])。

## 已确认安全(请复核)
- ckpt save rank-0-only: train_graph_vae.py:932 `if do_val and is_main:` 守卫 val + last/best/
  best_recon save(都在该块缩进内); :1114 is_main periodic; :1131 dist.barrier。→ 无写文件 race。
- out_dir.mkdir(exist_ok=True):484 安全; overwrite check:480 两 node overwrite=True 跳过 raise。

## 审查点(请逐条判定)
1. **c10d rendezvous 正确性**: 两 alloc 各跑 `torchrun --nnodes=2 --nproc_per_node=2
   --rdzv_backend=c10d --rdzv_endpoint=swarmh1002-ib0:29500 --rdzv_id=fkB4card` → 自动凑 4 global
   ranks? **同节点**两 alloc 用同 rdzv_endpoint 会不会 port 冲突 / c10d 把两者当同一 host?
   需要手动 node_rank 吗? rdzv host(rank 0)在哪个 alloc 起 c10d store?
2. **NCCL IB cross-alloc**: NCCL_SOCKET_IFNAME=ib + NCCL_IB_DISABLE=0。同节点跨 cgroup(两 alloc)
   的 GPU 通信: P2P/NVLink 大概率被 Slurm cgroup 隔离 → 会 fallback IB 还是直接 hang? 需不需要
   显式 NCCL_P2P_DISABLE=1 / NCCL_SHM_DISABLE 设置避免它尝试不可用的 P2P?
3. **srun --overlap interactive alloc**: 944459/944460 是 interactive(salloc, JobName=inter_STJ)。
   `srun --jobid --overlap --nodes=1 --ntasks=1 bash -c "..."` 进 running interactive alloc 跑
   torchrun, 正确? 会不会和 salloc 持有的 shell 冲突? 2 卡/16-32 CPU 够 torchrun nproc=2 +
   dataloader workers?
4. **lr/global batch**: global 128(4×bs32), lr 8e-4(linear scaling from global-64 lr-4e-4),
   epochs 300 不变。正确? train_graph_vae 有无 warmup 需同步 scale?
5. **guard NNODES 例外**: cross-alloc 跳过 pgrep guard(同节点会误匹配 peer)。正确? 会不会反而放过
   真正的 double-launch(如用户手滑跑两次 orchestrator)?
6. **rank 0 node 不确定性**: c10d 下 global rank 0 = 哪个 alloc 不定。OUT 共享 fs(iridisfs),
   rank 0 写 ckpt。有无隐患?
7. **durable**: orchestrator 打算 setsid nohup(登录节点) + srun 提交到 alloc。**登录节点
   orchestrator 死(ssh 断)→ srun client 死 → srun task(训练)是否随之死?** 如何让 cross-alloc
   训练 durable? (我们既有 durable 模式是 setsid nohup 直接在 compute node, 但 cross-alloc 需要
   从能同时 srun 两 alloc 的地方启动。建议?)
8. **smoke 计划**: SMOKE=1 orchestrator → 两 alloc rendezvous + 5 iters。够验证 cross-alloc
   正确性(rendezvous + IB NCCL + bs32 no-OOM)吗? 还需测什么?

## 相关文件
- scripts/_launch_rot6d_fk_B.sh (改, 完整审)
- scripts/_launch_rot6d_fk_B_4card.sh (新 orchestrator, 完整审)
- scripts/train_graph_vae.py:230-245(_ddp_setup), :478-484(out_dir), :932/:1114/:1131(is_main/barrier)

请给最终 verdict: **PASS** 或 **NEEDS-FIX**(若 NEEDS-FIX, 列每个具体 fix)。
