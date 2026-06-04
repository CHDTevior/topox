# Codex Review: cross-NODE 8-card a100 bf16 VAE DDP infra

## 背景
bf16 rot6d_fk VAE 训练, 8 卡 a100 跨**物理节点** cross-alloc DDP。bf16 改本身已 codex PASS(thread 019e8b40: fp32 双 smoke bit-identical + bf16 finite)。本次审**跨节点 infra**(orchestrator + launch multi-node + amp_dtype 传参)。

## 跨节点验证(已做)
- swarma1004(944455) ib0=10.6.15.68 + swarma1001(944456) ib0=10.6.15.8, cross-node IB ping 0.22ms 0% loss, iface=ib0

## 改动(worktree)
### scripts/_launch_rot6d_fk_B.sh
- 加 `AMP_DTYPE` 参数(default fp32) + torchrun `--amp_dtype "$AMP_DTYPE"`
- multi-node(NNODES>1)分支**此前同节点 4card 已 PASS**(thread 019e84f9): static rendezvous(`--nnodes --node_rank --master_addr --master_port --nproc_per_node`) + NCCL P2P/SHM disable + SOCKET_IFNAME ib0
- NPROC = CVD 卡数(CVD=0,1,2,3 → 4)

### scripts/_launch_bf16_vae_8card_xnode.sh (新, 跨节点 orchestrator)
- JOB_A=944455(swarma1004, node_rank 0, master) + JOB_B=944456(swarma1001, node_rank 1)
- MASTER_IB=10.6.15.68(swarma1004 ib0), MASTER_PORT 29500
- COMMON_ENV: `NNODES=2 MASTER_ADDR=10.6.15.68 CVD=0,1,2,3 BS=32 LR=1.6e-3 AMP_DTYPE=bf16 ...`; WORLD_SIZE = 4×2 = 8
- run_node: `srun --jobid --overlap --nodes=1 --ntasks=1 --gres=gpu:4 --cpus-per-task=32 --no-kill bash -c "NODE_RANK=$noderank $COMMON_ENV bash _launch_rot6d_fk_B.sh"`
- run_node nodeA 944455 0 + nodeB 944456 1; flock 单实例; durable on master(swarma1004) setsid nohup

## 审查点(请逐一)
1. **跨节点 static rendezvous 正确?** orchestrator 在 swarma1004 跑, srun --jobid=944455(本地) + srun --jobid=944456(远程 swarma1001)。node_rank 0(swarma1004) 用 MASTER_ADDR=10.6.15.68(它自己的 IB) host TCPStore, node_rank 1(swarma1001) connect via IB。这套**跨物理节点**的 static rendezvous(master_addr=直接 IB IP)对吗? 比同节点 loopback 有什么新风险?
2. **NCCL 跨节点配置?** NCCL_SOCKET_IFNAME=ib0(两节点都 ib0)。P2P_DISABLE/SHM_DISABLE=1 是同节点跨 cgroup 用的; 跨节点本就走 IB net — disable P2P/SHM 在跨节点是无害冗余还是会有问题? 是否应该让 NCCL 用 IB RDMA(NCCL_IB_DISABLE=0 已设)?
3. **srun --jobid 跨节点语义?** 从 swarma1004 上的 orchestrator 发 srun --jobid=944456 --overlap, srun step 会在 944456 所属的 swarma1001 上执行吗? 两个 srun(本地+远程)由 master 节点的 orchestrator wait, 对吗?
4. **linear scaling?** global = NPROC(4)×NNODES(2)×BS(32) = 256; lr = 8e-4 × 256/128 = 1.6e-3(基线: B 同节点 4card global128 lr8e-4)。a100-80GB bf16 BS32 是否合理(smoke 验 OOM)?
5. **durable 跨节点?** orchestrator 在 master(swarma1004) setsid nohup PPID=1; 它 wait 两个 srun。若 master 节点 orchestrator 死, 两 srun step 都死? 跨节点 durable 比同节点有何不同注意?
6. **两 alloc walltime?** 944455/944456 是独立 job, 任一到时/失败拖死整个 DDP。
7. **ckpt rank-0-only?** train_graph_vae 是否 is_main 守卫 ckpt 写(8 rank 跨节点共享 fs, 只 global rank 0 写)?

请读 worktree scripts/_launch_bf16_vae_8card_xnode.sh + scripts/_launch_rot6d_fk_B.sh(multi-node 分支 :84-105) + scripts/train_graph_vae.py(ckpt save 守卫)。逐点 + PASS/NEEDS-FIX。这是 smoke 前的审(smoke 会真验跨节点 rendezvous + NCCL IB + bf16 + WORLD_SIZE=8)。
