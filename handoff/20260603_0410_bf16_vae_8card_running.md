# bf16 VAE 8卡跨节点正式训 RUNNING (2026-06-03 ~04:10 BST)

## STATE: bf16 VAE 8卡跨节点正式训跑起来了 (durable)
- worktree `/iridisfs/scratch/ts1v23/workspace/noKslot_bf16vae`(分支 bf16-vae), 主 diffusion(noKslot_clean/main)完全不碰
- **8卡 a100 跨节点**: swarma1004(944455, rank0-3, master)+swarma1001(944456, rank4-7), WORLD_SIZE=8, NCCL via NET/IB/0
- config: **bf16**, BS48 global384 **lr8e-4** epochs300, durable PPID=1. ⚠ lr 历经 1.6e-3→2.4e-3(提 util)→**8e-4(frozen fix 2026-06-03)**: Goyal-linear 2.4e-3 太高致 VAE 塌缩 mean-pose(val speed_ratio ~0.02 🥶, loss 卡 8.8x 假收敛), 降 lr8e-4 后 **ep4 speed_ratio 1.1168 ✓OK**(pred 0.1837≈gt 0.1676) frozen 解 + loss 正常降(9.4→3.5 by ep4)
- ⚠ **train.log 是 append**(--overwrite 清 ckpt 不清 log): grep -c epoch/val 会混历史 run; 监控取**最后一个 run**(tail epoch done / speed_ratio)
- OUT: `runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42`(worktree 内)
- orchestrator: swarma1004 setsid nohup `scripts/_launch_bf16_vae_8card_xnode.sh`, log `scripts/_train_bf16_vae_8card.log`

## DONE (这个 session, bf16 VAE 完整)
- **bf16-safe 改 6 文件**(worktree): attention.py guard 放宽 + encoder/motion_decoder/treeik 5 处 softmax fp32 + losses KL(mu/logvar fp32)+contact BCE fp32 + train_graph_vae --amp_dtype flag(default fp32)+autocast+Gate2 放宽
- **codex 3 PASS**: bf16 数值(thread 019e8b40) + fp32 path PASS + 跨节点 infra(019e8b67, blocker=launch P hardcode noKslot_clean 已修为 worktree)
- **fp32 双 smoke bit-identical**: worktree fp32 == main fp32 (ep0 it0 loss 10.3772 / epoch 19.0182 完全一致) → fp32 byte-for-byte 不变 = diffusion 续训 + 合并 main 安全
- bf16 单卡 smoke(loss 10.45 finite) + **8卡跨节点 smoke 全 PASS**(rendezvous WORLD_SIZE=8 + NCCL via IB/0 + bf16 loss 12.30 finite + no-OOM)
- 8卡跨节点 bf16 VAE 正式训 RUNNING

## NEXT
1. **监控 bf16 VAE**(8卡跨节点 durable). 判活: util>0 + epoch 递增 + orch_PPID=1. ⚠ **两 alloc walltime: 944455 先到期 2026-06-06 08:14**(effective deadline; 944456 到 06-06 20:21). 任一 alloc 到时拖死 DDP
2. 训出 useful ckpt → 渲染 **VAE recon QA**(rot6d_fk recon 视觉, CV 可视化优先于 metric; 三栏 GT_RIC|PRED_RIC|PRED_FK or recon dual-path)
3. **fp32 不变已证 → 可合并 bf16-vae 分支回 main**(用户 2026-06-03 要的, 方便管理). 合并: `git -C noKslot_clean merge bf16-vae`(或 cherry-pick). fp32 路径 bit-identical 保证 diffusion 续训安全
4. 非阻塞优化: NCCL P2P/SHM real run 放开(node内4卡 NVLink 更快, codex 建议, node内同alloc不跨cgroup安全); bf16 mem 44GB 还可加 BS(linear scaling)

## 复现/监控命令
监控: `ssh swarma1004 'cd /scratch/ts1v23/workspace/noKslot_bf16vae; O=runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42/train.log; echo EP=$(grep -c "epoch [0-9]* done" $O); grep -E "epoch [0-9]+ done|val" $O|tail -2; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader|head -4; echo PPID=$(pgrep -f [_]launch_bf16_vae_8card|head -1|xargs -r -I{} ps -p {} -o ppid=)'`
⚠ a100 共享节点 nvidia-smi util 可信(独占 alloc); 判活 util+epoch+PPID。

跨节点重启(durable): `ssh swarma1004 "cd /scratch/ts1v23/workspace/noKslot_bf16vae && setsid nohup bash scripts/_launch_bf16_vae_8card_xnode.sh > scripts/_train_bf16_vae_8card.log 2>&1 </dev/null &"`

## 跨节点 cross-alloc 新经验(vs CLAUDE.md 同节点8条)
- 真跨物理节点: MASTER_ADDR=直接 IB IP(swarma1004 10.6.15.68), 不靠 hostname; NCCL via NET/IB/0(RDMA, 不是同节点 loopback)
- orchestrator 在 master 节点(swarma1004), srun --jobid=944455(本地)+srun --jobid=944456(远程 swarma1001, srun 跨节点进远程 alloc)
- launch P 必须指 worktree(不能 hardcode 主 checkout, 否则 torchrun 跑错代码) — codex 抓到的 blocker
- 两独立 alloc walltime: effective = 较早到期那个

## 主线(不在此 worktree, 勿动)
- diffusion backbone **fp32 正式训**(swarmh1002 6卡 H100, main): cron `cd4b0801` 监控, ep~20min, val_denoise ep0 0.4011, D_EP~5
- old diffusion 已停; 评估产物 qa_sample_dual/qa_cross_skeleton(old VAE, 用户判断偏弱, 等 backbone 训完再说)
