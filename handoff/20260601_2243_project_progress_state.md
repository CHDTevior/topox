# TopoSlots 项目进度文档 (2026-06-01 22:43 BST)

**项目目标**: 多拓扑动作迁移 + 文本控制动作生成 → NeurIPS 2026。
**本文档**: 现状 / 待办 / 已完成 / 复现 / harness 流程 / 失败教训。
**最新 commit**: `6d5cef3` (rot6d-FK fix + w_fk=1.0 + 4-card cross-alloc DDP)。

---

## 1. 现在正在干的 (CURRENT)

**B rot6d_fk arm — 4 卡 cross-alloc 训练** (NeurIPS 主实验 arm B):
- 节点: swarmh1002, **4×H100 cross-alloc** (两个同节点 alloc `944459`+`944460`, 各 2 卡, IB rendezvous)
- 配置: `loss_mode=anytop13_world_rot6d_fk`, w_world=0.25 / **w_fk=1.0** / w_traj=0.10, global batch 128 (4×bs32), **lr 8e-4**
- 状态 (22:43): ep1 训练中, ep0 done **995s** (16.6min, **比 2 卡 1950s/ep 快 ~2×**), loss ~9 下降中, ERR 0
- OUT: `runs/m1_l2_anytop13_edgeseg_C128_rot6dfk_w025f100t010_g128_4card_seed42/`
- orchestrator durable (PPID=1, swarmh1002), 监控走 OUT/train.log

**监控**: /loop cron `6d384dbc` (每 30m, session-only, 7 天过期) — B + diffusion + world_geom。

---

## 2. 将要干的 (NEXT / PENDING)

**★ 等 user 定的两个决策:**
- **diffusion ep10 lr 决策** (swarma1004): val_denoise 卡在 **0.3734** (ep0 0.3738→ep10 0.3734, lr 1e-3 没收敛), > baseline 0.3688。选项: 降 lr 5e-4 重启 / 接受 ep40 的 0.3688 作 final / 再等。
- **world_geom resume** (swarma1001, walltime GONE): 表现极好 (val 1.395 持续大降), 但无空闲 4×A100。有 `--resume` + last_model.pt。去向: 等卡 / 2×A100 / 别 alloc。

**B 训练后续** (codex 把关, 不降级):
- ep30-50 → fresh codex thread 评估 loss 趋势 + 是否触发 long-chain QA。
- 训练完 → **三栏 PRED QA** (GT_RIC | PRED_RIC | PRED_FK, 需干净 pred, 本 session 只做了 GT 两路验证) + unseen-species eval。
- A/B 对比: baseline A (无 geometry loss) vs B (rot6d_fk) — loss 是唯一实验变量 (global batch / config 都 match)。

---

## 3. 已经做完的 (DONE — 本 session 2026-06-01)

**核心: rot6d-FK double-root-rotation bug 修复**
- **bug**: `recover_from_bvh_rot_np`(numpy) + `recover_rot6d_fk_positions_torch`(torch) 多乘一次 root global rotation (turn yaw 用了两遍) → FK 恢复的骨架比 RIC ~2× 全局旋转。
- **发现**: user 看 GT_RIC vs GT_FK 渲染 (Saiga) 肉眼抓到 — **所有数字检查都没抓到**。
- **诊断**: RIC 路 = ground truth (从 position 通道直接恢复, 无 FK 递归)。判据 FK==RIC; clean_L2 Saiga + 1070 老 truebones 最大旋转 clip, 删 root correction 后 absL1 **0.65→0.0000** (user 独立验证 1e-9)。
- **修复**: 删 2 文件各 1 行 root correction。numpy+torch smoke PASS (FK==RIC 2e-7, autograd OK)。codex PASS (`019e84c0`)。
- **污染修正**: gt_fk_mismatch=0.29 是 bug 产物 (修后→0); "哺乳动物 RIC-FK 分歧 60-87%" 全是 bug; 旧 2 卡 B (ep25) fk loss 被污染 → 停 + 重启。

**w_fk 标定**: post-fix calibration (frozen baseline VAE forward) → fk raw 0.176 ≈ world 0.156 (1.13×, 旧 buggy 1.82×)。user 定 **w_fk=1.0** (weighted 12.1% of base, 硬信号测 long-chain/wings; w_fk=0.5 的 6% 太软)。

**重渲 QA**: GT_RIC vs GT_FK 全 0.0000 (Saiga GIF 给 user 看, 绿==红)。

**cross-alloc 4 卡 DDP infra**: codex 两轮审 (`019e84f9`: NEEDS-FIX 5 项→PASS) + smoke PASS。详 §4/§5/§6。

**其他**: docstring 修 (FK 文件 "official"→"patched"); `~/.claude/CLAUDE.md` 加 cross-alloc DDP section (8 条跨项目经验); commit `6d5cef3`。

---

## 4. 怎么复现现在的训练 (REPRODUCE)

**环境**: cwd `/scratch/ts1v23/workspace/noKslot_clean` (= `/iridisfs/...`)。conda `/scratch/ts1v23/.conda`。数据 `data/anytop_planet_zoo_clean_L2` (13ch RIFKE)。

**B 4 卡 cross-alloc 启动** (前提: 两个同节点 H100 alloc, 各 gres=gpu:2):
1. **smoke 先验 (必, fail-loud)**:
   `SMOKE=1 NCCL_DEBUG=INFO bash scripts/_launch_rot6d_fk_B_4card.sh`
   → 确认 `WORLD_SIZE=4` + NCCL `via NET/IB/0` + rc_A=0 rc_B=0。
2. **durable 真跑 (compute node, 非登录节点)**:
   `ssh swarmh1002 "cd /scratch/ts1v23/workspace/noKslot_clean && setsid nohup bash scripts/_launch_rot6d_fk_B_4card.sh > scripts/_train_fkB_4card.log 2>&1 </dev/null &"`
3. **改 alloc**: orchestrator 顶部 `JOB_A`/`JOB_B` (默认 944459/944460), `RDZV_HOST=swarmh1002-ib0`。

**关键配置** (orchestrator → launch 的 COMMON_ENV): w_fk=1.0, BS=32, LR=8e-4, NNODES=2, static rendezvous (`--node_rank --master_addr --master_port`), NCCL `P2P_DISABLE=1 SHM_DISABLE=1 SOCKET_IFNAME=ib0 IB_DISABLE=0`。

**2 卡单 alloc 版** (对照/fallback): `CVD=0,1 setsid nohup bash scripts/_launch_rot6d_fk_B.sh` (默认 NNODES=1 standalone, lr 4e-4 global 64)。

**监控**: 走 `OUT/train.log` (rank0 直写), **不是** orchestrator log (sed buffer)。

---

## 5. 我们的 harness 流程 (WORKFLOW)

**每个代码改动的铁律流程**:
1. 改代码 → 2. **smoke 验证** (具体可验证标准: FK==RIC / no-OOM / rendezvous WORLD_SIZE) → 3. **codex review** (gpt-5.5 xhigh, **fresh thread**, brief 写 `scripts/_codex_*_brief.md`) → 4. NEEDS-FIX 则 fix + re-review (codex-reply 同 thread) → 5. PASS 后才真跑。
- codex MCP 断 → fallback `codex exec` CLI。代码改动**必经 codex 审**。

**CV 任务: visual QA primacy** — 渲染可视化 (gif / 并排 GT-vs-pred) 人眼/codex 看, **优先级高于 metric**。数字好 ≠ 视觉对 (FK bug 就是 metric 全过、visual 抓到)。训练前 QA 数据 + 训练后 QA 输出都做。

**训练监控**: durable monitor (compute node `setsid nohup` PPID=1, survive ssh 断) 或 /loop (session-only cron)。判活走 **rank0 log + GPU util**, 不靠可能 buffer 的 orchestrator log。

**Slurm 纪律**: **不 self-submit/cancel** (可 pkill 自己进程, pkill -f 别匹配自己 ssh 串); **不抢别项目正在用的卡** (启动前 nvidia-smi + squeue 核验)。

**handoff**: 时间戳前缀 + 内容后缀命名。STATE compact header 优先读, 不全读大文件 (防 context 溢出)。

**linear scaling** (Goyal): batch ×k → lr ×k, total_iter /k, epochs 不变。

---

## 6. 失败经验教训 (LESSONS — 这么多次失败)

**A. FK double-rotation bug (本 session 核心)**
- **数字检查会漏 bug, 必须 visual QA**: SMOKE1 验 torch FK==numpy FK (1e-6) 但两个一起 double; "<1% bbox" 用 root 不动的 idle 样本没暴露; user 看 Saiga 运动一眼看出。
- **ground truth 选最简、最被验证的那条** (RIC 无 FK 递归), 判据 FK==RIC; **不要拿"官方代码"当 ground truth** (官方也是 codex 写的、自带同 bug)。
- bug 污染下游 (gt_fk_mismatch / calibration / 训练 target / 渲染全错), 修复后 **re-measure 一切**。

**B. cross-alloc DDP (本 session, 见 CLAUDE.md 8 条)**
- **c10d rendezvous FAIL**: agent hostname (swarmh1002) ≠ IB rdzv host (swarmh1002-ib0) → 没人起 store → 都 client timeout。用 static + 显式 node_rank。**是 fail-loud smoke 抓到的 — cross-alloc 必先 smoke**。
- **NCCL hang**: 同节点跨 cgroup P2P/SHM 被 Slurm 隔离 → `NCCL_P2P_DISABLE=1 SHM_DISABLE=1` 强制 IB。
- **sed block-buffer**: orchestrator log 只有 launching echo, 看着像卡 (其实 GPU util 100%)。监控走 rank0 OUT/train.log; sed 要 `stdbuf -oL`。
- **durable 位置**: 登录节点 orchestrator 死 → srun step 死。要 compute node setsid nohup。

**C. 监控/进程 (历史 + 本 session)**
- **pkill -f 自匹配**: pattern 匹配执行 pkill 的 ssh 命令串 → 杀自己 (exit 255)。用 `[t]`/`[_]` grep trick (`ps|grep "[t]rain"|awk|kill`)。
- **nohup/subagent 死**: 登录节点 nohup & + Agent subagent 都 ~1.5h 死。长 monitor 用 ssh compute-node setsid nohup (PPID=1)。
- **共享节点 nvidia-smi util 假 0%**: cgroup 视图不稳。判活用 log mtime + iter 递增。

**D. 实验纪律 (历史)**
- **metric-乐观 false-PASS**: bone_cos 0.92 但视觉乱团。codex 审要能真暴露失败 (读数据/跑 smoke), 不盖章。
- **抢卡**: 别项目 --overlap 抢我方卡, 吞吐骤降。启动前核验卡空闲; 被抢要察觉上报。
- **数据-claim 错配**: 给 user 的"分歧 60-87%"实际是 bug。报数前自查来自哪条真实回显, 修 bug 后撤回受污染结论。

---

## 附: 关键文件
- **修复**: `src/data/anytop_rot6d_fk.py`, `src/models/graph_salad/rot6d_fk_recovery.py`
- **launcher**: `scripts/_launch_rot6d_fk_B.sh` (multi-node), `scripts/_launch_rot6d_fk_B_4card.sh` (orchestrator)
- **训练**: `scripts/train_graph_vae.py` (标准 torchrun DDP, ckpt rank-0-only :932)
- **诊断/smoke**: `_diag_fk_variants.py`, `_diag_oldset_fk_variants.py`, `_smoke_fk_fix_torch.py`, `_calibration_world_rot6d_fk.py`
- **codex briefs**: `_codex_fk_fix_brief.md`, `_codex_crossalloc_brief.md`
- **修复报告**: `handoff/20260601_2102_rot6d_fk_double_rotation_fix.md`
- **cross-alloc 跨项目经验**: `~/.claude/CLAUDE.md` → "同节点多 Slurm alloc 合并成 cross-alloc DDP" (8 条)
