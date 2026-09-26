#!/usr/bin/env bash
# UniMate 跨 alloc 8 卡 DDP 编排（user 2026-09-20: "你把那四个 allocation 给它，它能连起来的…就用 8 卡 DDP 训"）。
#
# 把 swarmh1001 上我持有的 4 个 2 卡 alloc 合成一个 8 进程 DDP job。这套模式此前在 swarmh1002 上以
# torchrun 验证过（2-alloc 与 3-alloc/6 卡），本脚本是它的 **accelerate 版**：UniMate 用
# `accelerate launch`，其 --machine_rank/--main_process_ip/--main_process_port/--num_machines
# 与 torchrun 的 --node_rank/--master_addr/--master_port 一一对应。
# `--num_processes` 是**总进程数**（accelerate --help 原文 "The total number of processes"），故为 8 而非 2。
#
# 为什么每条都不能省（全部来自那次跨 alloc 验证踩出来的坑）：
#  1. **静态 rendezvous + 显式 machine_rank**，不用自动选主：agent 看到的 hostname 是 swarmh1001，
#     而走 IB 时地址是 ib0 的 10.6.15.11，两者不等 → 没人起 TCPStore → 全体当 client 连超时。
#     所以 main_process_ip 直接给 **IB 的 IP**（不是主机名，免 DNS 别名）。
#  2. **NCCL 必须关 P2P/SHM 并强制走 IB**：8 张卡虽在同一物理节点，但分属 4 个 cgroup，
#     Slurm 隔离了 P2P/SHM，NCCL 默认去试会 error 或 hang。
#  3. **srun 必须显式 --gres/--cpus-per-task/--no-kill**，否则可能拿不到该 alloc 的卡、CPU 被限到默认配额。
#  4. 本脚本要在计算节点上 setsid 起（PPID=1），否则 ssh 一断 srun client 就死、step 随之死。
#  5. 存盘 rank-0-only：UniMate 自己用 accelerator.is_main_process 守着（train.py:57 取 is_main，
#     :263 的 if is_main 守存档、前后 :262/:268 各一个 wait_for_everyone，落盘在 :362），已满足。
#     （审查 2026-09-20 纠正：我原先写的 :430 守的是日志打印，不是 checkpoint。）
#  6. 判活看 rank0 的日志与 GPU util，不要看本脚本的聚合输出（srun|sed 会块缓冲，看着像卡住其实在跑）。
#  7. 双启动防护用 flock：同节点 pgrep 会误匹配 peer alloc 的 rank，不能用它当防护。
#  8. 同型号卡：4 个 alloc 全是 H100，不混型号。
#  9. **计算节点无外网**：文本塔 google/flan-t5-base 必须先在登录节点预取进 ~/.cache/huggingface
#     （已完成，944 MB），这里再设 HF_HUB_OFFLINE=1/TRANSFORMERS_OFFLINE=1，否则首次加载会去试网络然后失败或挂住。
#
# 用法：
#   SMOKE=1 bash scripts/_unimate_crossalloc_launch.sh      # 先验 rendezvous，必做
#   bash scripts/_unimate_crossalloc_launch.sh              # 真跑
#   RESUME=<ckpt> bash scripts/_unimate_crossalloc_launch.sh  # 从存档续（见下）
#
# 续训（审查 2026-09-20 必修 1）：save_interval=10000 / num_steps=100000 → 全程只有 10 个存档，
# 每个间隔约 1-1.5 h。UniMate 自己支持恢复 model/EMA/optimizer/lr_scheduler/step（train.py:46 的
# --resume，:154-159 调用，:366-411 恢复；README:108 原文 "a run continues exactly where it stopped"），
# 但本编排此前根本没把这个开关接出来 —— 崩在第 12 小时就是丢 12 小时而不是丢 1.5 小时。
# 而 UniMate README:145 自己写明 Objaverse-XL 有相当比例的坏 rig/clip "can destabilize or collapse a run"，
# 所以崩的概率不是零。续法：
#   RESUME=$OUT/checkpoints/checkpoint_step_90000.pt bash scripts/_unimate_crossalloc_launch.sh
# durable（真跑时）：
#   ssh swarmh1001 "cd /scratch/ts1v23/workspace/noKslot_clean || exit 1
#     { setsid nohup bash scripts/_unimate_crossalloc_launch.sh ; } > runs/_unimate/orch.log 2>&1 </dev/null &
#     exit 0"
set -uo pipefail
P=/scratch/ts1v23/workspace/noKslot_clean
cd "$P" || exit 1
U="$P/outside_docs/UniMate"
V=/iridisfs/scratch/ts1v23/workspace/unimate_venv
NODE=${NODE:-swarmh1001}
SMOKE=${SMOKE:-0}
PORT=${PORT:-29600}
# 精度：UniMate 全仓库从不指定 —— 配置 JSON 无精度字段、README:103 的 8 卡命令与 scripts/run_train.sh:33
# 都不传 --mixed_precision、仓库无 accelerate 配置文件、~/.cache/huggingface/accelerate 不存在。
# 所以"按它自己的方式跑" = accelerate 默认的 no（fp32）。此前 smoke 用的 bf16 是我加的，已改回其默认。
# 要换速度就 MP=bf16，但那样得在汇报里标明"非其文档默认"。
MP=${MP:-no}
CFG=${CFG:-configs/objaverse_60frames_graph_adaln.json}
OUT=${OUT:-}
GPUS_PER_ALLOC=${GPUS_PER_ALLOC:-2}
# 每 alloc 跑 2 个进程 x num_workers 8 = 16 个 dataloader worker（unimate/configs/schema.py:257 的默认，
# 真跑配置没有覆盖 num_workers）。原来给 8 核是 2 倍超订；alloc 实有 32 核，给 16 核零代价。
CPUS=${CPUS:-16}
RESUME=${RESUME:-}                                  # 非空则续训，见文件头
TAG=${TAG:-$(date -u +%Y%m%d_%H%M%S)}               # 日志名前缀：每次启动一套新日志（审查必修 3）

ts() { date -u +%FT%TZ; }
say() { echo "$(ts) [orch] $*"; }

mkdir -p "$P/runs/_unimate" "$P/.aris/meta"
exec 9>"$P/.aris/meta/.unimate_crossalloc.lock"
# 锁冲突必须 fail-loud：exit 0 会让一次撞锁的重试看起来像启动成功（审查 2026-09-22 item 4）
flock -n 9 || { echo "$(ts) [orch] 已有实例在跑，拒绝第二个实例"; exit 1; }

# ---- 发现本节点上我的 2 卡 alloc；数量不对就 fail-loud，绝不凑合 ----
mapfile -t ALLOCS < <(squeue -u "$USER" -h -o '%i %N %b' \
  | awk -v n="$NODE" '$2==n && $3 ~ /gpu:2$/ {print $1}' | sort)
if [ "${#ALLOCS[@]}" -ne 4 ]; then
  say "拒绝：$NODE 上找到 ${#ALLOCS[@]} 个 2 卡 alloc（需要 4 个）：${ALLOCS[*]:-无}"
  exit 1
fi
NMACH=${#ALLOCS[@]}
NPROC=$(( NMACH * GPUS_PER_ALLOC ))

# ---- master 的 IB 地址：直接取 IP，不用主机名（坑 1）----
IB_IFACE=${IB_IFACE:-ib0}
IB_HCA=${IB_HCA:-mlx5_0}
MASTER_IP=$(timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=8 "$NODE" \
  "ip -4 -o addr show $IB_IFACE | awk '{print \$4}' | cut -d/ -f1" </dev/null 2>/dev/null)
[[ "$MASTER_IP" =~ ^[0-9]+(\.[0-9]+){3}$ ]] || { say "拒绝：读不到 $NODE 的 $IB_IFACE 地址（得到 '${MASTER_IP}'）"; exit 1; }

if [ "$SMOKE" = 1 ]; then
  CFG=${SMOKE_CFG:-configs/_smoke_objaverse_crossalloc.json}
  OUT=${OUT:-$P/runs/_unimate/smoke}
  NCCL_DEBUG_LEVEL=INFO          # 坑 2 的验收证据：日志里要出现 via NET/IB
else
  OUT=${OUT:-$P/runs/_unimate/objaverse_60f_graph_adaln}
  NCCL_DEBUG_LEVEL=WARN
fi
mkdir -p "$OUT"

say "节点=$NODE  alloc=${ALLOCS[*]}  机器数=$NMACH  总进程=$NPROC"
say "rendezvous=$MASTER_IP:$PORT ($IB_IFACE/$IB_HCA)  SMOKE=$SMOKE"
say "配置=$CFG  输出=$OUT"

# 审查必修 1：空数组在 bash 4.4+ 的 set -u 下展开安全（本机 4.4.20，已实测）。
resume_args=()
if [ -n "$RESUME" ]; then
  [ -f "$RESUME" ] || { say "拒绝：RESUME 指向的存档不存在：$RESUME"; exit 1; }
  resume_args=(--resume "$RESUME")
  say "续训：$RESUME"
fi

pids=()
for i in "${!ALLOCS[@]}"; do
  j="${ALLOCS[$i]}"
  # 审查必修 3：原来是固定名 + ">" 截断，重启的第一个动作会把上一次的崩溃日志清掉 ——
  # 恰好是诊断唯一需要的东西。加启动时间戳前缀，jobid 仍在名字里（自描述）。
  log="$P/runs/_unimate/${TAG}_rank${i}_alloc${j}.log"
  # /usr/bin/env 是必须的：直接调 venv 里的 python 会被 uv shim 截走（rc=13）
  # --chdir 必须给：配置里的 dataset/features/... 是相对 **UniMate 仓库根** 的，
  # 而本编排自身 cd 在 noKslot_clean 根下。我先前在 UniMate 目录里验过该路径可达，
  # 但运行时的 cwd 不是那里 —— 验证必须打在运行时真正所处的上下文上。
  # 审查必修 2：原来这行带 --mem=64G。本集群 ConstrainRAMSpace=yes，该值是**硬** cgroup 上限：
  # 实测带它 memory.max=64GiB，不带则 =200GiB（alloc 全额）。每 alloc 跑 2 个进程 x 8 个 dataloader
  # worker = 16 个 fork，COW 页随 Python 引用计数在 10-15 h 里逐步解共享，RSS 单调爬升；30 步的
  # smoke 定不了 100,000 步的峰值上界，越过就是 cgroup OOM-kill。节点 1987 GiB 空闲，不设限零代价。
  srun --jobid="$j" --overlap -N1 -n1 --chdir="$U" \
       --gres=gpu:"$GPUS_PER_ALLOC" --cpus-per-task="$CPUS" --no-kill \
    /usr/bin/env \
      NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=0 \
      NCCL_SOCKET_IFNAME="$IB_IFACE" NCCL_IB_HCA="$IB_HCA" \
      NCCL_DEBUG="$NCCL_DEBUG_LEVEL" TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      PYTHONPATH="$U" \
      "$V/bin/python" -m accelerate.commands.launch \
        --num_machines "$NMACH" --machine_rank "$i" --num_processes "$NPROC" \
        --main_process_ip "$MASTER_IP" --main_process_port "$PORT" \
        --same_network --mixed_precision "$MP" \
        -m unimate.training.train --config "$CFG" --output_dir "$OUT" "${resume_args[@]}" \
    > "$log" 2>&1 &
  pids+=($!)
  say "alloc $j -> machine_rank $i -> $log"
done

say "四个 step 已派发，等待全部结束（判活请看 rank0 日志与 GPU util，不要看本文件 —— 坑 6）"
rc_all=0
for k in "${!pids[@]}"; do
  wait "${pids[$k]}"; r=$?
  say "machine_rank $k 退出 rc=$r"
  [ "$r" = 0 ] || rc_all=$r
done
say "全部结束，总 rc=$rc_all"
exit "$rc_all"
