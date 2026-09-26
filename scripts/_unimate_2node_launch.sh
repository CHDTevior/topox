#!/usr/bin/env bash
# UniMate 两节点 × 4 卡编排（user 2026-09-22 选 A：UniMate 等我们那条训完，在 pink 的 8×H200 上续）。
#
# 与 scripts/_unimate_crossalloc_launch.sh 是同一个 8 进程 DDP，只是卡的形状不同：那边是同一台节点上
# 4 个 2 卡 alloc，这边是两台节点、各一个 4 卡 alloc。进程总数仍是 8，所以全局 batch（24 × 8）、按步的
# 学习率日程、每个进程的随机种子（seed + process_index，unimate/training/train.py:69）都和原跑一样；
# --resume 恢复模型 / EMA / optimizer / lr_scheduler / step（train.py:366-411）。原脚本不动，它仍是
# "4 个 H100 落在同一台节点"时不改代码的续训路径。
#
# 与原脚本不同的地方（其余照抄原脚本，各条坑的来历见原脚本头部）：
#  1. 两台节点：--num_machines 2 --num_processes 8，每台 machine_rank 0/1，每台一个 4 卡 alloc。
#  2. NCCL 不关 P2P/SHM：同一台节点的 4 张卡属于同一个 alloc，没有原脚本那种跨 cgroup 的隔离。
#     照搬我们在 pink 上跑 DDP 的设置（scripts/_launch_v2_ddp_2node_h200.sh:224-225），跨节点走 IB。
#  3. pink 的 IB 是 ib1 / mlx5_3（swarmh 是 ib0 / mlx5_0）；rendezvous 取 master 的 ib1 IP。
#  4. 派发前自己查卡：两台都要 compute-apps 为 0，查询失败按"被占用"处理（照 runs/_heldout/_resume.sh:74-80）。
#  5. flock 在这个 GPFS 上只在单节点内互斥（2026-09-22 审查实测），挡不住别的节点上的第二个 UniMate；
#     所以本脚本只许在 master（NODES[0]）上跑（同节点内 flock 够用），真跑再加一道：runs/_unimate 下
#     10 分钟内有非 smoke 的 rank0 日志在写就拒绝（UniMate 每步都刷进度条；崩溃后要等 10 分钟才能重起）。
#  6. 本脚本只做续训（审查 2026-09-22 P1）：真跑必须给 RESUME，OUT 与 RESUME 必须是绝对路径（训练经 --chdir
#     在 UniMate 目录下解析相对路径），RESUME 必须是 OUT/checkpoints 里编号最大的存档 —— 忘给 RESUME 会从头训、
#     从第 5000 步起覆盖原跑的存档；邻目录 objaverse_60f_graph_adaln 也有结构兼容的同名存档，指错了照样能载入。
#  7. UniMate 起跑时会重写 OUT/config.json 与 OUT/dataset_stats.npy（train.py:94-101）。派发前先各备份一份
#     *.before_<TAG>，起跑后用 cmp 比对，确认续训用的归一化与原跑一致。SMOKE 不接受外部 OUT，免得把 smoke
#     （objaverse 全集）的统计写进真跑目录。
#
# 用法（在 master 节点上 setsid，PPID=1 才活得过会话；形状照原脚本头部，cd 在前台、重定向绑在 { } 上）：
#   ssh pink7006 "cd /scratch/ts1v23/workspace/noKslot_clean || exit 1
#     { setsid nohup env SMOKE=1 bash scripts/_unimate_2node_launch.sh ; } >> runs/_unimate/<log> 2>&1 </dev/null &
#     exit 0"
#   续训：把 SMOKE=1 换成 CFG=configs/objaverse_common_v2_60frames_graph_adaln.json OUT=<绝对路径> RESUME=<绝对路径>
set -uo pipefail
P=/scratch/ts1v23/workspace/noKslot_clean
cd "$P" || exit 1
U="$P/outside_docs/UniMate"
V=/iridisfs/scratch/ts1v23/workspace/unimate_venv
read -r -a NODES <<< "${NODES:-pink7006 pink7008}"
SMOKE=${SMOKE:-0}
PORT=${PORT:-29600}
MP=${MP:-no}                      # UniMate 自己的默认（fp32），见原脚本
CFG=${CFG:-}
OUT=${OUT:-}
GPUS_PER_ALLOC=${GPUS_PER_ALLOC:-4}
CPUS=${CPUS:-16}                  # pink 的 4 卡 alloc 实有 16 核
IB_IFACE=${IB_IFACE:-ib1}
IB_HCA=${IB_HCA:-mlx5_3}
RESUME=${RESUME:-}
if [ "$SMOKE" = 1 ]; then TAG=${TAG:-smoke_$(date -u +%Y%m%d_%H%M%S)}; else TAG=${TAG:-$(date -u +%Y%m%d_%H%M%S)}; fi

ts() { date -u +%FT%TZ; }
say() { echo "$(ts) [orch2] $*"; }

[ "${#NODES[@]}" -eq 2 ] || { say "拒绝：NODES 需要恰好 2 台，得到 '${NODES[*]}'"; exit 1; }
# 第 5 条：只在 master 上跑，同节点内的 flock 才挡得住第二个实例
[ "$(hostname -s)" = "${NODES[0]}" ] || { say "拒绝：本脚本须在 master ${NODES[0]} 上运行（本机 $(hostname -s)）"; exit 1; }

mkdir -p "$P/runs/_unimate" "$P/.aris/meta"
exec 9>"$P/.aris/meta/.unimate_crossalloc.lock"      # 与原脚本同一把锁（同一台节点内互斥）
flock -n 9 || { say "已有实例在跑，拒绝第二个实例"; exit 1; }

if [ "$SMOKE" = 1 ]; then
  [ -z "$OUT" ] || { say "拒绝：SMOKE 不接受外部 OUT（'$OUT'），免得把 smoke 的统计写进别的目录"; exit 1; }
  CFG=${SMOKE_CFG:-configs/_smoke_objaverse_crossalloc.json}
  OUT=$P/runs/_unimate/smoke_2node
  NCCL_DEBUG_LEVEL=INFO          # 验收证据：日志里要出现 via NET/IB
else
  # 第 6 条：只做续训，路径全部显式、绝对，且续的是本 OUT 最新的存档
  [ -n "$CFG" ] && [ -n "$OUT" ] && [ -n "$RESUME" ] || { say "拒绝：真跑必须显式给 CFG、OUT、RESUME（本脚本只做续训）"; exit 1; }
  case "$OUT" in /*) ;; *) say "拒绝：OUT 必须是绝对路径：$OUT"; exit 1;; esac
  case "$RESUME" in /*) ;; *) say "拒绝：RESUME 必须是绝对路径：$RESUME"; exit 1;; esac
  [ -f "$RESUME" ] || { say "拒绝：RESUME 指向的存档不存在：$RESUME"; exit 1; }
  latest=$(ls "$OUT"/checkpoints/checkpoint_step_*.pt 2>/dev/null | sort -V | tail -n 1)
  [ -n "$latest" ] && [ "$(readlink -f "$RESUME")" = "$(readlink -f "$latest")" ] \
    || { say "拒绝：RESUME 不是 $OUT/checkpoints 里编号最大的存档（最大的是 '${latest:-无}'）"; exit 1; }
  case "$CFG" in /*) cfg_path=$CFG ;; *) cfg_path=$U/$CFG ;; esac
  [ -f "$cfg_path" ] || { say "拒绝：配置不存在：$cfg_path"; exit 1; }
  # CFG 里写了的每个字段都必须与原跑写在 OUT 里的 config.json 一致，只跳过自动算出的 dataset.max_joints：
  # 误给旧配置（objaverse 全集）模型结构兼容，会静默换数据续训（审查 2026-09-22 第 2 轮 P2）。读不到也拒绝。
  # config.json 里多出来的字段是 schema 默认值与自动推算值（同一份代码从同一份 CFG 得出），不比。
  cfg_diff=$(python3 - "$cfg_path" "$OUT/config.json" <<'PY'
import json, sys
def flat(d, p=""):
    o = {}
    for k, v in d.items():
        if isinstance(v, dict):
            o.update(flat(v, p + k + "."))
        else:
            o[p + k] = v
    return o
a, b = (flat(json.load(open(f))) for f in sys.argv[1:3])
print(" ".join(k for k in sorted(a) if k != "dataset.max_joints" and a[k] != b.get(k)))
PY
  ); cst=$?
  [ "$cst" -eq 0 ] && [ -z "$cfg_diff" ] || { say "拒绝：CFG 与 $OUT/config.json 不一致或读取失败（status $cst）：$cfg_diff"; exit 1; }
  NCCL_DEBUG_LEVEL=WARN
  # 第 5 条：跨节点的第二道防护；find 本身出错按"有人在跑"处理
  live=$(find "$P/runs/_unimate" -maxdepth 1 -name '*_rank0_alloc*.log' ! -name 'smoke_*' -mmin -10 2>&1); fst=$?
  [ "$fst" -eq 0 ] && [ -z "$live" ] || { say "拒绝：10 分钟内还有 UniMate rank0 日志在写，或查询失败（status $fst）：$live"; exit 1; }
fi

# ---- 每台节点恰好一个我的 4 卡 alloc；数量不对就 fail-loud ----
ALLOCS=()
for n in "${NODES[@]}"; do
  mapfile -t ids < <(squeue -u "$USER" -t RUNNING -h -o '%i %N %b' \
    | awk -v n="$n" -v g="$GPUS_PER_ALLOC" '$2==n && $3 ~ ("gpu:([a-z0-9]+:)?" g "$") {print $1}')
  [ "${#ids[@]}" -eq 1 ] || { say "拒绝：$n 上找到 ${#ids[@]} 个 ${GPUS_PER_ALLOC} 卡 alloc（需要 1 个）：${ids[*]:-无}"; exit 1; }
  ALLOCS+=("${ids[0]}")
done
NMACH=${#NODES[@]}
NPROC=$(( NMACH * GPUS_PER_ALLOC ))

# ---- 第 4 条：两台的卡都空着 ----
for n in "${NODES[@]}"; do
  out=$(timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=10 "$n" \
        'set -o pipefail; nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l' </dev/null 2>/dev/null); st=$?
  [ $st -eq 0 ] && [[ "$out" =~ ^[0-9]+$ ]] || { say "拒绝：$n 查卡失败（status $st，输出 '${out}'），按被占用处理"; exit 1; }
  [ "$out" = 0 ] || { say "拒绝：$n 的卡上有 $out 个进程"; exit 1; }
done

# ---- 第 3 条：master 的 IB 地址，直接取 IP ----
MASTER_IP=$(timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=8 "${NODES[0]}" \
  "ip -4 -o addr show $IB_IFACE | awk '{print \$4}' | cut -d/ -f1" </dev/null 2>/dev/null)
[[ "$MASTER_IP" =~ ^[0-9]+(\.[0-9]+){3}$ ]] || { say "拒绝：读不到 ${NODES[0]} 的 $IB_IFACE 地址（得到 '${MASTER_IP}'）"; exit 1; }

mkdir -p "$OUT"
say "节点=${NODES[*]}  alloc=${ALLOCS[*]}  机器数=$NMACH  总进程=$NPROC"
say "rendezvous=$MASTER_IP:$PORT ($IB_IFACE/$IB_HCA)  SMOKE=$SMOKE"
say "配置=$CFG  输出=$OUT"

resume_args=()
if [ -n "$RESUME" ]; then
  [ -f "$RESUME" ] || { say "拒绝：RESUME 指向的存档不存在：$RESUME"; exit 1; }
  resume_args=(--resume "$RESUME")
  say "续训：$RESUME"
fi
if [ "$SMOKE" != 1 ]; then
  # 第 7 条：起跑会重写这两个文件，先留底，起跑后 cmp
  for f in config.json dataset_stats.npy; do
    if [ -f "$OUT/$f" ]; then
      cp -p "$OUT/$f" "$OUT/$f.before_$TAG" || { say "拒绝：备份 $OUT/$f 失败"; exit 1; }
      say "已备份 $OUT/$f -> $f.before_$TAG（起跑后 cmp 比对）"
    fi
  done
fi

pids=()
for i in "${!ALLOCS[@]}"; do
  j="${ALLOCS[$i]}"
  log="$P/runs/_unimate/${TAG}_rank${i}_alloc${j}.log"
  srun --jobid="$j" --overlap -N1 -n1 --chdir="$U" \
       --gres=gpu:"$GPUS_PER_ALLOC" --cpus-per-task="$CPUS" --no-kill \
    /usr/bin/env \
      NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=0 NCCL_IB_DISABLE=0 \
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
  say "alloc $j (${NODES[$i]}) -> machine_rank $i -> $log"
done

say "两个 step 已派发，等待全部结束（判活看 rank0 日志与 GPU util，不要看本文件）"
rc_all=0
for k in "${!pids[@]}"; do
  wait "${pids[$k]}"; r=$?
  say "machine_rank $k 退出 rc=$r"
  [ "$r" = 0 ] || rc_all=$r
done
say "全部结束，总 rc=$rc_all"
exit "$rc_all"
