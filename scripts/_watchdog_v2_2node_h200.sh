#!/usr/bin/env bash
# DURABLE auto-resume watchdog for the v2 in-context KTJD-17 run on 2x2 H200.
# DERIVED from _watchdog_h200_backbone.sh (proven across many alloc expiries); only the run it
# manages and the environment it forwards differ. User 2026-08-21: "不要担心卡断，之后会有续上，
# 你注意一下能保存检查点和resume" -- a 500-epoch run is ~5 days against 13-19h allocs, so an
# expiry is the NORMAL case, not an incident.
# Runs setsid nohup on a STABLE compute node (swarmh1002, ~4d alloc) so it survives
# SSH/session death (PPID=1) — fixing the overnight gap where the cross-node H200
# backbone dropped (alloc expiry) and nothing resumed it.
#
# 2026-06-15 (user: "H200 掉了就看有没有分配新的, 有就还是四卡H200老样子续上"):
# GENERALIZED off the hardcoded flamingo01+blossom03. The replacement H200 alloc may
# land on flamingo02 (dual_h200 = {flamingo01,flamingo02}) or blossom01/02/04
# (quad_h200 = {blossom01..04}). So every cycle we DYNAMICALLY DISCOVER the two H200
# nodes hosting MY gpu:2 allocs: exactly ONE dual_h200 alloc (-> MASTER, node_rank 0)
# + exactly ONE quad_h200 alloc (-> WORKER, node_rank 1). At resume we DETECT the UP
# IB iface/HCA/IP on the master (no longer assume ib1/mlx5_1) and pass them to the
# (now parameterized) launcher. If the two nodes' UP IB ifaces differ, we REFUSE to
# launch (fail-loud) rather than wire a mismatched NCCL_SOCKET_IFNAME.
#
# Healthy = a train_v2_incontext proc on BOTH discovered H200 nodes. DOWN for 2
# consecutive checks AND both nodes idle AND last_model.pt exists -> clean ONLY this
# run's orphans (exact jobids) + relaunch cross-node orchestrator RESUME_CKPT=last_model.pt.
# Manages ONLY the H200 backbone. flock single-instance. NEVER scancel.
set -uo pipefail
P=/scratch/ts1v23/workspace/noKslot_clean
OUT_REL="${OUT_REL:?output dir, relative to the repo root}"
OUT="$P/$OUT_REL"
LOG="$P/.aris/meta/watchdog_v2_2node.log"
CHECK_SEC="${CHECK_SEC:-300}"        # poll every 5 min
CTRL_NODE="${CTRL_NODE:-swarma1002}" # stable node used for squeue queries
STALE_SRUN_JOB_IDS="${STALE_SRUN_JOB_IDS:-}"   # prior known-dead alloc jobids to also pkill (empty for a fresh run)
# Resume config -- MUST match the original launch; the resume FORWARDS these to the launcher.
# Every one is REQUIRED: a watchdog that silently resumes with a different lr, batch or objective
# produces a run whose later epochs were trained under settings no log records.
# Forward the CONFIG FILE, not a hand-copied variable list. The launcher now requires 31 settings
# (architecture, schedule and the full objective), and a resume that forwards a stale SUBSET dies
# on the first `${VAR:?}` -- which silently converts "resumes across a card change" into "never
# resumes again". One file, sourced by the gate, the launch and this resume.
CFG="${CFG:?config file that the gate and the launcher both source}"
[ -f "$P/$CFG" ] || { echo "watchdog: config $CFG not found under $P" >&2; exit 1; }
# Source it HERE too. Two reasons: this script logs the run's settings, and under `set -u` an
# unset EPOCHS/LR/BATCH kills the watchdog on its very first log line -- turning "resumes across a
# card change" into "never even started". Sourcing also means an operator override exported before
# launching this watchdog is carried INTO the resume rather than silently reverting to the file
# default (codex 2026-08-23 round 3).
set -a; . "$P/$CFG"; set +a
EXTRA="${EXTRA:-}"
mkdir -p "$P/.aris/meta"


exec 9>"$P/.aris/meta/.watchdog_v2_2node.lock"
flock -n 9 || { echo "[wd] already running, exit"; exit 0; }

ts() { date -u +%FT%TZ; }
log() { echo "[wd $(ts)] $*" >> "$LOG"; }

log "START v2 watchdog (CHECK_SEC=$CHECK_SEC, out=$OUT_REL, epochs=$EPOCHS lr=$LR batch=$BATCH) pid=$$ host=$(hostname)"
down_streak=0

# Discover the two H200 nodes hosting MY gpu:2 allocs: exactly ONE dual_h200 (master)
# + exactly ONE quad_h200 (worker). Echo "mnode mjob wnode wjob", else return 1 (never guess).
discover_h200() {
    local sq dline qline mj mn wj wn
    sq=$(timeout 25 ssh "$CTRL_NODE" "squeue -u ts1v23 -t RUNNING -h -o '%i|%P|%N|%b' 2>/dev/null" 2>/dev/null) || return 1
    # exact partition match; GRES anchored on ':2$' so gpu:20 / gpu:h200:20 do NOT match.
    dline=$(printf '%s\n' "$sq" | awk -F'|' '$2=="dual_h200" && $4 ~ /:2$/ {print $1"|"$3}')
    qline=$(printf '%s\n' "$sq" | awk -F'|' '$2=="quad_h200" && $4 ~ /:2$/ {print $1"|"$3}')
    [ "$(printf '%s\n' "$dline" | grep -c .)" -eq 1 ] || return 1
    [ "$(printf '%s\n' "$qline" | grep -c .)" -eq 1 ] || return 1
    mj=${dline%%|*}; mn=${dline##*|}
    wj=${qline%%|*}; wn=${qline##*|}
    [ -n "$mn" ] && [ -n "$wn" ] && [ -n "$mj" ] && [ -n "$wj" ] || return 1
    # validate single, expected hostnames (guards against node ranges / unexpected nodes).
    [[ "$mn" =~ ^flamingo0[12]$ ]] || { log "DISCOVER-REJECT: master node '$mn' not flamingo0[12]"; return 1; }
    [[ "$wn" =~ ^blossom0[1-4]$ ]] || { log "DISCOVER-REJECT: worker node '$wn' not blossom0[1-4]"; return 1; }
    [[ "$mj" =~ ^[0-9]+$ ]] && [[ "$wj" =~ ^[0-9]+$ ]] || return 1
    printf '%s %s %s %s\n' "$mn" "$mj" "$wn" "$wj"
}

procs_on() {  # count of THIS run's train procs on node $1 (scoped to OUT_REL; 0 if unreachable)
    local n
    n=$(timeout 25 ssh "$1" "pgrep -u ts1v23 -fc '[t]rain_v2_incontext.*$OUT_REL'" 2>/dev/null || echo 0)
    echo "${n:-0}"
}
alive() {  # healthy = a train proc on BOTH currently-discovered H200 nodes
    local nodes mn wn
    nodes=$(discover_h200) || return 1   # can't resolve 2 H200 nodes -> treat as down
    read -r mn _ wn _ <<<"$nodes"
    [ "$(procs_on "$mn")" -ge 1 ] && [ "$(procs_on "$wn")" -ge 1 ]
}

# MY allocated GPUs idle (<500 MiB total), scoped to alloc $2 via srun --jobid so a
# quad_h200 co-tenant on the node's OTHER GPUs never blocks resume; fail -> 9999 (not idle).
alloc_gpus_idle() {
    local used
    used=$(timeout 45 ssh "$CTRL_NODE" "srun --jobid=$2 --overlap --quiet nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd+ | bc" 2>/dev/null || echo 9999)
    [ "${used:-9999}" -lt 500 ]
}

# /22 network address of IPv4 $1 (NCCL uses native IB verbs; ICMP/ping is filtered on
# this IPoIB fabric, so we sanity-check same-fabric by subnet, not by ping).
net22() {
    local o1 o2 o3 o4
    IFS=. read -r o1 o2 o3 o4 <<<"$1"
    [ -n "$o3" ] || { echo "x"; return; }
    echo "$o1.$o2.$(( (o3/4)*4 )).0"
}

# Detect the UP IB iface on node $1 -> echo "iface hca ip" (e.g. "ib1 mlx5_1 10.6.15.127"), else empty.
ib_info() {
    timeout 20 ssh "$1" '
        for i in $(ls /sys/class/net 2>/dev/null | grep -E "^ib[0-9]+$"); do
            st=$(cat /sys/class/net/$i/operstate 2>/dev/null)
            ip=$(ip -o -4 addr show "$i" 2>/dev/null | grep -oE "10\.[0-9.]+" | head -1)
            if [ "$st" = up ] && [ -n "$ip" ]; then
                hca=$(ls /sys/class/net/$i/device/infiniband/ 2>/dev/null | head -1)
                [ -n "$hca" ] && { echo "$i $hca $ip"; break; }
            fi
        done' 2>/dev/null
}

cleanup_orphans() {  # $1=job_a $2=job_b, rest = nodes. kill ONLY my train/launch + EXACT-jobid srun clients.
    local job_a="$1" job_b="$2" n jid
    shift 2
    for n in "$@"; do
        # scope the train-proc kill to THIS run's OUT (its cmdline has '--out $OUT_REL'),
        # so a different train_v2_incontext run of mine on the same node is never killed.
        timeout 25 ssh "$n" "pkill -u ts1v23 -9 -f \"[t]rain_v2_incontext.*$OUT_REL\" 2>/dev/null || true; pkill -u ts1v23 -f '[_]launch_graph_pscf_2node_h200.sh' 2>/dev/null || true" 2>/dev/null || true
        for jid in "$job_a" "$job_b" ${STALE_SRUN_JOB_IDS:-}; do
            [[ "$jid" =~ ^[0-9]+$ ]] || continue
            timeout 25 ssh "$n" "pkill -u ts1v23 -f '[s]run .*--jobid(=| )${jid}([^0-9]|\$)' 2>/dev/null || true" 2>/dev/null || true
        done
    done
}

resume() {
    local nodes mn mj wn wj mib wib mface mhca mip rest wface wrest whca wip out
    nodes=$(discover_h200) || { log "RESUME-WAIT: need EXACTLY ONE dual_h200 + ONE quad_h200 gpu:2 alloc of mine (ambiguous/missing); retry"; return 1; }
    read -r mn mj wn wj <<<"$nodes"
    if [ ! -f "$OUT/last_model.pt" ]; then
        log "RESUME-ABORT: no last_model.pt yet; retry"; return 1
    fi
    # IB: detect UP rail on both nodes; require SAME iface AND SAME HCA AND same /22 fabric.
    mib=$(ib_info "$mn"); wib=$(ib_info "$wn")
    [ -n "$mib" ] || { log "RESUME-WAIT: cannot detect UP IB on master $mn; retry"; return 1; }
    [ -n "$wib" ] || { log "RESUME-WAIT: cannot detect UP IB on worker $wn; retry"; return 1; }
    mface=${mib%% *}; rest=${mib#* }; mhca=${rest%% *}; mip=${rest##* }
    wface=${wib%% *}; wrest=${wib#* }; whca=${wrest%% *}; wip=${wrest##* }
    if [ "$mface" != "$wface" ] || [ "$mhca" != "$whca" ]; then
        log "RESUME-WAIT: IB mismatch master($mn=$mface/$mhca) vs worker($wn=$wface/$whca); refuse; retry"; return 1
    fi
    if [ "$(net22 "$mip")" != "$(net22 "$wip")" ] || [ "$(net22 "$mip")" = "x" ]; then
        log "RESUME-WAIT: IB subnet mismatch master $mip vs worker $wip (not same /22); refuse; retry"; return 1
    fi
    # final scoped recheck: if the run is actually alive (the 2 downs were transient ssh
    # blips), DO NOT cleanup/relaunch a healthy run — back off this cycle.
    if [ "$(procs_on "$mn")" -ge 1 ] && [ "$(procs_on "$wn")" -ge 1 ]; then
        log "RESUME-CANCEL: target alive on final scoped recheck (master=$mn worker=$wn); no cleanup"; return 0
    fi
    # clean MY orphans FIRST, then verify MY allocated GPUs are free (alloc-scoped, ignores co-tenants).
    log "RESUME: master=$mn($mj) worker=$wn($wj) IB=$mface/$mhca rdzv=$mip; cleanup orphans"
    cleanup_orphans "$mj" "$wj" "$mn" "$wn"
    sleep 5
    if ! alloc_gpus_idle "$mn" "$mj" || ! alloc_gpus_idle "$wn" "$wj"; then
        log "RESUME-WAIT: my allocated GPUs not free after cleanup (master=$mn worker=$wn); retry"; return 1
    fi
    # launch detached; the launcher writes its PID to .aris/meta/.v2_2node_orch.pid AFTER
    # acquiring flock. rm that pidfile first, launch, then 8s later confirm the PID is alive
    # AND is the launcher (PID-based, so it cannot false-match the ssh wrapper's own argv).
    # WHICH checkpoint to resume from. last_model.pt is the most RECENT state, not necessarily a
    # usable one: run7 blew up in ep29, that epoch's validation overwrote last_model.pt with the
    # wrecked weights, and this watchdog resumed them -- eight hours of training a dead model.
    # The trainer now stamps `healthy`, so ask before relaunching and fall back to best_model.pt.
    # WHICH checkpoint to resume from. last_model.pt is the most RECENT state, not necessarily a
    # usable one: run7 blew up in ep29, that epoch's validation overwrote last_model.pt with the
    # wrecked weights, and this watchdog resumed them -- eight hours of training a dead model.
    # The trainer stamps `healthy`; anything other than a clear HEALTHY is treated as unusable,
    # because an unreadable probe or a missing stamp is exactly the case where guessing caused the
    # original loss (codex 2026-08-24). `timeout -k` guarantees the probe cannot wedge this loop.
    probe_health() {   # $1 = checkpoint path relative to $P
      [ -f "$P/$1" ] || { echo MISSING; return; }
      (cd "$P" && CUDA_VISIBLE_DEVICES='' timeout -k 10 180 python3 -c "
import sys, torch
try:
    c = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
except Exception:
    print('UNREADABLE'); raise SystemExit
h = c.get('healthy')
if h is True:
    print('HEALTHY')
elif h is None:
    print('UNSTAMPED')
else:
    print('WRECKED %.1fx' % (c.get('val_over_best') or 0))
" "$P/$1" 2>/dev/null | head -1) || echo PROBE_FAILED
    }
    RESUME_CKPT=""
    lh=$(probe_health "$OUT_REL/last_model.pt")
    if [ "$lh" = HEALTHY ]; then
      RESUME_CKPT="$OUT_REL/last_model.pt"
    else
      bh=$(probe_health "$OUT_REL/best_model.pt")
      if [ "$bh" = HEALTHY ]; then
        log "last_model.pt is ${lh:-?} -- falling back to best_model.pt (verified HEALTHY)"
        RESUME_CKPT="$OUT_REL/best_model.pt"
      else
        log "REFUSING to relaunch: last_model.pt=${lh:-?} best_model.pt=${bh:-?}; neither is a "\
            "verified-healthy checkpoint. Manual inspection needed."
        return 1
      fi
    fi
    out=$(timeout 80 ssh "$mn" "cd $P && rm -f .aris/meta/.v2_2node_orch.pid && setsid nohup env JOB_A=$mj JOB_B=$wj MASTER_NODE=$mn WORKER_NODE=$wn RDZV_HOST=$mip NCCL_SOCKET_IFNAME=$mface NCCL_IB_HCA=$mhca CFG=$CFG OUT=$OUT_REL EXTRA='$EXTRA' RESUME=$RESUME_CKPT bash scripts/_launch_v2_ddp_2node_h200.sh > $OUT/orch_resume_wd.log 2>&1 </dev/null & sleep 8; pid=\$(cat .aris/meta/.v2_2node_orch.pid 2>/dev/null || true); { [ -n \"\$pid\" ] && ps -p \"\$pid\" -o args= 2>/dev/null | grep -qF '_launch_v2_ddp_2node_h200.sh'; } && echo STARTED || echo DIEDFAST" 2>/dev/null)
    if [ "$out" = STARTED ]; then
        log "RESUME launched OK on $mn (orchestrator PID alive after 8s); sleep 600 before next check"; return 0
    else
        log "RESUME-LAUNCH-FAIL on $mn (probe='${out:-<ssh-fail>}'; see orch_resume_wd.log); retry next cycle"; return 1
    fi
}

while true; do
    if alive; then
        down_streak=0
    else
        down_streak=$((down_streak+1))
        log "v2 run DOWN (streak=$down_streak)"
        if [ "$down_streak" -ge 2 ]; then
            if resume; then down_streak=0; sleep 600; continue; fi
        fi
    fi
    sleep "$CHECK_SEC"
done
