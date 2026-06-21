#!/usr/bin/env bash
# DURABLE H200 auto-resume watchdog for the n8192 GRAPH-VQVAE cross-node run
# (scripts/train_graph_vqvae.py via scripts/_launch_graph_vqvae_2node_h200.sh).
# ADAPTED VERBATIM from the PROVEN scripts/_watchdog_h200_backbone.sh — ONLY the run
# identity (OUT_REL, train-proc name train_graph_vqvae, the 2node launcher it relaunches,
# the resume env) differs; the H200 cross-node infra (dynamic dual_h200+quad_h200 node
# discovery, UP-IB detection, /22 fabric check, alloc-scoped idle check, scoped kills,
# fail-loud guards, flock single-instance, NEVER scancel) is unchanged.
#
# Runs setsid nohup on a STABLE compute node (swarmh1002, ~4d alloc), PPID=1, so it
# survives SSH/session death. Healthy = a train_graph_vqvae proc on BOTH discovered H200
# nodes. DOWN for 2 consecutive checks AND both my allocated GPUs idle AND last_model.pt
# exists -> clean ONLY this run's orphans (exact jobids) + relaunch RESUME_CKPT=last_model.pt.
# MUST be the ONLY H200 watchdog running (the backbone watchdog is stopped before this).
set -uo pipefail
P=/scratch/ts1v23/workspace/noKslot_clean
# Run identity + tokenizer config — ALL env-overridable so resume relaunches the CORRECT
# dataset/shape. A hardcoded L4safeTB/C96 here would silently corrupt an L4safeHuman resume.
OUT_REL="${OUT_REL:-runs/vqvae_L4safeHuman_C72_J144_d512_Q4_n8192_b16g64_300ep_seed42}"
ANYTOP_ROOT="${ANYTOP_ROOT:-data/animo4d_anytop_clean_L4_safe_plus_humanml3d}"
MAX_COARSE="${MAX_COARSE:-72}"
MAX_JOINTS="${MAX_JOINTS:-144}"
MAX_FRAMES="${MAX_FRAMES:-64}"
NUM_CODES="${NUM_CODES:-8192}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-6.65e-5}"
OUT="$P/$OUT_REL"
# Fail-fast config guard: an L4safeHuman run MUST use the humanml3d root + MAX_COARSE=72.
# Refuse to start (hence refuse to ever resume) on mismatch, so a stale old-default env can
# never silently relaunch this run as L4safeTB/C96.
if [[ "$OUT_REL" == *L4safeHuman* ]]; then
  if [ "$ANYTOP_ROOT" != "data/animo4d_anytop_clean_L4_safe_plus_humanml3d" ] || [ "$MAX_COARSE" != "72" ]; then
    echo "[wd-vq] ABORT: L4safeHuman run but ANYTOP_ROOT='$ANYTOP_ROOT' / MAX_COARSE='$MAX_COARSE' mismatch (need humanml3d root + 72)" >&2
    exit 1
  fi
fi
LOG="$P/.aris/meta/watchdog_h200_vqvae.log"
CHECK_SEC="${CHECK_SEC:-300}"        # poll every 5 min
CTRL_NODE="${CTRL_NODE:-swarmh1002}" # stable node used for squeue queries
STALE_SRUN_JOB_IDS="${STALE_SRUN_JOB_IDS:-}"   # none known-dead for this run
mkdir -p "$P/.aris/meta"

exec 9>"$P/.aris/meta/.watchdog_h200_vqvae.lock"
flock -n 9 || { echo "[wd-vq] already running, exit"; exit 0; }

ts() { date -u +%FT%TZ; }
log() { echo "[wd-vq $(ts)] $*" >> "$LOG"; }

log "START vqvae watchdog (CHECK_SEC=$CHECK_SEC, out=$OUT_REL, dynamic H200 node discovery) pid=$$ host=$(hostname)"

# Safety self-guard: MUST be the ONLY H200 watchdog. The backbone watchdog uses a
# DIFFERENT lock, so it would still discover + relaunch PSCF onto the same dual+quad
# H200 allocs. Abort if it is alive (it must be stopped before this VQVAE watchdog starts).
if pgrep -u ts1v23 -f "[_]watchdog_h200_backbone\.sh" >/dev/null 2>&1; then
    log "ABORT: _watchdog_h200_backbone.sh still running — only ONE H200 watchdog allowed; stop the backbone watchdog first"
    exit 1
fi
down_streak=0

# Discover the two H200 nodes hosting MY gpu:2 allocs: exactly ONE dual_h200 (master)
# + exactly ONE quad_h200 (worker). Echo "mnode mjob wnode wjob", else return 1 (never guess).
discover_h200() {
    local sq dline qline mj mn wj wn
    sq=$(timeout 25 ssh "$CTRL_NODE" "squeue -u ts1v23 -t RUNNING -h -o '%i|%P|%N|%b' 2>/dev/null" 2>/dev/null) || return 1
    dline=$(printf '%s\n' "$sq" | awk -F'|' '$2=="dual_h200" && $4 ~ /:2$/ {print $1"|"$3}')
    qline=$(printf '%s\n' "$sq" | awk -F'|' '$2=="quad_h200" && $4 ~ /:2$/ {print $1"|"$3}')
    [ "$(printf '%s\n' "$dline" | grep -c .)" -eq 1 ] || return 1
    [ "$(printf '%s\n' "$qline" | grep -c .)" -eq 1 ] || return 1
    mj=${dline%%|*}; mn=${dline##*|}
    wj=${qline%%|*}; wn=${qline##*|}
    [ -n "$mn" ] && [ -n "$wn" ] && [ -n "$mj" ] && [ -n "$wj" ] || return 1
    [[ "$mn" =~ ^flamingo0[12]$ ]] || { log "DISCOVER-REJECT: master node '$mn' not flamingo0[12]"; return 1; }
    [[ "$wn" =~ ^blossom0[1-4]$ ]] || { log "DISCOVER-REJECT: worker node '$wn' not blossom0[1-4]"; return 1; }
    [[ "$mj" =~ ^[0-9]+$ ]] && [[ "$wj" =~ ^[0-9]+$ ]] || return 1
    printf '%s %s %s %s\n' "$mn" "$mj" "$wn" "$wj"
}

procs_on() {  # count of THIS run's train procs on node $1 (scoped to OUT_REL; 0 if unreachable)
    local n
    n=$(timeout 25 ssh "$1" "pgrep -u ts1v23 -fc '[t]rain_graph_vqvae.*$OUT_REL'" 2>/dev/null || echo 0)
    echo "${n:-0}"
}
alive() {  # healthy = a train proc on BOTH currently-discovered H200 nodes
    local nodes mn wn
    nodes=$(discover_h200) || return 1
    read -r mn _ wn _ <<<"$nodes"
    [ "$(procs_on "$mn")" -ge 1 ] && [ "$(procs_on "$wn")" -ge 1 ]
}

# MY allocated GPUs idle (<500 MiB total), scoped to alloc $2 via srun --jobid; fail -> 9999.
alloc_gpus_idle() {
    local used
    used=$(timeout 45 ssh "$CTRL_NODE" "srun --jobid=$2 --overlap --quiet nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd+ | bc" 2>/dev/null || echo 9999)
    [ "${used:-9999}" -lt 500 ]
}

net22() {  # /22 network address of IPv4 $1 (IPoIB ICMP is filtered -> subnet check, not ping)
    local o1 o2 o3 o4
    IFS=. read -r o1 o2 o3 o4 <<<"$1"
    [ -n "$o3" ] || { echo "x"; return; }
    echo "$o1.$o2.$(( (o3/4)*4 )).0"
}

ib_info() {  # UP IB iface on node $1 -> "iface hca ip", else empty
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
        timeout 25 ssh "$n" "pkill -u ts1v23 -9 -f \"[t]rain_graph_vqvae.*$OUT_REL\" 2>/dev/null || true; pkill -u ts1v23 -f '[_]launch_graph_vqvae_2node_h200.sh' 2>/dev/null || true" 2>/dev/null || true
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
        log "RESUME-ABORT: no last_model.pt yet (fresh run not past first save); retry"; return 1
    fi
    # Safety: NEVER touch a node still hosting the CodeFlow backbone (its srun clients can
    # share these discovered jobids -> cleanup_orphans' jobid-scoped srun kill could hit them).
    # REFUSE if any train_graph_codeflow proc is present on either discovered node.
    local cfm cfw
    # pgrep -c prints "0" AND exits nonzero on no-match; keep the || INSIDE ssh and tr to
    # digits so cfm/cfw are always a clean integer (or empty if the node is unreachable).
    cfm=$(timeout 25 ssh "$mn" "pgrep -u ts1v23 -fc '[t]rain_graph_codeflow' 2>/dev/null || true" 2>/dev/null | tr -dc '0-9')
    cfw=$(timeout 25 ssh "$wn" "pgrep -u ts1v23 -fc '[t]rain_graph_codeflow' 2>/dev/null || true" 2>/dev/null | tr -dc '0-9')
    if [ "${cfm:-0}" -gt 0 ] || [ "${cfw:-0}" -gt 0 ]; then
        log "RESUME-REFUSE: CodeFlow backbone procs present (m=$cfm w=$cfw) on discovered nodes — not touching"; return 1
    fi
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
    if [ "$(procs_on "$mn")" -ge 1 ] && [ "$(procs_on "$wn")" -ge 1 ]; then
        log "RESUME-CANCEL: target alive on final scoped recheck (master=$mn worker=$wn); no cleanup"; return 0
    fi
    log "RESUME: master=$mn($mj) worker=$wn($wj) IB=$mface/$mhca rdzv=$mip; cleanup orphans"
    cleanup_orphans "$mj" "$wj" "$mn" "$wn"
    sleep 5
    if ! alloc_gpus_idle "$mn" "$mj" || ! alloc_gpus_idle "$wn" "$wj"; then
        log "RESUME-WAIT: my allocated GPUs not free after cleanup (master=$mn worker=$wn); retry"; return 1
    fi
    out=$(timeout 80 ssh "$mn" "cd $P && rm -f .aris/meta/.vqvae_h200_orch.pid && setsid nohup env JOB_A=$mj JOB_B=$wj MASTER_NODE=$mn WORKER_NODE=$wn RDZV_HOST=$mip NCCL_SOCKET_IFNAME=$mface NCCL_IB_HCA=$mhca ANYTOP_ROOT=$ANYTOP_ROOT MAX_COARSE=$MAX_COARSE MAX_JOINTS=$MAX_JOINTS MAX_FRAMES=$MAX_FRAMES BATCH_SIZE=$BATCH_SIZE LR=$LR NUM_CODES=$NUM_CODES RESUME_CKPT=last_model.pt OVERWRITE=0 OUT=$OUT_REL bash scripts/_launch_graph_vqvae_2node_h200.sh > $OUT/orch_resume_wd.log 2>&1 </dev/null & sleep 8; pid=\$(cat .aris/meta/.vqvae_h200_orch.pid 2>/dev/null || true); { [ -n \"\$pid\" ] && ps -p \"\$pid\" -o args= 2>/dev/null | grep -qF '_launch_graph_vqvae_2node_h200.sh'; } && echo STARTED || echo DIEDFAST" 2>/dev/null)
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
        log "vqvae DOWN (streak=$down_streak)"
        if [ "$down_streak" -ge 2 ]; then
            if resume; then down_streak=0; sleep 600; continue; fi
        fi
    fi
    sleep "$CHECK_SEC"
done
