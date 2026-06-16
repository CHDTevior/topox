#!/usr/bin/env bash
# DURABLE A100-backbone auto-resume watchdog for the 2048-codebook graph_pscf run
# (8xA100 cross-node DDP, swarm_a100 partition). Sibling of _watchdog_h200_backbone.sh,
# with codex-driven hardening (thread 019ecc8d). Runs setsid nohup on swarmh1002 (PPID=1).
#
# The 2048-backbone runs on TWO swarm_a100 gpu:4 nodes (currently swarma1003=976853 master
# + swarma1004=974143 worker; 974143 expires ~06:00 UTC). On expiry the cross-node DDP dies;
# this watchdog auto-resumes onto two ELIGIBLE+IDLE swarm_a100 gpu:4 allocs of mine.
#
# DESIGN:
#  * alive() PROC-BASED: a node is "up" if it carries >=4 of THIS run's train procs (the 4
#    local ranks); healthy = >=2 such nodes. Pause-safe + tolerant of extra idle allocs.
#  * resume() SELECTS the two targets FIRST (a node carrying THIS run's procs [survivor/wedged]
#    OR idle [fresh], AND no foreign codeflow proc), requires EXACTLY 2, rechecks alive(), THEN
#    cleans ONLY those two (frees a wedged survivor). master = lexicographically-first node.
#    Selecting before cleanup means a 3rd fresh alloc during a transient false-down -> cnt=3 ->
#    RESUME-WAIT, no cleanup -> a healthy 2-node run is never killed. Never guesses (cnt!=2 -> wait).
#  * launch probe runs IN $P (cd is foreground, only setsid is backgrounded) and POLLS up to
#    40s for the launcher pidfile+PID (fixes the H200 single-8s false-DIEDFAST).
# NEVER scancel; kill only THIS run's scoped train procs + exact-jobid srun + the recorded
# orchestrator PID. flock single-instance (fd 8, coexists with H200 wd fd9 + launcher fd9).
set -uo pipefail
P=/scratch/ts1v23/workspace/noKslot_clean
OUT_REL=runs/codeflow_graph_pscf_mergedL4TB_n2048_b8_lr8e5_8xa100_seed42
OUT="$P/$OUT_REL"
LOG="$P/.aris/meta/watchdog_a100.log"
PIDF_REL=.aris/meta/.gpscf_a100_orch.pid
CHECK_SEC="${CHECK_SEC:-300}"
CTRL_NODE="${CTRL_NODE:-swarmh1002}"
mkdir -p "$P/.aris/meta"

exec 8>"$P/.aris/meta/.watchdog_a100.lock"
flock -n 8 || { echo "[wd-a100] already running, exit"; exit 0; }

ts() { date -u +%FT%TZ; }
log() { echo "[wd-a100 $(ts)] $*" >> "$LOG"; }

log "START a100 watchdog (CHECK_SEC=$CHECK_SEC, out=$OUT_REL) pid=$$ host=$(hostname)"
down_streak=0

# "jobid node" lines for my RUNNING swarm_a100 gpu:4 allocs (0 lines if none/unreachable)
a100_allocs() {
    timeout 25 ssh "$CTRL_NODE" "squeue -u ts1v23 -t RUNNING -h -o '%i|%P|%N|%b' 2>/dev/null" </dev/null 2>/dev/null \
      | awk -F'|' '$2=="swarm_a100" && $4 ~ /:4$/ {print $1" "$3}'
}

# count of THIS run's train procs on node $1 (scoped to OUT_REL). Retries once on ssh FAILURE
# (empty/non-numeric) so a slow login to a heavily-loaded node isn't misread as 0/down; a genuine
# numeric "0" (node truly has none) returns immediately (no retry). timeout 40 for loaded swarma.
procs_on() {
    local n a
    for a in 1 2; do
        n=$(timeout 40 ssh "$1" "pgrep -u ts1v23 -fc '[t]rain_graph_codeflow.*$OUT_REL' 2>/dev/null || true" </dev/null 2>/dev/null)
        [[ "$n" =~ ^[0-9]+$ ]] && { echo "$n"; return; }
        sleep 2
    done
    echo 0
}
# count of OTHER (different OUT_REL) codeflow train procs on node $1 -> eligibility guard.
# IMPORTANT: only used to EXCLUDE candidates -> on ssh failure (non-numeric) default to a
# NON-ZERO sentinel so an unreachable node is treated as ineligible (never grab on uncertainty).
foreign_codeflow_on() {
    local n
    n=$(timeout 40 ssh "$1" "pgrep -u ts1v23 -af '[t]rain_graph_codeflow' 2>/dev/null | grep -vF '$OUT_REL' | grep -c . || true" </dev/null 2>/dev/null)
    [[ "$n" =~ ^[0-9]+$ ]] || n=9
    echo "$n"
}
alive() {  # healthy = >=2 of my swarm_a100 nodes each carry >=4 of THIS run's train procs
    local allocs node nodes_up=0
    allocs=$(a100_allocs) || return 1
    [ -n "$allocs" ] || return 1
    while read -r _jid node; do
        [ -n "$node" ] || continue
        [ "$(procs_on "$node")" -ge 4 ] && nodes_up=$((nodes_up+1))
    done <<<"$allocs"
    [ "$nodes_up" -ge 2 ]
}

alloc_gpus_idle() {  # my 4 GPUs in alloc $1 idle (<500 MiB total) via srun --jobid; fail -> not idle
    local used
    used=$(timeout 45 ssh "$CTRL_NODE" "srun --jobid=$1 --overlap --quiet nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd+ | bc" </dev/null 2>/dev/null || echo 9999)
    [ "${used:-9999}" -lt 500 ]
}

net22() { local o1 o2 o3 o4; IFS=. read -r o1 o2 o3 o4 <<<"$1"; [ -n "$o3" ] || { echo x; return; }; echo "$o1.$o2.$(( (o3/4)*4 )).0"; }

ib_info() {  # UP IB iface on node $1 -> "iface hca ip", else empty
    timeout 30 ssh "$1" '
        for i in $(ls /sys/class/net 2>/dev/null | grep -E "^ib[0-9]+$"); do
            st=$(cat /sys/class/net/$i/operstate 2>/dev/null)
            ip=$(ip -o -4 addr show "$i" 2>/dev/null | grep -oE "10\.[0-9.]+" | head -1)
            if [ "$st" = up ] && [ -n "$ip" ]; then
                hca=$(ls /sys/class/net/$i/device/infiniband/ 2>/dev/null | head -1)
                [ -n "$hca" ] && { echo "$i $hca $ip"; break; }
            fi
        done' </dev/null 2>/dev/null
}

# kill THIS run's scoped orphans on node $1 (jobid $2): scoped train procs + recorded orch PID + exact-jobid srun
cleanup_node() {
    local n="$1" jid="$2"
    timeout 25 ssh "$n" "pkill -u ts1v23 -9 -f \"[t]rain_graph_codeflow.*$OUT_REL\" 2>/dev/null || true; p=\$(cat $PIDF_REL 2>/dev/null || true); [ -n \"\$p\" ] && ps -p \"\$p\" -o args= 2>/dev/null | grep -qF '_launch_graph_pscf_2node_a100.sh' && kill \"\$p\" 2>/dev/null || true" 2>/dev/null || true
    [[ "$jid" =~ ^[0-9]+$ ]] && timeout 25 ssh "$n" "pkill -u ts1v23 -f '[s]run .*--jobid(=| )${jid}([^0-9]|\$)' 2>/dev/null || true" 2>/dev/null || true
}

resume() {
    local allocs cand cnt mn mj wn wj mp wp jid n p mib wib mface mhca mip rest wface wrest whca wip out
    allocs=$(a100_allocs) || { log "RESUME-WAIT: squeue unreachable; retry"; return 1; }
    [ -n "$allocs" ] || { log "RESUME-WAIT: no swarm_a100 gpu:4 allocs of mine; retry"; return 1; }
    # Select the TWO resume targets BEFORE any cleanup: a node is a target if it carries THIS
    # run's procs (survivor/wedged) OR is idle (fresh replacement), AND has no foreign codeflow.
    # Requiring EXACTLY 2 here means a 3rd fresh alloc during a transient false-down -> cnt=3 ->
    # no cleanup -> we never kill a healthy 2-node run, then strand it.
    # record each candidate's proc count (3rd field) so we can authoritatively cancel below
    # if both targets are proc-healthy (avoids relying on yet another flaky alive() call).
    cand=$(while read -r jid n; do
             [ -n "$n" ] || continue
             [ "$(foreign_codeflow_on "$n")" -eq 0 ] || continue
             p=$(procs_on "$n")
             if [ "$p" -gt 0 ] || alloc_gpus_idle "$jid"; then echo "$n $jid $p"; fi
           done <<<"$allocs" | sort)
    cnt=$(printf '%s\n' "$cand" | grep -c .)
    if [ "$cnt" -ne 2 ]; then
        log "RESUME-WAIT: need EXACTLY TWO target allocs (proc-bearing-survivor OR idle, no foreign); have $cnt (cand='$(echo $cand)'); retry"; return 1
    fi
    mn=$(printf '%s\n' "$cand" | sed -n '1p' | awk '{print $1}'); mj=$(printf '%s\n' "$cand" | sed -n '1p' | awk '{print $2}'); mp=$(printf '%s\n' "$cand" | sed -n '1p' | awk '{print $3}')
    wn=$(printf '%s\n' "$cand" | sed -n '2p' | awk '{print $1}'); wj=$(printf '%s\n' "$cand" | sed -n '2p' | awk '{print $2}'); wp=$(printf '%s\n' "$cand" | sed -n '2p' | awk '{print $3}')
    [[ "$mn" =~ ^swarma100[1-9]$ ]] && [[ "$wn" =~ ^swarma100[1-9]$ ]] || { log "RESUME-WAIT: node names unexpected (m=$mn w=$wn); retry"; return 1; }
    # AUTHORITATIVE cancel: if BOTH targets were proc-healthy (>=4) at selection, the run is alive
    # and the alive()-flake that triggered resume was spurious -> do NOT cleanup (uses the proc
    # counts just gathered, not another flaky alive() call).
    if [ "${mp:-0}" -ge 4 ] && [ "${wp:-0}" -ge 4 ]; then
        log "RESUME-CANCEL: both targets proc-healthy at selection (m=$mp w=$wp); run alive, no cleanup"; return 1
    fi
    [ -f "$OUT/last_model.pt" ] || { log "RESUME-ABORT: no last_model.pt yet; retry"; return 1; }
    # extra guard: final alive() recheck before destructive cleanup
    if alive; then log "RESUME-CANCEL: alive on pre-cleanup recheck; no cleanup"; return 1; fi
    # cleanup ONLY the two selected target nodes (frees a wedged survivor), then require both idle
    log "RESUME: targets master=$mn($mj) worker=$wn($wj); cleanup these two then launch"
    cleanup_node "$mn" "$mj"; cleanup_node "$wn" "$wj"
    sleep 5
    if ! alloc_gpus_idle "$mj" || ! alloc_gpus_idle "$wj"; then
        log "RESUME-WAIT: targets not idle after cleanup (m=$mn w=$wn); retry"; return 1
    fi
    mib=$(ib_info "$mn"); wib=$(ib_info "$wn")
    [ -n "$mib" ] && [ -n "$wib" ] || { log "RESUME-WAIT: cannot detect UP IB (m='$mib' w='$wib'); retry"; return 1; }
    mface=${mib%% *}; rest=${mib#* }; mhca=${rest%% *}; mip=${rest##* }
    wface=${wib%% *}; wrest=${wib#* }; whca=${wrest%% *}; wip=${wrest##* }
    if [ "$mface" != "$wface" ] || [ "$mhca" != "$whca" ]; then
        log "RESUME-WAIT: IB mismatch master($mn=$mface/$mhca) vs worker($wn=$wface/$whca); refuse; retry"; return 1
    fi
    if [ "$(net22 "$mip")" != "$(net22 "$wip")" ] || [ "$(net22 "$mip")" = "x" ]; then
        log "RESUME-WAIT: IB subnet mismatch master $mip vs worker $wip; refuse; retry"; return 1
    fi
    log "RESUME: master=$mn($mj) worker=$wn($wj) IB=$mface/$mhca rdzv=$mip; launching"
    # launch IN $P (cd foreground; only setsid backgrounded) + POLL up to 40s for the launcher pidfile PID.
    out=$(timeout 90 ssh "$mn" "cd $P || exit 10; rm -f $PIDF_REL; setsid nohup env JOB_A=$mj JOB_B=$wj MASTER_NODE=$mn WORKER_NODE=$wn RDZV_HOST=$mip NCCL_SOCKET_IFNAME=$mface NCCL_IB_HCA=$mhca BATCH_SIZE=8 LR=8e-5 RESUME_CKPT=last_model.pt OVERWRITE=0 OUT=$OUT_REL bash scripts/_launch_graph_pscf_2node_a100.sh > $OUT/orch_resume_wd.log 2>&1 </dev/null & ok=DIEDFAST; for i in 1 2 3 4 5 6 7 8; do sleep 5; pid=\$(cat $PIDF_REL 2>/dev/null || true); if [ -n \"\$pid\" ] && ps -p \"\$pid\" -o args= 2>/dev/null | grep -qF '_launch_graph_pscf_2node_a100.sh'; then ok=STARTED; break; fi; done; echo \$ok" 2>/dev/null)
    if [ "$out" = STARTED ]; then
        log "RESUME launched OK on $mn (orchestrator PID alive); sleep 600 before next check"; return 0
    else
        log "RESUME-LAUNCH-FAIL on $mn (probe='${out:-<ssh-fail>}'; see orch_resume_wd.log); retry next cycle"; return 1
    fi
}

while true; do
    if alive; then
        down_streak=0
    else
        down_streak=$((down_streak+1))
        log "2048-backbone DOWN (streak=$down_streak)"
        if [ "$down_streak" -ge 2 ]; then
            if alive; then log "RESUME-CANCEL: alive on recheck; no cleanup"; down_streak=0; sleep "$CHECK_SEC"; continue; fi
            if resume; then down_streak=0; sleep 600; continue; fi
        fi
    fi
    sleep "$CHECK_SEC"
done
