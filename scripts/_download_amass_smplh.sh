#!/usr/bin/env bash
# Download AMASS SMPL+H (gender_specific, mosh_results) per-dataset tarballs used by HumanML3D,
# for the human real-twist recovery pipeline (see handoff/20260701_235711_human_twist_recovery_v4_implementation.md).
#
# WHY SMPL+H gender_specific: HumanML3D was built from the SMPL+H `*_poses.npz` release (poses[:, :66] =
# root + 21 body joints axis-angle -> real per-joint twist). The 2024 re-release names files by new
# short names (e.g. PosePrior.tar.bz2) but the tarball's INTERNAL top-level dir uses the original
# HumanML3D name (e.g. MPI_Limits/), so extracted folder names align with amass_annotations.json.
#
# MECHANISM (empirically verified 2026-07-02): download.is.tue.mpg.de accepts credentials POSTed to
# download.php; the 302 sets a download-domain PHPSESSID that authorizes subsequent GETs. So we
# authenticate ONCE then fetch every file with the session cookie only. On session expiry a fetch
# returns an HTML login page (not "BZh") -> re-auth.
#
# CREDENTIALS: read ONLY as file paths AMASS_USER_FILE / AMASS_PASS_FILE (each a chmod-600 file with
# NO trailing newline). curl reads the values via --data-urlencode @file (kept out of argv/`ps`). The
# password is NEVER placed in this script or in the environment (env is readable via /proc/<pid>/environ
# for the whole durable run). The launcher creates + owns + shreds the cred files.
#
# SAFETY: non-destructive + atomic. Every subset is extracted to a temp dir, VERIFIED (pose count ==
# the tarball's pose count), then atomically published to dest ONLY if dest is absent or is an
# incomplete dir this script itself created (tracked by a .done marker). A pre-existing foreign dir is
# never replaced. KIT is special-cased: the local motion_data/KIT/ is SMPL-X and is preserved; SMPL+H
# KIT publishes to KIT_smplh/. IDEMPOTENT: .done records "dest|pose_count"; resume + count-match make
# re-runs safe and cheap.
#
# Usage:
#   AMASS_USER_FILE=/path/user AMASS_PASS_FILE=/path/pass bash scripts/_download_amass_smplh.sh [subset ...]
#   (no args -> the default 16-subset list; CMU/EKUT skipped, already SMPL+H locally)

set -uo pipefail

# ---------------------------------------------------------------------------- paths
AMASS_ROOT="${AMASS_ROOT:-/iridisfs/scratch/ts1v23/workspace/motion-latent-diffusion-main/datasets/amass}"
MOTION_DIR="$AMASS_ROOT/motion_data"
STAGE="$AMASS_ROOT/_smplh_tarballs"
STATUS_DIR="/scratch/ts1v23/workspace/noKslot_clean/.aris/meta"
STATUS_FILE="$STATUS_DIR/amass_download_status"
LOCK="$STATUS_DIR/.amass_download.lock"
LOG="$STATUS_DIR/amass_download.log"
COOKIE="$STAGE/.dl_cookies.txt"
BASE_URL="https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplh/gender_specific/mosh_results"
LOGIN_PROBE_SUBSET="PosePrior"           # small file used to (re)establish the session

mkdir -p "$MOTION_DIR" "$STAGE" "$STATUS_DIR"

# ---------------------------------------------------------------------------- single instance
exec 9>"$LOCK"
if ! flock -n 9; then echo "another amass download is already running; exiting" ; exit 0 ; fi

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG" ; }
status() {  # atomic one-line status
  local tmp="$STATUS_FILE.$$"
  printf '%s | amass_smplh_download | %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" > "$tmp" && mv "$tmp" "$STATUS_FILE"
}
count_poses() { find "$1" -name '*_poses.npz' 2>/dev/null | wc -l ; }

# ---------------------------------------------------------------------------- credentials (file paths only)
: "${AMASS_USER_FILE:?set AMASS_USER_FILE to a chmod-600 file (no trailing newline)}"
: "${AMASS_PASS_FILE:?set AMASS_PASS_FILE to a chmod-600 file (no trailing newline)}"
[ -r "$AMASS_USER_FILE" ] && [ -r "$AMASS_PASS_FILE" ] || { echo "credential files unreadable"; exit 1; }
for f in "$AMASS_USER_FILE" "$AMASS_PASS_FILE"; do
  p=$(stat -c '%a' "$f" 2>/dev/null)
  [ "$p" = "600" ] || { echo "credential file $f must be chmod 600 (is ${p:-?})"; exit 1; }
done
touch "$COOKIE"; chmod 600 "$COOKIE"

# ---------------------------------------------------------------------------- auth + fetch
amass_login() {   # POST creds -> download-domain session cookie. Returns curl rc; logs failures.
  local rc
  curl -fsS --show-error --max-time 120 -c "$COOKIE" -b "$COOKIE" \
    --data-urlencode "username@$AMASS_USER_FILE" \
    --data-urlencode "password@$AMASS_PASS_FILE" \
    "$BASE_URL/$LOGIN_PROBE_SUBSET.tar.bz2" -o /dev/null 2>>"$LOG"; rc=$?
  [ "$rc" -eq 0 ] || log "  amass_login: curl rc=$rc (see log)"
  return "$rc"
}

# fetch <subset> <outfile>: resume-capable; HTML(login) vs truncated-bz2 distinguished; integrity-gated.
fetch() {
  local subset="$1" out="$2" url="$BASE_URL/$1.tar.bz2" try rc prev cur
  if [ -s "$out" ] && [ "$(head -c 3 "$out" 2>/dev/null)" = "BZh" ] && bzip2 -t "$out" 2>/dev/null; then
    return 0   # already complete + valid
  fi
  for try in 1 2 3 4 5 6; do
    prev=0; [ -f "$out" ] && prev=$(stat -c%s "$out" 2>/dev/null || echo 0)
    curl -sS --max-time 7200 --retry 3 --retry-delay 10 -b "$COOKIE" -c "$COOKIE" -C - "$url" -o "$out" 2>>"$LOG"; rc=$?
    if [ ! -s "$out" ]; then
      log "  $subset: empty response (attempt $try, curl rc=$rc) -> re-auth"; amass_login || true; continue
    fi
    if [ "$(head -c 3 "$out" 2>/dev/null)" != "BZh" ]; then
      log "  $subset: non-bz2 response (attempt $try, curl rc=$rc) -> discard + re-auth"
      rm -f "$out"; amass_login || true; continue
    fi
    if bzip2 -t "$out" 2>/dev/null; then return 0 ; fi          # complete + valid
    cur=$(stat -c%s "$out" 2>/dev/null || echo 0)
    if [ "$cur" -le "$prev" ]; then
      log "  $subset: bz2 truncated, NO progress (attempt $try, curl rc=$rc) -> re-auth"; amass_login || true
    else
      log "  $subset: bz2 truncated, progressed $prev->$cur (attempt $try, curl rc=$rc) -> resume"
    fi
  done
  rm -f "$out"   # give up on this partial so the next script run starts clean
  return 1
}

# ---------------------------------------------------------------------------- subset list
if [ "$#" -gt 0 ]; then
  SUBSETS=("$@")
else
  # 16 subsets = HumanML3D's 18 AMASS subsets minus CMU/EKUT (already SMPL+H locally); KIT re-fetched
  # in SMPL+H to replace the local SMPL-X release.
  SUBSETS=(ACCAD BMLhandball BMLmovi BMLrub DFaust EyesJapanDataset HumanEva HDM05 \
           MoSh PosePrior SFU SSM TCDHands TotalCapture Transitions KIT)
fi

log "=== AMASS SMPL+H download start: ${#SUBSETS[@]} subset(s): ${SUBSETS[*]} ==="
amass_login || log "initial amass_login failed; will retry per-subset"
ok=0; fail=0; skip=0; i=0
for subset in "${SUBSETS[@]}"; do
  i=$((i+1))
  status "[$i/${#SUBSETS[@]}] $subset : working (ok=$ok skip=$skip fail=$fail)"
  done_marker="$STAGE/$subset.done"
  tarball="$STAGE/$subset.tar.bz2"

  # pre-fetch honor: skip only if the marker's dest still holds the exact recorded pose count
  if [ -f "$done_marker" ]; then
    IFS='|' read -r md me < "$done_marker"
    if [ -n "${md:-}" ] && [[ "${me:-}" =~ ^[0-9]+$ ]] && [ -d "$md" ] && [ "$me" -ge 1 ] && [ "$(count_poses "$md")" -eq "$me" ]; then
      log "[$i/${#SUBSETS[@]}] $subset: already done ($md, $me poses), skip"; skip=$((skip+1)); continue
    fi
    log "[$i/${#SUBSETS[@]}] $subset: stale .done -> redoing"; rm -f "$done_marker"
  fi

  log "[$i/${#SUBSETS[@]}] $subset: downloading -> $tarball"
  if ! fetch "$subset" "$tarball"; then
    log "[$i/${#SUBSETS[@]}] $subset: DOWNLOAD FAILED after retries"; fail=$((fail+1)); continue
  fi

  # authoritative expected pose count = poses inside the (integrity-verified) tarball
  expected=$(tar -tjf "$tarball" 2>/dev/null | grep -c '_poses\.npz$')
  if [ "${expected:-0}" -lt 1 ]; then
    log "[$i/${#SUBSETS[@]}] $subset: tarball has 0 *_poses.npz -> fail"; fail=$((fail+1)); continue
  fi
  log "[$i/${#SUBSETS[@]}] $subset: integrity OK ($(du -h "$tarball" 2>/dev/null | cut -f1), expect $expected poses)"

  # detect the tarball's internal top-level dir; require EXACTLY ONE non-license dir
  mapfile -t topdirs < <(tar -tjf "$tarball" 2>/dev/null | cut -d/ -f1 | grep -v -e '^LICENSE' -e '^$' | sort -u)
  if [ "${#topdirs[@]}" -ne 1 ]; then
    log "[$i/${#SUBSETS[@]}] $subset: expected 1 top dir, found ${#topdirs[@]} (${topdirs[*]:-none}) -> skip"; fail=$((fail+1)); continue
  fi
  topdir="${topdirs[0]}"

  # KIT SMPL+H must NOT clobber the existing SMPL-X KIT/
  if [ "$subset" = "KIT" ]; then dest_name="KIT_smplh"; else dest_name="$topdir"; fi
  dest="$MOTION_DIR/$dest_name"

  # already complete on disk?
  if [ -d "$dest" ] && [ "$(count_poses "$dest")" -eq "$expected" ]; then
    log "[$i/${#SUBSETS[@]}] $subset: dest $dest already complete ($expected poses) -> marking done"
    printf '%s|%s' "$dest" "$expected" > "$done_marker"; skip=$((skip+1)); continue
  fi

  # extract to a unique temp dir, verify count, then publish atomically
  tmpx="$STAGE/_x_$subset"; rm -rf "$tmpx"; mkdir -p "$tmpx"
  if ! tar -xjf "$tarball" -C "$tmpx" "$topdir" 2>>"$LOG"; then
    log "[$i/${#SUBSETS[@]}] $subset: EXTRACT FAILED -> fail"; rm -rf "$tmpx"; fail=$((fail+1)); continue
  fi
  tcount=$(count_poses "$tmpx/$topdir")
  if [ "$tcount" -ne "$expected" ]; then
    log "[$i/${#SUBSETS[@]}] $subset: extracted $tcount != expected $expected -> fail"; rm -rf "$tmpx"; fail=$((fail+1)); continue
  fi

  if [ ! -e "$dest" ]; then
    mv -T "$tmpx/$topdir" "$dest" 2>>"$LOG"
  elif [ -f "$done_marker" ]; then
    # dest is an incomplete dir THIS script created (has our marker) -> safe to replace
    log "[$i/${#SUBSETS[@]}] $subset: replacing our own incomplete $dest"
    rm -rf "$dest" && mv -T "$tmpx/$topdir" "$dest" 2>>"$LOG"
  else
    log "[$i/${#SUBSETS[@]}] $subset: dest $dest exists (foreign, incomplete) -> NOT replacing; manual check"; rm -rf "$tmpx"; fail=$((fail+1)); continue
  fi
  rm -rf "$tmpx"

  if [ -d "$dest" ] && [ "$(count_poses "$dest")" -eq "$expected" ]; then
    log "[$i/${#SUBSETS[@]}] $subset: OK -> $dest ($expected *_poses.npz files)"
    printf '%s|%s' "$dest" "$expected" > "$done_marker"; ok=$((ok+1))
  else
    log "[$i/${#SUBSETS[@]}] $subset: publish mismatch (dest has $(count_poses "$dest"), expected $expected) -> fail"; fail=$((fail+1))
  fi
done

status "DONE ok=$ok skip=$skip fail=$fail of ${#SUBSETS[@]}"
log "=== AMASS SMPL+H download finished: ok=$ok skip=$skip fail=$fail of ${#SUBSETS[@]} ==="
[ "$fail" -eq 0 ]
