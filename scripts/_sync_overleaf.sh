#!/bin/bash
# Mirror the paper sources to TWO places on every paper change (user 2026-09-10 standing authorisation;
# 2026-09-12: push straight to Overleaf as well, so the web project updates with no button to press).
#   1. the PUBLIC GitHub repo CHDTevior/TopX-ICLR-  (backup)
#   2. the Overleaf project's own git remote           (what the user reads)
# ALLOWLIST, not an exclude list: a file reaches either remote only by being named here. Build artifacts,
# internal notes, figure sources and every run/ artifact stay out.
#
# What the file checks guard against: a symlink, hard link, missing or replaced file under paper/ -- mistakes by
# the operator, who is the only writer of paper/ and of the two clones. They are not a defence against another
# process of the same user racing this script: such a process can simply write anything into a .tex file.
# Credentials never touch this file or a remote URL: the Overleaf token lives in ~/.config/overleaf/token
# (mode 600) and reaches git only through GIT_ASKPASS. The project id is in ~/.config/overleaf/project_id.
set -euo pipefail
# a local refs/replace/* object would redirect every commit id git resolves (reset, diff, ancestry, push);
# these clones must see the objects as they are (codex r20)
export GIT_NO_REPLACE_OBJECTS=1
SRC=${PAPER_SRC:-/iridisfs/scratch/ts1v23/workspace/noKslot_clean/paper}
GH_DIR=${OVERLEAF_DIR:-/iridisfs/scratch/ts1v23/workspace/overleaf_topx}
OL_DIR=${OVERLEAF_DIRECT_DIR:-/iridisfs/scratch/ts1v23/workspace/overleaf_direct}
CFG=${OVERLEAF_CFG:-$HOME/.config/overleaf}
MSG=${1:-"sync from the working repository"}

# every file the submission compiles from, and nothing else
FILES=(
  main.tex
  references.bib
  main.bbl
  math_commands.tex
  iclr2027_conference.sty
  iclr2027_conference.bst
  fancyhdr.sty
  natbib.sty
  Makefile
  sections/introduction.tex
  sections/related_work.tex
  sections/method.tex
  sections/experiments.tex
  sections/conclusion.tex
  sections/appendix.tex
  figures/fig1_align/figure1.pdf
  figures/qualitative.pdf
  figures/tb_filmstrips.pdf
  figures/tb_main.pdf
  figures/tb_world.pdf
  figures/model_details.png
)

# refuse to publish a paper that does not build here
[ -f "$SRC/main.pdf" ] || { echo "[sync] no main.pdf -- build the paper before syncing"; exit 1; }
# one sync at a time: two overlapping runs can interleave fetch/reset/guard/commit/push and record a far-side
# commit as published (codex 2026-09-12 r2 P1). The lock lives beside the clones on the shared filesystem.
exec 8>"$(dirname "$GH_DIR")/.sync_overleaf.lock"
flock -w 900 8 || { echo "[sync] another sync is still running after 15 min -- not starting a second one"; exit 1; }

# Everything below reads a PRIVATE SNAPSHOT of the allowlisted files, not $SRC: the helper opens every path
# component with O_NOFOLLOW, fstat's the descriptor it copies from (regular file, one hard link) and refuses
# otherwise, so a symlink or hard link anywhere under $SRC -- present, or swapped in while this runs -- cannot
# publish a file from outside the paper directory (codex r4 P1, r5 P1 #1/#2). Both mirrors compare against and
# copy from the snapshot. A missing file fails there too (a partial paper is never pushed).
SNAP=$(mktemp -d "$(dirname "$GH_DIR")/.sync_snapshot.XXXXXX") || { echo "[sync] cannot create the snapshot dir"; exit 1; }
trap 'rm -rf -- "$SNAP"' EXIT
python3 "$(dirname "$0")/_sync_snapshot.py" "$SRC" "$SNAP" "${FILES[@]}" || exit 1
# the figures main.tex includes must be exactly the ones on the list
want=$(printf '%s\n' "${FILES[@]}" | grep '^figures/' | sort)
have=$(grep -rhoE '\\includegraphics(\[[^]]*\])?\{[^}]+\}' "$SNAP"/sections/*.tex "$SNAP"/main.tex \
       | sed -E 's/.*\{(.*)\}/\1/' | sort -u)
if [ "$want" != "$have" ]; then
  echo "[sync] the figures main.tex includes differ from the allowlist -- update FILES in this script"
  diff <(echo "$want") <(echo "$have") || true; exit 1
fi

# mirror_into <clone dir> <label> <branch> <baseline dir|-> [env for git...]
#
# The clone is DISPOSABLE and the remote is the truth: every run resets the clone to origin/<branch>, so no
# local state can diverge from it (a rejected push leaves nothing behind). What the script must not do is
# revert an edit made on the far side, so it remembers the last commit it PUBLISHED in a local ref,
# refs/sync/published, and compares COMMITS, not working trees: if the remote moved past that ref and the
# move touched an allowlisted file, someone edited it over there -- refuse, name the files, and keep refusing
# until the working repository has absorbed the edit (the ref is only advanced on success or on a clean
# remote). A second local ref, refs/sync/pending, names the commit about to be pushed: a later run that finds
# that commit on the remote records it as published (a push that landed but was never recorded -- a crash, a
# failed ref write -- is recovered), and clears it otherwise (codex r8). On the very first run there is no
# published ref; the remote's allowlisted files must then equal what we last
# published to the OTHER mirror (<baseline dir>), or the run refuses. ACCEPT_REMOTE_AS_BASELINE=1 overrides
# that one check deliberately. (codex 2026-09-12: four reproduced failure modes of the previous version.)
mirror_into() {
  local dir=$1 label=$2 br=$3 base=$4; shift 4
  [ -d "$dir/.git" ] || { echo "[sync:$label] $dir is not a git clone"; return 1; }
  (
    cd "$dir" || exit 1
    say() { echo "[sync:$label] $*"; }
    # read a local ref: prints its hash; prints nothing when the ref does not exist; FAILS on any other outcome --
    # a git error, or a ref that IS on disk (loose file or packed-refs line) yet does not resolve, i.e. an
    # unreadable or corrupt ref, which rev-parse reports with the same exit 1 as absence (codex r16, r17)
    # The lookup must be EXACT: rev-parse also resolves a tag named refs/tags/refs/sync/<name> that a clone can
    # bring in from the far side (codex r18). rev-parse gives the hash (the exact ref wins when both exist);
    # `show-ref --verify` then confirms the exact ref exists -- it does no name expansion (it cannot replace
    # rev-parse outright: on this git it exits 128 for "absent" and for errors alike, which rounds 16-17 forbid).
    read_ref() { local h rc; h=$(git rev-parse -q --verify "$1" 2>/dev/null); rc=$?
                 if [ "$rc" -eq 0 ] && ! git show-ref --verify -q -- "$1"; then rc=1; fi   # resolved to another ref: OURS is absent
                 [ "$rc" -eq 0 ] && { printf '%s' "$h"; return 0; }
                 [ "$rc" -eq 1 ] || return "$rc"
                 if [ -e "$gd/$1" ] || { [ -f "$gd/packed-refs" ] && grep -Eq "^[0-9a-f]+ $1\$" "$gd/packed-refs"; }; then return 3; fi
                 return 0; }
    # a run killed inside git leaves *.lock files (index.lock, refs/**/*.lock, ...) and every later git call
    # fails with "File exists"; under the flock we are the only writer of this clone, so any lock here is stale
    # (codex r9, r10)
    local gd
    gd=$(git rev-parse --git-dir) || { say "not a git clone"; exit 1; }
    find "$gd" -name '*.lock' -type f -delete || { say "could not clear stale git locks"; exit 1; }
    rm -f -- "$gd/packed-refs.new" || { say "could not clear a stale packed-refs.new"; exit 1; }   # git's other temp file (codex r11)
    # pending recovery, first pass: against the tracking ref as this clone last saw it, BEFORE the fetch replaces
    # it. A push that succeeded updated origin/<branch> here; if the marker write then failed, the push is recorded
    # now, so a remote history rewritten afterwards is caught by the published-ancestry guard below instead of
    # being mistaken for a clean remote (codex r13). A push that failed on the way back did not update the
    # tracking ref and is handled by the second pass after the fetch.
    local pend anc
    pend=$(read_ref refs/sync/pending) || { say "could not read refs/sync/pending -- refusing"; exit 1; }
    local old_remote
    if [ -n "$pend" ] && old_remote=$(git show-ref --verify --hash -- "refs/remotes/origin/$br" 2>/dev/null); then
      anc=0; git merge-base --is-ancestor "$pend" "$old_remote" || anc=$?
      if [ "$anc" -eq 0 ]; then
        git update-ref refs/sync/published "$pend" || { say "could not record the recovered marker"; exit 1; }
        git update-ref -d refs/sync/pending || { say "could not clear refs/sync/pending"; exit 1; }
        say "recovered: the previous run's push $(git rev-parse --short "$pend") had reached the remote"
      elif [ "$anc" -ne 1 ]; then
        say "could not check the pending push against the tracking ref (git exit $anc) -- keeping refs/sync/pending; run again"; exit 1
      fi
    fi
    # exact refspec, no tags, and every later use of the remote's tip goes through the exact tracking ref's
    # commit id: the short name origin/<branch> would also match a tag of that name (codex r19)
    env "$@" git fetch -q --no-tags origin "+refs/heads/$br:refs/remotes/origin/$br" || { say "fetch failed"; exit 1; }
    local remote
    remote=$(git show-ref --verify --hash -- "refs/remotes/origin/$br") || { say "origin/$br has no commit"; exit 1; }
    git reset -q --hard "$remote" || { say "reset to origin/$br failed"; exit 1; }
    git clean -ffdxq || { say "could not clear untracked files"; exit 1; }
    # a symlink the remote tracks survives reset/clean and a copy onto it would write through it; a gitlink
    # (submodule entry) makes git add fail on the path (codex r5 P1 #3, r6)
    # (captured first, not piped into grep -q: with pipefail an early match would SIGPIPE git and the test would
    # read as "no match" -- codex r7 #2)
    local index
    index=$(git ls-files -s) || { say "git ls-files failed"; exit 1; }
    if grep -Eq '^(120000|160000) ' <<<"$index"; then
      say "the remote tracks a symlink or submodule ($(awk '$1=="120000"||$1=="160000"{print $4}' <<<"$index" | tr '\n' ' ')) -- refusing"; exit 1
    fi
    # this function runs in an || list, so set -e is off in here: every command must be checked explicitly
    local pub
    # second pass, against the fetched remote (a push that landed but returned failure)
    pend=$(read_ref refs/sync/pending) || { say "could not read refs/sync/pending -- refusing"; exit 1; }
    if [ -n "$pend" ]; then
      anc=0; git merge-base --is-ancestor "$pend" "$remote" || anc=$?
      if [ "$anc" -eq 0 ]; then
        git update-ref refs/sync/published "$pend" || { say "could not record the recovered marker"; exit 1; }
        say "recovered: the previous run's push $(git rev-parse --short "$pend") is on the remote"
      elif [ "$anc" -ne 1 ]; then
        # exit 1 means "not an ancestor"; anything else is an error, and pending must survive it (codex r10)
        say "could not check whether the pending push reached the remote (git exit $anc) -- keeping refs/sync/pending; run again"; exit 1
      elif [ "${SYNC_DROP_PENDING:-0}" != 1 ]; then
        # neither the tracking ref nor the fetched remote contains the commit a previous run pushed without a
        # verdict. Either it never landed (safe to forget) or the remote's history was rewritten after it did --
        # and that rewrite could hide an edit made there. The script cannot tell; the operator can (codex r14).
        say "a previous run pushed $(git rev-parse --short "$pend") without a verdict and the remote does not contain it."
        say "If the remote history was NOT rewritten since, run once with SYNC_DROP_PENDING=1 to forget that push; otherwise reconcile by hand."
        exit 1
      else
        say "forgetting the unconfirmed push $(git rev-parse --short "$pend") (SYNC_DROP_PENDING=1)"
      fi
      git update-ref -d refs/sync/pending || { say "could not clear refs/sync/pending"; exit 1; }
    fi
    pub=$(read_ref refs/sync/published) || { say "could not read refs/sync/published -- refusing"; exit 1; }
    if [ -z "$pub" ]; then
      if [ "$base" != "-" ] && [ "${ACCEPT_REMOTE_AS_BASELINE:-0}" != 1 ]; then
        local bad=()
        for f in "${FILES[@]}"; do
          if [ -e "$f" ] || [ -e "$base/$f" ]; then
            { [ -e "$f" ] && [ -e "$base/$f" ] && cmp -s "$f" "$base/$f"; } || bad+=("$f")
          fi
        done
        if [ ${#bad[@]} -gt 0 ]; then
          say "first run: the remote's copies of these files differ from what was last published, so they may"
          say "carry edits made on the far side; bring them into the working repository first, or run once with"
          say "ACCEPT_REMOTE_AS_BASELINE=1 to declare the remote the baseline knowingly:"
          for f in "${bad[@]}"; do say "    $f"; done
          exit 1
        fi
      fi
      pub=$remote
    elif [ "$pub" != "$remote" ]; then
      if ! git merge-base --is-ancestor "$pub" "$remote"; then
        say "the remote no longer contains the commit last published here -- resolve by hand"; exit 1
      fi
      # files edited on the far side since we last published, and NOT yet absorbed into SRC (the clone is at
      # the remote, so comparing the working file with SRC asks exactly "would a push revert this edit")
      local unabsorbed=() changed f
      changed=$(git diff --name-only --no-renames "$pub" "$remote" -- "${FILES[@]}") \
        || { say "could not diff $pub..$remote -- refusing rather than guessing"; exit 1; }
      while IFS= read -r f; do
        [ -n "$f" ] || continue
        # a file deleted on the far side is "unabsorbed" unless SRC has also dropped it
        if [ -e "$f" ]; then
          { [ -e "$SNAP/$f" ] && cmp -s "$f" "$SNAP/$f"; } || unabsorbed+=("$f")
        else
          [ ! -e "$SNAP/$f" ] || unabsorbed+=("$f")
        fi
      done <<< "$changed"
      if [ ${#unabsorbed[@]} -gt 0 ]; then
        say "allowlisted files were edited on the far side since the last publish; a push would revert them."
        say "Bring these into the working repository first (the run clears once SRC matches the remote copy):"
        for f in "${unabsorbed[@]}"; do say "    $f"; done
        exit 1
      fi
      pub=$remote     # the remote moved, but every allowlisted change is already in SRC (or outside the list)
    fi
    git update-ref refs/sync/published "$pub" || { say "could not record the publication marker -- refusing"; exit 1; }

    for f in "${FILES[@]}"; do
      mkdir -p -- "$(dirname "$f")" || { say "mkdir failed for $f"; exit 1; }
      cp -pT --remove-destination -- "$SNAP/$f" "$f" || { say "copy failed for $f (is something else in its place?)"; exit 1; }
    done
    git add -A -- "${FILES[@]}" || { say "git add failed"; git reset -q --hard "$remote"; exit 1; }
    if git diff --cached --quiet; then say "nothing changed"; exit 0; fi
    git -c user.name="Tengjiao Sun" -c user.email="tevior@outlook.com" commit -q -m "$MSG" \
      || { say "commit failed"; git reset -q --hard "$remote"; exit 1; }
    # the candidate is resolved ONCE to a commit id and that id is what gets pushed and recorded; the name HEAD
    # in a push refspec could match a tag called HEAD (codex r19)
    local cand
    cand=$(git rev-parse --verify -q 'HEAD^{commit}') || { say "could not resolve the new commit"; git reset -q --hard "$remote"; exit 1; }
    # record the candidate BEFORE pushing: if this run dies or cannot write the marker after the push, the next
    # run finds the commit on the remote and records it (codex r7 #4, r8 #1/#2)
    git update-ref refs/sync/pending "$cand" \
      || { say "could not record refs/sync/pending -- not pushing"; git reset -q --hard "$remote"; exit 1; }
    local pushout
    local rej
    rej=$(printf '^!\t[^\t:]*:refs/heads/%s\t\\[(remote )?rejected\\]' "$br")
    if ! pushout=$(LC_ALL=C env "$@" git push --porcelain --no-follow-tags origin "$cand:refs/heads/$br" 2>&1); then
      # only an explicit [rejected] / [remote rejected] on OUR BRANCH's row proves the push did not land: "!" alone
      # also covers "[remote failure] (remote failed to report status)", and another ref's rejection (a tag that
      # push.followTags would have sent -- hence --no-follow-tags) says nothing about the branch (codex r15, r16)
      if grep -Eq -- "$rej" <<<"$pushout"; then
        # the remote answered and refused the ref update: the push certainly did not land
        git update-ref -d refs/sync/pending || true
        say "push rejected by the remote -- the clone is reset to the remote; run again"
      else
        # no verdict from the remote (connection lost, receive-pack died): the push may have landed, so
        # refs/sync/pending is KEPT and the next run decides (codex r9, r14)
        say "push failed without a verdict from the remote -- refs/sync/pending is kept; run again"
      fi
      git reset -q --hard "$remote"; exit 1
    fi
    git update-ref refs/sync/published "$cand" \
      || { say "pushed: $(git log --oneline -1) -- but the marker could not be recorded; the next run recovers it from refs/sync/pending"; exit 1; }
    git update-ref -d refs/sync/pending \
      || { say "pushed and recorded: $(git log --oneline -1) -- refs/sync/pending could not be cleared; the next run clears it"; exit 1; }
    say "pushed: $(git log --oneline -1)"
  )
}

rc=0
# GitHub: nobody edits there, so on the first run the remote IS the baseline
# a GitHub refusal stops the run: its clone is the first-run baseline for Overleaf, and a refused clone
# may hold the very far-side edit the baseline check exists to protect (codex r4 P1)
mirror_into "$GH_DIR" github main - || exit 1

if [ -s "$CFG/token" ] && [ -s "$CFG/project_id" ]; then
  if [ -e "$OL_DIR" ] && ! git -C "$OL_DIR" rev-parse -q --verify HEAD >/dev/null 2>&1; then
    echo "[sync:overleaf] $OL_DIR exists but is not a usable clone (an interrupted clone?) -- remove it and run again"; exit 1
  fi
  if [ ! -d "$OL_DIR/.git" ]; then
    # clone beside the final path and move it there only once it is complete, so an interrupted clone can
    # never occupy $OL_DIR (codex r10)
    echo "[sync:overleaf] first run: cloning the project's git remote into $OL_DIR"
    tmp=$(mktemp -d "$(dirname "$OL_DIR")/.overleaf_clone.XXXXXX") || { echo "[sync:overleaf] cannot create a temp dir"; exit 1; }
    GIT_ASKPASS="$CFG/askpass" GIT_TERMINAL_PROMPT=0 \
      git clone -q "https://git@git.overleaf.com/$(cat "$CFG/project_id")" "$tmp/clone" \
      || { rm -rf -- "$tmp"; echo "[sync:overleaf] clone failed -- check the token and project id"; exit 1; }
    mv -T -- "$tmp/clone" "$OL_DIR" || { rm -rf -- "$tmp"; echo "[sync:overleaf] could not move the clone into place"; exit 1; }
    rmdir -- "$tmp"
  fi
  # Overleaf serves and accepts exactly one branch, master; the GitHub mirror is what we last published
  mirror_into "$OL_DIR" overleaf master "$GH_DIR" GIT_ASKPASS="$CFG/askpass" GIT_TERMINAL_PROMPT=0 || rc=1
else
  echo "[sync:overleaf] no $CFG/token or project_id -- Overleaf not pushed (GitHub only)"
fi
exit $rc