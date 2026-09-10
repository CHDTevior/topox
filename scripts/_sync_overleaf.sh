#!/bin/bash
# Mirror the paper sources into the Overleaf-backed GitHub repo (user 2026-09-10: sync on every paper change,
# standing authorisation). ALLOWLIST, not an exclude list: this pushes to a PUBLIC repository, so a file reaches
# it only by being named here. Build artifacts, internal notes, figure sources and every run/ artifact stay out.
set -euo pipefail
SRC=/iridisfs/scratch/ts1v23/workspace/noKslot_clean/paper
DST=${OVERLEAF_DIR:-/iridisfs/scratch/ts1v23/workspace/overleaf_topx}
MSG=${1:-"sync from the working repository"}

[ -d "$DST/.git" ] || { echo "[sync] $DST is not a git clone -- clone https://github.com/CHDTevior/TopX-ICLR-.git first"; exit 1; }

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
)

# refuse to publish a paper that does not build here
[ -f "$SRC/main.pdf" ] || { echo "[sync] no main.pdf -- build the paper before syncing"; exit 1; }
for f in "${FILES[@]}"; do
  [ -f "$SRC/$f" ] || { echo "[sync] missing $f -- refusing to push a partial paper"; exit 1; }
done

# the figures main.tex includes must be exactly the ones on the list
want=$(printf '%s\n' "${FILES[@]}" | grep '^figures/' | sort)
have=$(grep -rhoE '\\includegraphics(\[[^]]*\])?\{[^}]+\}' "$SRC"/sections/*.tex "$SRC"/main.tex \
       | sed -E 's/.*\{(.*)\}/\1/' | sort -u)
if [ "$want" != "$have" ]; then
  echo "[sync] the figures main.tex includes differ from the allowlist -- update FILES in this script"
  diff <(echo "$want") <(echo "$have") || true; exit 1
fi

for f in "${FILES[@]}"; do mkdir -p "$DST/$(dirname "$f")"; cp -p "$SRC/$f" "$DST/$f"; done

cd "$DST"
git add -A "${FILES[@]}"
if git diff --cached --quiet; then echo "[sync] nothing changed"; exit 0; fi
git -c user.name="Tengjiao Sun" -c user.email="tevior@outlook.com" commit -q -m "$MSG"
git push -q origin main
echo "[sync] pushed: $(git log --oneline -1)"
