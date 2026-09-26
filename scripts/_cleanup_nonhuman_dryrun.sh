#!/bin/bash
# DRY-RUN enumerator for non-Human+animal run cleanup (NO rm here).
# Resolves batch1 (old non-human experiments) + batch2 (smoke/QA/bench cruft) globs to an
# EXPLICIT path list, HARD-excludes anything containing L4safeHuman/curric/humanml3d, and a
# PROTECTED allow-list (active mainline + the 3 Human+animal ablations + codebook_compare).
# Prints the explicit delete list + sizes + total, confirms PROTECTED untouched, and lists
# any leftover dirs matched by NEITHER batch (your call). Writes the explicit list to
# .aris/meta/delete_list_nonhuman.txt for a separate, human-verified rm step.
set -uo pipefail
RUNS=/scratch/ts1v23/workspace/noKslot_clean/runs
cd "$RUNS" || exit 1

PROTECTED=(
  vqvae_L4safeHuman_C72_J144_d512_Q4_n8192_b16g64_300ep_curric50to60_seed42
  vqvae_L4safeHuman_C72_J144_d512_Q4_n8192_b16g64_300ep_seed42
  codeflow_graph_pscf_L4safeHuman_n8192_b16g64_lr8e5_4xh200_seed42
  anytop_t2m_evaluator_distilbert_coemb512_gb128_lr1e-4_mfd12_l4human_seed42
  vqvae_L4safeHuman_C72_J144_d512_Q4_n4096_b32g128_300ep_seed42
  vqvae_L4safeHuman_C72_J144_d512_Q4_n128_b32g128_300ep_seed42
  vqvae_L4safeHuman_C72_J144_d512_Q4_n512_b32g128_300ep_seed42
  _codebook_compare_ep50
)
is_protected() { local d="$1" p; for p in "${PROTECTED[@]}"; do [ "$d" = "$p" ] && return 0; done; return 1; }

shopt -s nullglob
# batch1 = non-human experiments; batch2 = cruft. Evaluators by EXACT name (never glob l4human).
CANDS=(
  codeflow_graph_pscf_mergedL4TB_*/ codeflow_graph_pscf_L5_*/
  vqvae_L4safeTB_*/ vqvae_L5_*/
  anytop_t2m_evaluator_distilbert_coemb512_gb128_lr1e-4_seed42/
  anytop_t2m_evaluator_distilbert_coemb512_gb128_lr1e-4_mfd12_seed42/
  m1_*/ m2_*/ _archive_cont1_preretrain_*/ _exp_m1_l2_cleanL2_*/
  _smoke_*/ _bench_*/ _qa_*/ sanity_*/ diagnostics/ vqvae_smoke/ _eval_l4human_smoke/
)
declare -A inDel
DEL=()
for c in "${CANDS[@]}"; do
  d="${c%/}"; [ -d "$d" ] || continue
  [ -n "${inDel[$d]:-}" ] && continue
  if [[ "$d" == *L4safeHuman* || "$d" == *curric* || "$d" == *humanml3d* ]]; then continue; fi
  if is_protected "$d"; then continue; fi
  inDel[$d]=1; DEL+=("$d")
done

echo "=== DELETE CANDIDATES (${#DEL[@]} dirs, explicit) ==="
for d in "${DEL[@]}"; do du -sh "$d"; done | sort -h
echo "=== TOTAL to free ==="
du -sch "${DEL[@]}" 2>/dev/null | tail -1
echo
echo "=== PROTECTED (all must exist; NONE may be in the delete list) ==="
fail=0
for p in "${PROTECTED[@]}"; do
  if [ -n "${inDel[$p]:-}" ]; then echo "  !!! ERROR protected in delete list: $p"; fail=1; fi
  if [ -d "$p" ]; then echo "  keep: $p ($(du -sh "$p" 2>/dev/null|cut -f1))"; else echo "  (absent): $p"; fi
done
# extra safety: assert no delete path contains the human marker
for d in "${DEL[@]}"; do case "$d" in *L4safeHuman*|*curric*|*humanml3d*) echo "  !!! ERROR human-marker in delete: $d"; fail=1;; esac; done
echo "  GUARD: $([ $fail -eq 0 ] && echo PASS || echo FAIL)"
echo
echo "=== LEFTOVER dirs matched by NEITHER batch (your call — not in delete list) ==="
for x in */; do
  d="${x%/}"
  [ -n "${inDel[$d]:-}" ] && continue
  is_protected "$d" && continue
  du -sh "$d"
done | sort -h
printf '%s\n' "${DEL[@]}" > /scratch/ts1v23/workspace/noKslot_clean/.aris/meta/delete_list_nonhuman.txt
echo
echo "=== wrote explicit list -> .aris/meta/delete_list_nonhuman.txt (${#DEL[@]} paths) ==="
