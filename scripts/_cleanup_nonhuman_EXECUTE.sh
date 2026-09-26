#!/bin/bash
# ACTUAL deletion executor — runs ONLY after human verification of the dry-run.
# Reads the explicit list .aris/meta/delete_list_nonhuman.txt and rm -rf each, but RE-APPLIES
# every safety guard at rm time (defense in depth): each path must be a real dir directly under
# runs/, must NOT contain L4safeHuman/curric/humanml3d, must NOT be in PROTECTED. If ANY entry
# fails a guard, NOTHING is deleted (abort). No wildcards in rm — explicit per-path.
set -uo pipefail
ROOT=/scratch/ts1v23/workspace/noKslot_clean
RUNS="$ROOT/runs"
LIST="$ROOT/.aris/meta/delete_list_nonhuman.txt"
[ -f "$LIST" ] || { echo "ABORT: no list $LIST"; exit 1; }

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

mapfile -t ITEMS < "$LIST"
echo "validating ${#ITEMS[@]} paths..."
fail=0; VALID=()
for d in "${ITEMS[@]}"; do
  [ -z "$d" ] && continue
  case "$d" in */*|.*|"") echo "REJECT (not a bare run name): '$d'"; fail=1; continue;; esac
  case "$d" in *L4safeHuman*|*curric*|*humanml3d*) echo "REJECT (human marker): $d"; fail=1; continue;; esac
  if is_protected "$d"; then echo "REJECT (protected): $d"; fail=1; continue; fi
  if [ ! -d "$RUNS/$d" ]; then echo "REJECT (missing): $d"; fail=1; continue; fi
  VALID+=("$d")
done
if [ "$fail" -ne 0 ]; then echo "=== ABORT: a guard failed, NOTHING deleted ==="; exit 1; fi
echo "all ${#VALID[@]} validated. deleting..."
n=0
for d in "${VALID[@]}"; do rm -rf -- "$RUNS/$d" && n=$((n+1)) && echo "  rm: $d"; done
echo "=== deleted $n dirs ==="
echo "runs/ now: $(du -sh "$RUNS" 2>/dev/null | cut -f1)"
