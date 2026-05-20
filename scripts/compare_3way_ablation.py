#!/usr/bin/env python3
"""M1.5 3-way ablation comparison — reads 3 eval JSONs, emits comparison table.

Usage:
  # First, run eval on each variant's best_recon_model.pt:
  python scripts/eval_graph_vae.py --ckpt runs/m1_5_graph_vae_dynamic_seed42/best_recon_model.pt \
    --out runs/m1_5_graph_vae_dynamic_seed42/eval_best_recon.json --split val
  python scripts/eval_graph_vae.py --ckpt runs/m1_5_graph_vae_deterministic_seed42/best_recon_model.pt \
    --out runs/m1_5_graph_vae_deterministic_seed42/eval_best_recon.json --split val
  python scripts/eval_graph_vae.py --ckpt runs/m1_5_graph_vae_none_seed42/best_recon_model.pt \
    --out runs/m1_5_graph_vae_none_seed42/eval_best_recon.json --split val

  # Then compare:
  python scripts/compare_3way_ablation.py \
    --dynamic runs/m1_5_graph_vae_dynamic_seed42/eval_best_recon.json \
    --deterministic runs/m1_5_graph_vae_deterministic_seed42/eval_best_recon.json \
    --none runs/m1_5_graph_vae_none_seed42/eval_best_recon.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_eval(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dynamic", required=True)
    p.add_argument("--deterministic", required=True)
    p.add_argument("--none", required=True)
    p.add_argument("--out", default="runs/m1_5_3way_compare.json")
    args = p.parse_args()

    variants = {
        "dynamic": load_eval(args.dynamic),
        "deterministic": load_eval(args.deterministic),
        "none": load_eval(args.none),
    }

    # Sanity checks
    for name, ev in variants.items():
        if ev.get("pool_type") != name:
            print(f"[WARN] {name} JSON has pool_type={ev.get('pool_type')}", file=sys.stderr)

    # Overall comparison (macro pos_l1 is the canonical 3-way ranking metric)
    print(f"\n{'='*78}")
    print(f"=== 3-WAY ABLATION (n={variants['dynamic'].get('n_samples')}) ===")
    print(f"{'='*78}")
    print(f"\n{'Variant':<16}{'macro_pos':>12}{'macro_vel':>12}{'micro_pos':>12}{'micro_vel':>12}")
    print("-" * 64)
    for name in ("dynamic", "deterministic", "none"):
        ev = variants[name]
        macro = ev["overall_macro"]
        micro = ev["overall_micro"]
        print(f"{name:<16}{macro['pos_l1']:>12.4f}{macro['vel_l1']:>12.4f}"
              f"{micro['pos_l1']:>12.4f}{micro['vel_l1']:>12.4f}")

    # Find winner
    winners = {}
    for metric_set, key in [("macro", "overall_macro"), ("micro", "overall_micro")]:
        best_name = min(variants.keys(),
                       key=lambda n: variants[n][key]["pos_l1"])
        winners[metric_set] = (best_name, variants[best_name][key]["pos_l1"])

    print(f"\n{'='*78}")
    print("=== WINNER ===")
    print(f"{'='*78}")
    print(f"macro pos_l1 winner: {winners['macro'][0]} ({winners['macro'][1]:.4f})")
    print(f"micro pos_l1 winner: {winners['micro'][0]} ({winners['micro'][1]:.4f})")
    if winners["macro"][0] == winners["micro"][0]:
        print(f"\n→ CONSISTENT WINNER: {winners['macro'][0]}")
    else:
        print(f"\n→ DISAGREEMENT: macro={winners['macro'][0]} micro={winners['micro'][0]}")
        print(f"  (likely small-species bias; trust macro for fair multi-topology comparison)")

    # Per-species detail
    all_species = set()
    for ev in variants.values():
        all_species |= set(ev["per_species"].keys())

    print(f"\n{'='*78}")
    print("=== PER-SPECIES pos_l1 mean±std ===")
    print(f"{'='*78}")
    header = f"{'species':<16}{'n':>4}"
    for name in ("dynamic", "deterministic", "none"):
        header += f"{name:>18}"
    print(header)
    print("-" * len(header))
    species_rankings = {}
    for sid in sorted(all_species):
        row = f"{sid:<16}"
        n = None
        per_variant_pos = {}
        for name in ("dynamic", "deterministic", "none"):
            ps = variants[name]["per_species"].get(sid, None)
            if ps is None:
                row += f"{'—':>18}"
                per_variant_pos[name] = None
            else:
                n = ps["n"]
                m, s = ps["pos_l1_mean"], ps["pos_l1_std"]
                row += f"{m:>10.4f}±{s:<6.3f}"
                per_variant_pos[name] = m
        # Per-species winner
        non_none = {k: v for k, v in per_variant_pos.items() if v is not None}
        if len(non_none) >= 2:
            best = min(non_none, key=non_none.get)
            species_rankings[sid] = best
        row_n = f"{n if n else '?':>4}"
        print(f"{sid:<16}{row_n:>4} " + row[16:])

    # Per-species win counts
    print(f"\n=== PER-SPECIES WIN COUNT ===")
    win_count = {"dynamic": 0, "deterministic": 0, "none": 0}
    for sid, winner in species_rankings.items():
        win_count[winner] += 1
    total = len(species_rankings)
    for name in ("dynamic", "deterministic", "none"):
        print(f"  {name:<16} wins {win_count[name]:>3} / {total} species "
              f"({100*win_count[name]/total:>5.1f}%)")

    # Pool diagnostic comparison
    print(f"\n{'='*78}")
    print("=== POOL DIAGNOSTICS (active_coarse mean) ===")
    print(f"{'='*78}")
    for name in ("dynamic", "deterministic", "none"):
        diag = variants[name].get("pool_diag", {})
        print(f"  {name:<16} active_coarse mean={diag.get('active_coarse_mean')} "
              f"range=[{diag.get('active_coarse_min')}, {diag.get('active_coarse_max')}]")

    # Write comparison JSON
    out_data = {
        "winners": winners,
        "per_species_winner": species_rankings,
        "win_count": win_count,
        "summary": {
            name: {
                "macro_pos_l1": variants[name]["overall_macro"]["pos_l1"],
                "macro_vel_l1": variants[name]["overall_macro"]["vel_l1"],
                "micro_pos_l1": variants[name]["overall_micro"]["pos_l1"],
                "micro_vel_l1": variants[name]["overall_micro"]["vel_l1"],
                "active_coarse_mean": variants[name].get("pool_diag", {}).get("active_coarse_mean"),
            }
            for name in ("dynamic", "deterministic", "none")
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\n→ comparison JSON written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
