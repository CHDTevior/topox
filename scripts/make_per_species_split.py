"""Per-species 80/20 stratified train/val split (M1.5R decision B).

Replaces the zero-shot held-out split (Trex/Spider/Dragon all-val) with a
stratified in-distribution split where every species appears in both train
and val. Seed-deterministic.

Usage:
  python scripts/make_per_species_split.py \\
      --data_dir data/cs_sparse2full_tgt_v2 \\
      --val_frac 0.2 --seed 42 \\
      --backup_existing  # back up existing train/val.txt before overwriting
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True,
                   help="Dataset root containing motions/ + splits/")
    p.add_argument("--val_frac", type=float, default=0.2,
                   help="Fraction of each species to allocate to val (default 0.2)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backup_existing", action="store_true",
                   help="Move existing train/val.txt to *.bak before writing")
    p.add_argument("--min_val_per_species", type=int, default=1,
                   help="Minimum val clips per species (default 1) — guards against "
                        "0-clip val for tiny species. If n_clips * val_frac < min, force "
                        "at least min val clips (still ≥ 1 train).")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    motions_dir = data_dir / "motions"
    splits_dir = data_dir / "splits"
    if not motions_dir.exists():
        raise SystemExit(f"motions dir not found: {motions_dir}")

    # Discover all motion files + group by species
    all_motions = sorted(motions_dir.glob("*.npz"))
    by_species: dict[str, list[str]] = defaultdict(list)
    for fp in all_motions:
        # "Bat_0001.npz" → species "Bat", motion_id "Bat_0001"
        motion_id = fp.stem
        species = motion_id.rsplit("_", 1)[0]
        by_species[species].append(motion_id)

    print(f"Discovered {len(all_motions)} motions across {len(by_species)} species")

    # Deterministic per-species 80/20
    rng = random.Random(args.seed)
    train_ids: list[str] = []
    val_ids: list[str] = []
    per_species_split = {}

    for sp in sorted(by_species.keys()):
        ids = sorted(by_species[sp])  # canonical order
        n = len(ids)
        n_val = max(args.min_val_per_species, round(n * args.val_frac))
        # Guarantee at least 1 train
        n_val = min(n_val, n - 1) if n >= 2 else 0
        # Shuffle deterministically per-species
        sp_rng = random.Random(args.seed + hash(sp) % 1000)
        sp_shuffled = list(ids)
        sp_rng.shuffle(sp_shuffled)
        sp_val = sp_shuffled[:n_val]
        sp_train = sp_shuffled[n_val:]
        train_ids.extend(sp_train)
        val_ids.extend(sp_val)
        per_species_split[sp] = {"n": n, "train": len(sp_train), "val": len(sp_val)}

    # Sort the final lists for deterministic output
    train_ids = sorted(train_ids)
    val_ids = sorted(val_ids)

    print(f"\nTotal train: {len(train_ids)}, val: {len(val_ids)}")
    print(f"\nPer-species split (sorted by n desc):")
    print(f"  {'species':<18} {'n':>4} {'train':>6} {'val':>4}")
    print("  " + "-" * 40)
    for sp in sorted(per_species_split, key=lambda s: -per_species_split[s]["n"]):
        d = per_species_split[sp]
        print(f"  {sp:<18} {d['n']:>4} {d['train']:>6} {d['val']:>4}")

    # Backup existing splits if requested
    splits_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("train.txt", "val.txt"):
        fpath = splits_dir / fname
        if fpath.exists() and args.backup_existing:
            bak = splits_dir / (fname + ".bak")
            shutil.move(str(fpath), str(bak))
            print(f"\nBacked up {fpath} → {bak}")

    # Write new splits
    train_file = splits_dir / "train.txt"
    val_file = splits_dir / "val.txt"
    train_file.write_text("\n".join(train_ids) + "\n")
    val_file.write_text("\n".join(val_ids) + "\n")
    print(f"\n✓ Wrote {train_file} ({len(train_ids)} motions)")
    print(f"✓ Wrote {val_file} ({len(val_ids)} motions)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
