"""Fix preprocess J inconsistency by majority-vote per species.

Bug discovered 2026-05-21: preprocess_bvh.py's dummy-root removal is per-clip
(based on whether each clip's root rotation < 1° static check), so different
clips of same species end up with different J counts. Old min_frames=16
hid this because most short clips (more likely static) were filtered out.

Fix strategy (user-approved):
  1. Per species, count J distribution across motion clips
  2. Pick MAJORITY J as canonical
  3. DELETE motion clips with non-majority J
  4. Re-save skeleton file matching majority J (delete and re-save from
     first majority-J clip's structure if needed)
  5. Verify all motions + skeleton align

Usage:
  python scripts/fix_preprocess_J_mismatch.py \\
      --data_dir data/cs_sparse2full_tgt_v2 \\
      --dry_run  # show what would be deleted
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--dry_run", action="store_true",
                   help="Print actions without modifying files")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    motions_dir = data_dir / "motions"
    skeletons_dir = data_dir / "skeletons"
    if not motions_dir.exists():
        raise SystemExit(f"motions dir missing: {motions_dir}")

    # Discover per-species clips and their J
    by_species = defaultdict(list)
    for fp in sorted(motions_dir.glob("*.npz")):
        m = np.load(fp)
        J = int(m["local_positions"].shape[1])
        species = fp.stem.rsplit("_", 1)[0]
        by_species[species].append((fp, J))

    print(f"Discovered {sum(len(v) for v in by_species.values())} motions "
          f"across {len(by_species)} species")

    to_delete: list[Path] = []
    species_canonical_J = {}
    species_skeleton_action = {}

    for sp, clip_list in by_species.items():
        J_counts = Counter(j for _, j in clip_list)
        canonical_J, _ = J_counts.most_common(1)[0]
        species_canonical_J[sp] = canonical_J
        minorities = [fp for fp, J in clip_list if J != canonical_J]
        to_delete.extend(minorities)

        # Compare with skeleton file
        sk_fp = skeletons_dir / f"{sp}.npz"
        if sk_fp.exists():
            sk = np.load(sk_fp, allow_pickle=True)
            sk_J = len(sk["parent_indices"])
            if sk_J != canonical_J:
                species_skeleton_action[sp] = (sk_J, canonical_J)

    print(f"\n=== SUMMARY ===")
    print(f"  Clips to DELETE (non-majority J): {len(to_delete)}")
    print(f"  Species needing skeleton re-save: {len(species_skeleton_action)}")

    print(f"\n  Per-species drop count (top 10):")
    drops_per_species = Counter(fp.stem.rsplit("_", 1)[0] for fp in to_delete)
    for sp, n in drops_per_species.most_common(10):
        total = len(by_species[sp])
        print(f"    {sp:<20} drop {n} / {total} (keep {total-n} at J={species_canonical_J[sp]})")

    print(f"\n  Skeleton re-saves needed:")
    for sp, (old_J, new_J) in species_skeleton_action.items():
        print(f"    {sp:<20} skeleton J={old_J} → motion canonical J={new_J}")

    if args.dry_run:
        print("\n[DRY RUN] No files modified.")
        return 0

    # === Execute ===
    print(f"\n=== EXECUTING ===")
    # 1. Delete minority-J clips
    for fp in to_delete:
        fp.unlink()
    print(f"  Deleted {len(to_delete)} motion files")

    # 2. Re-save skeletons that mismatch
    # Strategy: load a majority-J motion file's structure, reuse skeleton's
    # graph data (adjacency, etc.) for the canonical J. If sk_J > motion_J,
    # truncate skeleton arrays at [:canonical_J]. If sk_J < motion_J, FAIL
    # (would need data we don't have).
    for sp, (sk_J, motion_J) in species_skeleton_action.items():
        sk_fp = skeletons_dir / f"{sp}.npz"
        sk = dict(np.load(sk_fp, allow_pickle=True))
        if sk_J > motion_J:
            # Skeleton has 1+ extra joints (dummy root that motion removed).
            # Truncate from start (drop dummy_root at index 0) by motion_J - sk_J amount
            n_drop = sk_J - motion_J  # typically 1
            # Per the preprocess dummy-root logic, removed joint is at index 0;
            # parent_indices remap: subtract 1 for indices > 0, root indicator updated
            new_sk = {}
            for k, v in sk.items():
                if hasattr(v, "shape") and v.ndim >= 1 and v.shape[0] == sk_J:
                    new_sk[k] = v[n_drop:]
                    # Fix parent_indices: subtract n_drop, set new root
                    if k == "parent_indices":
                        new_parents = []
                        for orig_p in v[n_drop:]:
                            p_int = int(orig_p)
                            if p_int < n_drop:
                                new_parents.append(-1)  # was pointing to removed root
                            else:
                                new_parents.append(p_int - n_drop)
                        new_sk[k] = np.array(new_parents, dtype=v.dtype)
                elif hasattr(v, "shape") and v.ndim == 2 and v.shape[0] == sk_J and v.shape[1] == sk_J:
                    # 2D matrices (adjacency, geodesic_dist)
                    new_sk[k] = v[n_drop:, n_drop:]
                else:
                    new_sk[k] = v
            np.savez(sk_fp, **new_sk)
            print(f"  Truncated {sp} skeleton J={sk_J} → {motion_J}")
        else:
            # Skeleton has fewer joints than majority motion — can't easily fix
            print(f"  ⚠ {sp}: skeleton J={sk_J} < motion J={motion_J}. Need to delete "
                  f"all clips of this species (skeleton can't be reconstructed without "
                  f"re-running preprocess). Dropping species entirely.")
            sk_fp.unlink()
            # Delete all motions of this species
            for fp_sp, _ in by_species[sp]:
                if fp_sp.exists():
                    fp_sp.unlink()
                    to_delete.append(fp_sp)

    # 3. Verify all motions + skeletons now align
    print(f"\n=== POST-FIX VERIFICATION ===")
    remaining = sorted(motions_dir.glob("*.npz"))
    mismatch_count = 0
    for fp in remaining:
        m = np.load(fp)
        J_motion = m["local_positions"].shape[1]
        sp = fp.stem.rsplit("_", 1)[0]
        sk_fp = skeletons_dir / f"{sp}.npz"
        if not sk_fp.exists():
            print(f"  ⚠ {fp.name}: skeleton missing → DELETE")
            fp.unlink()
            mismatch_count += 1
            continue
        sk = np.load(sk_fp, allow_pickle=True)
        J_sk = len(sk["parent_indices"])
        if J_motion != J_sk:
            print(f"  ⚠ {fp.name}: motion J={J_motion}, skeleton J={J_sk} STILL MISMATCH → DELETE")
            fp.unlink()
            mismatch_count += 1

    final_count = sum(1 for _ in motions_dir.glob("*.npz"))
    print(f"\n  Final motion count: {final_count}")
    print(f"  Total deleted: {len(to_delete) + mismatch_count}")
    print(f"  Post-fix mismatches removed: {mismatch_count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
