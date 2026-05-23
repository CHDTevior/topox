"""Preflight: T5 caption coverage check for Phase-2 denoiser training.

Confirms that for both train + val splits of AnyTopDataset, every sample has
`has_text=True` (caption_emb is loaded from cache, not zero-fill). Phase-2
text-conditional denoising requires 100% coverage — if any sample falls back to
zero embedding, `cond_drop_prob` CFG semantics break (the sample would be
silently treated as uncond).

Usage:
    python scripts/preflight_t5_coverage.py \
        --cache data/anytop_caption_t5_1070.npz \
        --texts_json data/anytop_truebones/motion_texts_by_file_with_codex_drafts.json
"""
from __future__ import annotations

import argparse, os, sys
import numpy as np
import torch
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.data.anytop_dataset import AnyTopDataset


def check_split(split: str, cache_path: str) -> tuple[int, int, list[tuple[str, str]]]:
    """Returns (n_total, n_has_text, list of (motion_id, species) with has_text=False)."""
    ds = AnyTopDataset(
        split=split, num_frames=64, max_joints=143,
        load_captions=True,
        caption_emb_cache=cache_path,
    )
    n = len(ds)
    has_text_count = 0
    missing: list[tuple[str, str]] = []
    species_missing: Counter = Counter()
    for i in range(n):
        item = ds[i]
        ht = bool(item.get("has_text", False))
        if ht:
            # Sanity: caption_emb must be a [768] float tensor and non-zero
            emb = item.get("caption_emb")
            if emb is None:
                missing.append((item.get("motion_id", "?"), item.get("object_type", "?")))
                species_missing[item.get("object_type", "?")] += 1
                continue
            if isinstance(emb, torch.Tensor):
                e = emb
            else:
                e = torch.as_tensor(emb)
            if e.shape != (768,):
                missing.append((item.get("motion_id", "?") + f" [bad_shape={tuple(e.shape)}]",
                                item.get("object_type", "?")))
                species_missing[item.get("object_type", "?")] += 1
                continue
            if torch.abs(e).sum().item() < 1e-6:
                missing.append((item.get("motion_id", "?") + " [zero_emb]",
                                item.get("object_type", "?")))
                species_missing[item.get("object_type", "?")] += 1
                continue
            has_text_count += 1
        else:
            missing.append((item.get("motion_id", "?"), item.get("object_type", "?")))
            species_missing[item.get("object_type", "?")] += 1
    return n, has_text_count, missing, species_missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/anytop_caption_t5_1070.npz",
                    help="path to T5 cache npz")
    ap.add_argument("--strict", action="store_true",
                    help="fail loud (exit nonzero) if coverage < 100%")
    args = ap.parse_args()

    # Cache existence + sanity
    cache_path = args.cache
    if not os.path.exists(cache_path):
        print(f"[FAIL] cache not found: {cache_path}", file=sys.stderr)
        return 2
    cache = np.load(cache_path, allow_pickle=True)
    print(f"=== Cache: {cache_path} ({len(cache.files)} keys, dim={cache[cache.files[0]].shape[0]}) ===")

    overall_ok = True
    for split in ("train", "val"):
        print(f"\n--- split={split} ---")
        n, n_has, missing, species_missing = check_split(split, cache_path)
        cov = n_has / n if n > 0 else 0.0
        print(f"  total={n} has_text={n_has} coverage={cov:.4%}")
        if missing:
            print(f"  MISSING {len(missing)}:")
            for mid, sp in missing[:15]:
                print(f"    {mid} (species={sp})")
            if len(missing) > 15:
                print(f"    ... + {len(missing)-15} more")
            print(f"  species hit:")
            for sp, c in sorted(species_missing.items(), key=lambda x: -x[1])[:10]:
                print(f"    {sp}: {c}")
            overall_ok = False
        else:
            print(f"  [OK] all {n} samples have valid T5 caption embedding")

    if overall_ok:
        print("\n=== PREFLIGHT PASS: 100% coverage on train + val ===")
        return 0
    else:
        print("\n=== PREFLIGHT FAIL: some samples missing/zero caption_emb ===")
        return 1 if args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
