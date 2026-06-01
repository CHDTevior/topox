"""Smoke gate (batch.py change): does anytop_collate_fn emit anytop_mean/std,
and does GraphMotionBatch.from_collate_dict expose them as [B,Jmax,13]?

Runs a real collate of a few clips and checks end-to-end. Prints PASS/FAIL.
Run: python scripts/_smoke_batch_anytop_meanstd.py
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.anytop_dataset import AnyTopDataset, collate_fn  # noqa: E402
from src.models.graph_salad.batch import GraphMotionBatch  # noqa: E402

ANYTOP_ROOT = str(ROOT / "data" / "anytop_planet_zoo_clean_L2")


def main():
    ds = AnyTopDataset(
        split="val", val_frac=0.05, seed=42, data_root=ANYTOP_ROOT,
        num_frames=64, max_joints=144, caption_emb_cache=None,
    )
    items = [ds[i] for i in range(4)]
    # anytop path collate is anytop_dataset.collate_fn (train_graph_vae.py:46
    # imports it as `anytop_collate_fn`); takes the list[dict] of items.
    d = collate_fn(items)

    print("collate dict has anytop_mean:", "anytop_mean" in d, flush=True)
    print("collate dict has anytop_std :", "anytop_std" in d, flush=True)
    if "anytop_mean" in d:
        print("  anytop_mean shape:", tuple(d["anytop_mean"].shape),
              "dtype:", d["anytop_mean"].dtype, flush=True)
    if "anytop_std" in d:
        print("  anytop_std  shape:", tuple(d["anytop_std"].shape),
              "dtype:", d["anytop_std"].dtype, flush=True)

    # construct typed batch
    batch = GraphMotionBatch.from_collate_dict(d)
    ok_mean = batch.anytop_mean is not None
    ok_std = batch.anytop_std is not None
    print("batch.anytop_mean is not None:", ok_mean, flush=True)
    print("batch.anytop_std  is not None:", ok_std, flush=True)
    if ok_mean:
        B = batch.batch_size
        Jm = batch.max_joints
        shp = tuple(batch.anytop_mean.shape)
        print(f"  batch.anytop_mean shape={shp} expected=({B},{Jm},13)", flush=True)
        shape_ok = shp == (B, Jm, 13)
    else:
        shape_ok = False

    # round-trip check: denorm anytop_x with mean/std should be finite + sane
    if ok_mean and batch.anytop_x is not None:
        # anytop_x is [B,Jm,13,T] normalized; denorm = norm*(std+1e-6) + mean
        # (must match anytop_dataset._STD_FLOOR=1e-6 used in forward normalize).
        ax = batch.anytop_x                      # [B,Jm,13,T]
        mean = batch.anytop_mean.unsqueeze(-1)   # [B,Jm,13,1]
        std = batch.anytop_std.unsqueeze(-1)     # [B,Jm,13,1]
        raw = ax * (std + 1e-6) + mean
        finite = bool(torch.isfinite(raw).all().item())
        print(f"  denorm(anytop_x) finite: {finite}  raw range "
              f"[{raw.min().item():.3f},{raw.max().item():.3f}]", flush=True)
    else:
        finite = False

    if ok_mean and ok_std and shape_ok and finite:
        print("BATCH_GATE PASS", flush=True)
    else:
        print(f"BATCH_GATE FAIL (mean={ok_mean} std={ok_std} "
              f"shape={shape_ok} finite={finite})", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
