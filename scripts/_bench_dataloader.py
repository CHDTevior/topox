#!/usr/bin/env python3
"""One-off dataloader throughput benchmark (NOT committed to training).

Measures: single-thread __getitem__ cost (and the world-recovery fraction), then
DataLoader items/sec at several num_workers, to size num_workers for the real run.
Run on a compute node:  python scripts/_bench_dataloader.py
"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
from torch.utils.data import DataLoader
from src.data.anytop_dataset import AnyTopDataset, collate_fn

ROOT_DATA = "data/animo4d_anytop_clean_L5"
ds = AnyTopDataset(split="train", num_frames=64, max_joints=64,
                   val_frac=0.05, load_captions=False, data_root=ROOT_DATA)
print(f"[bench] train dataset size = {len(ds)}", flush=True)

# ---- single-thread __getitem__ cost (warm: 2nd pass so .npy is page-cached) ----
idxs = np.random.RandomState(0).randint(0, len(ds), size=60)
for i in idxs[:10]:
    _ = ds[i]                       # warm page cache
t0 = time.time()
for i in idxs:
    _ = ds[i]
dt = (time.time() - t0) / len(idxs)
print(f"[bench] single-thread __getitem__: {dt*1000:.1f} ms/item  "
      f"({1.0/dt:.1f} items/s/worker)", flush=True)

# ---- isolate world-recovery cost ----
from src.data.anytop_dataset import _recover_world_positions
info = ds.samples[int(idxs[0])]
c = ds.cond[info["object_type"]]
raw = np.load(info["path"]).astype(np.float32)[:, c["new_to_old_perm"], :]
N = 30
t0 = time.time()
for _ in range(N):
    _ = _recover_world_positions(raw)
wr = (time.time() - t0) / N
print(f"[bench] _recover_world_positions: {wr*1000:.1f} ms/clip "
      f"(J={raw.shape[1]} T={raw.shape[0]})", flush=True)

# ---- DataLoader items/sec at several worker counts ----
BATCH = 16
for nw in [8, 16, 24, 28]:
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, sampler=None,
                    collate_fn=collate_fn, num_workers=nw, drop_last=True,
                    pin_memory=True, persistent_workers=True, prefetch_factor=4)
    it = iter(dl)
    next(it)                        # spin up workers (excluded from timing)
    t0 = time.time(); nb = 0
    for _ in range(20):             # time 20 batches
        next(it); nb += 1
    dt = time.time() - t0
    ips = nb * BATCH / dt
    print(f"[bench] num_workers={nw:2d}: {ips:7.1f} items/s  "
          f"({dt/nb*1000:6.1f} ms/batch of {BATCH})", flush=True)
    del it, dl
print("[bench] done", flush=True)
