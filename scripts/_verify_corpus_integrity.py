#!/usr/bin/env python3
"""Open every motion file. The first conversion wrote np.savez directly to the final path, so an
interruption could have left a truncated archive that a rerun would have silently kept."""
import json, os, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "dataset/ktjd17_pzh312_noik_v1")
NEED = {"motion", "heading_valid", "clip_id", "rig_id", "fps_target", "origin_xz"}
def check(p):
    try:
        with np.load(p) as z:
            missing = NEED - set(z.files)
            if missing:
                return (p.name, f"missing {sorted(missing)}")
            m = z["motion"]
            if m.dtype != np.float32 or m.ndim != 3 or m.shape[-1] != 17:
                return (p.name, f"bad motion {m.dtype} {m.shape}")
            if not np.isfinite(m).all():
                return (p.name, "non-finite")
            if z["heading_valid"].shape != (m.shape[0],):
                return (p.name, "heading_valid length mismatch")
    except Exception as e:
        return (p.name, f"{type(e).__name__}: {e}")
    return None
fs = sorted((ROOT / "motions").glob("*.npz"))
print(f"[verify] opening {len(fs)} files", flush=True)
bad, done = [], 0
with ProcessPoolExecutor(int(os.environ.get("WORKERS", "16"))) as ex:
    for r in ex.map(check, fs, chunksize=64):
        done += 1
        if r: bad.append(r)
        if done % 20000 == 0: print(f"  {done}/{len(fs)}", flush=True)
print(f"[verify] {len(fs)} files, {len(bad)} corrupt")
for n, why in bad[:10]: print(f"   {n}: {why}")
tmp = list((ROOT / "motions").glob("*.tmp"))
print(f"[verify] leftover .tmp files: {len(tmp)}")
sys.exit(1 if bad else 0)
