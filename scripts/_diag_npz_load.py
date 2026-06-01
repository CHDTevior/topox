"""Confirm the caption-cache load bottleneck + best fix (read-only timing).
Compares 3 ways to load the 1.47GB / 409970-key npz:
  A. per-key  npz[key]  (current code, slow)
  B. dict(np.load(...))  (one-shot)
  C. np.load(..., mmap)  then iterate (lazy)
"""
import time, numpy as np

P = "data/anytop_caption_t5_cleanL2_multi.npz"

# A. per-key, first 3000 keys → extrapolate
t = time.time()
with np.load(P) as z:
    files = z.files
    for k in files[:3000]:
        _ = z[k].astype(np.float32)
dtA = time.time() - t
print(f"A per-key 3000:     {dtA:.1f}s  ({dtA/3000*1000:.1f}ms/key) -> full {len(files)}: ~{len(files)*dtA/3000/60:.1f}min")

# B. dict(np.load(...)) — full one-shot
t = time.time()
d = dict(np.load(P))
dtB = time.time() - t
print(f"B dict(np.load):    {dtB:.1f}s  (full {len(d)} keys at once)")

# verify B has same content as a per-key sample
k0 = files[0]
with np.load(P) as z:
    same = np.array_equal(z[k0].astype(np.float32), d[k0].astype(np.float32))
print(f"   content match (key0): {same}")
print(f"SPEEDUP B vs A(extrapolated): {(len(files)*dtA/3000)/max(dtB,0.001):.0f}x")
