"""Convert the slow per-key caption npz cache to a fast (single .npy + keys.json)
format, per codex decision 2026-05-31.

Root cause of slowness: `for key in npz.files: npz[key]` re-parses the zip
central directory + builds a ZipExtFile + np.lib.format header per key. With
409970 keys that is ~O(N^2)-ish (~68 min). Sequential zip-member reads via
zipfile.ZipFile + np.lib.format.read_array are ~1 order of magnitude faster.

Output:
  data/anytop_caption_t5_cleanL2_multi_embs.npy   [N, 768] float32  (order = keys)
  data/anytop_caption_t5_cleanL2_multi_keys.json  list[str] of "<motion_id>__cap<i>"

Read-only on the source npz. One-shot offline. Usage:
  python scripts/convert_caption_npz_to_npy.py
"""
import json
import time
import zipfile

import numpy as np
from numpy.lib.format import read_array

# Sidecar naming MUST match the dataset auto-detect (codex decision): the loader
# checks cache_path.with_suffix(".embs.npy") / ".keys.json". For
# "...multi.npz", with_suffix(".embs.npy") -> "...multi.embs.npy".
SRC = "data/anytop_caption_t5_cleanL2_multi.npz"
DST_EMB = "data/anytop_caption_t5_cleanL2_multi.embs.npy"
DST_KEY = "data/anytop_caption_t5_cleanL2_multi.keys.json"

t0 = time.time()
keys = []
arrs = []
with zipfile.ZipFile(SRC) as zf:
    names = [n for n in zf.namelist() if n.endswith(".npy")]
    print(f"npz members: {len(names)}", flush=True)
    for i, n in enumerate(names):
        with zf.open(n) as f:
            arrs.append(read_array(f, allow_pickle=False))
        keys.append(n[:-4])  # strip trailing ".npy"
        if (i + 1) % 50000 == 0:
            print(f"  {i+1}/{len(names)}  ({time.time()-t0:.0f}s)", flush=True)

embs = np.stack(arrs).astype(np.float32)  # [N, 768]
print(f"stacked {embs.shape} dtype={embs.dtype} in {time.time()-t0:.0f}s", flush=True)

np.save(DST_EMB, embs)
with open(DST_KEY, "w") as f:
    json.dump(keys, f)

# sanity: re-read + spot-check vs source
emb_chk = np.load(DST_EMB, mmap_mode="r")
keys_chk = json.load(open(DST_KEY))
assert emb_chk.shape[0] == len(keys_chk) == len(names), "count mismatch"
with zipfile.ZipFile(SRC) as zf:
    k0 = keys_chk[0]
    with zf.open(k0 + ".npy") as f:
        src0 = read_array(f, allow_pickle=False).astype(np.float32)
    ok = np.array_equal(src0, np.asarray(emb_chk[0], dtype=np.float32))
print(f"WROTE {DST_EMB} {emb_chk.shape} + {DST_KEY} ({len(keys_chk)} keys)  "
      f"key0_match={ok}  total {time.time()-t0:.0f}s", flush=True)
print("DONE", flush=True)
