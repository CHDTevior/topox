import sys, numpy as np
a, b = np.load(sys.argv[1], allow_pickle=True), np.load(sys.argv[2], allow_pickle=True)
ka, kb = set(a.files), set(b.files)
bad = []
if ka != kb: bad.append(f"key sets differ: only_pre={sorted(ka-kb)[:5]} only_post={sorted(kb-ka)[:5]}")
for k in sorted(ka & kb):
    x, y = a[k], b[k]
    if x.dtype.kind in "USO" or y.dtype.kind in "USO":
        if not np.array_equal(x.astype(str), y.astype(str)): bad.append(f"{k}: string mismatch")
    elif x.shape != y.shape: bad.append(f"{k}: shape {x.shape} vs {y.shape}")
    elif not np.array_equal(x, y): bad.append(f"{k}: max|diff|={np.abs(x.astype(np.float64)-y.astype(np.float64)).max()}")
print("compared", len(ka & kb), "keys;", "IDENTICAL" if not bad else "DIFFERENCES:"); [print("  ", m) for m in bad[:20]]
sys.exit(1 if bad else 0)
