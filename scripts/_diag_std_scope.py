"""Scope the bad-std damage BEFORE deciding whether to retrain.

Two questions:
  (1) How many of the 473 object_types have a bad std (any channel >> sane)?
  (2) CRITICAL: does a bad std corrupt the NORMALIZED training data?
      normalized = (raw - mean) / (std + eps).  If std=1e21, normalized -> ~0
      => training saw degenerate (all-zero) channels for that joint => bad data
         => retrain needed for those species.
      If the dataset clamps/floors std before normalizing, training data may be
      fine even though de-norm (viz) blows up => retrain NOT needed.

Read-only. Usage: python scripts/_diag_std_scope.py
"""
import glob
import numpy as np

DATA = "data/anytop_planet_zoo_clean_L2"
cond = np.load(f"{DATA}/cond.npy", allow_pickle=True).item()
keys = list(cond.keys())

BAD_THRESH = 100.0   # a per-channel std this large is non-physical (sane ~ O(1))

print(f"#### (1) SCOPE: scanning {len(keys)} object_types for bad std ####")
bad = []   # (sp, J, [bad_channels], max_std)
for sp in keys:
    try:
        std = np.asarray(cond[sp]["std"], dtype=np.float64)   # [J,13]
    except Exception:
        continue
    chmax = np.abs(std).max(axis=0)        # [13] worst over joints per channel
    bad_ch = [int(c) for c in np.where(chmax > BAD_THRESH)[0]]
    if bad_ch:
        bad.append((sp, std.shape[0], bad_ch, float(chmax.max())))

print(f"  bad species: {len(bad)} / {len(keys)}")
# channel-frequency of corruption
from collections import Counter
chc = Counter()
for _, _, bch, _ in bad:
    for c in bch:
        chc[c] += 1
print(f"  corrupted-channel frequency: {dict(sorted(chc.items()))}")
print(f"  (channels 9/10/11 = root velocity x/yvel/z used in cumsum recovery)")
print("  --- all bad species (sp | J | bad_channels | max_std) ---")
for sp, J, bch, mx in sorted(bad, key=lambda x: -x[3]):
    print(f"    {sp:42s} J={J:3d} ch={bch} max={mx:.4g}")

# (2) Does a bad std corrupt NORMALIZED training data? Sample 2 bad species.
print(f"\n#### (2) CRITICAL: is the NORMALIZED training data degenerate? ####")
_STD_FLOOR = 1e-6
sample = [b[0] for b in bad[:3]] if bad else []
for sp in sample:
    std = np.asarray(cond[sp]["std"], dtype=np.float64)
    mean = np.asarray(cond[sp]["mean"], dtype=np.float64)
    Jref = std.shape[0]
    files = sorted(glob.glob(f"{DATA}/motions/{sp}_*.npy"))
    m = None
    for f in files:
        mm = np.load(f)
        if mm.ndim == 3 and mm.shape[1] == Jref and mm.shape[2] == 13:
            m = mm.astype(np.float64); break
    if m is None:
        print(f"  {sp}: no clip"); continue
    # normalize exactly as dataset would: (raw - mean) / (std + floor)
    norm = (m - mean[None]) / (std[None] + _STD_FLOOR)   # [T,J,13]
    bch = [c for c in range(13) if np.abs(std[:, c]).max() > BAD_THRESH]
    print(f"\n  == {sp}  bad_channels={bch} ==")
    for c in bch:
        # at the joints where std is huge, what does normalized data look like?
        jbad = np.where(np.abs(std[:, c]) > BAD_THRESH)[0]
        nv = norm[:, jbad, c]
        rawv = m[:, jbad, c]
        print(f"    ch{c}: bad at joints {list(jbad)[:5]}{'...' if len(jbad)>5 else ''} "
              f"({len(jbad)} joints)")
        print(f"         RAW    absmax={np.abs(rawv).max():.4g}  "
              f"NORMALIZED absmax={np.abs(nv).max():.4g}  "
              f"(near-0 => training saw DEGENERATE data here)")
print("\n#### DONE ####")
