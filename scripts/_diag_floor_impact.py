"""Check if the std floor (std = max(std, 1e-6)) was applied too aggressively.

A floor is "too heavy" if it RAISES many std values that were legitimately
small-but-nonzero, artificially suppressing real variance in those channels.

For the cleaned cond.npy we measure, across all 473 objects x J x 13:
  - how many std elements are EXACTLY at the floor (==1e-6) -> floored
  - of those, how many had a real underlying std just below 1e-6 (harmless:
    truly-constant channel) vs were meaningfully nonzero before flooring
  - the distribution of std just above the floor (1e-6 .. 1e-3) to see if a
    cliff of real values got clamped

Read-only. Usage: python scripts/_diag_floor_impact.py
"""
import numpy as np

DATA = "data/anytop_planet_zoo_clean_L2"
FLOOR = 1e-6
cond = np.load(f"{DATA}/cond.npy", allow_pickle=True).item()

all_std = []
per_obj_floored = []
for sp, c in cond.items():
    try:
        std = np.asarray(c["std"], dtype=np.float64).reshape(-1)
    except Exception:
        continue
    all_std.append(std)
    n_at_floor = int(np.isclose(std, FLOOR, rtol=0, atol=0).sum() + (std == FLOOR).sum() > 0) \
        if False else int((std <= FLOOR * 1.0000001).sum())
    per_obj_floored.append((sp, std.size, n_at_floor))

S = np.concatenate(all_std)
N = S.size
print(f"#### Floor impact on cleaned cond.npy ####")
print(f"  total std elements: {N}  (473 objects x J x 13)")
print(f"  std == floor(1e-6) exactly: {(S == FLOOR).sum()}  "
      f"({100*(S==FLOOR).sum()/N:.3f}%)")
print(f"  std <= 1e-6 (floored region): {(S <= FLOOR*1.0001).sum()}  "
      f"({100*(S<=FLOOR*1.0001).sum()/N:.3f}%)")

# Distribution buckets — is there a pile-up just above the floor (= clamped)?
print(f"\n  std distribution buckets:")
buckets = [(0, 1e-6), (1e-6, 1e-5), (1e-5, 1e-4), (1e-4, 1e-3),
           (1e-3, 1e-2), (1e-2, 1e-1), (1e-1, 1.0), (1.0, 10.0)]
for lo, hi in buckets:
    cnt = int(((S > lo) & (S <= hi)).sum())
    print(f"    ({lo:>8.0e}, {hi:>8.0e}]: {cnt:>9d}  ({100*cnt/N:6.3f}%)")

print(f"\n  std stats: min={S.min():.4g} median={np.median(S):.4g} "
      f"max={S.max():.4g} mean={S.mean():.4g}")

# Which objects have the most floored elements? (concentration check)
per_obj_floored.sort(key=lambda x: -x[2])
print(f"\n  top objects by #floored-std elements:")
for sp, sz, nf in per_obj_floored[:8]:
    print(f"    {sp:42s} floored={nf:4d}/{sz:5d} ({100*nf/sz:.2f}%)")

# Interpretation hint
floored_pct = 100 * (S <= FLOOR * 1.0001).sum() / N
print(f"\n  VERDICT HINT:")
if floored_pct < 0.5:
    print(f"    floored {floored_pct:.3f}% — NEGLIGIBLE. Floor only catches truly-")
    print(f"    constant channels (root joint never moves on some channels). NOT heavy.")
elif floored_pct < 5:
    print(f"    floored {floored_pct:.3f}% — minor. Worth a glance but likely fine.")
else:
    print(f"    floored {floored_pct:.3f}% — NON-TRIVIAL. Many channels clamped; floor")
    print(f"    may be suppressing real small-variance signal. Flag to user.")
print("#### DONE ####")
