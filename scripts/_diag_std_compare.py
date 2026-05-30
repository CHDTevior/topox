"""Validate our std-recompute method on KNOWN-GOOD species, then contrast with
the two failing ones. If cond.std == recomputed-std on the good species, our
method is correct, and the 1e21 values on seal/monitor are confirmed bad data.

Read-only. Usage: python scripts/_diag_std_compare.py
"""
import glob
import numpy as np

DATA = "data/anytop_planet_zoo_clean_L2"
cond = np.load(f"{DATA}/cond.npy", allow_pickle=True).item()

SPECIES = [
    # known-good (speed_ratio 0.97-1.04 in QA)
    "PZ_Reticulated_Giraffe_Female",
    "PZ_Siberian_Tiger_Female",
    "PZ_Saltwater_Crocodile_Female",
    "PZ_California_Sea_Lion_Female",
    "PZ_American_Alligator_Female",
    "PZ_Indian_Elephant_Male",
    # failing
    "PZ_Grey_Seal_Female",
    "PZ_Asian_Water_Monitor_Female",
]

for sp in SPECIES:
    if sp not in cond:
        print(f"{sp}: NOT IN COND")
        continue
    std = np.asarray(cond[sp]["std"], dtype=np.float64)   # [J,13]
    Jref = std.shape[0]
    files = sorted(glob.glob(f"{DATA}/motions/{sp}_*.npy"))
    allm = []
    for f in files[:60]:
        m = np.load(f)
        if m.ndim == 3 and m.shape[1] == Jref and m.shape[2] == 13:
            allm.append(m.reshape(-1, Jref, 13))
    if not allm:
        print(f"{sp}: no shape-matching clips (Jref={Jref})")
        continue
    M = np.concatenate(allm, axis=0).astype(np.float64)
    my = M.std(axis=0)                                     # [J,13]

    print(f"\n###### {sp}  J={Jref}  frames={M.shape[0]} ######")
    header = "  ch  {:>12}  {:>12}  {:>14}".format("cond_max", "mine_max", "ratio c/m")
    print(header)
    for ch in range(13):
        cm = float(np.abs(std[:, ch]).max())
        mm = float(np.abs(my[:, ch]).max())
        r = cm / max(mm, 1e-12)
        if r > 100:
            flag = " <<< BAD cond"
        elif 0.3 < r < 3.0:
            flag = " ~ok"
        else:
            flag = " ? diff"
        print("  {:>2}  {:>12.4g}  {:>12.4g}  {:>14.4g}{}".format(ch, cm, mm, r, flag))
