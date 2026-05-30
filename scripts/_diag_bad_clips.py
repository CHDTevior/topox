"""Test the user's hypothesis: a FEW bad source-motion clips poison the whole
species' std (and thus degrade thousands of otherwise-good clips).

For each of the 15 bad-std species, scan EVERY clip frame-by-frame on the
root-velocity channels (9/10/11) and find which clips carry non-physical values
(nan / inf / |x| beyond a sane bound). Report: how many clips bad vs total, and
what the std would be if those bad clips were EXCLUDED.

Read-only. Usage: python scripts/_diag_bad_clips.py
"""
import glob
import numpy as np

DATA = "data/anytop_planet_zoo_clean_L2"
cond = np.load(f"{DATA}/cond.npy", allow_pickle=True).item()

BAD_SPECIES = [
    "PZ_Snow_Leopard_Male", "PZ_Honey_Badger_Male", "PZ_North_Island_Brown_Kiwi_Female",
    "PZ_North_Island_Brown_Kiwi_Male", "PZ_California_Sea_Lion_Juvenile",
    "PZ_Japanese_Raccoon_Dog_Female", "PZ_Asian_Small_Clawed_Otter_Male",
    "PZ_Giant_Otter_Male", "PZ_Maned_Wolf_Female", "PZ_Grey_Seal_Female",
    "PZ_Proboscis_Monkey_Male", "PZ_Asian_Water_Monitor_Male",
    "PZ_Honey_Badger_Female", "PZ_Pallas_Cat_Female", "PZ_Asian_Water_Monitor_Female",
]
ROOT_VEL_CH = [9, 10, 11]
SANE_ABS = 50.0   # a root-velocity component beyond this is non-physical

print("#### Per-species: which clips poison the std? ####")
print(f"  (scanning root-vel channels {ROOT_VEL_CH}, sane bound |x|<{SANE_ABS})\n")
summary = []
for sp in BAD_SPECIES:
    Jref = int(np.asarray(cond[sp]["std"]).shape[0])
    files = sorted(glob.glob(f"{DATA}/motions/{sp}_*.npy"))
    n_total = len(files)
    bad_clips = []          # (filename, worst_abs, n_bad_frames)
    good_frames = []        # collect root-vel from GOOD clips to recompute std
    for f in files:
        m = np.load(f)
        if not (m.ndim == 3 and m.shape[1] == Jref and m.shape[2] == 13):
            continue
        rv = m[:, :, ROOT_VEL_CH].astype(np.float64)     # [T,J,3]
        finite = np.isfinite(rv)
        worst = np.abs(rv[finite]).max() if finite.any() else np.inf
        n_nan = int((~finite).sum())
        n_extreme = int((np.abs(np.where(finite, rv, 0)) > SANE_ABS).sum())
        if n_nan or n_extreme:
            bad_clips.append((f.split("/")[-1], float(worst), n_nan + n_extreme))
        else:
            good_frames.append(rv.reshape(-1, Jref, 3))
    # recompute std from GOOD clips only (per joint, take channel-9 slot of full 13)
    if good_frames:
        G = np.concatenate(good_frames, axis=0)          # [N,J,3]
        clean_std_max = float(G.std(axis=0).max())
    else:
        clean_std_max = float("nan")
    cond_std_max = float(np.abs(np.asarray(cond[sp]["std"])[:, ROOT_VEL_CH]).max())
    summary.append((sp, n_total, len(bad_clips), cond_std_max, clean_std_max))
    print(f"== {sp}  J={Jref}  total_clips={n_total}  BAD_clips={len(bad_clips)} ==")
    print(f"   cond.std(root-vel) max = {cond_std_max:.4g}   "
          f"clean-recomputed(good clips only) max = {clean_std_max:.4g}")
    for fn, w, nb in bad_clips[:6]:
        print(f"     BAD: {fn[:80]}  worst|x|={w:.4g}  n_bad_vals={nb}")
    if len(bad_clips) > 6:
        print(f"     ... +{len(bad_clips)-6} more bad clips")
    print()

print("#### SUMMARY: bad-clips / total-clips per species ####")
print(f"  {'species':42s} {'total':>6} {'bad':>5} {'cond_std':>12} {'clean_std':>12}")
for sp, nt, nb, cs, cls in summary:
    print(f"  {sp:42s} {nt:>6} {nb:>5} {cs:>12.4g} {cls:>12.4g}")
tot_clips = sum(s[1] for s in summary)
tot_bad = sum(s[2] for s in summary)
print(f"\n  TOTAL across 15 species: {tot_bad} bad clips / {tot_clips} clips "
      f"= {100*tot_bad/max(tot_clips,1):.2f}% bad  "
      f"(=> {tot_clips-tot_bad} good clips poisoned by them)")
print("#### DONE ####")
