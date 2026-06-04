#!/usr/bin/env python3
"""(Re)generate the train/val split list files for the AnyTop dataset.

WHAT: AnyTopDataset now READS data_root/splits/{train,val}.txt by default
(use_split_file=True). This script regenerates those files by FORCING the
underlying per-object md5-seeded stratified algorithm (use_split_file=False), so
the files always reflect the algorithm rather than copying a stale file. The
split is deterministic: self.samples depends only on (data_root, val_frac, seed,
split) (never filtered by captions or frame params).

Both trainings use the SAME data dir with val_frac=0.05, seed=42:
  - diffusion : scripts/train_denoiser.py
  - VAE       : scripts/train_graph_vae.py  (--anytop_root path)
so this single pair of lists is exact for BOTH (the loader hard-errors on drift:
overlap / duplicate / missing / uncovered / empty).

Run with the training interpreter:
  /iridisfs/scratch/ts1v23/.conda/bin/python3.12 scripts/_export_split_lists.py
Regenerate after ANY change to data/anytop_planet_zoo_clean_L2/motions or cond.npy.
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data.anytop_dataset import AnyTopDataset  # noqa: E402

DATA_ROOT = ROOT / "data" / "anytop_planet_zoo_clean_L2"
VAL_FRAC, SEED = 0.05, 42

# load_captions=False: split set is independent of captions (self.samples is only
# assigned at anytop_dataset.py:578/599, never filtered afterwards). Skipping the
# caption json.load just avoids a multi-minute init stall and changes nothing.
# max_joints=144 matches the trainings so we REUSE their _cond_normalized_J144.pkl
# cache instead of generating a stray J143 one (max_joints does not affect the
# split set either -- it is not used in sample construction at 559-602).
# use_split_file=False: FORCE the stratified algorithm so this export reflects the
# ALGORITHM, not whatever splits/*.txt already exists -- otherwise regenerating after
# a data change would just copy the stale file and silently drop the new clips.
common = dict(data_root=str(DATA_ROOT), val_frac=VAL_FRAC, seed=SEED,
              max_joints=144, load_captions=False, use_split_file=False)

ds_tr = AnyTopDataset(split="train", **common)
ds_va = AnyTopDataset(split="val", **common)
tr = [pathlib.Path(s["path"]).name for s in ds_tr.samples]
va = [pathlib.Path(s["path"]).name for s in ds_va.samples]

# ---- fail-loud sanity ----
inter = sorted(set(tr) & set(va))
assert not inter, f"train/val OVERLAP ({len(inter)}): {inter[:5]}"
on_disk = sum(1 for _ in (DATA_ROOT / "motions").glob("*.npy"))
matched = len(tr) + len(va)
print(f"[split] data_root      = {DATA_ROOT}")
print(f"[split] motions on disk = {on_disk}")
print(f"[split] matched(tr+va)  = {matched}   (unmatched skipped = {on_disk - matched})")
print(f"[split] train={len(tr)}  val={len(va)}  effective_val_frac={len(va)/matched:.4f}")

HDR = (
    "# AUTO-GENERATED -- do not hand-edit; regenerate via scripts/_export_split_lists.py\n"
    "# Materialized {split} split of data/anytop_planet_zoo_clean_L2, used by BOTH\n"
    "# trainings: diffusion (scripts/train_denoiser.py) + VAE (scripts/train_graph_vae.py).\n"
    "# val_frac={vf} seed={sd}.\n"
    "# AnyTopDataset READS this file by default (use_split_file=True): one .npy basename\n"
    "# per line, '#' lines ignored. If train.txt/val.txt are absent it falls back to the\n"
    "# per-object md5-seeded stratified algorithm. This file was generated FROM that\n"
    "# algorithm -- regenerate after ANY data change to keep them in sync; the loader\n"
    "# HARD-ERRORS on drift (overlap / duplicate / missing / uncovered).\n"
    "# {n} motions in this split.\n"
)
out_dir = DATA_ROOT / "splits"
out_dir.mkdir(exist_ok=True)
for name, lst in (("train", tr), ("val", va)):
    p = out_dir / f"{name}.txt"
    p.write_text(HDR.format(split=name, vf=VAL_FRAC, sd=SEED, n=len(lst))
                 + "\n".join(lst) + "\n")
    print(f"[split] wrote {p}  ({len(lst)} motions)")
