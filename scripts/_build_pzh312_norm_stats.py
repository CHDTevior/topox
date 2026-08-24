#!/usr/bin/env python3
"""Training-space normalization for the PZ+Human 312-rig corpus (2026-08-21).

SOURCE: the corpus ships official per-(rig, joint, channel) statistics --
dataset/ktjd17_pz_human312_species_stats/rig_stats.npz, [312,102,17] with mean/std/count/
valid_mask/minimum/maximum. We consume them; we do not recompute them.

TWO CORRECTIONS, from the user's own numerical analysis of those stats (2026-08-21), which
refined an over-stated claim of mine (I had said 2% of cells would amplify errors 1e4x; measured
against the real min/max the largest normalized magnitude among low-variance cells is 16.19, and
NONE exceeds 100 -- because 5,274 of the 5,294 low-variance cells are EXACT constants whose
x-mean is identically zero):

 1. EXACT-CONSTANT CELLS JOIN THE VALIDITY MASK. 5,274 valid cells have zero variance, 5,219 of
    them ch12 contact -- joints that never touch the ground for that rig. Supervising them as
    constant-zero targets does not teach anything; it inflates the contact group's denominator and
    dilutes the joints that DO make contact. They are excluded from supervision instead.
 2. NON-ZERO SMALL VARIANCES ARE FLOORED AT 1e-4. Only 20 cells sit in (0, 1e-4); their physical
    jitter is meaningless, and dividing by 2.9e-6 promotes it to full loss weight. The floor is a
    real floor -- unlike _STD_FLOOR=1e-6, which is a divide-by-zero guard, and which my TrueBones
    builder additionally cancelled by storing (std - floor) for an exact round-trip.

The artifact keeps the repo convention raw = x*(std+_STD_FLOOR)+mean, so every existing consumer
(renderer, gamma7 pack, dynamics terms) inverts it unchanged, and carries the extra mask so the
trainer can drop the excluded cells.
"""
import hashlib, json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.anytop_dataset import _STD_FLOOR

SRC = Path(os.environ.get("SRC_NPZ",
                          "dataset/ktjd17_pz_human312_species_stats/rig_stats.npz"))
OUT = Path(os.environ.get("OUT_NPZ", "data/pzh312_norm_stats_v1.npz"))
# A REAL floor, not a divide-by-zero guard. 1e-4 was far too permissive. Four rigs have a root
# rot6d component that is numerically zero across the ENTIRE corpus (codex scanned 999 clips /
# 112,232 frames: |motion[:,0,3]| <= 2.8e-16), so its std sits just under 1e-4 and the analytic
# rest demo -- which writes rot6d IDENTITY, ch3 = 1 -- normalized to 1.0/1e-4 = 10,000. That value
# never reaches the loss (the rest frame is context, not a target) but it does reach x_in, an
# un-normalized nn.Linear, whose weight gradient scales with the input.
# 0.05 is codex's call (gpt-5.6-terra/xhigh, 2026-08-21): the smallest of the candidates that puts
# every rest-context maximum under 20.1 across all 279 train rigs. It floors 2,372 of 251,528
# supervised cells (0.94%); the median supervised std is 0.3804, so the compressed cells are all
# genuinely near-constant channels. It does NOT bound every target -- an Aldabra tortoise toe
# world-velocity cell reaches ~80 off a real std of 0.145 -- and that one is deliberately left
# alone pending a GT visualization rather than masked pre-emptively.
STD_MIN = float(os.environ.get("STD_MIN", "0.05"))

z = np.load(SRC, allow_pickle=True)
rig_ids = [str(r) for r in z["rig_ids"]]
mean = np.asarray(z["mean"], np.float64)
std = np.asarray(z["std"], np.float64)
vm = np.asarray(z["valid_mask"])
jc = np.asarray(z["joint_count"], np.int64)

const = vm & (std == 0.0)
tiny = vm & (std > 0.0) & (std < STD_MIN)
sup = vm & ~const                                   # cells that will be SUPERVISED
std_eff = np.where(sup, np.maximum(std, STD_MIN), 1.0)
# EVERY valid cell keeps its source mean, constants included (codex 2026-08-21 (A)1). With
# mean=const and std=1 the cell normalizes to (const-const)/1 = 0 -- matching the zero-projection
# the model applies to unsupervised cells -- and de-normalizes as 0*1+const = const, returning the
# physical value. Writing 0 here got BOTH directions wrong: the cell normalized to `const` instead
# of 0, and decoded to 0 instead of `const`. Three cells carry non-zero constants (Bongo j0 ch6
# -0.01084, Bongo j1 ch0 0.31799, Red River Hog j1 ch0 0.37080), and ch0 is a direct-position
# channel that also feeds the FK term.
mean_eff = np.where(vm, mean, 0.0)

print(f"rigs={len(rig_ids)}  valid={vm.sum():,}")
print(f"  exact-constant -> dropped from supervision: {const.sum():,} "
      f"(ch12 contact {int((const & (np.arange(17)==12)).sum()):,})")
print(f"  non-zero but < {STD_MIN:g} -> floored: {tiny.sum():,}")
print(f"  supervised cells: {sup.sum():,} ({sup.sum()/vm.sum()*100:.2f}% of valid)")
print(f"  effective divisor: min={std_eff[sup].min():.3e} median={np.median(std_eff[sup]):.4f} "
      f"max={std_eff[sup].max():.3f}")

payload = {"rig_ids": np.array(rig_ids), "joint_count": jc,
           "mean": mean_eff.astype(np.float32),
           "std": (std_eff - _STD_FLOOR).astype(np.float32),   # repo convention
           "supervise_mask": sup, "was_constant": const, "was_floored": tiny,
           "__meta": json.dumps({
               "source_npz_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(),
               "exclusion_sha256": (str(np.load(SRC, allow_pickle=True)["__exclusion_sha256"])
                                    if "__exclusion_sha256" in np.load(SRC, allow_pickle=True)
                                    else None),
               "generation_id": json.loads(
                   Path("dataset/ktjd17_pz_human312/generation.json").read_text())["generation_id"],
               "source": str(SRC), "source_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(),
               "std_min": STD_MIN, "std_floor": float(_STD_FLOOR),
               "convention": "raw = x * (std + _STD_FLOOR) + mean",
               "excluded_policy": "exact-constant valid cells are removed from supervision "
                                  "(mean 0, std 1) so they normalize to exactly 0",
               "n_rigs": len(rig_ids), "n_valid": int(vm.sum()),
               "n_constant_excluded": int(const.sum()), "n_floored": int(tiny.sum()),
           })}
OUT.parent.mkdir(exist_ok=True)
np.savez(OUT, **payload)
print(f"[OK] {OUT} ({OUT.stat().st_size/1e6:.1f} MB) sha256 "
      f"{hashlib.sha256(OUT.read_bytes()).hexdigest()[:16]}")
