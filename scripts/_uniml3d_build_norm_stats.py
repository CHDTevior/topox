#!/usr/bin/env python3
"""Training-space normalization sidecar for the UniML3D corpus (2026-09-16).

Same policy and artifact contract as scripts/_build_pzh312_norm_stats.py (exact-constant valid
cells leave supervision and keep their constant as the mean; non-zero variances below STD_MIN are
floored; the repo convention raw = x*(std+_STD_FLOOR)+mean is preserved). A SEPARATE script rather
than a flag on that one: the PZ+Human builder hard-codes its corpus's generation.json path and is
byte-bound to the active runs, and dataset/ktjd17_uniml3d_v1/stats/rig_stats.npz carries its own
`__generation_id` that must be cross-checked against the corpus.

WHY ALL-CLIP REPORTING STATISTICS ARE THE RIGHT SOURCE FOR THE `rest` SERVING VIEW
The corpus ships reporting statistics over all 6,881 accepted clips, while training applies
configs/uniml3d_v1_visual_exclusions.json and serves 6,880 (the cut clip's rig keeps 3 others, so
its stats row really does see the extreme motion). Under --rep_norm rest the loader takes exactly
two things from this artifact: `supervise_mask`, and `mean` on the cells that mask drops. Neither
can be made wrong by the extra clip:
  * a cell that is exact-constant over the SUPERSET is exact-constant over the subset with the
    SAME constant, so every restored constant is correct for the served view;
  * a cell that the extra clip makes non-constant is merely SUPERVISED here where a subset-derived
    artifact would have dropped it -- supervision with the rest-frame mean, i.e. the ordinary path.
So the superset can only under-drop, never mis-restore. Recomputing per-clip statistics over the
cut view would cost a full corpus pass to move cells in the safe direction only, and would break
the PZ+Human precedent of consuming the corpus's official statistics rather than recomputing them.
The exclusion artifact's sha256 is recorded in __meta so the choice is auditable.

STD_MIN IS INERT UNDER `rest` and is kept only so the artifact stays a drop-in for the other two
serving normalizations: Ktjd17Base._stats() under "rest" builds its divisor from s_rig and the
frozen block gains (_std_eff), and never reads `std` from here. It still has to pass the loader's
positivity/finiteness validation, which it does.
"""
import hashlib, json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.anytop_dataset import _STD_FLOOR

ROOT = Path(os.environ.get("CORPUS_ROOT", "dataset/ktjd17_uniml3d_v1"))
SRC = Path(os.environ.get("SRC_NPZ", ROOT / "stats" / "rig_stats.npz"))
OUT = Path(os.environ.get("OUT_NPZ", "data/uniml3d_norm_stats_v1.npz"))
EXCL = Path(os.environ.get("EXCLUSION_JSON", "configs/uniml3d_v1_visual_exclusions.json"))
# Same value and same reasoning as the PZ+Human builder (codex 2026-08-21): a REAL floor on the
# supervised divisor, not the divide-by-zero guard _STD_FLOOR.
STD_MIN = float(os.environ.get("STD_MIN", "0.05"))

if OUT.exists() and os.environ.get("FORCE") != "1":
    raise SystemExit(f"REFUSED: {OUT} exists; set FORCE=1 to overwrite")

z = np.load(SRC, allow_pickle=False)
rig_ids = [str(r) for r in z["rig_ids"]]
mean = np.asarray(z["mean"], np.float64)
std = np.asarray(z["std"], np.float64)
vm = np.asarray(z["valid_mask"])
jc = np.asarray(z["joint_count"], np.int64)

# The statistics artifact names the generation it was measured on; it must be THIS corpus, or the
# loader's own generation check would fail later on a file that looks superficially fine.
gen = json.loads((ROOT / "generation.json").read_text())["generation_id"]
src_gen = str(z["__generation_id"]) if "__generation_id" in z.files else None
if src_gen != gen:
    raise SystemExit(f"REFUSED: {SRC} was measured on generation {src_gen!r}, "
                     f"{ROOT}/generation.json says {gen!r}")

# Padding rows beyond joint_count must carry no valid cell, or the loader's [:J] slice would be
# reading statistics for joints that do not exist.
for i, J in enumerate(jc):
    if vm[i, J:].any():
        raise SystemExit(f"REFUSED: rig {rig_ids[i]} has valid cells beyond joint_count {J}")
if not (np.isfinite(mean[vm]).all() and np.isfinite(std[vm]).all()):
    raise SystemExit("REFUSED: non-finite statistics on valid cells")
if (std[vm] < 0).any():
    raise SystemExit("REFUSED: negative std on a valid cell")

const = vm & (std == 0.0)
tiny = vm & (std > 0.0) & (std < STD_MIN)
sup = vm & ~const                                   # cells that will be SUPERVISED
std_eff = np.where(sup, np.maximum(std, STD_MIN), 1.0)
# EVERY valid cell keeps its source mean, constants included: under `rest` the loader copies the
# mean of every UNSUPERVISED cell over the skeleton-derived rest value, so this is the value the
# decoder restores for a cell the model is never asked to predict.
mean_eff = np.where(vm, mean, 0.0)

struct = np.zeros_like(vm)
for i, J in enumerate(jc):
    struct[i, :J, :13] = True
    struct[i, 0, 13:17] = True
sup_rig = sup.sum(axis=(1, 2))
struct_rig = np.maximum(struct.sum(axis=(1, 2)), 1)
frac = sup_rig / struct_rig

print(f"rigs={len(rig_ids)}  joints={int(jc.sum()):,}  valid={int(vm.sum()):,}")
print(f"  exact-constant -> dropped from supervision: {int(const.sum()):,} "
      f"(ch12 contact {int(const[:, :, 12].sum()):,}; "
      f"non-zero constants {int((const & (mean != 0.0)).sum()):,})")
print(f"  non-zero but < {STD_MIN:g} -> floored: {int(tiny.sum()):,}")
print(f"  supervised cells: {int(sup.sum()):,} ({sup.sum()/vm.sum()*100:.2f}% of valid)")
print(f"  effective divisor: min={std_eff[sup].min():.3e} median={np.median(std_eff[sup]):.4f} "
      f"max={std_eff[sup].max():.3f}")
print(f"  structural cells (loader channel_valid before the mask): {int(struct.sum()):,}; "
      f"structural-but-invalid: {int((struct & ~vm).sum()):,}")
print(f"  per-rig supervised/structural: min={frac.min():.4f} p1={np.percentile(frac, 1):.4f} "
      f"median={np.median(frac):.4f}")
print(f"  rigs with ZERO supervised cells: {int((sup_rig == 0).sum())}; "
      f"below 50%: {int((frac < 0.5).sum())}; below 25%: {int((frac < 0.25).sum())}")
worst = np.argsort(frac)[:8]
print("  least-supervised rigs: "
      + ", ".join(f"{rig_ids[i]}({sup_rig[i]}/{struct_rig[i]})" for i in worst))

payload = {"rig_ids": np.array(rig_ids), "joint_count": jc,
           "mean": mean_eff.astype(np.float32),
           "std": (std_eff - _STD_FLOOR).astype(np.float32),   # repo convention
           "supervise_mask": sup, "was_constant": const, "was_floored": tiny,
           "__meta": json.dumps({
               "generation_id": gen,
               "cohort": "uniml3d_v1_all_accepted_clips",
               "source": str(SRC), "source_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(),
               "source_npz_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(),
               "corpus_root": str(ROOT),
               "exclusion_sha256": (hashlib.sha256(EXCL.read_bytes()).hexdigest()
                                    if EXCL.is_file() else None),
               "exclusion_applied": False,
               "exclusion_note": "reporting statistics cover all accepted clips; under rep_norm "
                                 "rest the superset can only under-drop constants, never restore "
                                 "a wrong one (see module docstring)",
               "std_min": STD_MIN, "std_floor": float(_STD_FLOOR),
               "convention": "raw = x * (std + _STD_FLOOR) + mean",
               "excluded_policy": "exact-constant valid cells are removed from supervision "
                                  "(mean kept, std 1) so they normalize to exactly 0",
               "n_rigs": len(rig_ids), "n_valid": int(vm.sum()),
               "n_constant_excluded": int(const.sum()), "n_floored": int(tiny.sum()),
               "n_supervised": int(sup.sum()),
               "n_structural": int(struct.sum()),
               "n_structural_but_invalid": int((struct & ~vm).sum()),
               "min_rig_supervised_fraction": float(frac.min()),
           })}
OUT.parent.mkdir(exist_ok=True)
np.savez(OUT, **payload)
print(f"[OK] {OUT} ({OUT.stat().st_size/1e6:.1f} MB) sha256 "
      f"{hashlib.sha256(OUT.read_bytes()).hexdigest()[:16]}")
