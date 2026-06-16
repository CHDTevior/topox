"""One-off GT-render fidelity check (2026-06-10, user request).

Verifies the QA gif's red GT panel against the dataset-export reference renders
in data/animo4d_anytop_clean_L5/animations/l5_dense_random10_20260608:

  Layer 1 (numeric): dataset path  item["motion_features"][:gt_T, :J, :3]
            vs reference path      _recover_world_positions(np.load(motion.npy))
            -> shapes + max|diff|. Both must be the SAME world-space recovery
            (right-handed, y-up, facing z+), so any crop/joint-remap/resample
            distortion shows up here.
  Layer 2 (visual): render the SAME clip's GT through the QA renderer
            (make_t2m_large_gif red rightmost panel) so it can be eyeballed
            against the existing reference gif.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.anytop_dataset import AnyTopDataset, _recover_world_positions  # noqa: E402
from scripts.animate_denoiser import make_t2m_large_gif  # noqa: E402
from scripts.animate import fk_rest_pose  # noqa: E402

ROOT = PROJECT_ROOT / "data/animo4d_anytop_clean_L5"
REF_DIR = ROOT / "animations/l5_dense_random10_20260608"
OUT_DIR = PROJECT_ROOT / "runs/codeflow_graph_pscf_L5_b16_lr1p2e4_seed42/_gt_fidelity_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)

manifest = json.loads((REF_DIR / "manifest.json").read_text())
targets = [m for m in manifest if not m.get("skipped")][:2]


def find_split(stem: str) -> str | None:
    for sp in ("val", "train"):
        if stem in (ROOT / "splits" / f"{sp}.txt").read_text():
            return sp
    return None


for t in targets:
    stem = Path(t["motion"]).stem
    sp = find_split(stem)
    print(f"\n=== {t['object']} / {stem[:60]}... split={sp} ref_shape={t['shape']}")
    if sp is None:
        print("  [FAIL] motion not found in train/val splits")
        continue

    ds = AnyTopDataset(split=sp, num_frames=300, max_joints=64, val_frac=0.05,
                       seed=42, data_root=str(ROOT))
    idx = next((i for i, s in enumerate(ds.samples) if s["motion_id"] == stem), None)
    if idx is None:
        print("  [FAIL] motion_id not found in dataset samples")
        continue
    item = ds[idx]
    J = int(item["num_joints"])
    gt_T = int(item["num_frames"])
    ds_world = item["motion_features"][:gt_T, :J, :3].numpy()

    raw = np.load(ROOT / "motions" / t["motion"])
    # dataset reorders the J axis to BFS/FK order BEFORE recovery
    # (anytop_dataset.py L987); apply the same perm for an apples-to-apples diff.
    perm = ds.cond[t["object"]]["new_to_old_perm"]
    ref_world = _recover_world_positions(raw[:, perm, :])

    print(f"  dataset: T={gt_T} J={J}  reference: T={ref_world.shape[0]} J={ref_world.shape[1]}")
    if ds_world.shape == ref_world.shape:
        d = np.abs(ds_world - ref_world)
        print(f"  [L1 numeric] max|diff|={d.max():.6e} mean|diff|={d.mean():.6e} "
              f"-> {'PASS (identical world recovery)' if d.max() < 1e-4 else 'MISMATCH'}")
    else:
        print(f"  [L1 numeric] SHAPE MISMATCH (dataset crop/resample differs from raw) "
              f"{ds_world.shape} vs {ref_world.shape}")

    # Layer 2: render GT through the QA path (red rightmost panel).
    parents = [int(p) for p in item["parent_indices"][:J]]
    rest_off = item["rest_offsets"][:J].numpy()
    static_pose = fk_rest_pose(rest_off, parents)
    gif = OUT_DIR / f"{t['object']}_gtcheck.gif"
    make_t2m_large_gif(ds_world, ds_world, static_pose, parents,
                       f"GT fidelity check: {t['object']} (compare vs reference render)",
                       str(gif), fps=8, gt=ds_world,
                       pred_labels=("(same GT, panel2)", "(same GT, panel3)"))
    print(f"  [L2 visual] -> {gif}")

print("\nDONE")
