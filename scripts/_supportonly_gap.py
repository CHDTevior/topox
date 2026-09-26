"""FK--pose gap of sampled arms, in the rig's mean rest bone length.

The geometry comparator measures jitter, articulation and root speed, and computes a pose-vs-FK gap only for the
external BVH set; neither of our own arms is aggregated there (codex 2026-09-10 r2 #3). Table 2 of the paper needs
that column for our arms, measured on the SAME sampled sequences the ratios come from, so this reads the dumps and
takes it directly: mean over frames and joints of ||gen_fk - gen_ric||, divided by the rig's mean rest bone length
with the root offset excluded -- the convention the trainer's fk_dist uses (src/models/v2/fk_torch.py:133-140).

usage: python scripts/_supportonly_gap.py --view <view> --rig Buffalo \
         --dump zero=renders/... --dump lora=renders/... --out runs/_figs/....json
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", required=True)
    ap.add_argument("--rig", required=True)
    ap.add_argument("--dump", action="append", required=True, metavar="LABEL=DIR")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sk = np.load(Path(a.view) / "skeletons" / f"{a.rig}.npz", allow_pickle=True)
    off = np.asarray(sk["offset_parent_local"], dtype=np.float64)
    # The paper defines the gap as the distance in the rig's MEAN BONE LENGTH, and that is what this divides by.
    # The trainer's own fk_dist adds a 1e-3 guard against a degenerate rig (src/models/v2/fk_torch.py:134), which
    # on these four rigs makes its denominator 0.4% larger; every sampled gap the paper reports is measured here,
    # under this definition, so the two are not mixed (codex r3 #3).
    bone = float(np.linalg.norm(off[1:], axis=-1).mean())
    if not np.isfinite(bone) or bone <= 0:
        raise SystemExit(f"[refuse] {a.rig}: mean rest bone length {bone}")
    n_joints = int(off.shape[0])

    out = {"rig": a.rig, "view": a.view, "mean_rest_bone_length": bone, "arms": {}}
    for spec in a.dump:
        if "=" not in spec:
            raise SystemExit(f"[refuse] --dump wants LABEL=DIR, got {spec!r}")
        label, d = spec.split("=", 1)
        files = sorted(glob.glob(str(Path(d) / "*.world.npz")))
        if not files:
            raise SystemExit(f"[refuse] {label}: no dumps under {d}")
        per = []
        for f in files:
            z = np.load(f, allow_pickle=True)
            ric = np.asarray(z["gen_ric"], dtype=np.float64)
            fk = np.asarray(z["gen_fk"], dtype=np.float64)
            # matching shapes are not enough: an empty array averages to NaN and exits zero (codex r3 #4)
            if ric.shape != fk.shape or ric.ndim != 3:
                raise SystemExit(f"[refuse] {f}: gen_ric {ric.shape} vs gen_fk {fk.shape}")
            if ric.shape[0] < 1 or ric.shape[2] != 3 or ric.shape[1] != n_joints:
                raise SystemExit(f"[refuse] {f}: sampled array {ric.shape}, expected [T>=1, {n_joints}, 3]")
            if not (np.isfinite(ric).all() and np.isfinite(fk).all()):
                raise SystemExit(f"[refuse] {f}: non-finite samples")
            per.append({"motion_id": str(z["motion_id"]), "frames": int(ric.shape[0]),
                        "ckpt": str(z["ckpt"]), "epoch": int(z["epoch"]),
                        "gap_bl": float(np.linalg.norm(fk - ric, axis=-1).mean() / bone)})
        out["arms"][label] = {"n": len(per), "gap_bl": float(np.mean([r["gap_bl"] for r in per])),
                              "per_clip": per}
        print(f"[gap] {a.rig} {label:14s} n={len(per)} gap={out['arms'][label]['gap_bl']:.3f} bl", flush=True)
    ids = [sorted(r["motion_id"] for r in v["per_clip"]) for v in out["arms"].values()]
    if len({tuple(i) for i in ids}) != 1:
        raise SystemExit("[refuse] the arms do not cover the same clips")
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"[gap] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
