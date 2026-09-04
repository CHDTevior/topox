#!/usr/bin/env python
"""Export AnyTop's DIRECT output positions for the samples written by its sample.generate.

Run INSIDE the AnyTop checkout with its own conda env (it needs AnyTop's data_loaders + Motion pkg):

    cd <AnyTop repo> && <anytop python> <this file> --gen_dir gen_out/paper_cmp_v2 [--gen_dir ...]

sample.generate writes, per sample, <rig>_rep_<r>_#<k>.npy -- the de-normalised [T, J, 13] feature
tensor (generate.py:88, saved at :105) -- and a .bvh obtained by IK-fitting the recovered positions and
exporting through the Motion package (generate.py:93-107). The BVH is therefore RIC -> IK -> BVH, one
fit away from what the model produced. This script recovers the model's own positions from the .npy
with AnyTop's own recover_from_bvh_ric_np (the function generate.py itself calls at :93) and stores
them together with the rig's joint names / parents / offsets from the SAME conditioning table
generate.py used, as <sample>.ric_world.npz.

Our compare script (scripts/_compare_external_bvh_geometry.py) reads the pair
(.ric_world.npz = "pose", .bvh = "fk") and validates the BVH node order against these names/offsets.
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np  # noqa: E402
from data_loaders.truebones.truebones_utils.get_opt import get_opt                          # noqa: E402

POSE_SUFFIX = ".ric_world.npz"


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen_dir", action="append", required=True, help="output dir of sample.generate (repeatable)")
    ap.add_argument("--cond", default=None, help="conditioning table; default = the one get_opt points generate.py at")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    opt = get_opt("cpu")
    cond_path = str(Path(a.cond or opt.cond_file).resolve())
    cond = np.load(cond_path, allow_pickle=True).item()
    cond_sha = sha256_file(cond_path)
    # code provenance of THIS export (codex 2026-09-04 round 2): the consumer must know which exporter, which
    # generate.py and which recovery source produced a file, not which happen to be on disk when it reads it
    gen_py = Path(os.getcwd()) / "sample" / "generate.py"
    recover_src = Path(os.getcwd()) / "data_loaders" / "truebones" / "truebones_utils" / "motion_process.py"
    if not gen_py.is_file() or not recover_src.is_file():
        raise SystemExit("[refuse] run from the AnyTop checkout root (sample/generate.py and the recovery source must exist)")
    code_prov = {"exporter_sha256": sha256_file(Path(__file__).resolve()),
                 "generate_py_sha256": sha256_file(gen_py), "recover_src_sha256": sha256_file(recover_src)}
    n = 0
    for d in a.gen_dir:
        files = sorted(Path(d).glob("*.npy"))
        if not files:
            raise SystemExit(f"[refuse] no .npy samples in {d}")
        for f in files:
            rig = f.name.split("_rep_")[0]
            if rig not in cond:
                raise SystemExit(f"[refuse] {f.name}: rig {rig!r} is not in {cond_path}")
            c = cond[rig]
            names = [str(x) for x in c["joints_names"]]
            motion = np.load(f)                                   # [T, J, 13], de-normalised (generate.py:88)
            if motion.ndim != 3 or motion.shape[1] != len(names) or motion.shape[2] != int(opt.feature_len):
                raise SystemExit(f"[refuse] {f.name}: shape {motion.shape} vs J={len(names)} F={opt.feature_len}")
            if not np.isfinite(motion).all():
                raise SystemExit(f"[refuse] {f.name}: non-finite features")
            pos = np.asarray(recover_from_bvh_ric_np(motion), dtype=np.float64)   # [T, J, 3] (generate.py:93)
            if pos.shape != (motion.shape[0], len(names), 3) or not np.isfinite(pos).all():
                raise SystemExit(f"[refuse] {f.name}: recovered positions {pos.shape} malformed")
            bvh = f.with_suffix(".bvh")
            if not bvh.exists():
                raise SystemExit(f"[refuse] {f.name} has no sibling {bvh.name}")
            out = f.with_suffix(POSE_SUFFIX)
            if out.exists() and not a.overwrite:
                raise SystemExit(f"[refuse] {out} exists (pass --overwrite)")
            np.savez(out, positions=pos, joint_names=np.array(names),
                     parents=np.asarray(c["parents"], dtype=np.int64),
                     offsets=np.asarray(c["offsets"], dtype=np.float64),
                     fps=np.array(float(opt.fps)), rig=np.array(rig),
                     source_npy=np.array(f.name), source_npy_path=np.array(str(f.resolve())),
                     source_npy_sha256=np.array(sha256_file(f)), sibling_bvh_sha256=np.array(sha256_file(bvh)),
                     cond_path=np.array(cond_path), cond_sha256=np.array(cond_sha),
                     recover_fn=np.array("data_loaders.truebones.truebones_utils.motion_process.recover_from_bvh_ric_np"),
                     **{k: np.array(v) for k, v in code_prov.items()},
                     format=np.array("anytop-ric-world-v2"))
            n += 1
            print(f"[export] {out.name}: T={pos.shape[0]} J={pos.shape[1]} fps={opt.fps}", flush=True)
    print(f"[export] DONE {n} samples; cond {cond_path} sha256 {cond_sha[:16]}")


if __name__ == "__main__":
    main()
