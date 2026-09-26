"""Stratified ~120-clip human subset for the v3 re-encode gates
(handoff/20260630_033233_human_rot6d_v3_converter_implementation.md, sampling rule).
Strata: (a) highest v2 rot6d 2nd-diff accel; (b) top v2 gt_fk_mismatch; (c) 180-stress
motions (deep flex / kick / sit / jump / turn, by caption keyword); (d) length spread;
(e) random locomotion. Records motion-ids + why each was chosen to a json. NO GPU.

Scans a bounded prefix (--scan_limit) for the accel/fk-mismatch strata (full 25k scan is
slow + unnecessary for a representative gate subset). Usage:
  python scripts/_v3_subset_select.py --n 120 --scan_limit 1500 --out scratch/v3_subset.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

REPO = "/iridisfs/scratch/ts1v23/workspace/noKslot_clean"
HM = "/iridisfs/scratch/ts1v23/workspace/motion-latent-diffusion-main"
sys.path.insert(0, HM); sys.path.insert(0, REPO)
import importlib.util
_spec = importlib.util.spec_from_file_location("cv", REPO + "/scripts/convert_humanml3d_to_anytop13.py")
cv = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(cv)
from src.models.graph_salad.rot6d_fk_recovery import recover_rot6d_fk_positions_torch
from src.models.graph_salad.world_recovery import recover_world_positions_torch

J = cv.J; PARENTS = cv.PARENTS
SC_TOK = [j for j in range(1, J) if PARENTS[j] in cv._SINGLE_CHILD_PARENTS]
STRESS_KW = ["kick", "jump", "sit", "turn", "crouch", "flip", "squat", "kneel", "spin", "cartwheel"]
LOCO_KW = ["walk", "run", "jog", "stroll"]


def _accel6d_singlechild(v2):
    if v2.shape[0] < 3:
        return 0.0
    a = v2[2:, SC_TOK, 3:9] - 2 * v2[1:-1, SC_TOK, 3:9] + v2[:-2, SC_TOK, 3:9]
    return float(np.linalg.norm(a, axis=-1).mean())


def _fk_mismatch(v2, off):
    t = torch.from_numpy(v2[None].astype(np.float32))
    pj = [[int(z) for z in PARENTS]]; ro = torch.from_numpy(off[None].astype(np.float32))
    jm = torch.ones(1, J, dtype=torch.bool)
    fk = recover_rot6d_fk_positions_torch(t, pj, ro, jm)[0].numpy()
    ric = recover_world_positions_torch(t)[0].numpy()
    return float(np.linalg.norm(fk - ric, axis=-1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--scan_limit", type=int, default=1500)
    ap.add_argument("--out", default=REPO + "/scratch/v3_subset.json")
    args = ap.parse_args()
    off = cv.compute_offsets()
    present = [i for i in cv.read_split("all")
               if (Path(cv.SRC) / "new_joint_vecs" / f"{i}.npy").exists()]
    lens = {}
    for i in present:
        try:
            lens[i] = int(np.load(Path(cv.SRC) / "new_joint_vecs" / f"{i}.npy", mmap_mode="r").shape[0])
        except Exception:
            pass
    present = [i for i in present if lens.get(i, 0) >= 5]   # need T>=3 for accel/training

    picked, tags = [], {}

    def add(i, tag):
        if i in lens and i not in picked and len(picked) < args.n:
            picked.append(i); tags[i] = tag

    # (a)/(b) scan a bounded prefix for accel + fk-mismatch
    scan = present[:args.scan_limit]
    accel, fkmm = {}, {}
    for i in scan:
        try:
            x = np.load(Path(cv.SRC) / "new_joint_vecs" / f"{i}.npy")
            v2 = cv.reencode_rot6d(cv.convert_263_to_13(x), cv.world_positions(x), off, "v2")
            accel[i] = _accel6d_singlechild(v2)
            fkmm[i] = _fk_mismatch(v2, off)
        except Exception:
            pass
    for i in sorted(accel, key=lambda k: -accel[k])[:25]:
        add(i, f"worst_accel={accel[i]:.3f}")
    for i in sorted(fkmm, key=lambda k: -fkmm[k])[:20]:
        add(i, f"top_fkmm={fkmm[i]:.4f}")
    # (c) 180-stress by caption keyword
    for kw in STRESS_KW:
        cnt = 0
        for i in present:
            if cnt >= 4:
                break
            caps = cv.parse_captions(i)
            if caps and kw in caps[0].lower():
                add(i, f"stress:{kw}"); cnt += 1
    # (d) length spread (short..long buckets)
    by_len = sorted(lens, key=lambda k: lens[k])
    for q in np.linspace(0, len(by_len) - 1, 15).astype(int):
        add(by_len[int(q)], f"len={lens[by_len[int(q)]]}")
    # (e) random ordinary locomotion
    loco = [i for i in present if (cv.parse_captions(i) and
            any(k in cv.parse_captions(i)[0].lower() for k in LOCO_KW))]
    rng = np.random.default_rng(42)
    for i in rng.permutation(loco)[:20]:
        add(str(i), "locomotion")
    # fill deterministically if short of n
    for i in present[:: max(1, len(present) // args.n)]:
        add(i, "spread")

    out = {"ids": picked, "tags": tags, "n": len(picked), "scan_limit": args.scan_limit}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"[subset] {len(picked)} ids -> {args.out}")
    from collections import Counter
    print("  strata:", dict(Counter(t.split("=")[0].split(":")[0] for t in tags.values())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
