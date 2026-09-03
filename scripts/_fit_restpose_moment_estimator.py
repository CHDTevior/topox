"""Predict a rig's AnyTop-13 normalisation moments from its STATIC rest pose alone.

Why. Generation ends in `raw = normed·(std + 1e-6) + mean`, and those moments are per-object
statistics computed from that rig's own motions. For a rig we have never seen there are none, so
the last step of generation cannot be taken. This fits a predictor whose only inputs are the
static skeleton — `offsets` and `parents` — which exist for any rig, seen or not.

Scope, deliberately small. The per-rig moments used by every retained rig are UNCHANGED; the
existing pipeline is correct and stays that way (see FACTS.md C9). This estimator is used only
where real moments do not exist: the k=0 point of the few-shot curve. At k>=1 the moments are
measured from the k available clips, which is the normal deployment state — Motion2Motion needs
example motions of the target character, and Maya 2027.2 MotionMaker requires either one of four
standard character definitions or a studio-trained model built from the studio's own data.

Fit on retained topologies only; the held topologies are the test set and are never fitted to.

    python scripts/_fit_restpose_moment_estimator.py --out data/moment_estimator_v1.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import sys

sys.setrecursionlimit(20000)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.holdout_guard import canonical_form, verify_artifact  # noqa: E402

# Channel groups. Root (slot 0) is RIFKE and means different things from every other slot, so it
# is fitted separately throughout. Within non-root slots the groups are the natural blocks of the
# 13-channel layout.
GROUPS = {
    "nonroot_pos":  ("nonroot", slice(0, 3)),
    "nonroot_rot":  ("nonroot", slice(3, 9)),
    "nonroot_vel":  ("nonroot", slice(9, 12)),
    "nonroot_ct":   ("nonroot", slice(12, 13)),
    # The root slot is RIFKE: pooling its 13 channels averages the live ones with channels that
    # are identically zero on 381 of 382 rigs, which hides the one that decides whether an animal
    # floats. Split into the channels the recovery path actually reads.
    "root_height":  ("root",    slice(1, 2)),      # ch1
    "root_heading": ("root",    slice(3, 9)),      # ch3:9, a pure-yaw 6D matrix
    "root_velxz":   ("root",    [9, 11]),          # ch9 / ch11 only; ch10 is inert
}
# Which groups scale with the rig's linear size. Positions and velocities are lengths and
# lengths-per-frame; rotations and contacts are not. This is a hypothesis and is TESTED below,
# not assumed — the fitted report prints the scale-invariance check for every group.
LENGTH_LIKE = {"nonroot_pos", "nonroot_vel", "root_height", "root_velxz"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/animo4d_L4TB_plus_human_v4b272neutral")
    ap.add_argument("--holdout_artifact", default="data/holdout_topologies_v1.json")
    ap.add_argument("--max_joints", type=int, default=144)
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def fk_rest(parents: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Rest joint positions by accumulating offsets down the chain. `offsets` and `parents` are
    the unambiguous static skeleton; unlike `tpos_first_frame` they are not a posed frame, so no
    reviewer can call this motion-derived."""
    J = len(parents)
    q = np.zeros((J, 3), dtype=np.float64)
    for j in range(J):
        p = int(parents[j])
        q[j] = offsets[j] if (p < 0 or p == j) else q[p] + offsets[j]
    return q


def main():
    args = parse_args()
    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(f"REFUSED: {out} exists; pass --force")

    root = Path(args.data_root)
    art = verify_artifact(args.holdout_artifact, root)
    held_canon = {t["canonical_form_sha256"] for t in art["held_out_trees"]}

    # Use the dataset's own reindexed cond: `mean`/`std` live in the permuted (FK) joint order
    # and are applied to motion that has been reordered the same way. Re-deriving the order here
    # would be a second implementation of a thing this project has already been bitten by.
    from src.data.anytop_dataset import AnyTopDataset
    ds = AnyTopDataset(data_root=str(root), split="all", num_frames=64,
                       max_joints=args.max_joints, load_captions=False)
    cond = ds.cond

    rows = []
    for obj, c in cond.items():
        par = np.asarray(c["parents"]).ravel().astype(int)
        off = np.asarray(c["offsets"], dtype=np.float64).reshape(len(par), 3)
        q = fk_rest(par, off)
        s = float(np.linalg.norm(q - q[0], axis=1).max())
        canon = hashlib.sha256(
            canonical_form(tuple(int(x) for x in par)).encode()).hexdigest()
        rows.append({"obj": obj, "s": s, "held": canon in held_canon,
                     "mean": np.asarray(c["mean"], np.float64),
                     "std": np.asarray(c["std"], np.float64)})

    n_held = sum(r["held"] for r in rows)
    print(f"[est] {len(rows)} object types, {n_held} held / {len(rows)-n_held} retained")
    ss = np.array([r["s"] for r in rows])
    print(f"[est] rest size s: min={ss.min():.3f} median={np.median(ss):.3f} max={ss.max():.3f}")
    if not np.isfinite(ss).all() or (ss <= 0).any():
        raise SystemExit("REFUSED: non-finite or non-positive rest size for some rig")

    def gather(r, group):
        where, sl = GROUPS[group]
        m, sd = r["mean"], r["std"]
        block = slice(1, None) if where == "nonroot" else slice(0, 1)
        idx = sl if not isinstance(sl, list) else np.asarray(sl)
        return m[block][:, idx].ravel(), sd[block][:, idx].ravel()

    # ---- fit ---------------------------------------------------------------------------
    fit_rows = [r for r in rows if not r["held"]]
    est = {}
    report = {"n_retained": len(fit_rows), "n_held": n_held, "groups": {}}
    for g in GROUPS:
        scale_by_s = g in LENGTH_LIKE
        mus, sigs, keep = [], [], []
        for r in fit_rows:
            m, sd = gather(r, g)
            if m.size == 0:
                continue
            k = r["s"] if scale_by_s else 1.0
            mu_r = float(np.mean(m)) / k
            sg_r = float(np.mean(sd)) / k
            # A rig whose motion is degenerate in this group (an in-place clip set) contributes
            # a near-zero scale that a geometric mean cannot survive. Excluded from the FIT only;
            # it is still predicted for, and still scored.
            if sg_r > 1e-8:
                sigs.append(sg_r); keep.append(r["obj"])
            mus.append(mu_r)
        mu = float(np.mean(mus)) if mus else 0.0
        # Geometric mean: scales are positive and span orders of magnitude, so the arithmetic
        # mean is dominated by the largest rigs and a pooled second moment additionally absorbs
        # between-rig variance and over-predicts.
        sg = float(np.exp(np.mean(np.log(sigs)))) if sigs else 1.0
        est[g] = {"mu": mu, "sigma": sg, "scale_by_s": scale_by_s}
        report["groups"][g] = {"mu": mu, "sigma": sg, "scale_by_s": scale_by_s,
                               "n_fit": len(sigs), "n_excluded_degenerate": len(mus) - len(sigs)}

        # Scale-invariance check: the whole premise of predicting from `s` is that after dividing
        # by it the residual is roughly constant across rigs. If it is not, say so here rather
        # than discovering it in a render.
        if sigs:
            lr = np.log(np.array(sigs) / sg)
            report["groups"][g]["residual_log_spread"] = {
                "std": float(lr.std()), "p90_abs": float(np.percentile(np.abs(lr), 90)),
                "max_abs": float(np.abs(lr).max())}

    # Control: refit each length-like group WITHOUT the s scaling, so the report shows whether
    # dividing by rest size earns its place rather than assuming it.
    ctrl = {}
    for g in GROUPS:
        if g not in LENGTH_LIKE:
            continue
        sigs = [v for r in fit_rows
                for v in [float(np.mean(gather(r, g)[1]))] if v > 1e-8]
        ctrl[g] = float(np.exp(np.mean(np.log(sigs)))) if sigs else 1.0

    # ---- validate on the held topologies -----------------------------------------------
    held_rows = [r for r in rows if r["held"]]
    for g in GROUPS:
        errs = []
        for r in held_rows:
            m, sd = gather(r, g)
            if m.size == 0:
                continue
            k = r["s"] if est[g]["scale_by_s"] else 1.0
            true_sg = float(np.mean(sd))
            pred_sg = est[g]["sigma"] * k
            if true_sg > 1e-8 and pred_sg > 1e-8:
                errs.append(abs(np.log(true_sg / pred_sg)))
        if g in ctrl:
            ce = [abs(np.log(float(np.mean(gather(r, g)[1])) / ctrl[g]))
                  for r in held_rows
                  if float(np.mean(gather(r, g)[1])) > 1e-8]
            report["groups"][g]["held_log_error_no_s_control"] = (
                {"mean": float(np.mean(ce)), "mean_fold": float(np.exp(np.mean(ce)))}
                if ce else None)
        report["groups"][g]["held_log_error"] = (
            {"mean": float(np.mean(errs)), "median": float(np.median(errs)),
             "p90": float(np.percentile(errs, 90)), "n": len(errs),
             "mean_fold": float(np.exp(np.mean(errs)))} if errs else None)

    print(f"\n{'group':14s} {'scale~s':>8s} {'fit n':>6s} {'resid σ':>8s}  "
          f"{'held |log err|':>14s} {'fold':>6s} {'no-s fold':>10s}")
    for g, d in report["groups"].items():
        h = d.get("held_log_error")
        rs = d.get("residual_log_spread", {})
        print(f"{g:14s} {str(d['scale_by_s']):>8s} {d['n_fit']:>6d} "
              f"{rs.get('std', float('nan')):>8.3f}  "
              f"{(h['mean'] if h else float('nan')):>14.3f} "
              f"{(h['mean_fold'] if h else float('nan')):>6.2f}x "
              f"{(d.get('held_log_error_no_s_control') or {}).get('mean_fold', float('nan')):>9.2f}x")

    np.savez(out,
             groups=json.dumps({g: est[g] for g in est}),
             report=json.dumps(report),
             artifact_sha256=art["artifact_sha256"],
             obj_s=json.dumps({r["obj"]: r["s"] for r in rows}))
    Path(str(out) + ".report.json").write_text(json.dumps(report, indent=2))
    print(f"\n[est] -> {out}  (+ .report.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
