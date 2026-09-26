#!/usr/bin/env python3
"""How much of the per-cell normalisation comes from the validation clips.

The published statistics (data/noik_rig_stats_v2.npz) are taken over all of a rig's clips. This
measures what they would have been over the training clips alone, and reports the difference in the
units the model actually sees: the served value is (x - mu) / (max(sigma, std_min) + floor), so a
shift of the mean matters in sigma and a change of sigma matters relatively.

It does not rescan the corpus. The artifact carries float64 mean, std and a per-cell count, so the
all-clip sums are exact; scanning the 3,899 validation clips and subtracting gives the training-only
sums exactly. --noop_rigs re-derives the ALL-clip statistics of a few rigs from their clips and
checks them against the artifact, because a subtraction is only as good as the convention it
assumes (three channel partitions, src/data/ktjd17/species_stats.py:220-250).
"""
import argparse, json, os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

C = 17


def accumulate(args):
    """Sums over one rig's clip list, in the corpus's three channel partitions."""
    rig, paths, Jmax = args
    n = np.zeros((Jmax, C), np.float64)
    s = np.zeros((Jmax, C), np.float64)
    q = np.zeros((Jmax, C), np.float64)
    for p in paths:
        with np.load(p) as z:
            m = np.asarray(z["motion"], dtype=np.float64)
            hv = np.asarray(z["heading_valid"], dtype=bool)
        T, J = m.shape[0], m.shape[1]
        blk = m[:, :, 0:13]                       # every frame, every joint
        n[:J, 0:13] += T; s[:J, 0:13] += blk.sum(0); q[:J, 0:13] += (blk * blk).sum(0)
        r = m[:, 0, 13:15]                        # every frame, root row only
        n[0, 13:15] += T; s[0, 13:15] += r.sum(0); q[0, 13:15] += (r * r).sum(0)
        if hv.any():                              # root row, heading-valid frames only
            h = m[hv, 0, 15:17]
            n[0, 15:17] += h.shape[0]; s[0, 15:17] += h.sum(0); q[0, 15:17] += (h * h).sum(0)
    return rig, n, s, q


def run(pool, jobs):
    out = {}
    for rig, n, s, q in pool.map(accumulate, jobs, chunksize=1):
        out[rig] = (n, s, q)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset/ktjd17_pzh312_noik_v2")
    ap.add_argument("--rig_stats", default="data/noik_rig_stats_v2.npz")
    ap.add_argument("--percell", default="data/noik_norm_stats_v2.npz")
    ap.add_argument("--exclude", default="configs/pilot_animal_only_exclusions.json")
    ap.add_argument("--std_min", type=float, default=0.05)
    ap.add_argument("--std_floor", type=float, default=1e-6)
    ap.add_argument("--noop_rigs", type=int, default=3,
                    help="re-derive this many rigs' ALL-clip statistics and check them against the artifact")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "16")))
    ap.add_argument("--json_out")
    a = ap.parse_args()

    root = Path(a.root)
    S = np.load(a.rig_stats, allow_pickle=True)
    rig_ids = [str(r) for r in S["rig_ids"]]
    idx = {r: i for i, r in enumerate(rig_ids)}
    mean_all, std_all, cnt_all = S["mean"], S["std"], S["count"]
    R, Jmax, _ = mean_all.shape

    P = np.load(a.percell, allow_pickle=True)
    pmeta = json.loads(str(P["__meta"]))
    if float(pmeta["std_min"]) != a.std_min or float(pmeta["std_floor"]) != a.std_floor:
        raise SystemExit(f"[refuse] the percell artifact was built at std_min={pmeta['std_min']} "
                         f"floor={pmeta['std_floor']}, not {a.std_min}/{a.std_floor}")
    sup = np.asarray(P["supervise_mask"], dtype=bool)
    vm = np.asarray(S["valid_mask"], dtype=bool)
    prig = [str(r) for r in P["rig_ids"]]
    if prig != rig_ids:
        raise SystemExit("[refuse] the percell artifact and the rig statistics list different rigs")

    drop = json.loads(Path(a.exclude).read_text())["clips"]
    drop = set(drop) if isinstance(drop, dict) else set(drop)
    rows = [json.loads(l) for l in open(root / "manifests" / "clips.jsonl")]
    val, allc, kept_rigs = defaultdict(list), defaultdict(list), set()
    for r in rows:
        if r.get("status") != "accept":
            continue
        rig, cid = str(r["rig_id"]), str(r["clip_id"])
        allc[rig].append(root / r["motion_relpath"])
        if cid in drop:
            continue
        kept_rigs.add(rig)
        if str(r.get("split")) == "val":
            val[rig].append(root / r["motion_relpath"])
    nval = sum(len(v) for v in val.values())
    print(f"[scan] {nval} validation clips over {len(val)} rigs; {len(kept_rigs)} rigs survive the exclusion")

    with ProcessPoolExecutor(a.workers) as ex:
        noop_rigs = sorted(val, key=lambda r: len(allc[r]))[:a.noop_rigs]
        got = run(ex, [(r, val[r], Jmax) for r in sorted(val)]
                  + [("__all__" + r, allc[r], Jmax) for r in noop_rigs])

    # the artifact's own convention, re-derived from the clips of a few rigs
    worst = 0.0
    for rig in noop_rigs:
        n, s, q = got["__all__" + rig]
        i = idx[rig]
        mu = np.where(n > 0, s / np.maximum(n, 1), 0.0)
        var = np.where(n > 0, q / np.maximum(n, 1) - mu * mu, 0.0)
        sd = np.sqrt(np.maximum(var, 0.0))
        if not np.array_equal(n, cnt_all[i]):
            raise SystemExit(f"[refuse] no-op check: {rig} re-derives a different frame count")
        d = max(np.abs(mu - mean_all[i]).max(), np.abs(sd - std_all[i]).max())
        worst = max(worst, float(d))
        print(f"[no-op] {rig:22s} {len(allc[rig]):5d} clips  max|delta| vs artifact {d:.3e}")
    if worst > 1e-8:
        raise SystemExit(f"[refuse] no-op check off by {worst:.3e}; the accumulation convention differs "
                         "from the one that built the artifact, so the subtraction below is not valid")

    # training-only statistics by exact subtraction
    s1_all = mean_all * cnt_all
    s2_all = (std_all ** 2 + mean_all ** 2) * cnt_all
    n_v = np.zeros_like(cnt_all); s1_v = np.zeros_like(s1_all); s2_v = np.zeros_like(s2_all)
    for rig, (n, s, q) in got.items():
        if rig.startswith("__all__"):
            continue
        i = idx[rig]
        n_v[i] = n; s1_v[i] = s; s2_v[i] = q
    n_t = cnt_all - n_v
    if (n_t < 0).any():
        raise SystemExit("[refuse] a cell has more validation frames than the artifact counts in total")
    keep = np.zeros(R, bool)
    for r in kept_rigs:
        keep[idx[r]] = True
    region = keep[:, None, None] & vm & (cnt_all > 0)     # the animal rigs' real cells
    if (n_t[region & sup] <= 0).any():
        raise SystemExit("[refuse] a supervised cell has no training frames left")

    with np.errstate(invalid="ignore", divide="ignore"):
        mu_t = np.where(n_t > 0, (s1_all - s1_v) / np.maximum(n_t, 1), 0.0)
        var_t = np.where(n_t > 0, (s2_all - s2_v) / np.maximum(n_t, 1) - mu_t ** 2, 0.0)
    sd_t = np.sqrt(np.maximum(var_t, 0.0))

    # what the model is SERVED, through the artifact's own derivation and float32 round trip
    # (scripts/_build_pzh312_norm_stats.py:55-84, read back in src/data/ktjd17_incontext.py:671):
    # a valid cell keeps its mean, a supervised cell divides by max(sigma, std_min), a cell that is
    # exactly constant is removed from supervision and divides by 1 (codex 2026-09-12)
    def serve(mu, sd, supervised):
        m = np.where(vm, mu, 0.0).astype(np.float32)
        d = np.where(supervised, np.maximum(sd, a.std_min), 1.0)
        d = (d - a.std_floor).astype(np.float32) + np.float32(a.std_floor)
        return m, d
    sup_t = vm & (sd_t != 0.0)       # a cell that is exactly constant on the training clips leaves supervision
    mu_serv_all, den_all = np.asarray(P["mean"]), np.asarray(P["std"]) + np.float32(a.std_floor)
    mu_serv_t, den_t = serve(mu_t, sd_t, sup_t)
    flip = region & (sup != sup_t)
    cells = region & sup & sup_t     # compare only where BOTH are supervised
    shift = np.abs(mu_serv_t - mu_serv_all)[cells] / den_all[cells]
    scale = np.abs(den_t[cells] / den_all[cells] - 1.0)

    def report(name, v):
        print(f"  {name:26s} median {np.median(v):.2e}   p99 {np.quantile(v, 0.99):.2e}   "
              f"max {v.max():.2e}")
        return {"median": float(np.median(v)), "p99": float(np.quantile(v, 0.99)), "max": float(v.max())}

    fr_v = int(n_v[keep, 0, 0].sum()); fr_a = int(cnt_all[keep, 0, 0].sum())
    print(f"[delta] over {int(cells.sum()):,} cells supervised under both, of {int(keep.sum())} rigs "
          f"({fr_v:,} validation frames removed of {fr_a:,}); "
          f"{int(flip.sum())} cells change supervision status")
    out = {"n_cells": int(cells.sum()), "n_rigs": int(keep.sum()), "n_val_clips": nval,
           "n_frames_val": fr_v, "n_frames_all": fr_a, "n_supervision_flips": int(flip.sum()),
           "noop_max_abs_delta": worst,
           "mean_shift_in_sigma": report("|d mu| / sigma", shift),
           "scale_change": report("|d sigma| / sigma", scale)}
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(out, indent=2))
        print(f"wrote {a.json_out}")


if __name__ == "__main__":
    main()
