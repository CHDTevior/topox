#!/usr/bin/env python3
"""Mean and 95% CI over the seed repetitions of the frozen protocol.

Every report must be a strict-acceptance score of the SAME checkpoint under the SAME
protocol at a DIFFERENT seed; anything else refuses, because a mean over points that do
not share the protocol is not a repetition study.
"""
import argparse, json, sys
import numpy as np
from scipy import stats

# what has to agree across the repetitions for their spread to be seed variance
SHARED = ["gen_ckpt_sha256", "eval_ckpt_sha256", "pool", "steps", "cfg_text",
          "subset", "val_n", "strict_acceptance", "cohort_exclude_clips",
          # gen_batch sets the chunk boundaries and generation reseeds per chunk, so a different
          # batch is a different sampling procedure; eval_order fixes which clips share a pool
          "gen_batch", "eval_order_sha256"]

METRICS = [
    ("R@1",           lambda d: d["text_to_gen"]["rprec"]["1"]),
    ("R@2",           lambda d: d["text_to_gen"]["rprec"]["2"]),
    ("R@3",           lambda d: d["text_to_gen"]["rprec"]["3"]),
    ("ceil R@1",      lambda d: d["text_to_gt_ceiling"]["rprec"]["1"]),
    ("ceil R@2",      lambda d: d["text_to_gt_ceiling"]["rprec"]["2"]),
    ("ceil R@3",      lambda d: d["text_to_gt_ceiling"]["rprec"]["3"]),
    ("match text-gen", lambda d: d["matching"]["text_gen_cos"]),
    ("match text-GT",  lambda d: d["matching"]["text_gt_cos"]),
    ("match gen-GT",   lambda d: d["matching"]["gen_gt_cos"]),
    ("FID",           lambda d: d["fid_gen_vs_gt"]),
]


def arithmetic(p):
    """The matmul arithmetic a report was produced with. allow_tf32_matmul alone does not fix it:
    torch's "high" and "medium" both report True and are not the same arithmetic (codex 2026-09-12)."""
    rt = p.get("runtime") or (p.get("generation") or {}).get("runtime") or {}
    return (rt.get("allow_tf32_matmul"), rt.get("float32_matmul_precision"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--json_out")
    a = ap.parse_args()

    reps = []
    for f in a.reports:
        d = json.load(open(f))
        p = d["protocol"]
        if not p.get("strict_acceptance"):
            sys.exit(f"[refuse] {f} was not scored with --strict_acceptance")
        reps.append((f, d, p))

    ref_f, _, ref = reps[0]
    for f, _, p in reps[1:]:
        for k in SHARED:
            if p.get(k) != ref.get(k):
                sys.exit(f"[refuse] {f} differs from {ref_f} in {k}: {p.get(k)!r} vs {ref.get(k)!r}")
        if arithmetic(p) != arithmetic(ref):
            sys.exit(f"[refuse] {f} ran with (allow_tf32_matmul, float32_matmul_precision)="
                     f"{arithmetic(p)}, {ref_f} with {arithmetic(ref)}; a mean may not mix the two arithmetics")
        gp = (p.get("generation") or {}).get("plan_sha256")
        gr = (ref.get("generation") or {}).get("plan_sha256")
        if gp != gr:
            sys.exit(f"[refuse] {f} generated a different plan ({gp}) than {ref_f} ({gr})")
    seeds = [p.get("seed") for _, _, p in reps]
    if len(set(seeds)) != len(seeds):
        sys.exit(f"[refuse] the seeds repeat: {seeds}")
    reps.sort(key=lambda r: r[2]["seed"])   # so the per-seed column reads in the order the header names
    seeds = [p["seed"] for _, _, p in reps]

    n = len(reps)
    if n < 2:
        sys.exit("[refuse] a confidence interval needs at least two repetitions")
    tcrit = float(stats.t.ppf(0.975, n - 1))

    print(f"{n} repetitions of pool {ref['pool']} / {ref['steps']} steps / cfg {ref['cfg_text']} "
          f"at seeds {seeds}, arithmetic={arithmetic(ref)}, strict acceptance")
    print(f"{'metric':16s} {'mean':>10s} {'sd':>10s} {'95% CI':>22s}   per-seed")
    out = {"n": n, "seeds": seeds, "arithmetic": list(arithmetic(ref)), "t_crit": tcrit, "metrics": {}}
    for name, get in METRICS:
        v = np.array([get(d) for _, d, _ in reps], dtype=float)
        m, sd = float(v.mean()), float(v.std(ddof=1))
        half = tcrit * sd / np.sqrt(n)
        print(f"{name:16s} {m:10.4f} {sd:10.4f}   [{m-half:8.4f},{m+half:8.4f}]   "
              + " ".join(f"{x:.4f}" for x in v))
        out["metrics"][name] = {"mean": m, "sd": sd, "half_width": float(half),
                                "lo": float(m - half), "hi": float(m + half),
                                "values": [float(x) for x in v],
                                "by_seed": dict(zip([str(s) for s in seeds], [float(x) for x in v]))}
    if a.json_out:
        json.dump(out, open(a.json_out, "w"), indent=2)
        print(f"\nwrote {a.json_out}")


if __name__ == "__main__":
    main()
