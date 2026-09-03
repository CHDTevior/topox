"""Per-skeleton generation quality against per-skeleton training volume (measurement M1).

The question: among SEEN skeletons, is the generator's shortfall relative to its own
paired ground-truth reference associated with how many training clips that skeleton had?
This is a decision tool for where to spend compute. It is NOT evidence about unseen
topology — a null association cannot establish unseen-topology ability, and a positive one
only indicates volume dependence among skeletons the model trained on.

Reads the embedding dump written by `_eval_codeflow_gen_in_evalspace.py --emb_dump`; no
GPU, no model, no motion data. It only regroups rows that were already scored.

WHY THE PRIMARY OUTCOME IS A DIFFERENCE, AND WHAT IT DOES NOT FIX.
The frozen evaluator was trained on the same unbalanced corpus, so it may embed
high-volume skeletons better regardless of generator quality. Subtracting each clip's
paired `text.gt` removes a shared ADDITIVE evaluator/text-difficulty term. It does not
remove multiplicative compression, nonlinear evaluator error, generated-vs-real domain
distortion, or differing action difficulty. `text.gt` is a paired reference score, not a
mathematical ceiling: a valid alternative generation can exceed it. A ratio is rejected
because these are cosines whose denominator can be zero or negative.

CONFOUNDS ARE REPORTED, NOT ASSUMED AWAY. Within PZ, training volume co-varies with the
unique-caption inventory (rho 0.889), joint count (+0.251) and mean clip length (-0.283).
`val_clips` is NOT partialled out despite correlating at 0.998: it is a precision term, not
a confound, and conditioning on it would remove most of the signal by construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.anytop_dataset import _longest_prefix_match  # noqa: E402

SCHEMA = 2
N_BOOT = 10000
N_PERM = 10000
NORM_TOL = 5e-2
MEAN_TOL = 1e-4
RESID_TOL = 1e-8      # partial correlation is undefined below this residual-norm ratio

REQUIRED = ("gen_emb", "text_emb", "gt_emb", "motion_ids", "row_keys", "selection",
            "overall_matching_mean", "data_root", "target_frames", "decoded_frames",
            "n_soft_clamped", "train_split_md5", "val_split_md5")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dump", required=True)
    ap.add_argument("--token_index", default=None,
                    help="train/index.jsonl of the token cache; supplies the unique-caption "
                         "inventory and joint count used as covariates. Required.")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--allow_root_override", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sensitivity_min_val", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


# ---------------------------------------------------------------- statistics


def _rankdata(v):
    v = np.asarray(v, float)
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v))
    sv = v[order]
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def _corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def spearman(x, y):
    return _corr(_rankdata(x), _rankdata(y))


def partial_spearman(x, y, controls):
    """Rank-partial correlation. Returns NaN when the controls span x (or y) to numerical
    precision — in that case the statistic is undefined, and returning a number computed
    from floating-point residue would be worse than returning nothing. An earlier version
    of this file lacked this gate and silently reported ~1e-3 'after controlling' values
    that were pure round-off, because the control was an exact copy of x."""
    rx, ry = _rankdata(x), _rankdata(y)
    Z = np.column_stack([np.ones(len(rx))] + [_rankdata(c) for c in controls])
    def resid(v):
        beta, *_ = np.linalg.lstsq(Z, v, rcond=None)
        r = v - Z @ beta
        centred = v - v.mean()
        ratio = np.linalg.norm(r) / max(np.linalg.norm(centred), 1e-30)
        return r, ratio
    ex, rax = resid(rx)
    ey, ray = resid(ry)
    if rax < RESID_TOL or ray < RESID_TOL:
        return float("nan"), {"residual_ratio_x": float(rax), "residual_ratio_y": float(ray),
                              "undefined": True,
                              "why": "controls span the variable to numerical precision"}
    return _corr(ex, ey), {"residual_ratio_x": float(rax), "residual_ratio_y": float(ray),
                           "undefined": False}


def boot_ci(groups, xmap, rng, controls_by_skel=None, n_boot=N_BOOT):
    """Two-stage bootstrap: resample skeletons, then clips within each resampled skeleton.
    Per-skeleton means rest on 3-46 clips, so a skeleton-only bootstrap would understate
    the interval. When `controls_by_skel` is given the ADJUSTED statistic is bootstrapped,
    which is the one the paper would quote."""
    names = list(groups)
    out = np.full(n_boot, np.nan)
    for b in range(n_boot):
        pick = rng.integers(0, len(names), len(names))
        xs, ys, cs = [], [], []
        for p in pick:
            nm = names[p]
            v = groups[nm]
            ys.append(float(np.mean(v[rng.integers(0, len(v), len(v))])))
            xs.append(xmap[nm])
            if controls_by_skel is not None:
                cs.append(controls_by_skel[nm])
        xs, ys = np.array(xs, float), np.array(ys, float)
        if controls_by_skel is None:
            out[b] = spearman(xs, ys)
        else:
            C = np.array(cs, float)
            r, _ = partial_spearman(xs, ys, [C[:, k] for k in range(C.shape[1])])
            out[b] = r
    if np.all(np.isnan(out)):
        return None, None
    lo, hi = np.nanpercentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def perm_p(x, y, rng, n_perm=N_PERM):
    """Permutation test for the MARGINAL Spearman. Valid only if skeleton outcomes are
    exchangeable under the null; here they are not exactly (heterogeneous precision from
    3-46 clips, systematic covariates), so this is reported as unadjusted."""
    obs = abs(spearman(x, y))
    y = np.asarray(y, float).copy()
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(y)
        if abs(spearman(x, y)) >= obs:
            hits += 1
    return float((hits + 1) / (n_perm + 1))


# ---------------------------------------------------------------- loading


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_and_validate(args):
    d = torch.load(args.emb_dump, map_location="cpu", weights_only=False)

    if d.get("schema") != SCHEMA:
        raise SystemExit(f"REFUSED: dump schema {d.get('schema')} != {SCHEMA}. Re-run the eval "
                         f"with the current --emb_dump code; an older dump lacks the provenance "
                         f"and soft-clamp fields this analysis requires.")
    missing = [k for k in REQUIRED if d.get(k) is None]
    if missing:
        raise SystemExit(f"REFUSED: dump missing required fields {missing}")

    if d.get("gt_baseline"):
        raise SystemExit("REFUSED: GT-baseline dump — it holds no generated motion, so the "
                         "deficit is identically zero.")

    if int(d["n_soft_clamped"]) != 0:
        raise SystemExit(f"REFUSED: {d['n_soft_clamped']} clips were soft-clamped (decode shorter "
                         f"than GT, scored as a prefix against full-duration GT). If clamping "
                         f"covaries with clip length or skeleton it biases this analysis "
                         f"directly. Fix the decode length, do not analyse around it.")

    gen, txt, gt = (d[k].float() for k in ("gen_emb", "text_emb", "gt_emb"))
    mids = [str(m) for m in d["motion_ids"]]
    n = len(mids)
    for name, t in (("gen_emb", gen), ("text_emb", txt), ("gt_emb", gt)):
        if t.ndim != 2 or t.shape[0] != n:
            raise SystemExit(f"REFUSED: {name} shape {tuple(t.shape)} incompatible with {n} rows")
        if not torch.isfinite(t).all():
            raise SystemExit(f"REFUSED: {name} has non-finite values")
        if (t.norm(dim=-1) - 1.0).abs().max().item() > NORM_TOL:
            raise SystemExit(f"REFUSED: {name} rows are not unit-norm; the dot products would "
                             f"not be cosines")

    for k in ("target_frames", "decoded_frames", "row_keys"):
        if len(d[k]) != n:
            raise SystemExit(f"REFUSED: len({k})={len(d[k])} != {n} rows")
    if len(set(tuple(r) for r in d["row_keys"])) != n:
        raise SystemExit("REFUSED: row_keys are not unique")

    sel = d["selection"]
    for k in ("n_total", "n_evaluated", "idxs", "balanced", "exclude_truebones"):
        if k not in sel:
            raise SystemExit(f"REFUSED: selection missing {k}")
    if sel["balanced"] or sel["exclude_truebones"]:
        raise SystemExit(f"REFUSED: cohort-altering flags set: {sel}")
    idxs = list(sel["idxs"])
    if len(idxs) != n or len(set(idxs)) != n:
        raise SystemExit(f"REFUSED: selection.idxs has {len(idxs)} entries "
                         f"({len(set(idxs))} unique) for {n} rows")
    if sorted(idxs) != list(range(int(sel["n_total"]))):
        raise SystemExit(f"REFUSED: selection.idxs is not the full cohort 0..{sel['n_total']-1}; "
                         f"per-skeleton means would be over an unstated subset")
    mm = d["overall_matching_mean"]
    if not np.isfinite(mm):
        raise SystemExit("REFUSED: overall_matching_mean is not finite")

    match_gen = (txt * gen).sum(-1).numpy()
    if abs(float(match_gen.mean()) - float(mm)) > MEAN_TOL:
        raise SystemExit(f"REFUSED: recomputed matching mean {match_gen.mean():.6f} != reported "
                         f"{float(mm):.6f}; dump and report disagree")

    root = Path(args.data_root or d["data_root"])
    if args.data_root and Path(args.data_root).resolve() != Path(d["data_root"]).resolve():
        if not args.allow_root_override:
            raise SystemExit(f"REFUSED: --data_root {args.data_root} != dump's {d['data_root']}")

    # The independent variable is read from the CURRENT split files, so they must be the ones
    # the dump was produced against or the regression mixes two corpora.
    for fname, key in (("train.txt", "train_split_md5"), ("val.txt", "val_split_md5")):
        got = _md5(root / "splits" / fname)
        if got != d[key]:
            raise SystemExit(f"REFUSED: splits/{fname} md5 {got} != dump's {d[key]}; the split "
                             f"changed after generation, so training volume no longer matches "
                             f"the evaluated model")
    return d, gen, txt, gt, mids, root


def main():
    args = parse_args()
    if not args.token_index:
        raise SystemExit("--token_index is required: the unique-caption inventory covariate "
                         "cannot be derived from filenames (an earlier version tried, and the "
                         "result was an exact copy of the clip count).")
    rng = np.random.default_rng(args.seed)
    d, gen, txt, gt, mids, root = load_and_validate(args)

    m_gen = (txt * gen).sum(-1).numpy()
    m_gt = (txt * gt).sum(-1).numpy()
    deficit = m_gen - m_gt
    pair = (gen * gt).sum(-1).numpy()

    keys = sorted(np.load(root / "cond.npy", allow_pickle=True).item().keys(), key=len, reverse=True)

    # Covariates from the token-cache index: the per-clip primary caption (whose per-skeleton
    # unique count is the action-inventory proxy) and the joint count.
    cap_by_mid, joints_by_obj, tokens_by_obj = {}, {}, defaultdict(list)
    for line in Path(args.token_index).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        cap_by_mid[r["motion_id"]] = r["text"]
        joints_by_obj[r["object_type"]] = int(r["num_joints"])
        tokens_by_obj[r["object_type"]].append(int(r["n_valid_tokens"]))

    train_clips, uniq_caps = Counter(), defaultdict(set)
    for fn in (root / "splits" / "train.txt").read_text().splitlines():
        fn = fn.strip()
        if not fn or fn.startswith("#"):
            continue
        o = _longest_prefix_match(fn, keys)
        if o is None:
            raise SystemExit(f"REFUSED: unresolved train filename {fn!r}")
        train_clips[o] += 1
        mid = fn[:-4] if fn.endswith(".npy") else fn
        c = cap_by_mid.get(mid)
        if c is None:
            raise SystemExit(f"REFUSED: train clip {mid!r} absent from --token_index; the index "
                             f"and the split file describe different corpora")
        uniq_caps[o].add(c)

    g_gen, g_gt, g_def, g_pair = (defaultdict(list) for _ in range(4))
    for i, m in enumerate(mids):
        fn = m if m.endswith(".npy") else f"{m}.npy"
        o = _longest_prefix_match(fn, keys)
        if o is None:
            raise SystemExit(f"REFUSED: unresolved motion_id {m!r}")
        g_gen[o].append(m_gen[i]); g_gt[o].append(m_gt[i])
        g_def[o].append(deficit[i]); g_pair[o].append(pair[i])

    rows = []
    for o in sorted(g_gen):
        src = ("human" if o.upper().startswith("HML")
               else "pz" if o.startswith("PZ_") else "truebones")
        toks = tokens_by_obj.get(o, [])
        rows.append({
            "skeleton": o, "source": src,
            "train_clips": int(train_clips.get(o, 0)),
            "uniq_captions": int(len(uniq_caps.get(o, ()))),
            "num_joints": int(joints_by_obj.get(o, 0)),
            "mean_valid_tokens": float(np.mean(toks)) if toks else float("nan"),
            "val_clips": len(g_gen[o]),
            "matching_gen": float(np.mean(g_gen[o])),
            "matching_gt_reference": float(np.mean(g_gt[o])),
            "deficit": float(np.mean(g_def[o])),
            "pair_gen_gt": float(np.mean(g_pair[o])),
        })

    report = {
        "schema": SCHEMA, "emb_dump": args.emb_dump, "data_root": str(root),
        "flow_ckpt": d.get("flow_ckpt"), "flow_epoch": d.get("flow_epoch"),
        "eval_ckpt": d.get("eval_ckpt"), "cfg_scale": d.get("cfg_scale"),
        "steps": d.get("steps"), "seed_generation": d.get("seed"), "seed_analysis": args.seed,
        "n_generation_seeds": 1,
        "single_seed_caveat": ("All intervals are conditional on ONE generation seed. They "
                               "describe sampling over skeletons and clips, not over generations."),
        "n_soft_clamped": int(d["n_soft_clamped"]),
        "n_clips": len(mids), "n_skeletons": len(rows),
        "primary_outcome": "deficit = mean(text.gen - text.gt) per skeleton",
        "reference_note": ("matching_gt_reference is a PAIRED REFERENCE score, not a ceiling; a "
                           "valid alternative generation can exceed it. Subtracting it removes a "
                           "shared additive evaluator/text-difficulty term only."),
        "scope_note": ("Observational within-source association among SEEN skeletons. It does not "
                       "identify a causal data-volume effect and does not test unseen-topology "
                       "generalization."),
        "metric_version_note": ("R-precision in the companion report uses 20 shuffled pool-32 "
                                "repetitions (MotionMillion regime), which differs from the "
                                "consecutive-pool version at git HEAD. Matching, FID and the "
                                "quantities used here are unaffected."),
        "per_skeleton": rows,
    }

    print(f"[M1] {len(mids)} clips over {len(rows)} skeletons "
          f"(ep{d.get('flow_epoch')}, cfg={d.get('cfg_scale')}, soft-clamped={d['n_soft_clamped']})")

    COVS = ("uniq_captions", "num_joints", "mean_valid_tokens")
    for src in ("pz", "truebones"):
        sel = [r for r in rows if r["source"] == src]
        if len(sel) < 8:
            print(f"[M1] {src}: only {len(sel)} skeletons — correlations not computed")
            report[f"corr_{src}"] = None
            continue
        x = np.array([r["train_clips"] for r in sel], float)
        out = {"n_skeletons": len(sel),
               "train_clips_range": [int(x.min()), int(x.max())],
               "val_clips_range": [int(min(r["val_clips"] for r in sel)),
                                   int(max(r["val_clips"] for r in sel))]}
        for label in ("deficit", "matching_gen", "matching_gt_reference", "pair_gen_gt"):
            out[f"spearman_trainclips_vs_{label}"] = spearman(
                x, np.array([r[label] for r in sel], float))
        out["evaluator_volume_bias"] = out["spearman_trainclips_vs_matching_gt_reference"]
        out["evaluator_volume_bias_note"] = (
            "The reference score's own correlation with volume. If it is non-zero, the raw "
            "matching correlation cannot be read as generator quality.")

        for c in COVS:
            out[f"spearman_trainclips_vs_{c}"] = spearman(
                x, np.array([r[c] for r in sel], float))

        y = np.array([r["deficit"] for r in sel], float)
        C = np.array([[r[c] for c in COVS] for r in sel], float)
        if np.isfinite(C).all():
            rho_adj, diag = partial_spearman(x, y, [C[:, k] for k in range(C.shape[1])])
            out["partial_spearman_deficit_adjusted"] = rho_adj
            out["partial_diagnostics"] = diag
            out["adjusted_controls"] = list(COVS)
            groups = {r["skeleton"]: np.asarray(g_def[r["skeleton"]]) for r in sel}
            xmap = {r["skeleton"]: r["train_clips"] for r in sel}
            cmap = {r["skeleton"]: [r[c] for c in COVS] for r in sel}
            out["ci95_deficit_adjusted"] = boot_ci(groups, xmap, rng, controls_by_skel=cmap)
            out["ci95_deficit_marginal"] = boot_ci(groups, xmap, rng)
        else:
            out["partial_spearman_deficit_adjusted"] = None
            out["adjusted_controls"] = None

        out["perm_p_deficit_marginal_unadjusted"] = perm_p(x, y.copy(), rng)
        out["perm_p_note"] = ("Marginal statistic only, and it assumes skeleton outcomes are "
                              "exchangeable under the null. They are not exactly: per-skeleton "
                              "means have heterogeneous precision (3-46 clips) and systematic "
                              "covariates. Read it as indicative, not as a decision threshold.")

        keep = [r for r in sel if r["val_clips"] >= args.sensitivity_min_val]
        out["sensitivity_min_val_clips"] = {
            "threshold": args.sensitivity_min_val,
            "n_dropped": len(sel) - len(keep), "n_kept": len(keep),
            "spearman_deficit": (spearman(np.array([r["train_clips"] for r in keep], float),
                                          np.array([r["deficit"] for r in keep], float))
                                 if len(keep) >= 8 else None),
            "note": "Secondary only. val_clips correlates with train_clips at ~0.998, so this "
                    "threshold drops exactly the scarce skeletons under study."}
        report[f"corr_{src}"] = out

        adj = out.get("partial_spearman_deficit_adjusted")
        ci = out.get("ci95_deficit_adjusted") or (None, None)
        print(f"[M1] {src}: N={len(sel)}  train_clips {int(x.min())}..{int(x.max())}")
        print(f"[M1] {src}:   deficit  marginal rho={out['spearman_trainclips_vs_deficit']:+.3f}  "
              f"p={out['perm_p_deficit_marginal_unadjusted']:.4f} (unadjusted)")
        if adj is not None and np.isfinite(adj):
            print(f"[M1] {src}:   deficit  ADJUSTED rho={adj:+.3f} "
                  f"CI95=[{ci[0]:+.3f},{ci[1]:+.3f}]  controls={list(COVS)}")
        else:
            print(f"[M1] {src}:   deficit  ADJUSTED rho=UNDEFINED  {out.get('partial_diagnostics')}")
        print(f"[M1] {src}:   evaluator volume bias (ref score vs volume) = "
              f"{out['evaluator_volume_bias']:+.3f}")
        for c in COVS:
            print(f"[M1] {src}:   confound {c:18s} rho={out[f'spearman_trainclips_vs_{c}']:+.3f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, allow_nan=False))
        print(f"[M1] report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
