"""Does the joint-description text carry cross-rig structural information the raw name does not?

The gate for whether joint semantics is worth a slot in the tokenizer retrain. It must not be
circular: two earlier probes used cosine-argmax retrieval between name strings and were both
dominated by phrase length and function words rather than by anatomy, which made them useless as
evidence in either direction.

So the target here comes from the SKELETON GRAPH, never from the text:

  depth_norm    the joint's depth divided by the rig's maximum depth
  radius_norm   its rest-pose distance from the root, divided by the rig's rest radius
  height_norm   its rest-pose height relative to the rig's vertical extent
  subtree_norm  the fraction of the rig's joints in its subtree
  n_children    branching at that joint
  is_leaf       whether it terminates a chain

A linear probe is fitted on RETAINED rigs to map a joint's text embedding to those six numbers,
and scored on HELD rigs it never saw. Raw name embeddings and description embeddings go through
an identical pipeline, so the comparison isolates the rewrite.

Reading the result. If descriptions beat raw names on held rigs, the text carries structural
information that transfers across naming conventions, which is exactly what a shared codebook
needs. If they tie, the rewrite buys nothing and joint semantics should not consume a slot in the
one serial chain we can afford. Either answer is worth having before spending 141 GPU-hours.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import sys
import torch

sys.setrecursionlimit(20000)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.holdout_guard import canonical_form, verify_artifact  # noqa: E402

TARGETS = ("depth_norm", "radius_norm", "height_norm", "subtree_norm", "n_children", "is_leaf")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--descriptions", default="data/joint_descriptions_v1.json")
    ap.add_argument("--data_root", default="data/animo4d_L4TB_plus_human_v4b272neutral")
    ap.add_argument("--holdout_artifact", default="data/holdout_topologies_v1.json")
    ap.add_argument("--text_encoder", default="distilbert", choices=["distilbert", "llm2vec"],
                    help="which frozen sentence encoder to probe. The probe is non-circular, so "
                         "it can rank encoders directly — that is how the choice is made here "
                         "rather than by reputation.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--eval_on", default="held",
                    choices=["held", "retained_dev", "cross_source"],
                    help="held = score on the frozen held-out topologies. retained_dev = score "
                         "on a fold carved out of the RETAINED topologies, leaving the held set "
                         "untouched. Model selection must use retained_dev: choosing between "
                         "encoders or estimators on held scores turns the held set into "
                         "development feedback and it stops being a confirmatory test. "
                         "cross_source = fit on one naming convention and score on another, "
                         "which is what the description rewrite actually targets: the "
                         "structural probe can be satisfied by convention-local cues, so a "
                         "within-convention score under-measures cross-convention alignment.")
    ap.add_argument("--fit_source", default="pz", choices=["pz", "truebones"])
    ap.add_argument("--dev_frac", type=float, default=0.25)
    ap.add_argument("--dev_seed", type=int, default=20260802)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def structural_targets(parents: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    J = len(parents)
    q = np.zeros((J, 3))
    for j, p in enumerate(parents):
        p = int(p)
        q[j] = offsets[j] if (p < 0 or p == j) else q[p] + offsets[j]
    depth = np.zeros(J)
    for j in range(J):
        d, k, g = 0, j, 0
        while int(parents[k]) >= 0 and int(parents[k]) != k and g < J + 1:
            k = int(parents[k]); d += 1; g += 1
        depth[j] = d
    nch = np.zeros(J)
    for j, p in enumerate(parents):
        p = int(p)
        if 0 <= p != j:
            nch[p] += 1
    sub = np.ones(J)
    for j in np.argsort(-depth):
        p = int(parents[j])
        if 0 <= p != j:
            sub[p] += sub[j]
    r = np.linalg.norm(q - q[0], axis=1)
    hy = q[:, 1]
    span = max(hy.max() - hy.min(), 1e-9)
    return np.stack([
        depth / max(depth.max(), 1),
        r / max(r.max(), 1e-9),
        (hy - hy.min()) / span,
        sub / J,
        nch,
        (nch == 0).astype(float),
    ], axis=1)


def ridge_r2(Xtr, Ytr, Xte, Yte, lam=1.0):
    """Ridge on centred features; R^2 per target on the held set, against the TRAIN mean as the
    null predictor so a negative score means "worse than knowing nothing"."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A = (Xtr - mu) / sd
    B = (Xte - mu) / sd
    ym = Ytr.mean(0)
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (Ytr - ym))
    P = B @ W + ym
    ss_res = ((Yte - P) ** 2).sum(0)
    ss_tot = ((Yte - ym) ** 2).sum(0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def main():
    args = parse_args()
    root = Path(args.data_root)
    art = verify_artifact(args.holdout_artifact, root)
    held_canon = {t["canonical_form_sha256"] for t in art["held_out_trees"]}
    desc = json.loads(Path(args.descriptions).read_text())

    from src.data.anytop_dataset import AnyTopDataset
    ds = AnyTopDataset(data_root=str(root), split="all", num_frames=64, max_joints=144,
                       load_captions=False)

    import hashlib
    names, descs, Y, held, obj_of = [], [], [], [], []
    missing = 0
    for obj, c in ds.cond.items():
        par = np.asarray(c["parents"]).ravel().astype(int)
        off = np.asarray(c["offsets"], float).reshape(len(par), 3)
        jn = [str(x) for x in c["joint_names"]]
        t = structural_targets(par, off)
        isheld = hashlib.sha256(
            canonical_form(tuple(int(x) for x in par)).encode()).hexdigest() in held_canon
        for j, nm in enumerate(jn):
            d = desc.get(nm)
            if d is None:
                missing += 1
                continue
            names.append(nm); descs.append(d["description"])
            Y.append(t[j]); held.append(isheld); obj_of.append(obj)
    if missing:
        raise SystemExit(f"REFUSED: {missing} joints have no description; the comparison would "
                         f"be over different sets for the two arms")
    Y = np.asarray(Y); held = np.asarray(held)
    obj_of = np.asarray(obj_of)
    if args.eval_on == "retained_dev":
        # Carve a dev fold out of the RETAINED topologies, grouped by object type so no rig
        # straddles the split. The frozen held set is dropped entirely and never scored here.
        keep = ~held
        objs = sorted(set(obj_of[keep]))
        rs = np.random.default_rng(args.dev_seed)
        dev = set(np.asarray(objs)[rs.permutation(len(objs))[:max(1, int(round(
            args.dev_frac * len(objs))))]].tolist())
        score = np.array([o in dev for o in obj_of]) & keep
        fit = keep & ~score
        print(f"[probe] retained-dev mode: {len(dev)}/{len(objs)} retained object types in the "
              f"dev fold; the frozen held set is NOT touched")
    elif args.eval_on == "cross_source":
        src = np.array(["human" if o.upper().startswith("HML") else
                        "pz" if o.startswith("PZ_") else "truebones" for o in obj_of])
        keep = ~held                      # the frozen held set stays untouched here too
        fit = keep & (src == args.fit_source)
        other = "truebones" if args.fit_source == "pz" else "pz"
        score = keep & (src == other)
        print(f"[probe] cross-source: fit on {args.fit_source}, score on {other}; "
              f"held set untouched")
    else:
        score, fit = held, ~held
    print(f"[probe] {len(names)} joint instances, fit={fit.sum()} score={score.sum()} "
          f"(eval_on={args.eval_on})")

    from src.data.text_encoders import build as build_encoder
    torch.set_num_threads(8)
    enc = build_encoder(args.text_encoder, device=args.device)
    print(f"[probe] encoder={args.text_encoder} dim={enc.dim}")

    # Encode the UNIQUE strings once, then expand — the same name recurs on many rigs.
    res = {}
    for arm, texts in (("raw_name", names), ("description", descs)):
        uniq = sorted(set(texts))
        idx = {t: i for i, t in enumerate(uniq)}
        E = enc.encode(uniq).astype(np.float64)
        X = E[[idx[t] for t in texts]]
        r2 = ridge_r2(X[fit], Y[fit], X[score], Y[score])
        res[arm] = {k: float(v) for k, v in zip(TARGETS, r2)}
        res[arm]["mean"] = float(np.mean(r2))
        print(f"[probe] {arm:12s} unique={len(uniq):5d}  held R^2 mean={np.mean(r2):+.4f}")

    print(f"\n{'target':14s} {'raw name':>10s} {'description':>12s} {'delta':>8s}")
    for k in TARGETS:
        a, b = res["raw_name"][k], res["description"][k]
        print(f"{k:14s} {a:>10.4f} {b:>12.4f} {b-a:>+8.4f}")
    a, b = res["raw_name"]["mean"], res["description"]["mean"]
    print(f"{'MEAN':14s} {a:>10.4f} {b:>12.4f} {b-a:>+8.4f}")
    verdict = ("descriptions carry structural information the raw names do not"
               if b - a > 0.02 else
               "no material gain from the rewrite under this probe")
    print(f"\n[probe] {verdict}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"text_encoder": args.text_encoder, "encoder_dim": enc.dim,
             "eval_on": args.eval_on, "n_instances": len(names),
             "n_fit": int(fit.sum()), "n_score": int(score.sum()),
             "targets": list(TARGETS), "results": res, "verdict": verdict}, indent=2))
        print(f"[probe] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
