"""Select and freeze the held-out TOPOLOGY set for the unseen-topology protocol.

Held out at parent-tree level and closed under object_type: sex and age variants of a
species share one rig, so holding out `PZ_Bongo_Male` while training on `PZ_Bongo_Female`
would leave the "unseen" topology in training under a sibling's name.

A held-out tree contributes its ENTIRE clip inventory (train + val) to the unseen
evaluation set. TrueBones trees carry a median of 1 val clip, so a val-only holdout would
be unscorable.

Selection is deterministic and stratified so the held-out set spans morphology rather than
being sampled uniformly (a uniform sample of 194 trees would be dominated by mid-sized PZ
quadrupeds, and the result would say nothing about the bodies that motivate the paper).
For each held-out tree we record the distance to its nearest RETAINED tree, so the
degradation can be reported against how far the body is from anything seen — which is the
question, not the aggregate.

This script writes the list once. It is a pre-registration: run it, commit the JSON, and do
not re-run it after seeing any result.

    python scripts/_build_holdout_trees.py --out data/holdout_trees_v1.json \
        --max_train_frac 0.08
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import sys
sys.setrecursionlimit(20000)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.anytop_dataset import _longest_prefix_match  # noqa: E402

# Two buckets, because one number cannot answer two questions. A uniform draw estimates
# average unseen performance over the eligible frame; a descriptor-diverse panel probes
# morphological extremes. Both are excluded in the SAME retrain, so they cost one chain.
N_PZ_REP, N_TB_REP = 12, 8      # representative: uniform over eligible topologies
N_PZ_STRESS, N_TB_STRESS = 8, 7  # stress: farthest-point, morphological outliers


EXCLUDED_TREE_IDS: frozenset = frozenset()   # set from --exclude_artifact before any draw


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/animo4d_L4TB_plus_human_v4b272neutral")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_pz_rep", type=int, default=N_PZ_REP)
    ap.add_argument("--n_tb_rep", type=int, default=N_TB_REP)
    ap.add_argument("--n_pz_stress", type=int, default=N_PZ_STRESS)
    ap.add_argument("--n_tb_stress", type=int, default=N_TB_STRESS)
    ap.add_argument("--sample_seed", type=int, default=20260801,
                    help="seed for the representative bucket's uniform draw; fixed and recorded "
                         "in the frozen artifact so the draw is reproducible")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing freeze whose content differs. Re-registering "
                         "after seeing results invalidates the pre-registration; only use this "
                         "before any training has run against the old list.")
    ap.add_argument("--exclude_artifact", type=str, default=None,
                    help="a previously frozen holdout artifact whose topologies are removed from "
                         "CANDIDACY, so this draw comes from the pool that draw retained. Used "
                         "when an earlier holdout has been demoted to development data: its "
                         "topologies go back into training and must not be re-drawn, or the new "
                         "set would inherit the old one's contamination. The eligibility cap is "
                         "still computed over ALL topologies of the source, so the size "
                         "restriction is identical to the original draw — only candidacy narrows, "
                         "and the artifact records that narrowing as part of its sampling frame.")
    # 0.08 because that is the value the frozen v1 draw was made with. A default that does not
    # reproduce the pre-registration is a trap: re-running the builder to regenerate the freeze
    # gave "pz: budget exhausted at 7/8 trees" instead of the frozen set. It failed loudly rather
    # than silently producing a different set, but the default should simply be the frozen one.
    ap.add_argument("--max_train_frac", type=float, default=0.08,
                    help="hard cap on the fraction of the corpus's training clips the holdout "
                         "may remove; the run fails rather than silently holding out fewer trees")
    return ap.parse_args()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_form(parents: tuple[int, ...]) -> str:
    """AHU canonical form of a rooted UNORDERED tree: recursively sort the children's
    canonical strings. Two rigs that are the same skeleton written with different joint
    indexings collapse to the same string here, which a raw `parents` tuple does not —
    grouping on the raw tuple counted 194 topologies where there are 179, and put two
    trees in the holdout that are isomorphic to retained ones."""
    ch = defaultdict(list)
    root = None
    for j, p in enumerate(parents):
        if p < 0 or p == j:
            if root is None:
                root = j
        else:
            ch[p].append(j)
    if root is None:
        root = 0
    def rec(v):
        return "()" if not ch[v] else "(" + "".join(sorted(rec(c) for c in ch[v])) + ")"
    return rec(root)


def tree_features(parents: tuple[int, ...]) -> dict:
    """Morphology descriptors that do not depend on joint naming or on the motions."""
    J = len(parents)
    children = Counter(p for p in parents if p >= 0)
    leaves = sum(1 for j in range(J) if j not in children)
    depths = []
    for j in range(J):
        d, k, guard = 0, j, 0
        while k >= 0 and parents[k] >= 0 and parents[k] != k and guard < J + 1:
            k = parents[k]
            d += 1
            guard += 1
        depths.append(d)
    return {"J": J,
            "max_branching": max(children.values()) if children else 0,
            "n_leaves": leaves,
            "max_depth": max(depths),
            "mean_depth": float(np.mean(depths)),
            "n_branch_nodes": sum(1 for c in children.values() if c >= 2)}


FEAT_KEYS = ("J", "max_branching", "n_leaves", "max_depth", "mean_depth", "n_branch_nodes")


def main():
    args = parse_args()
    # Loaded before anything is drawn: an exclusion applied after the draw would be a filter on a
    # result rather than a restriction of the sampling frame, and the two are not the same thing.
    global EXCLUDED_TREE_IDS
    EXCLUDED_TREE_IDS = frozenset()
    if args.exclude_artifact:
        _ex = json.loads(Path(args.exclude_artifact).read_text())
        EXCLUDED_TREE_IDS = frozenset(t["tree_id"] for t in _ex["held_out_trees"])
        if not EXCLUDED_TREE_IDS:
            raise SystemExit(f"--exclude_artifact {args.exclude_artifact} lists no held trees; "
                             f"an exclusion that excludes nothing is a silent no-op.")
    root = Path(args.data_root)
    cond = np.load(root / "cond.npy", allow_pickle=True).item()
    keys = sorted(cond.keys(), key=len, reverse=True)

    def read_split(n):
        return [l.strip() for l in (root / "splits" / n).read_text().splitlines()
                if l.strip() and not l.startswith("#")]

    n_train, n_val = Counter(), Counter()
    for fn in read_split("train.txt"):
        o = _longest_prefix_match(fn, keys)
        if o is None:
            raise SystemExit(f"unresolved train file {fn!r}")
        n_train[o] += 1
    for fn in read_split("val.txt"):
        o = _longest_prefix_match(fn, keys)
        if o is None:
            raise SystemExit(f"unresolved val file {fn!r}")
        n_val[o] += 1

    def parents_of(o):
        v = cond[o]
        p = v["parents"] if isinstance(v, dict) else v[0]
        return tuple(int(x) for x in np.asarray(p).ravel())

    by_tree = defaultdict(list)          # canonical form -> object types
    raw_forms = defaultdict(set)         # canonical form -> distinct raw parents arrays
    for o in cond:
        p = parents_of(o)
        c = canonical_form(p)
        by_tree[c].append(o)
        raw_forms[c].add(p)

    trees = []
    for canon, objs in by_tree.items():
        objs = sorted(objs)
        sig = parents_of(objs[0])        # a representative indexing, for the descriptors
        src = ("human" if any(o.upper().startswith("HML") for o in objs)
               else "pz" if objs[0].startswith("PZ_") else "truebones")
        f = tree_features(sig)
        trees.append({"tree_id": hashlib.sha256(canon.encode()).hexdigest()[:16],
                      "canonical_form_sha256": hashlib.sha256(canon.encode()).hexdigest(),
                      "n_raw_parent_arrays": len(raw_forms[canon]),
                      "source": src, "object_types": objs, "n_object_types": len(objs),
                      "train_clips": sum(n_train[o] for o in objs),
                      "val_clips": sum(n_val[o] for o in objs), **f})
    trees.sort(key=lambda t: t["tree_id"])
    if len(trees) != 179:
        raise SystemExit(f"expected 179 canonical topologies, found {len(trees)}. The corpus "
                         f"changed; re-derive the pre-registration deliberately.")

    total_train_clips = sum(t["train_clips"] for t in trees)

    # Feature space, standardized so no descriptor dominates the spread purely by scale.
    F = np.array([[t[k] for k in FEAT_KEYS] for t in trees], float)
    mu, sd = F.mean(0), F.std(0)
    sd[sd == 0] = 1.0
    Z = (F - mu) / sd

    def eligible(source: str) -> list[int]:
        """Candidates for holdout: topologies of this source whose removal does not take a
        disproportionate slice of training data. The size cap is a real restriction on the
        sampling frame and is named as such in the frozen artifact — the representative
        bucket is uniform over ELIGIBLE topologies, not over the corpus."""
        idx = [i for i, t in enumerate(trees) if t["source"] == source]
        # The cap is derived from every topology of this source, INCLUDING any that are excluded
        # from candidacy below: it defines how large a topology may be before removing it would
        # distort the retained corpus, which does not depend on who is eligible to be drawn.
        med = float(np.median([trees[i]["train_clips"] for i in idx]))
        out = [i for i in idx if trees[i]["train_clips"] <= max(2.0 * med, 40)]
        return [i for i in out if trees[i]["tree_id"] not in EXCLUDED_TREE_IDS]

    def select_uniform(source: str, n: int, taken: set) -> list[int]:
        """Uniform draw over eligible topologies, under the shared clip budget."""
        cand = [i for i in eligible(source) if i not in taken]
        rs = np.random.default_rng(args.sample_seed + (0 if source == "pz" else 1))
        order = list(rs.permutation(len(cand)))
        chosen, spent = [], 0
        for k in order:
            i = cand[k]
            if len(chosen) == n:
                break
            if spent + trees[i]["train_clips"] > budget_left[0]:
                continue
            chosen.append(i)
            spent += trees[i]["train_clips"]
        if len(chosen) < n:
            raise SystemExit(f"{source} representative: only {len(chosen)}/{n} fit the budget")
        budget_left[0] -= spent
        return sorted(chosen)

    def select(source: str, n: int, taken: set = frozenset()) -> list[int]:
        """Farthest-point sampling in descriptor space, seeded at the most extreme tree,
        so the held-out set spans morphology deterministically.

        Two guards, for different reasons. (i) The largest trees are excluded from
        candidacy: a tree carrying many times the source median would remove a large slice
        of training data for one topology, and the retained corpus must stay representative
        — the experiment is about unseen topology, not about training on less data.
        (ii) A single budget SHARED across sources, spent on TrueBones first. A per-source
        cap cannot work here: TrueBones is 1 % of the corpus, so any cap loose enough to
        admit 15 creature trees is far looser than PZ needs, and any cap tight enough for PZ
        blocks the very creature rigs the holdout exists to test."""
        cand = [i for i in eligible(source) if i not in taken]
        if len(cand) < n:
            raise SystemExit(f"{source}: only {len(cand)} trees under the size cap, need {n}")
        budget = budget_left[0]
        centre = Z[cand].mean(0)
        seed = max(cand, key=lambda i: float(np.linalg.norm(Z[i] - centre)))
        chosen, spent = [seed], trees[seed]["train_clips"]
        while len(chosen) < n:
            best, bestd = None, -1.0
            for i in cand:
                if i in chosen or spent + trees[i]["train_clips"] > budget:
                    continue
                dmin = min(float(np.linalg.norm(Z[i] - Z[c])) for c in chosen)
                if dmin > bestd:
                    best, bestd = i, dmin
            if best is None:
                raise SystemExit(f"{source}: budget exhausted at {len(chosen)}/{n} trees — "
                                 f"raise the cap deliberately rather than silently holding out fewer")
            chosen.append(best)
            spent += trees[best]["train_clips"]
        budget_left[0] -= spent
        return sorted(chosen)

    # One budget for the whole holdout, spent creatures-first: they are morphologically the
    # most informative and cost almost nothing, so they must not be crowded out by PZ.
    budget_left = [args.max_train_frac * total_train_clips]
    # REPRESENTATIVE FIRST, and deliberately so. Drawing the stress panel first would make
    # the representative bucket a uniform draw over the eligible-MINUS-stress complement,
    # which cannot estimate average performance over the eligible frame. A uniform sample is
    # allowed to contain an outlier; a representative maximum exceeding the stress maximum is
    # not an inversion, it is sampling. The stress panel is therefore what remains after the
    # representative draw, and is labelled as such rather than as a worst case.
    rep_tb = select_uniform("truebones", args.n_tb_rep, set())
    rep_pz = select_uniform("pz", args.n_pz_rep, set(rep_tb))
    taken = set(rep_tb) | set(rep_pz)
    st_tb = select("truebones", args.n_tb_stress, taken)
    st_pz = select("pz", args.n_pz_stress, taken | set(st_tb))
    bucket = {}
    for i in rep_tb + rep_pz:
        bucket[i] = "representative"
    for i in st_tb + st_pz:
        bucket[i] = "stress"
    held = sorted(bucket)
    held_set = set(held)
    retained = [i for i in range(len(trees)) if i not in held_set]

    # Distance from each held-out tree to the nearest retained tree: the axis the unseen
    # result should be broken out along.
    for i in held:
        dmin = min(float(np.linalg.norm(Z[i] - Z[j])) for j in retained)
        near = min(retained, key=lambda j: np.linalg.norm(Z[i] - Z[j]))
        trees[i]["bucket"] = bucket[i]
        trees[i]["dist_to_nearest_retained"] = round(dmin, 4)
        trees[i]["nearest_retained_tree"] = trees[near]["tree_id"]
        trees[i]["nearest_retained_example"] = trees[near]["object_types"][0]

    held_canon = {trees[i]["canonical_form_sha256"] for i in held}
    ret_canon = {trees[i]["canonical_form_sha256"] for i in retained}
    if held_canon & ret_canon:
        raise SystemExit(f"LEAK: {len(held_canon & ret_canon)} canonical forms appear in BOTH "
                         f"held-out and retained sets")
    held_obj_set = {o for i in held for o in trees[i]["object_types"]}
    ret_obj_set = {o for i in retained for o in trees[i]["object_types"]}
    if held_obj_set & ret_obj_set:
        raise SystemExit(f"LEAK: object types in both sets: {sorted(held_obj_set & ret_obj_set)[:5]}")
    held_objs = sorted(held_obj_set)
    tot_train = sum(t["train_clips"] for t in trees)
    held_train = sum(trees[i]["train_clips"] for i in held)
    held_eval = sum(trees[i]["train_clips"] + trees[i]["val_clips"] for i in held)

    out = {
        "protocol": "unseen-topology holdout (pre-registered)",
        "excluded_from_candidacy": {
            "artifact": args.exclude_artifact,
            "n_tree_ids": len(EXCLUDED_TREE_IDS),
            "why": ("topologies frozen by an earlier draw that has since been demoted to "
                    "development data. They return to training and are not eligible here, so "
                    "this set cannot inherit the earlier set's contamination."),
        } if args.exclude_artifact else None,
        "selection": {
            "representative": f"uniform draw (seed {args.sample_seed}) over ELIGIBLE topologies"
                              + (f" MINUS the {len(EXCLUDED_TREE_IDS)} excluded by "
                                 f"{args.exclude_artifact}" if args.exclude_artifact else "")
                              + f", "
                              f"i.e. those whose training-clip count is at most max(2x source "
                              f"median, 40). Estimates average unseen-topology performance over "
                              f"the ELIGIBLE frame, which the size cap restricts — not over the "
                              f"corpus. Fixed per-source quotas restrict it further.",
            "stress": "descriptor-diverse remainder panel: farthest-point sampling over the "
                      "eligible topologies REMAINING after the representative draw. It "
                      "over-samples descriptor extremes by construction, so it probes "
                      "descriptor-stress, NOT a worst case — farthest-point sampling does not "
                      "establish one. Report separately, never pooled with the representative "
                      "bucket.",
            "shared_budget_frac": args.max_train_frac,
            "reporting": "topology-macro averages within each bucket, not clip-weighted "
                         "aggregates, so a few clip-rich topologies cannot dominate.",
        },
        "descriptor_caveat": ("The six descriptors are adequate to STRATIFY this corpus but are "
                              "not a defensible skeleton similarity metric; "
                              "dist_to_nearest_retained is a stratification aid, not a distance "
                              "to be regressed against."),
        "descriptors": list(FEAT_KEYS),
        # The exact command, so the artifact says how to regenerate itself rather than relying on
        # defaults that can drift away from the frozen draw.
        "argv": sys.argv[1:],
        "n_trees_total": len(trees),
        "n_held_out_trees": len(held),
        "n_held_out_object_types": len(held_objs),
        "held_out_train_clips_removed": held_train,
        "held_out_train_clips_removed_frac": round(held_train / tot_train, 4),
        "held_out_eval_clips": held_eval,
        "retained_train_clips": tot_train - held_train,
        "held_out_object_types": held_objs,
        "held_out_trees": [trees[i] for i in held],
        "note": "A held-out tree contributes ALL its clips (train+val) to the unseen "
                "evaluation set. Retained splits keep their existing per-clip stratification.",
    }
    # The frozen artifact must be self-verifying: it records what it was derived FROM, so a
    # later reader can prove the corpus has not moved under it, and it refuses to overwrite an
    # existing freeze (a pre-registration that can be silently regenerated is not one).
    import subprocess, sys as _sys
    try:
        git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=Path(__file__).resolve().parents[1]).stdout.strip()
    except Exception:
        git_sha = None
    out["provenance"] = {
        "script_sha256": _file_sha256(Path(__file__).resolve()),
        "git_sha": git_sha or None,
        "argv": _sys.argv[1:],
        "args": vars(args),
        "numpy": np.__version__, "python": _sys.version.split()[0],
        "eligible_frame": {s_: sorted(trees[i]["tree_id"] for i in eligible(s_))
                           for s_ in ("pz", "truebones")},
        "counts_by_source": {s_: {"canonical": sum(1 for t in trees if t["source"] == s_),
                                  "raw_parent_arrays": sum(t["n_raw_parent_arrays"]
                                                           for t in trees if t["source"] == s_),
                                  "object_types": sum(t["n_object_types"]
                                                      for t in trees if t["source"] == s_)}
                             for s_ in ("pz", "truebones", "human")},
    }
    out["inputs"] = {
        "data_root": str(root.resolve()),
        "cond_npy_sha256": _file_sha256(root / "cond.npy"),
        "train_split_sha256": _file_sha256(root / "splits" / "train.txt"),
        "val_split_sha256": _file_sha256(root / "splits" / "val.txt"),
        "n_canonical_topologies": len(trees),
        "canonicalization": "AHU rooted-unordered-tree canonical form; tree_id = sha256 prefix",
    }
    out["retained_topologies"] = len(retained)
    def body_sha(d):
        return hashlib.sha256(json.dumps({k: v for k, v in d.items() if k != "artifact_sha256"},
                                         indent=2, sort_keys=True).encode()).hexdigest()
    out["artifact_sha256"] = body_sha(out)

    dest = Path(args.out)
    if dest.exists() and not args.force:
        prev = json.loads(dest.read_text())
        # RECOMPUTE the stored file's body hash rather than trusting the hash it carries:
        # otherwise a hand-edited held list that keeps the old hash is accepted as "identical".
        prev_actual = body_sha(prev)
        if prev_actual != prev.get("artifact_sha256"):
            raise SystemExit(f"REFUSED: {dest} is TAMPERED — its body hashes to {prev_actual} "
                             f"but it carries {prev.get('artifact_sha256')}. Investigate before "
                             f"doing anything else with it.")
        if prev_actual == out["artifact_sha256"]:
            print(f"[holdout] identical freeze already at {dest}; nothing to do")
            return 0
        raise SystemExit(f"REFUSED: {dest} exists with a DIFFERENT artifact_sha256 "
                         f"({prev_actual} vs {out['artifact_sha256']}). A frozen "
                         f"pre-registration must not be silently replaced. Pass --force only if "
                         f"you intend to re-register, and say so in the paper.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))

    for b in ("representative", "stress"):
        sel = [trees[i] for i in held if trees[i]["bucket"] == b]
        print(f"[holdout] {b:15s}: {len(sel):2d} topologies, "
              f"{sum(t['train_clips'] for t in sel):5d} train clips removed, "
              f"{sum(t['train_clips']+t['val_clips'] for t in sel):5d} eval clips, "
              f"dist {min(t['dist_to_nearest_retained'] for t in sel):.2f}"
              f"-{max(t['dist_to_nearest_retained'] for t in sel):.2f}")
    print(f"[holdout] {len(held)} trees / {len(held_objs)} object types held out")
    print(f"[holdout] training clips removed: {held_train} / {tot_train} "
          f"({100*held_train/tot_train:.2f}%)")
    print(f"[holdout] unseen evaluation clips: {held_eval}")
    for src in ("pz", "truebones"):
        sel = [trees[i] for i in held if trees[i]["source"] == src]
        if not sel:
            continue
        print(f"[holdout] {src}: {len(sel)} trees, J {min(t['J'] for t in sel)}-"
              f"{max(t['J'] for t in sel)}, branching {min(t['max_branching'] for t in sel)}-"
              f"{max(t['max_branching'] for t in sel)}, "
              f"eval clips {sum(t['train_clips']+t['val_clips'] for t in sel)}")
        for t in sorted(sel, key=lambda t: -t["dist_to_nearest_retained"])[:4]:
            print(f"            far: {t['object_types'][0]:34s} J={t['J']:3d} "
                  f"d={t['dist_to_nearest_retained']:.2f} (nearest kept: {t['nearest_retained_example']})")
    print(f"[holdout] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
