"""Derive held-out-aware evaluator manifests from the frozen topology pre-registration.

The evaluator reads `eval_splits/*.json` manifests rather than the `splits/*.txt` clip lists, so
the split filtering done for the VQVAE and the backbone does not reach it. Left alone it would
train on every held topology and select its checkpoint on them, and then be used to score the
"unseen" numbers — measuring rigs it had memorised.

Membership is resolved from the manifest `filename` through longest-prefix object resolution and
then the AHU canonical form. It is deliberately NOT resolved from the manifest's own
`object_type` field: for every TrueBones record that field is the *string* `"None"`, so filtering
on it would silently retain every creature rig.

Writes, for each input manifest, a `<name>_retained.json` containing only retained topologies,
plus one `held_<bucket>.json` per bucket pooled across all inputs. Existing manifests are never
modified.

    python scripts/_build_holdout_eval_manifests.py --out_dir data/holdout_eval_splits_v1
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
from src.data.holdout_guard import canonical_form, verify_artifact  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout_artifact", default="data/holdout_topologies_v1.json")
    ap.add_argument("--data_root", default="data/animo4d_L4TB_plus_human_v4b272neutral")
    ap.add_argument("--eval_splits", default=None, help="default <data_root>/eval_splits")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    root = Path(args.data_root)
    art = verify_artifact(args.holdout_artifact, root)
    bucket_of = {t["canonical_form_sha256"]: t["bucket"] for t in art["held_out_trees"]}

    cond = np.load(root / "cond.npy", allow_pickle=True).item()
    keys = sorted(cond.keys(), key=len, reverse=True)
    obj_bucket = {}
    for o, v in cond.items():
        par = v["parents"] if isinstance(v, dict) else v[0]
        c = canonical_form(tuple(int(x) for x in np.asarray(par).ravel()))
        obj_bucket[o] = bucket_of.get(hashlib.sha256(c.encode()).hexdigest())

    src_dir = Path(args.eval_splits) if args.eval_splits else root / "eval_splits"
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    held_pool: dict[str, list] = defaultdict(list)
    held_seen: dict[str, set] = defaultdict(set)
    report = {"holdout_artifact": str(args.holdout_artifact),
              "artifact_sha256": art["artifact_sha256"], "inputs": {}, "outputs": {}}

    names = ["train_main.json"] + sorted(p.name for p in src_dir.glob("val_*.json"))
    for name in names:
        src = src_dir / name
        if not src.exists():
            continue
        items = json.loads(src.read_text())
        if not isinstance(items, list):
            raise SystemExit(f"REFUSED: {src} is not a list of records")
        keep, drop = [], Counter()
        for it in items:
            fn = it.get("filename")
            if not fn:
                raise SystemExit(f"REFUSED: a record in {src} has no 'filename'")
            o = _longest_prefix_match(fn, keys)
            if o is None:
                raise SystemExit(f"REFUSED: {fn!r} in {src} does not resolve to an object type")
            b = obj_bucket[o]
            if b is None:
                keep.append(it)
            else:
                drop[b] += 1
                mid = it.get("motion_id") or fn
                if mid not in held_seen[b]:
                    held_seen[b].add(mid)
                    held_pool[b].append(it)
        dest = outdir / name.replace(".json", "_retained.json")
        if dest.exists() and not args.force:
            raise SystemExit(f"REFUSED: {dest} exists; pass --force")
        dest.write_text(json.dumps(keep))
        report["inputs"][name] = len(items)
        report["outputs"][dest.name] = {"kept": len(keep), "dropped": dict(drop)}
        print(f"[eval-manifest] {name:28s} {len(items):6d} -> {len(keep):6d} retained "
              f"(held: {dict(drop)})")

    for b, items in sorted(held_pool.items()):
        dest = outdir / f"held_{b}.json"
        if dest.exists() and not args.force:
            raise SystemExit(f"REFUSED: {dest} exists; pass --force")
        dest.write_text(json.dumps(items))
        report["outputs"][dest.name] = {"n": len(items)}
        print(f"[eval-manifest] held_{b}.json{'':14s} {len(items):6d} unique held clips")

    # These manifests are test queries used only AFTER the evaluator and generator are frozen.
    (outdir / "README.txt").write_text(
        "Derived from " + str(args.holdout_artifact) + "\n"
        "sha256=" + art["artifact_sha256"] + "\n\n"
        "*_retained.json : retained canonical topologies only. train_main_retained is the ONLY\n"
        "                  evaluator training set; val_all_retained is the ONLY signal allowed to\n"
        "                  select its checkpoint.\n"
        "held_*.json     : TEST QUERIES, used only after the evaluator AND the generator are\n"
        "                  frozen. No checkpoint, threshold, architecture or normalisation may be\n"
        "                  tuned against them.\n")
    (outdir / "manifest_report.json").write_text(json.dumps(report, indent=2))
    print(f"[eval-manifest] -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
