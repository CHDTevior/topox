"""Materialise the four clip lists implied by the frozen held-out-topology artifact.

Membership is resolved from `cond.npy` by longest-prefix object resolution and then by AHU
canonical form. It is deliberately NOT resolved from manifest metadata: TrueBones records
carry `object_type: None`, so filtering on that field would silently leak every creature rig.

Produces, under `<out_dir>/`:

  train.txt                   original train minus every held canonical topology
  val.txt                     original val   minus every held canonical topology
                              — this alone controls early stopping and checkpoint selection
  held_representative.txt     ALL clips (train+val) of the representative topologies
  held_stress.txt             ALL clips (train+val) of the stress topologies
  splits_manifest.json        counts, hashes, and the artifact SHA256 they derive from

The retained lists are named `train.txt`/`val.txt` deliberately: any tool pointed at this
directory reads RETAINED data by default and cannot pick up a held clip by accident. The
held lists carry distinct names so reading one is always an explicit act.

A held topology contributes its entire inventory to the test side, which is why the two
held lists draw from both original partitions.

The held lists are TEST QUERIES ONLY, used after the evaluator and the generator are frozen.
Nothing — checkpoint, threshold, architecture, normalisation — may be tuned against them.
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
from src.data.anytop_dataset import _longest_prefix_match  # noqa: E402
from scripts._build_holdout_trees import canonical_form, _file_sha256  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout_artifact", default="data/holdout_topologies_v1.json")
    ap.add_argument("--data_root", default="data/animo4d_L4TB_plus_human_v4b272neutral")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    art_path = Path(args.holdout_artifact)
    art = json.loads(art_path.read_text())

    # Re-verify the freeze before deriving anything from it.
    body = json.dumps({k: v for k, v in art.items() if k != "artifact_sha256"},
                      indent=2, sort_keys=True)
    got = hashlib.sha256(body.encode()).hexdigest()
    if got != art.get("artifact_sha256"):
        raise SystemExit(f"REFUSED: {art_path} is tampered — body hashes to {got}, "
                         f"artifact carries {art.get('artifact_sha256')}")

    root = Path(args.data_root)
    if _file_sha256(root / "cond.npy") != art["inputs"]["cond_npy_sha256"]:
        raise SystemExit("REFUSED: cond.npy has changed since the freeze")
    for f, k in (("train.txt", "train_split_sha256"), ("val.txt", "val_split_sha256")):
        if _file_sha256(root / "splits" / f) != art["inputs"][k]:
            raise SystemExit(f"REFUSED: splits/{f} has changed since the freeze")

    cond = np.load(root / "cond.npy", allow_pickle=True).item()
    keys = sorted(cond.keys(), key=len, reverse=True)

    def canon_of(obj):
        v = cond[obj]
        p = v["parents"] if isinstance(v, dict) else v[0]
        return hashlib.sha256(
            canonical_form(tuple(int(x) for x in np.asarray(p).ravel())).encode()).hexdigest()

    bucket_of_canon = {}
    for t in art["held_out_trees"]:
        bucket_of_canon[t["canonical_form_sha256"]] = t["bucket"]
    held_canon = set(bucket_of_canon)

    # Object-type -> (canonical form, bucket or None). Built for EVERY object type in the
    # corpus, so an object type that shares a held canonical form is caught even if it was
    # not itself listed in the artifact.
    obj_bucket = {}
    for o in cond:
        c = canon_of(o)
        obj_bucket[o] = bucket_of_canon.get(c)
    listed = {o for t in art["held_out_trees"] for o in t["object_types"]}
    caught = {o for o, b in obj_bucket.items() if b is not None}
    if caught != listed:
        raise SystemExit(f"REFUSED: canonical-form membership ({len(caught)} object types) "
                         f"disagrees with the artifact's explicit list ({len(listed)}); "
                         f"difference: {sorted(caught ^ listed)[:5]}")

    def read_split(n):
        return [l.strip() for l in (root / "splits" / n).read_text().splitlines()
                if l.strip() and not l.startswith("#")]

    out = defaultdict(list)
    per_bucket_obj = defaultdict(set)
    for part in ("train", "val"):
        for fn in read_split(f"{part}.txt"):
            o = _longest_prefix_match(fn, keys)
            if o is None:
                raise SystemExit(f"REFUSED: unresolved filename {fn!r}")
            b = obj_bucket[o]
            if b is None:
                out[part].append(fn)
            else:
                out[f"held_{b}"].append(fn)
                per_bucket_obj[b].add(o)

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = {"holdout_artifact": str(art_path), "artifact_sha256": art["artifact_sha256"],
                "data_root": str(root.resolve()), "counts": {}, "files": {}}
    header = ("# derived from " + str(art_path) + " sha256=" + art["artifact_sha256"][:16]
              + " — do not hand-edit")
    for name in ("train", "val", "held_representative", "held_stress"):
        lines = sorted(out[name])
        if len(set(lines)) != len(lines):
            raise SystemExit(f"REFUSED: {name} has duplicates")
        p = outdir / f"{name}.txt"
        if p.exists() and not args.force:
            raise SystemExit(f"REFUSED: {p} exists; pass --force to regenerate")
        p.write_text(header + "\n" + "\n".join(lines) + "\n")
        manifest["counts"][name] = len(lines)
        manifest["files"][name] = _file_sha256(p)

    tot_in = len(read_split("train.txt")) + len(read_split("val.txt"))
    tot_out = sum(manifest["counts"].values())
    if tot_in != tot_out:
        raise SystemExit(f"REFUSED: {tot_in} input clips but {tot_out} written — clips lost")
    overlap = set(out["train"] + out["val"]) & \
              set(out["held_representative"] + out["held_stress"])
    if overlap:
        raise SystemExit(f"REFUSED: {len(overlap)} clips in both retained and held sets")

    manifest["held_object_types"] = {b: sorted(v) for b, v in per_bucket_obj.items()}
    (outdir / "splits_manifest.json").write_text(json.dumps(manifest, indent=2))

    for k, v in manifest["counts"].items():
        print(f"[splits] {k:22s} {v:6d} clips")
    print(f"[splits] total {tot_out} == input {tot_in}  OK")
    print(f"[splits] -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
