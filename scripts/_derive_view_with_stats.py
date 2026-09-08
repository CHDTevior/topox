"""Sibling of a derived KTJD-17 training view that serves a DIFFERENT per-cell stats file (zero-shot item 1, 2026-09-07).

A derived view (derivation.json) pins the sha256 of the stats file it is served with, and Ktjd17Base refuses any other
file -- correct for training, but the zero-shot statistics ablation must render the SAME rigs / SAME validation targets
with borrowed statistics. This tool creates <dst> next to <src>: every entry of <src> is symlinked (relative), except
derivation.json, which is copied with `norm_stats` re-pointed at the new file and a `stats_override` record. Nothing in
<src> is touched; the new view is a scratch artifact for rendering / evaluation, never for training.

usage: python scripts/_derive_view_with_stats.py --src dataset/ktjd17_truebones_lora_v2_mainbody \
          --dst dataset/ktjd17_truebones_lora_v2_mainbody_bstats --stats data/tb_norm_stats_borrowed_v1.npz
"""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--stats", required=True, help="the stats npz this view will be served with")
    ap.add_argument("--purpose", choices=("eval", "train"), default="eval",
                    help="eval (default): a scratch view for rendering / evaluation only. train: a TRAINING view -- allowed only "
                         "for a stats artifact whose cohort excludes the evaluation clips (support-only protocol, 2026-09-08)")
    a = ap.parse_args()
    src, dst, stats = Path(a.src), Path(a.dst), Path(a.stats)
    if not (src / "derivation.json").is_file():
        raise SystemExit(f"[refuse] {src} is not a derived view (no derivation.json)")
    if dst.exists():
        raise SystemExit(f"[refuse] {dst} exists -- remove it yourself if you mean to rebuild")
    if not stats.is_file():
        raise SystemExit(f"[refuse] stats file {stats} missing")
    if src.resolve() == dst.resolve() or dst.resolve().parent != src.resolve().parent:
        raise SystemExit("[refuse] dst must be a sibling directory of src")
    deriv = json.loads((src / "derivation.json").read_text())
    if not isinstance(deriv.get("norm_stats"), dict) or "sha256" not in deriv["norm_stats"]:
        raise SystemExit("[refuse] src derivation.json has no norm_stats pin to replace")
    # the stats file must belong to the same frozen generation, else the loader refuses it anyway; check here to fail early
    import numpy as np
    with np.load(stats, allow_pickle=False) as z:
        meta = json.loads(str(z["__meta"]))
    if str(meta.get("generation_id")) != str(deriv.get("parent_generation_id")):
        raise SystemExit(f"[refuse] stats generation {meta.get('generation_id')} != view parent generation "
                         f"{deriv.get('parent_generation_id')}")
    if a.purpose == "train":
        # the cohort NAME is a claim; bind it to THIS view's manifest and THIS view's support set, or a support-only artifact
        # measured on another split of the same generation would pass while containing these validation clips (codex r1 #2)
        if str(meta.get("cohort")) != "per_rig_train_clips_only":
            raise SystemExit(f"[refuse] --purpose train needs support-only statistics (cohort per_rig_train_clips_only); "
                             f"{stats} declares {meta.get('cohort')!r}, which contains the evaluation clips")
        man = src / "manifests" / "clips.jsonl"
        if str(meta.get("manifest_sha256")) != sha256_file(man):
            raise SystemExit(f"[refuse] {stats} was measured on manifest {str(meta.get('manifest_sha256'))[:12]}, this view serves "
                             f"{sha256_file(man)[:12]}")
        train_ids = sorted({str(json.loads(l)["clip_id"]) for l in open(man)
                            if json.loads(l).get("status") == "accept" and str(json.loads(l).get("split")) == "train"})
        want = hashlib.sha256("\n".join(train_ids).encode()).hexdigest()
        if str(meta.get("train_ids_sha256")) != want:
            raise SystemExit(f"[refuse] {stats} declares support set {str(meta.get('train_ids_sha256'))[:12]}, this view's training "
                             f"clips hash {want[:12]} -- the statistics were measured on another split")
    dst.mkdir()
    rel = os.path.relpath(src.resolve(), dst.resolve())
    for entry in sorted(src.iterdir()):
        if entry.name == "derivation.json":
            continue
        os.symlink(os.path.join(rel, entry.name), dst / entry.name)
    new = dict(deriv)
    new["norm_stats"] = {"path": str(stats), "sha256": sha256_file(stats)}
    _KIND = {"eval": "zero-shot statistics ablation view (scripts/_derive_view_with_stats.py)",
             "train": "support-only TRAINING view: identical clips, splits and skeletons; the served per-cell statistics are "
                      "measured on each rig's training clips only (scripts/_derive_view_with_stats.py --purpose train)"}
    _NOTE = {"eval": "identical manifest / splits / skeletons / motions / cuts; only the served per-cell "
                     "statistics differ. Render and evaluate only -- never train on this view.",
             "train": "identical manifest / splits / skeletons / motions / cuts; only the served per-cell statistics differ. "
                      "Train, render and evaluate: the statistics exclude the evaluation clips, so numbers measured on this "
                      "view are held-out."}
    new["stats_override"] = {"kind": _KIND[a.purpose],
                             "source_view": str(src), "source_derivation_sha256": sha256_file(src / "derivation.json"),
                             "original_norm_stats": deriv["norm_stats"],
                             # an eval-purpose derivation stays byte-identical to what this tool wrote before the training
                             # purpose existed (codex r1 #6): the extra records appear only for a training view
                             **({"purpose": "train", "stats_cohort": str(meta.get("cohort")),
                                 "support_train_ids_sha256": str(meta.get("train_ids_sha256"))} if a.purpose == "train" else {}),
                             "note": _NOTE[a.purpose]}
    (dst / "derivation.json").write_text(json.dumps(new, indent=1))
    # self-check: the pins the loader verifies must resolve through the symlinks
    for what, p in (("generation.json", dst / "generation.json"), ("manifest", dst / "manifests" / "clips.jsonl"),
                    ("rig_table", dst / "splits" / "lora_v1" / "rig_table.json")):
        if not p.is_file():
            raise SystemExit(f"[refuse] {what} not reachable in the new view ({p})")
    if sha256_file(dst / "manifests" / "clips.jsonl") != str(new.get("derived_manifest_sha256")):
        raise SystemExit("[refuse] manifest sha does not match the derivation pin (source view inconsistent?)")
    print(f"[OK] {dst}: {len(list(dst.iterdir()))} entries, norm_stats -> {stats} ({new['norm_stats']['sha256'][:16]})")


if __name__ == "__main__":
    main()
