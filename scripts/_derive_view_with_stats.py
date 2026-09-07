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
    dst.mkdir()
    rel = os.path.relpath(src.resolve(), dst.resolve())
    for entry in sorted(src.iterdir()):
        if entry.name == "derivation.json":
            continue
        os.symlink(os.path.join(rel, entry.name), dst / entry.name)
    new = dict(deriv)
    new["norm_stats"] = {"path": str(stats), "sha256": sha256_file(stats)}
    new["stats_override"] = {"kind": "zero-shot statistics ablation view (scripts/_derive_view_with_stats.py)",
                             "source_view": str(src), "source_derivation_sha256": sha256_file(src / "derivation.json"),
                             "original_norm_stats": deriv["norm_stats"],
                             "note": "identical manifest / splits / skeletons / motions / cuts; only the served per-cell "
                                     "statistics differ. Render and evaluate only -- never train on this view."}
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
