#!/usr/bin/env python3
"""Derive a TrueBones LoRA view whose split moves EVERY usable clip of the named rigs into train
(user 2026-09-03: "把龙的全部加进去训练"). Nothing else changes: motions, skeletons, stats, joint
semantics, captions and the clip-level exclusion artifacts are the source view's, reached through
symlinks / identical files; only the manifest `split` field (and the split lists / rig_table derived
from it) differ. The result is declared in derivation.json (schema 2, parent_view = the source view)
so Ktjd17Base verifies and pins it like any derived view.

A rig moved to all-train has n_val = 0: the trainer skips validation for it (no best_model.pt,
last_model.pt is the deliverable) and the launcher needs ALLOW_NO_VAL=1. Read-only w.r.t. the
source view.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
from pathlib import Path


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src_view", default="dataset/ktjd17_truebones_lora_v2_mainbody")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--all_train_rigs", default="", help="comma-separated rig ids whose val clips move to train (may be empty)")
    ap.add_argument("--declare_exclusion", default="",
                    help="comma-separated clip-level exclusion artifacts to DECLARE in derivation.json in addition to the "
                         "source view's (e.g. a multi-rig group cut written by scripts/_make_tb_group_exclusion.py); "
                         "Ktjd17Base refuses undeclared cuts")
    ap.add_argument("--batch", type=int, default=8, help="batch used for the steps_per_epoch column of rig_table")
    a = ap.parse_args()
    src, out = Path(a.src_view), Path(a.out_root)
    rigs = [r.strip() for r in a.all_train_rigs.split(",") if r.strip()]
    deriv_src = json.loads((src / "derivation.json").read_text())
    if sha256_file(src / "manifests" / "clips.jsonl") != str(deriv_src.get("derived_manifest_sha256")):
        raise SystemExit(f"[refuse] {src} manifest does not match its own derivation.json")
    gen = json.loads((src / "generation.json").read_text())
    if str(deriv_src.get("parent_generation_id")) != str(gen["generation_id"]):
        raise SystemExit("[refuse] source view's derivation names a different generation than its generation.json")
    if out.exists():
        raise SystemExit(f"[refuse] {out} exists; remove it explicitly before rebuilding")

    rows = [json.loads(l) for l in open(src / "manifests" / "clips.jsonl")]
    have = {str(r["rig_id"]) for r in rows if r.get("status") == "accept"}
    missing = [r for r in rigs if r not in have]
    if missing:
        raise SystemExit(f"[refuse] rigs not in the source view: {missing}")
    moved = []
    for r in rows:
        if r.get("status") == "accept" and str(r["rig_id"]) in rigs and r.get("split") == "val":
            r["split_before_override"] = "val"; r["split"] = "train"; moved.append(str(r["clip_id"]))
    if rigs and not moved:
        raise SystemExit(f"[refuse] no val clip to move for {rigs}")
    extra = [e.strip() for e in a.declare_exclusion.split(",") if e.strip()]
    for e in extra:
        if not Path(e).is_file():
            raise SystemExit(f"[refuse] exclusion artifact to declare does not exist: {e}")
        ej = json.loads(Path(e).read_text())
        if ej.get("mode") != "clip" or not isinstance(ej.get("clips"), dict):
            raise SystemExit(f"[refuse] {e} is not a clip-mode exclusion artifact")
    if not rigs and not extra:
        raise SystemExit("[refuse] nothing to derive: no --all_train_rigs and no --declare_exclusion")

    out.mkdir(parents=True); (out / "manifests").mkdir(); (out / "splits" / "lora_v1").mkdir(parents=True)
    for d in ("motions", "skeletons", "stats", "config", "qa", "evidence"):
        if (src / d).exists():
            (out / d).symlink_to(os.path.relpath((src / d).resolve(), out))
    for f in ("generation.json", "schema.json"):
        shutil.copyfile(src / f, out / f)
    with open(out / "manifests" / "clips.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    split_of = {str(r["clip_id"]): r["split"] for r in rows if r.get("status") == "accept"}
    for sp in ("train", "val"):
        (out / "splits" / "lora_v1" / f"{sp}.txt").write_text(
            "\n".join(sorted(c for c, s in split_of.items() if s == sp)) + "\n")
    table = json.loads((src / "splits" / "lora_v1" / "rig_table.json").read_text())
    for rig in rigs:
        t = table[rig]
        n_tr = sum(1 for r in rows if r.get("status") == "accept" and str(r["rig_id"]) == rig and r["split"] == "train")
        n_va = sum(1 for r in rows if r.get("status") == "accept" and str(r["rig_id"]) == rig and r["split"] == "val")
        t.update({"n_train": n_tr, "n_val": n_va, "steps_per_epoch_b%d" % a.batch: n_tr // a.batch,
                  "eligible": False, "reason": "no_val_group (all-train override; needs ALLOW_NO_VAL=1)",
                  "all_train_override": True})
    (out / "splits" / "lora_v1" / "rig_table.json").write_text(json.dumps(table, indent=1))

    excl = dict(deriv_src.get("exclusions") or {})
    for p, want in excl.items():
        if not Path(p).is_file() or sha256_file(p) != want:
            raise SystemExit(f"[refuse] exclusion artifact {p} missing or changed since the source view declared it")
    for e in extra:
        excl[str(e)] = sha256_file(e)
    for key in ("texts_json", "norm_stats", "joint_semantics"):
        ent = deriv_src.get(key) or {}
        if not ent or sha256_file(ent["path"]) != ent["sha256"]:
            raise SystemExit(f"[refuse] source view's {key} {ent.get('path')} missing or changed")
    n_tr = sum(1 for s in split_of.values() if s == "train"); n_va = sum(1 for s in split_of.values() if s == "val")
    deriv = {"schema_version": "2", "created_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "kind": "training_view_of_frozen_generation+joint_pruning" + ("+split_override" if moved else "") + ("+declared_exclusions" if extra else ""),
             "parent_root": deriv_src["parent_root"], "parent_generation_id": deriv_src["parent_generation_id"],
             "parent_generation_json_sha256": deriv_src["parent_generation_json_sha256"],
             "parent_manifest_sha256": deriv_src["parent_manifest_sha256"],
             "parent_view": {"root": str(src), "derivation_sha256": sha256_file(src / "derivation.json"),
                             "manifest_sha256": sha256_file(src / "manifests" / "clips.jsonl")},
             "derived_manifest_sha256": sha256_file(out / "manifests" / "clips.jsonl"),
             "split": {"name": "lora_v1",
                       "policy": f"source view's split with ALL usable clips of {rigs} in train (val -> train: {moved}); "
                                 f"every other rig unchanged",
                       "n_train": n_tr, "n_val": n_va, "n_unusable": deriv_src["split"].get("n_unusable"),
                       "rig_table_sha256": sha256_file(out / "splits" / "lora_v1" / "rig_table.json"),
                       "all_train_rigs": rigs, "moved_clips": moved},
             "joint_pruning": deriv_src.get("joint_pruning"),
             "texts_json": deriv_src["texts_json"], "norm_stats": deriv_src["norm_stats"],
             "joint_semantics": deriv_src["joint_semantics"], "exclusions": excl,
             "builder": {"path": "scripts/_derive_tb_view_split_override.py", "sha256": sha256_file(__file__)}}
    (out / "derivation.json").write_text(json.dumps(deriv, indent=1))
    print(f"[override] {out}: moved {len(moved)} clip(s) of {rigs} to train" +
          (" -> " + ", ".join(f"{r}: train {table[r]['n_train']} / val {table[r]['n_val']}" for r in rigs) if rigs else "") +
          (f"; declared {len(extra)} extra exclusion artifact(s) {extra}" if extra else "") +
          f"; derived manifest {deriv['derived_manifest_sha256'][:12]}")


if __name__ == "__main__":
    main()
