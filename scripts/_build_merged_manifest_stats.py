#!/usr/bin/env python3
"""Manifest + per-cell stats + schema for the merged no-IK corpus.

Splits: the OLD corpus already has an authoritative split for every Human clip; reuse it verbatim
so the human half stays comparable across corpora. New PZ clips are split per-rig so every rig
appears in train (the model is per-rig conditioned; a rig seen only in val teaches nothing).
"""
import json, hashlib, os, sys
from pathlib import Path
import numpy as np

NEW = Path(os.environ.get("NEW_ROOT", "data/animo4d_anytop_noik_canonical/data"))
OLD = Path("dataset/ktjd17_pz_human312")
OUT = Path(os.environ.get("OUT_ROOT", "dataset/ktjd17_pzh312_noik_v2"))
VAL_FRAC = 0.05

# ---- manifest ---------------------------------------------------------------
old_rows = {}
for l in open(OLD / "manifests" / "clips.jsonl"):
    r = json.loads(l)
    if r["rig_id"] == "HML3D_Human":
        old_rows[r["clip_id"]] = r
print(f"[man] human rows carried over: {len(old_rows)}")

new_rows = [json.loads(l) for l in open(NEW / "manifests" / "clips.jsonl")]
by_rig = {}
for r in new_rows:
    by_rig.setdefault(r["rig_id"], []).append(r)

rng = np.random.default_rng(0)
out, n_val = [], 0
for rig, rows in sorted(by_rig.items()):
    rows = sorted(rows, key=lambda r: r["clip_id"])
    idx = rng.permutation(len(rows))
    k = max(1, int(round(len(rows) * VAL_FRAC))) if len(rows) > 1 else 0
    val = set(idx[:k].tolist())
    for j, r in enumerate(rows):
        split = "val" if j in val else "train"
        n_val += split == "val"
        caption = r.get("caption") or {}
        texts = caption.get("texts") or []
        out.append({
            "clip_id": r["clip_id"], "rig_id": rig,
            "motion_relpath": f"motions/{r['clip_id']}.npz",
            "skeleton_relpath": f"skeletons/{rig}.npz",
            "split": split, "status": "accept",
            "J_phys": None, "T_target": r.get("stored_frames"),
            "fps_target": 30.0,
            "official_id": r.get("official_id"),
            "source_action_name": r.get("source_action_name"),
            "captions": texts,
            "topology_family": "planetzoo",
            "topology_distance_bucket": "train_seen_topology",
        })
for cid, r in sorted(old_rows.items()):
    out.append({
        "clip_id": cid, "rig_id": "HML3D_Human",
        "motion_relpath": f"motions/{cid}.npz",
        "skeleton_relpath": "skeletons/HML3D_Human.npz",
        "split": r["split"], "status": r["status"],
        "J_phys": r.get("J_phys"), "T_target": r.get("T_target"),
        "fps_target": 30.0,
        "official_id": None, "source_action_name": None, "captions": [],
        "topology_family": "human",
        "topology_distance_bucket": r.get("topology_distance_bucket"),
    })
(OUT / "manifests").mkdir(parents=True, exist_ok=True)
with open(OUT / "manifests" / "clips.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
from collections import Counter
c = Counter(r["split"] for r in out)
fam = Counter(r["topology_family"] for r in out)
print(f"[man] {len(out)} rows  splits={dict(c)}  families={dict(fam)}")
assert len({r['clip_id'] for r in out}) == len(out), "duplicate clip_id"

# ---- per-cell stats ---------------------------------------------------------
# The new corpus ships per-rig stats, but they were computed on the ABSOLUTE root track we just
# re-based, so ch13:15 would be wrong. Recompute from the converted files -- and for the human rig
# reuse nothing, so both halves come from one code path.
print("[stats] recomputing from the converted corpus (the shipped stats predate the re-basing)")
