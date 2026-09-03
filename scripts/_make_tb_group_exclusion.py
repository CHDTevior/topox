#!/usr/bin/env python3
"""Clip-level exclusion artifact that keeps the usable clips of a GROUP of TrueBones rigs (user 2026-09-03:
a "flying group" LoRA -- Bat, Bird, Buzzard, Dragon, Eagle, Giantbee, Parrot, Pteranodon -- to give the
Dragon val clips a flight prior the per-species LoRA cannot learn from 10 clips). Everything else is
excluded: the unusable set (no caption / rest pose) and every clip of a rig outside the group. Same
loader contract as the per-rig artifacts written by scripts/_build_tb_lora_corpus.py. The view that
serves it must DECLARE it (scripts/_derive_tb_view_split_override.py --declare_exclusion)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--view", default="dataset/ktjd17_truebones_lora_v2_mainbody")
    ap.add_argument("--rigs", required=True, help="comma-separated rig ids to KEEP")
    ap.add_argument("--unusable", default="configs/tb_lora_v1_exclusions_unusable.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rigs = [r.strip() for r in a.rigs.split(",") if r.strip()]
    rows = [json.loads(l) for l in open(Path(a.view) / "manifests" / "clips.jsonl")]
    acc = [r for r in rows if r.get("status") == "accept"]
    have = {str(r["rig_id"]) for r in acc}
    missing = [r for r in rigs if r not in have]
    if missing:
        raise SystemExit(f"[refuse] rigs not in {a.view}: {missing}")
    unusable = set(json.load(open(a.unusable))["clips"])
    drop = {str(r["clip_id"]) for r in acc if str(r["rig_id"]) not in rigs} | unusable
    keep = [r for r in acc if str(r["rig_id"]) in rigs and str(r["clip_id"]) not in unusable]
    n_tr = sum(1 for r in keep if r["split"] == "train"); n_va = sum(1 for r in keep if r["split"] == "val")
    per = {rig: {"train": sum(1 for r in keep if str(r["rig_id"]) == rig and r["split"] == "train"),
                 "val": sum(1 for r in keep if str(r["rig_id"]) == rig and r["split"] == "val")} for rig in rigs}
    Path(a.out).write_text(json.dumps({
        "mode": "clip", "n_clips": len(drop),
        "note": f"group LoRA cut: keep the usable clips of {rigs} (train {n_tr} / val {n_va}: {per}); "
                f"everything else excluded (other rigs + unusable set {Path(a.unusable).name})",
        "group_rigs": rigs, "group_counts": {"train": n_tr, "val": n_va, "per_rig": per},
        "clips": {c: "all" for c in sorted(drop)}}, indent=1))
    print(f"[group] {a.out}: keep {len(keep)} clips of {len(rigs)} rigs (train {n_tr} / val {n_va}), exclude {len(drop)}; per rig {per}")


if __name__ == "__main__":
    main()
