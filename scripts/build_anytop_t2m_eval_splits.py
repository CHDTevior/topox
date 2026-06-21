#!/usr/bin/env python3
"""M0: build AnyTop T2M evaluator eval-split manifests (pure data, no torch).

VQ/CodeFlow revision (2026-06-16, handoff/20260614_anytop_t2m_evaluator_vq_codeflow_revision.md):
- Target the MERGED dataset data/animo4d_anytop_clean_L4_safe_plus_truebones (was L2).
- Variable per-motion caption count (1..7), NOT a fixed N_CAP=5.
- source_motion_id: truebones carry a real grouping id (e.g. Alligator_0000, shared across
  augmented clips); animo4d clips are 1:1 with their source (verified: 0 shared source_file),
  so source_motion_id := motion_id for them (correct, not degenerate).
- DistilBERT raw-caption is the PRIMARY text tower, so the T5 key cache is OPTIONAL here
  (t5_keys are still emitted for the T5 ablation path, validated only if --t5_keys given).
- Adds per-source manifests val_animo4d.json / val_truebones.json and `source` + `object_type`
  metadata on every record (metadata only: grouping / multi-positive mask / per-source reports).
- Blacklists the handful of hard-corrupt caption strings (single-letter A/B/C/D, NT-repeats);
  every motion retains >=1 valid caption (verified 0 all-corrupt motions on the merged set).
- val_action_clean / val_action_overlap counts are COMPUTED from data (no stale L2 assert);
  only train/val totals + disjointness are hard-asserted.

Run: scripts/build_anytop_t2m_eval_splits.py [--data_root ...] [--t5_keys ...]
"""
import argparse
import json
import re
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

ANIMAL_RE = re.compile(r"\b(an|the) animal\b", re.IGNORECASE)
# Hard-corrupt caption blacklist (spec §5): single-char placeholders + NT-repeat junk.
_CORRUPT_RE = re.compile(r"(?:NT){3,}|[NT]{6,}", re.IGNORECASE)


def fail(msg):
    raise SystemExit(f"[M0 PREFLIGHT FAIL] {msg}")


def read_split(p):
    return [l.strip() for l in p.read_text().splitlines()
            if l.strip() and not l.startswith("#")]


def is_corrupt_caption(s: str) -> bool:
    t = (s or "").strip()
    return len(t) <= 1 or bool(_CORRUPT_RE.fullmatch(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/animo4d_anytop_clean_L4_safe_plus_truebones")
    ap.add_argument("--cap_json", default=None,
                    help="caption JSON; default <data_root>/motion_texts_by_file_with_codex_drafts.json")
    ap.add_argument("--t5_keys", default=None,
                    help="optional T5 key-cache JSON; if given, validate emitted t5_keys exist "
                         "(T5 is fallback/ablation only — DistilBERT raw caption is primary)")
    ap.add_argument("--expect_train", type=int, default=71784)
    ap.add_argument("--expect_val", type=int, default=3808)
    args = ap.parse_args()

    data = ROOT / args.data_root
    splits = data / "splits"
    cap_json = pathlib.Path(args.cap_json) if args.cap_json else (
        data / "motion_texts_by_file_with_codex_drafts.json")
    out = data / "eval_splits"

    train = read_split(splits / "train.txt")
    val = read_split(splits / "val.txt")
    caps = json.loads(cap_json.read_text())

    t5_keys = None
    if args.t5_keys:
        t5_raw = json.loads(pathlib.Path(args.t5_keys).read_text())
        t5_keys = set(t5_raw.keys()) if isinstance(t5_raw, dict) else set(t5_raw)

    # ---- split integrity (hard-fail on count drift / dup / overlap) ----
    if len(train) != args.expect_train:
        fail(f"train {len(train)} != {args.expect_train}")
    if len(val) != args.expect_val:
        fail(f"val {len(val)} != {args.expect_val}")
    ts, vs = set(train), set(val)
    if len(ts) != len(train):
        fail("duplicate entries in train.txt")
    if len(vs) != len(val):
        fail("duplicate entries in val.txt")
    if ts & vs:
        fail(f"train/val overlap: {len(ts & vs)} clips")

    corrupt_seen = set()

    def build_record(fname):
        if fname not in caps:
            fail(f"no caption JSON entry for {fname}")
        c = caps[fname]
        stem = fname[:-4] if fname.endswith(".npy") else fname
        # variable caption count; drop hard-corrupt strings (every motion keeps >=1 valid).
        raw_caps = c.get("captions") or []
        capl = []
        for cap in raw_caps:
            if is_corrupt_caption(cap):
                corrupt_seen.add(cap.strip())
                continue
            capl.append(cap)
        if not capl:
            fail(f"{fname} has 0 valid captions after blacklist (raw={raw_caps})")
        # source_motion_id: real (truebones) or motion_id fallback (animo4d 1:1).
        smid = c.get("source_motion_id") or stem
        source = c.get("source_dataset") or "unknown"
        object_type = c.get("object_name")  # None for truebones; base dataset re-derives it
        t5k = [f"{stem}__cap{i}" for i in range(len(capl))]
        if t5_keys is not None:
            for k in t5k:
                if k not in t5_keys:
                    fail(f"missing T5 cache key {k} (--t5_keys given)")
        ss_idx = next((i for i, cap in enumerate(capl) if ANIMAL_RE.search(cap)), None)
        return {
            "filename": fname,
            "motion_id": stem,
            "source_motion_id": smid,            # grouping/mask/metadata only
            "source": source,                    # animo4d_L4_safe | truebones (metadata only)
            "object_type": object_type,          # species/object (metadata only; may be null)
            "captions": capl,                    # variable-length, blacklist-filtered
            "t5_keys": t5k,                       # for T5 ablation path only
            "species_stripped_cap_idx": ss_idx,
            "has_species_stripped": ss_idx is not None,
        }

    train_recs = [build_record(m) for m in train]
    val_recs = [build_record(m) for m in val]

    # ---- action-clean / action-overlap by source_motion_id (COMPUTED, no assert) ----
    train_smids = {r["source_motion_id"] for r in train_recs}
    val_clean = [r for r in val_recs if r["source_motion_id"] not in train_smids]
    val_overlap = [r for r in val_recs if r["source_motion_id"] in train_smids]

    # ---- per-source val subsets: human (HumanML3D) vs animal (everything else) ----
    # This merged set has no truebones; build_record sets source='HumanML3D' for human rows
    # and 'unknown' for animal rows. Split human vs animal so the eval can report R-precision
    # separately (human is the new addition we must validate the evaluator can discriminate).
    def is_human(r):
        return ("humanml3d" in str(r["source"]).lower()
                or str(r["motion_id"]).upper().startswith("HML3D"))
    val_human = [r for r in val_recs if is_human(r)]
    val_animal = [r for r in val_recs if not is_human(r)]

    ss_hits = sum(r["has_species_stripped"] for r in val_recs)
    ss_cov = ss_hits / len(val_recs)

    out.mkdir(exist_ok=True)

    def dump(name, recs):
        (out / f"{name}.json").write_text(json.dumps(recs))

    dump("train_main", train_recs)
    dump("val_all", val_recs)
    dump("val_action_clean", val_clean)
    dump("val_action_overlap", val_overlap)
    dump("val_human", val_human)
    dump("val_animal", val_animal)

    audit = {
        "data_root": args.data_root,
        "canonical_action_key": "source_motion_id (HumanML3D=real grouping, can repeat across train/val; animal=motion_id, 1:1)",
        "train_main": len(train_recs),
        "val_all": len(val_recs),
        "total": len(train_recs) + len(val_recs),
        "val_action_clean": len(val_clean),
        "val_action_overlap": len(val_overlap),
        "val_human": len(val_human),
        "val_animal": len(val_animal),
        "val_species_stripped_hits": ss_hits,
        "val_species_stripped_coverage": round(ss_cov, 4),
        "blacklisted_corrupt_captions": sorted(corrupt_seen),
        "labels": {
            "val_all": "held-out CLIP validation, NOT an unseen-action benchmark",
            "val_action_clean": "source_motion_id not in train (animo4d mostly clean by 1:1)",
            "species_stripped": "sanity-only view (spec §5); full caption is the primary view",
        },
    }
    (out / "split_audit.json").write_text(json.dumps(audit, indent=2))

    print(f"[M0 OK] train_main={len(train_recs)} val_all={len(val_recs)} "
          f"clean={len(val_clean)} overlap={len(val_overlap)} "
          f"human={len(val_human)} animal={len(val_animal)}")
    print(f"[M0 OK] blacklisted corrupt captions: {sorted(corrupt_seen)}")
    print(f"[M0 OK] val species_stripped coverage={ss_cov:.4f} ({ss_hits}/{len(val_recs)})")
    print(f"[M0 OK] wrote {out}/*.json")


if __name__ == "__main__":
    main()
