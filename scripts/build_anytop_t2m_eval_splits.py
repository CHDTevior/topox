#!/usr/bin/env python3
"""M0: build AnyTop T2M evaluator eval-split manifests (pure data, no torch).

Reads the materialized splits/{train,val}.txt + caption JSON + T5 key cache,
derives val_action_clean / val_action_overlap by canonical_action_key
(=source_motion_id), and HARD-FAILS on any drift. The manifests drive the
independent text-motion evaluator (train_main) and its reporting subsets.

`val_all` is held-out CLIP validation, NOT an unseen-action benchmark
(source_motion_id is species+gender+action 组合级 holdout). See
handoff/20260604_0121_anytop_t2m_evaluator_impl_proposal.md §1c / §5-M0.

Run: /iridisfs/scratch/ts1v23/.conda/bin/python3.12 scripts/build_anytop_t2m_eval_splits.py
"""
import json
import re
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data/anytop_planet_zoo_clean_L2"
SPLITS = DATA / "splits"
CAP_JSON = DATA / "motion_texts_by_file_with_codex_drafts.json"
T5_KEYS = ROOT / "data/anytop_caption_t5_cleanL2_multi.keys.json"
OUT = DATA / "eval_splits"

N_CAP = 5
ANIMAL_RE = re.compile(r"\b(an|the) animal\b", re.IGNORECASE)
EXPECT = dict(train=77882, val=4112, total=81994, clean=824, overlap=3288)


def fail(msg):
    raise SystemExit(f"[M0 PREFLIGHT FAIL] {msg}")


def read_split(p):
    return [l.strip() for l in p.read_text().splitlines()
            if l.strip() and not l.startswith("#")]


# ---- load ----
train = read_split(SPLITS / "train.txt")
val = read_split(SPLITS / "val.txt")
caps = json.loads(CAP_JSON.read_text())
t5_raw = json.loads(T5_KEYS.read_text())
t5_keys = set(t5_raw.keys()) if isinstance(t5_raw, dict) else set(t5_raw)

# ---- split integrity (hard-fail) ----
if len(train) != EXPECT["train"]:
    fail(f"train {len(train)} != {EXPECT['train']}")
if len(val) != EXPECT["val"]:
    fail(f"val {len(val)} != {EXPECT['val']}")
ts, vs = set(train), set(val)
if len(ts) != len(train):
    fail("duplicate entries in train.txt")
if len(vs) != len(val):
    fail("duplicate entries in val.txt")
if ts & vs:
    fail(f"train/val overlap: {len(ts & vs)} clips")
if len(ts | vs) != EXPECT["total"]:
    fail(f"train+val cover {len(ts | vs)} != {EXPECT['total']} motions")


def build_record(fname):
    """fname = .npy basename (= split / caption JSON key)."""
    if fname not in caps:
        fail(f"no caption JSON entry for {fname}")
    c = caps[fname]
    smid = c.get("source_motion_id")
    if not smid:
        fail(f"no source_motion_id for {fname}")
    capl = c.get("captions") or []
    if len(capl) < N_CAP:
        fail(f"{fname} has {len(capl)} captions < {N_CAP}")
    capl = capl[:N_CAP]
    stem = fname[:-4] if fname.endswith(".npy") else fname
    t5k = [f"{stem}__cap{i}" for i in range(N_CAP)]
    for k in t5k:
        if k not in t5_keys:
            fail(f"missing T5 cache key {k}")
    # species_stripped (方案 A): first caption matching regex \b(an|the) animal\b
    ss_idx = next((i for i, cap in enumerate(capl) if ANIMAL_RE.search(cap)), None)
    return {
        "filename": fname,                       # with .npy (split / caption key)
        "motion_id": stem,                       # stem (AnyTopDataset motion_id)
        "source_motion_id": smid,                # canonical_action_key (metadata/group/mask only)
        "captions": capl,                        # cap0..cap4
        "t5_keys": t5k,                          # {motion_id}__cap{0..4}
        "species_stripped_cap_idx": ss_idx,      # animal-level cap index, or null
        "has_species_stripped": ss_idx is not None,
    }


train_recs = [build_record(m) for m in train]
val_recs = [build_record(m) for m in val]

# ---- derive action-clean / action-overlap by canonical key ----
train_smids = {r["source_motion_id"] for r in train_recs}
val_clean = [r for r in val_recs if r["source_motion_id"] not in train_smids]
val_overlap = [r for r in val_recs if r["source_motion_id"] in train_smids]
if len(val_clean) != EXPECT["clean"]:
    fail(f"val_action_clean {len(val_clean)} != {EXPECT['clean']}")
if len(val_overlap) != EXPECT["overlap"]:
    fail(f"val_action_overlap {len(val_overlap)} != {EXPECT['overlap']}")
if len(val_clean) + len(val_overlap) != EXPECT["val"]:
    fail("val_action_clean + val_action_overlap != val_all")

ss_hits = sum(r["has_species_stripped"] for r in val_recs)
ss_cov = ss_hits / len(val_recs)

# ---- write manifests ----
OUT.mkdir(exist_ok=True)


def dump(name, recs):
    (OUT / f"{name}.json").write_text(json.dumps(recs))  # compact


dump("train_main", train_recs)
dump("val_all", val_recs)
dump("val_action_clean", val_clean)
dump("val_action_overlap", val_overlap)

audit = {
    "canonical_action_key": "source_motion_id",
    "train_main": len(train_recs),
    "val_all": len(val_recs),
    "total": len(train_recs) + len(val_recs),
    "val_action_clean": len(val_clean),
    "val_action_overlap": len(val_overlap),
    "val_species_stripped_hits": ss_hits,
    "val_species_stripped_coverage": round(ss_cov, 4),
    "labels": {
        "val_all": "held-out CLIP validation, NOT an unseen-action benchmark",
        "val_action_clean": "species+gender+action combo holdout (NOT action-template unseen)",
    },
}
(OUT / "split_audit.json").write_text(json.dumps(audit, indent=2))

print(f"[M0 OK] train_main={len(train_recs)} val_all={len(val_recs)} "
      f"clean={len(val_clean)} overlap={len(val_overlap)}")
print(f"[M0 OK] val species_stripped coverage={ss_cov:.4f} ({ss_hits}/{len(val_recs)})")
print(f"[M0 OK] wrote {OUT}/*.json")
