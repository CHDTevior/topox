"""Same clip IDs on both sides: build the training sets for the UniMate-vs-ours controlled comparison
(user 2026-09-22: "让两边训同样的数据量，对应的 ID 也相同 ... 每边用自己的格式").

A clip is COMMON when it exists in our v2 corpus (manifest official_id) and in UniMate's feature set as exactly one
file <official_id>-000.npz whose object UniMate can train on (5 <= J <= 60, its loader's joint-count window). Our
visually excluded clips are dropped from the common set on both sides. The split is ours: common clips in our train
split are the shared TRAINING set, common clips in our val split the shared HELD-OUT set (neither side trains on it).

Writes (all read-only on the sources):
  <exclusions>   our loader's exclude_clips artifact: every v2 clip that is neither common-train nor common-held-out
  <unimate_dir>  UniMate feature dir: motions/ = symlinks to the common-TRAINING files only; cond.npy and captions.json
                 pruned to them; SUBSET.json with the lists and source hashes
  <split>        both lists (official ids and our clip ids)
Refuses when a common clip's caption or joint count differs between the corpora.

usage: python scripts/_build_common_unimate_subset.py
"""
from __future__ import annotations
import argparse, hashlib, json, os, re
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", type=Path, default=REPO / "dataset/ktjd17_uniml3d_v2")
    ap.add_argument("--visual_exclusions", type=Path, default=REPO / "configs/uniml3d_v2_visual_exclusions.json")
    ap.add_argument("--unimate_src", type=Path, default=REPO / "outside_docs/UniMate/dataset/features/objaverse")
    ap.add_argument("--unimate_dir", type=Path, default=REPO / "outside_docs/UniMate/dataset/features/objaverse_common_v2")
    ap.add_argument("--exclusions", type=Path, default=REPO / "configs/uniml3d_v2_common_unimate_exclusions.json")
    ap.add_argument("--split", type=Path, default=REPO / "runs/_unimate/common_v2_split.json")
    a = ap.parse_args()
    # refuse BEFORE any write: a re-run that rewrote our cut and then stopped here would leave the two sides on
    # different id sets with nothing binding them together (review 2026-09-22 item 3)
    if a.unimate_dir.exists():
        raise SystemExit(f"[refuse] {a.unimate_dir} exists; remove it deliberately before rebuilding")

    gen = json.loads((a.ours / "generation.json").read_text())["generation_id"]
    vis = json.loads(a.visual_exclusions.read_text())
    if vis["generation_id"] != gen:
        raise SystemExit(f"[refuse] visual exclusions bound to {vis['generation_id']}, corpus is {gen}")
    vis_clips = set(vis["clips"])
    man = [json.loads(l) for l in open(a.ours / "manifests/clips.jsonl")]
    by_oid = {}
    for r in man:
        if r["official_id"] in by_oid:
            raise SystemExit(f"[refuse] official_id {r['official_id']} appears twice in our manifest")
        by_oid[r["official_id"]] = r
    J_ours = {}
    for r in man:
        if r["rig_id"] not in J_ours:
            J_ours[r["rig_id"]] = len(np.load(a.ours / "skeletons" / f"{r['rig_id']}.npz")["parents"])

    cond = np.load(a.unimate_src / "cond.npy", allow_pickle=True).item()
    caps_u = json.loads((a.unimate_src / "captions.json").read_text())
    J_u = {k: len(v["parents"]) for k, v in cond.items()}
    slices = {}
    for f in os.listdir(a.unimate_src / "motions"):
        m = re.fullmatch(r"(.+)-(\d{3})\.npz", f)
        if m:
            slices.setdefault(m.group(1), []).append(f)
    usable = {b for b, fs in slices.items() if fs == [f"{b}-000.npz"] and b.split("-", 1)[0] in J_u
              and 5 <= J_u[b.split("-", 1)[0]] <= 60}

    common = sorted(o for o in by_oid if o in usable and by_oid[o]["clip_id"] not in vis_clips)
    bad = []
    for o in common:
        r = by_oid[o]; obj = o.split("-", 1)[0]
        if J_ours[r["rig_id"]] != J_u[obj]:
            bad.append(f"{o}: J ours {J_ours[r['rig_id']]} vs UniMate {J_u[obj]}")
        cu = caps_u.get(f"{o}-000")
        if cu is None or cu not in r["captions"]:
            bad.append(f"{o}: caption ours {r['captions']} vs UniMate {cu!r}")
    if bad:
        raise SystemExit("[refuse] common clips disagree between the corpora:\n  " + "\n  ".join(bad[:20]))
    train = [o for o in common if by_oid[o]["split"] == "train"]
    held = [o for o in common if by_oid[o]["split"] == "val"]
    if len(train) + len(held) != len(common):
        raise SystemExit("[refuse] a common clip has a split other than train/val")

    # ours: exclude everything that is not common (and keep the visual exclusions excluded)
    keep = {by_oid[o]["clip_id"] for o in common}
    reasons = {}
    for r in man:
        c = r["clip_id"]
        if c in keep:
            continue
        o, obj = r["official_id"], r["official_id"].split("-", 1)[0]
        if c in vis_clips:
            reasons[c] = "visual exclusion carried from configs/uniml3d_v2_visual_exclusions.json"
        elif obj not in J_u:
            reasons[c] = "rig not in UniMate's feature set"
        elif not (5 <= J_u[obj] <= 60):
            reasons[c] = f"UniMate cannot train this rig: J={J_u[obj]} outside its loader window [5, 60]"
        elif o not in slices:
            reasons[c] = "clip absent from UniMate's features (filtered by its preprocessing, e.g. low activity)"
        else:
            reasons[c] = f"UniMate stores this clip as {sorted(slices[o])}, not one -000 file"
    excl = {"generation_id": gen, "clips": {c: "all" for c in sorted(reasons)}, "reasons": reasons,
            "scope": ("Controlled UniMate-vs-ours comparison: our training is restricted to the clips UniMate also "
                      "trains on (same official ids). Not a quality judgement of the excluded clips."),
            "built_by": "scripts/_build_common_unimate_subset.py",
            "counts": {"v2_clips": len(man), "common": len(common), "common_train": len(train),
                       "common_heldout": len(held), "excluded": len(reasons)},
            "sources": {"manifest_sha256": sha(a.ours / "manifests/clips.jsonl"),
                        "visual_exclusions_sha256": sha(a.visual_exclusions),
                        "unimate_cond_sha256": sha(a.unimate_src / "cond.npy"),
                        "unimate_captions_sha256": sha(a.unimate_src / "captions.json")}}
    a.exclusions.write_text(json.dumps(excl, indent=1, ensure_ascii=False) + "\n")

    # UniMate: a feature dir holding only the common TRAINING clips
    (a.unimate_dir / "motions").mkdir(parents=True)
    src_motions = (a.unimate_src / "motions").resolve()
    for o in train:
        os.symlink(src_motions / f"{o}-000.npz", a.unimate_dir / "motions" / f"{o}-000.npz")
    objs = sorted({o.split("-", 1)[0] for o in train})
    np.save(a.unimate_dir / "cond.npy", {k: cond[k] for k in objs}, allow_pickle=True)
    (a.unimate_dir / "captions.json").write_text(json.dumps({f"{o}-000": caps_u[f"{o}-000"] for o in train}, indent=1))
    subset = {"train_official_ids": train, "heldout_official_ids": held, "objects": objs,
              "counts": {"train_clips": len(train), "objects": len(objs), "heldout_clips": len(held)},
              "sources": excl["sources"], "built_by": "scripts/_build_common_unimate_subset.py"}
    (a.unimate_dir / "SUBSET.json").write_text(json.dumps(subset, indent=1) + "\n")
    # the COMPARISON cohort: held-out clips whose rig has at least one training clip. UniMate cannot generate for an
    # object absent from its pruned cond.npy, and our pair loader skips a rig with no demo pool -- a generation script
    # that self-pairs the val set would keep all of them, so the cohort is pinned here (review 2026-09-22 item 6)
    objset = set(objs)
    comparable = [o for o in held if o.split("-", 1)[0] in objset]
    a.split.parent.mkdir(parents=True, exist_ok=True)
    a.split.write_text(json.dumps({"train_official_ids": train, "heldout_official_ids": held,
                                   "heldout_comparable_official_ids": comparable,
                                   "train_clip_ids": [by_oid[o]["clip_id"] for o in train],
                                   "heldout_clip_ids": [by_oid[o]["clip_id"] for o in held],
                                   "heldout_comparable_clip_ids": [by_oid[o]["clip_id"] for o in comparable],
                                   "note": "comparison cohort = heldout_comparable (rig has a training clip on both sides); "
                                           "generate our side from last_model.pt, never best_model.pt (val = held-out)"},
                                  indent=1) + "\n")
    held_rigs_trained = len(comparable)
    print(f"v2 clips {len(man)} | UniMate usable {len(usable)} | common {len(common)} "
          f"(train {len(train)}, held-out {len(held)}; held-out whose rig UniMate trains on: {held_rigs_trained})")
    print(f"ours excluded {len(reasons)} -> {a.exclusions}")
    from collections import Counter
    for k, v in Counter(v.split(":")[0] for v in reasons.values()).most_common():
        print(f"   {v:5d}  {k}")
    print(f"UniMate subset: {len(train)} motion symlinks, {len(objs)} objects -> {a.unimate_dir}")


if __name__ == "__main__":
    main()
