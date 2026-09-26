#!/usr/bin/env python3
"""Joint-description table for the UniML3D corpus, in the shape the KTJD semantic builder reads.

scripts/_build_joint_semantic_embeddings_ktjd17.py wants ONE dict keyed by joint NAME, with
{"description", "source", "confident"} per entry and no name missing. The corpus already carries a
description per (rig, joint) inside each skeleton npz (`joint_descriptions`, provisional anatomical
templates derived from the upstream clean names), so this script only collapses that per-rig table
into the global name-keyed one -- and reports, loudly, exactly what the collapse costs.

THE COLLAPSE IS LOSSY AND THE REPORT SAYS BY HOW MUCH. `joint_names` are per-rig unique raw IDs
(e.g. "mixamorig:RightUpLeg_00") whose numeric suffix is a per-rig index, so the SAME name string
can appear in many rigs, and in a small number of rigs the upstream clean name attached to it
disagrees -- including left/right flips. This script resolves each name by MAJORITY VOTE over the
joint rows that carry it, records every alternative in a `conflicts` field, and prints the number
of joint rows whose corpus description the winner does not match. Those extra fields are inert for
the semantic builder (it reads only `description` and `confident`) but make the artifact auditable,
and the printed count is the number to weigh when deciding whether the templates are worth redoing.

`confident` is False for a name whose description is contested across rigs or is a content-free
placeholder (LOW_INFO below: "Bone joint.", "Root joint." and their left/right/end variants -- the
upstream name carried no anatomy). It feeds only the semantic builder's per-rig confidence report.
"""
import hashlib, json, os, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get("CORPUS_ROOT", "dataset/ktjd17_uniml3d_v1"))
SKEL = Path(os.environ.get("SKELETONS", ROOT / "skeletons"))
OUT = Path(os.environ.get("OUT_JSON", "data/joint_descriptions_uniml3d_v1.json"))
TOP_N = int(os.environ.get("TOP_N", "20"))
if OUT.exists() and os.environ.get("FORCE") != "1":
    raise SystemExit(f"REFUSED: {OUT} exists; set FORCE=1 to overwrite")

# A placeholder stem carries no anatomy: the upstream clean name was a generic rig-editor label.
LOW_INFO = re.compile(
    r"^(?:(?:Left|Right|Upper|Lower|Front|Back|Hind|Middle)\s+)*"
    r"(?:Bone|Root|Object|Joint|Node|Armature|Dummy|Mesh|Null|Empty|Group|Unnamed|Unknown)"
    r"(?:\s+End)?$", re.I)

paths = sorted(SKEL.glob("*.npz"))
if not paths:
    raise SystemExit(f"REFUSED: no skeletons under {SKEL}")
name2desc: dict[str, Counter] = defaultdict(Counter)
per_rig: dict[str, tuple[list[str], list[str]]] = {}
for p in paths:
    z = np.load(p, allow_pickle=False)
    nm = [str(x) for x in z["joint_names"]]
    ds = [str(x) for x in z["joint_descriptions"]]
    if len(nm) != len(ds):
        raise SystemExit(f"REFUSED: {p} has {len(nm)} joint_names but {len(ds)} descriptions")
    if any(not d.strip() for d in ds):
        raise SystemExit(f"REFUSED: {p} has an empty joint description")
    per_rig[p.stem] = (nm, ds)
    for n, d in zip(nm, ds):
        name2desc[n][d] += 1

rows = sum(len(v[0]) for v in per_rig.values())
desc_rows = Counter(d for _, ds in per_rig.values() for d in ds)
# Majority vote, ties broken by the description string so the artifact is reproducible.
winner = {n: sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] for n, c in name2desc.items()}
conflicted = {n for n, c in name2desc.items() if len(c) > 1}

mis = lr = 0
mis_rigs: set[str] = set()
examples: list[tuple[str, str, str, str]] = []
for rig, (nm, ds) in per_rig.items():
    for n, d in zip(nm, ds):
        if winner[n] != d:
            mis += 1
            mis_rigs.add(rig)
            if ("left" in winner[n].lower()) != ("left" in d.lower()) or \
               ("right" in winner[n].lower()) != ("right" in d.lower()):
                lr += 1
                if len(examples) < 5:
                    examples.append((rig, n, d, winner[n]))

low_desc = {d for d in desc_rows if LOW_INFO.match(d[:-len(" joint.")] if d.endswith(" joint.") else d)}
low_rows = sum(desc_rows[d] for d in low_desc)
low_frac = np.array([sum(1 for d in ds if d in low_desc) / len(ds) for _, ds in per_rig.values()])

out = {}
for n, c in name2desc.items():
    w = winner[n]
    e = {"description": w,
         "source": "uniml3d_skeleton_joint_descriptions",
         "confident": bool(n not in conflicted and w not in low_desc),
         "rows": int(sum(c.values()))}
    if len(c) > 1:
        e["conflicts"] = {k: int(v) for k, v in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))}
    out[n] = e

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(out, indent=1, sort_keys=True))

print(f"rigs={len(per_rig):,}  joint rows={rows:,}  unique joint names={len(out):,}  "
      f"unique descriptions={len(desc_rows):,}")
print(f"confident names: {sum(1 for v in out.values() if v['confident']):,} of {len(out):,}")
print()
print(f"COLLAPSE COST: {mis:,} of {rows:,} joint rows ({mis/rows*100:.3f}%) get a description the "
      f"corpus does not give them, across {len(mis_rigs)} of {len(per_rig)} rigs; "
      f"{lr} of those are left/right flips.")
print(f"  contested names: {len(conflicted):,} of {len(out):,}")
for rig, n, had, got in examples:
    print(f"    {rig} {n!r}: corpus {had!r} -> table {got!r}")
print()
print(f"LOW-INFORMATION (placeholder) descriptions: {len(low_desc)} unique, {low_rows:,} joint "
      f"rows ({low_rows/rows*100:.2f}%)")
for d in sorted(low_desc, key=lambda d: -desc_rows[d]):
    print(f"    {desc_rows[d]:7,}  {d!r}")
print(f"  per-rig placeholder fraction: mean={low_frac.mean():.4f} median={np.median(low_frac):.4f}"
      f"  >50%: {int((low_frac > 0.5).sum())} rigs   100%: {int((low_frac >= 1.0).sum())} rigs")
dpr = np.array([len(set(ds)) for _, ds in per_rig.values()])
print(f"  distinct descriptions per rig: min={dpr.min()} median={int(np.median(dpr))} "
      f"max={dpr.max()}; rigs with only one: {int((dpr == 1).sum())}")
print()
print(f"TOP {TOP_N} descriptions by joint rows:")
for d, c in desc_rows.most_common(TOP_N):
    print(f"    {c:7,}  {d!r}")
print()
print(f"[OK] {OUT} ({OUT.stat().st_size/1e6:.2f} MB) sha256 "
      f"{hashlib.sha256(OUT.read_bytes()).hexdigest()[:16]}")
