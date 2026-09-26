#!/usr/bin/env python3
"""Audit the per-(rig,joint) descriptions of a KTJD-17 corpus, and calibrate the LEFT/RIGHT axis.

Independent re-measurement (it re-reads the skeleton npz files; it trusts no earlier log):
  A  placeholder ("low information") descriptions: unique strings, joint rows, per-rig fractions
  B  what a global NAME-keyed majority-vote table would cost: rows it re-describes, rigs, L/R flips
  C  the lateral axis, calibrated against the rigs that DO carry Left/Right anatomy:
     rest x = P_rest_global[:,0] is already root-centred in x/z by the corpus builder, so its sign
     is the side.  The sign->word mapping is MEASURED here, not assumed.
  D  shape of the problem rigs: joints, depth, branch chains, leaves

`--descriptions_json <file>` overrides the npz field with a per-rig {rig: [desc, ...]} map, so the
same audit runs before and after a fill.
"""
import argparse, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# A placeholder stem carries no anatomy: the upstream clean name was a generic rig-editor label.
LOW_INFO = re.compile(
    r"^(?:(?:Left|Right|Upper|Lower|Front|Back|Hind|Middle)\s+)*"
    r"(?:Bone|Root|Object|Joint|Node|Armature|Dummy|Mesh|Null|Empty|Group|Unnamed|Unknown)"
    r"(?:\s+End)?$", re.I)


def is_low_info(d: str) -> bool:
    s = d[:-len(" joint.")] if d.endswith(" joint.") else d
    return bool(LOW_INFO.match(s))


def load_rigs(skel_dir: Path, override: dict | None):
    rigs = {}
    for p in sorted(skel_dir.glob("*.npz")):
        z = np.load(p, allow_pickle=False)
        nm = [str(x) for x in z["joint_names"]]
        ds = [str(x) for x in z["joint_descriptions"]]
        if override is not None and p.stem in override:
            ds = [str(x) for x in override[p.stem]]
            if len(ds) != len(nm):
                raise SystemExit(f"REFUSED: override for {p.stem} has {len(ds)} != {len(nm)} entries")
        rigs[p.stem] = dict(names=nm, desc=ds,
                            parents=np.asarray(z["parents"], np.int64),
                            P=np.asarray(z["P_rest_global"], np.float64),
                            s_rig=float(z["s_rig"]))
    return rigs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skeletons", default="dataset/ktjd17_uniml3d_v1/skeletons")
    ap.add_argument("--meta", default="dataset/ktjd17_uniml3d_v1/source_metadata")
    ap.add_argument("--descriptions_json", default=None)
    ap.add_argument("--tag", default="BEFORE")
    ap.add_argument("--dump_problem_rigs", default=None)
    a = ap.parse_args()

    override = json.loads(Path(a.descriptions_json).read_text()) if a.descriptions_json else None
    if override is not None and "rigs" in override and isinstance(override["rigs"], dict):
        override = override["rigs"]
    rigs = load_rigs(Path(a.skeletons), override)
    rows = sum(len(r["names"]) for r in rigs.values())
    print(f"##### {a.tag} #####")
    print(f"rigs={len(rigs):,}  joint rows={rows:,}  "
          f"unique joint NAMES={len({n for r in rigs.values() for n in r['names']}):,}  "
          f"unique DESCRIPTIONS={len({d for r in rigs.values() for d in r['desc']}):,}")

    # ---------------- A. placeholders ----------------
    desc_rows = Counter(d for r in rigs.values() for d in r["desc"])
    low = {d for d in desc_rows if is_low_info(d)}
    low_rows = sum(desc_rows[d] for d in low)
    frac = np.array([sum(1 for d in r["desc"] if d in low) / len(r["desc"]) for r in rigs.values()])
    dpr = np.array([len(set(r["desc"])) for r in rigs.values()])
    print(f"[A] placeholder descriptions: {len(low)} unique, {low_rows:,} joint rows "
          f"({low_rows / rows * 100:.3f}% of all rows)")
    for d in sorted(low, key=lambda d: -desc_rows[d])[:15]:
        print(f"      {desc_rows[d]:7,}  {d!r}")
    print(f"[A] rigs with >=1 placeholder: {int((frac > 0).sum())}   >50%: {int((frac > 0.5).sum())}"
          f"   ==100%: {int((frac >= 1.0).sum())}   mean per-rig frac={frac.mean():.4f}"
          f"  median={np.median(frac):.4f}")
    print(f"[A] distinct descriptions per rig: min={dpr.min()} p25={int(np.percentile(dpr,25))} "
          f"median={int(np.median(dpr))} max={dpr.max()};  rigs with only ONE: {int((dpr==1).sum())}"
          f"   with <=2: {int((dpr<=2).sum())}")

    # ---------------- B. cost of a global name-keyed table ----------------
    name2desc = defaultdict(Counter)
    for r in rigs.values():
        for n, d in zip(r["names"], r["desc"]):
            name2desc[n][d] += 1
    winner = {n: sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] for n, c in name2desc.items()}
    contested = {n for n, c in name2desc.items() if len(c) > 1}
    mis = lr = 0
    mis_rigs, lr_rigs = set(), set()
    for rig, r in rigs.items():
        for n, d in zip(r["names"], r["desc"]):
            if winner[n] != d:
                mis += 1
                mis_rigs.add(rig)
                if (("left" in winner[n].lower()) != ("left" in d.lower())) or \
                   (("right" in winner[n].lower()) != ("right" in d.lower())):
                    lr += 1
                    lr_rigs.add(rig)
    print(f"[B] global name-keyed majority-vote table would re-describe {mis:,} of {rows:,} rows "
          f"({mis / rows * 100:.3f}%) across {len(mis_rigs)} rigs; {lr} of those rows are L/R flips, "
          f"in {len(lr_rigs)} rigs.  contested names: {len(contested):,} of {len(name2desc):,}")

    # ---------------- C. lateral-axis calibration ----------------
    # rest x is already root-centred (builder: origin = [P[0,0], min y, P[0,2]]), so sign(x) is the side.
    xs = {"Left": [], "Right": []}
    per_rig_agree = []
    for rig, r in rigs.items():
        P, s = r["P"], r["s_rig"]
        lo, ro = [], []
        for j, d in enumerate(r["desc"]):
            if d.startswith("Left "):
                lo.append(P[j, 0] / s)
            elif d.startswith("Right "):
                ro.append(P[j, 0] / s)
        xs["Left"] += lo
        xs["Right"] += ro
        if lo and ro:
            per_rig_agree.append((rig, float(np.mean(lo)), float(np.mean(ro))))
    for k in ("Left", "Right"):
        v = np.asarray(xs[k])
        if v.size:
            print(f"[C] joints described {k!r:8}: n={v.size:,}  mean x/s_rig={v.mean():+.4f}  "
                  f"median={np.median(v):+.4f}  frac x>0={float((v>0).mean()):.4f}  "
                  f"frac |x|<1e-4={float((np.abs(v)<1e-4).mean()):.4f}")
    if per_rig_agree:
        ok = sum(1 for _, l, rr in per_rig_agree if l > rr)
        print(f"[C] rigs with both sides labelled: {len(per_rig_agree)};  mean(x of Left) > "
              f"mean(x of Right) in {ok} ({ok/len(per_rig_agree)*100:.2f}%)  "
              f"-> LEFT is the {'+X' if ok*2>len(per_rig_agree) else '-X'} side")

    # ---------------- D. shape of the problem rigs ----------------
    prob = [rig for rig, r in rigs.items() if any(d in low for d in r["desc"])]
    allplace = [rig for rig, r in rigs.items() if all(d in low for d in r["desc"])]
    J = np.array([len(rigs[r]["names"]) for r in prob])
    if J.size:
        print(f"[D] rigs with any placeholder: {len(prob)} (all-placeholder: {len(allplace)}); "
              f"their J: min={J.min()} median={int(np.median(J))} max={J.max()} sum={int(J.sum()):,}")
        deg, dep, leaf = [], [], []
        for rig in prob:
            par = rigs[rig]["parents"]
            ch = Counter(int(p) for p in par if p >= 0)
            deg += [ch.get(j, 0) for j in range(len(par))]
            leaf.append(sum(1 for j in range(len(par)) if ch.get(j, 0) == 0))
            d = np.zeros(len(par), int)
            for j in range(1, len(par)):
                d[j] = d[par[j]] + 1
            dep.append(int(d.max()))
        cd = Counter(deg)
        print(f"[D] child-count histogram over those rigs: "
              f"{ {k: cd[k] for k in sorted(cd)} }")
        print(f"[D] max depth per rig: min={min(dep)} median={int(np.median(dep))} max={max(dep)};"
              f"  leaves per rig: median={int(np.median(leaf))} max={max(leaf)}")
    if a.dump_problem_rigs:
        Path(a.dump_problem_rigs).write_text(json.dumps(sorted(prob), indent=0))
        print(f"[D] problem-rig list -> {a.dump_problem_rigs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
