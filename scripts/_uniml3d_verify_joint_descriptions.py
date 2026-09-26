#!/usr/bin/env python3
"""Verify the LEFT/RIGHT side of the refilled UniML3D joint descriptions.

The side criterion under test: the corpus rest pose is root-centred in x and z, the schema frame is
right-handed with +Y up and rest forward +Z, so left = up x forward = +X and the sagittal plane is
x = 0; a joint is called left / right when x / s_rig is above / below +-SIDE_TOL and mid-line
otherwise.  Four independent checks, none of which assumes the criterion it is testing:

 1  CONVENTION   over every rig that carries upstream Left/Right anatomy: does sign(x) agree with
                 the corpus's own side word?  (the criterion is calibrated, not asserted)
 2  OVERWRITTEN  534 of the rewritten rows had a side word in the placeholder they replaced
                 ("Left Bone joint.").  Ground truth for exactly those rows: does the new sentence
                 keep that side?
 3  MIRROR PAIRS 20 sampled rewritten rigs with bilateral structure: for each mirror pair
                 (x ~ -x', same y, z, depth, chain length) the +x member must read "left" and the
                 -x member "right".
 4  ANCESTOR     a rewritten joint whose nearest named ancestor is a "Left ..." / "Right ..." part
                 must not contradict it.
 5  ANCHOR       every anchor clause is re-derived from the skeleton: the hop count must be the real
                 number of bones to that ancestor, the quoted name must be that ancestor's own
                 description, any above/below word must agree with the MEASURED height difference,
                 and a side word quoted inside the anchor must not contradict the ancestor's own x.
                 Reads the sentences out of the changelog's `new` field, so the same instrument
                 measures any generation of the wording (codex r1 P1-1: the superseded generation
                 wrote "below" for all 978 anchored rows, 295 of which sit above their ancestor).
"""
import argparse, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

SIDE_TOL = 0.02
PLACE_TOL = 0.02
ANCHOR_MAX_HOPS = 4

LOW_INFO = re.compile(
    r"^(?:(?:Left|Right|Upper|Lower|Front|Back|Hind|Middle)\s+)*"
    r"(?:Bone|Root|Object|Joint|Node|Armature|Dummy|Mesh|Null|Empty|Group|Unnamed|Unknown)"
    r"(?:\s+End)?$", re.I)

# both wordings: the superseded "<hops> below the <name>" and the current
# "<hops> down the chain from the <name>[ and above|below it]"
ANCHOR_CLAUSE = re.compile(
    r"\b(one bone|\w+ bones) (down the chain from|below|above) the "
    r"(.+?)(?: and (above|below) it)?(?:, |$)")
NUMWORD = {"one": 1, "two": 2, "three": 3, "four": 4}


def stem_of(d):
    return (d[:-len(" joint.")] if d.endswith(" joint.") else d).strip().lower()


def is_low_info(d):
    return bool(LOW_INFO.match(d[:-len(" joint.")] if d.endswith(" joint.") else d))


def side_word(s: str):
    if re.search(r"\bon the left side\b", s) or s.startswith("Left "):
        return "left"
    if re.search(r"\bon the right side\b", s) or s.startswith("Right "):
        return "right"
    if re.search(r"\bon the mid-line\b", s):
        return "mid"
    return None


def geo_side(xr):
    return "left" if xr > SIDE_TOL else ("right" if xr < -SIDE_TOL else "mid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skeletons", default="dataset/ktjd17_uniml3d_v1/skeletons")
    ap.add_argument("--changelog", default="scratch/uniml3d_sidecars/joint_description_fill_v2.json")
    ap.add_argument("--n_rigs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    SK = Path(a.skeletons)
    log = json.loads(Path(a.changelog).read_text())["rigs"]

    # ---------------- 1. convention, on the rigs that carry upstream anatomy ----------------
    c1 = Counter()
    for p in sorted(SK.glob("*.npz")):
        z = np.load(p, allow_pickle=False)
        ds = [str(v) for v in z["joint_descriptions"]]
        P = np.asarray(z["P_rest_global"], np.float64); s = float(z["s_rig"])
        for j, d in enumerate(ds):
            w = "left" if d.startswith("Left ") else ("right" if d.startswith("Right ") else None)
            if w is None:
                continue
            g = geo_side(P[j, 0] / s)
            c1["agree" if g == w else ("mid-line (side dropped)" if g == "mid" else "CONTRADICTS")] += 1
    n1 = sum(c1.values())
    print(f"[1] upstream-anatomy joints with a side word: {n1:,}")
    for k, v in c1.most_common():
        print(f"      {v:8,}  {v/n1*100:6.3f}%  {k}")

    # ---------------- 2. rows whose placeholder already carried a side ----------------------
    c2, ex2 = Counter(), []
    for rig, rows in log.items():
        for r in rows:
            w = side_word(r["old"])
            if w not in ("left", "right"):
                continue
            g = side_word(r["new"])
            c2["agree" if g == w else (f"mid-line (side dropped)" if g == "mid" else "CONTRADICTS")] += 1
            if g not in (w, "mid") and len(ex2) < 10:
                ex2.append((rig, r["j"], r["old"], r["new"]))
    n2 = sum(c2.values())
    print(f"\n[2] rewritten rows whose PLACEHOLDER carried a side word: {n2:,}")
    for k, v in c2.most_common():
        print(f"      {v:8,}  {v/n2*100:6.3f}%  {k}")
    for e in ex2:
        print("      contradiction:", e)

    # ---------------- 3. mirror pairs on sampled rewritten rigs -----------------------------
    rng = np.random.default_rng(a.seed)
    cand = []
    for rig, rows in log.items():
        z = np.load(SK / f"{rig}.npz", allow_pickle=False)
        P = np.asarray(z["P_rest_global"], np.float64); s = float(z["s_rig"])
        idx = [r["j"] for r in rows]
        xr = P[:, 0] / s
        if sum(1 for j in idx if xr[j] > SIDE_TOL) >= 3 and sum(1 for j in idx if xr[j] < -SIDE_TOL) >= 3:
            cand.append(rig)
    pick = [cand[i] for i in rng.permutation(len(cand))[:a.n_rigs]]
    print(f"\n[3] rigs with >=3 rewritten joints on each side: {len(cand)}; checking {len(pick)}")
    tot_pairs = bad = 0
    for rig in pick:
        z = np.load(SK / f"{rig}.npz", allow_pickle=False)
        P = np.asarray(z["P_rest_global"], np.float64); s = float(z["s_rig"])
        par = np.asarray(z["parents"], np.int64)
        ds = [str(v) for v in z["joint_descriptions"]]
        J = len(ds)
        dep = np.zeros(J, int)
        for j in range(1, J):
            dep[j] = dep[int(par[j])] + 1
        idx = [r["j"] for r in log[rig]]
        key = defaultdict(list)
        for j in idx:
            key[(int(dep[j]), round(abs(P[j, 0]) / s, 3), round(P[j, 1] / s, 3),
                 round(P[j, 2] / s, 3))].append(j)
        npair = nbad = 0
        for k, js in key.items():
            if len(js) != 2 or k[1] <= SIDE_TOL:
                continue
            l = [j for j in js if P[j, 0] > 0]
            r = [j for j in js if P[j, 0] < 0]
            if len(l) != 1 or len(r) != 1:
                continue
            npair += 1
            if not (side_word(ds[l[0]]) == "left" and side_word(ds[r[0]]) == "right"):
                nbad += 1
                print(f"      MISMATCH {rig} j={l[0]}/{r[0]}: {ds[l[0]]!r} / {ds[r[0]]!r}")
        tot_pairs += npair; bad += nbad
        print(f"      {rig[:26]:26} J={J:3d} mirror pairs={npair:3d} wrong={nbad}")
    print(f"[3] total mirror pairs {tot_pairs}, wrong {bad}")

    # ---------------- 4. named ancestor must not be contradicted ----------------------------
    c4, ex4 = Counter(), []
    for rig, rows in log.items():
        z = np.load(SK / f"{rig}.npz", allow_pickle=False)
        ds = [str(v) for v in z["joint_descriptions"]]
        par = np.asarray(z["parents"], np.int64)
        rewritten = {r["j"] for r in rows}
        old_of = {r["j"]: r["old"] for r in rows}
        for j in sorted(rewritten):
            aj, hops = int(par[j]), 1
            while aj >= 0 and aj in rewritten:
                aj, hops = int(par[aj]), hops + 1
            if aj < 0 or hops > 4:
                continue
            w = "left" if ds[aj].startswith("Left ") else ("right" if ds[aj].startswith("Right ") else None)
            if w is None:
                continue
            g = side_word(ds[j])
            c4["agree" if g == w else ("mid-line" if g == "mid" else "differs")] += 1
            if g not in (w, "mid") and len(ex4) < 8:
                ex4.append((rig, j, ds[aj], ds[j], old_of[j]))
    n4 = sum(c4.values())
    print(f"\n[4] rewritten joints within 4 bones of a Left/Right-named ancestor: {n4:,}")
    for k, v in c4.most_common():
        print(f"      {v:8,}  {v/max(n4,1)*100:6.3f}%  {k}")
    for e in ex4:
        print("      differs:", e)

    # ---------------- 5. anchor clauses re-derived from the skeleton ------------------------
    c5, ex5 = Counter(), []
    n_anchored = n_sentences = 0
    for rig, rows in log.items():
        z = np.load(SK / f"{rig}.npz", allow_pickle=False)
        cur = [str(v) for v in z["joint_descriptions"]]
        par = np.asarray(z["parents"], np.int64)
        P_ = np.asarray(z["P_rest_global"], np.float64)
        s_ = float(z["s_rig"])
        # the description vector the generator SAW: the changelog's `old` where it rewrote, the
        # file's own row everywhere else (those rows are never touched)
        old_of = {r["j"]: r["old"] for r in rows}
        src = [old_of.get(j, cur[j]) for j in range(len(cur))]
        low = [is_low_info(d) for d in src]
        for r in rows:
            j, new = r["j"], r["new"]
            n_sentences += 1
            m = ANCHOR_CLAUSE.search(new)
            if not m:
                continue
            n_anchored += 1
            hops_txt, joiner, name, tail = m.group(1), m.group(2), m.group(3), m.group(4)
            said = tail if joiner == "down the chain from" else joiner
            w0 = hops_txt.split()[0]
            hops_said = NUMWORD.get(w0, None) or (int(w0) if w0.isdigit() else None)
            # re-derive the ancestor the generator must have used
            a, hops = int(par[j]), 1
            while a >= 0 and low[a] and hops <= ANCHOR_MAX_HOPS:
                a, hops = int(par[a]), hops + 1
            if not (a >= 0 and not low[a] and hops <= ANCHOR_MAX_HOPS):
                c5["ANCHOR DOES NOT RESOLVE"] += 1
                continue
            if hops_said != hops:
                c5["HOP COUNT WRONG"] += 1
                if len(ex5) < 6:
                    ex5.append(("hops", rig, j, hops_said, hops, new))
                continue
            stem = stem_of(src[a])
            toks = stem.split()
            xa = P_[a, 0] / s_
            g_anc = "left" if xa > SIDE_TOL else ("right" if xa < -SIDE_TOL else "mid")
            dropped = toks and toks[0] in ("left", "right") and g_anc not in (toks[0], "mid")
            if name not in (stem, " ".join(toks[1:])):
                c5["QUOTED NAME IS NOT THE ANCESTOR'S"] += 1
                if len(ex5) < 6:
                    ex5.append(("name", rig, j, name, stem, new))
                continue
            # (a) the side word the clause actually quotes must not contradict the ancestor's x
            nw = name.split()[0] if name else ""
            if nw in ("left", "right"):
                if g_anc == "mid":
                    c5["side word on a mid-line ancestor (deadband, allowed)"] += 1
                elif g_anc != nw:
                    c5["SIDE WORD CONTRADICTS THE ANCESTOR'S GEOMETRY"] += 1
                    if len(ex5) < 6:
                        ex5.append(("side", rig, j, nw, round(float(xa), 4), new))
                else:
                    c5["side word agrees with the ancestor's geometry"] += 1
            elif dropped:
                c5["contradicted side word correctly omitted"] += 1
            # (b) any above/below word must be the measured one
            dy = (P_[j, 1] - P_[a, 1]) / s_
            truth = "above" if dy > PLACE_TOL else ("below" if dy < -PLACE_TOL else None)
            if said == truth:
                c5["height word agrees with the measured height" if said
                   else "no height word, and the height is within the deadband"] += 1
            elif said is None:
                c5["HEIGHT MEASURABLE BUT UNSTATED (safe)"] += 1
            elif truth is None:
                # a height word where the measurement cannot support one (inside the deadband)
                c5["HEIGHT WORD ASSERTED INSIDE THE DEADBAND"] += 1
                if len(ex5) < 6:
                    ex5.append(("deadband", rig, j, said, round(float(dy), 4), new))
            else:
                # the reversal codex counted: said below, measured above (or the mirror)
                c5["HEIGHT WORD CONTRADICTS THE MEASURED HEIGHT"] += 1
                if len(ex5) < 6:
                    ex5.append(("height", rig, j, said, round(float(dy), 4), new))
    print(f"\n[5] rewritten sentences: {n_sentences:,}; carrying an anchor clause: {n_anchored:,}")
    for k, v in c5.most_common():
        print(f"      {v:8,}  {k}")
    for e in ex5:
        print("      example:", e)
    FAIL5 = ("ANCHOR DOES NOT RESOLVE", "HOP COUNT WRONG", "QUOTED NAME IS NOT THE ANCESTOR'S",
             "SIDE WORD CONTRADICTS THE ANCESTOR'S GEOMETRY",
             "HEIGHT WORD CONTRADICTS THE MEASURED HEIGHT",
             "HEIGHT WORD ASSERTED INSIDE THE DEADBAND")
    print(f"[5] anchor clauses contradicted by the skeleton: {sum(c5[k] for k in FAIL5)} "
          + "  ".join(f"({k}: {c5[k]})" for k in FAIL5))
    return 0


if __name__ == "__main__":
    sys.exit(main())
