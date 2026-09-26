#!/usr/bin/env python3
"""Replace the content-free joint descriptions of the UniML3D corpus with structural ones.

WHAT IS BROKEN.  Every description in dataset/ktjd17_uniml3d_v1 is the template "<clean name>
joint.", where the clean name comes from the upstream annotation
dataset/uniml3d/export/objaverse/clean_joint_names.json.  For 402 rigs that annotation is itself
content-free ("Bone", "Root"): 4,167 joint rows carry no identity at all, 97 rigs are 100%
placeholder and 77 rigs have a single description for the whole body.  The upstream file is the
SOURCE of the placeholders, so it cannot repair them, and the raw joint names of those rigs are
"joint12_13" / "Bone.002_Armature" -- a body-part word search over them is 11% recall and most of
those hits are false ("Armature" contains "arm").  The only honest signal left is the skeleton.

WHAT THIS WRITES.  For a placeholder joint only (every other row is left byte-identical), a
natural-language sentence built from quantities read off the rest pose and the parent array:

  side          sign of the rest x coordinate.  The corpus centres the rest pose on the root in x
                and z (builder: origin = [P[0,0], min y, P[0,2]]), so x = 0 IS the sagittal plane.
                LEFT = +X: calibrated on the 178,301 joints that DO carry Left/Right anatomy --
                99.75% of "Left ..." joints sit at x>0 and 99.65% of "Right ..." at x<0, and
                mean-x(Left) > mean-x(Right) in 5,156 of the 5,158 rigs that label both sides.
                It also matches the frame the schema declares (right-handed, +Y up, forward +Z,
                so left = up x forward = +X).  |x| <= SIDE_TOL * s_rig is called mid-line: at
                SIDE_TOL=0.02 that mis-files 1.48% of genuinely lateral joints as central and
                correctly calls 97.9% of the non-lateral ones central.
  limb          the maximal unbranched chain the joint lies on: its length, the joint's position
                along it, and whether it ends in a tip or in a branch.
  limb index    among sibling chains that leave the same parent on the same side, ranked front to
                back by rest z, so an insect's five left legs do not all read alike.
  placement     above / below / level with the root in y, in front of / behind it in z.
  anchor        if a non-placeholder ancestor is within 4 bones, the sentence hangs off it
                ("two bones down the chain from the left hand") instead of off the root.  The
                ancestry is stated as a HOP COUNT only; above/below is appended only when the
                measured height difference clears PLACE_TOL, and a leading left/right word whose
                own x refutes it is dropped from the quoted name (codex r1 P1-1: the old wording
                said "below" for all 978 anchored rows, 295 of which sit above their ancestor,
                and copied a "right front shoulder" sitting at x/s_rig=+0.1048).

Nothing anatomical is invented: no sentence names a body part that the corpus does not already
carry for that joint.  A rig whose body plan is "uncertain" is described as "a skeleton".

OUTPUT.  --apply rewrites `joint_descriptions` inside the touched skeleton npz files.  The 11
preserved fields are taken from a SNAPSHOT OF THE DESTINATION, never from a backup (codex r1 P1-2:
a stale backup silently reverted s_rig 28.2336 -> 14.1168 while the script reported eleven fields
bit-identical); an already-existing backup is instead validated as this skeleton's own history,
those eleven fields byte for byte.  The new file is written to a temporary sibling, verified there,
and only then os.replace()d into place, so an interrupted run leaves the destination readable and
unchanged (codex r1 P2-3).  The landed file is re-read and the run fails loudly unless the other 11
fields are bit-identical and joint_order_sha256 is unchanged.  A changelog of every (rig, joint,
old, new) is always written.  Without --apply nothing is touched (dry run).

--descriptions_from <dir> reads the description vector to be REWRITTEN from a directory of earlier
skeleton npz files (this script's own backups) instead of from the live corpus, so a corrected
generator can be re-run after an earlier one already replaced the placeholders.  Only
`joint_descriptions` is read from there, and only for rigs that directory actually holds; every
other field still comes from the live file.
"""
import argparse, hashlib, json, os, re, shutil, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SIDE_TOL = 0.02          # |x| / s_rig below this is the mid-line (calibrated, see docstring)
PLACE_TOL = 0.02         # same deadband for the above/below and front/behind clauses
ANCHOR_MAX_HOPS = 4      # beyond this an inherited part name stops being informative

LOW_INFO = re.compile(
    r"^(?:(?:Left|Right|Upper|Lower|Front|Back|Hind|Middle)\s+)*"
    r"(?:Bone|Root|Object|Joint|Node|Armature|Dummy|Mesh|Null|Empty|Group|Unnamed|Unknown)"
    r"(?:\s+End)?$", re.I)

PLAN_WORD = {"bipedal": "bipedal", "quadrupedal": "quadrupedal", "insectoid": "insectoid",
             "avian": "avian", "marine": "marine", "serpentine": "serpentine",
             "articulated_rigid": "rigid articulated", "uncertain": None}

NUM = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
       "eleven", "twelve"]
ORD = ["zeroth", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
       "ninth", "tenth", "eleventh", "twelfth"]


def num(n: int) -> str:
    return NUM[n] if n < len(NUM) else str(n)


def ordn(n: int) -> str:
    return ORD[n] if n < len(ORD) else f"{n}th"


def is_low_info(d: str) -> bool:
    s = d[:-len(" joint.")] if d.endswith(" joint.") else d
    return bool(LOW_INFO.match(s))


def plan_phrase(plan: str | None) -> str:
    w = PLAN_WORD.get(plan or "", None)
    if w is None:
        return "a skeleton"
    return ("an " if w[0] in "aeiou" else "a ") + w + " skeleton"


def describe_rig(names, desc, parents, P, s_rig, plan):
    """Return the rig's description list with every placeholder row replaced."""
    J = len(names)
    low = [is_low_info(d) for d in desc]
    children = defaultdict(list)
    for j in range(J):
        p = int(parents[j])
        if p >= 0:
            children[p].append(j)
    depth = np.zeros(J, np.int64)
    for j in range(1, J):
        depth[j] = depth[int(parents[j])] + 1

    x = P[:, 0] / s_rig
    y = (P[:, 1] - P[0, 1]) / s_rig          # root-relative height
    z = P[:, 2] / s_rig                      # already root-centred in z

    def side_of(j):
        if x[j] > SIDE_TOL:
            return "left"
        if x[j] < -SIDE_TOL:
            return "right"
        return "mid-line"

    # ---- maximal unbranched chains -------------------------------------------------------
    heads = [j for j in range(1, J) if int(parents[j]) == 0 or len(children[int(parents[j])]) >= 2]
    chain_of, chain_pos, chain_len, chain_end = {}, {}, {}, {}
    for h in heads:
        cur, seq = h, [h]
        while len(children[cur]) == 1:
            cur = children[cur][0]
            seq.append(cur)
        for i, j in enumerate(seq):
            chain_of[j] = h
            chain_pos[j] = i + 1
            chain_len[j] = len(seq)
            chain_end[j] = seq[-1]

    # ---- sibling limb ranking: same parent, same side, front (max z) first ---------------
    limb_rank, limb_group = {}, {}
    for p, ch in children.items():
        hs = [c for c in ch if c in chain_of and chain_of[c] == c]
        by_side = defaultdict(list)
        for c in hs:
            by_side[side_of(c)].append(c)
        for sd, group in by_side.items():
            group.sort(key=lambda c: (-z[c], -y[c], c))
            for i, c in enumerate(group):
                limb_rank[c], limb_group[c] = i + 1, len(group)

    # ---- nearest non-placeholder ancestor -------------------------------------------------
    stat = {"anchored": 0, "side_dropped": 0, "anchor_dropped": 0, "above": 0, "below": 0,
            "level": 0}

    def anchor(j):
        """(name, hops, ancestor index) of the nearest named ancestor, or (None, 0, -1).

        The name loses a LEADING left/right word that the ancestor's own rest x refutes: the anchor
        is quoted verbatim out of the corpus, and an inherited side word the geometry contradicts
        would put a claim in the sentence that nothing supports.  A side word on a mid-line ancestor
        is left alone -- the deadband means "not measurable here", not "wrong", and SIDE_TOL
        mis-files 1.48% of genuinely lateral joints as central.
        """
        hops, a = 1, int(parents[j])
        while a >= 0 and low[a] and hops <= ANCHOR_MAX_HOPS:
            a, hops = int(parents[a]), hops + 1
        if not (a >= 0 and not low[a] and hops <= ANCHOR_MAX_HOPS):
            return None, 0, -1
        stem = desc[a][:-len(" joint.")] if desc[a].endswith(" joint.") else desc[a]
        toks = stem.strip().lower().split()
        if toks and toks[0] in ("left", "right") and side_of(a) not in (toks[0], "mid-line"):
            stat["side_dropped"] += 1
            toks = toks[1:]
        if not toks:                       # the name was NOTHING but a refuted side word
            stat["anchor_dropped"] += 1
            return None, 0, -1
        return " ".join(toks), hops, a

    def place_clause(j):
        v = []
        if y[j] > PLACE_TOL:
            v.append("above")
        elif y[j] < -PLACE_TOL:
            v.append("below")
        h = None
        if z[j] > PLACE_TOL:
            h = "in front of"
        elif z[j] < -PLACE_TOL:
            h = "behind"
        if v and h:
            return f"{v[0]} and {h} the root"
        if v:
            return f"{v[0]} the root"
        if h:
            return f"{h} the root"
        return "level with the root"

    out = list(desc)
    for j in range(J):
        if not low[j]:
            continue
        pl = plan_phrase(plan)
        if j == 0:
            out[j] = f"the root joint at the base of {pl}"
            continue
        h = chain_of[j]
        L, pos, end = chain_len[j], chain_pos[j], chain_end[j]
        nkids = len(children[end])
        # "limb" ends in a tip; "branching chain" runs on into further limbs.
        kind = "limb" if nkids == 0 else "branching chain"
        if L == 1:
            role = "a single-bone limb" if nkids == 0 else "a single joint"
        elif pos == L and nkids == 0:
            role = f"the tip of a {num(L)}-bone limb"
        else:
            role = f"the {ordn(pos)} joint of a {num(L)}-bone {kind}"
        sd = side_of(j)
        parts = [f"{role} " + ("on the mid-line of " if sd == "mid-line"
                               else f"on the {sd} side of ") + pl]
        if limb_group.get(h, 1) > 1:
            parts.append(f"{ordn(limb_rank[h])} limb from the front")
        a_name, a_hops, a_idx = anchor(j)
        if a_name:
            # ancestry = hops along the tree; vertical wording only where the height says so
            hop = "one bone" if a_hops == 1 else f"{num(a_hops)} bones"
            dy = y[j] - y[a_idx]                       # measured, in rig scales
            rel = (" and above it" if dy > PLACE_TOL else
                   " and below it" if dy < -PLACE_TOL else "")
            stat["above" if dy > PLACE_TOL else "below" if dy < -PLACE_TOL else "level"] += 1
            stat["anchored"] += 1
            parts.append(f"{hop} down the chain from the {a_name}{rel}")
        else:
            parts.append(place_clause(j))
        if pos == L and nkids >= 2:
            parts.append(f"where {num(nkids)} limbs branch")
        out[j] = ", ".join(parts)
    return out, low, stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skeletons", default="dataset/ktjd17_uniml3d_v1/skeletons")
    ap.add_argument("--meta", default="dataset/ktjd17_uniml3d_v1/source_metadata")
    ap.add_argument("--changelog", default="scratch/uniml3d_sidecars/joint_description_fill_v2.json")
    ap.add_argument("--backup", default="scratch/uniml3d_sidecars/superseded_20260916/skeletons_orig")
    ap.add_argument("--descriptions_from", default=None,
                    help="directory of earlier skeleton npz whose joint_descriptions are the vector "
                         "to rewrite (this script's own backups).  Lets a corrected generator re-run "
                         "after an earlier one already replaced the placeholders.  Every field other "
                         "than joint_descriptions still comes from the live corpus file.")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    skel = Path(a.skeletons)
    paths = sorted(skel.glob("*.npz"))
    if not paths:
        raise SystemExit(f"REFUSED: no skeletons under {skel}")
    KEEP = ("joint_names", "parents", "P_rest_global", "R_rest_global", "R_rest_local",
            "offset_parent_local", "rotation_source_kind", "contact_joint_indices",
            "face_joint_names", "joint_order_sha256", "s_rig")

    src_dir = Path(a.descriptions_from) if a.descriptions_from else None
    changes, n_rows, n_low, rigs_touched = {}, 0, 0, 0
    new_desc_all, dup_in_rig, plans = {}, 0, Counter()
    tally, n_from_src = Counter(), 0
    for p in paths:
        z = np.load(p, allow_pickle=False)
        names = [str(v) for v in z["joint_names"]]
        desc = [str(v) for v in z["joint_descriptions"]]
        if src_dir is not None and (src_dir / p.name).is_file():
            with np.load(src_dir / p.name, allow_pickle=False) as sz:
                if [str(v) for v in sz["joint_names"]] != names:
                    raise SystemExit(f"REFUSED: {src_dir / p.name} is a different rig's joint order")
                desc = [str(v) for v in sz["joint_descriptions"]]
            n_from_src += 1
        n_rows += len(names)
        meta = json.loads((Path(a.meta) / f"{p.stem}.json").read_text())
        plan = meta.get("body_plan")
        new, low, st = describe_rig(names, desc, np.asarray(z["parents"], np.int64),
                                    np.asarray(z["P_rest_global"], np.float64), float(z["s_rig"]), plan)
        if not any(low):
            continue
        tally.update(st)
        rigs_touched += 1
        plans[plan] += 1
        n_low += sum(low)
        idx = [j for j in range(len(names)) if low[j]]
        seen = Counter(new[j] for j in idx)
        dup_in_rig += sum(c - 1 for c in seen.values() if c > 1)
        changes[p.stem] = [{"j": j, "name": names[j], "old": desc[j], "new": new[j]} for j in idx]
        new_desc_all[p.stem] = new

    print(f"rigs={len(paths):,}  joint rows={n_rows:,}")
    if src_dir is not None:
        print(f"description vector read from {src_dir} for {n_from_src} of {len(paths)} rigs "
              f"(the rest keep the corpus's own)")
    print(f"anchored sentences: {tally['anchored']:,}  (measured above the anchor "
          f"{tally['above']:,} / below {tally['below']:,} / level {tally['level']:,}); "
          f"anchor side words dropped as contradicted: {tally['side_dropped']:,}; "
          f"anchors dropped entirely: {tally['anchor_dropped']:,}")
    print(f"rigs with placeholders: {rigs_touched}  placeholder rows rewritten: {n_low:,} "
          f"({n_low/n_rows*100:.3f}%)  body plans: {dict(plans)}")
    uniq_new = {c['new'] for v in changes.values() for c in v}
    print(f"new unique sentences: {len(uniq_new):,}  within-rig duplicate sentences among "
          f"rewritten rows: {dup_in_rig:,} ({dup_in_rig/max(n_low,1)*100:.2f}%)")
    ln = np.array([len(s.split()) for s in uniq_new])
    print(f"sentence length (words): min={ln.min()} median={int(np.median(ln))} max={ln.max()}")
    print("\n--- 12 sample sentences ---")
    for s in sorted(uniq_new)[:: max(1, len(uniq_new) // 12)][:12]:
        print("   ", s)

    Path(a.changelog).parent.mkdir(parents=True, exist_ok=True)
    Path(a.changelog).write_text(json.dumps(
        {"__meta": {"side_tol": SIDE_TOL, "place_tol": PLACE_TOL, "anchor_max_hops": ANCHOR_MAX_HOPS,
                    "left_is": "+X", "rows_rewritten": n_low, "rigs": rigs_touched,
                    "generator": Path(__file__).name},
         "rigs": changes}, indent=1, sort_keys=True))
    print(f"\n[changelog] {a.changelog} "
          f"({Path(a.changelog).stat().st_size/1e6:.2f} MB) sha256 "
          f"{hashlib.sha256(Path(a.changelog).read_bytes()).hexdigest()[:16]}")

    if not a.apply:
        print("\nDRY RUN: no skeleton npz was touched.  Re-run with --apply to write.")
        return 0

    bak = Path(a.backup)
    bak.mkdir(parents=True, exist_ok=True)
    written, reused_bak = 0, 0
    for rig, new in new_desc_all.items():
        p = skel / f"{rig}.npz"
        b = bak / f"{rig}.npz"
        # THE DESTINATION IS THE SOURCE OF THE PRESERVED FIELDS.  Reading them from a backup lets a
        # stale one silently revert real data while the rewrite still reports "11 fields identical".
        with np.load(p, allow_pickle=False) as zc:
            if set(zc.files) != set(KEEP) | {"joint_descriptions"}:
                raise SystemExit(f"REFUSED: {rig} has fields {sorted(zc.files)}, not the 12 expected")
            cur = {k: np.array(zc[k]) for k in zc.files}
        if b.exists():
            # An existing backup must be THIS skeleton's own history: the eleven preserved fields
            # byte for byte.  Only joint_descriptions may differ -- that is what an earlier run moved.
            with np.load(b, allow_pickle=False) as zb:
                if set(zb.files) != set(cur):
                    raise SystemExit(f"REFUSED: backup {b} has fields {sorted(zb.files)}, "
                                     f"not the 12 the destination carries")
                for k in KEEP:
                    u, v = np.array(zb[k]), cur[k]
                    if u.dtype != v.dtype or u.shape != v.shape or u.tobytes() != v.tobytes():
                        raise SystemExit(f"REFUSED: backup {b} disagrees with the current {rig} on "
                                         f"{k}; it is not this skeleton's backup, delete or move it")
            reused_bak += 1
        else:
            shutil.copy2(p, b)
        payload = {k: cur[k] for k in KEEP}
        payload["joint_descriptions"] = np.array(new)

        def audit(path):
            """FAIL LOUD: the rewrite must move exactly one field, off the CURRENT destination."""
            with np.load(path, allow_pickle=False) as chk:
                if set(chk.files) != set(KEEP) | {"joint_descriptions"}:
                    raise SystemExit(f"REFUSED: {rig} field set changed: {sorted(chk.files)}")
                for k in KEEP:
                    u, v = cur[k], np.array(chk[k])
                    if u.dtype != v.dtype or u.shape != v.shape or u.tobytes() != v.tobytes():
                        raise SystemExit(f"REFUSED: {rig} field {k} changed on rewrite")
                if str(chk["joint_order_sha256"]) != hashlib.sha256(
                        "|".join(str(v) for v in chk["joint_names"]).encode()).hexdigest():
                    raise SystemExit(f"REFUSED: {rig} joint_order_sha256 no longer matches joint_names")
                if [str(v) for v in chk["joint_descriptions"]] != list(new):
                    raise SystemExit(f"REFUSED: {rig} joint_descriptions did not land")

        # ATOMIC: build and verify a sibling, then rename over the destination.  A crash anywhere
        # before the rename leaves the destination exactly as it was.
        tmp = p.with_name(f"{p.name}.tmp{os.getpid()}.npz")
        try:
            np.savez_compressed(tmp, **payload)
            audit(tmp)
            os.replace(tmp, p)
        finally:
            if tmp.exists():
                tmp.unlink()
        audit(p)
        written += 1
    print(f"\n[apply] rewrote {written} skeleton npz atomically; originals under {bak} "
          f"({reused_bak} backups already existed and were validated against the destination)")
    print(f"[apply] verified for every one, against a snapshot of the DESTINATION: 11 other fields "
          f"bit-identical, joint_order_sha256 still == sha256('|'.join(joint_names))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
