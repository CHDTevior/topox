#!/usr/bin/env python
"""Joint-name lexicon for scripts/deploy_generate.py: normalized joint-name key -> the corpus's clean anatomical name.

The training corpus (dataset/ktjd17_uniml3d_v2) describes a joint whose upstream clean name carries anatomy as
"<Side> <Clean> joint." ("mixamorig:LeftUpLeg_056" -> "Left Thigh joint.", "UpLeg.L" -> "Left Thigh joint."). A new rig's
joint names are mapped the same way through deploy_generate.name_side_key (side + key) and this table (key -> Clean),
so its descriptions come from the same vocabulary the joint-semantic embeddings were trained on. Rows whose description
is a structural sentence (the corpus's refill for content-free names) or a content-free placeholder do not vote.

Evaluation (printed and stored): rigs split 90 / 10 by a hash of the rig id; the table is built on the 90% and applied to
the held 10%'s name-template joints -- coverage (key found with share >= 0.5) and exact-match accuracy of the regenerated
description. The shipped table is then built on all rigs.

usage: python scripts/_build_deploy_joint_lexicon.py [--skeletons dataset/ktjd17_uniml3d_v2/skeletons]
           [--out configs/deploy/joint_name_lexicon_uniml3d_v2.json]
"""
import argparse, hashlib, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.deploy_generate import name_side_key                                    # noqa: E402
from scripts._uniml3d_fill_joint_descriptions import is_low_info                      # noqa: E402

TEMPLATE = re.compile(r"^(?:(Left|Right) )?(.+) joint\.$")


def rows(paths):
    for p in paths:
        z = np.load(p, allow_pickle=False)
        xs = np.asarray(z["P_rest_global"], dtype=np.float64)[:, 0] / float(z["s_rig"])
        for j, (nm, d) in enumerate(zip(z["joint_names"], z["joint_descriptions"])):
            nm, d = str(nm), str(d)
            m = TEMPLATE.match(d)
            if not m or is_low_info(d):
                continue
            yield p.stem, nm, (m.group(1) or "").lower() or None, m.group(2), float(xs[j])


def build(paths):
    votes, sides = defaultdict(Counter), defaultdict(Counter)
    for _, nm, dside, clean, _ in rows(paths):
        nside, key = name_side_key(nm)
        if key:
            votes[key][clean] += 1
            if nside is None:
                sides[key]["sided" if dside else "none"] += 1
    ent = {}
    for key, c in votes.items():
        clean, n = c.most_common(1)[0]
        tot = sum(c.values())
        # sided: the corpus describes this key with a side although the key function sees none in the names ('lleg',
        # 'mao esquerda'); deploy reads WHICH side off the geometry (a learnt side is inverted on mirrored assets)
        ns = sides[key]["sided"]
        sided = ns >= 2 and ns >= 0.95 * sum(sides[key].values())
        ent[key] = {"clean": clean, "count": tot, "share": round(n / tot, 4), "sided": bool(sided),
                    "alternatives": [[k, v] for k, v in c.most_common(4)[1:]]}
    return ent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skeletons", default="dataset/ktjd17_uniml3d_v2/skeletons")
    ap.add_argument("--out", default="configs/deploy/joint_name_lexicon_uniml3d_v2.json")
    a = ap.parse_args()
    paths = sorted(Path(a.skeletons).glob("*.npz"))
    held = [p for p in paths if int(hashlib.sha256(p.stem.encode()).hexdigest(), 16) % 10 == 0]
    train = [p for p in paths if p not in set(held)]
    ent = build(train)
    lex = {k: (v["clean"], v["sided"]) for k, v in ent.items() if v["share"] >= 0.5}
    n = hit = exact = side_ok = 0
    miss = Counter()
    for _, nm, dside, clean, x in rows(held):
        n += 1
        side, key = name_side_key(nm)
        if key in lex:
            hit += 1
            geo = "left" if x > 0.02 else ("right" if x < -0.02 else None)
            if side is None and lex[key][1]:
                side = geo
            elif side is not None and geo is not None and geo != side:
                side = geo                            # deploy's rule: a name mirrored against the geometry follows the geometry
            want = f"{dside.capitalize() + ' ' if dside else ''}{clean} joint."
            got = f"{side.capitalize() + ' ' if side else ''}{lex[key][0]} joint."
            exact += got == want
            side_ok += (side == dside)
        else:
            miss[key] += 1
    ev = {"held_rigs": len(held), "held_template_joints": n, "coverage": round(hit / max(n, 1), 4),
          "exact_given_covered": round(exact / max(hit, 1), 4), "side_agrees_given_covered": round(side_ok / max(hit, 1), 4),
          "exact_overall": round(exact / max(n, 1), 4), "top_missing_keys": miss.most_common(15)}
    print(json.dumps(ev, indent=1))
    full = build(paths)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"source": str(a.skeletons), "n_rigs": len(paths), "rigs": "every skeleton file of the corpus (train, val and excluded rigs alike)",
                               "key_function": "scripts/deploy_generate.py name_side_key",
                               "evaluation_90_10": ev, "entries": dict(sorted(full.items()))}, indent=0))
    print(f"[lexicon] {len(full)} keys ({sum(1 for v in full.values() if v['share'] >= 0.5)} with share >= 0.5) -> {out}")


if __name__ == "__main__":
    main()
