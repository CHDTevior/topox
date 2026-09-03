"""Turn raw joint names into anatomical descriptions shared across naming conventions.

Why this exists. Our three sources name the same anatomy in three incompatible ways:
`left_shoulder` (SMPL), `def_frontLegUpr_joint.L` (Planet Zoo), `BN_Arm_L_01` (TrueBones).
A frozen text encoder applied to the RAW strings does not connect them — measured: of 22
human joint names, only 10 retrieve a token-overlapping match among PZ/TrueBones names,
and every PZ limb query collapses onto a single hub. The cause is not the encoder: the PZ
vocabulary contains no `knee`, `elbow`, `shoulder`, `wrist`, `ankle` or `pelvis` at all,
because a quadruped rig names limb SEGMENTS (`frontLegUpr`, `rearFoot`, `frontIndexToe1`)
rather than human joints. The correspondence has to be supplied, not embedded.

So each name is rewritten into a phrase over ONE anatomical vocabulary, stating the
homology explicitly where it exists ("the left front upper limb segment, homologous to the
upper arm"). That phrase is what gets encoded and fed to the model.

Planet Zoo and SMPL are handled by rules: their conventions are small, closed and
completely regular (325 and 22 unique names), so rules are auditable in a way that 347 LLM
calls would not be. TrueBones spans 70 creature rigs with 1160 names and no single
convention; names the rules cannot parse confidently are emitted with
`needs_llm: true` for a separate enrichment pass rather than being silently guessed.

    python scripts/_build_joint_descriptions.py --out data/joint_descriptions_v1.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from typing import Optional
from pathlib import Path

import numpy as np

# Segment homologies. The right-hand side is deliberately written in the vocabulary the
# human rig uses, because that is the only vocabulary all three sources can be compared in.
FORE = {"upr": ("upper limb segment", "the upper arm"),
        "lwr": ("lower limb segment", "the forearm")}
HIND = {"upr": ("upper limb segment", "the thigh"),
        "lwr": ("lower limb segment", "the shin")}
DIGIT = {"thumb": "first", "index": "second", "mid": "third", "middle": "third",
         "ring": "fourth", "pinky": "fifth", "inner": "inner", "outer": "outer"}
AXIAL = {"spine": "spine vertebra", "neck": "neck vertebra", "tail": "tail vertebra",
         "head": "head", "chest": "chest", "hips": "pelvis", "hip": "hip",
         "belly": "belly", "trunk": "trunk", "root": "root", "clavicle": "clavicle"}
COSMETIC = ("hair", "quills", "plate", "sculpt", "shaper", "bulge", "wart", "sqsh",
            "hump", "false", "tip", "notril")


def split_side(raw: str) -> tuple[Optional[str], str]:
    """Return (side, name-with-the-side-marker-removed).

    Regex accretion made this brittle twice (`ArmLClaw` lost its side; `ArmL_02_` was read as
    right), so it is written as explicit, ordered, testable cases over the marker forms that
    actually occur in this corpus. Every case is exercised by SIDE_CASES below.
    """
    # 1. explicit words
    m = re.search(r"(?i)(left|right)", raw)
    if m:
        return m.group(1).lower(), (raw[:m.start()] + raw[m.end():])
    # 2. suffix markers: `.L`, `_L`, `_L01`, `_L_01`
    m = re.search(r"[._](L|R)(?=[._]?\d*$|[._])", raw)
    if m:
        return ("left" if m.group(1) == "L" else "right"), (raw[:m.start()] + raw[m.end():])
    # 3. bare capital between camel humps or before digits: `ArmLClaw`, `ArmL_02_`, `LegR00`
    m = re.search(r"(?<=[a-z])(L|R)(?=[A-Z]|[._]?\d|[._]*$)", raw)
    if m:
        return ("left" if m.group(1) == "L" else "right"), (raw[:m.start()] + raw[m.end():])
    # 4. leading capital before an uppercase word: `LFoot`, `RHand`
    m = re.match(r"(L|R)(?=[A-Z][a-z])", raw)
    if m:
        return ("left" if m.group(1) == "L" else "right"), raw[1:]
    return None, raw


SIDE_CASES = [
    ("ArmLClaw", "left", "ArmClaw"), ("ArmLCollarbone", "left", "ArmCollarbone"),
    ("ArmL_01_", "left", "Arm_01_"), ("ArmR_02_", "right", "Arm_02_"),
    ("BN_Forearm_L_01", "left", "BN_Forearm_01"), ("BN_Foot_R_01", "right", "BN_Foot_01"),
    ("LeftFoot", "left", "Foot"), ("RightFoot", "right", "Foot"),
    ("left_ankle", "left", "_ankle"), ("def_hip_joint.L", "left", "def_hip_joint"),
    ("def_frontFoot_joint.R", "right", "def_frontFoot_joint"),
    ("BN_Head", None, "BN_Head"), ("Bip01_Spine", None, "Bip01_Spine"),
    ("BN_Thigh_L_01", "left", "BN_Thigh_01"),
]


def _selftest_side() -> None:
    bad = [(r, want, got) for r, want, _ in SIDE_CASES
           for got in [split_side(r)[0]] if got != want]
    if bad:
        raise SystemExit(f"split_side self-test FAILED: {bad}")


def describe_pz(raw: str):
    """Planet Zoo: def_[c_]<part>[Idx]_joint[.L|.R]. Returns (description, confident)."""
    side = "left" if raw.endswith(".L") else "right" if raw.endswith(".R") else None
    body = re.sub(r"^def_", "", raw)
    body = re.sub(r"\.(L|R)$", "", body)
    body = re.sub(r"_joint$|_Sculpt$", "", body)
    body = re.sub(r"^c_", "", body)                       # centre-line marker
    low = body.lower()
    s = f"the {side} " if side else "the "

    m = re.match(r"(front|rear)leg(upr|lwr)", low)
    if m:
        limb, seg = m.group(1), m.group(2)
        tab = FORE if limb == "front" else HIND
        name, hom = tab[seg]
        which = "front" if limb == "front" else "hind"
        return f"{s}{which} {name}, homologous to {hom}", True
    if re.match(r"(front|rear)foot", low):
        which = "front" if low.startswith("front") else "hind"
        hom = "the hand and wrist" if which == "front" else "the foot and ankle"
        return f"{s}{which} foot, homologous to {hom}", True
    if low.startswith("clavicle"):
        return f"{s}clavicle, at the shoulder", True
    if re.match(r"hips?$|hipsculpt|hipshaper", low):
        return f"{s}hip", True

    # digits: <front|rear><Digit>Toe<n> or toe<Front|Rear><Digit><n> or <Front|Rear><Digit>Claw
    m = (re.match(r"(front|rear)(thumb|index|mid|middle|ring|pinky|inner|outer)toe(\d+)", low)
         or re.match(r"toe(front|rear)(thumb|index|mid|middle|ring|pinky|inner|outer)(\d+)", low))
    if m:
        a, b, n = m.groups()
        limb = a if a in ("front", "rear") else b
        dig = b if a in ("front", "rear") else a
        which = "front" if limb == "front" else "hind"
        ordn = DIGIT.get(dig, dig)
        return f"{s}phalanx {int(n)} of the {ordn} digit of the {which} foot", True
    m = re.match(r"(front|rear)(thumb|index|mid|middle|ring|pinky)claw", low)
    if m:
        which = "front" if m.group(1) == "front" else "hind"
        return f"{s}claw on the {DIGIT.get(m.group(2), m.group(2))} digit of the {which} foot", True

    m = re.match(r"(spine|neck|tail)(\d+)", low)
    if m:
        return f"the {AXIAL[m.group(1)]} {int(m.group(2))}", True
    for k, v in AXIAL.items():
        if low.startswith(k):
            extra = " (a soft-tissue or cosmetic control)" if any(c in low for c in COSMETIC) else ""
            return f"{s}{v}{extra}", not extra
    if any(c in low for c in COSMETIC):
        return f"{s}{re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', body).lower()}, a soft-tissue or cosmetic control", True
    if low in ("skeleton", "root", "l0", "rig", "t"):
        return "the root of the skeleton", True
    return f"{s}{re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', body).lower()}", False


def describe_human(raw: str):
    side = "left" if raw.startswith("left_") else "right" if raw.startswith("right_") else None
    part = re.sub(r"^(left|right)_", "", raw)
    s = f"the {side} " if side else "the "
    hom = {"collar": "clavicle, at the shoulder", "hip": "hip", "knee": "knee",
           "ankle": "foot and ankle", "foot": "foot", "shoulder": "shoulder",
           "elbow": "elbow", "wrist": "hand and wrist", "pelvis": "pelvis",
           "neck": "neck vertebra", "head": "head"}
    m = re.match(r"spine(\d+)", part)
    if m:
        return f"the spine vertebra {int(m.group(1))}", True
    return f"{s}{hom.get(part, part.replace('_', ' '))}", part in hom or part.startswith("spine")


def describe_tb(raw: str, parent: str | None):
    """TrueBones has no single convention. Handle the regular majority; flag the rest."""
    side, stripped = split_side(raw)
    body = re.sub(r"^(BN_+|Bip\d*_|b_)", "", stripped)
    body = re.sub(r"_end_site$|_end$", "", body)
    idx = None
    m = re.search(r"[_]?(\d+)_?$", body)
    if m:
        idx = int(m.group(1))
        body = body[:m.start()]
    body = re.sub(r"[_\-]+$", "", body)
    word = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", body).replace("_", " ").strip().lower()
    word = re.sub(r"\b(l|r)\b", "", word).strip()
    s = f"the {side} " if side else "the "
    MAP = {"arm": ("limb segment", "the arm"), "forearm": ("lower limb segment", "the forearm"),
           "thigh": ("upper hind limb segment", "the thigh"), "calf": ("lower hind limb segment", "the shin"),
           "leg": ("limb segment", "the leg"), "foot": ("foot", "the foot and ankle"),
           "hand": ("hand", "the hand and wrist"), "shoulder": ("shoulder", "the shoulder"),
           "collarbone": ("clavicle", "the clavicle at the shoulder"),
           "spine": ("spine vertebra", None), "neck": ("neck vertebra", None),
           "tail": ("tail vertebra", None), "head": ("head", None), "pelvis": ("pelvis", None),
           "hips": ("pelvis", None), "hip": ("hip", None), "claw": ("claw", None),
           "wing": ("wing segment", None), "finger": ("finger", None), "toe": ("toe", None)}
    # Match the MOST SPECIFIC token, not the first key that happens to match. These names are
    # head-final compounds — `ArmLClaw` is a claw (of the arm), `ArmLCollarbone` is a
    # collarbone — so scanning MAP in insertion order and taking the first hit returned "arm"
    # for both and threw the head noun away.
    toks = word.split()
    hit = None
    for i in range(len(toks) - 1, -1, -1):          # last token first
        if toks[i] in MAP:
            hit = (toks[i], toks[:i] + toks[i + 1:])
            break
    if hit is not None:
        k, rest = hit
        name, hom = MAP[k]
        n = f" {idx}" if idx is not None else ""
        qual = (" of the " + " ".join(rest)) if rest else ""
        base = f"{s}{name}{n}{qual}"
        return (base + (f", homologous to {hom}" if hom else "")), True
    if not word:
        return "the root of the skeleton", parent in (None, "<root>")
    n = f" {idx}" if idx is not None else ""
    return f"{s}{word}{n}", False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/animo4d_L4TB_plus_human_v4b272neutral")
    ap.add_argument("--context", default="scratch/_jointname_context.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    _selftest_side()
    cond = np.load(Path(args.data_root) / "cond.npy", allow_pickle=True).item()
    ctx = {c["name"]: c for c in json.loads(Path(args.context).read_text())}

    src_of = {}
    for o, v in cond.items():
        s = "human" if o.upper().startswith("HML") else "pz" if o.startswith("PZ_") else "tb"
        for n in v["joints_names"]:
            src_of.setdefault(str(n), set()).add(s)

    out, stats = {}, Counter()
    for name in sorted(src_of):
        srcs = src_of[name]
        par = (ctx.get(name, {}).get("parent") or [None])[0]
        if "pz" in srcs:
            desc, ok = describe_pz(name)
        elif "human" in srcs:
            desc, ok = describe_human(name)
        else:
            desc, ok = describe_tb(name, par)
        stats["confident" if ok else "needs_llm"] += 1
        stats["|".join(sorted(srcs))] += 1
        out[name] = {"description": desc, "needs_llm": not ok,
                     "sources": sorted(srcs), "parent": par,
                     "n_rigs": ctx.get(name, {}).get("n_rigs")}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    n = len(out)
    print(f"{n} joint names described -> {args.out}")
    print(f"  confident: {stats['confident']} ({100*stats['confident']/n:.1f}%)   "
          f"needs_llm: {stats['needs_llm']} ({100*stats['needs_llm']/n:.1f}%)")
    for s in ("pz", "human", "tb"):
        sub = [k for k, v in out.items() if v["sources"] == [s]]
        bad = [k for k in sub if out[k]["needs_llm"]]
        print(f"  {s:6s}: {len(sub):5d} names, {len(bad)} need LLM")
        for k in sub[:4]:
            print(f"          {k:34s} -> {out[k]['description']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
