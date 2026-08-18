#!/usr/bin/env python3
"""Lock the properties of the [demo|target] split that fail SILENTLY if broken.

Each test states the damage it prevents. Run: python3 tests/test_incontext_split.py
"""
import subprocess, sys, zlib
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.incontext_pairs import read_split, truebones_types

FAILS = []
def check(name, cond, damage):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        FAILS.append(f"{name}: {damage}")

print("[1] clip permutation is process-stable")
# The real dataset needs GPU-free but slow IO, so assert on the primitive the split relies on.
code = "import zlib;print(zlib.crc32(b'Alligator'))"
outs = {subprocess.run([sys.executable,"-c",code],capture_output=True,text=True).stdout.strip()
        for _ in range(3)}
check("crc32 identical across processes", len(outs) == 1,
      "train and eval processes would hold out DIFFERENT clips -> bucket A trains on its own eval set")
code_h = "print(abs(hash('Alligator')))"
outs_h = {subprocess.run([sys.executable,"-c",code_h],capture_output=True,text=True).stdout.strip()
          for _ in range(3)}
check("hash() is NOT stable (why crc32 is required)", len(outs_h) > 1,
      "if hash() were stable this test is vacuous, but the code must still not depend on it")
src = (Path(__file__).resolve().parents[1] / "src/data/incontext_pairs.py").read_text()
check("source does not use hash() for the split", "abs(hash(" not in src,
      "reintroducing hash() silently breaks the train/eval split")

print("[2] the FROZEN protocol is a real holdout")
SD = Path(__file__).resolve().parents[1] / "data/holdout_splits_v1"
tr, va = read_split(SD, "train"), read_split(SD, "val")
hr, hs = read_split(SD, "held_representative"), read_split(SD, "held_stress")
check("train and val are disjoint", not (tr & va), "val would be scored on trained clips")
check("held lists are disjoint from train+val", not ((hr | hs) & (tr | va)),
      "the holdout is not a holdout -- bucket B is meaningless")
check("held_representative and held_stress are disjoint", not (hr & hs),
      "the same rig would be both the fair test and the stress test")
rig = lambda ns: {n.split("___")[0] for n in ns if "___" in n}
check("held rigs never appear in train", not (rig(hr) & rig(tr)),
      "a rig split at CLIP level would leak the rig itself into bucket B")
check("all four lists are non-empty", all(len(x) for x in (tr, va, hr, hs)), "a split file is empty")

print("[3] source no longer re-derives its own split")
check("no home-grown skeleton split remains", "def split_skeletons" not in src,
      "a split by object_type leaks: distinct object types can share one canonical topology")

print("[4] pair-construction invariants (stub base, no data loading)")
from src.data.incontext_pairs import InContextPairs

class _StubBase:
    """Only what InContextPairs.__init__ touches: .samples with object_type + path."""
    def __init__(self, names_by_rig):
        self.samples = [{"object_type": ot, "path": f"/x/{n}.npy"}
                        for ot, ns in names_by_rig.items() for n in ns]

rig4 = {"RigA": [f"RigA___w_{i}" for i in range(4)]}
names4 = {f"RigA___w_{i}" for i in range(4)}
base = _StubBase(rig4)

same = InContextPairs(base, names4, names4)          # bucket-B style: same object on both sides
check("same-object (self-demo bucket) constructs", len(same) == 4,
      "bucket B legitimately uses one list for targets and demos; the overlap guard must not fire")
check("every target keeps a distinct demo",
      all(any(d != t for d in v["demos"]) for v in same.by_type.values() for t in v["targets"]),
      "a target whose only demo is itself becomes its own clean demonstration -- identity copy")

tr3 = {f"RigA___w_{i}" for i in range(3)}; va1 = {"RigA___w_3"}
ok_ab = InContextPairs(base, va1, tr3)               # bucket-A style: disjoint lists
check("disjoint target/demo lists construct", len(ok_ab) == 1, "bucket A must build")

try:
    InContextPairs(base, {"RigA___w_0", "RigA___w_3"}, tr3)   # 50% overlap, different objects
    raised = False
except ValueError:
    raised = True
check("PARTIAL overlap between different lists raises", raised,
      "a 30% leaked bucket scores the model on clips its demo pool trained on -- silently")

two = {"RigB": ["RigB___a_0", "RigB___b_1"]}
names2 = {"RigB___a_0", "RigB___b_1"}
ds2 = InContextPairs(_StubBase(two), names2, names2)
check("2-clip rig: both targets legal, each demos the other",
      sorted(ds2.by_type["RigB"]["targets"]) == sorted(ds2.by_type["RigB"]["demos"]) and len(ds2) == 2,
      "the smallest rigs (Chicken: 2 clips) must remain usable under K=1")

print()
if FAILS:
    print("FAILED:"); [print("  -", f) for f in FAILS]; sys.exit(1)
print("all split invariants hold")
