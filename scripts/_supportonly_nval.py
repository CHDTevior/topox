"""How many validation clips of one rig a cut leaves in a view -- the number of samples each arm must produce.

The chain guards its dump steps on this count: a directory that merely holds SOME .world.npz files would let a
rerun skip an interrupted sampling pass, and the comparator only checks that the two arms carry the SAME targets,
not that they carry all of them (codex 2026-09-10 r1 #5).
"""
import json, sys

view, cut, rig = sys.argv[1], sys.argv[2], sys.argv[3]
drop = set(json.load(open(cut))["clips"])
rows = [json.loads(l) for l in open(view + "/manifests/clips.jsonl")]
print(sum(1 for r in rows if r.get("status") == "accept" and str(r.get("split")) == "val"
          and str(r["rig_id"]) == rig and str(r["clip_id"]) not in drop))
