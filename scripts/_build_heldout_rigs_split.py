"""Held-out-rig split for the library (post-submission study, user 2026-09-13): hold out N_UNIQUE rigs whose kinematic
tree is unique in the library and N_SHARED rigs whose tree is shared with rigs that stay in training, remove every clip of
those rigs from training, and ALSO remove (a) every training clip on any other rig whose motion file is byte-identical to a
held-out clip (same-source exports) and (b) every training clip on a tree-sibling of a held-out rig that carries the same
source_action_name (the same animation retargeted within the tree). The held-out rigs' clips (all of them) become the
evaluation targets. Output: a merged exclusion list in the launcher's format (base cut + held-out exclusions) and a
companion JSON describing the held-out set.
  python scripts/_build_heldout_rigs_split.py --root dataset/ktjd17_pzh312_noik_v2 --hashes runs/_heldout/motion_sha256_v2.json \
      --base_cut configs/pilot_animal_only_exclusions.json --out configs/heldout20_v1_exclusions.json --seed 0"""
import argparse, collections, glob, hashlib, json, os, random, sys
import numpy as np
sys.path.insert(0, os.getcwd())
from src.data.holdout_guard import canonical_form   # AHU canonical form: the same unordered tree under any joint ordering

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True); ap.add_argument("--hashes", required=True); ap.add_argument("--base_cut", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n_unique", type=int, default=10); ap.add_argument("--n_shared", type=int, default=10)
ap.add_argument("--min_clips", type=int, default=100)
a = ap.parse_args()

base = json.load(open(a.base_cut)); base_drop = set(base["clips"])
recs = [json.loads(l) for l in open(os.path.join(a.root, "manifests/clips.jsonl"))]
recs = [r for r in recs if r["status"] == "accept" and r["clip_id"] not in base_drop]      # the active set of the base cut
sha = json.load(open(a.hashes))
by_rig = collections.defaultdict(list)
for r in recs: by_rig[r["rig_id"]].append(r)
rigs = sorted(by_rig)
tree = {}
for rig in rigs:
    z = np.load(os.path.join(a.root, "skeletons", rig + ".npz"), allow_pickle=True)
    par = tuple(int(x) for x in np.asarray(z["parents"]).ravel())
    tree[rig] = (len(par), canonical_form(par))                  # joint count + unordered-tree identity (codex r1 P1)
groups = collections.defaultdict(list)
for rig in rigs: groups[tree[rig]].append(rig)
unique = [rig for rig in rigs if len(groups[tree[rig]]) == 1 and len(by_rig[rig]) >= a.min_clips]
shared = [rig for rig in rigs if len(groups[tree[rig]]) >= 3 and len(by_rig[rig]) >= a.min_clips]   # >= 2 siblings stay in training

def stratified(cands, n, rng):
    """n rigs spread over joint count: sort by J, cut into n bins, one random rig per bin."""
    cands = sorted(cands, key=lambda r: (tree[r][0], r)); picks = []
    for k in range(n):
        lo, hi = k * len(cands) // n, (k + 1) * len(cands) // n
        picks.append(rng.choice(cands[lo:hi]))
    return picks

rng = random.Random(a.seed)
held_u = stratified(unique, a.n_unique, rng)
shared_trees = sorted({tree[r] for r in shared}, key=lambda t: (t[0], t[1]))   # eligible trees, by joint count (codex r1 P2)
held_s = []
for k in range(a.n_shared):                                  # one tree per joint-count bin, one eligible rig from it
    lo, hi = k * len(shared_trees) // a.n_shared, (k + 1) * len(shared_trees) // a.n_shared
    t = rng.choice(shared_trees[lo:hi])
    held_s.append(rng.choice(sorted(r for r in shared if tree[r] == t)))
assert len(held_s) == a.n_shared and len({tree[r] for r in held_s}) == a.n_shared
held = held_u + held_s; held_set = set(held)

held_clips = [r for r in recs if r["rig_id"] in held_set]
held_sha = {sha[r["clip_id"]] for r in held_clips}
action = lambda r: r["source_action_name"].split("@", 1)[-1]          # "aardvark_female@burrowexit..." -> the action part
held_actions = {(tree[r["rig_id"]], action(r)) for r in held_clips}
drop = {}; why = collections.Counter()
for r in recs:
    cid = r["clip_id"]
    if r["rig_id"] in held_set: drop[cid] = "all"; why["held-out rig"] += 1
    elif sha[cid] in held_sha: drop[cid] = "all"; why["byte-identical to a held-out clip"] += 1
    elif (tree[r["rig_id"]], action(r)) in held_actions: drop[cid] = "all"; why["same source action on a tree-sibling"] += 1
survivors = {}                                               # the tree must remain seen: >= 2 siblings with TRAINING clips left
for rig in held_s:
    left = [s for s in groups[tree[rig]] if s not in held_set
            and any(r["clip_id"] not in drop and r["split"] == "train" for r in by_rig[s])]
    assert len(left) >= 2, (rig, "siblings left with training clips:", left)
    survivors[rig] = len(left)
merged = {"clips": {**base["clips"], **drop},
          "note": (f"{base['note']} PLUS held-out-rig split v1 (seed {a.seed}): {len(held)} rigs held out "
                   f"({a.n_unique} unique-tree, {a.n_shared} shared-tree) with all their clips, byte-identical clips on other rigs "
                   f"and same-source clips on tree-siblings removed from training; see {os.path.splitext(a.out)[0]}_heldout.json")}
json.dump(merged, open(a.out, "w"), indent=0)
info = {"seed": a.seed, "root": a.root, "hashes_sha256": hashlib.sha256(open(a.hashes, "rb").read()).hexdigest(),
        "heldout_unique_tree": [{"rig": r, "J": tree[r][0], "n_clips": len(by_rig[r])} for r in held_u],
        "heldout_shared_tree": [{"rig": r, "J": tree[r][0], "n_clips": len(by_rig[r]), "siblings_in_training": survivors[r]} for r in held_s],
        "removed_from_training": dict(why), "active_before": len(recs), "active_after": len(recs) - len(drop),
        "eval_clips": [r["clip_id"] for r in held_clips]}
json.dump(info, open(os.path.splitext(a.out)[0] + "_heldout.json", "w"), indent=1)
print(json.dumps({k: v for k, v in info.items() if k != "eval_clips"}, indent=1))
