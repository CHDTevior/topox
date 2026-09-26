"""Clean validation subset of the active PZ corpus (codex 2026-09-06 P0-1: duplicate asset exports straddle the random
clip split).  A val clip is CLEAN iff (1) its exact motion array (all channels) does not occur in ANY active train clip
of any rig, and (2) no active train clip of the same rig shares its source_action_name (cuts / re-exports of the same
source animation).  Writes the clip ids with the rule, counts and content hashes so the subset is reproducible.
usage: python scripts/_build_pz_clean_val_subset.py <ktjd_root> <exclusions.json> <out.json>"""
import hashlib, json, sys, collections
import numpy as np
root, excl_path, out = sys.argv[1:4]
# snapshot the three identity inputs ONCE and derive both the selection and the recorded hashes from these bytes
# (codex 2026-09-06 r2 P1-2: re-reading them after the hashing pass could bind the record to different content)
excl_bytes = open(excl_path, "rb").read(); manifest_bytes = open(f"{root}/manifests/clips.jsonl", "rb").read()
gen_bytes = open(f"{root}/generation.json", "rb").read()
excl = json.loads(excl_bytes); exset = set(excl["clips"].keys())
rows = [json.loads(l) for l in manifest_bytes.decode().splitlines() if l.strip()]
act = [r for r in rows if r["status"] == "accept" and r["clip_id"] not in exset]
tr = [r for r in act if r["split"] == "train"]; va = [r for r in act if r["split"] == "val"]
def motion_sha(r):
    z = np.load(f"{root}/{r['motion_relpath']}"); k = [k for k in z.files if z[k].ndim >= 2][0]
    a = np.ascontiguousarray(z[k]); return hashlib.sha256(a.tobytes() + str(a.shape).encode()).hexdigest()
train_hash = {}
for i, r in enumerate(tr):
    train_hash.setdefault(motion_sha(r), []).append(r["clip_id"])
    if i % 10000 == 0: print(f"  hashed {i}/{len(tr)} train", flush=True)
train_src = collections.defaultdict(set)
for r in tr: train_src[r["rig_id"]].add(r["source_action_name"])
clean, dup_exact, dup_src, dup_both = [], 0, 0, 0
for r in va:
    h = motion_sha(r)
    e = h in train_hash; s_ = r["source_action_name"] in train_src[r["rig_id"]]
    dup_exact += int(e); dup_src += int(s_); dup_both += int(e and s_)       # independent predicates + intersection
    if e or s_: continue
    clean.append(dict(clip_id=r["clip_id"], rig_id=r["rig_id"], source_action_name=r["source_action_name"], T_target=int(r["T_target"]), motion_sha256=h))
rigs = sorted({c["rig_id"] for c in clean})
gen = json.loads(gen_bytes)
rep = dict(format="pz-clean-val-subset-v1", rule="exact motion sha256 absent from all active train clips AND (rig_id, source_action_name) absent from active train",
           ktjd_root=root, exclusions=excl_path,
           manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
           exclusions_sha256=hashlib.sha256(excl_bytes).hexdigest(), generation_id=str(gen["generation_id"]),
           n_active_train=len(tr), n_active_val=len(va), n_val_exact_dup_in_train=dup_exact, n_val_same_source_in_train=dup_src,
           n_val_both=dup_both, n_val_removed=len(va) - len(clean), n_clean=len(clean), n_clean_rigs=len(rigs), clean=clean,
           note="counts are independent predicates over the active val set (exact / same-source / both); removed = exact OR same-source")
json.dump(rep, open(out, "w"), indent=1)
print(f"active val {len(va)}: exact-dup {dup_exact}, same-source {dup_src} (both {dup_both}), removed {len(va)-len(clean)}, CLEAN {len(clean)} over {len(rigs)} rigs -> {out}")
