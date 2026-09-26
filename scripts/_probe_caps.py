import json, os
base = "data/animo4d_L4TB_plus_human_v4b272neutral"
texts = json.load(open(os.path.join(base, "motion_texts_by_file.json")))
def cap_for(fn):
    v = texts.get(fn) or texts.get(fn.replace(".npy","")) or texts.get(fn+".npy")
    if isinstance(v, list) and v: return v[0]
    if isinstance(v, str): return v
    if isinstance(v, dict):
        for kk in ("caption","text","captions"):
            if kk in v:
                x = v[kk]; return x[0] if isinstance(x,list) else x
    return None
for name, n in [("val_human.json", 40), ("val_animal.json", 24)]:
    d = json.load(open(os.path.join(base, "eval_splits", name)))
    print(f"\n===== {name} (showing {n}) =====")
    for e in d[:n]:
        fn = e["filename"]
        print(f"{fn:40s} | {str(cap_for(fn))[:85]}")
