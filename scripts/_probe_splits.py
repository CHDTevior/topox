import json, os
base = "data/animo4d_L4TB_plus_human_v4b272neutral"
for name in ["val_human.json", "val_animal.json"]:
    p = os.path.join(base, "eval_splits", name)
    d = json.load(open(p))
    print(f"\n=== {name}: type={type(d).__name__}", end=" ")
    if isinstance(d, dict):
        print("keys=", list(d.keys())[:6], "len=", len(d))
        items = list(d.items())[:5]
        for k, v in items:
            vs = v if not isinstance(v, (list, dict)) else (str(v)[:80])
            print("  ", k, "->", vs)
    elif isinstance(d, list):
        print("len=", len(d))
        for e in d[:5]:
            print("  ", str(e)[:120])
# also list a few motion filenames on disk
mdir = os.path.join(base, "motions")
fs = sorted(os.listdir(mdir))
print("\n=== motions/ sample basenames (first 3 + any HML3D*) ===")
print("first:", fs[:3])
hml = [f for f in fs if f.startswith("HML3D")][:5]
pz  = [f for f in fs if f.startswith("PZ_")][:5]
print("HML3D*:", hml)
print("PZ_*:", pz)
print("total motions:", len(fs))
