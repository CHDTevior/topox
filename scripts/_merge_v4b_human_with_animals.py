#!/usr/bin/env python3
"""Merge the v4b-272 human AnyTop dataset with an animal AnyTop dataset into ONE training dir for the shared VQVAE.

Replicates the finished-run merge (data/animo4d_anytop_clean_L4_safe_plus_humanml3d_v3a): the loader reads ONE
data_root, so we physically union cond.npy (animal object types + HML3D_Human), hardlink both motions/ (+ animal
motions_heldout/), union captions, regenerate FULLY-COVERING splits via the loader's own deterministic md5-stratified
algorithm (AnyTopDataset use_split_file=False, val_frac=0.05 seed=42), and write MERGE_SUMMARY.json. No file is
modified in the source dirs; motions are hardlinked (symlink fallback). Do NOT copy any _cond_normalized_J*.pkl cache
— the loader rebuilds it from the new cond.npy.

Usage: python scripts/_merge_v4b_human_with_animals.py --human DIR --animal DIR --out DIR
"""
import os, json, argparse, pathlib, sys
import numpy as np
ROOT = pathlib.Path("/scratch/ts1v23/workspace/noKslot_clean"); sys.path.insert(0, str(ROOT))

def link_all(src, dst):
    if not os.path.isdir(src): return 0
    os.makedirs(dst, exist_ok=True); n=0
    for f in os.listdir(src):
        if not f.endswith(".npy"): continue
        s=os.path.join(src,f); d=os.path.join(dst,f)
        if os.path.exists(d): continue
        try: os.link(s,d)
        except OSError: os.symlink(os.path.abspath(s),d)
        n+=1
    return n

def load_caps(base):
    for cf in ("motion_texts_by_file_with_codex_drafts.json","motion_texts_by_file.json"):
        p=f"{base}/{cf}"
        if os.path.exists(p): return json.load(open(p))
    return {}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--human", default=str(ROOT/"data/humanml3d_anytop13_v4b_272_neutral"))
    ap.add_argument("--animal", required=True, help="e.g. data/animo4d_anytop_clean_L4_safe_plus_truebones")
    ap.add_argument("--out", required=True)
    a=ap.parse_args()
    if os.path.exists(a.out): print(f"ERROR: {a.out} exists; remove or pick fresh --out."); return
    for d in (a.human,a.animal):
        assert os.path.isfile(f"{d}/cond.npy") and os.path.isdir(f"{d}/motions"), f"{d} not a valid AnyTop dataset"
    os.makedirs(a.out)
    # 1) union cond.npy (per-object mean/std/offsets kept as-is; no cross-object renorm)
    ac=np.load(f"{a.animal}/cond.npy",allow_pickle=True).item()
    hc=np.load(f"{a.human}/cond.npy",allow_pickle=True).item()
    dup=set(ac)&set(hc); assert not dup, f"cond object-key COLLISION animal vs human: {sorted(dup)[:5]}"
    merged={**ac,**hc}; np.save(f"{a.out}/cond.npy", merged, allow_pickle=True)
    # 2) hardlink motions/ (+ heldout) from both
    na=link_all(f"{a.animal}/motions", f"{a.out}/motions"); nh=link_all(f"{a.human}/motions", f"{a.out}/motions")
    hh=link_all(f"{a.animal}/motions_heldout", f"{a.out}/motions_heldout")+link_all(f"{a.human}/motions_heldout", f"{a.out}/motions_heldout")
    # 3) union captions
    caps={}; caps.update(load_caps(a.animal)); caps.update(load_caps(a.human))
    json.dump(caps, open(f"{a.out}/motion_texts_by_file.json","w"))
    # 4) regenerate fully-covering splits via the loader's own algorithm (fail-loud on drift)
    from src.data.anytop_dataset import AnyTopDataset
    common=dict(data_root=a.out, val_frac=0.05, seed=42, max_joints=144, load_captions=False, use_split_file=False)
    tr=[pathlib.Path(s["path"]).name for s in AnyTopDataset(split="train",**common).samples]
    va=[pathlib.Path(s["path"]).name for s in AnyTopDataset(split="val",**common).samples]
    assert not (set(tr)&set(va)), "train/val OVERLAP"
    on_disk=len([f for f in os.listdir(f"{a.out}/motions") if f.endswith(".npy")])
    assert len(tr)+len(va)==on_disk, f"splits do NOT cover all motions: {len(tr)+len(va)} vs {on_disk} on disk"
    os.makedirs(f"{a.out}/splits",exist_ok=True)
    open(f"{a.out}/splits/train.txt","w").write("# auto-generated (merge)\n"+"\n".join(tr)+"\n")
    open(f"{a.out}/splits/val.txt","w").write("# auto-generated (merge)\n"+"\n".join(va)+"\n")
    # 5) summary
    n_h=sum(1 for k in merged if k.upper().startswith("HML"))
    summ=dict(cond_objects=len(merged), animal_objects=len(merged)-n_h, human_objects=n_h,
              motions_total=na+nh, motions_animal=na, motions_human=nh, motions_heldout=hh,
              split_train=len(tr), split_val=len(va), coverage_ok=(len(tr)+len(va)==on_disk),
              human_dir=a.human, animal_dir=a.animal)
    json.dump(summ, open(f"{a.out}/MERGE_SUMMARY.json","w"), indent=1)
    print("=== MERGE DONE ===\n"+json.dumps(summ,indent=1))
    print(f"\nlaunch VQVAE with ANYTOP_ROOT={a.out} MAX_JOINTS=144 MAX_COARSE=96 NUM_CODES=8192 (name OUT WITHOUT 'L4safeHuman' for the watchdog guard).")

if __name__=="__main__": main()
