"""Verify the FULL v3a human dataset against the AnyTop13/animal contract (checks 1-6).
READ-ONLY on the data, NO GPU, NO writes to v2/canonical. Check 7 (loader smoke) is separate.
"""
from __future__ import annotations
import sys, json, csv, time
from pathlib import Path
import numpy as np

REPO = "/iridisfs/scratch/ts1v23/workspace/noKslot_clean"
HM = "/iridisfs/scratch/ts1v23/workspace/motion-latent-diffusion-main"
sys.path.insert(0, HM); sys.path.insert(0, REPO)
import importlib.util as _u
def _load(name, path):
    s = _u.spec_from_file_location(name, path); m = _u.module_from_spec(s); s.loader.exec_module(m); return m
cv = _load("cv", REPO + "/scripts/convert_humanml3d_to_anytop13.py")
gr = _load("gr", REPO + "/scripts/_v3_gate_runner.py")

V2 = Path(REPO) / "data/humanml3d_anytop13_v2_shared_reencoded"
V3 = Path(REPO) / "data/humanml3d_anytop13_v3a_shared_reencoded"
OT = "HML3D_Human"
SC = gr.SINGLE_CHILD_JOINTS
results = []
def rec(name, ok, msg): results.append((name, ok, msg)); print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {msg}")

c3 = np.load(V3 / "cond.npy", allow_pickle=True).item()
c2 = np.load(V2 / "cond.npy", allow_pickle=True).item()
o3, o2 = c3[OT], c2[OT]

# ---- 1 STRUCTURE ----
print("=== 1 STRUCTURE (cond v3a vs v2) ===")
struct_keys = ["joints_names", "parents", "offsets", "kinematic_chains", "joint_relations", "joints_graph_dist", "tpos_first_frame"]
def _eq(a, b):
    try:
        return bool(np.array_equal(np.asarray(a), np.asarray(b)))
    except Exception:
        return repr(a) == repr(b)
bad = [k for k in struct_keys if not _eq(o3[k], o2[k])]
rec("1.struct_identical", len(bad) == 0 and list(c3.keys()) == [OT] and o3["object_type"] == OT,
    f"object_type={o3['object_type']}; differing keys={bad if bad else 'NONE'}")

# ---- 2 cond mean/std ----
print("=== 2 cond mean/std ===")
m3, s3, m2, s2 = o3["mean"], o3["std"], o2["mean"], o2["std"]
finite = bool(np.isfinite(m3).all() and np.isfinite(s3).all())
grp = {"ch0:3": slice(0, 3), "ch3:9": slice(3, 9), "ch9:12": slice(9, 12), "ch12": slice(12, 13)}
dmean = {g: float(np.abs(m3[:, sl] - m2[:, sl]).max()) for g, sl in grp.items()}
dstd = {g: float(np.abs(s3[:, sl] - s2[:, sl]).max()) for g, sl in grp.items()}
ch39_differs = dmean["ch3:9"] > 1e-9 or dstd["ch3:9"] > 1e-9
others_same = all(dmean[g] < 1e-9 and dstd[g] < 1e-9 for g in ("ch0:3", "ch9:12", "ch12"))
rec("2.mean_std", finite and ch39_differs and others_same,
    f"finite={finite}; ch3:9 dmean={dmean['ch3:9']:.4g}/dstd={dstd['ch3:9']:.4g} (DIFFER); "
    f"ch0:3/9:12/12 dmean={dmean['ch0:3']:.1e}/{dmean['ch9:12']:.1e}/{dmean['ch12']:.1e} (~0)")

# ---- helpers for motion-level checks ----
def list_motions(loc):
    return sorted(p.name for p in (V3 / loc).glob("*.npy"))
def src_id(fname):
    return fname.replace(f"{OT}_", "").replace(".npy", "")
def load_motion(root, loc, fname):
    return np.load(root / loc / fname)

# ---- 3 PARITY (~20 clips, train+heldout) ----
print("=== 3 PARITY (v3a vs v2 motions) ===")
mo = list_motions("motions"); hd = list_motions("motions_heldout")
sample = [("motions", f) for f in mo[::len(mo)//10][:10]] + [("motions_heldout", f) for f in hd[::len(hd)//10][:10]]
worst_other, min_nonroot, n3 = 0.0, 1e9, 0
for loc, f in sample:
    a = load_motion(V3, loc, f); b = load_motion(V2, loc, f)
    o = max(float(np.abs(a[:, :, 0:3] - b[:, :, 0:3]).max()),
            float(np.abs(a[:, :, 9:12] - b[:, :, 9:12]).max()),
            float(np.abs(a[:, :, 12] - b[:, :, 12]).max()),
            float(np.abs(a[:, 0, 3:9] - b[:, 0, 3:9]).max()))   # root ch3:9
    worst_other = max(worst_other, o)
    min_nonroot = min(min_nonroot, float(np.abs(a[:, 1:, 3:9] - b[:, 1:, 3:9]).max()))
    n3 += 1
rec("3.parity", worst_other == 0.0 and min_nonroot > 0.0,
    f"{n3} clips; OTHER-channel(pos/vel/contact/root-ch3:9) max delta={worst_other:.2e} (must be 0); "
    f"non-root ch3:9 min-clip max-delta={min_nonroot:.4g} (must be >0)")

# ---- 4 PERSISTED == VALIDATED CODE (~5 clips) ----
print("=== 4 PERSISTED == reencode_rot6d(v3a) ===")
off = cv.compute_offsets()
worst4, n4 = 0.0, 0
for loc, f in [("motions", mo[i]) for i in (0, 5000, 12000, 20000, 24000)]:
    sid = src_id(f)
    x = np.load(Path(cv.SRC) / "new_joint_vecs" / f"{sid}.npy")
    recomp = cv.reencode_rot6d(cv.convert_263_to_13(x), cv.world_positions(x), off, rot6d_mode="v3a")
    pers = load_motion(V3, loc, f)
    worst4 = max(worst4, float(np.abs(recomp[:, :, 3:9] - pers[:, :, 3:9]).max())); n4 += 1
rec("4.persisted_eq_code", worst4 < 1e-6, f"{n4} clips; persisted ch3:9 vs in-memory v3a max delta={worst4:.2e} (must be ~0)")

# ---- 5 TWIST-SMOOTHNESS on persisted v3a (~30 clips) ----
print("=== 5 TWIST-SMOOTHNESS on persisted v3a ===")
u = o3["offsets"][SC].astype(np.float64); u = u / (np.linalg.norm(u, axis=-1, keepdims=True) + 1e-12)
vals = []
for f in mo[::len(mo)//30][:30]:
    a = load_motion(V3, "motions", f)
    if a.shape[0] < 3:
        continue
    R = gr._sixd_to_mat(a[:, SC, 3:9].astype(np.float64))
    vals.append(gr._so3_accel_vals_deg(R))
pooled = np.concatenate(vals)
med = float(np.median(pooled)); fg30 = float((pooled > 30).mean()); p95 = float(np.percentile(pooled, 95))
rec("5.twist_smoothness", med <= 13.0 and fg30 < 0.05,
    f"{len(vals)} clips; SO(3)-accel median={med:.2f}deg p95={p95:.2f} frac>30={fg30*100:.3f}% (animal band ~1-2deg)")

# ---- 6 COMPLETENESS / fail-loud ----
print("=== 6 COMPLETENESS / fail-loud ===")
mo2 = sorted(p.name for p in (V2 / "motions").glob("*.npy"))
hd2 = sorted(p.name for p in (V2 / "motions_heldout").glob("*.npy"))
set_ok = (set(mo) == set(mo2)) and (set(hd) == set(hd2))
cnt_ok = len(mo) == len(mo2) == 24838 and len(hd) == len(hd2) == 4388
# scan ALL for non-finite
t0 = time.time(); nonfinite = []; scanned = 0
for loc, names in (("motions", mo), ("motions_heldout", hd)):
    for f in names:
        arr = np.load(V3 / loc / f, mmap_mode="r")
        if not np.isfinite(np.asarray(arr)).all():
            nonfinite.append(f)
        scanned += 1
        if scanned % 8000 == 0:
            print(f"    ...scanned {scanned} ({time.time()-t0:.0f}s)")
# aux files counts vs v2
def nlines(p):
    return sum(1 for _ in open(p)) if Path(p).exists() else -1
splits_ok = all(nlines(V3 / "splits" / s) == nlines(V2 / "splits" / s) for s in ("train.txt", "val.txt", "test.txt", "all.txt"))
oi_ok = nlines(V3 / "object_index.csv") == nlines(V2 / "object_index.csv")
cap3 = len(json.load(open(V3 / "motion_texts_by_file.json"))); cap2 = len(json.load(open(V2 / "motion_texts_by_file.json")))
cap_ok = cap3 == cap2
rec("6.completeness", set_ok and cnt_ok and len(nonfinite) == 0 and splits_ok and oi_ok and cap_ok,
    f"counts 24838+4388 set-match={set_ok}; scanned {scanned} non-finite={len(nonfinite)}; "
    f"splits_match={splits_ok} object_index_match={oi_ok} caption_match={cap_ok}({cap3}=={cap2})")

print("\n=== CHECKS 1-6 VERDICT ===")
allok = all(ok for _, ok, _ in results)
for name, ok, _ in results:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"=== {'ALL 1-6 PASS' if allok else 'SOME FAILED'} ===")
sys.exit(0 if allok else 1)
