#!/usr/bin/env python3
"""
Stage-0/1 frame-align probe for the human real-twist recovery (v4) pipeline.
See handoff/20260701_235711_human_twist_recovery_v4_implementation.md (Rev 2), gate G-slicing / G-srcalign.

GOAL (no GPU, no training): decide the AMASS frame-slicing convention BEFORE any twist conversion.
index.csv gives (source_path, start_frame, end_frame) but does NOT state whether start/end index the
ORIGINAL-fps poses or the already-DOWNSAMPLED (20 fps) poses:
    A) downsample-then-slice : poses[::ds][start:end]     (start/end are 20fps-indexed)
    B) slice-then-downsample : poses[start:end][::ds]     (start/end are raw-fps-indexed)
For a NON-ZERO start clip these select different raw frames; only the true convention aligns to our
stored v3a clip. Guessing wrong -> twist injected at the WRONG time onto correct positions (positions
may pass, skinning breaks silently). So confirm at the CONTENT level.

TWO independent numeric checks (both must agree), computed on both conventions:
  1. LAG: internal flexion angle (knees 4/5, elbows 18/19) cross-correlated over integer lags.
     - AMASS side: numpy SMPL FK from poses[:, :66] axis-angle (with the clip's own betas) -> joints.
     - v3a side: same angle from stored RIC positions ch0:3.
     Correct convention -> lag 0, high corr.
  2. ABSOLUTE mm: per-frame similarity-Procrustes (Umeyama) MPJPE between AMASS-FK joints and v3a
     joints (coordinate-convention-invariant: removes global R,t,scale). Wrong convention -> poses
     mismatch per frame -> large MPJPE.

NUMERIC PASS (per user Stage-0 gate): the winning convention must, on EVERY non-zero-start clip,
have |lag|<=1 AND mean corr>0.95 AND per-frame Procrustes MPJPE < MM_THRESH, AND be CLEARLY better
than the loser (loser fails >=1 of these). If not, VERDICT=INCONCLUSIVE (do NOT proceed).

Also validates id-remap: our clip -> primary_caption -> amass_annotations key (NOT identity; identity
breaks by id~10000) -> index.csv row -> source npz + start/end, fail-loud on 0/ambiguous match, and
mirrored 'M' keys are filtered out (index.csv holds base ids only).

Usage: python scripts/_v4_framealign_probe.py
"""
import os, json, csv
import numpy as np

HM       = "/iridisfs/scratch/ts1v23/workspace/motion-latent-diffusion-main"
REPO     = "/scratch/ts1v23/workspace/noKslot_clean"
INDEX    = f"{HM}/datasets/humanml3d/index.csv"
ANN      = f"{HM}/datasets/humanml3d/amass_annotations.json"
AMASS_MD = f"{HM}/datasets/amass/motion_data"
V3A      = f"{REPO}/data/humanml3d_anytop13_v3a_shared_reencoded"
SIDECAR  = f"{V3A}/motion_texts_by_file.json"
SMPLH_DIR= f"{HM}/datasets/humanml3d/body_models/smplh"

PARENTS = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]
HINGES  = {"L_knee":(4,1,7), "R_knee":(5,2,8), "L_elbow":(18,16,20), "R_elbow":(19,17,21)}
TEST_CLIPS = ["000004","000013","000025","000559","000043","000058"]
ZERO_START = {"000043","000058"}
MM_THRESH  = 30.0     # per-frame Procrustes MPJPE ceiling (mm) for a PASS
CORR_THRESH= 0.95
LAG_TOL    = 1

# ---------------------------------------------------------------- SMPL FK (numpy, betas-aware)
def rodrigues(aa):                      # aa [N,3] -> R [N,3,3]
    th = np.linalg.norm(aa, axis=1, keepdims=True)
    r  = np.where(th < 1e-8, 0.0, aa/np.where(th<1e-8,1.0,th))
    c, s = np.cos(th)[:,None], np.sin(th)[:,None]
    x,y,z = r[:,0:1],r[:,1:2],r[:,2:3]
    K = np.zeros((aa.shape[0],3,3))
    K[:,0,1]=-z[:,0]; K[:,0,2]=y[:,0]; K[:,1,0]=z[:,0]; K[:,1,2]=-x[:,0]; K[:,2,0]=-y[:,0]; K[:,2,1]=x[:,0]
    return np.eye(3)[None] + s*K + (1-c)*(K@K)

_MODEL = {}
def load_model(gender):
    if gender not in _MODEL:
        p = f"{SMPLH_DIR}/{gender}/model.npz"
        if not os.path.exists(p): p = f"{SMPLH_DIR}/neutral/model.npz"
        m = np.load(p, allow_pickle=True)
        Jr = m["J_regressor"]
        if not isinstance(Jr, np.ndarray) or Jr.dtype == object:
            Jr = Jr.item()
        if hasattr(Jr, "toarray"): Jr = Jr.toarray()
        Jr = np.asarray(Jr, dtype=np.float64)
        _MODEL[gender] = (Jr, np.asarray(m["v_template"], np.float64), np.asarray(m["shapedirs"], np.float64))
    return _MODEL[gender]

def rest_joints(gender, betas):         # betas-aware rest joints [22,3]
    Jr, vt, sd = load_model(gender)
    if betas is not None:
        n = min(sd.shape[2], betas.shape[0])
        vt = vt + np.einsum("vcn,n->vc", sd[:,:,:n], betas[:n])
    return (Jr @ vt)[:22]

def fk_positions(poses66, gender, betas):
    T = poses66.shape[0]; restJ = rest_joints(gender, betas)
    aa = poses66.reshape(T,22,3)
    R = np.stack([rodrigues(aa[:,j,:]) for j in range(22)],axis=1)   # [T,22,3,3]
    Gr = np.zeros((T,22,3,3)); Gp = np.zeros((T,22,3))
    Gr[:,0] = R[:,0]; Gp[:,0] = restJ[0]
    for j in range(1,22):
        pj = PARENTS[j]
        Gr[:,j] = Gr[:,pj] @ R[:,j]
        Gp[:,j] = Gp[:,pj] + np.einsum("tij,j->ti", Gr[:,pj], restJ[j]-restJ[pj])
    return Gp

def internal_angles(pos):               # pos [T,22,3] -> [T,4] deg
    out = []
    for _,(j,pa,ch) in HINGES.items():
        v1 = pos[:,j]-pos[:,pa]; v2 = pos[:,ch]-pos[:,j]
        v1n = v1/np.clip(np.linalg.norm(v1,axis=1,keepdims=True),1e-8,None)
        v2n = v2/np.clip(np.linalg.norm(v2,axis=1,keepdims=True),1e-8,None)
        out.append(np.degrees(np.arccos(np.clip((v1n*v2n).sum(1),-1,1))))
    return np.stack(out,axis=1)

def procrustes_mpjpe_mm(A, B):          # per-frame similarity align B->A; A,B [T,22,3] (meters) -> mm
    T = min(len(A),len(B)); res=[]
    for t in range(T):
        X, Y = A[t], B[t]
        Xc, Yc = X - X.mean(0), Y - Y.mean(0)
        U,S,Vt = np.linalg.svd(Yc.T @ Xc)
        d = np.sign(np.linalg.det(U@Vt)); D=np.diag([1,1,d])
        Rm = U@D@Vt
        scale = (S*[1,1,d]).sum() / max((Yc**2).sum(), 1e-12)
        Yt = scale*(Yc@Rm) + X.mean(0)
        res.append(np.linalg.norm(Yt-X,axis=1).mean())
    return float(np.mean(res))*1000.0

# ---------------------------------------------------------------- cross-correlation lag
def best_lag(a, b, maxlag=10):
    a = a-a.mean(); b = b-b.mean()
    n = min(len(a),len(b)); a=a[:n]; b=b[:n]
    if n < 8 or a.std()<1e-6 or b.std()<1e-6: return None, 0.0
    best=(0,-2.0)
    for L in range(-maxlag,maxlag+1):
        if L>=0: x,y = a[L:], b[:n-L]
        else:    x,y = a[:n+L], b[-L:]
        if len(x)<8: continue
        c = np.corrcoef(x,y)[0,1]
        if np.isfinite(c) and c>best[1]: best=(L,c)
    return best[0], best[1]

# ---------------------------------------------------------------- id remap (caption -> key)
def norm(t): return " ".join(str(t).strip().lower().split())
def build_caption_index(ann):
    idx = {}
    for k,v in ann.items():
        for a in v.get("annotations",[]):
            idx.setdefault(norm(a.get("text","")), set()).add(k)
    return idx
def load_index_csv():
    rows = {}
    with open(INDEX) as f:
        for r in csv.DictReader(f):
            rows[os.path.splitext(r["new_name"])[0]] = r
    return rows
def resolve(our_id, sidecar, cap_idx, idx_rows, ann):
    fn = f"HML3D_Human_{our_id}.npy"; meta = sidecar.get(fn)
    if meta is None: return None, f"no sidecar entry {fn}"
    cap = norm(meta.get("primary_caption") or (meta.get("captions") or [""])[0])
    keys = {k for k in cap_idx.get(cap, set()) if k in idx_rows}   # drop mirror 'M' keys (not in index.csv)
    if len(keys)==0: return None, f"no index-caption match for '{cap[:48]}'"
    if len(keys)>1:  return None, f"ambiguous caption ({len(keys)} keys): {sorted(keys)[:5]}"
    key = next(iter(keys)); row = idx_rows[key]
    ann_path = ann[key].get("path","")
    src = row["source_path"]; src_rel = src.replace("./pose_data/","").replace("_poses.npy","_poses")
    if src_rel != ann_path:
        return None, f"path mismatch: index '{src_rel}' vs ann '{ann_path}'"   # enforce (Codex M4)
    npz = f"{AMASS_MD}/" + src.replace("./pose_data/","").replace("_poses.npy","_poses.npz")
    if not os.path.exists(npz):
        alt = npz.replace("_poses.npz","_stageii.npz"); npz = alt if os.path.exists(alt) else npz
    if not os.path.exists(npz): return None, f"npz not found: {npz}"
    return dict(key=key, npz=npz, start=int(row["start_frame"]), end=int(row["end_frame"]),
                src_rel=src_rel), None

# ---------------------------------------------------------------- main
def main():
    ann = json.load(open(ANN)); sidecar = json.load(open(SIDECAR))
    cap_idx = build_caption_index(ann); idx_rows = load_index_csv()
    print(f"loaded: annotations={len(ann)} sidecar={len(sidecar)} index_rows={len(idx_rows)} caption_keys={len(cap_idx)}\n")

    r10, err10 = resolve("010000", sidecar, cap_idx, idx_rows, ann)
    if r10: print(f"[remap-check] our 010000 -> amass key {r10['key']} (identity-break {'CONFIRMED' if r10['key']!='010000' else 'NOT triggered'}); src={r10['src_rel']}")
    else:   print(f"[remap-check] 010000 -> {err10}")
    print()

    def find_motion(oid):
        for sub in ("motions","motions_heldout"):
            p = f"{V3A}/{sub}/HML3D_Human_{oid}.npy"
            if os.path.exists(p): return p
        return None

    rows = []; unresolved = []
    for oid in TEST_CLIPS:
        info, err = resolve(oid, sidecar, cap_idx, idx_rows, ann)
        if err: print(f"[{oid}] REMAP FAIL: {err}"); unresolved.append(oid); continue
        mp = find_motion(oid)
        if mp is None: print(f"[{oid}] stored motion not found"); unresolved.append(oid); continue
        clip = np.load(mp).astype(np.float64)
        d = np.load(info["npz"], allow_pickle=True)
        poses = d["poses"][:, :66].astype(np.float64)
        betas = np.asarray(d["betas"], np.float64) if "betas" in d else None
        fps = float(d["mocap_framerate"]); ds_f = fps/20.0
        assert abs(ds_f-round(ds_f))<1e-6, f"{oid}: fps {fps} not a multiple of 20"
        ds = int(round(ds_f)); s,e = info["start"], info["end"]
        g = str(d["gender"]) if "gender" in d else "neutral"
        g = g if g in ("male","female","neutral") else "neutral"
        Tstore = clip.shape[0]
        dv = internal_angles(clip[:,:, :3]); pv = clip[:,:, :3]   # v3a angles + positions

        res = {}
        for conv, seg in (("A_ds_then_slice", poses[::ds][s:e]),
                          ("B_slice_then_ds", poses[s:e][::ds])):
            len_ok = seg.shape[0] in (Tstore, Tstore+1)           # Codex H2: gate length first
            if not len_ok or seg.shape[0] < 8:
                res[conv] = dict(len=seg.shape[0], len_ok=False); continue
            pa = fk_positions(seg, g, betas); da = internal_angles(pa)
            n = min(len(da), Tstore)
            lags=[]; corrs=[]
            for c in range(4):
                L,cc = best_lag(da[:n,c], dv[:n,c])
                if L is not None: lags.append(L); corrs.append(cc)
            mm = procrustes_mpjpe_mm(pa[:n, 1:], pv[:n, 1:])   # exclude root j0 (ch0:3 not pure XYZ for root)
            res[conv] = dict(len=seg.shape[0], len_ok=True,
                             medlag=float(np.median(lags)) if lags else None,
                             meancorr=float(np.mean(corrs)) if corrs else None,
                             mm=mm, perlag=lags, percorr=[round(x,3) for x in corrs])
        tag = "ZERO-start(control)" if oid in ZERO_START else "NONZERO-start"
        print(f"[{oid}] {tag} src={info['src_rel']} fps={fps:.0f} ds={ds} s:e={s}:{e} Tstore={Tstore}")
        for conv in ("A_ds_then_slice","B_slice_then_ds"):
            v = res[conv]
            if v.get("len_ok"):
                print(f"     {conv}: len={v['len']} medlag={v['medlag']:+.0f} meancorr={v['meancorr']:.3f} procrustes_mpjpe={v['mm']:.1f}mm  lags={v['perlag']} corr={v['percorr']}")
            else:
                print(f"     {conv}: len={v['len']} LEN_MISMATCH(Tstore={Tstore}) -> not scored")
        rows.append((oid, oid not in ZERO_START, res)); print()

    # ---- verdict from NON-ZERO-start clips ----
    def passes(v):
        return bool(v.get("len_ok") and v.get("medlag") is not None and abs(v["medlag"])<=LAG_TOL
                    and v["meancorr"] is not None and v["meancorr"]>CORR_THRESH and v["mm"] is not None and v["mm"]<MM_THRESH)
    n_nz = sum(1 for _,nz,_ in rows if nz)
    planned_nz = sum(1 for o in TEST_CLIPS if o not in ZERO_START)
    A_ok = sum(1 for _,nz,r in rows if nz and passes(r["A_ds_then_slice"]))
    B_ok = sum(1 for _,nz,r in rows if nz and passes(r["B_slice_then_ds"]))
    print("="*72)
    print(f"resolved nonzero-start clips: {n_nz}/{planned_nz}  |  A passes {A_ok}/{n_nz}  |  B passes {B_ok}/{n_nz}  (thresh: |lag|<={LAG_TOL}, corr>{CORR_THRESH}, mpjpe<{MM_THRESH}mm)")
    if unresolved:
        print(f"VERDICT: INCONCLUSIVE — unresolved planned clips {unresolved} (fix remap before trusting any verdict)."); return 2
    if n_nz>0 and A_ok==n_nz and B_ok==0:
        print("VERDICT: convention A (downsample-then-slice, 20fps-indexed start/end) CORRECT — A passes all, B none."); return 0
    if n_nz>0 and B_ok==n_nz and A_ok==0:
        print("VERDICT: convention B (slice-then-downsample, raw-fps-indexed) CORRECT — B passes all, A none."); return 0
    print("VERDICT: INCONCLUSIVE — neither convention cleanly wins; inspect rows (do NOT proceed to conversion)."); return 2

if __name__ == "__main__":
    import sys; sys.exit(main())
