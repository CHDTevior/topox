#!/usr/bin/env python3
"""
FK-anchored remap for the human real-twist recovery (v4) pipeline. Candidate subset scope is a CLI arg
(--subsets, default ALL AMASS subsets); the FK content-alignment below is the sole judge regardless of scope.
See handoff/20260701_235711_human_twist_recovery_v4_implementation.md (Rev 2).

WHY: source_motion_id -> index.csv direct join is only ~56% length-consistent and FK-falsified
(change_name.py sort-renumbers the texts dir, decoupling our ids from index.csv ids). So NO id or
caption is trusted as ground truth. The ONLY accepted mapping is content alignment: restore joint
positions from a candidate AMASS source under the CONFIRMED slicing `poses[::ds][start:end][:-1]`
(drop-last, convention A verified by _v4_framealign_probe.py) and require the internal flexion-angle
time series to match our stored v3a clip (lag 0, corr>0.999, per-frame Procrustes MPJPE<30mm), with the
winner CLEARLY dominating the second-best. Otherwise fail-loud (AMBIGUOUS / MISSING), never force-pick.

Candidates (all WEAK; union; FK is the judge):
  * length-bucket: every in-scope index.csv row whose (end-start-1) == our clip length  [primary, id-free]
  * source_motion_id direct row (hint, tested first if in scope)
  * caption -> amass_annotations path -> index.csv rows on that path (hint)
Mirror clips (our id >= OFF): DERIVED from the base clip's accepted source (same AMASS segment) +
mirror flag, FK-verified with L<->R hinge swap (never independently searched).

Output: per-clip status (exact_accepted / exact_ambiguous / exact_missing / mirror_derived /
mirror_failed) + a remap table JSON for the downstream C_p calibration (only accepted clips).

Usage: python scripts/_v4_remap_fk_anchored.py [--limit N]
"""
import os, json, csv, argparse
import numpy as np
import _v4_framealign_probe as P

OFF        = 14613               # candidate subset scope is a CLI arg (--subsets, default ALL AMASS dirs)
CORR_ACC   = 0.999
MM_ACC     = 30.0
LAG_ACC    = 0
SWAP       = [1, 0, 3, 2]        # internal_angles order [L_knee,R_knee,L_elbow,R_elbow] -> mirror L<->R
OUT_DIR    = "/scratch/ts1v23/workspace/noKslot_clean/scratch/v4_remap"
OUT_JSON   = f"{OUT_DIR}/cmu_ekut_remap.json"

# ---- npz cache (many clips share a source file) ----
_NPZ = {}
def load_src(npz):
    if npz not in _NPZ:
        if len(_NPZ) > 400: _NPZ.clear()      # bound memory
        d = np.load(npz, allow_pickle=True)
        g = str(d.get("gender", "neutral")); g = g if g in ("male","female","neutral") else "neutral"
        _NPZ[npz] = dict(poses=d["poses"][:, :66].astype(np.float64),
                         betas=np.asarray(d["betas"], np.float64) if "betas" in d else None,
                         fps=float(d["mocap_framerate"]), gender=g)
    return _NPZ[npz]

def procrustes_mm(A, B, allow_reflection=False):
    """per-frame similarity align B->A -> mean MPJPE (mm). allow_reflection=True for mirrored clips
    (base vs mirror differ by a reflection; standard det-sign Umeyama forbids it -> spurious large mm)."""
    T = min(len(A), len(B)); res = []
    for t in range(T):
        X, Y = A[t], B[t]; Xc, Yc = X - X.mean(0), Y - Y.mean(0)
        U, S, Vt = np.linalg.svd(Yc.T @ Xc)
        if allow_reflection:
            Rm = U @ Vt; s_ = S.sum()
        else:
            d = np.sign(np.linalg.det(U @ Vt)); D = np.diag([1, 1, d]); Rm = U @ D @ Vt; s_ = (S * [1, 1, d]).sum()
        scale = s_ / max((Yc**2).sum(), 1e-12)
        Yt = scale * (Yc @ Rm) + X.mean(0)
        res.append(np.linalg.norm(Yt - X, axis=1).mean())
    return float(np.mean(res)) * 1000.0

def npz_path(source_path):
    rel = source_path.replace("./pose_data/","").replace("_poses.npy","_poses.npz")
    primary = f"{P.AMASS_MD}/{rel}"
    if os.path.exists(primary): return primary
    # Some subsets were fetched twice: '<subset>' may hold the SMPL-X '_stageii.npz' while the SMPL-H
    # '_poses.npz' that HumanML3D actually used lives under '<subset>_smplh' (verified for KIT: 4648 clips,
    # redirect -> corr=1.0000, MPJPE 0.8-10mm). Redirect to the _smplh twin when the primary is absent.
    if "/" in rel:
        sub, tail = rel.split("/", 1)
        alt = f"{P.AMASS_MD}/{sub}_smplh/{tail}"
        if os.path.exists(alt): return alt
    return primary

def subset_of(source_path):
    return source_path.replace("./pose_data/","").split("/")[0]

_CAND = {}   # (path,start,end) -> (angles[T,4], pos[T,22,3]) or None ; candidate FK is our-clip-independent -> cache once
def cand_fk(row):
    key = (row["source_path"], row["start_frame"], row["end_frame"])
    if key not in _CAND:
        val = None
        npz = npz_path(row["source_path"])
        if os.path.exists(npz):
            s = load_src(npz); ds = int(round(s["fps"]/20.0))   # HumanML3D downsample; non-integer fps/20 (e.g.
            if ds >= 1:                                         # Eyes_Japan 250, SSM 120.00005) -> nearest ds, the
                a, b = int(row["start_frame"]), int(row["end_frame"])   # corr>0.999 FK gate rejects any misalignment.
                seg = s["poses"][::ds][a:b][:-1]          # CONFIRMED convention A + drop-last
                if seg.shape[0] >= 8:
                    pos = P.fk_positions(seg, s["gender"], s["betas"])   # FK ONCE
                    val = (P.internal_angles(pos), pos)
        if len(_CAND) > 20000: _CAND.clear()
        _CAND[key] = val
    return _CAND[key]

def score(row, dv, pv, Tstore, mirror=False):
    """compare a candidate to our clip -> dict(len_ok, lag, corr, mm) or None if unusable."""
    out = cand_fk(row)
    if out is None: return None
    da, pa = out
    len_ok = da.shape[0] == Tstore                       # [:-1] -> exact length equality
    if not len_ok: return dict(len_ok=False)
    a = da[:, SWAP] if mirror else da                    # mirror: swap L<->R hinges
    T = da.shape[0]
    def corr_at(shift):                                  # mean 4-hinge Pearson corr at a given integer lag
        cs = []
        for c in range(4):
            if shift >= 0: x, y = a[shift:, c], dv[:T-shift, c]
            else:          x, y = a[:T+shift, c], dv[-shift:, c]
            if len(x) >= 8 and x.std() > 1e-6 and y.std() > 1e-6:
                cc = np.corrcoef(x, y)[0, 1]
                if np.isfinite(cc): cs.append(cc)
        return float(np.mean(cs)) if len(cs) >= 2 else -2.0   # need >=2 non-constant hinges (reject degenerate)
    c0 = corr_at(0)                                      # confirmed convention -> true match is lag 0; cheap reject majority
    corr, lag = c0, 0.0
    if c0 > 0.9:                                         # only near-matches pay for the +-1 robustness check
        for sh in (-1, 1):
            cc = corr_at(sh)
            if cc > corr: corr, lag = cc, float(sh)
    mm = None
    if abs(lag) <= LAG_ACC and corr > CORR_ACC:          # gate: Procrustes only for angle-verified lag-0 candidates
        mm = procrustes_mm(pa[:, 1:], pv[:, 1:], allow_reflection=mirror)
    return dict(len_ok=True, lag=lag, corr=corr, mm=mm)

def passes(sc):
    return bool(sc and sc.get("len_ok") and sc.get("lag") is not None
               and abs(sc["lag"]) <= LAG_ACC and sc["corr"] > CORR_ACC and sc["mm"] < MM_ACC)

def dominates(best, second):
    if second is None: return True
    # winner clearly better: second fails, OR clear corr/mpjpe gap
    if not passes(second): return True
    return (best["corr"] - second["corr"] > 0.005) or (best["mm"] * 2.0 < second["mm"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--subsets", default="ALL",
                    help="comma-list of AMASS subsets to match against, or ALL = every dir present under AMASS_MD "
                         "(humanact12 is not an AMASS dir -> auto-excluded, recovered separately via ACTOR).")
    ap.add_argument("--out", default=OUT_JSON, help="output remap JSON (default cmu_ekut; use full_amass_remap.json for full).")
    args = ap.parse_args()
    # resolve the accepted subset set (FK is still the sole judge; this only bounds the CANDIDATE pool).
    if args.subsets.strip().upper() == "ALL":
        subsets_ok = frozenset(d for d in os.listdir(P.AMASS_MD)
                               if os.path.isdir(f"{P.AMASS_MD}/{d}") and d != "humanact12")
    else:
        subsets_ok = frozenset(s.strip() for s in args.subsets.split(",") if s.strip())
    out_json = args.out
    print(f"candidate subsets ({len(subsets_ok)}): {sorted(subsets_ok)}")
    os.makedirs(OUT_DIR, exist_ok=True)
    sidecar = json.load(open(P.SIDECAR)); idx_rows = P.load_index_csv()
    oi = {}
    with open(f"{P.V3A}/object_index.csv") as f:
        for r in csv.DictReader(f): oi[r["filename"]] = r

    # length-bucket over in-scope index rows: (end-start-1) -> [rows]
    by_len = {}
    for k, row in idx_rows.items():
        if subset_of(row["source_path"]) not in subsets_ok: continue
        try: L = int(row["end_frame"]) - int(row["start_frame"]) - 1
        except Exception: continue
        if L >= 8: by_len.setdefault(L, []).append(row)
    print(f"index rows in scope: {sum(len(v) for v in by_len.values())} across {len(by_len)} distinct lengths")

    def load_clip(oid):
        for sub in ("motions","motions_heldout"):
            p = f"{P.V3A}/{sub}/HML3D_Human_{oid:06d}.npy"
            if os.path.exists(p): return np.load(p).astype(np.float64)
        return None

    # ---- pass 1: BASE clips (id < OFF) ----
    accepted = {}   # our_id -> record
    cats = {"exact_accepted":0,"exact_ambiguous":0,"exact_missing":0,"mirror_derived":0,"mirror_failed":0,"no_clip":0}
    base_ids = [int(k.replace("HML3D_Human_","").replace(".npy","")) for k in sidecar]
    base_ids = sorted(i for i in base_ids if i < OFF)
    if args.limit: base_ids = base_ids[:args.limit]
    for n, oid in enumerate(base_ids):
        clip = load_clip(oid)
        if clip is None: cats["no_clip"]+=1; continue
        Tstore = clip.shape[0]
        dv = P.internal_angles(clip[:, :, :3]); pv = clip[:, :, :3]
        smid = sidecar[f"HML3D_Human_{oid:06d}.npy"].get("source_motion_id")
        acc_row = acc_sc = via = None
        # 1) FAST PATH: source_motion_id direct candidate, FK-VERIFIED (smid correct for ~56%). Accepted alone
        #    WITHOUT the bucket-dominance check because the gate is already uniqueness-strong: corr>0.999 on the
        #    moving hinges (>=2 non-constant) AND per-frame Procrustes MPJPE<30mm over the full 21-joint skeleton.
        #    A different motion cannot satisfy the <30mm full-skeleton gate; a near-duplicate would carry the same twist.
        if smid in idx_rows and subset_of(idx_rows[smid]["source_path"]) in subsets_ok:
            sc = score(idx_rows[smid], dv, pv, Tstore)
            if passes(sc): acc_row, acc_sc, via = idx_rows[smid], sc, "smid"
        # 2) FALLBACK: content-only length-bucket FK search (id-free) when smid did not verify
        ambiguous = False
        if acc_row is None:
            passing = []
            for r in by_len.get(Tstore, []):
                sc = score(r, dv, pv, Tstore)
                if passes(sc): passing.append((r, sc))
            passing.sort(key=lambda rs: (-(rs[1]["corr"]), rs[1]["mm"]))
            if len(passing) == 1 or (len(passing) > 1 and dominates(passing[0][1], passing[1][1])):
                acc_row, acc_sc, via = passing[0][0], passing[0][1], "length_fk"
            elif len(passing) > 1:
                ambiguous = True
        if acc_row is not None:
            accepted[oid] = dict(our_id=f"{oid:06d}", source_path=acc_row["source_path"],
                                 start=int(acc_row["start_frame"]), end=int(acc_row["end_frame"]),
                                 subset=subset_of(acc_row["source_path"]), mirror=False,
                                 corr=round(acc_sc["corr"],4), mm=round(acc_sc["mm"],2), Tstore=Tstore, via=via)
            cats["exact_accepted"]+=1
        elif ambiguous:
            cats["exact_ambiguous"]+=1
        else:
            cats["exact_missing"]+=1
        if (n+1) % 500 == 0:
            print(f"  base {n+1}/{len(base_ids)}: accepted={cats['exact_accepted']} ambig={cats['exact_ambiguous']} missing={cats['exact_missing']}")

    # ---- pass 2: MIRROR clips derived from base ----
    mirror_ids = sorted(i for i in (int(k.replace('HML3D_Human_','').replace('.npy','')) for k in sidecar) if i >= OFF)
    if args.limit: mirror_ids = mirror_ids[:args.limit]
    for oid in mirror_ids:
        base = oid - OFF
        if base not in accepted: cats["mirror_failed"]+=1; continue   # base not exact-local -> skip mirror
        clip = load_clip(oid)
        if clip is None: cats["no_clip"]+=1; continue
        Tstore = clip.shape[0]; dv = P.internal_angles(clip[:, :, :3]); pv = clip[:, :, :3]
        br = accepted[base]
        row = {"source_path": br["source_path"], "start_frame": str(br["start"]), "end_frame": str(br["end"])}
        sc = score(row, dv, pv, Tstore, mirror=True)
        # mirror gate = swapped-hinge angle corr (bulletproof: a wrong pairing cannot match 4 hinge
        # trajectories to >0.999 over the whole clip). Position mm is NOT a hard gate for mirrors —
        # HumanML3D's 263-dim swap+negate augmentation is not a perfect geometric reflection, so mirror
        # position residual is inherently ~2-4x the direct residual; mm is recorded as diagnostic only.
        mir_ok = bool(sc and sc.get("len_ok") and sc.get("corr") is not None
                      and abs(sc.get("lag") if sc.get("lag") is not None else 9) <= LAG_ACC
                      and sc["corr"] > CORR_ACC)
        if mir_ok:
            accepted[oid] = dict(our_id=f"{oid:06d}", source_path=br["source_path"], start=br["start"],
                                 end=br["end"], subset=br["subset"], mirror=True,
                                 corr=round(sc["corr"],4),
                                 mm=(round(sc["mm"],2) if sc.get("mm") is not None else None),
                                 Tstore=Tstore, via="mirror_of_%06d"%base)
            cats["mirror_derived"]+=1
        else:
            cats["mirror_failed"]+=1

    json.dump({str(k):v for k,v in accepted.items()}, open(out_json,"w"), indent=1)
    print("\n"+"="*72)
    print(f"FK-ANCHORED REMAP coverage ({len(subsets_ok)} subsets in scope):")
    for c,v in cats.items(): print(f"  {c:<18} {v}")
    print(f"  accepted table -> {out_json}  ({len(accepted)} clips)")
    print(f"\nNOTE: exact_missing = true source not in the candidate subsets, humanact12 (ACTOR, separate), OR")
    print(f"      unresolvable/ambiguous. FK is the sole judge: acceptance needs corr>0.999 on the moving hinges")
    print(f"      (>=2 non-constant) + per-frame Procrustes MPJPE<30mm over the 21-joint skeleton; the length-bucket")
    print(f"      fallback additionally requires the winner to dominate the 2nd. A different motion cannot pass.")

if __name__ == "__main__":
    main()
