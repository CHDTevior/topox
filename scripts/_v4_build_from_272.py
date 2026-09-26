#!/usr/bin/env python3
"""
Build the v4-B AnyTop HUMAN dataset from MotionStreamer-272's shipped motion_data (HF lxxiao/272-dim-HumanML3D).

WHY (user 2026-07-02): 272 ships the REAL SMPL rotations for every HumanML3D clip (incl Eyes_Japan/MPI our AMASS
FK-remap couldn't source). We build the human AnyTop dataset PURELY from 272's motion_data — NOT via our v3a clips:
the v3a numbering only matches 272 at LOW oid and DIVERGES at high oid (internal-angle corr drops to ~0 by oid 10000,
verified; v3a renumbers humanact12 contiguously while 272 has AMASS-only gaps). So prep_272_full is 272-ONLY.
Verified (prep_272_full): motion corr 0.98-0.999 vs 272 original across LOW+HIGH oids; ANATOMICAL twist diff 0.000deg
(faithful, user-confirmed incl fast 000004); no jitter; self-consistent selfcon~0.

PIPELINE (prep_272_full, per clip): 272-dim -> decode to GLOBAL joint positions (recover_from_local_position) +
world rotations (g272_world). Resample 30fps -> 20fps (positions interp, rotations slerp). extract_features(global
pos) -> HumanML3D 263 (correct root channels + RIC + vel + contact) -> convert_263_to_13 -> clip13. A = Kabsch(272
body -> clip13 frame). build_clip({}, d, offs): ch3:9 = 272 rotations (per-parent, real twist), ch0:3 = FK(ch3:9,
offs) self-consistent, ch9:12 re-derived, root+contact from clip13. NO v3a dependency.

COVERAGE: 272 = 13423 base + 13423 mirror = 26846 clips (all HumanML3D minus humanact12, which has no AMASS).
Keyed + captioned by the 272 name (base '000006', mirror 'M000006'; caption = texts/<name>.txt, motion-consistent).
humanact12 (~1190, absent in 272) -> separate ACTOR recovery later. Mirrors build natively (no v3a counterpart).

TWO REST VARIANTS (user wants both):
  --rest neutral : OFF_S = SMPL-neutral (single fixed human skeleton, like the animals' per-species fixed skeleton);
                   ch0:3 = FK(ch3:9, neutral) self-consistent; FK-consistent with a single cond.offsets.
  --rest perbetas: per-clip offsets = SMPL-neutral bone DIRECTIONS x each clip's real bone LENGTHS (measured from
                   272's positions) -> ch0:3 = FK(ch3:9, per-clip) self-consistent per clip, matching 272's real
                   proportions; cond.offsets = MEAN of per-clip offsets (representative; FK-loss has small slack).

Usage: python scripts/_v4_build_from_272.py --test                 (agreement vs AMASS on matched clips)
       python scripts/_v4_build_from_272.py --write --rest neutral  --out DIR
       python scripts/_v4_build_from_272.py --write --rest perbetas --out DIR
"""
import os, json, argparse, shutil
import numpy as np
from scipy.spatial.transform import Rotation as Rt, Slerp
import _v4_cp_calibration as C
import _v4_build_dataset as B
conv=C.conv; J=C.J; PARENTS=C.PARENTS; CHILDREN=C.CHILDREN; SINGLE=C.SINGLE
from src.data.anytop_rot6d_fk import recover_from_bvh_rot_np
_norm=C._norm
REPO="/scratch/ts1v23/workspace/noKslot_clean"
H272=f"{REPO}/scratch/humanml3d_272"; M272=f"{H272}/motion_data"; T272=f"{H272}/texts"
V3A=C.V3A; OFFN=14613; OFF=conv.compute_offsets()
OFF_S=B.OFF_S                                    # SMPL-neutral rest (shared with the builder)
_UNIT_S={j:_norm(OFF_S[j][None])[0] for j in range(1,J)}   # neutral rest bone DIRECTIONS

def load_272(name):
    p=f"{M272}/{name}.npy"; return np.load(p) if os.path.exists(p) else None
def decode_6d_rows(d6):
    a=d6[...,:3]; b=d6[...,3:]
    r0=a/(np.linalg.norm(a,axis=-1,keepdims=True)+1e-12)
    b=b-(r0*b).sum(-1,keepdims=True)*r0
    r1=b/(np.linalg.norm(b,axis=-1,keepdims=True)+1e-12)
    return np.stack([r0,r1,np.cross(r0,r1)],axis=-2)
def g272_world(x272):
    T=x272.shape[0]; rl=decode_6d_rows(x272[:,140:140+132].reshape(T,J,6))
    G=np.zeros((T,J,3,3)); G[:,0]=rl[:,0]
    for j in range(1,J): G[:,j]=G[:,PARENTS[j]]@rl[:,j]
    return G
def resample_rot(G, Tt):
    T=G.shape[0]
    if T==Tt: return G
    src=np.arange(T); tgt=np.linspace(0,T-1,Tt); out=np.zeros((Tt,J,3,3))
    for j in range(J): out[:,j]=Slerp(src,Rt.from_matrix(G[:,j]))(tgt).as_matrix()
    return out
def perbetas_offsets(x272):
    """per-clip rest offsets = neutral bone DIRECTIONS x this clip's real bone LENGTHS (from 272 positions,
    constant over frames -> median). Matches 272's proportions while keeping the SMPL rest directions."""
    pos=x272[:,8:8+66].reshape(len(x272),J,3)
    off=np.zeros((J,3))
    for j in range(1,J):
        L=np.median(np.linalg.norm(pos[:,j]-pos[:,PARENTS[j]],axis=-1))
        off[j]=_UNIT_S[j]*L
    return off

def prep_272(v3a_oid, m272_name, offs):
    """prep()-compatible dict; SMPL rotations from 272 (resampled to the v3a clip length), FK rest = offs."""
    clip=C.load_clip(v3a_oid); x=load_272(m272_name)
    if clip is None or x is None or clip.shape[0]<3: return None
    G=resample_rot(g272_world(x), clip.shape[0])
    Pw=recover_from_bvh_rot_np(clip.astype(np.float32),PARENTS,OFF).astype(np.float64)
    Psm=np.zeros((clip.shape[0],J,3))
    for j in range(1,J): Psm[:,j]=Psm[:,PARENTS[j]]+np.einsum("tij,j->ti",G[:,PARENTS[j]],offs[j])
    U=C.bone_dirs(Psm); V=C.bone_dirs(Pw); Uu,_,Vt=np.linalg.svd(U.T@V)
    A=Vt.T@np.diag([1,1,np.sign(np.linalg.det(Vt.T@Uu.T))])@Uu.T
    Gtrue=np.einsum("ij,tpjk->tpik",A,G)
    return dict(clip=clip,Gtrue=Gtrue,G=G,A=A,T=clip.shape[0],WR=None,Rloc=None,off_smpl=None,dcur=None)

# ---- clip list: every 272 clip (base + mirror). caption = 272 texts/<name>.txt (consistent with the motion). ----
def clip_list():
    return sorted(f[:-4] for f in os.listdir(M272) if f.endswith(".npy"))

def read_caption(nm):
    """AnyTopDataset caption schema (dict; a list value is SKIPPED by the loader): primary_caption/captions/
    source_dataset/source_motion_id."""
    p=f"{T272}/{nm}.txt"
    if not os.path.exists(p): return None
    caps=[ln.split("#")[0].strip() for ln in open(p) if ln.strip()]
    if not caps: return None
    return {"primary_caption": caps[0], "captions": caps, "source_dataset": "HumanML3D", "source_motion_id": nm}

def build_all(rest):
    """FULL 272-ONLY re-derive (v3a numbering DIVERGES from 272 at high oid -> cannot use v3a; build purely from
    272's shipped motion_data). Every 272 clip (base+mirror) -> AnyTop, keyed by its 272 name."""
    names=clip_list(); toks={}; caps={}; perb_offs=[]; fails=[]
    for k,nm in enumerate(names):
        try:                                               # defensive: any per-clip exception -> record fail (hard-fail
            x=load_272(nm)                                 # aborts the write later), never crash the whole run mid-way
            offs = OFF_S if rest=="neutral" else perbetas_offsets(x)
            d=prep_272_full(nm)                            # 272-only (uses OFF_S for the A-align; version-agnostic)
            if d is None: fails.append((nm,"prep_none")); continue
            tok,phi,g=B.build_clip({}, d=d, offs=offs)
        except Exception as e:
            fails.append((nm, f"{type(e).__name__}: {str(e)[:70]}")); continue
        if tok is None: fails.append((nm,phi)); continue
        if g["selfcon_mm"]>1.0 or g["rotval_max"]>1e-3: fails.append((nm,f"gate selfcon={g['selfcon_mm']:.2f} rotval={g['rotval_max']:.1e}")); continue
        toks[nm]=tok; caps[nm]=read_caption(nm)
        if rest=="perbetas": perb_offs.append(offs)
        if (k+1)%2000==0: print(f"  {k+1}/{len(names)} ok={len(toks)} fail={len(fails)}")
    cond_offsets = OFF_S if rest=="neutral" else np.mean(perb_offs,axis=0)
    return toks, caps, cond_offsets, fails

def regen_cond(toks, cond_offsets):
    """cond['HML3D_Human']: copy topology/graph fields from v3a cond; regen offsets, tpos_first_frame, mean, std."""
    base=np.load(f"{V3A}/cond.npy",allow_pickle=True).item()["HML3D_Human"]
    allv=np.concatenate([t for t in toks.values()],axis=0)          # [sumT,J,13]
    mean=allv.mean(0); std=allv.std(0); std[std<1e-6]=1.0
    # tpos_first_frame: rest skeleton in 13ch (identity rot, rest positions from cond_offsets)
    restpos=np.zeros((J,3))
    for j in range(1,J): restpos[j]=restpos[PARENTS[j]]+cond_offsets[j]
    tpos=np.zeros((J,13)); tpos[:,0:3]=restpos; tpos[:,3]=1; tpos[:,7]=1   # identity 6D=[1,0,0,0,1,0]->cols? keep like v3a
    tpos=base["tpos_first_frame"].copy(); tpos[:,0:3]=restpos            # reuse v3a tpos structure, update positions
    return dict(parents=base["parents"], offsets=cond_offsets.astype(np.float32),
                tpos_first_frame=tpos.astype(np.float32), joints_names=base["joints_names"],
                kinematic_chains=base["kinematic_chains"], joint_relations=base["joint_relations"],
                joints_graph_dist=base["joints_graph_dist"], mean=mean.astype(np.float32),
                std=std.astype(np.float32), object_type="HML3D_Human")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--test",action="store_true"); ap.add_argument("--n",type=int,default=6)
    ap.add_argument("--write",action="store_true"); ap.add_argument("--rest",choices=["neutral","perbetas"],default="neutral")
    ap.add_argument("--out",default=None); a=ap.parse_args()
    if a.test:
        remap=json.load(open(C.REMAP)); recs=[v for v in remap.values() if not v["mirror"]][:a.n]
        for rec in recs:
            oid=rec["our_id"]; tA,_,gA=B.build_clip(rec); d=prep_272(oid,oid,OFF_S)
            if d is None: print(f"{oid} 272 missing"); continue
            t2,_,g2=B.build_clip({}, d=d)
            n=min(len(tA),len(t2)) if tA is not None and t2 is not None else 0
            if n: print(f"{oid} ch3:9diff={np.abs(tA[:n,:,3:9]-t2[:n,:,3:9]).mean():.4f} 272selfcon={g2['selfcon_mm']:.3f} twist={g2['twist_absmean']:.0f}")
        return
    if not a.write:
        # dry-run: 272-only build a spread of base+mirror + low/high oids, confirm no missing
        cl=clip_list(); print(f"272 clips: {len(cl)} (base+mirror). 272-only sample build (rest={a.rest}):")
        samp=cl[:3]+[cl[len(cl)//2]]+cl[-3:]+[c for c in cl if c in ("010000","013000","M000006")]
        for nm in samp:
            x=load_272(nm); offs=OFF_S if a.rest=="neutral" else perbetas_offsets(x)
            d=prep_272_full(nm); tok,phi,g=B.build_clip({},d=d,offs=offs) if d else (None,None,None)
            cap=read_caption(nm)
            print(f"  {nm}: {'OK selfcon=%.3f twist=%.0f cap=%r'%(g['selfcon_mm'],g['twist_absmean'],(cap['primary_caption'][:32] if cap else None)) if tok is not None else 'FAIL '+str(phi)}")
        return
    out=a.out or f"{REPO}/data/humanml3d_anytop13_v4b_272_{a.rest}"
    if os.path.exists(out): print(f"ERROR: {out} exists; remove or pick fresh --out."); return
    print(f"v4-B 272 build rest={a.rest} -> {out}\n")
    toks,caps,cond_off,fails=build_all(a.rest)
    print(f"\nbuilt {len(toks)} clips ({len(fails)} fails) of {len(clip_list())} 272 clips")
    if fails:                                                              # user requires ALL 272 clips -> never write a partial dataset
        print("  FAILS (must be 0 for full 272 coverage):", fails[:15]); print("  ABORTING write."); return
    missing_cap=[k for k,v in caps.items() if not v]
    if missing_cap: print(f"  {len(missing_cap)} clips missing captions e.g. {missing_cap[:10]}; ABORTING write."); return
    os.makedirs(f"{out}/motions");
    for v3a,tok in toks.items(): np.save(f"{out}/motions/HML3D_Human_{v3a}.npy", tok)
    cond={"HML3D_Human":regen_cond(toks,cond_off)}; np.save(f"{out}/cond.npy", cond, allow_pickle=True)
    json.dump({f"HML3D_Human_{k}.npy":v for k,v in caps.items() if v}, open(f"{out}/motion_texts_by_file.json","w"))
    # readback gate: load a saved clip + saved cond, recover with cond offsets -> RIC -> self-consistency
    import numpy as _np
    c=_np.load(f"{out}/cond.npy",allow_pickle=True).item()["HML3D_Human"]; off=c["offsets"]
    k0=next(iter(toks)); t=_np.load(f"{out}/motions/HML3D_Human_{k0}.npy").astype(_np.float64)
    w=recover_from_bvh_rot_np(t.astype(_np.float32),PARENTS,off).astype(_np.float64)
    r=B.world_to_ric(w,t[:,0]); rb=float(_np.abs(r[:,1:]-t[:,1:,0:3]).max()*1000)
    print(f"\n=== v4-B 272 ({a.rest}) written ===")
    print(f"  clips={len(toks)} captions={sum(1 for v in caps.values() if v)}  cond.offsets/mean/std regen'd")
    print(f"  READBACK self-consistency (saved clip via saved cond.offsets): {rb:.4f} mm  ({'PASS' if rb<(1.0 if a.rest=='neutral' else 300) else 'CHECK'})")
    print(f"  missing/failed: {len(fails)}  (MUST be 0 for full 272 coverage)")


# ========== FULL 272-ONLY re-derive (no v3a dependency; v3a numbering diverges from 272 at high oid) ==========
from mld.data.humanml.utils.paramUtil import t2m_raw_offsets as _RAWOFF, t2m_kinematic_chain as _KC
from mld.data.humanml.scripts.motion_process import extract_features as _extract_features
import torch as _t
_RAWOFF=_t.from_numpy(np.asarray(_RAWOFF,dtype=np.float32))
_FACE=[2,1,17,16]; _FIDL=[7,10]; _FIDR=[8,11]; _FEET_THRE=0.002

def _decode_272_global(x):
    """272-dim -> global joint positions [T,22,3] (recover_from_local_position; VERBATIM 272 logic)."""
    nfrm=x.shape[0]; pnh=x[:,8:8+3*J].reshape(nfrm,J,3); vxy=x[:,:2]; gdr=x[:,2:8]
    ghr=[]; R=decode_6d_rows(gdr); acc=[R[0]]
    for r in R[1:]: acc.append(r@acc[-1])
    ghr=np.array(acc); inv=np.transpose(ghr,(0,2,1))
    ph=np.matmul(np.repeat(inv[:,None],J,axis=1), pnh[...,None]).squeeze(-1)
    v3=np.zeros((nfrm,3)); v3[:,0]=vxy[:,0]; v3[:,2]=vxy[:,1]
    v3[1:]=np.matmul(inv[:-1], v3[1:,:,None]).squeeze(-1)
    rt=np.cumsum(v3,axis=0); ph[:,:,0]+=rt[:,0:1]; ph[:,:,2]+=rt[:,2:]
    return ph

def _resample_pos(pos,Tt):
    if len(pos)==Tt: return pos
    src=np.arange(len(pos)); tgt=np.linspace(0,len(pos)-1,Tt)
    return np.stack([np.stack([np.interp(tgt,src,pos[:,j,c]) for c in range(3)],1) for j in range(pos.shape[1])],1)

def prep_272_full(m272_name, T20=None):
    """272-ONLY prep: decode 272->global pos, resample 30->20 (T20 from ratio if None), extract_features->263->
    convert_263_to_13 for root channels+contact, G272 for rotations, A aligns G272 to that canonical frame."""
    x=load_272(m272_name)
    if x is None: return None
    if x.shape[0]<6:                                                       # pad the 4 tiny 3-frame 272 clips so they build (no drop)
        pad=np.repeat(x[-1:], 6-x.shape[0], axis=0)
        pad[:,0:2]=0.0; pad[:,2:8]=np.array([1,0,0,0,1,0],x.dtype)         # zero root xz-vel + identity heading in the padded frames (no spurious trajectory)
        x=np.concatenate([x,pad],axis=0)
    if T20 is None: T20=max(4,int(round(x.shape[0]*20.0/30.0)))
    gpos=_resample_pos(_decode_272_global(x), T20)                        # [T20,J,3] global (30->20fps grid)
    feats=np.asarray(_extract_features(gpos.astype(np.float32),_FEET_THRE,_RAWOFF,_KC,_FACE,_FIDR,_FIDL))  # [T20-1,263]
    clip=conv.convert_263_to_13(feats)                                    # [T20-1,22,13]
    Tc=clip.shape[0]
    G=resample_rot(g272_world(x), T20)[:Tc]                               # rotations on the SAME T20 grid, then truncate to Tc
    Pw=conv.world_positions(feats).astype(np.float64)                     # RIC route (clip13 ch3:9 is raw HML rot6d, NOT AnyTop-reencoded -> its bvh-FK is wrong; use recover_from_ric)
    Psm=np.zeros((Tc,J,3))
    for j in range(1,J): Psm[:,j]=Psm[:,PARENTS[j]]+np.einsum("tij,j->ti",G[:,PARENTS[j]],OFF_S[j])
    U=C.bone_dirs(Psm); V=C.bone_dirs(Pw); Uu,_,Vt=np.linalg.svd(U.T@V)
    A=Vt.T@np.diag([1,1,np.sign(np.linalg.det(Vt.T@Uu.T))])@Uu.T
    return dict(clip=clip,Gtrue=np.einsum("ij,tpjk->tpik",A,G),G=G,A=A,T=Tc,WR=None,Rloc=None,off_smpl=None,dcur=None)

if __name__=="__main__":
    main()
