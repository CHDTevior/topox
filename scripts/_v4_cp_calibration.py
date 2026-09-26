#!/usr/bin/env python3
"""
Stage-1 step-2: GLOBAL-C_p twist-fidelity calibration for human real-twist recovery (v4).
See handoff/20260701_235711_human_twist_recovery_v4_implementation.md (Rev 2), §3.3 (load-bearing go/no-go).

QUESTION: the PoC extracted twist with a PER-CLIP frame-averaged C_p (motion-dependent). Production
wants a MOTION-INDEPENDENT C_p. Does a single GLOBAL per-bone C_p preserve the PoC's validated
twist fidelity? (An earlier off-axis metric was WRONG: it penalised the large constant rest-gap
000021-arms-down vs SMPL-T-pose, which is ABSORBED by the bind pose and cancels in frame-to-frame
dynamics — the metric that actually matters for skinning.)

Reuse the PoC's VALIDATED metrics (positions preserved to um; twist matches SMPL <1-2deg on 2 CMU clips):
  (c) DYNAMIC twist fidelity = mean_t |geo(injected stored-local rotation) - geo(SMPL local rotation)|
      -> Cp-CONSTANT-INVARIANT (constant cancels frame-to-frame), so per-clip==global here; small => real twist dynamics.
  (d) SMOOTHNESS: injected stored-local geodesic frame-delta vs SMPL local (must be SMPL-like, NOT v2 90-180deg).
  + ABSOLUTE twist: |phi| range under global C_p (sane, not blown up).
  + CROSS-CLIP BIND STABILITY: mean(phi - SMPL_anatomical_twist) per clip -> spread across clips small
    => one global C_p gives a CONSISTENT absolute twist / bind pose (the real motion-independence test).
  + POSITIONS: gauge identity (injecting any roll about the current bone preserves FK) -> spot-checked.

GO if global-C_p dynamic fidelity is small (<~3deg) at every joint AND smoothness ~SMPL AND cross-clip
bind spread is small. Then global C_p is production-ready. Else report specifics.

No GPU, read-only. Usage: python scripts/_v4_cp_calibration.py [--sample N]
"""
import os, json, argparse, importlib.util
import numpy as np

REPO="/scratch/ts1v23/workspace/noKslot_clean"
HM="/iridisfs/scratch/ts1v23/workspace/motion-latent-diffusion-main"
AMASS=HM+"/datasets/amass/motion_data"; BM=HM+"/datasets/humanml3d/body_models/smplh"
V3A=REPO+"/data/humanml3d_anytop13_v3a_shared_reencoded"
REMAP=REPO+"/scratch/v4_remap/cmu_ekut_remap.json"; R2D=180.0/np.pi

spec=importlib.util.spec_from_file_location("conv",REPO+"/scripts/convert_humanml3d_to_anytop13.py")
conv=importlib.util.module_from_spec(spec); spec.loader.exec_module(conv)
from src.data.anytop_rot6d_fk import recover_from_bvh_rot_np, _rotation_6d_to_matrix_np
J=conv.J; PARENTS=conv.PARENTS; CHILDREN=conv.CHILDREN; SINGLE=conv._SINGLE_CHILD_PARENTS
NAMES=getattr(conv,"JOINT_NAMES",[str(i) for i in range(J)])
LIMB={1:"L_thigh",2:"R_thigh",4:"L_shin",5:"R_shin",7:"L_ankle",8:"R_ankle",3:"spine1",6:"spine2",
      12:"neck",13:"L_collar",14:"R_collar",16:"L_uparm",17:"R_uparm",18:"L_forearm",19:"R_forearm"}

# ---- rotation helpers (verbatim from validated PoC) ----
def _skew(w):
    K=np.zeros(w.shape[:-1]+(3,3))
    K[...,0,1]=-w[...,2];K[...,0,2]=w[...,1];K[...,1,0]=w[...,2]
    K[...,1,2]=-w[...,0];K[...,2,0]=-w[...,1];K[...,2,1]=w[...,0];return K
def _norm(v): return v/(np.linalg.norm(v,axis=-1,keepdims=True)+1e-12)
def rodrigues(aa):
    th=np.linalg.norm(aa,axis=-1,keepdims=True); ax=aa/np.clip(th,1e-12,None)
    K=_skew(ax); th=th[...,None]; I=np.broadcast_to(np.eye(3),K.shape).copy()
    return I+np.sin(th)*K+(1-np.cos(th))*(K@K)
def axis_angle_batch(axis,angle): return rodrigues(_norm(axis)*angle[:,None])
def minimal_arc(a,b):
    a=_norm(a);b=_norm(b);v=np.cross(a,b);c=np.sum(a*b,-1);K=_skew(v)
    I=np.broadcast_to(np.eye(3),K.shape).copy();coef=1.0/(1.0+c)
    R=I+K+(K@K)*coef[...,None,None];bad=c<-1+1e-6
    for t in np.where(bad)[0]:
        ref=np.array([1.0,0,0]) if abs(a[t,0])<0.9 else np.array([0,1.0,0])
        R[t]=rodrigues(_norm(np.cross(a[t],ref))*np.pi)
    return R
def signed_twist(M,a):
    # a may be a single axis (3,) or per-frame axes (T,3); broadcast elementwise
    w=0.5*np.stack([M[:,2,1]-M[:,1,2],M[:,0,2]-M[:,2,0],M[:,1,0]-M[:,0,1]],-1)   # [T,3] = sin(th)*axis
    sin_s=(w*np.broadcast_to(np.asarray(a,dtype=float),w.shape)).sum(-1)
    return np.arctan2(sin_s,np.clip((np.trace(M,axis1=1,axis2=2)-1.0)/2.0,-1,1))
def twist_about_rest(R,a):
    T=R.shape[0]; b=np.einsum("tij,j->ti",R,a); S=minimal_arc(np.broadcast_to(a,(T,3)),b)
    return signed_twist(np.transpose(S,(0,2,1))@R,a)
def proj_so3(M):
    U,_,Vt=np.linalg.svd(M); return U@np.diag([1,1,np.sign(np.linalg.det(U@Vt))])@Vt
def mat_to_6d(R): return np.concatenate([R[...,:,0],R[...,:,1]],axis=-1)
def geo_seq(R):
    d=np.matmul(np.transpose(R[:-1],(0,2,1)),R[1:])
    return np.degrees(np.arccos(np.clip((np.trace(d,axis1=1,axis2=2)-1)/2,-1,1)))
def smpl_world_rot(ro,pb):
    T=ro.shape[0]; local=np.concatenate([ro[:,None,:],pb.reshape(T,21,3)],1); Rloc=rodrigues(local)
    G=np.zeros((T,J,3,3));G[:,0]=Rloc[:,0]
    for j in range(1,J): G[:,j]=G[:,PARENTS[j]]@Rloc[:,j]
    return G,Rloc
_MODEL={}
def smpl_rest_joints(gender,betas):
    g=gender if gender in("male","female","neutral") else "neutral"
    if g not in _MODEL:
        m=np.load(f"{BM}/{g}/model.npz",allow_pickle=True)
        kt=np.asarray(m["kintree_table"])[0,:J].copy();kt[0]=-1
        assert kt.tolist()==list(PARENTS)
        _MODEL[g]=(np.asarray(m["v_template"],np.float64),np.asarray(m["shapedirs"],np.float64),np.asarray(m["J_regressor"],np.float64))
    v_t,sd,Jr=_MODEL[g];nb=min(len(betas),sd.shape[2])
    return (Jr@(v_t+sd[:,:,:nb]@betas[:nb]))[:J]
def bone_dirs(P): return np.concatenate([_norm(P[:,j]-P[:,PARENTS[j]]) for j in range(1,J)],0)
def wr_from_tokens(new13):
    T=new13.shape[0]; rotq=np.broadcast_to(np.eye(3),(J,T,3,3)).copy()
    for p in range(J):
        if CHILDREN[p]: rotq[p]=_rotation_6d_to_matrix_np(new13[:,CHILDREN[p][0],3:9].astype(np.float64))
    WR=np.broadcast_to(np.eye(3),(J,T,3,3)).copy();WR[0]=rotq[0]
    for p in range(1,J): WR[p]=WR[PARENTS[p]]@rotq[p]
    return WR

_NPZ={}
def load_amass(rel):
    if rel not in _NPZ:
        if len(_NPZ)>300:_NPZ.clear()
        d=np.load(f"{AMASS}/{rel}",allow_pickle=True)
        _NPZ[rel]=(np.asarray(d["poses"],np.float64),float(d["mocap_framerate"]),str(d.get("gender","neutral")),np.asarray(d["betas"],np.float64))
    return _NPZ[rel]
def load_clip(oid):
    for sub in("motions","motions_heldout"):
        p=f"{V3A}/{sub}/HML3D_Human_{oid}.npy"
        if os.path.exists(p): return np.load(p).astype(np.float64)
    return None

_OFF=None
def prep(rec):
    """load + compute Gtrue, WR (v3a swing), Rloc(SMPL local), dcur, off_smpl, all in the SAME absolute
    world frame Pw=recover(clip) (matching the PoC's world_positions). return None if unusable."""
    global _OFF
    if _OFF is None: _OFF=conv.compute_offsets()
    rel=rec["source_path"].replace("./pose_data/","").replace("_poses.npy","_poses.npz")
    if not os.path.exists(f"{AMASS}/{rel}") and "/" in rel:   # KIT etc: SMPL-H _poses.npz lives under <subset>_smplh
        sub,tail=rel.split("/",1); alt=f"{sub}_smplh/{tail}"
        if os.path.exists(f"{AMASS}/{alt}"): rel=alt
    if not os.path.exists(f"{AMASS}/{rel}"): return None
    poses,fps,gender,betas=load_amass(rel); ds=int(round(fps/20.0))
    s,e=rec["start"],rec["end"]; ro=poses[::ds,:3][s:e][:-1]; pb=poses[::ds,3:66][s:e][:-1]
    clip=load_clip(rec["our_id"])
    if clip is None or ro.shape[0]!=clip.shape[0] or clip.shape[0]<3: return None
    # ABSOLUTE world positions of the stored clip (NOT ch0:3 which is RIC/rotation-invariant) —
    # this is the frame the converter's WR / recover FK live in; use it consistently.
    Pw=recover_from_bvh_rot_np(clip.astype(np.float32),PARENTS,_OFF).astype(np.float64)
    _,WR=conv.reencode_rot6d(clip,Pw,_OFF,"v3a",return_wr=True); WR=WR.astype(np.float64)
    G,Rloc=smpl_world_rot(ro,pb); Jrest=smpl_rest_joints(gender,betas)
    off=np.zeros((J,3))
    for j in range(1,J): off[j]=Jrest[j]-Jrest[PARENTS[j]]
    P_smpl=np.zeros((clip.shape[0],J,3))
    for j in range(1,J): P_smpl[:,j]=P_smpl[:,PARENTS[j]]+np.einsum("tij,j->ti",G[:,PARENTS[j]],off[j])
    U=bone_dirs(P_smpl);V=bone_dirs(Pw);Uu,_,Vt=np.linalg.svd(U.T@V)
    A=Vt.T@np.diag([1,1,np.sign(np.linalg.det(Vt.T@Uu.T))])@Uu.T
    Gtrue=np.einsum("ij,tpjk->tpik",A,G)
    off_smpl={p:_norm(Jrest[CHILDREN[p][0]]-Jrest[p]) for p in SINGLE}
    dcur={p:_norm(Pw[:,CHILDREN[p][0]]-Pw[:,p]) for p in SINGLE}
    return dict(clip=clip,Gtrue=Gtrue,WR=WR,Rloc=Rloc,G=G,off_smpl=off_smpl,dcur=dcur,T=clip.shape[0])

def cp_perclip(d,p):
    return proj_so3(np.mean(np.transpose(d["Gtrue"][:,p],(0,2,1))@d["WR"][p],axis=0))

def extract_phi(d,p,Cp):
    target=d["Gtrue"][:,p]@Cp
    resid=np.einsum("tij,tjk->tik",target,np.transpose(d["WR"][p],(0,2,1)))
    return signed_twist(resid,d["dcur"][p])   # roll about CURRENT bone

def inject_and_decode(d,phi_by_p):
    """inject phi at single-child joints, repack, decode stored-local rotations per single-child joint."""
    WR_new=d["WR"].copy()
    for p in SINGLE:
        WR_new[p]=axis_angle_batch(d["dcur"][p],phi_by_p[p])@d["WR"][p]
    rq={}
    for p in SINGLE:
        gp=PARENTS[p]
        rq[p]=WR_new[p] if gp<0 else np.transpose(WR_new[gp],(0,2,1))@WR_new[p]
    return WR_new,rq

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--sample",type=int,default=600);a=ap.parse_args()
    remap=json.load(open(REMAP))
    base=sorted((v for v in remap.values() if not v["mirror"]),key=lambda v:v["our_id"])
    ntot=len(base)
    if a.sample and ntot>a.sample:
        base=[base[int(round(i*(ntot-1)/(a.sample-1)))] for i in range(a.sample)]
        base=[base[i] for i in sorted(set(range(len(base))))]
    print(f"calibrating on {len(base)} base accepted clips (of {ntot})\n")

    # pass 1: per-clip C_p -> global C_p per bone
    Csum={p:np.zeros((3,3)) for p in SINGLE}; cache=[]; nfail=0
    for rec in base:
        d=prep(rec)
        if d is None: nfail+=1; continue
        for p in SINGLE: Csum[p]+=cp_perclip(d,p)
        cache.append(d)
    nok=len(cache); Cglob={p:proj_so3(Csum[p]) for p in SINGLE}
    print(f"processed {nok} clips ({nfail} skipped)\n")

    # pass 2: inject with GLOBAL C_p; measure PoC metrics + absolute + cross-clip bind
    dyn={p:[] for p in SINGLE}          # |geo(inj-local)-geo(smpl-local)| mean per clip
    offax={p:[] for p in SINGLE}        # PRIMARY: off-axis residual after removing the roll (deg) = twist-fidelity error
    jit_inj={p:[] for p in SINGLE}; jit_smpl={p:[] for p in SINGLE}   # mean geodesic frame-delta
    absphi={p:[] for p in SINGLE}       # mean |phi| deg
    binderr={p:[] for p in SINGLE}      # per-clip mean(phi - anatomical_twist) deg
    gauge_max=0.0
    OFF=conv.compute_offsets()
    for k,d in enumerate(cache):
        phi={p:extract_phi(d,p,Cglob[p]) for p in SINGLE}
        WR_new,rq=inject_and_decode(d,phi)
        for p in SINGLE:
            gI=geo_seq(rq[p]); gS=geo_seq(d["Rloc"][:,p])
            dyn[p].append(np.mean(np.abs(gI-gS)))
            # PRIMARY metric (codex): is the true-vs-swing residual actually a PURE roll about dcur?
            resid=np.einsum("tij,tjk->tik",d["Gtrue"][:,p]@Cglob[p],np.transpose(d["WR"][p],(0,2,1)))
            roll=axis_angle_batch(d["dcur"][p],phi[p])
            leftover=np.transpose(roll,(0,2,1))@resid
            offax[p].append(np.degrees(np.arccos(np.clip((np.trace(leftover,axis1=1,axis2=2)-1)/2,-1,1))))
            jit_inj[p].append(gI.mean()); jit_smpl[p].append(gS.mean())
            absphi[p].append(np.mean(np.abs(phi[p]))*R2D)
            anat=twist_about_rest(d["Rloc"][:,p],d["off_smpl"][p])   # true SMPL anatomical twist
            binderr[p].append(np.mean(phi[p]-anat)*R2D)
        # gauge spot-check on a few clips (positions preserved for ANY phi). FULL repack over ALL joints
        # (must recompute rotq[i] for every i incl multi-child 9 whose parent 6 is injected), like the PoC.
        if k<3:
            rotq_full={}
            for i in range(J):
                if not CHILDREN[i]: continue
                gp=PARENTS[i]
                rotq_full[i]=WR_new[i] if gp<0 else np.transpose(WR_new[gp],(0,2,1))@WR_new[i]
            new_inj=d["clip"].astype(np.float64).copy()
            for j in range(1,J): new_inj[:,j,3:9]=mat_to_6d(rotq_full[PARENTS[j]])
            fk_inj=recover_from_bvh_rot_np(new_inj.astype(np.float32),PARENTS,OFF).astype(np.float64)
            fk_v3a=recover_from_bvh_rot_np(d["clip"].astype(np.float32),PARENTS,OFF).astype(np.float64)  # same frame
            gauge_max=max(gauge_max,np.linalg.norm(fk_inj-fk_v3a,axis=-1).max())   # gauge: inject vs v3a (both FK world)

    def pct(lst,q): return np.percentile(np.concatenate(lst),q)
    print(f"{'joint':<11}{'Cp_glob':>8}  {'OFF-AXIS mean/p95/max(deg)':>28}  {'dyn(deg)':>9}  {'|phi|':>7}  {'bind spread':>11}")
    worst_off=0.0
    for p in SINGLE:
        cpg=np.degrees(np.arccos(np.clip((np.trace(Cglob[p])-1)/2,-1,1)))
        om=pct(offax[p],50); o95=pct(offax[p],95); omx=np.concatenate(offax[p]).max()
        worst_off=max(worst_off,o95)
        print(f"{LIMB.get(p,NAMES[p]):<11}{cpg:7.1f}  {om:8.2f} /{o95:7.2f} /{omx:7.2f}       {np.mean(dyn[p]):7.3f}  {np.mean(absphi[p]):6.1f}  {np.std(binderr[p]):8.1f}")
    print("\n"+"="*74)
    alloff=np.concatenate([np.concatenate(offax[p]) for p in SINGLE])
    print(f"gauge (FK(inject) vs FK(v3a)) max = {gauge_max*1e6:.3f} um  (positions preserved regardless of phi)")
    print(f"OFF-AXIS residual (twist-fidelity error under GLOBAL C_p): overall p95={np.percentile(alloff,95):.2f}deg  worst-joint p95={worst_off:.2f}deg")
    print("NOTE: dynamic-fidelity col is C_p-INVARIANT (even zero-twist v3a passes ~2deg) -> NOT the gate. OFF-AXIS is the gate.")
    if worst_off<5.0 and gauge_max*1e6<10.0:
        print("VERDICT: GO — under a global C_p the true-vs-swing residual is a near-pure roll (<5deg) at every joint;")
        print("         swing+roll faithfully carries the absolute twist. Positions exact.")
    else:
        print("VERDICT: NO-GO for skinning-grade absolute twist via constant C_p — off-axis residual is large at")
        print("         large-rest-gap joints (arms): the 000021-vs-SMPL rest holonomy is POSE-DEPENDENT and no")
        print("         constant C_p removes it. Twist DYNAMICS are captured; ABSOLUTE frame is not. Report to user.")

if __name__=="__main__":
    main()
