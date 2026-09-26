#!/usr/bin/env python3
"""
v4 dataset build (Option B — FULL RE-DERIVE from SMPL, 272-consistent) for human AnyTop clips.

DECISION (user 2026-07-02): align ALL AnyTop attribute processing to MotionStreamer-272. 272's positions are
SMPLX Jtr = rigid-FK(SMPL rotations, SMPL rest) (LBS only moves mesh verts, NOT joints), so 272 is self-consistent
under the SMPL rest. Our decoder recover_from_bvh_rot_np is ALSO rigid FK, so we match 272 EXACTLY by using the
SMPL-neutral rest AND deriving positions by FK of the SMPL rotations. This is uniform with the animals (whose
positions are also rigid-FK of their native rotations) -> clean merge for the new shared VQVAE.

CONSTRUCTION (validated by the per-clip build gates below + KIT/CMU smoke):
  WR = Gtrue = A @ G_smpl (SMPL world rotations aligned to the clip frame; A orthogonal from prep()'s Kabsch).
  Per-parent packing: rotq[p] = WR[parent[p]]^T @ WR[p] (rotq[0]=WR[0]); token[j].ch3:9 = 6d(rotq[parent[j]]).
    => non-root LOCAL rotations = G[gp]^T @ G[p] (A cancels) = RAW SMPL local values (272-consistent, real twist).
       root rotq[0] = A@G[0] = SMPL pelvis orientation in the clip frame (frame choice; recover uses it for FK).
  world_smpl = recover_from_bvh_rot_np(tokens, OFF_S)      # SMPL body (SMPL rest lengths) on the clip trajectory
  ch0:3[j>=1] = world_to_ric(world_smpl)                   # EXACT inverse of recover_from_ric -> self-consistent RIC
  ch9:12[j>=1] = qrot(r_rot_quat, d/dt world_smpl)         # extract_features local-vel convention (re-derived)
  root channels (ch0:3[0], ch3:9[0], ch9/ch11[0]) + ch12 contact: kept from the HumanML3D clip (trajectory + contact).

Result: ch0:3 = FK(ch3:9) EXACT (self-consistent, like the animals); ch3:9 = SMPL rotation values (272); real twist.
ch0:3 changes from HumanML3D by bone-LENGTH (~13mm/bone) + small retarget direction (~3.5deg) — expected/clean.

SCOPE (this stage): validate the PER-CLIP CONSTRUCTION on BASE clips (id<14613) from the FK-anchored remap table
(CMU+EKUT) via dry-run gates + twist QA. Production --write is DEFERRED to the merge stage (see the --write guard
in main()): it needs the full-AMASS remap + a single-SMPL-rest unification + cond.npy regen. Mirror + humanact12
also deferred.

Usage: python scripts/_v4_build_dataset.py [--limit N] [--write] [--out DIR]  (dry-run gates by default).
"""
import json, argparse
import numpy as np
import _v4_cp_calibration as C
conv=C.conv; J=C.J; PARENTS=C.PARENTS; CHILDREN=C.CHILDREN; SINGLE=C.SINGLE
import src.data.anytop_rot6d_fk as FK
recover=FK.recover_from_bvh_rot_np
_qmulvec=FK._quat_mul_vec; _recover_root=FK._recover_root_quat_and_pos_np
_norm=C._norm; signed_twist=C.signed_twist
REPO="/scratch/ts1v23/workspace/noKslot_clean"
REMAP=C.REMAP; OFF=conv.compute_offsets(); R2D=C.R2D

# SMPL-neutral T-pose rest offsets (single canonical human skeleton, betas=0) — 272 / SMPL convention.
_Jn=C.smpl_rest_joints("neutral", np.zeros(16))
OFF_S=np.zeros((J,3))
for j in range(1,J): OFF_S[j]=_Jn[j]-_Jn[PARENTS[j]]

def _mat6d(R): return np.concatenate([R[...,:,0],R[...,:,1]],axis=-1)

def world_to_ric(world, root_token):
    """EXACT inverse of recover_from_ric (motion_process.py:419): xz root-relative (Y absolute), then
    RIC = qrot(r_rot_quat, rel) — recover uses qrot(qinv(r_rot_quat),RIC) so the inverse uses POSITIVE r_rot_quat."""
    T,Jn,_=world.shape
    r_rot_quat,r_pos=_recover_root(root_token)
    rel=world.copy(); rel[...,0]-=r_pos[:,None,0]; rel[...,2]-=r_pos[:,None,2]
    return _qmulvec(np.broadcast_to(r_rot_quat[:,None,:],(T,Jn,4)), rel)

def _rotq_from_WR(WR):
    """per-parent rotq[p]=WR[gp]^T@WR[p] (rotq[0]=WR[0]). WR [T,J,3,3]."""
    T=WR.shape[0]; rotq=np.zeros((J,T,3,3))
    for p in range(J):
        gp=PARENTS[p]
        rotq[p]=WR[:,p] if gp<0 else np.matmul(np.transpose(WR[:,gp],(0,2,1)),WR[:,p])
    return rotq

def build_clip(rec, d=None, offs=None):
    """returns (v4b_motion[T,22,13], phi_dict, gate) or (None,None,reason).
    d: optional precomputed prep() dict (same structure). If given, the SMPL rotations come from d (e.g. a 272
    source via _v4_build_from_272.prep_272) instead of the AMASS FK-remap. Everything else is identical.
    offs: optional rest offsets [J,3] for the FK (default OFF_S = SMPL-neutral). Pass per-clip betas offsets for
    the per-betas variant so ch0:3 = FK(ch3:9, per-clip rest) is self-consistent under that clip's own skeleton."""
    if offs is None: offs=OFF_S
    if d is None: d=C.prep(rec)
    if d is None: return None,None,"prep_failed(source/len)"
    clip=d["clip"]; Gtrue=d["Gtrue"]; T=d["T"]
    # ---- ch3:9 = SMPL rotations (per-parent packing of WR=A@G); root ch3:9 kept from clip ----
    rotq=_rotq_from_WR(Gtrue)
    tok=clip.astype(np.float64).copy()
    for j in range(1,J): tok[:,j,3:9]=_mat6d(rotq[PARENTS[j]])
    # ---- ch0:3 = self-consistent SMPL-FK RIC ----
    world_smpl=recover(tok.astype(np.float32),PARENTS,offs).astype(np.float64)
    ric=world_to_ric(world_smpl, tok[:,0])
    tok[:,1:,0:3]=ric[:,1:]
    # ---- ch9:12 non-root = re-derived local velocity (extract_features convention); keep root + last frame ----
    r_rot_quat,_=_recover_root(tok[:,0])
    dv=world_smpl[1:]-world_smpl[:-1]                                   # [T-1,J,3]
    lv=_qmulvec(np.broadcast_to(r_rot_quat[:-1,None,:],(T-1,J,4)), dv)  # heading-frame velocity
    vel_consistency=float(np.abs(lv[:,1:]-clip[:-1,1:,9:12]).mean()*1000)  # vs clip vel (should be small: pos moved ~13mm)
    tok[:-1,1:,9:12]=lv[:,1:]
    # ---- gates ----
    fk2=recover(tok.astype(np.float32),PARENTS,offs).astype(np.float64)
    ric2=world_to_ric(fk2, tok[:,0])
    selfcon_mm=float(np.abs(ric2[:,1:]-tok[:,1:,0:3]).max()*1000)       # ch0:3 == FK(ch3:9)->RIC : MUST ~0
    # twist present: geodesic diff between SMPL rotq and the zero-twist swing (v3a) at single-child joints.
    # WRv3a (the v3a-swing reference) is QA-ONLY (facing gate); reencode_rot6d("v3a") calls _swing_batch which strict-
    # asserts on near-anti-parallel bones (more common in 272-derived clips) -> never let it kill the build.
    try:
        _,WRv3a=conv.reencode_rot6d(clip,recover(clip.astype(np.float32),PARENTS,OFF).astype(np.float64),OFF,"v3a",return_wr=True)
    except AssertionError:
        WRv3a=None
    phi={}
    for p in SINGLE:
        c=CHILDREN[p][0]; curv=world_smpl[:,c]-world_smpl[:,p]
        try:                                              # twist is a QA-ONLY metric; do NOT let a near-anti-parallel
            Sw=conv._swing_batch(offs[c],curv)            # bone's strict-swing assertion kill the build (tokens use
            phi[p]=signed_twist(np.einsum("tij,tkj->tik",Gtrue[:,p],Sw),_norm(curv))  # Gtrue/FK, not this swing)
        except AssertionError:
            phi[p]=np.zeros(T)                            # degenerate bone -> report 0 twist for this joint (metric only)
    # rotation values match raw SMPL local (A cancels) — verify on ALL non-root joints (incl multi-child 0/9).
    # Use prep()'s raw G (it resolved the AMASS source WITH the _smplh redirect); do NOT re-load here (that would
    # bypass the redirect and FileNotFoundError on KIT).
    rq_raw=_rotq_from_WR(d["G"])
    val_max=float(max(np.abs(rq_raw[p]-rotq[p]).max() for p in range(1,J)))  # ALL non-root -> A cancels -> ~0
    # root FACING preservation: body orientation of v4-B (A@G[0]) vs the v3a/HumanML3D body orientation (Kabsch).
    # This is the CORRECT facing check (NOT root6d-yaw which is the trajectory heading, decoupled by design).
    if WRv3a is not None:                                 # WRv3a is [J,T,3,3] (joint-first); WRv3a[0]=[T,3,3]
        rel0=np.einsum("tij,tkj->tik", Gtrue[:,0], WRv3a[0])
        facing=np.degrees(np.arccos(np.clip((np.trace(rel0,axis1=1,axis2=2)-1)/2,-1,1)))
    else:
        facing=np.zeros(T)                                # degenerate-bone v3a-swing reference -> skip facing QA metric
    # position change vs HumanML3D (informational)
    posdiff=float(np.linalg.norm(ric[:,1:]-clip[:,1:,0:3],axis=-1).mean()*1000)
    gate=dict(selfcon_mm=selfcon_mm, rotval_max=val_max, vel_consistency_mm=vel_consistency,
              root_facing_deg=float(facing.mean()), root_facing_max=float(facing.max()),
              twist_absmean=float(np.mean([np.mean(np.abs(phi[p]))*R2D for p in SINGLE])),
              twist_dphi=float(np.mean([np.mean(np.abs(np.diff(phi[p])))*R2D for p in SINGLE])),
              posdiff_mm=posdiff)
    return tok.astype(np.float32), phi, gate

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--limit",type=int,default=0)
    ap.add_argument("--out",default=REPO+"/data/humanml3d_anytop13_v4_shared_reencoded")
    ap.add_argument("--write",action="store_true",help="write the v4 dataset (default: dry-run gates only)")
    a=ap.parse_args()
    remap=json.load(open(REMAP))
    bases=sorted((v for v in remap.values() if not v["mirror"]), key=lambda v:v["our_id"])
    if a.limit: bases=bases[:a.limit]
    print(f"v4 build (Option B, FULL RE-DERIVE from SMPL, BASE clips): {len(bases)} clips; write={a.write} out={a.out}\n")
    if a.write:
        # --write DEFERRED to the merge stage (codex 019f23f4, P0). The correct write is NOT "copy v3a + overwrite
        # base": that mixes 000021-rest v3a clips with SMPL-rest v4-B clips under one cond.npy (one offsets can't be
        # FK-consistent for both) and reuses v3a mean/std. The merge-stage write must instead: (a) build ALL human
        # clips under OFF_S (real twist where AMASS available, zero-twist swing UNDER OFF_S elsewhere, positions
        # re-derived), (b) regenerate cond.npy (offsets=OFF_S, mean/std over the new clips, tpos_first_frame),
        # (c) add a readback gate that loads a saved clip + saved cond through AnyTopDataset and re-checks
        # self-consistency. That needs the full-AMASS remap first, so it is not implemented here.
        print("  --write is DEFERRED to the merge stage (unification to one SMPL rest + cond.npy regen; codex P0).")
        print("  This stage validates the per-clip construction only (dry-run gates + QA).")
        return
    gates=[]; ok=0; fail=0
    for n,rec in enumerate(bases):
        tok,phi,gate=build_clip(rec)
        if tok is None:
            fail+=1; continue
        gates.append(gate); ok+=1
        if (n+1)%500==0: print(f"  {n+1}/{len(bases)} ok={ok} fail={fail}")
    import numpy as _np
    def col(k): return _np.array([g[k] for g in gates])
    print(f"\n=== v4-B build gates over {len(gates)} base clips ===")
    print(f"  [CRITICAL] self-consistency ch0:3==FK(ch3:9): p95={_np.percentile(col('selfcon_mm'),95):.4f} max={col('selfcon_mm').max():.4f} mm  (MUST ~0)")
    print(f"  [CRITICAL] rotation values == raw SMPL local: max={col('rotval_max').max():.2e}                 (MUST ~0, A cancels)")
    print(f"  vel-convention check (re-derived vs clip vel): mean={col('vel_consistency_mm').mean():.2f} max={col('vel_consistency_mm').max():.2f} mm  (small = convention OK)")
    print(f"  root FACING (v4-B body vs v3a/HumanML3D body): mean={col('root_facing_deg').mean():.2f} max={col('root_facing_max').max():.2f} deg  (small = facing preserved, NOT the trajectory-heading)")
    print(f"  real twist |phi| absmean: {col('twist_absmean').mean():.1f} deg   |dphi/frame|: {col('twist_dphi').mean():.2f} deg  (>0 = real twist present)")
    print(f"  [info] ch0:3 change vs HumanML3D: mean={col('posdiff_mm').mean():.1f} mm  (bone-length + retarget dir; expected)")
    print(f"  ok={ok} fail={fail}")
    scpass = col('selfcon_mm').max()<1.0 and col('rotval_max').max()<1e-3
    print(f"  SELF-CONSISTENCY + SMPL-VALUE gate: {'PASS' if scpass else 'FAIL'}")

if __name__=="__main__":
    main()
