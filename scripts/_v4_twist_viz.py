#!/usr/bin/env python3
"""
v4 TWIST visual-QA: twist is a GAUGE (does not move joints), so a normal skeleton render can't show it.
Render each single-child bone's coordinate frame as a short perpendicular "flag" that ROLLS with the
bone's axial twist. v3a (zero twist) -> flags stay in a fixed plane; v4 (real twist) -> flags roll about
the bone. Side-by-side animated GIF for user verdict (esp. spine, shoulders, elbows/forearms).

Usage: python scripts/_v4_twist_viz.py <our_id> [--out PATH]   (renders v3a-vs-v4 for that clip)
"""
import os, sys, argparse, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import _v4_cp_calibration as C
import _v4_build_dataset as B
conv=C.conv; J=C.J; PARENTS=C.PARENTS; CHILDREN=C.CHILDREN; SINGLE=C.SINGLE; NAMES=C.NAMES; LIMB=C.LIMB
OFF=conv.compute_offsets(); _norm=C._norm
from src.data.anytop_rot6d_fk import recover_from_bvh_rot_np
# joints to flag (twist-carrying single-child limbs/spine)
FLAG=[3,6,12,16,17,18,19,1,2,4,5]   # spine1,spine2,neck,collars? uparm,forearm,thigh,shin

def _perp0(u):
    perp=np.cross(u,[0,1.0,0])
    if np.linalg.norm(perp)<1e-3: perp=np.cross(u,[1.0,0,0])
    return perp/np.linalg.norm(perp)

def flags_from_WR(pos, WRdict, offs):
    """per-single-child flag endpoints [T,J,3] = bone-midpoint + L*(WR @ rest_perp) -> rolls with twist."""
    T=pos.shape[0]; flags=np.full((T,J,3),np.nan)
    for p in SINGLE:
        c=CHILDREN[p][0]
        mid=0.5*(pos[:,p]+pos[:,c]); L=0.5*np.linalg.norm(offs[c])
        flags[:,p]=mid+L*np.einsum("tij,j->ti", WRdict[p], _perp0(_norm(offs[c][None])[0]))
    return flags

def frames_and_flags(clip, offs):
    pos=recover_from_bvh_rot_np(clip.astype(np.float32),PARENTS,offs).astype(np.float64)
    WR=C.wr_from_tokens(clip)
    return pos, flags_from_WR(pos, {p:WR[p] for p in SINGLE}, offs)

def render(oid, out):
    v3a=C.load_clip(oid)
    if v3a is None: print(f"no v3a clip {oid}"); return
    # build v4 for this clip
    remap=json.load(open(C.REMAP)); rec=None
    for v in remap.values():
        if v["our_id"]==oid: rec=v; break
    if rec is None: print(f"{oid} not in remap (no AMASS twist)"); return
    v4,phi,gate=B.build_clip(rec)
    if v4 is None: print(f"build failed {oid}: {phi}"); return
    n=min(len(v3a),len(v4)); v3a=v3a[:n].astype(np.float64); v4=v4[:n]
    p3,f3=frames_and_flags(v3a, OFF); p4,f4=frames_and_flags(v4, B.OFF_S)   # v4 uses SMPL rest offsets
    # SMPL GROUND-TRUTH flags (blue) from the real AMASS SMPL frame Gtrue, drawn on the v4 skeleton
    d=C.prep(rec); Gt=d["Gtrue"]
    fS=flags_from_WR(p4, {p:Gt[:n,p] for p in SINGLE}, B.OFF_S)
    absmax=np.mean([np.mean(np.abs(phi[p]))*C.R2D for p in SINGLE])
    print(f"clip {oid}: T={n} mean|phi|={absmax:.1f}deg  gate selfcon={gate['selfcon_mm']:.4f}mm rotval={gate['rotval_max']:.1e}")

    fig=plt.figure(figsize=(11,6));
    axes=[fig.add_subplot(1,2,i+1,projection="3d") for i in range(2)]
    titles=["v3a (zero twist — flag fixed)","v4 RED  vs  SMPL truth BLUE  (roll together = correct)"]
    allp=np.concatenate([p3,p4],0).reshape(-1,3); ctr=allp.mean(0); rng=(allp.max(0)-allp.min(0)).max()*0.55
    def draw(fr):
        for k,(ax,P,F,ttl) in enumerate([(axes[0],p3,f3,titles[0]),(axes[1],p4,f4,titles[1])]):
            ax.clear(); ax.set_title(ttl,fontsize=9)
            for j in range(1,J):
                a,b=P[fr,PARENTS[j]],P[fr,j]
                ax.plot([a[0],b[0]],[a[2],b[2]],[a[1],b[1]],c="0.5",lw=1.5)
            for p in FLAG:
                if p in SINGLE:
                    m=0.5*(P[fr,p]+P[fr,CHILDREN[p][0]]); e=F[fr,p]
                    ax.plot([m[0],e[0]],[m[2],e[2]],[m[1],e[1]],c="crimson",lw=2)
                    if k==1:                                    # overlay SMPL ground truth (blue) on v4 panel
                        eS=fS[fr,p]
                        ax.plot([m[0],eS[0]],[m[2],eS[2]],[m[1],eS[1]],c="royalblue",lw=2,alpha=0.85)
            ax.set_xlim(ctr[0]-rng,ctr[0]+rng); ax.set_ylim(ctr[2]-rng,ctr[2]+rng); ax.set_zlim(ctr[1]-rng,ctr[1]+rng)
            ax.set_box_aspect((1,1,1)); ax.view_init(elev=12,azim=fr*2%360); ax.set_axis_off()
        fig.suptitle(f"{oid}  frame {fr}/{n}   mean|twist|={absmax:.0f}°   RED=v4 injected, BLUE=SMPL truth — should roll IN SYNC (offset ok, opposite=reversed)",fontsize=10)
    step=max(1,n//80)
    anim=FuncAnimation(fig,draw,frames=range(0,n,step),interval=80)
    anim.save(out,writer=PillowWriter(fps=12)); plt.close(fig)
    print(f"WROTE {out}")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("oid"); ap.add_argument("--out",default=None); a=ap.parse_args()
    out=a.out or f"/scratch/ts1v23/workspace/noKslot_clean/scratch/v4_twist_{a.oid}.gif"
    render(a.oid,out)
