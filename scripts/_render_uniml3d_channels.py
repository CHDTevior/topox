#!/usr/bin/env python3
"""Channel-aware diagnostics using the active project's skeleton drawing/projection functions.

Run on a compute node. Left: canonical rest (XOZ); middle: animated side; right: animated XOZ.
Purple thick bones are rotation-FK; red thin bones are direct positions. Green dots are contact,
teal arrows are world velocity over 0.1 seconds, orange arrows are heading, blue traces root XZ.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts._render_noik_gt import flatten,fit,panel,PANEL_W,PANEL_H,GAP
from scripts._build_uniml3d_ktjd17 import json_write
from src.data.ktjd17.decoder import decode_ktjd17


def arrow(draw,start,end,color):
    draw.line([tuple(start),tuple(end)],fill=color,width=2)
    if np.linalg.norm(end-start)>3:
        a=np.arctan2(*(end-start)[::-1])
        for turn in (-2.6,2.6):
            tip=end+6*np.array([np.cos(a+turn),np.sin(a+turn)])
            draw.line([tuple(end),tuple(tip)],fill=color,width=2)


def render(root,row,out):
    with np.load(root/row["skeleton_relpath"],allow_pickle=False) as z:sk={k:z[k] for k in z.files}
    with np.load(root/row["motion_relpath"],allow_pickle=False) as z:
        m=z["motion"][:240].astype(np.float64);hv=z["heading_valid"][:240]
    args={k:sk[k] for k in ("parents","R_rest_global","R_rest_local","offset_parent_local","rotation_source_kind")}
    dec=decode_ktjd17(m,**args)
    P=dec.positions_direct;fk=dec.positions_fk;par=sk["parents"];s=float(sk["s_rig"])
    rest=sk["P_rest_global"][None];meta=json.loads((root/"source_metadata"/(row["rig_id"]+".json")).read_text())
    heading=P[:,0].copy();heading[:,0]+=.22*s*m[:,0,16];heading[:,2]+=.22*s*m[:,0,15]
    velocity_tip=P+.1*m[...,9:12]
    views={}
    for name in ("side","top"):
        allp=np.concatenate([P,heading[:,None],velocity_tip],axis=1)
        to_px,ground,_=fit([flatten(allp,name)])
        views[name]=(to_px(flatten(P,name)),to_px(flatten(fk,name)),to_px(flatten(heading[:,None],name))[:,0],
                     to_px(flatten(velocity_tip,name)),ground)
    rest_tip=rest[:,0].copy();rest_tip[:,2]+=.22*s
    rx,rg,_=fit([flatten(np.concatenate([rest,rest_tip[:,None]],axis=1),"top")])
    restxy=rx(flatten(rest,"top"))[0];resthead=rx(flatten(rest_tip[:,None],"top"))[0,0]
    W=PANEL_W*3+GAP*2;frames=[]
    for t in range(len(m)):
        im=Image.new("RGB",(W,PANEL_H+134),(246,248,247))
        panel(im,0,restxy,par,"top",rg,"REST XOZ / forward +Z",resthead,True)
        for k,name in enumerate(("side","top"),1):
            x0=k*(PANEL_W+GAP);xy,pfk,head,vel,ground=views[name]
            panel(im,x0,xy[t],par,name,ground,"MOTION "+name.upper(),head[t],bool(hv[t]))
            d=ImageDraw.Draw(im)
            for j in range(1,len(par)):
                a=pfk[t,j]+[x0,0];b=pfk[t,par[j]]+[x0,0]
                d.line([tuple(a),tuple(b)],fill=(130,90,185),width=5)
                a=xy[t,j]+[x0,0];b=xy[t,par[j]]+[x0,0]
                d.line([tuple(a),tuple(b)],fill=(185,28,28),width=2)
            # Ground line on the side view and axis orientation are inherited unchanged.
            roottrack=xy[:,0]+[x0,0]
            d.line([tuple(p) for p in roottrack],fill=(85,130,210),width=1)
            for j in range(len(par)):
                at=xy[t,j]+[x0,0]
                if m[t,j,12]>.5:
                    d.ellipse([at[0]-3,at[1]-3,at[0]+3,at[1]+3],fill=(20,155,55))
                if j==0 or j in sk["contact_joint_indices"]:
                    arrow(d,at,vel[t,j]+[x0,0],(10,130,135))
            at=xy[t,0]+[x0,0]
            d.ellipse([at[0]-7,at[1]-7,at[0]+7,at[1]+7],outline=(20,40,200),width=2)
            if hv[t]:arrow(d,at,head[t]+[x0,0],(224,96,0))
        d=ImageDraw.Draw(im);y=PANEL_H+5
        lines=[f"{row['rig_id']} | {row['body_plan']} | frame {t+1}/{len(m)} | 30fps",
               "purple/red overlap = rotation-FK/direct position | green = ch12 contact | teal = ch9:12 velocity x 0.1s",
               f"ch0:3 root q: {np.array2string(m[t,0,:3],precision=3)}   ch13:15 smooth XZ: {np.array2string(m[t,0,13:15],precision=3)}",
               f"ch15:17 heading: {np.array2string(m[t,0,15:17],precision=3)} valid={bool(hv[t])} carrier={meta['heading_carrier_joint']}",
               f"contact joints={int(m[t,:,12].sum())}/{len(par)} | min Y / rig scale={P[t,:,1].min()/s:.5f} | facing need not equal travel",
               (row['captions'][0] if row['captions'] else '')]
        for line in lines:d.text((8,y),line,fill=(30,42,50));y+=20
        frames.append(im)
    stem=row["rig_id"]+"__"+row["clip_id"]
    dest=out/(stem+".gif")
    frames[0].save(dest,save_all=True,append_images=frames[1:],duration=33,loop=0)
    sheet=Image.new("RGB",(W,4*(PANEL_H+134)),"white")
    for k,t in enumerate(np.linspace(0,len(frames)-1,4).astype(int)):sheet.paste(frames[t],(0,k*(PANEL_H+134)))
    sheet.save(out/(stem+"_sequence.png"))
    print(f"channel preview {row['rig_id']} {len(m)} frames",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("root",type=Path)
    p.add_argument("--count",type=int,default=7);a=p.parse_args()
    selection=json.loads((a.root/"analysis/visuals/selection.json").read_text())[:a.count]
    check=a.root/"analysis/channel_semantics_verification.json"
    if check.exists():
        checks=json.loads(check.read_text())
        alternate=[r["rig_id"] for r in checks["rig_checks"] if r["heading_carrier_joint"]!=0][:2]
        rows=[json.loads(l) for l in (a.root/"manifests/clips.jsonl").open()]
        extra=[next(r for r in rows if r["rig_id"]==rig) for rig in alternate]
        extra += [min(rows,key=lambda r:r["qa"]["heading_valid_fraction"]),
                  max(rows,key=lambda r:r["qa"]["max_step_rig_units"])]
        existing={r["clip_id"] for r in selection}
        for r in extra:
            if r["clip_id"] not in existing:
                selection.append(r);existing.add(r["clip_id"])
    out=a.root/"analysis/channel_visuals";out.mkdir(parents=True,exist_ok=True)
    for row in selection:render(a.root,row,out)
    json_write(out/"selection.json",selection)
