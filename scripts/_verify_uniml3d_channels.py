#!/usr/bin/env python3
"""Independent source-to-channel semantics checks, with the active no-IK corpus as reference.

Run on a compute node inside srun. No training/embedding sidecars are required.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts._build_uniml3d_ktjd17 import json_write,sha
from src.data.ktjd17.decoder import decode_ktjd17


def scalar_max(x):return float(np.max(np.abs(x)))


def reference_probe(root,n=100):
    """Run the same direct/FK and world-velocity interpretation on active on-disk data."""
    rows=[json.loads(l) for l in (root/"manifests/clips.jsonl").open()]
    rng=np.random.default_rng(0);results=[]
    for i in rng.choice(len(rows),min(n,len(rows)),replace=False):
        row=rows[i]
        with np.load(root/row["skeleton_relpath"],allow_pickle=False) as z:
            sk={k:z[k] for k in ("parents","R_rest_global","R_rest_local","offset_parent_local","rotation_source_kind")}
            s=float(z["s_rig"])
        with np.load(root/row["motion_relpath"],allow_pickle=False) as z:
            m=z["motion"].astype(np.float64);fps=float(z["fps_target"])
        dec=decode_ktjd17(m,**sk);P=dec.positions_direct
        v=np.diff(P,axis=0)*fps;v=np.concatenate([v,v[-1:]],axis=0)
        results.append(dict(clip_id=row["clip_id"],fk_direct_norm=scalar_max(P-dec.positions_fk)/s,
                            velocity_norm=scalar_max(v-m[...,9:12])/(fps*s),
                            velocity_interior_norm=scalar_max(v[:-1]-m[:-1,:,9:12])/(fps*s),
                            velocity_tail_norm=scalar_max(v[-1]-m[-1,:,9:12])/(fps*s)))
    return dict(checked_clips=len(results),max_fk_direct_norm=max(r["fk_direct_norm"] for r in results),
                max_world_velocity_norm=max(r["velocity_norm"] for r in results),
                max_world_velocity_interior_norm=max(r["velocity_interior_norm"] for r in results),
                max_world_velocity_tail_norm=max(r["velocity_tail_norm"] for r in results),
                tail_mismatch_clips=sum(r["velocity_tail_norm"]>2e-5 for r in results),
                worst_clips=sorted(results,key=lambda r:r["velocity_norm"],reverse=True)[:10],
                interpretation="Active corpus interior velocity has the same world forward-difference semantics. "
                    "Tail mismatches are reported separately; current UniML3D follows the frozen schema's repeated-last-difference rule. "
                    "Reference data is not modified.")


def verify(root,raw,reference,n):
    rows=[json.loads(l) for l in (root/"manifests/clips.jsonl").open()]
    groups=defaultdict(list)
    for r in rows:groups[r["rig_id"]].append(r)
    rng=np.random.default_rng(0)
    # Cover every rig's rest; clip checks sample n plus one per body plan and altered carriers.
    chosen={rows[i]["clip_id"] for i in rng.choice(len(rows),min(n,len(rows)),replace=False)}
    spec={"generation_id":json.loads((root/"generation.json").read_text())["generation_id"],"rigs":{}}
    rest_results=[]
    for rig,clips in sorted(groups.items()):
        meta=json.loads((root/"source_metadata"/(rig+".json")).read_text())
        with np.load(root/clips[0]["skeleton_relpath"],allow_pickle=False) as z:
            sk={k:z[k] for k in z.files}
        with np.load(raw/meta["first_source_npz"],allow_pickle=False) as z:
            names=list(z["names"].astype(str));par=z["parents"]
        # Derive an independent parent-before-child order from SOURCE names/parents.
        queue=[int(np.flatnonzero(par==-1)[0])];perm=[]
        while queue:
            j=queue.pop(0);perm.append(j);queue.extend(np.flatnonzero(par==j).tolist())
        want=[names[j] for j in perm]
        assert list(sk["joint_names"])==want,rig
        clean=dict(zip(meta["raw_joint_names"],meta["clean_joint_names"],strict=True))
        spec["rigs"][rig]={"joints":[{"name":name,"description":f"{clean[name]} joint."} for name in want]}
        assert list(sk["joint_descriptions"])==[j["description"] for j in spec["rigs"][rig]["joints"]],rig
        P=sk["P_rest_global"];s=float(sk["s_rig"]);carrier=meta["heading_carrier_joint"]
        C=np.array(meta["source_to_canonical_yaw"])
        face=meta["annotations_face_pair"]
        r,l=[want.index(face[k]["raw"]) for k in ("r_hip","l_hip")]
        forward=P[r]-P[l]
        if not face.get("body_axis",False):forward=np.cross([0.,1.,0.],forward)
        forward=forward[[0,2]];forward/=np.linalg.norm(forward)
        carrier_fwd=sk["R_rest_global"][carrier] @ np.array(meta["u_forward_local"])
        residual=max(scalar_max(forward-[0,1]),scalar_max(carrier_fwd-[0,0,1]),
                     scalar_max(P[0,[0,2]])/s,abs(float(P[:,1].min()))/s,
                     abs(float(np.linalg.det(C))-1),scalar_max(C @ [0,1,0]-[0,1,0]))
        assert residual<1e-7,(rig,residual)
        rest_results.append({"rig_id":rig,"max_rest_contract_error":residual,"heading_carrier_joint":carrier})
        if carrier!=0:chosen.add(clips[0]["clip_id"])
    json_write(root/"analysis/joint_spec.json",spec)
    results=[]
    for row in rows:
        if row["clip_id"] not in chosen:continue
        rig=row["rig_id"]
        meta=json.loads((root/"source_metadata"/(rig+".json")).read_text())
        with np.load(root/row["skeleton_relpath"],allow_pickle=False) as z:sk={k:z[k] for k in z.files}
        with np.load(root/row["motion_relpath"],allow_pickle=False) as z:
            m=z["motion"].astype(np.float64);hv=z["heading_valid"];oxz=z["origin_xz"]
        s=float(sk["s_rig"]);par=sk["parents"]
        args={k:sk[k] for k in ("parents","R_rest_global","R_rest_local","offset_parent_local","rotation_source_kind")}
        dec=decode_ktjd17(m,**args);P=dec.positions_direct
        # ch0:3 direct position and ch3:9 FK agree without integrating any channel.
        gap=scalar_max(P-dec.positions_fk)/s
        v=np.diff(P,axis=0)*30.;v=np.concatenate([v,v[-1:]],axis=0)
        verror=scalar_max(v-m[...,9:12])/(30*s)
        assert gap<2e-5 and verror<2e-5,(row["clip_id"],gap,verror)
        assert not np.any(m[:,1:,13:17]) and not np.any(m[~hv,0,15:17])
        assert not np.any(m[0,0,13:15])
        assert abs(float(P[...,1].min()))/s<2e-6
        G=dec.global_rotations
        f=G[:,meta["heading_carrier_joint"]] @ np.array(meta["u_forward_local"])
        hnorm=np.linalg.norm(f[:,[0,2]],axis=-1)
        heading_boundary_mismatch=hv!=(hnorm>=.05)
        # The saved flag was computed in float64 BEFORE 6D was quantized to float32.
        # Quantization can move a projection a few ulps across the frozen 0.05 threshold.
        assert not np.any(heading_boundary_mismatch & (np.abs(hnorm-.05)>2e-6))
        heading=np.zeros((len(m),2));heading[hv]=f[hv][:,[2,0]]/hnorm[hv,None]
        herror=scalar_max(heading-m[:,0,15:17]);assert herror<2e-5
        # Recompute contact, allowing only float32 threshold-boundary ambiguity.
        speed=np.linalg.norm(m[...,9:12],axis=-1)
        contact=(P[...,1]<=.05*s)&(speed<=.25*s)
        contact[-1]=contact[-2]
        mismatch=contact!=(m[...,12]>.5)
        near=(np.abs(P[...,1]-.05*s)<2e-6*s)|(np.abs(speed-.25*s)<2e-6*s)
        near[-1]=near[-2]
        assert not np.any(mismatch & ~near),(row["clip_id"],"contact")
        # Independent source quaternions: reconstruct full rotations in source parent order.
        with np.load(raw/row["source_npz"],allow_pickle=False) as z:
            names=list(z["names"].astype(str));order=[names.index(n) for n in sk["joint_names"]]
            q=z["anim_local_rot"][:,order];pos=z["anim_local_pos"][:,order];fps=float(z["fps"])
        source_error=None;source_position_error=None
        if fps==30. and len(q)==len(m):
            L=Rotation.from_quat(q[...,[1,2,3,0]].reshape(-1,4)).as_matrix().reshape(q.shape[:-1]+(3,3))
            C=np.array(meta["source_to_canonical_yaw"]);o=np.array(meta["source_origin"])
            L[:,0]=C @ L[:,0];srcG=L.copy();srcP=np.empty_like(pos)
            srcP[:,0]=(pos[:,0]-o) @ C.T
            for j in range(1,len(par)):
                srcG[:,j]=srcG[:,par[j]] @ L[:,j]
                srcP[:,j]=srcP[:,par[j]]+np.einsum("tij,tj->ti",srcG[:,par[j]],pos[:,j])
            srcP[...,1]+=row["qa"]["ground_shift_y"];srcP[...,0]-=oxz[0];srcP[...,2]-=oxz[1]
            source_error=scalar_max(srcG-G)
            source_position_error=float(np.linalg.norm(srcP-P,axis=-1).max()/s)
            assert source_error<2e-5,(row["clip_id"],source_error)
        results.append(dict(clip_id=row["clip_id"],rig_id=rig,fk_direct_norm=gap,velocity_norm=verror,
                            heading_error=herror,contact_threshold_ambiguous_cells=int(mismatch.sum()),
                            heading_threshold_ambiguous_frames=int(heading_boundary_mismatch.sum()),
                            source_rotation_error=source_error,source_dynamic_offset_position_error_norm=source_position_error))
    ref_alignment={name:sha(root/name)==sha(reference/name) for name in ("schema.json","stats/train_block_gains.npz")}
    assert all(ref_alignment.values())
    summary=dict(status="pass",checked_rigs=len(rest_results),checked_clips=len(results),
                 active_reference=str(reference),byte_identical_contract=ref_alignment,
                 active_reference_numeric_probe=reference_probe(reference),
                 max_fk_direct_norm=max(r["fk_direct_norm"] for r in results),
                 max_velocity_norm=max(r["velocity_norm"] for r in results),
                 max_heading_error=max(r["heading_error"] for r in results),
                 max_rest_contract_error=max(r["max_rest_contract_error"] for r in rest_results),
                 nonroot_heading_carriers=sum(r["heading_carrier_joint"]!=0 for r in rest_results),
                 rig_checks=rest_results,clip_checks=results,
                 interpretation="Heading is body orientation, not necessarily travel direction (backward/sideways motion is valid). "
                                "Source-position residual measures the documented fixed-offset approximation; rotations are source-faithful.")
    json_write(root/"analysis/channel_semantics_verification.json",summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('rig_checks','clip_checks')},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path);p.add_argument("--samples",type=int,default=400)
    p.add_argument("--raw-root",type=Path,default=Path("dataset/uniml3d"))
    p.add_argument("--reference",type=Path,default=Path("dataset/ktjd17_pzh312_noik_v2"))
    a=p.parse_args();verify(a.root,a.raw_root,a.reference,a.samples)
