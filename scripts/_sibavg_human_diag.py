"""Standalone sibling-averaged-FK diagnostic on a SMALL set of human clips (avoids the full-val
eval's per-batch memory accumulation: here each clip is processed alone with del+empty_cache).
Reports, for human: last-child rot6d-FK MPJPE vs sibling-AVERAGED-parent rot6d-FK MPJPE (both vs
GT position), the recon sibling 6D dispersion, and the GT dispersion (~0 self-check). No evaluator.
Usage: python scripts/_sibavg_human_diag.py <vqvae_ckpt> <n_human> <num_frames>"""
import sys, importlib.util
import numpy as np
import torch

P = "/scratch/ts1v23/workspace/noKslot_clean"
sys.path.insert(0, P)
CK = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 12
NF = int(sys.argv[3]) if len(sys.argv) > 3 else 288

spec = importlib.util.spec_from_file_location("ev", P + "/scripts/_eval_vqvae_recon_in_evalspace.py")
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
load_vq = ev._import_load_vq_tokenizer()
AnyTopDataset, collate_fn = ev.AnyTopDataset, ev.collate_fn
GraphMotionBatch = ev.GraphMotionBatch
_STD_FLOOR = ev._STD_FLOOR
rec_world = ev.recover_world_positions_torch
rec_fk = ev.recover_rot6d_fk_positions_torch
sib_avg = ev._sibling_avg_rot6d

dev = torch.device("cuda")
vqvae, ta, vck = load_vq(CK, dev)
amp = (ta.get("amp_dtype", "bf16") == "bf16")
root = ta.get("anytop_root") or ta.get("data_root")
ds = AnyTopDataset(data_root=root, split="val", num_frames=NF, max_joints=ta.get("max_joints", 144))
hidx = [i for i in range(len(ds.samples)) if str(ds.samples[i].get("object_type", "")).upper().startswith("HML")][:N]
print(f"  ckpt ep={vck.get('epoch')} | {len(hidx)} human clips | num_frames={NF}", flush=True)

lc_s = lc_c = sa_s = sa_c = 0.0          # last-child / sibling-avg FK: sum, count (masked)
disp_s = disp_c = gdisp_s = gdisp_c = 0.0
for n, i in enumerate(hidx):
    with torch.no_grad():
        raw_coll = collate_fn([ds[i]])
        coll = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in raw_coll.items()}
        batch = GraphMotionBatch.from_collate_dict(coll)
        def _recon():
            enc = vqvae.encode(batch)
            z = vqvae.nearest_residual_ids(enc["h_lat"], enc["token_mask"])["z_snap"]
            return vqvae.decode(z, enc, batch)
        if amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = _recon()
        else:
            out = _recon()
        pred = out["pred_motion"].float()                       # [1,T,J,13] norm
        fmr = out["frame_mask_recovered"].bool()                # [1,T]
        eff = batch.frame_mask & fmr                            # [1,T]
        gtn = batch.anytop_x.permute(0, 3, 1, 2).float()        # [1,T,J,13]
        mean = batch.anytop_mean[:, None].float(); std = batch.anytop_std[:, None].float() + _STD_FLOOR
        gt_raw = gtn * std + mean; rc_raw = pred * std + mean
        vmask = (eff[:, :, None] & batch.joint_mask[:, None, :].bool())   # [1,T,J]
        gt_w = rec_world(gt_raw)
        rc_fk = rec_fk(rc_raw, batch.parent_indices, batch.rest_offsets, batch.joint_mask)
        rc_fk_sa = rec_fk(sib_avg(rc_raw, batch.parent_indices, batch.joint_mask), batch.parent_indices, batch.rest_offsets, batch.joint_mask)
        dlc = torch.where(vmask, (rc_fk - gt_w).norm(dim=-1), torch.zeros_like(vmask, dtype=torch.float))
        dsa = torch.where(vmask, (rc_fk_sa - gt_w).norm(dim=-1), torch.zeros_like(vmask, dtype=torch.float))
        nval = float(vmask.float().sum().item())
        lc_s += float(dlc.sum().item()); lc_c += nval
        sa_s += float(dsa.sum().item()); sa_c += nval
        ds_, dc_ = ev._sibling_dispersion(rc_raw, batch.parent_indices, batch.joint_mask, eff)
        gs_, gc_ = ev._sibling_dispersion(gt_raw, batch.parent_indices, batch.joint_mask, eff)
        disp_s += ds_; disp_c += dc_; gdisp_s += gs_; gdisp_c += gc_
        print(f"   clip{n}: last-child FK={1000*float(dlc.sum()/max(nval,1)):.1f}mm  sib-avg FK={1000*float(dsa.sum()/max(nval,1)):.1f}mm", flush=True)
        del out, pred, gt_raw, rc_raw, gt_w, rc_fk, rc_fk_sa, dlc, dsa, batch, coll
    torch.cuda.empty_cache()

print(f"  ===== HUMAN ({len(hidx)} clips, {NF}f) =====", flush=True)
print(f"  last-child rot6d-FK MPJPE = {1000*lc_s/max(lc_c,1):.1f} mm", flush=True)
print(f"  sibling-AVG rot6d-FK MPJPE = {1000*sa_s/max(sa_c,1):.1f} mm", flush=True)
print(f"  recon sibling 6D dispersion = {disp_s/max(disp_c,1):.5f}  (GT dispersion = {gdisp_s/max(gdisp_c,1):.2e}, ~0 expected)", flush=True)
print("  DONE", flush=True)
