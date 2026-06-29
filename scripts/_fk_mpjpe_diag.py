"""Leak-free FK/position MPJPE diagnostic: processes clips ONE AT A TIME (no_grad + del +
empty_cache per clip) to avoid the full-val eval's per-batch GPU accumulation (OOMs at 288f).
Per species (human=HML*, animal=PZ_*), over N clips, reports:
  position MPJPE  = recon RIC->world (rc_w) vs GT RIC->world (gt_w)
  rot6d-FK MPJPE  = recon rot6d->FK (rc_fk) vs GT positions (gt_w)   <- the user's ask
  FK-floor        = GT rot6d->FK (gt_fk) vs GT positions (gt_w), recon-independent baseline
All ABSOLUTE world-pos, root-incl, de-norm meters. Usage: python _fk_mpjpe_diag.py <ckpt> <n_per> <num_frames>"""
import sys, importlib.util
import torch

P = "/scratch/ts1v23/workspace/noKslot_clean"
sys.path.insert(0, P)
CK = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 80
NF = int(sys.argv[3]) if len(sys.argv) > 3 else 288
CONT = any(a in ("continuous", "--continuous") for a in sys.argv[4:])   # pre-VQ h_lat (skip RVQ snap); accepts positional OR --continuous

spec = importlib.util.spec_from_file_location("ev", P + "/scripts/_eval_vqvae_recon_in_evalspace.py")
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
load_vq = ev._import_load_vq_tokenizer()
AnyTopDataset, collate_fn = ev.AnyTopDataset, ev.collate_fn
GraphMotionBatch = ev.GraphMotionBatch
_STD_FLOOR = ev._STD_FLOOR
rec_world = ev.recover_world_positions_torch
rec_fk = ev.recover_rot6d_fk_positions_torch

def _jitter(pos, eff, jmask):
    """Mean per-joint acceleration magnitude ||x[t+1]-2x[t]+x[t-1]|| over frames where t-1,t,t+1
    are ALL valid (eff) and the joint is valid. Units = de-norm meters/frame^2. Higher = shakier.
    Returns (sum, count). This is the temporal-jitter measure MPJPE (per-frame avg) is blind to."""
    accel = pos[:, 2:] - 2 * pos[:, 1:-1] + pos[:, :-2]          # [1,T-2,J,3]
    fm = (eff[:, :-2] & eff[:, 1:-1] & eff[:, 2:])               # [1,T-2] all-3-frames valid
    m = fm[:, :, None] & jmask[:, None, :]                       # [1,T-2,J]
    z = torch.zeros_like(m, dtype=torch.float)
    s = float(torch.where(m, accel.norm(dim=-1), z).sum().item())
    return s, float(m.float().sum().item())


def _speed(pos, eff, jmask):
    """Mean per-joint VELOCITY magnitude ||x[t+1]-x[t]|| over frames where t,t+1 valid + joint
    valid. fk_speed_ratio = recon-FK speed / GT speed detects OVER-smoothing (motion damped) vs
    fk_accel_ratio (jitter). Returns (sum,count)."""
    vel = pos[:, 1:] - pos[:, :-1]                               # [1,T-1,J,3]
    fm = (eff[:, 1:] & eff[:, :-1])                              # [1,T-1] both frames valid
    m = fm[:, :, None] & jmask[:, None, :]
    z = torch.zeros_like(m, dtype=torch.float)
    return float(torch.where(m, vel.norm(dim=-1), z).sum().item()), float(m.float().sum().item())


dev = torch.device("cuda")
vqvae, ta, vck = load_vq(CK, dev)
amp = (ta.get("amp_dtype", "bf16") == "bf16")
root = ta.get("anytop_root") or ta.get("data_root")
ds = AnyTopDataset(data_root=root, split="val", num_frames=NF, max_joints=ta.get("max_joints", 144))
print(f"  ckpt ep={vck.get('epoch')} | num_frames={NF} | n_per_species={N} | recon={'CONTINUOUS(pre-VQ)' if CONT else 'QUANTIZED'}", flush=True)

def idxs_for(prefix):
    return [i for i in range(len(ds.samples)) if str(ds.samples[i].get("object_type", "")).upper().startswith(prefix)][:N]

acc = {}  # species -> dict of sum/cnt for pos, fk, floor
for sp, pref in [("human", "HML"), ("animal", "PZ_")]:
    s_pos = s_fk = s_fl = c = 0.0
    j_gtw = j_rcw = j_gtfk = j_rcfk = jc = 0.0   # accel(jitter) sums + count
    v_gtw = v_rcfk = vc = 0.0                     # velocity sums + count (for speed ratio)
    sel = idxs_for(pref)
    for i in sel:
        with torch.no_grad():
            coll = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in collate_fn([ds[i]]).items()}
            batch = GraphMotionBatch.from_collate_dict(coll)
            def _recon():
                enc = vqvae.encode(batch)
                if CONT:
                    return vqvae.decode(enc["h_lat"], enc, batch)   # pre-VQ continuous (no RVQ snap)
                z = vqvae.nearest_residual_ids(enc["h_lat"], enc["token_mask"])["z_snap"]
                return vqvae.decode(z, enc, batch)
            if amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = _recon()
            else:
                out = _recon()
            pred = out["pred_motion"].float()
            eff = batch.frame_mask & out["frame_mask_recovered"].bool()
            gtn = batch.anytop_x.permute(0, 3, 1, 2).float()
            mean = batch.anytop_mean[:, None].float(); std = batch.anytop_std[:, None].float() + _STD_FLOOR
            gt_raw = gtn * std + mean; rc_raw = pred * std + mean
            vmask = (eff[:, :, None] & batch.joint_mask[:, None, :].bool())
            z0 = torch.zeros_like(vmask, dtype=torch.float)
            gt_w = rec_world(gt_raw); rc_w = rec_world(rc_raw)
            gt_fk = rec_fk(gt_raw, batch.parent_indices, batch.rest_offsets, batch.joint_mask)
            rc_fk = rec_fk(rc_raw, batch.parent_indices, batch.rest_offsets, batch.joint_mask)
            nval = float(vmask.float().sum().item())
            s_pos += float(torch.where(vmask, (rc_w - gt_w).norm(dim=-1), z0).sum().item())
            s_fk += float(torch.where(vmask, (rc_fk - gt_w).norm(dim=-1), z0).sum().item())
            s_fl += float(torch.where(vmask, (gt_fk - gt_w).norm(dim=-1), z0).sum().item())
            c += nval
            jm = batch.joint_mask.bool()
            for tgt, key in ((gt_w, "gtw"), (rc_w, "rcw"), (gt_fk, "gtfk"), (rc_fk, "rcfk")):
                js, jn = _jitter(tgt, eff, jm)
                if key == "gtw": j_gtw += js
                elif key == "rcw": j_rcw += js
                elif key == "gtfk": j_gtfk += js
                else: j_rcfk += js; jc += jn   # jc counted once (same mask for all)
            vg, vn = _speed(gt_w, eff, jm); v_gtw += vg; vc += vn
            vr, _ = _speed(rc_fk, eff, jm); v_rcfk += vr
            del out, pred, gt_raw, rc_raw, gt_w, rc_w, gt_fk, rc_fk, batch, coll
        torch.cuda.empty_cache()
    acc[sp] = (1000 * s_pos / max(c, 1), 1000 * s_fk / max(c, 1), 1000 * s_fl / max(c, 1), len(sel))
    print(f"  {sp} ({len(sel)} clips): position={acc[sp][0]:.1f}mm  rot6d-FK={acc[sp][1]:.1f}mm  FK-floor={acc[sp][2]:.1f}mm", flush=True)
    jd = max(jc, 1)
    print(f"  {sp} JITTER (mean accel, mm/frame^2): GT-pos={1000*j_gtw/jd:.2f}  recon-pos={1000*j_rcw/jd:.2f}  GT-FK={1000*j_gtfk/jd:.2f}  recon-FK={1000*j_rcfk/jd:.2f}  | recon-FK/GT-pos ratio={j_rcfk/max(j_gtw,1e-9):.1f}x", flush=True)
    print(f"  {sp} RATIOS: fk_accel_ratio={j_rcfk/max(j_gtw,1e-9):.2f}x (jitter; >1=shaky)  fk_speed_ratio={v_rcfk/max(v_gtw,1e-9):.2f}x (velocity; <1=over-smoothed/damped, ~1=ok)", flush=True)
print("  DONE", flush=True)
