"""Smoke gates (plan §8) for loss_mode=anytop13_world_rot6d_fk, via run_loss on
a REAL VAE forward. Gates:
  G2: default loss_mode="anytop13" total == direct compute_total_loss_13ch (no
      world/fk/traj keys).
  G3: anytop13_world_rot6d_fk returns finite world/fk/traj/gt_fk_mismatch/total.
  G4: backward -> grad(pred_motion[:, :, 1:, 3:9]) > 0 (non-root rot6d, the FK
      signature — the KEY difference from world_geometry).
  G5: grad(pred_motion[:, :, 1:, 0:3]) > 0 (world/RIC pose route still supervised).
  G7: gt_fk_mismatch logged, NOT asserted zero (it's the dataset's route floor).
(G1 numpy parity already PASS in _smoke_rot6d_fk_torch.py; G6 frame_mask_recovered
 is what run_loss passes as effective_frame_mask.)

Run on rose11: python scripts/_smoke_world_rot6d_fk_wiring.py
"""
import sys
import importlib.util
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.anytop_dataset import AnyTopDataset, collate_fn  # noqa
from src.models.graph_salad import GraphMotionBatch, GraphMotionVAE, compute_total_loss_13ch  # noqa
spec = importlib.util.spec_from_file_location("tgv", str(ROOT / "scripts" / "train_graph_vae.py"))
tgv = importlib.util.module_from_spec(spec); spec.loader.exec_module(tgv)
run_loss = tgv.run_loss

BASE = str(ROOT / "runs/_baseline_cleanL2_ep34_for_p1diag_compare/best_recon_model.pt")
LW = {"pos": 1.0, "rot": 1.0, "vel": 1.0, "contact": 0.1, "kl": 1e-3, "pool_aux": 0.5}


def build_vae(dev):
    ck = torch.load(BASE, map_location="cpu", weights_only=True); ta = ck["args"]
    vae = GraphMotionVAE(
        pool_type=ta["pool_type"], pool_tau=ta.get("pool_tau"),
        d_model=ta["d_model"], n_heads=ta["n_heads"], d_ff=ta["d_ff"],
        n_graph_layers=ta["n_graph_layers"], n_enc_temporal_layers=ta["n_enc_temporal_layers"],
        n_cross_layers=ta["n_cross_layers"], n_dec_temporal_layers=ta["n_dec_temporal_layers"],
        n_treeik_layers=ta["n_treeik_layers"], max_coarse=ta["max_coarse"],
        local_radius=ta["local_radius"], temporal_stride=ta["temporal_stride"],
        temporal_kernel=ta["temporal_kernel"], dropout=ta["dropout"], feat_mode="anytop13",
        attn_mode=ta.get("attn_mode") or "scalar", decoder_mode=ta.get("decoder_mode") or "unpool_identity",
        n_graph_temporal_layers=ta.get("n_graph_temporal_layers", 4)).to(dev)
    vae.load_state_dict(ck["model_state_dict"], strict=True)
    vae.encoder.use_name_embed = bool(ta.get("use_name_embed", False)); vae.eval()
    return vae


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = AnyTopDataset(split="val", val_frac=0.05, seed=42,
                       data_root=str(ROOT / "data/anytop_planet_zoo_clean_L2"),
                       num_frames=64, max_joints=144, caption_emb_cache=None)
    d = collate_fn([ds[i] for i in range(4)])
    d = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in d.items()}
    batch = GraphMotionBatch.from_collate_dict(d)
    vae = build_vae(dev)
    out = vae(batch, sample=False)
    eff_fm = out["frame_mask_recovered"]
    gt_motion = batch.anytop_x.permute(0, 3, 1, 2).contiguous()

    # G2: default == direct
    ref = compute_total_loss_13ch(
        pred_motion=out["pred_motion"], gt_motion=gt_motion,
        foot_contact_per_joint=batch.foot_contact_per_joint, mu=out["mu"], logvar=out["logvar"],
        pool_aux_outputs=out["pool_aux_outputs"], joint_mask=batch.joint_mask, frame_mask=eff_fm,
        coarse_mask=out["coarse_mask"], frame_mask_lat=out["frame_mask_lat"], weights=LW)
    deflt = run_loss(out, batch, "anytop13", LW, eff_fm, dev, loss_mode="anytop13",
                     w_world=0.25, w_fk=0.25, w_traj=0.1)
    g2 = bool(torch.equal(ref["total"], deflt["total"])) and ("fk" not in deflt and "world" not in deflt)
    print(f"G2 default==direct & no geo keys: {g2} (ref={ref['total'].item():.6f} def={deflt['total'].item():.6f})", flush=True)

    # G3: new mode finite keys
    wg = run_loss(out, batch, "anytop13", LW, eff_fm, dev, loss_mode="anytop13_world_rot6d_fk",
                  w_world=0.25, w_fk=0.25, w_traj=0.1)
    keys = ("world", "fk", "traj", "gt_fk_mismatch", "total")
    g3 = all(k in wg and torch.isfinite(wg[k]).all() for k in keys)
    print(f"G3 keys finite: {g3}  world={wg['world'].item():.4f} fk={wg['fk'].item():.4f} "
          f"traj={wg['traj'].item():.4f} gt_fk_mismatch={wg['gt_fk_mismatch'].item():.4f} "
          f"total={wg['total'].item():.4f}", flush=True)

    # G4/G5: grads. re-forward with grad, hook pred_motion.
    out2 = vae(batch, sample=False)
    pm = out2["pred_motion"]; pm.retain_grad()
    wg2 = run_loss(out2, batch, "anytop13", LW, out2["frame_mask_recovered"], dev,
                   loss_mode="anytop13_world_rot6d_fk", w_world=0.25, w_fk=0.25, w_traj=0.1)
    wg2["total"].backward()
    g = pm.grad
    nr_rot = float(g[:, :, 1:, 3:9].abs().sum().item())   # G4
    nr_pos = float(g[:, :, 1:, 0:3].abs().sum().item())   # G5
    g4 = nr_rot > 0; g5 = nr_pos > 0
    print(f"G4 nonroot rot6d(3:9) grad>0: {g4} ({nr_rot:.4e}) [FK signature]", flush=True)
    print(f"G5 nonroot pos(0:3) grad>0: {g5} ({nr_pos:.4e}) [world/RIC route]", flush=True)

    # G7: gt_fk_mismatch is logged, not zero (informational)
    print(f"G7 gt_fk_mismatch (route floor, NOT asserted 0): {wg['gt_fk_mismatch'].item():.4f}", flush=True)

    ok = g2 and g3 and g4 and g5
    print(f"\nWIRING_SMOKE {'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
