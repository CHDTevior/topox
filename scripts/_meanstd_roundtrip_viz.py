"""mean/std round-trip sanity viz: 5 human GT clips, GT(denorm) vs norm->denorm round-trip.
Reuses the render helpers + the SAME _recover_world_positions used by the recon QA. Reports
the numerical round-trip error (should be ~machine eps if mean/std + the denorm formula are
clean). Identical panels + ~0 error => the per-skeleton mean/std pipeline preserves GT exactly.
NO model, NO GPU — pure data + recovery + PIL render."""
import sys, importlib.util
from pathlib import Path
import numpy as np

P = "/scratch/ts1v23/workspace/noKslot_clean"
sys.path.insert(0, P)
spec = importlib.util.spec_from_file_location("avr", P + "/scripts/animate_vqvae_recon_large.py")
avr = importlib.util.module_from_spec(spec); spec.loader.exec_module(avr)
from src.data.anytop_dataset import AnyTopDataset, _STD_FLOOR, _recover_world_positions

ds = AnyTopDataset(data_root=P + "/data/animo4d_anytop_clean_L4_safe_plus_humanml3d",
                   split="val", num_frames=288, max_joints=144)
hidx = [i for i in range(len(ds.samples))
        if str(ds.samples[i].get("object_type", "")).upper().startswith("HML")][:5]
print(f"  human clips: {hidx}")
font = avr.get_font(13); cell = (900, 760)
outdir = Path(P + "/runs/vqvae_L4safeHuman_C72_J144_d512_Q4_n8192_b16g64_300ep_curric50to60_seed42/qa_meanstd_roundtrip_fk")
outdir.mkdir(parents=True, exist_ok=True)
recover_fk = avr.recover_from_bvh_rot_np   # rot6d(ch3:9) -> forward kinematics (official)

for n, i in enumerate(hidx):
    item = ds[i]
    J = int(item["num_joints"]); T = int(item["num_frames"])
    stored = np.asarray(item["anytop_x"]).transpose(2, 0, 1)[:T, :J, :].astype(np.float64)  # [T,J,13] normalized
    std = np.asarray(item["anytop_std"])[:J].astype(np.float64)
    mean = np.asarray(item["anytop_mean"])[:J].astype(np.float64)
    raw = stored * (std[None] + _STD_FLOOR) + mean[None]            # denorm -> GT raw motion
    norm2 = (raw - mean[None]) / (std[None] + _STD_FLOOR)           # re-normalize with same mean/std
    rt = norm2 * (std[None] + _STD_FLOOR) + mean[None]              # round-trip denorm
    rt_err = float(np.abs(rt - raw).max())                          # round-trip fidelity (should be ~0)
    renorm_err = float(np.abs(norm2 - stored).max())               # re-norm vs stored norm (~0)
    nan_ct = int(np.isnan(std).sum() + np.isnan(mean).sum())
    zero_std = int((std == 0).sum())                               # zero-std channels (floor-handled)
    parents = np.asarray([int(p) for p in item["parent_indices"][:J]], dtype=int)
    offsets = np.asarray(item["rest_offsets"])[:J].astype(np.float64)
    raw_w = recover_fk(raw, parents, offsets).astype(np.float64)    # rot6d->FK world pos of GT raw
    rt_w = recover_fk(rt, parents, offsets).astype(np.float64)      # rot6d->FK world pos of round-trip
    gt_ric = _recover_world_positions(raw).astype(np.float64)       # RIC truth (for FK-vs-RIC floor)
    fk_vs_ric = float(np.linalg.norm(raw_w - gt_ric, axis=-1).mean())  # human FK-route floor (~8.6mm avg)
    height = float(raw_w[..., 1].max() - raw_w[..., 1].min())       # scale sanity (~1.7 if meters)
    for arr in (raw_w, rt_w):
        arr[..., 1] -= arr[..., 1].min()
    idxs = avr.sample_indices(T, 48)
    ps = []
    for arr in (raw_w, rt_w):
        c = arr.copy(); roots = c[:, 0].copy()
        c[..., 0] -= roots[:, None, 0]; c[..., 2] -= roots[:, None, 2]
        ps += [c[k] for k in idxs]
    transform = avr.compute_transform(ps, cell, 0.06, 1.15)
    frames = [avr.make_frame_2panel(raw_w, rt_w, parents, k, transform, cell, 3, 5, True, font,
                                    titles=("GT rot6d->FK", "ROUND-TRIP rot6d->FK"))
              for k in idxs]
    op = outdir / f"human_clip{n}_meanstd_roundtrip_FK.gif"
    frames[0].save(op, save_all=True, append_images=frames[1:], duration=83, loop=0, optimize=True)
    print(f"  human clip{n} (J={J} T={T}): roundtrip_max_err={rt_err:.3e} renorm_vs_stored_max={renorm_err:.3e} "
          f"FK_vs_RIC_floor={fk_vs_ric:.4f} nan={nan_ct} zero_std={zero_std} GT_height={height:.3f} -> {op.name}")
print("  DONE")
