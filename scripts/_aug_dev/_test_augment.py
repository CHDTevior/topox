"""Invariant tests for the skeleton-robustness augmentation (unit level on base items + integration through
InContextPairs/collate/ktjd_prep and a CPU cfm_loss smoke). usage: python scripts/_aug_dev/_test_augment.py (snapshots/logs under runs/_aug_dev/)"""
import sys, os, json, importlib.util
import numpy as np, torch
sys.path.insert(0, os.getcwd())
from src.data.ktjd17_incontext import Ktjd17Base, ktjd17_split_names
from src.data.incontext_pairs import InContextPairs, collate, _graph_v2_tables
from src.data.ktjd17_augment import AugConfig, make_transform, apply_motion, apply_semantics, hop_matrix
from src.data.anytop_dataset import _STD_FLOOR
from src.data.ktjd17.codec import decode_column_cont6d, fk_from_global_rotations

ROOT, CUT = "dataset/ktjd17_pzh312_noik_v2", "configs/pilot_animal_only_exclusions.json"
base = Ktjd17Base(ROOT, caption_emb_cache="data/noik_caption_llm2vec_v1", joint_semantics="data/joint_semantics_llm2vec_pzh312_v1.npz",
                  texts_json="data/noik_pzh312_motion_texts_v1.json", percell_stats="data/noik_norm_stats_v2.npz",
                  exclude_clips=CUT, random_caption=False, normalization="percell")
names = ktjd17_split_names(ROOT, exclude=CUT)
cfg = AugConfig(p=1.0, drop_max_frac=0.4, rest_deg=15.0, sem_noise=0.1, sem_drop_p=0.2, stats_logsd=0.2, stats_shift=0.2)
cfg_tips = AugConfig(p=1.0, drop_max_frac=0.4, drop_mode="tips", rest_deg=15.0, sem_noise=0.1, sem_drop_p=0.2, stats_logsd=0.2, stats_shift=0.2)
rng = np.random.default_rng(123)
fails = []
def check(cond, msg):
    if not cond: fails.append(msg); print("FAIL:", msg)

# ---------------- unit level: 10 random base items, both drop modes ----------------
idxs = rng.choice(len(base), size=10, replace=False)
for mode_cfg in (cfg, cfg_tips):
  fk_gap_new, fk_gap_old, n_drop_tot, n_reparented = [], [], 0, 0
  for bi in idxs:
      it = base[int(bi)]
      J0, T = int(it["num_joints"]), int(it["num_frames"])
      x = np.asarray(it["anytop_x"])[:J0, :, :T].transpose(2, 0, 1)            # [T,J,18]
      mu0, sd0 = np.asarray(it["anytop_mean"])[:J0, :17], np.asarray(it["anytop_std"])[:J0, :17]
      cv0 = base.static_masks(it["object_type"])["channel_valid"][:J0]
      raw0 = x[..., :17].astype(np.float64) * (sd0 + _STD_FLOOR) + mu0
      contact = (raw0[..., 12] > 0.5).any(0)
      sk = base.skeleton(it["object_type"])
      par0, P0, R0, O0 = (np.asarray(sk["parents"])[:J0], np.asarray(sk["P_rest_global"], dtype=np.float64)[:J0],
                          np.asarray(sk["R_rest_global"], dtype=np.float64)[:J0], np.asarray(sk["offset_parent_local"], dtype=np.float64)[:J0])
      tr = make_transform(rng, mode_cfg, parents=par0, P_rest_global=P0, R_rest_global=R0, offset_parent_local=O0,
                          channel_valid=cv0, mu=mu0, sd=sd0, contact_joints=contact)
      keep = tr.keep; Jn = len(keep); n_drop_tot += J0 - Jn
      check(keep[0] == 0 and np.all(np.diff(keep) > 0), "root kept / keep sorted")
      check(all(int(j) in set(keep.tolist()) for j in np.where(contact)[0]), "contact joints protected")
      check(tr.parents[0] == -1 and np.all(tr.parents[1:] < np.arange(1, Jn)) and np.all(tr.parents[1:] >= 0), "FK order of new parents")
      xn = apply_motion(x, tr)
      check(xn.shape == (T, Jn, 18), "shape")
      check(np.array_equal(xn[..., 17], x[:, keep, 17]), "plane 17 copied")
      rawn = xn[..., :17].astype(np.float64) * (tr.sd + _STD_FLOOR) + tr.mu
      for ch in (slice(0, 3), slice(9, 13)):
          d = np.abs(rawn[..., ch] - raw0[:, keep, ch]).max()
          check(d < 2e-4, f"positions/velocity/contact unchanged on kept rows (max diff {d:.2e})")
      rows = np.where(tr.rot_rows)[0]
      G0 = decode_column_cont6d(raw0[:, keep[rows], 3:9]) @ R0[keep[rows]][None]
      Gn = decode_column_cont6d(rawn[:, rows, 3:9]) @ tr.R_rest[rows][None]
      check(np.abs(G0 - Gn).max() < 1e-4, f"global rotations invariant under rest-convention change (max {np.abs(G0-Gn).max():.2e})")
      # stats perturbed only on channel_valid cells 0:13
      changed = (tr.mu != tr.mu0) | (tr.sd != tr.sd0)
      check(not changed[:, 13:17].any(), "root cells 13:17 stats untouched")
      check(not changed[~tr.channel_valid].any(), "invalid cells stats untouched")
      # FK with the new skeleton vs direct positions (kept joints only; world = q_position + smooth root)
      def world(raw, root_rows):
          w = raw[..., 0:3].copy(); w[..., 0] += raw[:, 0:1, 13]; w[..., 2] += raw[:, 0:1, 14]; return w
      w0, wn = world(raw0, 0), world(rawn, 0)
      # FK needs all rotations: use rot_rows only rigs (skip rigs with invalid rotation rows)
      if tr.rot_rows.all() and cv0[:, 3:9].all():
          fk0 = fk_from_global_rotations(par0, w0[:, 0], decode_column_cont6d(raw0[..., 3:9]) @ R0[None], O0)
          fkn = fk_from_global_rotations(tr.parents, wn[:, 0], Gn, tr.offsets)
          bl = np.linalg.norm(O0[1:], axis=-1).mean()
          fk_gap_old.append(np.abs(fk0 - w0).mean() / bl); fk_gap_new.append(np.abs(fkn - wn).mean() / bl)
          # joints whose whole ancestor chain is kept must have identical FK
          keep_set = set(keep.tolist()); chain_ok = []
          for n, j in enumerate(keep):
              q, ok = int(par0[j]), True
              while q >= 0:
                  ok &= q in keep_set; q = int(par0[q])
              chain_ok.append(ok)
          chain_ok = np.array(chain_ok); n_reparented += int((~chain_ok).sum())
          d = np.abs(fkn[:, chain_ok] - fk0[:, keep[chain_ok]]).max()
          check(d < 1e-3 * bl, f"FK identical on fully-kept chains (max diff {d:.2e}, bl {bl:.3f})")
      # rest demo
      rest = base.rest_frame_normalized(it["object_type"])[:J0]
      rn = apply_motion(rest, tr)
      rraw = rn[:, :17].astype(np.float64) * (tr.sd + _STD_FLOOR) + tr.mu
      Qt = decode_column_cont6d(rraw[rows, 3:9])
      check(np.abs(Qt - np.swapaxes(tr.Q[rows], -1, -2)).max() < 1e-5, "rest demo rotations = Q^T")
      rest_raw0 = rest[:, :17].astype(np.float64) * (sd0 + _STD_FLOOR) + mu0
      check(np.abs(rraw[:, 0:3] - rest_raw0[keep, 0:3]).max() < 2e-4, "rest demo positions unchanged")
      # struct tables consistent with the new tree (internal assert) + geodesic
      geo = hop_matrix(tr.parents); feats, ud = _graph_v2_tables(tr.parents, tr.offsets, geo)
      check(feats.shape == (Jn, 8) and ud.shape == (Jn, Jn, 2), "graph-v2 tables")
      # semantics
      sem = np.asarray(it["joint_semantics"])[:J0]; sn = apply_semantics(sem, tr, rng)
      check(sn.shape == (Jn, sem.shape[1]), "sem shape")
      if tr.sem_drop: check(not sn.any(), "sem drop zeroes")
      else:
          rel = np.linalg.norm(sn - sem[keep], axis=1) / np.linalg.norm(sem[keep], axis=1)
          check(0.05 < rel.mean() < 0.2, f"sem noise magnitude ~0.1 (got {rel.mean():.3f})")
  print(f"[unit:{mode_cfg.drop_mode}] dropped {n_drop_tot} joints over 10 items; re-parented joints {n_reparented}; "
        f"FK-pose gap (bl) original {np.mean(fk_gap_old) if fk_gap_old else float('nan'):.4f} -> augmented {np.mean(fk_gap_new) if fk_gap_new else float('nan'):.4f} "
        f"(n={len(fk_gap_new)})")
  if mode_cfg.drop_mode == "tips" and fk_gap_new:
      check(max(fk_gap_new) < 1e-3, f"tips mode keeps FK exact (max gap {max(fk_gap_new):.2e} bl)")

# ---------------- codex r1 regressions ----------------
# (1) the cached rest frame must stay pristine whatever the demo path does (in-place clamp used to mutate it)
from src.data.incontext_pairs import REST_DEMO_CLAMP
rig_big = "PZ_Blue_Wildebeest_Female"
pristine = base.rest_frame_normalized(rig_big).copy()
check(np.abs(pristine[:, :17]).max() > REST_DEMO_CLAMP, "test rig has a rest cell beyond the clamp (precondition)")
ds_mix = InContextPairs(base, names["train"], names["train"], object_types=[rig_big], demo_frames=1, target_frames=240,
                        balance_skeletons=True, seed=3, emit_fk_fields=True, emit_graph_v2=True, demo_rest=True,
                        augment=AugConfig(p=0.5, drop_max_frac=0.3, rest_deg=10.0))
_ = [ds_mix[i] for i in range(6)]
check(np.array_equal(base.rest_frame_normalized(rig_big), pristine), "cached rest frame untouched after mixed aug / non-aug items")
# (3) degenerate 6D cells (zero first column; parallel columns) must pass through untouched, others rotate
it0 = base[int(idxs[0])]; J0 = int(it0["num_joints"]); sk0 = base.skeleton(it0["object_type"])
rest0 = base.rest_frame_normalized(it0["object_type"])[:J0].copy()
cv = base.static_masks(it0["object_type"])["channel_valid"][:J0]
mu0, sd0 = np.asarray(it0["anytop_mean"])[:J0, :17], np.asarray(it0["anytop_std"])[:J0, :17]
raw = rest0[:, :17].astype(np.float64) * (sd0 + _STD_FLOOR) + mu0
rows_ok = np.where(cv[:, 3:9].all(1))[0]
zj, pj = rows_ok[1], rows_ok[2]
raw[zj, 3:9] = 0.0; raw[pj, 3:9] = np.array([0.6, 0.8, 0.0, 0.6, 0.8, 0.0])           # zero / parallel columns
x_deg = rest0.copy(); x_deg[:, :17] = ((raw - mu0) / (sd0 + _STD_FLOOR)).astype(np.float32)
trd = make_transform(np.random.default_rng(5), AugConfig(p=1.0, rest_deg=20.0), parents=np.asarray(sk0["parents"])[:J0],
                     P_rest_global=np.asarray(sk0["P_rest_global"])[:J0], R_rest_global=np.asarray(sk0["R_rest_global"])[:J0],
                     offset_parent_local=np.asarray(sk0["offset_parent_local"])[:J0], channel_valid=cv, mu=mu0, sd=sd0,
                     contact_joints=np.zeros(J0, bool))
try:
    xd = apply_motion(x_deg, trd)
    rawd = xd[:, :17].astype(np.float64) * (trd.sd + _STD_FLOOR) + trd.mu
    check(np.abs(rawd[zj, 3:9] - raw[zj, 3:9]).max() < 1e-4 and np.abs(rawd[pj, 3:9] - raw[pj, 3:9]).max() < 1e-4, "degenerate 6D cells kept as served")
    other = rows_ok[3]
    check(np.abs(rawd[other, 3:9] - raw[other, 3:9]).max() > 1e-3, "non-degenerate cell rotated")
except Exception as e:
    check(False, f"degenerate 6D raised {type(e).__name__}: {e}")
# (4) statistics strengths: unsupported values refused; supported maximum never vanishes
try:
    AugConfig(p=1.0, stats_logsd=10.0); check(False, "stats_logsd=10 accepted")
except ValueError:
    pass
worst = np.inf
for k in range(200):
    trs = make_transform(np.random.default_rng(1000 + k), AugConfig(p=1.0, stats_logsd=1.0, stats_shift=3.0), parents=np.asarray(sk0["parents"])[:J0],
                         P_rest_global=np.asarray(sk0["P_rest_global"])[:J0], R_rest_global=np.asarray(sk0["R_rest_global"])[:J0],
                         offset_parent_local=np.asarray(sk0["offset_parent_local"])[:J0], channel_valid=cv, mu=mu0, sd=sd0,
                         contact_joints=np.zeros(J0, bool))
    worst = min(worst, float((trs.sd.astype(np.float64) + _STD_FLOOR).min()))
    xs = apply_motion(rest0, trs); check(np.isfinite(xs).all(), "finite output at max strength")
print(f"[stats] smallest effective scale over 200 max-strength transforms: {worst:.3e}")
check(worst > 1e-5, "effective scale stays positive at max strength")

# ---------------- integration ----------------
ds = InContextPairs(base, names["train"], names["train"], demo_frames=1, target_frames=240, balance_skeletons=True, seed=0,
                    emit_fk_fields=True, emit_graph_v2=True, identity_p=0.0, emit_ref_text=False, demo_rest=True, augment=cfg)
items = [ds[i] for i in range(8)]
for it in items:
    J = it["n_joints"]
    check(it["channel_valid"].shape == (J, 17), "item channel_valid")
    check(it["parents"].shape[0] == J and it["rest_offsets"].shape == (J, 3) and it["R_rest_global"].shape == (J, 3, 3), "fk fields sized to J'")
    check(it["struct_feats"].shape == (J, 8) and it["updown"].shape == (J, J, 2) and it["joint_sem"].shape[0] == J, "graph/sem sized to J'")
    check(it["x"].shape[1] == J and it["geodesic"].shape == (J, J), "x/geodesic sized to J'")
b = collate(items[:4])
check(b["channel_valid"].shape == (4, b["x"].shape[2], 17), "collated channel_valid")
spec = importlib.util.spec_from_file_location("trainer", "scripts/train_v2_incontext.py"); tr_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(tr_mod)
lut = tr_mod.ktjd_channel_lut(base)
calib = json.load(open("configs/pilot_animal_r1acc_gamma_calibration_b16_v2.json"))
gammas = {k: float(v) for k, v in calib["gammas"].items()}
x17, kt = tr_mod.ktjd_prep(b, lut, gammas)
check(torch.equal(kt["channel_valid"], b["channel_valid"]), "ktjd_prep uses per-sample channel_valid")
from src.models.v2.dit_motion import InContextMotionDiT, cfm_loss
torch.manual_seed(0)
model = InContextMotionDiT(in_ch=17, dim=64, depth=1, n_heads=2, use_struct_feats=True, use_dir_bias=True, qk_norm=True)
loss = cfm_loss(model, x17, is_target=b["is_target"], valid=b["valid"], t_sampler="uniform", v_space=True, sigma_min=0.2, huber_delta=10.0,
                gamma_fk=0.07, gamma_vel=0.01, gamma_lock=0.01, gamma_acc=1.0, fk_pack=tr_mod.fk_pack_of(b), **kt, **tr_mod.cond_of(b))
check(torch.isfinite(loss).item(), f"cfm_loss finite ({loss.item()})")
loss.backward(); check(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()), "grads finite")
print("[integration] loss", float(loss), "| Jm", b["x"].shape[2], "| n_joints", b["n_joints"].tolist())
print("RESULT:", "PASS" if not fails else f"FAIL ({len(fails)})")
sys.exit(1 if fails else 0)
