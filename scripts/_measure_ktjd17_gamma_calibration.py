#!/usr/bin/env python3
"""KTJD-17 gamma calibration -- versioned artifact builder (codex round-S0 item 3).

PREREGISTERED PROTOCOL (fixed BEFORE measurement; every field is recorded in the artifact):
  cohort          ALL train-split targets, ONE full pass, clip-balanced
                  (InContextPairs balance_skeletons=False, seed 0, num_workers 4)
  windows         exactly the training crop policy: demo = the run's demo condition (DEMO_REST=1:
                  1-frame rest pose; DEMO_REST=0: random re-based DEMO_FRAMES-frame window),
                  target = head window; energies measured on TARGET frames only
                  (that is where the loss lives). Crop re-base ACTIVE (post-fix).
  space           normalized model space (s_rig + frozen train gains), v_space=False,
                  huber_delta=HUBER -- the mechanism check must run through the SAME
                  objective the run trains on, or the measured shares describe a
                  different loss than the one the gammas are frozen into
                  t_sampler recorded as "uniform" (energy is t-independent: E_i = mean x1^2)
  mask            the FULL effective-validity mask, byte-identical semantics to cfm_loss:
                  target-frame & frame_valid & joint_valid & channel_valid & heading gate
  target shares   KIMODO Eq.1 IMPLIED PROFILE (user 2026-08-20: "kimodo-like loss, 能一致尽量
                  一致"). Under Kimodo's per-element-unit-energy normalization, Eq.1's gammas
                  (10/2/10/10/3/4) imply gradient-share ~ gamma^2: r^p 30.4% r^a 1.2% j^p 30.4%
                  j^a 30.4% j^v 2.7% f 4.9%. Kimodo families map onto KTJD groups as:
                  r^p->{root_pos,smooth_root} r^a->{heading} j^p->{body_pos}
                  j^a->{body_rot,root_rot} j^v->{body_vel,root_vel} f->{contact}
                  (root_rot: Kimodo's rep has no root orientation beyond heading; KTJD's root
                  rot6d is a rotation channel, so it joins the rotation family.)
  mapping         ONE gamma per FAMILY (Eq.1 has one gamma per term):
                  share_fam = gamma_fam^2 * sum_{i in fam} E_i  =>
                  gamma_fam = sqrt(share_fam / sum E_i), scaled so the j^a family anchors at
                  Kimodo's gamma4 = 10.0
  verification    TWO checks with different jobs (codex round-2):
                  [solve] hard gate -- gamma_fam^2 * sum(E_x1) over families, normalized, must
                    equal the preregistered Kimodo profile to 1e-4 (4-decimal gamma rounding
                    costs ~3e-6; a mis-assigned group or wrong energy costs >1e-2). TRUE BY
                    CONSTRUCTION of the solve: it gates solve/serialization correctness, and is
                    explicitly NOT evidence that the profile is attained during training.
                  [mechanism] 30 optimizer-free steps on the REAL arm model (dim384/depth7/
                    heads8, struct+dir ON, B=CALIB_BATCH, default 8): per-group share of sum |dLoss/d x1_pred|^2 must
                    match gamma^2 * E_err within 25%. This catches wiring bugs (dropped group,
                    gamma not reaching the loss, mask zeroing a group). It deliberately does NOT
                    compare against the Kimodo profile: the init error energy carries an
                    untrained-output-variance transient (~0.5-0.9 on every group), so the
                    init-gradient profile differs from the data-energy profile by construction
                    -- in Kimodo's own setup too.
                  plus tiling assertions (groups cover the valid-cell space exactly once, every
                  group count > 0).

The artifact (configs/ktjd17_gamma_calibration_v2.json) is written ONLY if every assertion
passes. Training refuses to start without it and re-checks generation/gains/schema hashes.
"""
import hashlib
import os
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
EXCLUDE = os.environ.get("EXCLUDE", "configs/pzh312_extreme_cut_K100.json")
from src.data.incontext_pairs import InContextPairs, collate, DEMO_FRAMES, TARGET_FRAMES  # noqa: E402
from src.data.ktjd17_incontext import Ktjd17Base, ktjd17_split_names                      # noqa: E402
from src.models.v2.dit_motion import (InContextMotionDiT, cfm_loss,                       # noqa: E402
                                      _GROUP_SPEC_KTJD17, KTJD17_MASK_POLICY)
from scripts.train_v2_incontext import ktjd_channel_lut, ktjd_prep, cond_of, to_dev       # noqa: E402

# The grouped loss normalises per BATCH, so the mechanism check certifies the objective only at the batch
# size the run will use (codex 2026-09-03, mirrors the view script). Recorded as protocol.batch; the
# trainer refuses a run whose --batch differs.
CALIB_BATCH = int(os.environ.get("CALIB_BATCH", "8"))
if CALIB_BATCH <= 0:
    raise SystemExit("[FAIL] CALIB_BATCH must be a positive integer")

OUT = Path(os.environ.get("CALIB_OUT", "configs/pzh312_gamma_calibration_v5.json"))
# Kimodo Eq.1 families -> KTJD groups, with the Eq.1-implied share profile (gamma^2-normalized:
# 100/4/100/100/9/16 over a 329 total)
FAMILIES = {
    "r_p": {"groups": ["root_pos", "smooth_root"], "share": 100 / 329},
    "r_a": {"groups": ["heading"], "share": 4 / 329},
    "j_p": {"groups": ["body_pos"], "share": 100 / 329},
    "j_a": {"groups": ["body_rot", "root_rot"], "share": 100 / 329},
    "j_v": {"groups": ["body_vel", "root_vel"], "share": 9 / 329},
    "f": {"groups": ["contact"], "share": 16 / 329},
}
ANCHOR_FAMILY, ANCHOR_GAMMA = "j_a", 10.0        # Kimodo gamma4
HUBER = float(os.environ["HUBER"])   # REQUIRED: must match --huber_delta of the run (0 = pure MSE); made an env input when step-2 (Huber->MSE) landed, same no-default rule as V_SPACE/SIGMA_MIN/T_SAMPLER
# The docstring's own principle -- "the mechanism check must run through the SAME objective the
# run trains on" -- was not implemented for v_space/sigma_min/t_sampler until codex 2026-08-26
# round 4 refuted the share-invariance shortcut: realized shares ride on
# gamma^2 * E_t[w(t)^2 * min(r(t)^2, delta^2)], and the residual profile is t-dependent even
# though the DATA energies E[x1^2] (which the gamma SOLVE uses) are not. Hence: the solve is
# unchanged, but the check runs under the objective's true weighting, and the three knobs are
# REQUIRED (no defaults -- a forgotten export must fail here, not silently certify the old loss).
V_SPACE = os.environ["V_SPACE"] == "1"
SIGMA_MIN = float(os.environ["SIGMA_MIN"])
T_SAMPLER = os.environ["T_SAMPLER"]
# auxiliary acc-matching weight of the run these gammas are for (0 = off). The gamma SOLVE and
# the primary mechanism check stay on the grouped flow objective (gamma_acc=0), where the
# analytic share prediction gamma^2*E_err is valid; a SECOND, diagnostic-only pass then runs
# with gamma_acc active and RECORDS how the measured group shares shift plus the acc_match
# magnitude -- no hard tolerance, because the acc gradient has no analytic share model
# (codex 2026-08-28 item 2: the artifact must characterize the objective it certifies).
GAMMA_ACC = float(os.environ["GAMMA_ACC"])
VERIFY_STEPS, VERIFY_TOL = 30, 1.25    # mechanism check band, recorded verbatim in the artifact
ARM = dict(dim=384, depth=7, heads=8)  # the Step-1 arm config the gammas will train
# Demo condition of the mechanism check (codex 2026-09-04 P0): the arm model is driven with the SAME demo
# the run trains with -- 1-frame rest (legacy default) or a DEMO_FRAMES-frame real clip of the same rig.
# Recorded as protocol.demo_rest / demo_frames; the trainer refuses a run whose demo differs.
DEMO_REST = int(os.environ.get("DEMO_REST", "1"))
DEMO_FRAMES = int(os.environ.get("DEMO_FRAMES", "1"))
if DEMO_REST not in (0, 1) or DEMO_FRAMES < 1 or (DEMO_REST == 1 and DEMO_FRAMES != 1):
    raise SystemExit("[FAIL] DEMO_REST must be 0/1 and DEMO_FRAMES >= 1 (rest demo implies exactly 1 frame)")


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    R = os.environ.get("KTJD_ROOT", "dataset/ktjd17_pz_human312")
    base = Ktjd17Base(R, caption_emb_cache=os.environ.get(
                          "CAPTION_CACHE", "data/anytop_caption_llm2vec_v4b272neutral_multi"),
                      joint_semantics="data/joint_semantics_llm2vec_pzh312_v1.npz",
                      percell_stats=os.environ.get("PERCELL", "data/pzh312_norm_stats_v4.npz"),
                      texts_json=os.environ.get(
                          "TEXTS_JSON", "motion_texts_by_file_clean_v1.json"),
                  exclude_clips=EXCLUDE)
    names = ktjd17_split_names(R, exclude=EXCLUDE)
    ds = InContextPairs(base, names["train"], names["train"], balance_skeletons=False, seed=0,
                        emit_graph_v2=True, demo_rest=bool(DEMO_REST), demo_frames=DEMO_FRAMES)
    lut = ktjd_channel_lut(base)

    # ---- tiling assertion: per rig, the 9 groups cover channel_valid cells EXACTLY once ----
    for rig, cv in ((r, base.static_masks(r)["channel_valid"]) for r in sorted(lut)):
        J = cv.shape[0]
        hit = np.zeros((J, 17), dtype=np.int32)
        for name, (js, cs) in _GROUP_SPEC_KTJD17.items():
            sub = np.zeros((J, 17), dtype=bool)
            sub[js, :] = True
            sub[:, [c for c in range(17) if c not in cs]] = False
            hit += sub.astype(np.int32)
        if not (hit[cv] == 1).all():
            raise SystemExit(f"[FAIL] group spec does not tile valid cells exactly once ({rig})")
    print(f"[tiling] all {len(lut)} rigs: groups tile valid cells exactly once")

    # ---- pass 1: energies over the full train cohort ----
    from torch.utils.data import DataLoader
    gen = torch.Generator(); gen.manual_seed(0)
    dl = DataLoader(ds, batch_size=CALIB_BATCH, shuffle=False, num_workers=24, collate_fn=collate,
                    generator=gen)
    e_sum = {g: 0.0 for g in _GROUP_SPEC_KTJD17}
    e_cnt = {g: 0 for g in _GROUP_SPEC_KTJD17}
    t0 = time.time()
    for bi, b in enumerate(dl):
        # FULL pass. A truncated prefix is NOT a corpus sample here: the index is ordered by rig
        # (incontext_pairs.py:231) and the loader runs shuffle=False, so the first N batches are
        # one rig's clips -- on this corpus, 4,000 all-human windows out of a mixed PZ+human
        # corpus (codex 2026-08-21 (A)2). These gammas are frozen into a 500-epoch objective.
        x17, kt = ktjd_prep(b, lut, gammas=None)
        m = (b["is_target"][..., None] & b["valid"]).float()[..., None].expand_as(x17).clone()
        cvb = kt["channel_valid"][:, None].float().expand_as(x17).clone()
        cvb[:, :, 0, 15:17] *= kt["heading_valid"].float()[:, :, None]
        m = m * cvb
        for gname, (js, cs) in _GROUP_SPEC_KTJD17.items():
            e = (x17[:, :, js][..., cs] ** 2 * m[:, :, js][..., cs])
            e_sum[gname] += float(e.sum()); e_cnt[gname] += int(m[:, :, js][..., cs].sum())
    energies = {g: e_sum[g] / max(e_cnt[g], 1) for g in _GROUP_SPEC_KTJD17}
    n_windows, n_rigs = len(ds), len(ds.types)
    print(f"[cohort] {n_windows} train targets over {n_rigs} rigs, one full pass", flush=True)
    if min(e_cnt.values()) <= 0:
        raise SystemExit(f"[FAIL] empty group in cohort: {e_cnt}")
    print(f"[energy] {bi+1} batches in {time.time()-t0:.0f}s: "
          + " ".join(f"{g}={energies[g]:.3f}" for g in energies))

    # ---- family gammas from the Kimodo-implied profile ----
    fam_of = {g: f for f, spec in FAMILIES.items() for g in spec["groups"]}
    assert sorted(fam_of) == sorted(_GROUP_SPEC_KTJD17), "family map must cover every group once"
    raw_fam = {f: (spec["share"] / max(sum(energies[g] for g in spec["groups"]), 1e-8)) ** 0.5
               for f, spec in FAMILIES.items()}
    scale = ANCHOR_GAMMA / raw_fam[ANCHOR_FAMILY]
    gam_fam = {f: round(v * scale, 4) for f, v in raw_fam.items()}
    gammas = {g: gam_fam[fam_of[g]] for g in _GROUP_SPEC_KTJD17}
    ts = {f: FAMILIES[f]["share"] for f in FAMILIES}          # family target shares
    print("[gammas] " + " ".join(f"{f}={gam_fam[f]}({'+'.join(FAMILIES[f]['groups'])})"
                                 for f in gam_fam))

    # ---- SOLVE-CONSISTENCY CHECK ----
    # HONEST SCOPE (codex round-2 blocker 2, and its criticism of the first attempt at this):
    # gamma_fam = sqrt(share_fam / sum E) makes gamma^2 * sum E == share BY CONSTRUCTION, so
    # this check CANNOT prove that the Kimodo profile is "attained" -- nothing measured at
    # calibration time can, because the profile is a DESIGN CHOICE realized by a solve, not an
    # empirical claim. What it does prove, and what it exists for, is that the solve was
    # executed on the MEASURED energies with the DECLARED family map and anchor: a wrong energy
    # vector, a mis-assigned group, a wrong anchor family, or a serialization slip all break it
    # by >1e-2. It is a solve/serialization gate, not evidence of attainment.
    #
    # WHY E_x1 IS THE RIGHT NORMALIZER: Kimodo's gamma^2 share profile holds BECAUSE each of its
    # representation components is normalized to unit variance -- the gammas alone then set the
    # shares. KTJD normalization (s_rig + block gains) does NOT equalize per-group energy, so
    # the faithful transfer is gamma_i = gamma_kimodo_i / sqrt(E_i), which makes each group's
    # term "unit-variance-normalized component x Kimodo's gamma".
    #
    # The empirically meaningful quantity -- the gradient share actually realized during
    # training -- is NOT verifiable here (it depends on the error energies, which evolve). The
    # mechanism check below reports it at init; tracking it over training is an open item.
    ach = {f: gam_fam[f] ** 2 * sum(energies[g] for g in FAMILIES[f]["groups"])
           for f in FAMILIES}
    tot_a = sum(ach.values())
    ach = {f: ach[f] / tot_a for f in ach}
    off = {f: abs(ach[f] - ts[f]) for f in ts}
    print("[solve] gamma^2*E_x1 family shares: "
          + " ".join(f"{f}={ach[f]:.4f}(target {ts[f]:.4f})" for f in ach))
    # tolerance is rounding-aware: the gammas are stored to 4 decimals (that is what training
    # consumes), which perturbs the shares by ~3e-6. A real defect -- wrong family map, wrong
    # energy, wrong anchor -- moves them by >1e-2, so 1e-4 separates the two cleanly.
    if max(off.values()) > 1e-4:
        raise SystemExit(f"[FAIL] gamma solve does not reproduce the preregistered Kimodo "
                         f"share profile: max deviation {max(off.values()):.2e} "
                         f"({ {f: round(ach[f], 4) for f in off} })")
    print(f"[solve] PASS (solve/serialization consistency, max dev {max(off.values()):.1e}) "
          f"-- NOT a claim that the profile is attained during training")

    # ---- verification: 30 optimizer-free steps on the real arm model ----
    torch.manual_seed(0)
    # grad_ckpt: activation checkpointing only (recompute, bit-identical gradients); at CALIB_BATCH=16
    # the fp32 arm model without it exceeds a 140 GB H200 on the J=142 rigs (2026-09-04).
    model = InContextMotionDiT(in_ch=17, dim=ARM["dim"], depth=ARM["depth"],
                               n_heads=ARM["heads"], d_text=4096, d_joint_sem=4096,
                               use_struct_feats=True, use_dir_bias=True, grad_ckpt=True).to(dev).train()
    holder = {}
    def hook(_m, _i, out):
        out.retain_grad(); holder["out"] = out
    model.register_forward_hook(hook)
    g_sum = {g: 0.0 for g in _GROUP_SPEC_KTJD17}
    # verify batches must be cohort-representative: the index is alphabetical by rig, so an
    # unshuffled head-240 slice is one corner of the corpus (first run: r_p share off 2x on
    # exactly that bias). Fixed-seed shuffle, preregistered.
    vgen = torch.Generator(); vgen.manual_seed(1)
    it = iter(DataLoader(ds, batch_size=CALIB_BATCH, shuffle=True, generator=vgen, num_workers=2,
                         collate_fn=collate))
    err_sum = {g: 0.0 for g in _GROUP_SPEC_KTJD17}
    err_cnt = {g: 0 for g in _GROUP_SPEC_KTJD17}
    sat_hit = sat_tot = 0            # how often the knee actually binds, on RESIDUALS not targets
    for step in range(VERIFY_STEPS):
        b = to_dev(next(it), dev)
        x17, kt = ktjd_prep(b, lut, gammas=gammas)
        # Seed immediately before the call, then REPLAY the t draw for the prediction side:
        # t is the FIRST rng consumption inside cfm_loss (dit_motion.py:614-619, before the
        # x0 randn), so the replayed draw below is bit-identical to the t the loss used.
        torch.manual_seed(10000 + step)
        loss = cfm_loss(model, x17, is_target=b["is_target"], valid=b["valid"],
                        huber_delta=HUBER, t_sampler=T_SAMPLER, v_space=V_SPACE,
                        sigma_min=SIGMA_MIN, gamma_acc=0.0, **kt, **cond_of(b))
        torch.manual_seed(10000 + step)
        if T_SAMPLER == "logitnormal":
            t_used = torch.sigmoid(torch.randn(x17.shape[0], device=dev) * 0.8 + 0.8)  # mirrors dit_motion.py:624 (our-convention +0.8)
        else:
            t_used = torch.rand(x17.shape[0], device=dev)
        model.zero_grad(set_to_none=True)
        loss.backward()
        gr = holder["out"].grad
        # diagnostic: per-group ERROR energy (what the shares actually ride on at init),
        # rebuilt with the same effective mask as cfm_loss
        with torch.no_grad():
            eff = kt["channel_valid"][:, None].float().expand_as(x17).clone()
            eff[:, :, 0, 15:17] *= kt["heading_valid"].float()[:, :, None]
            mm = (b["is_target"][..., None] & b["valid"]).float()[..., None] * eff
            r = holder["out"].detach() - x17 * eff
            if HUBER > 0.0:
                # Under Huber the per-cell gradient is 2*min(|r|,delta)*sign(r), so the gradient
                # ENERGY that the shares ride on is 4*min(r^2,delta^2) -- the factor 4 is common to
                # every group and cancels in a share. Predicting with r^2 would describe an MSE
                # objective while the backward pass runs Huber (codex 2026-08-21 (A)3).
                e2 = torch.clamp(r ** 2, max=HUBER ** 2)
            else:
                e2 = r ** 2
            if V_SPACE:
                # the loss element is huber(r) * w(t)/const, so the gradient ENERGY carries
                # w(t)^2; const is a global scalar and cancels in every share.
                w = 1.0 / torch.clamp(1.0 - t_used, min=SIGMA_MIN) ** 2
                e2 = e2 * (w ** 2)[:, None, None, None]
            sat_hit += int(((r.abs() > HUBER) & (mm > 0)).sum()) if HUBER > 0 else 0
            sat_tot += int((mm > 0).sum())
        for gname, (js, cs) in _GROUP_SPEC_KTJD17.items():
            g_sum[gname] += float((gr[:, :, js][..., cs] ** 2).sum())
            err_sum[gname] += float((e2[:, :, js][..., cs] * mm[:, :, js][..., cs]).sum())
            err_cnt[gname] += int(mm[:, :, js][..., cs].sum())
    e_err = {g: err_sum[g] / max(err_cnt[g], 1) for g in err_sum}
    sat_frac = sat_hit / max(sat_tot, 1)
    print(f"[verify-diag] Huber knee delta={HUBER}: residual saturation {sat_frac:.3e} of "
          f"supervised cells at init ({sat_hit:,}/{sat_tot:,}). Reported target-magnitude rates "
          f"are NOT this number -- the knee acts on residuals (codex 2026-08-21).")
    print("[verify-diag] E_err vs E_x1 per group: "
          + " ".join(f"{g}={e_err[g]:.3f}/{energies[g]:.3f}" for g in e_err))
    pred_share_t = {g: gammas[g] ** 2 * e_err[g] for g in e_err}
    tot_p = sum(pred_share_t.values())
    print("[verify-diag] gamma^2*E_err predicted shares: "
          + " ".join(f"{g}={pred_share_t[g]/tot_p:.3f}" for g in pred_share_t))
    tot = sum(g_sum.values())
    measured = {g: g_sum[g] / tot for g in g_sum}
    fam_measured = {f: sum(measured[g] for g in FAMILIES[f]["groups"]) for f in FAMILIES}
    print("[verify] measured family shares: "
          + " ".join(f"{f}={fam_measured[f]:.3f}" for f in fam_measured))
    # MECHANISM CHECK (diagnostic, NOT the design-target verification -- that one is the
    # [solve] check above). This only asserts that the implemented loss weights gradients the
    # way its own formula says: measured output-grad share == gamma^2 * E_err per group. It
    # CANNOT verify the Kimodo share profile, because the init error energy E_err carries a
    # ~0.5-0.9 untrained-output-variance transient on every group (measured 2026-08-20:
    # contact E .134 -> E_err .663, heading .500 -> 1.431), so the init-gradient profile differs
    # from the data-energy profile by construction -- and would in Kimodo's own setup too.
    # Its value is catching wiring bugs (a group silently dropped, a gamma not reaching the
    # loss, a mask zeroing a group), which it does exactly.
    pred_g = {g: gammas[g] ** 2 * e_err[g] for g in e_err}
    tp = sum(pred_g.values())
    mech_bad = {g for g in measured
                if not (pred_g[g] / tp / VERIFY_TOL <= measured[g] <= pred_g[g] / tp * VERIFY_TOL)}
    if mech_bad:
        raise SystemExit(f"[FAIL] implemented weighting deviates from gamma^2*E_err mechanism "
                         f"for {sorted(mech_bad)}: measured="
                         f"{ {g: round(measured[g],3) for g in mech_bad} } predicted="
                         f"{ {g: round(pred_g[g]/tp,3) for g in mech_bad} }")
    acc_diag = None
    if GAMMA_ACC > 0.0:
        # diagnostic pass: same batches/seeds, acc term ACTIVE; record share drift + magnitude
        g_sum2 = {g: 0.0 for g in _GROUP_SPEC_KTJD17}
        acc_vals = []
        vgen2 = torch.Generator(); vgen2.manual_seed(1)
        it2 = iter(DataLoader(ds, batch_size=CALIB_BATCH, shuffle=True, generator=vgen2, num_workers=2,
                              collate_fn=collate))
        for step in range(VERIFY_STEPS):
            b = to_dev(next(it2), dev)
            x17, kt = ktjd_prep(b, lut, gammas=gammas)
            torch.manual_seed(10000 + step)
            loss2, parts2 = cfm_loss(model, x17, is_target=b["is_target"], valid=b["valid"],
                                     huber_delta=HUBER, t_sampler=T_SAMPLER, v_space=V_SPACE,
                                     sigma_min=SIGMA_MIN, gamma_acc=GAMMA_ACC,
                                     return_parts=True, **kt, **cond_of(b))
            model.zero_grad(set_to_none=True)
            loss2.backward()
            gr2 = holder["out"].grad
            for gname, (js, cs) in _GROUP_SPEC_KTJD17.items():
                g_sum2[gname] += float((gr2[:, :, js][..., cs] ** 2).sum())
            acc_vals.append(parts2["acc_match"])
        tot2 = sum(g_sum2.values())
        fam2 = {f: sum(g_sum2[g] / tot2 for g in FAMILIES[f]["groups"]) for f in FAMILIES}
        acc_diag = {"gamma_acc": GAMMA_ACC,
                    "acc_match_mean": float(np.mean(acc_vals)),
                    "family_shares_with_acc": {f: round(v, 6) for f, v in fam2.items()},
                    "family_shares_without_acc": {f: round(v, 6) for f, v in fam_measured.items()},
                    "note": "diagnostic only -- no analytic prediction exists for the acc term"}
        print(f"[acc-diag] acc_match(init)={acc_diag['acc_match_mean']:.4f} "
              f"shares with acc: " + " ".join(f"{f}={fam2[f]:.3f}" for f in fam2))
    print("[verify] PASS (mechanism): measured shares match gamma^2*E_err within 25% per group")

    code_sha = hashlib.sha256(
        Path("src/models/v2/dit_motion.py").read_bytes()
        + Path(__file__).read_bytes()).hexdigest()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "version": "pzh312_v1",
        "generation_id": base.generation_id,
        "target_centering": base.provenance["target_centering"],
        "percell_sha256": base.provenance["percell_sha256"],
        "residual_saturation_at_init": sat_frac,
        "protocol": {"cohort": "train_all_targets_one_pass",
                     "cohort_windows": n_windows, "cohort_rigs": n_rigs,
                     "weighting": "clip_balanced",
                     "seed": 0, "batch": CALIB_BATCH, "huber_delta": HUBER,
                     "windows": ("demo_rest1/target_head" if DEMO_REST else "demo_random_rebased/target_head"),
                     "demo_rest": bool(DEMO_REST), "demo_frames": DEMO_FRAMES,
                     "crop_rebase_active": True, "space": "normalized_model_space",
                     "t_sampler": T_SAMPLER, "v_space": V_SPACE, "sigma_min": SIGMA_MIN,
                     "gamma_acc": GAMMA_ACC,
                     "energy_stat": "mean_x1_sq_over_effective_valid_cells",
                     "mapping": "kimodo_eq1_family_shares: share_fam ~ gamma_fam^2 * sum E_i; "
                                "gamma_fam = sqrt(share_fam / sum E_i)",
                     "families": {f: FAMILIES[f]["groups"] for f in FAMILIES},
                     "anchor": {ANCHOR_FAMILY: ANCHOR_GAMMA},
                     "mask_policy_version": KTJD17_MASK_POLICY,
                     "verify": {"steps": VERIFY_STEPS, "tolerance_x": VERIFY_TOL,
                                "arm_model": ARM, "arm_grad_ckpt": True}},
        "counts": e_cnt, "energies": {g: round(v, 6) for g, v in energies.items()},
        "target_family_shares": {f: round(v, 6) for f, v in ts.items()},
        "solve_consistency_check": {
            "statement": "gamma_fam^2 * sum(E_x1) over families, normalized, == the "
                         "preregistered Kimodo Eq.1 share profile (hard gate, 1e-4)",
            "scope": "TRUE BY CONSTRUCTION of the solve -- this gate proves the solve ran on "
                     "the measured energies with the declared family map and anchor, and that "
                     "nothing was mis-serialized. It is NOT evidence that the profile is "
                     "attained during training; the profile is a design choice, and the "
                     "realized gradient share depends on error energies that evolve.",
            "rationale": "Kimodo's gamma^2 profile holds because its components are "
                         "unit-variance normalized; gamma_i = gamma_kimodo_i / sqrt(E_i) is "
                         "the faithful transfer into KTJD units",
            "achieved": {f: round(v, 6) for f, v in ach.items()},
            "max_deviation": max(off.values()),
        },
        "acc_diagnostic": acc_diag,
        "mechanism_check": {
            "statement": "measured output-grad share == gamma^2 * E_err within 25%/group "
                         "(diagnostic: catches wiring bugs; CANNOT verify the design target, "
                         "because init E_err carries an untrained-output-variance transient)",
            "init_error_energies": {g: round(v, 6) for g, v in e_err.items()},
            "measured_grad_shares_init": {g: round(v, 6) for g, v in measured.items()},
            "measured_family_shares_init": {f: round(v, 6) for f, v in fam_measured.items()},
        },
        "gammas": gammas,
        "hashes": {"gains_sha256": base.provenance["gains_sha256"],
                   "schema_sha256": base.provenance["schema_sha256"],
                   "joint_sem_sha256": base.provenance["joint_sem_sha256"],
                   "code_sha256": code_sha,
                   "code_script": "scripts/_measure_ktjd17_gamma_calibration.py",
                   # cut / cohort / manifest seals, same as the view producer (codex 2026-09-04 P0)
                   "exclusion_sha256": (base.provenance_exclusion or {}).get("sha256") or "none",
                   "train_ids_sha256": hashlib.sha256("\n".join(sorted(names["train"])).encode()).hexdigest(),
                   "manifest_sha256": hashlib.sha256((Path(R) / "manifests" / "clips.jsonl").read_bytes()).hexdigest()},
    }, indent=2))
    print(f"[OK] wrote {OUT}")


if __name__ == "__main__":
    main()
