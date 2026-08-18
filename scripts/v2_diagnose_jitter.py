#!/usr/bin/env python3
"""Root-cause instrument for generation jitter. Read-only; discriminates four hypotheses so the
fix is chosen by evidence, not by menu:

  H1 sampler accumulation   jitter vs ODE steps (5/10/25/50), same ckpt/noise. Falls steeply -> H1.
  H2 rough x0 prediction    one-step denoise from lightly-noised GT (t=0.7/0.9/0.97): if the
                            model's own x_hat0 is jittery even with a nearly-clean input, the
                            training objective under-supervises the near-data regime -> H2
                            (JiT-style v-space loss is the targeted fix).
  H3 undertraining          compare this report across epoch ckpts (300/400/500) -- falling
                            curves mean "keep training", handled by running this per milestone.
  H4 channel inconsistency  ||FK - RIC|| on GENERATED motion, normalised by motion extent.
                            GT measures 0.000% (preflight); whatever the generation shows IS the
                            rotation-family-vs-position-family disagreement -> consistency term.

Reuses the reviewed renderer's world_of/jitter_ratio and the official recovery paths -- nothing
re-implemented. Output: one table per clip + a JSON for cross-epoch comparison.
"""
import argparse, importlib.util, json, pickle, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.anytop_dataset import AnyTopDataset, _STD_FLOOR, _recover_world_positions  # noqa: E402
from src.data.anytop_rot6d_fk import recover_from_bvh_rot_np                             # noqa: E402
from src.data.incontext_pairs import (InContextPairs, collate, read_split,               # noqa: E402
                                      truebones_types, DEMO_FRAMES)
from src.models.v2.dit_motion import InContextMotionDiT, sample                          # noqa: E402

_spec = importlib.util.spec_from_file_location("v2render", ROOT / "scripts/v2_render_incontext.py")
_v2r = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_v2r)
world_of, jitter_ratio = _v2r.world_of, _v2r.jitter_ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rigs_A", default="Alligator,Trex")
    ap.add_argument("--rigs_B", default="BrownBear,Elephant")
    ap.add_argument("--steps_sweep", default="5,10,25,50")
    ap.add_argument("--onestep_ts", default="0.7,0.9,0.97")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out_json", default="")
    ap.add_argument("--data_root", default="data/animo4d_L4TB_plus_human_v4b272neutral")
    ap.add_argument("--splits_dir", default="data/holdout_splits_v1")
    ap.add_argument("--joint_sem", default="data/joint_semantics_llm2vec_v1.npz")
    ap.add_argument("--caption_cache", default="data/anytop_caption_llm2vec_v4b272neutral_multi")
    ap.add_argument("--texts_json", default="motion_texts_by_file_clean_v1.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    ca = ck["args"]
    model = InContextMotionDiT(in_ch=13, dim=ca["dim"], depth=ca["depth"], n_heads=ca["heads"],
                               d_text=4096, d_joint_sem=4096).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    ep = ck.get("epoch", -1)
    print(f"[diag] ckpt {a.ckpt} (epoch {ep})", flush=True)

    cond = pickle.load(open(f"{a.data_root}/_cond_normalized_J144.pkl", "rb"))
    tb = truebones_types(cond.keys())
    names = {k: read_split(a.splits_dir, k) for k in ("train", "val", "held_representative")}
    base = AnyTopDataset(data_root=a.data_root, split="all", num_frames=300, max_joints=144,
                         load_captions=True, caption_emb_cache=a.caption_cache,
                         random_caption=False, augment=False, joint_semantics=a.joint_sem,
                         species_whitelist=tb, splits_dir=a.splits_dir,
                         texts_json_name=a.texts_json)
    dsA = InContextPairs(base, names["val"], names["train"], object_types=tb,
                         balance_skeletons=False, seed=a.seed)
    dsB = InContextPairs(base, names["held_representative"], names["held_representative"],
                         object_types=tb, balance_skeletons=False, seed=a.seed)

    steps_sweep = [int(x) for x in a.steps_sweep.split(",")]
    onestep_ts = [float(x) for x in a.onestep_ts.split(",")]
    report = {"ckpt": a.ckpt, "epoch": int(ep), "clips": []}

    jobs = [("A", r, dsA) for r in a.rigs_A.split(",") if r] + \
           [("B", r, dsB) for r in a.rigs_B.split(",") if r]
    for bucket, rig, ds in jobs:
        if rig not in ds.types:
            continue
        ds._wrng_key = None
        pos = next(i for i, (ot, _) in enumerate(ds.index) if ot == rig)
        item = ds[pos]
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in collate([item]).items()}
        J = int(item["n_joints"])
        t_real = int(item["frame_valid"][DEMO_FRAMES:].sum())
        t_item = base[ds.index[pos][1]]
        mean = np.asarray(t_item["anytop_mean"])[:J]; std = np.asarray(t_item["anytop_std"])[:J]
        parents = [int(p) for p in t_item["parent_indices"][:J]]
        offsets = np.asarray(t_item["rest_offsets"])[:J]
        gt = b["x"][0].float().cpu().numpy()
        gt_seg = gt[DEMO_FRAMES:DEMO_FRAMES + t_real, :J]
        gt_w = world_of(gt_seg, mean, std)
        ckw = dict(joint_bias=b["joint_bias"], frame_valid=b["frame_valid"],
                   joint_valid=b["joint_valid"], text=b["text"], joint_sem=b["joint_sem"])
        row = {"bucket": bucket, "rig": rig, "motion": str(item["motion_id"]), "J": J}

        # ---- H1: jitter vs ODE steps ----
        h1 = {}
        for st in steps_sweep:
            torch.manual_seed(a.seed)
            with torch.no_grad():
                gen = sample(model, b["x"], b["is_target"], st, **ckw)
            gseg = gen[0].float().cpu().numpy()[DEMO_FRAMES:DEMO_FRAMES + t_real, :J]
            ja, jr, _, _ = jitter_ratio(world_of(gseg, mean, std), gt_w)
            h1[st] = round(ja, 2)
        row["H1_jitter_vs_steps"] = h1
        # H4's reference generation is produced explicitly at a FIXED step count so it does not
        # depend on what --steps_sweep happens to contain, and stays comparable across epochs.
        torch.manual_seed(a.seed)
        with torch.no_grad():
            gen_ref = sample(model, b["x"], b["is_target"], 10, **ckw)
        gen10 = gen_ref[0].float().cpu().numpy()[DEMO_FRAMES:DEMO_FRAMES + t_real, :J]

        # ---- H2: one-step x0 from lightly-noised GT ----
        h2 = {}
        for tv in onestep_ts:
            torch.manual_seed(a.seed + 1)
            x1 = b["x"].clone()
            x0n = torch.randn_like(x1)
            tt = torch.full((1,), tv, device=dev)
            xt = (1 - tv) * x0n + tv * x1
            xt = torch.where((~b["is_target"])[..., None, None], x1, xt)
            with torch.no_grad():
                xhat = model(xt, tt, is_target=b["is_target"], **ckw)
            xseg = xhat[0].float().cpu().numpy()[DEMO_FRAMES:DEMO_FRAMES + t_real, :J]
            ja, _, _, _ = jitter_ratio(world_of(xseg, mean, std), gt_w)
            h2[tv] = round(ja, 2)
        row["H2_onestep_x0_jitter"] = h2

        # ---- H4: FK-vs-RIC disagreement on generated motion (GT is 0.000% by preflight) ----
        # BOTH sides normalise by the SAME GT extent: a collapsed/static generation is internally
        # consistent (FK==RIC) and would score ~0 under its own tiny extent -- the shared
        # denominator plus an explicit amplitude ratio make collapse visible instead of hidden.
        def world_and_extent(seg):
            raw = (seg * (std[None] + _STD_FLOOR) + mean[None]).astype(np.float64)
            fk = recover_from_bvh_rot_np(raw, parents, offsets)
            ric = _recover_world_positions(raw)
            ext = float(np.linalg.norm(ric - ric.mean(axis=(0, 1)), axis=-1).max())
            return fk, ric, ext
        fk_g, ric_g, ext_g = world_and_extent(gen10)
        fk_t, ric_t, ext_t = world_and_extent(gt_seg)
        denom = max(ext_t, 1e-9)
        row["H4_fk_ric_rel_gen"] = round(float(np.linalg.norm(fk_g - ric_g, axis=-1).mean()) / denom, 4)
        row["H4_fk_ric_rel_gt"] = round(float(np.linalg.norm(fk_t - ric_t, axis=-1).mean()) / denom, 6)
        row["H4_amp_ratio_gen_vs_gt"] = round(ext_g / denom, 3)   # ~1 healthy; <<1 collapse; >>1 runaway

        print(f"[diag] {bucket}:{rig:11s} H1 {h1} | H2 {h2} | "
              f"H4 gen {row['H4_fk_ric_rel_gen']:.3%} vs gt {row['H4_fk_ric_rel_gt']:.4%} "
              f"amp {row['H4_amp_ratio_gen_vs_gt']:.2f}x", flush=True)
        report["clips"].append(row)

    if a.out_json:
        Path(a.out_json).write_text(json.dumps(report, indent=1))
        print(f"[diag] -> {a.out_json}", flush=True)


if __name__ == "__main__":
    main()
