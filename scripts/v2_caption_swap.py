#!/usr/bin/env python3
"""Caption-swap probe: THE direct text-effect measurement (designed in run-1 Phase A, run now).

For one rig: keep the SAME demo, SAME target slot, SAME sampling noise (identical seed per
variant), and swap ONLY the caption embedding across N variants (the base clip's caption plus
donor captions from other clips of the same rig). If the outputs barely differ, the text
pathway is being ignored at generation time -- no interpretation needed.

Output per rig:
  <rig>_capswap.gif   panels: DEMO | GEN@cap1 | GEN@cap2 | GEN@cap3 | GT(base clip), shared
                      scale, ground line; all generated panels use the RIC recovery.
  summary.txt         full captions + the numbers:
                        text effect = mean |gen_i - gen_j| over real target frames (normalized
                        space), pairwise; reference scale = mean |gen_1 - GT| (how far a
                        generation sits from GT at all). effect << reference  =>  text ignored.
"""
import argparse, pickle, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.anytop_dataset import (AnyTopDataset, _STD_FLOOR,                   # noqa: E402
                                     _recover_world_positions)
from src.data.incontext_pairs import (InContextPairs, collate, read_split,       # noqa: E402
                                      truebones_types, pzh_types, DEMO_FRAMES, TARGET_FRAMES)
from src.models.v2.dit_motion import InContextMotionDiT, sample                   # noqa: E402
from scripts.v2_render_incontext import (COLS, render_gif, world_of)              # noqa: E402

# extra panel colors for the caption variants (orange kept for cap1 = the clip's own caption)
COLS.setdefault("gen_c2", (30, 90, 190))
COLS.setdefault("gen_c3", (20, 140, 60))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rigs", default="Trex,Cat", help="rigs, rendered one gif each")
    ap.add_argument("--bucket", choices=("train", "val"), default="train",
                    help="train = seen clips (upper bound for text adherence)")
    ap.add_argument("--n_caps", type=int, default=3)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cfg_text", type=float, default=1.0,
                    help=">1 needs a CFG-trained ckpt (run-3 family)")
    ap.add_argument("--corpus", choices=("truebones", "pzh"), default="truebones")
    ap.add_argument("--data_root", default="data/animo4d_L4TB_plus_human_v4b272neutral")
    ap.add_argument("--splits_dir", default="data/holdout_splits_v1")
    ap.add_argument("--joint_sem", default="data/joint_semantics_llm2vec_v1.npz")
    ap.add_argument("--caption_cache", default="data/anytop_caption_llm2vec_v4b272neutral_multi")
    ap.add_argument("--texts_json", default="motion_texts_by_file_clean_v1.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    ca = ck["args"]
    model = InContextMotionDiT(in_ch=13, dim=ca["dim"], depth=ca["depth"], n_heads=ca["heads"],
                               d_text=4096, d_joint_sem=4096,
                               use_struct_feats=bool(ca.get("struct_feats", False)),
                               use_dir_bias=bool(ca.get("dir_bias", False))).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    DF = int(ca.get("demo_frames", DEMO_FRAMES))
    TF = int(ca.get("target_frames", 240))
    print(f"[capswap] ckpt {a.ckpt} (epoch {ck.get('epoch', -1)}) demo={DF} target={TF}")

    cond = pickle.load(open(f"{a.data_root}/_cond_normalized_J144.pkl", "rb"))
    tb = truebones_types(cond.keys()) if a.corpus == "truebones" else pzh_types(cond.keys())
    names = {k: read_split(a.splits_dir, k) for k in ("train", "val")}
    base = AnyTopDataset(data_root=a.data_root, split="all", num_frames=300, max_joints=144,
                         load_captions=True, caption_emb_cache=a.caption_cache,
                         random_caption=False, augment=False, joint_semantics=a.joint_sem,
                         species_whitelist=tb, splits_dir=a.splits_dir,
                         texts_json_name=a.texts_json)
    tgt_names = names["train"] if a.bucket == "train" else names["val"]
    ds = InContextPairs(base, tgt_names, names["train"], object_types=tb,
                        balance_skeletons=False, seed=a.seed, demo_frames=DF, target_frames=TF,
                        emit_graph_v2=bool(ca.get("struct_feats", False))
                        or bool(ca.get("dir_bias", False)))

    lines = []
    for rig in [r.strip() for r in a.rigs.split(",") if r.strip()]:
        if rig not in ds.types:
            print(f"[capswap] SKIP {rig}: not in {a.bucket} bucket"); continue
        positions = [i for i, (ot, _) in enumerate(ds.index) if ot == rig]

        def energy(ix):
            gt_it = base[ds.index[ix][1]]
            Jn = int(gt_it["num_joints"]); Tn = min(int(gt_it["num_frames"]), ds.Tt)
            if Tn < 2:
                return 0.0
            xn = np.asarray(gt_it["anytop_x"])[:Jn, :, :Tn].transpose(2, 0, 1)
            mn = np.asarray(gt_it["anytop_mean"])[:Jn]; sd = np.asarray(gt_it["anytop_std"])[:Jn]
            raw = (xn * (sd[None] + _STD_FLOOR) + mn[None]).astype(np.float64)
            xw = _recover_world_positions(raw)                 # WORLD, incl. root translation
            return float(np.linalg.norm(np.diff(xw, axis=0), axis=-1).mean())

        ranked = sorted(positions, key=energy, reverse=True)
        pos = ranked[0]
        base_cap = str(base[ds.index[pos][1]].get("caption", ""))
        # donors: most-energetic OTHER clips of this rig whose caption STRING differs; distinct
        # captions are the whole point -- a synonym swap would under-measure the effect.
        donors, seen_caps = [], {base_cap}
        for ix in ranked[1:]:
            cap = str(base[ds.index[ix][1]].get("caption", ""))
            if cap and cap not in seen_caps:
                donors.append(ix); seen_caps.add(cap)
            if len(donors) >= a.n_caps - 1:
                break
        if len(donors) < a.n_caps - 1:
            print(f"[capswap] SKIP {rig}: not enough distinct captions"); continue

        ds._wrng_key = None
        item = ds[pos]
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in collate([item]).items()}
        J = int(item["n_joints"])
        t_real = int(item["frame_valid"][DF:].sum())
        t_item = base[ds.index[pos][1]]
        mean = np.asarray(t_item["anytop_mean"])[:J]; std = np.asarray(t_item["anytop_std"])[:J]
        parents = [int(p) for p in t_item["parent_indices"][:J]]

        caps = [base_cap] + [str(base[ds.index[ix][1]].get("caption", "")) for ix in donors]
        texts = [b["text"]] + \
            [torch.as_tensor(np.asarray(base[ds.index[ix][1]]["caption_emb"]),
                             dtype=torch.float32, device=dev)[None] for ix in donors]

        gens = []
        g2kw = {k: b[k] for k in ("struct_feats", "updown") if k in b}
        cfg_kw = {}
        if a.cfg_text != 1.0:
            cfg_kw = dict(cfg_text=a.cfg_text, demo_frames=DF)
        for te in texts:
            torch.manual_seed(a.seed)                    # SAME noise for every caption variant
            with torch.no_grad():
                g = sample(model, b["x"], b["is_target"], a.steps,
                           joint_bias=b["joint_bias"], frame_valid=b["frame_valid"],
                           joint_valid=b["joint_valid"], text=te, joint_sem=b["joint_sem"],
                           **cfg_kw, **g2kw)
            gens.append(g[0].float().cpu().numpy()[DF:DF + t_real, :J])

        gt = b["x"][0].float().cpu().numpy()
        gtseg = gt[DF:DF + t_real, :J]
        demo_seg = gt[:int(item["frame_valid"][:DF].sum()), :J]

        # ---- the numbers ----
        eff = [float(np.abs(gens[i] - gens[j]).mean())
               for i in range(len(gens)) for j in range(i + 1, len(gens))]
        ref = float(np.abs(gens[0] - gtseg).mean())
        line = (f"{rig}: text-effect pairwise |d| = {['%.4f' % e for e in eff]}  "
                f"reference |gen1-GT| = {ref:.4f}  ratio(mean_eff/ref) = "
                f"{(sum(eff) / len(eff)) / max(ref, 1e-9):.2%}  cfg_text={a.cfg_text}")
        print("[capswap] " + line, flush=True)
        lines.append(line)
        for i, c in enumerate(caps):
            lines.append(f"  cap{i + 1}: {c}")

        panels = [("demo", "DEMO", world_of(demo_seg, mean, std)),
                  ("gen_ric", "GEN cap1", world_of(gens[0], mean, std)),
                  ("gen_c2", "GEN cap2", world_of(gens[1], mean, std)),
                  ("gen_c3", "GEN cap3", world_of(gens[2], mean, std)),
                  ("gt", "GT (cap1's clip)", world_of(gtseg, mean, std))]
        render_gif(out / f"{rig}_capswap.gif", panels, parents,
                   " | ".join(f"{i+1}) {c[:55]}" for i, c in enumerate(caps)), rig)
        print(f"[capswap] {rig}_capswap.gif", flush=True)

    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"[capswap] DONE -> {out}")


if __name__ == "__main__":
    main()
