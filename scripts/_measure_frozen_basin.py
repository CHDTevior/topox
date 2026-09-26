#!/usr/bin/env python3
"""Full-cohort frozen-pose basin (codex round-S8: the 0.416 figure had no computation trace).

What fraction of the gradient-weighted objective can a per-clip CONSTANT prediction capture?
Reference points measured with the same code: scale-only KTJD 0.750 (the collapse), 13ch AnyTop
0.452 (the corpus that produced coherent motion). A 24-window sample is NOT stable -- redraws span
0.334..0.475 -- so this runs the WHOLE train cohort and prints the count it used.
"""
import json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
EXCLUDE = "configs/pzh312_extreme_cut_K100.json"   # extreme-tail cut (ratio>100), user-agreed 2026-08-21
from src.data.ktjd17_incontext import Ktjd17Base, ktjd17_split_names
from src.data.incontext_pairs import InContextPairs, collate
from src.models.v2.dit_motion import grouped_loss, _GROUP_SPEC_KTJD17
from torch.utils.data import DataLoader

CAL = sys.argv[1] if len(sys.argv) > 1 else "configs/pzh312_gamma_calibration_v5.json"
ROOT = sys.argv[2] if len(sys.argv) > 2 else "dataset/ktjd17_pz_human312"
base = Ktjd17Base(ROOT,
                  caption_emb_cache="data/anytop_caption_llm2vec_v4b272neutral_multi",
                  joint_semantics="data/joint_semantics_llm2vec_pzh312_v1.npz",
                  percell_stats="data/pzh312_norm_stats_v4.npz",
                  exclude_clips=EXCLUDE)
names = ktjd17_split_names(ROOT, exclude=EXCLUDE)
ds = InContextPairs(base, names["train"], names["train"], balance_skeletons=False, seed=0,
                    demo_rest=True, demo_frames=1)
lut = {r: torch.from_numpy(base.static_masks(r)["channel_valid"]) for r in ds.types}
G = {k: float(v) for k, v in json.load(open(CAL))["gammas"].items()}
g = torch.Generator(); g.manual_seed(0)
dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate, generator=g)
LS = LD = 0.0; n = 0
for b in dl:
    x = b["x"][..., :17]; hv = b["x"][:, :, 0, 17] > 0.5
    cv = torch.zeros(x.shape[0], x.shape[2], 17, dtype=torch.bool)
    for k, ot in enumerate(b["object_type"]):
        cv[k, :lut[ot].shape[0]] = lut[ot]
    m = (b["is_target"][..., None] & b["valid"]).float()[..., None] * cv[:, None].float()
    m = m.clone(); m[:, :, 0, 15:17] *= hv.float()[:, :, None]
    mu = (x * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp_min(1e-6)
    ls, _ = grouped_loss((mu ** 2).expand_as(x), m, G, group_spec=_GROUP_SPEC_KTJD17)
    ld, _ = grouped_loss((x - mu) ** 2, m, G, group_spec=_GROUP_SPEC_KTJD17)
    LS += float(ls); LD += float(ld); n += x.shape[0]
share = LS / (LS + LD)
print(f"cohort            = {n} train windows (full pass), gammas from {CAL}")
print(f"frozen_share      = {share:.4f}")
print(f"reference points  = scale-only KTJD 0.750 | 13ch AnyTop 0.452 | this corpus target < 0.60")
print("VERDICT:", "OK" if share < 0.60 else "REGRESSION")
