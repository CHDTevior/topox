#!/usr/bin/env python3
"""Locate the source of the ep11 gradient explosion, on the ep9 weights that sit inside the
0.217-0.24 fkdist band where all six runs have died.

The crash is NOT gradual divergence: epoch 10 ends at grad mean 2.489, epoch 11 at 502.357 with a
max of 439175. Individual batches produce 1e9-scale gradients while most stay near 2. So the
question is which TERM of the objective explodes once the model is accurate, and on which samples.

Strategy: run many batches through the full objective, rank them by gradient norm, then re-run the
worst ones with one term disabled at a time. A term whose removal collapses the gradient is the
culprit; if none does, the blow-up is in the shared flow-matching path itself.
"""
import json, os, sys
from pathlib import Path
import torch
sys.path.insert(0, "/iridisfs/scratch/ts1v23/workspace/noKslot_clean")
from src.data.ktjd17_incontext import Ktjd17Base, ktjd17_split_names
from src.data.incontext_pairs import InContextPairs, collate
from src.models.v2.dit_motion import InContextMotionDiT, cfm_loss
from scripts.train_v2_incontext import ktjd_channel_lut, ktjd_prep, cond_of, to_dev, fk_pack_of
from torch.utils.data import DataLoader

CKPT = os.environ.get("CKPT", "runs/v2_pzh312_run6/best_model.pt")
NB   = int(os.environ.get("NBATCH", "300"))
dev  = "cuda:0"
EXC  = "configs/pzh312_extreme_cut_K100.json"
ROOT = "dataset/ktjd17_pz_human312"

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
a = ck["args"]
print(f"[diag] {CKPT}: epoch={ck['epoch']} gstep={ck['gstep']} val={ck.get('val')}")

base = Ktjd17Base(ROOT, caption_emb_cache="data/anytop_caption_llm2vec_v4b272neutral_multi",
                  joint_semantics="data/joint_semantics_llm2vec_pzh312_v1.npz",
                  percell_stats="data/pzh312_norm_stats_v4.npz", exclude_clips=EXC)
names = ktjd17_split_names(ROOT, exclude=EXC)
ds = InContextPairs(base, names["train"], names["train"], balance_skeletons=True, seed=0,
                    emit_fk_fields=True, emit_graph_v2=True, demo_rest=True, demo_frames=1)
lut = ktjd_channel_lut(base)
G = {k: float(v) for k, v in
     json.loads(Path("configs/pzh312_gamma_calibration_v5.json").read_text())["gammas"].items()}

m = InContextMotionDiT(in_ch=17, dim=a["dim"], depth=a["depth"], n_heads=a["heads"],
                       d_text=4096, d_joint_sem=4096, use_struct_feats=True,
                       use_dir_bias=True).to(dev)
m.load_state_dict(ck["model"]); m.train()

def run(b, **over):
    """One forward+backward with the training objective, optionally overriding terms."""
    kw = dict(v_space=a["v_space"], sigma_min=a["sigma_min"], huber_delta=a["huber_delta"],
              t_sampler=a["t_sampler"])
    fk = {}
    if a["gamma_vel"] > 0 or a["gamma_lock"] > 0:
        fk = dict(gamma_vel=a["gamma_vel"], gamma_lock=a["gamma_lock"], fk_pack=fk_pack_of(b))
    if a["gamma_fk"] > 0:
        fk.update(gamma_fk=a["gamma_fk"], fk_pack=fk_pack_of(b))
    kw.update(fk); kw.update(over)
    for k in [k for k, v in kw.items() if v is None]:
        kw.pop(k)
    x17, kt = ktjd_prep(b, lut, gammas=G)
    m.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a["bf16"]):
        loss = cfm_loss(m, x17, is_target=b["is_target"], valid=b["valid"], **kt, **kw, **cond_of(b))
    loss.float().backward()
    gn = torch.nn.utils.clip_grad_norm_(m.parameters(), float("inf")).item()  # measure, don't clip
    return float(loss.detach()), gn

dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=3, collate_fn=collate,
                generator=torch.Generator().manual_seed(7))
rows, it = [], iter(dl)
for i in range(NB):
    b = to_dev(next(it), dev)
    l, g = run(b)
    rows.append((g, l, i, b))
    if (i + 1) % 50 == 0:
        gs = sorted(r[0] for r in rows)
        print(f"[diag] {i+1} batches: grad median={gs[len(gs)//2]:.2f} "
              f"p90={gs[int(len(gs)*0.9)]:.2f} max={gs[-1]:.3e}", flush=True)

rows.sort(key=lambda r: -r[0])
gs = sorted(r[0] for r in rows)
print(f"\n[diag] === {NB} batches on ep{ck['epoch']} weights ===")
print(f"[diag] grad norm: median={gs[len(gs)//2]:.2f} p90={gs[int(len(gs)*0.9)]:.2f} "
      f"p99={gs[int(len(gs)*0.99)]:.2f} max={gs[-1]:.4e}")
print(f"[diag] batches above 100: {sum(1 for g in gs if g > 100)}/{NB}   "
      f"above 1000: {sum(1 for g in gs if g > 1000)}/{NB}")

ABL = [("full objective",        {}),
       ("gamma_fk=0",            dict(gamma_fk=0.0)),
       ("gamma_vel=0",           dict(gamma_vel=0.0)),
       ("gamma_lock=0",          dict(gamma_lock=0.0)),
       ("all FK aux off",        dict(gamma_fk=0.0, gamma_vel=0.0, gamma_lock=0.0)),
       ("v_space off",           dict(v_space=False)),
       ("huber off (pure MSE)",  dict(huber_delta=0.0))]
print(f"\n[diag] ablating the {min(5, len(rows))} worst batches (same batch, same seed):")
print(f"[diag] {'variant':<24}" + "".join(f"{'b'+str(i):>13}" for i in range(min(5, len(rows)))))
for name, over in ABL:
    out = []
    for g0, l0, idx, b in rows[:5]:
        torch.manual_seed(1234 + idx)      # same t / noise draw across variants
        _, g = run(b, **over)
        out.append(g)
    print(f"[diag] {name:<24}" + "".join(f"{g:>13.3e}" for g in out), flush=True)
