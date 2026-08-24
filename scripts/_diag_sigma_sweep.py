#!/usr/bin/env python3
"""Does a larger sigma_min tame the UNSTABLE state?

sigma_min floors the v-space weight 1/clamp(1-t, s)^2 at 1/s^2, so 0.2 caps it at 25 and 0.3 at
~11. run5 (sigma_min 0.2) pushed the death from ep11 to ep18 -- the only knob with positive
evidence. Measuring on the HEALTHY ep9 weights is useless (every batch is fine there); what matters
is whether a larger floor lowers gradients in the state the model actually blows up in. So this
runs on a post-blow-up snapshot.
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

CKPT = os.environ["CKPT"]; NB = int(os.environ.get("NBATCH", "120"))
dev = "cuda:0"; ROOT = "dataset/ktjd17_pz_human312"; EXC = "configs/pzh312_extreme_cut_K100.json"
ck = torch.load(CKPT, map_location="cpu", weights_only=False); a = ck["args"]
print(f"[sweep] {CKPT}  epoch={ck['epoch']} gstep={ck['gstep']}")
base = Ktjd17Base(ROOT, caption_emb_cache="data/anytop_caption_llm2vec_v4b272neutral_multi",
                  joint_semantics="data/joint_semantics_llm2vec_pzh312_v1.npz",
                  percell_stats="data/pzh312_norm_stats_v4.npz", exclude_clips=EXC)
names = ktjd17_split_names(ROOT, exclude=EXC)
ds = InContextPairs(base, names["train"], names["train"], balance_skeletons=True, seed=0,
                    emit_fk_fields=True, emit_graph_v2=True, demo_rest=True, demo_frames=1)
lut = ktjd_channel_lut(base)
G = {k: float(v) for k, v in
     json.loads(Path("configs/pzh312_gamma_calibration_v5.json").read_text())["gammas"].items()}
m = InContextMotionDiT(in_ch=17, dim=a["dim"], depth=a["depth"], n_heads=a["heads"], d_text=4096,
                       d_joint_sem=4096, use_struct_feats=True, use_dir_bias=True).to(dev)
m.load_state_dict(ck["model"]); m.train()

def grads(sig, batches):
    out = []
    for i, b in enumerate(batches):
        torch.manual_seed(9000 + i)          # identical t/noise draw across variants
        fk = dict(gamma_vel=a["gamma_vel"], gamma_lock=a["gamma_lock"], fk_pack=fk_pack_of(b))
        fk.update(gamma_fk=a["gamma_fk"])
        x17, kt = ktjd_prep(b, lut, gammas=G)
        m.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a["bf16"]):
            loss = cfm_loss(m, x17, is_target=b["is_target"], valid=b["valid"], v_space=True,
                            sigma_min=sig, huber_delta=a["huber_delta"],
                            t_sampler=a["t_sampler"], **fk, **kt, **cond_of(b))
        if not torch.isfinite(loss):
            out.append(float("inf")); continue
        loss.float().backward()
        g = torch.nn.utils.clip_grad_norm_(m.parameters(), float("inf")).item()
        out.append(g)
    return out

dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=3, collate_fn=collate,
                generator=torch.Generator().manual_seed(11))
it = iter(dl); batches = [to_dev(next(it), dev) for _ in range(NB)]
print(f"[sweep] {NB} identical batches per variant\n")
print(f"{'sigma_min':>10} {'w_cap':>7} {'median':>10} {'p90':>10} {'max':>12} {'>200':>6} {'inf':>5}")
for sig in (0.2, 0.3, 0.4, 0.5):
    v = grads(sig, batches)
    fin = sorted(x for x in v if x != float("inf"))
    ninf = sum(1 for x in v if x == float("inf"))
    n = len(fin)
    print(f"{sig:>10} {1/sig**2:>7.1f} {fin[n//2]:>10.2f} {fin[int(n*0.9)]:>10.2f} "
          f"{fin[-1]:>12.4g} {sum(1 for x in fin if x>200):>6} {ninf:>5}", flush=True)
