"""Diagnostic: why is the evaluator tiny-overfit stuck at ln(B)? Check whether the
two towers produce DISCRIMINATIVE embeddings at init, and whether the model CAN
overfit 8 distinct samples at a few learning rates. NOT a gate — a probe."""
from __future__ import annotations
import sys, math
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.anytop_t2m_eval_dataset import AnyTopT2MEvalDataset, collate_fn
from src.models.graph_salad.batch import GraphMotionBatch
from src.models.graph_salad.t2m_evaluator import AnyTopT2MEvaluator, build_multi_positive_mask

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/animo4d_anytop_clean_L4_safe_plus_truebones"
DB = str(ROOT / "checkpoints/text_encoders/distilbert-base-uncased")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

ds = AnyTopT2MEvalDataset(manifest_path=DATA/"eval_splits/val_all.json", data_root=DATA,
    caption_emb_cache=None, split="val", view="full", num_frames=64, max_joints=144)
B = 8
spread = [int(round(k*(len(ds)-1)/(B-1))) for k in range(B)]
coll = collate_fn([ds[i] for i in spread])
coll = {k:(v.to(dev) if isinstance(v,torch.Tensor) else v) for k,v in coll.items()}
batch = GraphMotionBatch.from_collate_dict(coll)
caps = list(coll["caption_text"])
print("captions:", [c[:35] for c in caps])

def offdiag_cos(emb):
    e = torch.nn.functional.normalize(emb.float(), dim=-1)
    s = e @ e.t()
    od = s[~torch.eye(len(e), dtype=torch.bool, device=e.device)]
    return float(od.mean()), float(od.max())

def mk(): return AnyTopT2MEvaluator(coemb_dim=512, text_tower="distilbert", distilbert_path=DB,
    n_heads=8, d_ff=2048, n_graph_layers=6, n_temporal_layers=4, dropout=0.0,
    learnable_temperature=True, temperature=0.07).to(dev)

m = mk(); m.eval()
with torch.no_grad():
    te = m.encode_text(caps); me = m.encode_motion(batch)
tm, tx = offdiag_cos(te); mm, mx = offdiag_cos(me)
print(f"INIT discrimination: text off-diag cos mean={tm:.3f} max={tx:.3f} | "
      f"motion off-diag cos mean={mm:.3f} max={mx:.3f}  (near 1.0 => collapsed)")

fmask = build_multi_positive_mask(coll["motion_id"], coll["source_motion_id"], coll["caption_text"])
for lr in [2e-3, 5e-4, 2e-4]:
    m = mk(); m.train()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    L = []
    for step in range(200):
        o = m(batch, caps); l = m.contrastive_loss(o, false_neg_mask=fmask)
        opt.zero_grad(set_to_none=True); l.backward()
        torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.0)
        opt.step(); L.append(l.item())
    m.eval()
    with torch.no_grad():
        te = m.encode_text(caps); me = m.encode_motion(batch)
    tm, _ = offdiag_cos(te); mm, _ = offdiag_cos(me)
    print(f"lr={lr:.0e}: loss {L[0]:.3f} -> {L[-1]:.3f} (best {min(L):.3f}, ln8={math.log(8):.3f}); "
          f"final text_cos={tm:.3f} motion_cos={mm:.3f}")
