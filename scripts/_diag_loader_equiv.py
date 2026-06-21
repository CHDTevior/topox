"""DIAGNOSTIC (not a gate): does my recon-eval GT loader (base AnyTopDataset, val.txt /
val_frac+seed) load motions IDENTICALLY to the evaluator's canonical loader
(AnyTopT2MEvalDataset, val_all.json manifest)? If yes, the recon-eval GT embeddings are
valid and the low recon→GT cosine is real; if no, the recon-eval must switch to the
evaluator's loader. For matched motion_ids: compare anytop_x (max abs diff) + the
evaluator motion embedding cosine (want ~0 and ~1.0)."""
from __future__ import annotations
import random
import numpy as np
import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.anytop_t2m_eval_dataset import AnyTopT2MEvalDataset, collate_fn as ev_collate
from src.data.anytop_dataset import AnyTopDataset, collate_fn as base_collate
from src.models.graph_salad.batch import GraphMotionBatch
from src.models.graph_salad.t2m_evaluator import AnyTopT2MEvaluator

D = "data/animo4d_anytop_clean_L4_safe_plus_truebones"
EVCK = "runs/anytop_t2m_evaluator_distilbert_coemb512_gb128_lr1e-4_seed42/best_model.pt"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ev = AnyTopT2MEvalDataset(manifest_path=f"{D}/eval_splits/val_all.json", data_root=D,
                          caption_emb_cache=None, split="val", view="full",
                          num_frames=300, max_joints=144)
base = AnyTopDataset(split="val", num_frames=300, max_joints=144, val_frac=0.05, seed=42,
                     load_captions=False, data_root=D)
print(f"[diag] ev(AnyTopT2MEvalDataset/val_all.json)={len(ev)}  base(AnyTopDataset/val)={len(base)}")

# cheap motion_id sets
ev_ids, ev_id2idx = [], {}
for i in range(len(ev)):
    mid = ev._plan[i][1]["motion_id"]
    ev_ids.append(mid); ev_id2idx[mid] = i
base_ids = [base.samples[i].get("motion_id", base.samples[i].get("path")) for i in range(len(base.samples))] \
    if hasattr(base, "samples") else None
if base_ids is not None:
    es, bs = set(ev_ids), set(base_ids)
    print(f"[diag] id-set: ev={len(es)} base={len(bs)} common={len(es & bs)} "
          f"ev_only={len(es - bs)} base_only={len(bs - es)}")

eck = torch.load(EVCK, map_location="cpu")
ea = eck["args"]
core = AnyTopT2MEvaluator(coemb_dim=ea["coemb_dim"], text_tower=ea["text_tower"],
                          distilbert_path=ea["distilbert_path"], text_max_length=ea["text_max_length"],
                          n_heads=ea["n_heads"], d_ff=ea["d_ff"], n_graph_layers=ea["n_graph_layers"],
                          n_temporal_layers=ea["n_temporal_layers"],
                          motion_feat_dim=ea.get("motion_feat_dim", 13),  # 12ch ckpts rebuild at 12
                          dropout=ea["dropout"],
                          learnable_temperature=not ea["fixed_temperature"], temperature=ea["temperature"])
core.load_state_dict(eck["model"], strict=False)
core.to(dev).eval()


def emb(item, collate):
    coll = collate([item])
    coll = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in coll.items()}
    b = GraphMotionBatch.from_collate_dict(coll)
    with torch.no_grad():
        return core.encode_motion(b).float().cpu()[0]


random.seed(0)
sample = random.sample(range(len(base)), 12)
print("[diag] per-clip loader equivalence (same motion_id, base vs ev):")
n_ok = n_miss = 0
for bi in sample:
    b_item = base[bi]
    mid = b_item.get("motion_id")
    if mid not in ev_id2idx:
        print(f"   {str(mid)[:48]:48s}  NOT in ev val_all")
        n_miss += 1
        continue
    e_item = ev[ev_id2idx[mid]]
    bx, ex = np.asarray(b_item["anytop_x"]), np.asarray(e_item["anytop_x"])
    shape_ok = bx.shape == ex.shape
    maxdiff = (np.abs(bx - ex).max() if shape_ok else float("nan"))
    cos = float((emb(b_item, base_collate) * emb(e_item, ev_collate)).sum())
    print(f"   {str(mid)[:48]:48s}  shape {bx.shape}=={ex.shape}? {shape_ok}  "
          f"anytop_x_maxdiff={maxdiff:.2e}  embed_cos={cos:.4f}")
    n_ok += 1
print(f"[diag] {n_ok} compared, {n_miss} base-ids missing from ev val_all")
