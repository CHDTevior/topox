"""Pre-change snapshot of served items/batches for the no-op equivalence check of the skeleton augmentation
(36M r1acc recipe settings: rest demo, graph-v2, fk fields). Deterministic: main-process rng key (seed 0), fixed index draws.
usage: python scripts/_aug_dev/_snapshot_items.py <out.npz>"""
import sys, os
import numpy as np, torch
sys.path.insert(0, os.getcwd())
from src.data.ktjd17_incontext import Ktjd17Base, ktjd17_split_names
from src.data.incontext_pairs import InContextPairs, collate
ROOT, CUT = "dataset/ktjd17_pzh312_noik_v2", "configs/pilot_animal_only_exclusions.json"
base = Ktjd17Base(ROOT, caption_emb_cache="data/noik_caption_llm2vec_v1", joint_semantics="data/joint_semantics_llm2vec_pzh312_v1.npz",
                  texts_json="data/noik_pzh312_motion_texts_v1.json", percell_stats="data/noik_norm_stats_v2.npz",
                  exclude_clips=CUT, random_caption=False, normalization="percell")
names = ktjd17_split_names(ROOT, exclude=CUT)
ds = InContextPairs(base, names["train"], names["train"], demo_frames=1, target_frames=240, balance_skeletons=True, seed=0,
                    emit_fk_fields=True, emit_graph_v2=True, identity_p=0.0, emit_ref_text=False, demo_rest=True)
items = [ds[i] for i in range(12)]
out = {}
for k in range(3):
    b = collate(items[4 * k:4 * k + 4])
    for key, v in b.items():
        if torch.is_tensor(v): out[f"b{k}__{key}"] = v.numpy()
        else: out[f"b{k}__{key}"] = np.array(v)
rigs = sorted({it["object_type"] for it in items})
for r in rigs:
    out[f"cv__{r}"] = base.static_masks(r)["channel_valid"]
    out[f"rest__{r}"] = base.rest_frame_normalized(r)
    out[f"anchor__{r}"] = base.rest_anchor_frame(r)
np.savez(sys.argv[1], **out)
print("saved", sys.argv[1], "rigs", rigs, "keys", len(out))
