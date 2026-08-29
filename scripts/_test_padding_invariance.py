"""Padding-invariance test for the KTJD-17 evaluator motion tower (codex fix #1).

Two assertions on a real val batch containing genuinely padded clips (num_frames<240):
  1. strict_frame_masking=True  (evaluator default): scribbling large noise into the
     padded frames of anytop_x must NOT change motion_emb (invariance).
  2. strict_frame_masking=False (legacy behaviour): the SAME scribble MUST change
     motion_emb -- proves this test actually catches the leak it guards against.
CPU/fp32, fresh random weights (the leak is structural, not learned).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.anytop_t2m_eval_dataset import collate_fn
from src.data.ktjd17_incontext import Ktjd17Base
from src.data.ktjd17_t2m_eval_dataset import Ktjd17T2MEvalDataset
from src.models.graph_salad.batch import GraphMotionBatch
from src.models.graph_salad.t2m_evaluator import AnyTopT2MEvaluator

EXC = "configs/pilot_animal_only_exclusions.json"   # pure-PZ: same exclude as the trainer


def build_batch():
    base = Ktjd17Base("dataset/ktjd17_pzh312_noik_v2",
                      caption_emb_cache="data/noik_caption_llm2vec_v1",
                      joint_semantics="data/joint_semantics_llm2vec_pzh312_v1.npz",
                      percell_stats="data/noik_norm_stats_v2.npz",
                      exclude_clips=EXC,
                      texts_json="data/noik_pzh312_motion_texts_v1.json")
    ds = Ktjd17T2MEvalDataset(base, "val", max_frames=240, exclude=EXC)
    items, want_padded = [], 3
    for i in range(len(ds)):
        it = ds[i]
        n_valid = int(it["frame_mask"].sum())
        if n_valid < 200 and want_padded > 0:          # genuinely padded clip
            items.append(it); want_padded -= 1
        elif len(items) - (3 - want_padded) < 1:       # plus one near-full clip
            items.append(it)
        if len(items) >= 4 and want_padded == 0:
            break
    assert want_padded == 0, "val split has no clip with <200 valid frames?"
    coll = collate_fn(items)
    global _COLL
    _COLL = coll
    return GraphMotionBatch.from_collate_dict(coll)


@torch.no_grad()
def emb_with(model, batch, scribble: bool):
    import dataclasses
    x = batch.anytop_x.clone()                          # [B, J, C, T]
    if scribble:
        pad = ~batch.frame_mask.bool()                  # [B, T]
        noise = torch.randn_like(x) * 10.0
        x = torch.where(pad[:, None, None, :].expand_as(x), noise, x)
    return model.encode_motion(dataclasses.replace(batch, anytop_x=x))


def main():
    torch.manual_seed(0)
    batch = build_batch()
    print(f"[test] batch: anytop_x {tuple(batch.anytop_x.shape)} "
          f"valid-frames per clip: {[int(v) for v in batch.frame_mask.sum(1)]}")
    results = {}
    for strict in (True, False):
        torch.manual_seed(1234)
        model = AnyTopT2MEvaluator(coemb_dim=64, n_heads=4, d_ff=128,
                                   n_graph_layers=1, n_temporal_layers=2,
                                   motion_feat_dim=16, dropout=0.0,
                                   text_tower="t5_cache",
                                   strict_frame_masking=strict).eval()
        assert all(b.strict_mask is strict for b in model.motion_encoder.temporal_layers)
        e_clean = emb_with(model, batch, scribble=False)
        e_dirty = emb_with(model, batch, scribble=True)
        delta = (e_clean - e_dirty).abs().max().item()
        results[strict] = delta
        print(f"[test] strict_frame_masking={strict}: max|emb_clean-emb_scribbled| = {delta:.3e}")
    assert results[True] < 1e-5, f"LEAK with strict masking: {results[True]:.3e}"
    assert results[False] > 1e-3, (
        f"legacy path shows no leak ({results[False]:.3e}) -- test cannot catch failures")
    print("PADDING-INVARIANCE TEST PASS")

    # ---- regression: wrapper -> collate metadata reaches sanity's collectors ----
    # (batch above already went through collate_fn; _COLL is that very dict)
    for key in ("object_type", "motion_id", "source_motion_id", "caption_text", "source"):
        assert key in _COLL, f"collate dict lost metadata key {key!r}"
        assert len(_COLL[key]) == batch.anytop_x.shape[0]
    assert all(isinstance(v, str) and v for v in _COLL["object_type"]), "object_type must be rig strings"
    print("METADATA CHAIN TEST PASS:", {k: _COLL[k][0] for k in ("object_type", "source_motion_id")})

    # ---- regression: ckpt-args round trip restores the strict flag ----
    for stored, want in (({}, False), ({"strict_frame_masking": True}, True)):
        g = lambda k, d: stored.get(k, d)
        m2 = AnyTopT2MEvaluator(coemb_dim=64, n_heads=4, d_ff=128, n_graph_layers=1,
                                n_temporal_layers=2, motion_feat_dim=16, dropout=0.0,
                                text_tower="t5_cache",
                                strict_frame_masking=g("strict_frame_masking", False))
        assert all(b.strict_mask is want for b in m2.motion_encoder.temporal_layers), \
            f"ckpt args {stored} -> strict should rebuild {want}"
    print("CKPT-ARGS ROUND-TRIP TEST PASS (missing field -> False, True -> True)")


if __name__ == "__main__":
    main()
