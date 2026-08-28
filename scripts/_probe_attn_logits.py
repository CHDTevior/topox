#!/usr/bin/env python3
"""Measure attention-logit magnitude on real checkpoints.

Unnormalised q@k logits growing with training is the canonical source of late-training sharpness
(ViT-22B / PaLM instabilities; qk-norm is the standard fix). Measured 2026-08-26 on run10: max
logit 1372-1483 while HEALTHY, 12392-23391 across the ep34 damage step -- the evidence that put
qk-norm into the architecture. The probe is CHECKPOINT-AWARE (codex 2026-08-26 follow-up item 4):
corpus root / exclusion cut / gamma calib / arch flags all come from the checkpoint's own args,
and for a qk_norm=True checkpoint the logits are measured AFTER q/k RMS-normalisation -- the
quantity softmax actually sees. Data setup happens lazily per checkpoint (grouped by identical
data args), so mixed-lineage invocations stay correct.
"""
import json, os, sys
from pathlib import Path
import torch
sys.path.insert(0, "/iridisfs/scratch/ts1v23/workspace/noKslot_clean")
from src.data.ktjd17_incontext import Ktjd17Base, ktjd17_split_names
from src.data.incontext_pairs import InContextPairs, collate
from src.models.v2.dit_motion import InContextMotionDiT
from scripts.train_v2_incontext import ktjd_channel_lut, ktjd_prep, cond_of, to_dev
from torch.utils.data import DataLoader

dev = "cuda:0"
_data_cache: dict[tuple, tuple] = {}

def data_for(a: dict):
    """(lut, gammas, batches) for a checkpoint's OWN data args. Every argument that reaches
    Ktjd17Base or InContextPairs is taken from the checkpoint and is part of the cache key, so
    probing checkpoints from different lineages in one invocation can never share wrong data.
    A checkpoint missing any of these keys is an unsupported lineage and fails on the KeyError."""
    key = (a["ktjd_root"], a.get("exclude_clips") or "", a["ktjd_gamma_calib"],
           a["caption_cache"], a["joint_sem"], a["ktjd_percell_stats"], a["texts_json"],
           bool(a.get("random_caption", False)), a["balance"], int(a["seed"]),
           int(a.get("demo_frames", 1)), int(a["target_frames"]),
           float(a.get("identity_p", 0.0)), bool(a.get("ref_text", False)),
           bool(a.get("demo_rest", False)))
    if key in _data_cache:
        return _data_cache[key]
    R = a["ktjd_root"]
    cut = a.get("exclude_clips") or None
    base = Ktjd17Base(R, caption_emb_cache=a["caption_cache"],
                      joint_semantics=a["joint_sem"],
                      percell_stats=a["ktjd_percell_stats"],
                      exclude_clips=cut,
                      texts_json=a["texts_json"],
                      random_caption=bool(a.get("random_caption", False)))
    names = ktjd17_split_names(R, exclude=cut)
    ds = InContextPairs(base, names["train"], names["train"],
                        balance_skeletons=(a["balance"] == "rig"), seed=int(a["seed"]),
                        emit_fk_fields=True, emit_graph_v2=True,
                        demo_frames=int(a.get("demo_frames", 1)),
                        target_frames=int(a["target_frames"]),
                        identity_p=float(a.get("identity_p", 0.0)),
                        emit_ref_text=bool(a.get("ref_text", False)),
                        demo_rest=bool(a.get("demo_rest", False)))
    lut = ktjd_channel_lut(base)
    G = {k: float(v) for k, v in
         json.loads(Path(a["ktjd_gamma_calib"]).read_text())["gammas"].items()}
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2, collate_fn=collate,
                    generator=torch.Generator().manual_seed(3))
    it = iter(dl)
    batches = [to_dev(next(it), dev) for _ in range(4)]
    _data_cache[key] = (lut, G, batches)
    return _data_cache[key]

def probe(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    m = InContextMotionDiT(in_ch=17, dim=a["dim"], depth=a["depth"], n_heads=a["heads"],
                           d_text=4096, d_joint_sem=4096,
                           use_struct_feats=bool(a.get("struct_feats", False)),
                           use_dir_bias=bool(a.get("dir_bias", False)),
                           use_ref_text=bool(a.get("ref_text", False)),
                           qk_norm=bool(a.get("qk_norm", False))).to(dev)
    m.load_state_dict(ck["model"]); m.eval()
    stats = {}   # name -> max logit seen
    hooks = []
    def mk(name, att):
        def hook(mod, inp, out):
            x = inp[0]
            N = x.shape[-2]
            qkv = att.qkv(x).reshape(*x.shape[:-2], N, 3, att.h, att.dh)
            q, k, _ = (t.transpose(-3, -2) for t in qkv.movedim(-3, 0))
            if getattr(att, "qk_norm", False):
                q, k = att.q_norm(q), att.k_norm(k)
            lg = (q.float() @ k.float().transpose(-1, -2)) / (att.dh ** 0.5)
            s = stats.setdefault(name, [0.0, 0.0])
            s[0] = max(s[0], float(lg.max()))
            s[1] = max(s[1], float(lg.abs().mean()))
        return hook
    for i, blk in enumerate(m.blocks):
        hooks.append(blk.t_attn.register_forward_hook(mk(f"b{i:02d}.t", blk.t_attn)))
        hooks.append(blk.s_attn.register_forward_hook(mk(f"b{i:02d}.s", blk.s_attn)))
    lut, G, batches = data_for(a)
    use_bf16 = bool(a.get("bf16", False))
    with torch.no_grad():
        for b in batches:
            x17, kt = ktjd_prep(b, lut, gammas=G)
            # xt EXACTLY as cfm_loss builds it (dit_motion.py:620-645): effective-validity
            # projection of clean data AND base noise, OT interpolation at t=0.5, demo frames
            # substituted with clean (projected) x1. anchor=none for every probed lineage --
            # the trainer refuses --anchor rest, and no run has used demo-anchoring.
            torch.manual_seed(7)
            x1 = x17
            eff = kt["channel_valid"][:, None].to(x1.dtype).expand_as(x1).clone()
            eff[:, :, 0, 15:17] *= kt["heading_valid"].to(x1.dtype)[:, :, None]
            x1 = x1 * eff
            x0 = torch.randn_like(x1) * eff
            tt = torch.full((x1.shape[0],), 0.5, device=dev)
            xt = 0.5 * x0 + 0.5 * x1
            xt = torch.where((~b["is_target"])[..., None, None], x1, xt)
            c = cond_of(b)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                m(xt, tt, is_target=b["is_target"],
                  joint_sem=c.get("joint_sem"), text=c.get("text"),
                  joint_bias=c.get("joint_bias"), frame_valid=c.get("frame_valid"),
                  joint_valid=c.get("joint_valid"), struct_feats=c.get("struct_feats"),
                  updown=c.get("updown"))
    for h in hooks: h.remove()
    del m; torch.cuda.empty_cache()
    mx = max(v[0] for v in stats.values())
    top = sorted(stats.items(), key=lambda kv: -kv[1][0])[:4]
    return ck["epoch"], mx, top

for path in sys.argv[1:]:
    ep, mx, top = probe(path)
    print(f"{Path(path).name}  epoch={ep}  MAX logit={mx:8.2f}   "
          f"top: {[(n, round(v[0],1)) for n, v in top]}", flush=True)
