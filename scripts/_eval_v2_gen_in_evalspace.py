"""Frozen-evaluator generation eval for the v2 in-context DiT (KTJD-17 PZ corpus).

Protocol (user 2026-08-29): FULL val (3,899 clips) + pool size 32, dataset-order pools --
identical chunking to _eval_evaluator_sanity.py, so text->GEN R@K is directly comparable
to that script's text->GT ceiling (0.975 for evaluator_ktjd16_pz_v1). Inference is the
deployment config: 20-step ODE, cfg_text=2, 1-frame rest demo.

Metrics: text->gen R@1/2/3 (group-aware, pool 32) | text->GT same-pool ceiling |
matching score (diagonal cos) | FID(gen, GT) in the 512-d evaluator space | gen<->GT cos.
Scores are PZ-only/16ch/T=240 -- NOT comparable to legacy 13ch evaluator numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.incontext_pairs import InContextPairs, collate                     # noqa: E402
from src.data.ktjd17_incontext import Ktjd17Base, ktjd17_split_names             # noqa: E402
from src.data.ktjd17_t2m_eval_dataset import Ktjd17T2MEvalDataset, KEEP_CH, J_MAX  # noqa: E402
from src.data.anytop_t2m_eval_dataset import collate_fn as eval_collate          # noqa: E402
from src.models.graph_salad.batch import GraphMotionBatch                        # noqa: E402
from src.models.graph_salad.t2m_evaluator import AnyTopT2MEvaluator              # noqa: E402
from src.models.v2.dit_motion import InContextMotionDiT, sample                  # noqa: E402
from scripts._eval_evaluator_sanity import avg_over_pools                        # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen_ckpt", required=True, help="v2 DiT ckpt (best_model.pt).")
    ap.add_argument("--eval_ckpt", required=True, help="frozen evaluator ckpt.")
    ap.add_argument("--out", default=None, help="JSON report path.")
    ap.add_argument("--cfg_text", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gen_batch", type=int, default=16)
    ap.add_argument("--encode_batch", type=int, default=64)
    ap.add_argument("--pool", type=int, default=32)
    a = ap.parse_args()
    # The protocol is FROZEN (user 2026-08-29: full val + pool 32; deployment inference
    # 20-step/cfg2). Changing any of these is a protocol change -> edit this pin on purpose.
    if (a.steps, a.cfg_text, a.pool) != (20, 2.0, 32):
        raise SystemExit(f"[refuse] protocol pin is steps=20/cfg_text=2.0/pool=32; "
                         f"got {a.steps}/{a.cfg_text}/{a.pool}")
    return a


PROTOCOL_VAL_N = 3899   # frozen val size; a different corpus cut must re-pin on purpose


def hash_load(path):
    """sha256 + torch.load from ONE read of the file, so the reported hash is the hash
    of the weights actually evaluated (best_model.pt gets atomically replaced mid-training;
    codex fix3 observed epoch 344 -> 354 during one review)."""
    import hashlib
    import io
    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False), sha


def load_gen_model(ck, dev):
    ca = ck["args"]
    if str(ca.get("corpus")) != "ktjd17":
        raise SystemExit(f"[refuse] gen ckpt corpus={ca.get('corpus')!r}; this eval is KTJD-17 only")
    if bool(ca.get("two_stage", False)):
        raise SystemExit("[refuse] two_stage ckpts are not wired here")
    model = InContextMotionDiT(
        in_ch=17, dim=ca["dim"], depth=ca["depth"], n_heads=ca["heads"],
        d_text=4096, d_joint_sem=4096,
        use_struct_feats=bool(ca.get("struct_feats", False)),
        use_dir_bias=bool(ca.get("dir_bias", False)),
        qk_norm=bool(ca.get("qk_norm", False)),
        use_ref_text=bool(ca.get("ref_text", False))).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ca


def load_evaluator(path, dev):
    ck, sha = hash_load(path)
    m = ck.get("args", {})
    g = (lambda k, d: m.get(k, d)) if isinstance(m, dict) else (lambda k, d: getattr(m, k, d))
    if int(g("motion_feat_dim", 13)) != 16:
        raise SystemExit(f"[refuse] evaluator motion_feat_dim={g('motion_feat_dim', 13)}; "
                         "this eval feeds 16ch KTJD tensors")
    core = AnyTopT2MEvaluator(
        coemb_dim=g("coemb_dim", 512), text_tower=g("text_tower", "distilbert"),
        distilbert_path=g("distilbert_path", "checkpoints/text_encoders/distilbert-base-uncased"),
        text_max_length=g("text_max_length", 64),
        n_heads=g("n_heads", 8), d_ff=g("d_ff", 2048),
        n_graph_layers=g("n_graph_layers", 6), n_temporal_layers=g("n_temporal_layers", 4),
        motion_feat_dim=16, dropout=g("dropout", 0.1),
        learnable_temperature=not g("fixed_temperature", False), temperature=g("temperature", 0.07),
        strict_frame_masking=g("strict_frame_masking", False))
    core.load_state_dict(ck["model"])
    core.to(dev).eval()
    print(f"[gen-eval] evaluator {path} (epoch={ck.get('epoch', '?')} "
          f"val={ck.get('val', {}).get('r1', '?')} sha256={sha[:16]})", flush=True)
    return core, sha


@torch.no_grad()
def generate_all(model, ca, base, names, dev, a):
    """One generated [T,J,17] (normalized space) per val target, keyed by motion_id."""
    PK = dict(demo_rest=bool(ca.get("demo_rest", False)),
              emit_ref_text=bool(ca.get("ref_text", False)),
              demo_frames=int(ca.get("demo_frames", 1)),
              target_frames=int(ca["target_frames"]),
              emit_graph_v2=bool(ca.get("struct_feats", False)) or bool(ca.get("dir_bias", False)))
    if PK["demo_rest"] and PK["demo_frames"] != 1:
        raise SystemExit("[refuse] demo_rest ckpt with demo_frames != 1")
    ds = InContextPairs(base, names["val"], names["train"], object_types=None,
                        balance_skeletons=False, seed=a.seed, **PK)
    anc_mode = str(ca.get("anchor", "none"))
    rest_lut = {}
    if anc_mode == "rest":
        from scripts.train_v2_incontext import ktjd_anchor  # noqa: F401 (used below)
    by_rig: dict[str, list[int]] = {}
    for i, (rig, _) in enumerate(ds.index):
        by_rig.setdefault(rig, []).append(i)
    gen_by_clip: dict[str, np.ndarray] = {}
    df = PK["demo_frames"]
    n_done = 0
    for rig in sorted(by_rig):
        pos = by_rig[rig]
        cvj = torch.from_numpy(base.static_masks(rig)["channel_valid"]).to(dev)
        for s in range(0, len(pos), a.gen_batch):
            chunk = pos[s:s + a.gen_batch]
            items = []
            for p in chunk:
                ds._wrng_key = None          # per-item stream reset (render-script convention)
                items.append(ds[p])
            b = {k: (v.to(dev) if torch.is_tensor(v) else v)
                 for k, v in collate(items).items()}
            x_in = b["x"][..., :17].contiguous()
            g2kw = {k: b[k] for k in ("struct_feats", "updown") if k in b}
            cv = torch.zeros(x_in.shape[0], x_in.shape[2], 17, dtype=torch.bool, device=dev)
            cv[:, :cvj.shape[0]] = cvj
            g2kw["channel_valid"] = cv
            g2kw["heading_valid"] = b["x"][:, :, 0, 17] > 0.5
            if "demo_text" in b:
                g2kw["demo_text"] = b["demo_text"]
            if anc_mode != "none":
                from scripts.train_v2_incontext import ktjd_anchor
                if anc_mode == "rest" and rig not in rest_lut:
                    rest_lut[rig] = torch.from_numpy(base.rest_anchor_frame(rig))
                g2kw["anchor"] = ktjd_anchor(b, x_in, anc_mode,
                                             rest_lut if anc_mode == "rest" else None, df)
            torch.manual_seed(a.seed)
            gen = sample(model, x_in, b["is_target"], a.steps, cfg_text=a.cfg_text,
                         demo_frames=df,
                         joint_bias=b["joint_bias"], frame_valid=b["frame_valid"],
                         joint_valid=b["joint_valid"], text=b["text"],
                         joint_sem=b["joint_sem"], **g2kw)
            gen = gen.float().cpu().numpy()   # [B, df+Tt, J, 17]
            for bi, p in enumerate(chunk):
                mid = str(items[bi]["motion_id"])
                if mid in gen_by_clip:
                    raise SystemExit(f"[refuse] duplicate generated motion_id {mid}")
                gen_by_clip[mid] = gen[bi, df:].astype(np.float32)   # target frames only
                # fp32 kept on purpose: fp16 caching quantizes the samples that feed the
                # evaluator/FID (codex fix2); ~3GB RAM at 3,899 clips, fine on these nodes
            n_done += len(chunk)
        if n_done and (len(gen_by_clip) % 512 < a.gen_batch):
            print(f"[gen-eval] generated {n_done}/{len(ds.index)}", flush=True)
    if len(gen_by_clip) != len(ds.index):
        raise SystemExit(f"[refuse] generated {len(gen_by_clip)} != targets {len(ds.index)}")
    if len(gen_by_clip) != PROTOCOL_VAL_N:
        raise SystemExit(f"[refuse] protocol val is {PROTOCOL_VAL_N}, generated {len(gen_by_clip)}")
    print(f"[gen-eval] generated {len(gen_by_clip)} clips", flush=True)
    return gen_by_clip


def gen_to_anytop_x(gen_tjc: np.ndarray, J: int, T: int, Tt: int) -> torch.Tensor:
    """[T_target, J_serve, 17] normalized -> evaluator anytop_x [J_MAX, 16, Tt]."""
    x = gen_tjc[:T, :J, :].transpose(1, 2, 0)                  # [J, 17, T]
    x16 = np.zeros((J_MAX, 16, Tt), dtype=np.float32)
    x16[:J, :, :T] = x[:, KEEP_CH, :]
    return torch.from_numpy(x16)


@torch.no_grad()
def encode_split(core, eval_ds, gen_by_clip, dev, a):
    te, me_gt, me_gen = [], [], []
    meta = {"motion_id": [], "source_motion_id": [], "caption_text": []}
    missing = []
    buf = []

    def flush():
        if not buf:
            return
        coll = eval_collate([it for it, _ in buf])
        batch = GraphMotionBatch.from_collate_dict(
            {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in coll.items()})
        te.append(core.encode_text(coll["caption_text"]).float().cpu())
        me_gt.append(core.encode_motion(batch).float().cpu())
        gx = torch.stack([gx_ for _, gx_ in buf]).to(dev)
        import dataclasses
        me_gen.append(core.encode_motion(
            dataclasses.replace(batch, anytop_x=gx)).float().cpu())
        buf.clear()

    for i in range(len(eval_ds)):
        it = eval_ds[i]
        mid = str(it["motion_id"])
        if mid not in gen_by_clip:
            missing.append(mid)
            continue
        J = int(it["num_joints"]); T = int(it["num_frames"])
        buf.append((it, gen_to_anytop_x(gen_by_clip[mid], J, T, eval_ds.Tt)))
        meta["motion_id"].append(mid)
        meta["source_motion_id"].append(str(it["source_motion_id"]))
        meta["caption_text"].append(str(it["caption_text"]))
        if len(buf) >= a.encode_batch:
            flush()
        if (i + 1) % 1024 == 0:
            print(f"[gen-eval] encoded {i+1}/{len(eval_ds)}", flush=True)
    flush()
    if missing:
        raise SystemExit(f"[refuse] {len(missing)} val clips have no generation "
                         f"(e.g. {missing[:5]}); generation must cover the full protocol set")
    return (torch.nn.functional.normalize(torch.cat(te), dim=-1),
            torch.nn.functional.normalize(torch.cat(me_gt), dim=-1),
            torch.nn.functional.normalize(torch.cat(me_gen), dim=-1), meta)


def fid(a_emb: torch.Tensor, b_emb: torch.Tensor) -> float:
    from scipy import linalg
    x, y = a_emb.double().numpy(), b_emb.double().numpy()
    mu1, mu2 = x.mean(0), y.mean(0)
    s1 = np.cov(x, rowvar=False)
    s2 = np.cov(y, rowvar=False)
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2.0 * covmean))


def main():
    a = parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck, gen_sha = hash_load(a.gen_ckpt)
    ca = ck["args"]
    model, ca = load_gen_model(ck, dev)
    print(f"[gen-eval] gen ckpt {a.gen_ckpt} (epoch {ck.get('epoch', -1)} "
          f"sha256={gen_sha[:16]}) cfg_text={a.cfg_text} steps={a.steps}", flush=True)

    root = ca["ktjd_root"]
    excl = ca.get("exclude_clips") or None
    base = Ktjd17Base(root, caption_emb_cache=ca["caption_cache"],
                      joint_semantics=ca["joint_sem"], texts_json=ca["texts_json"],
                      percell_stats=ca.get("ktjd_percell_stats",
                                           "data/ktjd17_percell_stats_v1.npz"),
                      exclude_clips=excl)
    pins_ck = ck.get("ktjd_pins") or {}
    live = {**base.provenance, "exclusion": base.provenance_exclusion}
    drift = sorted(k for k, v in pins_ck.items() if k in live and live[k] != v)
    if drift:
        raise SystemExit(f"[refuse] data drift vs gen-ckpt pins: {drift}")
    names = ktjd17_split_names(root, exclude=excl)

    gen_by_clip = generate_all(model, ca, base, names, dev, a)
    del model
    torch.cuda.empty_cache()

    core, eval_sha = load_evaluator(a.eval_ckpt, dev)
    # eval-side base: SAME root/percell/exclude as the generator ckpt, so both towers see
    # one normalization space and one val set.
    base_eval = Ktjd17Base(root, caption_emb_cache=ca["caption_cache"],
                           joint_semantics=ca["joint_sem"], texts_json=ca["texts_json"],
                           percell_stats=ca.get("ktjd_percell_stats",
                                                "data/ktjd17_percell_stats_v1.npz"),
                           exclude_clips=excl)
    eval_ds = Ktjd17T2MEvalDataset(base_eval, "val", max_frames=240, exclude=excl)
    if len(eval_ds) != PROTOCOL_VAL_N:
        raise SystemExit(f"[refuse] eval val has {len(eval_ds)} clips, protocol pins "
                         f"{PROTOCOL_VAL_N}")
    te, me_gt, me_gen, meta = encode_split(core, eval_ds, gen_by_clip, dev, a)

    gpool = torch.Generator().manual_seed(a.seed)
    n = te.shape[0]
    report = {"protocol": {"val_n": n, "pool": a.pool, "cfg_text": a.cfg_text,
                           "steps": a.steps, "seed": a.seed,
                           "gen_ckpt": a.gen_ckpt, "gen_epoch": int(ck.get("epoch", -1)),
                           "gen_ckpt_sha256": gen_sha,
                           "eval_ckpt": a.eval_ckpt, "eval_ckpt_sha256": eval_sha,
                           "note": "PZ-only/16ch/T=240; pools chunk dataset order like "
                                   "_eval_evaluator_sanity.py"}}
    for tag, me in (("text_to_gen", me_gen), ("text_to_gt_ceiling", me_gt)):
        rr, npool = avg_over_pools(te, me, meta, a.pool, masked=True, shuffled=False, gen=gpool)
        used = npool * a.pool
        report[tag] = {"rprec": rr, "n_pools": npool, "n_used": used}
        print(f"[gen-eval] {tag:<19} R@1={rr[1]:.3f} R@2={rr[2]:.3f} R@3={rr[3]:.3f} "
              f"({npool} pools of {a.pool}, used {used}/{n})", flush=True)
    report["matching"] = {
        "text_gen_cos": float((te * me_gen).sum(-1).mean()),
        "text_gt_cos": float((te * me_gt).sum(-1).mean()),
        "gen_gt_cos": float((me_gen * me_gt).sum(-1).mean())}
    report["fid_gen_vs_gt"] = fid(me_gen, me_gt)
    print(f"[gen-eval] matching text-gen={report['matching']['text_gen_cos']:.3f} "
          f"text-GT={report['matching']['text_gt_cos']:.3f} "
          f"gen-GT={report['matching']['gen_gt_cos']:.3f}", flush=True)
    print(f"[gen-eval] FID(gen, GT) = {report['fid_gen_vs_gt']:.4f}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2))
        print(f"[gen-eval] report -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
