"""Generate ONE clip per requested rig with a v2 in-context DiT and dump it as a
NORMALIZED [T,J,17] npy (the serve space), for downstream skinning via
_ktjd17_to_bvh.py --gen_npy. Same inference surface as v2_render_incontext.py /
_eval_v2_gen_in_evalspace.py (ckpt-derived PK, per-rig channel_valid/heading, anchor,
deployment cfg_text=2 / 20 steps by default). Target clip per rig: --pick longest or
energetic (max GT motion energy = mean frame-to-frame displacement of the DE-NORMALIZED
KTJD positions over the target window, the v2_render_incontext.py definition).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.incontext_pairs import InContextPairs, collate                     # noqa: E402
from src.data.ktjd17_incontext import Ktjd17Base, ktjd17_split_names             # noqa: E402
from src.models.v2.dit_motion import InContextMotionDiT, sample                  # noqa: E402
from scripts.v2_render_incontext import world_of_ktjd                            # noqa: E402


def gt_energy(base, rig, gt_it, Tt):
    """v2_render_incontext.py `--pick energetic` definition: mean frame-to-frame displacement
    of the de-normalized world positions over the head window (physical space, GT-only)."""
    Jn, Tn = int(gt_it["num_joints"]), min(int(gt_it["num_frames"]), Tt)
    if Tn < 2:
        return 0.0
    xn = np.asarray(gt_it["anytop_x"])[:Jn, :, :Tn].transpose(2, 0, 1)
    xw, _ = world_of_ktjd(xn, base, rig, strict_gt=True)
    return float(np.linalg.norm(np.diff(xw, axis=0), axis=-1).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rigs", required=True, help="comma-separated rig ids")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cfg_text", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pick", choices=("longest", "energetic"), default="energetic",
                    help="target clip per rig (energetic = the render scripts' max-GT-motion-energy rule)")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    ckpt_sha = hashlib.sha256(Path(a.ckpt).read_bytes()).hexdigest()
    ca = ck["args"]
    percell_path = ca.get("ktjd_percell_stats", "data/ktjd17_percell_stats_v1.npz")
    percell_sha = hashlib.sha256(Path(percell_path).read_bytes()).hexdigest()
    if str(ca.get("corpus")) != "ktjd17" or bool(ca.get("two_stage", False)):
        raise SystemExit("[refuse] expects a single-stage KTJD-17 ckpt")
    model = InContextMotionDiT(in_ch=17, dim=ca["dim"], depth=ca["depth"], n_heads=ca["heads"],
                               d_text=4096, d_joint_sem=4096,
                               use_struct_feats=bool(ca.get("struct_feats", False)),
                               use_dir_bias=bool(ca.get("dir_bias", False)),
                               qk_norm=bool(ca.get("qk_norm", False)),
                               use_ref_text=bool(ca.get("ref_text", False))).to(dev)
    model.load_state_dict(ck["model"]); model.eval()

    root, excl = ca["ktjd_root"], (ca.get("exclude_clips") or None)
    base = Ktjd17Base(root, caption_emb_cache=ca["caption_cache"], joint_semantics=ca["joint_sem"],
                      texts_json=ca["texts_json"],
                      percell_stats=ca.get("ktjd_percell_stats", "data/ktjd17_percell_stats_v1.npz"),
                      exclude_clips=excl)
    pins = ck.get("ktjd_pins") or {}
    live = {**base.provenance, "exclusion": base.provenance_exclusion}
    drift = sorted(k for k, v in pins.items() if k in live and live[k] != v)
    if drift:
        raise SystemExit(f"[refuse] data drift vs ckpt pins: {drift}")
    names = ktjd17_split_names(root, exclude=excl)
    PK = dict(demo_rest=bool(ca.get("demo_rest", False)), emit_ref_text=bool(ca.get("ref_text", False)),
              demo_frames=int(ca.get("demo_frames", 1)), target_frames=int(ca["target_frames"]),
              emit_graph_v2=bool(ca.get("struct_feats", False)) or bool(ca.get("dir_bias", False)))
    ds = InContextPairs(base, names["val"], names["train"], object_types=None,
                        balance_skeletons=False, seed=a.seed, **PK)
    anc_mode = str(ca.get("anchor", "none"))
    df = PK["demo_frames"]
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    meta = {}
    for rig in [r.strip() for r in a.rigs.split(",") if r.strip()]:
        pos = [i for i, (ot, _) in enumerate(ds.index) if ot == rig]
        if not pos:
            print(f"[gen] SKIP {rig}: no val targets", flush=True); continue
        if a.pick == "longest":
            p = max(pos, key=lambda i: int(base[ds.index[i][1]]["num_frames"]))
        else:
            p = max(pos, key=lambda i: gt_energy(base, rig, base[ds.index[i][1]], ds.Tt))
        ds._wrng_key = None
        item = ds[p]
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in collate([item]).items()}
        x_in = b["x"][..., :17].contiguous()
        g2kw = {k: b[k] for k in ("struct_feats", "updown") if k in b}
        cvj = torch.from_numpy(base.static_masks(rig)["channel_valid"]).to(dev)
        cv = torch.zeros(1, x_in.shape[2], 17, dtype=torch.bool, device=dev); cv[0, :cvj.shape[0]] = cvj
        g2kw["channel_valid"] = cv
        g2kw["heading_valid"] = b["x"][:, :, 0, 17] > 0.5
        if "demo_text" in b:
            g2kw["demo_text"] = b["demo_text"]
        if anc_mode != "none":
            from scripts.train_v2_incontext import ktjd_anchor
            lut = {rig: torch.from_numpy(base.rest_anchor_frame(rig))} if anc_mode == "rest" else None
            g2kw["anchor"] = ktjd_anchor(b, x_in, anc_mode, lut, df)
        torch.manual_seed(a.seed)
        with torch.no_grad():
            gen = sample(model, x_in, b["is_target"], a.steps, cfg_text=a.cfg_text, demo_frames=df,
                         joint_bias=b["joint_bias"], frame_valid=b["frame_valid"],
                         joint_valid=b["joint_valid"], text=b["text"], joint_sem=b["joint_sem"], **g2kw)
        J = int(item["n_joints"])
        T = int(item["frame_valid"][df:].sum())
        arr = gen[0, df:df + T, :J, :].float().cpu().numpy()             # [T,J,17] normalized
        cid = str(item["motion_id"])
        npy_path = out / f"{rig}__{cid}.npy"
        np.save(npy_path, arr)
        rec = {"rig": rig, "clip_id": cid, "frames": T, "joints": J, "cfg_text": a.cfg_text, "steps": a.steps,
               "seed": a.seed, "pick": a.pick, "caption": str(item.get("caption_text", item.get("caption", ""))),
               "npy_sha256": hashlib.sha256(npy_path.read_bytes()).hexdigest(),
               "ckpt": a.ckpt, "ckpt_sha256": ckpt_sha, "ckpt_epoch": int(ck.get("epoch", -1)),
               "percell_stats": percell_path, "percell_sha256": percell_sha, "ktjd_pins": pins}
        # sibling manifest: _ktjd17_to_bvh.py refuses a generated npy without it (provenance gate)
        npy_path.with_suffix(".manifest.json").write_text(json.dumps(rec, indent=2))
        meta[rig] = rec
        print(f"[gen] {rig} clip={cid} T={T} J={J} -> {out / (rig + '__' + cid + '.npy')}", flush=True)
    (out / "meta.json").write_text(json.dumps({"ckpt": a.ckpt, "epoch": int(ck.get("epoch", -1)),
                                               "clips": meta}, indent=2))


if __name__ == "__main__":
    main()
