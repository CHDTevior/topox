"""Sampling latency of a v2 checkpoint (measurement only; touches nothing in the protocol).

Builds the SAME val items generate_all() in scripts/_eval_v2_gen_in_evalspace.py builds (same dataset, seed,
demo construction, anchor, masks) for one rig, then times sample() at batch 1 for several Euler step counts.
With cfg_text != 1 the network runs twice per step (conditional + unconditional), so NFE = 2 * steps.
Run inside an allocation on one idle GPU:
  CUDA_VISIBLE_DEVICES=k python scripts/_measure_latency.py --ckpt <best_model.pt> --rig <rig> --steps 20,10,5 --out <json>
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import _eval_v2_gen_in_evalspace as ev                                 # noqa: E402
from src.data.incontext_pairs import InContextPairs, collate                        # noqa: E402
from src.data.ktjd17_incontext import Ktjd17Base, ktjd17_split_names                # noqa: E402
from src.models.v2.dit_motion import sample                                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rig", required=True, help="val rig; its first val target (dataset order) is timed")
    ap.add_argument("--steps", default="20,10,5", help="comma-separated Euler step counts")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--cfg_text", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    step_list = [int(x) for x in a.steps.split(",")]
    if any(S < 1 for S in step_list):
        raise SystemExit(f"[refuse] steps {step_list}: a non-positive count returns the base noise without calling the model")
    if not torch.cuda.is_available():
        raise SystemExit("[refuse] latency is a GPU measurement; no CUDA device visible")
    dev = torch.device("cuda")
    ck, sha = ev.hash_load(a.ckpt)
    model, ca = ev.load_gen_model(ck, dev)
    excl = ca.get("exclude_clips") or None
    base = Ktjd17Base(ca["ktjd_root"], caption_emb_cache=ca["caption_cache"], joint_semantics=ca["joint_sem"],
                      texts_json=ca["texts_json"],
                      percell_stats=ca.get("ktjd_percell_stats", "data/ktjd17_percell_stats_v1.npz"),
                      exclude_clips=excl, normalization=str(ca.get("rep_norm", "percell")))
    names = ktjd17_split_names(ca["ktjd_root"], exclude=excl)
    PK = dict(demo_rest=bool(ca.get("demo_rest", False)), emit_ref_text=bool(ca.get("ref_text", False)),
              demo_frames=int(ca.get("demo_frames", 1)), target_frames=int(ca["target_frames"]),
              emit_graph_v2=bool(ca.get("struct_feats", False)) or bool(ca.get("dir_bias", False)))
    ds = InContextPairs(base, names["val"], names["train"], object_types=None, balance_skeletons=False, seed=42, **PK)
    pos = [i for i, (rig, _) in enumerate(ds.index) if rig == a.rig]
    if not pos:
        raise SystemExit(f"[refuse] rig {a.rig!r} has no val target in this checkpoint's corpus")
    df = PK["demo_frames"]
    ds._wrng_key = None
    item = ds[pos[0]]
    b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in collate([item]).items()}
    x_in = b["x"][..., :17].contiguous()
    cvj = torch.from_numpy(base.static_masks(a.rig)["channel_valid"]).to(dev)
    cv = torch.zeros(x_in.shape[0], x_in.shape[2], 17, dtype=torch.bool, device=dev)
    cv[:, :cvj.shape[0]] = cvj
    g2kw = {k: b[k] for k in ("struct_feats", "updown") if k in b}
    g2kw["channel_valid"] = cv
    g2kw["heading_valid"] = b["x"][:, :, 0, 17] > 0.5
    if "demo_text" in b:
        g2kw["demo_text"] = b["demo_text"]
    anc_mode = str(ca.get("anchor", "none"))
    if anc_mode != "none":
        from scripts.train_v2_incontext import ktjd_anchor
        rest_lut = {a.rig: torch.from_numpy(base.rest_anchor_frame(a.rig))} if anc_mode == "rest" else None
        g2kw["anchor"] = ktjd_anchor(b, x_in, anc_mode, rest_lut, df)
    T_target = int(b["frame_valid"][0, df:].sum())
    J = int(cvj.shape[0])
    print(f"[latency] ckpt {a.ckpt} sha256 {sha[:12]} | rig {a.rig} J={J} target frames={T_target} (padded {x_in.shape[1] - df}) "
          f"| {torch.cuda.get_device_name(0)} fp32 batch 1 cfg_text {a.cfg_text}", flush=True)

    def run(S):                                  # reseed OUTSIDE the timed region (codex 2026-09-06 P3)
        return sample(model, x_in, b["is_target"], S, cfg_text=a.cfg_text, demo_frames=df,
                      joint_bias=b["joint_bias"], frame_valid=b["frame_valid"], joint_valid=b["joint_valid"],
                      text=b["text"], joint_sem=b["joint_sem"], **g2kw)

    res = {}
    with torch.no_grad():
        for S in step_list:
            for _ in range(2):                       # warm-up: kernels, allocator
                torch.manual_seed(42); run(S)
            torch.cuda.synchronize()
            ts = []
            for _ in range(a.repeats):
                torch.manual_seed(42)
                t0 = time.perf_counter()
                run(S)
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            res[str(S)] = {"ms_median": 1000 * statistics.median(ts), "ms_min": 1000 * min(ts),
                           "ms_max": 1000 * max(ts), "nfe": 2 * S if a.cfg_text != 1.0 else S}
            print(f"[latency] steps {S:3d}: median {res[str(S)]['ms_median']:8.1f} ms  min {res[str(S)]['ms_min']:.1f}  "
                  f"max {res[str(S)]['ms_max']:.1f}  ({a.repeats} timed runs, NFE {res[str(S)]['nfe']})", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps({
            "ckpt": a.ckpt, "ckpt_sha256": sha, "rig": a.rig, "J": J, "target_frames": T_target,
            "padded_frames": int(x_in.shape[1] - df), "device": torch.cuda.get_device_name(0), "dtype": "fp32",
            "cfg_text": a.cfg_text, "batch": 1, "repeats": a.repeats, "results": res}, indent=2))
        print(f"[latency] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
