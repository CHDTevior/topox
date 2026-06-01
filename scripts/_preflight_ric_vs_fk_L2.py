"""Preflight (plan §7): scan RIC(gt) vs FK(gt_rot6d) mismatch on clean_L2.

For a sample of train+val clips, compute || recover_from_bvh_rot_np(gt) -
_recover_world_positions(gt) ||, as % of motion bbox diagonal. This is the FK
loss FLOOR: even pred==gt won't give zero fk loss if the two GT routes disagree.

Uses item's own rest_offsets/parent_indices (same new_to_old_perm as anytop_x),
matching how the loss will run. helper-name rule from the user's script.

Run on rose11/login: python scripts/_preflight_ric_vs_fk_L2.py
"""
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.anytop_dataset import AnyTopDataset, _recover_world_positions, _STD_FLOOR  # noqa
from src.data.anytop_rot6d_fk import recover_from_bvh_rot_np  # noqa

HELPER_TOKENS = ["end_site", "twist", "srb", "breath"]


def stats(v):
    v = np.asarray(v, np.float64)
    if v.size == 0:
        return {k: float("nan") for k in ("mean", "median", "p95", "p99", "max")}
    return {"mean": float(v.mean()), "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)), "p99": float(np.percentile(v, 99)),
            "max": float(v.max())}


def scan(split, n):
    ds = AnyTopDataset(split=split, val_frac=0.05, seed=42,
                       data_root=str(ROOT / "data/anytop_planet_zoo_clean_L2"),
                       num_frames=64, max_joints=144, caption_emb_cache=None)
    idxs = np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int)
    all_pct, main_pct, root_pct = [], [], []
    per_obj = {}
    for i in idxs:
        it = ds[int(i)]
        J = int(it["num_joints"]); T = int(it["num_frames"])
        ax = np.asarray(it["anytop_x"], np.float32)
        mean = np.asarray(it["anytop_mean"], np.float32); std = np.asarray(it["anytop_std"], np.float32)
        raw = np.transpose(ax, (2, 0, 1))[:T, :J, :] * (std[:J][None] + _STD_FLOOR) + mean[:J][None]
        parents = np.asarray([int(p) for p in it["parent_indices"][:J]], int)
        offsets = np.asarray(it["rest_offsets"], np.float32)[:J]
        jn = [str(x) for x in (it.get("joint_names") or [""] * J)][:J]
        ric = _recover_world_positions(raw)                       # [T,J,3]
        fk = recover_from_bvh_rot_np(raw, parents, offsets)       # [T,J,3]
        err = np.linalg.norm(fk - ric, axis=-1)                   # [T,J]
        bbox = ric.reshape(-1, 3)
        diag = float(np.linalg.norm(bbox.max(0) - bbox.min(0))) or 1e-9
        helper = np.array([any(t in n.lower() for t in HELPER_TOKENS) for n in jn]) if len(jn) == J else np.zeros(J, bool)
        main = ~helper; main[0] = False
        all_pct.extend((err / diag * 100).reshape(-1).tolist())
        if main.any():
            main_pct.extend((err[:, main] / diag * 100).reshape(-1).tolist())
        root_pct.extend((err[:, 0] / diag * 100).tolist())
        o = it["object_type"]
        per_obj.setdefault(o, []).append(float(np.percentile(err[:, main] / diag * 100, 95)) if main.any() else 0.0)
    return {"split": split, "n_clips": len(idxs),
            "all_joints_pct_bbox": stats(all_pct),
            "main_nonhelper_pct_bbox": stats(main_pct),
            "root_pct_bbox": stats(root_pct),
            "worst_objects_p95": sorted(((o, max(v)) for o, v in per_obj.items()), key=lambda x: -x[1])[:8]}


def main():
    out = {"train": scan("train", 60), "val": scan("val", 40)}
    p95 = out["val"]["main_nonhelper_pct_bbox"]["p95"]
    verdict = ("GOOD (p95<=2%, FK target consistent)" if p95 <= 2 else
               "USABLE (2-5%, report mismatch floor)" if p95 <= 5 else
               "HIGH (>5%, lower w_fk first or fix samples)")
    out["verdict_val_main_p95"] = {"value": p95, "verdict": verdict}
    (ROOT / "runs").mkdir(exist_ok=True)
    op = ROOT / "runs" / "_preflight_ric_vs_fk_L2.json"
    op.write_text(json.dumps(out, indent=2))
    print(json.dumps({"train_main_p95": out["train"]["main_nonhelper_pct_bbox"]["p95"],
                      "val_main_p95": p95, "verdict": verdict,
                      "train_main_mean": out["train"]["main_nonhelper_pct_bbox"]["mean"],
                      "val_main_mean": out["val"]["main_nonhelper_pct_bbox"]["mean"]}, indent=2), flush=True)
    print(f"WROTE {op}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
