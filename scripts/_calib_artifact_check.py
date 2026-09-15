#!/usr/bin/env python3
"""Validate a gamma-calibration artifact against the arm it is meant to certify -- the same environment the calibration runner
sourced (the arm's config): the complete augmentation protocol record must equal AugConfig(...).protocol() built from the
AUG_* variables (the trainer compares the whole record; codex 2026-09-15 unimate r5 P2: a partial comparison accepted an
artifact measured under an earlier augmentation rule), the model record must be the arm's (DIM / DEPTH / HEADS / QK_NORM /
STRUCT_FEATS / DIR_BIAS / ARM_GEO_BIAS / ARM_FREEZE_ZERO_JOINT_SEM), the mechanism check must have run VERIFY_STEPS steps,
the solve must be the pinned GAMMA_SOLVE with non-uniform gammas, and the producer must be the allowlisted script named
by EXPECT_CODE_SCRIPT. usage: python scripts/_calib_artifact_check.py <artifact.json>  (exit 1 = refused)"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.ktjd17_augment import AugConfig   # noqa: E402


def main(path: str) -> int:
    art = json.load(open(path)); pr = art["protocol"]; ver = pr.get("verify") or {}
    e = os.environ
    cfg = AugConfig(p=float(e.get("AUG_P", "0")), drop_max_frac=float(e.get("AUG_DROP_MAX_FRAC", "0")),
                    drop_mode=e.get("AUG_DROP_MODE", "any"), rest_deg=float(e.get("AUG_REST_DEG", "0")),
                    sem_noise=float(e.get("AUG_SEM_NOISE", "0")), sem_drop_p=float(e.get("AUG_SEM_DROP_P", "0")),
                    stats_logsd=float(e.get("AUG_STATS_LOGSD", "0")), stats_shift=float(e.get("AUG_STATS_SHIFT", "0")),
                    bone_scale=float(e.get("AUG_BONE_SCALE", "0")), pool_frac=float(e.get("AUG_POOL_FRAC", "0")),
                    add_p=float(e.get("AUG_ADD_P", "0")), mode=e.get("AUG_MODE", "joint"))
    want_aug = cfg.protocol()
    want_model = {"dim": int(e["DIM"]), "depth": int(e["DEPTH"]), "heads": int(e["HEADS"]), "qk_norm": e["QK_NORM"] == "1",
                  "struct_feats": e["STRUCT_FEATS"] == "1", "dir_bias": e["DIR_BIAS"] == "1",
                  "geo_bias": e["ARM_GEO_BIAS"] == "1", "freeze_zero_joint_sem": e["ARM_FREEZE_ZERO_JOINT_SEM"] == "1"}
    want_solve, want_steps, want_script = e["GAMMA_SOLVE"], int(e["VERIFY_STEPS"]), e["EXPECT_CODE_SCRIPT"]
    bad = []
    if pr.get("augmentation") != want_aug: bad.append(f"augmentation={pr.get('augmentation')!r} != {want_aug!r}")
    if ver.get("arm_model") != want_model: bad.append(f"verify.arm_model={ver.get('arm_model')!r} != {want_model!r}")
    if ver.get("steps") != want_steps or ver.get("arm_grad_ckpt") is not True:
        bad.append(f"verify.steps/arm_grad_ckpt={ver.get('steps')!r}/{ver.get('arm_grad_ckpt')!r}")
    if pr.get("gamma_solve") != want_solve: bad.append(f"gamma_solve={pr.get('gamma_solve')!r} != {want_solve!r}")
    if want_solve != "uniform" and all(abs(float(g) - 1.0) < 1e-12 for g in art["gammas"].values()): bad.append("all gammas are 1.0")
    if art["hashes"].get("code_script") != want_script: bad.append(f"code_script={art['hashes'].get('code_script')!r} != {want_script!r}")
    if bad:
        print("[calib] REFUSED artifact:", "; ".join(bad)); return 1
    print("[calib] artifact checked:", want_solve, "gammas", {k: round(float(v), 3) for k, v in art["gammas"].items()},
          "| augmentation", (want_aug or {}).get("version"), "| model", want_model)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
