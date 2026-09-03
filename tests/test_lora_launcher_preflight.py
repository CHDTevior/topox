"""The exact minimal launch commands of the two derived-view LoRA runs resolve through the launcher's
PREFLIGHT mode (no srun, no GPU): view profile -> calibration / sidecars / split settings / GROUP_RIGS from the
cut artifact, rank-derived alpha, 5/6 decay, linear-scaled lr (codex 2026-09-03 r4)."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "_launch_lora_species.sh"
VIEWS = {"dragon_all": ROOT / "dataset/ktjd17_truebones_lora_v2_mainbody_dragon_all",
         "flyers": ROOT / "dataset/ktjd17_truebones_lora_v2_mainbody_flyers"}


def preflight(**env):
    e = {k: v for k, v in os.environ.items() if k not in ("SKIN_JOBID", "RIG", "GROUP", "GROUP_RIGS", "OUT", "BATCH",
                                                           "CALIB", "PERCELL", "JOINT_SEM", "KTJD_ROOT", "VIEW_ENV",
                                                           "ALLOW_NO_VAL", "LORA_ALPHA", "LR_DECAY_EPOCHS", "SMOKE")}
    e.update({"PREFLIGHT": "1", **env})
    r = subprocess.run(["bash", str(LAUNCHER)], cwd=ROOT, env=e, capture_output=True, text=True, timeout=600)
    argv = None
    for ln in r.stdout.splitlines():
        if ln.startswith("[lora] PREFLIGHT OK -- would run: "):
            argv = shlex.split(ln.split("would run: ", 1)[1])
    return r, argv


def flag(argv, name):
    return argv[argv.index(name) + 1]


@unittest.skipUnless(all(v.is_dir() for v in VIEWS.values()), "derived LoRA views not on this machine")
class LauncherPreflightTests(unittest.TestCase):
    def test_dragon_all_minimal_command_resolves(self):
        r, argv = preflight(VIEW_ENV="configs/lora_view_dragon_all.env", LORA_R="128", EPOCHS="900")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIsNotNone(argv, r.stdout)
        self.assertIn("ALLOW_NO_VAL=1", r.stdout)
        self.assertEqual(flag(argv, "--ktjd_gamma_calib"), "configs/tb_dragon_all_mainbody_gamma_calibration_v1.json")
        self.assertEqual(flag(argv, "--ktjd_percell_stats"), "data/tb_norm_stats_v2_mainbody.npz")
        self.assertEqual(flag(argv, "--joint_sem"), "data/joint_semantics_llm2vec_ktjd17_v1_mainbody.npz")
        self.assertEqual(flag(argv, "--ktjd_root"), "dataset/ktjd17_truebones_lora_v2_mainbody_dragon_all")
        self.assertEqual(flag(argv, "--exclude_clips"), "configs/tb_lora_Dragon_only_exclusions.json")
        self.assertEqual(flag(argv, "--init_from"), "runs/lora_tb_init_run12_best_snapshot.pt")
        self.assertEqual((flag(argv, "--batch"), flag(argv, "--lr")), ("13", "0.0001625"))
        self.assertEqual((flag(argv, "--lora_r"), flag(argv, "--lora_alpha")), ("128", "128"))
        self.assertEqual((flag(argv, "--epochs"), flag(argv, "--lr_decay_epochs")), ("900", "750"))
        self.assertEqual(flag(argv, "--out"), "runs/lora_tb_Dragon_r128_v2_mainbody_dragon_all")
        self.assertEqual(flag(argv, "--ckpt_every"), "1000000")

    def test_flyers_minimal_command_resolves_with_group_rigs_from_the_cut(self):
        r, argv = preflight(VIEW_ENV="configs/lora_view_flyers.env", LORA_R="128", EPOCHS="200")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIsNotNone(argv, r.stdout)
        cut = json.loads((ROOT / "configs/tb_lora_group_flyers_exclusions.json").read_text())
        self.assertIn("[lora] GROUP_RIGS taken from configs/tb_lora_group_flyers_exclusions.json: "
                      + ",".join(cut["group_rigs"]), r.stdout)
        self.assertEqual(flag(argv, "--ktjd_gamma_calib"), "configs/tb_group_flyers_mainbody_gamma_calibration_v1.json")
        self.assertEqual(flag(argv, "--ktjd_percell_stats"), "data/tb_norm_stats_v2_mainbody.npz")
        self.assertEqual(flag(argv, "--joint_sem"), "data/joint_semantics_llm2vec_ktjd17_v1_mainbody.npz")
        self.assertEqual(flag(argv, "--exclude_clips"), "configs/tb_lora_group_flyers_exclusions.json")
        self.assertEqual((flag(argv, "--batch"), flag(argv, "--lr")), ("6", "7.5e-05"))
        self.assertEqual((flag(argv, "--lora_r"), flag(argv, "--lora_alpha")), ("128", "128"))
        self.assertEqual((flag(argv, "--epochs"), flag(argv, "--lr_decay_epochs")), ("200", "166"))
        self.assertEqual(flag(argv, "--out"), "runs/lora_tb_group_flyers_r128_v2_mainbody_flyers")

    def test_explicit_group_rigs_must_match_the_cut(self):
        r, argv = preflight(VIEW_ENV="configs/lora_view_flyers.env", LORA_R="128", EPOCHS="200",
                            GROUP_RIGS="Bat,Bird,Buzzard,Dragon,Eagle,Giantbee,Parrot")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("!= the cut artifact's group_rigs", r.stdout + r.stderr)
        self.assertIsNone(argv)

    def test_derived_view_without_its_profile_stops_before_srun(self):
        # the footgun codex reproduced: RIG-derived calibration default does not exist for a derived view
        r, argv = preflight(RIG="Dragon", KTJD_ROOT="dataset/ktjd17_truebones_lora_v2_mainbody_dragon_all",
                            LORA_R="128", EPOCHS="900", BATCH="13")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("missing configs/tb_dragon_gamma_calibration_v1.json", r.stdout + r.stderr)
        self.assertIsNone(argv)

    def test_preflight_never_needs_an_allocation_but_a_launch_does(self):
        r, _ = preflight(VIEW_ENV="configs/lora_view_dragon_all.env", LORA_R="128", EPOCHS="900", PREFLIGHT="0")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("SKIN_JOBID", r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
