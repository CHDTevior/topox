"""Unit tests for src/models/graph_salad/vae.py — M1.3 step 1."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.graph_salad.batch import GraphMotionBatch
from src.models.graph_salad.vae import GraphMotionVAE


def _adj_line(N: int) -> torch.Tensor:
    A = torch.zeros(N, N)
    for j in range(1, N):
        A[j, j - 1] = 1.0
        A[j - 1, j] = 1.0
    return A


def _geo_line(N: int) -> torch.Tensor:
    return torch.tensor([[abs(i - j) for j in range(N)] for i in range(N)],
                        dtype=torch.float32)


def _make_batch(B: int = 1, T: int = 8, J: int = 11, D: int = 64):
    """Build a minimal GraphMotionBatch with a line skeleton (uniform across batch)."""
    nj = [J] * B
    nf = [T] * B
    joint_mask = torch.zeros(B, J, dtype=torch.bool)
    frame_mask = torch.zeros(B, T, dtype=torch.bool)
    for i in range(B):
        joint_mask[i, :nj[i]] = True
        frame_mask[i, :nf[i]] = True
    adj = torch.zeros(B, J, J)
    for i in range(B):
        adj[i, :nj[i], :nj[i]] = _adj_line(nj[i])
    geo = torch.zeros(B, J, J)
    for i in range(B):
        geo[i, :nj[i], :nj[i]] = _geo_line(nj[i])
    line_parents = [[-1] + list(range(J - 1)) for _ in range(B)]

    d = {
        "motion_features": torch.randn(B, T, J, 6),
        "skeleton_features": torch.randn(B, J, 9),
        "joint_mask": joint_mask,
        "frame_mask": frame_mask,
        "adjacency": adj,
        "geodesic_dist": geo,
        "name_hashes": torch.zeros(B, J, dtype=torch.long),
        "root_position": torch.randn(B, T, 3),
        "root_velocity": torch.randn(B, T, 3),
        "local_rotations_6d": torch.randn(B, T, J, 6),
        "foot_contact": torch.zeros(B, T, 4),
        "bone_lengths": torch.ones(B, T, J),
        "rest_offsets": torch.randn(B, J, 3),
        "num_joints": torch.tensor(nj, dtype=torch.long),
        "num_frames": torch.tensor(nf, dtype=torch.long),
        "fps": torch.full((B,), 20.0),
        "has_rotations": torch.ones(B, dtype=torch.bool),
        "parent_indices": line_parents,
        "joint_names": [[f"j{j}" for j in range(J)] for _ in range(B)],
        "canonical_names": [[f"c{j}" for j in range(J)] for _ in range(B)],
        "bone_lengths_rest": [[0.1] * J for _ in range(B)],
        "text": [f"sample {i}" for i in range(B)],
        "skeleton_id": [f"skel_{i}" for i in range(B)],
        "motion_id": [f"mot_{i}" for i in range(B)],
    }
    return GraphMotionBatch.from_collate_dict(d)


class GraphMotionVAETests(unittest.TestCase):

    def test_dynamic_forward(self):
        # Small d_model for speed
        D = 32
        vae = GraphMotionVAE(
            pool_type="dynamic",
            d_model=D, n_heads=4, d_ff=64,
            n_graph_layers=2, n_enc_temporal_layers=1,
            n_cross_layers=1, n_dec_temporal_layers=1,
            n_treeik_layers=1,
            max_coarse=16, local_radius=4,
            temporal_stride=4,
        )
        batch = _make_batch(B=1, T=8, J=11, D=D)
        out = vae(batch)
        # Shape contract
        self.assertEqual(out["pred_pos"].shape, (1, 8, 11, 3))
        self.assertEqual(out["pred_vel"].shape, (1, 8, 11, 3))
        # z latent: [B, T/4, C, D] = [1, 2, ≤16, 32]
        self.assertEqual(out["z"].shape[:2], (1, 2))
        self.assertEqual(out["z"].shape[3], D)
        # mu/logvar same shape as z
        self.assertEqual(out["mu"].shape, out["z"].shape)
        self.assertEqual(out["logvar"].shape, out["z"].shape)
        # Finite
        self.assertTrue(torch.isfinite(out["pred_pos"]).all())

    def test_deterministic_forward(self):
        D = 32
        vae = GraphMotionVAE(
            pool_type="deterministic",
            d_model=D, n_heads=4, d_ff=64,
            n_graph_layers=2, n_enc_temporal_layers=1,
            n_cross_layers=1, n_dec_temporal_layers=1,
            n_treeik_layers=1,
            max_coarse=16, local_radius=4,
            temporal_stride=4,
        )
        batch = _make_batch(B=1, T=8, J=11, D=D)
        out = vae(batch)
        self.assertEqual(out["pred_pos"].shape, (1, 8, 11, 3))

    def test_nopool_forward(self):
        D = 32
        vae = GraphMotionVAE(
            pool_type="none",
            d_model=D, n_heads=4, d_ff=64,
            n_graph_layers=2, n_enc_temporal_layers=1,
            n_cross_layers=1, n_dec_temporal_layers=1,
            n_treeik_layers=1,
            temporal_stride=4,
        )
        batch = _make_batch(B=1, T=8, J=11, D=D)
        out = vae(batch)
        self.assertEqual(out["pred_pos"].shape, (1, 8, 11, 3))
        # In no-pool, C == J (no skeletal compression)
        self.assertEqual(out["z"].shape, (1, 2, 11, D))

    def test_invalid_pool_type_raises(self):
        with self.assertRaisesRegex(ValueError, "pool_type must be"):
            GraphMotionVAE(pool_type="bogus")

    def test_invalid_temporal_stride_raises(self):
        with self.assertRaisesRegex(TypeError, "temporal_stride must be strict int"):
            GraphMotionVAE(pool_type="none", temporal_stride=4.0)

    def test_invalid_dropout_raises(self):
        with self.assertRaisesRegex(ValueError, "dropout must be in"):
            GraphMotionVAE(pool_type="none", dropout=1.0)
        with self.assertRaisesRegex(ValueError, "dropout must be in"):
            GraphMotionVAE(pool_type="none", dropout=-0.1)

    def test_d_model_not_divisible_by_n_heads_raises(self):
        with self.assertRaisesRegex(ValueError, "must be divisible by n_heads"):
            GraphMotionVAE(pool_type="none", d_model=20, n_heads=8)

    def test_n_heads_zero_raises(self):
        with self.assertRaisesRegex(ValueError, "n_heads must be > 0"):
            GraphMotionVAE(pool_type="none", n_heads=0)

    def test_d_model_zero_raises(self):
        with self.assertRaisesRegex(ValueError, "d_model must be > 0"):
            GraphMotionVAE(pool_type="none", d_model=0)

    def test_dim_hparam_strict_int(self):
        with self.assertRaisesRegex(TypeError, "n_heads must be strict int"):
            GraphMotionVAE(pool_type="none", n_heads=True)
        with self.assertRaisesRegex(TypeError, "d_model must be strict int"):
            GraphMotionVAE(pool_type="none", d_model=64.0)

    def test_padded_frame_output_zero(self):
        # Codex M1.3 R6 regression: padded frames in pred_pos/pred_vel must be 0.
        D = 32
        vae = GraphMotionVAE(
            pool_type="dynamic",
            d_model=D, n_heads=4, d_ff=64,
            n_graph_layers=2, n_enc_temporal_layers=1,
            n_cross_layers=1, n_dec_temporal_layers=1,
            n_treeik_layers=1,
            max_coarse=16, local_radius=4,
            temporal_stride=4,
        )
        # T=8 total, but only 4 valid frames
        batch = _make_batch(B=1, T=8, J=6, D=D)
        # Override num_frames + frame_mask to 4 valid frames
        from src.models.graph_salad.batch import GraphMotionBatch
        # Rebuild batch with B=1, T=8, J=6, nf=[4]
        nj = [6]
        nf = [4]
        joint_mask = torch.zeros(1, 6, dtype=torch.bool)
        frame_mask = torch.zeros(1, 8, dtype=torch.bool)
        joint_mask[0, :6] = True
        frame_mask[0, :4] = True
        adj = torch.zeros(1, 6, 6)
        adj[0, :6, :6] = _adj_line(6)
        geo = torch.zeros(1, 6, 6)
        geo[0, :6, :6] = _geo_line(6)
        d = {
            "motion_features": torch.randn(1, 8, 6, 6),
            "skeleton_features": torch.randn(1, 6, 9),
            "joint_mask": joint_mask, "frame_mask": frame_mask,
            "adjacency": adj, "geodesic_dist": geo,
            "name_hashes": torch.zeros(1, 6, dtype=torch.long),
            "root_position": torch.randn(1, 8, 3),
            "root_velocity": torch.randn(1, 8, 3),
            "local_rotations_6d": torch.randn(1, 8, 6, 6),
            "foot_contact": torch.zeros(1, 8, 4),
            "bone_lengths": torch.ones(1, 8, 6),
            "rest_offsets": torch.randn(1, 6, 3),
            "num_joints": torch.tensor(nj, dtype=torch.long),
            "num_frames": torch.tensor(nf, dtype=torch.long),
            "fps": torch.full((1,), 20.0),
            "has_rotations": torch.ones(1, dtype=torch.bool),
            "parent_indices": [[-1, 0, 1, 2, 3, 4]],
            "joint_names": [[f"j{j}" for j in range(6)]],
            "canonical_names": [[f"c{j}" for j in range(6)]],
            "bone_lengths_rest": [[0.1] * 6],
            "text": ["sample 0"], "skeleton_id": ["s0"], "motion_id": ["m0"],
        }
        batch2 = GraphMotionBatch.from_collate_dict(d)
        out = vae(batch2)
        # Padded frames (indices 4-7) of pred_pos/pred_vel must be exactly 0
        self.assertEqual(out["pred_pos"][0, 4:, :, :].abs().max().item(), 0.0,
                        f"padded frames have non-zero pred_pos: max = {out['pred_pos'][0, 4:].abs().max().item()}")
        self.assertEqual(out["pred_vel"][0, 4:, :, :].abs().max().item(), 0.0)

    def test_even_temporal_kernel_raises(self):
        with self.assertRaisesRegex(ValueError, "temporal_kernel must be odd"):
            GraphMotionVAE(pool_type="none", temporal_kernel=2)

    def test_nopool_T_not_divisible_raises(self):
        D = 32
        vae = GraphMotionVAE(
            pool_type="none",
            d_model=D, n_heads=4, d_ff=64,
            n_graph_layers=2, n_enc_temporal_layers=1,
            n_cross_layers=1, n_dec_temporal_layers=1,
            n_treeik_layers=1,
            temporal_stride=4,
        )
        # T=5 not divisible by stride=4
        batch = _make_batch(B=1, T=5, J=6, D=D)
        with self.assertRaisesRegex(ValueError, "T=5 must be divisible by"):
            vae(batch)

    def test_ckpt_compat_real_baseline_load(self):
        """Real ckpt smoke: load runs/baseline_noKslot_ep399/last_model.pt into
        VAE. Codex M1.3 R2 requested replacing synthetic test with real-file
        verification. Skip if ckpt file absent.
        """
        ckpt_path = Path("runs/baseline_noKslot_ep399/last_model.pt")
        if not ckpt_path.exists():
            self.skipTest(f"baseline ckpt {ckpt_path} not present")
        # weights_only=True for security (codex M1.3 R3 fix; verified to work
        # with this ckpt since it contains only torch.Tensor state_dict).
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        # Locate state_dict (training script saves as 'model_state_dict')
        if "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        else:
            sd = ckpt
        # Filter slot_assignment.* keys (PLAN_GAP_REPORT §6 #4: only these are
        # allowed as 'unexpected' for ckpt-compat).
        sd_filtered = {k: v for k, v in sd.items() if not k.startswith("slot_assignment.")}
        sd_dropped = [k for k in sd if k.startswith("slot_assignment.")]
        # Infer dims from real ckpt:
        # encoder.graph_layers.0.geodesic_bias.weight shape = [n_heads, 1]
        # encoder.graph_layers.0.ff.0.weight shape = [d_ff, d_model]
        n_heads_real = sd_filtered["encoder.graph_layers.0.geodesic_bias.weight"].shape[0]
        ff0 = sd_filtered["encoder.graph_layers.0.ff.0.weight"]
        d_ff_real = ff0.shape[0]
        d_model_real = ff0.shape[1]
        vae = GraphMotionVAE(
            pool_type="dynamic",
            d_model=d_model_real, n_heads=n_heads_real, d_ff=d_ff_real,
            n_graph_layers=4, n_enc_temporal_layers=2,
            n_cross_layers=3, n_dec_temporal_layers=2,
            n_treeik_layers=1,  # small treeik for test speed
            max_coarse=32, local_radius=6,
            temporal_stride=4,
            dropout=0.0,
        )
        load_result = vae.load_state_dict(sd_filtered, strict=False)
        # All filtered baseline keys must load (no unexpected)
        self.assertEqual(
            list(load_result.unexpected_keys), [],
            f"baseline keys not consumed by VAE: {load_result.unexpected_keys}",
        )
        # Missing keys are only pool.* / dist.* / treeik_head.* / unpool.*
        for key in load_result.missing_keys:
            if not (key.startswith("pool.") or key.startswith("dist.") or
                   key.startswith("treeik_head.") or key.startswith("unpool.")):
                self.fail(f"unexpected missing key {key} — not pool/dist/treeik_head/unpool")
        # Documented allowed-dropped:
        self.assertTrue(
            all(k.startswith("slot_assignment.") for k in sd_dropped),
            f"unexpected drop list: {sd_dropped}",
        )

    def test_ckpt_compat_baseline_load(self):
        """Verify baseline keys (encoder.* + slot_norm.norm.* + decoder.*)
        load 1:1 into GraphMotionVAE — treeik_head/pool/dist are 'missing'
        (acceptable). Codex M1.3 R1 F1 ckpt-compat gate.
        """
        from src.models.model import Model
        D = 32
        # Build baseline Model with same dims
        baseline = Model(
            d_model=D, n_heads=4, d_ff=64,
            n_graph_layers=2, n_enc_temporal_layers=1,
            n_cross_layers=1, n_dec_temporal_layers=1,
            motion_feat_dim=6, joint_feat_dim=9,
            temporal_kernel=9, dropout=0.0,
        )
        # Build VAE with same encoder/decoder dims
        vae = GraphMotionVAE(
            pool_type="dynamic",
            d_model=D, n_heads=4, d_ff=64,
            n_graph_layers=2, n_enc_temporal_layers=1,
            n_cross_layers=1, n_dec_temporal_layers=1,
            n_treeik_layers=1,
            max_coarse=16, local_radius=4,
            temporal_stride=4,
            dropout=0.0,
        )
        # Load baseline state_dict with strict=False
        baseline_sd = baseline.state_dict()
        load_result = vae.load_state_dict(baseline_sd, strict=False)
        # All baseline keys should be loaded — no unexpected keys from baseline
        self.assertEqual(
            list(load_result.unexpected_keys), [],
            f"baseline keys not loaded by VAE: {load_result.unexpected_keys}",
        )
        # Missing keys: pool.*, dist.*, treeik_head.* (and unpool has no params)
        # All starts with 'pool.', 'dist.', or 'treeik_head.'
        for key in load_result.missing_keys:
            self.assertTrue(
                key.startswith("pool.") or key.startswith("dist.") or
                key.startswith("treeik_head.") or key.startswith("unpool."),
                f"unexpected missing key {key} — expected only pool/dist/treeik_head",
            )

    def test_aux_losses_present(self):
        D = 32
        vae = GraphMotionVAE(
            pool_type="dynamic",
            d_model=D, n_heads=4, d_ff=64,
            n_graph_layers=2, n_enc_temporal_layers=1,
            n_cross_layers=1, n_dec_temporal_layers=1,
            n_treeik_layers=1,
            max_coarse=16, local_radius=4,
            temporal_stride=4,
        )
        batch = _make_batch(B=1, T=8, J=11, D=D)
        out = vae(batch)
        # pool_aux_outputs is a list of one dict (single-level pool)
        self.assertEqual(len(out["pool_aux_outputs"]), 1)
        aux = out["pool_aux_outputs"][0]
        for key in ("mincut", "mincut_cut", "mincut_ortho", "locality", "entropy"):
            self.assertIn(key, aux)


if __name__ == "__main__":
    unittest.main(verbosity=2)
