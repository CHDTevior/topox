"""Padded-clean smoke check for GraphTemporalDecoderLayer (decoder_mode=graph_temporal).

Verifies the padded-token re-masking hard point: GraphTemporalDecoderLayer reuses
AnyTopGraphAttentionBlock / TemporalSelfAttention, which only KEY-mask — a padded
joint/frame acting as a QUERY would otherwise produce a non-zero (dirty) output.
The layer must re-mask after each sub-block. Given a clean (zero-on-padded) input,
this asserts the layer output is EXACTLY 0 on padded joints AND padded frames, and
that the valid region is non-trivially non-zero (the layer is not a no-op).

Synthetic tensors only — fast, CPU-safe, no dataset needed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.models.motion_decoder import GraphTemporalDecoderLayer  # noqa: E402


def main() -> int:
    torch.manual_seed(0)
    B, T, J, D = 2, 8, 10, 32
    n_heads, d_ff = 4, 64
    J_valid, T_valid = 6, 5  # joints 6..9 padded, frames 5..7 padded

    layer = GraphTemporalDecoderLayer(D, n_heads, d_ff, dropout=0.0).eval()

    joint_mask = torch.zeros(B, J, dtype=torch.bool)
    joint_mask[:, :J_valid] = True
    frame_mask = torch.zeros(B, T, dtype=torch.bool)
    frame_mask[:, :T_valid] = True
    graph_dist = torch.randint(0, 6, (B, J, J))
    joint_relations = torch.randint(0, 6, (B, J, J))

    # Clean input: zero on padded joints/frames (mirrors MotionDecoder's masked
    # return_features output, which is what feeds the graph_temporal stack).
    x = torch.randn(B, T, J, D)
    x = x * joint_mask[:, None, :, None] * frame_mask[:, :, None, None]

    with torch.no_grad():
        y = layer(x, graph_dist, joint_relations, joint_mask, frame_mask)

    assert y.shape == (B, T, J, D), f"bad output shape {tuple(y.shape)}"
    pad_joint_max = y[:, :, J_valid:, :].abs().max().item()
    pad_frame_max = y[:, T_valid:, :, :].abs().max().item()
    valid_max = y[:, :T_valid, :J_valid, :].abs().max().item()
    print(f"padded-joint max|y|={pad_joint_max:.3e}  "
          f"padded-frame max|y|={pad_frame_max:.3e}  "
          f"valid max|y|={valid_max:.3e}")

    assert pad_joint_max == 0.0, f"padded JOINTS not clean: max|y|={pad_joint_max}"
    assert pad_frame_max == 0.0, f"padded FRAMES not clean: max|y|={pad_frame_max}"
    assert valid_max > 1e-6, f"valid region all-zero — layer is a no-op (max={valid_max})"
    assert torch.isfinite(y).all(), "non-finite values in output"

    print("PADDED-CLEAN SMOKE PASS: graph_temporal layer output is clean on "
          "padded joints + frames, valid region non-trivial.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
