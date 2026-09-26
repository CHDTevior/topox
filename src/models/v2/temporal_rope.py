"""Sinusoidal temporal RoPE for the frame axis, ported from UniMate's RopeND
(outside_docs/UniMate/unimate/models/denoiser/rope.py `_generate_cos_sin` and its `auto_base` rule).

Why it replaces the learned table. `InContextMotionDiT.t_pos` is an ABSOLUTE learned position table of
4096 rows, of which training only ever touches the frames inside its window (1 demo + 240 target here):
every later row keeps its initialisation, so a sequence longer than the trained window is addressed by
vectors that no gradient has seen. A rotary encoding has no table: the attention logit depends on the
DIFFERENCE of two frame indices, which is defined at any length.

The frequency base. UniMate's rule is base = 8L/pi rounded up to the next hundred, where L is the model's
maximum sequence length; it places the SLOWEST channel pair at about a sixteenth of a revolution across L
frames (measured 0.055 at head_dim 64). The faster pairs do wrap -- at head_dim 64 and base 700, 18 of the
32 pairs complete a revolution within 240 frames of offset, and the slowest completes one at about 3584 --
which is the ordinary state of any RoPE: the band of frequencies is what resolves several distance scales
at once, and only the slow end has to stay unwrapped for long offsets to remain distinguishable. At
L = 241 (1 + 240) the rule gives 700; the 10000 of language models would leave the slowest pair at half a
percent of a revolution over the window and spend most of the head's channels on distances this data does
not contain. No base is established as better here (codex trope r1 P3); 700 is UniMate's rule at our
window, and a different horizon is a different arm that needs its own training run and calibration.
"""
import math

import torch


def unimate_auto_base(max_len: int) -> float:
    """UniMate's RopeND auto_base: (int(8 * L / pi) // 100 + 1) * 100."""
    if max_len < 1:
        raise ValueError(f"max_len must be positive, got {max_len}")
    return float((int(8 * max_len / math.pi) // 100 + 1) * 100)


def sinusoidal_cos_sin(n_pos: int, head_dim: int, base: float, *, device=None, dtype=torch.float32):
    """(cos, sin), each [n_pos, head_dim], for positions 0..n_pos-1.

    inv_freq[i] = base^(-2i/head_dim) for i in [0, head_dim/2), the frequency table concatenated with
    itself so it pairs channel j with channel j + head_dim/2 -- the `rotate_half` convention of
    src/models/v2/spec_rope.py, and UniMate's.
    """
    if head_dim % 2:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    if not (base > 1.0):
        raise ValueError(f"base must be > 1, got {base}")
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(n_pos, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)                     # [n_pos, head_dim/2]
    freqs = torch.cat([freqs, freqs], dim=-1)            # [n_pos, head_dim]
    return freqs.cos().to(dtype), freqs.sin().to(dtype)
