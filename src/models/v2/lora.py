"""LoRA (Hu et al. 2021) for the in-context motion DiT -- per-species fine-tuning on a frozen
backbone (user 2026-09-02: one LoRA per TrueBones species, rank 64, on attention q/k/v/o, the
FFN and the conditioning pathways).

Design choices that matter downstream:
  * `LoRALinear` wraps an EXISTING nn.Linear (kept as `.base`, frozen) and adds
    B @ A with A ~ Kaiming-uniform, B = 0, scaled by alpha / r -- so at step 0 the model is exactly
    the backbone. The wrapper is a drop-in: callers keep calling the module like the Linear.
  * `merged_state_dict()` folds every adapter into its base weight and returns a PLAIN state_dict
    with the original key names. Checkpoints written this way load into an unmodified
    InContextMotionDiT, so v2_render_incontext / gen-eval / the skinning generator need no LoRA
    knowledge. The adapter-only tensors are saved alongside (`lora_state_dict`) for bookkeeping.
  * Target selection is by module PATH regex, not by class, so the set is explicit and auditable:
      attn : blocks.*.{t_attn,s_attn}.{qkv,proj}
      ffn  : blocks.*.mlp.{0,2}
      cond : blocks.*.ada.1 (AdaLN), joint_sem, text_mlp.{1,3}, struct_mlp.{0,2}, ada_out.1
    (x_in / out / t_mlp / ref_text_mlp / bp_mlp are NOT adapted.)
"""
from __future__ import annotations

import math
import re
from typing import Iterable

import torch
import torch.nn as nn

TARGET_GROUPS = {
    "attn": [r"^blocks\.\d+\.(t_attn|s_attn)\.(qkv|proj)$"],
    "ffn": [r"^blocks\.\d+\.mlp\.(0|2)$"],
    "cond": [r"^blocks\.\d+\.ada\.1$", r"^joint_sem$", r"^text_mlp\.(1|3)$",
             r"^struct_mlp\.(0|2)$", r"^ada_out\.1$"],
}


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.alpha = int(r), float(alpha)
        self.scale = self.alpha / self.r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, dtype=base.weight.dtype,
                                               device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=base.weight.dtype,
                                               device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scale

    def merged_weight(self) -> torch.Tensor:
        return self.base.weight.detach() + (self.lora_B @ self.lora_A).detach() * self.scale


def _matches(name: str, patterns: Iterable[re.Pattern]) -> bool:
    return any(p.search(name) for p in patterns)


def inject_lora(model: nn.Module, groups: Iterable[str], r: int, alpha: float,
                dropout: float = 0.0) -> list[str]:
    """Replace every nn.Linear whose path matches the selected groups with a LoRALinear.
    Returns the sorted list of adapted module paths. Refuses an unknown group or an empty match."""
    pats = []
    for g in groups:
        if g not in TARGET_GROUPS:
            raise ValueError(f"unknown LoRA target group {g!r}; choose from {sorted(TARGET_GROUPS)}")
        pats += [re.compile(p) for p in TARGET_GROUPS[g]]
    adapted = []
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Linear) and _matches(name, pats):
            parent_name, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, attr, LoRALinear(mod, r, alpha, dropout))
            adapted.append(name)
    if not adapted:
        raise RuntimeError(f"LoRA: no nn.Linear matched groups {list(groups)}")
    return sorted(adapted)


def freeze_non_lora(model: nn.Module) -> tuple[int, int]:
    """requires_grad only on lora_A / lora_B. Returns (n_trainable, n_total)."""
    n_tr = n_tot = 0
    for n, p in model.named_parameters():
        is_lora = n.endswith(".lora_A") or n.endswith(".lora_B")
        p.requires_grad_(is_lora)
        n_tot += p.numel()
        n_tr += p.numel() if is_lora else 0
    return n_tr, n_tot


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if k.endswith(".lora_A") or k.endswith(".lora_B")}


@torch.no_grad()
def merged_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Plain state_dict of the equivalent backbone: adapters folded into `.weight`, the `.base.`
    segment removed from the keys, adapter tensors dropped."""
    out = {}
    lora_mods = {n for n, m in model.named_modules() if isinstance(m, LoRALinear)}
    for k, v in model.state_dict().items():
        if k.endswith(".lora_A") or k.endswith(".lora_B"):
            continue
        owner = k.rsplit(".base.", 1)[0] if ".base." in k else None
        if owner in lora_mods:
            leaf = k.rsplit(".", 1)[1]
            if leaf == "weight":
                v = model.get_submodule(owner).merged_weight()
            out[f"{owner}.{leaf}"] = v.detach().cpu()
        else:
            out[k] = v.detach().cpu()
    return out
