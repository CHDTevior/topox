"""LoRA correctness tests (CPU, small dims):
  1. injection touches exactly the intended module paths and nothing else;
  2. at init (B = 0) the adapted model equals the backbone bit-for-bit;
  3. after random adapters, merged_state_dict() loaded into a PLAIN InContextMotionDiT reproduces
     the adapted model's output (this is what every downstream consumer relies on);
  4. freeze_non_lora leaves only lora_A/lora_B trainable.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.v2.dit_motion import InContextMotionDiT                       # noqa: E402
from src.models.v2.lora import (LoRALinear, inject_lora, freeze_non_lora,   # noqa: E402
                                merged_state_dict, lora_state_dict, TARGET_GROUPS)

torch.manual_seed(0)
KW = dict(in_ch=17, dim=64, depth=2, n_heads=4, d_text=4096, d_joint_sem=4096,
          use_struct_feats=True, use_dir_bias=True, qk_norm=True)
B, T, J = 2, 9, 7


def batch():
    x = torch.randn(B, T, J, 17)
    return dict(x=x, t=torch.rand(B), text=torch.randn(B, 4096), joint_sem=torch.randn(B, J, 4096),
                struct_feats=torch.randn(B, J, 8), updown=torch.randint(0, 3, (B, J, J, 2)),
                joint_bias=torch.zeros(B, J, J), frame_valid=torch.ones(B, T, dtype=torch.bool),
                joint_valid=torch.ones(B, J, dtype=torch.bool))


def fwd(m, b):
    is_target = torch.ones(B, T, dtype=torch.bool, device=b["x"].device); is_target[:, 0] = False   # 1 demo frame
    return m(b["x"], b["t"], is_target=is_target, text=b["text"], joint_sem=b["joint_sem"], joint_bias=b["joint_bias"],
             frame_valid=b["frame_valid"], joint_valid=b["joint_valid"],
             struct_feats=b["struct_feats"], updown=b["updown"])


def main():
    base = InContextMotionDiT(**KW).eval()
    ref_sd = {k: v.clone() for k, v in base.state_dict().items()}
    b = batch()
    with torch.no_grad():
        y0 = fwd(base, b)

    paths = inject_lora(base, ["attn", "ffn", "cond"], r=8, alpha=8.0)
    exp_per_block = {"t_attn.qkv", "t_attn.proj", "s_attn.qkv", "s_attn.proj", "mlp.0", "mlp.2", "ada.1"}
    got_blocks = {p.split(".", 2)[2] for p in paths if p.startswith("blocks.")}
    assert got_blocks == exp_per_block, got_blocks
    top = {p for p in paths if not p.startswith("blocks.")}
    assert top == {"joint_sem", "text_mlp.1", "text_mlp.3", "struct_mlp.0", "struct_mlp.2", "ada_out.1"}, top
    assert len(paths) == 2 * len(exp_per_block) + len(top)
    for name, mod in base.named_modules():                      # untouched Linears stay plain
        if isinstance(mod, torch.nn.Linear) and name in ("x_in", "out", "t_mlp.0", "t_mlp.2"):
            pass
        assert not (isinstance(mod, LoRALinear) and name not in paths)
    print(f"[1] injected {len(paths)} layers, paths as specified")

    with torch.no_grad():
        y1 = fwd(base, b)
    assert torch.equal(y0, y1), "B=0 init must be the identity adapter"
    print("[2] init equals backbone bit-for-bit")

    n_tr, n_tot = freeze_non_lora(base)
    trainable = {n for n, p in base.named_parameters() if p.requires_grad}
    assert all(n.endswith((".lora_A", ".lora_B")) for n in trainable) and n_tr > 0
    print(f"[4] trainable {n_tr} / {n_tot} ({100 * n_tr / n_tot:.1f}%), all lora_A/B")

    with torch.no_grad():                                        # perturb adapters
        for n, p in base.named_parameters():
            if n.endswith((".lora_A", ".lora_B")):
                p.add_(torch.randn_like(p) * 0.3)
        y2 = fwd(base, b)
    assert not torch.allclose(y2, y0), "perturbed adapters must change the output"
    merged = merged_state_dict(base)
    assert set(merged) == set(ref_sd), (set(merged) ^ set(ref_sd))
    plain = InContextMotionDiT(**KW).eval()
    plain.load_state_dict(merged, strict=True)
    with torch.no_grad():
        y3 = fwd(plain, b)
    err = (y3 - y2).abs().max().item()
    assert err < 1e-4, f"merged model deviates {err}"
    ls = lora_state_dict(base)
    assert len(ls) == 2 * len(paths)
    # weights that were NOT adapted must be untouched by the merge
    for k in ("x_in.weight", "out.weight", "t_mlp.0.weight"):
        assert torch.equal(merged[k], ref_sd[k])
    print(f"[3] merged plain model reproduces adapted output (max err {err:.2e}); non-adapted weights untouched")

    # 5. the training path is bf16 autocast on CUDA: the two adapter GEMMs and the one merged GEMM
    #    are not bit-identical there, so bound the gap the way the trainer/validator will see it
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        bc = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        ad, pl = base.to(dev).eval(), plain.to(dev).eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ya, yp = fwd(ad, bc).float(), fwd(pl, bc).float()
        rel = ((ya - yp).abs().max() / ya.abs().max().clamp_min(1e-6)).item()
        assert rel < 2e-2, f"bf16 adapted-vs-merged relative gap {rel}"
        print(f"[5] CUDA bf16 autocast: adapted vs merged max rel gap {rel:.2e} (fp32 path {err:.2e})")
    else:
        print("[5] skipped (no CUDA): bf16 adapted-vs-merged gap not measured")
    print("LORA-TESTS-PASS")


if __name__ == "__main__":
    main()
