import torch, sys
ck = torch.load("runs/codeflow_graph_pscf_v4b272neutral_n8192_b16g64_lr8e5_4xh200_seed42/last_model.pt",
                map_location="cpu", weights_only=False)
ta = ck.get("train_args", {}) or {}
for k in ["anytop_root", "max_joints", "max_frames", "model_variant"]:
    print(k, "=", ta.get(k, "?"))
print("ckpt_epoch =", ck.get("epoch", ta.get("epoch", "?")))
