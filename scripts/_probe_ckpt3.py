import torch
ck = torch.load("runs/codeflow_graph_pscf_v4b272neutral_n8192_b16g64_lr8e5_4xh200_seed42/last_model.pt",
                map_location="cpu", weights_only=False)
a = ck["args"]
d = vars(a) if hasattr(a, "__dict__") else a
for k in ["anytop_root","data_root","gen_eval_data_root","max_joints","num_frames","stride",
          "temporal_stride","max_T_lat","code_dim","hidden_size","frozen_vqvae_ckpt","evaluator_ckpt"]:
    print(f"{k} = {d.get(k, '<MISSING>')}")
print("epoch =", ck.get("epoch"), "best_val =", ck.get("best_val"), "val_flow =", ck.get("val_flow"))
