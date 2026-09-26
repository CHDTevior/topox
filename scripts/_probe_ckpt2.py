import torch
ck = torch.load("runs/codeflow_graph_pscf_v4b272neutral_n8192_b16g64_lr8e5_4xh200_seed42/last_model.pt",
                map_location="cpu", weights_only=False)
print("TOP KEYS:", [k for k in ck.keys() if k != "model"])
for cand in ["train_args", "args", "config", "cfg"]:
    if cand in ck:
        v = ck[cand]
        d = vars(v) if hasattr(v, "__dict__") else (v if isinstance(v, dict) else {})
        print(f"--- {cand} keys:", sorted(list(d.keys()))[:40])
        for k in ["anytop_root","data_root","max_joints","max_frames","model_variant","num_frames","stride","temporal_stride"]:
            if k in d: print(f"   {k} =", d[k])
