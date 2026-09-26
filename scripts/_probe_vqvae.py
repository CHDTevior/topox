import torch
ck = torch.load("runs/vqvae_v4b272neutral_C96_J144_d512_Q4_n8192_b16g64_300ep_curric50to60_seed42/best_model.pt",
                map_location="cpu", weights_only=False)
for cand in ["train_args","args","config"]:
    if cand in ck:
        a = ck[cand]; d = vars(a) if hasattr(a,"__dict__") else a
        print("USING", cand)
        for k in ["max_joints","temporal_stride","anytop_root","data_root","num_frames",
                  "code_dim","d_model","num_quantizers","codebook_size","max_frames"]:
            if k in d: print(f"  {k} = {d[k]}")
        break
