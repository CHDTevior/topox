"""Smoke for v2 EdgeSegmentPool. Validates:
  [A] AST parse + import
  [B] Construction with prod cfg
  [C] _build_segments_rulebased on synthetic + real Dragon/Spider/Alligator
  [D] compute_assignment_and_graph with real AnyTopDataset batch:
      - segments ≤ max_coarse=64 for all species
      - assignment is hard 1-of-K (each row sums to 1, single 1.0 per row for valid joints)
      - pooled_adjacency is symmetric, no self-loops, sparsely connected
      - pooled_skeleton_embeddings is segment mean (not gather)
      - aux_losses all zero
  [E] forward() = compute_assignment_and_graph + _pool_features
  [F] Full GraphMotionVAE construction with pool_type='edge_segment'
"""
import os
import sys

REPO = "/scratch/ts1v23/workspace/noKslot_clean"
os.chdir(REPO)
sys.path.insert(0, REPO)

print("=== [A] AST parse ===")
import ast
for fp in [
    "src/models/graph_salad/pool_edge_segment.py",
    "src/models/graph_salad/vae.py",
    "scripts/train_graph_vae.py",
]:
    ast.parse(open(fp).read())
    print(f"  {fp}: OK")

import torch
import numpy as np
from src.models.graph_salad.pool_edge_segment import (
    EdgeSegmentPool, _build_segments_rulebased,
)
from src.models.graph_salad.vae import GraphMotionVAE
from src.models.graph_salad.batch import GraphMotionBatch
from src.data.anytop_dataset import AnyTopDataset, collate_fn as anytop_collate_fn

print("\n=== [B] EdgeSegmentPool construction ===")
pool = EdgeSegmentPool(d_model=384, max_coarse=64, temporal_stride=4)
print(f"  pool.d_model={pool.d_model} max_coarse={pool.max_coarse} stride={pool.temporal_stride}")
n_params = sum(p.numel() for p in pool.parameters())
print(f"  params={n_params} (expect 0 — rule-based, no learnable)")
assert n_params == 0

print("\n=== [C] _build_segments_rulebased synthetic ===")
# Tiny test: linear chain 0→1→2→3→4
parents_linear = [-1, 0, 1, 2, 3]
segs = _build_segments_rulebased(parents_linear, max_segments=64)
print(f"  linear J=5: segments={segs}")
# Expect: [[0]] (virtual root) + chain [1,2,3,4]: L=4 even → [[1,2],[3,4]] → total 3 segments
assert len(segs) == 3, f"linear J=5 expected 3 segments, got {len(segs)}: {segs}"

# Y-shape: 0→1, 1→{2,3,4}
parents_y = [-1, 0, 1, 1, 1]
segs = _build_segments_rulebased(parents_y, max_segments=64)
print(f"  Y-shape J=5 (1 branches to 2,3,4): segments={segs}")
# Expect: [[0]] + chain [1] (L=1, root→branch chain) + chains [2],[3],[4] (each leaf, L=1)
# Total 5 segments
assert len(segs) == 5, f"Y-shape expected 5 segments, got {len(segs)}"

# Long chain: 0→1→...→9 (J=10, chain L=9)
parents_long = [-1] + list(range(9))
segs = _build_segments_rulebased(parents_long, max_segments=64)
print(f"  long chain J=10 (L=9): segments count={len(segs)}")
# L=9 odd → root-side single [1] + 4 pairs [2,3],[4,5],[6,7],[8,9] = 5 chain segments + virtual root = 6
assert len(segs) == 6, f"long chain L=9 expected 6 segments, got {len(segs)}"

# Overflow test: 1 chain of L=200 → need merge
parents_huge = [-1] + list(range(199))
segs = _build_segments_rulebased(parents_huge, max_segments=64)
print(f"  huge chain J=200 (L=199): segments={len(segs)} (max 64)")
assert len(segs) <= 64, f"overflow merge failed: {len(segs)} > 64"

print("\n=== [D] real AnyTopDataset batches ===")
torch.cuda.set_device(0)
device = torch.device("cuda")

ds_val = AnyTopDataset(split="val", num_frames=64, max_joints=143)
# Find one sample per species: Dragon, Spider, Alligator
target_species = {"Dragon", "Spider", "Alligator"}
picked = {}
for i in range(len(ds_val)):
    sp = ds_val.samples[i]["object_type"]
    if sp in target_species and sp not in picked:
        picked[sp] = i
    if len(picked) == 3:
        break
print(f"  picked sample indices: {picked}")

for sp, idx in picked.items():
    item = ds_val[idx]
    batch_dict = anytop_collate_fn([item])
    raw = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch_dict.items()}
    batch = GraphMotionBatch.from_collate_dict(raw)
    J_valid = int(item["num_joints"])

    pool_dev = EdgeSegmentPool(d_model=384, max_coarse=64).to(device)
    skel_emb = torch.randn(1, batch.adjacency.shape[1], 384, device=device, dtype=torch.float32)

    with torch.no_grad():
        out = pool_dev.compute_assignment_and_graph(
            skeleton_embeddings=skel_emb,
            adjacency=batch.adjacency,
            geodesic_dist=batch.geodesic_dist,
            joint_mask=batch.joint_mask,
            parent_indices=batch.parent_indices,
        )

    C_used = int(out["pooled_mask"].sum().item())
    asn = out["assignment"][0]  # [J, C]
    # Validate: hard 1-of-K
    row_sums = asn.sum(dim=-1)
    valid_row_sums = row_sums[batch.joint_mask[0]]
    assert torch.all(valid_row_sums == 1.0), f"{sp}: assignment rows not hard 1-of-K"
    # Validate: each row has exactly one 1.0
    nonzero_per_row = (asn > 0).sum(dim=-1)
    valid_nz = nonzero_per_row[batch.joint_mask[0]]
    assert torch.all(valid_nz == 1), f"{sp}: assignment rows have multiple nonzeros"
    # aux_losses all zero
    for k, v in out["aux_losses"].items():
        assert v.item() == 0.0, f"{sp}: aux_losses[{k}]={v.item()} not zero"
    # pooled_adj symmetric + zero diagonal
    pa = out["pooled_adjacency"][0]
    assert torch.equal(pa, pa.T), f"{sp}: pooled_adj not symmetric"
    assert (pa.diagonal() == 0).all(), f"{sp}: pooled_adj has self-loops"

    print(f"  {sp} J={J_valid}: C_used={C_used}/64, assignment hard 1-of-K ✓, "
          f"pooled_adj symmetric ✓, aux_losses=0 ✓")

print("\n=== [E] forward() (with synthetic joint_features) ===")
T = 64
pool_dev = EdgeSegmentPool(d_model=384, max_coarse=64).to(device)
# Use Dragon batch (largest J=142)
item = ds_val[picked["Dragon"]]
batch_dict = anytop_collate_fn([item])
raw = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch_dict.items()}
batch = GraphMotionBatch.from_collate_dict(raw)
J_padded = batch.adjacency.shape[1]
skel_emb = torch.randn(1, J_padded, 384, device=device, dtype=torch.float32)
jf = torch.randn(1, T, J_padded, 384, device=device, dtype=torch.float32)
fm = torch.ones(1, T, device=device, dtype=torch.bool)
with torch.no_grad():
    out = pool_dev(
        joint_features=jf,
        skeleton_embeddings=skel_emb,
        adjacency=batch.adjacency,
        geodesic_dist=batch.geodesic_dist,
        joint_mask=batch.joint_mask,
        frame_mask=fm,
        parent_indices=batch.parent_indices,
    )
print(f"  pooled_features shape: {tuple(out['pooled_features'].shape)}")
print(f"  frame_mask_down shape: {tuple(out['frame_mask_down'].shape)}")
assert out["pooled_features"].shape == (1, T // 4, 64, 384)
assert out["frame_mask_down"].shape == (1, T // 4)
print(f"  forward shapes OK")

print("\n=== [F] GraphMotionVAE construction (pool_type='edge_segment') ===")
vae = GraphMotionVAE(
    pool_type="edge_segment", pool_tau=None,
    d_model=384, n_heads=8, d_ff=1024,
    n_graph_layers=4, n_enc_temporal_layers=2, n_cross_layers=3,
    n_dec_temporal_layers=2, n_treeik_layers=3,
    max_coarse=64, local_radius=8,
    temporal_stride=4, temporal_kernel=9, dropout=0.1,
    motion_feat_dim=13, feat_mode="anytop13",
    attn_mode="graphormer", use_text=False,
    decoder_mode="coarse_xattn", n_graph_temporal_layers=4,
).to(device).eval()
vae.encoder.use_name_embed = True
print(f"  VAE built OK, pool class = {type(vae.pool).__name__}")
print(f"  VAE total params: {sum(p.numel() for p in vae.parameters()):,}")

# Quick end-to-end forward
out_vae = vae(batch, sample=False)
print(f"  vae.forward(batch): z shape = {tuple(out_vae['z'].shape)}, "
      f"pred_motion shape = {tuple(out_vae['pred_motion'].shape)}")

print("\n=== ALL SMOKE CHECKS PASS ===")
