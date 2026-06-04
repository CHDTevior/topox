"""noKslot_clean / src.models.treeik_decoder — TreeIK series + FK helpers
surgical-extracted from source repo's scripts/topofk_decoder.py:16-277.

Contains (verbatim from source where noted):

  Module-level FK helpers (source lines 23-80, verbatim):
    - rot6d_to_matrix       (source line 23-29)
    - validate_fk_tree      (source line 32-42)
    - fk_one                (source line 45-62)
    - fk_persample          (source line 65-80)

  Module-level loss helper (source line 265-277, verbatim):
    - rot_geodesic_loss

  Classes (verbatim from source where noted):
    - TopoFKDecoder         (source line 83-114) — wraps a MotionDecoder base
                              with per-joint MLP rot head + hard FK. Not used
                              by no_k_slot path's `topofk_treeik` head, but kept
                              because it shares base/fk_persample/rest-pose-init
                              idiom with TopoFKTreeIKDecoder and is the minimum
                              FK-head reference; future no_k_slot variants may
                              swap rot heads (MLP vs TreeIK) using the same
                              base. Drop only if Step 4 truly never instantiates
                              it (re-review at Step 12).
    - RestFiLM              (source line 124-136) — rest-offset FiLM modulator
    - TreeGraphAttention    (source line 139-176) — spatial graph attention
    - TreeIKBlock           (source line 179-200) — block: FiLM → attn → FFN
    - TopoFKTreeIKDecoder   (source line 203-262) — main no_k_slot head:
                              TreeIK rot head + hard FK + IK rot supervision.

Intentionally NOT brought over from source scripts/topofk_decoder.py
(K-slot research path — user 2026-05-19 Hold-not-sunset):
  - TopoCondSynth                (source line 306-437)  — C1 topology generator
  - TopoFKTopoCondDecoder        (source line 438-547)  — C1 wrapper
  - JointMemoryCrossAttention    (source line 548-609)  — §4-mem memory pool
  - TreeIKMemBlock               (source line 610-639)  — §4-mem block
  - TopoFKTreeIKMemDecoder       (source line 640-end)  — §4-mem head

All five are K-slot / paired-gate / cross-species memory architectures from
the K-slot research path that user placed on Hold (not sunset). The no_k_slot
clean baseline uses TopoFKTreeIKDecoder only (the ② TreeIK head without memory
pool) — that is what the noKslot diagnostic at runs/baseline_noKslot_ep399/
last_model.pt was trained with.

TopoFKTreeIKDecoder.__init__(base_decoder, d_model, n_layers=3, n_heads=8,
                             d_ff=None, dropout=0.1, max_hop=16.0)
where base_decoder is a src.models.motion_decoder.MotionDecoder instance
(see source train_paired_gate.py:1274 — TopoFKTreeIKDecoder(model.decoder, ...)).
"""
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F


def rot6d_to_matrix(r6):
    a1, a2 = r6[..., :3], r6[..., 3:]
    b1 = a1 / (a1.norm(dim=-1, keepdim=True) + 1e-8)
    a2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = a2 / (a2.norm(dim=-1, keepdim=True) + 1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def validate_fk_tree(parents):
    """codex 019e2cdb C1/C2 fail-fast guardrail (verbatim). FK silently
    depends on a single root at index 0 + strict parent-before-child order;
    codex verified the live cs_sparse2full_* data passes (roots_bad=0,
    bad_parent_order=0 over all 72 skeletons) — this asserts it stays true."""
    roots = [i for i, p in enumerate(parents) if p < 0]
    if roots != [0]:
        raise RuntimeError(f"TopoFK requires single root at index 0, got {roots}")
    bad = [(j, p) for j, p in enumerate(parents) if p >= j and p >= 0]
    if bad:
        raise RuntimeError(f"TopoFK requires parent-before-child order, bad={bad[:8]}")


def fk_one(local_rot6d_b, root_local_b, parents_b, rest_b):
    """One sample. local_rot6d_b [T,J,6]; root_local_b [T,3];
    parents_b list[int] len J; rest_b [J,3] tensor.
    Returns root-relative local positions [T,J,3] (FK; bone lengths exact)."""
    T, J, _ = local_rot6d_b.shape
    R = rot6d_to_matrix(local_rot6d_b)              # [T,J,3,3]
    gpos = [None] * J
    grot = [None] * J
    for j in range(J):
        p = parents_b[j]
        off = rest_b[j].view(1, 3, 1)
        if p < 0:
            grot[j] = R[:, j]
            gpos[j] = root_local_b                  # [T,3]
        else:
            grot[j] = grot[p] @ R[:, j]
            gpos[j] = gpos[p] + (grot[p] @ off).squeeze(-1)
    return torch.stack(gpos, dim=1)                 # [T,J,3]


def fk_persample(local_rot6d, root_local, parents_list, rest_tensor, joint_mask):
    """Batched variable-skeleton FK (codex: per-sample tree, loop over B).
    local_rot6d [B,T,Jpad,6]; root_local [B,T,3];
    parents_list: list length B, each list[int] of that sample's real joints;
    rest_tensor [B,Jpad,3]; joint_mask [B,Jpad]. Returns [B,T,Jpad,3] with
    padded joints zeroed."""
    B, T, Jpad, _ = local_rot6d.shape
    out = local_rot6d.new_zeros(B, T, Jpad, 3)
    for b in range(B):
        par = parents_list[b]
        Jb = min(len(par), Jpad)
        validate_fk_tree(par[:Jb])          # codex C1/C2: before fk_one
        pos_b = fk_one(local_rot6d[b, :, :Jb], root_local[b],
                       par[:Jb], rest_tensor[b, :Jb])
        out[b, :, :Jb] = pos_b
    return out * joint_mask[:, None, :, None].float()


class TopoFKDecoder(nn.Module):
    """Wraps an existing MotionDecoder (used with return_features=True) and a
    small per-joint head -> local 6D rotation + root; positions via per-sample
    FK on target rest tree. velocities = fps * finite-diff(positions)."""

    def __init__(self, base_decoder, d_model):
        super().__init__()
        self.base = base_decoder                    # MotionDecoder (frozen-able)
        self.rot = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 6))
        self.root = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 3))
        # init rot head -> identity 6D, root head -> 0 (start at rest pose,
        # learn motion as it helps; no free Cartesian collapse possible).
        nn.init.zeros_(self.rot[-1].weight)
        with torch.no_grad():
            self.rot[-1].bias[:] = torch.tensor([1, 0, 0, 0, 1, 0],
                                                dtype=torch.float32)
        nn.init.zeros_(self.root[-1].weight); nn.init.zeros_(self.root[-1].bias)

    def forward(self, slot_features, s_j, asg, joint_mask, frame_mask,
                parents_list, rest_tensor, fps, root_idx=0):
        feats = self.base(slot_features, s_j, asg, joint_mask, frame_mask,
                          return_features=True)      # [B,T,Jpad,D]
        r6 = self.rot(feats)                          # [B,T,Jpad,6]
        root_local = self.root(feats[:, :, root_idx])  # [B,T,3]
        pos = fk_persample(r6, root_local, parents_list, rest_tensor, joint_mask)
        vel = torch.zeros_like(pos)
        if pos.shape[1] > 1:
            vel[:, 1:] = (pos[:, 1:] - pos[:, :-1]) * fps
            vel[:, 0] = vel[:, 1]
        return torch.cat([pos, vel], dim=-1)          # [B,T,Jpad,6]


# ===================================================================
# ② TopoFK-TreeIK — tree-aware rotation decoder (codex TreeIK-design
# 20260516, thread 019e2cdb). Spatial-only (no temporal block: base
# decoder already did temporal; no memory cross-attn: SECOND per codex
# synth). Replaces ONLY the per-joint MLP rot head; root/fk_persample/
# hard-FK/rest-pose-identity-init UNCHANGED (13 anchors preserved).
# ===================================================================
class RestFiLM(nn.Module):
    """codex spec (§1 MoCapAnything FiLMCondition idiom, independent
    reimpl): rest-offset FiLM with a gentle 0.1 gate. cond=rest_embed
    [B,J,D] broadcast over T; x [B,T,J,D]."""

    def __init__(self, d_model):
        super().__init__()
        self.to_scale_shift = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 2 * d_model))

    def forward(self, x, rest_embed):
        scale, shift = self.to_scale_shift(rest_embed).chunk(2, dim=-1)
        return x * (1.0 + 0.1 * scale.unsqueeze(1)) + shift.unsqueeze(1)


class TreeGraphAttention(nn.Module):
    """codex spec: spatial-only per-frame multi-head attention over the
    joint axis using the PROJECT's geodesic_bias+adjacency_bias additive
    idiom (skeleton_encoder-consistent, anchor-compatible). Full valid
    tree, NOT ancestor-only (codex: distal rotation needs parent/sibling/
    root/whole-body phase; hard ancestor-only = a new bottleneck)."""

    def __init__(self, d_model, n_heads, max_hop):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        self.max_hop = max_hop
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        # project idiom (src/models/skeleton_encoder.py): scalar→per-head
        self.geodesic_bias = nn.Linear(1, n_heads, bias=False)
        self.adjacency_bias = nn.Linear(1, n_heads, bias=False)

    def forward(self, x, adjacency, geodesic_dist, joint_mask):
        B, T, J, D = x.shape
        q = self.q(x).view(B, T, J, self.h, self.dh).permute(0, 1, 3, 2, 4)
        k = self.k(x).view(B, T, J, self.h, self.dh).permute(0, 1, 3, 2, 4)
        v = self.v(x).view(B, T, J, self.h, self.dh).permute(0, 1, 3, 2, 4)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.dh ** 0.5)
        geo = geodesic_dist.clamp(0.0, self.max_hop).unsqueeze(-1)
        gb = self.geodesic_bias(geo).permute(0, 3, 1, 2)          # [B,H,J,J]
        ab = self.adjacency_bias(adjacency.unsqueeze(-1)).permute(0, 3, 1, 2)
        scores = scores + (gb + ab).unsqueeze(1)                  # +T bcast
        km = joint_mask.bool()[:, None, None, None, :]            # mask KEY
        scores = scores.masked_fill(~km, -1e9)
        attn = F.softmax(scores.float(), dim=-1).to(scores.dtype)  # bf16-safe: fp32 softmax (fp32 path no-op)
        out = torch.matmul(attn, v)                               # [B,T,H,J,dh]
        out = out.permute(0, 1, 3, 2, 4).reshape(B, T, J, D)
        out = self.o(out)
        return out * joint_mask[:, None, :, None].to(out.dtype)   # zero QUERY


class TreeIKBlock(nn.Module):
    """codex spec block order: RestFiLM → +TreeGraphAttention(LN) →
    +FFN(LN) → mask by joint+frame. NO temporal block."""

    def __init__(self, d_model, n_heads, d_ff, dropout, max_hop):
        super().__init__()
        self.film = RestFiLM(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = TreeGraphAttention(d_model, n_heads, max_hop)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout))

    def forward(self, x, rest_embed, adjacency, geodesic_dist,
                joint_mask, frame_mask):
        x = self.film(x, rest_embed)
        x = x + self.attn(self.ln1(x), adjacency, geodesic_dist, joint_mask)
        x = x + self.ffn(self.ln2(x))
        x = x * joint_mask[:, None, :, None].to(x.dtype)
        x = x * frame_mask[:, :, None, None].to(x.dtype)
        return x


class TopoFKTreeIKDecoder(nn.Module):
    """codex TreeIK-design 20260516 (thread 019e2cdb) — ② tree-aware
    rotation decoder. Replaces TopoFKDecoder's per-joint MLP rot head with
    a stack of spatial tree-aware blocks (rest-FiLM + project geodesic/
    adjacency graph attention + FFN). root head + fk_persample + hard FK +
    rest-pose identity-6D init UNCHANGED (13 anchors preserved; only the
    rot head upgrades). Memory cross-attn OUT of ② (SECOND per codex
    synth). IK-derived rotation supervision via train_paired_gate
    --w_rot_ik (the ① genuinely-PASS foundation)."""

    def __init__(self, base_decoder, d_model, n_layers=3, n_heads=8,
                 d_ff=None, dropout=0.1, max_hop=16.0):
        super().__init__()
        self.base = base_decoder
        d_ff = d_ff or 4 * d_model              # codex Recommended defaults
        self.rest_proj = nn.Linear(3, d_model)
        self.blocks = nn.ModuleList([
            TreeIKBlock(d_model, n_heads, d_ff, dropout, max_hop)
            for _ in range(n_layers)])
        self.rot_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 6))
        self.root = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 3))
        # codex: preserve rest-pose identity-6D output at step 0 regardless
        # of random TreeIKBlock weights; root zero-init (same as TopoFK).
        nn.init.zeros_(self.rot_head[-1].weight)
        with torch.no_grad():
            self.rot_head[-1].bias[:] = torch.tensor(
                [1, 0, 0, 0, 1, 0], dtype=torch.float32)
        nn.init.zeros_(self.root[-1].weight)
        nn.init.zeros_(self.root[-1].bias)

    def forward(self, slot_features, s_j, asg, joint_mask, frame_mask,
                parents_list, rest_tensor, fps, adjacency, geodesic_dist,
                root_idx=0, return_rot=False):
        feats = self.base(slot_features, s_j, asg, joint_mask, frame_mask,
                          return_features=True)         # [B,T,J,D]
        rest_embed = self.rest_proj(rest_tensor)        # [B,J,D]
        x = feats
        for blk in self.blocks:
            x = blk(x, rest_embed, adjacency, geodesic_dist,
                    joint_mask, frame_mask)
        r6 = self.rot_head(x)                            # [B,T,J,6]
        root_local = self.root(feats[:, :, root_idx])    # [B,T,3] unchanged
        pos = fk_persample(r6, root_local, parents_list, rest_tensor,
                           joint_mask)
        vel = torch.zeros_like(pos)
        if pos.shape[1] > 1:
            vel[:, 1:] = (pos[:, 1:] - pos[:, :-1]) * fps
            vel[:, 0] = vel[:, 1]
        out = torch.cat([pos, vel], dim=-1)              # [B,T,Jpad,6]
        return (out, r6) if return_rot else out

    def new_parameters(self):
        # codex C3 lesson: EXCLUDE self.base (== model.decoder) to avoid
        # optimizer param duplication. Only the ② head modules.
        return chain(self.rest_proj.parameters(),
                     self.blocks.parameters(),
                     self.rot_head.parameters(),
                     self.root.parameters())


def rot_geodesic_loss(pred_r6, ik_r6, rot_mask):
    """codex TreeIK-design spec: geodesic-on-SO(3) (NOT 6D MSE). pred_r6/
    ik_r6 [...,6]; rot_mask [...] (frame valid & joint valid & non-leaf
    only — leaf rotations are position-unobservable & IK leaf targets
    arbitrary; root included iff it has children)."""
    Rp = rot6d_to_matrix(pred_r6)
    Rt = rot6d_to_matrix(ik_r6)
    rel = Rp.transpose(-1, -2) @ Rt
    tr = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos = ((tr - 1.0) / 2.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    angle2 = torch.acos(cos) ** 2
    m = rot_mask.to(angle2.dtype)
    return (angle2 * m).sum() / m.sum().clamp(min=1.0)
