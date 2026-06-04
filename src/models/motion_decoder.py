"""
Motion Decoder (Component E) for TopoSlots.

Reconstructs per-joint motion features from slot representations.
Architecture: slot unpool → graph cross-attention → temporal refinement → output projection.

For Step 1 (continuous AE), takes continuous slot features.
Later (Step 2+), takes quantized slot codes from SlotQuantizer.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .encoder import AnyTopGraphAttentionBlock


class SlotToJointCrossAttention(nn.Module):
    """Cross-attention: target joint queries attend to slot features."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Learnable coefficient for assignment bias (starts at 1.0, can learn to soften)
        self.assign_bias_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        joint_queries: torch.Tensor,   # [B, J, D]
        slot_features: torch.Tensor,   # [B, K, D]
        assignment: torch.Tensor,      # [B, J, K] — used as attention bias
    ) -> torch.Tensor:
        B, J, D = joint_queries.shape
        K = slot_features.shape[1]

        q = self.q_proj(self.norm_q(joint_queries))
        kv_in = self.norm_kv(slot_features)
        k = self.k_proj(kv_in)
        v = self.v_proj(kv_in)

        q = q.view(B, J, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        k = k.view(B, K, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        v = v.view(B, K, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        # Attention scores [B, H, J, K]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Bias with assignment matrix (joints attend more to their assigned slots)
        # clamp BEFORE log to avoid log(0) = -inf → NaN in backward
        # scale is learnable — model can soften the assignment constraint
        assign_bias = assignment.unsqueeze(1).clamp(min=1e-8).log()  # [B, 1, J, K]
        scores = scores + self.assign_bias_scale * assign_bias

        attn = F.softmax(scores.float(), dim=-1).to(scores.dtype)  # bf16-safe: fp32 softmax (fp32 path no-op)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B, H, J, d_head]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, J, D)
        return joint_queries + self.o_proj(out)


class TemporalRefineBlock(nn.Module):
    """1D temporal convolution for smoothing decoded motion."""

    def __init__(self, d_model: int, kernel_size: int = 9, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size, padding=padding),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B*J, T, D]"""
        residual = x
        x = self.norm(x).permute(0, 2, 1)
        x = self.conv(x).permute(0, 2, 1)
        return residual + x


class MotionDecoder(nn.Module):
    """
    Decode slot features → per-joint motion features.

    Pipeline:
    1. Unpool slots to joints via assignment matrix (initial estimate)
    2. Cross-attention refinement (joints attend to all slots)
    3. Temporal refinement (smooth per-joint trajectories)
    4. Output projection → [T, J, motion_dim]
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_cross_layers: int = 3,
        n_temporal_layers: int = 2,
        motion_feat_dim: int = 6,    # output: local_pos(3) + velocity(3)
        temporal_kernel: int = 9,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.motion_feat_dim = motion_feat_dim

        # Cross-attention layers: joints query slots
        self.cross_layers = nn.ModuleList([
            SlotToJointCrossAttention(d_model, n_heads, dropout)
            for _ in range(n_cross_layers)
        ])

        # FFN after each cross-attention
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(n_cross_layers)
        ])

        # Temporal refinement
        self.temporal_layers = nn.ModuleList([
            TemporalRefineBlock(d_model, temporal_kernel, dropout)
            for _ in range(n_temporal_layers)
        ])

        # Output projection
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, motion_feat_dim)

    def forward(
        self,
        slot_features: torch.Tensor,       # [B, T, K, D]
        skeleton_embeddings: torch.Tensor,  # [B, J, D] — target skeleton static embeddings
        assignment: torch.Tensor,           # [B, J, K] — target skeleton assignment
        joint_mask: torch.Tensor,           # [B, J]
        frame_mask: torch.Tensor,           # [B, T]
        return_features: bool = False,      # codex 019e2cdb G1: FK-head feature tap
    ) -> torch.Tensor:
        """
        Decode slot features to per-joint motion.

        Returns: [B, T, J, motion_feat_dim], or — when return_features=True —
        the masked pre-output-projection per-joint features [B, T, J, D]
        (used by the FK/rotation head; position-head behavior unchanged).
        """
        B, T, K, D = slot_features.shape
        J = skeleton_embeddings.shape[1]

        # 1. Initial unpool: [B, T, J, D]
        unpool_features = torch.einsum('bjk,btkd->btjd', assignment, slot_features)

        # 2. Initialize joint queries from skeleton embeddings (topology-conditioned)
        # skeleton_embeddings [B, J, D] carry target topology info
        joint_features = unpool_features + skeleton_embeddings.unsqueeze(1).expand(-1, T, -1, -1)

        # 3. Cross-attention refinement (per frame): joints query all slots
        for cross_attn, ffn in zip(self.cross_layers, self.ffn_layers):
            jf = joint_features.reshape(B * T, J, D)
            sf = slot_features.reshape(B * T, K, D)
            assign_expand = assignment.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, J, K)

            jf = cross_attn(jf, sf, assign_expand)
            jf = jf + ffn(jf)
            joint_features = jf.reshape(B, T, J, D)

        # 3. Temporal refinement (per joint across time)
        jf = joint_features.permute(0, 2, 1, 3).reshape(B * J, T, D)  # [B*J, T, D]
        for temp_layer in self.temporal_layers:
            jf = temp_layer(jf)
        joint_features = jf.reshape(B, J, T, D).permute(0, 2, 1, 3)  # [B, T, J, D]

        # 4. Output projection
        features = self.output_norm(joint_features)
        if return_features:
            features = features * joint_mask[:, None, :, None].float()
            features = features * frame_mask[:, :, None, None].float()
            return features
        output = self.output_proj(features)  # [B, T, J, motion_dim]

        # Mask invalid joints and frames
        output = output * joint_mask[:, None, :, None].float()
        output = output * frame_mask[:, :, None, None].float()

        return output


class TemporalSelfAttention(nn.Module):
    """Multi-head self-attention over the temporal axis (frames, per joint).

    AnyTop's decoder coordinates a joint across the whole clip with full-sequence
    temporal attention (not the conv used by TemporalRefineBlock). Pre-norm;
    padded frames are key-masked. Used by GraphTemporalDecoderLayer.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        """x: [N, T, D]  frame_mask: [N, T]  ->  [N, T, D]   (N = B*J)."""
        N, T, D = x.shape
        residual = x
        x = self.norm(x)

        q = self.q_proj(x).view(N, T, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(N, T, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(N, T, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        # Key-mask padded frames (large finite negative — avoid all-(-inf) NaN).
        mask = frame_mask.bool().unsqueeze(1).unsqueeze(2)   # [N, 1, 1, T]
        scores = scores.masked_fill(~mask, -1e9)

        attn = F.softmax(scores.float(), dim=-1).to(scores.dtype)  # bf16-safe: fp32 softmax (fp32 path no-op)
        attn = attn.nan_to_num(0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)                          # [N, H, T, d_head]
        out = out.permute(0, 2, 1, 3).contiguous().view(N, T, D)
        return residual + self.dropout(self.o_proj(out))


class GraphTemporalDecoderLayer(nn.Module):
    """One AnyTop-style decoder refine layer: spatial graph-attention over joints
    (per frame) then temporal self-attention over frames (per joint).

    Used only by GraphMotionVAE decoder_mode='graph_temporal' — runs on the fine
    [B,T,J,D] features the coarse_xattn path produces, adding the joint↔joint +
    long-range temporal coordination MotionDecoder lacks.

    The reused AnyTopGraphAttentionBlock and TemporalSelfAttention only KEY-mask
    (a padded joint/frame as a QUERY still attends valid tokens → non-zero, dirty
    output). So this layer explicitly re-masks padded joints after the spatial
    sub-block and padded joints+frames after the temporal sub-block — each
    layer's output is then clean by construction and the next layer never
    inherits dirt.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.spatial = AnyTopGraphAttentionBlock(d_model, n_heads, d_ff, dropout)
        self.temporal = TemporalSelfAttention(d_model, n_heads, dropout)

    def forward(
        self,
        x: torch.Tensor,                 # [B, T, J, D]
        graph_dist: torch.Tensor,        # [B, J, J]
        joint_relations: torch.Tensor,   # [B, J, J]
        joint_mask: torch.Tensor,        # [B, J]
        frame_mask: torch.Tensor,        # [B, T]
    ) -> torch.Tensor:
        B, T, J, D = x.shape
        jm = joint_mask[:, None, :, None].to(x.dtype)   # [B, 1, J, 1]
        fm = frame_mask[:, :, None, None].to(x.dtype)   # [B, T, 1, 1]

        # --- spatial: graph attention over joints, per frame ([B*T, J, D]) ---
        xs = x.reshape(B * T, J, D)
        gd = graph_dist.unsqueeze(1).expand(B, T, J, J).reshape(B * T, J, J)
        jr = joint_relations.unsqueeze(1).expand(B, T, J, J).reshape(B * T, J, J)
        jm_e = joint_mask.unsqueeze(1).expand(B, T, J).reshape(B * T, J)
        xs = self.spatial(xs, gd, jr, jm_e)
        x = xs.reshape(B, T, J, D) * jm                 # re-mask padded joints

        # --- temporal: self-attention over frames, per joint ([B*J, T, D]) ---
        xt = x.permute(0, 2, 1, 3).reshape(B * J, T, D)
        fm_e = frame_mask.unsqueeze(1).expand(B, J, T).reshape(B * J, T)
        xt = self.temporal(xt, fm_e)
        x = xt.reshape(B, J, T, D).permute(0, 2, 1, 3)
        return x * jm * fm                              # re-mask padded joints + frames
