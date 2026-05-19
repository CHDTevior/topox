"""
Skeleton Encoder (Component A) for TopoSlots.

Encodes arbitrary skeleton graphs into per-joint embeddings via:
1. Static joint features (rest offset, bone length, depth, degree, side tag)
2. CLIP-projected joint name embeddings (semantic anchoring)
3. Graph transformer with topology-aware edge biases
4. Temporal fusion with dynamic per-frame motion features

Output: h_{t,j} ∈ R^{d_model} for each joint j at each frame t.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GraphAttentionBlock(nn.Module):
    """
    Graph transformer block with topology-aware edge biases.

    Attention is biased by:
    - Parent/child relationship (edge type)
    - Geodesic distance between joints
    - Rest-pose direction vector
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        # Edge bias projections
        self.geodesic_bias = nn.Linear(1, n_heads, bias=False)  # scalar distance → per-head bias
        self.adjacency_bias = nn.Linear(1, n_heads, bias=False)  # binary adjacency → per-head bias

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,            # [B, J, D]
        adjacency: torch.Tensor,     # [B, J, J]
        geodesic_dist: torch.Tensor, # [B, J, J]
        joint_mask: torch.Tensor,    # [B, J] bool
    ) -> torch.Tensor:
        B, J, D = x.shape

        # Self-attention with edge biases
        residual = x
        x = self.norm1(x)

        q = self.q_proj(x).view(B, J, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(B, J, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(B, J, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        # [B, H, J, J]

        # Add topology biases
        geo_bias = self.geodesic_bias(geodesic_dist.unsqueeze(-1))  # [B, J, J, H]
        adj_bias = self.adjacency_bias(adjacency.unsqueeze(-1))     # [B, J, J, H]
        topo_bias = (geo_bias + adj_bias).permute(0, 3, 1, 2)      # [B, H, J, J]
        scores = scores + topo_bias

        # Mask invalid joints (use large finite negative to avoid NaN from softmax on all -inf)
        mask = joint_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, J]
        scores = scores.masked_fill(~mask, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = attn.nan_to_num(0.0)  # safety: all-masked rows → 0 attention
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B, H, J, d_head]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, J, D)
        out = self.o_proj(out)
        x = residual + self.dropout(out)

        # FFN
        x = x + self.ff(self.norm2(x))

        return x


class TemporalBlock(nn.Module):
    """Temporal convolution + attention for fusing motion dynamics."""

    def __init__(self, d_model: int, kernel_size: int = 9, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding=padding, groups=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size, padding=padding, groups=1),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] or [B*J, T, D]
            mask: [B, T] or [B*J, T] bool
        """
        residual = x
        x = self.norm(x)
        # Conv expects [B, D, T]
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()
        return residual + x


class SkeletonEncoder(nn.Module):
    """
    Full skeleton encoder: static joint features + CLIP names + graph transformer + temporal fusion.

    Produces per-joint, per-frame embeddings h_{t,j} for slot assignment.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        n_graph_layers: int = 6,
        n_temporal_layers: int = 4,
        joint_feat_dim: int = 9,       # from SkeletonGraph.get_joint_features()
        motion_feat_dim: int = 6,      # local_pos(3) + velocity(3)
        clip_embed_dim: int = 768,     # CLIP text embedding dim
        clip_proj_dim: int = 128,      # projected CLIP dim
        temporal_kernel: int = 9,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Static joint feature projection
        self.joint_feat_proj = nn.Sequential(
            nn.Linear(joint_feat_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )

        # CLIP joint name embedding projection (optional, used when available)
        self.clip_proj = nn.Sequential(
            nn.Linear(clip_embed_dim, clip_proj_dim),
            nn.GELU(),
            nn.Linear(clip_proj_dim, d_model),
        )
        self.use_clip = False  # Set True when CLIP embeddings are available

        # Learnable canonical name embedding (lightweight alternative to CLIP)
        # Maps a hash of the canonical name string to a d_model embedding
        self.name_embed_dim = 128
        self.name_vocab_size = 1024  # hash buckets
        self.name_embedding = nn.Embedding(self.name_vocab_size, d_model)
        self.use_name_embed = False  # Set True to enable

        # Graph transformer layers (operate on static skeleton)
        self.graph_layers = nn.ModuleList([
            GraphAttentionBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_graph_layers)
        ])

        # Motion feature projection (dynamic per-frame features)
        self.motion_proj = nn.Sequential(
            nn.Linear(motion_feat_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )

        # Fusion: combine static joint embedding + dynamic motion features
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Temporal layers (operate along time for each joint)
        self.temporal_layers = nn.ModuleList([
            TemporalBlock(d_model, temporal_kernel, dropout)
            for _ in range(n_temporal_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

    def encode_skeleton(
        self,
        skeleton_features: torch.Tensor,   # [B, J, 9]
        adjacency: torch.Tensor,           # [B, J, J]
        geodesic_dist: torch.Tensor,       # [B, J, J]
        joint_mask: torch.Tensor,          # [B, J]
        clip_embeddings: Optional[torch.Tensor] = None,  # [B, J, clip_dim]
        name_hashes: Optional[torch.Tensor] = None,      # [B, J] int64
    ) -> torch.Tensor:
        """
        Encode static skeleton graph into per-joint embeddings.
        This is computed ONCE per skeleton and reused across all frames.

        Returns: s_j ∈ R^{d_model} for each joint j. Shape: [B, J, D]
        """
        # Project joint features
        s = self.joint_feat_proj(skeleton_features)  # [B, J, D]

        # Add CLIP embeddings if available
        if clip_embeddings is not None and self.use_clip:
            clip_feat = self.clip_proj(clip_embeddings)
            s = s + clip_feat

        # Add canonical name embeddings if enabled
        if self.use_name_embed and name_hashes is not None:
            name_feat = self.name_embedding(name_hashes)  # [B, J, D]
            s = s + name_feat

        # Graph transformer: topology-aware message passing
        for layer in self.graph_layers:
            s = layer(s, adjacency, geodesic_dist, joint_mask)

        # Zero out padded joints
        s = s * joint_mask.unsqueeze(-1).float()

        return s  # [B, J, D]

    def forward(
        self,
        motion_features: torch.Tensor,    # [B, T, J, 6]
        skeleton_features: torch.Tensor,  # [B, J, 9]
        adjacency: torch.Tensor,          # [B, J, J]
        geodesic_dist: torch.Tensor,      # [B, J, J]
        joint_mask: torch.Tensor,         # [B, J]
        frame_mask: torch.Tensor,         # [B, T]
        clip_embeddings: Optional[torch.Tensor] = None,
        name_hashes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full forward: static skeleton encoding + dynamic motion fusion + temporal processing.

        Returns: h_{t,j} ∈ R^{d_model}. Shape: [B, T, J, D]
        """
        B, T, J, _ = motion_features.shape

        # 1. Static skeleton encoding (computed once, broadcast across time)
        s_j = self.encode_skeleton(
            skeleton_features, adjacency, geodesic_dist, joint_mask,
            clip_embeddings, name_hashes,
        )  # [B, J, D]

        # 2. Dynamic motion feature projection
        m_tj = self.motion_proj(motion_features)  # [B, T, J, D]

        # 3. Fuse static + dynamic
        s_j_expanded = s_j.unsqueeze(1).expand(-1, T, -1, -1)  # [B, T, J, D]
        h_tj = self.fusion(torch.cat([s_j_expanded, m_tj], dim=-1))  # [B, T, J, D]

        # 4. Temporal processing (per-joint across time)
        # Reshape: [B*J, T, D]
        h_flat = h_tj.permute(0, 2, 1, 3).reshape(B * J, T, self.d_model)
        # Expand frame mask for all joints
        frame_mask_expanded = frame_mask.unsqueeze(1).expand(-1, J, -1).reshape(B * J, T)

        for layer in self.temporal_layers:
            h_flat = layer(h_flat, frame_mask_expanded)

        # Reshape back: [B, T, J, D]
        h_tj = h_flat.reshape(B, J, T, self.d_model).permute(0, 2, 1, 3)
        h_tj = self.final_norm(h_tj)

        # Mask invalid joints
        h_tj = h_tj * joint_mask.unsqueeze(1).unsqueeze(-1).float()

        return h_tj
