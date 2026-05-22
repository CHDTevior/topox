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


class AnyTopGraphAttentionBlock(nn.Module):
    """Graph transformer block with AnyTop-style Graphormer edge biases.

    Replaces GraphAttentionBlock's scalar `Linear(1, n_heads)` geodesic/adjacency
    bias with AnyTop's richer embedding-gather bias (model/motion_transformer.py
    GraphMultiHeadAttention): each hop-distance bucket and each edge-type has its
    own learned d_model embedding, projected against the query AND key and added
    to the attention score. Two relation tensors drive it:
      - graph_dist      [B, J, J] integer hop distance, clamped to [0, 5]
                        (6 buckets — AnyTop's max_path_len=5 convention)
      - joint_relations [B, J, J] edge type in [0, 5]
                        (0 self / 1 parent / 2 child / 3 sibling / 4 none / 5 EE)
    Same outer structure (q/k/v/o proj, norms, FFN, residuals, masking) as
    GraphAttentionBlock so it is a drop-in for the encoder's graph layers.
    """

    N_HOP_BUCKETS = 6   # AnyTop max_path_len(5) + 1
    N_EDGE_TYPES = 6    # self/parent/child/sibling/no_relation/end_effector

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
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

        # Graphormer-style relative embeddings (query + key side).
        self.topo_q_emb = nn.Embedding(self.N_HOP_BUCKETS, d_model)
        self.topo_k_emb = nn.Embedding(self.N_HOP_BUCKETS, d_model)
        self.edge_q_emb = nn.Embedding(self.N_EDGE_TYPES, d_model)
        self.edge_k_emb = nn.Embedding(self.N_EDGE_TYPES, d_model)

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

    def _rel_bias(self, q, k, emb_q, emb_k, rel_idx):
        """Graphormer bias: gather per-(i,j) query+key bias for a relation.

        Args:
            q, k: [B, H, J, d_head]
            emb_q, emb_k: nn.Embedding(n_types, d_model)
            rel_idx: [B, J, J] long, values in [0, n_types)
        Returns: [B, H, J, J] additive bias.
        """
        B, H, J, _ = q.shape
        n_types = emb_q.num_embeddings
        # emb.weight [n_types, d_model] -> [1, H, n_types, d_head]
        wq = emb_q.weight.view(1, n_types, H, self.d_head).transpose(1, 2)
        wk = emb_k.weight.view(1, n_types, H, self.d_head).transpose(1, 2)
        # score against every relation type: [B, H, J, n_types]
        q_rel = torch.matmul(q, wq.transpose(2, 3))
        k_rel = torch.matmul(k, wk.transpose(2, 3))
        # gather the bucket for each (i, j) pair: index [B, H, J, J]
        idx = rel_idx.unsqueeze(1).expand(-1, H, -1, -1)
        q_gathered = torch.gather(q_rel, 3, idx)   # [B, H, J, J]
        k_gathered = torch.gather(k_rel, 3, idx)
        return q_gathered + k_gathered

    def forward(
        self,
        x: torch.Tensor,                 # [B, J, D]
        graph_dist: torch.Tensor,        # [B, J, J] hop distance
        joint_relations: torch.Tensor,   # [B, J, J] edge type
        joint_mask: torch.Tensor,        # [B, J] bool
    ) -> torch.Tensor:
        B, J, D = x.shape
        residual = x
        x = self.norm1(x)

        q = self.q_proj(x).view(B, J, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(B, J, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(B, J, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        # Defensive clamp: padded regions are 0 (valid index); AnyTop edge maps
        # may carry a 7th 'ts_token' type (idx 6) absent from our data — clamp
        # so a stray value never indexes past the embedding table.
        gd = graph_dist.clamp(0, self.N_HOP_BUCKETS - 1).long()
        jr = joint_relations.clamp(0, self.N_EDGE_TYPES - 1).long()
        # AnyTop scales the SUMMED score (qk + topo_bias + edge_bias) by
        # 1/sqrt(d_head) — motion_transformer.py:92-94 — so the bias terms are
        # scaled too. Add raw, then scale once.
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores + self._rel_bias(q, k, self.topo_q_emb, self.topo_k_emb, gd)
        scores = scores + self._rel_bias(q, k, self.edge_q_emb, self.edge_k_emb, jr)
        scores = scores / math.sqrt(self.d_head)

        # Mask invalid joints (large finite negative — avoid all-(-inf) NaN)
        mask = joint_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, J]
        scores = scores.masked_fill(~mask, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = attn.nan_to_num(0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B, H, J, d_head]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, J, D)
        out = self.o_proj(out)
        x = residual + self.dropout(out)

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
        motion_feat_dim: int = 6,      # fk6: local_pos(3)+vel(3); anytop13: 13ch
        clip_embed_dim: int = 768,     # CLIP text embedding dim
        clip_proj_dim: int = 128,      # projected CLIP dim
        temporal_kernel: int = 9,
        dropout: float = 0.1,
        motion_mode: str = "fk6_shared",
        attn_mode: str = "scalar",
    ):
        super().__init__()
        self.d_model = d_model
        # motion_mode controls how per-frame motion features are projected:
        #   "fk6_shared"     — one shared MLP for all joints (legacy 6ch path)
        #   "anytop13_split" — separate Linears for root vs non-root joints
        #     (AnyTop InputProcess: root channels 0:3 are RIFKE root-state,
        #      non-root 0:3 are relative position — different semantics).
        if motion_mode not in ("fk6_shared", "anytop13_split"):
            raise ValueError(
                f"motion_mode must be 'fk6_shared' or 'anytop13_split', "
                f"got {motion_mode!r}"
            )
        self.motion_mode = motion_mode
        # attn_mode controls the graph-layer attention bias:
        #   "scalar"     — GraphAttentionBlock (Linear(1,n_heads) geo/adj bias)
        #   "graphormer" — AnyTopGraphAttentionBlock (per-edge-type embedding
        #                  bias, gathered by graph_dist + joint_relations)
        if attn_mode not in ("scalar", "graphormer"):
            raise ValueError(
                f"attn_mode must be 'scalar' or 'graphormer', got {attn_mode!r}"
            )
        self.attn_mode = attn_mode

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
        if attn_mode == "graphormer":
            self.graph_layers = nn.ModuleList([
                AnyTopGraphAttentionBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_graph_layers)
            ])
        else:
            self.graph_layers = nn.ModuleList([
                GraphAttentionBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_graph_layers)
            ])

        # Motion feature projection (dynamic per-frame features)
        if motion_mode == "fk6_shared":
            self.motion_proj = nn.Sequential(
                nn.Linear(motion_feat_dim, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, d_model),
            )
        else:  # anytop13_split — AnyTop-style root/non-root separate Linears
            self.motion_proj_root = nn.Linear(motion_feat_dim, d_model)
            self.motion_proj_nonroot = nn.Linear(motion_feat_dim, d_model)

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
        graph_dist: Optional[torch.Tensor] = None,        # [B, J, J] (graphormer)
        joint_relations: Optional[torch.Tensor] = None,   # [B, J, J] (graphormer)
    ) -> torch.Tensor:
        """
        Encode static skeleton graph into per-joint embeddings.
        This is computed ONCE per skeleton and reused across all frames.

        In `attn_mode="graphormer"` the graph layers consume `graph_dist` +
        `joint_relations` (AnyTop edge biases) instead of `adjacency` +
        `geodesic_dist`; both must then be provided.

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
        if self.attn_mode == "graphormer":
            if graph_dist is None or joint_relations is None:
                raise ValueError(
                    "SkeletonEncoder(attn_mode=graphormer) requires graph_dist "
                    "and joint_relations"
                )
            for layer in self.graph_layers:
                s = layer(s, graph_dist, joint_relations, joint_mask)
        else:
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
        graph_dist: Optional[torch.Tensor] = None,        # [B, J, J] (graphormer)
        joint_relations: Optional[torch.Tensor] = None,   # [B, J, J] (graphormer)
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
            graph_dist=graph_dist, joint_relations=joint_relations,
        )  # [B, J, D]

        # 2. Dynamic motion feature projection
        if self.motion_mode == "anytop13_split":
            # Root (joint 0) and non-root joints use separate Linears — their
            # channels 0:3 carry different semantics (RIFKE root-state vs
            # relative position). Joint 0 is guaranteed root by the dataset's
            # FK re-ordering. J >= 9 for all AnyTop skeletons so the [1:] slice
            # is always non-empty.
            m_root = self.motion_proj_root(motion_features[:, :, 0:1, :])    # [B,T,1,D]
            m_nonroot = self.motion_proj_nonroot(motion_features[:, :, 1:, :])  # [B,T,J-1,D]
            m_tj = torch.cat([m_root, m_nonroot], dim=2)                     # [B,T,J,D]
        else:
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
