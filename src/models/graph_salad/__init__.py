"""graph_salad — graph-aware latent VAE + denoiser stub for multi-topology motion.

M1 milestone scope: Phase 1 (GraphMotionVAE reconstruction across 3 pool variants —
dynamic learned / deterministic-anchor / no-pool) + Phase 2 denoiser interface stub
(signature-only). No denoiser training in M1.

See docs/PLAN_GAP_REPORT.md for design decisions and acceptance gates.
"""

from .attention import GraphAttentionBlock
from .batch import GraphMotionBatch
from .denoiser_stub import GraphSaladDenoiserStub
from .pool_deterministic import DeterministicGraphPool
from .pool_dynamic import DynamicGraphPool
from .unpool import DynamicGraphUnpool
from .losses import (
    aggregate_pool_aux,
    compute_total_loss,
    compute_total_loss_13ch,
    masked_bone_length,
    masked_contact_bce,
    masked_kl_gaussian,
    masked_l1_pos,
    masked_l1_vel,
    masked_vel_consistency,
)
from .vae import GraphMotionVAE
from .graph_utils import (
    floyd_shortest_path,
    build_coarse_adjacency_from_hard_assign,
    find_anchors_rulebased,
    decompose_chains,
    topological_order_with_root_first,
    assert_root_first_parent_order,
)

__all__ = [
    # M1.2 core modules
    "GraphAttentionBlock",
    "DynamicGraphPool",
    "DeterministicGraphPool",
    "DynamicGraphUnpool",
    # M1.2 step 5 losses
    "masked_l1_pos",
    "masked_l1_vel",
    "masked_vel_consistency",
    "masked_kl_gaussian",
    "masked_bone_length",
    "masked_contact_bce",
    "aggregate_pool_aux",
    "compute_total_loss",
    "compute_total_loss_13ch",
    # M1.3 VAE wrapper
    "GraphMotionVAE",
    # M1.1 scaffolding
    "GraphMotionBatch",
    "GraphSaladDenoiserStub",
    # M1.0 graph utilities
    "floyd_shortest_path",
    "build_coarse_adjacency_from_hard_assign",
    "find_anchors_rulebased",
    "decompose_chains",
    "topological_order_with_root_first",
    "assert_root_first_parent_order",
]
