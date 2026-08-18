"""Graph-VQVAE (coarse-slot structured RVQ) tokenizer package.

A SEPARATE VQ tokenizer pipeline for AniMo4D AnyTop L5, built per
handoff/20260608_graph_vqvae_l5_pipeline_plan.md +
handoff/20260608_0600_graph_vqvae_review_verdict_and_forks.md. It reuses the
Gaussian Graph-VAE's SkeletonEncoder / SlotNorm / EdgeSegmentPool / graph
attention / loss terms by IMPORT but does NOT modify or subclass them — the VAE
and diffusion pipelines keep working unchanged.
"""

from .quantizer import MaskedResidualVQ
from .masked_motion_decoder import MaskedMotionDecoder
from .graph_vq_tokenizer import GraphVQTokenizer, semantic_config_from_ckpt
from .losses import compute_vq_loss_13ch
from .utils import root_drift_jitter_qa

__all__ = [
    "MaskedResidualVQ",
    "MaskedMotionDecoder",
    "GraphVQTokenizer", "semantic_config_from_ckpt",
    "compute_vq_loss_13ch",
    "root_drift_jitter_qa",
]
