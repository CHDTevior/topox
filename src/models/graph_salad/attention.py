"""GraphAttentionBlock — graph-aware multi-head self-attention.

Used in M1.2+ by:
- No-pool VAE variant: full skeletal self-attention over fine joints
- Phase 2 denoiser: latent self-attention over coarse nodes with pooled
  adjacency / geodesic bias
- Optional pool refinement (post-assignment)

Bias formulation: per-head scalar projection (Linear(1, n_heads)) on
adjacency + geodesic. Matches encoder.py::GraphAttentionBlock formulation
for ckpt compatibility across the codebase. Lit survey (Graphormer NeurIPS
2021, lit_survey §4.2 rec 2) recommended SPD-bucketed bias as a later
upgrade — deferred to a separate ablation milestone if Phase 2 needs it.

Why not reuse encoder.py::GraphAttentionBlock directly:
1. encoder's mask arg is named `joint_mask`; graph_salad needs to call this
   on coarse nodes too where the mask is `coarse_mask`. A second module with
   neutral `node_mask` naming avoids semantic drift.
2. graph_salad/attention.py state_dict keys are namespaced under graph_salad
   instead of mingling with encoder's keys — preserves M1.0 ckpt envelope
   (PLAN_GAP_REPORT.md §3.6: missing=[] + unexpected limited to
   slot_assignment.*).

Note: this module is structurally a near-copy of encoder.py:20-99. We do not
inherit because (a) coupling the two modules through inheritance would propagate
encoder-side refactors into graph_salad unintentionally, and (b) the parameter
names need to be free to evolve independently for the denoiser case.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_utils import floyd_shortest_path


class GraphAttentionBlock(nn.Module):
    """Graph-aware multi-head self-attention with adjacency + geodesic bias.

    Args:
        d_model: feature dim of every node. Must be divisible by n_heads.
        n_heads: number of attention heads.
        d_ff: hidden dim of the post-attention feedforward block.
        dropout: dropout probability in attention + FFN.

    Forward args:
        x:             [B, N, d_model]  — node features (finite)
        adjacency:     [B, N, N]        — binary-or-soft in [0, 1], symmetric,
                                           zero diagonal. All local producers
                                           (GraphMotionBatch, pool_dynamic,
                                           pool_deterministic) emit binary {0,1}
                                           skeleton-or-pooled adjacency; values
                                           outside [0, 1] disproportionately
                                           shift the additive bias (codex round 7).
        geodesic_dist: [B, N, N]        — non-negative finite hop-count distances
                                           or +Inf for unreachable pairs; symmetric
                                           in finite/+Inf pattern AND in finite
                                           values; zero diagonal at valid nodes
        node_mask:     [B, N]           — bool, True = valid node;
                                           at least one True per batch element
        validate_inputs: bool           — when True (default), runs the full
                                           ~14 contract checks. Hot-path callers
                                           (e.g. diffusion denoiser timestep
                                           loop) can pass False to skip; codex
                                           round 7 recommendation.

    Forward returns:
        [B, N, d_model]
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_graph_bias: bool = True,
    ) -> None:
        super().__init__()
        if d_model <= 0 or n_heads <= 0:
            raise ValueError(f"d_model and n_heads must be > 0, got {d_model}, {n_heads}")
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        if d_ff <= 0:
            raise ValueError(f"d_ff must be > 0, got {d_ff}")
        # Dropout contract is [0, 1) per nn.Dropout docs; p=1 zeros all outputs
        # (everything dropped) which corrupts gradient flow silently. Codex M1.2
        # round 1 R12 fix.
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.use_graph_bias = use_graph_bias

        # Q/K/V/O projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        # Edge bias projections (scalar → per-head). Matches encoder.py:41-42.
        # Only the graph-aware variant adds adjacency/geodesic bias to the scores.
        # The no_graph_spatial ablation (use_graph_bias=False) is a plain slot
        # self-attention: it drops these two tiny projections (~2*n_heads params/
        # block, negligible vs d_model² Q/K/V/O+FFN) and skips the bias in _compute,
        # but keeps node_mask + the rest of the block byte-identical (param-aligned).
        if use_graph_bias:
            self.geodesic_bias = nn.Linear(1, n_heads, bias=False)
            self.adjacency_bias = nn.Linear(1, n_heads, bias=False)

        # Norms (pre-norm)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feedforward block
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
        x: torch.Tensor,
        adjacency: torch.Tensor,
        geodesic_dist: torch.Tensor,
        node_mask: torch.Tensor,
        validate_inputs: bool = True,
    ) -> torch.Tensor:
        if not validate_inputs:
            # Caller asserts inputs already validated (hot-path, e.g. denoiser
            # timestep loop where adjacency / geodesic / mask are static across
            # the loop). Forward straight to compute.
            return self._compute(x, adjacency, geodesic_dist, node_mask)

        if x.dim() != 3 or x.shape[-1] != self.d_model:
            raise ValueError(
                f"x must be [B, N, {self.d_model}], got {tuple(x.shape)}"
            )
        B, N, _ = x.shape
        # Reject empty batches loud: zero-sized B/N silently produce zero-sized
        # output, which downstream pool/VAE consumes as "everything padded" and
        # cascades into shape-only-OK-but-meaningless errors (codex M1.2 round 2 R12).
        if B <= 0 or N <= 0:
            raise ValueError(
                f"GraphAttentionBlock: batch B={B} and node count N={N} must be > 0"
            )
        if adjacency.shape != (B, N, N) or geodesic_dist.shape != (B, N, N):
            raise ValueError(
                f"adjacency/geodesic_dist must be [B={B}, N={N}, N={N}], "
                f"got {tuple(adjacency.shape)} and {tuple(geodesic_dist.shape)}"
            )
        if node_mask.shape != (B, N) or node_mask.dtype != torch.bool:
            raise ValueError(
                f"node_mask must be [B={B}, N={N}] bool, got "
                f"shape {tuple(node_mask.shape)} dtype {node_mask.dtype}"
            )

        # --- R12 fail-loud: dtype consistency (codex M1.2 round 5) ---
        # All float tensors must (a) be fp32 or fp64 (fp16/bf16 overflow at
        # softmax with -1e9 mask sentinel and at large bias terms), and
        # (b) match the module's parameter dtype (mixed-dtype matmul crashes
        # opaquely deep in attention compute).
        # bf16-safe (2026-06-03): bf16 IS allowed — its 8-bit exponent (range ±3e38,
        # same as fp32) does NOT overflow the -1e9 softmax sentinel or the additive
        # topology bias; softmax is forced to fp32 in _compute below. fp16 is STILL
        # rejected (5-bit exponent overflows at -1e9). The fp32/fp64 path is
        # byte-for-byte unchanged. Under autocast(bf16), x may be bf16 while module
        # weights stay fp32 — a valid autocast pattern (matmul casts internally), so
        # the strict x.dtype==weight.dtype check is enforced ONLY on the fp32/64 path.
        expected_dtype = self.q_proj.weight.dtype
        for name, t in (("x", x), ("adjacency", adjacency), ("geodesic_dist", geodesic_dist)):
            if t.dtype not in (torch.float32, torch.float64, torch.bfloat16):
                raise ValueError(
                    f"GraphAttentionBlock: {name}.dtype must be float32/float64/bfloat16, "
                    f"got {t.dtype} (fp16 unsupported: 5-bit exponent overflows the "
                    f"-1e9 softmax sentinel + additive bias)"
                )
            if t.dtype in (torch.float32, torch.float64) and t.dtype != expected_dtype:
                raise ValueError(
                    f"GraphAttentionBlock: {name}.dtype {t.dtype} != module dtype "
                    f"{expected_dtype} (cast inputs OR module to match)"
                )

        # --- R12 fail-loud: finite + topology semantic checks ---
        # (codex M1.2 round 1 + 3)
        # x: all entries must be finite.
        if not torch.isfinite(x).all():
            raise ValueError(
                "GraphAttentionBlock: x contains NaN or Inf"
            )
        # adjacency contract: finite, non-negative, symmetric, zero diagonal.
        # We allow weighted edges (not only binary {0,1}) to support pool
        # variants that may emit soft-weighted pooled adjacency, but the
        # geometric meaning must remain: undirected graph with no self-loops.
        if not torch.isfinite(adjacency).all():
            raise ValueError(
                "GraphAttentionBlock: adjacency contains NaN or Inf"
            )
        if (adjacency < 0).any():
            raise ValueError(
                "GraphAttentionBlock: adjacency contains negative values "
                "(edges must be non-negative weights)"
            )
        if (adjacency > 1.0).any():
            raise ValueError(
                "GraphAttentionBlock: adjacency contains values > 1.0; "
                "contract is binary {0,1} or soft [0,1] (large magnitudes "
                "would dominate the additive bias projection)"
            )
        # Symmetry with rtol=0 so large absolute asymmetry (e.g. 1e6 vs 1e6+1)
        # cannot slip past allclose's default relative tolerance (codex M1.2
        # round 4 R12 fix).
        if not torch.allclose(
            adjacency, adjacency.transpose(-2, -1), atol=1e-6, rtol=0.0
        ):
            raise ValueError(
                "GraphAttentionBlock: adjacency is not symmetric "
                "(undirected graph required)"
            )
        if (adjacency.diagonal(dim1=-2, dim2=-1) != 0).any():
            raise ValueError(
                "GraphAttentionBlock: adjacency has non-zero diagonal "
                "(self-loops not permitted)"
            )
        # geodesic_dist contract: no NaN, no -Inf (+Inf is legitimate per Floyd
        # unreachable-pair contract), non-negative on finite entries, symmetric
        # on finite entries, zero diagonal at valid nodes.
        if torch.isnan(geodesic_dist).any():
            raise ValueError(
                "GraphAttentionBlock: geodesic_dist contains NaN"
            )
        if (geodesic_dist == float("-inf")).any():
            raise ValueError(
                "GraphAttentionBlock: geodesic_dist contains -Inf "
                "(bug; only +Inf is legitimate for unreachable pairs)"
            )
        # Negative finite distances are nonsense (Floyd output is hop count ≥ 0).
        finite_geo = geodesic_dist[torch.isfinite(geodesic_dist)]
        if (finite_geo < 0).any():
            raise ValueError(
                "GraphAttentionBlock: geodesic_dist has negative finite entries "
                "(distances must be ≥ 0)"
            )
        # Floyd hop-count upper bound: max hops ≤ N-1 (line-graph case on N
        # nodes). Larger finite values would dominate the additive bias and
        # silently corrupt attention. Codex M1.2 round 5 R12 fix.
        if (finite_geo > (N - 1)).any():
            raise ValueError(
                f"GraphAttentionBlock: geodesic_dist has finite entries > {N - 1} "
                f"(max hop-count on {N} nodes); not a valid Floyd shortest-path output"
            )
        # Symmetry: two-stage check (codex M1.2 round 4 R12 fix).
        # (1) The finite/+Inf pattern must be symmetric — i.e. cell is finite
        #     iff its transpose is. Otherwise asymmetric reachability slips
        #     through e.g. geo[0,1]=+Inf, geo[1,0]=1.0.
        # (2) Where both sides are finite, values must allclose with rtol=0.
        gt = geodesic_dist.transpose(-2, -1)
        finite_g = torch.isfinite(geodesic_dist)
        finite_gt = torch.isfinite(gt)
        if not torch.equal(finite_g, finite_gt):
            raise ValueError(
                "GraphAttentionBlock: geodesic_dist finite/+Inf pattern is not "
                "symmetric (asymmetric reachability)"
            )
        both_finite = finite_g & finite_gt
        if not torch.allclose(
            geodesic_dist[both_finite], gt[both_finite], atol=1e-6, rtol=0.0
        ):
            raise ValueError(
                "GraphAttentionBlock: geodesic_dist is not symmetric on finite entries"
            )
        # Diagonal of geodesic at valid nodes must be zero (i->i distance).
        diag = geodesic_dist.diagonal(dim1=-2, dim2=-1)  # [B, N]
        if ((diag != 0) & node_mask).any():
            raise ValueError(
                "GraphAttentionBlock: geodesic_dist has non-zero diagonal "
                "at valid nodes (i→i distance must be 0)"
            )

        # Per-sample sanity: at least one valid node per batch element.
        # This MUST run before adj/geo cross-consistency below, since Floyd on
        # an all-False mask sample produces all-+Inf which would falsely trip
        # the reachability-pattern check.
        if (~node_mask.any(dim=1)).any():
            bad = (~node_mask.any(dim=1)).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(
                f"GraphAttentionBlock: node_mask has all-False rows for "
                f"batch element(s) {bad} (no valid nodes; attention undefined)"
            )

        # --- R12 fail-loud: adj/geo cross-consistency (codex M1.2 round 6) ---
        # geodesic_dist must equal floyd_shortest_path(adjacency, node_mask).
        # Without this, a bounded-but-wrong geo (e.g. correct adj, geo[0,3]=2
        # instead of 3 on a 4-node line) silently corrupts the additive topology
        # bias and skews attention. Costs O(B·N^3) per forward — acceptable at
        # N≤160 (~4M ops, <1ms on GPU).
        expected_geo = floyd_shortest_path(adjacency, node_mask)
        both_valid = node_mask[:, :, None] & node_mask[:, None, :]
        # Pattern check: reachability (finite/+Inf) must match on valid pairs.
        finite_actual = torch.isfinite(geodesic_dist) & both_valid
        finite_expected = torch.isfinite(expected_geo) & both_valid
        if not torch.equal(finite_actual, finite_expected):
            raise ValueError(
                "GraphAttentionBlock: geodesic_dist reachability pattern "
                "inconsistent with adjacency (Floyd-recomputed)"
            )
        # Value check on entries that are finite in BOTH:
        compare_mask = finite_actual & finite_expected
        if not torch.allclose(
            geodesic_dist[compare_mask], expected_geo[compare_mask],
            atol=1e-6, rtol=0.0,
        ):
            raise ValueError(
                "GraphAttentionBlock: geodesic_dist values inconsistent with "
                "shortest-path over adjacency (Floyd-recomputed)"
            )

        return self._compute(x, adjacency, geodesic_dist, node_mask)

    def _compute(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        geodesic_dist: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pure compute path; assumes inputs already validated."""
        B, N, _ = x.shape
        # --- Pre-norm + self-attn ---
        residual = x
        x_norm = self.norm1(x)

        q = self.q_proj(x_norm).view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        k = self.k_proj(x_norm).view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        v = self.v_proj(x_norm).view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        # q/k/v: [B, H, N, d_head]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        # [B, H, N, N]

        # Topology biases. geodesic_dist may contain +inf for legitimate
        # unreachable pairs (from floyd_shortest_path). Substitute +inf with
        # 0.0 BEFORE projecting — this gives a neutral additive bias on those
        # pairs. The key-mask masks out padded keys, so the neutral bias only
        # affects unmasked-but-disconnected pairs (rare; deferred to a later
        # learnable "unreachable" bucket per lit survey if it shows up in
        # generation eval). NaN/-Inf were rejected above.
        # Graph-aware variant only; the no_graph_spatial ablation skips the topo
        # bias entirely → plain slot self-attention (still node-masked below).
        if self.use_graph_bias:
            geo = geodesic_dist.clone()
            geo[torch.isinf(geo)] = 0.0
            geo_bias = self.geodesic_bias(geo.unsqueeze(-1))         # [B, N, N, H]
            adj_bias = self.adjacency_bias(adjacency.unsqueeze(-1))  # [B, N, N, H]
            topo_bias = (geo_bias + adj_bias).permute(0, 3, 1, 2)    # [B, H, N, N]
            scores = scores + topo_bias

        # Mask invalid nodes (key side). Use large finite negative for softmax
        # numerical safety; matches encoder.py:84-85.
        mask = node_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
        scores = scores.masked_fill(~mask, -1e9)

        # Softmax. The earlier `all-False node_mask` per-sample guard ensures
        # at least one valid key per batch element, so no row of `scores` is
        # entirely -1e9 → no NaN in softmax output → no nan_to_num needed.
        # Padded-query rows still compute attention (over valid keys); their
        # output is zeroed downstream by the caller's joint_mask multiplication.
        # softmax in fp32 for bf16-safety (sentinel + reduction precision). On the
        # fp32 path scores.float() is a no-op and .to(scores.dtype) returns fp32, so
        # behavior is byte-for-byte unchanged; on the bf16 path softmax runs in fp32
        # then casts the probabilities back to bf16 for the attn@v matmul.
        attn = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B, H, N, d_head]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, self.d_model)
        out = self.o_proj(out)
        x = residual + self.dropout(out)

        # --- Pre-norm + FFN ---
        x = x + self.ff(self.norm2(x))

        return x
