"""GraphMotionBatch — typed wrapper around UnifiedMotionDataset.collate_fn output.

The existing src/data/unified_dataset.py::collate_fn already produces a dict
with all fields graph_salad needs (padded tensors + variable-length per-sample
lists). This module wraps that dict in a typed dataclass so pool / VAE /
denoiser code can use attribute access instead of string keys, and so any
schema drift is caught loud at construction.

Plan §3 originally proposed a GraphMotionBatch dataclass with custom collate;
codex pre-scaffold review confirmed the existing collate_fn covers everything,
so this dataclass is just a typed view — no new collate, no tensor copies.

R12 fail-loud (codex M1.1 review 2026-05-20): from_collate_dict validates
every tensor's rank + last-dim + dtype + batched-scalar shape/dtype, raising
ValueError on first mismatch. Wrong-shape or wrong-dtype inputs would
otherwise surface as opaque matmul errors deep in pool/VAE forward passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .graph_utils import assert_root_first_parent_order, validate_parent_tree


# Tensor shape spec: (key, expected_rank, last_dim_or_None, expected_dtype).
# Batch dim B is always axis 0. Cross-tensor (T_max, J_max) consistency is
# checked separately after per-tensor validation. "last_dim_or_None=None"
# means the last axis is variable (J_max, T_max, or J_max-square).
_TENSOR_SHAPE_SPEC: tuple[tuple[str, int, int | None, torch.dtype], ...] = (
    ("motion_features",      4, 6,    torch.float32),
    ("skeleton_features",    3, 9,    torch.float32),
    ("joint_mask",           2, None, torch.bool),
    ("frame_mask",           2, None, torch.bool),
    ("adjacency",            3, None, torch.float32),  # +square check below
    ("geodesic_dist",        3, None, torch.float32),  # +square check below
    ("name_hashes",          2, None, torch.int64),
    ("root_position",        3, 3,    torch.float32),
    ("root_velocity",        3, 3,    torch.float32),
    ("local_rotations_6d",   4, 6,    torch.float32),
    ("foot_contact",         3, 4,    torch.float32),
    ("bone_lengths",         3, None, torch.float32),
    ("rest_offsets",         3, 3,    torch.float32),
)

# Batched-scalar spec: (key, expected_dtype). All must be [B]-shape tensors
# after collate_fn (Python int/float/bool → torch.tensor via the int/float
# branch — `isinstance(True, int) == True`, so bools go through too).
_BATCHED_SCALAR_SPEC: tuple[tuple[str, torch.dtype], ...] = (
    ("num_joints",    torch.int64),
    ("num_frames",    torch.int64),
    ("fps",           torch.float32),
    ("has_rotations", torch.bool),
)

# Per-sample variable-length lists (collate_fn falls through to the else-list
# branch because the underlying field is a Python list or dict — never a
# scalar). These remain list[<per-sample-payload>] of length B.
_PER_SAMPLE_LIST_KEYS: tuple[str, ...] = (
    "parent_indices",       # list[list[int]]
    "joint_names",          # list[list[str]]
    "canonical_names",      # list[list[str]]
    "bone_lengths_rest",    # list[list[float]]
)

# Per-sample string lists (collate emits list[str] via the text-special branch
# for 'text', else-branch for others).
_PER_SAMPLE_STRING_KEYS: tuple[str, ...] = (
    "text",
    "skeleton_id",
    "motion_id",
)


@dataclass
class GraphMotionBatch:
    """Typed view of a `collate_fn` output dict from UnifiedMotionDataset.

    Constructed via ``GraphMotionBatch.from_collate_dict(d)`` — does NOT copy
    tensors; just holds references. Mutating tensors in-place mutates the
    underlying dict too.

    Shape / dtype conventions (B=batch_size, T_max=max_frames, J_max=max_joints):
      See ``_TENSOR_SHAPE_SPEC`` for padded-tensor contract.
      See ``_BATCHED_SCALAR_SPEC`` for [B]-shape scalar tensors.
      Lists: per-sample variable-length payloads (NOT stacked).
    """

    # Padded tensors
    motion_features: torch.Tensor
    skeleton_features: torch.Tensor
    joint_mask: torch.Tensor
    frame_mask: torch.Tensor
    adjacency: torch.Tensor
    geodesic_dist: torch.Tensor
    name_hashes: torch.Tensor
    root_position: torch.Tensor
    root_velocity: torch.Tensor
    local_rotations_6d: torch.Tensor
    foot_contact: torch.Tensor
    bone_lengths: torch.Tensor
    rest_offsets: torch.Tensor

    # Batched scalars (always torch.Tensor of shape [B], dtype per spec)
    num_joints: torch.Tensor
    num_frames: torch.Tensor
    fps: torch.Tensor
    has_rotations: torch.Tensor

    # Per-sample variable-length lists
    parent_indices: list[list[int]]
    joint_names: list[list[str]]
    canonical_names: list[list[str]]
    bone_lengths_rest: list[list[float]]

    # Per-sample string fields
    text: list[str]
    skeleton_id: list[str]
    motion_id: list[str]

    @classmethod
    def from_collate_dict(cls, d: dict[str, Any]) -> "GraphMotionBatch":
        """Construct from `collate_fn` output, fail loud on schema drift.

        Validates (in order):
          1. All required keys present.
          2. Padded tensors: type, rank, last-dim, dtype (per ``_TENSOR_SHAPE_SPEC``).
          3. Square check for adjacency / geodesic_dist.
          4. Batched scalars: type, shape == [B], dtype (per ``_BATCHED_SCALAR_SPEC``).
          5. Per-sample lists: type list, length == B.
          6. Cross-tensor T_max + J_max consistency.

        Raises ValueError with offending key on first mismatch.
        """
        # --- 1. Required-key presence ---
        all_required_keys = (
            [k for k, *_ in _TENSOR_SHAPE_SPEC]
            + [k for k, *_ in _BATCHED_SCALAR_SPEC]
            + list(_PER_SAMPLE_LIST_KEYS)
            + list(_PER_SAMPLE_STRING_KEYS)
        )
        missing = [k for k in all_required_keys if k not in d]
        if missing:
            raise ValueError(
                f"GraphMotionBatch: collate dict missing required keys: {missing}"
            )

        # --- 2-3. Padded tensors ---
        mf = d["motion_features"]
        if not isinstance(mf, torch.Tensor):
            raise ValueError(
                f"GraphMotionBatch: 'motion_features' must be torch.Tensor, "
                f"got {type(mf).__name__}"
            )
        if mf.dim() < 1:
            raise ValueError(
                f"GraphMotionBatch: 'motion_features' must have rank >= 1 to "
                f"derive batch size, got rank {mf.dim()} (scalar)"
            )
        B = mf.shape[0]
        if B <= 0:
            raise ValueError(
                f"GraphMotionBatch: batch size must be > 0, got B={B}"
            )

        # Device-consistency reference (all tensors must share device)
        ref_device = mf.device

        for key, expected_rank, last_dim, expected_dtype in _TENSOR_SHAPE_SPEC:
            t = d[key]
            if not isinstance(t, torch.Tensor):
                raise ValueError(
                    f"GraphMotionBatch: '{key}' must be torch.Tensor, "
                    f"got {type(t).__name__}"
                )
            if t.dim() != expected_rank:
                raise ValueError(
                    f"GraphMotionBatch: '{key}' must have rank {expected_rank}, "
                    f"got shape {tuple(t.shape)} (rank {t.dim()})"
                )
            if t.shape[0] != B:
                raise ValueError(
                    f"GraphMotionBatch: '{key}' batch dim {t.shape[0]} "
                    f"!= motion_features batch dim {B}"
                )
            if last_dim is not None and t.shape[-1] != last_dim:
                raise ValueError(
                    f"GraphMotionBatch: '{key}' last-dim must be {last_dim}, "
                    f"got shape {tuple(t.shape)}"
                )
            if t.dtype != expected_dtype:
                raise ValueError(
                    f"GraphMotionBatch: '{key}' dtype must be {expected_dtype}, "
                    f"got {t.dtype}"
                )
            if t.device != ref_device:
                raise ValueError(
                    f"GraphMotionBatch: '{key}' device {t.device} != "
                    f"motion_features device {ref_device}; all batch tensors "
                    f"must be on the same device"
                )
            if t.dtype.is_floating_point and not torch.isfinite(t).all():
                raise ValueError(
                    f"GraphMotionBatch: '{key}' contains NaN or Inf "
                    f"(non-finite values would corrupt loss / gradients)"
                )

        # Square check for graph tensors
        for key in ("adjacency", "geodesic_dist"):
            t = d[key]
            if t.shape[1] != t.shape[2]:
                raise ValueError(
                    f"GraphMotionBatch: '{key}' must be square (J_max x J_max), "
                    f"got shape {tuple(t.shape)}"
                )

        # --- 4. Batched scalars ---
        for key, expected_dtype in _BATCHED_SCALAR_SPEC:
            t = d[key]
            if not isinstance(t, torch.Tensor):
                raise ValueError(
                    f"GraphMotionBatch: batched-scalar '{key}' must be torch.Tensor "
                    f"(collate_fn should tensorize int/float/bool); "
                    f"got {type(t).__name__}"
                )
            if t.shape != (B,):
                raise ValueError(
                    f"GraphMotionBatch: batched-scalar '{key}' shape must be ({B},), "
                    f"got {tuple(t.shape)}"
                )
            if t.dtype != expected_dtype:
                raise ValueError(
                    f"GraphMotionBatch: batched-scalar '{key}' dtype must be "
                    f"{expected_dtype}, got {t.dtype}"
                )
            if t.device != ref_device:
                raise ValueError(
                    f"GraphMotionBatch: batched-scalar '{key}' device {t.device} "
                    f"!= motion_features device {ref_device}"
                )

        # --- 4b. Cross-tensor T_max + J_max consistency (structural, before value-level) ---
        T_max_val = d["motion_features"].shape[1]
        J_max_val = d["motion_features"].shape[2]
        for key in ("frame_mask", "root_position", "root_velocity", "foot_contact",
                    "local_rotations_6d", "bone_lengths"):
            if d[key].shape[1] != T_max_val:
                raise ValueError(
                    f"GraphMotionBatch: '{key}' T-dim {d[key].shape[1]} "
                    f"!= motion_features T_max {T_max_val}"
                )
        for key in ("skeleton_features", "joint_mask", "name_hashes", "rest_offsets",
                    "adjacency", "geodesic_dist"):
            if d[key].shape[1] != J_max_val:
                raise ValueError(
                    f"GraphMotionBatch: '{key}' J-dim {d[key].shape[1]} "
                    f"!= motion_features J_max {J_max_val}"
                )
        for key in ("local_rotations_6d", "bone_lengths"):
            if d[key].shape[2] != J_max_val:
                raise ValueError(
                    f"GraphMotionBatch: '{key}' J-dim {d[key].shape[2]} "
                    f"!= motion_features J_max {J_max_val}"
                )

        # --- 4c. Value-level guards on batched scalars (codex M1.1 round 3 R12) ---
        # num_joints / num_frames are authoritative lengths consumed by M1.2
        # pool / VAE; they must be in range AND match the corresponding mask.

        nj = d["num_joints"]
        if (nj <= 0).any() or (nj > J_max_val).any():
            raise ValueError(
                f"GraphMotionBatch: num_joints out of range [1, J_max={J_max_val}], "
                f"got {nj.tolist()}"
            )
        expected_nj = d["joint_mask"].sum(dim=1).to(nj.dtype)
        if not torch.equal(nj, expected_nj):
            raise ValueError(
                f"GraphMotionBatch: num_joints {nj.tolist()} != "
                f"joint_mask.sum(1) {expected_nj.tolist()}; collate-output "
                f"inconsistency would corrupt M1.2 pool length assumptions"
            )

        nf = d["num_frames"]
        if (nf <= 0).any() or (nf > T_max_val).any():
            raise ValueError(
                f"GraphMotionBatch: num_frames out of range [1, T_max={T_max_val}], "
                f"got {nf.tolist()}"
            )
        expected_nf = d["frame_mask"].sum(dim=1).to(nf.dtype)
        if not torch.equal(nf, expected_nf):
            raise ValueError(
                f"GraphMotionBatch: num_frames {nf.tolist()} != "
                f"frame_mask.sum(1) {expected_nf.tolist()}; collate-output "
                f"inconsistency would corrupt M1.2 temporal length assumptions"
            )

        # fps: NaN/Inf check + must be > 0 (fps is float, not covered by
        # _TENSOR_SHAPE_SPEC's isfinite guard)
        fps_t = d["fps"]
        if not torch.isfinite(fps_t).all():
            raise ValueError(
                f"GraphMotionBatch: fps contains NaN or Inf, got {fps_t.tolist()}"
            )
        if (fps_t <= 0).any():
            raise ValueError(
                f"GraphMotionBatch: fps must be > 0, got {fps_t.tolist()}"
            )

        # --- 5. Per-sample lists ---
        # Element-type contracts (codex M1.1 round 2 R12 fix):
        #   parent_indices    → list[list[int]]    inner ints (negatives allowed: -1 = root)
        #   joint_names       → list[list[str]]    inner str
        #   canonical_names   → list[list[str]]    inner str
        #   bone_lengths_rest → list[list[float]]  inner float (or int — Python numeric)
        _LIST_OF_LIST_ELEMENT_TYPE: dict[str, type | tuple[type, ...]] = {
            "parent_indices":    int,
            "joint_names":       str,
            "canonical_names":   str,
            "bone_lengths_rest": (int, float),
        }
        # num_joints already validated against joint_mask in step 4c above; use it
        # to constrain per-sample list cardinality (collate_fn emits variable-length
        # lists at length J_i, NOT padded to J_max).
        nj_list = d["num_joints"].tolist()

        for key in _PER_SAMPLE_LIST_KEYS:
            v = d[key]
            if not isinstance(v, list):
                raise ValueError(
                    f"GraphMotionBatch: '{key}' must be list, got {type(v).__name__}"
                )
            if len(v) != B:
                raise ValueError(
                    f"GraphMotionBatch: list '{key}' length {len(v)} != batch size {B}"
                )
            inner_type = _LIST_OF_LIST_ELEMENT_TYPE[key]
            for i, inner in enumerate(v):
                if not isinstance(inner, list):
                    raise ValueError(
                        f"GraphMotionBatch: '{key}'[{i}] must be list, "
                        f"got {type(inner).__name__}"
                    )
                # Cardinality must equal num_joints[i] (codex M1.1 round 4 R12 fix:
                # upstream collate emits variable-length per-sample lists at J_i,
                # not J_max-padded; M1.2 pool will iterate these as authoritative
                # joint enumerations).
                if len(inner) != nj_list[i]:
                    raise ValueError(
                        f"GraphMotionBatch: '{key}'[{i}] length {len(inner)} "
                        f"!= num_joints[{i}]={nj_list[i]}; per-sample lists "
                        f"must match the un-padded joint count"
                    )
                for j, item in enumerate(inner):
                    # Strict int / float: reject bool because `bool ⊂ int`
                    # in Python — codex M1.1 round 3 R12 fix.
                    if isinstance(item, bool):
                        raise ValueError(
                            f"GraphMotionBatch: '{key}'[{i}][{j}] is bool; "
                            f"must be strict {inner_type} (bool subclasses int "
                            f"but is semantically distinct)"
                        )
                    if not isinstance(item, inner_type):
                        raise ValueError(
                            f"GraphMotionBatch: '{key}'[{i}][{j}] type "
                            f"{type(item).__name__} != expected {inner_type}"
                        )

            # parent_indices needs structural validation (single root + range +
            # acyclic + connected) AND the FK-topology ordering invariant
            # required by treeik_decoder.py: root=0 + parents[j]<j for j>0.
            # Without the ordering invariant, downstream FK would consume
            # parents-after-children and crash deeper (codex M1.1 round 6
            # R12 category #8).
            if key == "parent_indices":
                for i, inner in enumerate(v):
                    try:
                        validate_parent_tree(inner)
                    except ValueError as e:
                        raise ValueError(
                            f"GraphMotionBatch: parent_indices[{i}] is not a "
                            f"valid rooted tree — {e}"
                        ) from e
                    try:
                        assert_root_first_parent_order(inner)
                    except ValueError as e:
                        raise ValueError(
                            f"GraphMotionBatch: parent_indices[{i}] violates "
                            f"FK ordering invariant (root must be index 0, "
                            f"parents must precede children) — {e}"
                        ) from e

        # --- 5b. Mask prefix-contiguous (codex M1.1 round 5 partial-leak fix) ---
        # joint_mask / frame_mask must be a contiguous prefix of True followed
        # by False (matches unified_dataset.py:151,158 which sets `mask[:J]=True`).
        # Sum-only checks accept holes like [1,0,1,...] that violate the contract.
        jm = d["joint_mask"]
        fm = d["frame_mask"]
        for b in range(B):
            nj_b = nj_list[b]
            if not jm[b, :nj_b].all():
                raise ValueError(
                    f"GraphMotionBatch: joint_mask[{b}] is not contiguous "
                    f"prefix-True; expected all True in [:{nj_b}], got holes"
                )
            if nj_b < J_max_val and jm[b, nj_b:].any():
                raise ValueError(
                    f"GraphMotionBatch: joint_mask[{b}] has True in padded "
                    f"region [{nj_b}:{J_max_val}]; must be all False"
                )
            nf_b = int(d["num_frames"][b].item())
            if not fm[b, :nf_b].all():
                raise ValueError(
                    f"GraphMotionBatch: frame_mask[{b}] is not contiguous "
                    f"prefix-True; expected all True in [:{nf_b}], got holes"
                )
            if nf_b < T_max_val and fm[b, nf_b:].any():
                raise ValueError(
                    f"GraphMotionBatch: frame_mask[{b}] has True in padded "
                    f"region [{nf_b}:{T_max_val}]; must be all False"
                )

        # --- 5c. Graph-semantics consistency (codex M1.1 round 5 R12 fix) ---
        # adjacency[b,:nj,:nj] must match the undirected graph derived from
        # parent_indices[b]: binary {0,1}, symmetric, zero diagonal, edge iff
        # parent-child relation. Padded rows/cols must be zero.
        adj = d["adjacency"]
        parents_per_sample = d["parent_indices"]
        for b in range(B):
            nj_b = nj_list[b]
            sub = adj[b, :nj_b, :nj_b]

            # Binary: every entry in {0, 1}
            if ((sub != 0) & (sub != 1)).any():
                raise ValueError(
                    f"GraphMotionBatch: adjacency[{b},:{nj_b},:{nj_b}] has "
                    f"non-binary entries (must be 0 or 1)"
                )
            # Symmetric
            if not torch.equal(sub, sub.T):
                raise ValueError(
                    f"GraphMotionBatch: adjacency[{b},:{nj_b},:{nj_b}] not "
                    f"symmetric (must be undirected)"
                )
            # Zero diagonal
            if sub.diagonal().any():
                raise ValueError(
                    f"GraphMotionBatch: adjacency[{b}] has non-zero diagonal "
                    f"(no self-loops permitted)"
                )
            # Build expected adj from parents and compare
            expected_sub = torch.zeros(nj_b, nj_b, dtype=adj.dtype, device=adj.device)
            for j, p in enumerate(parents_per_sample[b]):
                if p >= 0:
                    expected_sub[j, p] = 1.0
                    expected_sub[p, j] = 1.0
            if not torch.equal(sub, expected_sub):
                raise ValueError(
                    f"GraphMotionBatch: adjacency[{b},:{nj_b},:{nj_b}] does "
                    f"not match the undirected graph derived from "
                    f"parent_indices[{b}]"
                )
            # Padded rows / cols must be all zero
            if nj_b < J_max_val:
                if adj[b, nj_b:, :].any() or adj[b, :, nj_b:].any():
                    raise ValueError(
                        f"GraphMotionBatch: adjacency[{b}] has non-zero "
                        f"entries in padded rows/cols [{nj_b}:{J_max_val}]"
                    )

        for key in _PER_SAMPLE_STRING_KEYS:
            v = d[key]
            if not isinstance(v, list):
                raise ValueError(
                    f"GraphMotionBatch: '{key}' must be list, got {type(v).__name__}"
                )
            if len(v) != B:
                raise ValueError(
                    f"GraphMotionBatch: list '{key}' length {len(v)} != batch size {B}"
                )
            for i, s in enumerate(v):
                if not isinstance(s, str):
                    raise ValueError(
                        f"GraphMotionBatch: '{key}'[{i}] must be str, "
                        f"got {type(s).__name__}"
                    )

        return cls(
            motion_features=d["motion_features"],
            skeleton_features=d["skeleton_features"],
            joint_mask=d["joint_mask"],
            frame_mask=d["frame_mask"],
            adjacency=d["adjacency"],
            geodesic_dist=d["geodesic_dist"],
            name_hashes=d["name_hashes"],
            root_position=d["root_position"],
            root_velocity=d["root_velocity"],
            local_rotations_6d=d["local_rotations_6d"],
            foot_contact=d["foot_contact"],
            bone_lengths=d["bone_lengths"],
            rest_offsets=d["rest_offsets"],
            num_joints=d["num_joints"],
            num_frames=d["num_frames"],
            fps=d["fps"],
            has_rotations=d["has_rotations"],
            parent_indices=d["parent_indices"],
            joint_names=d["joint_names"],
            canonical_names=d["canonical_names"],
            bone_lengths_rest=d["bone_lengths_rest"],
            text=d["text"],
            skeleton_id=d["skeleton_id"],
            motion_id=d["motion_id"],
        )

    @property
    def batch_size(self) -> int:
        return int(self.motion_features.shape[0])

    @property
    def max_frames(self) -> int:
        return int(self.motion_features.shape[1])

    @property
    def max_joints(self) -> int:
        return int(self.motion_features.shape[2])
