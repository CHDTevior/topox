"""Minimal data subpackage for the topology-aware motion baseline.

Only exports what scripts/{train,eval,animate,self_test}.py and src/models/*.py
need on the supported path: the dataset class, its collate_fn, and SkeletonGraph
(used internally by unified_dataset for skeleton dict reconstruction).

bvh_parser / fbx_reader / humanml3d_converter / canonical_names from the source
repo are intentionally NOT exported here — they are convenience exports in the
source repo's __init__.py but are NOT referenced by unified_dataset.py's body
on the supported path, so they are out of scope for this clean baseline.
The legacy exports are loaded lazily.  This keeps schema-only utilities such as
``src.data.ktjd17.schema`` independent from Torch and the training stack while
preserving ``from src.data import UnifiedMotionDataset`` compatibility.
"""

from __future__ import annotations

from typing import Any


__all__ = ["UnifiedMotionDataset", "collate_fn", "SkeletonGraph"]


def __getattr__(name: str) -> Any:
    if name in {"UnifiedMotionDataset", "collate_fn"}:
        from .unified_dataset import UnifiedMotionDataset, collate_fn

        value = {
            "UnifiedMotionDataset": UnifiedMotionDataset,
            "collate_fn": collate_fn,
        }[name]
    elif name == "SkeletonGraph":
        from .skeleton_graph import SkeletonGraph

        value = SkeletonGraph
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
