"""Streaming all-data KTJD-17 mean/std grouped by biological species."""

from __future__ import annotations

import concurrent.futures
import dataclasses
import datetime as _datetime
import hashlib
import json
import os
import re
import shutil
import uuid
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .loader import load_motion_npz


STATS_VERSION = "ktjd17-pz-human312-species-and-rig-stats-v2"
EXPECTED_FULL_STATUS = "full_numeric_pass_visual_gate_bound"
EXPECTED_RIG_COUNT = 312
EXPECTED_SPECIES_COUNT = 117
CHANNEL_NAMES = (
    "q_x",
    "q_y",
    "q_z",
    "rest_delta_6d_c0_x",
    "rest_delta_6d_c0_y",
    "rest_delta_6d_c0_z",
    "rest_delta_6d_c1_x",
    "rest_delta_6d_c1_y",
    "rest_delta_6d_c1_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "contact",
    "smooth_root_x",
    "smooth_root_z",
    "heading_cos",
    "heading_sin",
)
_PZ_RIG = re.compile(r"PZ_(.+)_(Female|Male|Juvenile)\Z")


class SpeciesStatsError(RuntimeError):
    """Raised when a full generation cannot yield trustworthy statistics."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def species_from_rig_id(rig_id: str) -> str:
    """Collapse PZ sex/age rigs to species; keep the Human corpus separate."""

    value = str(rig_id)
    if value == "HML3D_Human":
        return "Human"
    match = _PZ_RIG.fullmatch(value)
    if match is None:
        raise SpeciesStatsError(f"unsupported PZ/Human rig identifier: {value}")
    return match.group(1)


@dataclasses.dataclass
class ChannelMoments:
    """Per-channel population moments using deterministic Chan merges."""

    count: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(17, dtype=np.int64)
    )
    mean: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(17, dtype=np.float64)
    )
    m2: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(17, dtype=np.float64)
    )
    minimum: np.ndarray = dataclasses.field(
        default_factory=lambda: np.full(17, np.inf, dtype=np.float64)
    )
    maximum: np.ndarray = dataclasses.field(
        default_factory=lambda: np.full(17, -np.inf, dtype=np.float64)
    )

    def update(self, values: np.ndarray, channels: slice) -> None:
        array = np.asarray(values, dtype=np.float64)
        indices = np.arange(17)[channels]
        if array.ndim != 2 or array.shape[1] != len(indices):
            raise SpeciesStatsError(
                f"invalid moment batch {array.shape} for channels {channels}"
            )
        if array.shape[0] == 0:
            return
        if not np.isfinite(array).all():
            raise SpeciesStatsError("non-finite value encountered while computing stats")
        batch_count = np.full(len(indices), array.shape[0], dtype=np.int64)
        batch_mean = np.mean(array, axis=0, dtype=np.float64)
        centered = array - batch_mean[None]
        batch_m2 = np.sum(centered * centered, axis=0, dtype=np.float64)
        batch_min = np.min(array, axis=0)
        batch_max = np.max(array, axis=0)
        self._merge_arrays(
            indices,
            batch_count,
            batch_mean,
            batch_m2,
            batch_min,
            batch_max,
        )

    def merge(self, other: "ChannelMoments") -> None:
        self._merge_arrays(
            np.arange(17),
            other.count,
            other.mean,
            other.m2,
            other.minimum,
            other.maximum,
        )

    def _merge_arrays(
        self,
        indices: np.ndarray,
        count: np.ndarray,
        mean: np.ndarray,
        m2: np.ndarray,
        minimum: np.ndarray,
        maximum: np.ndarray,
    ) -> None:
        for local, channel in enumerate(indices.tolist()):
            incoming = int(count[local])
            if incoming == 0:
                continue
            previous = int(self.count[channel])
            if previous == 0:
                self.count[channel] = incoming
                self.mean[channel] = float(mean[local])
                self.m2[channel] = float(m2[local])
                self.minimum[channel] = float(minimum[local])
                self.maximum[channel] = float(maximum[local])
                continue
            total = previous + incoming
            delta = float(mean[local]) - float(self.mean[channel])
            self.mean[channel] += delta * incoming / total
            self.m2[channel] += float(m2[local]) + (
                delta * delta * previous * incoming / total
            )
            self.count[channel] = total
            self.minimum[channel] = min(
                float(self.minimum[channel]), float(minimum[local])
            )
            self.maximum[channel] = max(
                float(self.maximum[channel]), float(maximum[local])
            )

    def population_std(self) -> np.ndarray:
        if np.any(self.count <= 0):
            missing = np.flatnonzero(self.count <= 0).tolist()
            raise SpeciesStatsError(f"channels have no valid observations: {missing}")
        variance = self.m2 / self.count.astype(np.float64)
        tolerance = np.finfo(np.float64).eps * np.maximum(1.0, np.abs(self.mean)) ** 2
        if np.any(variance < -tolerance):
            raise SpeciesStatsError("negative variance beyond floating tolerance")
        return np.sqrt(np.maximum(variance, 0.0))


@dataclasses.dataclass
class RigCellMoments:
    """Population moments for every physical joint/channel cell of one rig."""

    rig_id: str
    count: np.ndarray | None = None
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None
    minimum: np.ndarray | None = None
    maximum: np.ndarray | None = None
    clip_count: int = 0
    frame_count: int = 0
    heading_valid_frame_count: int = 0

    def _initialize(self, joint_count: int) -> None:
        shape = (int(joint_count), 17)
        self.count = np.zeros(shape, dtype=np.int64)
        self.mean = np.zeros(shape, dtype=np.float64)
        self.m2 = np.zeros(shape, dtype=np.float64)
        self.minimum = np.full(shape, np.inf, dtype=np.float64)
        self.maximum = np.full(shape, -np.inf, dtype=np.float64)

    def _require_arrays(self) -> tuple[np.ndarray, ...]:
        arrays = (self.count, self.mean, self.m2, self.minimum, self.maximum)
        if any(value is None for value in arrays):
            raise SpeciesStatsError(f"rig moments are uninitialized: {self.rig_id}")
        return tuple(np.asarray(value) for value in arrays)

    def update(self, motion: np.ndarray, heading_valid: np.ndarray) -> None:
        values = np.asarray(motion, dtype=np.float64)
        headings = np.asarray(heading_valid, dtype=bool)
        T, J, D = values.shape
        if D != 17 or headings.shape != (T,) or not np.isfinite(values).all():
            raise SpeciesStatsError(f"invalid rig-cell batch: {self.rig_id}")
        if self.count is None:
            self._initialize(J)
        elif self.count.shape != (J, 17):
            raise SpeciesStatsError(f"joint count changed within rig {self.rig_id}")

        batch_count = np.zeros((J, 17), dtype=np.int64)
        batch_mean = np.zeros((J, 17), dtype=np.float64)
        batch_m2 = np.zeros((J, 17), dtype=np.float64)
        batch_min = np.full((J, 17), np.inf, dtype=np.float64)
        batch_max = np.full((J, 17), -np.inf, dtype=np.float64)

        common = values[..., :13]
        common_mean = np.mean(common, axis=0, dtype=np.float64)
        batch_count[:, :13] = T
        batch_mean[:, :13] = common_mean
        batch_m2[:, :13] = np.sum(
            (common - common_mean[None]) ** 2, axis=0, dtype=np.float64
        )
        batch_min[:, :13] = np.min(common, axis=0)
        batch_max[:, :13] = np.max(common, axis=0)

        root_xz = values[:, 0, 13:15]
        root_xz_mean = np.mean(root_xz, axis=0, dtype=np.float64)
        batch_count[0, 13:15] = T
        batch_mean[0, 13:15] = root_xz_mean
        batch_m2[0, 13:15] = np.sum(
            (root_xz - root_xz_mean[None]) ** 2, axis=0, dtype=np.float64
        )
        batch_min[0, 13:15] = np.min(root_xz, axis=0)
        batch_max[0, 13:15] = np.max(root_xz, axis=0)

        heading_count = int(np.count_nonzero(headings))
        if heading_count:
            heading = values[headings, 0, 15:17]
            heading_mean = np.mean(heading, axis=0, dtype=np.float64)
            batch_count[0, 15:17] = heading_count
            batch_mean[0, 15:17] = heading_mean
            batch_m2[0, 15:17] = np.sum(
                (heading - heading_mean[None]) ** 2, axis=0, dtype=np.float64
            )
            batch_min[0, 15:17] = np.min(heading, axis=0)
            batch_max[0, 15:17] = np.max(heading, axis=0)
        self._merge_arrays(batch_count, batch_mean, batch_m2, batch_min, batch_max)
        self.clip_count += 1
        self.frame_count += T
        self.heading_valid_frame_count += heading_count

    def _merge_arrays(
        self,
        incoming_count: np.ndarray,
        incoming_mean: np.ndarray,
        incoming_m2: np.ndarray,
        incoming_min: np.ndarray,
        incoming_max: np.ndarray,
    ) -> None:
        count, mean, m2, minimum, maximum = self._require_arrays()
        if incoming_count.shape != count.shape:
            raise SpeciesStatsError(f"rig-cell shape mismatch: {self.rig_id}")
        valid = incoming_count > 0
        empty = valid & (count == 0)
        mean[empty] = incoming_mean[empty]
        m2[empty] = incoming_m2[empty]
        minimum[empty] = incoming_min[empty]
        maximum[empty] = incoming_max[empty]
        count[empty] = incoming_count[empty]

        both = valid & ~empty
        previous = count[both].astype(np.float64)
        incoming = incoming_count[both].astype(np.float64)
        total = previous + incoming
        delta = incoming_mean[both] - mean[both]
        mean[both] += delta * incoming / total
        m2[both] += incoming_m2[both] + delta * delta * previous * incoming / total
        minimum[both] = np.minimum(minimum[both], incoming_min[both])
        maximum[both] = np.maximum(maximum[both], incoming_max[both])
        count[both] += incoming_count[both]

    def merge(self, other: "RigCellMoments") -> None:
        if self.rig_id != other.rig_id:
            raise SpeciesStatsError(f"cannot merge rig cells {self.rig_id}/{other.rig_id}")
        if other.count is None:
            return
        if self.count is None:
            self._initialize(other.count.shape[0])
        other_arrays = other._require_arrays()
        self._merge_arrays(*other_arrays)
        self.clip_count += other.clip_count
        self.frame_count += other.frame_count
        self.heading_valid_frame_count += other.heading_valid_frame_count

    def finalized(self) -> dict[str, np.ndarray | int]:
        count, mean, m2, minimum, maximum = self._require_arrays()
        expected = np.zeros_like(count)
        expected[:, :13] = self.frame_count
        expected[0, 13:15] = self.frame_count
        expected[0, 15:17] = self.heading_valid_frame_count
        if not np.array_equal(count, expected):
            raise SpeciesStatsError(f"rig-cell valid-count mismatch: {self.rig_id}")
        valid = count > 0
        variance = np.zeros_like(mean)
        variance[valid] = m2[valid] / count[valid].astype(np.float64)
        tolerance = np.finfo(np.float64).eps * np.maximum(1.0, np.abs(mean)) ** 2
        if np.any(variance[valid] < -tolerance[valid]):
            raise SpeciesStatsError(f"negative rig-cell variance: {self.rig_id}")
        std = np.zeros_like(mean)
        std[valid] = np.sqrt(np.maximum(variance[valid], 0.0))
        output_min = np.zeros_like(mean)
        output_max = np.zeros_like(mean)
        output_min[valid] = minimum[valid]
        output_max[valid] = maximum[valid]
        return {
            "count": count.copy(),
            "mean": mean.copy(),
            "std": std,
            "minimum": output_min,
            "maximum": output_max,
            "joint_count": count.shape[0],
            "clip_count": self.clip_count,
            "frame_count": self.frame_count,
            "heading_valid_frame_count": self.heading_valid_frame_count,
        }


@dataclasses.dataclass
class SpeciesAggregate:
    species_id: str
    moments: ChannelMoments = dataclasses.field(default_factory=ChannelMoments)
    rig_ids: set[str] = dataclasses.field(default_factory=set)
    clip_count: int = 0
    frame_count: int = 0
    physical_joint_frame_count: int = 0
    heading_valid_frame_count: int = 0
    rig_moments: dict[str, RigCellMoments] = dataclasses.field(default_factory=dict)

    def merge(self, other: "SpeciesAggregate") -> None:
        if self.species_id != other.species_id:
            raise SpeciesStatsError(
                f"cannot merge species {self.species_id} and {other.species_id}"
            )
        self.moments.merge(other.moments)
        self.rig_ids.update(other.rig_ids)
        self.clip_count += other.clip_count
        self.frame_count += other.frame_count
        self.physical_joint_frame_count += other.physical_joint_frame_count
        self.heading_valid_frame_count += other.heading_valid_frame_count
        for rig_id, partial in other.rig_moments.items():
            target = self.rig_moments.setdefault(rig_id, RigCellMoments(rig_id))
            target.merge(partial)


def accumulate_motion(
    aggregate: SpeciesAggregate,
    *,
    motion: np.ndarray,
    heading_valid: np.ndarray,
    rig_id: str,
) -> None:
    values = np.asarray(motion)
    headings = np.asarray(heading_valid, dtype=bool)
    if (
        values.dtype != np.float32
        or values.ndim != 3
        or values.shape[-1] != 17
        or headings.shape != (values.shape[0],)
        or values.shape[0] <= 0
        or values.shape[1] <= 0
    ):
        raise SpeciesStatsError(
            f"invalid KTJD-17 payload for statistics: {values.dtype} {values.shape}"
        )
    if np.any(values[:, 1:, 13:17] != 0.0):
        raise SpeciesStatsError("non-root channels 13:17 are not exact zero")
    T, J, _ = values.shape
    aggregate.moments.update(values[..., :13].reshape(T * J, 13), slice(0, 13))
    aggregate.moments.update(values[:, 0, 13:15], slice(13, 15))
    aggregate.moments.update(values[headings, 0, 15:17], slice(15, 17))
    rig_key = str(rig_id)
    rig_moments = aggregate.rig_moments.setdefault(rig_key, RigCellMoments(rig_key))
    rig_moments.update(values, headings)
    aggregate.rig_ids.add(str(rig_id))
    aggregate.clip_count += 1
    aggregate.frame_count += T
    aggregate.physical_joint_frame_count += T * J
    aggregate.heading_valid_frame_count += int(np.count_nonzero(headings))


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SpeciesStatsError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise SpeciesStatsError(f"non-object at {path}:{line_number}")
            records.append(record)
    return records


def _compute_shard(task: tuple[str, str, float, list[dict[str, Any]]]) -> SpeciesAggregate:
    root_text, species_id, fps_target, records = task
    root = Path(root_text)
    aggregate = SpeciesAggregate(species_id)
    for record in records:
        rig_id = str(record["rig_id"])
        if species_from_rig_id(rig_id) != species_id:
            raise SpeciesStatsError(f"species shard mismatch: {species_id}/{rig_id}")
        motion_path = root / str(record["motion_relpath"])
        payload = load_motion_npz(motion_path, expected_fps_target=fps_target)
        motion = np.asarray(payload["motion"])
        heading_valid = np.asarray(payload["heading_valid"], dtype=bool)
        if (
            payload["clip_id"] != record["clip_id"]
            or payload["rig_id"] != rig_id
            or motion.shape[0] != int(record["T_target"])
            or motion.shape[1] != int(record["J_phys"])
        ):
            raise SpeciesStatsError(f"manifest/payload mismatch: {record['clip_id']}")
        accumulate_motion(
            aggregate,
            motion=motion,
            heading_valid=heading_valid,
            rig_id=rig_id,
        )
    return aggregate


def _chunks(values: Sequence[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def _aggregate_record(aggregate: SpeciesAggregate) -> dict[str, Any]:
    std = aggregate.moments.population_std()
    counts = aggregate.moments.count
    if (
        not np.all(counts[:13] == aggregate.physical_joint_frame_count)
        or not np.all(counts[13:15] == aggregate.frame_count)
        or not np.all(counts[15:17] == aggregate.heading_valid_frame_count)
    ):
        raise SpeciesStatsError(f"valid-count contract failed: {aggregate.species_id}")
    return {
        "rig_ids": sorted(aggregate.rig_ids),
        "rig_count": len(aggregate.rig_ids),
        "clip_count": aggregate.clip_count,
        "frame_count": aggregate.frame_count,
        "physical_joint_frame_count": aggregate.physical_joint_frame_count,
        "heading_valid_frame_count": aggregate.heading_valid_frame_count,
        "valid_value_count": counts.tolist(),
        "mean": aggregate.moments.mean.tolist(),
        "std": std.tolist(),
        "minimum": aggregate.moments.minimum.tolist(),
        "maximum": aggregate.moments.maximum.tolist(),
    }


def _write_outputs(
    output: Path,
    *,
    generation: Mapping[str, Any],
    generation_sha256: str,
    aggregates: Mapping[str, SpeciesAggregate],
) -> dict[str, Any]:
    species_ids = sorted(aggregates)
    records = {species: _aggregate_record(aggregates[species]) for species in species_ids}
    global_aggregate = SpeciesAggregate("ALL")
    for species in species_ids:
        global_aggregate.moments.merge(aggregates[species].moments)
        global_aggregate.rig_ids.update(aggregates[species].rig_ids)
        global_aggregate.clip_count += aggregates[species].clip_count
        global_aggregate.frame_count += aggregates[species].frame_count
        global_aggregate.physical_joint_frame_count += aggregates[
            species
        ].physical_joint_frame_count
        global_aggregate.heading_valid_frame_count += aggregates[
            species
        ].heading_valid_frame_count
    global_record = _aggregate_record(global_aggregate)
    rig_moments: dict[str, RigCellMoments] = {}
    rig_species: dict[str, str] = {}
    for species_id in species_ids:
        for rig_id, moments in aggregates[species_id].rig_moments.items():
            if rig_id in rig_moments:
                raise SpeciesStatsError(f"rig appears in two species: {rig_id}")
            rig_moments[rig_id] = moments
            rig_species[rig_id] = species_id
    if (
        len(species_ids) != EXPECTED_SPECIES_COUNT
        or global_record["rig_count"] != EXPECTED_RIG_COUNT
        or global_record["clip_count"] != int(generation["accepted_clip_count"])
        or len(rig_moments) != EXPECTED_RIG_COUNT
    ):
        raise SpeciesStatsError(
            "species/global coverage mismatch: "
            f"species={len(species_ids)}, rigs={global_record['rig_count']}, "
            f"clips={global_record['clip_count']}"
        )

    payload = {
        "stats_version": STATS_VERSION,
        "created_at_utc": _datetime.datetime.now(_datetime.UTC).isoformat(),
        "source_generation_id": generation["generation_id"],
        "source_generation_json_sha256": generation_sha256,
        "source_generation_status": generation["status"],
        "scope": "all accepted clips across train/val/test",
        "value_domain": "raw float32 KTJD-17 storage before normalization or padding",
        "std_definition": "population standard deviation (ddof=0)",
        "channel_names": list(CHANNEL_NAMES),
        "validity": {
            "channels_0_12": "all frames and physical joints",
            "channels_13_14": "all frames, physical root only",
            "channels_15_16": "heading_valid frames, physical root only",
            "padding": "excluded",
            "invalid_heading_zero_sentinel": "excluded",
        },
        "grouping": {
            "planetzoo": "strip PZ_ prefix and final _Female/_Male/_Juvenile suffix",
            "motionstreamer272": "HML3D_Human maps to Human",
            "species_count": len(species_ids),
        },
        "rig_cell_statistics": {
            "artifact": "rig_stats.npz",
            "scope": "each of 312 physical rigs over all accepted clips",
            "shape": "[rig, padded_joint, channel] with count>0 as validity mask",
            "mean_std": "population moments per physical joint and KTJD-17 channel",
            "invalid_cells": "count=0 and stored mean/std/min/max=0",
            "padding": "count=0; joint_count identifies physical extent",
            "normalization_note": "std is empirical ddof=0; consumers choose their own floor",
        },
        "global": global_record,
        "species": records,
    }

    parent = output.parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise SpeciesStatsError(f"refusing to overwrite stats output: {output}")
    staging = parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        json_path = staging / "species_stats.json"
        json_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        mean = np.asarray([records[s]["mean"] for s in species_ids], dtype=np.float64)
        std = np.asarray([records[s]["std"] for s in species_ids], dtype=np.float64)
        count = np.asarray(
            [records[s]["valid_value_count"] for s in species_ids], dtype=np.int64
        )
        minimum = np.asarray(
            [records[s]["minimum"] for s in species_ids], dtype=np.float64
        )
        maximum = np.asarray(
            [records[s]["maximum"] for s in species_ids], dtype=np.float64
        )
        np.savez_compressed(
            staging / "species_stats.npz",
            species_ids=np.asarray(species_ids),
            channel_names=np.asarray(CHANNEL_NAMES),
            mean=mean,
            std=std,
            count=count,
            minimum=minimum,
            maximum=maximum,
            clip_count=np.asarray(
                [records[s]["clip_count"] for s in species_ids], dtype=np.int64
            ),
            frame_count=np.asarray(
                [records[s]["frame_count"] for s in species_ids], dtype=np.int64
            ),
            rig_count=np.asarray(
                [records[s]["rig_count"] for s in species_ids], dtype=np.int64
            ),
            global_mean=np.asarray(global_record["mean"], dtype=np.float64),
            global_std=np.asarray(global_record["std"], dtype=np.float64),
            global_count=np.asarray(
                global_record["valid_value_count"], dtype=np.int64
            ),
        )
        rig_ids = sorted(rig_moments)
        finalized_rigs = {rig_id: rig_moments[rig_id].finalized() for rig_id in rig_ids}
        J_max = max(int(finalized_rigs[rig_id]["joint_count"]) for rig_id in rig_ids)
        rig_shape = (len(rig_ids), J_max, 17)
        rig_mean = np.zeros(rig_shape, dtype=np.float64)
        rig_std = np.zeros(rig_shape, dtype=np.float64)
        rig_count = np.zeros(rig_shape, dtype=np.int64)
        rig_minimum = np.zeros(rig_shape, dtype=np.float64)
        rig_maximum = np.zeros(rig_shape, dtype=np.float64)
        joint_count = np.zeros(len(rig_ids), dtype=np.int64)
        rig_clip_count = np.zeros(len(rig_ids), dtype=np.int64)
        rig_frame_count = np.zeros(len(rig_ids), dtype=np.int64)
        rig_heading_count = np.zeros(len(rig_ids), dtype=np.int64)
        for index, rig_id in enumerate(rig_ids):
            record = finalized_rigs[rig_id]
            joints = int(record["joint_count"])
            joint_count[index] = joints
            rig_mean[index, :joints] = np.asarray(record["mean"])
            rig_std[index, :joints] = np.asarray(record["std"])
            rig_count[index, :joints] = np.asarray(record["count"])
            rig_minimum[index, :joints] = np.asarray(record["minimum"])
            rig_maximum[index, :joints] = np.asarray(record["maximum"])
            rig_clip_count[index] = int(record["clip_count"])
            rig_frame_count[index] = int(record["frame_count"])
            rig_heading_count[index] = int(record["heading_valid_frame_count"])
        np.savez_compressed(
            staging / "rig_stats.npz",
            rig_ids=np.asarray(rig_ids),
            biological_species_ids=np.asarray([rig_species[rig_id] for rig_id in rig_ids]),
            channel_names=np.asarray(CHANNEL_NAMES),
            joint_count=joint_count,
            mean=rig_mean,
            std=rig_std,
            count=rig_count,
            valid_mask=rig_count > 0,
            minimum=rig_minimum,
            maximum=rig_maximum,
            clip_count=rig_clip_count,
            frame_count=rig_frame_count,
            heading_valid_frame_count=rig_heading_count,
        )
        files = {
            name: {
                "size_bytes": (staging / name).stat().st_size,
                "sha256": _sha256_file(staging / name),
            }
            for name in ("species_stats.json", "species_stats.npz", "rig_stats.npz")
        }
        manifest = {
            "stats_version": STATS_VERSION,
            "created_at_utc": payload["created_at_utc"],
            "status": "pass",
            "source_generation_id": generation["generation_id"],
            "source_generation_json_sha256": generation_sha256,
            "species_count": len(species_ids),
            "rig_count": global_record["rig_count"],
            "clip_count": global_record["clip_count"],
            "rig_cell_J_max": J_max,
            "rig_cell_shape": list(rig_shape),
            "files": files,
        }
        (staging / "generation.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        with np.load(staging / "species_stats.npz", allow_pickle=False) as compact:
            if (
                compact["mean"].shape != (len(species_ids), 17)
                or compact["std"].shape != (len(species_ids), 17)
                or compact["count"].shape != (len(species_ids), 17)
                or compact["species_ids"].astype(str).tolist() != species_ids
                or not np.array_equal(compact["count"], count)
                or not np.allclose(compact["mean"], mean, rtol=0.0, atol=0.0)
                or not np.allclose(compact["std"], std, rtol=0.0, atol=0.0)
            ):
                raise SpeciesStatsError("JSON/NPZ species statistics disagree")
        with np.load(staging / "rig_stats.npz", allow_pickle=False) as compact:
            if (
                compact["mean"].shape != rig_shape
                or compact["std"].shape != rig_shape
                or compact["count"].shape != rig_shape
                or compact["rig_ids"].astype(str).tolist() != rig_ids
                or not np.array_equal(compact["count"], rig_count)
                or not np.array_equal(compact["valid_mask"], rig_count > 0)
                or not np.allclose(compact["mean"], rig_mean, rtol=0.0, atol=0.0)
                or not np.allclose(compact["std"], rig_std, rtol=0.0, atol=0.0)
            ):
                raise SpeciesStatsError("written rig-cell statistics disagree")
        for path in staging.iterdir():
            os.chmod(path, 0o444)
        os.chmod(staging, 0o555)
        os.replace(staging, output)
    except BaseException:
        if staging.exists():
            os.chmod(staging, 0o755)
            for path in staging.iterdir():
                os.chmod(path, 0o644)
            shutil.rmtree(staging)
        raise
    return {
        "output": str(output),
        "source_generation_id": generation["generation_id"],
        "species_count": len(species_ids),
        "rig_count": global_record["rig_count"],
        "clip_count": global_record["clip_count"],
        "generation_json_sha256": _sha256_file(output / "generation.json"),
    }


def compute_species_stats(
    generation_root: str | Path,
    output_root: str | Path,
    *,
    workers: int = 16,
    shard_size: int = 256,
) -> dict[str, Any]:
    root = Path(generation_root).expanduser().resolve()
    output = Path(output_root).expanduser().absolute()
    generation_path = root / "generation.json"
    generation = _load_json(generation_path)
    if (
        generation.get("mode") != "full"
        or generation.get("status") != EXPECTED_FULL_STATUS
        or generation.get("full_conversion_authorized") is not True
        or int(generation.get("rig_count", -1)) != EXPECTED_RIG_COUNT
        or int(generation.get("accepted_clip_count", -1)) <= 0
    ):
        raise SpeciesStatsError("input is not a complete PZ-311 plus Human-1 full build")
    generation_sha256 = _sha256_file(generation_path)
    records = _load_jsonl(root / "manifests/clips.jsonl")
    if len(records) != int(generation["accepted_clip_count"]):
        raise SpeciesStatsError("clip manifest count does not match generation")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rig_ids: set[str] = set()
    fps_values: set[float] = set()
    for record in records:
        if record.get("status") != "accept":
            raise SpeciesStatsError(f"non-accepted clip in full manifest: {record}")
        rig_id = str(record["rig_id"])
        grouped[species_from_rig_id(rig_id)].append(record)
        rig_ids.add(rig_id)
        fps_values.add(float(record["fps_target"]))
    if (
        len(grouped) != EXPECTED_SPECIES_COUNT
        or len(rig_ids) != EXPECTED_RIG_COUNT
        or len(fps_values) != 1
    ):
        raise SpeciesStatsError(
            f"unexpected grouping: species={len(grouped)}, rigs={len(rig_ids)}, "
            f"fps={sorted(fps_values)}"
        )
    fps_target = next(iter(fps_values))
    tasks = [
        (str(root), species, fps_target, shard)
        for species in sorted(grouped)
        for shard in _chunks(grouped[species], int(shard_size))
    ]
    aggregates = {species: SpeciesAggregate(species) for species in sorted(grouped)}
    worker_count = max(1, int(workers))
    if worker_count == 1:
        results = map(_compute_shard, tasks)
        for index, partial in enumerate(results, start=1):
            aggregates[partial.species_id].merge(partial)
            if index % 50 == 0 or index == len(tasks):
                print(f"stats shards {index}/{len(tasks)}", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as pool:
            for index, partial in enumerate(
                pool.map(_compute_shard, tasks, chunksize=1), start=1
            ):
                aggregates[partial.species_id].merge(partial)
                if index % 50 == 0 or index == len(tasks):
                    print(f"stats shards {index}/{len(tasks)}", flush=True)
    return _write_outputs(
        output,
        generation=generation,
        generation_sha256=generation_sha256,
        aggregates=aggregates,
    )
