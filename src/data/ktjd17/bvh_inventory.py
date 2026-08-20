"""Small, dependency-free BVH hierarchy/header reader for KTJD inventory.

This is deliberately *not* the source motion parser used by the later KTJD
converter.  T02 only needs auditable hierarchy/channel evidence, native timing,
and frame counts.  Numeric Euler decoding and source-FK reproduction belong to
T03, so this module stops immediately after the BVH motion header.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class BvhInventoryError(ValueError):
    """A BVH cannot provide unambiguous inventory evidence."""


_DECL_RE = re.compile(r"^(ROOT|JOINT)\s+(.+?)\s*$", re.IGNORECASE)
_END_RE = re.compile(r"^End\s+Site(?:\s+#name:\s*(.+?))?\s*$", re.IGNORECASE)
_FRAMES_RE = re.compile(r"^Frames\s*:\s*(\d+)\s*$", re.IGNORECASE)
_FRAME_TIME_RE = re.compile(
    r"^Frame\s+Time\s*:\s*([^\s]+)\s*$", re.IGNORECASE
)


@dataclass(frozen=True)
class BvhJointHeader:
    """One hierarchy node exactly as declared by the BVH."""

    name: str
    parent: int
    node_kind: str
    offset: tuple[float, float, float]
    channels: tuple[str, ...]
    channels_declared: bool

    def rotation_source_kind(self) -> str:
        """Return the only two admissible KTJD lossless-v1 source kinds."""
        lowered = [channel.lower() for channel in self.channels]
        unknown = [
            channel
            for channel in lowered
            if not channel.endswith("rotation") and not channel.endswith("position")
        ]
        if unknown:
            raise BvhInventoryError(
                f"joint {self.name!r} has unsupported channels {unknown}"
            )

        rotation = [channel for channel in lowered if channel.endswith("rotation")]
        axes = [channel[0] for channel in rotation]
        if rotation:
            if len(rotation) != 3 or sorted(axes) != ["x", "y", "z"]:
                raise BvhInventoryError(
                    f"joint {self.name!r} does not expose one X/Y/Z rotation "
                    f"channel each: {self.channels}"
                )
            return "animated_dof"

        if self.node_kind == "end_site":
            if self.channels:
                raise BvhInventoryError(
                    f"end site {self.name!r} unexpectedly has channels"
                )
            return "fixed_dof"

        if self.channels_declared and not self.channels:
            return "fixed_dof"

        raise BvhInventoryError(
            f"joint {self.name!r} has no rotation channels and is not a "
            "source-declared fixed DOF"
        )


@dataclass(frozen=True)
class BvhHeader:
    """Hierarchy plus timing metadata, without loading motion samples."""

    path: str
    joints: tuple[BvhJointHeader, ...]
    frames: int
    frame_time: float
    channel_count: int

    @property
    def fps(self) -> float:
        return 1.0 / self.frame_time

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(joint.name for joint in self.joints)

    @property
    def parents(self) -> tuple[int, ...]:
        return tuple(joint.parent for joint in self.joints)

    def rotation_source_kinds(self) -> tuple[str, ...]:
        return tuple(joint.rotation_source_kind() for joint in self.joints)

    def rotation_layout_sha256(self) -> str:
        """Hash hierarchy/channel semantics while ignoring rest offsets."""
        payload = [
            {
                "name": joint.name,
                "parent": joint.parent,
                "node_kind": joint.node_kind,
                "channels": list(joint.channels),
                "channels_declared": joint.channels_declared,
            }
            for joint in self.joints
        ]
        raw = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def rest_layout_sha256(self) -> str:
        """Hash the hierarchy/channel semantics and declared rest offsets."""
        payload = [
            {
                "name": joint.name,
                "parent": joint.parent,
                "node_kind": joint.node_kind,
                "offset": list(joint.offset),
                "channels": list(joint.channels),
                "channels_declared": joint.channels_declared,
            }
            for joint in self.joints
        ]
        raw = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def compact_dict(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "frame_time": self.frame_time,
            "fps_src": self.fps,
            "source_joint_count": len(self.joints),
            "source_channel_count": self.channel_count,
            "rotation_layout_sha256": self.rotation_layout_sha256(),
            "rest_layout_sha256": self.rest_layout_sha256(),
        }


def _error(path: Path, line_number: int, message: str) -> BvhInventoryError:
    return BvhInventoryError(f"{path}:{line_number}: {message}")


def _parse_offset(path: Path, line_number: int, text: str) -> tuple[float, float, float]:
    fields = text.split()
    if len(fields) != 4:
        raise _error(path, line_number, f"malformed OFFSET line: {text!r}")
    try:
        values = tuple(float(value) for value in fields[1:])
    except ValueError as exc:
        raise _error(path, line_number, f"non-numeric OFFSET line: {text!r}") from exc
    if not all(math.isfinite(value) for value in values):
        raise _error(path, line_number, f"non-finite OFFSET line: {text!r}")
    return values  # type: ignore[return-value]


def parse_bvh_header(path: str | Path) -> BvhHeader:
    """Read one BVH through ``Frame Time`` and fail closed on ambiguity.

    Named End Sites emitted by the local AnyTop BVH tooling are preserved as
    physical fixed-DOF joints.  Unnamed End Sites receive deterministic names;
    they cannot accidentally match a named BTJD joint.
    """
    source = Path(path)
    if not source.is_file():
        raise BvhInventoryError(f"BVH source does not exist: {source}")

    mutable: list[dict[str, Any]] = []
    stack: list[int] = []
    pending: int | None = None
    unnamed_end_count: dict[int, int] = {}
    saw_hierarchy = False
    saw_motion = False
    motion_line = 0

    with source.open("r", encoding="utf-8", errors="strict") as handle:
        iterator = enumerate(handle, start=1)
        for line_number, raw_line in iterator:
            text = raw_line.strip()
            if not text:
                continue
            if not saw_hierarchy:
                if text.upper() != "HIERARCHY":
                    raise _error(source, line_number, "expected HIERARCHY")
                saw_hierarchy = True
                continue
            if text.upper() == "MOTION":
                if pending is not None or stack:
                    raise _error(source, line_number, "unclosed hierarchy before MOTION")
                saw_motion = True
                motion_line = line_number
                break

            declaration = _DECL_RE.match(text)
            if declaration:
                if pending is not None:
                    raise _error(source, line_number, "joint declaration missing opening brace")
                kind = declaration.group(1).lower()
                name = declaration.group(2).strip()
                if not name:
                    raise _error(source, line_number, "empty joint name")
                if kind == "root" and mutable:
                    raise _error(source, line_number, "multiple ROOT declarations")
                if kind == "joint" and not stack:
                    raise _error(source, line_number, "JOINT declared without a parent")
                mutable.append(
                    {
                        "name": name,
                        "parent": -1 if kind == "root" else stack[-1],
                        "node_kind": "joint",
                        "offset": None,
                        "channels": None,
                        "channels_declared": False,
                    }
                )
                pending = len(mutable) - 1
                continue

            end_site = _END_RE.match(text)
            if end_site:
                if pending is not None or not stack:
                    raise _error(source, line_number, "End Site has no unambiguous parent")
                parent = stack[-1]
                explicit_name = end_site.group(1)
                if explicit_name is None:
                    count = unnamed_end_count.get(parent, 0)
                    unnamed_end_count[parent] = count + 1
                    name = f"{mutable[parent]['name']}__unnamed_end_site_{count}"
                else:
                    name = explicit_name.strip()
                    if not name:
                        raise _error(source, line_number, "empty named End Site")
                mutable.append(
                    {
                        "name": name,
                        "parent": parent,
                        "node_kind": "end_site",
                        "offset": None,
                        "channels": (),
                        "channels_declared": True,
                    }
                )
                pending = len(mutable) - 1
                continue

            if text == "{":
                if pending is None:
                    raise _error(source, line_number, "opening brace without declaration")
                stack.append(pending)
                pending = None
                continue
            if text == "}":
                if pending is not None:
                    raise _error(source, line_number, "declaration missing opening brace")
                if not stack:
                    raise _error(source, line_number, "unmatched closing brace")
                stack.pop()
                continue
            if text.upper().startswith("OFFSET"):
                if not stack:
                    raise _error(source, line_number, "OFFSET outside a joint block")
                index = stack[-1]
                if mutable[index]["offset"] is not None:
                    raise _error(source, line_number, "duplicate OFFSET")
                mutable[index]["offset"] = _parse_offset(source, line_number, text)
                continue
            if text.upper().startswith("CHANNELS"):
                if not stack:
                    raise _error(source, line_number, "CHANNELS outside a joint block")
                index = stack[-1]
                if mutable[index]["node_kind"] == "end_site":
                    raise _error(source, line_number, "End Site cannot declare CHANNELS")
                if mutable[index]["channels_declared"]:
                    raise _error(source, line_number, "duplicate CHANNELS")
                fields = text.split()
                if len(fields) < 2:
                    raise _error(source, line_number, "malformed CHANNELS line")
                try:
                    count = int(fields[1])
                except ValueError as exc:
                    raise _error(source, line_number, "non-integer CHANNELS count") from exc
                channels = tuple(fields[2:])
                if count < 0 or count != len(channels):
                    raise _error(
                        source,
                        line_number,
                        f"CHANNELS count {count} does not match {len(channels)} labels",
                    )
                mutable[index]["channels"] = channels
                mutable[index]["channels_declared"] = True
                continue

            raise _error(source, line_number, f"unsupported hierarchy line {text!r}")

        if not saw_motion:
            raise _error(source, motion_line or 1, "missing MOTION section")

        frames: int | None = None
        frame_time: float | None = None
        for line_number, raw_line in iterator:
            text = raw_line.strip()
            if not text:
                continue
            if frames is None:
                match = _FRAMES_RE.match(text)
                if not match:
                    raise _error(source, line_number, "expected Frames: <int>")
                frames = int(match.group(1))
                if frames <= 0:
                    raise _error(source, line_number, "frame count must be positive")
                continue
            match = _FRAME_TIME_RE.match(text)
            if not match:
                raise _error(source, line_number, "expected Frame Time: <float>")
            try:
                frame_time = float(match.group(1))
            except ValueError as exc:
                raise _error(source, line_number, "non-numeric frame time") from exc
            if not math.isfinite(frame_time) or frame_time <= 0:
                raise _error(source, line_number, "frame time must be finite and positive")
            break

    if frames is None or frame_time is None:
        raise BvhInventoryError(f"{source}: incomplete motion header")
    if not mutable or mutable[0]["parent"] != -1:
        raise BvhInventoryError(f"{source}: hierarchy has no valid physical root")

    joints: list[BvhJointHeader] = []
    names: set[str] = set()
    for index, record in enumerate(mutable):
        if record["offset"] is None:
            raise BvhInventoryError(
                f"{source}: joint {record['name']!r} has no OFFSET"
            )
        if record["node_kind"] != "end_site" and not record["channels_declared"]:
            raise BvhInventoryError(
                f"{source}: joint {record['name']!r} has no CHANNELS declaration"
            )
        if record["name"] in names:
            raise BvhInventoryError(
                f"{source}: duplicate joint name {record['name']!r}"
            )
        names.add(record["name"])
        parent = int(record["parent"])
        if index == 0 and parent != -1:
            raise BvhInventoryError(f"{source}: first joint is not root")
        if index > 0 and not (0 <= parent < index):
            raise BvhInventoryError(
                f"{source}: joint {record['name']!r} violates parent-before-child order"
            )
        joints.append(
            BvhJointHeader(
                name=str(record["name"]),
                parent=parent,
                node_kind=str(record["node_kind"]),
                offset=record["offset"],
                channels=tuple(record["channels"] or ()),
                channels_declared=bool(record["channels_declared"]),
            )
        )

    header = BvhHeader(
        path=str(source.resolve()),
        joints=tuple(joints),
        frames=frames,
        frame_time=frame_time,
        channel_count=sum(len(joint.channels) for joint in joints),
    )
    # Force the binary provenance check while the file context is available.
    header.rotation_source_kinds()
    return header
