"""Perspective multi-frame visual QA for KTJD-17 source/direct/FK paths."""

from __future__ import annotations

import dataclasses
import datetime as _datetime
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .calibration import derive_position_anchor_heading
from .decoder import decode_ktjd17
from .encoder import load_skeleton
from .fixed_qa import reconstruct_aligned_source_reference
from .loader import load_motion_npz


VISUAL_QA_VERSION = "ktjd17-perspective-source-direct-fk-v2"
VISUAL_GENERATION_DIRECTORY = ".ktjd17_visual_qa_generations"
VISUAL_LINK_NAME = "ktjd17_visual_qa"
COORDINATE_CONTRACT = (
    "right-handed; +Y is screen-up; +Z points out of the screen toward the viewer"
)
_HISTORICAL_PARENT_MANIFEST_HASHES = {
    "20260819T145535975831Z-ed48b3fd2745": {
        "clips.jsonl": (
            "f7cd0d05ad2208924c43ede43f31a13d6ec893a2af92ec3229f344e76b23e9f3"
        ),
        "rigs.jsonl": (
            "108cb904684617e2bdcf24eaa80f9e5976d14a38acf56054b3996237e5cc5271"
        ),
    }
}


class VisualQaError(RuntimeError):
    """Visual-QA inputs or publication violate the immutable contract."""


@dataclasses.dataclass(frozen=True)
class PerspectiveCamera:
    center_x: float
    center_y: float
    camera_z: float
    focal_px: float

    def project(
        self, points: np.ndarray, *, width: int, height: int
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(points, dtype=np.float64)
        if values.shape[-1] != 3 or not np.isfinite(values).all():
            raise VisualQaError(f"invalid projection points {values.shape}")
        depth = self.camera_z - values[..., 2]
        if np.any(depth <= 0.0):
            raise VisualQaError("geometry crossed the +Z camera plane")
        x = width * 0.5 + self.focal_px * (values[..., 0] - self.center_x) / depth
        y = height * 0.54 - self.focal_px * (values[..., 1] - self.center_y) / depth
        return np.stack((x, y), axis=-1), depth

    def as_record(self) -> dict[str, Any]:
        return {
            "model": "pinhole_perspective",
            "position": [self.center_x, self.center_y, self.camera_z],
            "look_direction": "-Z",
            "screen_right": "+X",
            "screen_up": "+Y",
            "toward_viewer": "+Z",
            "center_x": self.center_x,
            "center_y": self.center_y,
            "camera_z": self.camera_z,
            "focal_px_for_310x300_panel": self.focal_px,
        }


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise VisualQaError(f"cannot read JSON {path}: {exc}") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise VisualQaError(f"{path}:{line_number}: blank JSONL row")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise VisualQaError(f"{path}:{line_number}: row is not an object")
                records.append(value)
    except VisualQaError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise VisualQaError(f"cannot read JSONL {path}: {exc}") from exc
    return records


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _resolve_generation_path(root: Path, relpath: Any, *, label: str) -> Path:
    if not isinstance(relpath, str) or not relpath:
        raise VisualQaError(f"{label}: invalid relative path")
    relative = Path(relpath)
    if relative.is_absolute():
        raise VisualQaError(f"{label}: absolute path is forbidden")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise VisualQaError(f"{label}: path escapes prototype generation")
    return resolved


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


def _safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    return f"{cleaned[:72]}-{digest}"


def _select_diagnostic_frames(
    frame_errors: np.ndarray, *, frame_count: int, count: int = 6
) -> np.ndarray:
    """Mix temporal coverage with separated worst heading-disagreement frames."""
    if frame_count <= 0 or count <= 0:
        raise VisualQaError("diagnostic frame selection requires positive extents")
    target = min(int(count), int(frame_count))
    errors = np.asarray(frame_errors, dtype=np.float64)
    if errors.shape != (frame_count,):
        raise VisualQaError("diagnostic frame errors must have shape [T]")
    selected: set[int] = {0, frame_count - 1}
    separation = max(1, frame_count // (2 * target))
    finite_indices = np.flatnonzero(np.isfinite(errors))
    ordered = finite_indices[np.argsort(errors[finite_indices])[::-1]]
    for index in ordered:
        value = int(index)
        if all(abs(value - existing) >= separation for existing in selected):
            selected.add(value)
        if len(selected) >= target:
            break
    uniform = np.rint(np.linspace(0, frame_count - 1, target)).astype(np.int64)
    for index in uniform:
        selected.add(int(index))
        if len(selected) >= target:
            break
    if len(selected) < target:
        for index in range(frame_count):
            selected.add(index)
            if len(selected) >= target:
                break
    ranked = sorted(
        selected,
        key=lambda index: (
            0 if index in {0, frame_count - 1} else 1,
            -float(errors[index]) if np.isfinite(errors[index]) else math.inf,
            index,
        ),
    )[:target]
    return np.asarray(sorted(ranked), dtype=np.int64)


def _fit_camera(
    point_sets: Sequence[np.ndarray], *, panel_width: int, panel_height: int, s_rig: float
) -> PerspectiveCamera:
    points = np.concatenate(
        [np.asarray(values, dtype=np.float64).reshape(-1, 3) for values in point_sets],
        axis=0,
    )
    if points.size == 0 or not np.isfinite(points).all():
        raise VisualQaError("cannot fit camera to empty/non-finite geometry")
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    center_x = float((minimum[0] + maximum[0]) * 0.5)
    center_y = float((minimum[1] + maximum[1]) * 0.5)
    span = max(float(np.max(maximum - minimum)), float(s_rig), 1e-6)
    camera_z = float(maximum[2] + 2.5 * span)
    depth = camera_z - points[:, 2]
    normalized_x = np.abs((points[:, 0] - center_x) / depth)
    normalized_y = np.abs((points[:, 1] - center_y) / depth)
    max_x = max(float(np.max(normalized_x)), 1e-9)
    max_y = max(float(np.max(normalized_y)), 1e-9)
    focal = min(
        0.43 * panel_width / max_x,
        0.40 * panel_height / max_y,
        4.0 * panel_width,
    )
    return PerspectiveCamera(
        center_x=center_x,
        center_y=center_y,
        camera_z=camera_z,
        focal_px=float(focal),
    )


def _xy(points: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(int(round(value)) for value in point) for point in points]


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: Sequence[float],
    end: Sequence[float],
    *,
    color: tuple[int, int, int],
    width: int = 3,
) -> None:
    p0 = np.asarray(start, dtype=np.float64)
    p1 = np.asarray(end, dtype=np.float64)
    draw.line([tuple(p0), tuple(p1)], fill=color, width=width)
    direction = p1 - p0
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        return
    unit = direction / norm
    normal = np.asarray([-unit[1], unit[0]])
    tip1 = p1 - 9.0 * unit + 4.5 * normal
    tip2 = p1 - 9.0 * unit - 4.5 * normal
    draw.polygon([tuple(p1), tuple(tip1), tuple(tip2)], fill=color)


def _draw_heading_compass(
    draw: ImageDraw.ImageDraw,
    *,
    rotation_heading: np.ndarray,
    rotation_valid: bool,
    position_heading: np.ndarray,
    position_valid: bool,
    width: int,
) -> None:
    center = np.asarray([width - 40.0, 77.0])
    radius = 25.0
    draw.ellipse(
        (
            int(center[0] - radius),
            int(center[1] - radius),
            int(center[0] + radius),
            int(center[1] + radius),
        ),
        outline=(105, 115, 132),
        width=1,
    )
    draw.line(
        [(center[0] - radius, center[1]), (center[0] + radius, center[1])],
        fill=(235, 78, 78),
        width=1,
    )
    draw.line(
        [(center[0], center[1] - radius), (center[0], center[1] + radius)],
        fill=(80, 145, 255),
        width=1,
    )
    draw.text((int(center[0] + radius - 8), int(center[1] + 2)), "+X", fill=(235, 78, 78), font=_font(9))
    draw.text((int(center[0] + 2), int(center[1] - radius - 10)), "+Z", fill=(80, 145, 255), font=_font(9))
    draw.text((int(center[0] - radius), int(center[1] + radius + 2)), "heading XZ", fill=(180, 188, 202), font=_font(8))
    for heading, valid, color in (
        (rotation_heading, rotation_valid, (255, 88, 212)),
        (position_heading, position_valid, (255, 158, 52)),
    ):
        if not valid:
            continue
        endpoint = center + radius * 0.82 * np.asarray(
            [float(heading[1]), -float(heading[0])]
        )
        _draw_arrow(draw, center, endpoint, color=color, width=2)


def _nice_grid_step(extent: float) -> float:
    raw = max(float(extent) / 8.0, 1e-6)
    power = 10.0 ** math.floor(math.log10(raw))
    scaled = raw / power
    factor = 1.0 if scaled <= 1.0 else 2.0 if scaled <= 2.0 else 5.0
    return factor * power


def _draw_ground_and_axes(
    draw: ImageDraw.ImageDraw,
    camera: PerspectiveCamera,
    *,
    width: int,
    height: int,
    bounds: tuple[float, float, float, float],
    axis_length: float,
) -> None:
    x_min, x_max, z_min, z_max = bounds
    step = _nice_grid_step(max(x_max - x_min, z_max - z_min))
    x_values = np.arange(math.floor(x_min / step) * step, x_max + step, step)
    z_values = np.arange(math.floor(z_min / step) * step, z_max + step, step)
    for x in x_values:
        line = np.asarray([[x, 0.0, z_min], [x, 0.0, z_max]], dtype=np.float64)
        projected, _ = camera.project(line, width=width, height=height)
        draw.line(_xy(projected), fill=(50, 58, 70), width=1)
    for z in z_values:
        line = np.asarray([[x_min, 0.0, z], [x_max, 0.0, z]], dtype=np.float64)
        projected, _ = camera.project(line, width=width, height=height)
        draw.line(_xy(projected), fill=(50, 58, 70), width=1)
    origin = np.asarray([0.0, 0.0, 0.0])
    axes = {
        "+X": (np.asarray([axis_length, 0.0, 0.0]), (235, 78, 78)),
        "+Y": (np.asarray([0.0, axis_length, 0.0]), (73, 205, 110)),
    }
    origin_2d, _ = camera.project(origin[None], width=width, height=height)
    for label, (endpoint, color) in axes.items():
        endpoint_2d, _ = camera.project(endpoint[None], width=width, height=height)
        _draw_arrow(draw, origin_2d[0], endpoint_2d[0], color=color, width=2)
        draw.text(tuple(endpoint_2d[0] + np.asarray([3.0, -10.0])), label, fill=color, font=_font(11))
    ox, oy = (int(round(value)) for value in origin_2d[0])
    draw.ellipse((ox - 5, oy - 5, ox + 5, oy + 5), outline=(80, 145, 255), width=2)
    draw.ellipse((ox - 1, oy - 1, ox + 1, oy + 1), fill=(80, 145, 255))
    draw.text((ox + 7, oy + 3), "+Z toward viewer", fill=(80, 145, 255), font=_font(10))


def _draw_panel(
    *,
    positions: np.ndarray,
    all_positions: np.ndarray,
    smooth_root_xz: np.ndarray,
    parents: np.ndarray,
    contact: np.ndarray | None,
    heading: np.ndarray,
    heading_valid: bool,
    position_heading: np.ndarray,
    position_heading_valid: bool,
    anchor_indices: Sequence[int],
    frame_index: int,
    title: str,
    camera: PerspectiveCamera,
    width: int,
    height: int,
    bounds: tuple[float, float, float, float],
    s_rig: float,
    metrics_line: str,
) -> Image.Image:
    image = Image.new("RGB", (width, height), (17, 21, 29))
    draw = ImageDraw.Draw(image)
    _draw_ground_and_axes(
        draw,
        camera,
        width=width,
        height=height,
        bounds=bounds,
        axis_length=0.30 * s_rig,
    )
    root_path = all_positions[:, 0]
    root_2d, _ = camera.project(root_path, width=width, height=height)
    if len(root_2d) >= 2:
        draw.line(_xy(root_2d), fill=(165, 165, 165), width=2)
    smooth_path = np.zeros((len(smooth_root_xz), 3), dtype=np.float64)
    smooth_path[:, 0] = smooth_root_xz[:, 0]
    smooth_path[:, 2] = smooth_root_xz[:, 1]
    smooth_2d, _ = camera.project(smooth_path, width=width, height=height)
    if len(smooth_2d) >= 2:
        draw.line(_xy(smooth_2d), fill=(30, 215, 225), width=2)
    joints_2d, _ = camera.project(positions, width=width, height=height)
    for child in range(1, len(parents)):
        parent = int(parents[child])
        draw.line(
            [tuple(joints_2d[parent]), tuple(joints_2d[child])],
            fill=(228, 232, 239),
            width=2,
        )
    for joint, point in enumerate(joints_2d):
        active_contact = bool(contact[joint]) if contact is not None else False
        radius = 4 if joint == 0 else 3
        color = (255, 190, 45) if active_contact else (116, 192, 255)
        x, y = (int(round(value)) for value in point)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        if joint in anchor_indices:
            draw.ellipse(
                (x - radius - 3, y - radius - 3, x + radius + 3, y + radius + 3),
                outline=(255, 158, 52),
                width=2,
            )
    root = positions[0]
    root_2d_current, _ = camera.project(root[None], width=width, height=height)
    if heading_valid:
        direction = np.asarray([heading[1], 0.0, heading[0]], dtype=np.float64)
        endpoint = root + 0.45 * s_rig * direction
        endpoint_2d, _ = camera.project(endpoint[None], width=width, height=height)
        _draw_arrow(
            draw,
            root_2d_current[0],
            endpoint_2d[0],
            color=(255, 88, 212),
            width=3,
        )
    else:
        x, y = (int(round(value)) for value in root_2d_current[0])
        draw.line([(x - 7, y - 7), (x + 7, y + 7)], fill=(255, 72, 72), width=3)
        draw.line([(x - 7, y + 7), (x + 7, y - 7)], fill=(255, 72, 72), width=3)
        draw.text((x + 8, y - 8), "heading invalid", fill=(255, 72, 72), font=_font(10))
    if position_heading_valid:
        direction = np.asarray(
            [position_heading[1], 0.0, position_heading[0]], dtype=np.float64
        )
        endpoint = root + 0.36 * s_rig * direction
        endpoint_2d, _ = camera.project(endpoint[None], width=width, height=height)
        _draw_arrow(
            draw,
            root_2d_current[0],
            endpoint_2d[0],
            color=(255, 158, 52),
            width=2,
        )
    _draw_heading_compass(
        draw,
        rotation_heading=heading,
        rotation_valid=heading_valid,
        position_heading=position_heading,
        position_valid=position_heading_valid,
        width=width,
    )
    draw.rectangle((0, 0, width, 36), fill=(9, 12, 18))
    draw.text((8, 5), title, fill=(245, 245, 245), font=_font(15))
    draw.text((8, 23), metrics_line, fill=(185, 195, 210), font=_font(9))
    draw.text(
        (8, height - 15),
        f"frame {frame_index} root=gray smooth=cyan contact=gold rot-H=magenta pos-H=orange",
        fill=(176, 184, 198),
        font=_font(9),
    )
    return image


def _draw_rest_panel(
    *,
    rest: np.ndarray,
    parents: np.ndarray,
    camera: PerspectiveCamera,
    width: int,
    height: int,
    bounds: tuple[float, float, float, float],
    s_rig: float,
    rig_id: str,
    anchor_indices: Sequence[int],
) -> Image.Image:
    heading = np.asarray([1.0, 0.0])
    return _draw_panel(
        positions=rest,
        all_positions=rest[None],
        smooth_root_xz=np.asarray([[rest[0, 0], rest[0, 2]]]),
        parents=parents,
        contact=None,
        heading=heading,
        heading_valid=True,
        position_heading=heading,
        position_heading_valid=True,
        anchor_indices=anchor_indices,
        frame_index=0,
        title=f"canonical rest | {rig_id}",
        camera=camera,
        width=width,
        height=height,
        bounds=bounds,
        s_rig=s_rig,
        metrics_line="rest forward +Z | physical joints only",
    )


def _select_default_clips(
    manifests: Sequence[Mapping[str, Any]],
    calibration_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    manifest_by_id = {str(record["clip_id"]): record for record in manifests}
    selected: list[dict[str, str]] = []
    for role in (
        "human",
        "quadruped",
        "winged",
        "spider_crab",
        "dragon_or_deep_topology",
    ):
        candidates = [
            record for record in calibration_records if record["family_role"] == role
        ]
        if not candidates:
            raise VisualQaError(f"no train calibration visual candidate for {role}")
        chosen = max(
            candidates,
            key=lambda record: float(
                record["metrics"]["position_anchor_heading_circular_p99_rad"]
            ),
        )
        selected.append(
            {
                "clip_id": str(chosen["clip_id"]),
                "visual_role": role,
                "selection_reason": "worst_train_position_anchor_heading_p99",
            }
        )
    preferences = {
        "snake": "KingCobra___CircleBite_502",
        "dragon_exact": "Dragon___Fly_297",
    }
    for role, preferred in preferences.items():
        candidates = [
            record for record in manifests if record.get("family_role") == role
        ]
        if not candidates:
            raise VisualQaError(f"no held read-only visual candidate for {role}")
        clip_id = preferred if preferred in manifest_by_id else str(candidates[0]["clip_id"])
        selected.append(
            {
                "clip_id": clip_id,
                "visual_role": role,
                "selection_reason": "declared_held_read_only_representative",
            }
        )
    return selected


def _file_manifest(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise VisualQaError(
                f"symlink is forbidden inside visual generation: {path}"
            )
        if path.is_file():
            relpath = path.relative_to(root).as_posix()
            if relpath == "generation.json":
                continue
            result[relpath] = {
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    return result


def verify_visual_generation(root: str | Path) -> dict[str, Any]:
    """Verify immutable visual-generation file closure, hashes, and sizes."""
    generation_root = Path(root).expanduser().resolve()
    generation = _load_json(generation_root / "generation.json")
    expected = generation.get("files")
    if not isinstance(expected, dict):
        raise VisualQaError("visual generation files map is absent")
    observed = _file_manifest(generation_root)
    if set(observed) != set(expected):
        raise VisualQaError(
            "visual generation file closure failed: "
            f"missing={sorted(set(expected) - set(observed))}, "
            f"extra={sorted(set(observed) - set(expected))}"
        )
    for relpath, metadata in expected.items():
        if not isinstance(relpath, str) or not relpath:
            raise VisualQaError("visual generation contains an invalid file path")
        relative = Path(relpath)
        if relative.is_absolute() or ".." in relative.parts:
            raise VisualQaError(f"visual generation path escapes root: {relpath}")
        if not isinstance(metadata, Mapping):
            raise VisualQaError(f"invalid visual file metadata: {relpath}")
        if observed[relpath] != dict(metadata):
            raise VisualQaError(f"visual generation hash/size drift: {relpath}")
    return generation


def _verify_input_generation(root: Path, *, label: str) -> dict[str, Any]:
    """Require a closed immutable producer generation before rendering it."""
    try:
        generation = _load_json(root / "generation.json")
        expected = generation.get("files")
        if not isinstance(expected, dict):
            raise VisualQaError(f"{label} generation files map is absent")
        observed = _file_manifest(root)
        if set(observed) != set(expected):
            raise VisualQaError(
                f"{label} generation file closure failed: "
                f"missing={sorted(set(expected) - set(observed))}, "
                f"extra={sorted(set(observed) - set(expected))}"
            )
        for relpath, metadata in expected.items():
            if not isinstance(metadata, Mapping) or observed[relpath] != dict(metadata):
                raise VisualQaError(
                    f"{label} generation hash/size drift: {relpath}"
                )
        return generation
    except VisualQaError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise VisualQaError(f"cannot verify {label} generation: {exc}") from exc


def verify_parent_manifest_authority(
    selection_record: Mapping[str, Any],
) -> tuple[Path, dict[str, str]]:
    """Resolve and hash-check both parent inputs used by the source route."""
    authority = selection_record.get("selection_authority")
    if not isinstance(authority, Mapping):
        raise VisualQaError("selection authority is absent")
    parent_value = authority.get("parent_manifest_root")
    if not isinstance(parent_value, str) or not parent_value:
        raise VisualQaError("parent manifest root is absent")
    parent_root = Path(parent_value).expanduser().resolve()
    if not parent_root.is_dir():
        raise VisualQaError(f"parent manifest root is unavailable: {parent_root}")
    verified: dict[str, str] = {}
    historical = _HISTORICAL_PARENT_MANIFEST_HASHES.get(parent_root.name, {})
    for filename, field in (
        ("clips.jsonl", "parent_clips_jsonl_sha256"),
        ("rigs.jsonl", "parent_rigs_jsonl_sha256"),
    ):
        expected = authority.get(field)
        if expected is None:
            expected = historical.get(filename)
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise VisualQaError(f"selection authority lacks valid {field}")
        path = parent_root / filename
        if path.is_symlink() or not path.is_file():
            raise VisualQaError(f"parent manifest input is not a regular file: {path}")
        observed = _sha256_file(path)
        if observed != expected:
            raise VisualQaError(
                f"parent manifest hash drift: {filename}: {observed} != {expected}"
            )
        verified[filename] = observed
    return parent_root, verified


def _replace_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() and not link.is_symlink():
        raise VisualQaError(f"refusing to replace non-symlink {link}")
    temporary = link.parent / f".{link.name}.{uuid.uuid4().hex}.tmp"
    os.symlink(os.path.relpath(target, start=link.parent), temporary)
    os.replace(temporary, link)
    _fsync_directory(link.parent)


def render_prototype_visual_qa(
    *,
    prototype_root: str | Path,
    calibration_root: str | Path | None,
    output_root: str | Path,
    clip_ids: Sequence[str] | None = None,
    max_gif_frames: int = 36,
    update_link: bool = True,
) -> dict[str, Any]:
    """Render synchronized source/direct/FK animations without frame recentering."""
    root = Path(prototype_root).expanduser().resolve()
    output = Path(output_root).expanduser().absolute()
    prototype_generation = _verify_input_generation(root, label="prototype")
    if calibration_root is None:
        if not clip_ids:
            raise VisualQaError(
                "calibration_root may be omitted only with explicit clip_ids"
            )
        calibration_generation = {
            "generation_id": "not_applicable_explicit_visual_audit",
            "prototype_generation_id": prototype_generation.get("generation_id"),
        }
        calibration_records: list[dict[str, Any]] = []
    else:
        calibration = Path(calibration_root).expanduser().resolve()
        calibration_generation = _verify_input_generation(
            calibration, label="calibration"
        )
        if (
            calibration_generation.get("prototype_generation_id")
            != prototype_generation.get("generation_id")
        ):
            raise VisualQaError("calibration/prototype generation mismatch")
        calibration_records = _load_jsonl(calibration / "train_clip_metrics.jsonl")
    manifests = _load_jsonl(root / "manifests/clips.jsonl")
    manifest_by_id = {str(record["clip_id"]): record for record in manifests}
    if len(manifest_by_id) != len(manifests):
        raise VisualQaError("duplicate prototype manifest clip ids")
    if clip_ids:
        selection = [
            {
                "clip_id": str(clip_id),
                "visual_role": str(manifest_by_id[str(clip_id)]["family_role"]),
                "selection_reason": "explicit_cli_selection",
            }
            for clip_id in clip_ids
        ]
    else:
        selection = _select_default_clips(manifests, calibration_records)
    if len({item["clip_id"] for item in selection}) != len(selection):
        raise VisualQaError("visual selection contains duplicate clip ids")
    selection_authority = _load_json(root / "manifests/prototype_selection.json")
    parent_root, verified_parent_hashes = verify_parent_manifest_authority(
        selection_authority
    )
    parent_clip_records = _load_jsonl(parent_root / "clips.jsonl")
    parent_rig_records = _load_jsonl(parent_root / "rigs.jsonl")
    parent_clips = {record["clip_id"]: record for record in parent_clip_records}
    parent_rigs = {record["rig_id"]: record for record in parent_rig_records}
    if len(parent_clips) != len(parent_clip_records):
        raise VisualQaError("duplicate parent manifest clip ids")
    if len(parent_rigs) != len(parent_rig_records):
        raise VisualQaError("duplicate parent manifest rig ids")
    post_load_parent_root, post_load_hashes = verify_parent_manifest_authority(
        selection_authority
    )
    if (
        post_load_parent_root != parent_root
        or post_load_hashes != verified_parent_hashes
    ):
        raise VisualQaError("parent manifest authority changed while loading")
    encoder_config = _load_json(root / "config/encoder_candidate.json")
    generation_id = (
        _datetime.datetime.now(_datetime.UTC).strftime("%Y%m%dT%H%M%S%fZ")
        + "-"
        + uuid.uuid4().hex[:12]
    )
    generations = output / VISUAL_GENERATION_DIRECTORY
    generations.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{generation_id}.", dir=generations))
    final = generations / generation_id
    index_records: list[dict[str, Any]] = []
    rest_cache: dict[str, Any] = {}
    try:
        for item in selection:
            clip_id = item["clip_id"]
            if clip_id not in manifest_by_id or clip_id not in parent_clips:
                raise VisualQaError(f"unknown selected clip {clip_id}")
            manifest = manifest_by_id[clip_id]
            motion_path = _resolve_generation_path(
                root, manifest["motion_relpath"], label=f"{clip_id}: motion"
            )
            skeleton_path = _resolve_generation_path(
                root, manifest["skeleton_relpath"], label=f"{clip_id}: skeleton"
            )
            if _sha256_file(motion_path) != manifest["motion_sha256"]:
                raise VisualQaError(f"{clip_id}: motion hash drifted")
            if _sha256_file(skeleton_path) != manifest["skeleton_sha256"]:
                raise VisualQaError(f"{clip_id}: skeleton hash drifted")
            skeleton = load_skeleton(skeleton_path)
            payload = load_motion_npz(
                motion_path,
                expected_fps_target=float(encoder_config["fps_target"]),
            )
            motion = np.asarray(payload["motion"], dtype=np.float64)
            heading_valid = np.asarray(payload["heading_valid"], dtype=bool)
            decoded = decode_ktjd17(
                motion,
                parents=skeleton.parents,
                R_rest_global=skeleton.R_rest_global,
                R_rest_local=skeleton.R_rest_local,
                offset_parent_local=skeleton.offset_parent_local,
                rotation_source_kind=skeleton.rotation_source_kind,
                strict_gt=True,
            )
            heading_provenance = skeleton.metadata.get(
                "heading_payload_provenance"
            )
            if not isinstance(heading_provenance, Mapping):
                raise VisualQaError(f"{clip_id}: heading provenance is absent")
            anchor_indices = tuple(
                int(value)
                for value in heading_provenance["forward_anchor_indices"]
            )
            anchor_names = tuple(
                str(value) for value in heading_provenance["forward_anchor_names"]
            )
            if tuple(skeleton.joint_names[index] for index in anchor_indices) != anchor_names:
                raise VisualQaError(f"{clip_id}: heading anchor mapping drifted")
            position_headings, position_heading_valid, _ = (
                derive_position_anchor_heading(
                    decoded.positions_direct,
                    method=str(heading_provenance["forward_method"]),
                    anchor_indices=anchor_indices,
                    s_rig=skeleton.s_rig,
                )
            )
            stored_headings = motion[:, 0, 15:17]
            heading_compare = heading_valid & position_heading_valid
            heading_frame_error = np.full(motion.shape[0], np.nan, dtype=np.float64)
            heading_cross = (
                stored_headings[heading_compare, 0]
                * position_headings[heading_compare, 1]
                - stored_headings[heading_compare, 1]
                * position_headings[heading_compare, 0]
            )
            heading_dot = np.sum(
                stored_headings[heading_compare]
                * position_headings[heading_compare],
                axis=-1,
            )
            heading_frame_error[heading_compare] = np.abs(
                np.arctan2(heading_cross, heading_dot)
            )
            source = reconstruct_aligned_source_reference(
                parent_clip=parent_clips[clip_id],
                parent_rig=parent_rigs[str(manifest["rig_id"])],
                skeleton=skeleton,
                fps_target=float(payload["fps_target"]),
                origin_xz=np.asarray(payload["origin_xz"], dtype=np.float64),
                rest_cache=rest_cache,
            )
            if source.positions_clip.shape != decoded.positions_direct.shape:
                raise VisualQaError(
                    f"{clip_id}: source/direct shape mismatch "
                    f"{source.positions_clip.shape} != {decoded.positions_direct.shape}"
                )
            source_error = np.linalg.norm(
                source.positions_clip - decoded.positions_direct, axis=-1
            ) / skeleton.s_rig
            fk_error = np.linalg.norm(
                source.positions_clip - decoded.positions_fk, axis=-1
            ) / skeleton.s_rig
            paths = {
                "source": source.positions_clip,
                "position-direct": decoded.positions_direct,
                "rotation-FK": decoded.positions_fk,
            }
            point_sets = list(paths.values()) + [skeleton.P_rest_global[None]]
            panel_width, panel_height = 310, 300
            camera = _fit_camera(
                point_sets,
                panel_width=panel_width,
                panel_height=panel_height,
                s_rig=skeleton.s_rig,
            )
            all_points = np.concatenate(
                [values.reshape(-1, 3) for values in point_sets], axis=0
            )
            x_margin = max(0.25 * skeleton.s_rig, 0.05 * np.ptp(all_points[:, 0]))
            z_margin = max(0.25 * skeleton.s_rig, 0.05 * np.ptp(all_points[:, 2]))
            bounds = (
                float(np.min(all_points[:, 0]) - x_margin),
                float(np.max(all_points[:, 0]) + x_margin),
                float(np.min(all_points[:, 2]) - z_margin),
                float(np.max(all_points[:, 2]) + z_margin),
            )
            frame_count = motion.shape[0]
            animation_frames = np.unique(
                np.rint(
                    np.linspace(0, frame_count - 1, min(max_gif_frames, frame_count))
                ).astype(np.int64)
            )
            smooth_root = motion[:, 0, 13:15]
            headings = stored_headings
            contacts = motion[..., 12].astype(bool)
            animation_images: list[Image.Image] = []
            for frame in animation_frames:
                panels: list[Image.Image] = []
                metric_lines = {
                    "source": "independent raw-source route",
                    "position-direct": f"source max={np.max(source_error):.2e} s_rig",
                    "rotation-FK": f"source max={np.max(fk_error):.2e} s_rig",
                }
                frame_heading_error = heading_frame_error[int(frame)]
                heading_suffix = (
                    f" | Hdiff={math.degrees(frame_heading_error):.1f}deg"
                    if np.isfinite(frame_heading_error)
                    else " | Hdiff=n/a"
                )
                for title, positions in paths.items():
                    panels.append(
                        _draw_panel(
                            positions=positions[int(frame)],
                            all_positions=positions,
                            smooth_root_xz=smooth_root,
                            parents=skeleton.parents,
                            contact=contacts[int(frame)],
                            heading=headings[int(frame)],
                            heading_valid=bool(heading_valid[int(frame)]),
                            position_heading=position_headings[int(frame)],
                            position_heading_valid=bool(
                                position_heading_valid[int(frame)]
                            ),
                            anchor_indices=anchor_indices,
                            frame_index=int(frame),
                            title=title,
                            camera=camera,
                            width=panel_width,
                            height=panel_height,
                            bounds=bounds,
                            s_rig=skeleton.s_rig,
                            metrics_line=metric_lines[title] + heading_suffix,
                        )
                    )
                panels.append(
                    _draw_rest_panel(
                        rest=skeleton.P_rest_global,
                        parents=skeleton.parents,
                        camera=camera,
                        width=panel_width,
                        height=panel_height,
                        bounds=bounds,
                        s_rig=skeleton.s_rig,
                        rig_id=skeleton.rig_id,
                        anchor_indices=anchor_indices,
                    )
                )
                canvas = Image.new(
                    "RGB", (panel_width * 4, panel_height + 42), (5, 8, 13)
                )
                header = ImageDraw.Draw(canvas)
                header.text(
                    (10, 6),
                    f"{clip_id} | {COORDINATE_CONTRACT} | fixed camera, no frame recenter",
                    fill=(245, 245, 245),
                    font=_font(14),
                )
                for index, panel in enumerate(panels):
                    canvas.paste(panel, (index * panel_width, 42))
                animation_images.append(canvas)
            safe = _safe_stem(clip_id)
            gif_relpath = f"clips/{safe}.gif"
            filmstrip_relpath = f"clips/{safe}_filmstrip.png"
            rest_relpath = f"clips/{safe}_rest.png"
            gif_path = staging / gif_relpath
            filmstrip_path = staging / filmstrip_relpath
            rest_path = staging / rest_relpath
            gif_path.parent.mkdir(parents=True, exist_ok=True)
            if len(animation_frames) >= 2:
                frame_stride = float(
                    np.mean(np.diff(animation_frames.astype(np.float64)))
                )
            else:
                frame_stride = 1.0
            duration_ms = max(
                20, int(round(1000.0 * frame_stride / float(payload["fps_target"])))
            )
            animation_images[0].save(
                gif_path,
                save_all=True,
                append_images=animation_images[1:],
                duration=duration_ms,
                loop=0,
                optimize=False,
            )
            film_frames = _select_diagnostic_frames(
                heading_frame_error,
                frame_count=frame_count,
                count=min(6, frame_count),
            )
            cell_width, cell_height = 310, 250
            filmstrip = Image.new(
                "RGB",
                (cell_width * len(film_frames), cell_height * 3 + 48),
                (5, 8, 13),
            )
            film_draw = ImageDraw.Draw(filmstrip)
            film_draw.text(
                (10, 7),
                f"{clip_id} | synchronized time columns | {COORDINATE_CONTRACT}",
                fill=(245, 245, 245),
                font=_font(14),
            )
            for row, (title, positions) in enumerate(paths.items()):
                for column, frame in enumerate(film_frames):
                    panel = _draw_panel(
                        positions=positions[int(frame)],
                        all_positions=positions,
                        smooth_root_xz=smooth_root,
                        parents=skeleton.parents,
                        contact=contacts[int(frame)],
                        heading=headings[int(frame)],
                        heading_valid=bool(heading_valid[int(frame)]),
                        position_heading=position_headings[int(frame)],
                        position_heading_valid=bool(
                            position_heading_valid[int(frame)]
                        ),
                        anchor_indices=anchor_indices,
                        frame_index=int(frame),
                        title=title,
                        camera=camera,
                        width=cell_width,
                        height=cell_height,
                        bounds=bounds,
                        s_rig=skeleton.s_rig,
                        metrics_line=(
                            "independent raw-source route"
                            if title == "source"
                            else f"source max={np.max(source_error if title == 'position-direct' else fk_error):.2e} s_rig"
                        )
                        + (
                            f" | Hdiff={math.degrees(heading_frame_error[int(frame)]):.1f}deg"
                            if np.isfinite(heading_frame_error[int(frame)])
                            else " | Hdiff=n/a"
                        ),
                    )
                    filmstrip.paste(
                        panel, (column * cell_width, 48 + row * cell_height)
                    )
            filmstrip.save(filmstrip_path)
            rest_image = _draw_rest_panel(
                rest=skeleton.P_rest_global,
                parents=skeleton.parents,
                camera=camera,
                width=620,
                height=520,
                bounds=bounds,
                s_rig=skeleton.s_rig,
                rig_id=skeleton.rig_id,
                anchor_indices=anchor_indices,
            )
            rest_image.save(rest_path)
            for image in animation_images:
                image.close()
            index_records.append(
                {
                    **item,
                    "rig_id": skeleton.rig_id,
                    "source_family": manifest["source_family"],
                    "topology_family": manifest["topology_family"],
                    "topology_distance_bucket": manifest[
                        "topology_distance_bucket"
                    ],
                    "split": manifest["split"],
                    "calibration_eligible": bool(manifest["calibration_eligible"]),
                    "held_read_only": not bool(manifest["calibration_eligible"]),
                    "T": int(frame_count),
                    "J": int(motion.shape[1]),
                    "fps_target": float(payload["fps_target"]),
                    "animation_frame_indices": animation_frames.tolist(),
                    "filmstrip_frame_indices": film_frames.tolist(),
                    "gif_frame_duration_ms": duration_ms,
                    "camera": camera.as_record(),
                    "fixed_camera_across_frames_and_paths": True,
                    "frame_recenter_applied": False,
                    "ground_changed": False,
                    "face_direction_changed": False,
                    "coordinate_contract": COORDINATE_CONTRACT,
                    "source_direct_max_norm": float(np.max(source_error)),
                    "source_fk_max_norm": float(np.max(fk_error)),
                    "heading_invalid_frame_count": int(np.count_nonzero(~heading_valid)),
                    "position_heading_invalid_frame_count": int(
                        np.count_nonzero(~position_heading_valid)
                    ),
                    "position_anchor_heading_circular_median_rad": float(
                        np.nanmedian(heading_frame_error)
                    ),
                    "position_anchor_heading_circular_p99_rad": float(
                        np.nanpercentile(heading_frame_error, 99)
                    ),
                    "position_anchor_heading_circular_max_rad": float(
                        np.nanmax(heading_frame_error)
                    ),
                    "heading_anchor_method": str(
                        heading_provenance["forward_method"]
                    ),
                    "heading_anchor_names": list(anchor_names),
                    "contact_positive_rate": float(np.mean(contacts)),
                    "gif_relpath": gif_relpath,
                    "filmstrip_relpath": filmstrip_relpath,
                    "rest_relpath": rest_relpath,
                    "inspection_status": "pending_human_and_codex_visual_review",
                }
            )
            print(
                f"[ktjd17-visual] rendered {item['visual_role']}: {clip_id}",
                flush=True,
            )
        visual_index = {
            "visual_qa_version": VISUAL_QA_VERSION,
            "status": "pending_human_and_codex_visual_review",
            "prototype_generation_id": prototype_generation["generation_id"],
            "calibration_generation_id": calibration_generation["generation_id"],
            "source_plan_commit": prototype_generation["source_plan_commit"],
            "coordinate_contract": COORDINATE_CONTRACT,
            "required_paths": ["source", "position-direct", "rotation-FK"],
            "perspective_camera": True,
            "fixed_camera_across_frames_and_paths": True,
            "frame_recenter_applied": False,
            "ground_changed": False,
            "face_direction_changed": False,
            "verified_parent_manifest_root": str(parent_root),
            "verified_parent_manifest_hashes": verified_parent_hashes,
            "legend": {
                "root_trajectory": "gray",
                "smooth_root_trajectory": "cyan",
                "contact_joint": "gold",
                "rotation_heading": "magenta arrow and XZ compass needle",
                "position_anchor_heading": "orange arrow, anchor rings, and XZ compass needle",
                "invalid_heading": "red X",
                "rest_forward": "+Z",
            },
            "clips": index_records,
            "freeze_authorized": False,
            "full_conversion_authorized": False,
        }
        _write_json(staging / "visual_qa_index.json", visual_index)
        files = _file_manifest(staging)
        visual_generation = {
            "visual_qa_version": VISUAL_QA_VERSION,
            "generation_id": generation_id,
            "created_at_utc": _datetime.datetime.now(_datetime.UTC).isoformat(),
            "prototype_generation_id": prototype_generation["generation_id"],
            "calibration_generation_id": calibration_generation["generation_id"],
            "status": visual_index["status"],
            "verified_parent_manifest_root": str(parent_root),
            "verified_parent_manifest_hashes": verified_parent_hashes,
            "files": files,
            "freeze_authorized": False,
            "full_conversion_authorized": False,
        }
        _write_json(staging / "generation.json", visual_generation)
        for path in sorted(item for item in staging.rglob("*") if item.is_file()):
            _fsync_file(path)
        for directory in sorted(
            (item for item in staging.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            _fsync_directory(directory)
        _fsync_directory(staging)
        if final.exists():
            raise VisualQaError(f"visual generation already exists: {final}")
        os.replace(staging, final)
        _fsync_directory(generations)
        verify_visual_generation(final)
        if update_link:
            _replace_symlink(output / VISUAL_LINK_NAME, final)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {
        "status": "pending_human_and_codex_visual_review",
        "generation_id": generation_id,
        "generation_root": str(final),
        "compatibility_link": str(output / VISUAL_LINK_NAME),
        "compatibility_link_updated": bool(update_link),
        "clip_count": len(index_records),
        "visual_roles": [record["visual_role"] for record in index_records],
        "coordinate_contract": COORDINATE_CONTRACT,
        "freeze_authorized": False,
        "full_conversion_authorized": False,
    }
