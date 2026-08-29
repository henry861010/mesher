"""Plan circular imprint and extension operations for process-flow geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .domain import (
    ModelDomain as _ModelDomain,
    SymmetryMode,
    domain_center_x as _domain_center_x,
    domain_center_y as _domain_center_y,
)
from .translation import _geometry_to_face

JsonObject = dict[str, Any]


CIRCLE_CLEARANCE_TOLERANCE = 1e-6
CIRCLE_CENTER_TOLERANCE = 1e-6
CIRCLE_MINIMUM_QUAD_SCALED_JACOBIAN = 0.3


@dataclass(frozen=True, order=True)
class _CirclePattern:
    center_x: float
    center_y: float
    radius: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.center_x, self.center_y)


@dataclass(frozen=True)
class _CircleExtension:
    inner: _CirclePattern
    outer: _CirclePattern
    center: tuple[float, float]


@dataclass(frozen=True)
class _CircleMeshingPlan:
    imprint_patterns: tuple[_CirclePattern, ...]
    extensions: tuple[_CircleExtension, ...]

    @property
    def extended_patterns(self) -> frozenset[_CirclePattern]:
        return frozenset(extension.outer for extension in self.extensions)



def _validate_circle_domain_topology(
    circle_patterns: list[_CirclePattern],
    domain: _ModelDomain,
    *,
    band_width: float,
) -> None:
    if domain.symmetry is SymmetryMode.FULL:
        return

    # The external open-circle mesher supports sectors bounded by rays from
    # the circle center. Include its outer search margin so near-axis bands
    # fail here with a model-specific error instead of a topology error later.
    open_radius_margin = 0.6 * band_width
    for pattern in circle_patterns:
        selection_radius = pattern.radius + open_radius_margin
        boundaries = []
        if domain.restrict_x:
            boundaries.append(("x", pattern.center_x, _domain_center_x(domain)))
        if domain.restrict_y:
            boundaries.append(("y", pattern.center_y, _domain_center_y(domain)))
        for axis, circle_center, boundary in boundaries:
            offset = abs(circle_center - boundary)
            if (
                CIRCLE_CENTER_TOLERANCE < offset
                < selection_radius - CIRCLE_CENTER_TOLERANCE
            ):
                raise ValueError(
                    "Open circle meshing requires each intersecting symmetry "
                    f"boundary to pass through the circle center: symmetry="
                    f"{domain.symmetry.value}, circle {_circle_label(pattern)}, "
                    f"boundary {axis}={boundary:.12g}."
                )



def _collect_circle_patterns(faces: list[JsonObject]) -> list[_CirclePattern]:
    patterns: set[_CirclePattern] = set()
    for face in faces:
        if face.get("type") != "CIRCLE":
            continue
        patterns.add(_circle_pattern_from_face(face))
    return sorted(patterns)


def _collect_circle_source_refs(
    container: JsonObject,
) -> dict[_CirclePattern, list[str]]:
    refs: dict[_CirclePattern, list[str]] = {}

    def visit(current: JsonObject, path: str) -> None:
        for field in ("bodies", "vias", "circuits", "bumps"):
            items = current.get(field, [])
            if not isinstance(items, list):
                continue
            for index, item in enumerate(items):
                if not isinstance(item, dict) or not isinstance(item.get("geometry"), dict):
                    continue
                face = _geometry_to_face(item["geometry"])
                if face.get("type") != "CIRCLE":
                    continue
                pattern = _circle_pattern_from_face(face)
                refs.setdefault(pattern, []).append(
                    str(item.get("id") or f"{path}.{field}[{index}]")
                )
        children = current.get("children", [])
        if isinstance(children, list):
            for child_index, child in enumerate(children):
                if isinstance(child, dict):
                    visit(child, f"{path}.children[{child_index}]")

    visit(container, "root")
    return refs


def _circle_pattern_from_face(face: JsonObject) -> _CirclePattern:
    dim = face.get("dim")
    if not isinstance(dim, list) or len(dim) != 3:
        raise ValueError("CIRCLE face dim must be [x, y, radius].")
    center_x, center_y, radius = (
        _finite_number(value, "CIRCLE face dim") for value in dim
    )
    if radius <= 0.0:
        raise ValueError("CIRCLE face radius must be greater than 0.")
    return _CirclePattern(center_x, center_y, radius)


def _build_circle_meshing_plan(
    base_face: JsonObject,
    faces: list[JsonObject],
    circle_patterns: list[_CirclePattern],
    *,
    band_width: float,
) -> _CircleMeshingPlan:
    imprint_only = _CircleMeshingPlan(tuple(circle_patterns), ())
    if base_face.get("type") != "CIRCLE" or len(circle_patterns) < 2:
        return imprint_only

    base_circle = _circle_pattern_from_face(base_face)
    if not all(
        _face_is_contained_by_circle(face, base_circle)
        for face in faces
    ):
        return imprint_only

    concentric = sorted(
        (
            pattern
            for pattern in circle_patterns
            if _same_circle_center(pattern, base_circle)
            and pattern.radius <= base_circle.radius + CIRCLE_CENTER_TOLERANCE
        ),
        key=lambda pattern: pattern.radius,
    )
    if len(concentric) < 2 or concentric[-1] != base_circle:
        return imprint_only

    non_concentric = [
        pattern
        for pattern in circle_patterns
        if pattern not in concentric
    ]
    source_index = None
    for index, candidate in enumerate(concentric[:-1]):
        if _line_pattern_crosses_annulus(
            faces,
            center=candidate.center,
            inner_radius=candidate.radius,
            outer_radius=base_circle.radius,
        ):
            continue
        if not all(
            _circle_band_is_inside_circle(
                pattern,
                candidate,
                band_width=band_width,
            )
            for pattern in non_concentric
        ):
            continue
        source_index = index
        break

    if source_index is None:
        return imprint_only

    extension_chain = concentric[source_index:]
    extension_center = extension_chain[0].center
    extensions = tuple(
        _CircleExtension(inner, outer, extension_center)
        for inner, outer in zip(extension_chain, extension_chain[1:])
    )
    extended_patterns = {
        extension.outer for extension in extensions
    }
    imprint_patterns = tuple(
        pattern
        for pattern in circle_patterns
        if pattern not in extended_patterns
    )
    return _CircleMeshingPlan(imprint_patterns, extensions)


def _same_circle_center(
    left: _CirclePattern,
    right: _CirclePattern,
) -> bool:
    return (
        abs(left.center_x - right.center_x) <= CIRCLE_CENTER_TOLERANCE
        and abs(left.center_y - right.center_y) <= CIRCLE_CENTER_TOLERANCE
    )


def _face_is_contained_by_circle(
    face: JsonObject,
    container: _CirclePattern,
) -> bool:
    if face.get("type") == "CIRCLE":
        pattern = _circle_pattern_from_face(face)
        center_distance = math.hypot(
            pattern.center_x - container.center_x,
            pattern.center_y - container.center_y,
        )
        return (
            center_distance + pattern.radius
            <= container.radius + CIRCLE_CENTER_TOLERANCE
        )

    points = [point for segment in _face_boundary_segments(face) for point in segment]
    return all(
        math.hypot(
            point[0] - container.center_x,
            point[1] - container.center_y,
        )
        <= container.radius + CIRCLE_CENTER_TOLERANCE
        for point in points
    )


def _line_pattern_crosses_annulus(
    faces: list[JsonObject],
    *,
    center: tuple[float, float],
    inner_radius: float,
    outer_radius: float,
) -> bool:
    for face in faces:
        if face.get("type") == "CIRCLE":
            continue
        for start, end in _face_boundary_segments(face):
            if _segment_intersects_annulus(
                start,
                end,
                center=center,
                inner_radius=inner_radius,
                outer_radius=outer_radius,
            ):
                return True
    return False


def _collect_pattern_segments(
    faces: list[JsonObject],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    segments = []
    for face in faces:
        if face.get("type") == "CIRCLE":
            continue
        segments.extend(_face_boundary_segments(face))
    return segments


def _face_boundary_segments(
    face: JsonObject,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    face_type = face.get("type")
    dim = face.get("dim")
    loops: list[list[tuple[float, float]]] = []

    if face_type == "BOX":
        if not isinstance(dim, list) or len(dim) != 4:
            raise ValueError("BOX face dim must be [xMin, yMin, xMax, yMax].")
        x1, y1, x2, y2 = (
            _finite_number(value, "BOX face dim") for value in dim
        )
        loops.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
    elif face_type == "POLYGON":
        if not isinstance(dim, list):
            raise ValueError("POLYGON face dim must be a list of polygon loops.")
        for polygon in dim:
            if not isinstance(polygon, list):
                raise ValueError("POLYGON face loop must be a list.")
            loop: list[tuple[float, float]] = []
            for point in polygon:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    raise ValueError("POLYGON face point must be [x, y].")
                loop.append(
                    (
                        _finite_number(point[0], "POLYGON face point x"),
                        _finite_number(point[1], "POLYGON face point y"),
                    )
                )
            loops.append(loop)
    else:
        raise ValueError(f"Face type {face_type} is not supported by CDB export.")

    segments = []
    for loop in loops:
        if len(loop) < 2:
            continue
        segments.extend(zip(loop, [*loop[1:], loop[0]]))
    return segments


def _segment_intersects_annulus(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    center: tuple[float, float],
    inner_radius: float,
    outer_radius: float,
) -> bool:
    start_offset = np.asarray(start, dtype=np.float64) - np.asarray(
        center,
        dtype=np.float64,
    )
    end_offset = np.asarray(end, dtype=np.float64) - np.asarray(
        center,
        dtype=np.float64,
    )
    direction = end_offset - start_offset
    length_squared = float(np.dot(direction, direction))
    if length_squared == 0.0:
        minimum_distance = maximum_distance = float(np.linalg.norm(start_offset))
    else:
        projection = float(-np.dot(start_offset, direction) / length_squared)
        projection = min(1.0, max(0.0, projection))
        closest = start_offset + projection * direction
        minimum_distance = float(np.linalg.norm(closest))
        maximum_distance = max(
            float(np.linalg.norm(start_offset)),
            float(np.linalg.norm(end_offset)),
        )
    return (
        maximum_distance >= inner_radius - CIRCLE_CENTER_TOLERANCE
        and minimum_distance <= outer_radius + CIRCLE_CENTER_TOLERANCE
    )


def _circle_band_is_inside_circle(
    pattern: _CirclePattern,
    container: _CirclePattern,
    *,
    band_width: float,
) -> bool:
    center_distance = math.hypot(
        pattern.center_x - container.center_x,
        pattern.center_y - container.center_y,
    )
    return (
        center_distance + pattern.radius + band_width
        < container.radius - CIRCLE_CLEARANCE_TOLERANCE
    )


def _planar_element_size(
    element_size: float,
    circle_patterns: list[_CirclePattern],
) -> float:
    if not circle_patterns:
        return element_size
    minimum_radius = min(pattern.radius for pattern in circle_patterns)
    return min(element_size, minimum_radius / 3.0)


def _validate_circle_clearances(
    circle_patterns: list[_CirclePattern],
    band_width: float,
) -> None:
    required_clearance = band_width + CIRCLE_CLEARANCE_TOLERANCE
    for left_index, left in enumerate(circle_patterns):
        for right in circle_patterns[left_index + 1 :]:
            center_distance = math.hypot(
                right.center_x - left.center_x,
                right.center_y - left.center_y,
            )
            clearance = max(
                center_distance - left.radius - right.radius,
                abs(left.radius - right.radius) - center_distance,
            )
            if clearance <= required_clearance:
                raise ValueError(
                    "Circle patterns have intersecting, tangent, or overlapping "
                    "imprint bands: "
                    f"{_circle_label(left)} and {_circle_label(right)}; "
                    f"clearance={clearance:.12g}, required>{required_clearance:.12g}."
                )


def _add_circle_support_lines(
    circle_patterns: list[_CirclePattern],
    band_width: float,
    x_lines: list[float],
    y_lines: list[float],
) -> None:
    for pattern in circle_patterns:
        support_radius = pattern.radius + 2.0 * band_width
        x_lines.extend(
            [
                pattern.center_x - support_radius,
                pattern.center_x + support_radius,
            ]
        )
        y_lines.extend(
            [
                pattern.center_y - support_radius,
                pattern.center_y + support_radius,
            ]
        )


def _circle_label(pattern: _CirclePattern) -> str:
    return (
        f"center=({pattern.center_x:.12g}, {pattern.center_y:.12g}), "
        f"radius={pattern.radius:.12g}"
    )


def _finite_number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite.") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result
