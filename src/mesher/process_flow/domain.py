"""Symmetry-domain modeling and planar geometry clipping."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from .translation.standard_v1 import _geometry_to_face

JsonObject = dict[str, Any]


class SymmetryMode(StrEnum):
    """Supported process-flow domain reductions."""

    FULL = "full"
    UPPER_HALF = "upper_half"
    RIGHT_HALF = "right_half"
    UPPER_RIGHT_QUARTER = "upper_right_quarter"


@dataclass(frozen=True)
class ModelDomain:
    symmetry: SymmetryMode
    center_x: float | None = None
    center_y: float | None = None

    @property
    def restrict_x(self) -> bool:
        return self.symmetry in {
            SymmetryMode.UPPER_RIGHT_QUARTER,
            SymmetryMode.RIGHT_HALF,
        }

    @property
    def restrict_y(self) -> bool:
        return self.symmetry in {
            SymmetryMode.UPPER_RIGHT_QUARTER,
            SymmetryMode.UPPER_HALF,
        }


def normalize_symmetry(value: Any) -> SymmetryMode:
    try:
        return SymmetryMode(value)
    except (TypeError, ValueError) as error:
        allowed = ", ".join(mode.value for mode in SymmetryMode)
        raise ValueError(f"symmetry must be one of: {allowed}.") from error


def model_domain(symmetry: SymmetryMode, faces: list[JsonObject]) -> ModelDomain:
    if symmetry is SymmetryMode.FULL:
        return ModelDomain(symmetry=symmetry)
    bounds = [face_bounds(face) for face in faces]
    return ModelDomain(
        symmetry=symmetry,
        center_x=(min(value[0] for value in bounds) + max(value[2] for value in bounds))
        / 2.0,
        center_y=(min(value[1] for value in bounds) + max(value[3] for value in bounds))
        / 2.0,
    )


def filter_container_to_domain(container: JsonObject, domain: ModelDomain) -> None:
    for field in ("bodies", "vias", "circuits", "bumps"):
        items = container.get(field, [])
        if items is None:
            items = []
        if not isinstance(items, list):
            raise ValueError(f"container.{field} must be a list")
        retained = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(f"container.{field}[{index}] must be an object")
            geometry = item.get("geometry")
            if not isinstance(geometry, dict):
                raise ValueError(
                    f"container.{field}[{index}].geometry must be an object"
                )
            if face_intersects_domain(_geometry_to_face(geometry), domain):
                retained.append(item)
        container[field] = retained

    children = container.get("children", [])
    if children is None:
        children = []
    if not isinstance(children, list):
        raise ValueError("container.children must be a list")
    for index, child in enumerate(children):
        if not isinstance(child, dict):
            raise ValueError(f"container.children[{index}] must be an object")
        filter_container_to_domain(child, domain)


def face_bounds(face: JsonObject) -> tuple[float, float, float, float]:
    face_type = face.get("type")
    dimensions = face.get("dim")
    if face_type == "BOX":
        if not isinstance(dimensions, list) or len(dimensions) != 4:
            raise ValueError("BOX face dim must be [xMin, yMin, xMax, yMax].")
        x1, y1, x2, y2 = (
            _finite_number(value, "BOX face dim") for value in dimensions
        )
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
    if face_type == "CIRCLE":
        if not isinstance(dimensions, list) or len(dimensions) != 3:
            raise ValueError("CIRCLE face dim must be [centerX, centerY, radius].")
        center_x, center_y, radius = (
            _finite_number(value, "CIRCLE face dim") for value in dimensions
        )
        if radius <= 0:
            raise ValueError("CIRCLE radius must be positive.")
        return (
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
        )
    if face_type == "POLYGON":
        loops = polygon_loops(face)
        xs = [point[0] for loop in loops for point in loop]
        ys = [point[1] for loop in loops for point in loop]
        return min(xs), min(ys), max(xs), max(ys)
    raise ValueError(f"Face type {face_type} is not supported by CDB export.")


def face_intersects_domain(face: JsonObject, domain: ModelDomain) -> bool:
    if domain.symmetry is SymmetryMode.FULL:
        return True
    face_type = face.get("type")
    if face_type == "BOX":
        x_min, y_min, x_max, y_max = face_bounds(face)
        if domain.restrict_x and x_max <= domain_center_x(domain):
            return False
        if domain.restrict_y and y_max <= domain_center_y(domain):
            return False
        return x_max > x_min and y_max > y_min
    if face_type == "CIRCLE":
        x_min, y_min, x_max, y_max = face_bounds(face)
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        radius = (x_max - x_min) / 2.0
        dx = (
            max(domain_center_x(domain) - center_x, 0.0)
            if domain.restrict_x
            else 0.0
        )
        dy = (
            max(domain_center_y(domain) - center_y, 0.0)
            if domain.restrict_y
            else 0.0
        )
        return math.hypot(dx, dy) < radius
    if face_type == "POLYGON":
        return polygon_intersection_area(face, domain) > polygon_area_tolerance(face)
    raise ValueError(f"Face type {face_type} is not supported by CDB export.")


def polygon_loops(face: JsonObject) -> list[list[tuple[float, float]]]:
    dimensions = face.get("dim")
    if not isinstance(dimensions, list) or not dimensions:
        raise ValueError("POLYGON face dim must be a non-empty list of polygon loops.")
    loops: list[list[tuple[float, float]]] = []
    for index, polygon in enumerate(dimensions):
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise ValueError(
                f"POLYGON face loop {index} must contain at least 3 points."
            )
        loop = []
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
    return loops


def polygon_intersection_area(face: JsonObject, domain: ModelDomain) -> float:
    loops = polygon_loops(face)
    total_area = 0.0
    for loop_index, loop in enumerate(loops):
        depth = sum(
            _point_in_polygon(loop[0], candidate)
            for candidate_index, candidate in enumerate(loops)
            if candidate_index != loop_index
        )
        clipped = loop
        if domain.restrict_x:
            clipped = _clip_polygon_to_lower_bound(
                clipped,
                axis=0,
                bound=domain_center_x(domain),
            )
        if domain.restrict_y:
            clipped = _clip_polygon_to_lower_bound(
                clipped,
                axis=1,
                bound=domain_center_y(domain),
            )
        clipped_area = _polygon_area(clipped)
        total_area += clipped_area if depth % 2 == 0 else -clipped_area
    return max(0.0, total_area)


def _clip_polygon_to_lower_bound(
    polygon: list[tuple[float, float]],
    *,
    axis: int,
    bound: float,
) -> list[tuple[float, float]]:
    if not polygon:
        return []
    clipped: list[tuple[float, float]] = []
    start = polygon[-1]
    start_inside = start[axis] >= bound
    for end in polygon:
        end_inside = end[axis] >= bound
        if start_inside != end_inside:
            ratio = (bound - start[axis]) / (end[axis] - start[axis])
            intersection = [
                start[coordinate] + ratio * (end[coordinate] - start[coordinate])
                for coordinate in (0, 1)
            ]
            intersection[axis] = bound
            clipped.append((intersection[0], intersection[1]))
        if end_inside:
            clipped.append(end)
        start = end
        start_inside = end_inside
    return clipped


def _point_in_polygon(
    point: tuple[float, float],
    polygon: list[tuple[float, float]],
) -> bool:
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        if (y1 > y) != (y2 > y):
            x_intersection = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_intersection:
                inside = not inside
        previous = current
    return inside


def _polygon_area(polygon: list[tuple[float, float]]) -> float:
    if len(polygon) < 3:
        return 0.0
    return 0.5 * abs(
        sum(
            start[0] * end[1] - start[1] * end[0]
            for start, end in zip(polygon, [*polygon[1:], polygon[0]])
        )
    )


def polygon_area_tolerance(face: JsonObject) -> float:
    loops = polygon_loops(face)
    coordinate_scale = max(
        1.0,
        *(abs(value) for loop in loops for point in loop for value in point),
    )
    return 64.0 * np.finfo(np.float64).eps * coordinate_scale**2


def restrict_grid_lines_to_domain(
    x_lines: list[float],
    y_lines: list[float],
    domain: ModelDomain,
) -> tuple[list[float], list[float]]:
    if domain.restrict_x:
        center_x = domain_center_x(domain)
        x_lines = [value for value in x_lines if value > center_x]
        x_lines.append(center_x)
    if domain.restrict_y:
        center_y = domain_center_y(domain)
        y_lines = [value for value in y_lines if value > center_y]
        y_lines.append(center_y)
    return x_lines, y_lines


def domain_center_x(domain: ModelDomain) -> float:
    if domain.center_x is None:
        raise ValueError("The selected model domain does not define center_x.")
    return domain.center_x


def domain_center_y(domain: ModelDomain) -> float:
    if domain.center_y is None:
        raise ValueError("The selected model domain does not define center_y.")
    return domain.center_y


def _finite_number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite.") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


__all__ = ["ModelDomain", "SymmetryMode"]
