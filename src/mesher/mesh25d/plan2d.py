"""2D mesh stage: plan and execute grid/circular meshing from patterns."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..mesh2d.circular.extend import extend_circular_mesh
from ..mesh2d.circular.imprint_v2 import imprint_circle
from ..mesh2d.controls import XYRegionControl
from ..mesh2d.generators import generate_rectilinear_mesh
from ..mesh2d.model import Mesh2D
from .circle_planning import (
    _CircleExtension, _CirclePattern, _CircleMeshingPlan,
    _add_circle_support_lines, _build_circle_meshing_plan, _circle_label,
    _circle_pattern_from_face, _collect_circle_patterns, _collect_pattern_segments,
    _planar_element_size, _validate_circle_clearances, _validate_circle_domain_topology,
)
from .controls import ResolvedControls
from .domain import (
    ModelDomain, face_bounds,
    restrict_grid_lines_to_domain as _restrict_grid_lines_to_domain,
)
from .geometry.patterns import Pattern2D
from .progress import ProgressCallback, _emit_progress, _emit_item_completed

JsonObject = dict[str, Any]
_CIRCLE_PROJECTION_TOLERANCE_RATIO = 0.2
# 判斷 node 是否在 guide segment 上面；而非用來找 segment 附近 node。
_CIRCLE_GUIDE_TOLERANCE_RATIO = 0.001
_CIRCLE_GUIDE_TOLERANCE_MINIMUM = 1.0
_CIRCLE_MERGE_TOLERANCE_RATIO = 0.002
_CIRCLE_MINIMUM_AREA_RATIO = 0.0004


@dataclass(frozen=True)
class Plan2D:
    pattern: Pattern2D
    element_size: float
    circle_patterns: list[_CirclePattern]
    circle_plan: _CircleMeshingPlan
    pattern_segments: list[tuple[tuple[float, float], tuple[float, float]]]
    x_lines: list[float]
    y_lines: list[float]
    circle_source_refs: dict[_CirclePattern, list[str]]

    @property
    def feature_total(self) -> int:
        return len(self.circle_plan.imprint_patterns) + len(self.circle_plan.extensions)


def plan_2d(
    pattern: Pattern2D,
    controls: ResolvedControls,
    *,
    element_size: float,
    domain: ModelDomain,
) -> Plan2D:
    """Combine patterns and sizing requests before executing mesh algorithms."""
    base_face = pattern.base_face
    if base_face is None:
        raise ValueError("Cannot plan a mesh without a geometry footprint.")
    all_faces = pattern.all_faces
    circle_patterns = _collect_circle_patterns(all_faces)
    circle_source_refs = {_CirclePattern(*key): refs for key, refs in pattern.circle_sources.items()}
    koz_circle_patterns = {_CirclePattern(*key) for key in pattern.koz_circle_sources}
    planar_element_size = _planar_element_size(
        element_size,
        circle_patterns,
        koz_patterns=koz_circle_patterns,
    )
    circle_band_width = 2.0 * planar_element_size
    _validate_circle_domain_topology(
        circle_patterns,
        domain,
        band_width=circle_band_width,
    )
    circle_plan = _build_circle_meshing_plan(
        base_face,
        all_faces,
        circle_patterns,
        band_width=circle_band_width,
    )
    pattern_segments = _collect_pattern_segments(all_faces)
    circle_clearance_tolerance = pattern.clearance_tolerance
    if koz_circle_patterns:
        # Refinement may be driven by the gap to a neighboring feature, which
        # can be smaller than the source feature's own KOZ width.
        circle_clearance_tolerance = min(
            circle_clearance_tolerance, planar_element_size / 100.0
        )
    _validate_circle_clearances(
        list(circle_plan.imprint_patterns),
        circle_band_width,
        tolerance=circle_clearance_tolerance,
    )

    x_lines: list[float] = []
    y_lines: list[float] = []
    extended_patterns = circle_plan.extended_patterns
    for face in all_faces:
        if (
            face.get("type") == "CIRCLE"
            and _circle_pattern_from_face(face) in extended_patterns
        ):
            continue
        xs, ys = _face_grid_lines(face)
        x_lines.extend(xs)
        y_lines.extend(ys)

    if controls.xy:
        bounds = [face_bounds(face) for face in all_faces]
        x_bounds, y_bounds = _restrict_grid_lines_to_domain(
            [min(bound[0] for bound in bounds), max(bound[2] for bound in bounds)],
            [min(bound[1] for bound in bounds), max(bound[3] for bound in bounds)],
            domain,
        )
        _add_xy_control_lines(
            controls.xy, x_lines, y_lines,
            x_bounds=(min(x_bounds), max(x_bounds)),
            y_bounds=(min(y_bounds), max(y_bounds)),
        )

    _add_circle_support_lines(
        list(circle_plan.imprint_patterns),
        circle_band_width,
        x_lines,
        y_lines,
    )
    x_lines, y_lines = _restrict_grid_lines_to_domain(
        x_lines,
        y_lines,
        domain,
    )

    return Plan2D(
        pattern, planar_element_size, circle_patterns, circle_plan,
        pattern_segments, x_lines, y_lines, circle_source_refs,
    )


def build_2d_mesh(plan: Plan2D, *, progress: ProgressCallback | None = None) -> Mesh2D:
    mesh = generate_rectilinear_mesh(
        plan.element_size, plan.x_lines, plan.y_lines,
        coordinate_tolerance=plan.pattern.coordinate_tolerance,
    )
    _imprint_circle_patterns(
        mesh, list(plan.circle_plan.imprint_patterns),
        guide_segments=plan.pattern_segments, element_size=plan.element_size,
        progress=progress, completed_offset=0, total_operations=plan.feature_total,
        source_refs=plan.circle_source_refs,
    )
    _extend_circle_patterns(
        mesh, plan.circle_plan.extensions, element_size=plan.element_size,
        progress=progress, completed_offset=len(plan.circle_plan.imprint_patterns),
        total_operations=plan.feature_total, source_refs=plan.circle_source_refs,
    )
    return mesh


def _add_xy_control_lines(
    controls: list[XYRegionControl],
    x_lines: list[float],
    y_lines: list[float],
    *,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> None:
    """Union independent region cuts without changing the geometry footprint."""
    for control in controls:
        xs, ys = _face_grid_lines(control["region"])
        if (
            max(xs) < x_bounds[0] or min(xs) > x_bounds[1]
            or max(ys) < y_bounds[0] or min(ys) > y_bounds[1]
        ):
            continue
        size = control["element_size"]
        x_lines.extend(_section_avg_lines(xs, size, x_bounds))
        y_lines.extend(_section_avg_lines(ys, size, y_bounds))


def _section_avg_lines(
    coordinates: list[float],
    element_size: float,
    bounds: tuple[float, float],
) -> list[float]:
    """Subdivide vertex intervals before clipping, preserving each region's cuts."""
    ordered = sorted(set(coordinates))
    lines = [value for value in ordered if bounds[0] <= value <= bounds[1]]
    for start, end in zip(ordered, ordered[1:]):
        if end <= bounds[0] or start >= bounds[1]:
            continue
        count = max(1, math.floor((end - start) / element_size))
        lines.extend(
            value for value in np.linspace(start, end, count + 1)[1:-1].tolist()
            if bounds[0] <= value <= bounds[1]
        )
    return lines


def _imprint_circle_patterns(
    mesh_2d: Any,
    circle_patterns: list[_CirclePattern],
    *,
    guide_segments: list[
        tuple[tuple[float, float], tuple[float, float]]
    ],
    element_size: float,
    progress: ProgressCallback | None = None,
    completed_offset: int = 0,
    total_operations: int = 0,
    source_refs: dict[_CirclePattern, list[str]] | None = None,
) -> None:
    for index, pattern in enumerate(circle_patterns):
        operation_index = completed_offset + index + 1
        started = time.perf_counter()
        _emit_progress(
            progress,
            current=operation_index - 1,
            total=total_operations,
            unit="features",
            message=f"Imprinting feature {operation_index} of {total_operations}.",
        )
        try:
            imprint_circle(
                mesh_2d,
                center=pattern.center,
                radius=pattern.radius,
                projection_tolerance=(
                    _CIRCLE_PROJECTION_TOLERANCE_RATIO * element_size
                ),
                guide_tolerance=max(
                    _CIRCLE_GUIDE_TOLERANCE_RATIO * element_size,
                    _CIRCLE_GUIDE_TOLERANCE_MINIMUM,
                ),
                merge_tolerance=(
                    _CIRCLE_MERGE_TOLERANCE_RATIO * element_size
                ),
                minimum_area=(
                    _CIRCLE_MINIMUM_AREA_RATIO * element_size**2
                ),
                guide_segments=guide_segments,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Failed to imprint circle pattern {_circle_label(pattern)}: {exc}"
            ) from exc
        refs = (source_refs or {}).get(pattern, [])
        _emit_item_completed(
            progress,
            stage="building_2d_mesh",
            data={
                "featureRef": refs[0] if refs else f"circle-pattern:{operation_index}",
                "sourceRefs": refs,
                "featureType": "circle",
                "geometryType": "CylinderGeometry",
                "operation": "imprint",
                "wallDurationMs": int(round((time.perf_counter() - started) * 1000)),
            },
        )
        _emit_progress(
            progress,
            current=operation_index,
            total=total_operations,
            unit="features",
            message=f"Imprinted feature {operation_index} of {total_operations}.",
        )


def _extend_circle_patterns(
    mesh_2d: Any,
    extensions: tuple[_CircleExtension, ...],
    *,
    element_size: float,
    progress: ProgressCallback | None = None,
    completed_offset: int = 0,
    total_operations: int = 0,
    source_refs: dict[_CirclePattern, list[str]] | None = None,
) -> None:
    for index, extension in enumerate(extensions):
        operation_index = completed_offset + index + 1
        started = time.perf_counter()
        _emit_progress(
            progress,
            current=operation_index - 1,
            total=total_operations,
            unit="features",
            message=f"Extending feature {operation_index} of {total_operations}.",
        )
        try:
            extend_circular_mesh(
                mesh_2d,
                element_size=element_size,
                center_x=extension.center[0],
                center_y=extension.center[1],
                inner_radius=extension.inner.radius,
                outer_radius=extension.outer.radius,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(
                "Failed to extend circular mesh from "
                f"{_circle_label(extension.inner)} to "
                f"{_circle_label(extension.outer)}: {exc}"
            ) from exc
        refs = (source_refs or {}).get(extension.outer, [])
        _emit_item_completed(
            progress,
            stage="building_2d_mesh",
            data={
                "featureRef": refs[0] if refs else f"circle-pattern:{operation_index}",
                "sourceRefs": refs,
                "featureType": "circle",
                "geometryType": "CylinderGeometry",
                "operation": "extension",
                "wallDurationMs": int(round((time.perf_counter() - started) * 1000)),
            },
        )
        _emit_progress(
            progress,
            current=operation_index,
            total=total_operations,
            unit="features",
            message=f"Extended feature {operation_index} of {total_operations}.",
        )


def _face_grid_lines(face: JsonObject) -> tuple[list[float], list[float]]:
    face_type = face.get("type")
    dim = face.get("dim")

    if face_type == "BOX":
        if not isinstance(dim, list) or len(dim) != 4:
            raise ValueError("BOX face dim must be [xMin, yMin, xMax, yMax].")
        x1, y1, x2, y2 = (_finite_number(value, "BOX face dim") for value in dim)
        return [x1, x2], [y1, y2]

    if face_type == "POLYGON":
        if not isinstance(dim, list):
            raise ValueError("POLYGON face dim must be a list of polygon loops.")
        xs: list[float] = []
        ys: list[float] = []
        for polygon in dim:
            if not isinstance(polygon, list):
                raise ValueError("POLYGON face loop must be a list.")
            for point in polygon:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    raise ValueError("POLYGON face point must be [x, y].")
                xs.append(_finite_number(point[0], "POLYGON face point x"))
                ys.append(_finite_number(point[1], "POLYGON face point y"))
        return xs, ys

    if face_type == "CIRCLE":
        if not isinstance(dim, list) or len(dim) != 3:
            raise ValueError("CIRCLE face dim must be [x, y, radius].")
        x, y, radius = (_finite_number(value, "CIRCLE face dim") for value in dim)
        if radius <= 0:
            raise ValueError("CIRCLE face radius must be greater than 0.")
        return [x - radius, x + radius], [y - radius, y + radius]

    raise ValueError(f"Face type {face_type} is not supported by CDB export.")


def _finite_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number.")
    return number
