"""High-level Standard V1 geometry-to-solid-mesh pipeline."""

from __future__ import annotations

import copy
import math
import resource
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from process_flow_kernel import validate_geometry_semantic_keys

from ..mesh2d import Mesh2D
from ..mesh2d.circular import extend_circular_mesh, imprint_circle
from ..mesh2d.generators import generate_rectilinear_mesh
from ..mesh3d import Mesh3D, extrude_mesh
from .circle_planning import (
    CIRCLE_MINIMUM_QUAD_SCALED_JACOBIAN,
    _CircleExtension,
    _CirclePattern,
    _add_circle_support_lines,
    _build_circle_meshing_plan,
    _circle_label,
    _circle_pattern_from_face,
    _collect_circle_patterns,
    _collect_circle_source_refs,
    _collect_pattern_segments,
    _planar_element_size,
    _validate_circle_clearances,
    _validate_circle_domain_topology,
)
from .domain import (
    ModelDomain as _ModelDomain,
    SymmetryMode,
    domain_center_x as _domain_center_x,
    domain_center_y as _domain_center_y,
    filter_container_to_domain as _filter_container_to_domain,
    model_domain as _model_domain,
    normalize_symmetry as _normalize_symmetry,
    restrict_grid_lines_to_domain as _restrict_grid_lines_to_domain,
)
from .material_assignment import MaterialAssignmentResolver
from .translation import (
    _geometry_to_face,
    translate_layer_assignments,
    translate_planar_pattern,
)

JsonObject = dict[str, Any]
ProgressCallback = Callable[[JsonObject], None]


@dataclass(frozen=True)
class _StageTimer:
    wall_started: float
    cpu_started: float


def build_mesh_from_structure(
    geometry_structure: JsonObject,
    *,
    element_size: float,
    symmetry: SymmetryMode | str = SymmetryMode.FULL,
    progress: ProgressCallback | None = None,
) -> Mesh3D:
    """Build a full or symmetry-reduced 2.5D mesh from a geometry structure.

    Reduced models use the center of the complete XY footprint bounds as
    their symmetry origin. ``upper_half`` retains the upper half,
    ``right_half`` retains the right half, and ``upper_right_quarter`` retains
    the upper-right quarter.
    """
    stage = _start_stage(progress, "validating", "Checking geometry input.")
    normalized_element_size = _positive_finite_number(element_size, "elementSize")
    normalized_symmetry = _normalize_symmetry(symmetry)
    validate_geometry_semantic_keys(geometry_structure)
    root = _root_container(geometry_structure)
    _complete_stage(progress, "validating", stage)

    # The translator annotates containers with priority during 3D pattern
    # extraction, so keep the caller's preview snapshot immutable.
    stage = _start_stage(progress, "analyzing_geometry", "Analyzing geometry patterns.")
    container = copy.deepcopy(root)
    planar_pattern = translate_planar_pattern(container)
    base_face = planar_pattern.base_face
    faces = list(planar_pattern.feature_faces)
    if base_face is None:
        raise ValueError("CDB export requires at least one geometry body or feature.")

    domain = _model_domain(
        normalized_symmetry,
        [base_face, *faces],
    )
    if normalized_symmetry is not SymmetryMode.FULL:
        _filter_container_to_domain(container, domain)
        planar_pattern = translate_planar_pattern(container)
        base_face = planar_pattern.base_face
        faces = list(planar_pattern.feature_faces)
        if base_face is None:
            raise ValueError(
                f"CDB export has no geometry with positive XY area in "
                f"{normalized_symmetry.value}."
            )

    all_faces = [base_face, *faces]
    circle_patterns = _collect_circle_patterns(all_faces)
    circle_source_refs = _collect_circle_source_refs(container)
    planar_element_size = _planar_element_size(
        normalized_element_size,
        circle_patterns,
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
    _validate_circle_clearances(
        list(circle_plan.imprint_patterns),
        circle_band_width,
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

    layer_infos = translate_layer_assignments(container)
    _complete_stage(
        progress,
        "analyzing_geometry",
        stage,
        data={
            "faceCount": len(all_faces),
            "circlePatternCount": len(circle_patterns),
            "imprintOperationCount": len(circle_plan.imprint_patterns),
            "extensionOperationCount": len(circle_plan.extensions),
            "xGridLineCount": len(set(x_lines)),
            "yGridLineCount": len(set(y_lines)),
            "layerBoundaryCount": len(layer_infos),
            "layerIntervalCount": max(0, len(layer_infos) - 1),
            "assignmentCount": sum(
                len(layer_info.assignments) for layer_info in layer_infos[:-1]
            ),
            "planarElementSize": planar_element_size,
            "symmetry": normalized_symmetry.value,
        },
    )

    feature_total = len(circle_plan.imprint_patterns) + len(circle_plan.extensions)
    stage = _start_stage(progress, "building_2d_mesh", "Generating base grid.")
    mesh_2d = generate_rectilinear_mesh(
        planar_element_size,
        x_lines,
        y_lines,
    )

    _imprint_circle_patterns(
        mesh_2d,
        list(circle_plan.imprint_patterns),
        guide_segments=pattern_segments,
        band_width=circle_band_width,
        target_edge_size=planar_element_size,
        progress=progress,
        completed_offset=0,
        total_operations=feature_total,
        source_refs=circle_source_refs,
    )
    _extend_circle_patterns(
        mesh_2d,
        circle_plan.extensions,
        element_size=planar_element_size,
        progress=progress,
        completed_offset=len(circle_plan.imprint_patterns),
        total_operations=feature_total,
        source_refs=circle_source_refs,
    )
    elements_2d = _clockwise_elements(mesh_2d.elements)
    _complete_stage(
        progress,
        "building_2d_mesh",
        stage,
        data={
            "featureOperationCount": feature_total,
            "node2DCount": int(len(mesh_2d.nodes)),
            "element2DCount": int(len(elements_2d)),
        },
    )

    stage = _start_stage(
        progress,
        "building_3d_mesh",
        "Building 3D mesh layers.",
        current=0,
        total=max(0, len(layer_infos) - 1),
        unit="layers",
    )
    extrusion_source = Mesh2D(nodes=mesh_2d.nodes, elements=elements_2d)
    resolver = MaterialAssignmentResolver(extrusion_source)
    layer_started_at: dict[int, float] = {}

    def on_layer_started(layer_index: int) -> None:
        layer_started_at[layer_index] = time.perf_counter()

    def on_layer_completed(
        layer_index: int,
        nodes_added: int,
        elements_added: int,
    ) -> None:
        if progress is not None:
            progress(
                {
                    "event": "item.completed",
                    "stage": "building_3d_mesh",
                    "data": {
                        "itemType": "layer",
                        "layerIndex": layer_index + 1,
                        "assignmentCount": len(layer_infos[layer_index].assignments),
                        "nodesAdded": nodes_added,
                        "elementsAdded": elements_added,
                        "wallDurationMs": int(
                            round(
                                (
                                    time.perf_counter()
                                    - layer_started_at.get(
                                        layer_index,
                                        time.perf_counter(),
                                    )
                                )
                                * 1000
                            )
                        ),
                    },
                }
            )
        _emit_progress(
            progress,
            current=layer_index + 1,
            total=max(0, len(layer_infos) - 1),
            unit="layers",
            message=(
                f"Built layer {layer_index + 1} of "
                f"{max(0, len(layer_infos) - 1)}."
            ),
        )

    mesh = extrude_mesh(
        extrusion_source,
        resolver.iter_extrusion_layers(
            layer_infos,
            progress=progress,
            _on_layer_started=on_layer_started,
        ),
        element_size=normalized_element_size,
        component_ids_by_name=resolver.component_ids_by_name,
        _on_layer_completed=on_layer_completed,
    )
    _complete_stage(
        progress,
        "building_3d_mesh",
        stage,
        data={
            "nodeCount": mesh.node_count,
            "elementCount": mesh.element_count,
            "componentCount": mesh.component_count,
        },
    )
    return mesh




def _imprint_circle_patterns(
    mesh_2d: Any,
    circle_patterns: list[_CirclePattern],
    *,
    guide_segments: list[
        tuple[tuple[float, float], tuple[float, float]]
    ],
    band_width: float,
    target_edge_size: float,
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
                band_width=band_width,
                guide_segments=guide_segments,
                target_edge_size=target_edge_size,
                min_quad_scaled_jacobian=(
                    CIRCLE_MINIMUM_QUAD_SCALED_JACOBIAN
                ),
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


def _clockwise_elements(elements: Any) -> np.ndarray:
    source = np.asarray(elements)
    if source.ndim != 2 or source.shape[1] != 4:
        raise ValueError("mesh2D.elements must have shape (m, 4).")

    clockwise = np.empty_like(source)
    triangle_mask = source[:, 2] == source[:, 3]
    clockwise[~triangle_mask] = source[~triangle_mask][:, [0, 3, 2, 1]]
    clockwise[triangle_mask] = source[triangle_mask][:, [0, 2, 1, 1]]
    return clockwise



def _root_container(geometry_structure: JsonObject) -> JsonObject:
    if not isinstance(geometry_structure, dict):
        raise ValueError("geometryStructure must be an object.")
    root = geometry_structure.get("root")
    if not isinstance(root, dict):
        raise ValueError("geometryStructure.root must be an object.")
    return root


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


def _positive_finite_number(value: float, name: str) -> float:
    number = _finite_number(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be greater than 0.")
    return number


def _finite_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number.")
    return number


def _start_stage(
    progress: ProgressCallback | None,
    stage: str,
    message: str,
    *,
    current: int | None = None,
    total: int | None = None,
    unit: str | None = None,
) -> _StageTimer:
    _emit(
        progress,
        {
            "event": "stage.started",
            "stage": stage,
            "current": current,
            "total": total,
            "unit": unit,
            "message": message,
            "data": {},
        },
    )
    return _StageTimer(time.perf_counter(), time.process_time())


def _complete_stage(
    progress: ProgressCallback | None,
    stage: str,
    timer: _StageTimer,
    *,
    data: JsonObject | None = None,
) -> None:
    metrics = dict(data or {})
    metrics.update(
        {
            "wallDurationMs": int(round((time.perf_counter() - timer.wall_started) * 1000)),
            "cpuDurationMs": int(round((time.process_time() - timer.cpu_started) * 1000)),
            "peakRssBytes": _peak_rss_bytes(),
        }
    )
    _emit(
        progress,
        {
            "event": "stage.completed",
            "stage": stage,
            "data": metrics,
        },
    )


def _emit_progress(
    progress: ProgressCallback | None,
    *,
    current: int | None,
    total: int | None,
    unit: str | None,
    message: str,
    data: JsonObject | None = None,
) -> None:
    _emit(
        progress,
        {
            "event": "progress",
            "current": current,
            "total": total,
            "unit": unit,
            "message": message,
            "data": data or {},
        },
    )


def _emit_item_completed(
    progress: ProgressCallback | None,
    *,
    stage: str,
    data: JsonObject,
) -> None:
    _emit(
        progress,
        {
            "event": "item.completed",
            "stage": stage,
            "data": data,
        },
    )


def _emit(progress: ProgressCallback | None, payload: JsonObject) -> None:
    if progress is not None:
        progress(payload)


def _peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


__all__ = ["SymmetryMode", "build_mesh_from_structure"]
