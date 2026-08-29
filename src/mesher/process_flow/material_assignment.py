"""Resolve process-flow material events into typed extrusion layers."""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any, Callable

import numpy as np
from matplotlib.path import Path

from ..mesh2d import Mesh2D
from ..mesh3d import ExtrusionLayer
from .translation import AssignmentKind, LayerAssignments

JsonObject = dict[str, Any]
ProgressCallback = Callable[[JsonObject], None]
LayerStartedCallback = Callable[[int], None]


class MaterialAssignmentResolver:
    """Statefully rasterize translated process events onto a planar mesh.

    Priority and density-occupancy arrays intentionally persist across Z
    intervals.  This is the typed replacement for the assignment portion of
    the former ``Dragger`` implementation.
    """

    def __init__(self, mesh: Mesh2D) -> None:
        self._elements = mesh.elements
        self._nodes_xy = mesh.nodes[:, :2]
        self._element_areas = _element_areas(self._nodes_xy, self._elements)
        self._component_ids = np.zeros(mesh.element_count, dtype=np.int32)
        self._priorities = np.zeros(mesh.element_count, dtype=np.float64)
        self._density_occupancy = np.zeros(mesh.element_count, dtype=np.float64)
        self.component_ids_by_name = {"EMPTY": 0}

    def iter_extrusion_layers(
        self,
        layers: tuple[LayerAssignments, ...],
        *,
        progress: ProgressCallback | None = None,
        _on_layer_started: LayerStartedCallback | None = None,
    ) -> Iterator[ExtrusionLayer]:
        """Yield resolved intervals lazily so progress ordering is preserved."""

        self._component_ids[:] = 0
        layer_total = max(0, len(layers) - 1)
        for index, layer in enumerate(layers[:-1]):
            if _on_layer_started is not None:
                _on_layer_started(index)
            self._apply_assignments(
                layer.assignments,
                layer_index=index,
                layer_total=layer_total,
                progress=progress,
            )
            yield ExtrusionLayer(
                z_min=layer.z,
                z_max=layers[index + 1].z,
                element_component_ids=self._component_ids.copy(),
            )

    def _select_elements(
        self,
        face: JsonObject | None,
        *,
        keep_out: float = 0.0,
        indices: np.ndarray | None = None,
        tolerance: float = 0.01,
    ) -> np.ndarray:
        selected_indices = _normalize_indices(indices, len(self._elements))
        if face is None:
            return np.ones(len(selected_indices), dtype=bool)
        if len(selected_indices) == 0:
            return np.zeros(0, dtype=bool)

        coordinates = self._nodes_xy[self._elements[selected_indices]]
        face_type = face["type"]
        dimensions = face["dim"]
        if face_type == "BOX":
            lower_x = dimensions[0] + keep_out - tolerance < coordinates[:, :, 0]
            lower_y = dimensions[1] + keep_out - tolerance < coordinates[:, :, 1]
            upper_x = coordinates[:, :, 0] < dimensions[2] - keep_out + tolerance
            upper_y = coordinates[:, :, 1] < dimensions[3] - keep_out + tolerance
            return np.all(lower_x & lower_y & upper_x & upper_y, axis=1)
        if face_type == "POLYGON":
            mask = np.zeros(len(coordinates), dtype=bool)
            flat_coordinates = coordinates.reshape(-1, 2)
            for polygon in dimensions:
                radius = tolerance if _signed_area(polygon) > 0 else -tolerance
                path = Path(np.asarray(polygon, dtype=np.float64))
                point_mask = path.contains_points(flat_coordinates, radius=radius)
                element_mask = point_mask.reshape(len(coordinates), 4).all(axis=1)
                mask = np.logical_xor(mask, element_mask)
            return mask
        if face_type == "CIRCLE":
            center_x, center_y, radius = dimensions
            dx = coordinates[:, :, 0] - center_x
            dy = coordinates[:, :, 1] - center_y
            return np.all(
                dx * dx + dy * dy
                <= (radius - keep_out + tolerance) ** 2,
                axis=1,
            )
        raise ValueError(f"The face type {face_type} is not supported.")

    def _component_id(self, name: str) -> int:
        if name not in self.component_ids_by_name:
            self.component_ids_by_name[name] = len(self.component_ids_by_name)
        return self.component_ids_by_name[name]

    def _select_density(
        self,
        indices: np.ndarray,
        density: float,
        total_area: float,
        random_seed: int,
    ) -> np.ndarray:
        # Preserve the existing deterministic selection semantics exactly.
        if density == 0:
            return np.empty(0, dtype=np.int32)
        areas = self._element_areas[indices]
        target_indices = np.arange(len(areas), dtype=np.int32)
        target = (density / 100.0) * total_area
        rng = np.random.default_rng(random_seed)
        random_indices = target_indices[rng.permutation(len(areas))]
        cumulative = np.cumsum(areas[random_indices])
        count = np.searchsorted(cumulative, target, side="right")
        if count > 0:
            chosen = target_indices[random_indices[: count + 1]]
            return indices[chosen]
        return np.empty(0, dtype=np.int32)

    def _apply_assignments(
        self,
        assignments: tuple[JsonObject, ...],
        *,
        layer_index: int,
        layer_total: int,
        progress: ProgressCallback | None,
    ) -> None:
        ordered = sorted(
            assignments,
            key=lambda item: (item["type"], -item["areas"][0]["priority"]),
        )
        for assignment_index, assignment in enumerate(ordered):
            started = time.perf_counter()
            _emit_progress(
                progress,
                current=layer_index,
                total=layer_total,
                unit="layers",
                message=(
                    f"Assigning feature {assignment_index + 1} of {len(ordered)} "
                    f"in layer {layer_index + 1} of {layer_total}."
                ),
            )
            potential = np.flatnonzero(self._select_elements(assignment["face"]))
            selected = np.empty(0, dtype=np.int32)
            if len(potential):
                kind = AssignmentKind(assignment["type"])
                areas = assignment["areas"]
                if kind is AssignmentKind.START_NORMAL:
                    area = areas[0]
                    selected = potential[
                        self._priorities[potential] <= area["priority"]
                    ]
                    self._component_ids[selected] = self._component_id(
                        area["material"]
                    )
                    self._priorities[selected] = area["priority"]
                elif kind is AssignmentKind.END:
                    original_priority = areas[0]["priority_o"]
                    release_mask = (
                        self._density_occupancy[potential] == original_priority
                    )
                    self._density_occupancy[potential[release_mask]] = 0
                    remaining = potential[
                        self._priorities[potential] == original_priority
                    ]
                    for area in sorted(
                        areas,
                        key=lambda item: item["priority"],
                        reverse=True,
                    ):
                        local_mask = self._select_elements(
                            area["face"],
                            indices=remaining,
                        )
                        selected = remaining[local_mask]
                        remaining = remaining[~local_mask]
                        self._component_ids[selected] = self._component_id(
                            area["material"]
                        )
                        self._priorities[selected] = area["priority"]
                        if len(remaining) == 0:
                            break
                elif kind is AssignmentKind.START_DENSITY:
                    area = areas[0]
                    selected = potential[
                        self._priorities[potential] <= area["priority"]
                    ]
                    selected = selected[
                        self._density_occupancy[selected] <= area["priority"]
                    ]
                    self._density_occupancy[selected] = area["priority"]
                    keep_out_mask = self._select_elements(
                        assignment["face"],
                        keep_out=area["koz"],
                        indices=selected,
                    )
                    selected = selected[keep_out_mask]
                    selected = self._select_density(
                        selected,
                        area["density"],
                        float(np.sum(self._element_areas[selected])),
                        random_seed=layer_index * 100000 + assignment_index,
                    )
                    self._component_ids[selected] = self._component_id(
                        area["material"]
                    )
                    self._priorities[selected] = area["priority"]

            _emit_assignment_completed(
                progress,
                assignment,
                layer=layer_index,
                assignment_index=assignment_index,
                duration_ms=int(round((time.perf_counter() - started) * 1000)),
                selected_element_count=len(selected),
            )


def _normalize_indices(indices: np.ndarray | None, count: int) -> np.ndarray:
    if indices is None:
        return np.arange(count, dtype=np.int32)
    result = np.asarray(indices, dtype=np.int32)
    return result.reshape(1) if result.ndim == 0 else result


def _signed_area(points: list[list[float]]) -> float:
    total = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return total


def _element_areas(nodes_xy: np.ndarray, elements: np.ndarray) -> np.ndarray:
    coordinates = nodes_xy[elements]
    x = coordinates[:, :, 0]
    y = coordinates[:, :, 1]
    return (
        0.5
        * np.abs(
            np.sum(x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1), axis=1)
        )
    ).astype(np.float64, copy=False)


def _emit_progress(
    progress: ProgressCallback | None,
    *,
    current: int,
    total: int,
    unit: str,
    message: str,
) -> None:
    if progress is not None:
        progress(
            {
                "event": "progress",
                "current": current,
                "total": total,
                "unit": unit,
                "message": message,
                "data": {},
            }
        )


def _emit_assignment_completed(
    progress: ProgressCallback | None,
    assignment: JsonObject,
    *,
    layer: int,
    assignment_index: int,
    duration_ms: int,
    selected_element_count: int,
) -> None:
    if progress is None:
        return
    diagnostic = assignment.get("diagnostic")
    data = diagnostic if isinstance(diagnostic, dict) else {}
    progress(
        {
            "event": "item.completed",
            "stage": "building_3d_mesh",
            "data": {
                "itemType": "assignment",
                "layerIndex": layer + 1,
                "assignmentIndex": assignment_index + 1,
                "sourceRef": data.get("sourceRef"),
                "containerRef": data.get("containerRef"),
                "featureType": data.get("featureType"),
                "geometryType": data.get("geometryType"),
                "operation": data.get("operation"),
                "selectedElementCount": selected_element_count,
                "wallDurationMs": duration_ms,
            },
        }
    )


__all__ = ["MaterialAssignmentResolver"]
