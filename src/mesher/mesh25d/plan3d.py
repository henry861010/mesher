"""3D mesh stage: plan Z layers and extrude the 2D mesh with Dragger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..mesh2d.model import Mesh2D
from ..mesh3d.controls import LayerInfo, ZControl
from ..mesh3d.extrusion import Dragger
from ..mesh3d.model import Mesh3D
from .progress import ProgressCallback


@dataclass(frozen=True)
class LayerPlan:
    layer_infos: list[LayerInfo]


def plan_layers(
    assignments: list[dict[str, Any]],
    controls: list[ZControl],
    *,
    element_size: float,
) -> LayerPlan:
    """Combine boundaries without interpreting control method names.

    Points add a cut only. Intervals refine by the smallest active size and
    are clipped to the material event extent. Material event ordering and
    payloads remain unchanged, including events on the terminal boundary.
    """
    if not assignments:
        return LayerPlan([])
    events: dict[float, list[dict[str, Any]]] = {}
    for assignment in sorted(assignments, key=lambda item: (item["z"], item["type"])):
        events.setdefault(assignment["z"], []).append(assignment)
    model_start, model_end = min(events), max(events)
    boundaries = set(events)
    intervals = []
    for control in controls:
        start, end = control["start"], control["end"]
        if start == end:
            if model_start <= start <= model_end:
                boundaries.add(start)
        else:
            start, end = max(start, model_start), min(end, model_end)
            if start < end:
                boundaries.update((start, end))
                intervals.append((start, end, control["element_size"]))

    infos = []
    ordered = sorted(boundaries)
    for index, z in enumerate(ordered):
        size = element_size
        if index < len(ordered) - 1:
            end = ordered[index + 1]
            size = min([size, *(value for start, stop, value in intervals
                                if start <= z and end <= stop)])
        infos.append(LayerInfo(z=z, element_size=size, assignments=events.get(z, [])))
    return LayerPlan(infos)


def build_3d_mesh(
    mesh: Mesh2D,
    plan: LayerPlan,
    *,
    progress: ProgressCallback | None = None,
) -> Mesh3D:
    dragger = Dragger()
    dragger.set_2D(mesh.nodes, _clockwise_elements(mesh.elements))
    return dragger.build(plan.layer_infos, progress=progress)


def _clockwise_elements(elements: Any) -> np.ndarray:
    source = np.asarray(elements)
    if source.ndim != 2 or source.shape[1] != 4:
        raise ValueError("mesh2D.elements must have shape (m, 4).")

    clockwise = np.empty_like(source)
    triangle_mask = source[:, 2] == source[:, 3]
    clockwise[~triangle_mask] = source[~triangle_mask][:, [0, 3, 2, 1]]
    clockwise[triangle_mask] = source[triangle_mask][:, [0, 2, 1, 1]]
    return clockwise
