"""Resolve controls for the retained geometry and collect mesher interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..mesh2d.controls import XYRegionControl
from ..mesh3d.controls import ZControl
from ..meshcontrol import get_control_obj
from ..meshcontrol.methods import _find_target_geometry
from ..tolerance import GeometryToleranceError, canonicalize_points, resolve_tolerance
from .geometry.patterns import _collect_faces
from .geometry.normalization import _validate_polygon_topology


@dataclass(frozen=True)
class ResolvedControls:
    xy: list[XYRegionControl]
    z: list[ZControl]
    xy_sources: list[str]


def resolve_controls(
    original: dict[str, Any],
    retained: dict[str, Any],
    entries: list[dict[str, Any]],
    *,
    global_element_size: float,
    tolerance=None,
) -> ResolvedControls:
    # Validate against the original model so a typo cannot disappear when a
    # symmetry cut removes its purported target. Re-resolve retained references
    # because a surviving container's bounding subtree may have changed.
    policy = resolve_tolerance(tolerance)
    original_objects = get_control_obj(original, entries)
    objects = []
    for index, entry in enumerate(entries):
        if entry["coordinateMode"] == "absolute":
            active = [obj for obj in original_objects if obj.source_index == index]
        else:
            if not _find_target_geometry(entry, retained):
                continue
            try:
                active = get_control_obj(retained, [entry])
            except GeometryToleranceError as exc:
                _raise_control_error(exc, f"controls[{index}]", index, policy)
            except ValueError as exc:
                raise ValueError(str(exc).replace("controls[0]", f"controls[{index}]", 1)) from exc
        for obj in active:
            obj.source_index = index
        objects.extend(active)

    xy, z, sources = [], [], []
    for obj in objects:
        context = f"controls[{obj.source_index}] ({obj.label!r})"
        try:
            xy_constraints = obj.get_25D_xy_controls()
            if xy_constraints and obj.element_size <= policy.absolute:
                raise GeometryToleranceError(
                    "control element size must exceed geometryTolerance",
                    code="feature_below_resolution", stage="control_normalization",
                    measured_distance=float(obj.element_size), tolerance=policy.absolute,
                )
            z.extend(obj.get_25D_z_controls(tolerance=policy, global_element_size=global_element_size,
                                           preserve_z=True))
        except GeometryToleranceError as exc:
            _raise_control_error(exc, context, obj.source_index, policy)
        except ValueError as exc:
            raise ValueError(f"{context}: {exc}") from exc
        xy.extend(xy_constraints)
        sources.extend([context] * len(xy_constraints))
    if xy:
        _normalize_xy_controls(xy, sources, retained, policy)
    return ResolvedControls(xy, z, sources)


def _raise_control_error(error, source, index, policy):
    context = dict(error.context)
    if "source" in context:
        context["original_source"] = context["source"]
    context.update(source=source, control_index=index)
    context.setdefault("tolerance", policy.absolute)
    raise GeometryToleranceError(error.message, code=error.code, **context) from error


def _normalize_xy_controls(controls, sources, container, policy):
    """Attach movable control cuts to fixed, normalized material boundaries."""
    anchors = []
    for face in _collect_faces(container):
        kind, dim = face["type"], face["dim"]
        if kind == "BOX":
            anchors.extend((dim[:2], dim[2:]))
        elif kind == "POLYGON":
            anchors.extend(point for loop in dim for point in loop)
        else:
            x, y, radius = dim
            anchors.extend(((x - radius, y - radius), (x + radius, y + radius)))
    points = list(anchors)
    records = []
    for control, source in zip(controls, sources):
        face = control["region"]
        start = len(points)
        if control["element_size"] <= policy.absolute:
            raise GeometryToleranceError(
                "control element size must exceed geometryTolerance",
                code="feature_below_resolution", stage="control_normalization", source=source,
                measured_distance=float(control["element_size"]), tolerance=policy.absolute,
            )
        if face["type"] == "BOX":
            points.extend((face["dim"][:2], face["dim"][2:]))
            records.append((start, None))
        else:
            sizes = [len(loop) for loop in face["dim"]]
            points.extend(point for loop in face["dim"] for point in loop)
            records.append((start, sizes))
    policy.check_resolution(points, stage="control_normalization", source=sources)
    normalized = canonicalize_points(
        points, policy, fixed=[True] * len(anchors) + [False] * (len(points) - len(anchors)),
        stage="control_normalization", source=sources,
    )
    for control, source, (start, sizes) in zip(controls, sources, records):
        if sizes is None:
            a, b = normalized[start:start + 2]
            if np.any(np.abs(b - a) <= policy.absolute):
                raise GeometryToleranceError(
                    "normalization would collapse a control region",
                    code="feature_below_resolution", stage="control_normalization", source=source,
                    measured_distances=np.abs(b - a).tolist(), tolerance=policy.absolute,
                )
            control["region"]["dim"] = [*a, *b]
        else:
            loops, cursor = [], start
            for size in sizes:
                loop = normalized[cursor:cursor + size]
                cursor += size
                loops.append(loop.tolist())
            _validate_polygon_topology(control["region"]["dim"], loops, policy, source,
                                       stage="control_normalization")
            control["region"]["dim"] = loops
