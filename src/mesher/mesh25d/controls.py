"""Resolve controls for the retained geometry and collect mesher interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..mesh2d.controls import XYRegionControl
from ..mesh3d.controls import ZControl
from ..meshcontrol import get_control_obj
from ..meshcontrol.methods import _find_target_geometry


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
) -> ResolvedControls:
    # Validate against the original model so a typo cannot disappear when a
    # symmetry cut removes its purported target. Re-resolve retained references
    # because a surviving container's bounding subtree may have changed.
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
            z.extend(obj.get_25D_z_controls(global_element_size=global_element_size))
        except ValueError as exc:
            raise ValueError(f"{context}: {exc}") from exc
        xy.extend(xy_constraints)
        if xy_constraints:
            sources.append(context)
    return ResolvedControls(xy, z, sources)
