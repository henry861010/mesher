"""Shared planar keep-out-zone geometry for translation and meshing."""

from __future__ import annotations

import math
from typing import Any

JsonObject = dict[str, Any]
FEATURE_FIELDS = ("bumps", "circuits", "vias")


def koz_value(item: JsonObject) -> float:
    """Read an optional, finite non-negative XY inset distance."""
    try:
        value = float(item.get("koz", 0.0))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("koz must be a finite non-negative number") from error
    if not math.isfinite(value) or value < 0:
        raise ValueError("koz must be a finite non-negative number")
    return value


def koz_tolerance(container: JsonObject, default: float) -> float:
    """Keep positive KOZ widths distinct from their original boundaries."""
    tolerance = default
    for field in FEATURE_FIELDS:
        for item in container.get(field) or []:
            value = koz_value(item)
            if value > 0:
                tolerance = min(tolerance, value / 100.0)
    for child in container.get("children") or []:
        tolerance = min(tolerance, koz_tolerance(child, default))
    return tolerance


def inset_face(face: JsonObject, distance: float) -> JsonObject | None:
    """Return the eligible region inside a KOZ, or None if it is empty.

    Polygon loops use even/odd nesting, independent of order or winding.
    Shapely is loaded only for positive polygon insets, at the process-flow
    boundary; reusable mesh2d/mesh3d modules do not depend on it.
    """
    if distance == 0:
        return face
    kind, dim = face["type"], face["dim"]
    if kind == "BOX":
        x1, y1, x2, y2 = dim
        left, bottom = min(x1, x2) + distance, min(y1, y2) + distance
        right, top = max(x1, x2) - distance, max(y1, y2) - distance
        if left >= right or bottom >= top:
            return None
        return {"type": "BOX", "dim": [left, bottom, right, top]}
    if kind == "CIRCLE":
        x, y, radius = dim
        if radius <= distance:
            return None
        return {"type": "CIRCLE", "dim": [x, y, radius - distance]}
    if kind != "POLYGON":
        raise ValueError(f"KOZ is not supported for face type {kind}")

    from shapely.geometry import Polygon
    from shapely.geometry.polygon import orient

    region = Polygon()
    for loop in dim:
        polygon = Polygon(loop)
        if not polygon.is_valid:
            raise ValueError("Polygon KOZ requires valid, non-self-intersecting loops")
        region = region.symmetric_difference(polygon)
    # Mitre joins preserve straight corners, including rectilinear holes.
    region = region.buffer(-distance, join_style="mitre")
    if region.is_empty:
        return None
    polygons = [region] if region.geom_type == "Polygon" else list(region.geoms)
    polygons.sort(key=lambda polygon: (*polygon.bounds, -polygon.area))
    loops = []
    for polygon in polygons:
        polygon = orient(polygon, sign=1.0)
        loops.append(_canonical_loop(polygon.exterior.coords))
        holes = [_canonical_loop(ring.coords) for ring in polygon.interiors]
        loops.extend(sorted(holes))
    return {"type": "POLYGON", "dim": loops}


def _canonical_loop(coordinates: Any) -> list[list[float]]:
    points = [[float(x), float(y)] for x, y in list(coordinates)[:-1]]
    start = min(range(len(points)), key=lambda index: points[index])
    return points[start:] + points[:start]
