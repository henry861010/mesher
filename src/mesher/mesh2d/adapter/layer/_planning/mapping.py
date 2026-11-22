"""Geometry checks for existing-node section interfaces."""

from __future__ import annotations

import numpy as np

from .._registry import _BoundedCoordinates
from .._geometry import point_location, segment_contacts


def same_point(a, b, tol):
    return bool(np.linalg.norm(np.asarray(a) - b) <= tol)


def corner_distances(points, source, data):
    """Rank corner contacts using only axes that distinguish candidates."""
    points = np.asarray(points)
    axes = np.any(points != points[0], axis=0)
    coordinates = points[:, axes]
    source = source[axes]
    distances = np.linalg.norm(coordinates - source, axis=1)
    # A constant normal coordinate adds the same squared distance to every
    # candidate. Its world offset must not turn distinct tangential gaps into
    # ties. Keep roundoff from the varying coordinates for transformed ties.
    local_extent = max(np.abs(coordinates).max(initial=0.), np.abs(source).max(initial=0.))
    origin_ulp = np.abs(np.spacing(data["origin"][axes])).max(initial=0.)
    roundoff = 4 * origin_ulp / data["scale"] + 8 * np.finfo(float).eps * local_extent
    return distances, min(data["tol"], roundoff)


def compatible_interfaces(first, second, tol):
    """Interfaces may share endpoints, but cannot cross or overlap."""
    for point in segment_contacts(*first, *second, tol):
        if (not any(same_point(point, p, tol) for p in first)
                or not any(same_point(point, p, tol) for p in second)):
            return False
    return True


def valid_interface(source, target, data, interfaces=()):
    """An open source-to-front segment must stay strictly inside the band."""
    tol = data["tol"]
    if same_point(source, target, tol):
        return False
    midpoint = (source + target) / 2
    if (point_location(midpoint, data["outer"], data["numeric_tol"]) != 1
            or point_location(midpoint, data["inner"], data["numeric_tol"]) != -1):
        return False
    for loop in (data["source"], data["front"]):
        for a, b in zip(loop, np.roll(loop, -1, axis=0)):
            for point in segment_contacts(source, target, a, b, tol):
                if not same_point(point, source, tol) and not same_point(point, target, tol):
                    return False
    return all(compatible_interfaces((source, target), interface, tol)
               for interface in interfaces)


def target_node_id(target, data):
    """Resolve a supplied target ID; planned automatic contacts have no ID."""
    if "outline_anchors" not in data:
        return None
    distances = np.linalg.norm(data["outline_anchors"] - target, axis=1)
    index = int(np.argmin(distances))
    return int(data["outline_anchor_ids"][index]) if distances[index] <= data["tol"] else None


def canonical_target(point, data):
    """Resolve movable contacts before sampling or evaluating their direction."""
    if "outline_anchors" not in data:
        return point
    index = data.get("_target_coordinate_index")
    if index is None:
        index = _BoundedCoordinates(data["tol"], stage="section_partition")
        for owner, anchor in zip(data["outline_anchor_ids"], data["outline_anchors"]):
            index.register(anchor, int(owner), fixed=True)
        data["_target_coordinate_index"] = index
    owner = index.match(point)
    return point if owner is None else data["xy"][owner].copy()


def dense_angle(source, target, tangent, normal, role, limit):
    """Check the directed angular interval using arithmetic roundoff only."""
    delta = np.asarray(target) - source
    along, outward = float(delta @ tangent), float(delta @ normal)
    angle = float(np.degrees(np.arctan2(along, outward)))
    # Local-coordinate subtraction, dot products, and atan2 rounding.
    error = 16 * np.finfo(float).eps * max(float(np.max(np.abs((source, target)))),
                                           float(np.hypot(along, outward)))
    angular_error = float(np.degrees(np.arctan2(error, max(outward, error))))
    lower, upper = (-limit, 0.) if role == "begin" else (0., limit)
    return outward > 0 and lower - angular_error <= angle <= upper + angular_error, angle
