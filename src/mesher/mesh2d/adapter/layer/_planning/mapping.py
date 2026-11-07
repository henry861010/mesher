"""Geometry checks for existing-node section interfaces."""

from __future__ import annotations

import numpy as np

from .._geometry import point_location, segment_contacts


def same_point(a, b, tol):
    return bool(np.linalg.norm(np.asarray(a) - b) <= tol)


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
    if (point_location(midpoint, data["outer"], tol) != 1
            or point_location(midpoint, data["inner"], tol) != -1):
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
