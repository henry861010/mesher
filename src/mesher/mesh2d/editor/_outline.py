"""Split ordered boundaries into directed logical edges."""

import numpy as np

from ..utils.topology import _get_boundary


def _loop_windings(mesh, loops):
    """Find each loop's owning cell without classifying nested polygons."""
    node_count = mesh.node_count
    seeds = np.array([int(loop[0]) * node_count + int(loop[1]) for loop in loops])
    order = np.argsort(seeds)
    seeds = seeds[order]
    owners = np.full(len(loops), -1, dtype=np.int64)
    for begin in range(0, mesh.element_count, 65_536):
        cells = mesh.elements[begin:begin + 65_536].astype(np.int64)
        keys = (cells * node_count + np.roll(cells, -1, axis=1)).ravel()
        positions = np.searchsorted(seeds, keys)
        candidates = np.flatnonzero(positions < len(seeds))
        matches = candidates[seeds[positions[candidates]] == keys[candidates]]
        owners[order[positions[matches]]] = begin + matches // 4

    points = mesh.nodes[mesh.elements[owners], :2]
    points = points - points[:, :1]
    scales = np.max(np.abs(points), axis=(1, 2))
    if not np.all(np.isfinite(scales)) or np.any(scales == 0):
        raise ValueError("boundary cells must have finite, nonzero XY area")
    # Only the sign matters; normalize to avoid squaring world-scale lengths.
    points = points / scales[:, None, None]
    following = np.roll(points, -1, axis=1)
    areas = np.sum(points[:, :, 0] * following[:, :, 1]
                   - following[:, :, 0] * points[:, :, 1], axis=1)
    if not np.all(np.isfinite(areas)) or np.any(areas == 0):
        raise ValueError("boundary cells must have finite, nonzero XY area")
    return areas


def _outlines_by_edge(mesh, angle_tolerance):
    if isinstance(angle_tolerance, (bool, np.bool_)):
        raise ValueError("angle_tolerance must be finite and in [0, pi/2) radians")
    try:
        tolerance = float(angle_tolerance)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("angle_tolerance must be finite and in [0, pi/2) radians") from error
    if not np.isfinite(tolerance) or not 0 <= tolerance < np.pi / 2:
        raise ValueError("angle_tolerance must be finite and in [0, pi/2) radians")

    loops = _get_boundary(mesh)
    if not loops:
        return []

    outlines = []
    for loop, winding in zip(loops, _loop_windings(mesh, loops)):
        # Reversing CCW cell boundaries also gives holes their required CCW
        # direction, and handles disconnected components of either winding.
        if winding > 0:
            loop = loop[::-1]
        points = mesh.nodes[loop, :2]
        if not np.all(np.isfinite(points)):
            raise ValueError("boundary nodes must have finite XY coordinates")
        vectors = np.roll(points, -1, axis=0) - points
        lengths = np.hypot(vectors[:, 0], vectors[:, 1])
        if not np.all(np.isfinite(lengths)) or np.any(lengths == 0):
            raise ValueError("boundary segments must have finite, nonzero XY length")
        directions = vectors / lengths[:, None]
        previous = np.roll(directions, 1, axis=0)
        turns = previous[:, 0] * directions[:, 1] - previous[:, 1] * directions[:, 0]
        dot = np.sum(previous * directions, axis=1)
        corners = np.flatnonzero((np.abs(turns) > np.sin(tolerance)) | (dot <= 0))
        if len(corners) < 3:
            raise ValueError("boundary has fewer than three logical edges; reduce angle_tolerance")

        # Choose the lowest directed corner edge for a stable starting point.
        first = np.lexsort((loop[(corners + 1) % len(loop)], loop[corners]))[0]
        corners = np.roll(corners, -first)
        edges = []
        for start, end in zip(corners, np.roll(corners, -1)):
            if end <= start:
                end += len(loop)
            edges.append(loop[np.arange(start, end + 1) % len(loop)].copy())
        outlines.append(edges)
    return outlines
