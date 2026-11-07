"""Return ordered boundary loops with existing mesh on their right."""

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


def _outlines(mesh):
    loops = _get_boundary(mesh)
    if not loops:
        return []

    outlines = []
    for loop, winding in zip(loops, _loop_windings(mesh, loops)):
        if winding > 0:
            loop = loop[::-1]
        points = mesh.nodes[loop, :2]
        if not np.all(np.isfinite(points)):
            raise ValueError("boundary nodes must have finite XY coordinates")
        vectors = np.roll(points, -1, axis=0) - points
        lengths = np.hypot(vectors[:, 0], vectors[:, 1])
        if not np.all(np.isfinite(lengths)) or np.any(lengths == 0):
            raise ValueError("boundary segments must have finite, nonzero XY length")
        first = np.lexsort((np.roll(loop, -1), loop))[0]
        outlines.append(np.roll(loop, -first).astype(np.int64, copy=True))
    return outlines
