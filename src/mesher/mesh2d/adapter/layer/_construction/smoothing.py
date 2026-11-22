"""One guarded Laplacian pass over private nodes of selected Quad4 units."""

from __future__ import annotations

import numpy as np


_THRESHOLD = 4.0
_EPS = 64 * np.finfo(float).eps


def quad_shape(cells, *, min_edge=None):
    """Return each quad's edge aspect ratio and least signed scaled Jacobian.

    For a proposed move, mark unresolved edges and turns with an invalid
    Jacobian. The turn floor uses the same cell span as the layer validator.
    """
    edges = np.roll(cells, -1, axis=1) - cells
    scale = np.ptp(cells, axis=1).max(axis=1)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        edges /= scale[:, None, None]
        lengths = np.hypot(edges[..., 0], edges[..., 1])
        following = np.roll(edges, -1, axis=1)
        turns = edges[..., 0] * following[..., 1] - edges[..., 1] * following[..., 0]
        shortest = lengths.min(axis=1)
        aspect = lengths.max(axis=1) / shortest
        jacobian = (turns / (lengths * np.roll(lengths, -1, axis=1))).min(axis=1)
        if min_edge is not None:
            valid = (shortest > min_edge / scale) & (turns.min(axis=1) > _EPS)
            jacobian[~valid] = -np.inf
    return aspect, jacobian


def smooth_selected_units(nodes, quads, units, *, element_offset, original_count, tol):
    """Update represented XY coordinates, returning original XY for rollback."""
    originals = {}
    for unit in units:
        start = unit["element_start"] - element_offset
        stop = unit["element_end"] - element_offset
        if stop - start == 1 or unit["quality"] <= _THRESHOLD:
            continue
        cells = quads[start:stop]
        aspect, jacobian = quad_shape(nodes[cells, :2])
        if not np.all(jacobian > 0):
            continue
        scores = np.maximum(aspect, 1 / jacobian)
        poor = np.flatnonzero(scores > _THRESHOLD)
        if not len(poor):
            continue
        # Moves use previously updated neighbors. Visit the first equally
        # poor quad when conversion of length units perturbs a score by ulps.
        poor_scores = scores[poor]
        worst = np.argmax(poor_scores >= poor_scores.max() * (1 - _EPS))
        poor[0], poor[worst] = poor[worst], poor[0]
        fixed = set(unit["_polygon_ids"])
        fixed.update(unit.get("_fixed_interior", ()))
        # Each candidate owns its complete ring, including initially good cells.
        rings = {node: [] for quad in cells[poor].tolist() for node in quad
                 if node >= original_count and node not in fixed}
        if not rings:
            continue
        for row, quad in enumerate(cells.tolist()):
            for corner, node in enumerate(quad):
                if node in rings:
                    rings[node].append((row, corner))

        for node_id, ring in rings.items():
            rows, corners = np.asarray(ring).T
            ring_aspect = aspect[rows].max()
            ring_jacobian = jacobian[rows].min()
            if ring_aspect <= _THRESHOLD and ring_jacobian >= 1 / _THRESHOLD:
                continue
            # In a CCW interior ring each outgoing edge appears once. Incoming
            # edges give the same neighbors. Diagonal vertices play no part.
            neighbors = cells[rows, (corners + 1) % 4]
            point = nodes[node_id, :2]
            displacement = (nodes[neighbors, :2] - point).mean(axis=0)
            ring_xy = nodes[cells[rows], :2]
            local_rows = np.arange(len(rows))
            for step in (.5, .25):
                candidate = point + step * displacement
                ring_xy[local_rows, corners] = candidate
                trial_aspect, trial_jacobian = quad_shape(ring_xy, min_edge=tol)
                worst_aspect = trial_aspect.max()
                least_jacobian = trial_jacobian.min()
                if not (worst_aspect <= ring_aspect * (1 + _EPS)
                        and least_jacobian >= ring_jacobian - _EPS):
                    continue
                if worst_aspect < ring_aspect * (1 - _EPS) or least_jacobian > ring_jacobian + _EPS:
                    originals[node_id] = point.copy()
                    nodes[node_id, :2] = candidate
                    aspect[rows], jacobian[rows] = trial_aspect, trial_jacobian
                    break
    return originals
