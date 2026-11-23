"""Local Quad4 quality checks shared by unit construction and layer smoothing."""

from __future__ import annotations

import numpy as np


SHAPE_EPS = 64 * np.finfo(float).eps


def quad_shape(cells, *, min_edge=None, span=None):
    """Return each quad's edge aspect ratio and least signed scaled Jacobian.

    For a proposed move, mark unresolved edges and turns with an invalid
    Jacobian. Unit builders validate turns at the patch span, layers at the
    individual cell span.
    """
    edges = np.roll(cells, -1, axis=1) - cells
    scale = np.ptp(cells, axis=1).max(axis=1) if span is None else np.full(len(cells), span)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        edges /= scale[:, None, None]
        lengths = np.hypot(edges[..., 0], edges[..., 1])
        following = np.roll(edges, -1, axis=1)
        turns = edges[..., 0] * following[..., 1] - edges[..., 1] * following[..., 0]
        shortest = lengths.min(axis=1)
        aspect = lengths.max(axis=1) / shortest
        jacobian = (turns / (lengths * np.roll(lengths, -1, axis=1))).min(axis=1)
        if min_edge is not None:
            valid = (shortest > min_edge / scale) & (turns.min(axis=1) > SHAPE_EPS)
            jacobian[~valid] = -np.inf
    return aspect, jacobian


def improves_ring(aspect, jacobian, trial_aspect, trial_jacobian):
    """Both ring extrema must hold, and at least one must improve beyond roundoff."""
    return (trial_jacobian > 0
            and trial_aspect <= aspect * (1 + SHAPE_EPS)
            and trial_jacobian >= jacobian - SHAPE_EPS
            and (trial_aspect < aspect * (1 - SHAPE_EPS)
                 or trial_jacobian > jacobian + SHAPE_EPS))
