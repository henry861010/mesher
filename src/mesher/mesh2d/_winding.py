"""Classify planar element winding before committing connectivity."""

from fractions import Fraction

import numpy as np

from ..tolerance import GeometryToleranceError


def _exact_sign(points):
    """Resolve concavity, crossings, collinearity and uncertain float signs."""
    points = [tuple(map(Fraction, point)) for point in points]
    if len(set(points)) != len(points):
        raise ValueError("coincident vertices or overlapping edges")

    def turn(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    if len(points) == 4:
        for i in (0, 1):
            a, b, c, d = points[i:] + points[:i]
            if (turn(a, b, c) * turn(a, b, d) <= 0 and
                    turn(c, d, a) * turn(c, d, b) <= 0 and
                    all(max(min(a[k], b[k]), min(c[k], d[k])) <=
                        min(max(a[k], b[k]), max(c[k], d[k])) for k in (0, 1))):
                raise ValueError("intersecting or overlapping edges")
    area = sum(turn(points[0], points[i], points[i + 1]) for i in range(1, len(points) - 1))
    if not area:
        raise ValueError("zero area")
    return 1 if area > 0 else -1


def ccw_elements(nodes, elements):
    """Return original rows when possible, copying only to reverse CW rows."""
    result = elements
    for start in range(0, len(elements), 32768):
        block = elements[start:start + 32768]
        points = nodes[block, :2]
        finite = np.isfinite(points).all(axis=(1, 2))
        if not finite.all():
            raise GeometryToleranceError("element references nonfinite XY coordinates", code="invalid_input_geometry",
                                         element_indices=(start + np.flatnonzero(~finite)).tolist())
        triangle = block[:, 2] == block[:, 3]
        # Consistently signed turns prove strict convexity and simple winding.
        # Tri3 has one independent turn. Ignore its two padded zero turns.
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            edges = np.roll(points, -1, axis=1) - points
            following = np.roll(edges, -1, axis=1)
            left = edges[..., 0] * following[..., 1]
            right = edges[..., 1] * following[..., 0]
            det = left - right
            bound = (8 * np.finfo(float).eps * (abs(left) + abs(right))
                     + np.finfo(float).tiny)
            positive, negative = det > bound, det < -bound
        ccw = np.where(triangle, positive[:, 0], positive.all(axis=1))
        cw = np.where(triangle, negative[:, 0], negative.all(axis=1))
        invalid = {}
        for i in np.flatnonzero(~(cw | ccw)):
            try:
                cw[i] = _exact_sign(points[i, :3 if triangle[i] else 4]) < 0
            except ValueError as error:
                invalid[int(start + i)] = str(error)
        if invalid:
            raise GeometryToleranceError(next(iter(invalid.values())), code="invalid_input_geometry",
                                         element_indices=list(invalid), reasons=list(invalid.values()))
        if cw.any():
            if result is elements:
                result = elements.copy()
            destination = result[start:start + len(block)]
            triangles = cw & triangle
            quads = cw & ~triangle
            destination[triangles] = block[triangles][:, [0, 2, 1, 1]]
            destination[quads] = block[quads][:, [0, 3, 2, 1]]
    return result
