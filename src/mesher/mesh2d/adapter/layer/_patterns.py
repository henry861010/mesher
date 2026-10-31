"""Finite axis-aligned guides, using absolute axial intervals throughout.

The source-normalized frame is retained, but guide endpoints never define an
intersection parameter. Intersections are computed on the local mesh edge or
cell, so extending an irrelevant tail cannot destroy a nearby intersection.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import hypot

import numpy as np

from ._geometry import cross
from ._types import QuadLayerError


PATTERN_RANGE_LIMIT = 1_000_000.0


@dataclass(frozen=True)
class AxisPattern:
    axis: int
    support: float
    lo: float
    hi: float
    ids: tuple[int, ...] = ()

    @classmethod
    def from_points(cls, points, ids=()):
        points = np.asarray(points, dtype=float)
        axis = int(np.argmax(np.abs(points[1] - points[0])))
        return cls(axis, float(points[0, 1 - axis]),
                   float(points[:, axis].min()), float(points[:, axis].max()), tuple(ids))

    def point(self, value):
        point = np.full(2, self.support, dtype=float)
        point[self.axis] = value
        return point

    @property
    def points(self):
        return np.asarray((self.point(self.lo), self.point(self.hi)))

    def bounds(self, tol=0.0):
        return self.point(self.lo) - tol, self.point(self.hi) + tol


def _finite(values):
    if not np.all(np.isfinite(values)):
        raise QuadLayerError("pattern geometry exceeds the supported numeric range",
                             code="numeric_range_exceeded")


def _canonical_point(point, a, b, tol):
    """Preserve supplied endpoints only when the full contact is close enough."""
    for endpoint in (a, b):
        if hypot(*(point - endpoint)) <= tol:
            return endpoint.copy()
    return point


def pattern_contacts(pattern, a, b, tol):
    """Contacts with a local segment, preserving exact shared endpoints.

    Near-collinearity and finite-endpoint tolerance match edge coverage. For
    transverse edges, only the local edge's parameter is used; normal distance
    alone must not snap a shallow crossing to an unrelated endpoint.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    _finite((pattern.support, pattern.lo, pattern.hi))
    _finite((a, b))
    axis, other = pattern.axis, 1 - pattern.axis
    offsets = np.asarray((a[other], b[other])) - pattern.support
    if np.all(np.abs(offsets) <= tol):
        lo = max(pattern.lo, float(min(a[axis], b[axis])))
        hi = min(pattern.hi, float(max(a[axis], b[axis])))
        if hi < lo - tol:
            return []
        values = (lo + (hi - lo) * .5,) if hi - lo <= tol else (lo, hi)
        return [_canonical_point(pattern.point(value), a, b, tol) for value in values]
    with np.errstate(over="ignore", invalid="ignore"):
        delta = b - a
    _finite(delta)
    length = hypot(*delta)
    _finite(length)
    if length <= tol or delta[other] == 0:
        return []
    # Reject remote supports before dividing by a potentially shallow slope.
    if (pattern.support < min(a[other], b[other]) - tol
            or pattern.support > max(a[other], b[other]) + tol):
        return []
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        fraction = float((pattern.support - a[other]) / delta[other])
    _finite(fraction)
    if not -tol / length <= fraction <= 1.0 + tol / length:
        return []
    if fraction <= 0:
        point = a.copy()
    elif fraction >= 1:
        point = b.copy()
    else:
        point = (a + fraction * delta if fraction <= .5
                 else b - (1.0 - fraction) * delta)
        point[other] = pattern.support
        point = _canonical_point(point, a, b, tol)
    _finite(point)
    if not pattern.lo - tol <= point[axis] <= pattern.hi + tol:
        return []
    return [point]


def covered_intervals(pattern, edges, tol):
    """Absolute axial intervals covered by collinear finite mesh edges."""
    edges = np.asarray(edges, dtype=float)
    _finite((pattern.support, pattern.lo, pattern.hi))
    _finite(edges)
    if not len(edges):
        return []
    collinear = np.all(np.abs(edges[:, :, 1 - pattern.axis] - pattern.support) <= tol,
                       axis=1)
    values = edges[collinear, :, pattern.axis]
    if not len(values):
        return []
    lower = np.maximum(pattern.lo, values.min(axis=1))
    upper = np.minimum(pattern.hi, values.max(axis=1))
    good = upper > lower
    return list(zip(lower[good].tolist(), upper[good].tolist()))


def occupied_intervals(pattern, cells, tol):
    """Clip a finite guide against convex cells of either orientation.

    Each clipping interval starts inside the cell's axial bounding box. Roots
    are interpolated only between opposite signs on that local interval, never
    between distant guide endpoints. Parallel-edge tolerance follows the
    existing convex clipping contract.
    """
    cells = np.asarray(cells, dtype=float)
    _finite((pattern.support, pattern.lo, pattern.hi))
    _finite(cells)
    if not len(cells):
        return []
    lower = np.maximum(pattern.lo, cells[:, :, pattern.axis].min(axis=1))
    upper = np.minimum(pattern.hi, cells[:, :, pattern.axis].max(axis=1))
    selected = upper > lower
    cells, lower, upper = cells[selected], lower[selected], upper[selected]
    if not len(cells):
        return []
    with np.errstate(over="ignore", invalid="ignore"):
        shifted = cells - cells[:, :1]
        signs = np.sum(cross(shifted, np.roll(shifted, -1, axis=1)), axis=1)
    _finite(shifted)
    _finite(signs)
    cells = cells.copy()
    cells[signs < 0] = cells[signs < 0, ::-1]
    valid = np.ones(len(cells), dtype=bool)
    axis, other = pattern.axis, 1 - pattern.axis
    for index in range(cells.shape[1]):
        a, b = cells[:, index], cells[:, (index + 1) % cells.shape[1]]
        with np.errstate(over="ignore", invalid="ignore"):
            edge = b - a
            first = np.empty_like(a)
            first[:, axis] = lower - a[:, axis]
            first[:, other] = pattern.support - a[:, other]
            last = first.copy()
            last[:, axis] = upper - a[:, axis]
            first_side, last_side = cross(edge, first), cross(edge, last)
        _finite(edge)
        _finite(first)
        _finite(last)
        _finite(first_side)
        _finite(last_side)
        parallel = edge[:, other] == 0
        with np.errstate(over="ignore", invalid="ignore"):
            epsilon = tol * np.hypot(edge[:, 0], edge[:, 1])
        _finite(epsilon)
        valid &= ~(parallel & (first_side < -epsilon))
        valid &= ~(~parallel & (first_side < 0) & (last_side < 0))
        crossing = valid & ~parallel & ((first_side < 0) != (last_side < 0))
        indices = np.flatnonzero(crossing)
        if len(indices):
            left, right = np.abs(first_side[indices]), np.abs(last_side[indices])
            scale = np.maximum(left, right)
            left, right = left / scale, right / scale
            fraction = left / (left + right)
            lo, hi = lower[indices], upper[indices]
            contact = np.where(fraction <= .5, lo + fraction * (hi - lo),
                               hi - (1.0 - fraction) * (hi - lo))
            _finite(contact)
            entering = first_side[indices] < 0
            lower[indices[entering]] = contact[entering]
            upper[indices[~entering]] = contact[~entering]
        valid &= upper > lower
    return list(zip(lower[valid].tolist(), upper[valid].tolist()))
