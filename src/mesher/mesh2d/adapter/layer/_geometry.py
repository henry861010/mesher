"""Planar predicates and spatial queries; these never create mesh elements."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from mesher.tolerance import within_distance


def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def area(p):
    p = np.asarray(p) - p[0]
    return float(np.sum(cross(p, np.roll(p, -1, axis=0))) * 0.5)


@dataclass(frozen=True, slots=True)
class _BoxNode:
    lower: np.ndarray
    upper: np.ndarray
    ids: np.ndarray | None = None
    left: _BoxNode | None = None
    right: _BoxNode | None = None


class BoxIndex:
    """Static, balanced AABB tree with each primitive stored in one leaf.

    Local median splits keep distant components from crowding unrelated
    geometry into one query bucket. Bounds include the constructor tolerance;
    queries retain inclusive intersection and return ascending original IDs.
    """
    _LEAF_SIZE = 32

    def __init__(self, lower, upper, tol=0.0):
        self.lower = np.asarray(lower).reshape(-1, 2) - tol
        self.upper = np.asarray(upper).reshape(-1, 2) + tol
        centers = .5 * self.lower + .5 * self.upper
        self._root = (self._build(np.arange(len(self.lower)), centers)
                      if len(self.lower) else None)

    def _build(self, ids, centers):
        lower, upper = self.lower[ids].min(axis=0), self.upper[ids].max(axis=0)
        if len(ids) <= self._LEAF_SIZE:
            return _BoxNode(lower, upper, ids)
        local = centers[ids]
        # Halving before subtraction avoids overflow for distant finite bounds.
        # argmax chooses X when both centroid spans are equal.
        axis = int(np.argmax(.5 * local.max(axis=0) - .5 * local.min(axis=0)))
        values, middle = local[:, axis], len(ids) // 2
        pivot = np.partition(values, middle)[middle]
        left = values < pivot
        needed = middle - int(np.count_nonzero(left))
        # Root IDs start sorted, and Boolean indexing preserves that invariant.
        # Equal centroids therefore split by original ID without another sort,
        # including when every centroid is identical.
        equal = np.flatnonzero(values == pivot)
        left[equal[:needed]] = True
        return _BoxNode(lower, upper, left=self._build(ids[left], centers),
                        right=self._build(ids[~left], centers))

    def _filter_leaf(self, ids, lo, hi):
        return ids[np.all(self.upper[ids] >= lo, axis=1)
                   & np.all(self.lower[ids] <= hi, axis=1)]

    def query(self, lo, hi):
        if self._root is None:
            return np.empty(0, dtype=int)
        lo, hi = np.asarray(lo), np.asarray(hi)
        stack, matches = [self._root], []
        while stack:
            node = stack.pop()
            if (node.upper[0] < lo[0] or node.lower[0] > hi[0]
                    or node.upper[1] < lo[1] or node.lower[1] > hi[1]):
                continue
            if node.ids is None:
                stack.extend((node.right, node.left))
            else:
                found = self._filter_leaf(node.ids, lo, hi)
                if len(found):
                    matches.append(found)
        if not matches:
            return np.empty(0, dtype=int)
        if len(matches) == 1:
            return matches[0]
        return np.sort(np.concatenate(matches))


def on_segment(p, a, b, tol):
    d, v = b - a, p - a
    length = float(np.linalg.norm(d))
    if length <= tol:
        return False
    if tol == 0:
        return bool(cross(d, v) == 0 and 0 <= v @ d <= length * length)
    nearest = a + np.clip(float(v @ d) / (length * length), 0., 1.) * d
    return within_distance(p, nearest, tol)


def segment_contacts(a, b, c, d, tol):
    """Return contact coordinates (two endpoints for an overlap)."""
    u, v = b - a, d - c
    lu, lv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    determinant = float(cross(u, v))
    products = abs(float(u[0] * v[1])) + abs(float(u[1] * v[0]))
    exact = None
    if products and abs(determinant) <= 8 * np.finfo(float).eps * products:
        # Resolve cancellation using the actual supplied floating coordinates.
        # A length tolerance must never turn shallow crossings into parallels.
        ax, ay, bx, by, cx, cy, dx, dy = map(Fraction, map(float, (*a, *b, *c, *d)))
        ux, uy, vx, vy = bx - ax, by - ay, dx - cx, dy - cy
        exact = ux * vy - uy * vx
        determinant = float(exact)
    if determinant != 0 or exact:
        if exact is not None:
            t = float(((cx - ax) * vy - (cy - ay) * vx) / exact)
            s = float(((cx - ax) * uy - (cy - ay) * ux) / exact)
        else:
            t = float(cross(c - a, v)) / determinant
            s = float(cross(c - a, u)) / determinant
        if -tol / lu <= t <= 1 + tol / lu and -tol / lv <= s <= 1 + tol / lv:
            return [a + np.clip(t, 0, 1) * u]
        return []
    if abs(float(cross(u, c - a))) > tol * lu:
        return []
    values = sorted((float((c - a) @ u) / (lu * lu),
                     float((d - a) @ u) / (lu * lu)))
    lo, hi = max(0., values[0]), min(1., values[1])
    if hi < lo - tol / lu:
        return []
    if (hi - lo) * lu <= tol:
        return [a + np.clip((lo + hi) * .5, 0., 1.) * u]
    return [a + lo * u, a + hi * u]


def point_location(point, polygon, tol):
    """-1 outside, 0 boundary, 1 inside a simple loop of either orientation."""
    a, b = polygon, np.roll(polygon, -1, axis=0)
    delta = b - a
    length = np.linalg.norm(delta, axis=1)
    relative = point - a
    projection = np.sum(relative * delta, axis=1)
    fraction = np.divide(projection, length * length, out=np.zeros_like(projection),
                         where=length != 0)
    nearest = a + np.clip(fraction, 0., 1.)[:, None] * delta
    # A zero tolerance is reserved for exact topological/numerical predicates.
    on_boundary = ((cross(delta, relative) == 0) & (projection >= 0)
                   & (projection <= length * length) if tol == 0
                   else within_distance(point, nearest, tol))
    if np.any(on_boundary):
        return 0
    crossing = (a[:, 1] > point[1]) != (b[:, 1] > point[1])
    selected_a, selected_b = a[crossing], b[crossing]
    x = selected_a[:, 0] + (point[1] - selected_a[:, 1]) * (
        selected_b[:, 0] - selected_a[:, 0]) / (selected_b[:, 1] - selected_a[:, 1])
    return 1 if np.count_nonzero(x > point[0]) % 2 else -1


def segment_convex_interval(start, end, polygon, tol):
    """Analytic clipping of a segment against a convex polygon."""
    if area(polygon) < 0:
        polygon = polygon[::-1]
    delta = end - start
    lo, hi = 0., 1.
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = b - a
        offset, slope = float(cross(edge, start - a)), float(cross(edge, delta))
        epsilon = tol * np.linalg.norm(edge)
        if slope == 0:
            if offset < -epsilon:
                return None
        elif slope > 0:
            lo = max(lo, -offset / slope)
        else:
            hi = min(hi, -offset / slope)
        if hi < lo:
            return None
    return (lo, hi) if hi > lo else None


def collinear_interval(start, end, a, b, tol):
    delta = end - start
    length = float(np.linalg.norm(delta))
    if max(abs(float(cross(delta, a - start))), abs(float(cross(delta, b - start)))) > tol * length:
        return None
    values = sorted((float((a - start) @ delta) / length**2,
                     float((b - start) @ delta) / length**2))
    lo, hi = max(0., values[0]), min(1., values[1])
    return (lo, hi) if hi > lo else None


def intervals_cover(required, covering, epsilon):
    merged = []
    for lo, hi in sorted(covering):
        if merged and lo <= merged[-1][1] + epsilon:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return all(hi - lo <= epsilon or any(a <= lo + epsilon and b >= hi - epsilon
                                       for a, b in merged) for lo, hi in required)


def convex_overlap(a, b, tol):
    """Separating-axis test for positive-area overlap of convex polygons."""
    for polygon, other in ((a, b), (b, a)):
        if area(polygon) < 0:
            polygon = polygon[::-1]
        edges = np.roll(polygon, -1, axis=0) - polygon
        sides = cross(edges[:, None, :], other[None, :, :] - polygon[:, None, :])
        if np.any(sides.max(axis=1) <= tol * np.linalg.norm(edges, axis=1)):
            return False
    return True
