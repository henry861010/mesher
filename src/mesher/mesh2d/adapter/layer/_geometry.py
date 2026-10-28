"""Planar predicates and spatial queries; these never create mesh elements."""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from math import floor

import numpy as np


def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def area(p):
    p = np.asarray(p) - p[0]
    return float(np.sum(cross(p, np.roll(p, -1, axis=0))) * 0.5)


class BoxIndex:
    """Static box grid with bounded storage for unusually long primitives."""
    def __init__(self, lower, upper, tol=0.0):
        self.lower = np.asarray(lower).reshape(-1, 2) - tol
        self.upper = np.asarray(upper).reshape(-1, 2) + tol
        self.bins = defaultdict(list)
        self.large = []
        if not len(self.lower):
            self.origin, self.step = np.zeros(2), np.ones(2)
            return
        self.origin = self.lower.min(axis=0)
        extent = self.upper.max(axis=0) - self.origin
        self.step = np.maximum(extent / max(1, int(np.sqrt(len(lower)))), 1e-15)
        for i, (lo, hi) in enumerate(zip(self.lower, self.upper)):
            bounds = self._bounds(lo, hi)
            if (bounds[1] - bounds[0] + 1) * (bounds[3] - bounds[2] + 1) > 1024:
                self.large.append(i)
            else:
                for x in range(bounds[0], bounds[1] + 1):
                    for y in range(bounds[2], bounds[3] + 1):
                        self.bins[x, y].append(i)

    def _bounds(self, lo, hi):
        a, b = (lo - self.origin) / self.step, (hi - self.origin) / self.step
        return floor(a[0]), floor(b[0]), floor(a[1]), floor(b[1])

    def query(self, lo, hi):
        bounds = self._bounds(np.asarray(lo), np.asarray(hi))
        if (bounds[1] - bounds[0] + 1) * (bounds[3] - bounds[2] + 1) > 4096:
            ids = np.arange(len(self.lower))
        else:
            found = set(self.large)
            for x in range(bounds[0], bounds[1] + 1):
                for y in range(bounds[2], bounds[3] + 1):
                    found.update(self.bins.get((x, y), ()))
            ids = np.asarray(sorted(found), dtype=int)
        if not len(ids):
            return ids
        return ids[np.all(self.upper[ids] >= lo, axis=1)
                   & np.all(self.lower[ids] <= hi, axis=1)]


def on_segment(p, a, b, tol):
    d, v = b - a, p - a
    length = float(np.linalg.norm(d))
    return (length > tol and abs(float(cross(d, v))) <= tol * length
            and -tol * length <= float(v @ d) <= length * length + tol * length)


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
    if np.any((np.abs(cross(delta, relative)) <= tol * length)
              & (projection >= -tol * length)
              & (projection <= length * length + tol * length)):
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
