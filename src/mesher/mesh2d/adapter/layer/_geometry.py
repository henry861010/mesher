"""Planar predicates and spatial queries; these never create mesh elements."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from ....tolerance import within_distance


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


def area_epsilon(points, numeric_tol=0.):
    """Roundoff budget for signed area; independent of physical resolution."""
    xy = np.asarray(points)[:, :2]
    perimeter = float(np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1).sum())
    span = float(np.ptp(xy, axis=0).max())
    return max(numeric_tol * perimeter, 64 * np.finfo(float).eps * len(xy) * span**2)


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


def _exact_segment_contact(a, b, c, d, tol):
    """Resolve uncertain intersections/distances using the encoded input floats."""
    a, b, c, d = [tuple(map(Fraction, map(float, p))) for p in (a, b, c, d)]

    def subtract(p, q):
        return (p[0] - q[0], p[1] - q[1])

    def dot(p, q):
        return p[0] * q[0] + p[1] * q[1]

    def det(p, q):
        return p[0] * q[1] - p[1] * q[0]

    def interpolate(p, v, t):
        return (p[0] + t * v[0], p[1] + t * v[1])

    def project(p, start, vector):
        squared = dot(vector, vector)
        if not squared:
            return start
        t = max(Fraction(0), min(Fraction(1), dot(subtract(p, start), vector) / squared))
        return interpolate(start, vector, t)

    u, v, offset = subtract(b, a), subtract(d, c), subtract(c, a)
    determinant = det(u, v)
    if determinant:
        t, s = det(offset, v) / determinant, det(offset, u) / determinant
        if 0 <= t <= 1 and 0 <= s <= 1:
            point = np.array(interpolate(a, u, t), dtype=float)
            return point, point.copy()
    pairs = [(a, project(a, c, v)), (b, project(b, c, v)),
             (project(c, a, u), c), (project(d, a, u), d)]
    squared, first = min((dot(subtract(p, q), subtract(p, q)), p) for p, q in pairs)
    if squared > Fraction(float(tol)) ** 2:
        return None
    if not determinant and dot(u, u) and dot(v, v):
        low, high = sorted((dot(offset, u) / dot(u, u),
                            dot(subtract(d, a), u) / dot(u, u)))
        low, high = max(Fraction(0), low), min(Fraction(1), high)
        if low <= high:
            return (np.array(interpolate(a, u, low), dtype=float),
                    np.array(interpolate(a, u, high), dtype=float))
    point = np.array(first, dtype=float)
    return point, point.copy()


def _uncertain_cross(first, second, value):
    products = (np.abs(first[..., 0] * second[..., 1])
                + np.abs(first[..., 1] * second[..., 0]))
    return (products != 0) & (np.abs(value) <= 8 * np.finfo(float).eps * products)


def segment_relations(a, b, others, tol, *, represented=None):
    """Batch finite-segment contacts and their endpoints on the first segment.

    Return (contact mask, first points, last points). A nonzero interval denotes
    overlapping projections of parallel segments within Euclidean distance T;
    a single point denotes an intersection or nearest-point contact. Inputs use
    the caller's safe local frame. No angular or length slack inflates T.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    others = np.asarray(others, dtype=float).reshape(-1, 2, 2)
    count = len(others)
    contact = np.zeros(count, dtype=bool)
    first = np.broadcast_to(a, (count, 2)).copy()
    last = first.copy()
    if not count:
        return contact, first, last
    c, d = others[:, 0], others[:, 1]
    u, v, offset = b - a, d - c, c - a
    determinant = cross(u, v)
    numerator_t, numerator_s = cross(offset, v), cross(offset, u)
    t = np.divide(numerator_t, determinant, out=np.zeros(count), where=determinant != 0)
    s = np.divide(numerator_s, determinant, out=np.zeros(count), where=determinant != 0)
    uncertain = _uncertain_cross(u, v, determinant)
    # Cancellation near either endpoint can change whether the finite segments
    # actually intersect, even when their directions are well conditioned.
    uncertain |= ((determinant != 0)
                  & (_uncertain_cross(offset, v, numerator_t)
                     | _uncertain_cross(offset, u, numerator_s)
                     | _uncertain_cross(c - b, v, cross(c - b, v))
                     | _uncertain_cross(d - a, u, cross(d - a, u))))
    contact = ((determinant != 0) & ~uncertain
               & (t >= 0) & (t <= 1) & (s >= 0) & (s <= 1))
    first[contact] = a + t[contact, None] * u
    last[contact] = first[contact]
    ids = np.flatnonzero(~contact & ~uncertain)
    if len(ids):
        cv, dv, vv = c[ids], d[ids], v[ids]
        u2, v2 = float(u @ u), np.sum(vv * vv, axis=1)

        def project_on_others(point):
            fraction = np.divide(np.sum((point - cv) * vv, axis=1), v2,
                                 out=np.zeros(len(ids)), where=v2 != 0)
            return cv + np.clip(fraction, 0, 1)[:, None] * vv

        tc = (cv - a) @ u / u2 if u2 else np.zeros(len(ids))
        td = (dv - a) @ u / u2 if u2 else np.zeros(len(ids))
        first_points = np.stack((np.broadcast_to(a, cv.shape), np.broadcast_to(b, cv.shape),
                                 a + np.clip(tc, 0, 1)[:, None] * u,
                                 a + np.clip(td, 0, 1)[:, None] * u), axis=1)
        second_points = np.stack((project_on_others(a), project_on_others(b), cv, dv), axis=1)
        delta = first_points - second_points
        distances = np.hypot(delta[..., 0], delta[..., 1])
        nearest = np.argmin(distances, axis=1)
        rows = np.arange(len(ids))
        distance = distances[rows, nearest]
        contact[ids] = distance <= tol
        first[ids] = first_points[rows, nearest]
        last[ids] = first[ids]
        low, high = np.maximum(0., np.minimum(tc, td)), np.minimum(1., np.maximum(tc, td))
        parallel = (determinant[ids] == 0) & (u2 > 0) & (v2 > 0) & (low <= high)
        overlap_ids = ids[parallel & contact[ids]]
        first[overlap_ids] = a + low[parallel & contact[ids], None] * u
        last[overlap_ids] = a + high[parallel & contact[ids], None] * u
        # Account for subtraction and projection roundoff; only uncertain cases
        # pay for rational projection and an exact squared-distance comparison.
        scale = np.maximum(np.max(np.abs(others[ids]), axis=(1, 2)),
                           max(float(np.max(np.abs((a, b)))), float(tol)))
        uncertain[ids] |= np.abs(distance - tol) <= 32 * np.finfo(float).eps * scale
    for index in np.flatnonzero(uncertain):
        if represented is None:
            exact = _exact_segment_contact(a, b, c[index], d[index], tol)
        else:
            world_edge, world_others, origin, frame_scale, physical_tol = represented
            exact = _exact_segment_contact(*world_edge, *world_others[index], physical_tol)
            if exact is not None:
                exact = tuple((point - origin) / frame_scale for point in exact)
        contact[index] = exact is not None
        if exact is not None:
            first[index], last[index] = exact
    return contact, first, last


def segment_contacts(a, b, c, d, tol):
    """Return finite contact coordinates, preserving parallel overlap endpoints."""
    # Planning visits many obviously disjoint pairs. This conservative broad
    # phase avoids allocating batch arrays; its roundoff padding never decides
    # contact, which remains the core's exact/inclusive distance predicate.
    padding = tol + 8 * np.finfo(float).eps * max(map(abs, (*a, *b, *c, *d)))
    if (max(a[0], b[0]) + padding < min(c[0], d[0])
            or max(c[0], d[0]) + padding < min(a[0], b[0])
            or max(a[1], b[1]) + padding < min(c[1], d[1])
            or max(c[1], d[1]) + padding < min(a[1], b[1])):
        return []
    contact, first, last = segment_relations(a, b, np.asarray([[c, d]]), tol)
    if not contact[0]:
        return []
    if np.linalg.norm(last[0] - first[0]) > tol:
        return [first[0], last[0]]
    return [(first[0] + last[0]) * .5]


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
