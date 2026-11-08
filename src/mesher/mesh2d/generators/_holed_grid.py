"""Count guards and the complete Cartesian fallback for orthogonal domains."""

from __future__ import annotations

import math
import operator

import numpy as np
from mesher.tolerance import subdivision_count


_SIZE_RTOL = float(32 * np.finfo(float).eps)
_INDEX_LIMIT = np.iinfo(np.int32).max


def _fits_size(length, target):
    return length <= target or math.isclose(length, target, rel_tol=_SIZE_RTOL)


class _ElementBudget:
    def __init__(self, limit=1_000_000):
        if isinstance(limit, (bool, np.bool_)):
            raise ValueError("max_elements must be a positive integer")
        try:
            self.limit = operator.index(limit)
        except TypeError as exc:
            raise ValueError("max_elements must be a positive integer") from exc
        if self.limit <= 0:
            raise ValueError("max_elements must be a positive integer")

    def check(self, count, stage):
        if count > self.limit:
            raise ValueError(
                f"max_elements exceeded: limit={self.limit}, "
                f"required_at_least={count}, stage={stage}"
            )
        # A connected planar Quad4 domain has E <= 2V, independently of
        # boundary count. Reject only a proven index-capacity excess here;
        # assembly checks the actual node count before casting connectivity.
        if count > 2 * _INDEX_LIMIT:
            raise ValueError(f"requested mesh exceeds the int32 node-index capacity, stage={stage}")

    def preflight(self, rectangles, target):
        """Lower bound for refinements that retain the initial rectangle edges.

        A rectangle needs at least ceil(width/h) * ceil(height/h) tiles: place
        a product of sample points farther than h apart in each coordinate.
        No admissible tile covers two of them. Enlarge h and round quotients
        downward to accommodate the same size tolerance as the actual mesher.
        """
        upper = math.nextafter(target / (1 - _SIZE_RTOL), math.inf)
        total = 0
        for x0, y0, x1, y1 in rectangles:
            counts = []
            for length in (x1 - x0, y1 - y0):
                ratio = length / upper
                if not math.isfinite(ratio):
                    self.check(max(self.limit, 2 * _INDEX_LIMIT) + 1, "sizing_preflight")
                counts.append(max(1, math.ceil(math.nextafter(ratio, -math.inf))))
            total += counts[0] * counts[1]
            self.check(total, "sizing_preflight")


def _segment_count(length, target, budget, stage):
    if _fits_size(length, target):
        return 1
    ratio = length / target
    if not math.isfinite(ratio):
        budget.check(max(budget.limit, 2 * _INDEX_LIMIT) + 1, stage)
    count = subdivision_count(length, target)
    budget.check(count, stage)
    return count


def _required_coordinates(domain, patterns):
    """Collect only input constraints, independent of a failed local attempt."""
    sources = [{}, {}]
    for loop in domain.loops:
        for point in loop:
            for axis in (0, 1):
                sources[axis][float(point[axis])] = "geometry"

    def retain(point):
        for axis in (0, 1):
            sources[axis].setdefault(float(point[axis]), "pattern")

    for pattern in patterns:
        axis = 0 if pattern[0, 1] == pattern[1, 1] else 1
        fixed = 1 - axis
        coordinate = float(pattern[0, fixed])
        lo, hi = sorted(map(float, pattern[:, axis]))
        for interval in domain.intervals(axis, coordinate):
            start, end = max(lo, interval.start), min(hi, interval.end)
            if start < end:
                for value in (start, end):
                    point = (value, coordinate) if axis == 0 else (coordinate, value)
                    retain(point)
        for point in pattern:
            if domain.contains(point) or domain.boundary_owner(point) is not None:
                retain(point)
    return sources


def _planned_prefix(coordinates, target, budget):
    prefix = [0]
    for a, b in zip(coordinates, coordinates[1:]):
        count = _segment_count(b - a, target, budget, "fallback_axis_preflight")
        prefix.append(prefix[-1] + count)
        # Every X (or Y) slab of a connected domain contains at least a cell.
        budget.check(prefix[-1], "fallback_axis_preflight")
    return prefix


def _grid_count(domain, mandatory, prefix, budget):
    y_ranks = dict(zip(mandatory[1], prefix[1]))
    total = 0
    sections = domain.iter_sections_after(1, mandatory[0][:-1])
    for i, (_, intervals) in enumerate(sections):
        nx = prefix[0][i + 1] - prefix[0][i]
        ny = sum(y_ranks[entry.end] - y_ranks[entry.start] for entry in intervals)
        total += nx * ny
        budget.check(total, "fallback_grid_preflight")
    return total


def _densify_axis(coordinates, target, budget):
    """Keep input coordinates exact and resolve rounding-induced long gaps."""
    values = [coordinates[0]]
    prefix = [0]
    for start, end in zip(coordinates, coordinates[1:]):
        pending = [(start, end)]
        while pending:
            a, b = pending.pop()
            count = _segment_count(b - a, target, budget, "fallback_axis_sampling")
            if count == 1:
                budget.check(len(values), "fallback_axis_sampling")
                values.append(b)
                continue
            # Bound temporary sampling as well as the final coordinate list.
            budget.check(len(values) - 1 + len(pending) + count, "fallback_axis_sampling")
            points = [a, *(a + (b - a) * (i / count) for i in range(1, count)), b]
            if any(left >= right for left, right in zip(points, points[1:])):
                raise ValueError("target size cannot be represented at these coordinates")
            pending.extend(reversed(list(zip(points, points[1:]))))
        prefix.append(len(values) - 1)
    return np.asarray(values, dtype=np.float64), prefix


def build_global_grid(domain, patterns, target, budget):
    """Return rectangles, exact boundary points and final-grid cut counts.

    Every retained rectangle is an elementary cell of the same Cartesian
    grid. Its boundary graph is bipartite, so each closed outline is even.
    Classification uses symbolic open slabs, never a floating midpoint.
    """
    sources = _required_coordinates(domain, patterns)
    mandatory = [sorted(entries) for entries in sources]
    planned = [_planned_prefix(axis, target, budget) for axis in mandatory]
    _grid_count(domain, mandatory, planned, budget)
    axes, prefix = [], []
    for axis in mandatory:
        values, ranks = _densify_axis(axis, target, budget)
        axes.append(values)
        prefix.append(ranks)
    count = _grid_count(domain, mandatory, prefix, budget)
    rectangles = np.empty((count, 4), dtype=np.float64)
    y_ranks = dict(zip(mandatory[1], prefix[1]))
    x, y = axes
    offset = 0
    for i, (_, intervals) in enumerate(domain.iter_sections_after(1, mandatory[0][:-1])):
        for ix in range(prefix[0][i], prefix[0][i + 1]):
            for interval in intervals:
                lo, hi = y_ranks[interval.start], y_ranks[interval.end]
                block = rectangles[offset:offset + hi - lo]
                block[:, 0], block[:, 2] = x[ix], x[ix + 1]
                block[:, 1], block[:, 3] = y[lo:hi], y[lo + 1:hi + 1]
                offset += hi - lo
    if offset != count or not count:
        raise ValueError("global grid count does not match its symbolic sections")

    boundary_points = [{tuple(map(float, point)) for point in loop} for loop in domain.loops]
    cuts = dict.fromkeys(("geometry", "pattern", "sizing", "parity"), 0)
    for fixed in (0, 1):
        varying = 1 - fixed
        for coordinate in axes[fixed]:
            coordinate = float(coordinate)
            kind = sources[fixed].get(coordinate, "sizing")
            for interval in domain.intervals(varying, coordinate):
                cuts[kind] += 1
                for value, owner in ((interval.start, interval.start_owner),
                                     (interval.end, interval.end_owner)):
                    point = (coordinate, value) if fixed == 0 else (value, coordinate)
                    boundary_points[owner].add(point)
    return rectangles, boundary_points, cuts
