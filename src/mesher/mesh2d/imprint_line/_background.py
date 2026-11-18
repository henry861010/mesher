"""Sparse rectangular grid around reserved transition cells.

Reuse the existing generator's sweep and rectangle BVH, without its hole-parity
repair (parity belongs to each mixed-element transition unit here).
"""
from __future__ import annotations

import numpy as np

from ..generators._holed_geometry import OrthogonalDomain, _AxisIndex, _logical_corners
from ..generators.holed_rectilinear import _Builder, _RectangleIndex, _initial_rectangles
from ...tolerance import subdivision_count
from ._types import ImprintLineError, loop


class ReservedDomain(OrthogonalDomain):
    """Validated exterior minus planner-owned disjoint rectangular reservations.

    Reservations may meet the exterior. Unlike user-supplied holes, these
    rectangles have already been checked for containment and non-overlap.
    Coincident boundary crossings produce zero-width sections, which are
    removed before passing them to the sweep.
    """

    def __init__(self, outer, transitions):
        self.policy, self.tolerance = outer.policy, outer.tolerance
        self.bounds = outer.bounds
        self.loops = [outer.loops[0]]
        for transition in transitions:
            b = transition.bounds
            corners = loop(b)
            # Retain the subdivision corners on both transverse boundaries.
            points = []
            for p, q in zip(corners, np.roll(corners, -1, axis=0)):
                points.append(p)
                axis = 1 - transition.axis
                if p[axis] != q[axis]:
                    levels = [v for v in transition.splits if min(p[axis], q[axis]) < v < max(p[axis], q[axis])]
                    for value in sorted(levels, reverse=p[axis] > q[axis]):
                        xy = p.copy()
                        xy[axis] = value
                        points.append(xy)
            self.loops.append(np.asarray(points))
        self.area = outer.area - sum((t.bounds[2] - t.bounds[0]) * (t.bounds[3] - t.bounds[1]) for t in transitions)
        self._axes = [_AxisIndex([_logical_corners(p) for p in self.loops], a) for a in (0, 1)]
        self._cache = {}
        self.stats = dict.fromkeys(outer.stats, 0)

    def iter_sections_after(self, axis, coordinates):
        for coordinate, intervals in super().iter_sections_after(axis, coordinates):
            yield coordinate, tuple(i for i in intervals if i.start < i.end)


class Background(_Builder):
    def __init__(self, domain, budget, transitions):
        super().__init__(domain, budget)
        self.transitions = transitions
        self.origins = {}

    def anchor(self, point, kind):
        for axis in (0, 1):
            coordinate, value = float(point[1 - axis]), float(point[axis])
            for interval in self.domain.intervals(axis, coordinate):
                if interval.start <= value <= interval.end:
                    self.add(axis, coordinate, interval, kind)
                    self.origins.setdefault((axis, coordinate), set()).add(value)

    def seed_lines(self, cores, patterns, bands, size):
        for outline in self.domain.loops:
            for point in outline:
                self.anchor(point, "geometry")
        for core in cores:
            for point in loop(core):
                self.anchor(point, "control")
        for pattern in patterns:
            axis = 0 if pattern[0, 1] == pattern[1, 1] else 1
            lo, hi = sorted(pattern[:, axis])
            for interval in self.domain.intervals(axis, pattern[0, 1 - axis]):
                if max(lo, interval.start) < min(hi, interval.end):
                    self.add(axis, pattern[0, 1 - axis], interval, "pattern")
                    self.origins.setdefault((axis, float(pattern[0, 1 - axis])), set()).update(
                        (max(lo, interval.start), min(hi, interval.end)))
            for point in pattern:
                if self.domain.contains(point) or self.domain.boundary_owner(point) is not None:
                    self.anchor(point, "pattern")
        for band in bands:
            for coordinate in band.coordinates:
                for interval in self.domain.intervals(band.axis, coordinate):
                    if max(band.start, interval.start) < min(band.end, interval.end):
                        self.add(band.axis, coordinate, interval, "dense")
                        self.origins.setdefault((band.axis, float(coordinate)), set()).update(
                            (max(band.start, interval.start), min(band.end, interval.end)))
    def partition(self, size):
        # Inspect all hard contacts before allocating a possibly invalid grid.
        rectangles = _initial_rectangles(self.domain, self.cuts.values(), self.budget)
        self.budget.preflight(rectangles, size)
        self.index = _RectangleIndex(rectangles, self.budget)

    def size_locked(self, size):
        """Choose sizing cuts at unit corners instead of splitting locked sides."""
        for fixed in (0, 1):
            varying = 1 - fixed
            while True:
                bounds = self.index.largest(fixed)
                length = bounds[fixed + 2] - bounds[fixed]
                if length <= size * (1 + 32 * np.finfo(float).eps):
                    break
                count = subdivision_count(length, size)
                self.budget.check(self.index.leaf_count + count - 1, "imprint_sizing")
                coordinate = bounds[fixed] + length * ((count // 2) / count)
                visited = set()
                while True:
                    if coordinate in visited:
                        raise ImprintLineError("cannot size background beside a reserved unit",
                                               code="locked_sizing_conflict", stage="background")
                    visited.add(coordinate)
                    interval = next((i for i in self.domain.intervals(varying, coordinate)
                                     if i.start <= bounds[varying] and bounds[varying + 2] <= i.end), None)
                    if interval is None:
                        raise ImprintLineError("sizing cut leaves its rectangle", code="grid_topology", stage="background")
                    replacement = None
                    for owner, endpoint in ((interval.start_owner, interval.start), (interval.end_owner, interval.end)):
                        if not owner:
                            continue
                        t = self.transitions[owner - 1]
                        if varying == t.axis and endpoint == t.inner:
                            continue
                        allowed = (t.splits if varying == t.axis else
                                   (t.bounds[fixed], t.bounds[fixed + 2]))
                        if coordinate in allowed:
                            continue
                        candidates = [float(v) for v in allowed if bounds[fixed] < v < bounds[fixed + 2]]
                        if not candidates:
                            raise ImprintLineError("locked side prevents background sizing",
                                                   code="locked_sizing_conflict", stage="background")
                        replacement = min(candidates, key=lambda v: (abs(v - coordinate), v))
                        break
                    if replacement is None:
                        break
                    coordinate = replacement
                self.add(varying, coordinate, interval, "sizing")


def background(outer, transitions, cores, patterns, bands, size, budget):
    domain = ReservedDomain(outer, transitions)
    builder = Background(domain, budget, transitions)
    builder.seed_lines(cores, patterns, bands, size)
    return builder
