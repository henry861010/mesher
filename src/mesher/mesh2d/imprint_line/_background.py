"""Sparse rectangular grid around reserved transition cells.

Reuse the existing generator's sweep and rectangle BVH, without its hole-parity
repair (parity belongs to each mixed-element transition unit here).
"""
from __future__ import annotations

import heapq

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
                    # Python 3.11 parses reverse as an integer; use a native bool.
                    for value in sorted(levels, reverse=bool(p[axis] > q[axis])):
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


class BackgroundConflict(Exception):
    """A generated reservation must move before the grid can be resolved."""

    def __init__(self, owners):
        self.owners = tuple(sorted(owners))
        super().__init__(f"background conflicts with reservations {self.owners}")


class Background(_Builder):
    def __init__(self, domain, budget, transitions, constraints):
        super().__init__(domain, budget)
        self.transitions = transitions
        self.origins = {}
        self.constraints = constraints

    def support_coordinate(self, axis, coordinate):
        # Input supports are already fixed. Generated cuts are resolved before
        # querying visibility, never by the shared builder's distance snapping.
        return float(coordinate)

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

    def _failure(self, message, code, bounds, owners=()):
        owners = set(owners)
        for owner, t in enumerate(self.transitions, 1):
            for axis in (0, 1):
                other = 1 - axis
                # Reservation corners cast chords far beyond the reservation.
                # A remote thin rectangle can therefore require replanning its
                # source even when it does not touch that reservation directly.
                generated = (t.bounds[axis], t.bounds[axis + 2])
                if axis != t.axis:
                    generated = (*generated, *t.splits)
                for value in (bounds[axis], bounds[axis + 2]):
                    fixed = self.constraints.supports[axis]
                    index = np.searchsorted(fixed, value)
                    if (index == len(fixed) or fixed[index] != value) and value in generated:
                        owners.add(owner)
                if (bounds[axis] == t.bounds[axis + 2] or bounds[axis + 2] == t.bounds[axis]):
                    if min(bounds[other + 2], t.bounds[other + 2]) > max(bounds[other], t.bounds[other]):
                        owners.add(owner)
        if owners:
            raise BackgroundConflict(owners)
        raise ImprintLineError(message, code=code, stage="background", bounds=bounds,
                               tolerance=self.domain.tolerance)

    def _resolvable(self, length, size):
        return length > 0 and length / subdivision_count(length, size) > self.domain.tolerance

    def partition(self, size):
        rectangles = _initial_rectangles(self.domain, self.cuts.values(), self.budget)
        self.budget.preflight(rectangles, size)
        for bounds in rectangles:
            if any(not self._resolvable(bounds[a + 2] - bounds[a], size) for a in (0, 1)):
                self._failure("rectangle cannot satisfy both size and resolution",
                              "feature_below_resolution", bounds)
        self.index = _RectangleIndex(rectangles, self.budget)

    def _cut_feasible(self, fixed, coordinate, interval, size):
        """Check every leaf touched by this complete visibility chord."""
        varying = 1 - fixed
        pending = [self.index.root]
        while pending:
            node = self.index.nodes[pending.pop()]
            b = node.bounds
            if not (b[fixed] < coordinate < b[fixed + 2]):
                continue
            if b[varying + 2] <= interval.start or b[varying] >= interval.end:
                continue
            if node.children is not None:
                pending.extend(node.children)
            elif not all(self._resolvable(length, size) for length in
                         (coordinate - b[fixed], b[fixed + 2] - coordinate)):
                return False
        return True

    def size_locked(self, size):
        """Each successful iteration splits its target; retries are finite."""
        for fixed in (0, 1):
            varying = 1 - fixed
            while True:
                bounds = self.index.largest(fixed)
                # largest() has removed stale heap entries; retain this exact
                # leaf so a successful cut must split the selected rectangle.
                target = self.index.nodes[self.index.heaps[fixed][0][-1]]
                length = bounds[fixed + 2] - bounds[fixed]
                if length <= size * (1 + 32 * np.finfo(float).eps):
                    break
                count = subdivision_count(length, size)
                self.budget.check(self.index.leaf_count + count - 1, "imprint_sizing")
                ideal = bounds[fixed] + length * ((count // 2) / count)
                pending = [(0., ideal)]
                visited, owners = set(), set()
                expanded = False
                while True:
                    if not pending:
                        if expanded:
                            self._failure("no admissible sizing cut", "locked_sizing_conflict", bounds, owners)
                        expanded = True
                        # Fallback positions are bounded by the subdivision budget.
                        for k in range(1, count):
                            value = bounds[fixed] + length * (k / count)
                            heapq.heappush(pending, (abs(value - ideal), value))
                        for t in self.transitions:
                            allowed = t.splits if varying == t.axis else (t.bounds[fixed], t.bounds[fixed + 2])
                            for value in allowed:
                                if bounds[fixed] < value < bounds[fixed + 2]:
                                    heapq.heappush(pending, (abs(value - ideal), float(value)))
                        if not pending:
                            self._failure("no admissible sizing cut", "locked_sizing_conflict", bounds, owners)
                    _, original = heapq.heappop(pending)
                    coordinate = float(self.constraints.align_generated(fixed, original, self.domain.tolerance))
                    if coordinate in visited:
                        continue
                    visited.add(coordinate)
                    if not (bounds[fixed] < coordinate < bounds[fixed + 2]):
                        continue
                    interval = next((i for i in self.domain.intervals(varying, coordinate)
                                     if i.start <= bounds[varying] and bounds[varying + 2] <= i.end), None)
                    if interval is None:
                        continue
                    blocked = False
                    for owner, endpoint in ((interval.start_owner, interval.start), (interval.end_owner, interval.end)):
                        if not owner:
                            continue
                        t = self.transitions[owner - 1]
                        if varying == t.axis and endpoint == t.inner:
                            continue
                        allowed = t.splits if varying == t.axis else (t.bounds[fixed], t.bounds[fixed + 2])
                        if coordinate in allowed:
                            continue
                        owners.add(owner)
                        blocked = True
                        for value in allowed:
                            if bounds[fixed] < value < bounds[fixed + 2]:
                                heapq.heappush(pending, (abs(value - ideal), float(value)))
                    if blocked or not self._cut_feasible(fixed, coordinate, interval, size):
                        continue
                    before = self.index.leaf_count
                    added = self.add(varying, coordinate, interval, "sizing")
                    if not added or self.index.leaf_count <= before or target.children is None:
                        raise ImprintLineError("sizing cut made no progress", code="sizing_no_progress",
                                               stage="background", bounds=bounds, coordinate=coordinate)
                    break


def background(outer, transitions, cores, patterns, bands, size, budget, constraints):
    domain = ReservedDomain(outer, transitions)
    builder = Background(domain, budget, transitions, constraints)
    builder.seed_lines(cores, patterns, bands, size)
    return builder
