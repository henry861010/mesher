"""Rectilinear Quad4 meshing with independently even hole boundaries.

Cuts always terminate on geometry, never on another cut.  Reflex-corner rays
therefore partition the domain into rectangles without hanging nodes.  A sparse
sweep builds the initial rectangles; a bounding-volume tree updates only cells
intersected by subsequent sizing and parity cuts.
If local parity repair is unavailable, a complete Cartesian grid provides
even boundaries without another repair step.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right, insort
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
import heapq
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..editor import EditorMesh2D
from ..model import Mesh2D
from ._holed_geometry import Interval, OrthogonalDomain, normalize_patterns
from ._holed_grid import _ElementBudget, _fits_size, build_global_grid


@dataclass(frozen=True)
class RectilinearHolesResult:
    """A new mesh, complete boundary node chains, and construction diagnostics.

    The outer chain is clockwise and hole chains are counterclockwise, with
    implicit closure. Holes retain their input order. Arrays own their data;
    callers' geometry and patterns are never modified.
    """

    mesh: Mesh2D
    outer_outline: NDArray[np.int64]
    hole_outlines: list[NDArray[np.int64]]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class _Cut:
    # axis is the varying coordinate: 0 = horizontal, 1 = vertical.
    axis: int
    coordinate: float
    interval: Interval
    kind: str

    @property
    def key(self):
        return (self.axis, self.coordinate, self.interval.start, self.interval.end)

    @property
    def points(self):
        a, b = self.interval.start, self.interval.end
        if self.axis == 0:
            return (a, self.coordinate), (b, self.coordinate)
        return (self.coordinate, a), (self.coordinate, b)


class _ParityRepairUnavailable(Exception):
    """The local repair has no representable candidate; a grid can retry."""


def _initial_rectangles(domain, cuts, budget=None):
    """Sweep active horizontal rows without allocating a coordinate product."""
    budget = budget if budget is not None else _ElementBudget()
    starts, ends, vertical = defaultdict(list), defaultdict(list), defaultdict(list)
    # Only vertical boundary edges change a vertical section of the domain.
    # In particular, a mandatory point inside a horizontal boundary edge does
    # not require another domain sample or a rebuild of all active rows.
    geometry_events = sorted({float(a[0]) for loop in domain.loops
                              for a, b in zip(loop, np.roll(loop, -1, axis=0))
                              if a[0] == b[0]})
    refresh_events = set(geometry_events)
    events = set(geometry_events)
    for cut in cuts:
        lo, hi = cut.interval.start, cut.interval.end
        if cut.axis == 0:
            starts[lo].append(cut.coordinate)
            ends[hi].append(cut.coordinate)
            events.update((lo, hi))
            refresh_events.update((lo, hi))
        else:
            vertical[cut.coordinate].append((lo, hi))
            events.add(cut.coordinate)
    events = sorted(events)
    active = Counter()
    levels = []
    opened = {}
    rectangles = []
    ordered_rows = []
    lower = []
    sections = domain.iter_sections_after(1, (x for x in events if x in refresh_events))

    def close_rectangle(bounds):
        budget.check(len(rectangles) + 1, "initial_sweep")
        rectangles.append(bounds)

    for x in events:
        for y in ends[x]:
            active[y] -= 1
            if not active[y]:
                levels.pop(bisect_left(levels, y))
        for y in starts[x]:
            if not active[y]:
                insort(levels, y)
            active[y] += 1

        if x in refresh_events:
            rows = set()
            _, intervals = next(sections)
            for interval in intervals:
                ys = [interval.start]
                ys.extend(levels[bisect_right(levels, interval.start):
                                 bisect_left(levels, interval.end)])
                ys.append(interval.end)
                for row in zip(ys, ys[1:]):
                    budget.check(len(rows) + 1, "initial_sweep")
                    rows.add(row)
            for row in sorted(opened.keys() - rows):
                x0 = opened.pop(row)
                close_rectangle((x0, row[0], x, row[1]))
            ordered_rows = sorted(rows)
            lower = [row[0] for row in ordered_rows]
            for row in ordered_rows:
                if row not in opened:
                    budget.check(len(rectangles) + len(opened) + 1, "initial_sweep")
                    opened[row] = x

        # A vertical-only event leaves the horizontal partition unchanged.
        # Touch just its intersected rows; other rectangles remain open across
        # this X coordinate without copying or scanning their row set.
        if vertical[x]:
            for lo, hi in vertical[x]:
                begin = max(0, bisect_right(lower, lo) - 1)
                for index in range(begin, len(ordered_rows)):
                    row = ordered_rows[index]
                    if row[0] >= hi:
                        break
                    if row[1] <= lo:
                        continue
                    if opened[row] == x:
                        continue
                    if not (lo <= row[0] and row[1] <= hi):
                        raise ValueError("a grid cut would end inside a rectangle")
                    x0 = opened[row]
                    # The closed left cell and open right cell both count.
                    budget.check(len(rectangles) + len(opened) + 1, "initial_sweep")
                    close_rectangle((x0, row[0], x, row[1]))
                    opened[row] = x
    if opened or not rectangles:
        raise ValueError("could not construct a nonempty rectangular domain")
    return rectangles


class _Cell:
    __slots__ = ("bounds", "children")

    def __init__(self, bounds, children=None):
        self.bounds = bounds
        self.children = children


class _RectangleIndex:
    """Mutable rectangle partition with BVH traversal and longest-edge heaps."""

    def __init__(self, rectangles, budget=None):
        self.budget = budget if budget is not None else _ElementBudget()
        self.budget.check(len(rectangles), "rectangle_index")
        self.nodes = []
        self.heaps = [[], []]

        def build(bounds):
            if len(bounds) == 1:
                return self._leaf(bounds[0])
            box = (min(r[0] for r in bounds), min(r[1] for r in bounds),
                   max(r[2] for r in bounds), max(r[3] for r in bounds))
            axis = 0 if box[2] - box[0] >= box[3] - box[1] else 1
            bounds.sort(key=lambda r: (r[axis] / 2 + r[axis + 2] / 2, r))
            middle = len(bounds) // 2
            children = (build(bounds[:middle]), build(bounds[middle:]))
            index = len(self.nodes)
            self.nodes.append(_Cell(box, children))
            return index

        self.root = build(list(rectangles))
        self.leaf_count = len(rectangles)

    def _leaf(self, bounds):
        index = len(self.nodes)
        self.nodes.append(_Cell(bounds))
        for axis in (0, 1):
            length = bounds[axis + 2] - bounds[axis]
            heapq.heappush(self.heaps[axis], (-length, *bounds, index))
        return index

    def largest(self, axis):
        heap = self.heaps[axis]
        while self.nodes[heap[0][-1]].children is not None:
            heapq.heappop(heap)
        return self.nodes[heap[0][-1]].bounds

    def split(self, cut):
        varying, fixed = cut.axis, 1 - cut.axis
        coordinate = cut.coordinate
        lo, hi = cut.interval.start, cut.interval.end
        pending = [self.root]
        count = 0
        while pending:
            node = self.nodes[pending.pop()]
            b = node.bounds
            if not (b[fixed] < coordinate < b[fixed + 2]):
                continue
            if b[varying + 2] <= lo or b[varying] >= hi:
                continue
            if node.children is not None:
                pending.extend(node.children)
                continue
            if not (lo <= b[varying] and b[varying + 2] <= hi):
                raise ValueError("a grid cut would create a hanging node")
            left, right = list(b), list(b)
            left[fixed + 2] = right[fixed] = coordinate
            self.budget.check(self.leaf_count + 1, "split")
            node.children = (self._leaf(tuple(left)), self._leaf(tuple(right)))
            self.leaf_count += 1
            count += 1
        return count

    def rectangles(self):
        return sorted(node.bounds for node in self.nodes if node.children is None)


class _Builder:
    def __init__(self, domain, budget=None):
        self.domain = domain
        self.budget = budget if budget is not None else _ElementBudget()
        self.cuts = {}
        self.boundary_points = [{tuple(map(float, p)) for p in loop}
                                for loop in domain.loops]
        self.index = None
        self._size_preflight_done = False

    def add(self, axis, coordinate, interval, kind):
        cut = _Cut(axis, float(coordinate), interval, kind)
        if cut.key in self.cuts:
            return False
        if self.index is not None and not self.index.split(cut):
            raise ValueError("a new visibility cut did not split any rectangle")
        self.cuts[cut.key] = cut
        a, b = cut.points
        self.boundary_points[interval.start_owner].add(a)
        self.boundary_points[interval.end_owner].add(b)
        return True

    def anchor(self, point, kind):
        """Make a retained point a node using complete visibility chords."""
        for axis in (0, 1):
            coordinate = float(point[1 - axis])
            value = float(point[axis])
            for interval in self.domain.intervals(axis, coordinate):
                if interval.start <= value <= interval.end:
                    self.add(axis, coordinate, interval, kind)

    def seed(self, patterns, target=None):
        for loop in self.domain.loops:
            for point in loop:
                self.anchor(point, "geometry")
        for pattern in patterns:
            axis = 0 if pattern[0, 1] == pattern[1, 1] else 1
            coordinate = float(pattern[0, 1 - axis])
            lo, hi = sorted(map(float, pattern[:, axis]))
            for interval in self.domain.intervals(axis, coordinate):
                if max(lo, interval.start) < min(hi, interval.end):
                    self.add(axis, coordinate, interval, "pattern")
            for point in pattern:
                if self.domain.contains(point) or self.domain.boundary_owner(point) is not None:
                    self.anchor(point, "pattern")
        rectangles = _initial_rectangles(self.domain, self.cuts.values(), self.budget)
        if target is not None:
            self.budget.preflight(rectangles, target)
            self._size_preflight_done = True
        self.index = _RectangleIndex(rectangles, self.budget)

    def size(self, target):
        if not self._size_preflight_done:
            self.budget.preflight(self.index.rectangles(), target)
            self._size_preflight_done = True
        for fixed in (0, 1):
            while True:
                bounds = self.index.largest(fixed)
                length = bounds[fixed + 2] - bounds[fixed]
                if _fits_size(length, target):
                    break
                ratio = length / target
                if not math.isfinite(ratio):
                    self.budget.check(max(self.budget.limit, 2 * np.iinfo(np.int32).max) + 1,
                                      "sizing")
                count = math.ceil(ratio)
                self.budget.check(self.index.leaf_count + count - 1, "sizing")
                # Median-first insertion avoids a linear-depth subdivision tree.
                ranges = [(1, count)]
                while ranges:
                    begin, end = ranges.pop()
                    if begin >= end:
                        continue
                    i = (begin + end) // 2
                    coordinate = bounds[fixed] + length * (i / count)
                    if not bounds[fixed] < coordinate < bounds[fixed + 2]:
                        raise ValueError("target size cannot be represented at these coordinates")
                    varying = 1 - fixed
                    intervals = self.domain.intervals(varying, coordinate)
                    interval = next((entry for entry in intervals
                                     if entry.start <= bounds[varying]
                                     and bounds[varying + 2] <= entry.end), None)
                    if interval is None:
                        raise ValueError("a sizing cut is not contained in its geometry interval")
                    if not self.add(varying, coordinate, interval, "sizing"):
                        raise ValueError("target size cannot be resolved without duplicate cuts")
                    ranges.extend(((i + 1, end), (begin, i)))

    def repair_parity(self):
        before = [len(points) for points in self.boundary_points]
        if sum(before) % 2:
            raise ValueError("rectangular partition has inconsistent boundary parity")
        repairs = []
        minima = [float(np.min(loop[:, 0])) for loop in self.domain.loops]
        order = sorted(range(1, len(minima)), key=lambda owner: (-minima[owner], owner))
        y_levels = {float(p[1]) for loop in self.domain.loops for p in loop}
        y_levels.update(c.coordinate for c in self.cuts.values() if c.axis == 0)
        for owner in order:
            if len(self.boundary_points[owner]) % 2 == 0:
                continue
            x = minima[owner]
            loop = self.domain.loops[owner]
            levels = sorted(y_levels)
            candidates = []
            for a, b in zip(loop, np.roll(loop, -1, axis=0)):
                if a[0] != x or b[0] != x:
                    continue
                lo, hi = sorted((float(a[1]), float(b[1])))
                ys = [lo, *levels[bisect_right(levels, lo):bisect_left(levels, hi)], hi]
                candidates.extend((-(end - start), start, end)
                                  for start, end in zip(ys, ys[1:]))
            interval = None
            for _, lo, hi in sorted(candidates):
                y = lo + (hi - lo) / 2
                if not lo < y < hi:
                    continue
                interval = next((entry for entry in self.domain.intervals(0, y)
                                 if entry.end == x and entry.end_owner == owner), None)
                if interval is not None:
                    break
            if interval is None:
                raise _ParityRepairUnavailable(
                    "no representable parity repair interval for hole " + str(owner - 1)
                )
            parent = interval.start_owner
            if parent and minima[parent] >= x:
                raise ValueError("parity repair did not advance toward the outer boundary")
            if ((interval.start, y) in self.boundary_points[parent]
                    or (x, y) in self.boundary_points[owner]):
                raise ValueError("parity repair endpoints must be new boundary nodes")
            self.add(0, y, interval, "parity")
            y_levels.add(y)
            repairs.append({"segment": [[interval.start, y], [x, y]],
                            "hole_index": owner - 1,
                            "target_hole_index": parent - 1 if parent else None})
        after = [len(points) for points in self.boundary_points]
        if any(count % 2 for count in after):
            raise ValueError("could not make every boundary even")
        return before, after, repairs


def _assemble(domain, rectangles, boundary_points, target, budget, *, patterns=()):
    budget.check(len(rectangles), "assembly")
    rectangles = np.asarray(rectangles, dtype=np.float64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        widths = rectangles[:, 2] - rectangles[:, 0]
        heights = rectangles[:, 3] - rectangles[:, 1]
        areas = widths * heights
        aspects = np.maximum(widths / heights, heights / widths)
    if (not np.all(np.isfinite(areas)) or not np.all(areas > 0)
            or not np.all(np.isfinite(aspects))):
        raise ValueError("rectangle area or aspect ratio is not numerically representable")
    corners = rectangles[:, [0, 1, 2, 1, 2, 3, 0, 3]].reshape(-1, 2)
    nodes, inverse = np.unique(corners, axis=0, return_inverse=True)
    if len(nodes) > np.iinfo(np.int32).max:
        raise ValueError("requested mesh exceeds the int32 node-index capacity")
    mesh = Mesh2D(nodes, inverse.reshape(-1, 4).astype(np.int32))
    chains = EditorMesh2D.outline(mesh)
    if len(chains) != len(domain.loops):
        raise ValueError("grid boundary topology does not match the requested holes")
    point_ids = {tuple(p): i for i, p in enumerate(nodes)}
    for pattern in patterns:
        for point in pattern:
            if (tuple(point) not in point_ids
                    and (domain.contains(point) or domain.boundary_owner(point) is not None)):
                raise ValueError("grid omitted a retained pattern endpoint")
    chain_for_node = {int(node): chain for chain in chains for node in chain}
    outlines = []
    for owner, source in enumerate(domain.loops):
        start = point_ids.get(tuple(source[0]))
        if start is None or start not in chain_for_node:
            raise ValueError("grid omitted a geometry anchor from its boundary")
        chain = chain_for_node[start][::-1].copy()
        chain = np.roll(chain, -int(np.argmin(chain))).astype(np.int64)
        if {tuple(p) for p in nodes[chain]} != boundary_points[owner]:
            raise ValueError("grid omitted an anchor or contains an unmatched boundary edge")
        if len(chain) % 2:
            raise ValueError("grid boundary must have an even node count")
        outlines.append(chain)

    maximum = float(max(np.max(widths), np.max(heights)))
    if not _fits_size(maximum, target):
        raise ValueError("grid exceeded the requested maximum edge size")
    area = math.fsum(map(float, areas))
    if not math.isfinite(area) or not math.isclose(area, domain.area, rel_tol=1e-11):
        raise ValueError("rectangular grid does not cover the requested domain area")
    quality = {"minimum_edge_length": float(min(np.min(widths), np.min(heights))),
               "maximum_edge_length": maximum,
               "maximum_aspect_ratio": float(np.max(aspects))}
    return mesh, outlines, quality


def generate_rectilinear_mesh_optimal(
    target_edge_size: float,
    outer_outline: ArrayLike,
    *,
    holes: Iterable[ArrayLike] = (),
    pattern_lines: Iterable[ArrayLike] = (),
    coordinate_tolerance: float = 1e-3,
    max_elements: int = 1_000_000,
) -> RectilinearHolesResult:
    """Mesh an orthogonal polygon minus disjoint holes with rectangular Quad4s.

    Geometry and retained pattern endpoints are fixed. Local geometric and
    pattern cuts stop at the first outer/hole boundary, while crossing other
    grid lines. Local equal subdivisions enforce the maximum edge size.
    At most one extra cut per hole repairs local boundary parity. If no local
    repair candidate is representable, a global grid automatically provides
    even boundaries; its coordinate lines may also appear beyond a hole.

    Loops are finite ``(n, 2)`` XY arrays with optional repeated closure;
    patterns are finite horizontal/vertical ``(m, 2, 2)`` segments. Patterns
    are clipped to the domain. Holes must be strictly interior, separated,
    and nonnested. ``coordinate_tolerance`` rejects ambiguous geometric
    contacts; it never snaps coordinates or discards mandatory features.
    ``max_elements`` is a positive integer budget checked before cell
    allocation; it is not a process-memory limit. Invalid, over-budget or
    numerically unrepresentable geometry raises ``ValueError``. Before area
    arithmetic, finite loop spans W/H must satisfy sum(2*n*W*H) <= float64.max/2.
    No optional dependency is imported and no caller input is modified.
    """
    if isinstance(target_edge_size, (bool, np.bool_)):
        raise ValueError("target_edge_size must be finite and positive")
    try:
        target = float(target_edge_size)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("target_edge_size must be finite and positive") from exc
    if not math.isfinite(target) or target <= 0:
        raise ValueError("target_edge_size must be finite and positive")
    budget = _ElementBudget(max_elements)
    domain = OrthogonalDomain(outer_outline, holes, coordinate_tolerance)
    patterns = normalize_patterns(pattern_lines)
    builder = _Builder(domain, budget)
    builder.seed(patterns, target)
    builder.size(target)
    strategy, fallback_reason = "local", None
    try:
        before, after, repairs = builder.repair_parity()
    except _ParityRepairUnavailable:
        # Leave the exception handler before allocating: its traceback would
        # otherwise retain the failed builder through repair_parity's frame.
        strategy, fallback_reason = "global_grid_fallback", "parity_repair_unavailable"
    if fallback_reason is not None:
        del builder
        rectangles, boundary_points, cuts = build_global_grid(domain, patterns, target, budget)
        before = after = [len(points) for points in boundary_points]
        repairs = []
    else:
        rectangles = builder.index.rectangles()
        boundary_points = builder.boundary_points
        cuts = {kind: sum(c.kind == kind for c in builder.cuts.values())
                for kind in ("geometry", "pattern", "sizing", "parity")}
        del builder
    mesh, outlines, quality = _assemble(
        domain, rectangles, boundary_points, target, budget, patterns=patterns)
    diagnostics = {
        "strategy": strategy,
        "fallback_reason": fallback_reason,
        "cuts": cuts,
        "parity_repairs": repairs,
        "boundary_node_counts_before": before,
        "boundary_node_counts_after": after,
        "quality": quality,
        "node_count": mesh.node_count,
        "element_count": mesh.element_count,
    }
    return RectilinearHolesResult(mesh, outlines[0], outlines[1:], diagnostics)


__all__ = ["RectilinearHolesResult", "generate_rectilinear_mesh_optimal"]
