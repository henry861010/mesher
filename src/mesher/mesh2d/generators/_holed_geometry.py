"""Exact orthogonal geometry and visibility for the exterior grid generator.

Input coordinates are never snapped.  The tolerance rejects physically close
boundaries whose topology would be ambiguous; collinear outline anchors remain
available to the mesher.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right, insort
from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    start_owner: int
    end_owner: int


def _xy(value, name: str) -> np.ndarray:
    try:
        point = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite XY point") from exc
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must be a finite XY point")
    return point


def _signed_area(loop: np.ndarray) -> float:
    # Translating first avoids cancellation from a large coordinate origin.
    points = loop - loop[0]
    following = np.roll(points, -1, axis=0)
    return 0.5 * math.fsum(
        float(first[0] * second[1] - second[0] * first[1])
        for first, second in zip(points, following)
    )


def _logical_corners(loop: np.ndarray) -> np.ndarray:
    incoming = loop - np.roll(loop, 1, axis=0)
    outgoing = np.roll(loop, -1, axis=0) - loop
    # Axis identity suffices because backtracking was rejected separately.
    turns = (incoming[:, 0] == 0) != (outgoing[:, 0] == 0)
    return loop[turns]


def _segment_distance(first, second, other_first, other_second) -> float:
    lower = np.maximum(np.minimum(first, second), np.minimum(other_first, other_second))
    upper = np.minimum(np.maximum(first, second), np.maximum(other_first, other_second))
    # Separate finite loops may have origins whose distance overflows.  Such
    # pairs are simply farther apart than any admitted finite tolerance.
    with np.errstate(over="ignore"):
        gaps = np.maximum(lower - upper, 0.0)
    # Orthogonal segment boxes are the segments themselves.
    return math.hypot(float(gaps[0]), float(gaps[1]))


def _parse_loop(value, name: str) -> np.ndarray:
    try:
        loop = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite XY points with shape (n, 2)") from exc
    if loop.ndim != 2 or loop.shape[1:] != (2,) or not np.all(np.isfinite(loop)):
        raise ValueError(f"{name} must be finite XY points with shape (n, 2)")
    if len(loop) > 1 and np.array_equal(loop[0], loop[-1]):
        loop = loop[:-1].copy()
    if len(loop) < 4:
        raise ValueError(f"{name} needs at least four vertices")
    return loop


def _check_numeric_envelope(loops) -> None:
    """Bound every shoelace intermediate before performing area arithmetic.

    Products of float64 spans are dyadic rationals.  Exact rational arithmetic
    makes this small, once-per-loop check safe even at the exponent limits;
    neither an overflowing product nor an underflowed budget can hide a term.
    The bound is translation independent and counts retained input anchors.
    """
    remaining = Fraction.from_float(float(np.finfo(np.float64).max)) / 2
    for loop in loops:
        lower = np.min(loop, axis=0)
        upper = np.max(loop, axis=0)
        width = float(upper[0]) - float(lower[0])
        height = float(upper[1]) - float(lower[1])
        if not math.isfinite(width) or not math.isfinite(height):
            raise ValueError("geometry coordinate spans exceed the float64 numeric range")
        remaining -= (2 * len(loop) * Fraction.from_float(width)
                      * Fraction.from_float(height))
        if remaining < 0:
            raise ValueError("geometry exceeds the float64 area-accumulation numeric range")


def _normalize_loop(loop: np.ndarray, name: str, *, clockwise: bool) -> np.ndarray:
    following = np.roll(loop, -1, axis=0)
    equal = loop == following
    if np.any(np.all(equal, axis=1)):
        raise ValueError(f"{name} contains a zero-length edge")
    if np.any(~np.any(equal, axis=1)):
        raise ValueError(f"{name} edges must be exactly axis aligned")
    incoming = loop - np.roll(loop, 1, axis=0)
    outgoing = following - loop
    same_axis = (incoming[:, 0] == 0) == (outgoing[:, 0] == 0)
    # The inactive component is zero on both vectors, so compare the active
    # component explicitly rather than using a dot product that may overflow.
    active = np.where(incoming[:, 0] != 0, 0, 1)
    rows = np.arange(len(loop))
    same_direction = np.sign(incoming[rows, active]) == np.sign(outgoing[rows, active])
    if np.any(same_axis & ~same_direction):
        raise ValueError(f"{name} contains adjacent backtracking edges")

    corners = _logical_corners(loop)
    if len(corners) < 4:
        raise ValueError(f"{name} needs at least four distinct corners")
    signed = _signed_area(loop)
    if not math.isfinite(signed) or signed == 0:
        raise ValueError(f"{name} must enclose finite positive area")
    if (signed < 0) != clockwise:
        loop = loop[::-1].copy()
    first = int(np.lexsort((loop[:, 1], loop[:, 0]))[0])
    loop = np.roll(loop, -first, axis=0)
    loop.setflags(write=False)
    return loop


def _inside_loop(point: np.ndarray, loop: np.ndarray) -> bool:
    """Even/odd point location; the caller excludes boundary points."""
    following = np.roll(loop, -1, axis=0)
    vertical = loop[:, 0] == following[:, 0]
    lower = np.minimum(loop[:, 1], following[:, 1])
    upper = np.maximum(loop[:, 1], following[:, 1])
    crossings = vertical & (lower <= point[1]) & (point[1] < upper) & (loop[:, 0] > point[0])
    return bool(np.count_nonzero(crossings) % 2)


class _AABBIndex:
    """Static bounding-volume tree; queries return candidate item indices."""

    def __init__(self, boxes):
        self.boxes = boxes
        self.nodes = []

        def build(indices):
            bounds = (min(boxes[i][0] for i in indices),
                      min(boxes[i][1] for i in indices),
                      max(boxes[i][2] for i in indices),
                      max(boxes[i][3] for i in indices))
            if len(indices) <= 8:
                index = len(self.nodes)
                self.nodes.append((bounds, None, tuple(indices)))
                return index
            axis = 0 if bounds[2] - bounds[0] >= bounds[3] - bounds[1] else 1
            indices.sort(key=lambda i: (boxes[i][axis] / 2 + boxes[i][axis + 2] / 2, i))
            middle = len(indices) // 2
            children = (build(indices[:middle]), build(indices[middle:]))
            index = len(self.nodes)
            self.nodes.append((bounds, children, ()))
            return index

        self.root = build(list(range(len(boxes)))) if boxes else None

    @staticmethod
    def _overlaps(a, b):
        return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

    def query(self, bounds):
        pending = [] if self.root is None else [self.root]
        while pending:
            node_bounds, children, items = self.nodes[pending.pop()]
            if not self._overlaps(node_bounds, bounds):
                continue
            if children is not None:
                pending.extend(children)
            else:
                for index in items:
                    if self._overlaps(self.boxes[index], bounds):
                        yield index


def _validate_boundaries(corner_loops, tolerance, stats):
    segments = []
    boxes = []
    for owner, loop in enumerate(corner_loops):
        for ordinal, (a, b) in enumerate(zip(loop, np.roll(loop, -1, axis=0))):
            segments.append((a, b, owner, ordinal, len(loop)))
            boxes.append((min(float(a[0]), float(b[0])), min(float(a[1]), float(b[1])),
                          max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))))
    index = _AABBIndex(boxes)
    for first, (a, b, owner, ordinal, count) in enumerate(segments):
        box = boxes[first]
        # Round the broad phase outward; the exact distance check below keeps
        # the public <= tolerance decision unchanged at floating-point limits.
        query = (math.nextafter(box[0] - tolerance, -math.inf),
                 math.nextafter(box[1] - tolerance, -math.inf),
                 math.nextafter(box[2] + tolerance, math.inf),
                 math.nextafter(box[3] + tolerance, math.inf))
        for second in index.query(query):
            if second <= first:
                continue
            c, d, other_owner, other_ordinal, _ = segments[second]
            if owner == other_owner and ((ordinal - other_ordinal) % count in (1, count - 1)):
                continue
            stats["validation_pairs"] += 1
            if _segment_distance(a, b, c, d) > tolerance:
                continue
            if owner == other_owner:
                name = "outer_outline" if owner == 0 else f"holes[{owner - 1}]"
                raise ValueError(
                    f"{name} self-intersects, touches itself, or has nonadjacent "
                    "edges within coordinate_tolerance"
                )
            raise ValueError(
                f"boundaries {owner} and {other_owner} intersect, touch, "
                "or lie within coordinate_tolerance"
            )


class _AxisIndex:
    """Half-open interval stabbing plus exact collinear-boundary lookups."""

    def __init__(self, loops, axis):
        fixed = 1 - axis
        self.edges = []
        self.ends = {}
        self.parallel = {}
        for owner, loop in enumerate(loops):
            for a, b in zip(loop, np.roll(loop, -1, axis=0)):
                if a[fixed] == b[fixed]:
                    low, high = sorted((float(a[axis]), float(b[axis])))
                    self.parallel.setdefault(float(a[fixed]), []).append((low, high, owner))
                else:
                    low, high = sorted((float(a[fixed]), float(b[fixed])))
                    edge = (low, high, float(a[axis]), owner)
                    self.edges.append(edge)
                    self.ends.setdefault(high, []).append(edge)
        for entries in self.parallel.values():
            entries.sort()
        self.parallel_starts = {coordinate: [entry[0] for entry in entries]
                                for coordinate, entries in self.parallel.items()}
        self.starts_order = sorted(range(len(self.edges)), key=lambda i: (self.edges[i][0], i))
        self.ends_order = sorted(range(len(self.edges)), key=lambda i: (self.edges[i][1], i))
        self.nodes = []

        def build(indices):
            if not indices:
                return None
            # A median start guarantees a nonempty middle and balanced side
            # subtrees without relying on a representable interval midpoint.
            starts = sorted(self.edges[i][0] for i in indices)
            center = starts[len(starts) // 2]
            left, middle, right = [], [], []
            for i in indices:
                low, high, _, _ = self.edges[i]
                if high <= center:
                    left.append(i)
                elif low > center:
                    right.append(i)
                else:
                    middle.append(i)
            by_start = sorted(middle, key=lambda i: (self.edges[i][0], i))
            by_end = sorted(middle, key=lambda i: (self.edges[i][1], i))
            child_left, child_right = build(left), build(right)
            node = (center, by_start, [self.edges[i][0] for i in by_start],
                    by_end, [self.edges[i][1] for i in by_end], child_left, child_right)
            index = len(self.nodes)
            self.nodes.append(node)
            return index

        self.root = build(list(range(len(self.edges))))

    def active(self, coordinate):
        node_id = self.root
        while node_id is not None:
            center, starts, lows, ends, highs, left, right = self.nodes[node_id]
            if coordinate < center:
                for i in starts[:bisect_right(lows, coordinate)]:
                    yield self.edges[i]
                node_id = left
            else:
                for i in ends[bisect_right(highs, coordinate):]:
                    yield self.edges[i]
                node_id = right

    def boundary_owner(self, coordinate, position):
        entries = self.parallel.get(coordinate, ())
        starts = self.parallel_starts.get(coordinate, ())
        index = bisect_right(starts, position) - 1
        if index >= 0 and position <= entries[index][1]:
            return entries[index][2]
        return None


def _section_intervals(crossings):
    """Pair ordered crossings in the open slab immediately after an event."""
    if len(crossings) % 2:
        raise ValueError("geometry section has an inconsistent crossing count")
    return tuple(Interval(a[0], b[0], a[1], b[1])
                 for a, b in zip(crossings[::2], crossings[1::2]))


class OrthogonalDomain:
    """A simple exterior loop containing mutually disjoint interior holes.

    Owner zero is the outer loop; owners one onward retain input hole order.
    ``intervals(0, y)`` varies X and ``intervals(1, x)`` varies Y. Intervals
    exclude all boundary points, including isolated tangential contacts.
    """

    def __init__(self, outer_outline, holes=(), coordinate_tolerance: float = 1e-9):
        if isinstance(coordinate_tolerance, (bool, np.bool_)):
            raise ValueError("coordinate_tolerance must be finite and non-negative")
        try:
            tolerance = float(coordinate_tolerance)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("coordinate_tolerance must be finite and non-negative") from exc
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("coordinate_tolerance must be finite and non-negative")
        self.coordinate_tolerance = tolerance
        try:
            hole_inputs = list(holes)
        except TypeError as exc:
            raise ValueError("holes must be a sequence of orthogonal outlines") from exc
        names = ["outer_outline", *(f"holes[{i}]" for i in range(len(hole_inputs)))]
        parsed = [_parse_loop(value, name)
                  for value, name in zip([outer_outline, *hole_inputs], names)]
        _check_numeric_envelope(parsed)
        self.loops = [_normalize_loop(loop, name, clockwise=owner != 0)
                      for owner, (loop, name) in enumerate(zip(parsed, names))]
        self.stats = {"validation_pairs": 0, "section_edges": 0, "interval_edges": 0,
                      "interval_queries": 0, "interval_cache_hits": 0, "boundary_queries": 0}
        corner_loops = [_logical_corners(loop) for loop in self.loops]
        _validate_boundaries(corner_loops, tolerance, self.stats)
        loop_boxes = [(float(np.min(loop[:, 0])), float(np.min(loop[:, 1])),
                       float(np.max(loop[:, 0])), float(np.max(loop[:, 1])))
                      for loop in corner_loops]
        loop_index = _AABBIndex(loop_boxes)
        for owner, hole in enumerate(corner_loops[1:], start=1):
            if not _inside_loop(hole[0], corner_loops[0]):
                raise ValueError(f"holes[{owner - 1}] must lie strictly inside outer_outline")
            point = (float(hole[0, 0]), float(hole[0, 1]))
            for other_owner in loop_index.query((*point, *point)):
                if other_owner in (0, owner):
                    continue
                if _inside_loop(hole[0], corner_loops[other_owner]):
                    raise ValueError("holes must not overlap or contain another hole")
        lower = np.min(self.loops[0], axis=0)
        upper = np.max(self.loops[0], axis=0)
        self.bounds = (float(lower[0]), float(lower[1]), float(upper[0]), float(upper[1]))
        self.area = math.fsum(_signed_area(loop) for loop in self.loops)
        if not math.isfinite(self.area) or self.area <= 0:
            raise ValueError("the domain must have finite positive exterior area")
        self._axes = [_AxisIndex(corner_loops, axis) for axis in (0, 1)]
        self._cache: dict[tuple[int, float], tuple[Interval, ...]] = {}

    def boundary_owner(self, point) -> int | None:
        point = _xy(point, "point")
        self.stats["boundary_queries"] += 1
        for axis, index in enumerate(self._axes):
            owner = index.boundary_owner(float(point[1 - axis]), float(point[axis]))
            if owner is not None:
                return owner
        return None

    def contains(self, point) -> bool:
        point = _xy(point, "point")
        if self.boundary_owner(point) is not None:
            return False
        intervals = self.intervals(0, float(point[1]))
        position = float(point[0])
        index = bisect_left(intervals, position, key=lambda entry: entry.end)
        return index < len(intervals) and intervals[index].start < position < intervals[index].end

    @staticmethod
    def _query(axis, coordinate):
        if axis not in (0, 1):
            raise ValueError("axis must be 0 (horizontal) or 1 (vertical)")
        try:
            coordinate = float(coordinate)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("coordinate must be finite") from exc
        if not math.isfinite(coordinate):
            raise ValueError("coordinate must be finite")
        return coordinate

    def section_after(self, axis: int, coordinate: float) -> tuple[Interval, ...]:
        """Domain section just after a coordinate, without a float midpoint.

        The fixed coordinate varies infinitesimally toward positive infinity.
        This differs from ``intervals`` at boundary events: parallel edges and
        ending contacts do not obstruct the open slab to their positive side.
        """
        coordinate = self._query(axis, coordinate)
        active = list(self._axes[axis].active(coordinate))
        self.stats["section_edges"] += len(active)
        crossings = sorted((position, owner) for _, _, position, owner in active)
        return _section_intervals(crossings)

    def iter_sections_after(self, axis: int, coordinates):
        """Yield ``(coordinate, section)`` for nondecreasing fixed coordinates.

        Each geometric edge enters and leaves the active set at most once.
        Collinear input anchors are absent from this geometric sweep, while
        remaining available in ``loops`` for mandatory mesh nodes.
        """
        self._query(axis, 0.0)
        index = self._axes[axis]
        start_cursor = end_cursor = 0
        ordered = []
        previous = -math.inf
        for value in coordinates:
            coordinate = self._query(axis, value)
            if coordinate < previous:
                raise ValueError("section coordinates must be nondecreasing")
            previous = coordinate
            while start_cursor < len(index.starts_order):
                edge_id = index.starts_order[start_cursor]
                low, high, position, owner = index.edges[edge_id]
                if low > coordinate:
                    break
                if high > coordinate:
                    insort(ordered, (position, owner, edge_id))
                self.stats["section_edges"] += 1
                start_cursor += 1
            while end_cursor < len(index.ends_order):
                edge_id = index.ends_order[end_cursor]
                _, high, position, owner = index.edges[edge_id]
                if high > coordinate:
                    break
                item = (position, owner, edge_id)
                offset = bisect_left(ordered, item)
                if offset < len(ordered) and ordered[offset] == item:
                    ordered.pop(offset)
                self.stats["section_edges"] += 1
                end_cursor += 1
            self.stats["section_edges"] += len(ordered)
            yield coordinate, _section_intervals(ordered)

    def intervals(self, axis: int, coordinate: float) -> tuple[Interval, ...]:
        coordinate = self._query(axis, coordinate)
        self.stats["interval_queries"] += 1
        key = (axis, coordinate)
        if key in self._cache:
            self.stats["interval_cache_hits"] += 1
            return self._cache[key]
        index = self._axes[axis]
        contacts: dict[float, int] = {}
        crossings: list[float] = []
        for _, _, position, owner in index.active(coordinate):
            contacts[position] = owner
            crossings.append(position)
            self.stats["interval_edges"] += 1
        for _, _, position, owner in index.ends.get(coordinate, ()):
            contacts[position] = owner
            self.stats["interval_edges"] += 1
        overlaps = index.parallel.get(coordinate, ())
        for low, high, owner in overlaps:
            contacts[low] = contacts[high] = owner
            self.stats["interval_edges"] += 1
        positions = sorted(contacts)
        crossings.sort()
        result = []
        crossing_cursor = overlap_cursor = 0
        for start, end in zip(positions, positions[1:]):
            while overlap_cursor < len(overlaps) and overlaps[overlap_cursor][1] <= start:
                overlap_cursor += 1
            if (overlap_cursor < len(overlaps) and overlaps[overlap_cursor][0] <= start
                    and end <= overlaps[overlap_cursor][1]):
                continue
            # Classify an open interval symbolically.  This also works when
            # no representable floating-point midpoint lies between its ends.
            while crossing_cursor < len(crossings) and crossings[crossing_cursor] < end:
                crossing_cursor += 1
            count = len(crossings) - crossing_cursor
            if count % 2:
                result.append(Interval(start, end, contacts[start], contacts[end]))
        answer = tuple(result)
        self._cache[key] = answer
        return answer


def normalize_patterns(pattern_lines) -> tuple[np.ndarray, ...]:
    """Return exact, canonical nonzero axis-aligned finite pattern segments."""
    try:
        entries = list(pattern_lines)
    except TypeError as exc:
        raise ValueError("pattern_lines must be a sequence of endpoint pairs") from exc
    unique = {}
    for index, value in enumerate(entries):
        try:
            points = np.array(value, dtype=np.float64, copy=True)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"pattern_lines[{index}] must be finite endpoint pairs") from exc
        if points.shape != (2, 2) or not np.all(np.isfinite(points)):
            raise ValueError(f"pattern_lines[{index}] must have finite shape (2, 2)")
        equal = points[0] == points[1]
        if np.all(equal) or not np.any(equal):
            raise ValueError(f"pattern_lines[{index}] must be nonzero and exactly axis aligned")
        if tuple(points[1]) < tuple(points[0]):
            points = points[::-1].copy()
        key = tuple(float(value) for value in points.ravel())
        points.setflags(write=False)
        unique[key] = points
    return tuple(unique[key] for key in sorted(unique))


__all__ = ["Interval", "OrthogonalDomain", "normalize_patterns"]
