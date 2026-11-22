"""Find local die-corner control areas, including adjoining die gaps."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mesher.mesh2d.imprint_line import BoxPairSpec

_Line = TypeVar("_Line", bound=ArrayLike)


def find_adapter_group(
    group_size: float, lines: Iterable[_Line], min_group_count: int = 3
) -> tuple[list[BoxPairSpec], list[_Line]]:
    """Return adapter box controls and the remaining original pattern lines.

    Parallel segments must have identical spans. A gap greater than
    `group_size` separates runs. A run qualifies at `min_group_count`
    distinct offsets, counting duplicate and reversed segments once.
    All copies of grouped lines are consumed. Remaining lines preserve
    input order, endpoint orientation, and duplicates. Inputs are not mutated.
    Horizontal lines supply Y samples in x_points, verticals X in y_points.
    """
    if not np.isfinite(group_size) or group_size <= 0:
        raise ValueError("group_size must be finite and positive")
    if not isinstance(min_group_count, (int, np.integer)) or min_group_count < 2:
        raise ValueError("min_group_count must be an integer of at least 2")

    lines = list(lines)
    offsets: dict[tuple[str, float, float], dict[float, list[int]]] = {}
    for index, line in enumerate(lines):
        endpoints = np.asarray(line, dtype=float)
        if endpoints.shape != (2, 2) or not np.isfinite(endpoints).all():
            raise ValueError(f"lines[{index}] requires two finite XY endpoints")
        (x1, y1), (x2, y2) = endpoints.tolist()
        if y1 == y2 and x1 != x2:
            key = ("horizontal", min(x1, x2), max(x1, x2))
            offset = y1
        elif x1 == x2 and y1 != y2:
            key = ("vertical", min(y1, y2), max(y1, y2))
            offset = x1
        else:
            raise ValueError(f"lines[{index}] must be a nonzero axis-aligned segment")
        offsets.setdefault(key, {}).setdefault(offset, []).append(index)

    result: list[BoxPairSpec] = []
    grouped = [False] * len(lines)
    for (orientation, low, high), indices_by_offset in sorted(offsets.items()):
        positions = sorted(indices_by_offset)
        start = 0
        for stop in range(1, len(positions) + 1):
            if stop < len(positions) and positions[stop] - positions[stop - 1] <= group_size:
                continue
            run = positions[start:stop]
            start = stop
            if len(run) < min_group_count:
                continue
            first, last = run[0], run[-1]
            if orientation == "horizontal":
                result.append(BoxPairSpec(box=[low, first, high, last],
                                          x_points=[[low, y] for y in run]))
            else:
                result.append(BoxPairSpec(box=[first, low, last, high],
                                          y_points=[[x, low] for x in run]))
            for offset in run:
                for index in indices_by_offset[offset]:
                    grouped[index] = True
    return result, [line for index, line in enumerate(lines) if not grouped[index]]


def get_mm2_control_area(
    boxes: ArrayLike, *, corner_extent: float = 25.0, max_gap: float | None = None
) -> NDArray[np.float64]:
    """Return the rectangular union of local corner/gap control areas.

    ``boxes`` contains axis-aligned dies as ``[x1, y1, x2, y2]`` rows.
    Each corner starts with a square extending ``corner_extent`` in both
    directions along each axis. A positive-width gap adjoining that corner
    extends the square along the gap axis through the opposite die's facing
    edge, plus ``corner_extent``. A gap requires positive transverse overlap
    and no intervening die in the visible portion of that overlap. Its closed
    transverse span must contain the source corner. Only gaps incident to the
    source die participate; extended patches do not discover further gaps.
    A finite nonnegative ``max_gap`` limits the edge-to-edge distance along
    each axis (inclusive). Zero disables gap extension. None allows any width.

    X and Y extensions at the same corner combine into one rectangle, filling
    crossed gaps between four dies. Unrelated distant corner bands never
    intersect globally. Overlapping patches are returned as a deterministic,
    lexicographically sorted union of interior-disjoint rectangles, not a
    bounding box; the decomposition need not use the fewest rectangles.

    The result has shape ``(m, 4)`` and dtype ``float64``, including ``(0, 4)``
    for empty input. Coordinates and extent use the same model units. No
    sampling, snapping, tolerance, domain clipping, or meshing is performed.
    A mesh spacing such as 5 belongs to subsequent control-line sampling.
    Inputs are never modified; output storage is independent.

    Raises:
        ValueError: For malformed/nonfinite coordinates, nonpositive die
            dimensions, overlapping die interiors (including duplicates), or
            a nonpositive/nonfinite extent. Touching dies are allowed but
            zero-width gaps do not extend controls. Unrepresentable expanded
            coordinates also raise ValueError.
    """
    dies, extent = _validate(boxes, corner_extent)
    if max_gap is not None:
        raw_gap = np.asarray(max_gap)
        if raw_gap.ndim != 0 or raw_gap.dtype.kind not in "iuf":
            raise ValueError("max_gap must be a finite nonnegative scalar or None")
        max_gap = float(raw_gap)
        if not np.isfinite(max_gap) or max_gap < 0:
            raise ValueError("max_gap must be a finite nonnegative scalar or None")
    if len(dies) == 0:
        return np.empty((0, 4), dtype=np.float64)

    # Corner order: lower-left, upper-left, lower-right, upper-right.
    corners = np.stack(
        [dies[:, [0, 1]], dies[:, [0, 3]], dies[:, [2, 1]], dies[:, [2, 3]]],
        axis=1,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        lower, upper = corners - extent, corners + extent
    if (not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper))
            or np.any(lower >= corners) or np.any(upper <= corners)):
        raise ValueError("corner_extent cannot be represented at these coordinates")
    patches = np.concatenate([lower, upper], axis=2)

    for axis in (0, 1):
        _extend_across_gaps(dies, corners, patches, extent, axis, max_gap)

    return _rectangular_union(patches.reshape(-1, 4))


def _validate(boxes: ArrayLike, corner_extent: float) -> tuple[NDArray, float]:
    try:
        raw_extent = np.asarray(corner_extent)
        if raw_extent.ndim != 0:
            raise ValueError("not a scalar")
        extent = float(raw_extent)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("corner_extent must be a finite positive scalar") from error
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("corner_extent must be a finite positive scalar")

    try:
        dies = np.array(boxes, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("boxes must contain finite coordinates with shape (n, 4)") from error
    if dies.shape == (0,):
        dies = dies.reshape(0, 4)
    if dies.ndim != 2 or dies.shape[1] != 4:
        raise ValueError("boxes must have shape (n, 4)")
    if not np.all(np.isfinite(dies)):
        raise ValueError("boxes must contain finite coordinates")
    with np.errstate(over="ignore", invalid="ignore"):
        spans = dies[:, 2:] - dies[:, :2]
    invalid = np.flatnonzero(np.any((spans <= 0) | ~np.isfinite(spans), axis=1))
    if invalid.size:
        raise ValueError(
            f"boxes[{invalid[0]}] requires x1 < x2 and y1 < y2 with finite side lengths"
        )
    return dies, extent


def _extend_across_gaps(dies, corners, patches, extent, axis, max_gap):
    """Sweep transverse slabs; consecutive dies are the visible neighbors."""
    transverse = 1 - axis
    levels = np.unique(dies[:, [transverse, transverse + 2]])
    for low, high in zip(levels[:-1], levels[1:]):
        active = np.flatnonzero(
            (dies[:, transverse] < high) & (dies[:, transverse + 2] > low)
        )
        active = active[np.argsort(dies[active, axis], kind="stable")]
        for first, second in zip(active[:-1], active[1:]):
            end, start = dies[first, axis + 2], dies[second, axis]
            if end > start:
                raise ValueError(
                    f"boxes[{first}] and boxes[{second}] have overlapping interiors"
                )
            if end == start or (max_gap is not None and float(start) - float(end) > max_gap):
                continue

            # Include slab endpoints: gap ends are precisely where corners lie.
            first_corners = (
                (corners[first, :, axis] == end)
                & (corners[first, :, transverse] >= low)
                & (corners[first, :, transverse] <= high)
            )
            second_corners = (
                (corners[second, :, axis] == start)
                & (corners[second, :, transverse] >= low)
                & (corners[second, :, transverse] <= high)
            )
            patches[first, first_corners, axis + 2] = np.maximum(
                patches[first, first_corners, axis + 2], start + extent
            )
            patches[second, second_corners, axis] = np.minimum(
                patches[second, second_corners, axis], end - extent
            )


def _rectangular_union(boxes):
    """Merge Y intervals per X slab and continue identical runs across slabs."""
    levels = np.unique(boxes[:, [0, 2]])
    open_rectangles = {}
    finished = []
    for left, right in zip(levels[:-1], levels[1:]):
        active = boxes[(boxes[:, 0] < right) & (boxes[:, 2] > left)]
        intervals = sorted(map(tuple, active[:, [1, 3]]))
        merged = []
        for bottom, top in intervals:
            if merged and bottom <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], top))
            else:
                merged.append((bottom, top))

        next_rectangles = {}
        for interval in merged:
            rectangle = open_rectangles.pop(interval, None)
            if rectangle is None:
                rectangle = [left, interval[0], right, interval[1]]
            else:
                rectangle[2] = right
            next_rectangles[interval] = rectangle
        finished.extend(open_rectangles.values())
        open_rectangles = next_rectangles
    finished.extend(open_rectangles.values())
    return np.asarray(sorted(finished), dtype=np.float64).reshape(-1, 4)
