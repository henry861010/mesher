"""Find local die-corner control areas, including adjoining die gaps."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def get_mm2_control_area(
    boxes: ArrayLike, *, corner_extent: float = 25.0
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
        _extend_across_gaps(dies, corners, patches, extent, axis)

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


def _extend_across_gaps(dies, corners, patches, extent, axis):
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
            if end == start:
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
