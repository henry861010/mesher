"""Build directional line-pair controls from axis-aligned boxes."""
from __future__ import annotations

import math

import numpy as np

from ..generators._holed_geometry import OrthogonalDomain
from ._input import contains_box, scalar
from ._types import LinePair


def _scalar(value, name, *, zero=False):
    value = np.asarray(value)
    if value.ndim != 0:
        raise ValueError(f"{name} must be a scalar distance")
    return scalar(value.item(), name, zero=zero)


def _coordinates(low, high, spacing):
    length = high - low
    ratio = length / spacing
    if not math.isfinite(ratio) or ratio >= np.iinfo(np.intp).max:
        raise ValueError("spacing requires too many points to represent")
    values = low + np.arange(max(1, math.ceil(ratio)), dtype=float) * spacing
    # Only absorb arithmetic roundoff at the terminal point, never a physical
    # remainder using the mesher's (much larger) geometric tolerance.
    rounding = min(spacing / 2, 4 * max(map(math.ulp, (low, high, length))))
    values = values[(values == low) | ((values < high) & (high - values > rounding))]
    values = np.r_[values, high]
    if np.any(np.diff(values) <= 0):
        raise ValueError("spacing cannot produce distinct points at these coordinates")
    return values.tolist()


def _distances(domain, box, axis):
    start, end = box[axis], box[axis + 2]
    low, high = box[1 - axis], box[3 - axis]
    outline = domain.loops[0]
    levels = sorted({low, *(float(v) for v in outline[:, 1 - axis] if low < v < high)})
    left, right = -math.inf, math.inf
    for value in levels:
        interval = next(i for i in domain.section_after(axis, value)
                        if i.start <= start and end <= i.end)
        left, right = max(left, interval.start), min(right, interval.end)

    # Open slabs above cover the whole side interior. Also retain first
    # contacts at its endpoints (e.g. a notch ending exactly at a box corner).
    for first, last in zip(outline, np.roll(outline, -1, axis=0)):
        if first[axis] != last[axis]:
            continue
        a, b = sorted((first[1 - axis], last[1 - axis]))
        if b < low or a > high:
            continue
        position = float(first[axis])
        if position <= start:
            left = max(left, position)
        if position >= end:
            right = min(right, position)
    return start - left, right - end


def get_pair_line(boxes, expand_size: float, outer_outline=None, distance=None,
                  *, spacing: float) -> list[LinePair]:
    """Return X then Y controls for each ``[x1, y1, x2, y2]`` box.

    ``spacing`` is a required positive distance. Points start at the lower
    coordinate, advance by this fixed spacing, and include the exact endpoint;
    the final interval may be shorter. Each pair stores points on ``line1``.

    All four sides use the finite, nonnegative ``expand_size`` by default.
    Supply both ``outer_outline`` and nonnegative ``distance`` to replace each
    side's expansion with its outward normal distance to the outline when
    strictly below ``distance`` (even if larger than ``expand_size``). Concave
    outlines use the first contact across the entire side, including endpoints.

    The outline is a simple orthogonal XY loop, validated with the package's
    default geometric tolerance. Boxes must lie within it; touching is allowed.
    Boxes and sampled points are not snapped: meshing resolution is checked by
    ``imprint_lines``. Invalid inputs raise ``ValueError``. Inputs are not mutated.
    """
    expand = _scalar(expand_size, "expand_size", zero=True)
    step = _scalar(spacing, "spacing")
    if (outer_outline is None) != (distance is None):
        raise ValueError("outer_outline and distance must be provided together")
    threshold = None if distance is None else _scalar(distance, "distance", zero=True)
    domain = None if outer_outline is None else OrthogonalDomain(outer_outline)
    try:
        bounds = np.array(boxes if isinstance(boxes, np.ndarray) else list(boxes), dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("boxes must have shape (n, 4) with finite coordinates") from error
    if bounds.shape == (0,):
        return []
    if bounds.ndim != 2 or bounds.shape[1:] != (4,) or not np.all(np.isfinite(bounds)):
        raise ValueError("boxes must have shape (n, 4) with finite coordinates")
    with np.errstate(over="ignore", invalid="ignore"):
        spans = bounds[:, 2:] - bounds[:, :2]
    if np.any(spans <= 0) or not np.all(np.isfinite(spans)):
        raise ValueError("boxes require x1 < x2 and y1 < y2 with finite side lengths")

    result: list[LinePair] = []
    for index, box in enumerate(bounds.tolist()):
        if domain is not None and not contains_box(domain, box):
            raise ValueError(f"boxes[{index}] must lie within outer_outline")
        x1, y1, x2, y2 = box
        for axis in (0, 1):
            expands = [expand, expand]
            if domain is not None:
                expands = [d if d < threshold else expand for d in _distances(domain, box, axis)]
            if axis == 0:
                line1, line2 = [[x1, y1], [x1, y2]], [[x2, y1], [x2, y2]]
                points = [[x1, y] for y in _coordinates(y1, y2, step)]
            else:
                line1, line2 = [[x1, y1], [x2, y1]], [[x1, y2], [x2, y2]]
                points = [[x, y1] for x in _coordinates(x1, x2, step)]
            result.append(LinePair(direction="x" if axis == 0 else "y",
                                   line1=line1, line2=line2, points=points,
                                   line1_expand_size=expands[0], line2_expand_size=expands[1]))
    return result
