"""Build directional line-pair controls from axis-aligned boxes."""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import numpy as np

from ..generators._holed_geometry import OrthogonalDomain
from ._input import contains_box, scalar
from ._types import BoxPairSpec, LinePair


def _scalar(value, name, *, zero=False):
    value = np.asarray(value)
    if value.ndim != 0:
        raise ValueError(f"{name} must be a scalar distance")
    return scalar(value.item(), name, zero=zero)


def _finite_array(value, name):
    try:
        array = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must contain finite coordinates") from error
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite coordinates")
    return array


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


def get_pair_line(boxes: Iterable[BoxPairSpec], expand_size: float,
                  outer_outline=None, distance=None) -> list[LinePair]:
    """Build selected X then Y controls for each box specification.

    Each mapping requires ``box=[x1, y1, x2, y2]``. Optional ``x_points`` and
    ``y_points`` enable the respective pairs, even when empty. Points are finite
    (n, 2) absolute XY coordinates: X pairs use only Y, and Y pairs use only X.
    That transverse component must lie within the box; the other is unconstrained.
    Samples are copied without sorting, deduplication, projection or endpoint
    insertion. ``imprint_lines`` includes endpoints and checks mesh resolution.
    Omitting both point fields produces no pairs but still validates the box.

    Selected sides use the finite, nonnegative ``expand_size`` by default.
    Supply both ``outer_outline`` and nonnegative ``distance`` to replace each
    side's expansion with its outward normal distance to the outline when
    strictly below ``distance`` (even if larger than ``expand_size``). Concave
    outlines use the first contact across the entire side, including endpoints.

    The outline is a simple orthogonal XY loop, validated with the package's
    default geometric tolerance. Boxes must lie within it; touching is allowed.
    Empty input returns an empty list. Invalid inputs raise ``ValueError``.
    Inputs are not mutated and returned geometry is independently owned.
    """
    expand = _scalar(expand_size, "expand_size", zero=True)
    if (outer_outline is None) != (distance is None):
        raise ValueError("outer_outline and distance must be provided together")
    threshold = None if distance is None else _scalar(distance, "distance", zero=True)
    domain = None if outer_outline is None else OrthogonalDomain(outer_outline)
    try:
        specs = iter(boxes)
    except TypeError as error:
        raise ValueError("boxes must be an iterable of box specifications") from error

    result: list[LinePair] = []
    for index, spec in enumerate(specs):
        name = f"boxes[{index}]"
        if not isinstance(spec, Mapping):
            raise ValueError(f"{name} must be a mapping")
        if "box" not in spec:
            raise ValueError(f"{name}.box is required")
        unknown = spec.keys() - {"box", "x_points", "y_points"}
        if unknown:
            raise ValueError(f"{name} has unknown fields: {sorted(map(str, unknown))}")
        bounds = _finite_array(spec["box"], f"{name}.box")
        if bounds.shape != (4,):
            raise ValueError(f"{name}.box must have shape (4,)")
        with np.errstate(over="ignore", invalid="ignore"):
            spans = bounds[2:] - bounds[:2]
        if np.any(spans <= 0) or not np.all(np.isfinite(spans)):
            raise ValueError(f"{name}.box requires x1 < x2 and y1 < y2 with finite side lengths")
        box = bounds.tolist()
        if domain is not None and not contains_box(domain, box):
            raise ValueError(f"{name}.box must lie within outer_outline")
        x1, y1, x2, y2 = box
        for axis, field in enumerate(("x_points", "y_points")):
            if field not in spec:
                continue
            points = _finite_array(spec[field], f"{name}.{field}")
            if points.shape == (0,):
                points = points.reshape(0, 2)
            if points.ndim != 2 or points.shape[1:] != (2,):
                raise ValueError(f"{name}.{field} must have shape (n, 2)")
            transverse = points[:, 1 - axis]
            if np.any(transverse < box[1 - axis]) or np.any(transverse > box[3 - axis]):
                raise ValueError(f"{name}.{field} lie outside the transverse span")
            expands = [expand, expand]
            if domain is not None:
                expands = [d if d < threshold else expand for d in _distances(domain, box, axis)]
            if axis == 0:
                line1, line2 = [[x1, y1], [x1, y2]], [[x2, y1], [x2, y2]]
            else:
                line1, line2 = [[x1, y1], [x2, y1]], [[x1, y2], [x2, y2]]
            result.append(LinePair(direction="x" if axis == 0 else "y",
                                   line1=line1, line2=line2, points=points.tolist(),
                                   line1_expand_size=expands[0], line2_expand_size=expands[1]))
    return result
