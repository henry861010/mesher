"""Normalize all fixed constraints once, with the package tolerance policy."""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from ...tolerance import canonicalize_points, resolve_tolerance
from ..generators._holed_geometry import OrthogonalDomain
from ..generators._holed_grid import _ElementBudget
from ._types import Band, ImprintLineError, Patterns


def scalar(value, name, *, zero=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite real number") from error
    if not math.isfinite(result) or (result < 0 if zero else result <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return result


def contains_box(domain, box):
    x0, y0, x1, y1 = box
    levels = sorted({y0, *(float(y) for y in domain.loops[0][:, 1] if y0 < y < y1)})
    return all(any(i.start <= x0 and x1 <= i.end for i in domain.section_after(0, y))
               for y in levels)


def prepare(element_size, line_pairs, pattern_lines, outer_outline,
            min_transition_size, pattern_snap_distance, tolerance, max_elements):
    policy = resolve_tolerance(tolerance)
    size = scalar(element_size, "element_size")
    minimum = scalar(min_transition_size, "min_transition_size")
    snap = scalar(pattern_snap_distance, "pattern_snap_distance", zero=True)
    if minimum <= policy.absolute or size < minimum:
        raise ValueError("tolerance < min_transition_size <= element_size is required")
    budget = _ElementBudget(max_elements)
    outer = np.array(outer_outline, dtype=float, copy=True)
    if outer.ndim != 2 or outer.shape[1:] != (2,) or len(outer) < 4:
        raise ValueError("outer_outline must have shape (n, 2), n >= 4")
    patterns = np.array(list(pattern_lines), dtype=float)
    if patterns.size == 0:
        patterns = np.empty((0, 2, 2))
    if patterns.ndim != 3 or patterns.shape[1:] != (2, 2):
        raise ValueError("pattern_lines must have shape (n, 2, 2)")
    policy.check_resolution(patterns, stage="imprint_input", source="pattern_lines")
    equal = patterns[:, 0] == patterns[:, 1]
    if np.any(np.sum(equal, axis=1) != 1):
        raise ValueError("pattern_lines must be nonzero and exactly axis aligned; fixed endpoints cannot move")
    raw = []
    arrays = [outer, patterns.reshape(-1, 2)]
    for index, pair in enumerate(line_pairs):
        if not isinstance(pair, Mapping):
            raise ValueError(f"line_pairs[{index}] must be a mapping")
        missing = {"direction", "line1", "line2", "points",
                   "line1_expand_size", "line2_expand_size"} - pair.keys()
        if missing:
            raise ValueError(f"line_pairs[{index}] is missing {sorted(missing)}")
        if pair.get("direction") not in ("x", "y"):
            raise ValueError(f"line_pairs[{index}].direction must be x or y")
        axis = 0 if pair["direction"] == "x" else 1
        lines = np.array([pair["line1"], pair["line2"]], dtype=float)
        points = np.array(pair["points"], dtype=float)
        if points.size == 0:
            points = np.empty((0, 2))
        if lines.shape != (2, 2, 2) or points.ndim != 2 or points.shape[1:] != (2,):
            raise ValueError(f"line_pairs[{index}] needs two (2, 2) lines and (n, 2) points")
        if not np.all(np.isfinite(points)):
            raise ValueError(f"line_pairs[{index}].points must be finite")
        # The longitudinal component of points is explicitly not a constraint.
        points[:, axis] = lines[0, 0, axis]
        expands = []
        for name in ("line1_expand_size", "line2_expand_size"):
            value = np.asarray(pair[name])
            if value.ndim != 0:
                raise ValueError(f"line_pairs[{index}].{name} must be a scalar distance")
            expands.append(scalar(value.item(), f"line_pairs[{index}].{name}", zero=True))
        raw.append((axis, expands, len(points)))
        arrays.extend((lines.reshape(-1, 2), points))
    points = np.vstack(arrays)
    policy.check_resolution(points, stage="imprint_input", source="fixed_constraints")
    fixed = np.zeros(len(points), dtype=bool)
    fixed[len(outer):len(outer) + patterns.size // 2] = True
    merged = canonicalize_points(points, policy, fixed=fixed,
                                 stage="imprint_input", source="fixed_constraints")
    offsets = np.cumsum([0, *map(len, arrays)])
    parts = [merged[a:b] for a, b in zip(offsets, offsets[1:])]
    domain = OrthogonalDomain(parts[0], tolerance=policy)
    span = np.asarray(domain.bounds[2:]) - domain.bounds[:2]
    for length in span:
        ratio = float(length) / size
        if not math.isfinite(ratio):
            budget.check(budget.limit + 1, "imprint_extent_preflight")
        budget.check(math.ceil(ratio), "imprint_extent_preflight")
    # Sort and deduplicate exact segments only. Do not normalize patterns again.
    unique = {}
    for index, p in enumerate(parts[1].reshape(-1, 2, 2)):
        p = p[::-1].copy() if tuple(p[1]) < tuple(p[0]) else p.copy()
        p.setflags(write=False)
        unique.setdefault(tuple(p.ravel()), (p, index))
    ordered = [unique[key] for key in sorted(unique)]
    patterns = Patterns((p for p, _ in ordered), (i for _, i in ordered))
    bands = []
    cores = []
    for index, (axis, expands, _) in enumerate(raw):
        lines = parts[2 + 2 * index].reshape(2, 2, 2)
        points = parts[3 + 2 * index]
        t = 1 - axis
        spans = np.sort(lines[:, :, t], axis=1)
        if (np.any(lines[:, 0, axis] != lines[:, 1, axis])
                or not np.array_equal(spans[0], spans[1])):
            raise ValueError(f"line_pairs[{index}] must be matching transverse cross sections")
        low, high = map(float, spans[0])
        ends = lines[:, 0, axis]
        order = np.argsort(ends)
        start, end = map(float, ends[order])
        if min(end - start, high - low) <= policy.absolute:
            raise ValueError(f"line_pairs[{index}] has no resolved rectangular area")
        if np.any(points[:, t] < low) or np.any(points[:, t] > high):
            raise ValueError(f"line_pairs[{index}].points lie outside the transverse span")
        coordinates = np.unique(np.r_[low, points[:, t], high])
        if np.any(np.diff(coordinates) <= policy.absolute):
            raise ValueError(f"line_pairs[{index}] has unresolved dense spacing")
        fronts = [start - expands[int(order[0])], end + expands[int(order[1])]]
        if not np.all(np.isfinite(fronts)):
            raise ValueError(f"line_pairs[{index}] expansion exceeds numeric range")
        band = Band(axis, start, end, low, high, coordinates, fronts, (index,))
        if not contains_box(domain, band.bounds):
            raise ImprintLineError("control rectangle leaves the domain", code="control_outside_domain",
                                   stage="input_validation", pair_indices=[index])
        bands.append(band)
        cores.append(band.bounds)
    bands.sort(key=lambda b: (b.axis, b.start, b.end, b.low, b.high, tuple(b.coordinates)))
    return domain, bands, cores, patterns, size, minimum, snap, policy, budget
