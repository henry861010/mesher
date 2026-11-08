"""Bounded inward Quad4 rings with a frozen, finely sampled perimeter.

This module fills one already selected repair face. It never discovers slots
or invokes the layer/repair pipeline again.
"""
from __future__ import annotations

from math import ceil

import numpy as np
from mesher.tolerance import subdivision_count

from ...unit.build_unit_1_1_x_y import _validate_patch
from .._geometry import area, cross
from .._limits import ElementLimitExceeded, check_element_limit
from .._construction.line import _plan_count
from .._construction.units import CoordinatePool, UnitPlan, merge_unit_plans, quality_score


def _logical_sides(boundary, tol):
    """Retain every boundary segment, grouping collinear runs only."""
    points = np.asarray(boundary, dtype=float)
    if area(points[:, :2]) < 0:
        points = points[::-1].copy()
    vectors = np.roll(points[:, :2], -1, axis=0) - points[:, :2]
    previous = np.roll(vectors, 1, axis=0)
    turns = cross(previous, vectors)
    threshold = tol * np.maximum(np.linalg.norm(previous, axis=1),
                                 np.linalg.norm(vectors, axis=1))
    if np.any(turns < -threshold):
        raise ValueError("adaptive rings require a convex logical face")
    corners = np.flatnonzero(turns > threshold)
    if len(corners) < 3:
        raise ValueError("adaptive face has fewer than three corners")
    sides = []
    for index, start in enumerate(corners):
        end = int(corners[(index + 1) % len(corners)])
        indices = np.arange(int(start), end + (len(points) if end <= start else 0) + 1)
        sides.append(points[indices % len(points)].copy())
    return tuple(sides)


def _offset(sides, tol):
    """Return linear inset vertex velocities and the first side collapse."""
    corners = np.asarray([side[0] for side in sides])
    vectors = np.roll(corners[:, :2], -1, axis=0) - corners[:, :2]
    lengths = np.linalg.norm(vectors, axis=1)
    tangent = vectors / lengths[:, None]
    normals = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    velocities = []
    for previous, current in zip(np.roll(normals, 1, axis=0), normals):
        matrix = np.asarray((previous, current))
        if abs(float(np.linalg.det(matrix))) <= 64 * np.finfo(float).eps:
            raise ValueError("adaptive face has parallel adjacent supports")
        velocities.append(np.linalg.solve(matrix, np.ones(2)))
    velocities = np.asarray(velocities)
    slope = np.sum((np.roll(velocities, -1, axis=0) - velocities) * tangent, axis=1)
    contracting = slope < -64 * np.finfo(float).eps
    events = -lengths[contracting] / slope[contracting]
    if not len(events):
        raise ValueError("adaptive face has no finite inset limit")
    return corners, velocities, float(events.min())


def _inner_counts(corners, target):
    lengths = np.linalg.norm(np.roll(corners[:, :2], -1, axis=0) - corners[:, :2], axis=1)
    with np.errstate(over="ignore", invalid="ignore"):
        ratios = lengths / target
    if not np.all(np.isfinite(ratios)):
        raise ValueError("adaptive sampling count is not representable")
    # Python integers keep preflight arithmetic exact even when a requested
    # array would exceed NumPy's platform-sized index range.
    counts = [subdivision_count(float(length), target) for length in lengths]
    if sum(counts) % 2:
        # A closed Quad4 core needs an even boundary. Add one segment where
        # its normalized size-error penalty is smallest; never move a corner.
        index = min(range(len(counts)), key=lambda i: (
            (counts[i] + 1) * (ratios[i] / (counts[i] + 1) - 1) ** 2
            - counts[i] * (ratios[i] / counts[i] - 1) ** 2, i))
        counts[index] += 1
    return counts


def _sample_inner(corners, counts):
    if sum(counts) > np.iinfo(np.intp).max:
        raise ValueError("adaptive sampling exceeds the platform node-index range")
    sides = []
    for a, b, count in zip(corners, np.roll(corners, -1, axis=0), counts):
        sides.append(np.asarray([a, *(a + (b - a) * (i / int(count))
                                        for i in range(1, int(count))), b]))
    return tuple(sides)


def _as_unit(patch, method):
    return UnitPlan(patch.nodes, patch.quads,
                    (patch.source_ids, patch.right_ids,
                     patch.outer_ids[::-1], patch.left_ids[::-1]),
                    method, patch.quality)


def _ring(sides, inner, tol, limit, cache):
    """Two cyclic seam-parity assignments, with no recursive search."""
    candidates, limits = [], []
    for initial in (1, 2):
        seam_counts = [initial]
        for outer_side, inner_side in zip(sides, inner):
            bit = ((len(outer_side) + len(inner_side) - 2) % 2) ^ (seam_counts[-1] % 2)
            seam_counts.append(1 if bit else 2)
        if seam_counts[-1] != initial:
            continue
        seams = []
        for outer_side, inner_side, count in zip(sides, inner, seam_counts):
            a, b = outer_side[0], inner_side[0]
            seams.append(np.asarray((a, b)) if count == 1
                         else np.asarray((a, .5 * (a + b), b)))
        units, total = [], 0
        for index, (outer_side, inner_side) in enumerate(zip(sides, inner)):
            try:
                patch = _plan_count(outer_side, inner_side, seams[index],
                                    seams[(index + 1) % len(seams)], 0., tol,
                                    build_cache=cache,
                                    max_added_elements=None if limit is None else limit - total)
            except ElementLimitExceeded as error:
                if limit is not None:
                    error.context.update(limit=limit,
                                         required_elements=error.context['required_elements'] + total)
                limits.append(error)
                break
            except ValueError:
                break
            if patch is None:
                break
            units.append(_as_unit(patch, "adaptive_strip"))
            total += len(patch.quads)
        else:
            candidates.append(units)
    if not candidates:
        if limits:
            raise limits[0]
        raise ValueError("neither cyclic seam assignment builds an inward ring")
    return min(candidates, key=lambda units: (max(unit.quality for unit in units),
                                              sum(len(unit.quads) for unit in units)))


def _structured_core(sides, tol, limit):
    if len(sides) != 4:
        raise ValueError("structured core needs four logical sides")
    bottom, right, top, left = sides
    n, m = len(bottom) - 1, len(right) - 1
    if len(top) != n + 1 or len(left) != m + 1:
        raise ValueError("structured core has unequal opposite counts")
    check_element_limit(n * m, limit, stage="slot_repair_core")
    top, left = top[::-1], left[::-1]
    nodes = []
    for j in range(m + 1):
        v = j / m
        for i in range(n + 1):
            u = i / n
            if j == 0:
                point = bottom[i]
            elif j == m:
                point = top[i]
            elif i == 0:
                point = left[j]
            elif i == n:
                point = right[j]
            else:
                bilinear = ((1-u)*(1-v)*bottom[0] + u*(1-v)*bottom[-1]
                            + u*v*top[-1] + (1-u)*v*top[0])
                point = ((1-v)*bottom[i] + v*top[i] + (1-u)*left[j]
                         + u*right[j] - bilinear)
            nodes.append(point)
    nodes = np.asarray(nodes)
    grid = np.arange((n + 1) * (m + 1), dtype=np.int64).reshape(m + 1, n + 1)
    quads = np.column_stack((grid[:-1, :-1].ravel(), grid[:-1, 1:].ravel(),
                            grid[1:, 1:].ravel(), grid[1:, :-1].ravel()))
    ids = (grid[0], grid[:, -1], grid[-1, ::-1], grid[::-1, 0])
    _validate_patch(nodes, quads, ids)
    return [UnitPlan(nodes, quads, ids, "structured_core", quality_score(nodes, quads))]


def _core(sides, tol, limit, cache, legacy):
    errors = []
    try:
        return _structured_core(sides, tol, limit), "structured"
    except ElementLimitExceeded as error:
        errors.append(error)
    except ValueError:
        pass
    if len(sides) == 4:
        # Close elongated cores across their narrow dimension. This preserves
        # all existing perimeter samples instead of making a long fan.
        for rotation in sorted((0, 1), key=lambda i: -max(
                np.linalg.norm(sides[i][-1, :2] - sides[i][0, :2]),
                np.linalg.norm(sides[i + 2][-1, :2] - sides[i + 2][0, :2]))):
            ordered = sides[rotation:] + sides[:rotation]
            bottom, right, top, left = ordered
            try:
                patch = _plan_count(bottom, top[::-1], left[::-1], right, 0., tol,
                                    build_cache=cache, max_added_elements=limit)
            except ElementLimitExceeded as error:
                errors.append(error)
                continue
            except ValueError:
                continue
            if patch is not None:
                return [_as_unit(patch, "strip_core")], "strip"
    boundary = np.vstack([side[:-1] for side in sides])
    try:
        return legacy(boundary, limit), "direct"
    except ValueError:
        if errors:
            raise errors[0]
        raise


def build_adaptive_face(boundary, boundary_size, max_size, tol, limit, cache, legacy):
    """Fill one fixed face, returning isolated units and sizing diagnostics."""
    try:
        sides = _logical_sides(boundary, tol)
        _, _, first_event = _offset(sides, tol)
    except ValueError:
        return legacy(boundary, limit), {"ring_count": 0, "ring_target_sizes": [],
                                        "stop_reason": "unsupported_geometry", "core_method": "direct"}
    units, targets, total = [], [], 0
    current_size = boundary_size
    stop_reason = "offset_limit"
    # Every successful inset consumes at least boundary_size. The bound is
    # derived once, so no candidate failure can restart or extend this loop.
    bound = max(1, ceil(first_event / boundary_size))
    if max_size <= boundary_size * (1 + 64 * np.finfo(float).eps):
        bound = 0
        stop_reason = "no_coarsening"
    for _ in range(bound):
        if max(len(side) - 1 for side in sides) <= 2:
            stop_reason = "core_size"
            break
        corners, velocities, event = _offset(sides, tol)
        target = min(2 * current_size, max_size)
        if target > .5 * event + tol:
            break
        inner_corners = corners.copy()
        inner_corners[:, :2] += target * velocities
        outer_area, inner_area = area(corners[:, :2]), area(inner_corners[:, :2])
        perimeter = float(np.linalg.norm(np.roll(corners[:, :2], -1, axis=0)
                                         - corners[:, :2], axis=1).sum())
        area_margin = tol * perimeter
        if (not np.isfinite(inner_area) or inner_area <= area_margin
                or outer_area - inner_area <= area_margin):
            stop_reason = "offset_limit"
            break
        counts = _inner_counts(inner_corners, target)
        outside = sum(len(side) - 1 for side in sides)
        inside = sum(counts)
        # A quadrangulated annulus needs at least (B_outer+B_inner)/2
        # cells, and its core at least B_inner/2-1. Guard both before any
        # inner-side arrays are allocated.
        lower_bound = (outside + inside) // 2 + inside // 2 - 1
        check_element_limit(total + lower_bound, limit, stage="slot_repair_ring_count_preflight",
                            count_kind="lower_bound")
        inner = _sample_inner(inner_corners, counts)
        try:
            ring = _ring(sides, inner, tol, None if limit is None else limit - total, cache)
        except ElementLimitExceeded as error:
            if limit is not None:
                error.context.update(limit=limit,
                                     required_elements=error.context['required_elements'] + total)
            raise
        except ValueError:
            stop_reason = "ring_infeasible"
            break
        count = sum(len(unit.quads) for unit in ring)
        check_element_limit(total + count, limit, stage="slot_repair_ring")
        units.extend(ring)
        total += count
        targets.append(float(target))
        sides, current_size = inner, target
    try:
        core, core_method = _core(sides, tol, None if limit is None else limit - total, cache, legacy)
    except ElementLimitExceeded as error:
        if limit is not None:
            error.context.update(limit=limit,
                                 required_elements=error.context['required_elements'] + total)
        raise
    units.extend(core)
    check_element_limit(sum(len(unit.quads) for unit in units), limit, stage="slot_repair_face")
    nodes, quads, _ = merge_unit_plans(units, tol)
    pool = CoordinatePool(tol)
    for point in nodes:
        pool.intern(point)
    boundary_ids = pool.chain(boundary)
    if len(pool.points) != len(nodes):
        raise ValueError("adaptive face omitted a frozen boundary node")
    _validate_patch(nodes, quads, (np.r_[boundary_ids, boundary_ids[0]],))
    return units, {"ring_count": len(targets), "ring_target_sizes": targets,
                   "stop_reason": stop_reason, "core_method": core_method}


__all__ = ["build_adaptive_face"]
