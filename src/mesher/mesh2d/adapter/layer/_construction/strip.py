"""Partition a LINE into even Quad4 patches with unsplit transverse interfaces."""

from __future__ import annotations

import numpy as np

from .....tolerance import GeometryToleranceError, resolve_tolerance
from ...unit.build_unit_1_1_1_x import _physical_boundary
from .._limits import ElementLimitExceeded, check_element_limit
from .units import UnitPlan, assemble_section, build_unit_patch, xyz_chain


def _frame(source, outer):
    vector = outer[-1, :2] - outer[0, :2]
    tangent = vector / np.hypot(*vector)
    basis = np.column_stack((tangent, (-tangent[1], tangent[0])))
    return ((source[:, :2] - outer[0, :2]) @ basis,
            (outer[:, :2] - outer[0, :2]) @ basis)


def _shear_metrics(source_xy, outer_xy, internal, tol):
    internal = np.asarray(internal, dtype=int)
    if not len(internal):
        return 0., 0.
    delta = outer_xy[internal[:, 1]] - source_xy[internal[:, 0]]
    shear = np.abs(delta[:, 0]) / np.maximum(np.abs(delta[:, 1]), tol)
    return float(np.mean(shear ** 2)), float(shear.max())


def alignment_metrics(source, outer, cuts, tol):
    """Mean squared and maximum absolute shear over the internal (i, j) cuts."""
    source_xy, outer_xy = _frame(np.asarray(source), np.asarray(outer))
    return _shear_metrics(source_xy, outer_xy, cuts, tol)


def _nearest(values, target, low, high, step=1):
    # low and high have the same admissible parity for step=2.
    index = int(np.searchsorted(values, target))
    upper = min(high, max(low, index + (low - index) % step))
    lower = max(low, upper - step)
    return lower if abs(values[lower] - target) <= abs(values[upper] - target) else upper


def _sweep(source_xy, outer_xy, left, right, tol, cap):
    n, m = len(source_xy) - 1, len(outer_xy) - 1
    coarse = n >= m
    short, long = (outer_xy, source_xy) if coarse else (source_xy, outer_xy)
    a, b = len(short) - 1, len(long) - 1
    long_stations = long[:, 0]

    def interval(start, end):
        width = end - start
        entry = left if start == 0 else 1
        exit_ = right if end == a else 1
        parity = (entry + exit_ + width) % 2
        low, high = 1, b
        if cap is not None:
            if coarse:
                high = min(high, width * cap)
            else:
                low = max(low, (width + cap - 1) // cap)
        return low + (parity - low) % 2, high - (high - parity) % 2

    # The sums of legal advances all have the same parity at each station.
    # Two singleton units can be combined into a width-two unit at the
    # same long indices; its allowed interval includes their summed advances.
    # Thus storing just the extrema is sufficient for count feasibility.
    minima, maxima = [b + 1] * (a + 1), [-1] * (a + 1)
    minima[a] = maxima[a] = 0
    for start in range(a - 1, -1, -1):
        for end in range(start + 1, min(start + 2, a) + 1):
            low, high = interval(start, end)
            if low <= high and minima[end] <= b:
                minima[start] = min(minima[start], low + minima[end])
                maxima[start] = max(maxima[start], min(b, high + maxima[end]))
    if not minima[0] <= b <= maxima[0]:
        return None

    cuts = [(0, 0)]
    start = position = 0
    while start < a:
        options = []
        for end in range(start + 1, min(start + 2, a) + 1):
            low, high = interval(start, end)
            lo = max(position + low, b - maxima[end])
            hi = min(position + high, b - minima[end])
            lo += (position + low - lo) % 2
            hi -= (hi - position - low) % 2
            if lo > hi:
                continue
            index = _nearest(long_stations, short[end, 0], lo, hi, 2)
            delta = long[index] - short[end]
            slope = abs(delta[0]) / max(abs(delta[1]), tol) if end != a else 0.
            # A full cut is preferable to building a single all-subdivided cap
            # when the strip has an admissible interior cut.
            isolated = start == 0 and end == a and min(a, b, left, right) > 1
            options.append((isolated, slope, end, index))
        if not options:
            return None
        _, _, start, position = min(options)
        cuts.append((position, start) if coarse else (start, position))
    return cuts


def fixed_cut_plans(source, outer, left, right, tol, cap=None, *, cut_stations=None):
    """Yield at most two partitions, with one or two short-side segments per unit.

    Each full-height cut has even prefix boundary parity, independent of its
    predecessor. Reachable suffix counts reserve room for both physical ends.
    The two sweeps start at opposite ends and are ranked on actual cut shear.
    """
    source_xy, outer_xy = _frame(np.asarray(source), np.asarray(outer))
    if cut_stations is not None:
        source_xy[:, 0] = cut_stations
    if any(np.any(np.diff(chain[:, 0]) <= tol) for chain in (source_xy, outer_xy)):
        raise ValueError("LINE nodes must be strictly ordered along the line axis")
    n, m = len(source_xy) - 1, len(outer_xy) - 1
    lseg, rseg = len(left) - 1, len(right) - 1
    if (n + m + lseg + rseg) % 2 or (cap is not None and n > m * cap):
        return
    if min(n, m) == 1:
        yield [(0, 0), (n, m)]
        return
    if n == m and lseg % 2 and rseg % 2 and np.array_equal(source_xy[:, 0], outer_xy[:, 0]):
        yield [(index, index) for index in range(n + 1)]
        return
    forward = _sweep(source_xy, outer_xy, lseg, rseg, tol, cap)
    backward = _sweep(-source_xy[::-1], -outer_xy[::-1], rseg, lseg, tol, cap)
    candidates = []
    if forward:
        candidates.append(forward)
    if backward:
        reversed_cuts = [(n - i, m - j) for i, j in reversed(backward)]
        if reversed_cuts != forward:
            candidates.append(reversed_cuts)

    def score(cuts):
        mean, peak = _shear_metrics(source_xy, outer_xy, cuts[1:-1], tol)
        return peak, mean, -len(cuts), cuts

    yield from sorted(candidates, key=score)


def _edge_assignment(source, outer, cap):
    """Assign slices of the source chain to individual ordered outer edges."""
    n, m = len(source) - 1, len(outer) - 1
    if m == 1:
        return [n]
    if n == m:
        return [1] * m
    lower = int(n >= m)
    upper = min(n, cap) if cap is not None else n
    previous = 0
    counts = []
    for j in range(1, m):
        remaining = m - j
        low = max(previous + lower, n - remaining * upper)
        high = min(previous + upper, n - remaining * lower)
        index = _nearest(source, outer[j], low, high)
        counts.append(index - previous)
        previous = index
    return [*counts, n - previous]


_DIRECT_SIDES = tuple(np.asarray((i, (i + 1) % 4)) for i in range(4))
_CCW = np.array([[0, 1, 2, 3]])
_CW = np.array([[0, 3, 2, 1]])


def _direct_quads(source, outer, cuts, direct, tol):
    """Validate independent single Quad4s together, in each cell's local scale."""
    if not len(direct):
        return {}
    i, j = cuts[direct].T
    points = np.stack((source[i], source[i + 1], outer[j + 1], outer[j]), axis=1)
    xy = points[:, :, :2]
    resolve_tolerance(tol).check_resolution(xy, stage="unit_boundary", source="fixed_boundary")
    vectors = np.roll(xy, -1, axis=1) - xy
    lengths = np.hypot(vectors[:, :, 0], vectors[:, :, 1])
    if np.any(lengths <= tol):
        bad = np.flatnonzero(np.any(lengths <= tol, axis=1))[0]
        _physical_boundary(points[bad], np.arange(4), tol)
    scale = np.ptp(xy, axis=1).max(axis=1)
    vectors /= scale[:, None, None]
    following = np.roll(vectors, -1, axis=1)
    crosses = vectors[:, :, 0] * following[:, :, 1] - vectors[:, :, 1] * following[:, :, 0]
    sign = np.sign(crosses[:, 0])
    if np.any(crosses * sign[:, None] <= 64 * np.finfo(float).eps):
        raise ValueError("single-cell LINE unit is nonconvex or degenerate")
    norm = lengths / scale[:, None]
    inv_j = norm * np.roll(norm, -1, axis=1) / np.abs(crosses)
    quality = np.maximum(lengths.max(axis=1) / lengths.min(axis=1), inv_j.max(axis=1))
    return {int(k): UnitPlan(nodes, _CCW if winding > 0 else _CW,
                             _DIRECT_SIDES, "direct", float(q))
            for k, nodes, winding, q in zip(direct, points, sign, quality)}


def _paired_unit(sides, i, j, tol, cache, stats):
    """Join two validated patches along their prescribed private transition.

    Each outer edge of a two-edge coarsening unit owns its exact source slice.
    The exterior cuts of the combined unit stay full height.
    """
    from ...unit.build_unit_1_1_x_y import _validate_patch

    source, right, reverse_outer, back = sides
    outer = reverse_outer[::-1]
    link = np.asarray((source[i], outer[j]))
    if (i + j + len(back)) % 2:
        link = np.asarray((source[i], .5 * (source[i] + outer[j]), outer[j]))
    parts = ((source[:i + 1], link, outer[:j + 1][::-1], back),
             (source[i:], right, outer[j:][::-1], link[::-1]))
    children = []
    for part in parts:
        stats["unit_trial_count"] = stats.get("unit_trial_count", 0) + 1
        children.append(build_unit_patch(part, tol, _cache=cache))
    first, second = children
    shared_a, shared_b = first.sides[1], second.sides[3][::-1]
    if not np.array_equal(first.nodes[shared_a], second.nodes[shared_b]):
        raise ValueError("two LINE transition patches disagree on their common edge")
    unique = np.ones(len(second.nodes), dtype=bool)
    unique[shared_b] = False
    remap = np.empty(len(second.nodes), dtype=int)
    remap[shared_b] = shared_a
    remap[unique] = len(first.nodes) + np.arange(np.sum(unique))
    nodes = np.vstack((first.nodes, second.nodes[unique]))
    quads = np.vstack((first.quads, remap[second.quads]))
    ids = (np.r_[first.sides[0], remap[second.sides[0]][1:]], remap[second.sides[1]],
           np.r_[remap[second.sides[2]], first.sides[2][1:]], first.sides[3])

    # Strict separation proves that the two conforming child meshes can only
    # meet along their shared chain. Use the full crossing check for bent
    # seams that leave the two half-planes, or near-collinear vertices.
    vector = link[-1, :2] - link[0, :2]
    normal = np.array((-vector[1], vector[0]))
    dist_a = (first.nodes[:, :2] - link[0, :2]) @ normal
    dist_b = (second.nodes[unique, :2] - link[0, :2]) @ normal
    dist_a = np.delete(dist_a, shared_a)
    margin = 64. * np.finfo(float).eps * np.ptp(nodes[:, :2], axis=0).max() ** 2
    if not ((dist_a.max() < -margin and dist_b.min() > margin) or
            (dist_b.max() < -margin and dist_a.min() > margin)):
        _validate_patch(nodes, quads, ids)
    fixed = (*first.fixed_interior, *shared_a[1:-1],
             *(remap[node] for node in second.fixed_interior))
    return UnitPlan(nodes, quads, ids, "paired", max(first.quality, second.quality),
                    metadata={"paired_cut": (i, j)}, fixed_interior=fixed)


def _adjacent_cuts(pairs, at, source_xy, outer_xy, tol, cap, cut_stations=None):
    """Best two local parity-preserving moves; both neighboring spans stay valid."""
    (si, sj), (i, j), (ei, ej) = pairs[at - 1:at + 2].tolist()
    options = []
    for di, dj in ((-2, 0), (2, 0), (0, -2), (0, 2),
                   (-1, -1), (1, 1), (-1, 1), (1, -1)):
        a, b = i + di, j + dj
        if not si < a < ei or not sj < b < ej:
            continue
        # A repaired cut on a widened front must still join matching stations.
        if cut_stations is not None and abs(cut_stations[a] - outer_xy[b, 0]) > tol:
            continue
        spans = ((a - si, b - sj), (ei - a, ej - b))
        if any(min(s, o) > 2 or (cap is not None and s > cap * o) for s, o in spans):
            continue
        delta = outer_xy[b] - source_xy[a]
        options.append((abs(delta[0]) / max(abs(delta[1]), tol), a, b))
    return [(a, b) for _, a, b in sorted(options)[:2]]


def build_strip(source, outer, left, right, cuts, size_error, tol, *, cache, stats,
                max_added_elements=None, cap=None, cut_stations=None):
    """Build one fixed partition, returning None for an unbuildable geometry."""
    source, outer, left, right = map(xyz_chain, (source, outer, left, right))
    pairs = np.asarray(cuts, dtype=np.int64)
    n, m = len(source) - 1, len(outer) - 1
    if (pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) < 2
            or tuple(pairs[0]) != (0, 0) or tuple(pairs[-1]) != (n, m)):
        raise ValueError("LINE cuts must contain the section endpoints")
    sizes = np.diff(pairs, axis=0)
    if np.any(sizes < 1):
        raise ValueError("LINE cuts must consume both sides in order")
    direct = np.all(sizes == 1, axis=1)
    direct[0] &= len(left) == 2
    direct[-1] &= len(right) == 2
    source_xy, outer_xy = _frame(source, outer)
    assignment_x = source_xy[:, 0] if cut_stations is None else cut_stations
    units = [None] * len(sizes)
    pairs = pairs.copy()
    try:
        single = _direct_quads(source, outer, pairs, np.flatnonzero(direct), tol)
    except GeometryToleranceError:
        raise
    except ValueError:
        single = {}
    stats["direct_quad_count"] = stats.get("direct_quad_count", 0) + len(single)
    # Several seam states can have precisely the same interior unit in the
    # same world frame. Such units share the already validated restored patch.
    world_units = cache.setdefault("world_units", {}) if cache is not None else {}

    def construct(k, start, end, predefined=None):
        i, j = start
        ni, nj = end
        entry = left if k == 0 else np.asarray((source[i], outer[j]))
        exit_ = right if k == len(sizes) - 1 else np.asarray((source[ni], outer[nj]))
        ns, no = int(ni - i), int(nj - j)
        if (min(ns, no) > 2 or (ns + no + len(entry) + len(exit_)) % 2
                or (cap is not None and ns > cap * no)):
            raise ValueError("illegal LINE unit boundary counts")
        if predefined is not None:
            return predefined
        sides = (source[i:ni + 1], exit_, outer[j:nj + 1][::-1], entry[::-1])
        key = (tol, cap, *(side.tobytes() for side in sides))
        if key in world_units:
            return world_units[key]
        isolated = min(map(len, sides)) > 2
        capped_pair = cap is not None and no == 2 and ns > no
        if isolated or capped_pair or (cap == 1 and ns == 2 and no > ns):
            # Also handles an isolated short cap whose four edges are all
            # split: whichever longitudinal chain has two segments supplies
            # a one-segment side to each child.
            if no <= ns:
                v = 1
                u = _edge_assignment(assignment_x[i:ni + 1], outer_xy[j:nj + 1, 0], cap)[0]
            else:
                u = 1
                v = _nearest(outer_xy[j:nj + 1, 0], assignment_x[i + 1], 1, no - 1)
            unit = _paired_unit(sides, u, v, tol, cache, stats)
        else:
            stats["unit_trial_count"] = stats.get("unit_trial_count", 0) + 1
            unit = build_unit_patch(sides, tol, _cache=cache)
        world_units[key] = unit
        return unit

    quad_count = 0
    for k in range(len(units)):
        if units[k] is not None:  # Already built as the second owner of a repaired cut.
            continue
        try:
            units[k] = construct(k, pairs[k], pairs[k + 1], single.get(k))
            quad_count += len(units[k].quads)
        except (GeometryToleranceError, ElementLimitExceeded):
            raise
        except ValueError:
            repaired = False
            repair_limit = None
            for at in (k + 1, k):
                if not 0 < at < len(units):
                    continue
                for replacement in _adjacent_cuts(pairs, at, source_xy, outer_xy, tol, cap, cut_stations):
                    stats["local_repair_count"] = stats.get("local_repair_count", 0) + 1
                    try:
                        first = construct(at - 1, pairs[at - 1], replacement)
                        second = construct(at, replacement, pairs[at + 1])
                    except (GeometryToleranceError, ElementLimitExceeded):
                        raise
                    except ValueError:
                        continue
                    old_count = sum(len(unit.quads) for unit in units[at - 1:at + 1]
                                    if unit is not None)
                    replacement_count = quad_count - old_count + len(first.quads) + len(second.quads)
                    try:
                        check_element_limit(replacement_count, max_added_elements, stage="section_builder")
                    except ElementLimitExceeded as error:
                        repair_limit = error
                        continue
                    units[at - 1:at + 1] = [first, second]
                    quad_count = replacement_count
                    pairs[at] = replacement
                    repaired = True
                    break
                if repaired:
                    break
            if not repaired:
                if repair_limit is not None:
                    raise repair_limit
                return None
        check_element_limit(quad_count, max_added_elements, stage="section_builder")
    result = assemble_section(units, source, outer, left, right, size_error, tol)
    maximum_count = 0
    for record, unit, (i, j), (ni, nj) in zip(result.units, units, pairs[:-1], pairs[1:]):
        source_stations, outer_stations = assignment_x[i:ni + 1], outer_xy[j:nj + 1, 0]
        if "paired_cut" in unit.metadata:
            # Each child owns the source slice used during construction.
            a, b = unit.metadata["paired_cut"]
            counts = (_edge_assignment(source_stations[:a + 1], outer_stations[:b + 1], cap)
                      + _edge_assignment(source_stations[a:], outer_stations[b:], cap))
        else:
            counts = _edge_assignment(source_stations, outer_stations, cap)
        record["source_segments_per_outer"] = counts
        maximum_count = max(maximum_count, max(counts))
    result.alignment_error, peak = _shear_metrics(source_xy, outer_xy, pairs[1:-1], tol)
    result.diagnostics.update(
        alignment_error=result.alignment_error, max_connector_shear=peak,
        max_alignment_angle=float(np.degrees(np.arctan(peak))),
        internal_cuts=pairs[1:-1].tolist(),
        max_actual_transition_count=maximum_count,
    )
    return result
