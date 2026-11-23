"""Plan adapter cuts in a background grid with reserved transition rectangles."""
from __future__ import annotations

import numpy as np

from ._reserved_grid import background, BackgroundConflict
from ._constraints import Constraints
from ._unit import build_unit
from ._input import contains_box
from ._bands import limits, merge_bands, overlap
from ._types import CutTransition, ImprintPlan, ImprintLineError, rectangle
from ...tolerance import subdivision_count


def partition(low, high, mandatory, size, constraints=None, axis=0,
              tolerance=0, max_parts=1_000_000):
    """Split across the line, bounded by size and geometric resolution only."""
    values = sorted({low, high, *(float(v) for v in mandatory if low < v < high)})
    result = [low]
    rounding = 32 * np.finfo(float).eps
    for a, b in zip(values, values[1:]):
        first = subdivision_count(b - a, size)
        last = max_parts - len(result) + 1
        if tolerance > 0:
            last = min(last, int(np.ceil((b - a) / tolerance)) - 1)
        for n in range(first, last + 1):
            cuts = np.linspace(a, b, n + 1)
            if constraints is not None:
                cuts[1:-1] = constraints.align_generated(axis, cuts[1:-1], tolerance)
            widths = np.diff(cuts)
            if (np.all(widths > tolerance)
                    and np.all(widths <= size * (1 + rounding))):
                result.extend(cuts[1:].tolist())
                break
        else:
            return None
    return np.asarray(result)


def restore_supports(domain, bands, cores, patterns, transitions, policy):
    """Remove arithmetic roundtrip noise, preserving every fixed input support.

    This is an ULP-scale identity repair, not physical-distance snapping: a
    genuine small gap remains a constraint and is never silently erased.
    """
    for axis in (0, 1):
        fixed = set(map(float, domain.loops[0][:, axis]))
        for b in cores:
            fixed.update((b[axis], b[axis + 2]))
        for p in patterns:
            fixed.update(map(float, p[:, axis]))
        for b in bands:
            if b.axis != axis:
                fixed.update(map(float, b.coordinates))
        values = set(fixed)
        for t in transitions:
            values.update((t.bounds[axis], t.bounds[axis + 2]))
            if t.axis != axis:
                values.update(map(float, t.splits))
        ordered = sorted(values)
        mapping = {}
        i = 0
        while i < len(ordered):
            first = ordered[i]
            rounding = min(policy.absolute / 2, 8 * abs(float(np.spacing(first))))
            j = i + 1
            while j < len(ordered) and ordered[j] - first <= rounding:
                j += 1
            anchors = [v for v in ordered[i:j] if v in fixed]
            representative = anchors[0] if anchors else first
            # Input canonicalization already separates distinct fixed supports.
            if len(anchors) > 1:
                raise ImprintLineError("fixed supports are numerically indistinguishable",
                                       code="numeric_resolution_exceeded", stage="region_planning")
            mapping.update((v, representative) for v in ordered[i:j])
            i = j
        for t in transitions:
            b = list(t.bounds)
            b[axis], b[axis + 2] = mapping[b[axis]], mapping[b[axis + 2]]
            t.bounds = tuple(b)
            if t.axis != axis:
                t.splits = np.asarray([mapping[float(v)] for v in t.splits])
            else:
                bands[t.band].fronts[t.side] = t.inner


def candidate(domain, band, index, side, patterns, size, minimum, constraints, budget):
    sign = -1 if side == 0 else 1
    limit = limits(domain, band)[side]
    if not band.active[side]:
        return None
    transverse = 1 - band.axis
    visited = set()
    while True:
        inner = band.fronts[side]
        if inner in visited:
            raise ImprintLineError("transition candidate made no progress", code="planning_limit",
                                   stage="region_planning", reason="no_progress", band=index, side=side)
        visited.add(inner)
        # A perpendicular pattern can bound a transition before its nominal outer edge.
        supports = sorted({float(p[0, band.axis]) for p in patterns
                           if p[0, band.axis] == p[1, band.axis]
                           and min(p[:, transverse]) < band.high and max(p[:, transverse]) > band.low},
                          reverse=side == 0)
        outer = inner + sign * size
        crossing = [v for v in supports if 0 < sign * (v - inner) <= size]
        if crossing:
            # A pattern-cut grid cell may be smaller than global size. Retain
            # that legal cell instead of skipping every closely spaced pattern.
            outer = crossing[0]
        elif outer in supports:
            # Preserve the exact-face arithmetic of the zero-distance cut planner.
            inner = outer - sign * size
        if sign * (outer - limit) > 0:
            outer = limit
        if sign * (outer - inner) < minimum and sign * (outer - limit) < 0:
            band.fronts[side] = outer
            band.splits[side].clear()
            continue
        if sign * (outer - inner) < minimum or sign * (inner - limit) >= 0:
            band.active[side] = False
            return None
        # No inward snap can reduce an E request, including after clipping.
        band.fronts[side] = inner
        box = rectangle(band.axis, min(inner, outer), max(inner, outer), band.low, band.high)
        mandatory = set(band.splits[side])
        blockers = []
        # An exterior corner on a reservation can have no visibility chord.
        # Retain it explicitly rather than relying on background feedback.
        for p in domain.loops[0]:
            position, level = p[band.axis], p[transverse]
            if min(inner, outer) <= position <= max(inner, outer) and band.low <= level <= band.high:
                if position == outer and band.low < level < band.high:
                    mandatory.add(float(level))
                elif min(inner, outer) < position < max(inner, outer):
                    blockers.append(float(position))
        for p in patterns:
            a, b = sorted(p[:, band.axis])
            low, high = sorted(p[:, transverse])
            if a == b:
                if min(inner, outer) < a < max(inner, outer) and low < band.high and high > band.low:
                    blockers.append(a)
            elif band.low < low < band.high and max(a, min(inner, outer)) < min(b, max(inner, outer)):
                if a <= min(inner, outer) and b >= max(inner, outer):
                    mandatory.add(low)
                else:
                    blockers.append(a if side == 0 else b)
        splits = partition(band.low, band.high, mandatory, size, constraints,
                           transverse, domain.tolerance, budget.limit)
        if splits is None:
            for p in patterns:
                if p[0, transverse] == p[1, transverse] and band.low < p[0, transverse] < band.high:
                    if max(min(p[:, band.axis]), min(inner, outer)) < min(max(p[:, band.axis]), max(inner, outer)):
                        blockers.append(float(min(p[:, band.axis]) if side == 0 else max(p[:, band.axis])))
            if not blockers:
                band.active[side] = False
                return None
        if blockers:
            value = min(blockers) if side == 0 else max(blockers)
            if sign * (value - inner) <= 0:
                value = outer
            band.fronts[side] = value
            band.splits[side].clear()
            continue
        if not contains_box(domain, box):
            band.active[side] = False
            return None
        return CutTransition(index, band.axis, side, box, splits)


def neighbouring_caps(builder, transitions):
    """Pairs linked by a transverse visibility chord or a shared lateral wall."""
    for cut in builder.cuts.values():
        owners = (cut.interval.start_owner, cut.interval.end_owner)
        if all(owners) and owners[0] != owners[1]:
            first, second = (transitions[o - 1] for o in owners)
            if first.axis == second.axis and cut.axis != first.axis:
                yield first, second
    starts = {}
    for t in transitions:
        starts.setdefault((t.axis, t.bounds[1 - t.axis]), []).append(t)
    for t in transitions:
        transverse = 1 - t.axis
        for other in starts.get((t.axis, t.bounds[transverse + 2]), ()):
            yield t, other


def boundary_feedback(builder, transitions, bands, size, constraints):
    """Hard cut contacts either become whole unit dividers or move the front."""
    # Align parallel neighbouring caps together so they cannot chase each
    # other's corners to the exterior.
    aligned = False
    for first, second in neighbouring_caps(builder, transitions):
        if first.side != second.side or first.inner == second.inner:
            continue
        axis = first.axis
        if min(first.bounds[axis + 2], second.bounds[axis + 2]) <= max(first.bounds[axis], second.bounds[axis]):
            continue
        side = first.side
        choose = min if side == 0 else max
        value = choose(bands[first.band].fronts[side], bands[second.band].fronts[side])
        for t in (first, second):
            if bands[t.band].fronts[side] != value:
                bands[t.band].fronts[side] = value
                bands[t.band].splits[side].clear()
                aligned = True
    if aligned:
        return True
    changed = False
    for owner, t in enumerate(transitions, 1):
        points = np.array(sorted(builder.boundary_points[owner]))
        axis, transverse = t.axis, 1 - t.axis
        low, high = t.bounds[axis], t.bounds[axis + 2]
        lateral = points[(points[:, axis] > low) & (points[:, axis] < high)]
        band = bands[t.band]
        if len(lateral):
            value = float(lateral[:, axis].min() if t.side == 0 else lateral[:, axis].max())
            band.fronts[t.side] = value
            band.splits[t.side].clear()
            changed = True
            continue
        outer = t.bounds[axis + (0 if t.side == 0 else 2)]
        levels = points[points[:, axis] == outer, transverse]
        new = set(map(float, levels)) - set(t.splits)
        if new:
            splits = partition(t.bounds[transverse], t.bounds[transverse + 2],
                               set(t.splits) | new, size, constraints, transverse,
                               builder.domain.tolerance, builder.budget.limit)
            if splits is None:
                # An incoming hard ray may originate far away. Jump to its
                # last retained anchor, not one global-size step at a time.
                sign = -1 if t.side == 0 else 1
                events = [outer]
                for level in new:
                    origins = builder.origins.get((axis, float(level)), ())
                    for interval in builder.domain.intervals(axis, level):
                        meets = ((t.side == 1 and interval.start_owner == owner and interval.start == outer)
                                 or (t.side == 0 and interval.end_owner == owner and interval.end == outer))
                        if meets:
                            events.extend(v for v in origins if interval.start <= v <= interval.end
                                          and sign * (v - outer) > 0)
                band.fronts[t.side] = min(events) if t.side == 0 else max(events)
                band.splits[t.side].clear()
            else:
                band.splits[t.side].update(map(float, splits))
            changed = True
    return changed


def repair_exterior_parity(builder, transitions, bands, size, count):
    # Each internal edge is counted twice. With rectangular background cells,
    # the exterior parity is exactly the parity of the number of unit triangles,
    # even when a reservation shares part of the exterior.
    odd, even = [], []
    for owner, t in enumerate(transitions, 1):
        for i, chain in enumerate(t.chains):
            group = odd if len(chain) % 2 else even
            group.append((owner, t, i, chain))
    if len(odd) % 2 == 0:
        return True
    # Prefer removing an existing triangle. An even chain can also absorb a
    # cut when no odd chain sees the exterior.
    for owner, t, i, chain in odd + even:
        # One extra dense node adds two unit cells to an odd chain, one to an even chain.
        extra_cells = 1 + len(chain) % 2
        remaining = builder.budget.limit - count - extra_cells
        if remaining < 1:
            continue
        repaired = builder.exterior_cut(owner, t, chain, size, remaining)
        if repaired is not None:
            t.chains[i] = repaired
            return True
    # This end now continues straight to the exterior. Rebuild the partition
    # before counting again: removing a reservation can change other chains.
    _, t, _, _ = odd[0]
    bands[t.band].active[t.side] = False
    return False


def plan(domain, bands, cores, patterns, size, minimum, policy, budget):
    merges = []
    constraints = Constraints.capture(domain, bands, cores, patterns)
    seen = set()
    # Each retry either merges components, advances a front, adds hard dividers,
    # or terminates an end at the exterior. Exhaustion is explicit, never a mesh fallback.
    search_limit = max(32, 8 * len(bands) + 4 * len(patterns) + 4 * len(domain.loops[0]))
    for iteration in range(search_limit):
        state = tuple((b.axis, b.start, b.end, b.low, b.high, tuple(b.coordinates),
                       tuple(b.fronts), tuple(b.active), tuple(tuple(sorted(s)) for s in b.splits)) for b in bands)
        if state in seen:
            raise ImprintLineError("region planning made no progress", code="planning_limit",
                                   stage="region_planning", reason="no_progress", iterations=iteration)
        seen.add(state)
        previous = len(bands)
        merge_bands(bands, domain, margin=size)
        if len(bands) != previous:
            merges.append(previous - len(bands))
        transitions = []
        for i, band in enumerate(bands):
            for side in (0, 1):
                t = candidate(domain, band, i, side, patterns, size, minimum, constraints, budget)
                if t is not None:
                    transitions.append(t)
        restore_supports(domain, bands, cores, patterns, transitions, policy)
        changed = False
        for t in transitions:
            band = bands[t.band]
            sign = -1 if t.side == 0 else 1
            for j, other in enumerate(bands):
                if j == t.band:
                    continue
                ends = limits(domain, other)
                protected = rectangle(other.axis,
                                      other.fronts[0] if other.active[0] else ends[0],
                                      other.fronts[1] if other.active[1] else ends[1], other.low, other.high)
                if overlap(t.bounds, protected):
                    value = protected[t.axis + (0 if t.side == 0 else 2)]
                    band.fronts[t.side] = min(band.fronts[t.side], value) if sign < 0 else max(band.fronts[t.side], value)
                    band.splits[t.side].clear()
                    changed = True
            for other in transitions:
                if other is t or not overlap(t.bounds, other.bounds):
                    continue
                value = other.bounds[t.axis + (0 if t.side == 0 else 2)]
                band.fronts[t.side] = min(band.fronts[t.side], value) if sign < 0 else max(band.fronts[t.side], value)
                band.splits[t.side].clear()
                changed = True
        if changed:
            continue
        builder = background(domain, transitions, cores, patterns, bands, budget, constraints)
        if boundary_feedback(builder, transitions, bands, size, constraints):
            continue
        try:
            builder.partition(size)
            builder.size_locked(size)
        except BackgroundConflict as conflict:
            for owner in conflict.owners:
                t = transitions[owner - 1]
                bands[t.band].fronts[t.side] = t.bounds[t.axis + (0 if t.side == 0 else 2)]
                bands[t.band].splits[t.side].clear()
            continue
        if boundary_feedback(builder, transitions, bands, size, constraints):
            continue
        for owner, t in enumerate(transitions, 1):
            pts = np.array(sorted(builder.boundary_points[owner]))
            coordinates = pts[pts[:, t.axis] == t.inner, 1 - t.axis]
            for low, high in zip(t.splits, t.splits[1:]):
                chain = np.unique(np.r_[low, coordinates[(coordinates > low) & (coordinates < high)], high])
                t.chains.append(chain)
        count = builder.index.leaf_count + sum((3 * len(chain) - 4) // 2
                                                for t in transitions for chain in t.chains)
        if not repair_exterior_parity(builder, transitions, bands, size, count):
            continue
        budget.check(count, "imprint_assembly")
        units = []
        retry = False
        for t in transitions:
            transverse = 1 - t.axis
            for low, high, chain in zip(t.splits, t.splits[1:], t.chains):
                bounds = list(t.bounds)
                bounds[transverse], bounds[transverse + 2] = low, high
                try:
                    units.append(build_unit(tuple(bounds), t.axis, t.side, chain, policy.absolute,
                                              outline=domain.loops[0]))
                except ImprintLineError as error:
                    if error.code not in ("unit_below_resolution", "exterior_triangle"):
                        raise
                    band = bands[t.band]
                    depth = t.bounds[t.axis + 2] - t.bounds[t.axis]
                    if error.code == "exterior_triangle" or depth >= size * (1 - 32 * np.finfo(float).eps):
                        band.active[t.side] = False
                    else:
                        band.fronts[t.side] = t.bounds[t.axis + (0 if t.side == 0 else 2)]
                    band.splits[t.side].clear()
                    retry = True
                    break
        if retry:
            continue
        rectangles = np.array(builder.index.rectangles(), dtype=float).reshape(-1, 4)
        diagnostics = {"planning_iterations": iteration + 1, "merged_band_count": sum(merges),
                       "band_count": len(bands), "transition_count": sum(len(t.chains) for t in transitions),
                       "boundary_ends": [(b.sources, side) for b in bands for side in (0, 1) if not b.active[side]]}
        return ImprintPlan(bands, transitions, rectangles, units, diagnostics, constraints)
    raise ImprintLineError("outward planning exhausted its event budget", code="planning_limit",
                           stage="region_planning", iterations=search_limit)
