"""Monotone directional band merging and transition placement."""
from __future__ import annotations

from math import nextafter

import numpy as np

from ._background import Background
from ._cells import overlapping_cells, rebuild_cells
from ._constraints import Constraints
from ._input import contains_box
from ._reference import ReferenceCoordinates
from ._types import Band, Transition, ImprintPlan, rectangle


def overlap(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def limits(domain, band):
    levels = sorted({band.low, *(float(v) for v in domain.loops[0][:, 1 - band.axis]
                                if band.low < v < band.high)})
    sections = []
    for value in levels:
        interval = next((i for i in domain.section_after(band.axis, value)
                         if i.start <= band.start and band.end <= i.end), None)
        if interval is None:
            return (band.start, band.end)
        sections.append(interval)
    return max(i.start for i in sections), min(i.end for i in sections)


def envelope(band):
    return rectangle(band.axis, band.fronts[0], band.fronts[1],
                     band.low, band.high)


def merge_bands(bands, domain):
    changed = True
    while changed:
        changed = False
        for i, first in enumerate(bands):
            for j in range(i + 1, len(bands)):
                second = bands[j]
                if first.axis != second.axis or not overlap(envelope(first), envelope(second)):
                    continue
                merged = Band(first.axis, min(first.start, second.start), max(first.end, second.end),
                              min(first.low, second.low), max(first.high, second.high),
                              np.unique(np.r_[first.coordinates, second.coordinates]),
                              [min(first.fronts[0], second.fronts[0]), max(first.fronts[1], second.fronts[1])],
                              tuple(sorted(first.sources + second.sources)))
                if not contains_box(domain, merged.bounds):
                    continue  # Concave exteriors can separate otherwise overlapping envelopes.
                bands[i] = merged
                bands.pop(j)
                changed = True
                break
            if changed:
                break
    bands.sort(key=lambda b: (b.axis, b.start, b.end, b.low, b.high, tuple(b.coordinates)))


def candidate(index, band, band_index, side, limit, minimum, original):
    """Find the first outward reference face with one usable row of cells."""
    axis = band.axis
    sign = 1 if side else -1
    entry = axis + 2 * (1 - side)
    exit_face = axis + 2 * side
    requested = band.fronts[side]
    inner = requested
    while sign * (limit - inner) > 0:
        # Probe immediately beyond the front so an exact face selects only its
        # outward owners. The domain limit guarantees coverage of the band.
        probe = nextafter(inner, limit)
        box = rectangle(axis, min(inner, probe), max(inner, probe), band.low, band.high)
        cells = tuple(sorted(overlapping_cells(index, box)))
        # Every cell in the probe must end before another full-width face can
        # begin. This also skips a shallow layer without cutting through it.
        outer = (max if side else min)(b[exit_face] for b in cells)
        # A working level can be displaced to a nearby fixed control. Its
        # original face must also qualify wherever that control is absent.
        ref_inner = original.get(inner, inner)
        ref_outer = original.get(outer, outer)
        depth = min(sign * (ref_outer - ref_inner), sign * (outer - inner))
        if (sign * (ref_inner - requested) >= 0 and depth >= minimum and sign * (limit - outer) >= 0
                and all(b[entry] == inner and b[exit_face] == outer for b in cells)):
            band.fronts[side] = inner
            bounds = rectangle(axis, min(inner, outer), max(inner, outer), band.low, band.high)
            return Transition(band_index, axis, side, bounds, cells)
        inner = outer
    band.active[side] = False
    return None


def plan(domain, bands, cores, patterns, size, minimum, budget):
    geometry = Constraints.capture(domain, [], [], patterns)
    builder = Background(domain, budget, geometry)
    builder.partition(patterns, size)
    constraints = Constraints.capture(domain, bands, cores, patterns) if bands else geometry
    coordinates = ReferenceCoordinates(builder, constraints, geometry, domain.tolerance)
    index = coordinates.working_index(builder.index, budget) if coordinates.active else builder.index
    sources = list(bands)
    merge_bands(bands, domain)
    transitions = []
    for band_index, band in enumerate(bands):
        for side, limit in enumerate(limits(domain, band)):
            end = candidate(index, band, band_index, side, limit, minimum, coordinates.original[band.axis])
            if end is not None:
                transitions.append(end)
    rectangles, units = rebuild_cells(index, domain, sources, bands, transitions, budget, coordinates)
    if coordinates.active:
        rectangles, units = coordinates.restore(rectangles, units)
        for end in transitions:
            end.cells = tuple(map(coordinates.reference_box, end.cells))
            # Report the axial bounds of the chosen reference layer. The
            # transverse interval is the exact input band.
            a = end.axis
            cell = end.cells[0]
            band = bands[end.band]
            end.bounds = rectangle(a, cell[a], cell[a + 2], band.low, band.high)
            band.fronts[end.side] = end.inner
    diagnostics = {
        "merged_band_count": len(sources) - len(bands),
        "band_count": len(bands),
        "transition_count": len(transitions),
        "boundary_ends": [(b.sources, side) for b in bands for side in (0, 1)
                          if not b.active[side]],
    }
    return ImprintPlan(bands, transitions, rectangles, units, diagnostics, constraints)
