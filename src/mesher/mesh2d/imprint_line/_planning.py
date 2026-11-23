"""Monotone directional band merging and transition placement."""
from __future__ import annotations

from math import nextafter

from ._background import Background
from ._cells import overlapping_cells, rebuild_cells
from ._constraints import Constraints
from ._bands import limits, merge_bands
from ._reference import ReferenceCoordinates
from ._types import CellTransition, ImprintPlan, rectangle


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
            return CellTransition(band_index, axis, side, bounds, cells)
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
