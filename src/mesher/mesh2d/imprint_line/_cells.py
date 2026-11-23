"""Claim complete reference cells and collect their fixed contacts before meshing."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ._cell_mesh import mesh_cell
from ._constraints import _closed_spans, on_outline
from ._types import ImprintLineError, loop, rectangle


def overlapping_cells(index, box):
    """Positive-area BVH query. An edge or corner touch claims no cell."""
    stack = [index.root]
    while stack:
        node = index.nodes[stack.pop()]
        b = node.bounds
        if any(b[a] >= box[a + 2] or box[a] >= b[a + 2] for a in (0, 1)):
            continue
        if node.children is None:
            yield b
        else:
            stack.extend(node.children)


def _face(b, fixed, side):
    t = 1 - fixed
    return fixed, b[fixed + 2 * side], b[t], b[t + 2]


def _faces(b):
    return tuple(_face(b, a, side) for a in (0, 1) for side in (0, 1))


def _clip_levels(levels, lo, hi):
    return levels[levels.searchsorted(lo):levels.searchsorted(hi, side="right")]


def _straight(index, interior, contacts, axis, lo, hi, levels, low, high, coordinates):
    """Retain endpoints and crossings of finite parallel segments on both owners."""
    if coordinates.active:
        coordinates.protect_lines(axis, lo, hi, levels)
    transverse = 1 - axis
    for b in overlapping_cells(index, rectangle(axis, lo, hi, low, high)):
        local = _clip_levels(levels, b[transverse], b[transverse + 2])
        if not len(local):
            continue
        start, end = max(b[axis], lo), min(b[axis + 2], hi)
        cuts = interior[b]
        cuts[axis].update((start, end))
        cuts[transverse].update(local)
        for side in (0, 1):
            if lo <= b[axis + 2 * side] <= hi:
                contacts[_face(b, axis, side)].update(local)
            if b[transverse + 2 * side] in (local[0], local[-1]):
                contacts[_face(b, transverse, side)].update((start, end))


def _straight_band(index, domain, interior, contacts, band, coordinates):
    # Concave outlines can put two disjoint closed intervals on the same line.
    # A straight run reaching the exterior stops at its own component.
    axis = band.axis
    groups = defaultdict(list)
    for level in band.coordinates:
        lo, hi = band.fronts
        if not all(band.active):
            left, right = next((a, b) for a, b in _closed_spans(domain, axis, level)
                               if a <= band.start and band.end <= b)
            lo = lo if band.active[0] else left
            hi = hi if band.active[1] else right
        groups[lo, hi].append(level)
    for (lo, hi), levels in groups.items():
        _straight(index, interior, contacts, axis, lo, hi, np.asarray(levels),
                  band.low, band.high, coordinates)


def _repair_exterior_parity(contacts, domain):
    # The rectangular reference grid already has even exterior parity. Only
    # newly required contacts on actual exterior edges can change it.
    if not contacts:
        return
    faces = list(contacts)
    edges = [((value, lo), (value, hi)) if axis == 0 else ((lo, value), (hi, value))
             for axis, value, lo, hi in faces]
    boundary = on_outline(np.asarray(edges), domain.loops[0])
    touched = [face for face, exterior in zip(faces, boundary) if exterior]
    if sum(len(contacts[face]) for face in touched) % 2 == 0:
        return
    candidates = []
    for face in touched:
        levels = sorted({face[2], *contacts[face], face[3]})
        for lo, hi in zip(levels, levels[1:]):
            mid = lo + (hi - lo) / 2
            if min(mid - lo, hi - mid) > domain.tolerance:
                candidates.append((lo - hi, face, mid))
    if not candidates:
        raise ImprintLineError("exterior has no resolvable parity split",
                               code="invalid_boundary_parity", stage="region_planning")
    _, face, mid = min(candidates)
    contacts[face].add(mid)


def rebuild_cells(index, domain, sources, bands, transitions, budget, coordinates):
    rectangles = np.asarray(index.rectangles()).reshape(-1, 4)
    if not bands:
        return rectangles, []
    interior = defaultdict(lambda: (set(), set()))
    contacts = defaultdict(set)
    for band in bands:
        _straight_band(index, domain, interior, contacts, band, coordinates)
    for source in sources:
        # The original cross sections and their samples remain exact after a merge.
        _straight(index, interior, contacts, 1 - source.axis, source.low, source.high,
                  np.array([source.start, source.end]), source.start, source.end, coordinates)
    claimed = {cell for end in transitions for cell in end.cells}
    for face, points in contacts.items():
        points.difference_update(face[2:])
    contacts = {face: points for face, points in contacts.items() if points}
    _repair_exterior_parity(contacts, domain)

    keep = np.ones(len(rectangles), dtype=bool)
    units = []
    total = len(rectangles)
    for i, row in enumerate(rectangles):
        b = tuple(row)
        levels = interior.get(b, ((), ()))
        sides = [contacts.get(face, ()) for face in _faces(b)]
        internal_cut = any(b[a] < value < b[a + 2] for a in (0, 1) for value in levels[a])
        if internal_cut or any(sides):
            xy, cells = mesh_cell(b, levels, sides, domain, budget, total - 1)
        elif b in claimed:
            xy, cells = loop(b), np.array([[0, 1, 2, 3]])
        else:
            continue
        total += len(cells) - 1
        budget.check(total, "imprint_cell_rebuild")
        keep[i] = False
        units.append((xy, cells))
    return rectangles[keep], units
