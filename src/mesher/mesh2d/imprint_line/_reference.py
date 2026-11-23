"""Resolve near-coincident controls locally on an independently sized reference grid."""
from __future__ import annotations

import numpy as np

from ..generators.holed_rectilinear import _RectangleIndex
from ._cell_mesh import invalid_cells


class ReferenceCoordinates:
    """Use one working identity where a generated grid level nearly meets a control.

    A corner can belong to a straight segment and its neighbouring cell at once.
    It moves onto the exact control there. Elsewhere the same reference row keeps
    its original coordinate. Both owners restore a vertex using the same rule.
    """

    def __init__(self, reference, fixed, geometry, tolerance):
        self.tolerance = tolerance
        self.original = [{}, {}]
        self.protected = [{}, {}]
        for axis in (0, 1):
            levels = np.asarray(reference.supports[1 - axis])
            aligned = fixed.align_generated(axis, levels, tolerance)
            changed = levels != aligned
            if not changed.any():
                continue
            # Outline and physical pattern identities are immutable.
            changed &= ~np.isin(levels, geometry.supports[axis])
            # A concave domain can have nearby sizing lines in disjoint
            # sections. Keep other identities when one already matches, and
            # let only the nearest generated level use each control support.
            changed &= ~np.isin(aligned, levels)
            candidates = np.flatnonzero(changed)
            by_distance = np.argsort(abs(aligned[candidates] - levels[candidates]), kind="stable")
            for i in candidates[by_distance]:
                self.original[axis].setdefault(aligned[i], levels[i])
            self.protected[axis] = {value: [] for value in self.original[axis]}

    @property
    def active(self):
        return any(self.original)

    def working_index(self, index, budget):
        rectangles = np.asarray(index.rectangles())
        for axis, original in enumerate(self.original):
            for value, previous in original.items():
                for column in (axis, axis + 2):
                    rows = rectangles[:, column] == previous
                    rectangles[rows, column] = value
        return _RectangleIndex(list(map(tuple, rectangles)), budget)

    def protect_lines(self, axis, start, end, levels):
        # Each straight and each source cross section fixes its transverse
        # coordinate. A generated endpoint follows the local reference face.
        for value, intervals in self.protected[1 - axis].items():
            i = levels.searchsorted(value)
            if i < len(levels) and levels[i] == value:
                intervals.append((start, end))

    def reference_box(self, box):
        return tuple(self.original[i % 2].get(v, v) for i, v in enumerate(box))

    def _restore(self, xy):
        result = xy.copy()
        for axis, original in enumerate(self.original):
            for value, previous in original.items():
                ids = np.flatnonzero(xy[:, axis] == value)
                other = xy[ids, 1 - axis]
                keep = np.zeros(len(ids), dtype=bool)
                for lo, hi in self.protected[axis][value]:
                    keep |= (lo <= other) & (other <= hi)
                result[ids[~keep], axis] = previous
        return result

    def restore(self, rectangles, units):
        # A reference rectangle adjacent to an exact contact may have one
        # displaced corner. Keep it as a single quad with that shared vertex.
        xy = rectangles[:, [0, 1, 2, 1, 2, 3, 0, 3]].reshape(-1, 2)
        points = [xy, *(nodes for nodes, _ in units)]
        offsets = np.cumsum([0, *map(len, points)])
        working, inverse = np.unique(np.concatenate(points), axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        elements = [np.arange(len(xy)).reshape(-1, 4)]
        elements.extend(cells + offset for offset, (_, cells) in zip(offsets[1:], units))
        elements = inverse[np.concatenate(elements)]
        actual = self._restore(working)
        moved = np.any(actual != working, axis=1)
        # A second fixed line can pass just beyond a reference corner. If
        # restoring that corner would collapse an incident edge, keep the
        # locally aligned point for all owners. Every pass pins new vertices.
        while np.any(moved):
            incident = elements[np.any(moved[elements], axis=1)]
            invalid = incident[invalid_cells(actual, incident, self.tolerance)]
            if not len(invalid):
                break
            pinned = np.unique(invalid)
            actual[pinned] = working[pinned]
            moved[pinned] = False
        corners = actual[inverse[:len(xy)]].reshape(-1, 4, 2)
        simple = ((corners[:, 0, 0] == corners[:, 3, 0]) & (corners[:, 1, 0] == corners[:, 2, 0])
                  & (corners[:, 0, 1] == corners[:, 1, 1]) & (corners[:, 2, 1] == corners[:, 3, 1]))
        restored = [(actual[inverse[lo:hi]], cells)
                    for lo, hi, (_, cells) in zip(offsets[1:-1], offsets[2:], units)]
        quad = np.array([[0, 1, 2, 3]])
        restored.extend((points, quad) for points in corners[~simple])
        return corners[simple][:, [0, 2]].reshape(-1, 4), restored
