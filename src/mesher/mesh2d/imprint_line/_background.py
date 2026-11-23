"""Size the fixed background before adding any directional controls."""
from __future__ import annotations

import numpy as np

from ..generators.holed_rectilinear import _Builder
from ...tolerance import subdivision_count
from ._types import ImprintLineError


class ImprintGrid(_Builder):
    def __init__(self, domain, budget, constraints):
        super().__init__(domain, budget)
        self.constraints = constraints

    def support_coordinate(self, axis, coordinate):
        return float(coordinate)

    def _resolvable(self, length, size):
        return length > 0 and length / subdivision_count(length, size) > self.domain.tolerance

    def _cut_cost(self, fixed, coordinate, interval, size):
        """Number of rectangles split by a resolved chord, or zero if infeasible."""
        varying = 1 - fixed
        count = 0
        pending = [self.index.root]
        while pending:
            node = self.index.nodes[pending.pop()]
            b = node.bounds
            if not (b[fixed] < coordinate < b[fixed + 2]):
                continue
            if b[varying + 2] <= interval.start or b[varying] >= interval.end:
                continue
            if node.children is not None:
                pending.extend(node.children)
            elif not all(self._resolvable(length, size) for length in
                         (coordinate - b[fixed], b[fixed + 2] - coordinate)):
                return 0
            else:
                count += 1
        return count


class Background(ImprintGrid):
    def partition(self, patterns, size):
        self.seed(patterns, size)
        for bounds in self.index.rectangles():
            if any(not self._resolvable(bounds[a + 2] - bounds[a], size) for a in (0, 1)):
                raise ImprintLineError("rectangle cannot satisfy both size and resolution",
                                       code="feature_below_resolution", stage="background",
                                       bounds=bounds, tolerance=self.domain.tolerance)
        for fixed in (0, 1):
            while True:
                bounds = self.index.largest(fixed)
                length = bounds[fixed + 2] - bounds[fixed]
                if length <= size * (1 + 32 * np.finfo(float).eps):
                    break
                count = subdivision_count(length, size)
                self.budget.check(self.index.leaf_count + count - 1, "imprint_sizing")
                target = self.index.nodes[self.index.heaps[fixed][0][-1]]
                middle = count // 2
                # Prefer a balanced split, then try each other location once.
                candidates = (k for d in range(count) for k in
                              ((middle,) if d == 0 else (middle - d, middle + d))
                              if 0 < k < count)
                for k in candidates:
                    ideal = bounds[fixed] + length * (k / count)
                    coordinate = float(self.constraints.align_generated(fixed, ideal, self.domain.tolerance))
                    if not bounds[fixed] < coordinate < bounds[fixed + 2]:
                        continue
                    varying = 1 - fixed
                    interval = next((i for i in self.domain.intervals(varying, coordinate)
                                     if i.start <= bounds[varying] and bounds[varying + 2] <= i.end), None)
                    if interval is None or not self._cut_cost(fixed, coordinate, interval, size):
                        continue
                    if not self.add(varying, coordinate, interval, "sizing") or target.children is None:
                        raise ImprintLineError("sizing cut made no progress", code="sizing_no_progress",
                                               stage="background", bounds=bounds, coordinate=coordinate)
                    break
                else:
                    raise ImprintLineError("no resolvable sizing cut", code="locked_sizing_conflict",
                                           stage="background", bounds=bounds)
