"""Fixed support identities and indexed checks of the returned constraints."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._types import ImprintLineError


@dataclass(frozen=True)
class Segment:
    axis: int
    coordinate: float
    start: float
    end: float
    source: str
    index: int


def _closed_spans(domain, axis, coordinate):
    """Include collinear exterior edges, which visibility intervals exclude."""
    spans = [(i.start, i.end) for i in domain.intervals(axis, coordinate)]
    outline = domain.loops[0]
    for a, b in zip(outline, np.roll(outline, -1, axis=0)):
        if a[1 - axis] == b[1 - axis] == coordinate:
            spans.append(tuple(sorted((float(a[axis]), float(b[axis])))))
        elif a[axis] == b[axis] and min(a[1 - axis], b[1 - axis]) <= coordinate <= max(a[1 - axis], b[1 - axis]):
            spans.append((float(a[axis]), float(a[axis])))
    merged = []
    for lo, hi in sorted(spans):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
        else:
            merged.append((lo, hi))
    return merged


@dataclass(frozen=True)
class Constraints:
    supports: tuple[np.ndarray, np.ndarray]
    segments: tuple[Segment, ...]

    @classmethod
    def capture(cls, domain, bands, cores, patterns):
        levels = [set(map(float, domain.loops[0][:, a])) for a in (0, 1)]
        segments = []
        for core in cores:
            for axis in (0, 1):
                levels[axis].update((core[axis], core[axis + 2]))
        for band in bands:
            levels[1 - band.axis].update(map(float, band.coordinates))
            segments.extend(Segment(band.axis, float(v), band.start, band.end,
                                    "line_pairs", band.sources[0]) for v in band.coordinates)
        for index, p in enumerate(patterns):
            for axis in (0, 1):
                levels[axis].update(map(float, p[:, axis]))
            axis = int(p[0, 0] == p[1, 0])
            coordinate = float(p[0, 1 - axis])
            start, end = sorted(map(float, p[:, axis]))
            for lo, hi in _closed_spans(domain, axis, coordinate):
                low, high = max(lo, start), min(hi, end)
                if low <= high:
                    source = patterns.source_indices[index]
                    segments.append(Segment(axis, coordinate, low, high, "pattern_lines", source))
        supports = tuple(np.array(sorted(values)) for values in levels)
        for values in supports:
            values.setflags(write=False)
        return cls(supports, tuple(segments))

    def align_generated(self, axis, values, tolerance):
        """Only generated coordinates may move to a nearby fixed identity."""
        result = np.array(values, dtype=float, copy=True)
        fixed = self.supports[axis]
        if not len(fixed):
            return result
        positions = np.searchsorted(fixed, result)
        left = fixed[np.maximum(0, positions - 1)]
        right = fixed[np.minimum(len(fixed) - 1, positions)]
        closest = np.where(abs(result - left) <= abs(result - right), left, right)
        return np.where(abs(result - closest) <= tolerance, closest, result)

    def validate(self, mesh, edges):
        if not self.segments:
            return
        points = mesh.nodes[edges, :2]
        for axis in (0, 1):
            transverse = 1 - axis
            selected = points[points[:, 0, transverse] == points[:, 1, transverse]]
            starts = np.minimum(selected[:, 0, axis], selected[:, 1, axis])
            ends = np.maximum(selected[:, 0, axis], selected[:, 1, axis])
            coordinates = selected[:, 0, transverse]
            order = np.lexsort((starts, coordinates))
            coordinates, starts, ends = coordinates[order], starts[order], ends[order]
            keys, begin = np.unique(coordinates, return_index=True)
            groups = {}
            for key, lo, hi in zip(keys, begin, np.r_[begin[1:], len(starts)]):
                a, b = starts[lo:hi], ends[lo:hi]
                # Merge contiguous mesh edges into maximal covered intervals.
                gaps = np.r_[True, a[1:] > np.maximum.accumulate(b)[:-1]]
                indices = np.flatnonzero(gaps)
                groups[float(key)] = (a[indices], np.maximum.reduceat(b, indices))
            for s in self.segments:
                if s.axis != axis or s.start == s.end:
                    continue
                group = groups.get(s.coordinate)
                covered = False
                if group is not None:
                    a, b = group
                    index = np.searchsorted(a, s.start, side="right") - 1
                    covered = index >= 0 and b[index] >= s.end
                if not covered:
                    self._fail(s, "fixed segment is not covered by mesh edges")

        # Assembly already sorts nodes lexicographically. Search endpoint pairs
        # in bulk without allocating a Python tuple/set for every mesh node.
        dtype = np.dtype([("x", np.float64), ("y", np.float64)])
        nodes = np.ascontiguousarray(mesh.nodes[:, :2]).view(dtype).ravel()
        anchors = np.array([((s.start, s.coordinate), (s.end, s.coordinate)) if s.axis == 0
                            else ((s.coordinate, s.start), (s.coordinate, s.end))
                            for s in self.segments], dtype=float).reshape(-1, 2)
        query = anchors.view(dtype).ravel()
        ids = np.searchsorted(nodes, query)
        present = (ids < len(nodes)) & (nodes[np.minimum(ids, len(nodes) - 1)] == query)
        if not np.all(present):
            index = int(np.flatnonzero(~present)[0])
            self._fail(self.segments[index // 2], "fixed endpoint is not a mesh node")

    @staticmethod
    def _fail(segment, message):
        raise ImprintLineError(message, code="constraint_not_preserved", stage="validation",
                               source=segment.source, source_index=segment.index,
                               axis=segment.axis, coordinate=segment.coordinate,
                               interval=(segment.start, segment.end))
