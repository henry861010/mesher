"""Shared boundary nodes and front-chain lookup for section assembly."""
from __future__ import annotations

from collections import defaultdict
from math import floor

import numpy as np

from ._types import QuadLayerError, UnmeshableDirectQuadLayout


class _NodePool:
    """Merge planned boundary geometry, without merging unrelated old nodes."""

    def __init__(self, original: np.ndarray, source_ids: np.ndarray, tol: float):
        self.original = original
        self.added: list[np.ndarray] = []
        self.tol = tol
        self.buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        for node_id in source_ids:
            node_id = int(node_id)
            existing = self.find(original[node_id, :2])
            if existing is not None and existing != node_id:
                raise QuadLayerError(
                    "distinct source nodes have indistinguishable coordinates",
                    code="source_node_collision", node_ids=(existing, node_id),
                )
            self.buckets[self.key(original[node_id, :2])].append(node_id)

    def key(self, xy: np.ndarray) -> tuple[int, int]:
        return (floor(float(xy[0]) / self.tol), floor(float(xy[1]) / self.tol))

    def point(self, node_id: int) -> np.ndarray:
        if node_id < len(self.original):
            return self.original[node_id]
        return self.added[node_id - len(self.original)]

    def find(self, xy: np.ndarray) -> int | None:
        x, y = self.key(xy)
        found: list[tuple[float, int]] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for node_id in self.buckets.get((x + dx, y + dy), ()):
                    distance = float(np.linalg.norm(self.point(node_id)[:2] - xy))
                    if distance <= self.tol:
                        found.append((distance, node_id))
        return min(found)[1] if found else None

    def boundary_node(self, xy: np.ndarray) -> int:
        found = self.find(xy)
        if found is not None:
            return found
        node_id = len(self.original) + len(self.added)
        self.added.append(np.array((xy[0], xy[1], 0.0), dtype=float))
        self.buckets[self.key(xy)].append(node_id)
        return node_id

    def interior_nodes(self, points: np.ndarray) -> np.ndarray:
        """Interior points are private to a section, never coordinate-merged."""
        first = len(self.original) + len(self.added)
        self.added.extend(point.copy() for point in points)
        return np.arange(first, first + len(points), dtype=np.int64)

    def array(self) -> np.ndarray:
        if not self.added:
            return self.original.copy()
        return np.vstack((self.original, np.asarray(self.added)))


class _PointIndex:
    """A sorted-axis range index for noding straight front boundaries."""

    def __init__(self, pool: _NodePool, ids: list[int]):
        self.ids = np.asarray(sorted(set(ids)), dtype=np.int64)
        self.xy = np.asarray([pool.point(int(i))[:2] for i in self.ids])
        self.orders = tuple(np.argsort(self.xy[:, axis], kind="stable") for axis in (0, 1))
        self.sorted_values = tuple(self.xy[order, axis] for axis, order in enumerate(self.orders))
        self.tol = pool.tol

    def on_segment(self, start: np.ndarray, end: np.ndarray) -> list[int]:
        minimum = np.minimum(start, end) - self.tol
        maximum = np.maximum(start, end) + self.tol
        ranges = [
            (int(np.searchsorted(values, minimum[axis], side="left")),
             int(np.searchsorted(values, maximum[axis], side="right")))
            for axis, values in enumerate(self.sorted_values)
        ]
        axis = min((0, 1), key=lambda i: ranges[i][1] - ranges[i][0])
        lo, hi = ranges[axis]
        candidates = self.orders[axis][lo:hi]
        points = self.xy[candidates]
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length <= self.tol:
            raise UnmeshableDirectQuadLayout("the front has a zero-length side")
        delta = points - start
        crosses = direction[0] * delta[:, 1] - direction[1] * delta[:, 0]
        projections = delta @ direction / (length * length)
        mask = (
            (np.abs(crosses) <= self.tol * length)
            & (projections >= -self.tol / length)
            & (projections <= 1.0 + self.tol / length)
            & np.all(points >= minimum, axis=1)
            & np.all(points <= maximum, axis=1)
        )
        candidates = candidates[mask]
        order = np.argsort(projections[mask], kind="stable")
        return self.ids[candidates[order]].tolist()
