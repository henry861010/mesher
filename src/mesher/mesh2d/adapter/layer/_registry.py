"""Shared boundary nodes and front-chain lookup for section assembly."""
from __future__ import annotations

from collections import defaultdict
from math import floor

import numpy as np
from ....tolerance import GeometryToleranceError, within_distance

from ._types import QuadLayerError, UnmeshableDirectQuadLayout


class _BoundedCoordinates:
    """Remember every alias so incremental proximity cannot hide a long chain.

    Representatives belong to their original owners and never move. A later
    point may join only one component whose complete diameter remains <= T.
    """

    def __init__(self, tol, *, stage):
        self.tol, self.stage = tol, stage
        self.buckets = defaultdict(list)
        self.groups = defaultdict(list)
        self.exact = {}
        self.fixed = set()

    def key(self, point):
        return tuple(floor(float(value) / self.tol) for value in point[:2])

    def match(self, point):
        point = np.asarray(point)[:2]
        exact = self.exact.get(tuple(point))
        if exact is not None:
            return exact
        x, y = self.key(point)
        owners = set()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for alias, owner in self.buckets.get((x + dx, y + dy), ()):
                    if within_distance(alias, point, self.tol):
                        owners.add(owner)
        if not owners:
            return None
        aliases = np.vstack([self.groups[owner] for owner in sorted(owners)])
        distances = np.linalg.norm(aliases - point, axis=1)
        if len(owners) > 1 or not np.all(within_distance(aliases, point, self.tol)):
            raise GeometryToleranceError(
                "planned coordinates bridge distinct or overwide identity components",
                code="fixed_constraint_conflict" if len(owners & self.fixed) > 1 else "ambiguous_geometry",
                stage=self.stage, source="boundary_coordinates", tolerance=self.tol,
                measured_distance=float(distances.max()), point=point.tolist(),
                owner_ids=sorted(owners))
        return next(iter(owners))

    def register(self, point, owner, *, fixed=False):
        point = np.asarray(point, dtype=float)[:2].copy()
        key = tuple(point)
        if key not in self.exact:
            self.exact[key] = owner
            self.buckets[self.key(point)].append((point, owner))
            self.groups[owner].append(point)
        if fixed:
            self.fixed.add(owner)


class _NodePool:
    """Merge planned boundary geometry, without merging unrelated old nodes."""

    def __init__(self, original: np.ndarray, source_ids: np.ndarray, tol: float):
        self.original = original
        self.added: list[np.ndarray] = []
        self.tol = tol
        self.coordinates = _BoundedCoordinates(tol, stage="layer_assembly")
        for node_id in source_ids:
            node_id = int(node_id)
            existing = self.find(original[node_id, :2])
            if existing is not None and existing != node_id:
                raise QuadLayerError(
                    "distinct source nodes have indistinguishable coordinates",
                    code="source_node_collision", node_ids=(existing, node_id),
                    stage="layer_assembly", source="fixed_source", tolerance=tol,
                    measured_distance=float(np.linalg.norm(original[existing, :2] - original[node_id, :2])),
                )
            self.coordinates.register(original[node_id, :2], node_id, fixed=True)

    def key(self, xy: np.ndarray) -> tuple[int, int]:
        return (floor(float(xy[0]) / self.tol), floor(float(xy[1]) / self.tol))

    def point(self, node_id: int) -> np.ndarray:
        if node_id < len(self.original):
            return self.original[node_id]
        return self.added[node_id - len(self.original)]

    def find(self, xy: np.ndarray) -> int | None:
        return self.coordinates.match(xy)

    def boundary_node(self, xy: np.ndarray) -> int:
        found = self.find(xy)
        if found is not None:
            self.coordinates.register(xy, found)
            return found
        node_id = len(self.original) + len(self.added)
        self.added.append(np.array((xy[0], xy[1], 0.0), dtype=float))
        self.coordinates.register(xy, node_id)
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
        nearest = start + np.clip(projections, 0., 1.)[:, None] * direction
        mask = (
            within_distance(points, nearest, self.tol)
            & np.all(points >= minimum, axis=1)
            & np.all(points <= maximum, axis=1)
        )
        candidates = candidates[mask]
        order = np.argsort(projections[mask], kind="stable")
        return self.ids[candidates[order]].tolist()
