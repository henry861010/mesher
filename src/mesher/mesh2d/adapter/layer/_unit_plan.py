"""Isolated unit plans and shared coordinates for section builders."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np

from ...model import Mesh2D, _validate_zero_z
from ..unit import build_unit_1_1_1_x, build_unit_1_1_x_y, build_unit_1_x_y_z
from ..unit.build_unit_1_1_x_y import _validate_patch


@dataclass
class UnitPlan:
    nodes: np.ndarray
    quads: np.ndarray
    sides: tuple[np.ndarray, ...]
    method: str
    quality: float
    metadata: dict = field(default_factory=dict)


@dataclass
class SectionPatch:
    nodes: np.ndarray
    quads: np.ndarray
    left_ids: np.ndarray
    right_ids: np.ndarray
    source_ids: np.ndarray
    outer_ids: np.ndarray
    units: list[dict]
    size_error: float
    quality: float
    diagnostics: dict = field(default_factory=dict)


def xyz_chain(values: np.ndarray) -> np.ndarray:
    points = np.asarray(values, dtype=float)
    if points.ndim != 2 or points.shape[1] not in (2, 3) or len(points) < 2:
        raise ValueError("a boundary chain needs at least two XY or XYZ points")
    if points.shape[1] == 2:
        points = np.column_stack((points, np.zeros(len(points))))
    _validate_zero_z(points)
    if not np.all(np.isfinite(points[:, :2])):
        raise ValueError("boundary coordinates must be finite")
    return points.copy()


def quality_score(nodes: np.ndarray, quads: np.ndarray) -> float:
    """Worst max(edge aspect ratio, inverse scaled Jacobian); lower is better."""
    xy = np.asarray(nodes)[np.asarray(quads), :2]
    vectors = np.roll(xy, -1, axis=1) - xy
    scale = np.max(np.abs(vectors), axis=(1, 2))
    if np.any(scale <= 0):
        raise ValueError("a unit contains a degenerate quadrilateral")
    vectors = vectors / scale[:, None, None]
    lengths = np.linalg.norm(vectors, axis=2)
    following = np.roll(vectors, -1, axis=1)
    crosses = vectors[..., 0] * following[..., 1] - vectors[..., 1] * following[..., 0]
    if np.any(lengths <= 0) or np.any(crosses <= 0):
        raise ValueError("a unit contains a nonconvex quadrilateral")
    scaled = crosses / (lengths * np.roll(lengths, -1, axis=1))
    result = float(max(np.max(lengths.max(axis=1) / lengths.min(axis=1)),
                       np.max(1.0 / scaled)))
    if not np.isfinite(result):
        raise ValueError("unit quality is not finite")
    return result


def build_unit_patch(sides: tuple[np.ndarray, ...], tol: float, *,
                     _cache: dict | None = None) -> UnitPlan:
    """Build four coordinate chains without changing any caller-owned arrays.

    The supplied sides are cyclic; either winding is accepted. Boundary IDs
    in the result retain the input side directions, independent of quad winding.
    """
    if len(sides) != 4 or not np.isfinite(tol) or tol <= 0:
        raise ValueError("a unit needs four sides and a positive tolerance")
    chains = tuple(xyz_chain(side) for side in sides)
    for current, following in zip(chains, chains[1:] + chains[:1]):
        if np.linalg.norm(current[-1, :2] - following[0, :2]) > tol:
            raise ValueError("unit coordinate chains must join cyclically")
    boundary = np.vstack([chain[:-1] for chain in chains])
    offsets = np.cumsum([0, *(len(chain) - 1 for chain in chains)])
    ids = tuple(np.r_[np.arange(offsets[i], offsets[i + 1]),
                      offsets[(i + 1) % 4]].astype(np.int64)
                for i in range(4))
    origin = boundary[0, :2].copy()
    scale = float(np.ptp(boundary[:, :2], axis=0).max())
    if scale <= tol:
        raise ValueError("unit boundary has no usable extent")
    normalized = np.column_stack(((boundary[:, :2] - origin) / scale,
                                  np.zeros(len(boundary))))
    cache_key = (tuple(map(len, ids)), normalized.tobytes())
    cached = None if _cache is None else _cache.get(cache_key)
    if cached is not None:
        raw_nodes, raw_quads, method = cached
        nodes = np.column_stack((origin + scale * raw_nodes[:, :2],
                                 np.zeros(len(raw_nodes))))
        nodes[:len(boundary)] = boundary
        quads = raw_quads.copy()
        return UnitPlan(nodes, quads, ids, method, quality_score(nodes, quads),
                        {"boundary_counts": tuple(map(len, ids))})
    mesh = Mesh2D(normalized, np.empty((0, 4), dtype=np.int64))
    diagnostics: dict = {}
    long_sides = [i for i, side in enumerate(ids) if len(side) > 2]
    if len(long_sides) <= 1:
        templates = (build_unit_1_1_1_x, build_unit_1_1_x_y)
    elif len(long_sides) == 2 and any((i + 1) % 4 in long_sides for i in long_sides):
        templates = (build_unit_1_1_x_y,)
    else:
        templates = ()
    for builder in templates:
        attempt = Mesh2D(mesh.nodes, np.empty((0, 4), dtype=np.int64))
        try:
            builder(attempt, *ids)
            # 11xy already performs the complete patch checks. The smaller
            # 111x public builder checks convexity only, so complete that here.
            if builder is build_unit_1_1_1_x:
                _validate_patch(attempt.nodes, attempt.elements, ids)
        except ValueError:
            continue
        mesh = attempt
        diagnostics["method"] = "template"
        break
    else:
        build_unit_1_x_y_z(mesh, *ids, _diagnostics=diagnostics)
    if _cache is not None:
        # Store the actual normalized builder output, before restoring world
        # coordinates. Keys are byte-exact: no rounding or fuzzy geometry
        # equivalence can bypass a unit validation.
        _cache[cache_key] = (mesh.nodes.copy(), np.asarray(mesh.elements, dtype=np.int64).copy(),
                             diagnostics["method"])
    nodes = np.column_stack((origin + scale * mesh.nodes[:, :2],
                             np.zeros(len(mesh.nodes))))
    # Coordinates on all fixed boundaries survive normalization exactly.
    nodes[:len(boundary)] = boundary
    quads = np.asarray(mesh.elements, dtype=np.int64)
    return UnitPlan(nodes, quads, ids, diagnostics["method"], quality_score(nodes, quads),
                    {"boundary_counts": tuple(map(len, ids))})


class CoordinatePool:
    """Merge shared boundary coordinates; keep every unit interior private."""

    def __init__(self, tol: float):
        self.tol = tol
        self.points: list[np.ndarray] = []
        self.buckets: dict[tuple[int, int], list[int]] = {}

    def _key(self, point: np.ndarray) -> tuple[int, int]:
        return tuple(int(np.floor(float(value) / self.tol)) for value in point[:2])

    def intern(self, point: np.ndarray) -> int:
        key = self._key(point)
        found = []
        for delta in product((-1, 0, 1), repeat=2):
            nearby = tuple(a + b for a, b in zip(key, delta))
            for index in self.buckets.get(nearby, ()):
                distance = float(np.linalg.norm(self.points[index][:2] - point[:2]))
                if distance <= self.tol:
                    found.append((distance, index))
        if found:
            return min(found)[1]
        index = len(self.points)
        self.points.append(np.array([point[0], point[1], 0.0], dtype=float))
        self.buckets.setdefault(key, []).append(index)
        return index

    def chain(self, points: np.ndarray) -> np.ndarray:
        return np.asarray([self.intern(point) for point in points], dtype=np.int64)

    def add_unit(self, unit: UnitPlan) -> np.ndarray:
        boundary = set(int(i) for side in unit.sides for i in side)
        mapping = np.empty(len(unit.nodes), dtype=np.int64)
        for index, point in enumerate(unit.nodes):
            if index in boundary:
                mapping[index] = self.intern(point)
            else:
                mapping[index] = len(self.points)
                self.points.append(np.array([point[0], point[1], 0.0], dtype=float))
        return mapping

    def array(self) -> np.ndarray:
        return np.asarray(self.points, dtype=float).reshape(-1, 3)


def merge_unit_plans(units: list[UnitPlan], tol: float
                     ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    pool = CoordinatePool(tol)
    mappings = [pool.add_unit(unit) for unit in units]
    quads = np.vstack([mapping[unit.quads] for unit, mapping in zip(units, mappings)])
    return pool.array(), quads, mappings


def assemble_section(units: list[UnitPlan], source: np.ndarray, outer: np.ndarray,
                     left: np.ndarray, right: np.ndarray, size_error: float,
                     tol: float) -> SectionPatch:
    pool = CoordinatePool(tol)
    source_ids, outer_ids = pool.chain(source), pool.chain(outer)
    left_ids, right_ids = pool.chain(left), pool.chain(right)
    quads, records = [], []
    offset = 0
    for unit in units:
        mapping = pool.add_unit(unit)
        quads.append(mapping[unit.quads])
        records.append({
            "element_start": offset, "element_end": offset + len(unit.quads),
            "boundary_counts": [len(side) for side in unit.sides],
            "method": unit.method, "quality": unit.quality,
            "polygon": unit.nodes[np.concatenate([side[:-1] for side in unit.sides])].tolist(),
        })
        offset += len(unit.quads)
    return SectionPatch(pool.array(), np.vstack(quads), left_ids, right_ids,
                        source_ids, outer_ids, records, float(size_error),
                        max(unit.quality for unit in units))


__all__ = ["UnitPlan", "SectionPatch", "build_unit_patch", "quality_score",
           "CoordinatePool", "merge_unit_plans", "assemble_section", "xyz_chain"]
