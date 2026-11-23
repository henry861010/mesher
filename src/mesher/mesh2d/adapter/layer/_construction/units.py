"""Isolated unit plans and shared coordinates for section builders."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ....model import Mesh2D, _validate_zero_z
from ...unit.build_unit_1_1_1_x import _build_unit_1_1_1_x as build_unit_1_1_1_x
from ...unit.build_unit_1_1_x_y import _build_unit_1_1_x_y as build_unit_1_1_x_y
from ...unit.build_unit_1_x_y_z import _build_unit_1_x_y_z as build_unit_1_x_y_z
from ...unit.build_unit_1_1_1_x import _physical_boundary, _physical_quads, _represented_ring_check
from ...unit.build_unit_1_1_x_y import _validate_patch
from .._registry import _BoundedCoordinates


@dataclass
class UnitPlan:
    nodes: np.ndarray
    quads: np.ndarray
    sides: tuple[np.ndarray, ...]
    method: str
    quality: float
    metadata: dict = field(default_factory=dict)
    fixed_interior: tuple[int, ...] = ()


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
    alignment_error: float = 0.0


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
                     _cache: dict | None = None, validate_unit=None) -> UnitPlan:
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
    effective_tol = tol / scale
    policy = _physical_boundary(boundary, np.arange(len(boundary)), tol)

    def to_world(xy):
        return origin + scale * xy

    check_move = _represented_ring_check(boundary, policy, to_world, lambda xy: (xy - origin) / scale)

    def restore_nodes(raw_nodes):
        nodes = np.column_stack((to_world(raw_nodes[:, :2]), np.zeros(len(raw_nodes))))
        # Every fixed coordinate survives normalization exactly, on either path.
        nodes[:len(boundary)] = boundary
        return nodes

    def restored(raw_nodes, raw_quads, details):
        method, fixed_interior = details
        nodes = restore_nodes(raw_nodes)
        quads = np.asarray(raw_quads, dtype=np.int64).copy()
        if not np.all(np.isfinite(nodes)):
            raise ValueError("restored unit coordinates must be finite")
        _physical_quads(nodes, quads, policy)
        if validate_unit is not None:
            validate_unit(nodes, quads)
        return UnitPlan(nodes, quads, ids, method, quality_score(nodes, quads),
                        {"boundary_counts": tuple(map(len, ids))}, fixed_interior)

    cache_key = (tuple(map(len, ids)), normalized.tobytes(), effective_tol)
    cached = None if _cache is None else _cache.get(cache_key)
    if cached is not None:
        try:
            return restored(*cached)
        except ValueError:
            # Restoring a different frame can expose an unrepresentable edge.
            # Retry the ordinary builders without modifying the stored arrays.
            pass
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
            builder(attempt, *ids, tolerance=effective_tol, _check_move=check_move)
            # 11xy already performs the complete patch checks. The smaller
            # 111x public builder checks convexity only, so complete that here.
            if builder is build_unit_1_1_1_x:
                _validate_patch(attempt.nodes, attempt.elements, ids)
            details = ("template", ())
            result = restored(attempt.nodes, attempt.elements, details)
        except ValueError:
            continue
        mesh = attempt
        break
    else:
        midpoints = []
        check = (None if validate_unit is None else
                 lambda points, cells: validate_unit(restore_nodes(points), cells))
        build_unit_1_x_y_z(mesh, *ids, tolerance=effective_tol, _diagnostics=diagnostics,
                           _fixed_interior=midpoints, _validate_candidate=check, _check_move=check_move)
        details = (diagnostics["method"], tuple(midpoints))
        result = restored(mesh.nodes, mesh.elements, details)
    if _cache is not None:
        # Store normalized output only after its restored representation passes
        # validation. Byte-exact shape AND effective tolerance govern reuse.
        _cache[cache_key] = (mesh.nodes.copy(), np.asarray(mesh.elements, dtype=np.int64).copy(),
                             details)
    return result


class CoordinatePool:
    """Merge shared boundary coordinates; keep every unit interior private."""

    def __init__(self, tol: float):
        self.tol = tol
        self.points: list[np.ndarray] = []
        self.coordinates = _BoundedCoordinates(tol, stage="unit_assembly")

    def intern(self, point: np.ndarray) -> int:
        found = self.coordinates.match(point)
        if found is not None:
            self.coordinates.register(point, found)
            return found
        index = len(self.points)
        self.points.append(np.array([point[0], point[1], 0.0], dtype=float))
        self.coordinates.register(point, index)
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
        boundary = np.concatenate([side[:-1] for side in unit.sides])
        records.append({
            "element_start": offset, "element_end": offset + len(unit.quads),
            "boundary_counts": [len(side) for side in unit.sides],
            "method": unit.method, "quality": unit.quality,
            "polygon": unit.nodes[boundary].tolist(),
            "_polygon_ids": mapping[boundary].tolist(),
        })
        if unit.fixed_interior:
            records[-1]["_fixed_interior"] = [int(mapping[node]) for node in unit.fixed_interior]
        offset += len(unit.quads)
    return SectionPatch(pool.array(), np.vstack(quads), left_ids, right_ids,
                        source_ids, outer_ids, records, float(size_error),
                        max(unit.quality for unit in units))


__all__ = ["UnitPlan", "SectionPatch", "build_unit_patch", "quality_score",
           "CoordinatePool", "merge_unit_plans", "assemble_section", "xyz_chain"]
