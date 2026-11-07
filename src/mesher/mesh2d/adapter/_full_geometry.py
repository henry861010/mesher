"""Boundary forests for complete adaptive Quad4 meshing.

The forest alternates material exteriors (clockwise) and voids
(counterclockwise).  Growing a boundary must preserve this forest; the void
nodes, together with the user exterior, then describe disjoint global grids.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..editor import EditorMesh2D
from ..generators._holed_geometry import OrthogonalDomain
from ..model import Mesh2D, _validate_zero_z
from .layer._geometry import BoxIndex, area, cross, on_segment, point_location, segment_contacts
from .layer._planning.source import _simple_loop
from .layer._types import QuadLayerError


@dataclass(frozen=True)
class BoundaryTopology:
    outer_outline: np.ndarray
    outlines: list[list[int]]
    parents: tuple[int | None, ...]
    signs: tuple[int, ...]
    coordinate_tolerance: float


@dataclass(frozen=True)
class ComplementDomain:
    boundary_index: int | None
    outer_outline: np.ndarray
    hole_boundary_indices: tuple[int, ...]
    holes: list[np.ndarray]


def section_outline_points(plan) -> np.ndarray:
    """Collect the complete planned front, including all section seam anchors."""
    front = np.asarray(plan.front)
    origin = front[0]
    scale = float(np.max(np.ptp(front, axis=0)))
    normalized = (front - origin) / scale
    tolerance = max(128 * np.finfo(float).eps,
                    4 * abs(float(np.spacing(np.max(np.abs(front)))) / scale))

    def canonical(point):
        # Planning restores each seam independently to world coordinates.
        # Correct roundoff only in its fixed coordinate so the optimal grid
        # receives exactly orthogonal geometry with every tangent anchor.
        point = np.asarray(point, dtype=float).copy()
        local = (point - origin) / scale
        distances = np.linalg.norm(normalized - local, axis=1)
        if np.min(distances) <= tolerance:
            return front[int(np.argmin(distances))].copy()
        for edge, (first, last) in enumerate(zip(normalized, np.roll(normalized, -1, axis=0))):
            if on_segment(local, first, last, tolerance):
                fixed = int(np.argmin(np.abs(last - first)))
                point[fixed] = front[edge, fixed]
                return point
        _fail("a section anchor does not lie on the planned front",
              code="invalid_section_boundary", stage="front_planning")

    points = []
    for section in plan.sections:
        for point in section.outer_boundary:
            point = canonical(point)
            if not points or not np.array_equal(point, points[-1]):
                points.append(point)
    if len(points) > 1 and np.array_equal(points[0], points[-1]):
        points.pop()
    if len(points) < 4:
        raise QuadLayerError("planned front has fewer than four points",
                             code="invalid_section_boundary", stage="front_planning")
    return np.asarray(points)


def _fail(message, *, code="invalid_input", **context):
    raise QuadLayerError(message, code=code, **context)


def _points(value, boundary_index):
    try:
        points = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise QuadLayerError("boundary must be finite XY points", code="invalid_input",
                             boundary_index=boundary_index) from error
    if (points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 4
            or not np.all(np.isfinite(points))):
        _fail("boundary must be finite XY points with shape (n, 2), n >= 4",
              boundary_index=boundary_index)
    if np.array_equal(points[0], points[-1]):
        points = points[:-1]
    if len(points) < 4:
        _fail("boundary needs at least four distinct vertices", boundary_index=boundary_index)
    return points


def _distance(a, b, c, d):
    """Euclidean distance between arbitrary planar segments in a safe frame."""
    if segment_contacts(a, b, c, d, 0.0):
        return 0.0

    def point_distance(point, first, last):
        delta = last - first
        fraction = np.clip(float((point - first) @ delta / (delta @ delta)), 0.0, 1.0)
        return float(np.linalg.norm(point - (first + fraction * delta)))

    return min(point_distance(a, c, d), point_distance(b, c, d),
               point_distance(c, a, b), point_distance(d, a, b))


def _forest(outer, loops, tolerance):
    """Classify simple, separated loops without requiring exact axis alignment."""
    all_points = np.vstack((outer, *loops))
    origin = all_points[0]
    with np.errstate(over="ignore", invalid="ignore"):
        scale = float(np.max(np.ptp(all_points, axis=0)))
        normalized = [(points - origin) / scale for points in (outer, *loops)]
    if (not np.isfinite(scale) or scale <= 0
            or any(not np.all(np.isfinite(points)) for points in normalized)):
        _fail("boundary coordinates have an incompatible numeric range")
    separation = tolerance / scale
    for index, (original, points) in enumerate(zip((outer, *loops), normalized)):
        if np.any(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1) == 0):
            _fail("boundary contains a zero-length segment", boundary_index=index - 1)
        local_scale = float(np.max(np.ptp(original, axis=0)))
        local = (original - original[0]) / local_scale
        local_tol = max(128 * np.finfo(float).eps,
                        4 * abs(float(np.spacing(np.max(np.abs(original)))) / local_scale))
        try:
            _simple_loop(local, local_tol, "boundary")
        except QuadLayerError as error:
            error.context["boundary_index"] = index - 1
            raise

    segments, owners = [], []
    for owner, points in enumerate(normalized):
        segments.extend(zip(points, np.roll(points, -1, axis=0)))
        owners.extend([owner] * len(points))
    segments = np.asarray(segments)
    owners = np.asarray(owners)
    lower, upper = segments.min(axis=1), segments.max(axis=1)
    index = BoxIndex(lower, upper, separation)
    for first, (a, b) in enumerate(segments):
        for second in index.query(lower[first], upper[first]):
            if second <= first or owners[first] == owners[second]:
                continue
            if _distance(a, b, *segments[second]) <= separation:
                _fail("boundaries intersect, touch, or lie within coordinate_tolerance",
                      code="topology_change", boundary_indices=(int(owners[first]) - 1,
                                                                 int(owners[second]) - 1))

    outer, *loops = normalized
    signed_areas = [area(points) for points in loops]
    if any(not np.isfinite(value) or value == 0 for value in signed_areas):
        _fail("each boundary must enclose a nonzero finite area")
    signs = tuple(-1 if value < 0 else 1 for value in signed_areas)
    boxes = BoxIndex([points.min(axis=0) for points in loops],
                     [points.max(axis=0) for points in loops])
    parents = []
    for boundary_index, points in enumerate(loops):
        point = points[0]
        if point_location(point, outer, 0.0) != 1:
            _fail("every focus boundary must lie strictly inside outer_outline",
                  code="topology_change", boundary_index=boundary_index)
        containers = [int(other) for other in boxes.query(point, point)
                      if other != boundary_index
                      and point_location(point, loops[other], 0.0) == 1]
        parent = min(containers, key=lambda other: abs(signed_areas[other])) if containers else None
        if (parent is None and signs[boundary_index] != -1
                or parent is not None and signs[parent] == signs[boundary_index]):
            _fail("focus boundaries must alternate material exteriors and interior holes",
                  code="topology_change", boundary_index=boundary_index, parent_boundary=parent)
        parents.append(parent)
    return tuple(parents), signs


def _validate_mesh(mesh):
    if not isinstance(mesh, Mesh2D):
        _fail("mesh must be a Mesh2D", parameter="mesh")
    if not mesh.element_count:
        _fail("full adaptive meshing requires an existing Quad4 mesh", code="empty_mesh")
    if not np.all(np.isfinite(mesh.nodes)):
        _fail("all mesh nodes must be finite", parameter="mesh")
    try:
        _validate_zero_z(mesh.nodes)
    except ValueError as error:
        raise QuadLayerError(str(error), code="invalid_input", parameter="mesh") from error
    elements = mesh.elements
    if (np.any(elements < 0) or np.any(elements >= mesh.node_count)
            or np.any(np.diff(np.sort(elements, axis=1), axis=1) == 0)):
        _fail("input elements must be Quad4 with four distinct valid node IDs",
              code="quad_only_input", parameter="mesh")
    for begin in range(0, len(elements), 65_536):
        points = mesh.nodes[elements[begin:begin + 65_536], :2]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            points = points - points[:, :1]
            scales = np.max(np.abs(points), axis=(1, 2))
            points = points / scales[:, None, None]
        edges = np.roll(points, -1, axis=1) - points
        turns = cross(edges, np.roll(edges, -1, axis=1))
        lengths = np.linalg.norm(edges, axis=2)
        threshold = 128 * np.finfo(float).eps * np.maximum(lengths, np.roll(lengths, -1, axis=1))
        if not np.all(np.isfinite(turns)) or np.any(turns <= threshold):
            _fail("existing Quad4 elements must be nondegenerate, strictly convex, and CCW",
                  parameter="mesh")


def discover_boundaries(mesh, outer_outline, *, coordinate_tolerance=1e-3) -> BoundaryTopology:
    """Validate source data and record its directed boundary containment forest."""
    try:
        _validate_mesh(mesh)
        try:
            domain = OrthogonalDomain(outer_outline, coordinate_tolerance=coordinate_tolerance)
            raw = EditorMesh2D.outlines(mesh)
        except (TypeError, ValueError, IndexError) as error:
            raise QuadLayerError(str(error), code="invalid_input") from error
        if not raw:
            _fail("source mesh has no exposed boundary", code="empty_mesh")
        outlines = [list(map(int, outline)) for outline in raw]
        loops = []
        for boundary_index, outline in enumerate(outlines):
            ids = np.asarray(outline, dtype=np.int64)
            if len(ids) % 2:
                _fail("a frozen odd boundary cannot enclose a complete Quad4 mesh",
                      code="unsupported_boundary_parity", boundary_index=boundary_index,
                      boundary_node_count=len(ids))
            loops.append(mesh.nodes[ids, :2].copy())
        outer = domain.loops[0].copy()
        parents, signs = _forest(outer, loops, domain.coordinate_tolerance)
        return BoundaryTopology(outer, outlines, parents, signs, domain.coordinate_tolerance)
    except QuadLayerError as error:
        error.context.setdefault("stage", "input_validation")
        raise


def _orthogonal_front(points):
    """Use exact support lines for validation of generated, rounded fronts."""
    following = np.roll(points, -1, axis=0)
    if np.all(np.any(points == following, axis=1)):
        return points
    origin = points[0]
    scale = float(np.max(np.ptp(points, axis=0)))
    if not np.isfinite(scale) or scale <= 0:
        _fail("front has no finite positive extent", code="topology_change")
    local = (points - origin) / scale
    vectors = np.roll(local, -1, axis=0) - local
    axes = np.argmax(np.abs(vectors), axis=1)
    rows = np.arange(len(points))
    tolerance = max(128 * np.finfo(float).eps,
                    4 * abs(float(np.spacing(np.max(np.abs(points)))) / scale))
    if np.any(np.abs(vectors[rows, 1 - axes]) > tolerance):
        _fail("generated front must be axis aligned within numerical precision", code="topology_change")
    directions = np.zeros_like(vectors)
    directions[rows, axes] = np.sign(vectors[rows, axes])
    corners = np.flatnonzero(np.any(directions != np.roll(directions, 1, axis=0), axis=1))
    if len(corners) < 4:
        _fail("generated front has fewer than four logical corners", code="topology_change")
    result = points.copy()
    for first, last in zip(corners, np.roll(corners, -1)):
        indices = np.arange(first, (last if last > first else last + len(points)) + 1) % len(points)
        fixed = 1 - axes[first]
        result[indices, fixed] = points[first, fixed]
    if np.any(np.abs((result - points) / scale) > tolerance):
        _fail("generated front deviates from its support lines", code="topology_change")
    return result


def validate_fronts(topology, fronts, *, stage="topology_validation", layer_index=None) -> None:
    """Reject collisions and changed nesting before using a new set of fronts."""
    try:
        if len(fronts) != len(topology.parents):
            _fail("the number of adaptive boundaries changed", code="topology_change")
        loops = [_orthogonal_front(_points(front, index)) for index, front in enumerate(fronts)]
        parents, signs = _forest(topology.outer_outline, loops, topology.coordinate_tolerance)
        if parents != topology.parents or signs != topology.signs:
            _fail("adaptive growth changed the boundary containment forest", code="topology_change")
        # Validate material shells as well as complementary global domains.
        for parent in (None, *range(len(loops))):
            outer = topology.outer_outline if parent is None else loops[parent]
            holes = [loops[index] for index, owner in enumerate(parents) if owner == parent]
            try:
                OrthogonalDomain(outer, holes, topology.coordinate_tolerance)
            except ValueError as error:
                raise QuadLayerError(str(error), code="topology_change", boundary_index=parent) from error
    except QuadLayerError as error:
        error.context.setdefault("stage", stage)
        if layer_index is not None:
            error.context.setdefault("layer_index", layer_index)
        raise


def complement_domains(topology, fronts) -> list[ComplementDomain]:
    """Return the exterior grid and each remaining hole-core grid in stable order."""
    domains = []
    for parent in (None, *(index for index, sign in enumerate(topology.signs) if sign > 0)):
        children = tuple(index for index, owner in enumerate(topology.parents) if owner == parent)
        outer = topology.outer_outline if parent is None else fronts[parent]
        domains.append(ComplementDomain(parent, np.array(outer, dtype=float, copy=True), children,
                                        [np.array(fronts[index], dtype=float, copy=True)
                                         for index in children]))
    return domains
