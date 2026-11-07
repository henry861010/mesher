"""Validate source mesh and patterns, then normalize their coordinate frame."""
from __future__ import annotations

import numpy as np

from ....model import Mesh2D, _validate_zero_z
from .._geometry import BoxIndex, area, cross, intervals_cover, segment_contacts
from .._old_mesh import old_edge_keys
from .._patterns import (PATTERN_RANGE_LIMIT, AxisPattern, covered_intervals,
                         occupied_intervals, pattern_contacts)
from .._types import QuadLayerError, UnmeshableDirectQuadLayout


def _fail(message, code="invalid_input", **context):
    raise QuadLayerError(message, code=code, **context)


def _input_scalar(value, parameter):
    try:
        return float(value)
    except QuadLayerError:
        raise
    except (TypeError, ValueError, OverflowError) as error:
        raise QuadLayerError(f"{parameter} must be a finite scalar",
                             code="invalid_input", stage="input_validation",
                             parameter=parameter) from error


def _simple_loop(polygon, tol, name):
    following = np.roll(polygon, -1, axis=0)
    lower, upper = np.minimum(polygon, following), np.maximum(polygon, following)
    index = BoxIndex(lower, upper, tol)
    for i, (a, b) in enumerate(zip(polygon, following)):
        for j in index.query(lower[i] - tol, upper[i] + tol):
            if j <= i or j == i + 1 or (i == 0 and j == len(polygon) - 1):
                continue
            if segment_contacts(a, b, polygon[j], following[j], tol):
                _fail(f"{name} intersects or touches itself", "topology_change", edges=(i, int(j)))


def _normalize_patterns(pattern_lines, origin, scale, tol, source):
    def range_failure(reason, index=None):
        raise QuadLayerError("patterns exceed the source-relative numeric range",
                             code="numeric_range_exceeded", stage="input_validation",
                             pattern_index=index, limit=PATTERN_RANGE_LIMIT, reason=reason)

    try:
        patterns = np.asarray(pattern_lines, dtype=float)
    except OverflowError:
        range_failure("coordinate_conversion")
    except (ValueError, TypeError) as exc:
        raise QuadLayerError("patterns must have shape (n, 2, 2)") from exc
    if patterns.size == 0:
        return np.empty((0, 2, 2))
    if patterns.ndim != 3 or patterns.shape[1:] != (2, 2) or not np.all(np.isfinite(patterns)):
        _fail("patterns must be finite XY segments with shape (n, 2, 2)")
    with np.errstate(over="ignore", invalid="ignore"):
        patterns = (patterns - origin) / scale
    # Source coordinates are within one unit of the chosen source-node origin.
    # Bound endpoints before subtraction, lengths, or spatial-index queries.
    lower, upper = source.min(axis=0), source.max(axis=0)
    result = []
    for i, segment in enumerate(patterns):
        if (not np.all(np.isfinite(segment))
                or np.any(np.abs(segment) > PATTERN_RANGE_LIMIT + 1.0)):
            range_failure("endpoint_distance", i)
        distance = np.maximum(lower - segment, segment - upper)
        if np.any(distance > PATTERN_RANGE_LIMIT):
            range_failure("endpoint_distance", i)
        delta = segment[1] - segment[0]
        axis = int(np.argmax(np.abs(delta)))
        if abs(delta[axis]) > PATTERN_RANGE_LIMIT:
            range_failure("length", i)
        if abs(delta[axis]) <= tol or abs(delta[1 - axis]) > tol:
            _fail("patterns must be nonzero horizontal or vertical segments", pattern_index=i)
        segment[:, 1 - axis] = segment[:, 1 - axis].mean()
        if delta[axis] < 0:
            segment = segment[::-1]
        result.append(segment)
    # Geometric classification merges duplicates later, retaining original IDs.
    return np.asarray(result)


def _input_loop(outline, node_count, *, parameter="outline_i"):
    """Read a complete public loop without accepting grouped legacy input."""
    context = dict(stage="input_validation", parameter=parameter)
    try:
        raw = np.asarray(outline)
    except QuadLayerError:
        raise
    except (TypeError, ValueError, OverflowError) as error:
        raise QuadLayerError(f"{parameter} must be a one-dimensional integer node-ID loop",
                             code="invalid_input", **context) from error
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
        _fail(f"{parameter} must be a one-dimensional integer node-ID loop", **context)
    if np.any(raw < 0) or np.any(raw >= node_count):
        _fail(f"{parameter} contains out-of-range node IDs", **context)
    if len(raw) > 1 and raw[-1] == raw[0]:
        raw = raw[:-1]
    if len(raw) < 4:
        _fail(f"{parameter} needs at least four boundary nodes", **context)
    if len(np.unique(raw)) != len(raw):
        _fail(f"{parameter} repeats a node other than its closing point", **context)
    # A stable initial frame makes normalization independent of input start
    # and winding, including when the lowest node is in the middle of an edge.
    return np.roll(raw.astype(np.int64, copy=True), -int(np.argmin(raw)))


def _source_chains(flat, xy, tol):
    """Group a caller-oriented frozen boundary into cyclic cardinal runs."""
    context = dict(stage="input_validation", parameter="outline_i")
    vectors = xy[np.roll(flat, -1)] - xy[flat]
    if np.any(np.linalg.norm(vectors, axis=1) <= tol):
        _fail("outline_i segments must have nonzero XY length", **context)
    axes = np.argmax(np.abs(vectors), axis=1)
    directions = np.zeros_like(vectors)
    rows = np.arange(len(flat))
    directions[rows, axes] = np.sign(vectors[rows, axes])
    corners = np.flatnonzero(np.any(directions != np.roll(directions, 1, axis=0), axis=1))
    if len(corners) < 4:
        _fail("outline_i needs at least four logical edges", **context)
    first = np.lexsort((flat[(corners + 1) % len(flat)], flat[corners]))[0]
    corners = np.roll(corners, -first)
    return [flat[np.arange(start, (end if end > start else end + len(flat)) + 1)
                 % len(flat)].copy()
            for start, end in zip(corners, np.roll(corners, -1))]


def _prepare(mesh, outline_i, element_size, pattern_lines, expand_ratio, axis_tolerance,
             *, expand_size=None, _semantic_directions=None, _frame=None):
    if not isinstance(mesh, Mesh2D):
        _fail("mesh must be a Mesh2D")
    try:
        _validate_zero_z(mesh.nodes)
    except ValueError as exc:
        raise QuadLayerError(str(exc)) from exc
    if not len(mesh.elements):
        _fail("layer extension requires at least one existing Quad4 element", "empty_mesh")
    h = _input_scalar(element_size, "element_size")
    ratio = _input_scalar(expand_ratio, "expand_ratio")
    axis_tol = _input_scalar(axis_tolerance, "axis_tolerance")
    if not np.isfinite(h) or h <= 0:
        _fail("element_size must be finite and positive",
              stage="input_validation", parameter="element_size")
    expand_h = h if expand_size is None else _input_scalar(expand_size, "expand_size")
    if not np.isfinite(expand_h) or expand_h <= 0:
        _fail("expand_size must be finite and positive",
              stage="input_validation", parameter="expand_size")
    if not np.isfinite(ratio) or ratio < 1:
        _fail("expand_ratio must be finite and at least 1",
              stage="input_validation", parameter="expand_ratio")
    if not np.isfinite(axis_tol) or not 0 <= axis_tol < 1:
        _fail("axis_tolerance must be finite and in [0, 1)",
              stage="input_validation", parameter="axis_tolerance")
    if not np.all(np.isfinite(mesh.nodes)):
        _fail("all mesh nodes must be finite")
    if len(mesh.elements) and (np.any(mesh.elements < 0) or np.any(mesh.elements >= len(mesh.nodes))):
        _fail("mesh contains out-of-range node IDs")
    if len(mesh.elements) and np.any(np.diff(np.sort(mesh.elements, axis=1), axis=1) == 0):
        _fail("input elements must be Quad4 with four distinct node IDs", "quad_only_input")
    if _semantic_directions is None:
        flat = _input_loop(outline_i, len(mesh.nodes))
        ids = None
    else:
        # Staged repairs carry directions for possibly oblique segments. They
        # must retain their semantic grouping and original numeric frame.
        ids = []
        for chain in outline_i:
            raw = np.asarray(chain)
            if (raw.ndim != 1 or len(raw) < 2
                    or not np.issubdtype(raw.dtype, np.integer)
                    or np.any(raw < 0) or np.any(raw >= len(mesh.nodes))):
                _fail("prepared source has an invalid node chain", stage="repair_outline")
            ids.append(raw.astype(np.int64, copy=True))
        if len(ids) < 4 or any(a[-1] != b[0] for a, b in zip(ids, ids[1:] + ids[:1])):
            _fail("prepared source chains must form a closed loop", stage="repair_outline")
        flat = np.concatenate([chain[:-1] for chain in ids])
        if len(np.unique(flat)) != len(flat):
            _fail("prepared source repeats a node", stage="repair_outline")
    original = mesh.nodes[flat, :2]
    origin = original[0].copy() if _frame is None else _frame["origin"].copy()
    scale = float(np.max(np.ptp(original, axis=0))) if _frame is None else _frame["scale"]
    if not np.isfinite(scale) or scale <= 0:
        _fail("source boundary has zero or unrepresentable extent")
    xy = (mesh.nodes[:, :2] - origin) / scale
    h /= scale
    expand_h /= scale
    if not np.isfinite(h) or h <= 0 or not np.all(np.isfinite(xy)):
        _fail("geometry and target size have incompatible numeric scales")
    if not np.isfinite(expand_h) or expand_h <= 0:
        _fail("geometry and expand_size have incompatible numeric scales",
              stage="input_validation", parameter="expand_size")
    tol = max(128 * np.finfo(float).eps,
              4 * abs(float(np.spacing(np.max(np.abs(original)))) / scale))
    if _frame is not None:
        tol = max(tol, _frame["tol"])
    if h <= 32 * tol:
        raise UnmeshableDirectQuadLayout("element_size is below geometric resolution")
    if expand_h <= 32 * tol:
        raise UnmeshableDirectQuadLayout("expand_size is below geometric resolution",
                                        stage="input_validation", parameter="expand_size")
    if ids is None:
        ids = _source_chains(flat, xy, tol)
        flat = np.concatenate([chain[:-1] for chain in ids])
    source = xy[flat]
    chains = [xy[chain].copy() for chain in ids]
    if _semantic_directions is not None:
        semantic = np.asarray(_semantic_directions, dtype=float)
        if (semantic.shape != (len(chains), 2)
                or not np.all(np.isin(semantic, (-1., 0., 1.)))
                or not np.all(np.sum(np.abs(semantic), axis=1) == 1)):
            _fail("prepared source directions must be cardinal unit vectors",
                  code="unsupported_slot_repair", stage="repair_outline")
    directions = []
    for i, chain in enumerate(chains):
        delta = chain[-1] - chain[0]
        axis = int(np.argmax(np.abs(delta if _semantic_directions is None else semantic[i])))
        length = abs(delta[axis])
        direction = np.zeros(2)
        direction[axis] = 1 if delta[axis] >= 0 else -1
        if _semantic_directions is not None:
            direction = semantic[i].copy()
        if length <= tol or (_semantic_directions is None
                             and np.ptp(chain[:, 1 - axis]) > axis_tol * length + tol):
            _fail("outline_i edges must be near a global X or Y axis",
                  parameter="outline_i", edge_index=i)
        if np.any(np.diff(chain @ direction) <= tol):
            _fail("nodes on each logical edge must be strictly ordered",
                  parameter="outline_i", edge_index=i)
        directions.append(direction)
    directions = np.asarray(directions)
    if np.any(np.sum(directions * np.roll(directions, -1, axis=0), axis=1) != 0):
        _fail("adjacent logical edges must be perpendicular; combine collinear chains")
    _simple_loop(source, tol, "source")
    if abs(area(source)) <= tol:
        _fail("source boundary has zero area")
    normals = np.column_stack((-directions[:, 1], directions[:, 0]))
    frozen = np.vstack([np.column_stack((edge[:-1], edge[1:])) for edge in ids])
    cells = xy[mesh.elements]
    if len(cells):
        edges = np.roll(cells, -1, axis=1) - cells
        turns = cross(edges, np.roll(edges, -1, axis=1))
        threshold = tol * np.maximum(np.linalg.norm(edges, axis=2),
                                    np.linalg.norm(np.roll(edges, -1, axis=1), axis=2))
        if np.any(~(np.all(turns > threshold, axis=1) | np.all(turns < -threshold, axis=1))):
            _fail("existing Quad4 elements must be nondegenerate and strictly convex")
    patterns = _normalize_patterns(pattern_lines, origin, scale, tol, source)
    pattern_records = tuple(AxisPattern.from_points(points, (i,))
                            for i, points in enumerate(patterns))
    return dict(xy=xy, ids=ids, source=source, chains=chains, normals=normals,
                directions=directions, frozen=frozen, patterns=patterns,
                pattern_records=pattern_records,
                h=h, expand_h=expand_h, ratio=ratio, tol=tol, origin=origin, scale=scale)


def _validate_old_patterns(mesh, data):
    patterns, xy, tol = data["pattern_records"], data["xy"], data["tol"]
    if not len(patterns):
        return
    frozen_points = xy[data["frozen"]]
    index = BoxIndex(frozen_points.min(axis=1), frozen_points.max(axis=1), tol)
    for i, pattern in enumerate(patterns):
        for j in index.query(*pattern.bounds(tol)):
            c, d = frozen_points[j]
            if covered_intervals(pattern, frozen_points[j:j + 1], tol):
                continue
            for point in pattern_contacts(pattern, c, d, tol):
                if min(np.linalg.norm(point - c), np.linalg.norm(point - d)) > tol:
                    _fail("pattern would split a frozen source segment", "fixed_boundary_conflict", pattern_index=i)
    if not len(mesh.elements):
        return
    cells = xy[mesh.elements]
    cell_index = BoxIndex(cells.min(axis=1), cells.max(axis=1), tol)
    keys = old_edge_keys(mesh.elements, data)
    edge_points = xy[keys]
    edge_index = BoxIndex(edge_points.min(axis=1), edge_points.max(axis=1), tol)
    for i, pattern in enumerate(patterns):
        lo, hi = pattern.bounds(tol)
        required = occupied_intervals(pattern, cells[cell_index.query(lo, hi)], tol)
        covering = covered_intervals(pattern, edge_points[edge_index.query(lo, hi)], tol)
        if not intervals_cover(required, covering, tol):
            _fail("pattern cuts through an existing element", "fixed_boundary_conflict", pattern_index=i)
