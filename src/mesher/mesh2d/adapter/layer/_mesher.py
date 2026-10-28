"""Transactional single-layer growth using exclusively direct Quad4 builders."""
from __future__ import annotations

from collections import defaultdict
from time import perf_counter

import numpy as np

from ...model import Mesh2D
from ._geometry import (BoxIndex, area, collinear_interval, cross, intervals_cover,
                        on_segment, point_location, segment_contacts, segment_convex_interval)
from ._partition import partition_layer
from ._registry import fill_blocks
from ._types import QuadLayerError, QuadLayerResult, UnmeshableDirectQuadLayout


def _fail(message, code="invalid_input", **context):
    raise QuadLayerError(message, code=code, **context)


def _edge_incidence(elements):
    incidence = defaultdict(list)
    for i, cell in enumerate(elements):
        for a, b in zip(cell, np.roll(cell, -1)):
            incidence[tuple(sorted((int(a), int(b))))].append((i, int(a), int(b)))
    return incidence


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


def _normalize_patterns(pattern_lines, origin, scale, tol):
    try:
        patterns = np.asarray(pattern_lines, dtype=float)
    except (ValueError, TypeError) as exc:
        raise QuadLayerError("patterns must have shape (n, 2, 2)") from exc
    if patterns.size == 0:
        return np.empty((0, 2, 2))
    if patterns.ndim != 3 or patterns.shape[1:] != (2, 2) or not np.all(np.isfinite(patterns)):
        _fail("patterns must be finite XY segments with shape (n, 2, 2)")
    with np.errstate(over="ignore", invalid="ignore"):
        patterns = (patterns - origin) / scale
    if not np.all(np.isfinite(patterns)):
        _fail("patterns and source have incompatible numeric scales")
    result = []
    for i, segment in enumerate(patterns):
        delta = segment[1] - segment[0]
        axis = int(np.argmax(np.abs(delta)))
        if abs(delta[axis]) <= tol or abs(delta[1 - axis]) > tol:
            _fail("patterns must be nonzero horizontal or vertical segments", pattern_index=i)
        segment[:, 1 - axis] = segment[:, 1 - axis].mean()
        if delta[axis] < 0:
            segment = segment[::-1]
        result.append(segment)
    # Keep overlap endpoints as anchors, while dropping exact duplicates.
    values = np.asarray(result)
    _, first = np.unique(values.reshape(-1, 4), axis=0, return_index=True)
    return values[np.sort(first)]


def _prepare(mesh, outline, element_size, pattern_lines, expand_ratio, axis_tolerance):
    if not isinstance(mesh, Mesh2D):
        _fail("mesh must be a Mesh2D")
    try:
        h, ratio, axis_tol = float(element_size), float(expand_ratio), float(axis_tolerance)
    except (TypeError, ValueError) as exc:
        raise QuadLayerError("size, ratio and tolerance must be finite scalars") from exc
    if not np.isfinite(h) or h <= 0:
        _fail("element_size must be finite and positive")
    if not np.isfinite(ratio) or ratio < 1:
        _fail("expand_ratio must be finite and at least 1")
    if not np.isfinite(axis_tol) or not 0 <= axis_tol < 1:
        _fail("axis_tolerance must be finite and in [0, 1)")
    if not np.all(np.isfinite(mesh.nodes)):
        _fail("all mesh nodes must be finite")
    if len(mesh.elements) and (np.any(mesh.elements < 0) or np.any(mesh.elements >= len(mesh.nodes))):
        _fail("mesh contains out-of-range node IDs")
    if len(mesh.elements) and np.any(np.diff(np.sort(mesh.elements, axis=1), axis=1) == 0):
        _fail("input elements must be Quad4 with four distinct node IDs", "quad_only_input")
    try:
        if len(outline) < 4:
            _fail("outline needs at least four logical edges")
        ids = []
        for chain in outline:
            raw = np.asarray(chain)
            if raw.ndim != 1 or len(raw) < 2 or not np.issubdtype(raw.dtype, np.integer):
                _fail("each outline edge must be a chain of at least two integer node IDs")
            if np.any(raw < 0) or np.any(raw >= len(mesh.nodes)):
                _fail("outline contains out-of-range node IDs")
            ids.append(raw.astype(np.int64, copy=True))
    except TypeError as exc:
        raise QuadLayerError("outline must be a sequence of node-index chains") from exc
    if any(a[-1] != b[0] for a, b in zip(ids, ids[1:] + ids[:1])):
        _fail("each edge end must equal the next edge start, including loop closure")
    flat = np.concatenate([edge[:-1] for edge in ids])
    if len(set(flat.tolist())) != len(flat):
        _fail("outline repeats a node other than at consecutive edge joins")
    original = mesh.nodes[flat, :2]
    origin = original[0].copy()
    scale = float(np.max(np.ptp(original, axis=0)))
    if not np.isfinite(scale) or scale <= 0:
        _fail("source boundary has zero or unrepresentable extent")
    xy = (mesh.nodes[:, :2] - origin) / scale
    h /= scale
    if not np.isfinite(h) or h <= 0 or not np.all(np.isfinite(xy)):
        _fail("geometry and target size have incompatible numeric scales")
    tol = max(128 * np.finfo(float).eps,
              4 * abs(float(np.spacing(np.max(np.abs(original)))) / scale))
    if h <= 32 * tol:
        raise UnmeshableDirectQuadLayout("element_size is below geometric resolution")
    if np.ptp(mesh.nodes[:, 2]) > tol * scale:
        _fail("all nodes must lie in one plane parallel to XY")
    source = xy[flat]
    chains = [xy[chain].copy() for chain in ids]
    directions = []
    for i, chain in enumerate(chains):
        delta = chain[-1] - chain[0]
        axis = int(np.argmax(np.abs(delta)))
        length = abs(delta[axis])
        direction = np.zeros(2)
        direction[axis] = 1 if delta[axis] >= 0 else -1
        if length <= tol or np.ptp(chain[:, 1 - axis]) > axis_tol * length + tol:
            _fail("outline edges must be near a global X or Y axis", edge_index=i)
        if np.any(np.diff(chain @ direction) <= tol):
            _fail("nodes on each logical edge must be strictly ordered", edge_index=i)
        directions.append(direction)
    directions = np.asarray(directions)
    if np.any(np.sum(directions * np.roll(directions, -1, axis=0), axis=1) != 0):
        _fail("adjacent logical edges must be perpendicular; combine collinear chains")
    _simple_loop(source, tol, "source")
    if abs(area(source)) <= tol:
        _fail("source boundary has zero area")
    normals = np.column_stack((-directions[:, 1], directions[:, 0]))
    frozen = np.vstack([np.column_stack((edge[:-1], edge[1:])) for edge in ids])
    incidence = _edge_incidence(mesh.elements)
    cells = xy[mesh.elements]
    if len(cells):
        edges = np.roll(cells, -1, axis=1) - cells
        turns = cross(edges, np.roll(edges, -1, axis=1))
        threshold = tol * np.maximum(np.linalg.norm(edges, axis=2),
                                    np.linalg.norm(np.roll(edges, -1, axis=1), axis=2))
        if np.any(~(np.all(turns > threshold, axis=1) | np.all(turns < -threshold, axis=1))):
            _fail("existing Quad4 elements must be nondegenerate and strictly convex")
        for a, b in frozen:
            owners = incidence.get(tuple(sorted((int(a), int(b)))), ())
            if len(owners) != 1:
                _fail("each source segment must be an existing boundary edge", "fixed_boundary_conflict")
            centroid = cells[owners[0][0]].mean(axis=0)
            if cross(xy[b] - xy[a], centroid - xy[a]) >= -tol * np.linalg.norm(xy[b] - xy[a]):
                _fail("outline direction must place existing mesh on its right", "outline_orientation")
    patterns = _normalize_patterns(pattern_lines, origin, scale, tol)
    return dict(xy=xy, ids=ids, source=source, chains=chains, normals=normals,
                directions=directions, frozen=frozen, patterns=patterns,
                h=h, ratio=ratio, tol=tol, origin=origin, scale=scale,
                old_incidence=incidence)


def _front(data):
    chains, normals = data["chains"], data["normals"]
    tol, h = data["tol"], data["h"]
    supports, distances, snapped = [], [], []
    for chain, normal, tangent in zip(chains, normals, data["directions"]):
        support = float(np.max(chain @ normal))
        lo, hi = min(chain @ tangent), max(chain @ tangent)
        nearest, pattern_id = float("inf"), None
        for j, pattern in enumerate(data["patterns"]):
            if abs(float((pattern[1] - pattern[0]) @ normal)) > tol:
                continue
            interval = sorted(pattern @ tangent)
            if interval[1] < lo - tol or interval[0] > hi + tol:
                continue
            distance = float(pattern[0] @ normal) - support
            if tol < distance < nearest:
                nearest, pattern_id = distance, j
        distance = nearest if np.isfinite(nearest) and nearest / h <= data["ratio"] + tol else h
        supports.append(support + distance)
        distances.append(distance)
        snapped.append(pattern_id if distance == nearest else None)
    front = np.asarray([np.linalg.solve(np.asarray((normals[i - 1], normals[i])),
                                       np.asarray((supports[i - 1], supports[i])))
                        for i in range(len(chains))])
    if (not np.all(np.isfinite(front))
            or np.max(np.abs(front)) > np.sqrt(np.finfo(float).max) / 16):
        raise UnmeshableDirectQuadLayout("expanded front exceeds the numeric coordinate range")
    vectors = np.roll(front, -1, axis=0) - front
    if np.any(np.sum(vectors * data["directions"], axis=1) <= tol):
        _fail("expansion closes a slot, edge, or hole", "topology_change")
    _simple_loop(front, tol, "front")
    source = data["source"]
    if area(source) * area(front) <= 0 or area(source) - area(front) <= tol:
        _fail("expansion collapses or reverses the boundary", "topology_change")
    a, b = source, np.roll(source, -1, axis=0)
    index = BoxIndex(np.minimum(a, b), np.maximum(a, b), tol)
    for c, d in zip(front, np.roll(front, -1, axis=0)):
        for j in index.query(np.minimum(c, d) - tol, np.maximum(c, d) + tol):
            if segment_contacts(a[j], b[j], c, d, tol):
                _fail("front touches or crosses the source boundary", "topology_change")
    outer, inner = (front, source) if area(source) < 0 else (source, front)
    if point_location(inner[0], outer, tol) != 1:
        _fail("source and front do not bound a single annular layer", "topology_change")
    data.update(front=front, outer=outer, inner=inner,
                expansion=distances, snapped=snapped)


def _validate_old_patterns(mesh, data):
    patterns, xy, tol = data["patterns"], data["xy"], data["tol"]
    if not len(patterns):
        return
    frozen_points = xy[data["frozen"]]
    index = BoxIndex(frozen_points.min(axis=1), frozen_points.max(axis=1), tol)
    for i, (a, b) in enumerate(patterns):
        for j in index.query(np.minimum(a, b) - tol, np.maximum(a, b) + tol):
            c, d = frozen_points[j]
            if collinear_interval(a, b, c, d, tol) is not None:
                continue
            for point in segment_contacts(a, b, c, d, tol):
                if min(np.linalg.norm(point - c), np.linalg.norm(point - d)) > tol:
                    _fail("pattern would split a frozen source segment", "fixed_boundary_conflict", pattern_index=i)
    if not len(mesh.elements):
        return
    cells = xy[mesh.elements]
    cell_index = BoxIndex(cells.min(axis=1), cells.max(axis=1), tol)
    keys = np.asarray(list(data["old_incidence"]), dtype=int)
    edge_points = xy[keys]
    edge_index = BoxIndex(edge_points.min(axis=1), edge_points.max(axis=1), tol)
    for i, (a, b) in enumerate(patterns):
        lo, hi = np.minimum(a, b) - tol, np.maximum(a, b) + tol
        required = [interval for j in cell_index.query(lo, hi)
                    if (interval := segment_convex_interval(a, b, cells[j], tol)) is not None]
        covering = [interval for j in edge_index.query(lo, hi)
                    if (interval := collinear_interval(a, b, *edge_points[j], tol)) is not None]
        if not intervals_cover(required, covering, tol / np.linalg.norm(b - a)):
            _fail("pattern cuts through an existing element", "fixed_boundary_conflict", pattern_index=i)


def _anchors(data):
    patterns, tol = data["patterns"], data["tol"]
    candidates = list(patterns.reshape(-1, 2))
    if len(patterns):
        index = BoxIndex(patterns.min(axis=1), patterns.max(axis=1), tol)
        for i, (a, b) in enumerate(patterns):
            for j in index.query(np.minimum(a, b) - tol, np.maximum(a, b) + tol):
                if j > i:
                    candidates.extend(segment_contacts(a, b, *patterns[j], tol))
    frozen = data["xy"][data["frozen"]]
    index = BoxIndex(frozen.min(axis=1), frozen.max(axis=1), tol)
    result = []
    for p in candidates:
        # A covered subsegment of an old edge needs no endpoint insertion.
        interior = any(on_segment(p, a, b, tol)
                       and min(np.linalg.norm(p - a), np.linalg.norm(p - b)) > tol
                       for a, b in frozen[index.query(p - tol, p + tol)])
        if not interior:
            result.append(p)
    return np.asarray(result).reshape(-1, 2)


def _validate_patch(*args):
    # Isolated validator is also an injectable final transaction boundary.
    from ._validation import validate_patch
    return validate_patch(*args)


def extend_quad_layer(mesh: Mesh2D, outline, element_size: float, *, pattern_lines=(),
                      expand_ratio: float = 1.0,
                      axis_tolerance: float = 1e-3) -> QuadLayerResult:
    """Append one conforming Quad4 layer, committing only after validation.

    ``outline`` is a closed chain of near-axis logical edges, CW for an
    exterior boundary and CCW for a hole. ``pattern_lines`` contains finite
    XY segments of shape ``(n, 2, 2)``. Growth is to the left of each directed
    edge. A parallel pattern within ``element_size * expand_ratio`` determines
    that whole edge's expansion; otherwise the expansion is ``element_size``.

    Source nodes, segments, and elements are immutable. Size is a target,
    not an upper bound. Unsupported direct-quad layouts raise
    :class:`UnmeshableDirectQuadLayout`; failures leave ``mesh`` untouched.
    No triangular mesh is constructed, including inside temporary builders.
    """
    start = perf_counter()
    data = _prepare(mesh, outline, element_size, pattern_lines, expand_ratio, axis_tolerance)
    _validate_old_patterns(mesh, data)
    _front(data)
    blocks, partition_diagnostics = partition_layer(
        data["chains"], data["front"], data["patterns"], data["h"], data["tol"])
    local_nodes = mesh.nodes.copy()
    local_nodes[:, :2] = data["xy"]
    anchors = _anchors(data)
    try:
        nodes, quads, new_outline, registry_diagnostics = fill_blocks(
            local_nodes, blocks, data["frozen"], data["h"], data["tol"],
            anchors=anchors, front=data["front"],
            prefer_cycle=partition_diagnostics.get("path") == "structured")
    except UnmeshableDirectQuadLayout:
        if partition_diagnostics.get("path") != "structured":
            raise
        # Staged nodes from the first attempt are discarded in their entirety.
        blocks, partition_diagnostics = partition_layer(
            data["chains"], data["front"], data["patterns"], data["h"], data["tol"],
            force_direct=True)
        nodes, quads, new_outline, registry_diagnostics = fill_blocks(
            local_nodes, blocks, data["frozen"], data["h"], data["tol"],
            anchors=anchors, front=data["front"])
        registry_diagnostics["structured_retry"] = True
    _validate_patch(mesh, nodes, quads, new_outline, data)
    world_nodes = nodes.copy()
    world_nodes[len(mesh.nodes):, :2] = nodes[len(mesh.nodes):, :2] * data["scale"] + data["origin"]
    world_nodes[:len(mesh.nodes)] = mesh.nodes
    restored_xy = (world_nodes[len(mesh.nodes):, :2] - data["origin"]) / data["scale"]
    if np.any(np.abs(restored_xy - nodes[len(mesh.nodes):, :2]) > data["tol"]):
        raise UnmeshableDirectQuadLayout("mapping new nodes back exceeds the geometric tolerance")
    # Verify strict convexity again after restoring the caller's coordinate frame.
    represented = (world_nodes[quads, :2] - data["origin"]) / data["scale"]
    vectors = np.roll(represented, -1, axis=1) - represented
    if not np.all(np.isfinite(world_nodes)) or np.any(cross(vectors, np.roll(vectors, -1, axis=1)) <= 0):
        raise UnmeshableDirectQuadLayout("new quads cannot be represented at the input coordinate scale")
    elements = np.vstack((mesh.elements, quads))
    diagnostics = {**partition_diagnostics, **registry_diagnostics,
                   "added_nodes": len(nodes) - len(mesh.nodes), "added_elements": len(quads),
                   "edge_expansion": [d * data["scale"] for d in data["expansion"]],
                   "snapped_patterns": data["snapped"], "elapsed_seconds": perf_counter() - start}
    # This is the only mutation of the caller's mesh.
    mesh.replace_data(nodes=world_nodes, elements=elements)
    return QuadLayerResult(mesh=mesh, outline=new_outline, diagnostics=diagnostics)
