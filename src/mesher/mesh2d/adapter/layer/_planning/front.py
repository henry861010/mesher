"""Plan and validate automatic expansion or a caller-specified target front."""
from __future__ import annotations

import numpy as np

from .._geometry import BoxIndex, area, on_segment, point_location, segment_contacts
from .._types import QuadLayerError, UnmeshableDirectQuadLayout
from .source import _fail, _simple_loop


def _expansion_distances(data):
    """Plan supports and pattern distances before any front can collapse."""
    chains, normals = data["chains"], data["normals"]
    tol, h = data["tol"], data["h"]
    supports, distances, snapped = [], [], []
    for chain, normal, tangent in zip(chains, normals, data["directions"]):
        support = float(np.max(chain @ normal))
        lo, hi = min(chain @ tangent), max(chain @ tangent)
        nearest, pattern_id = float("inf"), None
        for j, pattern in enumerate(data["pattern_records"]):
            if normal[pattern.axis] != 0:
                continue
            interval = sorted((pattern.lo * tangent[pattern.axis],
                               pattern.hi * tangent[pattern.axis]))
            if interval[1] < lo - tol or interval[0] > hi + tol:
                continue
            distance = float(pattern.support * normal[1 - pattern.axis]) - support
            if tol < distance < nearest:
                nearest, pattern_id = distance, j
        distance = nearest if np.isfinite(nearest) and nearest / h <= data["ratio"] + tol else h
        supports.append(support + distance)
        distances.append(distance)
        snapped.append(pattern_id if distance == nearest else None)
    data.update(expansion=distances, snapped=snapped, supports=supports)
    return supports


def _front(data):
    chains, normals = data["chains"], data["normals"]
    supports = _expansion_distances(data)
    front = np.asarray([np.linalg.solve(np.asarray((normals[i - 1], normals[i])),
                                       np.asarray((supports[i - 1], supports[i])))
                        for i in range(len(chains))])
    _validate_front(data, front)


def _validate_front(data, front):
    """Certify either an automatically expanded or explicitly supplied front."""
    tol = data["tol"]
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
    data.update(front=front, outer=outer, inner=inner)


def _normalize_new_outline(new_outline, data):
    """Read an ordered XY loop without moving any of its mandatory points."""
    context = dict(parameter="new_outline", stage="input_validation")
    try:
        points = np.asarray(new_outline, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise QuadLayerError("new_outline must be finite XY points with shape (n, 2)",
                             code="invalid_input", **context) from error
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 4
            or not np.all(np.isfinite(points))):
        _fail("new_outline must be finite XY points with shape (n, 2), n >= 4", **context)
    with np.errstate(over="ignore", invalid="ignore"):
        points = (points - data["origin"]) / data["scale"]
    if (not np.all(np.isfinite(points))
            or np.max(np.abs(points)) > np.sqrt(np.finfo(float).max) / 16):
        _fail("new_outline exceeds the numeric coordinate range",
              "numeric_range_exceeded", **context)
    tol = data["tol"]
    if np.linalg.norm(points[-1] - points[0]) <= tol:
        points = points[:-1]
    if len(points) < 4:
        _fail("new_outline needs at least four distinct corners", **context)
    directions = _new_outline_directions(points, tol)
    turns = np.sum(directions * np.roll(directions, 1, axis=0), axis=1)
    if np.any(turns < 0):
        _fail("new_outline cannot reverse along an adjacent edge", **context)
    try:
        _simple_loop(points, tol, "new_outline")
    except QuadLayerError as error:
        error.context.update(context)
        raise
    if abs(area(points)) <= tol:
        _fail("new_outline has zero area", **context)
    return points


def _new_outline_directions(points, tol):
    vectors = np.roll(points, -1, axis=0) - points
    axes = np.argmax(np.abs(vectors), axis=1)
    rows = np.arange(len(points))
    along = vectors[rows, axes]
    invalid = (np.abs(along) <= tol) | (np.abs(vectors[rows, 1 - axes]) > tol)
    if np.any(invalid):
        _fail("new_outline edges must be nonzero horizontal or vertical segments",
              parameter="new_outline", stage="input_validation",
              edge_index=int(np.flatnonzero(invalid)[0]))
    directions = np.zeros_like(vectors)
    directions[rows, axes] = np.sign(along)
    return directions


def _specified_front(data, points):
    """Match an explicit target to the final, possibly repaired source loop."""
    tol = data["tol"]
    context = dict(parameter="new_outline", stage="expansion")
    if np.sign(area(points)) != np.sign(area(data["source"])):
        points = points[::-1]
    directions = _new_outline_directions(points, tol)
    corners = np.flatnonzero(np.any(directions != np.roll(directions, 1, axis=0), axis=1))
    front = points[corners]
    matches = []
    if len(corners) == len(data["chains"]):
        matches = [i for i in range(len(corners))
                   if np.array_equal(np.roll(directions[corners], -i, axis=0),
                                     data["directions"])]
    if len(matches) != 1:
        _fail("new_outline must have the same cyclic turns as the prepared source",
              "unsupported_new_outline", source_edge_count=len(data["chains"]),
              target_edge_count=len(corners), **context)
    shift = matches[0]
    # Include the wraparound run when the input begins in the middle of an edge.
    for first, last in zip(corners, np.roll(corners, -1)):
        indices = np.arange(first, last if last > first else last + len(points)) % len(points)
        if any(not on_segment(points[i], points[first], points[last], tol) for i in indices):
            _fail("new_outline points must follow their logical straight edge", **context)
    front = np.roll(front, -shift, axis=0).copy()
    anchors = np.roll(points, -int(corners[shift]), axis=0).copy()
    supports = np.sum(front * data["normals"], axis=1)
    expansion = [float(support - np.max(chain @ normal))
                 for support, chain, normal in zip(supports, data["chains"], data["normals"])]
    if any(distance <= tol for distance in expansion):
        _fail("new_outline edges must lie strictly on the expansion side of the prepared source",
              "topology_change", **context)
    try:
        _validate_front(data, front)
    except QuadLayerError as error:
        error.context.update(context)
        raise
    data.update(supports=supports.tolist(), expansion=expansion,
                snapped=[None] * len(front), outline_anchors=anchors)
