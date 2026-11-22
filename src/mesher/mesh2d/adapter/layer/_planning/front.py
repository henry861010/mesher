"""Plan and validate automatic expansion or a caller-specified target front."""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from .._geometry import BoxIndex, area, area_epsilon, point_location, segment_contacts
from .._types import QuadLayerError, UnmeshableDirectQuadLayout
from .source import _fail, _input_loop, _simple_loop


def _within_snap_reach(ordinates, sign, source_support, expand_size, ratio):
    """Compare the guide midline distance to the physical expansion reach."""
    first, last = (float(value) * sign for value in ordinates)
    a, b = first - source_support, last - source_support
    gap = .5 * a + .5 * b
    reach = expand_size * ratio
    if math.isfinite(gap) and math.isfinite(reach):
        error = 8 * math.ulp(max(abs(a), abs(b), reach))
        if abs(gap - reach) > error:
            return gap < reach
    # Roundoff can move either the midpoint or the product across the cutoff.
    # Decide borderline (and overflowing) comparisons from the supplied floats.
    exact_gap = (Fraction(first) + Fraction(last)) / 2 - Fraction(source_support)
    return exact_gap <= Fraction(expand_size) * Fraction(ratio)


def _expansion_distances(data):
    """Plan supports and pattern distances before any front can collapse."""
    chains, normals = data["chains"], data["normals"]
    tol, expand_h = data["tol"], data["expand_h"]
    supports, distances, snapped = [], [], []
    for edge, (chain, normal, tangent) in enumerate(zip(chains, normals, data["directions"])):
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
            axis = 1 - pattern.axis
            distance = float(pattern.support * normal[axis]) - support
            if (tol < distance < nearest and _within_snap_reach(
                    data["world_patterns"][j, :, axis], float(normal[axis]),
                    data["world_source_supports"][edge], data["world_expand_h"], data["ratio"])):
                nearest, pattern_id = distance, j
        distance = nearest if pattern_id is not None else expand_h
        supports.append(support + distance)
        distances.append(distance)
        snapped.append(pattern_id)
    data.update(expansion=distances, snapped=snapped, supports=supports)
    return supports


def _front(data):
    chains, normals = data["chains"], data["normals"]
    supports = _expansion_distances(data)
    front = np.asarray([np.linalg.solve(np.asarray((normals[i - 1], normals[i])),
                                       np.asarray((supports[i - 1], supports[i])))
                        for i in range(len(chains))])
    _validate_front(data, front)
    with np.errstate(over="ignore", invalid="ignore"):
        world_front = front * data["scale"] + data["origin"]
    if not np.all(np.isfinite(world_front)):
        raise UnmeshableDirectQuadLayout("expanded front exceeds the world coordinate range",
                                         stage="world_coordinates")
    data["tolerance"].check_resolution(world_front, stage="world_coordinates", source="layer_front")


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
    if area(source) * area(front) <= 0 or area(source) - area(front) <= data["numeric_tol"]:
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


def _normalize_outline_o(outline_o, data):
    """Resolve a target ID loop while retaining its exact node identities."""
    context = dict(parameter="outline_o", stage="input_validation")
    ids = _input_loop(outline_o, len(data["xy"]), parameter="outline_o")
    points = data["xy"][ids].copy()
    if (not np.all(np.isfinite(points))
            or np.max(np.abs(points)) > np.sqrt(np.finfo(float).max) / 16):
        _fail("outline_o exceeds the numeric coordinate range",
              "numeric_range_exceeded", **context)
    tol = data["tol"]
    directions = _outline_o_directions(points, tol)
    turns = np.sum(directions * np.roll(directions, 1, axis=0), axis=1)
    if np.any(turns < 0):
        _fail("outline_o cannot reverse along an adjacent edge", **context)
    corners = np.flatnonzero(turns == 0)
    for first, last in zip(corners, np.roll(corners, -1)):
        # A long oblique edge can pass a per-segment check after subdivision.
        # Apply the same total transverse-span limit as a reusable source edge.
        stop = last if last > first else last + len(points)
        indices = np.arange(first, stop + 1) % len(points)
        transverse = 1 - int(np.argmax(np.abs(directions[first])))
        if np.ptp(points[indices, transverse]) > tol:
            _fail("outline_o logical edges must be near a global X or Y axis",
                  edge_index=int(first), **context)
    try:
        _simple_loop(points, tol, "outline_o")
    except QuadLayerError as error:
        error.context.update(context)
        raise
    if abs(area(points)) <= area_epsilon(points, data["numeric_tol"]):
        _fail("outline_o has zero area", **context)
    return ids, points


def _outline_o_directions(points, tol):
    vectors = np.roll(points, -1, axis=0) - points
    axes = np.argmax(np.abs(vectors), axis=1)
    rows = np.arange(len(points))
    along = vectors[rows, axes]
    invalid = (np.abs(along) <= tol) | (np.abs(vectors[rows, 1 - axes]) > tol)
    if np.any(invalid):
        _fail("outline_o edges must be nonzero horizontal or vertical segments",
              parameter="outline_o", stage="input_validation",
              edge_index=int(np.flatnonzero(invalid)[0]))
    directions = np.zeros_like(vectors)
    directions[rows, axes] = np.sign(along)
    return directions


def _specified_front(data, target):
    """Match an explicit target to the final, possibly repaired source loop."""
    ids, points = target
    tol = data["tol"]
    context = dict(parameter="outline_o", stage="expansion")
    if np.sign(area(points)) != np.sign(area(data["source"])):
        ids, points = ids[::-1], points[::-1]
    directions = _outline_o_directions(points, tol)
    corners = np.flatnonzero(np.any(directions != np.roll(directions, 1, axis=0), axis=1))
    front = points[corners]
    matches = []
    if len(corners) == len(data["chains"]):
        matches = [i for i in range(len(corners))
                   if np.array_equal(np.roll(directions[corners], -i, axis=0),
                                     data["directions"])]
    if len(matches) != 1:
        _fail("outline_o must have the same cyclic turns as the prepared source",
              "unsupported_new_outline", source_edge_count=len(data["chains"]),
              target_edge_count=len(corners), **context)
    shift = matches[0]
    front = np.roll(front, -shift, axis=0).copy()
    anchors = np.roll(points, -int(corners[shift]), axis=0).copy()
    anchor_ids = np.roll(ids, -int(corners[shift])).copy()
    supports = np.sum(front * data["normals"], axis=1)
    expansion = [float(support - np.max(chain @ normal))
                 for support, chain, normal in zip(supports, data["chains"], data["normals"])]
    if any(distance <= tol for distance in expansion):
        _fail("outline_o edges must lie strictly on the expansion side of the prepared source",
              "topology_change", **context)
    try:
        _validate_front(data, front)
    except QuadLayerError as error:
        error.context.update(context)
        raise
    data.update(supports=supports.tolist(), expansion=expansion,
                snapped=[None] * len(front), outline_anchors=anchors,
                outline_anchor_ids=anchor_ids)
