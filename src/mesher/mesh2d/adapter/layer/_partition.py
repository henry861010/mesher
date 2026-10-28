"""Direct four-sided partitions of a single orthogonal boundary layer.

There is no simplicial mesh, even temporarily.  A small collar preserves every
old boundary segment.  A vertical slab sweep partitions the remainder into
rectangles and merges matching intervals, so unrelated pattern coordinates do
not produce a global Cartesian-product grid.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from ._types import Block, UnmeshableDirectQuadLayout


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _area(points: np.ndarray) -> float:
    shifted = points - points[0]
    return float(np.sum(_cross(shifted, np.roll(shifted, -1, axis=0))) / 2.0)


def _axis_direction(chain: np.ndarray) -> np.ndarray:
    delta = chain[-1] - chain[0]
    axis = int(np.argmax(np.abs(delta)))
    direction = np.zeros(2, dtype=float)
    direction[axis] = 1.0 if delta[axis] > 0.0 else -1.0
    return direction


def _unique(values: list[float], tol: float) -> list[float]:
    result: list[float] = []
    for value in sorted(values):
        if not result or value - result[-1] > tol:
            result.append(float(value))
    return result


def _convex_block(
    vertices: np.ndarray,
    *,
    kind: str,
    source_edge: int | None,
    tol: float,
    metadata: dict | None = None,
) -> Block | None:
    vertices = np.asarray(vertices, dtype=float)
    if _area(vertices) < 0.0:
        vertices = vertices[::-1].copy()
    edges = np.roll(vertices, -1, axis=0) - vertices
    lengths = np.linalg.norm(edges, axis=1)
    turns = _cross(edges, np.roll(edges, -1, axis=0))
    if np.any(lengths <= tol) or np.any(
        turns <= tol * np.maximum(lengths, np.roll(lengths, -1))
    ):
        return None
    return Block(vertices, kind, source_edge, metadata or {})


def _source_groups(
    chain: np.ndarray, mapped: np.ndarray, normal: np.ndarray,
    patterns: np.ndarray, element_size: float, tol: float,
) -> list[tuple[int, int]]:
    """Merge straight source runs while retaining bends and constrained seams.

    Frozen intermediate nodes remain on the resulting block side and are
    inserted by the shared registry.  They need not force matching nodes onto
    the new front: this is what permits an adaptive layer to coarsen.
    """
    breaks = {0, len(chain) - 1}
    start = 0
    direction = chain[1] - chain[0]
    for index in range(2, len(chain)):
        if abs(float(_cross(direction, chain[index] - chain[start]))) > (
            tol * 0.01 * float(np.linalg.norm(direction))
        ):
            breaks.add(index - 1)
            start = index - 1
            direction = chain[index] - chain[start]

    tangent = np.asarray((normal[1], -normal[0]))
    tangential = chain @ tangent
    source_normal, mapped_normal = chain @ normal, mapped @ normal
    for pattern in patterns:
        if abs(float((pattern[1] - pattern[0]) @ tangent)) > tol:
            continue
        low, high = sorted(pattern @ normal)
        matches = (np.abs(tangential - pattern[0] @ tangent) <= tol)
        matches &= np.minimum(high, mapped_normal) - np.maximum(low, source_normal) > tol
        breaks.update(int(index) for index in np.flatnonzero(matches))

    ordered = sorted(breaks)
    groups: list[tuple[int, int]] = []
    for first, last in zip(ordered, ordered[1:]):
        start = first
        for end in range(first + 1, last + 1):
            # Unit filling is deliberately local.  At most 16 frozen segments
            # enter one strip, allowing substantial coarsening without the
            # quadratic geometry checks of a thousand-node unit.  The span
            # limit also avoids very long blocks at ordinary target sizes.
            count_limit = end - start >= 16
            span_limit = float(np.linalg.norm(chain[end] - chain[start])) >= 8.0 * element_size
            if count_limit or span_limit or end == last:
                groups.append((start, end))
                start = end
    return groups


def _collar_blocks(
    chains: list[np.ndarray], front: np.ndarray, patterns: np.ndarray,
    element_size: float, tol: float,
) -> list[Block] | None:
    """Extrude frozen chains; convex corners get their own four-sided cap."""
    directions = np.asarray([_axis_direction(chain) for chain in chains])
    normals = np.column_stack((-directions[:, 1], directions[:, 0]))
    count = len(chains)
    turns = _cross(np.roll(directions, 1, axis=0), directions)
    supports = np.sum(normals * front, axis=1)
    blocks: list[Block] = []

    for index, chain in enumerate(chains):
        normal = normals[index]
        distances = supports[index] - chain @ normal
        if np.any(distances <= tol):
            return None
        mapped = chain + distances[:, None] * normal

        # A left turn consumes part of both incident strips, with a shared
        # miter seam.  Right turns leave the square cap between the strips.
        if turns[index] > 0.0:
            mapped[0] = front[index]
        if turns[(index + 1) % count] > 0.0:
            mapped[-1] = front[(index + 1) % count]

        for segment, end in _source_groups(chain, mapped, normal, patterns, element_size, tol):
            kind = "LINE"
            if segment == 0 and turns[index] > 0.0:
                kind = "INNER"
            if end == len(chain) - 1 and turns[(index + 1) % count] > 0.0:
                kind = "INNER"
            block = _convex_block(
                np.asarray((chain[segment], chain[end], mapped[end], mapped[segment])),
                kind=kind, source_edge=index, tol=tol,
                metadata={
                    "partition": "collar",
                    "source_segment": segment,
                    "source_segment_end": end,
                    "inner_at_start": segment == 0 and bool(turns[index] > 0.0),
                    "inner_at_end": end == len(chain) - 1
                    and bool(turns[(index + 1) % count] > 0.0),
                },
            )
            if block is None:
                return None
            blocks.append(block)

        if turns[index] < 0.0:
            previous = (index - 1) % count
            corner = chain[0]
            outgoing = corner + normal * (supports[index] - corner @ normal)
            incoming = corner + normals[previous] * (
                supports[previous] - corner @ normals[previous]
            )
            block = _convex_block(
                np.asarray((corner, outgoing, front[index], incoming)),
                kind="OUTER", source_edge=index, tol=tol,
                metadata={"partition": "corner"},
            )
            if block is None:
                return None
            blocks.append(block)

    source = np.concatenate([chain[:-1] for chain in chains])
    expected = abs(abs(_area(front)) - abs(_area(source)))
    actual = sum(_area(block.vertices) for block in blocks)
    if abs(actual - expected) > max(64.0 * tol, expected * 1e-9):
        return None
    return blocks


def _pattern_crosses_interior(pattern: np.ndarray, polygon: np.ndarray, tol: float) -> bool:
    """Clip a finite line against a convex block, retaining boundary contacts."""
    delta = pattern[1] - pattern[0]
    edges = np.roll(polygon, -1, axis=0) - polygon
    lengths = np.linalg.norm(edges, axis=1)
    offsets = _cross(edges, pattern[0] - polygon)
    slopes = _cross(edges, delta)
    start, end = 0.0, 1.0
    for offset, slope, length in zip(offsets, slopes, lengths):
        epsilon = tol * length
        if abs(slope) <= epsilon:
            if offset < -epsilon:
                return False
        elif slope > 0.0:
            start = max(start, float(-offset / slope))
        else:
            end = min(end, float(-offset / slope))
        if end < start:
            return False
    if (end - start) * np.linalg.norm(delta) <= tol:
        return False
    midpoint = pattern[0] + (start + end) * 0.5 * delta
    return bool(np.all(_cross(edges, midpoint - polygon) > tol * lengths))


def _patterns_follow_boundaries(blocks: list[Block], patterns: np.ndarray, tol: float) -> bool:
    if not len(patterns):
        return True
    lower = np.asarray([np.min(block.vertices, axis=0) for block in blocks])
    upper = np.asarray([np.max(block.vertices, axis=0) for block in blocks])
    for pattern in patterns:
        lo, hi = np.min(pattern, axis=0), np.max(pattern, axis=0)
        candidates = np.flatnonzero(np.all(upper >= lo - tol, axis=1)
                                    & np.all(lower <= hi + tol, axis=1))
        for index in candidates:
            if _pattern_crosses_interior(pattern, blocks[int(index)].vertices, tol):
                return False
    return True


def _offset_front(chains: list[np.ndarray], distance: float) -> np.ndarray:
    directions = np.asarray([_axis_direction(chain) for chain in chains])
    normals = np.column_stack((-directions[:, 1], directions[:, 0]))
    supports = np.asarray([
        float(np.max(chain @ normal)) + distance
        for chain, normal in zip(chains, normals)
    ])
    result = np.empty((len(chains), 2), dtype=float)
    for index in range(len(chains)):
        previous = (index - 1) % len(chains)
        matrix = np.asarray((normals[previous], normals[index]))
        if abs(float(np.linalg.det(matrix))) < 0.5:
            raise UnmeshableDirectQuadLayout("adjacent logical edges must be orthogonal")
        result[index] = np.linalg.solve(matrix, supports[[previous, index]])
    return result


def _simple_axis_loop(loop: np.ndarray, tol: float) -> bool:
    """Reject non-adjacent contacts of the temporary collar boundary."""
    starts = loop
    ends = np.roll(loop, -1, axis=0)
    lower, upper = np.minimum(starts, ends), np.maximum(starts, ends)
    count = len(loop)
    for index in range(count):
        candidates = np.flatnonzero(np.all(upper >= lower[index] - tol, axis=1)
                                    & np.all(lower <= upper[index] + tol, axis=1))
        for other in candidates:
            if other <= index or other == (index + 1) % count or index == (other + 1) % count:
                continue
            # Two axis-aligned segments with overlapping AABBs touch.
            return False
    return True


def _crossing_intervals(
    first: np.ndarray, second: np.ndarray, x: float, tol: float
) -> list[tuple[float, float]]:
    crossings: list[float] = []
    for loop in (first, second):
        for start, end in zip(loop, np.roll(loop, -1, axis=0)):
            if abs(start[1] - end[1]) <= tol and min(start[0], end[0]) < x < max(start[0], end[0]):
                crossings.append(float(start[1]))
    crossings.sort()
    if len(crossings) % 2:
        raise UnmeshableDirectQuadLayout("an orthogonal sweep encountered an open boundary")
    return [(crossings[i], crossings[i + 1])
            for i in range(0, len(crossings), 2)
            if crossings[i + 1] - crossings[i] > tol]


def _rectangle_core(
    collar: np.ndarray, front: np.ndarray, patterns: np.ndarray,
    element_size: float, tol: float,
) -> list[Block]:
    """Slab decomposition with finite H/V barriers and interval merging.

    A vertical pattern can terminate in the middle of a rectangle's side;
    the shared-interface registry inserts that endpoint on both incident
    blocks.  Horizontal finite endpoints change intervals only in slabs where
    the segment is present.  Thus closed pattern loops remain filled faces.
    """
    all_points = np.vstack((collar, front))
    xmin, xmax = float(np.min(all_points[:, 0])), float(np.max(all_points[:, 0]))
    xvalues = list(all_points[:, 0])
    horizontal: list[np.ndarray] = []
    vertical: list[np.ndarray] = []
    for pattern in patterns:
        if abs(pattern[1, 1] - pattern[0, 1]) <= tol:
            horizontal.append(pattern)
        else:
            vertical.append(pattern)
        xvalues.extend(float(x) for x in pattern[:, 0] if xmin - tol <= x <= xmax + tol)
    xs = _unique(xvalues, tol)

    # Values come from a small coordinate list, not a Cartesian grid.
    yvalues = list(all_points[:, 1]) + [float(p[0, 1]) for p in horizontal]
    canonical_y = np.asarray(_unique(yvalues, tol))

    def ykey(value: float) -> float:
        index = int(np.searchsorted(canonical_y, value))
        candidates = [j for j in (index - 1, index) if 0 <= j < len(canonical_y)]
        return float(canonical_y[min(candidates, key=lambda j: abs(canonical_y[j] - value))])

    active: dict[tuple[float, float], list[float]] = {}
    finished: list[list[float]] = []
    for left, right in zip(xs, xs[1:]):
        if right - left <= tol:
            continue
        middle = (left + right) / 2.0
        intervals = _crossing_intervals(collar, front, middle, tol)
        cuts = [float(p[0, 1]) for p in horizontal
                if min(p[:, 0]) < middle < max(p[:, 0])]
        keys: list[tuple[float, float]] = []
        for low, high in intervals:
            levels = _unique([low, high] + [y for y in cuts if low + tol < y < high - tol], tol)
            keys.extend((ykey(a), ykey(b)) for a, b in zip(levels, levels[1:]))

        next_active: dict[tuple[float, float], list[float]] = {}
        for low, high in keys:
            barrier = any(abs(float(p[0, 0]) - left) <= tol
                          and min(high, float(np.max(p[:, 1])))
                          - max(low, float(np.min(p[:, 1]))) > tol
                          for p in vertical)
            key = (low, high)
            if key in active and not barrier:
                rectangle = active.pop(key)
                rectangle[2] = right
            else:
                rectangle = [left, low, right, high]
            next_active[key] = rectangle
        finished.extend(active.values())
        active = next_active
    finished.extend(active.values())

    blocks: list[Block] = []
    for left, low, right, high in finished:
        # Subdivide only this rectangle, at a target-size multiple.  A long
        # unconstrained strip must not become one unit with thousands of edge
        # nodes; this introduces no global pattern-coordinate product.
        nx = max(1, int(np.ceil((right - left) / (16.0 * element_size))))
        ny = max(1, int(np.ceil((high - low) / (16.0 * element_size))))
        local_x = np.linspace(left, right, nx + 1)
        local_y = np.linspace(low, high, ny + 1)
        for x0, x1 in zip(local_x, local_x[1:]):
            for y0, y1 in zip(local_y, local_y[1:]):
                blocks.append(Block(
                    np.asarray(((x0, y0), (x1, y0), (x1, y1), (x0, y1))),
                    "LINE", None, {"partition": "rectangle"},
                ))
    expected = abs(abs(_area(front)) - abs(_area(collar)))
    actual = sum(_area(block.vertices) for block in blocks)
    if abs(actual - expected) > max(64.0 * tol, expected * 1e-9):
        raise UnmeshableDirectQuadLayout("rectangles do not cover the remaining layer")
    if not _patterns_follow_boundaries(blocks, patterns, tol):
        raise UnmeshableDirectQuadLayout("a finite pattern crosses a rectangular block")
    return blocks


def _classify_inner_sections(blocks: list[Block], patterns: np.ndarray, tol: float) -> None:
    """Classify paired concave-corner flanks in source traversal order.

    The direct partition uses two incident strip blocks instead of the old
    draft's one larger corner patch.  Both blocks carry that corner's label;
    ``corner_sections`` preserves both labels when a short strip participates
    in two distinct concave corners.
    """
    source_edges = [block.source_edge for block in blocks if block.source_edge is not None]
    edge_count = max(source_edges, default=-1) + 1
    corners: dict[int, dict[str, Block]] = {}
    for block in blocks:
        if block.kind != "INNER":
            continue
        index = int(block.source_edge)
        if block.metadata.get("inner_at_start"):
            corners.setdefault(index, {})["right"] = block
        if block.metadata.get("inner_at_end"):
            corners.setdefault((index + 1) % edge_count, {})["left"] = block

    def constrained(a: np.ndarray, b: np.ndarray) -> bool:
        edge = b - a
        length = float(np.linalg.norm(edge))
        if not len(patterns):
            return False
        collinear = np.all(np.abs(_cross(edge, patterns - a)) <= tol * length, axis=1)
        candidates = patterns[collinear]
        if not len(candidates):
            return False
        values = (candidates - a) @ (edge / length)
        overlaps = np.minimum(length, np.max(values, axis=1)) - np.maximum(0.0, np.min(values, axis=1))
        return bool(np.any(overlaps > tol))

    for corner, flanks in sorted(corners.items()):
        left = flanks.get("left")
        right = flanks.get("right")
        has_left = left is not None and constrained(left.vertices[3], left.vertices[0])
        has_right = right is not None and constrained(right.vertices[1], right.vertices[2])
        kind = ("INNER_BOTH" if has_left and has_right else
                "INNER_LEFT" if has_left else "INNER_RIGHT" if has_right else "INNER")
        for block in flanks.values():
            sections = block.metadata.setdefault("corner_sections", [])
            sections.append({"corner": corner, "kind": kind})
            # A block shared by two concave corners uses the first corner in
            # traversal order as its primary label; both are kept in metadata.
            if len(sections) == 1:
                block.kind = kind


def partition_layer(
    source_chains: list[np.ndarray],
    front: np.ndarray,
    patterns: np.ndarray,
    element_size: float,
    tol: float,
    *,
    force_direct: bool = False,
) -> tuple[list[Block], dict]:
    """Return convex CCW four-sided blocks without touching source nodes.

    ``front[i]`` corresponds to ``source_chains[i][0]``.  Arrays are normalized
    XY coordinates; validation of the requested front and existing mesh belongs
    to the transaction orchestrator.  Patterns may extend outside the band.
    """
    chains = [np.asarray(chain, dtype=float) for chain in source_chains]
    front = np.asarray(front, dtype=float)
    patterns = np.asarray(patterns, dtype=float).reshape((-1, 2, 2))

    structured = None if force_direct else _collar_blocks(chains, front, patterns, element_size, tol)
    if structured is not None and _patterns_follow_boundaries(structured, patterns, tol):
        _classify_inner_sections(structured, patterns, tol)
        return structured, {
            "path": "structured",
            "sections": dict(Counter(block.kind for block in structured)),
            "block_count": len(structured),
            "collar_block_count": len(structured),
            "rectangle_count": 0,
            "collar_distance": None,
        }

    directions = np.asarray([_axis_direction(chain) for chain in chains])
    normals = np.column_stack((-directions[:, 1], directions[:, 0]))
    headroom = min(float(front[i] @ normals[i] - np.max(chain @ normals[i]))
                   for i, chain in enumerate(chains))
    shortest = min(float(np.min(np.diff(chain, axis=0) @ direction))
                   for chain, direction in zip(chains, directions))
    distance = min(headroom, shortest, element_size) / 8.0

    for attempt in range(48):
        if distance <= 32.0 * tol:
            break
        collar_front = _offset_front(chains, distance)
        collar = _collar_blocks(chains, collar_front, patterns, element_size, tol)
        if (collar is not None and _simple_axis_loop(collar_front, tol)
                and _patterns_follow_boundaries(collar, patterns, tol)):
            try:
                core = _rectangle_core(collar_front, front, patterns, element_size, tol)
            except UnmeshableDirectQuadLayout:
                distance *= 0.5
                continue
            blocks = collar + core
            _classify_inner_sections(blocks, patterns, tol)
            return blocks, {
                "path": "direct",
                "sections": dict(Counter(block.kind for block in blocks)),
                "block_count": len(blocks),
                "collar_block_count": len(collar),
                "rectangle_count": len(core),
                "collar_distance": distance,
                "collar_attempts": attempt + 1,
            }
        distance *= 0.5

    raise UnmeshableDirectQuadLayout(
        "cannot construct a positive, pattern-preserving source transition",
        minimum_collar_distance=32.0 * tol,
    )
