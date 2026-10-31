"""Fill a four-sided patch whose two subdivided edges meet at a corner."""

from __future__ import annotations

from collections import Counter

import numpy as np

from ...model import Mesh2D
from .build_unit_1_1_1_x import (
    _cross,
    _polygon_area_twice,
    _segments_intersect,
    _stage_unit_1_1_1_x,
    _validate_boundary,
    _validate_cyclic_boundary,
    _validate_quads,
)


def _coons_point(nodes: np.ndarray, x: np.ndarray, y: np.ndarray, opposite: int) -> np.ndarray:
    """Interpolate the next diagonal node using the two current boundary arcs."""
    x_points = nodes[x, :2]
    y_points = nodes[y, :2]
    x_lengths = np.linalg.norm(np.diff(x_points[:, :2], axis=0), axis=1)
    y_lengths = np.linalg.norm(np.diff(y_points[:, :2], axis=0), axis=1)
    u = float(x_lengths[0] / x_lengths.sum())
    v = float(1.0 - y_lengths[-1] / y_lengths.sum())
    origin = x_points[0]
    shared = x_points[-1] - origin
    end_y = y_points[-1] - origin
    far_corner = nodes[opposite, :2] - origin
    next_x, previous_y = x_points[1] - origin, y_points[-2] - origin
    top = (1.0 - u) * far_corner + u * end_y
    left = v * far_corner
    bilinear = (
        u * (1.0 - v) * shared
        + u * v * end_y
        + (1.0 - u) * v * far_corner
    )
    return origin + (
        (1.0 - v) * next_x + v * top + (1.0 - u) * left + u * previous_y - bilinear
    )


def _clip_half_plane(
    polygon: list[np.ndarray], normal: np.ndarray, offset: float
) -> list[np.ndarray]:
    """Keep points satisfying normal @ point + offset >= 0."""
    if not polygon:
        return []
    result: list[np.ndarray] = []
    previous = polygon[-1]
    previous_value = float(normal @ previous + offset)
    for current in polygon:
        current_value = float(normal @ current + offset)
        if (previous_value >= 0.0) != (current_value >= 0.0):
            fraction = previous_value / (previous_value - current_value)
            result.append(previous + fraction * (current - previous))
        if current_value >= 0.0:
            result.append(current)
        previous, previous_value = current, current_value
    return result


def _feasible_points(
    nodes: np.ndarray,
    edges: tuple[np.ndarray, ...],
    target: np.ndarray,
) -> list[np.ndarray]:
    """Find deterministic XY candidates that make both peeled quads convex."""
    x, y, _, fourth = edges
    boundary = np.concatenate([edge[:-1] for edge in edges])
    boundary_xy = nodes[boundary, :2]
    origin = nodes[int(x[0]), :2]
    scale = float(np.max(np.ptp(boundary_xy, axis=0)))

    def normalized(index: int) -> np.ndarray:
        return (nodes[int(index), :2] - origin) / scale

    start = normalized(x[0])
    next_x = normalized(x[1])
    opposite = normalized(fourth[0])
    previous_y = normalized(y[-2])
    end_y = normalized(y[-1])

    def turns(point: np.ndarray) -> np.ndarray:
        first = np.stack((start, next_x, point, opposite))
        second = np.stack((opposite, point, previous_y, end_y))
        first_edges = np.roll(first, -1, axis=0) - first
        second_edges = np.roll(second, -1, axis=0) - second
        return np.concatenate(
            (
                _cross(first_edges, np.roll(first_edges, -1, axis=0)),
                _cross(second_edges, np.roll(second_edges, -1, axis=0)),
            )
        )

    zero = turns(np.zeros(2))
    dx = turns(np.array([1.0, 0.0])) - zero
    dy = turns(np.array([0.0, 1.0])) - zero
    minimum = np.min((boundary_xy - origin) / scale, axis=0)
    maximum = np.max((boundary_xy - origin) / scale, axis=0)
    lo_x, lo_y = minimum
    hi_x, hi_y = maximum
    region = [
        np.array([lo_x, lo_y]),
        np.array([hi_x, lo_y]),
        np.array([hi_x, hi_y]),
        np.array([lo_x, hi_y]),
    ]
    margin = 128.0 * np.finfo(float).eps
    for x_coefficient, y_coefficient, constant in zip(dx, dy, zero):
        region = _clip_half_plane(
            region,
            np.array([x_coefficient, y_coefficient]),
            float(constant - margin),
        )
        if len(region) < 3:
            raise ValueError("no position produces valid convex quadrilaterals")

    region_array = np.asarray(region)
    if abs(_polygon_area_twice(region_array)) <= margin:
        raise ValueError("no position produces valid convex quadrilaterals")
    centroid = np.mean(region_array, axis=0)
    normalized_target = (target[:2] - origin) / scale
    projections = []
    for start, end in zip(region_array, np.roll(region_array, -1, axis=0)):
        direction = end - start
        denominator = float(direction @ direction)
        fraction = 0.0 if denominator == 0.0 else float(
            np.clip(((normalized_target - start) @ direction) / denominator, 0.0, 1.0)
        )
        projections.append(start + fraction * direction)
    closest = min(projections, key=lambda point: float(np.sum((point - normalized_target) ** 2)))
    normalized_candidates = (
        normalized_target,
        0.5 * (closest + centroid),
        0.75 * centroid + 0.25 * closest,
        centroid,
    )
    return [origin + scale * candidate for candidate in normalized_candidates]


def _peel_pair(
    nodes: np.ndarray, edges: tuple[np.ndarray, ...]
) -> tuple[np.ndarray, tuple[np.ndarray, ...], np.ndarray]:
    x, y, _, fourth = edges
    target = _coons_point(nodes, x, y, int(fourth[0]))
    next_index = len(nodes)
    quads = np.asarray(
        (
            (int(x[0]), int(x[1]), next_index, int(fourth[0])),
            (int(fourth[0]), next_index, int(y[-2]), int(y[-1])),
        ),
        dtype=np.int64,
    )
    remaining = (
        x[1:],
        y[:-1],
        np.asarray((y[-2], next_index), dtype=np.int64),
        np.asarray((next_index, x[1]), dtype=np.int64),
    )
    for xy in _feasible_points(nodes, edges, target):
        candidate = np.r_[xy, 0.0]
        candidate_nodes = np.vstack((nodes, candidate))
        try:
            _validate_quads(candidate_nodes, quads)
            _validate_cyclic_boundary(
                candidate_nodes,
                remaining,
                require_convex=False,
            )
        except ValueError:
            continue
        return candidate_nodes, remaining, quads
    raise ValueError("no position produces a valid pair of quadrilaterals")


def _validate_patch(
    nodes: np.ndarray, quads: np.ndarray, original_edges: tuple[np.ndarray, ...]
) -> None:
    _validate_quads(nodes, quads)
    boundary = np.concatenate([edge[:-1] for edge in original_edges])
    boundary_area = abs(_polygon_area_twice(nodes[boundary]))
    quad_area = sum(_polygon_area_twice(nodes[quad]) for quad in quads)
    scale = float(np.max(np.ptp(nodes[boundary, :2], axis=0)))
    tolerance = max(
        1e-10 * boundary_area,
        512.0 * np.finfo(float).eps * scale * scale * (len(boundary) + len(quads)),
    )
    if abs(quad_area - boundary_area) > tolerance:
        raise ValueError("quadrilaterals do not cover the boundary")

    expected = {
        frozenset((int(start), int(end)))
        for edge in original_edges
        for start, end in zip(edge[:-1], edge[1:])
    }
    incidence = Counter(
        frozenset((int(quad[index]), int(quad[(index + 1) % 4])))
        for quad in quads
        for index in range(4)
    )
    if {edge for edge, count in incidence.items() if count == 1} != expected or any(
        count != (1 if edge in expected else 2) for edge, count in incidence.items()
    ):
        raise ValueError("quadrilaterals have an invalid edge incidence")

    unique_edges = [tuple(edge) for edge in incidence]
    length_tolerance = 64.0 * np.finfo(float).eps * scale
    area_tolerance = length_tolerance * scale
    endpoints = nodes[np.asarray(unique_edges, dtype=np.int64), :2]
    lower, upper = endpoints.min(axis=1), endpoints.max(axis=1)
    for first, (a, b) in enumerate(unique_edges):
        candidates = np.flatnonzero(np.all(
            np.maximum(lower[first], lower[first + 1:])
            <= np.minimum(upper[first], upper[first + 1:]) + length_tolerance,
            axis=1,
        )) + first + 1
        for second in candidates:
            c, d = unique_edges[second]
            shared = {a, b} & {c, d}
            if shared:
                common = shared.pop()
                other_first = b if a == common else a
                other_second = d if c == common else c
                first_vector = nodes[other_first, :2] - nodes[common, :2]
                second_vector = nodes[other_second, :2] - nodes[common, :2]
                if abs(_cross(first_vector, second_vector)) <= area_tolerance and (
                    first_vector @ second_vector > length_tolerance**2
                ):
                    raise ValueError("quadrilateral edges overlap")
                continue
            if _segments_intersect(
                nodes[a, :2], nodes[b, :2], nodes[c, :2], nodes[d, :2],
                length_tolerance, area_tolerance,
            ):
                raise ValueError("quadrilateral edges cross")


def build_unit_1_1_x_y(
    mesh: Mesh2D,
    edge1: np.ndarray,
    edge2: np.ndarray,
    edge3: np.ndarray,
    edge4: np.ndarray,
) -> Mesh2D:
    """Append Quad4 cells between two adjacent subdivided edges.

    The other two edges have one segment each. Mildly concave boundary chains
    are accepted when the resulting cells are all convex and nonoverlapping.
    Existing elements are preserved; the bounded patch is assumed to be empty.
    """
    if not isinstance(mesh, Mesh2D):
        raise TypeError("mesh must be a Mesh2D instance")
    edges = (edge1, edge2, edge3, edge4)
    boundary = _validate_cyclic_boundary(mesh, edges, require_convex=False)
    if sum(len(edge) for edge in edges) % 2:
        raise ValueError("the total number of edge nodes must be even")
    long_edges = [index for index, edge in enumerate(edges) if len(edge) > 2]
    if len(long_edges) > 2 or (
        len(long_edges) == 2 and not any(
            (index + 1) % 4 in long_edges for index in long_edges
        )
    ):
        raise ValueError("at most two adjacent edges may have more than two nodes")

    if _polygon_area_twice(mesh.nodes[boundary]) < 0.0:
        edges = tuple(edge[::-1] for edge in edges[::-1])
        long_edges = [index for index, edge in enumerate(edges) if len(edge) > 2]
    if len(long_edges) == 2:
        first = next(index for index in long_edges if (index + 1) % 4 in long_edges)
        edges = edges[first:] + edges[:first]

    nodes = mesh.nodes
    staged_quads: list[np.ndarray] = []
    while len(edges[0]) > 2 and len(edges[1]) > 2:
        nodes, edges, pair = _peel_pair(nodes, edges)
        staged_quads.append(pair)

    long_index = _validate_boundary(nodes, edges, require_convex=False)
    tail_nodes, tail_quads = _stage_unit_1_1_1_x(nodes, edges, long_index)
    nodes = np.concatenate((nodes, tail_nodes), axis=0)
    staged_quads.append(tail_quads)
    quads = np.concatenate(staged_quads, axis=0)
    _validate_patch(nodes, quads, (edge1, edge2, edge3, edge4))
    mesh.replace_data(nodes=nodes, elements=np.concatenate((mesh.elements, quads), axis=0))
    return mesh


__all__ = ["build_unit_1_1_x_y"]
