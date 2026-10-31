"""Fill a four-sided patch with a one-segment side using only convex Quad4 cells."""

from __future__ import annotations

import numpy as np

from ...model import Mesh2D
from .build_unit_1_1_1_x import (
    _cross,
    _polygon_area_twice,
    _validate_cyclic_boundary,
    _validate_quads,
    build_unit_1_1_1_x,
)
from .build_unit_1_1_x_y import (
    _clip_half_plane,
    _validate_patch,
    build_unit_1_1_x_y,
)

_Edges = tuple[np.ndarray, ...]
_Patch = tuple[np.ndarray, np.ndarray]


def _checked_patch(nodes: np.ndarray, quads: np.ndarray, edges: _Edges) -> _Patch:
    if np.any(np.diff(np.sort(quads, axis=1), axis=1) == 0):
        raise ValueError("each generated quadrilateral must have four distinct nodes")
    _validate_patch(nodes, quads, edges)
    return nodes, quads


def _check_ccw_boundary(nodes: np.ndarray, edges: _Edges) -> None:
    boundary = _validate_cyclic_boundary(nodes, edges, require_convex=False)
    if _polygon_area_twice(nodes[boundary]) <= 0.0:
        raise ValueError("a cut must leave two counterclockwise patches")


def _stage_template(nodes: np.ndarray, edges: _Edges) -> _Patch:
    """Try compatible existing templates, isolated from all previous attempts."""
    long = [index for index, edge in enumerate(edges) if len(edge) > 2]
    builders = ()
    if len(long) <= 1:
        # The second builder also accepts mildly concave single-chain patches.
        builders = (build_unit_1_1_1_x, build_unit_1_1_x_y)
    elif len(long) == 2 and any((index + 1) % 4 in long for index in long):
        builders = (build_unit_1_1_x_y,)

    last_error = None
    for builder in builders:
        work = Mesh2D(nodes, np.empty((0, 4), dtype=np.int64))
        try:
            builder(work, *edges)
            return _checked_patch(work.nodes, work.elements, edges)
        except ValueError as error:
            last_error = error
    raise ValueError("no existing template fills this patch") from last_error


def _stage_layers(nodes: np.ndarray, edges: _Edges) -> _Patch:
    """Peel both sides equally, reserving their last segments for the tail."""
    last_error = None
    for index, edge in enumerate(edges):
        if len(edge) != 2:
            continue
        ordered = edges[index:] + edges[:index]
        _, left, opposite, back = ordered
        right = back[::-1]
        count = min(len(left), len(right)) - 2
        if count == 0:
            continue
        quads = np.column_stack(
            (left[:count], left[1:count + 1], right[1:count + 1], right[:count])
        )
        remaining = (
            np.asarray((right[count], left[count]), dtype=np.int64),
            left[count:],
            opposite,
            right[count:][::-1],
        )
        try:
            _validate_quads(nodes, quads)
            _check_ccw_boundary(nodes, remaining)
            tail_nodes, tail_quads = _stage_template(nodes, remaining)
            return _checked_patch(tail_nodes, np.vstack((quads, tail_quads)), edges)
        except ValueError as error:
            last_error = error
    raise ValueError("synchronous layers cannot fill this patch") from last_error


def _stage_local_patch(nodes: np.ndarray, edges: _Edges) -> _Patch:
    """A cap or tail always retains at least one one-segment side."""
    try:
        return _stage_template(nodes, edges)
    except ValueError:
        return _stage_layers(nodes, edges)


def _distance_to_segment_squared(point: np.ndarray, segment: np.ndarray) -> float:
    """Distance in normalized XY; the segment starts at the origin."""
    fraction = float(np.clip((point @ segment) / (segment @ segment), 0.0, 1.0))
    delta = point - fraction * segment
    return float(delta @ delta)


def _stage_sweep(nodes: np.ndarray, edges: _Edges) -> _Patch:
    """Advance a primary node by one segment and match the nearest valid node."""
    frontier, left, opposite, back = edges
    right = back[::-1]
    boundary = np.concatenate([edge[:-1] for edge in edges])
    scale = float(np.max(np.ptp(nodes[boundary, :2], axis=0)))
    origin = nodes[frontier[0], :2]
    seed = (nodes[frontier[1], :2] - origin) / scale
    left_xy = (nodes[left, :2] - origin) / scale
    right_xy = (nodes[right, :2] - origin) / scale
    i = j = 0
    staged_quads: list[np.ndarray] = []

    # Never consume a bottom corner: a zero-length side would leave a triangle.
    while i < len(left) - 2 and j < len(right) - 2:
        left_distance = _distance_to_segment_squared(left_xy[i + 1], seed)
        right_distance = _distance_to_segment_squared(right_xy[j + 1], seed)
        primary_sides = (True, False) if left_distance >= right_distance else (False, True)
        accepted = False
        last_error = None
        for primary_left in primary_sides:
            primary_xy = left_xy[i + 1] if primary_left else right_xy[j + 1]
            other_xy = right_xy if primary_left else left_xy
            other_start = j if primary_left else i
            candidates = sorted(
                range(other_start + 1, len(other_xy) - 1),
                key=lambda candidate: (
                    float(np.sum((other_xy[candidate] - primary_xy) ** 2)), candidate
                ),
            )
            for match in candidates:
                next_i, next_j = (i + 1, match) if primary_left else (match, j + 1)
                connector = np.asarray((left[next_i], right[next_j]), dtype=np.int64)
                candidate_nodes = nodes
                consumed = next_i - i + next_j - j
                # Count segments, including every node on the OLD frontier.
                if (len(frontier) - 1 + consumed + 1) % 2:
                    midpoint_xy = (0.5 * nodes[connector[0], :2]
                                   + 0.5 * nodes[connector[1], :2])
                    midpoint = np.r_[midpoint_xy, 0.0]
                    connector = np.asarray(
                        (connector[0], len(nodes), connector[1]), dtype=np.int64
                    )
                    candidate_nodes = np.vstack((nodes, midpoint))

                cap = (
                    frontier, left[i:next_i + 1], connector, right[j:next_j + 1][::-1]
                )
                remaining = (
                    connector[::-1], left[next_i:], opposite, right[next_j:][::-1]
                )
                try:
                    # Simple, positively oriented subpatches ensure that the
                    # connector stays inside the current boundary.
                    _check_ccw_boundary(candidate_nodes, cap)
                    _check_ccw_boundary(candidate_nodes, remaining)
                    cap_nodes, cap_quads = _stage_local_patch(candidate_nodes, cap)
                except ValueError as error:
                    last_error = error
                    continue

                nodes = cap_nodes
                staged_quads.append(cap_quads)
                i, j = next_i, next_j
                # Reuse the entire connector, including its midpoint index.
                frontier = connector[::-1]
                accepted = True
                break
            if accepted:
                break
        if not accepted:
            raise ValueError("no valid primary-to-opposite cut") from last_error

    remaining = (frontier, left[i:], opposite, right[j:][::-1])
    nodes, tail = _stage_local_patch(nodes, remaining)
    staged_quads.append(tail)
    return _checked_patch(nodes, np.concatenate(staged_quads), edges)


def _stage_inner_ring(nodes: np.ndarray, edges: _Edges) -> _Patch:
    """Use radial convex quads when the visibility kernel has positive area.

    A strict kernel center sees consecutive vertices at angles in (0, pi).
    Moving them onto an interior circle makes each radial strip convex; the
    remaining even-sided polygon is convex and admits a quadrilateral fan.
    """
    boundary = np.concatenate([edge[:-1] for edge in edges])
    xy = nodes[boundary, :2]
    origin = xy[0]
    scale = float(np.max(np.ptp(xy, axis=0)))
    points = (xy - origin) / scale
    vectors = np.roll(points, -1, axis=0) - points
    normals = np.column_stack((-vectors[:, 1], vectors[:, 0]))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    minimum, maximum = points.min(axis=0), points.max(axis=0)
    kernel = [
        minimum.copy(), np.array([maximum[0], minimum[1]]),
        maximum.copy(), np.array([minimum[0], maximum[1]]),
    ]
    for point, normal in zip(points, normals):
        kernel = _clip_half_plane(kernel, normal, -float(normal @ point))
        if len(kernel) < 3:
            raise ValueError("the visibility kernel has no positive area")

    kernel_points = np.asarray(kernel)
    relative = kernel_points - kernel_points[0]
    following = np.roll(relative, -1, axis=0)
    crosses = _cross(relative, following)
    area_twice = float(crosses.sum())
    tolerance = 128.0 * np.finfo(float).eps
    if area_twice <= tolerance:
        raise ValueError("the visibility kernel has no numerically usable area")
    center = kernel_points[0] + np.sum(
        (relative + following) * crosses[:, None], axis=0
    ) / (3.0 * area_twice)
    clearance = float(np.min(np.sum(normals * (center - points), axis=1)))
    if clearance <= tolerance:
        raise ValueError("the visibility kernel has insufficient clearance")

    radius = 0.5 * clearance
    distances = np.linalg.norm(points - center, axis=1)
    directions = (points - center) / distances[:, None]
    inner_xy = origin + scale * (center + radius * directions)

    inner_nodes = np.column_stack((inner_xy, np.zeros(len(inner_xy))))
    inner = np.arange(len(nodes), len(nodes) + len(boundary), dtype=np.int64)
    strips = np.column_stack((boundary, np.roll(boundary, -1), np.roll(inner, -1), inner))
    fan = np.asarray(
        [(inner[0], inner[index], inner[index + 1], inner[index + 2])
         for index in range(1, len(inner) - 2, 2)],
        dtype=np.int64,
    )
    return _checked_patch(np.vstack((nodes, inner_nodes)), np.vstack((strips, fan)), edges)


def _stage_patch(nodes: np.ndarray, edges: _Edges, diagnostics: dict | None = None) -> _Patch:
    def selected(patch: _Patch, method: str) -> _Patch:
        if diagnostics is not None:
            diagnostics["method"] = method
        return patch

    try:
        return selected(_stage_template(nodes, edges), "template")
    except ValueError:
        pass
    for index, edge in enumerate(edges):
        if len(edge) != 2:
            continue
        try:
            return selected(_stage_sweep(nodes, edges[index:] + edges[:index]), "sweep")
        except ValueError:
            pass
    try:
        return selected(_stage_layers(nodes, edges), "layers")
    except ValueError:
        pass
    try:
        return selected(_stage_inner_ring(nodes, edges), "inner_ring")
    except ValueError as error:
        raise ValueError("the available strategies could not construct a valid Quad4 patch") from error


def build_unit_1_x_y_z(
    mesh: Mesh2D,
    edge1: np.ndarray,
    edge2: np.ndarray,
    edge3: np.ndarray,
    edge4: np.ndarray,
    *,
    _diagnostics: dict | None = None,
) -> Mesh2D:
    """Append convex Quad4 cells inside four cyclic boundary chains.

    Edges are one-dimensional integer NumPy arrays of existing node indices.
    At least one edge has two nodes and their total node count must be even.
    Both windings and mildly concave chains are accepted. Geometry is measured
    in XY; every new node has Z=0. The x/y/z in this function name denote
    side subdivision counts, not spatial coordinate axes.

    Existing templates are preferred. Otherwise, of the two next side nodes,
    the one farther from the initial short segment is matched to the nearest
    valid opposite node. Parity midpoints are shared with subsequent layers.
    Synchronous layers and a visibility-kernel inner ring provide fallbacks
    when these cuts cannot be meshed.

    Every successful result preserves the original boundary segments and adds
    only nonoverlapping, strictly convex, counterclockwise quadrilaterals.
    Some simple polygons cannot be handled by these strategies; they raise
    ValueError. All attempts are staged, so any failure leaves the original
    mesh unchanged. The patch interior is assumed to contain no elements.
    """
    if not isinstance(mesh, Mesh2D):
        raise TypeError("mesh must be a Mesh2D instance")
    edges = (edge1, edge2, edge3, edge4)
    boundary = _validate_cyclic_boundary(mesh, edges, require_convex=False)
    if not any(len(edge) == 2 for edge in edges):
        raise ValueError("at least one edge must contain exactly two nodes")
    if sum(len(edge) for edge in edges) % 2:
        raise ValueError("the total number of edge nodes must be even")
    if _polygon_area_twice(mesh.nodes[boundary]) < 0.0:
        edges = tuple(edge[::-1] for edge in edges[::-1])

    staged_diagnostics: dict = {}
    nodes, quads = _stage_patch(mesh.nodes, edges, staged_diagnostics)
    _checked_patch(nodes, quads, (edge1, edge2, edge3, edge4))
    mesh.replace_data(nodes=nodes, elements=np.vstack((mesh.elements, quads)))
    if _diagnostics is not None:
        _diagnostics.update(staged_diagnostics)
    return mesh


__all__ = ["build_unit_1_x_y_z"]
