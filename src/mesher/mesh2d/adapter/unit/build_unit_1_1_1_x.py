"""Fill a convex four-sided patch with the 1-1-1-odd Quad4 template."""

from __future__ import annotations

import numpy as np

from ...model import Mesh2D


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _polygon_area_twice(points: np.ndarray) -> float:
    # Moving the origin near the polygon avoids cancellation for translated meshes.
    xy = points[:, :2] - points[0, :2]
    return float(np.sum(_cross(xy, np.roll(xy, -1, axis=0))))


def _segments_intersect(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    length_tolerance: float,
    area_tolerance: float,
) -> bool:
    """Include crossings, endpoint touches, and collinear overlaps."""

    def orientation(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float(_cross(q - p, r - p))

    def on_segment(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
        return bool(
            np.all(r >= np.minimum(p, q) - length_tolerance)
            and np.all(r <= np.maximum(p, q) + length_tolerance)
        )

    if np.any(
        np.maximum(np.minimum(a, b), np.minimum(c, d))
        > np.minimum(np.maximum(a, b), np.maximum(c, d)) + length_tolerance
    ):
        return False

    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)
    if o1 * o2 < -area_tolerance**2 and o3 * o4 < -area_tolerance**2:
        return True
    return bool(
        (abs(o1) <= area_tolerance and on_segment(a, b, c))
        or (abs(o2) <= area_tolerance and on_segment(a, b, d))
        or (abs(o3) <= area_tolerance and on_segment(c, d, a))
        or (abs(o4) <= area_tolerance and on_segment(c, d, b))
    )


def _validate_cyclic_boundary(
    mesh: Mesh2D | np.ndarray, edges: tuple[np.ndarray, ...], *, require_convex: bool
) -> np.ndarray:
    nodes = mesh.nodes if isinstance(mesh, Mesh2D) else mesh
    for number, edge in enumerate(edges, start=1):
        if not isinstance(edge, np.ndarray) or edge.ndim != 1:
            raise ValueError(f"edge{number} must be a one-dimensional NumPy array")
        if not np.issubdtype(edge.dtype, np.integer):
            raise ValueError(f"edge{number} must contain integer node indices")
        if edge.size < 2:
            raise ValueError(f"edge{number} must contain at least two nodes")
        if np.any(edge < 0) or np.any(edge >= len(nodes)):
            raise ValueError(f"edge{number} contains an out-of-range node index")

    for number in range(4):
        if edges[number][-1] != edges[(number + 1) % 4][0]:
            raise ValueError("the four edges must join cyclically")

    boundary = np.concatenate([edge[:-1] for edge in edges])
    if np.unique(boundary).size != boundary.size:
        raise ValueError("boundary node indices must be distinct except at edge joins")

    points = nodes[boundary]
    if not np.all(np.isfinite(points)):
        raise ValueError("boundary node coordinates must be finite")
    xy = points[:, :2]
    scale = float(np.max(np.ptp(xy, axis=0)))
    if scale == 0.0:
        raise ValueError("boundary has zero XY extent")
    length_tolerance = 64.0 * np.finfo(float).eps * scale
    area_tolerance = 64.0 * np.finfo(float).eps * scale * scale
    vectors = np.roll(xy, -1, axis=0) - xy
    if np.any(np.linalg.norm(vectors, axis=1) <= length_tolerance):
        raise ValueError("boundary contains a zero-length segment")

    area_twice = _polygon_area_twice(points)
    if abs(area_twice) <= area_tolerance * len(boundary):
        raise ValueError("boundary has zero XY area")
    if require_convex:
        turns = _cross(vectors, np.roll(vectors, -1, axis=0))
        if np.any(np.sign(area_twice) * turns < -area_tolerance):
            raise ValueError("boundary must be convex")

    for first in range(len(boundary)):
        a, b = xy[first], xy[(first + 1) % len(boundary)]
        for second in range(first + 1, len(boundary)):
            if second == first + 1 or (first == 0 and second == len(boundary) - 1):
                continue
            c, d = xy[second], xy[(second + 1) % len(boundary)]
            if _segments_intersect(a, b, c, d, length_tolerance, area_tolerance):
                raise ValueError("boundary must not self-intersect")

    return boundary


def _validate_boundary(
    mesh: Mesh2D | np.ndarray, edges: tuple[np.ndarray, ...], *, require_convex: bool = True
) -> int:
    _validate_cyclic_boundary(mesh, edges, require_convex=require_convex)
    lengths = [len(edge) for edge in edges]
    long_edges = [index for index, length in enumerate(lengths) if length > 2]
    if len(long_edges) > 1 or any(length % 2 for length in lengths):
        raise ValueError("three edges must have two nodes and the fourth an even number")
    return long_edges[0] if long_edges else 0


def _validate_quads(nodes: np.ndarray, quads: np.ndarray) -> None:
    points = nodes[quads]
    if not np.all(np.isfinite(points)):
        raise ValueError("generated node coordinates must be finite")
    xy = points[:, :, :2]
    edges = np.roll(xy, -1, axis=1) - xy
    corners = _cross(edges, np.roll(edges, -1, axis=1))
    scale = float(np.max(np.ptp(xy.reshape(-1, 2), axis=0)))
    area_tolerance = 64.0 * np.finfo(float).eps * scale * scale
    if np.any(corners <= area_tolerance):
        raise ValueError("generated quadrilaterals are degenerate or inverted")


def _stage_unit_1_1_1_x(
    nodes: np.ndarray, edges: tuple[np.ndarray, ...], long_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the template on a validated CCW boundary without changing a mesh."""
    edges = edges[long_index:] + edges[:long_index]
    lower = edges[0].tolist()
    top_right = int(edges[1][-1])
    top_left = int(edges[3][0])
    original_node_count = len(nodes)
    added_nodes: list[np.ndarray] = []
    added_quads: list[list[int]] = []

    def coordinate(index: int) -> np.ndarray:
        return nodes[index] if index < original_node_count else added_nodes[index - original_node_count]

    while len(lower) > 2:
        lower_points = np.asarray([coordinate(index) for index in lower])
        lengths = np.linalg.norm(np.diff(lower_points[:, :2], axis=0), axis=1)
        total_length = float(np.sum(lengths))
        fraction_left = float(lengths[0] / total_length)
        fraction_right = float(1.0 - lengths[-1] / total_length)
        upper_left, upper_right = coordinate(top_left), coordinate(top_right)
        new_left = 0.5 * (
            lower_points[1]
            + (1.0 - fraction_left) * upper_left
            + fraction_left * upper_right
        )
        new_right = 0.5 * (
            lower_points[-2]
            + (1.0 - fraction_right) * upper_left
            + fraction_right * upper_right
        )
        left_index = original_node_count + len(added_nodes)
        right_index = left_index + 1
        added_nodes.extend((new_left, new_right))
        added_quads.extend(
            (
                [lower[0], lower[1], left_index, top_left],
                [top_left, left_index, right_index, top_right],
                [lower[-2], lower[-1], top_right, right_index],
            )
        )
        lower = lower[1:-1]
        top_left, top_right = left_index, right_index

    added_quads.append([lower[0], lower[1], top_right, top_left])
    new_nodes = np.asarray(added_nodes, dtype=np.float64).reshape(-1, 3)
    new_quads = np.asarray(added_quads, dtype=np.int64)
    return new_nodes, new_quads


def build_unit_1_1_1_x(
    mesh: Mesh2D,
    edge1: np.ndarray,
    edge2: np.ndarray,
    edge3: np.ndarray,
    edge4: np.ndarray,
) -> Mesh2D:
    """Append an all-Quad4 patch bounded by three one-segment edges.

    The fourth edge has an odd number of segments. Each layer consumes one
    segment from either end of that edge and adds three quads. Coordinates of
    new nodes are halfway from the next boundary node to its arc-length-aligned
    point on the opposite edge. A final quad closes the patch.

    All inputs are validated before the mesh is changed. The patch interior is
    assumed to be empty; existing nodes and elements are preserved.
    """
    if not isinstance(mesh, Mesh2D):
        raise TypeError("mesh must be a Mesh2D instance")

    edges = (edge1, edge2, edge3, edge4)
    long_index = _validate_boundary(mesh, edges)
    boundary = np.concatenate([edge[:-1] for edge in edges])
    if _polygon_area_twice(mesh.nodes[boundary]) < 0.0:
        edges = tuple(edge[::-1] for edge in edges[::-1])
        long_index = 3 - long_index

    new_nodes, new_quads = _stage_unit_1_1_1_x(mesh.nodes, edges, long_index)
    combined_nodes = np.concatenate((mesh.nodes, new_nodes), axis=0)
    _validate_quads(combined_nodes, new_quads)
    combined_elements = np.concatenate((mesh.elements, new_quads), axis=0)
    mesh.replace_data(nodes=combined_nodes, elements=combined_elements)
    return mesh


__all__ = ["build_unit_1_1_1_x"]
