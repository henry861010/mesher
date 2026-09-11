"""Fast element/circle-boundary intersection queries."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ...model import Mesh2D


def _find_intersect_element_and_sort(
    mesh: Mesh2D,
    center_x: float,
    center_y: float,
    radius: float,
) -> NDArray[np.int64]:
    """Return circle-intersecting element rows in angular order.

    Only intersections with an element's perimeter are considered.  A padded
    Tri3 row therefore contributes its three real edges, while a Quad4 row
    contributes four.  Tangencies and contacts at an edge endpoint count as
    intersections.

    Each element's unique intersection angles are normalized to ``[0, 2*pi)``
    and sorted.  Elements are then ordered lexicographically by those angle
    sequences.  A shorter sequence sorts before a longer sequence with the
    same prefix, and the original element row index is the final tie-breaker.

    Args:
        mesh: Mesh whose element perimeters are queried in the XY plane.
        center_x: Circle-center X coordinate.
        center_y: Circle-center Y coordinate.
        radius: Strictly positive circle radius.

    Returns:
        A one-dimensional int64 array of indices into ``mesh.elements``.

    Raises:
        TypeError: If ``mesh`` is not a :class:`Mesh2D`.
        ValueError: If the circle or mutable mesh data are invalid.
    """
    if not isinstance(mesh, Mesh2D):
        raise TypeError("mesh must be a Mesh2D instance")

    try:
        circle = np.asarray(
            [float(center_x), float(center_y), float(radius)],
            dtype=np.float64,
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "center_x, center_y, and radius must be real numbers"
        ) from error
    if not np.all(np.isfinite(circle)):
        raise ValueError("center_x, center_y, and radius must be finite")
    if circle[2] <= 0.0:
        raise ValueError("radius must be positive")

    nodes = np.asarray(mesh.nodes)
    elements = np.asarray(mesh.elements)
    if nodes.ndim != 2 or nodes.shape[1] not in (2, 3):
        raise ValueError("nodes must have shape (N, 2) or (N, 3)")
    if (
        not np.issubdtype(nodes.dtype, np.number)
        or np.issubdtype(nodes.dtype, np.bool_)
        or np.issubdtype(nodes.dtype, np.complexfloating)
    ):
        raise ValueError("nodes must have a real numeric dtype")
    if elements.ndim != 2 or elements.shape[1] != 4:
        raise ValueError("elements must have shape (M, 4)")
    if (
        not np.issubdtype(elements.dtype, np.integer)
        or np.issubdtype(elements.dtype, np.bool_)
    ):
        raise ValueError("elements must have an integer dtype")

    try:
        xy = nodes[:, :2].astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("nodes must be representable as float64") from error
    if not np.all(np.isfinite(xy)):
        raise ValueError("nodes must contain finite XY coordinates")
    if elements.size and (
        np.any(elements < 0) or np.any(elements >= nodes.shape[0])
    ):
        raise ValueError("elements contain an out-of-range node index")
    if elements.shape[0] == 0:
        return np.empty(0, dtype=np.int64)

    is_triangle = elements[:, 2] == elements[:, 3]
    triangle_valid = (
        (elements[:, 0] != elements[:, 1])
        & (elements[:, 0] != elements[:, 2])
        & (elements[:, 1] != elements[:, 2])
    )
    quad_valid = (
        (elements[:, 0] != elements[:, 1])
        & (elements[:, 0] != elements[:, 2])
        & (elements[:, 0] != elements[:, 3])
        & (elements[:, 1] != elements[:, 2])
        & (elements[:, 1] != elements[:, 3])
        & (elements[:, 2] != elements[:, 3])
    )
    if np.any(is_triangle & ~triangle_valid) or np.any(
        ~is_triangle & ~quad_valid
    ):
        raise ValueError(
            "elements must contain valid padded Tri3 or Quad4 connectivity"
        )

    center = circle[:2]
    circle_radius = float(circle[2])
    coordinate_scale = max(
        float(np.max(np.abs(center))),
        circle_radius,
    )
    coordinate_ulp = abs(float(np.spacing(coordinate_scale)))
    linear_tolerance = max(
        8.0 * coordinate_ulp,
        256.0 * np.finfo(np.float64).eps * circle_radius,
    )
    angular_tolerance = max(
        256.0 * np.finfo(np.float64).eps,
        linear_tolerance / circle_radius,
    )

    # Cheap broad phase: an edge can meet the circle only when its element's
    # axis-aligned bounding box overlaps the circle's bounding box.
    element_xy = xy[elements]
    element_minimum = np.min(element_xy, axis=1)
    element_maximum = np.max(element_xy, axis=1)
    with np.errstate(over="ignore", invalid="ignore"):
        circle_minimum = center - circle_radius - linear_tolerance
        circle_maximum = center + circle_radius + linear_tolerance
    if not np.all(np.isfinite([circle_minimum, circle_maximum])):
        raise ValueError("circle bounds exceed the float64 range")
    broad_phase_mask = np.all(
        element_maximum >= circle_minimum, axis=1
    ) & np.all(element_minimum <= circle_maximum, axis=1)
    candidate_indices = np.flatnonzero(broad_phase_mask)
    if candidate_indices.size == 0:
        return np.empty(0, dtype=np.int64)

    candidate_elements = elements[candidate_indices]
    edge_start_indices = candidate_elements
    edge_end_indices = np.roll(candidate_elements, -1, axis=1)

    # Give a shared edge one canonical orientation.  Adjacent elements then
    # calculate bit-identical intersection keys even when their perimeter
    # directions are opposite.
    canonical_start = np.minimum(edge_start_indices, edge_end_indices)
    canonical_end = np.maximum(edge_start_indices, edge_end_indices)
    with np.errstate(over="ignore", invalid="ignore"):
        start_offsets = xy[canonical_start] - center
        end_offsets = xy[canonical_end] - center
    if not np.all(np.isfinite(start_offsets)) or not np.all(
        np.isfinite(end_offsets)
    ):
        raise ValueError("node-to-center coordinate differences exceed float64")

    # Normalize each edge independently.  This keeps all quadratic terms near
    # unit scale and avoids overflow for large, but finite, coordinates.
    edge_scale = np.maximum.reduce(
        (
            np.max(np.abs(start_offsets), axis=2),
            np.max(np.abs(end_offsets), axis=2),
            np.full(start_offsets.shape[:2], circle_radius),
        )
    )
    normalized_start = start_offsets / edge_scale[..., None]
    normalized_end = end_offsets / edge_scale[..., None]
    normalized_radius = circle_radius / edge_scale
    normalized_tolerance = np.maximum(
        linear_tolerance / edge_scale,
        256.0 * np.finfo(np.float64).eps,
    )
    edge_vectors = normalized_end - normalized_start

    edge_squared_lengths = np.einsum(
        "...i,...i->...", edge_vectors, edge_vectors
    )
    start_dot_edge = np.einsum(
        "...i,...i->...", normalized_start, edge_vectors
    )
    circle_equation_at_start = (
        np.einsum("...i,...i->...", normalized_start, normalized_start)
        - normalized_radius * normalized_radius
    )
    half_discriminant = (
        start_dot_edge * start_dot_edge
        - edge_squared_lengths * circle_equation_at_start
    )
    discriminant_scale = (
        start_dot_edge * start_dot_edge
        + np.abs(edge_squared_lengths * circle_equation_at_start)
        + edge_squared_lengths * normalized_radius * normalized_radius
    )
    discriminant_tolerance = (
        512.0 * np.finfo(np.float64).eps * discriminant_scale
        + edge_squared_lengths * normalized_tolerance * normalized_tolerance
    )

    nondegenerate = edge_squared_lengths > (
        normalized_tolerance * normalized_tolerance
    )
    safe_squared_lengths = np.where(
        nondegenerate, edge_squared_lengths, 1.0
    )
    root_term = np.sqrt(np.maximum(half_discriminant, 0.0))
    parameters = np.stack(
        (
            (-start_dot_edge - root_term) / safe_squared_lengths,
            (-start_dot_edge + root_term) / safe_squared_lengths,
        ),
        axis=2,
    )
    parameter_tolerance = np.minimum(
        1.0,
        normalized_tolerance / np.sqrt(safe_squared_lengths),
    )
    valid_intersections = (
        nondegenerate[..., None]
        & (half_discriminant[..., None] >= -discriminant_tolerance[..., None])
        & (parameters >= -parameter_tolerance[..., None])
        & (parameters <= 1.0 + parameter_tolerance[..., None])
    )
    clipped_parameters = np.clip(parameters, 0.0, 1.0)
    intersection_offsets = (
        normalized_start[..., None, :]
        + clipped_parameters[..., None] * edge_vectors[..., None, :]
    )

    # A geometrically zero-length edge is a point/circle query.  This also
    # handles the padded fourth edge of every Tri3 without a special loop.
    start_distances = np.hypot(
        normalized_start[..., 0], normalized_start[..., 1]
    )
    degenerate_hits = (~nondegenerate) & (
        np.abs(start_distances - normalized_radius) <= normalized_tolerance
    )
    valid_intersections[..., 0] |= degenerate_hits
    intersection_offsets[..., 0, :] = np.where(
        degenerate_hits[..., None],
        normalized_start,
        intersection_offsets[..., 0, :],
    )

    angles = np.mod(
        np.arctan2(
            intersection_offsets[..., 1],
            intersection_offsets[..., 0],
        ),
        2.0 * np.pi,
    ).reshape(candidate_indices.size, 8)
    valid_intersections = valid_intersections.reshape(candidate_indices.size, 8)
    angles[~valid_intersections] = np.inf

    full_turn = 2.0 * np.pi
    seam_hits = np.isfinite(angles) & (
        (angles <= angular_tolerance)
        | (angles >= full_turn - angular_tolerance)
    )
    angles[seam_hits] = 0.0
    angles.sort(axis=1)

    adjacent_finite = np.isfinite(angles[:, 1:]) & np.isfinite(
        angles[:, :-1]
    )
    angle_differences = np.full(
        (angles.shape[0], angles.shape[1] - 1),
        np.inf,
        dtype=np.float64,
    )
    np.subtract(
        angles[:, 1:],
        angles[:, :-1],
        out=angle_differences,
        where=adjacent_finite,
    )
    duplicate_intersections = adjacent_finite & (
        angle_differences <= angular_tolerance
    )
    angles[:, 1:][duplicate_intersections] = np.inf
    angles.sort(axis=1)

    intersecting = np.isfinite(angles[:, 0])
    selected_indices = candidate_indices[intersecting].astype(
        np.int64, copy=False
    )
    if selected_indices.size == 0:
        return np.empty(0, dtype=np.int64)

    angle_keys = angles[intersecting]
    angle_keys[~np.isfinite(angle_keys)] = -np.inf
    lexicographic_keys = (selected_indices,) + tuple(
        angle_keys[:, column] for column in range(7, -1, -1)
    )
    order = np.lexsort(lexicographic_keys)
    return selected_indices[order]


def _get_areas(
    mesh: Mesh2D,
    center_x: float,
    center_y: float,
    radius: float,
    indices: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return selected element areas inside and outside a circle.

    The calculation is analytic: each directed element edge is split at its
    intersections with the circle.  A sub-edge inside the circle contributes
    its signed triangle area, while a sub-edge outside contributes the signed
    circular-sector area.  Summing those contributions around the perimeter
    gives the exact polygon/circle intersection area without approximating the
    circle with line segments.

    Tri3 rows must use the canonical ``[n0, n1, n2, n2]`` representation.
    Their zero-length padded edge contributes no area.  Quad4 rows use all
    four perimeter edges.  Input element winding does not affect the returned
    non-negative areas, and node Z coordinates are ignored.

    Args:
        mesh: Mesh providing valid, non-degenerate Tri3 or Quad4 elements.
        center_x: Circle-center X coordinate.
        center_y: Circle-center Y coordinate.
        radius: Strictly positive circle radius.
        indices: Unique one-dimensional integer sequence of element rows.  The
            result arrays follow this sequence's order.

    Returns:
        A pair ``(inner_areas, outer_areas)`` of float64 arrays.  The first
        contains each element's area inside the circle; the second contains
        its area outside.  Their element-wise sum is the full element area.

    Raises:
        TypeError: If ``mesh`` is not a Mesh2D or indices are not integers.
        ValueError: If the circle or indices shape is invalid, or indices
            contain duplicates.
        IndexError: If an element index is out of range.
    """
    if not isinstance(mesh, Mesh2D):
        raise TypeError("mesh must be a Mesh2D instance")

    try:
        circle = np.asarray(
            [float(center_x), float(center_y), float(radius)],
            dtype=np.float64,
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "center_x, center_y, and radius must be real numbers"
        ) from error
    if not np.all(np.isfinite(circle)):
        raise ValueError("center_x, center_y, and radius must be finite")
    if circle[2] <= 0.0:
        raise ValueError("radius must be positive")

    element_indices = np.asarray(indices)
    if element_indices.ndim != 1:
        raise ValueError("indices must be a one-dimensional sequence")
    if element_indices.size:
        if not np.issubdtype(
            element_indices.dtype, np.integer
        ) or np.issubdtype(element_indices.dtype, np.bool_):
            raise TypeError("indices must contain integers")
        if np.any(element_indices < 0) or np.any(
            element_indices >= mesh.element_count
        ):
            raise IndexError(
                "indices contain an element index that is out of range"
            )
        if np.unique(element_indices).size != element_indices.size:
            raise ValueError("indices must not contain duplicates")
        element_indices = element_indices.astype(np.int64, copy=False)
    else:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty.copy()

    circle_center = circle[:2]
    circle_radius = float(circle[2])
    circle_area = np.pi * circle_radius * circle_radius

    selected_elements = mesh.elements[element_indices]
    points = mesh.nodes[selected_elements, :2] - circle_center
    edge_starts = points
    edge_ends = np.roll(points, -1, axis=1)
    edge_vectors = edge_ends - edge_starts

    # Normalize each edge separately before solving its quadratic.  This keeps
    # the discriminant near unit scale for both small and large coordinates.
    edge_scale = np.maximum.reduce(
        (
            np.max(np.abs(edge_starts), axis=2),
            np.max(np.abs(edge_ends), axis=2),
            np.full(edge_starts.shape[:2], circle_radius),
        )
    )
    normalized_starts = edge_starts / edge_scale[..., None]
    normalized_vectors = edge_vectors / edge_scale[..., None]
    normalized_radius = circle_radius / edge_scale

    squared_lengths = np.einsum(
        "...i,...i->...", normalized_vectors, normalized_vectors
    )
    start_dot_vector = np.einsum(
        "...i,...i->...", normalized_starts, normalized_vectors
    )
    circle_equation_at_start = (
        np.einsum(
            "...i,...i->...", normalized_starts, normalized_starts
        )
        - normalized_radius * normalized_radius
    )
    half_discriminant = (
        start_dot_vector * start_dot_vector
        - squared_lengths * circle_equation_at_start
    )
    discriminant_scale = (
        start_dot_vector * start_dot_vector
        + np.abs(squared_lengths * circle_equation_at_start)
        + squared_lengths * normalized_radius * normalized_radius
    )
    discriminant_tolerance = (
        512.0 * np.finfo(np.float64).eps * discriminant_scale
    )

    nondegenerate = squared_lengths > np.finfo(np.float64).eps**2
    safe_squared_lengths = np.where(nondegenerate, squared_lengths, 1.0)
    root_term = np.sqrt(np.maximum(half_discriminant, 0.0))
    intersection_parameters = np.stack(
        (
            (-start_dot_vector - root_term) / safe_squared_lengths,
            (-start_dot_vector + root_term) / safe_squared_lengths,
        ),
        axis=2,
    )
    parameter_tolerance = 256.0 * np.finfo(np.float64).eps
    valid_intersections = (
        nondegenerate[..., None]
        & (
            half_discriminant[..., None]
            >= -discriminant_tolerance[..., None]
        )
        & (intersection_parameters >= -parameter_tolerance)
        & (intersection_parameters <= 1.0 + parameter_tolerance)
    )
    intersection_parameters = np.where(
        valid_intersections,
        np.clip(intersection_parameters, 0.0, 1.0),
        1.0,
    )

    # Invalid roots become 1.0.  Sorting therefore produces one complete
    # [0, 1] interval plus zero-length trailing intervals for a missed edge.
    cuts = np.concatenate(
        (
            np.zeros((*squared_lengths.shape, 1), dtype=np.float64),
            intersection_parameters,
            np.ones((*squared_lengths.shape, 1), dtype=np.float64),
        ),
        axis=2,
    )
    cuts.sort(axis=2)
    subedge_starts = (
        edge_starts[..., None, :]
        + cuts[..., :-1, None] * edge_vectors[..., None, :]
    )
    subedge_ends = (
        edge_starts[..., None, :]
        + cuts[..., 1:, None] * edge_vectors[..., None, :]
    )
    subedge_midpoints = 0.5 * (subedge_starts + subedge_ends)
    midpoint_inside = (
        np.einsum(
            "...i,...i->...", subedge_midpoints, subedge_midpoints
        )
        <= circle_radius * circle_radius
    )

    subedge_cross = (
        subedge_starts[..., 0] * subedge_ends[..., 1]
        - subedge_starts[..., 1] * subedge_ends[..., 0]
    )
    subedge_dot = (
        subedge_starts[..., 0] * subedge_ends[..., 0]
        + subedge_starts[..., 1] * subedge_ends[..., 1]
    )
    subedge_contributions = np.where(
        midpoint_inside,
        0.5 * subedge_cross,
        0.5
        * circle_radius
        * circle_radius
        * np.arctan2(subedge_cross, subedge_dot),
    )
    inner_areas = np.abs(np.sum(subedge_contributions, axis=(1, 2)))

    edge_cross = (
        edge_starts[..., 0] * edge_ends[..., 1]
        - edge_starts[..., 1] * edge_ends[..., 0]
    )
    total_areas = 0.5 * np.abs(np.sum(edge_cross, axis=1))

    # Snap round-off at the physical bounds.  In particular, sector terms for
    # a wholly external polygon mathematically cancel but may leave a tiny
    # residual in floating-point arithmetic.
    area_tolerance = (
        1024.0
        * np.finfo(np.float64).eps
        * np.maximum(total_areas, circle_area)
    )
    inner_areas[inner_areas <= area_tolerance] = 0.0
    nearly_full = total_areas - inner_areas <= area_tolerance
    inner_areas[nearly_full] = total_areas[nearly_full]
    np.clip(inner_areas, 0.0, total_areas, out=inner_areas)
    outer_areas = total_areas - inner_areas
    return inner_areas, outer_areas


def _tri_quad(    
    mesh: Mesh2D
):
    

__all__ = ["_find_intersect_element_and_sort", "_get_areas"]
