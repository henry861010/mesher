"""Vectorized utilities for querying and preparing circular 2D meshes.

The module has three groups of helpers:

* ``_normalize_indices`` gives every public query the same element-selection
  rules.
* The ``get_*`` functions inspect element topology or element/circle geometry
  without changing the mesh.
* ``to_circle`` moves existing nodes onto a circle.  Its two private helpers
  precompute guide/circle roots and resolve sparse node/guide constraints.

All geometric calculations use node X and Y coordinates.  A node's Z value is
ignored by read-only queries and preserved by ``to_circle``.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ...model import Mesh2D
from ..utils.pattern_segments import _PatternGuideSet, _coerce_pattern_guides


_GUIDE_PAIR_BATCH_SIZE = 262_144


def _normalize_indices(
    element_count: int,
    indices: ArrayLike | None = None,
) -> NDArray[np.int64]:
    """Normalize an optional element-row selection to a one-dimensional array.

    This is the shared selection contract for the public functions in this
    module.  Explicit selections preserve the caller's order.  Duplicates are
    rejected because several result arrays are row-aligned with the selection
    and a repeated element would make downstream ownership ambiguous.

    Args:
        element_count: Total number of rows in ``mesh.elements``.  Valid
            explicit indices are in the half-open range
            ``[0, element_count)``.
        indices: Optional one-dimensional integer sequence of element-row
            indices.  ``None`` selects every row in ascending order.  An empty
            sequence selects no rows.

    Returns:
        An int64 array containing the normalized element-row indices.  The
        array has shape ``(K,)``, where ``K`` is the selection size.

    Raises:
        TypeError: If an explicit selection contains non-integers or booleans.
        ValueError: If ``indices`` is not one-dimensional or has duplicates.
        IndexError: If an index is negative or greater than or equal to
            ``element_count``.
    """
    if indices is None:
        return np.arange(element_count, dtype=np.int64)

    element_indices = np.asarray(indices)
    if element_indices.ndim != 1:
        raise ValueError("indices must be a one-dimensional sequence")
    if element_indices.size == 0:
        return np.empty(0, dtype=np.int64)
    if not np.issubdtype(element_indices.dtype, np.integer) or np.issubdtype(
        element_indices.dtype, np.bool_
    ):
        raise TypeError("indices must contain integers")
    if np.any(element_indices < 0) or np.any(
        element_indices >= element_count
    ):
        raise IndexError(
            "indices contain an element index that is out of range"
        )
    if np.unique(element_indices).size != element_indices.size:
        raise ValueError("indices must not contain duplicates")
    return element_indices.astype(np.int64, copy=False)


def get_circle_intersect(
    mesh: Mesh2D,
    center_x: float,
    center_y: float,
    radius: float,
    indices: ArrayLike | None = None,
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
        indices: Optional unique one-dimensional integer sequence of element
            rows to process. ``None`` processes every element.

    Returns:
        A one-dimensional int64 array of indices into ``mesh.elements``.

    Raises:
        TypeError: If ``mesh`` is not a :class:`Mesh2D` or indices are not
            integers.
        ValueError: If the circle, mutable mesh data, or indices are invalid.
        IndexError: If an element index is out of range.

    Notes:
        The implementation first rejects elements whose axis-aligned bounding
        boxes cannot touch the circle.  It then solves at most two roots for
        each of the four connectivity edges in one NumPy batch.  Shared-vertex
        roots are deduplicated by angle before the final lexicographic sort.
        The mesh is never mutated.
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
    element_indices = _normalize_indices(elements.shape[0], indices)
    selected_elements = elements[element_indices]
    if selected_elements.shape[0] == 0:
        return np.empty(0, dtype=np.int64)

    is_triangle = selected_elements[:, 2] == selected_elements[:, 3]
    triangle_valid = (
        (selected_elements[:, 0] != selected_elements[:, 1])
        & (selected_elements[:, 0] != selected_elements[:, 2])
        & (selected_elements[:, 1] != selected_elements[:, 2])
    )
    quad_valid = (
        (selected_elements[:, 0] != selected_elements[:, 1])
        & (selected_elements[:, 0] != selected_elements[:, 2])
        & (selected_elements[:, 0] != selected_elements[:, 3])
        & (selected_elements[:, 1] != selected_elements[:, 2])
        & (selected_elements[:, 1] != selected_elements[:, 3])
        & (selected_elements[:, 2] != selected_elements[:, 3])
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
    element_xy = xy[selected_elements]
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
    candidate_positions = np.flatnonzero(broad_phase_mask)
    if candidate_positions.size == 0:
        return np.empty(0, dtype=np.int64)

    candidate_indices = element_indices[candidate_positions]
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


def get_inner_outer_areas(
    mesh: Mesh2D,
    center_x: float,
    center_y: float,
    radius: float,
    indices: ArrayLike | None = None,
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
        indices: Optional unique one-dimensional integer sequence of element
            rows. The result arrays follow this sequence's order. ``None``
            processes every element.

    Returns:
        A pair ``(inner_areas, outer_areas)`` of float64 arrays.  The first
        contains each element's area inside the circle; the second contains
        its area outside.  Their element-wise sum is the full element area.

    Raises:
        TypeError: If ``mesh`` is not a Mesh2D or indices are not integers.
        ValueError: If the circle or indices shape is invalid, or indices
            contain duplicates.
        IndexError: If an element index is out of range.

    Notes:
        Each edge is solved independently in normalized coordinates to avoid
        overflow.  Its intersection parameters split the edge into sub-edges.
        A midpoint test chooses whether a sub-edge contributes triangle area
        or circular-sector area.  The mesh is never mutated.
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

    element_indices = _normalize_indices(mesh.element_count, indices)
    if element_indices.size == 0:
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


def get_tri_quad(
    mesh: Mesh2D,
    indices: ArrayLike | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return triangle and quadrilateral element-row indices.

    Tri3 elements use the canonical padded representation
    ``[n0, n1, n2, n2]``.  Classification therefore depends only on whether
    the final two connectivity entries match; all other rows are classified
    as Quad4 elements. The returned indices retain the selected element order.

    Args:
        mesh: Mesh whose elements are classified.
        indices: Optional unique one-dimensional integer sequence of element
            rows to process. ``None`` processes every element.

    Returns:
        A pair ``(triangle_indices, quadrilateral_indices)`` containing
        one-dimensional int64 arrays of indices into ``mesh.elements``.

    Raises:
        TypeError: If ``mesh`` is not a :class:`Mesh2D` or indices are not
            integers.
        ValueError: If indices are not one-dimensional or contain duplicates.
        IndexError: If an element index is out of range.

    Notes:
        This function classifies connectivity only; it does not validate
        element geometry or node coordinates and never mutates the mesh.
    """
    if not isinstance(mesh, Mesh2D):
        raise TypeError("mesh must be a Mesh2D instance")

    element_indices = _normalize_indices(mesh.element_count, indices)
    selected_elements = mesh.elements[element_indices]
    is_triangle = selected_elements[:, 2] == selected_elements[:, 3]
    triangle_indices = element_indices[is_triangle]
    quadrilateral_indices = element_indices[~is_triangle]
    return triangle_indices, quadrilateral_indices


def get_intersect_nodes(
    mesh: Mesh2D,
    center_x: float,
    center_y: float,
    radius: float,
    indices: ArrayLike | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Return each element's unique intersections with a circle boundary.

    Intersections are ordered counter-clockwise by their angle around the
    circle, starting at the positive X axis.  A Tri3 can have at most six
    unique intersections and a Quad4 can have at most eight, so the result is
    padded to eight slots for both topologies.  Only the first
    ``intersection_counts[i]`` entries in result row ``i`` are valid; all
    remaining entries are ``(0, 0)``. Tangencies and contacts at an element
    vertex count as one intersection, even when the same vertex is found on
    two edges.

    The calculation is vectorized over every element and edge.  Node Z
    coordinates are ignored.

    Args:
        mesh: Mesh whose element perimeters are queried in the XY plane.
        center_x: Circle-center X coordinate.
        center_y: Circle-center Y coordinate.
        radius: Strictly positive circle radius.
        indices: Optional unique one-dimensional integer sequence of element
            rows to process. Result rows follow this sequence's order.
            ``None`` processes every element.

    Returns:
        A pair ``(intersection_counts, intersection_nodes)``. For ``K``
        selected elements, counts has shape ``(K,)`` and dtype int64, while
        nodes has shape ``(K, 8, 2)`` and dtype float64.

    Raises:
        TypeError: If ``mesh`` is not a :class:`Mesh2D` or indices are not
            integers.
        ValueError: If the circle, mutable mesh data, or indices are invalid.
        IndexError: If an element index is out of range.

    Notes:
        The result stays aligned with the caller's element selection even
        though an axis-aligned bounding-box broad phase skips the expensive
        solve for irrelevant elements.  Candidate edges are solved together,
        sorted by polar angle, and deduplicated before being copied into the
        fixed-width output.  The mesh is never mutated.
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

    element_indices = _normalize_indices(elements.shape[0], indices)
    selected_elements = elements[element_indices]
    selected_count = selected_elements.shape[0]
    intersection_counts = np.zeros(selected_count, dtype=np.int64)
    intersection_nodes = np.zeros((selected_count, 8, 2), dtype=np.float64)
    if selected_count == 0:
        return intersection_counts, intersection_nodes

    is_triangle = selected_elements[:, 2] == selected_elements[:, 3]
    triangle_valid = (
        (selected_elements[:, 0] != selected_elements[:, 1])
        & (selected_elements[:, 0] != selected_elements[:, 2])
        & (selected_elements[:, 1] != selected_elements[:, 2])
    )
    quad_valid = (
        (selected_elements[:, 0] != selected_elements[:, 1])
        & (selected_elements[:, 0] != selected_elements[:, 2])
        & (selected_elements[:, 0] != selected_elements[:, 3])
        & (selected_elements[:, 1] != selected_elements[:, 2])
        & (selected_elements[:, 1] != selected_elements[:, 3])
        & (selected_elements[:, 2] != selected_elements[:, 3])
    )
    if np.any(is_triangle & ~triangle_valid) or np.any(
        ~is_triangle & ~quad_valid
    ):
        raise ValueError(
            "elements must contain valid padded Tri3 or Quad4 connectivity"
        )

    center = circle[:2]
    circle_radius = float(circle[2])
    coordinate_scale = max(float(np.max(np.abs(center))), circle_radius)
    coordinate_ulp = abs(float(np.spacing(coordinate_scale)))
    linear_tolerance = max(
        8.0 * coordinate_ulp,
        256.0 * np.finfo(np.float64).eps * circle_radius,
    )
    angular_tolerance = max(
        256.0 * np.finfo(np.float64).eps,
        linear_tolerance / circle_radius,
    )

    # The result remains aligned with the selection, but the more expensive
    # quadratic solve only runs for bounding boxes that overlap the circle.
    element_xy = xy[selected_elements]
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
    candidate_positions = np.flatnonzero(broad_phase_mask)
    if candidate_positions.size == 0:
        return intersection_counts, intersection_nodes

    candidate_elements = selected_elements[candidate_positions]
    candidate_count = candidate_positions.size
    edge_start_indices = candidate_elements
    edge_end_indices = np.roll(candidate_elements, -1, axis=1)

    # A canonical edge direction gives adjacent elements bit-identical roots
    # on their shared edge, regardless of their perimeter winding.
    canonical_start = np.minimum(edge_start_indices, edge_end_indices)
    canonical_end = np.maximum(edge_start_indices, edge_end_indices)
    with np.errstate(over="ignore", invalid="ignore"):
        start_offsets = xy[canonical_start] - center
        end_offsets = xy[canonical_end] - center
    if not np.all(np.isfinite(start_offsets)) or not np.all(
        np.isfinite(end_offsets)
    ):
        raise ValueError("node-to-center coordinate differences exceed float64")

    # Normalize each edge independently to keep the quadratic stable for very
    # large and very small finite coordinates.
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

    # This includes the padded fourth edge of a Tri3.  If its point lies on
    # the circle it is later deduplicated with the two real adjacent edges.
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

    flat_offsets = intersection_offsets.reshape(candidate_count, 8, 2)
    flat_valid = valid_intersections.reshape(candidate_count, 8)
    angles = np.mod(
        np.arctan2(flat_offsets[..., 1], flat_offsets[..., 0]),
        2.0 * np.pi,
    )
    angles[~flat_valid] = np.inf

    full_turn = 2.0 * np.pi
    seam_hits = np.isfinite(angles) & (
        (angles <= angular_tolerance)
        | (angles >= full_turn - angular_tolerance)
    )
    angles[seam_hits] = 0.0

    order = np.argsort(angles, axis=1, kind="stable")
    angles = np.take_along_axis(angles, order, axis=1)
    flat_offsets = np.take_along_axis(
        flat_offsets, order[..., None], axis=1
    )
    flat_scales = np.take_along_axis(
        np.repeat(edge_scale, 2, axis=1), order, axis=1
    )

    adjacent_finite = np.isfinite(angles[:, 1:]) & np.isfinite(
        angles[:, :-1]
    )
    angle_differences = np.full(
        (candidate_count, 7), np.inf, dtype=np.float64
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

    # A second stable sort compacts unique intersections into the leading
    # slots while preserving their counter-clockwise order.
    compact_order = np.argsort(angles, axis=1, kind="stable")
    angles = np.take_along_axis(angles, compact_order, axis=1)
    flat_offsets = np.take_along_axis(
        flat_offsets, compact_order[..., None], axis=1
    )
    flat_scales = np.take_along_axis(flat_scales, compact_order, axis=1)
    valid_unique = np.isfinite(angles)
    candidate_intersection_counts = np.count_nonzero(
        valid_unique, axis=1
    ).astype(np.int64, copy=False)

    with np.errstate(over="ignore", invalid="ignore"):
        calculated_nodes = center + flat_offsets * flat_scales[..., None]
    if not np.all(np.isfinite(calculated_nodes[valid_unique])):
        raise ValueError("intersection coordinates exceed the float64 range")
    calculated_nodes[~valid_unique] = 0.0
    intersection_counts[candidate_positions] = candidate_intersection_counts
    intersection_nodes[candidate_positions] = calculated_nodes
    return intersection_counts, intersection_nodes


def _prepare_guide_circle_roots(
    pattern_guides: _PatternGuideSet,
    center: NDArray[np.float64],
    radius: float,
    numerical_tolerance: float,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Precompute finite-segment/circle roots for every pattern guide.

    A prepared guide stores one fixed axis and one varying axis.  Consequently
    a circle intersection can be represented by only its varying-axis value;
    the corresponding fixed coordinate is already available in
    ``pattern_guides.fixed_values``.  Both infinite-line roots are calculated
    in one vectorized operation, then marked valid only if they lie within the
    finite segment bounds.  A tangent occupies the first slot only.

    Args:
        pattern_guides: Validated horizontal and vertical finite segments.
            All per-guide metadata arrays have length ``L``.
        center: Float64 circle center with shape ``(2,)``.
        radius: Strictly positive circle radius.
        numerical_tolerance: Non-negative scale-aware tolerance used only for
            round-off at tangencies and finite segment endpoints.

    Returns:
        ``(root_values, valid_roots)``.  Both arrays have shape ``(L, 2)``.
        ``root_values[i, j]`` is a coordinate on guide ``i``'s varying axis;
        ``valid_roots[i, j]`` says whether that root is a usable intersection
        with the finite segment.  Values in invalid slots must not be used.

    Notes:
        This helper assumes its inputs were normalized by ``to_circle`` and
        does not mutate the guide set or mesh.
    """
    guide_count = len(pattern_guides)
    root_values = np.zeros((guide_count, 2), dtype=np.float64)
    valid_roots = np.zeros((guide_count, 2), dtype=np.bool_)
    if guide_count == 0:
        return root_values, valid_roots

    fixed_centers = center[pattern_guides.fixed_axes]
    varying_centers = center[pattern_guides.varying_axes]
    with np.errstate(over="ignore", invalid="ignore"):
        fixed_distances = np.abs(
            pattern_guides.fixed_values - fixed_centers
        )

    fixed_tolerances = np.maximum(
        pattern_guides.fixed_tolerances,
        numerical_tolerance,
    )
    line_intersects = (
        np.isfinite(fixed_distances)
        & (fixed_distances <= radius + fixed_tolerances)
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        normalized_distances = np.minimum(fixed_distances / radius, 1.0)
        root_offsets = radius * np.sqrt(
            np.maximum(
                0.0,
                (1.0 - normalized_distances)
                * (1.0 + normalized_distances),
            )
        )
        root_values[:, 0] = varying_centers - root_offsets
        root_values[:, 1] = varying_centers + root_offsets

    bound_tolerances = np.maximum(
        pattern_guides.bound_tolerances,
        numerical_tolerance,
    )
    valid_roots[:] = (
        line_intersects[:, None]
        & np.isfinite(root_values)
        & (
            root_values
            >= pattern_guides.lower_bounds[:, None]
            - bound_tolerances[:, None]
        )
        & (
            root_values
            <= pattern_guides.upper_bounds[:, None]
            + bound_tolerances[:, None]
        )
    )

    # A tangent is one geometric root, not an ambiguous choice between two
    # numerically identical roots.
    duplicate_roots = (
        2.0 * root_offsets
        <= np.maximum(fixed_tolerances, bound_tolerances)
    )
    valid_roots[duplicate_roots, 1] = False
    return root_values, valid_roots


def _accumulate_guide_constraints(
    candidate_xy: NDArray[np.float64],
    pattern_guides: _PatternGuideSet,
    guide_positions: NDArray[np.intp],
    effective_guide_tolerances: NDArray[np.float64],
    root_values: NDArray[np.float64],
    valid_roots: NDArray[np.bool_],
    root_choice_tolerances: NDArray[np.float64],
    constrained: NDArray[np.bool_],
    blocked: NDArray[np.bool_],
    target_minimums: NDArray[np.float64],
    target_maximums: NDArray[np.float64],
    reference_targets: NDArray[np.float64],
) -> None:
    """Resolve sparse candidate/guide matches for one fixed-axis orientation.

    The function receives either all vertical guides or all horizontal guides.
    It sorts them by fixed coordinate and uses ``searchsorted`` to build only
    plausible node/guide pairs.  Those pairs are processed in bounded batches:
    exact point-to-segment distance decides guide membership, valid circle
    roots are selected by proximity, and per-node target summaries are updated
    in place.  There is no Python loop over individual nodes or guides.

    Args:
        candidate_xy: XY coordinates of the circle-band candidates, shape
            ``(C, 2)``.  Row positions are the keys used by every accumulator.
        pattern_guides: Prepared metadata for all ``L`` pattern segments.
        guide_positions: Positions of guides that share one fixed axis.  The
            array has shape ``(G,)`` and may be unsorted.
        effective_guide_tolerances: Per-guide Euclidean hit tolerances with
            shape ``(L,)``.  Each value is the larger of the caller's guide
            tolerance and the guide's numerical tolerance.
        root_values: Precomputed varying-axis circle roots with shape
            ``(L, 2)``.
        valid_roots: Boolean mask with shape ``(L, 2)`` identifying roots that
            lie on their finite segments.
        root_choice_tolerances: Per-guide numerical thresholds, shape ``(L,)``,
            for detecting an equal-distance choice between two roots.
        constrained: Boolean output accumulator with shape ``(C,)``.  A row is
            set when at least one guide passes within its hit tolerance.
        blocked: Boolean output accumulator with shape ``(C,)``.  A row is set
            when any matching guide has no usable or uniquely nearest root.
        target_minimums: Float64 output accumulator with shape ``(C, 2)``.
            Component-wise minima across every usable target are written here.
        target_maximums: Float64 output accumulator with shape ``(C, 2)``.
            Component-wise maxima across every usable target are written here.
        reference_targets: Float64 output accumulator with shape ``(C, 2)``.
            The first usable target for each candidate is retained as the
            deterministic coordinate used if all constraints agree.

    Returns:
        ``None``.  The five accumulator arrays are modified in place.  The
        caller later compares target minima and maxima to detect conflicts
        across this and the other guide orientation.

    Notes:
        ``_GUIDE_PAIR_BATCH_SIZE`` bounds temporary pair arrays.  A candidate
        matched by conflicting guides is not rejected here unless a root is
        missing or ambiguous; geometric disagreement is resolved by
        ``to_circle`` after both orientations have been accumulated.
    """
    if candidate_xy.shape[0] == 0 or guide_positions.size == 0:
        return

    fixed_axis = int(pattern_guides.fixed_axes[guide_positions[0]])
    varying_axis = 1 - fixed_axis
    order = guide_positions[
        np.argsort(
            pattern_guides.fixed_values[guide_positions],
            kind="stable",
        )
    ]
    sorted_fixed_values = pattern_guides.fixed_values[order]
    maximum_lookup_tolerance = float(
        np.max(effective_guide_tolerances[order])
    )

    candidate_fixed_values = candidate_xy[:, fixed_axis]
    with np.errstate(over="ignore", invalid="ignore"):
        lookup_lower = candidate_fixed_values - maximum_lookup_tolerance
        lookup_upper = candidate_fixed_values + maximum_lookup_tolerance
    lower_positions = np.searchsorted(
        sorted_fixed_values,
        lookup_lower,
        side="left",
    )
    upper_positions = np.searchsorted(
        sorted_fixed_values,
        lookup_upper,
        side="right",
    )
    pair_counts = (upper_positions - lower_positions).astype(
        np.int64,
        copy=False,
    )
    pair_prefix = np.empty(candidate_xy.shape[0] + 1, dtype=np.int64)
    pair_prefix[0] = 0
    np.cumsum(pair_counts, out=pair_prefix[1:])

    node_start = 0
    candidate_count = candidate_xy.shape[0]
    while node_start < candidate_count:
        maximum_pair_end = (
            int(pair_prefix[node_start]) + _GUIDE_PAIR_BATCH_SIZE
        )
        node_end = int(
            np.searchsorted(
                pair_prefix,
                maximum_pair_end,
                side="right",
            )
            - 1
        )
        node_end = min(candidate_count, max(node_start + 1, node_end))

        block_counts = pair_counts[node_start:node_end]
        pair_count = int(np.sum(block_counts, dtype=np.int64))
        if pair_count == 0:
            node_start = node_end
            continue

        node_positions = np.repeat(
            np.arange(node_start, node_end, dtype=np.intp),
            block_counts,
        )
        block_pair_starts = np.cumsum(
            block_counts,
            dtype=np.int64,
        ) - block_counts
        offsets_within_node = (
            np.arange(pair_count, dtype=np.int64)
            - np.repeat(block_pair_starts, block_counts)
        )
        sorted_guide_positions = (
            np.repeat(lower_positions[node_start:node_end], block_counts)
            + offsets_within_node
        )
        paired_guides = order[sorted_guide_positions]

        paired_points = candidate_xy[node_positions]
        fixed_differences = np.abs(
            paired_points[:, fixed_axis]
            - pattern_guides.fixed_values[paired_guides]
        )
        varying_values = paired_points[:, varying_axis]
        varying_distances = np.maximum(
            pattern_guides.lower_bounds[paired_guides] - varying_values,
            0.0,
        )
        varying_distances = np.maximum(
            varying_distances,
            varying_values - pattern_guides.upper_bounds[paired_guides],
        )
        point_segment_distances = np.hypot(
            fixed_differences,
            varying_distances,
        )
        guide_hits = (
            point_segment_distances
            <= effective_guide_tolerances[paired_guides]
        )
        if not np.any(guide_hits):
            node_start = node_end
            continue

        hit_nodes = node_positions[guide_hits]
        hit_guides = paired_guides[guide_hits]
        hit_varying_values = varying_values[guide_hits]
        constrained[hit_nodes] = True

        hit_root_values = root_values[hit_guides]
        hit_root_validity = valid_roots[hit_guides]
        root_distances = np.abs(
            hit_root_values - hit_varying_values[:, None]
        )
        root_distances[~hit_root_validity] = np.inf
        nearest_root_positions = np.argmin(root_distances, axis=1)
        nearest_root_distances = np.take_along_axis(
            root_distances,
            nearest_root_positions[:, None],
            axis=1,
        )[:, 0]
        has_root = np.isfinite(nearest_root_distances)
        root_distance_differences = np.full(
            hit_nodes.size,
            np.inf,
            dtype=np.float64,
        )
        both_roots_valid = np.all(hit_root_validity, axis=1)
        np.subtract(
            root_distances[:, 0],
            root_distances[:, 1],
            out=root_distance_differences,
            where=both_roots_valid,
        )
        ambiguous_root = (
            both_roots_valid
            & (
                np.abs(root_distance_differences)
                <= root_choice_tolerances[hit_guides]
            )
        )
        invalid_constraints = ~has_root | ambiguous_root
        blocked[hit_nodes[invalid_constraints]] = True

        usable = ~invalid_constraints
        if not np.any(usable):
            node_start = node_end
            continue

        usable_nodes = hit_nodes[usable]
        usable_guides = hit_guides[usable]
        usable_root_positions = nearest_root_positions[usable]
        usable_targets = np.empty(
            (usable_nodes.size, 2),
            dtype=np.float64,
        )
        usable_targets[:, fixed_axis] = pattern_guides.fixed_values[
            usable_guides
        ]
        usable_targets[:, varying_axis] = root_values[
            usable_guides,
            usable_root_positions,
        ]

        for axis in (0, 1):
            np.minimum.at(
                target_minimums[:, axis],
                usable_nodes,
                usable_targets[:, axis],
            )
            np.maximum.at(
                target_maximums[:, axis],
                usable_nodes,
                usable_targets[:, axis],
            )

        unique_nodes, first_positions = np.unique(
            usable_nodes,
            return_index=True,
        )
        unset = ~np.isfinite(reference_targets[unique_nodes, 0])
        reference_targets[unique_nodes[unset]] = usable_targets[
            first_positions[unset]
        ]
        node_start = node_end


def _merge_close_circle_targets(
    target_xy: NDArray[np.float64],
    guided: NDArray[np.bool_],
    center: NDArray[np.float64],
    tolerance: float,
    numerical_tolerance: float,
) -> NDArray[np.float64]:
    """Merge nearby projected targets in one clockwise circular sweep.

    Targets are ordered clockwise from the positive X axis.  Each free target
    closer than ``tolerance`` to the current group representative receives the
    representative's XY coordinate.  A guided target promotes itself to the
    representative of a free group so guide/circle intersections remain
    fixed.  Two distinct guided representatives form a group boundary even
    when they are closer than ``tolerance``.

    The first and last groups are compared after the linear pass so targets
    straddling the positive-X seam receive the same treatment.  Only the
    returned coordinate array changes; node identities and connectivity are
    deliberately outside this helper's scope.
    """
    target_count = target_xy.shape[0]
    if target_count < 2 or tolerance <= 0.0:
        return target_xy.copy()

    offsets = target_xy - center
    clockwise_angles = np.mod(
        -np.arctan2(offsets[:, 1], offsets[:, 0]),
        2.0 * np.pi,
    )
    order = np.argsort(clockwise_angles, kind="stable")

    # Each tuple describes a contiguous slice of ``order`` and the target
    # within that slice whose coordinate is the group's representative.
    groups: list[tuple[int, int, int, bool]] = []
    group_start = 0
    representative = int(order[0])
    representative_is_guided = bool(guided[representative])

    for ordered_position in range(1, target_count):
        candidate = int(order[ordered_position])
        difference = target_xy[candidate] - target_xy[representative]
        distance = float(np.hypot(difference[0], difference[1]))
        candidate_is_guided = bool(guided[candidate])
        distinct_guides = (
            representative_is_guided
            and candidate_is_guided
            and distance > numerical_tolerance
        )
        if distance >= tolerance or distinct_guides:
            groups.append(
                (
                    group_start,
                    ordered_position,
                    representative,
                    representative_is_guided,
                )
            )
            group_start = ordered_position
            representative = candidate
            representative_is_guided = candidate_is_guided
            continue

        if candidate_is_guided and not representative_is_guided:
            representative = candidate
            representative_is_guided = True

    groups.append(
        (
            group_start,
            target_count,
            representative,
            representative_is_guided,
        )
    )

    # Reconcile the circular seam.  The first group is earlier in the sweep,
    # except that a guide representative always takes precedence over a free
    # representative on either side of the seam.
    if len(groups) > 1:
        first_start, first_end, first_rep, first_guided = groups[0]
        last_start, last_end, last_rep, last_guided = groups[-1]
        seam_difference = target_xy[last_rep] - target_xy[first_rep]
        seam_distance = float(
            np.hypot(seam_difference[0], seam_difference[1])
        )
        distinct_guides = (
            first_guided
            and last_guided
            and seam_distance > numerical_tolerance
        )
        if seam_distance < tolerance and not distinct_guides:
            if last_guided and not first_guided:
                first_rep = last_rep
                first_guided = True
            else:
                last_rep = first_rep
                last_guided = first_guided
            groups[0] = (
                first_start,
                first_end,
                first_rep,
                first_guided,
            )
            groups[-1] = (
                last_start,
                last_end,
                last_rep,
                last_guided,
            )

    merged_xy = target_xy.copy()
    for start, end, group_representative, _ in groups:
        merged_xy[order[start:end]] = target_xy[group_representative]
    return merged_xy


def to_circle(
    mesh: Mesh2D,
    center_x: float,
    center_y: float,
    radius: float,
    tolerance: float,
    guide_tolerance: float,
    guide_segments,
    indices: ArrayLike | None = None,
) -> Mesh2D:
    """Move nodes near a circle onto its boundary, honoring pattern guides.

    ``tolerance`` selects nodes by radial distance from the circle, while
    ``guide_tolerance`` independently selects nodes close to a finite pattern
    segment.  Unconstrained nodes move radially.  A constrained node moves to
    the nearest finite-segment/circle intersection only when every segment
    touching it agrees on the same target; otherwise that node is left alone.
    After projection, movable circle targets are swept clockwise from the
    positive X axis.  A target strictly closer than ``tolerance`` to the
    current representative receives the same XY coordinate.  Guide targets
    remain fixed and take precedence over nearby unconstrained targets.

    When ``indices`` is provided it contains element-row indices, and only
    nodes referenced by those elements are considered.  ``None`` considers
    every mesh node, including unreferenced nodes.  Node Z coordinates and all
    element connectivity are preserved.  Validation and target construction
    finish before the mesh is mutated.

    Args:
        mesh: Mesh whose existing nodes may be moved in the XY plane.
        center_x: Circle-center X coordinate.
        center_y: Circle-center Y coordinate.
        radius: Strictly positive circle radius.
        tolerance: Non-negative radial distance from the circle to process.
        guide_tolerance: Non-negative Euclidean point-to-segment tolerance.
        guide_segments: Finite horizontal or vertical pattern segments with
            shape ``(L, 2, 2)``, a prepared guide set, or ``None``.
        indices: Optional unique one-dimensional element-row indices.

    Returns:
        The same ``mesh`` instance after an atomic batch coordinate update.

    Raises:
        TypeError: If ``mesh`` or ``indices`` has an invalid type.
        ValueError: If mesh data, circle values, tolerances, or guides are
            invalid.
        IndexError: If an element index is out of range.

    Notes:
        The implementation is organized into five stages that can be followed
        directly in the code below:

        1. Validate all scalar, mesh, selection, and guide inputs.
        2. Convert element selection to unique nodes and keep only nodes in the
           inclusive radial tolerance band.
        3. Precompute guide/circle roots and accumulate sparse guide matches.
        4. Radially project free nodes; accept a constrained target only when
           every touching guide agrees.  Missing, ambiguous, or conflicting
           constraints leave that node unchanged.
        5. Merge nearby movable targets in a clockwise circular sweep while
           preserving guide anchors.
        6. Validate every proposed target, then perform one XY assignment.

        Nodes exactly at the circle center cannot be projected radially and
        remain unchanged.  Per-node guide conflicts are normal outcomes, not
        exceptions.  The merge changes coordinates only, so it can create
        zero-length element edges.  No element-quality, topology, or inversion
        check is performed.
    """
    if not isinstance(mesh, Mesh2D):
        raise TypeError("mesh must be a Mesh2D instance")

    try:
        circle_values = np.asarray(
            [
                float(center_x),
                float(center_y),
                float(radius),
                float(tolerance),
                float(guide_tolerance),
            ],
            dtype=np.float64,
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "center_x, center_y, radius, tolerance, and guide_tolerance "
            "must be real numbers"
        ) from error
    if not np.all(np.isfinite(circle_values)):
        raise ValueError(
            "center_x, center_y, radius, tolerance, and guide_tolerance "
            "must be finite"
        )
    circle_radius = float(circle_values[2])
    radial_tolerance = float(circle_values[3])
    guide_tolerance = float(circle_values[4])
    if circle_radius <= 0.0:
        raise ValueError("radius must be positive")
    if radial_tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    if guide_tolerance < 0.0:
        raise ValueError("guide_tolerance must be non-negative")

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
        float_nodes = nodes.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("nodes must be representable as float64") from error
    if not np.all(np.isfinite(float_nodes)):
        raise ValueError("nodes must contain finite coordinates")
    if elements.size and (
        np.any(elements < 0) or np.any(elements >= nodes.shape[0])
    ):
        raise ValueError("elements contain an out-of-range node index")

    center = circle_values[:2]
    coordinate_scale = max(
        1.0,
        float(np.max(np.abs(center))),
        circle_radius,
    )
    numerical_tolerance = (
        128.0 * np.finfo(np.float64).eps * coordinate_scale
    )
    pattern_guides = _coerce_pattern_guides(
        guide_segments,
        coordinate_scale=coordinate_scale,
        minimum_tolerance=numerical_tolerance,
    )

    if indices is None:
        selected_node_indices = np.arange(nodes.shape[0], dtype=np.int64)
    else:
        element_indices = _normalize_indices(elements.shape[0], indices)
        if element_indices.size == 0:
            return mesh
        selected_node_indices = np.unique(
            elements[element_indices].reshape(-1)
        ).astype(np.int64, copy=False)
    if selected_node_indices.size == 0:
        return mesh

    selected_xy = float_nodes[selected_node_indices, :2]
    with np.errstate(over="ignore", invalid="ignore"):
        selected_offsets = selected_xy - center
        selected_distances = np.hypot(
            selected_offsets[:, 0],
            selected_offsets[:, 1],
        )
        radial_residuals = np.abs(selected_distances - circle_radius)
        candidate_threshold = radial_tolerance + numerical_tolerance
    candidate_selection = radial_residuals <= candidate_threshold
    candidate_positions = np.flatnonzero(candidate_selection)
    if candidate_positions.size == 0:
        return mesh

    candidate_node_indices = selected_node_indices[candidate_positions]
    candidate_xy = selected_xy[candidate_positions]
    candidate_offsets = selected_offsets[candidate_positions]
    candidate_distances = selected_distances[candidate_positions]
    candidate_count = candidate_positions.size

    constrained = np.zeros(candidate_count, dtype=np.bool_)
    blocked = np.zeros(candidate_count, dtype=np.bool_)
    target_minimums = np.full(
        (candidate_count, 2),
        np.inf,
        dtype=np.float64,
    )
    target_maximums = np.full(
        (candidate_count, 2),
        -np.inf,
        dtype=np.float64,
    )
    reference_targets = np.full(
        (candidate_count, 2),
        np.nan,
        dtype=np.float64,
    )

    if len(pattern_guides):
        root_values, valid_roots = _prepare_guide_circle_roots(
            pattern_guides,
            center,
            circle_radius,
            numerical_tolerance,
        )
        guide_numerical_tolerances = np.maximum.reduce(
            (
                pattern_guides.fixed_tolerances,
                pattern_guides.bound_tolerances,
                np.full(len(pattern_guides), numerical_tolerance),
            )
        )
        effective_guide_tolerances = np.maximum(
            guide_tolerance,
            guide_numerical_tolerances,
        )
        root_choice_tolerances = guide_numerical_tolerances

        for fixed_axis in (0, 1):
            orientation_guides = np.flatnonzero(
                pattern_guides.fixed_axes == fixed_axis
            ).astype(np.intp, copy=False)
            _accumulate_guide_constraints(
                candidate_xy,
                pattern_guides,
                orientation_guides,
                effective_guide_tolerances,
                root_values,
                valid_roots,
                root_choice_tolerances,
                constrained,
                blocked,
                target_minimums,
                target_maximums,
                reference_targets,
            )

    has_constraint_target = np.isfinite(reference_targets[:, 0])
    target_spreads = np.zeros(candidate_count, dtype=np.float64)
    target_spreads[has_constraint_target] = np.hypot(
        target_maximums[has_constraint_target, 0]
        - target_minimums[has_constraint_target, 0],
        target_maximums[has_constraint_target, 1]
        - target_minimums[has_constraint_target, 1],
    )
    target_scales = np.full(candidate_count, coordinate_scale)
    if np.any(has_constraint_target):
        target_scales[has_constraint_target] = np.maximum(
            target_scales[has_constraint_target],
            np.max(
                np.abs(reference_targets[has_constraint_target]),
                axis=1,
            ),
        )
    agreement_tolerances = (
        256.0 * np.finfo(np.float64).eps * target_scales
    )
    conflicting = target_spreads > agreement_tolerances
    guided_movable = (
        constrained
        & ~blocked
        & has_constraint_target
        & ~conflicting
    )
    radial_movable = (~constrained) & (candidate_distances > 0.0)
    movable = radial_movable | guided_movable
    if not np.any(movable):
        return mesh

    target_xy = candidate_xy.copy()
    radial_positions = np.flatnonzero(radial_movable)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        target_xy[radial_positions] = (
            center
            + circle_radius
            * candidate_offsets[radial_positions]
            / candidate_distances[radial_positions, None]
        )
    target_xy[guided_movable] = reference_targets[guided_movable]

    movable_positions = np.flatnonzero(movable)
    target_xy[movable_positions] = _merge_close_circle_targets(
        target_xy[movable_positions],
        guided_movable[movable_positions],
        center,
        radial_tolerance,
        numerical_tolerance,
    )

    if not np.all(np.isfinite(target_xy[movable])):
        raise ValueError("projected circle coordinates exceed float64 range")
    projected_offsets = target_xy[movable] - center
    projected_distances = np.hypot(
        projected_offsets[:, 0],
        projected_offsets[:, 1],
    )
    validation_tolerance = max(
        4.0 * numerical_tolerance,
        circle_radius * 1.0e-12,
    )
    if np.any(
        np.abs(projected_distances - circle_radius)
        > validation_tolerance
    ):
        raise ValueError("a projected node is not on the target circle")

    mesh.nodes[candidate_node_indices[movable], :2] = target_xy[movable]
    return mesh


def imprint_circle(
    mesh: Mesh2D,
    center_x: float,
    center_y: float,
    radius: float,
    tolerance: float,
    indices: ArrayLike | None = None,
) -> Mesh2D:
    """Split selected elements where a circle properly crosses two edges.

    The two circle/element intersections are joined by a straight chord.  A
    Tri3 is split into a Tri3 and a Quad4, while a Quad4 produces either two
    Quad4 elements (opposite crossed edges) or three mixed elements (adjacent
    crossed edges).  Tangencies, same-edge double intersections, and elements
    with any other number of proper crossings are left unchanged.

    ``tolerance`` is a geometric merge distance.  An intersection within that
    distance of an edge endpoint reuses the endpoint after projecting its XY
    coordinate radially onto the circle.  Intersections within that distance
    of one another are treated as a single contact.  Newly inserted nodes on
    a shared selected edge are deduplicated; their Z coordinate is linearly
    interpolated along the source edge.

    When ``indices`` is provided, only those element rows may have their
    connectivity split.  Reused endpoint nodes are shared mesh nodes, so
    snapping one can also move an unselected neighbouring element.  No
    conformity closure is performed across the selection boundary.

    The operation is transactional: all geometry and connectivity are built
    and validated before :meth:`Mesh2D.replace_data` is called.

    Args:
        mesh: Mesh to update in place.
        center_x: Circle-center X coordinate.
        center_y: Circle-center Y coordinate.
        radius: Strictly positive circle radius.
        tolerance: Non-negative endpoint/intersection merge distance.
        indices: Optional unique one-dimensional element-row selection.

    Returns:
        The same ``mesh`` instance after a successful batch update.

    Raises:
        TypeError: If ``mesh`` or ``indices`` has an invalid type.
        ValueError: If mesh data, circle values, tolerance, or selected
            element geometry are invalid.
        IndexError: If an element index is out of range.
    """
    if not isinstance(mesh, Mesh2D):
        raise TypeError("mesh must be a Mesh2D instance")

    try:
        circle_values = np.asarray(
            [
                float(center_x),
                float(center_y),
                float(radius),
                float(tolerance),
            ],
            dtype=np.float64,
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "center_x, center_y, radius, and tolerance must be real numbers"
        ) from error
    if not np.all(np.isfinite(circle_values)):
        raise ValueError(
            "center_x, center_y, radius, and tolerance must be finite"
        )

    center = circle_values[:2]
    circle_radius = float(circle_values[2])
    merge_tolerance = float(circle_values[3])
    if circle_radius <= 0.0:
        raise ValueError("radius must be positive")
    if merge_tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")

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
        float_nodes = nodes.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("nodes must be representable as float64") from error
    if not np.all(np.isfinite(float_nodes)):
        raise ValueError("nodes must contain finite coordinates")
    if elements.size and (
        np.any(elements < 0) or np.any(elements >= nodes.shape[0])
    ):
        raise ValueError("elements contain an out-of-range node index")

    element_indices = _normalize_indices(elements.shape[0], indices)
    if element_indices.size == 0:
        return mesh

    selected_elements = elements[element_indices]
    is_triangle = selected_elements[:, 2] == selected_elements[:, 3]
    triangle_valid = (
        (selected_elements[:, 0] != selected_elements[:, 1])
        & (selected_elements[:, 0] != selected_elements[:, 2])
        & (selected_elements[:, 1] != selected_elements[:, 2])
    )
    sorted_quads = np.sort(selected_elements, axis=1)
    quad_valid = np.all(np.diff(sorted_quads, axis=1) != 0, axis=1)
    if np.any(is_triangle & ~triangle_valid) or np.any(
        ~is_triangle & ~quad_valid
    ):
        raise ValueError(
            "elements must contain valid padded Tri3 or Quad4 connectivity"
        )

    xy = float_nodes[:, :2]
    selected_xy = xy[selected_elements]
    topology_counts = np.where(is_triangle, 3, 4).astype(np.int64)
    edge_positions = np.arange(4, dtype=np.int64)[None, :]
    valid_edges = edge_positions < topology_counts[:, None]

    # Validate the documented simple Tri3/convex Quad4 precondition.  This is
    # done before the broad phase so invalid selected geometry cannot be hidden
    # merely because the requested circle is far away.
    next_xy = np.roll(selected_xy, -1, axis=1)
    previous_xy = np.roll(selected_xy, 1, axis=1)
    triangle_rows = np.flatnonzero(is_triangle)
    if triangle_rows.size:
        next_xy[triangle_rows, 2] = selected_xy[triangle_rows, 0]
        next_xy[triangle_rows, 3] = selected_xy[triangle_rows, 3]
        previous_xy[triangle_rows, 0] = selected_xy[triangle_rows, 2]
    outgoing = next_xy - selected_xy
    incoming_reverse = previous_xy - selected_xy
    corner_cross = (
        outgoing[..., 0] * incoming_reverse[..., 1]
        - outgoing[..., 1] * incoming_reverse[..., 0]
    )
    edge_lengths = np.hypot(outgoing[..., 0], outgoing[..., 1])
    element_scales = np.max(
        np.where(valid_edges, edge_lengths, 0.0), axis=1
    )
    geometry_tolerances = (
        2048.0
        * np.finfo(np.float64).eps
        * np.maximum(element_scales, 1.0) ** 2
    )
    triangle_cross = corner_cross[:, 0]
    triangle_geometry_valid = np.abs(triangle_cross) > geometry_tolerances
    quad_positive = np.all(
        corner_cross > geometry_tolerances[:, None], axis=1
    )
    quad_negative = np.all(
        corner_cross < -geometry_tolerances[:, None], axis=1
    )
    if np.any(is_triangle & ~triangle_geometry_valid) or np.any(
        ~is_triangle & ~(quad_positive | quad_negative)
    ):
        raise ValueError(
            "selected elements must be non-degenerate Tri3 or convex Quad4"
        )

    coordinate_scale = max(
        float(np.max(np.abs(center))),
        circle_radius,
    )
    coordinate_ulp = abs(float(np.spacing(coordinate_scale)))
    numerical_tolerance = max(
        8.0 * coordinate_ulp,
        256.0 * np.finfo(np.float64).eps * circle_radius,
    )
    effective_merge_tolerance = max(
        merge_tolerance,
        numerical_tolerance,
    )

    element_minimum = np.min(selected_xy, axis=1)
    element_maximum = np.max(selected_xy, axis=1)
    with np.errstate(over="ignore", invalid="ignore"):
        circle_minimum = center - circle_radius - numerical_tolerance
        circle_maximum = center + circle_radius + numerical_tolerance
    if not np.all(np.isfinite([circle_minimum, circle_maximum])):
        raise ValueError("circle bounds exceed the float64 range")
    broad_phase = np.all(
        element_maximum >= circle_minimum, axis=1
    ) & np.all(element_minimum <= circle_maximum, axis=1)
    candidate_positions = np.flatnonzero(broad_phase)
    if candidate_positions.size == 0:
        return mesh

    candidate_elements = selected_elements[candidate_positions]
    candidate_xy = selected_xy[candidate_positions]
    candidate_counts = topology_counts[candidate_positions]
    candidate_valid_edges = valid_edges[candidate_positions]
    candidate_count = candidate_positions.size

    local_edge_starts = candidate_elements
    local_edge_ends = np.roll(candidate_elements, -1, axis=1)
    candidate_triangle_rows = np.flatnonzero(candidate_counts == 3)
    if candidate_triangle_rows.size:
        local_edge_ends[candidate_triangle_rows, 2] = candidate_elements[
            candidate_triangle_rows, 0
        ]
        local_edge_ends[candidate_triangle_rows, 3] = candidate_elements[
            candidate_triangle_rows, 3
        ]
    canonical_starts = np.minimum(local_edge_starts, local_edge_ends)
    canonical_ends = np.maximum(local_edge_starts, local_edge_ends)
    with np.errstate(over="ignore", invalid="ignore"):
        start_offsets = xy[canonical_starts] - center
        end_offsets = xy[canonical_ends] - center
    if not np.all(np.isfinite(start_offsets)) or not np.all(
        np.isfinite(end_offsets)
    ):
        raise ValueError("node-to-center coordinate differences exceed float64")

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
    normalized_numerical_tolerance = np.maximum(
        numerical_tolerance / edge_scale,
        256.0 * np.finfo(np.float64).eps,
    )
    edge_vectors = normalized_end - normalized_start
    squared_lengths = np.einsum(
        "...i,...i->...", edge_vectors, edge_vectors
    )
    start_dot_edge = np.einsum(
        "...i,...i->...", normalized_start, edge_vectors
    )
    circle_at_start = (
        np.einsum("...i,...i->...", normalized_start, normalized_start)
        - normalized_radius * normalized_radius
    )
    half_discriminant = (
        start_dot_edge * start_dot_edge
        - squared_lengths * circle_at_start
    )
    discriminant_scale = (
        start_dot_edge * start_dot_edge
        + np.abs(squared_lengths * circle_at_start)
        + squared_lengths * normalized_radius * normalized_radius
    )
    discriminant_tolerance = (
        512.0 * np.finfo(np.float64).eps * discriminant_scale
        + squared_lengths
        * normalized_numerical_tolerance
        * normalized_numerical_tolerance
    )
    nondegenerate = candidate_valid_edges & (
        squared_lengths
        > normalized_numerical_tolerance * normalized_numerical_tolerance
    )
    safe_squared_lengths = np.where(nondegenerate, squared_lengths, 1.0)
    root_term = np.sqrt(np.maximum(half_discriminant, 0.0))
    canonical_parameters = np.stack(
        (
            (-start_dot_edge - root_term) / safe_squared_lengths,
            (-start_dot_edge + root_term) / safe_squared_lengths,
        ),
        axis=2,
    )
    parameter_tolerance = np.minimum(
        1.0,
        normalized_numerical_tolerance / np.sqrt(safe_squared_lengths),
    )
    valid_roots = (
        nondegenerate[..., None]
        & (
            half_discriminant[..., None]
            >= -discriminant_tolerance[..., None]
        )
        & (canonical_parameters >= -parameter_tolerance[..., None])
        & (canonical_parameters <= 1.0 + parameter_tolerance[..., None])
    )
    canonical_parameters = np.clip(canonical_parameters, 0.0, 1.0)
    normalized_root_offsets = (
        normalized_start[..., None, :]
        + canonical_parameters[..., None] * edge_vectors[..., None, :]
    )
    with np.errstate(over="ignore", invalid="ignore"):
        root_xy = (
            center
            + normalized_root_offsets * edge_scale[..., None, None]
        )
    if not np.all(np.isfinite(root_xy[valid_roots])):
        raise ValueError("intersection coordinates exceed the float64 range")

    # Work in local perimeter orientation for endpoint tests and later
    # insertion order, while retaining canonical parameters for shared-edge
    # node keys and Z interpolation.
    canonical_is_local = local_edge_starts == canonical_starts
    local_parameters = np.where(
        canonical_is_local[..., None],
        canonical_parameters,
        1.0 - canonical_parameters,
    )
    root_edge_positions = np.broadcast_to(
        np.arange(4, dtype=np.int64)[None, :, None],
        valid_roots.shape,
    )
    root_ordinals = np.broadcast_to(
        np.arange(2, dtype=np.int64)[None, None, :],
        valid_roots.shape,
    )

    local_start_xy = xy[local_edge_starts][..., None, :]
    local_end_xy = xy[local_edge_ends][..., None, :]
    start_distances = np.hypot(
        root_xy[..., 0] - local_start_xy[..., 0],
        root_xy[..., 1] - local_start_xy[..., 1],
    )
    end_distances = np.hypot(
        root_xy[..., 0] - local_end_xy[..., 0],
        root_xy[..., 1] - local_end_xy[..., 1],
    )
    start_snap = (
        valid_roots
        & (start_distances <= effective_merge_tolerance)
        & (start_distances <= end_distances)
    )
    end_snap = (
        valid_roots
        & ~start_snap
        & (end_distances <= effective_merge_tolerance)
    )
    snap_vertex_positions = np.full(valid_roots.shape, -1, dtype=np.int64)
    snap_vertex_positions[start_snap] = root_edge_positions[start_snap]
    end_vertex_positions = (
        root_edge_positions + 1
    ) % candidate_counts[:, None, None]
    snap_vertex_positions[end_snap] = end_vertex_positions[end_snap]

    candidate_offsets = candidate_xy - center
    candidate_distances = np.hypot(
        candidate_offsets[..., 0], candidate_offsets[..., 1]
    )
    projectable_vertices = candidate_distances > 0.0
    projected_vertices = candidate_xy.copy()
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        projected_vertices[projectable_vertices] = (
            center
            + circle_radius
            * candidate_offsets[projectable_vertices]
            / candidate_distances[projectable_vertices, None]
        )

    # A snapped vertex is a proper boundary crossing only when the two local
    # perimeter directions immediately leave it on opposite sides of the
    # circle.  A zero radial derivative is tangent and therefore outside.
    previous_positions = (
        np.arange(4, dtype=np.int64)[None, :] - 1
    ) % candidate_counts[:, None]
    next_positions = (
        np.arange(4, dtype=np.int64)[None, :] + 1
    ) % candidate_counts[:, None]
    row_positions = np.arange(candidate_count, dtype=np.int64)[:, None]
    previous_vertices = candidate_xy[row_positions, previous_positions]
    next_vertices = candidate_xy[row_positions, next_positions]
    projected_offsets = projected_vertices - center
    previous_derivative = 2.0 * np.einsum(
        "...i,...i->...",
        projected_offsets,
        previous_vertices - projected_vertices,
    )
    next_derivative = 2.0 * np.einsum(
        "...i,...i->...",
        projected_offsets,
        next_vertices - projected_vertices,
    )
    derivative_scales = np.maximum.reduce(
        (
            np.abs(previous_derivative),
            np.abs(next_derivative),
            np.full(previous_derivative.shape, circle_radius**2),
        )
    )
    derivative_tolerances = (
        2048.0 * np.finfo(np.float64).eps * derivative_scales
    )
    previous_inside = previous_derivative < -derivative_tolerances
    next_inside = next_derivative < -derivative_tolerances
    vertex_crossings = (
        projectable_vertices
        & (previous_inside != next_inside)
        & valid_edges[candidate_positions]
    )

    flat_valid_roots = valid_roots.reshape(candidate_count, 8)
    flat_discriminants = np.repeat(half_discriminant, 2, axis=1)
    flat_discriminant_tolerances = np.repeat(
        discriminant_tolerance, 2, axis=1
    )
    flat_snap_vertices = snap_vertex_positions.reshape(candidate_count, 8)
    snapped = flat_snap_vertices >= 0
    safe_snap_vertices = np.maximum(flat_snap_vertices, 0)
    snapped_crossings = vertex_crossings[
        np.arange(candidate_count)[:, None], safe_snap_vertices
    ]
    interior_crossings = (
        flat_discriminants > flat_discriminant_tolerances
    )
    flat_valid_roots &= np.where(
        snapped,
        snapped_crossings,
        interior_crossings,
    )

    flat_root_xy = root_xy.reshape(candidate_count, 8, 2)
    flat_edge_positions = root_edge_positions.reshape(candidate_count, 8)
    flat_root_ordinals = root_ordinals.reshape(candidate_count, 8)
    flat_canonical_parameters = canonical_parameters.reshape(
        candidate_count, 8
    )
    flat_local_parameters = local_parameters.reshape(candidate_count, 8)
    flat_snap_nodes = np.full((candidate_count, 8), -1, dtype=np.int64)
    snap_rows, snap_slots = np.nonzero(snapped)
    if snap_rows.size:
        flat_snap_nodes[snap_rows, snap_slots] = candidate_elements[
            snap_rows,
            flat_snap_vertices[snap_rows, snap_slots],
        ]

    root_support = (
        np.left_shift(np.uint8(1), flat_edge_positions.astype(np.uint8))
    )
    if snap_rows.size:
        snap_vertices = flat_snap_vertices[snap_rows, snap_slots]
        previous_edges = (
            snap_vertices - 1
        ) % candidate_counts[snap_rows]
        support = np.left_shift(
            np.uint8(1), snap_vertices.astype(np.uint8)
        ) | np.left_shift(np.uint8(1), previous_edges.astype(np.uint8))
        root_support[snap_rows, snap_slots] = support

    angles = np.mod(
        np.arctan2(
            flat_root_xy[..., 1] - center[1],
            flat_root_xy[..., 0] - center[0],
        ),
        2.0 * np.pi,
    )
    angles[~flat_valid_roots] = np.inf
    angle_order = np.argsort(angles, axis=1, kind="stable")

    def _take_rows(values):
        if values.ndim == 3:
            return np.take_along_axis(values, angle_order[..., None], axis=1)
        return np.take_along_axis(values, angle_order, axis=1)

    sorted_angles = _take_rows(angles)
    sorted_xy = _take_rows(flat_root_xy)
    sorted_edges = _take_rows(flat_edge_positions)
    sorted_ordinals = _take_rows(flat_root_ordinals)
    sorted_canonical_parameters = _take_rows(flat_canonical_parameters)
    sorted_local_parameters = _take_rows(flat_local_parameters)
    sorted_snap_nodes = _take_rows(flat_snap_nodes)
    sorted_support = _take_rows(root_support)
    sorted_valid = _take_rows(flat_valid_roots)
    valid_counts = np.count_nonzero(sorted_valid, axis=1).astype(np.int64)

    tolerance_ratio = min(
        1.0,
        0.5 * (effective_merge_tolerance / circle_radius),
    )
    angular_merge_tolerance = 2.0 * np.arcsin(tolerance_ratio)
    circular_gaps = np.full(
        (candidate_count, 8), -np.inf, dtype=np.float64
    )
    if candidate_count:
        consecutive_valid = (
            np.arange(7, dtype=np.int64)[None, :]
            < (valid_counts[:, None] - 1)
        )
        consecutive_gaps = np.full(
            (candidate_count, 7), -np.inf, dtype=np.float64
        )
        np.subtract(
            sorted_angles[:, 1:],
            sorted_angles[:, :-1],
            out=consecutive_gaps,
            where=consecutive_valid,
        )
        circular_gaps[:, :7] = consecutive_gaps
        rows_with_roots = valid_counts > 0
        root_rows = np.flatnonzero(rows_with_roots)
        last_positions = valid_counts[rows_with_roots] - 1
        circular_gaps[root_rows, last_positions] = (
            sorted_angles[root_rows, 0]
            + 2.0 * np.pi
            - sorted_angles[root_rows, last_positions]
        )

    largest_gap_positions = np.argmax(circular_gaps, axis=1)
    rotation_starts = np.where(
        valid_counts > 0,
        (largest_gap_positions + 1) % np.maximum(valid_counts, 1),
        0,
    )
    rotated_positions = (
        rotation_starts[:, None] + np.arange(8, dtype=np.int64)[None, :]
    ) % np.maximum(valid_counts[:, None], 1)
    rotated_valid = (
        np.arange(8, dtype=np.int64)[None, :] < valid_counts[:, None]
    )

    def _rotate_rows(values):
        if values.ndim == 3:
            return np.take_along_axis(
                values, rotated_positions[..., None], axis=1
            )
        return np.take_along_axis(values, rotated_positions, axis=1)

    rotated_angles = _rotate_rows(sorted_angles)
    rotated_xy = _rotate_rows(sorted_xy)
    rotated_edges = _rotate_rows(sorted_edges)
    rotated_ordinals = _rotate_rows(sorted_ordinals)
    rotated_canonical_parameters = _rotate_rows(
        sorted_canonical_parameters
    )
    rotated_local_parameters = _rotate_rows(sorted_local_parameters)
    rotated_snap_nodes = _rotate_rows(sorted_snap_nodes)
    rotated_support = _rotate_rows(sorted_support)

    group_starts = np.zeros((candidate_count, 8), dtype=np.bool_)
    group_starts[:, 0] = rotated_valid[:, 0]
    rotated_gaps = np.zeros(
        (candidate_count, 7), dtype=np.float64
    )
    rotated_pair_valid = rotated_valid[:, 1:] & rotated_valid[:, :-1]
    np.subtract(
        rotated_angles[:, 1:],
        rotated_angles[:, :-1],
        out=rotated_gaps,
        where=rotated_pair_valid,
    )
    np.mod(rotated_gaps, 2.0 * np.pi, out=rotated_gaps)
    group_starts[:, 1:] = (
        rotated_valid[:, 1:]
        & (rotated_gaps > angular_merge_tolerance)
    )
    group_ids = np.cumsum(group_starts, axis=1, dtype=np.int64) - 1
    unique_counts = np.count_nonzero(group_starts, axis=1).astype(np.int64)

    group_xy = np.zeros((candidate_count, 8, 2), dtype=np.float64)
    group_edges = np.zeros((candidate_count, 8), dtype=np.int64)
    group_ordinals = np.zeros((candidate_count, 8), dtype=np.int64)
    group_canonical_parameters = np.zeros(
        (candidate_count, 8), dtype=np.float64
    )
    group_local_parameters = np.zeros(
        (candidate_count, 8), dtype=np.float64
    )
    start_rows, start_slots = np.nonzero(group_starts)
    start_groups = group_ids[start_rows, start_slots]
    if start_rows.size:
        group_xy[start_rows, start_groups] = rotated_xy[
            start_rows, start_slots
        ]
        group_edges[start_rows, start_groups] = rotated_edges[
            start_rows, start_slots
        ]
        group_ordinals[start_rows, start_groups] = rotated_ordinals[
            start_rows, start_slots
        ]
        group_canonical_parameters[start_rows, start_groups] = (
            rotated_canonical_parameters[start_rows, start_slots]
        )
        group_local_parameters[start_rows, start_groups] = (
            rotated_local_parameters[start_rows, start_slots]
        )

    group_support = np.zeros((candidate_count, 8), dtype=np.uint8)
    group_snap_minimum = np.full(
        (candidate_count, 8), np.iinfo(np.int64).max, dtype=np.int64
    )
    group_snap_maximum = np.full(
        (candidate_count, 8), -1, dtype=np.int64
    )
    aggregate_rows, aggregate_slots = np.nonzero(rotated_valid)
    aggregate_groups = group_ids[aggregate_rows, aggregate_slots]
    if aggregate_rows.size:
        np.bitwise_or.at(
            group_support,
            (aggregate_rows, aggregate_groups),
            rotated_support[aggregate_rows, aggregate_slots],
        )
        aggregate_snap_nodes = rotated_snap_nodes[
            aggregate_rows, aggregate_slots
        ]
        has_snap_node = aggregate_snap_nodes >= 0
        np.minimum.at(
            group_snap_minimum,
            (
                aggregate_rows[has_snap_node],
                aggregate_groups[has_snap_node],
            ),
            aggregate_snap_nodes[has_snap_node],
        )
        np.maximum.at(
            group_snap_maximum,
            (
                aggregate_rows[has_snap_node],
                aggregate_groups[has_snap_node],
            ),
            aggregate_snap_nodes[has_snap_node],
        )

    ambiguous_snap = (
        (group_snap_maximum >= 0)
        & (group_snap_minimum != group_snap_maximum)
    )
    group_has_snap = group_snap_maximum >= 0
    support_as_int = group_support.astype(np.int64)
    merged_across_edges_without_vertex = (
        ~group_has_snap
        & (support_as_int != 0)
        & ((support_as_int & (support_as_int - 1)) != 0)
    )
    two_crossings = unique_counts == 2
    different_edges = (
        group_support[:, 0] & group_support[:, 1]
    ) == 0
    eligible_candidates = (
        two_crossings
        & different_edges
        & ~ambiguous_snap[:, 0]
        & ~ambiguous_snap[:, 1]
        & ~merged_across_edges_without_vertex[:, 0]
        & ~merged_across_edges_without_vertex[:, 1]
    )
    eligible_candidate_positions = np.flatnonzero(eligible_candidates)
    if eligible_candidate_positions.size == 0:
        return mesh

    eligible_selection_positions = candidate_positions[
        eligible_candidate_positions
    ]
    eligible_element_indices = element_indices[eligible_selection_positions]
    eligible_elements = candidate_elements[eligible_candidate_positions]
    eligible_counts = candidate_counts[eligible_candidate_positions]
    eligible_count = eligible_candidate_positions.size
    eligible_group_edges = group_edges[eligible_candidate_positions, :2]
    eligible_group_ordinals = group_ordinals[
        eligible_candidate_positions, :2
    ]
    eligible_group_xy = group_xy[eligible_candidate_positions, :2]
    eligible_group_canonical_parameters = group_canonical_parameters[
        eligible_candidate_positions, :2
    ]
    eligible_group_local_parameters = group_local_parameters[
        eligible_candidate_positions, :2
    ]
    eligible_snap_nodes = group_snap_maximum[
        eligible_candidate_positions, :2
    ]

    # Assign one node to every unique unsnapped edge/root.  The root ordinal
    # disambiguates the two possible intersections of the same undirected
    # edge, even though eligible elements normally use only one of them.
    intersection_node_indices = eligible_snap_nodes.copy()
    unsnapped = eligible_snap_nodes < 0
    unsnapped_rows, unsnapped_groups = np.nonzero(unsnapped)
    new_node_coordinates = np.empty((0, 3), dtype=np.float64)
    if unsnapped_rows.size:
        unsnapped_edges = eligible_group_edges[
            unsnapped_rows, unsnapped_groups
        ]
        edge_start_nodes = eligible_elements[
            unsnapped_rows, unsnapped_edges
        ]
        edge_end_nodes = eligible_elements[
            unsnapped_rows,
            (unsnapped_edges + 1) % eligible_counts[unsnapped_rows],
        ]
        key_starts = np.minimum(edge_start_nodes, edge_end_nodes)
        key_ends = np.maximum(edge_start_nodes, edge_end_nodes)
        edge_keys = np.column_stack(
            (
                key_starts,
                key_ends,
                eligible_group_ordinals[unsnapped_rows, unsnapped_groups],
            )
        )
        unique_keys, first_key_positions, key_inverse = np.unique(
            edge_keys,
            axis=0,
            return_index=True,
            return_inverse=True,
        )
        if nodes.shape[0] + unique_keys.shape[0] > np.iinfo(np.int32).max:
            raise ValueError("imprint would exceed the int32 node-index range")

        unique_rows = unsnapped_rows[first_key_positions]
        unique_groups = unsnapped_groups[first_key_positions]
        unique_parameters = eligible_group_canonical_parameters[
            unique_rows, unique_groups
        ]
        unique_start_nodes = unique_keys[:, 0]
        unique_end_nodes = unique_keys[:, 1]
        new_node_coordinates = np.empty(
            (unique_keys.shape[0], 3), dtype=np.float64
        )
        new_node_coordinates[:, :2] = eligible_group_xy[
            unique_rows, unique_groups
        ]
        new_node_coordinates[:, 2] = (
            float_nodes[unique_start_nodes, 2]
            + unique_parameters
            * (
                float_nodes[unique_end_nodes, 2]
                - float_nodes[unique_start_nodes, 2]
            )
        )
        assigned_nodes = nodes.shape[0] + key_inverse
        intersection_node_indices[unsnapped_rows, unsnapped_groups] = (
            assigned_nodes
        )

    proposed_nodes = np.concatenate(
        (float_nodes.copy(), new_node_coordinates), axis=0
    )
    snapped_node_indices = eligible_snap_nodes[eligible_snap_nodes >= 0]
    if snapped_node_indices.size:
        snapped_node_indices = np.unique(snapped_node_indices)
        snap_offsets = xy[snapped_node_indices] - center
        snap_distances = np.hypot(snap_offsets[:, 0], snap_offsets[:, 1])
        if np.any(snap_distances <= 0.0):
            raise ValueError("a circle-center node cannot be snapped")
        proposed_nodes[snapped_node_indices, :2] = (
            center
            + circle_radius
            * snap_offsets
            / snap_distances[:, None]
        )

    # Insert both chord endpoints into each original perimeter.  Existing
    # snapped vertices appear twice in the six-item work array and are then
    # compacted back to one perimeter occurrence.
    perimeter_nodes = np.full((eligible_count, 6), -1, dtype=np.int64)
    perimeter_positions = np.full(
        (eligible_count, 6), np.inf, dtype=np.float64
    )
    perimeter_valid = np.zeros((eligible_count, 6), dtype=np.bool_)
    perimeter_nodes[:, :4] = eligible_elements
    perimeter_positions[:, :4] = np.arange(4, dtype=np.float64)
    perimeter_valid[:, :4] = (
        np.arange(4, dtype=np.int64)[None, :] < eligible_counts[:, None]
    )
    perimeter_nodes[:, 4:] = intersection_node_indices
    for group_position in (0, 1):
        snapped_group = eligible_snap_nodes[:, group_position] >= 0
        group_edge = eligible_group_edges[:, group_position]
        local_parameter = eligible_group_local_parameters[:, group_position]
        group_position_value = group_edge.astype(np.float64) + local_parameter
        if np.any(snapped_group):
            group_node = intersection_node_indices[:, group_position]
            matches = eligible_elements == group_node[:, None]
            snapped_positions = np.argmax(matches, axis=1)
            group_position_value[snapped_group] = snapped_positions[
                snapped_group
            ]
        perimeter_positions[:, 4 + group_position] = group_position_value
        perimeter_valid[:, 4 + group_position] = True

    perimeter_positions[~perimeter_valid] = np.inf
    perimeter_order = np.argsort(
        perimeter_positions, axis=1, kind="stable"
    )
    sorted_perimeter_nodes = np.take_along_axis(
        perimeter_nodes, perimeter_order, axis=1
    )
    sorted_perimeter_valid = np.take_along_axis(
        perimeter_valid, perimeter_order, axis=1
    )
    keep_perimeter = sorted_perimeter_valid.copy()
    keep_perimeter[:, 1:] &= (
        sorted_perimeter_nodes[:, 1:] != sorted_perimeter_nodes[:, :-1]
    )
    compact_keys = np.where(
        keep_perimeter,
        np.arange(6, dtype=np.int64)[None, :],
        6,
    )
    compact_order = np.argsort(compact_keys, axis=1, kind="stable")
    compact_perimeter = np.take_along_axis(
        sorted_perimeter_nodes, compact_order, axis=1
    )
    perimeter_counts = np.count_nonzero(keep_perimeter, axis=1).astype(
        np.int64
    )
    compact_valid = (
        np.arange(6, dtype=np.int64)[None, :] < perimeter_counts[:, None]
    )

    first_cut_positions = np.argmax(
        compact_valid
        & (compact_perimeter == intersection_node_indices[:, 0, None]),
        axis=1,
    )
    second_cut_positions = np.argmax(
        compact_valid
        & (compact_perimeter == intersection_node_indices[:, 1, None]),
        axis=1,
    )
    perimeter_steps = np.arange(6, dtype=np.int64)[None, :]
    rotated_perimeter_positions = (
        first_cut_positions[:, None] + perimeter_steps
    ) % perimeter_counts[:, None]
    rotated_perimeter = np.take_along_axis(
        compact_perimeter, rotated_perimeter_positions, axis=1
    )
    second_offsets = (
        second_cut_positions - first_cut_positions
    ) % perimeter_counts
    first_path_counts = second_offsets + 1
    second_path_counts = perimeter_counts - second_offsets + 1
    if np.any(
        (first_path_counts < 3)
        | (first_path_counts > 5)
        | (second_path_counts < 3)
        | (second_path_counts > 5)
    ):
        raise ValueError("circle chord does not split an element interior")

    first_paths = rotated_perimeter.copy()
    second_paths = np.full((eligible_count, 5), -1, dtype=np.int64)
    for offset in range(5):
        source_positions = (
            second_offsets + offset
        ) % perimeter_counts
        second_paths[:, offset] = rotated_perimeter[
            np.arange(eligible_count), source_positions
        ]

    child_elements = np.full(
        (eligible_count, 3, 4), -1, dtype=np.int64
    )
    child_counts = np.full(eligible_count, 2, dtype=np.int64)

    def _encode_short_paths(paths, path_counts, child_positions):
        triangle_rows = np.flatnonzero(path_counts == 3)
        if triangle_rows.size:
            triangles = paths[triangle_rows, :3]
            encoded = np.column_stack((triangles, triangles[:, 2]))
            child_elements[
                triangle_rows, child_positions[triangle_rows]
            ] = encoded
        quad_rows = np.flatnonzero(path_counts == 4)
        if quad_rows.size:
            child_elements[
                quad_rows, child_positions[quad_rows]
            ] = paths[quad_rows, :4]

    first_child_positions = np.zeros(eligible_count, dtype=np.int64)
    _encode_short_paths(
        first_paths, first_path_counts, first_child_positions
    )

    # Split a convex five-node side into the best-conditioned Tri3/Quad4 pair.
    # Five fixed ear candidates keep this stage fully vectorized.
    def _encode_pentagons(paths, path_counts, first_slot):
        pentagon_rows = np.flatnonzero(path_counts == 5)
        if pentagon_rows.size == 0:
            return pentagon_rows
        pentagons = paths[pentagon_rows, :5]
        ears = np.arange(5, dtype=np.int64)
        triangle_positions = np.stack(
            (ears, (ears + 1) % 5, (ears + 2) % 5), axis=1
        )
        quad_positions = np.stack(
            (ears, (ears + 2) % 5, (ears + 3) % 5, (ears + 4) % 5),
            axis=1,
        )
        triangle_candidates = pentagons[:, triangle_positions]
        quad_candidates = pentagons[:, quad_positions]
        triangle_points = proposed_nodes[triangle_candidates, :2]
        quad_points = proposed_nodes[quad_candidates, :2]

        def _minimum_corner_sine(points):
            previous = np.roll(points, 1, axis=2) - points
            following = np.roll(points, -1, axis=2) - points
            cross = (
                following[..., 0] * previous[..., 1]
                - following[..., 1] * previous[..., 0]
            )
            denominator = np.hypot(
                following[..., 0], following[..., 1]
            ) * np.hypot(previous[..., 0], previous[..., 1])
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.min(np.abs(cross) / denominator, axis=2)

        candidate_scores = np.minimum(
            _minimum_corner_sine(triangle_points),
            _minimum_corner_sine(quad_points),
        )
        candidate_scores[~np.isfinite(candidate_scores)] = -np.inf
        best_candidates = np.argmax(candidate_scores, axis=1)
        best_scores = candidate_scores[
            np.arange(pentagon_rows.size), best_candidates
        ]
        if np.any(best_scores <= 2048.0 * np.finfo(np.float64).eps):
            raise ValueError("cannot split a five-node side into valid elements")
        chosen_triangles = triangle_candidates[
            np.arange(pentagon_rows.size), best_candidates
        ]
        chosen_quads = quad_candidates[
            np.arange(pentagon_rows.size), best_candidates
        ]
        encoded_triangles = np.column_stack(
            (chosen_triangles, chosen_triangles[:, 2])
        )
        slots = first_slot[pentagon_rows]
        child_elements[pentagon_rows, slots] = encoded_triangles
        child_elements[pentagon_rows, slots + 1] = chosen_quads
        child_counts[pentagon_rows] = 3
        return pentagon_rows

    first_pentagon_rows = _encode_pentagons(
        first_paths, first_path_counts, first_child_positions
    )
    second_child_positions = np.ones(eligible_count, dtype=np.int64)
    second_child_positions[first_pentagon_rows] = 2
    _encode_short_paths(
        second_paths, second_path_counts, second_child_positions
    )
    second_pentagon_rows = _encode_pentagons(
        second_paths, second_path_counts, second_child_positions
    )
    if second_pentagon_rows.size:
        # The other side is necessarily the three-node side and already
        # occupies slot zero, so the pentagon starts at slot one.
        child_counts[second_pentagon_rows] = 3

    child_valid = (
        np.arange(3, dtype=np.int64)[None, :] < child_counts[:, None]
    )
    if np.any(child_elements[child_valid] < 0):
        raise ValueError("failed to construct every imprinted child element")

    flat_children = child_elements[child_valid]
    child_is_triangle = flat_children[:, 2] == flat_children[:, 3]
    sorted_child_triangles = np.sort(flat_children[:, :3], axis=1)
    valid_child_triangles = np.all(
        np.diff(sorted_child_triangles, axis=1) != 0, axis=1
    )
    sorted_child_quads = np.sort(flat_children, axis=1)
    valid_child_quads = np.all(
        np.diff(sorted_child_quads, axis=1) != 0, axis=1
    )
    if np.any(child_is_triangle & ~valid_child_triangles) or np.any(
        ~child_is_triangle & ~valid_child_quads
    ):
        raise ValueError("imprint produced duplicate child node indices")

    child_points = proposed_nodes[flat_children, :2]
    child_next_points = np.roll(child_points, -1, axis=1)
    if np.any(child_is_triangle):
        child_next_points[child_is_triangle, 2] = child_points[
            child_is_triangle, 0
        ]
        child_next_points[child_is_triangle, 3] = child_points[
            child_is_triangle, 3
        ]
    child_edges = child_next_points - child_points
    child_valid_edges = np.ones((flat_children.shape[0], 4), dtype=np.bool_)
    child_valid_edges[child_is_triangle, 3] = False
    child_relative_points = child_points - child_points[:, :1]
    child_relative_next = child_next_points - child_points[:, :1]
    child_cross = (
        child_relative_points[..., 0] * child_relative_next[..., 1]
        - child_relative_points[..., 1] * child_relative_next[..., 0]
    )
    child_signed_areas = 0.5 * np.sum(
        np.where(child_valid_edges, child_cross, 0.0), axis=1
    )
    child_scales = np.max(
        np.where(
            child_valid_edges,
            np.hypot(child_edges[..., 0], child_edges[..., 1]),
            0.0,
        ),
        axis=1,
    )
    child_area_tolerances = (
        4096.0
        * np.finfo(np.float64).eps
        * np.maximum(child_scales, 1.0) ** 2
    )

    proposed_parent_points = proposed_nodes[eligible_elements, :2]
    parent_next_points = np.roll(proposed_parent_points, -1, axis=1)
    parent_previous_points = np.roll(proposed_parent_points, 1, axis=1)
    eligible_triangle_rows = np.flatnonzero(eligible_counts == 3)
    if eligible_triangle_rows.size:
        parent_next_points[eligible_triangle_rows, 2] = (
            proposed_parent_points[eligible_triangle_rows, 0]
        )
        parent_next_points[eligible_triangle_rows, 3] = (
            proposed_parent_points[eligible_triangle_rows, 3]
        )
        parent_previous_points[eligible_triangle_rows, 0] = (
            proposed_parent_points[eligible_triangle_rows, 2]
        )
    parent_relative_points = (
        proposed_parent_points - proposed_parent_points[:, :1]
    )
    parent_relative_next = parent_next_points - proposed_parent_points[:, :1]
    parent_cross = (
        parent_relative_points[..., 0] * parent_relative_next[..., 1]
        - parent_relative_points[..., 1] * parent_relative_next[..., 0]
    )
    parent_valid_edges = (
        np.arange(4, dtype=np.int64)[None, :] < eligible_counts[:, None]
    )
    parent_signed_areas = 0.5 * np.sum(
        np.where(parent_valid_edges, parent_cross, 0.0), axis=1
    )
    original_parent_signs = np.sign(
        np.where(
            eligible_counts == 3,
            triangle_cross[eligible_selection_positions],
            corner_cross[eligible_selection_positions, 0],
        )
    )
    parent_corner_cross = (
        (parent_next_points[..., 0] - proposed_parent_points[..., 0])
        * (
            parent_previous_points[..., 1]
            - proposed_parent_points[..., 1]
        )
        - (parent_next_points[..., 1] - proposed_parent_points[..., 1])
        * (
            parent_previous_points[..., 0]
            - proposed_parent_points[..., 0]
        )
    )
    parent_geometry_tolerances = geometry_tolerances[
        eligible_selection_positions
    ]
    if np.any(
        parent_valid_edges
        & (
            parent_corner_cross * original_parent_signs[:, None]
            <= parent_geometry_tolerances[:, None]
        )
    ) or np.any(
        parent_signed_areas * original_parent_signs
        <= parent_geometry_tolerances
    ):
        raise ValueError(
            "snapping an intersection would invert or fold its parent element"
        )
    parent_signs = original_parent_signs
    child_parent_rows = np.repeat(
        np.arange(eligible_count, dtype=np.int64), child_counts
    )
    if np.any(
        child_signed_areas * parent_signs[child_parent_rows]
        <= child_area_tolerances
    ):
        raise ValueError("imprint produced a degenerate or inverted child")

    child_previous_points = np.roll(child_points, 1, axis=1)
    if np.any(child_is_triangle):
        child_previous_points[child_is_triangle, 0] = child_points[
            child_is_triangle, 2
        ]
    child_corner_cross = (
        child_edges[..., 0]
        * (child_previous_points[..., 1] - child_points[..., 1])
        - child_edges[..., 1]
        * (child_previous_points[..., 0] - child_points[..., 0])
    )
    if np.any(
        child_valid_edges
        & (
            child_corner_cross * parent_signs[child_parent_rows, None]
            <= child_area_tolerances[:, None]
        )
    ):
        raise ValueError("imprint produced a folded child element")

    child_absolute_areas = np.abs(child_signed_areas)
    area_sums = np.zeros(eligible_count, dtype=np.float64)
    np.add.at(area_sums, child_parent_rows, child_absolute_areas)
    parent_absolute_areas = np.abs(parent_signed_areas)
    area_tolerances = (
        8192.0
        * np.finfo(np.float64).eps
        * np.maximum.reduce(
            (
                parent_absolute_areas,
                element_scales[eligible_selection_positions] ** 2,
                np.ones(eligible_count, dtype=np.float64),
            )
        )
    )
    if np.any(
        np.abs(area_sums - parent_absolute_areas) > area_tolerances
    ):
        raise ValueError("imprinted child areas do not cover their parent")

    proposed_elements = elements.astype(np.int64, copy=True)
    proposed_elements[eligible_element_indices] = child_elements[:, 0]
    appended_elements = child_elements[:, 1:].reshape(-1, 4)
    appended_valid = child_valid[:, 1:].reshape(-1)
    proposed_elements = np.concatenate(
        (proposed_elements, appended_elements[appended_valid]), axis=0
    )
    mesh.replace_data(nodes=proposed_nodes, elements=proposed_elements)
    return mesh


def normalize_mesh(
    mesh: Mesh2D,
    tolerance: float,
) -> Mesh2D:


def trim_element(
    mesh: Mesh2D,
    tolerance: float,
):
    print()

__all__ = [
    "get_circle_intersect",
    "get_inner_outer_areas",
    "get_intersect_nodes",
    "get_tri_quad",
    "imprint_circle",
    "to_circle",
]
