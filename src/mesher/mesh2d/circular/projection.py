"""Project mesh boundary chains onto circular target boundaries."""

from typing import NamedTuple

import numpy as np

from ..model import Mesh2D

from .geometry import (
    _minimum_scaled_jacobian,
    _smooth_circle_nodes,
    _validate_generated_strip,
)
from .pattern_segments import (
    _PatternGuideSet,
    _circle_line_intersections,
    _circle_segment_intersections,
    _coerce_pattern_guides,
)


def _find_segment_vertex_hits(
    segment,
    *,
    vertical,
    selected_xy,
    pair_start_positions,
    pair_end_positions,
    chain_lengths,
    segment_tolerance,
):
    """Find boundary vertices touched by one axis-aligned pattern segment.

    Args:
        segment: Two XY endpoints of the pattern segment.
        vertical: Whether the segment uses a fixed X coordinate.
        selected_xy: Coordinates of the ordered source boundary.
        pair_start_positions: Start positions of source-boundary edges.
        pair_end_positions: End positions of source-boundary edges.
        chain_lengths: Length of each source-boundary edge.
        segment_tolerance: Linear tolerance local to the pattern segment.

    Returns:
        A tuple containing the hit vertex positions, fixed-axis coordinate,
        lower varying-axis coordinate, and upper varying-axis coordinate.

    Raises:
        ValueError: If the segment crosses the interior of an edge without
            passing through one of its vertices.
    """
    fixed_axis = 0 if vertical else 1
    varying_axis = 1 - fixed_axis
    fixed = segment[0, fixed_axis]
    lower, upper = np.sort(segment[:, varying_axis])
    vertex_hits: set[int] = set()

    for edge_position, (start_position, end_position) in enumerate(
        zip(pair_start_positions, pair_end_positions)
    ):
        point_a = selected_xy[start_position]
        point_b = selected_xy[end_position]
        fixed_a = point_a[fixed_axis]
        fixed_b = point_b[fixed_axis]
        varying_a = point_a[varying_axis]
        varying_b = point_b[varying_axis]
        a_on_axis = abs(fixed_a - fixed) <= segment_tolerance
        b_on_axis = abs(fixed_b - fixed) <= segment_tolerance

        if a_on_axis and b_on_axis:
            overlap_lower = max(min(varying_a, varying_b), lower)
            overlap_upper = min(max(varying_a, varying_b), upper)
            a_on_segment = (
                lower - segment_tolerance
                <= varying_a
                <= upper + segment_tolerance
            )
            b_on_segment = (
                lower - segment_tolerance
                <= varying_b
                <= upper + segment_tolerance
            )
            if overlap_upper > overlap_lower + segment_tolerance:
                overlap_candidates: set[int] = set()
                if a_on_segment:
                    overlap_candidates.add(int(start_position))
                if b_on_segment:
                    overlap_candidates.add(int(end_position))
                if not overlap_candidates:
                    raise ValueError(
                        "a pattern segment intersects a boundary edge "
                        "without a vertex"
                    )
                vertex_hits.update(overlap_candidates)
                continue
            if a_on_segment:
                vertex_hits.add(int(start_position))
            if b_on_segment:
                vertex_hits.add(int(end_position))
            continue

        fixed_delta = fixed_b - fixed_a
        if abs(fixed_delta) <= segment_tolerance:
            continue
        parameter = (fixed - fixed_a) / fixed_delta
        parameter_tolerance = max(
            64.0 * np.finfo(np.float64).eps,
            segment_tolerance / chain_lengths[edge_position],
        )
        if (
            parameter < -parameter_tolerance
            or parameter > 1.0 + parameter_tolerance
        ):
            continue
        varying_intersection = varying_a + parameter * (
            varying_b - varying_a
        )
        if not (
            lower - segment_tolerance
            <= varying_intersection
            <= upper + segment_tolerance
        ):
            continue

        edge_parameter_tolerance = (
            segment_tolerance / chain_lengths[edge_position]
        )
        if (
            edge_parameter_tolerance
            < parameter
            < 1.0 - edge_parameter_tolerance
        ):
            raise ValueError(
                "a pattern segment intersects a boundary edge without a vertex"
            )
        if parameter <= edge_parameter_tolerance:
            vertex_hits.add(int(start_position))
        else:
            vertex_hits.add(int(end_position))

    return vertex_hits, fixed, lower, upper


def _nearest_circle_root(
    vertex_position,
    *,
    selected_xy,
    vertical,
    varying_roots,
    segment_tolerance,
):
    """Choose the unambiguous circle root nearest one source vertex.

    Args:
        vertex_position: Position of the vertex in the ordered source chain.
        selected_xy: Coordinates of the ordered source boundary.
        vertical: Whether the pattern segment uses a fixed X coordinate.
        varying_roots: Candidate coordinates on the varying segment axis.
        segment_tolerance: Linear tolerance local to the pattern segment.

    Returns:
        A pair containing the root index and its distance from the vertex on
        the varying segment axis.

    Raises:
        ValueError: If two roots are indistinguishably close to the vertex.
    """
    source = selected_xy[vertex_position]
    varying_source = source[1 if vertical else 0]
    root_distances = np.abs(np.asarray(varying_roots) - varying_source)
    nearest = int(np.argmin(root_distances))
    if root_distances.size > 1:
        ordered_distances = np.sort(root_distances)
        if ordered_distances[1] - ordered_distances[0] <= segment_tolerance:
            raise ValueError(
                "a pattern segment has ambiguous circle intersections"
            )
    return nearest, float(root_distances[nearest])


def _lift_pattern_angle(
    target,
    *,
    center,
    angular_direction,
    reference,
    full_turn,
):
    """Lift an ordinary target angle near a directed reference angle.

    Args:
        target: Exact XY target coordinate.
        center: Circle center.
        angular_direction: Positive one for CCW traversal, negative one for CW.
        reference: Preferred unwrapped directed angle.
        full_turn: Angular size of one complete revolution.

    Returns:
        The equivalent directed target angle nearest the reference.
    """
    ordinary_angle = np.arctan2(
        target[1] - center[1],
        target[0] - center[0],
    )
    directed_angle = angular_direction * ordinary_angle
    return directed_angle + full_turn * np.rint(
        (reference - directed_angle) / full_turn
    )


def _validate_circle_targets(
    candidate_xy,
    *,
    center,
    radius,
    tolerance,
    closed,
    angular_direction,
    angular_tolerance,
    full_turn,
    winding_tolerance,
):
    """Validate radius, finiteness, order, and winding of projected nodes.

    Args:
        candidate_xy: Proposed target-circle coordinates.
        center: Circle center.
        radius: Target circle radius.
        tolerance: Linear geometry tolerance.
        closed: Whether the source boundary is a closed loop.
        angular_direction: Positive one for CCW traversal, negative one for CW.
        angular_tolerance: Minimum distinguishable angular separation.
        full_turn: Angular size of one complete revolution.
        winding_tolerance: Allowed closed-loop winding error.

    Raises:
        ValueError: If a coordinate is non-finite, off-circle, out of order,
            or spans an invalid winding.
    """
    if not np.all(np.isfinite(candidate_xy)):
        raise ValueError("projected circle coordinates exceed float64 range")

    target_offsets = candidate_xy - center
    target_distances = np.hypot(
        target_offsets[:, 0],
        target_offsets[:, 1],
    )
    if np.any(
        np.abs(target_distances - radius)
        > max(tolerance, radius * 1.0e-12)
    ):
        raise ValueError("a generated target node is not on the circle")

    target_raw_angles = np.arctan2(
        target_offsets[:, 1],
        target_offsets[:, 0],
    )
    if closed:
        target_steps = angular_direction * np.arctan2(
            np.sin(np.roll(target_raw_angles, -1) - target_raw_angles),
            np.cos(np.roll(target_raw_angles, -1) - target_raw_angles),
        )
        if np.any(target_steps <= angular_tolerance) or abs(
            float(np.sum(target_steps)) - full_turn
        ) > winding_tolerance:
            raise ValueError(
                "circle target nodes must preserve strict angular order"
            )
        return

    target_angles = np.unwrap(target_raw_angles)
    target_angles *= angular_direction
    if np.any(np.diff(target_angles) <= angular_tolerance):
        raise ValueError(
            "circle target nodes must preserve strict angular order"
        )
    if target_angles[-1] - target_angles[0] >= (
        full_turn - angular_tolerance
    ):
        raise ValueError("an open chain cannot span a complete circle")


class _ProjectionInputs(NamedTuple):
    """Normalized state shared by boundary-projection stages.

    Attributes:
        nodes: Original mesh node array.
        elements: Original mesh element array.
        float_nodes: Finite float64 view or copy of all node coordinates.
        center: Target-circle center.
        radius: Target-circle radius.
        indices: Ordered source-boundary node indices.
        pattern_guides: Prepared pattern segments.
        requested_closed: Explicit closure mode or None for inference.
        selected_nodes: Copied source-boundary coordinates.
        selected_xy: Planar view of selected_nodes.
        coordinate_scale: Local magnitude used for numerical tolerances.
        tolerance: Linear geometry tolerance.
        angular_tolerance: Angular geometry tolerance.
    """

    nodes: np.ndarray
    elements: np.ndarray
    float_nodes: np.ndarray
    center: np.ndarray
    radius: float
    indices: np.ndarray
    pattern_guides: _PatternGuideSet
    requested_closed: bool | None
    selected_nodes: np.ndarray
    selected_xy: np.ndarray
    coordinate_scale: float
    tolerance: float
    angular_tolerance: float


class _BoundaryChain(NamedTuple):
    """Resolved source-boundary topology.

    Attributes:
        closed: Whether the source nodes form a closed loop.
        pair_start_positions: Start positions of consecutive source edges.
        pair_end_positions: End positions of consecutive source edges.
        chain_lengths: Length of every source edge.
        matches_element_direction: Whether input traversal matches element
            perimeter direction.
    """

    closed: bool
    pair_start_positions: np.ndarray
    pair_end_positions: np.ndarray
    chain_lengths: np.ndarray
    matches_element_direction: bool


class _PreferredAngles(NamedTuple):
    """Preferred directed angular coordinates for projected source nodes.

    Attributes:
        full_turn: Angular size of one complete revolution.
        direction: Positive one for CCW traversal and negative one for CW.
        winding_tolerance: Allowed accumulated closed-loop angular error.
        values: Strictly increasing preferred directed angles.
    """

    full_turn: float
    direction: float
    winding_tolerance: float
    values: np.ndarray


def _read_projection_inputs(
    mesh,
    center_x,
    center_y,
    radius,
    node_indices,
    guide_segments,
    closed,
):
    """Normalize every input needed by boundary projection.

    Args:
        mesh: Mesh2D whose exposed boundary will be projected.
        center_x: X coordinate of the target-circle center.
        center_y: Y coordinate of the target-circle center.
        radius: Positive target-circle radius.
        node_indices: Ordered source-boundary node indices.
        guide_segments: Optional axis-aligned pattern segments. Non-empty input must
            have shape (L, 2, 2); empty input is normalized to no segments.
        closed: Explicit closure mode or None for topology inference.

    Returns:
        Normalized arrays, scalar values, selected coordinates, and tolerances.

    Raises:
        TypeError: If mesh, closed, or node-index types are invalid.
        ValueError: If mesh arrays, scalars, pattern guide_segments, or coordinates are
            invalid.
        IndexError: If a source node index is out of range.
    """
    if not isinstance(mesh, Mesh2D):
        raise TypeError("mesh must be a Mesh2D instance")

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
    if not np.issubdtype(elements.dtype, np.integer) or np.issubdtype(
        elements.dtype,
        np.bool_,
    ):
        raise ValueError("elements must have an integer dtype")

    try:
        center = np.asarray([center_x, center_y], dtype=np.float64)
        radius = float(radius)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "center_x, center_y, and radius must be real numbers"
        ) from error
    if not np.all(np.isfinite(center)) or not np.isfinite(radius):
        raise ValueError("center_x, center_y, and radius must be finite")
    if radius <= 0.0:
        raise ValueError("radius must be positive")

    indices = np.asarray(node_indices)
    if indices.ndim != 1:
        raise ValueError("node_indices must be a one-dimensional sequence")
    if indices.size < 2:
        raise ValueError("node_indices must contain at least two nodes")
    if not np.issubdtype(indices.dtype, np.integer) or np.issubdtype(
        indices.dtype,
        np.bool_,
    ):
        raise TypeError("node_indices must contain integers")
    if np.any(indices < 0) or np.any(indices >= nodes.shape[0]):
        raise IndexError("node_indices contain an out-of-range node index")
    indices = indices.astype(np.intp, copy=False)
    if np.unique(indices).size != indices.size:
        raise ValueError("node_indices must not contain duplicates")
    if closed is not None and not isinstance(closed, (bool, np.bool_)):
        raise TypeError("closed must be True, False, or None")
    requested_closed = None if closed is None else bool(closed)

    if np.any(elements < 0) or np.any(elements >= nodes.shape[0]):
        raise ValueError("elements contain an out-of-range node index")
    try:
        float_nodes = nodes.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("nodes must be representable as float64") from error
    if not np.all(np.isfinite(float_nodes)):
        raise ValueError("nodes must contain finite float64 coordinates")

    selected_nodes = float_nodes[indices].copy()
    selected_xy = selected_nodes[:, :2]
    scale_values = [1.0, abs(radius), *np.abs(center).tolist()]
    if selected_xy.size:
        scale_values.append(float(np.max(np.abs(selected_xy))))
    coordinate_scale = max(scale_values)
    tolerance = 64.0 * np.finfo(np.float64).eps * coordinate_scale
    pattern_guides = _coerce_pattern_guides(
        guide_segments,
        coordinate_scale=coordinate_scale,
        minimum_tolerance=tolerance,
    )
    angular_tolerance = max(
        64.0 * np.finfo(np.float64).eps,
        tolerance / radius,
    )
    return _ProjectionInputs(
        nodes,
        elements,
        float_nodes,
        center,
        radius,
        indices,
        pattern_guides,
        requested_closed,
        selected_nodes,
        selected_xy,
        coordinate_scale,
        tolerance,
        angular_tolerance,
    )


def _read_boundary_chain(
    elements,
    indices,
    selected_xy,
    requested_closed,
    tolerance,
):
    """Validate source topology and resolve consecutive boundary edges.

    Args:
        elements: Existing mixed Tri3/Quad4 connectivity.
        indices: Ordered source-boundary node indices.
        selected_xy: Planar coordinates of the selected source nodes.
        requested_closed: Explicit closure mode or None for inference.
        tolerance: Linear geometry tolerance.

    Returns:
        Closure mode, consecutive edge positions and lengths, and source
        traversal direction.

    Raises:
        ValueError: If existing topology or the selected boundary is invalid.
    """
    edge_uses: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for element in elements:
        element_ids = [int(value) for value in element]
        if element_ids[2] == element_ids[3]:
            perimeter = element_ids[:3]
            if len(set(perimeter)) != 3:
                raise ValueError(
                    "a padded Tri3 must contain three distinct perimeter nodes"
                )
        else:
            perimeter = element_ids
            if len(set(perimeter)) != 4:
                raise ValueError("a Quad4 must contain four distinct nodes")

        for start, end in zip(perimeter, perimeter[1:] + perimeter[:1]):
            key = (min(start, end), max(start, end))
            occurrences = edge_uses.setdefault(key, [])
            occurrences.append((start, end))
            if len(occurrences) > 2:
                raise ValueError(
                    "mesh contains a non-manifold edge shared by more than two "
                    "elements"
                )

    for occurrences in edge_uses.values():
        if len(occurrences) == 2 and occurrences[0] != occurrences[1][::-1]:
            raise ValueError(
                "elements sharing an edge must traverse it in opposite "
                "directions"
            )

    closing_edge_key = (
        min(int(indices[-1]), int(indices[0])),
        max(int(indices[-1]), int(indices[0])),
    )
    closing_edge_uses = edge_uses.get(closing_edge_key)
    closing_edge_is_exposed = (
        indices.size >= 3
        and closing_edge_uses is not None
        and len(closing_edge_uses) == 1
    )
    is_closed = (
        closing_edge_is_exposed
        if requested_closed is None
        else requested_closed
    )
    if is_closed and indices.size < 3:
        raise ValueError(
            "a closed boundary loop must contain at least three nodes"
        )

    pair_start_positions = np.arange(
        indices.size if is_closed else indices.size - 1,
        dtype=np.intp,
    )
    pair_end_positions = (pair_start_positions + 1) % indices.size
    chain_pairs = np.column_stack(
        (indices[pair_start_positions], indices[pair_end_positions])
    )
    direction_matches = []
    for start, end in chain_pairs:
        edge_key = (min(int(start), int(end)), max(int(start), int(end)))
        directed_uses = edge_uses.get(edge_key)
        if directed_uses is None or len(directed_uses) != 1:
            edge_description = (
                "each boundary-loop edge"
                if is_closed
                else "each consecutive node pair"
            )
            raise ValueError(
                f"{edge_description} must be an exposed boundary edge"
            )
        directed_edge = directed_uses[0]
        direction_matches.append(directed_edge == (int(start), int(end)))
    if len(set(direction_matches)) != 1:
        raise ValueError(
            "node_indices do not follow one consistent boundary direction"
        )

    chain_vectors = (
        selected_xy[pair_end_positions]
        - selected_xy[pair_start_positions]
    )
    chain_lengths = np.hypot(chain_vectors[:, 0], chain_vectors[:, 1])
    if np.any(chain_lengths <= tolerance):
        raise ValueError(
            "the boundary chain must not contain zero-length edges"
        )
    return _BoundaryChain(
        is_closed,
        pair_start_positions,
        pair_end_positions,
        chain_lengths,
        direction_matches[0],
    )


def _preferred_projection_angles(
    center,
    radius,
    selected_xy,
    indices,
    closed,
    tolerance,
    angular_tolerance,
):
    """Project source directions into strictly increasing angular coordinates.

    Args:
        center: Target-circle center.
        radius: Target-circle radius.
        selected_xy: Planar source-boundary coordinates.
        indices: Ordered source-boundary node indices.
        closed: Whether the boundary is a closed loop.
        tolerance: Linear geometry tolerance.
        angular_tolerance: Minimum distinguishable angular separation.

    Returns:
        Full-turn size, traversal direction, winding tolerance, and preferred
        directed angles.

    Raises:
        ValueError: If offsets overflow, a source lies on or across the target
            circle, angular order is not strict, or winding is invalid.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        offsets = selected_xy - center
    overflowed = ~np.all(np.isfinite(offsets), axis=1)
    if np.any(overflowed):
        offsets[overflowed] = (
            selected_xy[overflowed] * 0.5 - center * 0.5
        )
    if not np.all(np.isfinite(offsets)):
        raise ValueError(
            "selected node offsets are not representable as float64"
        )

    offset_scale = np.max(np.abs(offsets), axis=1)
    if np.any(offset_scale == 0.0):
        raise ValueError(
            "a selected node cannot coincide with the circle center"
        )
    scaled_offsets = offsets / offset_scale[:, None]
    scaled_norms = np.hypot(scaled_offsets[:, 0], scaled_offsets[:, 1])
    unit_offsets = scaled_offsets / scaled_norms[:, None]
    with np.errstate(over="ignore", invalid="ignore"):
        distances = offset_scale * scaled_norms

    strictly_inside = distances < radius - tolerance
    strictly_outside = distances > radius + tolerance
    if not (np.all(strictly_inside) or np.all(strictly_outside)):
        raise ValueError(
            "all chain nodes must lie strictly on the same side of the circle"
        )

    full_turn = 2.0 * np.pi
    source_raw_angles = np.arctan2(
        unit_offsets[:, 1],
        unit_offsets[:, 0],
    )
    if closed:
        source_raw_steps = np.arctan2(
            np.sin(np.roll(source_raw_angles, -1) - source_raw_angles),
            np.cos(np.roll(source_raw_angles, -1) - source_raw_angles),
        )
        if np.all(source_raw_steps > angular_tolerance):
            angular_direction = 1.0
        elif np.all(source_raw_steps < -angular_tolerance):
            angular_direction = -1.0
        else:
            raise ValueError(
                "circle target nodes must preserve strict angular order"
            )
        directed_source_steps = angular_direction * source_raw_steps
        winding_tolerance = max(
            indices.size * angular_tolerance,
            512.0 * np.finfo(np.float64).eps,
        )
        if abs(
            float(np.sum(directed_source_steps)) - full_turn
        ) > winding_tolerance:
            raise ValueError(
                "a closed boundary loop must wind exactly once around the "
                "circle center"
            )
        preferred_angles = np.empty(indices.size, dtype=np.float64)
        preferred_angles[0] = angular_direction * source_raw_angles[0]
        preferred_angles[1:] = preferred_angles[0] + np.cumsum(
            directed_source_steps[:-1]
        )
    else:
        unwrapped_source_angles = np.unwrap(source_raw_angles)
        source_angle_steps = np.diff(unwrapped_source_angles)
        if np.all(source_angle_steps > angular_tolerance):
            angular_direction = 1.0
        elif np.all(source_angle_steps < -angular_tolerance):
            angular_direction = -1.0
        else:
            raise ValueError(
                "circle target nodes must preserve strict angular order"
            )
        preferred_angles = angular_direction * unwrapped_source_angles
        winding_tolerance = 0.0
        if preferred_angles[-1] - preferred_angles[0] >= (
            full_turn - angular_tolerance
        ):
            raise ValueError("an open chain cannot span a complete circle")

    return _PreferredAngles(
        full_turn,
        angular_direction,
        winding_tolerance,
        preferred_angles,
    )


def _collect_projection_constraints(
    pattern_guides,
    *,
    radius,
    selected_xy,
    pair_start_positions,
    pair_end_positions,
    chain_lengths,
    center,
    tolerance,
):
    """Resolve active pattern segments into exact target coordinates.

    Args:
        pattern_guides: Prepared finite axis-aligned pattern segments.
        radius: Target-circle radius.
        selected_xy: Planar source-boundary coordinates.
        pair_start_positions: Start positions of source-boundary edges.
        pair_end_positions: End positions of source-boundary edges.
        chain_lengths: Length of every source-boundary edge.
        center: Target-circle center.
        tolerance: Linear projection tolerance.

    Returns:
        A mapping from source-chain positions to exact target-circle points.

    Raises:
        ValueError: If an active segment is malformed, unsupported, ambiguous,
            conflicting, or fails to reach the target circle.
    """
    constrained_targets: dict[int, np.ndarray] = {}
    for segment in pattern_guides:
        segment_tolerance = segment.tolerance

        vertex_hits, fixed, _, _ = _find_segment_vertex_hits(
            segment.coordinates,
            vertical=segment.vertical,
            selected_xy=selected_xy,
            pair_start_positions=pair_start_positions,
            pair_end_positions=pair_end_positions,
            chain_lengths=chain_lengths,
            segment_tolerance=segment_tolerance,
        )
        if not vertex_hits:
            continue

        line_intersections = _circle_line_intersections(
            segment,
            center,
            radius,
        )
        if line_intersections.shape[0] == 0:
            raise ValueError(
                "an active pattern segment does not intersect the target circle"
            )
        segment_intersections = _circle_segment_intersections(
            segment,
            center,
            radius,
        )
        if segment_intersections.shape[0] == 0:
            raise ValueError(
                "an active pattern segment does not reach the target circle"
            )
        varying_roots = segment_intersections[:, segment.varying_axis]

        # One analytic circle intersection can support only one connector from
        # a given pattern segment.  This also collapses every collinear overlap
        # to the boundary vertex closest to the corresponding circle root.
        representatives: dict[int, tuple[float, int]] = {}
        for vertex_position in sorted(vertex_hits):
            nearest, root_distance = _nearest_circle_root(
                vertex_position,
                selected_xy=selected_xy,
                vertical=segment.vertical,
                varying_roots=varying_roots,
                segment_tolerance=segment_tolerance,
            )
            representative = (root_distance, vertex_position)
            previous = representatives.get(nearest)
            if previous is None or representative < previous:
                representatives[nearest] = representative
        vertex_hits = {
            vertex_position
            for _, vertex_position in representatives.values()
        }

        for vertex_position in vertex_hits:
            nearest, _ = _nearest_circle_root(
                vertex_position,
                selected_xy=selected_xy,
                vertical=segment.vertical,
                varying_roots=varying_roots,
                segment_tolerance=segment_tolerance,
            )
            if segment.vertical:
                candidate = np.array(
                    [fixed, varying_roots[nearest]], dtype=np.float64
                )
            else:
                candidate = np.array(
                    [varying_roots[nearest], fixed], dtype=np.float64
                )

            previous = constrained_targets.get(vertex_position)
            if (
                previous is not None
                and np.hypot(*(candidate - previous))
                > max(tolerance, segment_tolerance)
            ):
                raise ValueError("pattern segments impose conflicting constraints")
            constrained_targets[vertex_position] = candidate

    return constrained_targets


def _redistribute_projection_angles(
    preferred_angles,
    constrained_targets,
    *,
    is_closed,
    node_count,
    center,
    angular_direction,
    full_turn,
    angular_tolerance,
    radius,
    tolerance,
    winding_tolerance,
):
    """Reconcile exact pattern anchors with strict circular node ordering.

    Args:
        preferred_angles: Directed source angles before pattern constraints.
        constrained_targets: Exact target coordinates keyed by chain position.
        is_closed: Whether target nodes form a closed loop.
        node_count: Number of target nodes.
        center: Target-circle center.
        angular_direction: Positive one for CCW traversal, negative one for CW.
        full_turn: Angular size of one revolution.
        angular_tolerance: Minimum distinguishable angular separation.
        radius: Target-circle radius.
        tolerance: Linear projection tolerance.
        winding_tolerance: Allowed accumulated closed-loop winding error.

    Returns:
        Finite target-circle coordinates with exact pattern anchors restored.

    Raises:
        ValueError: If constraints cannot preserve strict angular order or the
            resulting targets fail circle and winding validation.
    """
    indices = np.arange(node_count, dtype=np.intp)
    # Project the preferred radial angles onto the strictly ordered angles
    # allowed by the exact pattern anchors.  Subtracting ``position * gap``
    # converts strict ordering into ordinary monotonic ordering; clipping in
    # that space moves only the neighbouring nodes whose order an anchor would
    # otherwise reverse.
    adjusted_angles = preferred_angles.copy()
    if constrained_targets:
        if is_closed:
            first_anchor_position = min(constrained_targets)
            rotated_positions = (
                first_anchor_position + np.arange(indices.size, dtype=np.intp)
            ) % indices.size
            completed_turns = (
                first_anchor_position + np.arange(indices.size, dtype=np.intp)
            ) // indices.size
            rotated_preferred = (
                preferred_angles[rotated_positions]
                + completed_turns.astype(np.float64) * full_turn
            )

            local_anchor_positions = sorted(
                (position - first_anchor_position) % indices.size
                for position in constrained_targets
            )
            local_anchor_values = []
            for local_position in local_anchor_positions:
                original_position = int(rotated_positions[local_position])
                local_anchor_values.append(
                    _lift_pattern_angle(
                        constrained_targets[original_position],
                        center=center,
                        angular_direction=angular_direction,
                        reference=rotated_preferred[local_position],
                        full_turn=full_turn,
                    )
                )

            extended_preferred = np.concatenate(
                (rotated_preferred, [rotated_preferred[0] + full_turn])
            )
            extended_anchor_positions = np.asarray(
                [*local_anchor_positions, indices.size], dtype=np.intp
            )
            extended_anchor_values = np.asarray(
                [*local_anchor_values, local_anchor_values[0] + full_turn],
                dtype=np.float64,
            )
            base_capacities = np.diff(extended_preferred)
            anchor_capacities = np.diff(extended_anchor_values) / np.diff(
                extended_anchor_positions
            )
        else:
            anchor_positions = np.asarray(
                sorted(constrained_targets), dtype=np.intp
            )
            anchor_values = np.asarray(
                [
                    _lift_pattern_angle(
                        constrained_targets[int(position)],
                        center=center,
                        angular_direction=angular_direction,
                        reference=preferred_angles[int(position)],
                        full_turn=full_turn,
                    )
                    for position in anchor_positions
                ],
                dtype=np.float64,
            )
            base_capacities = np.diff(preferred_angles)
            if anchor_positions.size > 1:
                anchor_capacities = np.diff(anchor_values) / np.diff(
                    anchor_positions
                )
            else:
                anchor_capacities = np.empty(0, dtype=np.float64)

        angular_capacity = float(
            np.min(np.concatenate((base_capacities, anchor_capacities)))
        )
        if not np.isfinite(angular_capacity) or (
            angular_capacity <= angular_tolerance
        ):
            raise ValueError(
                "pattern constraints cannot preserve strict angular order"
            )
        redistribution_gap = angular_tolerance + 0.5 * (
            angular_capacity - angular_tolerance
        )

        if is_closed:
            positions = np.arange(indices.size + 1, dtype=np.float64)
            preferred_without_gap = (
                extended_preferred - positions * redistribution_gap
            )
            adjusted_without_gap = preferred_without_gap.copy()
            for interval in range(extended_anchor_positions.size - 1):
                lower_position = int(extended_anchor_positions[interval])
                upper_position = int(extended_anchor_positions[interval + 1])
                lower_value = (
                    extended_anchor_values[interval]
                    - lower_position * redistribution_gap
                )
                upper_value = (
                    extended_anchor_values[interval + 1]
                    - upper_position * redistribution_gap
                )
                adjusted_without_gap[lower_position : upper_position + 1] = (
                    np.clip(
                        preferred_without_gap[
                            lower_position : upper_position + 1
                        ],
                        lower_value,
                        upper_value,
                    )
                )
                adjusted_without_gap[lower_position] = lower_value
                adjusted_without_gap[upper_position] = upper_value

            rotated_adjusted = (
                adjusted_without_gap[:-1]
                + positions[:-1] * redistribution_gap
            )
            adjusted_angles[rotated_positions] = rotated_adjusted
        else:
            positions = np.arange(indices.size, dtype=np.float64)
            preferred_without_gap = (
                preferred_angles - positions * redistribution_gap
            )
            adjusted_without_gap = preferred_without_gap.copy()
            anchor_values_without_gap = (
                anchor_values - anchor_positions * redistribution_gap
            )

            first_position = int(anchor_positions[0])
            adjusted_without_gap[:first_position] = np.minimum(
                preferred_without_gap[:first_position],
                anchor_values_without_gap[0],
            )
            for interval in range(anchor_positions.size - 1):
                lower_position = int(anchor_positions[interval])
                upper_position = int(anchor_positions[interval + 1])
                adjusted_without_gap[lower_position : upper_position + 1] = (
                    np.clip(
                        preferred_without_gap[
                            lower_position : upper_position + 1
                        ],
                        anchor_values_without_gap[interval],
                        anchor_values_without_gap[interval + 1],
                    )
                )
            last_position = int(anchor_positions[-1])
            adjusted_without_gap[last_position + 1 :] = np.maximum(
                preferred_without_gap[last_position + 1 :],
                anchor_values_without_gap[-1],
            )
            adjusted_without_gap[anchor_positions] = anchor_values_without_gap
            adjusted_angles = adjusted_without_gap + positions * redistribution_gap

    ordinary_target_angles = angular_direction * adjusted_angles
    with np.errstate(over="ignore", invalid="ignore"):
        target_xy = center + radius * np.column_stack(
            (np.cos(ordinary_target_angles), np.sin(ordinary_target_angles))
        )
    for vertex_position, target in constrained_targets.items():
        # Preserve the analytic pattern/circle intersection exactly, including
        # its fixed horizontal or vertical coordinate.
        target_xy[vertex_position] = target

    _validate_circle_targets(
        target_xy,
        center=center,
        radius=radius,
        tolerance=tolerance,
        closed=is_closed,
        angular_direction=angular_direction,
        angular_tolerance=angular_tolerance,
        full_turn=full_turn,
        winding_tolerance=winding_tolerance if is_closed else 0.0,
    )

    return target_xy


def _build_projected_strip_elements(
    nodes,
    indices,
    pair_start_positions,
    pair_end_positions,
    chain_matches_element_direction,
    selected_xy,
    target_xy,
    constrained_targets,
    angular_tolerance,
):
    """Build strip connectivity and split pattern-collinear quads.

    Args:
        nodes: Existing mesh node array.
        indices: Ordered source-boundary node indices.
        pair_start_positions: Start positions of source-boundary edges.
        pair_end_positions: End positions of source-boundary edges.
        chain_matches_element_direction: Whether source traversal follows
            existing element winding.
        selected_xy: Planar source-boundary coordinates.
        target_xy: Projected target-circle coordinates.
        constrained_targets: Exact pattern anchors keyed by chain position.
        angular_tolerance: Collinearity tolerance for connector corners.

    Returns:
        New-node start index, appended target-node indices, and mixed
        Tri3/Quad4 strip connectivity.

    Raises:
        ValueError: If both endpoints of one source edge would create
            collinear pattern connectors.
        RuntimeError: If internal anchor-to-edge bookkeeping is inconsistent.
    """
    new_node_start = nodes.shape[0]
    target_indices = np.arange(
        new_node_start,
        new_node_start + indices.size,
        dtype=np.int64,
    )
    source_starts = indices[pair_start_positions]
    source_ends = indices[pair_end_positions]
    target_starts = target_indices[pair_start_positions]
    target_ends = target_indices[pair_end_positions]
    if chain_matches_element_direction:
        new_quads = np.column_stack(
            (
                source_ends,
                source_starts,
                target_starts,
                target_ends,
            )
        )
        quad_source_positions = np.column_stack(
            (pair_end_positions, pair_start_positions)
        )
    else:
        new_quads = np.column_stack(
            (
                source_starts,
                source_ends,
                target_ends,
                target_starts,
            )
        )
        quad_source_positions = np.column_stack(
            (pair_start_positions, pair_end_positions)
        )

    # A pattern connector can be collinear with an incident source edge, either
    # because the pattern overlaps that edge or ends exactly at its vertex.
    # The corresponding four-node polygon has a straight corner, so split it
    # along the opposite diagonal into two conforming Tri3 elements instead of
    # emitting a degenerate Quad4.
    collinear_edge_anchors: dict[int, set[int]] = {}
    for edge_position, source_positions in enumerate(quad_source_positions):
        first_source_position = int(source_positions[0])
        second_source_position = int(source_positions[1])
        for source_position, neighbour_position in (
            (first_source_position, second_source_position),
            (second_source_position, first_source_position),
        ):
            if source_position not in constrained_targets:
                continue
            boundary_vector = (
                selected_xy[neighbour_position] - selected_xy[source_position]
            )
            connector_vector = (
                target_xy[source_position] - selected_xy[source_position]
            )
            boundary_length = np.hypot(*boundary_vector)
            connector_length = np.hypot(*connector_vector)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                boundary_unit = boundary_vector / boundary_length
                connector_unit = connector_vector / connector_length
                normalized_cross = (
                    boundary_unit[0] * connector_unit[1]
                    - boundary_unit[1] * connector_unit[0]
                )
            if abs(normalized_cross) <= angular_tolerance:
                collinear_edge_anchors.setdefault(edge_position, set()).add(
                    source_position
                )

    if collinear_edge_anchors:
        new_element_rows: list[np.ndarray] = []
        for edge_position, quad in enumerate(new_quads):
            anchors = collinear_edge_anchors.get(edge_position)
            if not anchors:
                new_element_rows.append(quad)
                continue

            first_source_position = int(quad_source_positions[edge_position, 0])
            second_source_position = int(quad_source_positions[edge_position, 1])
            first_is_anchor = first_source_position in anchors
            second_is_anchor = second_source_position in anchors
            if first_is_anchor and second_is_anchor:
                raise ValueError(
                    "pattern connectors cannot be collinear at both endpoints "
                    "of one boundary edge"
                )
            if first_is_anchor:
                new_element_rows.extend(
                    (
                        np.array([quad[0], quad[1], quad[2], quad[2]]),
                        np.array([quad[0], quad[2], quad[3], quad[3]]),
                    )
                )
            elif second_is_anchor:
                new_element_rows.extend(
                    (
                        np.array([quad[0], quad[1], quad[3], quad[3]]),
                        np.array([quad[1], quad[2], quad[3], quad[3]]),
                    )
                )
            else:
                raise RuntimeError(
                    "collinear anchor is not an endpoint of its boundary edge"
                )
        new_elements = np.asarray(new_element_rows, dtype=np.int64)
    else:
        new_elements = new_quads

    return new_node_start, target_indices, new_elements


def _to_circle(
    mesh: Mesh2D,
    center_x,
    center_y,
    radius,
    node_indices,
    guide_segments=None,
    closed=None,
) -> Mesh2D:
    """Extend an ordered boundary chain with a mixed Tri3/Quad4 strip.

    One circle node is created for every node in ``node_indices`` and normally
    one quadrilateral is created for every boundary edge.  A quadrilateral with
    a pattern-induced straight corner is split into two padded Tri3 elements.
    By default a chain is treated as closed when its last and first nodes also
    form an exposed boundary edge; pass ``closed=False`` to keep such a chain
    open explicitly.  Closed input lists contain each node once (the first node
    is not repeated).

    ``guide_segments`` contains finite horizontal or vertical pattern segments.  When a
    segment meets a chain vertex, that vertex is connected to the matching
    segment/circle intersection.  If a segment meets multiple vertices,
    including along overlapping boundary edges, the vertex nearest each
    matching circle intersection is used.  Neighbouring circle nodes are
    redistributed locally when necessary so the exact pattern connector does
    not reverse the circle-node order.  Unconstrained circle nodes are then
    smoothed along the circle to improve short angular gaps, but a move is
    accepted only when the incident strip's minimum scaled Jacobian does not
    decrease.  Pattern intersections and open-chain endpoints remain fixed.

    The mesh is mutated only after every geometric and topological validation
    succeeds, and the same :class:`Mesh2D` instance is returned.

    Args:
        mesh: Mesh2D whose exposed boundary is extended.
        center_x: X coordinate of the target-circle center.
        center_y: Y coordinate of the target-circle center.
        radius: Positive target-circle radius.
        node_indices: Ordered source-boundary node indices.
        guide_segments: Optional finite horizontal or vertical pattern segments.
            Non-empty input must have shape (L, 2, 2); empty input is treated
            as no segments.
        closed: True for a closed loop, False for an open chain, or None to
            infer closure from the last-to-first exposed edge.

    Returns:
        The same mesh instance with projected nodes and strip elements added.

    Raises:
        TypeError: If mesh, closed, or node-index types are invalid.
        ValueError: If inputs, source topology, pattern constraints, or
            generated geometry are invalid.
        IndexError: If a selected node index is out of range.

    Note:
        Mutation is atomic. Both replacement arrays are materialized only
        after projection, smoothing, topology, and geometry validation finish.
    """
    projection = _read_projection_inputs(
        mesh,
        center_x,
        center_y,
        radius,
        node_indices,
        guide_segments,
        closed,
    )
    (
        nodes,
        elements,
        float_nodes,
        center,
        radius,
        indices,
        pattern_guides,
        requested_closed,
        selected_nodes,
        selected_xy,
        coordinate_scale,
        tolerance,
        angular_tolerance,
    ) = projection

    boundary = _read_boundary_chain(
        elements,
        indices,
        selected_xy,
        requested_closed,
        tolerance,
    )
    (
        is_closed,
        pair_start_positions,
        pair_end_positions,
        chain_lengths,
        chain_matches_element_direction,
    ) = boundary

    preferred = _preferred_projection_angles(
        center,
        radius,
        selected_xy,
        indices,
        is_closed,
        tolerance,
        angular_tolerance,
    )
    (
        full_turn,
        angular_direction,
        winding_tolerance,
        preferred_angles,
    ) = preferred

    constrained_targets = _collect_projection_constraints(
        pattern_guides,
        radius=radius,
        selected_xy=selected_xy,
        pair_start_positions=pair_start_positions,
        pair_end_positions=pair_end_positions,
        chain_lengths=chain_lengths,
        center=center,
        tolerance=tolerance,
    )

    target_xy = _redistribute_projection_angles(
        preferred_angles,
        constrained_targets,
        is_closed=is_closed,
        node_count=indices.size,
        center=center,
        angular_direction=angular_direction,
        full_turn=full_turn,
        angular_tolerance=angular_tolerance,
        radius=radius,
        tolerance=tolerance,
        winding_tolerance=winding_tolerance,
    )

    (
        new_node_start,
        target_indices,
        new_elements,
    ) = _build_projected_strip_elements(
        nodes,
        indices,
        pair_start_positions,
        pair_end_positions,
        chain_matches_element_direction,
        selected_xy,
        target_xy,
        constrained_targets,
        angular_tolerance,
    )

    proposed_xy = np.vstack((float_nodes[:, :2], target_xy))
    _validate_generated_strip(
        proposed_xy,
        new_elements,
        tolerance,
        angular_tolerance,
    )
    baseline_quality = _minimum_scaled_jacobian(
        proposed_xy,
        new_elements,
        tolerance,
    )

    fixed_target_indices = target_indices[
        np.asarray(sorted(constrained_targets), dtype=np.intp)
    ]
    smoothed_xy = _smooth_circle_nodes(
        proposed_xy,
        new_elements,
        center,
        radius,
        target_indices,
        fixed_target_indices,
        closed=is_closed,
        tolerance=tolerance,
        angular_tolerance=angular_tolerance,
    )
    try:
        _validate_generated_strip(
            smoothed_xy,
            new_elements,
            tolerance,
            angular_tolerance,
        )
        smoothed_target_xy = smoothed_xy[target_indices]
        _validate_circle_targets(
            smoothed_target_xy,
            center=center,
            radius=radius,
            tolerance=tolerance,
            closed=is_closed,
            angular_direction=angular_direction,
            angular_tolerance=angular_tolerance,
            full_turn=full_turn,
            winding_tolerance=winding_tolerance if is_closed else 0.0,
        )
        smoothed_quality = _minimum_scaled_jacobian(
            smoothed_xy,
            new_elements,
            tolerance,
        )
        quality_epsilon = 256.0 * np.finfo(np.float64).eps * max(
            1.0,
            abs(baseline_quality) if baseline_quality is not None else 1.0,
        )
        if (
            baseline_quality is None
            or smoothed_quality is None
            or smoothed_quality < baseline_quality - quality_epsilon
        ):
            raise ValueError("circle smoothing lowered strip quality")
    except ValueError:
        # The unsmoothed strip was already proven valid.  A non-local crossing
        # or a numerical edge case therefore falls back atomically instead of
        # turning a previously valid _to_circle operation into an error.
        smoothed_xy = proposed_xy
    target_xy = smoothed_xy[target_indices]

    target_nodes = selected_nodes
    target_nodes[:, :2] = target_xy
    combined_nodes = np.concatenate((float_nodes, target_nodes), axis=0)

    element_dtype = elements.dtype
    maximum_new_index = new_node_start + indices.size - 1
    if maximum_new_index > np.iinfo(element_dtype).max:
        element_dtype = np.dtype(np.int64)
    combined_elements = np.concatenate(
        (
            elements.astype(element_dtype, copy=False),
            new_elements.astype(element_dtype, copy=False),
        ),
        axis=0,
    )

    # Both arrays are fully materialized before either attribute is assigned.
    mesh.nodes = combined_nodes
    mesh.elements = combined_elements
    return mesh
