"""Geometry and quality helpers for circular mesh construction."""

import numpy as np


def _minimum_scaled_jacobian(xy, element_rows, tolerance):
    """Return the worst scaled Jacobian across mixed Tri3/Quad4 rows.

    Args:
        xy: Planar node coordinates.
        element_rows: Four-column connectivity, with Tri3 rows padded by
            repeating their third node.
        tolerance: Minimum valid edge length.

    Returns:
        The minimum scaled Jacobian, positive infinity for no rows, or None
        when any supplied element is invalid.
    """
    element_rows = np.asarray(element_rows, dtype=np.intp)
    minima = []
    is_triangle = element_rows[:, 2] == element_rows[:, 3]
    for rows, factor in (
        (element_rows[is_triangle, :3], 2.0 / np.sqrt(3.0)),
        (element_rows[~is_triangle], 1.0),
    ):
        if rows.size == 0:
            continue
        points = xy[rows]
        following = np.concatenate((points[:, 1:], points[:, :1]), axis=1)
        preceding = np.concatenate((points[:, -1:], points[:, :-1]), axis=1)
        forward = following - points
        backward = preceding - points
        forward_lengths = np.hypot(forward[:, :, 0], forward[:, :, 1])
        backward_lengths = np.hypot(backward[:, :, 0], backward[:, :, 1])
        if np.any(forward_lengths <= tolerance):
            return None
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            scaled = (
                forward[:, :, 0] * backward[:, :, 1]
                - forward[:, :, 1] * backward[:, :, 0]
            ) / (forward_lengths * backward_lengths)
        scaled *= factor
        if not np.all(np.isfinite(scaled)) or np.any(scaled <= 0.0):
            return None
        minima.append(float(np.min(scaled)))
    return min(minima, default=np.inf)


def _strip_edges(elements):
    """Collect one directed representative of each topological strip edge.

    Args:
        elements: Mixed Tri3/Quad4 connectivity rows.

    Returns:
        A mapping from sorted endpoint pairs to one directed occurrence.
    """
    edges: dict[tuple[int, int], tuple[int, int]] = {}
    for element in np.asarray(elements):
        perimeter = element[:3] if element[2] == element[3] else element
        for start, end in zip(perimeter, np.roll(perimeter, -1)):
            start = int(start)
            end = int(end)
            key = (min(start, end), max(start, end))
            edges.setdefault(key, (start, end))
    return edges


def _orientation(start, end, point):
    """Return the signed twice-area of an oriented point-edge triangle.

    Args:
        start: Start coordinate of the directed edge.
        end: End coordinate of the directed edge.
        point: Coordinate whose side of the edge is measured.

    Returns:
        A positive value to the left of the edge, negative to the right, and
        zero for collinear coordinates.
    """
    edge = end - start
    offset = point - start
    return edge[0] * offset[1] - edge[1] * offset[0]


def _segments_intersect_xy(xy, first_edge, second_edge, tolerance):
    """Return whether two planar edges intersect within a tolerance.

    Args:
        xy: Planar node coordinates.
        first_edge: Node-index pair for the first segment.
        second_edge: Node-index pair for the second segment.
        tolerance: Linear distance tolerance.

    Returns:
        True when the closed segments touch, overlap, or cross.
    """
    first_start, first_end = xy[list(first_edge)]
    second_start, second_end = xy[list(second_edge)]
    if (
        max(min(first_start[0], first_end[0]), min(second_start[0], second_end[0]))
        > min(max(first_start[0], first_end[0]), max(second_start[0], second_end[0]))
        + tolerance
        or max(
            min(first_start[1], first_end[1]),
            min(second_start[1], second_end[1]),
        )
        > min(
            max(first_start[1], first_end[1]),
            max(second_start[1], second_end[1]),
        )
        + tolerance
    ):
        return False

    first_vector = first_end - first_start
    second_vector = second_end - second_start
    length_scale = max(
        np.hypot(*first_vector),
        np.hypot(*second_vector),
        tolerance,
    )
    area_tolerance = tolerance * length_scale

    orientations = (
        _orientation(first_start, first_end, second_start),
        _orientation(first_start, first_end, second_end),
        _orientation(second_start, second_end, first_start),
        _orientation(second_start, second_end, first_end),
    )
    signs = tuple(
        1 if value > area_tolerance else -1 if value < -area_tolerance else 0
        for value in orientations
    )
    return signs[0] * signs[1] <= 0 and signs[2] * signs[3] <= 0


def _validate_generated_strip(
    proposed_xy,
    new_elements,
    tolerance,
    angular_tolerance,
):
    """Validate geometry and non-local edge intersections for a new strip.

    Args:
        proposed_xy: Planar coordinates including any newly proposed nodes.
        new_elements: Mixed Tri3/Quad4 connectivity for the proposed strip.
        tolerance: Minimum valid edge length and intersection tolerance.
        angular_tolerance: Minimum positive normalized corner orientation.

    Raises:
        ValueError: If an element is degenerate, clockwise, or crossed by a
            non-neighbouring strip edge.
    """
    is_triangle = new_elements[:, 2] == new_elements[:, 3]
    if np.any(is_triangle):
        triangle_points = proposed_xy[new_elements[is_triangle, :3]]
        triangle_edges = np.roll(triangle_points, -1, axis=1) - triangle_points
        triangle_lengths = np.hypot(
            triangle_edges[:, :, 0], triangle_edges[:, :, 1]
        )
        if np.any(triangle_lengths <= tolerance):
            raise ValueError("a generated triangle has a zero-length edge")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            first_triangle_edge = (
                triangle_edges[:, 0] / triangle_lengths[:, 0, None]
            )
            closing_triangle_edge = (
                -triangle_edges[:, 2] / triangle_lengths[:, 2, None]
            )
            triangle_orientation = (
                first_triangle_edge[:, 0] * closing_triangle_edge[:, 1]
                - first_triangle_edge[:, 1] * closing_triangle_edge[:, 0]
            )
        if not np.all(np.isfinite(triangle_orientation)) or np.any(
            triangle_orientation <= angular_tolerance
        ):
            raise ValueError(
                "generated triangles must be non-degenerate and counter-clockwise"
            )

    if np.any(~is_triangle):
        quad_points = proposed_xy[new_elements[~is_triangle]]
        forward_edges = np.roll(quad_points, -1, axis=1) - quad_points
        backward_edges = np.roll(quad_points, 1, axis=1) - quad_points
        forward_lengths = np.hypot(
            forward_edges[:, :, 0], forward_edges[:, :, 1]
        )
        backward_lengths = np.hypot(
            backward_edges[:, :, 0], backward_edges[:, :, 1]
        )
        if np.any(forward_lengths <= tolerance):
            raise ValueError("a generated quadrilateral has a zero-length edge")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            forward_unit = forward_edges / forward_lengths[:, :, None]
            backward_unit = backward_edges / backward_lengths[:, :, None]
            corner_orientation = (
                forward_unit[:, :, 0] * backward_unit[:, :, 1]
                - forward_unit[:, :, 1] * backward_unit[:, :, 0]
            )
        if not np.all(np.isfinite(corner_orientation)) or np.any(
            corner_orientation <= angular_tolerance
        ):
            raise ValueError(
                "generated quadrilaterals must be non-degenerate and "
                "counter-clockwise"
            )

    edge_items = list(_strip_edges(new_elements).items())
    for first_position, (first_key, first_edge) in enumerate(edge_items):
        for second_key, second_edge in edge_items[first_position + 1 :]:
            if set(first_key).intersection(second_key):
                continue
            if _segments_intersect_xy(
                proposed_xy,
                first_edge,
                second_edge,
                tolerance,
            ):
                raise ValueError(
                    "non-neighbouring edges in the generated strip cross"
                )


def _smooth_circle_nodes(
    proposed_xy,
    new_elements,
    center,
    radius,
    circle_node_indices,
    fixed_node_indices,
    *,
    closed,
    tolerance,
    angular_tolerance,
):
    """Improve circular spacing without lowering incident element quality.

    Nodes move only along the target circle.  Pattern anchors and open-chain
    endpoints remain fixed.  Invalid candidates are ignored, so this helper
    always has the original coordinates as a safe fallback.

    Args:
        proposed_xy: Planar coordinates including the target-circle nodes.
        new_elements: Connectivity of the generated strip.
        center: Target-circle center.
        radius: Target-circle radius.
        circle_node_indices: Indices of nodes eligible to move on the circle.
        fixed_node_indices: Circle-node indices that must remain unchanged.
        closed: Whether the target nodes form a closed loop.
        tolerance: Minimum valid edge length.
        angular_tolerance: Smallest meaningful angular move.

    Returns:
        A copied coordinate array containing every accepted smoothing move.
    """
    smoothed_xy = np.array(proposed_xy, dtype=np.float64, copy=True)
    circle_node_indices = np.asarray(circle_node_indices, dtype=np.int64)
    if circle_node_indices.size < 3:
        return smoothed_xy

    full_turn = 2.0 * np.pi
    raw_angles = np.mod(
        np.arctan2(
            smoothed_xy[circle_node_indices, 1] - center[1],
            smoothed_xy[circle_node_indices, 0] - center[0],
        ),
        full_turn,
    )
    if closed:
        order = np.argsort(raw_angles, kind="stable")
    else:
        directed_steps = np.arctan2(
            np.sin(np.diff(raw_angles)),
            np.cos(np.diff(raw_angles)),
        )
        order = np.arange(circle_node_indices.size, dtype=np.intp)
        if np.all(directed_steps < 0.0):
            order = order[::-1]

    ordered_nodes = circle_node_indices[order]
    ordered_raw_angles = raw_angles[order]
    angles = np.empty_like(ordered_raw_angles)
    angles[0] = ordered_raw_angles[0]
    if angles.size > 1:
        directed_gaps = np.mod(np.diff(ordered_raw_angles), full_turn)
        angles[1:] = angles[0] + np.cumsum(directed_gaps)

    fixed_nodes = {int(value) for value in np.asarray(fixed_node_indices).flat}
    if not closed:
        fixed_nodes.update((int(ordered_nodes[0]), int(ordered_nodes[-1])))
    fixed = np.asarray(
        [int(node) in fixed_nodes for node in ordered_nodes],
        dtype=bool,
    )
    if np.all(fixed):
        return smoothed_xy

    incident_rows = [
        np.flatnonzero(np.any(new_elements == node, axis=1))
        for node in ordered_nodes
    ]
    quality_epsilon = 256.0 * np.finfo(np.float64).eps
    gap_epsilon = max(angular_tolerance, 64.0 * np.finfo(np.float64).eps)
    relaxation_values = 2.0 ** -np.arange(13, dtype=np.float64)

    for sweep in range(50):
        accepted_in_sweep = False
        maximum_move = 0.0
        positions = (
            range(angles.size)
            if sweep % 2 == 0
            else range(angles.size - 1, -1, -1)
        )
        for position in positions:
            if fixed[position]:
                continue

            if position == 0:
                previous_angle = angles[-1] - full_turn
            else:
                previous_angle = angles[position - 1]
            if position == angles.size - 1:
                next_angle = angles[0] + full_turn
            else:
                next_angle = angles[position + 1]

            preceding_gap = angles[position] - previous_angle
            following_gap = next_angle - angles[position]
            midpoint_move = 0.5 * (following_gap - preceding_gap)
            if abs(midpoint_move) <= angular_tolerance:
                continue

            rows = incident_rows[position]
            current_quality = _minimum_scaled_jacobian(
                smoothed_xy,
                new_elements[rows],
                tolerance,
            )
            if current_quality is None:
                continue
            current_gap = min(preceding_gap, following_gap)
            node = int(ordered_nodes[position])
            original_xy = smoothed_xy[node].copy()

            for relaxation in relaxation_values:
                move = float(relaxation * midpoint_move)
                candidate_angle = angles[position] + move
                candidate_gap = min(
                    candidate_angle - previous_angle,
                    next_angle - candidate_angle,
                )
                if candidate_gap <= current_gap + gap_epsilon:
                    break

                with np.errstate(over="ignore", invalid="ignore"):
                    candidate_xy = center + radius * np.array(
                        [np.cos(candidate_angle), np.sin(candidate_angle)],
                        dtype=np.float64,
                    )
                if not np.all(np.isfinite(candidate_xy)):
                    continue
                smoothed_xy[node] = candidate_xy
                candidate_quality = _minimum_scaled_jacobian(
                    smoothed_xy,
                    new_elements[rows],
                    tolerance,
                )
                allowed_loss = quality_epsilon * max(1.0, abs(current_quality))
                if (
                    candidate_quality is not None
                    and candidate_quality >= current_quality - allowed_loss
                ):
                    angles[position] = candidate_angle
                    accepted_in_sweep = True
                    maximum_move = max(maximum_move, abs(move))
                    break
                smoothed_xy[node] = original_xy
            else:
                smoothed_xy[node] = original_xy

        if not accepted_in_sweep or maximum_move <= angular_tolerance:
            break

    return smoothed_xy
