"""Generate circular pattern rings and arcs with exact line constraints."""

from typing import NamedTuple

import numpy as np

from ..mesh import Mesh2D


def _add_pattern_anchor(
    anchor_records,
    point,
    fixed_axis,
    fixed_value,
    *,
    center,
    radius,
    linear_tolerance,
    fit_tolerance,
):
    """Add or merge one exact pattern-line intersection anchor.

    Args:
        anchor_records: Mutable collection of existing anchor records.
        point: Candidate XY intersection coordinate.
        fixed_axis: Coordinate axis fixed by the contributing pattern line.
        fixed_value: Exact coordinate on the fixed axis.
        center: Pattern-circle center.
        radius: Pattern-circle radius.
        linear_tolerance: Distance used to identify duplicate anchors.
        fit_tolerance: Maximum radial error allowed after merging constraints.

    Raises:
        ValueError: If the candidate ambiguously matches multiple anchors,
            conflicts with an existing fixed coordinate, or cannot satisfy the
            target circle after constraint merging.
    """
    point = np.asarray(point, dtype=np.float64)
    matches = [
        position
        for position, record in enumerate(anchor_records)
        if np.hypot(*(point - record["point"])) <= linear_tolerance
    ]
    if len(matches) > 1:
        raise ValueError("pattern guide_segments produce ambiguous circle anchors")
    if not matches:
        anchor_records.append(
            {
                "point": point.copy(),
                "constraints": {int(fixed_axis): float(fixed_value)},
            }
        )
        return

    record = anchor_records[matches[0]]
    constraints = record["constraints"]
    previous = constraints.get(int(fixed_axis))
    if previous is not None and previous != float(fixed_value):
        raise ValueError("pattern guide_segments produce conflicting circle anchors")
    constraints[int(fixed_axis)] = float(fixed_value)
    merged = np.asarray(record["point"], dtype=np.float64).copy()
    for axis, value in constraints.items():
        merged[int(axis)] = value
    radial_distance = float(np.hypot(*(merged - center)))
    if abs(radial_distance - radius) > fit_tolerance:
        raise ValueError("pattern line anchors cannot lie on one circle node")
    record["point"] = merged


def _subdivisions_for_gap(
    gap,
    *,
    radius,
    target_edge_size,
    angular_step,
    maximum_node_count,
):
    """Return the subdivisions needed to respect one arc-length bound.

    Args:
        gap: Angular size of the arc in radians.
        radius: Pattern-circle radius.
        target_edge_size: Maximum allowed arc length.
        angular_step: Element-size bound expressed as an angle.
        maximum_node_count: Safety limit on generated node count.

    Returns:
        A positive subdivision count for the arc.

    Raises:
        ValueError: If the request is non-finite or exceeds the safety limit.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        arc_length = gap * radius
    requested = (
        arc_length / target_edge_size
        if np.isfinite(arc_length)
        else gap / angular_step
    )
    if not np.isfinite(requested) or requested > maximum_node_count:
        raise ValueError("target_edge_size would require too many circle nodes")
    return max(1, int(np.ceil(requested)))


class _PatternCircleInputs(NamedTuple):
    """Normalized inputs for pattern-circle generation.

    Attributes:
        nodes: Original mesh node array.
        center: Pattern-circle center.
        radius: Pattern-circle radius.
        target_edge_size: Maximum requested arc length.
        pattern_lines: Normalized finite axis-aligned pattern segments.
    """

    nodes: np.ndarray
    center: np.ndarray
    radius: float
    target_edge_size: float
    pattern_lines: np.ndarray


def _read_pattern_circle_inputs(
    mesh,
    center_x,
    center_y,
    radius,
    target_edge_size,
    guide_segments,
):
    """Normalize pattern-circle mesh, scalar, and line inputs.

    Args:
        mesh: Mesh2D whose coordinate dimension controls the output shape.
        center_x: X coordinate of the pattern-circle center.
        center_y: Y coordinate of the pattern-circle center.
        radius: Positive pattern-circle radius.
        target_edge_size: Positive maximum arc length.
        guide_segments: Optional finite axis-aligned pattern segments.

    Returns:
        Validated node array, center, scalar values, and pattern guide_segments.

    Raises:
        TypeError: If mesh is not a Mesh2D instance.
        ValueError: If nodes, scalars, or pattern guide_segments are invalid.
    """
    if not isinstance(mesh, Mesh2D):
        raise TypeError("mesh must be a Mesh2D instance")

    nodes = np.asarray(mesh.nodes)
    if nodes.ndim != 2 or nodes.shape[1] not in (2, 3):
        raise ValueError("nodes must have shape (N, 2) or (N, 3)")
    if (
        not np.issubdtype(nodes.dtype, np.number)
        or np.issubdtype(nodes.dtype, np.bool_)
        or np.issubdtype(nodes.dtype, np.complexfloating)
    ):
        raise ValueError("nodes must have a real numeric dtype")
    try:
        float_nodes = nodes.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("nodes must be representable as float64") from error
    if not np.all(np.isfinite(float_nodes)):
        raise ValueError("nodes must contain finite float64 coordinates")

    try:
        center = np.asarray([center_x, center_y], dtype=np.float64)
        radius = float(radius)
        target_edge_size = float(target_edge_size)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "center_x, center_y, radius, and target_edge_size must be real numbers"
        ) from error
    if not np.all(np.isfinite(center)) or not np.all(
        np.isfinite([radius, target_edge_size])
    ):
        raise ValueError(
            "center_x, center_y, radius, and target_edge_size must be finite"
        )
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if target_edge_size <= 0.0:
        raise ValueError("target_edge_size must be positive")

    if guide_segments is None:
        pattern_lines = np.empty((0, 2, 2), dtype=np.float64)
    else:
        try:
            pattern_lines = np.asarray(guide_segments, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("guide_segments must have shape (L, 2, 2)") from error
        if pattern_lines.size == 0:
            if pattern_lines.shape not in ((0,), (0, 2, 2)):
                raise ValueError("guide_segments must have shape (L, 2, 2)")
            pattern_lines = np.empty((0, 2, 2), dtype=np.float64)
        elif pattern_lines.ndim != 3 or pattern_lines.shape[1:] != (2, 2):
            raise ValueError("guide_segments must have shape (L, 2, 2)")
    if not np.all(np.isfinite(pattern_lines)):
        raise ValueError("guide_segments must contain finite coordinates")
    return _PatternCircleInputs(
        nodes,
        center,
        radius,
        target_edge_size,
        pattern_lines,
    )


def _collect_pattern_circle_anchors(
    pattern_lines,
    *,
    center,
    radius,
    coordinate_scale,
    linear_tolerance,
    fit_tolerance,
):
    """Collect and merge exact pattern-line intersections on a circle.

    Args:
        pattern_lines: Normalized finite axis-aligned segments.
        center: Pattern-circle center.
        radius: Pattern-circle radius.
        coordinate_scale: Local circle coordinate magnitude.
        linear_tolerance: Tolerance for duplicate anchor coordinates.
        fit_tolerance: Maximum radial error for merged anchors.

    Returns:
        Constraint-preserving anchor records in deterministic input order.

    Raises:
        ValueError: If a segment is malformed or its anchors are ambiguous,
            conflicting, or not representable on one circle node.
    """
    # Each record retains the exact fixed-axis constraints that produced the
    # anchor.  A vertical and a horizontal segment may legitimately meet at
    # the same circle point; combining their two exact coordinates preserves
    # both constraints with a single node.
    anchor_records: list[dict[str, object]] = []

    for segment in pattern_lines:
        segment_scale = max(
            coordinate_scale,
            float(np.max(np.abs(segment))),
        )
        segment_tolerance = max(
            linear_tolerance,
            64.0 * np.finfo(np.float64).eps * segment_scale,
        )
        delta = segment[1] - segment[0]
        vertical = (
            abs(delta[0]) <= segment_tolerance
            and abs(delta[1]) > segment_tolerance
        )
        horizontal = (
            abs(delta[1]) <= segment_tolerance
            and abs(delta[0]) > segment_tolerance
        )
        if vertical == horizontal:
            raise ValueError(
                "each pattern segment must be non-zero and horizontal or vertical"
            )

        fixed_axis = 0 if vertical else 1
        varying_axis = 1 - fixed_axis
        fixed_value = float(segment[0, fixed_axis])
        varying_lower = float(np.min(segment[:, varying_axis]))
        varying_upper = float(np.max(segment[:, varying_axis]))
        with np.errstate(over="ignore", invalid="ignore"):
            fixed_offset = fixed_value - center[fixed_axis]
        if not np.isfinite(fixed_offset):
            # Opposite extreme finite coordinates can overflow on subtraction;
            # their separation necessarily places the line outside this finite
            # local circle.
            continue
        fixed_distance = abs(float(fixed_offset))
        if fixed_distance > radius + segment_tolerance:
            continue

        normalized_distance = min(fixed_distance / radius, 1.0)
        root = radius * np.sqrt(
            max(
                0.0,
                (1.0 - normalized_distance) * (1.0 + normalized_distance),
            )
        )
        varying_values = [float(center[varying_axis] - root)]
        if root > segment_tolerance:
            varying_values.append(float(center[varying_axis] + root))

        for varying_value in varying_values:
            if not (
                varying_lower - segment_tolerance
                <= varying_value
                <= varying_upper + segment_tolerance
            ):
                continue
            point = np.empty(2, dtype=np.float64)
            point[fixed_axis] = fixed_value
            point[varying_axis] = varying_value
            if not np.all(np.isfinite(point)):
                raise ValueError("pattern circle coordinates exceed float64 range")
            _add_pattern_anchor(
                anchor_records,
                point,
                fixed_axis,
                fixed_value,
                center=center,
                radius=radius,
                linear_tolerance=linear_tolerance,
                fit_tolerance=fit_tolerance,
            )

    return anchor_records


def _sample_pattern_circle(
    anchor_records,
    *,
    center,
    radius,
    target_edge_size,
    angular_tolerance,
    full_turn,
):
    """Sample a circle uniformly between any exact anchor coordinates.

    Args:
        anchor_records: Exact pattern anchor records.
        center: Pattern-circle center.
        radius: Pattern-circle radius.
        target_edge_size: Maximum arc length.
        angular_tolerance: Minimum distinguishable angular gap.
        full_turn: Angular size of one revolution.

    Returns:
        Counter-clockwise planar pattern-circle coordinates.

    Raises:
        ValueError: If the requested spacing requires too many nodes or
            anchors occupy duplicate angular positions.
    """
    angular_step = target_edge_size / radius
    if np.isnan(angular_step) or angular_step <= 0.0:
        raise ValueError("target_edge_size is too small for the requested radius")
    maximum_node_count = min(int(np.iinfo(np.intp).max), 10_000_000)

    if not anchor_records:
        node_count = max(
            3,
            _subdivisions_for_gap(
                full_turn,
                radius=radius,
                target_edge_size=target_edge_size,
                angular_step=angular_step,
                maximum_node_count=maximum_node_count,
            ),
        )
        angles = np.arange(node_count, dtype=np.float64) * (
            full_turn / node_count
        )
        with np.errstate(over="ignore", invalid="ignore"):
            circle_xy = center + radius * np.column_stack(
                (np.cos(angles), np.sin(angles))
            )
    else:
        anchor_points = np.asarray(
            [record["point"] for record in anchor_records],
            dtype=np.float64,
        )
        anchor_angles = np.mod(
            np.arctan2(
                anchor_points[:, 1] - center[1],
                anchor_points[:, 0] - center[0],
            ),
            full_turn,
        )
        order = np.lexsort(
            (anchor_points[:, 1], anchor_points[:, 0], anchor_angles)
        )
        anchor_points = anchor_points[order]
        anchor_angles = anchor_angles[order]
        if anchor_angles.size == 1:
            gaps = np.asarray([full_turn], dtype=np.float64)
        else:
            gaps = np.mod(
                np.roll(anchor_angles, -1) - anchor_angles,
                full_turn,
            )
        if np.any(gaps <= angular_tolerance):
            raise ValueError("pattern guide_segments produce duplicate angular anchors")

        subdivisions = np.asarray(
            [
                _subdivisions_for_gap(
                    float(gap),
                    radius=radius,
                    target_edge_size=target_edge_size,
                    angular_step=angular_step,
                    maximum_node_count=maximum_node_count,
                )
                for gap in gaps
            ],
            dtype=np.int64,
        )
        node_count = sum(int(value) for value in subdivisions)
        while node_count < 3:
            largest = int(np.argmax(gaps / subdivisions))
            subdivisions[largest] += 1
            node_count += 1
        if node_count > maximum_node_count:
            raise ValueError("target_edge_size would require too many circle nodes")

        generated_points = []
        for anchor_point, start_angle, gap, division_count in zip(
            anchor_points,
            anchor_angles,
            gaps,
            subdivisions,
        ):
            generated_points.append(anchor_point.copy())
            for division in range(1, int(division_count)):
                angle = start_angle + gap * division / int(division_count)
                with np.errstate(over="ignore", invalid="ignore"):
                    point = center + radius * np.array(
                        [np.cos(angle), np.sin(angle)],
                        dtype=np.float64,
                    )
                generated_points.append(point)
        circle_xy = np.asarray(generated_points, dtype=np.float64)

    return circle_xy


def _validate_pattern_circle(
    circle_xy,
    *,
    center,
    radius,
    target_edge_size,
    fit_tolerance,
    angular_tolerance,
    full_turn,
):
    """Validate finiteness, radial fit, uniqueness, and sampled spacing.

    Args:
        circle_xy: Sampled planar pattern-circle coordinates.
        center: Pattern-circle center.
        radius: Pattern-circle radius.
        target_edge_size: Maximum requested arc length.
        fit_tolerance: Allowed radial and spacing error.
        angular_tolerance: Minimum distinguishable angular gap.
        full_turn: Angular size of one revolution.

    Raises:
        ValueError: If the sample is too small, non-finite, off-circle,
            duplicated, or spaced too coarsely.
    """
    angular_step = target_edge_size / radius
    if circle_xy.shape[0] < 3 or not np.all(np.isfinite(circle_xy)):
        raise ValueError("pattern circle must contain at least three finite nodes")
    offsets = circle_xy - center
    distances = np.hypot(offsets[:, 0], offsets[:, 1])
    if np.any(np.abs(distances - radius) > fit_tolerance):
        raise ValueError("generated pattern nodes do not lie on the target circle")
    generated_angles = np.mod(
        np.arctan2(offsets[:, 1], offsets[:, 0]),
        full_turn,
    )
    sorted_angles = np.sort(generated_angles)
    generated_gaps = np.diff(
        np.concatenate((sorted_angles, sorted_angles[:1] + full_turn))
    )
    if np.any(generated_gaps <= angular_tolerance):
        raise ValueError("target_edge_size produces indistinguishable circle nodes")
    spacing_tolerance = fit_tolerance / radius
    if np.any(generated_gaps > angular_step + spacing_tolerance):
        raise ValueError("generated pattern-circle spacing exceeds target_edge_size")


def _sample_pattern_arc(
    anchor_records,
    *,
    arc_endpoints,
    center,
    radius,
    target_edge_size,
    angular_tolerance,
    full_turn,
):
    """Sample one counter-clockwise endpoint-bounded circular arc."""
    endpoint_offsets = arc_endpoints - center
    endpoint_angles = np.arctan2(
        endpoint_offsets[:, 1],
        endpoint_offsets[:, 0],
    )
    start_angle = float(endpoint_angles[0])
    span = float(np.mod(endpoint_angles[1] - start_angle, full_turn))
    if span <= angular_tolerance or span >= full_turn - angular_tolerance:
        raise ValueError(
            "pattern arc endpoints must determine a non-empty open arc"
        )

    interior_anchors = []
    for record in anchor_records:
        point = np.asarray(record["point"], dtype=np.float64)
        angle = float(
            np.arctan2(point[1] - center[1], point[0] - center[0])
        )
        position = float(np.mod(angle - start_angle, full_turn))
        if (
            position <= angular_tolerance
            or abs(position - span) <= angular_tolerance
        ):
            continue
        if position >= span - angular_tolerance:
            continue
        interior_anchors.append((position, point.copy()))
    interior_anchors.sort(
        key=lambda item: (item[0], item[1][0], item[1][1])
    )
    if len(interior_anchors) > 1:
        anchor_positions = np.asarray(
            [item[0] for item in interior_anchors],
            dtype=np.float64,
        )
        if np.any(np.diff(anchor_positions) <= angular_tolerance):
            raise ValueError(
                "pattern guide_segments produce duplicate angular anchors"
            )

    knot_positions = np.asarray(
        [0.0, *[item[0] for item in interior_anchors], span],
        dtype=np.float64,
    )
    knot_points = [
        np.asarray(arc_endpoints[0], dtype=np.float64).copy(),
        *[item[1] for item in interior_anchors],
        np.asarray(arc_endpoints[1], dtype=np.float64).copy(),
    ]
    gaps = np.diff(knot_positions)
    angular_step = target_edge_size / radius
    if np.isnan(angular_step) or angular_step <= 0.0:
        raise ValueError("target_edge_size is too small for the requested radius")
    maximum_node_count = min(int(np.iinfo(np.intp).max), 10_000_000)
    subdivisions = np.asarray(
        [
            _subdivisions_for_gap(
                float(gap),
                radius=radius,
                target_edge_size=target_edge_size,
                angular_step=angular_step,
                maximum_node_count=maximum_node_count,
            )
            for gap in gaps
        ],
        dtype=np.int64,
    )
    node_count = 1 + sum(int(value) for value in subdivisions)
    if node_count > maximum_node_count:
        raise ValueError("target_edge_size would require too many circle nodes")

    generated_points = [knot_points[0]]
    for interval, division_count in enumerate(subdivisions):
        lower = float(knot_positions[interval])
        gap = float(gaps[interval])
        for division in range(1, int(division_count)):
            angle = start_angle + gap * division / int(division_count) + lower
            with np.errstate(over="ignore", invalid="ignore"):
                point = center + radius * np.asarray(
                    [np.cos(angle), np.sin(angle)],
                    dtype=np.float64,
                )
            generated_points.append(point)
        generated_points.append(knot_points[interval + 1])
    return np.asarray(generated_points, dtype=np.float64)


def _validate_pattern_arc(
    arc_xy,
    *,
    arc_endpoints,
    center,
    radius,
    target_edge_size,
    fit_tolerance,
    angular_tolerance,
    full_turn,
):
    """Validate radial fit, ordering, endpoints, and spacing of an open arc."""
    if arc_xy.shape[0] < 2 or not np.all(np.isfinite(arc_xy)):
        raise ValueError("pattern arc must contain at least two finite nodes")
    if not np.allclose(
        arc_xy[[0, -1]],
        arc_endpoints,
        rtol=0.0,
        atol=fit_tolerance,
    ):
        raise ValueError("generated pattern-arc endpoints were not preserved")

    offsets = arc_xy - center
    distances = np.hypot(offsets[:, 0], offsets[:, 1])
    if np.any(np.abs(distances - radius) > fit_tolerance):
        raise ValueError("generated pattern nodes do not lie on the target circle")
    start_offset = arc_endpoints[0] - center
    end_offset = arc_endpoints[1] - center
    start_angle = float(np.arctan2(start_offset[1], start_offset[0]))
    end_angle = float(np.arctan2(end_offset[1], end_offset[0]))
    span = float(np.mod(end_angle - start_angle, full_turn))
    node_angles = np.arctan2(offsets[:, 1], offsets[:, 0])
    positions = np.mod(node_angles - start_angle, full_turn)
    positions[0] = 0.0
    positions[-1] = span
    gaps = np.diff(positions)
    if np.any(gaps <= angular_tolerance):
        raise ValueError("target_edge_size produces indistinguishable arc nodes")
    spacing_tolerance = fit_tolerance / radius
    if np.any(gaps > target_edge_size / radius + spacing_tolerance):
        raise ValueError("generated pattern-arc spacing exceeds target_edge_size")


def _generate_pattern_circle_nodes(
    mesh: Mesh2D,
    center_x,
    center_y,
    radius,
    target_edge_size,
    guide_segments=None,
    arc_endpoints=None,
) -> np.ndarray:
    """Return a CCW ring or endpoint-bounded arc with bounded spacing.

    Finite horizontal and vertical pattern segments contribute exact circle
    intersection anchors.  The arcs between those anchors are subdivided
    independently, which preserves the constrained coordinates without
    creating the very short edges caused by inserting anchors into an already
    uniform ring.  The first node is not repeated at the end.

    Args:
        mesh: Mesh2D whose coordinate dimension determines the result shape.
        center_x: X coordinate of the pattern-circle center.
        center_y: Y coordinate of the pattern-circle center.
        radius: Positive pattern-circle radius.
        target_edge_size: Positive maximum arc length between adjacent nodes.
        guide_segments: Optional finite horizontal or vertical pattern segments with
            shape (L, 2, 2).
        arc_endpoints: Optional two planar points on the pattern circle. When
            provided, generate only the counter-clockwise open arc from the
            first point to the second.

    Returns:
        Float64 circle coordinates with shape (N, 2) or (N, 3). Generated Z
        coordinates are zero and the first node is not repeated.

    Raises:
        TypeError: If mesh is not a Mesh2D instance.
        ValueError: If inputs, pattern constraints, requested spacing, or
            generated circle geometry are invalid.
    """
    inputs = _read_pattern_circle_inputs(
        mesh,
        center_x,
        center_y,
        radius,
        target_edge_size,
        guide_segments,
    )
    nodes, center, radius, target_edge_size, pattern_lines = inputs

    # Irrelevant far-away pattern segments must not coarsen the tolerances of
    # this local circle.  Each segment receives its own scale below.
    scale_values = [abs(radius), *np.abs(center).tolist()]
    coordinate_scale = max(scale_values)
    linear_tolerance = max(
        64.0 * np.finfo(np.float64).eps * coordinate_scale,
        256.0 * np.finfo(np.float64).eps * radius,
    )
    fit_tolerance = max(linear_tolerance, 1.0e-9 * radius)
    angular_tolerance = max(
        256.0 * np.finfo(np.float64).eps,
        linear_tolerance / radius,
    )
    full_turn = 2.0 * np.pi

    anchor_records = _collect_pattern_circle_anchors(
        pattern_lines,
        center=center,
        radius=radius,
        coordinate_scale=coordinate_scale,
        linear_tolerance=linear_tolerance,
        fit_tolerance=fit_tolerance,
    )

    if arc_endpoints is None:
        circle_xy = _sample_pattern_circle(
            anchor_records,
            center=center,
            radius=radius,
            target_edge_size=target_edge_size,
            angular_tolerance=angular_tolerance,
            full_turn=full_turn,
        )

        _validate_pattern_circle(
            circle_xy,
            center=center,
            radius=radius,
            target_edge_size=target_edge_size,
            fit_tolerance=fit_tolerance,
            angular_tolerance=angular_tolerance,
            full_turn=full_turn,
        )
    else:
        try:
            arc_endpoints = np.asarray(arc_endpoints, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("arc_endpoints must have shape (2, 2)") from error
        if arc_endpoints.shape != (2, 2) or not np.all(
            np.isfinite(arc_endpoints)
        ):
            raise ValueError("arc_endpoints must have shape (2, 2)")
        endpoint_offsets = arc_endpoints - center
        endpoint_radii = np.hypot(
            endpoint_offsets[:, 0],
            endpoint_offsets[:, 1],
        )
        if np.any(np.abs(endpoint_radii - radius) > fit_tolerance):
            raise ValueError("arc_endpoints must lie on the pattern circle")
        circle_xy = _sample_pattern_arc(
            anchor_records,
            arc_endpoints=arc_endpoints,
            center=center,
            radius=radius,
            target_edge_size=target_edge_size,
            angular_tolerance=angular_tolerance,
            full_turn=full_turn,
        )
        _validate_pattern_arc(
            circle_xy,
            arc_endpoints=arc_endpoints,
            center=center,
            radius=radius,
            target_edge_size=target_edge_size,
            fit_tolerance=fit_tolerance,
            angular_tolerance=angular_tolerance,
            full_turn=full_turn,
        )

    if nodes.shape[1] == 2:
        return circle_xy.astype(np.float64, copy=False)
    return np.column_stack(
        (circle_xy, np.zeros(circle_xy.shape[0], dtype=np.float64))
    )
