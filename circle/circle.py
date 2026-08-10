"""Circle-based element selection helpers."""

import numpy as np

from mesh import Mesh
from viewer import view_mesh

def _search_circle(
    mesh: Mesh,
    x: float,
    y: float,
    radius: float,
    type: str = "ALL",
    tolerance: float = 0.01,
) -> np.ndarray:
    """Return the indices of elements selected by a circle in the XY plane.

    ``ALL`` selects an element only when all its nodes are inside the circle;
    ``PART`` selects it when at least one node is inside.  Nodes on the circle,
    including the supplied tolerance, count as inside.
    """
    nodes = np.asarray(mesh.nodes)
    elements = np.asarray(mesh.elements)

    if nodes.ndim != 2 or nodes.shape[1] not in (2, 3):
        raise ValueError("nodes must have shape (N, 2) or (N, 3)")
    if not np.issubdtype(nodes.dtype, np.number) or np.issubdtype(
        nodes.dtype, np.complexfloating
    ):
        raise ValueError("nodes must have a real numeric dtype")
    if elements.ndim != 2 or elements.shape[1] != 4:
        raise ValueError("elements must have shape (M, 4)")
    if not np.issubdtype(elements.dtype, np.integer) or np.issubdtype(
        elements.dtype, np.bool_
    ):
        raise ValueError("elements must have an integer dtype")
    if type not in {"ALL", "PART"}:
        raise ValueError("type must be either 'ALL' or 'PART'")

    center_x = float(x)
    center_y = float(y)
    radius = float(radius)
    tolerance = float(tolerance)
    if not np.all(np.isfinite([center_x, center_y, radius, tolerance])):
        raise ValueError("x, y, radius, and tolerance must be finite")
    if radius < 0.0:
        raise ValueError("radius must be non-negative")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")

    node_count = nodes.shape[0]
    if np.any(elements < 0) or np.any(elements >= node_count):
        raise ValueError("elements contain an out-of-range node index")

    xy = nodes[:, :2].astype(np.float64, copy=False)
    offset = xy - np.array([center_x, center_y], dtype=np.float64)
    distances = np.hypot(offset[:, 0], offset[:, 1])
    effective_radius = radius + tolerance
    node_inside = distances <= effective_radius
    element_nodes_inside = node_inside[elements]

    if type == "ALL":
        selected = np.all(element_nodes_inside, axis=1)
    else:
        selected = np.any(element_nodes_inside, axis=1)

    return np.flatnonzero(selected).astype(np.int64, copy=False)


def _delete_element(mesh: Mesh, indices) -> Mesh:
    """Remove the selected element rows from ``mesh`` in place.

    Node coordinates are left untouched; removing unused nodes would require
    remapping every node index in the remaining elements.
    """
    if not isinstance(mesh, Mesh):
        raise TypeError("mesh must be a Mesh instance")

    elements = np.asarray(mesh.elements)
    if elements.ndim != 2 or elements.shape[1] != 4:
        raise ValueError("elements must have shape (M, 4)")
    if not np.issubdtype(elements.dtype, np.integer) or np.issubdtype(
        elements.dtype, np.bool_
    ):
        raise ValueError("elements must have an integer dtype")

    delete_indices = np.asarray(indices)
    if delete_indices.ndim != 1:
        raise ValueError("indices must be a one-dimensional sequence")
    if delete_indices.size:
        if not np.issubdtype(delete_indices.dtype, np.integer) or np.issubdtype(
            delete_indices.dtype, np.bool_
        ):
            raise TypeError("indices must contain integers")
        if np.any(delete_indices < 0) or np.any(
            delete_indices >= elements.shape[0]
        ):
            raise IndexError("indices contain an element index that is out of range")
        delete_indices = delete_indices.astype(np.int64, copy=False)
    else:
        delete_indices = np.empty(0, dtype=np.int64)

    if delete_indices.size == 0:
        return mesh

    mesh.elements = np.delete(elements, delete_indices, axis=0)
    return mesh


def _delete(mesh: Mesh, indices) -> Mesh:
    """Backward-compatible name for :func:`_delete_element`."""
    return _delete_element(mesh, indices)


def _clear_node(mesh: Mesh) -> np.ndarray:
    """Remove unreferenced nodes and return their old-to-new index mapping.

    The mesh is updated in place.  The returned one-dimensional array has one
    entry for every node in the original mesh, so retained node indices can be
    remapped with ``new_indices = index_map[old_indices]``.  Entries for nodes
    removed from the mesh are ``-1``.
    """
    if not isinstance(mesh, Mesh):
        raise TypeError("mesh must be a Mesh instance")

    nodes = np.asarray(mesh.nodes)
    elements = np.asarray(mesh.elements)

    if nodes.ndim != 2 or nodes.shape[1] not in (2, 3):
        raise ValueError("nodes must have shape (N, 2) or (N, 3)")
    if not np.issubdtype(nodes.dtype, np.number) or np.issubdtype(
        nodes.dtype, np.complexfloating
    ):
        raise ValueError("nodes must have a real numeric dtype")
    if elements.ndim != 2 or elements.shape[1] != 4:
        raise ValueError("elements must have shape (M, 4)")
    if not np.issubdtype(elements.dtype, np.integer) or np.issubdtype(
        elements.dtype, np.bool_
    ):
        raise ValueError("elements must have an integer dtype")

    node_count = nodes.shape[0]
    if np.any(elements < 0) or np.any(elements >= node_count):
        raise ValueError("elements contain an out-of-range node index")

    referenced_nodes = np.unique(elements)
    old_to_new = np.full(node_count, -1, dtype=np.intp)
    old_to_new[referenced_nodes] = np.arange(referenced_nodes.size, dtype=np.intp)

    if referenced_nodes.size == node_count:
        return old_to_new

    compact_elements = old_to_new[elements].astype(elements.dtype, copy=False)
    mesh.nodes = nodes[referenced_nodes]
    mesh.elements = compact_elements
    return old_to_new


def _get_boundary(mesh: Mesh, indices=None) -> list[np.ndarray]:
    """Return every ordered boundary loop as node indices.

    An element edge belongs to the boundary when no other element shares that
    edge within the selected elements.  ``indices=None`` searches all elements;
    otherwise, only the elements at the supplied indices are considered.  Each
    returned array contains one closed loop in traversal order; closure is
    implicit, so the first node is not repeated at the end.  The element winding
    is preserved (for counter-clockwise elements, outer loops are
    counter-clockwise and hole loops are clockwise).

    Padded triangles of the form ``[n0, n1, n2, n2]`` are supported alongside
    quadrilaterals.  Results use ``int64`` node indices and are ordered
    deterministically by their lexicographically smallest cyclic rotation.
    """
    if not isinstance(mesh, Mesh):
        raise TypeError("mesh must be a Mesh instance")

    nodes = np.asarray(mesh.nodes)
    elements = np.asarray(mesh.elements)

    if nodes.ndim != 2 or nodes.shape[1] not in (2, 3):
        raise ValueError("nodes must have shape (N, 2) or (N, 3)")
    if not np.issubdtype(nodes.dtype, np.number) or np.issubdtype(
        nodes.dtype, np.complexfloating
    ):
        raise ValueError("nodes must have a real numeric dtype")
    if elements.ndim != 2 or elements.shape[1] != 4:
        raise ValueError("elements must have shape (M, 4)")
    if not np.issubdtype(elements.dtype, np.integer) or np.issubdtype(
        elements.dtype, np.bool_
    ):
        raise ValueError("elements must have an integer dtype")

    if indices is not None:
        element_indices = np.asarray(indices)
        if element_indices.ndim != 1:
            raise ValueError("indices must be a one-dimensional sequence")
        if element_indices.size:
            if not np.issubdtype(
                element_indices.dtype, np.integer
            ) or np.issubdtype(element_indices.dtype, np.bool_):
                raise TypeError("indices must contain integers")
            if np.any(element_indices < 0) or np.any(
                element_indices >= elements.shape[0]
            ):
                raise IndexError(
                    "indices contain an element index that is out of range"
                )
            if np.unique(element_indices).size != element_indices.size:
                raise ValueError("indices must not contain duplicates")
            element_indices = element_indices.astype(np.int64, copy=False)
            elements = elements[element_indices]
        else:
            elements = elements[:0]

    node_count = nodes.shape[0]
    if np.any(elements < 0) or np.any(elements >= node_count):
        raise ValueError("elements contain an out-of-range node index")
    if elements.shape[0] == 0:
        return []

    # A triangle is represented by repeating its third node in the fourth
    # column.  Any other repeated node makes the element perimeter ambiguous.
    is_triangle = elements[:, 2] == elements[:, 3]
    sorted_triangle_nodes = np.sort(elements[:, :3], axis=1)
    triangle_is_valid = np.all(np.diff(sorted_triangle_nodes, axis=1) != 0, axis=1)
    sorted_quad_nodes = np.sort(elements, axis=1)
    quad_is_valid = np.all(np.diff(sorted_quad_nodes, axis=1) != 0, axis=1)
    if np.any(is_triangle & ~triangle_is_valid) or np.any(
        ~is_triangle & ~quad_is_valid
    ):
        raise ValueError(
            "each element must be a Quad4 with four distinct nodes or a "
            "Tri3 encoded as [n0, n1, n2, n2]"
        )

    # Build directed half-edges in each element's perimeter order.  The
    # repeated edge in a padded triangle is a self-edge and is omitted.
    edge_starts = elements
    edge_ends = np.roll(elements, -1, axis=1)
    valid_edge = edge_starts != edge_ends
    half_edges = np.column_stack(
        (edge_starts[valid_edge], edge_ends[valid_edge])
    ).astype(np.int64, copy=False)

    half_edge_count = half_edges.shape[0]
    half_edge_ids = np.full(elements.shape, -1, dtype=np.int64)
    half_edge_ids[valid_edge] = np.arange(half_edge_count, dtype=np.int64)

    # Find the next non-self half-edge in each element.  There are at most four
    # slots, so four vectorized passes also cover padded triangles.
    next_ids = np.full(elements.shape, -1, dtype=np.int64)
    for offset in range(1, elements.shape[1] + 1):
        candidate = np.roll(half_edge_ids, -offset, axis=1)
        use_candidate = (half_edge_ids >= 0) & (next_ids < 0) & (candidate >= 0)
        next_ids[use_candidate] = candidate[use_candidate]
    next_half_edge = next_ids[valid_edge]

    # Bucket half-edges by their undirected endpoints.  One occurrence is a
    # boundary edge, two opposite occurrences form an interior pair, and more
    # than two occurrences are non-manifold.
    undirected_edges = np.sort(half_edges, axis=1)
    _, inverse, edge_use_count = np.unique(
        undirected_edges,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    if np.any(edge_use_count > 2):
        raise ValueError(
            "mesh contains a non-manifold edge shared by more than two elements"
        )

    order = np.argsort(inverse, kind="stable")
    group_start = np.cumsum(edge_use_count) - edge_use_count
    interior_groups = np.flatnonzero(edge_use_count == 2)
    first_interior = order[group_start[interior_groups]]
    second_interior = order[group_start[interior_groups] + 1]

    if first_interior.size and np.any(
        (half_edges[first_interior, 0] != half_edges[second_interior, 1])
        | (half_edges[first_interior, 1] != half_edges[second_interior, 0])
    ):
        raise ValueError(
            "elements sharing an edge must traverse it in opposite directions"
        )

    twin = np.full(half_edge_count, -1, dtype=np.int64)
    twin[first_interior] = second_interior
    twin[second_interior] = first_interior

    boundary_half_edges = np.flatnonzero(edge_use_count[inverse] == 1)
    if boundary_half_edges.size == 0:
        return []

    # At the end node of each boundary half-edge, walk around the incident
    # element fan until the next boundary half-edge is reached.  Tracing by
    # half-edge rather than only by node keeps point-touching loops separate.
    boundary_successor = next_half_edge[boundary_half_edges].copy()
    for _ in range(half_edge_count):
        crosses_interior = twin[boundary_successor] >= 0
        if not np.any(crosses_interior):
            break
        crossed = twin[boundary_successor[crosses_interior]]
        boundary_successor[crosses_interior] = next_half_edge[crossed]
    else:
        raise ValueError("boundary topology contains a non-terminating element fan")

    is_boundary = np.zeros(half_edge_count, dtype=bool)
    is_boundary[boundary_half_edges] = True
    if (
        np.any(~is_boundary[boundary_successor])
        or np.any(
            half_edges[boundary_half_edges, 1]
            != half_edges[boundary_successor, 0]
        )
        or np.unique(boundary_successor).size != boundary_half_edges.size
    ):
        raise ValueError("boundary edges do not form closed, non-branching loops")

    successor = np.full(half_edge_count, -1, dtype=np.int64)
    successor[boundary_half_edges] = boundary_successor

    unvisited = set(boundary_half_edges.tolist())
    boundaries: list[np.ndarray] = []
    while unvisited:
        first = min(
            unvisited,
            key=lambda index: tuple(half_edges[index]),
        )
        current = first
        loop_half_edges = []

        while current in unvisited:
            loop_half_edges.append(current)
            unvisited.remove(current)
            current = int(successor[current])

        if current != first:
            raise ValueError("boundary edges do not form closed loops")

        loop_half_edges_array = np.asarray(loop_half_edges, dtype=np.int64)
        loop_nodes = half_edges[loop_half_edges_array, 0]

        # Rotate without reversing so winding is preserved.  Comparing the
        # first directed edge also disambiguates a repeated minimum node in a
        # point-touching topology.
        start_position = min(
            range(loop_nodes.size),
            key=lambda position: (
                int(loop_nodes[position]),
                int(loop_nodes[(position + 1) % loop_nodes.size]),
            ),
        )
        loop_nodes = np.roll(loop_nodes, -start_position).astype(
            np.int64, copy=False
        )
        boundaries.append(loop_nodes)

    boundaries.sort(key=lambda boundary: tuple(boundary.tolist()))
    return boundaries


def _to_circle(
    mesh: Mesh,
    center_x,
    center_y,
    radius,
    node_indices,
    lines=None,
    closed=None,
) -> Mesh:
    """Extend an ordered boundary chain to a circle with Quad4 elements.

    One circle node is created for every node in ``node_indices`` and one
    quadrilateral is created for every boundary edge.  By default a chain is
    treated as closed when its last and first nodes also form an exposed
    boundary edge; pass ``closed=False`` to keep such a chain open explicitly.
    Closed input lists contain each node once (the first node is not repeated).

    ``lines`` contains finite horizontal or vertical pattern segments.  When a
    segment meets a chain vertex, that vertex is connected to the matching
    segment/circle intersection.  Neighbouring circle nodes are redistributed
    locally when necessary so the exact pattern connector does not reverse the
    circle-node order.

    The mesh is mutated only after every geometric and topological validation
    succeeds, and the same :class:`Mesh` instance is returned.
    """
    if not isinstance(mesh, Mesh):
        raise TypeError("mesh must be a Mesh instance")

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
        elements.dtype, np.bool_
    ):
        raise ValueError("elements must have an integer dtype")

    try:
        center = np.asarray([center_x, center_y], dtype=np.float64)
        radius = float(radius)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("center_x, center_y, and radius must be real numbers") from error
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
        indices.dtype, np.bool_
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

    if lines is None:
        pattern_lines = np.empty((0, 2, 2), dtype=np.float64)
    else:
        try:
            pattern_lines = np.asarray(lines, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("lines must have shape (L, 2, 2)") from error
        if pattern_lines.size == 0:
            pattern_lines = np.empty((0, 2, 2), dtype=np.float64)
        elif pattern_lines.ndim != 3 or pattern_lines.shape[1:] != (2, 2):
            raise ValueError("lines must have shape (L, 2, 2)")
    if not np.all(np.isfinite(pattern_lines)):
        raise ValueError("lines must contain finite coordinates")

    selected_nodes = float_nodes[indices].copy()
    selected_xy = selected_nodes[:, :2]
    scale_values = [1.0, abs(radius), *np.abs(center).tolist()]
    if selected_xy.size:
        scale_values.append(float(np.max(np.abs(selected_xy))))
    coordinate_scale = max(scale_values)
    tolerance = 64.0 * np.finfo(np.float64).eps * coordinate_scale
    angular_tolerance = max(
        64.0 * np.finfo(np.float64).eps,
        tolerance / radius,
    )

    # Validate the repository's mixed Quad4 / padded-Tri3 connectivity contract
    # while collecting directed occurrences of every topological edge.
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
                    "mesh contains a non-manifold edge shared by more than two elements"
                )

    for occurrences in edge_uses.values():
        if len(occurrences) == 2 and occurrences[0] != occurrences[1][::-1]:
            raise ValueError(
                "elements sharing an edge must traverse it in opposite directions"
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
        raise ValueError("a closed boundary loop must contain at least three nodes")

    pair_start_positions = np.arange(
        indices.size if is_closed else indices.size - 1,
        dtype=np.intp,
    )
    pair_end_positions = (pair_start_positions + 1) % indices.size
    chain_pairs = np.column_stack(
        (indices[pair_start_positions], indices[pair_end_positions])
    )
    chain_direction_matches = []
    for start, end in chain_pairs:
        edge_key = (min(int(start), int(end)), max(int(start), int(end)))
        directed_uses = edge_uses.get(edge_key)
        if directed_uses is None or len(directed_uses) != 1:
            edge_description = "each boundary-loop edge" if is_closed else (
                "each consecutive node pair"
            )
            raise ValueError(
                f"{edge_description} must be an exposed boundary edge"
            )
        directed_edge = directed_uses[0]
        chain_direction_matches.append(directed_edge == (int(start), int(end)))
    if len(set(chain_direction_matches)) != 1:
        raise ValueError("node_indices do not follow one consistent boundary direction")
    chain_matches_element_direction = chain_direction_matches[0]

    chain_vectors = (
        selected_xy[pair_end_positions] - selected_xy[pair_start_positions]
    )
    chain_lengths = np.hypot(chain_vectors[:, 0], chain_vectors[:, 1])
    if np.any(chain_lengths <= tolerance):
        raise ValueError("the boundary chain must not contain zero-length edges")

    # Compute stable unit vectors from the circle center.  Scaling each offset
    # before hypot avoids overflow and underflow for extreme finite inputs.
    with np.errstate(over="ignore", invalid="ignore"):
        offsets = selected_xy - center
    overflowed = ~np.all(np.isfinite(offsets), axis=1)
    if np.any(overflowed):
        offsets[overflowed] = selected_xy[overflowed] * 0.5 - center * 0.5
    if not np.all(np.isfinite(offsets)):
        raise ValueError("selected node offsets are not representable as float64")

    offset_scale = np.max(np.abs(offsets), axis=1)
    if np.any(offset_scale == 0.0):
        raise ValueError("a selected node cannot coincide with the circle center")
    scaled_offsets = offsets / offset_scale[:, None]
    scaled_norms = np.hypot(scaled_offsets[:, 0], scaled_offsets[:, 1])
    unit_offsets = scaled_offsets / scaled_norms[:, None]
    with np.errstate(over="ignore", invalid="ignore"):
        distances = offset_scale * scaled_norms

    strictly_inside = distances < radius - tolerance
    strictly_outside = distances > radius + tolerance
    if np.all(strictly_inside):
        pass
    elif np.all(strictly_outside):
        pass
    else:
        raise ValueError(
            "all chain nodes must lie strictly on the same side of the circle"
        )

    with np.errstate(over="ignore", invalid="ignore"):
        target_xy = center + radius * unit_offsets
    if not np.all(np.isfinite(target_xy)):
        raise ValueError("projected circle coordinates exceed float64 range")

    # Represent angular traversal as one strictly increasing scalar coordinate,
    # regardless of whether the supplied boundary runs clockwise or
    # counter-clockwise.  Pattern constraints are solved in this coordinate
    # and converted back to ordinary XY angles afterwards.
    full_turn = 2.0 * np.pi
    source_raw_angles = np.arctan2(unit_offsets[:, 1], unit_offsets[:, 0])
    if is_closed:
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
        if abs(float(np.sum(directed_source_steps)) - full_turn) > winding_tolerance:
            raise ValueError(
                "a closed boundary loop must wind exactly once around the circle center"
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
        if preferred_angles[-1] - preferred_angles[0] >= (
            full_turn - angular_tolerance
        ):
            raise ValueError("an open chain cannot span a complete circle")

    def segment_vertex_hits(segment, vertical, segment_tolerance):
        """Return chain vertex positions hit by one axis-aligned segment."""
        fixed_axis = 0 if vertical else 1
        varying_axis = 1 - fixed_axis
        # The segment has already been validated as axis-aligned, so either
        # endpoint supplies the fixed coordinate without a potentially
        # overflowing endpoint sum.
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
                if overlap_upper > overlap_lower + segment_tolerance:
                    raise ValueError(
                        "a pattern segment must not overlap a boundary edge"
                    )
                if lower - segment_tolerance <= varying_a <= upper + segment_tolerance:
                    vertex_hits.add(int(start_position))
                if lower - segment_tolerance <= varying_b <= upper + segment_tolerance:
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
            if parameter < -parameter_tolerance or parameter > 1.0 + parameter_tolerance:
                continue
            varying_intersection = varying_a + parameter * (varying_b - varying_a)
            if not (
                lower - segment_tolerance
                <= varying_intersection
                <= upper + segment_tolerance
            ):
                continue

            edge_parameter_tolerance = segment_tolerance / chain_lengths[edge_position]
            if edge_parameter_tolerance < parameter < 1.0 - edge_parameter_tolerance:
                raise ValueError(
                    "a pattern segment intersects a boundary edge without a vertex"
                )
            if parameter <= edge_parameter_tolerance:
                vertex_hits.add(int(start_position))
            else:
                vertex_hits.add(int(end_position))

        return vertex_hits, fixed, lower, upper

    constrained_targets: dict[int, np.ndarray] = {}
    for segment in pattern_lines:
        segment_scale = max(
            coordinate_scale,
            float(np.max(np.abs(segment))),
        )
        segment_tolerance = 64.0 * np.finfo(np.float64).eps * segment_scale
        segment_angular_tolerance = max(
            64.0 * np.finfo(np.float64).eps,
            segment_tolerance / radius,
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

        vertex_hits, fixed, lower, upper = segment_vertex_hits(
            segment,
            vertical,
            segment_tolerance,
        )
        if not vertex_hits:
            continue

        fixed_center = center[0 if vertical else 1]
        varying_center = center[1 if vertical else 0]
        normalized_fixed_offset = abs(fixed - fixed_center) / radius
        if normalized_fixed_offset > 1.0 + segment_angular_tolerance:
            raise ValueError(
                "an active pattern segment does not intersect the target circle"
            )
        normalized_fixed_offset = min(normalized_fixed_offset, 1.0)
        root_offset = radius * np.sqrt(
            max(0.0, 1.0 - normalized_fixed_offset**2)
        )
        varying_roots = [varying_center - root_offset]
        if root_offset > segment_tolerance:
            varying_roots.append(varying_center + root_offset)
        varying_roots = [
            value
            for value in varying_roots
            if lower - segment_tolerance <= value <= upper + segment_tolerance
        ]
        if not varying_roots:
            raise ValueError(
                "an active pattern segment does not reach the target circle"
            )

        for vertex_position in vertex_hits:
            source = selected_xy[vertex_position]
            varying_source = source[1 if vertical else 0]
            root_distances = np.abs(np.asarray(varying_roots) - varying_source)
            nearest = int(np.argmin(root_distances))
            if root_distances.size > 1:
                ordered_distances = np.sort(root_distances)
                if (
                    ordered_distances[1] - ordered_distances[0]
                    <= segment_tolerance
                ):
                    raise ValueError(
                        "a pattern segment has ambiguous circle intersections"
                    )
            if vertical:
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

    def lifted_pattern_angle(target, reference):
        ordinary_angle = np.arctan2(target[1] - center[1], target[0] - center[0])
        directed_angle = angular_direction * ordinary_angle
        return directed_angle + full_turn * np.rint(
            (reference - directed_angle) / full_turn
        )

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
                    lifted_pattern_angle(
                        constrained_targets[original_position],
                        rotated_preferred[local_position],
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
                    lifted_pattern_angle(
                        constrained_targets[int(position)],
                        preferred_angles[int(position)],
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

    target_offsets = target_xy - center
    target_distances = np.hypot(target_offsets[:, 0], target_offsets[:, 1])
    if np.any(
        np.abs(target_distances - radius)
        > max(tolerance, radius * 1.0e-12)
    ):
        raise ValueError("a generated target node is not on the circle")

    target_raw_angles = np.arctan2(target_offsets[:, 1], target_offsets[:, 0])
    if is_closed:
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
    else:
        target_angles = np.unwrap(target_raw_angles)
        target_angles *= angular_direction
        angle_steps = np.diff(target_angles)
        if np.any(angle_steps <= angular_tolerance):
            raise ValueError(
                "circle target nodes must preserve strict angular order"
            )
        if target_angles[-1] - target_angles[0] >= (
            full_turn - angular_tolerance
        ):
            raise ValueError("an open chain cannot span a complete circle")

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
        new_elements = np.column_stack(
            (
                source_ends,
                source_starts,
                target_starts,
                target_ends,
            )
        )
    else:
        new_elements = np.column_stack(
            (
                source_starts,
                source_ends,
                target_ends,
                target_starts,
            )
        )

    proposed_xy = np.vstack((float_nodes[:, :2], target_xy))
    quad_points = proposed_xy[new_elements]
    forward_edges = np.roll(quad_points, -1, axis=1) - quad_points
    backward_edges = np.roll(quad_points, 1, axis=1) - quad_points
    forward_lengths = np.hypot(forward_edges[:, :, 0], forward_edges[:, :, 1])
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
            "generated quadrilaterals must be non-degenerate and counter-clockwise"
        )

    # Positive corner Jacobians rule out a folded individual quad.  A separate
    # edge check prevents two otherwise-valid, non-neighbouring quads from
    # crossing elsewhere in the strip.
    strip_edges: dict[tuple[int, int], tuple[int, int]] = {}
    for element in new_elements:
        for start, end in zip(element, np.roll(element, -1)):
            start = int(start)
            end = int(end)
            key = (min(start, end), max(start, end))
            strip_edges.setdefault(key, (start, end))

    def segments_intersect(first_edge, second_edge):
        first_start, first_end = proposed_xy[list(first_edge)]
        second_start, second_end = proposed_xy[list(second_edge)]
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

        def orientation(start, end, point):
            edge = end - start
            offset = point - start
            return edge[0] * offset[1] - edge[1] * offset[0]

        orientations = (
            orientation(first_start, first_end, second_start),
            orientation(first_start, first_end, second_end),
            orientation(second_start, second_end, first_start),
            orientation(second_start, second_end, first_end),
        )
        signs = tuple(
            1 if value > area_tolerance else -1 if value < -area_tolerance else 0
            for value in orientations
        )
        return signs[0] * signs[1] <= 0 and signs[2] * signs[3] <= 0

    edge_items = list(strip_edges.items())
    for first_position, (first_key, first_edge) in enumerate(edge_items):
        for second_key, second_edge in edge_items[first_position + 1 :]:
            if set(first_key).intersection(second_key):
                continue
            if segments_intersect(first_edge, second_edge):
                raise ValueError("non-neighbouring edges in the generated strip cross")

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


def circle(
    mesh: Mesh,
    x: float,
    y: float,
    radius: float,
    buffer: float,
    lines=None,
):
    indices_inner = _search_circle(mesh, x, y, radius - buffer)
    indices_outer = _search_circle(mesh, x, y, radius + buffer)

    indices_delete = indices_outer[~np.isin(indices_outer, indices_inner)]

    indices_list = _get_boundary(mesh, indices=indices_delete)
    node_indices_outer = indices_list[0]
    node_indices_inner = indices_list[1]

    mesh = _delete_element(mesh, indices=indices_delete)
    node_map = _clear_node(mesh)
    node_indices_outer = node_map[node_indices_outer]
    node_indices_inner = node_map[node_indices_inner]

    mesh = _to_circle(
        mesh,
        x,
        y,
        radius,
        node_indices_outer,
        lines=lines,
        closed=True,
    )

    view_mesh(mesh, node_indices=node_indices_outer)

    return mesh
