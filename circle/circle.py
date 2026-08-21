"""Circle-based element selection helpers."""

from typing import NamedTuple

import numpy as np

from mesh import Mesh
from viewer import view_mesh

def _equivalence(mesh: Mesh, tolerance: float = 0.01) -> np.ndarray:
    nodes = mesh.nodes
    elements = mesh.elements

    # Quantize
    qx = np.round(nodes[:, 0] / tolerance).astype(np.int64)
    qy = np.round(nodes[:, 1] / tolerance).astype(np.int64)
    qz = np.round(nodes[:, 2] / tolerance ).astype(np.int64)

    # Sort by (qz, qx, qy)
    order = np.lexsort((qy, qx, qz))  # primary is last key => qz, then qx, then qy
    qz_s = qz[order]; qx_s = qx[order]; qy_s = qy[order]

    # Run-length unique over sorted triples
    same_as_prev = np.zeros(qz_s.shape[0], dtype=bool)
    same_as_prev[1:] = (qz_s[1:] == qz_s[:-1]) & (qx_s[1:] == qx_s[:-1]) & (qy_s[1:] == qy_s[:-1])
    group_id_sorted = np.cumsum(~same_as_prev) - 1  # 0..n_unique-1

    # Invert sorting to get "inverse" mapping for original node order
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(order.size, dtype=order.dtype)
    inverse = group_id_sorted[inv_order].astype(np.int32, copy=False)

    # Extract unique nodes (keep first occurrence)
    unique_mask_sorted = ~same_as_prev
    unique_nodes = nodes[order[unique_mask_sorted]].astype(np.float64, copy=False)

    # Remap elements IN-PLACE (no extra big allocation)
    np.take(inverse, elements, out=elements)

    mesh.nodes = unique_nodes
    mesh.elements = elements

    return inverse


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

    One circle node is created for every node in ``node_indices`` and normally
    one quadrilateral is created for every boundary edge.  A quadrilateral with
    a pattern-induced straight corner is split into two padded Tri3 elements.
    By default a chain is treated as closed when its last and first nodes also
    form an exposed boundary edge; pass ``closed=False`` to keep such a chain
    open explicitly.  Closed input lists contain each node once (the first node
    is not repeated).

    ``lines`` contains finite horizontal or vertical pattern segments.  When a
    segment meets a chain vertex, that vertex is connected to the matching
    segment/circle intersection.  If a segment meets multiple vertices,
    including along overlapping boundary edges, the vertex nearest each
    matching circle intersection is used.  Neighbouring circle nodes are
    redistributed locally when necessary so the exact pattern connector does
    not reverse the circle-node order.

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

        def nearest_root(vertex_position):
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
            return nearest, float(root_distances[nearest])

        # One analytic circle intersection can support only one connector from
        # a given pattern segment.  This also collapses every collinear overlap
        # to the boundary vertex closest to the corresponding circle root.
        representatives: dict[int, tuple[float, int]] = {}
        for vertex_position in sorted(vertex_hits):
            nearest, root_distance = nearest_root(vertex_position)
            representative = (root_distance, vertex_position)
            previous = representatives.get(nearest)
            if previous is None or representative < previous:
                representatives[nearest] = representative
        vertex_hits = {
            vertex_position
            for _, vertex_position in representatives.values()
        }

        for vertex_position in vertex_hits:
            nearest, _ = nearest_root(vertex_position)
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

    proposed_xy = np.vstack((float_nodes[:, :2], target_xy))
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
            first_triangle_edge = triangle_edges[:, 0] / triangle_lengths[:, 0, None]
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

    # Positive element orientations rule out folded individual cells.  A
    # separate edge check prevents two otherwise-valid, non-neighbouring cells
    # from crossing elsewhere in the strip.
    strip_edges: dict[tuple[int, int], tuple[int, int]] = {}
    for element in new_elements:
        for start, end in zip(element, np.roll(element, -1)):
            start = int(start)
            end = int(end)
            if start == end:
                continue
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


class _StripScore(NamedTuple):
    worst_mismatch: float
    total_squared_mismatch: float
    minimum_quality: float


class _CircularStripMesher:
    """Build a conforming triangular strip between two concentric boundaries."""

    def __init__(self, mesh, inner_nodes, outer_nodes, lines, closed):
        self.mesh = mesh
        self.inner_input = inner_nodes
        self.outer_input = outer_nodes
        self.lines_input = lines
        self.closed_input = closed

    def build(self):
        self._read_inputs()
        self._fit_concentric_geometry()
        self._read_existing_topology()
        self._order_boundaries()
        self._validate_existing_geometry()
        self._collect_pattern_connectors()
        triangles = self._triangulate()
        self._validate_completed_strip(triangles)
        self._commit(triangles)
        return self.mesh

    def _read_inputs(self):
        if not isinstance(self.mesh, Mesh):
            raise TypeError("mesh must be a Mesh instance")
        if not isinstance(self.closed_input, (bool, np.bool_)):
            raise TypeError("closed must be True or False")
        self.closed = bool(self.closed_input)

        self.nodes = np.asarray(self.mesh.nodes)
        self.elements = np.asarray(self.mesh.elements)
        if self.nodes.ndim != 2 or self.nodes.shape[1] not in (2, 3):
            raise ValueError("nodes must have shape (N, 2) or (N, 3)")
        if (
            not np.issubdtype(self.nodes.dtype, np.number)
            or np.issubdtype(self.nodes.dtype, np.bool_)
            or np.issubdtype(self.nodes.dtype, np.complexfloating)
        ):
            raise ValueError("nodes must have a real numeric dtype")
        if self.elements.ndim != 2 or self.elements.shape[1] != 4:
            raise ValueError("elements must have shape (M, 4)")
        if (
            not np.issubdtype(self.elements.dtype, np.integer)
            or np.issubdtype(self.elements.dtype, np.bool_)
        ):
            raise ValueError("elements must have an integer dtype")

        try:
            float_nodes = self.nodes.astype(np.float64, copy=False)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("nodes must be representable as float64") from error
        if not np.all(np.isfinite(float_nodes)):
            raise ValueError("nodes must contain finite float64 coordinates")
        self.xy = float_nodes[:, :2]

        self.inner_indices = self._normalize_indices(
            self.inner_input, "inner_nodes"
        )
        self.outer_indices = self._normalize_indices(
            self.outer_input, "outer_nodes"
        )
        if np.intersect1d(self.inner_indices, self.outer_indices).size:
            raise ValueError("inner_nodes and outer_nodes must be disjoint")
        if np.any(self.elements < 0) or np.any(
            self.elements >= self.nodes.shape[0]
        ):
            raise ValueError("elements contain an out-of-range node index")

        if self.lines_input is None:
            self.pattern_lines = np.empty((0, 2, 2), dtype=np.float64)
            return
        try:
            pattern_lines = np.asarray(self.lines_input, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("lines must have shape (L, 2, 2)") from error
        if pattern_lines.size == 0:
            if pattern_lines.shape not in ((0,), (0, 2, 2)):
                raise ValueError("lines must have shape (L, 2, 2)")
            pattern_lines = np.empty((0, 2, 2), dtype=np.float64)
        elif pattern_lines.ndim != 3 or pattern_lines.shape[1:] != (2, 2):
            raise ValueError("lines must have shape (L, 2, 2)")
        if not np.all(np.isfinite(pattern_lines)):
            raise ValueError("lines must contain finite coordinates")
        self.pattern_lines = pattern_lines

    def _normalize_indices(self, values, name):
        indices = np.asarray(values)
        if indices.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional sequence")
        minimum = 3 if self.closed else 2
        if indices.size < minimum:
            kind = "ring" if self.closed else "arc"
            raise ValueError(
                f"{name} must contain at least {minimum} nodes for a {kind}"
            )
        if (
            not np.issubdtype(indices.dtype, np.integer)
            or np.issubdtype(indices.dtype, np.bool_)
        ):
            raise TypeError(f"{name} must contain integers")
        if np.any(indices < 0) or np.any(indices >= self.nodes.shape[0]):
            raise IndexError(f"{name} contain an out-of-range node index")
        indices = indices.astype(np.int64, copy=False)
        if np.unique(indices).size != indices.size:
            raise ValueError(f"{name} must not contain duplicates")
        return indices

    def _fit_concentric_geometry(self):
        selected = np.vstack(
            (self.xy[self.inner_indices], self.xy[self.outer_indices])
        )
        origin = selected[0].copy()
        centered = selected - origin
        if not np.all(np.isfinite(centered)):
            raise ValueError("selected circle-coordinate differences exceed float64")
        scale = float(np.max(np.abs(centered)))
        if scale == 0.0:
            raise ValueError("the selected nodes do not determine circle radii")

        normalized = centered / scale
        design = np.zeros((selected.shape[0], 4), dtype=np.float64)
        design[:, :2] = 2.0 * normalized
        design[: self.inner_indices.size, 2] = 1.0
        design[self.inner_indices.size :, 3] = 1.0
        rhs = np.einsum("ij,ij->i", normalized, normalized)
        try:
            fit, _, rank, _ = np.linalg.lstsq(design, rhs, rcond=None)
        except np.linalg.LinAlgError as error:
            raise ValueError("the common circle fit could not be solved") from error
        if rank != 4 or not np.all(np.isfinite(fit)):
            raise ValueError(
                "the selected arcs do not determine two concentric circles"
            )

        normalized_center = fit[:2]
        center_norm_squared = float(np.dot(normalized_center, normalized_center))
        inner_squared = float(fit[2] + center_norm_squared)
        outer_squared = float(fit[3] + center_norm_squared)
        if (
            not np.isfinite(inner_squared)
            or not np.isfinite(outer_squared)
            or inner_squared <= 0.0
            or outer_squared <= 0.0
        ):
            raise ValueError("the fitted circle radii must be positive")

        with np.errstate(over="ignore", invalid="ignore"):
            self.center = origin + scale * normalized_center
            self.inner_radius = scale * np.sqrt(inner_squared)
            self.outer_radius = scale * np.sqrt(outer_squared)
        if (
            not np.all(np.isfinite(self.center))
            or not np.isfinite(self.inner_radius)
            or not np.isfinite(self.outer_radius)
        ):
            raise ValueError("the fitted circle geometry exceeds float64")

        coordinate_scale = max(
            float(np.max(np.abs(selected))),
            float(np.max(np.abs(self.center))),
            self.outer_radius,
        )
        coordinate_ulp = abs(float(np.spacing(coordinate_scale)))
        local_scale = max(scale, self.outer_radius)
        machine_tolerance = max(
            8.0 * coordinate_ulp,
            256.0 * np.finfo(np.float64).eps * local_scale,
        )
        self.geometry_tolerance = max(
            machine_tolerance, 1.0e-10 * self.outer_radius
        )
        if (
            self.inner_radius
            >= self.outer_radius - self.geometry_tolerance
        ):
            raise ValueError("inner_nodes must lie on the smaller fitted circle")

        inner_distances = np.hypot(
            self.xy[self.inner_indices, 0] - self.center[0],
            self.xy[self.inner_indices, 1] - self.center[1],
        )
        outer_distances = np.hypot(
            self.xy[self.outer_indices, 0] - self.center[0],
            self.xy[self.outer_indices, 1] - self.center[1],
        )
        self.fit_tolerance = max(
            self.geometry_tolerance, 1.0e-9 * self.outer_radius
        )
        if np.any(
            np.abs(inner_distances - self.inner_radius) > self.fit_tolerance
        ) or np.any(
            np.abs(outer_distances - self.outer_radius) > self.fit_tolerance
        ):
            raise ValueError(
                "inner_nodes and outer_nodes must lie on concentric circles"
            )
        self.angular_tolerance = max(
            256.0 * np.finfo(np.float64).eps,
            self.geometry_tolerance / self.inner_radius,
        )
        offsets = self.xy - self.center
        self.node_angles = np.mod(
            np.arctan2(offsets[:, 1], offsets[:, 0]), 2.0 * np.pi
        )

    def _read_existing_topology(self):
        self.existing_edge_uses = {}
        self.existing_perimeters = []
        for element in self.elements:
            values = [int(value) for value in element]
            if values[2] == values[3]:
                perimeter = values[:3]
                if len(set(perimeter)) != 3:
                    raise ValueError(
                        "a padded Tri3 must contain three distinct perimeter nodes"
                    )
            else:
                perimeter = values
                if len(set(perimeter)) != 4:
                    raise ValueError(
                        "a Quad4 must contain four distinct nodes"
                    )
            self.existing_perimeters.append(tuple(perimeter))
            for start, end in zip(
                perimeter, perimeter[1:] + perimeter[:1]
            ):
                key = self._edge_key((start, end))
                uses = self.existing_edge_uses.setdefault(key, [])
                uses.append((start, end))
                if len(uses) > 2:
                    raise ValueError(
                        "mesh contains a non-manifold edge shared by more than "
                        "two elements"
                    )
        for uses in self.existing_edge_uses.values():
            if len(uses) == 2 and uses[0] != uses[1][::-1]:
                raise ValueError(
                    "elements sharing an edge must traverse it in opposite "
                    "directions"
                )

    @staticmethod
    def _edge_key(edge):
        first, second = map(int, edge)
        return (min(first, second), max(first, second))

    @staticmethod
    def _cross(first, second):
        return first[0] * second[1] - first[1] * second[0]

    def _polygon_area(self, indices):
        points = self.xy[np.asarray(indices, dtype=np.intp)]
        points = points - points[0]
        following = np.roll(points, -1, axis=0)
        return 0.5 * float(
            np.sum(
                points[:, 0] * following[:, 1]
                - following[:, 0] * points[:, 1]
            )
        )

    def _area_tolerance(self, vertex_count):
        return (
            self.geometry_tolerance
            * max(self.geometry_tolerance, self.outer_radius)
            * max(1, int(vertex_count))
        )

    @staticmethod
    def _boundary_edges(indices, closing):
        indices = np.asarray(indices, dtype=np.int64)
        count = indices.size if closing else indices.size - 1
        return [
            (
                int(indices[position]),
                int(indices[(position + 1) % indices.size]),
            )
            for position in range(count)
        ]

    def _segment_relation(self, first_edge, second_edge):
        first_start, first_end = self.xy[list(first_edge)]
        second_start, second_end = self.xy[list(second_edge)]
        first_vector = first_end - first_start
        second_vector = second_end - second_start
        length_scale = max(
            self.geometry_tolerance,
            float(np.hypot(*first_vector)),
            float(np.hypot(*second_vector)),
        )
        area_tolerance = self.geometry_tolerance * length_scale

        if (
            max(
                min(first_start[0], first_end[0]),
                min(second_start[0], second_end[0]),
            )
            > min(
                max(first_start[0], first_end[0]),
                max(second_start[0], second_end[0]),
            )
            + self.geometry_tolerance
            or max(
                min(first_start[1], first_end[1]),
                min(second_start[1], second_end[1]),
            )
            > min(
                max(first_start[1], first_end[1]),
                max(second_start[1], second_end[1]),
            )
            + self.geometry_tolerance
        ):
            return "none"

        orientations = np.asarray(
            [
                self._cross(first_vector, second_start - first_start),
                self._cross(first_vector, second_end - first_start),
                self._cross(second_vector, first_start - second_start),
                self._cross(second_vector, first_end - second_start),
            ],
            dtype=np.float64,
        )
        if np.all(np.abs(orientations) <= area_tolerance):
            axis = int(np.argmax(np.abs(first_vector)))
            first_interval = sorted((first_start[axis], first_end[axis]))
            second_interval = sorted((second_start[axis], second_end[axis]))
            overlap = (
                min(first_interval[1], second_interval[1])
                - max(first_interval[0], second_interval[0])
            )
            if overlap > self.geometry_tolerance:
                return "overlap"
            if overlap >= -self.geometry_tolerance:
                return "touch"
            return "none"

        signs = np.where(
            orientations > area_tolerance,
            1,
            np.where(orientations < -area_tolerance, -1, 0),
        )
        if signs[0] * signs[1] <= 0 and signs[2] * signs[3] <= 0:
            return "cross"
        return "none"

    def _edges_are_compatible(self, first_edge, second_edge):
        if self._edge_key(first_edge) == self._edge_key(second_edge):
            return True
        relation = self._segment_relation(first_edge, second_edge)
        if relation == "none":
            return True
        shared = set(first_edge).intersection(second_edge)
        return bool(shared) and relation != "overlap"

    def _point_on_segment(self, point, start, end):
        vector = end - start
        offset = point - start
        length = float(np.hypot(*vector))
        if length <= self.geometry_tolerance:
            return float(np.hypot(*offset)) <= self.geometry_tolerance
        if (
            abs(self._cross(vector, offset))
            > self.geometry_tolerance * length
        ):
            return False
        projection = float(np.dot(offset, vector))
        margin = self.geometry_tolerance * length
        return -margin <= projection <= length * length + margin

    def _point_on_polygon(self, point, polygon_ids):
        polygon = self.xy[np.asarray(polygon_ids, dtype=np.intp)]
        return any(
            self._point_on_segment(point, start, end)
            for start, end in zip(
                polygon, np.roll(polygon, -1, axis=0)
            )
        )

    def _point_in_polygon(self, point, polygon_ids):
        polygon = self.xy[np.asarray(polygon_ids, dtype=np.intp)]
        following = np.roll(polygon, -1, axis=0)
        crosses_y = (polygon[:, 1] > point[1]) != (
            following[:, 1] > point[1]
        )
        positions = np.flatnonzero(crosses_y)
        if positions.size == 0:
            return False
        starts = polygon[positions]
        ends = following[positions]
        intersection_x = starts[:, 0] + (
            (point[1] - starts[:, 1])
            * (ends[:, 0] - starts[:, 0])
            / (ends[:, 1] - starts[:, 1])
        )
        return bool(np.count_nonzero(point[0] < intersection_x) % 2)

    def _polygon_is_simple(self, polygon_ids):
        edges = self._boundary_edges(polygon_ids, True)
        for first_position, first_edge in enumerate(edges):
            for second_position in range(first_position + 1, len(edges)):
                second_edge = edges[second_position]
                adjacent = (
                    second_position == first_position + 1
                    or (
                        first_position == 0
                        and second_position == len(edges) - 1
                    )
                )
                relation = self._segment_relation(first_edge, second_edge)
                if relation == "none":
                    continue
                if adjacent and relation != "overlap":
                    continue
                return False
        return True

    def _closed_order(self, indices, name):
        angles = self.node_angles[indices]
        order = np.lexsort((indices, angles))
        ordered = indices[order]
        ordered_angles = angles[order]
        steps = np.mod(
            np.roll(ordered_angles, -1) - ordered_angles, 2.0 * np.pi
        )
        if np.any(steps <= self.angular_tolerance):
            raise ValueError(f"{name} contain duplicate angular positions")
        return ordered.astype(np.int64, copy=False)

    def _open_orders(self, indices, name):
        angles = self.node_angles[indices]
        interior_positions = np.arange(1, indices.size - 1, dtype=np.intp)
        candidates = []
        endpoints = ((0, indices.size - 1), (indices.size - 1, 0))
        for start_position, end_position in endpoints:
            start_angle = angles[start_position]
            span = float(
                np.mod(angles[end_position] - start_angle, 2.0 * np.pi)
            )
            if span <= self.angular_tolerance:
                continue
            steps = np.mod(
                angles[interior_positions] - start_angle, 2.0 * np.pi
            )
            if steps.size and (
                np.any(steps <= self.angular_tolerance)
                or np.any(steps >= span - self.angular_tolerance)
            ):
                continue
            order = interior_positions[
                np.lexsort((indices[interior_positions], steps))
            ]
            if order.size:
                sorted_steps = np.mod(
                    angles[order] - start_angle, 2.0 * np.pi
                )
                if np.any(
                    np.diff(sorted_steps) <= self.angular_tolerance
                ):
                    continue
            candidate = np.concatenate(
                (
                    indices[[start_position]],
                    indices[order],
                    indices[[end_position]],
                )
            ).astype(np.int64, copy=False)
            signature = tuple(map(int, candidate))
            if all(tuple(map(int, item)) != signature for item in candidates):
                candidates.append(candidate)
        if not candidates:
            raise ValueError(
                f"the interior nodes of {name} do not lie on one "
                "endpoint-bounded arc"
            )
        return candidates

    def _order_boundaries(self):
        if self.closed:
            self.inner_order_candidates = [
                self._closed_order(self.inner_indices, "inner_nodes")
            ]
            self.outer_order_candidates = [
                self._closed_order(self.outer_indices, "outer_nodes")
            ]
            self.open_geometry_candidates = []
            return

        self.inner_order_candidates = self._open_orders(
            self.inner_indices, "inner_nodes"
        )
        self.outer_order_candidates = self._open_orders(
            self.outer_indices, "outer_nodes"
        )
        self.open_geometry_candidates = []
        for outer_order in self.outer_order_candidates:
            for inner_order in self.inner_order_candidates:
                boundary = np.concatenate(
                    (outer_order, inner_order[::-1])
                ).astype(np.int64, copy=False)
                if self._polygon_area(boundary) <= self._area_tolerance(
                    boundary.size
                ):
                    continue
                if not self._polygon_is_simple(boundary):
                    continue
                self.open_geometry_candidates.append(
                    (outer_order, inner_order, boundary)
                )

    def _triangle_quality(self, first, second, third):
        if len({int(first), int(second), int(third)}) != 3:
            return -np.inf
        points = self.xy[[first, second, third]]
        forward = np.roll(points, -1, axis=0) - points
        lengths = np.hypot(forward[:, 0], forward[:, 1])
        if np.any(lengths <= self.geometry_tolerance):
            return -np.inf
        backward = np.roll(points, 1, axis=0) - points
        corner_cross = (
            forward[:, 0] * backward[:, 1]
            - forward[:, 1] * backward[:, 0]
        )
        length_scale = max(
            self.geometry_tolerance, float(np.max(lengths))
        )
        if np.any(
            corner_cross <= self.geometry_tolerance * length_scale
        ):
            return -np.inf
        quality = (
            (2.0 / np.sqrt(3.0))
            * corner_cross
            / (lengths * np.roll(lengths, 1))
        )
        if not np.all(np.isfinite(quality)):
            return -np.inf
        return float(np.min(quality))

    def _validate_existing_geometry(self):
        for perimeter in self.existing_perimeters:
            if not self._polygon_is_simple(perimeter):
                raise ValueError(
                    "existing elements must have simple perimeters"
                )
            if len(perimeter) == 3:
                if not np.isfinite(self._triangle_quality(*perimeter)):
                    raise ValueError(
                        "existing Tri3 elements must be non-degenerate and "
                        "counter-clockwise"
                    )
                continue

            points = self.xy[np.asarray(perimeter, dtype=np.intp)]
            forward = np.roll(points, -1, axis=0) - points
            backward = np.roll(points, 1, axis=0) - points
            forward_lengths = np.hypot(
                forward[:, 0], forward[:, 1]
            )
            backward_lengths = np.hypot(
                backward[:, 0], backward[:, 1]
            )
            if np.any(forward_lengths <= self.geometry_tolerance):
                raise ValueError(
                    "existing Quad4 elements have a zero-length edge"
                )
            corner_cross = (
                forward[:, 0] * backward[:, 1]
                - forward[:, 1] * backward[:, 0]
            )
            normalized_turn = corner_cross / (
                forward_lengths * backward_lengths
            )
            if (
                not np.all(np.isfinite(normalized_turn))
                or np.any(
                    normalized_turn <= self.angular_tolerance
                )
            ):
                raise ValueError(
                    "existing Quad4 elements must be non-degenerate and "
                    "counter-clockwise"
                )

    def _axis_interval_enters_open_strip(
        self,
        fixed_axis,
        varying_axis,
        fixed_value,
        interval_lower,
        interval_upper,
        tolerance,
    ):
        if self.closed:
            return interval_upper - interval_lower > tolerance
        if interval_upper - interval_lower <= tolerance:
            return False

        for _, _, polygon_ids in self.open_geometry_candidates:
            polygon = self.xy[np.asarray(polygon_ids, dtype=np.intp)]
            cuts = [float(interval_lower), float(interval_upper)]
            for start, end in zip(
                polygon, np.roll(polygon, -1, axis=0)
            ):
                fixed_start = float(start[fixed_axis])
                fixed_end = float(end[fixed_axis])
                if (
                    fixed_value
                    < min(fixed_start, fixed_end) - tolerance
                    or fixed_value
                    > max(fixed_start, fixed_end) + tolerance
                ):
                    continue
                fixed_delta = fixed_end - fixed_start
                if abs(fixed_delta) <= tolerance:
                    if abs(fixed_value - fixed_start) <= tolerance:
                        cuts.extend(
                            (
                                float(start[varying_axis]),
                                float(end[varying_axis]),
                            )
                        )
                    continue
                fraction = (
                    fixed_value - fixed_start
                ) / fixed_delta
                if -tolerance <= fraction <= 1.0 + tolerance:
                    cuts.append(
                        float(
                            start[varying_axis]
                            + fraction
                            * (end[varying_axis] - start[varying_axis])
                        )
                    )

            clipped = sorted(
                max(interval_lower, min(interval_upper, value))
                for value in cuts
                if interval_lower - tolerance
                <= value
                <= interval_upper + tolerance
            )
            distinct = []
            for value in clipped:
                if (
                    not distinct
                    or value - distinct[-1] > tolerance
                ):
                    distinct.append(value)
                else:
                    distinct[-1] = 0.5 * (
                        distinct[-1] + value
                    )
            for lower, upper in zip(distinct, distinct[1:]):
                if upper - lower <= tolerance:
                    continue
                midpoint = np.empty(2, dtype=np.float64)
                midpoint[fixed_axis] = fixed_value
                midpoint[varying_axis] = 0.5 * (lower + upper)
                if self._point_on_polygon(midpoint, polygon_ids):
                    continue
                if self._point_in_polygon(midpoint, polygon_ids):
                    return True
        return False

    def _match_circle_node(
        self, indices, target, description, tolerance
    ):
        distances = np.hypot(
            self.xy[indices, 0] - target[0],
            self.xy[indices, 1] - target[1],
        )
        matches = np.flatnonzero(distances <= tolerance)
        if matches.size == 0:
            return None
        if matches.size > 1:
            raise ValueError(
                f"an active pattern segment has an ambiguous "
                f"{description} node"
            )
        return int(indices[int(matches[0])])

    def _collect_pattern_connectors(self):
        connectors = set()
        for segment in self.pattern_lines:
            delta = segment[1] - segment[0]
            axis_scales = np.max(np.abs(segment), axis=0)
            axis_tolerances = np.maximum(
                self.geometry_tolerance,
                8.0 * np.abs(np.spacing(axis_scales)),
            )
            vertical = (
                abs(delta[0]) <= axis_tolerances[0]
                and abs(delta[1]) > axis_tolerances[1]
            )
            horizontal = (
                abs(delta[1]) <= axis_tolerances[1]
                and abs(delta[0]) > axis_tolerances[0]
            )
            if vertical == horizontal:
                raise ValueError(
                    "each pattern segment must be non-zero and horizontal "
                    "or vertical"
                )

            fixed_axis = 0 if vertical else 1
            varying_axis = 1 - fixed_axis
            fixed_tolerance = float(axis_tolerances[fixed_axis])
            bound_tolerance = float(axis_tolerances[varying_axis])
            intersection_tolerance = max(
                self.geometry_tolerance, fixed_tolerance
            )
            fixed = float(segment[0, fixed_axis])
            segment_lower = float(
                np.min(segment[:, varying_axis])
            )
            segment_upper = float(
                np.max(segment[:, varying_axis])
            )
            fixed_distance = abs(fixed - self.center[fixed_axis])
            if (
                fixed_distance
                > self.outer_radius + fixed_tolerance
            ):
                continue
            if (
                abs(fixed_distance - self.outer_radius)
                <= fixed_tolerance
            ):
                continue

            normalized_outer = min(
                fixed_distance / self.outer_radius, 1.0
            )
            outer_root = self.outer_radius * np.sqrt(
                max(0.0, 1.0 - normalized_outer**2)
            )
            if outer_root <= intersection_tolerance:
                continue
            outer_lower = (
                self.center[varying_axis] - outer_root
            )
            outer_upper = (
                self.center[varying_axis] + outer_root
            )

            if (
                fixed_distance
                > self.inner_radius + fixed_tolerance
            ):
                overlap_lower = max(segment_lower, outer_lower)
                overlap_upper = min(segment_upper, outer_upper)
                active = (
                    overlap_upper - overlap_lower
                    > intersection_tolerance
                    and self._axis_interval_enters_open_strip(
                        fixed_axis,
                        varying_axis,
                        fixed,
                        overlap_lower,
                        overlap_upper,
                        intersection_tolerance,
                    )
                )
                if active:
                    raise ValueError(
                        "an active pattern segment forms an unsupported "
                        "outer-circle chord"
                    )
                continue

            if (
                abs(fixed_distance - self.inner_radius)
                <= fixed_tolerance
            ):
                inner_lower = inner_upper = self.center[varying_axis]
            else:
                normalized_inner = min(
                    fixed_distance / self.inner_radius, 1.0
                )
                inner_root = self.inner_radius * np.sqrt(
                    max(0.0, 1.0 - normalized_inner**2)
                )
                inner_lower = (
                    self.center[varying_axis] - inner_root
                )
                inner_upper = (
                    self.center[varying_axis] + inner_root
                )

            for first_value, second_value in (
                (outer_lower, inner_lower),
                (inner_upper, outer_upper),
            ):
                branch_lower = min(first_value, second_value)
                branch_upper = max(first_value, second_value)
                if (
                    branch_upper - branch_lower
                    <= intersection_tolerance
                ):
                    continue
                overlap_lower = max(segment_lower, branch_lower)
                overlap_upper = min(segment_upper, branch_upper)
                if (
                    overlap_upper - overlap_lower
                    <= intersection_tolerance
                ):
                    continue
                active = self._axis_interval_enters_open_strip(
                    fixed_axis,
                    varying_axis,
                    fixed,
                    overlap_lower,
                    overlap_upper,
                    intersection_tolerance,
                )
                if not active:
                    continue
                if not (
                    segment_lower
                    <= branch_lower + bound_tolerance
                    and segment_upper
                    >= branch_upper - bound_tolerance
                ):
                    raise ValueError(
                        "a pattern segment ends inside the annular material"
                    )

                if first_value in (outer_lower, outer_upper):
                    outer_value = first_value
                    inner_value = second_value
                else:
                    outer_value = second_value
                    inner_value = first_value
                outer_target = np.empty(2, dtype=np.float64)
                inner_target = np.empty(2, dtype=np.float64)
                outer_target[fixed_axis] = fixed
                outer_target[varying_axis] = outer_value
                inner_target[fixed_axis] = fixed
                inner_target[varying_axis] = inner_value
                match_tolerance = max(
                    self.fit_tolerance,
                    fixed_tolerance,
                    self.geometry_tolerance,
                )
                outer_node = self._match_circle_node(
                    self.outer_indices,
                    outer_target,
                    "outer circle",
                    match_tolerance,
                )
                inner_node = self._match_circle_node(
                    self.inner_indices,
                    inner_target,
                    "inner circle",
                    match_tolerance,
                )
                if outer_node is None:
                    raise ValueError(
                        "an active pattern segment reaches outer circle "
                        "without a node"
                    )
                if inner_node is None:
                    raise ValueError(
                        "an active pattern segment reaches inner circle "
                        "without a node"
                    )
                connectors.add((outer_node, inner_node))

        self.forced_connectors = sorted(connectors)
        for position, first in enumerate(self.forced_connectors):
            for second in self.forced_connectors[position + 1 :]:
                if not self._edges_are_compatible(first, second):
                    raise ValueError(
                        "pattern connectors cross or overlap"
                    )

    def _connector_mismatch(self, outer_node, inner_node):
        difference = (
            self.node_angles[int(outer_node)]
            - self.node_angles[int(inner_node)]
        )
        return abs(
            float(np.arctan2(np.sin(difference), np.cos(difference)))
        )

    def _score_better(self, candidate, current):
        """Compare path scores in connector-locality priority order."""
        if current is None:
            return True
        angular_epsilon = max(
            self.angular_tolerance, 32.0 * np.finfo(np.float64).eps
        )
        if (
            candidate.worst_mismatch
            < current.worst_mismatch - angular_epsilon
        ):
            return True
        if (
            candidate.worst_mismatch
            > current.worst_mismatch + angular_epsilon
        ):
            return False
        sum_epsilon = max(
            1.0e-15,
            angular_epsilon
            * max(
                1.0,
                abs(candidate.total_squared_mismatch),
                abs(current.total_squared_mismatch),
            ),
        )
        if (
            candidate.total_squared_mismatch
            < current.total_squared_mismatch - sum_epsilon
        ):
            return True
        if (
            candidate.total_squared_mismatch
            > current.total_squared_mismatch + sum_epsilon
        ):
            return False
        quality_epsilon = 1.0e-15
        return (
            candidate.minimum_quality
            > current.minimum_quality + quality_epsilon
        )

    @staticmethod
    def _combine_scores(scores):
        return _StripScore(
            max(score.worst_mismatch for score in scores),
            sum(score.total_squared_mismatch for score in scores),
            min(score.minimum_quality for score in scores),
        )

    def _triangulate_sector(
        self, outer_arc, inner_arc, connector_is_valid
    ):
        """Return the globally most local monotone triangulation of one sector."""
        outer_arc = np.asarray(outer_arc, dtype=np.int64)
        inner_arc = np.asarray(inner_arc, dtype=np.int64)
        outer_steps = outer_arc.size - 1
        inner_steps = inner_arc.size - 1
        worst = np.full(
            (outer_steps + 1, inner_steps + 1),
            np.inf,
            dtype=np.float64,
        )
        total = np.full_like(worst, np.inf)
        minimum_quality = np.full_like(worst, -np.inf)
        predecessor = np.zeros(worst.shape, dtype=np.int8)

        start_outer = int(outer_arc[0])
        start_inner = int(inner_arc[0])
        if not connector_is_valid(start_outer, start_inner):
            raise ValueError(
                "a sector start connector leaves the annular material"
            )
        start_mismatch = self._connector_mismatch(
            start_outer, start_inner
        )
        worst[0, 0] = start_mismatch
        total[0, 0] = start_mismatch**2
        minimum_quality[0, 0] = np.inf

        for outer_position in range(outer_steps + 1):
            for inner_position in range(inner_steps + 1):
                if outer_position == 0 and inner_position == 0:
                    continue
                outer_node = int(outer_arc[outer_position])
                inner_node = int(inner_arc[inner_position])
                if not connector_is_valid(outer_node, inner_node):
                    continue
                mismatch = self._connector_mismatch(
                    outer_node, inner_node
                )
                chosen_score = None
                chosen_source = 0

                if (
                    outer_position > 0
                    and np.isfinite(
                        worst[outer_position - 1, inner_position]
                    )
                ):
                    quality = self._triangle_quality(
                        int(outer_arc[outer_position - 1]),
                        outer_node,
                        inner_node,
                    )
                    if np.isfinite(quality):
                        candidate = _StripScore(
                            max(
                                worst[
                                    outer_position - 1, inner_position
                                ],
                                mismatch,
                            ),
                            total[
                                outer_position - 1, inner_position
                            ]
                            + mismatch**2,
                            min(
                                minimum_quality[
                                    outer_position - 1, inner_position
                                ],
                                quality,
                            ),
                        )
                        if self._score_better(
                            candidate, chosen_score
                        ):
                            chosen_score = candidate
                            chosen_source = 1

                if (
                    inner_position > 0
                    and np.isfinite(
                        worst[outer_position, inner_position - 1]
                    )
                ):
                    quality = self._triangle_quality(
                        outer_node,
                        inner_node,
                        int(inner_arc[inner_position - 1]),
                    )
                    if np.isfinite(quality):
                        candidate = _StripScore(
                            max(
                                worst[
                                    outer_position, inner_position - 1
                                ],
                                mismatch,
                            ),
                            total[
                                outer_position, inner_position - 1
                            ]
                            + mismatch**2,
                            min(
                                minimum_quality[
                                    outer_position,
                                    inner_position - 1,
                                ],
                                quality,
                            ),
                        )
                        if self._score_better(
                            candidate, chosen_score
                        ):
                            chosen_score = candidate
                            chosen_source = 2

                if chosen_score is not None:
                    worst[outer_position, inner_position] = (
                        chosen_score.worst_mismatch
                    )
                    total[outer_position, inner_position] = (
                        chosen_score.total_squared_mismatch
                    )
                    minimum_quality[
                        outer_position, inner_position
                    ] = chosen_score.minimum_quality
                    predecessor[
                        outer_position, inner_position
                    ] = chosen_source

        score = _StripScore(
            worst[outer_steps, inner_steps],
            total[outer_steps, inner_steps],
            minimum_quality[outer_steps, inner_steps],
        )
        if not np.all(np.isfinite(score)):
            raise ValueError(
                "the constrained sector has no valid triangulation"
            )

        rows = []
        outer_position = outer_steps
        inner_position = inner_steps
        while outer_position or inner_position:
            source = int(
                predecessor[outer_position, inner_position]
            )
            if source == 1:
                triangle = [
                    int(outer_arc[outer_position - 1]),
                    int(outer_arc[outer_position]),
                    int(inner_arc[inner_position]),
                ]
                outer_position -= 1
            elif source == 2:
                triangle = [
                    int(outer_arc[outer_position]),
                    int(inner_arc[inner_position]),
                    int(inner_arc[inner_position - 1]),
                ]
                inner_position -= 1
            else:
                raise RuntimeError(
                    "missing triangulation predecessor"
                )
            rows.append([*triangle, triangle[-1]])
        rows.reverse()
        return np.asarray(rows, dtype=np.int64), score

    def _protected_connector_is_valid(self, edge, protected_edges):
        return all(
            self._edges_are_compatible(edge, protected)
            for protected in protected_edges
        )

    def _validate_closed_boundaries(self, outer_order, inner_order):
        if (
            self._polygon_area(outer_order) <= 0.0
            or self._polygon_area(inner_order) <= 0.0
        ):
            raise ValueError(
                "circle rings could not be normalized counter-clockwise"
            )
        for order, name in (
            (outer_order, "outer"),
            (inner_order, "inner"),
        ):
            points = self.xy[order]
            previous = points - np.roll(points, 1, axis=0)
            following = np.roll(points, -1, axis=0) - points
            turns = (
                previous[:, 0] * following[:, 1]
                - previous[:, 1] * following[:, 0]
            )
            local_scale = np.maximum(
                self.geometry_tolerance,
                np.maximum(
                    np.hypot(previous[:, 0], previous[:, 1]),
                    np.hypot(following[:, 0], following[:, 1]),
                ),
            )
            if np.any(
                turns <= self.geometry_tolerance * local_scale
            ):
                raise ValueError(
                    f"the {name} circle polygon must be strictly convex"
                )

        outer_points = self.xy[outer_order]
        inner_points = self.xy[inner_order]
        for start, end in zip(
            outer_points, np.roll(outer_points, -1, axis=0)
        ):
            edge = end - start
            signed_offsets = (
                edge[0] * (inner_points[:, 1] - start[1])
                - edge[1] * (inner_points[:, 0] - start[0])
            )
            threshold = self.geometry_tolerance * max(
                self.geometry_tolerance, float(np.hypot(*edge))
            )
            if np.any(signed_offsets <= threshold):
                raise ValueError(
                    "the outer circle polygon must strictly contain the "
                    "inner polygon"
                )

    def _triangulate_closed(self):
        outer_order = self.outer_order_candidates[0]
        inner_order = self.inner_order_candidates[0]
        self._validate_closed_boundaries(outer_order, inner_order)
        protected = [tuple(edge) for edge in self.forced_connectors]
        inner_positions = {
            int(node): position
            for position, node in enumerate(inner_order)
        }
        connector_cache = {}

        def connector_is_valid(outer_node, inner_node):
            key = (int(outer_node), int(inner_node))
            if key in connector_cache:
                return connector_cache[key]
            inner_position = inner_positions[int(inner_node)]
            inner_point = self.xy[int(inner_node)]
            vector = self.xy[int(outer_node)] - inner_point
            incoming = inner_point - self.xy[
                inner_order[
                    (inner_position - 1) % inner_order.size
                ]
            ]
            outgoing = (
                self.xy[
                    inner_order[
                        (inner_position + 1) % inner_order.size
                    ]
                ]
                - inner_point
            )
            local_scale = max(
                self.geometry_tolerance,
                float(np.hypot(*vector)),
                float(np.hypot(*incoming)),
                float(np.hypot(*outgoing)),
            )
            tolerance = self.geometry_tolerance * local_scale
            valid = (
                self._cross(incoming, vector) < -tolerance
                or self._cross(outgoing, vector) < -tolerance
            )
            if valid:
                valid = self._protected_connector_is_valid(
                    key, protected
                )
            connector_cache[key] = valid
            return valid

        outer_positions = {
            int(node): position
            for position, node in enumerate(outer_order)
        }
        inner_position_by_node = {
            int(node): position
            for position, node in enumerate(inner_order)
        }
        anchors = [
            (
                outer_positions[int(outer_node)],
                inner_position_by_node[int(inner_node)],
                int(outer_node),
                int(inner_node),
            )
            for outer_node, inner_node in self.forced_connectors
        ]
        if not anchors:
            seam_candidates = []
            for outer_node in outer_order:
                for inner_node in inner_order:
                    outer_node = int(outer_node)
                    inner_node = int(inner_node)
                    if not connector_is_valid(
                        outer_node, inner_node
                    ):
                        continue
                    seam_candidates.append(
                        (
                            self._connector_mismatch(
                                outer_node, inner_node
                            ),
                            float(
                                np.hypot(
                                    *(
                                        self.xy[outer_node]
                                        - self.xy[inner_node]
                                    )
                                )
                            ),
                            outer_node,
                            inner_node,
                        )
                    )
            if not seam_candidates:
                raise ValueError(
                    "no valid seam exists between the two circles"
                )
            _, _, outer_node, inner_node = min(seam_candidates)
            anchors = [
                (
                    outer_positions[outer_node],
                    inner_position_by_node[inner_node],
                    outer_node,
                    inner_node,
                )
            ]

        outer_count = outer_order.size
        inner_count = inner_order.size
        anchor_orders = []
        for seed in anchors:
            seed_outer, seed_inner = seed[:2]
            ordered = sorted(
                anchors,
                key=lambda anchor: (
                    (anchor[0] - seed_outer) % outer_count,
                    (anchor[1] - seed_inner) % inner_count,
                    anchor[2],
                    anchor[3],
                ),
            )
            if ordered[0] != seed:
                continue
            outer_offsets = np.asarray(
                [
                    (anchor[0] - seed_outer) % outer_count
                    for anchor in ordered
                ],
                dtype=np.int64,
            )
            inner_offsets = np.asarray(
                [
                    (anchor[1] - seed_inner) % inner_count
                    for anchor in ordered
                ],
                dtype=np.int64,
            )
            if np.any(np.diff(inner_offsets) < 0):
                continue
            if any(
                outer_offsets[position]
                == outer_offsets[position - 1]
                and inner_offsets[position]
                == inner_offsets[position - 1]
                for position in range(1, len(ordered))
            ):
                continue
            signature = tuple(
                (anchor[2], anchor[3]) for anchor in ordered
            )
            anchor_orders.append(
                (
                    signature,
                    seed_outer,
                    seed_inner,
                    ordered,
                    outer_offsets,
                    inner_offsets,
                )
            )
        if not anchor_orders:
            raise ValueError(
                "pattern connectors do not preserve cyclic order"
            )

        best_rows = None
        best_score = None
        best_signature = None
        for (
            signature,
            seed_outer,
            seed_inner,
            ordered,
            outer_offsets,
            inner_offsets,
        ) in anchor_orders:
            outer_limits = np.append(outer_offsets, outer_count)
            inner_limits = np.append(inner_offsets, inner_count)
            sector_rows = []
            sector_scores = []
            try:
                for position in range(len(ordered)):
                    outer_start = int(outer_limits[position])
                    outer_end = int(outer_limits[position + 1])
                    inner_start = int(inner_limits[position])
                    inner_end = int(inner_limits[position + 1])
                    outer_arc = outer_order[
                        (
                            seed_outer
                            + np.arange(
                                outer_start, outer_end + 1
                            )
                        )
                        % outer_count
                    ]
                    inner_arc = inner_order[
                        (
                            seed_inner
                            + np.arange(
                                inner_start, inner_end + 1
                            )
                        )
                        % inner_count
                    ]
                    rows, score = self._triangulate_sector(
                        outer_arc, inner_arc, connector_is_valid
                    )
                    sector_rows.append(rows)
                    sector_scores.append(score)
            except ValueError:
                continue

            score = self._combine_scores(sector_scores)
            if self._score_better(score, best_score) or (
                best_score is not None
                and not self._score_better(best_score, score)
                and not self._score_better(score, best_score)
                and (
                    best_signature is None
                    or signature < best_signature
                )
            ):
                best_rows = np.concatenate(sector_rows, axis=0)
                best_score = score
                best_signature = signature

        if best_rows is None:
            raise ValueError(
                "pattern connectors do not bound triangulatable sectors"
            )
        self.chosen_outer_order = outer_order
        self.chosen_inner_order = inner_order
        self.chosen_boundary_polygon = None
        return best_rows

    def _triangulate_open(self):
        best_rows = None
        best_score = None
        best_signature = None
        best_orders = None
        errors = []

        for outer_order, inner_order, boundary in (
            self.open_geometry_candidates
        ):
            polygon_edges = self._boundary_edges(boundary, True)
            side_edges = [
                (int(outer_order[0]), int(inner_order[0])),
                (int(outer_order[-1]), int(inner_order[-1])),
            ]
            side_keys = {
                self._edge_key(edge) for edge in side_edges
            }
            protected = [
                *[tuple(edge) for edge in self.forced_connectors],
                *side_edges,
            ]
            if any(
                not self._edges_are_compatible(first, second)
                for position, first in enumerate(protected)
                for second in protected[position + 1 :]
            ):
                continue

            edge_array = np.asarray(polygon_edges, dtype=np.int64)
            edge_points = self.xy[edge_array]
            edge_minimums = np.min(edge_points, axis=1)
            edge_maximums = np.max(edge_points, axis=1)
            connector_cache = {}

            def connector_is_valid(outer_node, inner_node):
                key = (int(outer_node), int(inner_node))
                if key in connector_cache:
                    return connector_cache[key]
                if self._edge_key(key) in side_keys:
                    connector_cache[key] = True
                    return True
                points = self.xy[list(key)]
                minimum = np.min(points, axis=0)
                maximum = np.max(points, axis=0)
                candidates = np.flatnonzero(
                    np.all(
                        (
                            minimum
                            <= edge_maximums
                            + self.geometry_tolerance
                        )
                        & (
                            maximum
                            >= edge_minimums
                            - self.geometry_tolerance
                        ),
                        axis=1,
                    )
                )
                valid = True
                for position in candidates:
                    polygon_edge = polygon_edges[int(position)]
                    if (
                        self._edge_key(key)
                        == self._edge_key(polygon_edge)
                    ):
                        continue
                    if not self._edges_are_compatible(
                        key, polygon_edge
                    ):
                        valid = False
                        break
                if valid:
                    midpoint = 0.5 * (
                        self.xy[key[0]] + self.xy[key[1]]
                    )
                    valid = self._point_in_polygon(
                        midpoint, boundary
                    )
                if valid:
                    valid = self._protected_connector_is_valid(
                        key, protected
                    )
                connector_cache[key] = valid
                return valid

            outer_positions = {
                int(node): position
                for position, node in enumerate(outer_order)
            }
            inner_positions = {
                int(node): position
                for position, node in enumerate(inner_order)
            }
            anchors = [(0, 0)]
            anchors.extend(
                (
                    outer_positions[int(outer_node)],
                    inner_positions[int(inner_node)],
                )
                for outer_node, inner_node
                in self.forced_connectors
            )
            anchors.append(
                (outer_order.size - 1, inner_order.size - 1)
            )
            anchors = sorted(set(anchors))
            if (
                anchors[0] != (0, 0)
                or anchors[-1]
                != (
                    outer_order.size - 1,
                    inner_order.size - 1,
                )
                or any(
                    anchors[position][1]
                    < anchors[position - 1][1]
                    for position in range(1, len(anchors))
                )
            ):
                continue

            sector_rows = []
            sector_scores = []
            try:
                for start, end in zip(anchors, anchors[1:]):
                    rows, score = self._triangulate_sector(
                        outer_order[start[0] : end[0] + 1],
                        inner_order[start[1] : end[1] + 1],
                        connector_is_valid,
                    )
                    sector_rows.append(rows)
                    sector_scores.append(score)
            except ValueError as error:
                errors.append(error)
                continue

            score = self._combine_scores(sector_scores)
            signature = (
                tuple(map(int, outer_order)),
                tuple(map(int, inner_order)),
            )
            if self._score_better(score, best_score) or (
                best_score is not None
                and not self._score_better(best_score, score)
                and not self._score_better(score, best_score)
                and (
                    best_signature is None
                    or signature < best_signature
                )
            ):
                best_rows = np.concatenate(sector_rows, axis=0)
                best_score = score
                best_signature = signature
                best_orders = (
                    outer_order,
                    inner_order,
                    boundary,
                )

        if best_rows is None:
            if errors:
                raise ValueError(
                    "the open circular arcs have no valid constrained "
                    "triangulation"
                ) from errors[-1]
            raise ValueError(
                "the open circular arcs do not bound a simple strip"
            )
        (
            self.chosen_outer_order,
            self.chosen_inner_order,
            self.chosen_boundary_polygon,
        ) = best_orders
        return best_rows

    def _triangulate(self):
        if self.closed:
            return self._triangulate_closed()
        return self._triangulate_open()

    def _geometry_bounds(self, connectivity):
        connectivity = [
            tuple(map(int, item)) for item in connectivity
        ]
        if not connectivity:
            empty = np.empty((0, 2), dtype=np.float64)
            return empty, empty
        minimums = np.empty((len(connectivity), 2))
        maximums = np.empty((len(connectivity), 2))
        for position, item in enumerate(connectivity):
            points = self.xy[np.asarray(item, dtype=np.intp)]
            minimums[position] = np.min(points, axis=0)
            maximums[position] = np.max(points, axis=0)
        return minimums, maximums

    def _bbox_pairs(
        self,
        first_minimums,
        first_maximums,
        second_minimums,
        second_maximums,
        same_collection=False,
    ):
        chunk_size = 256
        for offset in range(
            0, first_minimums.shape[0], chunk_size
        ):
            stop = min(
                offset + chunk_size, first_minimums.shape[0]
            )
            overlaps = np.all(
                (
                    first_minimums[offset:stop, None, :]
                    <= second_maximums[None, :, :]
                    + self.geometry_tolerance
                )
                & (
                    first_maximums[offset:stop, None, :]
                    >= second_minimums[None, :, :]
                    - self.geometry_tolerance
                ),
                axis=2,
            )
            first_positions, second_positions = np.nonzero(
                overlaps
            )
            first_positions += offset
            if same_collection:
                keep = second_positions > first_positions
                first_positions = first_positions[keep]
                second_positions = second_positions[keep]
            yield first_positions, second_positions

    def _validate_completed_strip(self, triangles):
        expected_count = (
            self.chosen_outer_order.size
            + self.chosen_inner_order.size
            if self.closed
            else self.chosen_outer_order.size
            + self.chosen_inner_order.size
            - 2
        )
        if triangles.shape != (expected_count, 4):
            raise RuntimeError(
                "the triangulation produced an unexpected element count"
            )
        if np.any(triangles[:, 2] != triangles[:, 3]):
            raise RuntimeError(
                "the triangulation did not produce padded Tri3 elements"
            )
        triangle_signatures = {
            tuple(sorted(map(int, row[:3]))) for row in triangles
        }
        if len(triangle_signatures) != triangles.shape[0]:
            raise ValueError(
                "the triangulation contains duplicate triangles"
            )

        new_edge_uses = {}
        triangle_area = 0.0
        for row in triangles:
            first, second, third = map(int, row[:3])
            quality = self._triangle_quality(
                first, second, third
            )
            if not np.isfinite(quality) or quality <= 0.0:
                raise ValueError(
                    "the triangulation contains an invalid triangle"
                )
            triangle_area += 0.5 * self._cross(
                self.xy[second] - self.xy[first],
                self.xy[third] - self.xy[first],
            )
            for edge in (
                (first, second),
                (second, third),
                (third, first),
            ):
                new_edge_uses.setdefault(
                    self._edge_key(edge), []
                ).append(edge)

        outer_edges = self._boundary_edges(
            self.chosen_outer_order, self.closed
        )
        inner_edges = self._boundary_edges(
            self.chosen_inner_order, self.closed
        )
        expected_boundary = {
            self._edge_key(edge)
            for edge in (*outer_edges, *inner_edges)
        }
        open_side_keys = set()
        if not self.closed:
            open_side_keys = {
                self._edge_key(
                    (
                        int(self.chosen_outer_order[0]),
                        int(self.chosen_inner_order[0]),
                    )
                ),
                self._edge_key(
                    (
                        int(self.chosen_outer_order[-1]),
                        int(self.chosen_inner_order[-1]),
                    )
                ),
            }
            expected_boundary.update(open_side_keys)

        for key, uses in new_edge_uses.items():
            expected_uses = 1 if key in expected_boundary else 2
            if len(uses) != expected_uses:
                raise ValueError(
                    "the triangulation has an unexpected boundary edge"
                )
            if len(uses) == 2 and uses[0] != uses[1][::-1]:
                raise ValueError(
                    "new triangles sharing an edge must traverse it "
                    "oppositely"
                )
        for key in expected_boundary:
            if len(new_edge_uses.get(key, ())) != 1:
                raise ValueError(
                    "a circular boundary edge is missing from the mesh"
                )
        for connector in self.forced_connectors:
            key = self._edge_key(connector)
            expected_uses = 1 if key in open_side_keys else 2
            if len(new_edge_uses.get(key, ())) != expected_uses:
                raise ValueError(
                    "a pattern connector is missing from the triangulation"
                )

        new_edges = [uses[0] for uses in new_edge_uses.values()]
        new_minimums, new_maximums = self._geometry_bounds(
            new_edges
        )
        for first_positions, second_positions in self._bbox_pairs(
            new_minimums,
            new_maximums,
            new_minimums,
            new_maximums,
            same_collection=True,
        ):
            for first_position, second_position in zip(
                first_positions, second_positions
            ):
                if not self._edges_are_compatible(
                    new_edges[int(first_position)],
                    new_edges[int(second_position)],
                ):
                    raise ValueError(
                        "non-neighbouring generated edges cross or overlap"
                    )

        existing_edges = [
            uses[0] for uses in self.existing_edge_uses.values()
        ]
        existing_minimums, existing_maximums = (
            self._geometry_bounds(existing_edges)
        )
        for new_positions, existing_positions in self._bbox_pairs(
            new_minimums,
            new_maximums,
            existing_minimums,
            existing_maximums,
        ):
            for new_position, existing_position in zip(
                new_positions, existing_positions
            ):
                new_edge = new_edges[int(new_position)]
                existing_edge = existing_edges[
                    int(existing_position)
                ]
                if (
                    self._edge_key(new_edge)
                    == self._edge_key(existing_edge)
                ):
                    continue
                if not self._edges_are_compatible(
                    new_edge, existing_edge
                ):
                    raise ValueError(
                        "generated edges intersect existing mesh geometry"
                    )

        new_perimeters = [
            tuple(map(int, row[:3])) for row in triangles
        ]
        new_element_minimums, new_element_maximums = (
            self._geometry_bounds(new_perimeters)
        )
        (
            existing_element_minimums,
            existing_element_maximums,
        ) = self._geometry_bounds(self.existing_perimeters)
        for new_positions, existing_positions in self._bbox_pairs(
            new_element_minimums,
            new_element_maximums,
            existing_element_minimums,
            existing_element_maximums,
        ):
            for new_position, existing_position in zip(
                new_positions, existing_positions
            ):
                new_perimeter = new_perimeters[int(new_position)]
                existing_perimeter = self.existing_perimeters[
                    int(existing_position)
                ]
                existing_inside_new = any(
                    not self._point_on_polygon(
                        self.xy[node], new_perimeter
                    )
                    and self._point_in_polygon(
                        self.xy[node], new_perimeter
                    )
                    for node in existing_perimeter
                )
                new_inside_existing = any(
                    not self._point_on_polygon(
                        self.xy[node], existing_perimeter
                    )
                    and self._point_in_polygon(
                        self.xy[node], existing_perimeter
                    )
                    for node in new_perimeter
                )
                if existing_inside_new or new_inside_existing:
                    raise ValueError(
                        "generated elements overlap existing mesh elements"
                    )

        if self.closed:
            expected_area = self._polygon_area(
                self.chosen_outer_order
            ) - self._polygon_area(self.chosen_inner_order)
            area_boundaries = (
                self.chosen_outer_order,
                self.chosen_inner_order,
            )
        else:
            expected_area = self._polygon_area(
                self.chosen_boundary_polygon
            )
            area_boundaries = (self.chosen_boundary_polygon,)
        boundary_length = 0.0
        for boundary in area_boundaries:
            points = self.xy[np.asarray(boundary, dtype=np.intp)]
            edges = np.roll(points, -1, axis=0) - points
            boundary_length += float(
                np.sum(np.hypot(edges[:, 0], edges[:, 1]))
            )
        area_tolerance = max(
            self.geometry_tolerance
            * max(self.geometry_tolerance, boundary_length),
            1.0e-10
            * max(
                self.geometry_tolerance**2,
                abs(expected_area),
            ),
        )
        if (
            expected_area <= area_tolerance
            or abs(triangle_area - expected_area) > area_tolerance
        ):
            raise ValueError(
                "triangles do not cover the complete circular strip"
            )

        combined_uses = {
            key: list(uses)
            for key, uses in self.existing_edge_uses.items()
        }
        for key, uses in new_edge_uses.items():
            combined = combined_uses.setdefault(key, [])
            combined.extend(uses)
            if len(combined) > 2:
                raise ValueError(
                    "the completed mesh would contain a non-manifold edge"
                )
            if len(combined) == 2 and combined[0] != combined[1][::-1]:
                raise ValueError(
                    "new elements would traverse an existing shared edge "
                    "incorrectly"
                )

    def _commit(self, triangles):
        element_dtype = self.elements.dtype
        dtype_info = np.iinfo(element_dtype)
        minimum = int(np.min(triangles))
        maximum = int(np.max(triangles))
        if minimum < dtype_info.min or maximum > dtype_info.max:
            element_dtype = np.dtype(np.int64)
        self.mesh.elements = np.concatenate(
            (
                self.elements.astype(element_dtype, copy=False),
                triangles.astype(element_dtype, copy=False),
            ),
            axis=0,
        )


def _mesh_inner_outer_circle(
    mesh: Mesh,
    inner_nodes,
    outer_nodes,
    lines=None,
    closed: bool = True,
) -> Mesh:
    """Triangulate the strip between two concentric circular boundaries.

    Existing nodes remain fixed.  Cross-ring connectors are selected by a
    global monotone matching that first minimizes the worst angular mismatch,
    then the total squared mismatch, and finally maximizes the worst triangle
    quality.  Pattern lines become mandatory connector edges.
    """
    return _CircularStripMesher(
        mesh, inner_nodes, outer_nodes, lines, closed
    ).build()


def _generate_pattern_circle_nodes(
    mesh: Mesh,
    center_x,
    center_y,
    radius,
    element_size,
    lines=None,
) -> np.ndarray:
    """Return a CCW ring whose maximum arc spacing is ``element_size``.

    Finite horizontal and vertical pattern segments contribute exact circle
    intersection anchors.  The arcs between those anchors are subdivided
    independently, which preserves the constrained coordinates without
    creating the very short edges caused by inserting anchors into an already
    uniform ring.  The first node is not repeated at the end.
    """
    if not isinstance(mesh, Mesh):
        raise TypeError("mesh must be a Mesh instance")

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
        element_size = float(element_size)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "center_x, center_y, radius, and element_size must be real numbers"
        ) from error
    if not np.all(np.isfinite(center)) or not np.all(
        np.isfinite([radius, element_size])
    ):
        raise ValueError(
            "center_x, center_y, radius, and element_size must be finite"
        )
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if element_size <= 0.0:
        raise ValueError("element_size must be positive")

    if lines is None:
        pattern_lines = np.empty((0, 2, 2), dtype=np.float64)
    else:
        try:
            pattern_lines = np.asarray(lines, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("lines must have shape (L, 2, 2)") from error
        if pattern_lines.size == 0:
            if pattern_lines.shape not in ((0,), (0, 2, 2)):
                raise ValueError("lines must have shape (L, 2, 2)")
            pattern_lines = np.empty((0, 2, 2), dtype=np.float64)
        elif pattern_lines.ndim != 3 or pattern_lines.shape[1:] != (2, 2):
            raise ValueError("lines must have shape (L, 2, 2)")
    if not np.all(np.isfinite(pattern_lines)):
        raise ValueError("lines must contain finite coordinates")

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

    # Each record retains the exact fixed-axis constraints that produced the
    # anchor.  A vertical and a horizontal segment may legitimately meet at
    # the same circle point; combining their two exact coordinates preserves
    # both constraints with a single node.
    anchor_records: list[dict[str, object]] = []

    def add_anchor(point, fixed_axis, fixed_value):
        point = np.asarray(point, dtype=np.float64)
        matches = [
            position
            for position, record in enumerate(anchor_records)
            if np.hypot(*(point - record["point"])) <= linear_tolerance
        ]
        if len(matches) > 1:
            raise ValueError("pattern lines produce ambiguous circle anchors")
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
            raise ValueError("pattern lines produce conflicting circle anchors")
        constraints[int(fixed_axis)] = float(fixed_value)
        merged = np.asarray(record["point"], dtype=np.float64).copy()
        for axis, value in constraints.items():
            merged[int(axis)] = value
        radial_distance = float(np.hypot(*(merged - center)))
        if abs(radial_distance - radius) > fit_tolerance:
            raise ValueError("pattern line anchors cannot lie on one circle node")
        record["point"] = merged

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
            add_anchor(point, fixed_axis, fixed_value)

    angular_step = element_size / radius
    if np.isnan(angular_step) or angular_step <= 0.0:
        raise ValueError("element_size is too small for the requested radius")
    maximum_node_count = min(int(np.iinfo(np.intp).max), 10_000_000)

    def subdivisions_for_gap(gap):
        with np.errstate(over="ignore", invalid="ignore"):
            arc_length = gap * radius
        requested = (
            arc_length / element_size
            if np.isfinite(arc_length)
            else gap / angular_step
        )
        if not np.isfinite(requested) or requested > maximum_node_count:
            raise ValueError("element_size would require too many circle nodes")
        return max(1, int(np.ceil(requested)))

    if not anchor_records:
        node_count = max(3, subdivisions_for_gap(full_turn))
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
            raise ValueError("pattern lines produce duplicate angular anchors")

        subdivisions = np.asarray(
            [subdivisions_for_gap(float(gap)) for gap in gaps],
            dtype=np.int64,
        )
        node_count = sum(int(value) for value in subdivisions)
        while node_count < 3:
            largest = int(np.argmax(gaps / subdivisions))
            subdivisions[largest] += 1
            node_count += 1
        if node_count > maximum_node_count:
            raise ValueError("element_size would require too many circle nodes")

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
        raise ValueError("element_size produces indistinguishable circle nodes")
    spacing_tolerance = fit_tolerance / radius
    if np.any(generated_gaps > angular_step + spacing_tolerance):
        raise ValueError("generated pattern-circle spacing exceeds element_size")

    if nodes.shape[1] == 2:
        return circle_xy.astype(np.float64, copy=False)
    return np.column_stack(
        (circle_xy, np.zeros(circle_xy.shape[0], dtype=np.float64))
    )


def circle(
    mesh: Mesh,
    x: float,
    y: float,
    radius: float,
    buffer: float,
    lines=None,
    *,
    element_size=None,
) -> Mesh:
    """Insert a conforming circular pattern into a planar mesh.

    The material around ``radius`` is rebuilt as two triangular strips sharing
    one internal pattern-circle edge loop.  ``element_size`` bounds the arc
    spacing of that loop and defaults to ``buffer``.  The original mesh is
    updated only after the complete operation succeeds.
    """
    if not isinstance(mesh, Mesh):
        raise TypeError("mesh must be a Mesh instance")
    try:
        center_x = float(x)
        center_y = float(y)
        circle_radius = float(radius)
        circle_buffer = float(buffer)
        pattern_element_size = (
            circle_buffer if element_size is None else float(element_size)
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "x, y, radius, buffer, and element_size must be real numbers"
        ) from error
    scalar_values = np.asarray(
        [
            center_x,
            center_y,
            circle_radius,
            circle_buffer,
            pattern_element_size,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(scalar_values)):
        raise ValueError("x, y, radius, buffer, and element_size must be finite")
    if circle_buffer <= 0.0:
        raise ValueError("buffer must be positive")
    if circle_radius <= circle_buffer:
        raise ValueError("radius must be greater than buffer")
    if pattern_element_size <= 0.0:
        raise ValueError("element_size must be positive")

    # Work on independent arrays so every downstream geometric or topological
    # error leaves both the identity and values of the caller's arrays intact.
    working_mesh = Mesh(
        nodes=np.array(mesh.nodes, copy=True),
        elements=np.array(mesh.elements, copy=True),
    )

    inner_target_radius = circle_radius - circle_buffer
    outer_target_radius = circle_radius + circle_buffer
    inner_search_margin = min(circle_buffer / 10.0, inner_target_radius / 10.0)
    indices_inner = _search_circle(
        working_mesh,
        center_x,
        center_y,
        inner_target_radius - inner_search_margin,
        type="ALL",
    )
    indices_outer = _search_circle(
        working_mesh,
        center_x,
        center_y,
        outer_target_radius + circle_buffer / 10.0,
        type="PART",
    )
    indices_delete = indices_outer[~np.isin(indices_outer, indices_inner)]

    boundary_loops = _get_boundary(working_mesh, indices=indices_delete)
    if len(boundary_loops) != 2:
        raise ValueError(
            "the selected circular band must have exactly two boundary loops"
        )
    working_nodes = np.asarray(working_mesh.nodes)
    center = np.asarray([center_x, center_y], dtype=np.float64)
    boundary_radii = np.asarray(
        [
            np.mean(
                np.hypot(
                    working_nodes[boundary, 0] - center[0],
                    working_nodes[boundary, 1] - center[1],
                )
            )
            for boundary in boundary_loops
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(boundary_radii)):
        raise ValueError("circular boundary radii must be finite")
    boundary_order = np.argsort(boundary_radii, kind="stable")
    if not boundary_radii[boundary_order[0]] < boundary_radii[boundary_order[1]]:
        raise ValueError("the selected circular boundaries cannot be ordered")
    node_indices_inner = boundary_loops[int(boundary_order[0])]
    node_indices_outer = boundary_loops[int(boundary_order[1])]

    _delete_element(working_mesh, indices=indices_delete)
    node_map = _clear_node(working_mesh)
    node_indices_outer = node_map[node_indices_outer]
    node_indices_inner = node_map[node_indices_inner]
    if np.any(node_indices_outer < 0) or np.any(node_indices_inner < 0):
        raise ValueError("a selected circular boundary was removed unexpectedly")

    outer_circle_start = np.asarray(working_mesh.nodes).shape[0]
    _to_circle(
        working_mesh,
        center_x,
        center_y,
        outer_target_radius,
        node_indices_outer,
        lines=lines,
        closed=True,
    )
    outer_circle_end = np.asarray(working_mesh.nodes).shape[0]
    if outer_circle_end - outer_circle_start != node_indices_outer.size:
        raise ValueError("outer circle did not create the expected node count")
    outer_circle_nodes = np.arange(
        outer_circle_start,
        outer_circle_end,
        dtype=np.int64,
    )

    inner_circle_start = outer_circle_end
    _to_circle(
        working_mesh,
        center_x,
        center_y,
        inner_target_radius,
        node_indices_inner,
        lines=lines,
        closed=True,
    )
    inner_circle_end = np.asarray(working_mesh.nodes).shape[0]
    if inner_circle_end - inner_circle_start != node_indices_inner.size:
        raise ValueError("inner circle did not create the expected node count")
    inner_circle_nodes = np.arange(
        inner_circle_start,
        inner_circle_end,
        dtype=np.int64,
    )

    # ``element_size`` is a maximum, so a coarse request may be refined.  The
    # middle polygon must geometrically contain the complete inner target
    # circle; otherwise a 3- or 4-node pattern ring cannot form an annular
    # strip even though its nominal arc spacing satisfies the request.
    containment_angle = 2.0 * np.arccos(
        min(inner_target_radius / circle_radius, 1.0)
    )
    containment_angle = float(np.nextafter(containment_angle, 0.0))
    if containment_angle <= 0.0:
        raise ValueError(
            "radius and buffer cannot form a resolvable circular strip"
        )
    effective_pattern_size = min(
        pattern_element_size,
        circle_radius * containment_angle,
    )
    pattern_coordinates = _generate_pattern_circle_nodes(
        working_mesh,
        center_x,
        center_y,
        circle_radius,
        effective_pattern_size,
        lines=lines,
    )
    pattern_circle_start = inner_circle_end
    combined_nodes = np.concatenate(
        (np.asarray(working_mesh.nodes), pattern_coordinates),
        axis=0,
    )
    working_mesh.nodes = combined_nodes
    pattern_circle_nodes = np.arange(
        pattern_circle_start,
        combined_nodes.shape[0],
        dtype=np.int64,
    )

    _mesh_inner_outer_circle(
        working_mesh,
        inner_circle_nodes,
        pattern_circle_nodes,
        lines=lines,
        closed=True,
    )
    _mesh_inner_outer_circle(
        working_mesh,
        pattern_circle_nodes,
        outer_circle_nodes,
        lines=lines,
        closed=True,
    )

    completed_nodes = working_mesh.nodes
    completed_elements = working_mesh.elements
    mesh.nodes = completed_nodes
    mesh.elements = completed_elements
    return mesh
