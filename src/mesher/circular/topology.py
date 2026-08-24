"""Mesh selection, compaction, and boundary-topology helpers."""

import numpy as np

from ..mesh import Mesh2D


def _search_circle(
    mesh: Mesh2D,
    x: float,
    y: float,
    radius: float,
    type: str = "ALL",
    tolerance: float = 0.01,
) -> np.ndarray:
    """Return elements selected by a circle in the XY plane.

    ALL requires every element node to be inside; PART requires at least one.
    Nodes on the circle, including tolerance, count as inside.

    Args:
        mesh: Mesh2D providing node coordinates and element connectivity.
        x: X coordinate of the selection center.
        y: Y coordinate of the selection center.
        radius: Non-negative circle radius.
        type: Selection mode, either ALL or PART.
        tolerance: Non-negative radial inclusion tolerance.

    Returns:
        Selected element-row indices as an int64 array.

    Raises:
        AttributeError: If mesh does not provide nodes and elements.
        TypeError: If a scalar cannot be converted to float.
        ValueError: If mesh arrays, selection mode, converted scalar values,
            or connectivity are invalid.
        OverflowError: If scalar conversion exceeds the float range.
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


def _delete_element(mesh: Mesh2D, indices) -> Mesh2D:
    """Remove selected element rows from a mesh in place.

    Node coordinates are left untouched; removing unused nodes would require
    remapping every node index in the remaining elements.

    Args:
        mesh: Mesh2D whose element rows are removed.
        indices: One-dimensional integer sequence of element indices.

    Returns:
        The same mesh instance.

    Raises:
        TypeError: If mesh or an index has an invalid type.
        ValueError: If connectivity or the index sequence shape is invalid.
        IndexError: If an element index is out of range.
    """
    if not isinstance(mesh, Mesh2D):
        raise TypeError("mesh must be a Mesh2D instance")

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


def _delete(mesh: Mesh2D, indices) -> Mesh2D:
    """Provide the backward-compatible alias for :func:`_delete_element`.

    Args:
        mesh: Mesh2D whose element rows are removed.
        indices: One-dimensional integer sequence of element indices.

    Returns:
        The same mesh instance.

    Raises:
        TypeError: If mesh or an index has an invalid type.
        ValueError: If connectivity or the index sequence shape is invalid.
        IndexError: If an element index is out of range.
    """
    return _delete_element(mesh, indices)


def _clear_node(mesh: Mesh2D) -> np.ndarray:
    """Remove unreferenced nodes and return their old-to-new index mapping.

    The mesh is updated in place.  The returned one-dimensional array has one
    entry for every node in the original mesh, so retained node indices can be
    remapped through the result. Entries for removed nodes are -1.

    Args:
        mesh: Mesh2D to compact in place.

    Returns:
        An intp mapping from every old node index to its new index or -1.

    Raises:
        TypeError: If mesh is not a Mesh2D instance.
        ValueError: If mesh arrays or connectivity are invalid.
    """
    if not isinstance(mesh, Mesh2D):
        raise TypeError("mesh must be a Mesh2D instance")

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


def _select_boundary_elements(mesh, indices):
    """Validate a mesh and select the elements used for boundary tracing.

    Args:
        mesh: Mesh2D providing nodes and mixed Tri3/Quad4 connectivity.
        indices: Optional unique one-dimensional element-index sequence.

    Returns:
        The validated and selected element rows.

    Raises:
        TypeError: If mesh or an element index has an invalid type.
        ValueError: If mesh arrays, connectivity, or selection are invalid.
        IndexError: If a selected element index is out of range.
    """
    if not isinstance(mesh, Mesh2D):
        raise TypeError("mesh must be a Mesh2D instance")

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
                element_indices.dtype,
                np.integer,
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
    return elements


def _build_boundary_half_edges(elements):
    """Build half-edge successors, twins, and exposed-edge identifiers.

    Args:
        elements: Non-empty validated mixed Tri3/Quad4 connectivity.

    Returns:
        Half-edge endpoint rows, next-perimeter half-edge identifiers, twin
        identifiers, and identifiers of exposed boundary half-edges.

    Raises:
        ValueError: If an element perimeter is invalid, an edge is
            non-manifold, or paired elements use inconsistent directions.
    """
    is_triangle = elements[:, 2] == elements[:, 3]
    sorted_triangle_nodes = np.sort(elements[:, :3], axis=1)
    triangle_is_valid = np.all(
        np.diff(sorted_triangle_nodes, axis=1) != 0,
        axis=1,
    )
    sorted_quad_nodes = np.sort(elements, axis=1)
    quad_is_valid = np.all(
        np.diff(sorted_quad_nodes, axis=1) != 0,
        axis=1,
    )
    if np.any(is_triangle & ~triangle_is_valid) or np.any(
        ~is_triangle & ~quad_is_valid
    ):
        raise ValueError(
            "each element must be a Quad4 with four distinct nodes or a "
            "Tri3 encoded as [n0, n1, n2, n2]"
        )

    edge_starts = elements
    edge_ends = np.roll(elements, -1, axis=1)
    valid_edge = edge_starts != edge_ends
    half_edges = np.column_stack(
        (edge_starts[valid_edge], edge_ends[valid_edge])
    ).astype(np.int64, copy=False)

    half_edge_count = half_edges.shape[0]
    half_edge_ids = np.full(elements.shape, -1, dtype=np.int64)
    half_edge_ids[valid_edge] = np.arange(half_edge_count, dtype=np.int64)

    next_ids = np.full(elements.shape, -1, dtype=np.int64)
    for offset in range(1, elements.shape[1] + 1):
        candidate = np.roll(half_edge_ids, -offset, axis=1)
        use_candidate = (
            (half_edge_ids >= 0)
            & (next_ids < 0)
            & (candidate >= 0)
        )
        next_ids[use_candidate] = candidate[use_candidate]
    next_half_edge = next_ids[valid_edge]

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
    return half_edges, next_half_edge, twin, boundary_half_edges


def _find_boundary_successors(
    half_edges,
    next_half_edge,
    twin,
    boundary_half_edges,
):
    """Map every exposed half-edge to the next edge on its boundary loop.

    Args:
        half_edges: Directed half-edge endpoint rows.
        next_half_edge: Next perimeter edge for each half-edge.
        twin: Opposite half-edge identifier or -1 for exposed edges.
        boundary_half_edges: Identifiers of all exposed half-edges.

    Returns:
        An array mapping boundary half-edge identifiers to their loop successor.

    Raises:
        ValueError: If an element fan does not terminate or exposed edges do
            not form a closed, non-branching permutation.
    """
    half_edge_count = half_edges.shape[0]
    boundary_successor = next_half_edge[boundary_half_edges].copy()
    for _ in range(half_edge_count):
        crosses_interior = twin[boundary_successor] >= 0
        if not np.any(crosses_interior):
            break
        crossed = twin[boundary_successor[crosses_interior]]
        boundary_successor[crosses_interior] = next_half_edge[crossed]
    else:
        raise ValueError(
            "boundary topology contains a non-terminating element fan"
        )

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
        raise ValueError(
            "boundary edges do not form closed, non-branching loops"
        )

    successor = np.full(half_edge_count, -1, dtype=np.int64)
    successor[boundary_half_edges] = boundary_successor
    return successor


def _trace_boundary_loops(half_edges, boundary_half_edges, successor):
    """Trace and canonicalize every loop in an exposed half-edge permutation.

    Args:
        half_edges: Directed half-edge endpoint rows.
        boundary_half_edges: Identifiers of all exposed half-edges.
        successor: Boundary half-edge successor mapping.

    Returns:
        Deterministically ordered int64 node loops with preserved winding.

    Raises:
        ValueError: If a traced chain fails to close at its starting edge.
    """
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

        loop_nodes = half_edges[
            np.asarray(loop_half_edges, dtype=np.int64),
            0,
        ]
        start_position = min(
            range(loop_nodes.size),
            key=lambda position: (
                int(loop_nodes[position]),
                int(loop_nodes[(position + 1) % loop_nodes.size]),
            ),
        )
        boundaries.append(
            np.roll(loop_nodes, -start_position).astype(
                np.int64,
                copy=False,
            )
        )

    boundaries.sort(key=lambda boundary: tuple(boundary.tolist()))
    return boundaries


def _get_boundary(mesh: Mesh2D, indices=None) -> list[np.ndarray]:
    """Return every ordered boundary loop as node indices.

    An element edge belongs to the boundary when no other selected element
    shares it. Each result contains one loop in traversal order with implicit
    closure. Winding is preserved, so a counter-clockwise mesh produces
    counter-clockwise outer loops and clockwise hole loops. Padded Tri3 rows
    and Quad4 rows are both supported.

    Args:
        mesh: Mesh2D whose selected submesh is traced.
        indices: Optional unique one-dimensional element-index sequence.

    Returns:
        Deterministically ordered int64 boundary loops.

    Raises:
        TypeError: If mesh or an element index has an invalid type.
        ValueError: If arrays, elements, selection, winding, or topology are
            invalid.
        IndexError: If a selected element index is out of range.
    """
    elements = _select_boundary_elements(mesh, indices)
    if elements.shape[0] == 0:
        return []

    (
        half_edges,
        next_half_edge,
        twin,
        boundary_half_edges,
    ) = _build_boundary_half_edges(elements)
    if boundary_half_edges.size == 0:
        return []

    successor = _find_boundary_successors(
        half_edges,
        next_half_edge,
        twin,
        boundary_half_edges,
    )
    return _trace_boundary_loops(
        half_edges,
        boundary_half_edges,
        successor,
    )
