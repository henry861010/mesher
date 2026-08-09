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


def _node_to_circle(
    mesh: Mesh,
    center_x,
    center_y,
    radius,
    indices,
) -> Mesh:
    """Project selected nodes radially onto a circle in the XY plane.

    Each selected node is moved to the intersection of the circle and the ray
    that starts at ``(center_x, center_y)`` and passes through that node.  For
    three-dimensional nodes, the Z coordinate is left unchanged.  The mesh is
    updated in place and returned.

    Projection work and temporary storage are both O(K), where K is the number
    of selected nodes.  Node arrays not already stored as writable ``float64``
    are promoted once so projected coordinates cannot be truncated.
    """
    if not isinstance(mesh, Mesh):
        raise TypeError("mesh must be a Mesh instance")

    nodes = np.asarray(mesh.nodes)
    if nodes.ndim != 2 or nodes.shape[1] not in (2, 3):
        raise ValueError("nodes must have shape (N, 2) or (N, 3)")
    if not np.issubdtype(nodes.dtype, np.number) or np.issubdtype(
        nodes.dtype, np.complexfloating
    ):
        raise ValueError("nodes must have a real numeric dtype")

    center_x = float(center_x)
    center_y = float(center_y)
    radius = float(radius)
    if not np.all(np.isfinite([center_x, center_y, radius])):
        raise ValueError("center_x, center_y, and radius must be finite")
    if radius < 0.0:
        raise ValueError("radius must be non-negative")

    node_indices = np.asarray(indices)
    if node_indices.ndim != 1:
        raise ValueError("indices must be a one-dimensional sequence")
    if node_indices.size:
        if not np.issubdtype(node_indices.dtype, np.integer) or np.issubdtype(
            node_indices.dtype, np.bool_
        ):
            raise TypeError("indices must contain integers")
        if np.any(node_indices < 0) or np.any(node_indices >= nodes.shape[0]):
            raise IndexError("indices contain a node index that is out of range")
        node_indices = node_indices.astype(np.intp, copy=False)
    else:
        return mesh

    center = np.array([center_x, center_y], dtype=np.float64)
    # Advanced indexing already returns a copy, so this is also the single
    # persistent K-by-2 work array used throughout the projection.
    projected_xy = nodes[node_indices, :2].astype(np.float64, copy=False)
    if not np.all(np.isfinite(projected_xy)):
        raise ValueError("selected node coordinates must be finite")

    # Normalize after scaling each offset by its largest component.  This
    # avoids overflow in hypot for very large, but still finite, coordinates.
    with np.errstate(over="ignore", invalid="ignore"):
        projected_xy -= center
    overflowed_offset = ~np.all(np.isfinite(projected_xy), axis=1)
    if np.any(overflowed_offset):
        # Finite values can still overflow when opposite-signed float64
        # coordinates are subtracted.  Halving both operands preserves the
        # direction while keeping this rare fallback representable.
        overflowed_xy = nodes[
            node_indices[overflowed_offset], :2
        ].astype(np.float64, copy=False)
        projected_xy[overflowed_offset] = (
            overflowed_xy * 0.5 - center * 0.5
        )
    if not np.all(np.isfinite(projected_xy)):
        raise ValueError("selected node offsets must be representable as float64")

    offset_scale = np.empty(projected_xy.shape[0], dtype=np.float64)
    np.abs(projected_xy[:, 0], out=offset_scale)
    np.maximum(offset_scale, np.abs(projected_xy[:, 1]), out=offset_scale)
    if np.any(offset_scale == 0.0):
        raise ValueError("a selected node cannot coincide with the circle center")

    projected_xy /= offset_scale[:, None]
    np.hypot(projected_xy[:, 0], projected_xy[:, 1], out=offset_scale)
    projected_xy /= offset_scale[:, None]
    projected_xy *= radius
    with np.errstate(over="ignore", invalid="ignore"):
        projected_xy += center
    if not np.all(np.isfinite(projected_xy)):
        raise ValueError("projected node coordinates exceed float64 range")

    # Delay conversion and mutation until every input and projected coordinate
    # has passed validation, so failures leave the mesh unchanged.
    float_nodes = nodes.astype(np.float64, copy=False)
    if not float_nodes.flags.writeable:
        float_nodes = float_nodes.copy()
    float_nodes[node_indices, :2] = projected_xy
    mesh.nodes = float_nodes
    return mesh


def circle(
    mesh: Mesh,
    x: float,
    y: float,
    radius: float,
    buffer: float,
):
    indices_inner = _search_circle(mesh, x, y, radius-buffer)
    indices_outer = _search_circle(mesh, x, y, radius+buffer)
    
    indices_delete = indices_outer[~np.isin(indices_outer, indices_inner)]
    
    indices_list = _get_boundary(mesh, indices=indices_delete)
    node_indices_outer = indices_list[0]
    node_indices_inner = indices_list[1]

    
    mesh = _delete_element(mesh, indices=indices_delete)
    node_map = _clear_node(mesh)
    node_indices_outer = node_map[node_indices_outer]
    node_indices_inner = node_map[node_indices_inner]
    
    view_mesh(mesh)
    
    mesh = _node_to_circle(mesh, x, y, radius-buffer/3, node_indices_inner)
    mesh = _node_to_circle(mesh, x, y, radius+buffer/3, node_indices_outer)
    
    view_mesh(mesh)
    
    return mesh
