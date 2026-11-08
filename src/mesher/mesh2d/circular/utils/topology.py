"""Mesh selection, compaction, and boundary-topology helpers."""

import numpy as np

from ....tolerance import resolve_tolerance, circle_membership

from ...model import Mesh2D
from ...utils.topology import (
    _build_boundary_half_edges,
    _clear_node,
    _delete_element,
    _find_boundary_successors,
    _get_boundary,
    _select_boundary_elements,
    _trace_boundary_loops,
)


def _search_circle(
    mesh: Mesh2D,
    x: float,
    y: float,
    radius: float,
    type: str = "ALL",
    tolerance=None,
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
        tolerance: Positive shared geometric inclusion tolerance.

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
    tolerance = resolve_tolerance(tolerance).absolute
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
    node_inside = circle_membership(xy, [center_x, center_y], radius, tolerance)
    element_nodes_inside = node_inside[elements]

    if type == "ALL":
        selected = np.all(element_nodes_inside, axis=1)
    else:
        selected = np.any(element_nodes_inside, axis=1)

    return np.flatnonzero(selected).astype(np.int64, copy=False)


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


def _get_boundary_edge_groups(
    mesh: Mesh2D,
    indices,
) -> tuple[np.ndarray, np.ndarray]:
    """Split selected-submesh boundary edges by their full-mesh ownership.

    A selected boundary edge is an ``interface`` edge when the full mesh has
    an element on its other side that is not selected.  It is a ``domain``
    edge when it was already exposed on the full mesh boundary.  Edge
    directions follow the selected elements' winding.

    Args:
        mesh: Mesh2D containing the selected submesh.
        indices: Unique one-dimensional element indices selecting the submesh.

    Returns:
        A pair ``(interface_edges, domain_edges)``.  Each array has shape
        ``(E, 2)`` and contains directed node-index pairs.

    Raises:
        TypeError: If mesh or an element index has an invalid type.
        ValueError: If arrays, elements, selection, winding, or topology are
            invalid.
        IndexError: If a selected element index is out of range.
    """
    full_elements = _select_boundary_elements(mesh, indices=None)
    selected_elements = _select_boundary_elements(mesh, indices=indices)
    if selected_elements.shape[0] == 0:
        empty = np.empty((0, 2), dtype=np.int64)
        return empty, empty.copy()

    (
        full_half_edges,
        _,
        _,
        full_boundary_half_edges,
    ) = _build_boundary_half_edges(full_elements)
    full_domain_edges = {
        tuple(sorted(map(int, edge)))
        for edge in full_half_edges[full_boundary_half_edges]
    }

    (
        selected_half_edges,
        _,
        _,
        selected_boundary_half_edges,
    ) = _build_boundary_half_edges(selected_elements)
    selected_boundary_edges = selected_half_edges[
        selected_boundary_half_edges
    ].astype(np.int64, copy=False)
    is_domain = np.asarray(
        [
            tuple(sorted(map(int, edge))) in full_domain_edges
            for edge in selected_boundary_edges
        ],
        dtype=bool,
    )
    return (
        selected_boundary_edges[~is_domain],
        selected_boundary_edges[is_domain],
    )
