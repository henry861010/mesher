"""Public workflow for circular feature imprinting.

The implementation is separated by responsibility into topology, projection,
pattern-ring generation, and concentric-strip meshing modules. This module
keeps the public imprinting workflow small. Private helper re-exports are
retained only for focused unit tests inside this package.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike

from ..mesh import Mesh2D

# Private helper re-exports keep the implementation-focused test surface small.
from .geometry import (
    _minimum_scaled_jacobian,
    _segments_intersect_xy,
    _smooth_circle_nodes,
    _strip_edges,
    _validate_generated_strip,
)
from .pattern import _generate_pattern_circle_nodes
from .projection import _to_circle
from .strip_mesher import (
    _CircularStripMesher,
    _StripScore,
    _mesh_inner_outer_circle,
)
from .topology import (
    _clear_node,
    _delete,
    _delete_element,
    _get_boundary,
    _get_boundary_edge_groups,
    _search_circle,
)


@dataclass(frozen=True)
class _CircularImprintRequest:
    """Validated scalar configuration for one circular insertion.

    Attributes:
        center: Circle center in the XY plane.
        radius: Radius of the internal pattern ring.
        band_width: Total width of the rebuilt circular band.
        target_edge_size: Maximum arc length between pattern-ring nodes.
        minimum_quad_scaled_jacobian: Minimum quality required when two
            triangles are merged into a Quad4 element.
        topology: Requested band topology: auto, closed, or open.
    """

    center: np.ndarray
    radius: float
    band_width: float
    target_edge_size: float
    minimum_quad_scaled_jacobian: float
    topology: str

    @classmethod
    def from_values(
        cls,
        center,
        radius,
        band_width,
        target_edge_size,
        min_quad_scaled_jacobian,
        topology,
    ):
        """Validate user values and build a shallow-frozen request.

        Args:
            center: Two finite coordinates for the circle center.
            radius: Radius of the internal pattern ring.
            band_width: Total width of the band to rebuild.
            target_edge_size: Optional maximum pattern-ring arc length. None uses
                band_width.
            min_quad_scaled_jacobian: Minimum scaled Jacobian for Quad4 merging.
            topology: Requested band topology: auto, closed, or open.

        Returns:
            A shallow-frozen request with float64-compatible scalar values.

        Raises:
            ValueError: If a value is non-numeric, non-finite, or outside its
                supported range.
        """
        try:
            center_coordinates = np.asarray(center, dtype=np.float64)
            if center_coordinates.shape != (2,):
                raise ValueError("center must contain exactly two coordinates")
            center_x, center_y = map(float, center_coordinates)
            circle_radius = float(radius)
            circle_buffer = float(band_width)
            pattern_element_size = (
                circle_buffer if target_edge_size is None else float(target_edge_size)
            )
            minimum_quad_scaled_jacobian = float(min_quad_scaled_jacobian)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "center, radius, band_width, target_edge_size, and "
                "min_quad_scaled_jacobian must contain real numbers"
            ) from error

        scalar_values = np.asarray(
            [
                center_x,
                center_y,
                circle_radius,
                circle_buffer,
                pattern_element_size,
                minimum_quad_scaled_jacobian,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(scalar_values)):
            raise ValueError(
                "center, radius, band_width, target_edge_size, and "
                "min_quad_scaled_jacobian must be finite"
            )
        if circle_buffer <= 0.0:
            raise ValueError("band_width must be positive")
        if circle_radius <= circle_buffer:
            raise ValueError("radius must be greater than band_width")
        if pattern_element_size <= 0.0:
            raise ValueError("target_edge_size must be positive")
        if not 0.0 <= minimum_quad_scaled_jacobian <= 1.0:
            raise ValueError("min_quad_scaled_jacobian must be between 0 and 1")
        if not isinstance(topology, str) or topology not in {
            "auto",
            "closed",
            "open",
        }:
            raise ValueError("topology must be 'auto', 'closed', or 'open'")

        return cls(
            center=np.asarray([center_x, center_y], dtype=np.float64),
            radius=circle_radius,
            band_width=circle_buffer,
            target_edge_size=pattern_element_size,
            minimum_quad_scaled_jacobian=minimum_quad_scaled_jacobian,
            topology=topology,
        )

    @property
    def inner_radius(self):
        """Return the target radius of the inner band boundary."""
        return self.radius - self.band_width / 2.0

    @property
    def outer_radius(self):
        """Return the target radius of the outer band boundary."""
        return self.radius + self.band_width / 2.0


@dataclass(frozen=True)
class _SelectedBand:
    """Element and boundary indices for the circular band being replaced.

    Attributes:
        element_indices: Elements removed from the working mesh.
        inner_boundary: Ordered nodes on the selected inner boundary.
        outer_boundary: Ordered nodes on the selected outer boundary.
        closed: Whether both boundaries are complete loops.
        arc_endpoints: Exact pattern-radius endpoints for an open CCW arc, or
            None for a closed ring.
    """

    element_indices: np.ndarray
    inner_boundary: np.ndarray
    outer_boundary: np.ndarray
    closed: bool
    arc_endpoints: np.ndarray | None


@dataclass(frozen=True)
class _ProjectedBoundaries:
    """New node indices for the projected inner and outer circles.

    Attributes:
        inner: Nodes created on the inner target radius.
        outer: Nodes created on the outer target radius.
    """

    inner: np.ndarray
    outer: np.ndarray


def _copy_mesh(mesh):
    """Create the private working mesh used for transactional updates.

    Args:
        mesh: Source mesh whose arrays must remain untouched until commit.

    Returns:
        A mesh containing independent copies of both source arrays.
    """
    return Mesh2D(
        nodes=np.array(mesh.nodes, copy=True),
        elements=np.array(mesh.elements, copy=True),
    )


def _trace_directed_open_chains(edges, label):
    """Trace non-branching directed edges into deterministic open chains."""
    directed_edges = np.asarray(edges, dtype=np.int64)
    if directed_edges.ndim != 2 or directed_edges.shape[1] != 2:
        raise ValueError(f"{label} edges must have shape (E, 2)")
    if directed_edges.shape[0] == 0:
        return []

    outgoing: dict[int, int] = {}
    incoming: dict[int, int] = {}
    for raw_start, raw_end in directed_edges:
        start = int(raw_start)
        end = int(raw_end)
        if start == end:
            raise ValueError(f"{label} contains a zero-length topological edge")
        if start in outgoing or end in incoming:
            raise ValueError(f"{label} edges form a branching chain")
        outgoing[start] = end
        incoming[end] = start

    start_nodes = sorted(set(outgoing) - set(incoming))
    end_nodes = set(incoming) - set(outgoing)
    if not start_nodes or len(start_nodes) != len(end_nodes):
        raise ValueError(f"{label} edges do not form open chains")

    visited_edges: set[tuple[int, int]] = set()
    chains = []
    for start in start_nodes:
        chain = [start]
        current = start
        while current in outgoing:
            target = outgoing[current]
            edge = (current, target)
            if edge in visited_edges:
                raise ValueError(f"{label} edges contain a directed cycle")
            visited_edges.add(edge)
            chain.append(target)
            current = target
        if current not in end_nodes:
            raise ValueError(f"{label} edges do not terminate consistently")
        chains.append(np.asarray(chain, dtype=np.int64))

    if len(visited_edges) != directed_edges.shape[0]:
        raise ValueError(f"{label} edges contain a closed or disconnected cycle")
    chains.sort(key=lambda chain: tuple(map(int, chain)))
    return chains


def _mean_boundary_radius(nodes, boundary, center):
    """Return the finite mean radius of one boundary chain or loop."""
    coordinates = nodes[np.asarray(boundary, dtype=np.intp), :2]
    radii = np.hypot(
        coordinates[:, 0] - center[0],
        coordinates[:, 1] - center[1],
    )
    mean_radius = float(np.mean(radii))
    if not np.isfinite(mean_radius):
        raise ValueError("circular boundary radii must be finite")
    return mean_radius


def _radial_cut_direction(nodes, chain, center, tolerance):
    """Validate that a domain-boundary chain lies on one center ray."""
    coordinates = nodes[np.asarray(chain, dtype=np.intp), :2]
    offsets = coordinates - center
    distances = np.hypot(offsets[:, 0], offsets[:, 1])
    if np.any(distances <= tolerance):
        raise ValueError(
            "an open circular-band side cannot pass through the circle center"
        )
    reference_position = int(np.argmax(distances))
    direction = offsets[reference_position] / distances[reference_position]
    perpendicular_distances = np.abs(
        offsets[:, 0] * direction[1] - offsets[:, 1] * direction[0]
    )
    forward_distances = offsets @ direction
    if np.any(perpendicular_distances > tolerance) or np.any(
        forward_distances <= tolerance
    ):
        raise ValueError(
            "open circular-band sides must lie on rays from the circle center"
        )
    return direction


def _order_counter_clockwise_arc(nodes, chain, center, angular_tolerance):
    """Orient one angularly monotone open chain counter-clockwise."""
    ordered = np.asarray(chain, dtype=np.int64)
    coordinates = nodes[ordered, :2]
    angles = np.unwrap(
        np.arctan2(
            coordinates[:, 1] - center[1],
            coordinates[:, 0] - center[0],
        )
    )
    steps = np.diff(angles)
    if np.all(steps > angular_tolerance):
        pass
    elif np.all(steps < -angular_tolerance):
        ordered = ordered[::-1].copy()
        coordinates = coordinates[::-1]
        angles = np.unwrap(
            np.arctan2(
                coordinates[:, 1] - center[1],
                coordinates[:, 0] - center[0],
            )
        )
        steps = np.diff(angles)
    else:
        raise ValueError(
            "an open circular boundary must preserve strict angular order"
        )

    span = float(angles[-1] - angles[0])
    if (
        np.any(steps <= angular_tolerance)
        or span <= angular_tolerance
        or span >= 2.0 * np.pi - angular_tolerance
    ):
        raise ValueError(
            "an open circular boundary must span less than one complete circle"
        )
    return ordered, float(angles[0]), span


def _select_open_circular_band(
    mesh,
    request,
    band_elements,
    boundary_loops,
    interface_edges,
    domain_edges,
):
    """Build an open-sector band from interface and radial domain edges."""
    if len(boundary_loops) != 1:
        raise ValueError(
            "an open circular band must have one connected perimeter"
        )

    interface_chains = _trace_directed_open_chains(
        interface_edges,
        "circular-band interface",
    )
    cut_chains = _trace_directed_open_chains(
        domain_edges,
        "circular-band domain",
    )
    if len(interface_chains) != 2 or len(cut_chains) != 2:
        raise ValueError(
            "an open circular band must have two circular interfaces and "
            "two radial sides"
        )

    nodes = np.asarray(mesh.nodes)
    boundary_radii = np.asarray(
        [
            _mean_boundary_radius(nodes, chain, request.center)
            for chain in interface_chains
        ],
        dtype=np.float64,
    )
    boundary_order = np.argsort(boundary_radii, kind="stable")
    if not boundary_radii[boundary_order[0]] < boundary_radii[boundary_order[1]]:
        raise ValueError("the selected circular boundaries cannot be ordered")
    inner_boundary = interface_chains[int(boundary_order[0])]
    outer_boundary = interface_chains[int(boundary_order[1])]

    coordinate_scale = max(
        float(np.max(np.abs(nodes[:, :2]))) if nodes.size else 0.0,
        float(np.max(np.abs(request.center))),
        request.outer_radius,
    )
    radial_tolerance = max(
        1.0e-9 * request.outer_radius,
        256.0 * np.finfo(np.float64).eps * coordinate_scale,
    )
    for cut_chain in cut_chains:
        _radial_cut_direction(
            nodes,
            cut_chain,
            request.center,
            radial_tolerance,
        )

    inner_endpoints = {int(inner_boundary[0]), int(inner_boundary[-1])}
    outer_endpoints = {int(outer_boundary[0]), int(outer_boundary[-1])}
    used_inner_endpoints: set[int] = set()
    used_outer_endpoints: set[int] = set()
    for cut_chain in cut_chains:
        cut_endpoints = {int(cut_chain[0]), int(cut_chain[-1])}
        inner_matches = cut_endpoints & inner_endpoints
        outer_matches = cut_endpoints & outer_endpoints
        if len(inner_matches) != 1 or len(outer_matches) != 1:
            raise ValueError(
                "each radial side must connect one inner and one outer "
                "circular-interface endpoint"
            )
        used_inner_endpoints.update(inner_matches)
        used_outer_endpoints.update(outer_matches)
    if (
        used_inner_endpoints != inner_endpoints
        or used_outer_endpoints != outer_endpoints
    ):
        raise ValueError("open circular-band endpoints cannot be paired")

    angular_tolerance = max(
        256.0 * np.finfo(np.float64).eps,
        radial_tolerance / request.outer_radius,
    )
    outer_boundary, start_angle, span = _order_counter_clockwise_arc(
        nodes,
        outer_boundary,
        request.center,
        angular_tolerance,
    )

    inner_coordinates = nodes[inner_boundary, :2]
    inner_angles = np.arctan2(
        inner_coordinates[:, 1] - request.center[1],
        inner_coordinates[:, 0] - request.center[0],
    )
    inner_positions = np.mod(inner_angles - start_angle, 2.0 * np.pi)
    if np.any(inner_positions > span + angular_tolerance):
        raise ValueError(
            "inner and outer circular interfaces must cover the same sector"
        )
    inner_steps = np.diff(np.unwrap(inner_angles))
    if not (
        np.all(inner_steps > angular_tolerance)
        or np.all(inner_steps < -angular_tolerance)
    ):
        raise ValueError(
            "an open circular boundary must preserve strict angular order"
        )

    outer_endpoint_coordinates = nodes[outer_boundary[[0, -1]], :2]
    endpoint_offsets = outer_endpoint_coordinates - request.center
    endpoint_distances = np.hypot(
        endpoint_offsets[:, 0],
        endpoint_offsets[:, 1],
    )
    arc_endpoints = request.center + request.radius * (
        endpoint_offsets / endpoint_distances[:, None]
    )
    return _SelectedBand(
        element_indices=band_elements,
        inner_boundary=inner_boundary,
        outer_boundary=outer_boundary,
        closed=False,
        arc_endpoints=arc_endpoints,
    )


def _select_circular_band(mesh, request):
    """Find and order the closed loops or open arcs of the rebuild region.

    Args:
        mesh: Working mesh from which the circular band is selected.
        request: Validated circle configuration.

    Returns:
        Selected band elements, ordered boundaries, and resolved topology.

    Raises:
        ValueError: If selection does not produce a supported closed ring or
            center-radial open sector.
    """
    center_x, center_y = request.center
    inner_search_margin = min(
        request.band_width / 10.0,
        request.inner_radius / 10.0,
    )
    inner_elements = _search_circle(
        mesh,
        center_x,
        center_y,
        request.inner_radius - inner_search_margin,
        type="ALL",
    )
    outer_elements = _search_circle(
        mesh,
        center_x,
        center_y,
        request.outer_radius + request.band_width / 10.0,
        type="PART",
    )
    band_elements = outer_elements[
        ~np.isin(outer_elements, inner_elements)
    ]

    boundary_loops = _get_boundary(mesh, indices=band_elements)
    interface_edges, domain_edges = _get_boundary_edge_groups(
        mesh,
        indices=band_elements,
    )

    nodes = np.asarray(mesh.nodes)
    is_closed_band = len(boundary_loops) == 2 and domain_edges.shape[0] == 0
    if is_closed_band:
        if request.topology == "open":
            raise ValueError(
                "topology='open' requires an open circular sector"
            )
        boundary_radii = np.asarray(
            [
                _mean_boundary_radius(nodes, boundary, request.center)
                for boundary in boundary_loops
            ],
            dtype=np.float64,
        )
        boundary_order = np.argsort(boundary_radii, kind="stable")
        if not (
            boundary_radii[boundary_order[0]]
            < boundary_radii[boundary_order[1]]
        ):
            raise ValueError("the selected circular boundaries cannot be ordered")
        return _SelectedBand(
            element_indices=band_elements,
            inner_boundary=boundary_loops[int(boundary_order[0])],
            outer_boundary=boundary_loops[int(boundary_order[1])],
            closed=True,
            arc_endpoints=None,
        )

    if request.topology == "closed":
        raise ValueError(
            "topology='closed' requires exactly two closed circular "
            "boundary loops"
        )
    return _select_open_circular_band(
        mesh,
        request,
        band_elements,
        boundary_loops,
        interface_edges,
        domain_edges,
    )


def _remove_selected_band(mesh, band):
    """Delete the selected band and remap its retained boundary nodes.

    Args:
        mesh: Working mesh to mutate.
        band: Selected band with node indices from the pre-compaction mesh.

    Returns:
        A pair (inner_boundary, outer_boundary) using compacted node indices.

    Raises:
        ValueError: If compaction unexpectedly removes a boundary node.
    """
    _delete_element(mesh, indices=band.element_indices)
    node_map = _clear_node(mesh)
    inner_boundary = node_map[band.inner_boundary]
    outer_boundary = node_map[band.outer_boundary]
    if np.any(inner_boundary < 0) or np.any(outer_boundary < 0):
        raise ValueError("a selected circular boundary was removed unexpectedly")
    return inner_boundary, outer_boundary


def _append_projected_boundary(
    mesh,
    request,
    source_nodes,
    target_radius,
    label,
    guide_segments,
    closed,
):
    """Project one source boundary and return its newly appended node indices.

    Args:
        mesh: Working mesh to extend.
        request: Validated circle configuration.
        source_nodes: Ordered existing boundary-node indices.
        target_radius: Radius of the new circular boundary.
        label: Boundary name used in validation errors.
        guide_segments: Optional axis-aligned pattern segments.
        closed: Whether the source boundary is a closed loop.

    Returns:
        Indices of the nodes appended by the projection.

    Raises:
        ValueError: If projection creates a node count different from the
            one-to-one source boundary contract.
    """
    start = np.asarray(mesh.nodes).shape[0]
    _to_circle(
        mesh,
        request.center[0],
        request.center[1],
        target_radius,
        source_nodes,
        guide_segments=guide_segments,
        closed=closed,
    )
    end = np.asarray(mesh.nodes).shape[0]
    if end - start != source_nodes.size:
        raise ValueError(
            f"{label} circle did not create the expected node count"
        )
    return np.arange(start, end, dtype=np.int64)


def _project_band_boundaries(
    mesh,
    request,
    inner_boundary,
    outer_boundary,
    guide_segments,
    closed,
):
    """Create circular target nodes for both selected band boundaries.

    The outer boundary is projected first to preserve the historical node and
    element ordering of the algorithm.

    Args:
        mesh: Working mesh to extend.
        request: Validated circle configuration.
        inner_boundary: Compacted inner source-boundary indices.
        outer_boundary: Compacted outer source-boundary indices.
        guide_segments: Optional axis-aligned pattern segments.
        closed: Whether both source boundaries are closed loops.

    Returns:
        Newly created inner- and outer-circle node indices.
    """
    print("_project_band_boundaries: outer_nodes")
    outer_nodes = _append_projected_boundary(
        mesh,
        request,
        outer_boundary,
        request.outer_radius,
        "outer",
        guide_segments,
        closed,
    )
    print("_project_band_boundaries: inner_nodes")
    inner_nodes = _append_projected_boundary(
        mesh,
        request,
        inner_boundary,
        request.inner_radius,
        "inner",
        guide_segments,
        closed,
    )
    return _ProjectedBoundaries(inner=inner_nodes, outer=outer_nodes)


def _effective_pattern_size(request):
    """Return an arc-size bound that geometrically contains the inner ring.

    Args:
        request: Validated circle configuration.

    Returns:
        The requested element size, refined when the resulting polygon would
        otherwise cut through the inner target circle.

    Raises:
        ValueError: If the requested radii cannot form a resolvable strip.
    """
    containment_angle = 2.0 * np.arccos(
        min(request.inner_radius / request.radius, 1.0)
    )
    containment_angle = float(np.nextafter(containment_angle, 0.0))
    if containment_angle <= 0.0:
        raise ValueError(
            "radius and band_width cannot form a resolvable circular strip"
        )
    return min(
        request.target_edge_size,
        request.radius * containment_angle,
    )


def _append_pattern_ring(mesh, request, guide_segments, arc_endpoints):
    """Generate and append the internal pattern-circle ring or arc.

    Args:
        mesh: Working mesh to extend.
        request: Validated circle configuration.
        guide_segments: Optional axis-aligned pattern segments.
        arc_endpoints: Exact endpoints for an open arc, or None for a ring.

    Returns:
        Node indices of the appended counter-clockwise pattern ring.
    """
    coordinates = _generate_pattern_circle_nodes(
        mesh,
        request.center[0],
        request.center[1],
        request.radius,
        _effective_pattern_size(request),
        guide_segments=guide_segments,
        arc_endpoints=arc_endpoints,
    )
    start = np.asarray(mesh.nodes).shape[0]
    combined_nodes = np.concatenate(
        (np.asarray(mesh.nodes), coordinates),
        axis=0,
    )
    mesh.nodes = combined_nodes
    return np.arange(start, combined_nodes.shape[0], dtype=np.int64)


def _mesh_pattern_strips(
    mesh,
    projected,
    pattern_nodes,
    guide_segments,
    min_quad_scaled_jacobian,
    closed,
):
    """Fill the inner and outer halves of the rebuilt circular band.

    Args:
        mesh: Working mesh to extend with strip elements.
        projected: Projected inner- and outer-circle node indices.
        pattern_nodes: Internal pattern-circle node indices.
        guide_segments: Optional axis-aligned pattern segments.
        min_quad_scaled_jacobian: Minimum scaled Jacobian for Quad4 merging.
        closed: Whether both boundaries are closed rings rather than arcs.
    """
    for inner_nodes, outer_nodes in (
        (projected.inner, pattern_nodes),
        (pattern_nodes, projected.outer),
    ):
        _mesh_inner_outer_circle(
            mesh,
            inner_nodes,
            outer_nodes,
            guide_segments=guide_segments,
            closed=closed,
            min_quad_scaled_jacobian=min_quad_scaled_jacobian,
        )


def imprint_circle(
    mesh: Mesh2D,
    *,
    center: ArrayLike,
    radius: float,
    band_width: float,
    guide_segments: ArrayLike | None = None,
    target_edge_size: float | None = None,
    min_quad_scaled_jacobian: float = 0.3,
    topology: str = "auto",
) -> Mesh2D:
    """Imprint a conforming circular feature into a planar mesh.

    The material around radius is rebuilt as two mixed Tri3/Quad4 strips that
    share an internal pattern-circle edge loop or open arc. ``band_width`` is
    the total width, so the two projected boundaries lie at radius -
    band_width / 2 and radius + band_width / 2. Pattern guide segments preserve
    exact horizontal or vertical connectors through both strips.

    Args:
        mesh: Mesh2D to update. Nodes may have shape (N, 2) or (N, 3); elements
            must contain four connectivity columns, with Tri3 rows padded as
            [n0, n1, n2, n2].
        center: Two finite coordinates for the pattern-circle center.
        radius: Radius of the internal pattern circle. It must be larger than
            band_width.
        band_width: Positive total width of the circular band to rebuild.
        guide_segments: Optional array-like axis-aligned segments with shape (L, 2, 2).
            Active segment intersections become mandatory connector edges.
        target_edge_size: Optional maximum arc spacing of the pattern ring.
            Defaults to band_width and may be refined to contain the inner target
            circle geometrically.
        min_quad_scaled_jacobian: Minimum scaled Jacobian in [0, 1] required to merge an
            adjacent triangle pair into a Quad4. Defaults to 0.3.
        topology: Circular-band topology mode: ``"auto"`` detects a complete
            ring or a single sector bounded by two rays from ``center``;
            ``"closed"`` and ``"open"`` require the corresponding topology.
            Defaults to ``"auto"``.

    Returns:
        The same mesh instance after a successful circular insertion.

    Raises:
        TypeError: If mesh is not a Mesh2D instance.
        ValueError: If inputs, selected topology, pattern constraints, or
            generated geometry are invalid.

    Note:
        The operation is transactional. All work is performed on copied
        arrays, so any exception leaves both the values and identities of the
        caller's mesh.nodes and mesh.elements unchanged.
    """
    if not isinstance(mesh, Mesh2D):
        raise TypeError("mesh must be a Mesh2D instance")

    request = _CircularImprintRequest.from_values(
        center,
        radius,
        band_width,
        target_edge_size,
        min_quad_scaled_jacobian,
        topology,
    )
    working_mesh = _copy_mesh(mesh)

    started_at = perf_counter()
    selected_band = _select_circular_band(working_mesh, request)
    print(f"_select_circular_band {perf_counter() - started_at:.4f}")

    started_at = perf_counter()
    inner_boundary, outer_boundary = _remove_selected_band(
        working_mesh,
        selected_band,
    )
    print(f"_remove_selected_band {perf_counter() - started_at:.4f}")

    started_at = perf_counter()
    projected = _project_band_boundaries(
        working_mesh,
        request,
        inner_boundary,
        outer_boundary,
        guide_segments,
        selected_band.closed,
    )
    print(f"_project_band_boundaries {perf_counter() - started_at:.4f}")

    started_at = perf_counter()
    pattern_nodes = _append_pattern_ring(
        working_mesh,
        request,
        guide_segments,
        selected_band.arc_endpoints,
    )
    print(f"_append_pattern_ring {perf_counter() - started_at:.4f}")

    started_at = perf_counter()
    _mesh_pattern_strips(
        working_mesh,
        projected,
        pattern_nodes,
        guide_segments,
        request.minimum_quad_scaled_jacobian,
        selected_band.closed,
    )
    print(f"_mesh_pattern_strips {perf_counter() - started_at:.4f}")

    mesh.nodes = working_mesh.nodes
    mesh.elements = working_mesh.elements
    return mesh


__all__ = ["imprint_circle"]
