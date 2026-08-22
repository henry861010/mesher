"""Public workflow and compatibility facade for circular mesh generation.

The implementation is separated by responsibility into topology, projection,
pattern-ring generation, and concentric-strip meshing modules. This module
keeps the public circle workflow small and re-exports the historical private
helpers used by existing callers.
"""

from dataclasses import dataclass

import numpy as np

from mesh import Mesh

# Compatibility re-exports for callers that historically imported helpers
# from circle.circle before the implementation was split by responsibility.
from ._geometry import (
    _minimum_scaled_jacobian,
    _segments_intersect_xy,
    _smooth_circle_nodes,
    _strip_edges,
    _validate_generated_strip,
)
from ._pattern import _generate_pattern_circle_nodes
from ._projection import _to_circle
from ._strip_mesher import (
    _CircularStripMesher,
    _StripScore,
    _mesh_inner_outer_circle,
)
from ._topology import (
    _clear_node,
    _delete,
    _delete_element,
    _get_boundary,
    _search_circle,
)


@dataclass(frozen=True)
class _CircleRequest:
    """Validated scalar configuration for one circular insertion.

    Attributes:
        center: Circle center in the XY plane.
        radius: Radius of the internal pattern ring.
        buffer: Total width of the rebuilt circular band.
        element_size: Maximum arc length between pattern-ring nodes.
        minimum_quad_scaled_jacobian: Minimum quality required when two
            triangles are merged into a Quad4 element.
    """

    center: np.ndarray
    radius: float
    buffer: float
    element_size: float
    minimum_quad_scaled_jacobian: float

    @classmethod
    def from_values(
        cls,
        x,
        y,
        radius,
        buffer,
        element_size,
        jacobian,
    ):
        """Validate user values and build a shallow-frozen request.

        Args:
            x: X coordinate of the circle center.
            y: Y coordinate of the circle center.
            radius: Radius of the internal pattern ring.
            buffer: Total width of the band to rebuild.
            element_size: Optional maximum pattern-ring arc length. None uses
                buffer.
            jacobian: Minimum scaled Jacobian for Quad4 merging.

        Returns:
            A shallow-frozen request with float64-compatible scalar values.

        Raises:
            ValueError: If a value is non-numeric, non-finite, or outside its
                supported range.
        """
        try:
            center_x = float(x)
            center_y = float(y)
            circle_radius = float(radius)
            circle_buffer = float(buffer)
            pattern_element_size = (
                circle_buffer if element_size is None else float(element_size)
            )
            minimum_quad_scaled_jacobian = float(jacobian)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "x, y, radius, buffer, element_size, and jacobian must be "
                "real numbers"
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
                "x, y, radius, buffer, element_size, and jacobian must be "
                "finite"
            )
        if circle_buffer <= 0.0:
            raise ValueError("buffer must be positive")
        if circle_radius <= circle_buffer:
            raise ValueError("radius must be greater than buffer")
        if pattern_element_size <= 0.0:
            raise ValueError("element_size must be positive")
        if not 0.0 <= minimum_quad_scaled_jacobian <= 1.0:
            raise ValueError("jacobian must be between 0 and 1")

        return cls(
            center=np.asarray([center_x, center_y], dtype=np.float64),
            radius=circle_radius,
            buffer=circle_buffer,
            element_size=pattern_element_size,
            minimum_quad_scaled_jacobian=minimum_quad_scaled_jacobian,
        )

    @property
    def inner_radius(self):
        """Return the target radius of the inner band boundary."""
        return self.radius - self.buffer / 2.0

    @property
    def outer_radius(self):
        """Return the target radius of the outer band boundary."""
        return self.radius + self.buffer / 2.0


@dataclass(frozen=True)
class _SelectedBand:
    """Element and boundary indices for the circular band being replaced.

    Attributes:
        element_indices: Elements removed from the working mesh.
        inner_boundary: Ordered nodes on the selected inner boundary.
        outer_boundary: Ordered nodes on the selected outer boundary.
    """

    element_indices: np.ndarray
    inner_boundary: np.ndarray
    outer_boundary: np.ndarray


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
    return Mesh(
        nodes=np.array(mesh.nodes, copy=True),
        elements=np.array(mesh.elements, copy=True),
    )


def _select_circular_band(mesh, request):
    """Find and radially order the two boundaries of the rebuild region.

    Args:
        mesh: Working mesh from which the circular band is selected.
        request: Validated circle configuration.

    Returns:
        Selected band elements and its ordered inner and outer boundaries.

    Raises:
        ValueError: If selection does not produce exactly two finite,
            radially distinct boundary loops.
    """
    center_x, center_y = request.center
    inner_search_margin = min(
        request.buffer / 10.0,
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
        request.outer_radius + request.buffer / 10.0,
        type="PART",
    )
    band_elements = outer_elements[
        ~np.isin(outer_elements, inner_elements)
    ]

    boundary_loops = _get_boundary(mesh, indices=band_elements)
    if len(boundary_loops) != 2:
        raise ValueError(
            "the selected circular band must have exactly two boundary loops"
        )

    nodes = np.asarray(mesh.nodes)
    boundary_radii = np.asarray(
        [
            np.mean(
                np.hypot(
                    nodes[boundary, 0] - request.center[0],
                    nodes[boundary, 1] - request.center[1],
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

    return _SelectedBand(
        element_indices=band_elements,
        inner_boundary=boundary_loops[int(boundary_order[0])],
        outer_boundary=boundary_loops[int(boundary_order[1])],
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
    lines,
):
    """Project one source boundary and return its newly appended node indices.

    Args:
        mesh: Working mesh to extend.
        request: Validated circle configuration.
        source_nodes: Ordered existing boundary-node indices.
        target_radius: Radius of the new circular boundary.
        label: Boundary name used in validation errors.
        lines: Optional axis-aligned pattern segments.

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
        lines=lines,
        closed=True,
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
    lines,
):
    """Create circular target nodes for both selected band boundaries.

    The outer boundary is projected first to preserve the historical node and
    element ordering of the algorithm.

    Args:
        mesh: Working mesh to extend.
        request: Validated circle configuration.
        inner_boundary: Compacted inner source-boundary indices.
        outer_boundary: Compacted outer source-boundary indices.
        lines: Optional axis-aligned pattern segments.

    Returns:
        Newly created inner- and outer-circle node indices.
    """
    outer_nodes = _append_projected_boundary(
        mesh,
        request,
        outer_boundary,
        request.outer_radius,
        "outer",
        lines,
    )
    inner_nodes = _append_projected_boundary(
        mesh,
        request,
        inner_boundary,
        request.inner_radius,
        "inner",
        lines,
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
            "radius and buffer cannot form a resolvable circular strip"
        )
    return min(
        request.element_size,
        request.radius * containment_angle,
    )


def _append_pattern_ring(mesh, request, lines):
    """Generate and append the internal pattern-circle node loop.

    Args:
        mesh: Working mesh to extend.
        request: Validated circle configuration.
        lines: Optional axis-aligned pattern segments.

    Returns:
        Node indices of the appended counter-clockwise pattern ring.
    """
    coordinates = _generate_pattern_circle_nodes(
        mesh,
        request.center[0],
        request.center[1],
        request.radius,
        _effective_pattern_size(request),
        lines=lines,
    )
    start = np.asarray(mesh.nodes).shape[0]
    combined_nodes = np.concatenate(
        (np.asarray(mesh.nodes), coordinates),
        axis=0,
    )
    mesh.nodes = combined_nodes
    return np.arange(start, combined_nodes.shape[0], dtype=np.int64)


def _mesh_pattern_strips(mesh, projected, pattern_nodes, lines, jacobian):
    """Fill the inner and outer halves of the rebuilt circular band.

    Args:
        mesh: Working mesh to extend with strip elements.
        projected: Projected inner- and outer-circle node indices.
        pattern_nodes: Internal pattern-circle node indices.
        lines: Optional axis-aligned pattern segments.
        jacobian: Minimum scaled Jacobian for Quad4 merging.
    """
    for inner_nodes, outer_nodes in (
        (projected.inner, pattern_nodes),
        (pattern_nodes, projected.outer),
    ):
        _mesh_inner_outer_circle(
            mesh,
            inner_nodes,
            outer_nodes,
            lines=lines,
            closed=True,
            jacobian=jacobian,
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
    jacobian: float = 0.3,
) -> Mesh:
    """Insert a conforming circular pattern into a planar mesh.

    The material around radius is rebuilt as two mixed Tri3/Quad4 strips that
    share an internal pattern-circle edge loop. Buffer is the total band width,
    so the two projected boundaries lie at radius - buffer / 2 and
    radius + buffer / 2. Pattern lines preserve exact horizontal or vertical
    connectors through both strips.

    Args:
        mesh: Mesh to update. Nodes may have shape (N, 2) or (N, 3); elements
            must contain four connectivity columns, with Tri3 rows padded as
            [n0, n1, n2, n2].
        x: X coordinate of the pattern-circle center.
        y: Y coordinate of the pattern-circle center.
        radius: Radius of the internal pattern circle. It must be larger than
            buffer.
        buffer: Positive total width of the circular band to rebuild.
        lines: Optional array-like axis-aligned segments with shape (L, 2, 2).
            Active segment intersections become mandatory connector edges.
        element_size: Optional maximum arc spacing of the pattern ring.
            Defaults to buffer and may be refined to contain the inner target
            circle geometrically.
        jacobian: Minimum scaled Jacobian in [0, 1] required to merge an
            adjacent triangle pair into a Quad4. Defaults to 0.3.

    Returns:
        The same mesh instance after a successful circular insertion.

    Raises:
        TypeError: If mesh is not a Mesh instance.
        ValueError: If inputs, selected topology, pattern constraints, or
            generated geometry are invalid.

    Note:
        The operation is transactional. All work is performed on copied
        arrays, so any exception leaves both the values and identities of the
        caller's mesh.nodes and mesh.elements unchanged.
    """
    if not isinstance(mesh, Mesh):
        raise TypeError("mesh must be a Mesh instance")

    request = _CircleRequest.from_values(
        x,
        y,
        radius,
        buffer,
        element_size,
        jacobian,
    )
    working_mesh = _copy_mesh(mesh)

    selected_band = _select_circular_band(working_mesh, request)
    inner_boundary, outer_boundary = _remove_selected_band(
        working_mesh,
        selected_band,
    )
    projected = _project_band_boundaries(
        working_mesh,
        request,
        inner_boundary,
        outer_boundary,
        lines,
    )
    pattern_nodes = _append_pattern_ring(working_mesh, request, lines)
    _mesh_pattern_strips(
        working_mesh,
        projected,
        pattern_nodes,
        lines,
        request.minimum_quad_scaled_jacobian,
    )

    mesh.nodes = working_mesh.nodes
    mesh.elements = working_mesh.elements
    return mesh


__all__ = ["circle"]
