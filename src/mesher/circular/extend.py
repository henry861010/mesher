"""Extend an existing circular mesh boundary through concentric layers."""

from dataclasses import dataclass

import numpy as np

from ..mesh import Mesh2D
from .projection import _to_circle
from .topology import _clear_node, _get_boundary


@dataclass(frozen=True)
class _CircularExtensionRequest:
    """Validated geometry for one outward circular extension."""

    element_size: float
    center: np.ndarray
    inner_radius: float
    outer_radius: float
    topology: str

    @classmethod
    def from_values(
        cls,
        element_size,
        center_x,
        center_y,
        inner_radius,
        outer_radius,
        topology,
    ):
        """Convert and validate the public scalar inputs."""
        try:
            values = np.asarray(
                [
                    element_size,
                    center_x,
                    center_y,
                    inner_radius,
                    outer_radius,
                ],
                dtype=np.float64,
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "element_size, center_x, center_y, inner_radius, and "
                "outer_radius must be real numbers"
            ) from error

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "element_size, center_x, center_y, inner_radius, and "
                "outer_radius must be finite"
            )

        radial_size, x, y, inner, outer = map(float, values)
        if radial_size <= 0.0:
            raise ValueError("element_size must be positive")
        if inner <= 0.0:
            raise ValueError("inner_radius must be positive")
        if outer <= inner:
            raise ValueError("outer_radius must be greater than inner_radius")
        if not isinstance(topology, str) or topology not in {
            "auto",
            "closed",
            "open",
        }:
            raise ValueError("topology must be 'auto', 'closed', or 'open'")

        return cls(
            element_size=radial_size,
            center=np.asarray([x, y], dtype=np.float64),
            inner_radius=inner,
            outer_radius=outer,
            topology=topology,
        )


@dataclass(frozen=True)
class _CircularBoundary:
    """One ordered inner-radius loop or endpoint-bounded arc."""

    node_indices: np.ndarray
    closed: bool


def _copy_and_validate_mesh(mesh):
    """Return a private working copy after validating the array contract."""
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
    if (
        not np.issubdtype(elements.dtype, np.integer)
        or np.issubdtype(elements.dtype, np.bool_)
    ):
        raise ValueError("elements must have an integer dtype")

    try:
        float_nodes = nodes.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("nodes must be representable as float64") from error
    if not np.all(np.isfinite(float_nodes)):
        raise ValueError("nodes must contain finite float64 coordinates")
    if np.any(elements < 0) or np.any(elements >= nodes.shape[0]):
        raise ValueError("elements contain an out-of-range node index")

    return Mesh2D(
        nodes=np.array(nodes, copy=True),
        elements=np.array(elements, copy=True),
    )


def _circle_tolerance(request):
    """Return a scale-aware tolerance for radius classification."""
    scale_values = [
        1.0,
        request.inner_radius,
        *np.abs(request.center).tolist(),
    ]
    coordinate_scale = max(scale_values)
    return max(
        64.0 * np.finfo(np.float64).eps * coordinate_scale,
        1.0e-9 * request.inner_radius,
    )


def _retain_inner_mesh(mesh, request):
    """Discard all elements outside the requested inner circle."""
    nodes = np.asarray(mesh.nodes)
    elements = np.asarray(mesh.elements)
    float_nodes = nodes.astype(np.float64, copy=False)
    tolerance = _circle_tolerance(request)
    offsets = float_nodes[:, :2] - request.center
    distances = np.hypot(offsets[:, 0], offsets[:, 1])
    node_is_inside = distances <= request.inner_radius + tolerance
    retained = np.all(node_is_inside[elements], axis=1)
    mesh.elements = elements[retained]
    _clear_node(mesh)
    return tolerance


def _circular_edge_runs(loop, node_is_on_circle):
    """Return ordered open chains for cyclic runs of circular edges."""
    edge_is_on_circle = node_is_on_circle & np.roll(node_is_on_circle, -1)
    run_starts = np.flatnonzero(
        edge_is_on_circle & ~np.roll(edge_is_on_circle, 1)
    )
    runs = []
    for raw_start in run_starts:
        position = int(raw_start)
        run = [int(loop[position])]
        while edge_is_on_circle[position]:
            position = (position + 1) % loop.size
            run.append(int(loop[position]))
        runs.append(np.asarray(run, dtype=np.int64))
    return runs


def _validate_boundary_angles(mesh, boundary, request, tolerance):
    """Require strict angular order and the winding implied by topology."""
    nodes = np.asarray(mesh.nodes).astype(np.float64, copy=False)
    coordinates = nodes[boundary.node_indices, :2]
    offsets = coordinates - request.center
    raw_angles = np.arctan2(offsets[:, 1], offsets[:, 0])
    angular_tolerance = max(
        256.0 * np.finfo(np.float64).eps,
        tolerance / request.inner_radius,
    )

    if boundary.closed:
        steps = np.arctan2(
            np.sin(np.roll(raw_angles, -1) - raw_angles),
            np.cos(np.roll(raw_angles, -1) - raw_angles),
        )
        if not (
            np.all(steps > angular_tolerance)
            or np.all(steps < -angular_tolerance)
        ):
            raise ValueError(
                "a closed circular boundary must preserve strict angular order"
            )
        winding_tolerance = max(
            boundary.node_indices.size * angular_tolerance,
            512.0 * np.finfo(np.float64).eps,
        )
        if abs(abs(float(np.sum(steps))) - 2.0 * np.pi) > winding_tolerance:
            raise ValueError(
                "a closed circular boundary must wind exactly once around "
                "the circle center"
            )
        return

    unwrapped_angles = np.unwrap(raw_angles)
    steps = np.diff(unwrapped_angles)
    if not (
        np.all(steps > angular_tolerance)
        or np.all(steps < -angular_tolerance)
    ):
        raise ValueError(
            "an open circular boundary must preserve strict angular order"
        )
    angular_span = abs(float(unwrapped_angles[-1] - unwrapped_angles[0]))
    if (
        angular_span <= angular_tolerance
        or angular_span >= 2.0 * np.pi - angular_tolerance
    ):
        raise ValueError(
            "an open circular boundary must span less than one complete circle"
        )


def _find_inner_boundary(mesh, request, tolerance):
    """Find the unique exposed inner-radius loop or continuous open arc."""
    loops = _get_boundary(mesh)
    nodes = np.asarray(mesh.nodes).astype(np.float64, copy=False)
    candidates = []
    for loop in loops:
        offsets = nodes[loop, :2] - request.center
        distances = np.hypot(offsets[:, 0], offsets[:, 1])
        node_is_on_circle = (
            np.abs(distances - request.inner_radius) <= tolerance
        )
        if np.all(node_is_on_circle):
            if loop.size >= 3:
                candidates.append(
                    _CircularBoundary(
                        node_indices=loop,
                        closed=True,
                    )
                )
            continue

        candidates.extend(
            _CircularBoundary(node_indices=run, closed=False)
            for run in _circular_edge_runs(loop, node_is_on_circle)
        )

    if len(candidates) != 1:
        raise ValueError(
            "the retained mesh must have exactly one exposed circular "
            "boundary or continuous arc on inner_radius"
        )

    boundary = candidates[0]
    if request.topology == "closed" and not boundary.closed:
        raise ValueError(
            "topology='closed' requires a complete circular boundary"
        )
    if request.topology == "open" and boundary.closed:
        raise ValueError(
            "topology='open' requires an incomplete circular boundary"
        )
    _validate_boundary_angles(mesh, boundary, request, tolerance)
    return boundary


def _layer_count(mesh, ring_size, request):
    """Return a safe, representable number of radial layers."""
    thickness = request.outer_radius - request.inner_radius
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        count_value = np.ceil(thickness / request.element_size)
    if not np.isfinite(count_value) or count_value > np.iinfo(np.intp).max:
        raise ValueError("element_size requires too many circular layers")

    count = max(1, int(count_value))
    existing_nodes = np.asarray(mesh.nodes).shape[0]
    maximum_new_nodes = min(
        int(np.iinfo(np.intp).max),
        10_000_000,
        int(np.iinfo(np.int64).max) - existing_nodes,
    )
    if count > maximum_new_nodes // ring_size:
        raise ValueError("element_size requires too many circular nodes")

    radial_step = thickness / count
    if (
        not np.isfinite(radial_step)
        or request.inner_radius + radial_step <= request.inner_radius
        or request.outer_radius - radial_step >= request.outer_radius
    ):
        raise ValueError(
            "element_size produces indistinguishable circular layer radii"
        )
    return count


def _append_circular_layers(mesh, inner_boundary, request):
    """Append equally spaced concentric Quad4 strips through outer_radius."""
    layer_count = _layer_count(
        mesh,
        inner_boundary.node_indices.size,
        request,
    )
    thickness = request.outer_radius - request.inner_radius
    boundary = inner_boundary.node_indices
    previous_radius = request.inner_radius

    for layer in range(1, layer_count + 1):
        target_radius = (
            request.outer_radius
            if layer == layer_count
            else request.inner_radius + thickness * layer / layer_count
        )
        if not np.isfinite(target_radius) or target_radius <= previous_radius:
            raise ValueError(
                "element_size produces indistinguishable circular layer radii"
            )

        node_start = np.asarray(mesh.nodes).shape[0]
        _to_circle(
            mesh,
            request.center[0],
            request.center[1],
            target_radius,
            boundary,
            closed=inner_boundary.closed,
        )
        node_stop = np.asarray(mesh.nodes).shape[0]
        if node_stop - node_start != boundary.size:
            raise RuntimeError(
                "a circular layer created an unexpected number of nodes"
            )
        boundary = np.arange(node_start, node_stop, dtype=np.int64)
        previous_radius = target_radius


def extend_circular_mesh(
    mesh: Mesh2D,
    *,
    element_size: float,
    center_x: float,
    center_y: float,
    inner_radius: float,
    outer_radius: float,
    topology: str = "auto",
) -> Mesh2D:
    """Extend an existing circular boundary outward through concentric layers.

    Existing elements outside ``inner_radius`` are discarded. The remaining
    mesh must expose exactly one complete node loop or continuous open arc on
    ``inner_radius``. That boundary is projected outward repeatedly with a
    constant node count and angular coverage until the final layer reaches
    ``outer_radius``. ``element_size`` limits radial layer spacing only; it
    does not limit circumferential edge length.

    Args:
        mesh: Mesh2D to update transactionally in place.
        element_size: Positive maximum radial distance between adjacent rings.
        center_x: X coordinate shared by all circular layers.
        center_y: Y coordinate shared by all circular layers.
        inner_radius: Positive radius already represented by an exposed loop.
        outer_radius: Final radius, strictly greater than inner_radius.
        topology: Circular-boundary topology mode. ``"auto"`` detects a
            complete loop or continuous open arc. ``"closed"`` and ``"open"``
            require the corresponding topology.

    Returns:
        The same Mesh2D instance after the successful extension.

    Raises:
        TypeError: If mesh is not a Mesh2D instance.
        ValueError: If inputs, retained topology, or generated geometry are
            invalid.

    Note:
        The operation is transactional. An exception leaves both arrays on the
        caller's mesh unchanged.
    """
    if not isinstance(mesh, Mesh2D):
        raise TypeError("mesh must be a Mesh2D instance")

    request = _CircularExtensionRequest.from_values(
        element_size,
        center_x,
        center_y,
        inner_radius,
        outer_radius,
        topology,
    )
    working_mesh = _copy_and_validate_mesh(mesh)
    tolerance = _retain_inner_mesh(working_mesh, request)
    inner_boundary = _find_inner_boundary(
        working_mesh,
        request,
        tolerance,
    )
    _append_circular_layers(working_mesh, inner_boundary, request)

    mesh.nodes = working_mesh.nodes
    mesh.elements = working_mesh.elements
    return mesh


__all__ = ["extend_circular_mesh"]
