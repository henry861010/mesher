"""Public orchestration for the second-generation circular imprint."""

from __future__ import annotations

from numpy.typing import ArrayLike

from ...model import Mesh2D
from .local import _run_local_imprint


def imprint_circle(
    mesh: Mesh2D,
    *,
    center: ArrayLike,
    radius: float,
    projection_distance: float,
    minimum_area: float,
    guide_segments: ArrayLike | None = None,
    flat_quad_angle: float = 170.0,
    tolerance=None,
) -> Mesh2D:
    """Project and split a mesh along a circular feature.

    The operation extracts a conservative circle-band patch, projects nearby
    nodes with extended finite-guide barriers on radial paths and fixed-junction
    barriers on guided paths, cleans local topology, and
    splits crossed elements, then replaces local quads having an interior
    angle of at least ``flat_quad_angle`` degrees with two positive-area
    triangles, choosing the diagonal that maximizes the pair's smallest angle.
    Shared-node and merge closure protect the patch
    interface. Remote geometry is preserved; final node compaction may change
    IDs. All stages commit together, so errors leave the caller unchanged.

    Args:
        mesh: Mesh to update in place after every stage succeeds.
        center: Two finite coordinates for the circle center.
        radius: Strictly positive circle radius.
        projection_distance: Non-negative maximum node displacement during
            circle projection. Also extends finite-guide radial path barriers.
        tolerance: Shared absolute geometry policy. Defaults to 0.001 model
            units; controls guide membership, node identity and intersections.
        minimum_area: Non-negative minimum retained element area.
        guide_segments: Optional finite horizontal or vertical guide segments
            with shape ``(L, 2, 2)``.
        flat_quad_angle: Interior-angle threshold in degrees for splitting a
            nearly straight local quad into two triangles. Must be finite and
            in ``(0, 180]``. Defaults to 170 degrees.

    Returns:
        The same ``mesh`` instance after a successful atomic update.

    Raises:
        TypeError: If ``mesh`` is not a :class:`Mesh2D`.
        ValueError: If circle, tolerance, guide, or mesh data is invalid, or a
            generated element is unusable.

    Note:
        Cells cannot be discarded to make the imprint succeed. Unusable source
        geometry, unresolved identity or a final area/quality conflict raises
        a structured GeometryToleranceError before commit. Explicit cleanup is
        available separately when deletion is intended.
    """
    return _run_local_imprint(
        mesh, center=center, radius=radius,
        projection_distance=projection_distance, tolerance=tolerance,
        minimum_area=minimum_area,
        guide_segments=guide_segments,
        flat_quad_angle=flat_quad_angle,
    )


__all__ = ["imprint_circle"]
