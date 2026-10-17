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
    projection_tolerance: float,
    guide_tolerance: float,
    merge_tolerance: float,
    minimum_area: float,
    guide_segments: ArrayLike | None = None,
    flat_quad_angle: float = 170.0,
) -> Mesh2D:
    """Project and split a mesh along a circular feature.

    The operation extracts a conservative circle-band patch, projects nearby
    nodes without crossing fixed guide junctions, cleans local topology, and
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
        projection_tolerance: Non-negative radial projection and projected-node
            merge distance.
        guide_tolerance: Non-negative absolute distance tolerance, in model
            coordinate units, for deciding whether a node lies on a finite
            guide segment. This is an on-segment membership tolerance, not a
            radius for finding nearby nodes.
        merge_tolerance: Non-negative node-equivalence and intersection
            snapping distance.
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
    """
    return _run_local_imprint(
        mesh, center=center, radius=radius,
        projection_tolerance=projection_tolerance, guide_tolerance=guide_tolerance,
        merge_tolerance=merge_tolerance, minimum_area=minimum_area,
        guide_segments=guide_segments,
        flat_quad_angle=flat_quad_angle,
    )


__all__ = ["imprint_circle"]
