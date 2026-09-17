"""Public orchestration for the second-generation circular imprint."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ...model import Mesh2D
from .utils import (
    _imprint_circle,
    _normalize_circle,
    _normalize_nonnegative,
    _remove_redundant_element,
    _to_circle,
)


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
) -> Mesh2D:
    """Project and split a mesh along a circular feature.

    The operation first projects nearby nodes onto the circle, removes
    redundant topology, and then splits elements crossed by the circle.  All
    stages run on a private mesh copy and are committed together, so an error
    leaves the caller's mesh unchanged.

    Args:
        mesh: Mesh to update in place after every stage succeeds.
        center: Two finite coordinates for the circle center.
        radius: Strictly positive circle radius.
        projection_tolerance: Non-negative radial projection and projected-node
            merge distance.
        guide_tolerance: Non-negative point-to-guide distance.
        merge_tolerance: Non-negative node-equivalence and intersection
            snapping distance.
        minimum_area: Non-negative minimum retained element area.
        guide_segments: Optional finite horizontal or vertical guide segments
            with shape ``(L, 2, 2)``.

    Returns:
        The same ``mesh`` instance after a successful atomic update.

    Raises:
        TypeError: If ``mesh`` is not a :class:`Mesh2D`.
        ValueError: If circle, tolerance, guide, or mesh data is invalid, or a
            generated element is unusable.
    """
    if not isinstance(mesh, Mesh2D):
        raise TypeError("mesh must be a Mesh2D instance")

    center, radius = _normalize_circle(center, radius)
    projection_tolerance = _normalize_nonnegative(
        projection_tolerance,
        "projection_tolerance",
    )
    guide_tolerance = _normalize_nonnegative(
        guide_tolerance,
        "guide_tolerance",
    )
    merge_tolerance = _normalize_nonnegative(
        merge_tolerance,
        "merge_tolerance",
    )
    minimum_area = _normalize_nonnegative(minimum_area, "minimum_area")
    minimum_area = min(1, minimum_area)

    working_mesh = Mesh2D(nodes=mesh.nodes, elements=mesh.elements)
    _to_circle(
        working_mesh,
        center=center,
        radius=radius,
        projection_tolerance=projection_tolerance,
        guide_tolerance=guide_tolerance,
        guide_segments=guide_segments,
    )
    _remove_redundant_element(
        working_mesh,
        merge_tolerance=merge_tolerance,
        minimum_area=minimum_area,
    )
    _imprint_circle(
        working_mesh,
        center=center,
        radius=radius,
        merge_tolerance=merge_tolerance,
    )

    if np.array_equal(mesh.nodes, working_mesh.nodes) and np.array_equal(
        mesh.elements,
        working_mesh.elements,
    ):
        return mesh

    mesh.replace_data(
        nodes=working_mesh.nodes,
        elements=working_mesh.elements,
    )
    return mesh


__all__ = ["imprint_circle"]
