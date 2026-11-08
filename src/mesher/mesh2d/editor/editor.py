"""Stateless, batched editing operations for planar mixed-element meshes."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..utils.equivalence import _bounded_equivalent_node_mapping
from ..utils.topology import _delete_element, _get_boundary, _select_boundary_elements
from ..model import Mesh2D
from ._cleanup import _repair_collapsed_elements
from ._outline import _outlines
from ...tolerance import GeometryToleranceError, resolve_tolerance, within_distance


_BATCH_SIZE = 65_536


def _area_sign(points):
    """Signed polygon area using local coordinates and an exact fallback."""
    with np.errstate(over="ignore", invalid="ignore"):
        local = points - points[:, :1]
    overflow = np.any(~np.isfinite(local), axis=(1, 2))
    if np.any(overflow):
        large = points[overflow]
        large = large / np.max(np.abs(large), axis=(1, 2))[:, None, None]
        local[overflow] = large - large[:, :1]
    scale = np.max(np.abs(local), axis=(1, 2))
    scaled = np.divide(local, scale[:, None, None], out=np.zeros_like(local),
                       where=scale[:, None, None] != 0)
    following = np.roll(scaled, -1, axis=1)
    positive = scaled[..., 0] * following[..., 1]
    negative = scaled[..., 1] * following[..., 0]
    area = np.sum(positive - negative, axis=1)
    error = 32 * np.finfo(float).eps * np.sum(np.abs(positive) + np.abs(negative), axis=1)
    result = np.sign(area)
    for row in np.flatnonzero(np.abs(area) <= error):
        values = [[Fraction(float(coordinate)) for coordinate in point] for point in points[row]]
        total = sum(a[0] * b[1] - a[1] * b[0]
                    for a, b in zip(values, values[1:] + values[:1]))
        result[row] = (total > 0) - (total < 0)
    return result


def _orientation_sign(a, b, c):
    return _area_sign(np.stack((a, b, c), axis=1))


def _self_intersections(points, elements):
    """Detect opposite-edge intersection for Quad4 rows, without distance slack."""
    result = np.zeros(len(points), dtype=bool)
    distinct = np.all(np.diff(np.sort(elements, axis=1), axis=1) != 0, axis=1)
    ids = np.flatnonzero(distinct)
    selected = points[ids]
    if not len(ids):
        return result
    for i, j, k, l in ((0, 1, 2, 3), (1, 2, 3, 0)):
        a, b, c, d = (selected[:, index] for index in (i, j, k, l))
        first, second = _orientation_sign(a, b, c), _orientation_sign(a, b, d)
        third, fourth = _orientation_sign(c, d, a), _orientation_sign(c, d, b)
        boxes_overlap = np.all(np.maximum(np.minimum(a, b), np.minimum(c, d))
                               <= np.minimum(np.maximum(a, b), np.maximum(c, d)), axis=1)
        result[ids] |= boxes_overlap & (first * second <= 0) & (third * fourth <= 0)
    return result


def _validate_equivalence_topology(nodes, before, after, policy):
    for begin in range(0, len(before), _BATCH_SIZE):
        previous = before[begin:begin + _BATCH_SIZE]
        proposed = after[begin:begin + _BATCH_SIZE]
        changed = np.flatnonzero(np.any(previous != proposed, axis=1))
        if not len(changed):
            continue
        previous, proposed = previous[changed], proposed[changed]
        original_points, updated_points = nodes[previous, :2], nodes[proposed, :2]
        reversed_winding = _area_sign(original_points) != _area_sign(updated_points)
        new_intersection = (_self_intersections(updated_points, proposed)
                            & ~_self_intersections(original_points, previous))
        # A valid convex Quad4 must retain a consistently signed Jacobian.
        # Preserve preexisting concavity; adjacent-node collapse to Tri3 is
        # governed by the triangle area check above rather than quad corners.
        quad = (np.all(np.diff(np.sort(previous, axis=1), axis=1) != 0, axis=1)
                & np.all(np.diff(np.sort(proposed, axis=1), axis=1) != 0, axis=1))
        corner_failure = np.zeros(len(previous), dtype=bool)
        if np.any(quad):
            def convex(points):
                turns = np.stack([_orientation_sign(points[:, index], points[:, (index + 1) % 4],
                                                     points[:, (index + 2) % 4])
                                  for index in range(4)], axis=1)
                return np.all(turns > 0, axis=1) | np.all(turns < 0, axis=1)
            corner_failure[quad] = convex(original_points[quad]) & ~convex(updated_points[quad])
        invalid = reversed_winding | new_intersection | corner_failure
        if np.any(invalid):
            raise GeometryToleranceError(
                "node equivalence would invalidate an element's orientation or Jacobian",
                code="topology_conflict", stage="node_equivalence",
                element_indices=(begin + changed[np.flatnonzero(invalid)[:16]]).tolist(),
                tolerance=policy.absolute,
            )


class EditorMesh2D:
    """NumPy-only mesh APIs with no mesh ownership or cross-call cache.

    Element and node IDs are zero-based array rows. Deletion renumbers element
    rows; equivalence can renumber both nodes and elements. Queries always use
    the current mesh arrays, including direct edits made by the caller.
    """

    __slots__ = ()

    @staticmethod
    def delete_element(mesh: Mesh2D, indices: ArrayLike) -> Mesh2D:
        """Delete integer element IDs in place, keeping all nodes unchanged.

        Repeated IDs are deleted once and an empty selection is a no-op.
        Invalid shapes, non-integer IDs, and negative/out-of-range IDs raise
        ValueError, TypeError, and IndexError respectively before mutation.
        Returns the original mesh.
        """
        return _delete_element(mesh, indices)

    @staticmethod
    def merge(mesh_a: Mesh2D, mesh_b: Mesh2D) -> Mesh2D:
        """Return an independent concatenation, without welding either source.

        Source A's nodes/elements precede source B's. Neither input changes,
        even when both arguments refer to the same mesh. Connectivity offset
        overflow raises ValueError before constructing the result.
        """
        elements_a = _select_boundary_elements(mesh_a, None)
        elements_b = _select_boundary_elements(mesh_b, None)
        shifted = elements_b.astype(np.int64) + mesh_a.node_count
        if shifted.size and np.max(shifted) > np.iinfo(np.int32).max:
            raise ValueError("merged elements contain indices outside the int32 range")
        return Mesh2D(
            nodes=np.concatenate((mesh_a.nodes, mesh_b.nodes)),
            elements=np.concatenate((elements_a, shifted)),
        )

    @staticmethod
    def search_element(
        mesh: Mesh2D, x1: float, y1: float, x2: float, y2: float, *, tolerance=None
    ) -> NDArray[np.int64]:
        """Return ascending element IDs with every XY vertex in the rectangle.

        Corners can be supplied in either order. The distance bounds are inclusive and
        must be finite scalar numbers. Geometry uses XY; all node Z values
        must be zero.
        """
        elements = _select_boundary_elements(mesh, None)
        try:
            bounds = np.asarray([x1, y1, x2, y2], dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("rectangle corners must be finite scalar numbers") from error
        if bounds.shape != (4,) or not np.all(np.isfinite(bounds)):
            raise ValueError("rectangle corners must be finite scalar numbers")
        lower = np.minimum(bounds[:2], bounds[2:])
        upper = np.maximum(bounds[:2], bounds[2:])
        xy = mesh.nodes[:, :2]
        policy = resolve_tolerance(tolerance)
        policy.check_resolution(xy, stage="rectangle_selection")
        inside = within_distance(xy, np.clip(xy, lower, upper), policy)
        selected = np.empty(len(elements), dtype=bool)
        for begin in range(0, len(elements), _BATCH_SIZE):
            end = begin + _BATCH_SIZE
            selected[begin:end] = np.all(inside[elements[begin:end]], axis=1)
        return np.flatnonzero(selected).astype(np.int64, copy=False)

    @staticmethod
    def equivalence(mesh: Mesh2D, *, tolerance=None) -> Mesh2D:
        """Weld bounded XY proximity components, in place after validation.

        The lexicographically lowest XY supplies each group's coordinate with
        Z=0. A collapsed Quad4 becomes Tri3 when adjacent duplicates can be removed;
        fewer than three vertices, non-adjacent duplicates, and zero-area cells
        cause a feature_below_resolution error. Winding and concavity are retained.
        Unreferenced nodes are removed and connectivity is renumbered.

        Every component diameter is at most the positive absolute tolerance.
        Ambiguous transitive chains fail before changing either mesh array.
        Referenced XY coordinates must be finite. Invalid input raises before
        mutation; otherwise the original mesh is returned after atomic commit.
        """
        elements = _select_boundary_elements(mesh, None)
        policy = resolve_tolerance(tolerance)

        used = np.zeros(mesh.node_count, dtype=bool)
        used[elements] = True
        referenced = np.flatnonzero(used)
        if not np.all(np.isfinite(mesh.nodes[referenced, :2])):
            raise ValueError("referenced nodes must contain finite XY coordinates")
        policy.check_resolution(mesh.nodes[referenced, :2], stage="node_equivalence")
        mapping = _bounded_equivalent_node_mapping(mesh.nodes, referenced, policy)
        repaired = _repair_collapsed_elements(mesh.nodes, mapping[elements])
        if len(repaired) != len(elements):
            raise GeometryToleranceError(
                "node equivalence would remove a nonzero mesh feature",
                code="feature_below_resolution", stage="node_equivalence",
                tolerance=policy.absolute, removed_elements=len(elements) - len(repaired))
        _validate_equivalence_topology(mesh.nodes, elements, repaired, policy)
        used.fill(False)
        used[repaired] = True
        retained = np.flatnonzero(used)
        old_to_new = np.full(mesh.node_count, -1, dtype=np.int64)
        old_to_new[retained] = np.arange(len(retained), dtype=np.int64)
        proposed_nodes = mesh.nodes[retained]
        proposed_elements = old_to_new[repaired]
        if np.array_equal(proposed_nodes, mesh.nodes) and np.array_equal(
            proposed_elements, elements
        ):
            return mesh
        mesh.replace_data(nodes=proposed_nodes, elements=proposed_elements)
        return mesh

    @staticmethod
    def outline(mesh: Mesh2D) -> list[NDArray[np.int64]]:
        """Return ordered node-ID loops for all exteriors, holes, and islands.

        Loops have implicit closure and preserve mesh winding. Each starts at
        its lexicographically lowest directed edge and loops are sorted by
        that edge. Invalid connectivity, non-manifold edges, or inconsistent
        shared-edge directions raise ValueError. The mesh is not modified.
        """
        return _get_boundary(mesh)

    @staticmethod
    def outlines(mesh: Mesh2D) -> list[NDArray[np.int64]]:
        """Return complete node-ID loops with existing mesh on their right.

        Exteriors are clockwise and holes counterclockwise, regardless of
        element winding. Loops have implicit closure, retain every boundary
        node, and start at their lowest directed segment. Loop order follows
        ``outline``. Returned int64 arrays own their data; the mesh is unchanged.

        Invalid topology, nonfinite coordinates, zero-length XY segments, or
        boundary cells with zero or nonfinite XY area raise ValueError.
        One loop can be passed directly as ``outline_i`` to
        ``generate_layer_adaptive_mesh`` or ``plan_quad_sections``.
        """
        return _outlines(mesh)


__all__ = ["EditorMesh2D"]
