"""Stateless, batched editing operations for planar mixed-element meshes."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..utils.equivalence import _equivalent_node_mapping
from ..utils.topology import _delete_element, _get_boundary, _select_boundary_elements
from ..model import Mesh2D
from ._cleanup import _repair_collapsed_elements
from ._outline import _outlines


_BATCH_SIZE = 65_536


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
        mesh: Mesh2D, x1: float, y1: float, x2: float, y2: float
    ) -> NDArray[np.int64]:
        """Return ascending element IDs with every XY vertex in the rectangle.

        Corners can be supplied in either order. The bounds are inclusive and
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
        inside = np.all((xy >= lower) & (xy <= upper), axis=1)
        selected = np.empty(len(elements), dtype=bool)
        for begin in range(0, len(elements), _BATCH_SIZE):
            end = begin + _BATCH_SIZE
            selected[begin:end] = np.all(inside[elements[begin:end]], axis=1)
        return np.flatnonzero(selected).astype(np.int64, copy=False)

    @staticmethod
    def equivalence(mesh: Mesh2D, tol: float = 0.001) -> Mesh2D:
        """Weld referenced nodes transitively by XY distance <= tol, in place.

        The lowest original node ID supplies each group's XY coordinate with
        Z=0. A collapsed Quad4 becomes Tri3 when adjacent duplicates can be removed;
        fewer than three vertices, non-adjacent duplicates, and zero-area cells
        are discarded. Winding, concavity, and duplicate elements are retained.
        Unreferenced nodes are removed and connectivity is renumbered.

        tol must be finite and non-negative (zero means exact XY equality).
        Referenced XY coordinates must be finite. Invalid input raises before
        mutation; otherwise the original mesh is returned after atomic commit.
        """
        elements = _select_boundary_elements(mesh, None)
        if isinstance(tol, (bool, np.bool_)):
            raise ValueError("tol must be a finite non-negative number")
        try:
            tolerance = float(tol)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("tol must be a finite non-negative number") from error
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tol must be a finite non-negative number")

        used = np.zeros(mesh.node_count, dtype=bool)
        used[elements] = True
        referenced = np.flatnonzero(used)
        if not np.all(np.isfinite(mesh.nodes[referenced, :2])):
            raise ValueError("referenced nodes must contain finite XY coordinates")
        mapping = _equivalent_node_mapping(mesh.nodes, referenced, tolerance)
        repaired = _repair_collapsed_elements(mesh.nodes, mapping[elements])
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
