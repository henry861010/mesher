"""Named FEM boundary selections for three-dimensional meshes."""

from __future__ import annotations

from typing import Literal

import numpy as np

from ..model import Mesh3D, MeshSet
from ...tolerance import GeometryToleranceError, resolve_tolerance, within_distance


_MODEL_TYPES = ("full", "upper_right_quarter", "upper_half", "right_half")


class EditorMesh3D:
    """NumPy-only mesh operations with no mesh ownership or cross-call cache."""

    __slots__ = ()

    @staticmethod
    def set_boundary(
        mesh3d: Mesh3D,
        model_type: Literal["full", "upper_right_quarter", "upper_half", "right_half"],
        *,
        tolerance=None,
        exact_z=False,
    ) -> Mesh3D:
        """Replace ``fix_x``, ``fix_y`` and ``fix_z`` node sets in place.

        All three sets contain the bottom cross: its center and the nearest
        two nodes along each retained arm. The center is inferred from the
        bottom (minimum-Z) XY bounds: full uses both midpoints (nine nodes),
        upper-half uses midpoint-X/minimum-Y (seven nodes), right-half uses
        minimum-X/midpoint-Y (seven nodes), and quarter uses both minima (five
        nodes). Thus reduced models keep the original full model's symmetry
        center and omit arms outside their retained domain.

        Symmetry additionally fixes the entire minimum-X face in ``fix_x``
        for right-half/quarter models and the entire minimum-Y face in
        ``fix_y`` for upper-half/quarter models, across every Z layer.
        Left/right mean -X/+X; down/up mean -Y/+Y.

        Plane and center matching use the absolute geometry tolerance (default
        0.001). Missing or ambiguous centers and arms with fewer than two nodes
        raise ``GeometryToleranceError`` before changing any sets. No nodes
        are created or moved. Other sets and all mesh arrays are preserved;
        the original mesh is returned. Set indices are zero-based node rows.
        ``exact_z=True`` selects only the exact minimum Z plane while retaining
        the XY tolerance, so nearby thin layers cannot enter the bottom sets.
        """
        if not isinstance(mesh3d, Mesh3D):
            raise TypeError("mesh3d must be a Mesh3D instance")
        if not isinstance(model_type, str) or model_type not in _MODEL_TYPES:
            raise ValueError(f"model_type must be one of: {', '.join(_MODEL_TYPES)}")

        policy = resolve_tolerance(tolerance)
        nodes = mesh3d.nodes
        if not len(nodes):
            raise ValueError("mesh3d must contain nodes to set boundary conditions")
        policy.check_resolution(nodes[:, :2] if exact_z else nodes, stage="boundary_selection")

        lower = np.min(nodes, axis=0)
        bottom = np.flatnonzero(nodes[:, 2] == lower[2] if exact_z else
                               within_distance(nodes[:, 2:3], lower[2], policy))
        xy = nodes[bottom, :2]
        xy_min, xy_max = np.min(xy, axis=0), np.max(xy, axis=0)
        center_xy = xy_min / 2 + xy_max / 2
        restrict_x = model_type in ("right_half", "upper_right_quarter")
        restrict_y = model_type in ("upper_half", "upper_right_quarter")
        if restrict_x:
            center_xy[0] = xy_min[0]
        if restrict_y:
            center_xy[1] = xy_min[1]
        bottom_nodes = nodes[bottom]
        matches = bottom[within_distance(bottom_nodes, np.r_[center_xy, lower[2]], policy)]
        if len(matches) != 1:
            raise GeometryToleranceError(
                "bottom cross requires exactly one node at its center",
                code="ambiguous_geometry" if len(matches) else "fixed_constraint_conflict",
                stage="boundary_selection", model_type=model_type,
                center=[*center_xy.tolist(), float(lower[2])],
                matching_nodes=matches.tolist(), tolerance=policy.absolute,
            )

        center_id = int(matches[0])
        center = nodes[center_id]
        arms = [(0, 1, "right"), (1, 1, "up")]
        if not restrict_x:
            arms.append((0, -1, "left"))
        if not restrict_y:
            arms.append((1, -1, "down"))
        cross = [center_id]
        for axis, sign, direction in arms:
            # Distance to the ray's fixed coordinates keeps all arms on the
            # minimum-Z plane and within T of the central axis.
            fixed_axes = [1 - axis, 2]
            aligned = within_distance(
                bottom_nodes[:, fixed_axes], [center[1 - axis], lower[2]], policy,
            )
            same_position = within_distance(bottom_nodes[:, axis:axis + 1], center[axis], policy)
            distance = sign * (bottom_nodes[:, axis] - center[axis])
            candidates = bottom[aligned & ~same_position & (distance > 0)]
            if len(candidates) < 2:
                raise GeometryToleranceError(
                    f"bottom cross requires two nodes in the {direction} arm",
                    code="fixed_constraint_conflict", stage="boundary_selection",
                    model_type=model_type, direction=direction,
                    available_nodes=int(len(candidates)), tolerance=policy.absolute,
                )
            order = np.lexsort((candidates, sign * (nodes[candidates, axis] - center[axis])))
            cross.extend(candidates[order[:2]].tolist())

        selections = {name: cross for name in ("fix_x", "fix_y", "fix_z")}
        if restrict_x:
            face = np.flatnonzero(within_distance(nodes[:, 0:1], lower[0], policy))
            selections["fix_x"] = np.union1d(cross, face)
        if restrict_y:
            face = np.flatnonzero(within_distance(nodes[:, 1:2], lower[1], policy))
            selections["fix_y"] = np.union1d(cross, face)

        boundary = {name: MeshSet(kind="node", indices=indices)
                    for name, indices in selections.items()}
        mesh3d.sets.update(boundary)
        return mesh3d
