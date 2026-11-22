"""Validated data model for planar mixed-element meshes."""

from __future__ import annotations

from enum import IntEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._winding import ccw_elements


class ElementType2D(IntEnum):
    """Supported planar element topologies."""

    TRI3 = 3
    QUAD4 = 4


class Mesh2D:
    """Owned planar mesh arrays with a three-coordinate node invariant.

    Geometry uses only ``nodes[:, :2]``; the third column must be exactly zero.
    Two-coordinate input is accepted and padded with zeros. Nonzero or nonfinite
    Z values are rejected, without projection or a Z tolerance.

    Arrays remain writable. Direct edits can temporarily violate this contract;
    public mesh operations validate Z before consuming or changing the mesh.

    Each simple element is stored counterclockwise in XY.
    Connectivity has fixed width four.  Tri3 rows use the canonical padded
    representation ``[n0, n1, n2, n2]``; every other row is classified as a
    Quad4.  The object remains mutable because feature operations update it
    transactionally, but every array assigned through its public attributes is
    normalized to an owned, contiguous NumPy array.
    """

    __slots__ = ("_nodes", "_elements")

    def __init__(self, nodes: ArrayLike, elements: ArrayLike) -> None:
        self.replace_data(nodes=nodes, elements=elements)

    @classmethod
    def from_planar(cls, nodes: ArrayLike, elements: ArrayLike, *, tolerance=None):
        """Explicitly normalize finite XYZ plane noise to the exact Z=0 model.

        The constructor remains strict. This adapter owns a copy and only moves
        Z coordinates whose distance to the XY plane is within the given policy.
        """
        from ..tolerance import GeometryToleranceError, resolve_tolerance, within_distance

        policy = resolve_tolerance(tolerance)
        values = np.array(nodes, dtype=float, copy=True)
        if values.ndim != 2 or values.shape[1] not in (2, 3):
            raise ValueError("nodes must have shape (n, 2) or (n, 3).")
        policy.check_resolution(values, stage="planar_normalization")
        if values.shape[1] == 3:
            accepted = within_distance(values[:, 2, None], np.zeros((len(values), 1)), policy)
            if not np.all(accepted):
                index = int(np.flatnonzero(~accepted)[0])
                raise GeometryToleranceError(
                    "node is farther from the XY plane than the geometric tolerance",
                    code="fixed_constraint_conflict", stage="planar_normalization",
                    node_index=index, measured_distance=abs(float(values[index, 2])),
                    tolerance=policy.absolute)
            values[:, 2] = 0.0
        return cls(values, elements)

    @property
    def nodes(self) -> NDArray[np.float64]:
        return self._nodes

    @nodes.setter
    def nodes(self, values: ArrayLike) -> None:
        nodes = _normalize_nodes(values)
        _validate_connectivity(self._elements, len(nodes))
        elements = ccw_elements(nodes, self._elements)
        self._nodes, self._elements = nodes, elements

    def _ensure_ccw(self):
        """Revalidate writable arrays on a successful no-op path."""
        _validate_zero_z(self._nodes)
        _validate_connectivity(self._elements, len(self._nodes))
        self._elements = ccw_elements(self._nodes, self._elements)
        return self

    @property
    def elements(self) -> NDArray[np.int32]:
        return self._elements

    @elements.setter
    def elements(self, values: ArrayLike) -> None:
        _validate_zero_z(self._nodes)
        elements = _normalize_elements(values)
        _validate_connectivity(elements, len(self._nodes))
        self._elements = ccw_elements(self._nodes, elements)

    @property
    def element_types(self) -> NDArray[np.uint8]:
        """Return read-only topology codes aligned with ``elements``."""

        result = np.full(len(self._elements), ElementType2D.QUAD4, dtype=np.uint8)
        if len(result):
            result[self._elements[:, 2] == self._elements[:, 3]] = ElementType2D.TRI3
        result.setflags(write=False)
        return result

    @property
    def node_count(self) -> int:
        return int(len(self._nodes))

    @property
    def element_count(self) -> int:
        return int(len(self._elements))

    def replace_data(self, *, nodes: ArrayLike, elements: ArrayLike) -> None:
        """Atomically replace both arrays after validating them together."""

        normalized_nodes = _normalize_nodes(nodes)
        normalized_elements = _normalize_elements(elements)
        _validate_connectivity(normalized_elements, len(normalized_nodes))
        oriented = ccw_elements(normalized_nodes, normalized_elements)
        self._nodes, self._elements = normalized_nodes, oriented


def _normalize_nodes(values: ArrayLike) -> NDArray[np.float64]:
    nodes = np.array(values, dtype=np.float64, copy=True)
    if nodes.ndim != 2 or nodes.shape[1] not in (2, 3):
        raise ValueError("nodes must have shape (n, 2) or (n, 3).")
    if nodes.shape[1] == 2:
        nodes = np.column_stack((nodes, np.zeros(len(nodes), dtype=np.float64)))
    _validate_zero_z(nodes)
    return np.ascontiguousarray(nodes)


def _validate_zero_z(nodes: NDArray[np.float64]) -> None:
    """Check the Z contract without modifying coordinates or validating XY.

    Call after shape validation for raw arrays. Two-column arrays are already
    XY; extra columns accepted by the extrusion adapter retain their semantics.
    This check includes unused nodes and deliberately has no tolerance.
    """
    if nodes.shape[1] >= 3:
        invalid = nodes[:, 2] != 0.0  # Also rejects NaN and either infinity.
        if np.any(invalid):
            index = int(np.argmax(invalid))
            raise ValueError(
                f"Mesh2D requires Z=0; node {index} has Z={float(nodes[index, 2])!r}"
            )


def _normalize_elements(values: ArrayLike) -> NDArray[np.int32]:
    source = np.asarray(values)
    if source.ndim != 2 or source.shape[1] != 4:
        raise ValueError("elements must have shape (m, 4).")
    if not np.issubdtype(source.dtype, np.integer):
        raise ValueError("elements must contain integer node indices.")
    if source.size:
        minimum = int(np.min(source))
        maximum = int(np.max(source))
        limits = np.iinfo(np.int32)
        if minimum < limits.min or maximum > limits.max:
            raise ValueError("elements contain indices outside the int32 range.")
    return np.ascontiguousarray(source, dtype=np.int32).copy()


def _validate_connectivity(elements: NDArray[np.int32], node_count: int) -> None:
    if elements.size and (np.any(elements < 0) or np.any(elements >= node_count)):
        raise ValueError("elements contain an out-of-range node index.")


__all__ = ["ElementType2D", "Mesh2D"]
