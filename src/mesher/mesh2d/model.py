"""Validated data model for planar mixed-element meshes."""

from __future__ import annotations

from enum import IntEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray


class ElementType2D(IntEnum):
    """Supported planar element topologies."""

    TRI3 = 3
    QUAD4 = 4


class Mesh2D:
    """Owned planar mesh arrays with a three-coordinate node invariant.

    Planar algorithms operate on ``nodes[:, :2]``.  The Z coordinate is kept
    so a planar mesh can carry an embedding or become an extrusion source.
    Two-coordinate input is accepted and normalized with ``z = 0``.

    Connectivity has fixed width four.  Tri3 rows use the canonical padded
    representation ``[n0, n1, n2, n2]``; every other row is classified as a
    Quad4.  The object remains mutable because feature operations update it
    transactionally, but every array assigned through its public attributes is
    normalized to an owned, contiguous NumPy array.
    """

    __slots__ = ("_nodes", "_elements")

    def __init__(self, nodes: ArrayLike, elements: ArrayLike) -> None:
        normalized_nodes = _normalize_nodes(nodes)
        normalized_elements = _normalize_elements(elements)
        _validate_connectivity(normalized_elements, len(normalized_nodes))
        self._nodes = normalized_nodes
        self._elements = normalized_elements

    @property
    def nodes(self) -> NDArray[np.float64]:
        return self._nodes

    @nodes.setter
    def nodes(self, values: ArrayLike) -> None:
        # Feature operations replace nodes before connectivity at commit time,
        # so cross-array validation is performed by ``replace_data`` or the
        # subsequent elements assignment rather than by this setter.
        self._nodes = _normalize_nodes(values)

    @property
    def elements(self) -> NDArray[np.int32]:
        return self._elements

    @elements.setter
    def elements(self, values: ArrayLike) -> None:
        elements = _normalize_elements(values)
        _validate_connectivity(elements, len(self._nodes))
        self._elements = elements

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
        self._nodes = normalized_nodes
        self._elements = normalized_elements


def _normalize_nodes(values: ArrayLike) -> NDArray[np.float64]:
    nodes = np.array(values, dtype=np.float64, copy=True)
    if nodes.ndim != 2 or nodes.shape[1] not in (2, 3):
        raise ValueError("nodes must have shape (n, 2) or (n, 3).")
    if nodes.shape[1] == 2:
        nodes = np.column_stack((nodes, np.zeros(len(nodes), dtype=np.float64)))
    return np.ascontiguousarray(nodes)


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
