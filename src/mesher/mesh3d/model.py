"""Validated data model for mixed solid meshes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


class ElementType3D(IntEnum):
    """Supported solid element topologies."""

    WEDGE6 = 6
    HEX8 = 8


@dataclass(frozen=True, init=False)
class Mesh3D:
    """Owned 3D mesh arrays with fixed-width solid connectivity.

    Hex8 rows use eight distinct node slots.  Wedge6 rows keep the repository
    CDB contract by repeating the third bottom and top nodes in slots three and
    seven.  ``element_types`` makes that topology explicit for consumers such
    as visualization backends.
    """

    nodes: NDArray[np.float64]
    elements: NDArray[np.int32]
    element_types: NDArray[np.uint8]
    element_component_ids: NDArray[np.int32]
    component_ids_by_name: dict[str, int]

    def __init__(
        self,
        *,
        nodes: ArrayLike,
        elements: ArrayLike,
        element_component_ids: ArrayLike,
        component_ids_by_name: Mapping[str, int],
        element_types: ArrayLike | None = None,
    ) -> None:
        normalized_nodes = _normalize_nodes(nodes)
        normalized_elements = _normalize_elements(elements)
        _validate_connectivity(normalized_elements, len(normalized_nodes))
        normalized_types = _normalize_element_types(
            element_types,
            normalized_elements,
        )
        normalized_components = _normalize_element_components(
            element_component_ids,
            len(normalized_elements),
        )
        normalized_table = _normalize_component_table(component_ids_by_name)

        object.__setattr__(self, "nodes", normalized_nodes)
        object.__setattr__(self, "elements", normalized_elements)
        object.__setattr__(self, "element_types", normalized_types)
        object.__setattr__(self, "element_component_ids", normalized_components)
        object.__setattr__(self, "component_ids_by_name", normalized_table)

    @property
    def node_count(self) -> int:
        return int(len(self.nodes))

    @property
    def element_count(self) -> int:
        return int(len(self.elements))

    @property
    def component_count(self) -> int:
        return len(self.component_ids_by_name)


def _normalize_nodes(values: ArrayLike) -> NDArray[np.float64]:
    nodes = np.array(values, dtype=np.float64, copy=True)
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError("nodes must have shape (n, 3).")
    return np.ascontiguousarray(nodes)


def _normalize_elements(values: ArrayLike) -> NDArray[np.int32]:
    source = np.asarray(values)
    if source.ndim != 2 or source.shape[1] != 8:
        raise ValueError("elements must have shape (m, 8).")
    if not np.issubdtype(source.dtype, np.integer):
        raise ValueError("elements must contain integer node indices.")
    if source.size:
        limits = np.iinfo(np.int32)
        if int(np.min(source)) < limits.min or int(np.max(source)) > limits.max:
            raise ValueError("elements contain indices outside the int32 range.")
    return np.ascontiguousarray(source, dtype=np.int32).copy()


def _validate_connectivity(elements: NDArray[np.int32], node_count: int) -> None:
    if elements.size and (np.any(elements < 0) or np.any(elements >= node_count)):
        raise ValueError("elements contain an out-of-range node index.")


def _infer_element_types(elements: NDArray[np.int32]) -> NDArray[np.uint8]:
    result = np.full(len(elements), ElementType3D.HEX8, dtype=np.uint8)
    if len(result):
        wedge_mask = (elements[:, 2] == elements[:, 3]) & (
            elements[:, 6] == elements[:, 7]
        )
        result[wedge_mask] = ElementType3D.WEDGE6
    return result


def _normalize_element_types(
    values: ArrayLike | None,
    elements: NDArray[np.int32],
) -> NDArray[np.uint8]:
    inferred = _infer_element_types(elements)
    if values is None:
        return inferred

    element_types = np.array(values, dtype=np.uint8, copy=True)
    if element_types.ndim != 1 or len(element_types) != len(elements):
        raise ValueError("element_types must have shape (m,).")
    supported = np.isin(
        element_types,
        [ElementType3D.WEDGE6, ElementType3D.HEX8],
    )
    if not np.all(supported):
        raise ValueError("element_types contains an unsupported topology.")
    if not np.array_equal(element_types, inferred):
        raise ValueError("element_types do not match the fixed-width connectivity.")
    return np.ascontiguousarray(element_types)


def _normalize_element_components(
    values: ArrayLike,
    element_count: int,
) -> NDArray[np.int32]:
    source = np.asarray(values)
    if source.ndim != 1:
        raise ValueError("element_component_ids must have shape (m,).")
    if len(source) != element_count:
        raise ValueError("element_component_ids length must match element count.")
    if not np.issubdtype(source.dtype, np.integer):
        raise ValueError("element_component_ids must contain integers.")
    return np.ascontiguousarray(source, dtype=np.int32).copy()


def _normalize_component_table(values: Mapping[str, int]) -> dict[str, int]:
    result: dict[str, int] = {}
    for name, component_id in values.items():
        if not isinstance(name, str):
            raise ValueError("component names must be strings.")
        if not isinstance(component_id, (int, np.integer)):
            raise ValueError("component ids must be integers.")
        result[name] = int(component_id)
    return result


__all__ = ["ElementType3D", "Mesh3D"]
