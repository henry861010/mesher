"""Public mesh data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class MeshSet:
    """Explicit node- or element-row indices, independent of components.

    Indices are zero-based and copied into a sorted, unique ``int32`` array.
    Empty selections are valid. Fields are frozen, but the owned array remains
    writable, just like the arrays in :class:`Mesh3D`.
    """

    kind: Literal["node", "element"]
    indices: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or self.kind not in ("node", "element"):
            raise ValueError("kind must be 'node' or 'element'.")
        object.__setattr__(self, "indices", _normalize_set_indices(self.indices))


@dataclass(frozen=True)
class Mesh3D:
    """Owned 3D mesh arrays with fixed-width ANSYS connectivity.

    Every connectivity row has twenty slots. ``element_node_num`` records how
    many leading slots belong to the element; remaining slots repeat the last
    effective node id. Degenerate SOLID185 wedges still use eight effective
    slots because their repeated ids are part of the ANSYS connectivity.

    ``sets`` maps globally unique, case-sensitive names to explicit node or
    element selections. Membership may overlap between sets.
    """

    nodes: np.ndarray
    elements: np.ndarray
    element_comps: np.ndarray
    comps: dict[str, int]
    element_types: np.ndarray
    types: dict[int, int]
    element_reals: np.ndarray
    reals: dict[int, list]
    element_sections: np.ndarray
    sections: dict[int, list]
    element_node_num: np.ndarray
    sets: dict[str, MeshSet] = field(default_factory=dict)

    def __post_init__(self) -> None:
        nodes = np.array(self.nodes, dtype=np.float64, copy=True)
        if nodes.ndim != 2 or nodes.shape[1] < 3:
            raise ValueError("nodes must have shape (n, 3+).")

        elements = np.array(self.elements, dtype=np.int32, copy=True)
        if elements.ndim != 2 or elements.shape[1] != 20:
            raise ValueError("elements must have shape (m, 20).")

        element_comps = _normalize_element_array(
            self.element_comps,
            "element_comps",
            len(elements),
        )
        element_types = _normalize_element_array(
            self.element_types,
            "element_types",
            len(elements),
        )
        element_reals = _normalize_element_array(
            self.element_reals,
            "element_reals",
            len(elements),
        )
        element_sections = _normalize_element_array(
            self.element_sections,
            "element_sections",
            len(elements),
        )
        element_node_num = _normalize_element_array(
            self.element_node_num,
            "element_node_num",
            len(elements),
        )
        if np.any(element_node_num < 1) or np.any(element_node_num > 20):
            raise ValueError("element_node_num values must be between 1 and 20.")

        for node_num in np.unique(element_node_num):
            effective_count = int(node_num)
            if effective_count < 20:
                rows = element_node_num == effective_count
                elements[rows, effective_count:] = elements[
                    rows,
                    effective_count - 1,
                    None,
                ]

        object.__setattr__(self, "nodes", np.ascontiguousarray(nodes[:, :3]))
        object.__setattr__(self, "elements", np.ascontiguousarray(elements))
        object.__setattr__(self, "element_comps", element_comps)
        object.__setattr__(self, "comps", dict(self.comps))
        object.__setattr__(self, "element_types", element_types)
        object.__setattr__(
            self,
            "types",
            {
                int(type_id): int(ansys_type)
                for type_id, ansys_type in self.types.items()
            },
        )
        object.__setattr__(self, "element_reals", element_reals)
        object.__setattr__(
            self,
            "reals",
            {int(real_id): list(values) for real_id, values in self.reals.items()},
        )
        object.__setattr__(self, "element_sections", element_sections)
        object.__setattr__(
            self,
            "sections",
            {
                int(section_id): list(values)
                for section_id, values in self.sections.items()
            },
        )
        object.__setattr__(self, "element_node_num", element_node_num)
        object.__setattr__(
            self,
            "sets",
            _normalize_sets(
                self.sets,
                node_count=self.node_count,
                element_count=self.element_count,
            ),
        )

    @property
    def node_count(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def element_count(self) -> int:
        return int(self.elements.shape[0])

    @property
    def component_count(self) -> int:
        return len(self.comps)


def _normalize_set_indices(values: object) -> np.ndarray:
    indices = np.asarray(values)
    if indices.ndim != 1:
        raise ValueError("set indices must be a one-dimensional sequence.")
    if indices.size == 0:
        return np.empty(0, dtype=np.int32)

    if isinstance(values, np.ndarray):
        integer_values = np.issubdtype(indices.dtype, np.integer)
    else:
        # Check before coercion can hide mixed bool/int inputs or overflow
        # arbitrarily large Python integers into a narrower NumPy dtype.
        integer_values = all(
            isinstance(value, (int, np.integer))
            and not isinstance(value, (bool, np.bool_))
            for value in values
        )
    if not integer_values:
        raise TypeError("set indices must contain integers, not booleans.")
    if np.any(indices < 0):
        raise ValueError("set indices must be nonnegative.")
    if np.any(indices > np.iinfo(np.int32).max):
        raise OverflowError("set indices exceed the int32 range.")
    return np.unique(indices).astype(np.int32, copy=True)


def _normalize_sets(
    values: dict[str, MeshSet],
    *,
    node_count: int,
    element_count: int,
) -> dict[str, MeshSet]:
    """Validate and snapshot sets at construction and export boundaries."""
    if not isinstance(values, dict):
        raise TypeError("sets must be a dictionary of names to MeshSet objects.")

    result = {}
    for name, selection in values.items():
        if not isinstance(name, str):
            raise TypeError("set names must be strings.")
        if not name:
            raise ValueError("set names must not be empty.")
        if not isinstance(selection, MeshSet):
            raise TypeError(f"set {name!r} must be a MeshSet object.")
        try:
            selection = MeshSet(kind=selection.kind, indices=selection.indices)
        except (TypeError, ValueError, OverflowError) as exc:
            raise type(exc)(f"set {name!r}: {exc}") from exc
        count = node_count if selection.kind == "node" else element_count
        if selection.indices.size and selection.indices[-1] >= count:
            raise IndexError(
                f"set {name!r} contains an out-of-range {selection.kind} index."
            )
        result[name] = selection
    return result


def _normalize_element_array(
    values: object,
    name: str,
    element_count: int,
) -> np.ndarray:
    result = np.array(values, dtype=np.int32, copy=True)
    if result.ndim != 1:
        raise ValueError(f"{name} must have shape (m,).")
    if len(result) != element_count:
        raise ValueError(f"{name} length must match element count.")
    return np.ascontiguousarray(result)
