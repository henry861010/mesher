"""Public mesh data structures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Mesh3D:
    """Owned 3D mesh arrays with fixed-width ANSYS connectivity.

    Every connectivity row has twenty slots. ``element_node_num`` records how
    many leading slots belong to the element; remaining slots repeat the last
    effective node id. Degenerate SOLID185 wedges still use eight effective
    slots because their repeated ids are part of the ANSYS connectivity.
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

    @property
    def node_count(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def element_count(self) -> int:
        return int(self.elements.shape[0])

    @property
    def component_count(self) -> int:
        return len(self.comps)


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
