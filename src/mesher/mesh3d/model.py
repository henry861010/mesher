"""Public mesh data structures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Mesh3D:
    """Owned 3D mesh arrays with fixed-width solid connectivity.

    Hexahedra use eight distinct node slots. Wedge-like elements extruded from
    padded Tri3 faces retain eight slots by repeating the third bottom and top
    nodes.
    """

    nodes: np.ndarray
    elements: np.ndarray
    element_comps: np.ndarray
    comps: dict[str, int]

    def __post_init__(self) -> None:
        nodes = np.array(self.nodes, dtype=np.float64, copy=True)
        if nodes.ndim != 2 or nodes.shape[1] < 3:
            raise ValueError("nodes must have shape (n, 3+).")

        elements = np.array(self.elements, dtype=np.int32, copy=True)
        if elements.ndim != 2 or elements.shape[1] != 8:
            raise ValueError("elements must have shape (m, 8).")

        element_comps = np.array(self.element_comps, dtype=np.int32, copy=True)
        if element_comps.ndim != 1:
            raise ValueError("element_comps must have shape (m,).")
        if len(element_comps) != len(elements):
            raise ValueError("element_comps length must match element count.")

        object.__setattr__(self, "nodes", np.ascontiguousarray(nodes[:, :3]))
        object.__setattr__(self, "elements", np.ascontiguousarray(elements))
        object.__setattr__(
            self,
            "element_comps",
            np.ascontiguousarray(element_comps),
        )
        object.__setattr__(self, "comps", dict(self.comps))

    @property
    def node_count(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def element_count(self) -> int:
        return int(self.elements.shape[0])

    @property
    def component_count(self) -> int:
        return len(self.comps)
