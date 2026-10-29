"""Public results and geometric section plans for a direct Quad4 layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...model import Mesh2D


class QuadLayerError(ValueError):
    """A layer could not be built; the caller's mesh has not been modified."""

    def __init__(self, message: str, *, code: str = "invalid_layer", **context: Any):
        super().__init__(message)
        self.code = code
        self.context = context


class UnmeshableDirectQuadLayout(QuadLayerError):
    """The available direct-quad partitions cannot represent this geometry."""

    def __init__(self, message: str, **context: Any):
        super().__init__(message, code="unmeshable_direct_quad_layout", **context)


@dataclass(frozen=True)
class QuadLayerResult:
    """The committed mesh, its new logical outline, and construction diagnostics."""

    mesh: Mesh2D
    outline: list[list[int]]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class Seam:
    """A shared section interface, ordered from the source to the new front."""

    id: int
    points: np.ndarray
    owners: tuple[int, ...]
    pattern_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class Section:
    """A geometric section; unit subdivision is a separate planning step.

    Source chains follow the input outline. The outer boundary follows the
    same traversal, from the left interface to the right interface. A plain
    INNER section can have a one-point outer boundary. An OUTER section has
    a one-node source chain. Neither representation is a degenerate element:
    sections are allowed to contain several eventual quadrilateral units.
    """

    id: int
    kind: str
    source_chains: tuple[tuple[int, ...], ...]
    outer_boundary: np.ndarray
    left_seam: int
    right_seam: int
    source_edge: int | None = None
    corner: int | None = None
    pattern_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class QuadSectionPlan:
    """A complete cyclic section partition, without creating mesh elements."""

    sections: tuple[Section, ...]
    seams: tuple[Seam, ...]
    front: np.ndarray
    edge_expansion: tuple[float, ...]
    diagnostics: dict[str, Any]
