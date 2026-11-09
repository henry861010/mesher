"""Public results and geometric section plans for a direct Quad4 layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from ....tolerance import GeometryTolerance, resolve_tolerance

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
    """The committed mesh, its complete node-ID outline, and construction diagnostics."""

    mesh: Mesh2D
    outline: list[int]
    diagnostics: dict[str, Any]
    tolerance: GeometryTolerance = field(default_factory=resolve_tolerance)


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

    Source chains follow the normalized logical-edge order. The outer boundary
    follows the same traversal, from the left interface to the right interface. A plain
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
class PreparedLayerSource:
    """Owned, read-only world-coordinate source after slot and corner repairs.

    Original node and element IDs are preserved as array prefixes. Section
    source chains in the containing plan index these nodes, not the caller's
    still-unmodified mesh. ``outline`` is a complete node-ID loop with implicit
    closure; its individual source segments may be oblique.
    """

    nodes: np.ndarray
    elements: np.ndarray
    outline: tuple[int, ...]
    original_node_count: int
    original_element_count: int
    tolerance: GeometryTolerance = field(default_factory=resolve_tolerance)

    def __post_init__(self):
        for name in ("nodes", "elements"):
            source = np.asarray(getattr(self, name))
            # An immutable backing buffer also prevents callers from turning
            # WRITEABLE back on through a view of the preview snapshot.
            values = np.frombuffer(source.tobytes(), dtype=source.dtype).reshape(source.shape)
            object.__setattr__(self, name, values)
        object.__setattr__(self, "outline", tuple(map(int, self.outline)))
        object.__setattr__(self, "tolerance", resolve_tolerance(self.tolerance))


@dataclass(frozen=True)
class QuadSectionPlan:
    """A cyclic section preview, optionally including staged repair cells.

    Slot and corner fills are recorded separately in diagnostics and together
    in ``prepared_source`` before the previewed expansion is constructed.
    """

    sections: tuple[Section, ...]
    seams: tuple[Seam, ...]
    front: np.ndarray
    edge_expansion: tuple[float, ...]
    diagnostics: dict[str, Any]
    prepared_source: PreparedLayerSource | None = None
    tolerance: GeometryTolerance = field(default_factory=resolve_tolerance)

    def __post_init__(self):
        object.__setattr__(self, "tolerance", resolve_tolerance(self.tolerance))
