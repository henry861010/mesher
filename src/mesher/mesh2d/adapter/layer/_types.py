"""Public results and internal geometric blocks for a direct Quad4 layer."""

from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass
class Block:
    """A positively oriented polygon filled directly with Quad4 cells.

    Coordinates are in the orchestrator's normalized XY frame. Source nodes
    on its sides are retained by the shared-interface registry.
    """

    vertices: np.ndarray
    kind: str = "LINE"
    source_edge: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
