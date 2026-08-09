"""Shared mesh data container."""

from dataclasses import dataclass

from numpy.typing import ArrayLike


@dataclass
class Mesh:
    """Node coordinates and element connectivity.

    Validation is intentionally left to consumers such as
    :class:`mesh_quality.MeshQualityChecker`.
    """

    nodes: ArrayLike
    elements: ArrayLike


__all__ = ["Mesh"]
