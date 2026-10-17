"""Planar sizing data shared by control methods and meshing planners.

Regions use the existing BOX/CIRCLE/POLYGON face dictionaries. These are
requests for a region's size, not instructions to operate a meshing algorithm.
"""

from typing import Any, Literal, TypedDict


class Face2D(TypedDict):
    type: Literal["BOX", "CIRCLE", "POLYGON"]
    dim: list[Any]


class XYRegionControl(TypedDict):
    region: Face2D
    element_size: float
