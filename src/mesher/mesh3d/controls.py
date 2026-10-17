"""Z sizing constraints and material layers, independent of the extruder."""

from typing import Any, TypedDict


class ZControl(TypedDict):
    element_size: float
    method: str
    start: float
    end: float


class LayerInfo(TypedDict):
    z: float
    element_size: float
    assignments: list[dict[str, Any]]
