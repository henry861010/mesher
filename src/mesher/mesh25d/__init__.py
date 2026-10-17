"""Standard V1 geometry-to-solid meshing with separate 2D/3D stages."""

from __future__ import annotations

from typing import Any, Callable

from ..mesh3d.model import Mesh3D
from .domain import SymmetryMode


def build_mesh(
    geometry_structure: dict[str, Any],
    mesh_control: dict[str, Any],
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> Mesh3D:
    """Build a 2.5D mesh; load the optional meshing runtime only on use."""
    try:
        from .pipeline import build_mesh as build
    except ModuleNotFoundError as error:
        if error.name not in {"process_flow_kernel", "matplotlib", "shapely"}:
            raise
        raise ImportError(
            "Mesh25D meshing requires optional dependencies. "
            "Install them with `pip install mesher[process-flow]`."
        ) from error
    return build(geometry_structure, mesh_control, progress=progress)


__all__ = ["SymmetryMode", "build_mesh"]
