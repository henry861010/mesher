"""Optional Matplotlib pattern and PyVista mesh visualization."""

from typing import TYPE_CHECKING

from .pattern import plot_pattern2d

if TYPE_CHECKING:
    from .pyvista import build_faces, view_mesh

__all__ = ["build_faces", "view_mesh", "plot_pattern2d"]


def __getattr__(name):
    if name in {"build_faces", "view_mesh"}:
        from . import pyvista

        value = getattr(pyvista, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
