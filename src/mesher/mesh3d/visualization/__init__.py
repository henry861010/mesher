"""Optional mesh visualization API."""

try:
    from .viewer import MeshViewer
except ModuleNotFoundError as exc:
    if exc.name not in {"pyvista", "vtkmodules"}:
        raise
    raise ImportError(
        "Mesh visualization requires optional dependencies. "
        "Install them with `pip install mesher[visualization]`."
    ) from exc

__all__ = ["MeshViewer"]
