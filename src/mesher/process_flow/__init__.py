"""Process-flow translation, meshing pipeline, and CDB export."""

try:
    from .pipeline import SymmetryMode, build_mesh_from_structure
except ModuleNotFoundError as error:
    if error.name not in {"process_flow_kernel", "matplotlib"}:
        raise
    raise ImportError(
        "Process-flow meshing requires optional dependencies. "
        "Install them with `pip install mesher[process-flow]`."
    ) from error

from .exporters import write_cdb_text

__all__ = ["SymmetryMode", "build_mesh_from_structure", "write_cdb_text"]
