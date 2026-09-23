"""Process-flow translation, meshing pipeline, and CDB export."""

from ..contracts.process_flow_2_5d import validate_mesh_control

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

__all__ = [
    "SymmetryMode",
    "build_mesh_from_structure",
    "validate_mesh_control",
    "write_cdb_text",
]
