"""Text CDB exporter for mesher-owned 3D meshes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from ...mesh3d import Mesh3D

ProgressCallback = Callable[[dict[str, Any]], None]


def write_cdb_text(
    output_path: str | Path,
    *,
    mesh: Mesh3D,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Write a 3D mesh to a deterministic line-oriented CDB text artifact."""
    path = Path(output_path)
    total_records = (
        mesh.node_count
        + mesh.element_count
        + len(mesh.element_comps)
        + mesh.component_count
    )
    completed_records = 0
    report_interval = max(1, total_records // 100)

    def report(message: str, *, force: bool = False) -> None:
        if progress is None:
            return
        if force or completed_records % report_interval == 0:
            progress(
                {
                    "event": "progress",
                    "current": completed_records,
                    "total": total_records,
                    "unit": "records",
                    "message": message,
                    "data": {},
                }
            )

    with path.open("w", encoding="utf-8", buffering=1024 * 1024) as handle:
        handle.write("# Process Flow CDB text export\n")
        handle.write("# Format: raw mesh array sections\n")
        handle.write(f"node_count={mesh.node_count}\n")
        handle.write(f"element_count={mesh.element_count}\n")
        handle.write(f"component_count={mesh.component_count}\n")

        handle.write("\n*NODES,index,x,y,z\n")
        report("Writing CDB nodes.", force=True)
        for node_index, node in enumerate(mesh.nodes):
            handle.write(
                f"{node_index},{_format_float(node[0])},{_format_float(node[1])},{_format_float(node[2])}\n"
            )
            completed_records += 1
            report("Writing CDB nodes.")

        handle.write("\n*ELEMENTS,index,n0,n1,n2,n3,n4,n5,n6,n7\n")
        report("Writing CDB elements.", force=True)
        for element_index, element in enumerate(mesh.elements):
            node_ids = ",".join(str(int(node_id)) for node_id in element)
            handle.write(f"{element_index},{node_ids}\n")
            completed_records += 1
            report("Writing CDB elements.")

        handle.write("\n*ELEMENT_COMP,index,component_id\n")
        report("Writing CDB element components.", force=True)
        for element_index, component_id in enumerate(mesh.element_comps):
            handle.write(f"{element_index},{int(component_id)}\n")
            completed_records += 1
            report("Writing CDB element components.")

        handle.write("\n*COMPS,component_id,name\n")
        report("Writing CDB component table.", force=True)
        for name, component_id in sorted(
            mesh.comps.items(),
            key=lambda item: item[1],
        ):
            encoded_name = json.dumps(str(name), ensure_ascii=False)
            handle.write(f"{int(component_id)},{encoded_name}\n")
            completed_records += 1
            report("Writing CDB component table.")

    report("CDB output written.", force=True)

    return {
        "outputPath": str(path),
        "nodeCount": mesh.node_count,
        "elementCount": mesh.element_count,
        "componentCount": mesh.component_count,
    }


def _format_float(value: object) -> str:
    return f"{float(value):.12g}"
