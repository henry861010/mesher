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
        + 6 * mesh.element_count
        + mesh.component_count
        + len(mesh.types)
        + len(mesh.reals)
        + len(mesh.sections)
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
        handle.write(f"type_count={len(mesh.types)}\n")
        handle.write(f"real_count={len(mesh.reals)}\n")
        handle.write(f"section_count={len(mesh.sections)}\n")

        handle.write("\n*NODES,index,x,y,z\n")
        report("Writing CDB nodes.", force=True)
        for node_index, node in enumerate(mesh.nodes):
            handle.write(
                f"{node_index},{_format_float(node[0])},{_format_float(node[1])},{_format_float(node[2])}\n"
            )
            completed_records += 1
            report("Writing CDB nodes.")

        element_columns = ",".join(f"n{index}" for index in range(20))
        handle.write(f"\n*ELEMENTS,index,{element_columns}\n")
        report("Writing CDB elements.", force=True)
        for element_index, element in enumerate(mesh.elements):
            node_ids = ",".join(str(int(node_id)) for node_id in element)
            handle.write(f"{element_index},{node_ids}\n")
            completed_records += 1
            report("Writing CDB elements.")

        handle.write("\n*ELEMENT_NODE_NUM,index,node_num\n")
        report("Writing CDB element node counts.", force=True)
        for element_index, node_num in enumerate(mesh.element_node_num):
            handle.write(f"{element_index},{int(node_num)}\n")
            completed_records += 1
            report("Writing CDB element node counts.")

        handle.write("\n*ELEMENT_TYPE,index,type_id\n")
        report("Writing CDB element types.", force=True)
        for element_index, type_id in enumerate(mesh.element_types):
            handle.write(f"{element_index},{int(type_id)}\n")
            completed_records += 1
            report("Writing CDB element types.")

        handle.write("\n*TYPES,type_id,ansys_element_type\n")
        report("Writing CDB type table.", force=True)
        for type_id, ansys_element_type in sorted(mesh.types.items()):
            handle.write(f"{int(type_id)},{int(ansys_element_type)}\n")
            completed_records += 1
            report("Writing CDB type table.")

        handle.write("\n*ELEMENT_REAL,index,real_id\n")
        report("Writing CDB element real ids.", force=True)
        for element_index, real_id in enumerate(mesh.element_reals):
            handle.write(f"{element_index},{int(real_id)}\n")
            completed_records += 1
            report("Writing CDB element real ids.")

        handle.write("\n*REALS,real_id,values_json\n")
        report("Writing CDB real table.", force=True)
        for real_id, values in sorted(mesh.reals.items()):
            encoded_values = json.dumps(
                values,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            handle.write(f"{int(real_id)},{encoded_values}\n")
            completed_records += 1
            report("Writing CDB real table.")

        handle.write("\n*ELEMENT_SECTION,index,section_id\n")
        report("Writing CDB element section ids.", force=True)
        for element_index, section_id in enumerate(mesh.element_sections):
            handle.write(f"{element_index},{int(section_id)}\n")
            completed_records += 1
            report("Writing CDB element section ids.")

        handle.write("\n*SECTIONS,section_id,values_json\n")
        report("Writing CDB section table.", force=True)
        for section_id, values in sorted(mesh.sections.items()):
            encoded_values = json.dumps(
                values,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            handle.write(f"{int(section_id)},{encoded_values}\n")
            completed_records += 1
            report("Writing CDB section table.")

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
        "typeCount": len(mesh.types),
        "realCount": len(mesh.reals),
        "sectionCount": len(mesh.sections),
    }


def _format_float(value: object) -> str:
    return f"{float(value):.12g}"
