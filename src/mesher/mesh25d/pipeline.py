"""Coordinate Standard V1 geometry, mesh-control interfaces and 2.5D stages."""

from __future__ import annotations

from typing import Any

from process_flow_kernel import validate_geometry_semantic_keys

from ..contracts.process_flow_2_5d import mesh_control_settings
from ..mesh3d.model import Mesh3D
from .controls import resolve_controls
from .domain import normalize_symmetry
from .geometry.assignments import extract_assignments
from .geometry.context import prepare_geometry
from .plan3d import build_3d_mesh, plan_layers
from .plan2d import build_2d_mesh, plan_2d
from .progress import ProgressCallback, _complete_stage, _start_stage


def build_mesh(
    geometry_structure: dict[str, Any],
    mesh_control: dict[str, Any],
    *,
    progress: ProgressCallback | None = None,
) -> Mesh3D:
    """Build a full or symmetry-reduced solid mesh without mutating inputs.

    Symmetry is centered on the combined root body XY bounds, falling back to
    all footprints when the root has no bodies. Control references follow the
    retained geometry. XY regions contribute axis-aligned cuts before the base
    grid fills remaining gaps using the global planar element size.
    """
    stage = _start_stage(progress, "validating", "Checking geometry input.")
    element_size, symmetry, _ = mesh_control_settings(mesh_control)
    normalized_symmetry = normalize_symmetry(symmetry)
    validate_geometry_semantic_keys(geometry_structure)
    root = _root_container(geometry_structure)
    _complete_stage(progress, "validating", stage)

    stage = _start_stage(progress, "analyzing_geometry", "Analyzing geometry patterns.")
    geometry = prepare_geometry(root, normalized_symmetry)
    controls = resolve_controls(
        root, geometry.container, mesh_control["controls"],
        global_element_size=element_size,
    )
    plan2d = plan_2d(
        geometry.pattern, controls, element_size=element_size, domain=geometry.domain,
    )
    plan3d = plan_layers(
        extract_assignments(geometry.container), controls.z, element_size=element_size,
    )
    layer_infos = plan3d.layer_infos
    _complete_stage(
        progress, "analyzing_geometry", stage,
        data={
            "faceCount": len(geometry.pattern.all_faces),
            "circlePatternCount": len(plan2d.circle_patterns),
            "imprintOperationCount": len(plan2d.circle_plan.imprint_patterns),
            "extensionOperationCount": len(plan2d.circle_plan.extensions),
            "xGridLineCount": len(set(plan2d.x_lines)),
            "yGridLineCount": len(set(plan2d.y_lines)),
            "layerBoundaryCount": len(layer_infos),
            "layerIntervalCount": max(0, len(layer_infos) - 1),
            "assignmentCount": sum(len(info["assignments"]) for info in layer_infos[:-1]),
            "planarElementSize": plan2d.element_size,
            "symmetry": normalized_symmetry.value,
        },
    )

    stage = _start_stage(progress, "building_2d_mesh", "Generating base grid.")
    mesh_2d = build_2d_mesh(plan2d, progress=progress)
    _complete_stage(
        progress, "building_2d_mesh", stage,
        data={
            "featureOperationCount": plan2d.feature_total,
            "node2DCount": int(len(mesh_2d.nodes)),
            "element2DCount": int(len(mesh_2d.elements)),
        },
    )

    stage = _start_stage(
        progress, "building_3d_mesh", "Building 3D mesh layers.",
        current=0, total=max(0, len(layer_infos) - 1), unit="layers",
    )
    mesh = build_3d_mesh(mesh_2d, plan3d, progress=progress)
    _complete_stage(
        progress, "building_3d_mesh", stage,
        data={
            "nodeCount": mesh.node_count,
            "elementCount": mesh.element_count,
            "componentCount": mesh.component_count,
        },
    )
    return mesh


def _root_container(geometry_structure: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(geometry_structure, dict):
        raise ValueError("geometryStructure must be an object.")
    root = geometry_structure.get("root")
    if not isinstance(root, dict):
        raise ValueError("geometryStructure.root must be an object.")
    return root
