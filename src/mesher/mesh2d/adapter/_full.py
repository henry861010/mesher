"""Transactional multi-boundary adaptive meshing and rectilinear completion."""

from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..generators import generate_rectilinear_mesh_optimal
from ..generators._holed_geometry import normalize_patterns
from ..model import Mesh2D
from ._full_geometry import (complement_domains, discover_boundaries, flatten_outline,
                             section_outline_points, validate_fronts)
from .layer import QuadLayerError, generate_layer_adaptive_mesh, plan_quad_sections
from .layer._geometry import BoxIndex, area, cross
from .layer._line_policy import (DEFAULT_MAX_TRANSITION_COUNT, validate_dense_parameters,
                                 validate_line_parameters)
from .layer._mesher import _generate_layer_adaptive_mesh
from .layer._validation import _check_edge_geometry, _check_overlaps


@dataclass(frozen=True)
class FullAdaptiveMeshResult:
    """The committed mesh and its boundaries, indexed using final mesh node IDs.

    ``outer_outline`` is a clockwise loop with implicit closure.
    ``layer_outlines[layer][boundary][edge]`` is an ordered node-ID chain;
    boundary order follows the original ``EditorMesh2D.outlines_by_edge``.
    Diagnostics record that order's containment tree and each construction call.
    Layer and boundary indices are zero based.
    """

    mesh: Mesh2D
    outer_outline: NDArray[np.int64]
    layer_outlines: list[list[list[list[int]]]]
    diagnostics: dict[str, Any]


def _call(operation, *, stage, **context):
    """Attach orchestration context without replacing a typed builder failure."""
    try:
        return operation()
    except QuadLayerError as error:
        error.context.setdefault("stage", stage)
        for key, value in context.items():
            error.context.setdefault(key, value)
        raise
    except (ValueError, TypeError, OverflowError) as error:
        code = ("invalid_input" if stage == "input_validation" else
                "global_meshing_failed" if stage == "global_meshing" else
                "invalid_full_mesh")
        raise QuadLayerError(str(error), code=code, stage=stage, **context) from error


def _inputs(element_sizes, expand_ratio, pattern_lines):
    try:
        sizes = list(element_sizes)
    except TypeError as error:
        raise ValueError("element_sizes must be a nonempty sequence of positive numbers") from error
    if not sizes:
        raise ValueError("element_sizes must not be empty")
    for index, value in enumerate(sizes):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))):
            raise QuadLayerError("each element size must be finite and positive",
                                 code="invalid_input", parameter="element_sizes",
                                 layer_index=index)
        try:
            sizes[index] = float(value)
        except (ValueError, OverflowError) as error:
            raise QuadLayerError("each element size must be finite and positive",
                                 code="invalid_input", parameter="element_sizes",
                                 layer_index=index) from error
        if not math.isfinite(sizes[index]) or sizes[index] <= 0:
            raise QuadLayerError("each element size must be finite and positive",
                                 code="invalid_input", parameter="element_sizes",
                                 layer_index=index)
    if (isinstance(expand_ratio, (bool, np.bool_))
            or not isinstance(expand_ratio, (int, float, np.integer, np.floating))
            or not math.isfinite(float(expand_ratio)) or expand_ratio < 1):
        raise ValueError("expand_ratio must be finite and at least 1")
    patterns = np.array(list(pattern_lines), dtype=float)
    if patterns.shape == (0,):
        patterns = patterns.reshape(0, 2, 2)
    if patterns.ndim != 3 or patterns.shape[1:] != (2, 2):
        raise ValueError("pattern_lines must have shape (n, 2, 2)")
    # Validate global's stricter axis contract without reordering pattern IDs.
    normalize_patterns(patterns)
    return sizes, float(expand_ratio), patterns


def _edges(loop):
    return {tuple(sorted((int(a), int(b))))
            for a, b in zip(loop, np.roll(loop, -1))}


def _interface_ids(mesh, outline, global_mesh, global_ids):
    """Map only an explicitly paired interface, with exact canonical XY values."""
    local_ids = flatten_outline(outline)
    if len(local_ids) != len(global_ids):
        raise ValueError("adaptive and global interfaces have different node counts")
    lookup = {tuple(mesh.nodes[node, :2]): int(node) for node in local_ids}
    if len(lookup) != len(local_ids):
        raise ValueError("an interface has duplicate node coordinates")
    try:
        mapped = np.asarray([lookup[tuple(global_mesh.nodes[node, :2])]
                             for node in global_ids], dtype=np.int64)
    except KeyError as error:
        raise ValueError("adaptive and global interface coordinates differ") from error
    if len(np.unique(mapped)) != len(local_ids) or _edges(mapped) != _edges(local_ids):
        raise ValueError("adaptive and global interface edge cycles differ")
    return mapped


def _assemble(working, final_outlines, regions):
    """Append global cells without equivalencing or renumbering existing nodes."""
    nodes, elements = [working.nodes], [working.elements]
    node_count = working.node_count
    outer_ids = None
    for region_index, (domain, result) in enumerate(regions):
        mapping = np.full(result.mesh.node_count, -1, dtype=np.int64)
        interfaces = list(zip(domain.hole_boundary_indices, result.hole_outlines))
        if domain.boundary_index is not None:
            interfaces.append((domain.boundary_index, result.outer_outline))
        for boundary_index, global_ids in interfaces:
            mapping[global_ids] = _call(
                lambda: _interface_ids(working, final_outlines[boundary_index],
                                       result.mesh, global_ids),
                stage="interface_assembly", boundary_index=boundary_index,
                global_region_index=region_index)
        interior = np.flatnonzero(mapping < 0)
        next_count = node_count + len(interior)
        if next_count - 1 > np.iinfo(np.int32).max:
            raise QuadLayerError("combined node IDs exceed the int32 range",
                                 code="numeric_range_exceeded", stage="global_assembly",
                                 global_region_index=region_index)
        mapping[interior] = np.arange(node_count, next_count, dtype=np.int64)
        nodes.append(result.mesh.nodes[interior])
        elements.append(mapping[result.mesh.elements])
        if domain.boundary_index is None:
            outer_ids = mapping[result.outer_outline].copy()
        node_count = next_count
    if outer_ids is None:
        raise ValueError("global completion is missing its exterior region")
    return Mesh2D(np.vstack(nodes), np.vstack(elements)), outer_ids


def _validate_complete(mesh, original, outer, outer_ids):
    """Validate the assembled topology and geometry before the sole caller commit."""
    if (not np.array_equal(mesh.nodes[:original.node_count], original.nodes)
            or not np.array_equal(mesh.elements[:original.element_count], original.elements)):
        raise ValueError("full meshing changed original nodes or elements")
    referenced_new = np.unique(mesh.elements[mesh.elements >= original.node_count])
    if not np.array_equal(referenced_new, np.arange(original.node_count, mesh.node_count)):
        raise ValueError("full meshing generated unused nodes")
    scale = float(np.max(np.ptp(outer, axis=0)))
    xy = (mesh.nodes[:, :2] - outer[0]) / scale
    polygon = (outer - outer[0]) / scale
    if not np.all(np.isfinite(xy)):
        raise ValueError("combined coordinates exceed the numerical range")
    cells = xy[mesh.elements]
    vectors = np.roll(cells, -1, axis=1) - cells
    if np.any(cross(vectors, np.roll(vectors, -1, axis=1)) <= 0):
        raise ValueError("the complete mesh must contain strictly convex CCW Quad4s")
    shifted = cells - cells[:, :1]
    cell_areas = .5 * np.sum(cross(shifted, np.roll(shifted, -1, axis=1)), axis=1)
    actual = math.fsum(map(float, cell_areas))
    expected = abs(area(polygon))
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=0.):
        raise ValueError("the complete mesh does not cover the requested outer area")

    directed = np.stack((mesh.elements, np.roll(mesh.elements, -1, axis=1)), axis=2).reshape(-1, 2)
    keys, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0,
                                      return_inverse=True, return_counts=True)
    signs = np.where(directed[:, 0] < directed[:, 1], 1, -1)
    balance = np.bincount(inverse, weights=signs, minlength=len(keys))
    if np.any(counts > 2) or np.any(balance[counts == 2] != 0):
        raise ValueError("mesh edges must have at most two oppositely directed owners")
    if set(map(tuple, keys[counts == 1])) != _edges(outer_ids):
        raise ValueError("the complete mesh has a hole, a gap, or an unmatched interface")

    # Use the same normalized predicates as individual layers. This also checks
    # contacts across independently constructed global regions and interfaces.
    tol = max(128 * np.finfo(float).eps,
              4 * abs(float(np.spacing(np.max(np.abs(mesh.nodes[:, :2]))))) / scale)
    edge_points = xy[keys]
    edge_index = BoxIndex(edge_points.min(axis=1), edge_points.max(axis=1), tol)
    cell_index = BoxIndex(cells.min(axis=1), cells.max(axis=1), tol)
    empty_index = BoxIndex(np.empty((0, 2)), np.empty((0, 2)), tol)
    _check_edge_geometry(keys, edge_points, np.empty((0, 2), dtype=np.int64),
                         np.empty((0, 2, 2)), edge_index, empty_index, tol)
    _check_overlaps(cells, np.empty((0, 4, 2)), cell_index, empty_index, tol)


def generate_full_adaptive_mesh(
    mesh: Mesh2D,
    element_sizes: list[float],
    *,
    outer_outline,
    expand_ratio: float = 1.5,
    pattern_lines=(),
    max_transition_count: int = DEFAULT_MAX_TRANSITION_COUNT,
    min_size_ratio: float = 0.5,
    coordinate_tolerance: float = 1e-3,
    small_element_size: float | None = None,
    dense_region_tolerance: float = 1e-3,
) -> FullAdaptiveMeshResult:
    """Fill an orthogonal domain around existing focus meshes with Quad4s.

    Each original exterior grows outward and each hole grows inward once per
    size, in the supplied order (equal and decreasing sizes are allowed). The
    final size also controls rectilinear completion of every remaining void.
    The global generator determines the final transition's exact boundary
    nodes before that transition is built. Nested focus islands are supported.

    Sources must have convex CCW Quad4s, Z=0, and even boundary segment counts.
    Source boundaries follow the single-layer near-axis contract. The global
    outer outline is a simple axis-aligned XY loop, with optional closure.
    Boundary contacts, collisions, hole collapse and unsupported fixed-boundary
    layouts fail without retrying with different sizes or extra layers.

    Patterns retain the existing layer constraints and must also be exactly
    horizontal or vertical for global generation. Each underlying call retains
    its one-million-element budget; this API does not impose a cumulative limit.
    ``coordinate_tolerance`` is the finite, non-negative absolute tolerance
    for boundary separation and global geometry validation; it defaults to
    1e-3. Coordinates are not snapped by this tolerance.

    ``max_transition_count`` and ``min_size_ratio`` have the single-layer
    API's LINE constraints and defaults (20 and 0.5). They apply to ordinary
    intermediate layers only. The final transition preserves the global
    generator's complete fixed interface, without imposing these constraints.
    Both arguments are validated even when there is only one, fixed layer.

    ``small_element_size`` and ``dense_region_tolerance`` have the single-layer
    semantics and defaults (None and 0.001), in world length units. Dense seams
    are planned before global meshing and retained in the final fixed interface.

    All work is staged privately. Success updates and returns the original mesh;
    any failure leaves both original arrays unchanged. QuadLayerError context
    includes zero-based layer/boundary/global-region indices when applicable.
    """
    started = perf_counter()
    max_transition_count, min_size_ratio = validate_line_parameters(
        max_transition_count, min_size_ratio)
    small_element_size, dense_region_tolerance = validate_dense_parameters(
        small_element_size, dense_region_tolerance)
    sizes, ratio, patterns = _call(lambda: _inputs(element_sizes, expand_ratio, pattern_lines),
                                    stage="input_validation")
    topology = _call(lambda: discover_boundaries(
        mesh, outer_outline, coordinate_tolerance=coordinate_tolerance),
        stage="input_validation")
    working = Mesh2D(mesh.nodes, mesh.elements)
    outlines = topology.outlines
    layer_outlines, layer_diagnostics = [], []

    for layer_index, size in enumerate(sizes[:-1]):
        current, diagnostics = [], []
        for boundary_index, outline in enumerate(outlines):
            result = _call(
                lambda: generate_layer_adaptive_mesh(working, outline, size,
                            expand_ratio=ratio, pattern_lines=patterns,
                            max_transition_count=max_transition_count,
                            min_size_ratio=min_size_ratio, small_element_size=small_element_size,
                            dense_region_tolerance=dense_region_tolerance),
                stage="layer_construction", layer_index=layer_index, boundary_index=boundary_index)
            current.append(result.outline)
            diagnostics.append(result.diagnostics)
        validate_fronts(topology, [working.nodes[flatten_outline(o), :2] for o in current],
                        layer_index=layer_index)
        outlines = current
        layer_outlines.append(current)
        layer_diagnostics.append(diagnostics)

    final_index, final_size = len(sizes) - 1, sizes[-1]
    fronts, dense_targets = [], []
    for boundary_index, outline in enumerate(outlines):
        plan = _call(lambda: plan_quad_sections(working, outline, final_size,
                         expand_ratio=ratio, pattern_lines=patterns,
                         small_element_size=small_element_size,
                         dense_region_tolerance=dense_region_tolerance),
                     stage="final_layer_planning", layer_index=final_index,
                     boundary_index=boundary_index)
        fronts.append(_call(lambda: section_outline_points(plan), stage="final_layer_planning",
                            layer_index=final_index, boundary_index=boundary_index))
        # Chain positions survive repair-node renumbering when another boundary
        # is built first. Use the exact canonical contacts sent to the global grid.
        contacts = {}
        for record in plan.diagnostics["dense_seams"]:
            point = plan.seams[record["seam_id"]].points[-1]
            nearest = int(np.argmin(np.linalg.norm(fronts[-1] - point, axis=1)))
            contacts[record["source_edge"], record["source_index"]] = fronts[-1][nearest].copy()
        dense_targets.append(contacts)
    validate_fronts(topology, fronts, stage="final_front_validation", layer_index=final_index)

    regions, global_diagnostics = [], []
    targets = [None] * len(outlines)
    for region_index, domain in enumerate(complement_domains(topology, fronts)):
        result = _call(lambda: generate_rectilinear_mesh_optimal(
                            final_size, domain.outer_outline, holes=domain.holes,
                            pattern_lines=patterns, coordinate_tolerance=topology.coordinate_tolerance),
                       stage="global_meshing", layer_index=final_index,
                       global_region_index=region_index)
        if domain.boundary_index is not None:
            targets[domain.boundary_index] = result.mesh.nodes[result.outer_outline, :2].copy()
        for boundary_index, hole_ids in zip(domain.hole_boundary_indices, result.hole_outlines):
            targets[boundary_index] = result.mesh.nodes[hole_ids, :2].copy()
        regions.append((domain, result))
        global_diagnostics.append({"region_index": region_index,
                                   "outer_boundary_index": domain.boundary_index,
                                   "hole_boundary_indices": list(domain.hole_boundary_indices),
                                   "diagnostics": result.diagnostics})

    current, diagnostics = [], []
    for boundary_index, (outline, target) in enumerate(zip(outlines, targets)):
        if target is None:
            raise QuadLayerError("a boundary has no global interface", stage="final_layer_construction",
                                 boundary_index=boundary_index, layer_index=final_index)
        result = _call(lambda: _generate_layer_adaptive_mesh(
                            working, outline, final_size, expand_ratio=ratio,
                            pattern_lines=patterns, new_outline=target, fixed_outer_boundary=True,
                            small_element_size=small_element_size,
                            dense_region_tolerance=dense_region_tolerance,
                            _dense_seam_targets=dense_targets[boundary_index]),
                       stage="final_layer_construction", layer_index=final_index,
                       boundary_index=boundary_index)
        current.append(result.outline)
        diagnostics.append(result.diagnostics)
    layer_outlines.append(current)
    layer_diagnostics.append(diagnostics)
    completed, outer_ids = _call(lambda: _assemble(working, current, regions), stage="global_assembly")
    _call(lambda: _validate_complete(completed, mesh, topology.outer_outline, outer_ids),
          stage="full_mesh_validation")
    diagnostics = {
        "element_sizes": sizes,
        "max_transition_count": max_transition_count,
        "min_size_ratio": min_size_ratio,
        "small_element_size": small_element_size,
        "dense_region_tolerance": dense_region_tolerance,
        "line_policy_active": len(sizes) > 1,
        "boundaries": [{"boundary_index": index, "parent_index": parent,
                        "kind": "exterior" if sign < 0 else "hole"}
                       for index, (parent, sign) in enumerate(zip(topology.parents, topology.signs))],
        "layers": layer_diagnostics,
        "global_regions": global_diagnostics,
        "added_nodes": completed.node_count - mesh.node_count,
        "added_elements": completed.element_count - mesh.element_count,
        "elapsed_seconds": perf_counter() - started,
    }
    mesh.replace_data(nodes=completed.nodes, elements=completed.elements)
    return FullAdaptiveMeshResult(mesh, outer_ids, layer_outlines, diagnostics)


__all__ = ["generate_full_adaptive_mesh", "FullAdaptiveMeshResult"]
