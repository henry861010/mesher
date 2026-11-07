"""Transactional single-layer growth using exclusively direct Quad4 builders."""
from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import numpy as np

from ...model import Mesh2D
from ._fixed_boundary import (fixed_boundary_nodes, prepare_fixed_boundary,
                              restore_fixed_boundary)
from ._geometry import area, cross
from ._limits import check_element_limit
from ._line_policy import (DEFAULT_MAX_TRANSITION_COUNT, validate_dense_parameters,
                           validate_line_parameters)
from ._planning.front import (_expansion_distances, _front, _normalize_new_outline,
                              _specified_front)
from ._planning.source import _fail, _prepare, _validate_old_patterns
from ._planning.sections import partition_sections
from ._construction.sections import fill_sections, _quality_summary
from ._types import (PreparedLayerSource, QuadLayerError, QuadLayerResult,
                     QuadSectionPlan, UnmeshableDirectQuadLayout)


def _validate_patch(*args):
    # Isolated validator is also an injectable final transaction boundary.
    from ._validation import validate_patch
    return validate_patch(*args)


def _world_nodes(nodes, source_mesh, data):
    """Restore staged nodes while retaining every source coordinate exactly."""
    result = nodes.copy()
    count = len(source_mesh.nodes)
    result[count:, :2] = nodes[count:, :2] * data["scale"] + data["origin"]
    result[:count] = source_mesh.nodes
    restored = (result[count:, :2] - data["origin"]) / data["scale"]
    if (not np.all(np.isfinite(result))
            or np.any(np.abs(restored - nodes[count:, :2]) > data["tol"])):
        raise UnmeshableDirectQuadLayout("mapping new nodes back exceeds the geometric tolerance",
                                        stage="world_coordinates")
    return result


def _world_repair_diagnostics(value, data, key=None):
    if key in {"polygon", "mouth_points"}:
        return (np.asarray(value)[..., :2] * data["scale"] + data["origin"]).tolist()
    if key in {"boundary_element_size", "max_element_size", "ring_target_sizes"}:
        return (np.asarray(value) * data["scale"]).tolist()
    if isinstance(value, dict):
        return {name: _world_repair_diagnostics(item, data, name) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_world_repair_diagnostics(item, data) for item in value]
    return value


def _prepare_plan(mesh, outline, element_size, pattern_lines, expand_ratio, axis_tolerance,
                  *, repair_slots=True, max_added_elements=None,
                  repair_max_element_size=None, new_outline=None,
                  small_element_size=None, dense_region_tolerance=1e-3,
                  _dense_seam_targets=None):
    small_element_size, dense_region_tolerance = validate_dense_parameters(
        small_element_size, dense_region_tolerance)
    if not isinstance(repair_slots, (bool, np.bool_)):
        _fail("repair_slots must be a boolean", stage="input_validation", parameter="repair_slots")
    if repair_max_element_size is not None:
        if (isinstance(repair_max_element_size, (bool, np.bool_))
                or not isinstance(repair_max_element_size, (int, float, np.integer, np.floating))):
            _fail("repair_max_element_size must be a finite positive scalar or None",
                  stage="input_validation", parameter="repair_max_element_size")
        try:
            repair_max_element_size = float(repair_max_element_size)
        except (ValueError, OverflowError) as error:
            raise QuadLayerError("repair_max_element_size must be a finite positive scalar",
                                 code="invalid_input", stage="input_validation",
                                 parameter="repair_max_element_size") from error
        if not np.isfinite(repair_max_element_size) or repair_max_element_size <= 0:
            _fail("repair_max_element_size must be finite and positive",
                  stage="input_validation", parameter="repair_max_element_size")
    try:
        data = _prepare(mesh, outline, element_size, pattern_lines, expand_ratio, axis_tolerance)
    except QuadLayerError as error:
        error.context.setdefault("stage", "input_validation")
        raise
    target = None if new_outline is None else _normalize_new_outline(new_outline, data)
    try:
        _validate_old_patterns(mesh, data)
    except QuadLayerError as error:
        error.context.setdefault("stage", "pattern_validation")
        raise
    working = mesh
    repair_diagnostics = {"slot_repairs": [], "corner_repairs": [], "repair_added_nodes": 0,
                          "repair_added_elements": 0}
    prepared_source = None
    if repair_slots and area(data["source"]) < 0:
        from ._repair.slots import repair_slots as stage_repairs
        _expansion_distances(data)
        normalized = Mesh2D(np.column_stack((data["xy"], np.zeros(len(mesh.nodes)))), mesh.elements)
        # Only the repair target uses this cap. The original h continues to
        # determine slot/corner coverage, pattern snapping, and the front.
        repair_cap = (None if repair_max_element_size is None else
                      min(repair_max_element_size, float(element_size)) / data["scale"])
        try:
            repair = stage_repairs(normalized, data, max_added_elements=max_added_elements,
                                   repair_max_element_size=repair_cap)
        except QuadLayerError as error:
            error.context.setdefault("stage", "slot_repair")
            raise
        if repair is not None:
            original_data = data
            working = Mesh2D(_world_nodes(repair.mesh.nodes, mesh, data), repair.mesh.elements)
            if (not np.array_equal(working.nodes[:len(mesh.nodes)], mesh.nodes)
                    or not np.array_equal(working.elements[:len(mesh.elements)], mesh.elements)):
                _fail("slot repair modified original mesh data", stage="slot_repair")
            repair_diagnostics = _world_repair_diagnostics(repair.diagnostics, data)
            repair_diagnostics.update(
                repair_added_nodes=len(working.nodes) - len(mesh.nodes),
                repair_added_elements=len(working.elements) - len(mesh.elements),
                prepared_outline=[list(map(int, chain)) for chain in repair.outline],
                edge_provenance=[list(map(int, edges)) for edges in repair.provenance])
            try:
                data = _prepare(working, repair.outline, element_size, pattern_lines,
                                expand_ratio, axis_tolerance,
                                _semantic_directions=repair.directions, _frame=original_data)
            except QuadLayerError as error:
                error.context.setdefault("stage", "repair_outline")
                raise
            data["_original_data"] = original_data
            prepared_source = PreparedLayerSource(
                working.nodes, working.elements, tuple(map(tuple, repair.outline)),
                len(mesh.nodes), len(mesh.elements))
    data["_source_mesh"] = working
    data["_repair_diagnostics"] = repair_diagnostics
    data.update(small_element_size=small_element_size,
                dense_region_tolerance=dense_region_tolerance)
    if _dense_seam_targets is not None:
        data["_dense_seam_targets"] = {
            key: (np.asarray(point) - data["origin"]) / data["scale"]
            for key, point in _dense_seam_targets.items()}
    actions = []
    if prepared_source is not None:
        actions.append(("pattern_validation", lambda: _validate_old_patterns(working, data)))
    actions.extend((("expansion", lambda: (_front(data) if target is None
                                           else _specified_front(data, target))),
                    ("section_partition", lambda: partition_sections(data))))
    for stage, action in actions:
        try:
            result = action()
        except QuadLayerError as error:
            error.context.setdefault("stage", stage)
            raise
    result = replace(result, prepared_source=prepared_source,
                     diagnostics={**result.diagnostics, **repair_diagnostics})
    return data, result


def plan_quad_sections(mesh: Mesh2D, outline, element_size: float, *, pattern_lines=(),
                       new_outline=None,
                       expand_ratio: float = 1.0,
                       axis_tolerance: float = 1e-3,
                       repair_slots: bool = True,
                       repair_max_element_size: float | None = None,
                       small_element_size: float | None = None,
                       dense_region_tolerance: float = 1e-3) -> QuadSectionPlan:
    """Inspect repaired source and section geometry without modifying input.

    All returned coordinates and expansion distances use the caller's frame.
    Original IDs are retained. With repairs, section source chains index the
    read-only ``prepared_source`` snapshot, which includes staged fill cells.
    Outer-layer construction remains a geometric preview: the subsequent
    seam/unit search may still find no solution. ``repair_slots=False`` keeps
    the original strict single-layer behavior.
    ``repair_max_element_size`` has the same interior target-size semantics as
    :func:`generate_layer_adaptive_mesh`, including automatic sizing when it is None.
    ``new_outline`` also follows that function's contract: an ordered XY target
    loop matching the repaired source. ``front`` contains its logical corners;
    section outer boundaries retain the supplied collinear points as anchors.
    At least one existing Quad4 is required. Pattern lengths and endpoint
    L-infinity distances from the source bounding box cannot exceed one million
    times its largest XY span. This preview has no element budget and makes no
    guarantee that construction fits a later call's ``max_added_elements``.
    ``small_element_size`` and ``dense_region_tolerance`` have the construction
    API's world-length semantics. Dense seams are part of this same preview;
    None leaves the original corner/pattern partition unchanged.
    """
    data, plan = _prepare_plan(mesh, outline, element_size, pattern_lines, expand_ratio,
                              axis_tolerance, repair_slots=repair_slots,
                              repair_max_element_size=repair_max_element_size,
                              new_outline=new_outline, small_element_size=small_element_size,
                              dense_region_tolerance=dense_region_tolerance)

    def world(points):
        result = points * data["scale"] + data["origin"]
        result.setflags(write=False)
        return result

    return replace(plan,
                   sections=tuple(replace(section, outer_boundary=world(section.outer_boundary))
                                  for section in plan.sections),
                   seams=tuple(replace(seam, points=world(seam.points)) for seam in plan.seams),
                   front=world(plan.front),
                   edge_expansion=tuple(d * data["scale"] for d in plan.edge_expansion))


def generate_layer_adaptive_mesh(mesh: Mesh2D, outline, element_size: float, *, pattern_lines=(),
                      new_outline=None,
                      expand_ratio: float = 1.0,
                      axis_tolerance: float = 1e-3,
                      max_added_elements: int = 1_000_000,
                      max_transition_count: int = DEFAULT_MAX_TRANSITION_COUNT,
                      min_size_ratio: float = 0.5,
                      repair_slots: bool = True,
                      repair_max_element_size: float | None = None,
                      small_element_size: float | None = None,
                      dense_region_tolerance: float = 1e-3) -> QuadLayerResult:
    """Append one conforming Quad4 layer, committing only after validation.

    ``outline`` is a closed chain of near-axis logical edges, CW for an
    exterior boundary and CCW for a hole. ``pattern_lines`` contains finite
    XY segments of shape ``(n, 2, 2)``. Growth is to the left of each directed
    edge. A parallel pattern within ``element_size * expand_ratio`` determines
    that whole edge's expansion; otherwise the expansion is ``element_size``.

    ``new_outline=None`` keeps that automatic expansion. Otherwise, supply an
    ordered finite XY array of shape ``(n, 2)`` describing the complete target
    loop. The last point connects to the first; a repeated closing point is
    optional. Either winding and any starting point are accepted. Target edges
    must be horizontal or vertical, with the same cyclic turns as the prepared
    source after repairs, and strictly on each source edge's expansion side.
    All supplied points, including collinear points, are mandatory within the
    geometric tolerance. Additional LINE points obey the size and transition
    limits below as well as Quad4 parity. Patterns still
    constrain mesh edges but never move or snap an explicitly supplied target.

    Closing simple exterior slots and covered isolated stair corners are filled
    first, unless ``repair_slots=False``. A stair corner is covered when either
    incident wall's planned expansion reaches the complete opposite leg.
    Its two old legs are retained and joined by a two-segment free cap.
    Repaired chains retain cardinal logical
    directions even when individual segments are oblique. Patterns are applied
    again to that prepared outline. Both stages share one output budget and
    one atomic commit; original node and element IDs remain unchanged.
    With ``new_outline``, repair decisions still use the original size, ratio,
    and patterns. Only the subsequent expansion uses the explicit target. An
    incompatible target fails atomically; it does not trigger another repair.

    Repair mouth and chord sampling target the median frozen segment length,
    capped at ``element_size``. Interior targets double inward up to
    ``min(element_size, 4 * boundary_target)`` by default. A finite positive
    ``repair_max_element_size`` overrides that interior target cap, must be at
    least each repair's boundary target, and is capped at ``element_size``.
    It is not a hard maximum on every mesh edge. Inward construction stops before
    a boundary collapse and fills the remaining core; it never restarts repair.

    At least one existing Quad4 is required. Each pattern's length and each
    endpoint's L-infinity distance from the source bounding box must be at most
    one million times that box's largest XY span, inclusive. Out-of-range
    patterns raise ``QuadLayerError`` with code ``numeric_range_exceeded``.

    ``max_added_elements`` is a positive Python or NumPy integer limiting only
    the new Quad4 cells in this call; bool, float and None are not accepted.
    The default is one million, and equality is allowed. A count-only preflight
    and actual candidate/layer checks raise ``element_limit_exceeded`` before
    committing an oversized result. Trials do not consume cumulative budget.
    The solver does not search for a smaller assignment if its selected layer
    exceeds the limit. This is not a memory or execution-time guarantee.

    ``max_transition_count`` is a positive Python or NumPy integer, default 20.
    Each resampled LINE outer segment may receive at most this many source
    segments; reverse refinement is unrestricted. ``min_size_ratio`` is finite
    and in (0, 1], default 0.5. LINE outer spacing lies between
    ``min_size_ratio * element_size`` and ``element_size``, within tolerance.
    Adjacent mandatory anchors closer than that minimum remain one unsplit
    segment and are recorded as an exception; the transition cap still applies.
    These constraints do not change layer thickness, source edges, corner
    sections, or repair interiors. Density-based sampling and bounded local
    pairing favor source alignment without searching for a globally best mesh.

    Corners are reserved before straight LINE sections. Finite patterns may
    follow boundaries or form complete source-to-front section interfaces;
    interior endpoints and interior pattern junctions are unsupported.
    Source nodes, segments, and elements are immutable. Outside resampled LINE
    boundaries, size remains a target rather than a bound on every element edge.
    Unsupported sections or incompatible seam states raise :class:`QuadLayerError`;
    failures leave ``mesh`` untouched. There is no thin-collar retry.
    No triangular mesh is constructed, including inside temporary builders.

    ``small_element_size=None`` disables dense-run partitioning. A finite positive
    value splits each LINE at the ends of maximal runs of source edges strictly
    shorter than this length. These endpoints become shared seams to projected
    outer anchors. Existing anchors closer than ``dense_region_tolerance``
    (finite, non-negative, default 0.001) are reused. Both parameters use world
    length units; this merge tolerance never replaces geometric validation.
    """
    return _generate_layer_adaptive_mesh(
        mesh, outline, element_size, pattern_lines=pattern_lines, new_outline=new_outline,
        expand_ratio=expand_ratio, axis_tolerance=axis_tolerance,
        max_added_elements=max_added_elements, repair_slots=repair_slots,
        repair_max_element_size=repair_max_element_size,
        max_transition_count=max_transition_count, min_size_ratio=min_size_ratio,
        small_element_size=small_element_size, dense_region_tolerance=dense_region_tolerance)


def _generate_layer_adaptive_mesh(mesh: Mesh2D, outline, element_size: float, *,
                                 pattern_lines=(), new_outline=None,
                                 expand_ratio: float = 1.0,
                                 axis_tolerance: float = 1e-3,
                                 max_added_elements: int = 1_000_000,
                                 max_transition_count: int = DEFAULT_MAX_TRANSITION_COUNT,
                                 min_size_ratio: float = 0.5,
                                 repair_slots: bool = True,
                                 repair_max_element_size: float | None = None,
                                 fixed_outer_boundary: bool = False,
                                 small_element_size: float | None = None,
                                 dense_region_tolerance: float = 1e-3,
                                 _dense_seam_targets=None) -> QuadLayerResult:
    """Shared transaction; the private fixed mode consumes a complete interface.

    Fixed targets include every section's front contact and are never sampled
    again. Their exact world coordinates survive the normalized construction.
    """
    start = perf_counter()
    max_transition_count, min_size_ratio = validate_line_parameters(
        max_transition_count, min_size_ratio)
    if not isinstance(fixed_outer_boundary, (bool, np.bool_)):
        _fail("fixed_outer_boundary must be a boolean", stage="input_validation",
              parameter="fixed_outer_boundary")
    if fixed_outer_boundary and new_outline is None:
        _fail("fixed_outer_boundary requires new_outline", stage="input_validation",
              parameter="new_outline")
    if (isinstance(max_added_elements, (bool, np.bool_))
            or not isinstance(max_added_elements, (int, np.integer))
            or max_added_elements <= 0):
        _fail("max_added_elements must be a positive integer", "invalid_input",
              stage="input_validation", parameter="max_added_elements")
    max_added_elements = int(max_added_elements)
    data, plan = _prepare_plan(mesh, outline, element_size, pattern_lines, expand_ratio,
                              axis_tolerance, repair_slots=repair_slots,
                              max_added_elements=max_added_elements,
                              repair_max_element_size=repair_max_element_size,
                              new_outline=new_outline, small_element_size=small_element_size,
                              dense_region_tolerance=dense_region_tolerance,
                              _dense_seam_targets=_dense_seam_targets)
    data["max_transition_count"] = max_transition_count
    data["min_size_ratio"] = min_size_ratio
    if fixed_outer_boundary:
        prepare_fixed_boundary(plan, data, new_outline)
    working = data["_source_mesh"]
    repair_count = len(working.elements) - len(mesh.elements)
    data["max_added_elements"] = max_added_elements - repair_count
    local_nodes = working.nodes.copy()
    local_nodes[:, :2] = data["xy"]
    try:
        nodes, quads, result_outline, registry_diagnostics = fill_sections(
            local_nodes, plan, data, element_offset=len(working.elements))
    except QuadLayerError as error:
        if error.code == "element_limit_exceeded":
            error.context["limit"] = max_added_elements
            error.context["required_elements"] += repair_count
        error.context.setdefault("stage", "section_assembly")
        raise
    try:
        if fixed_outer_boundary:
            fixed_boundary_nodes(nodes, result_outline, data)
        _validate_patch(working, nodes, quads, result_outline, data)
        all_quads = np.vstack((working.elements[len(mesh.elements):], quads)) if repair_count else quads
        if repair_count:
            original_data = dict(data["_original_data"])
            original_data.update(front=data["front"], outer=data["front"],
                                 inner=original_data["source"], tol=data["tol"])
            _validate_patch(mesh, nodes, all_quads, result_outline, original_data)
    except QuadLayerError as error:
        error.context.setdefault("stage", "final_validation")
        raise
    world_nodes = _world_nodes(nodes, working, data)
    if fixed_outer_boundary:
        restore_fixed_boundary(world_nodes, nodes, result_outline, data, len(working.nodes))
        nodes = world_nodes.copy()
        nodes[:, :2] = (world_nodes[:, :2] - data["origin"]) / data["scale"]
        try:
            fixed_boundary_nodes(nodes, result_outline, data)
            _validate_patch(working, nodes, quads, result_outline, data)
            if repair_count:
                _validate_patch(mesh, nodes, all_quads, result_outline, original_data)
        except QuadLayerError as error:
            error.context.setdefault("stage", "fixed_boundary_world_validation")
            raise
    # Verify strict convexity again after restoring the caller's coordinate frame.
    represented = (world_nodes[all_quads, :2] - data["origin"]) / data["scale"]
    vectors = np.roll(represented, -1, axis=1) - represented
    if not np.all(np.isfinite(world_nodes)) or np.any(cross(vectors, np.roll(vectors, -1, axis=1)) <= 0):
        raise UnmeshableDirectQuadLayout("new quads cannot be represented at the input coordinate scale",
                                        stage="world_coordinates")
    check_element_limit(len(all_quads), max_added_elements, stage="precommit")
    elements = np.vstack((mesh.elements, all_quads))
    diagnostics = {**plan.diagnostics, **registry_diagnostics,
                   "added_nodes": len(nodes) - len(mesh.nodes), "added_elements": len(all_quads),
                   "expand_added_nodes": len(nodes) - len(working.nodes),
                   "expand_added_elements": len(quads),
                   "quality": _quality_summary(nodes, all_quads),
                   "max_added_elements": max_added_elements,
                   "max_transition_count": max_transition_count,
                   "min_size_ratio": min_size_ratio,
                   "line_policy_active": not fixed_outer_boundary,
                   "edge_expansion": [d * data["scale"] for d in data["expansion"]],
                   "snapped_patterns": data["snapped"], "elapsed_seconds": perf_counter() - start}
    # This is the only mutation of the caller's mesh.
    mesh.replace_data(nodes=world_nodes, elements=elements)
    return QuadLayerResult(mesh=mesh, outline=result_outline, diagnostics=diagnostics)
