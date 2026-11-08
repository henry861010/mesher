"""Transactional single-layer growth using exclusively direct Quad4 builders."""
from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import numpy as np
from mesher.tolerance import GeometryTolerance, resolve_tolerance
from numpy.typing import ArrayLike

from ...model import Mesh2D
from ._fixed_boundary import (fixed_boundary_nodes, prepare_fixed_boundary,
                              restore_fixed_boundary)
from ._geometry import area, cross
from ._limits import check_element_limit
from ._line_policy import (DEFAULT_MAX_TRANSITION_COUNT, validate_dense_parameters,
                           validate_line_parameters)
from ._planning.front import (_expansion_distances, _front, _normalize_outline_o,
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
            or np.any(np.abs(restored - nodes[count:, :2]) > data["numeric_tol"])):
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


def _prepare_plan(mesh, outline_i, element_size, pattern_lines, expand_ratio, tolerance,
                  *, expand_size=None, repair_slots=True, max_added_elements=None,
                  repair_max_element_size=None, outline_o=None, fixed_outer_boundary=False,
                  small_element_size=None, dense_mapping_angle=30.0,
                  _dense_seam_targets=None):
    try:
        tolerance = resolve_tolerance(tolerance)
    except (TypeError, ValueError, OverflowError) as error:
        raise QuadLayerError("tolerance must be finite and positive", code="invalid_input",
                             stage="input_validation", parameter="tolerance") from error
    if not isinstance(fixed_outer_boundary, (bool, np.bool_)):
        _fail("fixed_outer_boundary must be a boolean", stage="input_validation",
              parameter="fixed_outer_boundary")
    if fixed_outer_boundary and outline_o is None:
        _fail("fixed_outer_boundary requires outline_o", stage="input_validation",
              parameter="outline_o")
    small_element_size, dense_mapping_angle = validate_dense_parameters(
        small_element_size, dense_mapping_angle)
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
        data = _prepare(mesh, outline_i, element_size, pattern_lines, expand_ratio, tolerance,
                        expand_size=expand_size)
    except QuadLayerError as error:
        error.context.setdefault("stage", "input_validation")
        raise
    target = None if outline_o is None else _normalize_outline_o(outline_o, data)
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
        # Repair sampling uses element_size; coverage and the front use expand_h.
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
                prepared_outline=[int(node) for chain in repair.outline for node in chain[:-1]],
                edge_provenance=[list(map(int, edges)) for edges in repair.provenance])
            try:
                data = _prepare(working, repair.outline, element_size, pattern_lines,
                                expand_ratio, tolerance, expand_size=expand_size,
                                _semantic_directions=repair.directions, _frame=original_data)
            except QuadLayerError as error:
                error.context.setdefault("stage", "repair_outline")
                raise
            data["_original_data"] = original_data
            prepared_source = PreparedLayerSource(
                working.nodes, working.elements, tuple(repair_diagnostics["prepared_outline"]),
                len(mesh.nodes), len(mesh.elements), tolerance=data["tolerance"])
    data["_source_mesh"] = working
    data["_repair_diagnostics"] = repair_diagnostics
    data.update(fixed_outer_boundary=bool(fixed_outer_boundary),
                small_element_size=small_element_size,
                dense_mapping_angle=dense_mapping_angle)
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
    if fixed_outer_boundary:
        prepare_fixed_boundary(result, data)
    result = replace(result, prepared_source=prepared_source, tolerance=data["tolerance"],
                     diagnostics={**result.diagnostics, **repair_diagnostics,
                                  "geometry_tolerance": data["tolerance"].absolute,
                                  "numerical_error_budget": {
                                      "normalized_length": data["numeric_tol"],
                                      "world_length": data["numeric_tol"] * data["scale"],
                                      "normalization_scale": data["scale"]}})
    return data, result


def plan_quad_sections(mesh: Mesh2D, outline_i: ArrayLike, element_size: float, *, pattern_lines=(),
                       outline_o: ArrayLike | None = None,
                       fixed_outer_boundary: bool = False,
                       expand_size: float | None = None,
                       expand_ratio: float = 1.0,
                       tolerance: GeometryTolerance | float | None = None,
                       repair_slots: bool = True,
                       repair_max_element_size: float | None = None,
                       small_element_size: float | None = None,
                       dense_mapping_angle: float = 30.0) -> QuadSectionPlan:
    """Inspect repaired source and section geometry without modifying input.

    All returned coordinates and expansion distances use the caller's frame.
    ``expand_size`` has the construction API's world-length semantics: None
    uses ``element_size`` as the expansion baseline, including pattern snapping.
    ``outline_i`` follows the construction API's flat node-ID loop contract,
    including caller-supplied orientation and automatic logical-edge grouping.
    Growth is to the left: exterior loops must be clockwise, holes counterclockwise.
    The caller guarantees that every source segment is an existing boundary edge.
    Original IDs are retained. With repairs, section source chains index the
    read-only ``prepared_source`` snapshot, which includes staged fill cells.
    Outer-layer construction remains a geometric preview: the subsequent
    seam/unit search may still find no solution. ``repair_slots=False`` keeps
    the original strict single-layer behavior.
    ``repair_max_element_size`` has the same interior target-size semantics as
    :func:`generate_layer_adaptive_mesh`, including automatic sizing when it is None.
    ``outline_o`` also follows that function's contract: an integer node-ID target
    loop matching the repaired source. ``front`` contains its logical corners;
    section outer boundaries retain the supplied collinear points as anchors.
    At least one existing Quad4 is required. Pattern lengths and endpoint
    L-infinity distances from the source bounding box cannot exceed one million
    times its largest XY span. This preview has no element budget and makes no
    guarantee that construction fits a later call's ``max_added_elements``.
    ``fixed_outer_boundary=True`` requires ``outline_o`` and previews the same
    existing-node corner/dense mapping as construction, including fixed-boundary
    parity and contact validation. It does not guarantee a feasible unit mesh.
    ``small_element_size`` is a world length; None disables dense partitioning.
    ``dense_mapping_angle`` is in degrees, default 30, and must be in [0, 90).
    Each optional endpoint maps independently, as in the construction API.
    """
    data, plan = _prepare_plan(mesh, outline_i, element_size, pattern_lines, expand_ratio,
                              tolerance, expand_size=expand_size, repair_slots=repair_slots,
                              repair_max_element_size=repair_max_element_size,
                              outline_o=outline_o, fixed_outer_boundary=fixed_outer_boundary,
                              small_element_size=small_element_size,
                              dense_mapping_angle=dense_mapping_angle)

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


def generate_layer_adaptive_mesh(mesh: Mesh2D, outline_i: ArrayLike, element_size: float, *, pattern_lines=(),
                      outline_o: ArrayLike | None = None,
                      fixed_outer_boundary: bool = False,
                      expand_size: float | None = None,
                      expand_ratio: float = 1.0,
                      tolerance: GeometryTolerance | float | None = None,
                      max_added_elements: int = 1_000_000,
                      max_transition_count: int = DEFAULT_MAX_TRANSITION_COUNT,
                      min_size_ratio: float = 0.5,
                      repair_slots: bool = True,
                      repair_max_element_size: float | None = None,
                      small_element_size: float | None = None,
                      dense_mapping_angle: float = 30.0) -> QuadLayerResult:
    """Append one conforming Quad4 layer, committing only after validation.

    ``outline_i`` is a one-dimensional integer node-ID loop containing every
    source boundary node. The caller must supply clockwise exterior loops and
    counterclockwise hole loops, placing existing mesh to the right. The API
    preserves that direction and always grows to the left; it does not infer
    ownership or reverse the source. The caller guarantees that each consecutive
    pair is an existing boundary edge available for attachment. These source
    preconditions are not checked against element ownership; violations have no
    guaranteed early rejection or dedicated error code.
    Any starting point is accepted; closure is implicit, with an optional
    repeated first ID at the end. The API groups consecutive segments by their
    signed dominant X/Y direction into near-axis logical edges. It never
    moves source coordinates or omits intermediate nodes. ``result.outline``
    is a flat list of node IDs, reusable as the next call's ``outline_i``.
    ``tolerance`` is a policy or positive absolute distance, defaulting to
    1e-3. Every near-axis chain's total transverse drift must fit that distance;
    the criterion does not grow with the chain length. Numerical area and
    Jacobian checks retain their separate roundoff thresholds.
    ``pattern_lines`` contains finite XY segments of shape ``(n, 2, 2)``. Growth
    is to the left of each directed edge. ``expand_size=None`` uses
    ``element_size`` as the expansion baseline. A finite positive ``expand_size``
    sets that baseline independently in world length units, while ``element_size``
    continues to control mesh sizing. A parallel pattern within
    ``baseline * expand_ratio`` determines that whole edge's expansion;
    otherwise the expansion is the baseline.

    ``outline_o=None`` keeps that automatic expansion. Otherwise, supply an
    one-dimensional integer node-ID loop in ``mesh.nodes`` describing the complete
    target. Nodes may already belong to existing elements; consecutive IDs need
    not be existing edges. The last ID connects to the first; a repeated closing
    ID is optional. Either winding and any starting point are accepted. Target edges
    must be horizontal or vertical, with the same cyclic turns as the prepared
    source after repairs, and strictly on each source edge's expansion side.
    All supplied nodes, including collinear nodes, retain their original IDs
    and exact coordinates. Additional LINE points obey the size and transition
    limits below as well as Quad4 parity. Patterns still
    constrain mesh edges but never move or snap an explicitly supplied target.
    Existing elements are never split; an incompatible shared edge fails validation.

    ``fixed_outer_boundary=True`` requires ``outline_o`` and preserves its complete
    node-ID cycle and edges without adding, dropping, or moving boundary nodes.
    Interior nodes may still be created. LINE and corner outer resampling are
    disabled, as are the ``max_transition_count`` and ``min_size_ratio`` policies;
    both parameters are still validated. The element budget remains active.
    Free OUTER corner contacts use the nearest usable nodes on the corresponding
    target edges, excluding that corner itself. Conflicting corner assignments
    try further candidates without an angle cap. Pattern contacts must already
    exist in the supplied target. Fixed source/target boundary counts must have
    matching parity, and unit feasibility is still checked before committing.

    Closing simple exterior slots and covered isolated stair corners are filled
    first, unless ``repair_slots=False``. A stair corner is covered when either
    incident wall's planned expansion reaches the complete opposite leg.
    Its two old legs are retained and joined by a two-segment free cap.
    Repaired chains retain cardinal logical
    directions even when individual segments are oblique. Patterns are applied
    again to that prepared outline. Both stages share one output budget and
    one atomic commit; original node and element IDs remain unchanged.
    Repair coverage uses the expansion baseline, ratio, and patterns, including
    when ``outline_o`` is supplied. Only the subsequent expansion uses that
    explicit target. An incompatible target fails atomically; it does not
    trigger another repair.

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
    ``min_size_ratio * element_size`` and ``element_size``, within numerical roundoff.
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
    shorter than this world length. ``dense_mapping_angle`` is finite and in
    [0, 90) degrees, default 30. From the outward normal, each begin endpoint
    searches toward the preceding source direction in [-angle, 0]; each end
    searches forward in [0, angle]. Only the first geometrically usable outer
    contact, ordered by increasing absolute angle, is selected. Contacts stay
    inside their corner/pattern-delimited LINE. Each endpoint is independent:
    if none exists, ordinary mode creates a normal projection when valid;
    fixed mode skips only that optional cut. Mandatory seams remain unchanged.
    The removed ``dense_region_tolerance`` keyword has no compatibility alias.
    """
    return _generate_layer_adaptive_mesh(
        mesh, outline_i, element_size, pattern_lines=pattern_lines, outline_o=outline_o,
        fixed_outer_boundary=fixed_outer_boundary,
        expand_size=expand_size, expand_ratio=expand_ratio, tolerance=tolerance,
        max_added_elements=max_added_elements, repair_slots=repair_slots,
        repair_max_element_size=repair_max_element_size,
        max_transition_count=max_transition_count, min_size_ratio=min_size_ratio,
        small_element_size=small_element_size, dense_mapping_angle=dense_mapping_angle)


def _generate_layer_adaptive_mesh(mesh: Mesh2D, outline_i, element_size: float, *,
                                 pattern_lines=(), outline_o=None,
                                 expand_size: float | None = None,
                                 expand_ratio: float = 1.0,
                                 tolerance: GeometryTolerance | float | None = None,
                                 max_added_elements: int = 1_000_000,
                                 max_transition_count: int = DEFAULT_MAX_TRANSITION_COUNT,
                                 min_size_ratio: float = 0.5,
                                 repair_slots: bool = True,
                                 repair_max_element_size: float | None = None,
                                 fixed_outer_boundary: bool = False,
                                 small_element_size: float | None = None,
                                 dense_mapping_angle: float = 30.0,
                                 _dense_seam_targets=None) -> QuadLayerResult:
    """Shared transaction for public layers and full-mesh final interfaces.

    Fixed targets include every section's front contact and are never sampled
    again. Their exact world coordinates survive the normalized construction.
    """
    start = perf_counter()
    max_transition_count, min_size_ratio = validate_line_parameters(
        max_transition_count, min_size_ratio)
    if not isinstance(fixed_outer_boundary, (bool, np.bool_)):
        _fail("fixed_outer_boundary must be a boolean", stage="input_validation",
              parameter="fixed_outer_boundary")
    if fixed_outer_boundary and outline_o is None:
        _fail("fixed_outer_boundary requires outline_o", stage="input_validation",
              parameter="outline_o")
    if (isinstance(max_added_elements, (bool, np.bool_))
            or not isinstance(max_added_elements, (int, np.integer))
            or max_added_elements <= 0):
        _fail("max_added_elements must be a positive integer", "invalid_input",
              stage="input_validation", parameter="max_added_elements")
    max_added_elements = int(max_added_elements)
    data, plan = _prepare_plan(mesh, outline_i, element_size, pattern_lines, expand_ratio,
                              tolerance, expand_size=expand_size, repair_slots=repair_slots,
                              max_added_elements=max_added_elements,
                              repair_max_element_size=repair_max_element_size,
                              outline_o=outline_o, fixed_outer_boundary=fixed_outer_boundary,
                              small_element_size=small_element_size,
                              dense_mapping_angle=dense_mapping_angle,
                              _dense_seam_targets=_dense_seam_targets)
    data["max_transition_count"] = max_transition_count
    data["min_size_ratio"] = min_size_ratio
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
            if "outline_anchor_ids" in data:
                original_data.update(outline_anchor_ids=data["outline_anchor_ids"])
            _validate_patch(mesh, nodes, all_quads, result_outline, original_data)
    except QuadLayerError as error:
        error.context.setdefault("stage", "final_validation")
        raise
    world_nodes = _world_nodes(nodes, working, data)
    if fixed_outer_boundary:
        restore_fixed_boundary(world_nodes, nodes, result_outline, data)
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
                   "fixed_outer_boundary": bool(fixed_outer_boundary),
                   "line_policy_active": not fixed_outer_boundary,
                   "edge_expansion": [d * data["scale"] for d in data["expansion"]],
                   "snapped_patterns": data["snapped"], "elapsed_seconds": perf_counter() - start}
    # This is the only mutation of the caller's mesh.
    mesh.replace_data(nodes=world_nodes, elements=elements)
    return QuadLayerResult(
        mesh=mesh, outline=[int(node) for chain in result_outline for node in chain[:-1]],
        diagnostics=diagnostics, tolerance=data["tolerance"])
