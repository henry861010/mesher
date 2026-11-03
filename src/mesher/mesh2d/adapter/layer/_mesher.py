"""Transactional single-layer growth using exclusively direct Quad4 builders."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from time import perf_counter

import numpy as np

from ...model import Mesh2D, _validate_zero_z
from ._geometry import (BoxIndex, area, cross, intervals_cover,
                        point_location, segment_contacts)
from ._patterns import (PATTERN_RANGE_LIMIT, AxisPattern, covered_intervals,
                        occupied_intervals, pattern_contacts)
from ._limits import check_element_limit
from ._sections import partition_sections
from ._section_mesh import fill_sections, _quality_summary
from ._types import (PreparedLayerSource, QuadLayerError, QuadLayerResult,
                     QuadSectionPlan, UnmeshableDirectQuadLayout)


def _fail(message, code="invalid_input", **context):
    raise QuadLayerError(message, code=code, **context)


def _edge_incidence(elements):
    incidence = defaultdict(list)
    for i, cell in enumerate(elements):
        for a, b in zip(cell, np.roll(cell, -1)):
            incidence[tuple(sorted((int(a), int(b))))].append((i, int(a), int(b)))
    return incidence


def _simple_loop(polygon, tol, name):
    following = np.roll(polygon, -1, axis=0)
    lower, upper = np.minimum(polygon, following), np.maximum(polygon, following)
    index = BoxIndex(lower, upper, tol)
    for i, (a, b) in enumerate(zip(polygon, following)):
        for j in index.query(lower[i] - tol, upper[i] + tol):
            if j <= i or j == i + 1 or (i == 0 and j == len(polygon) - 1):
                continue
            if segment_contacts(a, b, polygon[j], following[j], tol):
                _fail(f"{name} intersects or touches itself", "topology_change", edges=(i, int(j)))


def _normalize_patterns(pattern_lines, origin, scale, tol, source):
    def range_failure(reason, index=None):
        raise QuadLayerError("patterns exceed the source-relative numeric range",
                             code="numeric_range_exceeded", stage="input_validation",
                             pattern_index=index, limit=PATTERN_RANGE_LIMIT, reason=reason)

    try:
        patterns = np.asarray(pattern_lines, dtype=float)
    except OverflowError:
        range_failure("coordinate_conversion")
    except (ValueError, TypeError) as exc:
        raise QuadLayerError("patterns must have shape (n, 2, 2)") from exc
    if patterns.size == 0:
        return np.empty((0, 2, 2))
    if patterns.ndim != 3 or patterns.shape[1:] != (2, 2) or not np.all(np.isfinite(patterns)):
        _fail("patterns must be finite XY segments with shape (n, 2, 2)")
    with np.errstate(over="ignore", invalid="ignore"):
        patterns = (patterns - origin) / scale
    # Source coordinates are within one unit of the chosen source-node origin.
    # Bound endpoints before subtraction, lengths, or spatial-index queries.
    lower, upper = source.min(axis=0), source.max(axis=0)
    result = []
    for i, segment in enumerate(patterns):
        if (not np.all(np.isfinite(segment))
                or np.any(np.abs(segment) > PATTERN_RANGE_LIMIT + 1.0)):
            range_failure("endpoint_distance", i)
        distance = np.maximum(lower - segment, segment - upper)
        if np.any(distance > PATTERN_RANGE_LIMIT):
            range_failure("endpoint_distance", i)
        delta = segment[1] - segment[0]
        axis = int(np.argmax(np.abs(delta)))
        if abs(delta[axis]) > PATTERN_RANGE_LIMIT:
            range_failure("length", i)
        if abs(delta[axis]) <= tol or abs(delta[1 - axis]) > tol:
            _fail("patterns must be nonzero horizontal or vertical segments", pattern_index=i)
        segment[:, 1 - axis] = segment[:, 1 - axis].mean()
        if delta[axis] < 0:
            segment = segment[::-1]
        result.append(segment)
    # Geometric classification merges duplicates later, retaining original IDs.
    return np.asarray(result)


def _prepare(mesh, outline, element_size, pattern_lines, expand_ratio, axis_tolerance,
             *, _semantic_directions=None, _frame=None):
    if not isinstance(mesh, Mesh2D):
        _fail("mesh must be a Mesh2D")
    try:
        _validate_zero_z(mesh.nodes)
    except ValueError as exc:
        raise QuadLayerError(str(exc)) from exc
    if not len(mesh.elements):
        _fail("layer extension requires at least one existing Quad4 element", "empty_mesh")
    try:
        h, ratio, axis_tol = float(element_size), float(expand_ratio), float(axis_tolerance)
    except (TypeError, ValueError) as exc:
        raise QuadLayerError("size, ratio and tolerance must be finite scalars") from exc
    if not np.isfinite(h) or h <= 0:
        _fail("element_size must be finite and positive")
    if not np.isfinite(ratio) or ratio < 1:
        _fail("expand_ratio must be finite and at least 1")
    if not np.isfinite(axis_tol) or not 0 <= axis_tol < 1:
        _fail("axis_tolerance must be finite and in [0, 1)")
    if not np.all(np.isfinite(mesh.nodes)):
        _fail("all mesh nodes must be finite")
    if len(mesh.elements) and (np.any(mesh.elements < 0) or np.any(mesh.elements >= len(mesh.nodes))):
        _fail("mesh contains out-of-range node IDs")
    if len(mesh.elements) and np.any(np.diff(np.sort(mesh.elements, axis=1), axis=1) == 0):
        _fail("input elements must be Quad4 with four distinct node IDs", "quad_only_input")
    try:
        if len(outline) < 4:
            _fail("outline needs at least four logical edges")
        ids = []
        for chain in outline:
            raw = np.asarray(chain)
            if raw.ndim != 1 or len(raw) < 2 or not np.issubdtype(raw.dtype, np.integer):
                _fail("each outline edge must be a chain of at least two integer node IDs")
            if np.any(raw < 0) or np.any(raw >= len(mesh.nodes)):
                _fail("outline contains out-of-range node IDs")
            ids.append(raw.astype(np.int64, copy=True))
    except TypeError as exc:
        raise QuadLayerError("outline must be a sequence of node-index chains") from exc
    if any(a[-1] != b[0] for a, b in zip(ids, ids[1:] + ids[:1])):
        _fail("each edge end must equal the next edge start, including loop closure")
    flat = np.concatenate([edge[:-1] for edge in ids])
    if len(set(flat.tolist())) != len(flat):
        _fail("outline repeats a node other than at consecutive edge joins")
    original = mesh.nodes[flat, :2]
    origin = original[0].copy() if _frame is None else _frame["origin"].copy()
    scale = float(np.max(np.ptp(original, axis=0))) if _frame is None else _frame["scale"]
    if not np.isfinite(scale) or scale <= 0:
        _fail("source boundary has zero or unrepresentable extent")
    xy = (mesh.nodes[:, :2] - origin) / scale
    h /= scale
    if not np.isfinite(h) or h <= 0 or not np.all(np.isfinite(xy)):
        _fail("geometry and target size have incompatible numeric scales")
    tol = max(128 * np.finfo(float).eps,
              4 * abs(float(np.spacing(np.max(np.abs(original)))) / scale))
    if _frame is not None:
        tol = max(tol, _frame["tol"])
    if h <= 32 * tol:
        raise UnmeshableDirectQuadLayout("element_size is below geometric resolution")
    source = xy[flat]
    chains = [xy[chain].copy() for chain in ids]
    if _semantic_directions is not None:
        semantic = np.asarray(_semantic_directions, dtype=float)
        if (semantic.shape != (len(chains), 2)
                or not np.all(np.isin(semantic, (-1., 0., 1.)))
                or not np.all(np.sum(np.abs(semantic), axis=1) == 1)):
            _fail("prepared source directions must be cardinal unit vectors",
                  code="unsupported_slot_repair", stage="repair_outline")
    directions = []
    for i, chain in enumerate(chains):
        delta = chain[-1] - chain[0]
        axis = int(np.argmax(np.abs(delta if _semantic_directions is None else semantic[i])))
        length = abs(delta[axis])
        direction = np.zeros(2)
        direction[axis] = 1 if delta[axis] >= 0 else -1
        if _semantic_directions is not None:
            direction = semantic[i].copy()
        if length <= tol or (_semantic_directions is None
                             and np.ptp(chain[:, 1 - axis]) > axis_tol * length + tol):
            _fail("outline edges must be near a global X or Y axis", edge_index=i)
        if np.any(np.diff(chain @ direction) <= tol):
            _fail("nodes on each logical edge must be strictly ordered", edge_index=i)
        directions.append(direction)
    directions = np.asarray(directions)
    if np.any(np.sum(directions * np.roll(directions, -1, axis=0), axis=1) != 0):
        _fail("adjacent logical edges must be perpendicular; combine collinear chains")
    _simple_loop(source, tol, "source")
    if abs(area(source)) <= tol:
        _fail("source boundary has zero area")
    normals = np.column_stack((-directions[:, 1], directions[:, 0]))
    frozen = np.vstack([np.column_stack((edge[:-1], edge[1:])) for edge in ids])
    incidence = _edge_incidence(mesh.elements)
    cells = xy[mesh.elements]
    if len(cells):
        edges = np.roll(cells, -1, axis=1) - cells
        turns = cross(edges, np.roll(edges, -1, axis=1))
        threshold = tol * np.maximum(np.linalg.norm(edges, axis=2),
                                    np.linalg.norm(np.roll(edges, -1, axis=1), axis=2))
        if np.any(~(np.all(turns > threshold, axis=1) | np.all(turns < -threshold, axis=1))):
            _fail("existing Quad4 elements must be nondegenerate and strictly convex")
        for a, b in frozen:
            owners = incidence.get(tuple(sorted((int(a), int(b)))), ())
            if len(owners) != 1:
                _fail("each source segment must be an existing boundary edge", "fixed_boundary_conflict")
            centroid = cells[owners[0][0]].mean(axis=0)
            if cross(xy[b] - xy[a], centroid - xy[a]) >= -tol * np.linalg.norm(xy[b] - xy[a]):
                _fail("outline direction must place existing mesh on its right", "outline_orientation")
    patterns = _normalize_patterns(pattern_lines, origin, scale, tol, source)
    pattern_records = tuple(AxisPattern.from_points(points, (i,))
                            for i, points in enumerate(patterns))
    return dict(xy=xy, ids=ids, source=source, chains=chains, normals=normals,
                directions=directions, frozen=frozen, patterns=patterns,
                pattern_records=pattern_records,
                h=h, ratio=ratio, tol=tol, origin=origin, scale=scale,
                old_incidence=incidence)


def _expansion_distances(data):
    """Plan supports and pattern distances before any front can collapse."""
    chains, normals = data["chains"], data["normals"]
    tol, h = data["tol"], data["h"]
    supports, distances, snapped = [], [], []
    for chain, normal, tangent in zip(chains, normals, data["directions"]):
        support = float(np.max(chain @ normal))
        lo, hi = min(chain @ tangent), max(chain @ tangent)
        nearest, pattern_id = float("inf"), None
        for j, pattern in enumerate(data["pattern_records"]):
            if normal[pattern.axis] != 0:
                continue
            interval = sorted((pattern.lo * tangent[pattern.axis],
                               pattern.hi * tangent[pattern.axis]))
            if interval[1] < lo - tol or interval[0] > hi + tol:
                continue
            distance = float(pattern.support * normal[1 - pattern.axis]) - support
            if tol < distance < nearest:
                nearest, pattern_id = distance, j
        distance = nearest if np.isfinite(nearest) and nearest / h <= data["ratio"] + tol else h
        supports.append(support + distance)
        distances.append(distance)
        snapped.append(pattern_id if distance == nearest else None)
    data.update(expansion=distances, snapped=snapped, supports=supports)
    return supports


def _front(data):
    chains, normals, tol = data["chains"], data["normals"], data["tol"]
    supports = _expansion_distances(data)
    front = np.asarray([np.linalg.solve(np.asarray((normals[i - 1], normals[i])),
                                       np.asarray((supports[i - 1], supports[i])))
                        for i in range(len(chains))])
    if (not np.all(np.isfinite(front))
            or np.max(np.abs(front)) > np.sqrt(np.finfo(float).max) / 16):
        raise UnmeshableDirectQuadLayout("expanded front exceeds the numeric coordinate range")
    vectors = np.roll(front, -1, axis=0) - front
    if np.any(np.sum(vectors * data["directions"], axis=1) <= tol):
        _fail("expansion closes a slot, edge, or hole", "topology_change")
    _simple_loop(front, tol, "front")
    source = data["source"]
    if area(source) * area(front) <= 0 or area(source) - area(front) <= tol:
        _fail("expansion collapses or reverses the boundary", "topology_change")
    a, b = source, np.roll(source, -1, axis=0)
    index = BoxIndex(np.minimum(a, b), np.maximum(a, b), tol)
    for c, d in zip(front, np.roll(front, -1, axis=0)):
        for j in index.query(np.minimum(c, d) - tol, np.maximum(c, d) + tol):
            if segment_contacts(a[j], b[j], c, d, tol):
                _fail("front touches or crosses the source boundary", "topology_change")
    outer, inner = (front, source) if area(source) < 0 else (source, front)
    if point_location(inner[0], outer, tol) != 1:
        _fail("source and front do not bound a single annular layer", "topology_change")
    data.update(front=front, outer=outer, inner=inner)


def _validate_old_patterns(mesh, data):
    patterns, xy, tol = data["pattern_records"], data["xy"], data["tol"]
    if not len(patterns):
        return
    frozen_points = xy[data["frozen"]]
    index = BoxIndex(frozen_points.min(axis=1), frozen_points.max(axis=1), tol)
    for i, pattern in enumerate(patterns):
        for j in index.query(*pattern.bounds(tol)):
            c, d = frozen_points[j]
            if covered_intervals(pattern, frozen_points[j:j + 1], tol):
                continue
            for point in pattern_contacts(pattern, c, d, tol):
                if min(np.linalg.norm(point - c), np.linalg.norm(point - d)) > tol:
                    _fail("pattern would split a frozen source segment", "fixed_boundary_conflict", pattern_index=i)
    if not len(mesh.elements):
        return
    cells = xy[mesh.elements]
    cell_index = BoxIndex(cells.min(axis=1), cells.max(axis=1), tol)
    keys = np.asarray(list(data["old_incidence"]), dtype=int)
    edge_points = xy[keys]
    edge_index = BoxIndex(edge_points.min(axis=1), edge_points.max(axis=1), tol)
    for i, pattern in enumerate(patterns):
        lo, hi = pattern.bounds(tol)
        required = occupied_intervals(pattern, cells[cell_index.query(lo, hi)], tol)
        covering = covered_intervals(pattern, edge_points[edge_index.query(lo, hi)], tol)
        if not intervals_cover(required, covering, tol):
            _fail("pattern cuts through an existing element", "fixed_boundary_conflict", pattern_index=i)


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
                  repair_max_element_size=None):
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
        from ._repair import repair_slots as stage_repairs
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
    actions = []
    if prepared_source is not None:
        actions.append(("pattern_validation", lambda: _validate_old_patterns(working, data)))
    actions.extend((("expansion", lambda: _front(data)),
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
                       expand_ratio: float = 1.0,
                       axis_tolerance: float = 1e-3,
                       repair_slots: bool = True,
                       repair_max_element_size: float | None = None) -> QuadSectionPlan:
    """Inspect repaired source and section geometry without modifying input.

    All returned coordinates and expansion distances use the caller's frame.
    Original IDs are retained. With repairs, section source chains index the
    read-only ``prepared_source`` snapshot, which includes staged fill cells.
    Outer-layer construction remains a geometric preview: the subsequent
    seam/unit search may still find no solution. ``repair_slots=False`` keeps
    the original strict single-layer behavior.
    ``repair_max_element_size`` has the same interior target-size semantics as
    :func:`extend_quad_layer`, including automatic sizing when it is None.
    At least one existing Quad4 is required. Pattern lengths and endpoint
    L-infinity distances from the source bounding box cannot exceed one million
    times its largest XY span. This preview has no element budget and makes no
    guarantee that construction fits a later call's ``max_added_elements``.
    """
    data, plan = _prepare_plan(mesh, outline, element_size, pattern_lines, expand_ratio,
                              axis_tolerance, repair_slots=repair_slots,
                              repair_max_element_size=repair_max_element_size)

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


def extend_quad_layer(mesh: Mesh2D, outline, element_size: float, *, pattern_lines=(),
                      expand_ratio: float = 1.0,
                      axis_tolerance: float = 1e-3,
                      max_added_elements: int = 1_000_000,
                      repair_slots: bool = True,
                      repair_max_element_size: float | None = None) -> QuadLayerResult:
    """Append one conforming Quad4 layer, committing only after validation.

    ``outline`` is a closed chain of near-axis logical edges, CW for an
    exterior boundary and CCW for a hole. ``pattern_lines`` contains finite
    XY segments of shape ``(n, 2, 2)``. Growth is to the left of each directed
    edge. A parallel pattern within ``element_size * expand_ratio`` determines
    that whole edge's expansion; otherwise the expansion is ``element_size``.

    Closing simple exterior slots and covered isolated stair corners are filled
    first, unless ``repair_slots=False``. A stair corner is covered when either
    incident wall's planned expansion reaches the complete opposite leg.
    Its two old legs are retained and joined by a two-segment free cap.
    Repaired chains retain cardinal logical
    directions even when individual segments are oblique. Patterns are applied
    again to that prepared outline. Both stages share one output budget and
    one atomic commit; original node and element IDs remain unchanged.

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

    Corners are reserved before straight LINE sections. Finite patterns may
    follow boundaries or form complete source-to-front section interfaces;
    interior endpoints and interior pattern junctions are unsupported.
    Source nodes, segments, and elements are immutable. Size is a target.
    Unsupported sections or incompatible seam states raise :class:`QuadLayerError`;
    failures leave ``mesh`` untouched. There is no thin-collar retry.
    No triangular mesh is constructed, including inside temporary builders.
    """
    start = perf_counter()
    if (isinstance(max_added_elements, (bool, np.bool_))
            or not isinstance(max_added_elements, (int, np.integer))
            or max_added_elements <= 0):
        _fail("max_added_elements must be a positive integer", "invalid_input",
              stage="input_validation", parameter="max_added_elements")
    max_added_elements = int(max_added_elements)
    data, plan = _prepare_plan(mesh, outline, element_size, pattern_lines, expand_ratio,
                              axis_tolerance, repair_slots=repair_slots,
                              max_added_elements=max_added_elements,
                              repair_max_element_size=repair_max_element_size)
    working = data["_source_mesh"]
    repair_count = len(working.elements) - len(mesh.elements)
    data["max_added_elements"] = max_added_elements - repair_count
    local_nodes = working.nodes.copy()
    local_nodes[:, :2] = data["xy"]
    try:
        nodes, quads, new_outline, registry_diagnostics = fill_sections(
            local_nodes, plan, data, element_offset=len(working.elements))
    except QuadLayerError as error:
        if error.code == "element_limit_exceeded":
            error.context["limit"] = max_added_elements
            error.context["required_elements"] += repair_count
        error.context.setdefault("stage", "section_assembly")
        raise
    try:
        _validate_patch(working, nodes, quads, new_outline, data)
        all_quads = np.vstack((working.elements[len(mesh.elements):], quads)) if repair_count else quads
        if repair_count:
            original_data = dict(data["_original_data"])
            original_data.update(front=data["front"], outer=data["front"],
                                 inner=original_data["source"], tol=data["tol"])
            _validate_patch(mesh, nodes, all_quads, new_outline, original_data)
    except QuadLayerError as error:
        error.context.setdefault("stage", "final_validation")
        raise
    world_nodes = _world_nodes(nodes, working, data)
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
                   "edge_expansion": [d * data["scale"] for d in data["expansion"]],
                   "snapped_patterns": data["snapped"], "elapsed_seconds": perf_counter() - start}
    # This is the only mutation of the caller's mesh.
    mesh.replace_data(nodes=world_nodes, elements=elements)
    return QuadLayerResult(mesh=mesh, outline=new_outline, diagnostics=diagnostics)
