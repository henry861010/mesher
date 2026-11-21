"""Shared seam states, concrete section candidates, and cyclic assembly."""
from __future__ import annotations

from collections import Counter

import numpy as np

from .corners import plan_corner
from .line import plan_line
from .sampling import validate_spacing
from .strip import alignment_metrics
from .._planning.mapping import dense_angle
from .._limits import ElementLimitExceeded, check_element_limit, preflight_element_limit
from .._registry import _NodePool, _PointIndex
from .._types import QuadLayerError
from .._line_policy import DEFAULT_MAX_SIZE_RATIO, DEFAULT_MAX_TRANSITION_COUNT


def _xyz(points):
    return np.column_stack((np.asarray(points), np.zeros(len(points))))


def seam_states(seam, tol):
    """Canonical mandatory chain, optionally with one longest-interval midpoint."""
    points = _xyz(seam.points)
    lengths = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    if not len(lengths) or np.any(lengths <= tol):
        raise QuadLayerError("a section seam has no usable length",
                             code="unmeshable_section_layout", seam_id=seam.id)
    longest = int(np.argmax(lengths))
    midpoint = np.r_[(points[longest, :2] + points[longest + 1, :2]) / 2, 0.0]
    return {len(points): points,
            len(points) + 1: np.insert(points, longest + 1, midpoint, axis=0)}


def solve_section_ring(sections, states, candidates):
    """Minimize the worst LINE cut angle of a closed ring, then additive errors."""
    for section, following in zip(sections, sections[1:] + sections[:1]):
        if section.right_seam != following.left_seam:
            raise QuadLayerError("sections do not form a shared-seam ring",
                                 code="unmeshable_section_layout", section_id=section.id)
    first_states = sorted(states[sections[0].left_seam])
    # A later, larger angle can mask an earlier peak. Determine the feasible
    # bottleneck first, then optimize the additive costs under that limit.
    peak_limit = float("inf")
    for initial in first_states:
        peaks = {initial: 0.0}
        for options in candidates:
            following = {}
            for (left, right), patch in options.items():
                if left in peaks:
                    angle = getattr(patch, "diagnostics", {}).get("max_alignment_angle", 0.0)
                    peak = max(peaks[left], angle)
                    following[right] = min(following.get(right, float("inf")), peak)
            peaks = following
        peak_limit = min(peak_limit, peaks.get(initial, float("inf")))
    if not np.isfinite(peak_limit):
        raise QuadLayerError("no compatible seam states close the section ring",
                             code="unmeshable_seam_assignment", stage="seam_solver",
                             section_ids=[section.id for section in sections])

    solutions = []
    for initial in first_states:
        current = {initial: (0.0, 0.0, 0.0)}
        parents = []
        for options in candidates:
            following, previous = {}, {}
            for (left, right), patch in sorted(options.items()):
                if (left not in current or
                        getattr(patch, "diagnostics", {}).get("max_alignment_angle", 0.0) > peak_limit):
                    continue
                alignment, error, quality = current[left]
                score = (alignment + getattr(patch, "alignment_error", 0.0),
                         error + patch.size_error, max(quality, patch.quality))
                if right not in following or score < following[right]:
                    following[right], previous[right] = score, left
            current = following
            parents.append(previous)
        if initial in current:
            path = []
            right = initial
            for previous in reversed(parents):
                left = previous[right]
                path.append((left, right))
                right = left
            solutions.append((current[initial], tuple(reversed(path))))
    return min(solutions)[1]


def _source_ids(section):
    return np.asarray([*section.source_chains[0],
                       *(node for chain in section.source_chains[1:] for node in chain[1:])],
                      dtype=np.int64)


def _quality_summary(nodes, quads):
    cells = nodes[quads, :2]
    vectors = np.roll(cells, -1, axis=1) - cells
    vectors /= np.max(np.abs(vectors), axis=(1, 2))[:, None, None]
    lengths = np.linalg.norm(vectors, axis=2)
    next_vectors = np.roll(vectors, -1, axis=1)
    scaled = (vectors[..., 0] * next_vectors[..., 1] - vectors[..., 1] * next_vectors[..., 0])
    scaled /= lengths * np.roll(lengths, -1, axis=1)
    aspect = lengths.max(axis=1) / lengths.min(axis=1)
    return {"max_aspect_ratio": float(aspect.max()),
            "min_scaled_jacobian": float(scaled.min()),
            "worst_score": float(max(aspect.max(), 1.0 / scaled.min()))}


def fill_sections(nodes, plan, data, *, element_offset=0):
    """Stage a layer; no caller-owned mesh data is changed during candidate search."""
    tol, h = data["tol"], data["h"]
    limit = data.get("max_added_elements")
    fixed_outer_boundary = data.get("fixed_outer_boundary", False)
    preflight_element_limit(plan, data)
    states = {seam.id: seam_states(seam, tol) for seam in plan.seams}
    candidates = []
    line_stats = {}
    limit_errors = []
    for section in plan.sections:
        source_chains = tuple(nodes[np.asarray(chain, dtype=int)] for chain in section.source_chains)
        outer = _xyz(section.outer_boundary)
        options = {}
        rejected = []
        section_limit_errors = []
        constraint_errors = []
        sampling_cache = {}
        if section.kind == "LINE":
            line_stats[section.id] = {"line_planner": "projection", "unit_trial_count": 0,
                                      "local_repair_count": 0, "outer_count_attempts": 0,
                                      "direct_quad_count": 0}
        for left_count, left in states[section.left_seam].items():
            for right_count, right in states[section.right_seam].items():
                attempt_stats = {}
                try:
                    if section.kind == "LINE":
                        patches = plan_line(source_chains[0], outer, left, right, h, tol,
                                            best_only=True, _diagnostics=attempt_stats,
                                            max_added_elements=limit,
                                            fixed_outer_boundary=fixed_outer_boundary,
                                            max_transition_count=data.get("max_transition_count", DEFAULT_MAX_TRANSITION_COUNT),
                                            min_size_ratio=data.get("min_size_ratio", .5),
                                            max_size_ratio=data.get("max_size_ratio", DEFAULT_MAX_SIZE_RATIO),
                                            _sampling_cache=sampling_cache)
                    else:
                        corner = np.r_[plan.front[section.corner], 0.0]
                        patches = plan_corner(section.kind, source_chains, outer, left, right,
                                              corner, h, tol, max_added_elements=limit,
                                              fixed_outer_boundary=fixed_outer_boundary)
                    if patches:
                        for candidate in patches:
                            check_element_limit(len(candidate.quads), limit,
                                                stage="section_builder", section_id=section.id)
                        options[left_count, right_count] = min(
                            patches, key=lambda patch: (patch.diagnostics.get("max_alignment_angle", 0.),
                                                        patch.alignment_error, patch.size_error, patch.quality))
                    else:
                        counts = "fixed outer count" if fixed_outer_boundary else "bounded projection candidates"
                        rejected.append(f"{left_count}/{right_count}: {counts} "
                                        "have no valid unit assembly")
                except ElementLimitExceeded as error:
                    error.context.setdefault("section_id", section.id)
                    section_limit_errors.append(error)
                except QuadLayerError as error:
                    if error.code != "line_constraints_infeasible":
                        raise
                    constraint_errors.append(error)
                    rejected.append(f"{left_count}/{right_count}: {error}")
                except ValueError as error:
                    rejected.append(f"{left_count}/{right_count}: {error}")
                finally:
                    if section.kind == "LINE":
                        for key in ("unit_trial_count", "local_repair_count", "outer_count_attempts", "direct_quad_count"):
                            line_stats[section.id][key] += attempt_stats.get(key, 0)
        if not options:
            if section_limit_errors:
                raise section_limit_errors[0]
            if len(constraint_errors) == len(states[section.left_seam]) * len(states[section.right_seam]):
                error = constraint_errors[0]
                error.context.update(section_id=section.id, causes=sorted(set(rejected)))
                raise error
            raise QuadLayerError("no seam pair can build this section with direct units",
                                 code="unmeshable_section_layout", stage="section_builder",
                                 section_id=section.id, kind=section.kind,
                                 source_edge=section.source_edge, corner=section.corner,
                                 pattern_ids=section.pattern_ids, causes=sorted(set(rejected)),
                                 **line_stats.get(section.id, {}))
        candidates.append(options)
        limit_errors.extend(section_limit_errors)
    try:
        selected = solve_section_ring(plan.sections, states, candidates)
    except QuadLayerError as error:
        if error.code == "unmeshable_seam_assignment" and limit_errors:
            raise limit_errors[0]
        raise
    added_elements = sum(len(options[pair].quads) for options, pair in zip(candidates, selected))
    check_element_limit(added_elements, limit, stage="selected_layer")
    source_ids = np.unique(data["frozen"])
    boundary_ids = source_ids
    if "outline_anchor_ids" in data:
        boundary_ids = np.union1d(source_ids, data["outline_anchor_ids"])
    pool = _NodePool(nodes, boundary_ids, tol)

    seam_ids, seam_details = {}, []
    for section, (left, right) in zip(plan.sections, selected):
        for seam_id, count in ((section.left_seam, left), (section.right_seam, right)):
            if seam_id in seam_ids:
                if len(seam_ids[seam_id]) != count:
                    raise AssertionError("cyclic solver selected inconsistent shared seams")
                continue
            points = states[seam_id][count]
            ids = np.asarray([pool.boundary_node(point[:2]) for point in points], dtype=np.int64)
            seam_ids[seam_id] = ids
            seam_details.append({"seam_id": seam_id, "node_count": count,
                                 "node_ids": ids.tolist(),
                                 "owners": list(plan.seams[seam_id].owners),
                                 "pattern_ids": list(plan.seams[seam_id].pattern_ids)})

    patches, details, units, front_ids = [], [], [], []
    offset = int(element_offset)
    for section, pair, options in zip(plan.sections, selected, candidates):
        patch = options[pair]
        boundary = set(np.concatenate((patch.source_ids, patch.outer_ids,
                                       patch.left_ids, patch.right_ids)).tolist())
        mapping = np.empty(len(patch.nodes), dtype=np.int64)
        for local, point in enumerate(patch.nodes):
            if local in boundary:
                mapping[local] = pool.boundary_node(point[:2])
            else:
                point = point.copy()
                point[2] = 0.0
                mapping[local] = pool.interior_nodes(point[None])[0]
        if (not np.array_equal(mapping[patch.left_ids], seam_ids[section.left_seam])
                or not np.array_equal(mapping[patch.right_ids], seam_ids[section.right_seam])
                or not np.array_equal(mapping[patch.source_ids], _source_ids(section))):
            raise QuadLayerError("a section changed a canonical boundary chain",
                                 code="invalid_section_boundary", section_id=section.id)
        quads = mapping[patch.quads]
        patches.append(quads)
        front_ids.extend(mapping[patch.outer_ids].tolist())
        sampling_details = {}
        if section.kind == "LINE":
            sampling_details = {key: value for key, value in patch.diagnostics.items()
                                if key not in line_stats[section.id]}
        if "line_sampling" in sampling_details:
            forced_ids = []
            forced = patch.diagnostics["forced_short_intervals"]
            if forced:
                endpoint_ids = {tuple(patch.nodes[i, :2]): int(mapping[i]) for i in patch.outer_ids}
                if len(endpoint_ids) != len(patch.outer_ids):
                    raise QuadLayerError("a mandatory LINE interval lost its endpoint",
                                         code="invalid_section_boundary", section_id=section.id)
                for interval in forced:
                    try:
                        forced_ids.append([endpoint_ids[tuple(interval[name])] for name in ("start", "end")])
                    except KeyError as error:
                        raise QuadLayerError("a mandatory LINE interval lost its endpoint",
                                             code="invalid_section_boundary", section_id=section.id) from error
            sampling_details["_forced_interval_ids"] = forced_ids
        details.append({"section_id": section.id, "kind": section.kind,
                        "source_edge": section.source_edge, "corner": section.corner,
                        "pattern_ids": list(section.pattern_ids),
                        "source_chains": [list(chain) for chain in section.source_chains],
                        "left_seam": section.left_seam, "right_seam": section.right_seam,
                        "left_num": pair[0], "right_num": pair[1],
                        "feasible_seam_pairs": [list(pair) for pair in sorted(options)],
                        "left_ids": mapping[patch.left_ids].tolist(),
                        "right_ids": mapping[patch.right_ids].tolist(),
                        "outer_ids": mapping[patch.outer_ids].tolist(),
                        "unit_ids": list(range(len(units), len(units) + len(patch.units))),
                        "element_start": offset, "element_end": offset + len(quads),
                        "_polygon_ids": mapping[np.concatenate((patch.source_ids, patch.right_ids[1:],
                                                patch.outer_ids[::-1][1:], patch.left_ids[::-1][1:]))].tolist(),
                        "size_error": patch.size_error,
                        "quality": patch.quality, **line_stats.get(section.id, {}),
                        **sampling_details})
        for record in patch.units:
            units.append({**record, "unit_id": len(units), "section_id": section.id,
                          "element_start": offset + record["element_start"],
                          "element_end": offset + record["element_end"],
                          "_polygon_ids": mapping[record["_polygon_ids"]].tolist()})
        offset += len(quads)
    index = _PointIndex(pool, front_ids)
    outline = [index.on_segment(a, b) for a, b in zip(plan.front, np.roll(plan.front, -1, axis=0))]
    result_nodes = pool.array()
    result_quads = np.vstack(patches)
    diagnostics = {"path": "sections", "sections": dict(Counter(s.kind for s in plan.sections)),
                   "section_count": len(plan.sections), "block_count": len(plan.sections),
                   "parity_solver": "section_cyclic_dp", "section_details": details,
                   "seam_states": sorted(seam_details, key=lambda s: s["seam_id"]),
                   "units": units, "unit_count": len(units),
                   "unit_builders": dict(Counter(unit["method"] for unit in units)),
                   "line_planner": "projection",
                   "line_unit_trial_count": sum(s["unit_trial_count"] for s in line_stats.values()),
                   "line_direct_quad_count": sum(s["direct_quad_count"] for s in line_stats.values()),
                   "line_local_repair_count": sum(s["local_repair_count"] for s in line_stats.values())}
    return result_nodes, result_quads, outline, diagnostics


def finalize_diagnostics(world_nodes, nodes, quads, diagnostics, plan, data, *, element_offset):
    """Validate final LINE spacing and report geometry by its committed IDs."""
    for seam in diagnostics["seam_states"]:
        seam["points"] = world_nodes[seam["node_ids"], :2].tolist()
    cells = nodes[quads, :2]
    vectors = np.roll(cells, -1, axis=1) - cells
    vectors /= np.max(np.abs(vectors), axis=(1, 2))[:, None, None]
    lengths = np.linalg.norm(vectors, axis=2)
    following = np.roll(vectors, -1, axis=1)
    scaled = (vectors[..., 0] * following[..., 1] - vectors[..., 1] * following[..., 0])
    scaled /= lengths * np.roll(lengths, -1, axis=1)
    scores = np.maximum(lengths.max(axis=1) / lengths.min(axis=1), 1 / scaled.min(axis=1))
    for record in [*diagnostics["section_details"], *diagnostics["units"]]:
        record["polygon"] = world_nodes[record.pop("_polygon_ids"), :2].tolist()
        record["quality"] = float(scores[record["element_start"] - element_offset:
                                         record["element_end"] - element_offset].max())
        if "internal_cuts" in record:
            source = nodes[record["source_chains"][0]]
            outer = nodes[record["outer_ids"]]
            mean, peak = alignment_metrics(source, outer, record["internal_cuts"], data["tol"])
            record.update(alignment_error=mean, max_connector_shear=peak,
                          max_alignment_angle=float(np.degrees(np.arctan(peak))))
        if "line_sampling" in record:
            points = world_nodes[record["outer_ids"], :2]
            record["forced_short_intervals"] = [
                {"start": world_nodes[a, :2].tolist(), "end": world_nodes[b, :2].tolist(),
                 "length": float(np.hypot.reduce(world_nodes[b, :2] - world_nodes[a, :2]))}
                for a, b in record.pop("_forced_interval_ids")]
            gaps = validate_spacing(points, data["h"] * data["scale"], data["min_size_ratio"],
                                    record["forced_short_intervals"], data["max_size_ratio"])
            record.update(min_outer_spacing=float(gaps.min()), max_outer_spacing=float(gaps.max()),
                          size_error=float(np.mean((gaps / (data["h"] * data["scale"]) - 1) ** 2)))
    details = diagnostics["section_details"]
    diagnostics["outer_size_error"] = sum(d["size_error"] for d in details)
    diagnostics["line_alignment_error"] = sum(d.get("alignment_error", 0.) for d in details)
    diagnostics["max_alignment_angle"] = max(d.get("max_alignment_angle", 0.) for d in details)
    seams = {record["seam_id"]: record for record in diagnostics["seam_states"]}
    mapped_dense = {}
    for record in plan.diagnostics.get("dense_seams", ()):
        ids = seams[record["seam_id"]]["node_ids"]
        first, last = ids[0], ids[-1]
        edge = record["source_edge"]
        valid, angle = dense_angle(nodes[first, :2], nodes[last, :2], data["directions"][edge],
                                   data["normals"][edge], record["endpoint"], data["dense_mapping_angle"])
        if not valid:
            raise QuadLayerError("represented dense seam exceeds its angular interval",
                                 code="unmeshable_section_layout", seam_id=record["seam_id"])
        record.update(target=world_nodes[last, :2].tolist(), angle=angle)
        mapped_dense[record["seam_id"]] = record
    for record in plan.diagnostics.get("dense_mappings", ()):
        if record.get("status") == "mapped":
            matched = mapped_dense[record["seam_id"]]
            record.update(target=matched["target"], angle=matched["angle"])
