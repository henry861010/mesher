"""Shared seam states, concrete section candidates, and cyclic assembly."""
from __future__ import annotations

from collections import Counter

import numpy as np

from .corners import plan_corner
from .line import plan_line
from .._limits import ElementLimitExceeded, check_element_limit, preflight_element_limit
from .._registry import _NodePool, _PointIndex
from .._types import QuadLayerError


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
    """Minimize size error, then worst quality, with deterministic closure."""
    for section, following in zip(sections, sections[1:] + sections[:1]):
        if section.right_seam != following.left_seam:
            raise QuadLayerError("sections do not form a shared-seam ring",
                                 code="unmeshable_section_layout", section_id=section.id)
    solutions = []
    for initial in sorted(states[sections[0].left_seam]):
        current = {initial: (0.0, 0.0, ())}
        for section, options in zip(sections, candidates):
            following = {}
            for (left, right), candidate in sorted(options.items()):
                if left not in current:
                    continue
                error, quality, path = current[left]
                score = (error + candidate.size_error, max(quality, candidate.quality),
                         path + ((left, right),))
                if right not in following or score < following[right]:
                    following[right] = score
            current = following
        if initial in current:
            solutions.append(current[initial])
    if not solutions:
        raise QuadLayerError("no compatible seam states close the section ring",
                             code="unmeshable_seam_assignment", stage="seam_solver",
                             section_ids=[section.id for section in sections])
    return min(solutions)[2]


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
        if section.kind == "LINE":
            line_stats[section.id] = {"line_planner": "outer_first", "unit_trial_count": 0,
                                      "local_repair_count": 0, "outer_count_attempts": 0}
        for left_count, left in states[section.left_seam].items():
            for right_count, right in states[section.right_seam].items():
                attempt_stats = {}
                try:
                    if section.kind == "LINE":
                        patches = plan_line(source_chains[0], outer, left, right, h, tol,
                                            best_only=True, _diagnostics=attempt_stats,
                                            max_added_elements=limit,
                                            fixed_outer_boundary=fixed_outer_boundary)
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
                            patches, key=lambda patch: (patch.size_error, patch.quality))
                    else:
                        counts = "fixed outer count" if fixed_outer_boundary else "nearest legal outer counts"
                        rejected.append(f"{left_count}/{right_count}: {counts} "
                                        "have no valid unit assembly")
                except ElementLimitExceeded as error:
                    error.context.setdefault("section_id", section.id)
                    section_limit_errors.append(error)
                except ValueError as error:
                    rejected.append(f"{left_count}/{right_count}: {error}")
                finally:
                    if section.kind == "LINE":
                        for key in ("unit_trial_count", "local_repair_count", "outer_count_attempts"):
                            line_stats[section.id][key] += attempt_stats.get(key, 0)
        if not options:
            if section_limit_errors:
                raise section_limit_errors[0]
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
    pool = _NodePool(nodes, source_ids, tol)

    def world(points):
        return (np.asarray(points)[:, :2] * data["scale"] + data["origin"]).tolist()

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
                                 "node_ids": ids.tolist(), "points": world(points),
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
        polygon = np.vstack((patch.nodes[patch.source_ids],
                             patch.nodes[patch.right_ids[1:]],
                             patch.nodes[patch.outer_ids[::-1][1:]],
                             patch.nodes[patch.left_ids[::-1][1:]]))
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
                        "polygon": world(polygon), "size_error": patch.size_error,
                        "quality": patch.quality, **line_stats.get(section.id, {})})
        for record in patch.units:
            units.append({**record, "unit_id": len(units), "section_id": section.id,
                          "element_start": offset + record["element_start"],
                          "element_end": offset + record["element_end"],
                          "polygon": world(record["polygon"])})
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
                   "line_planner": "outer_first",
                   "line_unit_trial_count": sum(s["unit_trial_count"] for s in line_stats.values()),
                   "line_local_repair_count": sum(s["local_repair_count"] for s in line_stats.values()),
                   "outer_size_error": sum(detail["size_error"] for detail in details),
                   "quality": _quality_summary(result_nodes, result_quads)}
    return result_nodes, result_quads, outline, diagnostics
