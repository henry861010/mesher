"""Nearest-first, backtracking corner contacts on an immutable target loop."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import numpy as np

from .._geometry import on_segment
from .._types import QuadLayerError
from .mapping import (compatible_interfaces, corner_distances, same_point,
                      target_node_id, valid_interface)


def _nearest_first(points, source, data):
    """Resolve numerically equal distances in canonical boundary order."""
    if len(points) < 2:
        return points
    distances, tie_tol = corner_distances(points, source, data)
    order = sorted(range(len(points)), key=distances.__getitem__)
    result, start = [], 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and distances[order[stop]] <= distances[order[start]] + tie_tol:
            stop += 1
        result.extend(points[i] for i in sorted(order[start:stop]))
        start = stop
    return result


def fixed_corner_layouts(data, chords, registry, corners):
    """Yield admissible assignments without changing boundary geometry.

    Only free OUTER contacts move. INNER reservations, repaired shared seams,
    and finite patterns remain mandatory. Candidate lists have no angle cap.
    Iterative DFS avoids a recursion limit on outlines with many corners.
    """
    stats = data["_corner_search_stats"] = {"candidate_visits": 0, "ordering_prunes": 0}
    tol, front = data["tol"], data["front"]
    anchors = data["outline_anchors"]
    owners = {}
    for index, corner in enumerate(corners):
        for seam in (corner.left, corner.right):
            owners.setdefault(seam, []).append(index)
    variables = []
    for index, corner in enumerate(corners):
        if corner.kind != "OUTER":
            continue
        for side, seam_id, edge in (("incoming", corner.left, (index - 1) % len(corners)),
                                    ("outgoing", corner.right, index)):
            seam = registry.items[seam_id]
            if len(owners[seam_id]) != 1 or seam.pattern_ids:
                continue
            source = seam.points[0]
            a, b = front[edge], front[(edge + 1) % len(front)]
            forced = [chord.points[-1] for chord in chords
                      if chord.edge == edge and same_point(chord.points[0], source, tol)]
            candidates = [point.copy() for point in anchors
                          if on_segment(point, a, b, tol)
                          and not same_point(point, front[index], tol)
                          and all(same_point(point, p, tol) for p in forced)]
            candidates = _nearest_first(candidates, source, data)
            variables.append((seam_id, index, side, source, candidates))
    variable_ids = {item[0] for item in variables}
    known = {seam.id: seam.points[-1] for seam in registry.items if seam.id not in variable_ids}
    immutable = [seam.points[[0, -1]] for seam in registry.items if seam.id not in variable_ids]
    immutable.extend(chord.points[[0, -1]] for chord in chords)
    contacts = []
    for edge, corner in enumerate(corners):
        following = corners[(edge + 1) % len(corners)]
        if corner.outgoing_index > following.incoming_index:
            raise QuadLayerError("neighboring corner reservations overlap",
                                 code="unmeshable_section_layout", stage="section_partition",
                                 source_edge=edge)
        contacts.append(sorted(
            [(corner.outgoing_index, corner.right, None),
             (following.incoming_index, following.left, None)]
            + [(chord.index, None, chord.points[-1]) for chord in chords if chord.edge == edge],
            key=lambda item: item[0]))

    positions = {item[0]: i for i, item in enumerate(variables)}
    constraints = []
    for edge, entries in enumerate(contacts):
        for first, second in zip(entries, entries[1:]):
            constraints.append((edge, first, second))

    def values(entry, domains):
        _, seam, point = entry
        if seam in positions:
            position = positions[seam]
            return np.asarray([variables[position][4][i] for i in domains[position]])
        return np.asarray([known[seam] if seam is not None else point])

    def propagate(domains):
        # Only ordered contacts are propagated eagerly. This cheap scalar
        # feasibility pass catches an impossible late edge before any DFS.
        changed = True
        while changed:
            changed = False
            if any(not domain for domain in domains):
                return None
            for edge, first, second in constraints:
                a, b = values(first, domains), values(second, domains)
                if first[0] == second[0]:
                    keep_a = np.array([np.any(np.linalg.norm(b - p, axis=1) <= tol) for p in a])
                    keep_b = np.array([np.any(np.linalg.norm(a - p, axis=1) <= tol) for p in b])
                else:
                    tangent = data["directions"][edge]
                    x, y = a @ tangent, b @ tangent
                    keep_a, keep_b = y.max() - x > tol, y - x.min() > tol
                if not np.any(keep_a) or not np.any(keep_b):
                    return None
                for entry, mask in ((first, keep_a), (second, keep_b)):
                    position = positions.get(entry[1])
                    if position is not None and not np.all(mask):
                        domains[position] = [i for i, keep in zip(domains[position], mask) if keep]
                        changed = True
        return domains

    domains = propagate([list(range(len(item[4]))) for item in variables])
    if domains is None:
        stats["ordering_prunes"] += 1
        return

    @lru_cache(maxsize=4096)
    def unary(position, candidate):
        _, _, _, source, points = variables[position]
        return valid_interface(source, points[candidate], data, immutable)

    @lru_cache(maxsize=8192)
    def compatible(first, first_candidate, second, second_candidate):
        a, b = variables[first], variables[second]
        return compatible_interfaces((a[3], a[4][first_candidate]),
                                     (b[3], b[4][second_candidate]), tol)

    # Keep the original variable and nearest-first candidate order. Domain
    # pruning changes only branches proven impossible, not preferred solutions.
    domain_stack = [domains]
    assigned = [-1] * len(variables)
    choices = [0] * len(variables)
    depth = 0
    while depth >= 0:
        if depth == len(variables):
            diagnostics = []
            items = list(registry.items)
            for position, (seam_id, index, side, source, _) in enumerate(variables):
                point = known[seam_id]
                original = items[seam_id]
                # Free OUTER seams have no mandatory interior pattern points.
                points = np.asarray((source, point))
                points.setflags(write=False)
                items[seam_id] = replace(original, points=points)
                target_id = target_node_id(point, data)
                world_target = (data["_source_mesh"].nodes[target_id, :2]
                                if target_id is not None and "_source_mesh" in data
                                else point * data["scale"] + data["origin"])
                diagnostics.append({"corner": index, "side": side, "seam_id": seam_id,
                                    "source_node": int(corners[index].source_chains[0][0]),
                                    "target_node": target_id, "target": world_target.tolist(),
                                    "candidate_rank": choices[position],
                                    "substituted": not same_point(point, original.points[-1], tol)})
            mapped = [replace(corner, outer=np.asarray((items[corner.left].points[-1], front[index],
                                                        items[corner.right].points[-1])))
                      if corner.kind == "OUTER" else replace(corner)
                      for index, corner in enumerate(corners)]
            yield items, mapped, diagnostics
            depth -= 1
            continue
        seam_id, _, _, source, candidates = variables[depth]
        known.pop(seam_id, None)
        accepted = False
        while choices[depth] < len(candidates):
            point = candidates[choices[depth]]
            choices[depth] += 1
            known[seam_id] = point
            candidate = choices[depth] - 1
            if candidate not in domain_stack[depth][depth]:
                continue
            stats["candidate_visits"] += 1
            following = [domain.copy() for domain in domain_stack[depth]]
            following[depth] = [candidate]
            following = propagate(following)
            if following is None:
                stats["ordering_prunes"] += 1
            if (following is not None and unary(depth, candidate)
                    and all(compatible(previous, assigned[previous], depth, candidate)
                            for previous in range(depth))):
                assigned[depth] = candidate
                domain_stack[depth + 1:] = [following]
                accepted = True
                break
        if accepted:
            depth += 1
            if depth < len(variables):
                choices[depth] = 0
        else:
            known.pop(seam_id, None)
            depth -= 1
