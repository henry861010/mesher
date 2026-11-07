"""Nearest-first, backtracking corner contacts on an immutable target loop."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .._geometry import on_segment
from .._types import QuadLayerError
from .mapping import same_point, target_node_id, valid_interface


def _nearest_first(points, source, tol):
    """Resolve numerically equal distances in canonical boundary order."""
    distances = [float(np.linalg.norm(point - source)) for point in points]
    order = sorted(range(len(points)), key=distances.__getitem__)
    result, start = [], 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and distances[order[stop]] <= distances[order[start]] + tol:
            stop += 1
        result.extend(points[i] for i in sorted(order[start:stop]))
        start = stop
    return result


def fixed_corner_layouts(data, chords, registry, corners):
    """Yield geometrically admissible assignments without changing the inputs.

    Only free OUTER contacts move. INNER reservations, repaired shared seams,
    and finite patterns remain mandatory. Candidate lists have no angle cap.
    Iterative DFS avoids a recursion limit on outlines with many corners.
    """
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
            candidates = _nearest_first(candidates, source, tol)
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

    def ordered_contacts():
        for edge, entries in enumerate(contacts):
            resolved = [(index, known[seam] if seam is not None else point)
                        for index, seam, point in entries if seam is None or seam in known]
            for (a, first), (b, second) in zip(resolved, resolved[1:]):
                if a == b:
                    if not same_point(first, second, tol):
                        return False
                elif float((second - first) @ data["directions"][edge]) <= tol:
                    return False
        return True

    choices = [0] * len(variables)
    selected = {}
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
                diagnostics.append({"corner": index, "side": side, "seam_id": seam_id,
                                    "source_node": int(corners[index].source_chains[0][0]),
                                    "target_node": target_node_id(point, data),
                                    "target": (point * data["scale"] + data["origin"]).tolist(),
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
        selected.pop(seam_id, None)
        accepted = False
        while choices[depth] < len(candidates):
            point = candidates[choices[depth]]
            choices[depth] += 1
            known[seam_id] = point
            if (ordered_contacts()
                    and valid_interface(source, point, data, [*immutable, *selected.values()])):
                selected[seam_id] = (source, point)
                accepted = True
                break
        if accepted:
            depth += 1
            if depth < len(variables):
                choices[depth] = 0
        else:
            known.pop(seam_id, None)
            depth -= 1
