"""Transactional filling of simple closing slots before semantic-axis growth.

Only the selected exterior outline is changed.  Every original mesh node and
element retains its ID, and the rebuilt outline records the cardinal meaning
of its edges even when a newly sampled mouth is geometrically slanted.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from ...model import Mesh2D
from ._geometry import BoxIndex, area, cross, point_location, segment_contacts
from ._limits import ElementLimitExceeded, check_element_limit
from ._types import QuadLayerError


@dataclass(frozen=True)
class RepairResult:
    mesh: Mesh2D
    outline: list[list[int]]
    directions: np.ndarray
    provenance: tuple[tuple[int, ...], ...]
    diagnostics: dict


@dataclass(frozen=True)
class _Slot:
    edges: tuple[int, int, int]
    tangent: np.ndarray
    shorter: int
    station: float
    candidates: tuple[int, ...]


@dataclass(frozen=True)
class _Candidate:
    slot: _Slot
    source_chains: tuple[np.ndarray, ...]
    source_ids: np.ndarray
    polygon: np.ndarray


def _fail(message, *, reason=None, code="unsupported_slot_repair", **context):
    context.setdefault("stage", "slot_repair")
    if reason is not None:
        context["reason"] = reason
    raise QuadLayerError(message, code=code, **context)


def _flat_chains(chains):
    return np.concatenate([chains[0], *(chain[1:] for chain in chains[1:])])


def _key(a, b):
    return tuple(sorted((int(a), int(b))))


def _incidence(elements):
    result = defaultdict(list)
    for index, cell in enumerate(elements):
        for a, b in zip(cell, np.roll(cell, -1)):
            result[_key(a, b)].append((index, int(a), int(b)))
    return result


def _simple_polygon(polygon, tol, *, reason):
    following = np.roll(polygon, -1, axis=0)
    lengths = np.linalg.norm(following - polygon, axis=1)
    if len(polygon) < 3 or np.any(lengths <= tol):
        _fail("a slot repair boundary has a zero-length edge", reason=reason)
    lower, upper = np.minimum(polygon, following), np.maximum(polygon, following)
    index = BoxIndex(lower, upper, tol)
    for i, (a, b) in enumerate(zip(polygon, following)):
        for j in index.query(lower[i] - tol, upper[i] + tol):
            if j <= i or j == i + 1 or (i == 0 and j == len(polygon) - 1):
                continue
            if segment_contacts(a, b, polygon[j], following[j], tol):
                _fail("a slot repair boundary intersects or touches itself", reason=reason)


def _widths(first, last, tangent, across, tol):
    """Widths at all piecewise-linear events in the common wall interval."""
    first = first[::-1]
    a, b = first @ tangent, last @ tangent
    low, high = max(a[0], b[0]), min(a[-1], b[-1])
    if high - low <= tol:
        return None
    events = np.unique(np.r_[low, high, a[(a > low) & (a < high)],
                             b[(b > low) & (b < high)]])
    return np.interp(events, b, last @ across) - np.interp(events, a, first @ across)


def _brackets(values, station, tol):
    order = np.argsort(values)
    ordered = values[order]
    at = int(np.searchsorted(ordered, station))
    choices = {max(0, min(len(order) - 1, at)), max(0, min(len(order) - 1, at - 1))}
    indices = sorted((int(order[i]) for i in choices),
                     key=lambda i: (abs(float(values[i] - station)),
                                    values[i] > station + tol, i))
    if len(indices) == 2:
        a, b = indices
        if abs(abs(values[a] - station) - abs(values[b] - station)) <= tol:
            indices.sort(key=lambda i: (values[i] > station + tol, i))
    exact = [i for i in indices if abs(values[i] - station) <= tol]
    return tuple(exact[:1] if exact else indices)


def _find_slots(data):
    """Recognize only complete opposing-wall closure on the original outline."""
    if area(data["source"]) >= 0:
        return []
    chains, directions, tol = data["chains"], data["directions"], data["tol"]
    expansion = data["expansion"]
    count = len(chains)
    slots = []
    for back in range(count):
        first, last = (back - 1) % count, (back + 1) % count
        if (float(cross(directions[first], directions[back])) <= 0
                or float(cross(directions[back], directions[last])) <= 0
                or not np.array_equal(directions[first], -directions[last])):
            continue
        tangent, across = directions[last], directions[back]
        widths = _widths(chains[first], chains[last], tangent, across, tol)
        if widths is None or float(np.min(widths)) <= tol:
            continue
        reach = float(expansion[first] + expansion[last])
        if reach < float(np.min(widths)) - tol:
            continue
        if reach < float(np.max(widths)) - tol:
            _fail("only part of a slot would close in this expansion",
                  reason="partial_slot_closure", stage="slot_detection", source_edges=(first, back, last))
        before, after = (first - 1) % count, (last + 1) % count
        if (float(cross(directions[before], directions[first])) >= 0
                or float(cross(directions[last], directions[after])) >= 0):
            _fail("the closing region is not a simple open C/U slot",
                  reason="complex_slot_mouth", stage="slot_detection", source_edges=(first, back, last))
        first_end = float(chains[first][0] @ tangent)
        last_end = float(chains[last][-1] @ tangent)
        shorter = 0 if first_end <= last_end + tol else 2
        station = min(first_end, last_end)
        longer = last if shorter == 0 else first
        candidates = _brackets(chains[longer] @ tangent, station, tol)
        slots.append(_Slot((first, back, last), tangent.copy(), shorter, station, candidates))
    slots.sort(key=lambda slot: (*data["chains"][slot.edges[1]].mean(axis=0), slot.edges[1]))
    return slots


def _candidate(slot, anchor, data):
    first, back, last = slot.edges
    chains = tuple(np.asarray(data["ids"][edge], dtype=np.int64).copy() for edge in slot.edges)
    if slot.shorter == 0:
        chains = (chains[0], chains[1], chains[2][:anchor + 1])
    else:
        chains = (chains[0][anchor:], chains[1], chains[2])
    source_ids = _flat_chains(chains)
    polygon = data["xy"][source_ids]
    tol = data["tol"]
    _simple_polygon(polygon, tol, reason="invalid_slot_mouth")
    perimeter = float(np.linalg.norm(np.roll(polygon, -1, axis=0) - polygon, axis=1).sum())
    if area(polygon) <= tol * perimeter:
        _fail("the slot mouth leaves no positive repair region", reason="invalid_slot_mouth")

    # An outward bracket is useful when old node spacing misses the mouth.
    # Admit its overhang only where the ORIGINAL one-step wall/lip expansion
    # already covers it.  This is a geometric bound, not a hard delta <= h.
    endpoint = polygon[-1] if slot.shorter == 0 else polygon[0]
    outside = float(endpoint @ slot.tangent - slot.station)
    if outside > tol:
        short_edge = first if slot.shorter == 0 else last
        long_edge = last if slot.shorter == 0 else first
        lip = (short_edge - 1) % len(data["ids"]) if slot.shorter == 0 else (short_edge + 1) % len(data["ids"])
        lip_reach = float(data["expansion"][lip])
        if outside > lip_reach + tol:
            short_point = polygon[0] if slot.shorter == 0 else polygon[-1]
            contact = short_point + (endpoint - short_point) * (lip_reach / outside)
            normal = data["normals"][long_edge]
            front_support = float(np.max(data["chains"][long_edge] @ normal)
                                  + data["expansion"][long_edge])
            if float(contact @ normal) > front_support + tol:
                _fail("the outward mouth bracket would fill beyond this expansion",
                      reason="mouth_outside_expansion", source_edges=slot.edges,
                      anchor_node=int(source_ids[-1] if slot.shorter == 0 else source_ids[0]))
    return _Candidate(slot, chains, source_ids, polygon)


def _validate_empty(candidate, mesh, data, edge_keys, edge_points, edge_index, cell_index):
    """A cap may meet old topology only at its two existing endpoint IDs."""
    tol = data["tol"]
    a, b = candidate.polygon[[0, -1]]
    endpoint_ids = (int(candidate.source_ids[0]), int(candidate.source_ids[-1]))
    for j in edge_index.query(np.minimum(a, b) - tol, np.maximum(a, b) + tol):
        c, d = edge_points[j]
        contacts = segment_contacts(a, b, c, d, tol)
        if len(contacts) > 1 and np.linalg.norm(contacts[-1] - contacts[0]) > tol:
            _fail("a repair mouth overlaps an existing edge", reason="mouth_hits_existing_mesh")
        for point in contacts:
            if not any(node in edge_keys[j] and np.linalg.norm(point - endpoint) <= tol
                       for node, endpoint in zip(endpoint_ids, (a, b))):
                _fail("a repair mouth crosses existing mesh or would create a hanging node",
                      reason="mouth_hits_existing_mesh")
    for j in cell_index.query(candidate.polygon.min(axis=0), candidate.polygon.max(axis=0)):
        cell = mesh.nodes[mesh.elements[j], :2]
        if (point_location(cell.mean(axis=0), candidate.polygon, tol) == 1
                or any(point_location(point, candidate.polygon, tol) == 1 for point in cell)):
            _fail("the proposed slot repair contains existing mesh",
                  reason="repair_overlaps_existing_mesh", old_cell=int(j))


def _check_slot_conflict(candidate, selected, tol):
    first = candidate.polygon
    source_keys = {_key(a, b) for a, b in zip(candidate.source_ids[:-1], candidate.source_ids[1:])}
    for other in selected:
        other_keys = {_key(a, b) for a, b in zip(other.source_ids[:-1], other.source_ids[1:])}
        if source_keys.intersection(other_keys):
            _fail("closing slots consume the same original source segment",
                  reason="interacting_slots", source_edges=(candidate.slot.edges, other.slot.edges))
        second = other.polygon
        if (np.any(first.max(axis=0) < second.min(axis=0) - tol)
                or np.any(second.max(axis=0) < first.min(axis=0) - tol)):
            continue
        for a, b in zip(first, np.roll(first, -1, axis=0)):
            for c, d in zip(second, np.roll(second, -1, axis=0)):
                if segment_contacts(a, b, c, d, tol):
                    _fail("slot repair regions intersect or touch", reason="interacting_slots",
                          source_edges=(candidate.slot.edges, other.slot.edges))
        if (point_location(first[0], second, tol) >= 0
                or point_location(second[0], first, tol) >= 0):
            _fail("slot repair regions overlap", reason="interacting_slots",
                  source_edges=(candidate.slot.edges, other.slot.edges))


def _regroup(mesh, data, nodes, quads, selected, mouths):
    """Extract the actual repaired boundary and attach semantic provenance."""
    consumed = {_key(a, b) for candidate in selected
                for a, b in zip(candidate.source_ids[:-1], candidate.source_ids[1:])}
    records = []
    for edge, chain in enumerate(data["ids"]):
        for a, b in zip(chain[:-1], chain[1:]):
            if _key(a, b) not in consumed:
                records.append((int(a), int(b), data["directions"][edge], (edge,)))
    for candidate, mouth in zip(selected, mouths):
        for a, b in zip(mouth[:-1], mouth[1:]):
            records.append((int(a), int(b), data["directions"][candidate.slot.edges[1]],
                            candidate.slot.edges))
    planned = {}
    for record in records:
        a, b = record[:2]
        if (a, b) in planned:
            _fail("slot repairs repeat a planned boundary edge", reason="repair_topology")
        planned[a, b] = record

    # Re-extract from actual element ownership.  Do not assume that the
    # builder's advertised mouth equals the boundary it actually produced.
    # Remote islands and holes remain in this inventory, but the selected
    # loop is located by a surviving ORIGINAL segment, never by area/order.
    elements = np.vstack((mesh.elements, quads))
    incidence = _incidence(elements)
    old_boundary = {key for key, owners in data["old_incidence"].items() if len(owners) == 1}
    expected_boundary = old_boundary.difference(consumed)
    expected_boundary.update(_key(a, b) for mouth in mouths for a, b in zip(mouth[:-1], mouth[1:]))
    actual_boundary = {key for key, owners in incidence.items() if len(owners) == 1}
    if actual_boundary != expected_boundary:
        _fail("the actual repaired mesh boundary differs from the planned slot mouths",
              reason="repair_topology")
    outgoing = defaultdict(list)
    for key in actual_boundary:
        owner, a, b = incidence[key][0]
        centroid = nodes[elements[owner], :2].mean(axis=0)
        if cross(nodes[b, :2] - nodes[a, :2], centroid - nodes[a, :2]) > 0:
            a, b = b, a
        outgoing[a].append(b)
    surviving = next(((int(a), int(b)) for chain in data["ids"]
                      for a, b in zip(chain[:-1], chain[1:]) if _key(a, b) not in consumed), None)
    if surviving is None:
        _fail("slot repairs leave no original segment for selecting the exterior loop",
              reason="repair_topology")
    start = surviving[0]
    ordered, node = [], start
    while True:
        if len(outgoing[node]) != 1:
            _fail("the selected repaired boundary is not manifold", reason="repair_topology")
        edge = (node, outgoing[node][0])
        if edge not in planned:
            _fail("the actual repaired loop contains an unplanned edge", reason="repair_topology", edge=edge)
        record = planned[edge]
        ordered.append(record)
        node = record[1]
        if node == start:
            break
        if len(ordered) >= len(records):
            _fail("slot repair boundary traversal does not close", reason="repair_topology")
    if len(ordered) != len(records):
        _fail("slot repairs would create more than one boundary loop", reason="repair_topology")
    polygon = nodes[np.asarray([record[0] for record in ordered]), :2]
    _simple_polygon(polygon, data["tol"], reason="repair_topology")
    if area(polygon) >= 0:
        _fail("slot repairs reversed the exterior outline", reason="repair_topology")
    groups = []
    for a, b, direction, provenance in ordered:
        if float((nodes[b, :2] - nodes[a, :2]) @ direction) <= data["tol"]:
            _fail("a repaired chain is not monotone along its semantic axis",
                  reason="nonmonotone_repaired_chain", edge=(a, b))
        if groups and np.array_equal(groups[-1][1], direction):
            groups[-1][0].append(b)
            groups[-1][2].update(provenance)
        else:
            groups.append(([a, b], direction.copy(), set(provenance)))
    if len(groups) > 1 and np.array_equal(groups[0][1], groups[-1][1]):
        last = groups.pop()
        groups[0] = (last[0][:-1] + groups[0][0], groups[0][1], last[2] | groups[0][2])
    if len(groups) < 4 or any(float(a[1] @ b[1]) != 0
                              for a, b in zip(groups, groups[1:] + groups[:1])):
        _fail("repaired semantic edges do not form an orthogonal logical outline",
              reason="invalid_semantic_outline")
    return ([group[0] for group in groups], np.asarray([group[1] for group in groups]),
            tuple(tuple(sorted(group[2])) for group in groups), consumed)


def _validate_repair(mesh, nodes, quads, outline, data, selected, mouths, consumed):
    """Certify the complete staged repair against the ORIGINAL mesh."""
    from ._validation import (_ccw_cells, _check_edge_geometry, _check_overlaps,
                              _check_patterns)

    tol = data["tol"]
    if not np.array_equal(nodes[:len(mesh.nodes)], mesh.nodes):
        _fail("slot repair changed original node coordinates", code="unmeshable_slot_repair")
    if not len(quads) or quads.shape[1:] != (4,) or np.any(np.diff(np.sort(quads, axis=1), axis=1) == 0):
        _fail("slot repair produced an invalid quadrilateral", code="unmeshable_slot_repair")
    cells = nodes[quads, :2]
    vectors = np.roll(cells, -1, axis=1) - cells
    threshold = 64 * np.finfo(float).eps * np.ptp(cells, axis=1).max(axis=1) ** 2
    if np.any(cross(vectors, np.roll(vectors, -1, axis=1)) <= threshold[:, None]):
        _fail("slot repair cells must be strictly convex and counterclockwise", code="unmeshable_slot_repair")
    incidence = _incidence(quads)
    expected = {_key(a, b): (int(a), int(b)) for candidate in selected
                for a, b in zip(candidate.source_ids[:-1], candidate.source_ids[1:])}
    for mouth in mouths:
        for a, b in zip(mouth[:-1], mouth[1:]):
            expected[_key(a, b)] = (int(b), int(a))
    if not set(expected).issubset(incidence):
        _fail("slot repair omitted a fixed boundary segment", code="unmeshable_slot_repair")
    for key, owners in incidence.items():
        if key in expected:
            valid = len(owners) == 1 and owners[0][1:] == expected[key]
        else:
            valid = len(owners) == 2 and owners[0][1:] == owners[1][1:][::-1]
        if not valid:
            _fail("slot repair has inconsistent edge ownership", code="unmeshable_slot_repair", edge=key)
    old_incidence = data["old_incidence"]
    if set(incidence).intersection(old_incidence).difference(consumed):
        _fail("slot repair meets old mesh away from its source walls", code="unmeshable_slot_repair")
    allowed_old = set(int(i) for candidate in selected for i in candidate.source_ids)
    if not set(quads[quads < len(mesh.nodes)].tolist()).issubset(allowed_old):
        _fail("slot repair used an unrelated original node", code="unmeshable_slot_repair")
    used_new = np.unique(quads[quads >= len(mesh.nodes)])
    if not np.array_equal(used_new, np.arange(len(mesh.nodes), len(nodes))):
        _fail("slot repair contains unused new nodes", code="unmeshable_slot_repair")
    new_keys = np.asarray(sorted(incidence), dtype=np.int64).reshape(-1, 2)
    old_keys = np.asarray(sorted(old_incidence), dtype=np.int64).reshape(-1, 2)
    new_edges, old_edges = nodes[new_keys, :2], mesh.nodes[old_keys, :2]
    new_edge_index = BoxIndex(new_edges.min(axis=1), new_edges.max(axis=1), tol)
    old_edge_index = BoxIndex(old_edges.min(axis=1), old_edges.max(axis=1), tol)
    _check_edge_geometry(new_keys, new_edges, old_keys, old_edges, new_edge_index, old_edge_index, tol)
    old_cells = _ccw_cells(mesh.nodes[mesh.elements, :2])
    new_cell_index = BoxIndex(cells.min(axis=1), cells.max(axis=1), tol)
    old_cell_index = BoxIndex(old_cells.min(axis=1), old_cells.max(axis=1), tol)
    _check_overlaps(cells, old_cells, new_cell_index, old_cell_index, tol)
    _check_patterns(data["pattern_records"], cells, old_cells, new_cell_index, old_cell_index,
                    new_edges, old_edges, new_edge_index, old_edge_index, tol)
    actual = sum(area(cell) for cell in cells)
    target = nodes[np.concatenate([np.asarray(chain[:-1]) for chain in outline]), :2]
    expected_area = area(data["source"]) - area(target)
    perimeter = sum(float(np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1).sum())
                    for p in (data["source"], target))
    if expected_area <= 0 or abs(actual - expected_area) > max(8 * tol * perimeter, 1e-10 * expected_area):
        _fail("slot repair does not cover the complete planned region", code="unmeshable_slot_repair",
              actual_area=actual, expected_area=expected_area)


def repair_slots(mesh, data, *, max_added_elements=None, repair_max_element_size=None):
    """Return a normalized staged source, or None when no slot needs repair.

    ``data`` is the original preparation dictionary with per-edge expansion
    distances populated.  This function never mutates ``mesh`` or ``data``.
    Slot detection runs once. Adaptive face builders do not re-enter this phase.
    """
    slots = _find_slots(data)
    if not slots:
        return None
    from ._repair_patch import build_repair_patch

    tol = data["tol"]
    edge_keys = np.asarray(sorted(data["old_incidence"]), dtype=np.int64).reshape(-1, 2)
    edge_points = mesh.nodes[edge_keys, :2]
    edge_index = BoxIndex(edge_points.min(axis=1), edge_points.max(axis=1), tol)
    old_cells = mesh.nodes[mesh.elements, :2]
    cell_index = BoxIndex(old_cells.min(axis=1), old_cells.max(axis=1), tol)
    nodes = mesh.nodes.copy()
    added, selected, mouths, details = [], [], [], []
    added_count = 0
    for slot_index, slot in enumerate(slots):
        chosen = None
        errors, limit_errors, pattern_errors = [], [], []
        for anchor in slot.candidates:
            try:
                candidate = _candidate(slot, anchor, data)
                _validate_empty(candidate, mesh, data, edge_keys, edge_points, edge_index, cell_index)
                _check_slot_conflict(candidate, selected, tol)
                remaining = None if max_added_elements is None else max_added_elements - added_count
                patch = build_repair_patch(tuple(mesh.nodes[chain] for chain in candidate.source_chains),
                                           data["pattern_records"], data["h"], tol,
                                           max_added_elements=remaining,
                                           repair_max_element_size=repair_max_element_size)
                check_element_limit(added_count + len(patch.quads), max_added_elements,
                                    stage="slot_repair", slot_id=slot_index)
                chosen = candidate, patch
                break
            except ElementLimitExceeded as error:
                if error.context.get("limit") == remaining:
                    error.context["required_elements"] += added_count
                if max_added_elements is not None:
                    error.context["limit"] = int(max_added_elements)
                limit_errors.append(error)
            except QuadLayerError as error:
                if (error.code == "invalid_input"
                        or error.context.get("reason") == "interacting_slots"):
                    raise
                if error.code in {"unsupported_pattern_layout", "unsupported_repair_pattern"}:
                    pattern_errors.append(error)
                errors.append({"anchor_index": int(anchor), "code": error.code,
                               "message": str(error), "context": deepcopy(error.context)})
            except ValueError as error:
                errors.append({"anchor_index": int(anchor), "message": str(error)})
        if chosen is None:
            if limit_errors:
                raise limit_errors[0]
            if pattern_errors and len(pattern_errors) == len(errors):
                error = pattern_errors[0]
                error.context.update(slot_id=slot_index, source_edges=slot.edges, causes=errors)
                raise error
            _fail("neither existing mouth anchor can produce a valid slot repair",
                  code="unmeshable_slot_repair", slot_id=slot_index,
                  source_edges=slot.edges, causes=errors)
        candidate, patch = chosen
        source_local = np.asarray(patch.source_ids, dtype=np.int64)
        if (len(source_local) != len(candidate.source_ids)
                or not np.array_equal(patch.nodes[source_local], mesh.nodes[candidate.source_ids])):
            _fail("the slot builder changed a fixed source chain", code="unmeshable_slot_repair")
        mapping = np.full(len(patch.nodes), -1, dtype=np.int64)
        mapping[source_local] = candidate.source_ids
        new_local = np.flatnonzero(mapping < 0)
        mapping[new_local] = np.arange(len(nodes), len(nodes) + len(new_local))
        nodes = np.vstack((nodes, patch.nodes[new_local]))
        quads = mapping[np.asarray(patch.quads, dtype=np.int64)]
        mouth = mapping[np.asarray(patch.mouth_ids, dtype=np.int64)][::-1]
        if mouth[0] != candidate.source_ids[0] or mouth[-1] != candidate.source_ids[-1]:
            _fail("the slot builder changed the mouth endpoints", code="unmeshable_slot_repair")
        element_start = len(mesh.elements) + added_count
        details.append({**deepcopy(patch.diagnostics), "slot_id": slot_index,
                        "source_edges": list(slot.edges),
                        "source_chains": [chain.tolist() for chain in candidate.source_chains],
                        "mouth_node_ids": mouth.tolist(), "mouth_points": nodes[mouth, :2].tolist(),
                        "polygon": candidate.polygon.tolist(),
                        "element_start": element_start, "element_end": element_start + len(quads)})
        for group in ("units", "faces"):
            for record in details[-1].get(group, ()):
                if "element_start" in record:
                    record["element_start"] += element_start
                if "element_end" in record:
                    record["element_end"] += element_start
        added.append(quads)
        added_count += len(quads)
        selected.append(candidate)
        mouths.append(mouth)
    quads = np.vstack(added)
    outline, directions, provenance, consumed = _regroup(mesh, data, nodes, quads, selected, mouths)
    try:
        _validate_repair(mesh, nodes, quads, outline, data, selected, mouths, consumed)
    except QuadLayerError as error:
        error.context["stage"] = "slot_repair_validation"
        raise
    repaired = Mesh2D(nodes, np.vstack((mesh.elements, quads)))
    diagnostics = {"slot_repairs": details, "repair_added_nodes": len(nodes) - len(mesh.nodes),
                   "repair_added_elements": added_count}
    return RepairResult(repaired, outline, directions, provenance, diagnostics)


__all__ = ["RepairResult", "repair_slots"]
