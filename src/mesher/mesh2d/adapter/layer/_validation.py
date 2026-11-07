"""Independent geometric and topological checks before a layer is committed."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ._geometry import BoxIndex, area, cross, intervals_cover, segment_contacts
from ._patterns import covered_intervals, occupied_intervals
from ._types import QuadLayerError, UnmeshableDirectQuadLayout


def _failure(message, **context):
    raise UnmeshableDirectQuadLayout(message, **context)


def _ccw_cells(points):
    if not len(points):
        return points.copy()
    shifted = points - points[:, :1]
    signs = np.sum(cross(shifted, np.roll(shifted, -1, axis=1)), axis=1)
    result = points.copy()
    result[signs < 0] = result[signs < 0, ::-1]
    return result


def _cell_overlap(first, others, tol):
    """Vectorized strict separating-axis test against convex CCW cells."""
    first_edges = np.roll(first, -1, axis=0) - first
    other_edges = np.roll(others, -1, axis=1) - others
    first_sides = cross(first_edges[:, None, None, :],
                        others[None, :, :, :] - first[:, None, None, :])
    first_separates = np.any(
        first_sides.max(axis=2) <= tol * np.linalg.norm(first_edges, axis=1)[:, None], axis=0,
    )
    other_sides = cross(other_edges[:, :, None, :],
                        first[None, None, :, :] - others[:, :, None, :])
    other_separates = np.any(
        other_sides.max(axis=2) <= tol * np.linalg.norm(other_edges, axis=2), axis=1,
    )
    return ~(first_separates | other_separates)


def _check_overlaps(new_cells, old_cells, new_index, old_index, tol):
    for i, cell in enumerate(new_cells):
        lower, upper = cell.min(axis=0) - tol, cell.max(axis=0) + tol
        candidates = new_index.query(lower, upper)
        candidates = candidates[candidates > i]
        if len(candidates):
            overlaps = candidates[_cell_overlap(cell, new_cells[candidates], tol)]
            if len(overlaps):
                _failure("new Quad4 cells overlap", cells=(i, int(overlaps[0])))
        candidates = old_index.query(lower, upper)
        if len(candidates):
            overlaps = candidates[_cell_overlap(cell, old_cells[candidates], tol)]
            if len(overlaps):
                raise QuadLayerError(
                    "the layer overlaps an existing element", code="topology_change",
                    new_cell=i, old_cell=int(overlaps[0]),
                )


def _edge_conflicts(edge_ids, edge, other_ids, others, tol):
    """Crossings, overlaps and coordinate-only contacts lack shared topology."""
    if not len(others):
        return np.zeros(0, dtype=bool)
    a, b = edge
    c, d = others[:, 0], others[:, 1]
    u, v = b - a, d - c
    length = float(np.linalg.norm(u))
    other_lengths = np.linalg.norm(v, axis=1)
    if length <= tol or np.any(other_lengths <= tol):
        _failure("mesh contains a zero-length geometric edge")
    determinant = cross(u, v)
    product0, product1 = u[0] * v[:, 1], u[1] * v[:, 0]
    uncertain = ((np.abs(determinant) <= 8.0 * np.finfo(float).eps
                  * (np.abs(product0) + np.abs(product1)))
                 & (product0 != 0) & (product1 != 0))
    # Geometric tolerance is not an angular cutoff: an arbitrarily shallow
    # nonparallel segment can still cross a short edge in its interior.
    transverse = (determinant != 0) & ~uncertain
    shared = np.any(edge_ids[:, None, None] == other_ids[None, :, :], axis=(0, 2))
    identical = np.all(other_ids == edge_ids, axis=1)
    conflicts = np.zeros(len(others), dtype=bool)
    if np.any(transverse):
        ids = np.flatnonzero(transverse)
        fraction = cross(c[ids] - a, v[ids]) / determinant[ids]
        other_fraction = cross(c[ids] - a, u) / determinant[ids]
        contact = ((fraction >= -tol / length) & (fraction <= 1 + tol / length)
                   & (other_fraction >= -tol / other_lengths[ids])
                   & (other_fraction <= 1 + tol / other_lengths[ids]))
        conflicts[ids] = contact & ~shared[ids]
    parallel = (determinant == 0) & ~uncertain
    collinear = parallel & (np.abs(cross(u, c - a)) <= tol * length)
    if np.any(collinear):
        ids = np.flatnonzero(collinear)
        t0 = (c[ids] - a) @ u / (length * length)
        t1 = (d[ids] - a) @ u / (length * length)
        lower = np.maximum(0.0, np.minimum(t0, t1))
        upper = np.minimum(1.0, np.maximum(t0, t1))
        overlap = (upper - lower) * length
        conflicts[ids] = ((overlap > tol)
                          | ((overlap >= -tol) & ~shared[ids])) & ~identical[ids]
    for index in np.flatnonzero(uncertain):
        if identical[index]:
            continue
        contacts = segment_contacts(a, b, c[index], d[index], tol)
        if len(contacts) == 2 and np.linalg.norm(contacts[1] - contacts[0]) > tol:
            conflicts[index] = True
        elif contacts and not shared[index]:
            conflicts[index] = True
    return conflicts


def _check_edge_geometry(new_keys, new_edges, old_keys, old_edges, new_index, old_index, tol):
    for i, edge in enumerate(new_edges):
        lower, upper = edge.min(axis=0) - tol, edge.max(axis=0) + tol
        candidates = new_index.query(lower, upper)
        candidates = candidates[candidates > i]
        if len(candidates):
            bad = candidates[_edge_conflicts(new_keys[i], edge, new_keys[candidates],
                                             new_edges[candidates], tol)]
            if len(bad):
                _failure("new mesh edges cross, overlap, or form a hanging node",
                         edges=(new_keys[i].tolist(), new_keys[int(bad[0])].tolist()))
        candidates = old_index.query(lower, upper)
        if len(candidates):
            bad = candidates[_edge_conflicts(new_keys[i], edge, old_keys[candidates],
                                             old_edges[candidates], tol)]
            if len(bad):
                raise QuadLayerError(
                    "a new edge crosses or touches existing mesh without a shared node",
                    code="topology_change", new_edge=new_keys[i].tolist(),
                    old_edge=old_keys[int(bad[0])].tolist(),
                )


def _check_patterns(patterns, new_cells, old_cells, new_cell_index, old_cell_index,
                    new_edges, old_edges, new_edge_index, old_edge_index, tol):
    for i, pattern in enumerate(patterns):
        lower, upper = pattern.bounds(tol)
        required = occupied_intervals(pattern, new_cells[new_cell_index.query(lower, upper)], tol)
        required += occupied_intervals(pattern, old_cells[old_cell_index.query(lower, upper)], tol)
        covering = covered_intervals(pattern, new_edges[new_edge_index.query(lower, upper)], tol)
        covering += covered_intervals(pattern, old_edges[old_edge_index.query(lower, upper)], tol)
        if not intervals_cover(required, covering, tol):
            _failure("a finite pattern cuts through a mesh element", pattern_index=i)


def validate_patch(mesh, nodes, quads, outline, data):
    """Certify a staged patch without mutating any input mesh or array."""
    tol = data["tol"]
    original_count = len(mesh.nodes)
    if (nodes.ndim != 2 or nodes.shape[1] != 3 or len(nodes) < original_count
            or not np.all(np.isfinite(nodes))):
        _failure("the staged node array is invalid")
    if not np.array_equal(nodes[:original_count, :2], data["xy"]):
        _failure("the staged layer modified original node coordinates")
    if (quads.ndim != 2 or quads.shape[1] != 4 or not len(quads)
            or not np.issubdtype(quads.dtype, np.integer)
            or np.any(quads < 0) or np.any(quads >= len(nodes))
            or np.any(np.diff(np.sort(quads, axis=1), axis=1) == 0)):
        _failure("every added element must be a Quad4 with four distinct valid node IDs")
    new_cells = nodes[quads, :2]
    vectors = np.roll(new_cells, -1, axis=1) - new_cells
    turns = cross(vectors, np.roll(vectors, -1, axis=1))
    local_scale = np.ptp(new_cells, axis=1).max(axis=1)
    threshold = 64.0 * np.finfo(float).eps * local_scale * local_scale
    if np.any(turns <= threshold[:, None]):
        _failure("added Quad4 cells must be nondegenerate, strictly convex, and CCW")
    new_used = np.unique(quads[quads >= original_count])
    if not np.array_equal(new_used, np.arange(original_count, len(nodes))):
        _failure("the layer contains unused newly generated nodes")

    incidence = defaultdict(list)
    for i, cell in enumerate(quads):
        for a, b in zip(cell, np.roll(cell, -1)):
            incidence[tuple(sorted((int(a), int(b))))].append((i, int(a), int(b)))
    expected = {tuple(sorted((int(a), int(b)))): (int(a), int(b))
                for a, b in data["frozen"]}
    if len(outline) != len(data["front"]):
        _failure("the new logical outline has the wrong edge count")
    for edge_index, chain in enumerate(outline):
        ids = np.asarray(chain)
        if (ids.ndim != 1 or len(ids) < 2 or not np.issubdtype(ids.dtype, np.integer)
                or np.any(ids < 0) or np.any(ids >= len(nodes))):
            _failure("the new outline has an invalid node chain", edge=edge_index)
        if chain[-1] != outline[(edge_index + 1) % len(outline)][0]:
            _failure("the new outline is not closed", edge=edge_index)
        start = data["front"][edge_index]
        end = data["front"][(edge_index + 1) % len(outline)]
        delta = end - start
        length = float(np.linalg.norm(delta))
        relative = nodes[ids, :2] - start
        projections = relative @ delta / (length * length)
        if (np.linalg.norm(relative[0]) > tol or np.linalg.norm(nodes[ids[-1], :2] - end) > tol
                or np.any(np.abs(cross(delta, relative)) > tol * length)
                or np.any(np.diff(projections) <= 0)):
            _failure("new outline nodes do not follow the planned front", edge=edge_index)
        for a, b in zip(chain, chain[1:]):
            key = tuple(sorted((int(a), int(b))))
            if key in expected:
                _failure("source and front boundary segments overlap")
            expected[key] = (int(b), int(a))
    target_ids = set(map(int, data.get("outline_anchor_ids", ())))
    front_ids = set(np.concatenate(outline).tolist())
    if not target_ids.issubset(front_ids):
        _failure("the new outline omitted a specified node ID",
                 parameter="outline_o", node_id=min(target_ids - front_ids))
    for key, owners in incidence.items():
        if key in expected:
            if len(owners) != 1 or owners[0][1:] != expected[key]:
                _failure("source/front boundary has incorrect directed incidence", edge=key)
        elif len(owners) != 2 or owners[0][1:] != owners[1][1:][::-1]:
            _failure("internal edge must have two oppositely oriented owners", edge=key)
    if not set(expected).issubset(incidence):
        _failure("the layer omitted a source or front boundary segment")

    source_keys = {tuple(sorted(map(int, edge))) for edge in data["frozen"]}
    front_keys = set(expected).difference(source_keys)
    shared_front = front_keys.intersection(data["old_incidence"])
    for key in shared_front:
        old_owners = data["old_incidence"][key]
        if not set(key).issubset(target_ids) or len(old_owners) != 1:
            raise QuadLayerError("the target would create a nonconforming shared edge",
                                 code="topology_change", parameter="outline_o", edge=key)
    unexpected_old_edges = (set(incidence).intersection(data["old_incidence"])
                            .difference(source_keys | shared_front))
    if unexpected_old_edges:
        raise QuadLayerError("the layer meets old mesh away from its source boundary",
                             code="topology_change", edge=min(unexpected_old_edges))
    source_ids = set(np.asarray(data["frozen"]).ravel().tolist())
    old_nodes_used = set(quads[quads < original_count].tolist())
    if not old_nodes_used.issubset(source_ids | target_ids):
        _failure("new elements reference old nodes outside the source and target boundaries")

    shifted = new_cells - new_cells[:, :1]
    actual_area = float(np.sum(cross(shifted, np.roll(shifted, -1, axis=1))) * .5)
    expected_area = abs(area(data["outer"])) - abs(area(data["inner"]))
    perimeter = sum(float(np.linalg.norm(np.roll(loop, -1, axis=0) - loop, axis=1).sum())
                    for loop in (data["outer"], data["inner"]))
    area_tolerance = max(8.0 * tol * perimeter, 1e-10 * expected_area)
    if expected_area <= 0 or abs(actual_area - expected_area) > area_tolerance:
        _failure("new Quad4 cells do not cover the complete planned band",
                 actual_area=actual_area, expected_area=expected_area)

    new_keys = np.asarray(sorted(incidence), dtype=np.int64).reshape(-1, 2)
    old_keys = np.asarray(sorted(data["old_incidence"]), dtype=np.int64).reshape(-1, 2)
    new_edges = nodes[new_keys, :2]
    old_edges = data["xy"][old_keys]
    new_edge_index = BoxIndex(new_edges.min(axis=1), new_edges.max(axis=1), tol)
    old_edge_index = BoxIndex(old_edges.min(axis=1), old_edges.max(axis=1), tol)
    _check_edge_geometry(new_keys, new_edges, old_keys, old_edges,
                         new_edge_index, old_edge_index, tol)

    old_cells = _ccw_cells(data["xy"][mesh.elements])
    new_cell_index = BoxIndex(new_cells.min(axis=1), new_cells.max(axis=1), tol)
    old_cell_index = BoxIndex(old_cells.min(axis=1), old_cells.max(axis=1), tol)
    _check_overlaps(new_cells, old_cells, new_cell_index, old_cell_index, tol)
    _check_patterns(data["pattern_records"], new_cells, old_cells, new_cell_index, old_cell_index,
                    new_edges, old_edges, new_edge_index, old_edge_index, tol)


__all__ = ["validate_patch"]
