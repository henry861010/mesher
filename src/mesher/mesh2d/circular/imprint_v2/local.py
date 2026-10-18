"""Conservative circle patches, topology closure, and atomic global stitching.

Only the broad phase, lazy indexes, and final assembly see the entire mesh.
Geometry repair and circle splitting operate on compact local connectivity.
"""
from __future__ import annotations

from collections import deque
from time import perf_counter

import numpy as np

from ...model import Mesh2D
from ..utils.pattern_segments import _coerce_pattern_guides
from .triangulation import _triangulate_flat_quads
from .utils import (
    _cleanup_data,
    _coordinate_bin,
    _counter_clockwise_elements,
    _node_groups_touch,
    _normalize_circle,
    _normalize_nonnegative,
    _project_circle,
    _split_circle,
    _warn_invalid_removals,
)

_BATCH_SIZE = 65_536


def _circle_patch_rows(nodes, elements, center, radius, width):
    """Conservative annulus/AABB overlap, including edges with distant vertices."""
    selected = np.zeros(len(elements), dtype=np.bool_)
    for begin in range(0, len(elements), _BATCH_SIZE):
        end = min(begin + _BATCH_SIZE, len(elements))
        xy = nodes[elements[begin:end], :2]
        with np.errstate(over="ignore", invalid="ignore"):
            lower, upper = xy.min(axis=1) - center, xy.max(axis=1) - center
        nearest = np.maximum(np.maximum(lower, -upper), 0.0)
        farthest = np.maximum(np.abs(lower), np.abs(upper))
        selected[begin:end] = (
            (np.hypot(nearest[:, 0], nearest[:, 1]) <= radius + width)
            & (np.hypot(farthest[:, 0], farthest[:, 1]) >= max(0.0, radius - width))
        )
    return np.flatnonzero(selected)


class _NodeElements:
    """CSR node incidence, built once for all closure queries in one operation."""

    def __init__(self, elements, node_count):
        flat = elements.ravel()
        counts = np.bincount(flat, minlength=node_count)
        self.offsets = np.r_[0, np.cumsum(counts)]
        self.owners = np.argsort(flat, kind="stable") // 4
        self.referenced = np.flatnonzero(counts)

    def incident(self, nodes):
        nodes = np.unique(nodes)
        counts = self.offsets[nodes + 1] - self.offsets[nodes]
        total = int(counts.sum())
        if not total:
            return np.empty(0, dtype=np.int64)
        starts = np.cumsum(counts) - counts
        positions = np.repeat(self.offsets[nodes], counts)
        positions += np.arange(total) - np.repeat(starts, counts)
        return np.unique(self.owners[positions])


class _MergeClosure:
    """Explore only equivalence components seeded by local nodes.

    Half-tolerance buckets are internally connected. Neighbouring buckets
    join only after an exact distance test. A global O(N) bucket construction
    replaces the former 49-neighbour search for every remote mesh node.
    """

    def __init__(self, nodes, referenced, tolerance):
        self.xy = nodes[:, :2]
        self.tolerance = tolerance
        self.width = tolerance * 0.5 or tolerance
        self.cells = {}
        self.keys = {}
        self.visited = set()
        xy = self.xy[referenced]
        if tolerance == 0.0:
            keys = zip(xy[:, 0].tolist(), xy[:, 1].tolist())
        else:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                bins = np.floor(xy / self.width)
            if np.all(np.isfinite(bins)) and np.all(np.abs(bins) <= 2.0**52):
                bins = bins.astype(np.int64)
                keys = zip(bins[:, 0].tolist(), bins[:, 1].tolist())
            else:
                keys = (
                    (_coordinate_bin(float(x), self.width),
                     _coordinate_bin(float(y), self.width))
                    for x, y in xy
                )
        for node, key in zip(referenced.tolist(), keys, strict=True):
            self.cells.setdefault(key, []).append(node)
            self.keys[node] = key
        self.neighbours = tuple(
            (x, y) for x in range(-3, 4) for y in range(-3, 4) if x or y
        )

    def merged_components(self, seeds):
        joined = []
        for node in seeds:
            key = self.keys.get(int(node))
            if key is None or key in self.visited:
                continue
            self.visited.add(key)
            queue = deque([key])
            component = []
            while queue:
                current = queue.popleft()
                members = self.cells[current]
                component.extend(members)
                if self.tolerance == 0.0:
                    continue
                for dx, dy in self.neighbours:
                    neighbour = current[0] + dx, current[1] + dy
                    if neighbour in self.visited or neighbour not in self.cells:
                        continue
                    if _node_groups_touch(
                        self.xy, members, self.cells[neighbour], self.tolerance,
                    ):
                        self.visited.add(neighbour)
                        queue.append(neighbour)
            if len(component) > 1:
                joined.extend(component)
        return np.asarray(joined, dtype=np.int64)


def _assemble(nodes, elements, rows, local_nodes, local_elements, origins, sources):
    """Stitch by provenance, then compact once; never recover IDs from coordinates."""
    existing = origins >= 0
    updated = nodes.copy()
    updated[origins[existing]] = local_nodes[existing]
    new_count = int(np.count_nonzero(~existing))
    if len(nodes) + new_count > np.iinfo(np.int32).max:
        raise ValueError("imprint would exceed the int32 node-index range")
    local_to_global = origins.copy()
    local_to_global[~existing] = len(nodes) + np.arange(new_count)
    if new_count:
        updated = np.concatenate((updated, local_nodes[~existing]))
    mapped = local_to_global[local_elements]
    global_sources = rows[sources]
    first_sources, first = np.unique(global_sources, return_index=True)
    retained = np.ones(len(elements), dtype=np.bool_)
    retained[rows] = False
    retained[first_sources] = True
    output = elements.copy()
    output[first_sources] = mapped[first]
    extra = np.ones(len(mapped), dtype=np.bool_)
    extra[first] = False
    output = np.concatenate((output[retained], mapped[extra]))
    referenced = np.unique(output)
    mapping = np.full(len(updated), -1, dtype=np.int64)
    mapping[referenced] = np.arange(len(referenced))
    return Mesh2D(updated[referenced], mapping[output])


def _run_local_imprint(
    mesh, *, center, radius, projection_tolerance, guide_tolerance,
    merge_tolerance, minimum_area, guide_segments=None, flat_quad_angle=170.0,
    observer=None, stats=None,
):
    """Run one circle atomically; observer/stats are private diagnostic hooks.

    Observer receives (stage, global_snapshot, provenance). Normal calls do
    not allocate stage snapshots. Structurally invalid inputs still fail
    before selection, while remote geometric defects remain untouched.
    """
    started = perf_counter()
    if not isinstance(mesh, Mesh2D):
        raise TypeError("mesh must be a Mesh2D instance")
    center, radius = _normalize_circle(center, radius)
    projection_tolerance = _normalize_nonnegative(
        projection_tolerance, "projection_tolerance",
    )
    guide_tolerance = _normalize_nonnegative(guide_tolerance, "guide_tolerance")
    merge_tolerance = _normalize_nonnegative(merge_tolerance, "merge_tolerance")
    minimum_area = min(1.0, _normalize_nonnegative(minimum_area, "minimum_area"))
    flat_quad_angle = _normalize_nonnegative(flat_quad_angle, "flat_quad_angle")
    if not 0.0 < flat_quad_angle <= 180.0:
        raise ValueError("flat_quad_angle must be greater than 0 and at most 180 degrees")
    nodes, elements = mesh.nodes, mesh.elements
    if not np.all(np.isfinite(nodes)):
        raise ValueError("nodes must contain finite coordinates")
    if elements.size and (elements.min() < 0 or elements.max() >= len(nodes)):
        raise ValueError("elements contain an out-of-range node index")
    scale = max(1.0, float(np.max(np.abs(center))), radius)
    epsilon = 1024 * np.finfo(float).eps * scale
    guides = _coerce_pattern_guides(
        guide_segments, coordinate_scale=scale,
        minimum_tolerance=128 * np.finfo(float).eps * scale,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        width = projection_tolerance + merge_tolerance + epsilon
        bounds = np.r_[center - radius - width, center + radius + width]
    if not np.all(np.isfinite(bounds)):
        raise ValueError("circle bounds exceed the float64 range")
    data = {} if stats is None else stats
    data.update(global_elements=len(elements), global_nodes=len(nodes))
    core = _circle_patch_rows(nodes, elements, center, radius, width)
    data.update(core_elements=len(core), selection_seconds=perf_counter() - started)
    if not len(core):
        data.update(
            patch_elements=0, patch_nodes=0,
            triangulated_quads=0, triangulation_seconds=0.0,
            total_seconds=perf_counter() - started,
        )
        if observer is not None:
            for stage in ("projection", "cleanup", "split"):
                observer(stage, Mesh2D(nodes, elements), {"patch_rows": core})
        return mesh
    index_started = perf_counter()
    adjacency = _NodeElements(elements, len(nodes))
    rows = adjacency.incident(np.unique(elements[core]))
    data["adjacency_seconds"] = perf_counter() - index_started
    extraction_started = perf_counter()
    node_ids, inverse = np.unique(elements[rows], return_inverse=True)
    local = Mesh2D(nodes[node_ids], inverse.reshape(-1, 4))
    data["extraction_seconds"] = perf_counter() - extraction_started
    projection_started = perf_counter()
    _project_circle(
        local, center, radius, projection_tolerance, guide_tolerance, guides,
        constraint_nodes=nodes,
    )
    working_nodes = nodes.copy()
    working_nodes[node_ids] = local.nodes
    data["projection_seconds"] = perf_counter() - projection_started
    closure_started = perf_counter()
    closure = _MergeClosure(working_nodes, adjacency.referenced, merge_tolerance)
    data["merge_index_seconds"] = perf_counter() - closure_started
    selected = np.zeros(len(elements), dtype=np.bool_)
    selected[rows] = True
    cleanup_seconds = split_seconds = 0.0
    iterations = 0
    closure_started = perf_counter()
    while True:
        # Newly included rows may expose further merge components. A single
        # ordinary shared node does not expand the patch indefinitely.
        while True:
            node_ids = np.unique(elements[rows])
            merged = closure.merged_components(node_ids)
            affected = adjacency.incident(merged)
            new_rows = affected[~selected[affected]]
            if not len(new_rows):
                break
            selected[new_rows] = True
            rows = np.flatnonzero(selected)
        node_ids, inverse = np.unique(elements[rows], return_inverse=True)
        projected_elements = _counter_clockwise_elements(
            working_nodes[node_ids], inverse.reshape(-1, 4),
        )
        cleanup_started = perf_counter()
        discarded = []
        cleanup = _cleanup_data(
            working_nodes[node_ids], projected_elements, merge_tolerance, minimum_area,
            discarded=discarded,
        )
        clean_nodes, clean_elements, surviving, representatives, cleanup_sources = cleanup
        cleanup_seconds += perf_counter() - cleanup_started
        local = Mesh2D(clean_nodes, clean_elements)
        split_started = perf_counter()
        local, split_sources, new_edges = _split_circle(
            local, center, radius, merge_tolerance,
        )
        split_seconds += perf_counter() - split_started
        iterations += 1
        clean_origins = node_ids[surviving]
        changed = np.any(
            local.nodes[:len(clean_nodes)] != working_nodes[clean_origins], axis=1,
        )
        affected_nodes = clean_origins[changed]
        # Every owner of an edge receiving a new intersection must participate.
        if len(new_edges):
            affected_nodes = np.union1d(
                affected_nodes, clean_origins[np.unique(new_edges)],
            )
        affected = adjacency.incident(affected_nodes)
        new_rows = affected[~selected[affected]]
        if not len(new_rows):
            break
        selected[new_rows] = True
        rows = np.flatnonzero(selected)
    origins = np.r_[
        clean_origins,
        np.full(len(local.nodes) - len(clean_origins), -1, dtype=np.int64),
    ]
    triangulation_started = perf_counter()
    triangulated, triangulation_sources = _triangulate_flat_quads(
        local.nodes, local.elements, flat_quad_angle=flat_quad_angle,
    )
    if triangulated is not local.elements:
        local.elements = triangulated
    data["triangulation_seconds"] = perf_counter() - triangulation_started
    data["triangulated_quads"] = len(triangulated) - len(split_sources)
    sources = cleanup_sources[split_sources[triangulation_sources]]
    data.update(
        patch_elements=len(rows), patch_nodes=len(node_ids),
        closure_iterations=iterations, cleanup_seconds=cleanup_seconds,
        split_seconds=split_seconds,
        closure_seconds=(perf_counter() - closure_started - cleanup_seconds
                         - split_seconds - data["triangulation_seconds"]),
    )
    metadata = {
        "patch_rows": rows, "node_ids": node_ids,
        "representatives": node_ids[representatives],
        "cleanup_sources": rows[cleanup_sources],
        "cleanup_elements": node_ids[surviving][clean_elements],
        "split_sources": rows[sources],
    }
    if observer is not None:
        projected = elements.copy()
        projected[rows] = node_ids[projected_elements]
        observer("projection", Mesh2D(working_nodes, projected), metadata)
        snapshot = _assemble(
            working_nodes, elements, rows, clean_nodes, clean_elements,
            clean_origins, cleanup_sources,
        )
        observer("cleanup", snapshot, metadata)
    stitch_started = perf_counter()
    result = _assemble(
        working_nodes, elements, rows, local.nodes, local.elements, origins, sources,
    )
    data["stitch_seconds"] = perf_counter() - stitch_started
    if observer is not None:
        observer("split", result, metadata)
    commit_started = perf_counter()
    _warn_invalid_removals(discarded, element_rows=rows, stacklevel=4)
    if not (np.array_equal(result.nodes, nodes) and np.array_equal(result.elements, elements)):
        mesh.replace_data(nodes=result.nodes, elements=result.elements)
    data["commit_seconds"] = perf_counter() - commit_started
    data["total_seconds"] = perf_counter() - started
    return mesh
