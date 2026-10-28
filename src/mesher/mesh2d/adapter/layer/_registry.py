"""Shared boundary chains, parity coordination, and direct Quad4 block filling.

Only the boundary of each block is passed to a unit builder.  Original mesh
nodes are never copied once per block, and no triangular mesh is constructed.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from math import floor

import numpy as np

from ...model import Mesh2D
from ..unit import build_unit_1_x_y_z
from ..unit.build_unit_1_1_1_x import _validate_quads
from ..unit.build_unit_1_x_y_z import _stage_inner_ring
from ._types import Block, QuadLayerError, UnmeshableDirectQuadLayout


@dataclass
class _Interface:
    endpoints: tuple[int, int]
    owners: list[int] = field(default_factory=list)
    directions: list[bool] = field(default_factory=list)
    frozen: bool = False
    count: int = 1
    extra_cost: float = 0.0
    node_ids: list[int] = field(default_factory=list)


class _NodePool:
    """Merge planned boundary geometry, without merging unrelated old nodes."""

    def __init__(self, original: np.ndarray, source_ids: np.ndarray, tol: float):
        self.original = original
        self.added: list[np.ndarray] = []
        self.tol = tol
        self.buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        self.z = float(original[int(source_ids[0]), 2])
        for node_id in source_ids:
            node_id = int(node_id)
            existing = self.find(original[node_id, :2])
            if existing is not None and existing != node_id:
                raise QuadLayerError(
                    "distinct source nodes have indistinguishable coordinates",
                    code="source_node_collision", node_ids=(existing, node_id),
                )
            self.buckets[self.key(original[node_id, :2])].append(node_id)

    def key(self, xy: np.ndarray) -> tuple[int, int]:
        return (floor(float(xy[0]) / self.tol), floor(float(xy[1]) / self.tol))

    def point(self, node_id: int) -> np.ndarray:
        if node_id < len(self.original):
            return self.original[node_id]
        return self.added[node_id - len(self.original)]

    def find(self, xy: np.ndarray) -> int | None:
        x, y = self.key(xy)
        found: list[tuple[float, int]] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for node_id in self.buckets.get((x + dx, y + dy), ()):
                    distance = float(np.linalg.norm(self.point(node_id)[:2] - xy))
                    if distance <= self.tol:
                        found.append((distance, node_id))
        return min(found)[1] if found else None

    def boundary_node(self, xy: np.ndarray) -> int:
        found = self.find(xy)
        if found is not None:
            return found
        node_id = len(self.original) + len(self.added)
        self.added.append(np.array((xy[0], xy[1], self.z), dtype=float))
        self.buckets[self.key(xy)].append(node_id)
        return node_id

    def interior_nodes(self, points: np.ndarray) -> np.ndarray:
        """Interior points are private to a block, never coordinate-merged."""
        first = len(self.original) + len(self.added)
        self.added.extend(point.copy() for point in points)
        return np.arange(first, first + len(points), dtype=np.int64)

    def array(self) -> np.ndarray:
        if not self.added:
            return self.original.copy()
        return np.vstack((self.original, np.asarray(self.added)))


class _PointIndex:
    """A sorted-axis range index for noding straight block boundaries."""

    def __init__(self, pool: _NodePool, ids: list[int]):
        self.ids = np.asarray(sorted(set(ids)), dtype=np.int64)
        self.xy = np.asarray([pool.point(int(i))[:2] for i in self.ids])
        self.orders = tuple(np.argsort(self.xy[:, axis], kind="stable") for axis in (0, 1))
        self.sorted_values = tuple(self.xy[order, axis] for axis, order in enumerate(self.orders))
        self.tol = pool.tol

    def on_segment(self, start: np.ndarray, end: np.ndarray) -> list[int]:
        minimum = np.minimum(start, end) - self.tol
        maximum = np.maximum(start, end) + self.tol
        ranges = [
            (int(np.searchsorted(values, minimum[axis], side="left")),
             int(np.searchsorted(values, maximum[axis], side="right")))
            for axis, values in enumerate(self.sorted_values)
        ]
        axis = min((0, 1), key=lambda i: ranges[i][1] - ranges[i][0])
        lo, hi = ranges[axis]
        candidates = self.orders[axis][lo:hi]
        points = self.xy[candidates]
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length <= self.tol:
            raise UnmeshableDirectQuadLayout("a block has a zero-length side")
        delta = points - start
        crosses = direction[0] * delta[:, 1] - direction[1] * delta[:, 0]
        projections = delta @ direction / (length * length)
        mask = (
            (np.abs(crosses) <= self.tol * length)
            & (projections >= -self.tol / length)
            & (projections <= 1.0 + self.tol / length)
            & np.all(points >= minimum, axis=1)
            & np.all(points <= maximum, axis=1)
        )
        candidates = candidates[mask]
        order = np.argsort(projections[mask], kind="stable")
        return self.ids[candidates[order]].tolist()


def _size_cost(length: float, count: int, size: float) -> float:
    # Difference between the squared segment-count errors for n+1 and n.
    return max(0.0, 2.0 * count + 1.0 - 2.0 * length / size)


def _cycle_parity(
    interfaces: dict[tuple[int, int], _Interface], count: int, parity: list[int],
) -> set[tuple[int, int]] | None:
    """Exact binary-state DP for a ring, including optional exterior edges."""
    connections: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    exterior: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for key, edge in interfaces.items():
        if edge.frozen:
            continue
        if len(edge.owners) == 2:
            connections[tuple(sorted(edge.owners))].append(key)
        else:
            exterior[edge.owners[0]].append(key)
    neighbors: dict[int, list[int]] = defaultdict(list)
    for first, second in connections:
        neighbors[first].append(second)
        neighbors[second].append(first)
    if count < 3 or any(len(neighbors[i]) != 2 for i in range(count)):
        return None

    order = [0]
    previous, current = -1, 0
    while True:
        following = min(node for node in neighbors[current] if node != previous)
        if following == 0:
            break
        if following in order:
            return None
        order.append(following)
        previous, current = current, following
    if len(order) != count:
        return None

    def best(keys: list[tuple[int, int]]) -> tuple[int, int]:
        return min(keys, key=lambda key: (interfaces[key].extra_cost, key))

    ring = [best(connections[tuple(sorted((node, order[(i + 1) % count])))])
            for i, node in enumerate(order)]
    boundary = {node: best(keys) for node, keys in exterior.items()}
    solutions: list[tuple[float, int, tuple[tuple[int, int], ...]]] = []
    for initial in (0, 1):
        states: dict[int, tuple[float, int, tuple[tuple[int, int], ...]]] = {
            initial: (0.0, 0, ())
        }
        for position, node in enumerate(order):
            next_states = {}
            allowed = (initial,) if position == count - 1 else (0, 1)
            for preceding, state in states.items():
                for following in allowed:
                    outside = parity[node] ^ preceding ^ following
                    if outside and node not in boundary:
                        continue
                    selected = []
                    if following:
                        selected.append(ring[position])
                    if outside:
                        selected.append(boundary[node])
                    candidate = (
                        state[0] + sum(interfaces[key].extra_cost for key in selected),
                        state[1] + len(selected),
                        state[2] + tuple(selected),
                    )
                    if following not in next_states or candidate < next_states[following]:
                        next_states[following] = candidate
            states = next_states
        if initial in states:
            solutions.append(states[initial])
    if not solutions:
        raise UnmeshableDirectQuadLayout("the cyclic block parity constraints are inconsistent")
    return set(min(solutions)[2])


def _tree_parity(
    interfaces: dict[tuple[int, int], _Interface], count: int, parity: list[int],
) -> set[tuple[int, int]]:
    """Resolve odd faces along a deterministic minimum-cost spanning forest."""
    outside = count
    parent = list(range(count + 1))

    def root(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    adjacency: dict[int, list[tuple[int, tuple[int, int]]]] = defaultdict(list)
    ordered = sorted(
        (key for key, edge in interfaces.items() if not edge.frozen),
        key=lambda key: (interfaces[key].extra_cost, key),
    )
    for key in ordered:
        edge = interfaces[key]
        first = edge.owners[0]
        second = edge.owners[1] if len(edge.owners) == 2 else outside
        first_root, second_root = root(first), root(second)
        if first_root == second_root:
            continue
        parent[first_root] = second_root
        adjacency[first].append((second, key))
        adjacency[second].append((first, key))

    remaining = parity[:] + [0]
    visited: set[int] = set()
    selected: set[tuple[int, int]] = set()
    for start in [outside, *range(count)]:
        if start in visited:
            continue
        visited.add(start)
        traversal = [start]
        predecessors: dict[int, tuple[int, tuple[int, int]]] = {}
        for node in traversal:
            for neighbor, key in adjacency[node]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                predecessors[neighbor] = (node, key)
                traversal.append(neighbor)
        for node in reversed(traversal[1:]):
            previous, key = predecessors[node]
            if remaining[node]:
                selected.add(key)
                remaining[node] ^= 1
                remaining[previous] ^= 1
        if start != outside and remaining[start]:
            raise UnmeshableDirectQuadLayout(
                "fixed source segments make a block component's parity inconsistent",
                block=start,
            )
    return selected


def _coons_patch(
    nodes: np.ndarray, sides: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """A regular quadrilateral grid when opposite side counts agree."""
    if len(sides) != 4 or len(sides[0]) != len(sides[2]) or len(sides[1]) != len(sides[3]):
        raise ValueError("opposite boundary chains have unequal counts")
    bottom, right, top, left = sides[0], sides[1], sides[2][::-1], sides[3][::-1]
    width, height = len(bottom) - 1, len(right) - 1
    grid = np.empty((height + 1, width + 1), dtype=np.int64)
    grid[0], grid[-1], grid[:, 0], grid[:, -1] = bottom, top, left, right
    added = []
    p00, p10, p11, p01 = nodes[[bottom[0], bottom[-1], top[-1], top[0]]]
    for row in range(1, height):
        v = row / height
        for column in range(1, width):
            u = column / width
            bilinear = ((1 - u) * (1 - v) * p00 + u * (1 - v) * p10
                        + u * v * p11 + (1 - u) * v * p01)
            point = ((1 - v) * nodes[bottom[column]] + v * nodes[top[column]]
                     + (1 - u) * nodes[left[row]] + u * nodes[right[row]] - bilinear)
            grid[row, column] = len(nodes) + len(added)
            added.append(point)
    result_nodes = np.vstack((nodes, np.asarray(added).reshape(-1, 3)))
    quads = np.column_stack((grid[:-1, :-1].ravel(), grid[:-1, 1:].ravel(),
                            grid[1:, 1:].ravel(), grid[1:, :-1].ravel()))
    _check_direct_patch(result_nodes, quads, sides)
    return result_nodes, quads


def _check_direct_patch(nodes, quads, sides):
    """Linear local checks; the independent layer validator checks crossings."""
    if np.any(np.diff(np.sort(quads, axis=1), axis=1) == 0):
        raise ValueError("a direct quad repeats a node")
    cells = nodes[quads, :2]
    vectors = np.roll(cells, -1, axis=1) - cells
    following = np.roll(vectors, -1, axis=1)
    turns = vectors[:, :, 0] * following[:, :, 1] - vectors[:, :, 1] * following[:, :, 0]
    scale = np.ptp(cells, axis=1).max(axis=1)
    if np.any(turns <= (64.0 * np.finfo(float).eps * scale * scale)[:, None]):
        raise ValueError("a direct quad is degenerate or inverted")
    boundary = np.concatenate([side[:-1] for side in sides])
    points = nodes[boundary, :2] - nodes[boundary[0], :2]
    following = np.roll(points, -1, axis=0)
    expected_area = float(np.sum(points[:, 0] * following[:, 1] - points[:, 1] * following[:, 0]))
    relative = cells - cells[:, :1]
    following = np.roll(relative, -1, axis=1)
    actual_area = float(np.sum(relative[:, :, 0] * following[:, :, 1]
                               - relative[:, :, 1] * following[:, :, 0]))
    if abs(actual_area - expected_area) > max(1e-10 * expected_area,
                                            512.0 * np.finfo(float).eps):
        raise ValueError("direct quads do not cover the block")
    expected = {tuple(sorted((int(a), int(b)))) for side in sides
                for a, b in zip(side, side[1:])}
    incidence = Counter(tuple(sorted((int(cell[i]), int(cell[(i + 1) % 4]))))
                        for cell in quads for i in range(4))
    if ({key for key, count in incidence.items() if count == 1} != expected
            or any(count != (1 if key in expected else 2) for key, count in incidence.items())):
        raise ValueError("direct quads have inconsistent boundary incidence")


def _large_inner_ring(nodes, sides):
    """Direct radial quads for a large convex boundary, without unit sweeps.

    Large unit sweeps and their all-pairs edge checks can be quadratic or worse.
    Blocks here are convex by construction; an interior circle and an even
    quadrilateral fan give a linear construction, checked globally afterward.
    """
    boundary = np.concatenate([side[:-1] for side in sides])
    points = nodes[boundary, :2]
    center = np.mean(nodes[[int(side[0]) for side in sides], :2], axis=0)
    vectors = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(vectors, axis=1)
    normals = np.column_stack((-vectors[:, 1], vectors[:, 0])) / lengths[:, None]
    clearance = float(np.min(np.sum(normals * (center - points), axis=1)))
    if clearance <= 128.0 * np.finfo(float).eps:
        raise ValueError("the convex block has insufficient interior clearance")
    radius = .5 * clearance
    directions = points - center
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    inner_nodes = np.column_stack((center + radius * directions,
                                  np.full(len(points), nodes[0, 2])))
    inner = np.arange(len(nodes), len(nodes) + len(boundary), dtype=np.int64)
    strips = np.column_stack((boundary, np.roll(boundary, -1), np.roll(inner, -1), inner))
    fan = np.asarray([(inner[0], inner[i], inner[i + 1], inner[i + 2])
                      for i in range(1, len(inner) - 2, 2)], dtype=np.int64)
    result_nodes, quads = np.vstack((nodes, inner_nodes)), np.vstack((strips, fan))
    _check_direct_patch(result_nodes, quads, sides)
    return result_nodes, quads


def _fill_block(
    pool: _NodePool, sides: tuple[list[int], ...], block_index: int,
) -> tuple[np.ndarray, str]:
    global_boundary = np.asarray([node for side in sides for node in side[:-1]], dtype=np.int64)
    if len(global_boundary) % 2 or len(set(global_boundary.tolist())) != len(global_boundary):
        raise UnmeshableDirectQuadLayout("a block boundary is not a simple even cycle", block=block_index)
    local_for_global = {int(node): index for index, node in enumerate(global_boundary)}
    local_sides = tuple(np.asarray([local_for_global[node] for node in side], dtype=np.int64) for side in sides)
    nodes = np.asarray([pool.point(int(node)) for node in global_boundary])
    origin = nodes[0].copy()
    scale = float(np.ptp(nodes[:, :2], axis=0).max())
    if scale <= pool.tol:
        raise UnmeshableDirectQuadLayout("a block has no usable area", block=block_index)
    normalized = (nodes - origin) / scale
    if len(local_sides) != 4:
        boundary = np.arange(len(nodes), dtype=np.int64)
        local_sides = (boundary[:2], boundary[1:3], boundary[2:4], np.r_[boundary[3:], 0])

    try:
        new_nodes, quads = _coons_patch(normalized, local_sides)
        method = "coons"
    except ValueError:
        try:
            if len(global_boundary) > 64:
                raise ValueError("large boundaries use the bounded direct ring strategy")
            if not any(len(side) == 2 for side in local_sides):
                raise ValueError("no one-segment side")
            local_mesh = Mesh2D(normalized, np.empty((0, 4), dtype=np.int64))
            build_unit_1_x_y_z(local_mesh, *local_sides)
            new_nodes, quads = local_mesh.nodes, local_mesh.elements
            method = "unit"
        except ValueError:
            try:
                ring_builder = _large_inner_ring if len(global_boundary) > 64 else _stage_inner_ring
                new_nodes, quads = ring_builder(normalized, local_sides)
                method = "inner_ring"
            except ValueError as error:
                raise UnmeshableDirectQuadLayout(
                    "direct Quad4 builders could not fill a block", block=block_index,
                ) from error

    added = origin + scale * new_nodes[len(nodes):]
    # Preserve the selected constant Z plane exactly.
    added[:, 2] = pool.z
    local_to_global = np.r_[global_boundary, pool.interior_nodes(added)]
    return local_to_global[quads], method


def fill_blocks(
    nodes: np.ndarray,
    blocks: list[Block],
    frozen_segments: np.ndarray,
    element_size: float,
    tol: float,
    *,
    anchors: np.ndarray,
    front: np.ndarray,
    prefer_cycle: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[list[int]], dict]:
    """Fill a conforming block partition, staging all results in fresh arrays.

    Block sides are first split at *all* incident boundary vertices. Every
    resulting atomic interface is owned and subdivided once, even when the
    original neighboring block sides only overlap partially.
    """
    if not blocks:
        raise UnmeshableDirectQuadLayout("the layer contains no blocks")
    source_ids = np.unique(frozen_segments)
    if not len(source_ids):
        raise QuadLayerError("the layer has no frozen source segments")
    pool = _NodePool(nodes, source_ids, tol)
    frozen = {tuple(sorted(map(int, segment))) for segment in frozen_segments}
    point_ids = source_ids.tolist()
    block_vertices: list[list[int]] = []
    for block in blocks:
        ids = [pool.boundary_node(xy) for xy in block.vertices]
        block_vertices.append(ids)
        point_ids.extend(ids)
    # Anchors outside all block edges must not introduce unused mesh nodes.
    anchor_xy = np.asarray(anchors, dtype=float).reshape(-1, 2)
    starts = np.vstack([block.vertices for block in blocks])
    ends = np.vstack([np.roll(block.vertices, -1, axis=0) for block in blocks])
    directions = ends - starts
    lengths = np.linalg.norm(directions, axis=1)
    for xy in anchor_xy:
        delta = xy - starts
        cross = directions[:, 0] * delta[:, 1] - directions[:, 1] * delta[:, 0]
        dot = np.sum(delta * directions, axis=1)
        if np.any((np.abs(cross) <= tol * lengths)
                  & (dot >= -tol * lengths) & (dot <= lengths * lengths + tol * lengths)):
            point_ids.append(pool.boundary_node(xy))
    index = _PointIndex(pool, point_ids)
    interfaces: dict[tuple[int, int], _Interface] = {}
    block_sides: list[list[list[tuple[tuple[int, int], bool]]]] = []
    for block_id, corner_ids in enumerate(block_vertices):
        sides = []
        for first, last in zip(corner_ids, corner_ids[1:] + corner_ids[:1]):
            chain = index.on_segment(pool.point(first)[:2], pool.point(last)[:2])
            if len(chain) < 2 or chain[0] != first or chain[-1] != last:
                raise UnmeshableDirectQuadLayout("block side endpoints did not survive noding")
            side = []
            for a, b in zip(chain, chain[1:]):
                key = tuple(sorted((a, b)))
                edge = interfaces.setdefault(key, _Interface(key, frozen=key in frozen))
                if block_id in edge.owners or len(edge.owners) >= 2:
                    raise UnmeshableDirectQuadLayout("a block interface is not manifold", edge=key)
                forward = a == key[0]
                if edge.directions and edge.directions[0] == forward:
                    raise UnmeshableDirectQuadLayout(
                        "neighboring blocks traverse a shared interface in the same direction", edge=key,
                    )
                edge.owners.append(block_id)
                edge.directions.append(forward)
                side.append((key, forward))
            sides.append(side)
        block_sides.append(sides)
    missing = frozen.difference(interfaces)
    if missing:
        raise QuadLayerError(
            "a mandatory anchor would split or omit a frozen source segment",
            code="fixed_boundary_conflict", segments=sorted(missing),
        )
    for key in frozen:
        if len(interfaces[key].owners) != 1:
            raise UnmeshableDirectQuadLayout("source segment must border exactly one new block", edge=key)

    parity = [0] * len(blocks)
    for edge in interfaces.values():
        length = float(np.linalg.norm(pool.point(edge.endpoints[1])[:2] - pool.point(edge.endpoints[0])[:2]))
        if not edge.frozen:
            edge.count = max(1, floor(length / element_size + 0.5))
            edge.extra_cost = _size_cost(length, edge.count, element_size)
        for owner in edge.owners:
            parity[owner] ^= edge.count % 2
    selected = _cycle_parity(interfaces, len(blocks), parity) if prefer_cycle else None
    solver = "cyclic_dp" if selected is not None else "dual_tree"
    if selected is None:
        selected = _tree_parity(interfaces, len(blocks), parity)
    for key in selected:
        interfaces[key].count += 1
        for owner in interfaces[key].owners:
            parity[owner] ^= 1
    if any(parity):
        raise UnmeshableDirectQuadLayout("internal error: parity solver left an odd block")

    for edge in interfaces.values():
        first, last = edge.endpoints
        start, end = pool.point(first)[:2], pool.point(last)[:2]
        interior = [pool.boundary_node(start + (end - start) * (i / edge.count))
                    for i in range(1, edge.count)]
        edge.node_ids = [first, *interior, last]
        if len(set(edge.node_ids)) != len(edge.node_ids):
            raise UnmeshableDirectQuadLayout("an interface subdivision is below numerical resolution")

    patches = []
    methods: dict[str, int] = defaultdict(int)
    for block_id, sides in enumerate(block_sides):
        chains = []
        for intervals in sides:
            chain = []
            for key, forward in intervals:
                ids = interfaces[key].node_ids
                oriented = ids if forward else ids[::-1]
                chain.extend(oriented[:-1])
            chain.append(oriented[-1])
            chains.append(chain)
        quads, method = _fill_block(pool, tuple(chains), block_id)
        patches.append(quads)
        methods[method] += 1

    boundary_ids = sorted({node for edge in interfaces.values() if len(edge.owners) == 1
                           for node in edge.node_ids})
    front_index = _PointIndex(pool, boundary_ids)
    outline = [front_index.on_segment(start, end)
               for start, end in zip(front, np.roll(front, -1, axis=0))]
    boundary_pairs = {tuple(sorted((a, b))) for edge in interfaces.values() if len(edge.owners) == 1
                      for a, b in zip(edge.node_ids, edge.node_ids[1:])}
    for side in outline:
        if len(side) < 2 or any(tuple(sorted(pair)) not in boundary_pairs for pair in zip(side, side[1:])):
            raise UnmeshableDirectQuadLayout("the new outline is not a conforming boundary chain")
    for side, following in zip(outline, outline[1:] + outline[:1]):
        if side[-1] != following[0]:
            raise UnmeshableDirectQuadLayout("the new outline is not closed")
    result_nodes = pool.array()
    result_quads = np.vstack(patches).astype(np.int64, copy=False)
    # Validate each cell locally: a remote old mesh extent must not set tolerance.
    for quad in result_quads:
        xy = result_nodes[quad, :2]
        scale = float(np.ptp(xy, axis=0).max())
        if scale <= 0.0:
            raise UnmeshableDirectQuadLayout("a generated Quad4 has zero extent")
        local = np.column_stack(((xy - xy[0]) / scale, np.zeros(4)))
        try:
            _validate_quads(local, np.array([[0, 1, 2, 3]], dtype=np.int64))
        except ValueError as error:
            raise UnmeshableDirectQuadLayout(
                "a generated Quad4 is degenerate or inverted", node_ids=quad.tolist(),
            ) from error
    return result_nodes, result_quads, outline, {
        "parity_solver": solver,
        "interface_count": len(interfaces),
        "parity_adjustments": len(selected),
        "block_builders": dict(methods),
    }


__all__ = ["fill_blocks"]
