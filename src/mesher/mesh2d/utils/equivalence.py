"""Shared XY node grouping without dense distance matrices."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


_BATCH_SIZE = 65_536
_CELL_DTYPE = np.dtype([("x", np.int64), ("y", np.int64)])
_NEIGHBOR_OFFSETS = tuple(
    (x, y)
    for x in range(4)
    for y in range(-3, 4)
    if x > 0 or y > 0
)


def _connected_components(count, left, right):
    """Connect cell pairs by batched root hooking and pointer doubling."""
    parent = np.arange(count, dtype=np.int64)
    while left.size:
        left_root = parent[left]
        right_root = parent[right]
        different = left_root != right_root
        if not np.any(different):
            break
        high = np.maximum(left_root[different], right_root[different])
        low = np.minimum(left_root[different], right_root[different])
        np.minimum.at(parent, high, low)
        while True:
            grandparent = parent[parent]
            if np.array_equal(parent, grandparent):
                break
            parent = grandparent
    return parent


def _equivalent_node_mapping(
    nodes: NDArray[np.float64],
    referenced_nodes: NDArray[np.int64],
    tolerance: float,
) -> NDArray[np.int64]:
    """Map finite referenced XY nodes to the lowest ID of each component.

    Half-tolerance cells are internally connected. Vectorized sorted lookups
    locate nearby cells, bounding boxes settle most pairs, and only ambiguous
    pairs need blocked point distances. No global point-pair matrix is built.
    """
    result = np.arange(len(nodes), dtype=np.int64)
    if len(referenced_nodes) < 2:
        return result
    xy = nodes[referenced_nodes, :2]
    width = tolerance * 0.5
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        bins = np.floor(xy / width) if width else np.full_like(xy, np.inf)

    if tolerance == 0 or not np.all(np.abs(bins) <= 2.0**50):
        # Preserve exact-ratio handling for extreme scales and subnormal tol.
        # First compress duplicate coordinates so this rare fallback never
        # compares a dense cloud of identical points pair by pair.
        _, inverse = np.unique(xy, axis=0, return_inverse=True)
        owners = np.full(int(inverse.max()) + 1, len(nodes), dtype=np.int64)
        np.minimum.at(owners, inverse, referenced_nodes)
        if tolerance:
            mapping = _fallback_equivalent_node_mapping(nodes, owners, tolerance)
            owners = mapping[owners]
        result[referenced_nodes] = owners[inverse]
        return result

    cells, inverse, counts = np.unique(
        bins.astype(np.int64), axis=0, return_inverse=True, return_counts=True
    )
    order = np.argsort(inverse, kind="stable")
    starts = np.cumsum(counts) - counts
    sorted_xy = xy[order]
    minimum = np.minimum.reduceat(sorted_xy, starts)
    maximum = np.maximum.reduceat(sorted_xy, starts)
    keys = cells.view(_CELL_DTYPE).reshape(-1)
    left_pairs = []
    right_pairs = []

    for offset in _NEIGHBOR_OFFSETS:
        for begin in range(0, len(cells), _BATCH_SIZE):
            end = min(begin + _BATCH_SIZE, len(cells))
            targets = (cells[begin:end] + offset).view(_CELL_DTYPE).reshape(-1)
            positions = np.searchsorted(keys, targets)
            within = positions < len(cells)
            source = np.arange(begin, end, dtype=np.int64)[within]
            target = positions[within]
            matches = keys[target] == targets[within]
            source, target = source[matches], target[matches]
            if not len(source):
                continue

            with np.errstate(over="ignore", invalid="ignore"):
                separation = np.maximum(
                    np.maximum(minimum[source] - maximum[target],
                               minimum[target] - maximum[source]), 0.0
                )
                possible = np.hypot(separation[:, 0], separation[:, 1]) <= tolerance
            source, target = source[possible], target[possible]
            if not len(source):
                continue

            with np.errstate(over="ignore", invalid="ignore"):
                span = np.maximum(
                    np.abs(minimum[source] - maximum[target]),
                    np.abs(maximum[source] - minimum[target]),
                )
                touching = np.hypot(span[:, 0], span[:, 1]) <= tolerance
            # Python iteration is restricted to ambiguous cell pairs; regular
            # grids, duplicate seams, and tightly clustered cells stay batched.
            for position in np.flatnonzero(~touching):
                left, right = source[position], target[position]
                touching[position] = _node_groups_touch(
                    xy,
                    order[starts[left] : starts[left] + counts[left]],
                    order[starts[right] : starts[right] + counts[right]],
                    tolerance,
                )
            if np.any(touching):
                left_pairs.append(source[touching])
                right_pairs.append(target[touching])

    if left_pairs:
        roots = _connected_components(
            len(cells), np.concatenate(left_pairs), np.concatenate(right_pairs)
        )
    else:
        roots = np.arange(len(cells), dtype=np.int64)
    node_roots = roots[inverse]
    owners = np.full(len(cells), len(nodes), dtype=np.int64)
    np.minimum.at(owners, node_roots, referenced_nodes)
    result[referenced_nodes] = owners[node_roots]
    return result


def _fallback_equivalent_node_mapping(
    nodes: NDArray[np.float64],
    referenced_nodes: NDArray[np.int64],
    tolerance: float,
) -> NDArray[np.int64]:
    """Map referenced XY-equivalent nodes to their lowest-index member."""
    old_to_representative = np.arange(nodes.shape[0], dtype=np.int64)
    if referenced_nodes.size < 2:
        return old_to_representative

    parent = np.arange(nodes.shape[0], dtype=np.int64)

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = int(parent[root])
        while parent[node] != node:
            next_node = int(parent[node])
            parent[node] = root
            node = next_node
        return root

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    xy = nodes[:, :2]
    if tolerance == 0.0:
        coordinate_owner: dict[tuple[float, float], int] = {}
        for node_index_value in referenced_nodes:
            node_index = int(node_index_value)
            key = (float(xy[node_index, 0]), float(xy[node_index, 1]))
            owner = coordinate_owner.setdefault(key, node_index)
            union(owner, node_index)
    else:
        # Half-tolerance square cells have a diagonal below the equivalence
        # distance, so all nodes in a normal cell form one connected group.
        cell_width = tolerance * 0.5
        tiny_cell_width = cell_width == 0.0
        if tiny_cell_width:
            cell_width = tolerance

        cells: dict[tuple[int, int], list[int]] = {}
        for node_index_value in referenced_nodes:
            node_index = int(node_index_value)
            key = (
                _coordinate_bin(float(xy[node_index, 0]), cell_width),
                _coordinate_bin(float(xy[node_index, 1]), cell_width),
            )
            cells.setdefault(key, []).append(node_index)

        for node_indices in cells.values():
            if tiny_cell_width:
                _union_close_node_groups(
                    xy,
                    node_indices,
                    node_indices,
                    tolerance,
                    union,
                    same_group=True,
                )
            else:
                anchor = node_indices[0]
                for node_index in node_indices[1:]:
                    union(anchor, node_index)

        neighbor_radius = 1 if tiny_cell_width else 3
        neighbor_offsets = tuple(
            (offset_x, offset_y)
            for offset_x in range(-neighbor_radius, neighbor_radius + 1)
            for offset_y in range(-neighbor_radius, neighbor_radius + 1)
            if offset_x != 0 or offset_y != 0
        )
        for key, node_indices in cells.items():
            for offset_x, offset_y in neighbor_offsets:
                neighbor_key = (key[0] + offset_x, key[1] + offset_y)
                if neighbor_key <= key or neighbor_key not in cells:
                    continue
                neighbor_indices = cells[neighbor_key]
                if tiny_cell_width:
                    _union_close_node_groups(
                        xy,
                        node_indices,
                        neighbor_indices,
                        tolerance,
                        union,
                    )
                elif _node_groups_touch(
                    xy,
                    node_indices,
                    neighbor_indices,
                    tolerance,
                ):
                    # Every cell is already internally connected, so one
                    # touching cross-cell pair joins both complete groups.
                    union(node_indices[0], neighbor_indices[0])

    for node_index_value in referenced_nodes:
        node_index = int(node_index_value)
        old_to_representative[node_index] = find(node_index)
    return old_to_representative


def _coordinate_bin(value: float, width: float) -> int:
    """Floor a finite float ratio without overflowing at extreme scales."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        quotient = np.float64(value) / np.float64(width)
    if np.isfinite(quotient) and abs(quotient) <= 2.0**52:
        return int(np.floor(quotient))

    value_numerator, value_denominator = value.as_integer_ratio()
    width_numerator, width_denominator = width.as_integer_ratio()
    return (value_numerator * width_denominator) // (
        value_denominator * width_numerator
    )


def _node_groups_touch(
    xy: NDArray[np.float64],
    left_indices: list[int],
    right_indices: list[int],
    tolerance: float,
) -> bool:
    """Return whether two spatial-cell groups contain an equivalent pair."""
    left = xy[left_indices]
    right = xy[right_indices]
    left_minimum = np.min(left, axis=0)
    left_maximum = np.max(left, axis=0)
    right_minimum = np.min(right, axis=0)
    right_maximum = np.max(right, axis=0)
    with np.errstate(over="ignore", invalid="ignore"):
        axis_separation = np.maximum(
            np.maximum(
                left_minimum - right_maximum,
                right_minimum - left_maximum,
            ),
            0.0,
        )
    if np.hypot(axis_separation[0], axis_separation[1]) > tolerance:
        return False

    block_size = 512
    for left_start in range(0, left.shape[0], block_size):
        left_block = left[left_start : left_start + block_size]
        for right_start in range(0, right.shape[0], block_size):
            right_block = right[right_start : right_start + block_size]
            with np.errstate(over="ignore", invalid="ignore"):
                difference = left_block[:, None, :] - right_block[None, :, :]
                distances = np.hypot(
                    difference[..., 0],
                    difference[..., 1],
                )
            if np.any(distances <= tolerance):
                return True
    return False


def _union_close_node_groups(
    xy: NDArray[np.float64],
    left_indices: list[int],
    right_indices: list[int],
    tolerance: float,
    union,
    *,
    same_group: bool = False,
) -> None:
    """Union exact close pairs for the subnormal cell-width fallback."""
    for left_position, left_index in enumerate(left_indices):
        right_start = left_position + 1 if same_group else 0
        for right_index in right_indices[right_start:]:
            with np.errstate(over="ignore", invalid="ignore"):
                difference = xy[left_index] - xy[right_index]
                distance = np.hypot(difference[0], difference[1])
            if distance <= tolerance:
                union(left_index, right_index)


