"""Array-based old-edge queries without a whole-mesh ownership table."""

import numpy as np


_EDGE_DTYPE = np.dtype([("a", np.int64), ("b", np.int64)])


def old_edge_keys(elements, data):
    """Lazily cache sorted unique undirected edges for one prepared mesh."""
    if "old_edge_keys" not in data:
        edges = np.stack((elements, np.roll(elements, -1, axis=1)), axis=-1)
        edges = np.sort(edges.reshape(-1, 2), axis=1)
        # A record view avoids Python tuples and retains lexicographic order.
        records = np.ascontiguousarray(edges, dtype=np.int64).view(_EDGE_DTYPE).ravel()
        data["old_edge_keys"] = np.unique(records).view(np.int64).reshape(-1, 2)
    return data["old_edge_keys"]


def edge_membership(keys, sorted_keys):
    """Find canonical edge pairs in a sorted unique inventory, without sets."""
    query = np.ascontiguousarray(keys, dtype=np.int64).reshape(-1, 2).view(_EDGE_DTYPE).ravel()
    inventory = sorted_keys.view(_EDGE_DTYPE).ravel()
    if not len(inventory):
        return np.zeros(len(query), dtype=bool)
    positions = np.searchsorted(inventory, query)
    return (positions < len(inventory)) & (inventory[np.minimum(positions, len(inventory) - 1)] == query)


def edge_owner_count(key, elements, xy, cell_index, tol):
    """Count only the owners of one shared target edge in nearby old cells."""
    points = xy[np.asarray(key)]
    candidates = cell_index.query(points.min(axis=0) - tol, points.max(axis=0) + tol)
    cells = elements[candidates]
    following = np.roll(cells, -1, axis=1)
    a, b = key
    return int(np.count_nonzero(((cells == a) & (following == b))
                                | ((cells == b) & (following == a))))
