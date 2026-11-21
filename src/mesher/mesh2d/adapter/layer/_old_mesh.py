"""Array-based old-edge queries without a whole-mesh ownership table."""

import numpy as np


_EDGE_DTYPE = np.dtype([("a", np.int64), ("b", np.int64)])


def old_edge_keys(elements, data, *, boundary_only=False):
    """Cache canonical edge pairs and the subset owned by exactly one cell."""
    if "old_edge_keys" not in data:
        edges = np.stack((elements, np.roll(elements, -1, axis=1)), axis=-1)
        edges = np.sort(edges.reshape(-1, 2), axis=1)
        # A record view avoids Python tuples and retains lexicographic order.
        records = np.ascontiguousarray(edges, dtype=np.int64).view(_EDGE_DTYPE).ravel()
        unique, counts = np.unique(records, return_counts=True)
        keys = unique.view(np.int64).reshape(-1, 2)
        data["old_edge_keys"] = keys
        data["old_boundary_edge_keys"] = keys[counts == 1]
    return data["old_boundary_edge_keys" if boundary_only else "old_edge_keys"]


def edge_membership(keys, sorted_keys):
    """Find canonical edge pairs in a sorted unique inventory, without sets."""
    query = np.ascontiguousarray(keys, dtype=np.int64).reshape(-1, 2).view(_EDGE_DTYPE).ravel()
    inventory = sorted_keys.view(_EDGE_DTYPE).ravel()
    if not len(inventory):
        return np.zeros(len(query), dtype=bool)
    positions = np.searchsorted(inventory, query)
    return (positions < len(inventory)) & (inventory[np.minimum(positions, len(inventory) - 1)] == query)
