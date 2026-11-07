"""Old-edge queries retain topology without building global owner lists."""

import unittest
from collections import Counter

import numpy as np

from mesher.mesh2d.adapter.layer._geometry import BoxIndex
from mesher.mesh2d.adapter.layer._old_mesh import edge_membership, edge_owner_count, old_edge_keys


class OldMeshEdgeTests(unittest.TestCase):
    def test_inventory_and_membership_match_independent_edge_counts(self):
        # Include shared edges, duplicate cells, either cell winding and IDs
        # too large to safely encode a pair by multiplying the node count.
        cells = np.array([[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 1, 0]], dtype=np.int64)
        for offset in (0, 2**62):
            elements = cells + offset
            counts = Counter(tuple(sorted((int(cell[i]), int(cell[(i + 1) % 4]))))
                             for cell in elements for i in range(4))
            data = {}
            keys = old_edge_keys(elements, data)
            np.testing.assert_array_equal(keys, sorted(counts))
            self.assertIs(old_edge_keys(elements, data), keys)
            query = np.array([[0, 1], [0, 4], [4, 5], [5, 6]], dtype=np.int64) + offset
            np.testing.assert_array_equal(edge_membership(query, keys),
                                          [tuple(edge) in counts for edge in query])
        empty = old_edge_keys(np.empty((0, 4), dtype=np.int32), {})
        self.assertEqual(empty.shape, (0, 2))
        self.assertFalse(edge_membership([[0, 1]], empty)[0])

    def test_target_query_counts_edges_not_diagonals_or_nearby_coordinates(self):
        xy = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 0], [2, 1],
                       [0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        elements = np.array([[0, 1, 2, 3], [2, 5, 4, 1], [6, 7, 8, 9]])
        cells = xy[elements]
        index = BoxIndex(cells.min(axis=1), cells.max(axis=1), 1e-10)
        for key, expected in (((0, 1), 1), ((1, 2), 2), ((0, 2), 0), ((6, 7), 1)):
            with self.subTest(key=key):
                self.assertEqual(edge_owner_count(key, elements, xy, index, 1e-10), expected)


if __name__ == "__main__":
    unittest.main()
