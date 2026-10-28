import unittest

import numpy as np

from mesher.mesh2d.utils.equivalence import _equivalent_node_mapping


class EquivalentNodeMappingTests(unittest.TestCase):
    def assert_matches_brute_force(self, xy, referenced, tolerance):
        nodes = np.column_stack((xy, np.zeros(len(xy))))
        parent = np.arange(len(xy))

        def root(node):
            while parent[node] != node:
                node = parent[node]
            return node

        for position, left in enumerate(referenced):
            for right in referenced[position + 1:]:
                with np.errstate(over="ignore", invalid="ignore"):
                    delta = xy[left] - xy[right]
                    distance = np.hypot(*delta)
                if distance <= tolerance:
                    a, b = root(left), root(right)
                    parent[max(a, b)] = min(a, b)
        expected = np.array([root(node) for node in range(len(xy))])
        actual = _equivalent_node_mapping(nodes, referenced, tolerance)
        np.testing.assert_array_equal(actual, expected)

    def test_randomized_components_match_independent_pairwise_reference(self):
        for seed in range(8):
            random = np.random.default_rng(seed)
            xy = random.uniform(-1, 1, (64, 2))
            xy[20:25] = xy[:5]
            referenced = random.permutation(64)[:50]
            for tolerance in (0, 0.05, 0.25, 0.75):
                with self.subTest(seed=seed, tolerance=tolerance):
                    self.assert_matches_brute_force(xy, referenced, tolerance)

    def test_ambiguous_cells_use_actual_members_instead_of_only_representatives(self):
        xy = np.array([[0, 0], [0.49, 0.49], [1.1, 0.49], [3, 3], [3.49, 3.49]])
        self.assert_matches_brute_force(xy, np.arange(5), 1)

    def test_extreme_scales_and_subnormal_tolerance(self):
        tiny = np.nextafter(0.0, 1.0)
        for xy, tolerance in (
            (np.array([[0, 0], [tiny, 0], [tiny * 2, 0], [1, 1]]), tiny),
            (np.array([[1e100, 0], [1e100, 0], [-1e100, 0]]), 1e-200),
            (np.zeros((200, 2)), tiny),
            (np.array([[-1e308, 0], [0, 0], [1e308, 0]]), 1e308),
        ):
            self.assert_matches_brute_force(xy, np.arange(len(xy)), tolerance)

