"""AABB queries preserve geometry while isolating distant mesh components."""

import unittest
from unittest.mock import patch

import numpy as np

from mesher.mesh2d.adapter.layer._geometry import BoxIndex


def _grid_boxes(n, kind, remote):
    y, x = np.mgrid[0:n + 1, 0:n + 1]
    nodes = np.column_stack((x.ravel() / n, y.ravel() / n))
    y, x = np.mgrid[0:n, 0:n]
    first = (y * (n + 1) + x).ravel()
    cells = np.column_stack((first, first + 1, first + n + 2, first + n + 1))
    if remote:
        start = len(nodes)
        nodes = np.vstack((nodes, [[1e6, 1e6], [1e6 + 1, 1e6],
                                  [1e6 + 1, 1e6 + 1], [1e6, 1e6 + 1]]))
        cells = np.vstack((cells, np.arange(start, start + 4)))
    if kind == "edges":
        pairs = np.stack((cells, np.roll(cells, -1, axis=1)), axis=2).reshape(-1, 2)
        cells = np.unique(np.sort(pairs, axis=1), axis=0)
    points = nodes[cells]
    return points.min(axis=1), points.max(axis=1)


def _perimeter_queries(n, padding):
    for k in range(n):
        a, b = k / n, (k + 1) / n
        for first, last in (([a, 0.], [b, 0.]), ([a, 1.], [b, 1.]),
                            ([0., a], [0., b]), ([1., a], [1., b])):
            yield np.minimum(first, last) - padding, np.maximum(first, last) + padding


def _subtree_ids(node):
    if node.ids is not None:
        return node.ids
    return np.concatenate((_subtree_ids(node.left), _subtree_ids(node.right)))


class LayerBoxIndexTests(unittest.TestCase):
    def assert_query(self, index, lower, upper, tol, lo, hi):
        expected = np.flatnonzero(np.all(upper + tol >= lo, axis=1)
                                  & np.all(lower - tol <= hi, axis=1))
        result = index.query(lo, hi)
        self.assertEqual(result.ndim, 1)
        self.assertTrue(np.issubdtype(result.dtype, np.integer))
        np.testing.assert_array_equal(result, expected)

    def test_empty_index_returns_an_integer_vector(self):
        bounds = np.empty((0, 2))
        index = BoxIndex(bounds, bounds)
        self.assert_query(index, bounds, bounds, 0., [-1., -1.], [1., 1.])

    def test_points_segments_long_boxes_and_duplicates_match_brute_force(self):
        lower = np.array([[0., 0.], [-2., 1.], [2., -3.], [-1e9, -.1],
                          [-.1, -1e9], [-1., -1.], [-1., -1.]])
        upper = np.array([[0., 0.], [3., 1.], [2., 4.], [1e9, .1],
                          [.1, 1e9], [1., 1.], [1., 1.]])
        # Exercise tree pruning as well as the single-leaf cases below.
        lower, upper = np.tile(lower, (11, 1)), np.tile(upper, (11, 1))
        for tol in (0., 1e-12, .125):
            index = BoxIndex(lower, upper, tol)
            for lo, hi in (([0., 0.], [0., 0.]), ([1., 1.], [1., 1.]),
                           ([1.5, -.2], [2., .2]), ([10., 10.], [11., 11.]),
                           ([-2e9, -2e9], [2e9, 2e9])):
                with self.subTest(tol=tol, lo=lo, hi=hi):
                    self.assert_query(index, lower, upper, tol, lo, hi)

    def test_tolerance_is_inclusive_and_applied_only_to_stored_bounds(self):
        lower, upper = np.array([[0., 0.]]), np.array([[1., 1.]])
        index = BoxIndex(lower, upper, .125)
        for point in ([1.125, .5], [-.125, .5], [.5, 1.125], [.5, -.125],
                      [np.nextafter(1.125, np.inf), .5],
                      [np.nextafter(-.125, -np.inf), .5]):
            with self.subTest(point=point):
                self.assert_query(index, lower, upper, .125, point, point)
        np.testing.assert_array_equal(index.query([1.125, .5], [1.125, .5]), [0])
        self.assertEqual(len(index.query([1.1251, .5], [1.1251, .5])), 0)

    def test_random_boxes_and_queries_match_brute_force(self):
        random = np.random.default_rng(275)
        centers = random.normal(size=(257, 2)) * 20.
        sizes = random.lognormal(mean=-1., sigma=2., size=(257, 2))
        sizes[::7, 0] = 0.
        sizes[::11, 1] = 0.
        lower, upper = centers - sizes, centers + sizes
        for tol in (0., 1e-12, .03):
            index = BoxIndex(lower, upper, tol)
            for _ in range(100):
                a, b = random.normal(size=(2, 2)) * 25.
                self.assert_query(index, lower, upper, tol, np.minimum(a, b), np.maximum(a, b))
            self.assert_query(index, lower, upper, tol, [-1e6, -1e6], [1e6, 1e6])

    def test_identical_centroids_split_by_id_and_remain_balanced(self):
        for count in (33, 65, 1000):
            with self.subTest(count=count):
                sizes = np.arange(1, count + 1, dtype=float)[:, None] * np.ones((1, 2))
                index = BoxIndex(-sizes, sizes)

                def inspect(node):
                    if node.ids is not None:
                        self.assertLessEqual(len(node.ids), 32)
                        self.assertTrue(np.all(np.diff(node.ids) > 0))
                        return len(node.ids)
                    left, right = inspect(node.left), inspect(node.right)
                    self.assertEqual(left, (left + right) // 2)
                    self.assertLessEqual(right - left, 1)
                    return left + right

                self.assertEqual(inspect(index._root), count)
                np.testing.assert_array_equal(_subtree_ids(index._root), np.arange(count))
                repeated = BoxIndex(-sizes, sizes)
                np.testing.assert_array_equal(_subtree_ids(repeated._root),
                                              _subtree_ids(index._root))
                self.assert_query(index, -sizes, sizes, 0., [0., 0.], [0., 0.])
                self.assert_query(index, -sizes, sizes, 0., [count, count], [count, count])

    def test_equal_centroid_axis_spans_choose_x(self):
        points = np.column_stack((np.tile([-1., 1.], 33),
                                  np.repeat([-1., 1., 0.], 22)))
        index = BoxIndex(points, points)
        np.testing.assert_array_equal(np.sort(_subtree_ids(index._root.left)), np.arange(0, 66, 2))
        np.testing.assert_array_equal(np.sort(_subtree_ids(index._root.right)), np.arange(1, 66, 2))

    def test_local_queries_do_not_scan_a_dense_component_due_to_one_remote_quad(self):
        tol = 1e-12
        totals = {}
        original = BoxIndex._filter_leaf
        for n in (40, 80):
            for kind in ("cells", "edges"):
                for remote in (False, True):
                    with self.subTest(n=n, kind=kind, remote=remote):
                        lower, upper = _grid_boxes(n, kind, remote)
                        index = BoxIndex(lower, upper, tol)
                        scanned = 0
                        counts = []

                        def counted(candidate, ids, lo, hi):
                            nonlocal scanned
                            self.assertIs(candidate, index)
                            scanned += len(ids)
                            return original(candidate, ids, lo, hi)

                        with patch.object(BoxIndex, "_filter_leaf", counted):
                            for lo, hi in _perimeter_queries(n, tol):
                                scanned = 0
                                self.assert_query(index, lower, upper, tol, lo, hi)
                                counts.append(scanned)
                        self.assertEqual(len(counts), 4 * n)
                        self.assertLessEqual(sum(counts), 96 * len(counts))
                        self.assertLessEqual(max(counts), 128)
                        totals[n, kind, remote] = sum(counts)
                self.assertLessEqual(totals[n, kind, True], 2 * totals[n, kind, False])
        for kind in ("cells", "edges"):
            self.assertLessEqual(totals[80, kind, True], 2.5 * totals[40, kind, True])


if __name__ == "__main__":
    unittest.main()
