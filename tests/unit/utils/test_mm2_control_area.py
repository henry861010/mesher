"""Geometric acceptance cases for local die-corner/gap control regions."""

import itertools
import unittest
from copy import deepcopy

import numpy as np

from mesher.utils import find_adapter_group, get_mm2_control_area


def product_boxes(xs, ys):
    return np.array(sorted([x1, y1, x2, y2]
                           for x1, x2 in xs for y1, y2 in ys), dtype=float)


class ControlAreaTests(unittest.TestCase):
    def assert_region(self, actual, expected):
        """Compare exact coverage on boundary-induced cells, not decomposition."""
        expected = np.asarray(expected, dtype=float).reshape(-1, 4)
        self.assertEqual(actual.dtype, np.float64)
        self.assertEqual(actual.ndim, 2)
        self.assertEqual(actual.shape[1], 4)
        self.assertEqual(actual.tolist(), sorted(actual.tolist()))
        self.assertTrue(np.all(actual[:, 2:] > actual[:, :2]))
        all_boxes = np.concatenate([actual, expected])
        xs = np.unique(all_boxes[:, [0, 2]])
        ys = np.unique(all_boxes[:, [1, 3]])
        for x1, x2 in zip(xs[:-1], xs[1:]):
            for y1, y2 in zip(ys[:-1], ys[1:]):
                x, y = (x1 + x2) / 2, (y1 + y2) / 2
                def count(boxes):
                    return np.count_nonzero((boxes[:, 0] < x) & (x < boxes[:, 2])
                                            & (boxes[:, 1] < y) & (y < boxes[:, 3]))
                self.assertLessEqual(count(actual), 1, (x, y))
                self.assertEqual(count(actual) > 0, count(expected) > 0, (x, y))
        actual_area = np.prod(actual[:, 2:] - actual[:, :2], axis=1).sum()
        # Expected fixtures in these cases have disjoint interiors.
        expected_area = np.prod(expected[:, 2:] - expected[:, :2], axis=1).sum()
        self.assertAlmostEqual(actual_area, expected_area)

    def test_single_die(self):
        expected = product_boxes([(-25, 25), (75, 125)], [(-25, 25), (75, 125)])
        np.testing.assert_array_equal(get_mm2_control_area([[0, 0, 100, 100]]), expected)

    def test_aligned_gap_is_filled_in_both_orientations(self):
        dies = np.array([[0, 0, 100, 100], [180, 0, 280, 100]])
        expected = product_boxes([(-25, 25), (75, 205), (255, 305)],
                                 [(-25, 25), (75, 125)])
        for transpose in (False, True):
            order = [1, 0, 3, 2] if transpose else [0, 1, 2, 3]
            with self.subTest(transpose=transpose):
                self.assert_region(get_mm2_control_area(dies[:, order]), expected[:, order])

    def test_max_gap_includes_threshold_in_both_orientations(self):
        dies = np.array([[0, 0, 100, 100], [180, 0, 280, 100]])
        for order in ([0, 1, 2, 3], [1, 0, 3, 2]):
            oriented = dies[:, order]
            separate = np.concatenate([get_mm2_control_area([die]) for die in oriented])
            self.assert_region(get_mm2_control_area(oriented, max_gap=79), separate)
            self.assert_region(get_mm2_control_area(oriented, max_gap=80),
                               get_mm2_control_area(oriented))

    def test_invalid_max_gap(self):
        for limit in (-1, np.nan, np.inf, [25]):
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, "max_gap"):
                get_mm2_control_area([], max_gap=limit)

    def test_four_dies_include_entire_crossed_gap(self):
        dies = product_boxes([(0, 100), (180, 280)], [(0, 100), (180, 280)])
        bands = [(-25, 25), (75, 205), (255, 305)]
        self.assert_region(get_mm2_control_area(dies), product_boxes(bands, bands))

    def test_t_junction_and_input_order(self):
        dies = [[0, 0, 100, 100], [0, 180, 100, 280], [180, 0, 280, 280]]
        bands = [(-25, 25), (75, 205), (255, 305)]
        expected = np.concatenate([product_boxes(bands[:2], bands),
                                   product_boxes(bands[2:], [bands[0], bands[2]])])
        reference = get_mm2_control_area(dies)
        self.assert_region(reference, expected)
        for permutation in itertools.permutations(dies):
            np.testing.assert_array_equal(get_mm2_control_area(permutation), reference)

    def test_diagonally_separated_dies_do_not_create_remote_intersections(self):
        first = product_boxes([(-25, 25), (75, 125)], [(-25, 25), (75, 125)])
        self.assert_region(get_mm2_control_area([[0, 0, 100, 100], [300, 300, 400, 400]]),
                           np.concatenate([first, first + 300]))

    def test_partial_blocker_limits_visible_gap(self):
        dies = [[0, 0, 100, 100], [200, 0, 300, 100], [140, 0, 160, 60]]
        expected = np.concatenate([
            product_boxes([(-5, 5), (295, 305)], [(-5, 5), (95, 105)]),
            product_boxes([(95, 145), (155, 205)], [(-5, 5), (55, 65)]),
            [[95, 95, 205, 105]],
        ])
        self.assert_region(get_mm2_control_area(dies, corner_extent=5), expected)

    def test_remote_gap_does_not_expand_an_unrelated_corner(self):
        dies = [[0, 0, 100, 100], [-200, 1000, -100, 1100], [200, 1000, 300, 1100]]
        expected = np.concatenate([
            product_boxes([(-25, 25), (75, 125)], [(-25, 25), (75, 125)]),
            product_boxes([(-225, -175), (-125, 225), (275, 325)],
                          [(975, 1025), (1075, 1125)]),
        ])
        self.assert_region(get_mm2_control_area(dies), expected)

    def test_touching_dies_and_point_only_projection_do_not_form_gaps(self):
        for second in ([100, 0, 200, 100], [100, 100, 200, 200], [180, 100, 280, 200]):
            with self.subTest(second=second):
                # Corner patches are far enough apart to be disjoint except
                # identical patches at shared corners, which are deduplicated.
                corners = {(x, y) for b in ([0, 0, 100, 100], second)
                           for x in (b[0], b[2]) for y in (b[1], b[3])}
                expected = [[x - 5, y - 5, x + 5, y + 5] for x, y in corners]
                self.assert_region(get_mm2_control_area([[0, 0, 100, 100], second],
                                                        corner_extent=5), expected)

    def test_small_die_corner_patches_merge(self):
        self.assert_region(get_mm2_control_area([[0, 0, 20, 20]]), [[-25, -25, 45, 45]])

    def test_union_preserves_notches_in_overlapping_corner_patches(self):
        # Diagonal dies have no gap. Two of their corner squares overlap,
        # creating a stepped union rather than its rectangular bounding box.
        expected = [
            [-25, -25, 25, 25], [-25, 75, 25, 125], [75, -25, 125, 25],
            [75, 75, 95, 125], [95, 75, 125, 145], [125, 95, 145, 145],
            [95, 195, 145, 245], [195, 95, 245, 145], [195, 195, 245, 245],
        ]
        self.assert_region(get_mm2_control_area([[0, 0, 100, 100], [120, 120, 220, 220]]),
                           expected)

    def test_custom_extent_translation_and_independent_storage(self):
        dies = np.array([[0.25, -0.75, 100.25, 99.25], [180.25, -0.75, 280.25, 99.25]])
        dies.setflags(write=False)
        before = dies.copy()
        result = get_mm2_control_area(dies, corner_extent=2.5)
        expected = product_boxes([(-2.5, 2.5), (97.5, 182.5), (277.5, 282.5)],
                                 [(-2.5, 2.5), (97.5, 102.5)])
        self.assert_region(result, expected + [0.25, -0.75, 0.25, -0.75])
        np.testing.assert_array_equal(dies, before)
        self.assertFalse(np.shares_memory(result, dies))
        result[:] = 0
        np.testing.assert_array_equal(dies, before)

    def test_empty(self):
        for empty in ([], np.empty((0, 4))):
            self.assert_region(get_mm2_control_area(empty), [])

    def test_invalid_boxes(self):
        for boxes in (None, [0, 0, 1, 1], [[0, 0, 1]], np.empty((0, 3)),
                      [[[0, 0, 1, 1]]], [[0, 0, 1, 1], [0, 0]],
                      [[0, 0, np.nan, 1]], [[0, 0, np.inf, 1]],
                      [[0, 0, 0, 1]], [[1, 0, 0, 1]], [[0, 1, 1, 0]],
                      [[0, 0, 10, 10], [1, 1, 2, 2]],
                      [[0, 0, 10, 10], [5, 5, 20, 20]],
                      [[0, 0, 1, 1], [0, 0, 1, 1]]):
            with self.subTest(boxes=boxes), self.assertRaises(ValueError):
                get_mm2_control_area(boxes)

    def test_invalid_extent(self):
        for extent in (0, -1, np.nan, np.inf, -np.inf, None, [25], "bad", 1 + 1j):
            with self.subTest(extent=extent), self.assertRaisesRegex(ValueError, "corner_extent"):
                get_mm2_control_area([], corner_extent=extent)

    def test_unrepresentable_expansion(self):
        for box, extent in (([0, 0, 1e308, 1e308], 1e308),
                            ([1e16, 1e16, 1e16 + 10, 1e16 + 10], .1)):
            with self.subTest(extent=extent), self.assertRaisesRegex(ValueError, "represented"):
                get_mm2_control_area([box], corner_extent=extent)


class FindAdapterGroupTests(unittest.TestCase):
    def test_default_groups_and_remaining_patterns(self):
        horizontal = [[[10, y], [0, y]] for y in (2, 0, 1)]
        vertical = [[[x, 0], [x, 10]] for x in (4, 6, 5)]
        short = [[2, 1], [10, 1]]
        pair = [[[10, y], [0, y]] for y in (6, 5)]
        lines = [pair[0], *horizontal, pair[0][::-1], short, *vertical,
                 horizontal[0][::-1], pair[1], short]
        before = deepcopy(lines)
        groups, remaining = find_adapter_group(3, lines)
        self.assertEqual(groups, [
            {"box": [0, 0, 10, 2], "x_points": [[0, 0], [0, 1], [0, 2]]},
            {"box": [4, 0, 6, 10], "y_points": [[4, 0], [5, 0], [6, 0]]},
        ])
        self.assertEqual(remaining, [pair[0], pair[0][::-1], short, pair[1], short])
        self.assertEqual(lines, before)

    def test_gap_equal_to_threshold_separates_groups(self):
        lines = [[[10, y], [0, y]] for y in (4, 0, 5, 1)]
        self.assertEqual(find_adapter_group(3, lines, min_group_count=2), ([
            {"box": [0, 0, 10, 1], "x_points": [[0, 0], [0, 1]]},
            {"box": [0, 4, 10, 5], "x_points": [[0, 4], [0, 5]]},
        ], []))

    def test_custom_minimum_counts_distinct_positions(self):
        lines = [[[x, 0], [x, 10]] for x in (0, 1, 2, 2)]
        self.assertEqual(find_adapter_group(2, lines, min_group_count=4), ([], lines))
        lines.append([[3, 10], [3, 0]])
        self.assertEqual(find_adapter_group(2, iter(lines), min_group_count=4), ([
            {"box": [0, 0, 3, 10], "y_points": [[0, 0], [1, 0], [2, 0], [3, 0]]},
        ], []))

    def test_empty(self):
        self.assertEqual(find_adapter_group(1, []), ([], []))

    def test_invalid_thresholds(self):
        for count in (0, 1, True, 3.0, "3"):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "min_group_count"):
                find_adapter_group(1, [], min_group_count=count)
        for size in (0, -1, np.nan, np.inf):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "group_size"):
                find_adapter_group(size, [])

    def test_invalid_segments(self):
        for line in ([[0, 0], [1, 1]], [[0, 0], [0, 0]], [[0, 0], [0, np.inf]], [[0, 0]]):
            with self.subTest(line=line), self.assertRaisesRegex(ValueError, r"lines\[0\]"):
                find_adapter_group(1, [line])


if __name__ == "__main__":
    unittest.main()
