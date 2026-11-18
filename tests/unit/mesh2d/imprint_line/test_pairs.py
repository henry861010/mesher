"""Box controls, outward distances, and public mesher interoperability."""
import copy
import unittest

import numpy as np

from mesher.mesh2d.imprint_line import get_pair_line, imprint_lines


OUTER = [[0, 0], [20, 0], [20, 20], [0, 20]]
NOTCHED = [[0, 0], [20, 0], [20, 20], [0, 20],
           [0, 8], [6, 8], [6, 6], [0, 6]]


def expansions(pairs):
    return [p[key] for p in pairs for key in ("line1_expand_size", "line2_expand_size")]


class GetPairLineTests(unittest.TestCase):
    def test_pair_geometry_and_input_order(self):
        pairs = get_pair_line([[2, 3, 8, 7], [1, 1, 3, 3]], 1.5, spacing=2)
        self.assertEqual([p["direction"] for p in pairs], ["x", "y", "x", "y"])
        self.assertEqual(pairs[0], dict(
            direction="x", line1=[[2, 3], [2, 7]], line2=[[8, 3], [8, 7]],
            points=[[2, 3], [2, 5], [2, 7]], line1_expand_size=1.5, line2_expand_size=1.5))
        self.assertEqual(pairs[1], dict(
            direction="y", line1=[[2, 3], [8, 3]], line2=[[2, 7], [8, 7]],
            points=[[2, 3], [4, 3], [6, 3], [8, 3]],
            line1_expand_size=1.5, line2_expand_size=1.5))
        self.assertEqual(pairs[2]["line1"], [[1, 1], [1, 3]])
        self.assertEqual(expansions(pairs), [1.5] * 8)

    def test_fixed_spacing_and_terminal_remainder(self):
        for step, expected in ((3, [0, 3, 6, 9, 10]), (2, [0, 2, 4, 6, 8, 10]),
                               (11, [0, 10]), (10, [0, 10])):
            with self.subTest(step=step):
                pairs = get_pair_line([[0, 0, 10, 10]], 0, spacing=step)
                self.assertEqual([p[1] for p in pairs[0]["points"]], expected)
                self.assertEqual([p[0] for p in pairs[1]["points"]], expected)

    def test_floating_endpoint_is_included_once(self):
        for low, high, step, expected in (
                (0, .3, .1, [0, .1, .2, .3]),
                (-.3, 0, .1, [-.3, -.2, -.1, 0]),
                (.1, .4, .1, [.1, .2, .3, .4]),
                (0, .30001, .1, [0, .1, .2, .3, .30001])):
            with self.subTest(low=low, high=high):
                pair = get_pair_line([[low, low, high, high]], 0, spacing=step)[0]
                values = [p[1] for p in pair["points"]]
                np.testing.assert_allclose(values, expected, atol=1e-16, rtol=0)
                self.assertEqual(values[-1], high)
                self.assertEqual(values.count(high), 1)

    def test_threshold_is_strict_and_replacement_can_increase(self):
        # Distances left/right/bottom/top are 2, 14, 3, 12.
        for threshold, expected in ((0, [.5] * 4), (2, [.5] * 4),
                                    (3, [2, .5, .5, .5]), (4, [2, .5, 3, .5]),
                                    (15, [2, 14, 3, 12])):
            with self.subTest(threshold=threshold):
                pairs = get_pair_line([[2, 3, 6, 8]], .5, OUTER, threshold, spacing=1)
                self.assertEqual(expansions(pairs), expected)

    def test_all_corners_and_touching_sides(self):
        for box, expected in (([1, 1, 3, 3], [1, 5, 1, 5]),
                              ([17, 1, 19, 3], [5, 1, 1, 5]),
                              ([1, 17, 3, 19], [1, 5, 5, 1]),
                              ([17, 17, 19, 19], [5, 1, 5, 1]),
                              ([0, 0, 20, 20], [0, 0, 0, 0])):
            with self.subTest(box=box):
                pairs = get_pair_line([box], 5, OUTER, 2, spacing=1)
                self.assertEqual(expansions(pairs), expected)

    def test_near_bottom_does_not_shorten_left_or_right(self):
        pairs = get_pair_line([[8, .5, 12, 2]], 3, OUTER, 1, spacing=.5)
        self.assertEqual(expansions(pairs), [3, 3, .5, 3])

    def test_concave_outline_checks_entire_span_in_both_axes(self):
        # The notch meets only y=6..8 of the side; its midpoint y=10 misses it.
        box = [8, 5, 10, 15]
        for transpose in (False, True):
            outer = np.array(NOTCHED)
            bounds = np.array([box])
            expected = [2, 10, 5, 5]
            if transpose:
                outer = outer[:, ::-1]
                bounds = bounds[:, [1, 0, 3, 2]]
                expected = [5, 5, 2, 10]
            with self.subTest(transpose=transpose):
                pairs = get_pair_line(bounds, 1, outer, 30, spacing=1)
                self.assertEqual(expansions(pairs), expected)

    def test_notch_touching_only_endpoint_is_still_first_contact(self):
        pairs = get_pair_line([[8, 8, 10, 15]], 1, NOTCHED, 30, spacing=1)
        self.assertEqual(expansions(pairs), [2, 10, 8, 5])

    def test_outline_winding_and_closure(self):
        expected = get_pair_line([[8, 5, 10, 15]], 1, NOTCHED, 30, spacing=1)
        for outer in (NOTCHED[::-1], NOTCHED + [NOTCHED[0]],
                      NOTCHED[::-1] + [NOTCHED[-1]]):
            self.assertEqual(get_pair_line([[8, 5, 10, 15]], 1, outer, 30, spacing=1), expected)

    def test_inputs_and_results_are_independent(self):
        boxes = np.array([[1., 1., 3., 3.], [1., 1., 3., 3.]])
        outline = np.array(OUTER, dtype=float)
        original_boxes, original_outline = boxes.copy(), outline.copy()
        first = get_pair_line(boxes, 1, outline, 2, spacing=.5)
        second = copy.deepcopy(first)
        first[0]["points"][0][0] = 999
        first[0]["line1"][0][0] = 999
        self.assertEqual(first[1:], second[1:])
        self.assertEqual(get_pair_line(boxes, 1, outline, 2, spacing=.5), second)
        np.testing.assert_array_equal(boxes, original_boxes)
        np.testing.assert_array_equal(outline, original_outline)

    def test_empty_and_iterable_boxes(self):
        for boxes in ([], np.empty((0, 4)), iter(())):
            self.assertEqual(get_pair_line(boxes, 1, spacing=1), [])
        self.assertEqual(get_pair_line([], 1, OUTER, 2, spacing=1), [])
        self.assertEqual(len(get_pair_line(iter([[0, 0, 1, 1]]), 1, spacing=1)), 2)

    def test_invalid_boxes(self):
        for boxes in (None, np.empty((0, 3)), [0, 0, 1, 1], [[0, 0, 1]], [[0, 0, 1, 1, 2]],
                      [[1, 0, 0, 1]], [[0, 1, 1, 0]], [[0, 0, 0, 1]],
                      [[0, 0, 1, 0]], [[0, 0, np.nan, 1]], [[0, 0, np.inf, 1]],
                      [[-1e308, 0, 1e308, 1]], [[0, 0, 1, 1], [0, 0, 1]]):
            with self.subTest(boxes=boxes), self.assertRaises(ValueError):
                get_pair_line(boxes, 1, spacing=1)

    def test_invalid_scalar_parameters(self):
        for name, values in (("expand_size", [-1, np.inf, np.nan, True, [1]]),
                             ("spacing", [0, -1, np.inf, np.nan, True, [1]]),
                             ("distance", [-1, np.inf, np.nan, True, [1]])):
            for value in values:
                kwargs = dict(expand_size=1, spacing=1, outer_outline=OUTER, distance=2)
                kwargs[name] = value
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    get_pair_line([[1, 1, 2, 2]], **kwargs)

    def test_missing_parameters(self):
        with self.assertRaises(TypeError):
            get_pair_line([[0, 0, 1, 1]], 1)
        for kwargs in (dict(outer_outline=OUTER), dict(distance=2)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                get_pair_line([[0, 0, 1, 1]], 1, spacing=1, **kwargs)

    def test_invalid_outline_and_outside_box(self):
        for outer in ([], [[0, 0], [20, 0], [19, 20], [0, 20]],
                      [[0, 0], [20, 20], [20, 0], [0, 20]]):
            with self.subTest(outer=outer), self.assertRaises(ValueError):
                get_pair_line([[1, 1, 2, 2]], 1, outer, 2, spacing=1)
        for box in ([-1, 2, 2, 4], [21, 2, 23, 4], [2, 19, 4, 21], [4, 5, 10, 10]):
            with self.subTest(box=box), self.assertRaisesRegex(ValueError, r"boxes\[0\]"):
                get_pair_line([box], 1, NOTCHED, 2, spacing=1)

    def test_public_mesher_preserves_sampled_cross_sections(self):
        outer = [[0, 0], [12, 0], [12, 12], [0, 12]]
        pairs = get_pair_line([[1, 4, 3, 6]], .5, outer, 2, spacing=.6)
        mesh = imprint_lines(2, pairs, outer_outline=outer,
                             min_transition_size=.5, pattern_snap_distance=.2)
        self.assertGreater(mesh.element_count, 0)
        self.assertTrue(np.all(mesh.nodes[:, 2] == 0))
        elements = mesh.elements
        edges = np.stack((elements, np.roll(elements, -1, axis=1)), axis=-1).reshape(-1, 2)
        xy = mesh.nodes[edges, :2]
        for pair in pairs:
            axis = 0 if pair["direction"] == "x" else 1
            start, end = pair["line1"][0][axis], pair["line2"][0][axis]
            for point in pair["points"]:
                on = np.all(np.isclose(xy[:, :, 1 - axis], point[1 - axis], atol=1e-12, rtol=0), axis=1)
                spans = np.sort(xy[on, :, axis], axis=1)
                cursor = start
                for a, b in sorted(map(tuple, spans)):
                    if b <= cursor or a >= end:
                        continue
                    self.assertLessEqual(a, cursor + 1e-12)
                    cursor = max(cursor, b)
                self.assertGreaterEqual(cursor, end)


if __name__ == "__main__":
    unittest.main()
