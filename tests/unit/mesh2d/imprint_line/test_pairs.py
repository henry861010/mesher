"""Box controls, outward distances, and public mesher interoperability."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from mesher.mesh2d.imprint_line import BoxPairSpec, get_pair_line, imprint_lines


OUTER = [[0, 0], [20, 0], [20, 20], [0, 20]]
NOTCHED = [[0, 0], [20, 0], [20, 20], [0, 20],
           [0, 8], [6, 8], [6, 6], [0, 6]]


def both(box):
    """Enable both pairs for tests concerned only with boundary geometry."""
    return [{"box": box, "x_points": [], "y_points": []}]


def expansions(pairs):
    return [p[key] for p in pairs for key in ("line1_expand_size", "line2_expand_size")]


class GetPairLineTests(unittest.TestCase):
    def test_pair_geometry_and_input_order(self):
        specs = [
            BoxPairSpec(box=[2, 3, 8, 7], y_points=[[2, 3], [4, 3], [8, 3]],
                        x_points=[[2, 3], [2, 5], [2, 7]]),
            {"box": [1, 1, 3, 3], "y_points": [[2, 1]]},
            {"box": [4, 4, 6, 6]},
            {"box": [7, 7, 9, 9], "x_points": [[7, 8]]},
        ]
        pairs = get_pair_line(specs, 1.5)
        self.assertEqual([p["direction"] for p in pairs], ["x", "y", "y", "x"])
        self.assertEqual(pairs[0], dict(
            direction="x", line1=[[2, 3], [2, 7]], line2=[[8, 3], [8, 7]],
            points=[[2, 3], [2, 5], [2, 7]], line1_expand_size=1.5, line2_expand_size=1.5))
        self.assertEqual(pairs[1], dict(
            direction="y", line1=[[2, 3], [8, 3]], line2=[[2, 7], [8, 7]],
            points=[[2, 3], [4, 3], [8, 3]],
            line1_expand_size=1.5, line2_expand_size=1.5))
        self.assertEqual(pairs[2]["line1"], [[1, 1], [3, 1]])
        self.assertEqual(pairs[3]["line1"], [[7, 7], [7, 9]])
        self.assertEqual(expansions(pairs), [1.5] * 8)

    def test_nonuniform_points_are_preserved_without_endpoints_or_reordering(self):
        specs = [{"box": [0, 0, 50, 50],
                  "x_points": [[100, y] for y in (30, 5, 10, 30)],
                  "y_points": [[x, -100] for x in (5, 25, 45)]}]
        pairs = get_pair_line(specs, 0)
        self.assertEqual(pairs[0]["points"], specs[0]["x_points"])
        self.assertEqual(pairs[1]["points"], specs[0]["y_points"])

    def test_empty_points_enable_pairs_but_omitted_fields_do_not(self):
        pairs = get_pair_line([
            {"box": [0, 0, 1, 1]},
            {"box": [0, 0, 1, 1], "x_points": []},
            {"box": [0, 0, 1, 1], "y_points": np.empty((0, 2))},
        ], 0)
        self.assertEqual([p["direction"] for p in pairs], ["x", "y"])
        self.assertEqual([p["points"] for p in pairs], [[], []])

    def test_only_enabled_directions_compute_outline_distances(self):
        for field, axis, expected in (("x_points", 0, [2, 14]),
                                      ("y_points", 1, [3, 12])):
            with self.subTest(field=field), patch(
                    "mesher.mesh2d.imprint_line._pairs._distances",
                    return_value=expected) as distances:
                pairs = get_pair_line([{"box": [2, 3, 6, 8], field: []}], .5, OUTER, 15)
                distances.assert_called_once()
                self.assertEqual(distances.call_args.args[2], axis)
                self.assertEqual(expansions(pairs), expected)

    def test_threshold_is_strict_and_replacement_can_increase(self):
        # Distances left/right/bottom/top are 2, 14, 3, 12.
        for threshold, expected in ((0, [.5] * 4), (2, [.5] * 4),
                                    (3, [2, .5, .5, .5]), (4, [2, .5, 3, .5]),
                                    (15, [2, 14, 3, 12])):
            with self.subTest(threshold=threshold):
                pairs = get_pair_line(both([2, 3, 6, 8]), .5, OUTER, threshold)
                self.assertEqual(expansions(pairs), expected)

    def test_all_corners_and_touching_sides(self):
        for box, expected in (([1, 1, 3, 3], [1, 5, 1, 5]),
                              ([17, 1, 19, 3], [5, 1, 1, 5]),
                              ([1, 17, 3, 19], [1, 5, 5, 1]),
                              ([17, 17, 19, 19], [5, 1, 5, 1]),
                              ([0, 0, 20, 20], [0, 0, 0, 0])):
            with self.subTest(box=box):
                pairs = get_pair_line(both(box), 5, OUTER, 2)
                self.assertEqual(expansions(pairs), expected)

    def test_near_bottom_does_not_shorten_left_or_right(self):
        pairs = get_pair_line(both([8, .5, 12, 2]), 3, OUTER, 1)
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
                pairs = get_pair_line(both(bounds[0]), 1, outer, 30)
                self.assertEqual(expansions(pairs), expected)

    def test_notch_touching_only_endpoint_is_still_first_contact(self):
        pairs = get_pair_line(both([8, 8, 10, 15]), 1, NOTCHED, 30)
        self.assertEqual(expansions(pairs), [2, 10, 8, 5])

    def test_outline_winding_and_closure(self):
        expected = get_pair_line(both([8, 5, 10, 15]), 1, NOTCHED, 30)
        for outer in (NOTCHED[::-1], NOTCHED + [NOTCHED[0]],
                      NOTCHED[::-1] + [NOTCHED[-1]]):
            self.assertEqual(get_pair_line(both([8, 5, 10, 15]), 1, outer, 30), expected)

    def test_inputs_and_results_are_independent(self):
        box = np.array([1., 1., 3., 3.])
        points = np.array([[1., 1.], [2., 2.], [3., 3.]])
        spec = dict(box=box, x_points=points, y_points=points)
        specs = [spec, spec]
        outline = np.array(OUTER, dtype=float)
        original_box, original_points, original_outline = box.copy(), points.copy(), outline.copy()
        first = get_pair_line(specs, 1, outline, 2)
        second = copy.deepcopy(first)
        first[0]["points"][0][0] = 999
        first[0]["line1"][0][0] = 999
        self.assertEqual(first[0]["line2"], second[0]["line2"])
        self.assertEqual(first[1:], second[1:])
        self.assertEqual(get_pair_line(specs, 1, outline, 2), second)
        np.testing.assert_array_equal(box, original_box)
        np.testing.assert_array_equal(points, original_points)
        np.testing.assert_array_equal(outline, original_outline)
        points[0, 0] = 100
        self.assertEqual(first[1:], second[1:])

    def test_empty_and_iterable_boxes(self):
        for boxes in ([], iter(())):
            self.assertEqual(get_pair_line(boxes, 1), [])
        self.assertEqual(get_pair_line([], 1, OUTER, 2), [])
        self.assertEqual(len(get_pair_line(iter(both([0, 0, 1, 1])), 1)), 2)

    def test_invalid_boxes_are_validated_even_without_directions(self):
        for box in (None, [0, 0, 1], [0, 0, 1, 1, 2], [[0, 0, 1, 1]],
                    [1, 0, 0, 1], [0, 1, 1, 0], [0, 0, 0, 1], [0, 0, 1, 0],
                    [0, 0, np.nan, 1], [0, 0, np.inf, 1], [-1e308, 0, 1e308, 1],
                    [0, 0, "bad", 1], [0, 0, 10**400, 1]):
            with self.subTest(box=box), self.assertRaisesRegex(ValueError, r"boxes\[0\].box"):
                get_pair_line([{"box": box}], 1)

    def test_invalid_specs_and_old_api(self):
        for boxes in (None, 1, [[0, 0, 1, 1]], np.array([[0, 0, 1, 1]]), [None], [{}],
                      [{"box": [0, 0, 1, 1], "z_points": []}],
                      [{"box": [0, 0, 1, 1], "spacing": 1}]):
            with self.subTest(boxes=boxes), self.assertRaisesRegex(ValueError, "boxes"):
                get_pair_line(boxes, 1)
        with self.assertRaises(TypeError):
            get_pair_line(both([0, 0, 1, 1]), 1, spacing=1)

    def test_invalid_points_report_field_and_index(self):
        for field in ("x_points", "y_points"):
            for points in (None, 1, [0, 1], [[0]], [[0, 0, 0]], np.empty((0, 3)),
                           [[[]]], [[0, 0], [0]], [[np.nan, 0]], [[0, np.inf]],
                           [["bad", 0]], [[10**400, 0]], [[-1, -1]], [[2, 2]]):
                specs = [{"box": [0, 0, 1, 1]}] * 2 + [{"box": [0, 0, 1, 1], field: points}]
                with self.subTest(field=field, points=points), self.assertRaisesRegex(
                        ValueError, rf"boxes\[2\].{field}"):
                    get_pair_line(specs, 1)

    def test_invalid_scalar_parameters(self):
        for name in ("expand_size", "distance"):
            for value in (-1, np.inf, np.nan, True, [1]):
                kwargs = dict(expand_size=1, outer_outline=OUTER, distance=2)
                kwargs[name] = value
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    get_pair_line(both([1, 1, 2, 2]), **kwargs)

    def test_missing_parameters(self):
        with self.assertRaises(TypeError):
            get_pair_line(both([0, 0, 1, 1]))
        for kwargs in (dict(outer_outline=OUTER), dict(distance=2)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                get_pair_line(both([0, 0, 1, 1]), 1, **kwargs)

    def test_invalid_outline_and_outside_box(self):
        for outer in ([], [[0, 0], [20, 0], [19, 20], [0, 20]],
                      [[0, 0], [20, 20], [20, 0], [0, 20]]):
            with self.subTest(outer=outer), self.assertRaises(ValueError):
                get_pair_line(both([1, 1, 2, 2]), 1, outer, 2)
        for box in ([-1, 2, 2, 4], [21, 2, 23, 4], [2, 19, 4, 21], [4, 5, 10, 10]):
            for spec in ({"box": box}, both(box)[0]):
                with self.subTest(spec=spec), self.assertRaisesRegex(ValueError, r"boxes\[0\].box"):
                    get_pair_line([spec], 1, NOTCHED, 2)

    def test_public_mesher_preserves_nonuniform_points_in_each_direction(self):
        outer = [[-20, -20], [80, -20], [80, 80], [-20, 80]]
        for axis, field in enumerate(("x_points", "y_points")):
            box = [10, 0, 30, 50] if axis == 0 else [0, 10, 50, 30]
            # Full 5 -> 20 intervals, omitted endpoints, duplicate/unsorted
            # interior samples, and no interior samples use the same contract.
            for coordinates in ((0, 5, 10, 30, 50), (30, 5, 10, 30), ()):
                with self.subTest(axis=axis, coordinates=coordinates):
                    points = [[-999, v] if axis == 0 else [v, 999] for v in coordinates]
                    specs = [dict(box=box, **{field: points})]
                    pairs = get_pair_line(specs, 2)
                    original_pairs = copy.deepcopy(pairs)
                    kwargs = dict(outer_outline=outer, min_transition_size=2)
                    mesh = imprint_lines(20, pairs, **kwargs)
                    self.assertEqual(len(pairs), 1)
                    self.assertEqual(pairs, original_pairs)
                    self.assertEqual(specs[0][field], points)
                    self.assert_cross_sections(mesh, pairs)

                    # Project onto line1 and normalize order explicitly: the
                    # generated mesh must be identical regardless of the unused
                    # component, input order, duplicates or missing endpoints.
                    reference = [[10, v] if axis == 0 else [v, 10]
                                 for v in sorted({0, 50, *coordinates})]
                    expected = imprint_lines(20, get_pair_line([
                        dict(box=box, **{field: reference})], 2), **kwargs)
                    np.testing.assert_array_equal(mesh.nodes, expected.nodes)
                    np.testing.assert_array_equal(mesh.elements, expected.elements)

    def test_public_mesher_preserves_sampled_cross_sections(self):
        outer = [[0, 0], [12, 0], [12, 12], [0, 12]]
        pairs = get_pair_line([{"box": [1, 4, 3, 6],
                                "x_points": [[1, y] for y in (4, 4.6, 5.2, 5.8, 6)],
                                "y_points": [[x, 4] for x in (1, 1.6, 2.2, 2.8, 3)]}],
                              .5, outer, 2)
        mesh = imprint_lines(2, pairs, outer_outline=outer,
                             min_transition_size=.5)
        self.assert_cross_sections(mesh, pairs)

    def assert_cross_sections(self, mesh, pairs):
        self.assertGreater(mesh.element_count, 0)
        self.assertTrue(np.all(mesh.nodes[:, 2] == 0))
        elements = mesh.elements
        edges = np.stack((elements, np.roll(elements, -1, axis=1)), axis=-1).reshape(-1, 2)
        xy = mesh.nodes[edges, :2]
        for pair in pairs:
            axis = 0 if pair["direction"] == "x" else 1
            start, end = pair["line1"][0][axis], pair["line2"][0][axis]
            for point in [*pair["points"], *pair["line1"]]:
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
