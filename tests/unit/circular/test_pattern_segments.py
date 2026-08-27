import unittest

import numpy as np

from mesher import Mesh2D
from mesher.circular.pattern import _generate_pattern_circle_nodes
from mesher.circular.pattern_segments import (
    _PatternGuideSet,
    _circle_line_intersections,
    _circle_segment_intersections,
    _interval_overlap,
)


class PatternGuideSetTests(unittest.TestCase):
    def test_prepares_axis_metadata_and_preserves_order_and_duplicates(self):
        values = [
            [[2.0, 4.0], [2.0, -1.0]],
            [[5.0, 3.0], [-2.0, 3.0]],
            [[2.0, 4.0], [2.0, -1.0]],
        ]

        guides = _PatternGuideSet.from_values(values, coordinate_scale=5.0)

        self.assertEqual(len(guides), 3)
        np.testing.assert_array_equal(guides.fixed_axes, [0, 1, 0])
        np.testing.assert_array_equal(guides.varying_axes, [1, 0, 1])
        np.testing.assert_array_equal(guides.fixed_values, [2.0, 3.0, 2.0])
        np.testing.assert_array_equal(guides.lower_bounds, [-1.0, -2.0, -1.0])
        np.testing.assert_array_equal(guides.upper_bounds, [4.0, 5.0, 4.0])
        np.testing.assert_array_equal(guides.segments, values)
        self.assertFalse(guides.segments.flags.writeable)

    def test_filters_by_finite_circle_intersection(self):
        center = np.array([2.0, -3.0])
        radius = 5.0
        values = [
            [[2.0, -10.0], [2.0, 10.0]],  # two intersections
            [[-10.0, 2.0], [10.0, 2.0]],  # tangent
            [[7.0, -3.0], [9.0, -3.0]],  # endpoint intersection
            [[2.0, -2.0], [2.0, 0.0]],  # wholly inside the circle
            [[8.0, -20.0], [8.0, 20.0]],  # fixed coordinate outside radius
            [[2.0, 20.0], [2.0, 30.0]],  # circle roots outside segment
            [[54.0, -70.0], [54.0, 70.0]],  # outer-band-only analogue
        ]
        guides = _PatternGuideSet.from_values(values, coordinate_scale=7.0)

        filtered = guides.intersecting_circle(center, radius)

        self.assertEqual(len(filtered), 3)
        np.testing.assert_array_equal(filtered.segments, values[:3])

    def test_circle_helpers_share_line_and_segment_roots(self):
        guides = _PatternGuideSet.from_values(
            [[[3.0, -20.0], [3.0, 1.0]]],
            coordinate_scale=5.0,
        )
        segment = next(iter(guides))

        line_points = _circle_line_intersections(segment, [0.0, 0.0], 5.0)
        segment_points = _circle_segment_intersections(
            segment,
            [0.0, 0.0],
            5.0,
        )

        np.testing.assert_allclose(line_points, [[3.0, -4.0], [3.0, 4.0]])
        np.testing.assert_allclose(segment_points, [[3.0, -4.0]])

    def test_interval_overlap_reports_only_nontrivial_intersections(self):
        self.assertEqual(_interval_overlap(0.0, 2.0, 1.0, 3.0, 0.1), (1.0, 2.0, True))
        self.assertEqual(_interval_overlap(0.0, 1.0, 1.0, 2.0, 0.1), (1.0, 1.0, False))

    def test_invalid_segments_raise_before_filtering(self):
        cases = (
            [[[0.0, 0.0], [1.0, 1.0]]],
            [[[0.0, 0.0], [0.0, 0.0]]],
            [[[100.0, 100.0], [101.0, np.nan]]],
            [[0.0, 0.0], [0.0, 1.0]],
            np.empty((1, 0, 2), dtype=np.float64),
        )

        for values in cases:
            with self.subTest(values=np.asarray(values).shape):
                with self.assertRaises(ValueError):
                    _PatternGuideSet.from_values(
                        values,
                        coordinate_scale=1.0,
                    )

    def test_empty_extreme_and_duplicate_segments_are_deterministic(self):
        empty = _PatternGuideSet.from_values([], coordinate_scale=1.0)
        self.assertEqual(empty.segments.shape, (0, 2, 2))

        values = [
            [[1.0e15, -1.0e6], [1.0e15, 1.0e6]],
            [[0.0, -2.0], [0.0, 2.0]],
            [[0.0, 2.0], [0.0, -2.0]],
        ]
        guides = _PatternGuideSet.from_values(values, coordinate_scale=1.0)
        filtered = guides.intersecting_circle([0.0, 0.0], 1.0)

        np.testing.assert_array_equal(filtered.segments, values[1:])

    def test_pattern_generator_accepts_raw_and_prepared_guides(self):
        mesh = Mesh2D(
            nodes=np.empty((0, 2), dtype=np.float64),
            elements=np.empty((0, 4), dtype=np.int32),
        )
        values = [[[0.0, -8.0], [0.0, 8.0]]]
        guides = _PatternGuideSet.from_values(values, coordinate_scale=5.0)

        raw = _generate_pattern_circle_nodes(
            mesh,
            0.0,
            0.0,
            5.0,
            1.0,
            guide_segments=values,
        )
        prepared = _generate_pattern_circle_nodes(
            mesh,
            0.0,
            0.0,
            5.0,
            1.0,
            guide_segments=guides,
        )

        np.testing.assert_array_equal(prepared, raw)


if __name__ == "__main__":
    unittest.main()
