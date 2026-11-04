"""Density sampling obeys count and spacing constraints without geometry search."""

import unittest
import warnings
from unittest.mock import patch

import numpy as np

from mesher.mesh2d.adapter.layer._construction.sampling import LineSampling, line_count_bounds
from mesher.mesh2d.adapter.layer._types import QuadLayerError


def _xyz(x, y=0):
    x = np.asarray(x, dtype=float)
    return np.column_stack((x, np.full(len(x), y), np.zeros(len(x))))


def _dense_fixture():
    return _xyz(np.r_[0, 500, np.arange(1000, 1101, 5), 1600, 2100]), _xyz([0, 2100], 500)


class LineSamplingTests(unittest.TestCase):
    def assert_infeasible(self, action):
        with self.assertRaises(QuadLayerError) as caught:
            action()
        self.assertEqual(caught.exception.code, "line_constraints_infeasible")

    def test_dense_region_has_the_planned_six_segment_distribution(self):
        sampler = LineSampling(*_dense_fixture(), 500, 1e-9)
        self.assertAlmostEqual(sampler.demand, 6)
        self.assertEqual(sampler.counts(2, 2), [6])
        points, error = sampler.sample(6)
        np.testing.assert_allclose(points[:, 0], [0, 425, 800, 1050, 1300, 1675, 2100])
        self.assertAlmostEqual(error, np.mean((np.diff(points[:, 0]) / 500 - 1) ** 2))
        self.assertTrue(np.all(np.diff(points[:, 0]) >= 250))
        self.assertTrue(np.all(np.diff(points[:, 0]) <= 500))

    def test_sampling_is_equivariant_under_reversal_rotation_and_scale(self):
        source, anchors = _dense_fixture()
        expected = LineSampling(source, anchors, 500, 1e-9).sample(6)[0]
        for reverse in (False, True):
            for vertical in (False, True):
                for scale in (1e-6, 1, 1e6):
                    with self.subTest(reverse=reverse, vertical=vertical, scale=scale):
                        transformed = []
                        for points in (source, anchors, expected):
                            points = points[::-1] if reverse else points
                            points = points[:, [1, 0, 2]] if vertical else points
                            transformed.append(points * scale)
                        sampler = LineSampling(*transformed[:2], 500 * scale, 1e-9 * scale)
                        np.testing.assert_allclose(sampler.sample(6)[0], transformed[2], rtol=1e-13)

    def test_parity_candidates_are_bounded_neighbors_not_enumeration(self):
        sampler = LineSampling(_xyz(np.arange(25)), _xyz([0, 24], 5), 5, 1e-10)
        self.assertEqual(sampler.counts(2, 2), [6])
        self.assertEqual(sampler.counts(2, 3), [5])
        dense = LineSampling(*_dense_fixture(), 500, 1e-9)
        self.assertEqual(dense.counts(2, 3), [5, 7])
        self.assertLessEqual(len(dense.counts(2, 3)), 2)

    def test_parity_can_make_an_otherwise_legal_count_infeasible(self):
        sampler = LineSampling(_xyz([0, 1, 2]), _xyz([0, 2], 1), 1, 1e-10,
                               min_size_ratio=1)
        self.assertEqual(sampler.counts(2, 2), [2])
        self.assert_infeasible(lambda: sampler.counts(2, 3))

    def test_floating_density_sum_does_not_add_an_extra_count_candidate(self):
        sampler = LineSampling(_xyz(np.linspace(0, 100, 21)), _xyz([0, 100], 50), 50, 1e-9)
        self.assertAlmostEqual(sampler.demand, 2)
        self.assertEqual(sampler.counts(2, 2), [2])

    def test_arbitrarily_large_integer_transition_limit_does_not_overflow(self):
        sampler = LineSampling(*_dense_fixture(), 500, 1e-9, max_transition_count=10 ** 400)
        self.assertAlmostEqual(sampler.demand, 2100 / 500)
        self.assertEqual(sampler.counts(2, 2), [6])
        sampler.validate_sample(sampler.sample(6)[0])

    def test_tiny_positive_size_ratios_keep_finite_count_bounds(self):
        for ratio in (np.nextafter(0.0, 1.0), 1e-320):
            with self.subTest(ratio=ratio), warnings.catch_warnings():
                warnings.simplefilter("error")
                sampler = LineSampling(_xyz([0, 0.25, 0.5]), _xyz([0, 0.5], 1),
                                       0.25, 1e-10, min_size_ratio=ratio)
                self.assertEqual(sampler.counts(2, 2), [2])
                self.assertLessEqual(sampler.upper_count, np.iinfo(np.intp).max)
                np.testing.assert_allclose(sampler.sample(2)[0][:, 0], [0, 0.25, 0.5])

    def test_transition_floor_can_exceed_the_demand_neighbor(self):
        # Source outside the shorter front still counts against the hard K cap.
        source = _xyz(np.arange(-20, 21))
        sampler = LineSampling(source, _xyz([-1, 1], 1), 1, 1e-10)
        self.assertEqual(sampler.counts(2, 2), [4])
        self.assertEqual(sampler.lower_count, 4)

    def test_source_projection_exterior_uses_the_global_density(self):
        sampler = LineSampling(_xyz([2, 2.1, 2.2, 3]), _xyz([0, 5], 1), 1, 1e-10,
                               max_transition_count=2)
        self.assertAlmostEqual(sampler.demand, 5.8)

    def test_short_anchor_interval_is_preserved_once_and_recorded(self):
        anchors = _xyz([0, 0.2, 2.2], 1)
        sampler = LineSampling(_xyz([0, 1.1, 2.2]), anchors, 1, 1e-10)
        points, _ = sampler.sample(4)
        np.testing.assert_array_equal(points[:2], anchors[:2])
        self.assertEqual(sampler.diagnostics["forced_short_intervals"], [
            {"start": [0, 1], "end": [0.2, 1], "length": 0.2},
        ])
        self.assertTrue(np.all(np.diff(points[1:, 0]) >= 0.5))
        split = np.insert(points, 1, [0.1, 1, 0], axis=0)
        self.assert_infeasible(lambda: sampler.validate_sample(split))

    def test_anchor_allocation_prioritizes_density_and_retains_exact_points(self):
        source, _ = _dense_fixture()
        anchors = _xyz([0, 1000, 1100, 2100], 500)
        sampler = LineSampling(source, anchors, 500, 1e-9)
        points, _ = sampler.sample(7)
        for anchor in anchors:
            self.assertEqual(np.count_nonzero(np.all(points == anchor, axis=1)), 1)
        dense_index = np.flatnonzero(points[:, 0] == 1000)[0]
        self.assertEqual(points[dense_index + 1, 0], 1100)
        self.assertEqual(sampler.diagnostics["forced_short_intervals"][0]["length"], 100)

    def test_conflicting_counts_fail_before_points_are_allocated(self):
        self.assert_infeasible(lambda: line_count_bounds([1], 21, 1, 1e-10, 10, 0.5))
        # One impossible interval cannot borrow count capacity from another.
        self.assert_infeasible(lambda: line_count_bounds([1.25, 10], 2, 1, 1e-10, 10, 0.9))
        self.assert_infeasible(lambda: line_count_bounds([1.5], 1, 1, 1e-10, 10, 1))

    def test_tolerance_accepts_almost_exact_size_bounds(self):
        for delta in (-5e-11, 5e-11):
            with self.subTest(delta=delta):
                sampler = LineSampling(_xyz([0, 1, 2 + delta]), _xyz([0, 2 + delta], 1),
                                       1, 1e-10, min_size_ratio=1)
                self.assertEqual(sampler.counts(2, 2), [2])
                sampler.validate_sample(sampler.sample(2)[0])

    def test_density_and_count_planning_do_not_allocate_the_output_front(self):
        with patch.object(LineSampling, "sample", side_effect=AssertionError("must not sample")):
            sampler = LineSampling(_xyz([0, 1]), _xyz([0, 1], 1), 1e-9, 1e-12)
            self.assertEqual(sampler.counts(2, 2), [1000000001])
            self.assertEqual(sampler._samples, {})
            self.assertLessEqual(len(sampler._positions), 4)

    def test_cached_samples_are_reused_without_reallocation(self):
        sampler = LineSampling(*_dense_fixture(), 500, 1e-9)
        result = sampler.sample(6)
        with patch.object(sampler, "_allocate", side_effect=AssertionError("must use cache")):
            self.assertIs(sampler.sample(6), result)
        self.assertFalse(result[0].flags.writeable)

    def test_validation_rejects_moved_anchors_and_out_of_band_gaps(self):
        sampler = LineSampling(_xyz([0, 1, 2]), _xyz([0, 1, 2], 1), 1, 1e-10)
        points = sampler.sample(4)[0].copy()
        points[2, 0] += 1e-12
        self.assert_infeasible(lambda: sampler.validate_sample(points))
        points = sampler.sample(4)[0].copy()
        points[1, 0] = 0.2
        self.assert_infeasible(lambda: sampler.validate_sample(points))


if __name__ == "__main__":
    unittest.main()
