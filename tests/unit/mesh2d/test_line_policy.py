"""LINE spacing and transition limits hold through bounded local repairs."""

import unittest
from unittest.mock import patch

import numpy as np

from mesher.mesh2d.adapter import QuadLayerError
from mesher.mesh2d.adapter.layer._construction import line, strip
from mesher.mesh2d.adapter.layer._construction.units import build_unit_patch
from tests.unit.mesh2d import test_section_line as existing


def _dense_fixture():
    values = np.r_[0., 500., np.arange(1000., 1101., 5.), 1600., 2100.]
    source, outer, left, right = existing._fixture(len(values) - 1, 2100., 500.)
    source[:, 0] = values
    return source, outer, left, right


class LinePolicyTests(unittest.TestCase):
    def assert_conforming(self, candidate):
        existing.SectionLineTests.assert_conforming(self, candidate)

    def assert_policy(self, candidate, h, ratio=.5, limit=10, tol=1e-8):
        spacing = np.linalg.norm(np.diff(candidate.nodes[candidate.outer_ids, :2], axis=0), axis=1)
        self.assertTrue(np.all(spacing >= ratio * h - tol), spacing)
        self.assertTrue(np.all(spacing <= candidate.diagnostics["max_size_ratio"] * h + tol), spacing)
        for unit in candidate.units:
            loads = unit["source_segments_per_outer"]
            self.assertEqual(len(loads), unit["boundary_counts"][2] - 1)
            self.assertEqual(sum(loads), unit["boundary_counts"][0] - 1)
            self.assertGreaterEqual(min(loads), 0)
            self.assertLessEqual(max(loads), limit)
        self.assert_conforming(candidate)

    def test_dense_run_uses_bounded_spacing_and_preserves_every_source_node(self):
        args = _dense_fixture()
        originals = [value.copy() for value in args]
        candidates = line.plan_line(*args, 500., 1e-9, max_transition_count=10)
        self.assertTrue(candidates)
        for candidate in candidates:
            self.assert_policy(candidate, 500.)
            np.testing.assert_array_equal(candidate.nodes[candidate.source_ids], args[0])
        for actual, original in zip(args, originals):
            np.testing.assert_array_equal(actual, original)

    def test_refinement_assignment_respects_the_built_child_boundary(self):
        source, _, left, right = existing._fixture(2, 1., 1.)
        source[1, 0] = .49
        outer = np.array([[x, 1., 0.] for x in (0., .01, .99, .995, 1.)])
        patch = strip.build_strip(source, outer, left, right, [(0, 0), (2, 4)],
                                  0., 1e-10, cache={}, stats={}, cap=1)
        self.assert_conforming(patch)
        unit, = patch.units
        self.assertEqual(unit["method"], "paired")
        # The first child spans one inner and one outer edge. A nearest-point
        # assignment over the whole group would attribute it to the next edge.
        loads = unit["source_segments_per_outer"]
        self.assertEqual(loads[0], 1)
        self.assertEqual(sum(loads[1:]), 1)

    def test_spacing_and_caps_hold_for_every_ordinary_seam_parity(self):
        for left_count in (2, 3):
            for right_count in (2, 3):
                with self.subTest(left_nodes=left_count, right_nodes=right_count):
                    args = existing._fixture(6, 12., 4., left_count, right_count)
                    candidates = line.plan_line(*args, 4., 1e-10, max_transition_count=2)
                    self.assertTrue(candidates)
                    for candidate in candidates:
                        self.assert_policy(candidate, 4., limit=2)
                        self.assertEqual(len(candidate.left_ids), left_count)
                        self.assertEqual(len(candidate.right_ids), right_count)

    def test_policy_is_independent_of_axis_direction_and_coordinate_scale(self):
        for scale in (1e-5, 1., 1e8):
            for rotation in range(4):
                for reverse in (False, True):
                    with self.subTest(scale=scale, rotation=rotation, reverse=reverse):
                        args = [value * scale for value in _dense_fixture()]
                        for value in args:
                            for _ in range(rotation):
                                value[:, :2] = value[:, [1, 0]] * [-1, 1]
                        if reverse:
                            args = [args[0][::-1], args[1][::-1], args[3], args[2]]
                        candidates = line.plan_line(*args, 500. * scale, 1e-9 * scale,
                                                    best_only=True, max_transition_count=10)
                        self.assertTrue(candidates)
                        self.assert_policy(candidates[0], 500. * scale, tol=1e-7 * scale)
                        np.testing.assert_array_equal(candidates[0].nodes[candidates[0].source_ids],
                                                      args[0])

    def test_forced_short_interval_is_preserved_once_and_reported(self):
        source, _, left, right = existing._fixture(8, 16., 4.)
        outer = np.array([[0., 4., 0.], [3., 4., 0.], [16., 4., 0.]])
        candidate = line.plan_line(source, outer, left, right, 8., 1e-10, best_only=True, max_transition_count=10)[0]
        values = candidate.nodes[candidate.outer_ids, 0]
        np.testing.assert_array_equal(values[:2], [0., 3.])
        self.assertTrue(np.all(np.diff(values[1:]) >= 4. - 1e-9))
        self.assertTrue(np.all(np.diff(values[1:]) <= 1.25 * 8. + 1e-9))
        exceptions = candidate.diagnostics["forced_short_intervals"]
        self.assertEqual(len(exceptions), 1)
        self.assertAlmostEqual(exceptions[0]["length"], 3.)
        self.assert_conforming(candidate)

    def test_forced_short_interval_does_not_waive_transition_limit(self):
        args = existing._fixture(12, 1., 4., right_nodes=3)
        originals = [value.copy() for value in args]
        with self.assertRaises(QuadLayerError) as caught:
            line.plan_line(*args, 10., 1e-10, max_transition_count=10)
        self.assertEqual(caught.exception.code, "line_constraints_infeasible")
        for actual, original in zip(args, originals):
            np.testing.assert_array_equal(actual, original)

    def test_dense_candidates_ignore_both_ratios_and_keep_mandatory_anchors(self):
        source, outer, left, right = existing._fixture(60, 300., 500.)
        anchor = [125., 500., 0.]
        outer = np.insert(outer, 1, anchor, axis=0)

        def candidates(minimum, maximum):
            return line.plan_line(source, outer, left, right, 1000., 1e-8,
                                  min_size_ratio=minimum, max_size_ratio=maximum,
                                  max_transition_count=20, size_band_active=False)

        reference = candidates(1., 1.)
        self.assertTrue(reference)
        alternatives = candidates(.01, 2.)
        self.assertEqual(len(alternatives), len(reference))
        for expected, actual in zip(reference, alternatives):
            np.testing.assert_array_equal(actual.nodes, expected.nodes)
            np.testing.assert_array_equal(actual.quads, expected.quads)
            front = actual.nodes[actual.outer_ids]
            self.assertEqual(np.count_nonzero(np.all(front == anchor, axis=1)), 1)
            self.assertTrue(np.all(np.diff(front[:, 0]) > 0.))
            self.assertEqual(actual.diagnostics['forced_short_intervals'], [])
            self.assertLessEqual(actual.diagnostics['max_actual_transition_count'], 20)
            self.assert_conforming(actual)

    def test_dense_sampling_survives_coincident_cut_projections(self):
        source, _, _, _ = existing._fixture(6, 6., 5.)
        outer = np.array([[x, 5., 0.] for x in (1., 2.08, 3.7)])
        left = np.array([source[0], outer[0]])
        right = np.array([source[-1], outer[-1]])
        candidates = line.plan_line(source, outer, left, right, 5., 1e-10,
                                    max_transition_count=1, size_band_active=False, best_only=True)
        self.assertTrue(candidates)
        candidate = candidates[0]
        front = candidate.nodes[candidate.outer_ids]
        self.assertTrue(np.all(np.diff(front[:, 0]) > 0.))
        for anchor in outer:
            self.assertEqual(np.count_nonzero(np.all(front == anchor, axis=1)), 1)
        loads = [count for unit in candidate.units for count in unit['source_segments_per_outer']]
        self.assertEqual(sum(loads), 6)
        self.assertLessEqual(max(loads), 1)
        self.assert_conforming(candidate)

    def test_equal_upper_and_lower_ratio_one_require_exact_target_spacing(self):
        candidates = line.plan_line(*existing._fixture(8, 16., 4.), 4., 1e-10,
                                    min_size_ratio=1., max_size_ratio=1., max_transition_count=10)
        self.assertTrue(candidates)
        for candidate in candidates:
            self.assert_policy(candidate, 4., ratio=1.)
        with self.assertRaises(QuadLayerError) as caught:
            line.plan_line(*existing._fixture(8, 15., 4.), 4., 1e-10, min_size_ratio=1., max_size_ratio=1., max_transition_count=10)
        self.assertEqual(caught.exception.code, "line_constraints_infeasible")

    def test_larger_minimum_ratio_can_make_transition_capacity_infeasible(self):
        args = existing._fixture(40, 100., 50.)
        candidates = line.plan_line(*args, 50., 1e-10, min_size_ratio=.5, max_transition_count=10)
        self.assertTrue(candidates)
        for candidate in candidates:
            self.assert_policy(candidate, 50.)
        with self.assertRaises(QuadLayerError) as caught:
            line.plan_line(*args, 50., 1e-10, min_size_ratio=.75, max_transition_count=10)
        self.assertEqual(caught.exception.code, "line_constraints_infeasible")

    def test_refinement_is_not_capped_by_max_transition_count(self):
        candidate = line.plan_line(*existing._fixture(2, 20., 4.), 1., 1e-10,
                                   max_transition_count=1, best_only=True)[0]
        self.assert_policy(candidate, 1., limit=1)
        self.assertTrue(any(unit["boundary_counts"][2] - 1 > 1 for unit in candidate.units))

    def test_two_outer_edges_physically_receive_their_capped_source_slices(self):
        source, _, left, right = existing._fixture(20, 20., 8.)
        outer = np.array([[0., 8., 0.], [4., 8., 0.], [20., 8., 0.]])
        candidate = strip.build_strip(source, outer, left, right, [(0, 0), (20, 2)],
                                      0., 1e-10, cache={}, stats={}, cap=10)
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.units[0]["source_segments_per_outer"], [10, 10])
        # The private transition reaches the tenth source edge even though the
        # geometric projection of the prescribed outer vertex would reach x=4.
        path = [candidate.source_ids[10],
                np.flatnonzero(np.all(candidate.nodes[:, :2] == (7., 4.), axis=1)).item(),
                candidate.outer_ids[1]]
        edges = {frozenset((a, b)) for cell in candidate.quads
                 for a, b in zip(cell, np.roll(cell, -1))}
        for edge in zip(path, path[1:]):
            self.assertIn(frozenset(edge), edges)
        self.assert_conforming(candidate)

    def test_fixed_outer_boundary_preserves_supplied_spacing(self):
        args = existing._fixture(12, 10., 5., right_nodes=3)
        candidate = line.plan_line(*args, 500., 1e-10, fixed_outer_boundary=True,
                                   max_transition_count=1, min_size_ratio=1.)[0]
        np.testing.assert_array_equal(candidate.nodes[candidate.outer_ids], args[1])
        self.assertEqual(candidate.units[0]["boundary_counts"][0] - 1, 12)
        self.assert_conforming(candidate)

    def test_trial_budget_grows_with_units_without_searching_cut_combinations(self):
        for count in (200, 400):
            with self.subTest(source_segments=count):
                with patch.object(strip, "build_unit_patch", wraps=build_unit_patch) as builds:
                    candidate = line.plan_line(*existing._fixture(count, float(count), 10.),
                                               10., 1e-10, best_only=True, max_transition_count=10)[0]
                self.assert_policy(candidate, 10.)
                self.assertLessEqual(builds.call_count, count)
                self.assertEqual(candidate.diagnostics["unit_trial_count"], builds.call_count)


if __name__ == "__main__":
    unittest.main()
