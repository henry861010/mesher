"""LINE spacing and transition limits hold through bounded local repairs."""

import unittest
from unittest.mock import patch

import numpy as np

from mesher.mesh2d.adapter import QuadLayerError
from mesher.mesh2d.adapter.layer._construction import line
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
        self.assertTrue(np.all(spacing <= h + tol), spacing)
        for unit in candidate.units:
            source, _, outer, _ = np.asarray(unit["boundary_counts"]) - 1
            if outer == 1:
                self.assertLessEqual(source, limit)
            else:
                self.assertEqual(source, 1, "refinement must keep one source segment per unit")
        self.assert_conforming(candidate)

    def test_dense_run_uses_bounded_spacing_and_preserves_every_source_node(self):
        args = _dense_fixture()
        originals = [value.copy() for value in args]
        candidates = line.plan_line(*args, 500., 1e-9)
        self.assertTrue(candidates)
        self.assertLessEqual(len(candidates), 2)
        for candidate in candidates:
            self.assert_policy(candidate, 500.)
            np.testing.assert_array_equal(candidate.nodes[candidate.source_ids], args[0])
        for actual, original in zip(args, originals):
            np.testing.assert_array_equal(actual, original)

    def test_alignment_can_prefer_smaller_valid_elements_over_target_size(self):
        source, outer, left, right = existing._fixture(20, 20., 5.)
        source[:, 0] = np.r_[0., np.linspace(4., 6., 17), 10., 15., 20.]
        candidates = line.plan_line(source, outer, left, right, 10., 1e-10)
        self.assertEqual(len(candidates), 2)
        aligned, target_sized = candidates
        self.assertEqual(len(aligned.outer_ids) - 1, 4)
        self.assertEqual(len(target_sized.outer_ids) - 1, 2)
        self.assertGreater(aligned.size_error, target_sized.size_error)
        self.assertLess(aligned.diagnostics["alignment_error"],
                        target_sized.diagnostics["alignment_error"])
        for candidate in candidates:
            self.assert_policy(candidate, 10.)
        with patch.object(line, "_plan_count", wraps=line._plan_count) as builds:
            chosen = line.plan_line(source, outer, left, right, 10., 1e-10, best_only=True)[0]
        self.assertEqual(builds.call_count, 1)
        np.testing.assert_array_equal(chosen.nodes[chosen.outer_ids], aligned.nodes[aligned.outer_ids])

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
                                                    best_only=True)
                        self.assertTrue(candidates)
                        self.assert_policy(candidates[0], 500. * scale, tol=1e-7 * scale)
                        np.testing.assert_array_equal(candidates[0].nodes[candidates[0].source_ids],
                                                      args[0])

    def test_ten_to_one_is_accepted_but_twelve_to_one_is_rejected(self):
        for count in (10, 12):
            with self.subTest(count=count):
                args = existing._fixture(count, 10., 5., right_nodes=3)
                result = line._plan_count(*args, 0., 1e-10, max_transition_count=10)
                if count == 10:
                    self.assertIsNotNone(result)
                    self.assertEqual(result.units[0]["boundary_counts"][0] - 1, 10)
                    self.assert_conforming(result)
                else:
                    self.assertIsNone(result)

    def test_remaining_capacity_prevents_twelve_plus_eight_assignment(self):
        source, _, left, right = existing._fixture(20, 20., 5.)
        outer = np.array([[0., 5., 0.], [12., 5., 0.], [20., 5., 0.]])
        result = line._plan_count(source, outer, left, right, 0., 1e-10,
                                  max_transition_count=10)
        self.assertIsNotNone(result)
        self.assertEqual([unit["boundary_counts"][0] - 1 for unit in result.units], [10, 10])
        self.assert_conforming(result)

    def test_neighbor_fallback_preserves_current_and_remaining_capacity(self):
        source, _, left, right = existing._fixture(19, 19., 5., right_nodes=3)
        outer = np.array([[0., 5., 0.], [10., 5., 0.], [19., 5., 0.]])
        attempted = []

        def reject_nearest(sides, *args, **kwargs):
            interval = tuple(sides[0][[0, -1], 0])
            attempted.append(interval)
            self.assertLessEqual(len(sides[0]) - 1, 10)
            if interval == (0., 10.):
                raise ValueError("injected nearest-cut failure")
            return build_unit_patch(sides, *args, **kwargs)

        with patch.object(line, "build_unit_patch", side_effect=reject_nearest):
            result = line._plan_count(source, outer, left, right, 0., 1e-10,
                                      max_transition_count=10)
        self.assertIsNotNone(result)
        self.assertIn((0., 9.), attempted)
        self.assertEqual([unit["boundary_counts"][0] - 1 for unit in result.units], [9, 10])
        self.assert_conforming(result)

    def test_closing_repair_cannot_move_excess_segments_to_its_neighbor(self):
        source, _, left, right = existing._fixture(20, 20., 5.)
        outer = np.array([[0., 5., 0.], [10., 5., 0.], [20., 5., 0.]])
        attempted = []

        def reject_closing(sides, *args, **kwargs):
            interval = tuple(sides[0][[0, -1], 0])
            attempted.append(interval)
            self.assertLessEqual(len(sides[0]) - 1, 10)
            if interval == (10., 20.):
                raise ValueError("injected closing-unit failure")
            return build_unit_patch(sides, *args, **kwargs)

        with patch.object(line, "build_unit_patch", side_effect=reject_closing):
            result = line._plan_count(source, outer, left, right, 0., 1e-10,
                                      max_transition_count=10)
        self.assertIsNone(result)
        self.assertEqual(attempted, [(0., 10.), (10., 20.)])

    def test_closing_repair_can_use_a_neighbor_that_preserves_both_caps(self):
        source, _, left, right = existing._fixture(19, 19., 5., right_nodes=3)
        outer = np.array([[0., 5., 0.], [10., 5., 0.], [19., 5., 0.]])
        attempted = []

        def reject_closing(sides, *args, **kwargs):
            interval = tuple(sides[0][[0, -1], 0])
            attempted.append(interval)
            self.assertLessEqual(len(sides[0]) - 1, 10)
            if interval == (10., 19.):
                raise ValueError("injected closing-unit failure")
            return build_unit_patch(sides, *args, **kwargs)

        with patch.object(line, "build_unit_patch", side_effect=reject_closing):
            result = line._plan_count(source, outer, left, right, 0., 1e-10,
                                      max_transition_count=10)
        self.assertIsNotNone(result)
        self.assertIn((9., 19.), attempted)
        self.assertEqual([unit["boundary_counts"][0] - 1 for unit in result.units], [9, 10])
        self.assert_conforming(result)

    def test_right_hand_closing_repair_checks_the_rebuilt_neighbor_capacity(self):
        source, _, left, right = existing._fixture(29, 29., 5.)
        outer = np.array([[0., 5., 0.], [10., 5., 0.], [19., 5., 0.], [29., 5., 0.]])
        attempted = []

        def reject_closing(sides, *args, **kwargs):
            interval = tuple(sides[0][[0, -1], 0])
            attempted.append(interval)
            self.assertLessEqual(len(sides[0]) - 1, 10)
            if interval == (10., 19.):
                raise ValueError("injected center-unit failure")
            return build_unit_patch(sides, *args, **kwargs)

        with patch.object(line, "build_unit_patch", side_effect=reject_closing):
            result = line._plan_count(source, outer, left, right, 0., 1e-10,
                                      max_transition_count=10)
        self.assertIsNotNone(result)
        self.assertIn((20., 29.), attempted)
        self.assertEqual([unit["boundary_counts"][0] - 1 for unit in result.units], [10, 10, 9])
        self.assert_conforming(result)

    def test_forced_short_interval_is_preserved_once_and_reported(self):
        source, _, left, right = existing._fixture(8, 16., 4.)
        outer = np.array([[0., 4., 0.], [3., 4., 0.], [16., 4., 0.]])
        candidate = line.plan_line(source, outer, left, right, 8., 1e-10, best_only=True)[0]
        values = candidate.nodes[candidate.outer_ids, 0]
        np.testing.assert_array_equal(values[:2], [0., 3.])
        self.assertTrue(np.all(np.diff(values[1:]) >= 4. - 1e-9))
        self.assertTrue(np.all(np.diff(values[1:]) <= 8. + 1e-9))
        exceptions = candidate.diagnostics["forced_short_intervals"]
        self.assertEqual(len(exceptions), 1)
        self.assertAlmostEqual(exceptions[0]["length"], 3.)
        self.assert_conforming(candidate)

    def test_forced_short_interval_does_not_waive_transition_limit(self):
        args = existing._fixture(12, 1., 4., right_nodes=3)
        originals = [value.copy() for value in args]
        with self.assertRaises(QuadLayerError) as caught:
            line.plan_line(*args, 10., 1e-10)
        self.assertEqual(caught.exception.code, "line_constraints_infeasible")
        for actual, original in zip(args, originals):
            np.testing.assert_array_equal(actual, original)

    def test_ratio_one_requires_exact_target_spacing(self):
        candidates = line.plan_line(*existing._fixture(8, 16., 4.), 4., 1e-10,
                                    min_size_ratio=1.)
        self.assertTrue(candidates)
        for candidate in candidates:
            self.assert_policy(candidate, 4., ratio=1.)
        with self.assertRaises(QuadLayerError) as caught:
            line.plan_line(*existing._fixture(8, 15., 4.), 4., 1e-10, min_size_ratio=1.)
        self.assertEqual(caught.exception.code, "line_constraints_infeasible")

    def test_larger_minimum_ratio_can_make_transition_capacity_infeasible(self):
        args = existing._fixture(40, 100., 50.)
        candidates = line.plan_line(*args, 50., 1e-10, min_size_ratio=.5)
        self.assertTrue(candidates)
        for candidate in candidates:
            self.assert_policy(candidate, 50.)
        with self.assertRaises(QuadLayerError) as caught:
            line.plan_line(*args, 50., 1e-10, min_size_ratio=.75)
        self.assertEqual(caught.exception.code, "line_constraints_infeasible")

    def test_refinement_is_not_capped_by_max_transition_count(self):
        candidate = line.plan_line(*existing._fixture(2, 20., 4.), 1., 1e-10,
                                   max_transition_count=1, best_only=True)[0]
        self.assert_policy(candidate, 1., limit=1)
        self.assertTrue(any(unit["boundary_counts"][2] - 1 > 1 for unit in candidate.units))

    def test_fixed_outer_boundary_and_repair_default_keep_legacy_behavior(self):
        args = existing._fixture(12, 10., 5., right_nodes=3)
        candidate = line.plan_line(*args, 500., 1e-10, fixed_outer_boundary=True,
                                   max_transition_count=1, min_size_ratio=1.)[0]
        np.testing.assert_array_equal(candidate.nodes[candidate.outer_ids], args[1])
        self.assertEqual(candidate.units[0]["boundary_counts"][0] - 1, 12)
        self.assert_conforming(candidate)
        repair = line._plan_count(*args, 0., 1e-10)
        self.assertIsNotNone(repair)
        self.assertEqual(repair.units[0]["boundary_counts"][0] - 1, 12)

    def test_trial_budget_grows_with_units_without_searching_cut_combinations(self):
        for count in (200, 400):
            with self.subTest(source_segments=count):
                with patch.object(line, "build_unit_patch", wraps=build_unit_patch) as builds:
                    candidate = line.plan_line(*existing._fixture(count, float(count), 10.),
                                               10., 1e-10, best_only=True)[0]
                self.assert_policy(candidate, 10.)
                self.assertEqual(len(candidate.units), count // 10)
                self.assertEqual(builds.call_count, len(candidate.units))
                self.assertLessEqual(candidate.diagnostics["outer_count_attempts"], 2)
                self.assertEqual(candidate.diagnostics["unit_trial_count"], builds.call_count)


if __name__ == "__main__":
    unittest.main()
