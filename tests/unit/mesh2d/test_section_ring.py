"""Concrete finite-state seam rings must close and optimize the whole layer."""

import unittest
from itertools import product
from types import SimpleNamespace

import numpy as np

from mesher.mesh2d.adapter.layer._construction.sections import seam_states, solve_section_ring
from mesher.mesh2d.adapter.layer._types import QuadLayerError


def _ring(count=3):
    sections = [SimpleNamespace(id=i, left_seam=i, right_seam=(i + 1) % count)
                for i in range(count)]
    states = {i: {2: None, 3: None} for i in range(count)}
    return sections, states


def _candidate(error=0.0, quality=1.0):
    return SimpleNamespace(size_error=error, quality=quality)


class SectionRingTests(unittest.TestCase):
    def test_solver_tries_second_initial_state_to_close(self):
        sections, states = _ring()
        options = [{(3, 2): _candidate()}, {(2, 3): _candidate()}, {(3, 3): _candidate()}]
        self.assertEqual(solve_section_ring(sections, states, options),
                         ((3, 2), (2, 3), (3, 3)))

    def test_individually_feasible_sections_can_have_no_closed_assignment(self):
        sections, states = _ring()
        options = [{(2, 3): _candidate(), (3, 2): _candidate()} for _ in sections]
        with self.assertRaises(QuadLayerError) as caught:
            solve_section_ring(sections, states, options)
        self.assertEqual(caught.exception.code, "unmeshable_seam_assignment")
        self.assertEqual(caught.exception.context["section_ids"], [0, 1, 2])

    def test_size_priority_precedes_better_quality(self):
        sections, states = _ring()
        options = [{(2, 2): _candidate(0, 100), (3, 3): _candidate(.01, 1)}
                   for _ in sections]
        self.assertEqual(solve_section_ring(sections, states, options), ((2, 2),) * 3)

    def test_solver_matches_exhaustive_objective_and_is_deterministic(self):
        sections, states = _ring(4)
        options = []
        for index in range(4):
            options.append({(left, right): _candidate(
                float((left + 2 * right + index) % 3),
                float(1 + (3 * left + right + index) % 5))
                for left, right in product((2, 3), repeat=2)})
        expected = []
        for assignment in product((2, 3), repeat=4):
            path = tuple(zip(assignment, assignment[1:] + assignment[:1]))
            patches = [candidate[pair] for candidate, pair in zip(options, path)]
            expected.append((sum(p.size_error for p in patches), max(p.quality for p in patches)))
        selected = solve_section_ring(sections, states, options)
        patches = [candidate[pair] for candidate, pair in zip(options, selected)]
        self.assertEqual((sum(p.size_error for p in patches), max(p.quality for p in patches)),
                         min(expected))
        self.assertEqual(selected, solve_section_ring(sections, states, options))

    def test_mismatched_seam_identity_is_rejected(self):
        sections, states = _ring()
        sections[1].left_seam = 99
        with self.assertRaises(QuadLayerError) as caught:
            solve_section_ring(sections, states, [{}] * 3)
        self.assertEqual(caught.exception.code, "unmeshable_section_layout")

    def test_pattern_kink_has_three_and_four_node_canonical_states(self):
        points = np.array([[0., 0], [0, 2], [1, 2]])
        seam = SimpleNamespace(id=4, points=points)
        states = seam_states(seam, 1e-10)
        self.assertEqual(set(states), {3, 4})
        np.testing.assert_array_equal(states[3][:, :2], points)
        np.testing.assert_array_equal(states[4][:, :2], [[0, 0], [0, 1], [0, 2], [1, 2]])
        self.assertTrue(np.all(states[4][:, 2] == 0))


if __name__ == "__main__":
    unittest.main()
