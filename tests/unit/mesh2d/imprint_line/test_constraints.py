"""Termination and exact-constraint regressions for line imprinting."""
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from mesher.tolerance import GeometryToleranceError
from mesher.mesh2d.imprint_line import imprint_lines, ImprintLineError
from mesher.mesh2d.imprint_line.mesher import _prepare_plan, _assemble, _validate
from mesher.mesh2d.imprint_line._background import Background
from . import test_imprint_lines as fixtures

pair = fixtures.pair


class ConstraintTests(unittest.TestCase):
    def mesh(self, pairs=(), patterns=(), **kwargs):
        options = dict(outer_outline=[[0, 0], [12, 0], [12, 12], [0, 12]],
                       tag="mesh_snap", min_transition_size=.5)
        options.update(kwargs)
        return imprint_lines(2, pairs, patterns, **options)

    def test_impossible_sizing_terminates_with_resolution_error(self):
        code = '''
from mesher.mesh2d.imprint_line import imprint_lines, ImprintLineError
try:
    imprint_lines(.00104, [], [[[.002,0],[.002,.002]],[[.0041,0],[.0041,.002]]],
        outer_outline=[[0,0],[.0061,0],[.0061,.002],[0,.002]],
        tag="mesh_snap", min_transition_size=.00102, max_elements=100)
except ImprintLineError as error:
    assert error.code == 'feature_below_resolution', error
else:
    raise AssertionError('unresolvable sizing succeeded')
'''
        result = subprocess.run([sys.executable, '-c', code], capture_output=True,
                                text=True, timeout=5, env=os.environ.copy())
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_fixed_pattern_and_dense_line_survive_generated_divider(self):
        for axis in (0, 1):
            with self.subTest(axis=axis):
                p = pair(0, (4, 4, 6, 8), [4, 5, 6.0012, 7, 8])
                patterns = np.array([[[11, 6.0005], [11.5, 6.0005]]])
                if axis:
                    p['direction'] = 'y'
                    for field in ('line1', 'line2', 'points'):
                        p[field] = np.asarray(p[field])[:, ::-1].tolist()
                    patterns = patterns[:, :, ::-1]
                before = patterns.copy()
                mesh = self.mesh([p], patterns)
                fixtures.ImprintTests.assert_segment(self, mesh, axis, 6.0005, 4, 6)
                fixtures.ImprintTests.assert_segment(self, mesh, axis, 6.0005, 11, 11.5)
                for point in patterns.reshape(-1, 2):
                    self.assertTrue(np.any(np.all(mesh.nodes[:, :2] == point, axis=1)))
                np.testing.assert_array_equal(patterns, before)

    def test_input_normalization_does_not_move_pattern(self):
        p = pair(0, (4, 4, 6, 6), [4, 5, 6])
        mesh = self.mesh([p], [[[1, 5.0005], [10, 5.0005]]])
        fixtures.ImprintTests.assert_segment(self, mesh, 0, 5.0005, 1, 10)
        self.assertFalse(np.any(mesh.nodes[:, 1] == 5))

    def test_distinct_nearby_patterns_are_not_merged(self):
        with self.assertRaises(GeometryToleranceError) as raised:
            self.mesh(patterns=[[[1, 5], [10, 5]], [[1, 5.0005], [10, 5.0005]]])
        self.assertEqual(raised.exception.code, 'fixed_constraint_conflict')

    def test_nearly_orthogonal_pattern_is_rejected_without_projection(self):
        with self.assertRaises(ValueError):
            self.mesh(patterns=[[[1, 5], [10, 5.0005]]])

    def test_exact_axis_conflicts_with_a_nearby_fixed_pattern(self):
        with self.assertRaises(GeometryToleranceError) as raised:
            self.mesh(patterns=[[[.0005, 1], [.0005, 5]]],
                      outer_outline=[[-3, -3], [6, -3], [6, 6], [-3, 6]])
        self.assertEqual(raised.exception.code, 'fixed_constraint_conflict')
        # Normalization can move a near exterior onto a fixed support across zero.
        with self.assertRaises(GeometryToleranceError) as raised:
            self.mesh(patterns=[[[-.0001, 2], [-.0001, 3]]],
                      outer_outline=[[.0004, 1], [6, 1], [6, 6], [.0004, 6]])
        self.assertEqual(raised.exception.code, 'fixed_constraint_conflict')

    def test_concave_corner_is_retained_on_transition(self):
        outer = np.array([[0, 0], [12, 0], [12, 5], [8, 5], [8, 12], [0, 12]])
        for transpose in (False, True):
            p = pair(0, (2, 4, 4, 6), [4, 5, 6])
            boundary = outer.copy()
            if transpose:
                p['direction'] = 'y'
                for field in ('line1', 'line2', 'points'):
                    p[field] = np.asarray(p[field])[:, ::-1].tolist()
                boundary = boundary[:, ::-1]
            mesh = self.mesh([p], outer_outline=boundary)
            self.assertTrue(np.any(np.all(mesh.nodes[:, :2] == ([5, 8] if transpose else [8, 5]), axis=1)))

    def test_transition_starts_at_the_fixed_wall_and_preserves_dense_contacts(self):
        for side in (0, 1):
            p = pair(0, (2, 4, 3, 5), [4, 4.4994, 4.5006, 5], expand=0)
            x = 1.5 if side == 0 else 3.5
            args = _prepare_plan(1, [p], [[[x, 0], [x, 10]]],
                                 outer_outline=[[0, 0], [10, 0], [10, 10], [0, 10]],
                                 tag="mesh_snap", min_transition_size=.2)
            mesh = _assemble(*args)
            fixtures.ImprintTests.assert_segment(self, mesh, 1, x, 0, 10)
            end = next(t for t in args[1].transitions if t.side == side)
            self.assertEqual(end.inner, x)
            for y in (4, 4.4994, 4.5006, 5):
                fixtures.ImprintTests.assert_segment(self, mesh, 0, y, 2, 3)

    def test_patterns_clipped_to_concave_domain_and_boundary(self):
        patterns = [[[0, 0], [12, 0]], [[-2, 3], [14, 3]],
                    [[6, -2], [6, 14]], [[4, 5], [8, 5]], [[4, 5], [8, 5]]]
        outer = [[0, 0], [12, 0], [12, 12], [8, 12], [8, 5], [4, 5], [4, 12], [0, 12]]
        first = self.mesh(patterns=patterns, outer_outline=outer)
        second = self.mesh(patterns=[p[::-1] for p in patterns[::-1]], outer_outline=outer[::-1])
        np.testing.assert_array_equal(first.nodes, second.nodes)
        np.testing.assert_array_equal(first.elements, second.elements)
        fixtures.ImprintTests.assert_segment(self, first, 0, 3, 0, 12)
        fixtures.ImprintTests.assert_segment(self, first, 1, 6, 0, 5)
        fixtures.ImprintTests.assert_segment(self, first, 0, 5, 4, 8)

    def test_sizing_detects_a_cut_that_makes_no_progress(self):
        original = Background.add

        def stalled(builder, axis, coordinate, interval, kind):
            return False if kind == 'sizing' else original(builder, axis, coordinate, interval, kind)

        with patch.object(Background, 'add', stalled), self.assertRaises(ImprintLineError) as raised:
            self.mesh()
        self.assertEqual(raised.exception.code, 'sizing_no_progress')

    def test_unexpected_unit_failure_is_not_hidden_by_replanning(self):
        error = ImprintLineError('inverted', code='invalid_unit', stage='unit_construction')
        with patch('mesher.mesh2d.imprint_line._cell_mesh.build_unit', side_effect=error), \
                self.assertRaises(ImprintLineError) as raised:
            self.mesh([pair(0, (4, 4, 6, 6))])
        self.assertIs(raised.exception, error)

    def test_validation_rejects_moved_patterns_and_missing_endpoints(self):
        for move_axis, old, new in ((1, 5, 5.1), (0, 1, 1.1)):
            args = _prepare_plan(2, [], [[[2, 9], [9, 9]], [[1, 5], [10, 5]]],
                                 outer_outline=[[0, 0], [12, 0], [12, 12], [0, 12]],
                                 tag="mesh_snap", min_transition_size=.5)
            mesh = _assemble(*args)
            mesh.nodes[mesh.nodes[:, move_axis] == old, move_axis] = new
            with self.assertRaises(ImprintLineError) as raised:
                _validate(mesh, args[0], args[1].constraints)
            self.assertEqual(raised.exception.code, 'constraint_not_preserved')
            self.assertEqual(raised.exception.context['source_index'], 1)

    def test_remote_generated_front_replans_without_moving_pattern(self):
        # The reservation's outer Y=3 line reaches x=0, where a remote pattern
        # endpoint at Y=3.0005 would otherwise create an unresolved strip.
        mesh = self.mesh([pair(1, (4, 6, 6, 8), expand=1)],
                         [[[.5, 3.0005], [1, 3.0005]]])
        fixtures.ImprintTests.assert_segment(self, mesh, 0, 3.0005, .5, 1)

    def test_generated_front_near_a_fixed_level_advances_outwards(self):
        # Requested front 6.0005 lies just beyond the second control's Y=6.
        # Rounding it backwards would shorten the required straight extension.
        vertical = pair(1, (4, 3, 6, 5), [4, 5, 6], expand=(0, 1.0005))
        horizontal = pair(0, (3, 5, 7, 7), [5, 6, 7], expand=0)
        args = _prepare_plan(2, [vertical, horizontal], outer_outline=[[0, 0], [12, 0], [12, 12], [0, 12]],
                             tag="mesh_snap", min_transition_size=.5)
        mesh = _assemble(*args)
        end = next(t for t in args[1].transitions if t.axis == 1 and t.side == 1)
        self.assertGreaterEqual(end.inner, 6.0005)
        fixtures.ImprintTests.assert_segment(self, mesh, 1, 5, 3, 6.0005)
        fixtures.ImprintTests.assert_segment(self, mesh, 0, 6, 3, 7)

    def test_boundary_fallback_stops_at_each_lines_first_outline(self):
        outer = [[0, 0], [12, 0], [12, 8], [10, 8], [10, 3], [8, 3], [8, 8], [0, 8]]
        control = pair(0, (2, 2, 4, 5), [2, 3, 4, 5], expand=40)
        mesh = self.mesh([control], outer_outline=outer)
        for level, end in ((2, 12), (4, 8), (5, 8)):
            fixtures.ImprintTests.assert_segment(self, mesh, 0, level, 0, end)
        self.assertFalse(np.any(np.all(mesh.nodes[:, :2] == (12, 5), axis=1)))

    def test_point_only_pattern_contact_is_retained(self):
        mesh = self.mesh(patterns=[[[-2, 0], [0, 0]]])
        self.assertTrue(np.any(np.all(mesh.nodes[:, :2] == [0, 0], axis=1)))

    def test_rounded_zero_length_unit_edge_is_replanned(self):
        origin = 1e11
        p = pair(0, (2, 4, 3, 5), [4, 4.4994, 4.5006, 5], expand=0)
        for name in ('line1', 'line2', 'points'):
            p[name] = np.asarray(p[name]) + origin
        patterns = np.array([[[3.0011, 0], [3.0011, 10]]]) + origin
        mesh = imprint_lines(1, [p], patterns,
                             outer_outline=np.array([[0, 0], [10, 0], [10, 10], [0, 10]]) + origin,
                             tag="mesh_snap", min_transition_size=.00102)
        fixtures.ImprintTests.assert_segment(self, mesh, 1, patterns[0, 0, 0], origin, origin + 10)


if __name__ == '__main__':
    unittest.main()
