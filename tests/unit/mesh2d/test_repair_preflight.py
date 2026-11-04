"""Repair validation stays complete while rejecting proven budget excess early."""

from contextlib import ExitStack
import unittest
from unittest.mock import patch

import numpy as np

from mesher.mesh2d.adapter import QuadLayerError, extend_quad_layer
from mesher.mesh2d.adapter.layer import _repair_adaptive, _repair_patch
from mesher.mesh2d.adapter.layer._repair_patch import build_repair_patch
from tests.unit.mesh2d.test_corner_repair import _step
from tests.unit.mesh2d.test_extend_quad_layer import _polyomino
from tests.unit.mesh2d.test_repair_patch import _pattern, _source, _xyz
from tests.unit.mesh2d.test_slot_repair import _blocks


class RepairPreflightTests(unittest.TestCase):
    def test_long_slot_rejects_before_patterns_sampling_or_building(self):
        xs = np.arange(1., 502., .5)
        mesh, outline = _blocks([
            ([0., .5, 1.], np.arange(0., 3.01, .5)),
            (xs, [0., .5, 1.]),
            (xs, [2., 2.5, 3.]),
        ])
        original_nodes, original_elements = mesh.nodes.copy(), mesh.elements.copy()
        original_contacts = _repair_patch.segment_contacts
        original_source = _repair_patch._source_path
        source_sizes = []

        def source_path(*args, **kwargs):
            source, boundary = original_source(*args, **kwargs)
            source_sizes.append(len(source))
            return source, boundary

        with ExitStack() as stack:
            contacts = stack.enter_context(patch.object(
                _repair_patch, 'segment_contacts', wraps=original_contacts))
            stack.enter_context(patch.object(_repair_patch, '_source_path', side_effect=source_path))
            blocked = [stack.enter_context(patch.object(
                module, name, side_effect=AssertionError(f'{name} must not run')))
                for module, name in (
                    (_repair_patch, '_pattern_geometry'),
                    (_repair_patch, '_sample_edges'),
                    (_repair_patch, '_four_sided_units'),
                    (_repair_adaptive, 'build_adaptive_face'),
                )]
            with self.assertRaises(QuadLayerError) as caught:
                extend_quad_layer(mesh, outline, .5, max_added_elements=1)
            self.assertEqual(source_sizes, [2005])
            self.assertGreaterEqual(contacts.call_count, 2005)
            self.assertLess(contacts.call_count, 10 * 2005)
            for operation in blocked:
                operation.assert_not_called()
        error = caught.exception
        self.assertEqual(error.code, 'element_limit_exceeded')
        self.assertEqual(error.context['stage'], 'slot_repair_count_preflight')
        self.assertEqual(error.context['count_kind'], 'lower_bound')
        self.assertEqual(error.context['required_elements'], 1002)
        self.assertEqual(error.context['limit'], 1)
        np.testing.assert_array_equal(mesh.nodes, original_nodes)
        np.testing.assert_array_equal(mesh.elements, original_elements)

    def test_odd_and_even_minimum_boundaries_accept_exact_two_cell_budget(self):
        cases = (
            # Four frozen segments plus one free interval: parity needs a
            # second mouth segment, yielding two stacked quadrilaterals.
            ((_xyz([(1., 1.), (0., 1.)]),
              _xyz([(0., 1.), (0., .5), (0., 0.)]),
              _xyz([(0., 0.), (1., 0.)])), .5, 5),
            # Five frozen segments plus one mouth interval already give an
            # even perimeter and two side-by-side quadrilaterals.
            ((_xyz([(2., 1.), (1., 1.), (0., 1.)]),
              _xyz([(0., 1.), (0., 0.)]),
              _xyz([(0., 0.), (1., 0.), (2., 0.)])), 1., 6),
        )
        for forward, size, boundary_min in cases:
            for source in (forward, tuple(chain[::-1] for chain in forward[::-1])):
                with self.subTest(boundary_min=boundary_min, start=source[0][0].tolist()):
                    with patch.object(_repair_patch, '_pattern_geometry') as patterns:
                        with self.assertRaises(QuadLayerError) as caught:
                            build_repair_patch(source, (), size, 1e-10, max_added_elements=1)
                        patterns.assert_not_called()
                    self.assertEqual(caught.exception.code, 'element_limit_exceeded')
                    self.assertEqual(caught.exception.context['required_elements'], 2)
                    result = build_repair_patch(source, (), size, 1e-10, max_added_elements=2)
                    self.assertEqual(len(result.quads), 2)
                    frozen = np.vstack((source[0], source[1][1:], source[2][1:]))
                    np.testing.assert_array_equal(result.nodes[result.source_ids], frozen)

    def test_source_and_cap_input_validation_precede_budget(self):
        invalid_sources = []
        for problem in ('shape', 'nonfinite', 'nonzero_z', 'unjoined', 'zero_length'):
            source = list(_source())
            if problem == 'shape':
                source[0] = source[0][:, :2]
            elif problem == 'nonfinite':
                source[0][1, 0] = np.nan
            elif problem == 'nonzero_z':
                source[0][1, 2] = 1.
            elif problem == 'unjoined':
                source[1][0, 0] = .1
            else:
                source[0][1] = source[0][0]
            invalid_sources.append((problem, source))
        for problem, source in invalid_sources:
            with self.subTest(problem=problem):
                with self.assertRaises(QuadLayerError) as caught:
                    build_repair_patch(source, (), .75, 1e-10, max_added_elements=1)
                self.assertNotEqual(caught.exception.code, 'element_limit_exceeded')
        with self.assertRaises(QuadLayerError) as caught:
            build_repair_patch(_source(), (), .75, 1e-10,
                               max_added_elements=1, repair_max_element_size=.25)
        self.assertEqual(caught.exception.code, 'invalid_input')
        self.assertEqual(caught.exception.context['parameter'], 'repair_max_element_size')

    def test_self_contacts_are_still_rejected_before_budget_in_both_windings(self):
        cases = (
            [(0., 0.), (3., 0.), (1., 2.), (1., -1.), (0., 2.)],  # Crossing.
            [(0., 0.), (2., 0.), (2., 2.), (1., 0.), (0., 2.)],   # Nonadjacent touch.
            [(0., 0.), (2., 0.), (1., 0.), (2., 2.), (0., 2.)],   # Adjacent backtrack.
            [(0., 0.), (2., 0.), (2., 2.), (0., 2.), (0., -1.)],  # Closing-edge overlap.
            [(0., 0.), (2., 0.), (2., 2.), (1., 5e-11), (0., 2.)],  # Within tolerance.
        )
        for points in cases:
            forward = _xyz(points)
            for boundary in (forward, forward[::-1]):
                with self.subTest(points=boundary.tolist()):
                    source = (boundary[:2], boundary[1:3], boundary[2:])
                    with self.assertRaises(QuadLayerError) as caught:
                        build_repair_patch(source, (), 1., 1e-10, max_added_elements=1)
                    self.assertEqual(caught.exception.code, 'unmeshable_slot_repair')
                    self.assertTrue('intersects itself' in str(caught.exception)
                                    or 'doubles back' in str(caught.exception))

    def test_proven_budget_excess_precedes_repair_local_pattern_layout(self):
        source = _source()
        patterns = (_pattern([(0., 1.), (2., 1.)]),)  # Ends inside the repair.
        with self.assertRaises(QuadLayerError) as caught:
            build_repair_patch(source, patterns, .75, 1e-10, max_added_elements=1)
        self.assertEqual(caught.exception.code, 'element_limit_exceeded')
        self.assertEqual(caught.exception.context['required_elements'], 5)
        with self.assertRaises(QuadLayerError) as caught:
            build_repair_patch(source, patterns, .75, 1e-10)
        self.assertEqual(caught.exception.code, 'unsupported_pattern_layout')

    def test_corner_preflight_reports_global_budget_and_preserves_prior_staged_repairs(self):
        cross = {(x, y) for x in range(3) for y in range(3) if x == 1 or y == 1}
        cases = ((_step(), 0, 6), (_polyomino(cross)[:2], 1, 2))
        for (mesh, outline), corner_id, required in cases:
            with self.subTest(corner_id=corner_id):
                original_nodes, original_elements = mesh.nodes.copy(), mesh.elements.copy()
                with self.assertRaises(QuadLayerError) as caught:
                    extend_quad_layer(mesh, outline, 1., max_added_elements=1)
                error = caught.exception
                self.assertEqual(error.code, 'element_limit_exceeded')
                self.assertEqual(error.context['stage'], 'corner_repair_count_preflight')
                self.assertEqual(error.context['count_kind'], 'lower_bound')
                self.assertEqual(error.context['corner_id'], corner_id)
                self.assertEqual(error.context['required_elements'], required)
                self.assertEqual(error.context['limit'], 1)
                np.testing.assert_array_equal(mesh.nodes, original_nodes)
                np.testing.assert_array_equal(mesh.elements, original_elements)


if __name__ == '__main__':
    unittest.main()
