"""Slot filling preserves frozen edges and shares patterned Quad4 interfaces."""
import unittest
from collections import Counter
from unittest.mock import patch

import numpy as np

from mesher.mesh2d.adapter.layer import _repair_adaptive, _repair_patch
from mesher.mesh2d.adapter.layer._geometry import intervals_cover
from mesher.mesh2d.adapter.layer._patterns import AxisPattern, covered_intervals, occupied_intervals
from mesher.mesh2d.adapter.layer._repair_patch import build_repair_patch
from mesher.mesh2d.adapter.layer._types import QuadLayerError
from mesher.mesh2d.adapter.unit.build_unit_1_1_x_y import _validate_patch


def _xyz(points):
    values = np.asarray(points, dtype=float)
    return np.column_stack((values, np.zeros(len(values))))


def _source(misaligned=False):
    top = [(x, 2.) for x in ((3., 2., 1., 0.) if misaligned else (4., 3., 2., 1., 0.))]
    bottom = [(x, 0.) for x in ((0., 1., 2., 3., 3.5, 4.) if misaligned else (0., 1., 2., 3., 4.))]
    return _xyz(top), _xyz([(0., 2.), (0., 1.), (0., 0.)]), _xyz(bottom)


def _pattern(points, identifier=0):
    return AxisPattern.from_points(points, (identifier,))


class RepairPatchTests(unittest.TestCase):
    def assert_patch(self, result, source, patterns=()):
        frozen = np.vstack((source[0], source[1][1:], source[2][1:]))
        np.testing.assert_array_equal(result.nodes[result.source_ids], frozen)
        np.testing.assert_array_equal(result.nodes[result.mouth_ids[[0, -1]]], frozen[[-1, 0]])
        _validate_patch(result.nodes, result.quads, (result.source_ids, result.mouth_ids))
        incidence = Counter(tuple(sorted((int(a), int(b)))) for quad in result.quads
                            for a, b in zip(quad, np.roll(quad, -1)))
        edges = result.nodes[np.asarray(list(incidence)), :2]
        for pattern in patterns:
            required = occupied_intervals(pattern, result.nodes[result.quads, :2], 1e-10)
            self.assertTrue(intervals_cover(required, covered_intervals(pattern, edges, 1e-10), 1e-10))
        self.assertEqual(set(result.quads.ravel()), set(range(len(result.nodes))))

    def test_misaligned_mouth_and_both_windings_preserve_input(self):
        forward = _source(True)
        for source in (forward, tuple(chain[::-1] for chain in forward[::-1])):
            original = tuple(chain.copy() for chain in source)
            result = build_repair_patch(source, (), .75, 1e-10)
            self.assert_patch(result, source)
            for before, after in zip(original, source):
                np.testing.assert_array_equal(after, before)

    def test_complete_wall_to_mouth_chord_adds_a_shared_mouth_anchor(self):
        source = _source(True)
        patterns = (_pattern([(0., 1.), (4., 1.)]),)
        result = build_repair_patch(source, patterns, .75, 1e-10)
        self.assert_patch(result, source, patterns)
        mouth = result.nodes[result.mouth_ids]
        self.assertEqual(np.count_nonzero(np.all(mouth == [3.5, 1., 0.], axis=1)), 1)
        self.assertEqual(result.diagnostics['face_count'], 2)

    def test_parallel_opposite_wall_chords_have_canonical_shared_segments(self):
        source = _source()
        patterns = (_pattern([(1., 0.), (1., 2.)]), _pattern([(3., 0.), (3., 2.)], 1))
        result = build_repair_patch(source, patterns, .6, 1e-10)
        self.assert_patch(result, source, patterns)
        self.assertEqual(result.diagnostics['face_count'], 3)
        self.assertEqual(result.diagnostics['pattern_ids'], [0, 1])

    def test_adjacent_side_chord_can_form_a_triangular_geometric_face(self):
        source = _source(True)
        patterns = (_pattern([(3.5, 0.), (3.5, 2.)]),)
        result = build_repair_patch(source, patterns, .75, 1e-10)
        self.assert_patch(result, source, patterns)
        self.assertEqual(result.diagnostics['face_count'], 2)
        self.assertTrue(np.all(np.diff(np.sort(result.quads, axis=1), axis=1) > 0))

    def test_collinear_pattern_pieces_are_merged_before_endpoint_checks(self):
        source = _source()
        patterns = (_pattern([(0., 1.), (2., 1.)]), _pattern([(2., 1.), (4., 1.)], 1))
        result = build_repair_patch(source, patterns, .75, 1e-10)
        self.assert_patch(result, source, patterns)
        self.assertEqual(result.diagnostics['pattern_chord_count'], 1)
        self.assertEqual(result.diagnostics['pattern_ids'], [0, 1])

    def test_single_node_source_side_is_supported_without_triangular_cells(self):
        _, back, bottom = _source()
        source = (_xyz([(0., 2.)]), back, bottom)
        result = build_repair_patch(source, (), .75, 1e-10)
        self.assert_patch(result, source)

    def test_invalid_patterns_reject_without_changing_source(self):
        cases = (
            (_pattern([(0., 1.), (2., 1.)]),),
            (_pattern([(0., .5), (4., .5)]),),
            (_pattern([(0., 1.), (4., 1.)]), _pattern([(2., 0.), (2., 2.)], 1)),
            (_pattern([(0., 1.), (4., 1.)]), _pattern([(2., 0.), (2., 1.)], 1)),
        )
        for patterns in cases:
            with self.subTest(patterns=patterns):
                source = _source()
                original = tuple(chain.copy() for chain in source)
                with self.assertRaises(QuadLayerError) as caught:
                    build_repair_patch(source, patterns, .75, 1e-10)
                self.assertEqual(caught.exception.code, 'unsupported_pattern_layout')
                for before, after in zip(original, source):
                    np.testing.assert_array_equal(after, before)

    def test_limit_rejects_before_sampling_and_accepts_exact_output_count(self):
        source = _source()
        with patch.object(_repair_patch, '_sample_edges') as sampler:
            with self.assertRaises(QuadLayerError) as caught:
                build_repair_patch(source, (), 1e-6, 1e-10, max_added_elements=2)
            self.assertEqual(caught.exception.code, 'element_limit_exceeded')
            sampler.assert_not_called()
        baseline = build_repair_patch(source, (), .75, 1e-10)
        result = build_repair_patch(source, (), .75, 1e-10, max_added_elements=len(baseline.quads))
        self.assert_patch(result, source)
        self.assertEqual(len(result.quads), len(baseline.quads))

    def test_exact_target_has_a_bounded_two_segment_refinement(self):
        source = (_xyz([(1., 1.), (0., 1.)]), _xyz([(0., 1.), (0., 0.)]),
                  _xyz([(0., 0.), (1., 0.)]))
        original_builder = _repair_patch._four_sided_units
        original_adaptive = _repair_adaptive.build_adaptive_face

        def needs_refinement(sides, *args, **kwargs):
            if sum(len(side) - 1 for side in sides) == 4:
                raise ValueError('the first boundary count has no usable patch')
            return original_builder(sides, *args, **kwargs)

        def adaptive_needs_refinement(boundary, *args, **kwargs):
            if len(boundary) == 4:
                raise ValueError('the first boundary count has no usable patch')
            return original_adaptive(boundary, *args, **kwargs)

        with patch.object(_repair_patch, '_four_sided_units', side_effect=needs_refinement), \
                patch.object(_repair_adaptive, 'build_adaptive_face',
                             side_effect=adaptive_needs_refinement):
            result = build_repair_patch(source, (), 1., 1e-10)
        self.assert_patch(result, source)
        self.assertEqual(len(result.mouth_ids) - 1, 3)

    def test_exhausted_candidates_report_the_face_and_bounded_causes(self):
        with patch.object(_repair_patch, '_four_sided_units',
                          side_effect=ValueError('fixed corner is degenerate')), \
                patch.object(_repair_adaptive, 'build_adaptive_face',
                             side_effect=ValueError('fixed corner is degenerate')):
            with self.assertRaises(QuadLayerError) as caught:
                build_repair_patch(_source(), (), .75, 1e-10)
        self.assertEqual(caught.exception.context['reason'], 'direct_quad_candidates_exhausted')
        failures = caught.exception.context['candidate_failures']
        self.assertTrue(0 < len(failures) <= 8)
        self.assertEqual(failures[0]['face_index'], 0)
        self.assertEqual(failures[0]['causes'], ['fixed corner is degenerate'])
        self.assertTrue(failures[0]['free_edge_counts'])


if __name__ == '__main__':
    unittest.main()
