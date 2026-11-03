"""Bent free repair boundaries preserve fixed source chains and Quad4 parity."""

from collections import Counter
import unittest

import numpy as np

from mesher.mesh2d.adapter.layer._geometry import intervals_cover
from mesher.mesh2d.adapter.layer._patterns import covered_intervals, occupied_intervals
from mesher.mesh2d.adapter.layer._repair_patch import build_repair_patch
from mesher.mesh2d.adapter.layer._types import QuadLayerError
from mesher.mesh2d.adapter.unit.build_unit_1_1_x_y import _validate_patch
from tests.unit.mesh2d.test_repair_patch import _pattern, _xyz


def _corner():
    source = (_xyz([(0, 0), (.5, 0), (1.25, 0), (2, 0)]),
              _xyz([(2, 0), (2, .5), (2, 1), (2, 2), (2, 3)]))
    return source, _xyz([(2, 3), (0, 3), (0, 0)])


class CornerRepairPatchTests(unittest.TestCase):
    def assert_patch(self, result, source, free_boundary, patterns=()):
        frozen = np.vstack((source[0], *(chain[1:] for chain in source[1:])))
        np.testing.assert_array_equal(result.nodes[result.source_ids], frozen)
        mouth = result.nodes[result.mouth_ids]
        np.testing.assert_array_equal(mouth[[0, -1]], frozen[[-1, 0]])
        positions = []
        for point in free_boundary:
            matches = np.flatnonzero(np.all(mouth == point, axis=1))
            self.assertEqual(len(matches), 1)
            positions.append(int(matches[0]))
        self.assertEqual(positions, sorted(positions))
        self.assertEqual((len(result.source_ids) + len(result.mouth_ids) - 2) % 2, 0)
        _validate_patch(result.nodes, result.quads, (result.source_ids, result.mouth_ids))
        self.assertEqual(set(result.quads.ravel()), set(range(len(result.nodes))))
        incidence = Counter(tuple(sorted((int(a), int(b)))) for quad in result.quads
                            for a, b in zip(quad, np.roll(quad, -1)))
        edges = result.nodes[np.asarray(list(incidence)), :2]
        for pattern in patterns:
            required = occupied_intervals(pattern, result.nodes[result.quads, :2], 1e-10)
            self.assertTrue(intervals_cover(required, covered_intervals(pattern, edges, 1e-10), 1e-10))

    def test_two_frozen_legs_with_unequal_counts_keep_cap_knee_and_even_parity(self):
        source, boundary = _corner()
        for source, boundary in ((source, boundary),
                                 (tuple(chain[::-1] for chain in source[::-1]), boundary[::-1])):
            with self.subTest(winding=source[0][0].tolist()):
                snapshot = tuple(chain.copy() for chain in source)
                original_boundary = boundary.copy()
                result = build_repair_patch(source, (), 1., 1e-10, free_boundary=boundary)
                self.assert_patch(result, source, boundary)
                for before, after in zip(snapshot, source):
                    np.testing.assert_array_equal(before, after)
                np.testing.assert_array_equal(boundary, original_boundary)
                # Seven immutable source segments require an odd total on the
                # two free arms; neither source leg may be resampled to fix it.
                self.assertEqual((len(result.mouth_ids) - 1) % 2, 1)

    def test_near_axis_source_bends_remain_exact(self):
        source = (_xyz([(0, 0), (.5, .0001), (1, 0)]),
                  _xyz([(1, 0), (1.0001, .5), (1, 1)]))
        boundary = _xyz([(1, 1), (0, 1), (0, 0)])
        result = build_repair_patch(source, (), 1., 1e-10, free_boundary=boundary)
        self.assert_patch(result, source, boundary)

    def test_complete_chords_can_end_on_either_free_arm(self):
        source, boundary = _corner()
        for pattern, contact in ((_pattern([(0, 2), (2, 2)]), [0, 2, 0]),
                                 (_pattern([(.5, 0), (.5, 3)]), [.5, 3, 0])):
            with self.subTest(contact=contact):
                result = build_repair_patch(source, (pattern,), 1., 1e-10,
                                            free_boundary=boundary)
                self.assert_patch(result, source, boundary, (pattern,))
                self.assertEqual(result.diagnostics['face_count'], 2)
                mouth = result.nodes[result.mouth_ids]
                self.assertEqual(np.count_nonzero(np.all(mouth == contact, axis=1)), 1)

    def test_finite_boundary_patterns_keep_contacts_on_both_arms(self):
        source, boundary = _corner()
        patterns = (_pattern([(.25, 3), (1.5, 3)]),
                    _pattern([(0, .25), (0, 2.5)], 1))
        result = build_repair_patch(source, patterns, 1., 1e-10, free_boundary=boundary)
        self.assert_patch(result, source, boundary, patterns)
        mouth = result.nodes[result.mouth_ids]
        for point in ((.25, 3, 0), (1.5, 3, 0), (0, .25, 0), (0, 2.5, 0)):
            self.assertEqual(np.count_nonzero(np.all(mouth == point, axis=1)), 1)

    def test_patterns_cannot_split_a_frozen_leg_or_cross_inside_the_cap(self):
        source, boundary = _corner()
        cases = ((_pattern([(0, .75), (2, .75)]),),
                 (_pattern([(.75, 0), (.75, 3)]),),
                 (_pattern([(0, 2), (2, 2)]), _pattern([(.5, 0), (.5, 3)], 1)))
        for patterns in cases:
            with self.subTest(patterns=patterns):
                with self.assertRaises(QuadLayerError) as caught:
                    build_repair_patch(source, patterns, 1., 1e-10, free_boundary=boundary)
                self.assertEqual(caught.exception.code, 'unsupported_pattern_layout')

    def test_explicit_boundary_checks_exact_endpoints_and_entire_closed_polygon(self):
        source, boundary = _corner()
        cases = (boundary[::-1],
                 _xyz([(2, 3), (0, 3), (0, 3), (0, 0)]),
                 _xyz([(2, 3), (-1, 3), (-1, -1), (1, -1), (1, 1), (0, 0)]))
        for invalid in cases:
            with self.subTest(boundary=invalid.tolist()):
                with self.assertRaises(QuadLayerError):
                    build_repair_patch(source, (), 1., 1e-10, free_boundary=invalid)
        with self.assertRaises(QuadLayerError):
            build_repair_patch(source, (), 1., 1e-10)


if __name__ == '__main__':
    unittest.main()
