import unittest

import numpy as np

from mesher.mesh2d.generators._holed_geometry import (
    Interval,
    OrthogonalDomain,
    normalize_patterns,
)


def box(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


class OrthogonalDomainTests(unittest.TestCase):
    def test_normalization_preserves_anchors_and_input_hole_owners(self):
        outer = [[6, 6], [6, 0], [3, 0], [0, 0], [0, 6], [6, 6]]
        domain = OrthogonalDomain(outer, [box(3, 3, 5, 5), box(1, 1, 2, 2)])
        np.testing.assert_array_equal(domain.loops[0], [[0, 0], [3, 0], [6, 0], [6, 6], [0, 6]])
        np.testing.assert_array_equal(domain.loops[1], [[3, 3], [3, 5], [5, 5], [5, 3]])
        self.assertEqual(domain.bounds, (0, 0, 6, 6))
        self.assertEqual(domain.area, 31)
        self.assertEqual(domain.boundary_owner([4, 3]), 1)
        self.assertEqual(domain.boundary_owner([1, 1.5]), 2)
        self.assertFalse(domain.loops[0].flags.writeable)

    def test_horizontal_vertical_visibility_and_cache(self):
        domain = OrthogonalDomain(box(0, 0, 10, 8), [box(3, 2, 5, 6)])
        self.assertEqual(domain.intervals(0, 4), (Interval(0, 3, 0, 1), Interval(5, 10, 1, 0)))
        self.assertEqual(domain.intervals(1, 4), (Interval(0, 2, 0, 1), Interval(6, 8, 1, 0)))
        self.assertIs(domain.intervals(0, 4), domain.intervals(0, 4))
        self.assertEqual(domain.intervals(0, -1), ())
        self.assertEqual(domain.intervals(1, 11), ())

    def test_collinear_boundary_overlaps_are_excluded(self):
        domain = OrthogonalDomain(box(0, 0, 10, 8), [box(3, 2, 5, 6)])
        for y in (2, 6):
            self.assertEqual(domain.intervals(0, y), (Interval(0, 3, 0, 1), Interval(5, 10, 1, 0)))
        for x in (3, 5):
            self.assertEqual(domain.intervals(1, x), (Interval(0, 2, 0, 1), Interval(6, 8, 1, 0)))
        self.assertEqual(domain.intervals(0, 0), ())
        self.assertEqual(domain.intervals(1, 10), ())

    def test_concave_outer_and_concave_hole_visibility(self):
        outer = [[0, 0], [8, 0], [8, 3], [5, 3], [5, 8], [0, 8]]
        hole = [[1, 1], [4, 1], [4, 2], [2, 2], [2, 5], [1, 5]]
        domain = OrthogonalDomain(outer, [hole])
        self.assertEqual(domain.intervals(0, 4), (Interval(0, 1, 0, 1), Interval(2, 5, 1, 0)))
        self.assertEqual(domain.intervals(0, 2), (Interval(0, 1, 0, 1), Interval(4, 8, 1, 0)))
        self.assertEqual(domain.intervals(0, 3), (Interval(0, 1, 0, 1), Interval(2, 5, 1, 0)))
        self.assertTrue(domain.contains([3, 4]))
        self.assertFalse(domain.contains([6, 4]))
        self.assertFalse(domain.contains([1.5, 4]))
        self.assertFalse(domain.contains([2, 4]))

    def test_disconnected_scanline_sections_keep_owner_information(self):
        outer = [[0, 0], [8, 0], [8, 8], [6, 8], [6, 2], [2, 2], [2, 8], [0, 8]]
        domain = OrthogonalDomain(outer)
        self.assertEqual(domain.intervals(0, 5), (Interval(0, 2, 0, 0), Interval(6, 8, 0, 0)))
        self.assertEqual(domain.intervals(0, 2), (Interval(0, 2, 0, 0), Interval(6, 8, 0, 0)))

    def test_strict_contains_and_exact_boundary_queries(self):
        domain = OrthogonalDomain(box(0, 0, 4, 4), [box(1, 1, 2, 2)], tolerance=0.01)
        self.assertTrue(domain.contains([0.001, 2]))
        self.assertIsNone(domain.boundary_owner([0.001, 2]))
        self.assertEqual(domain.boundary_owner([0, 2]), 0)
        self.assertFalse(domain.contains([0, 2]))
        self.assertFalse(domain.contains([1.5, 1.5]))

    def test_unresolved_collinear_anchors_are_rejected_without_modifying_input(self):
        outer = np.array([[0, 0], [0.0001, 0], [0.0002, 0], [4, 0], [4, 4], [0, 4]])
        original = outer.copy()
        with self.assertRaisesRegex(ValueError, "feature_below_resolution"):
            OrthogonalDomain(outer, tolerance=0.01)
        np.testing.assert_array_equal(outer, original)
        domain = OrthogonalDomain(outer, tolerance=1e-5)
        np.testing.assert_array_equal(domain.loops[0], outer)

    def test_similar_coordinate_levels_far_apart_are_allowed(self):
        domain = OrthogonalDomain(
            box(0, 0, 20, 20),
            [box(1, 1, 3, 3), box(10, 3.0001, 12, 5)],
            tolerance=0.01,
        )
        self.assertEqual(domain.loops[2][0, 1], 3.0)

    def test_physical_near_contacts_are_rejected(self):
        for holes in ([box(0.005, 1, 2, 3)], [box(1, 1, 3, 3), box(3.005, 1, 5, 3)]):
            with self.subTest(holes=holes), self.assertRaisesRegex(ValueError, "tolerance"):
                OrthogonalDomain(box(0, 0, 10, 10), holes, tolerance=0.01)
        with self.assertRaisesRegex(ValueError, "tolerance"):
            OrthogonalDomain(box(0, 0, 1, 0.005), tolerance=0.01)

    def test_invalid_loop_shapes_are_rejected(self):
        invalid = [
            [[0, 0], [1, 0], [1, 1]],
            [[0, 0], [2, 0], [2, 2], [1, 1], [0, 2]],
            [[0, 0], [2, 0], [2, 0], [2, 2], [0, 2]],
            [[0, 0], [2, 0], [1, 0], [1, 2], [0, 2]],
            [[0, 0], [4, 0], [4, 4], [2, 4], [2, -1], [1, -1], [1, 3], [0, 3]],
            [[0, 0], [4, 0], [4, 4], [2, 4], [2, 0], [1, 0], [1, 3], [0, 3]],
        ]
        for loop in invalid:
            with self.subTest(loop=loop), self.assertRaises(ValueError):
                OrthogonalDomain(loop)

    def test_holes_must_be_strictly_interior_disjoint_and_not_nested(self):
        invalid = [
            [box(-2, 1, -1, 2)],
            [box(0, 1, 2, 2)],
            [box(1, 1, 4, 4), box(2, 2, 3, 3)],
            [box(1, 1, 4, 4), box(3, 3, 5, 5)],
            [box(1, 1, 3, 3), box(3, 3, 5, 5)],
        ]
        for holes in invalid:
            with self.subTest(holes=holes), self.assertRaises(ValueError):
                OrthogonalDomain(box(0, 0, 8, 8), holes)

    def test_area_is_stable_for_large_coordinate_origin(self):
        origin = 1e12
        domain = OrthogonalDomain(box(origin, origin, origin + 8, origin + 8),
                                  [box(origin + 2, origin + 2, origin + 4, origin + 4)])
        self.assertEqual(domain.area, 60)

    def test_tolerance_and_query_validation(self):
        for value in (-1, float("inf"), float("nan"), True, "bad"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                OrthogonalDomain(box(0, 0, 4, 4), tolerance=value)
        domain = OrthogonalDomain(box(0, 0, 4, 4))
        with self.assertRaises(ValueError):
            domain.intervals(2, 1)
        with self.assertRaises(ValueError):
            domain.intervals(0, float("nan"))
        with self.assertRaises(ValueError):
            domain.contains([1, 2, 3])

    def test_symbolic_sections_distinguish_adjacent_float_events(self):
        first = 0.3
        second = 0.1 + 0.2
        self.assertEqual(np.nextafter(first, np.inf), second)
        domain = OrthogonalDomain(
            box(0, 0, 2, 6),
            [box(first, 1, 1, 2), box(second, 3, 1, 4)],
        )
        self.assertEqual(domain.section_after(1, first),
                         (Interval(0, 1, 0, 1), Interval(2, 3, 1, 2), Interval(4, 6, 2, 0)))
        self.assertEqual(domain.section_after(1, second),
                         (Interval(0, 1, 0, 1), Interval(2, 3, 1, 2), Interval(4, 6, 2, 0)))
        self.assertEqual(domain.section_after(1, 1), (Interval(0, 6, 0, 0),))
        # Exact-boundary queries must still exclude the vertical hole edges.
        self.assertEqual(domain.intervals(1, 1),
                         (Interval(0, 1, 0, 1), Interval(2, 3, 1, 2), Interval(4, 6, 2, 0)))
        self.assertEqual(domain.section_after(1, 2), ())

    def test_incremental_sections_match_queries_with_skips_and_repeated_events(self):
        outer = [[0, 0], [8, 0], [8, 3], [5, 3], [5, 8], [0, 8]]
        hole = [[1, 1], [4, 1], [4, 2], [2, 2], [2, 5], [1, 5]]
        domain = OrthogonalDomain(outer, [hole])
        for axis in (0, 1):
            coordinates = [-1, 0, 0, 1, 1.5, 2, 4.5, 5, 8, 9]
            expected = [(coordinate, domain.section_after(axis, coordinate))
                        for coordinate in coordinates]
            self.assertEqual(list(domain.iter_sections_after(axis, coordinates)), expected)
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            list(domain.iter_sections_after(0, [2, 1]))
        with self.assertRaises(ValueError):
            domain.section_after(2, 0)
        with self.assertRaises(ValueError):
            list(domain.iter_sections_after(0, [float("nan")]))

    def test_spatial_validation_and_visibility_visit_only_candidates(self):
        holes = [box(1 + 2 * x, 1 + 2 * y, 2 + 2 * x, 2 + 2 * y)
                 for y in range(8) for x in range(8)]
        domain = OrthogonalDomain(box(0, 0, 18, 18), holes)
        # Separate boxes and adjacent edges need no exact distance tests.
        self.assertEqual(domain.stats["validation_pairs"], 0)
        self.assertEqual(len(domain.intervals(0, 1.5)), 9)
        self.assertEqual(domain.stats["interval_edges"], 18)
        self.assertEqual(domain.section_after(0, -1), ())
        self.assertEqual(domain.stats["section_edges"], 0)
        domain.intervals(0, 1.5)
        self.assertEqual(domain.stats["interval_cache_hits"], 1)

    def test_collinear_anchors_do_not_expand_the_geometry_index(self):
        outer = [[float(x), 0] for x in np.linspace(0, 10, 101)]
        outer.extend([[10, 10], [0, 10]])
        domain = OrthogonalDomain(outer)
        self.assertEqual(len(domain.loops[0]), 103)
        self.assertEqual(domain.intervals(0, 5), (Interval(0, 10, 0, 0),))
        self.assertEqual(domain.stats["interval_edges"], 2)
        self.assertEqual(domain.boundary_owner([0.1, 0]), 0)

    def test_spatial_validation_preserves_inclusive_tolerance_decision(self):
        outer = box(0, 0, 20, 20)
        with self.assertRaisesRegex(ValueError, "tolerance"):
            OrthogonalDomain(outer, [box(2, 2, 5, 5), box(6, 2, 9, 5)],
                             tolerance=1)
        domain = OrthogonalDomain(
            outer, [box(2, 2, 5, 5), box(np.nextafter(6., np.inf), 2, 9, 5)],
            tolerance=1,
        )
        self.assertEqual(len(domain.loops), 3)

    def test_numeric_envelope_rejects_before_overflowing_area_arithmetic(self):
        with np.errstate(over="raise", invalid="raise"):
            with self.assertRaisesRegex(ValueError, "area-accumulation numeric range"):
                OrthogonalDomain(box(0, 0, 1e154, 1e154), tolerance=1e140)
            maximum = np.finfo(float).max
            with self.assertRaisesRegex(ValueError, "numeric_resolution_exceeded"):
                OrthogonalDomain(box(-maximum, 0, maximum, 1))

    def test_numeric_envelope_includes_all_loops(self):
        # Each loop fits separately, while their combined weighted bound does not.
        OrthogonalDomain(box(0, 0, 3e153, 3e153), tolerance=1e140)
        OrthogonalDomain(box(0, 0, 2e153, 2e153), tolerance=1e140)
        with self.assertRaisesRegex(ValueError, "area-accumulation numeric range"):
            OrthogonalDomain(box(0, 0, 3e153, 3e153),
                             [box(5e152, 5e152, 2.5e153, 2.5e153)], tolerance=1e140)

    def test_numeric_envelope_allows_small_scales_and_large_origins(self):
        small = OrthogonalDomain(box(0, 0, 1e-150, 1e-150), tolerance=1e-155)
        self.assertAlmostEqual(small.area / 1e-300, 1)
        origin, extent = 1e150, 1e135
        domain = OrthogonalDomain(box(origin, origin, origin + extent, origin + extent), tolerance=8e134)
        actual_span = (origin + extent) - origin
        self.assertEqual(domain.area, actual_span * actual_span)


class PatternNormalizationTests(unittest.TestCase):
    def test_patterns_are_canonical_exact_and_deduplicated(self):
        patterns = normalize_patterns([[[4, 1], [0, 1]], [[0, 1], [4, 1]], [[2, 3], [2, 0]]])
        self.assertEqual(len(patterns), 2)
        np.testing.assert_array_equal(patterns[0], [[0, 1], [4, 1]])
        np.testing.assert_array_equal(patterns[1], [[2, 0], [2, 3]])
        self.assertFalse(patterns[0].flags.writeable)
        self.assertEqual(normalize_patterns([]), ())

    def test_invalid_patterns_are_rejected(self):
        for pattern in ([[0, 0], [0, 0]], [[0, 0], [1, 1]], [[0, 0, 0], [1, 0, 0]],
                        [[0, 0], [float("nan"), 0]]):
            with self.subTest(pattern=pattern), self.assertRaises(ValueError):
                normalize_patterns([pattern])


if __name__ == "__main__":
    unittest.main()
