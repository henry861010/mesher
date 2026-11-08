"""Precision policy contracts independent of meshing algorithm choices."""

from fractions import Fraction
import unittest

import numpy as np

from mesher import GeometryTolerance, GeometryToleranceError, Mesh2D
from mesher.tolerance import (
    canonicalize_points, canonicalize_values, point_segment_distance,
    resolve_tolerance, subdivision_count, within_distance,
)
from mesher.mesh2d.editor import EditorMesh2D
from mesher.mesh2d.utils.equivalence import _bounded_equivalent_node_mapping, _component_diameter_exceeds


class TestGeometryTolerance(unittest.TestCase):
    def test_circle_boundaries_use_exact_radius_plus_distance(self):
        from fractions import Fraction
        from mesher.tolerance import circle_membership
        for origin in (0., 1e9, -1e9):
            center = [origin, -origin]
            for radius in (.001, 1., 100.):
                for sign in (-1, 1):
                    endpoint = origin + sign * (radius + .001)
                    for coordinate in (np.nextafter(endpoint, -np.inf), endpoint,
                                       np.nextafter(endpoint, np.inf)):
                        exact = abs(Fraction(float(coordinate)) - Fraction(origin))
                        residual = abs(exact - Fraction(radius))
                        self.assertEqual(circle_membership([coordinate, -origin], center, radius),
                                         exact <= Fraction(radius) + Fraction(.001))
                        self.assertEqual(circle_membership([coordinate, -origin], center, radius, boundary=True),
                                         residual <= Fraction(.001))

    def test_policy_is_positive_immutable_and_identity_preserving(self):
        policy = GeometryTolerance()
        self.assertEqual(policy.absolute, .001)
        self.assertIs(resolve_tolerance(policy), policy)
        self.assertEqual(resolve_tolerance(.002).absolute, .002)
        for value in (0, -1, True, np.bool_(False), np.nan, np.inf, ".001", 10**1000):
            with self.subTest(value=value), self.assertRaises(ValueError):
                GeometryTolerance(value)
        with self.assertRaises(AttributeError):
            policy.absolute = .1

    def test_inclusive_distance_at_adjacent_representable_values(self):
        limit = .001
        points = np.array([[np.nextafter(limit, 0), 0], [limit, 0],
                           [np.nextafter(limit, np.inf), 0]])
        np.testing.assert_array_equal(within_distance(points, [0, 0]), [True, True, False])
        self.assertFalse(within_distance([.0009, .0009], [0, 0]))
        self.assertTrue(within_distance([.0006, .0006], [0, 0]))

    def test_distance_agrees_with_exact_encoded_coordinates_near_large_origin(self):
        a = np.array([1e9, -1e9])
        b = a + [.0008, .0006]
        exact = sum((Fraction(float(x)) - Fraction(float(y))) ** 2 for x, y in zip(a, b))
        self.assertEqual(within_distance(a, b), exact <= Fraction(.001) ** 2)

    def test_scalar_canonicalization_is_order_independent_and_bounded(self):
        source = np.array([3., 1.0004, 1., 1.0008, 3.0005])
        expected = np.array([3., 1., 1., 1., 3.])
        np.testing.assert_array_equal(canonicalize_values(source), expected)
        order = np.array([4, 1, 0, 3, 2])
        np.testing.assert_array_equal(canonicalize_values(source[order]), expected[order])
        for values in ([0, .00075, .0015], [0, .0008, .0016]):
            with self.assertRaises(GeometryToleranceError) as raised:
                canonicalize_values(values)
            self.assertEqual(raised.exception.code, "ambiguous_geometry")

    def test_fixed_coordinates_win_and_conflicting_constraints_fail(self):
        np.testing.assert_array_equal(
            canonicalize_values([1., 1.0004], fixed=[False, True]), [1.0004, 1.0004])
        with self.assertRaises(GeometryToleranceError) as raised:
            canonicalize_values([1., 1.0004], fixed=[True, True])
        self.assertEqual(raised.exception.code, "fixed_constraint_conflict")

    def test_point_normalization_checks_total_displacement(self):
        with self.assertRaises(GeometryToleranceError) as raised:
            canonicalize_points([[0, 0], [.0009, .0009]])
        self.assertGreater(raised.exception.context["displacement"], .001)
        values = np.array([[0., 0.], [.0004, .0004]])
        result = canonicalize_points(values, fixed=[True, False])
        np.testing.assert_array_equal(result, [[0, 0], [0, 0]])
        np.testing.assert_array_equal(values[1], [.0004, .0004])

    def test_numeric_resolution_is_not_permission_to_enlarge_tolerance(self):
        GeometryTolerance().check_resolution([[1e7, 0]])
        with self.assertRaises(GeometryToleranceError) as raised:
            GeometryTolerance().check_resolution([[1e15, 0]])
        self.assertEqual(raised.exception.code, "numeric_resolution_exceeded")
        with np.errstate(all="raise"):
            GeometryTolerance(1e-150).check_resolution([[0, 1e-149]])

    def test_subdivision_count_uses_roundoff_not_geometry_slack(self):
        for length in (.3, .1 + .2):
            for mode in ("ceil", "floor"):
                self.assertEqual(subdivision_count(length, .1, rounding=mode), 3)
        self.assertEqual(subdivision_count(.300000001, .1, rounding="ceil"), 4)
        self.assertEqual(subdivision_count(.300000001, .1, rounding="floor"), 3)

    def test_point_segment_distance_uses_closed_endpoints(self):
        distances = point_segment_distance([[.5, .0005], [1.0006, .0008]], [0, 0], [1, 0])
        np.testing.assert_allclose(distances, [.0005, .001], rtol=1e-12)
        self.assertEqual(point_segment_distance([3, 4], [0, 0], [0, 0]), 5)

    def test_segment_membership_does_not_round_projected_coordinates(self):
        from fractions import Fraction
        from mesher.tolerance import within_segment
        for origin in (0., 1e9, -1e9):
            first = np.array([origin, origin])
            last = first + [3, 4]
            for position in (0., .5, 1.):
                for offset in (.0009999, .001, .0010001):
                    point = first + position * (last - first) + offset * np.array([.8, -.6])
                    q = [Fraction(float(p)) - Fraction(float(a)) for p, a in zip(point, first)]
                    v = [Fraction(float(b)) - Fraction(float(a)) for b, a in zip(last, first)]
                    length_squared = sum(x*x for x in v)
                    t = max(Fraction(0), min(Fraction(1), sum(x*y for x, y in zip(q, v)) / length_squared))
                    expected = sum((x-t*y)**2 for x, y in zip(q, v)) <= Fraction(.001)**2
                    self.assertEqual(within_segment(point, first, last), expected)
        self.assertTrue(within_segment([0, .001], [0, 0], [0, 0]))
        self.assertFalse(within_segment([0, np.nextafter(.001, np.inf)], [0, 0], [0, 0]))


class TestBoundedEquivalence(unittest.TestCase):
    def test_large_origin_groups_match_exact_pairwise_oracle(self):
        from fractions import Fraction
        from mesher.mesh2d.utils.equivalence import _bounded_equivalent_node_mapping
        random = np.random.default_rng(31)
        limit = Fraction(.001) ** 2
        for origin in (1e12, -1e12):
            for _ in range(20):
                xy = origin + random.uniform(-.003, .003, (8, 2))
                near = np.array([[sum((Fraction(float(a)) - Fraction(float(b))) ** 2
                                     for a, b in zip(first, second)) <= limit
                                  for second in xy] for first in xy])
                groups, remaining = [], set(range(len(xy)))
                while remaining:
                    group = {min(remaining)}
                    pending = list(group)
                    while pending:
                        additions = set(np.flatnonzero(near[pending.pop()])) - group
                        group.update(additions)
                        pending.extend(additions)
                    remaining.difference_update(group)
                    groups.append(sorted(group))
                ambiguous = any(not near[np.ix_(group, group)].all() for group in groups)
                nodes = np.column_stack((xy, np.zeros(len(xy))))
                if ambiguous:
                    with self.assertRaises(GeometryToleranceError):
                        _bounded_equivalent_node_mapping(nodes, np.arange(len(xy)), .001)
                else:
                    mapping = _bounded_equivalent_node_mapping(nodes, np.arange(len(xy)), .001)
                    self.assertEqual(len(set(mapping)), len(groups))
                    self.assertTrue(all(len(set(mapping[group])) == 1 for group in groups))

    def test_chains_fail_before_mutation(self):
        mesh = Mesh2D([[0, 0], [.00075, 0], [.0015, 0], [0, 1]], [[0, 1, 2, 3]])
        before = mesh.nodes.copy(), mesh.elements.copy()
        with self.assertRaises(GeometryToleranceError) as raised:
            EditorMesh2D.equivalence(mesh)
        self.assertEqual(raised.exception.code, "ambiguous_geometry")
        np.testing.assert_array_equal(mesh.nodes, before[0])
        np.testing.assert_array_equal(mesh.elements, before[1])

    def test_thin_feature_cannot_silently_disappear(self):
        mesh = Mesh2D([[0, 0], [1, 0], [1, .0005], [0, .0005]], [[0, 1, 2, 3]])
        with self.assertRaises(GeometryToleranceError) as raised:
            EditorMesh2D.equivalence(mesh)
        self.assertEqual(raised.exception.code, "feature_below_resolution")
        self.assertEqual(mesh.element_count, 1)

    def test_representative_coordinate_independent_of_input_order(self):
        nodes = np.array([[.0005, 0], [0, 0], [.0004, .0001]])
        mapping = _bounded_equivalent_node_mapping(nodes, np.arange(3), .001)
        np.testing.assert_array_equal(nodes[mapping], np.zeros((3, 2)))

    def test_hull_diameter_against_small_exact_oracle(self):
        rng = np.random.default_rng(413)
        for count in (2, 3, 10, 40):
            for _ in range(20):
                points = rng.uniform(-.0006, .0006, (count, 2))
                expected = any(not within_distance(a, b) for a in points for b in points)
                self.assertEqual(_component_diameter_exceeds(points, GeometryTolerance()), expected)

    def test_rectangle_distance_is_euclidean_and_tolerant(self):
        mesh = Mesh2D([[0, 0], [1.0005, 0], [1.0005, 1], [0, 1]], [[0, 1, 2, 3]])
        np.testing.assert_array_equal(EditorMesh2D.search_element(mesh, 0, 0, 1, 1), [0])
        mesh.nodes[2, :2] = [1.0009, 1.0009]
        self.assertEqual(len(EditorMesh2D.search_element(mesh, 0, 0, 1, 1)), 0)

    def test_planar_input_adapter_is_explicit_and_bounded(self):
        values = np.array([[0, 0, .0005], [1, 0, 0], [0, 1, 0]])
        mesh = Mesh2D.from_planar(values, [[0, 1, 2, 2]])
        np.testing.assert_array_equal(mesh.nodes[:, 2], 0)
        self.assertEqual(values[0, 2], .0005)
        with self.assertRaises(ValueError):
            Mesh2D(values, [[0, 1, 2, 2]])
        values[0, 2] = .0011
        with self.assertRaises(GeometryToleranceError):
            Mesh2D.from_planar(values, [[0, 1, 2, 2]])
