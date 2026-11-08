"""Shared geometry policy across projection, identity, repair and validation."""
import unittest

import numpy as np

from mesher import Mesh2D
from mesher.tolerance import GeometryTolerance, GeometryToleranceError
from mesher.mesh2d.circular import extend_circular_mesh
from mesher.mesh2d.circular.imprint_v2 import imprint_circle
from mesher.mesh2d.circular.imprint_v2.utils import _to_circle
from mesher.mesh2d.circular.utils.pattern_segments import _coerce_pattern_guides
from mesher.mesh2d.circular.utils.topology import _search_circle
from mesher.mesh2d.quality import MeshQualityChecker, QualityReason


class CircularTolerancePolicyTests(unittest.TestCase):
    def test_near_flat_corner_uses_same_repair_on_both_sides_of_zero(self):
        for height in (0., -1e-14, 1e-14, -1e-12, 1e-12, .0005):
            with self.subTest(height=height):
                mesh = Mesh2D([[0., 0.], [1., 0.], [2., height], [0., 1.]], [[0, 1, 2, 3]])
                imprint_circle(mesh, center=[0., 0.], radius=.5,
                               projection_distance=.001, minimum_area=1e-6)
                self.assertEqual(mesh.element_count, 3)
                self.assertEqual(MeshQualityChecker(mesh).calculate_jacobian().summary.invalid_count, 0)
                points = mesh.nodes[mesh.elements, :2]
                relative = points - points[:, :1]
                after = np.roll(relative, -1, axis=1)
                area = .5 * np.sum(relative[..., 0] * after[..., 1] - relative[..., 1] * after[..., 0])
                self.assertAlmostEqual(area, 1. + .5 * height)

    def test_extension_accepts_radius_residual_within_shared_policy(self):
        angles = np.arange(8) * np.pi / 4
        ring = np.arange(1, 9)
        elements = [[0, a, b, b] for a, b in zip(ring, np.roll(ring, -1))]
        for radius in (.9995, 1., 1.0005):
            with self.subTest(radius=radius):
                xy = radius * np.column_stack((np.cos(angles), np.sin(angles)))
                mesh = Mesh2D(np.vstack(([0., 0.], xy)), elements)
                extend_circular_mesh(mesh, element_size=.1, center_x=0., center_y=0.,
                                     inner_radius=1., outer_radius=1.2)
                self.assertGreater(mesh.element_count, 8)

    def test_guides_normalize_near_axis_and_parallel_supports_once(self):
        guides = [[[.5, -2.], [.5000001, 2.]], [[.5001, -2.], [.5001, 2.]]]
        prepared = _coerce_pattern_guides(guides, coordinate_scale=1., tolerance=.001)
        np.testing.assert_array_equal(prepared.fixed_values, [.5, .5])
        mesh = Mesh2D([[.50005, .8655]], np.empty((0, 4), dtype=np.int32))
        _to_circle(mesh, [0., 0.], 1., .001, .001, prepared)
        np.testing.assert_allclose(mesh.nodes[0, :2], [.5, np.sqrt(.75)])

    def test_prepared_guides_do_not_silently_reuse_another_policy(self):
        prepared = _coerce_pattern_guides([[[.5, -2.], [.5, 2.]]], coordinate_scale=1., tolerance=.001)
        rebuilt = _coerce_pattern_guides(prepared, coordinate_scale=1., tolerance=.0001)
        self.assertIsNot(prepared, rebuilt)
        self.assertEqual(rebuilt.policy.absolute, .0001)

    def test_cached_guides_revalidate_original_geometry_under_stricter_policy(self):
        prepared = _coerce_pattern_guides([[[.5, -2.], [.5001, 2.]]], coordinate_scale=1., tolerance=.001)
        with self.assertRaisesRegex(ValueError, 'horizontal or vertical'):
            _coerce_pattern_guides(prepared, coordinate_scale=1., tolerance=.00001)

    def test_guide_normalization_cannot_accumulate_two_displacement_budgets(self):
        guides = [[[0., -2.], [0., 2.]], [[.0009, -2.], [.0018, 2.]]]
        with self.assertRaises(GeometryToleranceError) as error:
            _coerce_pattern_guides(guides, coordinate_scale=1., tolerance=.001)
        self.assertEqual(error.exception.code, 'ambiguous_geometry')

    def test_projection_reach_is_not_node_equivalence(self):
        angles = np.array([0., .01])
        source = 1.01 * np.column_stack((np.cos(angles), np.sin(angles)))
        mesh = Mesh2D(source, np.empty((0, 4), dtype=np.int32))
        _to_circle(mesh, [0., 0.], 1., .1, .001, None)
        self.assertGreater(np.linalg.norm(mesh.nodes[1, :2] - mesh.nodes[0, :2]), .001)
        np.testing.assert_allclose(np.linalg.norm(mesh.nodes[:, :2], axis=1), 1.)

    def test_guided_projection_cannot_exceed_reach(self):
        mesh = Mesh2D([[.9995, .02]], np.empty((0, 4), dtype=np.int32))
        original = mesh.nodes.copy()
        _to_circle(mesh, [0., 0.], 1., .001, .001, [[[.9995, -.1], [.9995, .1]]])
        np.testing.assert_array_equal(mesh.nodes, original)

    def test_large_origin_cannot_expand_projection_reach(self):
        center = np.array([1e12, 1e12])
        mesh = Mesh2D([center + [1.01, 0.]], np.empty((0, 4), dtype=np.int32))
        original = mesh.nodes.copy()
        _to_circle(mesh, center, 1., .001, .001, None)
        np.testing.assert_array_equal(mesh.nodes, original)

    def test_encoded_projection_distance_is_inclusive_without_global_slack(self):
        for offset, moved in ((.125, True), (np.nextafter(.125, np.inf), False)):
            with self.subTest(offset=offset):
                mesh = Mesh2D([[0., offset]], np.empty((0, 4), dtype=np.int32))
                # Center and radius make the target exactly [0, 0].
                _to_circle(mesh, [0., -1.], 1., .125, .001, None)
                self.assertEqual(mesh.nodes[0, 1] == 0., moved)

    def test_zero_projection_reach_cannot_move_nodes(self):
        mesh = Mesh2D([[1. + 1e-15, 0.]], np.empty((0, 4), dtype=np.int32))
        original = mesh.nodes.copy()
        _to_circle(mesh, [0., 0.], 1., 0., .001, None)
        np.testing.assert_array_equal(mesh.nodes, original)

    def test_transitive_projected_targets_raise_without_mutation(self):
        angles = np.array([0., .00075, .0015])
        mesh = Mesh2D(1.0001 * np.column_stack((np.cos(angles), np.sin(angles))),
                      np.empty((0, 4), dtype=np.int32))
        original = mesh.nodes.copy()
        with self.assertRaises(GeometryToleranceError) as caught:
            _to_circle(mesh, [0., 0.], 1., .01, .001, None)
        self.assertEqual(caught.exception.code, 'ambiguous_geometry')
        np.testing.assert_array_equal(mesh.nodes, original)

    def test_minimum_area_is_final_and_transactional(self):
        mesh = Mesh2D([[0., 0.], [1., 0.], [1., 1.], [0., 1.]], [[0, 1, 2, 3]])
        nodes, elements = mesh.nodes, mesh.elements
        with self.assertRaises(GeometryToleranceError) as caught:
            imprint_circle(mesh, center=[0., 0.], radius=.01,
                           projection_distance=.001, minimum_area=.001)
        self.assertEqual(caught.exception.code, 'minimum_area_conflict')
        self.assertIs(mesh.nodes, nodes)
        self.assertIs(mesh.elements, elements)

    def test_minimum_area_above_one_is_not_capped_during_cleanup(self):
        mesh = Mesh2D([[0., 0.], [1., 0.], [1., 1.], [0., 1.]], [[0, 1, 2, 3]])
        nodes, elements = mesh.nodes, mesh.elements
        with self.assertRaises(GeometryToleranceError) as caught:
            imprint_circle(mesh, center=[0., 0.], radius=.5,
                           projection_distance=0., minimum_area=2.)
        self.assertEqual(caught.exception.code, 'minimum_area_conflict')
        self.assertEqual(caught.exception.context['source_areas'], [1.])
        self.assertIs(mesh.nodes, nodes)
        self.assertIs(mesh.elements, elements)

    def test_circle_search_default_is_same_policy(self):
        mesh = Mesh2D([[1.005, 0.], [1.006, 0.], [1.005, .001]], [[0, 1, 2, 2]])
        self.assertEqual(len(_search_circle(mesh, 0., 0., 1., type='ALL')), 0)

    def test_circle_selection_respects_encoded_boundary_at_large_origins(self):
        from fractions import Fraction
        for origin in (0., 1e9, -1e9):
            edge = origin + 1. + .001
            for x in (np.nextafter(edge, -np.inf), edge, np.nextafter(edge, np.inf)):
                with self.subTest(origin=origin, x=x):
                    mesh = Mesh2D([[x, 0.]], [[0, 0, 0, 0]])
                    selected = _search_circle(mesh, origin, 0., 1.)
                    expected = abs(Fraction(float(x)) - Fraction(origin)) <= Fraction(1.) + Fraction(.001)
                    self.assertEqual(bool(len(selected)), expected)

    def test_local_projection_error_reports_original_global_node_ids(self):
        angles = np.array([0., .00075, .0015])
        local = 1.0001 * np.column_stack((np.cos(angles), np.sin(angles)))
        remote = np.array([[10., 10.], [11., 10.], [11., 11.], [10., 11.]])
        mesh = Mesh2D(np.vstack((remote, local)), [[0, 1, 2, 3], [4, 5, 6, 6]])
        original = mesh.nodes.copy()
        with self.assertRaises(GeometryToleranceError) as caught:
            imprint_circle(mesh, center=[0., 0.], radius=1., projection_distance=.01,
                           minimum_area=0.)
        self.assertEqual(set(caught.exception.context['node_indices']), {4, 5, 6})
        self.assertEqual(caught.exception.context['source'], 'circle projection')
        np.testing.assert_array_equal(mesh.nodes, original)

    def test_distinct_supplied_fixed_anchor_values_conflict(self):
        from mesher.mesh2d.circular.imprint.pattern import _add_pattern_anchor
        records = []
        arguments = dict(fixed_axis=0, center=np.zeros(2), radius=1.,
                         linear_tolerance=.001, fit_tolerance=.001)
        _add_pattern_anchor(records, point=np.array([1., 0.]), fixed_value=1., **arguments)
        with self.assertRaises(GeometryToleranceError) as caught:
            _add_pattern_anchor(records, point=np.array([1.0005, 0.]), fixed_value=1.0005, **arguments)
        self.assertEqual(caught.exception.code, 'fixed_constraint_conflict')

    def test_equivalent_derived_roots_use_stable_owner(self):
        from mesher.mesh2d.circular.imprint_v2.utils import _merge_close_circle_targets
        roots = np.array([[1., 0.], [1., np.spacing(1.)]])
        for points in (roots, roots[::-1]):
            merged = _merge_close_circle_targets(points, np.ones(2, dtype=bool),
                                                 np.zeros(2), .001, 1e-14)
            np.testing.assert_array_equal(merged, [[1., 0.], [1., 0.]])

    def test_quality_uses_distance_to_detect_long_flat_triangle(self):
        from mesher.mesh2d.quality import GeometryTolerance as QualityTolerance
        self.assertIs(QualityTolerance, GeometryTolerance)
        mesh = Mesh2D([[0., 0.], [1000., 0.], [500., .0005]], [[0, 1, 2, 2]])
        result = MeshQualityChecker(mesh).calculate_jacobian()
        self.assertTrue(result.reason_flags[0] & QualityReason.DEGENERATE)
        with self.assertRaises(TypeError):
            GeometryTolerance(absolute=.001, relative=1e-12)

    def test_removed_legacy_identity_arguments_fail_explicitly(self):
        mesh = Mesh2D([[0., 0.], [1., 0.], [0., 1.]], [[0, 1, 2, 2]])
        for name in ('projection_tolerance', 'guide_tolerance', 'merge_tolerance'):
            with self.subTest(name=name), self.assertRaises(TypeError):
                imprint_circle(mesh, center=[0., 0.], radius=.5,
                               projection_distance=.001, minimum_area=0., **{name: .001})
