"""Physical precision, canonical interfaces and arithmetic stability regressions."""
import unittest
from itertools import permutations

import numpy as np

from mesher import Mesh2D
from mesher.tolerance import GeometryTolerance, GeometryToleranceError
from mesher.mesh2d.editor import EditorMesh2D
from mesher.mesh2d.adapter import (QuadLayerError, generate_full_adaptive_mesh,
                                   generate_layer_adaptive_mesh, plan_quad_sections)
from mesher.mesh2d.adapter.layer._planning.source import _prepare
from mesher.mesh2d.adapter.layer._construction.sampling import line_count_bounds
from mesher.mesh2d.adapter.layer._geometry import on_segment
from mesher.mesh2d.adapter.layer._registry import _NodePool
from mesher.mesh2d.adapter.layer._construction.units import CoordinatePool
from mesher.mesh2d.generators.rectilinear import generate_rectilinear_mesh
from mesher.mesh2d.generators.holed_rectilinear import generate_rectilinear_mesh_optimal


def box(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


class PhysicalToleranceTests(unittest.TestCase):
    def test_policy_is_shared_and_length_tolerance_is_normalized(self):
        mesh = Mesh2D(box(0, 0, 10, 10), [[0, 1, 2, 3]])
        policy = GeometryTolerance(.001)
        data = _prepare(mesh, [0, 3, 2, 1], 1, (), 1, policy)
        self.assertIs(data["tolerance"], policy)
        self.assertEqual(data["tol"], .0001)
        self.assertLess(data["numeric_tol"], 1e-10)

    def test_prepared_plan_and_committed_layer_share_parsed_policy(self):
        from tests.unit.mesh2d.test_corner_repair import _step
        from dataclasses import FrozenInstanceError
        for supplied in (None, GeometryTolerance(.001)):
            with self.subTest(supplied=supplied):
                mesh, outline = _step()
                plan = plan_quad_sections(mesh, outline, 1., tolerance=supplied)
                self.assertIsNotNone(plan.prepared_source)
                self.assertIs(plan.prepared_source.tolerance, plan.tolerance)
                if supplied is not None:
                    self.assertIs(plan.tolerance, supplied)
                self.assertGreater(plan.tolerance.absolute, 0.)
                self.assertEqual(plan.diagnostics["geometry_tolerance"], plan.tolerance.absolute)
                budget = plan.diagnostics["numerical_error_budget"]
                self.assertGreater(budget["world_length"], 0.)
                self.assertLess(budget["world_length"], plan.tolerance.absolute)
                with self.assertRaises(FrozenInstanceError):
                    plan.tolerance = GeometryTolerance(.01)
                result = generate_layer_adaptive_mesh(mesh, outline, 1., tolerance=plan.tolerance)
                self.assertIs(result.tolerance, plan.tolerance)
                self.assertEqual(result.diagnostics["geometry_tolerance"], plan.tolerance.absolute)

    def test_axis_alignment_is_absolute_and_preserves_source(self):
        for extent, drift, accepted in ((.01, .0005, True), (1000, .5, False)):
            mesh = Mesh2D([[0, 0], [extent, 0], [extent, extent + drift], [0, extent]],
                          [[0, 1, 2, 3]])
            before = mesh.nodes.copy()
            with self.subTest(extent=extent):
                if accepted:
                    _prepare(mesh, [0, 3, 2, 1], extent, (), 1, .001)
                else:
                    with self.assertRaises(QuadLayerError):
                        _prepare(mesh, [0, 3, 2, 1], extent, (), 1, .001)
                np.testing.assert_array_equal(mesh.nodes, before)

    def test_dense_classification_is_translation_invariant(self):
        for offset in (0, 1, 10, 1e6):
            mesh = generate_rectilinear_mesh(1, np.array([0, .3, .6, .9]) + offset, [0, 1])
            outline = EditorMesh2D.outlines(mesh)[0]
            plan = plan_quad_sections(mesh, outline, 1, small_element_size=.3, repair_slots=False)
            self.assertEqual(plan.diagnostics["dense_seams"], [])
            self.assertEqual(plan.diagnostics["dense_sections"], [])

    def test_adjacent_float_supports_do_not_create_tiny_interface_edges(self):
        for transpose in (False, True):
            outer = box(0, 0, 5, 5)
            holes = [box(1, 1, np.nextafter(2., np.inf), 2), box(1, 3, 2, 4)]
            if transpose:
                outer, holes = outer[:, ::-1], [hole[:, ::-1] for hole in holes]
            originals = [value.copy() for value in (outer, *holes)]
            result = generate_rectilinear_mesh_optimal(1, outer, holes=holes)
            for ids in result.hole_outlines:
                points = result.mesh.nodes[ids, :2]
                self.assertGreater(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1).min(), .001)
                self.assertEqual(len(ids) % 2, 0)
            for original, actual in zip(originals, (outer, *holes)):
                np.testing.assert_array_equal(actual, original)

    def test_full_canonical_fronts_preserve_frozen_mesh_and_complete_interface(self):
        first = Mesh2D(box(0, 0, np.nextafter(2., np.inf), 1), [[0, 1, 2, 3]])
        second = Mesh2D(box(0, 3, 2, 4), [[0, 1, 2, 3]])
        mesh = EditorMesh2D.merge(first, second)
        original_nodes, original_elements = mesh.nodes.copy(), mesh.elements.copy()
        policy = GeometryTolerance(.001)
        result = generate_full_adaptive_mesh(mesh, [.5], outer_outline=box(-3, -3, 5, 7),
                                             tolerance=policy)
        self.assertIs(result.tolerance, policy)
        self.assertEqual(result.diagnostics["geometry_tolerance"], policy.absolute)
        np.testing.assert_array_equal(mesh.nodes[:len(original_nodes)], original_nodes)
        np.testing.assert_array_equal(mesh.elements[:len(original_elements)], original_elements)
        edges = np.stack((mesh.elements, np.roll(mesh.elements, -1, axis=1)), axis=-1).reshape(-1, 2)
        keys, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
        self.assertTrue(np.all(np.isin(counts, [1, 2])))
        actual = set(map(tuple, keys[counts == 1]))
        outline = result.outer_outline
        expected = {tuple(sorted((int(a), int(b)))) for a, b in zip(outline, np.roll(outline, -1))}
        self.assertEqual(actual, expected)
        points = mesh.nodes[mesh.elements, :2]
        vectors = np.roll(points, -1, axis=1) - points
        self.assertGreater(np.linalg.norm(vectors, axis=2).min(), .001)
        following = np.roll(vectors, -1, axis=1)
        self.assertTrue(np.all(vectors[..., 0] * following[..., 1] - vectors[..., 1] * following[..., 0] > 0))

    def test_stable_subdivision_count_and_ambiguous_chains(self):
        for endpoint in (.3, .1 + .2):
            mesh = generate_rectilinear_mesh(.1, [0, endpoint], [0, .1])
            self.assertEqual(mesh.element_count, 3)
        with self.assertRaises(GeometryToleranceError) as caught:
            generate_rectilinear_mesh(1, [0, .00075, .0015, 1], [0, 1])
        self.assertEqual(caught.exception.code, "ambiguous_geometry")

    def test_line_counts_do_not_spend_geometric_tolerance_on_rounding(self):
        self.assertEqual(line_count_bounds([1.0005], 1, 1., .001), (2, 2))
        self.assertEqual(line_count_bounds([.9995], 1, 1., .001), (1, 1))
        self.assertEqual(line_count_bounds([.1 + .2], 1, .1, .001), (3, 6))

    def test_segment_endpoint_contact_uses_euclidean_distance(self):
        start, end = np.array([0., 0.]), np.array([1., 0.])
        self.assertFalse(on_segment(np.array([1.0008, .0008]), start, end, .001))
        self.assertTrue(on_segment(np.array([1.0006, .0006]), start, end, .001))

    def test_subdivision_cannot_generate_unresolved_edges(self):
        with self.assertRaises(GeometryToleranceError) as caught:
            generate_rectilinear_mesh(.00101, [0, .0011], [0, .01])
        self.assertEqual(caught.exception.code, "feature_below_resolution")
        self.assertLess(caught.exception.context["measured_distance"], .001)

    def test_coordinate_pools_reject_chains_in_every_insertion_order(self):
        for points in permutations(([0., 0.], [.00075, 0.], [.0015, 0.])):
            for source_pool in (False, True):
                with self.subTest(points=points, source_pool=source_pool):
                    pool = (_NodePool(np.empty((0, 3)), [], .001) if source_pool
                            else CoordinatePool(.001))
                    add = pool.boundary_node if source_pool else pool.intern
                    with self.assertRaises(GeometryToleranceError) as caught:
                        for point in points:
                            add(np.asarray(point))
                    self.assertEqual(caught.exception.code, "ambiguous_geometry")

    def test_coordinate_pool_preserves_fixed_coordinates_and_rejects_bridges(self):
        original = np.array([[0., 0., 0.], [.0015, 0., 0.]])
        pool = _NodePool(original, [0, 1], .001)
        self.assertEqual(pool.boundary_node(np.array([.0001, 0.])), 0)
        np.testing.assert_array_equal(pool.array(), original)
        with self.assertRaises(GeometryToleranceError) as caught:
            pool.boundary_node(np.array([.00075, 0.]))
        self.assertEqual(caught.exception.code, "fixed_constraint_conflict")
        np.testing.assert_array_equal(pool.array(), original)

    def test_unrepresentable_spacing_and_subresolution_sizes_fail(self):
        with self.assertRaises(GeometryToleranceError) as caught:
            generate_rectilinear_mesh(1e-5, [1e12, 1e12 + .002], [0, .002], tolerance=1e-6)
        self.assertEqual(caught.exception.code, "numeric_resolution_exceeded")
        with self.assertRaises(GeometryToleranceError) as caught:
            generate_rectilinear_mesh(.001, [0, 1], [0, 1])
        self.assertEqual(caught.exception.code, "feature_below_resolution")

    def test_removed_public_keywords_are_not_accepted(self):
        with self.assertRaises(TypeError):
            generate_rectilinear_mesh(1, [0, 1], [0, 1], coordinate_tolerance=.001)
        mesh = Mesh2D(box(0, 0, 1, 1), [[0, 1, 2, 3]])
        with self.assertRaises(TypeError):
            plan_quad_sections(mesh, [0, 3, 2, 1], .5, axis_tolerance=.001)


if __name__ == "__main__":
    unittest.main()
