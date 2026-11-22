"""Distance, area and cache regressions from the November adapter audit."""

import unittest
from unittest.mock import patch

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d.adapter import (QuadLayerError, generate_layer_adaptive_mesh,
                                  plan_quad_sections)
from mesher.mesh2d.adapter.layer._geometry import segment_contacts
from mesher.mesh2d.adapter.layer._validation import _edge_conflicts
from mesher.mesh2d.adapter.layer._construction import units


class SegmentContactTests(unittest.TestCase):
    def assert_contact(self, first, second, tol, expected):
        for first_side, second_side in ((first, second), (second, first)):
            for reverse_first, reverse_second in ((False, False), (True, False),
                                                   (False, True), (True, True)):
                a = first_side[::-1] if reverse_first else first_side
                b = second_side[::-1] if reverse_second else second_side
                scalar = segment_contacts(*a, *b, tol)
                batch = _edge_conflicts(np.array([0, 1]), a, np.array([[2, 3]]),
                                       b[None], tol)
                self.assertEqual(bool(scalar), expected)
                self.assertEqual(bool(batch[0]), expected)
                # Returned contacts belong to the first finite segment.
                for point in scalar:
                    delta = a[1] - a[0]
                    t = (point - a[0]) @ delta / (delta @ delta)
                    np.testing.assert_allclose(point, a[0] + np.clip(t, 0, 1)*delta,
                                               rtol=0, atol=1e-12)

    def test_euclidean_contact_is_invariant_under_order_frame_and_scale(self):
        first = np.array([[-1., 0.], [0., 0.]])
        cases = [(np.array([[.0008, .0008], [.0008, 1.]]), False),
                 (np.array([[-1., 1.0013], [1.0013, -1.]]), True)]
        angle = .37
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        for other, expected in cases:
            for scale in (.25, 1., 1024.):
                for matrix, offset in ((np.eye(2), np.zeros(2)),
                                       (rotation, np.array([4., -7.]))):
                    with self.subTest(expected=expected, scale=scale, offset=offset):
                        self.assert_contact((first @ matrix)*scale+offset,
                                            (other @ matrix)*scale+offset,
                                            .001*scale, expected)

    def test_exact_distance_threshold_and_adjacent_floats(self):
        first = np.array([[-1., 0.], [0., 0.]])
        for distance, expected in ((np.nextafter(.125, 0), True), (.125, True),
                                   (np.nextafter(.125, np.inf), False)):
            with self.subTest(distance=distance):
                self.assert_contact(first, np.array([[distance, -.5], [distance, .5]]),
                                    .125, expected)
                self.assert_contact(first, np.array([[-1., distance], [0., distance]]),
                                    .125, expected)
        # Endpoint distance is exactly 5/8, using a 3-4-5 triangle.
        other = np.array([[.375, .5], [.375, 2.]])
        for tol, expected in ((np.nextafter(.625, 0), False), (.625, True),
                              (np.nextafter(.625, np.inf), True)):
            self.assert_contact(first, other, tol, expected)
            # The closest point now lies inside a 3-4-5 sloping segment.
            self.assert_contact(first, np.array([[-1.625, 2.], [2.375, -1.]]),
                                tol, expected)

    def test_exact_and_shallow_intersections_survive(self):
        for tol in (0., 128*np.finfo(float).eps):
            self.assert_contact(np.array([[.4999995, 0.], [.5000005, 0.]]),
                                np.array([[0., -1e-8], [1., 1e-8]]), tol, True)
            self.assert_contact(np.array([[0., 0.], [1., 1.]]),
                                np.array([[0., 1.], [1., 0.]]), tol, True)
        self.assert_contact(np.array([[0., 0.], [1., 0.]]),
                            np.array([[1.0001, 0.], [2., 0.]]), 0., False)

    def test_topology_exemptions_do_not_allow_overlaps_or_hanging_nodes(self):
        edge = np.array([[0., 0.], [1., 0.]])
        others = np.array([[[0., 0.], [0., 1.]],  # shared endpoint
                           [[0., 0.], [.5, 0.]],  # shared endpoint + overlap
                           [[0., 0.], [1., 0.]],  # same topological edge
                           [[.5, 0.], [.5, 1.]],  # hanging node
                           [[0., 0.], [0., 1.]]]) # coordinate-only endpoint
        ids = np.array([[0, 2], [0, 3], [0, 1], [4, 5], [6, 7]])
        np.testing.assert_array_equal(_edge_conflicts(np.array([0, 1]), edge, ids, others, .001),
                                      [False, True, False, True, True])
        contacts = segment_contacts(*edge, *others[1], .001)
        self.assertEqual(len(contacts), 2)
        np.testing.assert_allclose(contacts, [[0, 0], [.5, 0]])


class LayerAuditTests(unittest.TestCase):
    def test_attaching_to_source_must_connect_incident_cells_in_one_fan(self):
        # The four strips form the square's external perimeter. Growing a
        # layer connects the strips around each corner through shared edges.
        points = [[0, 0], [4, 0], [4, 4], [0, 4],
                  [3.5, .2], [.5, .2], [3.8, 3.5], [3.8, .5],
                  [.5, 3.8], [3.5, 3.8], [.2, .5], [.2, 3.5],
                  [.8, .6], [1.2, 1.2], [.6, .8]]
        strips = [[0, 1, 4, 5], [1, 2, 6, 7], [2, 3, 8, 9], [3, 0, 10, 11]]
        mesh = Mesh2D(points, strips)
        generate_layer_adaptive_mesh(mesh, [0, 3, 2, 1], 1.)

        # This extra quad occupies the gap between strips. Its only
        # connection to the assembled domain would be the DOF at node 0.
        mesh = Mesh2D(points, strips + [[0, 12, 13, 14]])
        nodes, elements = mesh.nodes, mesh.elements
        with self.assertRaises(QuadLayerError) as caught:
            generate_layer_adaptive_mesh(mesh, [0, 3, 2, 1], 1.)
        self.assertEqual(caught.exception.code, "topology_change")
        self.assertEqual(caught.exception.context["parameter"], "outline_i")
        self.assertEqual(caught.exception.context["node_id"], 0)
        self.assertEqual(caught.exception.context["boundary_degree"], 4)
        self.assertIs(mesh.nodes, nodes)
        self.assertIs(mesh.elements, elements)

    def test_duplicate_target_cells_cannot_mask_a_point_only_connection(self):
        source = [[0, 0], [1, 0], [1, 1], [0, 1]]
        target = [[-1, -1], [-1, 0], [-1, 1], [-1, 2], [0, 2], [1, 2],
                  [2, 2], [2, 1], [2, 0], [2, -1], [1, -1], [0, -1]]
        for fixed in (False, True):
            with self.subTest(fixed=fixed):
                mesh = Mesh2D(source + target + [[3, 2], [3, 3], [2, 3]],
                              [[0, 1, 2, 3], [10, 16, 17, 18], [10, 16, 17, 18]])
                nodes, elements = mesh.nodes, mesh.elements
                with self.assertRaises(QuadLayerError) as caught:
                    generate_layer_adaptive_mesh(mesh, [0, 3, 2, 1], 1., outline_o=np.arange(4, 16),
                                                 fixed_outer_boundary=fixed)
                self.assertEqual(caught.exception.code, "topology_change")
                self.assertEqual(caught.exception.context["parameter"], "outline_o")
                self.assertEqual(caught.exception.context["node_id"], 10)
                self.assertIs(mesh.nodes, nodes)
                self.assertIs(mesh.elements, elements)

    def test_remote_cell_corner_jacobian_uses_physical_coordinates(self):
        source = [[.17, .63], [3.17, .63], [3.17, 3.63], [.17, 3.63]]
        distant_cells = (
            # Three collinear corners have zero corner Jacobian.
            [[1e6, 1e6 + 1], [1e6 + 1, 1e6 + 2], [1e6 + 2, 1e6 + 3], [1e6, 1e6 + 7]],
            # This corner is inverted by one ULP in the represented input.
            [[1001., 1000.], [1006., 1004.], [1026., np.nextafter(1020., 0.)], [1001., 1060.]],
        )
        for cell in distant_cells:
            for operation in (plan_quad_sections, generate_layer_adaptive_mesh):
                with self.subTest(cell=cell, operation=operation.__name__):
                    mesh = Mesh2D(source + cell, [[0, 1, 2, 3], [4, 5, 6, 7]])
                    nodes, elements = mesh.nodes, mesh.elements
                    with self.assertRaisesRegex(QuadLayerError, "strictly convex") as caught:
                        operation(mesh, [0, 3, 2, 1], 1.)
                    self.assertEqual(caught.exception.context["element_index"], 1)
                    self.assertIs(mesh.nodes, nodes)
                    self.assertIs(mesh.elements, elements)

    def test_separated_neighbor_is_accepted_and_close_diagonal_is_atomic(self):
        source = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        cases = [(np.array([[2.0008, 2.0008], [3., 2.0008], [3., 3.], [2.0008, 3.]]), True),
                 (np.array([[3., 1.0013], [4., 2.0013], [2.0013, 4.], [1.0013, 3.]]), False)]
        for other, accepted in cases:
            with self.subTest(accepted=accepted):
                mesh = Mesh2D(np.vstack((source, other)), [[0,1,2,3], [4,5,6,7]])
                nodes, elements = mesh.nodes, mesh.elements
                saved_nodes, saved_elements = nodes.copy(), elements.copy()
                if accepted:
                    generate_layer_adaptive_mesh(mesh, [0,3,2,1], 1., tolerance=.001,
                                                 repair_slots=False)
                    np.testing.assert_array_equal(mesh.nodes[:len(nodes)], saved_nodes)
                    np.testing.assert_array_equal(mesh.elements[:len(elements)], saved_elements)
                else:
                    with self.assertRaises(QuadLayerError) as caught:
                        generate_layer_adaptive_mesh(mesh, [0,3,2,1], 1., tolerance=.001,
                                                     repair_slots=False)
                    self.assertEqual(caught.exception.code, "topology_change")
                    self.assertIs(mesh.nodes, nodes)
                    self.assertIs(mesh.elements, elements)
                    np.testing.assert_array_equal(nodes, saved_nodes)
                    np.testing.assert_array_equal(elements, saved_elements)

    def test_positive_thin_layer_passes_preview_and_construction(self):
        for scale in (.25, 1., 1024.):
            with self.subTest(scale=scale):
                mesh = Mesh2D(np.array([[0,0],[.01,0],[.01,.01],[0,.01]])*scale,
                              [[0,1,2,3]])
                nodes, elements = mesh.nodes, mesh.elements
                plan_quad_sections(mesh, [0,3,2,1], .004*scale, tolerance=.001*scale,
                                   repair_slots=False)
                self.assertIs(mesh.nodes, nodes)
                self.assertIs(mesh.elements, elements)
                result = generate_layer_adaptive_mesh(mesh, [0,3,2,1], .004*scale,
                                                      tolerance=.001*scale, repair_slots=False)
                from tests.unit.mesh2d.test_extend_quad_layer import ExtendQuadLayerTests
                ExtendQuadLayerTests().assert_layer(mesh, [0,3,2,1], result, nodes, elements)
                cells = mesh.nodes[mesh.elements[1:], :2]
                self.assertGreater(np.linalg.norm(np.roll(cells,-1,axis=1)-cells, axis=2).min(),
                                   .001*scale)


class UnitCacheAuditTests(unittest.TestCase):
    @staticmethod
    def sides(scale=1.):
        x = np.array([0,.0625,.125,.25,.375,.5,.625,.75,.8125,.875,.9375,1.])
        return tuple(points*scale for points in (
            np.column_stack((x,np.zeros(len(x)))), np.array([[1.,0],[1,1]]),
            np.array([[1.,1],[0,1]]), np.array([[0.,1],[0,0]])))

    def test_scale_and_tolerance_do_not_reuse_an_invalid_unit(self):
        for reverse in (False, True):
            cache = {}
            calls = [(1., .01, True), (.25, .01, False), (1., .04, False)]
            for scale, tol, accepted in calls[::-1] if reverse else calls:
                with self.subTest(scale=scale, tol=tol, reverse=reverse):
                    size_before = len(cache)
                    if accepted:
                        units.build_unit_patch(self.sides(scale), tol, _cache=cache)
                    else:
                        with self.assertRaises(ValueError):
                            units.build_unit_patch(self.sides(scale), tol, _cache=cache)
                        self.assertEqual(len(cache), size_before)

    def test_equal_effective_tolerance_keeps_scale_and_translation_reuse(self):
        cache = {}
        units.build_unit_patch(self.sides(), .01, _cache=cache)
        sides = tuple(side + [4., 2.] for side in self.sides(.25))
        with patch.object(units, "build_unit_1_1_1_x", wraps=units.build_unit_1_1_1_x) as builder:
            cached = units.build_unit_patch(sides, .0025, _cache=cache)
        builder.assert_not_called()
        direct = units.build_unit_patch(sides, .0025)
        self.assertEqual(len(cache), 1)
        np.testing.assert_array_equal(cached.nodes, direct.nodes)
        np.testing.assert_array_equal(cached.quads, direct.quads)

    def test_invalid_cached_geometry_falls_back_without_modifying_entry(self):
        cache = {}
        expected = units.build_unit_patch(self.sides(), .01, _cache=cache)
        raw_nodes, quads, method = next(iter(cache.values()))
        raw_nodes[-1] = raw_nodes[-2]  # Exercise result validation on a cache hit.
        saved = raw_nodes.copy()
        actual = units.build_unit_patch(self.sides(), .01, _cache=cache)
        np.testing.assert_array_equal(raw_nodes, saved)
        np.testing.assert_array_equal(actual.nodes, expected.nodes)
        np.testing.assert_array_equal(actual.quads, expected.quads)


if __name__ == "__main__":
    unittest.main()
