"""Boundary classification and global-domain partition for full adaptation."""
import unittest

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d import EditorMesh2D
from mesher.mesh2d.adapter import QuadLayerError, generate_full_adaptive_mesh, plan_quad_sections
from mesher.mesh2d.adapter._full_geometry import (
    _validate_mesh, complement_domains, discover_boundaries, flatten_outline,
    section_outline_points, validate_fronts,
)


def _rectangle(x0=0., y0=0., x1=4., y1=3.):
    return Mesh2D([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], [[0, 1, 2, 3]])


def _ring():
    values = [0., 2., 8., 10.]
    return Mesh2D([[x, y] for y in values for x in values],
                  [[4*y+x, 4*y+x+1, 4*y+x+5, 4*y+x+4]
                   for y in range(3) for x in range(3) if (x, y) != (1, 1)])


def _outer():
    return np.array([[-5., -5.], [15., -5.], [15., 15.], [-5., 15.]])


def _fronts(mesh, topology):
    return [mesh.nodes[flatten_outline(outline), :2] for outline in topology.outlines]


class FullGeometryTests(unittest.TestCase):
    def test_nested_material_and_void_domains_partition_the_complement(self):
        mesh = EditorMesh2D.merge(_ring(), _rectangle(4., 4., 6., 6.))
        topology = discover_boundaries(mesh, _outer())
        self.assertEqual(topology.signs, (-1, 1, -1))
        self.assertEqual(topology.parents, (None, 0, 1))
        fronts = _fronts(mesh, topology)
        validate_fronts(topology, fronts)
        domains = complement_domains(topology, fronts)
        self.assertEqual([domain.boundary_index for domain in domains], [None, 1])
        self.assertEqual([domain.hole_boundary_indices for domain in domains], [(0,), (2,)])
        np.testing.assert_array_equal(domains[1].outer_outline, fronts[1])
        np.testing.assert_array_equal(domains[1].holes[0], fronts[2])

    def test_disconnected_regions_have_stable_root_domains(self):
        mesh = EditorMesh2D.merge(_rectangle(), _rectangle(7., 0., 9., 3.))
        topology = discover_boundaries(mesh, _outer())
        self.assertEqual(topology.parents, (None, None))
        self.assertEqual(topology.signs, (-1, -1))
        domains = complement_domains(topology, _fronts(mesh, topology))
        self.assertEqual(len(domains), 1)
        self.assertEqual(domains[0].hole_boundary_indices, (0, 1))

    def test_discovery_owns_inputs_without_changing_mesh(self):
        mesh, outer = _ring(), _outer()
        nodes, elements, before_outer = mesh.nodes.copy(), mesh.elements.copy(), outer.copy()
        topology = discover_boundaries(mesh, outer)
        topology.outer_outline[:] = 0.
        topology.outlines[0][0][0] = -1
        np.testing.assert_array_equal(mesh.nodes, nodes)
        np.testing.assert_array_equal(mesh.elements, elements)
        np.testing.assert_array_equal(outer, before_outer)

    def test_touching_and_intersecting_source_regions_are_rejected(self):
        for other in (_rectangle(4., 3., 5., 4.), _rectangle(3., 2., 5., 4.)):
            mesh = EditorMesh2D.merge(_rectangle(), other)
            with self.subTest(other=other.nodes.tolist()), self.assertRaises(QuadLayerError) as caught:
                discover_boundaries(mesh, _outer())
            self.assertEqual(caught.exception.code, "topology_change")

    def test_nested_exteriors_without_an_enclosing_void_are_rejected(self):
        mesh = EditorMesh2D.merge(_rectangle(), _rectangle(1., 1., 2., 2.))
        with self.assertRaises(QuadLayerError) as caught:
            discover_boundaries(mesh, _outer())
        self.assertEqual(caught.exception.code, "topology_change")

    def test_odd_frozen_boundary_is_rejected(self):
        inner = np.array([[-1., -1.], [0., -1.], [1., -1.], [1., 1.], [-1., 1.]])
        mesh = Mesh2D(np.vstack((inner, 2 * inner)),
                      [[i, i+5, (i+1) % 5+5, (i+1) % 5] for i in range(5)])
        with self.assertRaises(QuadLayerError) as caught:
            discover_boundaries(mesh, _outer())
        self.assertEqual(caught.exception.code, "unsupported_boundary_parity")
        self.assertEqual(caught.exception.context["boundary_node_count"], 5)

    def test_source_winding_triangle_and_outside_are_rejected(self):
        clockwise = _rectangle()
        clockwise.elements = clockwise.elements[:, ::-1]
        triangle = Mesh2D([[0, 0], [1, 0], [0, 1]], [[0, 1, 2, 2]])
        for mesh in (clockwise, triangle, _rectangle(20., 20., 21., 21.)):
            with self.subTest(mesh=mesh.elements.tolist()), self.assertRaises(QuadLayerError):
                discover_boundaries(mesh, _outer())

    def test_changed_nesting_and_front_contact_are_rejected(self):
        mesh = EditorMesh2D.merge(_ring(), _rectangle(4., 4., 6., 6.))
        topology = discover_boundaries(mesh, _outer())
        fronts = _fronts(mesh, topology)
        for shift in ([3., 0.], [8., 0.]):
            changed = [front.copy() for front in fronts]
            changed[2] += shift
            with self.subTest(shift=shift), self.assertRaises(QuadLayerError) as caught:
                validate_fronts(topology, changed, layer_index=2)
            self.assertEqual(caught.exception.code, "topology_change")
            self.assertEqual(caught.exception.context["layer_index"], 2)

    def test_front_must_stay_away_from_outer_outline(self):
        mesh = _rectangle()
        topology = discover_boundaries(mesh, _outer())
        front = _fronts(mesh, topology)[0] + [11., 0.]
        with self.assertRaises(QuadLayerError):
            validate_fronts(topology, [front])

    def test_front_separation_uses_global_coordinate_tolerance(self):
        mesh = EditorMesh2D.merge(_rectangle(), _rectangle(4.0005, 0., 6., 3.))
        with self.assertRaises(QuadLayerError):
            discover_boundaries(mesh, _outer())
        topology = discover_boundaries(mesh, _outer(), coordinate_tolerance=1e-4)
        validate_fronts(topology, _fronts(mesh, topology))

    def test_section_front_retains_every_anchor_and_implicit_closure(self):
        mesh = _rectangle()
        outline = EditorMesh2D.outlines_by_edge(mesh)[0]
        plan = plan_quad_sections(mesh, outline, .5)
        points = section_outline_points(plan)
        self.assertFalse(np.array_equal(points[0], points[-1]))
        self.assertTrue(np.all(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1) > 0))
        for section in plan.sections:
            for point in section.outer_boundary:
                self.assertTrue(np.any(np.all(points == point, axis=1)))

    def test_near_axis_source_can_plan_an_exact_orthogonal_front(self):
        mesh = _rectangle()
        mesh.nodes[2:, 0] += 1e-4
        topology = discover_boundaries(mesh, _outer())
        plan = plan_quad_sections(mesh, topology.outlines[0], .5)
        validate_fronts(topology, [section_outline_points(plan)])

    def test_source_convexity_scales_threshold_by_adjacent_edge_lengths(self):
        mesh = Mesh2D([[0., 0.], [1e-7, 0.], [1e-7, 1e-7], [-1., 1.]], [[0, 1, 2, 3]])
        _validate_mesh(mesh)

    def test_large_global_span_does_not_inflate_boundary_clearance(self):
        mesh = EditorMesh2D.merge(_rectangle(), _rectangle(4.01, 0., 6., 3.))
        outer = [[-5., -5.], [1e12, -5.], [1e12, 1e12], [-5., 1e12]]
        topology = discover_boundaries(mesh, outer)
        validate_fronts(topology, _fronts(mesh, topology))

    def test_generated_front_roundoff_validation_does_not_move_nodes(self):
        mesh = _rectangle()
        topology = discover_boundaries(mesh, _outer())
        front = np.array([[-.5, -.5], [-.5, 1.], [-.5 + 1e-16, 3.5],
                          [4.5, 3.5], [4.5, -.5]])
        before = front.copy()
        validate_fronts(topology, [front])
        np.testing.assert_array_equal(front, before)
        front[1, 0] += 1e-6
        with self.assertRaises(QuadLayerError):
            validate_fronts(topology, [front])

    def test_full_near_axis_source_survives_two_layer_stages(self):
        mesh = _rectangle()
        mesh.nodes[2:, 0] += 1e-4
        original = mesh.nodes.copy()
        result = generate_full_adaptive_mesh(mesh, [.5, .8], outer_outline=_outer())
        self.assertIs(result.mesh, mesh)
        np.testing.assert_array_equal(mesh.nodes[:len(original)], original)
        self.assertEqual(len(EditorMesh2D.outline(mesh)), 1)


if __name__ == "__main__":
    unittest.main()
