import unittest

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d import EditorMesh2D
from mesher.mesh2d.adapter import generate_layer_adaptive_mesh
from mesher.mesh2d.generators import generate_rectilinear_mesh

from ._fixtures import empty_mesh, two_quads


class EditorOutlinesByEdgeTests(unittest.TestCase):
    def assert_closed_on_right(self, mesh, edges):
        owners = {}
        for cell in mesh.elements:
            for a, b in zip(cell, np.roll(cell, -1)):
                if a != b:
                    owners.setdefault(tuple(sorted((a, b))), []).append(cell)
        for edge, following in zip(edges, edges[1:] + edges[:1]):
            self.assertEqual(edge.dtype, np.int64)
            self.assertEqual(edge[-1], following[0])
            for a, b in zip(edge[:-1], edge[1:]):
                cells = owners[tuple(sorted((a, b)))]
                self.assertEqual(len(cells), 1)
                direction = mesh.nodes[b, :2] - mesh.nodes[a, :2]
                inward = mesh.nodes[cells[0], :2].mean(axis=0) - mesh.nodes[a, :2]
                self.assertLess(direction[0] * inward[1] - direction[1] * inward[0], 0)

    def test_rectangle_keeps_all_nodes_and_normalizes_either_winding(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                mesh = two_quads()
                if reverse:
                    mesh.elements = mesh.elements[:, ::-1]
                nodes, elements = mesh.nodes.copy(), mesh.elements.copy()
                outlines = EditorMesh2D.outlines_by_edge(mesh)
                self.assertEqual(len(outlines), 1)
                self.assertEqual([edge.tolist() for edge in outlines[0]],
                                 [[0, 3], [3, 4, 5], [5, 2], [2, 1, 0]])
                self.assert_closed_on_right(mesh, outlines[0])
                outlines[0][0][:] = -1
                np.testing.assert_array_equal(mesh.nodes, nodes)
                np.testing.assert_array_equal(mesh.elements, elements)
                self.assertEqual(outlines[0][-1][-1], 0)

    def test_hole_island_and_disconnected_clockwise_component(self):
        lower = np.array([0, 1, 2, 4, 6, 8, 9, 10])
        ring = Mesh2D([[x, y] for y in range(4) for x in range(4)],
                      np.column_stack((lower, lower + 1, lower + 5, lower + 4)))
        island = Mesh2D([[1.25, 1.25], [1.75, 1.25], [1.75, 1.75], [1.25, 1.75]],
                        [[0, 1, 2, 3]])
        clockwise = Mesh2D([[5, 0], [6, 0], [6, 1], [5, 1]], [[3, 2, 1, 0]])
        mesh = EditorMesh2D.merge(EditorMesh2D.merge(ring, island), clockwise)
        outlines = EditorMesh2D.outlines_by_edge(mesh)
        self.assertEqual(len(outlines), 4)
        signs = []
        for edges in outlines:
            self.assertEqual(len(edges), 4)
            self.assert_closed_on_right(mesh, edges)
            points = mesh.nodes[np.concatenate([edge[:-1] for edge in edges]), :2]
            following = np.roll(points, -1, axis=0)
            signs.append(np.sign(np.sum(points[:, 0] * following[:, 1]
                                        - points[:, 1] * following[:, 0])))
        self.assertEqual(signs, [-1, 1, -1, -1])

    def test_point_touching_components_and_concave_outline(self):
        touching = Mesh2D([[0, 0], [1, 0], [1, 1], [0, 1],
                           [-1, 0], [-1, -1], [0, -1]],
                          [[0, 1, 2, 3], [0, 4, 5, 6]])
        outlines = EditorMesh2D.outlines_by_edge(touching)
        self.assertEqual(len(outlines), 2)
        for edges in outlines:
            self.assert_closed_on_right(touching, edges)
        concave = Mesh2D([[x, y] for y in range(3) for x in range(3)],
                         [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6]])
        edges = EditorMesh2D.outlines_by_edge(concave)[0]
        self.assertEqual(len(edges), 6)
        self.assert_closed_on_right(concave, edges)

    def test_rotated_rectangle_and_padded_triangle(self):
        mesh = two_quads()
        angle = 0.37
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        mesh.nodes[:, :2] = mesh.nodes[:, :2] @ rotation.T + [1e6, -1e6]
        self.assertEqual(len(EditorMesh2D.outlines_by_edge(mesh)[0]), 4)
        for cell in ([[0, 1, 2, 2]], [[2, 1, 0, 0]]):
            triangle = Mesh2D([[0, 0], [1, 0], [0, 1]], cell)
            edges = EditorMesh2D.outlines_by_edge(triangle)[0]
            self.assertEqual([edge.tolist() for edge in edges], [[0, 2], [2, 1], [1, 0]])
            self.assert_closed_on_right(triangle, edges)

    def test_angle_tolerance_controls_nearly_collinear_segments(self):
        mesh = two_quads()
        mesh.nodes[1, 1] = -1e-6
        self.assertEqual(len(EditorMesh2D.outlines_by_edge(mesh, angle_tolerance=1e-5)[0]), 4)
        self.assertEqual(len(EditorMesh2D.outlines_by_edge(mesh, angle_tolerance=1e-8)[0]), 5)
        self.assertEqual(len(EditorMesh2D.outlines_by_edge(two_quads(), angle_tolerance=0)[0]), 4)

    def test_winding_and_corner_detection_are_independent_of_world_scale(self):
        for scale in (1e-200, 1e200):
            with self.subTest(scale=scale):
                mesh = two_quads()
                mesh.nodes[:, :2] *= scale
                self.assertEqual([edge.tolist() for edge in EditorMesh2D.outlines_by_edge(mesh)[0]],
                                 [[0, 3], [3, 4, 5], [5, 2], [2, 1, 0]])

    def test_empty_invalid_tolerance_topology_and_geometry(self):
        self.assertEqual(EditorMesh2D.outlines_by_edge(empty_mesh()), [])
        for tolerance in (True, None, -1, np.nan, np.inf, np.pi / 2, 1j, [0]):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                EditorMesh2D.outlines_by_edge(empty_mesh(), angle_tolerance=tolerance)
        invalid = Mesh2D([[0, 0], [1, 0], [0, 1], [0, -1]],
                         [[0, 1, 2, 2], [0, 1, 3, 3]])
        with self.assertRaises(ValueError):
            EditorMesh2D.outlines_by_edge(invalid)
        for kind in ('zero_length', 'zero_area', 'nonfinite'):
            mesh = two_quads()
            if kind == 'zero_length':
                mesh.nodes[1] = mesh.nodes[0]
            elif kind == 'zero_area':
                mesh.nodes[:, 1] = 0
            else:
                mesh.nodes[2, 0] = np.nan
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                EditorMesh2D.outlines_by_edge(mesh)

    def test_outline_drives_quad_layer_without_changing_original_cells(self):
        mesh = generate_rectilinear_mesh(target_edge_size=5,
                                         x_coordinates=[50, 200], y_coordinates=[50, 200])
        nodes, elements = mesh.nodes.copy(), mesh.elements.copy()
        outline = EditorMesh2D.outlines_by_edge(mesh)[0]
        self.assertEqual([len(edge) for edge in outline], [31] * 4)
        result = generate_layer_adaptive_mesh(mesh, outline, 50, pattern_lines=[], expand_ratio=1.5)
        self.assertIs(result.mesh, mesh)
        np.testing.assert_array_equal(mesh.nodes[:len(nodes)], nodes)
        np.testing.assert_array_equal(mesh.elements[:len(elements)], elements)
        # The adapter's pairing policy may change the transition cell count.
        # Its contract is a complete, convex Quad4 layer with a reusable front.
        self.assertGreater(result.diagnostics["added_elements"], 0)
        self.assertEqual(mesh.element_count, len(elements) + result.diagnostics["added_elements"])
        added = mesh.nodes[mesh.elements[len(elements):], :2]
        vectors = np.roll(added, -1, axis=1) - added
        following = np.roll(vectors, -1, axis=1)
        self.assertTrue(np.all(vectors[..., 0] * following[..., 1]
                               - vectors[..., 1] * following[..., 0] > 0))
        rolled = np.roll(added, -1, axis=1)
        new_area = .5 * np.sum(added[..., 0] * rolled[..., 1]
                                - added[..., 1] * rolled[..., 0])
        self.assertAlmostEqual(new_area, 250 ** 2 - 150 ** 2)
        self.assert_closed_on_right(mesh, [np.asarray(edge, dtype=np.int64)
                                           for edge in result.outline])
        np.testing.assert_allclose(mesh.nodes[:, :2].min(axis=0), [0, 0])
        np.testing.assert_allclose(mesh.nodes[:, :2].max(axis=0), [250, 250])


if __name__ == '__main__':
    unittest.main()
