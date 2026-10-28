import unittest

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d import EditorMesh2D

from ._fixtures import empty_mesh


class EditorOutlineTests(unittest.TestCase):
    def test_mixed_elements_preserve_winding_and_do_not_alias_connectivity(self):
        mesh = Mesh2D([[0, 0], [1, 0], [1, 1], [0, 1], [2, 0]],
                      [[0, 1, 2, 3], [1, 4, 2, 2]])
        for reverse, expected in ((False, [0, 1, 4, 2, 3]), (True, [0, 3, 2, 4, 1])):
            if reverse:
                mesh.elements = [[3, 2, 1, 0], [2, 4, 1, 1]]
            outlines = EditorMesh2D.outline(mesh)
            np.testing.assert_array_equal(outlines[0], expected)
            self.assertEqual(outlines[0].dtype, np.int64)
            outlines[0][:] = -1
            self.assertTrue(np.all(mesh.elements >= 0))

    def test_holes_and_point_touching_components(self):
        lower = np.array([0, 1, 2, 4, 6, 8, 9, 10])
        elements = np.column_stack((lower, lower + 1, lower + 5, lower + 4))
        mesh = Mesh2D(np.zeros((16, 3)), elements)
        boundaries = EditorMesh2D.outline(mesh)
        np.testing.assert_array_equal(boundaries[0], [0, 1, 2, 3, 7, 11, 15, 14, 13, 12, 8, 4])
        np.testing.assert_array_equal(boundaries[1], [5, 9, 10, 6])
        mesh = Mesh2D(np.zeros((7, 3)), [[0, 1, 2, 3], [0, 4, 5, 6]])
        boundaries = EditorMesh2D.outline(mesh)
        self.assertEqual([loop.tolist() for loop in boundaries], [[0, 1, 2, 3], [0, 4, 5, 6]])
        self.assertEqual(EditorMesh2D.outline(empty_mesh()), [])

    def test_disconnected_loops_are_canonical_and_sorted(self):
        elements = np.arange(4000, dtype=np.int32).reshape(-1, 4)
        elements = np.roll(elements[::-1], 2, axis=1)
        mesh = Mesh2D(np.zeros((4000, 3)), elements)
        loops = EditorMesh2D.outline(mesh)
        self.assertEqual(len(loops), 1000)
        np.testing.assert_array_equal(np.stack(loops), np.arange(4000).reshape(-1, 4))

    def test_open_high_valence_fan_traces_the_complete_boundary(self):
        count = 3000
        angles = np.linspace(0, np.pi, count + 1)
        nodes = np.vstack(([0, 0], np.column_stack((np.cos(angles), np.sin(angles)))))
        starts = np.arange(1, count + 1, dtype=np.int32)
        elements = np.column_stack((np.zeros(count, dtype=np.int32), starts, starts + 1, starts + 1))
        loops = EditorMesh2D.outline(Mesh2D(nodes, elements))
        self.assertEqual(len(loops), 1)
        np.testing.assert_array_equal(loops[0], np.arange(count + 2))

    def test_invalid_topology_raises_without_mutation(self):
        for elements in (
            [[0, 1, 2, 2], [0, 1, 3, 3]],  # Same direction on shared edge.
            [[0, 1, 2, 2], [1, 0, 3, 3], [0, 1, 4, 4]],  # Three owners.
            [[0, 1, 0, 2]],
        ):
            mesh = Mesh2D(np.zeros((5, 3)), elements)
            nodes, original = mesh.nodes, mesh.elements
            with self.assertRaises(ValueError):
                EditorMesh2D.outline(mesh)
            self.assertIs(mesh.nodes, nodes)
            self.assertIs(mesh.elements, original)

