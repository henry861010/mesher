import unittest

import numpy as np

from circle.circle import _clear_node, _delete, _delete_element, _search_circle
from mesh import Mesh


class SearchCircleTests(unittest.TestCase):
    def setUp(self):
        self.mesh = Mesh(
            nodes=np.array(
                [
                    [0.0, 0.0, 4.0],
                    [1.0, 0.0, 4.0],
                    [1.0, 1.0, 4.0],
                    [0.0, 1.0, 4.0],
                    [2.0, 0.0, 4.0],
                    [2.0, 1.0, 4.0],
                ]
            ),
            elements=np.array(
                [[0, 1, 2, 3], [1, 4, 5, 2]],
                dtype=np.int64,
            ),
        )

    def test_all_requires_every_node_to_be_inside(self):
        result = _search_circle(
            self.mesh,
            x=0.5,
            y=0.5,
            radius=np.sqrt(0.5),
            type="ALL",
            tolerance=0.0,
        )

        np.testing.assert_array_equal(result, [0])

    def test_part_requires_at_least_one_node_to_be_inside(self):
        result = _search_circle(
            self.mesh,
            x=0.5,
            y=0.5,
            radius=np.sqrt(0.5),
            type="PART",
            tolerance=0.0,
        )

        np.testing.assert_array_equal(result, [0, 1])

    def test_tolerance_includes_a_node_just_outside_the_radius(self):
        result = _search_circle(
            self.mesh,
            x=0.0,
            y=0.0,
            radius=0.99,
            type="PART",
            tolerance=0.01,
        )

        np.testing.assert_array_equal(result, [0, 1])

    def test_invalid_type_raises(self):
        with self.assertRaisesRegex(ValueError, "ALL.*PART"):
            _search_circle(
                self.mesh,
                x=0.0,
                y=0.0,
                radius=1.0,
                type="SOME",
                tolerance=0.0,
            )


class DeleteElementTests(unittest.TestCase):
    def setUp(self):
        self.nodes = np.arange(18, dtype=np.float64).reshape(6, 3)
        self.mesh = Mesh(
            nodes=self.nodes,
            elements=np.array(
                [
                    [0, 1, 2, 3],
                    [1, 2, 3, 4],
                    [2, 3, 4, 5],
                    [0, 2, 4, 5],
                ],
                dtype=np.int32,
            ),
        )

    def test_removes_requested_elements_and_keeps_nodes(self):
        result = _delete_element(self.mesh, np.array([1, 3], dtype=np.int64))

        self.assertIs(result, self.mesh)
        self.assertIs(self.mesh.nodes, self.nodes)
        np.testing.assert_array_equal(
            self.mesh.elements,
            np.array([[0, 1, 2, 3], [2, 3, 4, 5]], dtype=np.int32),
        )

    def test_empty_indices_leave_all_elements(self):
        original = self.mesh.elements

        _delete_element(self.mesh, [])

        self.assertIs(self.mesh.elements, original)

    def test_rejects_out_of_range_indices_without_mutating_mesh(self):
        original = self.mesh.elements.copy()

        with self.assertRaises(IndexError):
            _delete_element(self.mesh, [4])

        np.testing.assert_array_equal(self.mesh.elements, original)

    def test_legacy_delete_name_delegates_to_delete_element(self):
        result = _delete(self.mesh, [0])

        self.assertIs(result, self.mesh)
        self.assertEqual(self.mesh.elements.shape, (3, 4))


class ClearNodeTests(unittest.TestCase):
    def test_removes_unreferenced_nodes_and_remaps_elements(self):
        nodes = np.arange(21, dtype=np.float32).reshape(7, 3)
        mesh = Mesh(
            nodes=nodes,
            elements=np.array(
                [[0, 2, 5, 6], [2, 3, 5, 6]],
                dtype=np.int32,
            ),
        )

        index_map = _clear_node(mesh)

        np.testing.assert_array_equal(index_map, [0, -1, 1, 2, -1, 3, 4])
        self.assertEqual(index_map.ndim, 1)
        self.assertEqual(index_map.dtype, np.dtype(np.intp))
        np.testing.assert_array_equal(index_map[[6, 2, 0, 5]], [4, 1, 0, 3])
        np.testing.assert_array_equal(mesh.nodes, nodes[[0, 2, 3, 5, 6]])
        np.testing.assert_array_equal(
            mesh.elements,
            np.array([[0, 1, 3, 4], [1, 2, 3, 4]], dtype=np.int32),
        )
        self.assertEqual(mesh.nodes.dtype, np.float32)
        self.assertEqual(mesh.elements.dtype, np.int32)

    def test_empty_elements_remove_all_nodes(self):
        mesh = Mesh(
            nodes=np.arange(12, dtype=np.float64).reshape(4, 3),
            elements=np.empty((0, 4), dtype=np.int64),
        )

        index_map = _clear_node(mesh)

        np.testing.assert_array_equal(index_map, [-1, -1, -1, -1])
        self.assertEqual(mesh.nodes.shape, (0, 3))
        self.assertEqual(mesh.elements.shape, (0, 4))

    def test_returns_identity_map_when_all_nodes_are_referenced(self):
        nodes = np.arange(12, dtype=np.float64).reshape(4, 3)
        elements = np.array([[0, 1, 2, 3]], dtype=np.int32)
        mesh = Mesh(nodes=nodes, elements=elements)

        index_map = _clear_node(mesh)

        np.testing.assert_array_equal(index_map, [0, 1, 2, 3])
        self.assertIs(mesh.nodes, nodes)
        self.assertIs(mesh.elements, elements)


if __name__ == "__main__":
    unittest.main()
