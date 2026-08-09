import unittest

import numpy as np

from circle.circle import (
    _clear_node,
    _delete,
    _delete_element,
    _node_to_circle,
    _search_circle,
)
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


class NodeToCircleTests(unittest.TestCase):
    def test_projects_selected_2d_nodes_radially_from_an_off_center_circle(self):
        nodes = np.array(
            [
                [4.0, 2.0],   # already on the circle: offset (3, 4)
                [7.0, -2.0],  # outside the circle
                [1.0, -4.0],  # inside the circle
                [-8.0, 9.0],  # unselected
            ],
            dtype=np.float64,
        )
        mesh = Mesh(
            nodes=nodes,
            elements=np.array([[0, 1, 2, 3]], dtype=np.int64),
        )

        _node_to_circle(
            mesh,
            center_x=1.0,
            center_y=-2.0,
            radius=5.0,
            indices=[2, 0, 1],
        )

        np.testing.assert_allclose(
            mesh.nodes,
            np.array(
                [
                    [4.0, 2.0],
                    [6.0, -2.0],
                    [1.0, -7.0],
                    [-8.0, 9.0],
                ],
                dtype=np.float64,
            ),
        )

    def test_updates_in_place_and_preserves_z_unselected_rows_and_elements(self):
        nodes = np.array(
            [
                [4.0, 2.0, 7.5],
                [7.0, -2.0, -3.25],
                [1.0, -4.0, 11.0],
                [-8.0, 9.0, 13.0],
            ],
            dtype=np.float64,
        )
        elements = np.array([[0, 1, 2, 3]], dtype=np.int32)
        mesh = Mesh(nodes=nodes, elements=elements)
        original_z = nodes[:, 2].copy()
        original_unselected = nodes[3].copy()

        result = _node_to_circle(mesh, 1.0, -2.0, 5.0, [0, 1, 2])

        self.assertIs(result, mesh)
        self.assertIs(mesh.nodes, nodes)
        self.assertIs(mesh.elements, elements)
        np.testing.assert_array_equal(mesh.nodes[:, 2], original_z)
        np.testing.assert_array_equal(mesh.nodes[3], original_unselected)
        np.testing.assert_array_equal(mesh.elements, [[0, 1, 2, 3]])

    def test_integer_nodes_are_promoted_to_float64_before_projection(self):
        nodes = np.array(
            [[3, 4], [0, 2], [9, 9]],
            dtype=np.int32,
        )
        mesh = Mesh(
            nodes=nodes,
            elements=np.array([[0, 1, 2, 2]], dtype=np.int64),
        )

        result = _node_to_circle(mesh, 0.0, 0.0, 5.0, [0, 1])

        self.assertIs(result, mesh)
        self.assertEqual(mesh.nodes.dtype, np.dtype(np.float64))
        self.assertIsNot(mesh.nodes, nodes)
        np.testing.assert_array_equal(
            mesh.nodes,
            np.array([[3.0, 4.0], [0.0, 5.0], [9.0, 9.0]], dtype=np.float64),
        )

    def test_empty_indices_are_a_no_op(self):
        nodes = np.array(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=np.float64,
        )
        mesh = Mesh(
            nodes=nodes,
            elements=np.empty((0, 4), dtype=np.int64),
        )
        original = nodes.copy()

        result = _node_to_circle(mesh, 0.0, 0.0, 5.0, [])

        self.assertIs(result, mesh)
        self.assertIs(mesh.nodes, nodes)
        np.testing.assert_array_equal(mesh.nodes, original)

    def test_node_at_center_raises_without_partial_mutation(self):
        for radius in (5.0, 0.0):
            with self.subTest(radius=radius):
                nodes = np.array(
                    [[7.0, -2.0], [1.0, -2.0], [4.0, 2.0]],
                    dtype=np.float64,
                )
                mesh = Mesh(
                    nodes=nodes,
                    elements=np.array([[0, 1, 2, 2]], dtype=np.int64),
                )
                original = nodes.copy()

                with self.assertRaisesRegex(ValueError, "coincide.*center"):
                    _node_to_circle(mesh, 1.0, -2.0, radius, [0, 1])

                self.assertIs(mesh.nodes, nodes)
                np.testing.assert_array_equal(mesh.nodes, original)

    def test_projection_is_stable_at_float64_extremes(self):
        largest = np.finfo(np.float64).max
        smallest = np.nextafter(0.0, 1.0)
        diagonal = 1.0 / np.sqrt(2.0)
        cases = (
            (
                (0.0, 0.0),
                (largest, largest),
                largest,
                (largest * diagonal, largest * diagonal),
            ),
            (
                (0.0, 0.0),
                (smallest, smallest),
                1.0,
                (diagonal, diagonal),
            ),
            (
                (0.0, 0.0),
                (smallest, 0.0),
                1.0e308,
                (1.0e308, 0.0),
            ),
            (
                (-largest, -largest),
                (largest, largest),
                largest,
                (-largest + largest * diagonal,) * 2,
            ),
        )

        for center, node, radius, expected in cases:
            with self.subTest(center=center, node=node, radius=radius):
                mesh = Mesh(
                    nodes=np.array([node], dtype=np.float64),
                    elements=np.empty((0, 4), dtype=np.int64),
                )

                _node_to_circle(mesh, *center, radius, [0])

                self.assertTrue(np.all(np.isfinite(mesh.nodes)))
                np.testing.assert_allclose(
                    mesh.nodes[0], expected, rtol=2.0e-15, atol=0.0
                )

    def test_unrepresentable_projection_raises_without_mutation(self):
        largest = np.finfo(np.float64).max
        nodes = np.array([[largest, 0.0]], dtype=np.float64)
        mesh = Mesh(nodes, np.empty((0, 4), dtype=np.int64))
        original = nodes.copy()

        with self.assertRaisesRegex(ValueError, "exceed.*float64"):
            _node_to_circle(mesh, largest * 0.5, 0.0, largest, [0])

        self.assertIs(mesh.nodes, nodes)
        np.testing.assert_array_equal(mesh.nodes, original)

    def test_invalid_radius_raises_without_mutating_nodes(self):
        for radius in (-1.0, np.nan, np.inf):
            with self.subTest(radius=radius):
                nodes = np.array([[3.0, 4.0]], dtype=np.float64)
                mesh = Mesh(
                    nodes=nodes,
                    elements=np.empty((0, 4), dtype=np.int64),
                )
                original = nodes.copy()

                with self.assertRaises(ValueError):
                    _node_to_circle(mesh, 0.0, 0.0, radius, [0])

                self.assertIs(mesh.nodes, nodes)
                np.testing.assert_array_equal(mesh.nodes, original)

    def test_rejects_invalid_node_indices_without_mutating_nodes(self):
        invalid_cases = (
            ([[0]], ValueError),
            ([0.5], TypeError),
            ([True], TypeError),
            ([-1], IndexError),
            ([2], IndexError),
        )

        for indices, expected_error in invalid_cases:
            with self.subTest(indices=indices):
                nodes = np.array(
                    [[3.0, 4.0], [0.0, 2.0]],
                    dtype=np.float64,
                )
                mesh = Mesh(
                    nodes=nodes,
                    elements=np.empty((0, 4), dtype=np.int64),
                )
                original = nodes.copy()

                with self.assertRaises(expected_error):
                    _node_to_circle(mesh, 0.0, 0.0, 5.0, indices)

                self.assertIs(mesh.nodes, nodes)
                np.testing.assert_array_equal(mesh.nodes, original)


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
        nodes = np.arange(21, dtype=np.float64).reshape(7, 3)
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
        self.assertEqual(mesh.nodes.dtype, np.float64)
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
