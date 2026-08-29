import unittest

import numpy as np

from mesher.mesh2d.circular.topology import (
    _get_boundary,
    _get_boundary_edge_groups,
)
from mesher import Mesh2D


class GetBoundaryTests(unittest.TestCase):
    def test_splits_selected_interface_edges_from_domain_edges(self):
        mesh = Mesh2D(
            nodes=np.zeros((6, 2), dtype=np.float64),
            elements=np.array(
                [[0, 1, 4, 3], [1, 2, 5, 4]],
                dtype=np.int32,
            ),
        )

        interface_edges, domain_edges = _get_boundary_edge_groups(mesh, [0])

        self.assertEqual(
            {tuple(map(int, edge)) for edge in interface_edges},
            {(1, 4)},
        )
        self.assertEqual(
            {tuple(map(int, edge)) for edge in domain_edges},
            {(0, 1), (4, 3), (3, 0)},
        )

    def test_returns_ordered_outer_boundary_without_repeating_first_node(self):
        mesh = Mesh2D(
            nodes=np.zeros((6, 3), dtype=np.float64),
            elements=np.array(
                [[0, 1, 4, 3], [1, 2, 5, 4]],
                dtype=np.int32,
            ),
        )

        boundaries = _get_boundary(mesh)

        self.assertEqual(len(boundaries), 1)
        np.testing.assert_array_equal(boundaries[0], [0, 1, 2, 5, 4, 3])
        self.assertEqual(boundaries[0].dtype, np.int64)

    def test_indices_limit_boundary_search_to_selected_elements(self):
        mesh = Mesh2D(
            nodes=np.zeros((6, 3), dtype=np.float64),
            elements=np.array(
                [[0, 1, 4, 3], [1, 2, 5, 4]],
                dtype=np.int32,
            ),
        )

        boundaries = _get_boundary(mesh, indices=np.array([0], dtype=np.int32))

        self.assertEqual(len(boundaries), 1)
        np.testing.assert_array_equal(boundaries[0], [0, 1, 4, 3])

    def test_multiple_indices_use_the_selected_submesh_regardless_of_order(self):
        mesh = Mesh2D(
            nodes=np.zeros((6, 3), dtype=np.float64),
            elements=np.array(
                [[0, 1, 4, 3], [1, 2, 5, 4]],
                dtype=np.int32,
            ),
        )

        boundaries = _get_boundary(mesh, indices=[1, 0])

        self.assertEqual(len(boundaries), 1)
        np.testing.assert_array_equal(boundaries[0], [0, 1, 2, 5, 4, 3])

    def test_empty_indices_have_no_boundaries(self):
        mesh = Mesh2D(
            nodes=np.zeros((4, 2), dtype=np.float64),
            elements=np.array([[0, 1, 2, 3]], dtype=np.int64),
        )

        self.assertEqual(_get_boundary(mesh, indices=[]), [])

    def test_rejects_invalid_element_indices(self):
        mesh = Mesh2D(
            nodes=np.zeros((4, 2), dtype=np.float64),
            elements=np.array([[0, 1, 2, 3]], dtype=np.int64),
        )

        invalid_cases = (
            ([1], IndexError),
            ([-1], IndexError),
            ([0.0], TypeError),
            ([True], TypeError),
            ([[0]], ValueError),
            ([0, 0], ValueError),
        )
        for indices, expected_error in invalid_cases:
            with self.subTest(indices=indices):
                with self.assertRaises(expected_error):
                    _get_boundary(mesh, indices=indices)

    def test_returns_outer_and_hole_boundaries_with_preserved_winding(self):
        nodes = np.zeros((16, 3), dtype=np.float64)
        elements = []
        for row in range(3):
            for column in range(3):
                if (row, column) == (1, 1):
                    continue
                lower_left = row * 4 + column
                elements.append(
                    [
                        lower_left,
                        lower_left + 1,
                        lower_left + 5,
                        lower_left + 4,
                    ]
                )
        mesh = Mesh2D(nodes=nodes, elements=np.asarray(elements, dtype=np.int64))

        boundaries = _get_boundary(mesh)

        self.assertEqual(len(boundaries), 2)
        np.testing.assert_array_equal(
            boundaries[0],
            [0, 1, 2, 3, 7, 11, 15, 14, 13, 12, 8, 4],
        )
        np.testing.assert_array_equal(boundaries[1], [5, 9, 10, 6])

    def test_supports_padded_triangles(self):
        mesh = Mesh2D(
            nodes=np.zeros((3, 2), dtype=np.float64),
            elements=np.array([[0, 1, 2, 2]], dtype=np.int64),
        )

        boundaries = _get_boundary(mesh)

        self.assertEqual(len(boundaries), 1)
        np.testing.assert_array_equal(boundaries[0], [0, 1, 2])

    def test_keeps_point_touching_components_as_separate_boundaries(self):
        mesh = Mesh2D(
            nodes=np.zeros((7, 2), dtype=np.float64),
            elements=np.array(
                [[0, 1, 2, 3], [0, 4, 5, 6]],
                dtype=np.int64,
            ),
        )

        boundaries = _get_boundary(mesh)

        self.assertEqual(len(boundaries), 2)
        np.testing.assert_array_equal(boundaries[0], [0, 1, 2, 3])
        np.testing.assert_array_equal(boundaries[1], [0, 4, 5, 6])

    def test_empty_mesh_has_no_boundaries(self):
        mesh = Mesh2D(
            nodes=np.empty((0, 3), dtype=np.float64),
            elements=np.empty((0, 4), dtype=np.int64),
        )

        self.assertEqual(_get_boundary(mesh), [])


if __name__ == "__main__":
    unittest.main()
