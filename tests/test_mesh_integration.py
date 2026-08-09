import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from checkerboard import checkerboard_box
from mesh import Mesh
from viewer import build_faces, view_mesh


class CheckerboardIntegrationTests(unittest.TestCase):
    def test_checkerboard_returns_mesh_with_counter_clockwise_elements(self):
        mesh = checkerboard_box(
            element_size=0.5,
            x_list=[0.0, 1.0, 2.0],
            y_list=[0.0, 1.0, 2.0],
        )

        self.assertIsInstance(mesh, Mesh)
        self.assertEqual(mesh.nodes.dtype, np.dtype(np.float64))
        points = mesh.nodes[mesh.elements, :2]
        first_edges = points[:, 1] - points[:, 0]
        second_edges = points[:, 3] - points[:, 0]
        signed_corner_areas = (
            first_edges[:, 0] * second_edges[:, 1]
            - first_edges[:, 1] * second_edges[:, 0]
        )
        self.assertGreater(mesh.elements.shape[0], 0)
        self.assertTrue(np.all(signed_corner_areas > 0.0))

    def test_build_faces_handles_mesh_with_triangles_and_quads(self):
        mesh = Mesh(
            nodes=np.zeros((7, 3), dtype=np.float64),
            elements=np.array(
                [[0, 1, 2, 2], [3, 4, 5, 6]],
                dtype=np.int32,
            ),
        )

        faces = build_faces(mesh)

        np.testing.assert_array_equal(
            faces,
            np.array([3, 0, 1, 2, 4, 3, 4, 5, 6]),
        )

    def test_build_faces_supports_empty_mesh(self):
        mesh = Mesh(
            nodes=np.empty((0, 3), dtype=np.float64),
            elements=np.empty((0, 4), dtype=np.int32),
        )

        faces = build_faces(mesh)

        self.assertEqual(faces.ndim, 1)
        self.assertEqual(faces.size, 0)

    def test_legacy_checkerboard_tuple_unpacking_is_removed(self):
        mesh = checkerboard_box(0.5, [0.0, 1.0], [0.0, 1.0])

        with self.assertRaises(TypeError):
            _nodes, _elements = mesh

    def test_legacy_build_faces_array_api_is_removed(self):
        elements = np.array([[0, 1, 2, 2]], dtype=np.int64)

        with self.assertRaises((TypeError, AttributeError)):
            build_faces(elements)

    def test_legacy_two_argument_view_mesh_api_is_removed(self):
        nodes = np.zeros((3, 3), dtype=np.float64)
        elements = np.array([[0, 1, 2, 2]], dtype=np.int64)

        with self.assertRaises(TypeError):
            view_mesh(nodes, elements)

    def test_view_mesh_highlights_requested_elements_in_red(self):
        mesh = Mesh(
            nodes=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [2.0, 1.0, 0.0],
                ]
            ),
            elements=np.array(
                [[0, 1, 2, 3], [1, 4, 5, 2]],
                dtype=np.int64,
            ),
        )
        plotter = MagicMock()

        with patch("viewer.viewer.pv.Plotter", return_value=plotter):
            view_mesh(mesh, element_indices=[1])

        displayed_mesh = plotter.add_mesh.call_args.args[0]
        np.testing.assert_array_equal(
            displayed_mesh.cell_data["face_colors"],
            np.array([[173, 216, 230], [255, 0, 0]], dtype=np.uint8),
        )
        self.assertEqual(plotter.add_mesh.call_args.kwargs["scalars"], "face_colors")
        self.assertTrue(plotter.add_mesh.call_args.kwargs["rgb"])
        plotter.add_points.assert_not_called()

    def test_view_mesh_highlights_requested_nodes_in_yellow(self):
        mesh = Mesh(
            nodes=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
            elements=np.array([[0, 1, 2, 3]], dtype=np.int64),
        )
        plotter = MagicMock()

        with patch("viewer.viewer.pv.Plotter", return_value=plotter):
            view_mesh(mesh, node_indices=[0, 2])

        highlighted_nodes = plotter.add_points.call_args.args[0]
        np.testing.assert_array_equal(highlighted_nodes, mesh.nodes[[0, 2]])
        self.assertEqual(plotter.add_points.call_args.kwargs["color"], "yellow")
        self.assertEqual(plotter.add_points.call_args.kwargs["point_size"], 12)
        self.assertTrue(
            plotter.add_points.call_args.kwargs["render_points_as_spheres"]
        )

    def test_view_mesh_can_highlight_nodes_without_elements(self):
        mesh = Mesh(
            nodes=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            ),
            elements=np.empty((0, 4), dtype=np.int64),
        )
        plotter = MagicMock()

        with patch("viewer.viewer.pv.Plotter", return_value=plotter):
            view_mesh(mesh, node_indices=[1])

        plotter.add_mesh.assert_not_called()
        np.testing.assert_array_equal(
            plotter.add_points.call_args.args[0],
            mesh.nodes[[1]],
        )

    def test_view_mesh_supports_an_empty_mesh(self):
        mesh = Mesh(
            nodes=np.empty((0, 3), dtype=np.float64),
            elements=np.empty((0, 4), dtype=np.int64),
        )
        plotter = MagicMock()

        with patch("viewer.viewer.pv.Plotter", return_value=plotter):
            view_mesh(mesh)

        plotter.add_mesh.assert_not_called()
        plotter.add_points.assert_not_called()
        plotter.show.assert_called_once_with()

    def test_view_mesh_validates_node_indices(self):
        mesh = Mesh(
            nodes=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
            elements=np.array([[0, 1, 2, 2]], dtype=np.int64),
        )
        invalid_cases = (
            ([[0]], ValueError),
            ([0.5], TypeError),
            ([-1], IndexError),
            ([3], IndexError),
        )

        for node_indices, expected_error in invalid_cases:
            with self.subTest(node_indices=node_indices):
                with self.assertRaises(expected_error):
                    view_mesh(mesh, node_indices=node_indices)


if __name__ == "__main__":
    unittest.main()
