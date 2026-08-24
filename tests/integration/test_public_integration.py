import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from mesher import Mesh2D
from mesher.circular import extend_circular_mesh, imprint_circle
from mesher.circular.extend import (
    extend_circular_mesh as extend_circular_mesh_implementation,
)
from mesher.circular.imprint import (
    imprint_circle as imprint_circle_implementation,
)
from mesher.generators import generate_rectilinear_mesh
from mesher.visualization import build_faces, view_mesh


class PublicIntegrationTests(unittest.TestCase):
    def test_public_api_imports_resolve_from_domain_namespaces(self):
        self.assertEqual(Mesh2D.__module__, "mesher.mesh")
        self.assertEqual(
            generate_rectilinear_mesh.__module__,
            "mesher.generators.rectilinear",
        )
        self.assertIs(imprint_circle, imprint_circle_implementation)
        self.assertIs(
            extend_circular_mesh,
            extend_circular_mesh_implementation,
        )

    def test_checkerboard_returns_mesh_with_counter_clockwise_elements(self):
        mesh = generate_rectilinear_mesh(
            target_edge_size=0.5,
            x_coordinates=[0.0, 1.0, 2.0],
            y_coordinates=[0.0, 1.0, 2.0],
        )

        self.assertIsInstance(mesh, Mesh2D)
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
        mesh = Mesh2D(
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
        mesh = Mesh2D(
            nodes=np.empty((0, 3), dtype=np.float64),
            elements=np.empty((0, 4), dtype=np.int32),
        )

        faces = build_faces(mesh)

        self.assertEqual(faces.ndim, 1)
        self.assertEqual(faces.size, 0)

    def test_legacy_checkerboard_tuple_unpacking_is_removed(self):
        mesh = generate_rectilinear_mesh(0.5, [0.0, 1.0], [0.0, 1.0])

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
        mesh = Mesh2D(
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

        with patch(
            "mesher.visualization.pyvista.pv.Plotter",
            return_value=plotter,
        ):
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
        mesh = Mesh2D(
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

        with patch(
            "mesher.visualization.pyvista.pv.Plotter",
            return_value=plotter,
        ):
            view_mesh(mesh, node_indices=[0, 2])

        highlighted_nodes = plotter.add_points.call_args.args[0]
        np.testing.assert_array_equal(highlighted_nodes, mesh.nodes[[0, 2]])
        self.assertEqual(plotter.add_points.call_args.kwargs["color"], "yellow")
        self.assertEqual(plotter.add_points.call_args.kwargs["point_size"], 12)
        self.assertTrue(
            plotter.add_points.call_args.kwargs["render_points_as_spheres"]
        )

    def test_view_mesh_can_highlight_nodes_without_elements(self):
        mesh = Mesh2D(
            nodes=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            ),
            elements=np.empty((0, 4), dtype=np.int64),
        )
        plotter = MagicMock()

        with patch(
            "mesher.visualization.pyvista.pv.Plotter",
            return_value=plotter,
        ):
            view_mesh(mesh, node_indices=[1])

        plotter.add_mesh.assert_not_called()
        np.testing.assert_array_equal(
            plotter.add_points.call_args.args[0],
            mesh.nodes[[1]],
        )

    def test_view_mesh_supports_an_empty_mesh(self):
        mesh = Mesh2D(
            nodes=np.empty((0, 3), dtype=np.float64),
            elements=np.empty((0, 4), dtype=np.int64),
        )
        plotter = MagicMock()

        with patch(
            "mesher.visualization.pyvista.pv.Plotter",
            return_value=plotter,
        ):
            view_mesh(mesh)

        plotter.add_mesh.assert_not_called()
        plotter.add_points.assert_not_called()
        plotter.show.assert_called_once_with()

    def test_view_mesh_draws_reference_circles_boxes_and_lines(self):
        mesh = Mesh2D(
            nodes=np.empty((0, 3), dtype=np.float64),
            elements=np.empty((0, 4), dtype=np.int64),
        )
        plotter = MagicMock()

        with patch(
            "mesher.visualization.pyvista.pv.Plotter",
            return_value=plotter,
        ):
            view_mesh(
                mesh,
                reference_circles=[1.0, 2.0, 0.5],
                reference_boxes=[[-1.0, -2.0], [3.0, 4.0]],
                reference_lines=[[-3.0, 1.0], [5.0, 2.0]],
            )

        reference_segments = plotter.add_lines.call_args.args[0]
        self.assertEqual(reference_segments.shape, (266, 3))
        np.testing.assert_array_equal(
            reference_segments[-2:],
            np.array([[-3.0, 1.0, 0.0], [5.0, 2.0, 0.0]]),
        )
        self.assertEqual(plotter.add_lines.call_args.kwargs["color"], "black")
        self.assertEqual(plotter.add_lines.call_args.kwargs["width"], 2)
        self.assertFalse(plotter.add_lines.call_args.kwargs["connected"])

    def test_view_mesh_accepts_multiple_reference_shapes(self):
        mesh = Mesh2D(
            nodes=np.empty((0, 3), dtype=np.float64),
            elements=np.empty((0, 4), dtype=np.int64),
        )
        plotter = MagicMock()

        with patch(
            "mesher.visualization.pyvista.pv.Plotter",
            return_value=plotter,
        ):
            view_mesh(
                mesh,
                reference_circles=[[0.0, 0.0, 1.0], [2.0, 2.0, 0.5]],
                reference_boxes=[
                    [[0.0, 0.0], [1.0, 1.0]],
                    [[2.0, 2.0], [3.0, 3.0]],
                ],
                reference_lines=[
                    [[0.0, 0.0], [1.0, 1.0]],
                    [[1.0, 0.0], [0.0, 1.0]],
                ],
            )

        self.assertEqual(plotter.add_lines.call_args.args[0].shape, (532, 3))

    def test_view_mesh_validates_reference_geometry(self):
        mesh = Mesh2D(
            nodes=np.empty((0, 3), dtype=np.float64),
            elements=np.empty((0, 4), dtype=np.int64),
        )
        invalid_cases = (
            ({"reference_circles": [0.0, 0.0, 0.0]}, ValueError),
            ({"reference_circles": [0.0, 1.0]}, ValueError),
            ({"reference_boxes": [[1.0, 0.0], [0.0, 1.0]]}, ValueError),
            ({"reference_lines": [[0.0, 0.0], [np.inf, 1.0]]}, ValueError),
            ({"reference_lines": [["x", 0.0], [1.0, 1.0]]}, TypeError),
        )

        for kwargs, expected_error in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(expected_error):
                    view_mesh(mesh, **kwargs)

    def test_view_mesh_validates_node_indices(self):
        mesh = Mesh2D(
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
