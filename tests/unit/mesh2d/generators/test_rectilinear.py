import unittest

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d.generators import generate_rectilinear_mesh


class GenerateRectilinearMeshTests(unittest.TestCase):
    def test_generates_counter_clockwise_quad_grid_with_edge_size_bound(self):
        mesh = generate_rectilinear_mesh(
            target_edge_size=0.4,
            x_coordinates=[0.0, 1.0],
            y_coordinates=[0.0, 1.0],
        )

        self.assertIsInstance(mesh, Mesh2D)
        self.assertEqual(mesh.elements.shape, (9, 4))
        points = mesh.nodes[mesh.elements, :2]
        first_edges = points[:, 1] - points[:, 0]
        second_edges = points[:, 3] - points[:, 0]
        signed_corner_areas = (
            first_edges[:, 0] * second_edges[:, 1]
            - first_edges[:, 1] * second_edges[:, 0]
        )
        self.assertTrue(np.all(signed_corner_areas > 0.0))

    def test_rejects_nonpositive_target_edge_size(self):
        for target_edge_size in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(target_edge_size=target_edge_size):
                with self.assertRaisesRegex(ValueError, "target_edge_size"):
                    generate_rectilinear_mesh(
                        target_edge_size=target_edge_size,
                        x_coordinates=[0.0, 1.0],
                        y_coordinates=[0.0, 1.0],
                    )


if __name__ == "__main__":
    unittest.main()
