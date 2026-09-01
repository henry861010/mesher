import re
import unittest

import numpy as np

from mesher import ElementType2D, Mesh2D, Mesh3D
from mesher.mesh3d.extrusion import Dragger


class MeshModelTests(unittest.TestCase):
    def test_mesh2d_normalizes_xy_and_infers_mixed_types(self):
        mesh = Mesh2D(
            nodes=[[0, 0], [1, 0], [0, 1], [1, 1]],
            elements=[[0, 1, 2, 2], [0, 1, 3, 2]],
        )

        self.assertEqual(mesh.nodes.shape, (4, 3))
        self.assertEqual(mesh.nodes.dtype, np.float64)
        self.assertEqual(mesh.elements.dtype, np.int32)
        np.testing.assert_array_equal(mesh.nodes[:, 2], 0.0)
        np.testing.assert_array_equal(
            mesh.element_types,
            [ElementType2D.TRI3, ElementType2D.QUAD4],
        )

    def test_mesh3d_normalizes_and_owns_mesh_data(self):
        nodes = np.array([[1, 2, 3, 99]], dtype=np.float32)
        elements = np.zeros((1, 8), dtype=np.int64)
        component_ids = np.array([1], dtype=np.int64)
        component_table = {"EMPTY": 0, "body": 1}

        mesh = Mesh3D(nodes, elements, component_ids, component_table)
        nodes[0, 0] = 100
        elements[0, 0] = 7
        component_ids[0] = 9
        component_table["body"] = 8

        self.assertEqual(mesh.nodes.dtype, np.float64)
        self.assertEqual(mesh.elements.dtype, np.int32)
        self.assertEqual(mesh.element_comps.dtype, np.int32)
        self.assertEqual(mesh.nodes.shape, (1, 3))
        self.assertEqual(mesh.nodes[0, 0], 1.0)
        self.assertEqual(mesh.elements[0, 0], 0)
        self.assertEqual(mesh.element_comps[0], 1)
        self.assertEqual(mesh.comps["body"], 1)
        self.assertEqual(mesh.node_count, 1)
        self.assertEqual(mesh.element_count, 1)
        self.assertEqual(mesh.component_count, 2)
        self.assertFalse(hasattr(mesh, "element_types"))
        self.assertFalse(hasattr(mesh, "element_component_ids"))
        self.assertFalse(hasattr(mesh, "component_ids_by_name"))

    def test_mesh3d_rejects_invalid_shapes(self):
        cases = (
            {
                "nodes": np.empty((1, 2)),
                "elements": np.empty((0, 8), dtype=np.int32),
                "components": np.empty(0, dtype=np.int32),
                "message": "nodes must have shape (n, 3+)",
            },
            {
                "nodes": np.empty((0, 3)),
                "elements": np.empty((1, 4), dtype=np.int32),
                "components": np.empty(1, dtype=np.int32),
                "message": "elements must have shape (m, 8)",
            },
            {
                "nodes": np.empty((0, 3)),
                "elements": np.empty((0, 8), dtype=np.int32),
                "components": np.empty((0, 1), dtype=np.int32),
                "message": "element_comps must have shape (m,)",
            },
            {
                "nodes": np.empty((0, 3)),
                "elements": np.empty((1, 8), dtype=np.int32),
                "components": np.empty(0, dtype=np.int32),
                "message": "element_comps length must match element count",
            },
        )
        for case in cases:
            with self.subTest(case["message"]):
                with self.assertRaisesRegex(ValueError, re.escape(case["message"])):
                    Mesh3D(
                        nodes=case["nodes"],
                        elements=case["elements"],
                        element_comps=case["components"],
                        comps={},
                    )

    def test_dragger_build_returns_only_owned_valid_rows(self):
        dragger = Dragger()
        dragger.node_num = 1
        dragger.nodes = np.array(
            [[0.0, 0.0, 0.0], [99.0, 99.0, 99.0]],
            dtype=np.float64,
        )

        mesh = dragger.build([], 1.0)
        dragger.nodes[0, 0] = 42.0

        self.assertEqual(mesh.node_count, 1)
        self.assertEqual(mesh.nodes[0, 0], 0.0)
        self.assertFalse(np.any(mesh.nodes == 99.0))

    def test_dragger_extrudes_a_padded_triangle_as_fixed_width_wedge(self):
        dragger = Dragger()
        dragger.set_2D(
            np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
            np.array([[0, 1, 2, 2]], dtype=np.int32),
        )

        mesh = dragger.build(_layers(), 1.0)

        np.testing.assert_array_equal(
            mesh.elements[0],
            [0, 1, 2, 2, 3, 4, 5, 5],
        )
        np.testing.assert_array_equal(mesh.element_comps, [1])
        self.assertEqual(mesh.comps, {"EMPTY": 0, "Cu": 1})
        self.assertAlmostEqual(dragger.element_2D_volume[0], 0.5)

    def test_dragger_extrudes_a_quad_as_hex(self):
        dragger = Dragger()
        dragger.set_2D(
            np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
            np.array([[0, 1, 2, 3]], dtype=np.int32),
        )

        mesh = dragger.build(_layers(), 1.0)

        np.testing.assert_array_equal(mesh.element_comps, [1])

    def test_dragger_reuses_adjacent_layer_nodes(self):
        dragger = Dragger()
        dragger.set_2D(
            np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
            np.array([[0, 1, 2, 3]], dtype=np.int32),
        )
        layers = _layers()
        layers.insert(1, {"z": 1.0, "assignments": []})
        layers[-1]["z"] = 2.0

        mesh = dragger.build(layers, 1.0)

        self.assertEqual(mesh.node_count, 12)
        self.assertEqual(mesh.element_count, 2)
        np.testing.assert_array_equal(mesh.elements[0, 4:], mesh.elements[1, :4])

    def test_dragger_skips_a_layer_without_material(self):
        dragger = Dragger()
        dragger.set_2D(
            np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
            np.array([[0, 1, 2, 3]], dtype=np.int32),
        )

        mesh = dragger.build(
            [{"z": 0.0, "assignments": []}, {"z": 1.0, "assignments": []}],
            1.0,
        )

        self.assertEqual(mesh.node_count, 0)
        self.assertEqual(mesh.element_count, 0)

    def test_dragger_rejects_malformed_mixed_connectivity(self):
        dragger = Dragger()
        nodes = np.zeros((4, 2), dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "Quad4 rows or padded Tri3"):
            dragger.set_2D(nodes, [[0, 1, 1, 2]])

        with self.assertRaisesRegex(ValueError, "out-of-range node index"):
            dragger.set_2D(nodes, [[0, 1, 2, 4]])


def _layers():
    return [
        {
            "z": 0.0,
            "assignments": [
                {
                    "type": 3,
                    "face": None,
                    "areas": [{"priority": 1.0, "material": "Cu"}],
                }
            ],
        },
        {"z": 1.0, "assignments": []},
    ]


if __name__ == "__main__":
    unittest.main()
