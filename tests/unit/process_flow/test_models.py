import re
import unittest

import numpy as np

from mesher import ElementType2D, ElementType3D, Mesh2D, Mesh3D
from mesher.mesh3d import ExtrusionLayer, extrude_mesh


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
        nodes = np.array([[1, 2, 3]], dtype=np.float32)
        elements = np.zeros((1, 8), dtype=np.int64)
        component_ids = np.array([1], dtype=np.int64)
        component_table = {"EMPTY": 0, "body": 1}

        mesh = Mesh3D(
            nodes=nodes,
            elements=elements,
            element_component_ids=component_ids,
            component_ids_by_name=component_table,
        )
        nodes[0, 0] = 100
        elements[0, 0] = 7
        component_ids[0] = 9
        component_table["body"] = 8

        self.assertEqual(mesh.nodes.dtype, np.float64)
        self.assertEqual(mesh.elements.dtype, np.int32)
        self.assertEqual(mesh.element_component_ids.dtype, np.int32)
        self.assertEqual(mesh.nodes[0, 0], 1.0)
        self.assertEqual(mesh.elements[0, 0], 0)
        self.assertEqual(mesh.element_component_ids[0], 1)
        self.assertEqual(mesh.component_ids_by_name["body"], 1)
        self.assertEqual(mesh.node_count, 1)
        self.assertEqual(mesh.element_count, 1)
        self.assertEqual(mesh.component_count, 2)

    def test_mesh3d_rejects_invalid_shapes(self):
        cases = (
            {
                "nodes": np.empty((1, 2)),
                "elements": np.empty((0, 8), dtype=np.int32),
                "components": np.empty(0, dtype=np.int32),
                "message": "nodes must have shape (n, 3)",
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
                "message": "element_component_ids must have shape (m,)",
            },
        )
        for case in cases:
            with self.subTest(case["message"]):
                with self.assertRaisesRegex(ValueError, re.escape(case["message"])):
                    Mesh3D(
                        nodes=case["nodes"],
                        elements=case["elements"],
                        element_component_ids=case["components"],
                        component_ids_by_name={},
                    )

    def test_typed_extrusion_preserves_padded_wedge_contract(self):
        planar = Mesh2D(
            nodes=[[0, 0], [0, 1], [1, 0]],
            elements=[[0, 1, 2, 2]],
        )
        mesh = extrude_mesh(
            planar,
            [ExtrusionLayer(0.0, 1.0, [1])],
            element_size=1.0,
            component_ids_by_name={"EMPTY": 0, "Cu": 1},
        )

        np.testing.assert_array_equal(
            mesh.elements[0],
            [0, 1, 2, 2, 3, 4, 5, 5],
        )
        np.testing.assert_array_equal(mesh.element_types, [ElementType3D.WEDGE6])
        np.testing.assert_array_equal(mesh.element_component_ids, [1])

    def test_typed_extrusion_reuses_adjacent_layer_nodes(self):
        planar = Mesh2D(
            nodes=[[0, 0], [1, 0], [1, 1], [0, 1]],
            elements=[[0, 1, 2, 3]],
        )
        mesh = extrude_mesh(
            planar,
            [
                ExtrusionLayer(0.0, 1.0, [1]),
                ExtrusionLayer(1.0, 2.0, [1]),
            ],
            element_size=1.0,
            component_ids_by_name={"EMPTY": 0, "body": 1},
        )

        self.assertEqual(mesh.node_count, 12)
        self.assertEqual(mesh.element_count, 2)
        np.testing.assert_array_equal(mesh.elements[0, 4:], mesh.elements[1, :4])
        np.testing.assert_array_equal(mesh.element_types, [ElementType3D.HEX8] * 2)

    def test_typed_extrusion_accepts_an_empty_planar_mesh(self):
        planar = Mesh2D(
            nodes=np.empty((0, 2)),
            elements=np.empty((0, 4), dtype=np.int32),
        )

        mesh = extrude_mesh(
            planar,
            [ExtrusionLayer(0.0, 1.0, [])],
            element_size=1.0,
            component_ids_by_name={"EMPTY": 0},
        )

        self.assertEqual(mesh.nodes.shape, (0, 3))
        self.assertEqual(mesh.elements.shape, (0, 8))
        self.assertEqual(mesh.element_types.shape, (0,))
        self.assertEqual(mesh.element_component_ids.shape, (0,))

    def test_typed_extrusion_skips_a_layer_without_material(self):
        planar = Mesh2D(
            nodes=[[0, 0], [1, 0], [1, 1], [0, 1]],
            elements=[[0, 1, 2, 3]],
        )

        mesh = extrude_mesh(
            planar,
            [ExtrusionLayer(0.0, 1.0, [0])],
            element_size=1.0,
            component_ids_by_name={"EMPTY": 0},
        )

        self.assertEqual(mesh.node_count, 0)
        self.assertEqual(mesh.element_count, 0)


if __name__ == "__main__":
    unittest.main()
