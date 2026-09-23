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
        elements = np.arange(20, dtype=np.int64).reshape(1, 20)
        component_ids = np.array([1], dtype=np.int64)
        component_table = {"EMPTY": 0, "body": 1}
        element_types = np.array([1], dtype=np.int64)
        types = {1: 185}
        element_reals = np.array([2], dtype=np.int64)
        real_values = [1.5, 2.5]
        reals = {2: real_values}
        element_sections = np.array([3], dtype=np.int64)
        section_values = [4, 5]
        sections = {3: section_values}
        element_node_num = np.array([8], dtype=np.int64)

        mesh = Mesh3D(
            nodes=nodes,
            elements=elements,
            element_comps=component_ids,
            comps=component_table,
            element_types=element_types,
            types=types,
            element_reals=element_reals,
            reals=reals,
            element_sections=element_sections,
            sections=sections,
            element_node_num=element_node_num,
        )
        nodes[0, 0] = 100
        elements[0, 0] = 7
        component_ids[0] = 9
        component_table["body"] = 8
        element_types[0] = 9
        types[1] = 186
        element_reals[0] = 9
        real_values[0] = 99
        element_sections[0] = 9
        section_values[0] = 99
        element_node_num[0] = 20

        self.assertEqual(mesh.nodes.dtype, np.float64)
        self.assertEqual(mesh.elements.dtype, np.int32)
        self.assertEqual(mesh.element_comps.dtype, np.int32)
        self.assertEqual(mesh.element_types.dtype, np.int32)
        self.assertEqual(mesh.element_reals.dtype, np.int32)
        self.assertEqual(mesh.element_sections.dtype, np.int32)
        self.assertEqual(mesh.element_node_num.dtype, np.int32)
        self.assertEqual(mesh.nodes.shape, (1, 3))
        self.assertEqual(mesh.elements.shape, (1, 20))
        self.assertEqual(mesh.nodes[0, 0], 1.0)
        self.assertEqual(mesh.elements[0, 0], 0)
        np.testing.assert_array_equal(mesh.elements[0, 8:], 7)
        self.assertEqual(mesh.element_comps[0], 1)
        self.assertEqual(mesh.comps["body"], 1)
        self.assertEqual(mesh.element_types[0], 1)
        self.assertEqual(mesh.types, {1: 185})
        self.assertEqual(mesh.element_reals[0], 2)
        self.assertEqual(mesh.reals, {2: [1.5, 2.5]})
        self.assertEqual(mesh.element_sections[0], 3)
        self.assertEqual(mesh.sections, {3: [4, 5]})
        self.assertEqual(mesh.element_node_num[0], 8)
        self.assertEqual(mesh.node_count, 1)
        self.assertEqual(mesh.element_count, 1)
        self.assertEqual(mesh.component_count, 2)
        self.assertFalse(hasattr(mesh, "element_component_ids"))
        self.assertFalse(hasattr(mesh, "component_ids_by_name"))

    def test_mesh3d_normalizes_connectivity_padding(self):
        elements = np.tile(np.arange(20, dtype=np.int32), (4, 1))
        mesh = Mesh3D(
            nodes=np.empty((0, 3)),
            elements=elements,
            element_comps=np.zeros(4),
            comps={},
            element_types=np.ones(4),
            types={1: 185},
            element_reals=np.zeros(4),
            reals={},
            element_sections=np.zeros(4),
            sections={},
            element_node_num=[2, 4, 8, 20],
        )

        for row, node_num in enumerate((2, 4, 8, 20)):
            np.testing.assert_array_equal(
                mesh.elements[row, :node_num],
                np.arange(node_num),
            )
            if node_num < 20:
                np.testing.assert_array_equal(
                    mesh.elements[row, node_num:],
                    node_num - 1,
                )

    def test_mesh3d_rejects_invalid_shapes(self):
        cases = [
            ("nodes", np.empty((1, 2)), "nodes must have shape (n, 3+)"),
            (
                "elements",
                np.empty((1, 8), dtype=np.int32),
                "elements must have shape (m, 20)",
            ),
            (
                "element_comps",
                np.empty((0, 1), dtype=np.int32),
                "element_comps must have shape (m,)",
            ),
            (
                "element_types",
                np.empty(0, dtype=np.int32),
                "element_types length must match element count",
            ),
            (
                "element_reals",
                np.empty((1, 1), dtype=np.int32),
                "element_reals must have shape (m,)",
            ),
            (
                "element_sections",
                np.empty(0, dtype=np.int32),
                "element_sections length must match element count",
            ),
            (
                "element_node_num",
                [0],
                "element_node_num values must be between 1 and 20",
            ),
            (
                "element_node_num",
                [21],
                "element_node_num values must be between 1 and 20",
            ),
        ]
        for field, value, message in cases:
            kwargs = {
                "nodes": np.empty((0, 3)),
                "elements": np.empty((1, 20), dtype=np.int32),
                "element_comps": np.zeros(1, dtype=np.int32),
                "comps": {},
                "element_types": np.ones(1, dtype=np.int32),
                "types": {1: 185},
                "element_reals": np.zeros(1, dtype=np.int32),
                "reals": {},
                "element_sections": np.zeros(1, dtype=np.int32),
                "sections": {},
                "element_node_num": np.full(1, 8, dtype=np.int32),
            }
            kwargs[field] = value
            with self.subTest(message):
                with self.assertRaisesRegex(ValueError, re.escape(message)):
                    Mesh3D(**kwargs)

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
            mesh.elements[0, :8],
            [0, 1, 2, 2, 3, 4, 5, 5],
        )
        np.testing.assert_array_equal(mesh.elements[0, 8:], 5)
        np.testing.assert_array_equal(mesh.element_comps, [1])
        self.assertEqual(mesh.comps, {"EMPTY": 0, "Cu": 1})
        np.testing.assert_array_equal(mesh.element_types, [1])
        self.assertEqual(mesh.types, {1: 185})
        np.testing.assert_array_equal(mesh.element_reals, [0])
        self.assertEqual(mesh.reals, {})
        np.testing.assert_array_equal(mesh.element_sections, [0])
        self.assertEqual(mesh.sections, {})
        np.testing.assert_array_equal(mesh.element_node_num, [8])
        self.assertAlmostEqual(dragger.element_2D_volume[0], 0.5)

    def test_dragger_extrudes_a_quad_as_hex(self):
        dragger = Dragger()
        dragger.set_2D(
            np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
            np.array([[0, 1, 2, 3]], dtype=np.int32),
        )

        mesh = dragger.build(_layers(), 1.0)

        np.testing.assert_array_equal(mesh.element_comps, [1])
        self.assertEqual(mesh.elements.shape, (1, 20))
        np.testing.assert_array_equal(mesh.elements[0, 8:], mesh.elements[0, 7])

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
        np.testing.assert_array_equal(mesh.elements[0, 4:8], mesh.elements[1, :4])

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
