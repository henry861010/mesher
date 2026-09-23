import unittest
from unittest import mock

import numpy as np

from mesher import Mesh3D
from mesher.mesh3d.utils import add_midside_node


EDGE_CORNERS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def _mesh(
    nodes,
    corner_rows,
    *,
    node_counts=None,
    midside_rows=None,
):
    corners = np.asarray(corner_rows, dtype=np.int32)
    elements = np.empty((len(corners), 20), dtype=np.int32)
    elements[:, :8] = corners
    elements[:, 8:] = corners[:, 7, None]
    if midside_rows:
        for row, midsides in midside_rows.items():
            elements[row, 8:20] = midsides
    if node_counts is None:
        node_counts = np.full(len(corners), 8, dtype=np.int32)

    return Mesh3D(
        nodes=nodes,
        elements=elements,
        element_comps=np.full(len(corners), 3, dtype=np.int32),
        comps={"solid": 3},
        element_types=np.full(len(corners), 4, dtype=np.int32),
        types={4: 185},
        element_reals=np.full(len(corners), 2, dtype=np.int32),
        reals={2: [1.25]},
        element_sections=np.full(len(corners), 7, dtype=np.int32),
        sections={7: [8, 9]},
        element_node_num=node_counts,
    )


class AddMidsideNodeTopologyTests(unittest.TestCase):
    def test_converts_all_supported_ansys_topologies(self):
        cases = [
            (
                "hex",
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [1, 1, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 1],
                    [1, 1, 1],
                    [0, 1, 1],
                ],
                [0, 1, 2, 3, 4, 5, 6, 7],
                [8, 11, 13, 9, 16, 18, 19, 17, 10, 12, 14, 15],
                12,
            ),
            (
                "prism",
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 1],
                    [0, 1, 1],
                ],
                [0, 1, 2, 2, 3, 4, 5, 5],
                [6, 9, 2, 7, 12, 14, 5, 13, 8, 10, 11, 11],
                9,
            ),
            (
                "pyramid",
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [1, 1, 0],
                    [0, 1, 0],
                    [0.5, 0.5, 1],
                ],
                [0, 1, 2, 3, 4, 4, 4, 4],
                [5, 8, 10, 6, 4, 4, 4, 4, 7, 9, 11, 12],
                8,
            ),
            (
                "tetra",
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ],
                [0, 1, 2, 2, 3, 3, 3, 3],
                [4, 7, 2, 5, 3, 3, 3, 3, 6, 8, 9, 9],
                6,
            ),
        ]

        for name, nodes, corners, expected_midsides, added_nodes in cases:
            with self.subTest(topology=name):
                mesh = _mesh(nodes, [corners])
                original_nodes = mesh.nodes.copy()
                original_elements = mesh.elements.copy()

                result = add_midside_node(mesh, [0])

                self.assertIsNot(result, mesh)
                self.assertEqual(result.node_count, len(nodes) + added_nodes)
                self.assertEqual(result.element_node_num[0], 20)
                np.testing.assert_array_equal(
                    result.elements[0, :8],
                    corners,
                )
                np.testing.assert_array_equal(
                    result.elements[0, 8:20],
                    expected_midsides,
                )
                for slot, (first, second) in enumerate(EDGE_CORNERS):
                    midside_id = result.elements[0, 8 + slot]
                    first_id = corners[first]
                    second_id = corners[second]
                    if first_id == second_id:
                        self.assertEqual(midside_id, first_id)
                    else:
                        np.testing.assert_allclose(
                            result.nodes[midside_id],
                            (
                                original_nodes[first_id]
                                + original_nodes[second_id]
                            )
                            * 0.5,
                        )

                np.testing.assert_array_equal(mesh.nodes, original_nodes)
                np.testing.assert_array_equal(mesh.elements, original_elements)
                np.testing.assert_array_equal(
                    result.element_comps,
                    mesh.element_comps,
                )
                np.testing.assert_array_equal(
                    result.element_types,
                    mesh.element_types,
                )
                np.testing.assert_array_equal(
                    result.element_reals,
                    mesh.element_reals,
                )
                np.testing.assert_array_equal(
                    result.element_sections,
                    mesh.element_sections,
                )
                self.assertEqual(result.comps, mesh.comps)
                self.assertEqual(result.types, mesh.types)
                self.assertEqual(result.reals, mesh.reals)
                self.assertEqual(result.sections, mesh.sections)

    def test_selected_elements_share_midside_nodes(self):
        nodes = [
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [2, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [2, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
            [2, 1, 1],
        ]
        corners = [
            [0, 1, 4, 3, 6, 7, 10, 9],
            [1, 2, 5, 4, 7, 8, 11, 10],
        ]
        mesh = _mesh(nodes, corners)

        result = add_midside_node(mesh, [0, 1])

        self.assertEqual(result.node_count, len(nodes) + 20)
        np.testing.assert_array_equal(result.element_node_num, [20, 20])
        np.testing.assert_array_equal(
            result.elements[0, [9, 13, 17, 18]],
            result.elements[1, [11, 15, 16, 19]],
        )

    def test_reuses_midsides_from_an_existing_twenty_node_element(self):
        base_nodes = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [2, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [2, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [2, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
                [2, 1, 1],
            ],
            dtype=np.float64,
        )
        corners = np.array(
            [
                [0, 1, 4, 3, 6, 7, 10, 9],
                [1, 2, 5, 4, 7, 8, 11, 10],
            ],
            dtype=np.int32,
        )
        existing_midpoints = np.array(
            [
                (base_nodes[corners[0, first]] + base_nodes[corners[0, second]])
                * 0.5
                for first, second in EDGE_CORNERS
            ]
        )
        nodes = np.vstack((base_nodes, existing_midpoints))
        mesh = _mesh(
            nodes,
            corners,
            node_counts=[20, 8],
            midside_rows={0: np.arange(12, 24, dtype=np.int32)},
        )

        result = add_midside_node(mesh, None)

        self.assertEqual(result.node_count, mesh.node_count + 8)
        np.testing.assert_array_equal(result.element_node_num, [20, 20])
        np.testing.assert_array_equal(
            result.elements[0, [9, 13, 17, 18]],
            result.elements[1, [11, 15, 16, 19]],
        )


class AddMidsideNodeValidationTests(unittest.TestCase):
    def setUp(self):
        self.nodes = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=np.float64,
        )
        self.corners = [0, 1, 2, 3, 4, 5, 6, 7]
        self.mesh = _mesh(self.nodes, [self.corners])

    def test_empty_and_twenty_node_selections_are_noops(self):
        self.assertIs(add_midside_node(self.mesh, []), self.mesh)

        converted = add_midside_node(self.mesh, [0])
        self.assertIs(add_midside_node(converted, [0]), converted)
        self.assertIs(add_midside_node(converted, None), converted)

    def test_rejects_non_mesh_input(self):
        with self.assertRaisesRegex(TypeError, "Mesh3D"):
            add_midside_node(object(), [0])

    def test_rejects_invalid_element_indices(self):
        cases = [
            (0, ValueError, "one-dimensional"),
            ([[0]], ValueError, "one-dimensional"),
            ([0.0], TypeError, "integers"),
            ([True], TypeError, "integers"),
            ([0, 0], ValueError, "duplicates"),
            ([-1], IndexError, "out of range"),
            ([1], IndexError, "out of range"),
        ]
        for indices, error_type, message in cases:
            with self.subTest(indices=indices):
                with self.assertRaisesRegex(error_type, message):
                    add_midside_node(self.mesh, indices)

    def test_rejects_unsupported_node_count_and_topology_atomically(self):
        four_node = _mesh(
            self.nodes,
            [self.corners],
            node_counts=[4],
        )
        with self.assertRaisesRegex(ValueError, "8 or 20"):
            add_midside_node(four_node, [0])

        invalid = _mesh(
            self.nodes,
            [[0, 1, 2, 3, 4, 5, 6, 6]],
        )
        original_nodes = invalid.nodes.copy()
        original_elements = invalid.elements.copy()
        with self.assertRaisesRegex(ValueError, "ANSYS hex"):
            add_midside_node(invalid, [0])
        np.testing.assert_array_equal(invalid.nodes, original_nodes)
        np.testing.assert_array_equal(invalid.elements, original_elements)

    def test_rejects_conflicting_existing_midsides_atomically(self):
        midside_coordinates = np.array(
            [
                (self.nodes[first] + self.nodes[second]) * 0.5
                for first, second in EDGE_CORNERS
            ]
        )
        nodes = np.vstack(
            (self.nodes, midside_coordinates, midside_coordinates[0])
        )
        first_midsides = np.arange(8, 20, dtype=np.int32)
        second_midsides = first_midsides.copy()
        second_midsides[0] = 20
        mesh = _mesh(
            nodes,
            [self.corners, self.corners, self.corners],
            node_counts=[20, 20, 8],
            midside_rows={
                0: first_midsides,
                1: second_midsides,
            },
        )
        original_nodes = mesh.nodes.copy()
        original_elements = mesh.elements.copy()

        with mock.patch(
            "mesher.mesh3d.utils._EDGE_SCAN_CHUNK_SIZE",
            1,
        ):
            with self.assertRaisesRegex(ValueError, "conflicting midside"):
                add_midside_node(mesh, [2])

        np.testing.assert_array_equal(mesh.nodes, original_nodes)
        np.testing.assert_array_equal(mesh.elements, original_elements)


if __name__ == "__main__":
    unittest.main()
