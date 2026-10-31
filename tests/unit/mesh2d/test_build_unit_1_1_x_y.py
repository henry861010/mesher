"""The adjacent-edge template fills its boundary with valid Quad4 cells."""

import unittest
from collections import Counter

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d.adapter.unit import build_unit_1_1_x_y


def _patch(x_segments, y_segments, *, wavy=False, nonuniform=False):
    if nonuniform:
        if (x_segments, y_segments) != (5, 3):
            raise ValueError("the nonuniform fixture has five and three segments")
        x_fraction = np.array([0.0, 0.08, 0.23, 0.40, 0.64, 1.0])
        y_fraction = np.array([0.0, 0.18, 0.57, 1.0])
    else:
        x_fraction = np.linspace(0.0, 1.0, x_segments + 1)
        y_fraction = np.linspace(0.0, 1.0, y_segments + 1)
    x_y = 0.035 * np.sin(2.0 * np.pi * x_fraction) if wavy else np.zeros_like(x_fraction)
    y_x = 1.0 + 0.03 * np.sin(2.0 * np.pi * y_fraction) if wavy else np.ones_like(y_fraction)
    bottom = np.column_stack((x_fraction, x_y, np.zeros_like(x_fraction)))
    right = np.column_stack((y_x[1:], y_fraction[1:], np.zeros_like(y_fraction[1:])))
    nodes = np.vstack((bottom, right, [[0.0, 1.0, 0.0]]))
    mesh = Mesh2D(nodes, np.empty((0, 4), dtype=np.int32))
    edges = (
        np.arange(x_segments + 1, dtype=np.int64),
        np.r_[x_segments, np.arange(x_segments + 1, x_segments + y_segments + 1)].astype(np.int64),
        np.array([x_segments + y_segments, x_segments + y_segments + 1], dtype=np.int64),
        np.array([x_segments + y_segments + 1, 0], dtype=np.int64),
    )
    return mesh, edges


def _area(points):
    xy = points[:, :2] - points[0, :2]
    following = np.roll(xy, -1, axis=0)
    return 0.5 * float(np.sum(xy[:, 0] * following[:, 1] - xy[:, 1] * following[:, 0]))


class BuildUnit11XYTests(unittest.TestCase):
    def assert_valid_patch(self, mesh, edges, initial_nodes, initial_elements=0):
        quads = mesh.elements[initial_elements:]
        boundary = np.concatenate([edge[:-1] for edge in edges])
        expected_boundary = {
            frozenset((int(a), int(b)))
            for edge in edges
            for a, b in zip(edge[:-1], edge[1:])
        }
        incidence = Counter(
            frozenset((int(quad[index]), int(quad[(index + 1) % 4])))
            for quad in quads
            for index in range(4)
        )
        self.assertEqual(
            {edge for edge, count in incidence.items() if count == 1},
            expected_boundary,
        )
        self.assertTrue(
            all(count == 2 for edge, count in incidence.items() if edge not in expected_boundary)
        )
        self.assertTrue(np.all(boundary < initial_nodes))

        xy = mesh.nodes[quads, :2]
        vectors = np.roll(xy, -1, axis=1) - xy
        following = np.roll(vectors, -1, axis=1)
        turns = vectors[..., 0] * following[..., 1] - vectors[..., 1] * following[..., 0]
        self.assertTrue(np.all(turns > 0.0))
        self.assertAlmostEqual(
            sum(_area(mesh.nodes[quad]) for quad in quads),
            abs(_area(mesh.nodes[boundary])),
        )

    def test_equal_and_unequal_edge_counts(self):
        for x_segments, y_segments in ((1, 1), (2, 2), (3, 3), (5, 3), (3, 5), (5, 1), (1, 5)):
            with self.subTest(x_segments=x_segments, y_segments=y_segments):
                mesh, edges = _patch(x_segments, y_segments)
                original_nodes = mesh.nodes.copy()
                self.assertIs(build_unit_1_1_x_y(mesh, *edges), mesh)
                self.assertEqual(
                    mesh.node_count - len(original_nodes), max(x_segments, y_segments) - 1
                )
                self.assertEqual(
                    mesh.element_count,
                    (x_segments + y_segments) // 2 + max(x_segments, y_segments) - 1,
                )
                np.testing.assert_array_equal(mesh.nodes[:len(original_nodes)], original_nodes)
                self.assert_valid_patch(mesh, edges, len(original_nodes))

    def test_long_edges_can_be_rotated_and_winding_reversed(self):
        for rotation in range(4):
            for clockwise in (False, True):
                with self.subTest(rotation=rotation, clockwise=clockwise):
                    mesh, edges = _patch(5, 3, wavy=True)
                    original_count = mesh.node_count
                    edges = edges[rotation:] + edges[:rotation]
                    if clockwise:
                        edges = tuple(edge[::-1] for edge in edges[::-1])
                    build_unit_1_1_x_y(mesh, *edges)
                    self.assert_valid_patch(mesh, edges, original_count)

    def test_nonuniform_curved_edges_have_zero_z(self):
        mesh, edges = _patch(5, 3, wavy=True, nonuniform=True)
        original_count = mesh.node_count
        boundary = np.concatenate([edge[:-1] for edge in edges])
        boundary_vectors = np.roll(mesh.nodes[boundary, :2], -1, axis=0) - mesh.nodes[boundary, :2]
        boundary_turns = (
            boundary_vectors[:, 0] * np.roll(boundary_vectors[:, 1], -1)
            - boundary_vectors[:, 1] * np.roll(boundary_vectors[:, 0], -1)
        )
        self.assertTrue(np.any(boundary_turns < 0.0))
        build_unit_1_1_x_y(mesh, *edges)
        self.assert_valid_patch(mesh, edges, original_count)
        np.testing.assert_allclose(
            mesh.nodes[original_count:, 2],
            0.0,
            atol=1e-12,
        )

    def test_first_pair_uses_arc_length_on_nonuniform_straight_edges(self):
        mesh, edges = _patch(5, 3, nonuniform=True)
        first_new_index = mesh.node_count
        build_unit_1_1_x_y(mesh, *edges)
        np.testing.assert_allclose(mesh.nodes[first_new_index], [0.08, 0.57, 0.0], atol=1e-12)

    def test_large_coordinate_offset_preserves_patch_geometry(self):
        mesh, edges = _patch(5, 3, wavy=True, nonuniform=True)
        mesh.nodes += [1.0e10, -2.0e10, 0.0]
        initial_count = mesh.node_count
        build_unit_1_1_x_y(mesh, *edges)
        self.assert_valid_patch(mesh, edges, initial_count)

    def test_existing_external_element_is_preserved(self):
        mesh, edges = _patch(3, 3)
        remote = np.array([[3.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 1.0, 0.0], [3.0, 1.0, 0.0]])
        original_count = mesh.node_count
        mesh.replace_data(
            nodes=np.vstack((mesh.nodes, remote)),
            elements=np.array([[original_count + index for index in range(4)]]),
        )
        original_nodes = mesh.nodes.copy()
        original_elements = mesh.elements.copy()
        build_unit_1_1_x_y(mesh, *edges)
        np.testing.assert_array_equal(mesh.nodes[:len(original_nodes)], original_nodes)
        np.testing.assert_array_equal(mesh.elements[:1], original_elements)
        self.assert_valid_patch(mesh, edges, len(original_nodes), 1)

    def test_invalid_inputs_leave_mesh_unchanged(self):
        mesh, edges = _patch(2, 2)
        bad_cases = (
            ("nonarray", (edges[0].tolist(), *edges[1:])),
            ("dimension", (edges[0].reshape(1, -1), *edges[1:])),
            ("dtype", (edges[0].astype(float), *edges[1:])),
            ("missing_join", (edges[0], np.array([0, 4]), *edges[2:])),
            ("out_of_range", (np.array([0, 1, 99]), *edges[1:])),
            ("duplicate", (np.array([0, 1, 0]), *edges[1:])),
        )
        for name, bad_edges in bad_cases:
            with self.subTest(name=name):
                nodes, elements = mesh.nodes.copy(), mesh.elements.copy()
                with self.assertRaises(ValueError):
                    build_unit_1_1_x_y(mesh, *bad_edges)
                np.testing.assert_array_equal(mesh.nodes, nodes)
                np.testing.assert_array_equal(mesh.elements, elements)

    def test_invalid_topology_and_geometry_are_transactional(self):
        odd, odd_edges = _patch(2, 3)
        opposite = Mesh2D(
            [[0, 0], [0.5, 0], [1, 0], [1, 1], [0.5, 1], [0, 1]],
            np.empty((0, 4), dtype=np.int32),
        )
        opposite_edges = tuple(np.array(edge) for edge in ([0, 1, 2], [2, 3], [3, 4, 5], [5, 0]))
        impossible, impossible_edges = _patch(2, 2)
        impossible.nodes[1, :2] = [-0.2, -0.2]
        crossed = Mesh2D([[0, 0], [1, 1], [0, 1], [1, 0]], np.empty((0, 4), dtype=np.int32))
        crossed_edges = tuple(np.array(edge) for edge in ([0, 1], [1, 2], [2, 3], [3, 0]))
        for name, mesh, edges in (
            ("odd", odd, odd_edges),
            ("opposite", opposite, opposite_edges),
            ("no_feasible_pair", impossible, impossible_edges),
            ("self_intersection", crossed, crossed_edges),
        ):
            with self.subTest(name=name):
                nodes, elements = mesh.nodes.copy(), mesh.elements.copy()
                with self.assertRaises(ValueError):
                    build_unit_1_1_x_y(mesh, *edges)
                np.testing.assert_array_equal(mesh.nodes, nodes)
                np.testing.assert_array_equal(mesh.elements, elements)


if __name__ == "__main__":
    unittest.main()
