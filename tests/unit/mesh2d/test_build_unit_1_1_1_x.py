"""The 1-1-1-x template must close a patch without changing its boundary."""

import unittest
from collections import Counter

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d.adapter.unit import build_unit_1_1_1_x


def _patch(segment_count, *, bulged=False, nonuniform=False):
    if nonuniform:
        fractions = np.array([0.0, 0.08, 0.23, 0.40, 0.64, 1.0])
        if segment_count != 5:
            raise ValueError("nonuniform fixture has five segments")
    else:
        fractions = np.linspace(0.0, 1.0, segment_count + 1)
    bottom_y = -0.12 * np.sin(np.pi * fractions) if bulged else np.zeros_like(fractions)
    bottom = np.column_stack((fractions, bottom_y, np.zeros_like(fractions)))
    top_left = np.array([0.03, 1.0, 0.0])
    top_right = np.array([0.96, 1.08, 0.0])
    nodes = np.vstack((bottom, top_left, top_right))
    mesh = Mesh2D(nodes, np.empty((0, 4), dtype=np.int32))
    edges = (
        np.arange(segment_count + 1, dtype=np.int64),
        np.array([segment_count, segment_count + 2], dtype=np.int64),
        np.array([segment_count + 2, segment_count + 1], dtype=np.int64),
        np.array([segment_count + 1, 0], dtype=np.int64),
    )
    return mesh, edges


def _signed_area(points):
    xy = points[:, :2]
    following = np.roll(xy, -1, axis=0)
    return 0.5 * float(np.sum(xy[:, 0] * following[:, 1] - xy[:, 1] * following[:, 0]))


class BuildUnit111XTests(unittest.TestCase):
    def assert_valid_patch(self, mesh, edges, original_node_count, original_element_count=0):
        new_quads = mesh.elements[original_element_count:]
        boundary = np.concatenate([edge[:-1] for edge in edges])
        expected_boundary_edges = {
            frozenset((int(a), int(b)))
            for edge in edges
            for a, b in zip(edge[:-1], edge[1:])
        }

        incidence = Counter(
            frozenset((int(quad[index]), int(quad[(index + 1) % 4])))
            for quad in new_quads
            for index in range(4)
        )
        self.assertEqual({edge for edge, count in incidence.items() if count == 1}, expected_boundary_edges)
        self.assertTrue(all(count == 2 for edge, count in incidence.items() if edge not in expected_boundary_edges))
        self.assertTrue(all(int(index) < original_node_count for index in boundary))

        quad_points = mesh.nodes[new_quads, :2]
        edge_vectors = np.roll(quad_points, -1, axis=1) - quad_points
        next_vectors = np.roll(edge_vectors, -1, axis=1)
        corner_cross = edge_vectors[..., 0] * next_vectors[..., 1] - edge_vectors[..., 1] * next_vectors[..., 0]
        self.assertTrue(np.all(corner_cross > 0.0))
        boundary_area = abs(_signed_area(mesh.nodes[boundary]))
        quad_area = sum(_signed_area(mesh.nodes[quad]) for quad in new_quads)
        self.assertAlmostEqual(quad_area, boundary_area, places=12)

    def test_odd_segment_counts_match_the_drawn_templates(self):
        for segment_count in (1, 3, 5, 7):
            with self.subTest(segment_count=segment_count):
                mesh, edges = _patch(segment_count)
                original_nodes = mesh.nodes.copy()
                result = build_unit_1_1_1_x(mesh, *edges)
                self.assertIs(result, mesh)
                self.assertEqual(mesh.node_count, len(original_nodes) + segment_count - 1)
                self.assertEqual(mesh.element_count, (3 * segment_count - 1) // 2)
                np.testing.assert_array_equal(mesh.nodes[:len(original_nodes)], original_nodes)
                self.assert_valid_patch(mesh, edges, len(original_nodes))

    def test_long_edge_can_be_any_edge_and_winding_can_be_reversed(self):
        for segment_count in (1, 5):
            for rotation in range(4):
                for clockwise in (False, True):
                    with self.subTest(segment_count=segment_count, rotation=rotation, clockwise=clockwise):
                        mesh, edges = _patch(segment_count, bulged=True)
                        original_node_count = mesh.node_count
                        edges = edges[rotation:] + edges[:rotation]
                        if clockwise:
                            edges = tuple(edge[::-1] for edge in edges[::-1])
                        build_unit_1_1_1_x(mesh, *edges)
                        self.assert_valid_patch(mesh, edges, original_node_count)

    def test_nonuniform_bulged_boundary_uses_arc_length_with_zero_z(self):
        mesh, edges = _patch(5, bulged=True, nonuniform=True)
        initial = mesh.nodes.copy()
        lengths = np.linalg.norm(np.diff(initial[edges[0], :2], axis=0), axis=1)
        fraction = lengths[0] / lengths.sum()
        expected = 0.5 * (
            initial[edges[0][1]]
            + (1.0 - fraction) * initial[edges[3][0]]
            + fraction * initial[edges[1][-1]]
        )
        build_unit_1_1_1_x(mesh, *edges)
        np.testing.assert_allclose(mesh.nodes[len(initial)], expected)
        self.assert_valid_patch(mesh, edges, len(initial))

    def test_existing_external_element_is_preserved(self):
        mesh, edges = _patch(3)
        remote = np.array([[3.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 1.0, 0.0], [3.0, 1.0, 0.0]])
        original_count = mesh.node_count
        mesh.replace_data(nodes=np.vstack((mesh.nodes, remote)), elements=np.array([[original_count + i for i in range(4)]]))
        original_nodes = mesh.nodes.copy()
        original_elements = mesh.elements.copy()
        build_unit_1_1_1_x(mesh, *edges)
        np.testing.assert_array_equal(mesh.nodes[:len(original_nodes)], original_nodes)
        np.testing.assert_array_equal(mesh.elements[:1], original_elements)
        self.assert_valid_patch(mesh, edges, len(original_nodes), 1)

    def test_invalid_inputs_leave_mesh_unchanged(self):
        mesh, edges = _patch(3)
        bad_cases = (
            ("nonarray", (edges[0].tolist(), *edges[1:])),
            ("dimension", (edges[0].reshape(1, -1), *edges[1:])),
            ("dtype", (edges[0].astype(float), *edges[1:])),
            ("missing_join", (edges[0], np.array([0, 5]), *edges[2:])),
            ("out_of_range", (np.array([0, 1, 2, 99]), *edges[1:])),
            ("duplicate", (np.array([0, 1, 0, 3]), *edges[1:])),
            ("odd_point_count", (np.array([0, 1, 3]), *edges[1:])),
            ("two_long_edges", (edges[0], np.array([3, 1, 2, 5]), edges[2], edges[3])),
        )
        for name, bad_edges in bad_cases:
            with self.subTest(name=name):
                nodes, elements = mesh.nodes.copy(), mesh.elements.copy()
                with self.assertRaises(ValueError):
                    build_unit_1_1_1_x(mesh, *bad_edges)
                np.testing.assert_array_equal(mesh.nodes, nodes)
                np.testing.assert_array_equal(mesh.elements, elements)

    def test_invalid_geometry_is_rejected_transactionally(self):
        for name, change in (
            ("nonfinite", (1, [np.nan, 0.0, 0.0])),
            ("zero_segment", (1, [0.0, 0.0, 0.0])),
            ("concave", (1, [1.0 / 3.0, 0.2, 0.0])),
        ):
            with self.subTest(name=name):
                mesh, edges = _patch(3)
                nodes = mesh.nodes.copy()
                nodes[change[0]] = change[1]
                mesh.nodes = nodes
                original_nodes = mesh.nodes.copy()
                original_elements = mesh.elements.copy()
                with self.assertRaises(ValueError):
                    build_unit_1_1_1_x(mesh, *edges)
                np.testing.assert_array_equal(mesh.nodes, original_nodes)
                np.testing.assert_array_equal(mesh.elements, original_elements)

        crossed = Mesh2D([[0, 0], [1, 1], [0, 1], [1, 0]], np.empty((0, 4), dtype=np.int32))
        edges = tuple(np.array(edge) for edge in ([0, 1], [1, 3], [3, 2], [2, 0]))
        with self.assertRaises(ValueError):
            build_unit_1_1_1_x(crossed, *edges)
        self.assertEqual(crossed.element_count, 0)


if __name__ == "__main__":
    unittest.main()
