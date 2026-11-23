"""The 1-1-1-x template must close a patch without changing its boundary."""

import unittest
from collections import Counter

import numpy as np

from ._fixtures import assert_ccw

from mesher import Mesh2D
from mesher.mesh2d.adapter.unit import build_unit_1_1_1_x, build_unit_1_1_x_y, build_unit_1_x_y_z
from mesher.mesh2d.quality import MeshQualityChecker


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


def _rectangle(xs):
    mesh, edges = _patch(len(xs) - 1)
    mesh.nodes[:-2, 0] = xs
    mesh.nodes[-2:, :2] = [[0, .75], [1, .75]]
    return mesh, edges


class BuildUnit111XTests(unittest.TestCase):
    def assert_valid_patch(self, mesh, edges, original_node_count, original_element_count=0):
        assert_ccw(mesh)
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

    def test_short_end_opens_in_the_local_frame(self):
        angle = .8
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        for mirror in (False, True):
            with self.subTest(mirror=mirror):
                mesh, edges = _rectangle([0, .4, .97, 1])
                # Midpoint of the bottom short-edge endpoint and the 2/3 point above it.
                expected = np.array([(.97 + 2 / 3) / 2, .375])
                if mirror:
                    mesh.nodes[:, 0] = 1 - mesh.nodes[:, 0]
                    expected[0] = 1 - expected[0]
                mesh.nodes[:, :2] = mesh.nodes[:, :2] @ rotation
                boundary = mesh.nodes.copy()
                build_unit_1_1_1_x(mesh, *edges)
                np.testing.assert_array_equal(mesh.nodes[:6], boundary)
                np.testing.assert_allclose(mesh.nodes[6 if mirror else 7, :2], expected @ rotation)
                self.assert_valid_patch(mesh, edges, 6)
                metrics = MeshQualityChecker(mesh)
                self.assertAlmostEqual(metrics.calculate_scaled_jacobian().summary.minimum, .435979, places=6)
                self.assertAlmostEqual(metrics.calculate_aspect_ratio().summary.maximum, 25.)

    def test_end_at_quarter_mean_uses_arc_length_after_rotation(self):
        mesh, edges = _rectangle([0, .4, 11 / 12, 1])
        angle = .8
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        mesh.nodes[:, :2] = mesh.nodes[:, :2] @ rotation
        build_unit_1_1_1_x(mesh, *edges)
        np.testing.assert_allclose(mesh.nodes[7, :2] @ rotation.T, [11 / 12, .375])

    def test_short_ends_in_multiple_peels_preserve_connectivity(self):
        for xs, expected in (
            ([0, .25, .5, .7, .97, 1], [.25, .885, .5, .7]),
            ([0, .1, .3, .6, .89, .90, .99, 1], [.1, (.99 + 6 / 7) / 2, .3, .90, .6, .795]),
        ):
            with self.subTest(segments=len(xs) - 1):
                mesh, edges = _rectangle(xs)
                regular, _ = _rectangle(np.linspace(0., 1., len(xs)))
                boundary = mesh.nodes.copy()
                build_unit_1_1_1_x(regular, *edges)
                build_unit_1_1_1_x(mesh, *edges)
                np.testing.assert_array_equal(mesh.elements, regular.elements)
                np.testing.assert_array_equal(mesh.nodes[:len(boundary)], boundary)
                np.testing.assert_allclose(mesh.nodes[len(boundary):, 0], expected)
                self.assert_valid_patch(mesh, edges, len(boundary))

    def test_short_end_stays_put_when_it_would_squeeze_the_next_peel(self):
        # At x=.895 the first peel would open, but the quad against x=.893
        # in the second peel would have a smaller minimum scaled Jacobian.
        mesh, edges = _rectangle([0, .3, .6, .893, .99, 1])
        build_unit_1_1_1_x(mesh, *edges)
        np.testing.assert_allclose(mesh.nodes[9], [.99, .375, 0])
        self.assert_valid_patch(mesh, edges, 8)

    def test_short_end_respects_generated_edge_tolerance(self):
        # In this shallow trapezoid the move improves both shape metrics, but
        # shortens a generated edge from .0336 to .0287.
        for tolerance, fraction in ((.025, 2 / 3), (.032, .94)):
            with self.subTest(tolerance=tolerance):
                mesh, edges = _rectangle([0, .38, .94, 1])
                mesh.nodes[-2:, :2] = [[-.28, .03], [1.6, .055]]
                left, right = mesh.nodes[-2:, :2]
                target = left + fraction * (right - left)
                expected = (mesh.nodes[2, :2] + target) / 2
                build_unit_1_1_1_x(mesh, *edges, tolerance=tolerance)
                np.testing.assert_allclose(mesh.nodes[7, :2], expected)

    def test_short_end_rounding_back_to_world_preserves_a_valid_template(self):
        # The candidate edge measures .028674406 locally, .028674390 after restoration.
        offset = np.array([-1e9, -3.8e8])
        for build in (build_unit_1_1_1_x, build_unit_1_1_x_y, build_unit_1_x_y_z):
            with self.subTest(build=build.__name__):
                mesh, edges = _rectangle([0, .38, .94, 1])
                mesh.nodes[-2:, :2] = [[-.28, .03], [1.6, .055]]
                mesh.nodes[:, :2] += offset
                build(mesh, *edges, tolerance=.0286744)
                np.testing.assert_allclose(mesh.nodes[7, :2] - offset, [1.2136, .02675], atol=2e-7)

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
