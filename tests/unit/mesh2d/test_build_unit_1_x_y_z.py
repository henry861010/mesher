"""Unequal boundary chains must form a conforming, transactional Quad4 patch."""

import unittest
from collections import Counter

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d.adapter.unit import (
    build_unit_1_1_1_x,
    build_unit_1_1_x_y,
    build_unit_1_x_y_z,
)


def _patch(left_y, right_y, bottom_x, *, wavy=False):
    left_y, right_y, bottom_x = map(np.asarray, (left_y, right_y, bottom_x))
    left_x = 0.025 * np.sin(2.0 * np.pi * left_y) if wavy else np.zeros_like(left_y)
    right_x = 1.0 + (0.03 * np.sin(2.0 * np.pi * right_y) if wavy else np.zeros_like(right_y))
    bottom_y = 0.02 * np.sin(2.0 * np.pi * bottom_x[1:-1]) if wavy else np.zeros(len(bottom_x) - 2)
    xy = np.vstack((
        np.column_stack((left_x, left_y)),
        np.column_stack((right_x, right_y)),
        np.column_stack((bottom_x[1:-1], bottom_y)),
    ))
    nodes = np.column_stack((xy, 0.3 * xy[:, 0] + 0.4 * xy[:, 1]))
    left = np.arange(len(left_y), dtype=np.int64)
    right = np.arange(len(left_y), len(left_y) + len(right_y), dtype=np.int64)
    bottom = np.r_[left[-1], np.arange(len(left_y) + len(right_y), len(nodes)), right[-1]]
    edges = (np.array([right[0], left[0]]), left, bottom, right[::-1])
    return Mesh2D(nodes, np.empty((0, 4), dtype=np.int32)), edges


def _uniform_patch(left, right, bottom, **kwargs):
    return _patch(np.linspace(1, 0, left + 1), np.linspace(1, 0, right + 1),
                  np.linspace(0, 1, bottom + 1), **kwargs)


def _sketch_patch(**kwargs):
    return _patch([1, 0.76, 0.54, 0.27, 0], [1, 0.64, 0.13, 0],
                  np.linspace(0, 1, 5), **kwargs)


def _polygon_patch(xy, edges, z=None):
    xy = np.asarray(xy, dtype=float)
    if z is None:
        z = 0.3 * xy[:, 0] + 0.4 * xy[:, 1]
    mesh = Mesh2D(np.column_stack((xy, z)), np.empty((0, 4), dtype=np.int32))
    return mesh, tuple(np.array(edge, dtype=np.int64) for edge in edges)


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _area(points):
    xy = np.asarray(points)
    xy = xy - xy[0]
    return 0.5 * sum(_cross(a, b) for a, b in zip(xy, np.roll(xy, -1, axis=0)))


def _overlap_area(first, second):
    """Independently intersect two convex cells to check interior overlap."""
    polygon = list(first)
    for start, end in zip(second, np.roll(second, -1, axis=0)):
        if not polygon:
            return 0.0
        clipped = []
        previous = polygon[-1]
        previous_side = _cross(end - start, previous - start)
        for current in polygon:
            side = _cross(end - start, current - start)
            if (side >= 0.0) != (previous_side >= 0.0):
                ratio = previous_side / (previous_side - side)
                clipped.append(previous + ratio * (current - previous))
            if side >= 0.0:
                clipped.append(current)
            previous, previous_side = current, side
        polygon = clipped
    return abs(_area(polygon)) if len(polygon) >= 3 else 0.0


class BuildUnit1XYZTests(unittest.TestCase):
    def assert_valid_patch(self, mesh, edges, original_nodes, original_elements=None):
        if original_elements is None:
            original_elements = np.empty((0, 4), dtype=np.int32)
        np.testing.assert_array_equal(mesh.nodes[:len(original_nodes)], original_nodes)
        np.testing.assert_array_equal(mesh.elements[:len(original_elements)], original_elements)
        quads = mesh.elements[len(original_elements):]
        self.assertGreater(len(quads), 0)
        self.assertTrue(all(len(set(quad)) == 4 for quad in quads))
        self.assertTrue(np.all(mesh.element_types[len(original_elements):] == 4))
        self.assertTrue(np.all(np.isfinite(mesh.nodes[len(original_nodes):])))
        self.assertTrue(set(range(len(original_nodes), mesh.node_count)) <= set(quads.ravel()))

        incidence = Counter(
            (int(quad[i]), int(quad[(i + 1) % 4])) for quad in quads for i in range(4)
        )
        expected = {frozenset((int(a), int(b))) for edge in edges
                    for a, b in zip(edge[:-1], edge[1:])}
        actual = {frozenset(pair) for pair in incidence}
        self.assertTrue(expected <= actual)
        for pair in actual:
            a, b = pair
            total = incidence[(a, b)] + incidence[(b, a)]
            self.assertEqual(total, 1 if pair in expected else 2)
            if pair not in expected:
                self.assertEqual(incidence[(a, b)], 1)
                self.assertEqual(incidence[(b, a)], 1)

        boundary = np.concatenate([edge[:-1] for edge in edges])
        scale = np.ptp(mesh.nodes[boundary, :2], axis=0).max()
        xy = (mesh.nodes[:, :2] - mesh.nodes[boundary[0], :2]) / scale
        cells = xy[quads]
        vectors = np.roll(cells, -1, axis=1) - cells
        following = np.roll(vectors, -1, axis=1)
        turns = vectors[:, :, 0] * following[:, :, 1] - vectors[:, :, 1] * following[:, :, 0]
        self.assertTrue(np.all(turns > 0.0))
        self.assertAlmostEqual(sum(_area(cell) for cell in cells), abs(_area(xy[boundary])), places=9)
        for first, cell in enumerate(cells):
            for other in cells[first + 1:]:
                self.assertLess(_overlap_area(cell, other), 1e-10)

    def assert_shared_connector(self, mesh, endpoints, midpoint=None):
        start, end = map(int, endpoints)
        if midpoint is not None:
            found = np.flatnonzero(np.linalg.norm(mesh.nodes - midpoint, axis=1) < 1e-12)
            self.assertEqual(len(found), 1, "the shared midpoint must have one node index")
            middle = int(found[0])
            segments = ((start, middle), (middle, end))
        else:
            segments = ((start, end),)
        incidence = Counter(
            frozenset((int(quad[i]), int(quad[(i + 1) % 4])))
            for quad in mesh.elements for i in range(4)
        )
        for segment in segments:
            self.assertEqual(incidence[frozenset(segment)], 2)
        if midpoint is not None:
            self.assertNotIn(frozenset((start, end)), incidence)

    def test_existing_templates_are_used_when_applicable(self):
        for counts, builder in (
            ((1, 1, 1), build_unit_1_1_1_x),
            ((1, 1, 5), build_unit_1_1_1_x),
            ((1, 3, 3), build_unit_1_1_x_y),
            ((5, 1, 3), build_unit_1_1_x_y),
        ):
            with self.subTest(counts=counts):
                mesh, edges = _uniform_patch(*counts)
                original = mesh.nodes.copy()
                expected = Mesh2D(mesh.nodes, mesh.elements)
                builder(expected, *edges)
                self.assertIs(build_unit_1_x_y_z(mesh, *edges), mesh)
                np.testing.assert_array_equal(mesh.nodes, expected.nodes)
                np.testing.assert_array_equal(mesh.elements, expected.elements)
                self.assert_valid_patch(mesh, edges, original)

    def test_sketch_preserves_midpoints_through_two_segment_front_and_tail(self):
        mesh, edges = _sketch_patch()
        original = mesh.nodes.copy()
        left, right = edges[1], edges[3][::-1]
        first = 0.5 * (original[left[2]] + original[right[1]])
        second = 0.5 * (original[left[3]] + original[right[2]])
        build_unit_1_x_y_z(mesh, *edges)
        self.assert_shared_connector(mesh, (left[2], right[1]), first)
        self.assert_shared_connector(mesh, (left[3], right[2]), second)
        self.assert_valid_patch(mesh, edges, original)

    def test_two_segment_front_can_return_to_one_segment(self):
        mesh, edges = _patch([1, .9, .65, .5, .25, 0], [1, .67, .2, 0], np.linspace(0, 1, 4))
        original = mesh.nodes.copy()
        left, right = edges[1], edges[3][::-1]
        first = 0.5 * (original[left[2]] + original[right[1]])
        build_unit_1_x_y_z(mesh, *edges)
        self.assert_shared_connector(mesh, (left[2], right[1]), first)
        self.assert_shared_connector(mesh, (left[4], right[2]))
        self.assert_valid_patch(mesh, edges, original)

    def test_opposite_subdivided_sides_and_unequal_counts(self):
        for counts in ((3, 5, 1), (2, 2, 1), (8, 3, 2), (3, 8, 2), (4, 4, 3)):
            with self.subTest(counts=counts):
                mesh, edges = _uniform_patch(*counts)
                original = mesh.nodes.copy()
                build_unit_1_x_y_z(mesh, *edges)
                self.assert_valid_patch(mesh, edges, original)

    def test_wavy_patch_rotation_and_winding_preserve_input_arrays(self):
        for rotation in range(4):
            for clockwise in (False, True):
                with self.subTest(rotation=rotation, clockwise=clockwise):
                    mesh, edges = _sketch_patch(wavy=True)
                    edges = edges[rotation:] + edges[:rotation]
                    if clockwise:
                        edges = tuple(edge[::-1] for edge in edges[::-1])
                    snapshots = tuple(edge.copy() for edge in edges)
                    original = mesh.nodes.copy()
                    build_unit_1_x_y_z(mesh, *edges)
                    self.assert_valid_patch(mesh, edges, original)
                    for edge, snapshot in zip(edges, snapshots):
                        np.testing.assert_array_equal(edge, snapshot)

    def test_nonuniform_wavy_boundary_counts(self):
        rng = np.random.default_rng(20261026)
        for trial in range(24):
            left, right, bottom = map(int, rng.integers(2, 8, 3))
            bottom += (1 + left + right + bottom) % 2
            with self.subTest(trial=trial, counts=(left, right, bottom)):
                left_y = np.r_[1, 1 - np.cumsum(rng.dirichlet(np.ones(left) * 2))]
                right_y = np.r_[1, 1 - np.cumsum(rng.dirichlet(np.ones(right) * 2))]
                bottom_x = np.r_[0, np.cumsum(rng.dirichlet(np.ones(bottom) * 2))]
                mesh, edges = _patch(left_y, right_y, bottom_x, wavy=True)
                original = mesh.nodes.copy()
                build_unit_1_x_y_z(mesh, *edges)
                self.assert_valid_patch(mesh, edges, original)

    def test_concave_four_corner_patch_uses_inner_ring_and_interpolates_z(self):
        mesh, edges = _polygon_patch(
            [[0, 0], [1, 0], [.35, .3], [0, 1]], [[0, 1], [1, 2], [2, 3], [3, 0]]
        )
        original = mesh.nodes.copy()
        build_unit_1_x_y_z(mesh, *edges)
        self.assertEqual(mesh.node_count, 8)
        self.assertEqual(mesh.element_count, 5)
        self.assert_valid_patch(mesh, edges, original)
        np.testing.assert_allclose(mesh.nodes[:, 2], 0.3 * mesh.nodes[:, 0] + 0.4 * mesh.nodes[:, 1], atol=1e-12)

    def test_inner_ring_z_stays_within_nonaffine_boundary_values(self):
        mesh, edges = _polygon_patch(
            [[0, 0], [1, 0], [.35, .3], [0, 1]], [[0, 1], [1, 2], [2, 3], [3, 0]],
            z=[-2, 4, 1, 0],
        )
        original = mesh.nodes.copy()
        build_unit_1_x_y_z(mesh, *edges)
        self.assert_valid_patch(mesh, edges, original)
        self.assertTrue(np.all(mesh.nodes[4:, 2] >= -2))
        self.assertTrue(np.all(mesh.nodes[4:, 2] <= 4))

    def test_empty_visibility_kernel_does_not_prevent_successful_layers(self):
        mesh, edges = _polygon_patch(
            [[0, 1], [0, .5], [0, 0], [.247, 0], [.25, .03], [.253, 0],
             [.747, 0], [.75, .03], [.753, 0], [1, 0], [1, .5], [1, 1]],
            [[11, 0], [0, 1, 2], [2, 3, 4, 5, 6, 7, 8, 9], [9, 10, 11]],
        )
        original = mesh.nodes.copy()
        build_unit_1_x_y_z(mesh, *edges)
        self.assert_valid_patch(mesh, edges, original)

    def test_scaling_and_translation(self):
        for ring in (False, True):
            for scale, shift in ((1e-7, [0, 0, 0]), (1e7, [1e10, -2e10, 3e10]),
                                 (1, [1e10, -2e10, 3e10])):
                with self.subTest(ring=ring, scale=scale, shift=shift):
                    if ring:
                        mesh, edges = _polygon_patch(
                            [[0, 0], [1, 0], [.35, .3], [0, 1]],
                            [[0, 1], [1, 2], [2, 3], [3, 0]],
                        )
                    else:
                        mesh, edges = _sketch_patch()
                    mesh.nodes = mesh.nodes * scale + shift
                    original = mesh.nodes.copy()
                    build_unit_1_x_y_z(mesh, *edges)
                    self.assert_valid_patch(mesh, edges, original)

    def test_existing_mixed_elements_are_preserved(self):
        mesh, edges = _sketch_patch()
        count = mesh.node_count
        remote = [[3, 0, 0], [4, 0, 0], [4, 1, 0], [3, 1, 0]]
        mesh.replace_data(nodes=np.vstack((mesh.nodes, remote)),
                          elements=np.array([[count, count + 1, count + 2, count + 2]]))
        original_nodes, original_elements = mesh.nodes.copy(), mesh.elements.copy()
        build_unit_1_x_y_z(mesh, *edges)
        self.assert_valid_patch(mesh, edges, original_nodes, original_elements)

    def assert_failure_unchanged(self, mesh, edges):
        nodes_reference, elements_reference = mesh.nodes, mesh.elements
        nodes, elements = mesh.nodes.copy(), mesh.elements.copy()
        with self.assertRaises(ValueError):
            build_unit_1_x_y_z(mesh, *edges)
        self.assertIs(mesh.nodes, nodes_reference)
        self.assertIs(mesh.elements, elements_reference)
        np.testing.assert_array_equal(mesh.nodes, nodes)
        np.testing.assert_array_equal(mesh.elements, elements)

    def test_invalid_edge_arrays_are_rejected_before_mutation(self):
        mesh, edges = _sketch_patch()
        for bad in (edges[0].tolist(), edges[0].reshape(1, -1), edges[0].astype(float),
                    edges[0].astype(bool), np.array([], dtype=int), np.array([0]),
                    np.array([-1, 0]), np.array([999, 0]), np.array([0, 1])):
            with self.subTest(edge=bad):
                self.assert_failure_unchanged(mesh, (bad, *edges[1:]))
        repeated = edges[1].copy()
        repeated[1] = repeated[2]
        self.assert_failure_unchanged(mesh, (edges[0], repeated, *edges[2:]))
        with self.assertRaises(TypeError):
            build_unit_1_x_y_z(None, *edges)

    def test_invalid_counts_and_geometry_are_transactional(self):
        odd, odd_edges = _uniform_patch(2, 2, 2)
        no_short, no_short_edges = _polygon_patch(
            [[0, 0], [.5, 0], [1, 0], [1, .5], [1, 1], [.5, 1], [0, 1], [0, .5]],
            [[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 0]],
        )
        crossed, crossed_edges = _polygon_patch(
            [[0, 0], [1, 1], [0, 1], [1, 0]], [[0, 1], [1, 2], [2, 3], [3, 0]]
        )
        for mesh, edges in ((odd, odd_edges), (no_short, no_short_edges), (crossed, crossed_edges)):
            self.assert_failure_unchanged(mesh, edges)
        for bad_coordinate in ([np.nan, 0, 0], [np.inf, 0, 0], [0, 1, .4]):
            mesh, edges = _sketch_patch()
            mesh.nodes[1] = bad_coordinate
            self.assert_failure_unchanged(mesh, edges)
        mesh, edges = _sketch_patch()
        mesh.nodes[:, 1] = 0
        self.assert_failure_unchanged(mesh, edges)

    def test_failed_geometry_discards_an_already_meshed_first_layer(self):
        # A lower rectangular strip can be filled, but the upper U has no
        # valid next crosscut and an empty visibility kernel.
        mesh, edges = _polygon_patch(
            [[0, 0], [3, 0], [3, 1], [3, 3], [2, 3],
             [2, 1.5], [1, 1.5], [1, 3], [0, 3], [0, 1]],
            [[0, 1], [1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 0]],
        )
        self.assert_failure_unchanged(mesh, edges)


if __name__ == "__main__":
    unittest.main()
