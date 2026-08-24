import unittest
from collections import Counter

import numpy as np

from mesher.generators import generate_rectilinear_mesh
from mesher.circular.imprint import (
    _clear_node,
    _delete,
    _delete_element,
    _generate_pattern_circle_nodes,
    _get_boundary,
    imprint_circle,
    _search_circle,
    _to_circle,
)
from mesher import Mesh2D
from mesher.quality import MeshQualityChecker


class SearchCircleTests(unittest.TestCase):
    def setUp(self):
        self.mesh = Mesh2D(
            nodes=np.array(
                [
                    [0.0, 0.0, 4.0],
                    [1.0, 0.0, 4.0],
                    [1.0, 1.0, 4.0],
                    [0.0, 1.0, 4.0],
                    [2.0, 0.0, 4.0],
                    [2.0, 1.0, 4.0],
                ]
            ),
            elements=np.array(
                [[0, 1, 2, 3], [1, 4, 5, 2]],
                dtype=np.int64,
            ),
        )

    def test_all_requires_every_node_to_be_inside(self):
        result = _search_circle(
            self.mesh,
            x=0.5,
            y=0.5,
            radius=np.sqrt(0.5),
            type="ALL",
            tolerance=0.0,
        )

        np.testing.assert_array_equal(result, [0])

    def test_part_requires_at_least_one_node_to_be_inside(self):
        result = _search_circle(
            self.mesh,
            x=0.5,
            y=0.5,
            radius=np.sqrt(0.5),
            type="PART",
            tolerance=0.0,
        )

        np.testing.assert_array_equal(result, [0, 1])

    def test_tolerance_includes_a_node_just_outside_the_radius(self):
        result = _search_circle(
            self.mesh,
            x=0.0,
            y=0.0,
            radius=0.99,
            type="PART",
            tolerance=0.01,
        )

        np.testing.assert_array_equal(result, [0, 1])

    def test_invalid_type_raises(self):
        with self.assertRaisesRegex(ValueError, "ALL.*PART"):
            _search_circle(
                self.mesh,
                x=0.0,
                y=0.0,
                radius=1.0,
                type="SOME",
                tolerance=0.0,
            )


class ToCircleTests(unittest.TestCase):
    @staticmethod
    def _outward_mesh(node_dtype=np.float64, element_dtype=np.int32):
        nodes = np.array(
            [
                [-1.0, 0.0],
                [0.5, 0.0],
                [1.0, 0.0],
                [-1.0, 0.5],
                [0.5, 0.5],
                [1.0, 0.5],
            ],
            dtype=node_dtype,
        )
        elements = np.array(
            [[0, 1, 4, 3], [1, 2, 5, 4]],
            dtype=element_dtype,
        )
        return Mesh2D(nodes=nodes, elements=elements)

    @staticmethod
    def _off_center_pattern_mesh():
        return Mesh2D(
            nodes=np.array(
                [[3.0, -4.0], [4.0, -2.0], [3.0, 1.0], [-1.0, -2.0]],
                dtype=np.float64,
            ),
            elements=np.array([[0, 1, 2, 3]], dtype=np.int32),
        )

    @staticmethod
    def _closed_pattern_mesh():
        pattern_angle = np.arccos(-2.5 / 4.0)
        boundary_angles = np.concatenate(
            (
                np.deg2rad([0.0, 60.0, 110.0, 125.0]),
                [pattern_angle],
                np.deg2rad([180.0, 240.0, 300.0]),
            )
        )
        boundary = 4.0 * np.column_stack(
            (np.cos(boundary_angles), np.sin(boundary_angles))
        )
        nodes = np.vstack(([0.0, 0.0], boundary))
        boundary_indices = np.arange(1, boundary.shape[0] + 1, dtype=np.int64)
        elements = []
        for position, source in enumerate(boundary_indices):
            target = boundary_indices[(position + 1) % boundary_indices.size]
            elements.append([0, source, target, target])
        source_y = np.sqrt(4.0**2 - 2.5**2)
        target_y = np.sqrt(5.0**2 - 2.5**2)
        line = [[-2.5, source_y], [-2.5, target_y]]
        return (
            Mesh2D(nodes=nodes, elements=np.asarray(elements, dtype=np.int32)),
            boundary_indices,
            line,
            np.array([-2.5, target_y]),
        )

    @staticmethod
    def snapshot_mesh(mesh):
        return (
            mesh.nodes,
            mesh.elements,
            np.asarray(mesh.nodes).copy(),
            np.asarray(mesh.elements).copy(),
        )

    def assert_mesh_unchanged(self, mesh, snapshot):
        original_nodes, original_elements, node_values, element_values = snapshot
        self.assertIs(mesh.nodes, original_nodes)
        self.assertIs(mesh.elements, original_elements)
        np.testing.assert_array_equal(mesh.nodes, node_values)
        np.testing.assert_array_equal(mesh.elements, element_values)

    def test_extends_outward_as_an_open_one_to_one_quad_strip(self):
        mesh = self._outward_mesh()
        original_nodes = mesh.nodes.copy()
        original_elements = mesh.elements.copy()

        result = _to_circle(mesh, 0.0, 0.0, 2.0, [3, 4, 5])

        self.assertIs(result, mesh)
        self.assertEqual(mesh.nodes.shape, (9, 2))
        self.assertEqual(mesh.elements.shape, (4, 4))
        np.testing.assert_array_equal(mesh.nodes[:6], original_nodes)
        np.testing.assert_array_equal(mesh.elements[:2], original_elements)
        expected_targets = original_nodes[[3, 4, 5]].astype(np.float64)
        expected_targets *= 2.0 / np.linalg.norm(expected_targets, axis=1)[:, None]
        # Open endpoints are fixed while the free middle node is smoothed to
        # the angular midpoint of its two neighbours.
        expected_targets[1] = [0.0, 2.0]
        np.testing.assert_allclose(
            mesh.nodes[6:], expected_targets, atol=1.0e-15
        )
        np.testing.assert_array_equal(
            mesh.elements[2:],
            np.array([[3, 4, 7, 6], [4, 5, 8, 7]], dtype=np.int32),
        )
        all_edges = {
            tuple(sorted((int(start), int(end))))
            for element in mesh.elements
            for start, end in zip(element, np.roll(element, -1))
            if start != end
        }
        self.assertNotIn((6, 8), all_edges)
        report = MeshQualityChecker(mesh).check_jacobian(indices=[2, 3])
        self.assertEqual(report.invalid_indices.size, 0)
        self.assertTrue(np.all(report.values > 0.0))

    def test_closed_smoothing_increases_short_gap_without_lowering_quality(self):
        mesh, boundary_indices, _, _ = self._closed_pattern_mesh()
        original_node_count = mesh.nodes.shape[0]
        original_element_count = mesh.elements.shape[0]
        radial_targets = (
            mesh.nodes[boundary_indices]
            * 5.0
            / np.linalg.norm(mesh.nodes[boundary_indices], axis=1)[:, None]
        )

        _to_circle(
            mesh,
            0.0,
            0.0,
            5.0,
            boundary_indices,
            closed=True,
        )

        target_indices = np.arange(original_node_count, mesh.nodes.shape[0])
        new_element_indices = np.arange(
            original_element_count,
            mesh.elements.shape[0],
        )

        def minimum_gap(points):
            angles = np.sort(
                np.mod(np.arctan2(points[:, 1], points[:, 0]), 2.0 * np.pi)
            )
            return float(
                np.min(
                    np.diff(np.concatenate((angles, angles[:1] + 2.0 * np.pi)))
                )
            )

        baseline_mesh = Mesh2D(
            nodes=np.asarray(mesh.nodes).copy(),
            elements=np.asarray(mesh.elements).copy(),
        )
        baseline_mesh.nodes[target_indices, :2] = radial_targets
        baseline_quality = MeshQualityChecker(
            baseline_mesh
        ).calculate_scaled_jacobian(indices=new_element_indices)
        smoothed_quality = MeshQualityChecker(mesh).calculate_scaled_jacobian(
            indices=new_element_indices
        )

        self.assertGreater(
            minimum_gap(mesh.nodes[target_indices, :2]),
            minimum_gap(radial_targets),
        )
        self.assertGreaterEqual(
            float(np.min(smoothed_quality.values)) + 1.0e-13,
            float(np.min(baseline_quality.values)),
        )
        np.testing.assert_allclose(
            np.linalg.norm(mesh.nodes[target_indices, :2], axis=1),
            np.full(target_indices.size, 5.0),
        )
        self.assertEqual(smoothed_quality.invalid_indices.size, 0)

    def test_pattern_can_fix_every_circle_node_during_smoothing(self):
        source_nodes = np.array(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
            dtype=np.float64,
        )
        mesh = Mesh2D(
            nodes=source_nodes.copy(),
            elements=np.array([[0, 1, 2, 3]], dtype=np.int32),
        )
        guide_segments = [
            [[1.0, 0.0], [2.0, 0.0]],
            [[0.0, 1.0], [0.0, 2.0]],
            [[-1.0, 0.0], [-2.0, 0.0]],
            [[0.0, -1.0], [0.0, -2.0]],
        ]

        _to_circle(
            mesh,
            0.0,
            0.0,
            2.0,
            [0, 1, 2, 3],
            guide_segments=guide_segments,
            closed=True,
        )

        np.testing.assert_array_equal(mesh.nodes[4:, :2], 2.0 * source_nodes)
        report = MeshQualityChecker(mesh).check_jacobian(indices=[1, 2, 3, 4])
        self.assertEqual(report.invalid_indices.size, 0)
        self.assertTrue(np.all(report.values > 0.0))

    def test_extends_a_closed_loop_with_one_quad_per_boundary_edge(self):
        mesh = Mesh2D(
            nodes=np.array(
                [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
                dtype=np.float64,
            ),
            elements=np.array([[0, 1, 2, 3]], dtype=np.int32),
        )

        result = _to_circle(mesh, 0.0, 0.0, 2.0, [0, 1, 2, 3])

        self.assertIs(result, mesh)
        self.assertEqual(mesh.nodes.shape, (8, 2))
        self.assertEqual(mesh.elements.shape, (5, 4))
        np.testing.assert_allclose(
            mesh.nodes[4:],
            np.sqrt(2.0)
            * np.array(
                [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
            ),
        )
        np.testing.assert_array_equal(
            mesh.elements[1:],
            np.array(
                [[1, 0, 4, 5], [2, 1, 5, 6], [3, 2, 6, 7], [0, 3, 7, 4]],
                dtype=np.int32,
            ),
        )
        report = MeshQualityChecker(mesh).check_jacobian(indices=[1, 2, 3, 4])
        self.assertEqual(report.invalid_indices.size, 0)
        self.assertTrue(np.all(report.values > 0.0))

    def test_extends_an_outer_boundary_inward_with_ccw_connectivity(self):
        mesh = Mesh2D(
            nodes=np.array(
                [[2.0, -1.0], [4.0, -1.0], [4.0, 1.0], [2.0, 1.0]],
                dtype=np.float64,
            ),
            elements=np.array([[0, 1, 2, 3]], dtype=np.int32),
        )

        _to_circle(mesh, 0.0, 0.0, 1.0, [0, 3])

        np.testing.assert_allclose(
            mesh.nodes[4:],
            np.array(
                [
                    [2.0 / np.sqrt(5.0), -1.0 / np.sqrt(5.0)],
                    [2.0 / np.sqrt(5.0), 1.0 / np.sqrt(5.0)],
                ]
            ),
        )
        np.testing.assert_array_equal(mesh.elements[-1], [0, 3, 5, 4])
        report = MeshQualityChecker(mesh).check_jacobian(indices=[1])
        self.assertEqual(report.invalid_indices.size, 0)
        self.assertGreater(report.values[0], 0.0)

    def test_promotes_nodes_preserves_element_dtype_and_copies_z(self):
        base = Mesh2D(
            nodes=np.array(
                [
                    [-2, 0],
                    [1, 0],
                    [2, 0],
                    [-2, 1],
                    [1, 1],
                    [2, 1],
                ],
                dtype=np.int32,
            ),
            elements=np.array(
                [[0, 1, 4, 3], [1, 2, 5, 4]],
                dtype=np.int16,
            ),
        )
        z = np.array([10, 20, 30, 40, 50, 60], dtype=np.int32)
        original_nodes = np.column_stack((base.nodes, z))
        mesh = Mesh2D(nodes=original_nodes, elements=base.elements)

        _to_circle(mesh, 0.0, 0.0, 4.0, [3, 4, 5])

        self.assertEqual(mesh.nodes.dtype, np.dtype(np.float64))
        self.assertEqual(mesh.elements.dtype, np.dtype(np.int16))
        np.testing.assert_array_equal(mesh.nodes[:6], original_nodes.astype(np.float64))
        np.testing.assert_array_equal(mesh.nodes[6:, 2], z[[3, 4, 5]])

    def test_promotes_element_dtype_when_appended_indices_do_not_fit(self):
        mesh = self._outward_mesh(element_dtype=np.int8)
        padding = np.column_stack(
            (
                np.arange(124, dtype=np.float64) + 10.0,
                np.full(124, 10.0),
            )
        )
        mesh.nodes = np.vstack((mesh.nodes, padding))

        _to_circle(mesh, 0.0, 0.0, 2.0, [3, 4, 5])

        self.assertEqual(mesh.elements.dtype, np.dtype(np.int64))
        np.testing.assert_array_equal(mesh.elements[-2:, -2:], [[131, 130], [132, 131]])

    def test_vertical_pattern_becomes_a_connector_edge(self):
        mesh = self._outward_mesh()
        target_y = np.sqrt(4.0 - 0.5**2)

        _to_circle(
            mesh,
            0.0,
            0.0,
            2.0,
            [3, 4, 5],
            guide_segments=[[[0.5, 0.5], [0.5, 2.0]]],
        )

        np.testing.assert_allclose(mesh.nodes[7, :2], [0.5, target_y])
        connector = {4, 7}
        connector_uses = 0
        for element in mesh.elements[2:]:
            edges = zip(element, np.roll(element, -1))
            connector_uses += sum({int(a), int(b)} == connector for a, b in edges)
        self.assertEqual(connector_uses, 2)

    def test_horizontal_pattern_and_reversed_endpoints_are_supported(self):
        mesh = Mesh2D(
            nodes=np.array(
                [
                    [0.0, -1.0],
                    [0.0, 0.5],
                    [0.0, 1.0],
                    [0.5, -1.0],
                    [0.5, 0.5],
                    [0.5, 1.0],
                ],
                dtype=np.float64,
            ),
            elements=np.array(
                [[0, 3, 4, 1], [1, 4, 5, 2]],
                dtype=np.int32,
            ),
        )

        _to_circle(
            mesh,
            0.0,
            0.0,
            2.0,
            [3, 4, 5],
            guide_segments=[[[2.0, 0.5], [0.5, 0.5]]],
        )

        np.testing.assert_allclose(mesh.nodes[7, :2], [np.sqrt(3.75), 0.5])

    def test_vertical_pattern_can_overlap_a_boundary_edge(self):
        mesh = self._outward_mesh()
        original_xy = mesh.nodes.copy()
        mesh.nodes = np.column_stack((-original_xy[:, 1], original_xy[:, 0]))

        _to_circle(
            mesh,
            0.0,
            0.0,
            2.0,
            [3, 4, 5],
            guide_segments=[[[-0.5, 0.5], [-0.5, 2.0]]],
        )

        np.testing.assert_allclose(mesh.nodes[8, :2], [-0.5, np.sqrt(3.75)])
        triangle_indices = np.flatnonzero(
            mesh.elements[:, 2] == mesh.elements[:, 3]
        )
        np.testing.assert_array_equal(triangle_indices, [3, 4])
        report = MeshQualityChecker(mesh).check_jacobian(indices=[2, 3, 4])
        self.assertEqual(report.invalid_indices.size, 0)
        self.assertTrue(np.all(report.values > 0.0))

    def test_multi_edge_overlap_uses_the_vertex_nearest_the_circle(self):
        x_coordinates = np.array([-1.0, 0.0, 0.5, 1.0])
        mesh = Mesh2D(
            nodes=np.vstack(
                (
                    np.column_stack((x_coordinates, np.zeros(4))),
                    np.column_stack((x_coordinates, np.full(4, 0.5))),
                )
            ),
            elements=np.array(
                [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6]],
                dtype=np.int32,
            ),
        )

        _to_circle(
            mesh,
            0.0,
            0.0,
            2.0,
            [4, 5, 6, 7],
            guide_segments=[[[0.0, 0.5], [2.0, 0.5]]],
        )

        np.testing.assert_allclose(mesh.nodes[11, :2], [np.sqrt(3.75), 0.5])
        self.assertEqual(
            np.count_nonzero(mesh.elements[:, 2] == mesh.elements[:, 3]),
            2,
        )
        new_indices = list(range(3, mesh.elements.shape[0]))
        report = MeshQualityChecker(mesh).check_jacobian(indices=new_indices)
        self.assertEqual(report.invalid_indices.size, 0)
        self.assertTrue(np.all(report.values > 0.0))

    def test_pattern_inside_collinear_edge_without_vertex_is_atomic_error(self):
        mesh = self._outward_mesh()
        snapshot = self.snapshot_mesh(mesh)

        with self.assertRaisesRegex(ValueError, "without a vertex"):
            _to_circle(
                mesh,
                0.0,
                0.0,
                2.0,
                [3, 4, 5],
                guide_segments=[[[0.6, 0.5], [0.9, 0.5]]],
            )

        self.assert_mesh_unchanged(mesh, snapshot)

    def test_closed_pattern_redistributes_neighbouring_circle_nodes(self):
        mesh, boundary_indices, line, expected_anchor = self._closed_pattern_mesh()
        original_node_count = mesh.nodes.shape[0]
        original_element_count = mesh.elements.shape[0]
        pattern_position = 4
        neighbour_position = pattern_position - 1
        neighbour_radial_target = (
            mesh.nodes[boundary_indices[neighbour_position]]
            * 5.0
            / np.linalg.norm(mesh.nodes[boundary_indices[neighbour_position]])
        )

        _to_circle(
            mesh,
            0.0,
            0.0,
            5.0,
            boundary_indices,
            guide_segments=[line],
            closed=True,
        )

        targets = mesh.nodes[original_node_count:]
        np.testing.assert_allclose(targets[pattern_position], expected_anchor)
        self.assertFalse(
            np.allclose(targets[neighbour_position], neighbour_radial_target)
        )
        target_angles = np.arctan2(targets[:, 1], targets[:, 0])
        cyclic_steps = np.mod(
            np.diff(np.concatenate((target_angles, target_angles[:1]))),
            2.0 * np.pi,
        )
        self.assertTrue(np.all(cyclic_steps > 0.0))
        np.testing.assert_allclose(np.sum(cyclic_steps), 2.0 * np.pi)
        self.assertEqual(
            mesh.elements.shape[0], original_element_count + boundary_indices.size
        )
        report = MeshQualityChecker(mesh).check_jacobian(
            indices=list(
                range(original_element_count, mesh.elements.shape[0])
            )
        )
        self.assertEqual(report.invalid_indices.size, 0)
        self.assertTrue(np.all(report.values > 0.0))

    def test_closed_pattern_redistribution_handles_the_loop_seam(self):
        mesh, boundary_indices, line, expected_anchor = self._closed_pattern_mesh()
        pattern_position = 4
        rotated_indices = np.roll(boundary_indices, -pattern_position)
        original_node_count = mesh.nodes.shape[0]
        preceding_source = rotated_indices[-1]
        preceding_radial_target = (
            mesh.nodes[preceding_source]
            * 5.0
            / np.linalg.norm(mesh.nodes[preceding_source])
        )

        _to_circle(
            mesh,
            0.0,
            0.0,
            5.0,
            rotated_indices,
            guide_segments=[line],
            closed=True,
        )

        targets = mesh.nodes[original_node_count:]
        np.testing.assert_allclose(targets[0], expected_anchor)
        self.assertFalse(np.allclose(targets[-1], preceding_radial_target))
        target_angles = np.arctan2(targets[:, 1], targets[:, 0])
        cyclic_steps = np.mod(
            np.diff(np.concatenate((target_angles, target_angles[:1]))),
            2.0 * np.pi,
        )
        self.assertTrue(np.all(cyclic_steps > 0.0))
        np.testing.assert_allclose(np.sum(cyclic_steps), 2.0 * np.pi)

    def test_closed_smoothing_is_independent_of_seam_and_traversal(self):
        _, boundary_indices, _, _ = self._closed_pattern_mesh()
        orders = (
            boundary_indices,
            np.roll(boundary_indices, -3),
            boundary_indices[::-1],
        )
        targets_by_source = []

        for order in orders:
            mesh, _, current_line, _ = self._closed_pattern_mesh()
            node_start = mesh.nodes.shape[0]
            _to_circle(
                mesh,
                0.0,
                0.0,
                5.0,
                order,
                guide_segments=[current_line],
                closed=True,
            )
            mapped = {
                int(source): mesh.nodes[node_start + position, :2]
                for position, source in enumerate(order)
            }
            targets_by_source.append(
                np.asarray([mapped[int(source)] for source in boundary_indices])
            )

        np.testing.assert_allclose(
            targets_by_source[1], targets_by_source[0], atol=1.0e-13
        )
        np.testing.assert_allclose(
            targets_by_source[2], targets_by_source[0], atol=1.0e-13
        )

    def test_irrelevant_finite_pattern_segment_is_ignored(self):
        plain = self._outward_mesh()
        with_irrelevant_line = self._outward_mesh()

        _to_circle(plain, 0.0, 0.0, 2.0, [3, 4, 5])
        _to_circle(
            with_irrelevant_line,
            0.0,
            0.0,
            2.0,
            [3, 4, 5],
            guide_segments=[[[0.5, -5.0], [0.5, -4.0]]],
        )

        np.testing.assert_allclose(with_irrelevant_line.nodes, plain.nodes)
        np.testing.assert_array_equal(with_irrelevant_line.elements, plain.elements)

    def test_pattern_intersection_inside_boundary_edge_is_atomic_error(self):
        mesh = self._outward_mesh()
        snapshot = self.snapshot_mesh(mesh)

        with self.assertRaisesRegex(ValueError, "without a vertex"):
            _to_circle(
                mesh,
                0.0,
                0.0,
                2.0,
                [3, 4, 5],
                guide_segments=[[[0.0, 0.5], [0.0, 2.0]]],
            )

        self.assert_mesh_unchanged(mesh, snapshot)

    def test_pattern_intersection_inside_closed_seam_is_atomic_error(self):
        mesh = Mesh2D(
            nodes=np.array(
                [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
                dtype=np.float64,
            ),
            elements=np.array([[0, 1, 2, 3]], dtype=np.int32),
        )
        snapshot = self.snapshot_mesh(mesh)

        with self.assertRaisesRegex(ValueError, "without a vertex"):
            _to_circle(
                mesh,
                0.0,
                0.0,
                2.0,
                [0, 1, 2, 3],
                guide_segments=[[[-2.0, 0.0], [-1.0, 0.0]]],
                closed=True,
            )

        self.assert_mesh_unchanged(mesh, snapshot)

    def test_active_pattern_that_does_not_reach_circle_is_atomic_error(self):
        mesh = self._outward_mesh()
        snapshot = self.snapshot_mesh(mesh)

        with self.assertRaisesRegex(ValueError, "does not reach"):
            _to_circle(
                mesh,
                0.0,
                0.0,
                2.0,
                [3, 4, 5],
                guide_segments=[[[0.5, 0.5], [0.5, 1.0]]],
            )

        self.assert_mesh_unchanged(mesh, snapshot)

    def test_conflicting_pattern_constraints_are_atomic_error(self):
        mesh = self._off_center_pattern_mesh()
        snapshot = self.snapshot_mesh(mesh)
        lower_y = -2.0 - np.sqrt(21.0)
        right_x = 1.0 + np.sqrt(21.0)

        with self.assertRaisesRegex(ValueError, "conflicting"):
            _to_circle(
                mesh,
                1.0,
                -2.0,
                5.0,
                [0, 1, 2, 3],
                guide_segments=[
                    [[3.0, -4.0], [3.0, lower_y]],
                    [[3.0, -4.0], [right_x, -4.0]],
                ],
            )

        self.assert_mesh_unchanged(mesh, snapshot)

    def test_vertical_and_horizontal_patterns_select_the_nearest_circle_roots(self):
        mesh = self._off_center_pattern_mesh()
        root = np.sqrt(21.0)

        _to_circle(
            mesh,
            1.0,
            -2.0,
            5.0,
            [0, 1, 2, 3],
            guide_segments=[
                [[3.0, -2.0 - root], [3.0, -2.0 + root]],
                [[-4.0, -2.0], [6.0, -2.0]],
            ],
        )

        np.testing.assert_allclose(
            mesh.nodes[4:],
            np.array(
                [
                    [3.0, -2.0 - root],
                    [6.0, -2.0],
                    [3.0, -2.0 + root],
                    [-4.0, -2.0],
                ]
            ),
        )

    def test_rejects_invalid_pattern_segments_without_mutation(self):
        invalid_lines = (
            [[[0.5, 0.5], [1.0, 1.0]]],
            [[[0.5, 0.5], [0.5, 0.5]]],
            [[0.5, 0.5], [0.5, 2.0]],
            [[[0.5, 0.5], [0.5, np.nan]]],
        )
        for guide_segments in invalid_lines:
            with self.subTest(guide_segments=guide_segments):
                mesh = self._outward_mesh()
                snapshot = self.snapshot_mesh(mesh)

                with self.assertRaises(ValueError):
                    _to_circle(mesh, 0.0, 0.0, 2.0, [3, 4, 5], guide_segments)

                self.assert_mesh_unchanged(mesh, snapshot)

    def test_rejects_invalid_circle_and_index_inputs_atomically(self):
        invalid_calls = (
            ((0.0, 0.0, 0.0, [3, 4, 5]), ValueError),
            ((0.0, 0.0, -1.0, [3, 4, 5]), ValueError),
            ((np.inf, 0.0, 2.0, [3, 4, 5]), ValueError),
            ((0.0, 0.0, np.nan, [3, 4, 5]), ValueError),
            ((0.0, 0.0, 2.0, []), ValueError),
            ((0.0, 0.0, 2.0, [3]), ValueError),
            ((0.0, 0.0, 2.0, [[3, 4]]), ValueError),
            ((0.0, 0.0, 2.0, [3.0, 4.0]), TypeError),
            ((0.0, 0.0, 2.0, [True, False]), TypeError),
            ((0.0, 0.0, 2.0, [3, 3]), ValueError),
            ((0.0, 0.0, 2.0, [-1, 3]), IndexError),
            ((0.0, 0.0, 2.0, [3, 6]), IndexError),
            ((0.0, 0.0, 2.0, [0, 4]), ValueError),
        )
        for arguments, expected_error in invalid_calls:
            with self.subTest(arguments=arguments):
                mesh = self._outward_mesh()
                snapshot = self.snapshot_mesh(mesh)

                with self.assertRaises(expected_error):
                    _to_circle(mesh, *arguments)

                self.assert_mesh_unchanged(mesh, snapshot)

    def test_rejects_chain_on_or_crossing_circle_atomically(self):
        cases = (
            (1.0, [3, 4, 5]),
            (np.sqrt(1.25), [3, 4, 5]),
        )
        for radius, indices in cases:
            with self.subTest(radius=radius):
                mesh = self._outward_mesh()
                snapshot = self.snapshot_mesh(mesh)

                with self.assertRaisesRegex(ValueError, "same side"):
                    _to_circle(mesh, 0.0, 0.0, radius, indices)

                self.assert_mesh_unchanged(mesh, snapshot)

    def test_rejects_duplicate_radial_targets(self):
        duplicate_target_mesh = Mesh2D(
            nodes=np.array(
                [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]],
                dtype=np.float64,
            ),
            elements=np.array([[0, 1, 2, 3]], dtype=np.int32),
        )
        with self.assertRaisesRegex(ValueError, "angular order"):
            _to_circle(duplicate_target_mesh, 0.0, 0.0, 2.0, [0, 1])

    def test_rejects_pattern_anchors_with_reversed_order_atomically(self):
        reversal_mesh = Mesh2D(
            nodes=np.array(
                [
                    [-0.6, -0.6],
                    [0.6, -0.6],
                    [0.8, 0.2],
                    [0.6, 0.35],
                    [-0.6, 0.6],
                ],
                dtype=np.float64,
            ),
            elements=np.array(
                [[0, 1, 2, 4], [2, 3, 4, 4]],
                dtype=np.int32,
            ),
        )
        snapshot = self.snapshot_mesh(reversal_mesh)
        with self.assertRaisesRegex(ValueError, "pattern constraints.*angular order"):
            _to_circle(
                reversal_mesh,
                0.0,
                0.0,
                1.0,
                [0, 1, 2, 3, 4],
                guide_segments=[
                    [[0.8, 0.2], [0.8, 0.6]],
                    [[0.6, 0.35], [np.sqrt(1.0 - 0.35**2), 0.35]],
                ],
                closed=True,
            )
        self.assert_mesh_unchanged(reversal_mesh, snapshot)

    def test_rejects_folded_quad_even_when_circle_order_is_valid(self):
        mesh = Mesh2D(
            nodes=np.array(
                [[0.1, -0.5], [0.5, 0.5], [-0.5, 0.25], [-0.4, -0.6]],
                dtype=np.float64,
            ),
            elements=np.array([[0, 1, 2, 3]], dtype=np.int32),
        )
        snapshot = self.snapshot_mesh(mesh)

        with self.assertRaisesRegex(ValueError, "counter-clockwise"):
            _to_circle(
                mesh,
                0.0,
                0.0,
                1.0,
                [0, 1, 2, 3],
                guide_segments=[
                    [[0.1, -0.5], [np.sqrt(0.75), -0.5]],
                    [[0.5, 0.5], [0.5, np.sqrt(0.75)]],
                ],
            )

        self.assert_mesh_unchanged(mesh, snapshot)


class CircleIntegrationTests(unittest.TestCase):
    center_x = 0.0
    center_y = 0.0
    radius = 50.0
    band_width = 5.0

    @staticmethod
    def _build_mesh(dimensions=3):
        mesh = generate_rectilinear_mesh(
            5.0,
            np.arange(-70.0, 71.0, 2.0),
            np.arange(-70.0, 71.0, 3.0),
        )
        if dimensions == 2:
            mesh.nodes = mesh.nodes[:, :2].copy()
        return mesh

    @staticmethod
    def _edge_counts(mesh):
        counts = Counter()
        for element in np.asarray(mesh.elements):
            if element[2] == element[3]:
                perimeter = element[:3]
            else:
                perimeter = element
            for start, end in zip(perimeter, np.roll(perimeter, -1)):
                counts[tuple(sorted((int(start), int(end))))] += 1
        return counts

    def _circle_node_indices_and_gaps(self, mesh, radius):
        offsets = np.asarray(mesh.nodes)[:, :2] - np.array(
            [self.center_x, self.center_y]
        )
        distances = np.hypot(offsets[:, 0], offsets[:, 1])
        tolerance = 1.0e-10 * max(1.0, radius)
        indices = np.flatnonzero(np.abs(distances - radius) <= tolerance)
        self.assertGreaterEqual(indices.size, 3)

        angles = np.mod(
            np.arctan2(offsets[indices, 1], offsets[indices, 0]),
            2.0 * np.pi,
        )
        order = np.argsort(angles)
        indices = indices[order]
        angles = angles[order]
        gaps = np.diff(np.concatenate((angles, angles[:1] + 2.0 * np.pi)))
        self.assertTrue(np.all(gaps > 0.0))
        return indices, gaps

    def assert_pattern_circle_edges(self, mesh):
        pattern_nodes, gaps = self._circle_node_indices_and_gaps(
            mesh, self.radius
        )
        edge_counts = self._edge_counts(mesh)
        for start, end in zip(pattern_nodes, np.roll(pattern_nodes, -1)):
            self.assertEqual(
                edge_counts[tuple(sorted((int(start), int(end))))],
                2,
            )
        return pattern_nodes, gaps

    def annular_quad_scaled_jacobians(self, mesh):
        offsets = np.asarray(mesh.nodes)[:, :2] - np.array(
            [self.center_x, self.center_y]
        )
        radii = np.hypot(offsets[:, 0], offsets[:, 1])
        target_radii = np.array(
            [
                self.radius - self.band_width / 2.0,
                self.radius,
                self.radius + self.band_width / 2.0,
            ]
        )
        on_annular_ring = np.any(
            np.isclose(
                radii[:, None], target_radii[None, :], atol=1.0e-9
            ),
            axis=1,
        )
        annular_elements = np.all(on_annular_ring[mesh.elements], axis=1)
        quad_indices = np.flatnonzero(
            annular_elements
            & (mesh.elements[:, 2] != mesh.elements[:, 3])
        )
        if quad_indices.size == 0:
            return np.empty(0, dtype=np.float64)
        return MeshQualityChecker(mesh).calculate_scaled_jacobian(
            indices=quad_indices
        ).values

    def _node_at(self, mesh, point):
        point = np.asarray(point, dtype=np.float64)
        distances = np.hypot(
            mesh.nodes[:, 0] - point[0],
            mesh.nodes[:, 1] - point[1],
        )
        node = int(np.argmin(distances))
        self.assertLessEqual(float(distances[node]), 1.0e-9)
        return node

    def assert_mesh_unchanged(self, mesh, snapshot):
        nodes, elements, node_values, element_values = snapshot
        self.assertIs(mesh.nodes, nodes)
        self.assertIs(mesh.elements, elements)
        np.testing.assert_array_equal(mesh.nodes, node_values)
        np.testing.assert_array_equal(mesh.elements, element_values)

    def test_pattern_circle_is_an_interior_edge_loop_and_fills_the_hole(self):
        mesh = self._build_mesh()

        result = imprint_circle(
            mesh,
            center=(self.center_x, self.center_y),
            radius=self.radius,
            band_width=self.band_width,
        )

        self.assertIs(result, mesh)
        pattern_nodes, _ = self.assert_pattern_circle_edges(mesh)
        boundaries = _get_boundary(mesh)
        self.assertEqual(len(boundaries), 1)
        self.assertTrue(set(pattern_nodes).isdisjoint(set(boundaries[0])))

        report = MeshQualityChecker(mesh).check_jacobian()
        self.assertEqual(report.invalid_indices.size, 0)
        self.assertTrue(np.all(report.values > 0.0))

    def test_pattern_overlap_is_supported_through_both_annular_strips(self):
        mesh = self._build_mesh()
        pattern_x = -20.0
        guide_segments = [[[pattern_x, 70.0], [pattern_x, -70.0]]]

        imprint_circle(
            mesh,
            center=(self.center_x, self.center_y),
            radius=self.radius,
            band_width=self.band_width,
            guide_segments=guide_segments,
        )

        edge_counts = self._edge_counts(mesh)
        for sign in (-1.0, 1.0):
            nodes = []
            for circle_radius in (
                self.radius - self.band_width / 2.0,
                self.radius,
                self.radius + self.band_width / 2.0,
            ):
                intersection_y = sign * np.sqrt(circle_radius**2 - pattern_x**2)
                nodes.append(self._node_at(mesh, [pattern_x, intersection_y]))
            self.assertEqual(edge_counts[tuple(sorted(nodes[:2]))], 2)
            self.assertEqual(edge_counts[tuple(sorted(nodes[1:]))], 2)

        self.assert_pattern_circle_edges(mesh)
        self.assertEqual(len(_get_boundary(mesh)), 1)
        report = MeshQualityChecker(mesh).check_jacobian()
        self.assertEqual(report.invalid_indices.size, 0)
        self.assertTrue(np.all(report.values > 0.0))

    def test_default_and_explicit_element_sizes_bound_pattern_arc_spacing(self):
        cases = (
            ("default", {}, self.band_width),
            ("explicit", {"target_edge_size": 3.0}, 3.0),
        )

        for name, kwargs, expected_size in cases:
            with self.subTest(name=name):
                mesh = self._build_mesh()
                imprint_circle(
                    mesh,
                    center=(self.center_x, self.center_y),
                    radius=self.radius,
                    band_width=self.band_width,
                    **kwargs,
                )

                pattern_nodes, gaps = self.assert_pattern_circle_edges(mesh)
                expected_count = max(
                    3,
                    int(np.ceil(2.0 * np.pi * self.radius / expected_size)),
                )
                self.assertEqual(pattern_nodes.size, expected_count)
                self.assertLessEqual(
                    float(np.max(self.radius * gaps)),
                    expected_size + 1.0e-10,
                )

    def test_jacobian_controls_quad_merge_quality(self):
        default_mesh = self._build_mesh()
        strict_mesh = self._build_mesh()

        imprint_circle(
            default_mesh,
            center=(self.center_x, self.center_y),
            radius=self.radius,
            band_width=self.band_width,
        )
        imprint_circle(
            strict_mesh,
            center=(self.center_x, self.center_y),
            radius=self.radius,
            band_width=self.band_width,
            min_quad_scaled_jacobian=0.8,
        )

        default_quality = self.annular_quad_scaled_jacobians(default_mesh)
        strict_quality = self.annular_quad_scaled_jacobians(strict_mesh)
        self.assertGreater(default_quality.size, 0)
        self.assertGreater(strict_quality.size, 0)
        self.assertTrue(np.all(default_quality >= 0.3))
        self.assertTrue(np.any(default_quality < 0.8))
        self.assertTrue(np.all(strict_quality >= 0.8))
        self.assertFalse(
            np.array_equal(default_mesh.elements, strict_mesh.elements)
        )
        self.assertGreaterEqual(
            strict_mesh.elements.shape[0], default_mesh.elements.shape[0]
        )

    def test_invalid_inputs_leave_the_original_mesh_unchanged(self):
        cases = (
            {"center": (np.nan, self.center_y)},
            {"center": (self.center_x, np.inf)},
            {"center": (0.0,)},
            {"center": (0.0, 0.0, 0.0)},
            {"radius": np.nan},
            {"radius": self.band_width},
            {"band_width": 0.0},
            {"band_width": np.inf},
            {"target_edge_size": 0.0},
            {"target_edge_size": -1.0},
            {"target_edge_size": np.nan},
            {"target_edge_size": np.inf},
            {"min_quad_scaled_jacobian": None},
            {"min_quad_scaled_jacobian": "low"},
            {"min_quad_scaled_jacobian": -0.01},
            {"min_quad_scaled_jacobian": 1.01},
            {"min_quad_scaled_jacobian": np.nan},
            {"min_quad_scaled_jacobian": np.inf},
            {"guide_segments": [[[0.0, 0.0], [1.0, 1.0]]]},
        )

        for overrides in cases:
            with self.subTest(overrides=overrides):
                mesh = self._build_mesh()
                snapshot = (
                    mesh.nodes,
                    mesh.elements,
                    mesh.nodes.copy(),
                    mesh.elements.copy(),
                )
                arguments = {
                    "center": (self.center_x, self.center_y),
                    "radius": self.radius,
                    "band_width": self.band_width,
                }
                arguments.update(overrides)

                with self.assertRaises(ValueError):
                    imprint_circle(mesh, **arguments)

                self.assert_mesh_unchanged(mesh, snapshot)

    def test_second_strip_failure_leaves_the_original_mesh_unchanged(self):
        mesh = self._build_mesh()
        snapshot = (
            mesh.nodes,
            mesh.elements,
            mesh.nodes.copy(),
            mesh.elements.copy(),
        )

        with self.assertRaisesRegex(ValueError, "unsupported outer-circle chord"):
            imprint_circle(
                mesh,
                center=(self.center_x, self.center_y),
                radius=self.radius,
                band_width=self.band_width,
                # This reaches r+band_width but lies outside the pattern circle,
                # so only the second triangular strip sees an active chord.
                guide_segments=[[[52.0, -70.0], [52.0, 70.0]]],
            )

        self.assert_mesh_unchanged(mesh, snapshot)

    def test_pattern_node_generation_handles_tangent_duplicate_and_irrelevant_lines(self):
        center_x = 7.0
        center_y = -11.0
        radius = 5.0
        target_edge_size = 1.3
        empty_mesh = Mesh2D(
            nodes=np.empty((0, 2), dtype=np.float64),
            elements=np.empty((0, 4), dtype=np.int32),
        )

        tangent = [[12.0, -20.0], [12.0, 0.0]]
        tangent_nodes = _generate_pattern_circle_nodes(
            empty_mesh,
            center_x,
            center_y,
            radius,
            target_edge_size,
            guide_segments=[tangent],
        )
        tangent_anchor_distances = np.hypot(
            tangent_nodes[:, 0] - 12.0,
            tangent_nodes[:, 1] + 11.0,
        )
        self.assertLessEqual(float(np.min(tangent_anchor_distances)), 1.0e-12)
        tangent_angles = np.sort(
            np.mod(
                np.arctan2(
                    tangent_nodes[:, 1] - center_y,
                    tangent_nodes[:, 0] - center_x,
                ),
                2.0 * np.pi,
            )
        )
        tangent_gaps = np.diff(
            np.concatenate((tangent_angles, tangent_angles[:1] + 2.0 * np.pi))
        )
        self.assertLessEqual(
            float(np.max(radius * tangent_gaps)),
            target_edge_size + 1.0e-10,
        )

        constrained_lines = [
            [[7.0, -20.0], [7.0, 0.0]],
            [[7.0, 0.0], [7.0, -20.0]],
            [[-5.0, -11.0], [20.0, -11.0]],
            [[20.0, -20.0], [20.0, 0.0]],
        ]
        constrained_nodes = _generate_pattern_circle_nodes(
            empty_mesh,
            center_x,
            center_y,
            radius,
            target_edge_size,
            guide_segments=constrained_lines,
        )
        for anchor in (
            [12.0, -11.0],
            [7.0, -6.0],
            [2.0, -11.0],
            [7.0, -16.0],
        ):
            distances = np.hypot(
                constrained_nodes[:, 0] - anchor[0],
                constrained_nodes[:, 1] - anchor[1],
            )
            self.assertLessEqual(float(np.min(distances)), 1.0e-12)

        plain_nodes = _generate_pattern_circle_nodes(
            empty_mesh,
            center_x,
            center_y,
            radius,
            target_edge_size,
        )
        far_irrelevant_nodes = _generate_pattern_circle_nodes(
            empty_mesh,
            center_x,
            center_y,
            radius,
            target_edge_size,
            guide_segments=[[[1.0e15, -1.0e6], [1.0e15, 1.0e6]]],
        )
        np.testing.assert_array_equal(far_irrelevant_nodes, plain_nodes)

    def test_pattern_circle_tangent_is_preserved_end_to_end(self):
        mesh = generate_rectilinear_mesh(
            0.5,
            np.arange(-8.0, 8.01, 0.5),
            np.arange(-8.0, 8.01, 0.5),
        )

        imprint_circle(
            mesh,
            center=(0.0, 0.0),
            radius=5.0,
            band_width=1.0,
            target_edge_size=1.3,
            guide_segments=[[[5.0, -8.0], [5.0, 8.0]]],
        )

        pattern_tangent = self._node_at(mesh, [5.0, 0.0])
        outer_offset = np.sqrt((5.0 + 1.0 / 2.0) ** 2 - 5.0**2)
        outer_lower = self._node_at(mesh, [5.0, -outer_offset])
        outer_upper = self._node_at(mesh, [5.0, outer_offset])
        edge_counts = self._edge_counts(mesh)
        self.assertEqual(
            edge_counts[tuple(sorted((pattern_tangent, outer_lower)))],
            2,
        )
        self.assertEqual(
            edge_counts[tuple(sorted((pattern_tangent, outer_upper)))],
            2,
        )
        self.assertEqual(len(_get_boundary(mesh)), 1)
        report = MeshQualityChecker(mesh).check_jacobian()
        self.assertEqual(report.invalid_indices.size, 0)

    def test_large_element_size_is_refined_enough_to_contain_the_inner_ring(self):
        mesh = generate_rectilinear_mesh(
            0.5,
            np.arange(-8.0, 8.01, 0.5),
            np.arange(-8.0, 8.01, 0.5),
        )

        imprint_circle(
            mesh,
            center=(0.0, 0.0),
            radius=5.0,
            band_width=1.0,
            target_edge_size=200.0,
        )

        distances = np.hypot(mesh.nodes[:, 0], mesh.nodes[:, 1])
        pattern_nodes = np.flatnonzero(
            np.isclose(distances, 5.0, rtol=0.0, atol=5.0e-10)
        )
        self.assertGreaterEqual(pattern_nodes.size, 5)
        self.assertEqual(len(_get_boundary(mesh)), 1)
        report = MeshQualityChecker(mesh).check_jacobian()
        self.assertEqual(report.invalid_indices.size, 0)

    def test_supports_two_and_three_coordinate_node_arrays(self):
        for dimensions in (2, 3):
            with self.subTest(dimensions=dimensions):
                mesh = self._build_mesh(dimensions)

                imprint_circle(
                    mesh,
                    center=(self.center_x, self.center_y),
                    radius=self.radius,
                    band_width=self.band_width,
                    target_edge_size=4.0,
                )

                self.assertEqual(mesh.nodes.shape[1], dimensions)
                pattern_nodes, _ = self.assert_pattern_circle_edges(mesh)
                if dimensions == 3:
                    np.testing.assert_array_equal(
                        mesh.nodes[pattern_nodes, 2],
                        np.zeros(pattern_nodes.size),
                    )
                self.assertEqual(len(_get_boundary(mesh)), 1)
                report = MeshQualityChecker(mesh).check_jacobian()
                self.assertEqual(report.invalid_indices.size, 0)
                self.assertTrue(np.all(report.values > 0.0))


class DeleteElementTests(unittest.TestCase):
    def setUp(self):
        self.nodes = np.arange(18, dtype=np.float64).reshape(6, 3)
        self.mesh = Mesh2D(
            nodes=self.nodes,
            elements=np.array(
                [
                    [0, 1, 2, 3],
                    [1, 2, 3, 4],
                    [2, 3, 4, 5],
                    [0, 2, 4, 5],
                ],
                dtype=np.int32,
            ),
        )

    def test_removes_requested_elements_and_keeps_nodes(self):
        result = _delete_element(self.mesh, np.array([1, 3], dtype=np.int64))

        self.assertIs(result, self.mesh)
        self.assertIs(self.mesh.nodes, self.nodes)
        np.testing.assert_array_equal(
            self.mesh.elements,
            np.array([[0, 1, 2, 3], [2, 3, 4, 5]], dtype=np.int32),
        )

    def test_empty_indices_leave_all_elements(self):
        original = self.mesh.elements

        _delete_element(self.mesh, [])

        self.assertIs(self.mesh.elements, original)

    def test_rejects_out_of_range_indices_without_mutating_mesh(self):
        original = self.mesh.elements.copy()

        with self.assertRaises(IndexError):
            _delete_element(self.mesh, [4])

        np.testing.assert_array_equal(self.mesh.elements, original)

    def test_legacy_delete_name_delegates_to_delete_element(self):
        result = _delete(self.mesh, [0])

        self.assertIs(result, self.mesh)
        self.assertEqual(self.mesh.elements.shape, (3, 4))


class ClearNodeTests(unittest.TestCase):
    def test_removes_unreferenced_nodes_and_remaps_elements(self):
        nodes = np.arange(21, dtype=np.float64).reshape(7, 3)
        mesh = Mesh2D(
            nodes=nodes,
            elements=np.array(
                [[0, 2, 5, 6], [2, 3, 5, 6]],
                dtype=np.int32,
            ),
        )

        index_map = _clear_node(mesh)

        np.testing.assert_array_equal(index_map, [0, -1, 1, 2, -1, 3, 4])
        self.assertEqual(index_map.ndim, 1)
        self.assertEqual(index_map.dtype, np.dtype(np.intp))
        np.testing.assert_array_equal(index_map[[6, 2, 0, 5]], [4, 1, 0, 3])
        np.testing.assert_array_equal(mesh.nodes, nodes[[0, 2, 3, 5, 6]])
        np.testing.assert_array_equal(
            mesh.elements,
            np.array([[0, 1, 3, 4], [1, 2, 3, 4]], dtype=np.int32),
        )
        self.assertEqual(mesh.nodes.dtype, np.float64)
        self.assertEqual(mesh.elements.dtype, np.int32)

    def test_empty_elements_remove_all_nodes(self):
        mesh = Mesh2D(
            nodes=np.arange(12, dtype=np.float64).reshape(4, 3),
            elements=np.empty((0, 4), dtype=np.int64),
        )

        index_map = _clear_node(mesh)

        np.testing.assert_array_equal(index_map, [-1, -1, -1, -1])
        self.assertEqual(mesh.nodes.shape, (0, 3))
        self.assertEqual(mesh.elements.shape, (0, 4))

    def test_returns_identity_map_when_all_nodes_are_referenced(self):
        nodes = np.arange(12, dtype=np.float64).reshape(4, 3)
        elements = np.array([[0, 1, 2, 3]], dtype=np.int32)
        mesh = Mesh2D(nodes=nodes, elements=elements)

        index_map = _clear_node(mesh)

        np.testing.assert_array_equal(index_map, [0, 1, 2, 3])
        self.assertIs(mesh.nodes, nodes)
        self.assertIs(mesh.elements, elements)


if __name__ == "__main__":
    unittest.main()
