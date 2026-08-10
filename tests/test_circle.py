import unittest

import numpy as np

from circle.circle import (
    _clear_node,
    _delete,
    _delete_element,
    _search_circle,
    _to_circle,
)
from mesh import Mesh
from mesh_quality import MeshQualityChecker


class SearchCircleTests(unittest.TestCase):
    def setUp(self):
        self.mesh = Mesh(
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
        return Mesh(nodes=nodes, elements=elements)

    @staticmethod
    def _off_center_pattern_mesh():
        return Mesh(
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
            Mesh(nodes=nodes, elements=np.asarray(elements, dtype=np.int32)),
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
        np.testing.assert_allclose(mesh.nodes[6:], expected_targets)
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

    def test_extends_a_closed_loop_with_one_quad_per_boundary_edge(self):
        mesh = Mesh(
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
        mesh = Mesh(
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
        base = Mesh(
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
        mesh = Mesh(nodes=original_nodes, elements=base.elements)

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
            lines=[[[0.5, 0.5], [0.5, 2.0]]],
        )

        np.testing.assert_allclose(mesh.nodes[7, :2], [0.5, target_y])
        connector = {4, 7}
        connector_uses = 0
        for element in mesh.elements[2:]:
            edges = zip(element, np.roll(element, -1))
            connector_uses += sum({int(a), int(b)} == connector for a, b in edges)
        self.assertEqual(connector_uses, 2)

    def test_horizontal_pattern_and_reversed_endpoints_are_supported(self):
        mesh = Mesh(
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
            lines=[[[2.0, 0.5], [0.5, 0.5]]],
        )

        np.testing.assert_allclose(mesh.nodes[7, :2], [np.sqrt(3.75), 0.5])

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
            lines=[line],
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
            lines=[line],
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
            lines=[[[0.5, -5.0], [0.5, -4.0]]],
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
                lines=[[[0.0, 0.5], [0.0, 2.0]]],
            )

        self.assert_mesh_unchanged(mesh, snapshot)

    def test_pattern_intersection_inside_closed_seam_is_atomic_error(self):
        mesh = Mesh(
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
                lines=[[[-2.0, 0.0], [-1.0, 0.0]]],
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
                lines=[[[0.5, 0.5], [0.5, 1.0]]],
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
                lines=[
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
            lines=[
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
        for lines in invalid_lines:
            with self.subTest(lines=lines):
                mesh = self._outward_mesh()
                snapshot = self.snapshot_mesh(mesh)

                with self.assertRaises(ValueError):
                    _to_circle(mesh, 0.0, 0.0, 2.0, [3, 4, 5], lines)

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
        duplicate_target_mesh = Mesh(
            nodes=np.array(
                [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]],
                dtype=np.float64,
            ),
            elements=np.array([[0, 1, 2, 3]], dtype=np.int32),
        )
        with self.assertRaisesRegex(ValueError, "angular order"):
            _to_circle(duplicate_target_mesh, 0.0, 0.0, 2.0, [0, 1])

    def test_rejects_pattern_anchors_with_reversed_order_atomically(self):
        reversal_mesh = Mesh(
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
                lines=[
                    [[0.8, 0.2], [0.8, 0.6]],
                    [[0.6, 0.35], [np.sqrt(1.0 - 0.35**2), 0.35]],
                ],
                closed=True,
            )
        self.assert_mesh_unchanged(reversal_mesh, snapshot)

    def test_rejects_folded_quad_even_when_circle_order_is_valid(self):
        mesh = Mesh(
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
                lines=[
                    [[0.1, -0.5], [np.sqrt(0.75), -0.5]],
                    [[0.5, 0.5], [0.5, np.sqrt(0.75)]],
                ],
            )

        self.assert_mesh_unchanged(mesh, snapshot)


class DeleteElementTests(unittest.TestCase):
    def setUp(self):
        self.nodes = np.arange(18, dtype=np.float64).reshape(6, 3)
        self.mesh = Mesh(
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
        mesh = Mesh(
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
        mesh = Mesh(
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
        mesh = Mesh(nodes=nodes, elements=elements)

        index_map = _clear_node(mesh)

        np.testing.assert_array_equal(index_map, [0, 1, 2, 3])
        self.assertIs(mesh.nodes, nodes)
        self.assertIs(mesh.elements, elements)


if __name__ == "__main__":
    unittest.main()
