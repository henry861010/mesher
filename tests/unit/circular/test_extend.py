import unittest
from unittest.mock import patch

import numpy as np

from mesher import Mesh2D
from mesher.circular import extend_circular_mesh
from mesher.circular import extend as extend_module
from mesher.circular.topology import _get_boundary
from mesher.quality import MeshQualityChecker


def _fan_mesh(
    *,
    center=(0.0, 0.0),
    radius=1.0,
    angles=None,
    dimensions=2,
    existing_outer_radius=None,
    include_far_component=False,
    closed=True,
):
    if angles is None:
        angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    angles = np.asarray(angles, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    ring_xy = center + radius * np.column_stack(
        (np.cos(angles), np.sin(angles))
    )
    nodes_xy = np.vstack((center, ring_xy))
    if dimensions == 2:
        nodes = nodes_xy
    else:
        z = np.concatenate(([7.0], np.arange(angles.size, dtype=np.float64)))
        nodes = np.column_stack((nodes_xy, z))

    ring = np.arange(1, angles.size + 1, dtype=np.int64)
    elements = []
    ring_starts = ring if closed else ring[:-1]
    ring_ends = np.roll(ring, -1) if closed else ring[1:]
    for start, end in zip(ring_starts, ring_ends):
        elements.append([0, start, end, end])

    if existing_outer_radius is not None:
        outer_xy = center + existing_outer_radius * np.column_stack(
            (np.cos(angles), np.sin(angles))
        )
        if dimensions == 2:
            outer_nodes = outer_xy
        else:
            outer_nodes = np.column_stack(
                (outer_xy, np.full(angles.size, 99.0))
            )
        outer_start = nodes.shape[0]
        nodes = np.vstack((nodes, outer_nodes))
        outer = np.arange(
            outer_start,
            outer_start + angles.size,
            dtype=np.int64,
        )
        outer_ends = np.roll(outer, -1) if closed else outer[1:]
        for inner_start, inner_end, outer_start_node, outer_end in zip(
            ring_starts,
            ring_ends,
            outer if closed else outer[:-1],
            outer_ends,
        ):
            elements.append(
                [inner_start, outer_start_node, outer_end, inner_end]
            )

    if include_far_component:
        far_xy = np.array(
            [[20.0, 20.0], [21.0, 20.0], [20.0, 21.0]],
            dtype=np.float64,
        )
        if dimensions == 3:
            far_nodes = np.column_stack((far_xy, np.zeros(3)))
        else:
            far_nodes = far_xy
        far_start = nodes.shape[0]
        nodes = np.vstack((nodes, far_nodes))
        elements.append(
            [far_start, far_start + 1, far_start + 2, far_start + 2]
        )

    return Mesh2D(
        nodes=np.asarray(nodes),
        elements=np.asarray(elements, dtype=np.int32),
    )


class ExtendCircularMeshTests(unittest.TestCase):
    @staticmethod
    def _snapshot(mesh):
        return (
            mesh.nodes,
            mesh.elements,
            np.asarray(mesh.nodes).copy(),
            np.asarray(mesh.elements).copy(),
        )

    def assert_mesh_unchanged(self, mesh, snapshot):
        nodes, elements, node_values, element_values = snapshot
        self.assertIs(mesh.nodes, nodes)
        self.assertIs(mesh.elements, elements)
        np.testing.assert_array_equal(mesh.nodes, node_values)
        np.testing.assert_array_equal(mesh.elements, element_values)

    def test_replaces_everything_outside_inner_circle_with_equal_layers(self):
        mesh = _fan_mesh(
            existing_outer_radius=1.5,
            include_far_component=True,
        )
        retained_nodes = mesh.nodes[:9].copy()
        retained_elements = mesh.elements[:8].copy()

        result = extend_circular_mesh(
            mesh,
            element_size=0.6,
            center_x=0.0,
            center_y=0.0,
            inner_radius=1.0,
            outer_radius=2.3,
        )

        self.assertIs(result, mesh)
        layer_count = 3
        ring_size = 8
        self.assertEqual(mesh.nodes.shape, (9 + layer_count * ring_size, 2))
        self.assertEqual(
            mesh.elements.shape,
            (8 + layer_count * ring_size, 4),
        )
        np.testing.assert_array_equal(mesh.nodes[:9], retained_nodes)
        np.testing.assert_array_equal(mesh.elements[:8], retained_elements)
        self.assertTrue(np.all(mesh.elements[8:, 2] != mesh.elements[8:, 3]))

        layer_nodes = mesh.nodes[9:].reshape(layer_count, ring_size, 2)
        layer_radii = np.hypot(layer_nodes[..., 0], layer_nodes[..., 1])
        expected_radii = np.array([1.0 + 1.3 / 3.0, 1.0 + 2.6 / 3.0, 2.3])
        np.testing.assert_allclose(
            layer_radii,
            np.broadcast_to(expected_radii[:, None], layer_radii.shape),
            rtol=0.0,
            atol=1.0e-12,
        )
        self.assertTrue(np.all(np.diff(np.r_[1.0, expected_radii]) <= 0.6))
        self.assertLessEqual(
            float(np.max(np.hypot(mesh.nodes[:, 0], mesh.nodes[:, 1]))),
            2.3 + 1.0e-12,
        )

        boundaries = _get_boundary(mesh)
        self.assertEqual(len(boundaries), 1)
        outer_distances = np.hypot(
            mesh.nodes[boundaries[0], 0],
            mesh.nodes[boundaries[0], 1],
        )
        np.testing.assert_allclose(outer_distances, 2.3, atol=1.0e-12)
        report = MeshQualityChecker(mesh).check_jacobian()
        self.assertEqual(report.invalid_indices.size, 0)

    def test_thickness_smaller_than_element_size_creates_one_layer(self):
        mesh = _fan_mesh()

        extend_circular_mesh(
            mesh,
            element_size=2.0,
            center_x=0.0,
            center_y=0.0,
            inner_radius=1.0,
            outer_radius=1.25,
        )

        self.assertEqual(mesh.nodes.shape, (17, 2))
        self.assertEqual(mesh.elements.shape, (16, 4))
        outer = mesh.nodes[-8:]
        np.testing.assert_allclose(
            np.hypot(outer[:, 0], outer[:, 1]),
            1.25,
            atol=1.0e-12,
        )

    def test_off_center_irregular_3d_ring_preserves_z_by_layer(self):
        angles = np.deg2rad([0.0, 37.0, 91.0, 156.0, 213.0, 278.0, 326.0])
        mesh = _fan_mesh(
            center=(4.0, -3.0),
            radius=2.0,
            angles=angles,
            dimensions=3,
        )
        source_z = mesh.nodes[1:, 2].copy()

        extend_circular_mesh(
            mesh,
            element_size=0.8,
            center_x=4.0,
            center_y=-3.0,
            inner_radius=2.0,
            outer_radius=3.5,
        )

        new_layers = mesh.nodes[8:].reshape(2, 7, 3)
        for layer, radius in zip(new_layers, (2.75, 3.5)):
            offsets = layer[:, :2] - np.array([4.0, -3.0])
            np.testing.assert_allclose(
                np.hypot(offsets[:, 0], offsets[:, 1]),
                radius,
                atol=1.0e-12,
            )
            np.testing.assert_array_equal(layer[:, 2], source_z)

    def test_auto_extends_open_half_circle_with_equal_layers(self):
        angles = np.deg2rad([0.0, 23.0, 61.0, 104.0, 143.0, 180.0])
        mesh = _fan_mesh(angles=angles, closed=False)
        retained_nodes = mesh.nodes.copy()
        retained_elements = mesh.elements.copy()

        result = extend_circular_mesh(
            mesh,
            element_size=0.6,
            center_x=0.0,
            center_y=0.0,
            inner_radius=1.0,
            outer_radius=2.0,
        )

        self.assertIs(result, mesh)
        layer_count = 2
        arc_size = angles.size
        self.assertEqual(
            mesh.nodes.shape,
            (retained_nodes.shape[0] + layer_count * arc_size, 2),
        )
        self.assertEqual(
            mesh.elements.shape,
            (
                retained_elements.shape[0]
                + layer_count * (arc_size - 1),
                4,
            ),
        )
        np.testing.assert_array_equal(
            mesh.nodes[: retained_nodes.shape[0]],
            retained_nodes,
        )
        np.testing.assert_array_equal(
            mesh.elements[: retained_elements.shape[0]],
            retained_elements,
        )

        layers = mesh.nodes[retained_nodes.shape[0] :].reshape(
            layer_count,
            arc_size,
            2,
        )
        for layer, radius in zip(layers, (1.5, 2.0)):
            np.testing.assert_allclose(
                np.hypot(layer[:, 0], layer[:, 1]),
                radius,
                rtol=0.0,
                atol=1.0e-12,
            )
            layer_angles = np.unwrap(
                np.arctan2(layer[:, 1], layer[:, 0])
            )
            np.testing.assert_allclose(
                layer_angles[[0, -1]],
                angles[[0, -1]],
                rtol=0.0,
                atol=1.0e-12,
            )
            self.assertTrue(np.all(np.diff(layer_angles) > 0.0))

        self.assertEqual(len(_get_boundary(mesh)), 1)
        report = MeshQualityChecker(mesh).check_jacobian()
        self.assertEqual(report.invalid_indices.size, 0)

    def test_open_arc_crossing_angle_seam_preserves_3d_coordinates(self):
        center = np.array([4.0, -3.0])
        angles = np.deg2rad([150.0, 205.0, 270.0, 350.0, 430.0])
        mesh = _fan_mesh(
            center=center,
            radius=2.0,
            angles=angles,
            dimensions=3,
            closed=False,
        )
        source_z = mesh.nodes[1:, 2].copy()

        extend_circular_mesh(
            mesh,
            element_size=2.0,
            center_x=center[0],
            center_y=center[1],
            inner_radius=2.0,
            outer_radius=3.0,
            topology="open",
        )

        outer = mesh.nodes[-angles.size :]
        offsets = outer[:, :2] - center
        np.testing.assert_allclose(
            np.hypot(offsets[:, 0], offsets[:, 1]),
            3.0,
            rtol=0.0,
            atol=1.0e-12,
        )
        outer_angles = np.unwrap(
            np.arctan2(offsets[:, 1], offsets[:, 0])
        )
        np.testing.assert_allclose(
            outer_angles[[0, -1]],
            angles[[0, -1]],
            rtol=0.0,
            atol=1.0e-12,
        )
        self.assertTrue(np.all(np.diff(outer_angles) > 0.0))
        np.testing.assert_array_equal(outer[:, 2], source_z)

    def test_circular_edge_run_wraps_boundary_array_seam(self):
        loop = np.array([10, 11, 12, 13, 14], dtype=np.int64)
        node_is_on_circle = np.array(
            [True, False, False, True, True]
        )

        runs = extend_module._circular_edge_runs(
            loop,
            node_is_on_circle,
        )

        self.assertEqual(len(runs), 1)
        np.testing.assert_array_equal(runs[0], [13, 14, 10])

    def test_non_monotone_open_arc_is_rejected_atomically(self):
        mesh = _fan_mesh(
            angles=np.deg2rad([0.0, 60.0, 30.0, 90.0]),
            closed=False,
        )
        snapshot = self._snapshot(mesh)

        with self.assertRaisesRegex(ValueError, "strict angular order"):
            extend_circular_mesh(
                mesh,
                element_size=0.5,
                center_x=0.0,
                center_y=0.0,
                inner_radius=1.0,
                outer_radius=2.0,
            )

        self.assert_mesh_unchanged(mesh, snapshot)

    def test_default_auto_and_explicit_closed_modes_match(self):
        auto_mesh = _fan_mesh()
        closed_mesh = Mesh2D(
            nodes=auto_mesh.nodes.copy(),
            elements=auto_mesh.elements.copy(),
        )
        kwargs = dict(
            element_size=0.6,
            center_x=0.0,
            center_y=0.0,
            inner_radius=1.0,
            outer_radius=2.0,
        )

        extend_circular_mesh(auto_mesh, **kwargs)
        extend_circular_mesh(closed_mesh, topology="closed", **kwargs)

        np.testing.assert_array_equal(auto_mesh.nodes, closed_mesh.nodes)
        np.testing.assert_array_equal(
            auto_mesh.elements,
            closed_mesh.elements,
        )

    def test_invalid_scalars_and_missing_ring_are_atomic_errors(self):
        invalid_cases = (
            ({"element_size": 0.0}, "element_size"),
            ({"element_size": np.nan}, "finite"),
            ({"inner_radius": 0.0}, "inner_radius"),
            ({"outer_radius": 1.0}, "outer_radius"),
        )
        defaults = dict(
            element_size=0.5,
            center_x=0.0,
            center_y=0.0,
            inner_radius=1.0,
            outer_radius=2.0,
        )
        for changes, error in invalid_cases:
            with self.subTest(changes=changes):
                mesh = _fan_mesh()
                snapshot = self._snapshot(mesh)
                kwargs = {**defaults, **changes}
                with self.assertRaisesRegex(ValueError, error):
                    extend_circular_mesh(mesh, **kwargs)
                self.assert_mesh_unchanged(mesh, snapshot)

        mesh = _fan_mesh()
        snapshot = self._snapshot(mesh)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            extend_circular_mesh(
                mesh,
                element_size=0.5,
                center_x=0.0,
                center_y=0.0,
                inner_radius=0.9,
                outer_radius=2.0,
            )
        self.assert_mesh_unchanged(mesh, snapshot)

    def test_invalid_topology_is_rejected_atomically(self):
        for topology in (None, "partial", 1):
            with self.subTest(topology=topology):
                mesh = _fan_mesh()
                snapshot = self._snapshot(mesh)
                with self.assertRaisesRegex(ValueError, "topology"):
                    extend_circular_mesh(
                        mesh,
                        element_size=0.5,
                        center_x=0.0,
                        center_y=0.0,
                        inner_radius=1.0,
                        outer_radius=2.0,
                        topology=topology,
                    )
                self.assert_mesh_unchanged(mesh, snapshot)

    def test_non_mesh_input_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "Mesh2D"):
            extend_circular_mesh(
                object(),
                element_size=0.5,
                center_x=0.0,
                center_y=0.0,
                inner_radius=1.0,
                outer_radius=2.0,
            )

    def test_topology_modes_reject_mismatched_meshes_atomically(self):
        open_mesh = _fan_mesh(
            angles=np.linspace(0.0, np.pi, 6),
            closed=False,
        )
        open_snapshot = self._snapshot(open_mesh)
        with self.assertRaisesRegex(ValueError, "topology='closed'"):
            extend_circular_mesh(
                open_mesh,
                element_size=0.5,
                center_x=0.0,
                center_y=0.0,
                inner_radius=1.0,
                outer_radius=2.0,
                topology="closed",
            )
        self.assert_mesh_unchanged(open_mesh, open_snapshot)

        closed_mesh = _fan_mesh()
        closed_snapshot = self._snapshot(closed_mesh)
        with self.assertRaisesRegex(ValueError, "topology='open'"):
            extend_circular_mesh(
                closed_mesh,
                element_size=0.5,
                center_x=0.0,
                center_y=0.0,
                inner_radius=1.0,
                outer_radius=2.0,
                topology="open",
            )
        self.assert_mesh_unchanged(closed_mesh, closed_snapshot)

    def test_duplicate_inner_boundaries_are_rejected_atomically(self):
        ring = np.array(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
        )
        duplicate_mesh = Mesh2D(
            nodes=np.vstack((ring, ring)),
            elements=np.array(
                [[0, 1, 2, 3], [4, 5, 6, 7]],
                dtype=np.int32,
            ),
        )
        duplicate_snapshot = self._snapshot(duplicate_mesh)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            extend_circular_mesh(
                duplicate_mesh,
                element_size=0.5,
                center_x=0.0,
                center_y=0.0,
                inner_radius=1.0,
                outer_radius=2.0,
            )
        self.assert_mesh_unchanged(duplicate_mesh, duplicate_snapshot)

    def test_disconnected_open_arcs_are_rejected_atomically(self):
        first = _fan_mesh(
            angles=np.deg2rad([0.0, 20.0, 45.0, 70.0]),
            closed=False,
        )
        second = _fan_mesh(
            angles=np.deg2rad([180.0, 205.0, 230.0, 255.0]),
            closed=False,
        )
        second_elements = second.elements + first.nodes.shape[0]
        mesh = Mesh2D(
            nodes=np.vstack((first.nodes, second.nodes)),
            elements=np.vstack((first.elements, second_elements)),
        )
        snapshot = self._snapshot(mesh)

        with self.assertRaisesRegex(ValueError, "exactly one"):
            extend_circular_mesh(
                mesh,
                element_size=0.5,
                center_x=0.0,
                center_y=0.0,
                inner_radius=1.0,
                outer_radius=2.0,
            )

        self.assert_mesh_unchanged(mesh, snapshot)

    def test_invalid_connectivity_is_rejected_atomically(self):
        mesh = _fan_mesh()
        mesh.elements[0, 0] = mesh.nodes.shape[0]
        snapshot = self._snapshot(mesh)

        with self.assertRaisesRegex(ValueError, "out-of-range"):
            extend_circular_mesh(
                mesh,
                element_size=0.5,
                center_x=0.0,
                center_y=0.0,
                inner_radius=1.0,
                outer_radius=2.0,
            )

        self.assert_mesh_unchanged(mesh, snapshot)

    def test_failure_on_a_later_layer_leaves_original_mesh_unchanged(self):
        mesh = _fan_mesh()
        snapshot = self._snapshot(mesh)
        original_to_circle = extend_module._to_circle
        call_count = 0

        def fail_on_second_layer(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ValueError("second layer failed")
            return original_to_circle(*args, **kwargs)

        with patch.object(
            extend_module,
            "_to_circle",
            side_effect=fail_on_second_layer,
        ):
            with self.assertRaisesRegex(ValueError, "second layer"):
                extend_circular_mesh(
                    mesh,
                    element_size=0.5,
                    center_x=0.0,
                    center_y=0.0,
                    inner_radius=1.0,
                    outer_radius=2.0,
                )

        self.assertEqual(call_count, 2)
        self.assert_mesh_unchanged(mesh, snapshot)

    def test_unrepresentably_small_element_size_is_rejected_atomically(self):
        mesh = _fan_mesh()
        snapshot = self._snapshot(mesh)

        with self.assertRaisesRegex(ValueError, "too many"):
            extend_circular_mesh(
                mesh,
                element_size=np.nextafter(0.0, 1.0),
                center_x=0.0,
                center_y=0.0,
                inner_radius=1.0,
                outer_radius=2.0,
            )

        self.assert_mesh_unchanged(mesh, snapshot)


if __name__ == "__main__":
    unittest.main()
