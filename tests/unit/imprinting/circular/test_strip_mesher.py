import unittest
from collections import Counter

import numpy as np

from mesher.imprinting.circular.strip_mesher import _CircularStripMesher, _mesh_inner_outer_circle
from mesher import Mesh
from mesher.quality import MeshQualityChecker


def _points_on_circle(radius, angles):
    angles = np.asarray(angles, dtype=np.float64)
    return radius * np.column_stack((np.cos(angles), np.sin(angles)))


def _polygon_signed_area(points):
    points = np.asarray(points, dtype=np.float64)
    following = np.roll(points, -1, axis=0)
    return 0.5 * np.sum(
        points[:, 0] * following[:, 1]
        - following[:, 0] * points[:, 1]
    )


def _element_perimeter(element):
    element = np.asarray(element)
    return element[:3] if element[2] == element[3] else element


def _element_signed_areas(nodes, elements):
    nodes = np.asarray(nodes)
    return np.asarray(
        [
            _polygon_signed_area(nodes[_element_perimeter(element), :2])
            for element in np.asarray(elements)
        ]
    )


def _edge_counts(elements):
    counts = Counter()
    for element in np.asarray(elements):
        perimeter = _element_perimeter(element)
        for start, end in zip(perimeter, np.roll(perimeter, -1)):
            counts[tuple(sorted((int(start), int(end))))] += 1
    return counts


def _chain_edges(indices, closed):
    indices = np.asarray(indices)
    edge_count = indices.size if closed else indices.size - 1
    return {
        tuple(
            sorted(
                (
                    int(indices[position]),
                    int(indices[(position + 1) % indices.size]),
                )
            )
        )
        for position in range(edge_count)
    }


class MeshInnerOuterCircleTests(unittest.TestCase):
    @staticmethod
    def _closed_fixture(element_dtype=np.int16):
        # Match the real circle() call path: outer nodes run counter-clockwise,
        # inner nodes run clockwise, and the rings have different node counts.
        sentinel = np.array(
            [[-0.10, -0.10], [0.10, -0.10], [0.0, 0.10]],
            dtype=np.float64,
        )
        inner_points = _points_on_circle(
            1.0,
            np.linspace(0.0, -2.0 * np.pi, 5, endpoint=False),
        )
        outer_points = _points_on_circle(
            2.0,
            0.19 + np.linspace(0.0, 2.0 * np.pi, 7, endpoint=False),
        )
        nodes = np.vstack((sentinel, inner_points, outer_points))
        inner_nodes = np.arange(3, 8, dtype=np.int64)
        outer_nodes = np.arange(8, 15, dtype=np.int64)
        elements = np.array([[0, 1, 2, 2]], dtype=element_dtype)
        return (
            Mesh(nodes=nodes, elements=elements),
            inner_nodes,
            outer_nodes,
        )

    @staticmethod
    def _open_fixture():
        # The combined boundary is outer start->end, then inner end->start.
        outer_angles = np.deg2rad([-60.0, -10.0, 35.0, 80.0])
        inner_angles = np.deg2rad([80.0, 45.0, 10.0, -25.0, -60.0])
        inner_points = _points_on_circle(1.0, inner_angles)
        outer_points = _points_on_circle(2.0, outer_angles)
        nodes = np.vstack((inner_points, outer_points))
        inner_nodes = np.arange(0, inner_points.shape[0], dtype=np.int64)
        outer_nodes = np.arange(
            inner_points.shape[0],
            inner_points.shape[0] + outer_points.shape[0],
            dtype=np.int64,
        )
        mesh = Mesh(
            nodes=nodes,
            elements=np.empty((0, 4), dtype=np.int32),
        )
        return mesh, inner_nodes, outer_nodes

    @staticmethod
    def _pattern_fixture(element_dtype=np.int32):
        inner_points = _points_on_circle(
            1.0,
            np.deg2rad([0.0, -90.0, -180.0, -270.0]),
        )
        outer_points = _points_on_circle(
            2.0,
            np.deg2rad([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]),
        )
        nodes = np.vstack((inner_points, outer_points))
        inner_nodes = np.arange(0, 4, dtype=np.int64)
        outer_nodes = np.arange(4, 12, dtype=np.int64)
        mesh = Mesh(
            nodes=nodes,
            elements=np.empty((0, 4), dtype=element_dtype),
        )
        return mesh, inner_nodes, outer_nodes

    @staticmethod
    def _inner_tangent_pattern_fixture():
        inner_points = _points_on_circle(
            1.0,
            np.deg2rad([0.0, -90.0, -180.0, -270.0]),
        )
        outer_points = _points_on_circle(
            2.0,
            np.deg2rad([0.0, 60.0, 120.0, 180.0, 240.0, 300.0]),
        )
        nodes = np.vstack((inner_points, outer_points))
        inner_nodes = np.arange(0, 4, dtype=np.int64)
        outer_nodes = np.arange(4, 10, dtype=np.int64)
        mesh = Mesh(
            nodes=nodes,
            elements=np.empty((0, 4), dtype=np.int32),
        )
        return mesh, inner_nodes, outer_nodes

    @staticmethod
    def _crossing_pattern_fixture():
        first_inner_angle = np.arctan2(0.6, 0.8)
        second_inner_angle = np.arctan2(0.8, 0.6)
        outer_root = np.sqrt(2.0**2 - 0.8**2)
        first_outer_angle = np.arctan2(0.8, outer_root)
        second_outer_angle = np.arctan2(outer_root, 0.8)
        inner_points = _points_on_circle(
            1.0,
            [
                0.0,
                first_inner_angle,
                second_inner_angle,
                0.5 * np.pi,
                np.pi,
                1.5 * np.pi,
            ],
        )
        outer_points = _points_on_circle(
            2.0,
            [
                0.0,
                first_outer_angle,
                second_outer_angle,
                0.5 * np.pi,
                np.pi,
                1.5 * np.pi,
            ],
        )
        nodes = np.vstack((inner_points, outer_points))
        inner_nodes = np.arange(0, 6, dtype=np.int64)
        outer_nodes = np.arange(6, 12, dtype=np.int64)
        mesh = Mesh(
            nodes=nodes,
            elements=np.empty((0, 4), dtype=np.int32),
        )
        return mesh, inner_nodes, outer_nodes

    def assert_valid_elements(self, mesh, new_elements):
        self.assertGreater(new_elements.shape[0], 0)
        for element in new_elements:
            perimeter = _element_perimeter(element)
            self.assertEqual(
                np.unique(perimeter).size,
                perimeter.size,
            )
        signed_areas = _element_signed_areas(mesh.nodes, new_elements)
        self.assertTrue(np.all(signed_areas > 0.0))

        report = MeshQualityChecker(
            Mesh(nodes=np.asarray(mesh.nodes)[:, :2], elements=new_elements)
        ).check_jacobian()
        self.assertEqual(report.invalid_indices.size, 0)
        self.assertTrue(np.all(report.values > 0.0))

    def assert_quad_scaled_jacobian_at_least(
        self, mesh, elements, minimum
    ):
        quad_indices = np.flatnonzero(elements[:, 2] != elements[:, 3])
        self.assertGreater(quad_indices.size, 0)
        report = MeshQualityChecker(
            Mesh(nodes=np.asarray(mesh.nodes)[:, :2], elements=elements)
        ).calculate_scaled_jacobian(indices=quad_indices)
        self.assertTrue(np.all(report.values >= minimum))

    def assert_atomic_value_error(
        self,
        mesh,
        inner_nodes,
        outer_nodes,
        *,
        error_regex,
        **kwargs,
    ):
        original_nodes = mesh.nodes
        original_elements = mesh.elements
        node_values = mesh.nodes.copy()
        element_values = mesh.elements.copy()

        with self.assertRaisesRegex(ValueError, error_regex):
            _mesh_inner_outer_circle(
                mesh,
                inner_nodes,
                outer_nodes,
                **kwargs,
            )

        self.assertIs(mesh.nodes, original_nodes)
        self.assertIs(mesh.elements, original_elements)
        np.testing.assert_array_equal(mesh.nodes, node_values)
        np.testing.assert_array_equal(mesh.elements, element_values)

    def test_closed_annulus_sorts_completely_shuffled_unequal_rings(self):
        mesh, inner_nodes, outer_nodes = self._closed_fixture()
        ordered_inner_nodes = inner_nodes.copy()
        ordered_outer_nodes = outer_nodes.copy()
        # Neither input is a cyclic rotation nor a reversal of circle order.
        inner_nodes = inner_nodes[[2, 0, 4, 1, 3]]
        outer_nodes = outer_nodes[[4, 1, 6, 2, 0, 5, 3]]
        original_nodes = mesh.nodes.copy()
        original_elements = mesh.elements.copy()
        original_element_count = mesh.elements.shape[0]

        result = _mesh_inner_outer_circle(
            mesh,
            inner_nodes,
            outer_nodes,
            closed=True,
        )

        self.assertIs(result, mesh)
        np.testing.assert_array_equal(mesh.nodes, original_nodes)
        np.testing.assert_array_equal(
            mesh.elements[:original_element_count],
            original_elements,
        )
        self.assertEqual(mesh.elements.dtype, np.dtype(np.int16))

        new_elements = mesh.elements[original_element_count:]
        original_triangle_count = inner_nodes.size + outer_nodes.size
        self.assertLess(
            new_elements.shape[0], original_triangle_count
        )
        self.assertTrue(np.any(new_elements[:, 2] != new_elements[:, 3]))
        self.assert_valid_elements(mesh, new_elements)
        self.assert_quad_scaled_jacobian_at_least(
            mesh, new_elements, 0.3
        )
        self.assertTrue(
            set(np.unique(new_elements)).issubset(
                set(inner_nodes.tolist()) | set(outer_nodes.tolist())
            )
        )

        boundary_edges = _chain_edges(ordered_inner_nodes, True) | _chain_edges(
            ordered_outer_nodes, True
        )
        edge_counts = _edge_counts(new_elements)
        self.assertEqual(
            {edge for edge, count in edge_counts.items() if count == 1},
            boundary_edges,
        )
        self.assertTrue(
            all(
                count == (1 if edge in boundary_edges else 2)
                for edge, count in edge_counts.items()
            )
        )

        element_area = np.sum(_element_signed_areas(mesh.nodes, new_elements))
        expected_area = abs(
            _polygon_signed_area(mesh.nodes[ordered_outer_nodes, :2])
        ) - abs(
            _polygon_signed_area(mesh.nodes[ordered_inner_nodes, :2])
        )
        self.assertAlmostEqual(element_area, expected_area, places=12)

    def test_closed_annulus_connects_angularly_nearby_nodes(self):
        # Two almost coincident boundary samples make the worst triangle
        # quality unavoidable.  A quality-only path score then has a large
        # tie plateau and can build a fan to angularly distant nodes.  The
        # connector objective must remain local even in that situation.
        inner_angles = np.deg2rad(
            [
                0.0,
                11.9600653,
                31.2597791,
                45.3320865,
                55.7621526,
                55.7632917,
                75.0217345,
                93.2229512,
                107.2561401,
                125.0991854,
                144.2167524,
                157.1339421,
                170.48104,
                189.4988172,
                207.0188829,
                220.6673213,
                235.8209563,
                251.0153793,
                263.7646428,
                277.302341,
                294.0263145,
                312.6962567,
                330.0856314,
                342.3749601,
            ]
        )
        outer_angles = np.deg2rad(
            [
                2.995327,
                18.816417,
                33.2808791,
                47.059638,
                63.6634727,
                76.9420354,
                87.9303608,
                100.6476494,
                112.7389054,
                126.6636618,
                140.2893247,
                152.1314883,
                165.5542033,
                175.7929724,
                192.0336435,
                204.3543776,
                220.0408482,
                234.9855166,
                249.6209874,
                265.7492046,
                277.9637563,
                292.7136709,
                303.1619617,
                303.1634555,
                312.8213537,
                323.0116091,
                339.5025626,
                348.7238431,
            ]
        )
        inner_points = _points_on_circle(9.0, inner_angles)
        outer_points = _points_on_circle(10.0, outer_angles)
        inner_nodes = np.arange(inner_angles.size, dtype=np.int64)
        outer_nodes = np.arange(
            inner_angles.size,
            inner_angles.size + outer_angles.size,
            dtype=np.int64,
        )
        mesh = Mesh(
            nodes=np.vstack((inner_points, outer_points)),
            elements=np.empty((0, 4), dtype=np.int32),
        )

        _mesh_inner_outer_circle(mesh, inner_nodes, outer_nodes, closed=True)

        inner_set = set(map(int, inner_nodes))
        outer_set = set(map(int, outer_nodes))
        connector_edges = set()
        connector_neighbours = {}
        for element in mesh.elements:
            perimeter = [
                int(value) for value in _element_perimeter(element)
            ]
            for first, second in zip(
                perimeter, perimeter[1:] + perimeter[:1]
            ):
                if not (
                    (first in inner_set and second in outer_set)
                    or (first in outer_set and second in inner_set)
                ):
                    continue
                edge = tuple(sorted((first, second)))
                connector_edges.add(edge)
                connector_neighbours.setdefault(first, set()).add(second)
                connector_neighbours.setdefault(second, set()).add(first)

        node_angles = np.arctan2(mesh.nodes[:, 1], mesh.nodes[:, 0])
        connector_angle_differences = np.asarray(
            [
                abs(
                    np.arctan2(
                        np.sin(node_angles[first] - node_angles[second]),
                        np.cos(node_angles[first] - node_angles[second]),
                    )
                )
                for first, second in connector_edges
            ]
        )
        self.assertLessEqual(
            float(np.max(connector_angle_differences)),
            np.deg2rad(16.0),
        )
        self.assertLessEqual(
            max(len(neighbours) for neighbours in connector_neighbours.values()),
            3,
        )

    def test_open_chains_form_a_strip_without_closing_either_ring(self):
        mesh, inner_nodes, outer_nodes = self._open_fixture()
        ordered_inner_nodes = inner_nodes.copy()
        ordered_outer_nodes = outer_nodes.copy()
        # Preserve both endpoints while fully scrambling each arc's interior.
        inner_nodes = inner_nodes[[0, 3, 1, 2, 4]]
        outer_nodes = outer_nodes[[0, 2, 1, 3]]
        original_nodes = mesh.nodes.copy()

        _mesh_inner_outer_circle(
            mesh,
            inner_nodes,
            outer_nodes,
            closed=False,
        )

        np.testing.assert_array_equal(mesh.nodes, original_nodes)
        new_elements = mesh.elements
        original_triangle_count = inner_nodes.size + outer_nodes.size - 2
        self.assertLess(
            new_elements.shape[0], original_triangle_count
        )
        self.assertTrue(np.any(new_elements[:, 2] != new_elements[:, 3]))
        self.assert_valid_elements(mesh, new_elements)
        self.assert_quad_scaled_jacobian_at_least(
            mesh, new_elements, 0.3
        )

        expected_boundary = _chain_edges(
            ordered_inner_nodes, False
        ) | _chain_edges(
            ordered_outer_nodes, False
        )
        expected_boundary.update(
            {
                tuple(sorted((int(outer_nodes[-1]), int(inner_nodes[0])))),
                tuple(sorted((int(inner_nodes[-1]), int(outer_nodes[0])))),
            }
        )
        edge_counts = _edge_counts(new_elements)
        self.assertEqual(
            {edge for edge, count in edge_counts.items() if count == 1},
            expected_boundary,
        )
        self.assertTrue(
            all(
                count == (1 if edge in expected_boundary else 2)
                for edge, count in edge_counts.items()
            )
        )

        strip_polygon = np.vstack(
            (
                mesh.nodes[ordered_outer_nodes, :2],
                mesh.nodes[ordered_inner_nodes, :2],
            )
        )
        element_area = np.sum(_element_signed_areas(mesh.nodes, new_elements))
        self.assertAlmostEqual(
            element_area,
            abs(_polygon_signed_area(strip_polygon)),
            places=12,
        )

    def test_nonconvex_quad_candidates_remain_two_triangles(self):
        triangles = np.array(
            [[0, 1, 2, 2], [1, 0, 3, 3]],
            dtype=np.int64,
        )
        cases = {
            "concave": [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [2.0, -1.0],
            ],
            "collinear corner": [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [2.0, -1.0],
            ],
        }

        for name, coordinates in cases.items():
            with self.subTest(name=name):
                nodes = np.asarray(coordinates, dtype=np.float64)
                mesher = _CircularStripMesher(
                    Mesh(
                        nodes=nodes,
                        elements=np.empty((0, 4), dtype=np.int32),
                    ),
                    [],
                    [],
                    None,
                    True,
                )
                mesher.xy = nodes
                mesher.geometry_tolerance = 1.0e-12
                mesher.angular_tolerance = 1.0e-12
                mesher.minimum_quad_scaled_jacobian = 0.3
                mesher.forced_connectors = []

                merged = mesher._merge_triangle_pairs(triangles)

                np.testing.assert_array_equal(merged, triangles)

    def test_jacobian_threshold_rejects_a_low_quality_quad(self):
        nodes = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.2, -0.3],
            ],
            dtype=np.float64,
        )
        triangles = np.array(
            [[0, 1, 2, 2], [1, 0, 3, 3]],
            dtype=np.int64,
        )
        mesher = _CircularStripMesher(
            Mesh(nodes=nodes, elements=np.empty((0, 4), dtype=np.int32)),
            [],
            [],
            None,
            True,
        )
        mesher.xy = nodes
        mesher.geometry_tolerance = 1.0e-12
        mesher.angular_tolerance = 1.0e-12
        mesher.forced_connectors = []

        mesher.minimum_quad_scaled_jacobian = 0.3
        rejected = mesher._merge_triangle_pairs(triangles)
        mesher.minimum_quad_scaled_jacobian = 0.1
        accepted = mesher._merge_triangle_pairs(triangles)

        np.testing.assert_array_equal(rejected, triangles)
        self.assertEqual(accepted.shape, (1, 4))
        self.assertNotEqual(accepted[0, 2], accepted[0, 3])
        quality = MeshQualityChecker(
            Mesh(nodes=nodes, elements=accepted)
        ).calculate_scaled_jacobian()
        self.assertGreaterEqual(float(quality.values[0]), 0.1)
        self.assertLess(float(quality.values[0]), 0.3)

    def test_open_strip_ignores_pattern_wholly_outside_selected_arcs(self):
        baseline_mesh, baseline_inner, baseline_outer = self._open_fixture()
        patterned_mesh, patterned_inner, patterned_outer = self._open_fixture()

        _mesh_inner_outer_circle(
            baseline_mesh,
            baseline_inner,
            baseline_outer,
            closed=False,
        )
        result = _mesh_inner_outer_circle(
            patterned_mesh,
            patterned_inner,
            patterned_outer,
            guide_segments=[[[-1.5, -3.0], [-1.5, 3.0]]],
            closed=False,
        )

        self.assertIs(result, patterned_mesh)
        self.assert_valid_elements(patterned_mesh, patterned_mesh.elements)
        np.testing.assert_array_equal(
            patterned_mesh.elements,
            baseline_mesh.elements,
        )

    def test_open_strip_keeps_only_the_active_branch_of_a_full_line(self):
        angles = np.deg2rad([-60.0, 0.0, 60.0])
        inner_points = _points_on_circle(1.0, angles)
        outer_points = _points_on_circle(2.0, angles)
        mesh = Mesh(
            nodes=np.vstack((inner_points, outer_points)),
            elements=np.empty((0, 4), dtype=np.int32),
        )

        _mesh_inner_outer_circle(
            mesh,
            np.arange(0, 3, dtype=np.int64),
            np.arange(3, 6, dtype=np.int64),
            guide_segments=[[[-3.0, 0.0], [3.0, 0.0]]],
            closed=False,
        )

        self.assert_valid_elements(mesh, mesh.elements)
        # The negative-x branch lies outside this right-hand strip.  The
        # positive-x branch is represented by the two zero-degree nodes.
        self.assertEqual(_edge_counts(mesh.elements)[(1, 4)], 2)

    def test_open_pattern_entering_through_a_side_is_an_atomic_error(self):
        angles = np.deg2rad([-60.0, 0.0, 60.0])
        mesh = Mesh(
            nodes=np.vstack(
                (_points_on_circle(1.0, angles), _points_on_circle(2.0, angles))
            ),
            elements=np.empty((0, 4), dtype=np.int32),
        )

        self.assert_atomic_value_error(
            mesh,
            np.arange(0, 3, dtype=np.int64),
            np.arange(3, 6, dtype=np.int64),
            error_regex="unsupported outer-circle chord",
            guide_segments=[[[-3.0, 1.2], [3.0, 1.2]]],
            closed=False,
        )

    def test_vertical_and_horizontal_patterns_become_connector_edges(self):
        cases = (
            (
                [[[0.0, -3.0], [0.0, 3.0]]],
                ((3, 6), (1, 10)),
            ),
            (
                [[[3.0, 0.0], [-3.0, 0.0]]],
                ((0, 4), (2, 8)),
            ),
        )

        for guide_segments, connectors in cases:
            with self.subTest(guide_segments=guide_segments):
                mesh, inner_nodes, outer_nodes = self._pattern_fixture()

                _mesh_inner_outer_circle(
                    mesh,
                    inner_nodes,
                    outer_nodes,
                    guide_segments=guide_segments,
                    closed=True,
                )

                self.assert_valid_elements(mesh, mesh.elements)
                edge_counts = _edge_counts(mesh.elements)
                for connector in connectors:
                    self.assertEqual(
                        edge_counts[tuple(sorted(connector))],
                        2,
                    )

    def test_duplicate_reversed_and_irrelevant_patterns_are_deterministic(self):
        baseline_mesh, baseline_inner, baseline_outer = self._pattern_fixture()
        patterned_mesh, patterned_inner, patterned_outer = self._pattern_fixture()

        _mesh_inner_outer_circle(
            baseline_mesh,
            baseline_inner,
            baseline_outer,
            guide_segments=[[[0.0, -3.0], [0.0, 3.0]]],
            closed=True,
        )
        _mesh_inner_outer_circle(
            patterned_mesh,
            patterned_inner,
            patterned_outer,
            guide_segments=[
                [[0.0, 1.0e14], [0.0, -1.0e14]],
                [[0.0, -3.0], [0.0, 3.0]],
                [[4.0, -1.0], [4.0, 1.0]],
            ],
            closed=True,
        )

        np.testing.assert_array_equal(
            patterned_mesh.elements,
            baseline_mesh.elements,
        )

    def test_inner_tangent_pattern_creates_two_connector_edges(self):
        mesh, inner_nodes, outer_nodes = self._inner_tangent_pattern_fixture()

        _mesh_inner_outer_circle(
            mesh,
            inner_nodes,
            outer_nodes,
            guide_segments=[[[1.0, -2.0], [1.0, 2.0]]],
            closed=True,
        )

        self.assert_valid_elements(mesh, mesh.elements)
        edge_counts = _edge_counts(mesh.elements)
        # The vertical line is tangent to the inner circle at node 0 and meets
        # the outer circle at the +60 and -60 degree nodes.
        self.assertEqual(edge_counts[(0, 5)], 2)
        self.assertEqual(edge_counts[(0, 9)], 2)

    def test_partial_annulus_pattern_segment_is_an_atomic_error(self):
        mesh, inner_nodes, outer_nodes = self._pattern_fixture()

        self.assert_atomic_value_error(
            mesh,
            inner_nodes,
            outer_nodes,
            error_regex="ends inside the annular material",
            guide_segments=[[[0.0, 1.5], [0.0, 3.0]]],
            closed=True,
        )

    def test_outer_only_pattern_chord_is_an_atomic_error(self):
        mesh, inner_nodes, outer_nodes = self._pattern_fixture()

        self.assert_atomic_value_error(
            mesh,
            inner_nodes,
            outer_nodes,
            error_regex="unsupported outer-circle chord",
            guide_segments=[[[1.5, -3.0], [1.5, 3.0]]],
            closed=True,
        )

    def test_pattern_missing_an_inner_ring_anchor_is_an_atomic_error(self):
        mesh, inner_nodes, outer_nodes = self._pattern_fixture()
        inner_without_upper_root = inner_nodes[:-1]

        self.assert_atomic_value_error(
            mesh,
            inner_without_upper_root,
            outer_nodes,
            error_regex="reaches inner circle without a node",
            guide_segments=[[[0.0, -3.0], [0.0, 3.0]]],
            closed=True,
        )

    def test_crossing_pattern_constraints_are_an_atomic_error(self):
        mesh, inner_nodes, outer_nodes = self._crossing_pattern_fixture()

        self.assert_atomic_value_error(
            mesh,
            inner_nodes,
            outer_nodes,
            error_regex="pattern connectors cross or overlap",
            guide_segments=[
                [[0.8, 0.5], [0.8, 2.0]],
                [[0.5, 0.8], [2.0, 0.8]],
            ],
            closed=True,
        )

    def test_invalid_pattern_is_an_atomic_error(self):
        invalid_lines = (
            [[[0.0, 0.0], [1.0, 1.0]]],
            [[[0.0, 0.0], [0.0, 0.0]]],
            [[[0.0, 0.0], [0.0, np.nan]]],
        )

        for guide_segments in invalid_lines:
            with self.subTest(guide_segments=guide_segments):
                mesh, inner_nodes, outer_nodes = self._pattern_fixture()
                original_nodes = mesh.nodes
                original_elements = mesh.elements
                node_values = mesh.nodes.copy()
                element_values = mesh.elements.copy()

                with self.assertRaises(ValueError):
                    _mesh_inner_outer_circle(
                        mesh,
                        inner_nodes,
                        outer_nodes,
                        guide_segments=guide_segments,
                        closed=True,
                    )

                self.assertIs(mesh.nodes, original_nodes)
                self.assertIs(mesh.elements, original_elements)
                np.testing.assert_array_equal(mesh.nodes, node_values)
                np.testing.assert_array_equal(mesh.elements, element_values)

    def test_malformed_empty_pattern_shape_is_an_atomic_error(self):
        mesh, inner_nodes, outer_nodes = self._pattern_fixture()

        self.assert_atomic_value_error(
            mesh,
            inner_nodes,
            outer_nodes,
            error_regex="guide_segments must have shape",
            guide_segments=np.empty((1, 0, 2), dtype=np.float64),
            closed=True,
        )

    def test_out_of_range_ring_index_is_an_atomic_error(self):
        mesh, inner_nodes, outer_nodes = self._pattern_fixture()
        original_nodes = mesh.nodes
        original_elements = mesh.elements
        node_values = mesh.nodes.copy()
        element_values = mesh.elements.copy()
        invalid_outer = outer_nodes.copy()
        invalid_outer[-1] = mesh.nodes.shape[0]

        with self.assertRaises(IndexError):
            _mesh_inner_outer_circle(
                mesh,
                inner_nodes,
                invalid_outer,
                closed=True,
            )

        self.assertIs(mesh.nodes, original_nodes)
        self.assertIs(mesh.elements, original_elements)
        np.testing.assert_array_equal(mesh.nodes, node_values)
        np.testing.assert_array_equal(mesh.elements, element_values)

    def test_closed_none_and_non_boolean_values_raise_type_error_atomically(self):
        for closed in (None, 1, "closed"):
            with self.subTest(closed=closed):
                mesh, inner_nodes, outer_nodes = self._pattern_fixture()
                original_nodes = mesh.nodes
                original_elements = mesh.elements
                node_values = mesh.nodes.copy()
                element_values = mesh.elements.copy()

                with self.assertRaisesRegex(TypeError, "closed must be"):
                    _mesh_inner_outer_circle(
                        mesh,
                        inner_nodes,
                        outer_nodes,
                        closed=closed,
                    )

                self.assertIs(mesh.nodes, original_nodes)
                self.assertIs(mesh.elements, original_elements)
                np.testing.assert_array_equal(mesh.nodes, node_values)
                np.testing.assert_array_equal(mesh.elements, element_values)

    def test_invalid_jacobian_is_an_atomic_error(self):
        for min_quad_scaled_jacobian in (None, "low", np.nan, np.inf, -0.01, 1.01):
            with self.subTest(min_quad_scaled_jacobian=min_quad_scaled_jacobian):
                mesh, inner_nodes, outer_nodes = self._pattern_fixture()
                self.assert_atomic_value_error(
                    mesh,
                    inner_nodes,
                    outer_nodes,
                    error_regex="min_quad_scaled_jacobian",
                    min_quad_scaled_jacobian=min_quad_scaled_jacobian,
                    closed=True,
                )

    def test_non_concentric_rings_are_an_atomic_fit_error(self):
        mesh, inner_nodes, outer_nodes = self._closed_fixture()
        mesh.nodes[outer_nodes, 0] += 0.25

        self.assert_atomic_value_error(
            mesh,
            inner_nodes,
            outer_nodes,
            error_regex="concentric circles",
            closed=True,
        )

    def test_small_scale_concentric_rings_are_meshed(self):
        inner_points = _points_on_circle(
            1.0e-7,
            np.linspace(0.0, 2.0 * np.pi, 5, endpoint=False),
        )
        outer_points = _points_on_circle(
            2.0e-7,
            0.17 + np.linspace(0.0, 2.0 * np.pi, 7, endpoint=False),
        )
        nodes = np.vstack((inner_points, outer_points))
        inner_nodes = np.arange(0, 5, dtype=np.int64)[[3, 0, 4, 1, 2]]
        outer_nodes = np.arange(5, 12, dtype=np.int64)[[4, 1, 6, 2, 0, 5, 3]]
        mesh = Mesh(
            nodes=nodes,
            elements=np.empty((0, 4), dtype=np.int32),
        )
        original_nodes = mesh.nodes
        original_values = mesh.nodes.copy()

        result = _mesh_inner_outer_circle(
            mesh,
            inner_nodes,
            outer_nodes,
            closed=True,
        )

        self.assertIs(result, mesh)
        self.assertIs(mesh.nodes, original_nodes)
        np.testing.assert_array_equal(mesh.nodes, original_values)
        self.assertLess(mesh.elements.shape[0], 12)
        self.assertTrue(np.any(mesh.elements[:, 2] != mesh.elements[:, 3]))
        self.assert_valid_elements(mesh, mesh.elements)

    def test_existing_triangle_inside_annulus_is_an_atomic_error(self):
        inner_points = _points_on_circle(
            1.0,
            np.deg2rad([0.0, 90.0, 180.0, 270.0]),
        )
        outer_points = _points_on_circle(
            2.0,
            np.deg2rad([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]),
        )
        # These nodes form a small, independent CCW triangle strictly inside
        # the annular material and strictly inside one generated triangle.
        embedded_triangle = np.array(
            [[1.55, 0.12], [1.70, 0.12], [1.62, 0.25]],
            dtype=np.float64,
        )
        nodes = np.vstack((inner_points, outer_points, embedded_triangle))
        mesh = Mesh(
            nodes=nodes,
            elements=np.array([[12, 13, 14, 14]], dtype=np.int32),
        )

        self.assert_atomic_value_error(
            mesh,
            np.arange(0, 4, dtype=np.int64),
            np.arange(4, 12, dtype=np.int64),
            error_regex="existing",
            closed=True,
        )

    def test_existing_edge_crossing_generated_mesh_is_an_atomic_error(self):
        angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
        ring_nodes = np.vstack(
            (_points_on_circle(1.0, angles), _points_on_circle(2.0, angles))
        )
        crossing_triangle = np.array(
            [[1.5, -0.2], [1.7, 0.0], [1.5, 0.2]],
            dtype=np.float64,
        )
        mesh = Mesh(
            nodes=np.vstack((ring_nodes, crossing_triangle)),
            elements=np.array([[16, 17, 18, 18]], dtype=np.int32),
        )

        self.assert_atomic_value_error(
            mesh,
            np.arange(0, 8, dtype=np.int64),
            np.arange(8, 16, dtype=np.int64),
            error_regex="existing mesh geometry",
            closed=True,
        )

    def test_existing_edge_overlapping_generated_mesh_is_an_atomic_error(self):
        angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
        ring_nodes = np.vstack(
            (_points_on_circle(1.0, angles), _points_on_circle(2.0, angles))
        )
        overlapping_triangle = np.array(
            [[1.2, 0.0], [1.8, 0.0], [1.5, 0.1]],
            dtype=np.float64,
        )
        mesh = Mesh(
            nodes=np.vstack((ring_nodes, overlapping_triangle)),
            elements=np.array([[16, 17, 18, 18]], dtype=np.int32),
        )

        self.assert_atomic_value_error(
            mesh,
            np.arange(0, 8, dtype=np.int64),
            np.arange(8, 16, dtype=np.int64),
            error_regex="existing mesh geometry",
            closed=True,
        )

    def test_existing_exposed_ring_edge_is_paired_in_the_opposite_direction(self):
        inner_points = _points_on_circle(
            1.0, np.deg2rad([0.0, 90.0, 180.0, 270.0])
        )
        outer_points = _points_on_circle(
            2.0, np.deg2rad([0.0, 90.0, 180.0, 270.0])
        )
        nodes = np.vstack((inner_points, outer_points, [[3.0, 3.0]]))
        inner_nodes = np.arange(0, 4, dtype=np.int64)
        outer_nodes = np.arange(4, 8, dtype=np.int64)
        # New annulus triangles traverse 4 -> 5, so the existing outside
        # triangle exposes the same interface as 5 -> 4.
        mesh = Mesh(
            nodes=nodes,
            elements=np.array([[5, 4, 8, 8]], dtype=np.int32),
        )

        _mesh_inner_outer_circle(mesh, inner_nodes, outer_nodes, closed=True)

        directed_uses = []
        for element in mesh.elements:
            perimeter = _element_perimeter(element)
            for start, end in zip(perimeter, np.roll(perimeter, -1)):
                if {int(start), int(end)} == {4, 5}:
                    directed_uses.append((int(start), int(end)))
        self.assertCountEqual(directed_uses, [(5, 4), (4, 5)])

    def test_three_dimensional_nodes_are_not_replaced_or_modified(self):
        mesh, inner_nodes, outer_nodes = self._pattern_fixture()
        z_values = np.linspace(-3.0, 4.0, mesh.nodes.shape[0])
        mesh.nodes = np.column_stack((mesh.nodes, z_values))
        original_nodes = mesh.nodes
        original_values = mesh.nodes.copy()

        _mesh_inner_outer_circle(mesh, inner_nodes, outer_nodes, closed=True)

        self.assertIs(mesh.nodes, original_nodes)
        np.testing.assert_array_equal(mesh.nodes, original_values)
        self.assert_valid_elements(mesh, mesh.elements)

    def test_incompatible_existing_ring_edge_is_an_atomic_error(self):
        inner_points = _points_on_circle(
            1.0, np.deg2rad([0.0, 90.0, 180.0, 270.0])
        )
        outer_points = _points_on_circle(
            2.0, np.deg2rad([0.0, 90.0, 180.0, 270.0])
        )
        nodes = np.vstack((inner_points, outer_points, [[3.0, -2.0]]))
        mesh = Mesh(
            nodes=nodes,
            elements=np.array([[4, 5, 8, 8]], dtype=np.int32),
        )

        self.assert_atomic_value_error(
            mesh,
            np.arange(0, 4, dtype=np.int64),
            np.arange(4, 8, dtype=np.int64),
            error_regex="existing",
            closed=True,
        )

    def test_clockwise_existing_element_is_an_atomic_error(self):
        mesh, inner_nodes, outer_nodes = self._closed_fixture()
        mesh.elements = np.array([[0, 2, 1, 1]], dtype=np.int32)

        self.assert_atomic_value_error(
            mesh,
            inner_nodes,
            outer_nodes,
            error_regex="existing Tri3.*counter-clockwise",
            closed=True,
        )

    def test_existing_non_manifold_edge_is_an_atomic_error(self):
        mesh, inner_nodes, outer_nodes = self._closed_fixture()
        mesh.nodes = np.vstack(
            (
                mesh.nodes,
                [[-0.3, -0.3], [0.3, -0.3], [0.0, 0.35]],
            )
        )
        mesh.elements = np.array(
            [
                [0, 1, 15, 15],
                [1, 0, 16, 16],
                [0, 1, 17, 17],
            ],
            dtype=np.int32,
        )

        self.assert_atomic_value_error(
            mesh,
            inner_nodes,
            outer_nodes,
            error_regex="non-manifold edge",
            closed=True,
        )

    def test_promotes_element_dtype_when_ring_indices_do_not_fit(self):
        nodes = np.zeros((132, 2), dtype=np.float64)
        inner_nodes = np.arange(124, 128, dtype=np.int64)
        outer_nodes = np.arange(128, 132, dtype=np.int64)
        nodes[inner_nodes] = _points_on_circle(
            1.0,
            np.deg2rad([0.0, -90.0, -180.0, -270.0]),
        )
        nodes[outer_nodes] = _points_on_circle(
            2.0,
            np.deg2rad([0.0, 90.0, 180.0, 270.0]),
        )
        mesh = Mesh(
            nodes=nodes,
            elements=np.empty((0, 4), dtype=np.int8),
        )

        _mesh_inner_outer_circle(
            mesh,
            inner_nodes,
            outer_nodes,
            closed=True,
        )

        self.assertEqual(mesh.elements.dtype, np.dtype(np.int64))
        self.assert_valid_elements(mesh, mesh.elements)
        self.assertGreater(int(np.max(mesh.elements)), np.iinfo(np.int8).max)


if __name__ == "__main__":
    unittest.main()
