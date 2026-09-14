import inspect
import random
import time
import unittest

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d.generators import generate_rectilinear_mesh
from mesher.mesh2d.circular.imprint_v2.utils import (
    _imprint_circle,
    _remove_redundant_element,
    _to_circle,
)
from mesher.mesh2d.circular.utils.pattern_segments import _PatternGuideSet


class FunctionSignatureTests(unittest.TestCase):
    def test_all_functions_end_with_optional_indices_parameter(self):
        functions = (
            _imprint_circle,
            _to_circle,
        )

        for function in functions:
            with self.subTest(function=function.__name__):
                parameters = inspect.signature(function).parameters.values()
                parameter = tuple(parameters)[-1]
                self.assertEqual(parameter.name, "indices")
                self.assertIsNone(parameter.default)

    def test_imprint_circle_has_the_requested_public_signature(self):
        self.assertEqual(
            tuple(inspect.signature(_imprint_circle).parameters),
            (
                "mesh",
                "center_x",
                "center_y",
                "radius",
                "tolerance",
                "indices",
            ),
        )


class ToCircleTests(unittest.TestCase):
    @staticmethod
    def _mesh(nodes, elements=None):
        nodes = np.asarray(nodes, dtype=np.float64)
        if elements is None:
            elements = np.empty((0, 4), dtype=np.int32)
        return Mesh2D(nodes=nodes, elements=elements)

    @staticmethod
    def _snapshot(mesh):
        return mesh.nodes.copy(), mesh.elements.copy()

    @staticmethod
    def _signed_areas(mesh):
        points = mesh.nodes[mesh.elements, :2]
        relative_points = points - points[:, :1]
        following_points = np.roll(relative_points, -1, axis=1)
        return 0.5 * np.sum(
            relative_points[..., 0] * following_points[..., 1]
            - relative_points[..., 1] * following_points[..., 0],
            axis=1,
        )

    def test_normalizes_all_mixed_elements_when_selection_is_empty(self):
        nodes = [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [3.0, 1.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [5.0, 0.0],
            [4.0, 1.0],
        ]
        mesh = self._mesh(
            nodes,
            [
                [0, 1, 2, 2],
                [3, 4, 5, 6],
                [7, 8, 9, 9],
            ],
        )
        original_nodes = mesh.nodes.copy()

        result = _to_circle(
            mesh,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            None,
            indices=[],
        )

        self.assertIs(result, mesh)
        np.testing.assert_array_equal(mesh.nodes, original_nodes)
        np.testing.assert_array_equal(
            mesh.elements,
            [
                [0, 2, 1, 1],
                [3, 6, 5, 4],
                [7, 8, 9, 9],
            ],
        )
        self.assertTrue(np.all(self._signed_areas(mesh) > 0.0))

    def test_normalizes_clockwise_element_when_no_node_is_a_candidate(self):
        mesh = self._mesh(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
            [[0, 1, 2, 2]],
        )

        _to_circle(mesh, 0.0, 0.0, 10.0, 0.0, 0.0, None)

        np.testing.assert_array_equal(mesh.elements, [[0, 2, 1, 1]])

    def test_normalizes_with_projected_coordinates_after_an_inversion(self):
        mesh = self._mesh(
            [[0.995, 0.0], [0.999, 0.2], [0.999, -0.2]],
            [[0, 2, 1, 1]],
        )
        self.assertGreater(self._signed_areas(mesh)[0], 0.0)

        _to_circle(mesh, 0.0, 0.0, 1.0, 0.006, 0.0, None)

        np.testing.assert_allclose(mesh.nodes[0, :2], [1.0, 0.0])
        np.testing.assert_array_equal(mesh.elements, [[0, 1, 2, 2]])
        self.assertGreater(self._signed_areas(mesh)[0], 0.0)

    def test_preserves_zero_area_elements_without_reordering(self):
        mesh = self._mesh(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [4.0, 0.0],
                [5.0, 0.0],
                [6.0, 0.0],
            ],
            [[0, 2, 1, 1], [3, 5, 4, 6]],
        )
        original_elements = mesh.elements.copy()

        _to_circle(mesh, 0.0, 0.0, 1.0, 0.0, 0.0, None, indices=[])

        np.testing.assert_array_equal(mesh.elements, original_elements)
        np.testing.assert_array_equal(self._signed_areas(mesh), [0.0, 0.0])

    def test_normalizes_orientation_when_local_translation_overflows(self):
        large = np.finfo(np.float64).max * 0.75
        mesh = self._mesh(
            [[-large, 0.0], [large, 0.0], [0.0, -large]],
            [[0, 1, 2, 2]],
        )

        _to_circle(mesh, 0.0, 0.0, 1.0, 0.0, 0.0, None, indices=[])

        np.testing.assert_array_equal(mesh.elements, [[0, 2, 1, 1]])

    def test_orientation_normalization_is_idempotent(self):
        mesh = self._mesh(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
            [[0, 1, 2, 2]],
        )
        _to_circle(mesh, 0.0, 0.0, 1.0, 0.0, 0.0, None, indices=[])
        nodes = mesh.nodes
        elements = mesh.elements

        result = _to_circle(
            mesh,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            None,
            indices=[],
        )

        self.assertIs(result, mesh)
        self.assertIs(mesh.nodes, nodes)
        self.assertIs(mesh.elements, elements)

    def test_moves_inside_and_outside_nodes_radially_and_preserves_z(self):
        mesh = self._mesh(
            [
                [0.9, 0.0, 7.0],
                [0.0, 1.1, 8.0],
                [-1.0, 0.0, 9.0],
                [2.0, 2.0, 10.0],
                [0.0, 0.0, 11.0],
            ],
            [[0, 1, 2, 3]],
        )
        result = _to_circle(
            mesh,
            0.0,
            0.0,
            1.0,
            0.1,
            0.0,
            None,
        )

        self.assertIs(result, mesh)
        np.testing.assert_allclose(
            mesh.nodes[:, :2],
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [2.0, 2.0],
                [0.0, 0.0],
            ],
            atol=1.0e-15,
        )
        np.testing.assert_array_equal(mesh.nodes[:, 2], [7, 8, 9, 10, 11])
        np.testing.assert_array_equal(mesh.elements, [[0, 3, 2, 1]])

    def test_element_indices_limit_unique_referenced_nodes(self):
        mesh = self._mesh(
            [
                [0.9, 0.0],
                [0.0, 0.9],
                [-0.9, 0.0],
                [0.0, -0.9],
                [1.1, 0.0],
                [0.0, 1.1],
                [-1.1, 0.0],
            ],
            [
                [0, 1, 2, 3],
                [3, 4, 5, 6],
            ],
        )

        _to_circle(mesh, 0.0, 0.0, 1.0, 0.11, 0.0, None, indices=[0])

        np.testing.assert_allclose(
            mesh.nodes[:, :2],
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
                [1.1, 0.0],
                [0.0, 1.1],
                [-1.1, 0.0],
            ],
        )

        snapshot = self._snapshot(mesh)
        _to_circle(mesh, 0.0, 0.0, 1.0, 1.0, 1.0, None, indices=[])
        np.testing.assert_array_equal(mesh.nodes, snapshot[0])
        np.testing.assert_array_equal(mesh.elements, snapshot[1])

    def test_vertical_horizontal_and_reversed_guides_choose_nearest_roots(self):
        root = np.sqrt(0.75)
        mesh = self._mesh(
            [
                [0.5, 0.8],
                [0.5, -0.8],
                [0.8, 0.5],
                [-0.8, 0.5],
            ]
        )
        guides = [
            [[0.5, 1.0], [0.5, -1.0]],
            [[-1.0, 0.5], [1.0, 0.5]],
        ]

        _to_circle(mesh, 0.0, 0.0, 1.0, 0.1, 0.0, guides)

        np.testing.assert_allclose(
            mesh.nodes[:, :2],
            [
                [0.5, root],
                [0.5, -root],
                [root, 0.5],
                [-root, 0.5],
            ],
        )

    def test_guide_tolerance_uses_true_distance_and_is_independent(self):
        source = np.array([[0.52, 0.79]], dtype=np.float64)
        segment = [[[0.5, 0.8], [0.5, 1.0]]]
        radial = self._mesh(source)
        guided = self._mesh(source)

        _to_circle(radial, 0.0, 0.0, 1.0, 0.1, 0.021, segment)
        _to_circle(guided, 0.0, 0.0, 1.0, 0.1, 0.023, segment)

        expected_radial = source[0] / np.linalg.norm(source[0])
        np.testing.assert_allclose(radial.nodes[0, :2], expected_radial)
        np.testing.assert_allclose(
            guided.nodes[0, :2],
            [0.5, np.sqrt(0.75)],
        )

    def test_finite_guide_that_does_not_reach_circle_blocks_the_node(self):
        mesh = self._mesh([[0.5, 0.8]])
        original = mesh.nodes.copy()

        _to_circle(
            mesh,
            0.0,
            0.0,
            1.0,
            0.1,
            0.0,
            [[[0.5, 0.75], [0.5, 0.82]]],
        )

        np.testing.assert_array_equal(mesh.nodes, original)

    def test_compatible_duplicates_move_and_conflicting_guides_do_not(self):
        compatible = self._mesh([[0.5, 0.8]])
        conflict = self._mesh([[0.5, 0.8]])
        duplicate_guides = [
            [[0.5, -1.0], [0.5, 1.0]],
            [[0.5, 1.0], [0.5, -1.0]],
        ]

        _to_circle(
            compatible,
            0.0,
            0.0,
            1.0,
            0.1,
            0.0,
            duplicate_guides,
        )
        _to_circle(
            conflict,
            0.0,
            0.0,
            1.0,
            0.1,
            0.0,
            duplicate_guides + [[[-1.0, 0.8], [1.0, 0.8]]],
        )

        np.testing.assert_allclose(
            compatible.nodes[0, :2],
            [0.5, np.sqrt(0.75)],
        )
        np.testing.assert_array_equal(conflict.nodes[0, :2], [0.5, 0.8])

    def test_crossing_guides_can_agree_at_a_circle_anchor(self):
        anchor_y = np.sqrt(0.75)
        mesh = self._mesh([[0.49, anchor_y - 0.006]])

        _to_circle(
            mesh,
            0.0,
            0.0,
            1.0,
            0.02,
            0.011,
            [
                [[0.5, 0.0], [0.5, 1.0]],
                [[0.0, anchor_y], [1.0, anchor_y]],
            ],
        )

        np.testing.assert_allclose(mesh.nodes[0, :2], [0.5, anchor_y])

    def test_ambiguous_roots_and_center_node_remain_unchanged(self):
        mesh = self._mesh([[0.5, 0.0], [0.0, 0.0]])
        original = mesh.nodes.copy()

        _to_circle(
            mesh,
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            [[[0.5, -1.0], [0.5, 1.0]]],
        )

        np.testing.assert_array_equal(mesh.nodes, original)

    def test_accepts_prepared_guides_and_is_idempotent(self):
        mesh = self._mesh([[0.5, 0.8]])
        guides = _PatternGuideSet.from_values(
            [[[0.5, -1.0], [0.5, 1.0]]],
            coordinate_scale=1.0,
        )

        _to_circle(mesh, 0.0, 0.0, 1.0, 0.1, 0.0, guides)
        once = mesh.nodes.copy()
        result = _to_circle(mesh, 0.0, 0.0, 1.0, 0.1, 0.0, guides)

        self.assertIs(result, mesh)
        np.testing.assert_array_equal(mesh.nodes, once)

    def test_clockwise_merge_uses_current_representative_and_preserves_data(self):
        angles = np.deg2rad([0.0, -2.0, -4.0, 180.0])
        mesh = self._mesh(
            np.column_stack(
                (np.cos(angles), np.sin(angles), [1.0, 2.0, 3.0, 4.0])
            ),
            [[0, 1, 2, 3]],
        )
        original_nodes = mesh.nodes.copy()
        result = _to_circle(mesh, 0.0, 0.0, 1.0, 0.05, 0.0, None)

        self.assertIs(result, mesh)
        np.testing.assert_array_equal(mesh.nodes[1, :2], mesh.nodes[0, :2])
        np.testing.assert_allclose(mesh.nodes[2, :2], original_nodes[2, :2])
        np.testing.assert_array_equal(mesh.nodes[:, 2], original_nodes[:, 2])
        np.testing.assert_array_equal(mesh.elements, [[0, 3, 2, 1]])
        self.assertEqual(mesh.nodes.shape, original_nodes.shape)
        self.assertEqual(mesh.elements.shape, (1, 4))

        once = mesh.nodes.copy()
        _to_circle(mesh, 0.0, 0.0, 1.0, 0.05, 0.0, None)
        np.testing.assert_array_equal(mesh.nodes, once)

    def test_clockwise_merge_reconciles_the_positive_x_seam(self):
        angles = np.deg2rad([-1.0, 180.0, 1.0])
        mesh = self._mesh(np.column_stack((np.cos(angles), np.sin(angles))))

        _to_circle(mesh, 0.0, 0.0, 1.0, 0.04, 0.0, None)

        np.testing.assert_array_equal(mesh.nodes[2, :2], mesh.nodes[0, :2])
        np.testing.assert_allclose(
            mesh.nodes[1, :2],
            [-1.0, 0.0],
            atol=1.0e-15,
        )

    def test_merge_distance_is_strict_and_zero_disables_merging(self):
        exact_tolerance = float(np.hypot(1.0, 1.0))
        at_tolerance = self._mesh([[1.0, 0.0], [0.0, -1.0]])

        _to_circle(
            at_tolerance,
            0.0,
            0.0,
            1.0,
            exact_tolerance,
            0.0,
            None,
        )

        self.assertFalse(
            np.array_equal(
                at_tolerance.nodes[0, :2],
                at_tolerance.nodes[1, :2],
            )
        )

        disabled = self._mesh([[1.0, 0.0], [0.0, 1.0]])
        original = disabled.nodes.copy()
        _to_circle(disabled, 0.0, 0.0, 1.0, 0.0, 0.0, None)
        np.testing.assert_array_equal(disabled.nodes, original)

    def test_guided_targets_take_precedence_and_remain_distinct(self):
        first_angle = np.deg2rad(-2.0)
        first_anchor = np.array(
            [np.cos(first_angle), np.sin(first_angle)]
        )
        promoted = self._mesh([[1.0, 0.0], first_anchor])
        first_guide = [
            [[first_anchor[0], -0.1], [first_anchor[0], 0.0]]
        ]

        _to_circle(
            promoted,
            0.0,
            0.0,
            1.0,
            0.05,
            1.0e-6,
            first_guide,
        )

        np.testing.assert_allclose(
            promoted.nodes[:, :2],
            np.tile(first_anchor, (2, 1)),
        )

        second_angle = np.deg2rad(-4.0)
        second_anchor = np.array(
            [np.cos(second_angle), np.sin(second_angle)]
        )
        distinct = self._mesh([first_anchor, second_anchor])
        guides = first_guide + [
            [[second_anchor[0], -0.1], [second_anchor[0], 0.0]]
        ]

        _to_circle(
            distinct,
            0.0,
            0.0,
            1.0,
            0.05,
            1.0e-6,
            guides,
        )

        np.testing.assert_allclose(
            distinct.nodes[:, :2],
            [first_anchor, second_anchor],
            atol=1.0e-14,
        )

    def test_element_selection_limits_circle_target_merging(self):
        angles = np.deg2rad(
            [0.0, -2.0, 90.0, 180.0, 1.0, 45.0, 135.0, 225.0]
        )
        mesh = self._mesh(
            np.column_stack((np.cos(angles), np.sin(angles))),
            [[0, 1, 2, 3], [4, 5, 6, 7]],
        )
        unselected = mesh.nodes[4:].copy()

        _to_circle(
            mesh,
            0.0,
            0.0,
            1.0,
            0.05,
            0.0,
            None,
            indices=[0],
        )

        np.testing.assert_array_equal(mesh.nodes[1, :2], mesh.nodes[0, :2])
        np.testing.assert_array_equal(mesh.nodes[4:], unselected)

    def test_seeded_rectilinear_example_merges_twenty_four_pairs(self):
        generator = random.Random(1)

        def coordinates(begin, end, minimum_step):
            values = [begin]
            current = begin
            while True:
                current += generator.uniform(minimum_step, 2.0 * minimum_step)
                if current > end:
                    return values
                values.append(current)

        mesh = generate_rectilinear_mesh(
            5.0,
            coordinates(-100.0, 100.0, 1.0),
            coordinates(-100.0, 100.0, 2.0),
        )
        _to_circle(mesh, 0.0, 0.0, 57.0, 1.0, 1.0, [])

        radii = np.hypot(mesh.nodes[:, 0], mesh.nodes[:, 1])
        circle_nodes = mesh.nodes[
            np.isclose(radii, 57.0, rtol=0.0, atol=1.0e-10),
            :2,
        ]
        self.assertEqual(circle_nodes.shape[0], 150)
        self.assertEqual(np.unique(circle_nodes, axis=0).shape[0], 126)
        self.assertTrue(np.all(self._signed_areas(mesh) >= 0.0))

    def test_invalid_inputs_raise_before_mutation(self):
        invalid_calls = (
            ((0.0, 0.0, 0.0, 0.1, 0.1, None), ValueError),
            ((0.0, 0.0, 1.0, -0.1, 0.1, None), ValueError),
            ((0.0, 0.0, 1.0, 0.1, -0.1, None), ValueError),
            ((np.nan, 0.0, 1.0, 0.1, 0.1, None), ValueError),
            (
                (
                    0.0,
                    0.0,
                    1.0,
                    0.1,
                    0.1,
                    [[[0.0, 0.0], [1.0, 1.0]]],
                ),
                ValueError,
            ),
        )
        for arguments, error_type in invalid_calls:
            with self.subTest(arguments=arguments):
                mesh = self._mesh(
                    [[0.9, 0.0], [0.0, 0.9], [-0.9, 0.0], [0.0, -0.9]],
                    [[0, 1, 2, 3]],
                )
                snapshot = self._snapshot(mesh)

                with self.assertRaises(error_type):
                    _to_circle(mesh, *arguments)

                np.testing.assert_array_equal(mesh.nodes, snapshot[0])
                np.testing.assert_array_equal(mesh.elements, snapshot[1])

        mesh = self._mesh(
            [[0.9, 0.0], [0.0, 0.9], [-0.9, 0.0], [0.0, -0.9]],
            [[0, 1, 2, 3]],
        )
        snapshot = self._snapshot(mesh)
        with self.assertRaisesRegex(ValueError, "duplicates"):
            _to_circle(mesh, 0.0, 0.0, 1.0, 0.1, 0.1, None, [0, 0])
        np.testing.assert_array_equal(mesh.nodes, snapshot[0])
        np.testing.assert_array_equal(mesh.elements, snapshot[1])

        mesh.nodes[0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "finite coordinates"):
            _to_circle(mesh, 0.0, 0.0, 1.0, 0.1, 0.1, None)


class ToCirclePerformanceTests(unittest.TestCase):
    maximum_seconds = 5.0

    def test_one_hundred_thousand_nodes_and_two_thousand_guides(self):
        count = 100_000
        angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
        radii = 1.0 + 1.0e-4 * np.sin(37.0 * angles)
        nodes = np.column_stack(
            (
                radii * np.cos(angles),
                radii * np.sin(angles),
                np.zeros(count),
            )
        )
        elements = np.tile(
            np.array([[0, 1, 2, 3]], dtype=np.int32),
            (count, 1),
        )
        fixed_values = np.linspace(-0.999, 0.999, 1000)
        vertical = np.stack(
            (
                np.column_stack((fixed_values, np.full(1000, -1.1))),
                np.column_stack((fixed_values, np.full(1000, 1.1))),
            ),
            axis=1,
        )
        horizontal = np.stack(
            (
                np.column_stack((np.full(1000, -1.1), fixed_values)),
                np.column_stack((np.full(1000, 1.1), fixed_values)),
            ),
            axis=1,
        )
        mesh = Mesh2D(nodes=nodes, elements=elements)

        started = time.perf_counter()
        result = _to_circle(
            mesh,
            0.0,
            0.0,
            1.0,
            0.001,
            1.0e-5,
            np.concatenate((vertical, horizontal)),
        )
        elapsed = time.perf_counter() - started

        self.assertIs(result, mesh)
        self.assertLess(elapsed, self.maximum_seconds)


class ImprintCircleTests(unittest.TestCase):
    @staticmethod
    def _area(mesh, element):
        count = 3 if element[2] == element[3] else 4
        points = mesh.nodes[element[:count], :2]
        following = np.roll(points, -1, axis=0)
        return 0.5 * abs(
            np.sum(
                points[:, 0] * following[:, 1]
                - points[:, 1] * following[:, 0]
            )
        )

    @staticmethod
    def _snapshot(mesh):
        return (
            mesh.nodes.copy(),
            mesh.elements.copy(),
            id(mesh.nodes),
            id(mesh.elements),
        )

    def test_splits_triangle_into_triangle_and_quad_and_interpolates_z(self):
        mesh = Mesh2D(
            nodes=[[0.0, 0.0, 0.0], [2.0, 0.0, 2.0], [0.0, 2.0, 4.0]],
            elements=[[0, 1, 2, 2]],
        )

        result = _imprint_circle(mesh, 0.0, 0.0, 1.0, 0.0)

        self.assertIs(result, mesh)
        self.assertEqual(mesh.node_count, 5)
        self.assertEqual(mesh.element_count, 2)
        np.testing.assert_allclose(
            mesh.nodes[3:],
            [[1.0, 0.0, 1.0], [0.0, 1.0, 2.0]],
            atol=1.0e-14,
        )
        self.assertEqual(np.count_nonzero(mesh.element_types == 3), 1)
        self.assertEqual(np.count_nonzero(mesh.element_types == 4), 1)
        self.assertAlmostEqual(
            sum(self._area(mesh, element) for element in mesh.elements),
            2.0,
        )

    def test_preserves_clockwise_triangle_winding(self):
        mesh = Mesh2D(
            nodes=[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
            elements=[[0, 2, 1, 1]],
        )

        _imprint_circle(mesh, 0.0, 0.0, 1.0, 0.0)

        for element in mesh.elements:
            count = 3 if element[2] == element[3] else 4
            points = mesh.nodes[element[:count], :2]
            following = np.roll(points, -1, axis=0)
            signed_area = 0.5 * np.sum(
                points[:, 0] * following[:, 1]
                - points[:, 1] * following[:, 0]
            )
            self.assertLess(signed_area, 0.0)

    def test_handles_a_large_offset_circle_without_area_cancellation(self):
        center = np.array([1.0e12, -1.0e12])
        nodes = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]) + center
        mesh = Mesh2D(nodes=nodes, elements=[[0, 1, 2, 2]])

        _imprint_circle(mesh, center[0], center[1], 1.0, 0.0)

        self.assertEqual(mesh.node_count, 5)
        self.assertEqual(mesh.element_count, 2)
        np.testing.assert_allclose(
            np.hypot(
                mesh.nodes[3:, 0] - center[0],
                mesh.nodes[3:, 1] - center[1],
            ),
            1.0,
        )

    def test_splits_opposite_edge_quad_into_two_quads(self):
        mesh = Mesh2D(
            nodes=[
                [0.8, -0.2, 0.0],
                [1.2, -0.2, 1.0],
                [1.2, 0.2, 2.0],
                [0.8, 0.2, 3.0],
            ],
            elements=[[0, 1, 2, 3]],
        )

        _imprint_circle(mesh, 0.0, 0.0, 1.0, 0.0)

        self.assertEqual(mesh.node_count, 6)
        self.assertEqual(mesh.element_count, 2)
        np.testing.assert_array_equal(mesh.element_types, [4, 4])
        self.assertAlmostEqual(
            sum(self._area(mesh, element) for element in mesh.elements),
            0.16,
        )

    def test_splits_adjacent_edge_quad_into_two_triangles_and_quad(self):
        mesh = Mesh2D(
            nodes=[[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
            elements=[[0, 1, 2, 3]],
        )

        _imprint_circle(mesh, 0.0, 0.0, 1.0, 0.0)

        self.assertEqual(mesh.node_count, 6)
        self.assertEqual(mesh.element_count, 3)
        self.assertEqual(np.count_nonzero(mesh.element_types == 3), 2)
        self.assertEqual(np.count_nonzero(mesh.element_types == 4), 1)
        self.assertAlmostEqual(
            sum(self._area(mesh, element) for element in mesh.elements),
            4.0,
        )

    def test_reuses_one_new_node_on_a_shared_selected_edge(self):
        mesh = Mesh2D(
            nodes=[
                [0.8, -0.2, 0.0],
                [1.2, -0.2, 1.0],
                [0.8, 0.0, 2.0],
                [1.2, 0.0, 3.0],
                [0.8, 0.2, 4.0],
                [1.2, 0.2, 5.0],
            ],
            elements=[[0, 1, 3, 2], [2, 3, 5, 4]],
        )

        _imprint_circle(mesh, 0.0, 0.0, 1.0, 0.0)

        self.assertEqual(mesh.node_count, 9)
        self.assertEqual(mesh.element_count, 4)
        shared_hits = np.flatnonzero(
            np.all(np.isclose(mesh.nodes[:, :2], [1.0, 0.0]), axis=1)
        )
        np.testing.assert_array_equal(shared_hits, [7])
        self.assertEqual(np.count_nonzero(mesh.elements == 7), 4)

    def test_tolerance_snaps_endpoint_and_preserves_its_z(self):
        mesh = Mesh2D(
            nodes=[
                [0.95, 0.0, 7.0],
                [2.0, 1.0, 8.0],
                [0.0, 0.0, 9.0],
            ],
            elements=[[0, 1, 2, 2]],
        )

        _imprint_circle(mesh, 0.0, 0.0, 1.0, 0.1)

        np.testing.assert_allclose(mesh.nodes[0], [1.0, 0.0, 7.0])
        self.assertEqual(mesh.node_count, 4)
        self.assertEqual(mesh.element_count, 2)

    def test_selection_leaves_unselected_connectivity_but_shared_snap_moves(self):
        mesh = Mesh2D(
            nodes=[
                [0.95, 0.0],
                [2.0, 1.0],
                [0.0, 0.0],
                [2.0, -1.0],
                [3.0, 0.0],
            ],
            elements=[[0, 1, 2, 2], [0, 3, 4, 4]],
        )
        unselected = mesh.elements[1].copy()

        _imprint_circle(mesh, 0.0, 0.0, 1.0, 0.1, indices=[0])

        np.testing.assert_array_equal(mesh.elements[1], unselected)
        np.testing.assert_allclose(mesh.nodes[0, :2], [1.0, 0.0])
        self.assertEqual(mesh.element_count, 3)

    def test_other_intersection_topologies_are_no_ops(self):
        cases = (
            (
                [[-2.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
                [[0, 1, 2, 2]],
                "two intersections on one edge",
            ),
            (
                [
                    [-2.0, -0.2],
                    [2.0, -0.2],
                    [2.0, 0.2],
                    [-2.0, 0.2],
                ],
                [[0, 1, 2, 3]],
                "four intersections",
            ),
            (
                [[1.0, -0.5], [2.0, -0.5], [2.0, 0.5], [1.0, 0.5]],
                [[0, 1, 2, 3]],
                "tangent",
            ),
            (
                [[2.0, 2.0], [3.0, 2.0], [2.0, 3.0]],
                [[0, 1, 2, 2]],
                "no intersection",
            ),
        )
        for nodes, elements, label in cases:
            with self.subTest(label=label):
                mesh = Mesh2D(nodes=nodes, elements=elements)
                snapshot = self._snapshot(mesh)

                _imprint_circle(mesh, 0.0, 0.0, 1.0, 0.0)

                np.testing.assert_array_equal(mesh.nodes, snapshot[0])
                np.testing.assert_array_equal(mesh.elements, snapshot[1])
                self.assertEqual(id(mesh.nodes), snapshot[2])
                self.assertEqual(id(mesh.elements), snapshot[3])

    def test_close_intersections_merge_to_one_contact_without_snapping(self):
        mesh = Mesh2D(
            nodes=[[0.95, 0.0], [2.0, 0.1], [2.0, -0.1]],
            elements=[[0, 1, 2, 2]],
        )
        snapshot = self._snapshot(mesh)

        _imprint_circle(mesh, 0.0, 0.0, 1.0, 0.2)

        np.testing.assert_array_equal(mesh.nodes, snapshot[0])
        np.testing.assert_array_equal(mesh.elements, snapshot[1])
        self.assertEqual(id(mesh.nodes), snapshot[2])
        self.assertEqual(id(mesh.elements), snapshot[3])

    def test_is_idempotent(self):
        mesh = Mesh2D(
            nodes=[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
            elements=[[0, 1, 2, 2]],
        )
        _imprint_circle(mesh, 0.0, 0.0, 1.0, 0.0)
        snapshot = self._snapshot(mesh)

        _imprint_circle(mesh, 0.0, 0.0, 1.0, 0.0)

        np.testing.assert_array_equal(mesh.nodes, snapshot[0])
        np.testing.assert_array_equal(mesh.elements, snapshot[1])
        self.assertEqual(id(mesh.nodes), snapshot[2])
        self.assertEqual(id(mesh.elements), snapshot[3])

    def test_invalid_inputs_raise_before_mutation(self):
        with self.assertRaisesRegex(TypeError, "Mesh2D"):
            _imprint_circle(object(), 0.0, 0.0, 1.0, 0.0)

        invalid_arguments = (
            ((np.nan, 0.0, 1.0, 0.0), ValueError),
            ((0.0, 0.0, 0.0, 0.0), ValueError),
            ((0.0, 0.0, 1.0, -1.0), ValueError),
        )
        for arguments, error_type in invalid_arguments:
            mesh = Mesh2D(
                nodes=[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
                elements=[[0, 1, 2, 2]],
            )
            snapshot = self._snapshot(mesh)
            with self.assertRaises(error_type):
                _imprint_circle(mesh, *arguments)
            np.testing.assert_array_equal(mesh.nodes, snapshot[0])
            np.testing.assert_array_equal(mesh.elements, snapshot[1])

        mesh = Mesh2D(
            nodes=[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
            elements=[[0, 1, 2, 2]],
        )
        snapshot = self._snapshot(mesh)
        with self.assertRaisesRegex(ValueError, "duplicates"):
            _imprint_circle(mesh, 0.0, 0.0, 1.0, 0.0, [0, 0])
        np.testing.assert_array_equal(mesh.nodes, snapshot[0])
        np.testing.assert_array_equal(mesh.elements, snapshot[1])

    def test_empty_mesh_and_empty_selection_are_no_ops(self):
        empty = Mesh2D(
            nodes=np.empty((0, 3)),
            elements=np.empty((0, 4), dtype=np.int32),
        )
        self.assertIs(_imprint_circle(empty, 0.0, 0.0, 1.0, 0.0), empty)

        mesh = Mesh2D(
            nodes=[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
            elements=[[0, 1, 2, 2]],
        )
        snapshot = self._snapshot(mesh)
        _imprint_circle(mesh, 0.0, 0.0, 1.0, 0.0, indices=[])
        np.testing.assert_array_equal(mesh.nodes, snapshot[0])
        np.testing.assert_array_equal(mesh.elements, snapshot[1])


class RemoveRedundantElementTests(unittest.TestCase):
    @staticmethod
    def _snapshot(mesh):
        return mesh.nodes.copy(), mesh.elements.copy()

    @staticmethod
    def _edge_use_counts(mesh):
        counts = {}
        for element in mesh.elements:
            perimeter = element[:3] if element[2] == element[3] else element
            for start, end in zip(perimeter, np.roll(perimeter, -1)):
                endpoints = (
                    tuple(mesh.nodes[int(start), :2]),
                    tuple(mesh.nodes[int(end), :2]),
                )
                key = tuple(sorted(endpoints))
                counts[key] = counts.get(key, 0) + 1
        return counts

    def test_has_the_requested_public_signature(self):
        self.assertEqual(
            tuple(inspect.signature(_remove_redundant_element).parameters),
            ("mesh", "tolerance"),
        )

    def test_repairs_concave_and_removes_unusable_elements(self):
        nodes = [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [3.0, 0.0],
            [3.1, 0.0],
            [3.0, 0.1],
            [4.0, 0.0],
            [4.0, 1.0],
            [5.0, 0.0],
            [6.0, 0.0],
            [8.0, 0.0],
            [6.5, 0.2],
            [6.0, 1.0],
            [9.0, 0.0],
            [10.0, 1.0],
            [9.0, 1.0],
            [10.0, 0.0],
        ]
        elements = [
            [0, 1, 2, 3],       # valid Quad4
            [4, 5, 6, 6],       # area below tolerance
            [7, 8, 9, 9],       # clockwise Tri3
            [10, 11, 12, 13],   # concave Quad4
            [14, 15, 16, 17],   # folded Quad4
            [0, 1, 1, 2],       # invalid repeated-node Quad4
        ]
        mesh = Mesh2D(nodes=nodes, elements=elements)

        result = _remove_redundant_element(mesh, 0.01)

        self.assertIs(result, mesh)
        np.testing.assert_array_equal(
            mesh.nodes,
            np.column_stack(
                (
                    np.asarray(nodes)[[0, 1, 2, 3, 10, 11, 12, 13]],
                    np.zeros(8),
                )
            ),
        )
        np.testing.assert_array_equal(
            mesh.elements,
            [[0, 1, 2, 3], [4, 5, 6, 6], [4, 6, 7, 7]],
        )

    def test_area_equal_to_tolerance_is_removed(self):
        mesh = Mesh2D(
            nodes=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            elements=[[0, 1, 2, 2]],
        )

        _remove_redundant_element(mesh, 0.5)

        self.assertEqual(mesh.nodes.shape, (0, 3))
        self.assertEqual(mesh.elements.shape, (0, 4))

    def test_equivalences_nodes_globally_in_xy_and_preserves_representative(self):
        nodes = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 3.0],
            [1.0, 1.0, 4.0],
            [0.0, 1.0, 0.0],
            [1.01, 0.0, 30.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [1.01, 1.0, 40.0],
        ]
        mesh = Mesh2D(
            nodes=nodes,
            elements=[[0, 1, 2, 3], [4, 5, 6, 7]],
        )

        _remove_redundant_element(mesh, 0.02)

        np.testing.assert_array_equal(
            mesh.nodes,
            np.asarray(nodes)[[0, 1, 2, 3, 5, 6]],
        )
        np.testing.assert_array_equal(
            mesh.elements,
            [[0, 1, 2, 3], [1, 4, 5, 2]],
        )

    def test_equivalence_is_transitive_and_keeps_duplicate_elements(self):
        nodes = []
        elements = []
        for offset in (0.0, 0.1, 0.2):
            start = len(nodes)
            nodes.extend(
                [
                    [offset, 0.0],
                    [2.0 + offset, 0.0],
                    [offset, 2.0],
                ]
            )
            elements.append([start, start + 1, start + 2, start + 2])
        mesh = Mesh2D(nodes=nodes, elements=elements)

        _remove_redundant_element(mesh, 0.11)

        np.testing.assert_array_equal(
            mesh.nodes,
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        )
        np.testing.assert_array_equal(
            mesh.elements,
            np.tile([0, 1, 2, 2], (3, 1)),
        )

    def test_removes_element_degenerated_by_equivalence(self):
        mesh = Mesh2D(
            nodes=[[0.0, 0.0], [0.05, 0.0], [0.0, 100.0]],
            elements=[[0, 1, 2, 2]],
        )

        _remove_redundant_element(mesh, 0.1)

        self.assertEqual(mesh.nodes.shape, (0, 3))
        self.assertEqual(mesh.elements.shape, (0, 4))

    def test_converts_quad_collapsed_by_equivalence_to_triangle(self):
        mesh = Mesh2D(
            nodes=[
                [0.0, 0.0],
                [0.05, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ],
            elements=[[0, 1, 2, 3]],
        )

        _remove_redundant_element(mesh, 0.1)

        np.testing.assert_array_equal(
            mesh.nodes,
            [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        )
        np.testing.assert_array_equal(mesh.elements, [[0, 1, 2, 2]])

    def test_repairs_seeded_circle_without_creating_internal_boundaries(self):
        generator = random.Random(1)

        def coordinates(begin, end, minimum):
            result = [begin]
            current = begin
            while True:
                current += generator.uniform(minimum, minimum * 2.0)
                if current > end:
                    return result
                result.append(current)

        mesh = generate_rectilinear_mesh(
            5.0,
            coordinates(-100.0, 100.0, 1.0),
            coordinates(-100.0, 100.0, 2.0),
        )
        original_edge_uses = self._edge_use_counts(mesh)
        original_boundary = {
            edge for edge, count in original_edge_uses.items() if count == 1
        }

        _to_circle(mesh, 0.0, 0.0, 57.0, 1.0, 1.0, [])
        _remove_redundant_element(mesh, 0.01)

        repaired_edge_uses = self._edge_use_counts(mesh)
        repaired_boundary = {
            edge for edge, count in repaired_edge_uses.items() if count == 1
        }
        self.assertEqual(repaired_boundary, original_boundary)
        self.assertLessEqual(max(repaired_edge_uses.values()), 2)
        self.assertEqual(mesh.element_count, 8977)

    def test_removes_elements_with_nonfinite_xyz_coordinates(self):
        mesh = Mesh2D(
            nodes=[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, np.nan],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
            ],
            elements=[[0, 1, 2, 2], [3, 4, 5, 5]],
        )

        _remove_redundant_element(mesh, 0.0)

        np.testing.assert_array_equal(
            mesh.nodes,
            [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 1.0, 0.0]],
        )
        np.testing.assert_array_equal(mesh.elements, [[0, 1, 2, 2]])

    def test_area_is_stable_at_large_coordinate_offset(self):
        offset = 1.0e12
        mesh = Mesh2D(
            nodes=[
                [offset, offset],
                [offset + 1.0, offset],
                [offset + 1.0, offset + 1.0],
                [offset, offset + 1.0],
            ],
            elements=[[0, 1, 2, 3]],
        )

        _remove_redundant_element(mesh, 0.1)

        self.assertEqual(mesh.element_count, 1)

    def test_removes_corrupted_out_of_bounds_element(self):
        mesh = Mesh2D(
            nodes=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            elements=[[0, 1, 2, 2]],
        )
        mesh.elements[0, 0] = 99

        _remove_redundant_element(mesh, 0.0)

        self.assertEqual(mesh.nodes.shape, (0, 3))
        self.assertEqual(mesh.elements.shape, (0, 4))

    def test_removes_orphan_nodes_from_an_empty_mesh(self):
        mesh = Mesh2D(
            nodes=[[0.0, 0.0], [1.0, 1.0]],
            elements=np.empty((0, 4), dtype=np.int32),
        )

        result = _remove_redundant_element(mesh, 0.0)

        self.assertIs(result, mesh)
        self.assertEqual(mesh.nodes.shape, (0, 3))
        self.assertEqual(mesh.elements.shape, (0, 4))

    def test_no_op_preserves_owned_arrays(self):
        mesh = Mesh2D(
            nodes=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            elements=[[0, 1, 2, 2]],
        )
        node_array = mesh.nodes
        element_array = mesh.elements

        result = _remove_redundant_element(mesh, 0.0)

        self.assertIs(result, mesh)
        self.assertIs(mesh.nodes, node_array)
        self.assertIs(mesh.elements, element_array)

    def test_invalid_inputs_raise_before_mutation(self):
        with self.assertRaisesRegex(TypeError, "Mesh2D"):
            _remove_redundant_element(object(), 0.1)

        for tolerance in (-1.0, np.nan, np.inf, True, "invalid"):
            mesh = Mesh2D(
                nodes=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                elements=[[0, 1, 2, 2]],
            )
            original = self._snapshot(mesh)
            with self.subTest(tolerance=tolerance):
                with self.assertRaisesRegex(ValueError, "tolerance"):
                    _remove_redundant_element(mesh, tolerance)
                np.testing.assert_array_equal(mesh.nodes, original[0])
                np.testing.assert_array_equal(mesh.elements, original[1])


class RemoveRedundantElementPerformanceTests(unittest.TestCase):
    maximum_seconds = 10.0

    def test_one_hundred_thousand_element_grid(self):
        nx = 1000
        ny = 100
        x, y = np.meshgrid(
            np.arange(nx + 1, dtype=np.float64),
            np.arange(ny + 1, dtype=np.float64),
        )
        nodes = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        lower_left = np.arange(ny * (nx + 1), dtype=np.int32).reshape(
            ny, nx + 1
        )[:, :-1]
        elements = np.column_stack(
            (
                lower_left.ravel(),
                lower_left.ravel() + 1,
                lower_left.ravel() + nx + 2,
                lower_left.ravel() + nx + 1,
            )
        )
        mesh = Mesh2D(nodes=nodes, elements=elements)

        started = time.perf_counter()
        _remove_redundant_element(mesh, 0.01)
        elapsed = time.perf_counter() - started

        self.assertEqual(mesh.element_count, 100_000)
        self.assertEqual(mesh.node_count, nodes.shape[0])
        self.assertLess(elapsed, self.maximum_seconds)


class ImprintCirclePerformanceTests(unittest.TestCase):
    maximum_seconds = 5.0

    def test_one_hundred_thousand_element_grid(self):
        nx = 1000
        ny = 100
        x, y = np.meshgrid(
            np.arange(nx + 1, dtype=np.float64),
            np.arange(ny + 1, dtype=np.float64),
        )
        nodes = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        lower_left = np.arange(ny * (nx + 1), dtype=np.int32).reshape(
            ny, nx + 1
        )[:, :-1]
        elements = np.column_stack(
            (
                lower_left.ravel(),
                lower_left.ravel() + 1,
                lower_left.ravel() + nx + 2,
                lower_left.ravel() + nx + 1,
            )
        )
        mesh = Mesh2D(nodes=nodes, elements=elements)

        started = time.perf_counter()
        _imprint_circle(mesh, 500.0, 50.0, 30.0, 0.0)
        elapsed = time.perf_counter() - started

        self.assertGreater(mesh.element_count, 100_000)
        self.assertLess(elapsed, self.maximum_seconds)

    def test_one_hundred_thousand_elements_all_split(self):
        count = 100_000
        cell = np.array(
            [
                [0.8, -0.2, 0.0],
                [1.2, -0.2, 1.0],
                [1.2, 0.2, 2.0],
                [0.8, 0.2, 3.0],
            ]
        )
        nodes = np.tile(cell, (count, 1))
        elements = np.arange(count * 4, dtype=np.int32).reshape(count, 4)
        mesh = Mesh2D(nodes=nodes, elements=elements)

        started = time.perf_counter()
        _imprint_circle(mesh, 0.0, 0.0, 1.0, 0.0)
        elapsed = time.perf_counter() - started

        self.assertEqual(mesh.node_count, 600_000)
        self.assertEqual(mesh.element_count, 200_000)
        self.assertLess(elapsed, self.maximum_seconds)


if __name__ == "__main__":
    unittest.main()
