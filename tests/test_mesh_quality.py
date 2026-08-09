import math
import unittest
from unittest.mock import patch

import numpy as np

from checkerboard import checkerboard_box
from mesh import Mesh
from mesh_quality import (
    ElementType,
    GeometryTolerance,
    MeshQualityChecker,
    QualityReason,
    QualityReport,
    QualityResult,
)


def _checker(nodes, elements, *, tolerance=None):
    kwargs = {} if tolerance is None else {"tolerance": tolerance}
    return MeshQualityChecker(
        Mesh(np.asarray(nodes), np.asarray(elements)),
        **kwargs,
    )


def _triangle(nodes):
    return _checker(
        np.asarray(nodes, dtype=np.float64),
        np.array([[0, 1, 2, 2]], dtype=np.int64),
    )


def _quad(nodes):
    return _checker(
        np.asarray(nodes, dtype=np.float64),
        np.array([[0, 1, 2, 3]], dtype=np.int64),
    )


def _mixed_checker():
    height = math.sqrt(3.0) / 2.0
    nodes = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, height],
            [2.0, 0.0],
            [6.0, 0.0],
            [6.0, 1.0],
            [2.0, 1.0],
            [7.0, 0.0],
            [8.0, 0.0],
            [8.0, 1.0],
            [7.0, 1.0],
        ]
    )
    elements = np.array(
        [[0, 1, 2, 2], [3, 4, 5, 6], [7, 8, 9, 10]],
        dtype=np.int64,
    )
    return _checker(nodes, elements)


class TriangleQualityTests(unittest.TestCase):
    def test_equilateral_triangle_has_ideal_metrics(self):
        checker = _triangle(
            [[0.0, 0.0], [1.0, 0.0], [0.5, math.sqrt(3.0) / 2.0]]
        )

        raw = checker.calculate_jacobian()
        scaled = checker.calculate_scaled_jacobian()
        aspect = checker.calculate_aspect_ratio()

        self.assertAlmostEqual(raw.values[0], math.sqrt(3.0) / 2.0)
        self.assertAlmostEqual(raw.minimum_values[0], raw.values[0])
        self.assertAlmostEqual(raw.maximum_values[0], raw.values[0])
        self.assertAlmostEqual(scaled.values[0], 1.0)
        self.assertAlmostEqual(aspect.values[0], 1.0)
        self.assertEqual(raw.element_types, (ElementType.TRI3,))

    def test_right_triangle_metrics_follow_documented_definitions(self):
        checker = _triangle([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        self.assertAlmostEqual(checker.calculate_jacobian().values[0], 1.0)
        self.assertAlmostEqual(
            checker.calculate_scaled_jacobian().values[0],
            math.sqrt(2.0 / 3.0),
        )
        self.assertAlmostEqual(
            checker.calculate_aspect_ratio().values[0], math.sqrt(2.0)
        )

    def test_reversed_triangle_is_inverted_and_failed(self):
        checker = _checker(
            np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            np.array([[0, 2, 1, 1]], dtype=np.int64),
        )

        result = checker.calculate_jacobian()
        report = checker.check_jacobian()

        self.assertLess(result.values[0], 0.0)
        self.assertTrue(result.reason_flags[0] & QualityReason.INVERTED)
        self.assertFalse(result.reason_flags[0] & QualityReason.BELOW_MINIMUM)
        self.assertTrue(report.reason_flags[0] & QualityReason.BELOW_MINIMUM)
        np.testing.assert_array_equal(report.failed_indices, [0])
        np.testing.assert_array_equal(report.invalid_indices, [0])

    def test_collinear_triangle_is_degenerate(self):
        checker = _triangle([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

        result = checker.calculate_scaled_jacobian()
        report = checker.check_scaled_jacobian(minimum=0.0)

        self.assertTrue(result.reason_flags[0] & QualityReason.DEGENERATE)
        np.testing.assert_array_equal(report.failed_indices, [0])
        np.testing.assert_array_equal(report.invalid_indices, [0])


class QuadrilateralQualityTests(unittest.TestCase):
    def test_unit_square_has_exact_reference_metrics(self):
        checker = _quad([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

        results = (
            (checker.calculate_jacobian(), 0.25),
            (checker.calculate_scaled_jacobian(), 1.0),
            (checker.calculate_aspect_ratio(), 1.0),
        )

        for result, expected in results:
            self.assertAlmostEqual(result.values[0], expected)
            self.assertAlmostEqual(result.minimum_values[0], expected)
            self.assertAlmostEqual(result.maximum_values[0], expected)
            self.assertEqual(result.reason_flags, (QualityReason.NONE,))
        self.assertEqual(results[0][0].element_types, (ElementType.QUAD4,))

    def test_rectangle_aspect_ratio_is_longest_over_shortest_edge(self):
        checker = _quad([[0.0, 0.0], [4.0, 0.0], [4.0, 1.0], [0.0, 1.0]])

        self.assertAlmostEqual(checker.calculate_jacobian().values[0], 1.0)
        self.assertAlmostEqual(checker.calculate_scaled_jacobian().values[0], 1.0)
        self.assertAlmostEqual(checker.calculate_aspect_ratio().values[0], 4.0)

    def test_skew_parallelogram_metrics(self):
        checker = _quad([[0.0, 0.0], [2.0, 0.0], [3.0, 1.0], [1.0, 1.0]])

        self.assertAlmostEqual(checker.calculate_jacobian().values[0], 0.5)
        self.assertAlmostEqual(
            checker.calculate_scaled_jacobian().values[0], 1.0 / math.sqrt(2.0)
        )
        self.assertAlmostEqual(
            checker.calculate_aspect_ratio().values[0], math.sqrt(2.0)
        )

    def test_concave_quad_is_folded(self):
        result = _quad(
            [[0.0, 0.0], [1.0, 0.0], [0.2, 0.2], [0.0, 1.0]]
        ).calculate_jacobian()

        self.assertLess(result.minimum_values[0], 0.0)
        self.assertGreater(result.maximum_values[0], 0.0)
        self.assertTrue(result.reason_flags[0] & QualityReason.FOLDED)
        self.assertEqual(result.summary.invalid_count, 1)

    def test_bow_tie_quad_is_folded(self):
        checker = _quad([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])

        result = checker.calculate_scaled_jacobian()
        report = checker.check_scaled_jacobian(minimum=-1.0)

        self.assertTrue(result.reason_flags[0] & QualityReason.FOLDED)
        np.testing.assert_array_equal(report.failed_indices, [0])
        np.testing.assert_array_equal(report.invalid_indices, [0])

    def test_collapsed_quad_is_degenerate(self):
        checker = _quad([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [1.0, 1.0]])

        raw = checker.calculate_jacobian()
        aspect = checker.calculate_aspect_ratio()

        self.assertTrue(raw.reason_flags[0] & QualityReason.DEGENERATE)
        self.assertTrue(aspect.reason_flags[0] & QualityReason.DEGENERATE)
        self.assertTrue(aspect.reason_flags[0] & QualityReason.UNDEFINED_METRIC)
        self.assertTrue(math.isinf(aspect.values[0]))
        self.assertEqual(raw.summary.invalid_count, 1)


class CalculationAndSelectionTests(unittest.TestCase):
    def test_calculate_and_check_are_separate_result_types(self):
        checker = _mixed_checker()

        result = checker.calculate_aspect_ratio(indices=[1])
        report = checker.check_aspect_ratio(maximum=3.0, indices=[1])

        self.assertIsInstance(result, QualityResult)
        self.assertNotIsInstance(result, QualityReport)
        self.assertIsInstance(report, QualityReport)
        self.assertFalse(hasattr(result, "passed_indices"))
        self.assertFalse(result.reason_flags[0] & QualityReason.ABOVE_MAXIMUM)
        self.assertTrue(report.reason_flags[0] & QualityReason.ABOVE_MAXIMUM)

    def test_selected_values_and_metadata_follow_requested_order(self):
        result = _mixed_checker().calculate_aspect_ratio(indices=[2, 0, 1])

        np.testing.assert_array_equal(result.element_indices, [2, 0, 1])
        np.testing.assert_allclose(result.values, [1.0, 1.0, 4.0])
        self.assertEqual(
            result.element_types,
            (ElementType.QUAD4, ElementType.TRI3, ElementType.QUAD4),
        )
        self.assertEqual(len(result.reason_flags), 3)

    def test_report_indices_and_worst_index_remain_global(self):
        checker = _mixed_checker()
        result = checker.calculate_aspect_ratio(indices=[2, 1])
        report = checker.check_aspect_ratio(
            maximum=3.0,
            indices=[2, 1],
        )

        np.testing.assert_array_equal(report.element_indices, [2, 1])
        np.testing.assert_allclose(report.values, [1.0, 4.0])
        np.testing.assert_array_equal(report.passed_indices, [2])
        np.testing.assert_array_equal(report.failed_indices, [1])
        self.assertEqual(report.invalid_indices.size, 0)
        self.assertEqual(report.summary.worst_index, 1)
        self.assertEqual(report.summary.total_count, 2)
        self.assertEqual(result.summary.worst_index, 1)

    def test_invalid_indices_map_to_global_element_ids(self):
        nodes = np.array(
            [
                [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
                [2.0, 0.0], [3.0, 0.0], [4.0, 0.0],
                [5.0, 0.0], [6.0, 0.0], [6.0, 1.0], [5.0, 1.0],
            ]
        )
        elements = np.array(
            [[0, 1, 2, 3], [4, 5, 6, 6], [7, 8, 9, 10]], dtype=np.int64
        )

        checker = _checker(nodes, elements)
        result = checker.calculate_scaled_jacobian(indices=[2, 1])
        report = checker.check_scaled_jacobian(
            minimum=0.5,
            indices=[2, 1],
        )

        np.testing.assert_array_equal(result.invalid_indices, [1])
        np.testing.assert_array_equal(report.passed_indices, [2])
        np.testing.assert_array_equal(report.failed_indices, [1])
        np.testing.assert_array_equal(report.invalid_indices, [1])

    def test_none_list_ndarray_and_empty_selections(self):
        checker = _mixed_checker()

        all_result = checker.calculate_jacobian(indices=None)
        list_result = checker.calculate_jacobian(indices=[2, 0])
        array_result = checker.calculate_jacobian(
            indices=np.array([1], dtype=np.int32)
        )
        empty_result = checker.calculate_jacobian(indices=[])
        empty_array_result = checker.calculate_jacobian(
            indices=np.array([], dtype=np.int64)
        )

        np.testing.assert_array_equal(all_result.element_indices, [0, 1, 2])
        np.testing.assert_array_equal(list_result.element_indices, [2, 0])
        np.testing.assert_array_equal(array_result.element_indices, [1])
        for result in (empty_result, empty_array_result):
            self.assertEqual(result.element_indices.size, 0)
            self.assertEqual(result.values.size, 0)
            self.assertEqual(result.summary.total_count, 0)
            self.assertIsNone(result.summary.worst_index)

    def test_invalid_selection_forms_raise(self):
        checker = _mixed_checker()
        invalid = {
            "duplicate": [0, 0],
            "negative": [-1],
            "out_of_range": [3],
            "noninteger": [0.0],
            "bool": [True],
            "tuple": (0,),
            "slice": slice(0, 1),
            "non_1d": np.array([[0]], dtype=np.int64),
        }

        for name, indices in invalid.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    checker.calculate_jacobian(indices=indices)

    def test_check_methods_share_selection_validation(self):
        checker = _mixed_checker()

        for call in (
            lambda: checker.check_jacobian(indices=[0, 0]),
            lambda: checker.check_scaled_jacobian(0.0, indices=[-1]),
            lambda: checker.check_aspect_ratio(4.0, indices=np.array([[0]])),
        ):
            with self.assertRaises(ValueError):
                call()

    def test_calculation_only_processes_the_requested_subset(self):
        checker = _mixed_checker()

        with patch.object(
            checker,
            "_calculate_geometry",
            wraps=checker._calculate_geometry,
        ) as calculate_geometry:
            checker.calculate_jacobian(indices=[2, 0])

        np.testing.assert_array_equal(calculate_geometry.call_args.args[0], [2, 0])


class MixedMeshAndReportTests(unittest.TestCase):
    def test_translation_and_uniform_scaling_invariance(self):
        nodes = np.array(
            [
                [0.0, 0.0], [1.0, 0.0], [0.5, math.sqrt(3.0) / 2.0],
                [2.0, 0.0], [4.0, 0.0], [5.0, 1.0], [3.0, 1.0],
            ]
        )
        elements = np.array([[0, 1, 2, 2], [3, 4, 5, 6]], dtype=np.int64)
        original = _checker(nodes, elements)
        factor = 3.5
        transformed = _checker(
            nodes * factor + np.array([101.0, -37.0]), elements
        )

        np.testing.assert_allclose(
            transformed.calculate_jacobian().values,
            original.calculate_jacobian().values * factor**2,
        )
        np.testing.assert_allclose(
            transformed.calculate_scaled_jacobian().values,
            original.calculate_scaled_jacobian().values,
        )
        np.testing.assert_allclose(
            transformed.calculate_aspect_ratio().values,
            original.calculate_aspect_ratio().values,
        )

    def test_threshold_equality_passes_for_each_metric(self):
        checker = _quad([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        reports = (
            checker.check_jacobian(minimum=0.25),
            checker.check_scaled_jacobian(minimum=1.0),
            checker.check_aspect_ratio(maximum=1.0),
        )

        for report in reports:
            np.testing.assert_array_equal(report.passed_indices, [0])
            self.assertEqual(report.failed_indices.size, 0)

    def test_small_jacobian_can_fail_absolute_quality_threshold(self):
        tolerance = GeometryTolerance(absolute=0.0, relative=0.0, planar=0.0)
        checker = _checker(
            np.array([[0.0, 0.0], [1.0e-8, 0.0], [0.0, 1.0e-8]]),
            np.array([[0, 1, 2, 2]], dtype=np.int64),
            tolerance=tolerance,
        )

        report = checker.check_jacobian(minimum=1.0e-15)

        self.assertAlmostEqual(report.values[0], 1.0e-16)
        self.assertTrue(report.reason_flags[0] & QualityReason.BELOW_MINIMUM)

    def test_result_and_report_arrays_are_read_only(self):
        checker = _mixed_checker()
        objects = (
            checker.calculate_jacobian(indices=[2, 0]),
            checker.check_jacobian(indices=[2, 0]),
        )

        for item in objects:
            for values in (
                item.element_indices,
                item.values,
                item.minimum_values,
                item.maximum_values,
                item.invalid_indices,
            ):
                self.assertFalse(values.flags.writeable)
        self.assertFalse(objects[1].passed_indices.flags.writeable)
        self.assertFalse(objects[1].failed_indices.flags.writeable)

    def test_summary_excludes_invalid_metrics_and_maps_worst_globally(self):
        nodes = np.array(
            [
                [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
                [2.0, 0.0], [6.0, 0.0], [6.0, 1.0], [2.0, 1.0],
                [7.0, 0.0], [9.0, 0.0], [8.0, 0.0],
            ]
        )
        elements = np.array(
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 10]], dtype=np.int64
        )

        result = _checker(nodes, elements).calculate_aspect_ratio(indices=[2, 0, 1])

        self.assertEqual(result.summary.total_count, 3)
        self.assertEqual(result.summary.count, 2)
        self.assertEqual(result.summary.invalid_count, 1)
        self.assertAlmostEqual(result.summary.minimum, 1.0)
        self.assertAlmostEqual(result.summary.maximum, 4.0)
        self.assertAlmostEqual(result.summary.mean, 2.5)
        self.assertEqual(result.summary.worst_index, 1)
        np.testing.assert_array_equal(result.invalid_indices, [2])

    def test_calculation_arrays_use_float64(self):
        checker = _checker(
            np.array(
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                dtype=np.float32,
            ),
            np.array([[0, 1, 2, 3]], dtype=np.int32),
        )

        result = checker.calculate_jacobian()

        self.assertEqual(result.values.dtype, np.dtype(np.float64))
        self.assertEqual(result.minimum_values.dtype, np.dtype(np.float64))
        self.assertEqual(result.maximum_values.dtype, np.dtype(np.float64))

    def test_no_combined_check_all_api(self):
        self.assertFalse(hasattr(_mixed_checker(), "check_all"))


class InvalidMeshTests(unittest.TestCase):
    def test_empty_mesh_returns_empty_result_and_report(self):
        checker = _checker(
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 4), dtype=np.int64),
        )

        result = checker.calculate_scaled_jacobian()
        report = checker.check_aspect_ratio(maximum=1.0)

        for item in (result, report):
            self.assertEqual(item.values.size, 0)
            self.assertEqual(item.summary.total_count, 0)
            self.assertEqual(item.summary.count, 0)
            self.assertIsNone(item.summary.worst_index)
            self.assertTrue(math.isnan(item.summary.minimum))
            self.assertTrue(math.isnan(item.summary.maximum))
            self.assertTrue(math.isnan(item.summary.mean))
        self.assertEqual(result.summary.invalid_count, 0)
        self.assertEqual(result.invalid_indices.size, 0)
        self.assertEqual(report.invalid_indices.size, 0)

    def test_nonfinite_coordinates_are_element_local_invalidity(self):
        for bad_value in (math.nan, math.inf):
            with self.subTest(bad_value=bad_value):
                checker = _checker(
                    np.array(
                        [[0.0, 0.0], [1.0, 0.0], [1.0, bad_value], [0.0, 1.0]]
                    ),
                    np.array([[0, 1, 2, 3]], dtype=np.int64),
                )
                result = checker.calculate_jacobian()
                self.assertTrue(
                    result.reason_flags[0] & QualityReason.NONFINITE_COORDINATES
                )
                self.assertFalse(math.isfinite(result.values[0]))
                self.assertEqual(result.summary.invalid_count, 1)
                np.testing.assert_array_equal(result.invalid_indices, [0])

    def test_bad_connectivity_is_element_local_invalidity(self):
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cases = (
            (np.array([[0, 1, 2, 99]], dtype=np.int64), QualityReason.OUT_OF_BOUNDS),
            (np.array([[0, 1, 2, -1]], dtype=np.int64), QualityReason.OUT_OF_BOUNDS),
            (
                np.array([[0, 1, 1, 2]], dtype=np.int64),
                QualityReason.INVALID_CONNECTIVITY,
            ),
        )

        for elements, reason in cases:
            with self.subTest(reason=reason):
                result = _checker(nodes, elements).calculate_jacobian()
                self.assertTrue(result.reason_flags[0] & reason)
                self.assertEqual(result.summary.invalid_count, 1)
                np.testing.assert_array_equal(result.invalid_indices, [0])

    def test_wrong_mesh_shapes_raise_value_error_at_checker_boundary(self):
        valid_nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        valid_elements = np.array([[0, 1, 2, 2]], dtype=np.int64)
        cases = (
            (np.array([0.0, 1.0]), valid_elements),
            (np.zeros((3, 1)), valid_elements),
            (np.zeros((3, 4)), valid_elements),
            (valid_nodes, np.array([0, 1, 2, 2])),
            (valid_nodes, np.zeros((1, 3), dtype=np.int64)),
        )

        for nodes, elements in cases:
            with self.subTest(nodes_shape=nodes.shape, elements_shape=elements.shape):
                with self.assertRaises(ValueError):
                    MeshQualityChecker(Mesh(nodes, elements))

    def test_wrong_mesh_dtypes_raise_value_error(self):
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        elements = np.array([[0, 1, 2, 2]], dtype=np.int64)

        with self.assertRaises(ValueError):
            MeshQualityChecker(Mesh(nodes.astype(object), elements))
        with self.assertRaises(ValueError):
            MeshQualityChecker(Mesh(nodes, elements.astype(np.float64)))

    def test_xyz_plane_and_explicit_planar_tolerance(self):
        elements = np.array([[0, 1, 2, 2]], dtype=np.int64)
        within = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 5.0e-4], [0.0, 1.0, 0.0]]
        )
        outside = within.copy()
        outside[1, 2] = 2.0e-3
        tolerance = GeometryTolerance(absolute=0.0, relative=0.0, planar=1.0e-3)

        result = MeshQualityChecker(
            Mesh(within, elements), tolerance=tolerance
        ).calculate_jacobian()
        self.assertAlmostEqual(result.values[0], 1.0)
        with self.assertRaises(ValueError):
            MeshQualityChecker(Mesh(outside, elements), tolerance=tolerance)

    def test_legacy_two_array_checker_constructor_is_removed(self):
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        elements = np.array([[0, 1, 2, 2]], dtype=np.int64)

        with self.assertRaises(TypeError):
            MeshQualityChecker(nodes, elements)


class GeneratedMeshTests(unittest.TestCase):
    def test_checkerboard_mesh_passes_basic_jacobian_check(self):
        mesh = checkerboard_box(
            element_size=0.4,
            x_list=[0.0, 1.0, 2.0],
            y_list=[0.0, 1.5, 3.0],
        )

        self.assertIsInstance(mesh, Mesh)
        report = MeshQualityChecker(mesh).check_jacobian()
        self.assertEqual(report.failed_indices.size, 0)
        self.assertEqual(report.invalid_indices.size, 0)
        self.assertEqual(report.passed_indices.size, mesh.elements.shape[0])
        self.assertTrue(np.all(report.values > 0.0))


if __name__ == "__main__":
    unittest.main()
