"""LINE policy parameters are validated transactionally and routed by scope."""

import unittest
from unittest.mock import patch

import numpy as np

from mesher.mesh2d.adapter import (QuadLayerError, generate_full_adaptive_mesh,
                                   generate_layer_adaptive_mesh, plan_quad_sections)
from mesher.mesh2d.adapter import _full
from mesher.mesh2d.adapter.layer import _mesher
from mesher.mesh2d.adapter.layer._line_policy import validate_line_parameters
from mesher.mesh2d.editor import EditorMesh2D
from mesher.mesh2d.generators import generate_rectilinear_mesh


class LinePolicyInputTests(unittest.TestCase):
    @staticmethod
    def fixture():
        mesh = generate_rectilinear_mesh(0.5, [0, 2], [0, 2])
        return mesh, EditorMesh2D.outlines_by_edge(mesh)[0]

    def assert_invalid_atomic(self, parameter, value, *, full):
        mesh, outline = self.fixture()
        nodes, elements = mesh.nodes, mesh.elements
        saved_nodes, saved_elements = nodes.copy(), elements.copy()
        target = (_full, "discover_boundaries") if full else (_mesher, "_prepare_plan")
        with patch.object(*target) as prepare:
            with self.assertRaises(QuadLayerError) as caught:
                if full:
                    generate_full_adaptive_mesh(
                        mesh, [0.5], outer_outline=[[-2, -2], [4, -2], [4, 4], [-2, 4]],
                        **{parameter: value})
                else:
                    generate_layer_adaptive_mesh(mesh, outline, 0.5, **{parameter: value})
        prepare.assert_not_called()
        error = caught.exception
        self.assertEqual(error.code, "invalid_input")
        self.assertEqual(error.context["stage"], "input_validation")
        self.assertEqual(error.context["parameter"], parameter)
        self.assertIs(mesh.nodes, nodes)
        self.assertIs(mesh.elements, elements)
        np.testing.assert_array_equal(mesh.nodes, saved_nodes)
        np.testing.assert_array_equal(mesh.elements, saved_elements)
        return error

    def test_invalid_counts_fail_before_preparation_including_fixed_only_full_mesh(self):
        for full in (False, True):
            for value in (0, -1, 10.0, np.float64(10), True, np.bool_(False),
                          None, "10", [10], np.array(10), np.inf):
                with self.subTest(full=full, value=repr(value)):
                    self.assert_invalid_atomic("max_transition_count", value, full=full)

    def test_invalid_ratios_fail_before_preparation_including_fixed_only_full_mesh(self):
        for full in (False, True):
            for value in (0, -0.5, 1.01, np.nan, np.inf, -np.inf, True, np.bool_(False),
                          None, "0.5", [0.5], np.array(0.5), 0.5 + 0j):
                with self.subTest(full=full, value=repr(value)):
                    self.assert_invalid_atomic("min_size_ratio", value, full=full)
            error = self.assert_invalid_atomic("min_size_ratio", 10 ** 1000, full=full)
            self.assertIsInstance(error.__cause__, OverflowError)

    def test_numpy_values_and_ratio_endpoints_are_normalized(self):
        for count, ratio in ((np.int64(7), np.float32(0.5)), (np.uint64(3), 1),
                             (10 ** 1000, np.nextafter(0.0, 1.0))):
            with self.subTest(count_type=type(count), ratio=ratio):
                actual_count, actual_ratio = validate_line_parameters(count, ratio)
                self.assertEqual(actual_count, count)
                self.assertEqual(actual_ratio, ratio)
                self.assertIs(type(actual_count), int)
                self.assertIs(type(actual_ratio), float)

    def test_layer_assembly_receives_normalized_settings_and_reports_them(self):
        mesh, outline = self.fixture()
        with patch.object(_mesher, "fill_sections", wraps=_mesher.fill_sections) as assemble:
            result = generate_layer_adaptive_mesh(
                mesh, outline, 0.5,
                max_transition_count=np.int64(7), min_size_ratio=np.float64(0.6))
        data = assemble.call_args.args[2]
        self.assertEqual(data["max_transition_count"], 7)
        self.assertEqual(data["min_size_ratio"], 0.6)
        self.assertIs(type(data["max_transition_count"]), int)
        self.assertEqual(result.diagnostics["max_transition_count"], 7)
        self.assertEqual(result.diagnostics["min_size_ratio"], 0.6)
        self.assertIs(result.diagnostics["line_policy_active"], True)

    def test_public_defaults_and_geometric_preview_contract(self):
        mesh, outline = self.fixture()
        before_nodes, before_elements = mesh.nodes.copy(), mesh.elements.copy()
        preview = plan_quad_sections(mesh, outline, 0.5)
        self.assertTrue(preview.sections)
        np.testing.assert_array_equal(mesh.nodes, before_nodes)
        np.testing.assert_array_equal(mesh.elements, before_elements)
        result = generate_layer_adaptive_mesh(mesh, outline, 0.5)
        self.assertEqual(result.diagnostics["max_transition_count"], 10)
        self.assertEqual(result.diagnostics["min_size_ratio"], 0.5)

    def test_full_forwards_only_to_intermediate_layers(self):
        mesh, _ = self.fixture()
        with (patch.object(_full, "generate_layer_adaptive_mesh",
                           wraps=_full.generate_layer_adaptive_mesh) as ordinary,
              patch.object(_full, "_generate_layer_adaptive_mesh",
                           wraps=_full._generate_layer_adaptive_mesh) as fixed):
            result = generate_full_adaptive_mesh(
                mesh, [0.5, 0.5], outer_outline=[[-3, -3], [5, -3], [5, 5], [-3, 5]],
                max_transition_count=np.int32(3), min_size_ratio=np.float64(0.6))
        ordinary.assert_called_once()
        self.assertEqual(ordinary.call_args.kwargs["max_transition_count"], 3)
        self.assertEqual(ordinary.call_args.kwargs["min_size_ratio"], 0.6)
        fixed.assert_called_once()
        self.assertIs(fixed.call_args.kwargs["fixed_outer_boundary"], True)
        self.assertNotIn("max_transition_count", fixed.call_args.kwargs)
        self.assertNotIn("min_size_ratio", fixed.call_args.kwargs)
        self.assertEqual(result.diagnostics["max_transition_count"], 3)
        self.assertEqual(result.diagnostics["min_size_ratio"], 0.6)
        self.assertIs(result.diagnostics["line_policy_active"], True)
        self.assertIs(result.diagnostics["layers"][0][0]["line_policy_active"], True)
        self.assertIs(result.diagnostics["layers"][-1][0]["line_policy_active"], False)

    def test_fixed_only_full_reports_requested_settings_as_inactive(self):
        mesh, _ = self.fixture()
        result = generate_full_adaptive_mesh(
            mesh, [0.5], outer_outline=[[-2, -2], [4, -2], [4, 4], [-2, 4]],
            max_transition_count=1, min_size_ratio=1)
        self.assertEqual(result.diagnostics["max_transition_count"], 1)
        self.assertEqual(result.diagnostics["min_size_ratio"], 1.0)
        self.assertIs(result.diagnostics["line_policy_active"], False)
        self.assertIs(result.diagnostics["layers"][0][0]["line_policy_active"], False)


if __name__ == "__main__":
    unittest.main()
