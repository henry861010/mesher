"""Both layer APIs report malformed scalars and outlines atomically."""

import unittest

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d.adapter import QuadLayerError, generate_layer_adaptive_mesh, plan_quad_sections


_APIS = (plan_quad_sections, generate_layer_adaptive_mesh)


class _FailingFloat:
    def __init__(self, error):
        self.error = error

    def __float__(self):
        raise self.error


class _FailingArray:
    def __init__(self, error):
        self.error = error

    def __array__(self, dtype=None, copy=None):
        raise self.error


class QuadLayerInputErrorTests(unittest.TestCase):
    def call_failure(self, api, *, parameters=None, outline=None):
        mesh = Mesh2D([[0., 0.], [4., 0.], [4., 3.], [0., 3.]],
                      np.asarray([[0, 1, 2, 3]], dtype=np.int64))
        nodes, elements = mesh.nodes, mesh.elements
        saved_nodes, saved_elements = nodes.copy(), elements.copy()
        if outline is None:
            outline = [0, 3, 2, 1]
        with self.assertRaises(QuadLayerError) as caught:
            api(mesh, outline, **{"element_size": .5, **(parameters or {})})
        self.assertIs(mesh.nodes, nodes)
        self.assertIs(mesh.elements, elements)
        np.testing.assert_array_equal(mesh.nodes, saved_nodes)
        np.testing.assert_array_equal(mesh.elements, saved_elements)
        return caught.exception

    def assert_input_context(self, error, parameter, edge_index=None):
        self.assertEqual(error.code, "invalid_input")
        self.assertEqual(error.context["stage"], "input_validation")
        self.assertEqual(error.context["parameter"], parameter)
        if edge_index is not None:
            self.assertEqual(error.context["edge_index"], edge_index)

    def test_huge_integer_scalars_are_typed_for_both_apis(self):
        for api in _APIS:
            for parameter in ("element_size", "expand_ratio", "axis_tolerance"):
                with self.subTest(api=api.__name__, parameter=parameter):
                    error = self.call_failure(api, parameters={parameter: 10 ** 1000})
                    self.assert_input_context(error, parameter)
                    self.assertIsInstance(error.__cause__, OverflowError)

    def test_scalar_type_and_value_conversion_errors_are_typed(self):
        for api in _APIS:
            for parameter in ("element_size", "expand_ratio", "axis_tolerance"):
                for value in (None, "not-a-number", [1, 2]):
                    with self.subTest(api=api.__name__, parameter=parameter, value=value):
                        error = self.call_failure(api, parameters={parameter: value})
                        self.assert_input_context(error, parameter)
                        self.assertIsInstance(error.__cause__, (TypeError, ValueError))

    def test_existing_scalar_domain_messages_are_preserved(self):
        cases = (
            ("element_size", 0, "element_size must be finite and positive"),
            ("expand_ratio", .5, "expand_ratio must be finite and at least 1"),
            ("axis_tolerance", 1, "axis_tolerance must be finite and in [0, 1)"),
        )
        for api in _APIS:
            for parameter, invalid, message in cases:
                for value in (invalid, np.nan, np.inf):
                    with self.subTest(api=api.__name__, parameter=parameter, value=value):
                        error = self.call_failure(api, parameters={parameter: value})
                        self.assert_input_context(error, parameter)
                        self.assertEqual(str(error), message)
                        self.assertIsNone(error.__cause__)

    def test_malformed_flat_loops_are_typed_and_atomic(self):
        invalid = ([], [0, 3, 2], [0., 3., 2., 1.], [True] * 4,
                   [[0, 3], [3, 2], [2, 1], [1, 0]],
                   [[0, 3], [3, [2, 1]], [2, 1], [1, 0]],
                   [0, 3, 2, 99], [0, 3, 2, -1], [0, 3, 2, 3, 1],
                   [0, 3, 2, 1, 0, 0], [0, 3, 2, 10 ** 1000])
        for api in _APIS:
            for outline in invalid:
                with self.subTest(api=api.__name__, outline=outline):
                    error = self.call_failure(api, outline=outline)
                    self.assert_input_context(error, "outline_i")

    def test_outline_conversion_errors_preserve_the_cause(self):
        for api in _APIS:
            for exception in (TypeError, ValueError, OverflowError):
                with self.subTest(api=api.__name__, exception=exception):
                    cause = exception("cannot convert input")
                    error = self.call_failure(api, outline=_FailingArray(cause))
                    self.assert_input_context(error, "outline_i")
                    self.assertIs(error.__cause__, cause)

    def test_existing_quad_layer_errors_are_not_rewrapped(self):
        for api in _APIS:
            for location in ("scalar", "outline"):
                with self.subTest(api=api.__name__, location=location):
                    original = QuadLayerError("existing diagnostic", code="specific_failure",
                                              stage="existing_stage", parameter="original",
                                              edge_index=9)
                    if location == "scalar":
                        error = self.call_failure(
                            api, parameters={"element_size": _FailingFloat(original)})
                    else:
                        error = self.call_failure(api, outline=_FailingArray(original))
                    self.assertIs(error, original)
                    self.assertEqual(error.context, {"stage": "existing_stage",
                                                     "parameter": "original", "edge_index": 9})
                    self.assertIsNone(error.__cause__)

    def test_old_parameter_names_are_removed(self):
        for api in _APIS:
            mesh = Mesh2D([[0, 0], [4, 0], [4, 3], [0, 3]], [[0, 1, 2, 3]])
            with self.subTest(api=api.__name__, parameter="outline"):
                with self.assertRaises(TypeError):
                    api(mesh, outline=[0, 3, 2, 1], element_size=.5)
            with self.subTest(api=api.__name__, parameter="new_outline"):
                with self.assertRaises(TypeError):
                    api(mesh, [0, 3, 2, 1], .5, new_outline=None)
            error = self.call_failure(api, parameters={"outline_o": []})
            self.assert_input_context(error, "outline_o")


if __name__ == "__main__":
    unittest.main()
