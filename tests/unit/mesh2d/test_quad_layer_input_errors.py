"""Both layer APIs report malformed scalars and outlines atomically."""

import unittest

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d.adapter import QuadLayerError, extend_quad_layer, plan_quad_sections


_APIS = (plan_quad_sections, extend_quad_layer)


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


class _FailingOutline:
    def __init__(self, error):
        self.error = error

    def __len__(self):
        raise self.error


class QuadLayerInputErrorTests(unittest.TestCase):
    def call_failure(self, api, *, parameters=None, outline=None):
        mesh = Mesh2D([[0., 0.], [4., 0.], [4., 3.], [0., 3.]],
                      np.asarray([[0, 1, 2, 3]], dtype=np.int64))
        nodes, elements = mesh.nodes, mesh.elements
        saved_nodes, saved_elements = nodes.copy(), elements.copy()
        if outline is None:
            outline = [[0, 3], [3, 2], [2, 1], [1, 0]]
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

    def test_ragged_outline_chain_is_typed_and_identifies_the_edge(self):
        for api in _APIS:
            with self.subTest(api=api.__name__):
                error = self.call_failure(
                    api, outline=[[0, 3], [3, [2, 1]], [2, 1], [1, 0]])
                self.assert_input_context(error, "outline", 1)
                self.assertIsInstance(error.__cause__, ValueError)

    def test_outline_and_chain_conversion_errors_are_typed(self):
        for api in _APIS:
            for exception in (TypeError, ValueError, OverflowError):
                for location in ("outline", "chain"):
                    with self.subTest(api=api.__name__, exception=exception, location=location):
                        cause = exception("cannot convert input")
                        outline = (_FailingOutline(cause) if location == "outline" else
                                   [[0, 3], _FailingArray(cause), [2, 1], [1, 0]])
                        error = self.call_failure(api, outline=outline)
                        self.assert_input_context(error, "outline",
                                                  1 if location == "chain" else None)
                        self.assertIs(error.__cause__, cause)

    def test_explicit_outline_shape_and_index_errors_keep_their_messages(self):
        shape_message = "each outline edge must be a chain of at least two integer node IDs"
        cases = (
            ([[3, 2]], shape_message),
            ([3], shape_message),
            ([3., 2.], shape_message),
            ([3, 99], "outline contains out-of-range node IDs"),
            ([3, -1], "outline contains out-of-range node IDs"),
        )
        for api in _APIS:
            for chain, message in cases:
                with self.subTest(api=api.__name__, chain=chain):
                    error = self.call_failure(api, outline=[[0, 3], chain, [2, 1], [1, 0]])
                    self.assert_input_context(error, "outline", 1)
                    self.assertEqual(str(error), message)
                    self.assertIsNone(error.__cause__)
            with self.subTest(api=api.__name__, case="too_few_edges"):
                error = self.call_failure(api, outline=[[0, 3], [3, 2], [2, 0]])
                self.assert_input_context(error, "outline")
                self.assertEqual(str(error), "outline needs at least four logical edges")
                self.assertIsNone(error.__cause__)

    def test_existing_quad_layer_errors_are_not_rewrapped(self):
        for api in _APIS:
            for location in ("scalar", "outline", "chain"):
                with self.subTest(api=api.__name__, location=location):
                    original = QuadLayerError("existing diagnostic", code="specific_failure",
                                              stage="existing_stage", parameter="original",
                                              edge_index=9)
                    if location == "scalar":
                        error = self.call_failure(
                            api, parameters={"element_size": _FailingFloat(original)})
                    else:
                        outline = (_FailingOutline(original) if location == "outline" else
                                   [[0, 3], _FailingArray(original), [2, 1], [1, 0]])
                        error = self.call_failure(api, outline=outline)
                    self.assertIs(error, original)
                    self.assertEqual(error.code, "specific_failure")
                    self.assertEqual(error.context, {"stage": "existing_stage",
                                                     "parameter": "original", "edge_index": 9})
                    self.assertIsNone(error.__cause__)


if __name__ == "__main__":
    unittest.main()
