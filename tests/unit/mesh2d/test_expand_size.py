"""Layer thickness can vary independently of element and repair target sizes."""

import unittest

import numpy as np

from mesher.mesh2d.adapter import QuadLayerError, generate_layer_adaptive_mesh, plan_quad_sections
from tests.unit.mesh2d import test_extend_quad_layer as layer_checks
from tests.unit.mesh2d import test_outline_o as outline_checks
from tests.unit.mesh2d.test_corner_repair import _step
from tests.unit.mesh2d.test_slot_repair import _blocks, _c_slot


_APIS = (plan_quad_sections, generate_layer_adaptive_mesh)


def _refined_slot():
    return _blocks([
        (np.arange(0., 1.01, .25), np.arange(0., 3.01, .25)),
        (np.arange(1., 3.01, .25), np.arange(0., 1.01, .25)),
        (np.arange(1., 3.01, .25), np.arange(2., 3.01, .25)),
    ])


class _FailingFloat:
    def __init__(self, error):
        self.error = error

    def __float__(self):
        raise self.error


class ExpandSizeTests(unittest.TestCase):
    _snapshot = staticmethod(layer_checks.ExtendQuadLayerTests._snapshot)
    assert_unchanged = layer_checks.ExtendQuadLayerTests.assert_unchanged
    assert_layer = layer_checks.ExtendQuadLayerTests.assert_layer
    assert_target = outline_checks.OutlineTargetTests.assert_target

    def assert_front_bounds(self, points, minimum, maximum, *, atol=1e-9):
        np.testing.assert_allclose(np.min(points, axis=0), minimum, rtol=0, atol=atol)
        np.testing.assert_allclose(np.max(points, axis=0), maximum, rtol=0, atol=atol)

    def assert_line_spacing(self, mesh, result, element_size):
        lines = [section for section in result.diagnostics["section_details"]
                 if section["kind"] == "LINE"]
        self.assertTrue(lines)
        for section in lines:
            points = mesh.nodes[section["outer_ids"], :2]
            spacing = np.linalg.norm(np.diff(points, axis=0), axis=1)
            self.assertTrue(np.all(spacing >= .5 * element_size - 1e-9), spacing)
            self.assertTrue(np.all(spacing <= element_size + 1e-9), spacing)

    def test_omitted_none_and_equal_size_preserve_both_api_outputs(self):
        baseline = None
        for kwargs in ({}, {"expand_size": None}, {"expand_size": .5}):
            with self.subTest(kwargs=kwargs):
                mesh, outline = layer_checks._rectangle()
                snapshot = self._snapshot(mesh)
                patterns = [[[1., 3.4], [2., 3.4]]]
                plan = plan_quad_sections(mesh, outline, .5, pattern_lines=patterns, **kwargs)
                self.assert_unchanged(mesh, snapshot)
                result = generate_layer_adaptive_mesh(
                    mesh, outline, .5, pattern_lines=patterns, **kwargs)
                values = (mesh.nodes, mesh.elements, result.outline,
                          plan.front, plan.edge_expansion,
                          *(section.outer_boundary for section in plan.sections))
                if baseline is None:
                    baseline = [np.array(value, copy=True) for value in values]
                else:
                    self.assertEqual(len(values), len(baseline))
                    for actual, expected in zip(values, baseline):
                        np.testing.assert_array_equal(actual, expected)

    def test_growth_and_line_sampling_use_their_independent_sizes(self):
        counts = {}
        for element_size, expand_size in ((.5, .25), (.5, .8), (.25, .8)):
            with self.subTest(element_size=element_size, expand_size=expand_size):
                mesh, outline = layer_checks._rectangle()
                old_nodes, old_elements = mesh.nodes.copy(), mesh.elements.copy()
                plan = plan_quad_sections(mesh, outline, element_size, expand_size=expand_size)
                np.testing.assert_allclose(plan.edge_expansion, expand_size)
                self.assert_front_bounds(plan.front, [-expand_size] * 2,
                                         [4. + expand_size, 3. + expand_size])
                result = generate_layer_adaptive_mesh(
                    mesh, outline, element_size, expand_size=expand_size)
                self.assert_layer(mesh, outline, result, old_nodes, old_elements)
                np.testing.assert_allclose(result.diagnostics["edge_expansion"], expand_size)
                self.assert_front_bounds(mesh.nodes[result.outline, :2],
                                         [-expand_size] * 2, [4. + expand_size, 3. + expand_size])
                self.assert_line_spacing(mesh, result, element_size)
                counts[element_size, expand_size] = len(result.outline)
        self.assertGreater(counts[.25, .8], counts[.5, .8])

    def test_pattern_threshold_is_relative_to_expand_size(self):
        cases = ((.8, .6, 1., .6), (.8, .8, 1., .8), (.8, .9, 1., .8),
                 (.3, .4, 1., .3), (.3, .4, 1.5, .4),
                 (.3, .45, 1.5, .45), (.3, .46, 1.5, .3))
        for expand_size, distance, ratio, expected in cases:
            with self.subTest(expand_size=expand_size, distance=distance, ratio=ratio):
                mesh, outline = layer_checks._rectangle()
                pattern = [[1., 3. + distance], [2., 3. + distance]]
                kwargs = dict(expand_size=expand_size, expand_ratio=ratio,
                              pattern_lines=[pattern])
                plan = plan_quad_sections(mesh, outline, .5, **kwargs)
                np.testing.assert_allclose(plan.edge_expansion,
                                           [expand_size, expected, expand_size, expand_size])
                result = generate_layer_adaptive_mesh(mesh, outline, .5, **kwargs)
                np.testing.assert_allclose(result.diagnostics["edge_expansion"],
                                           plan.edge_expansion)
                self.assert_front_bounds(mesh.nodes[result.outline, :2],
                                         [-expand_size] * 2, [4. + expand_size, 3. + expected])

    def test_repair_triggers_and_replanned_growth_follow_expand_size(self):
        for fixture, key, element_size, expand_size in (
                (_refined_slot, "slot_repairs", .2, .6),
                (_step, "corner_repairs", .75, 1.2)):
            with self.subTest(repair=key):
                mesh, outline = fixture()
                snapshot = self._snapshot(mesh)
                baseline = plan_quad_sections(mesh, outline, element_size)
                self.assertEqual(baseline.diagnostics[key], [])
                plan = plan_quad_sections(mesh, outline, element_size, expand_size=expand_size)
                self.assert_unchanged(mesh, snapshot)
                self.assertIsNotNone(plan.prepared_source)
                self.assertEqual(len(plan.diagnostics[key]), 1)
                np.testing.assert_allclose(plan.edge_expansion, expand_size)
                repair = plan.diagnostics[key][0]
                self.assertAlmostEqual(repair["max_element_size"], element_size)
                self.assertLessEqual(repair["boundary_element_size"], element_size)
                result = generate_layer_adaptive_mesh(
                    mesh, outline, element_size, expand_size=expand_size)
                self.assert_layer(mesh, outline, result, snapshot[2], snapshot[3])
                self.assertEqual(len(result.diagnostics[key]), 1)
                np.testing.assert_allclose(result.diagnostics["edge_expansion"], expand_size)
                np.testing.assert_array_equal(mesh.nodes[:len(plan.prepared_source.nodes)],
                                              plan.prepared_source.nodes)
                np.testing.assert_array_equal(mesh.elements[:len(plan.prepared_source.elements)],
                                              plan.prepared_source.elements)
                self.assert_front_bounds(mesh.nodes[result.outline, :2],
                                         [-expand_size] * 2, [3. + expand_size] * 2)

    def test_smaller_expand_size_can_leave_slot_and_corner_unrepaired(self):
        for fixture, key, element_size, expand_size in (
                (_refined_slot, "slot_repairs", .5, .2),
                (_step, "corner_repairs", 1., .75)):
            with self.subTest(repair=key):
                mesh, outline = fixture()
                self.assertEqual(len(plan_quad_sections(mesh, outline, element_size).diagnostics[key]), 1)
                plan = plan_quad_sections(mesh, outline, element_size, expand_size=expand_size)
                self.assertIsNone(plan.prepared_source)
                self.assertEqual(plan.diagnostics[key], [])
                np.testing.assert_allclose(plan.edge_expansion, expand_size)
                result = generate_layer_adaptive_mesh(
                    mesh, outline, element_size, expand_size=expand_size)
                self.assertEqual(result.diagnostics[key], [])
                np.testing.assert_allclose(result.diagnostics["edge_expansion"], expand_size)

    def test_explicit_outline_has_precedence_over_expand_size_and_patterns(self):
        for expand_size in (.2, 1.2):
            with self.subTest(expand_size=expand_size):
                mesh, outline = layer_checks._rectangle()
                target = outline_checks._target()
                original = target.copy()
                ids = layer_checks._target_ids(mesh, target)
                kwargs = dict(outline_o=ids, expand_size=expand_size,
                              pattern_lines=[[[1., 4.], [2., 4.]]])
                plan = plan_quad_sections(mesh, outline, .5, **kwargs)
                np.testing.assert_allclose(plan.front, target)
                np.testing.assert_allclose(plan.edge_expansion, [.7, .8, .9, .4])
                result = generate_layer_adaptive_mesh(mesh, outline, .5, **kwargs)
                self.assert_target(mesh, result, ids)
                np.testing.assert_array_equal(target, original)

    def test_explicit_outline_keeps_expand_size_repair_decisions(self):
        for fixture, key, element_size, expand_size in (
                (_c_slot, "slot_repairs", .4, .6),
                (_step, "corner_repairs", .75, 1.2)):
            with self.subTest(repair=key):
                mesh, outline = fixture()
                target = np.array([[-.6, -.6], [-.6, 3.6], [3.6, 3.6], [3.6, -.6]])
                ids = layer_checks._target_ids(mesh, target)
                baseline = plan_quad_sections(mesh, outline, element_size, expand_size=expand_size)
                kwargs = dict(expand_size=expand_size, outline_o=ids)
                plan = plan_quad_sections(mesh, outline, element_size, **kwargs)
                self.assertEqual(len(plan.diagnostics[key]), 1)
                np.testing.assert_array_equal(plan.prepared_source.nodes,
                                              baseline.prepared_source.nodes)
                np.testing.assert_allclose(plan.front, target)
                result = generate_layer_adaptive_mesh(mesh, outline, element_size, **kwargs)
                self.assertEqual(len(result.diagnostics[key]), 1)
                self.assert_target(mesh, result, ids)

    def assert_atomic_error(self, api, value, *, code="invalid_input"):
        mesh, outline = layer_checks._rectangle()
        snapshot = self._snapshot(mesh)
        with self.assertRaises(QuadLayerError) as caught:
            api(mesh, outline, .5, expand_size=value)
        self.assert_unchanged(mesh, snapshot)
        error = caught.exception
        self.assertEqual(error.code, code)
        self.assertEqual(error.context["stage"], "input_validation")
        self.assertEqual(error.context["parameter"], "expand_size")
        return error

    def test_invalid_expand_size_domains_fail_atomically(self):
        for api in _APIS:
            for value in (0, -1, np.nan, np.inf, -np.inf):
                with self.subTest(api=api.__name__, value=value):
                    error = self.assert_atomic_error(api, value)
                    self.assertEqual(str(error), "expand_size must be finite and positive")

    def test_expand_size_conversion_failures_are_typed_and_atomic(self):
        for api in _APIS:
            for value in ("not-a-number", [1, 2], 10 ** 1000):
                with self.subTest(api=api.__name__, value_type=type(value)):
                    error = self.assert_atomic_error(api, value)
                    self.assertIsInstance(error.__cause__, (TypeError, ValueError, OverflowError))
            for exception in (TypeError, ValueError, OverflowError):
                with self.subTest(api=api.__name__, exception=exception):
                    cause = exception("cannot convert expansion size")
                    error = self.assert_atomic_error(api, _FailingFloat(cause))
                    self.assertIs(error.__cause__, cause)

    def test_unresolvable_expand_size_fails_atomically(self):
        for api in _APIS:
            with self.subTest(api=api.__name__, case="below_resolution"):
                self.assert_atomic_error(api, 1e-14, code="feature_below_resolution")
            with self.subTest(api=api.__name__, case="normalization_underflow"):
                self.assert_atomic_error(api, np.nextafter(0., 1.), code="feature_below_resolution")

    def test_expand_size_and_diagnostics_use_world_coordinate_units(self):
        cases = ((1e-7, np.zeros(2), np.eye(2)),
                 (1e6, np.array([10000., -20000.]), np.array([[0., -1.], [1., 0.]])),
                 (1., np.array([1e7, -2e7]), np.eye(2)))
        expected = np.array([[-.8, -.8], [-.8, 3.6], [4.8, 3.6], [4.8, -.8]])
        for scale, shift, rotation in cases:
            with self.subTest(scale=scale, shift=shift):
                mesh, outline = layer_checks._rectangle()
                mesh.nodes[:, :2] = mesh.nodes[:, :2] @ rotation.T * scale + shift
                patterns = np.array([[[1., 3.6], [2., 3.6]]]) @ rotation.T * scale + shift
                plan = plan_quad_sections(mesh, outline, .5 * scale,
                                          expand_size=.8 * scale, pattern_lines=patterns, tolerance=.001 * scale)
                np.testing.assert_allclose((plan.front - shift) @ rotation / scale,
                                           expected, rtol=0, atol=1e-8)
                result = generate_layer_adaptive_mesh(mesh, outline, .5 * scale,
                                                      expand_size=.8 * scale, pattern_lines=patterns, tolerance=.001 * scale)
                np.testing.assert_allclose(result.diagnostics["edge_expansion"],
                                           plan.edge_expansion, rtol=1e-8, atol=scale * 1e-8)
                front = (mesh.nodes[result.outline, :2] - shift) @ rotation / scale
                self.assert_front_bounds(front, [-.8, -.8], [4.8, 3.6], atol=1e-8)


if __name__ == "__main__":
    unittest.main()
