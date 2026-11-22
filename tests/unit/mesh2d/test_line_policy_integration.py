"""Public adaptive APIs apply LINE limits transactionally at their intended layers."""

import unittest
from unittest.mock import patch

import numpy as np

from mesher.mesh2d.adapter import (
    QuadLayerError,
    generate_full_adaptive_mesh,
    generate_layer_adaptive_mesh,
    plan_quad_sections,
)
from tests.unit.mesh2d import test_full_adaptive_mesh as full_existing
from tests.unit.mesh2d import test_section_integration as section_existing
from mesher.mesh2d.adapter.layer._construction import aligned
from tests.unit.mesh2d.test_dense_sections import rectangle
from tests.unit.mesh2d.test_extend_quad_layer import _target_ids


class LinePolicyIntegrationTests(unittest.TestCase):
    def test_optional_ratios_independently_bound_committed_line_spacing(self):
        for api in (generate_layer_adaptive_mesh, generate_full_adaptive_mesh):
            for settings in ({}, {"min_size_ratio": None, "max_size_ratio": None},
                             {"min_size_ratio": .5}, {"max_size_ratio": 1.25},
                             {"min_size_ratio": .5, "max_size_ratio": 1.25}):
                with self.subTest(api=api.__name__, settings=settings):
                    mesh, outline = rectangle([0, .1, .2, 1.6, 3.], height=2.)
                    if api is generate_layer_adaptive_mesh:
                        result = api(mesh, outline, 1., max_transition_count=1, **settings)
                        layer = result.diagnostics
                    else:
                        result = api(mesh, [1., 1.], max_transition_count=1, **settings,
                                     outer_outline=[[-4, -4], [7, -4], [7, 6], [-4, 6]])
                        layer = result.diagnostics["layers"][0][0]
                    minimum, maximum = settings.get("min_size_ratio"), settings.get("max_size_ratio")
                    for key, expected in (("min_size_ratio", minimum), ("max_size_ratio", maximum)):
                        self.assertEqual(result.diagnostics[key], expected)
                        self.assertEqual(layer[key], expected)
                    lines = [s for s in layer["section_details"] if s["kind"] == "LINE"]
                    gaps = np.concatenate([np.linalg.norm(np.diff(mesh.nodes[s["outer_ids"], :2],
                                                                 axis=0), axis=1) for s in lines])
                    for section in lines:
                        self.assertEqual(section["size_band_active"], minimum is not None or maximum is not None)
                        self.assertEqual(section["forced_short_intervals"], [])
                        self.assertLessEqual(section["max_actual_transition_count"], 1)
                    if minimum is None:
                        self.assertLess(gaps.min(), .5 - 1e-8)
                    else:
                        self.assertGreaterEqual(gaps.min(), minimum - 1e-8)
                    if maximum is None:
                        self.assertGreater(gaps.max(), 1.25 + 1e-8)
                    else:
                        self.assertLessEqual(gaps.max(), maximum + 1e-8)
                    if not settings:
                        default_nodes, default_elements = mesh.nodes.copy(), mesh.elements.copy()
                    elif minimum is None and maximum is None:
                        np.testing.assert_array_equal(mesh.nodes, default_nodes)
                        np.testing.assert_array_equal(mesh.elements, default_elements)

    def assert_unchanged(self, mesh, snapshot):
        full_existing.FullAdaptiveMeshTests.assert_unchanged(self, mesh, snapshot)

    def assert_policy(self, mesh, result, size, ratio=.5, limit=10):
        lines = [detail for detail in result.diagnostics["section_details"]
                 if detail["kind"] == "LINE"]
        self.assertTrue(lines)
        units = result.diagnostics["units"]
        for detail in lines:
            points = mesh.nodes[detail["outer_ids"], :2]
            spacing = np.linalg.norm(np.diff(points, axis=0), axis=1)
            self.assertTrue(np.all(spacing >= ratio * size - 1e-8), spacing)
            self.assertTrue(np.all(spacing <= result.diagnostics["max_size_ratio"] * size + 1e-8), spacing)
            for index in detail["unit_ids"]:
                unit = units[index]
                loads = unit["source_segments_per_outer"]
                self.assertEqual(len(loads), unit["boundary_counts"][2] - 1)
                self.assertEqual(sum(loads), unit["boundary_counts"][0] - 1)
                self.assertGreaterEqual(min(loads), 0)
                self.assertLessEqual(max(loads), limit)

    def test_ten_segment_cap_is_conforming_and_frozen_in_all_directions(self):
        for rotation in range(4):
            with self.subTest(rotation=rotation):
                mesh, outline = section_existing.SectionIntegrationTests._nonuniform_rectangle(rotation)
                nodes, elements = mesh.nodes.copy(), mesh.elements.copy()
                result = generate_layer_adaptive_mesh(mesh, outline, 500., max_transition_count=10,
                                                      min_size_ratio=.5, max_size_ratio=1.25)
                self.assert_policy(mesh, result, 500.)
                section_existing.SectionIntegrationTests.assert_built_partition(
                    self, mesh, nodes, elements, result)

    def test_explicit_target_anchor_is_preserved_with_new_spacing_constraints(self):
        mesh, outline = section_existing.SectionIntegrationTests._nonuniform_rectangle()
        nodes, elements = mesh.nodes.copy(), mesh.elements.copy()
        target = np.array([[-500., -500.], [-500., 1500.], [1007.5, 1500.],
                           [2600., 1500.], [2600., -500.]])
        original = target.copy()
        ids = _target_ids(mesh, target)
        nodes = mesh.nodes.copy()
        result = generate_layer_adaptive_mesh(mesh, outline, 500., outline_o=ids, max_transition_count=10,
                                              min_size_ratio=.5, max_size_ratio=1.25)
        self.assert_policy(mesh, result, 500.)
        np.testing.assert_array_equal(target, original)
        front = mesh.nodes[np.unique(result.outline), :2]
        for point in target:
            self.assertEqual(np.count_nonzero(np.linalg.norm(front - point, axis=1) < 1e-8), 1)
        section_existing.SectionIntegrationTests.assert_built_partition(
            self, mesh, nodes, elements, result)

    def test_custom_transition_limit_is_observed_in_public_unit_diagnostics(self):
        mesh, outline = section_existing.SectionIntegrationTests._nonuniform_rectangle()
        result = generate_layer_adaptive_mesh(mesh, outline, 500., max_transition_count=3,
                                              min_size_ratio=.5, max_size_ratio=1.25)
        self.assert_policy(mesh, result, 500., limit=3)

    def test_seam_candidates_reuse_each_line_sampling_profile(self):
        mesh, outline = section_existing.SectionIntegrationTests._nonuniform_rectangle()
        with patch.object(aligned, "LineSampling", wraps=aligned.LineSampling) as profiles:
            result = generate_layer_adaptive_mesh(mesh, outline, 500., max_transition_count=10)
        lines = [detail for detail in result.diagnostics["section_details"]
                 if detail["kind"] == "LINE"]
        self.assertLessEqual(profiles.call_count, len(lines))
        self.assertTrue(any(len(detail["feasible_seam_pairs"]) > 1 for detail in lines))

    def test_short_mandatory_anchor_interval_is_reported_in_world_coordinates(self):
        mesh, outline = section_existing.SectionIntegrationTests._nonuniform_rectangle()
        nodes, elements = mesh.nodes.copy(), mesh.elements.copy()
        target = np.array([[-500., -500.], [-500., 1500.], [1000., 1500.],
                           [1100., 1500.], [2600., 1500.], [2600., -500.]])
        original = target.copy()
        ids = _target_ids(mesh, target)
        nodes = mesh.nodes.copy()
        result = generate_layer_adaptive_mesh(mesh, outline, 500., outline_o=ids, max_transition_count=10,
                                              min_size_ratio=.5)
        details = [detail for detail in result.diagnostics["section_details"]
                   if detail.get("forced_short_intervals")]
        self.assertEqual(len(details), 1)
        exception, = details[0]["forced_short_intervals"]
        np.testing.assert_allclose(exception["start"], [1000., 1500.], atol=1e-8)
        np.testing.assert_allclose(exception["end"], [1100., 1500.], atol=1e-8)
        self.assertAlmostEqual(exception["length"], 100.)
        front = mesh.nodes[details[0]["outer_ids"], :2]
        middle = front[(front[:, 0] >= 1000. - 1e-8) & (front[:, 0] <= 1100. + 1e-8)]
        np.testing.assert_allclose(middle, [[1000., 1500.], [1100., 1500.]], atol=1e-8)
        for index in details[0]["unit_ids"]:
            self.assertLessEqual(max(result.diagnostics["units"][index]["source_segments_per_outer"]), 10)
        np.testing.assert_array_equal(target, original)
        section_existing.SectionIntegrationTests.assert_built_partition(
            self, mesh, nodes, elements, result)

    def test_incompatible_limits_are_typed_and_do_not_commit_any_mesh_arrays(self):
        mesh, outline = section_existing.SectionIntegrationTests._nonuniform_rectangle()
        snapshot = full_existing.FullAdaptiveMeshTests._snapshot(mesh)
        # Geometric preview remains available even though resampling cannot
        # place enough outer segments within the requested size interval.
        preview = plan_quad_sections(mesh, outline, 500.)
        self.assertTrue(preview.sections)
        with self.assertRaises(QuadLayerError) as caught:
            generate_layer_adaptive_mesh(mesh, outline, 500., max_transition_count=2, min_size_ratio=.5)
        self.assertEqual(caught.exception.code, "line_constraints_infeasible")
        self.assert_unchanged(mesh, snapshot)

    def test_budget_accounts_for_policy_output_and_failure_is_atomic(self):
        baseline, outline = section_existing.SectionIntegrationTests._nonuniform_rectangle()
        settings = dict(max_transition_count=10, min_size_ratio=.5, max_size_ratio=1.25)
        count = generate_layer_adaptive_mesh(baseline, outline, 500., **settings).diagnostics["added_elements"]
        mesh, outline = section_existing.SectionIntegrationTests._nonuniform_rectangle()
        result = generate_layer_adaptive_mesh(mesh, outline, 500., max_added_elements=count, **settings)
        self.assertEqual(result.diagnostics["added_elements"], count)
        self.assert_policy(mesh, result, 500.)
        mesh, outline = section_existing.SectionIntegrationTests._nonuniform_rectangle()
        snapshot = full_existing.FullAdaptiveMeshTests._snapshot(mesh)
        with self.assertRaises(QuadLayerError) as caught:
            generate_layer_adaptive_mesh(mesh, outline, 500., max_added_elements=count - 1, **settings)
        self.assertEqual(caught.exception.code, "element_limit_exceeded")
        self.assert_unchanged(mesh, snapshot)

if __name__ == "__main__":
    unittest.main()
